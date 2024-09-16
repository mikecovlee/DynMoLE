import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .abstracts import LLMMoeBlock
from .config import DynMoleConfig, LoraMoeConfig, MixLoraConfig, MolaConfig
from .lora_linear import Linear
from .mix_lora import (
    DynamicRouterLoss,
    DynamicSparseMoe,
    MixtralRouterLoss,
    MixtralSparseMoe,
    SwitchRouterLoss,
    SwitchSparseMoe,
)
from .moe_utils import tsallis_entropy


class LoraMoe(LLMMoeBlock):
    def __init__(
        self,
        in_features: int,
        device: torch.device,
        config: LoraMoeConfig,
        gate: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.adapter_name_: str = config.adapter_name
        self.dtype_: torch.dtype = torch.float32
        self.gate_ = torch.nn.Linear(
            in_features,
            config.num_experts_,
            bias=False,
            device=device,
            dtype=torch.float32,
        )
        self.experts_ = config.num_experts_
        self.router_logits_: torch.Tensor = None

        if gate is None:
            torch.nn.init.kaiming_uniform_(
                self.gate_.weight, a=math.sqrt(config.router_init_range_)
            )
        else:
            with torch.no_grad():
                self.gate_.weight.copy_(gate)

    def forward(
        self,
        residual: torch.Tensor,
        hidden_states: torch.Tensor,
        lora_linear: Optional[Linear] = None,
    ) -> Tuple:
        assert lora_linear is not None
        router_logits = self.gate_(hidden_states.to(self.dtype_))
        self.router_logits_ = router_logits.reshape(-1, self.experts_)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)

        for expert_idx in range(self.experts_):
            expert_lora = lora_linear.loras_[
                f"moe.{self.adapter_name_}.experts.{expert_idx}"
            ]
            residual = residual + (
                torch.unsqueeze(routing_weights[:, :, expert_idx], -1)
                * expert_lora.lora_forward(hidden_states)
            ).to(hidden_states.dtype)

        return residual


class MolaSparseMoe(LLMMoeBlock):
    def __init__(
        self,
        in_features: int,
        device: torch.device,
        config: MolaConfig,
        gate: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.adapter_name_: str = config.adapter_name
        self.dtype_: torch.dtype = torch.float32
        self.gate_ = torch.nn.Linear(
            in_features,
            config.num_experts_,
            bias=False,
            device=device,
            dtype=torch.float32,
        )
        self.experts_ = config.num_experts_
        self.topk_ = config.top_k_
        self.router_logits_: torch.Tensor = None

        if gate is None:
            torch.nn.init.kaiming_uniform_(
                self.gate_.weight, a=math.sqrt(config.router_init_range_)
            )
        else:
            with torch.no_grad():
                self.gate_.weight.copy_(gate)

    def forward(
        self,
        residual: torch.Tensor,
        hidden_states: torch.Tensor,
        lora_linear: Optional[Linear] = None,
    ):
        assert lora_linear is not None
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.view(-1, hidden_dim).to(self.dtype_)
        router_logits = self.gate_(hidden_states)
        self.router_logits_ = router_logits.reshape(-1, self.experts_)
        routing_weights_before = F.softmax(router_logits, dim=1, dtype=self.dtype_)

        routing_weights, selected_experts = torch.topk(
            routing_weights_before, self.topk_, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.experts_
        ).permute(2, 1, 0)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, lora_linear.out_features_),
            dtype=self.dtype_,
            device=hidden_states.device,
        )

        for expert_idx in range(self.experts_):
            expert_lora = lora_linear.loras_[
                f"moe.{self.adapter_name_}.experts.{expert_idx}"
            ]
            idx, top_x = torch.where(expert_mask[expert_idx])

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = (
                expert_lora.lora_forward(current_state)
                * routing_weights[top_x, idx, None]
            )
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(self.dtype_)
            )

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, lora_linear.out_features_
        ).to(input_dtype)

        return residual + final_hidden_states


@torch.jit.script
def _dynamic_routing(
    router_logits: torch.Tensor,
    entropy_threshold: float,
    entropy_index: float,
    entropy_eps: float,
    keep_top_k: int,
    top_p: float,
):
    # Top-p routing
    sorted_logits, sorted_indices = torch.sort(router_logits, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_logits, dim=-1)

    # Create a mask for Top-p experts
    expert_mask = (cumulative_probs <= top_p).to(router_logits.dtype)

    # Ensure at least top-k experts are kept
    expert_mask[..., :keep_top_k] = 1  # Keep at least top-k experts

    # Scatter final mask back to original shape using sorted indices
    final_mask = torch.zeros_like(router_logits)
    final_mask.scatter_(-1, sorted_indices, expert_mask)

    # Calculate entropy using Tsallis entropy
    router_entropy = tsallis_entropy(p=router_logits, q=entropy_index, eps=entropy_eps)

    # Broadcast if entropy is higher than threshold (set all experts active)
    high_entropy_mask = router_entropy > entropy_threshold
    final_mask[high_entropy_mask] = 1  # Activate all experts for high-entropy tokens

    # Apply mask to router logits (deactivate non-selected experts)
    router_logits = router_logits * final_mask

    return router_entropy, router_logits


def _dynamic_load_balancing_loss_func(
    router_logits: torch.Tensor,
    entropy_threshold: float,
    entropy_index: float,
    entropy_eps: float,
    keep_top_k: int,
    top_p: float,
    dyn_loss_coef: float,
    aux_loss_coef: float,
    attention_mask: Optional[torch.Tensor] = None,
) -> float:
    orig_routing_weights = F.softmax(router_logits, dim=-1)

    # Dynamic routing
    router_entropy, routing_weights = _dynamic_routing(
        router_logits=orig_routing_weights,
        entropy_threshold=entropy_threshold,
        entropy_index=entropy_index,
        entropy_eps=entropy_eps,
        keep_top_k=keep_top_k,
        top_p=top_p,
    )

    # Entropy loss
    entropy_loss = router_entropy.mean()

    # Router profile statistics
    router_profile = (routing_weights > 0.0).long().sum(-1).float()
    logging.info(f"    max activated: {router_profile.max()}")
    logging.info(f"    min activated: {router_profile.min()}")
    logging.info(f"    avg activated: {router_profile.mean()}")

    num_experts = routing_weights.size(-1)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each expert
        tokens_per_expert = torch.mean(routing_weights, dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(orig_routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = routing_weights.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0
        per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand(num_hidden_layers, batch_size, sequence_length, num_experts)
            .reshape(-1, num_experts)
            .to(routing_weights.device)
        )

        # Compute the sum of attention mask along the token dimension
        sum_per_expert_attention_mask = torch.sum(per_expert_attention_mask, dim=0)

        # Compute the percentage of tokens routed to each expert and the average routing probability
        tokens_per_expert = torch.sum(
            routing_weights * per_expert_attention_mask, dim=0
        ) / (sum_per_expert_attention_mask + 1e-8)

        router_prob_per_expert = torch.sum(
            orig_routing_weights * per_expert_attention_mask, dim=0
        ) / (sum_per_expert_attention_mask + 1e-8)

    # Load balance loss
    load_balance_loss = num_experts * torch.sum(
        tokens_per_expert * router_prob_per_expert
    )

    return dyn_loss_coef * entropy_loss + aux_loss_coef * load_balance_loss


class DynMoleRouterLoss(torch.nn.Module):
    def __init__(self, config: DynMoleConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, gate_logits, attention_mask) -> torch.Tensor:
        return _dynamic_load_balancing_loss_func(
            router_logits=gate_logits,
            entropy_threshold=self.config.entropy_threshold_,
            entropy_index=self.config.entropy_index_,
            entropy_eps=self.config.entropy_eps_,
            keep_top_k=self.config.keep_top_k_,
            top_p=self.config.top_p_,
            dyn_loss_coef=self.config.router_dyn_loss_coef_,
            aux_loss_coef=self.config.router_aux_loss_coef_,
            attention_mask=attention_mask,
        )


class DynMole(LLMMoeBlock):
    def __init__(
        self,
        in_features: int,
        device: torch.device,
        config: DynMoleConfig,
        gate: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.adapter_name_: str = config.adapter_name
        self.dtype_: torch.dtype = torch.float32
        self.gate_ = torch.nn.Linear(
            in_features,
            config.num_experts_,
            bias=False,
            device=device,
            dtype=torch.float32,
        )
        self.config_ = config
        self.experts_ = config.num_experts_
        self.router_logits_: torch.Tensor = None

        if gate is None:
            torch.nn.init.kaiming_uniform_(
                self.gate_.weight, a=math.sqrt(config.router_init_range_)
            )
        else:
            with torch.no_grad():
                self.gate_.weight.copy_(gate)

    def forward(
        self,
        residual: torch.Tensor,
        hidden_states: torch.Tensor,
        lora_linear: Optional[Linear] = None,
    ) -> Tuple:
        assert lora_linear is not None
        router_logits = self.gate_(hidden_states.to(self.dtype_))
        self.router_logits_ = router_logits.reshape(-1, self.experts_)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        _, routing_weights = _dynamic_routing(
            router_logits=routing_weights,
            entropy_threshold=self.config_.entropy_threshold_,
            entropy_index=self.config_.entropy_index_,
            entropy_eps=self.config_.entropy_eps_,
            keep_top_k=self.config_.keep_top_k_,
            top_p=self.config_.top_p_,
        )

        for expert_idx in range(self.experts_):
            expert_lora = lora_linear.loras_[
                f"moe.{self.adapter_name_}.experts.{expert_idx}"
            ]
            residual = residual + (
                routing_weights[..., expert_idx].unsqueeze(-1)
                * expert_lora.lora_forward(hidden_states)
            ).to(hidden_states.dtype)

        return residual


router_loss_dict = {
    "mixlora": MixtralRouterLoss,
    "mixlora-dynamic": DynamicRouterLoss,
    "mixlora-switch": SwitchRouterLoss,
    "dynmole": DynMoleRouterLoss,
}


def router_loss_factory(config: MixLoraConfig) -> torch.nn.Module:
    if config.routing_strategy_ not in router_loss_dict:
        return None
    if config.router_loss_:
        return router_loss_dict[config.routing_strategy_](config)
    else:
        return None


moe_layer_dict = {
    "mixlora": MixtralSparseMoe,
    "mixlora-dynamic": DynamicSparseMoe,
    "mixlora-switch": SwitchSparseMoe,
    "loramoe": LoraMoe,
    "mola": MolaSparseMoe,
    "dynmole": DynMole,
}


def moe_layer_factory(
    in_features: int,
    device: torch.device,
    config: MolaConfig,
    gate: Optional[torch.Tensor] = None,
) -> torch.nn.Module:
    if config.routing_strategy_ not in moe_layer_dict:
        raise ValueError(f"Unknown routing strategy {config.routing_strategy_}")
    return moe_layer_dict[config.routing_strategy_](in_features, device, config, gate)
