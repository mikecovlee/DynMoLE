import logging
import math
from typing import List, Optional, Tuple

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
    _entropy,
)


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
        self.router_profile_: bool = False
        self.profiler_: List[int] = None

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
        route_weight = torch.nn.functional.softmax(
            self.gate_(hidden_states.to(self.dtype_)), dim=-1, dtype=torch.float32
        )
        if self.router_profile_:
            logging.info(f"entropy: {_entropy(route_weight)}")

        for expert_idx in range(self.experts_):
            expert_lora = lora_linear.loras_[
                f"moe.{self.adapter_name_}.experts.{expert_idx}"
            ]
            residual = residual + (
                torch.unsqueeze(route_weight[:, :, expert_idx], -1)
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
        self.router_profile_: bool = False
        self.profiler_: List[int] = None

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
        routing_weights_before = F.softmax(router_logits, dim=1, dtype=self.dtype_)
        if self.router_profile_:
            logging.info(f"entropy: {_entropy(routing_weights_before)}")

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
    broadcast_threshhold: float,
    top_p: float,
    eps: float = 1e-5,
):
    # calculate router entropy
    router_entropy = _entropy(router_logits, -1, eps)
    # broadcast if higher than threshhold
    broadcast_index = torch.where(router_entropy >= broadcast_threshhold)[0]
    # calculate top-p routing
    sorted_logits = torch.sort(router_logits, dim=-1, descending=True)[0]
    cumulative_probs = sorted_logits.cumsum(dim=-1)
    expert_mask = cumulative_probs > top_p
    # maintain top-1 if no experts selected
    threshold_indices = expert_mask.long().argmax(dim=-1)
    threshold_mask = torch.nn.functional.one_hot(
        threshold_indices, num_classes=router_logits.size(-1)
    ).to(torch.bool)
    # calculate final mask
    expert_mask = (expert_mask & ~threshold_mask).index_fill(0, broadcast_index, False)
    sorted_logits = sorted_logits.masked_fill(expert_mask, 0.0)
    return router_entropy, sorted_logits


def _dynamic_load_balancing_loss_func(
    routing_logits: torch.Tensor,
    broadcast_threshhold: float,
    top_p: float,
    eps: float = 1e-5,
    aux_loss_coef: float = 0.001,
    dyn_loss_coef: float = 0.001,
    attention_mask: Optional[torch.Tensor] = None,
) -> float:
    routing_weights = F.softmax(routing_logits, dim=-1)

    router_entropy, routing_weights = _dynamic_routing(
        routing_weights, broadcast_threshhold, top_p, eps
    )

    entropy_loss = torch.mean(router_entropy, dim=0)

    expert_mask = (routing_weights > 0.0).long()

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        num_experts = routing_weights.size(-1)
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = routing_weights.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand(
                (
                    num_hidden_layers,
                    batch_size,
                    sequence_length,
                    num_experts,
                )
            )
            .reshape(-1, num_experts)
            .to(routing_weights.device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(
            expert_mask.float() * expert_attention_mask, dim=0
        ) / torch.sum(expert_attention_mask, dim=0)

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(routing_weights.device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(
            routing_weights * router_per_expert_attention_mask, dim=0
        ) / torch.sum(router_per_expert_attention_mask, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert)

    return entropy_loss * dyn_loss_coef + overall_loss * num_experts * aux_loss_coef


def _unpack_router_logits(gate_logits: List[torch.Tensor]):
    compute_device = gate_logits[0].device
    concatenated_gate_logits = torch.cat(
        [
            layer_gate.to(compute_device).reshape(-1, layer_gate.size(-1))
            for layer_gate in gate_logits
        ],
        dim=0,
    )
    return concatenated_gate_logits


class DynMoleRouterLoss(torch.nn.Module):
    def __init__(self, config: DynMoleConfig) -> None:
        super().__init__()
        self.aux_loss_coef = config.router_aux_loss_coef_
        self.dyn_loss_coef = config.router_dyn_loss_coef_
        self.broadcast_threshhold: float = config.broadcast_threshhold_
        self.top_p: float = config.top_p_
        self.eps: float = config.entropy_eps_

    def forward(self, gate_logits, attention_mask) -> torch.Tensor:
        concatenated_gate_logits = _unpack_router_logits(gate_logits)
        return _dynamic_load_balancing_loss_func(
            concatenated_gate_logits,
            self.broadcast_threshhold,
            self.top_p,
            self.eps,
            self.aux_loss_coef,
            self.dyn_loss_coef,
            attention_mask,
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
        self.broadcast_threshhold_: float = config.broadcast_threshhold_
        self.top_p_: float = config.top_p_
        self.eps_: float = config.entropy_eps_
        self.experts_: int = config.num_experts_
        self.router_logits_: torch.Tensor = None
        self.router_profile_: bool = False
        self.profiler_: List[int] = None

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
        self.router_logits_ = self.gate_(hidden_states.to(self.dtype_))
        routing_weights = F.softmax(self.router_logits_, dim=-1, dtype=torch.float32)
        router_entropy, routing_weights = _dynamic_routing(
            routing_weights, self.broadcast_threshhold_, self.top_p_, self.eps_
        )
        if self.router_profile_:
            logging.info(f"entropy: {router_entropy.mean()}")
            router_profile = (routing_weights > 0.0).long().sum(-1).float()
            logging.info(f"max activated: {router_profile.max()}")
            logging.info(f"min activated: {router_profile.min()}")
            logging.info(f"avg activated: {router_profile.mean()}")

        for expert_idx in range(self.experts_):
            expert_lora = lora_linear.loras_[
                f"moe.{self.adapter_name_}.experts.{expert_idx}"
            ]
            residual = residual + (
                routing_weights[:, :, expert_idx].unsqueeze(-1)
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
        raise ValueError(f"Unknown routing strategy {config.routing_strategy_}")
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
