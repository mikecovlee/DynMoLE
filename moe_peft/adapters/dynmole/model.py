import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from moe_peft.common import Linear, LLMMoeBlock, renyi_entropy, tsallis_entropy

from .config import DynMoleConfig


@torch.jit.script
def _dynamic_routing(
    router_logits: torch.Tensor,
    entropy_threshold: float,
    entropy_index: float,
    entropy_type: str,
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

    # Calculate entropy
    if entropy_type == "tsallis":
        router_entropy = tsallis_entropy(
            p=router_logits, a=entropy_index, eps=entropy_eps
        )
    elif entropy_type == "renyi":
        router_entropy = renyi_entropy(
            p=router_logits, a=entropy_index, eps=entropy_eps
        )
    else:
        raise NotImplementedError()

    if entropy_index > 0.0 and entropy_threshold < 1.0:
        # Broadcast if entropy is higher than threshold (set all experts active)
        high_entropy_mask = router_entropy > entropy_threshold
        # Activate all experts for high-entropy tokens
        final_mask[high_entropy_mask] = 1

    # Apply mask to router logits (deactivate non-selected experts)
    router_logits = router_logits * final_mask

    return router_entropy, router_logits


def _dynamic_load_balancing_loss_func(
    router_logits: torch.Tensor,
    entropy_threshold: float,
    entropy_index: float,
    entropy_type: str,
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
        entropy_type=entropy_type,
        entropy_eps=entropy_eps,
        keep_top_k=keep_top_k,
        top_p=top_p,
    )

    # Entropy loss
    entropy_loss = router_entropy.mean()

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
            entropy_type=self.config.entropy_type_,
            entropy_eps=self.config.entropy_eps_,
            keep_top_k=self.config.keep_top_k_,
            top_p=self.config.top_p_,
            dyn_loss_coef=self.config.router_dyn_loss_coef_,
            aux_loss_coef=self.config.router_aux_loss_coef_,
            attention_mask=attention_mask,
        )


class DynMoleSparseMoe(LLMMoeBlock):
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
            entropy_type=self.config_.entropy_type_,
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
