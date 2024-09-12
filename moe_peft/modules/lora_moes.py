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
    broadcast_index, _ = torch.where(router_entropy >= broadcast_threshhold)
    # calculate top-p routing
    sorted_logits, _ = torch.sort(router_logits, dim=-1, descending=True)
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
    # sorted_indices = sorted_indices.masked_fill(expert_mask, -1)
    return router_entropy, sorted_logits


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
        router_logits = self.gate_(hidden_states.to(self.dtype_))
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
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
