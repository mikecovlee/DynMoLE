import copy
from dataclasses import dataclass
from typing import Dict

from moe_peft.modules import LoraConfig


@dataclass
class DynMoleConfig(LoraConfig):
    entropy_threshold_: float = None
    entropy_index_: float = None
    entropy_eps_: float = None
    keep_top_k_: int = None
    top_p_: float = None
    num_experts_: int = None
    router_init_range_: float = None
    routing_strategy_: str = "dynmole"
    router_aux_loss_coef_: float = None
    router_dyn_loss_coef_: float = None
    router_loss_: bool = True

    def check(self) -> "DynMoleConfig":
        super().check()
        assert (
            isinstance(self.entropy_threshold_, float) and self.entropy_threshold_ > 0
        )
        assert (
            isinstance(self.entropy_index_, float)
            and self.entropy_index_ > 0
            and self.entropy_index_ <= 2.0
        )
        assert isinstance(self.entropy_eps_, float) and self.entropy_eps_ > 0
        assert isinstance(self.keep_top_k_, int) and self.keep_top_k_ > 0
        assert isinstance(self.top_p_, float) and self.top_p_ > 0 and self.top_p_ <= 1.0
        assert isinstance(self.num_experts_, int) and self.num_experts_ > 0
        assert (
            isinstance(self.router_init_range_, float) and self.router_init_range_ >= 0
        )
        assert (
            isinstance(self.router_aux_loss_coef_, float)
            and self.router_aux_loss_coef_ >= 0
        )
        assert (
            isinstance(self.router_dyn_loss_coef_, float)
            and self.router_dyn_loss_coef_ >= 0
        )
        assert isinstance(self.router_loss_, bool)

        return self

    @staticmethod
    def from_config(config: Dict[str, any]) -> "DynMoleConfig":
        return DynMoleConfig(
            entropy_threshold_=config.get("entropy_threshold", 0.9),
            entropy_index_=config.get("entropy_index", 1.1),
            entropy_eps_=config.get("entropy_eps", 1e-5),
            keep_top_k_=config.get("keep_top_k", 2),
            top_p_=config.get("top_p", 0.75),
            num_experts_=config["num_experts"],
            router_init_range_=config.get("router_init_range", 5.0),
            router_aux_loss_coef_=config.get(
                "router_aux_loss_coef", 0.001
            ),  # for training
            router_dyn_loss_coef_=config.get(
                "router_dyn_loss_coef", 0.01
            ),  # for training
            router_loss_=config.get("router_loss", True),
            **LoraConfig.from_config(config).__dict__,
        )

    def export(self) -> Dict[str, any]:
        config = super().export()
        config["peft_type"] = "DYNMOLE"
        config["routing_strategy"] = self.routing_strategy_
        config["num_experts"] = self.num_experts_
        config["entropy_threshold"] = self.entropy_threshold_
        config["entropy_index"] = self.entropy_index_
        config["entropy_eps"] = self.entropy_eps_
        config["keep_top_k"] = self.keep_top_k_
        config["top_p"] = self.top_p_

        return config

    def expert_config(self, expert_idx: int) -> LoraConfig:
        config = copy.deepcopy(super())
        config.adapter_name = f"moe.{self.adapter_name}.experts.{expert_idx}"
        return config
