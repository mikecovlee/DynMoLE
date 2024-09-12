import copy

import torch

from .abstracts import LLMDecoder, LLMModelInput


@torch.jit.script
def logits_entropy(
    logits: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-5,
) -> torch.Tensor:
    probs_neg_log = -torch.log(logits + eps)  # eps for 'p=0, -plogp=0'
    return (logits * probs_neg_log).sum(dim=dim)


def collect_plugin_router_logtis(
    router_logits, input_args: LLMModelInput, decoder_layer: LLMDecoder
):
    if router_logits is None or len(router_logits) == 0:
        router_logits = [None for _ in range(len(input_args.batch_configs_))]

    attn_proj, mlp_proj = decoder_layer.state_dict()
    all_proj = copy.copy(attn_proj)
    all_proj.update(mlp_proj)
    for idx, config in enumerate(input_args.batch_configs_):
        assert router_logits[idx] is None
        adapter_name = config.adapter_name_
        for proj in all_proj.values():
            if adapter_name in proj.moes_ and hasattr(
                proj.moes_[adapter_name], "router_logits_"
            ):
                if router_logits[idx] is None:
                    router_logits[idx] = []
                router_logits[idx].append(proj.moes_[adapter_name].router_logits_)
                proj.moes_[adapter_name].router_logits_ = None

    for idx, logits in enumerate(router_logits):
        if isinstance(logits, list):
            router_logits[idx] = torch.cat(logits, 0)

    return router_logits
