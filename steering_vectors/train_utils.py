from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn
from transformers import PreTrainedTokenizerBase

from steering_vectors.token_utils import adjust_read_indices_for_padding

from .layer_matching import LayerType, ModelLayerConfig
from .record_activations import record_activations


def extract_activations_raw(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    layer_type: LayerType,
    layer_config: ModelLayerConfig,
    layers: list[int] | None,
    read_token_indices: Sequence[int],
    no_grad: bool = True,
) -> dict[int, Tensor]:
    input = tokenizer(prompts, return_tensors="pt", padding=True)
    adjusted_read_indices = adjust_read_indices_for_padding(
        torch.tensor(read_token_indices), input["attention_mask"]
    )
    batch_indices = torch.arange(len(prompts))
    results: dict[int, Tensor] = {}
    with record_activations(
        model,
        layer_type,
        layer_config,
        layer_nums=layers,
        clone_activations=no_grad,
    ) as record:
        model(**input.to(model.device))
    for layer_num, activation in record.items():
        layer_activations = activation[-1][
            batch_indices.to(activation[-1].device),
            adjusted_read_indices.to(activation[-1].device),
        ]

        results[layer_num] = (
            layer_activations.detach() if no_grad else layer_activations
        )

    return results


def get_token_index(
    custom_idx: int | None,
    default_idx: int | Callable[[str], int],
    prompt: str,
) -> int:
    if custom_idx is None:
        if isinstance(default_idx, int):
            return default_idx
        else:
            return default_idx(prompt)
    else:
        return custom_idx
