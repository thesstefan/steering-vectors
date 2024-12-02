from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Protocol

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from .layer_matching import (
    LayerType,
    ModelLayerConfig,
    guess_and_enhance_layer_config,
)
from .steering_vector import SteeringVector
from .token_utils import fix_pad_token
from .train_utils import extract_activations_raw, get_token_index
from .utils import batchify


class LossFn(Protocol):
    def __call__(
        self, base_activations: Tensor, steered_activations: Tensor
    ) -> Tensor: ...


def extract_activations_single(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    training_prompts: Sequence[str],
    layers: list[int],
    layer_type: LayerType = "decoder_block",
    layer_config: ModelLayerConfig | None = None,
    move_to_cpu: bool = False,
    read_token_index: int | Callable[[str], int] = -1,
    no_grad: bool = True,
    show_progress: bool = False,
    batch_size: int = 1,
    tqdm_desc: str = "Extracting activations",
) -> dict[int, Tensor]:
    fix_pad_token(tokenizer)
    layer_config = guess_and_enhance_layer_config(model, layer_config, layer_type)
    acts_by_layer: dict[int, list[Tensor]] = defaultdict(list)

    for prompts in batchify(
        training_prompts,
        batch_size=batch_size,
        show_progress=show_progress,
        tqdm_desc=tqdm_desc,
    ):
        base_indices: list[int] = []

        for prompt in prompts:
            base_indices.append(get_token_index(None, read_token_index, prompt))

        base_acts = extract_activations_raw(
            model,
            tokenizer,
            prompts,
            layer_type=layer_type,
            layer_config=layer_config,
            layers=layers,
            read_token_indices=base_indices,
            no_grad=no_grad,
        )

        for layer_num, act in base_acts.items():
            if move_to_cpu:
                act = act.cpu()
            acts_by_layer[layer_num].append(act)

    return {
        layer_num: torch.stack(activations, dim=0)
        for layer_num, activations in acts_by_layer.items()
    }


def optimize_steering_objective(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    loss_fn: LossFn,
    training_prompts: Sequence[str],
    source_acts_by_layer: dict[int, Tensor],
    source_layer: int,
    target_layer: int,
    num_vectors: int = 1,
    num_steps: int = 50,
    layer_type: LayerType = "decoder_block",
    read_token_index: int | Callable[[str], int] = -1,
    layer_config: ModelLayerConfig | None = None,
    show_progress: bool = False,
    batch_size: int = 1,
    move_to_cpu: bool = False,
    tqdm_desc: str = "Training steering vector",
) -> tuple[list[SteeringVector], list[list[float]]]:
    all_losses: list[list[float]] = []
    steering_vectors: list[SteeringVector] = []

    for vec_i in tqdm(range(num_vectors)):
        losses: list[float] = []
        optimized_steering_vector = nn.Parameter(
            torch.rand(
                source_acts_by_layer[source_layer][0].shape[-1],
                device=next(model.parameters()).device,
            ),
        )
        optimizer = AdamW(
            [optimized_steering_vector],
            lr=1e-3,
            betas=(0.9, 0.98),
            weight_decay=0.0,
            amsgrad=True,
        )

        for step in range(num_steps):
            optimizer.zero_grad()

            steering_vector = SteeringVector(
                {source_layer: optimized_steering_vector},
                layer_type,
            )

            with steering_vector.apply(model):
                steered_activations = extract_activations_single(
                    model,
                    tokenizer,
                    training_prompts,
                    layers=[target_layer],
                    layer_type=layer_type,
                    layer_config=layer_config,
                    move_to_cpu=move_to_cpu,
                    read_token_index=read_token_index,
                    show_progress=show_progress,
                    batch_size=batch_size,
                    tqdm_desc=tqdm_desc,
                    no_grad=False,
                )

            loss = loss_fn(
                source_acts_by_layer[target_layer], steered_activations[target_layer]
            )
            _ = loss.backward()
            _ = optimizer.step()

            with torch.no_grad():
                optimized_steering_vector.data = nn.functional.normalize(
                    optimized_steering_vector.data, dim=0
                )

                losses.append(loss.detach().item())

        steering_vectors.append(
            SteeringVector(
                {source_layer: optimized_steering_vector.data.detach()},
                layer_type,
            )
        )

        all_losses.append(losses)

    return steering_vectors, all_losses


def unsupervised_train_steering_vector(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    loss_fn: LossFn,
    training_prompts: Sequence[str],
    source_layer: int,
    target_layer: int,
    layer_type: LayerType = "decoder_block",
    layer_config: ModelLayerConfig | None = None,
    move_to_cpu: bool = False,
    read_token_index: int | Callable[[str], int] = -1,
    show_progress: bool = False,
    batch_size: int = 1,
    tqdm_desc: str = "Training steering vector",
) -> tuple[list[SteeringVector], list[list[float]]]:
    layer_config = guess_and_enhance_layer_config(model, layer_config, layer_type)

    for param in model.parameters():
        param.requires_grad = False

    base_activations_by_layer = extract_activations_single(
        model=model,
        tokenizer=tokenizer,
        training_prompts=training_prompts,
        layers=[source_layer, target_layer],
        layer_type=layer_type,
        layer_config=layer_config,
        move_to_cpu=move_to_cpu,
        read_token_index=read_token_index,
        show_progress=show_progress,
        batch_size=batch_size,
        tqdm_desc=tqdm_desc,
        no_grad=False,
    )

    return optimize_steering_objective(
        model=model,
        tokenizer=tokenizer,
        training_prompts=training_prompts,
        loss_fn=loss_fn,
        source_acts_by_layer=base_activations_by_layer,
        source_layer=source_layer,
        target_layer=target_layer,
        layer_type=layer_type,
        layer_config=layer_config,
        move_to_cpu=move_to_cpu,
        read_token_index=read_token_index,
        show_progress=show_progress,
        batch_size=batch_size,
        tqdm_desc=tqdm_desc,
    )
