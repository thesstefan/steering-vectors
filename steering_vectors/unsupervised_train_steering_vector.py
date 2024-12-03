from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence

import torch
from torch import Tensor, nn
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


# TODO(stefanache): Basically taken from the MELBO implementation.
#                   Figure out if it can be done in a better way.
def project_to_orthogonal_subspace(
    vec: Tensor, subspace: Tensor, magnitude: float
) -> Tensor:
    U = subspace.t() / magnitude
    return vec - U @ U.t() @ vec


def project_to_sphere_tangent_space(
    vec: Tensor, point_on_shpere: Tensor, radius: float
) -> Tensor:
    return vec - point_on_shpere * (point_on_shpere @ vec) / radius


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
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    create_optimizer_fn: Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer],
    training_prompts: Sequence[str],
    source_acts_by_layer: dict[int, Tensor],
    source_layer: int,
    target_layer: int,
    learned_vectors: torch.Tensor,
    vec_magnitude: float = 1.0,
    num_steps: int = 50,
    layer_type: LayerType = "decoder_block",
    read_token_index: int | Callable[[str], int] = -1,
    layer_config: ModelLayerConfig | None = None,
    show_progress: bool = False,
    batch_size: int = 1,
    move_to_cpu: bool = False,
    tqdm_desc: str = "Training steering vector",
) -> tuple[torch.Tensor, list[float]]:
    layer_width = source_acts_by_layer[source_layer][0].shape[-1]

    optimized_steering_vector = nn.Parameter(
        nn.functional.normalize(
            project_to_orthogonal_subspace(
                torch.rand(layer_width, device=next(model.parameters()).device),
                subspace=learned_vectors,
                magnitude=vec_magnitude,
            ),
            dim=0,
        )
        * vec_magnitude,
    )
    optimizer = create_optimizer_fn([optimized_steering_vector])

    loss_history: list[float] = []
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

        with torch.no_grad():
            assert optimized_steering_vector.grad is not None

            optimized_steering_vector.grad = project_to_sphere_tangent_space(
                vec=project_to_orthogonal_subspace(
                    optimized_steering_vector.grad,
                    subspace=learned_vectors,
                    magnitude=vec_magnitude,
                ),
                point_on_shpere=optimized_steering_vector.data,
                radius=vec_magnitude,
            )

        _ = optimizer.step()

        with torch.no_grad():
            optimized_steering_vector.data = nn.functional.normalize(
                optimized_steering_vector.data, dim=0
            )
            loss_history.append(float(loss.detach().item()))

    return (
        optimized_steering_vector.data.detach(),
        loss_history,
    )


def unsupervised_train_steering_vector(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    training_prompts: Sequence[str],
    source_layer: int,
    target_layer: int,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    create_optimizer_fn: Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer],
    num_steps: int = 50,
    num_vectors: int = 1,
    layer_type: LayerType = "decoder_block",
    layer_config: ModelLayerConfig | None = None,
    move_to_cpu: bool = False,
    read_token_index: int | Callable[[str], int] = -1,
    show_progress: bool = False,
    batch_size: int = 1,
    tqdm_desc: str = "Training steering vector",
) -> list[tuple[SteeringVector, list[float]]]:
    layer_config = guess_and_enhance_layer_config(model, layer_config, layer_type)

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

    layer_width = base_activations_by_layer[source_layer][0].shape[-1]
    learned_vectors = torch.zeros(
        num_vectors,
        layer_width,
        device=next(model.parameters()).device,
    )

    vectors_and_losses: list[tuple[SteeringVector, list[float]]] = []
    for vec_i in tqdm(range(num_vectors)):
        raw_steering_vector, loss_history = optimize_steering_objective(
            model=model,
            tokenizer=tokenizer,
            training_prompts=training_prompts,
            learned_vectors=learned_vectors,
            loss_fn=loss_fn,
            create_optimizer_fn=create_optimizer_fn,
            num_steps=num_steps,
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

        learned_vectors[vec_i, :] = raw_steering_vector

        vectors_and_losses.append(
            (
                SteeringVector({source_layer: raw_steering_vector}, layer_type),
                loss_history,
            )
        )

    return vectors_and_losses
