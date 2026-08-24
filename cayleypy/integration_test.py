"""End-to-end scenario using the features of the whole series together.

This file lives only on the `integration/all-features` branch: it exercises features that are added by different pull
requests (the model contract and checkpoints, architectures, the trainer with its data sources, Bellman fine-tuning,
ensembles, symmetries and pruning by a lower bound), so it cannot belong to any single one of them.
"""

import os

import pytest
import torch

from . import (
    BfsLowerBound,
    CayleyGraph,
    CayleyGraphDef,
    EnsemblePredictor,
    PermutationGroups,
    Predictor,
    SymmetrizedPredictor,
    SymmetryGroup,
)
from .models import GroupTokenizer, ModelConfig, graph_hash, load_checkpoint
from .train import TrainConfig, Trainer

RUN_SLOW_TESTS = os.getenv("RUN_SLOW_TESTS") == "1"


class _CountingPredictor(Predictor):
    """Predictor counting how many times the search asked it to score children."""

    def __init__(self, graph: CayleyGraph, model):
        super().__init__(graph, model)
        self.score_children_calls = 0

    def score_children(self, states: torch.Tensor) -> torch.Tensor:
        self.score_children_calls += 1
        return super().score_children(states)


def _all_states_with_distances(graph: CayleyGraph) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns all states of the graph with their true distances, computed by exact BFS."""
    layers = graph.bfs(max_layer_size_to_store=None).layers
    states = torch.vstack([torch.as_tensor(layer) for layer in layers.values()])
    distances = torch.hstack([torch.full((len(layer),), int(distance)) for distance, layer in layers.items()])
    return states, distances


def _exact_child_distances(graph_def: CayleyGraphDef, states: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
    """Returns true distances of all children of given states, of shape ``[n_states, n_generators]``."""
    known = {tuple(state): int(distance) for state, distance in zip(states.tolist(), distances.tolist())}
    ans = torch.zeros((states.shape[0], graph_def.n_generators))
    for i, state in enumerate(states.tolist()):
        for j, permutation in enumerate(graph_def.generators_permutations):
            ans[i, j] = known[tuple(state[k] for k in permutation)]
    return ans


def _mean_absolute_error(predictor: Predictor, states: torch.Tensor, expected: torch.Tensor) -> float:
    with torch.no_grad():
        return float((predictor.score_children(states) - expected).abs().mean())


@pytest.mark.skipif(not RUN_SLOW_TESTS, reason="Slow test")
def test_end_to_end_trained_q_model_solves_lrx5_optimally(tmp_path):
    """Trains a Q-model with two heads and uses it in beam search with all the beam features of the series.

    The scenario is: train a QV-model on random walks with anchors, fine-tune it on Bellman targets, save and load it
    as a self-describing checkpoint, ensemble the two checkpoints, average the ensemble over symmetries, and run beam
    search scoring children in one pass, deduplicating the beam by symmetries, banning backtracking moves and pruning
    by a lower bound. All of this is checked against exact BFS on `lrx(5)`.
    """
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    n_generators = graph.definition.n_generators
    states, distances = _all_states_with_distances(graph)
    exact_child_scores = _exact_child_distances(graph.definition, states, distances)
    assert states.shape[0] == 120

    # A Q-model (one output per generator) with a V-head, on top of a residual MLP backbone.
    model_config = ModelConfig(
        model_type="QV",
        backbone_type="RESMLP",
        input_size=5,
        num_classes_for_one_hot=5,
        layers_sizes=[64, 64],
        n_outputs=n_generators,
        graph_hash=graph_hash(graph.definition),
    )

    # Pretraining: sparse Q-labels from random walks, with a few percent of anchors with exact distances.
    pretrain_config = TrainConfig(
        n_epochs=60,
        n_walks=64,
        rw_length=16,
        batch_size=128,
        lr=3e-3,
        lr_min=1e-4,
        anchors_depth=3,
        anchors_fraction=0.05,
        ema_decay=0.99,
        seed=42,
    )
    trainer = Trainer(graph, model_config, pretrain_config)
    pretrain_result = trainer.train()
    assert pretrain_result.losses[-1] < 0.5 * pretrain_result.losses[0]
    pretrain_path = str(tmp_path / "pretrain.pt")
    trainer.save(pretrain_path)

    # The checkpoint is self-describing: the model is rebuilt from it, and it predicts exactly what was saved.
    pretrained_model, pretrained_config = load_checkpoint(pretrain_path, graph_def=graph.definition)
    assert pretrained_config.model_type == "QV"
    assert pretrained_config.backbone_type == "RESMLP"
    assert pretrained_config.n_outputs == n_generators
    with torch.no_grad():
        expected_scores = Predictor(graph, trainer.model_for_inference()).score_children(states)
        assert torch.equal(Predictor(graph, pretrained_model).score_children(states), expected_scores)

    # Fine-tuning on Bellman targets, starting from the pretrained checkpoint with a lower learning rate.
    finetune_config = TrainConfig(
        n_epochs=150,
        n_walks=64,
        rw_length=16,
        batch_size=128,
        lr=1e-3,
        lr_min=1e-5,
        bellman_anchors_depth=1,
        anchors_fraction=0.05,
        ema_decay=0.99,
        seed=42,
        targets="bellman",
    )
    bellman_trainer = Trainer.from_checkpoint(pretrain_path, graph, finetune_config)
    bellman_trainer.train()
    finetune_path = str(tmp_path / "finetune.pt")
    bellman_trainer.save(finetune_path)
    finetuned_model, _ = load_checkpoint(finetune_path, graph_def=graph.definition)

    # Labels of random walks are loose upper estimates of the distance, and Bellman targets fix that: after fine-tuning
    # the Q-values are close to the true distances of the children.
    pretrained_predictor = Predictor(graph, pretrained_model)
    finetuned_predictor = Predictor(graph, finetuned_model)
    error_before = _mean_absolute_error(pretrained_predictor, states, exact_child_scores)
    error_after = _mean_absolute_error(finetuned_predictor, states, exact_child_scores)
    assert error_after < 1.0, error_after
    assert error_after < 0.4 * error_before, (error_before, error_after)

    # The two checkpoints are ensembled, and the ensemble is averaged over the symmetries of the graph (TTA).
    ensemble = EnsemblePredictor([pretrained_predictor, finetuned_predictor], [0.3, 0.7])
    with torch.no_grad():
        manual_sum = 0.3 * pretrained_predictor.score_children(states) + 0.7 * finetuned_predictor.score_children(
            states
        )
        assert torch.allclose(ensemble.score_children(states), manual_sum, atol=1e-5)
    symmetries = SymmetryGroup.reflections(graph.definition)
    symmetries.verify()
    assert len(symmetries.symmetries) == 2
    predictor = SymmetrizedPredictor(ensemble, symmetries)

    # A ball of radius 2 around the central state gives an admissible lower bound on the remaining distance.
    lower_bound = BfsLowerBound(graph, graph.bfs(max_diameter=2, return_all_hashes=True))
    assert lower_bound.radius == 2

    def solve(predictor_to_use: Predictor, state: list[int], distance: int):
        return graph.beam_search(
            start_state=state,
            predictor=predictor_to_use,
            beam_width=5,
            max_steps=30,
            return_path=True,
            use_child_scores=True,
            canonical_dedup=symmetries,
            non_backtracking=True,
            lower_bound=lower_bound,
            prune_above=distance,
        )

    # Every state is solved, and every found path is optimal (its length is the true distance) and valid (applying it
    # to the start state gives the central state).
    n_solved = 0
    for state, distance in zip(states.tolist(), distances.tolist()):
        if distance == 0:
            continue
        result = solve(predictor, state, int(distance))
        assert result.path_found, (state, distance)
        assert result.path_length == distance, (state, distance, result.path_length)
        assert result.path is not None
        assert graph.apply_path(state, result.path).reshape((-1)).tolist() == graph.central_state.tolist()
        n_solved += 1
    assert n_solved == 119

    # The same beam with the Hamming heuristic instead of the model solves far fewer states, so it is the model that
    # does the work here, not the beam features.
    hamming = Predictor(graph, "hamming")
    n_solved_by_hamming = sum(
        int(solve(hamming, state, int(distance)).path_found)
        for state, distance in zip(states.tolist(), distances.tolist())
        if distance > 0
    )
    assert n_solved_by_hamming < 100, n_solved_by_hamming


def test_tokenized_transformer_q_model_composes_with_the_beam():
    """Test that a Q-model with a transformer backbone over a tokenizer can be built, saved and used in the beam.

    Weights are random here - this checks that the architectures of the series compose with each other and with beam
    search, not that they predict anything. They are seeded, so that the search below is reproducible.
    """
    torch.manual_seed(0)
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    n_generators = graph.definition.n_generators
    # Every element of a state is its own token, which is the trivial (but lossless) tokenization of this graph.
    tokenizer_groups = [[1, 5]]
    GroupTokenizer(tokenizer_groups).verify(graph.definition)
    config = ModelConfig(
        model_type="QV",
        backbone_type="TRANSFORMER",
        input_size=5,
        num_classes_for_one_hot=5,
        layers_sizes=[32, 32],
        n_outputs=n_generators,
        tokenizer_groups=tokenizer_groups,
        n_heads=2,
        v_consistency_weight=0.5,
        graph_hash=graph_hash(graph.definition),
    )
    predictor = _CountingPredictor(graph, config.build_model())
    scores = predictor.score_children(torch.tensor([[1, 0, 2, 3, 4], [0, 1, 2, 3, 4]]))
    assert scores.shape == (2, n_generators)

    # The state farthest from the central state of this graph (10 moves away), so that the search really runs its loop
    # instead of finding the goal among the children of the start state.
    start_state = [1, 0, 4, 3, 2]
    predictor.score_children_calls = 0
    result = graph.beam_search(
        start_state=start_state,
        predictor=predictor,
        beam_width=30,
        max_steps=40,
        return_path=True,
        use_child_scores=True,
        canonical_dedup=SymmetryGroup.reflections(graph.definition),
        non_backtracking=True,
    )
    assert result.path_found
    assert result.path is not None
    assert graph.apply_path(start_state, result.path).reshape((-1)).tolist() == graph.central_state.tolist()
    # The search asked the model for children scores on several of its levels (it only needs them where the layer does
    # not fit in the beam), so the options above were applied to the model's scores and not bypassed.
    assert predictor.score_children_calls > 1
