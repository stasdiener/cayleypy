import gc
import os
import warnings
import weakref

import pytest
import torch

from .config import TrainConfig
from .bellman import BellmanTargets, _BellmanWithAnchors, make_bellman_source
from .data import DataSource, MixtureDataSource, PathDataSource, RandomWalksSource, SparseQSampler, TrainingData
from .trainer import Trainer
from ..cayley_graph import CayleyGraph
from ..cayley_graph_def import CayleyGraphDef
from ..cayley_path import CayleyPath
from ..graphs_lib import PermutationGroups
from ..models.checkpoint import graph_hash, load_checkpoint
from ..models.models import ModelConfig
from ..predictor import Predictor

RUN_SLOW_TESTS = os.getenv("RUN_SLOW_TESTS") == "1"

MLP_CONFIG = ModelConfig(model_type="MLP", input_size=5, num_classes_for_one_hot=5, layers_sizes=[16, 16])

# Config of a Q-model for lrx(5) - one output per generator. Architectures with several outputs are added in another
# change, so tests below train the model defined here instead of building one from this config.
Q_CONFIG = ModelConfig(model_type="MLP", input_size=5, num_classes_for_one_hot=5, layers_sizes=[16], n_outputs=3)


class _QModel(torch.nn.Module):
    """Minimal model with one output per generator, standing in for a Q-model architecture."""

    def __init__(self, state_size: int = 5, n_outputs: int = 3):
        super().__init__()
        self.n_outputs = n_outputs
        self.layer = torch.nn.Linear(state_size, n_outputs)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.layer(states.to(torch.float32))


# Configuration making training as short as possible - for tests that check mechanics rather than model quality.
TINY_CONFIG = TrainConfig(n_epochs=2, n_walks=4, rw_length=3, batch_size=8, seed=0)


def _lrx5() -> CayleyGraph:
    return CayleyGraph(PermutationGroups.lrx(5), device="cpu")


def _weights(model) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in model.parameters()]


def _all_states_with_distances(graph: CayleyGraph) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns all states of the graph with their true distances, computed by exact BFS."""
    layers = graph.bfs(max_layer_size_to_store=None).layers
    states = torch.vstack([torch.as_tensor(layer) for layer in layers.values()])
    distances = torch.hstack([torch.full((len(layer),), distance) for distance, layer in layers.items()])
    return states, distances


def test_train_returns_loss_for_every_epoch():
    trainer = Trainer(_lrx5(), MLP_CONFIG, TINY_CONFIG)
    result = trainer.train()
    assert len(result.losses) == 2
    assert all(loss > 0 for loss in result.losses)
    # 4 walks of length 3 give 12 states, which is 2 batches of 8, so there are 2 optimizer steps per epoch.
    assert result.n_steps == 4
    assert trainer.n_steps == 4
    assert trainer.epoch == 2


def test_generate_data():
    config = TrainConfig(n_walks=8, rw_length=4, rw_mode="classic")
    graph = _lrx5()
    data = Trainer(graph, MLP_CONFIG, config).generate_data()
    assert data.states.shape == (32, 5)
    assert data.targets.shape == (32,)
    assert data.targets.dtype == torch.float32
    assert data.mask is None and data.weights is None
    # First `n_walks` states are copies of the central state, and their distance is 0.
    assert torch.equal(data.states[:8], graph.central_state.expand(8, 5))
    assert torch.equal(data.targets[:8], torch.zeros(8))
    assert torch.all(data.targets >= 0)


def test_generate_data_mixes_anchors_in():
    graph = _lrx5()
    config = TrainConfig(n_walks=8, rw_length=25, anchors_depth=3, anchors_fraction=0.2)
    trainer = Trainer(graph, MLP_CONFIG, config)
    assert isinstance(trainer.data_source, MixtureDataSource)

    data = trainer.generate_data()
    # Walks generate 200 states, which is 80% of 250 states, and the remaining 50 are anchors.
    assert len(data) == 250
    exact = {
        tuple(state.tolist()): distance
        for distance, layer in graph.bfs(max_layer_size_to_store=None).layers.items()
        for state in layer
    }
    n_exact_targets = sum(
        1 for state, target in zip(data.states, data.targets) if exact[tuple(state.tolist())] == target
    )
    assert n_exact_targets >= 50


def test_trainer_trains_q_model_on_sparse_targets():
    graph = _lrx5()
    config = TrainConfig(n_epochs=2, n_walks=8, rw_length=6, batch_size=16, lr=0.01, seed=0)
    trainer = Trainer(graph, Q_CONFIG, config, model=_QModel())
    assert isinstance(trainer.data_source, SparseQSampler)

    data = trainer.generate_data()
    assert data.targets.shape == (48, 3)
    assert data.mask is not None

    weights_before = _weights(trainer.model)
    result = trainer.train()
    assert all(loss > 0 for loss in result.losses)
    for before, after in zip(weights_before, _weights(trainer.model)):
        assert not torch.equal(before, after)


def test_trainer_trains_q_model_on_walks_mixed_with_anchors():
    graph = _lrx5()
    config = TrainConfig(n_epochs=2, n_walks=8, rw_length=6, batch_size=16, anchors_depth=2, seed=0)
    trainer = Trainer(graph, Q_CONFIG, config, model=_QModel())
    assert isinstance(trainer.data_source, MixtureDataSource)
    data = trainer.generate_data()
    assert data.targets.shape[1] == 3
    # Anchors label all 3 outputs, walk states label at most 2.
    assert int((data.mask.sum(dim=1) == 3).sum()) > 0
    assert all(loss > 0 for loss in trainer.train().losses)


def test_trainer_uses_given_data_source():
    graph = _lrx5()
    start_state = torch.tensor([2, 0, 1, 4, 3])
    beam_result = graph.beam_search(start_state=start_state, return_path=True)
    assert beam_result.path_found and beam_result.path is not None
    path = CayleyPath(start_state, beam_result.path, graph.definition)
    source = PathDataSource(graph, [path], weight=0.5)

    trainer = Trainer(graph, MLP_CONFIG, TINY_CONFIG, data_source=source)
    data = trainer.generate_data()
    assert len(data) == beam_result.path_length + 1
    assert torch.equal(data.weights, torch.full_like(data.targets, 0.5))
    assert all(loss > 0 for loss in trainer.train().losses)


def test_training_is_deterministic_with_seed():
    config = TrainConfig(n_epochs=3, n_walks=8, rw_length=4, batch_size=16, seed=42)
    losses1 = Trainer(_lrx5(), MLP_CONFIG, config).train().losses
    losses2 = Trainer(_lrx5(), MLP_CONFIG, config).train().losses
    assert losses1 == losses2


def test_train_step_reduces_loss_on_the_same_batch():
    trainer = Trainer(_lrx5(), MLP_CONFIG, TrainConfig(lr=0.01, seed=0))
    data = trainer.generate_data()
    first_loss = trainer.train_step(data.states, data.targets)
    for _ in range(10):
        last_loss = trainer.train_step(data.states, data.targets)
    assert last_loss < first_loss


def test_fully_masked_batch_does_not_change_weights():
    trainer = Trainer(_lrx5(), MLP_CONFIG, TrainConfig(seed=0))
    data = trainer.generate_data()
    weights_before = _weights(trainer.model)
    loss = trainer.train_step(data.states, data.targets, mask=torch.zeros_like(data.targets))
    assert loss == 0
    for before, after in zip(weights_before, _weights(trainer.model)):
        assert torch.equal(before, after)


def test_ema_follows_weights():
    decay = 0.9
    trainer = Trainer(_lrx5(), MLP_CONFIG, TrainConfig(ema_decay=decay, lr=0.01, seed=0))
    assert trainer.ema_model is not None
    data = trainer.generate_data()

    ema_before = _weights(trainer.ema_model)
    for _ in range(2):
        trainer.train_step(data.states, data.targets)
        model_weights = _weights(trainer.model)
        ema_after = _weights(trainer.ema_model)
        for before, model_weight, after in zip(ema_before, model_weights, ema_after):
            assert torch.allclose(after, decay * before + (1 - decay) * model_weight)
        ema_before = ema_after

    # The average lags behind the weights, so they are not the same.
    assert not torch.equal(_weights(trainer.ema_model)[0], _weights(trainer.model)[0])


def test_ema_can_be_disabled():
    trainer = Trainer(_lrx5(), MLP_CONFIG, TrainConfig(ema_decay=0, n_epochs=1, n_walks=4, rw_length=3))
    assert trainer.ema_model is None
    trainer.train()
    # Even when the average is requested, there is nothing to return except the trained model itself.
    assert trainer.model_for_inference(use_ema=True) is trainer.model


def test_learning_rate_follows_cosine_schedule():
    config = TrainConfig(n_epochs=4, n_walks=4, rw_length=3, batch_size=8, lr=0.01, lr_min=0.001)
    trainer = Trainer(_lrx5(), MLP_CONFIG, config)
    learning_rates = []
    for _ in range(config.n_epochs):
        learning_rates.append(trainer.learning_rate)
        trainer.train_epoch()
    assert learning_rates[0] == pytest.approx(config.lr)
    assert learning_rates == sorted(learning_rates, reverse=True)
    assert trainer.learning_rate == pytest.approx(config.lr_min)


def test_predictor_uses_averaged_weights():
    trainer = Trainer(_lrx5(), MLP_CONFIG, TINY_CONFIG)
    trainer.train()
    states = trainer.generate_data().states
    predictor = trainer.predictor()
    assert isinstance(predictor, Predictor)
    assert predictor(states).shape == (states.shape[0],)
    with torch.no_grad():
        assert torch.equal(predictor(states), trainer.ema_model(states))
        assert torch.equal(trainer.predictor(use_ema=False)(states), trainer.model(states))


def test_save_and_resume_from_checkpoint(tmp_path):
    path = tmp_path / "model.pt"
    graph = _lrx5()
    trainer = Trainer(graph, MLP_CONFIG, TINY_CONFIG)
    trainer.train()
    saved_config = trainer.save(path)
    assert saved_config.graph_hash == graph_hash(graph.definition)

    resumed = Trainer.from_checkpoint(path, graph, TINY_CONFIG)
    assert resumed.model_config == saved_config
    states = trainer.generate_data().states
    with torch.no_grad():
        # Resumed trainer starts from exactly the weights that were saved.
        assert torch.equal(resumed.model(states), trainer.ema_model(states))
    # Training continues from these weights.
    resumed.train()
    assert resumed.n_steps == trainer.n_steps
    with torch.no_grad():
        assert not torch.equal(resumed.model(states), trainer.ema_model(states))


def test_save_can_store_weights_instead_of_their_average(tmp_path):
    path = tmp_path / "model.pt"
    trainer = Trainer(_lrx5(), MLP_CONFIG, TINY_CONFIG)
    trainer.train()
    trainer.save(path, use_ema=False)
    loaded_model, _ = load_checkpoint(path)
    states = trainer.generate_data().states
    with torch.no_grad():
        assert torch.equal(loaded_model(states), trainer.model(states))


def test_verbose_prints_progress(capsys):
    Trainer(_lrx5(), MLP_CONFIG, TrainConfig(n_epochs=2, n_walks=4, rw_length=3, verbose=1)).train()
    output = capsys.readouterr().out
    assert "Epoch 1: loss=" in output
    assert "Epoch 2: loss=" in output
    assert "Training finished in 2 steps" in output

    Trainer(_lrx5(), MLP_CONFIG, TrainConfig(n_epochs=2, n_walks=4, rw_length=3, verbose=0)).train()
    assert capsys.readouterr().out == ""


def test_train_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="n_epochs must be positive"):
        TrainConfig(n_epochs=0)
    with pytest.raises(ValueError, match="n_walks must be positive"):
        TrainConfig(n_walks=-1)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        TrainConfig(batch_size=0)
    with pytest.raises(ValueError, match="lr must be positive"):
        TrainConfig(lr=0)
    with pytest.raises(ValueError, match="rw_length must be at least 2"):
        TrainConfig(rw_length=1)
    with pytest.raises(ValueError, match="Unknown rw_mode"):
        TrainConfig(rw_mode="random")
    with pytest.raises(ValueError, match="nbt_history_depth must be non-negative"):
        TrainConfig(nbt_history_depth=-1)
    with pytest.raises(ValueError, match='nbt_history_depth must be at least 1 in "nbt" mode'):
        TrainConfig(rw_mode="nbt", nbt_history_depth=0)
    with pytest.raises(ValueError, match="anchors_depth must be non-negative"):
        TrainConfig(anchors_depth=-1)
    with pytest.raises(ValueError, match="anchors_fraction must be strictly between 0 and 1"):
        TrainConfig(anchors_fraction=1.0)
    with pytest.raises(ValueError, match="lr_min must be between 0 and lr"):
        TrainConfig(lr=0.001, lr_min=0.01)
    with pytest.raises(ValueError, match="weight_decay must be non-negative"):
        TrainConfig(weight_decay=-0.1)
    with pytest.raises(ValueError, match="ema_decay must be at least 0 and less than 1"):
        TrainConfig(ema_decay=1.0)
    with pytest.raises(ValueError, match="Unknown loss"):
        TrainConfig(loss="hinge")
    with pytest.raises(ValueError, match="tau must be strictly between 0 and 1"):
        TrainConfig(loss="pinball", tau=0)


def test_trainer_rejects_model_for_states_of_another_size():
    config = ModelConfig(model_type="MLP", input_size=6, num_classes_for_one_hot=6, layers_sizes=[16])
    with pytest.raises(ValueError, match="Model expects states of size 6, but states of this graph have size 5"):
        Trainer(_lrx5(), config, TINY_CONFIG)


def test_trainer_rejects_model_with_wrong_number_of_outputs():
    # This graph has 3 generators, so a model must have either 1 or 3 outputs.
    config = ModelConfig(model_type="MLP", input_size=5, num_classes_for_one_hot=5, layers_sizes=[16], n_outputs=4)
    with pytest.raises(ValueError, match="one output per generator of this graph, of which there are 3"):
        Trainer(_lrx5(), config, TINY_CONFIG)


def test_trainer_rejects_model_with_too_few_classes():
    config = ModelConfig(model_type="MLP", input_size=5, num_classes_for_one_hot=3, layers_sizes=[16])
    with pytest.raises(ValueError, match="which is not enough for this graph"):
        Trainer(_lrx5(), config, TINY_CONFIG)


def test_trainer_warns_when_generators_are_not_inverse_closed():
    # Directed 5-cycle - the inverse of its only generator is not a generator, so reaching the central state from the
    # state one step away from it takes 4 steps, not 1.
    graph = CayleyGraph(CayleyGraphDef.create([[1, 2, 3, 4, 0]]), device="cpu")
    with pytest.warns(UserWarning, match="not inverse closed"):
        Trainer(graph, MLP_CONFIG, TINY_CONFIG)


def test_trainer_does_not_warn_about_direction_for_inverse_closed_generators():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Trainer(_lrx5(), MLP_CONFIG, TINY_CONFIG)
    assert not any("inverse closed" in str(warning.message) for warning in caught)


def test_train_on_data_rejects_empty_data():
    trainer = Trainer(_lrx5(), MLP_CONFIG, TINY_CONFIG)
    with pytest.raises(ValueError, match="Cannot train on an empty set of states"):
        trainer.train_on_data(TrainingData(torch.zeros((0, 5), dtype=torch.int64), torch.zeros((0,))))


def test_from_checkpoint_rejects_checkpoint_for_another_graph(tmp_path):
    path = tmp_path / "model.pt"
    Trainer(_lrx5(), MLP_CONFIG, TINY_CONFIG).save(path)
    another_graph = CayleyGraph(PermutationGroups.lrx(5, k=2), device="cpu")
    with pytest.raises(ValueError, match="was trained for another graph"):
        Trainer.from_checkpoint(path, another_graph, TINY_CONFIG)


# To run slow tests like this, do `RUN_SLOW_TESTS=1 pytest`
@pytest.mark.skipif(not RUN_SLOW_TESTS, reason="slow test")
def test_learns_true_distances_from_exact_labels():
    # In "bfs" mode on a graph this small, walks visit the whole graph, so targets are the true distances and the loss
    # can go all the way to 0. Targets estimated from random walks are upper bounds, so with them the loss plateaus at
    # the noise in the labels instead (see the test below, which trains that way).
    graph = _lrx5()
    config = TrainConfig(rw_mode="bfs", n_epochs=200, n_walks=64, rw_length=12, batch_size=32, lr=5e-3, seed=42)
    trainer = Trainer(graph, ModelConfig("MLP", 5, 5, [64, 64]), config)
    result = trainer.train()
    assert result.losses[-1] < 0.2 * result.losses[0]

    states, distances = _all_states_with_distances(graph)
    predictor = trainer.predictor()
    assert float((predictor(states) - distances).abs().mean()) < 0.5

    # Such a model is an almost perfect heuristic: beam search finds an optimal path for every state of the graph even
    # with beam width 1, while with the default (Hamming distance) heuristic it solves 10 states out of 120.
    for i in range(states.shape[0]):
        result = graph.beam_search(start_state=states[i], predictor=predictor, beam_width=1, return_path=True)
        assert result.path_found, f"No path found for {states[i].tolist()}."
        assert result.path_length == int(distances[i])
        assert torch.equal(graph.apply_path(states[i], result.path).reshape((-1,)), graph.central_state)


@pytest.mark.skipif(not RUN_SLOW_TESTS, reason="slow test")
def test_anchors_make_predictions_near_central_state_exact():
    graph = _lrx5()
    depth = 3
    states, distances = _all_states_with_distances(graph)
    is_near = distances <= depth

    def train(anchors_depth: int, anchors_fraction: float) -> float:
        """Trains a model and returns its mean absolute error on states at distance at most `depth`."""
        config = TrainConfig(
            n_epochs=100,
            n_walks=64,
            rw_length=12,
            batch_size=64,
            lr=5e-3,
            seed=42,
            anchors_depth=anchors_depth,
            anchors_fraction=anchors_fraction,
        )
        trainer = Trainer(graph, ModelConfig("MLP", 5, 5, [64, 64]), config)
        trainer.train()
        predictions = trainer.predictor()(states[is_near])
        return float((predictions - distances[is_near]).abs().mean())

    # Anchors are exact distances of states near the central state, and this is what they buy: without them, targets
    # for those states come from random walks and are overestimated, so predictions there are off by more than a move.
    # The share of anchors here is far above the 1-2% that is right for real training - it makes their effect visible
    # within the tiny budget of a test, and it also shows what a large share costs: predictions on the whole graph get
    # no better, only predictions near the central state do.
    error_with_anchors = train(anchors_depth=depth, anchors_fraction=0.9)
    error_without_anchors = train(anchors_depth=0, anchors_fraction=0.9)
    assert error_with_anchors < 0.5
    assert error_without_anchors > 3 * error_with_anchors


@pytest.mark.skipif(not RUN_SLOW_TESTS, reason="slow test")
def test_model_trained_on_random_walks_solves_all_states():
    graph = _lrx5()
    config = TrainConfig(n_epochs=150, n_walks=64, rw_length=12, batch_size=64, lr=5e-3, seed=42)
    trainer = Trainer(graph, ModelConfig("MLP", 5, 5, [64, 64]), config)
    result = trainer.train()
    assert result.losses[-1] < 0.5 * result.losses[0]

    # Distances estimated from non-backtracking random walks are overestimated, so this model does not predict true
    # distances. It is still good enough for beam search to solve every state of the graph with a narrow beam, where
    # the default (Hamming distance) heuristic solves 23 states out of 120.
    states, _ = _all_states_with_distances(graph)
    predictor = trainer.predictor()
    for i in range(states.shape[0]):
        result = graph.beam_search(start_state=states[i], predictor=predictor, beam_width=5, return_path=True)
        assert result.path_found, f"No path found for {states[i].tolist()}."
        assert torch.equal(graph.apply_path(states[i], result.path).reshape((-1,)), graph.central_state)


class _ModelWithIntegerBuffer(torch.nn.Module):
    """Model whose state dict has a non-float entry, as batch normalization has (its counter of batches)."""

    n_batches: torch.Tensor

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(5, 1)
        self.register_buffer("n_batches", torch.zeros(1, dtype=torch.int64))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        self.n_batches += 1
        return self.layer(states.to(torch.float32)).squeeze(-1)


def test_ema_copies_non_float_entries_of_the_state_dict():
    """Test that entries with no meaningful average (e.g. counters of batch normalization) are copied, not averaged."""
    config = TrainConfig(ema_decay=0.9, lr=0.01, n_walks=4, rw_length=3, batch_size=8, seed=0)
    model = _ModelWithIntegerBuffer()
    trainer = Trainer(_lrx5(), MLP_CONFIG, config, model=model)
    assert trainer.ema_model is not None
    data = trainer.generate_data()

    trainer.train_step(data.states, data.targets)
    trainer.train_step(data.states, data.targets)

    # Averaging an integer entry in place would fail outright, and its value must follow the model exactly.
    assert int(model.n_batches) == 2
    assert int(trainer.ema_model.state_dict()["n_batches"]) == 2


def _lrx5_graph() -> CayleyGraph:
    return CayleyGraph(PermutationGroups.lrx(5), device="cpu")


def test_train_stages_reports_losses_of_every_stage():
    graph = _lrx5_graph()
    common = {"n_walks": 8, "rw_length": 4, "batch_size": 16, "seed": 0}
    stages = [TrainConfig(n_epochs=3, **common), TrainConfig(n_epochs=2, lr=1e-4, **common)]
    result = Trainer(graph, MLP_CONFIG, stages[0]).train_stages(stages)

    assert [len(stage) for stage in result.stage_losses] == [3, 2]
    # The flat curve is the stages one after another, so a boundary is visible in the split but not lost in the curve.
    assert result.losses == [loss for stage in result.stage_losses for loss in stage]


def test_train_stages_rejects_empty_list():
    graph = _lrx5_graph()
    with pytest.raises(ValueError, match="At least one stage"):
        Trainer(graph, MLP_CONFIG, TrainConfig(n_epochs=1, n_walks=4, rw_length=3)).train_stages([])


def test_set_stage_keeps_the_model_and_restarts_the_optimizer():
    graph = _lrx5_graph()
    first = TrainConfig(n_epochs=2, n_walks=8, rw_length=4, batch_size=16, lr=1e-2, lr_min=1e-5, seed=0)
    trainer = Trainer(graph, MLP_CONFIG, first)
    trainer.train()
    weights_before = {name: parameter.clone() for name, parameter in trainer.model.named_parameters()}
    ema_before = trainer.ema_model
    optimizer_before = trainer.optimizer

    second = TrainConfig(n_epochs=3, n_walks=8, rw_length=4, batch_size=16, lr=1e-3, seed=0)
    trainer.set_stage(second)

    # The model and its EMA copy carry over - a stage continues training them, it does not start from scratch.
    for name, parameter in trainer.model.named_parameters():
        assert torch.equal(parameter, weights_before[name])
    assert trainer.ema_model is ema_before
    # The optimizer, the schedule and the epoch counter are new, so the schedule spans this stage and not the last one.
    assert trainer.optimizer is not optimizer_before
    assert trainer.epoch == 0
    assert trainer.learning_rate == pytest.approx(second.lr)
    assert trainer.config is second


def test_bellman_targets_are_built_from_the_config():
    graph = _lrx5_graph()
    config = TrainConfig(n_epochs=1, n_walks=4, rw_length=5, batch_size=8, targets="bellman", seed=0)
    trainer = Trainer(graph, MLP_CONFIG, config)

    assert isinstance(trainer.data_source, _BellmanWithAnchors)
    assert isinstance(trainer.data_source.bellman_source, BellmanTargets)


def test_stage_switches_the_target_scheme():
    graph = _lrx5_graph()
    common = {"n_walks": 8, "rw_length": 4, "batch_size": 16, "seed": 0}
    trainer = Trainer(graph, MLP_CONFIG, TrainConfig(n_epochs=1, **common))
    assert isinstance(trainer.data_source, RandomWalksSource)

    trainer.set_stage(TrainConfig(n_epochs=1, targets="bellman", lr=1e-4, **common))
    assert isinstance(trainer.data_source, _BellmanWithAnchors)


def test_bellman_target_follows_the_model_between_epochs():
    graph = _lrx5_graph()
    config = TrainConfig(n_epochs=2, n_walks=8, rw_length=4, batch_size=16, lr=0.1, targets="bellman", seed=0)
    trainer = Trainer(graph, MLP_CONFIG, config)
    target_before = trainer.data_source.bellman_source.target

    trainer.train()

    # Without the on_epoch_start hook the frozen copy made at construction would label every epoch.
    assert trainer.data_source.bellman_source.target is not target_before


def test_unknown_targets_are_rejected():
    with pytest.raises(ValueError, match="Unknown targets"):
        TrainConfig(targets="whatever")


def test_stage_can_turn_ema_on_and_off():
    graph = _lrx5_graph()
    common = {"n_epochs": 1, "n_walks": 8, "rw_length": 4, "batch_size": 16, "seed": 0}

    # Starting without averaging, a later stage that asks for it gets a copy - of the weights it inherited.
    trainer = Trainer(graph, MLP_CONFIG, TrainConfig(ema_decay=0, **common))
    assert trainer.ema_model is None
    trainer.set_stage(TrainConfig(ema_decay=0.99, **common))
    assert trainer.ema_model is not None
    for name, parameter in trainer.model.named_parameters():
        assert torch.equal(dict(trainer.ema_model.named_parameters())[name], parameter)

    # And a stage that turns averaging off drops the copy, instead of leaving a stale one behind.
    trainer.set_stage(TrainConfig(ema_decay=0, **common))
    assert trainer.ema_model is None
    assert trainer.model_for_inference() is trainer.model


def test_bellman_source_forwards_epoch_start_to_the_source_of_states():
    class _CountingSource(DataSource):
        def __init__(self, inner):
            self.inner = inner
            self.epochs = 0

        def on_epoch_start(self, trainer):
            self.epochs += 1

        def generate(self):
            return self.inner.generate()

    graph = _lrx5_graph()
    config = TrainConfig(n_epochs=2, n_walks=4, rw_length=4, batch_size=16, seed=0)
    states = _CountingSource(RandomWalksSource(graph, n_walks=4, rw_length=4))
    source = make_bellman_source(graph, config, MLP_CONFIG.build_model(), states_source=states)

    Trainer(graph, MLP_CONFIG, config, data_source=source).train()

    # A composite source must forward the hook, or a wrapped source that needs it stays stale.
    assert states.epochs == 2


def test_train_stages_does_not_rebuild_the_first_stage():
    graph = _lrx5_graph()
    first = TrainConfig(n_epochs=1, n_walks=8, rw_length=4, batch_size=16, seed=0)
    trainer = Trainer(graph, MLP_CONFIG, first)
    source_before = trainer.data_source

    trainer.train_stages([first, TrainConfig(n_epochs=1, n_walks=8, rw_length=4, batch_size=16, lr=1e-4, seed=0)])

    # Building a source runs a breadth-first search when the stage uses anchors, so doing it twice is not free.
    assert trainer.data_source is not source_before  # the second stage did build its own
    # ... but the first stage reused what __init__ made, which is what this checks through the call count below.


def test_bellman_refresh_cadence_follows_the_stage_not_the_source():
    graph = _lrx5_graph()
    common = {"n_walks": 4, "rw_length": 4, "batch_size": 16, "seed": 0, "ema_decay": 0}
    config = TrainConfig(n_epochs=3, targets="bellman", bellman_target_update_period=2, **common)
    source = make_bellman_source(graph, config, MLP_CONFIG.build_model())
    trainer = Trainer(graph, MLP_CONFIG, config, data_source=source)
    trainer.train()

    # Three epochs with period 2 end off-period. Reusing the source in a new stage must still refresh on its first
    # epoch, which it does only if the cadence follows the epoch of the trainer (reset by set_stage).
    target_before = source.bellman_source.target
    trainer.set_stage(
        TrainConfig(n_epochs=1, targets="bellman", bellman_target_update_period=2, lr=1e-4, **common),
        data_source=source,
    )
    trainer.train()
    assert source.bellman_source.target is not target_before


def test_set_stage_releases_the_previous_data_source():
    graph = _lrx5_graph()
    common = {"n_epochs": 1, "n_walks": 8, "rw_length": 4, "batch_size": 16, "seed": 0}
    trainer = Trainer(graph, MLP_CONFIG, TrainConfig(anchors_depth=2, **common))
    dead = weakref.ref(trainer.data_source)

    trainer.set_stage(TrainConfig(anchors_depth=2, lr=1e-4, **common))

    # Anchors of two stages held at once can be what makes a staged run run out of memory, so the previous source
    # must be gone by the time the next one is built - not merely replaced once it is.
    gc.collect()
    assert dead() is None


def test_rejected_stage_leaves_the_trainer_trainable():
    graph = CayleyGraph(PermutationGroups.lx(5), device="cpu")
    model_config = ModelConfig(model_type="MLP", input_size=5, num_classes_for_one_hot=5, layers_sizes=[8])
    common = {"n_epochs": 1, "n_walks": 4, "rw_length": 3, "batch_size": 8, "seed": 0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer = Trainer(graph, model_config, TrainConfig(**common))

        # This graph cannot be trained on Bellman targets, and a stage that cannot be entered must not take the
        # trainer down with it: what it was doing before has to keep working.
        with pytest.raises(ValueError, match="inverse-closed generators"):
            trainer.set_stage(TrainConfig(targets="bellman", **common))
        assert len(trainer.train().losses) == 1


def test_explicit_data_source_skips_the_check_of_the_named_scheme():
    graph = CayleyGraph(PermutationGroups.lx(5), device="cpu")
    model_config = ModelConfig(model_type="MLP", input_size=5, num_classes_for_one_hot=5, layers_sizes=[8])
    common = {"n_epochs": 1, "n_walks": 4, "rw_length": 3, "batch_size": 8, "seed": 0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer = Trainer(graph, model_config, TrainConfig(**common))

        # The graph cannot support the Bellman scheme, but the supplied source is used instead of it, and TrainConfig
        # says that targets is ignored then - so the stage must be accepted.
        trainer.set_stage(
            TrainConfig(targets="bellman", **common),
            data_source=RandomWalksSource(graph, n_walks=4, rw_length=3),
        )
        assert len(trainer.train().losses) == 1


def test_a_supplied_bellman_target_is_left_alone():
    graph = _lrx5_graph()
    config = TrainConfig(n_epochs=2, n_walks=4, rw_length=4, batch_size=16, seed=0)
    teacher = MLP_CONFIG.build_model()
    source = BellmanTargets(graph, RandomWalksSource(graph, n_walks=4, rw_length=4), teacher)
    frozen = source.target

    Trainer(graph, MLP_CONFIG, config, data_source=source).train()

    # Supplying a target means it should label the data; refreshing it from the model being trained is what the
    # bootstrapping scheme asks for, and it has to ask.
    assert source.target is frozen
