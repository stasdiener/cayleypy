from .config import TrainConfig


def test_positional_arguments_are_not_shifted_by_targets():
    # Fields added to this dataclass go last, so that a caller passing the earlier ones positionally keeps working.
    config = TrainConfig(100, 128, 20, "classic")
    assert config.n_epochs == 100
    assert config.rw_mode == "classic"
    assert config.targets == "walks"
