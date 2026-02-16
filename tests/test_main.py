import numpy as np
import pytest

from main import (
    BeamParameters,
    SimpleMLPRegressor,
    StandardScaler,
    build_dataset,
    compute_beam_response,
    regression_metrics,
    split_train_test,
)


def test_compute_beam_response_known_values() -> None:
    params = BeamParameters(length=10.0, force=100.0, edge_margin=0.1)
    positions = np.array([2.0, 5.0, 8.0])

    actual = compute_beam_response(params, positions)
    expected = np.array(
        [
            [80.0, 20.0, 160.0],
            [50.0, 50.0, 250.0],
            [20.0, 80.0, 160.0],
        ]
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-10)


def test_beam_parameters_validate_rejects_invalid_margin() -> None:
    with pytest.raises(ValueError, match="Edge margin is too large"):
        BeamParameters(length=1.0, force=100.0, edge_margin=0.6).validate()


def test_split_train_test_counts_and_disjoint_sets() -> None:
    features = np.arange(20.0).reshape(-1, 1)
    targets = np.column_stack((features, features, features))

    x_train, x_test, y_train, y_test = split_train_test(
        features,
        targets,
        test_ratio=0.25,
        seed=3,
    )

    assert x_train.shape == (15, 1)
    assert x_test.shape == (5, 1)
    assert y_train.shape == (15, 3)
    assert y_test.shape == (5, 3)
    assert set(x_train[:, 0]).isdisjoint(set(x_test[:, 0]))


def test_mlp_learns_beam_mapping_with_high_r2() -> None:
    params = BeamParameters(length=10.0, force=100.0, edge_margin=0.1)
    features, targets = build_dataset(params, sample_count=200)
    x_train, x_test, y_train, y_test = split_train_test(
        features,
        targets,
        test_ratio=0.2,
        seed=42,
    )

    x_scaler = StandardScaler.fit(x_train)
    y_scaler = StandardScaler.fit(y_train)

    model = SimpleMLPRegressor(
        hidden_units=16,
        learning_rate=0.02,
        epochs=5000,
        random_seed=42,
    )
    model.fit(x_scaler.transform(x_train), y_scaler.transform(y_train))

    y_pred = y_scaler.inverse_transform(model.predict(x_scaler.transform(x_test)))
    _, _, r2 = regression_metrics(y_test, y_pred)

    assert np.all(r2 > np.array([0.999, 0.999, 0.995]))
