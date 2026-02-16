"""MLP approximation of static response for a simply supported beam.

Given a point load position on a simply supported rigid beam, this script:
1) Computes analytical reactions and maximum bending moment.
2) Trains a small MLP regressor to learn that mapping.
3) Reports regression metrics and optionally plots analytical vs predicted curves.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

TARGET_NAMES = (
    "Reaction at left support (N)",
    "Reaction at right support (N)",
    "Maximum bending moment (Nm)",
)


@dataclass(frozen=True)
class BeamParameters:
    """Physical parameters for the beam problem."""

    length: float = 10.0
    force: float = 100.0
    edge_margin: float = 0.1

    def validate(self) -> None:
        if self.length <= 0:
            raise ValueError("Beam length must be positive.")
        if self.edge_margin < 0:
            raise ValueError("Edge margin must be non-negative.")
        if self.edge_margin * 2 >= self.length:
            raise ValueError("Edge margin is too large for the beam length.")


@dataclass
class StandardScaler:
    """Small standard scaler (mean/std) for deterministic normalization."""

    mean_: np.ndarray
    scale_: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "StandardScaler":
        mean = values.mean(axis=0, keepdims=True)
        scale = values.std(axis=0, keepdims=True)
        scale = np.where(scale < 1e-12, 1.0, scale)
        return cls(mean_=mean, scale_=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean_) / self.scale_

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale_ + self.mean_


class SimpleMLPRegressor:
    """One hidden-layer MLP with tanh activation and full-batch gradient descent."""

    def __init__(
        self,
        hidden_units: int = 16,
        learning_rate: float = 0.01,
        epochs: int = 5000,
        random_seed: int = 42,
        verbose: bool = False,
    ) -> None:
        if hidden_units < 1:
            raise ValueError("hidden_units must be >= 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if epochs < 1:
            raise ValueError("epochs must be >= 1")

        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_seed = random_seed
        self.verbose = verbose

        self._fitted = False
        self.loss_history: list[float] = []

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "SimpleMLPRegressor":
        sample_count, feature_count = features.shape
        output_count = targets.shape[1]
        rng = np.random.default_rng(self.random_seed)

        self.w1 = rng.normal(
            loc=0.0,
            scale=np.sqrt(2.0 / feature_count),
            size=(feature_count, self.hidden_units),
        )
        self.b1 = np.zeros((1, self.hidden_units))
        self.w2 = rng.normal(
            loc=0.0,
            scale=np.sqrt(2.0 / self.hidden_units),
            size=(self.hidden_units, output_count),
        )
        self.b2 = np.zeros((1, output_count))

        checkpoint = max(1, self.epochs // 10)
        self.loss_history.clear()

        for epoch in range(self.epochs):
            hidden_linear = features @ self.w1 + self.b1
            hidden_activated = np.tanh(hidden_linear)
            predictions = hidden_activated @ self.w2 + self.b2

            errors = predictions - targets
            loss = float(np.mean(np.square(errors)))
            self.loss_history.append(loss)

            grad_predictions = (2.0 / sample_count) * errors
            grad_w2 = hidden_activated.T @ grad_predictions
            grad_b2 = grad_predictions.sum(axis=0, keepdims=True)

            grad_hidden = grad_predictions @ self.w2.T
            grad_hidden_linear = grad_hidden * (1.0 - np.square(hidden_activated))
            grad_w1 = features.T @ grad_hidden_linear
            grad_b1 = grad_hidden_linear.sum(axis=0, keepdims=True)

            self.w2 -= self.learning_rate * grad_w2
            self.b2 -= self.learning_rate * grad_b2
            self.w1 -= self.learning_rate * grad_w1
            self.b1 -= self.learning_rate * grad_b1

            if self.verbose and ((epoch + 1) % checkpoint == 0 or epoch == 0):
                print(f"Epoch {epoch + 1:>5}/{self.epochs}: scaled MSE={loss:.8f}")

        self._fitted = True
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before calling predict().")
        hidden_activated = np.tanh(features @ self.w1 + self.b1)
        return hidden_activated @ self.w2 + self.b2


def compute_beam_response(params: BeamParameters, positions: np.ndarray) -> np.ndarray:
    """Analytical statics response for each force position."""
    positions = np.asarray(positions, dtype=float).reshape(-1)
    if np.any(positions < 0.0) or np.any(positions > params.length):
        raise ValueError("All positions must lie in [0, beam_length].")

    reaction_left = params.force * (params.length - positions) / params.length
    reaction_right = params.force * positions / params.length
    max_bending = reaction_left * positions

    return np.column_stack((reaction_left, reaction_right, max_bending))


def build_dataset(
    params: BeamParameters,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate deterministic training samples along the beam."""
    if sample_count < 10:
        raise ValueError("sample_count must be at least 10.")

    positions = np.linspace(
        params.edge_margin,
        params.length - params.edge_margin,
        sample_count,
    )
    features = positions.reshape(-1, 1)
    targets = compute_beam_response(params, positions)
    return features, targets


def split_train_test(
    features: np.ndarray,
    targets: np.ndarray,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle and split into train/test subsets."""
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be in (0, 1).")

    sample_count = features.shape[0]
    test_count = max(1, int(round(sample_count * test_ratio)))
    if sample_count - test_count < 2:
        raise ValueError("Not enough samples for training after split.")

    rng = np.random.default_rng(seed)
    indices = np.arange(sample_count)
    rng.shuffle(indices)
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]

    return (
        features[train_idx],
        features[test_idx],
        targets[train_idx],
        targets[test_idx],
    )


def regression_metrics(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return MAE, RMSE, and R^2 for each output channel."""
    residuals = predicted_values - true_values
    mae = np.mean(np.abs(residuals), axis=0)
    rmse = np.sqrt(np.mean(np.square(residuals), axis=0))

    ss_res = np.sum(np.square(residuals), axis=0)
    centered = true_values - true_values.mean(axis=0, keepdims=True)
    ss_tot = np.sum(np.square(centered), axis=0)
    r2 = 1.0 - np.divide(ss_res, ss_tot, out=np.zeros_like(ss_res), where=ss_tot > 0.0)

    return mae, rmse, r2


def print_metrics(mae: np.ndarray, rmse: np.ndarray, r2: np.ndarray) -> None:
    """Print a compact evaluation table."""
    print("\nTest-set metrics")
    print("-" * 84)
    print(f"{'Target':42} {'MAE':>12} {'RMSE':>12} {'R^2':>10}")
    print("-" * 84)
    for idx, target_name in enumerate(TARGET_NAMES):
        print(f"{target_name:42} {mae[idx]:12.6f} {rmse[idx]:12.6f} {r2[idx]:10.6f}")
    print("-" * 84)


def plot_predictions(
    params: BeamParameters,
    model: SimpleMLPRegressor,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    features_test: np.ndarray,
    targets_test: np.ndarray,
    predictions_test: np.ndarray,
) -> None:
    """Plot analytical and learned response curves plus test points."""
    dense_positions = np.linspace(
        params.edge_margin,
        params.length - params.edge_margin,
        300,
    ).reshape(-1, 1)
    dense_analytical = compute_beam_response(params, dense_positions[:, 0])
    dense_predicted = y_scaler.inverse_transform(
        model.predict(x_scaler.transform(dense_positions))
    )

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    for idx, axis in enumerate(axes):
        axis.plot(
            dense_positions[:, 0],
            dense_analytical[:, idx],
            color="#1f77b4",
            linewidth=2.0,
            label="Analytical",
        )
        axis.plot(
            dense_positions[:, 0],
            dense_predicted[:, idx],
            color="#ff7f0e",
            linestyle="--",
            linewidth=2.0,
            label="MLP prediction",
        )
        axis.scatter(
            features_test[:, 0],
            targets_test[:, idx],
            color="#2ca02c",
            s=22,
            alpha=0.8,
            label="Test target",
        )
        axis.scatter(
            features_test[:, 0],
            predictions_test[:, idx],
            color="#d62728",
            s=26,
            marker="x",
            alpha=0.9,
            label="Test prediction",
        )
        axis.set_ylabel(TARGET_NAMES[idx])
        axis.grid(alpha=0.25)

    axes[-1].set_xlabel("Force position along beam (m)")
    fig.suptitle("MLP Approximation of Beam Statics", fontsize=14, y=0.995)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an MLP to predict support reactions and maximum bending moment.",
    )
    parser.add_argument("--length", type=float, default=10.0, help="Beam length in meters.")
    parser.add_argument("--force", type=float, default=100.0, help="Point load in Newtons.")
    parser.add_argument(
        "--margin",
        type=float,
        default=0.1,
        help="Distance from supports excluded from generated samples.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of generated data samples.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of samples reserved for test data.",
    )
    parser.add_argument(
        "--hidden-units",
        type=int,
        default=16,
        help="Number of neurons in the hidden layer.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.02,
        help="Gradient descent learning rate.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5000,
        help="Training iterations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for weight init and dataset split.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable matplotlib plots (useful in headless environments).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print training loss checkpoints.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    params = BeamParameters(length=args.length, force=args.force, edge_margin=args.margin)
    params.validate()

    features, targets = build_dataset(params, sample_count=args.samples)
    x_train, x_test, y_train, y_test = split_train_test(
        features,
        targets,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    x_scaler = StandardScaler.fit(x_train)
    y_scaler = StandardScaler.fit(y_train)

    model = SimpleMLPRegressor(
        hidden_units=args.hidden_units,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        random_seed=args.seed,
        verbose=args.verbose,
    )
    model.fit(x_scaler.transform(x_train), y_scaler.transform(y_train))

    y_pred_test = y_scaler.inverse_transform(model.predict(x_scaler.transform(x_test)))
    mae, rmse, r2 = regression_metrics(y_test, y_pred_test)

    print("Beam statics MLP run summary")
    print(f"- Beam length: {params.length:.3f} m")
    print(f"- Point load: {params.force:.3f} N")
    print(f"- Samples: {args.samples} (train={len(x_train)}, test={len(x_test)})")
    print(f"- Final scaled training MSE: {model.loss_history[-1]:.8f}")
    print_metrics(mae, rmse, r2)

    if not args.no_plot:
        plot_predictions(
            params=params,
            model=model,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            features_test=x_test,
            targets_test=y_test,
            predictions_test=y_pred_test,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
