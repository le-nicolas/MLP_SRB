# Load2Moment

Load2Moment is a crafty Multi-Layer Perceptron for **Statics of a Simply Supported Rigid Beam**.

This project generates analytical statics data for a point load on a beam and
trains a small neural network (MLP) to approximate the mapping:

`force position -> [left reaction, right reaction, maximum bending moment]`

## What It Does

- Computes exact support reactions and maximum bending moment from equilibrium.
- Trains a one-hidden-layer MLP (implemented from scratch in NumPy).
- Reports test metrics (`MAE`, `RMSE`, `R^2`) for all three targets.
- Plots analytical curves and MLP predictions for visual comparison.

## Governing Equations

For beam length `L`, point load `F`, and load position `x`:

- `R_left = F * (L - x) / L`
- `R_right = F * x / L`
- `M_max = R_left * x`

These are used as ground truth for supervised learning.

## Requirements

- Python 3.9+
- `numpy`
- `matplotlib`

Install dependencies:

```bash
pip install -r requirements.txt
```

For development/testing:

```bash
pip install -r requirements-dev.txt
```

## Quick Start

Run with defaults:

```bash
python main.py
```

Run in headless mode (no plot window):

```bash
python main.py --no-plot
```

Custom training run:

```bash
python main.py --length 12 --force 150 --samples 300 --hidden-units 24 --epochs 7000 --learning-rate 0.015
```

## CLI Options

```text
--length         Beam length in meters (default: 10.0)
--force          Applied point load in Newtons (default: 100.0)
--margin         Margin from supports for generated samples (default: 0.1)
--samples        Number of generated samples (default: 200)
--test-ratio     Test split ratio in (0, 1) (default: 0.2)
--hidden-units   Hidden neurons in MLP (default: 16)
--learning-rate  Gradient descent learning rate (default: 0.02)
--epochs         Training epochs (default: 5000)
--seed           Random seed (default: 42)
--no-plot        Disable plotting
--verbose        Print loss checkpoints during training
```

## Testing

Run the test suite:

```bash
pytest -q
```

## Expected Output

The script prints a short run summary and a table similar to:

```text
Test-set metrics
------------------------------------------------------------------------------------
Target                                          MAE         RMSE        R^2
------------------------------------------------------------------------------------
Reaction at left support (N)                 ...
Reaction at right support (N)                ...
Maximum bending moment (Nm)                  ...
------------------------------------------------------------------------------------
```

Low errors and `R^2` near `1.0` indicate the network learned the statics mapping well.

## Project Structure

- `main.py`: data generation, MLP training, evaluation, and plotting.
- `LICENSE`: MIT License.

## Notes

- This repository demonstrates supervised learning on a deterministic mechanics
  problem. It is educational and not intended as a production-grade structural
  analysis tool.
