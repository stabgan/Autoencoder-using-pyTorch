# Stacked Autoencoder for Movie Recommendations

A PyTorch implementation of a stacked autoencoder (SAE) trained on the MovieLens 100K dataset to learn latent representations of user–movie rating patterns.

## What It Does

The model learns to reconstruct a user's sparse rating vector (1682 movies) through a bottleneck layer, forcing it to capture meaningful patterns in viewing preferences. Once trained, the reconstructed output predicts ratings for movies a user hasn't seen — the basis for a recommendation system.

## Architecture

```
Input (1682) → FC(20) → FC(10) → FC(20) → Output (1682)
              σ         σ         σ        (linear)
```

- 4-layer fully connected network with sigmoid activations (linear output layer)
- Loss: MSE with a correction factor for sparsity (only rated movies contribute)
- Optimizer: RMSprop (lr=0.01, weight_decay=0.5)
- Trains for 200 epochs, iterating over each user per epoch

## Dataset

[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) — 100,000 ratings (1–5) from 943 users on 1682 movies. The repo includes both `ml-100k` and `ml-1m` as zip files in `AutoEncoders/`.

Training/test split uses the provided `u1.base` / `u1.test` (80/20).

## Dependencies

- Python 3.x
- PyTorch
- NumPy
- pandas

```bash
pip install torch numpy pandas
```

## Usage

1. Extract the dataset:
   ```bash
   cd AutoEncoders
   unzip ml-100k.zip
   unzip ml-1m.zip
   ```

2. Run training and evaluation:
   ```bash
   python ae.py
   ```

   The script trains for 200 epochs, printing the loss each epoch, then reports the test RMSE.

## Known Issues and Deprecations

The code was written for an older version of PyTorch (~0.3) and will not run correctly on modern PyTorch without fixes:

| Issue | Location | Severity |
|---|---|---|
| `torch.autograd.Variable` is deprecated | Throughout | Warning — tensors have built-in autograd since PyTorch 0.4 |
| `loss.data[0]` raises `IndexError` on modern PyTorch | Training & test loops | **Breaking** — must use `loss.item()` for scalar tensors (changed in 0.5) |
| `target.require_grad = False` is a typo | Training & test loops | Bug — the correct attribute is `requires_grad`; this line silently does nothing |
| `optimizer.zero_grad()` is never called | Training loop | Bug — gradients accumulate across all users instead of resetting per step |
| `ml-1m` data is loaded but never used | Lines 13–16 | Dead code — only `ml-100k` is used for training and testing |
| `pd.read_csv` with `sep='::'` | `ml-1m` loading | Warning — multi-char separator triggers `ParserWarning` in newer pandas |

## License

MIT
