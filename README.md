# Stacked Autoencoder for Movie Recommendations

A PyTorch implementation of a stacked autoencoder trained on MovieLens 100K to predict user movie ratings.

## What It Does

The model compresses each user's sparse rating vector (1682 movies) through a bottleneck layer, forcing it to learn latent patterns in viewing preferences. The reconstructed output predicts ratings for unseen movies — forming the basis of a collaborative filtering recommendation system.

## Architecture

```
Input (1682) → FC(20) → FC(10) → FC(20) → Output (1682)
              σ         σ         σ        (linear)
```

- **Encoder:** 1682 → 20 → 10 (sigmoid activations)
- **Decoder:** 10 → 20 → 1682 (linear output)
- **Loss:** MSE with sparsity correction (only rated movies contribute)
- **Optimizer:** RMSprop (lr=0.01, weight_decay=0.5)
- **Epochs:** 200, iterating over each user per epoch

## 🛠 Tech Stack

| | Technology | Purpose |
|---|---|---|
| 🔥 | PyTorch | Neural network framework |
| 🐍 | Python 3.8+ | Runtime |
| 🔢 | NumPy | Numerical operations |
| 🐼 | pandas | Data loading and preprocessing |
| 🎬 | MovieLens 100K | Training and evaluation dataset |

## Dependencies

```bash
pip install torch numpy pandas
```

## Usage

1. Extract the datasets:
   ```bash
   cd AutoEncoders
   unzip ml-100k.zip
   unzip ml-1m.zip
   ```

2. Train and evaluate:
   ```bash
   python ae.py
   ```

   Prints per-epoch training loss, then reports test RMSE.  
   Trained weights are saved to `sae_weights.pth`.

## Dataset

[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) — 100,000 ratings (1–5) from 943 users on 1682 movies. Uses the `u1.base` / `u1.test` 80/20 split.

## Modernization Notes

The codebase has been updated for compatibility with current PyTorch (2.x):

- Removed dead `ml-1m` loading code that was never used
- Replaced `target.requires_grad = False` with `target = input.clone().detach()`
- Replaced `target.data > 0` with `target > 0` (`.data` accessor is discouraged)
- Added `model.eval()` + `torch.no_grad()` for the test/inference loop
- Added `model.train()` before the training loop
- Added CUDA/CPU device selection
- Added model weight saving via `torch.save()`
- Wrapped execution in `if __name__ == "__main__"` guard
- Used `os.path` for file paths (works from any working directory)
- Refactored into clean functions (`load_data`, `convert`, `train`, `test`, `main`)
- Modernized `super()` call (no need to pass class name in Python 3)
- Renamed `input` variable to `input_data` to avoid shadowing the built-in

## ⚠️ Known Issues

- No GPU memory optimization — the full user×movie matrix is held in RAM. For larger datasets, consider batched DataLoader.
- No hyperparameter tuning or early stopping.
- The bottleneck (10 units) is very small; larger hidden layers may improve accuracy.

## License

MIT
