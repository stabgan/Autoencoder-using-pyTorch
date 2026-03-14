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

## Tech Stack

| | Technology | Purpose |
|---|---|---|
| 🔥 | PyTorch | Neural network framework |
| 🐍 | Python 3.x | Runtime |
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

## Dataset

[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) — 100,000 ratings (1–5) from 943 users on 1682 movies. Uses the `u1.base` / `u1.test` 80/20 split. The `ml-1m` dataset is loaded for reference but not used in training.

## Known Issues

- The `ml-1m` dataset is loaded at startup but never used — this is dead code carried over from the original tutorial.
- `pd.read_csv` with `sep='::'` triggers a `ParserWarning` in newer pandas versions (multi-char separator falls back to the Python engine).
- No GPU support — the model trains on CPU only. For large datasets, consider moving tensors to CUDA.
- No model checkpointing — trained weights are not saved to disk.

## License

MIT
