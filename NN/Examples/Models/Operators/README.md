# Operator-Learning Examples

This folder contains scientific ML operator-learning commands. Their target is a learned map
between functions or fields, rather than a fixed-dimensional classifier.

## Files

- `Fno1dBurgers.lean`: native TorchLean 1D Fourier neural operator for the viscous Burgers dataset.
  The command learns the operator `u0(x) -> u(x,T)` on a fixed grid and can export prediction CSVs
  for plotting.

## Data Boundary

The public Burgers data starts as a `.mat` file. Python owns download and conversion; Lean owns the
typed model, loss, optimizer, training loop, and prediction artifact.

```bash
python3 NN/Examples/Data/prepare_fno1d_burgers.py \
  --download --grid 32 --ntrain 128 --ntest 32
```

The script writes `.npy` arrays under `data/real/fno/`:

```text
burgers_train_X.npy
burgers_train_y.npy
burgers_test_X.npy
burgers_test_y.npy
burgers_meta.json
```

## Commands

Quick CUDA check:

```bash
lake -R -K cuda=true exe torchlean fno1d_burgers --device cuda --steps 1
```

Longer run with a prediction artifact:

```bash
lake -R -K cuda=true exe torchlean fno1d_burgers --device cuda \
  --steps 700 --lr 0.003 \
  --plot-csv data/real/fno/predictions.csv \
  --log data/real/fno/trainlog.json

python3 NN/Examples/Data/plot_fno1d_burgers.py \
  --csv data/real/fno/predictions.csv
```

## Runtime and Verification Boundary

On CUDA, this example uses a real-split FNO path with fused `spectralConv1dRfft` autograd support
and cuFFT-backed kernels. The fused runtime owns tensor parameters, cached Adam moments, prediction,
and tape disposal. Host updates retain their documented floating-point operation order. The generic
real-split FNO model supplies the dense mathematical reference.
Dataset metadata and prediction artifacts are visible in TorchLean; CUDA/cuFFT remain native runtime
boundaries.

For PINN residual certificates and scientific ML verification artifacts, use
`NN/Examples/Verification/PINN` and `NN/Verification/PINN`. The FNO command is a training/prediction
workflow; a later verification workflow should state which exported artifact it checks.
