# Operator-Learning Examples

This folder contains operator learning and physics-informed neural field examples.

## Files

- `ComplexRegression.lean`: fits both components of complex binary32 weights and bias using a
  real loss, then checks a held-out complex prediction and an optional checkpoint round trip.
- `Fno1dBurgers.lean`: native TorchLean 1D Fourier neural operator for the viscous Burgers dataset.
  The command learns the operator `u0(x) -> u(x,T)` on a fixed grid and can export prediction CSVs
  for plotting.
- `Pinn.lean`: trains a tanh MLP on `u'' = -2` and the boundary conditions `u(-1) = u(1) = 0`.
  Lean computes coordinate derivatives, differentiates the residual with respect to parameters,
  and updates the model. No Python process, dataset download, or solution labels are needed.

## A Lean-only PINN

```bash
scripts/lake.sh exe torchlean pinn
# Or choose the number of SGD steps:
scripts/lake.sh exe torchlean pinn 300
```

The default uses 1,500 steps. In the checked CPU run, the residual objective decreased from
`2.244093` to `0.006179`; at the held-out point `x = -0.75`, the prediction was `0.446573`
against the exact value `0.4375`. These are training diagnostics, not a uniform error bound.
The exact solution `1 - x²` is used only in the final report.

The reusable operations are `autograd.model.derivative` and `autograd.model.derivativeVjp`.
They support arbitrary input directions and repeated or mixed derivatives. The example chooses
one spatial coordinate and a second-order equation; those choices are not API restrictions.

## Complex Parameters

```bash
scripts/lake.sh exe torchlean complex_regression 50 /tmp/complex-regression.state
```

The real-coordinate gradient comes from `autograd.complex.grad`, followed by `nn.sgdStep`.
The example fits `z ↦ (2-i)*z + (1/2+3i/4)` at four training points. Predictions and saved state
retain both components. This is a small forward-AD training example, not the ordinary real-data
supervised trainer or a native complex GPU training benchmark.

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
real-split `nn.models.fno` model supports any spatial rank and any batch shape. Its default
`spectralPath := .automatic` transforms each spatial axis separately, using cuFFT in eager CUDA
execution and dense per-axis operations otherwise. Set `spectralPath := .denseReference` to use
the full-grid DFT matrices instead. These two implementations share weights and checkpoint layout;
tests compare their predictions, input gradients, and every parameter gradient.
The Burgers command's fused one-sided model has a different parameter layout.
Dataset metadata and prediction artifacts are visible in TorchLean; CUDA/cuFFT remain native runtime
boundaries.

For PINN residual certificates and scientific ML verification artifacts, use
`NN/Examples/Verification/PINN` and `NN/Verification/PINN`. The FNO command is a training/prediction
workflow; a later verification workflow should state which exported artifact it checks.
