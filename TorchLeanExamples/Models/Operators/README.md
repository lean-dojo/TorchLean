# Operator-Learning Examples

Here we train a physics-informed neural network (PINN) entirely in Lean, fit a model with
complex parameters, and learn the 1D viscous Burgers operator with a Fourier neural operator (FNO).

## Files

- `ComplexRegression.lean`: fits both components of complex binary32 weights and bias using a
  real loss, then reports a held-out complex prediction and saves an optional checkpoint.
- `FNOBurgers.lean`: native TorchLean 1D Fourier neural operator for the viscous Burgers dataset.
  The command learns the operator `u0(x) -> u(x,T)` on a fixed grid and can export prediction CSVs
  for plotting.
- `PINNPoisson.lean`: trains a tanh MLP on `u'' = -2` and the boundary conditions
  `u(-1) = u(1) = 0`. Lean computes coordinate derivatives, differentiates the residual with respect
  to parameters, and updates the model. No Python process, dataset download, or solution labels are
  needed.

## A Lean-only PINN

```bash
scripts/lake.sh exe torchlean pinn
# Or choose the number of SGD steps:
scripts/lake.sh exe torchlean pinn 300
```

The default is 1,500 steps. In the recorded CPU run the residual objective fell from `2.244093` to
`0.006179`, and at the held-out point `x = -0.75` the prediction was `0.446573` against the exact
value `0.4375`. These numbers are training diagnostics and give no uniform error bound. The exact
solution `1 - x²` appears only in the final report.

The derivatives come from `autograd.model.derivative` and `autograd.model.derivativeVJP`, which
accept arbitrary input directions and repeated or mixed derivatives. The example uses one spatial
coordinate and a second-order equation, but the API has neither restriction.

## Complex Parameters

```bash
scripts/lake.sh exe torchlean complex_regression \
  --steps 50 --save-checkpoint /tmp/complex-regression.state
```

The example fits `z ↦ (2-i)*z + (1/2+3i/4)` at four points on the unit circle. Parameters,
predictions, and saved state are complex, and only the squared-residual objective is real.
`autograd.complex.grad` differentiates that objective in forward mode with respect to the real and
imaginary coordinates of every parameter, and `nn.sgdStep` updates both. It is a small training
example, separate from the supervised trainer, and does not benchmark native complex GPU training.

## Data Boundary

The public Burgers data starts as a `.mat` file. Python downloads and converts it, and Lean
owns the typed model, loss, optimizer, training loop, and prediction artifact.

```bash
python3 TorchLeanExamples/Data/prepare_fno1d_burgers.py \
  --download --grid 32 --ntrain 128 --ntest 32
```

The script writes these files under `data/real/fno/`:

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
scripts/lake.sh -Kcuda=true exe torchlean fno1d_burgers --device cuda --steps 1
```

Longer run with a prediction artifact:

```bash
scripts/lake.sh -Kcuda=true exe torchlean fno1d_burgers --device cuda \
  --steps 700 --lr 0.003 \
  --plot-csv data/real/fno/predictions.csv \
  --log data/real/fno/trainlog.json

python3 TorchLeanExamples/Data/plot_fno1d_burgers.py \
  --csv data/real/fno/predictions.csv
```

## Runtime and Verification Boundary

The command trains the shared `nn.fno` model on a one-dimensional Burgers grid. The model
supports any spatial rank and batch shape, and the dataset fixes the dimensions used here. Its
default `spectralPath := automatic` transforms each spatial axis separately, calling LibTorch FFT
operations in eager CUDA execution and dense per-axis operations otherwise. Setting
`spectralPath := denseReference` in the model configuration selects full-grid DFT matrices. Both
paths share weights and checkpoint layout. The spectral-convolution checks in
[`TorchLeanTest/CUDA/FFT.lean`](../../../TorchLeanTest/CUDA/FFT.lean) compare
forward values, input gradients, and real and imaginary spectral-weight gradients against the dense
reference. They test the spectral operation on its own, without the rest of a complete FNO model.

TorchLean records the differentiation tape and runs the trainer and optimizer. LibTorch computes
the native tensor values, FFTs included. Dataset metadata and prediction artifacts are visible in
TorchLean, while native execution happens outside the Lean proof checker.

For PINN residual certificates and scientific ML verification artifacts, use
`TorchLeanExamples/Verification/PINN`. The FNO command trains and predicts and emits no
certificate.
