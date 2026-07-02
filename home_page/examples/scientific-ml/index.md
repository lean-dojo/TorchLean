---
title: Scientific ML
usemathjax: true
---

This example is about a basic scientific-ML question: can we train a neural model on PDE simulation
data, keep the data and model shapes explicit, export the run artifacts, and then connect the
resulting objects to checks that Lean can inspect?

The runnable path here is a one-dimensional Fourier neural operator for Burgers' equation. The
verification-facing path is PINN-style checking: exported PDE residuals, dataset samples, weights, or
certificates are small enough for Lean to reload and check. They are not the same model command, but
they are part of the same scientific workflow: train or import a neural surrogate, name the
mathematical contract, and check the artifacts that support the claim.

## The Problem

Burgers' equation is a standard nonlinear PDE benchmark. In viscous form:

$$
  u_t + u\,u_x = \nu u_{xx}.
$$

The solution is a scalar field $u(x,t)$. The nonlinear term $u\,u_x$ moves features around, while
the viscosity term $\nu u_{xx}$ smooths them. This makes the dataset a useful small test for
scientific ML: the model has to learn a whole time-evolution pattern, not just a static regression
label.

For the FNO run, TorchLean uses the common operator-learning version of the task. Each training
example is a pair:

$$
  \text{input } a = u_0(x),
  \qquad
  \text{target } u = u(x,T).
$$

So the learned object is a map between functions:

$$
  \mathcal{G}_\theta : u_0(x) \mapsto u(x,T).
$$

In the public `burgers_data_R10.mat` file used by many FNO tutorials, the field `a` stores the
initial conditions and the field `u` stores the final solution trajectories. The TorchLean data
script only does the ecosystem work: download or read the `.mat` file, choose a grid resolution,
pick train/test rows, and write `.npy` tensors. After that, the model run is native TorchLean.

## The Architecture

An FNO is built for operator learning. Instead of only applying local dense layers pointwise, it also
mixes information globally through Fourier modes. A simplified FNO block looks like:

$$
  v_{k+1}(x)
    =
    \sigma\!\left(
      W v_k(x)
      +
      \mathcal{F}^{-1}
        \left(R_\theta \cdot \mathcal{F}(v_k)\right)(x)
    \right).
$$

The two pieces have different jobs:

- the pointwise term $Wv_k(x)$ handles local channel mixing;
- the spectral term keeps a small number of Fourier modes and learns how those modes should evolve;
- the activation $\sigma$ makes the block nonlinear;
- stacking blocks gives a compact model for the map from the initial field to the final field.

The TorchLean example intentionally keeps the model small:

```text
grid   = 32
width  = 8
modes  = 8
blocks = 1
```

That is enough to exercise the actual operator-learning path without turning the example into a
large benchmark. On CUDA, the example uses a fused real-FFT FNO primitive backed by cuFFT. On CPU, it
falls back to a dense DFT reference path, which is slower but easier to inspect.

## What TorchLean Owns

The split is deliberate. Python is used for the parts where the Python ecosystem is the right tool:
download a public `.mat` file and plot a prediction CSV. TorchLean owns the pieces that should be
typed, inspectable, or connected to verification:

<div class="workflow-list">
  <a href="{{ '/docs/NN/Examples/Data/Loaders/Npy.html' | relative_url }}">
    <span>01</span>
    <strong>Typed data loading</strong>
    <em>The `.npy` arrays become fixed-shape supervised samples: one grid input, one grid target.</em>
  </a>
  <a href="{{ '/docs/NN/API/Models/Fno1d.html' | relative_url }}">
    <span>02</span>
    <strong>Model shape contract</strong>
    <em>The FNO config fixes the grid, channel width, Fourier modes, block count, and parameter shapes.</em>
  </a>
  <a href="{{ '/blueprint/Floating-Point-and-Native-Boundaries/GPU-and-CUDA-Boundaries/' | relative_url }}">
    <span>03</span>
    <strong>Runtime boundary</strong>
    <em>The CUDA path is fast, but its role is named: a fused real-FFT kernel implements the TorchLean FNO step.</em>
  </a>
  <a href="{{ '/docs/NN/Verification/PINN.html' | relative_url }}">
    <span>04</span>
    <strong>Lean-facing checks</strong>
    <em>PINN artifacts, dataset checks, residual checks, and certificates can be reloaded and checked in Lean.</em>
  </a>
</div>

## Run The FNO Example

Prepare a small Burgers split:

```bash
python3 NN/Examples/Data/prepare_fno1d_burgers.py \
  --download --grid 32 --ntrain 128 --ntest 32
```

This writes:

```text
data/real/fno/burgers_train_X.npy
data/real/fno/burgers_train_y.npy
data/real/fno/burgers_test_X.npy
data/real/fno/burgers_test_y.npy
data/real/fno/burgers_meta.json
```

Build TorchLean with CUDA support:

```bash
lake build -R -K cuda=true
```

Run the trainer:

```bash
lake exe -K cuda=true torchlean fno1d_burgers \
  --cuda --fast-kernels \
  --steps 700 \
  --lr 0.003 \
  --plot-csv data/real/fno/predictions.csv \
  --log data/real/fno/trainlog.json
```

For a quick smoke test:

```bash
lake exe -K cuda=true torchlean fno1d_burgers \
  --cuda --fast-kernels --steps 50 --log false
```

The command prints the device path, grid/model constants, train/test rows, and loss reports. With
the default artifact paths, it also writes:

- `trainlog.json`: train/test MSE history with run metadata;
- `predictions.csv`: one held-out input, target final field, and predicted final field.

Plot the prediction artifact:

```bash
python3 NN/Examples/Data/plot_fno1d_burgers.py \
  --csv data/real/fno/predictions.csv
```

Source entrypoint:
[`NN.Examples.Models.Operators.Fno1dBurgers`]({{ '/docs/NN/Examples/Models/Operators/Fno1dBurgers.html' | relative_url }}).

## What To Look For

A good run should show held-out MSE moving down together with training MSE. The point is not that a
tiny one-block FNO is the best possible Burgers solver. The point is that the whole scientific-ML
pipeline is visible:

- the dataset rows and grid resolution are explicit;
- the model shape is fixed by the Lean configuration;
- the CUDA path is named separately from the mathematical model;
- the run emits scalar logs and field-level prediction artifacts;
- those artifacts can be inspected outside the trainer.

That is the same discipline TorchLean wants for larger scientific models: the neural network should
not be a black box floating beside the proof. The data shape, runtime path, exported artifacts, and
verification claim should line up.

## PINN-Style Checks

A PINN takes a different route. Instead of learning only the operator $u_0 \mapsto u(T)$, a
physics-informed model represents a candidate solution $u_\theta(x,t)$ and checks the PDE residual:

$$
  r_\theta(x,t)
    =
    \partial_t u_\theta(x,t)
    +
    u_\theta(x,t)\,\partial_x u_\theta(x,t)
    -
    \nu\,\partial_{xx}u_\theta(x,t).
$$

The TorchLean PINN files are verification-facing. Python can produce compact weights, PDE
descriptions, or dataset samples; Lean reloads those artifacts and checks the residual or dataset
conditions through `NN.Verification.PINN`.

Run the small checked assets:

```bash
python3 scripts/verification/regenerate_assets.py --group pinn-small --run
lake exe verify -- pinn-cert
lake exe verify -- pinn-dataset-check
```

For a Burgers-style training/export path on the Python side:

```bash
python3 scripts/verification/pinn/train_pinn_1d.py \
  --steps 500 \
  --nu 0.01 \
  --pde-expr "u_t + u * u_x - nu * u_xx"
```

The dataset checker reports initial, boundary, and data terms. By default it is diagnostic: it prints
contained and missed counts. Add `--strict` when misses should make the command fail.

## Related Sources

- [`NN.Examples.Models.Operators.Fno1dBurgers`]({{ '/docs/NN/Examples/Models/Operators/Fno1dBurgers.html' | relative_url }})
- [`NN.API.Models.Fno1d`]({{ '/docs/NN/API/Models/Fno1d.html' | relative_url }})
- [`NN.Verification.PINN`]({{ '/docs/NN/Verification/PINN.html' | relative_url }})
- [`NN.Verification.PINN.DatasetCheck`]({{ '/docs/NN/Verification/PINN/DatasetCheck.html' | relative_url }})
- [`NN.Examples.Verification.PINN assets`](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/Verification/PINN)
