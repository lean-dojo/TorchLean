---
title: Scientific ML
---

Scientific ML examples are where TorchLean has to behave like both an ML library and a mathematics
library. The model should run on real data, but the run should also leave behind objects with clear
meaning: a grid, a PDE, a parameter file, a residual expression, a certificate, or a prediction
artifact that Lean can reload.

The first example is a one-dimensional Fourier neural operator for Burgers'
equation. It is a real training run: prepare simulation data, train an operator model, and export
loss logs plus prediction curves. The second is PINN-style checking. There the artifacts are kept
small enough for Lean to inspect directly: PDE residual expressions, dataset samples, compact
weights, intervals, and certificates. The two paths are different on purpose. One asks whether a
model learns the scientific map; the other asks which precise residual, dataset, or certificate claim
can be checked.

## The Problem

Burgers' equation is a standard nonlinear PDE benchmark. In viscous form:

$$
  u_t + u\,u_x = \nu u_{xx}.
$$

The solution is a scalar field $u(x,t)$. The nonlinear term $u\,u_x$ transports features through
the domain, while the viscosity term $\nu u_{xx}$ smooths them. That combination makes Burgers a
good compact benchmark for scientific ML: the model has to learn a time evolution pattern rather
than a static regression label.

"Benchmark" here means the standard test problem, not a timing measurement.

There are two common neural ways to approach this equation.

An operator-learning model sees examples of whole functions. Given an initial condition $u_0(x)$,
it learns to predict a later field $u(x,T)$. The FNO path below follows that setup.

A physics-informed model represents a candidate solution $u_\theta(x,t)$ and asks whether that
candidate satisfies the PDE residual at sampled points. Use the PINN-style checking path when the
artifact to trust includes a residual, a bound, or a certificate
attached to the PDE alongside the prediction curve.

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
- repeating the block gives a compact model for the map from the initial field to the final field.

The TorchLean example keeps the model compact enough to inspect:

```text
grid   = 32
width  = 8
modes  = 8
blocks = 1
```

The run is enough to exercise the actual operator-learning path while keeping the tensors, modes,
and artifacts readable. CPU execution uses the dense multidimensional real-split DFT. CUDA uses a
fused real-FFT FNO primitive backed by cuFFT.

## What TorchLean Owns

The split is deliberate. Python handles the parts where the Python ecosystem is the right tool:
downloading a public `.mat` file, preparing external datasets, or plotting a prediction CSV.
TorchLean owns the pieces that should be typed, inspectable, or connected to verification:

<div class="workflow-list">
  <a href="{{ '/blueprint/Building-Models/From-Files-To-Typed-Minibatches/' | relative_url }}">
    <span>01</span>
    <strong>Typed data loading</strong>
    <em>The <code>.npy</code> arrays become fixed-shape supervised samples: one grid input, one grid target.</em>
  </a>
  <a href="{{ '/docs/NN/API/Models/FNO.html' | relative_url }}">
    <span>02</span>
    <strong>Model shape contract</strong>
    <em>The FNO config fixes the grid, channel width, Fourier modes, block count, and parameter shapes.</em>
  </a>
  <a href="{{ '/blueprint/Floating-Point-and-Native-Boundaries/From-A-Tensor-Operation-To-A-GPU-Kernel/' | relative_url }}">
    <span>03</span>
    <strong>Runtime boundary</strong>
    <em>The CUDA path is fast, but its role is named: a fused real-FFT kernel implements the TorchLean FNO step.</em>
  </a>
  <a href="{{ '/examples/verification/' | relative_url }}">
    <span>04</span>
    <strong>Lean checks</strong>
    <em>PINN residual bounds, finite certificates, and dataset enclosure checks can be reloaded and checked in Lean.</em>
  </a>
</div>

That boundary matters. A plot can show that a prediction looks plausible. A Lean side artifact can
say more precisely which grid was used, which expression was parsed, which interval was checked,
and which command accepted or rejected the claim.

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
lake -R -K cuda=true build
```

Run the trainer:

```bash
lake -R -K cuda=true exe torchlean fno1d_burgers \
  --device cuda \
  --steps 700 \
  --lr 0.003 \
  --plot-csv data/real/fno/predictions.csv \
  --log data/real/fno/trainlog.json
```

For a short run that exercises the same path:

```bash
lake -R -K cuda=true exe torchlean fno1d_burgers \
  --device cuda --steps 50 --log false
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
[`NN.Examples.Models.Operators.Fno1dBurgers`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Operators/Fno1dBurgers.lean).

## What To Look For

A good run should show held-out MSE moving down together with training MSE. The compact one-block
FNO keeps the pipeline readable while still using the real operator-learning path:

- the dataset rows and grid resolution are explicit;
- the model shape is fixed by the Lean configuration;
- the CUDA path is named separately from the mathematical model;
- the run emits scalar logs and field-level prediction artifacts;
- those artifacts can be inspected outside the trainer.

For larger scientific models, the neural network should not be a black box floating beside the proof.
The data shape, runtime path, exported artifacts, and verification claim should line up.

Inspect both kinds of output. The loss log tells you whether the training run moved in the right
direction. The prediction CSV tells you what the learned operator does on a held-out trajectory. The
Lean verification commands tell you whether a much smaller, explicit artifact satisfies the residual,
dataset, or certificate condition it claims to satisfy.

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

The TorchLean PINN files are written for verification. Python can produce compact weights, PDE
descriptions, or dataset samples; Lean reloads those artifacts and checks the residual or dataset
conditions through `NN.Verification.PINN`.

Run the small checked assets:

```bash
python3 scripts/verification/pinn/export_pinn_cert.py
lake exe verify -- pinn-cert
lake exe verify -- pinn-cli -- "u_t + u*u_x - 0.01*u_xx" 0.0 0.5 0.01
lake exe verify -- pinn-dataset-check
```

For a Burgers-style training/export path on the Python side:

```bash
python3 scripts/verification/pinn/train_pinn_1d.py \
  --steps 500 \
  --nu 0.01 \
  --pde-expr "u_t + u * u_x - nu * u_xx"
```

The three commands check different objects. `pinn-cert` recomputes the compact certificate's residual
intervals and compares them with the exported values. `pinn-cli` bounds a residual expression over a
small box; the PDE parser accepts both compact names such as `uxx` and subscript-style names such as
`u_xx` and `u_t`, with `t` treated as the second input axis. `pinn-dataset-check` is separate: it
checks whether initial, boundary, and supervised data values are contained in the network output
intervals. By default the dataset command is diagnostic and prints contained/missed counts. Add
`--strict` when misses should make the command fail.

Read the outputs as three different forms of evidence:

- `pinn-cert` is a compact certificate replay.
- `pinn-cli` is a local residual-expression check over a stated box.
- `pinn-dataset-check` is a dataset containment diagnostic. It shows which samples are already
  covered by the exported intervals and which samples need tighter bounds or a different artifact.

Together, these commands split the scientific run into claims Lean can name: the residual
expression being bounded, the box or dataset being checked, and the exported certificate values being
replayed. Larger scientific models should follow the same pattern: ambitious training runs can feed
precise, checkable artifacts instead of relying on a plot or checkpoint alone.

## Related Sources

- [`NN.Examples.Models.Operators.Fno1dBurgers`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Operators/Fno1dBurgers.lean)
- [`NN.Runtime.Autograd.Engine.Cuda.Fno1dRfftFused`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/Engine/Cuda/Fno1dRfftFused.lean)
- [`NN.Verification.PINN`](https://github.com/lean-dojo/TorchLean/tree/main/NN/Verification/PINN)
- [`NN.Verification.PINN.DatasetCheck`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/PINN/DatasetCheck.lean)
- [`NN.Examples.Verification.PINN assets`](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/Verification/PINN)
