---
title: Scientific ML
---

Let's try two ways to train a neural network for a differential equation. First, we'll train
a Fourier neural operator (FNO) on simulated solutions of Burgers' equation. Then we'll train a
physics-informed neural network (PINN) using an equation and its boundary conditions as the loss.

We'll run both examples in TorchLean. After training, we'll look at separate Lean checkers for
exported PDE residuals and datasets, and work through what their results tell us.

## The Problem

We'll start with Burgers' equation, a standard nonlinear PDE benchmark. In viscous form:

$$
  u_t + u\,u_x = \nu u_{xx}.
$$

The solution is a scalar field $u(x,t)$. The nonlinear term $u\,u_x$ transports features through
the domain, while the viscosity term $\nu u_{xx}$ smooths them. That combination makes Burgers a
good compact benchmark for scientific ML: the model has to learn a time evolution pattern rather
than a static regression label.

There are two common neural ways to approach this equation.

An operator-learning model sees examples of whole functions. Given an initial condition $u_0(x)$,
it learns to predict a later field $u(x,T)$. The FNO path below follows that setup.

A physics-informed model represents a candidate solution $u_\theta(x,t)$ and asks whether that
candidate satisfies the PDE residual at sampled points. Use the PINN-style checking path when the
artifact to trust includes a residual, a bound, or a certificate
attached to the PDE alongside the prediction curve.

For our FNO, we'll pair each initial condition with the solution at a later time:

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
script downloads or reads the `.mat` file, chooses a grid resolution, selects train/test rows,
and writes `.npy` tensors. TorchLean then loads these arrays for training.

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

In this block:

- the pointwise term $Wv_k(x)$ handles local channel mixing;
- the spectral term keeps a small number of Fourier modes and learns how those modes should evolve;
- the activation $\sigma$ makes the block nonlinear;
- repeating the block gives a compact model for the map from the initial field to the final field.

We'll keep the model small enough to inspect:

```text
grid   = 32
width  = 8
modes  = 8
blocks = 1
```

CPU execution uses the dense multidimensional real-split DFT. CUDA
uses Lean composition of FFT, frequency mixing, and inverse FFT through LibTorch numerical
primitives. TorchLean also composes the local reverse calculation and owns the tape's saved buffers.

## Data preparation and training

Python downloads and converts the data and plots the prediction CSV. In Lean, we load the
tensors, construct the model, and train it. The relevant APIs are:

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
    <em>Lean composes the spectral layer and its reverse calculation; LibTorch supplies numerical primitives, and TorchLean owns the tape and saved buffers.</em>
  </a>
  <a href="{{ '/examples/verification/' | relative_url }}">
    <span>04</span>
    <strong>Lean checks</strong>
    <em>PINN residual bounds, finite certificates, and dataset enclosure checks can be reloaded and checked in Lean.</em>
  </a>
</div>

## Run The FNO Example

First, let's prepare a small training and test split:

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
scripts/lake.sh -Kcuda=true build
```

With the data ready, we can run the trainer:

```bash
scripts/lake.sh -Kcuda=true exe torchlean fno1d_burgers \
  --device cuda \
  --steps 700 \
  --lr 0.003 \
  --plot-csv data/real/fno/predictions.csv \
  --log data/real/fno/trainlog.json
```

For a short run that exercises the same path:

```bash
scripts/lake.sh -Kcuda=true exe torchlean fno1d_burgers \
  --device cuda --steps 50 --log false
```

The command prints the device path, grid/model constants, train/test rows, and loss reports. With
the default artifact paths, it also writes:

- `trainlog.json`: train/test MSE history with run metadata;
- `predictions.csv`: one held-out input, target final field, and predicted final field.

Let's plot the saved prediction beside its target:

```bash
python3 NN/Examples/Data/plot_fno1d_burgers.py \
  --csv data/real/fno/predictions.csv
```

Source entrypoint:
[`NN.Examples.Models.Operators.Fno1dBurgers`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Operators/Fno1dBurgers.lean).

## What To Look For

Compare the training and held-out MSE in the loss log. A falling training loss alone does not tell
you how well the operator predicts a new initial condition. Plot the prediction CSV to compare its
predicted final field with the held-out target.

These measurements evaluate the trained FNO. The PINN checks below concern separate residual
expressions and datasets; they do not certify this FNO training run.

## Train a PINN in Lean

Now let's train from an equation rather than simulated solution pairs. We'll solve $u''(x)=-2$
on $[-1,1]$ with $u(-1)=u(1)=0$, using a tanh MLP. We train on five interior collocation points
and the two boundary points, without using samples of the exact solution $1-x^2$ as targets.

From the repository root:

```bash
scripts/lake.sh -Kcuda=false exe torchlean pinn 1
```

That is a one-step runtime check. Omit `1` for the example's 1,500-step training run. The count is
a positional argument, not a `--steps` flag. The command prints the initial and final residual
objective, intermediate losses every 50 steps, and predictions beside the exact solution.

`autograd.model.derivative` computes coordinate derivatives; passing the same direction twice
computes the second derivative used by the residual. `autograd.model.derivativeVjp` propagates
the residual's cotangent through that derivative to the model parameters. The resulting gradient
updates the same typed model state with `nn.sgdStep`.

These sampled residuals and prediction comparisons are diagnostics, not a uniform PDE certificate.
The complete example is
[`NN.Examples.Models.Operators.Pinn`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Operators/Pinn.lean).

## PINN-Style Checks

For the remaining checks, we'll return to Burgers' equation. Given a candidate solution
$u_\theta(x,t)$, we can form its PDE residual:

$$
  r_\theta(x,t)
    =
    \partial_t u_\theta(x,t)
    +
    u_\theta(x,t)\,\partial_x u_\theta(x,t)
    -
    \nu\,\partial_{xx}u_\theta(x,t).
$$

The separate `NN.Verification.PINN` workflows check exported artifacts. Python can produce compact
weights, PDE descriptions, or dataset samples; Lean reloads those artifacts and checks the residual
or dataset conditions. This is independent of the native training example above.

The bundled `pinn_cert.json` in `NN/Examples/Verification/PINN` demonstrates the simpler residual
$u_{xx}$ with fixed seed weights, not a trained Burgers solution. `pinn-cert` recomputes bounds for
those built-in weights; it does not load the JSON's `model.weights` or validate its `domain`
metadata. We can run it without regenerating the certificate or training a model.
Its derivative-residual interval is roughly $[-25.65,25.65]$: matching those bounds checks replay,
not that the residual is close to zero.

`sample_dataset_1d.json` and `sample_dataset_2d.json` are small illustrative datasets for the
space-time and two-spatial-coordinate paths. Their labels are not certified PDE solutions.

Let's start with the small bundled examples:

```bash
python3 scripts/verification/pinn/export_pinn_cert.py
scripts/lake.sh exe verify -- pinn-cert
scripts/lake.sh exe verify -- pinn-cli -- "u_t + u*u_x - 0.01*u_xx" 0.0 0.5 0.01
scripts/lake.sh exe verify -- pinn-dataset-check
```

For a Burgers-style training/export path on the Python side:

```bash
python3 scripts/verification/pinn/train_pinn.py evolution \
  --steps 500 \
  --nu 0.01 \
  --pde-expr "u_t + u * u_x - nu * u_xx"
```

Use `train_pinn.py stationary` for a problem with two spatial coordinates. Both trainers export
checkpoints and TorchLean weight JSON. To convert an existing checkpoint or initialize a fresh
model, use `python3 scripts/verification/pinn/train_pinn.py export --help`.
`import_burgers_shock_mat.py` in the same directory converts the external Burgers `.mat` dataset
to JSON.

When importing weights, set `meta.activation` to the activation used during training. The loader
accepts `tanh`, `relu`, and `sin`/`sine`/`siren`. An absent metadata object or activation field
selects `tanh`; malformed metadata and unsupported names reject the checkpoint. The loader checks
the name, but cannot establish that it matches the model used to produce the weights.

Certificates require a `pinn.pde` residual expression, such as `"uxx"` for $u_{xx}=0$.
Missing, null, or non-string expressions are rejected. Write the residual, not an equation.

The three commands check different objects. `pinn-cert` recomputes the compact certificate's residual
intervals and compares them with the exported values. `pinn-cli` bounds a residual expression over a
small box; the PDE parser accepts both compact names such as `uxx` and subscript-style names such as
`u_xx` and `u_t`, with `t` treated as the second input axis. `pinn-dataset-check` is separate: it
checks whether initial, boundary, and supervised data values are contained in the network output
intervals. By default the dataset command is diagnostic and prints contained/missed counts. Add
`--strict` when misses should make the command fail.

## Related Sources

- [`NN.Examples.Models.Operators.Fno1dBurgers`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Operators/Fno1dBurgers.lean)
- [`NN.API.Models.FNO`](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Models/FNO.lean)
- [`NN.Verification.PINN`](https://github.com/lean-dojo/TorchLean/tree/main/NN/Verification/PINN)
- [`NN.Verification.PINN.DatasetCheck`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/PINN/DatasetCheck.lean)
- [`NN.Examples.Verification.PINN assets`](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/Verification/PINN)
