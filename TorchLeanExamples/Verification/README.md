# Verification Examples

This directory holds the `verify` command (`Main.lean`, with its command registry in `CLI.lean`),
the example checkers and workflows it runs, the small artifacts they read, and a CROWN margin
theorem for a small ReLU MLP. Reusable checkers and proof APIs live under `TorchLean/Verify`.

## Start Here

List the available commands:

```bash
scripts/lake.sh exe verify -- list
```

Start with a small model whose bounds Lean recomputes:

```bash
scripts/lake.sh exe verify -- torchlean-ibp
```

Then run the routine local suite:

```bash
scripts/lake.sh exe verify -- all
```

It runs `ibp-cert`, `camera-box3d-cert`, `pinn-cert`, `spline-cert` and `abcrown-leaf`, in that
order, each on its bundled artifact.

The example runner and verifier are intentionally separate:

```text
scripts/lake.sh exe torchlean ...   runs models, training, data, and numerical demonstrations
scripts/lake.sh exe verify -- ...   checks a stated property or certificate
```

Successful training is not a verification result.

## Example Groups

| Directory | What it holds |
| --- | --- |
| [`IBPCert/`](IBPCert/README.md) | An IBP certificate for a 3 → 4 → 2 ReLU MLP and its producer. |
| [`AbCROWN/`](AbCROWN/README.md) | Consistency checks of alpha-beta-CROWN leaf reports. |
| [`PINN/`](PINN/README.md) | PINN certificate and dataset checks, and a residual-bounding CLI. |
| [`ODE/`](ODE/README.md) | Corridor certificates for a scalar ODE `u' = f(t, u)`. |
| [`Splines/`](Splines/README.md) | Exact checks that polynomial pieces meet at their endpoints. |
| [`Geometry3D/`](Geometry3D/README.md) | Camera projection checks of 3D box certificates. |
| [`VNNComp/`](VNNComp/README.md) | VNN-COMP 2022 `mnist_fc` checks on files under `_external/`. |
| [`Lyapunov/`](Lyapunov/README.md) | `Examples.Lyapunov.Cert` and an all-in-Lean CEGIS training pipeline. |
| Top level | `Main.lean`, `CLI.lean`, `BoundArithmetic.lean` and four graph workflows. |

Three more Lean files sit at the top level. `Lyapunov.lean` and `PINN.lean` hold the shared cores
of the `Lyapunov/` and `PINN/` workflows, and `RobustMLP.lean` holds the margin theorem below.

Representative commands:

```bash
scripts/lake.sh exe verify -- ibp-cert
scripts/lake.sh exe verify -- abcrown-leaf
scripts/lake.sh exe verify -- pinn-cert
scripts/lake.sh exe verify -- spline-cert
scripts/lake.sh exe verify -- camera-box3d-cert
scripts/lake.sh exe verify -- ode --cert=TorchLeanExamples/Verification/ODE/sample_ode_cert.json
scripts/lake.sh exe verify -- twostage-torchlean-cegis-van
scripts/lake.sh exe verify -- vnncomp-mnistfc
scripts/lake.sh exe verify -- all
```

`verify -- list` prints every command with a one-line description and, for a command that takes
an optional artifact path, the bundled default. `ode`, `twostage-torchlean-cegis-van` and
`vnncomp-mnistfc` are not part of `all`. By default `vnncomp-mnistfc` reads its weights and suite
from `_external/vnncomp/mnist_fc/`, which the repository does not ship. `abcrown-leaf` checks that
a converted report is consistent and does not recompute the producer's network bounds.

## Graph Workflows

The workflows in `IBPWorkflow.lean`, `CROWNOpsWorkflow.lean`, `TransformerIBPWorkflow.lean` and
`MLPTrainVerifyWorkflow.lean` lower or build small TorchLean models, attach input regions, and run
native bound passes:

```bash
scripts/lake.sh exe verify -- torchlean-ibp
scripts/lake.sh exe verify -- torchlean-crown-ops
scripts/lake.sh exe verify -- torchlean-transformer-ibp
scripts/lake.sh exe verify -- torchlean-mlp-workflow
```

The MLP workflow trains a classifier and calls
`trained.verify center (radius := 0.10) (property := .topLabel 0)`. Its graph and bound state stay
internal. The default algorithm is `crown`, which is directed backward CROWN on rounded scalars
such as this workflow's. Use these commands when changing graph lowering, bound propagation, typed
graph execution, or the verification API.

## IBP Certificate

The example in [`IBPCert/`](IBPCert/README.md), `verify -- ibp-cert`, checks
`IBPCert/mlp_cert.json` against Lean's IBP pass over the 3 → 4 → 2 ReLU MLP of `IBPCert/MLP.lean`.
Its input box is `[0, 2] × [1, 3] × [2, 4]`, of radius 1 around `(1, 2, 3)`. The checker,
`TorchLean/Verify/Certificate/IBP.lean`, rounds each decimal endpoint once to binary64, lower
endpoints toward `-∞` and upper endpoints toward `+∞`, and accepts when every certified interval
contains the one Lean recomputes. The certified output intervals are about `[230, 554]` and
`[291, 699]`.

`IBPCert/export_cert.py` writes the certificate by reproducing Lean's own host-`Float` IBP pass
operation for operation, so the certificate carries no independent evidence. The check runs on
host `Float`, and no theorem covers it. It exercises the artifact parser and the containment test.

## CROWN margin theorem for a small ReLU MLP

`RobustMLP.lean` (namespace `Examples.RobustMLP`) proves that a `1 → 2 → 2` ReLU network keeps its
predicted class on an interval. The first layer has weights `[[1], [-1]]` and zero bias, then comes
ReLU, then the identity with bias `[1, 0]`, so the two scores are `1 + relu(x)` and `relu(-x)`.
The module has no `verify` command. Building it checks the proof:

```bash
scripts/lake.sh build TorchLeanExamples.Verification.RobustMLP
```

### The `robust` theorem

```lean
theorem robust (x : Tensor ℝ [1]) (hx : |x.getScalar 0| ≤ 1 / 4) :
    (3 : ℝ) / 4 ≤ (network.toCertificate.model.eval network.toCertificate.params x).getScalar 0 -
      (network.toCertificate.model.eval network.toCertificate.params x).getScalar 1
```

It covers every real input with `|x| ≤ 1/4`, without sampling. The first score exceeds the second
by `1 + x`, so the margin 3/4 is attained at `x = -1/4` and cannot be raised. Both ReLU inputs
cross zero inside the interval.

### Proof by a CROWN query

`query` asks for `score₁ - score₀ + 3/4 ≤ 0` on `[-1/4, 1/4]`. `Certificate.CROWNQuery.Query.check`,
in `TorchLean/Verify/Certificate/CROWNQuery.lean`, propagates affine bounds through the network in
exact rational arithmetic. The lower ReLU slopes are the zeros stored in `network`, and the upper
bound of a ReLU whose input crosses zero is the chord from `(l, 0)` to `(u, u)`. For this network
the propagated upper bound of `score₁ - score₀ + 3/4` is `-x/2 - 1/8`, which is at most 0 on the
interval. `decide +kernel` evaluates the check, `Query.check_sound` turns it into `Query.Safe` for
every real input in the box, and the proof unpacks that inequality into the two scores and closes
with `linarith`.

### Floating-point execution

The theorem is about real arithmetic. `Network.realTensor_forward` shows that exact rational
execution agrees with the real model, but nothing here bounds `Float` execution, which would need
rounding error bounds. The two `#eval` lines in the module docstring give the scores 1.25 and 0
for the `Float` input 0.25, and 1 and 1/4 for the rational input -1/4.

## What Acceptance Means

External JSON, weights, bounds, and solver output are untrusted inputs. A command accepts an
artifact only after Lean parses its schema and recomputes the checker's predicate. Acceptance
alone does not show that:

- an external producer computed its candidate correctly
- decimal floating-point bounds are exact real-number bounds
- a report-consistency checker establishes model soundness
- a runtime kernel implements the proved graph semantics

The command output and the checker's documentation name the checked predicate. The IBP and CROWN
soundness theorems are under `TorchLean/Verify/CROWN` and `TorchLean/Verify/CROWN/Op`, and
the lowering of TorchLean programs to the IR is in `TorchLean/NN/IR/Lowering.lean`.

## Artifact Policy

Keep only small artifacts that make a checker reproducible offline. Large trained checkpoints,
benchmark suites, and generated dumps stay outside git, and their paths go on the command line.
The subdirectory READMEs give the format and producer of each artifact family, and
`TorchLean/Verify/README.md` covers the reusable verification API and its architecture.
