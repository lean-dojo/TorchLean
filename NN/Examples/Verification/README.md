# Verification Fixtures and Workflows

This directory contains the small verification fixtures and wrapper modules used by TorchLean's
unified verification CLI.

Reusable verification code lives under `NN/Verification/*`.
These fixtures are the small assets and wrapper modules that keep `lake exe verify`
reproducible without pulling in large benchmark dumps.

## What To Run

- `lake exe verify -- all`
  Runs the fast certificate checkers that are safe for routine regression checks.

- `lake exe verify -- digits --eps=0.02 --max=360`
  Loads the bundled sklearn digits weights and test set, compiles the linear classifier through the
  TorchLean verifier bridge, and reports IBP/CROWN certified accuracy.

- `lake exe verify -- margin-cert`
  Checks the exported digits logit margin certificate. This recomputes the margin predicate from
  the JSON bounds and checks the summary fields.

- `lake exe verify -- torchlean-robustness`
  Builds a compact TorchLean classifier, compiles it to verifier IR, and checks the margin with
  IBP, forward affine CROWN, and backward objective CROWN.

- `lake exe verify -- torchlean-crown-ops`
  Exercises nonlinear verifier ops such as softmax and MSE loss on compact TorchLean graphs.

- `lake exe verify -- spline-cert`
  Checks an exact rational piecewise polynomial certificate. With `--regen`, Julia is used only as
  an untrusted producer and Lean checks the regenerated JSON payload.

## Workflow Tiers

- Native TorchLean verification: `TorchLean/*` and `Robustness/TorchLeanRobustness.lean` build
  models in TorchLean, compile them to verifier IR, and run bound propagation directly. These do
  not depend on an external verifier exporter.

- Exporter-backed verification: `LiRPA/*` and `VNNComp/*` include example wrappers. `AbCrown/*`,
  `ODE/*`, and `PINN/*` hold bundled artifacts consumed by reusable CLI/checker code under
  `NN/Verification`. Python or external tools may produce candidate JSON artifacts, but Lean still
  parses and checks the artifact before accepting it.

- Certificate checkers: `LiRPA/*`, `AbCrown/*`, `Robustness/VerifyMarginCert.lean`, and
  `Splines/PiecewiseLinearVerify.lean` parse external artifacts and recompute the relevant
  certificate condition inside Lean.

- Data-backed robustness: `lake exe verify -- digits` runs `NN.Verification.Robustness.Digits`,
  which loads the exported sklearn digits weights and test data stored in `Robustness/`.

- ODE/PINN workflows: `ODE/*` and `PINN/*` hold compact certificate/dataset assets. The checker
  implementations live under `NN.Verification.ODE` and `NN.Verification.PINN`, with Python scripts
  under `scripts/verification/` used as untrusted producers for weights or candidate certificates.

- Proof map: theorem-level CROWN/LiRPA soundness pointers live under
  `NN.Verification.ProofBackedCertificates` and `NN.Entrypoint.Verification`, not in this fixtures
  directory.

Reusable Lean code for ODE/PINN and certificate checking belongs under `NN/Verification`.
The `ODE/`, `PINN/`, `AbCrown/`, and `LiRPA/` folders here should contain only small fixtures,
notes, or thin runnable wrappers. Producers generally belong under `scripts/verification/`.

## Trust Boundaries

External tools may produce JSON, weights, alpha slopes, or candidate bounds. Those artifacts are
not trusted. Treat a workflow as checked only when Lean parses the artifact, checks shapes, and
recomputes the relevant predicate or bound.

Some JSON checkers compare decimal serialized floating point values with an explicit
tolerance. That is an artifact format check, not a theorem that the external producer is sound. For
the theorem path, use `NN.Entrypoint.Verification`, which states checker-style soundness
over the Lean graph semantics once the local certificate hypotheses are discharged.

## Compact Constants Versus Real Data

Small hand written tensors are acceptable in TorchLean native operator workflows because their job
is to exercise a verifier path quickly and reproducibly. Workflows that make data claims should
load weights and datasets from documented assets. Digits fixtures are bundled; large VNN-COMP
exports are kept outside git and passed to the checker explicitly.

## Artifact Parsers And Assets

Reusable parsing belongs in `NN.Verification`, not in individual example files. In particular,
`NN.Verification.Util.Json` provides the shared artifact boundary: read a JSON file, require a
schema `format`, and extract typed fields such as objects, arrays, natural numbers, booleans, and
float arrays with contextual errors.

Small JSON files are kept only when they make an example reproducible with one command. Larger
benchmark assets should be generated or downloaded by the documented scripts and treated like data
artifacts, not hand maintained source code.

Use the asset catalog to see or run the available regeneration commands:

```bash
python3 scripts/verification/regenerate_assets.py --list
python3 scripts/verification/regenerate_assets.py --group digits --run
python3 scripts/verification/regenerate_assets.py --group lirpa --run
```

Current asset policy:

| Asset class | Keep in git? | Regeneration path |
| --- | --- | --- |
| Small checker fixtures (`LiRPA/*.json`, `AbCrown/sample_*.json`, `Splines/*.json`) | Yes, if they keep CLI checks offline and small. | `regenerate_assets.py --group lirpa`, `lake exe verify -- spline-cert --regen`, or the local exporter. |
| Digits robustness fixtures | Yes, while they keep the certified accuracy example reproducible offline. | `regenerate_assets.py --group digits --run`. |
| PINN compact certs/datasets | Keep only compact fixtures; store trained checkpoints outside git. | `regenerate_assets.py --group pinn-small --run` and `PINN/train_*.py` for local runs. |
| PINN trained checkpoints/weight dumps | No. They are generated local outputs. | `regenerate_assets.py --group pinn-train --run`; outputs land in ignored paths. |
| ODE compact certificates/weights | Yes, if they remain curated and small. | `regenerate_assets.py --group ode --run` checks the default curated fixture. |
| VNN-COMP snapshots | No. Keep model/suite exports outside git. | Store under `_external/vnncomp/...` or pass explicit `--weights=... --suite=...` paths. |
| Two stage controller/Lyapunov weights | No. Treat as local experiment output. | `regenerate_assets.py --group two-stage --run`, which writes to `_external/` by default. |
