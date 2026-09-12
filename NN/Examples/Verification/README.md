# Verification Examples

This directory contains small artifacts consumed by TorchLean's verification commands. Reusable
checkers and proof APIs live under `NN/Verification`; this directory is data and documentation, not
a second verification library.

## Start Here

List the commands available in the current checkout:

```bash
lake exe verify -- list
```

Start with a small model whose bounds Lean recomputes:

```bash
lake exe verify -- torchlean-ibp
```

Then run the routine local suite:

```bash
lake exe verify -- all
```

The example runner and verifier are intentionally separate:

```text
lake exe torchlean ...   runs models, training, data, and numerical demonstrations
lake exe verify -- ...   checks a stated property or certificate
```

Successful training is not a verification result.

## Example Groups

| Directory | Purpose |
| --- | --- |
| `TorchLean/` | Models originating in TorchLean and lowered into graph verification workflows. |
| `Robustness/` | Bundled digits weights, test data, and margin artifacts. |
| `LiRPA/` | Small exported bound artifacts for supported network fragments. |
| `AbCrown/` | Sanity checks for exported leaf reports; does not recompute network bounds. |
| `VNNComp/` | Compact VNN-COMP-style network and property inputs. |
| `PINN/` | Residual certificates and small containment datasets. |
| `ODE/` | Corridor and learned-function certificate inputs. |
| `Splines/` | Exact rational checks that polynomial pieces meet their stated endpoints. |

Representative commands:

```bash
lake exe verify -- torchlean-mlp-workflow
lake exe verify -- digits --eps=0.02 --max=360
lake exe verify -- abcrown-leaf
lake exe verify -- vnncomp-mnistfc
lake exe verify -- pinn-cert
lake exe verify -- spline-cert
```

Some commands accept a positional artifact path and others accept flags; `verify -- list` shows
the entry points. `vnncomp-mnistfc` needs separately prepared files under `_external/` and is not
an offline first example. The bundled `abcrown-leaf` and `margin-report` commands check report
consistency; neither independently proves the producer's network bounds.

## What Acceptance Means

External JSON, weights, bounds, and solver output are untrusted inputs. A command accepts an
artifact only after Lean parses its schema and recomputes the predicate implemented by that
checker. Acceptance does not automatically prove that:

- an external producer computed its candidate correctly;
- decimal floating-point bounds are exact real-number bounds;
- a report-consistency checker establishes model soundness;
- a runtime kernel implements the proved graph semantics.

The command output and checker documentation name the predicate actually checked. Theorem-level
IBP/CROWN soundness lives under `NN/MLTheory/CROWN/Proofs`; TorchLean-to-IR correctness support
lives under `NN/Verification/Builtin/Proved`.

## Artifact Policy

Keep only small artifacts that make a checker reproducible offline. Store large trained
checkpoints, benchmark suites, and generated dumps outside git and pass their paths explicitly.
Subdirectory READMEs document the format and producer for each artifact family. The reusable
verification API and its architecture are documented in `NN/Verification/README.md`.
