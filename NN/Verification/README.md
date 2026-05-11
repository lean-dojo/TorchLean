# `NN/Verification` (verification library support)

This directory is the library layer for TorchLean verification workflows: it contains reusable
checkers, parsers, certificate kernels, and bridges from models to verifier IR that examples and
CLIs import.

Key themes:

- **Graph verification**: most checkers operate on the op tagged IR (`NN.IR.Graph`) together
  with a `CROWN`/LiRPA `ParamStore`.
- **Runtime numerics**: executable scalars such as `Float` and `IEEE32Exec` can be swapped into
  bound propagation and evaluation to study floating point effects.
- **Explicit trust boundaries**: when using external toolchains (e.g. PyTorch exported weights or
  certificates), we keep the parsing, validation, and recomputation inside Lean explicit.
- **Proof layer**: `NN.Entrypoint.Verification` includes theorem level CROWN/LiRPA
  soundness statements alongside the reusable verifier APIs. JSON checkers are executable
  recomputation checks; theorem credit requires discharging the corresponding Lean hypotheses.
- **Data workflows**: reusable dataset and weight loading should live under `NN.Verification`
  rather than directly inside examples. For example, sklearn digits certified accuracy lives in
  `NN.Verification.Robustness.Digits`.

## Subfolders

- `TorchLean/`: compilation and spec evaluation glue for reusing the same TorchLean model
  definition/program for execution and verification.
- `ODE/`: ODE corridor verifier for subsolution and supersolution certificates, inspired by
  arXiv:2601.19818.
- `PINN/`: reusable PINN graph builders, PDE DSL/parser, residual bound helpers, certificate
  checker, CLI implementation, and dataset backed checker.
- `Util/`: small shared utilities (JSON parsing helpers, numeric comparisons).
- `Cert/`: certificate format checkers, such as "IBP output bounds agree with Lean recomputation".
- `Robustness/`: reusable robustness workflows, including dataset backed certified accuracy.

## Public Imports

- `NN.Entrypoint.Verification`: reusable verification APIs and theorem level CROWN/LiRPA soundness
  entry points.
- `NN.Verification.Cert`: executable certificate checker surface (JSON parsers + recomputation checks).
- `NN.Verification.ODE`: ODE corridor verifier surface (AST + parser + checker).
- `NN.Verification.Splines`: spline and piecewise polynomial certificate checker surface.
- `NN.Verification.Robustness`: reusable robustness workflows and loaders.
- `NN.Verification.PINN`: reusable PINN verification support.
- `NN.Verification.CLI`: runnable CLI registry used by `lake exe verify`.

## JSON Artifacts And Tolerances

External JSON artifacts are treated as untrusted. Checkers parse them, validate shapes, and compare
them against Lean recomputation. When decimal serialized floats are involved, the tolerance is an
explicit artifact format tolerance; it is not a proof that the external producer is sound.

## References (informal)

- IBP (interval bound propagation): Gowal et al., 2018 ("On the Effectiveness of Interval Bound
  Propagation for Training Verifiably Robust Models").
- CROWN / DeepPoly style linear relaxations: Zhang et al., 2018 ("Efficient Neural Network
  Verification with CROWN").
- LiRPA unification viewpoint: Xu et al., 2020 ("Automatic Perturbation Analysis for Scalable
  Certified Robustness and Beyond").
- ODE learn and verify corridor enclosures: Tanaka and Yatabe, 2026 (arXiv:2601.19818).
