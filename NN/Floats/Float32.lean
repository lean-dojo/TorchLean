/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.FP32.Core
public import NN.Floats.IEEEExec.Exec32
import Mathlib.Algebra.Order.Algebra

/-!
# Float32

Unified Float32 entrypoint (TorchLean).

This file keeps several common meanings of "float32" separate and explicit: the trusted runtime
`Float` implementation, a proof-oriented rounding model (`FP32`), and an executable bit-level model
(`IEEE32Exec`).

## What “32-bit precision” means here

Throughout TorchLean, *float32* refers to **IEEE-754 binary32**: a 32-bit floating-point format with

- 1 sign bit,
- 8 exponent bits,
- 23 fraction bits (24 bits of precision including the implicit leading 1 for normals).

This is a widely supported baseline dtype for ML workloads; other formats (bf16/fp16/tf32, etc.) can
be added on top of the same structure.

## The three meanings we support

- **Lean runtime `Float` / `Float32`** are fast and convenient, but their arithmetic is implemented
  by external/runtime code. That behavior is *not* kernel-reducible, so we treat it as
  an explicit trust boundary.

- **`FP32`** is our *proof-oriented* float32 semantics: a finite-only “rounding-on-ℝ” model
  (in the style of Flocq) where each primitive operation is specified as “compute in `ℝ`, then
  round to the float32 grid”. Concretely, it fixes binary32-style parameters (radix 2, exponent
  function for gradual underflow, round-to-nearest ties-to-even). It does **not** model NaN/Inf.

- **`IEEE32Exec`** is our *execution-oriented* float32 semantics: an executable, bit-level
  IEEE-754 binary32 kernel implemented in Lean (raw `UInt32` bits, with signed zeros, subnormals,
  NaNs/Infs, and IEEE rules for core arithmetic). (Transcendentals are not specified by IEEE-754;
  we provide deterministic executable definitions, but we do not claim they match any
  particular hardware/libm.) This gives a concrete meaning to “float32 execution” inside Lean,
  independent of a particular platform’s runtime/libm.

The intent is to let the rest of the codebase depend on a single *name* (`Float32`/`F32`) while
keeping the boundary easy to see and easy to swap:

- theorem statements and error bounds typically use `FP32`,
- runnable demos typically use `IEEE32Exec`,
- runtime `Float32` is treated as an explicitly trusted/assumed implementation detail.

This design is described in the TorchLean paper appendix ("Appendix C (Numerical Semantics)"):
`arXiv:2602.22631` (https://arxiv.org/abs/2602.22631).
-/

@[expose] public section


namespace TorchLean.Floats

/-! ## Backend selection -/

/--
Selects which float32 semantics TorchLean should use.

`.fp32` is the proof-oriented rounding-on-`ℝ` model.
`.ieee754Exec` is the executable, bit-level IEEE-754 binary32 model.
-/
inductive Float32Mode where
  /-- Finite float32 rounding model (`FP32`). -/
  | fp32
  /-- Executable IEEE754 binary32 kernel (`IEEE32Exec`): bit-level float32 with NaN/Inf. -/
  | ieee754Exec
  deriving DecidableEq, Repr

/--
Executable float32 backend (bit-level IEEE-754 binary32).

This is the scalar type you pick when you want runs inside Lean to have an explicit float32 meaning
(including NaN/Inf and signed-zero behavior), rather than depending on the platform runtime.
-/
abbrev IEEE32Exec : Type := TorchLean.Floats.IEEE754.IEEE32Exec

/--
TorchLean’s “float32” surface with selectable semantics.

Default is `.ieee754Exec` because it is the closest to real float32 execution you can *define*
inside Lean. For theorem statements and compositional error reasoning, prefer `.fp32`.
-/
abbrev Float32 (mode : Float32Mode := .ieee754Exec) : Type :=
  match mode with
  | .fp32 => FP32
  | .ieee754Exec => IEEE32Exec

/-- Short alias used in demos/docs. -/
abbrev F32 (mode : Float32Mode := .ieee754Exec) : Type := Float32 mode

/-! ## Lightweight logging (CLI/examples) -/

/-- Human-readable summary of the selected float32 semantics. -/
def float32ModeSummary : Float32Mode → String
  | .fp32 =>
      "FP32: proof semantics (round-on-ℝ), finite-only; no NaN/Inf"
  | .ieee754Exec =>
      "IEEE32Exec: executable IEEE-754 binary32 kernel (bit-level; includes NaN/Inf)"

/-- Print a one-line summary of the selected float32 semantics. -/
def logFloat32Mode (mode : Float32Mode) : IO Unit :=
  IO.println s!"[TorchLean] Float32 mode: {float32ModeSummary mode}"

end TorchLean.Floats
