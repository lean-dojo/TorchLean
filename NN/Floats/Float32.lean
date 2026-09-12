/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import Mathlib.Algebra.Order.Algebra
public import Mathlib.Analysis.SpecialFunctions.Log.Basic
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
public import NN.Floats.IEEEExec.Exec32.Core

/-!
# Float32

Unified Float32 entrypoint (TorchLean).

This file keeps several common meanings of "float32" separate: TorchLean's proof-oriented
rounded-real model (`FP32`), its executable bit-level model (`IEEE32Exec`), and Lean's builtin
`Float32` with its logical `Float32.Model`.

## What “32-bit precision” means here

Throughout TorchLean, *float32* refers to **IEEE-754 binary32**: a 32-bit floating-point format with

- 1 sign bit,
- 8 exponent bits,
- 23 fraction bits (24 bits of precision including the implicit leading 1 for normals).

This is the binary32 specialization used by TorchLean's runtime and numerical proofs.

## The three meanings we support

- **Lean `Float` / `Float32`** now have logical models that the kernel can reduce. Compiled programs
  replace core operations with native implementations, so compiler and hardware conformance remain
  explicit runtime boundaries. `NN.Floats.IEEEExec.Bridge.LeanFloat32` connects the logical
  binary32 model to TorchLean's independent executor where agreement has been proved or stated.

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

The three types deliberately have different names:

- theorem statements and error bounds typically use `FP32`,
- ordinary native runs use Lean's `Float32`,
- reference executions use `IEEE32Exec`,
- ordinary Lean `Float32` proofs unfold through `Float32.Model`, while native execution is
  covered by a separate provider contract.

This design is described in the TorchLean paper appendix ("Appendix C (Numerical Semantics)"):
`arXiv:2602.22631` (https://arxiv.org/abs/2602.22631).
-/

@[expose] public section


namespace TorchLean.Floats

/--
Executable model of the IEEE-754 binary32 operations implemented by `IEEE32Exec`.

This is the scalar type you pick when you want runs inside Lean to have an explicit float32 meaning
(including NaN/Inf and signed-zero behavior), rather than depending on the platform runtime.
-/
abbrev IEEE32Exec : Type := TorchLean.Floats.IEEE754.IEEE32Exec

end TorchLean.Floats
