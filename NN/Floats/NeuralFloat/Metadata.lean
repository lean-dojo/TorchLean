/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Core
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# NeuralFloat Metadata

The Flocq-style core model is rounded arithmetic on `ℝ`. TorchLean attaches two pieces of
non-semantic information when converting values for runtime-refinement arguments:

- the phase in which a value was produced;
- a named hardware format and a conservative absolute error bound.

These notions are deliberately separate from `NeuralFloat`, which remains the pure
mantissa/exponent representation used by the generic format theory.
-/

@[expose] public section


namespace TorchLean.Floats

/--
Training phases for neural networks.

This is a coarse provenance tag, not a model of optimizer state or numerical execution.
-/
inductive TrainingPhase
  | forward
  | backward
  | update
  | inference

/--
Named precision levels commonly used in ML.

These carry the intended mantissa/exponent widths. Bit-level IEEE-754 behavior lives elsewhere
(`NN/Floats/IEEEExec`), and the “finite, rounding-only” float32 semantics used in most proofs is
`NN/Floats/FP32`.
-/
inductive NeuralPrecision
  | brainFloat16
  | ieeeHalf
  | ieeeSingle
  | ieeeDouble
  | tensorFloat32

namespace NeuralPrecision

/-- Exponent bit width (informational). -/
def expBits : NeuralPrecision → ℕ
  | brainFloat16 => 8
  | ieeeHalf => 5
  | ieeeSingle => 8
  | ieeeDouble => 11
  | tensorFloat32 => 8

/-- Stored mantissa (fraction) bit width (informational). -/
def mantissaBits : NeuralPrecision → ℕ
  | brainFloat16 => 7
  | ieeeHalf => 10
  | ieeeSingle => 23
  | ieeeDouble => 52
  | tensorFloat32 => 10

/-- A common “machine epsilon” proxy: `2^{-mantissaBits}` for binary-like formats. -/
noncomputable def machineEpsilon (p : NeuralPrecision) : ℝ :=
  (2 : ℝ) ^ (-(p.mantissaBits : ℤ))

end NeuralPrecision

/-- Non-semantic annotations attached to a rounded value by conversion and analysis utilities. -/
structure NeuralFloatMetadata where
  /-- Named precision used to produce the value. -/
  precision : NeuralPrecision
  /-- Nonnegative absolute error budget attached by the producing conversion. -/
  errorBound : ℝ
  /-- The attached error budget is nonnegative. -/
  errorBound_nonneg : 0 ≤ errorBound
  /-- Training phase in which the value was produced. -/
  phase : TrainingPhase

/-- A pure `NeuralFloat` together with ML-specific analysis metadata. -/
structure AnnotatedNeuralFloat (β : NeuralRadix) where
  /-- Mathematical mantissa/exponent value. -/
  value : NeuralFloat β
  /-- Metadata that does not affect the real denotation of `value`. -/
  metadata : NeuralFloatMetadata

namespace AnnotatedNeuralFloat

/-- Real denotation of an annotated value; metadata has no semantic effect. -/
noncomputable def toReal {β : NeuralRadix} (x : AnnotatedNeuralFloat β) : ℝ :=
  neuralToReal x.value

end AnnotatedNeuralFloat

end TorchLean.Floats
