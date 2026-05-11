/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# NeuralFloat metadata (training phase, named precisions)

The Flocq-style core model is "rounded arithmetic on `ℝ`". In TorchLean we sometimes want a little
extra structure *around* that core:

- a coarse notion of which part of training we're in (forward vs backward vs parameter update),
- named precision levels commonly used in ML (FP16/bfloat16/TF32/FP32/FP64).

We keep these notions in a separate file so that `Core.lean` can stay focused on the Flocq-style
mantissa/exponent machinery while still letting higher-level layers talk about mixed precision.
-/

@[expose] public section


namespace TorchLean.Floats

/--
Training phases for neural networks.

This is a coarse classifier used by a few mixed-precision policies and specifications; it is not a
model of the full optimizer state.
-/
inductive TrainingPhase
  | forward
  | backward
  | update
  | inference

namespace TrainingPhase

/-- Phases where we typically want to be more conservative about rounding/error. -/
def requiresHighPrecision : TrainingPhase → Bool
  | backward => true
  | update => true
  | forward => false
  | inference => false

/-- `forward` does not request extra precision. -/
@[simp] lemma requires_high_precision_forward : requiresHighPrecision forward = false := rfl
/-- `backward` requests extra precision (more conservative bounds). -/
@[simp] lemma requires_high_precision_backward : requiresHighPrecision backward = true := rfl
/-- `update` requests extra precision (more conservative bounds). -/
@[simp] lemma requires_high_precision_update : requiresHighPrecision update = true := rfl
/-- `inference` does not request extra precision. -/
@[simp] lemma requires_high_precision_inference : requiresHighPrecision inference = false := rfl

end TrainingPhase

/--
Named precision levels commonly used in ML.

These carry the intended mantissa/exponent widths. Bit-level IEEE-754 behavior lives elsewhere
(`NN/Floats/IEEEExec`), and the “finite, rounding-only” float32 semantics used in most proofs is
`NN/Floats/FP32`.
-/
inductive NeuralPrecision
  | brain_float16
  | ieee_half
  | ieee_single
  | ieee_double
  | tensor_float32

namespace NeuralPrecision

/-- Exponent bit width (informational). -/
def expBits : NeuralPrecision → ℕ
  | brain_float16 => 8
  | ieee_half => 5
  | ieee_single => 8
  | ieee_double => 11
  | tensor_float32 => 8

/-- Stored mantissa (fraction) bit width (informational). -/
def mantissaBits : NeuralPrecision → ℕ
  | brain_float16 => 7
  | ieee_half => 10
  | ieee_single => 23
  | ieee_double => 52
  | tensor_float32 => 10

/-- Total bit width (sign + exponent + mantissa bits). -/
def totalBits (p : NeuralPrecision) : ℕ :=
  1 + p.expBits + p.mantissaBits

/-- A common “machine epsilon” proxy: `2^{-mantissa_bits}` for binary-like formats. -/
noncomputable def machineEpsilon (p : NeuralPrecision) : ℝ :=
  (2 : ℝ) ^ (-(p.mantissaBits : ℤ))

end NeuralPrecision

/--
Mixed-precision configuration: which named precision to use in each stage.

This is a convenience record used by a few demos/spec layers; it is not part of the Flocq format
definitions (`FIX`/`FLX`/`FLT`), but it gives a simple way to state “forward in FP16, gradients in
FP32”, etc.
-/
structure NeuralMixedFormat where
  /-- Precision used for the forward pass. -/
  forward_format : NeuralPrecision
  /-- Precision used for the backward pass (gradients/VJPs). -/
  backward_format : NeuralPrecision
  /-- Precision used for stored parameters (weights/biases). -/
  param_format : NeuralPrecision
  /-- Precision used for accumulated gradients. -/
  grad_format : NeuralPrecision
  /-- Precision used for the scalar loss / reductions. -/
  loss_format : NeuralPrecision

/--
A conservative default used by TorchLean demos:

- FP16 forward (for speed),
- FP32 for gradients/params/loss (for stability).
-/
def NeuralMixedFormat.default : NeuralMixedFormat where
  forward_format := NeuralPrecision.ieee_half
  backward_format := NeuralPrecision.ieee_single
  param_format := NeuralPrecision.ieee_single
  grad_format := NeuralPrecision.ieee_single
  loss_format := NeuralPrecision.ieee_single

end TorchLean.Floats
