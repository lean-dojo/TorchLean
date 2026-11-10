/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Real

/-!
# Native arithmetic contract

The LibTorch backend computes in opaque native tensors, and the Lean proofs never inspect the
compiled ATen operations, their library dependencies or the device. This module states what a
native result has to satisfy for FloatLib's rounding theorems to apply to it, in every IEEE format
of FloatLib's configured binary family (binary32 and binary64 among them), and proves what
follows. A native kernel returns a bit word `w`, which enters as the value `ofBits w`. Two values
of a format are equal exactly when their stored bit words are.

- `EqUpToNaN native reference`: the two values are equal, or both are NaNs, whose encodings
  may differ.
- `Agreement operation native`: on every operand triple the native function agrees up to
  NaN encodings with `reference operation`, FloatLib's addition, subtraction, multiplication,
  division, square root or fused multiply-add.
- `Agreement.eq_of_isFinite`: a finite native result is FloatLib's result.
- `Agreement.abs_toReal_sub_le`: on the operation's `Domain`, a finite native result is
  within half an ulp of the exact real result.

The agreement is a hypothesis about one build, not a theorem about native code.
`scripts/checks/cuda.sh parity` tests it in binary32 for addition, multiplication, division,
square root and fused multiply-add, against compiled host arithmetic and ATen GPU operations, with
identical bits except NaN encodings. No parity sweep covers subtraction or binary64, and a passing
case is evidence for that input and build only. A reduction also needs its schedule, which
`LibTorch.reduceSumTree` specifies for the deterministic CUDA sum.

We wrote the contract by hand. Its reference is FloatLib's IEEE 754-2019 arithmetic under
roundTiesToEven, on a grid with gradual underflow. `TorchLean.Test.NativePrimitives.run` checks
Lean's host `Float32` addition, multiplication, division and square root against it bit for bit on
15 curated cases, among them `1 + 2^-24`, which ties to `1`, and `2^-126 * 0.5`, which lands on a
subnormal. `torch.set_flush_denormal(True)` (PyTorch 2.8 docs) disables subnormals on x86 CPUs
with SSE3 and on AArch64. On x86, PyTorch 2.8.0 sets both the flush-to-zero and the
denormals-are-zero bit. A CPU result computed in that mode breaks `Agreement` wherever the
reference is subnormal and at some subnormal operands too: `sqrt (2^-148)` gives `0` against the
reference `2^-74`.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat.Binary (isFinite isNaN isZero signBit toReal toModel)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

namespace TorchLean.LibTorch.Contract

open FloatLib.Floats.Formats.BinaryInterchange

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- A native result agrees with a reference result when the two are equal or both are NaNs,
whose encodings may differ. -/
def EqUpToNaN (native reference : Value) : Prop :=
  native = reference ∨ (isNaN native = true ∧ isNaN reference = true)

/-- A finite native result equals the reference result it agrees with. -/
theorem EqUpToNaN.eq_of_isFinite {native reference : Value}
    (h : EqUpToNaN native reference) (hresult : isFinite native = true) :
    native = reference := by
  rcases h with heq | ⟨hnan, _⟩
  · exact heq
  · change Model.isFinite (toModel native) = true at hresult
    change Model.isNaN (toModel native) = true at hnan
    simp [Model.isNaN_eq_false_of_isFinite_eq_true _ hresult] at hnan

/-- The exact real result of `operation`, which reads as many operands as its arity. -/
noncomputable def exactResult : ExecFloat.Backend.Operation → ℝ → ℝ → ℝ → ℝ
  | .add, x, y, _ => x + y
  | .sub, x, y, _ => x - y
  | .mul, x, y, _ => x * y
  | .div, x, y, _ => x / y
  | .sqrt, x, _, _ => Real.sqrt x
  | .fma, x, y, z => x * y + z

/-- The operands on which a finite result of `operation` is the rounded exact result: a finite
divisor for division, a finite operand that is zero or has sign bit `false` for square root, and
all operands for the other operations. -/
def Domain : ExecFloat.Backend.Operation → Value → Value → Value → Prop
  | .div, _, y, _ => isFinite y = true
  | .sqrt, x, _, _ => isFinite x = true ∧ (isZero x = true ∨ signBit x = false)
  | _, _, _, _ => True

section Reference

variable [capAdd : ExecFloat.Add (Configured.Family format code plan)]
  [capSub : ExecFloat.Sub (Configured.Family format code plan)]
  [capMul : ExecFloat.Mul (Configured.Family format code plan)]
  [capDiv : ExecFloat.Div (Configured.Family format code plan)]
  [capSqrt : ExecFloat.Sqrt (Configured.Family format code plan)]
  [capFma : ExecFloat.Fma (Configured.Family format code plan)]

/-- FloatLib's result of `operation`, which reads as many operands as its arity. -/
def reference : ExecFloat.Backend.Operation → Value → Value → Value → Value
  | .add, x, y, _ => x + y
  | .sub, x, y, _ => x - y
  | .mul, x, y, _ => x * y
  | .div, x, y, _ => x / y
  | .sqrt, x, _, _ => ExecFloat.sqrt x
  | .fma, x, y, z => ExecFloat.fma x y z

/-- The native function `native` agrees with FloatLib's `operation` up to NaN encodings on every
operand triple. -/
def Agreement (operation : ExecFloat.Backend.Operation)
    (native : Value → Value → Value → Value) : Prop :=
  ∀ x y z, EqUpToNaN (native x y z) (reference operation x y z)

/-- A finite result of a native function that agrees with `operation` is FloatLib's result. -/
theorem Agreement.eq_of_isFinite {operation : ExecFloat.Backend.Operation}
    {native : Value → Value → Value → Value} (h : Agreement operation native)
    (x y z : Value) (hresult : isFinite (native x y z) = true) :
    native x y z = reference operation x y z :=
  (h x y z).eq_of_isFinite hresult

section Rounding

variable [ExecFloat.Capability.HasSpec capAdd Configured.Spec.add]
  [ExecFloat.Capability.HasSpec capSub Configured.Spec.sub]
  [ExecFloat.Capability.HasSpec capMul Configured.Spec.mul]
  [ExecFloat.Capability.HasSpec capDiv Configured.Spec.div]
  [ExecFloat.Capability.HasSpec capSqrt Configured.Spec.sqrt]
  [ExecFloat.Capability.HasSpec capFma Configured.Spec.fma]

/-- On its `Domain`, a finite FloatLib result of `operation` is the nearest-even rounding of the
exact result, in an IEEE format. -/
theorem toReal_reference_eq_roundAt {operation : ExecFloat.Backend.Operation} {x y z : Value}
    (hdomain : Domain operation x y z) (hresult : isFinite (reference operation x y z) = true)
    (hformat : format.isIEEE = true := by ieee_format) :
    toReal (reference operation x y z) =
      Model.roundAt format (exactResult operation (toReal x) (toReal y) (toReal z)) := by
  cases operation
  · exact ExecFloat.Binary.toReal_add_eq_roundAt hresult hformat
  · exact ExecFloat.Binary.toReal_sub_eq_roundAt hresult hformat
  · exact ExecFloat.Binary.toReal_mul_eq_roundAt hresult hformat
  · exact ExecFloat.Binary.toReal_div_eq_roundAt hdomain hresult hformat
  · exact ExecFloat.Binary.toReal_execFloat_sqrt_eq_roundAt hdomain.1 hdomain.2 hformat
  · exact ExecFloat.Binary.toReal_fma_eq_roundAt hresult hformat

/-- On its `Domain`, a finite result of a native function that agrees with `operation` is within
half an ulp of the exact result, in an IEEE format. -/
theorem Agreement.abs_toReal_sub_le {operation : ExecFloat.Backend.Operation}
    {native : Value → Value → Value → Value} (h : Agreement operation native)
    {x y z : Value} (hdomain : Domain operation x y z) (hresult : isFinite (native x y z) = true)
    (hformat : format.isIEEE = true := by ieee_format) :
    |toReal (native x y z) - exactResult operation (toReal x) (toReal y) (toReal z)| ≤
      Model.halfUlpAt format (exactResult operation (toReal x) (toReal y) (toReal z)) := by
  have heq := h.eq_of_isFinite x y z hresult
  rw [heq] at hresult ⊢
  rw [toReal_reference_eq_roundAt hdomain hresult hformat]
  exact Model.abs_roundAt_sub_le format _

end Rounding

end Reference

end TorchLean.LibTorch.Contract
