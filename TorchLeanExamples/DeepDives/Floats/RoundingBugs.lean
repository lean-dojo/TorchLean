/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.VJPOp
public import TorchLean.Tensor.Elementwise
public import TorchLean.Tensor.Movement
public import TorchLean.Tensor.Layers.Dropout
public import TorchLean.Tensor.Layers.Embedding
public import TorchLean.Tensor.Layers.Linear
public import TorchLean.Tensor.Layers.Loss
public import TorchLean.Tensor.Layers.Softmax
public import TorchLean.Tensor.ConfiguredBinary
public meta import TorchLean.Tensor.Einops.Elab.Einsum
public meta import TorchLean.Tensor.Einops.Elab.Pack
public meta import TorchLean.Tensor.Literal
public meta import TorchLean.Tensor.Einops.Elab.Transform
public meta import TorchLean.Tensor.Tactic.Einops.Report
public import FloatLib.Numerics.Enclosure.Elementary.Runtime
public import TorchLean.Tensor.Context.Real
public import FloatLib.Floats.Formats.BinaryInterchange.Analysis.StandardModel
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Real
public import FloatLib.Numerics.MixedError.Basic
public import TorchLean.Verify.RuntimeApprox.Constants
public import TorchLean.Verify.RuntimeApprox.Roundoff.Coverage
public import TorchLean.Verify.RuntimeApprox.NF.CROWN
public import TorchLean.Verify.NumericalCertificate
public import TorchLean.Verify.Floats.IEEE754.Expressions
public import TorchLean.Verify.RuntimeApprox.NF.Attention
public import TorchLean.Verify.RuntimeApprox.NF.Convolution
public import TorchLean.Verify.RuntimeApprox.NF.EndToEnd
public import TorchLean.Verify.RuntimeApprox.NF.Normalization
public import TorchLean.Verify.RuntimeApprox.NF.Optimizers
public import TorchLean.Verify.RuntimeApprox.NF.ReductionOps
public import TorchLean.Verify.Floats.IEEE754.Reductions
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Autograd -- shake: keep
public import TorchLean.Verify.RuntimeApprox.TransformerBlock
public import TorchLean.Verify.RuntimeApprox.Roundoff.SoftmaxShift
public import TorchLean.Tensor.Layers.Loss.Axis
public import TorchLean.NN.Models.GradientBoostedTrees
import FloatLib.Numerics.Enclosure.Elementary.Proof
import TorchLean.NN.Softmax.Correctness
import TorchLean.Verify.RuntimeApprox.Roundoff.CrossEntropy
import TorchLean.NN.Loss.Clamp
import TorchLean.NN.Loss.Cosine
public import FloatLib.Numerics.Automation.Attributes
public import TorchLean.Tensor.Literal
public import TorchLean.Tensor.ShapeChange.Flatten
public import TorchLean.Tensor.Tactic.Init
public import TorchLean.Verify.RuntimeApprox.Roundoff.Distillation

/-!
# Rounding bugs in standard formulas, and their stable replacements

Floating-point bugs in standard (or direct) formulas, with counterexamples that the kernel checks
in IEEE formats and theorems about the stable formulas TorchLean computes.

* `SoftmaxCrossEntropy` and `SoftmaxCollapse`: Softmax Collapse. At logits `[9, -9, -9]` the
  max-shifted softmax and cross-entropy round `1 + 2 exp (-18)` to one in binary32, so the target
  log-probability, the loss and the target gradient come out zero. `SoftmaxCrossEntropy` runs
  TorchLean's operators on the example in native `Float32` as part of `lake test`, and
  `SoftmaxCollapse` proves the collapse.
* `SoftmaxSign`: under conditions on the computed values, the stable cross-entropy gradient of a
  one-hot target over two or more classes is negative at the target and nonnegative elsewhere, in
  every IEEE format and at every length.
* `Cancellation`: the cosine loss `1 - p̂ · t̂` and the Gini impurity `1 - Σ p²` cancel
  catastrophically, which the stable formulas avoid.
* `Clamp`: at the upper clamp the standard cross-entropy of probabilities and binary
  cross-entropy take the logarithm of the rounded `1 - epsilon`, and the loss of a prediction with
  probability one, about `epsilon`, can come out zero. Taking the logarithm as `log1p (-epsilon)`
  keeps it.
* `CosineGuard`: a cosine loss guarded by a positive epsilon falls back to `1 - p̂ · t̂` at a
  nonpositive epsilon, although the stable formula holds there. `cosineLoss` uses the stable
  formula whenever both norms are positive and above their floors.
-/

@[expose] public section

namespace TorchLean.Examples

/-!
## Softmax and cross-entropy at large margins

These checks run the examples of Barron, Gnanapragasam and Tschannen,
[A Fix for Softmax and Cross-Entropy](https://arxiv.org/abs/2610.31642), through TorchLean's
log-softmax, softmax and cross-entropy operators in native `Float32`, the host's binary32
arithmetic. At logits `[9, -9, -9]` with target `0`, the max-shifted formulas round
`1 + 2 exp (-18)` to one in binary32 and return zero for the target log-probability, the loss and
the target gradient. TorchLean's operators keep every result within 32 units of roundoff, and
`distillLossBackward` vanishes at the equal teacher logits `[2, 1, 0]`. Reference values are
midpoints of FloatLib's rational exponential and logarithm enclosures.

`Standard` holds the max-shifted formulas, which reproduce the collapse. `SoftmaxCollapse`
evaluates them with the correctly rounded `exp` and `log` of FloatLib's IEEE formats, so the
collapse comes from the formulas and not from the primitives.
-/


section

open Tensor
open FloatLib.Floats FloatLib.Numerics
open Softmax (maxShiftedExpVec maxVec)

/-- Fail with `label` unless `condition` holds. -/
def SoftmaxCrossEntropy.expect (label : String) (condition : Bool) : IO Unit := do
  unless condition do
    throw <| IO.userError s!"softmax and cross-entropy: {label}"

/-! ## Standard formulas

These formulas divide the max-shifted exponentials of `Softmax.maxShiftedExpVec` by their
rounded sum and take the logarithm of that sum. -/

namespace StandardSoftmax

section

variable {α : Type} [Storage α] [Context α]

/-- Standard vector softmax: max-shifted exponentials divided by their rounded sum. -/
def softmaxVec {n : Nat} (t : Tensor α [n]) : Tensor α [n] :=
  match n with
  | 0 => t
  | Nat.succ _ =>
      let ex := maxShiftedExpVec t
      let denom : α := Tensor.sum ex
      ex / Tensor.full _ denom

/-- Standard last-axis softmax. -/
def softmaxInnermost : {s : Shape} → Tensor α s → Tensor α s :=
  Tensor.lastAxisMap 1 fun _ => softmaxVec

/-- Standard last-axis softmax VJP, `y * (dY - <dY, y>)`, without centering the upstream. -/
def softmaxInnermostBackward : {s : Shape} → Tensor α s → Tensor α s → Tensor α s
  | .scalar, _x, _dY => Tensor.scalar 0
  | .dim n .scalar, x, dY =>
      let y := softmaxVec (α := α) (n := n) x
      let s : α := Tensor.sum (dY * y)
      y * (dY - Tensor.full _ s)
  | .dim n inner, x, dY =>
      Tensor.dim fun i : Fin n =>
        softmaxInnermostBackward (s := inner)
          (Tensor.unstack x i) (Tensor.unstack dY i)

/-- Standard softmax along `axis`. -/
def softmax {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (x : Tensor α s) : Tensor α s :=
  Tensor.alongAxis ⟨axis, Shape.AxisInBounds.proof⟩ (fun x => softmaxInnermost x) x

/-- Standard softmax VJP along `axis`. -/
def softmaxBackward {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (x dY : Tensor α s) : Tensor α s :=
  Tensor.alongAxis₂ ⟨axis, Shape.AxisInBounds.proof⟩ softmaxInnermostBackward x dY

/-- Standard vector log-softmax: the shifted logits minus the logarithm of the rounded sum. -/
def logSoftmaxVec {n : Nat} (t : Tensor α [n]) : Tensor α [n] :=
  match n with
  | 0 => t
  | Nat.succ n' =>
      let maxT : Tensor α .scalar := maxVec t
      let shifted : Tensor α [Nat.succ n'] := t - Tensor.full _ maxT.item
      let ex := maxShiftedExpVec t
      let denom : α := Tensor.sum ex
      let logDenom : α := MathFunctions.log denom
      shifted - Tensor.full _ logDenom

/-- Standard last-axis log-softmax. -/
def logSoftmaxInnermost : {s : Shape} → Tensor α s → Tensor α s :=
  Tensor.lastAxisMap 0 fun _ => logSoftmaxVec

/-- Standard log-softmax along `axis`. -/
def logSoftmax {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (x : Tensor α s) : Tensor α s :=
  Tensor.alongAxis ⟨axis, Shape.AxisInBounds.proof⟩ (fun x => logSoftmaxInnermost x) x

end

variable {α : Type} [Storage α] [Context α]

/-- Standard cross-entropy on logits, weighting the standard log-softmax by the target. -/
def crossEntropyLogits {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (logits : Tensor α s) (target : Tensor α s) : α :=
  let logp := logSoftmax (α := α) (s := s) axis logits
  let total := Tensor.sum (target * logp)
  meanOverAxisSlices (s := s) ⟨axis, Shape.AxisInBounds.proof⟩ (-total)

/-- Standard cross-entropy gradient, `softmax(logits) * sum(target) - target`. -/
def crossEntropyLogitsDeriv {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (logits : Tensor α s) (target : Tensor α s) :
    Tensor α s :=
  let probs := softmax (α := α) (s := s) axis logits
  let targetMass :=
    castShape (Tensor.reduce .sum target [⟨axis, Shape.AxisInBounds.proof⟩] false)
      (Shape.reduce_singleton _ _)
  let grad := (probs *
    Tensor.broadcast targetMass s _ (s.eraseIdx_broadcastDims axis)) - target
  Tensor.scale grad (Context.divNat 1 (axisMeanDenom s ⟨axis, Shape.AxisInBounds.proof⟩))

end StandardSoftmax

/-! ## Precisions and references -/

/-- A scalar type under test, with its exact decoding and unit roundoff. -/
structure SoftmaxCrossEntropy.Precision (α : Type) where
  /-- Name used in failure messages. -/
  name : String
  /-- The exact value of a finite scalar, or `none` for NaN and infinities. -/
  decode : α → Option Rat
  /-- Half the distance from one to the next larger value of the format. -/
  unitRoundoff : Rat

/-- Native `Float32`, decoded from its binary32 word. -/
def SoftmaxCrossEntropy.float32 : Precision Float32 :=
  ⟨"native Float32",
    fun x => ExecFloat.Binary.toRat? (ExecFloat.Binary.ofNatBits x.toBits.toNat : Binary32),
    1 / 2 ^ 24⟩

/-- The relative error allowed by the checks, in units of roundoff. Rounding a log-probability
near `-18` can by itself move its exponential by about 18 units. -/
def SoftmaxCrossEntropy.tolerance : Rat := 32

/-- Whether `actual` is finite and within `tolerance` units of roundoff of `exact`, relative to
`exact`. -/
def SoftmaxCrossEntropy.Precision.near {α : Type} (precision : Precision α) (actual : α)
    (exact : Rat) : Bool :=
  match precision.decode actual with
  | some value => |value - exact| ≤ tolerance * precision.unitRoundoff * |exact|
  | none => false

/-- Whether every coordinate of a three-class result is near its exact value. -/
def SoftmaxCrossEntropy.Precision.nearAll {α : Type} [Storage α] (precision : Precision α)
    (actual : Tensor α [3]) (exact : Fin 3 → Rat) : Bool :=
  (List.finRange 3).all fun i => precision.near (actual.getScalar i) (exact i)

namespace SoftmaxCrossEntropy.Reference

/-- FloatLib's rational enclosure of `exp x`. -/
def expBounds (x : Rat) : RationalInterval := Enclosure.exp x 30

/-- FloatLib's rational enclosure of `log x`. -/
def logBounds (x : Rat) : RationalInterval := Enclosure.log x 30

/-- The midpoint of `expBounds x`. -/
def exp (x : Rat) : Rat := ((expBounds x).lo + (expBounds x).hi) / 2

/-- The midpoint of `logBounds x`. -/
def log (x : Rat) : Rat := ((logBounds x).lo + (logBounds x).hi) / 2

/-- `exp (-18)`, the weight of each non-target class at logits `[9, -9, -9]`. -/
def tail : Rat := exp (-18)

end SoftmaxCrossEntropy.Reference

/-- Check that the enclosures behind the references are narrower than `2 ^ -100` of their
values, far below every tolerance used here. -/
def SoftmaxCrossEntropy.checkReferences : IO Unit := do
  let tight (bounds : RationalInterval) := bounds.hi - bounds.lo ≤ bounds.lo / 2 ^ 100
  expect "exponential enclosure at -18" (tight (Reference.expBounds (-18)))
  expect "logarithm enclosure" (tight (Reference.logBounds (1 + 2 * Reference.tail)))

/-! ## Softmax Collapse -/

section Collapse

variable {α : Type} [Storage α] [Context α]

/-- Logits `[9, -9, -9]`. Their margin `18 - log 2` is past the binary32 precision cliff. -/
def SoftmaxCrossEntropy.logits : Tensor α [3] := Tensor.ofFn fun i => if i = 0 then 9 else -9

/-- The one-hot target of class `0`. As an upstream gradient it selects `softmax(z)[0]`. -/
def SoftmaxCrossEntropy.target : Tensor α [3] := Tensor.ofFn fun i => if i = 0 then 1 else 0

/-- Log-softmax, the gradient of `softmax(z)[0]`, and the cross-entropy value and gradient at
`[9, -9, -9]` stay near their exact values. -/
def SoftmaxCrossEntropy.checkCollapse (precision : Precision α) : IO Unit := do
  let e := Reference.tail
  let normalizer := Reference.log (1 + 2 * e)
  let z : Tensor α [3] := logits
  let t : Tensor α [3] := target
  expect s!"{precision.name} log-softmax keeps the target log-probability"
    (precision.nearAll (Tensor.logSoftmax z ⟨0, Nat.one_pos⟩) fun i =>
      if i = 0 then -normalizer else -18 - normalizer)
  expect s!"{precision.name} softmax keeps the gradient of its target probability"
    (precision.nearAll (Tensor.softmaxBackward z t ⟨0, Nat.one_pos⟩) fun i =>
      if i = 0 then 2 * e / (1 + 2 * e) ^ 2 else -e / (1 + 2 * e) ^ 2)
  expect s!"{precision.name} cross-entropy keeps the loss"
    (precision.near (Tensor.crossEntropy .logits z t ⟨0, Nat.one_pos⟩) normalizer)
  expect s!"{precision.name} cross-entropy keeps the target gradient"
    (precision.nearAll
      (Tensor.scale (Tensor.crossEntropyBackward .logits z t ⟨0, Nat.one_pos⟩) 1) fun i =>
      if i = 0 then -2 * e / (1 + 2 * e) else e / (1 + 2 * e))

end Collapse

/-! ## Softmax Cancellation -/

/-- Teacher logits `[2, 1, 0]`. -/
def SoftmaxCrossEntropy.teacher {α : Type} [Storage α] [Context α] : Tensor α [3] :=
  Tensor.ofFn fun i => if i = 0 then 2 else if i = 1 then 1 else 0

section Cancellation

variable {α : Type} [Storage α] [Context α]

/-- The distillation gradient is exactly zero when the student logits equal the teacher's. -/
def SoftmaxCrossEntropy.checkDistillZero (precision : Precision α) : IO Unit := do
  let z : Tensor α [3] := teacher
  let gradient := Tensor.distillLossBackward ⟨0, Shape.AxisInBounds.proof⟩ z z
  expect s!"{precision.name} distillation gradient vanishes at equal logits"
    ((List.finRange 3).all fun i => precision.decode (gradient.getScalar i) == some 0)

end Cancellation

/-- Run the softmax and cross-entropy checks in native `Float32`. -/
def SoftmaxCrossEntropy.run : IO Unit := do
  checkReferences
  checkCollapse float32
  checkDistillZero float32
  IO.println "  softmax and cross-entropy at large margins: passed"

end

/-!
## Softmax Collapse, checked by the kernel

`SoftmaxCrossEntropy` runs the Softmax Collapse example of Barron, Gnanapragasam and Tschannen
in native `Float32`. Here the same example in FloatLib binary32, and its bfloat16 and binary64
counterparts, are theorems.

At logits `[m, -m, -m]` with target `0`, the target gradient of the real cross-entropy is `-c`,
where `c = 2 exp (-2 m) / (1 + 2 exp (-2 m))` (`gradient_real`). The standard formulas of
`Examples.StandardSoftmax` add the two small exponentials to one. When `2 exp (-2 m)` is below
the unit roundoff `u`, the sum rounds to one in any order of addition. The theorems take binary32
at the paper's `m = 9`, bfloat16 at `m = 4` and binary64 at `m = 19`. In each format the standard
target log-probability, loss and target gradient decode to zero, and an absolute tolerance of
about one machine epsilon accepts the zero gradient. The mixed bound
`MixedError · (-c) (32 u) (32 η)`, which adds 32 units of the format's underflow allowance `η` to
the relative tolerance of `SoftmaxCrossEntropy`, rejects it and accepts the target gradient of
`Tensor.crossEntropyBackward .logits`, which is finite and negative.

The kernel evaluates the floating-point formulas with the `MathFunctions` and `Context`
instances of each format. In these IEEE formats `exp`, `log`, `log1p` and `expm1` are FloatLib's
correctly rounded functions. FloatLib's rational enclosure of `exp` bounds `c`.
-/


namespace SoftmaxCollapse

open Tensor
open FloatLib.Floats FloatLib.Numerics
open FloatLib.Floats.Formats.BinaryInterchange

section Formulas

variable {α : Type} [Storage α] [Context α]

/-- Logits `[m, -m, -m]`. The target class `0` leads the other two by `2 * m`. -/
def logits (m : α) : Tensor α [3] := Tensor.ofFn fun i => if i = 0 then m else -m

/-- The target entry of the standard log-softmax at `logits m`. -/
def standardLogProbability (m : α) : α :=
  (StandardSoftmax.logSoftmax 0 (logits m)).getScalar 0

/-- The standard cross-entropy of `logits m` against the target `0`. -/
def standardLoss (m : α) : α :=
  StandardSoftmax.crossEntropyLogits 0 (logits m) SoftmaxCrossEntropy.target

/-- The target entry of the standard cross-entropy gradient at `logits m`. -/
def standardGradient (m : α) : α :=
  (StandardSoftmax.crossEntropyLogitsDeriv 0 (logits m)
    SoftmaxCrossEntropy.target).getScalar 0

/-- The target entry of `Tensor.crossEntropyBackward .logits` at `logits m`. -/
def gradient (m : α) : α :=
  (Tensor.crossEntropyBackward .logits (logits m) SoftmaxCrossEntropy.target ⟨0,
      by exact Nat.zero_lt_succ _⟩).getScalar 0

end Formulas

/-- The softmax probability `2 exp (-gap) / (1 + 2 exp (-gap))` of the two classes of
`logits (gap / 2)` other than the target. -/
noncomputable def tailMass (gap : ℝ) : ℝ :=
  2 * Real.exp (-gap) / (1 + 2 * Real.exp (-gap))

/-- Over the reals, the target gradient at `logits m` is `-tailProbability (2 * m)`. -/
theorem gradient_real (m : ℝ) : gradient m = -tailMass (2 * m) := by
  have connected := congrArg (fun value : Vec 3 => value 0)
    (Tensor.getScalarE_logSoftmaxBackward (logits m)
      (Tensor.map (fun x => -x) SoftmaxCrossEntropy.target))
  have coordinate : gradient m = -1 + Real.exp m / (Real.exp m + 2 * Real.exp (-m)) := by
    have hdenom : axisMeanDenom [3] ⟨0, Shape.AxisInBounds.proof⟩ = 1 := rfl
    simp only [gradient, Tensor.crossEntropyBackward, CrossEntropy.logProbs_real, reduceIte, hdenom]
    simp [Vec.logSoftmaxVJP, Vec.softmax, Vec.sumExp,
      Fin.sum_univ_three, logits, SoftmaxCrossEntropy.target] at connected ⊢
    rw [connected]
    ring
  have hexp : Real.exp (-(2 * m)) = Real.exp (-m) * Real.exp (-m) := by
    rw [← Real.exp_add]
    ring_nf
  have hinverse : Real.exp m * Real.exp (-m) = 1 := by
    rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]
  rw [coordinate, tailMass, hexp]
  have hpositive : 0 < Real.exp (-m) := Real.exp_pos _
  field_simp
  nlinarith [hinverse]

/-- Rational bounds `lo` and `hi` on the endpoints of FloatLib's enclosure of `exp (-gap)` bound
`tailMass gap` by `2 lo / (1 + 2 lo)` and `2 hi / (1 + 2 hi)`, so by any `a` below the
first and any `b` above the second. -/
private theorem tailMass_mem_Icc (gap degree : ℕ) (lo hi : ℚ) {a b : ℝ}
    (hlo₀ : 0 ≤ lo) (hlo : lo ≤ (Enclosure.exp (-gap) degree).lo)
    (hhi : (Enclosure.exp (-gap) degree).hi ≤ hi)
    (ha : a ≤ ((2 * lo / (1 + 2 * lo) : ℚ) : ℝ)) (hb : ((2 * hi / (1 + 2 * hi) : ℚ) : ℝ) ≤ b) :
    tailMass gap ∈ Set.Icc a b := by
  obtain ⟨lower, upper⟩ := Enclosure.contains_exp (-gap) degree
  have increasing {s t : ℝ} (hs : 0 ≤ s) (hst : s ≤ t) :
      2 * s / (1 + 2 * s) ≤ 2 * t / (1 + 2 * t) := by
    rw [div_le_div_iff₀ (by positivity) (by linarith)]
    nlinarith
  push_cast at lower upper ha hb ⊢
  exact ⟨ha.trans (increasing (by exact_mod_cast hlo₀) ((Rat.cast_le.mpr hlo).trans lower)),
    (increasing (Real.exp_pos _).le (upper.trans (Rat.cast_le.mpr hhi))).trans hb⟩

section Bounds

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" =>
  ExecFloat.Binary format.expWidth format.fracWidth format.encoding format.exponentBias
    format.expWidth_ge_two format.fracWidth_pos format.exponentBias_pos
    format.exponentBias_le_maxFinite plan code

/-- A value that decodes to zero is not within 32 units of roundoff and underflow of `-c`, for
`c ∈ [lo, hi]` with `0 < lo`, when `32 (u hi + 2 ^ e)` is below `lo` for a power `2 ^ e` at least
the underflow allowance. -/
private theorem not_mixedError_of_eq_zero {x : Value} {c lo hi : ℝ} (e : ℤ)
    (hx : ExecFloat.Binary.toRat? x = some 0) (hc : c ∈ Set.Icc lo hi) (hlo₀ : 0 < lo)
    (he : FloatFormat.minSubnormalExponent format - 1 ≤ e)
    (hallowance : 32 * (Model.unitRoundoffAt format * hi + 2 ^ e) < lo) :
    ¬ MixedError (ExecFloat.Binary.toReal x) (-c) (32 * Model.unitRoundoffAt format)
      (32 * Model.underflowErrorAt format) := by
  rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hx, Rat.cast_zero]
  intro bound
  obtain ⟨hlo, hhi⟩ := hc
  have hu := Model.unitRoundoffAt_pos format
  have hη := Model.underflowErrorAt_le_zpow format he
  have hscaled := mul_le_mul_of_nonneg_left hhi hu.le
  have hpositive : 0 < c := hlo₀.trans_le hlo
  have h := bound.bound
  rw [zero_sub, neg_neg, abs_neg, abs_of_pos hpositive] at h
  linarith

/-- A value that decodes to a negative `q` is finite, negative and within 32 units of roundoff
and underflow of `-c`, for `c ∈ [lo, hi]`, when `-32 u lo ≤ q + lo` and `q + hi ≤ 32 u lo`. -/
private theorem mixedError_of_eq_some {x : Value} {q : ℚ} {c lo hi : ℝ}
    (hx : ExecFloat.Binary.toRat? x = some q) (hq : q < 0) (hc : c ∈ Set.Icc lo hi)
    (hlower : -(32 * Model.unitRoundoffAt format * lo) ≤ q + lo)
    (hupper : q + hi ≤ 32 * Model.unitRoundoffAt format * lo) :
    ExecFloat.Binary.isFinite x = true ∧
      MixedError (ExecFloat.Binary.toReal x) (-c) (32 * Model.unitRoundoffAt format)
        (32 * Model.underflowErrorAt format) ∧
      ExecFloat.Binary.toReal x < 0 := by
  obtain ⟨hlo, hhi⟩ := hc
  have hu := Model.unitRoundoffAt_pos format
  have hscaled : 32 * Model.unitRoundoffAt format * lo ≤ 32 * Model.unitRoundoffAt format * |-c| :=
    mul_le_mul_of_nonneg_left (hlo.trans ((le_abs_self c).trans (abs_neg c).ge)) (by positivity)
  rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hx]
  refine ⟨Model.Algebraic.isFinite_of_toRat?_eq_some hx,
    (MixedError.of_abs_sub_le_mul (by positivity) ?_).mono le_rfl
      (mul_nonneg (by norm_num) (Model.underflowErrorAt_pos format).le),
    by exact_mod_cast hq⟩
  rw [sub_neg_eq_add, abs_le]
  constructor <;> linarith

end Bounds

/-! ## Binary32

The paper's logits `[9, -9, -9]`. With a gap of `18`, `2 exp (-18)` is below the binary32 unit
roundoff `2^-24`. -/

/-- FloatLib's enclosure of `exp (-18)` gives `tailMass 18` to eight digits. -/
private theorem binary32_tailMass_mem_Icc :
    tailMass 18 ∈ Set.Icc (30459958 / 10 ^ 15) (30459959 / 10 ^ 15) :=
  tailMass_mem_Icc 18 30 (152299797 / 10 ^ 16) (152299798 / 10 ^ 16)
    (by norm_num) (by decide +kernel) (by decide +kernel) (by norm_num) (by norm_num)

/-- The standard formulas evaluated with the binary32 primitives return zero for the target
log-probability, the loss and the target gradient. -/
theorem binary32_standard_toRat? :
    ExecFloat.Binary.toRat? (standardLogProbability (9 : Binary32)) = some 0 ∧
      ExecFloat.Binary.toRat? (standardLoss (9 : Binary32)) = some 0 ∧
      ExecFloat.Binary.toRat? (standardGradient (9 : Binary32)) = some 0 := by
  decide +kernel

/-- The exact probability of the non-target classes lies between `3e-8` and `3.1e-8`. -/
theorem binary32_tailMass_mem_Ioo :
    tailMass 18 ∈ Set.Ioo (3 / 10 ^ 8) (31 / 10 ^ 9) := by
  obtain ⟨lower, upper⟩ := binary32_tailMass_mem_Icc
  exact ⟨lt_of_lt_of_le (by norm_num) lower, lt_of_le_of_lt upper (by norm_num)⟩

/-- An absolute tolerance of `1e-7` accepts the standard target gradient. -/
theorem binary32_abs_sub_lt :
    |ExecFloat.Binary.toReal (standardGradient (9 : Binary32)) - -tailMass 18| <
      1 / 10 ^ 7 := by
  obtain ⟨lower, upper⟩ := binary32_tailMass_mem_Ioo
  rw [ExecFloat.Binary.toReal_of_toRat?_eq_some binary32_standard_toRat?.2.2, Rat.cast_zero,
    zero_sub, neg_neg, abs_of_pos (by linarith)]
  linarith

/-- The mixed bound with 32 units of binary32 roundoff and underflow rejects the standard target
gradient. -/
theorem binary32_not_mixedError :
    ¬ MixedError (ExecFloat.Binary.toReal (standardGradient (9 : Binary32))) (-tailMass 18)
      (32 * Model.unitRoundoffAt FloatFormat.binary32)
      (32 * Model.underflowErrorAt FloatFormat.binary32) :=
  not_mixedError_of_eq_zero (-64) binary32_standard_toRat?.2.2
    binary32_tailMass_mem_Icc (by norm_num) (by decide)
    (by norm_num [Model.unitRoundoffAt_eq, FloatFormat.binary32])

/-- The target gradient `gradient 9` is finite, negative and within the mixed bound with 32
units of binary32 roundoff and underflow. -/
theorem binary32_mixedError :
    ExecFloat.Binary.isFinite (gradient (9 : Binary32)) = true ∧
      MixedError (ExecFloat.Binary.toReal (gradient (9 : Binary32))) (-tailMass 18)
        (32 * Model.unitRoundoffAt FloatFormat.binary32)
        (32 * Model.underflowErrorAt FloatFormat.binary32) ∧
      ExecFloat.Binary.toReal (gradient (9 : Binary32)) < 0 :=
  mixedError_of_eq_some (q := -2143429 / 2 ^ 46) (by decide +kernel) (by norm_num)
    binary32_tailMass_mem_Icc (by norm_num [Model.unitRoundoffAt_eq, FloatFormat.binary32])
    (by norm_num [Model.unitRoundoffAt_eq, FloatFormat.binary32])

/-! ## Bfloat16

Logits `[4, -4, -4]`. With a gap of `8`, `2 exp (-8)` is below the bfloat16 unit roundoff
`2^-8`. -/

/-- FloatLib's enclosure of `exp (-8)` gives `tailMass 8` to seven digits. -/
private theorem bfloat16_tailMass_mem_Icc :
    tailMass 8 ∈ Set.Icc (6704754 / 10 ^ 10) (6704755 / 10 ^ 10) :=
  tailMass_mem_Icc 8 30 (335462627 / 10 ^ 12) (335462628 / 10 ^ 12)
    (by norm_num) (by decide +kernel) (by decide +kernel) (by norm_num) (by norm_num)

/-- The standard formulas evaluated with the bfloat16 primitives return zero for the target
log-probability, the loss and the target gradient. -/
theorem bfloat16_standard_toRat? :
    ExecFloat.Binary.toRat? (standardLogProbability (4 : BFloat16)) = some 0 ∧
      ExecFloat.Binary.toRat? (standardLoss (4 : BFloat16)) = some 0 ∧
      ExecFloat.Binary.toRat? (standardGradient (4 : BFloat16)) = some 0 := by
  decide +kernel

/-- The exact probability of the non-target classes lies between `6.7e-4` and `6.71e-4`. -/
theorem bfloat16_tailMass_mem_Ioo :
    tailMass 8 ∈ Set.Ioo (67 / 10 ^ 5) (671 / 10 ^ 6) := by
  obtain ⟨lower, upper⟩ := bfloat16_tailMass_mem_Icc
  exact ⟨lt_of_lt_of_le (by norm_num) lower, lt_of_le_of_lt upper (by norm_num)⟩

/-- An absolute tolerance of `1e-2` accepts the standard target gradient. -/
theorem bfloat16_abs_sub_lt :
    |ExecFloat.Binary.toReal (standardGradient (4 : BFloat16)) - -tailMass 8| <
      1 / 10 ^ 2 := by
  obtain ⟨lower, upper⟩ := bfloat16_tailMass_mem_Ioo
  rw [ExecFloat.Binary.toReal_of_toRat?_eq_some bfloat16_standard_toRat?.2.2, Rat.cast_zero,
    zero_sub, neg_neg, abs_of_pos (by linarith)]
  linarith

/-- The mixed bound with 32 units of bfloat16 roundoff and underflow rejects the standard target
gradient. -/
theorem bfloat16_not_mixedError :
    ¬ MixedError (ExecFloat.Binary.toReal (standardGradient (4 : BFloat16))) (-tailMass 8)
      (32 * Model.unitRoundoffAt FloatFormat.bfloat16)
      (32 * Model.underflowErrorAt FloatFormat.bfloat16) :=
  not_mixedError_of_eq_zero (-64) bfloat16_standard_toRat?.2.2
    bfloat16_tailMass_mem_Icc (by norm_num) (by decide)
    (by norm_num [Model.unitRoundoffAt_eq, FloatFormat.bfloat16])

/-- The target gradient `gradient 4` is finite, negative and within the mixed bound with 32
units of bfloat16 roundoff and underflow. -/
theorem bfloat16_mixedError :
    ExecFloat.Binary.isFinite (gradient (4 : BFloat16)) = true ∧
      MixedError (ExecFloat.Binary.toReal (gradient (4 : BFloat16))) (-tailMass 8)
        (32 * Model.unitRoundoffAt FloatFormat.bfloat16)
        (32 * Model.underflowErrorAt FloatFormat.bfloat16) ∧
      ExecFloat.Binary.toReal (gradient (4 : BFloat16)) < 0 :=
  mixedError_of_eq_some (q := -11 / 2 ^ 14) (by decide +kernel) (by norm_num)
    bfloat16_tailMass_mem_Icc (by norm_num [Model.unitRoundoffAt_eq, FloatFormat.bfloat16])
    (by norm_num [Model.unitRoundoffAt_eq, FloatFormat.bfloat16])

/-! ## Binary64

Logits `[19, -19, -19]`. With a gap of `38`, `2 exp (-38)` is below the binary64 unit roundoff
`2^-53`. -/

/-- FloatLib's enclosure of `exp (-38)` gives `tailMass 38` to seventeen digits. -/
private theorem binary64_tailMass_mem_Icc :
    tailMass 38 ∈
      Set.Icc (62782655840960588 / 10 ^ 33) (62782655840960589 / 10 ^ 33) :=
  tailMass_mem_Icc 38 30 (313913279204802962 / 10 ^ 34)
    (313913279204802963 / 10 ^ 34) (by norm_num) (by decide +kernel) (by decide +kernel)
    (by norm_num) (by norm_num)

/-- The standard formulas evaluated with the binary64 primitives return zero for the target
log-probability, the loss and the target gradient. -/
theorem binary64_standard_toRat? :
    ExecFloat.Binary.toRat? (standardLogProbability (19 : Binary64)) = some 0 ∧
      ExecFloat.Binary.toRat? (standardLoss (19 : Binary64)) = some 0 ∧
      ExecFloat.Binary.toRat? (standardGradient (19 : Binary64)) = some 0 := by
  decide +kernel

/-- The exact probability of the non-target classes lies between `6e-17` and `6.3e-17`. -/
theorem binary64_tailMass_mem_Ioo :
    tailMass 38 ∈ Set.Ioo (6 / 10 ^ 17) (63 / 10 ^ 18) := by
  obtain ⟨lower, upper⟩ := binary64_tailMass_mem_Icc
  exact ⟨lt_of_lt_of_le (by norm_num) lower, lt_of_le_of_lt upper (by norm_num)⟩

/-- An absolute tolerance of `1e-16` accepts the standard target gradient. -/
theorem binary64_abs_sub_lt :
    |ExecFloat.Binary.toReal (standardGradient (19 : Binary64)) - -tailMass 38| <
      1 / 10 ^ 16 := by
  obtain ⟨lower, upper⟩ := binary64_tailMass_mem_Ioo
  rw [ExecFloat.Binary.toReal_of_toRat?_eq_some binary64_standard_toRat?.2.2, Rat.cast_zero,
    zero_sub, neg_neg, abs_of_pos (by linarith)]
  linarith

/-- The mixed bound with 32 units of binary64 roundoff and underflow rejects the standard target
gradient. -/
theorem binary64_not_mixedError :
    ¬ MixedError (ExecFloat.Binary.toReal (standardGradient (19 : Binary64))) (-tailMass 38)
      (32 * Model.unitRoundoffAt FloatFormat.binary64)
      (32 * Model.underflowErrorAt FloatFormat.binary64) :=
  not_mixedError_of_eq_zero (-64) binary64_standard_toRat?.2.2
    binary64_tailMass_mem_Icc (by norm_num) (by decide)
    (by norm_num [Model.unitRoundoffAt_eq, FloatFormat.binary64])

/-- The target gradient `gradient 19` is finite, negative and within the mixed bound with 32
units of binary64 roundoff and underflow. -/
theorem binary64_mixedError :
    ExecFloat.Binary.isFinite (gradient (19 : Binary64)) = true ∧
      MixedError (ExecFloat.Binary.toReal (gradient (19 : Binary64))) (-tailMass 38)
        (32 * Model.unitRoundoffAt FloatFormat.binary64)
        (32 * Model.underflowErrorAt FloatFormat.binary64) ∧
      ExecFloat.Binary.toReal (gradient (19 : Binary64)) < 0 :=
  mixedError_of_eq_some (q := -5093534167085827 / 2 ^ 106) (by decide +kernel) (by norm_num)
    binary64_tailMass_mem_Icc (by norm_num [Model.unitRoundoffAt_eq, FloatFormat.binary64])
    (by norm_num [Model.unitRoundoffAt_eq, FloatFormat.binary64])

end SoftmaxCollapse

/-!
## The sign of the cross-entropy gradient

With upstream `-e_t`, the log-softmax pullback `LogSoftmax.vecBackward`
has a finite, negative entry at the target `t` and finite, nonnegative entries elsewhere, for
every IEEE format and every length, with no condition on the length times the unit roundoff.
Both sums of the upstream that the pullback forms are exact: the row sums to `-1`, and away from
the pivot it sums to zero or to `-1`. So when `t` is the pivot its entry is the correctly rounded
`expm1` of its log-probability, negative because that log-probability is, and the other entries
are correctly rounded exponentials. When `t` is not the pivot its entry rounds `exp y_t - 1`
once and is negative when the computed exponential is below one. `roundoff` proves both
theorems.

The cross-entropy gradient `Tensor.crossEntropyBackward .logits` of a one-hot target is this
pullback at the computed log-probabilities `CrossEntropy.logProbs`. For finite logits with two or
more classes they are negative wherever the forward computation is finite, provided that
whenever the computed tail is zero, the computed exponential of the log-sum-exp of the other
classes is finite and positive and every entry of the two-level result is finite: on the
one-level branch the failed test `CrossEntropy.twoLevelAt` then makes the computed tail positive
(`Roundoff.CrossEntropy.vectorLogProbs_neg`). So the summed gradient has these signs. Mean
reduction divides each entry once by the slice count `m`: the entries stay finite, nonpositive at
the target and nonnegative elsewhere, and the target entry stays negative when `η < |g_t| / m`.
For a single class slice the division is exact.

At the logits `[9, -9, -9]` the kernel decides every condition in binary32, bfloat16 and binary16,
and in bfloat16 also for a batch of two rows. In binary32 the standard formulas of
`SoftmaxCollapse` give the target gradient at these logits the value zero. In binary16 each
computed `exp (-18)` of the tail is zero, so the one-level formula gives the target gradient the
value zero, while the two-level normalizer keeps it at `-2⁻²⁴` (`binary16_toRat?`). The theorem
also specializes to bfloat16 logits over a vocabulary of 50257 classes, where the length times the
unit roundoff is about 196. In the IEEE format with exponent width 3 and fraction width 5, at the
logits `[0, -6, …, -6]` of 18 classes, the computed exponential of the log-sum-exp of the other
classes overflows, so the log-probabilities are the one-level ones.
-/


namespace SoftmaxSign

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange FloatLib.Numerics
open TorchLean.Roundoff

/-! ## Rows of a batch -/

section Rows

variable {α : Type} [Storage α] [Context α]

/-- Row `b` of the cross-entropy gradient of a batch of class rows is the summed cross-entropy
gradient of that row, each entry divided once by the batch's slice count. -/
theorem unstack_crossEntropyBackward {batch n : Nat}
    (logits target : Tensor α [batch, n]) (b : Fin batch) (i : Fin n) :
    (Tensor.unstack (Tensor.crossEntropyBackward .logits logits target ⟨1,
        by exact Nat.one_lt_succ_succ _⟩) b).getScalar i =
      (Tensor.map (Context.divNat · (Tensor.axisMeanDenom [batch, n] ⟨1, Shape.AxisInBounds.proof⟩))
        (Tensor.crossEntropyBackward .logits (Tensor.unstack logits b) (Tensor.unstack target b)
          ⟨0, by exact Nat.zero_lt_succ _⟩ false)).getScalar i := by
  change (Tensor.unstack (Tensor.map (Context.divNat · _) (Tensor.dim fun b =>
      LogSoftmax.vecBackward
        (Tensor.unstack
          (Tensor.dim fun b =>
            CrossEntropy.logProbs (Tensor.unstack logits b) ⟨0, Nat.one_pos⟩) b)
        (Tensor.unstack (Tensor.map (fun x => -x) target) b))) b).getScalar i = _
  simp only [Tensor.unstack_map, Tensor.unstack_dim]
  rfl

end Rows

section General

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-! ## Division by the slice count -/

/-- A finite value divided by the count one is finite and keeps its real value. -/
private theorem divNat_one_of_isFinite {g : Value} (hformat : format.isIEEE = true)
    (hg : ExecFloat.Binary.isFinite g = true) :
    ExecFloat.Binary.isFinite (Context.divNat g 1) = true ∧
      ExecFloat.Binary.toReal (Context.divNat g 1) = ExecFloat.Binary.toReal g := by
  refine ⟨isFinite_divNat hformat (by norm_num) hg, ?_⟩
  rw [toReal_divNat hformat (by norm_num) hg, Nat.cast_one, div_one]
  exact Model.roundAt_toReal_eq _ hg

/-! ## The log-softmax pullback of `-e_t` -/

/-- For every IEEE format and every length: if the log-probabilities `y` are finite and
nonpositive, `t` is the pivot the pullback chooses and `y t < 0`, then with upstream `-e_t` the
pullback is finite and negative at `t` and finite and nonnegative elsewhere. -/
theorem pullback_sign_of_pivot [Storage Value] {n : Nat}
    (y : Tensor Value [n + 1]) (t : Fin (n + 1)) (hformat : format.isIEEE = true)
    (hfinite : ∀ i, ExecFloat.Binary.isFinite (y.getScalar i) = true)
    (hnonpos : ∀ i, ExecFloat.Binary.toReal (y.getScalar i) ≤ 0)
    (hpivot : Softmax.pivot y = t)
    (htarget : ExecFloat.Binary.toReal (y.getScalar t) < 0) :
    let gradient := LogSoftmax.vecBackward y
      (Tensor.map (fun x => -x) (Tensor.ofFn fun i => if i = t then 1 else 0))
    (ExecFloat.Binary.isFinite (gradient.getScalar t) = true ∧
        ExecFloat.Binary.toReal (gradient.getScalar t) < 0) ∧
      ∀ i, i ≠ t → ExecFloat.Binary.isFinite (gradient.getScalar i) = true ∧
        0 ≤ ExecFloat.Binary.toReal (gradient.getScalar i) := by
  roundoff

/-- For every IEEE format and every length: if the log-probabilities `y` are finite and
nonpositive, `t` is not the pivot the pullback chooses and the computed `exp y_t` is below one,
then with upstream `-e_t` the pullback is finite and negative at `t` and finite and nonnegative
elsewhere. -/
theorem pullback_sign_of_exp_lt_one [Storage Value] {n : Nat}
    (y : Tensor Value [n + 1]) (t : Fin (n + 1)) (hformat : format.isIEEE = true)
    (hfinite : ∀ i, ExecFloat.Binary.isFinite (y.getScalar i) = true)
    (hnonpos : ∀ i, ExecFloat.Binary.toReal (y.getScalar i) ≤ 0)
    (hpivot : Softmax.pivot y ≠ t)
    (hexp : ExecFloat.Binary.toReal
      (MathFunctions.exp (self := Context.toMathFunctions) (y.getScalar t)) < 1) :
    let gradient := LogSoftmax.vecBackward y
      (Tensor.map (fun x => -x) (Tensor.ofFn fun i => if i = t then 1 else 0))
    (ExecFloat.Binary.isFinite (gradient.getScalar t) = true ∧
        ExecFloat.Binary.toReal (gradient.getScalar t) < 0) ∧
      ∀ i, i ≠ t → ExecFloat.Binary.isFinite (gradient.getScalar i) = true ∧
        0 ≤ ExecFloat.Binary.toReal (gradient.getScalar i) := by
  roundoff

/-! ## The cross-entropy gradient -/

/-- The summed cross-entropy gradient of a one-hot target `t`, for finite logits with two or more
classes in an IEEE format: if the forward computation is finite, `t` is the pivot of the pullback
or the pullback's computed `exp` at `t` is below one, and, whenever the computed tail is zero, the
computed exponential `others` of the log-sum-exp of the other classes is finite and positive and
every entry `(logits i - logits pivot) - log1p others` of the two-level result is finite, the
gradient is finite and negative at `t` and finite and nonnegative elsewhere. -/
theorem gradient_sign [Storage Value] {n : Nat}
    (logits : Tensor Value [n + 1]) (t : Fin (n + 1)) (hformat : format.isIEEE = true)
    (hfinite : ∀ i, ExecFloat.Binary.isFinite (logits.getScalar i) = true) (hn : n ≠ 0) :
    let pivot : Fin (n + 1) := Softmax.pivot logits
    let runner := CrossEntropy.runnerUp logits pivot
    let shift := logits - Tensor.full _ (logits.getScalar pivot)
    let tail : Tensor Value [n + 1] := Tensor.ofFn fun j => if j = pivot then 0
      else MathFunctions.exp (shift.getScalar j)
    let inner : Tensor Value [n + 1] := Tensor.ofFn fun j =>
      if j = pivot ∨ j = runner then 0
      else MathFunctions.exp (logits.getScalar j - logits.getScalar runner)
    let others := MathFunctions.exp (shift.getScalar runner + Context.log1p (Tensor.sum inner))
    let twoLevel := shift - Tensor.full _ (Context.log1p others)
    let logp := CrossEntropy.logProbs logits ⟨0, Nat.one_pos⟩
    let gradient := Tensor.crossEntropyBackward .logits logits
      (Tensor.ofFn fun i => if i = t then 1 else 0) ⟨0, by exact Nat.zero_lt_succ _⟩ false
    (∀ i, ExecFloat.Binary.isFinite (shift.getScalar i) = true) →
      ExecFloat.Binary.isFinite (Tensor.sum tail) = true →
      (∀ i, ExecFloat.Binary.isFinite (logp.getScalar i) = true) →
      (Softmax.pivot logp = t ∨ ExecFloat.Binary.toReal
        (MathFunctions.exp (self := Context.toMathFunctions) (logp.getScalar t)) < 1) →
      ((Tensor.sum tail == 0) = true →
        ExecFloat.Binary.isFinite others = true ∧ 0 < ExecFloat.Binary.toReal others ∧
          ∀ i, ExecFloat.Binary.isFinite (twoLevel.getScalar i) = true) →
      (ExecFloat.Binary.isFinite (gradient.getScalar t) = true ∧
          ExecFloat.Binary.toReal (gradient.getScalar t) < 0) ∧
        ∀ i, i ≠ t → ExecFloat.Binary.isFinite (gradient.getScalar i) = true ∧
          0 ≤ ExecFloat.Binary.toReal (gradient.getScalar i) := by
  intro pivot runner shift tail inner others twoLevel logp gradient hshift hsum hlogp hbackward
    htwo
  have hneg (i : Fin (n + 1)) :=
    Roundoff.CrossEntropy.vectorLogProbs_neg logits hformat hfinite i hn (hshift i) hsum
      (hlogp i) htwo
  by_cases hpivot : Softmax.pivot logp = t
  · exact pullback_sign_of_pivot logp t hformat hlogp (fun i => (hneg i).le) hpivot (hneg t)
  · exact pullback_sign_of_exp_lt_one logp t hformat hlogp (fun i => (hneg i).le) hpivot
      (hbackward.resolve_left hpivot)

/-! ## Mean reduction -/

/-- The mean cross-entropy gradient of a one-hot target `t` that is the pivot of the pullback,
under the other conditions of `gradient_sign`: the division by a single class slice is exact, so
the gradient is again finite and negative at `t` and finite and nonnegative elsewhere. -/
theorem mean_sign_of_pivot [Storage Value] {n : Nat}
    (logits : Tensor Value [n + 1]) (t : Fin (n + 1)) (hformat : format.isIEEE = true)
    (hfinite : ∀ i, ExecFloat.Binary.isFinite (logits.getScalar i) = true) (hn : n ≠ 0) :
    let pivot : Fin (n + 1) := Softmax.pivot logits
    let runner := CrossEntropy.runnerUp logits pivot
    let shift := logits - Tensor.full _ (logits.getScalar pivot)
    let tail : Tensor Value [n + 1] := Tensor.ofFn fun j => if j = pivot then 0
      else MathFunctions.exp (shift.getScalar j)
    let inner : Tensor Value [n + 1] := Tensor.ofFn fun j =>
      if j = pivot ∨ j = runner then 0
      else MathFunctions.exp (logits.getScalar j - logits.getScalar runner)
    let others := MathFunctions.exp (shift.getScalar runner + Context.log1p (Tensor.sum inner))
    let twoLevel := shift - Tensor.full _ (Context.log1p others)
    let logp := CrossEntropy.logProbs logits ⟨0, Nat.one_pos⟩
    let gradient := Tensor.crossEntropyBackward .logits logits
      (Tensor.ofFn fun i => if i = t then 1 else 0) ⟨0, by exact Nat.zero_lt_succ _⟩
    (∀ i, ExecFloat.Binary.isFinite (shift.getScalar i) = true) →
      ExecFloat.Binary.isFinite (Tensor.sum tail) = true →
      (∀ i, ExecFloat.Binary.isFinite (logp.getScalar i) = true) →
      Softmax.pivot logp = t →
      ((Tensor.sum tail == 0) = true →
        ExecFloat.Binary.isFinite others = true ∧ 0 < ExecFloat.Binary.toReal others ∧
          ∀ i, ExecFloat.Binary.isFinite (twoLevel.getScalar i) = true) →
      (ExecFloat.Binary.isFinite (gradient.getScalar t) = true ∧
          ExecFloat.Binary.toReal (gradient.getScalar t) < 0) ∧
        ∀ i, i ≠ t → ExecFloat.Binary.isFinite (gradient.getScalar i) = true ∧
          0 ≤ ExecFloat.Binary.toReal (gradient.getScalar i) := by
  intro pivot runner shift tail inner others twoLevel logp gradient hshift hsum hlogp hbackward
    htwo
  obtain ⟨⟨hfiniteTarget, htarget⟩, hothers⟩ := gradient_sign logits t hformat hfinite hn hshift
    hsum hlogp (.inl hbackward) htwo
  have hentry (i : Fin (n + 1)) : gradient.getScalar i =
      Context.divNat ((Tensor.crossEntropyBackward .logits logits
        (Tensor.ofFn fun i => if i = t then 1 else 0)
        ⟨0, by exact Nat.zero_lt_succ _⟩ false).getScalar i)
        (Tensor.axisMeanDenom [n + 1] ⟨0, Shape.AxisInBounds.proof⟩) := by
    simp only [gradient, Tensor.crossEntropyBackward, Bool.false_eq_true, ↓reduceIte,
      Tensor.getScalar_map]
  have hcount : Tensor.axisMeanDenom [n + 1] ⟨0, Shape.AxisInBounds.proof⟩ = 1 := rfl
  refine ⟨?_, fun i hi => ?_⟩
  · obtain ⟨hvalue, hreal⟩ := divNat_one_of_isFinite hformat hfiniteTarget
    rw [hentry, hcount]
    exact ⟨hvalue, hreal ▸ htarget⟩
  · obtain ⟨hfiniteOther, hother⟩ := hothers i hi
    obtain ⟨hvalue, hreal⟩ := divNat_one_of_isFinite hformat hfiniteOther
    rw [hentry, hcount]
    exact ⟨hvalue, hreal ▸ hother⟩

/-- Dividing a vector that is finite and negative at `t` and finite and nonnegative elsewhere
entrywise by a count `1 ≤ m` gives a vector that is finite and nonpositive at `t` and finite and
nonnegative elsewhere. It is negative at `t` when `η < |g_t| / m`. -/
theorem divNat_sign [Storage Value] {n : Nat} (gradient : Tensor Value [n]) (t : Fin n) {m : ℕ}
    (hformat : format.isIEEE = true) (hm : 1 ≤ m)
    (hgradient : (ExecFloat.Binary.isFinite (gradient.getScalar t) = true ∧
        ExecFloat.Binary.toReal (gradient.getScalar t) < 0) ∧
      ∀ i, i ≠ t → ExecFloat.Binary.isFinite (gradient.getScalar i) = true ∧
        0 ≤ ExecFloat.Binary.toReal (gradient.getScalar i)) :
    ((ExecFloat.Binary.isFinite ((Tensor.map (Context.divNat · m) gradient).getScalar t) = true ∧
        ExecFloat.Binary.toReal ((Tensor.map (Context.divNat · m) gradient).getScalar t) ≤ 0) ∧
      ∀ i, i ≠ t →
        ExecFloat.Binary.isFinite ((Tensor.map (Context.divNat · m) gradient).getScalar i) = true ∧
          0 ≤ ExecFloat.Binary.toReal ((Tensor.map (Context.divNat · m) gradient).getScalar i)) ∧
      (Model.underflowErrorAt format < |ExecFloat.Binary.toReal (gradient.getScalar t)| / m →
        ExecFloat.Binary.toReal ((Tensor.map (Context.divNat · m) gradient).getScalar t) < 0) := by
  obtain ⟨⟨hfiniteTarget, htarget⟩, hothers⟩ := hgradient
  have hpos : (0 : ℝ) < m := by exact_mod_cast hm
  simp only [Tensor.getScalar_map]
  refine ⟨⟨⟨isFinite_divNat hformat hpos.ne' hfiniteTarget, ?_⟩, fun i hi => ?_⟩, fun hlarge => ?_⟩
  · rw [toReal_divNat hformat hpos.ne' hfiniteTarget]
    exact Model.roundAt_nonpos format (div_nonpos_of_nonpos_of_nonneg htarget.le hpos.le)
  · obtain ⟨hfiniteOther, hother⟩ := hothers i hi
    refine ⟨isFinite_divNat hformat hpos.ne' hfiniteOther, ?_⟩
    rw [toReal_divNat hformat hpos.ne' hfiniteOther]
    exact Model.roundAt_nonneg format (div_nonneg hother hpos.le)
  · rw [toReal_divNat hformat hpos.ne' hfiniteTarget]
    refine Model.roundAt_lt_zero format ?_
    rw [abs_of_neg htarget, neg_div] at hlarge
    linarith

/-- Row `b` of the mean cross-entropy gradient of a batch with one-hot labels, under the
conditions of `gradient_sign` on the row: it is finite and nonpositive at the label and finite and
nonnegative elsewhere. For the slice count `m` it is negative at the label when `η < |g| / m`, for
the row's summed gradient `g` at the label. -/
theorem batch_sign [Storage Value] {batch n : Nat}
    (logits : Tensor Value [batch, n + 1]) (labels : Fin batch → Fin (n + 1)) (b : Fin batch)
    (hformat : format.isIEEE = true)
    (hfinite : ∀ i, ExecFloat.Binary.isFinite ((Tensor.unstack logits b).getScalar i) = true)
    (hn : n ≠ 0) :
    let row := Tensor.unstack logits b
    let pivot : Fin (n + 1) := Softmax.pivot row
    let runner := CrossEntropy.runnerUp row pivot
    let shift := row - Tensor.full _ (row.getScalar pivot)
    let tail : Tensor Value [n + 1] := Tensor.ofFn fun j => if j = pivot then 0
      else MathFunctions.exp (shift.getScalar j)
    let inner : Tensor Value [n + 1] := Tensor.ofFn fun j =>
      if j = pivot ∨ j = runner then 0
      else MathFunctions.exp (row.getScalar j - row.getScalar runner)
    let others := MathFunctions.exp (shift.getScalar runner + Context.log1p (Tensor.sum inner))
    let twoLevel := shift - Tensor.full _ (Context.log1p others)
    let logp := CrossEntropy.logProbs row ⟨0, Nat.one_pos⟩
    let summed := Tensor.crossEntropyBackward .logits row
      (Tensor.ofFn fun i => if i = labels b then 1 else 0) ⟨0, by exact Nat.zero_lt_succ _⟩ false
    let gradient := Tensor.crossEntropyBackward .logits logits
      (Tensor.dim fun b => Tensor.ofFn fun i => if i = labels b then 1 else 0)
      ⟨1, by exact Nat.one_lt_succ_succ _⟩
    (∀ i, ExecFloat.Binary.isFinite (shift.getScalar i) = true) →
      ExecFloat.Binary.isFinite (Tensor.sum tail) = true →
      (∀ i, ExecFloat.Binary.isFinite (logp.getScalar i) = true) →
      (Softmax.pivot logp = labels b ∨ ExecFloat.Binary.toReal
        (MathFunctions.exp (self := Context.toMathFunctions) (logp.getScalar (labels b))) < 1) →
      ((Tensor.sum tail == 0) = true →
        ExecFloat.Binary.isFinite others = true ∧ 0 < ExecFloat.Binary.toReal others ∧
          ∀ i, ExecFloat.Binary.isFinite (twoLevel.getScalar i) = true) →
      ((ExecFloat.Binary.isFinite ((Tensor.unstack gradient b).getScalar (labels b)) = true ∧
          ExecFloat.Binary.toReal ((Tensor.unstack gradient b).getScalar (labels b)) ≤ 0) ∧
          ∀ i, i ≠ labels b →
            ExecFloat.Binary.isFinite ((Tensor.unstack gradient b).getScalar i) = true ∧
              0 ≤ ExecFloat.Binary.toReal ((Tensor.unstack gradient b).getScalar i)) ∧
        (Model.underflowErrorAt format < |ExecFloat.Binary.toReal (summed.getScalar (labels b))| /
            Tensor.axisMeanDenom [batch, n + 1] ⟨1, Shape.AxisInBounds.proof⟩ →
          ExecFloat.Binary.toReal ((Tensor.unstack gradient b).getScalar (labels b)) < 0) := by
  intro row pivot runner shift tail inner others twoLevel logp summed gradient hshift hsum hlogp
    hbackward htwo
  have hsign := divNat_sign summed (labels b)
    (m := Tensor.axisMeanDenom [batch, n + 1] ⟨1, Shape.AxisInBounds.proof⟩) hformat
    (Tensor.axisMeanDenom_pos _ _)
    (gradient_sign row (labels b) hformat hfinite hn hshift hsum hlogp hbackward htwo)
  simp only [gradient, unstack_crossEntropyBackward, Tensor.unstack_dim]
  exact hsign

end General

/-! ## Instances

At the logits `[9, -9, -9]` of Barron, Gnanapragasam and Tschannen, with target `0`, the
conditions of `mean_sign_of_pivot` hold in binary32, bfloat16 and binary16, and those of
`batch_sign` hold for two such rows in bfloat16. The kernel decides them. In binary32 and
bfloat16 the computed tail is positive and the one-level branch is taken. In binary16 both
computed exponentials `exp (-18)` of the tail are zero, the two-level branch is taken, and the
computed exponential of the log-sum-exp of the other classes is `2⁻²⁴`. -/


/-- The binary32 mean cross-entropy gradient at `[9, -9, -9]` is finite and negative at the
target `0` and finite and nonnegative elsewhere. -/
theorem gradient_sign_binary32 :
    let gradient := Tensor.crossEntropyBackward .logits (SoftmaxCollapse.logits (9 : Binary32))
      SoftmaxCrossEntropy.target ⟨0, by exact Nat.zero_lt_succ _⟩
    (ExecFloat.Binary.isFinite (gradient.getScalar 0) = true ∧
        ExecFloat.Binary.toReal (gradient.getScalar 0) < 0) ∧
      ∀ i, i ≠ 0 → ExecFloat.Binary.isFinite (gradient.getScalar i) = true ∧
        0 ≤ ExecFloat.Binary.toReal (gradient.getScalar i) :=
  mean_sign_of_pivot _ 0 (by decide) (by decide +kernel) (by decide) (by decide +kernel)
    (by decide +kernel) (by decide +kernel) (by decide +kernel)
    (fun h => absurd h (by decide +kernel))

/-- The bfloat16 mean cross-entropy gradient at `[9, -9, -9]` is finite and negative at the
target `0` and finite and nonnegative elsewhere. -/
theorem gradient_sign_bfloat16 :
    let gradient := Tensor.crossEntropyBackward .logits (SoftmaxCollapse.logits (9 : BFloat16))
      SoftmaxCrossEntropy.target ⟨0, by exact Nat.zero_lt_succ _⟩
    (ExecFloat.Binary.isFinite (gradient.getScalar 0) = true ∧
        ExecFloat.Binary.toReal (gradient.getScalar 0) < 0) ∧
      ∀ i, i ≠ 0 → ExecFloat.Binary.isFinite (gradient.getScalar i) = true ∧
        0 ≤ ExecFloat.Binary.toReal (gradient.getScalar i) :=
  mean_sign_of_pivot _ 0 (by decide) (by decide +kernel) (by decide) (by decide +kernel)
    (by decide +kernel) (by decide +kernel) (by decide +kernel)
    (fun h => absurd h (by decide +kernel))

/-- The binary16 mean cross-entropy gradient at `[9, -9, -9]` is finite and negative at the
target `0` and finite and nonnegative elsewhere. The computed tail is zero there, and the
two-level branch is taken. -/
theorem gradient_sign_binary16 :
    let gradient := Tensor.crossEntropyBackward .logits (SoftmaxCollapse.logits (9 : Binary16))
      SoftmaxCrossEntropy.target ⟨0, by exact Nat.zero_lt_succ _⟩
    (ExecFloat.Binary.isFinite (gradient.getScalar 0) = true ∧
        ExecFloat.Binary.toReal (gradient.getScalar 0) < 0) ∧
      ∀ i, i ≠ 0 → ExecFloat.Binary.isFinite (gradient.getScalar i) = true ∧
        0 ≤ ExecFloat.Binary.toReal (gradient.getScalar i) :=
  mean_sign_of_pivot _ 0 (by decide) (by decide +kernel) (by decide) (by decide +kernel)
    (by decide +kernel) (by decide +kernel) (by decide +kernel)
    (fun _ => ⟨by decide +kernel, by
      rw [ExecFloat.Binary.toReal_of_toRat?_eq_some (q := 1 / 2 ^ 24) (by decide +kernel)]
      norm_num, by decide +kernel⟩)

/-- In binary16 the target entry of the mean cross-entropy gradient at `[9, -9, -9]` decodes to
`-2⁻²⁴`. The log-softmax VJP of the negated one-hot target at the one-level log-softmax
`Tensor.logSoftmax`, whose computed tail is zero there, decodes to zero. -/
theorem binary16_toRat? :
    ExecFloat.Binary.toRat? ((Tensor.crossEntropyBackward .logits
        (SoftmaxCollapse.logits (9 : Binary16)) SoftmaxCrossEntropy.target
        ⟨0, by exact Nat.zero_lt_succ _⟩).getScalar 0) = some (-1 / 2 ^ 24) ∧
      ExecFloat.Binary.toRat? ((Tensor.logSoftmaxBackward
        (Tensor.logSoftmax (SoftmaxCollapse.logits (9 : Binary16)) ⟨0, Nat.one_pos⟩)
        (Tensor.map (fun x => -x) SoftmaxCrossEntropy.target) ⟨0, Nat.one_pos⟩).getScalar 0) =
        some 0 := by
  decide +kernel

/-- In the IEEE format with exponent width 3 and fraction width 5, whose largest finite value is
`15.75`, at the logits `[0, -6, …, -6]` of 18 classes the computed tail is zero and the computed
sum of the sixteen exponentials `exp 0` of the other classes overflows, so the two-level test
fails and the log-probabilities are the one-level `LogSoftmax.vec`. -/
example :
    let logits : Tensor (ExecFloat.Binary 3 5) [18] := Tensor.ofFn fun i => if i = 0 then 0 else -6
    CrossEntropy.vectorLogProbs logits = LogSoftmax.vec logits := by
  have h : CrossEntropy.twoLevelAt
      (Tensor.ofFn fun i => if i = 0 then 0 else -6 : Tensor (ExecFloat.Binary 3 5) [18]) =
        false := by
    decide +kernel
  simp only [CrossEntropy.vectorLogProbs, h, Bool.false_eq_true, ↓reduceIte]

/-- `gradient_sign` for bfloat16 logits over a vocabulary of 50257 classes, where the length times
the unit roundoff is about 196. -/
theorem gradient_sign_vocabulary (logits : Tensor BFloat16 [50257])
    (t : Fin 50257) (hfinite : ∀ i, ExecFloat.Binary.isFinite (logits.getScalar i) = true) :
    let pivot : Fin 50257 := Softmax.pivot logits
    let runner := CrossEntropy.runnerUp logits pivot
    let shift := logits - Tensor.full _ (logits.getScalar pivot)
    let tail : Tensor BFloat16 [50257] := Tensor.ofFn fun j => if j = pivot then 0
      else MathFunctions.exp (shift.getScalar j)
    let inner : Tensor BFloat16 [50257] := Tensor.ofFn fun j =>
      if j = pivot ∨ j = runner then 0
      else MathFunctions.exp (logits.getScalar j - logits.getScalar runner)
    let others := MathFunctions.exp (shift.getScalar runner + Context.log1p (Tensor.sum inner))
    let twoLevel := shift - Tensor.full _ (Context.log1p others)
    let logp := CrossEntropy.logProbs logits ⟨0, Nat.one_pos⟩
    let gradient := Tensor.crossEntropyBackward .logits logits
      (Tensor.ofFn fun i => if i = t then 1 else 0) ⟨0, by exact Nat.zero_lt_succ _⟩ false
    (∀ i, ExecFloat.Binary.isFinite (shift.getScalar i) = true) →
      ExecFloat.Binary.isFinite (Tensor.sum tail) = true →
      (∀ i, ExecFloat.Binary.isFinite (logp.getScalar i) = true) →
      (Softmax.pivot logp = t ∨ ExecFloat.Binary.toReal
        (MathFunctions.exp (self := Context.toMathFunctions) (logp.getScalar t)) < 1) →
      ((Tensor.sum tail == 0) = true →
        ExecFloat.Binary.isFinite others = true ∧ 0 < ExecFloat.Binary.toReal others ∧
          ∀ i, ExecFloat.Binary.isFinite (twoLevel.getScalar i) = true) →
      (ExecFloat.Binary.isFinite (gradient.getScalar t) = true ∧
          ExecFloat.Binary.toReal (gradient.getScalar t) < 0) ∧
        ∀ i, i ≠ t → ExecFloat.Binary.isFinite (gradient.getScalar i) = true ∧
          0 ≤ ExecFloat.Binary.toReal (gradient.getScalar i) :=
  gradient_sign logits t (by decide) hfinite (by decide)

/-- Row `0` of the bfloat16 mean cross-entropy gradient of two rows `[9, -9, -9]` with labels `0`
is finite and negative at the label and finite and nonnegative elsewhere. Each entry is divided by
the slice count `2`. -/
theorem batch_sign_bfloat16 :
    let gradient := Tensor.unstack (Tensor.crossEntropyBackward .logits
      (Tensor.dim fun _ => SoftmaxCollapse.logits (9 : BFloat16) : Tensor BFloat16 [2, 3])
      (Tensor.dim fun _ => Tensor.ofFn fun i => if i = 0 then 1 else 0)
      ⟨1, by exact Nat.one_lt_succ_succ _⟩) 0
    (ExecFloat.Binary.isFinite (gradient.getScalar 0) = true ∧
        ExecFloat.Binary.toReal (gradient.getScalar 0) < 0) ∧
      ∀ i, i ≠ 0 → ExecFloat.Binary.isFinite (gradient.getScalar i) = true ∧
        0 ≤ ExecFloat.Binary.toReal (gradient.getScalar i) := by
  obtain ⟨⟨⟨hfinite, -⟩, hothers⟩, hstrict⟩ := batch_sign
    (Tensor.dim fun _ => SoftmaxCollapse.logits 9 : Tensor BFloat16 [2, 3]) (fun _ => 0) 0
    (by decide) (by decide +kernel) (by decide) (by decide +kernel) (by decide +kernel)
    (by decide +kernel) (.inl (by decide +kernel)) (fun h => absurd h (by decide +kernel))
  refine ⟨⟨hfinite, hstrict ?_⟩, hothers⟩
  have hcount : Tensor.axisMeanDenom [2, 3] ⟨1, Shape.AxisInBounds.proof⟩ = 2 := rfl
  rw [hcount, ExecFloat.Binary.toReal_of_toRat?_eq_some (q := -131 / 2 ^ 32) (by decide +kernel)]
  refine (Model.underflowErrorAt_le_zpow _ (e := -100) (by decide)).trans_lt ?_
  norm_num

/-! ## The standard formulas -/

/-- The standard formulas give the binary32 target gradient at `[9, -9, -9]` a finite value that
is not negative: it decodes to zero. -/
theorem standard_not_neg_binary32 :
    ExecFloat.Binary.isFinite (SoftmaxCollapse.standardGradient (9 : Binary32)) = true ∧
      ¬ ExecFloat.Binary.toReal (SoftmaxCollapse.standardGradient (9 : Binary32)) < 0 := by
  have hzero := SoftmaxCollapse.binary32_standard_toRat?.2.2
  refine ⟨Model.Algebraic.isFinite_of_toRat?_eq_some hzero, ?_⟩
  rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hzero]
  simp

end SoftmaxSign

/-!
## Cancellation in the cosine loss and the Gini impurity, checked by the kernel

For unit vectors `p̂` and `t̂` at a small angle `θ`, the cosine loss `1 - p̂ · t̂` is about
`θ² / 2`, while the rounded dot product is within about the unit roundoff `u` of one.
Subtracting it from one keeps an absolute error of about `u`, and no correct digit once `θ²` is
below `u`. Since a relative perturbation `u` of the inputs moves the angle by about `u`, the
inputs determine the loss `L` only to about `u θ = u √(2 L)`. When both norms are positive and
above their floors, `cosineLoss` computes `‖p̂ - t̂‖² / 2`, which equals `1 - p̂ · t̂` over the
reals (`cosineLoss_real`). `Direct` keeps the formula `1 - p̂ · t̂`.

The inputs below are four-vectors of natural numbers that each format represents exactly, at its
precision, whose last entries differ. In bfloat16 the exact loss is about `2.1e-3`, in binary32
about `9.2e-9` and in binary64 about `1.9e-33`. The kernel evaluates the direct loss to zero in
the first two formats and to `2⁻⁵²` in binary64, which an absolute tolerance of `1e-15` accepts.
The loss of `cosineLoss` decodes to `2.17e-3`, `9.19e-9` and `6.2e-33`. Each lies within
`u √(2 L)` of the exact loss of its format, and no direct value does.

The Gini impurity `1 - Σ p²` of `n` labels of which one differs from the others is
`2 (n - 1) / n²`. The direct formula subtracts the sum of the rounded squared frequencies from
one. `DecisionTree.Classifier.gini` computes the impurity exactly from the counts, as
a rational, and converts it with one rounding in binary32 (`gini_real` shows that over the reals
it is the direct formula). At `n = 300` the direct binary32 impurity is more than 32 units of
roundoff off, and that of `gini` less than one.
-/


namespace Cancellation

open TorchLean.Tensor
open FloatLib.Floats FloatLib.Numerics
open FloatLib.Floats.Formats.BinaryInterchange

section Formulas

variable {α : Type} [Storage α] [Context α]

/-! ## Direct formulas -/

namespace Direct

/-- Direct cosine loss: one minus the dot product of the normalized vectors. -/
def cosineLoss {s : Shape} (predicted : Tensor α s) (target : Tensor α s)
    (epsilon : α := Context.defaultEpsilon) : α :=
  let (predScale, predNorm) := CosineLoss.normFactors predicted epsilon
  let (targetScale, targetNorm) := CosineLoss.normFactors target epsilon
  let normalizedPred := Tensor.map (fun value => (value / predScale) / predNorm) predicted
  let normalizedTarget := Tensor.map (fun value => (value / targetScale) / targetNorm) target
  (1 : α) - Tensor.sum (normalizedPred * normalizedTarget)

/-- Direct Gini impurity: one minus the sum of the rounded squared class frequencies. -/
def gini {β : Type} [DecidableEq β] (ys : Array β) : α :=
  if ys.isEmpty then
    0
  else
    let n : α := (ys.size : α)
    let labels := ys.toList.eraseDups
    let sumSq :=
      labels.foldl (fun acc lbl =>
        let c : α := (ys.count lbl : Nat)
        let p := c / n
        acc + (p * p)
      ) 0
    (1 : α) - sumSq

end Direct

/-! ## Inputs -/

/-- A four-vector of natural numbers cast into the scalar type. -/
def vector (entries : List ℕ) : Tensor α [4] :=
  Tensor.ofFn fun i => ((entries.getD i.1 0 : ℕ) : α)

end Formulas

/-- `n - 1` labels `0` and one label `1`. -/
def labels (n : ℕ) : Array ℕ := (Array.replicate (n - 1) 0).push 1

/-- One minus the cosine of the angle between two four-vectors of natural numbers. -/
noncomputable def oneSubCos (p t : List ℕ) : ℝ :=
  1 - (∑ i : Fin 4, (p.getD i 0 : ℝ) * t.getD i 0) /
    (Real.sqrt (∑ i : Fin 4, (p.getD i 0 : ℝ) ^ 2) * Real.sqrt (∑ i : Fin 4, (t.getD i 0 : ℝ) ^ 2))

/-! ## The same real values -/

/-- Over the reals the cosine loss over every axis is the direct one. -/
theorem cosineLoss_real {s : Shape} (p t : Tensor ℝ s) (keepdim : Bool) (epsilon : ℝ)
    (o : (s.reduce (List.finRange s.rank) keepdim).Coord) :
    cosineLoss p t (List.finRange s.rank) keepdim epsilon o = Direct.cosineLoss p t epsilon :=
  Tensor.cosineLoss_eq_one_sub_dot p t keepdim epsilon o

/-- Over the reals the Gini impurity is the direct one. -/
theorem gini_real {β : Type} [DecidableEq β] (ys : Array β) :
    (DecisionTree.Classifier.gini ys : ℝ) = Direct.gini ys := by
  unfold DecisionTree.Classifier.gini Direct.gini
  split_ifs with hempty
  · rfl
  · have hn : (ys.size : ℝ) ≠ 0 := by
      exact_mod_cast fun h => hempty (Array.isEmpty_iff_size_eq_zero.mpr h)
    dsimp only
    have hsquares : ys.toList.eraseDups.foldl (fun total label => total + ys.count label ^ 2)
        0 = (ys.toList.eraseDups.map fun label => ys.count label ^ 2).sum := by
      rw [List.sum_eq_foldl, List.foldl_map]
    have hfrequencies : ys.toList.eraseDups.foldl (fun acc lbl =>
          acc + (ys.count lbl : ℝ) / ys.size * ((ys.count lbl : ℝ) / ys.size)) 0 =
        (ys.toList.eraseDups.map fun lbl => (ys.count lbl : ℝ) ^ 2).sum *
          (1 / (ys.size : ℝ) ^ 2) := by
      rw [← List.sum_map_mul_right, List.sum_eq_foldl, List.foldl_map]
      congr 1
      funext acc lbl
      ring
    rw [hsquares, hfrequencies]
    push_cast
    field_simp
    simp only [List.map_map, Function.comp_def]
    push_cast
    rfl

/-! ## Accuracy -/

/-- A value within `u √(2 lo)` of both ends of an interval `[lo, hi]` of nonnegative reals is
within `u √(2 L)` of every `L` in it. -/
private theorem abs_sub_le_mul_sqrt {c L lo hi u : ℝ} (hL : L ∈ Set.Icc lo hi) (hlo₀ : 0 ≤ lo)
    (hu : 0 ≤ u) (hlower : (c - lo) ^ 2 ≤ 2 * u ^ 2 * lo)
    (hupper : (hi - c) ^ 2 ≤ 2 * u ^ 2 * lo) :
    |c - L| ≤ u * Real.sqrt (2 * L) := by
  obtain ⟨hloL, hLhi⟩ := hL
  have hscaled : 2 * u ^ 2 * lo ≤ 2 * u ^ 2 * L := by nlinarith [sq_nonneg u]
  have hsquare : (c - L) ^ 2 ≤ (u * Real.sqrt (2 * L)) ^ 2 := by
    rw [mul_pow, Real.sq_sqrt (by linarith)]
    rcases le_total c L with h | h
    · nlinarith
    · nlinarith
  calc |c - L| = Real.sqrt ((c - L) ^ 2) := (Real.sqrt_sq_eq_abs _).symm
    _ ≤ Real.sqrt ((u * Real.sqrt (2 * L)) ^ 2) := Real.sqrt_le_sqrt hsquare
    _ = u * Real.sqrt (2 * L) := Real.sqrt_sq (by positivity)

/-- A value at distance at least `d` from `L ∈ [0, hi]` is not within `u √(2 L)` of it when
`2 u² hi < d²`. -/
private theorem mul_sqrt_lt_abs_sub {c L hi u d : ℝ} (hL₀ : 0 ≤ L) (hLhi : L ≤ hi) (hd : 0 ≤ d)
    (hdistance : d ≤ |c - L|) (hfar : 2 * u ^ 2 * hi < d ^ 2) :
    u * Real.sqrt (2 * L) < |c - L| := by
  have hsquare : (u * Real.sqrt (2 * L)) ^ 2 < d ^ 2 := by
    rw [mul_pow, Real.sq_sqrt (by linarith)]
    nlinarith [sq_nonneg u]
  exact lt_of_lt_of_le (abs_lt_of_sq_lt_sq' hsquare hd).2 hdistance

/-- Lower and upper bounds on `1 - D / (√A √B)` from comparisons of squares. -/
private theorem oneSubCos_mem_Icc {D A B lo hi : ℝ} (hA : 0 < A) (hB : 0 < B) (hD : 0 ≤ D)
    (hhi : hi ≤ 1) (hlo : lo ≤ 1) (hlower : D ^ 2 ≤ (1 - lo) ^ 2 * (A * B))
    (hupper : (1 - hi) ^ 2 * (A * B) ≤ D ^ 2) :
    1 - D / (Real.sqrt A * Real.sqrt B) ∈ Set.Icc lo hi := by
  rw [← Real.sqrt_mul hA.le]
  have hS : 0 < Real.sqrt (A * B) := Real.sqrt_pos.2 (mul_pos hA hB)
  have hlower' : D ≤ (1 - lo) * Real.sqrt (A * B) := by
    have := Real.sqrt_le_sqrt hlower
    rwa [Real.sqrt_sq hD, Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq (by linarith)] at this
  have hupper' : (1 - hi) * Real.sqrt (A * B) ≤ D := by
    have := Real.sqrt_le_sqrt hupper
    rwa [Real.sqrt_sq hD, Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq (by linarith)] at this
  constructor
  · have := (div_le_iff₀ hS).2 hlower'
    linarith
  · have := (le_div_iff₀ hS).2 hupper'
    linarith

/-! ## Cosine loss in bfloat16

The target moves the last entry of the prediction by `32`. -/

/-- The bfloat16 prediction. -/
def bfloat16Prediction : List ℕ := [77, 128, 179, 230]

/-- The bfloat16 target. -/
def bfloat16Target : List ℕ := [77, 128, 179, 262]

/-- The exact bfloat16 loss. -/
noncomputable def bfloat16ExactLoss : ℝ := oneSubCos bfloat16Prediction bfloat16Target

/-- The direct bfloat16 loss. -/
def bfloat16DirectLoss : BFloat16 :=
  Direct.cosineLoss (vector bfloat16Prediction) (vector bfloat16Target)

/-- The bfloat16 loss of `cosineLoss` over the one axis. -/
def bfloat16StableLoss : BFloat16 :=
  cosineLoss (vector bfloat16Prediction) (vector bfloat16Target) (List.finRange _) false
    Context.defaultEpsilon ()

/-- The bfloat16 inputs decode to their natural numbers, so `bfloat16ExactLoss` is the exact loss
of the decoded inputs. -/
theorem bfloat16_vector_toRat? : ∀ i : Fin 4,
    ExecFloat.Binary.toRat? ((vector bfloat16Prediction : Tensor BFloat16 [4]).getScalar i) =
        some (bfloat16Prediction.getD i.1 0 : ℕ) ∧
      ExecFloat.Binary.toRat? ((vector bfloat16Target : Tensor BFloat16 [4]).getScalar i) =
        some (bfloat16Target.getD i.1 0 : ℕ) := by
  decide +kernel

/-- The exact loss lies between `2.111e-3` and `2.112e-3`. -/
theorem bfloat16ExactLoss_mem_Icc :
    bfloat16ExactLoss ∈ Set.Icc (2111 / 10 ^ 6) (2112 / 10 ^ 6) := by
  unfold bfloat16ExactLoss oneSubCos bfloat16Prediction bfloat16Target
  apply oneSubCos_mem_Icc <;> norm_num [Fin.sum_univ_four]

/-- The direct loss decodes to zero, which is not within `u √(2 L)` of the exact loss `L`. -/
theorem bfloat16_direct :
    ExecFloat.Binary.toRat? bfloat16DirectLoss = some 0 ∧
      Model.unitRoundoffAt FloatFormat.bfloat16 * Real.sqrt (2 * bfloat16ExactLoss) <
        |ExecFloat.Binary.toReal bfloat16DirectLoss - bfloat16ExactLoss| := by
  have hvalue : ExecFloat.Binary.toRat? bfloat16DirectLoss = some 0 := by
    decide +kernel
  obtain ⟨lo, hi⟩ := bfloat16ExactLoss_mem_Icc
  refine ⟨hvalue, mul_sqrt_lt_abs_sub (by linarith) hi (d := 2111 / 10 ^ 6) (by norm_num) ?_ ?_⟩
  · rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hvalue, Rat.cast_zero, zero_sub, abs_neg,
      abs_of_nonneg (by linarith)]
    exact lo
  · norm_num [Model.unitRoundoffAt_eq, FloatFormat.bfloat16]

/-- The loss of `cosineLoss` decodes to `71 / 2¹⁵`, within `u √(2 L)` of the exact
loss `L`. -/
theorem bfloat16_stable :
    ExecFloat.Binary.toRat? bfloat16StableLoss = some (71 / 2 ^ 15) ∧
      |ExecFloat.Binary.toReal bfloat16StableLoss - bfloat16ExactLoss| ≤
        Model.unitRoundoffAt FloatFormat.bfloat16 * Real.sqrt (2 * bfloat16ExactLoss) := by
  have hvalue : ExecFloat.Binary.toRat? bfloat16StableLoss = some (71 / 2 ^ 15) := by
    decide +kernel
  refine ⟨hvalue, ?_⟩
  rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hvalue]
  exact abs_sub_le_mul_sqrt bfloat16ExactLoss_mem_Icc (by norm_num)
    (Model.unitRoundoffAt_pos _).le
    (by norm_num [Model.unitRoundoffAt_eq, FloatFormat.bfloat16])
    (by norm_num [Model.unitRoundoffAt_eq, FloatFormat.bfloat16])

/-! ## Cosine loss in binary32

The bfloat16 inputs at binary32 precision: the target moves the last entry by `2¹²`. -/

/-- The binary32 prediction. -/
def binary32Prediction : List ℕ := [5033165, 8388608, 11744051, 15099494]

/-- The binary32 target. -/
def binary32Target : List ℕ := [5033165, 8388608, 11744051, 15103590]

/-- The exact binary32 loss. -/
noncomputable def binary32ExactLoss : ℝ := oneSubCos binary32Prediction binary32Target

/-- The direct binary32 loss. -/
def binary32DirectLoss : Binary32 :=
  Direct.cosineLoss (vector binary32Prediction) (vector binary32Target)

/-- The binary32 loss of `cosineLoss` over the one axis. -/
def binary32StableLoss : Binary32 :=
  cosineLoss (vector binary32Prediction) (vector binary32Target) (List.finRange _) false
    Context.defaultEpsilon ()

/-- The binary32 inputs decode to their natural numbers, so `binary32ExactLoss` is the exact loss
of the decoded inputs. -/
theorem binary32_vector_toRat? : ∀ i : Fin 4,
    ExecFloat.Binary.toRat? ((vector binary32Prediction : Tensor Binary32 [4]).getScalar i) =
        some (binary32Prediction.getD i.1 0 : ℕ) ∧
      ExecFloat.Binary.toRat? ((vector binary32Target : Tensor Binary32 [4]).getScalar i) =
        some (binary32Target.getD i.1 0 : ℕ) := by
  decide +kernel

/-- The exact loss lies between `9.1944e-9` and `9.1945e-9`. -/
theorem binary32ExactLoss_mem_Icc :
    binary32ExactLoss ∈ Set.Icc (91944 / 10 ^ 13) (91945 / 10 ^ 13) := by
  unfold binary32ExactLoss oneSubCos binary32Prediction binary32Target
  apply oneSubCos_mem_Icc <;> norm_num [Fin.sum_univ_four]

/-- The direct loss decodes to zero, which is not within `u √(2 L)` of the exact loss `L`. -/
theorem binary32_direct :
    ExecFloat.Binary.toRat? binary32DirectLoss = some 0 ∧
      Model.unitRoundoffAt FloatFormat.binary32 * Real.sqrt (2 * binary32ExactLoss) <
        |ExecFloat.Binary.toReal binary32DirectLoss - binary32ExactLoss| := by
  have hvalue : ExecFloat.Binary.toRat? binary32DirectLoss = some 0 := by
    decide +kernel
  obtain ⟨lo, hi⟩ := binary32ExactLoss_mem_Icc
  refine ⟨hvalue, mul_sqrt_lt_abs_sub (by linarith) hi (d := 91944 / 10 ^ 13) (by norm_num) ?_ ?_⟩
  · rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hvalue, Rat.cast_zero, zero_sub, abs_neg,
      abs_of_nonneg (by linarith)]
    exact lo
  · norm_num [Model.unitRoundoffAt_eq, FloatFormat.binary32]

/-- The loss of `cosineLoss` decodes to `2588045 / 2⁴⁸`, within `u √(2 L)` of the
exact loss `L`. -/
theorem binary32_stable :
    ExecFloat.Binary.toRat? binary32StableLoss = some (2588045 / 2 ^ 48) ∧
      |ExecFloat.Binary.toReal binary32StableLoss - binary32ExactLoss| ≤
        Model.unitRoundoffAt FloatFormat.binary32 * Real.sqrt (2 * binary32ExactLoss) := by
  have hvalue : ExecFloat.Binary.toRat? binary32StableLoss = some (2588045 / 2 ^ 48) := by
    decide +kernel
  refine ⟨hvalue, ?_⟩
  rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hvalue]
  exact abs_sub_le_mul_sqrt binary32ExactLoss_mem_Icc (by norm_num)
    (Model.unitRoundoffAt_pos _).le
    (by norm_num [Model.unitRoundoffAt_eq, FloatFormat.binary32])
    (by norm_num [Model.unitRoundoffAt_eq, FloatFormat.binary32])

/-! ## Cosine loss in binary64

The same inputs at binary64 precision: the target moves the last entry by `2`. -/

/-- The binary64 prediction. -/
def binary64Prediction : List ℕ :=
  [5404319552844595, 9007199254740992, 12610078956637388, 16212958658533786]

/-- The binary64 target. -/
def binary64Target : List ℕ :=
  [5404319552844595, 9007199254740992, 12610078956637388, 16212958658533788]

/-- The exact binary64 loss. -/
noncomputable def binary64ExactLoss : ℝ := oneSubCos binary64Prediction binary64Target

/-- The direct binary64 loss. -/
def binary64DirectLoss : Binary64 :=
  Direct.cosineLoss (vector binary64Prediction) (vector binary64Target)

/-- The binary64 loss of `cosineLoss` over the one axis. -/
def binary64StableLoss : Binary64 :=
  cosineLoss (vector binary64Prediction) (vector binary64Target) (List.finRange _) false
    Context.defaultEpsilon ()

/-- The binary64 inputs decode to their natural numbers, so `binary64ExactLoss` is the exact loss
of the decoded inputs. -/
theorem binary64_vector_toRat? : ∀ i : Fin 4,
    ExecFloat.Binary.toRat? ((vector binary64Prediction : Tensor Binary64 [4]).getScalar i) =
        some (binary64Prediction.getD i.1 0 : ℕ) ∧
      ExecFloat.Binary.toRat? ((vector binary64Target : Tensor Binary64 [4]).getScalar i) =
        some (binary64Target.getD i.1 0 : ℕ) := by
  decide +kernel

/-- The exact loss lies between `1.90e-33` and `1.91e-33`. -/
theorem binary64ExactLoss_mem_Icc :
    binary64ExactLoss ∈ Set.Icc (190 / 10 ^ 35) (191 / 10 ^ 35) := by
  unfold binary64ExactLoss oneSubCos binary64Prediction binary64Target
  apply oneSubCos_mem_Icc <;> norm_num [Fin.sum_univ_four]

/-- The direct loss decodes to `2⁻⁵²`. It is not within `u √(2 L)` of the exact loss `L`, more
than `10¹⁷` times larger, yet an absolute tolerance of `1e-15` accepts it. -/
theorem binary64_direct :
    ExecFloat.Binary.toRat? binary64DirectLoss = some (1 / 2 ^ 52) ∧
      Model.unitRoundoffAt FloatFormat.binary64 * Real.sqrt (2 * binary64ExactLoss) <
        |ExecFloat.Binary.toReal binary64DirectLoss - binary64ExactLoss| ∧
      |ExecFloat.Binary.toReal binary64DirectLoss - binary64ExactLoss| < 1 / 10 ^ 15 := by
  have hvalue : ExecFloat.Binary.toRat? binary64DirectLoss = some (1 / 2 ^ 52) := by
    decide +kernel
  obtain ⟨lo, hi⟩ := binary64ExactLoss_mem_Icc
  rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hvalue]
  have hdistance : ((1 / 2 ^ 52 : ℚ) : ℝ) - binary64ExactLoss =
      |((1 / 2 ^ 52 : ℚ) : ℝ) - binary64ExactLoss| := by
    rw [abs_of_nonneg]
    push_cast
    linarith
  refine ⟨hvalue, mul_sqrt_lt_abs_sub (by linarith) hi (d := 1 / 2 ^ 52 - 191 / 10 ^ 35)
    (by norm_num) ?_ ?_, ?_⟩
  · rw [← hdistance]
    push_cast
    linarith
  · norm_num [Model.unitRoundoffAt_eq, FloatFormat.binary64]
  · rw [← hdistance]
    push_cast
    linarith

/-- The loss of `cosineLoss` decodes to `2⁻¹⁰⁷`, within `u √(2 L)` of the exact
loss `L`. -/
theorem binary64_stable :
    ExecFloat.Binary.toRat? binary64StableLoss = some (1 / 2 ^ 107) ∧
      |ExecFloat.Binary.toReal binary64StableLoss - binary64ExactLoss| ≤
        Model.unitRoundoffAt FloatFormat.binary64 * Real.sqrt (2 * binary64ExactLoss) := by
  have hvalue : ExecFloat.Binary.toRat? binary64StableLoss = some (1 / 2 ^ 107) := by
    decide +kernel
  refine ⟨hvalue, ?_⟩
  rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hvalue]
  exact abs_sub_le_mul_sqrt binary64ExactLoss_mem_Icc (by norm_num)
    (Model.unitRoundoffAt_pos _).le
    (by norm_num [Model.unitRoundoffAt_eq, FloatFormat.binary64])
    (by norm_num [Model.unitRoundoffAt_eq, FloatFormat.binary64])

/-! ## Gini impurity in binary32

`labels 300`: 299 labels `0` and one label `1`, with impurity `2 · 299 / 300² = 299 / 45000`. -/

/-- The direct impurity is more than 32 units of roundoff from the exact impurity. -/
theorem binary32_direct_gini :
    ExecFloat.Binary.toRat? (Direct.gini (labels 300) : Binary32) = some (27869 / 2 ^ 22) ∧
      ¬ MixedError (ExecFloat.Binary.toReal (Direct.gini (labels 300) : Binary32)) (299 / 45000)
        (32 * Model.unitRoundoffAt FloatFormat.binary32) 0 := by
  have hvalue : ExecFloat.Binary.toRat? (Direct.gini (labels 300) : Binary32) =
      some (27869 / 2 ^ 22) := by
    decide +kernel
  refine ⟨hvalue, fun bound => ?_⟩
  have h := bound.bound
  rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hvalue,
    abs_of_pos (by norm_num : (0 : ℝ) < ((27869 / 2 ^ 22 : ℚ) : ℝ) - 299 / 45000),
    abs_of_pos (by norm_num : (0 : ℝ) < 299 / 45000)] at h
  norm_num [Model.unitRoundoffAt_eq, FloatFormat.binary32] at h

/-- The impurity of `gini` is within one unit of roundoff of the exact impurity. -/
theorem binary32_gini :
    ExecFloat.Binary.toRat? (DecisionTree.Classifier.gini (labels 300) : Binary32) =
        some (3567209 / 2 ^ 29) ∧
      MixedError (ExecFloat.Binary.toReal (DecisionTree.Classifier.gini (labels 300) :
        Binary32)) (299 / 45000) (Model.unitRoundoffAt FloatFormat.binary32) 0 := by
  have hvalue : ExecFloat.Binary.toRat? (DecisionTree.Classifier.gini (labels 300) :
      Binary32) = some (3567209 / 2 ^ 29) := by
    decide +kernel
  refine ⟨hvalue, MixedError.of_abs_sub_le_mul (Model.unitRoundoffAt_pos _).le ?_⟩
  rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hvalue,
    abs_of_pos (by norm_num : (0 : ℝ) < ((3567209 / 2 ^ 29 : ℚ) : ℝ) - 299 / 45000),
    abs_of_pos (by norm_num : (0 : ℝ) < 299 / 45000)]
  norm_num [Model.unitRoundoffAt_eq, FloatFormat.binary32]

end Cancellation


/-!
## The clamp at one minus epsilon, checked by the kernel

Cross-entropy of probabilities and binary cross-entropy clamp a probability into
`[epsilon, 1 - epsilon]` before its logarithm. A probability at the upper clamp has the loss
`-log (1 - epsilon)`, about `epsilon`. The standard formulas of `Standard` round `1 - epsilon` and
take the logarithm of the rounded value. That rounding is an absolute error of up to half the unit
roundoff `u`, which the logarithm passes on to a loss of about `epsilon`. Once `epsilon` is below
`u / 2`, `1 - epsilon` rounds to one and the loss to zero.
`Tensor.crossEntropy (.probabilities epsilon)` and `BinaryCrossEntropy.term` take that
logarithm as `log1p (-epsilon)`, so no `1 - epsilon` is rounded before it. Over the reals both are
the standard formulas (`crossEntropy_real`, `binaryCrossEntropy_real`).

The input is a certain prediction: one class of probability one, with target one, whose exact
loss is `-log (1 - epsilon)` at the decoded epsilon. In binary32 at `epsilon = 1e-8` and in
binary64 at `epsilon = 1e-17` the standard loss decodes to zero, and in binary64 an absolute
tolerance of `1e-15` accepts it. In bfloat16 at the default epsilon both standard losses decode to
zero, more than 32 units of roundoff from the exact loss. In binary32 at the default epsilon the
standard loss is more than one percent off. In each case the loss of
`Tensor.crossEntropy` and `BinaryCrossEntropy.term` is within one unit of roundoff.
-/


namespace Clamp

open Tensor
open FloatLib.Floats FloatLib.Numerics
open FloatLib.Floats.Formats.BinaryInterchange

section Formulas

variable {α : Type} [Storage α] [Context α]

/-! ## Standard formulas -/

namespace Standard

/-- Standard cross-entropy of probabilities: the logarithm of each probability clamped into
`[epsilon, 1 - epsilon]`, with `1 - epsilon` rounded. -/
def crossEntropy {s : Shape} (epsilon : α) (x target : Tensor α s) (dim : Fin s.rank)
    (mean : Bool := true) : α :=
  let clamp := fun v : α =>
    let v := if v > epsilon then v else epsilon
    if v < (1 : α) - epsilon then v else (1 : α) - epsilon
  let total := Tensor.sum (target * Tensor.log (map clamp x))
  if mean then meanOverAxisSlices (s := s) dim (-total) else -total

/-- Standard binary cross-entropy: at the upper clamp the probability is the rounded
`1 - epsilon` and its complement `epsilon`. -/
def binaryCrossEntropy (predicted target : α) (epsilon : α := Context.defaultEpsilon) : α :=
  let p := if predicted > epsilon then predicted else epsilon
  let (p, logComplement) :=
    if p < (1 : α) - epsilon then (p, Context.log1p (-p))
    else ((1 : α) - epsilon, MathFunctions.log epsilon)
  let t := target * MathFunctions.log p + ((1 : α) - target) * logComplement
  (0 : α) - t

end Standard

/-! ## Inputs -/

/-- A one-class prediction of probability one. -/
def certain : Tensor α [1] := Tensor.ofFn fun _ => 1

/-- The cross-entropy of `certain` against the target `certain`. -/
def loss (epsilon : α) : α :=
  Tensor.crossEntropy (.probabilities epsilon) certain certain ⟨0, by decide⟩

/-- The standard cross-entropy of `certain` against the target `certain`. -/
def standardLoss (epsilon : α) : α :=
  Standard.crossEntropy epsilon certain certain ⟨0, by decide⟩

end Formulas

/-- The loss `-log (1 - x)` of a probability clamped at `1 - x`. -/
noncomputable def exactLoss (x : ℝ) : ℝ := -Real.log (1 - x)

/-! ## The same real values -/

/-- Over the reals the cross-entropy of probabilities is the standard one. -/
theorem crossEntropy_real {s : Shape} (x target : Tensor ℝ s) (dim : Fin s.rank) (epsilon : ℝ)
    (mean : Bool) :
    Tensor.crossEntropy (.probabilities epsilon) x target dim mean =
      Standard.crossEntropy epsilon x target dim mean :=
  Tensor.crossEntropy_probabilities_eq x target dim epsilon mean

/-- Over the reals binary cross-entropy is the standard one. -/
theorem binaryCrossEntropy_real (predicted target epsilon : ℝ) :
    BinaryCrossEntropy.term predicted target epsilon =
      Standard.binaryCrossEntropy predicted target epsilon := by
  have hlog : Context.log1p (-epsilon) = MathFunctions.log (1 - epsilon) := by
    rw [sub_eq_add_neg]
    exact Context.log1p_eq_log_one_add (-epsilon)
  simp only [BinaryCrossEntropy.term, Standard.binaryCrossEntropy]
  split_ifs <;> simp [hlog]

/-! ## Accuracy -/

/-- For `0 ≤ x ≤ 1 / 2`, `exactLoss x` lies within `2 x³` of `x + x² / 2`. -/
theorem exactLoss_mem_Icc {x : ℝ} (h₀ : 0 ≤ x) (h₁ : x ≤ 1 / 2) :
    exactLoss x ∈ Set.Icc (x + x ^ 2 / 2 - 2 * x ^ 3) (x + x ^ 2 / 2 + 2 * x ^ 3) := by
  have hx : |x| < 1 := by rw [abs_of_nonneg h₀]; linarith
  have h := Real.abs_log_sub_add_sum_range_le hx 2
  norm_num [Finset.sum_range_succ, abs_of_nonneg h₀] at h
  have hcube : x ^ 3 / (1 - x) ≤ 2 * x ^ 3 := by
    rw [div_le_iff₀ (by linarith)]
    nlinarith [pow_nonneg h₀ 3]
  rw [abs_le] at h
  unfold exactLoss
  constructor <;> linarith [h.1, h.2]

/-- A value at most `1 + u` times the lower end and at least `1 - u` times the upper end of the
enclosure of `exactLoss x` from `exactLoss_mem_Icc` is within relative error `u` of `exactLoss x`,
for `0 < x ≤ 1 / 4` and `0 ≤ u ≤ 1`. -/
private theorem mixedError_exactLoss {q x u : ℝ} (h₀ : 0 < x) (h₁ : x ≤ 1 / 4) (hu₀ : 0 ≤ u)
    (hu₁ : u ≤ 1) (hlower : q ≤ (1 + u) * (x + x ^ 2 / 2 - 2 * x ^ 3))
    (hupper : (1 - u) * (x + x ^ 2 / 2 + 2 * x ^ 3) ≤ q) : MixedError q (exactLoss x) u 0 := by
  obtain ⟨hlo, hhi⟩ := exactLoss_mem_Icc h₀.le (by linarith)
  have hpos : 0 < x + x ^ 2 / 2 - 2 * x ^ 3 := by
    nlinarith [mul_nonneg (sub_nonneg.2 h₁) (pow_pos h₀ 2).le]
  refine MixedError.of_abs_sub_le_mul hu₀ ?_
  rw [abs_of_pos (hpos.trans_le hlo), abs_le]
  constructor
  · nlinarith [mul_nonneg (sub_nonneg.2 hu₁) (sub_nonneg.2 hhi)]
  · nlinarith [mul_nonneg (by linarith : (0 : ℝ) ≤ 1 + u) (sub_nonneg.2 hlo)]

/-- A value below the lower end of the enclosure of `exactLoss x` by more than `r` times its upper
end, or above the upper end by more than that, is not within relative error `r` of `exactLoss x`,
for `0 < x ≤ 1 / 4`. -/
private theorem not_mixedError_exactLoss {q x r : ℝ} (h₀ : 0 < x) (h₁ : x ≤ 1 / 4) (hr : 0 ≤ r)
    (hfar : r * (x + x ^ 2 / 2 + 2 * x ^ 3) < x + x ^ 2 / 2 - 2 * x ^ 3 - q ∨
      r * (x + x ^ 2 / 2 + 2 * x ^ 3) < q - (x + x ^ 2 / 2 + 2 * x ^ 3)) :
    ¬ MixedError q (exactLoss x) r 0 := fun bound => by
  obtain ⟨hlo, hhi⟩ := exactLoss_mem_Icc h₀.le (by linarith)
  have hpos : 0 < x + x ^ 2 / 2 - 2 * x ^ 3 := by
    nlinarith [mul_nonneg (sub_nonneg.2 h₁) (pow_pos h₀ 2).le]
  have h := bound.bound
  rw [abs_of_pos (hpos.trans_le hlo), add_zero, abs_le] at h
  have hscaled := mul_le_mul_of_nonneg_left hhi hr
  rcases hfar with hfar | hfar <;> linarith [h.1, h.2]

/-! ## Binary32 at `epsilon = 1e-8` -/

/-- `1e-8` rounded to binary32. -/
def binary32Epsilon : Binary32 := 1 / (10 ^ 8 : ℕ)

/-- It decodes to `11258999 / 2⁵⁰`. -/
theorem binary32Epsilon_toRat? :
    ExecFloat.Binary.toRat? binary32Epsilon = some (11258999 / 2 ^ 50) := by
  decide +kernel

/-- The standard loss decodes to zero, which is not within 32 units of roundoff of the exact
loss. -/
theorem binary32_standard :
    ExecFloat.Binary.toRat? (standardLoss binary32Epsilon) = some 0 ∧
      ¬ MixedError (ExecFloat.Binary.toReal (standardLoss binary32Epsilon))
        (exactLoss (ExecFloat.Binary.toReal binary32Epsilon))
        (32 * Model.unitRoundoffAt FloatFormat.binary32) 0 := by
  have hvalue : ExecFloat.Binary.toRat? (standardLoss binary32Epsilon) = some 0 := by
    decide +kernel
  refine ⟨hvalue, ?_⟩
  rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hvalue,
    ExecFloat.Binary.toReal_of_toRat?_eq_some binary32Epsilon_toRat?]
  exact not_mixedError_exactLoss (by norm_num) (by norm_num)
    (mul_pos (by norm_num) (Model.unitRoundoffAt_pos _)).le
    (Or.inl (by norm_num [Model.unitRoundoffAt_eq, FloatFormat.binary32]))

/-- The loss decodes to `11258999 / 2⁵⁰`, the decoded epsilon, within one unit of roundoff of
the exact loss. -/
theorem binary32_stable :
    ExecFloat.Binary.toRat? (loss binary32Epsilon) = some (11258999 / 2 ^ 50) ∧
      MixedError (ExecFloat.Binary.toReal (loss binary32Epsilon))
        (exactLoss (ExecFloat.Binary.toReal binary32Epsilon))
        (Model.unitRoundoffAt FloatFormat.binary32) 0 := by
  have hvalue : ExecFloat.Binary.toRat? (loss binary32Epsilon) = some (11258999 / 2 ^ 50) := by
    decide +kernel
  refine ⟨hvalue, ?_⟩
  rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hvalue,
    ExecFloat.Binary.toReal_of_toRat?_eq_some binary32Epsilon_toRat?]
  exact mixedError_exactLoss (by norm_num) (by norm_num) (Model.unitRoundoffAt_pos _).le
    (by norm_num [Model.unitRoundoffAt_eq, FloatFormat.binary32])
    (by norm_num [Model.unitRoundoffAt_eq, FloatFormat.binary32])
    (by norm_num [Model.unitRoundoffAt_eq, FloatFormat.binary32])

/-! ## Binary32 at the default epsilon

The default epsilon `1e-6` is above the binary32 unit roundoff, so `1 - epsilon` does not round
to one. Its rounding error moves the standard loss by more than one percent. -/

/-- The binary32 default epsilon decodes to `8796093 / 2⁴³`. -/
theorem binary32DefaultEpsilon_toRat? :
    ExecFloat.Binary.toRat? (Context.defaultEpsilon : Binary32) = some (8796093 / 2 ^ 43) := by
  decide +kernel

/-- The standard loss decodes to `8912901 / 2⁴³`, more than one percent from the exact loss. -/
theorem binary32_default_standard :
    ExecFloat.Binary.toRat? (standardLoss (Context.defaultEpsilon : Binary32)) =
        some (8912901 / 2 ^ 43) ∧
      ¬ MixedError (ExecFloat.Binary.toReal (standardLoss (Context.defaultEpsilon : Binary32)))
        (exactLoss (ExecFloat.Binary.toReal (Context.defaultEpsilon : Binary32))) (1 / 100) 0 := by
  have hvalue : ExecFloat.Binary.toRat? (standardLoss (Context.defaultEpsilon : Binary32)) =
      some (8912901 / 2 ^ 43) := by
    decide +kernel
  refine ⟨hvalue, ?_⟩
  rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hvalue,
    ExecFloat.Binary.toReal_of_toRat?_eq_some binary32DefaultEpsilon_toRat?]
  exact not_mixedError_exactLoss (by norm_num) (by norm_num) (by norm_num)
    (Or.inr (by norm_num))

/-- The loss decodes to `8796097 / 2⁴³`, within one unit of roundoff of the exact loss. -/
theorem binary32_default_stable :
    ExecFloat.Binary.toRat? (loss (Context.defaultEpsilon : Binary32)) =
        some (8796097 / 2 ^ 43) ∧
      MixedError (ExecFloat.Binary.toReal (loss (Context.defaultEpsilon : Binary32)))
        (exactLoss (ExecFloat.Binary.toReal (Context.defaultEpsilon : Binary32)))
        (Model.unitRoundoffAt FloatFormat.binary32) 0 := by
  have hvalue : ExecFloat.Binary.toRat? (loss (Context.defaultEpsilon : Binary32)) =
      some (8796097 / 2 ^ 43) := by
    decide +kernel
  refine ⟨hvalue, ?_⟩
  rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hvalue,
    ExecFloat.Binary.toReal_of_toRat?_eq_some binary32DefaultEpsilon_toRat?]
  exact mixedError_exactLoss (by norm_num) (by norm_num) (Model.unitRoundoffAt_pos _).le
    (by norm_num [Model.unitRoundoffAt_eq, FloatFormat.binary32])
    (by norm_num [Model.unitRoundoffAt_eq, FloatFormat.binary32])
    (by norm_num [Model.unitRoundoffAt_eq, FloatFormat.binary32])

/-! ## Bfloat16 at the default epsilon -/

/-- The bfloat16 default epsilon decodes to `67 / 2²⁶`. -/
theorem bfloat16Epsilon_toRat? :
    ExecFloat.Binary.toRat? (Context.defaultEpsilon : BFloat16) = some (67 / 2 ^ 26) := by
  decide +kernel

/-- The standard cross-entropy and binary cross-entropy decode to zero, which is not within 32
units of roundoff of the exact loss. -/
theorem bfloat16_standard :
    ExecFloat.Binary.toRat? (standardLoss (Context.defaultEpsilon : BFloat16)) = some 0 ∧
      ExecFloat.Binary.toRat? (Standard.binaryCrossEntropy (1 : BFloat16) 1) = some 0 ∧
      ¬ MixedError (ExecFloat.Binary.toReal (standardLoss (Context.defaultEpsilon : BFloat16)))
        (exactLoss (ExecFloat.Binary.toReal (Context.defaultEpsilon : BFloat16)))
        (32 * Model.unitRoundoffAt FloatFormat.bfloat16) 0 ∧
      ¬ MixedError (ExecFloat.Binary.toReal (Standard.binaryCrossEntropy (1 : BFloat16) 1))
        (exactLoss (ExecFloat.Binary.toReal (Context.defaultEpsilon : BFloat16)))
        (32 * Model.unitRoundoffAt FloatFormat.bfloat16) 0 := by
  have hce : ExecFloat.Binary.toRat? (standardLoss (Context.defaultEpsilon : BFloat16)) =
      some 0 := by
    decide +kernel
  have hbce : ExecFloat.Binary.toRat? (Standard.binaryCrossEntropy (1 : BFloat16) 1) = some 0 := by
    decide +kernel
  have hfar : ¬ MixedError 0 (exactLoss ((67 / 2 ^ 26 : ℚ) : ℝ))
      (32 * Model.unitRoundoffAt FloatFormat.bfloat16) 0 :=
    not_mixedError_exactLoss (by norm_num) (by norm_num)
      (mul_pos (by norm_num) (Model.unitRoundoffAt_pos _)).le
      (Or.inl (by norm_num [Model.unitRoundoffAt_eq, FloatFormat.bfloat16]))
  refine ⟨hce, hbce, ?_, ?_⟩
  · rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hce, Rat.cast_zero,
      ExecFloat.Binary.toReal_of_toRat?_eq_some bfloat16Epsilon_toRat?]
    exact hfar
  · rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hbce, Rat.cast_zero,
      ExecFloat.Binary.toReal_of_toRat?_eq_some bfloat16Epsilon_toRat?]
    exact hfar

/-- The cross-entropy and binary cross-entropy decode to `67 / 2²⁶`, the decoded epsilon, within
one unit of roundoff of the exact loss. -/
theorem bfloat16_stable :
    ExecFloat.Binary.toRat? (loss (Context.defaultEpsilon : BFloat16)) = some (67 / 2 ^ 26) ∧
      ExecFloat.Binary.toRat? (BinaryCrossEntropy.term (1 : BFloat16) 1) = some (67 / 2 ^ 26) ∧
      MixedError (ExecFloat.Binary.toReal (loss (Context.defaultEpsilon : BFloat16)))
        (exactLoss (ExecFloat.Binary.toReal (Context.defaultEpsilon : BFloat16)))
        (Model.unitRoundoffAt FloatFormat.bfloat16) 0 ∧
      MixedError (ExecFloat.Binary.toReal (BinaryCrossEntropy.term (1 : BFloat16) 1))
        (exactLoss (ExecFloat.Binary.toReal (Context.defaultEpsilon : BFloat16)))
        (Model.unitRoundoffAt FloatFormat.bfloat16) 0 := by
  have hce : ExecFloat.Binary.toRat? (loss (Context.defaultEpsilon : BFloat16)) =
      some (67 / 2 ^ 26) := by
    decide +kernel
  have hbce : ExecFloat.Binary.toRat? (BinaryCrossEntropy.term (1 : BFloat16) 1) =
      some (67 / 2 ^ 26) := by
    decide +kernel
  have hnear : MixedError ((67 / 2 ^ 26 : ℚ) : ℝ) (exactLoss ((67 / 2 ^ 26 : ℚ) : ℝ))
      (Model.unitRoundoffAt FloatFormat.bfloat16) 0 :=
    mixedError_exactLoss (by norm_num) (by norm_num) (Model.unitRoundoffAt_pos _).le
      (by norm_num [Model.unitRoundoffAt_eq, FloatFormat.bfloat16])
      (by norm_num [Model.unitRoundoffAt_eq, FloatFormat.bfloat16])
      (by norm_num [Model.unitRoundoffAt_eq, FloatFormat.bfloat16])
  refine ⟨hce, hbce, ?_, ?_⟩
  · rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hce,
      ExecFloat.Binary.toReal_of_toRat?_eq_some bfloat16Epsilon_toRat?]
    exact hnear
  · rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hbce,
      ExecFloat.Binary.toReal_of_toRat?_eq_some bfloat16Epsilon_toRat?]
    exact hnear

/-! ## Binary64 at `epsilon = 1e-17` -/

/-- `1e-17` rounded to binary64. -/
def binary64Epsilon : Binary64 := 1 / (10 ^ 17 : ℕ)

/-- It decodes to `6490371073168535 / 2¹⁰⁹`. -/
theorem binary64Epsilon_toRat? :
    ExecFloat.Binary.toRat? binary64Epsilon = some (6490371073168535 / 2 ^ 109) := by
  decide +kernel

/-- The standard loss decodes to zero, which is not within 32 units of roundoff of the exact
loss, yet an absolute tolerance of `1e-15` accepts it. -/
theorem binary64_standard :
    ExecFloat.Binary.toRat? (standardLoss binary64Epsilon) = some 0 ∧
      ¬ MixedError (ExecFloat.Binary.toReal (standardLoss binary64Epsilon))
        (exactLoss (ExecFloat.Binary.toReal binary64Epsilon))
        (32 * Model.unitRoundoffAt FloatFormat.binary64) 0 ∧
      |ExecFloat.Binary.toReal (standardLoss binary64Epsilon) -
        exactLoss (ExecFloat.Binary.toReal binary64Epsilon)| < 1 / 10 ^ 15 := by
  have hvalue : ExecFloat.Binary.toRat? (standardLoss binary64Epsilon) = some 0 := by
    decide +kernel
  rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hvalue,
    ExecFloat.Binary.toReal_of_toRat?_eq_some binary64Epsilon_toRat?]
  obtain ⟨lo, hi⟩ := exactLoss_mem_Icc (x := ((6490371073168535 / 2 ^ 109 : ℚ) : ℝ))
    (by norm_num) (by norm_num)
  refine ⟨hvalue, not_mixedError_exactLoss (by norm_num) (by norm_num)
    (mul_pos (by norm_num) (Model.unitRoundoffAt_pos _)).le
    (Or.inl (by norm_num [Model.unitRoundoffAt_eq, FloatFormat.binary64])), ?_⟩
  rw [Rat.cast_zero, zero_sub, abs_neg, abs_lt]
  constructor <;> norm_num at lo hi ⊢ <;> linarith

/-- The loss decodes to `6490371073168535 / 2¹⁰⁹`, the decoded epsilon, within one unit of
roundoff of the exact loss. -/
theorem binary64_stable :
    ExecFloat.Binary.toRat? (loss binary64Epsilon) = some (6490371073168535 / 2 ^ 109) ∧
      MixedError (ExecFloat.Binary.toReal (loss binary64Epsilon))
        (exactLoss (ExecFloat.Binary.toReal binary64Epsilon))
        (Model.unitRoundoffAt FloatFormat.binary64) 0 := by
  have hvalue : ExecFloat.Binary.toRat? (loss binary64Epsilon) =
      some (6490371073168535 / 2 ^ 109) := by
    decide +kernel
  refine ⟨hvalue, ?_⟩
  rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hvalue,
    ExecFloat.Binary.toReal_of_toRat?_eq_some binary64Epsilon_toRat?]
  exact mixedError_exactLoss (by norm_num) (by norm_num) (Model.unitRoundoffAt_pos _).le
    (by norm_num [Model.unitRoundoffAt_eq, FloatFormat.binary64])
    (by norm_num [Model.unitRoundoffAt_eq, FloatFormat.binary64])
    (by norm_num [Model.unitRoundoffAt_eq, FloatFormat.binary64])

end Clamp

/-!
## The cosine loss at a nonpositive epsilon, checked by the kernel

`CosineLoss.fiberLoss` computes the stable `‖p̂ - t̂‖² / 2` of `Cancellation` when both
norms are positive and above their floors. `Standard.fiberLoss` requires a positive epsilon and
norms above their floors. At `epsilon = 0`, or at any negative epsilon, it computes the direct
`1 - p̂ · t̂` (`Standard.fiberLoss_of_not_pos`), although `p̂` and `t̂` are unit vectors over the
reals whenever both norms are positive and above their floors. Both guards take `1 - p̂ · t̂` at a
zero norm. Over the reals the two guards give the same loss (`fiberLoss_real`).

On the inputs of `Cancellation` at `epsilon = 0`, the standard loss decodes to zero in bfloat16
and binary32 and to `2⁻⁵²` in binary64, none of them within `u √(2 L)` of the exact loss `L`, and
an absolute tolerance of `1e-15` accepts the binary64 value. `cosineLoss` at `epsilon = 0` decodes
to its values of `Cancellation` at the default epsilon, within `u √(2 L)` in each format.
-/


namespace CosineGuard

open TorchLean.Tensor
open FloatLib.Floats FloatLib.Numerics
open FloatLib.Floats.Formats.BinaryInterchange
open Cancellation (vector bfloat16Prediction bfloat16Target bfloat16ExactLoss
  bfloat16ExactLoss_mem_Icc binary32Prediction binary32Target binary32ExactLoss
  binary32ExactLoss_mem_Icc binary64Prediction binary64Target binary64ExactLoss
  binary64ExactLoss_mem_Icc)

section Formulas

variable {α : Type} [Storage α] [Context α]

/-! ## Standard formula -/

namespace Standard

/-- Standard cosine fiber loss: `‖p̂ - t̂‖² / 2` at a positive epsilon with both norms above their
floors, and `1 - p̂ · t̂` otherwise. -/
def fiberLoss {s : Shape} (predicted : Tensor α s) (target : Tensor α s)
    (epsilon : α := Context.defaultEpsilon) : α :=
  let (predScale, predNorm) := CosineLoss.normFactors predicted epsilon
  let (targetScale, targetNorm) := CosineLoss.normFactors target epsilon
  let normalizedPred := Tensor.map (fun value => (value / predScale) / predNorm) predicted
  let normalizedTarget := Tensor.map (fun value => (value / targetScale) / targetNorm) target
  if epsilon > 0 ∧ predNorm > epsilon / predScale ∧ targetNorm > epsilon / targetScale then
    let difference := normalizedPred - normalizedTarget
    Tensor.sum (difference * difference) / 2
  else
    (1 : α) - Tensor.sum (normalizedPred * normalizedTarget)

/-- At a nonpositive epsilon the standard fiber loss is the direct loss. -/
theorem fiberLoss_of_not_pos {s : Shape} (predicted target : Tensor α s) {epsilon : α}
    (h : ¬ epsilon > 0) :
    fiberLoss predicted target epsilon =
      Cancellation.Direct.cosineLoss predicted target epsilon := by
  unfold fiberLoss Cancellation.Direct.cosineLoss
  exact ite_eq_right fun hc => h hc.1

end Standard

end Formulas

/-! ## The same real values -/

/-- Over the reals the fiber loss is the standard one. -/
theorem fiberLoss_real {s : Shape} (p t : Tensor ℝ s) (epsilon : ℝ) :
    CosineLoss.fiberLoss p t epsilon = Standard.fiberLoss p t epsilon := by
  rw [CosineLoss.fiberLoss_eq_one_sub_dot]
  unfold Standard.fiberLoss
  set f := CosineLoss.normFactors p epsilon
  set g := CosineLoss.normFactors t epsilon
  set P := Tensor.map (fun value => (value / f.1) / f.2) p
  set Q := Tensor.map (fun value => (value / g.1) / g.2) t
  change 1 - Tensor.sum (P * Q) =
    @ite ℝ (epsilon > 0 ∧ f.2 > epsilon / f.1 ∧ g.2 > epsilon / g.1) _
      (Tensor.sum ((P - Q) * (P - Q)) / 2) (1 - Tensor.sum (P * Q))
  split_ifs with h
  · obtain ⟨he, hp, ht⟩ := h
    have hf : 0 < f.1 := lt_of_lt_of_le (abs_pos.2 he.ne') (le_max_right _ _)
    have hg : 0 < g.1 := lt_of_lt_of_le (abs_pos.2 he.ne') (le_max_right _ _)
    have hstable : CosineLoss.fiberLoss p t epsilon =
        Tensor.sum ((P - Q) * (P - Q)) / 2 := by
      unfold CosineLoss.fiberLoss
      exact ite_eq_left ⟨lt_trans (div_pos he hf) hp, hp, lt_trans (div_pos he hg) ht, ht⟩
    rw [← hstable, CosineLoss.fiberLoss_eq_one_sub_dot]
  · rfl

/-! ## Cosine loss in bfloat16 at `epsilon = 0` -/

/-- The standard bfloat16 loss at `epsilon = 0`. -/
def bfloat16StandardLoss : BFloat16 :=
  Standard.fiberLoss (vector bfloat16Prediction) (vector bfloat16Target) 0

/-- The bfloat16 loss of `cosineLoss` over the one axis at `epsilon = 0`. -/
def bfloat16StableLoss : BFloat16 :=
  cosineLoss (vector bfloat16Prediction) (vector bfloat16Target) (List.finRange _) false 0 ()

/-- The standard loss decodes to zero, which is not within `u √(2 L)` of the exact loss `L`. -/
theorem bfloat16_standard :
    ExecFloat.Binary.toRat? bfloat16StandardLoss = some 0 ∧
      Model.unitRoundoffAt FloatFormat.bfloat16 * Real.sqrt (2 * bfloat16ExactLoss) <
        |ExecFloat.Binary.toReal bfloat16StandardLoss - bfloat16ExactLoss| := by
  have hvalue : ExecFloat.Binary.toRat? bfloat16StandardLoss = some 0 := by
    decide +kernel
  obtain ⟨lo, hi⟩ := bfloat16ExactLoss_mem_Icc
  refine ⟨hvalue, Cancellation.mul_sqrt_lt_abs_sub (by linarith) hi (d := 2111 / 10 ^ 6)
    (by norm_num) ?_ ?_⟩
  · rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hvalue, Rat.cast_zero, zero_sub, abs_neg,
      abs_of_nonneg (by linarith)]
    exact lo
  · norm_num [Model.unitRoundoffAt_eq, FloatFormat.bfloat16]

/-- The loss of `cosineLoss` decodes to `71 / 2¹⁵`, within `u √(2 L)` of the exact loss `L`. -/
theorem bfloat16_stable :
    ExecFloat.Binary.toRat? bfloat16StableLoss = some (71 / 2 ^ 15) ∧
      |ExecFloat.Binary.toReal bfloat16StableLoss - bfloat16ExactLoss| ≤
        Model.unitRoundoffAt FloatFormat.bfloat16 * Real.sqrt (2 * bfloat16ExactLoss) := by
  have hvalue : ExecFloat.Binary.toRat? bfloat16StableLoss = some (71 / 2 ^ 15) := by
    decide +kernel
  refine ⟨hvalue, ?_⟩
  rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hvalue]
  exact Cancellation.abs_sub_le_mul_sqrt bfloat16ExactLoss_mem_Icc (by norm_num)
    (Model.unitRoundoffAt_pos _).le
    (by norm_num [Model.unitRoundoffAt_eq, FloatFormat.bfloat16])
    (by norm_num [Model.unitRoundoffAt_eq, FloatFormat.bfloat16])

/-! ## Cosine loss in binary32 at `epsilon = 0` -/

/-- The standard binary32 loss at `epsilon = 0`. -/
def binary32StandardLoss : Binary32 :=
  Standard.fiberLoss (vector binary32Prediction) (vector binary32Target) 0

/-- The binary32 loss of `cosineLoss` over the one axis at `epsilon = 0`. -/
def binary32StableLoss : Binary32 :=
  cosineLoss (vector binary32Prediction) (vector binary32Target) (List.finRange _) false 0 ()

/-- The standard loss decodes to zero, which is not within `u √(2 L)` of the exact loss `L`. -/
theorem binary32_standard :
    ExecFloat.Binary.toRat? binary32StandardLoss = some 0 ∧
      Model.unitRoundoffAt FloatFormat.binary32 * Real.sqrt (2 * binary32ExactLoss) <
        |ExecFloat.Binary.toReal binary32StandardLoss - binary32ExactLoss| := by
  have hvalue : ExecFloat.Binary.toRat? binary32StandardLoss = some 0 := by
    decide +kernel
  obtain ⟨lo, hi⟩ := binary32ExactLoss_mem_Icc
  refine ⟨hvalue, Cancellation.mul_sqrt_lt_abs_sub (by linarith) hi (d := 91944 / 10 ^ 13)
    (by norm_num) ?_ ?_⟩
  · rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hvalue, Rat.cast_zero, zero_sub, abs_neg,
      abs_of_nonneg (by linarith)]
    exact lo
  · norm_num [Model.unitRoundoffAt_eq, FloatFormat.binary32]

/-- The loss of `cosineLoss` decodes to `2588045 / 2⁴⁸`, within `u √(2 L)` of the exact loss
`L`. -/
theorem binary32_stable :
    ExecFloat.Binary.toRat? binary32StableLoss = some (2588045 / 2 ^ 48) ∧
      |ExecFloat.Binary.toReal binary32StableLoss - binary32ExactLoss| ≤
        Model.unitRoundoffAt FloatFormat.binary32 * Real.sqrt (2 * binary32ExactLoss) := by
  have hvalue : ExecFloat.Binary.toRat? binary32StableLoss = some (2588045 / 2 ^ 48) := by
    decide +kernel
  refine ⟨hvalue, ?_⟩
  rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hvalue]
  exact Cancellation.abs_sub_le_mul_sqrt binary32ExactLoss_mem_Icc (by norm_num)
    (Model.unitRoundoffAt_pos _).le
    (by norm_num [Model.unitRoundoffAt_eq, FloatFormat.binary32])
    (by norm_num [Model.unitRoundoffAt_eq, FloatFormat.binary32])

/-! ## Cosine loss in binary64 at `epsilon = 0` -/

/-- The standard binary64 loss at `epsilon = 0`. -/
def binary64StandardLoss : Binary64 :=
  Standard.fiberLoss (vector binary64Prediction) (vector binary64Target) 0

/-- The binary64 loss of `cosineLoss` over the one axis at `epsilon = 0`. -/
def binary64StableLoss : Binary64 :=
  cosineLoss (vector binary64Prediction) (vector binary64Target) (List.finRange _) false 0 ()

/-- The standard loss decodes to `2⁻⁵²`. It is not within `u √(2 L)` of the exact loss `L`, yet
an absolute tolerance of `1e-15` accepts it. -/
theorem binary64_standard :
    ExecFloat.Binary.toRat? binary64StandardLoss = some (1 / 2 ^ 52) ∧
      Model.unitRoundoffAt FloatFormat.binary64 * Real.sqrt (2 * binary64ExactLoss) <
        |ExecFloat.Binary.toReal binary64StandardLoss - binary64ExactLoss| ∧
      |ExecFloat.Binary.toReal binary64StandardLoss - binary64ExactLoss| < 1 / 10 ^ 15 := by
  have hvalue : ExecFloat.Binary.toRat? binary64StandardLoss = some (1 / 2 ^ 52) := by
    decide +kernel
  obtain ⟨lo, hi⟩ := binary64ExactLoss_mem_Icc
  rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hvalue]
  have hdistance : ((1 / 2 ^ 52 : ℚ) : ℝ) - binary64ExactLoss =
      |((1 / 2 ^ 52 : ℚ) : ℝ) - binary64ExactLoss| := by
    rw [abs_of_nonneg]
    push_cast
    linarith
  refine ⟨hvalue, Cancellation.mul_sqrt_lt_abs_sub (by linarith) hi
    (d := 1 / 2 ^ 52 - 191 / 10 ^ 35) (by norm_num) ?_ ?_, ?_⟩
  · rw [← hdistance]
    push_cast
    linarith
  · norm_num [Model.unitRoundoffAt_eq, FloatFormat.binary64]
  · rw [← hdistance]
    push_cast
    linarith

/-- The loss of `cosineLoss` decodes to `2⁻¹⁰⁷`, within `u √(2 L)` of the exact loss `L`. -/
theorem binary64_stable :
    ExecFloat.Binary.toRat? binary64StableLoss = some (1 / 2 ^ 107) ∧
      |ExecFloat.Binary.toReal binary64StableLoss - binary64ExactLoss| ≤
        Model.unitRoundoffAt FloatFormat.binary64 * Real.sqrt (2 * binary64ExactLoss) := by
  have hvalue : ExecFloat.Binary.toRat? binary64StableLoss = some (1 / 2 ^ 107) := by
    decide +kernel
  refine ⟨hvalue, ?_⟩
  rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hvalue]
  exact Cancellation.abs_sub_le_mul_sqrt binary64ExactLoss_mem_Icc (by norm_num)
    (Model.unitRoundoffAt_pos _).le
    (by norm_num [Model.unitRoundoffAt_eq, FloatFormat.binary64])
    (by norm_num [Model.unitRoundoffAt_eq, FloatFormat.binary64])

end CosineGuard

end TorchLean.Examples

/-!
## The distillation gradient at the teacher logits

At the teacher logits `SoftmaxCrossEntropy.teacher`, `[2, 1, 0]`, that
`SoftmaxCrossEntropy.checkDistillZero` uses, the kernel decides the conditions of
`TorchLean.Roundoff.distillLossBackward_self` in binary32 and
bfloat16. It also evaluates the whole gradient in binary32, bfloat16 and binary64, where every
coordinate decodes to zero.
-/

namespace TorchLean.Examples.Distillation

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange FloatLib.Numerics

/-- `Roundoff.distillLossBackward_self` in binary32 at the teacher logits `[2, 1, 0]`, whose
conditions the kernel decides. -/
theorem backward_self_binary32 (c : Shape.Coord [3]) :
    ExecFloat.Binary.toRat? (Tensor.distillLossBackward ⟨0, Shape.AxisInBounds.proof⟩
      (SoftmaxCrossEntropy.teacher : Tensor Binary32 [3]) SoftmaxCrossEntropy.teacher c) =
      some 0 :=
  Roundoff.distillLossBackward_self ⟨0, Shape.AxisInBounds.proof⟩ _ (by decide)
    (by decide +kernel) (by decide +kernel) c

/-- The same in bfloat16. -/
theorem backward_self_bfloat16 (c : Shape.Coord [3]) :
    ExecFloat.Binary.toRat? (Tensor.distillLossBackward ⟨0, Shape.AxisInBounds.proof⟩
      (SoftmaxCrossEntropy.teacher : Tensor BFloat16 [3]) SoftmaxCrossEntropy.teacher c) =
      some 0 :=
  Roundoff.distillLossBackward_self ⟨0, Shape.AxisInBounds.proof⟩ _ (by decide)
    (by decide +kernel) (by decide +kernel) c

/-- The check `SoftmaxCrossEntropy.checkDistillZero` in binary32: every coordinate of the
gradient at equal logits decodes to zero. -/
theorem backward_teacher_binary32 (i : Fin 3) :
    ExecFloat.Binary.toRat?
      ((Tensor.distillLossBackward ⟨0, Shape.AxisInBounds.proof⟩
        (SoftmaxCrossEntropy.teacher : Tensor Binary32 [3])
        SoftmaxCrossEntropy.teacher).getScalar i) = some 0 := by
  revert i
  decide +kernel

/-- The same in bfloat16. -/
theorem backward_teacher_bfloat16 (i : Fin 3) :
    ExecFloat.Binary.toRat?
      ((Tensor.distillLossBackward ⟨0, Shape.AxisInBounds.proof⟩
        (SoftmaxCrossEntropy.teacher : Tensor BFloat16 [3])
        SoftmaxCrossEntropy.teacher).getScalar i) = some 0 := by
  revert i
  decide +kernel

/-- The same in binary64. -/
theorem backward_teacher_binary64 (i : Fin 3) :
    ExecFloat.Binary.toRat?
      ((Tensor.distillLossBackward ⟨0, Shape.AxisInBounds.proof⟩
        (SoftmaxCrossEntropy.teacher : Tensor Binary64 [3])
        SoftmaxCrossEntropy.teacher).getScalar i) = some 0 := by
  revert i
  decide +kernel

end TorchLean.Examples.Distillation
