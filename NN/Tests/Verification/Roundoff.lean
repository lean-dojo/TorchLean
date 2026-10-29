/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Proofs.RuntimeApprox
public import NN.Proofs.Autograd.Tape.Algebra.Nodes
public import NN.Proofs.Autograd.FDeriv.SoftmaxSpec
public import NN.Runtime.Autograd.Torch.Core.TypedGraph
public import FloatLib.Floats.Formats.BinaryInterchange.Configured
public import NN.Spec.Core.FloatInstances
public import NN.Spec.Autograd.Ops
public import NN.Spec.Core.Tensor.Numerics
public import NN.Spec.Layers.Normalization.Core
public import NN.Tensor
import FloatLib.Numerics.Enclosure.Elementary.Proof

/-!
# Automatic numerical proof regressions

The symbolic checks quantify over the floating-point descriptor. They exercise source unfolding,
shared intermediates, and supplied input error, rather than a separate numerical specification.
The cancellation example has a small absolute error but loses an entire nonzero result.
-/

public section

namespace NN.Tests.Verification.Roundoff

open FloatLib.Floats
open FloatLib.Floats.Formats.BinaryInterchange
open TorchLean
open Proofs.RuntimeApprox.Roundoff
open Model.Transcendentals FloatLib.Numerics

variable {format : FloatFormat}

local notation "Value" => ExecFloat.Binary format.expWidth format.fracWidth
  format.encoding format.exponentBias format.expWidth_ge_two format.fracWidth_pos
  format.exponentBias_pos format.exponentBias_le_maxFinite

/-- The primitive proof applies to every IEEE descriptor, not just binary32 and binary64. -/
theorem square_error (x : Value) (hformat : format.isIEEE = true)
    (hx : ExecFloat.Binary.isFinite x = true)
    (hresult : ExecFloat.Binary.isFinite (x * x) = true) :
    |ExecFloat.Binary.toReal (x * x) - ExecFloat.Binary.toReal x * ExecFloat.Binary.toReal x| ≤
      Model.epsilonAt format (ExecFloat.Binary.toReal x * ExecFloat.Binary.toReal x) := by
  have bound := by roundoff (ExecFloat.Binary.toReal (x * x))
  simpa using bound

/-- Nonzero preservation requires a scale-sensitive bound; nonzero real inputs alone do not
exclude floating-point underflow. -/
theorem square_stays_positive (x : Value) (hformat : format.isIEEE = true)
    (hx : ExecFloat.Binary.isFinite x = true)
    (hresult : ExecFloat.Binary.isFinite (x * x) = true)
    (hmargin : Model.epsilonAt format (ExecFloat.Binary.toReal x * ExecFloat.Binary.toReal x) <
      ExecFloat.Binary.toReal x * ExecFloat.Binary.toReal x) :
    0 < ExecFloat.Binary.toReal (x * x) := by
  roundoff

def squareSum {α : Type} [Add α] [Mul α] (x y : α) : α :=
  let sum := x + y
  sum * sum

/-- A shared intermediate is analyzed once; both uses still contribute their input error. -/
theorem square_sum_error (x y : Value) (hformat : format.isIEEE = true)
    (hx : ExecFloat.Binary.isFinite x = true) (hy : ExecFloat.Binary.isFinite y = true)
    (hsum : ExecFloat.Binary.isFinite (x + y) = true)
    (hresult : ExecFloat.Binary.isFinite (squareSum x y) = true) :
    |ExecFloat.Binary.toReal (squareSum x y) -
        (ExecFloat.Binary.toReal x + ExecFloat.Binary.toReal y) *
          (ExecFloat.Binary.toReal x + ExecFloat.Binary.toReal y)| ≤
      |ExecFloat.Binary.toReal (x + y)| *
          Model.epsilonAt format (ExecFloat.Binary.toReal x + ExecFloat.Binary.toReal y) +
        |ExecFloat.Binary.toReal x + ExecFloat.Binary.toReal y| *
          Model.epsilonAt format (ExecFloat.Binary.toReal x + ExecFloat.Binary.toReal y) +
        Model.epsilonAt format
          (ExecFloat.Binary.toReal (x + y) * ExecFloat.Binary.toReal (x + y)) := by
  roundoff

/-- An approximate input is propagated, rather than silently replaced by an exact input. -/
theorem uncertain_square_error (x : Value) (reference inputError : ℝ)
    (hformat : format.isIEEE = true) (hx : ExecFloat.Binary.isFinite x = true)
    (hresult : ExecFloat.Binary.isFinite (x * x) = true)
    (hinput : |ExecFloat.Binary.toReal x - reference| ≤ inputError) :
    |ExecFloat.Binary.toReal (x * x) - reference * reference| ≤
      |ExecFloat.Binary.toReal x| * inputError + |reference| * inputError +
        Model.epsilonAt format (ExecFloat.Binary.toReal x * ExecFloat.Binary.toReal x) := by
  roundoff

/-- The same analyzer consumes the descriptor model without a configured storage carrier. -/
theorem model_square_error (x : Model format) (hformat : format.isIEEE = true)
    (hx : Model.isFinite x = true) (hresult : Model.isFinite (Model.mul x x) = true) :
    |Model.toReal (Model.mul x x) - Model.toReal x * Model.toReal x| ≤
      Model.epsilonAt format (Model.toReal x * Model.toReal x) := by
  roundoff

/-- FMA incurs one rounding allowance, not the two of an unfused multiply followed by add. -/
theorem fused_error (x y z : Model format) (hformat : format.isIEEE = true)
    (hx : Model.isFinite x = true) (hy : Model.isFinite y = true)
    (hz : Model.isFinite z = true) (hresult : Model.isFinite (Model.fma x y z) = true) :
    |Model.toReal (Model.fma x y z) - (Model.toReal x * Model.toReal y + Model.toReal z)| ≤
      Model.epsilonAt format (Model.toReal x * Model.toReal y + Model.toReal z) := by
  roundoff

/-- Casts retain their own destination precision inside a subsequent arithmetic operation. -/
theorem mixed_precision_error {source target : FloatFormat} (x y : Model source)
    (htarget : target.isIEEE = true) (hx : Model.isFinite x = true)
    (hy : Model.isFinite y = true)
    (hcx : Model.isFinite (Model.cast source target x) = true)
    (hcy : Model.isFinite (Model.cast source target y) = true)
    (hresult : Model.isFinite
      (Model.add (Model.cast source target x) (Model.cast source target y)) = true) :
    |Model.toReal (Model.add (Model.cast source target x) (Model.cast source target y)) -
        (Model.toReal x + Model.toReal y)| ≤
      Model.epsilonAt target (Model.toReal x) + Model.epsilonAt target (Model.toReal y) +
        Model.epsilonAt target (Model.toReal (Model.cast source target x) +
          Model.toReal (Model.cast source target y)) := by
  roundoff

/-- Tensor coordinates retain arbitrary shapes and use the same scalar proof generator. -/
theorem tensor_product_error {shape : Spec.Shape} (x y : Tensor Value shape)
    (i : shape.Coord) (hformat : format.isIEEE = true)
    (hx : ExecFloat.Binary.isFinite (x i) = true)
    (hy : ExecFloat.Binary.isFinite (y i) = true)
    (hresult : ExecFloat.Binary.isFinite (Tensor.mulSpec x y i) = true) :
    |ExecFloat.Binary.toReal (Tensor.mulSpec x y i) -
        ExecFloat.Binary.toReal (x i) * ExecFloat.Binary.toReal (y i)| ≤
      Model.epsilonAt format (ExecFloat.Binary.toReal (x i) * ExecFloat.Binary.toReal (y i)) := by
  roundoff

/-- Analyze the existing autograd scale pullback itself, with no alternative backward program. -/
theorem scale_pullback_error {n : Nat} (input seed : Tensor Value [n]) (factor : Value)
    (i : Fin n) (hformat : format.isIEEE = true)
    (hseed : ExecFloat.Binary.isFinite (seed.getScalar i) = true)
    (hfactor : ExecFloat.Binary.isFinite factor = true)
    (hresult : ExecFloat.Binary.isFinite
      (((Spec.scaleOp factor).backward input seed).getScalar i) = true) :
    |ExecFloat.Binary.toReal (((Spec.scaleOp factor).backward input seed).getScalar i) -
        ExecFloat.Binary.toReal (seed.getScalar i) * ExecFloat.Binary.toReal factor| ≤
      Model.epsilonAt format
        (ExecFloat.Binary.toReal (seed.getScalar i) * ExecFloat.Binary.toReal factor) := by
  roundoff

/-- The captured-division pullback incurs one division rounding, with no separately rounded
reciprocal and no extra real-denominator assumption when the inputs are exact observations. -/
theorem division_pullback_error {shape : Spec.Shape} (input rhs seed : Tensor Value shape)
    (i : shape.Coord) (hformat : format.isIEEE = true)
    (hseed : ExecFloat.Binary.isFinite (seed i) = true)
    (hrhs : ExecFloat.Binary.isFinite (rhs i) = true)
    (hzero : ExecFloat.Binary.isZero (rhs i) = false)
    (hresult : ExecFloat.Binary.isFinite ((Spec.divOp rhs).backward input seed i) = true) :
    |ExecFloat.Binary.toReal ((Spec.divOp rhs).backward input seed i) -
        ExecFloat.Binary.toReal (seed i) / ExecFloat.Binary.toReal (rhs i)| ≤
      Model.epsilonAt format
        (ExecFloat.Binary.toReal (seed i) / ExecFloat.Binary.toReal (rhs i)) := by
  roundoff

/-- For a subnormal denominator, an isolated reciprocal overflows although the weighted gradient
is exactly one. This checks the actual pullback, including its tensor operations. -/
theorem division_pullback_subnormal :
    let tiny : Binary32 := Rat.cast (1 / 2 ^ 140 : Rat)
    let value : Tensor Binary32 [1] := Tensor.full [1] tiny
    ExecFloat.Binary.toRat? (((Spec.divOp value).backward value value).getScalar 0) = some 1 := by
  dsimp only
  roundoff
  all_goals decide +kernel

/-- The accumulator may round at every step. Its error is composed in the actual traversal
order, for any tensor shape; range evidence concerns only states visited by that traversal. -/
theorem reduction_error [Storage Value] {shape : Spec.Shape}
    (input : Tensor Value shape) (initial : Value)
    (hformat : format.isIEEE = true)
    (hvalid : AllSteps (fun a b : Value => a + b)
      (fun a b => ExecFloat.Binary.isFinite a = true ∧
        ExecFloat.Binary.isFinite b = true ∧ ExecFloat.Binary.isFinite (a + b) = true)
      initial input) :
    |ExecFloat.Binary.toReal (input.foldl (fun a b => a + b) initial) -
        input.foldl (fun a b => a + ExecFloat.Binary.toReal b) (ExecFloat.Binary.toReal initial)| ≤
      foldError (fun a b : Value => a + b) (fun a b => a + ExecFloat.Binary.toReal b)
        (fun a _ error b => error +
          Model.epsilonAt format (ExecFloat.Binary.toReal a + ExecFloat.Binary.toReal b))
        initial (ExecFloat.Binary.toReal initial) 0 input := by
  roundoff

/-- Fuse the existing scale pullback with its reduction. Both multiplication and accumulation
contribute rounding; no independently written backward implementation is supplied. -/
theorem summed_scale_pullback_error [Storage Value] {shape : Spec.Shape}
    (input seed : Tensor Value shape) (factor : Value)
    (hformat : format.isIEEE = true) (hfactor : ExecFloat.Binary.isFinite factor = true)
    (hvalid : AllSteps (fun a b : Value => a + b * factor)
      (fun a b => ExecFloat.Binary.isFinite a = true ∧
        ExecFloat.Binary.isFinite (b * factor) = true ∧
        ExecFloat.Binary.isFinite (a + b * factor) = true ∧
        ExecFloat.Binary.isFinite b = true) 0 seed) :
    |ExecFloat.Binary.toReal (Tensor.sumSpec ((Spec.scaleOp factor).backward input seed)) -
        seed.foldl (fun a b => a + ExecFloat.Binary.toReal b * ExecFloat.Binary.toReal factor)
          (ExecFloat.Binary.toReal (0 : Value))| ≤
      foldError (fun a b : Value => a + b * factor)
        (fun a b => a + ExecFloat.Binary.toReal b * ExecFloat.Binary.toReal factor)
        (fun a _ error b => error +
          Model.epsilonAt format (ExecFloat.Binary.toReal b * ExecFloat.Binary.toReal factor) +
          Model.epsilonAt format (ExecFloat.Binary.toReal a + ExecFloat.Binary.toReal (b * factor)))
        0 (ExecFloat.Binary.toReal (0 : Value)) 0 seed := by
  roundoff

/-- Finite inputs alone do not imply that their sum stays finite. -/
theorem finite_reduction_check :
    let valid := fun a b : Binary32 =>
      ExecFloat.Binary.isFinite a = true ∧ ExecFloat.Binary.isFinite b = true ∧
        ExecFloat.Binary.isFinite (a + b) = true
    let large : Binary32 := Rat.cast (3 * 10 ^ 38 : Rat)
    AllSteps (· + ·) valid 0 (Tensor.full [2] 1) ∧
      ExecFloat.Binary.isFinite large = true ∧
      ¬ AllSteps (· + ·) valid 0 (Tensor.full [2] large) := by
  decide +kernel

-- Symbolic source analysis must also retain the reduction's unproved range conditions.
example {shape : Spec.Shape} (_input : Tensor Value shape) (_initial : Value)
    (_hformat : format.isIEEE = true)
    (_hinput : ∀ index : shape.Coord, ExecFloat.Binary.isFinite (_input index) = true)
    (_hinit : ExecFloat.Binary.isFinite _initial = true) : True := by
  fail_if_success
    have : |ExecFloat.Binary.toReal (_input.foldl (fun a b => a + b) _initial) -
        _input.foldl (fun a b => a + ExecFloat.Binary.toReal b)
          (ExecFloat.Binary.toReal _initial)| ≤
      foldError (fun a b : Value => a + b) (fun a b => a + ExecFloat.Binary.toReal b)
        (fun a _ error b => error +
          Model.epsilonAt format (ExecFloat.Binary.toReal a + ExecFloat.Binary.toReal b))
        _initial (ExecFloat.Binary.toReal _initial) 0 _input := by
      roundoff
  trivial

/-- Only the selected branch needs finite operands and output. The reference retains the
program's actual test; replacing it with a real comparison requires a separate argument. -/
theorem branch_error [Storage Value] {shape : Spec.Shape}
    (condition : Prop) [Decidable condition] (x y : Tensor Value shape) (i : shape.Coord)
    (hformat : format.isIEEE = true)
    (hx : condition → ExecFloat.Binary.isFinite (x i) = true)
    (hy : ¬condition → ExecFloat.Binary.isFinite (y i) = true)
    (hsx : condition → ExecFloat.Binary.isFinite (x i * x i) = true)
    (hsy : ¬condition → ExecFloat.Binary.isFinite (y i * y i) = true) :
    |ExecFloat.Binary.toReal ((if condition then Tensor.mulSpec x x else Tensor.mulSpec y y) i) -
        (if condition then ExecFloat.Binary.toReal (x i) * ExecFloat.Binary.toReal (x i)
        else ExecFloat.Binary.toReal (y i) * ExecFloat.Binary.toReal (y i))| ≤
      if condition then
        Model.epsilonAt format (ExecFloat.Binary.toReal (x i) * ExecFloat.Binary.toReal (x i))
      else Model.epsilonAt format
        (ExecFloat.Binary.toReal (y i) * ExecFloat.Binary.toReal (y i)) := by
  roundoff

/-- A proof cached under one guard cannot discharge the opposite branch's obligations. -/
example (condition : Prop) [Decidable condition] (x : Value)
    (_hformat : format.isIEEE = true) (_hx : ExecFloat.Binary.isFinite x = true)
    (_hresult : condition → ExecFloat.Binary.isFinite (x * x) = true) : True := by
  fail_if_success
    have : |ExecFloat.Binary.toReal (if condition then x * x else x * x) -
        (if condition then ExecFloat.Binary.toReal x * ExecFloat.Binary.toReal x
        else ExecFloat.Binary.toReal x * ExecFloat.Binary.toReal x)| ≤
        if condition then
          Model.epsilonAt format (ExecFloat.Binary.toReal x * ExecFloat.Binary.toReal x)
        else Model.epsilonAt format (ExecFloat.Binary.toReal x * ExecFloat.Binary.toReal x) := by
      roundoff
  trivial

/-- Omitting an intermediate's finiteness proof must not certify its real interpretation. -/
example (x : Value) (_hformat : format.isIEEE = true)
    (_hx : ExecFloat.Binary.isFinite x = true) : True := by
  fail_if_success
    have : |ExecFloat.Binary.toReal (x * x) -
        ExecFloat.Binary.toReal x * ExecFloat.Binary.toReal x| ≤
        Model.epsilonAt format (ExecFloat.Binary.toReal x * ExecFloat.Binary.toReal x) := by
      roundoff
  fail_if_success
    have bound := by roundoff (ExecFloat.Binary.toReal (x * x))
  trivial

/-- A local dictionary spelling addition as multiplication has no FloatLib multiplication
certificate. Matching the `*` syntax must not certify it. -/
example (x : Value) (_hformat : format.isIEEE = true)
    (_hx : ExecFloat.Binary.isFinite x = true)
    (_hresult : ExecFloat.Binary.isFinite (x + x) = true) : True := by
  let : Mul Value := ⟨fun a b => a + b⟩
  fail_if_success
    have : |ExecFloat.Binary.toReal (x * x) -
        ExecFloat.Binary.toReal x * ExecFloat.Binary.toReal x| ≤
        Model.epsilonAt format (ExecFloat.Binary.toReal x * ExecFloat.Binary.toReal x) := by
      roundoff
  fail_if_success
    have : ExecFloat.Binary.toModel (x * x) =
        Model.mul (ExecFloat.Binary.toModel x) (ExecFloat.Binary.toModel x) := by
      roundoff
  trivial

/-- A numerical proof cannot discard fresh rounding by asking for a zero budget. -/
example (x : Value) (_hformat : format.isIEEE = true)
    (_hx : ExecFloat.Binary.isFinite x = true)
    (_hresult : ExecFloat.Binary.isFinite (x * x) = true) : True := by
  fail_if_success
    have : |ExecFloat.Binary.toReal (x * x) -
        ExecFloat.Binary.toReal x * ExecFloat.Binary.toReal x| ≤ 0 := by
      roundoff
  trivial

def cancellation {α : Type} [Add α] [Sub α] (large small : α) : α :=
  (large + small) - large

/-- Ordinary finite arithmetic can lose the entire small result, even at modest magnitudes.
The Boolean computation is checked by reduction, without `native_decide`. -/
theorem cancellation_loses_one :
    ExecFloat.Binary.toRat? (cancellation (256 : BFloat16) 1) = some 0 := by
  simp only [cancellation, ExecFloat.Binary.toRat?, ExecFloat.Binary.toModel_hSub,
    ExecFloat.Binary.toModel_hAdd]
  decide +kernel

/-- The library's dot product uses the same proof generator across arbitrary tensor shapes. -/
theorem exact_dot_error [Storage Rat] {shape : Spec.Shape} (x y : Tensor Rat shape) :
    |((Tensor.dotSpec x y : Rat) : ℝ) -
        Tensor.dotSpec (x.map (fun value : Rat => (value : ℝ)))
          (y.map (fun value : Rat => (value : ℝ)))| ≤ 0 := by
  roundoff

/-- The existing matrix pullback uses only addition, multiplication, and zero; exact rationals
need no artificial transcendental context or separate gradient specification. -/
theorem exact_matmul_pullback_error [Storage Rat] {m n p : Nat}
    (left : Tensor Rat [m, n]) (right : Tensor Rat [n, p]) (seed : Tensor Rat [m, p])
    (row : Fin m) (column : Fin n) :
    |((Spec.get2 ((Spec.matmulRightOp (batchA := []) (batchB := []) (batch := [])
          (Spec.Shape.CanBroadcastTo.refl []) (Spec.Shape.CanBroadcastTo.refl []) right).backward
        left seed) row column : Rat) : ℝ) -
      Spec.get2 ((Spec.matmulRightOp (batchA := []) (batchB := []) (batch := [])
          (Spec.Shape.CanBroadcastTo.refl []) (Spec.Shape.CanBroadcastTo.refl [])
          (right.map fun value : Rat => (value : ℝ))).backward
        (left.map fun value : Rat => (value : ℝ))
        (seed.map fun value : Rat => (value : ℝ))) row column| ≤ 0 := by
  roundoff

/-- Exact arithmetic uses the same loop analysis, with a zero accumulated budget. -/
theorem exact_fold_error [Storage Rat] {shape : Spec.Shape}
    (input : Tensor Rat shape) (initial : Rat) :
    |((input.foldl (fun a b => a + b) initial : Rat) : ℝ) -
        input.foldl (fun (a : ℝ) (b : Rat) => a + (b : ℝ)) (initial : ℝ)| ≤ 0 := by
  roundoff

section Configured

variable {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "ConfiguredValue" => ExecFloat (Configured.Family format code plan)

/-- Conversion uses finiteness of the actual configured result, without a separate model premise. -/
theorem rational_cast_error (value : Rat) (hformat : format.isIEEE = true)
    (hfinite : ExecFloat.Binary.isFinite (Rat.cast value : ConfiguredValue) = true) :
    |ExecFloat.Binary.toReal (Rat.cast value : ConfiguredValue) - (value : ℝ)| ≤
      Model.epsilonAt format (value : ℝ) := by
  roundoff

/-- A decimal constructed inside the program is compared with the exact real decimal. -/
theorem decimal_literal_error (hformat : format.isIEEE = true)
    (hfinite : ExecFloat.Binary.isFinite (0.1 : ConfiguredValue) = true) :
    |ExecFloat.Binary.toReal (0.1 : ConfiguredValue) - 0.1| ≤
      Model.epsilonAt format 0.1 := by
  roundoff

/-- A runtime conversion cannot become an exact input by omitting its range evidence. -/
example (_value : Rat) (_hformat : format.isIEEE = true) : True := by
  fail_if_success
    have bound := by roundoff (ExecFloat.Binary.toReal (Rat.cast _value : ConfiguredValue))
  trivial

/-- Exact real arithmetic retains its totalized division without a floating-point domain rule. -/
example (x : ℝ) : |x / 0 - x / 0| ≤ 0 := by
  roundoff

/-- An explicitly chosen input reference is propagated once, even if it has its own bound. -/
example (a b reference firstError secondError : ℝ)
    (hfirst : |a - b| ≤ firstError) (_hnext : |b - reference| ≤ secondError) :
    |max a 0 - max b 0| ≤ max firstError 0 := by
  roundoff

/-- A successful recovery retains the error of the product supplied to it. Both the optional
result and its fallback follow the original branch and selected scalar implementation. -/
roundoff theorem recovered_coefficient_error :=
  fun (input upstream fallback : ConfiguredValue) (isTanh : Bool)
    (condition : Prop) [Decidable condition] (_hformat : format.isIEEE = true) =>
    ExecFloat.Binary.toReal (show ConfiguredValue from
      match (if condition then
          @Numeric.Internal.enclosedSlope? ConfiguredValue
            (Numeric.QuotientArithmetic.configured
              (format := format) (plan := plan) (code := code))
            (input * input) (some upstream) isTanh
        else none) with
      | none => fallback
      | some result => result)

/-- An optional result from an unknown program supplies no rounding evidence. -/
example (_producer : ConfiguredValue → Option ConfiguredValue)
    (_input _fallback : ConfiguredValue) (_condition : Prop) [Decidable _condition] : True := by
  fail_if_success
    have bound := by
      roundoff (ExecFloat.Binary.toReal (show ConfiguredValue from
        match (if _condition then _producer _input else none) with
        | none => _fallback
        | some result => result))
  trivial

/-- Several optional producers retain their computations through the shared match traversal. -/
roundoff theorem computed_options_error :=
  fun (input seed fallback : ConfiguredValue) (left right : Prop)
    [Decidable left] [Decidable right] (_hformat : format.isIEEE = true) =>
    ExecFloat.Binary.toReal (show ConfiguredValue from
      match (if left then some (input * input) else none),
          (if right then some seed else none) with
      | some result, some _ => result
      | _, _ => fallback)

/-- Splitting several optional results does not supply a law for an unknown payload. -/
example (_left _right : ConfiguredValue → Option ConfiguredValue)
    (_input _fallback : ConfiguredValue) : True := by
  fail_if_success
    have bound := by
      roundoff (ExecFloat.Binary.toReal (show ConfiguredValue from
        match _left _input, _right _input with
        | some result, some _ => result
        | _, _ => _fallback))
  trivial

/-- A successful decoder retains the arithmetic that produced its input. -/
roundoff theorem rational_payload_error :=
  fun (input : ConfiguredValue) (_hformat : format.isIEEE = true)
    (found : (ExecFloat.Binary.toRat? (input * input)).isSome) =>
    (((ExecFloat.Binary.toRat? (input * input)).get found : Rat) : Real)

/-- Coefficient encoding observes the same scalar through its selected implementation. -/
roundoff theorem coefficient_payload_error :=
  fun (input : ConfiguredValue) (_hformat : format.isIEEE = true)
    (found : (@Numeric.QuotientArithmetic.encode ConfiguredValue
      (Numeric.QuotientArithmetic.configured (format := format) (plan := plan) (code := code))
      (input * input)).isSome) =>
    ((Numeric.QuotientCoefficients.primal ((@Numeric.QuotientArithmetic.encode ConfiguredValue
      (Numeric.QuotientArithmetic.configured (format := format) (plan := plan) (code := code))
      (input * input)).get found) : Rat) : Real)

-- An arbitrary optional rational is still an unknown computation after exact transport.
example (_source : Option Rat) (_found : _source.isSome) : True := by
  fail_if_success
    have missing := by roundoff (((_source.get _found : Rat) : Real))
  trivial

-- Success of the decoder supplies no numerical law for the function that produced its input.
example (_unknown : ConfiguredValue → ConfiguredValue) (_input : ConfiguredValue)
    (_found : (ExecFloat.Binary.toRat? (_unknown _input)).isSome) : True := by
  fail_if_success
    have missing := by
      roundoff ((((ExecFloat.Binary.toRat? (_unknown _input)).get _found : Rat) : Real))
  trivial

-- Both positive proofs must still use the source floating-point multiplication bound.
open Lean Meta Elab Command in
run_elab do
  for name in #[``rational_payload_error, ``coefficient_payload_error] do
    let some (.thmInfo info) := (← getEnv).find? name
      | throwError "missing successful-payload numerical theorem: {name}"
    let proof ← withoutExporting <| withDefault <| whnf info.value
    unless (proof.find? fun expression =>
        expression.isConstOf ``ExecFloat.Binary.abs_toReal_mul_sub_le ||
        expression.isConstOf ``Model.abs_toReal_mul_sub_le).isSome do
      throwError "payload observation discarded its source multiplication bound: {name}"

namespace ScalarSources

open TorchLean.Numeric

/-- An established guard permits verification of the reachable branch alone. The unknown
callback is deliberately outside the numerical rule set. -/
theorem known_branch_error (p : Prop) [Decidable p] (hp : p)
    (input : ConfiguredValue) (unknown : ConfiguredValue → ConfiguredValue)
    (hformat : format.isIEEE = true) (hinput : ExecFloat.Binary.isFinite input = true)
    (houtput : ExecFloat.Binary.isFinite (input * input) = true) :
    |ExecFloat.Binary.toReal (if p then input * input else unknown input) -
        ExecFloat.Binary.toReal input * ExecFloat.Binary.toReal input| ≤
      Model.epsilonAt format (ExecFloat.Binary.toReal input * ExecFloat.Binary.toReal input) := by
  roundoff

/-- Without a proved guard, an unsupported alternative cannot disappear. -/
example (_p : Prop) [Decidable _p] (_input : ConfiguredValue)
    (_unknown : ConfiguredValue → ConfiguredValue) (_hformat : format.isIEEE = true)
    (_hinput : ExecFloat.Binary.isFinite _input = true)
    (_houtput : ExecFloat.Binary.isFinite (_input * _input) = true) : True := by
  fail_if_success
    have bound := by
      roundoff (ExecFloat.Binary.toReal (if _p then _input * _input else _unknown _input))
  trivial

/-- Ordinary predicate aliases remain usable as range evidence. -/
def finiteAlias (input : ConfiguredValue) : Bool := ExecFloat.Binary.isFinite input

theorem alias_square_error (input : ConfiguredValue) (hformat : format.isIEEE = true)
    (hinput : finiteAlias input = true) (houtput : finiteAlias (input * input) = true) :
    |ExecFloat.Binary.toReal (input * input) -
        ExecFloat.Binary.toReal input * ExecFloat.Binary.toReal input| ≤
      Model.epsilonAt format (ExecFloat.Binary.toReal input * ExecFloat.Binary.toReal input) := by
  roundoff

roundoff theorem computed_get_error :=
  fun (input : ConfiguredValue) (flag : Bool) (_hformat : format.isIEEE = true)
    (found : (if flag then some (input * input) else some (input + input)).isSome) =>
    ExecFloat.Binary.toReal
      ((if flag then some (input * input) else some (input + input)).get found)

example (_producer : ConfiguredValue → Option ConfiguredValue) (_input : ConfiguredValue)
    (_found : (_producer _input).isSome) : True := by
  fail_if_success
    have bound := by
      roundoff (ExecFloat.Binary.toReal ((_producer _input).get _found))
  trivial

open Lean Meta Elab Command in
run_elab do
  let some (.thmInfo info) := (← getEnv).find? ``computed_get_error
    | throwError "missing successful Option source theorem"
  let proof ← withoutExporting <| withDefault <| whnf info.value
  let names := proof.getUsedConstantsAsSet
  for rule in #[``ExecFloat.Binary.abs_toReal_mul_sub_le,
      ``ExecFloat.Binary.abs_toReal_add_sub_le] do
    unless names.contains rule do throwError "optional source omitted {rule}"

/-- A decoder with its own floating-point computation. -/
@[expose, instance_reducible] def squareDecoder (input : ConfiguredValue) :
    QuotientArithmetic ConfiguredValue :=
  { QuotientArithmetic.configured with decode := fun _ => some (input * input) }

theorem selected_decoder_error (input : ConfiguredValue) (coefficients : QuotientCoefficients 0)
    (found : (@QuotientArithmetic.decode ConfiguredValue (squareDecoder input) coefficients).isSome)
    (hformat : format.isIEEE = true) (hinput : ExecFloat.Binary.isFinite input = true)
    (houtput : ExecFloat.Binary.isFinite (input * input) = true) :
    |ExecFloat.Binary.toReal
        ((@QuotientArithmetic.decode ConfiguredValue
          (squareDecoder input) coefficients).get found) -
        ExecFloat.Binary.toReal input * ExecFloat.Binary.toReal input| ≤
      Model.epsilonAt format (ExecFloat.Binary.toReal input * ExecFloat.Binary.toReal input) := by
  roundoff

example (_decode : QuotientCoefficients 0 → Option ConfiguredValue) (_value : Rat)
    (_hformat : format.isIEEE = true)
    (_hrounded : ExecFloat.Binary.isFinite
      (ExecFloat.Binary.ofModel (Model.ofRat format _value) : ConfiguredValue) = true)
    (_found : (@QuotientArithmetic.decode ConfiguredValue
      { QuotientArithmetic.configured with decode := _decode } (.scalar _value)).isSome) :
    True := by
  fail_if_success
    have bound := by
      roundoff (ExecFloat.Binary.toReal
        ((@QuotientArithmetic.decode ConfiguredValue
          { QuotientArithmetic.configured with decode := _decode } (.scalar _value)).get _found))
  trivial

end ScalarSources

/-- An exact outer operation still retains the premises of a selected input bound. -/
example (condition : Prop) (value : ℝ)
    (_hinput : condition → |value - value| ≤ 0) : True := by
  fail_if_success
    have bound := by roundoff (max value 0)
  trivial

example (condition : Prop) (value : ℝ) (_hcondition : condition)
    (_hinput : condition → |value - value| ≤ 0) :
    |max value 0 - max value 0| ≤ 0 := by
  have bound := by roundoff (max value 0)
  exact bound

/-- Selection remains stable when rounding changes which operand is larger. -/
theorem uncertain_max_error (left right : ConfiguredValue) (x y firstError secondError : ℝ)
    (hleft : ExecFloat.Binary.isFinite left = true)
    (hright : ExecFloat.Binary.isFinite right = true)
    (hx : |ExecFloat.Binary.toReal left - x| ≤ firstError)
    (hy : |ExecFloat.Binary.toReal right - y| ≤ secondError) :
    |ExecFloat.Binary.toReal (max left right) - max x y| ≤ max firstError secondError := by
  roundoff

/-- A custom real numeral may contain floating-point work; its spelling does not erase it. -/
example (input : ConfiguredValue) (hformat : format.isIEEE = true)
    (hinput : ExecFloat.Binary.isFinite input = true)
    (hresult : ExecFloat.Binary.isFinite (input * input) = true) :
    let _ : OfNat ℝ 1 := ⟨ExecFloat.Binary.toReal (input * input)⟩
    |(1 : ℝ) - ExecFloat.Binary.toReal input * ExecFloat.Binary.toReal input| ≤
      Model.epsilonAt format (ExecFloat.Binary.toReal input * ExecFloat.Binary.toReal input) := by
  roundoff

/-- Exact max/abs selection retains the rounding in the tensor that supplied its entries. -/
roundoff theorem max_abs_square_error :=
  fun [Storage ConfiguredValue] {shape : Spec.Shape} (input : Tensor ConfiguredValue shape) =>
    ExecFloat.Binary.toReal (Tensor.maxAbs (Tensor.squareSpec input))

/-- Lean 4.35's lowered record branch retains its decision and branch-local rounding. -/
theorem reflected_record_error (input : ConfiguredValue) (condition : Prop)
    [decision : Decidable condition]
    (hformat : format.isIEEE = true) (hinput : ExecFloat.Binary.isFinite input = true)
    (hproduct : condition → ExecFloat.Binary.isFinite (input * input) = true) :
    |(let pair : ConfiguredValue × ConfiguredValue :=
        Bool.rec (motive := fun bit => bit.Reflects condition → ConfiguredValue × ConfiguredValue)
          (fun _ => (input, input)) (fun _ => (input * input, input))
          (@decide condition decision) (@Decidable.reflects_decide condition decision)
      ExecFloat.Binary.toReal pair.1) -
        (if condition then ExecFloat.Binary.toReal input * ExecFloat.Binary.toReal input
         else ExecFloat.Binary.toReal input)| ≤
      if condition then Model.epsilonAt format
        (ExecFloat.Binary.toReal input * ExecFloat.Binary.toReal input) else 0 := by
  roundoff

namespace BufferReads

open TorchLean.Tensor.Internal

/-- A generated buffer retains the arithmetic in its entry function. -/
def storedSquare [Storage ConfiguredValue] {shape : Spec.Shape}
    (input : Tensor ConfiguredValue shape) : Tensor ConfiguredValue shape where
  buffer := Storage.ofFn fun index : Fin (TorchLean.Tensor.Internal.Shape.size shape) =>
    let value := input (Coord.unlinearize index)
    value * value
  size_eq := Storage.size_ofFn _

roundoff theorem stored_square_error :=
  fun [Storage ConfiguredValue] {shape : Spec.Shape}
    (input : Tensor ConfiguredValue shape) (coordinate : Coord shape) =>
    ExecFloat.Binary.toReal ((storedSquare input) coordinate)

roundoff theorem flat_square_error :=
  fun [Storage ConfiguredValue] {shape : Spec.Shape} (input : Tensor ConfiguredValue shape)
    (index : Fin (TorchLean.Tensor.Internal.Shape.size shape)) =>
    ExecFloat.Binary.toReal ((storedSquare input).getFlat index)

/-- Reduction callbacks can retain a definitionally equal shape in their coordinate type. -/
def erasedAxisSquare [Storage ConfiguredValue] {rows columns : Nat}
    (input : Tensor ConfiguredValue [rows]) :
    Tensor ConfiguredValue
      (Tensor.shapeAfterSum [rows, columns] (Spec.Shape.rank [rows, columns] - 1)) where
  buffer := Storage.ofFn fun index : Fin (TorchLean.Tensor.Internal.Shape.size [rows]) =>
    let value := input (Coord.unlinearize index)
    value * value
  size_eq := by
    rw [Storage.size_ofFn]
    rfl

roundoff theorem erased_axis_square_error :=
  fun [Storage ConfiguredValue] {rows columns : Nat} (input : Tensor ConfiguredValue [rows])
    (row : Fin rows) (column : Fin columns) =>
    ExecFloat.Binary.toReal
      (Rep.get (erasedAxisSquare (columns := columns) input)
        (row, ((Tensor.Internal.axisCoordinateEquiv 0
          (shape := Spec.Shape.dim columns Spec.Shape.scalar)) (column, PUnit.unit)).2))

open Lean Meta Elab Command in
run_elab do
  for name in #[``stored_square_error, ``flat_square_error, ``erased_axis_square_error] do
    let some (.thmInfo info) := (← getEnv).find? name
      | throwError "missing generated-buffer numerical theorem"
    let proof ← withoutExporting <| withDefault <| whnf info.value
    unless (proof.find? fun expression =>
        expression.isConstOf ``ExecFloat.Binary.abs_toReal_mul_sub_le ||
        expression.isConstOf ``Model.abs_toReal_mul_sub_le).isSome do
      throwError "the buffer read discarded its callback's floating arithmetic"

/-- Unknown entry functions require their own numerical evidence. -/
def storedApply [Storage ConfiguredValue] {shape : Spec.Shape}
    (compute : ConfiguredValue → ConfiguredValue) (input : Tensor ConfiguredValue shape) :
    Tensor ConfiguredValue shape where
  buffer := Storage.ofFn fun index : Fin (TorchLean.Tensor.Internal.Shape.size shape) =>
    compute (input (Coord.unlinearize index))
  size_eq := Storage.size_ofFn _

example [Storage ConfiguredValue] {shape : Spec.Shape}
    (_compute : ConfiguredValue → ConfiguredValue)
    (_input : Tensor ConfiguredValue shape) (_coordinate : Coord shape) : True := by
  fail_if_success
    have bound := by
      roundoff (ExecFloat.Binary.toReal ((storedApply _compute _input) _coordinate))
  trivial

-- No storage theorem applies to an arbitrary buffer solely because it is in a Rep record.
example [storage : Storage ConfiguredValue] {shape : Spec.Shape} (_buffer : storage.Buffer)
    (_size : storage.size _buffer = TorchLean.Tensor.Internal.Shape.size shape)
    (_coordinate : Coord shape) : True := by
  fail_if_success
    have bound := by
      roundoff (ExecFloat.Binary.toReal (Rep.get (Rep.mk _buffer _size) _coordinate))
  trivial

-- Flat reads of an arbitrary buffer also retain the missing producer evidence.
example [storage : Storage ConfiguredValue] {shape : Spec.Shape} (_buffer : storage.Buffer)
    (_size : storage.size _buffer = TorchLean.Tensor.Internal.Shape.size shape)
    (_index : Fin (TorchLean.Tensor.Internal.Shape.size shape)) : True := by
  fail_if_success
    have bound := by
      roundoff (ExecFloat.Binary.toReal (Rep.getFlat (Rep.mk _buffer _size) _index))
  trivial

end BufferReads

/-- A named literal for checking helper selection through an ordinary definition. -/
def zeroValue : ConfiguredValue := 0

/-- Helper selection preserves ordinary value aliases without evaluating unrelated arithmetic. -/
theorem named_zero_product_error (input : ConfiguredValue) (hformat : format.isIEEE = true)
    (hinput : ExecFloat.Binary.isFinite input = true)
    (houtput : ExecFloat.Binary.isFinite (zeroValue * input) = true) :
    |ExecFloat.Binary.toReal (zeroValue * input) - 0| ≤
      Model.epsilonAt format (ExecFloat.Binary.toReal (zeroValue : ConfiguredValue) *
        ExecFloat.Binary.toReal input) := by
  roundoff

/-- The public LayerNorm computation retains both centering passes and its ordered reductions. -/
roundoff theorem layer_norm_forward_error :=
  fun [Storage ConfiguredValue] {rows columns : Nat}
    (input : Tensor ConfiguredValue [rows, columns])
    (scale bias : Tensor ConfiguredValue [columns]) (epsilon : ConfiguredValue)
    (rowsPositive : 0 < rows) (columnsPositive : 0 < columns)
    (row : Fin rows) (column : Fin columns) =>
  ExecFloat.Binary.toReal
    (Spec.get2 (Spec.layerNorm input scale bias rowsPositive columnsPositive epsilon) row column)

open Lean Meta Elab Command in
run_elab do
  let some (.thmInfo info) := (← getEnv).find? ``layer_norm_forward_error
    | throwError "missing LayerNorm numerical theorem"
  let proof ← withoutExporting <| withDefault <| whnf info.value
  for rule in #[``Proofs.RuntimeApprox.Roundoff.fold_error,
      ``Proofs.RuntimeApprox.Roundoff.sqrt_error] do
    unless (proof.find? (·.isConstOf rule)).isSome do
      throwError "the LayerNorm proof omitted {rule}"
  for (direct, model) in #[
      (``ExecFloat.Binary.abs_toReal_div_sub_le, ``Model.abs_toReal_div_sub_le),
      (``ExecFloat.Binary.abs_toReal_sub_sub_le, ``Model.abs_toReal_sub_sub_le)] do
    unless (proof.find? fun expression =>
        expression.isConstOf direct || expression.isConstOf model).isSome do
      throwError "the LayerNorm proof omitted {direct}"

/-- The public RMSNorm layer retains symbolic dimensions and configured scalar arithmetic. -/
roundoff theorem rms_norm_forward_error :=
  fun [Storage ConfiguredValue] {rows columns : Nat}
    (input : Tensor ConfiguredValue [rows, columns])
    (scale : Tensor ConfiguredValue [columns]) (epsilon : ConfiguredValue)
    (rowsPositive : 0 < rows) (columnsPositive : 0 < columns)
    (row : Fin rows) (column : Fin columns) =>
  ExecFloat.Binary.toReal
    (Spec.get2 (Spec.rmsNorm input scale rowsPositive columnsPositive epsilon) row column)

/-- Both vector normalization schedules retain their existing scaling and ordered reduction. -/
roundoff theorem normalized_vector_error :=
  fun [Storage ConfiguredValue] {width : Nat}
    (input : Tensor ConfiguredValue [width]) (epsilon : ConfiguredValue)
    (mean : Bool) (coordinate : Fin width) =>
    ExecFloat.Binary.toReal
      ((Spec.normalizeL2RegularizedSpec input epsilon mean).getScalar coordinate)

/-- Batch dimensions and broadcasting remain symbolic; each dot product keeps its left fold. -/
roundoff theorem batched_matmul_forward_error :=
  fun [Storage ConfiguredValue] {leftBatch rightBatch batch : Spec.Shape} {m n p : Nat}
    (leftBroadcast : Spec.Shape.CanBroadcastTo leftBatch batch)
    (rightBroadcast : Spec.Shape.CanBroadcastTo rightBatch batch)
    (left : Tensor ConfiguredValue (leftBatch.concat [m, n]))
    (right : Tensor ConfiguredValue (rightBatch.concat [n, p]))
    (index : (batch.concat [m, p]).Coord) =>
    ExecFloat.Binary.toReal
      ((Spec.matmulRightOp leftBroadcast rightBroadcast right).forward left index)

/-- Both public input gradients retain an arbitrary batch rank when the batch shapes agree. -/
roundoff theorem common_batch_right_input_gradient_error :=
  fun [Storage ConfiguredValue] {batch : Spec.Shape} {m n p : Nat}
    (left : Tensor ConfiguredValue (batch.concat [m, n]))
    (right : Tensor ConfiguredValue (batch.concat [n, p]))
    (upstream : Tensor ConfiguredValue (batch.concat [m, p]))
    (index : (batch.concat [n, p]).Coord) =>
    ExecFloat.Binary.toReal
      ((Spec.matmulLeftOp (Spec.Shape.CanBroadcastTo.refl batch)
        (Spec.Shape.CanBroadcastTo.refl batch) left).backward right upstream index)

roundoff theorem common_batch_left_input_gradient_error :=
  fun [Storage ConfiguredValue] {batch : Spec.Shape} {m n p : Nat}
    (left : Tensor ConfiguredValue (batch.concat [m, n]))
    (right : Tensor ConfiguredValue (batch.concat [n, p]))
    (upstream : Tensor ConfiguredValue (batch.concat [m, p]))
    (index : (batch.concat [m, n]).Coord) =>
    ExecFloat.Binary.toReal
      ((Spec.matmulRightOp (Spec.Shape.CanBroadcastTo.refl batch)
        (Spec.Shape.CanBroadcastTo.refl batch) right).backward left upstream index)

/-- A broadcast operand's existing pullback accumulates contributions from every batch. -/
roundoff theorem broadcast_matmul_pullback_error :=
  fun [Storage ConfiguredValue] {b m n p : Nat}
    (left : Tensor ConfiguredValue [b, m, n]) (right : Tensor ConfiguredValue [1, n, p])
    (upstream : Tensor ConfiguredValue [b, m, p]) (index : Spec.Shape.Coord [1, n, p]) =>
    ExecFloat.Binary.toReal
      ((Spec.matmulLeftOp (Spec.Shape.CanBroadcastTo.refl [b])
        (Spec.Shape.CanBroadcastTo.dim_1_to_n Spec.Shape.CanBroadcastTo.scalar) left).backward
          right upstream index)

/-- Reducing a computed tensor includes its pointwise arithmetic, for any valid axis. -/
roundoff theorem axis_square_sum_error :=
  fun [Storage ConfiguredValue] {shape : Spec.Shape} (axis : Nat)
    [Spec.Shape.AxisInBounds axis shape] (input : Tensor ConfiguredValue shape)
    (index : (Tensor.shapeAfterSum shape axis).Coord) =>
    ExecFloat.Binary.toReal
      (Tensor.reduceDim Tensor.sumSpec axis (Tensor.squareSpec input) index)

/-- The existing reduction pullback copies the upstream entries along the removed axis. -/
roundoff theorem axis_sum_pullback_error :=
  fun [Storage ConfiguredValue] {shape : Spec.Shape} (axis : Nat)
    [Spec.Shape.HasNonemptyAxis axis shape] [Spec.Shape.WellFormed shape]
    (input : Tensor ConfiguredValue shape)
    (upstream : Tensor ConfiguredValue (Tensor.shapeAfterSum shape axis))
    (index : shape.Coord) =>
    ExecFloat.Binary.toReal ((Spec.reduceSumOp axis).backward input upstream index)

/-- Public means retain the arithmetic that produced each entry and the rounded denominator. -/
roundoff theorem axis_square_mean_error :=
  fun [Storage ConfiguredValue] {shape : Spec.Shape} (axis : Nat)
    (nonempty : Spec.Shape.NonemptyAxis axis shape) (input : Tensor ConfiguredValue shape)
    (index : (Tensor.shapeAfterSum shape axis).Coord) =>
    ExecFloat.Binary.toReal
      (Tensor.reduceMean axis (Tensor.squareSpec input) nonempty index)

/-- Captured parameter fields are inputs; their arithmetic still uses the selected scalar. -/
roundoff theorem linear_forward_error :=
  fun [Storage ConfiguredValue] {inDim outDim : Nat}
    (layer : Spec.LinearSpec ConfiguredValue inDim outDim)
    (input : Tensor ConfiguredValue [inDim]) (index : Fin outDim) =>
    ExecFloat.Binary.toReal (((Spec.linearOp layer).forward input).getScalar index)

/-- Mean-loss analysis includes the cast of the tensor-size denominator. -/
roundoff theorem mse_forward_error :=
  fun [Storage ConfiguredValue] {shape : Spec.Shape}
    (input target : Tensor ConfiguredValue shape) =>
    ExecFloat.Binary.toReal (((Spec.mseLossOp target).forward input).item)

/-- A nonlinear pullback composes the existing forward and backward code. The interval law
accounts for the actual cosh result and the input error from squaring. -/
roundoff theorem sinh_square_gradient_error :=
  fun [Storage ConfiguredValue] {shape : Spec.Shape}
    (input upstream : Tensor ConfiguredValue shape) (index : shape.Coord) =>
    ExecFloat.Binary.toReal
      ((Spec.squareOp.compose Spec.sinhOp).backward input upstream index)

open Lean Elab Command in
run_elab do
  let some (.thmInfo info) := (← getEnv).find? ``mse_forward_error
    | throwError "missing numerical mean-loss theorem"
  let proof ← Lean.withoutExporting <| Lean.Meta.withDefault <| Lean.Meta.whnf info.value
  unless (proof.find? (·.isConstOf
      ``Proofs.RuntimeApprox.Roundoff.model_ofRat_error)).isSome do
    throwError "the mean-loss proof omitted its denominator's conversion error"
  let some (.thmInfo gradient) := (← getEnv).find? ``sinh_square_gradient_error
    | throwError "missing numerical nonlinear-gradient theorem"
  let proof ← Lean.withoutExporting <| Lean.Meta.withDefault <| Lean.Meta.whnf gradient.value
  for rule in #[``Proofs.RuntimeApprox.Roundoff.model_enclosure_error,
      ``Proofs.RuntimeApprox.Roundoff.cosh_error] do
    unless (proof.find? (·.isConstOf rule)).isSome do
      throwError "the nonlinear-gradient proof omitted {rule}"
  let some (.thmInfo recovered) := (← getEnv).find? ``recovered_coefficient_error
    | throwError "missing numerical recovery theorem"
  let proof ← Lean.withoutExporting <| Lean.Meta.withDefault <| Lean.Meta.whnf recovered.value
  for rule in #[``Proofs.RuntimeApprox.Roundoff.enclosedSlope_get_error,
      ``ExecFloat.Binary.abs_toReal_mul_sub_le] do
    unless (proof.find? (·.isConstOf rule)).isSome do
      throwError "the recovery proof omitted {rule}"
  let some (.thmInfo maximum) := (← getEnv).find? ``max_abs_square_error
    | throwError "missing numerical max-absolute-value theorem"
  let proof ← Lean.withoutExporting <| Lean.Meta.withDefault <| Lean.Meta.whnf maximum.value
  unless (proof.find? (·.isConstOf ``ExecFloat.Binary.abs_toReal_mul_sub_le)).isSome do
    throwError "the max-absolute-value proof omitted the input multiplication"
  let some (.thmInfo options) := (← getEnv).find? ``computed_options_error
    | throwError "missing multiple-Option numerical theorem"
  let proof ← Lean.withoutExporting <| Lean.Meta.withDefault <| Lean.Meta.whnf options.value
  unless (proof.find? (·.isConstOf ``ExecFloat.Binary.abs_toReal_mul_sub_le)).isSome do
    throwError "the optional producers lost the input multiplication"

/-- Storing a tensor beside a function supplies no numerical law for that function. -/
example {shape : Spec.Shape}
    (_functions : Tensor ConfiguredValue shape × (ConfiguredValue → ConfiguredValue))
    (_input : ConfiguredValue) : True := by
  fail_if_success
    have bound := by roundoff (ExecFloat.Binary.toReal (_functions.2 _input))
  trivial

private noncomputable def decodedSquare (input : ConfiguredValue) : ℝ :=
  ExecFloat.Binary.toReal (input * input)

private theorem decoded_square_echo (input : ConfiguredValue) (reference error : ℝ)
    (bound : |decodedSquare input - reference| ≤ error) :
    |decodedSquare input - reference| ≤ error := bound

attribute [local roundoff_rule] decoded_square_echo

/-- Registering a restatement of an unproved bound cannot turn an entire program into an
exact scalar decoder. Its multiplication still needs numerical evidence. -/
example (_input : ConfiguredValue) : True := by
  fail_if_success
    have bound := by roundoff (decodedSquare _input)
  trivial

/-- The existing padded window retains the input bound at valid indices and the padding bound
otherwise. Sizes and offsets remain symbolic; the bounds proof belongs to the selected branch. -/
theorem window_error [Storage ConfiguredValue] {n length : Nat}
    (input : Tensor ConfiguredValue [n]) (reference : Tensor ℝ [n])
    (pad : ConfiguredValue) (padReference inputError padError : ℝ)
    (offset : Nat) (index : Fin length)
    (hinput : ∀ i,
      |ExecFloat.Binary.toReal (input.getScalar i) - reference.getScalar i| ≤ inputError)
    (hpad : ¬ (offset + index.val < n) →
      |ExecFloat.Binary.toReal pad - padReference| ≤ padError) :
    |ExecFloat.Binary.toReal ((Tensor.window input length offset pad).getScalar index) -
        (Tensor.window reference length offset padReference).getScalar index| ≤
      if offset + index.val < n then inputError else padError := by
  roundoff

/-- A selected indexing instance must be followed, rather than assumed to read one entry. -/
example [Storage ConfiguredValue] {n : Nat}
    (_input : Tensor ConfiguredValue [n]) (_index : Fin n) : True := by
  let : GetElem (Tensor ConfiguredValue [n]) (Fin n) ConfiguredValue (fun _ _ => True) :=
    ⟨fun values i _ => values.getScalar i * values.getScalar i⟩
  fail_if_success
    have : |ExecFloat.Binary.toReal _input[_index] -
        ExecFloat.Binary.toReal (_input.getScalar _index)| ≤ 0 := by
      roundoff
  trivial

/-- A tensor receiver does not make an arbitrary function coercion an exact tensor read. -/
example {shape : Spec.Shape} (_input : Tensor ConfiguredValue shape) (_index : shape.Coord)
    [_coercion : CoeFun (Tensor ConfiguredValue shape)
      (fun _ => shape.Coord → ConfiguredValue)] : True := by
  fail_if_success
    have bound := by
      roundoff (ExecFloat.Binary.toReal
        (@CoeFun.coe (Tensor ConfiguredValue shape) _ _coercion _input _index))
  trivial

/-- A conditional input-error theorem cannot certify the opposite branch. -/
example (condition : Prop) [Decidable condition] (input : ConfiguredValue) (reference error : ℝ)
    (_hinput : condition → |ExecFloat.Binary.toReal input - reference| ≤ error) : True := by
  fail_if_success
    have : |ExecFloat.Binary.toReal (if condition then input else input) -
        (if condition then reference else reference)| ≤
        if condition then error else error := by
      roundoff
  trivial

/-- Bound the actual matrix pullback against its real-valued implementation, including errors
in the incoming gradient and captured weights. Each product and addition retains its own rounding
term. Range assumptions concern the accumulation states actually visited. -/
theorem matmul_pullback_error [Storage ConfiguredValue] {m n p : Nat}
    (input : Tensor ConfiguredValue [m, n]) (seed : Tensor ConfiguredValue [m, p])
    (right : Tensor ConfiguredValue [n, p])
    (seedReference : Tensor ℝ [m, p]) (rightReference : Tensor ℝ [n, p])
    (seedError rightError : ℝ)
    (hseed : ∀ index : Spec.Shape.Coord [m, p],
      |ExecFloat.Binary.toReal (seed index) - seedReference index| ≤ seedError)
    (hright : ∀ index : Spec.Shape.Coord [n, p],
      |ExecFloat.Binary.toReal (right index) - rightReference index| ≤ rightError)
    (row : Fin m) (column : Fin n) (hformat : format.isIEEE = true)
    (hvalid : AllSteps
      (fun state (k : Spec.Shape.Coord [p]) =>
        state + Spec.get2 seed row k.1 * Spec.get2 right column k.1)
      (fun state k => ExecFloat.Binary.isFinite state = true ∧
        ExecFloat.Binary.isFinite (Spec.get2 seed row k.1 * Spec.get2 right column k.1) = true ∧
        ExecFloat.Binary.isFinite
          (state + Spec.get2 seed row k.1 * Spec.get2 right column k.1) = true ∧
        ExecFloat.Binary.isFinite (Spec.get2 seed row k.1) = true ∧
        ExecFloat.Binary.isFinite (Spec.get2 right column k.1) = true)
      0 (Tensor.Internal.Rep.coordinates [p])) :
    |ExecFloat.Binary.toReal
        (Spec.get2 ((Spec.matmulRightOp (batchA := []) (batchB := []) (batch := [])
          (Spec.Shape.CanBroadcastTo.refl []) (Spec.Shape.CanBroadcastTo.refl []) right).backward
          input seed) row column) -
      Spec.get2 ((Spec.matmulRightOp (batchA := []) (batchB := []) (batch := [])
          (Spec.Shape.CanBroadcastTo.refl []) (Spec.Shape.CanBroadcastTo.refl [])
          rightReference).backward (input.map ExecFloat.Binary.toReal) seedReference) row column| ≤
      foldError
        (fun state (k : Spec.Shape.Coord [p]) =>
          state + Spec.get2 seed row k.1 * Spec.get2 right column k.1)
        (fun state k => state + Spec.get2 seedReference row k.1 *
          Spec.get2 rightReference column k.1)
        (fun state _ error k => error +
          (|ExecFloat.Binary.toReal (Spec.get2 right column k.1)| * seedError +
            |Spec.get2 seedReference row k.1| * rightError +
            Model.epsilonAt format (ExecFloat.Binary.toReal (Spec.get2 seed row k.1) *
              ExecFloat.Binary.toReal (Spec.get2 right column k.1))) +
          Model.epsilonAt format (ExecFloat.Binary.toReal state +
            ExecFloat.Binary.toReal (Spec.get2 seed row k.1 * Spec.get2 right column k.1)))
        0 0 0 (Tensor.Internal.Rep.coordinates [p]) := by
  roundoff

/-- Composition reuses the existing pullbacks, including an uncertain incoming cotangent. -/
theorem composed_pullback_error [Storage ConfiguredValue] {shape : Spec.Shape}
    (input seed : Tensor ConfiguredValue shape) (factor : ConfiguredValue)
    (i : shape.Coord) (reference error : ℝ) (hformat : format.isIEEE = true)
    (hseed : ExecFloat.Binary.isFinite (seed i) = true)
    (hfactor : ExecFloat.Binary.isFinite factor = true)
    (hresult : ExecFloat.Binary.isFinite (seed i * factor) = true)
    (hinput : |ExecFloat.Binary.toReal (seed i) - reference| ≤ error) :
    |ExecFloat.Binary.toReal
        ((Spec.negOp.compose (Spec.scaleOp factor)).backward input seed i) -
        -(reference * ExecFloat.Binary.toReal factor)| ≤
      |ExecFloat.Binary.toReal factor| * error +
        Model.epsilonAt format (ExecFloat.Binary.toReal (seed i) *
          ExecFloat.Binary.toReal factor) := by
  roundoff

/-- Negating an existing pullback introduces no rounding error, for any configured codec.
The input cotangent may already have an error bound. -/
theorem negation_pullback_error [Storage ConfiguredValue] {shape : Spec.Shape}
    (input seed : Tensor ConfiguredValue shape) (reference error : shape.Coord → ℝ)
    (hfinite : ∀ i, ExecFloat.Binary.isFinite (seed i) = true)
    (hseed : ∀ i, |ExecFloat.Binary.toReal (seed i) - reference i| ≤ error i) :
    ∀ i, |ExecFloat.Binary.toReal (Spec.negOp.backward input seed i) - -reference i| ≤ error i := by
  roundoff

/-- An exact-operation theorem cannot certify a different selected negation implementation. -/
example (x : ConfiguredValue) (_hx : ExecFloat.Binary.isFinite x = true) : True := by
  let : Neg ConfiguredValue := ⟨id⟩
  fail_if_success
    have : |ExecFloat.Binary.toReal (-x) - -ExecFloat.Binary.toReal x| ≤ 0 := by
      roundoff
  trivial

/-- Analyze generated entries and their accumulation together. Each product and addition
contributes its own rounding term; the tensor length and storage codec remain parameters. -/
theorem product_sum_error [Storage ConfiguredValue] {n : Nat}
    (x y : Tensor ConfiguredValue [n]) (initial : ConfiguredValue)
    (hformat : format.isIEEE = true)
    (hx : ∀ i, ExecFloat.Binary.isFinite (x.getScalar i) = true)
    (hy : ∀ i, ExecFloat.Binary.isFinite (y.getScalar i) = true)
    (hvalid : AllSteps (fun a b : ConfiguredValue => a + b)
      (fun a b => ExecFloat.Binary.isFinite a = true ∧ ExecFloat.Binary.isFinite b = true ∧
        ExecFloat.Binary.isFinite (a + b) = true) initial
      (Tensor.ofFn fun i => x.getScalar i * y.getScalar i)) :
    |ExecFloat.Binary.toReal
        ((Tensor.ofFn fun i => x.getScalar i * y.getScalar i).foldl (· + ·) initial) -
        (Tensor.ofFn fun i => ExecFloat.Binary.toReal (x.getScalar i) *
          ExecFloat.Binary.toReal (y.getScalar i)).foldl (· + ·)
            (ExecFloat.Binary.toReal initial)| ≤
      foldError (fun a (pair : ConfiguredValue × ConfiguredValue) => a + pair.1 * pair.2)
        (fun a pair => a + ExecFloat.Binary.toReal pair.1 * ExecFloat.Binary.toReal pair.2)
        (fun a _ error pair => error +
          Model.epsilonAt format (ExecFloat.Binary.toReal pair.1 * ExecFloat.Binary.toReal pair.2) +
          Model.epsilonAt format
            (ExecFloat.Binary.toReal a + ExecFloat.Binary.toReal (pair.1 * pair.2)))
        initial (ExecFloat.Binary.toReal initial) 0
        (Tensor.ofFn fun i => (x.getScalar i, y.getScalar i)) := by
  roundoff

/-- Square-root input error is propagated even at zero, where a derivative bound would need
a positive margin. The floating-point operation still needs its finite, nonnegative domain. -/
theorem uncertain_sqrt_error (value : ConfiguredValue) (reference inputError : ℝ)
    (hformat : format.isIEEE = true)
    (hfinite : ExecFloat.Binary.isFinite value = true)
    (hdomain : ExecFloat.Binary.isZero value = true ∨ ExecFloat.Binary.signBit value = false)
    (hinput : |ExecFloat.Binary.toReal value - reference| ≤ inputError) :
    |ExecFloat.Binary.toReal (MathFunctions.sqrt value) - Real.sqrt reference| ≤
      Real.sqrt inputError +
        Model.epsilonAt format (Real.sqrt (ExecFloat.Binary.toReal value)) := by
  roundoff

/-- Decode a full reduction without unrolling its tensor or changing the accumulation order.
Exact model decoding also covers non-IEEE encodings and arbitrary storage codecs. -/
theorem decoded_sum {shape : Spec.Shape} (tensor : Tensor ConfiguredValue shape) :
    ExecFloat.Binary.toModel (Tensor.sumSpec tensor) =
      Tensor.foldlSpec Model.add (ExecFloat.Binary.toModel (0 : ConfiguredValue))
        (tensor.map ExecFloat.Binary.toModel) := by
  roundoff

/-- The fold rule handles a nonlinear scalar step, including its recurrence on the accumulator.
`Model.exp` retains FloatLib's actual approximation; it is not replaced with `Real.exp`. -/
theorem decoded_recurrence {shape : Spec.Shape} (tensor : Tensor ConfiguredValue shape)
    (initial : ConfiguredValue) :
    ExecFloat.Binary.toModel
        (Tensor.foldlSpec (fun state value => state * state + MathFunctions.exp value)
          initial tensor) =
      Tensor.foldlSpec (fun state value => Model.add (Model.mul state state) (Model.exp value))
        (ExecFloat.Binary.toModel initial) (tensor.map ExecFloat.Binary.toModel) := by
  roundoff

/-- A compensated logarithm keeps its actual discrepancy from a certified reference. -/
theorem log1p_error (x : ConfiguredValue) (hx : ExecFloat.Binary.isFinite x = true)
    (hresult : ExecFloat.Binary.isFinite (Context.log1p x) = true)
    (hsuccess : (Certified.logPlus1 (ExecFloat.Binary.toModel x) {}).isOk = true) :
    |ExecFloat.Binary.toReal (Context.log1p x) - Real.log (1 + ExecFloat.Binary.toReal x)| ≤
      |ExecFloat.Binary.toReal (Context.log1p x) - Model.toReal
          ((Certified.logPlus1 (ExecFloat.Binary.toModel x) {}).toOption.getD
            (Model.zero format false))| +
        Model.epsilonAt format (Real.log (1 + ExecFloat.Binary.toReal x)) := by
  roundoff
/-- When certification returns the actual candidate, the discrepancy term disappears. -/
theorem checked_expm1_error (x : ConfiguredValue) (hx : ExecFloat.Binary.isFinite x = true)
    (hcertificate : Certified.expMinus1 (ExecFloat.Binary.toModel x) {} =
      .ok (ExecFloat.Binary.toModel (Context.expm1 x))) :
    |ExecFloat.Binary.toReal (Context.expm1 x) - (Real.exp (ExecFloat.Binary.toReal x) - 1)| ≤
      Model.epsilonAt format (Real.exp (ExecFloat.Binary.toReal x) - 1) := by
  roundoff

/-- Logarithmic input error retains both positive-domain conditions. -/
theorem uncertain_log_error (x : ConfiguredValue) (reference inputError : ℝ)
    (hx : ExecFloat.Binary.isFinite x = true)
    (hresult : ExecFloat.Binary.isFinite (MathFunctions.log x) = true)
    (hsuccess : (Certified.log (ExecFloat.Binary.toModel x) {}).isOk = true)
    (hpositive : 0 < ExecFloat.Binary.toReal x) (hreference : 0 < reference)
    (hinput : |ExecFloat.Binary.toReal x - reference| ≤ inputError) :
    |ExecFloat.Binary.toReal (MathFunctions.log x) - Real.log reference| ≤
      inputError / min (ExecFloat.Binary.toReal x) reference +
        (|ExecFloat.Binary.toReal (MathFunctions.log x) - Model.toReal
          ((Certified.log (ExecFloat.Binary.toModel x) {}).toOption.getD
            (Model.zero format false))| +
          Model.epsilonAt format (Real.log (ExecFloat.Binary.toReal x))) := by
  roundoff

/-- Analyze the existing exponential pullback, including errors in its input and upstream
cotangent. The selected codec, tensor shape and tensor storage remain parameters. -/
theorem exponential_pullback_error [Storage ConfiguredValue] {shape : Spec.Shape}
    (input seed : Tensor ConfiguredValue shape) (index : shape.Coord)
    (reference seedReference inputError seedError : ℝ)
    (hformat : format.isIEEE = true)
    (hx : ExecFloat.Binary.isFinite (input index) = true)
    (hseed : ExecFloat.Binary.isFinite (seed index) = true)
    (hexp : ExecFloat.Binary.isFinite (MathFunctions.exp (input index)) = true)
    (hresult : ExecFloat.Binary.isFinite (Spec.expOp.backward input seed index) = true)
    (hsuccess : (Certified.exp (ExecFloat.Binary.toModel (input index)) {}).isOk = true)
    (hinput : |ExecFloat.Binary.toReal (input index) - reference| ≤ inputError)
    (hupstream : |ExecFloat.Binary.toReal (seed index) - seedReference| ≤ seedError) :
    |ExecFloat.Binary.toReal (Spec.expOp.backward input seed index) -
        Real.exp reference * seedReference| ≤
      |ExecFloat.Binary.toReal (seed index)| *
        (Real.exp (max (ExecFloat.Binary.toReal (input index)) reference) * inputError +
          (|ExecFloat.Binary.toReal (MathFunctions.exp (input index)) - Model.toReal
            ((Certified.exp (ExecFloat.Binary.toModel (input index)) {}).toOption.getD
              (Model.zero format false))| +
            Model.epsilonAt format (Real.exp (ExecFloat.Binary.toReal (input index))))) +
        |Real.exp reference| * seedError +
          Model.epsilonAt format (ExecFloat.Binary.toReal (MathFunctions.exp (input index)) *
            ExecFloat.Binary.toReal (seed index)) := by
  roundoff

/-- Finite values do not establish that the reference certification succeeded. -/
example (_x : ConfiguredValue) (_hx : ExecFloat.Binary.isFinite _x = true)
    (_hresult : ExecFloat.Binary.isFinite (MathFunctions.exp _x) = true) : True := by
  fail_if_success
    have : |ExecFloat.Binary.toReal (MathFunctions.exp _x) -
          Real.exp (ExecFloat.Binary.toReal _x)| ≤
        |ExecFloat.Binary.toReal (MathFunctions.exp _x) - Model.toReal
          ((Certified.exp (ExecFloat.Binary.toModel _x) {}).toOption.getD
            (Model.zero format false))| +
          Model.epsilonAt format (Real.exp (ExecFloat.Binary.toReal _x)) := by
      roundoff
  trivial

/-- The forward bound is generated from the implementation, including its normalization sum
and `log1p`. It remains available as a reusable rule while checking the composed operator. -/
roundoff theorem log_softmax_forward_error :=
  fun [storage : Storage ConfiguredValue] {n : Nat}
    (input : Tensor ConfiguredValue [n + 1]) (index : Fin (n + 1)) =>
    ExecFloat.Binary.toReal ((Activation.logSoftmaxVecSpec input).getScalar index)

attribute [local roundoff_rule] log_softmax_forward_error

/-- Infer the range conditions, real reference and error budget from the existing log-softmax
VJP, including both branches and their reductions. No separate floating-point VJP is supplied. -/
roundoff theorem log_softmax_pullback_error :=
  fun [storage : Storage ConfiguredValue] {n : Nat}
    (y seed : Tensor ConfiguredValue [n + 1]) (index : Fin (n + 1)) =>
    ExecFloat.Binary.toReal
      ((Activation.Internal.logSoftmaxVecBackwardSpec y seed).getScalar index)

/-- Check the public autograd operator, composing forward error with the backward arithmetic.
The registered forward theorem is reused without substituting another implementation. -/
roundoff theorem log_softmax_input_gradient_error :=
  fun [storage : Storage ConfiguredValue] {n : Nat}
    (input seed : Tensor ConfiguredValue [n + 1]) (index : Fin (n + 1)) =>
    ExecFloat.Binary.toReal (((Spec.logSoftmaxOp 0).backward input seed).getScalar index)

end Configured

/-- A known invalid division cannot be advertised by generating a theorem with false premises. -/
example : True := by
  fail_if_success
    have invalid := roundoff_proof% (fun _ : Unit =>
      ExecFloat.Binary.toReal ((0 : Binary32) / 0))
  fail_if_success
    have invalid := roundoff_proof% (fun _ : Unit =>
      ExecFloat.Binary.toReal ((0 : Binary32) / 0) + 0)
  trivial

/-- A finite candidate cannot conceal a nonfinite input at the interval boundary. -/
example : True := by
  let ordinary : MathFunctions Binary32 := inferInstance
  let : MathFunctions Binary32 := { ordinary with sin := fun _ => 0 }
  fail_if_success
    have invalid := roundoff_proof% (fun _ : Unit =>
      ExecFloat.Binary.toReal (MathFunctions.sin ((0 : Binary32) / 0)))
  trivial

/-- An actual forward output and one-hot upstream gradient exercise the generated backward
theorem with successful kernel-checked premises. The stable VJP retains the small gradient
that the conventional subtraction rounds to zero. -/
theorem log_softmax_fp32_gradient :
    let input : Tensor Binary32 [2] := Tensor.ofFn fun i => if i = 0 then 0 else -20
    let seed : Tensor Binary32 [2] := Tensor.ofFn fun i => if i = 0 then 1 else 0
    let y := Activation.logSoftmaxVecSpec input
    ExecFloat.Binary.toRat?
        ((Activation.Internal.logSoftmaxVecBackwardSpec y seed).getScalar 0) =
      some (9282611 / 2 ^ 52) ∧
    ExecFloat.Binary.toRat?
        (seed.getScalar 0 - MathFunctions.exp (y.getScalar 0) * Tensor.sumSpec seed) = some 0 := by
  intro input seed y
  have bound := log_softmax_pullback_error y seed 0 (by
    repeat' first | apply And.intro | intro
    all_goals decide +kernel)
  clear bound
  decide +kernel

/-- FloatLib's degree-ten enclosure is sufficient at -20. Only the two rational endpoint
comparisons need concrete evaluation; the exponential itself stays a mathematical real. -/
private theorem exp_neg_twenty_bounds :
    (206115361 / 10 ^ 17 : ℝ) ≤ Real.exp (-20) ∧
      Real.exp (-20) ≤ (206115366 / 10 ^ 17 : ℝ) := by
  have enclosed := Enclosure.contains_exp (-20 : Rat) 10
  have lower : (206115361 / 10 ^ 17 : Rat) ≤ (Enclosure.exp (-20) 10).lo := by
    decide +kernel
  have upper : (Enclosure.exp (-20) 10).hi ≤ (206115366 / 10 ^ 17 : Rat) := by
    decide +kernel
  have lowerReal : (206115361 / 10 ^ 17 : ℝ) ≤
      ((Enclosure.exp (-20) 10).lo : ℝ) := by
    simpa only [Rat.cast_div, Rat.cast_pow, Rat.cast_ofNat] using
      (Rat.cast_le (K := ℝ)).mpr lower
  have upperReal : ((Enclosure.exp (-20) 10).hi : ℝ) ≤
      (206115366 / 10 ^ 17 : ℝ) := by
    simpa only [Rat.cast_div, Rat.cast_pow, Rat.cast_ofNat] using
      (Rat.cast_le (K := ℝ)).mpr upper
  norm_num only [Rat.cast_neg, Rat.cast_ofNat] at enclosed
  exact ⟨lowerReal.trans enclosed.1, enclosed.2.trans upperReal⟩

/-- The reference is the existing real public operator's VJP, using the existing derivative
correctness connection. No second backward implementation is introduced. -/
theorem log_softmax_real_gradient :
    let input : Tensor ℝ [2] := Tensor.ofFn fun i => if i = 0 then 0 else -20
    let seed : Tensor ℝ [2] := Tensor.ofFn fun i => if i = 0 then 1 else 0
    ((Spec.logSoftmaxOp 0).backward input seed).getScalar 0 =
      Real.exp (-20) / (1 + Real.exp (-20)) := by
  intro input seed
  have connected := congrArg (fun value : Proofs.Autograd.Vec 2 => value 0)
    (Proofs.Autograd.getScalarE_logSoftmaxBackwardSpec input seed)
  have coordinate : ((Spec.logSoftmaxOp 0).backward input seed).getScalar 0 =
      1 - 1 / (1 + Real.exp (-20)) := by
    simpa [Spec.logSoftmaxOp, Proofs.Autograd.logSoftmaxVjp,
      Proofs.Autograd.softmaxVec, Proofs.Autograd.sumExp, Fin.sum_univ_two,
      input, seed] using connected
  rw [coordinate]
  have denominator : (1 + Real.exp (-20)) ≠ 0 := by positivity
  field_simp
  ring

/-- The public FP32 VJP preserves this small original-logit gradient: its absolute error is at
most 10^-16, its relative error is at most 10^-7, and it is strictly positive. A zero result
cannot satisfy these conclusions, even though it would pass a fixed 10^-7 absolute tolerance. -/
theorem log_softmax_fp32_accuracy :
    let input : Tensor Binary32 [2] := Tensor.ofFn fun i => if i = 0 then 0 else -20
    let seed : Tensor Binary32 [2] := Tensor.ofFn fun i => if i = 0 then 1 else 0
    let actual := ExecFloat.Binary.toReal
      (((Spec.logSoftmaxOp 0).backward input seed).getScalar 0)
    let reference := Real.exp (-20) / (1 + Real.exp (-20))
    |actual - reference| ≤ (1 / 10 ^ 16 : ℝ) ∧
      |actual - reference| ≤ (1 / 10 ^ 7 : ℝ) * |reference| ∧ 0 < actual := by
  intro input seed actual reference
  have encoded : ExecFloat.Binary.toRat?
      (((Spec.logSoftmaxOp 0).backward input seed).getScalar 0) =
      some (9282611 / 2 ^ 52) := by
    change ExecFloat.Binary.toRat?
      ((Activation.Internal.logSoftmaxVecBackwardSpec
        (Activation.logSoftmaxVecSpec input) seed).getScalar 0) = some _
    exact log_softmax_fp32_gradient.1
  have observed : actual = (9282611 / 2 ^ 52 : ℝ) := by
    have decoded := ExecFloat.Binary.toReal_of_toRat?_eq_some encoded
    norm_num only at decoded ⊢
    exact decoded
  have exponential := exp_neg_twenty_bounds
  have denominator : 0 < 1 + Real.exp (-20) := by positivity
  have referenceBounds :
      (206115361 / 10 ^ 17 : ℝ) / (1 + 206115361 / 10 ^ 17) ≤ reference ∧
        reference ≤ (206115366 / 10 ^ 17 : ℝ) / (1 + 206115366 / 10 ^ 17) := by
    constructor
    · apply (div_le_div_iff₀ (by norm_num) denominator).mpr
      nlinarith [exponential.1]
    · apply (div_le_div_iff₀ denominator (by norm_num)).mpr
      nlinarith [exponential.2]
  have lowerReference : (1 / 10 ^ 9 : ℝ) < reference := by
    exact lt_of_lt_of_le (by norm_num) referenceBounds.1
  have absoluteError : |actual - reference| ≤ (1 / 10 ^ 16 : ℝ) := by
    rw [observed]
    apply abs_le.mpr
    constructor
    · have upper := referenceBounds.2.trans
        (show (206115366 / 10 ^ 17 : ℝ) / (1 + 206115366 / 10 ^ 17) ≤
          (9282611 / 2 ^ 52 : ℝ) + 1 / 10 ^ 16 by norm_num)
      linarith
    · have lower :=
        (show (9282611 / 2 ^ 52 : ℝ) - 1 / 10 ^ 16 ≤
          (206115361 / 10 ^ 17 : ℝ) / (1 + 206115361 / 10 ^ 17) by norm_num).trans
          referenceBounds.1
      linarith
  have referencePositive : 0 < reference := lt_trans (by norm_num) lowerReference
  refine ⟨absoluteError, ?_, ?_⟩
  · rw [abs_of_pos referencePositive]
    exact absoluteError.trans (by linarith)
  · exact positive_of_error_lt absoluteError (by linarith)

open Proofs.Autograd.Algebra Runtime.Autograd.Torch

/-- One existing addition node with a shared parent, over an arbitrary scalar and tensor shape. -/
@[expose] def doublingGraph {α : Type} [Storage α] [Add α] [Zero α] {shape : Spec.Shape} :
    TypedGraph α [shape] shape :=
  let input : Proofs.Idx ([shape] ++ []) shape := ⟨⟨0, by simp⟩, rfl⟩
  { nodeShapes := [shape]
    data := .snoc .nil (NodeData.add input input)
    output := ⟨⟨1, by simp⟩, rfl⟩ }

/-- The public pullback has no arithmetic error over exact rationals. Its initial zero is
simplified only in the real reference; the executed gradient additions are retained. -/
theorem exact_graph_vjp_error {shape : Spec.Shape} (input seed : Tensor Rat shape)
    (i : shape.Coord) :
    |(((doublingGraph.vjp (TensorPack.singleton input) seed).head i : Rat) : ℝ) -
        ((seed i : ℝ) + (seed i : ℝ))| ≤ 0 := by
  roundoff

section Graph

variable {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "ConfiguredValue" => ExecFloat (Configured.Family format code plan)

/-- The range premise comes from indexed execution; the bound is on the public forward API. -/
theorem graph_forward_error [Storage ConfiguredValue] {shape : Spec.Shape}
    (x : Tensor ConfiguredValue shape) (i : shape.Coord) (hformat : format.isIEEE = true)
    (hx : ExecFloat.Binary.isFinite (x i) = true)
    (hresult : ExecFloat.Binary.isFinite
      ((doublingGraph.data.evalWithArray (TensorPack.singleton x) ()).tail.head i) = true) :
    |ExecFloat.Binary.toReal (doublingGraph.forward (TensorPack.singleton x) i) -
        (ExecFloat.Binary.toReal (x i) + ExecFloat.Binary.toReal (x i))| ≤
      Model.epsilonAt format (ExecFloat.Binary.toReal (x i) + ExecFloat.Binary.toReal (x i)) := by
  roundoff

/-- The saved reverse program counts both parent contributions and the later gradient addition.
No real-number associativity is used to reorder floating-point accumulation. -/
theorem saved_graph_pullback_error [Storage ConfiguredValue] {shape : Spec.Shape}
    (input previous seed : Tensor ConfiguredValue shape) (i : shape.Coord)
    (hformat : format.isIEEE = true)
    (hprevious : ExecFloat.Binary.isFinite (previous i) = true)
    (hseed : ExecFloat.Binary.isFinite (seed i) = true)
    (hdouble : ExecFloat.Binary.isFinite (seed i + seed i) = true)
    (hresult : ExecFloat.Binary.isFinite
      ((doublingGraph.data.backpropCtxSaved (TensorPack.singleton input) ()
        (TensorPack.pair previous seed)).head i) = true) :
    |ExecFloat.Binary.toReal
        ((doublingGraph.data.backpropCtxSaved (TensorPack.singleton input) ()
          (TensorPack.pair previous seed)).head i) -
        (ExecFloat.Binary.toReal (previous i) +
          (ExecFloat.Binary.toReal (seed i) + ExecFloat.Binary.toReal (seed i)))| ≤
      Model.epsilonAt format (ExecFloat.Binary.toReal (seed i) + ExecFloat.Binary.toReal (seed i)) +
        Model.epsilonAt format
          (ExecFloat.Binary.toReal (previous i) + ExecFloat.Binary.toReal (seed i + seed i)) := by
  roundoff

/-- The configured storage selects dependency-aware execution through the public VJP API.
The bound includes fan-out and the addition to the input's zero-initialized gradient. -/
theorem graph_vjp_error {shape : Spec.Shape}
    (input seed : Tensor ConfiguredValue shape) (i : shape.Coord)
    (hformat : format.isIEEE = true)
    (hseed : ExecFloat.Binary.isFinite (seed i) = true)
    (hdouble : ExecFloat.Binary.isFinite (seed i + seed i) = true)
    (hresult : ExecFloat.Binary.isFinite
      ((doublingGraph.vjp (TensorPack.singleton input) seed).head i) = true) :
    |ExecFloat.Binary.toReal ((doublingGraph.vjp (TensorPack.singleton input) seed).head i) -
        (ExecFloat.Binary.toReal (0 : ConfiguredValue) +
          (ExecFloat.Binary.toReal (seed i) + ExecFloat.Binary.toReal (seed i)))| ≤
      Model.epsilonAt format (ExecFloat.Binary.toReal (seed i) + ExecFloat.Binary.toReal (seed i)) +
        Model.epsilonAt format
          (ExecFloat.Binary.toReal (0 : ConfiguredValue) +
            ExecFloat.Binary.toReal (seed i + seed i)) := by
  roundoff

/-- Graph normalization must retain the selected scalar dictionary. Calling an operation addition
does not give a subtraction implementation FloatLib's addition theorem. -/
example [Storage ConfiguredValue] {shape : Spec.Shape} (x : Tensor ConfiguredValue shape)
    (i : shape.Coord) (_hformat : format.isIEEE = true)
    (_hx : ExecFloat.Binary.isFinite (x i) = true)
    (_hresult : ExecFloat.Binary.isFinite (x i - x i) = true) : True := by
  let : Add ConfiguredValue := ⟨fun a b => a - b⟩
  fail_if_success
    have : |ExecFloat.Binary.toReal (doublingGraph.forward (TensorPack.singleton x) i) -
        (ExecFloat.Binary.toReal (x i) + ExecFloat.Binary.toReal (x i))| ≤
        Model.epsilonAt format (ExecFloat.Binary.toReal (x i) + ExecFloat.Binary.toReal (x i)) := by
      roundoff
  trivial

/-- Exact constant rules also retain the actual literal instance. -/
example : True := by
  let : OfNat ConfiguredValue 0 := ⟨1⟩
  fail_if_success
    have : |ExecFloat.Binary.toReal (0 : ConfiguredValue) - 0| ≤ 0 := by
      roundoff
  trivial

end Graph

/-- Instantiate the generated arbitrary-format theorem with an actual successful FP32 check.
The executable premises are discharged by kernel reduction. -/
theorem expm1_fp32_rounding_error :
    let x : Binary32 := Rat.cast (-1 / 2 ^ 25 : Rat)
    |ExecFloat.Binary.toReal (Context.expm1 x) -
        (Real.exp (ExecFloat.Binary.toReal x) - 1)| ≤
      Model.epsilonAt FloatFormat.binary32 (Real.exp (ExecFloat.Binary.toReal x) - 1) := by
  intro x
  have hx : ExecFloat.Binary.isFinite x = true := by decide +kernel
  have hcertificate : Certified.expMinus1 (ExecFloat.Binary.toModel x) {} =
      .ok (ExecFloat.Binary.toModel (Context.expm1 x)) := by
    have observed : (Certified.expMinus1 (ExecFloat.Binary.toModel x) {}).toOption =
        some (ExecFloat.Binary.toModel (Context.expm1 x)) := by decide +kernel
    cases h : Certified.expMinus1 (ExecFloat.Binary.toModel x) {} with
    | error _ =>
        rw [h] at observed
        change none = some _ at observed
        cases observed
    | ok result =>
        rw [h] at observed
        change some result = some _ at observed
        exact congrArg Except.ok (Option.some.inj observed)
  exact checked_expm1_error x hx hcertificate

end NN.Tests.Verification.Roundoff
