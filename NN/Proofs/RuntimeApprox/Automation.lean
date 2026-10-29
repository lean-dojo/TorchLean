/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Real
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Operations.Proof
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Transcendentals
public import FloatLib.Numerics.Enclosure.Interval.HyperbolicProof
public import NN.Tensor.Reductions
public import NN.Proofs.RuntimeApprox.Rounding
public import NN.Proofs.RuntimeApprox.Recovery
public import Mathlib.Data.Rat.Cast.Lemmas
public import NN.Proofs.RuntimeApprox.Automation.Fold
public meta import Mathlib.Util.Qq
public meta import Lean.Elab.Tactic.Basic
public meta import Lean.Elab.Tactic.Simp
public meta import Lean.Meta.DiscrTree
public meta import Lean.Meta.RecursorInfo
public meta import Lean.Meta.Match.MatcherInfo
public meta import Lean.Compiler.CSimpAttr
public meta import NN.Proofs.RuntimeApprox.Automation.Rules

/-!
# Numerical proofs from the implementation

`roundoff` constructs an absolute-error proof directly from configured floating-point arithmetic.
It preserves each operation and its selected capability, reuses FloatLib's primitive theorems,
and shares the analysis of repeated subexpressions. No model-specific expression language or
separately supplied implementation is involved.

To inspect a generated reference and budget without writing them first, use the term form in
a local proof: `have bound := by roundoff (ExecFloat.Binary.toReal (program input))`.
Its range and domain premises must already be provable. The ordinary target form leaves any
missing premises as proof goals. To generate a reusable conditional theorem, use
`roundoff theorem name := fun input => ExecFloat.Binary.toReal (program input)`.
The generated statement records the required conditions; applying it requires proving them.

Tensor folds and finite index loops reuse one scalar invariant theorem, without unrolling their
elements. Generated entries and operations on several tensors reduce over their shared coordinates.
Conditions concern the states actually visited. Exact decoding equalities into
FloatLib's descriptor model are also supported; decoding an approximation alone does not prove
its accuracy against the corresponding real function.

`[roundoff_rule]` registers an existing numerical bound or observation equality. It also accepts
imported, polymorphic theorems and local registrations. Equalities introduce no local rounding
error. Quantified input-error assumptions can be reused at each tensor coordinate.

Tensor observations expose operation records and typed graph lookups before scalar analysis.
Registered `csimp` theorems connect optimized programs to their proved source semantics. External
or `implemented_by` code needs its own correspondence evidence. A numerical VJP bound concerns the
computed reverse map; the existing derivative theorems establish its mathematical meaning.

For exponentials, logarithms, `expm1` and `log1p`, the bound includes the observed discrepancy
from a FloatLib-certified result. The ordinary implementation is not assumed correctly rounded.
The certificate must succeed; its failure leaves an acceptance condition unproved.
Sine, cosine and the hyperbolic functions reuse FloatLib's rational enclosures instead. The
distance from the actual candidate to the enclosure endpoints bounds its local error; input
perturbations are propagated by the same registered real-error theorems.

Successful coefficient recovery reuses its endpoint-enclosure proof. The original input and
seed arithmetic remain in the composed budget, and optional results retain their source program.

The descriptor, storage plan and scalar representation remain parameters. IEEE-format and
finiteness conditions are proved from the context when possible and otherwise remain goals.
Inputs are decoded as stored. Floating-point literals and casts inside the computation carry
conversion error relative to their exact real values.
The primitive float rules require finite operands and results. A supplied error tolerance
still has to bound the generated budget; a weak tolerance need not preserve a small gradient.
-/

public section

namespace Proofs.RuntimeApprox.Roundoff

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange TorchLean

/-- Compare an existing implementation with a certified reference, retaining their observed
discrepancy. The candidate need not itself be correctly rounded. Certificate failure supplies
no bound, and a nonfinite candidate cannot represent a real computation. -/
theorem certified_error {format : FloatFormat} {reference : ℝ}
    (candidate witness : Model format) (_houtput : Model.isFinite candidate = true)
    (hwitness : Model.toReal witness = Model.roundAt format reference) :
    |Model.toReal candidate - reference| ≤
      |Model.toReal candidate - Model.toReal witness| + Model.epsilonAt format reference := by
  have localError : |Model.toReal witness - reference| ≤ Model.epsilonAt format reference := by
    rw [hwitness]
    exact Model.abs_roundAt_sub_le _ _
  exact (abs_sub_le (Model.toReal candidate) (Model.toReal witness) reference).trans
    (add_le_add le_rfl localError)

section Enclosures

open FloatLib.Numerics

/-- The stored input belongs to its exact rational point interval. -/
theorem contains_decoded_point {format : FloatFormat} (input : Model format) :
    (Interval.point ((Model.toRat? input).getD 0)).ContainsReal some (Model.toReal input) := by
  apply (Interval.containsReal_some_iff _ _).mpr
  simp only [Interval.point, cast_toRat_eq_toReal, le_refl, and_self]

/-- Distance to the farther endpoint bounds the error of any candidate against a real value
in the interval. The candidate itself need not belong to the interval. -/
theorem enclosure_error {reference : ℝ} {bounds : Interval Rat} (candidate : ℝ)
    (contained : bounds.ContainsReal some reference) :
    |candidate - reference| ≤
      max |candidate - (bounds.lo : ℝ)| |candidate - (bounds.hi : ℝ)| := by
  obtain ⟨lower, upper⟩ := (Interval.containsReal_some_iff bounds reference).mp contained
  apply abs_le.mpr
  constructor
  · have lowerDistance := neg_abs_le (candidate - (bounds.hi : ℝ))
    have largest := le_max_right |candidate - (bounds.lo : ℝ)| |candidate - (bounds.hi : ℝ)|
    linarith
  · have upperDistance := le_abs_self (candidate - (bounds.lo : ℝ))
    have largest := le_max_left |candidate - (bounds.lo : ℝ)| |candidate - (bounds.hi : ℝ)|
    linarith

/-- Compare a finite implementation result with a certified rational enclosure. Both finite
premises remain part of the numerical interface. The rational budget can be evaluated separately
from the proof, without assuming that the implementation is correctly rounded. -/
theorem model_enclosure_error {inputFormat outputFormat : FloatFormat}
    (input : Model inputFormat) (candidate : Model outputFormat)
    (_hinput : Model.isFinite input = true) (_hcandidate : Model.isFinite candidate = true)
    {reference : ℝ} {bounds : Interval Rat}
    (contained : bounds.ContainsReal some reference) :
    |Model.toReal candidate - reference| ≤
      ((max |(Model.toRat? candidate).getD 0 - bounds.lo|
        |(Model.toRat? candidate).getD 0 - bounds.hi| : Rat) : ℝ) := by
  simpa only [Rat.cast_max, Rat.cast_abs, Rat.cast_sub, cast_toRat_eq_toReal] using
    enclosure_error (Model.toReal candidate) contained

end Enclosures

/-- Clearing a finite value's sign is exact in every format with signed zero. -/
theorem model_toReal_abs {format : FloatFormat} (value : Model format)
    (hformat : format.supportsSignedZero = true) (hvalue : Model.isFinite value = true) :
    Model.toReal (Model.abs value) = |Model.toReal value| := by
  obtain ⟨dyadic, decoded⟩ := Model.exists_toDyadic?_of_isFinite hvalue
  rw [Model.toReal_eq, Model.toDyadic?_abs_of_supportsSignedZero hformat decoded,
    Model.toReal_eq, decoded]
  rw [Model.Dyadic.abs_toReal]
  rfl

/-- Casts used inside an algorithm contribute rounding error relative to the exact rational. -/
theorem model_ofRat_error {format : FloatFormat} (value : Rat)
    (hformat : format.isIEEE = true) (hfinite : Model.isFinite (Model.ofRat format value) = true) :
    |Model.toReal (Model.ofRat format value) - (value : ℝ)| ≤
      Model.epsilonAt format (value : ℝ) := by
  rw [model_toReal_ofRat value hformat hfinite]
  exact Model.abs_roundAt_sub_le _ _

private theorem model_maximum_finite {format : FloatFormat} (left right : Model format)
    (signedZero : format.supportsSignedZero = true)
    (leftFinite : Model.isFinite left = true) (rightFinite : Model.isFinite right = true) :
    Model.isFinite (Model.maximum left right) = true := by
  have leftNotNaN := Model.isNaN_eq_false_of_isFinite_eq_true left leftFinite
  have rightNotNaN := Model.isNaN_eq_false_of_isFinite_eq_true right rightFinite
  have selected := Model.chooseNaN2_none_of_not_isNaN left right leftNotNaN rightNotNaN
  have zeroFinite (sign : Bool) : Model.isFinite (Model.zero format sign) = true := by
    cases sign <;> simp [Model.zero, signedZero]
  have ordinary : Model.maximum left right =
      match Model.compareNonNaN left right leftNotNaN rightNotNaN with
      | .lt => right
      | .gt => left
      | .eq =>
          if Model.isZero left && Model.isZero right then
            Model.zero format (Model.signBit left && Model.signBit right)
          else left := by
    simp only [Model.maximum, selected, Model.withNaNSelection, dite_eq_left True.intro]
    rfl
  rw [ordinary]
  cases comparison : Model.compareNonNaN left right leftNotNaN rightNotNaN with
  | lt => simpa only [comparison] using rightFinite
  | gt => simpa only [comparison] using leftFinite
  | eq =>
      by_cases zeros : (Model.isZero left && Model.isZero right) = true
      · simpa only [comparison, ite_eq_left zeros] using
          zeroFinite (Model.signBit left && Model.signBit right)
      · simpa only [comparison, ite_eq_right zeros] using leftFinite

section Configured

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- The binary decoder preserves values packed by the same configured family. -/
theorem decode_family (value : Model format) :
    ExecFloat.Binary.toModel
        (Configured.Family.ofModel (plan := plan) (code := code) value) = value := by
  change Configured.Family.toModel (Configured.Family.ofModel value) = value
  exact Configured.Family.toModel_ofModel value

/-- The configured zero literal denotes real zero for every codec and storage plan. -/
theorem toReal_zero :
    ExecFloat.Binary.toReal (0 : ExecFloat (Configured.Family format code plan)) = 0 := by
  change Model.toReal (Configured.Family.toModel
    (Configured.Family.ofModel (Model.ofRat format (0 : Rat)))) = 0
  rw [Configured.Family.toModel_ofModel]
  simp [Model.ofRat, Model.Internal.Directed.roundRat, Model.Internal.Directed.roundRatScaled]

/-- Zero-initialized accumulators are finite without a caller-supplied range premise. -/
theorem isFinite_zero :
    ExecFloat.Binary.isFinite (0 : ExecFloat (Configured.Family format code plan)) = true := by
  change Model.isFinite (Configured.Family.toModel
    (Configured.Family.ofModel (Model.ofRat format (0 : Rat)))) = true
  rw [Configured.Family.toModel_ofModel]
  simpa [Model.ofRat, Model.Internal.Directed.roundRat, Model.Internal.Directed.roundRatScaled]
    using Model.isFinite_eq_true_of_isZero_eq_true _ (Model.isZero_zero format false)

/-- The codec preserves any unary model operation, including the scalar context's functions. -/
theorem decode_unary (operation : Model format → Model format)
    (value : ExecFloat (Configured.Family format code plan)) :
    ExecFloat.Binary.toModel
        (ExecFloat.ModelCodec.liftUnary (Model := Model format) (plan := plan) operation value) =
      operation (ExecFloat.Binary.toModel value) :=
  ExecFloat.ModelCodec.decode_liftUnary operation value

/-- Binary scalar lifts use the same codec law regardless of their numerical operation. -/
theorem decode_binary (operation : Model format → Model format → Model format)
    (left right : ExecFloat (Configured.Family format code plan)) :
    ExecFloat.Binary.toModel
        (ExecFloat.ModelCodec.liftBinary (Model := Model format) (plan := plan)
          operation left right) =
      operation (ExecFloat.Binary.toModel left) (ExecFloat.Binary.toModel right) :=
  ExecFloat.ModelCodec.decode_liftBinary operation left right

/-- The codec preserves an outcome's value independently of its exception indicators. -/
theorem decode_outcome (outcome : Model.IEEEOutcome format) :
    ExecFloat.Binary.toModel
        ((ExecFloat.Binary.IEEEOutcome.ofModel (plan := plan) (code := code) outcome).value) =
      outcome.value := by
  simpa only [ExecFloat.Binary.IEEEOutcome.toModel] using
    congrArg Model.IEEEOutcome.value
      (ExecFloat.Binary.IEEEOutcome.toModel_ofModel (plan := plan) (code := code) outcome)

/-- A real observation can pass through a scalar lift before its numerical law is applied. -/
theorem toReal_liftUnary (operation : Model format → Model format)
    (value : ExecFloat (Configured.Family format code plan)) :
    ExecFloat.Binary.toReal
        (ExecFloat.ModelCodec.liftUnary (Model := Model format) (plan := plan) operation value) =
      Model.toReal (operation (ExecFloat.Binary.toModel value)) :=
  congrArg Model.toReal (decode_unary operation value)

/-- Binary lifts retain the chosen model operation and both operands. -/
theorem toReal_liftBinary (operation : Model format → Model format → Model format)
    (left right : ExecFloat (Configured.Family format code plan)) :
    ExecFloat.Binary.toReal
        (ExecFloat.ModelCodec.liftBinary (Model := Model format) (plan := plan)
          operation left right) =
      Model.toReal (operation (ExecFloat.Binary.toModel left) (ExecFloat.Binary.toModel right)) :=
  congrArg Model.toReal (decode_binary operation left right)

private theorem finite_beq_self (value : Value)
    (finite : ExecFloat.Binary.isFinite value = true) : (value == value) = true := by
  have compared := Model.compare_self_of_isNaN_eq_false (ExecFloat.Binary.toModel value)
    (Model.isNaN_eq_false_of_isFinite_eq_true _ finite)
  change (Model.compare (ExecFloat.Binary.toModel value) (ExecFloat.Binary.toModel value) ==
    some .eq) = true
  simp only [compared, beq_self_eq_true]

/-- Finite max-absolute reduction is an exact observation of the existing traversal.
The original tensor remains in the real fold, including any computation that produced it. -/
theorem toReal_maxAbs [Storage Value] {shape : Spec.Shape} (input : Tensor Value shape)
    (signedZero : format.supportsSignedZero = true)
    (finite : ∀ coordinate, ExecFloat.Binary.isFinite (input coordinate) = true) :
    ExecFloat.Binary.toReal (Tensor.maxAbs input) =
      input.foldl (fun (largest : ℝ) value => max largest |ExecFloat.Binary.toReal value|) 0 := by
  have related := Tensor.Internal.Rep.foldl_rel input input
    (fun largest value : Value =>
      if !(largest == largest) then largest
      else if !(value == value) then value
      else max largest (MathFunctions.abs value))
    (fun (largest : ℝ) value => max largest |ExecFloat.Binary.toReal value|)
    (fun (largest : Value) (reference : ℝ) =>
      ExecFloat.Binary.isFinite largest = true ∧ ExecFloat.Binary.toReal largest = reference)
    0 0 ⟨isFinite_zero, toReal_zero⟩ (by
      intro coordinate largest reference state
      obtain ⟨largestFinite, observedLargest⟩ := state
      have valueFinite := finite coordinate
      have largestSelf := finite_beq_self largest largestFinite
      have valueSelf := finite_beq_self (input coordinate) valueFinite
      simp only [largestSelf, valueSelf, Bool.not_true, Bool.false_eq_true, reduceIte]
      have absoluteFinite :
          ExecFloat.Binary.isFinite (MathFunctions.abs (input coordinate)) = true := by
        change Model.isFinite
          (ExecFloat.Binary.toModel (ExecFloat.Binary.abs (input coordinate))) = true
        rw [ExecFloat.Binary.toModel_abs, Model.isFinite_abs]
        exact valueFinite
      have observedAbsolute : ExecFloat.Binary.toReal (MathFunctions.abs (input coordinate)) =
          |ExecFloat.Binary.toReal (input coordinate)| := by
        change Model.toReal
            (ExecFloat.Binary.toModel (ExecFloat.Binary.abs (input coordinate))) =
          |Model.toReal (ExecFloat.Binary.toModel (input coordinate))|
        rw [ExecFloat.Binary.toModel_abs]
        exact model_toReal_abs _ signedZero valueFinite
      constructor
      · change Model.isFinite (ExecFloat.Binary.toModel
          (max largest (ExecFloat.Binary.abs (input coordinate)))) = true
        rw [ExecFloat.Binary.toModel_max]
        exact model_maximum_finite _ _ signedZero largestFinite absoluteFinite
      · calc
          ExecFloat.Binary.toReal (max largest (MathFunctions.abs (input coordinate))) =
              max (ExecFloat.Binary.toReal largest)
                (ExecFloat.Binary.toReal (MathFunctions.abs (input coordinate))) := by
            change Model.toReal (ExecFloat.Binary.toModel
                (max largest (ExecFloat.Binary.abs (input coordinate)))) =
              max (Model.toReal (ExecFloat.Binary.toModel largest))
                (Model.toReal
                  (ExecFloat.Binary.toModel (ExecFloat.Binary.abs (input coordinate))))
            rw [ExecFloat.Binary.toModel_max]
            exact Model.toReal_maximum_eq_max_of_isFinite _ _ largestFinite absoluteFinite
          _ = max reference |ExecFloat.Binary.toReal (input coordinate)| := by
            rw [observedLargest, observedAbsolute])
  simpa only [Tensor.maxAbs, Tensor.foldlSpec, Tensor.foldl] using related.2

end Configured

/-- Decode a tensor reduction using the codecs around its existing scalar step.
Input and accumulator formats may differ. Subsequent primitive rewrites remove the codecs from
the step body, retaining its arithmetic and precision choices without a second implementation. -/
theorem decode_foldl {inputFormat stateFormat : FloatFormat}
    {inputPlan : Configured.StoragePlan inputFormat}
    {statePlan : Configured.StoragePlan stateFormat} {inputCode stateCode : Type}
    [ExecFloat.ModelCodec inputPlan (Model inputFormat) inputCode]
    [ExecFloat.ModelCodec statePlan (Model stateFormat) stateCode]
    [Storage (ExecFloat (Configured.Family inputFormat inputCode inputPlan))]
    [Storage (Model inputFormat)]
    {shape : Spec.Shape}
    (input : Tensor (ExecFloat (Configured.Family inputFormat inputCode inputPlan)) shape)
    (step : ExecFloat (Configured.Family stateFormat stateCode statePlan) →
      ExecFloat (Configured.Family inputFormat inputCode inputPlan) →
      ExecFloat (Configured.Family stateFormat stateCode statePlan))
    (initial : ExecFloat (Configured.Family stateFormat stateCode statePlan)) :
    ExecFloat.Binary.toModel (Tensor.Internal.Rep.foldl step initial input) =
      Tensor.Internal.Rep.foldl
        (fun state value => ExecFloat.Binary.toModel
          (step (ExecFloat.Binary.ofModel state) (ExecFloat.Binary.ofModel value)))
        (ExecFloat.Binary.toModel initial)
        (Tensor.Internal.Rep.map ExecFloat.Binary.toModel input) := by
  apply Tensor.Internal.Rep.map_foldl
  intro accumulator value
  simp only [ExecFloat.Binary.ofModel_toModel]

universe u v

/-- Split an actual optional computation without replacing its payload by a fresh input. -/
theorem option_apply_eq_dite {α : Type u} {β : Sort v}
    (program : Option α → β) (source : Option α) :
    program source = if found : source.isSome then
      program (some (source.get found)) else program none := by
  cases source <;> rfl

end Proofs.RuntimeApprox.Roundoff

public meta section

namespace Proofs.RuntimeApprox.Roundoff

open Lean Meta Elab Tactic Qq
open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange
open FloatLib.Floats.Formats.BinaryInterchange.Model.Transcendentals

-- Built-in and user-supplied scalar theorems use the same rule mechanism.
attribute [roundoff_rule]
  configured_decode
  TorchLean.Numeric.QuotientCoefficients.primal_add
  TorchLean.Numeric.QuotientCoefficients.primal_sub
  TorchLean.Numeric.QuotientCoefficients.primal_mul
  TorchLean.Numeric.QuotientCoefficients.primal_div
  TorchLean.Numeric.QuotientCoefficients.primal_map
  TorchLean.Tensor.eq_mp_apply
  TorchLean.Tensor.LinearAlgebra.Internal.matmulCommonBatchSpec_apply
  TorchLean.Tensor.Reduction.Internal.reduceOuterAxis_apply
  TorchLean.Tensor.reduceFromBroadcastTo_self
  TorchLean.Tensor.reduceDim_apply
  TorchLean.Tensor.reduceSum_apply
  TorchLean.Tensor.broadcastAfterSum_apply
  Spec.Shape.swapAdjacentAtDepth_concat_rank
  ExecFloat.Binary.abs_toReal_add_sub_le
  ExecFloat.Binary.abs_toReal_sub_sub_le
  ExecFloat.Binary.abs_toReal_mul_sub_le
  ExecFloat.Binary.abs_toReal_div_sub_le
  Model.abs_toReal_add_sub_le
  Model.abs_toReal_sub_sub_le
  Model.abs_toReal_mul_sub_le
  Model.abs_toReal_div_sub_le
  Model.abs_toReal_sqrt_sub_le
  Model.abs_toReal_fma_sub_le
  Model.abs_toReal_cast_sub_le_of_isIEEE_destination
  Model.abs_roundAt_sub_le
  Model.toReal_pi
  Model.toReal_neg
  model_toReal_abs
  toReal_maxAbs
  Model.toReal_minimum_eq_min_of_isFinite
  Model.toReal_maximum_eq_max_of_isFinite
  model_ofRat_error
  model_toReal_ofRat
  decode_outcome
  Model.value_ofRat Model.value_add Model.value_sub Model.value_mul Model.value_div
  Model.value_fma Model.value_sqrt
  enclosedSlope_get_error
  enclosedLogarithm_get_error
  cast_toRat_eq_toReal
  Rat.cast_add Rat.cast_sub Rat.cast_mul Rat.cast_div Rat.cast_neg
  Rat.cast_zero Rat.cast_one Rat.cast_natCast Rat.cast_intCast Rat.cast_ofNat
  Rat.cast_ofScientific Nat.cast_zero Nat.cast_one
  toReal_liftUnary
  toReal_liftBinary
  toReal_zero
  add_error sub_error mul_error div_error
  neg_error abs_error min_error max_error
  sqrt_error exp_error log_error
  sin_error cos_error tanh_error sinh_error cosh_error

/-- Proof construction data only; the executed tensor program is never replaced by this record. -/
private structure Estimate where
  reference : Q(ℝ)
  error : Q(ℝ)
  proof : Lean.Expr

private structure Analysis where
  cache : Lean.ExprMap Estimate := {}
  pending : Array MVarId := #[]
  rules : DiscrTree (Name × Bool) := {}
  observations : NameSet := {}
  boundaries : NameSet := {}

private abbrev AnalyzeM := StateRefT Analysis TacticM

/-- A failed speculative derivation leaves no assignments or errors behind. Recursive source
analysis keeps the declaration's budget rather than sharing a local matching budget. -/
private def transaction? {α : Type} (action : TacticM (Option α)) (catchLimits : Bool := false) :
    TacticM (Option α) := do
  let saved ← saveState
  let (result?, _) ← tryFinally'
    (tryCatchRuntimeEx (do
      let result? ← action
      if (← Core.getMessageLog).hasErrors then pure none else pure result?) (fun error => do
        if error.isInterrupt || (!catchLimits && error.isRuntime) then throw error
        trace[TorchLean.roundoff] "proof search: {error.toMessageData}"
        pure none))
    (fun result => unless result.any (·.isSome) do saved.restore)
  return result?

/-- Bound local matching and proof search independently of recursive source analysis. -/
private def attempt? {α : Type} (heartbeats : Nat) (action : TacticM (Option α)) :
    TacticM (Option α) :=
  transaction? (catchLimits := true) <| withTheReader Core.Context
    (fun c => { c with maxHeartbeats := heartbeats * 1000 }) (withCurrHeartbeats action)

private def literalHeads : Array Name :=
  #[``OfNat.ofNat, ``OfScientific.ofScientific, ``Real.pi]

/-- Existing FloatLib certificates identify real functions, independently of the fast candidate
being checked. The numerical descriptor is inferred from that candidate's representation. -/
private def certificates : Array (Name × Array Name) := #[
  (``Certified.exp_correctlyRoundedCertificateOn,
    #[``MathFunctions.exp, ``FloatLib.Floats.Formats.BinaryInterchange.Model.exp]),
  (``Certified.log_correctlyRoundedCertificateOn,
    #[``MathFunctions.log, ``FloatLib.Floats.Formats.BinaryInterchange.Model.log]),
  (``Certified.expMinus1_correctlyRoundedCertificateOn,
    #[``Context.expm1, ``TorchLean.Numeric.expm1]),
  (``Certified.logPlus1_correctlyRoundedCertificateOn,
    #[``Context.log1p, ``TorchLean.Numeric.log1p])]

/-- Local proof construction for a loop body, before its accumulator variables are closed. -/
private structure StepEstimate where
  reference : Lean.Expr
  error : Lean.Expr
  condition : Lean.Expr
  proof : Lean.Expr
  actualCondition? : Option Lean.Expr

/-- Recognize tensor-containing program values from their type definitions. A shared visited set
handles recursive datatypes without enumerating graph, operation-record, or user wrapper types. -/
private partial def containsTensor (environment : Environment) (type : Lean.Expr) :
    StateM NameSet Bool := do
  if type.isAppOf ``TorchLean.Tensor || type.isAppOf ``TorchLean.Tensor.Internal.Rep then
    return true
  match type with
  | .app function argument =>
      if ← containsTensor environment argument then return true
      containsTensor environment function
  | .forallE _ domain body _ | .lam _ domain body _ =>
      if ← containsTensor environment body then return true
      containsTensor environment domain
  | .letE _ type value body _ =>
      if ← containsTensor environment body then return true
      if ← containsTensor environment type then return true
      containsTensor environment value
  | .mdata _ value | .proj _ _ value => containsTensor environment value
  | .const name _ =>
      if (← get).contains name || (`FloatLib).isPrefixOf name then return false
      modify (·.insert name)
      match environment.find? name with
      | some (.inductInfo info) =>
          for constructor in info.ctors do
            if let some info := environment.find? constructor then
              if ← containsTensor environment info.type then return true
          return false
      | some (.defnInfo info) =>
          if info.type.getForallBody.isSort then containsTensor environment info.value
          else return false
      | _ => return false
  | _ => return false

/-- Protect the tensor producers in the applied observation pattern. Generic functions can
return a tensor here even when their own declaration returns an arbitrary type parameter. -/
private partial def tensorProducerHeads (expression : Lean.Expr) (names : NameSet) :
    MetaM NameSet := do
  let expression ← instantiateMVars expression
  let mut names := names
  if let some name := expression.getAppFn.constName? then
    let type ← inferType expression
    if ((containsTensor (← getEnv) type.getForallBody).run {}).1 then
      names := names.insert name
  if let .proj _ _ value := expression then return ← tensorProducerHeads value names
  if let .mdata _ value := expression then return ← tensorProducerHeads value names
  let mut type ← inferType expression.getAppFn
  for argument in expression.getAppArgs do
    let .forallE _ _ body info := type | break
    if info == .default && !(← isProof argument) then
      names ← tensorProducerHeads argument names
    type := body.instantiate1 argument
  return names

/-- Keep registered producers available for observation equalities and numerical bounds.
Numerical rules protect their source functions without rewriting them to real arithmetic. -/
private def programRules : MetaM (Array (Name × Bool) × NameSet) := do
  let environment ← getEnv
  let mut rules := #[]
  let mut boundaries : NameSet := {}
  for name in roundoffRules.getState environment do
    unless environment.contains name do continue
    let inverse := Lean.Compiler.hasCSimpAttribute environment name
    let heads? ← withNewMCtxDepth do
      let rule ← mkConstWithFreshMVarLevels name
      let (_, _, conclusion) ← forallMetaTelescopeReducing (← inferType rule)
      do
        if let some (type, left, right) := conclusion.eq? then
          -- A polymorphic coordinate equality rewrites every scalar type. Classifying it
          -- by unifying its type with ℝ would accidentally discard its floating-point uses.
          let numerical := !inverse && (← withReducible (whnf type)).isConstOf ``Real
          let source := if inverse then right else left
          let mut heads ← tensorProducerHeads source {}
          unless numerical do
            if let some head := source.getAppFn.constName? then heads := heads.insert head
          return some (heads, !numerical)
        let conclusion : Q(Prop) ← pure conclusion
        match conclusion with
        | ~q(|($actual : ℝ) - $_reference| ≤ $_error) =>
            return some (← tensorProducerHeads actual {}, false)
        | _ => return none
    if let some (heads, rewrite) := heads? then
      if rewrite then
        rules := rules.push (name, inverse)
      -- The applied pattern fixes the result type, including generic transports such as Eq.mp.
      for head in heads do
        boundaries := boundaries.insert head
  return (rules, boundaries)

/-- Preserve FloatLib arithmetic, tensor storage, and registered tensor producers. -/
private def programCanUnfold (boundaries : NameSet) (conditionals : Bool) (config : Meta.Config)
    (info : ConstantInfo) : CoreM Bool := do
  if boundaries.contains info.name then
    -- A failed control-flow reduction retains the original expression. Generic wrappers may
    -- therefore reduce while inspecting a discriminant; actual tensor producers stay intact.
    unless conditionals &&
        !((containsTensor (← getEnv) info.type.getForallBody).run {}).1 do return false
  if (!conditionals && (info.name == ``ite || info.name == ``dite)) ||
      (`FloatLib).isPrefixOf info.name ||
      (`TorchLean.Tensor.Internal.Rep).isPrefixOf info.name then return false
  canUnfoldDefault config info

/-- Reduce control flow only when its actual discriminant computes. Selected branches retain
their numerical operations; unknown options and flags never become fresh exact inputs. -/
private def reduceControl? (boundaries : NameSet) (value : Lean.Expr) :
    MetaM (Option Lean.Expr) := do
  let some name := value.getAppFn.constName? | return none
  let reduce {α : Type} (action : MetaM α) : MetaM α :=
    withCanUnfoldPred (programCanUnfold boundaries true) (withDefault action)
  if (name == ``ite || name == ``dite) && value.getAppNumArgs ≥ 5 then
    let arguments := value.getAppArgs
    let decision ← reduce (whnf (mkApp2 (mkConst ``decide) arguments[1]! arguments[2]!))
    unless decision.isConstOf ``Bool.true || decision.isConstOf ``Bool.false do return none
    let selected := arguments[if decision.isConstOf ``Bool.true then 3 else 4]!
    let selected := if name == ``dite then
        mkApp selected (mkApp2 (mkConst ``Decidable.reflects_decide) arguments[1]! arguments[2]!)
      else selected
    return some (mkAppN selected (arguments.extract 5 arguments.size)).headBeta
  if (← getMatcherInfo? name).isSome ||
      ((← getEnv).find? name).any (fun info => match info with
        | .recInfo _ => true | _ => false) then
    return ← reduce (reduceRecMatcher? value)
  return none

/-- Recover an unfolded dependent conditional only when its motive and complete value agree
with `dite`. The predicate and decision instance are taken from the actual discriminant. -/
private def restoreBranch? (source : Expr) : MetaM (Option Expr) := do
  let source ← instantiateMVars source
  unless source.isAppOfArity ``Bool.rec 5 && !source.hasExprMVar do return none
  let arguments := source.getAppArgs
  let decision := arguments[3]!
  unless decision.isAppOfArity ``decide 2 do return none
  let predicate := decision.getAppArgs[0]!
  let selected := decision.getAppArgs[1]!
  withoutModifyingState <| withNewMCtxDepth do
    try
      let resultType ← inferType source
      let expectedMotive ← withLocalDeclD `bit q(Bool) fun bit => do
        let reflected := mkApp2 (mkConst ``Bool.Reflects) bit predicate
        mkLambdaFVars #[bit] (← mkArrow reflected resultType)
      unless ← withDefault (isDefEq arguments[0]! expectedMotive) do return none
      let restored ← withDefault <| mkAppOptM ``dite
        #[some resultType, some predicate, some selected,
          some arguments[2]!, some arguments[1]!]
      let restored ← instantiateMVars restored
      if restored.hasExprMVar then return none
      unless ← withDefault (isDefEq source restored) do return none
      return some restored
    catch error =>
      if error.isInterrupt || error.isRuntime then throw error
      return none

/-- Move a function application through the actual branch, preserving its decision instance. -/
private def applyBranch? (observe control : Expr) : MetaM (Option (Expr × Expr)) := do
  let dependent := control.isAppOfArity ``dite 5
  unless dependent || control.isAppOfArity ``ite 5 do return none
  let .forallE _ _ codomain _ ← withReducible (whnf (← inferType observe)) | return none
  -- Application through a branch has a fixed result type; dependent record fields need
  -- their own proved observation rather than an arbitrary transport.
  if codomain.hasLooseBVars then return none
  let arguments := control.getAppArgs
  let rule := if dependent then ``apply_dite else ``apply_ite
  let proof ← withDefault <| mkAppOptM rule #[none, none, some observe, some arguments[1]!,
    some arguments[2]!, some arguments[3]!, some arguments[4]!]
  let some (_, _, result) := (← inferType proof).eq? |
    throwError "roundoff: branch application did not produce an equality"
  return some (result.headBeta, proof)

/-- Split one unresolved Option discriminant using matcher metadata and a constant motive.
Constructor alternatives and all other discriminants retain their actual source expressions. -/
private def optionMatch? (source : Expr) : MetaM (Option (Expr × Expr)) := do
  let source ← instantiateMVars source
  if source.hasExprMVar then return none
  let some name := source.getAppFn.constName? | return none
  let arguments := source.getAppArgs
  let positions? : Option (Array Nat × Nat) ← if let some info ← getMatcherInfo? name then do
      -- Named discriminant equations can retain dependencies on the original values.
      unless info.numDiscrs > 0 && info.getNumDiscrEqs == 0 &&
          arguments.size ≥ info.arity do return none
      let positions := (Array.range info.numDiscrs).map (info.getFirstDiscrPos + ·)
      pure (some (positions, info.getMotivePos))
    else do
      let environment ← getEnv
      let recursor := match environment.find? name with
        | some (.recInfo _) => true
        | _ => isAuxRecursor environment name
      unless recursor do return none
      let info ← mkRecursorInfo name
      unless info.typeName == ``Option && info.numIndices == 0 && !info.recursive &&
          arguments.size ≥ info.numArgs do return none
      pure (some (#[info.majorPos], info.motivePos))
  let some (discriminantPositions, motivePosition) := positions? | return none
  let mut selected? : Option (Nat × Expr) := none
  for index in [:discriminantPositions.size] do
    let discriminant := arguments[discriminantPositions[index]!]!
    if discriminant.isAppOf ``Option.some || discriminant.isAppOf ``Option.none then continue
    let optionType ← withReducible (whnf (← inferType discriminant))
    unless optionType.isAppOfArity ``Option 1 do continue
    selected? := some (index, optionType)
    break
  let some (selected, optionType) := selected? | return none
  let discriminantPosition := discriminantPositions[selected]!
  let discriminant := arguments[discriminantPosition]!
  let motive := arguments[motivePosition]!
  let motiveArguments := discriminantPositions.map (arguments[·]!)
  let resultType ← inferType source
  let program? ← withoutModifyingState <| withNewMCtxDepth do
    try
      withLocalDeclD `option optionType fun option => do
        -- Trailing proof arguments may depend on the original discriminant even when the
        -- fully applied motive is constant. Check the entire replaced application.
        unless ← withReducible (isDefEq (mkAppN motive motiveArguments)
            (mkAppN motive (motiveArguments.set! selected option))) do return none
        let body := mkAppN source.getAppFn (arguments.set! discriminantPosition option)
        check body .default
        unless ← withReducible (isDefEq (← inferType body) resultType) do return none
        return some (← mkLambdaFVars #[option] body)
    catch error =>
      if error.isInterrupt || error.isRuntime then throw error
      return none
  let some program := program? | return none
  -- Split producer control first, so the payload retains its original source.
  let proof ← if let some (_, proof) ← applyBranch? program discriminant then pure proof
    else mkAppM ``option_apply_eq_dite #[program, discriminant]
  let some (_, original, normalized) := (← inferType proof).eq? | return none
  unless ← withReducible (isDefEq original source) do return none
  check proof
  return some (normalized, proof)

/-- A generic producer may be a definitionally trivial wrapper around one of its data
arguments. Remove only such identity wrappers; ordinary tensor producers remain protected. -/
private def identityProducer? (value : Lean.Expr) : MetaM (Option Lean.Expr) :=
  withoutModifyingState do
    let some name := value.getAppFn.constName? | return none
    let declaration ← getConstInfo name
    if ((containsTensor (← getEnv) declaration.type.getForallBody).run {}).1 then return none
    let resultType ← inferType value
    let arguments := value.getAppArgs
    let parameters ← getFunInfoNArgs value.getAppFn arguments.size
    for index in (List.range arguments.size).reverse do
      let some parameter := parameters.paramInfo[index]? | continue
      if parameter.isInstance || parameter.isImplicit || parameter.isStrictImplicit then continue
      let argument := arguments[index]!
      if (← isType argument) || (← isProof argument) then continue
      let sameType ← withNewMCtxDepth <| withDefault <|
        isDefEq resultType (← inferType argument)
      unless sameType do continue
      if ← withNewMCtxDepth <| withDefault <| isDefEq value argument then return some argument
    return none

/-- Expose source code before simplification visits its arguments. Definitional reduction
also works under dependent projections; observational equalities retain their explicit proofs. -/
private def programStep (boundaries : NameSet) : Simp.DSimproc := fun value => do
  if (← isType value) || (← isProof value) then return .continue
  if let .proj structureName index receiver := value.getAppFn then
    if isClass (← getEnv) structureName then return .continue
    let some producer := receiver.getAppFn.constName? | return .continue
    if boundaries.contains producer || (`FloatLib).isPrefixOf producer ||
        (`TorchLean.Tensor.Internal.Rep).isPrefixOf producer then return .continue
    if let some restored ← restoreBranch? receiver then
      return .visit (mkAppN (.proj structureName index restored) value.getAppArgs)
    if receiver.isAppOf ``ite || receiver.isAppOf ``dite then return .continue
    if let some exposed ← withCanUnfoldPred (programCanUnfold boundaries false)
        (withDefault (unfoldDefinition? receiver)) then
      unless exposed == receiver do
        return .visit (mkAppN (.proj structureName index exposed) value.getAppArgs)
    return .continue
  let some name := value.getAppFn.constName? | return .continue
  if (`FloatLib).isPrefixOf name ||
      (`TorchLean.Tensor.Internal.Rep).isPrefixOf name then return .continue
  if boundaries.contains name then
    if let some reduced ← identityProducer? value then return .visit reduced
    return .continue
  let reduce {α : Type} (action : MetaM α) : MetaM α :=
    withCanUnfoldPred (programCanUnfold boundaries false) (withDefault action)
  if let some reduced ← reduceControl? boundaries value then
    if reduced != value then return .visit reduced
  if (name == ``ite || name == ``dite) && value.getAppNumArgs ≥ 5 then
    let type ← withReducible (whnf (← inferType value))
    -- Scalar branches can simplify their coordinate reads without expanding tensor programs
    -- in inactive branches. Their numerical analysis still runs under the original guard.
    return if ((containsTensor (← getEnv) type.getForallBody).run {}).1 then
      .done value else .continue
  if (← getMatcherInfo? name).isSome ||
      ((← getEnv).find? name).any (fun info => match info with
        | .recInfo _ => true | _ => false) then
    -- Scalar alternatives may simplify their guards. Tensor-producing alternatives stay
    -- intact until the actual discriminant determines which program is executed.
    let type ← withReducible (whnf (← inferType value))
    return if ((containsTensor (← getEnv) type.getForallBody).run {}).1 then
      .done value else .continue
  if let some projection ← getProjectionFnInfo? name then
    if projection.fromClass then
      -- Only data fields of the selected dictionary are metadata. Operand-taking methods
      -- retain their source computation and numerical obligations.
      if value.getAppNumArgs == projection.numParams + 1 then
        let type ← withReducible (whnf (← inferType value))
        if type.isConstOf ``Bool || type.isConstOf ``Nat then
          if let some selected ← withReducibleAndInstances (unfoldProjInst? value) then
            -- Numerals are already selected data. Revisiting a raw Nat literal would let
            -- simp re-encode it through OfNat and repeat the same projection indefinitely.
            if selected.consumeMData.rawNatLit?.isSome then return .done selected
            unless selected == value do return .visit selected
      -- Optional results retain the selected method and its operands. Revealing that
      -- producer lets exact observations follow a successful result without treating it as data.
      let resultType ← withReducible (whnf (← inferType value))
      if resultType.isAppOf ``Option then
        if let some selected ← withReducibleAndInstances (unfoldProjInst? value) then
          unless selected == value do return .visit selected
      return .continue
  let type ← withReducible (whnf (← inferType value))
  unless ((containsTensor (← getEnv) type.getForallBody).run {}).1 do return .continue
  if let some reduced ← reduce (unfoldDefinition? value) then
    if reduced != value then return .visit reduced
  return .continue

/-- Read tensor and record branches through the same proved function-application rule. -/
private def observeBranch : Simp.Simproc := fun expression => do
  if let .proj structureName index receiver := expression then
    if isClass (← getEnv) structureName then return .continue
    unless receiver.isAppOfArity ``ite 5 || receiver.isAppOfArity ``dite 5 do
      return .continue
    let observe ← withLocalDeclD `record (← inferType receiver) fun record =>
      mkLambdaFVars #[record] (.proj structureName index record)
    let some (result, proof) ← applyBranch? observe receiver | return .continue
    return .visit { expr := result, proof? := some proof }
  unless expression.isAppOfArity ``TorchLean.Tensor.Internal.Rep.get 5 do return .continue
  let arguments := expression.getAppArgs
  let tensor := arguments[3]!
  unless tensor.isAppOfArity ``ite 5 || tensor.isAppOfArity ``dite 5 do return .continue
  let observe ← withLocalDeclD `tensor (← inferType tensor) fun tensor =>
    mkLambdaFVars #[tensor] (mkAppN expression.getAppFn (arguments.set! 3 tensor))
  let some (result, proof) ← applyBranch? observe tensor | return .continue
  return .visit { expr := result, proof? := some proof }

/-- A successful getter agrees with a constant-motive observation of the original option. -/
private def optionGet? (expression : Expr) : MetaM (Option (Expr × Expr)) := do
  unless expression.isAppOfArity ``Option.get 3 do return none
  if expression.hasExprMVar then return none
  let arguments := expression.getAppArgs
  let producer ← withCanUnfoldPred (programCanUnfold {} false) <|
    withDefault (whnf arguments[1]!)
  unless producer.isAppOf ``Option.some || producer.isAppOf ``Option.none ||
      producer.isAppOfArity ``ite 5 || producer.isAppOfArity ``dite 5 do return none
  let capability ← mkAppM ``OfNat #[arguments[0]!, mkNatLit 0]
  let .some numeral ← trySynthInstance capability | return none
  let fallback ← mkAppOptM ``OfNat.ofNat
    #[some arguments[0]!, some (mkNatLit 0), some numeral]
  let proof ← mkAppOptM ``Option.get_eq_getD
    #[some arguments[0]!, some fallback, some arguments[1]!, some arguments[2]!]
  let some (_, _, result) := (← inferType proof).eq? | return none
  return some (result, proof)

/-- A paired observation also supplies its component observations. Derive these proofs directly
from the existing equality rather than requiring two wrapper lemmas for every adapter. -/
private def projectedObservations (name : Name) : MetaM (Array Lean.Expr) := do
  let declaration ← getConstInfo name
  let rule := mkConst name (declaration.levelParams.map Level.param)
  forallTelescopeReducing declaration.type fun arguments conclusion => do
    let some (type, left, right) := conclusion.eq? | return #[]
    unless left.isAppOfArity ``Prod.mk 4 do return #[]
    let original := mkAppN rule arguments
    let mut proofs := #[]
    for index in [:2] do
      let projection ← withLocalDeclD `pair type fun pair =>
        mkLambdaFVars #[pair] (.proj ``Prod index pair)
      let proof ← mkAppM ``congrArg #[projection, original]
      -- Index the component observation, before reduction can expose its representation.
      let expected ← mkEq left.getAppArgs[2 + index]! (.proj ``Prod index right)
      let proof := mkApp2 (mkConst ``id [0]) expected proof
      proofs := proofs.push (← mkLambdaFVars arguments proof)
    return proofs

/-- Observational equalities remove representation plumbing. Compiler equalities recover the
source computation, including its operation order. Scalar accuracy is handled separately. -/
private def addProgramRewrites (theorems : SimpTheorems) :
    MetaM (SimpTheorems × Simp.SimprocsArray) := do
  let mut theorems := theorems
  -- Conditional coordinate laws can use the caller's equalities and range exclusions.
  for declaration in ← getLCtx do
    if declaration.isImplementationDetail then continue
    let type := declaration.type
    if type.isAppOf ``Eq || type.isAppOf ``Iff || type.isAppOf ``Not then
      theorems ← theorems.add (.fvar declaration.fvarId) #[] declaration.toExpr
    -- Derive the Boolean capability without rewriting descriptors inside scalar types.
    if let some (_, predicate, result) := type.eq? then
      if predicate.isAppOfArity ``FloatFormat.isIEEE 1 && result.isConstOf ``Bool.true then
        let descriptor : Q(FloatFormat) := predicate.getAppArgs[0]!
        let expected : Q(Prop) :=
          q((FloatFormat.encoding $descriptor == FloatFormat.Encoding.ieee) = true)
        let proof ← mkAppM ``FloatFormat.supportsInfinity_eq_true_of_isIEEE
          #[descriptor, declaration.toExpr]
        let proof := mkApp2 (mkConst ``id [0]) expected proof
        check proof
        theorems ← theorems.add (.fvar declaration.fvarId) #[] proof (post := false)
  -- Preserve the producer when observing a successful mapped coefficient. A rational
  -- payload can use the existing total decoder law once get is written as getD zero.
  theorems ← theorems.addConst ``Option.get_map (post := false)
  let ratGet : Q(∀ (option : Option Rat) (found : option.isSome),
      option.get found = option.getD (0 : Rat)) :=
    q(fun option found => @Option.get_eq_getD Rat 0 option found)
  theorems ← theorems.add (.decl ``Option.get_eq_getD) #[] ratGet (post := false)
  -- Simplify logical guards and identity transports before exposing their implementation.
  for name in #[``eq_self, ``not_false_eq_true, ``not_true_eq_false,
      ``Bool.and_false, ``Bool.and_true, ``Bool.false_and, ``Bool.true_and,
      ``Bool.or_false, ``Bool.or_true, ``Bool.false_or, ``Bool.true_or,
      ``Bool.not_false, ``Bool.not_true, ``bne_self_eq_false,
      ``ite_eq_left, ``ite_eq_right, ``dite_eq_left, ``dite_eq_right,
      ``TorchLean.Tensor.broadcastTo_self, ``TorchLean.Tensor.cast_shape_self,
      ``ite_apply, ``dite_apply] do
    theorems ← theorems.addConst name (post := false)
  let (rules, boundaries) ← programRules
  for (name, inverse) in rules do
    theorems ← theorems.addConst name (inv := inverse) (post := false)
    unless inverse do
      for proof in ← projectedObservations name do
        let declaration ← getConstInfo name
        theorems ← theorems.add (.decl name) declaration.levelParams.toArray proof (post := false)
  let procedures := ({} : Simp.Simprocs).addCore #[.star] ``programStep false
    (.inr (programStep boundaries))
  let procedures := procedures.addCore #[.star] ``observeBranch false (.inl observeBranch)
  return (theorems, #[procedures])

/-- Normalize tensor observations without unfolding their buffers or arithmetic dictionaries. -/
private def coordinateSimpContext : MetaM (Simp.Context × Simp.SimprocsArray) := do
  let mut theorems : SimpTheorems := {}
  for name in #[``TorchLean.Tensor.addSpec, ``TorchLean.Tensor.subSpec,
      ``TorchLean.Tensor.mulSpec, ``TorchLean.Tensor.divSpec, ``TorchLean.Tensor.scaleSpec,
      ``TorchLean.Tensor.squareSpec, ``TorchLean.Tensor.negSpec,
      ``TorchLean.Tensor.expSpec, ``TorchLean.Tensor.logSpec,
      ``TorchLean.Tensor.map, ``TorchLean.Tensor.item,
      ``TorchLean.Tensor.ofFn, ``TorchLean.Tensor.generate,
      ``TorchLean.Tensor.Internal.Rep.coordinates,
      ``TorchLean.Tensor.Internal.Rep.broadcast,
      ``TorchLean.Tensor.Internal.Coord.linearizeFast,
      ``TorchLean.Tensor.Internal.Coord.linearizeAux] do
    theorems ← theorems.addDeclToUnfold name
  for name in #[``Spec.get2_eq_apply,
      ``TorchLean.Tensor.getScalar_eq_apply, ``TorchLean.Tensor.map2Spec_apply,
      ``TorchLean.Tensor.full_apply, ``TorchLean.Tensor.scalar_apply,
      ``TorchLean.Tensor.Internal.Rep.getFlat_eq_get,
      ``TorchLean.Tensor.Internal.Rep.get_ofFn, ``TorchLean.Tensor.Internal.Rep.get_ofFlatFn,
      ``TorchLean.Tensor.Internal.Rep.const_apply, ``TorchLean.Tensor.Internal.Rep.map_apply,
      ``TorchLean.Tensor.Internal.Rep.zipWith_apply,
      ``TorchLean.Tensor.Internal.Rep.pull_apply, ``TorchLean.Tensor.Internal.Coord.broadcast_self,
      ``TorchLean.Tensor.Internal.Coord.linearize_eq_linearizeFast, ``Nat.zero_mul, ``Nat.zero_add,
      ``TorchLean.Tensor.Internal.Rep.stack_apply, ``TorchLean.Tensor.Internal.Rep.unstack_apply,
      ``TorchLean.Tensor.Internal.Rep.reshape_apply,
      ``TorchLean.Tensor.swapAdjacentAxes_apply, ``TorchLean.Tensor.permuteByAdjacentSwaps_apply,
      ``TorchLean.Tensor.sliceAxisRangeSpec_apply,
      ``Bool.cond_eq_ite,
      ``TorchLean.Tensor.Reduction.Internal.reduceOuterAxis_vector,
      ``TorchLean.Tensor.Reduction.Internal.reduceDimCore_scalar,
      ``TorchLean.Tensor.Reduction.Internal.reduceDimCore_dim_zero,
      ``TorchLean.Tensor.Reduction.Internal.reduceDimCore_dim_succ] do
    theorems ← theorems.addConst name (post := false)
  -- Converse slice laws keep checked backward slices at the level of their scalar reads.
  for name in #[`TorchLean.Tensor.sliceAxisRangeBackwardSpec_apply_coordinate,
      `TorchLean.Tensor.sliceAxisRangeBackwardSpec_apply_eq_zero] do
    if (← getEnv).contains name then
      theorems ← theorems.addConst name (post := false)
  let (prepared, procedures) ← addProgramRewrites theorems
  let context ← Simp.mkContext
    { failIfUnchanged := false, implicitDefEqProofs := false } #[prepared]
    (← getSimpCongrTheorems)
  return (context, procedures)

/-- Fuse source maps into loops using proved traversal equalities, without exposing storage. -/
private def foldSimpContext : MetaM (Simp.Context × Simp.SimprocsArray) := do
  let mut theorems : SimpTheorems := {}
  for name in #[``TorchLean.Tensor.sumSpec, ``TorchLean.Tensor.dotSpec,
      ``TorchLean.Tensor.foldlSpec, ``TorchLean.Tensor.foldl,
      ``TorchLean.Tensor.ofFn, ``TorchLean.Tensor.generate,
      ``TorchLean.Tensor.map, ``TorchLean.Tensor.map2Spec,
      ``TorchLean.Tensor.addSpec, ``TorchLean.Tensor.subSpec,
      ``TorchLean.Tensor.mulSpec, ``TorchLean.Tensor.divSpec, ``TorchLean.Tensor.scaleSpec,
      ``TorchLean.Tensor.squareSpec, ``TorchLean.Tensor.negSpec,
      ``TorchLean.Tensor.expSpec, ``TorchLean.Tensor.logSpec,
      ``TorchLean.Tensor.Internal.Rep.pull] do
    theorems ← theorems.addDeclToUnfold name
  for name in #[``TorchLean.Tensor.Internal.Rep.ofFn_eq_map_coordinates,
      ``TorchLean.Tensor.Internal.Rep.zipWith_eq_map_coordinates,
      ``TorchLean.Tensor.Internal.Rep.foldl_map, ``AllSteps.map, ``foldError_map,
      ``Bool.cond_eq_ite] do
    theorems ← theorems.addConst name
  theorems ← theorems.addConst ``Fin.foldl_eq_foldl_finRange (inv := true)
  theorems ← theorems.addConst ``fin_foldl_eq_coordinates
  let (prepared, procedures) ← addProgramRewrites theorems
  let context ← Simp.mkContext
    { implicitDefEqProofs := false, failIfUnchanged := false } #[prepared]
    (← getSimpCongrTheorems)
  return (context, procedures)

/-- Compare configured computations through their proved FloatLib meanings, rather than asking
different execution backends to have definitionally identical implementations. -/
private def semanticContext (decodeClassifiers : Bool := false) : MetaM Simp.Context := do
  let mut theorems : SimpTheorems := {}
  for name in #[``FloatLib.Floats.ExecFloat.Binary.toReal,
      ``FloatLib.Floats.ExecFloat.Binary.isFinite, ``FloatLib.Floats.ExecFloat.Binary.isZero,
      ``FloatLib.Floats.ExecFloat.Binary.signBit, ``Except.toOption, ``Option.getD] do
    theorems ← theorems.addDeclToUnfold name
  for field in (getStructureInfo (← getEnv) ``MathFunctions).fieldInfo do
    theorems ← theorems.addDeclToUnfold field.projFn
  if decodeClassifiers then
    -- This mode compares fresh Boolean propositions only. Other source normalization keeps
    -- the public decoder intact so an unknown producer cannot become an exact input.
    for name in #[``FloatLib.Floats.ExecFloat.Binary.toModel,
        ``FloatLib.Floats.ExecFloat.Binary.ofModel, ``Rat.cast, ``RatCast.ratCast,
        ``OfNat.ofNat, ``OfScientific.ofScientific] do
      theorems ← theorems.addDeclToUnfold name
  for name in #[``FloatLib.Floats.ExecFloat.Binary.toModel_hAdd,
      ``FloatLib.Floats.ExecFloat.Binary.toModel_hSub,
      ``FloatLib.Floats.ExecFloat.Binary.toModel_hMul,
      ``FloatLib.Floats.ExecFloat.Binary.toModel_hDiv,
      ``FloatLib.Floats.ExecFloat.Binary.toModel_neg,
      ``FloatLib.Floats.ExecFloat.Binary.toModel_abs,
      ``FloatLib.Floats.ExecFloat.Binary.toModel_ofModel, ``decode_family, ``decode_unary,
      ``sub_self, ``abs_zero, ``zero_add, ``add_zero, ``zero_mul, ``mul_zero,
      ``zero_div, ``Real.sqrt_zero, ``Nat.cast_zero, ``Nat.cast_one,
      ``Rat.cast_natCast, ``Rat.cast_intCast, ``Rat.cast_ofScientific,
      ``Rat.cast_zero, ``Rat.cast_one] do
    theorems ← theorems.addConst name (post := !decodeClassifiers)
  if decodeClassifiers then
    theorems ← theorems.addConst ``Configured.Family.toModel_ofModel (post := false)
  theorems ← theorems.addConst ``toReal_zero (post := false)
  -- A caller may have already checked the reference and recorded its exact returned value.
  for declaration in ← getLCtx do
    unless declaration.type.isAppOfArity ``Eq 3 do continue
    unless declaration.type.getAppArgs[0]!.isAppOf ``Except do continue
    theorems ← theorems.add (.fvar declaration.fvarId) #[] declaration.toExpr (post := false)
  Simp.mkContext { implicitDefEqProofs := false, zetaDelta := false } #[theorems]

/-- Build congruence without first comparing whole loop functions by reduction. That comparison
would expose their floating-point implementations before the decoding rules can be used. -/
private def congrLoopArguments (goal : MVarId) : MetaM (Array MVarId) := do
  let target ← goal.getType
  let args := target.getAppArgs
  let left := args[1]!
  let right := args[2]!
  let arity := if left.isAppOf ``AllSteps then 4 else if left.isAppOf ``foldError then 7 else 0
  if arity == 0 then throwError "roundoff: no loop congruence for this expression"
  let mut current := left.getAppArgs
  let desired := right.getAppArgs
  unless current.size == desired.size do throwError "roundoff: incompatible loop applications"
  for index in [:current.size - arity] do
    unless ← withTransparency .reducible (isDefEq current[index]! desired[index]!) do
      throwError "roundoff: loop types or storage differ"
  let mut proof ← mkEqRefl left
  let mut pending := #[]
  for index in [current.size - arity:current.size] do
    if current[index]! == desired[index]! then continue
    let equality ← mkFreshExprMVar (← mkEq current[index]! desired[index]!)
    let argumentType ← inferType current[index]!
    let replacement ← withLocalDeclD `argument argumentType fun argument =>
      mkLambdaFVars #[argument] (mkAppN left.getAppFn (current.set! index argument))
    let step ← mkAppM ``congrArg #[replacement, equality]
    proof ← mkAppM ``Eq.trans #[proof, step]
    current := current.set! index desired[index]!
    pending := pending.push equality.mvarId!
  goal.assign proof
  return pending

/-- Introduce function arguments without a reflexivity attempt on their numeric bodies. -/
private partial def pointwiseGoal (goal : MVarId) : MetaM MVarId := goal.withContext do
  let target ← goal.getType
  unless target.isAppOfArity ``Eq 3 do return goal
  let left := target.getAppArgs[1]!
  unless (← withTransparency .reducible (whnf (← inferType left))).isForall do return goal
  let [pointwise] ← goal.apply (← mkConstWithFreshMVarLevels ``funext)
    { synthAssignedInstances := false }
    | throwError "roundoff: unexpected function equality"
  let (_, body) ← pointwise.intro1
  pointwiseGoal body

/-- Prove equal observations of implementations with the same floating-point semantics.
Congruence stops at one application layer; scalar values are compared by exact decoding. -/
private def semanticEquality (goal : MVarId) : TacticM Bool := do
  let target ← goal.getType
  unless target.isAppOfArity ``Eq 3 do return false
  let arguments := target.getAppArgs
  unless arguments[1]!.getAppFn == arguments[2]!.getAppFn do return false
  let succeeded? ← attempt? 10000 do
    let context ← semanticContext
    let arguments ← congrLoopArguments goal
    let mut complete := true
    for argument in arguments do
      let pointwise ← pointwiseGoal argument
      let leaves ← Lean.Elab.Tactic.run pointwise do
        evalTactic (← `(tactic| try apply FloatLib.Floats.ExecFloat.Binary.toModel_inj.mp))
      for leaf in leaves do
        let (next, _) ← simpTarget leaf context
        if let some next := next then
          let closed ← try
            next.withContext (withTransparency .reducible next.refl)
            pure true
          catch _ => pure false
          unless closed do
            trace[TorchLean.roundoff] "semantic equality remains: {← next.getType}"
            complete := false
    return if complete then some () else none
  return succeeded?.isSome

/-- Transport an error proof across a proved normalization of its actual computation. -/
private def rewriteActual (estimate : Estimate) (equality : Lean.Expr) : MetaM Estimate := do
  let reference := estimate.reference
  let error := estimate.error
  let proposition ← mkAppM ``congrArg #[q(fun value : ℝ => |value - $reference| ≤ $error), equality]
  return { estimate with proof := ← mkAppM ``Eq.mpr #[proposition, estimate.proof] }

/-- Recognize scalar reads and coordinate maps that can still hide tensor indexing work. -/
private def hasTensorObservation (expression : Lean.Expr) : Bool :=
  (expression.find? fun term =>
    #[``Spec.get2, ``TorchLean.Tensor.getScalar, ``TorchLean.Tensor.item,
      ``TorchLean.Tensor.map2Spec, ``TorchLean.Tensor.Internal.Rep.get,
      ``TorchLean.Tensor.Internal.Rep.getFlat,
      ``CoeFun.coe, ``GetElem.getElem, ``TorchLean.Tensor.Internal.Coord.linearize].contains
        (term.getAppFn.constName?.getD .anonymous)).isSome

/-- Data fields of an input are inputs too. Class methods and applied function fields still
need their implementation or a numerical rule; treating them as data would hide computation. -/
private partial def isInputValue (value : Lean.Expr) : MetaM Bool := do
  match value with
  | .fvar id => return (← id.getDecl).value?.isNone
  | .proj name _ source =>
      if isClass (← getEnv) name then return false
      isInputValue source
  | .mdata _ source => isInputValue source
  | _ =>
      let some name := value.getAppFn.constName? | return false
      let some field ← getProjectionFnInfo? name | return false
      if field.fromClass || value.getAppNumArgs != field.numParams + 1 then return false
      isInputValue value.getAppArgs[field.numParams]!

/-- Exact discrete data uses the complete canonical real cast, including its dictionary.
Computed rational expressions instead follow their registered algebra and observation laws. -/
private def exactRealData? (actual : Q(ℝ)) : MetaM Bool := do
  let actual : Q(ℝ) ← instantiateMVars actual
  if actual.hasExprMVar then return false
  let head := actual.getAppFn.constName?.getD .anonymous
  let some argument := actual.getAppArgs.back? | return false
  if head == ``Nat.cast || head == ``NatCast.natCast then
    unless ← withReducible (isDefEq (← inferType argument) q(Nat)) do return false
    let value : Q(Nat) := argument
    return ← withNewMCtxDepth <| withReducible <| isDefEq actual q(($value : ℝ))
  if head == ``Int.cast || head == ``IntCast.intCast then
    unless ← withReducible (isDefEq (← inferType argument) q(Int)) do return false
    let value : Q(Int) := argument
    return ← withNewMCtxDepth <| withReducible <| isDefEq actual q(($value : ℝ))
  if head == ``Rat.cast || head == ``RatCast.ratCast then
    unless ← withReducible (isDefEq (← inferType argument) q(Rat)) do return false
    unless ← isInputValue argument do return false
    let value : Q(Rat) := argument
    return ← withNewMCtxDepth <| withReducible <| isDefEq actual q(($value : ℝ))
  return false

/-- Only canonical mathematical real literals are exact data. Constructed floating-point
literals retain their conversion through the selected codec and its numerical law. -/
private def literalInput? (actual : Q(ℝ)) (source : Lean.Expr) : MetaM Bool := do
  let source ← instantiateMVars source
  if source.hasExprMVar then return false
  unless ← withReducible (isDefEq (← inferType source) q(ℝ)) do return false
  let head := source.getAppFn.constName?.getD .anonymous
  unless literalHeads.contains head do return false
  if head == ``Real.pi then return actual == source
  let arguments := source.getAppArgs
  let expected? : Option Lean.Expr ←
    if head == ``OfNat.ofNat && arguments.size == 3 then do
      let value : Q(Nat) := arguments[1]!
      let some literal := value.consumeMData.rawNatLit? | return false
      let real : Q(ℝ) := if literal == 0 then q((0 : ℝ))
        else if literal == 1 then q((1 : ℝ)) else q(($value : ℝ))
      pure (some real)
    else if head == ``OfScientific.ofScientific && arguments.size == 5 then do
      let mantissa : Q(Nat) := arguments[2]!
      let negative : Q(Bool) := arguments[3]!
      let exponent : Q(Nat) := arguments[4]!
      unless mantissa.consumeMData.rawNatLit?.isSome &&
          exponent.consumeMData.rawNatLit?.isSome &&
          (negative.isConstOf ``Bool.true || negative.isConstOf ``Bool.false) do return false
      pure (some q((OfScientific.ofScientific $mantissa $negative $exponent : ℝ)))
    else pure none
  let some expected := expected? | return false
  return ← withNewMCtxDepth <| withReducibleAndInstances <| isDefEq source expected

/-- Read a real absolute-error proposition without changing its arithmetic dictionaries. -/
private def errorConclusion? (type : Lean.Expr) :
    MetaM (Option (Lean.Expr × Lean.Expr × Lean.Expr)) := do
  unless ← isProp type do return none
  let type : Q(Prop) ← pure type
  match type with
  | ~q(|($actual : ℝ) - $reference| ≤ $error) =>
      unless ← isDefEq (← inferType actual) q(ℝ) do return none
      unless ← isDefEq type q(|$actual - $reference| ≤ $error) do return none
      return some (actual, reference, error)
  | _ => return none

/-- Extract the observation and its proposition context without reducing scalar arithmetic. -/
private def numericalConclusion? (type : Lean.Expr) :
    MetaM (Option (Lean.Expr × Lean.Expr × Bool)) := do
  let type : Q(Prop) ← pure type
  if let some (scalar, actual, reference) := type.eq? then
    unless ← isDefEq scalar q(ℝ) do return none
    let reference : Q(ℝ) ← instantiateMVars reference
    return some (← instantiateMVars actual, q(fun value : ℝ => value = $reference), true)
  let some (actual, reference, error) ← errorConclusion? type | return none
  let reference : Q(ℝ) ← pure reference
  let error : Q(ℝ) ← pure error
  return some (actual, q(fun value : ℝ => |value - $reference| ≤ $error), false)

/-- Abstract the theorem parameters left after specializing its local rounding premise. -/
private partial def closeRuleArguments (arguments : Array Lean.Expr) (binders : Array BinderInfo)
    (index : Nat) (opened : Array Lean.Expr) (proof : Lean.Expr) : MetaM Lean.Expr := do
  if index == arguments.size then
    return ← mkLambdaFVars opened (← instantiateMVars proof)
  let argument ← instantiateMVars arguments[index]!
  unless argument.isMVar do
    return ← closeRuleArguments arguments binders (index + 1) opened proof
  let goal := argument.mvarId!
  let declaration ← goal.getDecl
  let type ← instantiateMVars declaration.type
  withLocalDecl declaration.userName binders[index]! type fun value => do
    goal.assign value
    closeRuleArguments arguments binders (index + 1) (opened.push value) proof

/-- A scalar error transformer also describes an exact real operation. Discharge its local
rounding premise with reflexivity, retaining its input-error and domain premises. -/
private def specializeLocalError (rule : Lean.Expr) : MetaM Lean.Expr := do
  let (arguments, binders, conclusion) ← forallMetaTelescopeReducing (← inferType rule)
  let some (actual, _, _) ← errorConclusion? conclusion | return rule
  unless actual.isMVar do return rule
  for argument in arguments do
    let type ← instantiateMVars (← inferType argument)
    let some (value, middle, _) ← errorConclusion? type | continue
    unless value == actual do continue
    let saved ← saveState
    let exactProof ← mkAppM ``exact_error #[middle]
    unless ← isDefEq type (← inferType exactProof) do
      saved.restore
      continue
    argument.mvarId!.assign exactProof
    return ← closeRuleArguments arguments binders 0 #[] (mkAppN rule arguments)
  return rule

/-- Normalize a rule's tensor reads by an equality proof, retaining its original premises,
reference and budget. Rigid parameters prevent simplification from choosing theorem arguments. -/
private def normalizeRule (rule : Lean.Expr) : MetaM Lean.Expr := do
  forallTelescopeReducing (← inferType rule) fun arguments conclusion => do
    let some (actual, proposition, _) ← numericalConclusion? conclusion | return rule
    unless hasTensorObservation actual do return rule
    let (context, procedures) ← coordinateSimpContext
    let (normalized, _) ← simp actual context procedures
    if normalized.expr == actual then return rule
    let equality ← mkAppM ``congrArg #[proposition, ← normalized.getProof' actual]
    let proof ← mkAppM ``Eq.mp #[equality, mkAppN rule arguments]
    mkLambdaFVars arguments proof

/-- Index complete computation patterns. The tree ignores instance arguments; applying a
candidate still checks its complete statement, including every selected dictionary. -/
private partial def numericalPattern (expression : Lean.Expr) (root : Bool := true) :
    MetaM Lean.Expr := do
  let expression ← DiscrTree.reduceDT expression root
  let mut function := expression.getAppFn
  if let .proj name index source := function then
    let source ← if isClass (← getEnv) name then pure (DiscrTree.mkNoindexAnnotation source)
      else numericalPattern source false
    function := .proj name index source
  let arguments := expression.getAppArgs
  let info ← getFunInfoNArgs function arguments.size
  let mut prepared := #[]
  for h : index in [:arguments.size] do
    let argument := arguments[index]
    let ignore := info.paramInfo[index]?.any fun parameter =>
      parameter.isInstance || parameter.isImplicit || parameter.isStrictImplicit
    let argument ← if ignore || (← isType argument) || (← isProof argument) then
        pure (DiscrTree.mkNoindexAnnotation argument)
      else numericalPattern argument false
    prepared := prepared.push argument
  return mkAppN function prepared

/-- A scalar observation has one explicit data input. Implicit data must be parameters of
that input's type, rather than captured tensor values or additional scalar operands.
Instance arguments remain intact and are checked when a numerical theorem is applied. -/
private def isScalarObservation (expression : Lean.Expr) : MetaM Bool := do
  let head := expression.getAppFn.constName?.getD .anonymous
  if #[``Nat.cast, ``NatCast.natCast, ``Int.cast, ``IntCast.intCast].contains head then return false
  -- Exact rational reductions use their canonical real observation. A different selected
  -- RatCast dictionary must expose its computation instead of becoming a decoder.
  if head == ``Rat.cast || head == ``RatCast.ratCast then
    let some argument := expression.getAppArgs.back? | return false
    unless ← withReducible (isDefEq (← inferType argument) q(Rat)) do return false
    let value : Q(Rat) := argument
    return ← withNewMCtxDepth <| withReducible <| isDefEq expression q(($value : ℝ))
  let arguments := expression.getAppArgs
  let some input := arguments.back? | return false
  let info ← getFunInfoNArgs expression.getAppFn arguments.size
  let mut functionType ← inferType expression.getAppFn
  let mut inputTypes := #[← instantiateMVars (← inferType input)]
  let mut implicitData := #[]
  let mut dataInputs := 0
  let mut finalInput := false
  for h : index in [:arguments.size] do
    let some parameter := info.paramInfo[index]? | return false
    let argument := arguments[index]
    let .forallE _ domain body _ ← withReducible (whnf functionType) | return false
    functionType := body.instantiate1 argument
    if index + 1 == arguments.size then
      inputTypes := inputTypes.push (← instantiateMVars domain)
    if parameter.isInstance then continue
    if (← isType argument) || (← isProof argument) then continue
    if parameter.isImplicit || parameter.isStrictImplicit then
      implicitData := implicitData.push (← instantiateMVars argument)
      continue
    dataInputs := dataInputs + 1
    finalInput := index + 1 == arguments.size
  if dataInputs != 1 || !finalInput then return false
  if ((containsTensor (← getEnv) inputTypes[0]!).run {}).1 then return false
  -- Retain the original types: reducing a carrier to its storage type can erase the
  -- descriptor and storage-plan parameters that its observation still needs.
  for type in inputTypes do
    inputTypes := inputTypes.push (← withReducible (whnf type))
  for argument in implicitData do
    if ((containsTensor (← getEnv) (← inferType argument)).run {}).1 then return false
    unless inputTypes.any (fun type =>
        (type.find? fun term => term.consumeMData == argument.consumeMData).isSome) do
      return false
  return !(← withReducible (whnf (← inferType input))).isConstOf ``Real

/-- An arithmetic observation law reads smaller operands with the same observer in its
real reference. An echo bound, or a reference that reuses the complete computation, does not
identify a decoder. The complete numerical theorem remains available in every case. -/
private def referenceObservesOperands (actual reference : Lean.Expr) : MetaM Bool := do
  unless ← isScalarObservation actual do return false
  let some name := actual.getAppFn.constName? | return false
  let some input := actual.getAppArgs.back? | return false
  let input := input.consumeMData
  let found ← IO.mkRef false
  reference.forEach' fun term => do
    if ← found.get then return false
    let term := term.consumeMData
    if term.getAppFn.constName? != some name ||
        term.getAppNumArgs != actual.getAppNumArgs || term.hasLooseBVars then return true
    let some operand := term.getAppArgs.back? | return true
    let operand := operand.consumeMData
    if operand == input ||
        (input.find? fun child => child.consumeMData == operand).isNone then return true
    let matched ← isScalarObservation term
    found.set matched
    return !matched
  return ← found.get

/-- Type arguments are wildcards even when a format is reconstructed from its widths. The
retrieved theorem must still unify with the original, fully typed computation. -/
private def indexRules (boundaries : NameSet) : TacticM (DiscrTree (Name × Bool) × NameSet) := do
  let environment ← getEnv
  let mut entries : DiscrTree (Name × Bool) := {}
  let mut observations : NameSet := {}
  for name in roundoffRules.getState environment do
    unless environment.contains name do continue
    let entry? ← withNewMCtxDepth do
      let rule ← specializeLocalError (← mkConstWithFreshMVarLevels name)
      let (_, _, conclusion) ← forallMetaTelescopeReducing (← inferType rule)
      do
        let some (actual, _, exact) ← numericalConclusion? conclusion | return none
        let reference? ← if exact then
            pure (conclusion.eq?.map fun (_, _, reference) => reference)
          else do
            pure ((← errorConclusion? conclusion).map fun (_, reference, _) => reference)
        let some reference := reference? | return none
        let reference ← instantiateMVars reference
        let actual ← if hasTensorObservation actual then withNewMCtxDepth do
            let (context, procedures) ← coordinateSimpContext
            let (normalized, _) ← simp actual context procedures
            pure normalized.expr
          else pure actual
        let pattern ← withReducible <|
          withCanUnfoldPred (programCanUnfold boundaries false) (numericalPattern actual)
        let indexed ← withCanUnfoldPred (fun _ _ => pure false)
          (entries.insertIfSpecific pattern (name, exact))
        let observation? ← if ← referenceObservesOperands actual reference then
            pure actual.getAppFn.constName?
          else pure none
        pure (some (indexed, observation?))
    if let some (indexed, observation?) := entry? then
      trace[TorchLean.roundoff] "rule {name}: {observation?.getD .anonymous}"
      entries := indexed
      if let some observation := observation? then observations := observations.insert observation
  return (entries, observations)

/-- Dispatch proof search by rigid proposition heads. Unknown and conjunctive conclusions
remain candidates; applying the proof still checks all arguments and selected instances. -/
private def conditionHead (type : Expr) (classifiers : Bool := false) : Option Name :=
  let conclusion := type.getForallBody.consumeMData
  if conclusion.isAppOf ``And || conclusion.isAppOf ``Iff || conclusion.isConstOf ``False then
    none
  else if classifiers && conclusion.isAppOfArity ``Eq 3 then
    if conclusion.getAppArgs[0]!.isConstOf ``Bool then
      conclusion.getAppArgs[1]!.getAppFn.constName?
    else none
  else conclusion.getAppFn.constName?

private def conditionsMayMatch (left right : Expr) (classifiers : Bool := false) : Bool :=
  match conditionHead left classifiers, conditionHead right classifiers with
  | some left, some right => left == right
  | _, _ => true

/-- Compare predicate heads without evaluating floating-point classifiers. Ordinary aliases may
expose those heads; dependent propositions and unresolved control flow remain conservative. -/
private def assumptionHead (type : Expr) : MetaM (Option Name) := do
  if type.isForall then return none
  let conclusion := type.consumeMData
  let some (scalar, predicate, _) := conclusion.eq? | return conditionHead type true
  unless scalar.isConstOf ``Bool do return none
  let classifier ← withCanUnfoldPred (fun config info => do
      if (← getProjectionFnInfo? info.name).any (·.fromClass) then return false
      programCanUnfold {} false config info) <| withDefault (whnf predicate)
  if classifier.isConstOf ``Bool.true || classifier.isConstOf ``Bool.false then return none
  if let some name := classifier.getAppFn.constName? then
    if (← getMatcherInfo? name).isSome || isAuxRecursor (← getEnv) name ||
        ((← getEnv).find? name).any (fun info => match info with
          | .recInfo _ => true | _ => false) then
      return predicate.getAppFn.constName?
  return classifier.getAppFn.constName?

/-- Attempt inexpensive side conditions without hiding an unsolved premise. -/
private def dischargeCore (goal : MVarId) : TacticM Bool := goal.withContext do
  let target ← instantiateMVars (← goal.getType)
  let mut supplied := false
  let targetHead ← assumptionHead target
  for declaration in ← getLCtx do
    if declaration.isImplementationDetail then continue
    let type ← instantiateMVars declaration.type
    if type == target then
      goal.assign declaration.toExpr
      return true
    if !supplied && (← isProp type) then
      supplied := match ← assumptionHead type, targetHead with
        | some source, some target => source == target
        | _, _ => true
  let mut attempts := #[
    ← `(tactic| simp (config := { zetaDelta := false, implicitDefEqProofs := false }) only
        [*, Except.isOk, Except.toBool, Except.toOption, Option.getD,
        sub_self, abs_zero, zero_add, add_zero, zero_mul, mul_zero, zero_div,
        toReal_zero, foldError_identity, le_refl])]
  if supplied then attempts := attempts.push (← `(tactic| assumption))
  let helpers : Array (Name × TSyntax `tactic) := #[
    (``isFinite_zero, ← `(tactic| apply isFinite_zero)),
    (``FloatFormat.supportsSignedZero_eq_true_of_isIEEE,
      ← `(tactic| (apply FloatFormat.supportsSignedZero_eq_true_of_isIEEE; assumption))),
    (``Real.exp_pos, ← `(tactic| apply Real.exp_pos))]
  for (name, attempt) in helpers do
    let compatible ← withoutModifyingState <| withNewMCtxDepth do
      let rule ← mkConstWithFreshMVarLevels name
      let (_, _, conclusion) ← forallMetaTelescopeReducing (← inferType rule)
      unless conditionsMayMatch conclusion target (classifiers := true) do return false
      -- Match the helper's complete computation before unification can evaluate an unrelated
      -- scalar program. Candidate application still checks every selected dictionary.
      let indexed ← withCanUnfoldPred (fun _ _ => pure false) do
        let pattern ← numericalPattern conclusion
        ({} : DiscrTree Unit).insertIfSpecific pattern ()
      if ← withCanUnfoldPred (fun _ _ => pure false) do
          return !(← indexed.getMatch target).isEmpty then return true
      -- Ordinary value aliases may hide a matching literal. Keep the helper's heads and
      -- selected class methods opaque while checking that definitional equality.
      let protectedNames := conclusion.getUsedConstantsAsSet
      withCanUnfoldPred (fun config info => do
          if protectedNames.contains info.name ||
              (← getProjectionFnInfo? info.name).any (·.fromClass) then return false
          programCanUnfold {} false config info) <| withDefault do
        isDefEq conclusion target
    if compatible then attempts := attempts.push attempt
  if supplied then
    attempts := attempts.push (← `(tactic| solve_by_elim (maxDepth := 3)))
  -- Registered reflexivity applies to relations beyond equality.
  let reflexive ← withoutModifyingState <| withNewMCtxDepth do
    let target ← whnfR target
    if target.isAppOf ``Eq || target.isAppOf ``HEq then return true
    if target.getAppNumArgs < 2 then return false
    let relation := target.appFn!.appFn!
    unless relation.getAppFn.isConst do return true
    return !(← (Lean.Meta.Rfl.reflExt.getState (← getEnv)).getMatch relation).isEmpty
  if reflexive then attempts := attempts.push (← `(tactic| rfl))
  if target.isForall then attempts := attempts.push (← `(tactic| (intros; rfl)))
  if target.isAppOf ``AllSteps then
    attempts := attempts.push (← `(tactic| (apply AllSteps.of_forall; intros; trivial)))
  -- Match decide's own preprocessing: local let-values may close an otherwise symbolic type.
  let decidableTarget ← if target.hasFVar then zetaReduce target else pure target
  if !decidableTarget.hasFVar && !decidableTarget.hasMVar then
    attempts := attempts.push (← `(tactic| decide))
  for attempt in attempts do
    let solved? ← attempt? 1000 do
      let remaining ← Lean.Elab.Tactic.run goal (evalTactic attempt)
      return if remaining.isEmpty then some () else none
    if solved?.isSome then return true
  -- Range evidence can describe the same operation through a different certified provider.
  -- This is an equality proof, not an assumption that similarly named backends agree.
  let target ← instantiateMVars (← goal.getType)
  let boolean? ← attempt? 1000 do
    unless target.isAppOfArity ``Eq 3 do return none
    unless ← withReducible (isDefEq target.getAppArgs[0]! q(Bool)) do return none
    return some ()
  if boolean?.isSome then
    let context ← semanticContext (decodeClassifiers := true)
    for declaration in ← getLCtx do
      if declaration.isImplementationDetail then continue
      let solved? ← attempt? 1000 do
        let source ← instantiateMVars declaration.type
        unless source.isAppOfArity ``Eq 3 do return none
        unless source.getAppArgs[2]! == target.getAppArgs[2]! do return none
        unless ← withReducible (isDefEq source.getAppArgs[0]! q(Bool)) do return none
        -- Prove equivalence without rewriting the caller's hypothesis or the numerical goal.
        let equality ← mkFreshExprMVar (← mkEq source target)
        let (remaining, _) ← simpTarget equality.mvarId! context
        if let some remaining := remaining then
          remaining.withContext (withReducible remaining.refl)
        goal.assign (← mkAppM ``Eq.mp #[← instantiateMVars equality, declaration.toExpr])
        return some ()
      if solved?.isSome then return true
  if target.isAppOf ``AllSteps then
    for declaration in ← getLCtx do
      unless declaration.type.isAppOf ``AllSteps do continue
      let equality ← mkFreshExprMVar (← mkEq declaration.type target)
      if ← semanticEquality equality.mvarId! then
        goal.assign (← mkAppM ``Eq.mp #[equality, declaration.toExpr])
        return true
  if target.isAppOfArity ``LE.le 4 then
    let args := target.getAppArgs
    let equality ← mkFreshExprMVar (← mkEq args[2]! args[3]!)
    if ← semanticEquality equality.mvarId! then
      goal.assign (← mkAppM ``le_of_eq #[equality])
      return true
    let solved? ← attempt? 1000 do
      let (next, _) ← simpTarget goal (← semanticContext)
      let some next := next | return some ()
      trace[TorchLean.roundoff] "normalized numerical budget: {← next.getType}"
      let remaining ← Lean.Elab.Tactic.run next do
        evalTactic (← `(tactic| apply le_refl))
      return if remaining.isEmpty then some () else none
    if solved?.isSome then return true
  return false

/-- Range assumptions and executed code may use different tensor indexing notation.
Normalize their observations by proofs before retrying the same bounded side-condition search. -/
private def discharge (goal : MVarId) : TacticM Bool := do
  if ← dischargeCore goal then return true
  unless hasTensorObservation (← instantiateMVars (← goal.getType)) do return false
  let result? ← attempt? 10000 do
    let (context, procedures) ← coordinateSimpContext
    let (next, _) ← simpTarget goal context procedures
    let some next := next | return some ()
    trace[TorchLean.roundoff] "normalized side condition: {← next.getType}"
    return if ← dischargeCore next then some () else none
  return result?.isSome

/-- Compare the generated real reference with existing tensor code. All normalization steps
are proved equalities; the floating-point execution and its error budget remain unchanged. -/
private def referenceEquality (goal : MVarId) : TacticM Bool := do
  if ← discharge goal then return true
  let result? ← attempt? 10000 do
    let mut current := some goal
    for context in #[coordinateSimpContext, foldSimpContext, coordinateSimpContext] do
      if let some goal := current then
        let (context, procedures) ← context
        let (next, _) ← simpTarget goal context procedures
        current := next
    if let some goal := current then
      return if ← discharge goal then some () else none
    return some ()
  return result?.isSome

/-- Apply a proved bound with its original instances. Input-error premises determine the
reference before ordinary domain conditions are discharged or retained as obligations. -/
private def applyRule? (actual : Q(ℝ)) (rule : Lean.Expr) (exact : Bool)
    (deriveInput : Q(ℝ) → AnalyzeM Estimate) : AnalyzeM (Option (Estimate × Bool)) := do
  let initial ← get
  let boundaries := initial.boundaries
  let label := rule.getAppFn.constName?.getD .anonymous
  let result? ← transaction? do
    let reference : Q(ℝ) ← mkFreshExprMVar q(ℝ)
    let error : Q(ℝ) ← if exact then pure q(0) else mkFreshExprMVar q(ℝ)
    let target : Q(Prop) :=
      if exact then q($actual = $reference) else q(|$actual - $reference| ≤ $error)
    let proof ← mkFreshExprMVar target
    let rule ← normalizeRule (← specializeLocalError rule)
    let (arguments, binders, conclusion) ← forallMetaTelescopeReducing (← inferType rule)
    let some (candidate, _, _) ← numericalConclusion? conclusion | return none
    let matched? ← attempt? 10000 do
      let matched ← withCanUnfoldPred (fun config info =>
        if boundaries.contains info.name then pure false else canUnfoldDefault config info) do
          -- Match the computation before its dependent representation parameters.
          if let some source := actual.getAppArgs.back? then
            if let some pattern := candidate.getAppArgs.back? then
              unless ← isDefEq pattern source do return false
          isDefEq candidate actual
      unless matched do return none
      unless ← isDefEq conclusion target do return none
      return some ()
    unless matched?.isSome do return none
    proof.mvarId!.assign (mkAppN rule arguments)
    trace[TorchLean.roundoff] "matched {label}; checking {arguments.size} parameters"
    -- Detect input-reference parameters before solving any premise. Two inputs may share a
    -- reference, so assigning it for the first must not hide the second input's error premise.
    let needed ← getMVars (← instantiateMVars reference)
    let mut inputs := #[]
    for h : index in [:arguments.size] do
      let argument ← instantiateMVars arguments[index]
      unless argument.isMVar do continue
      let type ← instantiateMVars (← argument.mvarId!.getType)
      let some (_, wantedReference, _) ← errorConclusion? type | continue
      let mut feedsReference := false
      for dependency in ← getMVars wantedReference do
        if needed.contains dependency && !(← isProp (← dependency.getType)) then
          feedsReference := true
      if feedsReference then inputs := inputs.push index
    let propagatesInputs := !inputs.isEmpty
    let mut state := initial
    let mut allInputsAnalyzed := true
    for index in inputs do
      let argument ← instantiateMVars arguments[index]!
      let type ← instantiateMVars (← inferType argument)
      let some (value, _, _) ← errorConclusion? type | do
        allInputsAnalyzed := false
        continue
      let value : Q(ℝ) ← instantiateMVars value
      if value.hasExprMVar || value == actual then
        allInputsAnalyzed := false
        continue
      let resolved? ← transaction? do
        let (bound, next) ← (deriveInput value).run state
        let matched? ← attempt? 10000 do
          unless ← isDefEq type (← inferType bound.proof) do return none
          if argument.isMVar then argument.mvarId!.assign bound.proof
          return some ()
        return matched?.map fun _ => next
      if let some next := resolved? then
        state := next
      else
        allInputsAnalyzed := false
    -- An unchanged real reference needs no new domain assumptions. Keep the child validity
    -- obligations, including finiteness and successful numerical certificates. Even a shared,
    -- already assigned premise must have its executed input analyzed before this shortcut.
    let reference : Q(ℝ) ← instantiateMVars reference
    if propagatesInputs && allInputsAnalyzed && !reference.hasExprMVar && !actual.hasExprMVar then
      let identical? ← if reference == actual then pure (some ()) else
        attempt? 1000 do
          return if ← withReducible (isDefEq reference actual) then some () else none
      if identical?.isSome then
        let estimate : Estimate := ⟨actual, q(0), ← mkAppM ``exact_error #[actual]⟩
        return some (estimate, true, state)
    -- Capabilities still need their existing instance evidence; none is guessed or assumed.
    for h : index in [:arguments.size] do
      let argument ← instantiateMVars arguments[index]
      if argument.isMVar && binders[index]!.isInstImplicit then
        argument.mvarId!.assign (← synthInstance (← argument.mvarId!.getType))
    let mut pending := #[]
    for argument in arguments do
      let argument ← instantiateMVars argument
      unless argument.isMVar do continue
      let goal := argument.mvarId!
      unless ← isProp (← goal.getType) do continue
      unless ← discharge goal do pending := pending.push goal
    for argument in arguments do
      let argument ← instantiateMVars argument
      unless argument.isMVar do continue
      unless ← isProp (← argument.mvarId!.getType) do return none
    let reference : Q(ℝ) ← instantiateMVars reference
    let error : Q(ℝ) ← instantiateMVars error
    if reference.hasExprMVar || error.hasExprMVar then
      trace[TorchLean.roundoff] "unresolved reference or budget after {label}"
      return none
    let estimate := { reference, error, proof := ← instantiateMVars proof : Estimate }
    let estimate ← if exact then
        rewriteActual { estimate with proof := ← mkAppM ``exact_error #[reference] } estimate.proof
      else pure estimate
    return some (estimate, propagatesInputs, { state with pending := state.pending ++ pending })
  if let some (estimate, propagatesInputs, state) := result? then
    set state
    return some (estimate, propagatesInputs)
  return none

/-- Retrieve candidate laws for this computation before checking their full statements. -/
private def registeredBound? (actual : Q(ℝ)) (deriveInput : Q(ℝ) → AnalyzeM Estimate) :
    AnalyzeM (Option (Estimate × Bool)) := do
  let state ← get
  let candidates ← withCanUnfoldPred (programCanUnfold state.boundaries false)
    (state.rules.getMatch actual)
  for (name, exact) in candidates do
    trace[TorchLean.roundoff] "trying {name}"
    if let some bound ← applyRule? actual (← mkConstWithFreshMVarLevels name) exact deriveInput then
      return some bound
  return none

/-- Bound an actual unary result by its discrepancy from FloatLib's certified result. The
certificate's executable acceptance test remains a premise; its fallback value is never trusted. -/
private def certifiedBound? (actual : Q(ℝ)) : AnalyzeM (Option Estimate) := do
  let configured := actual.isAppOf ``FloatLib.Floats.ExecFloat.Binary.toReal
  unless configured ||
      actual.isAppOf ``FloatLib.Floats.Formats.BinaryInterchange.Model.toReal do return none
  let source := actual.getAppArgs.back!
  let name := source.getAppFn.constName?.getD .anonymous
  let some (certificateName, _) := certificates.find? (fun entry => entry.2.contains name)
    | return none
  let argument := source.getAppArgs.back!
  let modelOf (value : Lean.Expr) : MetaM Lean.Expr :=
    if configured then mkAppM ``FloatLib.Floats.ExecFloat.Binary.toModel #[value] else pure value
  let input ← modelOf argument
  let candidate ← modelOf source
  let modelType ← inferType candidate
  unless modelType.isAppOfArity ``FloatLib.Floats.Formats.BinaryInterchange.Model 1 do
    throwError "roundoff: the candidate has no binary descriptor model"
  let options : Q(Certified.Options) := q({})
  let certificate ← mkAppM certificateName #[modelType.getAppArgs[0]!, options]
  let certificateType ← inferType certificate
  unless certificateType.isAppOfArity ``Contract.CorrectlyRoundedCertificateOn 4 do
    throwError "roundoff: expected a FloatLib correct-rounding certificate"
  let domain := certificateType.getAppArgs[3]!
  let finite (value model : Lean.Expr) : MetaM Lean.Expr := do
    let predicate ← if configured then
        mkAppM ``FloatLib.Floats.ExecFloat.Binary.isFinite #[value]
      else mkAppM ``FloatLib.Floats.Formats.BinaryInterchange.Model.isFinite #[model]
    mkEq predicate (mkConst ``Bool.true)
  let inputFinite ← mkFreshExprMVar (← finite argument input)
  let outputFinite ← mkFreshExprMVar (← finite source candidate)
  let acceptance := (mkApp domain input).headBeta
  let success ← mkFreshExprMVar acceptance
  for premise in #[inputFinite, outputFinite, success] do
    unless ← discharge premise.mvarId! do
      modify fun state => { state with pending := state.pending.push premise.mvarId! }
  let rounding ← mkAppM ``Contract.CorrectlyRoundedCertificateOn.toReal_eq_roundAt
    #[certificate, input, inputFinite, success]
  let witness := mkApp certificateType.getAppArgs[2]! input
  -- Match a previously checked return value before building a discrepancy expression. This
  -- avoids asking simplification to evaluate a closed numerical kernel just to cancel that gap.
  let check := acceptance.getAppArgs[1]!.getAppArgs.back!
  let checkType ← inferType check
  let returned ← mkAppOptM ``Except.ok #[some checkType.getAppArgs[0]!, none, some candidate]
  let matchType ← mkEq check returned
  for declaration in ← getLCtx do
    unless declaration.type.isAppOfArity ``Eq 3 do continue
    unless declaration.type.getAppArgs[1]!.getAppFn == check.getAppFn do continue
    let matched? ← attempt? 2000 do
      return if ← isDefEq declaration.type matchType then some () else none
    if matched?.isNone then continue
    let extract ← withLocalDeclD `result checkType fun result => do
      let value ← mkAppM ``Option.getD
        #[← mkAppM ``Except.toOption #[result], witness.headBeta.getAppArgs.back!]
      mkLambdaFVars #[result] value
    let valueEq ← mkAppM ``congrArg #[extract, declaration.toExpr]
    let valueEq ← mkAppM ``Eq.trans #[valueEq, ← mkEqRefl candidate]
    let observed ← mkAppM ``FloatLib.Floats.Formats.BinaryInterchange.Model.toReal #[candidate]
    let observe := mkAppN observed.getAppFn observed.getAppArgs.pop
    let candidateRounding ← mkAppM ``Eq.trans
      #[← mkAppM ``Eq.symm #[← mkAppM ``congrArg #[observe, valueEq]], rounding]
    if !(← outputFinite.mvarId!.isAssigned) then
      let finiteValue ← mkAppM ``FloatLib.Floats.Formats.BinaryInterchange.Model.isFinite
        #[candidate]
      let finiteObserve := mkAppN finiteValue.getAppFn finiteValue.getAppArgs.pop
      let finiteEq ← mkAppM ``congrArg #[finiteObserve, valueEq]
      let checkedFinite ← mkAppM ``Contract.CorrectlyRoundedCertificateOn.output_finite
        #[certificate, input, inputFinite, success]
      outputFinite.mvarId!.assign
        (← mkAppM ``Eq.trans #[← mkAppM ``Eq.symm #[finiteEq], checkedFinite])
    let observedArgument := mkAppN actual.getAppFn (actual.getAppArgs.pop.push argument)
    let reference : Q(ℝ) := (mkApp certificateType.getAppArgs[1]! observedArgument).headBeta
    let error : Q(ℝ) ← mkAppM
      ``FloatLib.Floats.Formats.BinaryInterchange.Model.epsilonAt
      #[modelType.getAppArgs[0]!, reference]
    let proof ← mkAppM ``FloatLib.Floats.Formats.BinaryInterchange.Model.abs_roundAt_sub_le
      #[modelType.getAppArgs[0]!, reference]
    return some (← rewriteActual ⟨reference, error, proof⟩ candidateRounding)
  let proof ← mkAppM ``certified_error #[candidate, witness, outputFinite, rounding]
  let type : Q(Prop) ← instantiateMVars (← inferType proof)
  match type with
  | ~q(|($_actual : ℝ) - $_reference| ≤ $error) =>
      let observedArgument := mkAppN actual.getAppFn (actual.getAppArgs.pop.push argument)
      let reference := (mkApp certificateType.getAppArgs[1]! observedArgument).headBeta
      return some ⟨reference, error, proof⟩
  | _ => throwError "roundoff: unexpected certified error theorem {type}"

/-- Existing FloatLib soundness theorems supply real enclosures independently of the selected
implementation's accuracy. -/
private def elementaryEnclosures : Array (Name × Array Name) := #[
  (``FloatLib.Numerics.Interval.containsReal_sinBounds,
    #[``MathFunctions.sin, ``ExecFloat.Binary.sin, ``Model.sin]),
  (``FloatLib.Numerics.Interval.containsReal_cosBounds,
    #[``MathFunctions.cos, ``ExecFloat.Binary.cos, ``Model.cos]),
  (``FloatLib.Numerics.Interval.containsReal_sinhBounds,
    #[``MathFunctions.sinh, ``ExecFloat.Binary.sinh, ``Model.sinh]),
  (``FloatLib.Numerics.Interval.containsReal_coshBounds,
    #[``MathFunctions.cosh, ``ExecFloat.Binary.cosh, ``Model.cosh]),
  (``FloatLib.Numerics.Interval.containsReal_tanhBounds,
    #[``MathFunctions.tanh, ``ExecFloat.Binary.tanh, ``Model.tanh])]

/-- Compare an actual primitive result with its certified rational enclosure. The shared
composition pass propagates input error afterward; endpoints stay symbolic during construction. -/
private def intervalBound? (actual : Q(ℝ)) : AnalyzeM (Option Estimate) := do
  let configured := actual.isAppOf ``ExecFloat.Binary.toReal
  unless configured || actual.isAppOf ``Model.toReal do return none
  let source := actual.getAppArgs.back!
  let name := source.getAppFn.constName?.getD .anonymous
  let some (soundnessName, _) := elementaryEnclosures.find? (fun entry => entry.2.contains name)
    | return none
  let argument := source.getAppArgs.back!
  let modelOf (value : Lean.Expr) : MetaM Lean.Expr :=
    if configured then mkAppM ``ExecFloat.Binary.toModel #[value] else pure value
  let input ← modelOf argument
  let candidate ← modelOf source
  let modelType ← inferType candidate
  unless modelType.isAppOfArity ``Model 1 do
    throwError "roundoff: the candidate has no binary descriptor model"
  let precision ← mkAppM ``FloatFormat.fracWidth #[modelType.getAppArgs[0]!]
  let terms ← mkAppM ``max #[mkNatLit 8, precision]
  let rational ← mkAppM ``Option.getD #[← mkAppM ``Model.toRat? #[input], q((0 : Rat))]
  let point ← mkAppM ``FloatLib.Numerics.Interval.point #[rational]
  let contained ← mkAppM soundnessName #[point, terms, ← mkAppM ``contains_decoded_point #[input]]
  let finite (value model : Lean.Expr) : MetaM Lean.Expr := do
    let predicate ← if configured then mkAppM ``ExecFloat.Binary.isFinite #[value]
      else mkAppM ``Model.isFinite #[model]
    mkEq predicate (mkConst ``Bool.true)
  let inputFinite ← mkFreshExprMVar (← finite argument input)
  let outputFinite ← mkFreshExprMVar (← finite source candidate)
  for premise in #[inputFinite, outputFinite] do
    unless ← discharge premise.mvarId! do
      modify fun state => { state with pending := state.pending.push premise.mvarId! }
  let proof ← mkAppM ``model_enclosure_error
    #[input, candidate, inputFinite, outputFinite, contained]
  let type : Q(Prop) ← instantiateMVars (← inferType proof)
  match type with
  | ~q(|($observed : ℝ) - $reference| ≤ $error) =>
      unless ← isDefEq observed actual do
        throwError "roundoff: enclosure result does not observe the requested computation"
      return some ⟨reference, error, proof⟩
  | _ => throwError "roundoff: unexpected interval error theorem {type}"

/-- Reuse scalar or pointwise bounds, discharging their conditional premises in the current
branch or leaving the required evidence visible to the caller. -/
private def givenBound? (actual : Q(ℝ)) (deriveInput : Q(ℝ) → AnalyzeM Estimate) :
    AnalyzeM (Option Estimate) := do
  for decl in ← getLCtx do
    if decl.isImplementationDetail then continue
    unless ← isProp decl.type do continue
    let compatible ← withNewMCtxDepth do
      let (_, _, type) ← forallMetaTelescope (← instantiateMVars decl.type)
      let type : Q(Prop) ← pure type
      match type with
      | ~q(|($value : ℝ) - $_reference| ≤ $_error) =>
          pure (actual.getAppFn.constName? == value.getAppFn.constName? &&
            actual.getAppArgs.back?.map (·.getAppFn.constName?) ==
              value.getAppArgs.back?.map (·.getAppFn.constName?))
      | _ => pure false
    unless compatible do continue
    if let some (bound, _) ← applyRule? actual decl.toExpr false deriveInput then return some bound
  return none

/-- Share repeated premises and bind the remaining ones as an explicit condition.
Branches, loop invariants and generated statements use the same condition handling. -/
private def withConditions {α : Type} (candidates : Array MVarId)
    (action : Q(Prop) → Lean.Expr → TacticM α) : TacticM α := do
  let mut goals := #[]
  let mut known : Lean.ExprMap MVarId := {}
  let mut indexed : Array ((Name × Name) × Lean.Expr × MVarId) := #[]
  for goal in candidates do
    if ← goal.isAssigned then continue
    let type ← instantiateMVars (← goal.getType)
    unless ← isProp type do throwError "roundoff: unresolved data parameter {type}"
    if let some earlier := known.get? type then
      goal.assign (mkMVar earlier)
    else
      -- Restrict definitional comparison to matching predicates and source heads, avoiding
      -- expansion of unrelated floating-point implementations during deduplication.
      let body := type.getForallBody
      let predicate := if body.isAppOfArity ``Eq 3 then body.getAppArgs[1]! else body
      let key := (predicate.getAppFn.constName?.getD .anonymous,
        predicate.getAppArgs.back?.bind (·.getAppFn.constName?) |>.getD .anonymous)
      let mut duplicate := false
      for (earlierKey, earlierType, earlier) in indexed do
        unless key == earlierKey do continue
        let matched? ← attempt? 5000 do
          return if ← isDefEq type earlierType then some () else none
        if matched?.isSome then
          goal.assign (mkMVar earlier)
          duplicate := true
          break
      unless duplicate do
        goals := goals.push goal
        known := known.insert type goal
        indexed := indexed.push (key, type, goal)
  let mut condition : Q(Prop) := q(True)
  for goal in goals.reverse do
    let type : Q(Prop) ← instantiateMVars (← goal.getType)
    condition := q($type ∧ $condition)
  let mut simplifications : SimpTheorems := {}
  for name in #[``and_true, ``true_and] do
    simplifications ← simplifications.addConst name
  let context ← Simp.mkContext { implicitDefEqProofs := false } #[simplifications]
  let (simplified, _) ← simp condition context
  let simplifiedCondition : Q(Prop) := simplified.expr
  withLocalDeclD `conditions simplifiedCondition fun evidence => do
    let mut remaining ← mkAppM ``Eq.mpr #[← simplified.getProof' condition, evidence]
    for goal in goals do
      goal.assign (← mkAppM ``And.left #[remaining])
      remaining ← mkAppM ``And.right #[remaining]
    action simplifiedCondition evidence

/-- Keep branch-specific side conditions under their guard. Neither a cached proof nor a
finiteness obligation from one branch may escape into the other. -/
private def inBranch (condition : Q(Prop)) (action : Lean.Expr → AnalyzeM Estimate) :
    AnalyzeM (Lean.Expr × Lean.Expr × Lean.Expr) := do
  let initial ← get
  let outerContext ← getLCtx
  let outerInstances ← getLocalInstances
  let (estimate, pending) ← withLocalDeclD `branch condition fun guard => do
    let estimate ← action guard
    let localGoals := (← get).pending.extract initial.pending.size (← get).pending.size
    let (numerical, pending) ← if localGoals.isEmpty then
        pure (← instantiateMVars estimate.proof, #[])
      else withConditions localGoals fun condition evidence => do
        let type ← mkForallFVars #[guard] condition
        let closed ← withLCtx outerContext outerInstances (mkFreshExprMVar type)
        let conditional ← mkLambdaFVars #[evidence] (← instantiateMVars estimate.proof)
        pure (mkApp conditional (mkApp closed guard), #[closed.mvarId!])
    let reference ← mkLambdaFVars #[guard] (← instantiateMVars estimate.reference)
    let error ← mkLambdaFVars #[guard] (← instantiateMVars estimate.error)
    let proof ← mkLambdaFVars #[guard] numerical
    pure ((reference, error, proof), pending)
  set { initial with pending := initial.pending ++ pending }
  return estimate

/-- Analyze a scalar step once under an incoming accumulator bound. Its new premises become
one condition on visited states; no local variables or cached proofs escape the loop. -/
private def deriveStep (observe step inputType stateType : Lean.Expr)
    (analyzeBody : Q(ℝ) → AnalyzeM Estimate) : AnalyzeM StepEstimate := do
  let before ← get
  let result ← withLocalDeclD `accumulator stateType fun accumulator =>
    withLocalDeclD `reference q(ℝ) fun reference =>
    withLocalDeclD `budget q(ℝ) fun budget =>
    withLocalDeclD `value inputType fun value => do
      let observed : Q(ℝ) := (mkApp observe accumulator).headBeta
      let reference : Q(ℝ) := reference
      let budget : Q(ℝ) := budget
      withLocalDeclD `accumulator_error q(|$observed - $reference| ≤ $budget) fun incoming => do
        let actual : Q(ℝ) := (mkApp observe (mkAppN step #[accumulator, value]).headBeta).headBeta
        let estimate ← analyzeBody actual
        let candidates := (← get).pending.extract before.pending.size (← get).pending.size
        withConditions candidates fun condition evidence => do
          if condition.containsFVar incoming.fvarId! then
            throwError "roundoff: a loop condition depends on its proof argument"
          if estimate.reference.containsFVar accumulator.fvarId! ||
              estimate.reference.containsFVar budget.fvarId! then
            throwError "roundoff: the loop's real reference depends on its rounded accumulator; \
              establish agreement of its branch conditions first"
          let referenceFunction ← mkLambdaFVars #[reference, value] estimate.reference
          let errorFunction ← mkLambdaFVars #[accumulator, reference, budget, value] estimate.error
          let conditionFunction ← mkLambdaFVars #[accumulator, reference, budget, value] condition
          let actualCondition? ← if condition.containsFVar reference.fvarId! ||
              condition.containsFVar budget.fvarId! then pure none
            else pure (some (← mkLambdaFVars #[accumulator, value] condition))
          let proof ← mkLambdaFVars #[accumulator, reference, budget, value, evidence, incoming]
            (← instantiateMVars estimate.proof)
          pure ⟨referenceFunction, errorFunction, conditionFunction, proof, actualCondition?⟩
  set before
  return result

/-- Metadata on an application spine carries elaboration information, not computation.
Keep every argument, including the selected dictionaries, while exposing the actual head. -/
private def stripHeadMetadata : Lean.Expr → Lean.Expr
  | .mdata _ expression => stripHeadMetadata expression
  | expression@(.app function argument) =>
      expression.updateApp! (stripHeadMetadata function) argument
  | expression => expression

/-- Apply one existing tensor-read equality after ordinary coordinate simplification stalls.
Only this fully typed observation uses default transparency and local-let expansion; the selected
storage, original producer, and scalar callback remain in its checked proof. -/
private def readObservation? (name : Name) (expression : Expr) :
    MetaM (Option (Expr × Expr)) := do
  let expression ← instantiateMVars expression
  if expression.hasExprMVar then return none
  withoutModifyingState <| withNewMCtxDepth do
    try
      let rule ← mkConstWithFreshMVarLevels name
      let (arguments, _, conclusion) ← forallMetaTelescopeReducing (← inferType rule)
      let some (_, left, right) := conclusion.eq? | return none
      unless ← withConfig (fun config =>
          { config with transparency := .default, zetaDelta := true })
          (isDefEq left expression) do return none
      let result ← instantiateMVars right
      let proof ← instantiateMVars (mkAppN rule arguments)
      if result.hasExprMVar || proof.hasExprMVar || proof.hasLevelMVar then return none
      let expected ← mkEq expression result
      let proof := mkApp2 (mkConst ``id [0]) expected proof
      check proof
      return some (result, proof)
    catch error =>
      if error.isInterrupt || error.isRuntime then throw error
      return none

/-- The analysis is over real interpretations, independent of scalar representation.
Primitive rules expose where rounding occurs; ordinary real arithmetic just propagates errors.
-/
private partial def analyze (actual : Q(ℝ)) (fuel : Nat := 256) :
    AnalyzeM Estimate := do
  let actual : Q(ℝ) ← instantiateMVars actual
  let actual : Q(ℝ) := stripHeadMetadata actual
  if actual.headBeta != actual then return ← analyze actual.headBeta fuel
  trace[TorchLean.roundoff] "analyze {actual.getAppFn.constName?.getD .anonymous}"
  if fuel == 0 then throwError "roundoff: source expansion exceeded its traversal limit"
  if let some estimate := (← get).cache.get? actual then return estimate
  let estimate ← build actual (fuel - 1)
  modify fun state => { state with cache := state.cache.insert actual estimate }
  return estimate
where
  build (actual : Q(ℝ)) (fuel : Nat) : AnalyzeM Estimate := do
    if let some bound ← givenBound? actual (analyze · fuel) then return bound
    let name := actual.getAppFn.constName?.getD .anonymous
    let selected? ← registeredBound? actual (analyze · fuel)
    let selected? ← if selected?.isSome then pure selected? else
      pure ((← certifiedBound? actual).map fun bound => (bound, false))
    let selected? ← if selected?.isSome then pure selected? else
      pure ((← intervalBound? actual).map fun bound => (bound, false))
    if let some (estimate, propagatesInputs) := selected? then
      if propagatesInputs || estimate.reference == actual then return estimate
      let inputEstimate ← analyze estimate.reference fuel
      return ⟨inputEstimate.reference, q($(inputEstimate.error) + $(estimate.error)),
        ← mkAppM ``compose_error #[estimate.proof, inputEstimate.proof]⟩
    -- An observation of an input or a literal is exact as a real value. The rounding when that
    -- literal was encoded is not silently identified with its unrounded source spelling.
    let observed ← if (← get).observations.contains name then
        isScalarObservation actual
      else pure false
    let source ← instantiateMVars (if observed then actual.getAppArgs.back! else actual)
    let source := stripHeadMetadata source
    if source.headBeta != source then
      let next : Q(ℝ) := if observed then
          mkAppN actual.getAppFn (actual.getAppArgs.pop.push source.headBeta)
        else source.headBeta
      return ← analyze next fuel
    if source.isAppOfArity ``Option.get 3 then
      let (context, procedures) ← coordinateSimpContext
      let (normalized, _) ← simp actual context procedures
      if normalized.expr != actual then
        let estimate ← analyze normalized.expr fuel
        return ← rewriteActual estimate (← normalized.getProof' actual)
    let optionalValue? ← optionGet? source
    if source.isAppOfArity ``Option.get 3 && optionalValue?.isNone then
      throwError "roundoff: no numerical rule for optional producer {source.getAppArgs[1]!}; \
        expose its computation or supply an explicit bound"
    if let some (value, equality) := optionalValue? then
      let observe := if observed then mkAppN actual.getAppFn actual.getAppArgs.pop
        else q(fun value : ℝ => value)
      let equality ← mkAppM ``congrArg #[observe, equality]
      let next : Q(ℝ) := (mkApp observe value).headBeta
      let estimate ← analyze next fuel
      return ← rewriteActual estimate equality
    if let some reduced ← reduceControl? (← get).boundaries source then
      unless reduced == source do
        let next : Q(ℝ) := if observed then
            mkAppN actual.getAppFn (actual.getAppArgs.pop.push reduced)
          else reduced
        return ← analyze next fuel
    if let some (_, equality) ← optionMatch? source then
      let observe := if observed then mkAppN actual.getAppFn actual.getAppArgs.pop
        else q(fun value : ℝ => value)
      let equality ← mkAppM ``congrArg #[observe, equality]
      let some (_, _, next) := (← inferType equality).eq? |
        throwError "roundoff: Option observation did not produce an equality"
      let estimate ← analyze next fuel
      return ← rewriteActual estimate equality
    if source.isAppOf ``cond ||
        ((source.isAppOf ``ite || source.isAppOf ``dite) && source.getAppNumArgs > 5) then
      let mut branches : SimpTheorems := {}
      for name in #[``Bool.cond_eq_ite, ``ite_apply, ``dite_apply] do
        branches ← branches.addConst name
      let context ← Simp.mkContext { implicitDefEqProofs := false } #[branches]
      let (normalized, _) ← simp actual context
      if normalized.expr != actual then
        let estimate ← analyze normalized.expr fuel
        return ← rewriteActual estimate (← normalized.getProof' actual)
    if name == ``FloatLib.Floats.Formats.BinaryInterchange.Model.toReal &&
        source.isAppOf ``FloatLib.Floats.ExecFloat.Binary.toModel then
      return ← analyze (← mkAppM ``FloatLib.Floats.ExecFloat.Binary.toReal
        #[source.getAppArgs.back!]) fuel
    if source.isAppOf ``TorchLean.Tensor.sumSpec ||
        source.isAppOf ``TorchLean.Tensor.foldlSpec ||
        source.isAppOf ``TorchLean.Tensor.Internal.Rep.foldl ||
        source.isAppOf ``Fin.foldl || source.isAppOf ``List.foldl then
      let (context, procedures) ← foldSimpContext
      let (normalized, _) ← simp actual context procedures
      if normalized.expr != actual then
        let estimate ← analyze normalized.expr fuel
        return ← rewriteActual estimate (← normalized.getProof' actual)
      if source.isAppOfArity ``TorchLean.Tensor.Internal.Rep.foldl 7 then
        let args := source.getAppArgs
        let observe := if observed then mkAppN actual.getAppFn actual.getAppArgs.pop
          else q(fun value : ℝ => value)
        let step := args[4]!
        let initial := args[5]!
        let input := args[6]!
        -- A computed tensor must expose its producer at each coordinate. Otherwise its
        -- already-rounded entries would silently become the inputs of the real reference.
        unless input.isAppOf ``TorchLean.Tensor.Internal.Rep.coordinates ||
            (← isInputValue input) do
          let traversal ← mkAppM ``TorchLean.Tensor.Internal.Rep.foldl_eq_coordinates
            #[step, initial, input]
          let equality ← mkAppM ``congrArg #[observe, traversal]
          let some (_, _, next) := (← inferType equality).eq? |
            throwError "roundoff: coordinate traversal did not produce an observation equality"
          let estimate ← analyze next fuel
          return ← rewriteActual estimate equality
        let initialValue : Q(ℝ) := (mkApp observe initial).headBeta
        let initialEstimate ← analyze initialValue fuel
        let body ← deriveStep observe step args[0]! args[1]! (analyze · fuel)
        let augmented ← mkAppM ``foldStep #[step, body.reference, body.error]
        let initialState ← mkAppM ``Prod.mk #[initial,
          ← mkAppM ``Prod.mk #[initialEstimate.reference, initialEstimate.error]]
        let stateType ← inferType initialState
        let condition ← withLocalDeclD `state stateType fun state =>
          withLocalDeclD `value args[0]! fun value => do
            let current := Lean.Expr.proj ``Prod 0 state
            let pair := Lean.Expr.proj ``Prod 1 state
            mkLambdaFVars #[state, value] (mkAppN body.condition
              #[current, .proj ``Prod 0 pair, .proj ``Prod 1 pair, value])
        let validType ← mkAppM ``AllSteps #[augmented, condition, initialState, input]
        let valid ← mkFreshExprMVar validType
        if let some actualCondition := body.actualCondition? then
          let project ← withLocalDeclD `state stateType fun state =>
            mkLambdaFVars #[state] (.proj ``Prod 0 state)
          let goals ← valid.mvarId!.apply (← mkAppM ``AllSteps.of_map
            #[augmented, condition, initialState, input, project, step, actualCondition])
          for goal in goals do
            unless ← discharge goal do
              modify fun state => { state with pending := state.pending.push goal }
        else
          modify fun state => { state with pending := state.pending.push valid.mvarId! }
        let proof ← mkAppM ``fold_error #[observe, step, body.reference, body.error, body.condition,
          initial, initialEstimate.reference, initialEstimate.error, input, initialEstimate.proof,
          body.proof, valid]
        let type : Q(Prop) ← instantiateMVars (← inferType proof)
        return ← match type with
        | ~q(|($_actual : ℝ) - $reference| ≤ $error) => pure ⟨reference, error, proof⟩
        | _ => throwError "roundoff: unexpected loop proof type {type}"
    if source.isAppOfArity ``ite 5 || source.isAppOfArity ``dite 5 then
      let args := source.getAppArgs
      let condition : Q(Prop) := args[1]!
      let observe ← if observed then
          pure (mkAppN actual.getAppFn actual.getAppArgs.pop)
        else pure q(fun value : ℝ => value)
      -- A guard already proved in this scope makes the other branch unreachable. Keep the
      -- source's decision procedure and, for dependent branches, its actual proof argument.
      for declaration in ← getLCtx do
        let type ← instantiateMVars declaration.type
        let positive := stripHeadMetadata type == stripHeadMetadata condition
        let negative := stripHeadMetadata type == q(¬$condition)
        unless positive || negative do continue
        let rule := if source.isAppOfArity ``dite 5 then
            if positive then ``dite_eq_left else ``dite_eq_right
          else if positive then ``ite_eq_left else ``ite_eq_right
        let equality ← mkAppOptM rule #[some args[1]!, some args[2]!,
          some declaration.toExpr, some args[0]!, some args[3]!, some args[4]!]
        check equality .default
        let equality ← mkAppM ``congrArg #[observe, equality]
        let some (_, _, next) := (← inferType equality).eq? |
          throwError "roundoff: conditional reduction did not produce an equality"
        let estimate ← analyze next fuel
        return ← rewriteActual estimate equality
      let branch (guard value : Lean.Expr) : MetaM Lean.Expr := do
        if source.isAppOfArity ``dite 5 then return value
        withLocalDeclD `branch guard fun evidence => mkLambdaFVars #[evidence] value
      let yes ← branch condition args[3]!
      let no ← branch q(¬$condition) args[4]!
      let (yesReference, yesError, yesProof) ← inBranch condition fun guard =>
        analyze (mkApp observe (mkApp yes guard).headBeta).headBeta fuel
      let (noReference, noError, noProof) ← inBranch q(¬$condition) fun guard =>
        analyze (mkApp observe (mkApp no guard).headBeta).headBeta fuel
      let proof ← mkAppOptM ``dite_error #[none, some observe, some condition, some args[2]!,
        some yes, some no, some yesReference, some yesError, some noReference, some noError,
        some yesProof, some noProof]
      let type : Q(Prop) ← instantiateMVars (← inferType proof)
      return ← match type with
      | ~q(|($_actual : ℝ) - $reference| ≤ $error) => pure ⟨reference, error, proof⟩
      | _ => throwError "roundoff: unexpected branch proof type {type}"
    if let .letE _ _ value body _ := source then
      let source := body.instantiate1 value
      let next : Q(ℝ) := if observed then
          mkAppN actual.getAppFn (actual.getAppArgs.pop.push source)
        else source
      return ← analyze next fuel
    if source.isFVar then
      if let some value := (← source.fvarId!.getDecl).value? then
        -- Canonical real literals need no source expansion. Floating-point conversions are
        -- analyzed below, including when their result has a local name.
        if ← literalInput? actual value then
          return ⟨actual, q(0), ← mkAppM ``exact_error #[actual]⟩
        let next : Q(ℝ) := if observed then
            mkAppN actual.getAppFn (actual.getAppArgs.pop.push value)
          else value
        return ← analyze next fuel
      return ⟨actual, q(0), ← mkAppM ``exact_error #[actual]⟩
    let sourceName := source.getAppFn.constName?.getD .anonymous
    -- A generated callback may use the flat getter. Recover its coordinate observation before
    -- generic source expansion can expose the selected storage's raw read method.
    if source.isAppOfArity' ``TorchLean.Tensor.Internal.Rep.getFlat 5 then
      let some (value, equality) ← readObservation?
          ``TorchLean.Tensor.Internal.Rep.getFlat_eq_get source |
        throwError "roundoff: could not recover the original tensor observation in {source}"
      let observe := if observed then mkAppN actual.getAppFn actual.getAppArgs.pop
        else q(fun value : ℝ => value)
      let equality ← mkAppM ``congrArg #[observe, equality]
      let next : Q(ℝ) := (mkApp observe value).headBeta
      let estimate ← analyze next fuel
      return ← rewriteActual estimate equality
    if sourceName == ``TorchLean.Tensor.getScalar ||
        sourceName == ``TorchLean.Tensor.Internal.Rep.get then
      let tensor := source.getAppArgs[source.getAppArgs.size - 2]!
      let tensorType ← withDefault (whnf (← inferType tensor))
      -- Normalize the receiver's type so ordinary aliases are accepted without evaluating
      -- its stored values. General function coercions follow source analysis below.
      if tensorType.isAppOf ``TorchLean.Tensor.Internal.Rep then
        let (context, procedures) ← coordinateSimpContext
        let (normalized, _) ← simp actual context procedures
        if normalized.expr != actual then
          let estimate ← analyze normalized.expr fuel
          return ← rewriteActual estimate (← normalized.getProof' actual)
        if ← isInputValue tensor then
          return ⟨actual, q(0), ← mkAppM ``exact_error #[actual]⟩
        -- Expose the producer while retaining its tensor read. A stalled producer must not
        -- unfold into storage access and hide the computation that created its entries.
        if let some producer ← unfoldDefinition? tensor then
          unless producer == tensor do
            let arguments := source.getAppArgs.set! (source.getAppArgs.size - 2) producer
            let source := mkAppN source.getAppFn arguments
            let next : Q(ℝ) := if observed then
                mkAppN actual.getAppFn (actual.getAppArgs.pop.push source)
              else source
            return ← analyze next fuel
        if let some (value, equality) ← readObservation?
            ``TorchLean.Tensor.Internal.Rep.get_ofFlatFn source then
          let observe := if observed then mkAppN actual.getAppFn actual.getAppArgs.pop
            else q(fun value : ℝ => value)
          let equality ← mkAppM ``congrArg #[observe, equality]
          let next : Q(ℝ) := (mkApp observe value).headBeta
          let estimate ← analyze next fuel
          return ← rewriteActual estimate equality
        throwError "roundoff: no numerical rule for tensor producer {tensor}; expose its \
          computation or supply an explicit bound"
    if let .proj .. := source.getAppFn then
      if ← isInputValue source then
        return ⟨actual, q(0), ← mkAppM ``exact_error #[actual]⟩
      -- Helpers returning a record or tuple retain their original control flow. Reuse the
      -- proved source normalizer when that helper appears after earlier scalar unfolding.
      let (context, procedures) ← coordinateSimpContext
      let (normalized, _) ← simp actual context procedures
      if normalized.expr != actual then
        let estimate ← analyze normalized.expr fuel
        return ← rewriteActual estimate (← normalized.getProof' actual)
      if let some implementation ← withReducibleAndInstances (reduceProj? source.getAppFn) then
        let source := (mkAppN implementation source.getAppArgs).headBeta
        let next : Q(ℝ) := if observed then
            mkAppN actual.getAppFn (actual.getAppArgs.pop.push source)
          else source
        return ← analyze next fuel
    -- Casts may observe a complete loop or tensor computation, handled above. Otherwise only
    -- canonical real literals and exact discrete-to-real casts remain data.
    if ← exactRealData? actual then
      return ⟨actual, q(0), ← mkAppM ``exact_error #[actual]⟩
    if ← literalInput? actual source then
      return ⟨actual, q(0), ← mkAppM ``exact_error #[actual]⟩
    if name == ``FloatLib.Floats.ExecFloat.Binary.toReal ||
        name == ``Rat.cast || name == ``RatCast.ratCast then
      let mut decoding : SimpTheorems := {}
      decoding ← decoding.addDeclToUnfold ``FloatLib.Floats.ExecFloat.Binary.toReal
      decoding ← decoding.addDeclToUnfold ``FloatLib.Floats.ExecFloat.Binary.toRat?
      for field in (getStructureInfo (← getEnv) ``MathFunctions).fieldInfo do
        decoding ← decoding.addDeclToUnfold field.projFn
      for declaration in #[``decode_unary, ``decode_binary,
          ``FloatLib.Floats.ExecFloat.Binary.toModel_ofModel,
          ``decode_family,
          ``FloatLib.Floats.ExecFloat.Binary.toModel_neg,
          ``FloatLib.Floats.ExecFloat.Binary.toModel_abs,
          ``FloatLib.Floats.ExecFloat.Binary.toModel_min,
          ``FloatLib.Floats.ExecFloat.Binary.toModel_max,
          ``FloatLib.Floats.ExecFloat.Binary.toModel_exp,
          ``FloatLib.Floats.ExecFloat.Binary.toModel_log] do
        decoding ← decoding.addConst declaration
      let (prepared, procedures) ← addProgramRewrites decoding
      let context ← Simp.mkContext { implicitDefEqProofs := false } #[prepared]
      let (normalized, _) ← simp actual context procedures
      trace[TorchLean.roundoff] "decoded observation: {normalized.expr}"
      -- Merely unfolding the observer makes no progress if the operation was not decoded.
      let decoded := normalized.expr.getAppArgs.back?.all fun value =>
        !value.isAppOf ``FloatLib.Floats.ExecFloat.Binary.toModel
      if decoded && normalized.expr != actual then
        let estimate ← analyze normalized.expr fuel
        return ← rewriteActual estimate (← normalized.getProof' actual)
    if [``HAdd.hAdd, ``HSub.hSub, ``HMul.hMul, ``HDiv.hDiv, ``Neg.neg].contains sourceName then
      throwError "roundoff: no proved arithmetic rule matches the selected implementation in \
        `{sourceName}`"
    -- A scalar method can select an implementation through several nested dictionaries.
    -- Expose that selected field, including FloatLib interfaces, before interpreting its code.
    if (← getProjectionFnInfo? sourceName).isSome then
      let implementation? ← unfoldProjInst? source
      let implementation? ← if implementation?.isSome then pure implementation?
        else unfoldDefinition? source
      if let some implementation := implementation? then
        unless implementation == source do
          let next : Q(ℝ) := if observed then
              mkAppN actual.getAppFn (actual.getAppArgs.pop.push implementation)
            else implementation
          return ← analyze next fuel
    -- Expanding a user definition preserves its exact arithmetic. Unsupported library primitives
    -- are rejected before exposing their implementation or dispatch dictionaries.
    unless (`FloatLib).isPrefixOf sourceName do
      if let some unfolded ← unfoldDefinition? source then
        unless unfolded == source do
          let next : Q(ℝ) := if observed then
              mkAppN actual.getAppFn (actual.getAppArgs.pop.push unfolded)
            else unfolded
          return ← analyze next fuel
    throwError "roundoff: no numerical rule for {source}; expose its arithmetic or supply an \
      explicit bound in the context"

/-- Exact decoding uses FloatLib's existing operation equalities. A higher-order fold rule
transports arbitrary scalar steps, so reductions need no per-layer decoding specifications. -/
private def decode : TacticM Unit := do
  evalTactic (← `(tactic| simp (config := {
        implicitDefEqProofs := false, failIfUnchanged := false }) only
          [TorchLean.Tensor.sumSpec, TorchLean.Tensor.foldlSpec, TorchLean.Tensor.foldl,
          TorchLean.Tensor.map, decode_foldl, TorchLean.Tensor.Internal.Rep.foldl_map,
          MathFunctions.exp, MathFunctions.log, MathFunctions.abs,
          FloatLib.Floats.ExecFloat.Binary.toRat?,
          FloatLib.Floats.ExecFloat.Binary.toModel_hAdd,
          FloatLib.Floats.ExecFloat.Binary.toModel_hSub,
          FloatLib.Floats.ExecFloat.Binary.toModel_hMul,
          FloatLib.Floats.ExecFloat.Binary.toModel_hDiv,
          FloatLib.Floats.ExecFloat.Binary.toModel_neg,
          FloatLib.Floats.ExecFloat.Binary.toModel_abs,
          FloatLib.Floats.ExecFloat.Binary.toModel_exp,
          FloatLib.Floats.ExecFloat.Binary.toModel_log,
          FloatLib.Floats.ExecFloat.Binary.toModel_pow,
          FloatLib.Floats.ExecFloat.Binary.toModel_ofModel]))

/-- Decode an implementation, derive its error bound, or preserve a nonzero/positive result.

Each primitive rule fixes its real interpretation, and arithmetic propagates the resulting errors.
Certified functions retain the difference between the candidate and the checked reference result.
Nonzero and positivity goals additionally require an error below the reference magnitude or positive
value. Unproved finiteness, domain and budget conditions remain visible goals.
Missing operation rules stop the error analysis.
For an exact decoding equality, `roundoff` uses FloatLib's operation equalities and the generic
tensor-fold transport rule. A concrete model computation or an unsupported operation may remain
as a proof goal after decoding.
-/
elab "roundoff" : tactic => do
  -- Keep named inputs and their local assumptions available to numerical rule matching.
  while (← getMainTarget).isForall || (← getMainTarget).isLet do
    let (_, goal) ← (← getMainGoal).intro1
    replaceMainGoal [goal]
  let target ← getMainTarget
  if !target.isAppOf ``Eq && (target.find? fun term =>
      #[``TorchLean.Tensor.sumSpec, ``TorchLean.Tensor.dotSpec,
        ``TorchLean.Tensor.foldlSpec, ``TorchLean.Tensor.foldl,
        ``TorchLean.Tensor.Internal.Rep.foldl, ``Fin.foldl, ``List.foldl,
        ``AllSteps, ``foldError].contains
          (term.getAppFn.constName?.getD .anonymous)).isSome then
    withMainContext do
      let (context, procedures) ← foldSimpContext
      discard <| simpLocation context procedures none .wildcard
  if (← getGoals).isEmpty then return
  trace[TorchLean.roundoff] "prepare tensor coordinates"
  -- Coordinate rules expose the existing tensor arithmetic, without flattening tensors or
  -- unfolding their storage. The same equalities normalize the finiteness assumptions.
  let target ← getMainTarget
  if hasTensorObservation target then
    withMainContext do
      let (context, procedures) ← coordinateSimpContext
      discard <| simpLocation context procedures none .wildcard
  if (← getGoals).isEmpty then return
  -- Exact model equalities expose the executed schedule before any real-error analysis.
  -- No assumption about the accuracy of a transcendental approximation is introduced here.
  if (← getMainTarget).isAppOf ``Eq then
    decode
    return
  withMainContext do
    let goal ← getMainGoal
    let target : Q(Prop) ← instantiateMVars (← goal.getType)
    let actual : Q(ℝ) ← match target with
      | ~q(|($actual : ℝ) - $_reference| ≤ $_budget) => pure actual
      | ~q(($actual : ℝ) ≠ 0) => pure actual
      | ~q(0 < ($actual : ℝ)) => pure actual
      | _ => throwError "roundoff expects an absolute-error bound or a nonzero/positive result"
    let (_, boundaries) ← programRules
    let (rules, observations) ← indexRules boundaries
    let (estimate, state) ← (analyze actual).run { rules, observations, boundaries }
    let error := estimate.error
    let (proof, remaining) ← match target with
      | ~q(|($_actual : ℝ) - $reference| ≤ $budget) => do
          let same ← mkFreshExprMVar (← mkEq estimate.reference reference)
          unless ← referenceEquality same.mvarId! do
            throwError "roundoff derived the reference {estimate.reference}, but the goal uses \
              {reference}; establish their equality before applying the numerical bound"
          let statement ← mkAppM ``congrArg
            #[q(fun value : ℝ => |$actual - value| ≤ $error), same]
          let numerical ← mkAppM ``Eq.mp #[statement, estimate.proof]
          let remaining ← mkFreshExprMVar q($error ≤ $budget)
          pure (← mkAppM ``le_trans #[numerical, remaining], remaining)
      | ~q(($_actual : ℝ) ≠ 0) => do
          let reference := estimate.reference
          let remaining ← mkFreshExprMVar q($error < |$reference|)
          pure (← mkAppM ``nonzero_of_error_lt #[estimate.proof, remaining], remaining)
      | ~q(0 < ($_actual : ℝ)) => do
          let reference := estimate.reference
          let remaining ← mkFreshExprMVar q($error < $reference)
          pure (← mkAppM ``positive_of_error_lt #[estimate.proof, remaining], remaining)
      | _ => throwError "roundoff: unsupported target"
    goal.assign proof
    let mut pending := state.pending.toList
    unless ← discharge remaining.mvarId! do pending := pending ++ [remaining.mvarId!]
    replaceMainGoal pending

/-- Expose selected implementation definitions before the numerical pass. -/
syntax "roundoff" " [" ident,+ "]" : tactic

macro_rules
  | `(tactic| roundoff [$[$names:ident],*]) =>
      `(tactic| (simp (config := {
        failIfUnchanged := false, implicitDefEqProofs := false }) only
        [$[$names:ident],*] at *; roundoff))

/-- Infer a reference and budget with the ordinary numerical pass, retaining its obligations. -/
private def inferBound (actual : Q(ℝ)) : TacticM (Lean.Expr × Array MVarId) := do
  let reference : Q(ℝ) ← mkFreshExprMVar q(ℝ)
  let budget : Q(ℝ) ← mkFreshExprMVar q(ℝ)
  let numerical ← mkFreshExprMVar q(|$actual - $reference| ≤ $budget)
  let pending ← Tactic.run numerical.mvarId! do evalTactic (← `(tactic| roundoff))
  let pending ← pending.filterM fun goal => return !(← goal.isAssigned)
  return (← instantiateMVars numerical, pending.toArray)

/-- Infer a numerical statement for a local `have`, reusing the ordinary proof-producing pass.
The term is a real observation of the computation; both its reference and budget are inferred. -/
elab "roundoff" "(" value:term ")" : tactic => withMainContext do
  let actual : Q(ℝ) ← Term.elabTermEnsuringType value q(ℝ)
  Term.synthesizeSyntheticMVarsNoPostponing
  let (proof, pending) ← inferBound actual
  let mut goals := pending
  for goal in (proof.collectMVars {}).result do
    unless goals.contains goal do goals := goals.push goal
  let mut types := #[]
  let mut premises := #[]
  for goal in goals do
    if ← goal.isAssigned then continue
    let type ← instantiateMVars (← goal.getType)
    unless types.contains type do
      types := types.push type
      premises := premises.push m!"{type}"
  unless premises.isEmpty do
    throwError "roundoff: prove these premises before inferring a local bound:\n\
      {MessageData.joinSep premises.toList (m!"\n")}"
  let goal ← (← getMainGoal).change (← inferType proof)
  goal.assign proof
  replaceMainGoal []

/-- Refuse a generated conditional theorem when a closed obligation computes to false.
This is a bounded check, not a decision procedure for the consistency of arbitrary premises. -/
private def checkGeneratedConditions (pending : Array MVarId) : TacticM Unit := do
  for goal in pending do
    if ← goal.isAssigned then continue
    let condition ← instantiateMVars (← goal.getType)
    if condition.hasFVar || condition.hasExprMVar then continue
    let refuted? ← attempt? 500 do
      let decision ← withDefault (whnf (← mkDecide condition))
      return if decision.isConstOf ``Bool.false then some () else none
    if refuted?.isSome then
      throwError "roundoff: a required numerical condition is false: {condition}"

/-- Construct the proof term inside Lean's ordinary declaration elaboration, which handles section
variables, universes and declaration checking. The function's arguments retain their binders. -/
elab "roundoff_proof% " computation:term : term => Term.withoutErrToSorry do
  let computation ← Term.elabTerm computation none
  Term.synthesizeSyntheticMVarsNoPostponing
  let computation ← instantiateMVars computation
  let value ← forallTelescope (← inferType computation) fun arguments result => do
    unless ← isDefEq result q(ℝ) do throwError "roundoff: expected a real observation"
    let actual : Q(ℝ) := (mkAppN computation arguments).headBeta
    let expected ← mkFreshExprMVar q(Prop)
    let proof ← mkFreshExprMVar expected
    let remaining ← Tactic.run proof.mvarId! do
      let (numerical, pending) ← inferBound actual
      checkGeneratedConditions pending
      let value ← if pending.isEmpty then pure numerical else
        withConditions pending fun _ evidence => do
          mkLambdaFVars #[evidence] (← instantiateMVars numerical)
      let goal ← (← getMainGoal).change (← inferType value)
      goal.assign value
      replaceMainGoal []
    unless remaining.isEmpty do throwError "roundoff: unfinished generated proof"
    mkLambdaFVars arguments (← instantiateMVars proof)
  if value.hasExprMVar || value.hasLooseBVars || value.hasSorry then
    throwError "roundoff: generated proof contains unresolved obligations"
  check value
  return value

-- Infer the proof once; the public theorem reuses its checked type and value.
elab doc:(docComment)? "roundoff" "theorem" name:ident " := " computation:term : command => do
  let proofName ← Command.liftCoreM <| Core.mkFreshUserName (name.getId ++ `proof)
  let proof := mkIdentFrom name proofName
  let errorsBefore := (← get).messages.toList.countP (·.severity == .error)
  Command.elabCommand (← `(command|
    private abbrev $proof := @(roundoff_proof% $computation)))
  if (← get).messages.toList.countP (·.severity == .error) > errorsBefore then return
  Command.elabCommand (← `(command|
    $[$doc:docComment]? theorem $name : type_of% (private_decl% @$proof) := @$proof))

end Proofs.RuntimeApprox.Roundoff
