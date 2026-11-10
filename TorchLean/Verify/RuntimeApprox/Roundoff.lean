/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import FloatLib.Floats.Formats.BinaryInterchange.Automation.Roundoff
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Transcendentals
public import TorchLean.Tensor.ConfiguredBinary
public import TorchLean.Tensor.SomeTensor
public import TorchLean.Tensor.Reduce.Backward
public import TorchLean.Tensor.Contract
public import TorchLean.Tensor.Cumulative
public import TorchLean.Tensor.Layers.Activation
public import TorchLean.Tensor.Reduce.Fold
public import TorchLean.Verify.RuntimeApprox.Rounding
public import TorchLean.Verify.RuntimeApprox.Roundoff.Recovery
public import Mathlib.Data.Rat.Cast.Lemmas
public import TorchLean.Verify.RuntimeApprox.Roundoff.Fold
public import TorchLean.Verify.RuntimeApprox.Roundoff.Mixed
public import TorchLean.Tensor.List
public meta import Mathlib.Util.Qq
public meta import Lean.Meta.Match.MatcherInfo
public meta import TorchLean.Verify.RuntimeApprox.Roundoff.Rules
public import FloatLib.Floats.Formats.BinaryInterchange.Automation.Roundoff.Extension
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules

/-!
# Tensors in `roundoff`

FloatLib's `roundoff` analysis constructs numerical proofs from configured floating-point code
(`FloatLib.Floats.Formats.BinaryInterchange.Automation.Roundoff`), and TorchLean's tensors join it
as one `Extension`. Tensors and their storage stay folded during program normalization, an entry
is read by `Tensor.Rep.get` at a coordinate, and the coordinate laws read it through the
specification operation that computes it. Tensor folds and finite index loops reuse one scalar
invariant theorem through the coordinates (`fold_error`, `fold_mixedError`), without unrolling
their elements, and operations on several tensors reduce over their shared coordinates. Sums are
`Tensor.sum` and the entries of `reduce .sum` and `cumsum`. `Tensor.sum` and a sum over every axis
are the flat fold of the input, a sum over one axis cast to the shape without it is the fold of the
vector along that axis, and an entry of `cumsum` is the fold of its prefix. Other entries of
`reduce .sum` are read by the rule `reduce_sum_apply`. A transpose, such as
`(Shape.scalar ++ [k, w]).swapAdjacentAtDepth 0`, has its output shape reduced to its extents,
so that matching a rule stated at literal extents against the transposed operand stays within the
matcher's budget.

A pattern match on a length `k` that is not a literal, such as the length of a vector softmax, is
read at `k - 1 + 1` when the read coordinate or a hypothesis shows `0 < k`, and the tensors it
matches on are cast to that length. In the same way a rule stated at a successor length `n + 1`,
such as a vector theorem, applies to a tensor whose outer extent `k` is not literal, read at
`k - 1 + 1`. A producer with a `roundoff_read` stays folded where its shape is not literal, and its
read applies, except where the entry it exposes reads that producer again at a shape with the same
axes, such as a row read of a vector.

A contraction is read as one fold for each output entry. Matrix-vector, vector-matrix and matrix
products are read before that, as their loops over the shared axis, so a layer that
multiplies by a matrix is analyzed as the loop it computes. A concatenation is read at the tensor
whose block holds the coordinate, and a two-block concatenation as its two branches. An index
selection, the gather that batches every axis but the selected one, is read at its selected index,
and a permutation by `Shape.adjacentSwap` at the coordinate with the two swapped indices exchanged.

On an IEEE descriptor, the `exp` and `log` of the configured `MathFunctions` TorchLean selects and
the scalar context's `expm1`, `log1p` and `Context.expMul` are registered as FloatLib's correctly
rounded `exp`, `log`, `expMinus1`, `logPlus1` and `expMul`. The theorems below give the real value
of a max-absolute reduction of finite values, bound every entry of a finite vector by its entry at
`argmax` or at `Softmax.pivot`, and decode tensor folds through the codecs around their
steps.

An IEEE `maximum`, `minimum`, `fmax`, `fmin` or `clamp` of exact data, such as inputs, their
entries, numerals and selections among them, is read through its definition outside loop steps,
so that its cases are analyzed exactly. A loop step, an extremum of computed values and a maximum
against zero keep the finite-operand rules (`toReal_maximum`).
-/

public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange

section Configured

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- On finite configured values, in a format with signed zeros, `Tensor.maxAbs` is exact: its real
value is the left fold of `max` over the absolute real values of the entries, from `0`. -/
theorem Roundoff.toReal_maxAbs [Storage Value] {shape : Shape} (input : Tensor Value shape)
    (signedZero : format.supportsSignedZero = true)
    (finite : ∀ coordinate, ExecFloat.Binary.isFinite (input coordinate) = true) :
    ExecFloat.Binary.toReal (Tensor.maxAbs input) =
      input.foldl (fun (largest : ℝ) value => max largest |ExecFloat.Binary.toReal value|) 0 := by
  have related := Tensor.Rep.foldl_rel input input
    (fun largest value : Value =>
      if !(largest == largest) then largest
      else if !(value == value) then value
      else max largest (MathFunctions.abs value))
    (fun (largest : ℝ) value => max largest |ExecFloat.Binary.toReal value|)
    (fun (largest : Value) (reference : ℝ) =>
      ExecFloat.Binary.isFinite largest = true ∧ ExecFloat.Binary.toReal largest = reference)
    0 0 ⟨ExecFloat.Binary.isFinite_zero, ExecFloat.Binary.toReal_zero⟩ (by
      intro coordinate largest reference state
      obtain ⟨largestFinite, observedLargest⟩ := state
      have valueFinite := finite coordinate
      have largestSelf :=
        (ExecFloat.Binary.beq_eq_decide_toReal_eq largestFinite largestFinite).trans
          (decide_eq_true rfl)
      have valueSelf :=
        (ExecFloat.Binary.beq_eq_decide_toReal_eq valueFinite valueFinite).trans
          (decide_eq_true rfl)
      simp only [largestSelf, valueSelf, Bool.not_true, Bool.false_eq_true, reduceIte]
      have absoluteFinite :
          ExecFloat.Binary.isFinite (MathFunctions.abs (input coordinate)) = true :=
        ExecFloat.Binary.isFinite_abs valueFinite
      have observedAbsolute : ExecFloat.Binary.toReal (MathFunctions.abs (input coordinate)) =
          |ExecFloat.Binary.toReal (input coordinate)| := by
        change Model.toReal
            (ExecFloat.Binary.toModel (ExecFloat.Binary.abs (input coordinate))) =
          |Model.toReal (ExecFloat.Binary.toModel (input coordinate))|
        rw [ExecFloat.Binary.toModel_abs]
        exact Model.toReal_abs_of_isFinite _ signedZero valueFinite
      constructor
      · change Model.isFinite (ExecFloat.Binary.toModel
          (max largest (ExecFloat.Binary.abs (input coordinate)))) = true
        rw [ExecFloat.Binary.toModel_max]
        exact Model.isFinite_maximum _ _ largestFinite absoluteFinite
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
  simpa only [Tensor.maxAbs] using related.2

/-- The entry a vector of finite configured values holds at its `argmax` has the largest real
value. -/
theorem Roundoff.toReal_le_argmax [Storage Value] {n : Nat} (input : Tensor Value [n]) (h : 0 < n)
    (finite : ∀ index, ExecFloat.Binary.isFinite (input.getScalar index) = true) (index : Fin n) :
    ExecFloat.Binary.toReal (input.getScalar index) ≤
      ExecFloat.Binary.toReal
        (input.getScalar (Tensor.argmax input ⟨0, Nat.one_pos⟩ false h).item) := by
  refine Tensor.le_getScalar_argmax input h ExecFloat.Binary.toReal
    (fun j => (ExecFloat.Binary.beq_eq_decide_toReal_eq (finite j) (finite j)).trans
      (decide_eq_true rfl)) (fun i j => ?_) index
  exact ExecFloat.Binary.lt_iff_compare_eq_lt.trans
    (Model.compare_eq_some_lt_iff_toReal_lt_of_isFinite _ _ (finite j) (finite i))

/-- At a vector of finite configured values, the entry at `Softmax.pivot` has the
largest real value. Log-softmax and `logsumexp` shift by this entry, so every shifted value is
nonpositive. -/
theorem Roundoff.toReal_le_pivot [Storage Value] {n : Nat} (input : Tensor Value [n]) (h : 0 < n)
    (finite : ∀ index, ExecFloat.Binary.isFinite (input.getScalar index) = true) (index : Fin n) :
    ExecFloat.Binary.toReal (input.getScalar index) ≤
      ExecFloat.Binary.toReal (input.getScalar (Softmax.pivot input h)) := by
  rw [Softmax.pivot_eq_argmax input h fun j =>
    (ExecFloat.Binary.beq_eq_decide_toReal_eq (finite j) (finite j)).trans (decide_eq_true rfl)]
  exact toReal_le_argmax input h finite index

end Configured

/-- For input and accumulator formats that may differ, the model value of a tensor fold is the fold
of the model values with the step conjugated by the codecs. -/
theorem Roundoff.decode_foldl {inputFormat stateFormat : FloatFormat}
    {inputPlan : Configured.StoragePlan inputFormat}
    {statePlan : Configured.StoragePlan stateFormat} {inputCode stateCode : Type}
    [ExecFloat.ModelCodec inputPlan (Model inputFormat) inputCode]
    [ExecFloat.ModelCodec statePlan (Model stateFormat) stateCode]
    [Storage (ExecFloat (Configured.Family inputFormat inputCode inputPlan))]
    [Storage (Model inputFormat)]
    {shape : Shape}
    (input : Tensor (ExecFloat (Configured.Family inputFormat inputCode inputPlan)) shape)
    (step : ExecFloat (Configured.Family stateFormat stateCode statePlan) →
      ExecFloat (Configured.Family inputFormat inputCode inputPlan) →
      ExecFloat (Configured.Family stateFormat stateCode statePlan))
    (initial : ExecFloat (Configured.Family stateFormat stateCode statePlan)) :
    ExecFloat.Binary.toModel (Tensor.foldl step initial input) =
      Tensor.foldl
        (fun state value => ExecFloat.Binary.toModel
          (step (ExecFloat.Binary.ofModel state) (ExecFloat.Binary.ofModel value)))
        (ExecFloat.Binary.toModel initial)
        (Tensor.map ExecFloat.Binary.toModel input) := by
  apply Tensor.Rep.map_foldl
  intro accumulator value
  simp only [ExecFloat.Binary.ofModel_toModel]

/-- An entry of a tensor computed from an outer extent `k` is the entry of the same computation
at an equal extent `j`, at the coordinate with its outer index cast. -/
theorem Tensor.get_eq_of_length_eq {α : Type} [Storage α] {k j : Nat} {inner : Shape}
    (program : (j : Nat) → k = j → Tensor α (.dim j inner)) (h : k = j)
    (c : Shape.Coord (.dim k inner)) :
    program k rfl c = program j h (Fin.cast h c.1, c.2) := by
  cases h
  rfl

/-- A tensor computed from a length `k` is the same computation at an equal length `j`, cast back
along its shape. -/
theorem Tensor.eq_castShape_of_length_eq {α : Type} [Storage α] {k j : Nat} (shape : Nat → Shape)
    (program : (j : Nat) → k = j → Tensor α (shape j)) (h : k = j) :
    program k rfl = Tensor.castShape (program j h) (congrArg shape h).symm := by
  cases h
  rfl

end TorchLean

public meta section

namespace TorchLean.Roundoff

open Lean Meta Qq
open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange
open FloatLib.Floats.Formats.BinaryInterchange.Roundoff (Extension)
open Tensor.Rep (foldl_coordinates_add_pos foldl_fst)

-- TorchLean's tensor reads, coefficient decoding and tensor observations.
attribute [roundoff_rule]
  configured_decode
  Numeric.QuotientCoefficients.primal_add
  Numeric.QuotientCoefficients.primal_sub
  Numeric.QuotientCoefficients.primal_mul
  Numeric.QuotientCoefficients.primal_div
  Numeric.QuotientCoefficients.primal_map
  Tensor.eq_mp_apply
  SomeTensor.cast_apply
  Tensor.matmulCommonBatch_apply
  Tensor.contract_apply
  Tensor.contractFast_eq
  Tensor.permuteFast_eq
  Tensor.cat_apply
  Tensor.reduceOuterAxis_apply
  Tensor.reduceBroadcast_self
  Tensor.reduceDimWith_apply
  Tensor.ofFun_broadcast_eraseIdx
  Shape.swapAdjacentAtDepth_append_rank
  toReal_maxAbs
  enclosedSlope_get_error

-- Casts into the format, divisions by a count, `argmax` and pivot reads, mean denominators and
-- the default safeguard. Rules of one conclusion are tried in this order: the finiteness of a
-- quotient by a count from the operand's enclosure before the operand alone, and its rounding
-- exact at the count one, then at a positive count, then under `u * |n| < |n|`.
attribute [roundoff_rule]
  natCast_rounding isFinite_natCast_of_le toReal_le_argmax toReal_le_pivot cast_axisMeanDenom_pos
  isFinite_defaultEpsilon isFinite_divNat_of_mixedError isFinite_divNat abs_toReal_divNat_sub_le
  divNat_one_rounding divNat_axisMeanDenom_rounding divNat_rounding
  isFinite_divNat_and_rounding cast_ite_eq_zero_one_pos cast_meanDenominator_pos cast_axisSize_pos

/-- A proof that `length` is positive: from the outer component of `coordinate` when it lies in
`Fin length`, or from a coordinate in `Fin length`, a hypothesis `0 < length` or a hypothesis
`length ≠ 0` in context. -/
private def positiveLength? (length coordinate : Expr) : MetaM (Option Expr) := do
  let same (other : Expr) : MetaM Bool := withNewMCtxDepth <| withReducible <| isDefEq other length
  -- Test the head before `appArg!`: the coordinate type of a rank-zero tensor is not an
  -- application.
  let isFin (type : Expr) : MetaM Bool := do
    let type ← whnfR type
    unless type.isAppOfArity ``Fin 1 do return false
    same type.appArg!
  let type ← whnfR (← inferType coordinate)
  if type.isAppOfArity ``Prod 2 then
    if ← isFin type.appFn!.appArg! then
      let outer ← mkAppOptM ``Prod.fst #[type.getArg! 0, type.getArg! 1, coordinate]
      return some (← mkAppM ``Fin.pos #[outer])
  for declaration in ← getLCtx do
    if declaration.isImplementationDetail then continue
    let type ← instantiateMVars declaration.type
    if ← isFin type then return some (← mkAppM ``Fin.pos #[declaration.toExpr])
    if type.isAppOfArity ``LT.lt 4 && (type.getArg! 2).nat? == some 0 then
      if ← same (type.getArg! 3) then return some declaration.toExpr
    if type.isAppOfArity ``Ne 3 && (type.getArg! 2).nat? == some 0 then
      if ← same (type.getArg! 1) then
        return some (← mkAppM ``Nat.pos_of_ne_zero #[declaration.toExpr])
  return none

/-- `head` applied to `arguments`, as a function of a length `j` and a proof `length = j`. At each
of `positions`, a tensor whose type mentions `length` is cast to the shape at `j`, and an argument
whose type does not mention it gets `j` in place of `length`. There is none when an argument
elsewhere mentions `length`, when the result does not depend on `j` (as for a variable or a
projection whose type alone carries `length`), when an argument at `positions` whose type mentions
`length` is not a tensor, or when the result does not typecheck. -/
private def lengthProgram? (head : Expr) (arguments : Array Expr) (positions : Array Nat)
    (length : Expr) : MetaM (Option Expr) := do
  let independent ← arguments.zipIdx.allM fun (argument, index) => do
    return positions.contains index || !(← kabstract argument length).hasLooseBVars
  unless independent do return none
  withLocalDeclD `j (mkConst ``Nat) fun j => do
    withLocalDeclD `h (← mkEq length j) fun h => do
      let mut generalized := arguments
      for position in positions do
        let argument := arguments[position]!
        let type ← inferType argument
        unless (← kabstract type length).hasLooseBVars do
          generalized := generalized.set! position ((← kabstract argument length).instantiate1 j)
          continue
        let tensorType ← whnfR type
        unless tensorType.isAppOfArity ``Tensor.Rep 3 do return none
        let source := tensorType.getArg! 1
        let family := Lean.mkLambda `j .default (mkConst ``Nat) (← kabstract source length)
        let target := family.beta #[j]
        let shapeEquality ← mkExpectedTypeHint (← mkCongrArg family h) (← mkEq source target)
        generalized := generalized.set! position (← mkAppOptM ``Tensor.castShape
          #[tensorType.getArg! 0, tensorType.getArg! 2, source, target, argument,
            shapeEquality])
      let body := mkAppN head generalized
      unless body.containsFVar j.fvarId! do return none
      try check body catch error =>
        if error.isInterrupt || error.isRuntime then throw error
        return none
      return some (← mkLambdaFVars #[j, h] body)

/-- Rewrite a tensor producer whose pattern match is stuck on a length `k` (a discriminant, or the
outer extent of a discriminant shape) at `k - 1 + 1`, where the match reduces. Either the read
coordinate `coordinate` or a hypothesis shows `0 < k` (`positiveLength?`). The discriminant
tensors are cast to the extent `k - 1 + 1`, and the result is cast back to its original shape. -/
private def splitLength? (tensor coordinate : Expr) : MetaM (Option (Expr × Expr)) := withDefault do
  let tensor ← instantiateMVars tensor
  if tensor.hasExprMVar then return none
  let some name := tensor.getAppFn.constName? | return none
  let some info ← getMatcherInfo? name | return none
  let arguments := tensor.getAppArgs
  unless arguments.size == info.arity do return none
  let resultType ← whnfR (← inferType tensor)
  unless resultType.isAppOfArity ``Tensor.Rep 3 do return none
  unless (← reduceMatcher? tensor) matches .stuck _ do return none
  let positions := (Array.range info.numDiscrs).map (info.getFirstDiscrPos + ·)
  let mut lengths := #[]
  for position in positions do
    let discriminant := arguments[position]!
    let type ← whnfR (← inferType discriminant)
    if type.isConstOf ``Nat then lengths := lengths.push discriminant
    else if type.isAppOfArity ``List 1 then
      let shape ← whnfR discriminant
      if shape.isAppOfArity ``List.cons 3 then lengths := lengths.push (shape.getArg! 1)
  for length in lengths do
    if length.nat?.isSome then continue
    let some positive ← positiveLength? length coordinate | continue
    -- Only the discriminants may mention the length.
    let some program ← lengthProgram? tensor.getAppFn arguments positions length | continue
    unless ← isDefEq (program.beta #[length, ← mkEqRefl length]) tensor do continue
    let successor ← mkAppM ``HAdd.hAdd #[← mkAppM ``HSub.hSub #[length, mkNatLit 1], mkNatLit 1]
    let equality ← mkEqSymm (← mkAppM ``Nat.sub_one_add_one_eq_of_pos #[positive])
    let split := program.beta #[successor, equality]
    unless (← reduceMatcher? split) matches .reduced _ do continue
    let shape := resultType.getArg! 1
    let family := Lean.mkLambda `j .default (mkConst ``Nat) (← kabstract shape length)
    let splitShape := family.beta #[successor]
    let back ← mkExpectedTypeHint (← mkEqSymm (← mkCongrArg family equality))
      (← mkEq splitShape shape)
    let result ← mkAppOptM ``Tensor.castShape
      #[resultType.getArg! 0, resultType.getArg! 2, splitShape, shape, split, back]
    let proof ← mkAppOptM ``Tensor.eq_castShape_of_length_eq #[resultType.getArg! 0,
      resultType.getArg! 2, length, successor, family, program, equality]
    return some (result, ← mkExpectedTypeHint proof (← mkEq tensor result))
  return none

/-- Restate a read `observe (t c)` of a tensor whose outer extent `k` is neither a numeral nor a
successor at `k - 1 + 1`, where the outer index of `c` or a hypothesis shows `0 < k`
(`positiveLength?`). The producer of `t` is computed at that extent from its inputs cast to it
(`get_eq_of_length_eq`), so that rules stated at a successor length, such as vector theorems,
apply. Returns the restated read, the observation it applies and a proof that the original entry
equals the restated one. -/
private def splitRead? (actual : Expr) : MetaM (Option (Expr × Expr × Expr)) := withDefault do
  let some source := actual.getAppArgs.back? | return none
  unless source.isAppOfArity ``Tensor.Rep.get 5 do return none
  if (← instantiateMVars source).hasExprMVar then return none
  let arguments := source.getAppArgs
  let (tensor, coordinate) := (arguments[3]!, arguments[4]!)
  let shape ← whnfR arguments[2]!
  unless shape.isAppOfArity ``List.cons 3 do return none
  let (length, inner) := (shape.getArg! 1, shape.getArg! 2)
  if length.nat?.isSome || length.isAppOfArity ``Nat.succ 1 ||
      (length.isAppOfArity ``HAdd.hAdd 6 && (length.getArg! 5).nat? == some 1) then
    return none
  if (← kabstract inner length).hasLooseBVars then return none
  let some positive ← positiveLength? length coordinate | return none
  let some program ← lengthProgram? tensor.getAppFn tensor.getAppArgs
      (Array.range tensor.getAppNumArgs) length | return none
  unless ← isDefEq (program.beta #[length, ← mkEqRefl length]) tensor do return none
  let successor ← mkAppM ``HAdd.hAdd #[← mkAppM ``HSub.hSub #[length, mkNatLit 1], mkNatLit 1]
  let equality ← mkEqSymm (← mkAppM ``Nat.sub_one_add_one_eq_of_pos #[positive])
  let (outer, rest) ← if coordinate.isAppOfArity ``Prod.mk 4 then
      pure (coordinate.getArg! 2, coordinate.getArg! 3)
    else pure (← mkAppM ``Prod.fst #[coordinate], ← mkAppM ``Prod.snd #[coordinate])
  let read ← mkAppOptM ``Tensor.Rep.get #[arguments[0]!, arguments[1]!,
    ← mkAppM ``List.cons #[successor, inner], program.beta #[successor, equality],
    ← mkAppM ``Prod.mk #[← mkAppM ``Fin.cast #[equality, outer], rest]]
  let proof ← mkAppOptM ``Tensor.get_eq_of_length_eq
    #[arguments[0]!, arguments[1]!, length, successor, inner, program, equality, coordinate]
  let proof ← mkExpectedTypeHint proof (← mkEq source read)
  let observe ← withLocalDeclD `value (← inferType source) fun value =>
    mkLambdaFVars #[value] (mkAppN actual.getAppFn (actual.getAppArgs.pop.push value))
  return some ((mkApp observe read).headBeta, observe, proof)

/-- The number of axes `shape` lists and whether the list ends there. A shape that does not end
is symbolic, such as a shape variable or a shape computed from one. -/
private partial def shapeSkeleton (shape : Expr) : MetaM (Nat × Bool) := do
  let shape ← whnf shape
  if shape.isAppOfArity ``List.cons 3 then
    let (axes, literal) ← shapeSkeleton shape.appArg!
    return (axes + 1, literal)
  return (0, shape.isAppOfArity ``List.nil 1)

/-- The skeleton of the shape of the tensor `value` (`shapeSkeleton`). -/
private def tensorSkeleton (value : Expr) : MetaM (Option (Nat × Bool)) := do
  let type ← whnfR (← inferType value)
  unless type.isAppOfArity ``Tensor.Rep 3 do return none
  shapeSkeleton (type.getArg! 1)

/-- Whether the tensor `value` or a tensor argument of its producer has a shape that is not
literal (`tensorSkeleton`). -/
private def hasSymbolicShape (value : Expr) : MetaM Bool :=
  (value.getAppArgs.push value).anyM fun term => do return (← tensorSkeleton term).any (!·.2)

/-- Read a tensor whose pattern match is stuck on a positive length at the successor length,
through `splitLength?`. -/
private def observeLength : Simp.Simproc := fun expression => do
  unless expression.isAppOfArity ``Tensor.Rep.get 5 do return .continue
  let arguments := expression.getAppArgs
  let some (tensor, equality) ← splitLength? arguments[3]! arguments[4]! | return .continue
  let observe ← withLocalDeclD `tensor (← inferType arguments[3]!) fun tensor =>
    mkLambdaFVars #[tensor] (mkAppN expression.getAppFn (arguments.set! 3 tensor))
  let read := mkAppN expression.getAppFn (arguments.set! 3 tensor)
  return .visit { expr := read, proof? := some (← mkCongrArg observe equality) }

/-- The numeral a closed count computes to, such as the number of slices of a concrete shape. A
count with free variables is not reduced: the slice count of a vector of any length has its own
rule (`divNat_axisMeanDenom_rounding`), and reducing a count of variables could unfold the
definitions they appear in without end. -/
private def countNumeral? (count : Expr) : MetaM (Option Nat) := do
  if count.hasFVar || count.hasMVar then return none
  let value ← withDefault (whnf count)
  return value.rawNatLit? <|> value.nat?

/-- Read a division by a count that computes to a numeral at that numeral, where the exact rules
apply, as the analysis reads a cast count. -/
private def reduceDivNatCount : Simp.DSimproc := fun expression => do
  unless expression.isAppOfArity ``Context.divNat 4 do return .continue
  let count := expression.appArg!
  if count.nat?.isSome || count.rawNatLit?.isSome then return .continue
  let some numeral ← countNumeral? count | return .continue
  return .done (mkApp expression.appFn! (mkNatLit numeral))

/-- The number of arguments of an IEEE extremum of `Context`, or `none` for another constant. -/
private def extremumArity? (name : Name) : Option Nat :=
  if name == ``Context.maximum || name == ``Context.minimum || name == ``Context.fmax ||
      name == ``Context.fmin then some 4
  else if name == ``Context.clamp then some 5
  else none

/-- Whether `operand` is the numeral zero. -/
private def isZeroNumeral (operand : Expr) : Bool :=
  let operand := operand.consumeMData
  (operand.isAppOfArity ``OfNat.ofNat 3 && operand.appFn!.appArg!.rawNatLit? == some 0) ||
    operand.isAppOfArity ``Zero.zero 2

/-- Whether `operand` is exact data: a variable, a numeral or its negation, an entry of a variable,
or a selection among exact data, such as an IEEE extremum of exact data or its cases. -/
private partial def isExactOperand (operand : Expr) : Bool :=
  let operand := operand.consumeMData
  if operand.isFVar || operand.isAppOfArity ``OfNat.ofNat 3 ||
      operand.isAppOfArity ``Zero.zero 2 || operand.isAppOfArity ``One.one 2 then true
  else if operand.isAppOfArity ``Neg.neg 3 || operand.isAppOfArity ``Context.quietNaN 3 ||
      operand.isAppOfArity ``Context.extremaTie 5 then isExactOperand operand.appArg!
  else if operand.isAppOfArity ``Tensor.Rep.get 5 then
    operand.appFn!.appArg!.consumeMData.isFVar
  else if operand.isAppOfArity ``ite 5 then
    isExactOperand operand.appFn!.appArg! && isExactOperand operand.appArg!
  else if operand.isAppOfArity ``dite 5 then
    let branch := fun | .lam _ _ body _ => isExactOperand body | _ => false
    branch operand.appFn!.appArg! && branch operand.appArg!
  else match operand.getAppFn.constName? >>= extremumArity? with
    | some arity => operand.getAppNumArgs == arity &&
        (operand.getAppArgs.extract 2 arity).all isExactOperand
    | none => false

/-- Read an IEEE extremum of exact data outside a loop step through its definition, so that the
analysis reads its cases exactly. An extremum of computed values keeps the finite-operand rules
(`toReal_maximum`), and so do a fold step, whose operands are bound in the simplified term or are
the accumulator of the loop step under analysis, and a maximum against zero. -/
private def unfoldExtremum : Simp.DSimproc := fun expression => do
  let some name := expression.getAppFn.constName? | return .continue
  let some arity := extremumArity? name | return .continue
  unless expression.getAppNumArgs == arity do return .continue
  let operands := expression.getAppArgs.extract 2 arity
  if name == ``Context.maximum && isZeroNumeral operands[1]! then return .continue
  unless operands.all isExactOperand do return .continue
  let initial := (← Simp.getContext).lctxInitIndices
  let context ← getLCtx
  let inLoop := operands.any fun operand => (operand.find? fun child => child.isFVar &&
    (context.find? child.fvarId!).any fun declaration =>
      initial ≤ declaration.index || declaration.userName == `accumulator).isSome
  if inLoop then return .continue
  let some unfolded ← delta? expression | return .continue
  return .visit unfolded

/-- Whether every extent of `shape` is a numeral, as in `[2, 3]`. -/
private partial def numeralShape (shape : Expr) : MetaM Bool := do
  let shape ← whnf shape
  unless shape.isAppOfArity ``List.cons 3 do return shape.isAppOfArity ``List.nil 1
  let extent ← instantiateMVars shape.appFn!.appArg!
  unless extent.nat?.isSome || extent.rawNatLit?.isSome do return false
  numeralShape shape.appArg!

/-- Whether `shape` is a literal list whose extents after the first are numerals, as in `[n, 2, 3]`,
so that its flat index mentions no symbolic extent. -/
private def numeralStrides (shape : Expr) : MetaM Bool := do
  let shape ← whnf shape
  unless shape.isAppOfArity ``List.cons 3 do return shape.isAppOfArity ``List.nil 1
  numeralShape shape.appArg!

/-- Compute the flat index of a coordinate of a literal shape whose extents after the first are
numerals, axis by axis through `linearizeFast`, in the normal forms that unfold `linearizeFast`. At
any other shape the index stays `linearize`, the form its laws state. -/
private def linearizeLiteral : Simp.Simproc := fun expression => do
  unless expression.isAppOfArity ``Shape.Coord.linearize 2 do
    return .continue
  let fast := ``Shape.Coord.linearizeFast
  unless (← Simp.getContext).isDeclToUnfold fast do return .continue
  let shape := expression.appFn!.appArg!
  unless (← numeralStrides shape) do return .continue
  let law := mkConst ``Shape.Coord.linearizeFast_eq
  let equality ← mkCongrFun (← mkCongrFun law shape) expression.appArg!
  return .visit { expr := mkAppN (mkConst fast) expression.getAppArgs, proof? := some equality }

/-- TorchLean's tensors: their reads, folds, coordinate and fold laws, the successor-length reads of
shape-recursive producers, the cases of the extrema of exact data, and the correctly rounded
functions of the scalars it selects. -/
def tensors : Extension where
  containers := #[``Tensor, ``Tensor.Rep]
  -- The `Tensor.Rep` modules define their basic operations under their public names.
  storage := #[`TorchLean.Tensor.Rep, ``Tensor.full, ``Tensor.map, ``Tensor.zipWith,
    ``Tensor.zipWith3, ``Tensor.dim, ``Tensor.unstack, ``Tensor.set, ``Tensor.modify,
    ``Tensor.foldl, ``Tensor.cast]
  reads := #[
    { head := ``Tensor.Rep.get, arity := 5,
      types := #[``Tensor.Rep], trusts := #[``Storage],
      observers := #[``Storage.toArray],
      laws := #[``Storage.toArray_injective, ``Storage.toArray_size,
        ``Storage.get_eq_toArray_getElem, ``Storage.uget_eq_toArray_uget,
        ``Storage.toArray_emptyWithCapacity, ``Storage.toArray_push,
        ``Storage.copyAt_eq_push_uget, ``Storage.toArray_appendSlice,
        ``Storage.toArray_foldl, ``Storage.toArray_ofArray,
        ``Storage.gather_eq_finFoldl, ``Storage.toArray_transpose2D],
      selection := some ``Tensor.Rep.get_eq_toArray_getElem },
    { head := ``Tensor.getScalar, arity := 5, canonical := false,
      types := #[``Tensor.Rep],
      selection := some ``Tensor.getScalar_eq_apply }]
  observations := #[``Tensor.get2, ``Tensor.getScalar, ``Tensor.item,
    ``Tensor.zipWith, ``Tensor.Rep.get,
    ``Tensor.Rep.getFlat, ``Shape.Coord.linearize]
  surfaces := #[``Tensor.getScalar, ``Tensor.get2, ``Tensor.item]
  loops := #[{
    fold := ``Tensor.foldl, arity := 7, element := 0, state := 1, step := 4,
    initial := 5, input := 6, foldStep := ``List.foldStep, allSteps := ``AllSteps,
    allStepsOfMap := ``AllSteps.of_map, allStepsOfForall := ``AllSteps.of_forall,
    foldError := ``foldError, foldErrorIdentity := ``foldError_identity,
    errorBound := ``fold_error, mixedFoldStep := ``List.mixedFoldStep,
    foldRelative := ``foldRelative, foldAbsolute := ``foldAbsolute,
    mixedBound := ``fold_mixedError,
    induction := ``Tensor.foldl_induction,
    coordinates := some (``Tensor.Rep.coordinates,
      ``Tensor.Rep.foldl_eq_coordinates),
    single := some ``fold_eq_of_single, positiveSum := some ``foldl_coordinates_add_pos,
    projection := some ``foldl_fst }]
  loopHeads := #[``Tensor.dot]
  -- `Tensor.sum` and a sum over every axis are the flat fold of the input, and a sum over one
  -- axis, read without that axis at plain coordinate pairs over the outer axis, is the flat fold
  -- of the vector along it. The rule `reduce_sum_apply` reads the other entries of a sum, and an
  -- entry of `cumsum` is the flat fold of its prefix.
  sums := #[
    { head := ``Tensor.sum, laws := #[(``Tensor.sum_eq_foldl, false)] },
    { head := ``Tensor.reduce, entries := true,
      laws := #[(``Tensor.castShape_reduce_sum_zero_apply, false),
        (``Tensor.reduce_sum_finRange_apply, false),
        (``Tensor.castShape_reduce_sum_singleton_mk_apply, false),
        (``Tensor.castShape_reduce_sum_singleton_apply_of_lt, false),
        (``Tensor.castShape_reduce_sum_singleton_apply, false)] },
    { head := ``Tensor.cumsum, entries := true,
      laws := #[(``Tensor.cumsum_apply, false)] }]
  coordinateUnfold := #[``Tensor.scale,
    ``Tensor.square, ``Tensor.neg,
    ``Tensor.exp, ``Tensor.log, ``Tensor.item,
    ``Tensor.ofFn, ``Tensor.generate,
    ``Tensor.Rep.coordinates,
    ``Tensor.Rep.broadcast,
    ``Shape.Coord.linearizeFast,
    ``Shape.Coord.linearizeAux,
    -- A transposed operand has the type of its computed output shape, and reducing that shape
    -- keeps the matching of a rule whose operand has literal extents within the matcher's budget.
    ``Shape.swapAdjacentAtDepth]
  readRewrites := #[``Tensor.get2_eq_apply,
    ``Tensor.getScalar_eq_apply, ``Tensor.Rep.get_ofFn,
    ``Tensor.full_apply, ``Tensor.map_apply,
    ``Tensor.zipWith_apply, ``Tensor.Rep.pull_apply,
    ``Tensor.dim_apply, ``Tensor.unstack_apply]
  -- Tensor `+`, `-`, `*` and `/` on one element type are read through their `zipWith`.
  coordinateRewrites := #[``Tensor.add_def, ``Tensor.sub_def,
    ``Tensor.mul_def, ``Tensor.div_def, ``Tensor.scalar_apply,
    ``Tensor.Rep.getFlat_eq_get, ``Tensor.Rep.get_ofFlatFn,
    ``Shape.Coord.broadcast_self,
    ``Tensor.Rep.reshape_apply,
    ``Tensor.reduceOuterAxis_vector,
    ``Tensor.reduceDimWith_scalar,
    ``Tensor.reduceDimWith_dim_zero,
    ``Tensor.reduceDimWith_dim_succ,
    -- Converse slice laws keep checked backward slices at the level of their scalar reads.
    `TorchLean.Tensor.sliceBackward_apply_sliceAxisRangeCoordinate,
    `TorchLean.Tensor.sliceBackward_apply_eq_zero]
  foldUnfold := #[``Tensor.dot,
    ``Tensor.ofFn, ``Tensor.generate, ``Tensor.scale,
    ``Tensor.square, ``Tensor.neg,
    ``Tensor.exp, ``Tensor.log,
    ``Tensor.Rep.pull]
  -- A loop over `Fin n` is the fold over the coordinates of a vector.
  foldRewrites := #[(``Tensor.add_def, false), (``Tensor.sub_def, false),
    (``Tensor.mul_def, false), (``Tensor.div_def, false),
    (``Tensor.Rep.ofFn_eq_map_coordinates, false),
    (``Tensor.Rep.zipWith_eq_map_coordinates, false),
    (``Tensor.Rep.foldl_map, false), (``AllSteps.map, false),
    (``foldError_map, false), (``Fin.foldl_eq_foldl_finRange, true),
    (``Tensor.fin_foldl_eq_coordinates, false)]
  -- A shape cast of a nonempty shape is read one axis at a time, before the registered general cast
  -- read, so that facts stated at coordinate pairs apply to the cast tensor. A sum over one axis
  -- cast to the shape without it is first written as the sums along that axis, so that the cast
  -- read cannot separate it from its law where the shapes are literal, at plain coordinate pairs
  -- over the outer axis. Ahead of the general contraction, gather and permutation reads, the
  -- matrix products are read as their loops over the shared axis, an index selection at its
  -- selected index and an adjacent swap at the coordinate with the two indices exchanged. A
  -- two-block `cat` is read as its two branches, and a `cat` of more blocks as its first block and
  -- the `cat` of the others (of one block as that block). `maximum` and `minimum` are read as their
  -- `zipWith`, and a policy argument of an elementwise operator as the branch it selects.
  programRewrites := #[(``Tensor.maximum_def, eval_prio high),
    (``Tensor.minimum_def, eval_prio high),
    (``Tensor.sqrt_raw, eval_prio high),
    (``Tensor.sqrt_clampNegative, eval_prio high),
    (``Tensor.clamp_propagate, eval_prio high),
    (``Tensor.clamp_interior, eval_prio high),
    (``Tensor.broadcast_self, eval_prio default),
    (``Tensor.castShape_self, eval_prio default),
    (``Tensor.castShape_dim_apply, eval_prio high),
    (``Tensor.castShape_reduce_sum_zero, eval_prio high + 1),
    (``Tensor.castShape_reduce_sum_singleton_mk, eval_prio high),
    (``Tensor.castShape_reduce_sum_singleton, eval_prio high - 1),
    (``Tensor.contract_matVec_apply, eval_prio high),
    (``Tensor.contract_vecMat_apply, eval_prio high),
    (``Tensor.contract_matMul_apply, eval_prio high),
    (``Tensor.cat_one_apply, eval_prio high + 1),
    (``Tensor.cat_two_apply, eval_prio high),
    (``Tensor.cat_cons_apply, eval_prio high),
    (``Tensor.cat_succ_apply, eval_prio high - 1),
    (``Tensor.gather_select_apply, eval_prio high),
    (``Tensor.permute_swapAdjacentAtDepth_apply, eval_prio high + 1),
    (``Tensor.permute_adjacentSwap_zero_apply, eval_prio high + 1),
    (``Tensor.permute_adjacentSwap_apply, eval_prio high),
    (``Tensor.unstack_apply, eval_prio default)]
  decodeLemmas := #[``decode_foldl,
    ``Tensor.Rep.foldl_map, ``Tensor.item,
    ``Tensor.reshape_apply, ``Tensor.castShape_apply,
    ``Tensor.reduce_sum_finRange_apply]
  -- A generated callback may use the flat getter.
  readRecoveries := #[(``Tensor.Rep.getFlat, 5,
    ``Tensor.Rep.getFlat_eq_get)]
  readFallbacks := #[``Tensor.Rep.get_ofFlatFn]
  bridges := #[
    (``ExecFloat.Binary.CorrectlyRounded.exp,
      ``ExecFloat.Binary.CorrectlyRounded.mathFunctions_exp_of_isIEEE),
    (``ExecFloat.Binary.CorrectlyRounded.log,
      ``ExecFloat.Binary.CorrectlyRounded.mathFunctions_log_of_isIEEE),
    (``ExecFloat.Binary.CorrectlyRounded.expMinus1, ``ConfiguredBinary.expm1_of_isIEEE),
    (``ExecFloat.Binary.CorrectlyRounded.logPlus1, ``ConfiguredBinary.log1p_of_isIEEE),
    (``ExecFloat.Binary.CorrectlyRounded.expMul, ``ConfiguredBinary.expMul_of_isIEEE)]
  exactOperations := #[(``toReal_natCast_one, none)]
  roundings := #[(``Nat.cast, ``natCast_rounding)]
  procedures := #[(``observeLength, true, .inl observeLength),
    (``reduceDivNatCount, true, .inr reduceDivNatCount),
    (``unfoldExtremum, true, .inr unfoldExtremum),
    (``linearizeLiteral, true, .inl linearizeLiteral)]
  restate := #[splitRead?]
  symbolic := #[hasSymbolicShape]
  skeleton := #[tensorSkeleton]
  readRules := #[reads.getState]

attribute [roundoff_extension] tensors

end TorchLean.Roundoff
