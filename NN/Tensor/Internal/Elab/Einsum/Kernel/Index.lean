/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Elab.Einsum.Index
public import NN.Tensor.Internal.Elab.Einsum.Symbolic
public meta import NN.Tensor.Internal.Elab.Einsum.Contraction.Loop -- shake: keep

/-!
# Certified einsum operand indexing

This module generates row-major operand indices, hoists reusable coordinate
contributions, and certifies direct tensor reads against the checked einsum
index plan.
-/

public meta section

namespace TorchLean.Tensor.Internal.Elab.Impl

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta

/--
Hoist multi-term output-coordinate stride products out of contraction loops.

Lean's native compiler does not reliably perform loop-invariant motion for
boxed natural-number arithmetic. Precomputing a multi-term base therefore
removes several products and additions from every contracted iteration.
Passing a one-term base through the generated loop is measurably slower than
recomputing that product, so singleton dimensions and strides that reduce to
zero or one are ignored and at least two remaining terms are required.
-/
def shouldHoistOutputIndexTerms
    (inputAxes : List Check.EinsumAxis)
    (inputDimensions : List Expr)
    (outputAxes : List Check.EinsumAxis) : MetaM Bool := do
  let rec
    /-- Count nontrivial output-coordinate stride terms that can be shared across the fold. -/
    countTerms
      (remainingAxes : List Check.EinsumAxis)
      (remainingDimensions : List Expr) : MetaM Nat := do
    match remainingAxes, remainingDimensions with
    | [], [] => pure 0
    | axis :: remainingAxes, dimension :: remainingDimensions => do
        let remainingCount ←
          countTerms remainingAxes remainingDimensions
        if !outputAxes.contains axis then
          pure remainingCount
        else
          match ← getNatValue? dimension with
          | some 1 => pure remainingCount
          | _ =>
              let stride ← runtimeShapeSizeExpr remainingDimensions
              match ← getNatValue? stride with
              | some 0 | some 1 => pure remainingCount
              | _ => pure (remainingCount + 1)
    | _, _ =>
        throwError
          "internal error: einsum axes and dimensions have different lengths"
  return decide (2 ≤ (← countTerms inputAxes inputDimensions))

/--
Add one physical-axis contribution to a row-major input index.

A literal singleton contributes nothing. When the physical length is
definitionally the logical axis length, the coordinate already belongs to
that same `Fin` type and is necessarily zero in the singleton case, so no
runtime broadcast branch is needed. Only genuinely ambiguous symbolic
broadcasts retain the equality test.
-/
def addInputIndexContribution
    (physicalLength logicalLength base contribution : Expr) :
    MetaM Expr := do
  match ← getNatValue? physicalLength with
  | some 1 => pure base
  | some _ => natAddExpr base contribution
  | none =>
      if ← withTransparency .reducible <|
          isDefEq physicalLength logicalLength then
        natAddExpr base contribution
      else
        let nonsingletonIndex ← natAddExpr base contribution
        let isSingleton ← mkEq physicalLength (mkNatLit 1)
        mkAppM ``ite #[isSingleton, base, nonsingletonIndex]

/--
Combine one operand's loop-invariant output contributions in row-major index
order.

Unknown physical dimensions retain the singleton-broadcast guard outside the
contraction loop. The generated index certificate treats natural-number
addition as associative and commutative, so output contributions can be
regrouped without changing the verified flat index.
-/
def hoistedOutputIndexBase
    (inputDimensions logicalDimensions : List Expr)
    (outputCoordinates : List (Option Expr)) : MetaM Expr := do
  match inputDimensions, logicalDimensions, outputCoordinates with
  | [], [], [] => pure (mkNatLit 0)
  | dimension :: inputDimensions, logicalDimension :: logicalDimensions,
      outputCoordinate? :: outputCoordinates => do
      let remainingBase ←
        hoistedOutputIndexBase inputDimensions logicalDimensions
          outputCoordinates
      let some outputCoordinate := outputCoordinate?
        | return remainingBase
      let outputTerm ←
        natMulExpr (← runtimeShapeSizeExpr inputDimensions) outputCoordinate
      addInputIndexContribution dimension logicalDimension
        remainingBase outputTerm
  | _, _, _ =>
      throwError "internal error: einsum physical dimensions, logical \
        dimensions, and output coordinates have different lengths"

/--
Collect one contracted logical axis's physical row-major contribution to an
operand index.

Repeated labels contribute once per physical occurrence. Symbolic singleton
dimensions retain the runtime broadcast guard required by the verified plan.
-/
def contractedAxisIndexContribution
    (contractedAxis : Check.EinsumAxis)
    (contractedCoordinate : Expr)
    (inputAxes : List Check.EinsumAxis)
    (inputDimensions logicalDimensions : List Expr) : MetaM Expr := do
  match inputAxes, inputDimensions, logicalDimensions with
  | [], [], [] => pure (mkNatLit 0)
  | axis :: inputAxes, dimension :: inputDimensions,
      logicalDimension :: logicalDimensions => do
      let remainingContribution ←
        contractedAxisIndexContribution contractedAxis contractedCoordinate
          inputAxes inputDimensions logicalDimensions
      if axis != contractedAxis then
        return remainingContribution
      let contribution ←
        natMulExpr
          (← runtimeShapeSizeExpr inputDimensions)
          contractedCoordinate
      addInputIndexContribution dimension logicalDimension
        remainingContribution contribution
  | _, _, _ =>
      throwError "internal error: einsum axes, physical dimensions, and \
        logical dimensions have different lengths"

/--
Generate one operand's physical row-major index while retaining the exact
addition order of the generic verified plan.

For operands selected by `shouldHoistOutputIndexTerms`, all loop-invariant
output contributions arrive as one precomputed base. Contracted axes already
included in a staged base are omitted from the remaining dynamic index.
-/
def compileInputFlatIndexValue
    (inputAxes : List Check.EinsumAxis)
    (inputDimensions logicalDimensions : List Expr)
    (outputCoordinates : List (Option Expr))
    (hoistedOutputBase : Option Expr)
    (hoistedContractedAxes : List Check.EinsumAxis)
    (contractionCoordinates : List (Check.EinsumAxis × Expr)) :
    MetaM Expr := do
  let rec
    /-- Accumulate the dynamic row-major contribution of every unhoisted physical axis. -/
    visit
      (remainingAxes : List Check.EinsumAxis)
      (remainingDimensions : List Expr)
      (remainingLogicalDimensions : List Expr)
      (remainingOutputCoordinates : List (Option Expr)) : MetaM Expr := do
    match remainingAxes, remainingDimensions, remainingLogicalDimensions,
        remainingOutputCoordinates with
    | [], [], [], [] => pure (mkNatLit 0)
    | axis :: remainingAxes, dimension :: remainingDimensions,
        logicalDimension :: remainingLogicalDimensions,
        outputCoordinate? :: remainingOutputCoordinates => do
        let remainingIndex ←
          visit remainingAxes remainingDimensions
            remainingLogicalDimensions remainingOutputCoordinates
        match outputCoordinate?, hoistedOutputBase with
        | some _, some _ => pure remainingIndex
        | _, _ =>
            if outputCoordinate?.isNone &&
                hoistedContractedAxes.contains axis then
              return remainingIndex
            let axisOffset ←
              match outputCoordinate? with
              | some outputCoordinate =>
                  natMulExpr
                    (← runtimeShapeSizeExpr remainingDimensions)
                    outputCoordinate
              | none =>
                  let some contractionCoordinate :=
                      einsumAxisExpression? contractionCoordinates axis
                    | throwError
                        "internal error: an einsum input axis is neither \
                          output nor contracted"
                  natMulExpr
                    (← runtimeShapeSizeExpr remainingDimensions)
                    contractionCoordinate
            addInputIndexContribution dimension logicalDimension
              remainingIndex axisOffset
    | _, _, _, _ =>
        throwError "internal error: einsum axes, physical dimensions, \
          logical dimensions, and output coordinates have different lengths"
  let dynamicIndex ←
    visit inputAxes inputDimensions logicalDimensions outputCoordinates
  match hoistedOutputBase with
  | some outputBase => natAddExpr dynamicIndex outputBase
  | none => pure dynamicIndex

/--
Transport an index bound across an equality of its natural-number value.

Generated input views use this helper before composing a logical operand index
with a certified source-index map.
-/
def indexBoundFromValueEquality
    (inputSize hIndexValue rightIndexBound : Expr) : MetaM Expr := do
  let indexBoundPredicate ←
    withLocalDeclD `inputIndex (mkConst ``Nat) fun inputIndex => do
      let bound ← mkLT inputIndex inputSize
      mkLambdaFVars #[inputIndex] bound
  let indexBoundEquality ←
    mkAppM ``congrArg #[indexBoundPredicate, hIndexValue]
  mkAppM ``Eq.mpr #[indexBoundEquality, rightIndexBound]

/--
Lift equality of two bounded index values to equality of their `Fin` terms.
-/
def finIndexEquality
    (inputSize inputIndex certifiedInputIndex hIndexValue : Expr) :
    MetaM Expr :=
  mkAppOptM ``Fin.ext #[
    some inputSize, some inputIndex,
    some certifiedInputIndex, some hIndexValue]

/--
Compile one operand read from a generated row-major index.

Both branches end at the same certified `Fin` index. Concrete buffers use
native word arithmetic and `Array.uget`; symbolic buffers retain ordinary
natural-number arithmetic. The native branch additionally proves that
`USize.toNat` recovers the generated reference index exactly, so wrapping
cannot affect execution.
-/
def compileInputRead
    (tensor inputSize inputIndexValue normalizedInputIndexValue
      certifiedInputIndex hInputIndexValue : Expr)
    (useNativeIndex : Bool)
    (directIndexBound? : Option Expr := none)
    (nativeIndexAssumptions : Array Expr := #[]) :
    TermElabM (Expr × Expr) := do
  let inputIndexType ← mkAppM ``Fin #[inputSize]
  let certifiedInputValue ←
    withTransparency .all <|
      mkAppM ``Rep.getFlat #[tensor, certifiedInputIndex]
  let readEquality
      (inputIndex hIndexValue : Expr) : MetaM Expr := do
    let inputIndexEquality ←
      finIndexEquality inputSize inputIndex certifiedInputIndex hIndexValue
    let readInput ←
      withLocalDeclD `inputIndex inputIndexType fun inputIndex => do
        let value ←
          withTransparency .all <|
            mkAppM ``Rep.getFlat #[tensor, inputIndex]
        mkLambdaFVars #[inputIndex] value
    mkAppM ``congrArg #[readInput, inputIndexEquality]
  let certifiedInputIndexBound ←
    mkAppM ``Fin.isLt #[certifiedInputIndex]
  let inputIndexBound ←
    indexBoundFromValueEquality inputSize hInputIndexValue
      certifiedInputIndexBound
  if useNativeIndex then
    let nativeIndexFits ← mkLT normalizedInputIndexValue inputSize
    let directIndexBound ←
      match directIndexBound? with
      | some directIndexBound =>
          withTransparency .all <|
            mkExpectedTypeHint directIndexBound nativeIndexFits
      | none =>
          let directIndexBound ←
            certifyInputIndexBound normalizedInputIndexValue inputSize
          withTransparency .all <|
            mkExpectedTypeHint directIndexBound nativeIndexFits
    let (nativeIndex, hNativeIndex) ←
      if nativeIndexAssumptions.isEmpty then
        let nativeIndex ← nativeIndexValue normalizedInputIndexValue
        let nativeIndexNat ← mkAppM ``USize.toNat #[nativeIndex]
        let nativeIndexEquality ←
          mkEq nativeIndexNat normalizedInputIndexValue
        let nativeIndexCertificate ←
          certifyNativeIndex (← mkArrow nativeIndexFits nativeIndexEquality)
        pure (nativeIndex, mkApp nativeIndexCertificate directIndexBound)
      else
        let compositionalAssumptions :=
          nativeIndexAssumptions.push directIndexBound
        let certifiedNativeIndex? ←
          certifiedNativeIndexValue? normalizedInputIndexValue
            compositionalAssumptions
        match certifiedNativeIndex? with
        | some result => pure result
        | none => do
          let (nativeIndex, nativeIntermediates) ←
            nativeIndexValueWithIntermediates normalizedInputIndexValue
          let nativeIndexNat ← mkAppM ``USize.toNat #[nativeIndex]
          let nativeIndexEquality ←
            mkEq nativeIndexNat normalizedInputIndexValue
          let mut nativeAssumptions := nativeIndexAssumptions
          let mut intermediateAssumptions :=
            nativeIndexAssumptions.push directIndexBound
          for intermediate in nativeIntermediates do
            let hIntermediate ←
              certifyNativeIntermediateBound intermediate
                intermediateAssumptions
            nativeAssumptions :=
              nativeAssumptions.push hIntermediate
            intermediateAssumptions :=
              intermediateAssumptions.push hIntermediate
          let mut nativeIndexProposition ←
            mkArrow nativeIndexFits nativeIndexEquality
          for assumption in nativeAssumptions.reverse do
            nativeIndexProposition ←
              mkArrow (← inferType assumption) nativeIndexProposition
          let nativeIndexCertificate ←
            certifyNativeIndex nativeIndexProposition
          let mut hNativeIndex := nativeIndexCertificate
          for assumption in nativeAssumptions do
            hNativeIndex := mkApp hNativeIndex assumption
          hNativeIndex := mkApp hNativeIndex directIndexBound
          pure (nativeIndex, hNativeIndex)
    let nativeIndexNat ← mkAppM ``USize.toNat #[nativeIndex]
    let hNativeIndex ←
      withTransparency .all <|
        mkExpectedTypeHint hNativeIndex
          (← mkEq nativeIndexNat normalizedInputIndexValue)
    let hNativeCertified ←
      mkAppM ``Eq.trans #[hNativeIndex, hInputIndexValue]
    let certifiedInputIndexValue ←
      mkAppM ``Fin.val #[certifiedInputIndex]
    let expectedNativeEquality ←
      mkEq nativeIndexNat certifiedInputIndexValue
    let hNativeCertified ←
      withTransparency .all <|
        mkExpectedTypeHint hNativeCertified expectedNativeEquality
    let nativeIndexBound ←
      indexBoundFromValueEquality inputSize hNativeIndex directIndexBound
    let nativeInputIndex ←
      mkAppOptM ``Fin.mk #[
        some inputSize, some nativeIndexNat, some nativeIndexBound]
    let nativeInputValue ←
      mkAppM ``Rep.getFlatUSize #[
        tensor, nativeIndex, nativeIndexBound]
    let finInputValue ←
      withTransparency .all <|
        mkAppM ``Rep.getFlat #[tensor, nativeInputIndex]
    let nativeReadEquality ←
      mkEq nativeInputValue finInputValue
    let hNativeRead ←
      withTransparency .all <|
        mkExpectedTypeHint
          (← mkAppM ``Rep.getFlatUSize_eq_getFlat #[
            tensor, nativeIndex, nativeIndexBound])
          nativeReadEquality
    let hFinRead ←
      readEquality nativeInputIndex hNativeCertified
    let inputValueCorrect ←
      mkAppM ``Eq.trans #[hNativeRead, hFinRead]
    return (nativeInputValue, inputValueCorrect)
  let inputIndex ←
    mkAppOptM ``Fin.mk #[
      some inputSize, some inputIndexValue, some inputIndexBound]
  let inputValue ←
    mkAppM ``Rep.getFlat #[tensor, inputIndex]
  let inputValueCorrect ←
    readEquality inputIndex hInputIndexValue
  let expectedInputValueEquality ←
    mkEq inputValue certifiedInputValue
  let inputValueCorrect ←
    withTransparency .all <|
      mkExpectedTypeHint inputValueCorrect expectedInputValueEquality
  return (inputValue, inputValueCorrect)

end TorchLean.Tensor.Internal.Elab.Impl
