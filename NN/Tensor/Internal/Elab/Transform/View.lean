/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Lowering.TransformFusion
public meta import NN.Tensor.Internal.Elab.Transform.Index
public meta import NN.Tensor.Internal.Elab.Transform.ViewAttribute
public meta import Lean.Meta.Tactic.Rewrite -- shake: keep
public import NN.Tensor.Internal.Elab.Native.Tensor -- shake: keep

/-!
# Certified input views for transform fusion

This module recovers one source tensor and one certified flat-index program
from a visible chain of shape-only tensor operations. Rearrange, repeat,
einsum, and reduction elaborators share this recognizer so every consumer
uses the same coordinate semantics and dependent shape transport.

The metaprogramming result is an ordinary tuple rather than a new public plan
type. Its coordinate map is the independent semantics, its flat map is the
native executable representation, one proof connects those maps, and another
identifies the visible tensor term with the flat pullback from its source.
-/

public meta section

namespace TorchLean.Tensor.Internal.Elab.Impl

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta

/-- All theorems tagged `[einops_view]`, imported ones first and then this module's own. -/
private def certifiedViewTheorems (environment : Environment) : Array Name :=
  let state :=
    ViewRegistry.einopsViewAttribute.ext.toEnvExtension.getState environment
  let imported :=
    state.importedEntries.foldl (init := #[]) fun declarations entries =>
      declarations ++ entries
  imported ++
    (ViewRegistry.einopsViewAttribute.ext.getState environment).toArray

/--
Try one certified equation without retaining failed metavariable assignments.

`MVarId.rewrite` instantiates the theorem against the visible application but
does not preprocess its left-hand side. This is important for opaque wrappers:
the registered function call stays the matching key.
-/
private def rewriteCertifiedView? (candidate : Expr)
    (theoremName : Name) : TermElabM (Option (Expr × Expr)) :=
  observing? do
    let theoremExpression ← mkConstWithFreshMVarLevels theoremName
    let scratchGoal ← mkFreshExprMVar (mkConst ``True)
    let result ← scratchGoal.mvarId!.rewrite candidate theoremExpression
    synthesizeSyntheticMVarsNoPostponing
    let rewritten ← instantiateMVars result.eNew
    let proof ← instantiateMVars result.eqProof
    if rewritten.hasMVar || proof.hasMVar || rewritten == candidate then
      failure
    return (rewritten, proof)

/--
Rewrite only with equations registered as certified tensor views.

The returned proof is an ordinary equality from the visible tensor expression
to the rewritten view. No global simp theorem or implementation unfolding is
used.
-/
private def rewriteCertifiedViews? (candidate : Expr) :
    TermElabM (Option (Expr × Expr)) := do
  let theoremNames := certifiedViewTheorems (← getEnv)
  for theoremName in theoremNames do
    if let some result ← rewriteCertifiedView? candidate theoremName then
      return some result
  return none

/-- Recover the physical input and output shapes stored in a checked transform. -/
def checkedTransformShapes (checked : Expr) :
    MetaM (Expr × Expr) := do
  let value ← mkAppM ``Check.CheckedTransform.value #[checked]
  let normalized ← mkAppM ``Check.TransformPlan.normalized #[value]
  let inputShape ←
    mkAppM ``Check.NormalizedTransform.input #[normalized]
  let outputShape ←
    mkAppM ``Check.TransformPlan.output #[value]
  return (inputShape, outputShape)

/-- Recover the input-axis subset proof certified by a rearrangement plan. -/
def rearrangeAxesProof (checked hKind : Expr) : MetaM Expr := do
  let valid ← mkAppM ``Check.CheckedTransform.valid #[checked]
  let normalizedValid ←
    mkAppM ``Check.TransformPlan.Valid.normalization #[valid]
  mkAppM
    ``Check.NormalizedTransform.Valid.input_axes_subset_output_of_rearrange #[
      normalizedValid, hKind]

/-- Recover the input-axis subset proof certified by a repeat plan. -/
def repeatAxesProof (checked hKind : Expr) : MetaM Expr := do
  let valid ← mkAppM ``Check.CheckedTransform.valid #[checked]
  let normalizedValid ←
    mkAppM ``Check.TransformPlan.Valid.normalization #[valid]
  mkAppM
    ``Check.NormalizedTransform.Valid.input_axes_subset_output_of_repeat #[
      normalizedValid, hKind]

/--
Build the checked coordinate projection, its compact flat representation, and
the theorem connecting them.
-/
def checkedProjection (checked hAxes : Expr) :
    MetaM (Expr × Expr × Expr) := do
  let coordinateMap ←
    mkAppM ``Check.CheckedTransform.inputCoordinateOfOutput #[
      checked, hAxes]
  let (flatMap, hFlatMap) ← checkedFlatProjection checked hAxes
  return (coordinateMap, flatMap, hFlatMap)

/--
Represent an arbitrary coordinate map by direct row-major linearization.

Checked transformations use `checkedProjection` instead. This fallback keeps
fusion general for user-defined `Rep.pull` and `Rep.reindex` inputs.
-/
def directFlatProjection (inputShape : Expr)
    (coordinateMap : Expr) : MetaM (Expr × Expr) := do
  let inputSize ← mkAppM ``Shape.size #[inputShape]
  let inputIndexType ← mkAppM ``Fin #[inputSize]
  let flatMap ←
    withLocalDeclD `inputIndex inputIndexType fun inputIndex => do
      let inputCoordinate ← mkAppM ``Coord.unlinearize #[inputIndex]
      let sourceCoordinate := mkApp coordinateMap inputCoordinate
      let sourceIndex ← mkAppM ``Coord.linearize #[sourceCoordinate]
      mkLambdaFVars #[inputIndex] sourceIndex
  let hFlatMap ←
    withLocalDeclD `inputIndex inputIndexType fun inputIndex => do
      let inputCoordinate ← mkAppM ``Coord.unlinearize #[inputIndex]
      let sourceCoordinate := mkApp coordinateMap inputCoordinate
      let sourceIndex ← mkAppM ``Coord.linearize #[sourceCoordinate]
      let flatValue := mkApp flatMap inputIndex
      let equality ← mkEq flatValue sourceIndex
      let proof ←
        withTransparency .reducible <|
          mkExpectedTypeHint (← mkEqRefl sourceIndex) equality
      mkLambdaFVars #[inputIndex] proof
  return (flatMap, hFlatMap)

/-- Compose coordinate maps, flat maps, and their correctness certificates. -/
def composeCertifiedProjections
    (outerMap outerFlatMap hOuterMap innerMap innerFlatMap hInnerMap : Expr) :
    MetaM (Expr × Expr × Expr) := do
  let coordinateMap ← mkAppM ``Function.comp #[outerMap, innerMap]
  let flatMap ← mkAppM ``Function.comp #[outerFlatMap, innerFlatMap]
  let hFlatMap ←
    mkAppM ``Lowering.flatMap_comp_correct #[
      outerMap, outerFlatMap, hOuterMap,
      innerMap, innerFlatMap, hInnerMap]
  return (coordinateMap, flatMap, hFlatMap)

/--
Recover one composed output-to-source map from a leading shape-only pipeline.

The recognized source may be an ordinary or already fused rearrangement or
repeat, or a direct `Rep.reindex` or `Rep.pull`. Generated lets are
preserved without duplicating their checked plans. The result includes
equalities for both the visible tensor and the compact logical reference used
by downstream denotational semantics.
-/
partial def fusedTransformInput?
    (inputShape tensor : Expr) :
    TermElabM
      (Option (Expr × Expr × Expr × Expr × Expr × Expr × Expr)) := do
  let rec
    /--
    Open generated lets while recovering synchronized coordinate and flat
    programs for the preceding transformation chain.
    -/
    visit (candidate : Expr) :
        TermElabM
          (Option
            (Expr × Expr × Expr × Expr × Expr × Expr × Expr × Expr)) := do
      match candidate with
      | .letE name type assignment body _ =>
          if type.consumeMData.isConstOf ``Check.CheckedTransform then
            visit (body.instantiate1 assignment)
          else
            withLetDecl name type assignment fun localValue => do
              let some
                  (outputShape, coordinateMap, flatMap, hFlatMap, sourceTensor,
                    hTensor, logicalTensor, hLogicalTensor) ←
                  visit (body.instantiate1 localValue)
                | return none
              let coordinateMap ←
                mkLetFVars (generalizeNondepLet := false) #[localValue]
                  coordinateMap
              let flatMap ←
                mkLetFVars (generalizeNondepLet := false) #[localValue]
                  flatMap
              let hFlatMap ←
                mkLetFVars (generalizeNondepLet := false) #[localValue]
                  hFlatMap
              let sourceTensor ←
                mkLetFVars (generalizeNondepLet := false) #[localValue]
                  sourceTensor
              let hTensor ←
                mkLetFVars (generalizeNondepLet := false) #[localValue]
                  hTensor
              let logicalTensor ←
                mkLetFVars (generalizeNondepLet := false) #[localValue]
                  logicalTensor
              let hLogicalTensor ←
                mkLetFVars (generalizeNondepLet := false) #[localValue]
                  hLogicalTensor
              return some
                (outputShape, coordinateMap, flatMap, hFlatMap, sourceTensor,
                  hTensor, logicalTensor, hLogicalTensor)
      | _ => do
          let candidate := candidate.consumeMData
          if let some (rewritten, hRewritten) ←
              rewriteCertifiedViews? candidate then
            let rewritten ←
              withTransparency .reducible <| whnf rewritten
            let some
                (outputShape, coordinateMap, flatMap, hFlatMap, sourceTensor,
                  hRewrittenTensor, logicalTensor, hLogicalTensor) ←
                  visit rewritten
              | return none
            let hTensor ←
              mkAppM ``Eq.trans #[hRewritten, hRewrittenTensor]
            return some
              (outputShape, coordinateMap, flatMap, hFlatMap, sourceTensor,
                hTensor, logicalTensor, hLogicalTensor)
          if candidate.isAppOfArity ``nativeTensorKernel 8 then
            let arguments := candidate.getAppArgs
            let compiled := arguments[4]!
            let some
                (outputShape, coordinateMap, flatMap, hFlatMap, sourceTensor,
                  hCompiledTensor, _, _) ←
                visit compiled
              | return none
            let hNativeCompiled ←
              mkAppM ``nativeTensorKernel_eq_compiled #[
                arguments[3]!, compiled, arguments[5]!, arguments[6]!,
                arguments[7]!]
            let hTensor ←
              mkAppM ``Eq.trans #[hNativeCompiled, hCompiledTensor]
            let hReferenceCompiled ←
              mkAppM ``Eq.symm #[arguments[6]!]
            let hLogicalTensor ←
              mkAppM ``Eq.trans #[hReferenceCompiled, hCompiledTensor]
            return some
              (outputShape, coordinateMap, flatMap, hFlatMap, sourceTensor,
                hTensor, arguments[3]!, hLogicalTensor)
          let mut previousOutputShape? : Option Expr := none
          let mut previousMap? : Option Expr := none
          let mut previousFlatMap? : Option Expr := none
          let mut hPreviousMap? : Option Expr := none
          let mut sourceTensor? : Option Expr := none
          let mut hPreviousTensor? : Option Expr := none
          if candidate.isAppOfArity ``Lowering.rearrangeTensor 5 then
            let arguments := candidate.getAppArgs
            let previousChecked := arguments[2]!
            let previousKind := arguments[3]!
            let (_, previousOutputShape) ←
              checkedTransformShapes previousChecked
            let hAxes ←
              rearrangeAxesProof previousChecked previousKind
            let (previousMap, previousFlatMap, hPreviousMap) ←
              checkedProjection previousChecked hAxes
            previousOutputShape? := some previousOutputShape
            previousMap? := some previousMap
            previousFlatMap? := some previousFlatMap
            hPreviousMap? := some hPreviousMap
            sourceTensor? := some arguments[4]!
            hPreviousTensor? := some <|
              ← mkAppM ``Lowering.rearrangeTensor_eq_pullFlat_of_correct #[
                previousChecked, previousKind, previousFlatMap,
                hPreviousMap, arguments[4]!]
          else if candidate.isAppOfArity ``Lowering.repeatTensor 5 then
            let arguments := candidate.getAppArgs
            let previousChecked := arguments[2]!
            let previousKind := arguments[3]!
            let (_, previousOutputShape) ←
              checkedTransformShapes previousChecked
            let hAxes ← repeatAxesProof previousChecked previousKind
            let (previousMap, previousFlatMap, hPreviousMap) ←
              checkedProjection previousChecked hAxes
            previousOutputShape? := some previousOutputShape
            previousMap? := some previousMap
            previousFlatMap? := some previousFlatMap
            hPreviousMap? := some hPreviousMap
            sourceTensor? := some arguments[4]!
            hPreviousTensor? := some <|
              ← mkAppM ``Lowering.repeatTensor_eq_pullFlat_of_correct #[
                previousChecked, previousKind, previousFlatMap,
                hPreviousMap, arguments[4]!]
          else if candidate.isAppOfArity
              ``Lowering.transformTensorFused 9 then
            let arguments := candidate.getAppArgs
            let previousChecked := arguments[3]!
            let hAxes := arguments[4]!
            let (_, previousOutputShape) ←
              checkedTransformShapes previousChecked
            let (checkedMap, checkedFlatMap, hCheckedMap) ←
              checkedProjection previousChecked hAxes
            let (previousMap, previousFlatMap, hPreviousMap) ←
              composeCertifiedProjections
                arguments[5]! arguments[6]! arguments[7]!
                checkedMap checkedFlatMap hCheckedMap
            previousOutputShape? := some previousOutputShape
            previousMap? := some previousMap
            previousFlatMap? := some previousFlatMap
            hPreviousMap? := some hPreviousMap
            sourceTensor? := some arguments[8]!
            let target ←
              mkAppM ``Rep.pullFlat #[previousFlatMap, arguments[8]!]
            hPreviousTensor? := some <|
              ← withTransparency .all <|
                mkExpectedTypeHint (← mkEqRefl candidate)
                  (← mkEq candidate target)
          else if candidate.isAppOfArity ``Rep.pullFlat 6 then
            let arguments := candidate.getAppArgs
            let flatMap := arguments[4]!
            let coordinateMap ←
              mkAppM ``Lowering.coordinateMapOfFlatMap #[flatMap]
            let hFlatMap ←
              mkAppM ``Lowering.flatMap_coordinateMapOfFlatMap #[flatMap]
            previousOutputShape? := some arguments[3]!
            previousMap? := some coordinateMap
            previousFlatMap? := some flatMap
            hPreviousMap? := some hFlatMap
            sourceTensor? := some arguments[5]!
            hPreviousTensor? := some (← mkEqRefl candidate)
          else if candidate.isAppOfArity ``Rep.reindex 6 then
            let arguments := candidate.getAppArgs
            let coordinateMap ← mkAppM ``Equiv.toFun #[arguments[4]!]
            let (flatMap, hFlatMap) ←
              directFlatProjection arguments[3]! coordinateMap
            previousOutputShape? := some arguments[3]!
            previousMap? := some coordinateMap
            previousFlatMap? := some flatMap
            hPreviousMap? := some hFlatMap
            sourceTensor? := some arguments[5]!
            let hPull ←
              mkAppM ``Rep.pullFlat_eq_pull #[
                flatMap, coordinateMap, hFlatMap, arguments[5]!]
            hPreviousTensor? := some <| ← mkAppM ``Eq.symm #[hPull]
          else if candidate.isAppOfArity ``Rep.pull 6 then
            let arguments := candidate.getAppArgs
            let (flatMap, hFlatMap) ←
              directFlatProjection arguments[3]! arguments[4]!
            previousOutputShape? := some arguments[3]!
            previousMap? := some arguments[4]!
            previousFlatMap? := some flatMap
            hPreviousMap? := some hFlatMap
            sourceTensor? := some arguments[5]!
            let hPull ←
              mkAppM ``Rep.pullFlat_eq_pull #[
                flatMap, arguments[4]!, hFlatMap, arguments[5]!]
            hPreviousTensor? := some <| ← mkAppM ``Eq.symm #[hPull]
          let some previousOutputShape := previousOutputShape?
            | return none
          let some previousMap := previousMap?
            | return none
          let some previousFlatMap := previousFlatMap?
            | return none
          let some hPreviousMap := hPreviousMap?
            | return none
          let some sourceTensor := sourceTensor?
            | return none
          let some hPreviousTensor := hPreviousTensor?
            | return none
          let sourceTensorType ← whnf (← inferType sourceTensor)
          let sourceTensorType := sourceTensorType.consumeMData
          unless sourceTensorType.isAppOfArity ``Rep 3 do
            return none
          let sourceShape := sourceTensorType.getAppArgs[1]!
          let inputCoordinateType ← mkAppM ``Coord #[inputShape]
          let sourceCoordinateType ← mkAppM ``Coord #[sourceShape]
          let expectedMapType ←
            mkArrow inputCoordinateType sourceCoordinateType
          let previousMapType ← inferType previousMap
          if ← withTransparency .all <|
              isDefEq previousMapType expectedMapType then
            let target ←
              mkAppM ``Rep.pullFlat #[previousFlatMap, sourceTensor]
            let hTensor ←
              withTransparency .all <|
                mkExpectedTypeHint hPreviousTensor
                  (← mkEq candidate target)
            return some
              (previousOutputShape, previousMap, previousFlatMap,
                hPreviousMap, sourceTensor, hTensor, candidate, hTensor)
          let hMiddleShape ←
            withTransparency .reducible <|
              mkExpectedTypeHint (← mkEqRefl previousOutputShape)
                (← mkEq previousOutputShape inputShape)
          let hMiddleShapeSymm ←
            mkAppM ``Eq.symm #[hMiddleShape]
          let hCoordinateShape ←
            mkAppM ``congrArg #[mkConst ``Coord, hMiddleShapeSymm]
          let middleCast ← mkAppM ``Equiv.cast #[hCoordinateShape]
          let middleCast ← mkAppM ``Equiv.toFun #[middleCast]
          let inputMap ←
            mkAppM ``Function.comp #[previousMap, middleCast]
          let hSize ←
            mkAppM ``congrArg #[mkConst ``Shape.size, hMiddleShapeSymm]
          let flatCast ← mkAppM ``finCongr #[hSize]
          let flatCast ← mkAppM ``Equiv.toFun #[flatCast]
          let inputFlatMap ←
            mkAppM ``Function.comp #[previousFlatMap, flatCast]
          let hInputMap ←
            mkAppM ``Lowering.flatMap_comp_cast_correct #[
              hMiddleShape, previousMap, previousFlatMap, hPreviousMap]
          let target ←
            mkAppM ``Rep.pullFlat #[inputFlatMap, sourceTensor]
          let hTensor ←
            withTransparency .all <|
              mkExpectedTypeHint hPreviousTensor
                (← mkEq candidate target)
          return some
            (previousOutputShape, inputMap, inputFlatMap, hInputMap,
              sourceTensor, hTensor, candidate, hTensor)
  let some
      (_, inputMap, inputFlatMap, hInputMap, sourceTensor, hTensor,
        logicalTensor, hLogicalTensor) ←
      visit tensor
    | return none
  return some
    (inputMap, inputFlatMap, hInputMap, sourceTensor, hTensor,
      logicalTensor, hLogicalTensor)

end TorchLean.Tensor.Internal.Elab.Impl
