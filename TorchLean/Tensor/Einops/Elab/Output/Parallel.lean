/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Native.Parallel
public import TorchLean.Tensor.Einops.Elab.Einsum.Index
public meta import TorchLean.Tensor.Einops.Elab.Output.Index -- shake: keep
public import TorchLean.Tensor.Einops.Elab

/-!
# Parallel lowering for generated einsum outputs

For large concrete einsums, `compileParallelEinsumOutput?` splits the output
traversal into Lean tasks and proves the assembled buffer equal to the
reference array, as the sequential path does. Each task receives a contiguous
range of the outer output axis and runs the arbitrary-rank output compiler on
it. Scalar contractions are left as they are, so their reduction order is
preserved.
-/

public meta section

namespace TorchLean.Einops.Elab

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta
open Native (CertifiedFlatPart OrderedFlatPartition)

/--
Choose an arbitrary positive output-task count from static work.

The estimate counts source contraction terms because it is available for every
scalar type and lowering strategy. Measurements show that task overhead still
dominates a 512-by-512 contraction, while four output chunks substantially
improve a 1024-by-1024 contraction. Once parallel execution is worthwhile, the
count grows logarithmically with work and is bounded by the outer axis, so
every generated chunk is nonempty.

The initial parallel width is a scheduling policy, not a correctness boundary:
the executor and its proof accept every list length.
-/
def einsumOutputTaskCount
    (outputShape : Shape)
    (contractionEntries : Nat) : Nat :=
  match outputShape with
  | [] => 1
  | outer :: _ =>
      let threshold := 1_000_000
      let work := Shape.size outputShape * contractionEntries
      if decide (threshold ≤ work) then
        min outer (4 + Nat.log2 (work / threshold))
      else
        1

/--
Compile any positive number of balanced contiguous ranges of a concrete output
shape.

`buildOutputLoops` is the ordinary arbitrary-rank output compiler.
`compileNativeOutputFold` replaces a finite outer fold with its certified
native-counter implementation when the chunk length is portable. The result is
`none` when shapes are symbolic or the static work policy rejects task launch.
-/
def compileParallelEinsumOutput?
    (scalarType storage bufferType reference : Expr)
    (outputLengths : List Expr) (contractionEntries? : Option Nat)
    (buildOutputLoops :
      List Expr → List Expr → Expr →
        TermElabM (Expr × Expr × Expr))
    (compileNativeOutputFold :
      Expr → Expr → Expr → Expr → Expr → Expr → Name →
        TermElabM (Expr × Expr)) :
    TermElabM (Option (Expr × Expr)) := do
  let scalarLevel ← getDecLevel scalarType
  let concreteOutputShape? ← concreteNatExpressions? outputLengths
  match outputLengths, concreteOutputShape?, contractionEntries? with
  | outerLength :: innerLengths, some (outer :: innerShape),
      some contractionEntries =>
      let taskCount :=
        einsumOutputTaskCount (outer :: innerShape)
          contractionEntries
      unless 1 < taskCount do
        return none
      let innerSize := Shape.size innerShape
      let outputSize := Shape.size (outer :: innerShape)
      let compileChunk
          (generatedOuterStep referenceOuterStep hOuterStep : Expr)
          (capacity start count : Nat) :
          TermElabM (Expr × Expr × Expr) := do
        let startExpr := mkNatLit start
        let countExpr := mkNatLit count
        let innerShapeExpr ← shapeExpr innerLengths
        let capacityExpr := mkNatLit capacity
        let rangeEnd ← mkAppM ``Nat.add #[startExpr, countExpr]
        let hRange ←
          withTransparency .all <|
            mkExpectedTypeHint
              (← mkDecideProof <|
                ← mkLE (mkNatLit (start + count)) (mkNatLit outer))
              (← mkLE rangeEnd outerLength)
        let emptyChunk :=
          mkAppN (mkConst ``Storage.emptyWithCapacity [scalarLevel]) #[
            scalarType, storage, capacityExpr]
        let localCoordinateType ← mkAppM ``Fin #[countExpr]
        let (rawChunk, _referenceChunk, hRawChunkReference) ←
          withLocalDeclD `output bufferType fun loopOutput =>
            withLocalDeclD `localOuterCoordinate localCoordinateType
                fun localCoordinate => do
              let localCoordinateAtStart ←
                mkAppM ``Fin.natAdd #[startExpr, localCoordinate]
              let globalCoordinate ←
                mkAppM ``Fin.castLE #[hRange, localCoordinateAtStart]
              let generatedBody :=
                mkAppN generatedOuterStep #[loopOutput, globalCoordinate]
              let referenceBody :=
                mkAppN referenceOuterStep #[loopOutput, globalCoordinate]
              let hAtOutput ←
                mkAppM ``congrFun #[hOuterStep, loopOutput]
              let hBody ←
                mkAppM ``congrFun #[hAtOutput, globalCoordinate]
              let generatedStep ←
                mkLambdaFVars #[loopOutput, localCoordinate] generatedBody
              let referenceStep ←
                mkLambdaFVars #[loopOutput, localCoordinate] referenceBody
              let hCoordinateFunction ←
                mkLambdaFVars #[localCoordinate] hBody
              let hForOutput ←
                mkAppM ``funext #[hCoordinateFunction]
              let hOutputFunction ←
                mkLambdaFVars #[loopOutput] hForOutput
              let hStep ← mkAppM ``funext #[hOutputFunction]
              let generatedFinLoop ←
                mkAppM ``Fin.foldl #[
                  countExpr, generatedStep, emptyChunk]
              let referenceLoop ←
                mkAppM ``Fin.foldl #[
                  countExpr, referenceStep, emptyChunk]
              let hFinLoops ←
                withLocalDeclD `step (← inferType generatedStep) fun step => do
                  let fold ←
                    mkAppM ``Fin.foldl #[countExpr, step, emptyChunk]
                  let preserveStep ← mkLambdaFVars #[step] fold
                  mkAppM ``congrArg #[preserveStep, hStep]
              match ← nativeLoopBound? countExpr with
              | none =>
                  pure (generatedFinLoop, referenceLoop, hFinLoops)
              | some (nativeBound, hBound) => do
                  let (nativeLoop, hNativeFin) ←
                    compileNativeOutputFold
                      countExpr nativeBound hBound generatedStep
                      emptyChunk generatedFinLoop
                      `nativeParallelOutputChunk
                  let hNativeReference ←
                    mkAppM ``Eq.trans #[hNativeFin, hFinLoops]
                  pure (nativeLoop, referenceLoop, hNativeReference)
        let hCoordinateChunk ←
          mkAppM ``Shape.Coord.toArray_foldl_push_outerRange #[
            outerLength, startExpr, countExpr, innerShapeExpr, capacityExpr,
            hRange, reference]
        let hCoordinateChunkType ←
          withTransparency .reducible <| whnf (← inferType hCoordinateChunk)
        let some (_, observedCoordinateFold, _) := hCoordinateChunkType.eq?
          | throwError
              "internal error: the parallel output theorem did not produce \
                an equality"
        let observedCoordinateFold := observedCoordinateFold.consumeMData
        unless observedCoordinateFold.isAppOfArity
            ``Storage.toArray 3 do
          throwError
            "internal error: the parallel output theorem did not expose a \
              physical buffer"
        let coordinateFold := observedCoordinateFold.getAppArgs[2]!
        let hChunkCoordinateFold ←
          withTransparency .all <|
            mkExpectedTypeHint hRawChunkReference
              (← mkEq rawChunk coordinateFold)
        let toArrayFunction ←
          withLocalDeclD `buffer bufferType fun buffer => do
            let observed :=
              mkAppN (mkConst ``Storage.toArray [scalarLevel]) #[
                scalarType, storage, buffer]
            mkLambdaFVars #[buffer] observed
        let hObservedChunkCoordinateFold ←
          mkAppM ``congrArg #[
            toArrayFunction, hChunkCoordinateFold]
        let hChunk ←
          mkAppM ``Eq.trans #[
            hObservedChunkCoordinateFold, hCoordinateChunk]
        let flatStart := start * innerSize
        let flatLength := count * innerSize
        let flatStartExpr := mkNatLit flatStart
        let flatLengthExpr := mkNatLit flatLength
        let flatEnd ← mkAppM ``Nat.add #[flatStartExpr, flatLengthExpr]
        let hFlatRange ←
          withTransparency .all <|
            mkExpectedTypeHint
              (← mkDecideProof <|
                ← mkLE (mkNatLit (flatStart + flatLength))
                  (mkNatLit outputSize))
              (← mkLE flatEnd (mkNatLit outputSize))
        let producer ←
          withLocalDeclD `unused (mkConst ``Unit) fun unused =>
            mkLambdaFVars #[unused] rawChunk
        let certifiedPart ←
          mkAppM ``CertifiedFlatPart.mk #[producer, hChunk]
        -- Keeping each task body behind an auxiliary definition prevents the final
        -- partition certificate from embedding every native output loop recursively.
        let certifiedPart ← sealAuxDefinition `_einops_parallel_chunk certifiedPart
        let part ←
          mkAppM ``CertifiedFlatPart.produce #[certifiedPart]
        let certificate ←
          mkAppM ``CertifiedFlatPart.toArray_produce #[certifiedPart]
        pure (part, hFlatRange, certificate)
      let outerCoordinateType ← mkAppM ``Fin #[outerLength]
      let (generatedOuterStep, referenceOuterStep, hOuterStep) ←
        withLocalDeclD `output bufferType fun loopOutput =>
          withLocalDeclD `outerCoordinate outerCoordinateType
              fun outerCoordinate => do
            let (generatedBody, referenceBody, hBody) ←
              buildOutputLoops innerLengths [outerCoordinate] loopOutput
            let generatedStep ←
              mkLambdaFVars #[loopOutput, outerCoordinate] generatedBody
            let referenceStep ←
              mkLambdaFVars #[loopOutput, outerCoordinate] referenceBody
            let hCoordinateFunction ←
              mkLambdaFVars #[outerCoordinate] hBody
            let hForOutput ← mkAppM ``funext #[hCoordinateFunction]
            let hOutputFunction ←
              mkLambdaFVars #[loopOutput] hForOutput
            let hStep ← mkAppM ``funext #[hOutputFunction]
            pure (generatedStep, referenceStep, hStep)
      let hOuterStepType ← inferType hOuterStep
      let hOuterStep ← sealCertificate hOuterStepType hOuterStep
      let base := outer / taskCount
      let remainder := outer % taskCount
      let extra (chunk : Nat) : Nat :=
        if chunk < remainder then 1 else 0
      let (buffer, hOutputArray) ←
        withGeneratedLetPair
            [(`parallelOutputStep, generatedOuterStep)] fun values => do
          let parallelOutputStep := values[0]!
          let hParallelOutputStep ←
            withTransparency .all <|
              mkExpectedTypeHint hOuterStep
                (← mkEq parallelOutputStep referenceOuterStep)
          let mut parts : Array Expr := #[]
          let mut rangeProofs : Array Expr := #[]
          let mut certificates : Array Expr := #[]
          let mut start := 0
          for chunkIndex in [:taskCount] do
            let count := base + extra chunkIndex
            let capacity :=
              if chunkIndex == 0 then outputSize else count * innerSize
            let (part, hRange, certificate) ←
              compileChunk parallelOutputStep referenceOuterStep
                hParallelOutputStep capacity start count
            parts := parts.push part
            rangeProofs := rangeProofs.push hRange
            certificates := certificates.push certificate
            start := start + count
          let partType ← mkArrow (mkConst ``Unit) bufferType
          let partsList ← mkListLit partType parts.toList
          let buffer :=
            mkAppN (mkConst ``Native.parallelBuffer [scalarLevel]) #[
              scalarType, storage, partsList]
          let total := mkNatLit outputSize
          let mut partition ←
            mkAppOptM ``OrderedFlatPartition.done #[
              some scalarType, some storage,
              some total, some reference]
          for chunkIndex in (List.range taskCount).reverse do
            partition ←
              mkAppM ``OrderedFlatPartition.next #[
                parts[chunkIndex]!, rangeProofs[chunkIndex]!,
                certificates[chunkIndex]!, partition]
            let partitionType ← inferType partition
            partition ← sealCertificate partitionType partition
          let hOutputArray ←
            mkAppM ``Native.toArray_parallelBuffer #[partition]
          pure (buffer, hOutputArray)
      let hOutputArrayType ← inferType hOutputArray
      let hOutputArray ← sealCertificate hOutputArrayType hOutputArray
      return some (buffer, hOutputArray)
  | _, _, _ => return none

end TorchLean.Einops.Elab
