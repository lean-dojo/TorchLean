/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Functional.Einsum

@[expose] public section


namespace Runtime
namespace Autograd
namespace Model

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace F

/-!
# Dynamic einsum

`einsum?` interprets a PyTorch-style subscript string at run time and lowers it to the verified op
set: matmul, reorder, reshape, broadcast, multiply, sum. The dynamic lowering lives in its own
module so users of the typed contractions and label bookkeeping can import `Functional.Einsum`
without elaborating it.
-/

open Einsum

/--
Runtime-checked `einsum` that returns an existential output shape.

Supported:
- multiple inputs, explicit/implicit output, and ellipsis (`...`).
- repeated labels within an operand (diagonal extraction / trace semantics).
- repeated labels in the output (diagonal embedding / zeroing off-diagonal entries).

Currently unsupported (returns `none`):
- non-broadcastable size mismatches.
- any case that would require gather/scatter-style indexing (not in the verifier-friendly op set).

Matrix contractions use batch-broadcast matmul after label normalization. Other equations use
reordering, reshaping, broadcasting, elementwise multiplication, and summing contracted axes.
-/
def einsum? {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    (equation : String)
    (xs : List (Σ s : Shape, RefTy (m := m) (α := α) s)) :
    m (Option (Σ s : Shape, RefTy (m := m) (α := α) s)) := do
  let computation : OptionT m (Σ s : Shape, RefTy (m := m) (α := α) s) := do
    let parsed : Einsum.Parsed ←
      match Einsum.parseEquation equation with
      | .ok p => pure p
      | .error _ => failure
    if parsed.inputs.length != xs.length then
      failure

    -- Count raw labels, before diagonal extraction: implicit `ii` is a scalar trace.
    -- Ellipsis axes precede the alphabetically ordered labels that occur exactly once.
    let implicitOutputLabels (inputs : List (List Label)) (ellipsisRank : Nat) : List Label :=
      let counts := Einsum.labelCounts inputs
      let chars : List Char := (inputs.foldl (fun acc labels => acc ++ labels) []).filterMap fun
        | .chr c => if Einsum.labelCount counts (.chr c) == 1 then some c else none
        | .ell _ => none
      (List.range ellipsisRank).map Label.ell ++
        (chars.mergeSort (fun a b => a.toNat ≤ b.toNat)).map Label.chr

    let shapes0 : List Shape := xs.map Sigma.fst
    let ranks : List Nat := shapes0.map Spec.Shape.rank

    -- Compute max ellipsis length across inputs.
    let mut maxEll : Nat := 0
    for (sub, r) in List.zip parsed.inputs ranks do
      if sub.hasEll then
        let fixed := sub.pre.length + sub.post.length
        if r < fixed then
          failure
        maxEll := Nat.max maxEll (r - fixed)

    -- Expand per-input labels (including ellipsis mapped to `ell k` labels), and apply diagonal
    -- extraction for repeated labels inside an operand (PyTorch semantics).
    let mut processed : List (Σ s : Shape, RefTy (m := m) (α := α) s) := []
    let mut inLabels : List (List Label) := []
    let mut inLabelsRaw : List (List Label) := []

    let rec diagonalizeOperand (fuel : Nat)
        (cur : Σ s : Shape, RefTy (m := m) (α := α) s) (labs : List Label) :
        OptionT m ((Σ s : Shape, RefTy (m := m) (α := α) s) × List Label) := do
      match fuel with
      | 0 =>
          if Einsum.hasDupLabels labs then
            failure
          else
            pure (cur, labs)
      | fuel + 1 =>
          match Einsum.firstDup? labs with
          | none => pure (cur, labs)
          | some (_, p, q) =>
              let curShape := cur.fst
              let dims := Shape.toList curShape
              let some dp := dims[p]? | failure
              let some dq := dims[q]? | failure
              if dp != dq then
                failure
              let maskT : Tensor α curShape := Einsum.diagMaskForShape (α := α) curShape p q
              let mask ← OptionT.lift <| const (m := m) (α := α) (s := curShape) maskT
              let xMasked ← OptionT.lift <| mul (m := m) (α := α) (s := curShape) cur.snd mask
              let r := Spec.Shape.rank curShape
              if hRank : r > 0 then
                if hq : q < r then
                  let perm := Einsum.permMoveAxisToLast r q
                  let swaps : List Nat ←
                    match Einsum.swapDepthsForPerm? perm r with
                    | some ss => pure ss
                    | none => failure
                  let ⟨sPerm, xPerm⟩ ← OptionT.lift <|
                    Einsum.permuteBySwaps (α := α) (m := m) (x := ⟨curShape, xMasked⟩) swaps
                  let rPerm := Spec.Shape.rank sPerm
                  if hRankPerm : rPerm > 0 then
                    if hw : sPerm.wellFormed then
                      let axis : Nat := rPerm - 1
                      let nextRef :=
                        (← OptionT.lift <|
                          (by
                            letI : Shape.WellFormed sPerm := ⟨hw⟩
                            haveI : Shape.HasNonemptyAxis axis sPerm :=
                              Shape.inferNonemptyAxis (by grind)
                            exact reduceSum (m := m) (α := α) (s := sPerm) axis xPerm))
                      let nextShape : Shape := TorchLean.Tensor.shapeAfterSum sPerm axis
                      diagonalizeOperand fuel ⟨nextShape, nextRef⟩ (Einsum.removeAt labs q)
                    else
                      failure
                  else
                    failure
                else
                  failure
              else
                failure

    for (sub, xSigma) in List.zip parsed.inputs xs do
      let s := xSigma.fst
      let x := xSigma.snd
      let labs0 : List Label ←
        match Einsum.expandInputLabels sub s maxEll with
        | .ok v => pure v
        | .error _ => failure
      inLabelsRaw := inLabelsRaw ++ [labs0]
      let (cur', labs') ← diagonalizeOperand labs0.length ⟨s, x⟩ labs0
      processed := processed ++ [cur']
      inLabels := inLabels ++ [labs']

    -- Use diagonalized shapes for the remaining checks/alignments.
    let shapes : List Shape := processed.map Sigma.fst

    -- Determine output labels.
    let allInOrder : List Label :=
      Einsum.orderedUnique (inLabels.foldl (fun acc xs => acc ++ xs) [])
    let outLabelsRaw : List Label ←
      match parsed.output? with
      | some outSub =>
          let labs : List Label ←
            match Einsum.expandOutputLabels outSub maxEll with
            | .ok v => pure v
            | .error _ => failure
          for l in labs do
            if !(allInOrder.contains l) then
              failure
          pure labs
      | none =>
          pure (implicitOutputLabels inLabelsRaw maxEll)

    let outLabels : List Label :=
      -- If explicit output repeats labels (e.g. `i->ii`), we contract w.r.t. unique labels
      -- and then "diag-embed" to the repeated output at the end.
      Einsum.orderedUnique outLabelsRaw

    -- Contracted labels are everything not in the output (in first-appearance order).
    let contracted : List Label :=
      allInOrder.filter (fun l => !(outLabels.contains l))
    let fullLabels : List Label := outLabels ++ contracted

    -- Infer broadcasted label sizes.
    let dimMap ←
      match Einsum.labelDimMap inLabels shapes with
      | .ok mp => pure mp
      | .error _ => failure

    -- Recognize axis roles after ellipsis expansion and output inference. A single shared
    -- contraction axis and two operand-exclusive output axes form a matrix product; all other
    -- output axes are batch axes. Their positions and the number of batch axes are unrestricted.
    let attemptFast : OptionT m (Σ s : Shape, RefTy (m := m) (α := α) s) := do
      let [a, b] := processed | failure
      let [labelsA, labelsB] := inLabels | failure
      let [contract] := contracted | failure
      -- Diagonal extraction/embedding retains the generic lowering's semantics.
      if inLabelsRaw.any Einsum.hasDupLabels || Einsum.hasDupLabels outLabelsRaw then
        failure
      if !(labelsA.contains contract) || !(labelsB.contains contract) then
        failure
      let matrixAxis (other : List Label) (label : Label) : Bool :=
        match label with
        | .chr _ => outLabels.contains label && !(other.contains label)
        | .ell _ => false
      let some row := labelsA.reverse.find? (matrixAxis labelsB) | failure
      let some col := labelsB.reverse.find? (matrixAxis labelsA) | failure
      let batchLabels := outLabels.filter (fun label => label != row && label != col)
      let operandDim (labels : List Label) (shape : Shape) (label : Label) : Option Nat := do
        let index ← labels.findIdx? (· == label)
        (Shape.toList shape)[index]?
      let some mDim := operandDim labelsA a.fst row | failure
      let some nDim := operandDim labelsA a.fst contract | failure
      let some nDimB := operandDim labelsB b.fst contract | failure
      let some pDim := operandDim labelsB b.fst col | failure
      -- Matmul does not broadcast its contracted dimension. Singleton contraction sizes
      -- still use the generic elementwise lowering.
      if nDim != nDimB then failure
      let batchDims (labels : List Label) (shape : Shape) : Option (List Nat) :=
        batchLabels.mapM fun label =>
          if labels.contains label then operandDim labels shape label else some 1
      let some dimsA := batchDims labelsA a.fst | failure
      let some dimsB := batchDims labelsB b.fst | failure
      let some dims := batchLabels.mapM (Einsum.dimFind? dimMap) | failure
      let batchA := Shape.ofList dimsA
      let batchB := Shape.ofList dimsB
      let batch := Shape.ofList dims
      let some ⟨broadcastA⟩ := Shape.canBroadcastTo? batchA batch | failure
      let some ⟨broadcastB⟩ := Shape.canBroadcastTo? batchB batch | failure
      let inputSwaps (labels axes : List Label) : Option (List Nat) := do
        let order := batchLabels.filter (fun label => labels.contains label) ++ axes
        let perm ← order.mapM fun label => labels.findIdx? (· == label)
        Einsum.swapDepthsForPerm? perm labels.length
      let some swapsA := inputSwaps labelsA [row, contract] | failure
      let some swapsB := inputSwaps labelsB [contract, col] | failure
      let some outputPerm :=
        Einsum.permForDuplicateLabels? (batchLabels ++ [row, col]) outLabels | failure
      let some outputSwaps :=
        Einsum.swapDepthsForPerm? outputPerm outLabels.length | failure
      -- Insert only missing batch axes. The matmul primitive owns batch broadcasting and
      -- its adjoint; neither operand is expanded across the other matrix's free axis.
      let align (x : Σ s : Shape, RefTy (m := m) (α := α) s)
          (swaps : List Nat) (target : Shape) :
          OptionT m (RefTy (m := m) (α := α) target) := do
        let ⟨shape, ref⟩ ← OptionT.lift <|
          Einsum.permuteBySwaps (α := α) (m := m) x swaps
        if h : shape = target then
          pure (h ▸ ref)
        else if hSize : shape.size = target.size then
          OptionT.lift <| reshape (m := m) (α := α) ref hSize
        else
          failure
      let a' ← align a swapsA (batchA.concat [mDim, nDim])
      let b' ← align b swapsB (batchB.concat [nDim, pDim])
      let output ← OptionT.lift <|
        letI : Shape.BroadcastTo batchA batch := ⟨broadcastA⟩
        letI : Shape.BroadcastTo batchB batch := ⟨broadcastB⟩
        Runtime.Autograd.Torch.matmul (m := m) (α := α)
          (batchA := batchA) (batchB := batchB) (batch := batch)
          (mDim := mDim) (nDim := nDim) (pDim := pDim) a' b'
      OptionT.lift <| Einsum.permuteBySwaps (α := α) (m := m)
        ⟨batch.concat [mDim, pDim], output⟩ outputSwaps

    let fastRes? ← OptionT.lift attemptFast.run
    if let some result := fastRes? then
      return result

    let fullDims : List Nat ← fullLabels.mapM fun label =>
      match Einsum.dimFind? dimMap label with
      | some dimension => pure dimension
      | none => failure
    let sCommon : Shape := Shape.ofList fullDims

    -- Align each operand to `fullLabels` (permute -> reshape insert ones -> broadcast).
    let mut aligned : List (RefTy (m := m) (α := α) sCommon) := []
    for ((⟨sIn, xIn⟩), labsIn) in List.zip processed inLabels do
      let targetOrder := fullLabels.filter (fun l => labsIn.contains l)
      let mut perm : List Nat := []
      for l in targetOrder do
        match labsIn.findIdx? (· == l) with
        | none => failure
        | some i => perm := perm ++ [i]
      let swaps : List Nat ←
        match Einsum.swapDepthsForPerm? perm (Spec.Shape.rank sIn) with
        | some ss => pure ss
        | none => failure
      let ⟨sPerm, xPerm⟩ ← OptionT.lift <|
        Einsum.permuteBySwaps (α := α) (m := m) (x := ⟨sIn, xIn⟩) swaps
      let dimsPerm := Shape.toList sPerm
      -- Reshape to insert singleton dims for missing labels.
      let mut di : Nat := 0
      let mut insertedDims : List Nat := []
      for l in fullLabels do
        if labsIn.contains l then
          let some d := dimsPerm[di]? | failure
          insertedDims := insertedDims ++ [d]
          di := di + 1
        else
          insertedDims := insertedDims ++ [1]
      let sInserted : Shape := Shape.ofList insertedDims
      let xInserted : RefTy (m := m) (α := α) sInserted ←
        if h : Spec.Shape.size sPerm = Spec.Shape.size sInserted then
          OptionT.lift <| reshape (m := m) (α := α) (s₁ := sPerm) (s₂ := sInserted) xPerm h
        else
          failure
      let xb ←
        match Shape.canBroadcastTo? sInserted sCommon with
        | some ⟨cb⟩ =>
            OptionT.lift <|
              broadcastTo (m := m) (α := α) (s₁ := sInserted) (s₂ := sCommon) cb xInserted
        | none => failure
      aligned := aligned ++ [xb]

    -- Multiply all aligned operands elementwise.
    let some prod0 := aligned.head? | failure
    let mut prod : RefTy (m := m) (α := α) sCommon := prod0
    for x in aligned.drop 1 do
      prod ← OptionT.lift <| mul (m := m) (α := α) (s := sCommon) prod x

    -- Sum-reduce contracted axes (a suffix by construction).
    let rec reduceContracted (n : Nat)
        (cur : Σ s : Shape, RefTy (m := m) (α := α) s) :
        OptionT m (Σ s : Shape, RefTy (m := m) (α := α) s) := do
      match n with
      | 0 => pure cur
      | n + 1 =>
          let curShape := cur.fst
          if hRank : Spec.Shape.rank curShape > 0 then
            if hw : curShape.wellFormed then
              let axis : Nat := Spec.Shape.rank curShape - 1
              let nextRef :=
                (← OptionT.lift <|
                  (by
                    letI : Shape.WellFormed curShape := ⟨hw⟩
                    haveI : Shape.HasNonemptyAxis axis curShape :=
                      Shape.inferNonemptyAxis (by grind)
                    exact reduceSum (m := m) (α := α) (s := curShape) axis cur.snd))
              let nextShape : Shape :=
                TorchLean.Tensor.shapeAfterSum curShape axis
              reduceContracted n ⟨nextShape, nextRef⟩
            else
              failure
          else
            failure

    let out0 ← reduceContracted contracted.length ⟨sCommon, prod⟩
    if outLabelsRaw = outLabels then
      pure out0
    else
      -- Diagonal embedding for repeated output labels:
      -- insert new axes (via reshape+broadcast) and zero out off-diagonal entries with a mask.
      let extras : List Label :=
        let rec go (seen : List Label) : List Label → List Label
          | .nil => []
          | .cons l ls =>
              if seen.contains l then
                l :: go seen ls
              else
                go (l :: seen) ls
        go [] outLabelsRaw
      let outCanon : List Label := outLabels ++ extras
      let mut cur : Σ s : Shape, RefTy (m := m) (α := α) s := out0
      for l in extras do
        let some baseIdx := outLabels.findIdx? (· == l) | failure
        let some d := Einsum.dimFind? dimMap l | failure
        let sReshape : Shape := Shape.appendDim cur.fst 1
        have hSz : Spec.Shape.size cur.fst = Spec.Shape.size sReshape := by
          simpa [sReshape] using (Spec.Shape.size_appendDim cur.fst 1).symm
        let xReshaped ← OptionT.lift <|
          reshape (m := m) (α := α) (s₁ := cur.fst) (s₂ := sReshape) cur.snd hSz
        let sBroad : Shape := Shape.appendDim cur.fst d
        let xExpanded ←
          match Shape.canBroadcastTo? sReshape sBroad with
          | some ⟨cb⟩ =>
              OptionT.lift <|
                broadcastTo (m := m) (α := α) (s₁ := sReshape) (s₂ := sBroad) cb xReshaped
          | none => failure
        let qIdx : Nat := Spec.Shape.rank sBroad - 1
        let maskT : Tensor α sBroad := Einsum.diagMaskForShape (α := α) sBroad baseIdx qIdx
        let mask ← OptionT.lift <| const (m := m) (α := α) (s := sBroad) maskT
        let xMasked ← OptionT.lift <| mul (m := m) (α := α) (s := sBroad) xExpanded mask
        cur := ⟨sBroad, xMasked⟩

      let some perm := Einsum.permForDuplicateLabels? outCanon outLabelsRaw | failure
      let some swaps := Einsum.swapDepthsForPerm? perm (Spec.Shape.rank cur.fst) | failure
      OptionT.lift <| Einsum.permuteBySwaps (α := α) (m := m) (x := cur) swaps
  computation.run

/-- `einsum` with an expected output shape. -/
def «einsum» {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {sOut : Shape}
    (equation : String)
    (xs : List (Σ s : Shape, RefTy (m := m) (α := α) s)) :
    m (Option (RefTy (m := m) (α := α) sOut)) := do
  let r? ← einsum? (α := α) (m := m) equation xs
  match r? with
  | none => pure none
  | some ⟨s, r⟩ =>
      if h : s = sOut then
        pure (some (h ▸ r))
      else
        pure none
end F
end Model
end Autograd
end Runtime
