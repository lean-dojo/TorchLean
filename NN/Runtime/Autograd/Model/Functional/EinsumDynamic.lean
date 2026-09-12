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
set: reorder, reshape, broadcast, multiply, sum. It is one large monadic definition and by far the
most expensive thing to elaborate in this corner of the tree, so it lives on its own rather than in
`Functional.Einsum`. Everything that only needs the typed contractions or the label bookkeeping
imports that module and does not pay for this one.
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

This is implemented purely by reordering, reshaping, broadcasting, elementwise multiplication,
and summing contracted axes.
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

    -- Fast paths for common contractions.
    --
    -- The generic lowering below broadcasts all operands to a common label shape, multiplies,
    -- then sums contracted axes. This is verifier-friendly but can allocate extremely large
    -- intermediate tensors for common patterns like matmul/bmm/attention.
    --
    -- These fast paths dispatch to existing primitives (`matmul`/`bmm` plus small reshapes/
    -- transposes), preserving semantics while avoiding the huge broadcasted intermediate.
    let attemptFast : OptionT m (Σ s : Shape, RefTy (m := m) (α := α) s) := do
      if parsed.inputs.length != 2 then
        failure
      if xs.length != 2 then
        failure
      let some sub0 := parsed.inputs[0]? | failure
      let some sub1 := parsed.inputs[1]? | failure
      let some x0 := xs[0]? | failure
      let some x1 := xs[1]? | failure
      let out? := parsed.output?

      -- Require "simple" subscripts for fast paths: no ellipsis and no post-ellipsis labels.
      if sub0.hasEll || sub1.hasEll then
        failure
      match sub0.post, sub1.post with
      | .nil, .nil => pure ()
      | _, _ => failure

      -- Primitive axis roles must name distinct labels. Aliases require diagonal extraction
      -- or embedding, which the generic lowering handles below.
      let requireDistinct (labels : List Char) : OptionT m Unit := do
        if Einsum.hasDupLabels (labels.map Label.chr) then failure

      let expectOut (expected : List Char) : OptionT m Unit := do
        match out? with
        | some o =>
            if o.hasEll then failure
            match o.post with
            | .nil =>
                if o.pre = expected then pure () else failure
            | _ => failure
        | none =>
            -- Match the generic implicit output. Shared batch/head labels are contracted
            -- unless explicitly retained, so batched primitives cannot keep them here.
            let inferred :=
              implicitOutputLabels [sub0.pre.map Label.chr, sub1.pre.map Label.chr] 0
            if inferred = expected.map Label.chr then pure () else failure

      -- Case dispatch is primarily driven by shapes.
      match x0 with
      -- Matmul: `[i,j] × [j,k] → [i,k]` and subscripts `ij,jk->ik` (or implicit output).
      | ⟨.dim iDim (.dim jDim .scalar), a⟩ =>
          match x1 with
          | ⟨.dim jDim2 (.dim kDim .scalar), b⟩ =>
              if hJ : jDim = jDim2 then
                let b' : RefTy (m := m) (α := α) (.dim jDim (.dim kDim .scalar)) := by
                  simpa [hJ] using b
                -- Extract labels (must be length-2 on both operands).
                let some li0 := sub0.pre[0]? | failure
                let some lj0 := sub0.pre[1]? | failure
                if sub0.pre.length != 2 then failure
                let some lj1 := sub1.pre[0]? | failure
                let some lk1 := sub1.pre[1]? | failure
                if sub1.pre.length != 2 then failure
                if lj0 != lj1 then failure
                requireDistinct [li0, lj0, lk1]
                expectOut [li0, lk1]
                let out ← OptionT.lift <|
                  einsumIjJkIk (m := m) (α := α)
                    (iDim := iDim) (jDim := jDim) (kDim := kDim) a b'
                pure ⟨.dim iDim (.dim kDim .scalar), out⟩
              else
                failure
          | _ => failure
      -- BMM: `[b,i,j] × [b,j,k] → [b,i,k]` and subscripts `bij,bjk->bik`.
      | ⟨.dim batch (.dim iDim (.dim jDim .scalar)), a⟩ =>
          match x1 with
          | ⟨.dim batch2 (.dim jDim2 (.dim kDim .scalar)), b⟩ =>
              if hB : batch = batch2 then
                if hJ : jDim = jDim2 then
                  let b' : RefTy (m := m) (α := α)
                      (.dim batch (.dim jDim (.dim kDim .scalar))) := by
                    simpa [hB, hJ] using b
                  let some lb0 := sub0.pre[0]? | failure
                  let some li0 := sub0.pre[1]? | failure
                  let some lj0 := sub0.pre[2]? | failure
                  if sub0.pre.length != 3 then failure
                  let some lb1 := sub1.pre[0]? | failure
                  let some lj1 := sub1.pre[1]? | failure
                  let some lk1 := sub1.pre[2]? | failure
                  if sub1.pre.length != 3 then failure
                  if lb0 != lb1 then failure
                  if lj0 != lj1 then failure
                  requireDistinct [lb0, li0, lj0, lk1]
                  expectOut [lb0, li0, lk1]
                  let out ← OptionT.lift <|
                    einsumBijBjkBik (m := m) (α := α)
                      (batch := batch) (iDim := iDim) (jDim := jDim) (kDim := kDim) a b'
                  pure ⟨.dim batch (.dim iDim (.dim kDim .scalar)), out⟩
                else
                  failure
              else
                failure
          | _ => failure
      -- 4D attention-like contractions (Q·Kᵀ or Attn·V), selected by label patterns.
      | ⟨.dim batch (.dim heads (.dim iDim (.dim tDim .scalar))), x0Ref⟩ =>
          match x1 with
          | ⟨.dim batch2 (.dim heads2 (.dim jDim (.dim dDim .scalar))), x1Ref⟩ =>
              if hB : batch = batch2 then
                if hH : heads = heads2 then
                  let x1Ref' : RefTy (m := m) (α := α)
                      (.dim batch (.dim heads (.dim jDim (.dim dDim .scalar)))) := by
                    simpa [hB, hH] using x1Ref
                  let some lb0 := sub0.pre[0]? | failure
                  let some lh0 := sub0.pre[1]? | failure
                  let some l2_0 := sub0.pre[2]? | failure
                  let some l3_0 := sub0.pre[3]? | failure
                  if sub0.pre.length != 4 then failure
                  let some lb1 := sub1.pre[0]? | failure
                  let some lh1 := sub1.pre[1]? | failure
                  let some l2_1 := sub1.pre[2]? | failure
                  let some l3_1 := sub1.pre[3]? | failure
                  if sub1.pre.length != 4 then failure
                  if lb0 != lb1 then failure
                  if lh0 != lh1 then failure
                  -- Two attention-like cases:
                  -- 1) Q·Kᵀ: `bhid,bhjd -> bhij`  (shared last label across inputs).
                  -- 2) Attn·V: `bhij,bhjd -> bhid` (contract sub0 last label with sub1
                  -- third label).
                  if l3_0 = l3_1 then
                    -- Q·Kᵀ: sub0 = [b,h,i,d], sub1 = [b,h,j,d], and shapes must agree on `d`.
                    if hD : dDim = tDim then
                      let x0Ref' : RefTy (m := m) (α := α)
                          (.dim batch (.dim heads (.dim iDim (.dim dDim .scalar)))) := by
                        simpa [hD] using x0Ref
                      requireDistinct [lb0, lh0, l2_0, l2_1, l3_0]
                      expectOut [lb0, lh0, l2_0, l2_1]
                      let out ← OptionT.lift <|
                        einsumBhidBhjdBhij (m := m) (α := α)
                          (batch := batch) (heads := heads) (iDim := iDim) (jDim := jDim)
                          (dDim := dDim) x0Ref' x1Ref'
                      pure ⟨.dim batch (.dim heads (.dim iDim (.dim jDim .scalar))), out⟩
                    else
                      failure
                  else if l3_0 = l2_1 then
                    -- Attn·V: sub0 = [b,h,i,j], sub1 = [b,h,j,d], and shapes must agree on `j`.
                    if hJ : jDim = tDim then
                      let x0Ref' : RefTy (m := m) (α := α)
                          (.dim batch (.dim heads (.dim iDim (.dim jDim .scalar)))) := by
                        simpa [hJ] using x0Ref
                      requireDistinct [lb0, lh0, l2_0, l3_0, l3_1]
                      expectOut [lb0, lh0, l2_0, l3_1]
                      let out ← OptionT.lift <|
                        einsumBhijBhjdBhid (m := m) (α := α)
                          (batch := batch) (heads := heads) (iDim := iDim) (jDim := jDim)
                          (dDim := dDim) x0Ref' x1Ref'
                      pure ⟨.dim batch (.dim heads (.dim iDim (.dim dDim .scalar))), out⟩
                    else
                      failure
                  else
                    failure
                else
                  failure
              else
                failure
          | _ => failure
      | _ => failure

    let fastRes? ← OptionT.lift attemptFast.run
    if let some r := fastRes? then
      return r

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
