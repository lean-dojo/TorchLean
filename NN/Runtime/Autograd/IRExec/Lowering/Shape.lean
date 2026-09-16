/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Lowering.Primitives
public import NN.Runtime.Autograd.IRExec.Lowering.Common

/-!
# Shape and Concatenation IR Lowering

Checked lowering for permutations, reshaping, flattening, concatenation, and transpose.

Concatenation along axis `0` reads each parent through a typed index and folds
`Tensor.concatAxisSpec` over the parents; concatenation along another axis first moves that axis to
the front of every parent, folds, and moves it back. Both branches use
`concatLeadingAxisFromInputs`, and the output-shape cast is justified by
`concatLeadingAxisFromInputs_size_eq_sum` rather than by a proof embedded in the runtime code.
The nonzero-axis branch validates each parent exactly as `NN.IR.Graph.permuteSomeTensor` does and
records that evidence in `ConcatFrontInput`, so the correctness proof can replay the evaluator's
permutation on every parent.

Each operation has its own small `lower*` definition. `lowerShape` only dispatches on the operation
kind, and the `lowerShape_*` equation lemmas let correctness proofs reduce a dispatch to the branch
they care about without unfolding the whole dispatcher.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace IRExec

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Proofs.Autograd.Algebra
-- Typed context indices come from `NN.Proofs.Autograd.Tape.Util.Idx`, the one place
-- `Idx` and `getIdx` are defined.
open Proofs (Idx getIdx)
open NN.IR

namespace Internal

/-- Checked lowering for `.permute perm` through a sequence of adjacent axis swaps. -/
def lowerPermute {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (perm : Array Nat) :
    NodeLoweringResult ctx := do
  let g := ctx.graph
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match unaryParent? n.parents with
  | some pId =>
      let pNode ← g.getNode pId
      let sIn := pNode.outShape
      let ip ← parentIdx pId sIn
      match Spec.Shape.permute? sIn perm.toList with
      | none =>
          throw s!"IRExec: node {i}: invalid permutation {repr perm} for shape {repr sIn}"
      | some expected =>
          let swaps ← NN.IR.Graph.swapDepthsForPerm perm (Spec.Shape.rank sIn)
          let sFinal : Shape := swapShapeBySwaps sIn swaps
          if hFinal : sFinal = expected then
            if hOut : expected = τ then
              let forward := fun ctx : TorchLean.TensorPack α Γ =>
                let x := getIdx (α := α) (xs := ctx) ip
                let y : Tensor α sFinal := applySwapsTensor (α := α) (s := sIn) (swaps :=
                  swaps) x
                let yExpected : Tensor α expected := Tensor.castShape y hFinal
                Tensor.castShape yExpected hOut
              pure <| fwd forward
            else
              throw <|
                s!"IRExec: node {i}: permute outShape mismatch: " ++
                  s!"expected={repr expected}, declared={repr τ}"
          else
            throw <|
              s!"IRExec: node {i}: permute shape mismatch: computed={repr sFinal}, " ++
                s!"expected={repr expected} ({n.summary})"
  | _ => throw s!"IRExec: node {i}: permute expects 1 parent ({n.summary})"

/-- Checked lowering for `.reshape inS outS`. -/
def lowerReshape {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (inS outS : Shape) :
    NodeLoweringResult ctx := do
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match unaryParent? n.parents with
  | some pId =>
      let ip ← parentIdx pId inS
      if hNumel : Spec.Shape.size inS = Spec.Shape.size outS then
        if hOut : outS = τ then
          let forward := fun ctx : TorchLean.TensorPack α Γ =>
            let x := getIdx (α := α) (xs := ctx) ip
            hOut ▸ Tensor.reshapeSpec (α := α) (source := inS) (target := outS) x hNumel
          pure <| fwd forward
        else
          throw <|
            s!"IRExec: node {i}: reshape outShape mismatch: kind={repr outS}, " ++
              s!"declared={repr τ}"
      else
        throw <|
          s!"IRExec: node {i}: reshape numel mismatch: {Spec.Shape.size inS} vs " ++
            s!"{Spec.Shape.size outS}"
  | _ => throw s!"IRExec: node {i}: reshape expects 1 parent ({n.summary})"

/-- Checked lowering for `.flatten s`. -/
def lowerFlatten {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (s : Shape) : NodeLoweringResult ctx := do
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match unaryParent? n.parents with
  | some pId =>
      let ip ← parentIdx pId s
      let expected : Shape := .dim (Spec.Shape.size s) .scalar
      if hOut : expected = τ then
        let forward := fun ctx : TorchLean.TensorPack α Γ =>
          let x := getIdx (α := α) (xs := ctx) ip
          let y : Tensor α expected := Tensor.flattenSpec (α := α) (shape := s) x
          hOut ▸ y
        pure <| fwd forward
      else
        throw <|
          s!"IRExec: node {i}: flatten outShape mismatch: " ++
            s!"expected={repr expected}, declared={repr τ} ({n.summary})"
  | _ => throw s!"IRExec: node {i}: flatten expects 1 parent ({n.summary})"

/--
Build the leading-axis concat inputs for `.concat 0`: one typed index per parent, all sharing the
tail shape `rest`.
-/
def concatAxisZeroInputs {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (rest : Shape) :
    Except String (Array (ConcatInput α Γ rest)) :=
  ctx.node.parents.mapM fun pid => do
    let pNode ← ctx.graph.getNode pid
    match pNode.outShape with
    | .dim nP restP =>
        if _hRest : restP = rest then
          let ip ← ctx.parentIdx pid (.dim nP rest)
          pure ⟨nP, fun context => getIdx (α := α) (xs := context) ip⟩
        else
          throw <|
            s!"IRExec: node {ctx.index}: concat axis=0 tail mismatch: {repr restP} vs " ++
              s!"{repr rest}"
    | _ =>
        throw <|
          s!"IRExec: node {ctx.index}: concat axis=0 expects rank≥1 parents, got " ++
            s!"{repr pNode.outShape}"

/--
One validated parent of a nonzero-axis concat: its typed index, the adjacent swaps that move the
concatenated axis to the front, and the resulting leading extent. The proof fields record exactly
the checks the IR evaluator performs in `NN.IR.Graph.permuteSomeTensor`, so the lowering and the
evaluator permute each parent identically.
-/
structure ConcatFrontInput (α : Type) [TorchLean.Storage α] (Γ : List Shape)
    (permFront : Array Nat) (restFront : Shape) where
  /-- The parent's declared shape. -/
  sIn : Shape
  /-- Typed index of the parent in the runtime context. -/
  ip : Idx Γ sIn
  /-- Adjacent swaps realizing `permFront` on a tensor of rank `sIn.rank`. -/
  swaps : Array Nat
  /-- Leading extent of the permuted parent. -/
  nP : Nat
  /-- `permFront` is a valid permutation of the parent shape with the shared tail. -/
  perm_eq : Spec.Shape.permute? sIn permFront.toList = some (.dim nP restFront)
  /-- The swaps were computed for the parent's own rank. -/
  swaps_eq : NN.IR.Graph.swapDepthsForPerm permFront (Spec.Shape.rank sIn) = .ok swaps
  /-- Applying the swaps yields the permuted shape. -/
  final_eq : swapShapeBySwaps sIn swaps = .dim nP restFront

/-- Read and permute the parent so the concatenated axis comes first. -/
def ConcatFrontInput.toInput {α : Type} [TorchLean.Storage α] [Context α] {Γ : List Shape}
    {permFront : Array Nat} {restFront : Shape}
    (input : ConcatFrontInput α Γ permFront restFront) : ConcatInput α Γ restFront :=
  ⟨input.nP, fun context =>
    Tensor.castShape
      (applySwapsTensor (α := α) (s := input.sIn) (swaps := input.swaps)
        (getIdx (α := α) (xs := context) input.ip))
      input.final_eq⟩

/--
Build the concat inputs for a nonzero axis: every parent is read through a typed index and
permuted by `permFront` so the concatenated axis comes first, sharing the tail `restFront`.
-/
def concatAxisFrontInputs {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (permFront : Array Nat)
    (restFront : Shape) :
    Except String (Array (ConcatFrontInput α Γ permFront restFront)) :=
  ctx.node.parents.mapM fun pid => do
    let pNode ← ctx.graph.getNode pid
    let sIn := pNode.outShape
    let ip ← ctx.parentIdx pid sIn
    match hPerm : Spec.Shape.permute? sIn permFront.toList with
    | none =>
        throw <|
          s!"IRExec: node {ctx.index}: concat: invalid permutation " ++
            s!"{repr permFront} for parent shape {repr sIn}"
    | some (.dim nP restP) =>
        if hRest : restP = restFront then
          match hSwaps : NN.IR.Graph.swapDepthsForPerm permFront (Spec.Shape.rank sIn) with
          | .error msg => throw s!"IRExec: node {ctx.index}: concat: {msg}"
          | .ok swaps =>
              if hFinal : swapShapeBySwaps sIn swaps = .dim nP restFront then
                pure
                  { sIn := sIn, ip := ip, swaps := swaps, nP := nP
                    perm_eq := by rw [hPerm, hRest]
                    swaps_eq := hSwaps
                    final_eq := hFinal }
              else
                throw <|
                  s!"IRExec: node {ctx.index}: concat permute shape mismatch: " ++
                    s!"computed={repr (swapShapeBySwaps sIn swaps)}, " ++
                    s!"expected={repr (Shape.dim nP restFront)} ({ctx.node.summary})"
        else
          throw <|
            s!"IRExec: node {ctx.index}: concat: permuted tail mismatch: " ++
              s!"{repr restP} vs {repr restFront} ({ctx.node.summary})"
    | some _ =>
        throw <|
          s!"IRExec: node {ctx.index}: concat expects rank≥1 parents, got {repr sIn}"

/--
Fold concat inputs along the leading axis into a tensor of the declared leading extent `nOut`.

`hSum` records that the input extents add up to `nOut`, so the cast is justified by
`concatLeadingAxisFromInputs_size_eq_sum`.
-/
def concatInputsForward {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} {rest : Shape} (inputs : Array (ConcatInput α Γ rest)) (nOut : Nat)
    (hSum : inputs.foldl (fun acc input => acc + input.1) 0 = nOut)
    (context : TorchLean.TensorPack α Γ) : Tensor α (.dim nOut rest) :=
  let out := concatLeadingAxisFromInputs (α := α) (Γ := Γ) (rest := rest) context inputs
  Tensor.castShape out.2
    (congrArg (fun k => Shape.dim k rest)
      ((concatLeadingAxisFromInputs_size_eq_sum (α := α) context inputs).trans hSum))

/-- Checked lowering for `.concat axis` along an arbitrary axis. -/
def lowerConcat {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (axis : Nat) : NodeLoweringResult ctx := do
  let g := ctx.graph
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let fwd (forward : TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  let parents := n.parents
  if parents.size < 2 then
    throw s!"IRExec: node {i}: concat expects at least 2 parents"

  let parentShapes : Array Shape ← parents.mapM (fun pid => do
    let pNode ← g.getNode pid
    pure pNode.outShape)
  let expected ←
    match OpContracts.inferConcatOutShape axis parentShapes with
    | .ok s => pure s
    | .error msg => throw s!"IRExec: node {i}: {msg} ({n.summary})"
  if expected != τ then
    throw <|
      s!"IRExec: node {i}: concat outShape mismatch: expected={repr expected}, " ++
        s!"declared={repr τ} ({n.summary})"

  if axis = 0 then
    match hτ : τ with
    | .dim nOut rest =>
        let inputs ← concatAxisZeroInputs ctx rest
        if hSum : inputs.foldl (fun acc input => acc + input.1) 0 = nOut then
          let forward := fun context : TorchLean.TensorPack α Γ =>
            Tensor.castShape (concatInputsForward inputs nOut hSum context) hτ.symm
          pure <| fwd forward
        else
          throw <|
            s!"IRExec: node {i}: concat out dim mismatch: declared {nOut}, computed " ++
              s!"{inputs.foldl (fun acc input => acc + input.1) 0} ({n.summary})"
    | _ =>
        throw s!"IRExec: node {i}: concat axis=0 expects rank≥1 outShape, got {repr τ}"
  else
    -- General axis concat: permute `axis` to the front, concatenate along axis 0, then permute
    -- back.
    let permFront ←
      match OpContracts.permMoveAxisToFront axis τ with
      | .ok perm => pure perm
      | .error msg => throw s!"IRExec: node {i}: concat: {msg}"
    let permBack ←
      match OpContracts.inversePerm permFront with
      | .ok perm => pure perm
      | .error msg => throw s!"IRExec: node {i}: concat: {msg}"
    match Spec.Shape.permute? τ permFront.toList with
    | none =>
        throw <|
          s!"IRExec: node {i}: concat: invalid permutation {repr permFront} for " ++
            s!"shape {repr τ}"
    | some outFrontExpected =>
        match hOutFrontExpected : outFrontExpected with
        | .dim nOutFront restFront =>
            match Spec.Shape.permute? outFrontExpected permBack.toList with
            | none =>
                throw <|
                  s!"IRExec: node {i}: concat: invalid inverse permutation {repr permBack} " ++
                    s!"for shape {repr outFrontExpected}"
            | some _ =>
            let swapsBack ←
              NN.IR.Graph.swapDepthsForPerm permBack (Spec.Shape.rank outFrontExpected)
            let τBackFinal : Shape := swapShapeBySwaps outFrontExpected swapsBack
            if hOutBackFinal : τBackFinal = τ then
              let frontInputs ← concatAxisFrontInputs ctx permFront restFront
              let inputs := frontInputs.map ConcatFrontInput.toInput
              if hSum : inputs.foldl (fun acc input => acc + input.1) 0 = nOutFront then
                let forward := fun context : TorchLean.TensorPack α Γ =>
                  let tFront : Tensor α outFrontExpected :=
                    Tensor.castShape (concatInputsForward inputs nOutFront hSum context)
                      hOutFrontExpected.symm
                  let tBack : Tensor α τBackFinal :=
                    applySwapsTensor (α := α) (s := outFrontExpected) (swaps := swapsBack)
                      tFront
                  Tensor.castShape tBack hOutBackFinal
                pure <| fwd forward
              else
                throw <|
                  s!"IRExec: node {i}: concat out dim mismatch: declared {nOutFront}, " ++
                    s!"computed {inputs.foldl (fun acc input => acc + input.1) 0} " ++
                    s!"({n.summary})"
            else
              throw <|
                s!"IRExec: node {i}: concat permute-back shape mismatch: " ++
                  s!"computed={repr τBackFinal}, expected={repr τ} ({n.summary})"
        | _ =>
            throw s!"IRExec: node {i}: concat expects rank≥1 outShape, got {repr τ}"

/-- Checked lowering for `.transpose axis₁ axis₂` through a sequence of adjacent axis swaps. -/
def lowerTranspose {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (axis₁ axis₂ : Nat) :
    NodeLoweringResult ctx := do
  let g := ctx.graph
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match unaryParent? n.parents with
  | some pId =>
      let pNode ← g.getNode pId
      let sIn := pNode.outShape
      let ip ← parentIdx pId sIn
      let perm ← OpContracts.transposePerm sIn.rank axis₁ axis₂
      let expected ←
        match Shape.permute? sIn perm.toList with
        | some expected => pure expected
        | none => throw s!"IRExec: node {i}: invalid transpose axes ({n.summary})"
      let swaps ← NN.IR.Graph.swapDepthsForPerm perm sIn.rank
      let computed : Shape := swapShapeBySwaps sIn swaps
      if hComputed : computed = expected then
        if hOut : expected = τ then
          let forward := fun ctx : TorchLean.TensorPack α Γ =>
            let x := getIdx (α := α) (xs := ctx) ip
            let y : Tensor α computed := applySwapsTensor (α := α) (s := sIn)
              (swaps := swaps) x
            Tensor.castShape (Tensor.castShape y hComputed) hOut
          pure <| fwd forward
        else
          throw s!"IRExec: node {i}: transpose outShape mismatch ({n.summary})"
      else
        throw s!"IRExec: node {i}: transpose lowering mismatch ({n.summary})"
  | _ => throw s!"IRExec: node {i}: transpose expects 1 parent ({n.summary})"

/-- Checked lowering for permutations, reshaping, flattening, concatenation, and transpose. -/
def lowerShape {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (kind : OpKind) :
    NodeLoweringResult ctx :=
  match kind with
  | .permute perm => lowerPermute ctx perm
  | .reshape inS outS => lowerReshape ctx inS outS
  | .flatten s => lowerFlatten ctx s
  | .concat axis => lowerConcat ctx axis
  | .transpose axis₁ axis₂ => lowerTranspose ctx axis₁ axis₂
  | _ => throw s!"IRExec: internal error: operation routed to lowerShape"

variable {α : Type} [TorchLean.Storage α] [Context α] {Γ : List Shape}

/-- Dispatch equation for `.permute perm`. -/
@[simp] theorem lowerShape_permute (ctx : NodeLoweringContext α Γ) (perm : Array Nat) :
    lowerShape ctx (.permute perm) = lowerPermute ctx perm := rfl

/-- Dispatch equation for `.reshape inS outS`. -/
@[simp] theorem lowerShape_reshape (ctx : NodeLoweringContext α Γ) (inS outS : Shape) :
    lowerShape ctx (.reshape inS outS) = lowerReshape ctx inS outS := rfl

/-- Dispatch equation for `.flatten s`. -/
@[simp] theorem lowerShape_flatten (ctx : NodeLoweringContext α Γ) (s : Shape) :
    lowerShape ctx (.flatten s) = lowerFlatten ctx s := rfl

/-- Dispatch equation for `.concat axis`. -/
@[simp] theorem lowerShape_concat (ctx : NodeLoweringContext α Γ) (axis : Nat) :
    lowerShape ctx (.concat axis) = lowerConcat ctx axis := rfl

/-- Dispatch equation for `.transpose axis₁ axis₂`. -/
@[simp] theorem lowerShape_transpose (ctx : NodeLoweringContext α Γ) (axis₁ axis₂ : Nat) :
    lowerShape ctx (.transpose axis₁ axis₂) = lowerTranspose ctx axis₁ axis₂ := rfl

end Internal
end IRExec
end Autograd
end Runtime
