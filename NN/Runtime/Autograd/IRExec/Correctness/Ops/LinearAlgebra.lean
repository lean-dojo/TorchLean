/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Correctness.Common

/-!
# Linear Algebra

Linear-algebra correctness lemmas for the IR-to-forward-executor lowering.

This file proves the forward-correctness step for lowering a `.matmul` IR node into a single SSA
node in the lowered `ForwardData`. Concretely, it shows that:

* if `buildFrom` successfully lowers a `.matmul` node at position `i`, and
* we compare the IR evaluator `NN.IR.Graph.denoteAllFrom` against the forward-graph evaluator
  `denoteAllState`,

then the value appended by the IR evaluator at step `i` is the same tensor as the value produced by
the forward-graph node's `forward`.

The lowering and the IR semantics use `OpContracts.matmulDims` to check batch broadcasting and
vector promotion. Both compute `NN.IR.Graph.matmulWithDims`, so the statement covers vector dot
products, matrix/vector products, and arbitrarily many broadcast batch axes.

This module is about semantic correctness. Performance backends (external BLAS libraries, kernel
fusion, and so on) are a separate lowering layer and are not involved here.

The shape rule matches the public PyTorch matrix multiplication API:

* `torch.matmul`: https://pytorch.org/docs/stable/generated/torch.matmul.html

## Main definitions

- `buildFrom_denoteAllFrom_matmul_success`: correctness step once the typed operands are known.
- `buildFrom_denoteAllFrom_matmul`: correctness step for `.matmul` lowering.

## Implementation notes

- The proof follows the lowering pass's shape checks, so each branch records the same preconditions
  that the lowering code enforces.
- A successful contract preserves the two original operand shapes, which identifies the layout
  recovered from runtime values with the layout used by the lowered closure.

## Tags

matmul, bmm, correctness, ir, runtime
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace IRExec

open Spec TorchLean
open Proofs.Autograd.Algebra
open NN.IR
open Internal
-- Typed context indices come from `NN.Proofs.Autograd.Tape.Util.Idx`, the one place
-- `Idx` and `getIdx` are defined.
open Proofs (Idx getIdx)

/--
Correctness lemma for the `.matmul` lowering step once the typed operands are known.

The shared contract supplies the operand and output shapes. Both evaluators apply that layout,
giving the equality needed to hand off to the tail-induction hypothesis.
-/
theorem buildFrom_denoteAllFrom_matmul_success
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .matmul) (hi : i < g.nodes.size)
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := Spec.SomeTensor.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x))
    (aId bId : Nat) (dims : OpContracts.MatmulDims)
    (hp : binaryParents? n.parents = some (aId, bId))
    (hDims : OpContracts.matmulDims dims.leftShape dims.rightShape = .ok dims)
    (ia : Idx ([inShape] ++ ss) dims.leftShape)
    (hIa : mkIdx (inShape := inShape) (ss := ss) aId dims.leftShape = .ok ia)
    (ib : Idx ([inShape] ++ ss) dims.rightShape)
    (hIb : mkIdx (inShape := inShape) (ss := ss) bId dims.rightShape = .ok ib)
    (hOut : dims.outShape = n.outShape)
    (hBuildNext :
      buildFrom (α := α) (g := g) (payload := payload)
        (inShape := inShape) (i := i + 1)
        (st := (⟨ss ++ [n.outShape],
          .snoc (ss := ss) gd
            (mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
              hOut ▸ NN.IR.Graph.matmulWithDims (α := α) dims
                (readTensor (α := α) (xs := ctx) ia) (readTensor (α := α) (xs := ctx) ib)))⟩ :
          State α inShape)) = .ok st') :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := Spec.SomeTensor.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (Spec.SomeTensor α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TorchLean.TensorPack α ([inShape] ++ ss) :=
    ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil)
  let input : Spec.SomeTensor α := Spec.SomeTensor.mk (α := α) inShape x
  let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
    mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
      hOut ▸ NN.IR.Graph.matmulWithDims (α := α) dims
        (readTensor (α := α) (xs := ctx) ia) (readTensor (α := α) (xs := ctx) ib))
  have hRec :
      buildFrom (α := α) (g := g) (payload := payload)
        (inShape := inShape) (i := i + 1)
        (st := (⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩ : State α inShape)) =
          .ok st' := by
    exact hBuildNext
  have hGetA :
      vals0[aId]? = some
        (Spec.SomeTensor.mk (α := α) dims.leftShape
          (getIdx (α := α) (xs := ctx) ia)) := by
    simpa only [vals0, ctx] using
      (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
        (gd := gd) (x := x) (pid := aId)
        (s := dims.leftShape) (idx := ia) hIa)
  have hGetB :
      vals0[bId]? = some
        (Spec.SomeTensor.mk (α := α) dims.rightShape
          (getIdx (α := α) (xs := ctx) ib)) := by
    simpa only [vals0, ctx] using
      (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
        (gd := gd) (x := x) (pid := bId)
        (s := dims.rightShape) (idx := ib) hIb)
  have hEval :
      NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
        (input := input) (vals := vals0) (i := i) =
        .ok (Spec.SomeTensor.mk (α := α) n.outShape (nodeData.eval ctx)) := by
    simpa only [nodeData, mkForwardNode_eval, readTensor_ofPack] using
      (evalAt_matmul_dims_ok (α := α)
        (g := g) (payload := payload) (input := input) (vals := vals0)
        (i := i) (n := n) (aId := aId) (bId := bId)
        (dims := dims)
        (aT := getIdx (α := α) (xs := ctx) ia)
        (bT := getIdx (α := α) (xs := ctx) ib)
        hN hk hp hDims hGetA hGetB hOut)
  have hStep :
      denoteAllState (α := α) inShape
        (st := (⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩ : State α inShape)) x =
          vals0.push (Spec.SomeTensor.mk (α := α) n.outShape (nodeData.eval ctx)) := by
    simpa [vals0, nodeData, ctx] using
      (denoteAllState_snoc (α := α) (inShape := inShape)
        (ss := ss) (τ := n.outShape) (gd := gd)
        (nodeData := nodeData) (x := x))
  have hTail := ih ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩ hRec
  exact buildFrom_denoteAllFrom_finish (α := α) (g := g) (payload := payload)
    (i := i) (x := x) (hi := hi) (τ := n.outShape)
    (nodeData := nodeData)
    (st1 := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩)
    (st' := st') (ctx := ctx) (vals0 := vals0) (input := input)
    hTail hEval hStep

/--
Correctness lemma for the `.matmul` node lowering pass.

The proof mirrors `lowerMatmul`: it follows the shared contract and typed-shape checks, then
hands the typed operands to
`buildFrom_denoteAllFrom_matmul_success`.
-/
theorem buildFrom_denoteAllFrom_matmul
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .matmul) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := Spec.SomeTensor.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := Spec.SomeTensor.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  unfold buildFrom at hBuild
  simp [hi, hN, -OpContracts.MatmulDims.outShape] at hBuild
  simp (config := { failIfUnchanged := false })
    [hk, lowerMatmul, -OpContracts.matmulDims, -OpContracts.MatmulDims.outShape,
      -NN.IR.Graph.matmulWithDims] at hBuild
  cases hp : binaryParents? n.parents with
  | none =>
      simp [hp] at hBuild
      cases hBuild
  | some parentIds =>
  rcases parentIds with ⟨aId, bId⟩
  cases hA : g.getNode aId with
  | error msg => simp [hp, hA] at hBuild
  | ok aNode =>
  cases hB : g.getNode bId with
  | error msg => simp [hp, hA, hB] at hBuild
  | ok bNode =>
  cases hDims : OpContracts.matmulDims aNode.outShape bNode.outShape with
  | error msg => simp [hp, hA, hB, hDims, -OpContracts.matmulDims] at hBuild
  | ok dims =>
  simp (config := { failIfUnchanged := false })
    [hp, hA, hB, hDims, -OpContracts.matmulDims, -OpContracts.MatmulDims.outShape,
      -NN.IR.Graph.matmulWithDims] at hBuild
  cases hIa : mkIdx (inShape := inShape) (ss := ss) aId dims.leftShape with
  | error msg => simp [hIa] at hBuild
  | ok ia =>
  cases hIb : mkIdx (inShape := inShape) (ss := ss) bId dims.rightShape with
  | error msg => simp [hIa, hIb] at hBuild
  | ok ib =>
  by_cases hOut : dims.outShape = n.outShape
  · obtain ⟨hLeft, hRight⟩ := OpContracts.matmulDims_shapes hDims
    have hLayout : OpContracts.matmulDims dims.leftShape dims.rightShape = .ok dims := by
      rw [← hLeft, ← hRight]
      exact hDims
    apply buildFrom_denoteAllFrom_matmul_success (α := α)
      (g := g) (payload := payload) (gd := gd) (i := i) (st' := st')
      (x := x) (n := n) hN hk hi ih
      (aId := aId) (bId := bId) (dims := dims) hp hLayout
      (ia := ia) hIa (ib := ib) hIb hOut
    simpa [hIa, hIb, hOut, Tensor.eqRec_eq_cast_shape,
      -OpContracts.MatmulDims.outShape, -NN.IR.Graph.matmulWithDims] using hBuild
  · exact False.elim <|
      throw_bind_ne_ok (by
        simpa [hIa, hIb, hOut, -OpContracts.MatmulDims.outShape,
          -NN.IR.Graph.matmulWithDims] using hBuild)

end IRExec
end Autograd
end Runtime
