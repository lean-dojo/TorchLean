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

The lowering and the IR semantics accept the same matmul shapes: both parents share an arbitrary
leading shape and end in the matrix axes `[m, n]` and `[n, p]`. Plain matrix multiplication,
batched multiplication with one batch axis, and multiplication over several batch axes are all
instances of one statement, since both sides compute `NN.IR.Graph.matmulLeading`.

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
- The leading shape is recovered from the reversed dimension lists exactly as in the lowering, so
  the typed operands line up with `evalAt_matmul_leading_ok` without further shape rewriting.

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

The parents share the leading shape `Shape.ofList leadingRev.reverse`, and this gives the exact
equality needed to hand off to the tail-induction hypothesis.
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
    (aId bId : Nat) (leadingRev : List Nat) (rows inner cols : Nat)
    (hp : binaryParents? n.parents = some (aId, bId))
    (ia : Idx ([inShape] ++ ss) ((Shape.ofList leadingRev.reverse).concat [rows, inner]))
    (hIa : mkIdx (inShape := inShape) (ss := ss) aId
      ((Shape.ofList leadingRev.reverse).concat [rows, inner]) = .ok ia)
    (ib : Idx ([inShape] ++ ss) ((Shape.ofList leadingRev.reverse).concat [inner, cols]))
    (hIb : mkIdx (inShape := inShape) (ss := ss) bId
      ((Shape.ofList leadingRev.reverse).concat [inner, cols]) = .ok ib)
    (hOut : (Shape.ofList leadingRev.reverse).concat [rows, cols] = n.outShape)
    (hBuildNext :
      buildFrom (α := α) (g := g) (payload := payload)
        (inShape := inShape) (i := i + 1)
        (st := (⟨ss ++ [n.outShape],
          .snoc (ss := ss) gd
            (mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
              hOut ▸ NN.IR.Graph.matmulLeading (α := α) (Shape.ofList leadingRev.reverse)
                (getIdx (α := α) (xs := ctx) ia) (getIdx (α := α) (xs := ctx) ib)))⟩ :
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
  let leading : Shape := Shape.ofList leadingRev.reverse
  let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
    mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
      hOut ▸ NN.IR.Graph.matmulLeading (α := α) leading
        (getIdx (α := α) (xs := ctx) ia) (getIdx (α := α) (xs := ctx) ib))
  have hRec :
      buildFrom (α := α) (g := g) (payload := payload)
        (inShape := inShape) (i := i + 1)
        (st := (⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩ : State α inShape)) =
          .ok st' := by
    simpa [leading, nodeData] using hBuildNext
  have hGetA :
      vals0[aId]? = some
        (Spec.SomeTensor.mk (α := α) (leading.concat [rows, inner])
          (getIdx (α := α) (xs := ctx) ia)) := by
    simpa [vals0, ctx, leading] using
      (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
        (gd := gd) (x := x) (pid := aId)
        (s := leading.concat [rows, inner]) (idx := ia) hIa)
  have hGetB :
      vals0[bId]? = some
        (Spec.SomeTensor.mk (α := α) (leading.concat [inner, cols])
          (getIdx (α := α) (xs := ctx) ib)) := by
    simpa [vals0, ctx, leading] using
      (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
        (gd := gd) (x := x) (pid := bId)
        (s := leading.concat [inner, cols]) (idx := ib) hIb)
  have hEval :
      NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
        (input := input) (vals := vals0) (i := i) =
        .ok (Spec.SomeTensor.mk (α := α) n.outShape (nodeData.eval ctx)) := by
    simpa [nodeData, leading, mkForwardNode] using
      (evalAt_matmul_leading_ok (α := α)
        (g := g) (payload := payload) (input := input) (vals := vals0)
        (i := i) (n := n) (aId := aId) (bId := bId)
        (leadingRev := leadingRev) (rows := rows) (inner := inner) (cols := cols)
        (aT := getIdx (α := α) (xs := ctx) ia)
        (bT := getIdx (α := α) (xs := ctx) ib)
        hN hk hp hGetA hGetB hOut)
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

The proof mirrors `lowerMatmul`: it reads both parent shapes as reversed dimension lists, follows
the contract and typed-shape checks, and hands the typed operands to
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
  simp [hi, hN] at hBuild
  simp (config := { failIfUnchanged := false }) [hk, lowerMatmul] at hBuild
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
  cases hExpected : OpContracts.inferMatmulOutShape aNode.outShape bNode.outShape with
  | error msg => simp [hp, hA, hB, hExpected] at hBuild
  | ok expected =>
  simp (config := { failIfUnchanged := false }) [hp, hA, hB, hExpected] at hBuild
  -- Decompose both parent shapes as the lowering does.
  cases hAS : aNode.outShape.toList.reverse with
  | nil => exact False.elim <| throw_bind_ne_ok (by simpa [hAS] using hBuild)
  | cons inner aTail =>
  cases aTail with
  | nil => exact False.elim <| throw_bind_ne_ok (by simpa [hAS] using hBuild)
  | cons rows leadingRev =>
  cases hBS : bNode.outShape.toList.reverse with
  | nil => exact False.elim <| throw_bind_ne_ok (by simpa [hAS, hBS] using hBuild)
  | cons cols bTail =>
  cases bTail with
  | nil => exact False.elim <| throw_bind_ne_ok (by simpa [hAS, hBS] using hBuild)
  | cons inner' leadingRev' =>
  simp (config := { failIfUnchanged := false }) [hAS, hBS] at hBuild
  by_cases hLeading : leadingRev = leadingRev'
  · by_cases hInner : inner = inner'
    · subst hLeading
      subst hInner
      simp (config := { failIfUnchanged := false }) at hBuild
      cases hIa :
          mkIdx (inShape := inShape) (ss := ss) aId
            ((Shape.ofList leadingRev.reverse).concat [rows, inner]) with
      | error msg => simp [hIa] at hBuild
      | ok ia =>
      cases hIb :
          mkIdx (inShape := inShape) (ss := ss) bId
            ((Shape.ofList leadingRev.reverse).concat [inner, cols]) with
      | error msg => simp [hIa, hIb] at hBuild
      | ok ib =>
      by_cases hExpectedEq : (Shape.ofList leadingRev.reverse).concat [rows, cols] = expected
      · by_cases hOut : expected = n.outShape
        · have hOut' : (Shape.ofList leadingRev.reverse).concat [rows, cols] = n.outShape :=
            hExpectedEq.trans hOut
          have hBuildNext := by
            simpa [hIa, hIb, hExpectedEq, hOut, Tensor.eqRec_eq_cast_shape] using hBuild
          exact buildFrom_denoteAllFrom_matmul_success (α := α)
            (g := g) (payload := payload) (gd := gd) (i := i) (st' := st')
            (x := x) (n := n) hN hk hi ih
            (aId := aId) (bId := bId) (leadingRev := leadingRev)
            (rows := rows) (inner := inner) (cols := cols)
            hp (ia := ia) hIa (ib := ib) hIb hOut'
            (by simpa [Tensor.eqRec_eq_cast_shape] using hBuildNext)
        · exact False.elim <|
            throw_bind_ne_ok (by simpa [hIa, hIb, hExpectedEq, hOut] using hBuild)
      · exact False.elim <|
          throw_bind_ne_ok (by simpa [hIa, hIb, hExpectedEq] using hBuild)
    · exact False.elim <| throw_bind_ne_ok (by simpa [hLeading, hInner] using hBuild)
  · exact False.elim <| throw_bind_ne_ok (by simpa [hLeading] using hBuild)

end IRExec
end Autograd
end Runtime
