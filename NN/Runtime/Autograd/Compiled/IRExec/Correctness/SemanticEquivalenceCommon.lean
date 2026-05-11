/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.Common

/-!
# SemanticEquivalenceCommon

Shared helper lemmas for the semantic equivalence proof in `NN.Runtime.Autograd.Compiled.IRExec.Correctness`.

This module contains correctness infrastructure used by the end-to-end semantic equivalence proof. We keep
these lemmas separate from `...Correctness.Common` because they are specific to the recursive
`buildFrom` proof shape rather than generally useful per-op infrastructure.

Reading map:

* `denoteAllState_nil`: base case table lemma for the empty compiled prefix.
* `permuteDVal_eq_applySwapsTensor` (+ helper): connects IR permutation semantics to the runtime
  swap-based implementation.
* `buildFrom_denoteAllFrom_{unary,binary}_exact`: convenience wrappers that package the standard
  semantic equivalence proof pattern once a `nodeData` closure has been constructed.

These helpers exist because the main correctness proof is expensive to elaborate. They keep the
recursive theorem from re-solving the same typed-context and dynamic-value facts for every operator.
When a new proof starts to repeat parent lookup, `DVal` casting, or tail-of-graph preservation
steps, it usually belongs here.

Maintenance note: add focused, well-named lemmas here instead of adding another large
tactic block to the recursive proof.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Compiled

open Spec
open Tensor
open Proofs.Autograd.Algebra
open NN.IR
open IRExec

/-!
## Correctness (Semantic Equivalence Helpers)

These lemmas are shared by the branch-by-branch proof in `Correctness.SemanticEquivalence`.
-/

/-- Base case: evaluating the empty compiled prefix yields the singleton table containing just the
  input. -/
@[simp]
theorem denoteAllState_nil {α : Type} [Context α]
    {inShape : Shape} (x : Tensor α inShape) :
    denoteAllState (α := α) inShape (st := (⟨[], .nil⟩ : State α inShape)) x =
      #[NN.IR.DVal.mk (α := α) inShape x] := by
  simp [denoteAllState, execOfState, ExecGraphData.denoteAll, ExecGraphData.eval,
    dValsOfCtx, dValOfAny, NN.IR.DVal.mk, AnyTensor.mk, GraphData.eval,
    Proofs.Autograd.Algebra.TList.toAnyArray, Proofs.Autograd.Algebra.TList.toAnyList]

attribute [grind =] denoteAllState_nil

/--
Relate the IR permutation lowering (`applySwapDepth` folded over swap depths) to `applySwapsTensor`.

This is used to connect `NN.IR.Graph.permuteDVal` to the runtime implementation used by the
compiler-generated node.
-/
theorem applySwapsTensor_eq_foldl_applySwapDepth
    {α : Type} [Context α] {s : Shape} (t : Tensor α s) (swaps : List Nat) :
    (swaps.foldl (fun acc d => NN.IR.Graph.applySwapDepth (α := α) acc d) (NN.IR.DVal.mk (α := α) s
      t)) =
      NN.IR.DVal.mk (α := α) (swapShapeBySwaps s swaps)
        (applySwapsTensor (α := α) (s := s) (swaps := swaps) t) := by
  induction swaps generalizing s t with
  | nil =>
      simp [swapShapeBySwaps, applySwapsTensor]
  | cons d ds ih =>
      -- Expand one swap step (foldl) and apply the IH to the updated shape/tensor.
      simp [List.foldl, swapShapeBySwaps, applySwapsTensor, NN.IR.Graph.applySwapDepth]
      -- The accumulator after one `applySwapDepth` is definitional.
      simpa using (ih (s := s.swapAdjacentAtDepth d) (t := Tensor.swapAtDepthHelper (tensor := t)
        d))

/--
Specialize `NN.IR.Graph.permuteDVal` to the swap-based implementation used by this compiler.

Given a successful `permute?` witness and computed swap depths, `permuteDVal` returns the tensor
produced by `applySwapsTensor` with the expected final shape.
-/
theorem permuteDVal_eq_applySwapsTensor
    {α : Type} [Context α] {sIn : Shape} (t : Tensor α sIn)
    (perm : List Nat) (expected : Shape) (swaps : List Nat)
    (hPerm : Spec.Shape.permute? sIn perm = some expected)
    (hSwaps : NN.IR.Graph.swapDepthsForPerm perm (Shape.rank sIn) = .ok swaps) :
    NN.IR.Graph.permuteDVal (α := α) (v := NN.IR.DVal.mk (α := α) sIn t) perm =
      .ok (NN.IR.DVal.mk (α := α) (swapShapeBySwaps sIn swaps)
        (applySwapsTensor (α := α) (s := sIn) (swaps := swaps) t)) := by
  -- Unfold `permuteDVal`, then rewrite by the permutation witness and swap computation.
  unfold NN.IR.Graph.permuteDVal
  simp [NN.IR.DVal.shape, NN.IR.DVal.mk, hPerm, hSwaps]
  -- Reduce the foldl over `applySwapDepth` to the recursive `applySwapsTensor`.
  simpa [NN.IR.DVal.mk] using
    (applySwapsTensor_eq_foldl_applySwapDepth (α := α) (s := sIn) (t := t) (swaps := swaps))

-- `permuteDVal_eq_applySwapsTensor` carries extra witness parameters (`expected`, `swaps`) that are
-- not uniquely determined from the left-hand side, so it cannot be registered as a `grind` rule.
attribute [grind =] applySwapsTensor_eq_foldl_applySwapDepth

/--
Semantic Equivalence lemma for a unary op compilation step (already-compiled `nodeData`).

This packages the common pattern for unary operators: compile the node, compute its forward value,
push it onto the value table, and appeal to the inductive hypothesis for the tail.
-/
theorem buildFrom_denoteAllFrom_unary_exact
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss)
    (i : Nat) (st' : State α inShape) (x : Tensor α inShape)
    (hi : i < g.nodes.size)
    (τ : Shape)
    (nodeData : NodeData α Unit ([inShape] ++ ss) τ)
    (hTail :
      NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := NN.IR.DVal.mk (α := α) inShape x) (i := i + 1)
          (vals := denoteAllState (α := α) inShape
            (st := (⟨ss ++ [τ], .snoc (ss := ss) gd nodeData⟩ : State α inShape)) x) =
        .ok (denoteAllState (α := α) inShape st' x))
    (hEval :
      NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
          (input := NN.IR.DVal.mk (α := α) inShape x)
          (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x)
          (i := i) =
        .ok
          (NN.IR.DVal.mk (α := α) τ
            (nodeData.forward
              (GraphData.eval (α := α) (Δ := Unit) (Γ := [inShape]) (ss := ss) gd (.cons x .nil) ())
                ()))) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
        (input := NN.IR.DVal.mk (α := α) inShape x) (i := i)
        (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (NN.IR.DVal α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TList α ([inShape] ++ ss) :=
    GraphData.eval (α := α) (Δ := Unit) (Γ := [inShape]) (ss := ss) gd (.cons x .nil) ()
  let input : NN.IR.DVal α := NN.IR.DVal.mk (α := α) inShape x
  have hStep :
      denoteAllState (α := α) inShape
          (st := (⟨ss ++ [τ], .snoc (ss := ss) gd nodeData⟩ : State α inShape)) x =
        vals0.push (NN.IR.DVal.mk (α := α) τ (nodeData.forward ctx ())) := by
    simpa [vals0, ctx] using
      (denoteAllState_snoc (α := α) (inShape := inShape) (ss := ss) (τ := τ)
        (gd := gd) (nodeData := nodeData) (x := x))
  exact buildFrom_denoteAllFrom_finish (α := α) (g := g) (payload := payload)
    (i := i) (x := x) (hi := hi) (τ := τ) (nodeData := nodeData)
    (st1 := ⟨ss ++ [τ], .snoc (ss := ss) gd nodeData⟩) (st' := st')
    (ctx := ctx) (vals0 := vals0) (input := input) hTail hEval hStep

/--
Semantic Equivalence lemma for a binary op compilation step (already-compiled `nodeData`).

This is the two-parent analogue of `buildFrom_denoteAllFrom_unary_exact`.
-/
theorem buildFrom_denoteAllFrom_binary_exact
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss)
    (i : Nat) (st' : State α inShape) (x : Tensor α inShape)
    (hi : i < g.nodes.size)
    (τ : Shape)
    (nodeData : NodeData α Unit ([inShape] ++ ss) τ)
    (hTail :
      NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := NN.IR.DVal.mk (α := α) inShape x) (i := i + 1)
          (vals := denoteAllState (α := α) inShape
            (st := (⟨ss ++ [τ], .snoc (ss := ss) gd nodeData⟩ : State α inShape)) x) =
        .ok (denoteAllState (α := α) inShape st' x))
    (hEval :
      NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
          (input := NN.IR.DVal.mk (α := α) inShape x)
          (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x)
          (i := i) =
        .ok
          (NN.IR.DVal.mk (α := α) τ
            (nodeData.forward
              (GraphData.eval (α := α) (Δ := Unit) (Γ := [inShape]) (ss := ss) gd (.cons x .nil) ())
                ()))) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
        (input := NN.IR.DVal.mk (α := α) inShape x) (i := i)
        (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (NN.IR.DVal α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TList α ([inShape] ++ ss) :=
    GraphData.eval (α := α) (Δ := Unit) (Γ := [inShape]) (ss := ss) gd (.cons x .nil) ()
  let input : NN.IR.DVal α := NN.IR.DVal.mk (α := α) inShape x
  have hStep :
      denoteAllState (α := α) inShape
          (st := (⟨ss ++ [τ], .snoc (ss := ss) gd nodeData⟩ : State α inShape)) x =
        vals0.push (NN.IR.DVal.mk (α := α) τ (nodeData.forward ctx ())) := by
    simpa [vals0, ctx] using
      (denoteAllState_snoc (α := α) (inShape := inShape) (ss := ss) (τ := τ)
        (gd := gd) (nodeData := nodeData) (x := x))
  exact buildFrom_denoteAllFrom_finish (α := α) (g := g) (payload := payload)
    (i := i) (x := x) (hi := hi) (τ := τ) (nodeData := nodeData)
    (st1 := ⟨ss ++ [τ], .snoc (ss := ss) gd nodeData⟩) (st' := st')
    (ctx := ctx) (vals0 := vals0) (input := input) hTail hEval hStep

/--
If a dynamic value `v` is definitionally a dependent pair `⟨s, t⟩`, then extracting its second
component requires a transport across the (propositional) shape equality.

This lemma packages the standard `Sigma`-transport pattern used throughout the semantic equivalence
proof: when `v = DVal.mk s t`, transporting `v.snd` across the induced shape equality yields `t`.
-/
theorem dval_fst_eq_of_eq_mk
    {α : Type} [Context α]
    {v : NN.IR.DVal α} {s : Shape} {t : Tensor α s}
    (h : v = NN.IR.DVal.mk (α := α) s t) :
    v.fst = s := by
  cases h
  rfl

/--
Transport the tensor component of a dynamic value across the induced shape equality.

We spell this as a separate theorem (rather than using `by simpa ...` inside the *statement*) so
that the module system does not need to elaborate a tactic block in a public header.
-/
theorem dval_snd_cast_of_eq_mk
    {α : Type} [Context α]
    {v : NN.IR.DVal α} {s : Shape} {t : Tensor α s}
    (h : v = NN.IR.DVal.mk (α := α) s t) :
    (dval_fst_eq_of_eq_mk (α := α) (v := v) (s := s) (t := t) h) ▸ v.snd = t := by
  cases h
  rfl

/-!
## Boolean Shape Equality Helpers

The IR evaluator uses *boolean* equality/inequality checks on `Shape` (via `BEq Shape`) in a few
places. For example, `evalAt`'s `.conv2d` case checks a computed output shape against the node's
declared `outShape` using `!=` (rather than a propositional `≠`) because it is part of the
runtime error-reporting path.

In the proof layer we frequently have a propositional equality `s = t` and need to discharge such
boolean guards. Since `BEq Shape` is defined as an explicit structural test (`Shape.areEqual`) and
we do not globally assume `LawfulBEq Shape`, we prove the small bridge lemmas locally here.
-/

/-- Reflexivity of the explicit structural boolean equality test `Shape.areEqual`. -/
theorem shape_areEqual_refl (s : Shape) : Shape.areEqual s s = true := by
  induction s with
  | scalar => rfl
  | dim n s ih =>
      simp [Shape.areEqual, ih]

/-- Reflexivity of `BEq Shape` (`==`). -/
theorem shape_beq_refl (s : Shape) : (s == s) = true := by
  -- `==` is definitionally `BEq.beq`, and `BEq Shape` is `Shape.areEqual`.
  simpa [BEq.beq] using shape_areEqual_refl (s := s)

/-- Reflexivity of boolean inequality (`!=`) on shapes. -/
theorem shape_bne_refl (s : Shape) : (s != s) = false := by
  -- `!=` is the boolean negation of `==`.
  simp [bne, shape_beq_refl (s := s)]

/-- Propositional shape equality implies the boolean inequality guard `s != t` is false. -/
theorem shape_bne_eq_false_of_eq {s t : Shape} (h : s = t) : (s != t) = false := by
  cases h
  simpa using shape_bne_refl (s := s)

end Compiled
end Autograd
end Runtime
