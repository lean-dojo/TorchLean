/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tape.RealNode.Elementwise
public import TorchLean.NN.Tape.RealNode.Reductions

/-!
# Differentiable graph composition

The `DGraph` wrapper packages a tape graph `Tape.Graph (α := ℝ) Unit Γ ss` together with node-local
`RealNode.FDerivCorrect` proofs at the environment `()`. A graph is built incrementally with
`DGraph.snoc` and `DGraph.weakenContext`, after which `Tape.Graph.backpropVec_eq_fderiv` applies to
`dg.g` with the bundled proof read at the input, `dg.hg.at xV`. The VJP theorem for a whole model
therefore comes from the certificates of its primitive nodes.
-/

@[expose] public section

namespace TorchLean

open TorchLean.Tensor

noncomputable section

open scoped BigOperators

/-- A tape graph bundled with the proof that every one of its nodes is analytically correct. -/
structure Tape.DGraph (Γ : List Shape) (ss : List Shape) where
  /-- The underlying tape/DAG graph. -/
  g : Graph (α := ℝ) Unit Γ ss
  /-- Proof that every node is analytically correct (`jvp = fderiv`). -/
  hg : RealGraph.FDerivCorrect (Γ := Γ) g.toData ()

namespace Tape.DGraph

/--
Projection that drops the middle block `extra` from a context `(Γ ++ extra) ++ ss`.

When a graph `dg : DGraph Γ ss` is reused in the larger input context `Γ ++ extra`, a node written
for `Γ ++ ss` runs in `(Γ ++ extra) ++ ss`. The projection keeps the inputs `Γ` and the
intermediates `ss` that the node reads and discards the carried inputs in `extra`.
-/
def dropMiddleCLM (Γ extra ss : List Shape) :
    CtxVec ((Γ ++ extra) ++ ss) →L[ℝ] CtxVec (Γ ++ ss) :=
  let splitAll : CtxVec ((Γ ++ extra) ++ ss) →L[ℝ] CtxVec (Γ ++ extra) × CtxVec ss :=
    (EuclideanSpace.finAddEquivProd (𝕜 := ℝ)).toContinuousLinearMap.comp
      (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ
        (finCongr (ctxSize_append (Γ ++ extra) ss))).toContinuousLinearEquiv.toContinuousLinearMap
  let baseExtra := (ContinuousLinearMap.fst ℝ (CtxVec (Γ ++ extra)) (CtxVec ss)).comp splitAll
  let saved := (ContinuousLinearMap.snd ℝ (CtxVec (Γ ++ extra)) (CtxVec ss)).comp splitAll
  let splitBaseExtra : CtxVec (Γ ++ extra) →L[ℝ] CtxVec Γ × CtxVec extra :=
    (EuclideanSpace.finAddEquivProd (𝕜 := ℝ)).toContinuousLinearMap.comp
      (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ
        (finCongr (ctxSize_append Γ extra))).toContinuousLinearEquiv.toContinuousLinearMap
  let base :=
    (ContinuousLinearMap.fst ℝ (CtxVec Γ) (CtxVec extra)).comp (splitBaseExtra.comp baseExtra)
  (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ
      (finCongr (ctxSize_append Γ ss).symm)).toContinuousLinearEquiv.toContinuousLinearMap.comp
    ((EuclideanSpace.finAddEquivProd (𝕜 := ℝ)).symm.toContinuousLinearMap.comp (base.prod saved))

/-- Unfolds the middle-dropping projection into coordinate casts and the components of
`EuclideanSpace.finAddEquivProd` and its inverse. -/
@[simp] theorem dropMiddleCLM_apply {Γ extra ss : List Shape}
    (x : CtxVec ((Γ ++ extra) ++ ss)) :
    dropMiddleCLM Γ extra ss x =
      LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr (ctxSize_append Γ ss).symm)
        ((EuclideanSpace.finAddEquivProd (𝕜 := ℝ)).symm
          ((EuclideanSpace.finAddEquivProd (𝕜 := ℝ)
              (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr (ctxSize_append Γ extra))
                (EuclideanSpace.finAddEquivProd (𝕜 := ℝ)
                  (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ
                    (finCongr (ctxSize_append (Γ ++ extra) ss)) x)).1)).1,
            (EuclideanSpace.finAddEquivProd (𝕜 := ℝ)
              (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ
                (finCongr (ctxSize_append (Γ ++ extra) ss)) x)).2)) := by
  rfl

/--
Reuse a node in a context that carries extra unused inputs between the original inputs and the
current SSA intermediates.

The node reads through `dropMiddleCLM` and its VJP goes back through the adjoint of that
projection. Gradients land in the original inputs and the earlier intermediates, while the
carried block `extra` gets zero from this node.
-/
def weakenNodeMiddle {Γ extra ss : List Shape} {τ : Shape}
    (node : LawfulNode (α := ℝ) Unit (Γ ++ ss) τ) :
    LawfulNode (α := ℝ) Unit ((Γ ++ extra) ++ ss) τ :=
  let L := dropMiddleCLM Γ extra ss
  LawfulNode.ofFn
    (Γ := (Γ ++ extra) ++ ss) (τ := τ)
    (f := fun x => node.forwardVec (Γ := Γ ++ ss) (τ := τ) () (L x))
    (jvp := fun x dx => node.jvpVec (Γ := Γ ++ ss) (τ := τ) () (L x) (L dx))
    (vjp := fun x δ => L.adjoint (node.vjpVec (Γ := Γ ++ ss) (τ := τ) () (L x) δ))
    (correct_inner := by
      intro x dx δ
      have hnode :=
        LawfulNode.jvpVec_vjpVec_adjoint (node := node) () (L x) (L dx) δ
      have hadj :
          inner ℝ dx (L.adjoint (node.vjpVec (Γ := Γ ++ ss) (τ := τ) () (L x) δ))
            =
          inner ℝ (L dx) (node.vjpVec (Γ := Γ ++ ss) (τ := τ) () (L x) δ) := by
        simpa using (ContinuousLinearMap.adjoint_inner_right (A := L) (x := dx)
          (y := node.vjpVec (Γ := Γ ++ ss) (τ := τ) () (L x) δ))
      exact hnode.trans hadj.symm)

/-- A global `RealNode.FDerivCorrect` certificate for `node` carries over to
`weakenNodeMiddle node`, with the derivative composed with `dropMiddleCLM`. -/
def weakenNodeMiddleFDeriv {Γ extra ss : List Shape} {τ : Shape}
    {node : LawfulNode (α := ℝ) Unit (Γ ++ ss) τ} (hn : RealNode.FDerivCorrect node.toNodeData ()) :
    RealNode.FDerivCorrect
      (weakenNodeMiddle (Γ := Γ) (extra := extra) (ss := ss) node).toNodeData () := by
  let L := dropMiddleCLM Γ extra ss
  refine
    { deriv := fun x => (hn.deriv (L x)).comp L
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · intro x
    have hnode :
        HasFDerivAt
          (fun y => node.forwardVec (Γ := Γ ++ ss) (τ := τ) () (L y))
          ((hn.deriv (L x)).comp L) x :=
      (hn.hasFDerivAt (L x)).comp x (L.hasFDerivAt (x := x))
    simpa [weakenNodeMiddle, LawfulNode.forwardVec_ofFn, L, -dropMiddleCLM_apply] using hnode
  · intro x dx
    simpa [weakenNodeMiddle, L, ContinuousLinearMap.comp_apply, -dropMiddleCLM_apply] using
      hn.jvp_eq (L x) (L dx)

/-- Empty differentiable graph. -/
def nil {Γ : List Shape} : DGraph Γ [] :=
  ⟨.nil, PUnit.unit⟩

/-- Append a node together with its `RealNode.FDerivCorrect` certificate. -/
def snoc {Γ : List Shape} {ss : List Shape} {τ : Shape}
    (dg : DGraph Γ ss) (node : LawfulNode (α := ℝ) Unit (Γ ++ ss) τ)
    (hn : RealNode.FDerivCorrect node.toNodeData ()) :
    DGraph Γ (ss ++ [τ]) :=
  ⟨.snoc dg.g node, ⟨dg.hg, hn⟩⟩

/-- Recursive implementation of `DGraph.weakenContext` on graph data `g` with its adjointness proof
`hc` and its derivative proof `hg`: each node is replaced by its `weakenNodeMiddle` form. -/
def _root_.TorchLean.Tape.RealGraph.weakenContext {Γ ss : List Shape} (extra : List Shape)
    (g : GraphData ℝ Unit Γ ss) (hc : g.Correct) (hg : RealGraph.FDerivCorrect (Γ := Γ) g ()) :
    DGraph (Γ ++ extra) ss := by
  induction g with
  | nil =>
      exact DGraph.nil
  | @snoc ssPrefix τ g node ih =>
      rcases hc with ⟨hcPrefix, hnode⟩
      rcases hg with ⟨hgPrefix, hn⟩
      let dgPrefix : DGraph (Γ ++ extra) ssPrefix :=
        ih hcPrefix hgPrefix
      exact DGraph.snoc
        (dg := dgPrefix)
        (node := weakenNodeMiddle (Γ := Γ) (extra := extra) (ss := ssPrefix) ⟨node, hnode⟩)
        (hn := weakenNodeMiddleFDeriv (Γ := Γ) (extra := extra) (ss := ssPrefix)
          (node := ⟨node, hnode⟩) hn)

/--
Run a proof-carrying graph in an input context with extra inputs.

If `dg : DGraph Γ ss`, then `weakenContext dg extra : DGraph (Γ ++ extra) ss` evaluates the same
nodes with the inputs `extra` carried along. Each node sees the projection `Γ ++ ss_so_far` of the
actual context `(Γ ++ extra) ++ ss_so_far`, and its gradient comes back through the adjoint of that
projection, so the carried inputs get zero gradient from every node of `dg`. A node appended later
with `Tape.DGraph.snoc` can read them, as the post-norm Transformer blocks do with the LayerNorm
parameters.
-/
def weakenContext {Γ ss : List Shape} (dg : DGraph Γ ss) (extra : List Shape) :
    DGraph (Γ ++ extra) ss :=
  RealGraph.weakenContext (Γ := Γ) (ss := ss) extra dg.g.toData dg.g.correct dg.hg

end Tape.DGraph

end

end TorchLean
