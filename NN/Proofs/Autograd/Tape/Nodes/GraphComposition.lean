/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Nodes.Piecewise

/-!
# Differentiable graph composition

The `DGraph` wrapper packages a tape graph together with node-local `NodeFDerivCorrect` proofs.
Composition lemmas here let us build large model-level VJP theorems from proved primitive nodes rather
than reproving backprop correctness for each architecture from scratch.
-/

@[expose] public section

namespace Proofs
namespace Autograd

open Spec
open Tensor

noncomputable section

open scoped BigOperators

open Graph

/-!
`DGraph` (“differentiable graph”) is a small wrapper bundling a `Graph` together with proofs that
every node in it satisfies `NodeFDerivCorrect`.

This is a convenience for end-to-end examples: you can build a graph incrementally with `snoc`,
and then immediately use `backpropVec_eq_adjoint_fderiv` without separately threading a proof
object.
-/

structure DGraph (Γ : List Shape) (ss : List Shape) where
  /-- The underlying tape/DAG graph. -/
  g : Graph Γ ss
  /-- Proof that every node is analytically correct (`jvp = fderiv`). -/
  hg : GraphFDerivCorrect (Γ := Γ) g

namespace DGraph

/-- Continuous linear map taking the left block of a concatenated Euclidean vector. -/
def takeLeftCLM (m n : Nat) : Vec (m + n) →L[ℝ] Vec m := by
  classical
  let fLin : Vec (m + n) →ₗ[ℝ] Vec m :=
    { toFun := TapeNodes.takeLeftVec (m := m) (n := n)
      map_add' := by
        intro x y
        ext i
        simp [TapeNodes.takeLeftVec]
      map_smul' := by
        intro r x
        ext i
        simp [TapeNodes.takeLeftVec] }
  refine { toLinearMap := fLin, cont := ?_ }
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

/-- Continuous linear map taking the right block of a concatenated Euclidean vector. -/
def takeRightCLM (m n : Nat) : Vec (m + n) →L[ℝ] Vec n := by
  classical
  let fLin : Vec (m + n) →ₗ[ℝ] Vec n :=
    { toFun := TapeNodes.takeRightVec (m := m) (n := n)
      map_add' := by
        intro x y
        ext i
        simp [TapeNodes.takeRightVec]
      map_smul' := by
        intro r x
        ext i
        simp [TapeNodes.takeRightVec] }
  refine { toLinearMap := fLin, cont := ?_ }
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

/--
Drop an unused middle context block from a graph node context.

When a graph `dg : DGraph Γ ss` is reused inside a larger context `Γ ++ extra`, a node that
originally reads `Γ ++ ss` is evaluated in the actual context `(Γ ++ extra) ++ ss`. This projection
keeps the original inputs `Γ` and the already-computed intermediates `ss`, and ignores the carried
parameters in `extra`.
-/
def dropMiddleCLM (Γ extra ss : List Shape) :
    CtxVec ((Γ ++ extra) ++ ss) →L[ℝ] CtxVec (Γ ++ ss) :=
  let splitAll :=
    (Graph.castCLM (h := ctxSize_append (Γ ++ extra) ss) :
      CtxVec ((Γ ++ extra) ++ ss) →L[ℝ] Vec (ctxSize (Γ ++ extra) + ctxSize ss))
  let baseExtra :=
    (takeLeftCLM (ctxSize (Γ ++ extra)) (ctxSize ss)).comp splitAll
  let saved :=
    (takeRightCLM (ctxSize (Γ ++ extra)) (ctxSize ss)).comp splitAll
  let splitBaseExtra :=
    (Graph.castCLM (h := ctxSize_append Γ extra) :
      CtxVec (Γ ++ extra) →L[ℝ] Vec (ctxSize Γ + ctxSize extra))
  let base :=
    (takeLeftCLM (ctxSize Γ) (ctxSize extra)).comp (splitBaseExtra.comp baseExtra)
  (Graph.castCLM (h := (ctxSize_append Γ ss).symm)).comp
    ((Graph.appendCLM (ctxSize Γ) (ctxSize ss)).comp (base.prod saved))

@[simp] lemma dropMiddleCLM_apply {Γ extra ss : List Shape}
    (x : CtxVec ((Γ ++ extra) ++ ss)) :
    dropMiddleCLM Γ extra ss x =
      castVec (ctxSize_append Γ ss).symm
        (appendVec
          (m := ctxSize Γ) (n := ctxSize ss)
          (TapeNodes.takeLeftVec (m := ctxSize Γ) (n := ctxSize extra)
            (castVec (ctxSize_append Γ extra)
              (TapeNodes.takeLeftVec (m := ctxSize (Γ ++ extra)) (n := ctxSize ss)
                (castVec (ctxSize_append (Γ ++ extra) ss) x))))
          (TapeNodes.takeRightVec (m := ctxSize (Γ ++ extra)) (n := ctxSize ss)
            (castVec (ctxSize_append (Γ ++ extra) ss) x))) := by
  rfl

/--
Reuse a node in a context that carries extra unused inputs between the original inputs and the
current SSA intermediates.

The VJP is obtained by applying the adjoint of `dropMiddleCLM`, so gradients land only in the
original inputs and previous intermediates; the extra carried parameters receive zero contribution
from this reused node.
-/
def weakenNodeMiddle {Γ extra ss : List Shape} {τ : Shape}
    (node : Node (Γ ++ ss) τ) : Node ((Γ ++ extra) ++ ss) τ :=
  let L := dropMiddleCLM Γ extra ss
  Node.ofVec
    (Γ := (Γ ++ extra) ++ ss) (τ := τ)
    (f := fun x => node.forwardVec (Γ := Γ ++ ss) (τ := τ) (L x))
    (jvp := fun x dx => node.jvpVec (Γ := Γ ++ ss) (τ := τ) (L x) (L dx))
    (vjp := fun x δ => L.adjoint (node.vjpVec (Γ := Γ ++ ss) (τ := τ) (L x) δ))
    (correct_inner := by
      intro x dx δ
      have hnode :=
        Node.correct_inner (node := node) (L x) (L dx) δ
      have hadj :
          inner ℝ dx (L.adjoint (node.vjpVec (Γ := Γ ++ ss) (τ := τ) (L x) δ))
            =
          inner ℝ (L dx) (node.vjpVec (Γ := Γ ++ ss) (τ := τ) (L x) δ) := by
        simpa using (ContinuousLinearMap.adjoint_inner_right (A := L) (x := dx)
          (y := node.vjpVec (Γ := Γ ++ ss) (τ := τ) (L x) δ))
      exact hnode.trans hadj.symm)

/-- Transport a global node derivative certificate across `weakenNodeMiddle`. -/
def weakenNodeMiddleFDerivCorrect {Γ extra ss : List Shape} {τ : Shape}
    {node : Node (Γ ++ ss) τ} (hn : NodeFDerivCorrect node) :
    NodeFDerivCorrect (weakenNodeMiddle (Γ := Γ) (extra := extra) (ss := ss) node) := by
  let L := dropMiddleCLM Γ extra ss
  refine
    { deriv := fun x => (hn.deriv (L x)).comp L
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · intro x
    have hnode := (hn.hasFDerivAt (L x)).comp x (L.hasFDerivAt (x := x))
    simpa [weakenNodeMiddle, L] using hnode
  · intro x dx
    simpa [weakenNodeMiddle, L, ContinuousLinearMap.comp_apply] using hn.jvp_eq (L x) (L dx)

/-- Empty differentiable graph. -/
def nil {Γ : List Shape} : DGraph Γ [] :=
  ⟨.nil, PUnit.unit⟩

/-- Append a node together with its `NodeFDerivCorrect` certificate. -/
def snoc {Γ : List Shape} {ss : List Shape} {τ : Shape}
    (dg : DGraph Γ ss) (node : Node (Γ ++ ss) τ) (hn : NodeFDerivCorrect node) :
    DGraph Γ (ss ++ [τ]) :=
  ⟨.snoc dg.g node, ⟨dg.hg, hn⟩⟩

/--
Transport a node across a definitional/context-list equality.

This is mostly used by graph composition: the second graph sees its context as `(Γ ++ ss₁) ++ ss₂`,
while the composed graph sees the same values as `Γ ++ (ss₁ ++ ss₂)`.
-/
def castNodeContext {Γ₁ Γ₂ : List Shape} {τ : Shape}
    (h : Γ₁ = Γ₂) (node : Node Γ₁ τ) : Node Γ₂ τ := by
  subst h
  exact node

/-- Transport a node F-derivative certificate along `castNodeContext`. -/
def castNodeFDerivCorrect {Γ₁ Γ₂ : List Shape} {τ : Shape}
    (h : Γ₁ = Γ₂) {node : Node Γ₁ τ} (hn : NodeFDerivCorrect node) :
    NodeFDerivCorrect (castNodeContext (τ := τ) h node) := by
  subst h
  exact hn

/--
Specialize an everywhere-correct graph proof to a pointwise graph proof.

We use this when a globally smooth block feeds a pointwise block such as LayerNorm. The first block
does not need any domain hypotheses, so its `GraphFDerivCorrect` certificate can be read at the
actual runtime point.
-/
def graphFDerivCorrectAtOfCorrect {Γ : List Shape} {ss : List Shape} {g : Graph Γ ss}
    (hg : GraphFDerivCorrect (Γ := Γ) g) (xV : CtxVec Γ) :
    GraphFDerivCorrectAt (Γ := Γ) (ss := ss) g xV := by
  induction g with
  | nil =>
      exact PUnit.unit
  | @snoc ss τ g node ih =>
      rcases hg with ⟨hgPrefix, hn⟩
      exact ⟨ih hgPrefix, NodeFDerivCorrect.at hn (Graph.evalVec (Γ := Γ) (ss := ss) g xV)⟩

/-- Recursive implementation for `append`, stated over an explicit graph and proof. -/
def appendCore {Γ : List Shape} {ss₁ ss₂ : List Shape}
    (dg₁ : DGraph Γ ss₁)
    (g₂ : Graph (Γ ++ ss₁) ss₂) (hg₂ : GraphFDerivCorrect (Γ := Γ ++ ss₁) g₂) :
    DGraph Γ (ss₁ ++ ss₂) := by
  induction g₂ with
  | nil =>
      simpa using dg₁
  | @snoc ss τ g node ih =>
      rcases hg₂ with ⟨hg, hn⟩
      let dgPrefix : DGraph Γ (ss₁ ++ ss) := ih hg
      let hctx : (Γ ++ ss₁) ++ ss = Γ ++ (ss₁ ++ ss) := by
        simp [List.append_assoc]
      let node' : Node (Γ ++ (ss₁ ++ ss)) τ := castNodeContext (τ := τ) hctx node
      let hn' : NodeFDerivCorrect node' := castNodeFDerivCorrect (τ := τ) hctx hn
      have htarget : (ss₁ ++ ss) ++ [τ] = ss₁ ++ (ss ++ [τ]) := by
        simp [List.append_assoc]
      exact htarget ▸ DGraph.snoc (dg := dgPrefix) (node := node') (hn := hn')

/--
Append a proof-carrying graph after another proof-carrying graph.

If `dg₁ : DGraph Γ ss₁` has already computed some SSA values, then a second graph
`dg₂ : DGraph (Γ ++ ss₁) ss₂` may use both the original inputs and those saved values. `append`
turns the pair into one `DGraph Γ (ss₁ ++ ss₂)`.

This is the general composition adapter needed for model-level proofs: residual attention can feed
LayerNorm, a recurrent cell can feed the next unrolled step, and larger blocks can be assembled
while reusing the existing node-level correctness proofs.
-/
def append {Γ : List Shape} {ss₁ ss₂ : List Shape}
    (dg₁ : DGraph Γ ss₁) (dg₂ : DGraph (Γ ++ ss₁) ss₂) :
    DGraph Γ (ss₁ ++ ss₂) :=
  appendCore (Γ := Γ) (ss₁ := ss₁) (ss₂ := ss₂) dg₁ dg₂.g dg₂.hg

/-- Recursive implementation for `weakenContext`, stated over an explicit graph and proof. -/
def weakenContextCore {Γ ss : List Shape} (extra : List Shape)
    (g : Graph Γ ss) (hg : GraphFDerivCorrect (Γ := Γ) g) :
    DGraph (Γ ++ extra) ss := by
  induction g with
  | nil =>
      exact DGraph.nil
  | @snoc ssPrefix τ g node ih =>
      rcases hg with ⟨hgPrefix, hn⟩
      let dgPrefix : DGraph (Γ ++ extra) ssPrefix :=
        ih hgPrefix
      exact DGraph.snoc
        (dg := dgPrefix)
        (node := weakenNodeMiddle (Γ := Γ) (extra := extra) (ss := ssPrefix) node)
        (hn := weakenNodeMiddleFDerivCorrect (Γ := Γ) (extra := extra) (ss := ssPrefix) hn)

/--
Run a proof-carrying graph while carrying extra unused inputs.

If `dg : DGraph Γ ss`, then `weakenContext dg extra : DGraph (Γ ++ extra) ss` evaluates the same
nodes while preserving an enlarged input context. Each reused node sees the projection
`Γ ++ ss_so_far` of the actual context `(Γ ++ extra) ++ ss_so_far`; gradients are inserted back by
the adjoint projection, so the carried extras receive no gradient contribution from nodes that do
not read them.
-/
def weakenContext {Γ ss : List Shape} (dg : DGraph Γ ss) (extra : List Shape) :
    DGraph (Γ ++ extra) ss :=
  weakenContextCore (Γ := Γ) (ss := ss) extra dg.g dg.hg

/--
VJP theorem for context-weakened proof graphs.

The statement is intentionally direct: after threading unused inputs through the graph, the ordinary
`backprop = (fderiv eval)†` theorem still applies. The useful content lives in `weakenNodeMiddle`,
where unused parameters receive zero contribution by the adjoint of the drop-middle projection.
-/
theorem weakenContext_backpropVec_eq_adjoint_fderiv
    {Γ ss : List Shape} (dg : DGraph Γ ss) (extra : List Shape) :
    ∀ (xV : CtxVec (Γ ++ extra)) (seedV : CtxVec ((Γ ++ extra) ++ ss)),
      Graph.backpropVec
          (Γ := Γ ++ extra) (ss := ss) (weakenContext dg extra).g xV seedV
        =
      (fderiv ℝ
          (Graph.evalVec (Γ := Γ ++ extra) (ss := ss) (weakenContext dg extra).g) xV).adjoint
        seedV :=
  Graph.backpropVec_eq_adjoint_fderiv
    (Γ := Γ ++ extra) (ss := ss) (weakenContext dg extra).g (weakenContext dg extra).hg

/--
VJP theorem for appended proof-carrying graphs.

This is just `Graph.backpropVec_eq_adjoint_fderiv` specialized to `append`, but the named theorem
makes model proofs read like the construction we are formalizing: prove block A, prove block B over A's
extended context, append them, and immediately get the end-to-end reverse-mode theorem.
-/
theorem append_backpropVec_eq_adjoint_fderiv
    {Γ : List Shape} {ss₁ ss₂ : List Shape}
    (dg₁ : DGraph Γ ss₁) (dg₂ : DGraph (Γ ++ ss₁) ss₂) :
    ∀ (xV : CtxVec Γ) (seedV : CtxVec (Γ ++ (ss₁ ++ ss₂))),
      Graph.backpropVec (Γ := Γ) (ss := ss₁ ++ ss₂) (append dg₁ dg₂).g xV seedV
        =
      (fderiv ℝ (Graph.evalVec (Γ := Γ) (ss := ss₁ ++ ss₂) (append dg₁ dg₂).g) xV).adjoint
        seedV :=
  Graph.backpropVec_eq_adjoint_fderiv
    (Γ := Γ) (ss := ss₁ ++ ss₂) (append dg₁ dg₂).g (append dg₁ dg₂).hg

/--
Helper: append a unary op specified by an `OpSpecFDerivCorrect` proof object.

This is the common pattern for parameterized ops such as `linear`.
-/
def snocUnaryOp {Γ : List Shape} {ss : List Shape} {inDim outDim : Nat}
    (dg : DGraph Γ ss) (idx : Idx (Γ ++ ss) (.dim inDim .scalar))
    (C : OpSpecFDerivCorrect inDim outDim) :
    DGraph Γ (ss ++ [.dim outDim .scalar]) :=
  snoc (dg := dg)
    (node := TapeNodes.unaryOp (Γ := Γ ++ ss) (inDim := inDim) (outDim := outDim) idx C)
    (hn := TapeNodes.unaryOpFderiv (Γ := Γ ++ ss) (inDim := inDim) (outDim := outDim) idx C)

/--
End-to-end analytic theorem for bundled graphs.

This is just `Graph.backpropVec_eq_adjoint_fderiv` with the bundled proof `dg.hg`.
-/
theorem backpropVec_eq_adjoint_fderiv
    {Γ : List Shape} {ss : List Shape} (dg : DGraph Γ ss) :
    ∀ (xV : CtxVec Γ) (seedV : CtxVec (Γ ++ ss)),
      Graph.backpropVec (Γ := Γ) (ss := ss) dg.g xV seedV
        =
      (fderiv ℝ (Graph.evalVec (Γ := Γ) (ss := ss) dg.g) xV).adjoint seedV :=
  Graph.backpropVec_eq_adjoint_fderiv (Γ := Γ) (ss := ss) dg.g dg.hg

end DGraph

end

end Autograd
end Proofs
