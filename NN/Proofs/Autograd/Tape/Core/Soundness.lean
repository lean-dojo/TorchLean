/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.SpecialFunctions.Pow.NNReal
public import Mathlib.Data.Sym.Sym2.Init
import Mathlib.Tactic.NormNum.GCD
public import NN.Proofs.Tensor.Basic.BoundsNorms
-- Typed context indices are shared with the generic and runtime-approximation
-- graph developments, so `Idx` and `getIdx` come from one module.
public import NN.Proofs.Autograd.Tape.Util.Idx
public import NN.Proofs.Autograd.Tape.Algebra.Soundness

/-!
# Soundness

Tape-style (SSA/DAG) reverse-mode soundness for the proved-correct layer.

We model a dynamic graph as a sequence of nodes that may reference **any** previously computed
values (so sharing/fan-out is allowed). For each node we assume a local JVP/VJP adjointness law,
then prove the global reverse-mode accumulation algorithm is sound.

This is a proof-only layer; the runtime engine in `NN.Runtime.Autograd.Engine` is an
executable implementation of the same idea.

## PyTorch correspondence / citations
- This file is the proof analogue of PyTorch’s dynamic autograd engine building a tape of nodes
  during the forward pass and running a reverse pass that accumulates gradients at shared inputs.
  https://pytorch.org/docs/stable/autograd.html

References (background):
- Reverse-mode AD as backpropagation on a computation graph is standard; see e.g. Baydin et al.
  (JMLR 2018) for an overview and terminology (JVP/VJP, duality, etc.).
-/

@[expose] public section


namespace Proofs
namespace Autograd

open Spec TorchLean
open TorchLean TorchLean.Tensor

noncomputable section

namespace TensorPack

variable {ss : List Shape}

/--
Dot product over contexts: sum of per-entry tensor dot products.

Informally: `dotList xs ys` is the “context inner product” used to state global adjointness for
tape evaluation and backprop.
-/
def dotList : {ss : List Shape} → TorchLean.TensorPack ℝ ss → TorchLean.TensorPack ℝ ss → ℝ
  | [], .nil, .nil => 0
  | _ :: ss, .cons a as, .cons b bs => dot a b + dotList (ss := ss) as bs

/-- At `ℝ` the context dot product agrees with the backend-generic one in `Algebra`, entry by entry
through `dot_eq_tensorAlgebra_dot`. -/
theorem dotList_eq_algebra_dotList {Γ : List Shape} (x y : TorchLean.TensorPack ℝ Γ) :
    dotList (ss := Γ) x y = Algebra.TensorPack.dotList (α := ℝ) x y := by
  induction Γ with
  | nil =>
    cases x
    cases y
    rfl
  | cons s Γ ih =>
    cases x with
    | cons xh xt =>
      cases y with
      | cons yh yt =>
        simp [dotList, Algebra.TensorPack.dotList, dot_eq_tensorAlgebra_dot, ih]

end TensorPack

/-- A node with local JVP/VJP and an adjointness proof against the tensor dot product. -/
structure Node (Γ : List Shape) (τ : Shape) where
  /-- The node's forward pass, reading the whole context and producing one output tensor. -/
  forward : TorchLean.TensorPack ℝ Γ → Tensor ℝ τ
  /-- Forward-mode derivative: a basepoint and a tangent context give an output tangent. -/
  jvp : TorchLean.TensorPack ℝ Γ → TorchLean.TensorPack ℝ Γ → Tensor ℝ τ
  /-- Reverse-mode derivative: a basepoint and an output cotangent give a context cotangent. -/
  vjp : TorchLean.TensorPack ℝ Γ → Tensor ℝ τ → TorchLean.TensorPack ℝ Γ
  /-- `vjp` is the adjoint of `jvp` with respect to the tensor dot product. Every node in the tape
  carries this proof, so backpropagation over a whole graph is a composition of adjoints rather
  than a separate correctness argument. -/
  correct : ∀ x dx δ, dot (jvp x dx) δ = TensorPack.dotList dx (vjp x δ)
  /-- Runtime precondition metadata, preserved by the algebraic bridge; pure semantics ignore it. -/
  validate : TorchLean.TensorPack ℝ Γ → Except String Unit := fun _ => .ok ()
  /-- Certified local preparation, retained by both directions of the algebraic bridge. -/
  prepare? : Option {prepare : Algebra.TensorLookup ℝ Γ → Algebra.PreparedNode ℝ Γ τ //
    ∀ ctx, (prepare (Algebra.TensorLookup.ofPack ctx)).toPreparedPrograms =
      { value := fun _ => forward ctx, vjp := vjp ctx,
        validate := fun _ => validate ctx }} := none

/-- A tape/SSA graph: nodes are appended in topological order and may reference any previous value.
  -/
inductive Graph (Γ : List Shape) : List Shape → Type where
  | nil : Graph Γ []
  | snoc {ss : List Shape} {τ : Shape} :
      Graph Γ ss → Node (Γ ++ ss) τ → Graph Γ (ss ++ [τ])

namespace Graph

variable {Γ : List Shape}

/-- Evaluate a tape/graph, returning the full context (`inputs ++ intermediates`). -/
def eval {ss : List Shape} (g : Graph Γ ss) (x : TorchLean.TensorPack ℝ Γ) :
    TorchLean.TensorPack ℝ (Γ ++ ss) :=
  match g with
  | .nil => TorchLean.TensorPack.cast (h := (List.append_nil Γ).symm) x
  | .snoc (ss := ss) (τ := τ) g node =>
      let ctx := eval (ss := ss) g x
      let y := node.forward ctx
      TorchLean.TensorPack.cast (h := List.append_assoc Γ ss [τ]) (TorchLean.TensorPack.snoc ctx y)

/--
Evaluate the JVP (“forward-mode tangent”) of a graph, producing tangents for all values in the
extended context `Γ ++ ss`.
-/
def jvpCtx {ss : List Shape} (g : Graph Γ ss) (x : TorchLean.TensorPack ℝ Γ)
    (dx : TorchLean.TensorPack ℝ Γ) : TorchLean.TensorPack ℝ (Γ ++ ss) :=
  match g with
  | .nil => TorchLean.TensorPack.cast (h := (List.append_nil Γ).symm) dx
  | .snoc (ss := ss) (τ := τ) g node =>
      let ctx := eval (ss := ss) g x
      let dctx := jvpCtx (ss := ss) g x dx
      let dy := node.jvp ctx dctx
      TorchLean.TensorPack.cast (h := List.append_assoc Γ ss [τ])
        (TorchLean.TensorPack.snoc dctx dy)

/--
Reverse-mode backpropagation on a tape/graph, returning gradients for the *inputs* `Γ`.

This is the proof model of what PyTorch calls “running backward” starting from an output seed
cotangent and accumulating gradients at shared parents.
-/
def backpropCtx {ss : List Shape} (g : Graph Γ ss) (x : TorchLean.TensorPack ℝ Γ)
    (seed : TorchLean.TensorPack ℝ (Γ ++ ss)) : TorchLean.TensorPack ℝ Γ :=
  match g with
  | .nil => TorchLean.TensorPack.cast (h := List.append_nil Γ) seed
  | .snoc (ss := ss) (τ := τ) g node =>
      -- Reassociate the context so we can `unsnoc`.
      let seed' : TorchLean.TensorPack ℝ ((Γ ++ ss) ++ [τ]) :=
        TorchLean.TensorPack.cast (h := (List.append_assoc Γ ss [τ]).symm) seed
      let seedPrev : TorchLean.TensorPack ℝ (Γ ++ ss) :=
        (TorchLean.TensorPack.unsnoc (ss := Γ ++ ss) seed').1
      let seedOut : Tensor ℝ τ := (TorchLean.TensorPack.unsnoc (ss := Γ ++ ss) seed').2
      let ctx := eval (ss := ss) g x
      let contrib := node.vjp ctx seedOut
      let seedPrev' := TorchLean.TensorPack.add seedPrev contrib
      backpropCtx (ss := ss) g x seedPrev'

end Graph

/-- Embed an analytic node as an algebraic node over `ℝ` with a trivial environment. -/
def Node.toAlgebra {Γ : List Shape} {τ : Shape} (node : Node Γ τ) :
    Algebra.Node (α := ℝ) (Δ := Unit) (Γ := Γ) τ where
  validate x _ := node.validate x
  forward x _ := node.forward x
  jvp x dx _ := node.jvp x dx
  vjp x _ δ := node.vjp x δ
  prepare? := node.prepare?.map fun implementation =>
    ⟨fun lookup _ => implementation.val lookup, fun ctx _ => implementation.property ctx⟩
  correct x dx _ δ := by
    simpa [dot_eq_tensorAlgebra_dot, TensorPack.dotList_eq_algebra_dotList] using
      node.correct x dx δ

/-- Embed an analytic graph as an algebraic graph over `ℝ` with a trivial environment. -/
def Graph.toAlgebra {Γ : List Shape} :
    {ss : List Shape} → Graph Γ ss → Algebra.Graph (α := ℝ) (Δ := Unit) (Γ := Γ) ss
  | _, .nil => .nil
  | _, .snoc g node => .snoc (Graph.toAlgebra g) node.toAlgebra

namespace Graph

variable {Γ : List Shape}

/-- Embedding preserves evaluation. -/
theorem eval_toAlgebra {ss : List Shape} (g : Graph Γ ss) (x : TorchLean.TensorPack ℝ Γ) :
    Algebra.Graph.eval g.toAlgebra x () = eval g x := by
  induction g with
  | nil => rfl
  | snoc g node ih =>
    simp only [toAlgebra, Algebra.Graph.eval, Algebra.Graph.toData, Algebra.GraphData.eval,
      eval] at ih ⊢
    rw [ih]
    rfl

/-- Embedding preserves the JVP. -/
theorem jvpCtx_toAlgebra {ss : List Shape} (g : Graph Γ ss) (x dx : TorchLean.TensorPack ℝ Γ) :
    Algebra.Graph.jvpCtx g.toAlgebra x dx () = jvpCtx g x dx := by
  induction g with
  | nil => rfl
  | snoc g node ih =>
    have he := eval_toAlgebra g x
    simp only [toAlgebra, Algebra.Graph.eval, Algebra.Graph.jvpCtx, Algebra.Graph.toData,
      Algebra.GraphData.jvpCtx, jvpCtx] at ih he ⊢
    rw [ih, he]
    rfl

/-- Embedding preserves the reverse pass. -/
theorem backpropCtx_toAlgebra {ss : List Shape} (g : Graph Γ ss) (x : TorchLean.TensorPack ℝ Γ)
    (seed : TorchLean.TensorPack ℝ (Γ ++ ss)) :
    Algebra.Graph.backpropCtx g.toAlgebra x () seed = backpropCtx g x seed := by
  induction g with
  | nil => rfl
  | snoc g node ih =>
    have he := eval_toAlgebra g x
    simp only [toAlgebra, Algebra.Graph.eval, Algebra.Graph.backpropCtx, Algebra.Graph.toData,
      Algebra.GraphData.backpropCtx, backpropCtx] at ih he ⊢
    rw [he]
    exact ih _

/--
**Global tape soundness**: if each node satisfies a local JVP/VJP adjointness law, then the global
reverse-mode accumulation algorithm (`backpropCtx`) is correct.

Informally: for any input perturbation `dx` and any output seed cotangent `seed`,

`⟪JVP(g, x, dx), seed⟫ = ⟪dx, backprop(g, x, seed)⟫`.

This is the formal analogue of PyTorch’s guarantee that `backward()` computes vector–Jacobian
products and accumulates them through a dynamic DAG/tape. It is the `ℝ`, `Δ := Unit` instance of
`Algebra.Graph.backprop_correct`.
-/
theorem backprop_correct {ss : List Shape} (g : Graph Γ ss) :
    ∀ x dx seed,
      TensorPack.dotList (jvpCtx (ss := ss) g x dx) seed =
        TensorPack.dotList dx (backpropCtx (ss := ss) g x seed) := by
  intro x dx seed
  rw [TensorPack.dotList_eq_algebra_dotList, TensorPack.dotList_eq_algebra_dotList,
    ← jvpCtx_toAlgebra, ← backpropCtx_toAlgebra]
  exact Algebra.Graph.backprop_correct g.toAlgebra x dx () seed

end Graph


end
end Autograd
end Proofs
