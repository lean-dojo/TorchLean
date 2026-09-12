/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Algebra.Order.Algebra
public import Mathlib.Analysis.SpecialFunctions.Pow.NNReal
public import Mathlib.Data.Sym.Sym2.Init
import Mathlib.Tactic.NormNum.GCD
public import NN.Proofs.Tensor.Basic.BoundsNorms
public import NN.Tensor.Pack
-- Typed context indices are shared with the generic and runtime-approximation
-- graph developments, so `Idx` and `getIdx` come from one module.
public import NN.Proofs.Autograd.Tape.Util.Idx

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

/-- A context cast can be moved from one side of the context dot product to the other.

Casts appear whenever two tapes are composed and their context shape lists are only propositionally
equal; without this lemma every adjointness proof would have to `cases` the equality by hand. -/
theorem dotList_cast_left {ss₁ ss₂ : List Shape} (h : ss₁ = ss₂) (x : TorchLean.TensorPack ℝ ss₁)
    (y : TorchLean.TensorPack ℝ ss₂) :
    dotList (TorchLean.TensorPack.cast h x) y =
      dotList x (TorchLean.TensorPack.cast h.symm y) := by
  cases h
  rfl

private theorem dot_add_right {s : Shape} (a b c : Tensor ℝ s) :
    dot a (addSpec b c) = dot a b + dot a c := by
  calc
    dot a (addSpec b c) = dot (addSpec b c) a := by
      simpa using (dot_comm (a := a) (b := addSpec b c))
    _ = dot b a + dot c a := by
      simpa using (dot_add_left (a := b) (b := c) (c := a))
    _ = dot a b + dot a c := by
      simp [dot_comm]

/--
`dotList` is linear in its right argument with respect to `TorchLean.TensorPack.add`.

Informally: `⟪x, y + z⟫ = ⟪x, y⟫ + ⟪x, z⟫` for contexts.
-/
theorem dotList_add_right {ss : List Shape} (x y z : TorchLean.TensorPack ℝ ss) :
    dotList x (TorchLean.TensorPack.add y z) = dotList x y + dotList x z := by
  induction ss with
  | nil =>
    cases x; cases y; cases z; simp [dotList, TorchLean.TensorPack.add]
  | cons s ss ih =>
    cases x with
    | cons xh xt =>
      cases y with
      | cons yh yt =>
        cases z with
        | cons zh zt =>
          simp [dotList, TorchLean.TensorPack.add, dot_add_right, ih, add_assoc,
            add_left_comm]

/--
`dotList` respects appending: dot of two `snoc`ed contexts splits into prefix + last entry.

Informally: `⟪(x,a), (y,b)⟫ = ⟪x,y⟫ + ⟪a,b⟫`.
-/
theorem dotList_snoc {ss : List Shape} {τ : Shape} (x y : TorchLean.TensorPack ℝ ss)
    (a b : Tensor ℝ τ) :
    dotList (TorchLean.TensorPack.snoc x a)
      (TorchLean.TensorPack.snoc y b) = dotList x y + dot a b := by
  revert x y
  induction ss with
  | nil =>
    intro x y
    cases x; cases y
    simp [dotList, TorchLean.TensorPack.snoc]
  | cons s ss ih =>
    intro x y
    cases x with
    | cons xh xt =>
      cases y with
      | cons yh yt =>
        simp [dotList, TorchLean.TensorPack.snoc, ih, add_left_comm, add_comm]

private theorem full_eq_scale_one {s : Shape} (c : ℝ) :
    Tensor.full s c = scaleSpec (α:=ℝ) (s:=s) (Tensor.full s (1 : ℝ)) c := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  simp [Tensor.full, scaleSpec, mapSpec, Tensor.map]

/--
Dotting any tensor with a zero-filled tensor gives `0`.

This is the tensor-level fact used to show that “one-hot” cotangents behave as expected.
-/
theorem dot_full_zero_right {s : Shape} (a : Tensor ℝ s) :
    dot a (Tensor.full s (0 : ℝ)) = 0 := by
  have hfill : Tensor.full s (0 : ℝ) = scaleSpec (α:=ℝ) (s:=s) (Tensor.full s (1 : ℝ)) 0 := by
    simpa using (full_eq_scale_one (s := s) (c := (0 : ℝ)))
  calc
    dot a (Tensor.full s (0 : ℝ))
        = dot a (scaleSpec (α:=ℝ) (s:=s) (Tensor.full s (1 : ℝ)) 0) := by
            simp [hfill]
    _ = dot (scaleSpec (α:=ℝ) (s:=s) (Tensor.full s (1 : ℝ)) 0) a := by
          simpa using (dot_comm (a := a) (b := scaleSpec (α:=ℝ) (s:=s) (Tensor.full s (1 : ℝ)) 0))
    _ = 0 * dot (Tensor.full s (1 : ℝ)) a := by
          simpa using (dot_scale_left (a := Tensor.full s (1 : ℝ)) (b := a) (k := (0 : ℝ)))
    _ = 0 := by ring

/-- `dotList x 0 = 0` for the all-zero context. -/
theorem dotList_zero_right {ss : List Shape} (x : TorchLean.TensorPack ℝ ss) :
    dotList x (TorchLean.TensorPack.zero (ss := ss)) = 0 := by
  induction ss with
  | nil =>
    cases x
    simp [dotList, TorchLean.TensorPack.zero]
  | cons s ss ih =>
    cases x with
    | cons xh xt =>
      simp [dotList, TorchLean.TensorPack.zero, dot_full_zero_right, ih]

end TensorPack

namespace TensorPack

/--
Build a sparse context with a single nonzero entry at `idx` and zeros elsewhere.

This is used to express “one-hot” cotangents when proving local-to-global backprop correctness.
-/
def single {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (v : Tensor ℝ s) :
    TorchLean.TensorPack ℝ Γ :=
  match Γ, idx with
  | [], ⟨i, _h⟩ =>
      match i with
      | ⟨val, isLt⟩ => False.elim ((Nat.not_lt_zero val) isLt)
  | s0 :: Γtail, ⟨i, h⟩ =>
      match i with
      | ⟨0, _⟩ =>
          .cons (Tensor.castShape v h.symm)
            (TorchLean.TensorPack.zero (ss := Γtail))
      | ⟨Nat.succ j, hj⟩ =>
          let iTail : Fin Γtail.length := ⟨j, Nat.lt_of_succ_lt_succ hj⟩
          let hTail : Γtail.get iTail = s := by
            simpa using h
          .cons (Tensor.full s0 (0 : ℝ)) (single (Γ := Γtail) (s := s) ⟨iTail, hTail⟩ v)

/--
`single idx v` is the “one-hot” context with value `v` at `idx`, and zeros elsewhere.

This lemma says the context dot product against `single idx v` picks out the corresponding entry
of `dx`:

`⟪dx, single idx v⟫ = ⟪dx[idx], v⟫`.
-/
theorem dotList_single {Γ : List Shape} {s : Shape}
    (dx : TorchLean.TensorPack ℝ Γ) (idx : Idx Γ s) (v : Tensor ℝ s) :
    TensorPack.dotList dx (single idx v) = dot (getIdx dx idx) v := by
  -- Structural recursion over the context list, tracking the index.
  revert dx idx
  induction Γ with
  | nil =>
    intro dx idx
    cases idx with
    | mk i _h =>
      cases i with
        | mk val isLt =>
          exact False.elim ((Nat.not_lt_zero val) isLt)
  | cons s0 Γtail ih =>
      intro dx idx
      cases dx with
      | cons dx0 dxRest =>
        cases idx with
        | mk i h =>
          cases i with
          | mk val isLt =>
            cases val with
            | zero =>
              -- head index
              have hs0 : (s0 :: Γtail).get ⟨0, isLt⟩ = s0 := by
                rfl
              have hs : s0 = s := by
                simpa [hs0] using h
              cases hs
              calc
                TensorPack.dotList (.cons dx0 dxRest) (single ⟨⟨0, isLt⟩, rfl⟩ v) =
                    dot dx0 v := by
                  simp [TensorPack.dotList, single, Tensor.castShape, TensorPack.dotList_zero_right]
                _ = dot (getIdx (.cons dx0 dxRest) ⟨⟨0, isLt⟩, rfl⟩) v := by
                  dsimp [getIdx, Tensor.castShape]
                  have hget0 :
                      TorchLean.TensorPack.get (.cons dx0 dxRest)
                        (0 : Fin (s :: Γtail).length) = dx0 := by
                    rfl
                  exact (congrArg (fun t => dot t v) hget0).symm
            | succ j =>
              -- tail index
              have h0 : dot dx0 (Tensor.full s0 (0 : ℝ)) = 0 := dot_full_zero_right (a := dx0)
              let iHead : Fin (s0 :: Γtail).length := ⟨Nat.succ j, isLt⟩
              let iTail : Fin Γtail.length := ⟨j, Nat.lt_of_succ_lt_succ isLt⟩
              let hTail : Γtail.get iTail = s := by
                simpa using h
              let idxTail : Idx Γtail s := ⟨iTail, hTail⟩
              have hget : getIdx (.cons dx0 dxRest) ⟨iHead, h⟩ = getIdx dxRest idxTail := by
                -- Reduce the `get` at a successor index, then discharge cast-proof mismatches by
                -- proof-irrelevance.
                dsimp [getIdx]
                simp [iHead]
                exact (Tensor.cast_shape_proof_irrel (dxRest.get iTail) :
                  Tensor.castShape (tensor := dxRest.get iTail) _ =
                    Tensor.castShape (tensor := dxRest.get iTail) _)
              calc
                TensorPack.dotList (.cons dx0 dxRest) (single ⟨iHead, h⟩ v)
                    = dot dx0 (Tensor.full s0 (0 : ℝ)) +
                        TensorPack.dotList dxRest (single idxTail v) := by
                        simp [TensorPack.dotList, single, idxTail, iHead, iTail]
                _ = TensorPack.dotList dxRest (single idxTail v) := by
                      simp [h0]
                _ = dot (getIdx dxRest idxTail) v := by
                      simpa using (ih (dx := dxRest) (idx := idxTail))
                _ = dot (getIdx (.cons dx0 dxRest) ⟨iHead, h⟩) v := by
                      simp [hget]

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

/--
**Global tape soundness**: if each node satisfies a local JVP/VJP adjointness law, then the global
reverse-mode accumulation algorithm (`backpropCtx`) is correct.

Informally: for any input perturbation `dx` and any output seed cotangent `seed`,

`⟪JVP(g, x, dx), seed⟫ = ⟪dx, backprop(g, x, seed)⟫`.

This is the formal analogue of PyTorch’s guarantee that `backward()` computes vector–Jacobian
products and accumulates them through a dynamic DAG/tape.
-/
theorem backprop_correct {ss : List Shape} (g : Graph Γ ss) :
    ∀ x dx seed,
      TensorPack.dotList (jvpCtx (ss := ss) g x dx) seed =
        TensorPack.dotList dx (backpropCtx (ss := ss) g x seed) := by
  induction g with
  | nil =>
    intro x dx seed
    -- `ss = []` so this is exactly the dotList/cast adjointness.
    simpa [jvpCtx, backpropCtx] using
      (TensorPack.dotList_cast_left (h := (List.append_nil Γ).symm) (x := dx) (y := seed))
  | snoc g node ih =>
    intro x dx seed
    rename_i ss τ
    let ctx := eval (ss := ss) g x
    let dctx := jvpCtx (ss := ss) g x dx
    let dy := node.jvp ctx dctx
    let assoc : (Γ ++ ss) ++ [τ] = Γ ++ (ss ++ [τ]) := List.append_assoc Γ ss [τ]
    let seed' : TorchLean.TensorPack ℝ ((Γ ++ ss) ++ [τ]) :=
      TorchLean.TensorPack.cast (h := assoc.symm) seed
    let seedPrev : TorchLean.TensorPack ℝ (Γ ++ ss) :=
      (TorchLean.TensorPack.unsnoc (ss := Γ ++ ss) seed').1
    let seedOut : Tensor ℝ τ := (TorchLean.TensorPack.unsnoc (ss := Γ ++ ss) seed').2
    have hseed : TorchLean.TensorPack.snoc seedPrev seedOut = seed' := by
      change TorchLean.TensorPack.snoc
        (TorchLean.TensorPack.unsnoc seed').1
        (TorchLean.TensorPack.unsnoc seed').2 = seed'
      exact TorchLean.TensorPack.snoc_unsnoc seed'
    have hjvp :
        TensorPack.dotList (jvpCtx (ss := ss ++ [τ]) (Graph.snoc g node) x dx) seed =
          TensorPack.dotList dctx seedPrev + dot dy seedOut := by
      -- Move casts so we can use `dotList_snoc` on reassociated contexts.
      have :
          TensorPack.dotList (TorchLean.TensorPack.snoc dctx dy) seed' =
            TensorPack.dotList dctx seedPrev + dot dy seedOut := by
        simpa [hseed] using
          (TensorPack.dotList_snoc (x := dctx) (y := seedPrev) (a := dy) (b := seedOut))
      -- Unfold `jvpCtx` at the snoc node, then apply `dotList_cast_left`.
      simpa [jvpCtx, ctx, dctx, dy, seed', assoc] using
        (TensorPack.dotList_cast_left (h := assoc) (x := TorchLean.TensorPack.snoc dctx dy)
          (y := seed) |>.trans this)
    have hlocal : dot dy seedOut = TensorPack.dotList dctx (node.vjp ctx seedOut) := by
      simpa [dy] using (node.correct ctx dctx seedOut)
    have hadd :
        TensorPack.dotList dctx seedPrev + TensorPack.dotList dctx (node.vjp ctx seedOut) =
          TensorPack.dotList dctx (TorchLean.TensorPack.add seedPrev (node.vjp ctx seedOut)) := by
      simpa using
        (TensorPack.dotList_add_right (x := dctx) (y := seedPrev) (z := node.vjp ctx seedOut)).symm
    calc
      TensorPack.dotList (jvpCtx (ss := ss ++ [τ]) (Graph.snoc g node) x dx) seed
          = TensorPack.dotList dctx seedPrev + dot dy seedOut := hjvp
      _ = TensorPack.dotList dctx seedPrev + TensorPack.dotList dctx (node.vjp ctx seedOut) := by
            simp [hlocal]
      _ = TensorPack.dotList dctx (TorchLean.TensorPack.add seedPrev (node.vjp ctx seedOut)) := by
            simp [hadd]
      _ = TensorPack.dotList dx
            (backpropCtx (ss := ss) g x
              (TorchLean.TensorPack.add seedPrev (node.vjp ctx seedOut))) := by
            simpa [dctx] using (ih x dx (TorchLean.TensorPack.add seedPrev (node.vjp ctx seedOut)))
      _ = TensorPack.dotList dx (backpropCtx (ss := ss ++ [τ]) (Graph.snoc g node) x seed) := by
            simp [backpropCtx, ctx, seed', seedPrev, seedOut]

end Graph

end
end Autograd
end Proofs
