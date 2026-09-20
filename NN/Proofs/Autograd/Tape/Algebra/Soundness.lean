/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Algebra
-- `Idx` and `getIdx` are shared with the real-valued tape proofs and the
-- runtime-approximation graphs; they live in one place so index lemmas transfer.
public import NN.Proofs.Autograd.Tape.Util.Idx

/-!
# Algebraic tape soundness

Tape-style (SSA/DAG) reverse-mode soundness (algebraic, backend-generic).

This is a backend-generic analogue of the tensor-tape soundness layer: it proves the
global reverse-mode accumulation algorithm is sound assuming only commutative semiring laws.

This file lives under `NN/Proofs/Autograd/Tape/Algebra/` because it is reused by both proof-only
and runtime-link developments that target exact backends (e.g. `ℚ`).


## PyTorch correspondence / citations
This corresponds to the high-level structure of PyTorch’s reverse-mode engine, but stated over an
arbitrary commutative semiring so we can reuse it for exact backends.
https://pytorch.org/docs/stable/autograd.html
-/

@[expose] public section


namespace Proofs
namespace Autograd
namespace Algebra

open Spec TorchLean TorchLean.Tensor TensorAlgebra

noncomputable section

namespace TensorPack

variable {α : Type} [TorchLean.Storage α]
variable {ss : List Shape}

section

variable [CommSemiring α]

/--
Dot product over contexts: sum of per-entry tensor dots.

This is the algebraic analogue of `Spec.dotList`: it uses `TensorAlgebra.dot` for the backend `α`.
-/
def dotList : {ss : List Shape} → TorchLean.TensorPack α ss → TorchLean.TensorPack α ss → α
  | [], .nil, .nil => 0
  | _ :: ss, .cons a as, .cons b bs => dot (α := α) a b + dotList (ss := ss) as bs

/-- `dotList` commutes with casting the left context along a shape-list equality. -/
theorem dotList_cast_left {ss₁ ss₂ : List Shape} (h : ss₁ = ss₂) (x : TorchLean.TensorPack α ss₁)
    (y : TorchLean.TensorPack α ss₂) :
    dotList (α := α) (TorchLean.TensorPack.cast (α := α) h x) y =
      dotList (α := α) x (TorchLean.TensorPack.cast (α := α) h.symm y) := by
  cases h
  rfl

/-- `dotList` is linear in its right argument with respect to `TorchLean.TensorPack.add`. -/
theorem dotList_add_right {ss : List Shape} (x y z : TorchLean.TensorPack α ss) :
    dotList (α := α) x (TorchLean.TensorPack.add (α := α) y z) =
      dotList (α := α) x y + dotList (α := α) x z := by
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
          simp [dotList, TorchLean.TensorPack.add,
            TensorAlgebra.dot_add_right (α := α) (a := xh) (b := yh) (c := zh),
            ih, add_assoc, add_left_comm]

/-- Dot respects appending: dot of two `snoc`ed contexts splits into prefix + last entry. -/
theorem dotList_snoc {ss : List Shape} {τ : Shape} (x y : TorchLean.TensorPack α ss)
    (a b : Tensor α τ) :
    dotList (α := α) (TorchLean.TensorPack.snoc (α := α) (ss := ss) x a)
        (TorchLean.TensorPack.snoc (α := α) (ss := ss) y b) =
      dotList (α := α) x y + dot (α := α) a b := by
  revert x y
  induction ss with
  | nil =>
    intro x y
    cases x; cases y
    simp [TorchLean.TensorPack.snoc, dotList]
  | cons s ss ih =>
    intro x y
    cases x with
    | cons xh xt =>
      cases y with
      | cons yh yt =>
        simp [TorchLean.TensorPack.snoc, dotList, ih, add_left_comm, add_comm]

/-- Dotting with the all-zero context on the right yields `0`. -/
theorem dotList_zero_right {ss : List Shape} (x : TorchLean.TensorPack α ss) :
    dotList (α := α) x (TorchLean.TensorPack.zero (α := α) (ss := ss)) = 0 := by
  induction ss with
  | nil =>
    cases x
    simp [dotList, TorchLean.TensorPack.zero]
  | cons s ss ih =>
    cases x with
    | cons xh xt =>
      simp [dotList, TorchLean.TensorPack.zero,
        TensorAlgebra.dot_full_zero_right (α := α) (s := s) (a := xh), ih]

end

end TensorPack

namespace TensorPack

variable {α : Type} [TorchLean.Storage α]

/-- Sparse context with a single nonzero entry at `idx` (all other tensors are `0`). -/
def single [Zero α] {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (v : Tensor α s) :
    TorchLean.TensorPack α Γ :=
  match Γ, idx with
  | [], ⟨i, _h⟩ =>
      match i with
      | ⟨val, isLt⟩ => False.elim ((Nat.not_lt_zero val) isLt)
  | s0 :: Γtail, ⟨i, h⟩ =>
      match i with
      | ⟨0, _⟩ =>
          .cons (Tensor.castShape v h.symm) (TorchLean.TensorPack.zero (α := α) (ss := Γtail))
      | ⟨Nat.succ j, hj⟩ =>
          let iTail : Fin Γtail.length := ⟨j, Nat.lt_of_succ_lt_succ hj⟩
          let hTail : Γtail.get iTail = s := by
            simpa using h
          .cons (Tensor.full s0 (0 : α)) (single (Γ := Γtail) (s := s) ⟨iTail, hTail⟩ v)

section

variable [CommSemiring α]

/--
`single` is adjoint to `getIdx` with respect to `dotList`.

Informally: `⟪dx, single idx v⟫ = ⟪getIdx dx idx, v⟫`.
-/
theorem dotList_single {Γ : List Shape} {s : Shape}
    (dx : TorchLean.TensorPack α Γ) (idx : Idx Γ s) (v : Tensor α s) :
    TensorPack.dotList (α := α) dx (single idx v) = dot (α := α) (getIdx (α := α) dx idx) v := by
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
                have hs0 : (s0 :: Γtail).get ⟨0, isLt⟩ = s0 := by
                  rfl
                have hs : s0 = s := by
                  simpa [hs0] using h
                cases hs
                calc
                  TensorPack.dotList (α := α) (TorchLean.TensorPack.cons dx0 dxRest)
                      (single ⟨⟨0, isLt⟩, rfl⟩ v)
                      = dot (α := α) dx0 v := by
                          simp [TensorPack.dotList, single, Tensor.castShape,
                            TensorPack.dotList_zero_right]
                  _ = dot (α := α)
                        (getIdx (α := α) (TorchLean.TensorPack.cons dx0 dxRest)
                          ⟨⟨0, isLt⟩, rfl⟩) v := by
                          -- `getIdx` at index `0` reduces definitionally to the head tensor.
                          dsimp [getIdx, Tensor.castShape]
                          have hget0 :
                              (TorchLean.TensorPack.cons dx0 dxRest).get
                                (i := (0 : Fin (s :: Γtail).length)) = dx0 := by
                            rfl
                          exact (congrArg (fun t => dot (α := α) t v) hget0).symm
            | succ j =>
                have h0 : dot (α := α) dx0 (Tensor.full s0 (0 : α)) = 0 :=
                  TensorAlgebra.dot_full_zero_right (α := α) (s := s0) (a := dx0)
                let iHead : Fin (s0 :: Γtail).length := ⟨Nat.succ j, isLt⟩
                let iTail : Fin Γtail.length := ⟨j, Nat.lt_of_succ_lt_succ isLt⟩
                let hTail : Γtail.get iTail = s := by
                  simpa using h
                let idxTail : Idx Γtail s := ⟨iTail, hTail⟩
                have hget :
                    getIdx (α := α) (TorchLean.TensorPack.cons dx0 dxRest) ⟨iHead, h⟩ =
                      getIdx (α := α) dxRest idxTail := by
                  -- Peel off the head entry, then use definitional equality for `cast_shape` after
                  -- rewriting the index proof.
                  dsimp [getIdx, idxTail, iHead, iTail]
                  have hcons :
                      TorchLean.TensorPack.get (α := α) (ss := s0 :: Γtail)
                          (TorchLean.TensorPack.cons dx0 dxRest) iHead =
                        TorchLean.TensorPack.get (α := α) (ss := Γtail) dxRest iTail := by
                    exact TorchLean.TensorPack.get_cons_succ (α := α) (s := s0)
                      (ss := Γtail) dx0 dxRest j isLt
                  -- After rewriting `h`, the two casts become the same.
                  cases h
                  rw [hcons]
                calc
                  TensorPack.dotList (α := α) (TorchLean.TensorPack.cons dx0 dxRest)
                      (single ⟨iHead, h⟩ v)
                      = dot (α := α) dx0 (Tensor.full s0 (0 : α)) +
                          TensorPack.dotList (α := α) dxRest (single idxTail v) := by
                            simp [TensorPack.dotList, single, idxTail, iHead, iTail]
                  _ = TensorPack.dotList (α := α) dxRest (single idxTail v) := by
                        simp [h0]
                  _ = dot (α := α) (getIdx (α := α) dxRest idxTail) v :=
                        ih (dx := dxRest) (idx := idxTail)
                  _ = dot (α := α)
                        (getIdx (α := α) (TorchLean.TensorPack.cons dx0 dxRest) ⟨iHead, h⟩) v := by
                        simp [hget]

end

end TensorPack

/--
Executable node payload (no correctness proof).

`Δ` is an extra non-differentiable environment threaded through evaluation (e.g. parameters,
auxiliary data). The VJP returns gradients only for the differentiable context `Γ`.
-/
structure NodeData (α : Type) [TorchLean.Storage α]
    (Δ : Type) (Γ : List Shape) (τ : Shape) where
  /-- Evaluate the node from the current differentiable context and auxiliary data. -/
  forward : TorchLean.TensorPack α Γ → Δ → Tensor α τ
  /-- Propagate one tangent context through the node. -/
  jvp : TorchLean.TensorPack α Γ → TorchLean.TensorPack α Γ → Δ → Tensor α τ
  /-- Pull an output cotangent back to the node's differentiable input context. -/
  vjp : TorchLean.TensorPack α Γ → Δ → Tensor α τ → TorchLean.TensorPack α Γ
  /-- Optional runtime precondition, enforced by checked execution before evaluating this node. -/
  validate : TorchLean.TensorPack α Γ → Δ → Except String Unit := fun _ _ => .ok ()

/--
Proof-carrying node: `NodeData` plus the local adjointness law.

The field `correct` is the algebraic version of the standard JVP/VJP inner-product law.
-/
structure Node {α : Type} [TorchLean.Storage α] [CommSemiring α]
    (Δ : Type) (Γ : List Shape) (τ : Shape)
    extends NodeData α Δ Γ τ where
  /-- The node's JVP and VJP satisfy the local dot-product adjoint identity. -/
  correct : ∀ x dx d δ, dot (α := α) (jvp x dx d) δ = TensorPack.dotList (α := α) dx (vjp x d δ)

/-- Executable-only graph: a snoc-list of `NodeData`. -/
inductive GraphData (α : Type) [TorchLean.Storage α]
    (Δ : Type) (Γ : List Shape) : List Shape → Type where
  /-- A graph with no computed nodes; its context consists only of the inputs `Γ`. -/
  | nil : GraphData α Δ Γ []
  /-- Append one node whose inputs may use the original and previously computed values. -/
  | snoc {ss : List Shape} {τ : Shape} :
      GraphData α Δ Γ ss → NodeData α Δ (Γ ++ ss) τ → GraphData α Δ Γ (ss ++ [τ])

namespace GraphData

variable {α : Type} [TorchLean.Storage α]
variable {Δ : Type}
variable {Γ : List Shape}

/-- Evaluate a `GraphData` on an input context `x`, producing the full context `Γ ++ ss`. -/
def eval {ss : List Shape} (g : GraphData α Δ Γ ss) (x : TorchLean.TensorPack α Γ) (d : Δ) :
    TorchLean.TensorPack α (Γ ++ ss) :=
  match g with
  | .nil => TorchLean.TensorPack.cast (α := α) (h := (List.append_nil Γ).symm) x
  | .snoc (ss := ss) (τ := τ) g node =>
      let ctx := eval (ss := ss) g x d
      let y := node.forward ctx d
      TorchLean.TensorPack.cast (α := α) (h := List.append_assoc Γ ss [τ])
        (TorchLean.TensorPack.snoc (α := α) (ss := Γ ++ ss) (τ := τ) ctx y)

/-- Compute the JVP of `eval`, producing a tangent context of shape `Γ ++ ss`. -/
def jvpCtx {ss : List Shape} (g : GraphData α Δ Γ ss) (x : TorchLean.TensorPack α Γ)
    (dx : TorchLean.TensorPack α Γ) (d : Δ) :
    TorchLean.TensorPack α (Γ ++ ss) :=
  match g with
  | .nil => TorchLean.TensorPack.cast (α := α) (h := (List.append_nil Γ).symm) dx
  | .snoc (ss := ss) (τ := τ) g node =>
      let ctx := eval (ss := ss) g x d
      let dctx := jvpCtx (ss := ss) g x dx d
      let dy := node.jvp ctx dctx d
      TorchLean.TensorPack.cast (α := α) (h := List.append_assoc Γ ss [τ])
        (TorchLean.TensorPack.snoc (α := α) (ss := Γ ++ ss) (τ := τ) dctx dy)

/-- Reverse-mode accumulation on contexts (VJP), given a seed cotangent for `Γ ++ ss`. -/
def backpropCtx [Add α] {ss : List Shape} (g : GraphData α Δ Γ ss) (x : TorchLean.TensorPack α Γ)
    (d : Δ) (seed : TorchLean.TensorPack α (Γ ++ ss)) : TorchLean.TensorPack α Γ :=
  match g with
  | .nil => TorchLean.TensorPack.cast (α := α) (h := List.append_nil Γ) seed
  | .snoc (ss := ss) (τ := τ) g node =>
      let seed' : TorchLean.TensorPack α ((Γ ++ ss) ++ [τ]) :=
        TorchLean.TensorPack.cast (α := α) (h := (List.append_assoc Γ ss [τ]).symm) seed
      let seedPrev : TorchLean.TensorPack α (Γ ++ ss) :=
        (TorchLean.TensorPack.unsnoc (α := α) (ss := Γ ++ ss) (τ := τ) seed').1
      let seedOut : Tensor α τ :=
        (TorchLean.TensorPack.unsnoc (α := α) (ss := Γ ++ ss) (τ := τ) seed').2
      let ctx := eval (ss := ss) g x d
      let contrib := node.vjp ctx d seedOut
      let seedPrev' := TorchLean.TensorPack.add (α := α) (ss := Γ ++ ss) seedPrev contrib
      backpropCtx (ss := ss) g x d seedPrev'

end GraphData

/-!
Proof-carrying tape/SSA graphs.

Nodes are appended in topological order and may reference any previously computed value (fan-out
and sharing are allowed). This mirrors the structure of PyTorch’s dynamic autograd graph, but with
shape-typed contexts.
-/

/--
A proof-carrying tape/SSA graph.

Nodes are appended in topological order and may reference any previously computed value.
-/
inductive Graph {α : Type} [TorchLean.Storage α] [CommSemiring α]
    (Δ : Type) (Γ : List Shape) : List Shape → Type where
  /-- A graph with no computed nodes; its context consists only of the inputs `Γ`. -/
  | nil : Graph Δ Γ []
  /-- Append one locally correct node that may use the inputs and all preceding results. -/
  | snoc {ss : List Shape} {τ : Shape} :
      Graph Δ Γ ss → Node (α := α) (Δ := Δ) (Γ := Γ ++ ss) τ → Graph Δ Γ (ss ++ [τ])

namespace Graph

variable {α : Type} [TorchLean.Storage α] [CommSemiring α]
variable {Δ : Type}
variable {Γ : List Shape}

/-- Forget local correctness proofs, yielding an executable `GraphData`. -/
def toData {ss : List Shape} : Graph (α := α) Δ Γ ss → GraphData α Δ Γ ss
  | .nil => .nil
  | .snoc g node => .snoc (toData (ss := _) g) node.toNodeData

end Graph

namespace Graph

variable {α : Type} [TorchLean.Storage α] [CommSemiring α]
variable {Δ : Type}
variable {Γ : List Shape}

/-- Evaluate a proof-carrying graph through its executable representation. -/
def eval {ss : List Shape} (g : Graph (α := α) Δ Γ ss) (x : TorchLean.TensorPack α Γ) (d : Δ) :
    TorchLean.TensorPack α (Γ ++ ss) :=
  GraphData.eval g.toData x d

/-- Propagate tangents with the same evaluator used by the executable graph. -/
def jvpCtx {ss : List Shape} (g : Graph (α := α) Δ Γ ss) (x : TorchLean.TensorPack α Γ)
    (dx : TorchLean.TensorPack α Γ) (d : Δ) :
    TorchLean.TensorPack α (Γ ++ ss) :=
  GraphData.jvpCtx g.toData x dx d

/-- Accumulate cotangents with the executable graph's reverse pass. -/
def backpropCtx {ss : List Shape} (g : Graph (α := α) Δ Γ ss) (x : TorchLean.TensorPack α Γ) (d : Δ)
    (seed : TorchLean.TensorPack α (Γ ++ ss)) : TorchLean.TensorPack α Γ :=
  GraphData.backpropCtx g.toData x d seed

/--
Global tape soundness (algebraic form).

Assuming each node satisfies its local adjointness law, `backpropCtx` is the adjoint of `jvpCtx`
with respect to `TensorPack.dotList`.
-/
theorem backprop_correct {ss : List Shape} (g : Graph (α := α) Δ Γ ss) :
    ∀ x dx d seed,
      TensorPack.dotList (α := α) (jvpCtx (ss := ss) g x dx d) seed =
        TensorPack.dotList (α := α) dx (backpropCtx (ss := ss) g x d seed) := by
  induction g with
  | nil =>
    intro x dx d seed
    simpa [jvpCtx, backpropCtx, toData, GraphData.jvpCtx, GraphData.backpropCtx] using
      (TensorPack.dotList_cast_left (α := α) (h := (List.append_nil Γ).symm) (x := dx) (y := seed))
  | snoc g node ih =>
    intro x dx d seed
    rename_i ss τ
    let ctx := eval (ss := ss) g x d
    let dctx := jvpCtx (ss := ss) g x dx d
    let dy := node.jvp ctx dctx d
    let assoc : (Γ ++ ss) ++ [τ] = Γ ++ (ss ++ [τ]) := List.append_assoc Γ ss [τ]
    let seed' : TorchLean.TensorPack α ((Γ ++ ss) ++ [τ]) :=
      TorchLean.TensorPack.cast (α := α) (h := assoc.symm) seed
    let seedPrev : TorchLean.TensorPack α (Γ ++ ss) :=
      (TorchLean.TensorPack.unsnoc (α := α) (ss := Γ ++ ss) (τ := τ) seed').1
    let seedOut : Tensor α τ :=
      (TorchLean.TensorPack.unsnoc (α := α) (ss := Γ ++ ss) (τ := τ) seed').2
    have hseed :
        TorchLean.TensorPack.snoc (α := α) (ss := Γ ++ ss) (τ := τ) seedPrev seedOut = seed' := by
      change TorchLean.TensorPack.snoc
        (TorchLean.TensorPack.unsnoc seed').1
        (TorchLean.TensorPack.unsnoc seed').2 = seed'
      exact TorchLean.TensorPack.snoc_unsnoc seed'
    have hjvp :
        TensorPack.dotList (α := α) (jvpCtx (ss := ss ++ [τ]) (Graph.snoc g node) x dx d) seed =
          TensorPack.dotList (α := α) dctx seedPrev + dot (α := α) dy seedOut := by
      have :
          TensorPack.dotList (α := α)
              (TorchLean.TensorPack.snoc (α := α) (ss := Γ ++ ss) (τ := τ) dctx dy) seed' =
            TensorPack.dotList (α := α) dctx seedPrev + dot (α := α) dy seedOut := by
        simpa [hseed] using
          (TensorPack.dotList_snoc (α := α) (ss := Γ ++ ss) (τ := τ) (x := dctx) (y := seedPrev)
            (a := dy) (b := seedOut))
      simpa [jvpCtx, ctx, dctx, dy, seed', assoc, toData, GraphData.jvpCtx, eval] using
        (TensorPack.dotList_cast_left (α := α) (h := assoc)
          (x := TorchLean.TensorPack.snoc (α := α) (ss := Γ ++ ss) (τ := τ) dctx dy) (y := seed)
          |>.trans this)
    have hlocal :
        dot (α := α) dy seedOut = TensorPack.dotList (α := α) dctx (node.vjp ctx d seedOut) := by
      simpa [dy] using (node.correct ctx dctx d seedOut)
    have hadd :
        TensorPack.dotList (α := α) dctx seedPrev +
            TensorPack.dotList (α := α) dctx (node.vjp ctx d seedOut) =
          TensorPack.dotList (α := α) dctx
            (TorchLean.TensorPack.add (α := α) (ss := Γ ++ ss) seedPrev
              (node.vjp ctx d seedOut)) := by
      simpa using
        (TensorPack.dotList_add_right (α := α) (x := dctx) (y := seedPrev) (z := node.vjp ctx d
          seedOut)).symm
    calc
      TensorPack.dotList (α := α) (jvpCtx (ss := ss ++ [τ]) (Graph.snoc g node) x dx d) seed
          = TensorPack.dotList (α := α) dctx seedPrev + dot (α := α) dy seedOut := hjvp
      _ = TensorPack.dotList (α := α) dctx seedPrev +
            TensorPack.dotList (α := α) dctx (node.vjp ctx d seedOut) := by
            simp [hlocal]
      _ = TensorPack.dotList (α := α) dctx
            (TorchLean.TensorPack.add (α := α) (ss := Γ ++ ss) seedPrev
              (node.vjp ctx d seedOut)) := by
            simp [hadd]
      _ = TensorPack.dotList (α := α) dx
            (backpropCtx (ss := ss) g x d
              (TorchLean.TensorPack.add (α := α) (ss := Γ ++ ss) seedPrev
                (node.vjp ctx d seedOut))) := by
            simpa [dctx] using
              (ih x dx d
                (TorchLean.TensorPack.add (α := α) (ss := Γ ++ ss) seedPrev
                  (node.vjp ctx d seedOut)))
      _ = TensorPack.dotList (α := α) dx
            (backpropCtx (ss := ss ++ [τ]) (Graph.snoc g node) x d seed) := by
            simp [backpropCtx, ctx, seed', seedPrev, seedOut, toData, GraphData.backpropCtx, eval]

end Graph

end

end Algebra
end Autograd
end Proofs
