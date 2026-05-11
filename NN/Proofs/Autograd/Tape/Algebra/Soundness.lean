/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Algebra

/-!
# Soundness

Tape-style (SSA/DAG) reverse-mode soundness (algebraic, backend-generic).

This is a backend-generic analogue of the tensor-tape soundness layer: it proves the
global reverse-mode accumulation algorithm is sound assuming only commutative semiring laws.

This file lives under `NN/Proofs/Autograd/Tape/Algebra/` because it is reused by both proof-only
and runtime-link developments that target exact backends (e.g. `ℚ`).

In particular, it can be instantiated for exact backends such as `ℚ`.

## PyTorch correspondence / citations
This corresponds to the high-level structure of PyTorch’s reverse-mode engine, but stated over an
arbitrary commutative semiring so we can reuse it for exact backends.
https://pytorch.org/docs/stable/autograd.html
-/

@[expose] public section


namespace Proofs
namespace Autograd
namespace Algebra

open Spec
open Tensor
open TensorAlgebra

noncomputable section

/--
A heterogeneous list of tensors indexed by a list of shapes.

`TList α Γ` is the algebraic (backend-generic) version of the typed tape context:
it stores one tensor of each shape in `Γ`.

PyTorch analogue: the engine carries a runtime list of saved tensors for backward; here `Γ`
tracks the shapes statically.
-/
inductive TList (α : Type) : List Shape → Type where
  | nil : TList α []
  | cons {s : Shape} {ss : List Shape} : Tensor α s → TList α ss → TList α (s :: ss)

namespace TList

variable {α : Type}
variable {ss : List Shape}

/-- Get the `i`th tensor from a context, with its shape tracked by `List.get`. -/
def get : {ss : List Shape} → TList α ss → (i : Fin ss.length) → Tensor α (ss.get i)
  | [], nil, i => nomatch i
  | _ :: _, cons x _xs, ⟨0, _⟩ => x
  | _ :: _ss, cons _x xs, ⟨Nat.succ i, hi⟩ => get xs ⟨i, Nat.lt_of_succ_lt_succ hi⟩

@[simp] theorem get_cons_zero {s : Shape} {ss : List Shape} (x : Tensor α s) (xs : TList α ss)
    (h : 0 < (s :: ss).length) :
    get (α := α) (ss := s :: ss) (cons x xs) ⟨0, h⟩ = x := by
  rfl

@[simp] theorem get_cons_succ {s : Shape} {ss : List Shape} (x : Tensor α s) (xs : TList α ss)
    (i : Nat) (h : Nat.succ i < (s :: ss).length) :
    get (α := α) (ss := s :: ss) (cons x xs) ⟨Nat.succ i, h⟩ =
      get (α := α) (ss := ss) xs ⟨i, Nat.lt_of_succ_lt_succ h⟩ := by
  rfl

/-- All-zero context (fills each tensor entry with zeros). -/
def zero [Zero α] : {ss : List Shape} → TList α ss
  | [] => nil
  | s :: ss => cons (fill (0 : α) s) (zero (ss := ss))

/-- Pointwise addition of two contexts of the same shape list. -/
def add [Add α] : {ss : List Shape} → TList α ss → TList α ss → TList α ss
  | [], nil, nil => nil
  | _ :: ss, cons a as, cons b bs => cons (addSpec a b) (add (ss := ss) as bs)

/--
Scale every tensor in a context by the same scalar.

This is the context-level analogue of `scaleSpec`. It lives beside `zero` and `add` because several
proof layers need scalar multiplication on heterogeneous parameter/gradient packs, not just the
training-step file.
-/
def scale [Mul α] (c : α) : {ss : List Shape} → TList α ss → TList α ss
  | [], nil => nil
  | _ :: ss, cons x xs => cons (scaleSpec (α := α) x c) (scale c (ss := ss) xs)

/--
Pointwise subtraction of two contexts with the same shape list.

This is the context-level analogue of `subSpec`; it is used to state pure optimizer updates over an
entire typed parameter context.
-/
def sub [Sub α] : {ss : List Shape} → TList α ss → TList α ss → TList α ss
  | [], nil, nil => nil
  | _ :: ss, cons a as, cons b bs => cons (subSpec (α := α) a b) (sub (ss := ss) as bs)

/-- Append a tensor to the end of a context. -/
def snoc {τ : Shape} : {ss : List Shape} → TList α ss → Tensor α τ → TList α (ss ++ [τ])
  | [], nil, t => cons t nil
  | _ :: ss, cons x xs, t => cons x (snoc (ss := ss) xs t)

/-- Split a context of shape list `ss ++ [τ]` into its prefix and last tensor. -/
def unsnoc {τ : Shape} : {ss : List Shape} → TList α (ss ++ [τ]) → TList α ss × Tensor α τ
  | [], cons t nil => (nil, t)
  | _s :: ss, cons x xs =>
      let (ys, last) := unsnoc (ss := ss) xs
      (cons x ys, last)

/-- Cast a context along an equality of shape lists. -/
def cast {ss₁ ss₂ : List Shape} (h : ss₁ = ss₂) (xs : TList α ss₁) : TList α ss₂ :=
  Eq.mp (congrArg (TList α) h) xs

@[simp] theorem cast_rfl {ss : List Shape} (xs : TList α ss) :
    cast (α := α) (ss₁ := ss) (ss₂ := ss) rfl xs = xs := by
  rfl

@[simp] theorem cast_cast {ss₁ ss₂ ss₃ : List Shape} (h₁ : ss₁ = ss₂) (h₂ : ss₂ = ss₃) (xs : TList α
  ss₁) :
    cast (α := α) h₂ (cast (α := α) h₁ xs) = cast (α := α) (h₁.trans h₂) xs := by
  cases h₁
  cases h₂
  rfl

@[simp] theorem cast_symm {ss₁ ss₂ : List Shape} (h : ss₁ = ss₂) (xs : TList α ss₁) :
    cast (α := α) h.symm (cast (α := α) h xs) = xs := by
  cases h
  rfl

section

variable [CommSemiring α]

/--
Dot product over contexts: sum of per-entry tensor dots.

This is the algebraic analogue of `Spec.dotList`: it uses `TensorAlgebra.dot` for the backend `α`.
-/
def dotList : {ss : List Shape} → TList α ss → TList α ss → α
  | [], nil, nil => 0
  | _ :: ss, cons a as, cons b bs => dot (α := α) a b + dotList (ss := ss) as bs

/-- `dotList` commutes with casting the left context along a shape-list equality. -/
theorem dotList_cast_left {ss₁ ss₂ : List Shape} (h : ss₁ = ss₂) (x : TList α ss₁) (y : TList α ss₂)
  :
    dotList (α := α) (cast (α := α) h x) y = dotList (α := α) x (cast (α := α) h.symm y) := by
  cases h
  rfl

/-- `dotList` is linear in its right argument with respect to `TList.add`. -/
theorem dotList_add_right {ss : List Shape} (x y z : TList α ss) :
    dotList (α := α) x (add (α := α) y z) = dotList (α := α) x y + dotList (α := α) x z := by
  induction ss with
  | nil =>
    cases x; cases y; cases z; simp [dotList, add]
  | cons s ss ih =>
    cases x with
    | cons xh xt =>
      cases y with
      | cons yh yt =>
        cases z with
        | cons zh zt =>
          simp [dotList, add, TensorAlgebra.dot_add_right (α := α) (a := xh) (b := yh) (c := zh),
            ih, add_assoc, add_left_comm]

/-- Dot respects appending: dot of two `snoc`ed contexts splits into prefix + last entry. -/
theorem dotList_snoc {ss : List Shape} {τ : Shape} (x y : TList α ss) (a b : Tensor α τ) :
    dotList (α := α) (snoc (α := α) (ss := ss) x a) (snoc (α := α) (ss := ss) y b) =
      dotList (α := α) x y + dot (α := α) a b := by
  revert x y
  induction ss with
  | nil =>
    intro x y
    cases x; cases y
    simp [snoc, dotList]
  | cons s ss ih =>
    intro x y
    cases x with
    | cons xh xt =>
      cases y with
      | cons yh yt =>
        simp [snoc, dotList, ih, add_left_comm, add_comm]

omit [CommSemiring α] in
/-- `unsnoc` is a left inverse of `snoc`. -/
theorem unsnoc_snoc {ss : List Shape} {τ : Shape} (xs : TList α ss) (t : Tensor α τ) :
    unsnoc (α := α) (ss := ss) (τ := τ) (snoc (α := α) (ss := ss) xs t) = (xs, t) := by
  induction ss with
  | nil =>
    cases xs
    simp [snoc, unsnoc]
  | cons s ss ih =>
    cases xs with
    | cons x xt =>
      simp [snoc, unsnoc, ih]

omit [CommSemiring α] in
/-- `snoc` is a right inverse of `unsnoc`. -/
theorem snoc_unsnoc {ss : List Shape} {τ : Shape} (xs : TList α (ss ++ [τ])) :
    snoc (α := α) (ss := ss) (τ := τ) (unsnoc (α := α) (ss := ss) (τ := τ) xs).1
      (unsnoc (α := α) (ss := ss) (τ := τ) xs).2 = xs := by
  induction ss with
  | nil =>
    cases xs with
    | cons t xs =>
      cases xs with
      | nil =>
        simp [unsnoc, snoc]
  | cons s ss ih =>
    cases xs with
    | cons x xs =>
      have hrec :
          snoc (α := α) (ss := ss) (τ := τ) (unsnoc (α := α) (ss := ss) (τ := τ) xs).1
              (unsnoc (α := α) (ss := ss) (τ := τ) xs).2 = xs := ih (xs := xs)
      simp [unsnoc, snoc, hrec]

/-- Dotting with the all-zero context on the right yields `0`. -/
theorem dotList_zero_right {ss : List Shape} (x : TList α ss) :
    dotList (α := α) x (zero (α := α) (ss := ss)) = 0 := by
  induction ss with
  | nil =>
    cases x
    simp [dotList, zero]
  | cons s ss ih =>
    cases x with
    | cons xh xt =>
      simp [dotList, zero, TensorAlgebra.dot_fill_zero_right (α := α) (s := s) (a := xh), ih]

end

end TList

/-!
`Idx Γ s` is a “typed index” into a heterogeneous context: it stores a `Fin Γ.length` together
with a proof that the shape at that position is `s`.
-/

/-- A typed index into a heterogeneous context `Γ`, carrying a proof of the expected shape `s`. -/
structure Idx (Γ : List Shape) (s : Shape) where
  /-- i. -/
  i : Fin Γ.length
  /-- h. -/
  h : Γ.get i = s

/-- Read a tensor from a context at a typed index, casting along the stored shape equality. -/
def getIdx {α : Type} {Γ : List Shape} {s : Shape} (xs : TList α Γ) (idx : Idx Γ s) :
    Tensor α s :=
  Tensor.castShape (xs.get (α := α) idx.i) idx.h

namespace TList

variable {α : Type}

/-- Sparse context with a single nonzero entry at `idx` (all other tensors are `0`). -/
def single [Zero α] {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (v : Tensor α s) : TList α Γ :=
  match Γ, idx with
  | [], ⟨i, _h⟩ =>
      match i with
      | ⟨val, isLt⟩ => False.elim ((Nat.not_lt_zero val) isLt)
  | s0 :: Γtail, ⟨i, h⟩ =>
      match i with
      | ⟨0, _⟩ =>
          cons (Tensor.castShape v h.symm) (zero (α := α) (ss := Γtail))
      | ⟨Nat.succ j, hj⟩ =>
          let iTail : Fin Γtail.length := ⟨j, Nat.lt_of_succ_lt_succ hj⟩
          let hTail : Γtail.get iTail = s := by
            simpa using h
          cons (fill (0 : α) s0) (single (Γ := Γtail) (s := s) ⟨iTail, hTail⟩ v)

section

variable [CommSemiring α]

/--
`single` is adjoint to `getIdx` with respect to `dotList`.

Informally: `⟪dx, single idx v⟫ = ⟪getIdx dx idx, v⟫`.
-/
theorem dotList_single {Γ : List Shape} {s : Shape}
    (dx : TList α Γ) (idx : Idx Γ s) (v : Tensor α s) :
    TList.dotList (α := α) dx (single idx v) = dot (α := α) (getIdx (α := α) dx idx) v := by
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
                  TList.dotList (α := α) (TList.cons dx0 dxRest) (single ⟨⟨0, isLt⟩, rfl⟩ v)
                      = dot (α := α) dx0 v := by
                          simp [TList.dotList, single, Tensor.castShape, TList.dotList_zero_right]
                  _ = dot (α := α) (getIdx (α := α) (TList.cons dx0 dxRest) ⟨⟨0, isLt⟩, rfl⟩) v :=
                    by
                          -- `getIdx` at index `0` reduces definitionally to the head tensor.
                          dsimp [getIdx, Tensor.castShape]
                          have hget0 :
                              (TList.cons dx0 dxRest).get (i := (0 : Fin (s :: Γtail).length)) = dx0
                                := by
                            rfl
                          exact (congrArg (fun t => dot (α := α) t v) hget0).symm
            | succ j =>
                have h0 : dot (α := α) dx0 (fill (0 : α) s0) = 0 :=
                  TensorAlgebra.dot_fill_zero_right (α := α) (s := s0) (a := dx0)
                let iHead : Fin (s0 :: Γtail).length := ⟨j + 1, isLt⟩
                let iTail : Fin Γtail.length := ⟨j, Nat.lt_of_succ_lt_succ isLt⟩
                let hTail : Γtail.get iTail = s := by
                  simpa using h
                let idxTail : Idx Γtail s := ⟨iTail, hTail⟩
                have hget :
                    getIdx (α := α) (TList.cons dx0 dxRest) ⟨iHead, h⟩ =
                      getIdx (α := α) dxRest idxTail := by
                  -- Peel off the head entry, then use definitional equality for `cast_shape` after
                  -- rewriting the index proof.
                  dsimp [getIdx, idxTail, iHead, iTail]
                  have hcons :
                      TList.get (α := α) (ss := s0 :: Γtail) (TList.cons dx0 dxRest) iHead =
                        TList.get (α := α) (ss := Γtail) dxRest iTail := by
                    -- `get_cons_succ` is stated with `Nat.succ j`; rewrite to `j + 1`.
                    simpa [Nat.succ_eq_add_one, TList.get, iHead, iTail] using
                      (get_cons_succ (α := α) (s := s0) (ss := Γtail) dx0 dxRest j isLt)
                  -- After rewriting `h`, the two casts become the same.
                  cases h
                  simpa [Tensor.castShape, iHead, iTail] using hcons
                calc
                  TList.dotList (α := α) (TList.cons dx0 dxRest) (single ⟨iHead, h⟩ v)
                      = dot (α := α) dx0 (fill (0 : α) s0) +
                          TList.dotList (α := α) dxRest (single idxTail v) := by
                            simp [TList.dotList, single, idxTail, iHead, iTail]
                  _ = TList.dotList (α := α) dxRest (single idxTail v) := by
                        simp [h0]
                  _ = dot (α := α) (getIdx (α := α) dxRest idxTail) v :=
                        ih (dx := dxRest) (idx := idxTail)
                  _ = dot (α := α) (getIdx (α := α) (TList.cons dx0 dxRest) ⟨iHead, h⟩) v := by
                        simp [hget]

end

end TList

-- Executable node payload (no algebraic assumptions).
--
-- `Δ` is an extra *non-differentiable* environment threaded through evaluation.
-- It is intentionally opaque to the reverse-mode accumulator: `vjp` returns gradients only for `Γ`.
/--
Executable node payload (no correctness proof).

`Δ` is an extra non-differentiable environment threaded through evaluation (e.g. parameters,
auxiliary data). The VJP returns gradients only for the differentiable context `Γ`.
-/
structure NodeData (α : Type) (Δ : Type) (Γ : List Shape) (τ : Shape) where
  /-- forward. -/
  forward : TList α Γ → Δ → Tensor α τ
  /-- jvp. -/
  jvp : TList α Γ → TList α Γ → Δ → Tensor α τ
  /-- vjp. -/
  vjp : TList α Γ → Δ → Tensor α τ → TList α Γ

-- A node with a VJP/JVP adjointness law (proof-carrying).
/--
Proof-carrying node: `NodeData` plus the local adjointness law.

The field `correct` is the algebraic version of the standard JVP/VJP inner-product law.
-/
structure Node {α : Type} [CommSemiring α] (Δ : Type) (Γ : List Shape) (τ : Shape)
    extends NodeData α Δ Γ τ where
  correct : ∀ x dx d δ, dot (α := α) (jvp x dx d) δ = TList.dotList (α := α) dx (vjp x d δ)

-- A tape/SSA graph without local correctness proofs (executable form).
/-- Executable-only graph: a snoc-list of `NodeData`. -/
inductive GraphData (α : Type) (Δ : Type) (Γ : List Shape) : List Shape → Type where
  | nil : GraphData α Δ Γ []
  | snoc {ss : List Shape} {τ : Shape} :
      GraphData α Δ Γ ss → NodeData α Δ (Γ ++ ss) τ → GraphData α Δ Γ (ss ++ [τ])

namespace GraphData

variable {α : Type}
variable {Δ : Type}
variable {Γ : List Shape}

/-- Evaluate a `GraphData` on an input context `x`, producing the full context `Γ ++ ss`. -/
def eval {ss : List Shape} (g : GraphData α Δ Γ ss) (x : TList α Γ) (d : Δ) : TList α (Γ ++ ss) :=
  match g with
  | .nil => TList.cast (α := α) (h := (List.append_nil Γ).symm) x
  | .snoc (ss := ss) (τ := τ) g node =>
      let ctx := eval (ss := ss) g x d
      let y := node.forward ctx d
      TList.cast (α := α) (h := List.append_assoc Γ ss [τ]) (TList.snoc (α := α) (ss := Γ ++ ss) (τ
        := τ) ctx y)

/-- Compute the JVP of `eval`, producing a tangent context of shape `Γ ++ ss`. -/
def jvpCtx {ss : List Shape} (g : GraphData α Δ Γ ss) (x : TList α Γ) (dx : TList α Γ) (d : Δ) :
    TList α (Γ ++ ss) :=
  match g with
  | .nil => TList.cast (α := α) (h := (List.append_nil Γ).symm) dx
  | .snoc (ss := ss) (τ := τ) g node =>
      let ctx := eval (ss := ss) g x d
      let dctx := jvpCtx (ss := ss) g x dx d
      let dy := node.jvp ctx dctx d
      TList.cast (α := α) (h := List.append_assoc Γ ss [τ]) (TList.snoc (α := α) (ss := Γ ++ ss) (τ
        := τ) dctx dy)

/-- Reverse-mode accumulation on contexts (VJP), given a seed cotangent for `Γ ++ ss`. -/
def backpropCtx [Add α] {ss : List Shape} (g : GraphData α Δ Γ ss) (x : TList α Γ) (d : Δ)
    (seed : TList α (Γ ++ ss)) : TList α Γ :=
  match g with
  | .nil => TList.cast (α := α) (h := List.append_nil Γ) seed
  | .snoc (ss := ss) (τ := τ) g node =>
      let seed' : TList α ((Γ ++ ss) ++ [τ]) :=
        TList.cast (α := α) (h := (List.append_assoc Γ ss [τ]).symm) seed
      let seedPrev : TList α (Γ ++ ss) := (TList.unsnoc (α := α) (ss := Γ ++ ss) (τ := τ) seed').1
      let seedOut : Tensor α τ := (TList.unsnoc (α := α) (ss := Γ ++ ss) (τ := τ) seed').2
      let ctx := eval (ss := ss) g x d
      let contrib := node.vjp ctx d seedOut
      let seedPrev' := TList.add (α := α) (ss := Γ ++ ss) seedPrev contrib
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
inductive Graph {α : Type} [CommSemiring α] (Δ : Type) (Γ : List Shape) : List Shape → Type where
  | nil : Graph Δ Γ []
  | snoc {ss : List Shape} {τ : Shape} :
      Graph Δ Γ ss → Node (α := α) (Δ := Δ) (Γ := Γ ++ ss) τ → Graph Δ Γ (ss ++ [τ])

namespace Graph

variable {α : Type} [CommSemiring α]
variable {Δ : Type}
variable {Γ : List Shape}

/-- Forget local correctness proofs, yielding an executable `GraphData`. -/
def toData {ss : List Shape} : Graph (α := α) Δ Γ ss → GraphData α Δ Γ ss
  | .nil => .nil
  | .snoc g node => .snoc (toData (ss := _) g) node.toNodeData

end Graph

namespace Graph

variable {α : Type} [CommSemiring α]
variable {Δ : Type}
variable {Γ : List Shape}

/-- Evaluate a proof-carrying `Graph` on an input context `x`. -/
def eval {ss : List Shape} (g : Graph (α := α) Δ Γ ss) (x : TList α Γ) (d : Δ) : TList α (Γ ++ ss)
  :=
  match g with
  | .nil => TList.cast (α := α) (h := (List.append_nil Γ).symm) x
  | .snoc (ss := ss) (τ := τ) g node =>
      let ctx := eval (ss := ss) g x d
      let y := node.forward ctx d
      TList.cast (α := α) (h := List.append_assoc Γ ss [τ]) (TList.snoc (α := α) (ss := Γ ++ ss) (τ
        := τ) ctx y)

/-- Compute the JVP of `eval`, producing a tangent context of shape `Γ ++ ss`. -/
def jvpCtx {ss : List Shape} (g : Graph (α := α) Δ Γ ss) (x : TList α Γ) (dx : TList α Γ) (d : Δ) :
    TList α (Γ ++ ss) :=
  match g with
  | .nil => TList.cast (α := α) (h := (List.append_nil Γ).symm) dx
  | .snoc (ss := ss) (τ := τ) g node =>
      let ctx := eval (ss := ss) g x d
      let dctx := jvpCtx (ss := ss) g x dx d
      let dy := node.jvp ctx dctx d
      TList.cast (α := α) (h := List.append_assoc Γ ss [τ]) (TList.snoc (α := α) (ss := Γ ++ ss) (τ
        := τ) dctx dy)

/-- Reverse-mode accumulation on contexts (VJP) for a proof-carrying `Graph`. -/
def backpropCtx {ss : List Shape} (g : Graph (α := α) Δ Γ ss) (x : TList α Γ) (d : Δ)
    (seed : TList α (Γ ++ ss)) : TList α Γ :=
  match g with
  | .nil => TList.cast (α := α) (h := List.append_nil Γ) seed
  | .snoc (ss := ss) (τ := τ) g node =>
      let seed' : TList α ((Γ ++ ss) ++ [τ]) :=
        TList.cast (α := α) (h := (List.append_assoc Γ ss [τ]).symm) seed
      let seedPrev : TList α (Γ ++ ss) := (TList.unsnoc (α := α) (ss := Γ ++ ss) (τ := τ) seed').1
      let seedOut : Tensor α τ := (TList.unsnoc (α := α) (ss := Γ ++ ss) (τ := τ) seed').2
      let ctx := eval (ss := ss) g x d
      let contrib := node.vjp ctx d seedOut
      let seedPrev' := TList.add (α := α) (ss := Γ ++ ss) seedPrev contrib
      backpropCtx (ss := ss) g x d seedPrev'

/--
Global tape soundness (algebraic form).

Assuming each node satisfies its local adjointness law, `backpropCtx` is the adjoint of `jvpCtx`
with respect to `TList.dotList`.
-/
theorem backprop_correct {ss : List Shape} (g : Graph (α := α) Δ Γ ss) :
    ∀ x dx d seed,
      TList.dotList (α := α) (jvpCtx (ss := ss) g x dx d) seed =
        TList.dotList (α := α) dx (backpropCtx (ss := ss) g x d seed) := by
  induction g with
  | nil =>
    intro x dx d seed
    simpa [jvpCtx, backpropCtx] using
      (TList.dotList_cast_left (α := α) (h := (List.append_nil Γ).symm) (x := dx) (y := seed))
  | snoc g node ih =>
    intro x dx d seed
    rename_i ss τ
    let ctx := eval (ss := ss) g x d
    let dctx := jvpCtx (ss := ss) g x dx d
    let dy := node.jvp ctx dctx d
    let assoc : (Γ ++ ss) ++ [τ] = Γ ++ (ss ++ [τ]) := List.append_assoc Γ ss [τ]
    let seed' : TList α ((Γ ++ ss) ++ [τ]) := TList.cast (α := α) (h := assoc.symm) seed
    let seedPrev : TList α (Γ ++ ss) := (TList.unsnoc (α := α) (ss := Γ ++ ss) (τ := τ) seed').1
    let seedOut : Tensor α τ := (TList.unsnoc (α := α) (ss := Γ ++ ss) (τ := τ) seed').2
    have hseed : TList.snoc (α := α) (ss := Γ ++ ss) (τ := τ) seedPrev seedOut = seed' := by
      simpa [seedPrev, seedOut] using (TList.snoc_unsnoc (α := α) (ss := Γ ++ ss) (τ := τ) (xs :=
        seed'))
    have hjvp :
        TList.dotList (α := α) (jvpCtx (ss := ss ++ [τ]) (Graph.snoc g node) x dx d) seed =
          TList.dotList (α := α) dctx seedPrev + dot (α := α) dy seedOut := by
      have :
          TList.dotList (α := α) (TList.snoc (α := α) (ss := Γ ++ ss) (τ := τ) dctx dy) seed' =
            TList.dotList (α := α) dctx seedPrev + dot (α := α) dy seedOut := by
        simpa [hseed] using
          (TList.dotList_snoc (α := α) (ss := Γ ++ ss) (τ := τ) (x := dctx) (y := seedPrev) (a :=
            dy) (b := seedOut))
      simpa [jvpCtx, ctx, dctx, dy, seed', assoc] using
        (TList.dotList_cast_left (α := α) (h := assoc) (x := TList.snoc (α := α) (ss := Γ ++ ss) (τ
          := τ) dctx dy) (y := seed)
          |>.trans this)
    have hlocal : dot (α := α) dy seedOut = TList.dotList (α := α) dctx (node.vjp ctx d seedOut) :=
      by
      simpa [dy] using (node.correct ctx dctx d seedOut)
    have hadd :
        TList.dotList (α := α) dctx seedPrev + TList.dotList (α := α) dctx (node.vjp ctx d seedOut)
          =
          TList.dotList (α := α) dctx (TList.add (α := α) (ss := Γ ++ ss) seedPrev (node.vjp ctx d
            seedOut)) := by
      simpa using
        (TList.dotList_add_right (α := α) (x := dctx) (y := seedPrev) (z := node.vjp ctx d
          seedOut)).symm
    calc
      TList.dotList (α := α) (jvpCtx (ss := ss ++ [τ]) (Graph.snoc g node) x dx d) seed
          = TList.dotList (α := α) dctx seedPrev + dot (α := α) dy seedOut := hjvp
      _ = TList.dotList (α := α) dctx seedPrev + TList.dotList (α := α) dctx (node.vjp ctx d
        seedOut) := by
            simp [hlocal]
      _ = TList.dotList (α := α) dctx (TList.add (α := α) (ss := Γ ++ ss) seedPrev (node.vjp ctx d
        seedOut)) := by
            simp [hadd]
      _ = TList.dotList (α := α) dx
            (backpropCtx (ss := ss) g x d (TList.add (α := α) (ss := Γ ++ ss) seedPrev (node.vjp ctx
              d seedOut))) := by
            simpa [dctx] using (ih x dx d (TList.add (α := α) (ss := Γ ++ ss) seedPrev (node.vjp ctx
              d seedOut)))
      _ = TList.dotList (α := α) dx (backpropCtx (ss := ss ++ [τ]) (Graph.snoc g node) x d seed) :=
        by
            simp [backpropCtx, ctx, seed', seedPrev, seedOut]

end Graph

end

end Algebra
end Autograd
end Proofs
