/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Core.RealCorrectness
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

open Spec
open Tensor

/-
In dependently-typed code, the same cast can arise with different proof terms. Since equality proofs
are proof-irrelevant, we can normalize these casts away when they show up during definitional
reductions (e.g. `List.get` on `Fin` indices).
-/
theorem tensor_cast_shape_proof_irrel {α : Type} {s₁ s₂ : Shape}
    (t : Tensor α s₁) (p q : s₁ = s₂) :
    Tensor.castShape (t := t) p = Tensor.castShape (t := t) q := by
  have : p = q := Subsingleton.elim _ _
  cases this
  rfl

noncomputable section

/--
A heterogeneous list of tensors indexed by a list of shapes.

This is the “typed context” used by the tape model: `TList Γ` stores one tensor for each shape in
the list `Γ`.

PyTorch analogy: the tape stores “saved tensors”/intermediates for backward, but PyTorch stores
them in an untyped runtime list; here the shapes are tracked in the type.

Implementation note: we reuse the *type-level* context container from the backend-generic tape
development (`Proofs.Autograd.Algebra.TList`) and specialize it to `ℝ`. This avoids duplicating the
basic “heterogeneous list indexed by shapes” encoding in two different places.
-/
abbrev TList : List Shape → Type :=
  Proofs.Autograd.Algebra.TList ℝ

namespace TList

variable {ss : List Shape}

/--
Constructor aliases for the specialized `TList`.

We reuse the inductive type from `Proofs.Autograd.Algebra.TList`, so its constructors are actually
`Proofs.Autograd.Algebra.TList.nil/cons`. A few analytic files expect the shorter names
`Proofs.Autograd.TList.nil/cons`, so we provide them here as abbreviations.
-/
abbrev nil : TList ([] : List Shape) :=
  Proofs.Autograd.Algebra.TList.nil (α := ℝ)

/-- Constructor alias for `TList.cons` specialized to `ℝ`. -/
abbrev cons {s : Shape} {ss : List Shape} (x : Tensor ℝ s) (xs : TList ss) : TList (s :: ss) :=
  Proofs.Autograd.Algebra.TList.cons (α := ℝ) (s := s) (ss := ss) x xs

/-- Get the `i`th tensor from a context, with its shape tracked by `List.get`. -/
abbrev get : {ss : List Shape} → TList ss → (i : Fin ss.length) → Tensor ℝ (ss.get i) :=
  Proofs.Autograd.Algebra.TList.get (α := ℝ)

/-- All-zero context (fills each tensor entry with zeros). -/
abbrev zero : {ss : List Shape} → TList ss :=
  Proofs.Autograd.Algebra.TList.zero (α := ℝ)

/-- Pointwise addition of two contexts of the same shape list. -/
abbrev add : {ss : List Shape} → TList ss → TList ss → TList ss :=
  Proofs.Autograd.Algebra.TList.add (α := ℝ)

/-- Append a tensor to the end of a context. -/
abbrev snoc {τ : Shape} : {ss : List Shape} → TList ss → Tensor ℝ τ → TList (ss ++ [τ]) :=
  Proofs.Autograd.Algebra.TList.snoc (α := ℝ) (τ := τ)

/-- Split a context of shape list `ss ++ [τ]` into its prefix and last tensor. -/
abbrev unsnoc {τ : Shape} : {ss : List Shape} → TList (ss ++ [τ]) → TList ss × Tensor ℝ τ :=
  Proofs.Autograd.Algebra.TList.unsnoc (α := ℝ) (τ := τ)

/--
Dot product over contexts: sum of per-entry tensor dot products.

Informally: `dotList xs ys` is the “context inner product” used to state global adjointness for
tape evaluation and backprop.
-/
def dotList : {ss : List Shape} → TList ss → TList ss → ℝ
  | [], .nil, .nil => 0
  | _ :: ss, .cons a as, .cons b bs => dot a b + dotList (ss := ss) as bs

/-- Cast a context along an equality of shape lists. -/
abbrev cast {ss₁ ss₂ : List Shape} (h : ss₁ = ss₂) (xs : TList ss₁) : TList ss₂ :=
  Proofs.Autograd.Algebra.TList.cast (α := ℝ) (ss₁ := ss₁) (ss₂ := ss₂) h xs

export Proofs.Autograd.Algebra.TList (cast_rfl cast_cast cast_symm)

theorem dotList_cast_left {ss₁ ss₂ : List Shape} (h : ss₁ = ss₂) (x : TList ss₁) (y : TList ss₂) :
    dotList (cast h x) y = dotList x (cast h.symm y) := by
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
`dotList` is linear in its right argument with respect to `TList.add`.

Informally: `⟪x, y + z⟫ = ⟪x, y⟫ + ⟪x, z⟫` for contexts.
-/
theorem dotList_add_right {ss : List Shape} (x y z : TList ss) :
    dotList x (add y z) = dotList x y + dotList x z := by
  induction ss with
  | nil =>
    cases x; cases y; cases z; simp [dotList, Proofs.Autograd.Algebra.TList.add]
  | cons s ss ih =>
    cases x with
    | cons xh xt =>
      cases y with
      | cons yh yt =>
        cases z with
        | cons zh zt =>
          simp [dotList, Proofs.Autograd.Algebra.TList.add, dot_add_right, ih, add_assoc,
            add_left_comm]

/--
`dotList` respects appending: dot of two `snoc`ed contexts splits into prefix + last entry.

Informally: `⟪(x,a), (y,b)⟫ = ⟪x,y⟫ + ⟪a,b⟫`.
-/
theorem dotList_snoc {ss : List Shape} {τ : Shape} (x y : TList ss) (a b : Tensor ℝ τ) :
    dotList (snoc x a) (snoc y b) = dotList x y + dot a b := by
  revert x y
  induction ss with
  | nil =>
    intro x y
    cases x; cases y
    simp [dotList, Proofs.Autograd.Algebra.TList.snoc]
  | cons s ss ih =>
    intro x y
    cases x with
    | cons xh xt =>
      cases y with
      | cons yh yt =>
        simp [dotList, Proofs.Autograd.Algebra.TList.snoc, ih, add_left_comm, add_comm]

/-- `unsnoc` is a left inverse of `snoc`. -/
theorem unsnoc_snoc {ss : List Shape} {τ : Shape} (xs : TList ss) (t : Tensor ℝ τ) :
    unsnoc (ss := ss) (snoc xs t) = (xs, t) := by
  induction ss with
  | nil =>
    cases xs
    simp [Proofs.Autograd.Algebra.TList.snoc, Proofs.Autograd.Algebra.TList.unsnoc]
  | cons s ss ih =>
    cases xs with
    | cons x xt =>
      simp [Proofs.Autograd.Algebra.TList.snoc, Proofs.Autograd.Algebra.TList.unsnoc, ih]

/-- `snoc` is a right inverse of `unsnoc`. -/
theorem snoc_unsnoc {ss : List Shape} {τ : Shape} (xs : TList (ss ++ [τ])) :
    snoc (unsnoc (ss := ss) xs).1 (unsnoc (ss := ss) xs).2 = xs := by
  induction ss with
  | nil =>
    cases xs with
    | cons t xs =>
      cases xs with
      | nil =>
        simp [Proofs.Autograd.Algebra.TList.unsnoc, Proofs.Autograd.Algebra.TList.snoc]
  | cons s ss ih =>
    cases xs with
    | cons x xs =>
      have hrec : snoc (unsnoc (ss := ss) xs).1 (unsnoc (ss := ss) xs).2 = xs := ih (xs := xs)
      simp [Proofs.Autograd.Algebra.TList.unsnoc, Proofs.Autograd.Algebra.TList.snoc, hrec]

private theorem fill_eq_scale_one {s : Shape} (c : ℝ) :
    fill c s = scaleSpec (α:=ℝ) (s:=s) (fill (1 : ℝ) s) c := by
  induction s with
  | scalar =>
    simp [fill, scaleSpec, mapSpec]
  | dim n s ih =>
    simp [fill, scaleSpec, mapSpec, ih]

/--
Dotting any tensor with a zero-filled tensor gives `0`.

This is the tensor-level fact used to show that “one-hot” cotangents behave as expected.
-/
theorem dot_fill_zero_right {s : Shape} (a : Tensor ℝ s) :
    dot a (fill (0 : ℝ) s) = 0 := by
  have hfill : fill (0 : ℝ) s = scaleSpec (α:=ℝ) (s:=s) (fill (1 : ℝ) s) 0 := by
    simpa using (fill_eq_scale_one (s := s) (c := (0 : ℝ)))
  calc
    dot a (fill (0 : ℝ) s)
        = dot a (scaleSpec (α:=ℝ) (s:=s) (fill (1 : ℝ) s) 0) := by
            simp [hfill]
    _ = dot (scaleSpec (α:=ℝ) (s:=s) (fill (1 : ℝ) s) 0) a := by
          simpa using (dot_comm (a := a) (b := scaleSpec (α:=ℝ) (s:=s) (fill (1 : ℝ) s) 0))
    _ = 0 * dot (fill (1 : ℝ) s) a := by
          simpa using (dot_scale_left (a := fill (1 : ℝ) s) (b := a) (k := (0 : ℝ)))
    _ = 0 := by ring

/-- `dotList x 0 = 0` for the all-zero context. -/
theorem dotList_zero_right {ss : List Shape} (x : TList ss) :
    dotList x (zero (ss := ss)) = 0 := by
  induction ss with
  | nil =>
    cases x
    simp [dotList, Proofs.Autograd.Algebra.TList.zero]
  | cons s ss ih =>
    cases x with
    | cons xh xt =>
      simp [dotList, Proofs.Autograd.Algebra.TList.zero, dot_fill_zero_right, ih]

end TList

/--
An index into a heterogeneous context, carrying a proof of the expected shape.

This lets us talk about “the `i`th saved tensor has shape `s`” without losing the shape invariant.
-/
structure Idx (Γ : List Shape) (s : Shape) where
  /-- i. -/
  i : Fin Γ.length
  /-- h. -/
  h : Γ.get i = s

/-- Read an element from a context using an index with an attached shape proof. -/
def getIdx {Γ : List Shape} {s : Shape} (xs : TList Γ) (idx : Idx Γ s) : Tensor ℝ s :=
  Tensor.castShape (xs.get idx.i) idx.h

namespace TList

/--
Build a sparse context with a single nonzero entry at `idx` and zeros elsewhere.

This is used to express “one-hot” cotangents when proving local-to-global backprop correctness.
-/
def single {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (v : Tensor ℝ s) : TList Γ :=
  match Γ, idx with
  | [], ⟨i, _h⟩ =>
      match i with
      | ⟨val, isLt⟩ => False.elim ((Nat.not_lt_zero val) isLt)
  | s0 :: Γtail, ⟨i, h⟩ =>
      match i with
      | ⟨0, _⟩ =>
          .cons (Tensor.castShape v h.symm) (zero (ss := Γtail))
      | ⟨Nat.succ j, hj⟩ =>
          let iTail : Fin Γtail.length := ⟨j, Nat.lt_of_succ_lt_succ hj⟩
          let hTail : Γtail.get iTail = s := by
            simpa using h
          .cons (fill (0 : ℝ) s0) (single (Γ := Γtail) (s := s) ⟨iTail, hTail⟩ v)

/--
`single idx v` is the “one-hot” context with value `v` at `idx`, and zeros elsewhere.

This lemma says the context dot product against `single idx v` picks out the corresponding entry
of `dx`:

`⟪dx, single idx v⟫ = ⟪dx[idx], v⟫`.
-/
theorem dotList_single {Γ : List Shape} {s : Shape}
    (dx : TList Γ) (idx : Idx Γ s) (v : Tensor ℝ s) :
    TList.dotList dx (single idx v) = dot (getIdx dx idx) v := by
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
              cases h
              have hs : (s0 :: Γtail).get ⟨0, isLt⟩ = s0 := by rfl
              cases hs
              -- Head case: `single` puts `v` in the head slot and zeros elsewhere, so `dotList`
              -- picks `dx0`.
              have hget0 : (Algebra.TList.cons dx0 dxRest).get 0 = dx0 := by
                have hz : (0 : Fin (s0 :: Γtail).length) = ⟨0, isLt⟩ := by
                  apply Fin.ext
                  simp
                -- `get_cons_zero` is the stable simp lemma for head access.
                simpa [hz, TList.get] using
                  (Proofs.Autograd.Algebra.TList.get_cons_zero (α := ℝ) (s := s0) (ss := Γtail) dx0
                    dxRest isLt)
              simpa [TList.dotList, single, getIdx, Tensor.castShape, TList.get, TList.zero,
                TList.dotList_zero_right] using congrArg (fun t => dot t v) hget0.symm
            | succ j =>
              -- tail index
              have h0 : dot dx0 (fill (0 : ℝ) s0) = 0 := dot_fill_zero_right (a := dx0)
              let iHead : Fin (s0 :: Γtail).length := ⟨Nat.succ j, isLt⟩
              let iTail : Fin Γtail.length := ⟨j, Nat.lt_of_succ_lt_succ isLt⟩
              let hTail : Γtail.get iTail = s := by
                simpa using h
              let idxTail : Idx Γtail s := ⟨iTail, hTail⟩
              have hget : getIdx (.cons dx0 dxRest) ⟨iHead, h⟩ = getIdx dxRest idxTail := by
                -- Reduce the `get` at a successor index, then discharge cast-proof mismatches by
                -- proof-irrelevance.
                dsimp [getIdx]
                simp [TList.get, iHead]
                exact tensor_cast_shape_proof_irrel (t := dxRest.get iTail) _ _
              calc
                TList.dotList (.cons dx0 dxRest) (single ⟨iHead, h⟩ v)
                    = dot dx0 (fill (0 : ℝ) s0) + TList.dotList dxRest (single idxTail v) := by
                        simp [TList.dotList, single, idxTail, iHead, iTail]
                _ = TList.dotList dxRest (single idxTail v) := by
                      simp [h0]
                _ = dot (getIdx dxRest idxTail) v := by
                      simpa using (ih (dx := dxRest) (idx := idxTail))
                _ = dot (getIdx (.cons dx0 dxRest) ⟨iHead, h⟩) v := by
                      simp [hget]

end TList

/-- A node with local JVP/VJP and an adjointness proof against the tensor dot product. -/
structure Node (Γ : List Shape) (τ : Shape) where
  /-- forward. -/
  forward : TList Γ → Tensor ℝ τ
  /-- jvp. -/
  jvp : TList Γ → TList Γ → Tensor ℝ τ
  /-- vjp. -/
  vjp : TList Γ → Tensor ℝ τ → TList Γ
  /-- correct. -/
  correct : ∀ x dx δ, dot (jvp x dx) δ = TList.dotList dx (vjp x δ)

/-- A tape/SSA graph: nodes are appended in topological order and may reference any previous value.
  -/
inductive Graph (Γ : List Shape) : List Shape → Type where
  | nil : Graph Γ []
  | snoc {ss : List Shape} {τ : Shape} :
      Graph Γ ss → Node (Γ ++ ss) τ → Graph Γ (ss ++ [τ])

namespace Graph

variable {Γ : List Shape}

/-- Evaluate a tape/graph, returning the full context (`inputs ++ intermediates`). -/
def eval {ss : List Shape} (g : Graph Γ ss) (x : TList Γ) : TList (Γ ++ ss) :=
  match g with
  | .nil => TList.cast (h := (List.append_nil Γ).symm) x
  | .snoc (ss := ss) (τ := τ) g node =>
      let ctx := eval (ss := ss) g x
      let y := node.forward ctx
      TList.cast (h := List.append_assoc Γ ss [τ]) (TList.snoc ctx y)

/--
Evaluate the JVP (“forward-mode tangent”) of a graph, producing tangents for all values in the
extended context `Γ ++ ss`.
-/
def jvpCtx {ss : List Shape} (g : Graph Γ ss) (x : TList Γ) (dx : TList Γ) : TList (Γ ++ ss) :=
  match g with
  | .nil => TList.cast (h := (List.append_nil Γ).symm) dx
  | .snoc (ss := ss) (τ := τ) g node =>
      let ctx := eval (ss := ss) g x
      let dctx := jvpCtx (ss := ss) g x dx
      let dy := node.jvp ctx dctx
      TList.cast (h := List.append_assoc Γ ss [τ]) (TList.snoc dctx dy)

/--
Reverse-mode backpropagation on a tape/graph, returning gradients for the *inputs* `Γ`.

This is the proof model of what PyTorch calls “running backward” starting from an output seed
cotangent and accumulating gradients at shared parents.
-/
def backpropCtx {ss : List Shape} (g : Graph Γ ss) (x : TList Γ) (seed : TList (Γ ++ ss)) : TList Γ
  :=
  match g with
  | .nil => TList.cast (h := List.append_nil Γ) seed
  | .snoc (ss := ss) (τ := τ) g node =>
      -- Reassociate the context so we can `unsnoc`.
      let seed' : TList ((Γ ++ ss) ++ [τ]) :=
        TList.cast (h := (List.append_assoc Γ ss [τ]).symm) seed
      let seedPrev : TList (Γ ++ ss) := (TList.unsnoc (ss := Γ ++ ss) seed').1
      let seedOut : Tensor ℝ τ := (TList.unsnoc (ss := Γ ++ ss) seed').2
      let ctx := eval (ss := ss) g x
      let contrib := node.vjp ctx seedOut
      let seedPrev' := TList.add seedPrev contrib
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
      TList.dotList (jvpCtx (ss := ss) g x dx) seed =
        TList.dotList dx (backpropCtx (ss := ss) g x seed) := by
  induction g with
  | nil =>
    intro x dx seed
    -- `ss = []` so this is exactly the dotList/cast adjointness.
    simpa [jvpCtx, backpropCtx] using
      (TList.dotList_cast_left (h := (List.append_nil Γ).symm) (x := dx) (y := seed))
  | snoc g node ih =>
    intro x dx seed
    rename_i ss τ
    let ctx := eval (ss := ss) g x
    let dctx := jvpCtx (ss := ss) g x dx
    let dy := node.jvp ctx dctx
    let assoc : (Γ ++ ss) ++ [τ] = Γ ++ (ss ++ [τ]) := List.append_assoc Γ ss [τ]
    let seed' : TList ((Γ ++ ss) ++ [τ]) := TList.cast (h := assoc.symm) seed
    let seedPrev : TList (Γ ++ ss) := (TList.unsnoc (ss := Γ ++ ss) seed').1
    let seedOut : Tensor ℝ τ := (TList.unsnoc (ss := Γ ++ ss) seed').2
    have hseed : TList.snoc seedPrev seedOut = seed' := by
      simpa [seedPrev, seedOut] using (TList.snoc_unsnoc (ss := Γ ++ ss) (τ := τ) (xs := seed'))
    have hjvp :
        TList.dotList (jvpCtx (ss := ss ++ [τ]) (Graph.snoc g node) x dx) seed =
          TList.dotList dctx seedPrev + dot dy seedOut := by
      -- Move casts so we can use `dotList_snoc` on reassociated contexts.
      have :
          TList.dotList (TList.snoc dctx dy) seed' =
            TList.dotList dctx seedPrev + dot dy seedOut := by
        simpa [hseed] using (TList.dotList_snoc (x := dctx) (y := seedPrev) (a := dy) (b :=
          seedOut))
      -- Unfold `jvpCtx` at the snoc node, then apply `dotList_cast_left`.
      simpa [jvpCtx, ctx, dctx, dy, seed', assoc] using
        (TList.dotList_cast_left (h := assoc) (x := TList.snoc dctx dy) (y := seed) |>.trans this)
    have hlocal : dot dy seedOut = TList.dotList dctx (node.vjp ctx seedOut) := by
      simpa [dy] using (node.correct ctx dctx seedOut)
    have hadd :
        TList.dotList dctx seedPrev + TList.dotList dctx (node.vjp ctx seedOut) =
          TList.dotList dctx (TList.add seedPrev (node.vjp ctx seedOut)) := by
      simpa using
        (TList.dotList_add_right (x := dctx) (y := seedPrev) (z := node.vjp ctx seedOut)).symm
    calc
      TList.dotList (jvpCtx (ss := ss ++ [τ]) (Graph.snoc g node) x dx) seed
          = TList.dotList dctx seedPrev + dot dy seedOut := hjvp
      _ = TList.dotList dctx seedPrev + TList.dotList dctx (node.vjp ctx seedOut) := by
            simp [hlocal]
      _ = TList.dotList dctx (TList.add seedPrev (node.vjp ctx seedOut)) := by
            simp [hadd]
      _ = TList.dotList dx (backpropCtx (ss := ss) g x (TList.add seedPrev (node.vjp ctx seedOut)))
        := by
            simpa [dctx] using (ih x dx (TList.add seedPrev (node.vjp ctx seedOut)))
      _ = TList.dotList dx (backpropCtx (ss := ss ++ [τ]) (Graph.snoc g node) x seed) := by
            simp [backpropCtx, ctx, seed', seedPrev, seedOut]

end Graph

end
end Autograd
end Proofs
