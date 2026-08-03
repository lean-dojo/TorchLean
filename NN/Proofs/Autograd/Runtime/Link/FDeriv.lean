/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Runtime.Link.BackwardGraph
public import NN.Proofs.Autograd.Tape.Core.FDeriv

/-!
# Analytic upgrade of the runtime link: reverse mode computes `(fderiv eval)†`

Two independent developments meet in this file.

* The **runtime link** (`Runtime/Link/BackwardGraph.lean`) proves that the tape engine's dense
  reverse pass (`Tape.backwardDenseFrom`) agrees with the algebraic model's full backpropagation
  `backpropAllCtx`, over any commutative semiring carrier and an arbitrary non-differentiable
  environment `Δ`.
* The **analytic tape model** (`Tape/Core/FDeriv.lean`) proves that reverse-mode accumulation
  computes the adjoint of the Fréchet derivative of the forward evaluation, over `ℝ`.

The two developments are stated on *different graph types*: the algebraic
`Algebra.Graph (α := α) Δ Γ ss` versus the `ℝ`-monomorphic, environment-free
`Proofs.Autograd.Graph Γ ss`. Their contexts already coincide (`TList Γ` is by definition
`Algebra.TList ℝ Γ`), and their `eval`/`jvpCtx`/`backpropCtx` recursions mirror each other
node for node — what has been missing is the formal connection. This file supplies it:

* **The slice isomorphism.** `Algebra.Node.toReal` / `Algebra.Graph.toReal` specialize an
  algebraic graph at `α := ℝ` and a fixed environment `d : Δ` to an analytic graph;
  `Node.toAlgebra` / `Graph.toAlgebra` embed an analytic graph back as the `Δ := Unit` slice,
  and the round trip is the identity (`toAlgebra_toReal`). The analytic model is therefore
  *exactly* the environment-free `ℝ` slice of the algebraic model. The commutation lemmas
  `toReal_eval`, `toReal_jvpCtx`, `toReal_backpropCtx` show the specialization preserves all
  three semantics.
* **Input-prefix extraction.** `TList.takeLeft` reads the input (`Γ`-prefix) block out of a
  full context, and `takeLeft_backpropAllCtx` (also in `GraphData` form) identifies the input
  block of the full backpropagation with the inputs-only `backpropCtx` — the missing lemma
  relating the runtime-facing `backpropAllCtx` to the proof-facing `backpropCtx`.
* **Vectorization transport.** The `flattenCtx_*` lemmas commute context vectorization with
  `cast`/`snoc`/`unsnoc`/`add`, so the `TList`-level graph semantics coincide with the
  Euclidean `CtxVec` semantics: `evalVec_flattenCtx`, `jvpVec_flattenCtx`,
  `backpropVec_flattenCtx`.
* **The composed endpoints.** `backpropCtx_eq_adjoint_fderiv` upgrades the algebraic
  backpropagation at `ℝ` to the Fréchet-adjoint characterization, and
  `backwardDenseFrom_compileAux_adjoint_fderiv` combines it with the runtime link: the tape
  model's dense reverse pass on a compiled graph, instantiated at `α := ℝ`, succeeds with the
  full backpropagation context, whose input prefix is exactly `(fderiv ℝ eval x)† seed`. The
  `_at` variants assume
  differentiability only at the actual execution point (`GraphFDerivCorrectAt`), covering
  graphs with non-smooth primitives (`relu`, `abs`, `min`/`max`, `log`, `sqrt`, …).

Throughout, "reverse pass" means the exact tape model: the `Tape.backwardDenseFrom` program of
`Runtime/Autograd` instantiated at the exact carrier `α := ℝ`. Nothing in this file is a
statement about the native `Float` evaluation or the CUDA execution path; relating those to the
exact model is a separate (approximation) concern.

A natural follow-up enabled by the round-trip lemmas would be to re-found the analytic model
as an abbreviation of the algebraic one at `α := ℝ`, `Δ := Unit` (as already done for
`TList`); the lemmas here are the correctness specification such a consolidation must satisfy.
-/

@[expose] public section

namespace Proofs
namespace Autograd

open Spec
open Tensor

noncomputable section

-- ---------------------------------------------------------------------------
-- The two dot products agree at `ℝ`
-- ---------------------------------------------------------------------------

/--
The analytic context inner product (`Spec.dot`-based) agrees with the algebraic one
(`TensorAlgebra.dot`-based) at `ℝ`.

Both recursions are the same sum of per-entry tensor dots; the entries agree by
`dot_eq_tensorAlgebra_dot`.
-/
theorem dotList_eq_algebra_dotList {Γ : List Shape} (x y : TList Γ) :
    TList.dotList (ss := Γ) x y = Algebra.TList.dotList (α := ℝ) x y := by
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
        simp [TList.dotList, Algebra.TList.dotList, dot_eq_tensorAlgebra_dot, ih]

-- ---------------------------------------------------------------------------
-- Vectorization transport: `flattenCtx` commutes with the context operations
-- ---------------------------------------------------------------------------

/-- `flattenCtx` commutes with casting a context along a shape-list equality. -/
theorem flattenCtx_cast {Γ₁ Γ₂ : List Shape} (h : Γ₁ = Γ₂) (xs : TList Γ₁) :
    flattenCtx (Γ := Γ₂) (TList.cast h xs) = castCtxVec (Γ₁ := Γ₁) (Γ₂ := Γ₂) h (flattenCtx xs) := by
  cases h
  simp

/-- Vectorization is additive: `toVecT` maps `addSpec` to vector addition. -/
theorem toVecT_addSpec {s : Shape} (a b : Tensor ℝ s) :
    toVecT (t := addSpec a b) = toVecT (t := a) + toVecT (t := b) := by
  refine ext_inner_right ℝ ?_
  intro w
  have hw : w = toVecT (t := ofVecT (s := s) w) := (toVecT_ofVecT (s := s) w).symm
  rw [hw, ← dot_eq_inner_toVecT, inner_add_left, ← dot_eq_inner_toVecT, ← dot_eq_inner_toVecT,
    dot_add_left]

/-- The context inner product is additive in its left argument. -/
private lemma dotList_add_left' {Γ : List Shape} (u v z : TList Γ) :
    TList.dotList (TList.add u v) z = TList.dotList u z + TList.dotList v z := by
  induction Γ with
  | nil =>
    cases u
    cases v
    cases z
    simp [TList.dotList, Algebra.TList.add]
  | cons s Γ ih =>
    cases u with
    | cons uh ut =>
      cases v with
      | cons vh vt =>
        cases z with
        | cons zh zt =>
          show TList.dotList (TList.cons (addSpec uh vh) (TList.add ut vt)) (TList.cons zh zt)
              = _
          simp only [TList.dotList, dot_add_left, ih]
          ring

/-- `flattenCtx` maps context addition to vector addition. -/
theorem flattenCtx_add {Γ : List Shape} (u v : TList Γ) :
    flattenCtx (TList.add u v) = flattenCtx u + flattenCtx v := by
  refine ext_inner_right ℝ ?_
  intro w
  have hw : w = flattenCtx (unflattenCtx (Γ := Γ) w) :=
    (flattenCtx_unflattenCtx (Γ := Γ) w).symm
  rw [hw, ← dotList_eq_inner_flattenCtx, inner_add_left, ← dotList_eq_inner_flattenCtx,
    ← dotList_eq_inner_flattenCtx]
  exact dotList_add_left' u v _

/-- `flattenCtx` maps `TList.snoc` to `snocCtx`. -/
theorem flattenCtx_snoc {Γ : List Shape} {τ : Shape} (xs : TList Γ) (y : Tensor ℝ τ) :
    flattenCtx (Γ := Γ ++ [τ]) (TList.snoc xs y)
      = snocCtx (Γ := Γ) (τ := τ) (flattenCtx xs) (toVecT (t := y)) := by
  induction Γ with
  | nil =>
    cases xs
    show flattenCtx (TList.cons y TList.nil) = _
    ext i
    have hi : (i : Nat) < Spec.Shape.size τ := by
      have h := i.isLt
      simp only [ctxSize, Nat.add_zero] at h
      exact h
    simp [flattenCtx, snocCtx, appendVec, castVec_apply, Fin.append, Fin.addCases, hi, ctxSize]
    all_goals rfl
  | cons s Γ ih =>
    cases xs with
    | cons xh xt =>
      show flattenCtx (TList.cons xh (TList.snoc xt y)) = _
      simp only [flattenCtx, ih]
      ext i
      induction i using Fin.addCases with
      | left i =>
        have h1 : (i : Nat) < ctxSize (s :: Γ) := by
          have h := i.isLt
          simp only [ctxSize]
          exact Nat.lt_of_lt_of_le h (Nat.le_add_right _ _)
        simp [snocCtx, appendVec, castVec_apply, Fin.append, Fin.addCases, h1]
        all_goals rfl
      | right i =>
        by_cases h2 : (i : Nat) < ctxSize Γ
        · have h3 : Spec.Shape.size s + (i : Nat) < ctxSize (s :: Γ) := by
            simp only [ctxSize]
            exact Nat.add_lt_add_left h2 _
          simp [snocCtx, appendVec, castVec_apply, Fin.append, Fin.addCases,
            h2, h3, Fin.subNat, Fin.natAdd]
          all_goals rfl
        · have h3 : ¬ Spec.Shape.size s + (i : Nat) < ctxSize (s :: Γ) := by
            simp only [ctxSize]
            exact fun hc => h2 (Nat.lt_of_add_lt_add_left hc)
          have h4 : Spec.Shape.size s + (i : Nat) - ctxSize (s :: Γ) = (i : Nat) - ctxSize Γ := by
            simp only [ctxSize]
            exact Nat.add_sub_add_left _ _ _
          simp [snocCtx, appendVec, castVec_apply, Fin.append, Fin.addCases,
            h2, h3, h4, Fin.subNat, Fin.natAdd]

/-- `flattenCtx` maps `TList.unsnoc` to `unsnocCtx`. -/
theorem unsnocCtx_flattenCtx {Γ : List Shape} {τ : Shape} (w : TList (Γ ++ [τ])) :
    unsnocCtx (Γ := Γ) (τ := τ) (flattenCtx w)
      = (flattenCtx (TList.unsnoc w).1, toVecT (t := (TList.unsnoc w).2)) := by
  conv_lhs => rw [← Algebra.TList.snoc_unsnoc (α := ℝ) (ss := Γ) (τ := τ) (xs := w)]
  rw [flattenCtx_snoc, unsnocCtx_snocCtx]

namespace Node

/-- The vectorized forward map, evaluated on a flattened context. -/
theorem forwardVec_flattenCtx {Γ : List Shape} {τ : Shape} (node : Node Γ τ) (x : TList Γ) :
    node.forwardVec (Γ := Γ) (τ := τ) (flattenCtx x) = toVecT (t := node.forward x) := by
  simp [Node.forwardVec]

/-- The vectorized JVP, evaluated on flattened contexts. -/
theorem jvpVec_flattenCtx {Γ : List Shape} {τ : Shape} (node : Node Γ τ) (x dx : TList Γ) :
    node.jvpVec (Γ := Γ) (τ := τ) (flattenCtx x) (flattenCtx dx)
      = toVecT (t := node.jvp x dx) := by
  simp [Node.jvpVec]

/-- The vectorized VJP, evaluated on a flattened context and a vectorized cotangent. -/
theorem vjpVec_flattenCtx {Γ : List Shape} {τ : Shape} (node : Node Γ τ) (x : TList Γ)
    (δ : Tensor ℝ τ) :
    node.vjpVec (Γ := Γ) (τ := τ) (flattenCtx x) (toVecT (t := δ))
      = flattenCtx (node.vjp x δ) := by
  simp [Node.vjpVec]

end Node

namespace Graph

variable {Γ : List Shape}

/-- The Euclidean graph evaluation is the flattening of the `TList` evaluation. -/
theorem evalVec_flattenCtx {ss : List Shape} (g : Graph Γ ss) (x : TList Γ) :
    evalVec (Γ := Γ) (ss := ss) g (flattenCtx x) = flattenCtx (eval (Γ := Γ) (ss := ss) g x) := by
  induction g with
  | nil =>
    simp [evalVec, eval, flattenCtx_cast]
  | snoc g node ih =>
    simp [evalVec, eval, flattenCtx_cast, flattenCtx_snoc, ih, Node.forwardVec_flattenCtx]

/-- The Euclidean graph JVP is the flattening of the `TList` JVP. -/
theorem jvpVec_flattenCtx {ss : List Shape} (g : Graph Γ ss) (x dx : TList Γ) :
    jvpVec (Γ := Γ) (ss := ss) g (flattenCtx x) (flattenCtx dx)
      = flattenCtx (jvpCtx (Γ := Γ) (ss := ss) g x dx) := by
  induction g with
  | nil =>
    simp [jvpVec, jvpCtx, flattenCtx_cast]
  | snoc g node ih =>
    simp [jvpVec, jvpCtx, flattenCtx_cast, flattenCtx_snoc, ih, evalVec_flattenCtx,
      Node.jvpVec_flattenCtx]

/-- The Euclidean reverse pass is the flattening of the `TList` reverse pass. -/
theorem backpropVec_flattenCtx {ss : List Shape} (g : Graph Γ ss) (x : TList Γ)
    (seed : TList (Γ ++ ss)) :
    backpropVec (Γ := Γ) (ss := ss) g (flattenCtx x) (flattenCtx seed)
      = flattenCtx (backpropCtx (Γ := Γ) (ss := ss) g x seed) := by
  induction g with
  | nil =>
    simp [backpropVec, backpropCtx, flattenCtx_cast]
  | snoc g node ih =>
    rename_i ss τ
    simp only [backpropVec, backpropCtx]
    rw [← flattenCtx_cast, unsnocCtx_flattenCtx, evalVec_flattenCtx,
      Node.vjpVec_flattenCtx, ← flattenCtx_add, ih]

end Graph

-- ---------------------------------------------------------------------------
-- Embedding the analytic model as the `Δ := Unit` slice of the algebraic model
-- ---------------------------------------------------------------------------

/-- Embed an analytic node as an algebraic node over `ℝ` with a trivial environment. -/
def Node.toAlgebra {Γ : List Shape} {τ : Shape} (node : Node Γ τ) :
    Algebra.Node (α := ℝ) (Δ := Unit) (Γ := Γ) τ where
  forward x _ := node.forward x
  jvp x dx _ := node.jvp x dx
  vjp x _ δ := node.vjp x δ
  correct x dx _ δ := by
    simpa [dot_eq_tensorAlgebra_dot, dotList_eq_algebra_dotList] using node.correct x dx δ

/-- Embed an analytic graph as an algebraic graph over `ℝ` with a trivial environment. -/
def Graph.toAlgebra {Γ : List Shape} :
    {ss : List Shape} → Graph Γ ss → Algebra.Graph (α := ℝ) (Δ := Unit) (Γ := Γ) ss
  | _, .nil => .nil
  | _, .snoc g node => .snoc (Graph.toAlgebra g) node.toAlgebra

namespace Algebra

-- ---------------------------------------------------------------------------
-- Specializing the algebraic model at `ℝ` and a fixed environment
-- ---------------------------------------------------------------------------

/--
Specialize an algebraic node at carrier `ℝ` and a fixed environment `d : Δ` to an analytic
node. The adjointness law transports along `dot_eq_tensorAlgebra_dot`.
-/
def Node.toReal {Δ : Type} {Γ : List Shape} {τ : Shape}
    (node : Node (α := ℝ) (Δ := Δ) (Γ := Γ) τ) (d : Δ) : _root_.Proofs.Autograd.Node Γ τ where
  forward x := node.forward x d
  jvp x dx := node.jvp x dx d
  vjp x δ := node.vjp x d δ
  correct x dx δ := by
    simpa [dot_eq_tensorAlgebra_dot, dotList_eq_algebra_dotList] using node.correct x dx d δ

/-- Specialize an algebraic graph at carrier `ℝ` and a fixed environment to an analytic graph. -/
def Graph.toReal {Δ : Type} {Γ : List Shape} :
    {ss : List Shape} → Graph (α := ℝ) (Δ := Δ) (Γ := Γ) ss → Δ →
      _root_.Proofs.Autograd.Graph Γ ss
  | _, .nil, _ => .nil
  | _, .snoc g node, d => .snoc (Graph.toReal g d) (node.toReal d)

/-- The round trip through the algebraic model is the identity on analytic nodes. -/
@[simp] theorem Node.toAlgebra_toReal {Γ : List Shape} {τ : Shape}
    (node : _root_.Proofs.Autograd.Node Γ τ) (d : Unit) :
    Node.toReal (node.toAlgebra) d = node := rfl

/-- The round trip through the algebraic model is the identity on analytic graphs. -/
@[simp] theorem Graph.toAlgebra_toReal {Γ : List Shape} :
    ∀ {ss : List Shape} (g : _root_.Proofs.Autograd.Graph Γ ss) (d : Unit),
      Graph.toReal (g.toAlgebra) d = g
  | _, .nil, _ => rfl
  | _, .snoc g node, d => by
      simp [_root_.Proofs.Autograd.Graph.toAlgebra, Graph.toReal,
        Graph.toAlgebra_toReal g d]

namespace Graph

variable {Δ : Type}
variable {Γ : List Shape}

/-- Specialization preserves evaluation. -/
theorem toReal_eval {ss : List Shape} (g : Graph (α := ℝ) (Δ := Δ) (Γ := Γ) ss) (x : TList ℝ Γ)
    (d : Δ) :
    _root_.Proofs.Autograd.Graph.eval (toReal g d) x = eval (α := ℝ) g x d := by
  induction g with
  | nil =>
    rfl
  | snoc g node ih =>
    simp [toReal, _root_.Proofs.Autograd.Graph.eval, eval, ih, Node.toReal]

/-- Specialization preserves the JVP. -/
theorem toReal_jvpCtx {ss : List Shape} (g : Graph (α := ℝ) (Δ := Δ) (Γ := Γ) ss)
    (x dx : TList ℝ Γ) (d : Δ) :
    _root_.Proofs.Autograd.Graph.jvpCtx (toReal g d) x dx = jvpCtx (α := ℝ) g x dx d := by
  induction g with
  | nil =>
    rfl
  | snoc g node ih =>
    simp [toReal, _root_.Proofs.Autograd.Graph.jvpCtx, jvpCtx, toReal_eval, ih, Node.toReal]

/-- Specialization preserves the reverse pass. -/
theorem toReal_backpropCtx {ss : List Shape} (g : Graph (α := ℝ) (Δ := Δ) (Γ := Γ) ss)
    (x : TList ℝ Γ) (d : Δ) (seed : TList ℝ (Γ ++ ss)) :
    _root_.Proofs.Autograd.Graph.backpropCtx (toReal g d) x seed
      = backpropCtx (α := ℝ) g x d seed := by
  induction g with
  | nil =>
    rfl
  | snoc g node ih =>
    simp [toReal, _root_.Proofs.Autograd.Graph.backpropCtx, backpropCtx, toReal_eval, ih,
      Node.toReal]

end Graph

-- ---------------------------------------------------------------------------
-- Input-prefix extraction from a full context
-- ---------------------------------------------------------------------------

namespace TList

/-- Read the input (`Γ`-prefix) block out of a full context over `Γ ++ ss`. -/
def takeLeft {α : Type} : {Γ ss : List Shape} → TList α (Γ ++ ss) → TList α Γ
  | [], _, _ => .nil
  | _ :: Γ, ss, .cons x xs => .cons x (takeLeft (Γ := Γ) (ss := ss) xs)

/-- Push a `cast` along a `cons` cell. -/
theorem cast_cons {α : Type} {s : Shape} {ss₁ ss₂ : List Shape} (h : s :: ss₁ = s :: ss₂)
    (h' : ss₁ = ss₂) (x : Tensor α s) (xs : TList α ss₁) :
    cast (α := α) h (cons x xs) = cons x (cast (α := α) h' xs) := by
  cases h'
  rfl

/-- On a context with no intermediates, `takeLeft` is the cast along `Γ ++ [] = Γ`. -/
theorem takeLeft_append_nil {α : Type} {Γ : List Shape} (w : TList α (Γ ++ [])) :
    takeLeft (Γ := Γ) (ss := []) w = cast (α := α) (List.append_nil Γ) w := by
  induction Γ with
  | nil =>
    cases w
    rfl
  | cons s Γ ih =>
    cases w with
    | cons x w' =>
      show cons x (takeLeft (Γ := Γ) (ss := []) w')
          = cast (α := α) (ss₁ := s :: (Γ ++ [])) (ss₂ := s :: Γ)
              (show s :: (Γ ++ []) = s :: Γ from List.append_nil (s :: Γ)) (cons x w')
      rw [cast_cons (α := α) (s := s) (ss₁ := Γ ++ []) (ss₂ := Γ)
        (show s :: (Γ ++ []) = s :: Γ from List.append_nil (s :: Γ)) (List.append_nil Γ) x w', ih]

/-- `takeLeft` ignores a snoc-ed final block (after reassociating the context). -/
theorem takeLeft_cast_snoc {α : Type} {Γ : List Shape} :
    ∀ {ss : List Shape} {τ : Shape} (h : (Γ ++ ss) ++ [τ] = Γ ++ (ss ++ [τ]))
      (w : TList α (Γ ++ ss)) (y : Tensor α τ),
      takeLeft (Γ := Γ) (ss := ss ++ [τ]) (cast (α := α) h (snoc w y))
        = takeLeft (Γ := Γ) (ss := ss) w := by
  induction Γ with
  | nil =>
    intro ss τ h w y
    rfl
  | cons s Γ ih =>
    intro ss τ h w y
    cases w with
    | cons x w' =>
      show takeLeft (Γ := s :: Γ) (ss := ss ++ [τ])
            (cast (α := α) (ss₁ := s :: ((Γ ++ ss) ++ [τ])) (ss₂ := s :: (Γ ++ (ss ++ [τ]))) h
              (cons x (snoc w' y)))
          = cons x (takeLeft (Γ := Γ) (ss := ss) w')
      rw [cast_cons (α := α) (s := s) (ss₁ := (Γ ++ ss) ++ [τ]) (ss₂ := Γ ++ (ss ++ [τ])) h
        (List.append_assoc Γ ss [τ]) x (snoc w' y)]
      show cons x (takeLeft (Γ := Γ) (ss := ss ++ [τ])
            (cast (α := α) (List.append_assoc Γ ss [τ]) (snoc w' y)))
          = cons x (takeLeft (Γ := Γ) (ss := ss) w')
      rw [ih]

end TList

namespace Graph

variable {α : Type} [CommSemiring α]
variable {Δ : Type}
variable {Γ : List Shape}

/--
The input (`Γ`-prefix) block of the full backpropagation is the inputs-only backpropagation.

This identifies the link-facing `backpropAllCtx` (which retains a cotangent for every
value, mirroring the tape engine's dense reverse pass) with the proof-facing `backpropCtx`
on the input block.
-/
theorem takeLeft_backpropAllCtx {ss : List Shape} (g : Graph (α := α) (Δ := Δ) (Γ := Γ) ss)
    (x : TList α Γ) (d : Δ) (seed : TList α (Γ ++ ss)) :
    TList.takeLeft (backpropAllCtx (α := α) g x d seed) = backpropCtx (α := α) g x d seed := by
  induction g with
  | nil =>
    exact TList.takeLeft_append_nil (α := α) seed
  | snoc g node ih =>
    simp only [backpropAllCtx, backpropCtx]
    rw [TList.takeLeft_cast_snoc]
    exact ih _
end Graph

namespace GraphData

variable {α : Type} [Add α]
variable {Δ : Type}
variable {Γ : List Shape}

/-- `GraphData` version of `Graph.takeLeft_backpropAllCtx`. -/
theorem takeLeft_backpropAllCtx {ss : List Shape} (g : GraphData α Δ Γ ss) (x : TList α Γ)
    (d : Δ) (seed : TList α (Γ ++ ss)) :
    TList.takeLeft (backpropAllCtx (α := α) g x d seed) = backpropCtx (α := α) g x d seed := by
  induction g with
  | nil =>
    exact TList.takeLeft_append_nil (α := α) seed
  | snoc g node ih =>
    simp only [backpropAllCtx, backpropCtx]
    rw [TList.takeLeft_cast_snoc]
    exact ih _

end GraphData

-- ---------------------------------------------------------------------------
-- Composed endpoints: the algebraic reverse pass at `ℝ` is `(fderiv eval)†`
-- ---------------------------------------------------------------------------

namespace Graph

variable {Δ : Type}
variable {Γ : List Shape}

open Runtime
open Runtime.Autograd

/--
**Analytic upgrade of the algebraic reverse pass.** At carrier `ℝ` and a fixed environment,
the inputs-only backpropagation computes the adjoint of the Fréchet derivative of the graph's
(vectorized) forward evaluation.
-/
theorem backpropCtx_eq_adjoint_fderiv {ss : List Shape}
    (g : Graph (α := ℝ) (Δ := Δ) (Γ := Γ) ss) (d : Δ)
    (hg : GraphFDerivCorrect (Γ := Γ) (toReal g d)) (x : TList ℝ Γ)
    (seed : TList ℝ (Γ ++ ss)) :
    flattenCtx (backpropCtx (α := ℝ) g x d seed)
      = (fderiv ℝ (_root_.Proofs.Autograd.Graph.evalVec (toReal g d))
          (flattenCtx x)).adjoint (flattenCtx seed) := by
  rw [← toReal_backpropCtx, ← _root_.Proofs.Autograd.Graph.backpropVec_flattenCtx]
  exact _root_.Proofs.Autograd.Graph.backpropVec_eq_adjoint_fderiv (toReal g d) hg _ _

/--
Pointwise variant of `backpropCtx_eq_adjoint_fderiv`: differentiability is assumed only at the
values actually encountered, admitting non-smooth primitives away from their kinks.
-/
theorem backpropCtx_eq_adjoint_fderiv_at {ss : List Shape}
    (g : Graph (α := ℝ) (Δ := Δ) (Γ := Γ) ss) (d : Δ) (x : TList ℝ Γ)
    (hg : GraphFDerivCorrectAt (Γ := Γ) (toReal g d) (flattenCtx x))
    (seed : TList ℝ (Γ ++ ss)) :
    flattenCtx (backpropCtx (α := ℝ) g x d seed)
      = (fderiv ℝ (_root_.Proofs.Autograd.Graph.evalVec (toReal g d))
          (flattenCtx x)).adjoint (flattenCtx seed) := by
  rw [← toReal_backpropCtx, ← _root_.Proofs.Autograd.Graph.backpropVec_flattenCtx]
  exact _root_.Proofs.Autograd.Graph.backpropVec_eq_adjoint_fderiv_at (toReal g d) _ _ hg

/-- The function differentiated in the endpoints is the graph's own forward evaluation. -/
theorem toReal_evalVec {ss : List Shape} (g : Graph (α := ℝ) (Δ := Δ) (Γ := Γ) ss) (d : Δ)
    (x : TList ℝ Γ) :
    _root_.Proofs.Autograd.Graph.evalVec (toReal g d) (flattenCtx x)
      = flattenCtx (eval (α := ℝ) g x d) := by
  rw [_root_.Proofs.Autograd.Graph.evalVec_flattenCtx, toReal_eval]

/--
**Tape-model reverse pass = adjoint of the Fréchet derivative.** Running the tape engine's
dense reverse pass (`Tape.backwardDenseFrom`) on a compiled graph — the exact tape model
instantiated at `α := ℝ`, not the native `Float` or CUDA execution path — succeeds and returns
the full backpropagation context, whose input (`Γ`-prefix) block is exactly the adjoint of the
Fréchet derivative of the graph's forward evaluation applied to the seed.

This composes the runtime link (`backwardDenseFrom_compileAux_eq_backpropAllCtx`) with the
analytic upgrade above.
-/
theorem backwardDenseFrom_compileAux_adjoint_fderiv [DecidableEq Shape] {ss : List Shape}
    (g : Graph (α := ℝ) (Δ := Δ) (Γ := Γ) ss) (x : TList ℝ Γ) (d0 : Δ)
    (seed : TList ℝ (Γ ++ ss)) (hg : GraphFDerivCorrect (Γ := Γ) (toReal g d0)) :
    Runtime.Autograd.Tape.backwardDenseFrom
        (t := (compileAux (α := ℝ) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0).1)
        (grads0 := TList.toAnyArray (α := ℝ) (ss := Γ ++ ss) seed)
      = .ok (TList.toAnyArray (α := ℝ) (ss := Γ ++ ss)
          (backpropAllCtx (α := ℝ) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0 seed))
    ∧ flattenCtx (TList.takeLeft (backpropAllCtx (α := ℝ) g x d0 seed))
      = (fderiv ℝ (_root_.Proofs.Autograd.Graph.evalVec (toReal g d0))
          (flattenCtx x)).adjoint (flattenCtx seed) := by
  refine ⟨backwardDenseFrom_compileAux_eq_backpropAllCtx (α := ℝ) g x d0 seed, ?_⟩
  rw [takeLeft_backpropAllCtx]
  exact backpropCtx_eq_adjoint_fderiv g d0 hg x seed

/-- Pointwise variant of `backwardDenseFrom_compileAux_adjoint_fderiv`. -/
theorem backwardDenseFrom_compileAux_adjoint_fderiv_at [DecidableEq Shape] {ss : List Shape}
    (g : Graph (α := ℝ) (Δ := Δ) (Γ := Γ) ss) (x : TList ℝ Γ) (d0 : Δ)
    (seed : TList ℝ (Γ ++ ss))
    (hg : GraphFDerivCorrectAt (Γ := Γ) (toReal g d0) (flattenCtx x)) :
    Runtime.Autograd.Tape.backwardDenseFrom
        (t := (compileAux (α := ℝ) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0).1)
        (grads0 := TList.toAnyArray (α := ℝ) (ss := Γ ++ ss) seed)
      = .ok (TList.toAnyArray (α := ℝ) (ss := Γ ++ ss)
          (backpropAllCtx (α := ℝ) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0 seed))
    ∧ flattenCtx (TList.takeLeft (backpropAllCtx (α := ℝ) g x d0 seed))
      = (fderiv ℝ (_root_.Proofs.Autograd.Graph.evalVec (toReal g d0))
          (flattenCtx x)).adjoint (flattenCtx seed) := by
  refine ⟨backwardDenseFrom_compileAux_eq_backpropAllCtx (α := ℝ) g x d0 seed, ?_⟩
  rw [takeLeft_backpropAllCtx]
  exact backpropCtx_eq_adjoint_fderiv_at g d0 x hg seed

end Graph

end Algebra

end
end Autograd
end Proofs
