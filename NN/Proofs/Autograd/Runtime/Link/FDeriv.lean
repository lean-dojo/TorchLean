/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Runtime.Link.Core
public import NN.Proofs.Autograd.Tape.Core.FDeriv
public import NN.Proofs.Autograd.Runtime.Link.BackwardGraph

/-!
# Analytic upgrade of the runtime link: reverse mode computes `(fderiv eval)†`

Two independent developments meet in this file.

* The **runtime link** (`NN.Proofs.Autograd.Runtime.Link.BackwardGraph`) proves that the tape
  engine's dense reverse pass (`Tape.backwardDenseFrom`) agrees with the algebraic model's full
  backpropagation `backpropAllCtx`, over any commutative semiring carrier and an arbitrary
  non-differentiable environment `Δ`.
* The **analytic tape model** (`NN.Proofs.Autograd.Tape.Core.FDeriv`) proves that reverse-mode
  accumulation computes the adjoint of the Fréchet derivative of the forward evaluation, over `ℝ`.

The two developments are stated on *different graph types*: the algebraic
`Algebra.Graph (α := α) Δ Γ ss` versus the `ℝ`-monomorphic, environment-free
`Proofs.Autograd.Graph Γ ss`. Their contexts already coincide (`TorchLean.TensorPack ℝ Γ` is by
definition `Algebra.TorchLean.TensorPack ℝ Γ`), and their `eval`/`jvpCtx`/`backpropCtx` recursions
mirror each other node for node; what has been missing is the formal connection. This file
supplies it:

* **The real, environment-free slice.** `Algebra.Node.toReal` / `Algebra.Graph.toReal` specialize an
  algebraic graph at `α := ℝ` and a fixed environment `d : Δ` to an analytic graph;
  `Node.toAlgebra` / `Graph.toAlgebra` embed an analytic graph back as the `Δ := Unit` slice,
  and both round trips are identities on that slice (`toAlgebra_toReal` and
  `toReal_toAlgebra`). The analytic model is therefore exactly the environment-free `ℝ` slice of
  the algebraic model. The commutation lemmas
  `toReal_eval`, `toReal_jvpCtx`, `toReal_backpropCtx` show the specialization preserves all
  three semantics.
* **Input-prefix extraction.** `TensorPack.takeLeft` reads the input (`Γ`-prefix) block out of a
  full context, and `takeLeft_backpropAllCtx` (also in `GraphData` form) identifies the input
  block of the full backpropagation with the inputs-only `backpropCtx`, the missing lemma
  relating the runtime-facing `backpropAllCtx` to the proof-facing `backpropCtx`.
* **Vectorization transport.** The `flattenCtx_*` lemmas commute context vectorization with
  `cast`/`snoc`/`unsnoc`/`add`, so the `TensorPack`-level graph semantics coincide with the
  Euclidean `CtxVec` semantics: `evalVec_flattenCtx`, `jvpVec_flattenCtx`,
  `backpropVec_flattenCtx`.
* **The composed endpoints.** `backpropCtx_eq_adjoint_fderiv` upgrades the algebraic
  backpropagation at `ℝ` to the Fréchet-adjoint characterization, and
  `backwardDenseFrom_lowerGraphToTape_adjoint_fderiv` combines it with the runtime link: the tape
  model's dense reverse pass on a lowered graph, instantiated at `α := ℝ`, succeeds with the
  full backpropagation context, whose input prefix is exactly `(fderiv ℝ eval x)† seed`. The
  `_at` variants assume
  differentiability only at the actual execution point (`GraphFDerivCorrectAt`), covering
  graphs with non-smooth primitives (`relu`, `abs`, `min`/`max`, `log`, `sqrt`, …).

Throughout, "reverse pass" means the exact tape model: the `Tape.backwardDenseFrom` program of
`Runtime/Autograd` instantiated at the exact carrier `α := ℝ`. Nothing in this file is a
statement about the native `Float` evaluation or the CUDA execution path; relating those to the
exact model is a separate (approximation) concern.
-/

@[expose] public section

namespace Proofs
namespace Autograd

open Spec TorchLean
open TorchLean TorchLean.Tensor

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
theorem dotList_eq_algebra_dotList {Γ : List Shape} (x y : TorchLean.TensorPack ℝ Γ) :
    TensorPack.dotList (ss := Γ) x y = Algebra.TensorPack.dotList (α := ℝ) x y := by
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
        simp [TensorPack.dotList, Algebra.TensorPack.dotList, dot_eq_tensorAlgebra_dot, ih]

-- ---------------------------------------------------------------------------
-- Vectorization transport: `flattenCtx` commutes with the context operations
-- ---------------------------------------------------------------------------

/-- `flattenCtx` commutes with casting a context along a shape-list equality. -/
theorem flattenCtx_cast {Γ₁ Γ₂ : List Shape} (h : Γ₁ = Γ₂) (xs : TorchLean.TensorPack ℝ Γ₁) :
    flattenCtx (Γ := Γ₂) (TorchLean.TensorPack.cast h xs) =
      castCtxVec (Γ₁ := Γ₁) (Γ₂ := Γ₂) h (flattenCtx xs) := by
  cases h
  simp

/-- Vectorization is additive: `tensorToVec` maps `addSpec` to vector addition. -/
theorem tensorToVec_addSpec {s : Shape} (a b : Tensor ℝ s) :
    tensorToVec (t := addSpec a b) = tensorToVec (t := a) + tensorToVec (t := b) := by
  refine ext_inner_right ℝ ?_
  intro w
  have hw : w = tensorToVec (t := vecToTensor (s := s) w) :=
    (tensorToVec_vecToTensor (s := s) w).symm
  rw [hw, ← dot_eq_inner_tensorToVec, inner_add_left, ← dot_eq_inner_tensorToVec,
    ← dot_eq_inner_tensorToVec, dot_add_left]

/-- The context inner product is additive in its left argument. -/
private theorem dotList_add_left' {Γ : List Shape} (u v z : TorchLean.TensorPack ℝ Γ) :
    TensorPack.dotList (TorchLean.TensorPack.add u v) z =
      TensorPack.dotList u z + TensorPack.dotList v z := by
  induction Γ with
  | nil =>
    cases u
    cases v
    cases z
    simp [TensorPack.dotList, TorchLean.TensorPack.add]
  | cons s Γ ih =>
    cases u with
    | cons uh ut =>
      cases v with
      | cons vh vt =>
        cases z with
        | cons zh zt =>
          show TensorPack.dotList
              (TorchLean.TensorPack.cons (addSpec uh vh) (TorchLean.TensorPack.add ut vt))
              (TorchLean.TensorPack.cons zh zt)
              = _
          simp only [TensorPack.dotList, dot_add_left, ih]
          ring

/-- `flattenCtx` maps context addition to vector addition. -/
theorem flattenCtx_add {Γ : List Shape} (u v : TorchLean.TensorPack ℝ Γ) :
    flattenCtx (TorchLean.TensorPack.add u v) = flattenCtx u + flattenCtx v := by
  refine ext_inner_right ℝ ?_
  intro w
  have hw : w = flattenCtx (unflattenCtx (Γ := Γ) w) :=
    (flattenCtx_unflattenCtx (Γ := Γ) w).symm
  rw [hw, ← dotList_eq_inner_flattenCtx, inner_add_left, ← dotList_eq_inner_flattenCtx,
    ← dotList_eq_inner_flattenCtx]
  exact dotList_add_left' u v _

/-- `flattenCtx` maps `TorchLean.TensorPack.snoc` to `snocCtx`. -/
theorem flattenCtx_snoc {Γ : List Shape} {τ : Shape} (xs : TorchLean.TensorPack ℝ Γ)
    (y : Tensor ℝ τ) :
    flattenCtx (Γ := Γ ++ [τ]) (TorchLean.TensorPack.snoc xs y)
      = snocCtx (Γ := Γ) (τ := τ) (flattenCtx xs) (tensorToVec (t := y)) := by
  induction Γ with
  | nil =>
    cases xs
    change flattenCtx (TorchLean.TensorPack.cons y TorchLean.TensorPack.nil) = _
    rw [flattenCtx_cons, flattenCtx_nil]
    let h : ctxSize [] + τ.size = τ.size + ctxSize [] := by simp [ctxSize]
    change appendVec (tensorToVec y) 0 = castVec h (appendVec 0 (tensorToVec y))
    apply PiLp.ext
    intro i
    induction i using Fin.addCases with
    | left i =>
      rw [appendVec_ofLp_castAdd, castVec_ofLp]
      have hidx :
          Fin.cast h.symm (Fin.castAdd (ctxSize []) i) = Fin.natAdd (ctxSize []) i := by
        apply Fin.ext
        simpa only [ctxSize, Fin.val_natAdd, Nat.zero_add, Fin.val_castAdd] using
          Fin.val_cast h.symm (Fin.castAdd (ctxSize []) i)
      rw [hidx, appendVec_ofLp_natAdd]
    | right i => exact i.elim0
  | cons s Γ ih =>
    cases xs with
    | cons x xs =>
      change flattenCtx (TorchLean.TensorPack.cons x (TorchLean.TensorPack.snoc xs y)) = _
      rw [flattenCtx_cons, ih, flattenCtx_cons, appendVec_snocCtx]

/-- `flattenCtx` maps `TorchLean.TensorPack.unsnoc` to `unsnocCtx`. -/
theorem unsnocCtx_flattenCtx {Γ : List Shape} {τ : Shape} (w : TorchLean.TensorPack ℝ (Γ ++ [τ])) :
    unsnocCtx (Γ := Γ) (τ := τ) (flattenCtx w)
      = (flattenCtx (TorchLean.TensorPack.unsnoc w).1,
          tensorToVec (t := (TorchLean.TensorPack.unsnoc w).2)) := by
  conv_lhs => rw [← TorchLean.TensorPack.snoc_unsnoc (α := ℝ) (ss := Γ) (τ := τ) (xs := w)]
  rw [flattenCtx_snoc, unsnocCtx_snocCtx]

namespace Node

/-- The vectorized forward map, evaluated on a flattened context. -/
theorem forwardVec_flattenCtx {Γ : List Shape} {τ : Shape} (node : Node Γ τ)
    (x : TorchLean.TensorPack ℝ Γ) :
    node.forwardVec (Γ := Γ) (τ := τ) (flattenCtx x) = tensorToVec (t := node.forward x) := by
  simp [Node.forwardVec]

/-- The vectorized JVP, evaluated on flattened contexts. -/
theorem jvpVec_flattenCtx {Γ : List Shape} {τ : Shape} (node : Node Γ τ)
    (x dx : TorchLean.TensorPack ℝ Γ) :
    node.jvpVec (Γ := Γ) (τ := τ) (flattenCtx x) (flattenCtx dx)
      = tensorToVec (t := node.jvp x dx) := by
  simp [Node.jvpVec]

/-- The vectorized VJP, evaluated on a flattened context and a vectorized cotangent. -/
theorem vjpVec_flattenCtx {Γ : List Shape} {τ : Shape} (node : Node Γ τ)
    (x : TorchLean.TensorPack ℝ Γ) (δ : Tensor ℝ τ) :
    node.vjpVec (Γ := Γ) (τ := τ) (flattenCtx x) (tensorToVec (t := δ))
      = flattenCtx (node.vjp x δ) := by
  simp [Node.vjpVec]

end Node

namespace Graph

variable {Γ : List Shape}

/-- The Euclidean graph evaluation is the flattening of the `TensorPack` evaluation. -/
theorem evalVec_flattenCtx {ss : List Shape} (g : Graph Γ ss) (x : TorchLean.TensorPack ℝ Γ) :
    evalVec (Γ := Γ) (ss := ss) g (flattenCtx x) = flattenCtx (eval (Γ := Γ) (ss := ss) g x) := by
  induction g with
  | nil =>
    simp [evalVec, eval, flattenCtx_cast]
  | snoc g node ih =>
    simp [evalVec, eval, flattenCtx_cast, flattenCtx_snoc, ih, Node.forwardVec_flattenCtx]

/-- The Euclidean graph JVP is the flattening of the `TensorPack` JVP. -/
theorem jvpVec_flattenCtx {ss : List Shape} (g : Graph Γ ss) (x dx : TorchLean.TensorPack ℝ Γ) :
    jvpVec (Γ := Γ) (ss := ss) g (flattenCtx x) (flattenCtx dx)
      = flattenCtx (jvpCtx (Γ := Γ) (ss := ss) g x dx) := by
  induction g with
  | nil =>
    simp [jvpVec, jvpCtx, flattenCtx_cast]
  | snoc g node ih =>
    simp [jvpVec, jvpCtx, flattenCtx_cast, flattenCtx_snoc, ih, evalVec_flattenCtx,
      Node.jvpVec_flattenCtx]

/-- The Euclidean reverse pass is the flattening of the `TensorPack` reverse pass. -/
theorem backpropVec_flattenCtx {ss : List Shape} (g : Graph Γ ss) (x : TorchLean.TensorPack ℝ Γ)
    (seed : TorchLean.TensorPack ℝ (Γ ++ ss)) :
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
  validate x _ := node.validate x
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
    (node : Node (α := ℝ) (Δ := Δ) (Γ := Γ) τ) (d : Δ) : Proofs.Autograd.Node Γ τ where
  validate x := node.validate x d
  forward x := node.forward x d
  jvp x dx := node.jvp x dx d
  vjp x δ := node.vjp x d δ
  correct x dx δ := by
    simpa [dot_eq_tensorAlgebra_dot, dotList_eq_algebra_dotList] using node.correct x dx d δ

/-- Specialize an algebraic graph at carrier `ℝ` and a fixed environment to an analytic graph. -/
def Graph.toReal {Δ : Type} {Γ : List Shape} :
    {ss : List Shape} → Graph (α := ℝ) (Δ := Δ) (Γ := Γ) ss → Δ →
      Proofs.Autograd.Graph Γ ss
  | _, .nil, _ => .nil
  | _, .snoc g node, d => .snoc (Graph.toReal g d) (node.toReal d)

/-- The round trip through the algebraic model is the identity on analytic nodes. -/
@[simp] theorem Node.toAlgebra_toReal {Γ : List Shape} {τ : Shape}
    (node : Proofs.Autograd.Node Γ τ) (d : Unit) :
    Node.toReal (node.toAlgebra) d = node := rfl

/-- Specializing an algebraic node with a trivial environment and embedding it back is the
identity. -/
@[simp] theorem Node.toReal_toAlgebra {Γ : List Shape} {τ : Shape}
    (node : Node (α := ℝ) (Δ := Unit) (Γ := Γ) τ) :
    Proofs.Autograd.Node.toAlgebra (Node.toReal node ()) = node := by
  cases node
  rfl

/-- The round trip through the algebraic model is the identity on analytic graphs. -/
@[simp] theorem Graph.toAlgebra_toReal {Γ : List Shape} :
    ∀ {ss : List Shape} (g : Proofs.Autograd.Graph Γ ss) (d : Unit),
      Graph.toReal (g.toAlgebra) d = g
  | _, .nil, _ => rfl
  | _, .snoc g node, d => by
      simp [Proofs.Autograd.Graph.toAlgebra, Graph.toReal,
        Graph.toAlgebra_toReal g d]

/-- Specializing an algebraic graph with a trivial environment and embedding it back is the
identity. -/
@[simp] theorem Graph.toReal_toAlgebra {Γ : List Shape} :
    ∀ {ss : List Shape} (g : Graph (α := ℝ) (Δ := Unit) (Γ := Γ) ss),
      Proofs.Autograd.Graph.toAlgebra (Graph.toReal g ()) = g
  | _, .nil => rfl
  | _, .snoc g node => by
      simp [Proofs.Autograd.Graph.toAlgebra, Graph.toReal,
        Graph.toReal_toAlgebra g]

namespace Graph

variable {Δ : Type}
variable {Γ : List Shape}

/-- Specialization preserves evaluation. -/
theorem toReal_eval {ss : List Shape} (g : Graph (α := ℝ) (Δ := Δ) (Γ := Γ) ss)
    (x : TorchLean.TensorPack ℝ Γ) (d : Δ) :
    Proofs.Autograd.Graph.eval (toReal g d) x = eval (α := ℝ) g x d := by
  induction g with
  | nil =>
    rfl
  | snoc g node ih =>
    simp [toReal, Proofs.Autograd.Graph.eval, eval, toData, GraphData.eval, ih, Node.toReal]

/-- Specialization preserves the JVP. -/
theorem toReal_jvpCtx {ss : List Shape} (g : Graph (α := ℝ) (Δ := Δ) (Γ := Γ) ss)
    (x dx : TorchLean.TensorPack ℝ Γ) (d : Δ) :
    Proofs.Autograd.Graph.jvpCtx (toReal g d) x dx = jvpCtx (α := ℝ) g x dx d := by
  induction g with
  | nil =>
    rfl
  | snoc g node ih =>
    simp [toReal, Proofs.Autograd.Graph.jvpCtx, jvpCtx, toData, GraphData.jvpCtx,
      toReal_eval, eval, ih, Node.toReal]

/-- Specialization preserves the reverse pass. -/
theorem toReal_backpropCtx {ss : List Shape} (g : Graph (α := ℝ) (Δ := Δ) (Γ := Γ) ss)
    (x : TorchLean.TensorPack ℝ Γ) (d : Δ) (seed : TorchLean.TensorPack ℝ (Γ ++ ss)) :
    Proofs.Autograd.Graph.backpropCtx (toReal g d) x seed
      = backpropCtx (α := ℝ) g x d seed := by
  induction g with
  | nil =>
    rfl
  | snoc g node ih =>
    simp [toReal, Proofs.Autograd.Graph.backpropCtx, backpropCtx, toData, GraphData.backpropCtx,
      toReal_eval, eval, ih, Node.toReal]

end Graph

-- ---------------------------------------------------------------------------
-- Input-prefix extraction from a full context
-- ---------------------------------------------------------------------------

namespace TensorPack

/-- Read the input (`Γ`-prefix) block out of a full context over `Γ ++ ss`. -/
def takeLeft {α : Type} [TorchLean.Storage α] :
    {Γ ss : List Shape} → TorchLean.TensorPack α (Γ ++ ss) → TorchLean.TensorPack α Γ
  | [], _, _ => .nil
  | _ :: Γ, ss, .cons x xs => .cons x (takeLeft (Γ := Γ) (ss := ss) xs)

/-- Push a `cast` along a `cons` cell. -/
theorem cast_cons {α : Type} [TorchLean.Storage α] {s : Shape} {ss₁ ss₂ : List Shape}
    (h : s :: ss₁ = s :: ss₂) (h' : ss₁ = ss₂) (x : Tensor α s) (xs : TorchLean.TensorPack α ss₁) :
    TorchLean.TensorPack.cast (α := α) h (.cons x xs) =
      .cons x (TorchLean.TensorPack.cast (α := α) h' xs) := by
  cases h'
  rfl

/-- On a context with no intermediates, `takeLeft` is the cast along `Γ ++ [] = Γ`. -/
theorem takeLeft_append_nil {α : Type} [TorchLean.Storage α] {Γ : List Shape}
    (w : TorchLean.TensorPack α (Γ ++ [])) :
    takeLeft (Γ := Γ) (ss := []) w =
      TorchLean.TensorPack.cast (α := α) (List.append_nil Γ) w := by
  induction Γ with
  | nil =>
    cases w
    rfl
  | cons s Γ ih =>
    cases w with
    | cons x w' =>
      show .cons x (takeLeft (Γ := Γ) (ss := []) w') =
          TorchLean.TensorPack.cast (α := α)
            (show s :: (Γ ++ []) = s :: Γ from List.append_nil (s :: Γ)) (.cons x w')
      rw [cast_cons (α := α) (s := s) (ss₁ := Γ ++ []) (ss₂ := Γ)
        (show s :: (Γ ++ []) = s :: Γ from List.append_nil (s :: Γ)) (List.append_nil Γ) x w', ih]

/-- `takeLeft` ignores a snoc-ed final block (after reassociating the context). -/
theorem takeLeft_cast_snoc {α : Type} [TorchLean.Storage α] {Γ : List Shape} :
    ∀ {ss : List Shape} {τ : Shape} (h : (Γ ++ ss) ++ [τ] = Γ ++ (ss ++ [τ]))
      (w : TorchLean.TensorPack α (Γ ++ ss)) (y : Tensor α τ),
      takeLeft (Γ := Γ) (ss := ss ++ [τ])
        (TorchLean.TensorPack.cast (α := α) h
          (TorchLean.TensorPack.snoc w y))
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
            (TorchLean.TensorPack.cast (α := α) h
              (.cons x (TorchLean.TensorPack.snoc w' y))) =
          .cons x (takeLeft (Γ := Γ) (ss := ss) w')
      rw [cast_cons (α := α) (s := s) (ss₁ := (Γ ++ ss) ++ [τ]) (ss₂ := Γ ++ (ss ++ [τ])) h
        (List.append_assoc Γ ss [τ]) x (TorchLean.TensorPack.snoc w' y)]
      show TorchLean.TensorPack.cons x (takeLeft (Γ := Γ) (ss := ss ++ [τ])
            (TorchLean.TensorPack.cast (α := α) (List.append_assoc Γ ss [τ])
              (TorchLean.TensorPack.snoc w' y))) =
          TorchLean.TensorPack.cons x (takeLeft (Γ := Γ) (ss := ss) w')
      rw [ih]

end TensorPack

namespace Graph

variable {α : Type} [TorchLean.Storage α] [CommSemiring α]
variable {Δ : Type}
variable {Γ : List Shape}

/--
The input (`Γ`-prefix) block of the full backpropagation is the inputs-only backpropagation.

This identifies the link-facing `backpropAllCtx` (which retains a cotangent for every
value, mirroring the tape engine's dense reverse pass) with the proof-facing `backpropCtx`
on the input block.
-/
theorem takeLeft_backpropAllCtx {ss : List Shape} (g : Graph (α := α) (Δ := Δ) (Γ := Γ) ss)
    (x : TorchLean.TensorPack α Γ) (d : Δ) (seed : TorchLean.TensorPack α (Γ ++ ss)) :
    TensorPack.takeLeft (backpropAllCtx (α := α) g x d seed) = backpropCtx (α := α) g x d seed := by
  induction g with
  | nil =>
    exact TensorPack.takeLeft_append_nil (α := α) seed
  | snoc g node ih =>
    simp only [backpropAllCtx, backpropCtx]
    rw [TensorPack.takeLeft_cast_snoc]
    exact ih _
end Graph

namespace GraphData

variable {α : Type} [TorchLean.Storage α] [Add α]
variable {Δ : Type}
variable {Γ : List Shape}

/-- `GraphData` version of `Graph.takeLeft_backpropAllCtx`. -/
theorem takeLeft_backpropAllCtx {ss : List Shape} (g : GraphData α Δ Γ ss)
    (x : TorchLean.TensorPack α Γ) (d : Δ) (seed : TorchLean.TensorPack α (Γ ++ ss)) :
    TensorPack.takeLeft (backpropAllCtx (α := α) g x d seed) = backpropCtx (α := α) g x d seed := by
  induction g with
  | nil =>
    exact TensorPack.takeLeft_append_nil (α := α) seed
  | snoc g node ih =>
    simp only [backpropAllCtx, backpropCtx]
    rw [TensorPack.takeLeft_cast_snoc]
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
    (hg : GraphFDerivCorrect (Γ := Γ) (toReal g d)) (x : TorchLean.TensorPack ℝ Γ)
    (seed : TorchLean.TensorPack ℝ (Γ ++ ss)) :
    flattenCtx (backpropCtx (α := ℝ) g x d seed)
      = (fderiv ℝ (Proofs.Autograd.Graph.evalVec (toReal g d))
          (flattenCtx x)).adjoint (flattenCtx seed) := by
  rw [← toReal_backpropCtx, ← Proofs.Autograd.Graph.backpropVec_flattenCtx]
  exact Proofs.Autograd.Graph.backpropVec_eq_adjoint_fderiv (toReal g d) hg _ _

/--
Pointwise variant of `backpropCtx_eq_adjoint_fderiv`: differentiability is assumed only at the
values actually encountered, admitting non-smooth primitives away from their kinks.
-/
theorem backpropCtx_eq_adjoint_fderiv_at {ss : List Shape}
    (g : Graph (α := ℝ) (Δ := Δ) (Γ := Γ) ss) (d : Δ) (x : TorchLean.TensorPack ℝ Γ)
    (hg : GraphFDerivCorrectAt (Γ := Γ) (toReal g d) (flattenCtx x))
    (seed : TorchLean.TensorPack ℝ (Γ ++ ss)) :
    flattenCtx (backpropCtx (α := ℝ) g x d seed)
      = (fderiv ℝ (Proofs.Autograd.Graph.evalVec (toReal g d))
          (flattenCtx x)).adjoint (flattenCtx seed) := by
  rw [← toReal_backpropCtx, ← Proofs.Autograd.Graph.backpropVec_flattenCtx]
  exact Proofs.Autograd.Graph.backpropVec_eq_adjoint_fderiv_at (toReal g d) _ _ hg

/-- The function differentiated in the endpoints is the graph's own forward evaluation. -/
theorem toReal_evalVec {ss : List Shape} (g : Graph (α := ℝ) (Δ := Δ) (Γ := Γ) ss) (d : Δ)
    (x : TorchLean.TensorPack ℝ Γ) :
    Proofs.Autograd.Graph.evalVec (toReal g d) (flattenCtx x)
      = flattenCtx (eval (α := ℝ) g x d) := by
  rw [Proofs.Autograd.Graph.evalVec_flattenCtx, toReal_eval]

/--
**Tape-model reverse pass = adjoint of the Fréchet derivative.** Running the tape engine's
dense reverse pass (`Tape.backwardDenseFrom`) on a lowered graph (the exact tape model
instantiated at `α := ℝ`, not the native `Float` or CUDA execution path) succeeds and returns
the full backpropagation context, whose input (`Γ`-prefix) block is exactly the adjoint of the
Fréchet derivative of the graph's forward evaluation applied to the seed.

This composes the runtime link (`backwardDenseFrom_lowerGraphToTape_eq_backpropAllCtx`) with the
analytic upgrade above.
-/
theorem backwardDenseFrom_lowerGraphToTape_adjoint_fderiv {ss : List Shape}
    (g : Graph (α := ℝ) (Δ := Δ) (Γ := Γ) ss) (x : TorchLean.TensorPack ℝ Γ) (d0 : Δ)
    (seed : TorchLean.TensorPack ℝ (Γ ++ ss)) (hg : GraphFDerivCorrect (Γ := Γ) (toReal g d0)) :
    Runtime.Autograd.Tape.backwardDenseFrom
        (t := (lowerGraphToTape (α := ℝ) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0).1)
        (grads0 := TorchLean.TensorPack.toShapeErasedArray (α := ℝ) (ss := Γ ++ ss) seed)
      = .ok (TorchLean.TensorPack.toShapeErasedArray (α := ℝ) (ss := Γ ++ ss)
          (backpropAllCtx (α := ℝ) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0 seed))
    ∧ flattenCtx (TensorPack.takeLeft (backpropAllCtx (α := ℝ) g x d0 seed))
      = (fderiv ℝ (Proofs.Autograd.Graph.evalVec (toReal g d0))
          (flattenCtx x)).adjoint (flattenCtx seed) := by
  refine ⟨backwardDenseFrom_lowerGraphToTape_eq_backpropAllCtx (α := ℝ) g x d0 seed, ?_⟩
  rw [takeLeft_backpropAllCtx]
  exact backpropCtx_eq_adjoint_fderiv g d0 hg x seed

/-- Pointwise variant of `backwardDenseFrom_lowerGraphToTape_adjoint_fderiv`. -/
theorem backwardDenseFrom_lowerGraphToTape_adjoint_fderiv_at {ss : List Shape}
    (g : Graph (α := ℝ) (Δ := Δ) (Γ := Γ) ss) (x : TorchLean.TensorPack ℝ Γ) (d0 : Δ)
    (seed : TorchLean.TensorPack ℝ (Γ ++ ss))
    (hg : GraphFDerivCorrectAt (Γ := Γ) (toReal g d0) (flattenCtx x)) :
    Runtime.Autograd.Tape.backwardDenseFrom
        (t := (lowerGraphToTape (α := ℝ) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0).1)
        (grads0 := TorchLean.TensorPack.toShapeErasedArray (α := ℝ) (ss := Γ ++ ss) seed)
      = .ok (TorchLean.TensorPack.toShapeErasedArray (α := ℝ) (ss := Γ ++ ss)
          (backpropAllCtx (α := ℝ) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0 seed))
    ∧ flattenCtx (TensorPack.takeLeft (backpropAllCtx (α := ℝ) g x d0 seed))
      = (fderiv ℝ (Proofs.Autograd.Graph.evalVec (toReal g d0))
          (flattenCtx x)).adjoint (flattenCtx seed) := by
  refine ⟨backwardDenseFrom_lowerGraphToTape_eq_backpropAllCtx (α := ℝ) g x d0 seed, ?_⟩
  rw [takeLeft_backpropAllCtx]
  exact backpropCtx_eq_adjoint_fderiv_at g d0 x hg seed

end Graph

end Algebra

end
end Autograd
end Proofs
