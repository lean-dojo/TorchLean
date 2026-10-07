/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Algebra
public import NN.Proofs.Autograd.Tape.Algebra.Contributions

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
          change dot xh (addSpec yh zh) + dotList xt (TorchLean.TensorPack.add yt zt) = _
          rw [TensorAlgebra.dot_add_right, ih]
          simp only [dotList]
          ac_rfl

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
      change dot xh (Tensor.full s 0) + dotList xt TorchLean.TensorPack.zero = 0
      rw [TensorAlgebra.dot_full_zero_right, ih, zero_add]

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
  obtain ⟨i, rfl⟩ := idx
  induction Γ with
  | nil => exact Fin.elim0 i
  | cons head shapes ih =>
    cases dx with
    | cons first rest =>
      obtain ⟨i, hi⟩ := i
      cases i with
      | zero =>
        simp [single, dotList, getIdx, Tensor.castShape, dotList_zero_right]
        rfl
      | succ i =>
        simpa [single, dotList, getIdx, Tensor.castShape,
          TensorAlgebra.dot_full_zero_right] using
          ih rest ⟨i, Nat.lt_of_succ_lt_succ hi⟩ v

end

end TensorPack

namespace Contributions

/-- A single-parent dense VJP agrees with one exception to the uniform zero reader. -/
theorem lookup_single {α : Type} [TorchLean.Storage α] [Zero α]
    {Γ : List Shape} {shape : Shape} (idx : Idx Γ shape) (value : Tensor α shape) :
    TensorLookup.ofPack (TensorPack.single idx value) =
      (TensorLookup.fill 0).set idx value := by
  apply TensorLookup.ext
  funext otherShape other
  obtain ⟨i, rfl⟩ := idx
  obtain ⟨j, rfl⟩ := other
  induction Γ with
  | nil => exact Fin.elim0 i
  | cons head shapes ih =>
      obtain ⟨i, hi⟩ := i
      obtain ⟨j, hj⟩ := j
      cases i with
      | zero =>
          cases j with
          | zero => rfl
          | succ j =>
              have zero := congrArg (fun (lookup : TensorLookup α shapes) =>
                lookup.read ⟨⟨j, Nat.lt_of_succ_lt_succ hj⟩, rfl⟩)
                (TensorLookup.ofPack_zero (α := α) (shapes := shapes))
              simpa [TensorLookup.ofPack, TensorPack.single, getIdx, TensorLookup.set,
                TensorLookup.fill] using zero
      | succ i =>
          cases j with
          | zero => rfl
          | succ j =>
              simpa [TensorLookup.ofPack, TensorPack.single, getIdx, TensorLookup.set,
                TensorLookup.fill] using ih ⟨i, Nat.lt_of_succ_lt_succ hi⟩ value
                  ⟨j, Nat.lt_of_succ_lt_succ hj⟩

/-- One parent contribution, including the original zero entries at every other position. -/
def single {α : Type} [TorchLean.Storage α] [Zero α]
    {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (value : Tensor α s) :
    Contributions α Γ :=
  { dense := fun _ => TensorPack.single idx value
    lookup := (TensorLookup.fill 0).set idx value
    uniform := 0
    support := [⟨s, idx⟩]
    correct := lookup_single idx value
    outside := by
      intro otherShape other absent
      rw [TensorLookup.read_set_other _ _ _ _ (absent ⟨s, idx⟩ (by simp))]
      rfl }

end Contributions

/-- The original programs of a node after saving its inputs for one execution. -/
structure PreparedPrograms (α : Type) [TorchLean.Storage α] (Γ : List Shape) (τ : Shape) where
  /-- Compute the primal tensor after domain validation has succeeded. -/
  value : Unit → Tensor α τ
  /-- The stored VJP with its primal inputs already captured. -/
  vjp : Tensor α τ → TorchLean.TensorPack α Γ
  /-- Domain validation performed before recording the node. -/
  validate : Unit → Except String Unit

/-- Saved node programs with an optional certified compact backward implementation. -/
structure PreparedNode (α : Type) [TorchLean.Storage α] (Γ : List Shape) (τ : Shape)
    extends PreparedPrograms α Γ τ where
  /-- A compact VJP preserves the complete dense contribution, at every parent position. -/
  compact? : Option {run : Tensor α τ → Contributions α Γ //
    ∀ seed, (run seed).dense () = vjp seed} := none

/--
Executable node payload (no derivative-correctness proof).

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
  An optional preparation program that saves only the node's own inputs. Its certificate
  preserves the complete dense VJP, including zero entries and the order of scalar operations.
  -/
  prepare? : Option {prepare : TensorLookup α Γ → Δ → PreparedNode α Γ τ //
    ∀ ctx d, (prepare (TensorLookup.ofPack ctx) d).toPreparedPrograms =
      { value := fun _ => forward ctx d, vjp := vjp ctx d,
        validate := fun _ => validate ctx d }} := none

namespace NodeData

/--
Build a node whose runtime preparation captures a fixed collection of tensors. The same scalar
programs define the pack interface and the prepared interface, so their agreement is definitional.
-/
def ofLocal {α Δ P : Type} [TorchLean.Storage α] {Γ : List Shape} {τ : Shape}
    (capture : TensorLookup α Γ → P)
    (forward : P → Δ → Tensor α τ)
    (jvp : P → P → Δ → Tensor α τ)
    (vjp : P → Δ → Tensor α τ → TorchLean.TensorPack α Γ)
    (validate : P → Δ → Except String Unit := fun _ _ => .ok ()) :
    NodeData α Δ Γ τ :=
  { forward := fun ctx d => forward (capture (TensorLookup.ofPack ctx)) d
    jvp := fun ctx dctx d =>
      jvp (capture (TensorLookup.ofPack ctx)) (capture (TensorLookup.ofPack dctx)) d
    vjp := fun ctx d => vjp (capture (TensorLookup.ofPack ctx)) d
    validate := fun ctx d => validate (capture (TensorLookup.ofPack ctx)) d
    prepare? := some ⟨fun lookup d =>
      let saved := capture lookup
      { value := fun _ => forward saved d, vjp := vjp saved d,
        validate := fun _ => validate saved d },
      fun _ _ => rfl⟩ }

/-- Local scalar programs with a compact VJP whose dense adapter remains the public VJP. -/
def ofLocalCompact {α Δ P : Type} [TorchLean.Storage α] {Γ : List Shape} {τ : Shape}
    (capture : TensorLookup α Γ → P)
    (forward : P → Δ → Tensor α τ)
    (jvp : P → P → Δ → Tensor α τ)
    (vjp : P → Δ → Tensor α τ → Contributions α Γ)
    (validate : P → Δ → Except String Unit := fun _ _ => .ok ()) :
    NodeData α Δ Γ τ :=
  { forward := fun ctx d => forward (capture (TensorLookup.ofPack ctx)) d
    jvp := fun ctx dctx d =>
      jvp (capture (TensorLookup.ofPack ctx)) (capture (TensorLookup.ofPack dctx)) d
    vjp := fun ctx d seed => (vjp (capture (TensorLookup.ofPack ctx)) d seed).dense ()
    validate := fun ctx d => validate (capture (TensorLookup.ofPack ctx)) d
    prepare? := some ⟨fun lookup d =>
      let saved := capture lookup
      { value := fun _ => forward saved d
        vjp := fun seed => (vjp saved d seed).dense ()
        validate := fun _ => validate saved d
        compact? := some ⟨vjp saved d, fun _ => rfl⟩ },
      fun _ _ => rfl⟩ }

/-- Prepare a node from array storage, using the pack adapter for arbitrary custom closures. -/
def prepare {α Δ : Type} [TorchLean.Storage α] {Γ : List Shape} {τ : Shape}
    (node : NodeData α Δ Γ τ) (ctx : TensorContext α Γ) (data : Δ) :
    PreparedNode α Γ τ :=
  match node.prepare? with
  | some implementation => implementation.val ctx.lookup data
  | none =>
      let inputs := ctx.toPack
      { value := fun _ => node.forward inputs data
        vjp := node.vjp inputs data
        validate := fun _ => node.validate inputs data }

/-- Preparation preserves the original forward, validation, and dense backward programs. -/
@[simp] theorem prepare_ofPack {α Δ : Type} [TorchLean.Storage α]
    {Γ : List Shape} {τ : Shape} (node : NodeData α Δ Γ τ)
    (ctx : TorchLean.TensorPack α Γ) (data : Δ) :
    (node.prepare (TensorContext.ofPack ctx) data).toPreparedPrograms =
      { value := fun _ => node.forward ctx data, vjp := node.vjp ctx data,
        validate := fun _ => node.validate ctx data } := by
  unfold prepare
  split
  next implementation h =>
    rw [TensorContext.lookup_ofPack]
    exact implementation.property ctx data
  next h => simp only [TensorContext.toPack_ofPack]

@[simp] theorem prepare_value_ofPack {α Δ : Type} [TorchLean.Storage α]
    {Γ : List Shape} {τ : Shape} (node : NodeData α Δ Γ τ)
    (ctx : TorchLean.TensorPack α Γ) (data : Δ) :
    (node.prepare (TensorContext.ofPack ctx) data).value = fun _ => node.forward ctx data :=
  congrArg PreparedPrograms.value (prepare_ofPack node ctx data)

@[simp] theorem prepare_vjp_ofPack {α Δ : Type} [TorchLean.Storage α]
    {Γ : List Shape} {τ : Shape} (node : NodeData α Δ Γ τ)
    (ctx : TorchLean.TensorPack α Γ) (data : Δ) :
    (node.prepare (TensorContext.ofPack ctx) data).vjp = node.vjp ctx data :=
  congrArg PreparedPrograms.vjp (prepare_ofPack node ctx data)

@[simp] theorem prepare_validate_ofPack {α Δ : Type} [TorchLean.Storage α]
    {Γ : List Shape} {τ : Shape} (node : NodeData α Δ Γ τ)
    (ctx : TorchLean.TensorPack α Γ) (data : Δ) :
    (node.prepare (TensorContext.ofPack ctx) data).validate = fun _ => node.validate ctx data :=
  congrArg PreparedPrograms.validate (prepare_ofPack node ctx data)

end NodeData

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

/-- Evaluate using indexed storage and save only each node's local input tensors. -/
def evalArray {ss : List Shape} (g : GraphData α Δ Γ ss) (x : TorchLean.TensorPack α Γ)
    (d : Δ) : TensorContext α (Γ ++ ss) :=
  match g with
  | .nil => TensorContext.cast (List.append_nil Γ).symm (TensorContext.ofPack x)
  | .snoc (ss := ss) (τ := τ) previous node =>
      let context := evalArray previous x d
      let prepared := node.prepare context d
      TensorContext.cast (List.append_assoc Γ ss [τ]) (context.push (prepared.value ()))

/-- Indexed forward execution computes the original full typed context. -/
theorem evalArray_eq {ss : List Shape} (g : GraphData α Δ Γ ss)
    (x : TorchLean.TensorPack α Γ) (d : Δ) :
    evalArray g x d = TensorContext.ofPack (g.eval x d) := by
  induction g with
  | nil => simp only [evalArray, eval, TensorContext.cast_ofPack]
  | snoc previous node ih =>
      simp only [evalArray, eval, ih, NodeData.prepare_value_ofPack,
        TensorContext.push_ofPack, TensorContext.cast_ofPack]

/-- Public typed forward result, decoded once after array execution. -/
def evalWithArray {ss : List Shape} (g : GraphData α Δ Γ ss) (x : TorchLean.TensorPack α Γ)
    (d : Δ) : TorchLean.TensorPack α (Γ ++ ss) :=
  (evalArray g x d).toPack

/-- Compile the public evaluator to the proved array execution path. -/
@[csimp] theorem eval_eq_array : @eval = @evalWithArray := by
  funext α storage Δ Γ ss graph inputs data
  simp only [evalWithArray, evalArray_eq, TensorContext.toPack_ofPack]

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

/--
A proof-carrying tape/SSA graph.

Nodes are appended in topological order and may reference any previously computed value (fan-out
and sharing are allowed). This mirrors the structure of PyTorch’s dynamic autograd graph, but with
shape-typed contexts.
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
