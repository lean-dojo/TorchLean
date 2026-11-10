/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.TypedGraph.Context

/-!
# Executable graph data

`NodeData` holds the forward, JVP, VJP and validation programs of one node, and `GraphData` is a
snoc-list of such nodes over the input shapes `Γ`. Neither carries a derivative law. The
proof-carrying `LawfulNode` and `Graph` of `TorchLean.NN.Tape.Graph` pair them
with the local adjointness law. `NodeData.prepare` saves one execution of a node as a
`PreparedNode`, whose optional compact VJP is certified to compute the dense contribution.

## Main declarations

- `TorchLean.Tape.NodeData`, with `NodeData.prepare`, `NodeData.ofLocalCompact` and the
  addition node `NodeData.add`.
- `TorchLean.Tape.GraphData`, with the evaluator `GraphData.eval` (compiled through
  `GraphData.eval_eq_array`), the JVP `GraphData.jvpCtx` and the reverse passes
  `GraphData.backpropCtx` and `GraphData.backpropAllCtx`.
- `TorchLean.TensorPack.single` and `Contributions.single`: the pack and the
  contribution that are a tensor at one context entry and zero elsewhere.
-/

@[expose] public section

namespace TorchLean
namespace Tape

open TorchLean.Tensor

noncomputable section

open TensorPack (getIdx)

section

variable {α : Type} [Storage α]

/-- Sparse context with a single nonzero entry at `idx` (all other tensors are `0`). -/
def _root_.TorchLean.TensorPack.single [Zero α] {Γ : List Shape} {s : Shape} (idx : Idx Γ s)
    (v : Tensor α s) :
    TensorPack α Γ :=
  match Γ, idx with
  | [], ⟨i, _h⟩ =>
      match i with
      | ⟨val, isLt⟩ => False.elim ((Nat.not_lt_zero val) isLt)
  | s0 :: Γtail, ⟨i, h⟩ =>
      match i with
      | ⟨0, _⟩ =>
          .cons (Tensor.castShape v h.symm) (TensorPack.zero (α := α) (ss := Γtail))
      | ⟨Nat.succ j, hj⟩ =>
          let iTail : Fin Γtail.length := ⟨j, Nat.lt_of_succ_lt_succ hj⟩
          let hTail : Γtail.get iTail = s := by
            simpa using h
          .cons (Tensor.full s0 (0 : α)) (single (Γ := Γtail) (s := s) ⟨iTail, hTail⟩ v)

end

namespace Contributions

/-- A single-parent dense VJP agrees with one exception to the uniform zero reader. -/
theorem lookup_single {α : Type} [Storage α] [Zero α]
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
def single {α : Type} [Storage α] [Zero α]
    {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (value : Tensor α s) :
    Contributions α Γ :=
  { dense := fun _ => TensorPack.single idx value
    lookup := (TensorLookup.fill 0).set idx value
    uniform := 0
    support := [⟨s, idx⟩]
    ofPack_dense := lookup_single idx value
    outside := by
      intro otherShape other absent
      rw [TensorLookup.read_set_other _ _ _ _ (absent ⟨s, idx⟩ (by simp))]
      rfl }

end Contributions

/-- The original programs of a node after saving its inputs for one execution. -/
structure PreparedPrograms (α : Type) [Storage α] (Γ : List Shape) (τ : Shape) where
  /-- Compute the primal tensor after domain validation has succeeded. -/
  value : Unit → Tensor α τ
  /-- The stored VJP with its primal inputs already captured. -/
  vjp : Tensor α τ → TensorPack α Γ
  /-- Domain validation performed before recording the node. -/
  validate : Unit → Except String Unit

/-- Saved node programs with an optional certified compact backward implementation. -/
structure PreparedNode (α : Type) [Storage α] (Γ : List Shape) (τ : Shape)
    extends PreparedPrograms α Γ τ where
  /-- A compact VJP preserves the complete dense contribution, at every parent position. -/
  compact? : Option {run : Tensor α τ → Contributions α Γ //
    ∀ seed, (run seed).dense () = vjp seed} := none

/--
Executable node payload (no derivative-correctness proof).

`Δ` is an extra non-differentiable environment threaded through evaluation (e.g. parameters,
auxiliary data). The VJP returns gradients only for the differentiable context `Γ`.
-/
structure NodeData (α : Type) [Storage α]
    (Δ : Type) (Γ : List Shape) (τ : Shape) where
  /-- Evaluate the node from the current differentiable context and auxiliary data. -/
  forward : TensorPack α Γ → Δ → Tensor α τ
  /-- Propagate one tangent context through the node. -/
  jvp : TensorPack α Γ → TensorPack α Γ → Δ → Tensor α τ
  /-- Pull an output cotangent back to the node's differentiable input context. -/
  vjp : TensorPack α Γ → Δ → Tensor α τ → TensorPack α Γ
  /-- Optional runtime precondition, enforced by checked execution before evaluating this node. -/
  validate : TensorPack α Γ → Δ → Except String Unit := fun _ _ => .ok ()
  /--
  An optional preparation program that saves only the node's own inputs. Its certificate
  preserves the complete dense VJP, including zero entries and the order of scalar operations.
  -/
  prepare? : Option {prepare : TensorLookup α Γ → Δ → PreparedNode α Γ τ //
    ∀ ctx d, (prepare (TensorLookup.ofPack ctx) d).toPreparedPrograms =
      { value := fun _ => forward ctx d, vjp := vjp ctx d,
        validate := fun _ => validate ctx d }} := none

namespace NodeData

/-- A node from local scalar programs with a compact VJP. Its `vjp` evaluates the dense program,
and its `prepare?` keeps the compact VJP as `compact?`. -/
def ofLocalCompact {α Δ P : Type} [Storage α] {Γ : List Shape} {τ : Shape}
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
def prepare {α Δ : Type} [Storage α] {Γ : List Shape} {τ : Shape}
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
@[simp] theorem prepare_ofPack {α Δ : Type} [Storage α]
    {Γ : List Shape} {τ : Shape} (node : NodeData α Δ Γ τ)
    (ctx : TensorPack α Γ) (data : Δ) :
    (node.prepare (TensorContext.ofPack ctx) data).toPreparedPrograms =
      { value := fun _ => node.forward ctx data, vjp := node.vjp ctx data,
        validate := fun _ => node.validate ctx data } := by
  unfold prepare
  split
  next implementation h =>
    rw [TensorContext.lookup_ofPack]
    exact implementation.property ctx data
  next h => simp only [TensorContext.toPack_ofPack]

/-- The prepared value of a node on the context of a pack is its `forward` on the pack. -/
@[simp] theorem prepare_value_ofPack {α Δ : Type} [Storage α]
    {Γ : List Shape} {τ : Shape} (node : NodeData α Δ Γ τ)
    (ctx : TensorPack α Γ) (data : Δ) :
    (node.prepare (TensorContext.ofPack ctx) data).value = fun _ => node.forward ctx data :=
  congrArg PreparedPrograms.value (prepare_ofPack node ctx data)

/-- The prepared VJP of a node on the context of a pack is its `vjp` on the pack. -/
@[simp] theorem prepare_vjp_ofPack {α Δ : Type} [Storage α]
    {Γ : List Shape} {τ : Shape} (node : NodeData α Δ Γ τ)
    (ctx : TensorPack α Γ) (data : Δ) :
    (node.prepare (TensorContext.ofPack ctx) data).vjp = node.vjp ctx data :=
  congrArg PreparedPrograms.vjp (prepare_ofPack node ctx data)

/-- The prepared validation of a node on the context of a pack is its `validate` on the pack. -/
@[simp] theorem prepare_validate_ofPack {α Δ : Type} [Storage α]
    {Γ : List Shape} {τ : Shape} (node : NodeData α Δ Γ τ)
    (ctx : TensorPack α Γ) (data : Δ) :
    (node.prepare (TensorContext.ofPack ctx) data).validate = fun _ => node.validate ctx data :=
  congrArg PreparedPrograms.validate (prepare_ofPack node ctx data)

/-- Executable binary add node (two parents of the same shape). -/
def add {α : Type} {Δ : Type} [Storage α] [Zero α] [Add α]
    {Γ : List Shape} {s : Shape}
    (a b : Idx Γ s) : NodeData α Δ Γ s :=
  { forward := fun ctx _d => (getIdx (xs := ctx) a + getIdx (xs := ctx) b)
    jvp := fun _ctx dctx _d => getIdx (xs := dctx) a + getIdx (xs := dctx) b
    vjp := fun _ctx _d δ =>
      TensorPack.add (α := α) (ss := Γ)
        (TensorPack.single (α := α) (Γ := Γ) a δ)
        (TensorPack.single (α := α) (Γ := Γ) b δ) }

end NodeData

/-- Executable-only graph: a snoc-list of `NodeData`. -/
inductive GraphData (α : Type) [Storage α]
    (Δ : Type) (Γ : List Shape) : List Shape → Type where
  /-- A graph with no computed nodes, whose context consists only of the inputs `Γ`. -/
  | nil : GraphData α Δ Γ []
  /-- Append one node whose inputs may use the original values and those computed before it. -/
  | snoc {ss : List Shape} {τ : Shape} :
      GraphData α Δ Γ ss → NodeData α Δ (Γ ++ ss) τ → GraphData α Δ Γ (ss ++ [τ])

namespace GraphData

variable {α : Type} [Storage α]
variable {Δ : Type}
variable {Γ : List Shape}

/-- Evaluate a `GraphData` on an input context `x`, producing the full context `Γ ++ ss`. -/
def eval {ss : List Shape} (g : GraphData α Δ Γ ss) (x : TensorPack α Γ) (d : Δ) :
    TensorPack α (Γ ++ ss) :=
  match g with
  | .nil => Pack.cast (h := (List.append_nil Γ).symm) x
  | .snoc (ss := ss) (τ := τ) g node =>
      let ctx := eval (ss := ss) g x d
      let y := node.forward ctx d
      Pack.cast (h := List.append_assoc Γ ss [τ])
        (Pack.snoc (ss := Γ ++ ss) (τ := τ) ctx y)

/-- Evaluate using indexed storage and save only each node's local input tensors. -/
def evalArray {ss : List Shape} (g : GraphData α Δ Γ ss) (x : TensorPack α Γ)
    (d : Δ) : TensorContext α (Γ ++ ss) :=
  match g with
  | .nil => TensorContext.cast (List.append_nil Γ).symm (TensorContext.ofPack x)
  | .snoc (ss := ss) (τ := τ) previous node =>
      let context := evalArray previous x d
      let prepared := node.prepare context d
      TensorContext.cast (List.append_assoc Γ ss [τ]) (context.push (prepared.value ()))

/-- Indexed forward execution computes the original full typed context. -/
theorem evalArray_eq {ss : List Shape} (g : GraphData α Δ Γ ss)
    (x : TensorPack α Γ) (d : Δ) :
    evalArray g x d = TensorContext.ofPack (g.eval x d) := by
  induction g with
  | nil => simp only [evalArray, eval, TensorContext.cast_ofPack]
  | snoc previous node ih =>
      simp only [evalArray, eval, ih, NodeData.prepare_value_ofPack,
        TensorContext.push_ofPack, TensorContext.cast_ofPack]

/-- Public typed forward result, decoded once after array execution. -/
def evalWithArray {ss : List Shape} (g : GraphData α Δ Γ ss) (x : TensorPack α Γ)
    (d : Δ) : TensorPack α (Γ ++ ss) :=
  (evalArray g x d).toPack

/-- Compile the public evaluator to the proved array execution path. -/
@[csimp] theorem eval_eq_array : @eval = @evalWithArray := by
  funext α storage Δ Γ ss graph inputs data
  simp only [evalWithArray, evalArray_eq, TensorContext.toPack_ofPack]

/-- Compute the JVP of `eval`, producing a tangent context of shape `Γ ++ ss`. -/
def jvpCtx {ss : List Shape} (g : GraphData α Δ Γ ss) (x : TensorPack α Γ)
    (dx : TensorPack α Γ) (d : Δ) :
    TensorPack α (Γ ++ ss) :=
  match g with
  | .nil => Pack.cast (h := (List.append_nil Γ).symm) dx
  | .snoc (ss := ss) (τ := τ) g node =>
      let ctx := eval (ss := ss) g x d
      let dctx := jvpCtx (ss := ss) g x dx d
      let dy := node.jvp ctx dctx d
      Pack.cast (h := List.append_assoc Γ ss [τ])
        (Pack.snoc (ss := Γ ++ ss) (τ := τ) dctx dy)

/-- Reverse-mode accumulation on contexts (VJP), given a seed cotangent for `Γ ++ ss`. -/
def backpropCtx [Add α] {ss : List Shape} (g : GraphData α Δ Γ ss) (x : TensorPack α Γ)
    (d : Δ) (seed : TensorPack α (Γ ++ ss)) : TensorPack α Γ :=
  match g with
  | .nil => Pack.cast (h := List.append_nil Γ) seed
  | .snoc (ss := ss) (τ := τ) g node =>
      let seed' : TensorPack α ((Γ ++ ss) ++ [τ]) :=
        Pack.cast (h := (List.append_assoc Γ ss [τ]).symm) seed
      let seedPrev : TensorPack α (Γ ++ ss) :=
        (Pack.unsnoc (ss := Γ ++ ss) (τ := τ) seed').1
      let seedOut : Tensor α τ :=
        (Pack.unsnoc (ss := Γ ++ ss) (τ := τ) seed').2
      let ctx := eval (ss := ss) g x d
      let contrib := node.vjp ctx d seedOut
      let seedPrev' := TensorPack.add (α := α) (ss := Γ ++ ss) seedPrev contrib
      backpropCtx (ss := ss) g x d seedPrev'

end GraphData

end

/-!
### Full backpropagation (dense) for proofs and runtime

The runtime engine computes a *dense* gradient array, accumulating cotangents for every node in the
tape (inputs and intermediates). `GraphData.backpropAllCtx` is the same pass on graph data, and the
theorems under `TorchLean/NN/Tape/Lowering/` prove that the runtime loop on a lowered
tape agrees with it.
-/

namespace GraphData

/--
"Full" backpropagation for `GraphData`: gradients for every value in `Γ ++ ss`, including inputs.

Peel off the last node, apply its VJP to the seed on that node, add the result into the previous
seed, and recurse. Only `[Add α]` is needed.
-/
def backpropAllCtx {α : Type} {Δ : Type} [Storage α] [Add α]
    {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss) (x : TensorPack α Γ)
    (d : Δ) (seed : TensorPack α (Γ ++ ss)) :
    TensorPack α (Γ ++ ss) :=
  match g with
  | .nil => seed
  | .snoc (ss := ssPrev) (τ := τ) g node =>
      let assoc : (Γ ++ ssPrev) ++ [τ] = Γ ++ (ssPrev ++ [τ]) := List.append_assoc Γ ssPrev [τ]
      let seed' : TensorPack α ((Γ ++ ssPrev) ++ [τ]) :=
        Pack.cast (h := assoc.symm) seed
      let seedPrev : TensorPack α (Γ ++ ssPrev) :=
        (Pack.unsnoc (ss := Γ ++ ssPrev) (τ := τ) seed').1
      let seedOut : Tensor α τ :=
        (Pack.unsnoc (ss := Γ ++ ssPrev) (τ := τ) seed').2
      let ctx := GraphData.eval (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d
      let contrib := node.vjp ctx d seedOut
      let seedPrev' := TensorPack.add (α := α) (ss := Γ ++ ssPrev) seedPrev contrib
      let gradsPrev := backpropAllCtx (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d seedPrev'
      Pack.cast (h := assoc)
        (Pack.snoc (ss := Γ ++ ssPrev) (τ := τ) gradsPrev seedOut)

end GraphData

end Tape
end TorchLean
