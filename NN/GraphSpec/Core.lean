/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.DAG.Core
public import NN.Runtime.Autograd.TorchLean.NN
import Mathlib.Algebra.Order.Algebra

/-!
# Sequential GraphSpec

This file defines the **sequential authoring surface** for GraphSpec.

The important design decision is:

- `DAG.Model` is the canonical general GraphSpec model representation.
- `Graph ps σ τ` is lightweight syntax for the common special case where the model is just a chain
  of layers.

So `Graph` is not a competing graph IR. It is a pleasant way to write:

```lean
Linear >>> ReLU >>> Linear
Conv >>> ReLU >>> Pool >>> Flatten >>> Linear
```

and then lower that chain to the general DAG representation when downstream tooling wants one
model shape for everything.

GraphSpec as a whole is a *typed* DSL for describing neural-network computations, with the explicit
goal of being usable in two complementary ways:

1. **Reference / proof semantics**: interpret the graph as a *pure* Lean function on tensors
   (`Interp.spec`). This is the semantics we want to reason about: shape safety, algebraic
   identities, equivalence of model refactorings, etc.
2. **Executable semantics**: compile the same graph into a backend-generic `TorchLean.Program`
   (`Compile.torchProgram`) so it can run on the TorchLean runtime (which can target eager or
   compiled execution backends).

Shapes and parameter shapes are part of the graph type.
Concretely, a graph is indexed by:

- `σ τ : Shape` — input/output tensor shapes, and
- `ps : List Shape` — the ordered list of parameter tensor shapes the graph expects.

That “parameter interface” is not a convention (like “whatever `state_dict()` happens to return”);
it is baked into the model type. Sequential composition concatenates parameter lists, and
evaluation splits them canonically.

## Why do this if PyTorch already exists?

PyTorch is excellent at *running* and *training* neural networks:

- `nn.Module` packages parameters and a `forward` method.
- Autograd records an operation tape during execution and provides gradients.
- Modern tooling can capture/transform graphs (`torch.fx`, `torch.export`) and compile them
  (`torch.compile`).

GraphSpec is not trying to replace any of that. Instead, it focuses on the pieces PyTorch does not
give us *inside Lean*:

- A machine-checkable **mathematical semantics** we can use for proofs.
- **Static shape discipline**: shapes appear in the type, not as runtime asserts.
- A **typed parameter interface**: parameter shapes and ordering are explicit, so “which tensor is
  weight #2?” is not an out-of-band convention.

In practice, the expected workflow is:

- use GraphSpec to write down a model architecture in a shape-typed way,
- run it via TorchLean for concrete execution/training experiments, and
- use `Interp.spec` and proof libraries to prove properties of the same architecture.

## Why do this if TorchLean already exists?

TorchLean is the **runtime and operator** layer: it gives us typed tensors, a backend interface,
and executable programs (`TorchLean.Program`) that can run under the autograd/training runtime.

GraphSpec is the **architecture/specification** layer: it gives us a small typed syntax for model
structure that comes with *two linked meanings*:

- a **pure** semantics (`Interp.spec`) that is amenable to proofs in Lean, and
- a **compiler** (`Compile.torchProgram`) that turns the same description into something runnable.

You *can* write models directly in TorchLean, but then the “thing you reason about” is already in
the executable world (monadic references + backend ops). For many proofs, it is much cleaner to
reason about a pure function `Params → Tensor → Tensor` and separately prove that compilation to
the runtime preserves that meaning.

In other words:

- TorchLean answers: “Given ops, how do we run/train them?”
- GraphSpec answers: “How do we describe models so we can both run them and prove things about
  them?”

## Mathematical View For Sequential Chains

For `g : Graph ps σ τ`, think of `g` as denoting a function

`⟦g⟧ : Params(ps) → Tensor σ → Tensor τ`.

In this file, that semantics is implemented by `Interp.spec`, and it is defined structurally:

- `⟦id⟧ params x = x`
- `⟦prim p⟧ params x = p.specFwd params x`
- `⟦g₁ >>> g₂⟧ params x`:
  split `params : Params(ps₁ ++ ps₂)` into `(params₁ : Params ps₁, params₂ : Params ps₂)`, then
  compute `⟦g₂⟧ params₂ (⟦g₁⟧ params₁ x)`.

The compiler `Compile.torchProgram` follows the same structure, but targets a monadic Torch
interface and expects arguments as `params ++ [input]` (matching `TorchLean.NN.Seq.program`).

## Scope of `Core.lean`

This file defines only the sequential core:

- `Primitive` — a single typed operation with both a pure spec and a TorchLean implementation.
- `Graph` — sequential composition (`>>>`) of primitives with a typed parameter list.
- `Interp.spec` — pure interpreter.
- `Compile.torchProgram` — compiler to `TorchLean.Program`.
- `LowerToDAG.Graph.toDAGTerm` / `toDAGModelZeroInit` — the bridge from chain syntax to the
  canonical DAG representation.

For skip connections, shared intermediates, residual adds, or other multi-input nodes, use
`NN.GraphSpec.DAG` directly.

## Direction

GraphSpec is intended to grow into a hygienic “write once, run/prove many” layer:

- richer primitive packs (vision, language, classical ML, …),
- richer DAG structure (limited control flow where it can be compiled),
- verified compilation passes (fusion, constant folding, layout transforms) with proofs that they
  preserve `Interp.spec`,
- a better parameter/initialization interface (explicit RNG threading, serialization, interop with
  PyTorch/ONNX exports),
- and a library of reusable theorems about common architectures (e.g. invariants of residual
  blocks, bounds for Lipschitz constants, etc.).

## References / citations

- PyTorch `nn.Module` and graph tooling (`torch.fx`, `torch.export`, `torch.compile`) for the
  practical “execution/training first” baseline.
- Automatic differentiation background: Baydin et al. (2018), “Automatic Differentiation in Machine
  Learning: a Survey”.
- ReLU: Nair & Hinton (2010).
-/

@[expose] public section


namespace NN
namespace GraphSpec

open Spec
open Tensor
open NN.Tensor

/-! ## Core graph language -/

/--
A *primitive node* in the GraphSpec language.

GraphSpec primitives package both sides of the “spec vs runtime” interface:

- a **pure spec forward** function (`specFwd`) used by the reference interpreter, and
- a **TorchLean program** (`torchProgram`) used by the compiler.

Optionally, a primitive may also provide a lowering to a TorchLean `LayerDef` (used to build a
`TorchLean.NN.Seq` for training ergonomics + deterministic parameter initialization). Not every
primitive needs this (e.g. control-flow-ish nodes kept outside the sequential layer).

Why a record?

- It lets us grow the op set by adding new primitives in new files (rather than editing a single
  global inductive just to extend the vocabulary).
- It keeps the “spec vs TorchLean” linkage explicit: when you add an op, you must define both
  interpretations side-by-side.

Type indices:

- `ps : List Shape` are the *parameter tensor shapes* this primitive expects, in order.
- `σ τ : Shape` are input/output tensor shapes.
-/
structure Primitive (ps : List Shape) (σ τ : Shape) where
  /-- Human-readable name used mainly for debugging / error messages. -/
  name : String
  /--
  Pure reference semantics (forward pass).

  This is the function used by `Interp.spec`.
  -/
  specFwd :
    ∀ {α : Type 0}, [Context α] →
      Runtime.Autograd.Torch.TList α ps → Tensor α σ → Tensor α τ
  /--
  Executable TorchLean program.

  The program expects its arguments as `ps ++ [σ]` (all parameters first, then the input).
  This convention aligns with how sequential TorchLean models (`TorchLean.NN.Seq`) expose their
  program interfaces.
  -/
  torchProgram :
    ∀ {α : Type 0}, [Context α] → [DecidableEq Shape] →
      Runtime.Autograd.TorchLean.Program α (ps ++ [σ]) τ
  /--
  Optional lowering to a TorchLean `LayerDef`.

  We thread an occurrence index (`Nat`) so primitives can implement deterministic “per-layer”
  initialization (e.g. seed = f(index)).
  -/
  toLayerDefM? :
    Option (Nat → { l : Runtime.Autograd.TorchLean.NN.LayerDef σ τ // l.paramShapes = ps }) := none
  /-- Whether encountering this primitive should increment the layer-occurrence counter. -/
  countsAsLayer : Bool := false

/--
`Graph ps σ τ` is a (restricted) model that:
- takes an input tensor of shape `σ`,
- produces an output tensor of shape `τ`,
- and uses parameters whose shapes are listed in `ps` (in order).

This is a *sequential* (chain) graph language: the only composition operator is `seq` (`>>>`).
For sharing/skip connections, use `NN.GraphSpec.DAG`.

Implementation note:
- We encode the parameter list at the type level so composition automatically concatenates
  parameter lists (`ps := ps₁ ++ ps₂`).
- This means every graph has a canonical “ABI” for parameters: a single typed list `TList α ps`.
  When composing `g₁ : Graph ps₁ σ τ` and `g₂ : Graph ps₂ τ υ`, the composite graph expects
  parameters of shape list `ps₁ ++ ps₂`, and evaluation splits that list into the pieces needed by
  each subgraph.
-/
inductive Graph : List Shape → Shape → Shape → Type 2 where
  /-- Identity graph: passes the input through unchanged and requires no parameters. -/
  | id (s : Shape) : Graph [] s s
  /-- Sequential composition. Parameter lists concatenate. -/
  | seq {ps₁ ps₂ : List Shape} {σ τ υ : Shape} :
      Graph ps₁ σ τ → Graph ps₂ τ υ → Graph (ps₁ ++ ps₂) σ υ
  /-- Embed a single primitive node as a graph. -/
  | prim {ps : List Shape} {σ τ : Shape} :
      Primitive ps σ τ → Graph ps σ τ

infixr:80 " >>> " => Graph.seq

/-! ## Standard primitives (initial op set) -/

namespace Primitive

/--
Primitive linear layer.

Mathematical semantics (vector case):

Let `x : Vec inDim`, `W : Mat outDim inDim`, and `b : Vec outDim`. Then:

`linear(W,b,x) = W * x + b`.

This matches the standard dense layer as in PyTorch `torch.nn.Linear` / `torch.nn.functional.linear`
(up to the usual row/column convention; here the shape indices make the intended dimensions
explicit).

Type-level parameter interface:

- parameter shapes are `[Mat outDim inDim, Vec outDim]`,
- input shape is `Vec inDim`,
- output shape is `Vec outDim`.

So a graph containing a `linear` node *forces* you to supply exactly a weight matrix and bias
vector of the right shapes, and it fixes their ordering in the model’s parameter list.

References:
- Dense layers are standard; for PyTorch behavior see `torch.nn.Linear` documentation.
- For the semantics used by the spec interpreter, see `NN.Spec.Module.Linear` (`Spec.linear_spec`).

Initialization semantics:
- we attach a TorchLean `LayerDef` so graphs can be lowered to `TorchLean.NN.Seq`,
- and we seed `W,b` deterministically from the layer-occurrence index:
  - `seedW = 2*i`, `seedB = 2*i + 1`.

The deterministic occurrence-index rule keeps end-to-end examples reproducible while preserving a
single GraphSpec → TorchLean → training path.
-/
def linear (inDim outDim : Nat) :
    Primitive
      [NN.Tensor.Shape.Mat outDim inDim, NN.Tensor.Shape.Vec outDim]
      (NN.Tensor.Shape.Vec inDim) (NN.Tensor.Shape.Vec outDim) :=
  { name := s!"linear({inDim},{outDim})"
    specFwd := fun {α} _ctx params x =>
      match params with
      | .cons w (.cons b .nil) =>
          let lin : Spec.LinearSpec α inDim outDim := { weights := w, bias := b }
          Spec.linearSpec (α := α) lin x
    torchProgram := fun {α} _ctx _deq =>
      -- Program takes args as `W → b → x → ...`.
      fun {m} _instM _instOps =>
        fun w b x =>
          Runtime.Autograd.TorchLean.linear (m := m) (α := α)
            (inDim := inDim) (outDim := outDim) w b x
    toLayerDefM? := some (fun i =>
      ⟨ Runtime.Autograd.TorchLean.NN.linear inDim outDim (seedW := 2 * i) (seedB := 2 * i + 1)
      , by rfl ⟩)
    countsAsLayer := true
  }

/--
ReLU activation (parameter-free).

Mathematical semantics: elementwise `relu(x) = max(x, 0)`.

This is *shape-preserving* and *parameter-free*, so its parameter list is `[]` and its input/output
shape indices are both `s`.

References:
- Nair & Hinton (2010), “Rectified Linear Units Improve Restricted Boltzmann Machines”.
- Spec definition: `Activation.relu_spec` in `NN.Spec.Layers.Activation`.
-/
def relu (s : Shape) : Primitive [] s s :=
  { name := "relu"
    specFwd := fun {α} _ctx _params x => Activation.reluSpec (α := α) x
    torchProgram := fun {α} _ctx _deq =>
      fun {m} _ _ =>
        fun x => Runtime.Autograd.TorchLean.relu (m := m) (α := α) (s := s) x
    toLayerDefM? := some (fun _i => ⟨Runtime.Autograd.TorchLean.NN.relu (s := s), by rfl⟩)
    countsAsLayer := false
  }

/--
Last-axis softmax (parameter-free).

Softmax turns “logits” into a probability distribution along the *last* axis:

`softmax(x)_i = exp(x_i) / (∑_j exp(x_j))`.

In TorchLean’s spec layer, this is implemented as a genuine last-axis tensor softmax (recursing
over outer dimensions), analogous to `torch.softmax(x, dim=-1)` in PyTorch.

Notes:
- Softmax is *not* elementwise; it normalizes across an axis, so it is a canonical example of a
  non-pointwise activation.
- For numerical stability, practical implementations often rewrite softmax using `logsumexp`.
  The spec semantics here follows the dedicated `Activation.softmax_spec`.

References:
- Spec definition: `Activation.softmax_spec` in `NN.Spec.Layers.Activation`.
- PyTorch API analogy: `torch.softmax(x, dim=-1)`.
-/
def softmax (s : Shape) : Primitive [] s s :=
  { name := "softmax"
    specFwd := fun {α} _ctx _params x => Activation.softmaxSpec (α := α) (s := s) x
    torchProgram := fun {α} _ctx _deq =>
      fun {m} _ _ =>
        fun x => Runtime.Autograd.TorchLean.softmax (m := m) (α := α) (s := s) x
    -- TorchLean has the op; treat it as a parameter-free layer for Seq lowering.
    toLayerDefM? := some (fun _i =>
      ⟨ { paramShapes := []
          initParams := .nil
          forward := fun _ {α} _ _ =>
            fun {m} _ _ =>
              fun x => Runtime.Autograd.TorchLean.softmax (m := m) (α := α) (s := s) x }
      , by rfl ⟩)
    countsAsLayer := false
  }

end Primitive

namespace Graph

/-- Graph constructor for `Primitive.linear`. -/
def linear (inDim outDim : Nat) :
    Graph [NN.Tensor.Shape.Mat outDim inDim, NN.Tensor.Shape.Vec outDim]
      (NN.Tensor.Shape.Vec inDim) (NN.Tensor.Shape.Vec outDim) :=
  .prim (Primitive.linear inDim outDim)

/-- Graph constructor for `Primitive.relu`. -/
def relu (s : Shape) : Graph [] s s :=
  .prim (Primitive.relu s)

/-- Graph constructor for `Primitive.softmax`. -/
def softmax (s : Shape) : Graph [] s s :=
  .prim (Primitive.softmax s)

end Graph

/-! ## Lowering: sequential Graph → DAG term/model -/

/-!
GraphSpec has two surface syntaxes:

- `NN.GraphSpec.Core`: a *sequential* DSL (`Graph` + `>>>`), ideal for pure pipelines.
- `NN.GraphSpec.DAG.Core`: a *general* SSA/A-normal-form term language, ideal for sharing/skip
  connections.

The DAG term/model language is GraphSpec’s “general graph” core: it is the representation that can
express sharing and skip connections.

Sequential `Graph` exists because it is the clearest way to write pipelines, and it has its own
direct Spec semantics (`Interp.spec`) and compiler (`Compile.torchProgram`).

This lowering is still useful whenever you want to *embed* a sequential pipeline into the DAG world
(e.g. to reuse DAG-only tooling, or to keep a single GraphSpec example surface that can export DAG
models).

This section provides a structural lowering:

- `Graph.toDAGTerm` produces a `DAG.Term (ps ++ [σ]) τ`, i.e. a DAG term whose environment starts
  with the parameter list `ps` and ends with the (single) data input `σ`.
- `Graph.toDAGModelZeroInit` wraps that term into a `DAG.Model ps [σ] τ` with a simple default
  parameter initialization (all zeros).

Notes:
- The lowering is *purely structural*: it introduces `let1` binders between stages to make the
  sequential flow explicit in SSA form.
- Each sequential `Primitive ps σ τ` is embedded as a DAG primitive op with inputs `ps ++ [σ]`.
  This embedding is generic: any custom GraphSpec primitive automatically becomes usable in the
  DAG world.
-/

namespace LowerToDAG

open Runtime.Autograd.Torch (TList)

/-!
### Lowering internals

The definitions in this section (`castTerm`, `toTerm`, …) are internal adapters for the structural
lowering. The intended public API is `Graph.toDAGTerm` / `Graph.toDAGModelZeroInit`.
-/

/-- Cast a `DAG.Term` across a proven equality of output shapes. -/
def castTerm {Γ : List Shape} {s t : Shape} (h : s = t) :
    DAG.Term Γ s → DAG.Term Γ t :=
  fun x => DAG.Term.cast x h

/-- Cast the environment of a `DAG.Term` across a proven equality of environments. -/
def castEnvTerm {Γ Γ' : List Shape} {τ : Shape} (h : Γ = Γ') :
    DAG.Term Γ τ → DAG.Term Γ' τ :=
  fun x => DAG.Term.castEnv x h

/-- Cast the environment of `DAG.Args` across a proven equality of environments. -/
def castEnvArgs {Γ Γ' : List Shape} {ins : List Shape} (h : Γ = Γ') :
    DAG.Args Γ ins → DAG.Args Γ' ins := by
  cases h
  intro xs
  exact xs

/-! ### `List.get` lemmas (small, self-contained) -/

/-- `List.get` into `as` is unchanged by appending a right list (Nat-index form). -/
lemma get_append_left_nat {α : Type} :
    ∀ (as bs : List α) (i : Nat) (hi : i < as.length),
      (as ++ bs).get ⟨i, by
        simpa [List.length_append] using Nat.lt_of_lt_of_le hi (Nat.le_add_right _ _)⟩
      =
      as.get ⟨i, hi⟩
  | [], _bs, _i, hi => by simp at hi
  | _a :: as, bs, 0, _hi => rfl
  | _a :: as, bs, (i + 1), hi => by
      have hi' : i < as.length := Nat.lt_of_succ_lt_succ hi
      -- Reduce to the tail case.
      -- (The definitional reduction of `List.get` on a successor index handles the index-shift.)
      exact get_append_left_nat as bs i hi'

/--
`List.get` into the right list after appending, using an explicit offset `as.length + j`
(Nat-index form).
-/
lemma get_append_right_offset_nat {α : Type} :
    ∀ (as bs : List α) (j : Nat) (hj : as.length + j < (as ++ bs).length),
      (as ++ bs).get ⟨as.length + j, hj⟩
      =
      bs.get ⟨j, by
        have : as.length + j < as.length + bs.length := by
          simpa [List.length_append] using hj
        exact Nat.lt_of_add_lt_add_left this⟩
  | [], bs, j, _hj => by
      -- `[] ++ bs = bs`, and the two `Fin` proofs are propositionally equal.
      let idxL : Fin bs.length := ⟨j, by simpa using _hj⟩
      let idxR : Fin bs.length := ⟨j, by
        have : ([] : List α).length + j < ([] : List α).length + bs.length := by
          simpa using (by simpa [List.nil_append] using _hj)
        exact Nat.lt_of_add_lt_add_left this⟩
      have hIdx : idxL = idxR := by
        apply Fin.ext
        rfl
      -- Both sides are `bs.get` at the same index.
      simp [idxL, idxR, hIdx]
  | a :: as, bs, j, hj => by
      -- Re-express the index as a successor so `List.get` reduces to the tail.
      have hIdx2 : Nat.succ (as.length + j) < ((a :: as) ++ bs).length := by
        simpa [List.length_append, Nat.succ_add, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]
          using hj
      let idx1 : Fin (((a :: as) ++ bs).length) := ⟨(a :: as).length + j, hj⟩
      let idx2 : Fin (((a :: as) ++ bs).length) := ⟨Nat.succ (as.length + j), hIdx2⟩
      have hidx : idx1 = idx2 := by
        apply Fin.ext
        simp [idx1, idx2, List.length, Nat.succ_add, Nat.add_assoc, Nat.add_comm]
      have hj' : as.length + j < (as ++ bs).length :=
        Nat.lt_of_succ_lt_succ (by
          -- `((a :: as) ++ bs).length = Nat.succ ((as ++ bs).length)`
          simpa [List.length_append] using hIdx2)
      calc
        ((a :: as) ++ bs).get idx1
            = ((a :: as) ++ bs).get idx2 := by simp [hidx]
        _ = (as ++ bs).get ⟨as.length + j, hj'⟩ := by
              -- definitional reduction of `List.get` on a successor index
              rfl
        _ = bs.get ⟨j, by
              have : as.length + j < as.length + bs.length := by
                simpa [List.length_append] using hj'
              exact Nat.lt_of_add_lt_add_left this⟩ := by
              exact get_append_right_offset_nat as bs j hj'

/-- `List.get` of the last element after appending a singleton list. -/
lemma get_append_last {α : Type} :
    ∀ (xs : List α) (x : α),
      (xs ++ [x]).get ⟨xs.length, by simp [List.length_append]⟩ = x
  | [], x => rfl
  | _a :: xs, x => by
      -- Reduce to tail.
      simp

/-! ### Primitive embedding: `Primitive` → `DAG.PrimOp` -/

/--
Embed a sequential GraphSpec primitive as a DAG primitive op.

The resulting op has input shapes `ps ++ [σ]` (parameters followed by the data input).
 -/
def Primitive.toDAGPrimOp {ps : List Shape} {σ τ : Shape} (p : Primitive ps σ τ) :
    DAG.PrimOp (ps ++ [σ]) τ :=
  { name := p.name
    specFwd := fun {α} _ctx xs =>
      let (params, xs') :=
        Runtime.Autograd.Torch.Proofs.Autograd.Algebra.TList.splitAppend
          (α := α) (ss₁ := ps) (ss₂ := [σ]) xs
      match xs' with
      | .cons x .nil => p.specFwd (α := α) params x
    torchProgram := fun {α} _ctx _deq => p.torchProgram (α := α)
  }

/-! ### Building well-typed DAG arguments for a primitive call -/

lemma get_succ
    {α : Type} (a : α) (as : List α) (i : Fin as.length) :
    (a :: as).get ⟨i.1 + 1, Nat.succ_lt_succ i.2⟩ = as.get i := by
  cases i with
  | mk i hi =>
    -- `List.get` on a successor index reduces definitionally to the tail.
    rfl

/--
Build a typed `DAG.Args` list from an index-based family of argument terms.

This is the bridge from “arguments as a function of `Fin ins.length`” to the inductive `DAG.Args`
encoding used by `DAG.Term.op`.
-/
def argsOfFn {Γ : List Shape} :
    {ins : List Shape} →
    (∀ i : Fin ins.length, DAG.Term Γ (ins.get i)) →
    DAG.Args Γ ins
  | [], _f => .nil
  | s :: ss, f =>
      -- Head: index 0.
      let head : DAG.Term Γ ((s :: ss).get ⟨0, by simp⟩) := f ⟨0, by simp⟩
      -- Tail: shift indices by 1, and cast the `List.get` result to match `ss.get i`.
      let tail : DAG.Args Γ ss :=
        argsOfFn (ins := ss) (fun i =>
          castTerm (Γ := Γ) (s := (s :: ss).get ⟨i.1 + 1, Nat.succ_lt_succ i.2⟩) (t := ss.get i)
            (get_succ (a := s) (as := ss) i) (f ⟨i.1 + 1, Nat.succ_lt_succ i.2⟩))
      .cons (by simpa using head) tail

/-- Append one final term to a typed DAG argument list. -/
def Args.append1 {Γ : List Shape} {ps : List Shape} {σ : Shape} :
    DAG.Args Γ ps → DAG.Term Γ σ → DAG.Args Γ (ps ++ [σ])
  | .nil, x => .cons x .nil
  | .cons t ts, x => .cons t (Args.append1 ts x)

/--
Reference the `i`th parameter block inside a larger environment layout.

The surrounding environment is split as `pre ++ ps ++ post ++ extra`; this helper returns the term
that points at parameter `i : Fin ps.length` while keeping the full ambient environment explicit.
-/
def mkParamTerm
    {pre ps post extra : List Shape}
    (i : Fin ps.length) :
    DAG.Term ((pre ++ ps ++ post) ++ extra) (ps.get i) := by
  let Γ : List Shape := (pre ++ ps ++ post) ++ extra
  let n : Nat := pre.length + i.1
  have hPrePsPost : n < (pre ++ ps ++ post).length := by
    have hi' : i.1 < (ps ++ post).length := by
      have : i.1 < ps.length + post.length := Nat.lt_of_lt_of_le i.2 (Nat.le_add_right _ _)
      simpa [List.length_append, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using this
    have : pre.length + i.1 < pre.length + (ps ++ post).length :=
      Nat.add_lt_add_left hi' pre.length
    simpa [n, List.length_append] using this
  have hΓ : n < Γ.length := by
    have : n < (pre ++ ps ++ post).length + extra.length :=
      Nat.lt_of_lt_of_le hPrePsPost (Nat.le_add_right _ _)
    simpa [Γ, List.length_append, Nat.add_assoc] using this
  let idx : Fin Γ.length := ⟨n, hΓ⟩
  have hGet : Γ.get idx = ps.get i := by
    -- Drop the trailing `extra`.
    have hExtra :
        Γ.get idx
          =
        (pre ++ ps ++ post).get ⟨n, hPrePsPost⟩ := by
      have hIdx :
          idx
            =
          ⟨n, by
            have : n < (pre ++ ps ++ post).length + extra.length :=
              Nat.lt_of_lt_of_le hPrePsPost (Nat.le_add_right _ _)
            simpa [Γ, List.length_append, Nat.add_assoc] using this⟩ := by
        apply Fin.ext
        rfl
      simpa [Γ, hIdx] using
        (get_append_left_nat (as := (pre ++ ps ++ post)) (bs := extra) (i := n) (hi := hPrePsPost))
    -- Strip the leading `pre` inside `pre ++ (ps ++ post)`.
    have hPre :
        (pre ++ ps ++ post).get ⟨n, hPrePsPost⟩
          =
        (ps ++ post).get ⟨i.1, by
          have : i.1 < (ps ++ post).length := by
            have : i.1 < ps.length + post.length := Nat.lt_of_lt_of_le i.2 (Nat.le_add_right _ _)
            simpa [List.length_append, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using this
          simpa [List.length_append] using this⟩ := by
      have hj : pre.length + i.1 < (pre ++ (ps ++ post)).length := by
        simpa [n, List.length_append, Nat.add_assoc] using hPrePsPost
      -- Strip `pre` via `get_append_right_offset_nat` (with `j = i.1`).
      simp [n]
    -- Drop the trailing `post`, focusing to `ps`.
    have hPost :
        (ps ++ post).get ⟨i.1, by
          have : i.1 < (ps ++ post).length := by
            have : i.1 < ps.length + post.length := Nat.lt_of_lt_of_le i.2 (Nat.le_add_right _ _)
            simpa [List.length_append, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using this
          simpa [List.length_append] using this⟩
          =
        ps.get i := by
      -- `get_append_left_nat` returns `ps.get ⟨i.1, i.2⟩`; align that index with `i`.
      have hFin : (⟨i.1, i.2⟩ : Fin ps.length) = i := by
        apply Fin.ext
        rfl
      -- Match the `Fin` proof used by `get_append_left_nat` on the left.
      have hIdx :
          (⟨i.1, by
            simpa [List.length_append] using Nat.lt_of_lt_of_le i.2 (Nat.le_add_right _ _)⟩ : Fin
              (ps ++ post).length)
            =
          ⟨i.1, by
            have : i.1 < (ps ++ post).length := by
              have : i.1 < ps.length + post.length := Nat.lt_of_lt_of_le i.2 (Nat.le_add_right _ _)
              simpa [List.length_append, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using this
            simpa [List.length_append] using this⟩ := by
        apply Fin.ext
        rfl
      have h0 :
          (ps ++ post).get ⟨i.1, by
            simpa [List.length_append] using Nat.lt_of_lt_of_le i.2 (Nat.le_add_right _ _)⟩
            =
          ps.get ⟨i.1, i.2⟩ :=
        get_append_left_nat (as := ps) (bs := post) (i := i.1) (hi := i.2)
      simp
    exact Eq.trans hExtra (Eq.trans hPre (Eq.trans hPost rfl))
  simpa [Γ] using castTerm hGet (DAG.Term.var (Γ := Γ) idx)

/--
Lower a unary `Primitive` application into the DAG term language.

Parameters are read from the middle `ps` segment of the ambient environment, in the same order as
the primitive's parameter ABI, and the final data input is supplied by `x`.
-/
def primCall
    {pre ps post extra : List Shape} {σ τ : Shape}
    (p : Primitive ps σ τ)
    (x : DAG.Term ((pre ++ ps ++ post) ++ extra) σ) :
    DAG.Term ((pre ++ ps ++ post) ++ extra) τ := by
  let Γ : List Shape := (pre ++ ps ++ post) ++ extra
  let op : DAG.PrimOp (ps ++ [σ]) τ := Primitive.toDAGPrimOp (ps := ps) (σ := σ) (τ := τ) p
  let paramsArgs : DAG.Args Γ ps :=
    argsOfFn (Γ := Γ) (ins := ps) (fun i => mkParamTerm (pre := pre) (ps := ps) (post := post)
      (extra := extra) i)
  let args : DAG.Args Γ (ps ++ [σ]) :=
    Args.append1 (Γ := Γ) (ps := ps) (σ := σ) paramsArgs x
  exact (by
    -- Discharge the local `Γ` abbreviation.
    simpa [Γ] using (DAG.Term.op (Γ := Γ) op args))

/-! ### Graph lowering -/

/-- Lower a sequential `Graph` to an SSA-style `DAG.Term`, with parameters read from the
  environment. -/
def toTerm
    {pre ps post extra : List Shape} {σ τ : Shape}
    (g : Graph ps σ τ)
    (x : DAG.Term ((pre ++ ps ++ post) ++ extra) σ) :
    DAG.Term ((pre ++ ps ++ post) ++ extra) τ := by
  let Γ : List Shape := (pre ++ ps ++ post) ++ extra
  match g with
  | .id _ =>
      simpa [Γ] using x
  | .prim p =>
      simpa [Γ] using primCall (pre := pre) (ps := ps) (post := post) (extra := extra) (p := p) x
  | .seq (ps₁ := ps₁) (ps₂ := ps₂) (σ := σ) (τ := τm) (υ := τ) g₁ g₂ =>
      -- Outer env: `(pre ++ (ps₁ ++ ps₂) ++ post) ++ extra`
      let Γ0 : List Shape := (pre ++ (ps₁ ++ ps₂) ++ post) ++ extra
      -- Left subgraph sees post = ps₂ ++ post.
      have hΓ1 :
          (pre ++ ps₁ ++ (ps₂ ++ post)) ++ extra
          =
          Γ0 := by
        simp [Γ0, List.append_assoc]
      let t₁ : DAG.Term Γ0 τm :=
        castEnvTerm (Γ := (pre ++ ps₁ ++ (ps₂ ++ post)) ++ extra) (Γ' := Γ0) hΓ1 <|
          toTerm (pre := pre) (ps := ps₁) (post := ps₂ ++ post) (extra := extra) g₁
            (by
              have x0 : DAG.Term Γ0 σ := by
                simpa [Γ, Γ0] using x
              exact
                castEnvTerm (Γ := Γ0) (Γ' := (pre ++ ps₁ ++ (ps₂ ++ post)) ++ extra) (by
                  simp [Γ0, List.append_assoc]) x0)
      -- `let1`-bind and translate the right subgraph.
      let bodyEnv : List Shape := Γ0 ++ [τm]
      -- Bound var in the body env (the last element, at index `Γ0.length`).
      let boundIdx : Fin bodyEnv.length := ⟨Γ0.length, by simp [bodyEnv, List.length_append]⟩
      let boundVar : DAG.Term bodyEnv τm :=
        -- Align the index used in `get_append_last` with `boundIdx`.
        let idx0 : Fin bodyEnv.length := ⟨Γ0.length, by simp [bodyEnv, List.length_append]⟩
        have hGet0 : bodyEnv.get idx0 = τm := by
          -- Avoid `simp` rewriting `Eq` goals into `True`.
          dsimp [bodyEnv]
          let idxStd : Fin (Γ0 ++ [τm]).length := ⟨Γ0.length, by simp [List.length_append]⟩
          have hidx : idx0 = idxStd := by
            apply Fin.ext
            rfl
          cases hidx
          exact get_append_last (xs := Γ0) (x := τm)
        have hIdx : boundIdx = idx0 := by
          apply Fin.ext
          rfl
        have hGet : bodyEnv.get boundIdx = τm := by
          simpa [hIdx] using hGet0
        castTerm hGet (DAG.Term.var (Γ := bodyEnv) boundIdx)
      -- Translate `g₂` under its own parenthesization, then cast back to `bodyEnv`.
      let rhsEnv : List Shape := ((pre ++ ps₁) ++ ps₂ ++ post) ++ (extra ++ [τm])
      have hRhs : rhsEnv = bodyEnv := by
        simp [rhsEnv, bodyEnv, Γ0, List.append_assoc]
      let boundVar' : DAG.Term rhsEnv τm :=
        castEnvTerm (Γ := bodyEnv) (Γ' := rhsEnv) hRhs.symm boundVar
      let t₂' : DAG.Term rhsEnv τ :=
        toTerm (pre := pre ++ ps₁) (ps := ps₂) (post := post) (extra := extra ++ [τm]) g₂ boundVar'
      let t₂ : DAG.Term bodyEnv τ :=
        castEnvTerm (Γ := rhsEnv) (Γ' := bodyEnv) hRhs t₂'
      let out : DAG.Term Γ0 τ := DAG.Term.let1 t₁ t₂
      -- Discharge the local `Γ` abbreviation.
      simpa [Γ, Γ0] using out

/-! ### Public API -/

/-- Initialize a parameter list by filling every tensor with zeros (useful for proofs and examples). -/
def zeroInitParams : (ps : List Shape) → TList Float ps
  | [] => .nil
  | s :: ss => .cons (Spec.zeros (α := Float) s) (zeroInitParams ss)

/-!
### Deterministic initialization for sequential graphs

`Graph.toDAGModelZeroInit` is total, but its parameters are all-zero tensors, which is convenient
for proofs and shape-only examples but not representative of training setups.

For graphs whose primitives provide `Primitive.toLayerDefM?`, we can reuse TorchLean’s deterministic
initializers (e.g. Xavier init for linear weights) in a way that matches `ToTorchLean.toSeq`:

- we thread an occurrence index `i : Nat`,
- primitives with `countsAsLayer = true` increment it,
- and each primitive’s `LayerDef.initParams` uses seeds derived from `i`.

We expose this as `Graph.toDAGModelDetInit? : Except String (DAG.Model ...)`:
it fails if any primitive lacks a `toLayerDefM?` lowering.
-/

/--
Compute deterministic initialization tensors for a sequential `Graph`, threading a “layer
occurrence index”.

This matches `ToTorchLean.toSeq`’s notion of “occurrence”: only primitives with
`countsAsLayer = true` advance the counter.
-/
def Graph.detInitParamsAux
    {ps : List Shape} {σ τ : Shape}
    (g : Graph ps σ τ) (i : Nat) :
    Except String (Runtime.Autograd.Torch.TList Float ps × Nat) :=
  match g with
  | .id _ => .ok (.nil, i)
  | .prim p =>
      match p.toLayerDefM? with
      | none =>
          .error <|
            (s!"graphspec.detInit: primitive `{p.name}` has no deterministic init " ++
              s!"(missing toLayerDefM?)")
      | some mk =>
          let ⟨l, hps⟩ := mk i
          let i' := if p.countsAsLayer then i + 1 else i
          match hps with
          | rfl => .ok (l.initParams, i')
  | .seq (ps₁ := ps₁) (ps₂ := ps₂) (σ := σ) g₁ g₂ =>
      match Graph.detInitParamsAux (ps := ps₁) (σ := σ) g₁ i with
      | .error e => .error e
      | .ok (xs, i') =>
          match Graph.detInitParamsAux (ps := ps₂) (σ := _) g₂ i' with
          | .error e => .error e
          | .ok (ys, i'') =>
              .ok
                ( Runtime.Autograd.Torch.Proofs.Autograd.Algebra.TList.append
                    (α := Float) (ss₁ := ps₁) (ss₂ := ps₂) xs ys
                , i'')

/-- Convenience wrapper: start the occurrence index at `0` and discard its final value. -/
def Graph.detInitParams?
    {ps : List Shape} {σ τ : Shape}
    (g : Graph ps σ τ) :
    Except String (Runtime.Autograd.Torch.TList Float ps) :=
  match Graph.detInitParamsAux (ps := ps) (σ := σ) (τ := τ) g 0 with
  | .error e => .error e
  | .ok (xs, _i) => .ok xs

/--
Lower a sequential `Graph` to a DAG term with environment `ps ++ [σ]`.
 -/
def Graph.toDAGTerm {ps : List Shape} {σ τ : Shape} (g : Graph ps σ τ) :
    DAG.Term (ps ++ [σ]) τ :=
  let x :
      let Γ : List Shape := ([] ++ ps ++ []) ++ [σ]
      DAG.Term Γ σ := by
    intro Γ
    have hLt : ps.length < Γ.length := by
      simp [Γ, List.length_append]
    let xIdx : Fin Γ.length := ⟨ps.length, hLt⟩
    have hGet0 :
        Γ.get ⟨ps.length, by simp [Γ, List.length_append]⟩ = σ := by
      -- `Γ` is definitional `(([] ++ ps ++ []) ++ [σ])`, so this is the last element.
      simp [Γ]
    have hxIdx : xIdx = ⟨ps.length, by simp [Γ, List.length_append]⟩ := by
      apply Fin.ext
      rfl
    have hGet : Γ.get xIdx = σ := by
      simpa [hxIdx] using hGet0
    exact castTerm hGet (DAG.Term.var (Γ := Γ) xIdx)
  -- `toTerm`’s environment is definitional `([] ++ ps ++ [] ++ [σ])`; normalize to `ps ++ [σ]`.
  by
    simpa [List.nil_append, List.append_nil, List.append_assoc] using
      (toTerm (pre := []) (ps := ps) (post := []) (extra := [σ]) g x)

/--
Lower a sequential `Graph` to a DAG `Model` with a simple default init (all zeros).

This is mainly a convenience for GraphSpec example organization; for training-oriented init,
see `NN.GraphSpec.ToTorchLean` (Seq lowering) and/or provide your own initializer.
 -/
def Graph.toDAGModelZeroInit {ps : List Shape} {σ τ : Shape} (g : Graph ps σ τ) :
    DAG.Model ps [σ] τ :=
  { initParams := zeroInitParams ps
    body := Graph.toDAGTerm (ps := ps) (σ := σ) (τ := τ) g
  }

/--
Lower a sequential `Graph` to a DAG `Model`, using deterministic initialization.

This is the DAG analogue of `ToTorchLean.toSeq`’s initialization semantics: it uses each primitive’s
`toLayerDefM?` to obtain a TorchLean `LayerDef`, then reuses the `LayerDef.initParams`.

This returns `Except String` because not every primitive necessarily admits a `LayerDef` lowering.
-/
def Graph.toDAGModelDetInit?
    {ps : List Shape} {σ τ : Shape} (g : Graph ps σ τ) :
    Except String (DAG.Model ps [σ] τ) :=
  match Graph.detInitParams? (ps := ps) (σ := σ) (τ := τ) g with
  | .error e => .error e
  | .ok params =>
      .ok { initParams := params
            body := Graph.toDAGTerm (ps := ps) (σ := σ) (τ := τ) g }

end LowerToDAG

/-! ## Semantics (sequential core) -/

/-!
The sequential DSL (`Graph` with `>>>`) has *direct* semantics:

- `Interp.spec` evaluates a sequential graph as a pure function on tensors, and
- `Compile.torchProgram` compiles it to a backend-generic `TorchLean.Program`.

Even though a sequential graph is semantically a path-shaped DAG, we keep the sequential
interpreter/compiler direct for two pragmatic reasons:

1. **Proof ergonomics.** For chain graphs, definitional reduction is much simpler when we evaluate
   directly rather than going through an SSA lowering.
2. **Engineering clarity.** The sequential and DAG languages have different invariants (parameter
   concatenation vs explicit `let1` sharing). Keeping each semantics close to its syntax makes the
   code easier to audit.

We still provide a structural lowering `LowerToDAG.Graph.toDAGModelZeroInit` so that DAG-only
  tooling
can consume sequential models. The DAG path becomes the canonical execution route when a caller
wants explicit sharing together with the corresponding simp lemmas / proof infrastructure.
 -/

namespace Interp

/-- A typed list of parameter tensors matching the parameter-shape ABI `ps`. -/
abbrev Params (α : Type 0) (ps : List Shape) : Type :=
  Runtime.Autograd.Torch.TList α ps

/--
Split a typed parameter list for a sequential composition.

If `ps = ps₁ ++ ps₂`, then a value of type `TList α ps` can be split into the prefix parameters for
the left subgraph and the remaining parameters for the right subgraph.
-/
def splitParams {α : Type 0} :
     {ps₁ ps₂ : List Shape} →
       Runtime.Autograd.Torch.TList α (ps₁ ++ ps₂) →
         Runtime.Autograd.Torch.TList α ps₁ × Runtime.Autograd.Torch.TList α ps₂
   | [], _ps₂, xs => (.nil, xs)
   | _s :: ps₁, ps₂, .cons x xs =>
       let (l, r) := splitParams (ps₁ := ps₁) (ps₂ := ps₂) xs
       (.cons x l, r)

/--
Pure Spec semantics of a sequential `Graph`.
 -/
def spec
    {ps : List Shape} {σ τ : Shape}
    (g : Graph ps σ τ)
    {α : Type 0} [Context α] :
    Params α ps → Tensor α σ → Tensor α τ :=
  fun params x =>
    match g with
    | .id _ => x
    | .prim p => p.specFwd (α := α) params x
    | .seq (ps₁ := ps₁) (ps₂ := ps₂) g₁ g₂ =>
        let (params₁, params₂) := splitParams (α := α) (ps₁ := ps₁) (ps₂ := ps₂) params
        let y := spec (α := α) g₁ params₁ x
        spec (α := α) g₂ params₂ y

end Interp

namespace Compile

/--
Compile a sequential `Graph` to a backend-generic TorchLean `Program`.
 -/
def torchProgram
    {ps : List Shape} {σ τ : Shape}
    (g : Graph ps σ τ)
    {α : Type 0} [Context α] [DecidableEq Shape] :
    Runtime.Autograd.TorchLean.Program α (ps ++ [σ]) τ :=
  fun {m} _instM _instOps =>
    let Ref := fun s => Runtime.Autograd.Torch.Ops.Ref (m := m) (α := α) s

    let rec splitParamsRef :
        {ps₁ ps₂ : List Shape} →
          Runtime.Autograd.Torch.RefList Ref (ps₁ ++ ps₂) →
            Runtime.Autograd.Torch.RefList Ref ps₁ × Runtime.Autograd.Torch.RefList Ref ps₂
      | [], _ps₂, xs => (.nil, xs)
      | _s :: ps₁, ps₂, .cons x xs =>
          let (l, r) := splitParamsRef (ps₁ := ps₁) (ps₂ := ps₂) xs
          (.cons x l, r)

    let rec compileRefList
        {ps : List Shape} {σ τ : Shape}
        (g : Graph ps σ τ)
        (rs : Runtime.Autograd.Torch.RefList Ref (ps ++ [σ])) :
        m (Ref τ) :=
      match g with
      | .id _ =>
          match rs with
          | .cons x .nil => pure x
      | .prim p =>
          Runtime.Autograd.Torch.CurriedRef.uncurry (ss := ps ++ [σ]) (Ref := Ref)
            (p.torchProgram (α := α)) rs
      | .seq (ps₁ := ps₁) (ps₂ := ps₂) (τ := τm) g₁ g₂ => do
          let (params12, x) :=
            Runtime.Autograd.Torch.RefList.splitAppend1 (Ref := Ref) (ss := ps₁ ++ ps₂) (τ := σ) rs
          let (params₁, params₂) := splitParamsRef (ps₁ := ps₁) (ps₂ := ps₂) params12
          let rs₁ :=
            Runtime.Autograd.Torch.RefList.append (Ref := Ref) (ss₁ := ps₁) (ss₂ := [σ])
              params₁ (.cons x .nil)
          let y ← compileRefList (ps := ps₁) (σ := σ) (τ := τm) g₁ rs₁
          let rs₂ :=
            Runtime.Autograd.Torch.RefList.append (Ref := Ref) (ss₁ := ps₂) (ss₂ := [τm])
              params₂ (.cons y .nil)
          compileRefList (ps := ps₂) (σ := τm) (τ := τ) g₂ rs₂

    Runtime.Autograd.Torch.CurriedRef.curry (ss := ps ++ [σ]) (Ref := Ref)
      (fun rs => compileRefList (ps := ps) (σ := σ) (τ := τ) g rs)

end Compile

end GraphSpec
end NN
