/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Algebra.Soundness
public import NN.Runtime.Autograd.Engine.Core
public import NN.Runtime.Autograd.TorchLean.Random

/-!
# GraphM

Proof-compiled graph authoring API.

`Proofs.Autograd.Algebra.GraphData` is an executable SSA/DAG graph used by the proof-compiled
pipeline (`Runtime.Autograd.Compiled`). Constructing it directly exposes dependent indices (`Idx`)
into the graph context and is therefore fairly low-level.

This module defines a small `StateT` builder (`GraphM`) that:
- hides `Idx` bookkeeping behind typed variables (`Var s`);
- tracks the growing context automatically;
- emits `GraphData` nodes with runtime shape checks for safety.

## Reading map

- `GraphM.arg` and `GraphM.args` name inputs from the initial context.
- `GraphM.const` and `GraphM.rand_uniform` add constant and deterministic runtime nodes.
- The remaining op builders mirror the `Runtime.Autograd.TorchLean.Backend` surface one-for-one.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace Compiled
namespace GraphM

open Spec
open Tensor
open Proofs.Autograd.Algebra
open Runtime.Autograd.TorchLean

/-- Shorthand for the underlying executable SSA graph type from `Proofs.Autograd.Algebra`. -/
abbrev PGraphData (α : Type) (Δ : Type) (Γ : List Shape) (ss : List Shape) : Type :=
  _root_.Proofs.Autograd.Algebra.GraphData α Δ Γ ss

/-- Executable node payload for the proof-compiled SSA graph (`GraphData`). -/
abbrev PNodeData (α : Type) (Δ : Type) (Γ : List Shape) (s : Shape) : Type :=
  _root_.Proofs.Autograd.Algebra.NodeData α Δ Γ s

/--
A typed handle to a value in the growing compiled context.

`Var s` carries its expected `Shape` at the type level, while `id` is the runtime index into the
concatenated context `Γ ++ ss`.
-/
structure Var (s : Shape) where
  /--
  Runtime id of the value inside the concatenated context `Γ ++ ss`.

  The shape index on `Var s` is the static guarantee; this numeric id is the executable handle used
  when constructing `Idx` proofs for `GraphData` nodes.
  -/
  id : Nat
deriving Repr

/-!
`GraphM.arg` is correct but a little noisy for examples (you must repeat the index and shape).

`VarList` + `args` give a small convenience layer: `args` returns one `Var` per entry in `Γ`,
in order, without spelling indices.
-/

/--
Dependent list of typed variables, aligned with a list of shapes.

`VarList Γ` contains exactly one `Var s` for each `s ∈ Γ`, in order.
-/
inductive VarList : List Shape → Type where
  | nil : VarList []
  | cons {s : Shape} {ss : List Shape} : Var s → VarList ss → VarList (s :: ss)

namespace VarList

/-- First variable in a nonempty `VarList`. -/
def head {s : Shape} {ss : List Shape} : VarList (s :: ss) → Var s
  | cons v _ => v

/-- Tail variables in a nonempty `VarList`. -/
def tail {s : Shape} {ss : List Shape} : VarList (s :: ss) → VarList ss
  | cons _ vs => vs

end VarList

/--
State for the `GraphM` builder.

It is a sigma pair of:
- the list of intermediate shapes `ss` produced so far, and
- the corresponding executable SSA graph payload `GraphData α Δ Γ ss`.
-/
abbrev StateWith (α : Type) (Δ : Type) (Γ : List Shape) : Type :=
  Σ ss : List Shape, PGraphData α Δ Γ ss

/-- Default `GraphM` state with no extra environment (`Δ := Unit`). -/
abbrev State (α : Type) (Γ : List Shape) : Type :=
  StateWith α Unit Γ

/-- `StateT` builder monad for authoring a `GraphData` program, with explicit environment `Δ`. -/
abbrev MWith (α : Type) (Δ : Type) (Γ : List Shape) : Type → Type :=
  StateT (StateWith α Δ Γ) (Runtime.Autograd.Result)

/-- Default `GraphM` builder monad with `Δ := Unit`. -/
abbrev M (α : Type) (Γ : List Shape) : Type → Type :=
  MWith α Unit Γ

/-- Empty builder state (no intermediate nodes yet). -/
def empty {α : Type} {Γ : List Shape} : State α Γ :=
  ⟨[], .nil⟩

/-- Empty builder state for an explicit environment type `Δ`. -/
def emptyWith {α : Type} {Δ : Type} {Γ : List Shape} : StateWith α Δ Γ :=
  ⟨[], .nil⟩

/-- Run a `GraphM` program from an empty state. -/
def run {α : Type} {Γ : List Shape} {β : Type} (m : M α Γ β) :
    Runtime.Autograd.Result (β × State α Γ) :=
  StateT.run m empty

/--
Build a `GraphData` by running a `GraphM` program.

This is the usual entry point: write a `do`-block that constructs the graph using `arg`, ops,
and returns `Unit`; get back the finalized builder state containing `ss` and the graph.
-/
def buildGraph {α : Type} {Γ : List Shape} (m : M α Γ Unit) :
    Runtime.Autograd.Result (State α Γ) := do
  let (_, st) ← run (α := α) (Γ := Γ) m
  pure st

/-- Length of the current context `Γ ++ ss` (inputs + intermediates). -/
def ctxLen {Γ : List Shape} (ss : List Shape) : Nat :=
  (Γ ++ ss).length

/--
Convert a `Var s` into a dependent `Idx (Γ ++ ss) s`.

This performs bounds checking and a runtime shape check, returning a structured error if the
variable points outside the current context or has the wrong shape.
-/
def mkIdx {_α : Type} [DecidableEq Shape] {Γ : List Shape} (ss : List Shape) {s : Shape}
    (v : Var s) : Runtime.Autograd.Result (Idx (Γ ++ ss) s) := by
  let n := v.id
  if h : n < ctxLen (Γ := Γ) ss then
    let i : Fin (ctxLen (Γ := Γ) ss) := ⟨n, h⟩
    let got : Shape := (Γ ++ ss).get i
    if hg : got = s then
      exact .ok ⟨i, hg⟩
    else
      exact .error <|
        s!"compiled GraphM: shape mismatch at id={n}: expected {Shape.pretty s}, " ++
          s!"got {Shape.pretty got}"
  else
    exact .error s!"compiled GraphM: invalid id={n} for ctxLen={ctxLen (Γ := Γ) ss}"

/--
Append a node to the graph state and return a fresh `Var` pointing to its output.

The returned variable id is `Γ.length + ss.length`, i.e. it points at the newly appended entry.
-/
def push {α : Type} {Δ : Type} {Γ : List Shape} {ss : List Shape} {s : Shape}
    (g : PGraphData α Δ Γ ss) (node : PNodeData α Δ (Γ ++ ss) s) : MWith α Δ Γ (Var s) := do
  set (σ := StateWith α Δ Γ) ⟨ss ++ [s], .snoc g node⟩
  pure { id := Γ.length + ss.length }

/-- Forward-mode JVP availability for a compiled graph builder op. -/
inductive JvpAvailability where
  /-- The op supplies a real forward-mode JVP rule. -/
  | implemented
  /-- The op supplies reverse-mode VJP only. Forward-mode requests fail loudly. -/
  | reverseOnly (op : String)
deriving Repr, DecidableEq

/--
Compiled ops that provide VJP for training but no forward-mode JVP rule.

Keeping the list executable gives callers a stable preflight hook instead of discovering the gap
only after a directional-derivative run reaches the node. The list is intentionally empty when all
compiled builder ops have concrete JVP rules.
-/
def reverseOnlyJvpOps : List String :=
  []

/-- Return the JVP status for a named compiled op. -/
def jvpAvailability (op : String) : JvpAvailability :=
  if reverseOnlyJvpOps.any (fun name => name == op) then
    .reverseOnly op
  else
    .implemented

/-- Human-readable message for reverse-only compiled ops. -/
def reverseOnlyJvpMessage (op : String) : String :=
  s!"compiled GraphM: forward-mode JVP requested for op `{op}`, " ++
  "but this compiled node is reverse-mode only. Use VJP/backprop, avoid this op in forward-mode " ++
  "graphs, or add a real JVP rule in `NN/Runtime/Autograd/Compiled/GraphM.lean`."

/--
Fail-fast marker for compiled nodes whose forward-mode JVP rule is intentionally absent.

Returning a zero tangent here would silently corrupt forward-mode autodiff. Reverse-mode users are
unaffected because these nodes still provide real `vjp` implementations. Forward-mode callers get a
loud error, and `reverseOnlyJvpOps` provides a preflight list for tools that want to reject such
graphs before running a JVP.
-/
def unsupportedJvp {α : Type} [Zero α] {s : Shape} (op : String) : Tensor α s :=
  let _ : Inhabited (Tensor α s) := ⟨fill (0 : α) s⟩
  panic! reverseOnlyJvpMessage op

/--
Reference an input variable from the initial context `Γ`.

This checks that the provided index is within bounds and that the requested shape matches the
shape at that position in `Γ`.

PyTorch comparison: this is like naming a graph input tensor in a traced graph.
-/
def arg {α : Type} {Δ : Type} [DecidableEq Shape] {Γ : List Shape} (i : Nat) (s : Shape) :
    MWith α Δ Γ (Var s) := do
  if h : i < Γ.length then
    let fin : Fin Γ.length := ⟨i, h⟩
    let got : Shape := Γ.get fin
    if _hg : got = s then
      pure { id := i }
    else
      throw <|
        s!"compiled GraphM: input shape mismatch at i={i}: expected " ++
          s!"{Shape.pretty s}, got {Shape.pretty got}"
  else
    throw s!"compiled GraphM: input index out of bounds i={i} (Γ.length={Γ.length})"

/-- Pure helper to build `VarList Γ` starting at a given id offset. -/
def argsAux : (Γ : List Shape) → Nat → VarList Γ
  | [], _i => .nil
  | _s :: ss, i => .cons { id := i } (argsAux ss (i + 1))

/--
Return one `Var` per entry of `Γ`, in order.

This is a convenience wrapper around `arg` that avoids manually writing indices in examples.
-/
def args {α : Type} {Δ : Type} {Γ : List Shape} : MWith α Δ Γ (VarList Γ) := do
  pure (argsAux Γ 0)

/--
Embed a constant tensor as a node in the compiled graph.

This node has no input dependencies (`vjp = 0`, `jvp = 0`), i.e. it is treated as a constant
with respect to the graph inputs.

PyTorch comparison: a constant literal captured into a traced/compiled graph.
-/
def const {α : Type} {Δ : Type} [Zero α] {Γ : List Shape} {s : Shape} (t : Tensor α s) :
    MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun _ctx _d => t
      jvp := fun _ctx _dctx _d => fill (0 : α) s
      vjp := fun _ctx _d _δ => TList.zero (α := α) (ss := Γ ++ ss) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/-- Deterministic `U[0,1)` tensor generator (seeded, pure). -/
def randUniform {α : Type} [Context α] {Δ : Type} {Γ : List Shape} {s : Shape} (seed : Nat) :
    MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let counter := ss.length
  let key := TorchLean.Random.keyOf seed counter
  let t : Tensor α s := TorchLean.Random.uniform (α := α) key (s := s)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun _ctx _d => t
      jvp := fun _ctx _dctx _d => fill (0 : α) s
      vjp := fun _ctx _d _δ => TList.zero (α := α) (ss := Γ ++ ss) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/--
Deterministic `{0,1}` mask generator (seeded, pure).

Note: for differentiation purposes, this node is treated as a **stop-gradient** op:
`jvp = 0` and `vjp = 0` for all inputs (including `keepProb`). This matches the intended use in
dropout where the probability is a hyperparameter (not differentiated), while keeping execution
deterministic in the `.compiled` backend.
-/
def bernoulliMask {α : Type} [Context α] [DecidableEq Shape]
    {Δ : Type} {Γ : List Shape} {s : Shape}
    (keepProb : Var Shape.scalar) (seed : Nat) :
    MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let counter := ss.length
  let key := TorchLean.Random.keyOf seed counter
  let ikp ← liftM (mkIdx (_α := α) (Γ := Γ) ss keepProb)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d =>
        let kpT := getIdx (α := α) (xs := ctx) ikp
        let kp : α :=
          match kpT with
          | Tensor.scalar v => v
        TorchLean.Random.mask (α := α) key kp (s := s)
      jvp := fun _ctx _dctx _d => fill (0 : α) s
      vjp := fun _ctx _d _δ => TList.zero (α := α) (ss := Γ ++ ss) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/--
Stop-gradient boundary.

Forward semantics: identity (`detach(x) = x`).
Backward semantics: no gradient flows to `x` (treated as constant w.r.t. the graph inputs).
-/
def detach {α : Type} [Context α] [DecidableEq Shape]
    {Δ : Type} {Γ : List Shape} {s : Shape}
    (x : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d => getIdx (α := α) (xs := ctx) ix
      jvp := fun _ctx _dctx _d => fill (0 : α) s
      vjp := fun _ctx _d _δ => TList.zero (α := α) (ss := Γ ++ ss) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/-!
JVP vs VJP in this module

Each compiled node stores both:
- `vjp`: reverse-mode vector-Jacobian product (used by backprop), and
- `jvp`: forward-mode Jacobian-vector product (directional derivative).

The `.compiled` runtime path is primarily exercised via reverse-mode (VJP) and compilation to the
eager tape. Basic elementwise/bilinear ops provide real JVP rules, shape-structural ops (for
example slice/concat) apply the same transformation to the tangent, and heavier ops should expose
named spec-layer JVP helpers before being wired here. Reverse-only ops
it must be listed in `reverseOnlyJvpOps` and call `unsupportedJvp` rather than returning a silent
zero tangent.

Forward-mode coverage is expanded by adding concrete `jvp` rules next to the corresponding
`forward` and `vjp` definitions.
-/

/--
Elementwise addition node (`y = a + b`).

PyTorch comparison: `torch.add(a, b)`.
-/
def add {α : Type} {Δ : Type} [Add α] [Zero α] [DecidableEq Shape] {Γ : List Shape} {s : Shape}
    (a b : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let ia ← liftM (mkIdx (_α := α) (Γ := Γ) ss a)
  let ib ← liftM (mkIdx (_α := α) (Γ := Γ) ss b)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d => addSpec (getIdx (α := α) (xs := ctx) ia) (getIdx (α := α) (xs :=
      ctx) ib)
      jvp := fun _ctx dctx _d =>
        addSpec (getIdx (α := α) (xs := dctx) ia) (getIdx (α := α) (xs := dctx) ib)
      vjp := fun _ctx _d δ =>
        TList.add (α := α) (ss := Γ ++ ss)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := s) ia δ)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := s) ib δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/--
Elementwise subtraction node (`y = a - b`).

PyTorch comparison: `torch.sub(a, b)`.
-/
def sub {α : Type} {Δ : Type} [Sub α] [Add α] [Zero α] [DecidableEq Shape] {Γ : List Shape} {s :
  Shape}
    (a b : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let ia ← liftM (mkIdx (_α := α) (Γ := Γ) ss a)
  let ib ← liftM (mkIdx (_α := α) (Γ := Γ) ss b)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d => subSpec (getIdx (α := α) (xs := ctx) ia) (getIdx (α := α) (xs :=
      ctx) ib)
      jvp := fun _ctx dctx _d =>
        subSpec (getIdx (α := α) (xs := dctx) ia) (getIdx (α := α) (xs := dctx) ib)
      vjp := fun _ctx _d δ =>
        let negδ : Tensor α s := subSpec (fill (0 : α) s) δ
        TList.add (α := α) (ss := Γ ++ ss)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := s) ia δ)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := s) ib negδ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/--
Elementwise multiplication node (`y = a ⊙ b`).

PyTorch comparison: `torch.mul(a, b)`.
-/
def mul {α : Type} {Δ : Type} [Mul α] [Add α] [Zero α] [DecidableEq Shape] {Γ : List Shape} {s :
  Shape}
    (a b : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let ia ← liftM (mkIdx (_α := α) (Γ := Γ) ss a)
  let ib ← liftM (mkIdx (_α := α) (Γ := Γ) ss b)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d => mulSpec (getIdx (α := α) (xs := ctx) ia) (getIdx (α := α) (xs :=
      ctx) ib)
      jvp := fun ctx dctx _d =>
        let av := getIdx (α := α) (xs := ctx) ia
        let bv := getIdx (α := α) (xs := ctx) ib
        let da := getIdx (α := α) (xs := dctx) ia
        let db := getIdx (α := α) (xs := dctx) ib
        addSpec (mulSpec da bv) (mulSpec av db)
      vjp := fun ctx _d δ =>
        let av := getIdx (α := α) (xs := ctx) ia
        let bv := getIdx (α := α) (xs := ctx) ib
        TList.add (α := α) (ss := Γ ++ ss)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := s) ia (mulSpec δ bv))
          (TList.single (α := α) (Γ := Γ ++ ss) (s := s) ib (mulSpec δ av)) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/-- Square `x ↦ x ⊙ x`. -/
def square {α : Type} {Δ : Type} [Mul α] [Add α] [Zero α] [DecidableEq Shape] {Γ : List Shape} {s :
  Shape}
    (x : Var s) : MWith α Δ Γ (Var s) :=
  mul (α := α) (Δ := Δ) (Γ := Γ) (s := s) x x

/--
Scale a tensor by a scalar constant `c` (`y = c * x`).

PyTorch comparison: `c * x` / `torch.mul(x, c)`.
-/
def scale {α : Type} {Δ : Type} [Mul α] [Add α] [Zero α] [DecidableEq Shape] {Γ : List Shape} {s :
  Shape}
    (x : Var s) (c : α) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d =>
        scaleSpec (α := α) (s := s) (getIdx (α := α) (xs := ctx) ix) c
      jvp := fun _ctx dctx _d =>
        scaleSpec (α := α) (s := s) (getIdx (α := α) (xs := dctx) ix) c
      vjp := fun _ctx _d δ =>
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix (scaleSpec (α := α) (s := s) δ c) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/--
Elementwise absolute value.

PyTorch comparison: `torch.abs(x)`.
-/
def abs {α : Type} [Context α] [Add α] [Zero α] [DecidableEq Shape]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d =>
        absSpec (α := α) (s := s) (getIdx (α := α) (xs := ctx) ix)
      jvp := fun ctx dctx _d =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dx := getIdx (α := α) (xs := dctx) ix
        let dabs := signSpec (α := α) (s := s) xval
        mulSpec dabs dx
      vjp := fun ctx _d δ =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dabs := signSpec (α := α) (s := s) xval
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix (mulSpec dabs δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/--
Elementwise square root.

PyTorch comparison: `torch.sqrt(x)`.
-/
def sqrt {α : Type} [Context α] [Add α] [Zero α] [DecidableEq Shape]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d =>
        sqrtSpec (α := α) (s := s) (getIdx (α := α) (xs := ctx) ix)
      jvp := fun ctx dctx _d =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dx := getIdx (α := α) (xs := dctx) ix
        let dsqrt : Tensor α s :=
          mapSpec (α := α) (s := s) (fun v =>
            if v > 0 then
              (1 : α) / (((2 : Nat) : α) * MathFunctions.sqrt v)
            else
              (0 : α)) xval
        mulSpec dsqrt dx
      vjp := fun ctx _d δ =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dsqrt : Tensor α s :=
          mapSpec (α := α) (s := s) (fun v =>
            if v > 0 then
              (1 : α) / (((2 : Nat) : α) * MathFunctions.sqrt v)
            else
              (0 : α)) xval
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix (mulSpec dsqrt δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/--
Elementwise clamp to `[minVal, maxVal]`.

PyTorch comparison: `torch.clamp(x, min=minVal, max=maxVal)`.
-/
def clamp {α : Type} [Context α] [Add α] [Zero α] [DecidableEq Shape]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) (minVal maxVal : α) : MWith α Δ Γ (Var s) :=
    do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d =>
        clampSpec (α := α) (s := s) (getIdx (α := α) (xs := ctx) ix) minVal maxVal
      jvp := fun ctx dctx _d =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dx := getIdx (α := α) (xs := dctx) ix
        let dclamp : Tensor α s :=
          mapSpec (α := α) (s := s) (fun v =>
            if v > minVal ∧ maxVal > v then (1 : α) else (0 : α)) xval
        mulSpec dclamp dx
      vjp := fun ctx _d δ =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dclamp : Tensor α s :=
          mapSpec (α := α) (s := s) (fun v =>
            if v > minVal ∧ maxVal > v then (1 : α) else (0 : α)) xval
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix (mulSpec dclamp δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/--
Elementwise maximum.

At ties we split the gradient equally (`0.5` / `0.5`), matching the tie-handling documented in
the eager tape (`NN.Runtime.Autograd.Engine.Core`).

PyTorch comparison: `torch.maximum(a, b)`.
-/
def max {α : Type} [Context α] [Add α] [Zero α] [DecidableEq Shape]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {Δ : Type} {Γ : List Shape} {s : Shape} (a b : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let ia ← liftM (mkIdx (_α := α) (Γ := Γ) ss a)
  let ib ← liftM (mkIdx (_α := α) (Γ := Γ) ss b)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d =>
        maxSpec (α := α) (s := s) (getIdx (α := α) (xs := ctx) ia) (getIdx (α := α) (xs := ctx) ib)
      jvp := fun ctx dctx _d =>
        let av := getIdx (α := α) (xs := ctx) ia
        let bv := getIdx (α := α) (xs := ctx) ib
        let da := getIdx (α := α) (xs := dctx) ia
        let db := getIdx (α := α) (xs := dctx) ib
        let half : α := (1 : α) / ((2 : Nat) : α)
        let maskA : Tensor α s :=
          map2Spec (α := α) (β := α) (γ := α) (s := s) (fun x y =>
            if x > y then (1 : α) else if y > x then (0 : α) else half) av bv
        let maskB : Tensor α s :=
          map2Spec (α := α) (β := α) (γ := α) (s := s) (fun x y =>
            if y > x then (1 : α) else if x > y then (0 : α) else half) av bv
        addSpec (mulSpec maskA da) (mulSpec maskB db)
      vjp := fun ctx _d δ =>
        let av := getIdx (α := α) (xs := ctx) ia
        let bv := getIdx (α := α) (xs := ctx) ib
        let half : α := (1 : α) / ((2 : Nat) : α)
        let maskA : Tensor α s :=
          map2Spec (α := α) (β := α) (γ := α) (s := s) (fun x y =>
            if x > y then (1 : α) else if y > x then (0 : α) else half) av bv
        let maskB : Tensor α s :=
          map2Spec (α := α) (β := α) (γ := α) (s := s) (fun x y =>
            if y > x then (1 : α) else if x > y then (0 : α) else half) av bv
        TList.add (α := α) (ss := Γ ++ ss)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := s) ia (mulSpec maskA δ))
          (TList.single (α := α) (Γ := Γ ++ ss) (s := s) ib (mulSpec maskB δ)) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/--
Elementwise minimum.

At ties we split the gradient equally (`0.5` / `0.5`).

PyTorch comparison: `torch.minimum(a, b)`.
-/
def min {α : Type} [Context α] [Add α] [Zero α] [DecidableEq Shape]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {Δ : Type} {Γ : List Shape} {s : Shape} (a b : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let ia ← liftM (mkIdx (_α := α) (Γ := Γ) ss a)
  let ib ← liftM (mkIdx (_α := α) (Γ := Γ) ss b)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d =>
        minSpec (α := α) (s := s) (getIdx (α := α) (xs := ctx) ia) (getIdx (α := α) (xs := ctx) ib)
      jvp := fun ctx dctx _d =>
        let av := getIdx (α := α) (xs := ctx) ia
        let bv := getIdx (α := α) (xs := ctx) ib
        let da := getIdx (α := α) (xs := dctx) ia
        let db := getIdx (α := α) (xs := dctx) ib
        let half : α := (1 : α) / ((2 : Nat) : α)
        let maskA : Tensor α s :=
          map2Spec (α := α) (β := α) (γ := α) (s := s) (fun x y =>
            if y > x then (1 : α) else if x > y then (0 : α) else half) av bv
        let maskB : Tensor α s :=
          map2Spec (α := α) (β := α) (γ := α) (s := s) (fun x y =>
            if x > y then (1 : α) else if y > x then (0 : α) else half) av bv
        addSpec (mulSpec maskA da) (mulSpec maskB db)
      vjp := fun ctx _d δ =>
        let av := getIdx (α := α) (xs := ctx) ia
        let bv := getIdx (α := α) (xs := ctx) ib
        let half : α := (1 : α) / ((2 : Nat) : α)
        let maskA : Tensor α s :=
          map2Spec (α := α) (β := α) (γ := α) (s := s) (fun x y =>
            if y > x then (1 : α) else if x > y then (0 : α) else half) av bv
        let maskB : Tensor α s :=
          map2Spec (α := α) (β := α) (γ := α) (s := s) (fun x y =>
            if x > y then (1 : α) else if y > x then (0 : α) else half) av bv
        TList.add (α := α) (ss := Γ ++ ss)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := s) ia (mulSpec maskA δ))
          (TList.single (α := α) (Γ := Γ ++ ss) (s := s) ib (mulSpec maskB δ)) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/--
Elementwise ReLU.

PyTorch comparison: `torch.nn.functional.relu(x)`.
-/
def relu {α : Type}
  [Mul α] [Add α] [Zero α] [Max α] [One α] [LT α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d =>
        Activation.reluSpec (α := α) (getIdx (α := α) (xs := ctx) ix)
      jvp := fun ctx dctx _d =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dx := getIdx (α := α) (xs := dctx) ix
        let drelu := Activation.reluDerivSpec (α := α) xval
        mulSpec drelu dx
      vjp := fun ctx _d δ =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let drelu := Activation.reluDerivSpec (α := α) xval
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix (mulSpec drelu δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/-- Elementwise sigmoid. PyTorch comparison: `torch.sigmoid(x)`. -/
def sigmoid {α : Type} [Context α] [Add α] [Zero α] [DecidableEq Shape]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d =>
        Activation.sigmoidSpec (α := α) (getIdx (α := α) (xs := ctx) ix)
      jvp := fun ctx dctx _d =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dx := getIdx (α := α) (xs := dctx) ix
        let dsig := Activation.sigmoidDerivSpec (α := α) xval
        mulSpec dsig dx
      vjp := fun ctx _d δ =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dsig := Activation.sigmoidDerivSpec (α := α) xval
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix (mulSpec dsig δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/-- Elementwise tanh. PyTorch comparison: `torch.tanh(x)`. -/
def tanh {α : Type} [Context α] [Add α] [Zero α] [DecidableEq Shape]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d =>
        Activation.tanhSpec (α := α) (getIdx (α := α) (xs := ctx) ix)
      jvp := fun ctx dctx _d =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dx := getIdx (α := α) (xs := dctx) ix
        let dtanh := Activation.tanhDerivSpec (α := α) xval
        mulSpec dtanh dx
      vjp := fun ctx _d δ =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dtanh := Activation.tanhDerivSpec (α := α) xval
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix (mulSpec dtanh δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/--
Softmax along the last axis (recursing over outer dimensions).

PyTorch comparison: `torch.softmax(x, dim=-1)`.
-/
def softmax {α : Type} [Context α] [Add α] [Zero α] [DecidableEq Shape]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d =>
        Activation.softmaxSpec (α := α) (s := s) (getIdx (α := α) (xs := ctx) ix)
      jvp := fun ctx dctx _d =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dx := getIdx (α := α) (xs := dctx) ix
        -- Softmax Jacobian is symmetric, so we can reuse the same JVP/VJP implementation.
        Activation.softmaxBackwardSpec (α := α) (s := s) xval dx
      vjp := fun ctx _d δ =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dx := Activation.softmaxBackwardSpec (α := α) (s := s) xval δ
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix dx }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/--
Stable log-softmax along the last axis.

This is intentionally a primitive in the compiled graph, not the composition
`log ∘ softmax`, so proof/IR execution and eager CUDA share the same PyTorch-style numerical
contract.
-/
def logSoftmax {α : Type} [Context α] [Add α] [Zero α] [DecidableEq Shape]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d =>
        Activation.logSoftmaxSpec (α := α) (s := s) (getIdx (α := α) (xs := ctx) ix)
      jvp := fun ctx dctx _d =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let yval := Activation.logSoftmaxSpec (α := α) (s := s) xval
        let dx := getIdx (α := α) (xs := dctx) ix
        Activation.logSoftmaxBackwardSpec (α := α) (s := s) yval dx
      vjp := fun ctx _d δ =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let yval := Activation.logSoftmaxSpec (α := α) (s := s) xval
        let dx := Activation.logSoftmaxBackwardSpec (α := α) (s := s) yval δ
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix dx }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/-- Elementwise softplus. PyTorch comparison: `torch.nn.functional.softplus(x)`. -/
def softplus {α : Type} [Context α] [Add α] [Zero α] [DecidableEq Shape]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d =>
        Activation.softplusSpec (α := α) (s := s) (getIdx (α := α) (xs := ctx) ix)
      jvp := fun ctx dctx _d =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dx := getIdx (α := α) (xs := dctx) ix
        let dsoft := Activation.softplusDerivSpec (α := α) (s := s) xval
        mulSpec dsoft dx
      vjp := fun ctx _d δ =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dsoft := Activation.softplusDerivSpec (α := α) (s := s) xval
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix (mulSpec dsoft δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/-- Elementwise exponential. PyTorch comparison: `torch.exp(x)`. -/
def exp {α : Type} [Context α] [Add α] [Zero α] [DecidableEq Shape]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d =>
        expSpec (α := α) (s := s) (getIdx (α := α) (xs := ctx) ix)
      jvp := fun ctx dctx _d =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dx := getIdx (α := α) (xs := dctx) ix
        mulSpec (expSpec (α := α) xval) dx
      vjp := fun ctx _d δ =>
        let xval := getIdx (α := α) (xs := ctx) ix
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix (mulSpec (expSpec (α := α) xval) δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/-- Elementwise natural logarithm. PyTorch comparison: `torch.log(x)`. -/
def log {α : Type} [Context α] [Add α] [Zero α] [DecidableEq Shape]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d =>
        let xval := getIdx (α := α) (xs := ctx) ix
        -- Keep runtime behavior consistent with the eager autograd engine:
        -- `log` rejects non-positive inputs; use `safe_log` for epsilon protection.
        if Tensor.allSpec (α := α) (s := s) (fun v => decide (v > (0 : α))) xval then
          logSpec (α := α) (s := s) xval
        else
          panic! "GraphM: log: input contains values <= 0 (or NaN); use `safe_log` if you want epsilon protection"
      jvp := fun ctx dctx _d =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dx := getIdx (α := α) (xs := dctx) ix
        mulSpec (invSpec (α := α) xval) dx
      vjp := fun ctx _d δ =>
        let xval := getIdx (α := α) (xs := ctx) ix
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix (mulSpec (invSpec (α := α) xval) δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/-- Elementwise reciprocal `x ↦ 1/x`. PyTorch comparison: `torch.reciprocal(x)`. -/
def inv {α : Type} [Context α] [Mul α] [Add α] [Zero α] [DecidableEq Shape]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d =>
        invSpec (α := α) (s := s) (getIdx (α := α) (xs := ctx) ix)
      jvp := fun ctx dctx _d =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dx0 := getIdx (α := α) (xs := dctx) ix
        let invx := invSpec (α := α) xval
        let invx2 := mulSpec invx invx
        scaleSpec (α := α) (s := s) (mulSpec dx0 invx2) (-1 : α)
      vjp := fun ctx _d δ =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let invx := invSpec (α := α) xval
        let invx2 := mulSpec invx invx
        let dx := scaleSpec (α := α) (s := s) (mulSpec δ invx2) (-1 : α)
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix dx }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/--
Elementwise numerically-stable log (uses an internal `ε`).

PyTorch comparison: commonly written `torch.log(x + eps)`.
-/
def safeLog {α : Type} [Context α] [Add α] [Zero α] [DecidableEq Shape]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) (ε : α := Numbers.epsilon) : MWith α Δ Γ (Var
    s) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d =>
        let xval := getIdx (α := α) (xs := ctx) ix
        Activation.safeLogSpec (α := α) (s := s) xval ε
      jvp := fun ctx dctx _d =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dx := getIdx (α := α) (xs := dctx) ix
        let dlog := Activation.safeLogDerivSpec (α := α) (s := s) xval ε
        mulSpec dlog dx
      vjp := fun ctx _d δ =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dlog := Activation.safeLogDerivSpec (α := α) (s := s) xval ε
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix (mulSpec dlog δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/--
Reduce-sum over all entries, producing a scalar.

PyTorch comparison: `torch.sum(x)`.
-/
def sum {α : Type} [Add α] [Zero α] [DecidableEq Shape]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var Shape.scalar) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) Shape.scalar :=
    { forward := fun ctx _d => Tensor.scalar (sumSpec (α := α) (s := s) (getIdx (α := α) (xs :=
      ctx) ix))
      jvp := fun _ctx dctx _d =>
        Tensor.scalar (sumSpec (α := α) (s := s) (getIdx (α := α) (xs := dctx) ix))
      vjp := fun _ctx _d dLdy =>
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix (replicate (α := α) (s := s) dLdy) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := Shape.scalar) g node

/--
Mean-squared error loss with `"mean"` reduction, producing a scalar.

PyTorch comparison: `torch.nn.functional.mse_loss(yhat, target, reduction=\"mean\")`.
-/
def mseLoss {α : Type}
  [Add α] [Sub α] [Mul α] [Div α] [Zero α] [One α] [Coe Nat α] [DecidableEq Shape]
  {Δ : Type} {Γ : List Shape} {s : Shape} (yhat target : Var s) : MWith α Δ Γ (Var Shape.scalar) :=
    do
  let ⟨ss, g⟩ ← get
  let iyhat ← liftM (mkIdx (_α := α) (Γ := Γ) ss yhat)
  let itarget ← liftM (mkIdx (_α := α) (Γ := Γ) ss target)
  let node : NodeData α Δ (Γ ++ ss) Shape.scalar :=
    { forward := fun ctx _d =>
        let yhatv := getIdx (α := α) (xs := ctx) iyhat
        let targetv := getIdx (α := α) (xs := ctx) itarget
        let diff := subSpec yhatv targetv
        let squared := mulSpec diff diff
        let total := sumSpec (α := α) (s := s) squared
        Tensor.scalar (total / (Shape.size s : α))
      jvp := fun ctx dctx _d =>
        let yhatv := getIdx (α := α) (xs := ctx) iyhat
        let targetv := getIdx (α := α) (xs := ctx) itarget
        let dyhat := getIdx (α := α) (xs := dctx) iyhat
        let dtarget := getIdx (α := α) (xs := dctx) itarget
        let diff := subSpec yhatv targetv
        let two : α := (1 : α) + 1
        let baseGrad : Tensor α s := scaleSpec (α := α) (s := s) diff (two / (Shape.size s : α))
        let ddiff := subSpec dyhat dtarget
        Tensor.scalar (sumSpec (α := α) (s := s) (mulSpec baseGrad ddiff))
      vjp := fun ctx _d dLdy =>
        let yhatv := getIdx (α := α) (xs := ctx) iyhat
        let targetv := getIdx (α := α) (xs := ctx) itarget
        let diff := subSpec yhatv targetv
        let two : α := (1 : α) + 1
        let baseGrad : Tensor α s := scaleSpec (α := α) (s := s) diff (two / (Shape.size s : α))
        let gscalar : α := Tensor.toScalar dLdy
        let dYhat : Tensor α s := scaleSpec (α := α) (s := s) baseGrad gscalar
        let dTarget : Tensor α s := subSpec (fill (0 : α) s) dYhat
        TList.add (α := α) (ss := Γ ++ ss)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := s) iyhat dYhat)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := s) itarget dTarget) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := Shape.scalar) g node

/--
  Affine layer `y = W x + b` in the compiled graph.

  PyTorch comparison: `torch.nn.functional.linear` / `torch.nn.Linear`.

  The JVP is the usual product rule:
  `d(Wx+b) = dW*x + W*dx + db`.
  -/
  def linear {α : Type} {Δ : Type} [Add α] [Mul α] [Zero α] [DecidableEq Shape]
    {Γ : List Shape} {inDim outDim : Nat}
    (w : Var (.dim outDim (.dim inDim .scalar)))
    (b : Var (.dim outDim .scalar))
    (x : Var (.dim inDim .scalar)) : MWith α Δ Γ (Var (.dim outDim .scalar)) := do
  let ⟨ss, g⟩ ← get
  let iW ← liftM (mkIdx (_α := α) (Γ := Γ) ss w)
  let ib ← liftM (mkIdx (_α := α) (Γ := Γ) ss b)
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) (.dim outDim .scalar) :=
    { forward := fun ctx _d =>
        let W := getIdx (α := α) (xs := ctx) iW
        let bv := getIdx (α := α) (xs := ctx) ib
        let xv := getIdx (α := α) (xs := ctx) ix
        let layer : Spec.LinearSpec α inDim outDim := { weights := W, bias := bv }
        Spec.linearSpec (α := α) layer xv
      jvp := fun ctx dctx _d =>
        let W := getIdx (α := α) (xs := ctx) iW
        let xv := getIdx (α := α) (xs := ctx) ix
        let dW := getIdx (α := α) (xs := dctx) iW
        let db := getIdx (α := α) (xs := dctx) ib
        let dx := getIdx (α := α) (xs := dctx) ix
        let dLayer : Spec.LinearSpec α inDim outDim := { weights := dW, bias := db }
        let xLayer : Spec.LinearSpec α inDim outDim := { weights := W, bias := fill (0 : α) (.dim outDim .scalar) }
        addSpec (Spec.linearSpec (α := α) dLayer xv) (Spec.linearSpec (α := α) xLayer dx)
      vjp := fun ctx _d dLdy =>
        let W := getIdx (α := α) (xs := ctx) iW
        let xv := getIdx (α := α) (xs := ctx) ix
        let dW := Spec.linearWeightsDerivSpec (α := α) (inDim := inDim) (outDim := outDim) xv
          dLdy
        let db := Spec.linearBiasDerivSpec (α := α) (inDim := inDim) (outDim := outDim) dW dLdy
          xv
        let dx := Spec.linearInputDerivSpec (α := α) (inDim := inDim) (outDim := outDim) W dLdy
        let z0 := TList.add (α := α) (ss := Γ ++ ss)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim outDim (.dim inDim .scalar)) iW dW)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim outDim .scalar) ib db)
        TList.add (α := α) (ss := Γ ++ ss) z0
          (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim inDim .scalar) ix dx) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := (.dim outDim .scalar)) g node

/--
  Matrix multiplication (`(m×n) @ (n×p) → (m×p)`).

  PyTorch comparison: `torch.matmul`.

  The JVP is the bilinear product rule `d(A @ B) = dA @ B + A @ dB`.
  -/
  def matmul {α : Type} {Δ : Type} [Context α] [Add α] [Zero α]
    [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
    {Γ : List Shape} {m n p : Nat}
    (a : Var (.dim m (.dim n .scalar))) (b : Var (.dim n (.dim p .scalar))) :
    MWith α Δ Γ (Var (.dim m (.dim p .scalar))) := do
  let ⟨ss, g⟩ ← get
  let ia ← liftM (mkIdx (_α := α) (Γ := Γ) ss a)
  let ib ← liftM (mkIdx (_α := α) (Γ := Γ) ss b)
  let node : NodeData α Δ (Γ ++ ss) (.dim m (.dim p .scalar)) :=
    { forward := fun ctx _d =>
        let av := getIdx (α := α) (xs := ctx) ia
        let bv := getIdx (α := α) (xs := ctx) ib
        Spec.matMulSpec av bv
      jvp := fun ctx dctx _d =>
        let av := getIdx (α := α) (xs := ctx) ia
        let bv := getIdx (α := α) (xs := ctx) ib
        let da := getIdx (α := α) (xs := dctx) ia
        let db := getIdx (α := α) (xs := dctx) ib
        addSpec (Spec.matMulSpec da bv) (Spec.matMulSpec av db)
      vjp := fun ctx _d dLdy =>
        let av := getIdx (α := α) (xs := ctx) ia
        let bv := getIdx (α := α) (xs := ctx) ib
        let (dA, dB) := Spec.Tensor.matMulBackwardSpec av bv dLdy
        TList.add (α := α) (ss := Γ ++ ss)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim m (.dim n .scalar)) ia dA)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim n (.dim p .scalar)) ib dB) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := (.dim m (.dim p .scalar))) g node

/--
  Batched matrix multiplication (`batch×m×n` with `batch×n×p`).

  PyTorch comparison: `torch.bmm`.

  The JVP is the batched bilinear product rule `d(A @ B) = dA @ B + A @ dB`.
  -/
  def bmm {α : Type} {Δ : Type} [Add α] [Mul α] [Zero α] [DecidableEq Shape]
    {Γ : List Shape} {batch m n p : Nat}
    (a : Var (.dim batch (.dim m (.dim n .scalar))))
    (b : Var (.dim batch (.dim n (.dim p .scalar)))) :
    MWith α Δ Γ (Var (.dim batch (.dim m (.dim p .scalar)))) := do
  let ⟨ss, g⟩ ← get
  let ia ← liftM (mkIdx (_α := α) (Γ := Γ) ss a)
  let ib ← liftM (mkIdx (_α := α) (Γ := Γ) ss b)
  let outS : Shape := .dim batch (.dim m (.dim p .scalar))
  let aS : Shape := .dim batch (.dim m (.dim n .scalar))
  let bS : Shape := .dim batch (.dim n (.dim p .scalar))
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        let av := getIdx (α := α) (xs := ctx) ia
        let bv := getIdx (α := α) (xs := ctx) ib
        Spec.Tensor.bmmSpec (α := α) (batch := batch) (m := m) (n := n) (p := p) av bv
      jvp := fun ctx dctx _d =>
        let av := getIdx (α := α) (xs := ctx) ia
        let bv := getIdx (α := α) (xs := ctx) ib
        let da := getIdx (α := α) (xs := dctx) ia
        let db := getIdx (α := α) (xs := dctx) ib
        addSpec
          (Spec.Tensor.bmmSpec (α := α) (batch := batch) (m := m) (n := n) (p := p) da bv)
          (Spec.Tensor.bmmSpec (α := α) (batch := batch) (m := m) (n := n) (p := p) av db)
      vjp := fun ctx _d dLdy =>
        let av := getIdx (α := α) (xs := ctx) ia
        let bv := getIdx (α := α) (xs := ctx) ib
        let (dA, dB) :=
          Spec.Tensor.bmmBackwardSpec (α := α) (batch := batch) (m := m) (n := n) (p := p) av bv
            dLdy
        TList.add (α := α) (ss := Γ ++ ss)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := aS) ia dA)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := bS) ib dB) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
  Concatenate two vectors (dim-0 concat).

  PyTorch comparison: `torch.cat([a, b], dim=0)` for 1D tensors.
  -/
  def concatVectors {α : Type} {Δ : Type} [Context α] [Add α] [Zero α]
    [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
    {Γ : List Shape} {n m : Nat}
    (a : Var (.dim n .scalar)) (b : Var (.dim m .scalar)) :
    MWith α Δ Γ (Var (.dim (n + m) .scalar)) := do
  let ⟨ss, g⟩ ← get
  let ia ← liftM (mkIdx (_α := α) (Γ := Γ) ss a)
  let ib ← liftM (mkIdx (_α := α) (Γ := Γ) ss b)
  let node : NodeData α Δ (Γ ++ ss) (.dim (n + m) .scalar) :=
      { forward := fun ctx _d =>
          let av := getIdx (α := α) (xs := ctx) ia
          let bv := getIdx (α := α) (xs := ctx) ib
          Spec.Tensor.concatVectorsSpec av bv
        jvp := fun _ctx dctx _d =>
          let da := getIdx (α := α) (xs := dctx) ia
          let db := getIdx (α := α) (xs := dctx) ib
          Spec.Tensor.concatVectorsSpec da db
        vjp := fun _ctx _d dLdy =>
          let dA := Spec.Tensor.sliceVectorSpec dLdy 0 n (by simp)
          let dB := Spec.Tensor.sliceVectorSpec dLdy n m (by exact Nat.le_refl _)
          TList.add (α := α) (ss := Γ ++ ss)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim n .scalar) ia dA)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim m .scalar) ib dB) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := (.dim (n + m) .scalar)) g node

/--
  Concatenate along the leading dimension (`dim=0`) for tensors of shape `.dim n s`.

  PyTorch comparison: `torch.cat([a, b], dim=0)`.
  -/
  def concatDim0 {α : Type} {Δ : Type} [Add α] [Zero α] [DecidableEq Shape]
    {Γ : List Shape} {n m : Nat} {s : Shape}
    (a : Var (.dim n s)) (b : Var (.dim m s)) :
    MWith α Δ Γ (Var (.dim (n + m) s)) := do
  let ⟨ss, g⟩ ← get
  let ia ← liftM (mkIdx (_α := α) (Γ := Γ) ss a)
  let ib ← liftM (mkIdx (_α := α) (Γ := Γ) ss b)
  let outS : Shape := .dim (n + m) s
  let aS : Shape := .dim n s
  let bS : Shape := .dim m s
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        let av := getIdx (α := α) (xs := ctx) ia
        let bv := getIdx (α := α) (xs := ctx) ib
        Spec.Tensor.concatDim0Spec (α := α) (n := n) (m := m) (s := s) av bv
      jvp := fun _ctx dctx _d =>
        let da := getIdx (α := α) (xs := dctx) ia
        let db := getIdx (α := α) (xs := dctx) ib
        Spec.Tensor.concatDim0Spec (α := α) (n := n) (m := m) (s := s) da db
      vjp := fun _ctx _d dLdy =>
        let dA := Spec.sliceRangeSpec (α := α) (n := n + m) (s := s) dLdy 0 n
          (by simp)
        let dB := Spec.sliceRangeSpec (α := α) (n := n + m) (s := s) dLdy n m
          (by simp [Nat.add_comm])
        TList.add (α := α) (ss := Γ ++ ss)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := aS) ia dA)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := bS) ib dB) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
  Slice a contiguous range along `dim=0`.

  PyTorch comparison: `x[start : start+len]` for tensors where the leading dimension is indexed.
  -/
  def sliceRange0 {α : Type} {Δ : Type} [Zero α] [DecidableEq Shape]
    {Γ : List Shape} {n : Nat} {s : Shape}
    (x : Var (.dim n s)) (start len : Nat) (h : len + start ≤ n) :
    MWith α Δ Γ (Var (.dim len s)) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let outS : Shape := .dim len s
  let inS : Shape := .dim n s
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        Spec.sliceRangeSpec (α := α) (n := n) (s := s) (getIdx (α := α) (xs := ctx) ix) start len
          h
      jvp := fun _ctx dctx _d =>
        let dx := getIdx (α := α) (xs := dctx) ix
        Spec.sliceRangeSpec (α := α) (n := n) (s := s) dx start len h
      vjp := fun _ctx _d δ =>
        TList.single (α := α) (Γ := Γ ++ ss) (s := inS) ix
          (Spec.Tensor.sliceRange0BackwardSpec (α := α) (n := n) (s := s) start len h δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
N-D max pooling (channels-first) on a single sample tensor (no batch axis).

PyTorch comparison: `torch.nn.functional.max_pool1d` / `max_pool2d` / `max_pool3d` depending on
the spatial rank `d`.

Forward-mode status: implemented. The JVP follows the primal argmax selected by
`Spec.maxPoolJvpSpec`, including the documented first-winner tie convention.
-/
def maxPool {α : Type} {Δ : Type} [Context α] [Add α] [Zero α] [DecidableEq Shape]
  {Γ : List Shape} {d C : Nat}
  {inSpatial kernel stride padding : Vector Nat d}
  {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (x : Var (Shape.ofList (C :: inSpatial.toList))) :
  MWith α Δ Γ (Var (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  if hStride : (∀ i : Fin d, stride.get i ≠ 0) then
    let layer : Spec.MaxPoolSpec d kernel stride padding hKernel hStride := {}
    let outSpatial := Spec.poolOutSpatialPad inSpatial kernel stride padding
    let outShape : Shape := Shape.ofList (C :: outSpatial.toList)
    let inShape : Shape := Shape.ofList (C :: inSpatial.toList)
    let node : NodeData α Δ (Γ ++ ss) outShape :=
      { forward := fun ctx _d =>
          let xv := getIdx (α := α) (xs := ctx) ix
          Spec.maxPoolSpec (α := α) (d := d) (C := C)
            (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
            (layer := layer) xv
        jvp := fun ctx dctx _d =>
          let xv := getIdx (α := α) (xs := ctx) ix
          let dx := getIdx (α := α) (xs := dctx) ix
          Spec.maxPoolJvpSpec (α := α) (d := d) (C := C)
            (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
            (layer := layer) xv dx
        vjp := fun ctx _d δ =>
          let xv := getIdx (α := α) (xs := ctx) ix
          let dx :=
            Spec.maxPoolBackwardSpec (α := α) (d := d) (C := C)
              (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
              (layer := layer) (input := xv) (grad_output := δ)
          TList.single (α := α) (Γ := Γ ++ ss) (s := inShape) ix dx }
    push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outShape) g node
  else
    throw "compiled GraphM: max_pool requires stride > 0 on every spatial axis"

/--
N-D average pooling (channels-first) on a single sample tensor (no batch axis).

PyTorch comparison: `torch.nn.functional.avg_pool1d` / `avg_pool2d` / `avg_pool3d` depending on
the spatial rank `d`.

  Forward-mode status: implemented. Average pooling is linear, so the JVP is the same average-pool
  map applied to the input tangent.
-/
def avgPool {α : Type} {Δ : Type} [Context α] [Add α] [Zero α] [DecidableEq Shape]
  {Γ : List Shape} {d C : Nat}
  {inSpatial kernel stride padding : Vector Nat d}
  (hKernel : ∀ i : Fin d, kernel.get i ≠ 0)
  (x : Var (Shape.ofList (C :: inSpatial.toList))) :
  MWith α Δ Γ (Var (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  if hStride : (∀ i : Fin d, stride.get i ≠ 0) then
    let layer : Spec.AvgPoolSpec d kernel stride padding hKernel hStride := {}
    let outSpatial := Spec.poolOutSpatialPad inSpatial kernel stride padding
    let outShape : Shape := Shape.ofList (C :: outSpatial.toList)
    let inShape : Shape := Shape.ofList (C :: inSpatial.toList)
    let node : NodeData α Δ (Γ ++ ss) outShape :=
      { forward := fun ctx _d =>
          let xv := getIdx (α := α) (xs := ctx) ix
          Spec.avgPoolSpec (α := α) (d := d) (C := C)
            (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
            (layer := layer) xv
        jvp := fun _ctx dctx _d =>
          let dx := getIdx (α := α) (xs := dctx) ix
          Spec.avgPoolSpec (α := α) (d := d) (C := C)
            (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
            (layer := layer) dx
        vjp := fun _ctx _d δ =>
          let dx :=
            Spec.avgPoolBackwardSpec (α := α) (d := d) (C := C)
              (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
              (layer := layer) (grad_output := δ)
          TList.single (α := α) (Γ := Γ ++ ss) (s := inShape) ix dx }
    push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outShape) g node
  else
    throw "compiled GraphM: avg_pool requires stride > 0 on every spatial axis"

/--
N-D smooth max pooling (log-sum-exp surrogate) on a single sample tensor (no batch axis).

PyTorch comparison: there is no direct primitive; this is a differentiable approximation to
max pooling.

Forward-mode status: implemented. The JVP is the softmax-weighted tangent of the
log-sum-exp pooling window.
-/
def smoothMaxPool {α : Type} {Δ : Type} [Context α] [Add α] [Zero α] [DecidableEq Shape]
  {Γ : List Shape} {d C : Nat}
  {inSpatial kernel stride padding : Vector Nat d}
  {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (x : Var (Shape.ofList (C :: inSpatial.toList))) (beta : α) :
  MWith α Δ Γ (Var (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  if hStride : (∀ i : Fin d, stride.get i ≠ 0) then
    let layer : Spec.MaxPoolSpec d kernel stride padding hKernel hStride := {}
    let outSpatial := Spec.poolOutSpatialPad inSpatial kernel stride padding
    let outShape : Shape := Shape.ofList (C :: outSpatial.toList)
    let inShape : Shape := Shape.ofList (C :: inSpatial.toList)
    let node : NodeData α Δ (Γ ++ ss) outShape :=
      { forward := fun ctx _d =>
          let xv := getIdx (α := α) (xs := ctx) ix
          Spec.smoothMaxPoolSpec (α := α) (d := d) (C := C)
            (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
            (layer := layer) (beta := beta) xv
        jvp := fun ctx dctx _d =>
          let xv := getIdx (α := α) (xs := ctx) ix
          let dx := getIdx (α := α) (xs := dctx) ix
          Spec.smoothMaxPoolJvpSpec (α := α) (d := d) (C := C)
            (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
            (layer := layer) (beta := beta) xv dx
        vjp := fun ctx _d δ =>
          let xv := getIdx (α := α) (xs := ctx) ix
          let dx :=
            Spec.smoothMaxPoolBackwardSpec (α := α) (d := d) (C := C)
              (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
              (layer := layer) (beta := beta) (input := xv) (grad_output := δ)
          TList.single (α := α) (Γ := Γ ++ ss) (s := inShape) ix dx }
    push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outShape) g node
  else
    throw "compiled GraphM: smooth_max_pool requires stride > 0 on every spatial axis"

/--
2D max-pooling (channel-first) on a single image tensor.

PyTorch comparison: `torch.nn.functional.max_pool2d` (without a batch dimension).

Forward-mode status: implemented. The JVP routes each output tangent through the
argmax selected by the primal input.
-/

def maxPool2d {α : Type} {Δ : Type} [Context α] [Add α] [Zero α] [DecidableEq Shape]
  {Γ : List Shape} {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (x : Var (.dim inC (.dim inH (.dim inW .scalar)))) :
  MWith α Δ Γ (Var (.dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1)
    .scalar)))) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  if hStride : stride ≠ 0 then
    let layer : Spec.MaxPool2DSpec kH kW stride h1 h2 hStride := {}
    let outH := (inH - kH) / stride + 1
    let outW := (inW - kW) / stride + 1
    let outShape : Shape := .dim inC (.dim outH (.dim outW .scalar))
    let inShape : Shape := .dim inC (.dim inH (.dim inW .scalar))
    let node : NodeData α Δ (Γ ++ ss) outShape :=
      { forward := fun ctx _d =>
          let xv := getIdx (α := α) (xs := ctx) ix
          Spec.maxPool2dMultiSpec (layer := layer) xv
        jvp := fun ctx dctx _d =>
          let xv := getIdx (α := α) (xs := ctx) ix
          let dx := getIdx (α := α) (xs := dctx) ix
          Spec.maxPool2dMultiJvpSpec (layer := layer) (input := xv) (tangent := dx)
        vjp := fun ctx _d δ =>
          let xv := getIdx (α := α) (xs := ctx) ix
          let dx :=
            Tensor.dim (fun c =>
              Spec.maxPool2dBackwardSpec (α := α) (_layer := layer)
                (input := getAtSpec xv c) (grad_output := getAtSpec δ c))
          TList.single (α := α) (Γ := Γ ++ ss) (s := inShape) ix dx }
    push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outShape) g node
  else
    throw "compiled GraphM: max_pool2d requires stride > 0"

/--
2D max-pooling with explicit padding.

PyTorch comparison: `torch.nn.functional.max_pool2d` with padding.

Forward-mode status: implemented. Padding is fixed and the JVP follows the real primal winner,
ignoring padded cells just like the forward pass.
-/
def maxPool2dPad {α : Type} {Δ : Type} [Context α] [Add α] [Zero α] [DecidableEq Shape]
  {Γ : List Shape} {kH kW inH inW inC stride padding : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (x : Var (.dim inC (.dim inH (.dim inW .scalar)))) :
  MWith α Δ Γ
    (Var (.dim inC (.dim ((inH + 2 * padding - kH) / stride + 1) (.dim ((inW + 2 * padding - kW) /
      stride + 1) .scalar)))) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  if hStride : stride ≠ 0 then
    let layer : Spec.MaxPool2DSpec kH kW stride h1 h2 hStride := {}
    let outH := (inH + 2 * padding - kH) / stride + 1
    let outW := (inW + 2 * padding - kW) / stride + 1
    let outShape : Shape := .dim inC (.dim outH (.dim outW .scalar))
    let inShape : Shape := .dim inC (.dim inH (.dim inW .scalar))
    let node : NodeData α Δ (Γ ++ ss) outShape :=
      { forward := fun ctx _d =>
          let xv := getIdx (α := α) (xs := ctx) ix
          Spec.maxPool2dMultiSpecPad (layer := layer) (padding := padding) xv
        jvp := fun ctx dctx _d =>
          let xv := getIdx (α := α) (xs := ctx) ix
          let dx := getIdx (α := α) (xs := dctx) ix
          Spec.maxPool2dMultiJvpSpecPad (layer := layer) (padding := padding)
            (input := xv) (tangent := dx)
        vjp := fun ctx _d δ =>
          let xv := getIdx (α := α) (xs := ctx) ix
          let dx :=
            Spec.maxPool2dMultiBackwardSpecPad (layer := layer) (padding := padding) xv δ
          TList.single (α := α) (Γ := Γ ++ ss) (s := inShape) ix dx }
    push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outShape) g node
  else
    throw "compiled GraphM: max_pool2d_pad requires stride > 0"

/--
Smooth (soft) max-pooling, controlled by `beta`.

This is a differentiable approximation to max-pooling.

Forward-mode status: implemented. The JVP is the softmax-weighted tangent of the
log-sum-exp pooling window.
-/
def smoothMaxPool2d {α : Type} {Δ : Type} [Context α] [Add α] [Zero α] [DecidableEq Shape]
  {Γ : List Shape} {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (x : Var (.dim inC (.dim inH (.dim inW .scalar)))) (beta : α) :
  MWith α Δ Γ (Var (.dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1)
    .scalar)))) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  if hStride : stride ≠ 0 then
    let layer : Spec.MaxPool2DSpec kH kW stride h1 h2 hStride := {}
    let outH := (inH - kH) / stride + 1
    let outW := (inW - kW) / stride + 1
    let outShape : Shape := .dim inC (.dim outH (.dim outW .scalar))
    let inShape : Shape := .dim inC (.dim inH (.dim inW .scalar))
    let node : NodeData α Δ (Γ ++ ss) outShape :=
      { forward := fun ctx _d =>
          let xv := getIdx (α := α) (xs := ctx) ix
          Spec.smoothMaxPool2dMultiSpec (layer := layer) (beta := beta) xv
        jvp := fun ctx dctx _d =>
          let xv := getIdx (α := α) (xs := ctx) ix
          let dx := getIdx (α := α) (xs := dctx) ix
          Spec.smoothMaxPool2dMultiJvpSpec (layer := layer) (beta := beta)
            (input := xv) (tangent := dx)
        vjp := fun ctx _d δ =>
          let xv := getIdx (α := α) (xs := ctx) ix
          let dx :=
            Spec.smoothMaxPool2dMultiBackwardSpec (layer := layer) (beta := beta) xv δ
          TList.single (α := α) (Γ := Γ ++ ss) (s := inShape) ix dx }
    push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outShape) g node
  else
    throw "compiled GraphM: smooth_max_pool2d requires stride > 0"

/--
Average pooling (channel-first) on a single image tensor.

PyTorch comparison: `torch.nn.functional.avg_pool2d` (without a batch dimension).

Forward-mode status: implemented. Average pooling is linear, so the JVP is average pooling of the
input tangent.
-/
def avgPool2d {α : Type} {Δ : Type} [Context α] [Add α] [Zero α] [DecidableEq Shape]
  {Γ : List Shape} {kH kW inH inW inC stride : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0)
  (x : Var (.dim inC (.dim inH (.dim inW .scalar)))) :
  MWith α Δ Γ (Var (.dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1)
    .scalar)))) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  if hStride : stride ≠ 0 then
    let layer : Spec.AvgPool2DSpec kH kW stride h1 h2 hStride := {}
    let outH := (inH - kH) / stride + 1
    let outW := (inW - kW) / stride + 1
    let outShape : Shape := .dim inC (.dim outH (.dim outW .scalar))
    let inShape : Shape := .dim inC (.dim inH (.dim inW .scalar))
    let node : NodeData α Δ (Γ ++ ss) outShape :=
      { forward := fun ctx _d =>
          let xv := getIdx (α := α) (xs := ctx) ix
          Spec.avgPool2dMultiSpec (h1 := h1) (h2 := h2) (layer := layer) xv
        jvp := fun _ctx dctx _d =>
          let dx := getIdx (α := α) (xs := dctx) ix
          Spec.avgPool2dMultiSpec (h1 := h1) (h2 := h2) (layer := layer) dx
        vjp := fun _ctx _d δ =>
          let dx :=
            Tensor.dim (fun c =>
              Spec.avgPool2dBackwardSpec (α := α) (_h1 := h1) (_h2 := h2) (_layer := layer)
                (grad_output := getAtSpec δ c))
          TList.single (α := α) (Γ := Γ ++ ss) (s := inShape) ix dx }
    push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outShape) g node
  else
    throw "compiled GraphM: avg_pool2d requires stride > 0"

/--
Average pooling with explicit padding.

PyTorch comparison: `torch.nn.functional.avg_pool2d` with padding.

  Forward-mode status: implemented. Padding is fixed and average pooling is linear, so the JVP is
  the padded average-pool map applied to the input tangent.
-/
def avgPool2dPad {α : Type} {Δ : Type} [Context α] [Add α] [Zero α] [DecidableEq Shape]
  {Γ : List Shape} {kH kW inH inW inC stride padding : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0)
  (x : Var (.dim inC (.dim inH (.dim inW .scalar)))) :
  MWith α Δ Γ
    (Var (.dim inC (.dim ((inH + 2 * padding - kH) / stride + 1) (.dim ((inW + 2 * padding - kW) /
      stride + 1) .scalar)))) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  if hStride : stride ≠ 0 then
    let layer : Spec.AvgPool2DSpec kH kW stride h1 h2 hStride := {}
    let outH := (inH + 2 * padding - kH) / stride + 1
    let outW := (inW + 2 * padding - kW) / stride + 1
    let outShape : Shape := .dim inC (.dim outH (.dim outW .scalar))
    let inShape : Shape := .dim inC (.dim inH (.dim inW .scalar))
    let node : NodeData α Δ (Γ ++ ss) outShape :=
      { forward := fun ctx _d =>
          let xv := getIdx (α := α) (xs := ctx) ix
          Spec.avgPool2dMultiSpecPad (h1 := h1) (h2 := h2) (layer := layer) (padding := padding)
            xv
        jvp := fun _ctx dctx _d =>
          let dx := getIdx (α := α) (xs := dctx) ix
          Spec.avgPool2dMultiSpecPad (h1 := h1) (h2 := h2) (layer := layer) (padding := padding)
            dx
        vjp := fun _ctx _d δ =>
          let dx :=
            Spec.avgPool2dMultiBackwardSpecPad (h1 := h1) (h2 := h2) (layer := layer)
              (padding := padding) δ
          TList.single (α := α) (Γ := Γ ++ ss) (s := inShape) ix dx }
    push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outShape) g node
  else
    throw "compiled GraphM: avg_pool2d_pad requires stride > 0"

/--
Flatten a tensor to a 1D vector (preserving total size).

PyTorch comparison: `torch.flatten(x)` (for a single tensor value).
-/
def flatten {α : Type} {Δ : Type} [Inhabited α] [Zero α] [DecidableEq Shape]
  {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var (.dim (Shape.size s) .scalar)) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let outS : Shape := .dim (Shape.size s) .scalar
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d => flattenSpec (α := α) (s := s) (getIdx (α := α) (xs := ctx) ix)
      jvp := fun _ctx dctx _d =>
        flattenSpec (α := α) (s := s) (getIdx (α := α) (xs := dctx) ix)
      vjp := fun _ctx _d δ =>
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix (unflattenSpec (α := α) s δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
Reshape a tensor, given a proof that the total sizes match.

PyTorch comparison: `torch.reshape(x, new_shape)`.
-/
def reshape {α : Type} {Δ : Type} [Inhabited α] [Zero α] [DecidableEq Shape]
  {Γ : List Shape} {s₁ s₂ : Shape} (x : Var s₁) (h : Shape.size s₁ = Shape.size s₂) :
    MWith α Δ Γ (Var s₂) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s₂ :=
    { forward := fun ctx _d =>
        Spec.Tensor.reshapeSpec (α := α) (s₁ := s₁) (s₂ := s₂) (getIdx (α := α) (xs := ctx) ix) h
      jvp := fun _ctx dctx _d =>
        Spec.Tensor.reshapeSpec (α := α) (s₁ := s₁) (s₂ := s₂) (getIdx (α := α) (xs := dctx) ix) h
      vjp := fun _ctx _d δ =>
        TList.single (α := α) (Γ := Γ ++ ss) (s := s₁) ix
          (Spec.Tensor.reshapeSpec (α := α) (s₁ := s₂) (s₂ := s₁) δ h.symm) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s₂) g node

/-- Transpose a 2D matrix. PyTorch comparison: `x.transpose(0, 1)` / `x.T` for matrices. -/
def transpose2d {α : Type} {Δ : Type} [Zero α] [DecidableEq Shape]
  {Γ : List Shape} {m n : Nat} (x : Var (.dim m (.dim n .scalar))) :
    MWith α Δ Γ (Var (.dim n (.dim m .scalar))) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let outS : Shape := .dim n (.dim m .scalar)
  let inS : Shape := .dim m (.dim n .scalar)
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        matrixTransposeSpec (α := α) (getIdx (α := α) (xs := ctx) ix)
      jvp := fun _ctx dctx _d =>
        matrixTransposeSpec (α := α) (getIdx (α := α) (xs := dctx) ix)
      vjp := fun _ctx _d δ =>
        TList.single (α := α) (Γ := Γ ++ ss) (s := inS) ix (matrixTransposeSpec (α := α) δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
Transpose a rank-3 tensor by moving the first axis to the last (`(a,b,c) → (b,c,a)`).

PyTorch comparison: `x.permute(1, 2, 0)`.
-/
def transpose3dFirstToLast {α : Type} {Δ : Type} [Zero α] [DecidableEq Shape]
  {Γ : List Shape} {a b c : Nat} (x : Var (.dim a (.dim b (.dim c .scalar)))) :
    MWith α Δ Γ (Var (.dim b (.dim c (.dim a .scalar)))) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let inS : Shape := .dim a (.dim b (.dim c .scalar))
  let outS : Shape := .dim b (.dim c (.dim a .scalar))
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        Spec.Tensor.transpose3DFirstToLastSpec (α := α) (a := a) (b := b) (c := c)
          (getIdx (α := α) (xs := ctx) ix)
      jvp := fun _ctx dctx _d =>
        Spec.Tensor.transpose3DFirstToLastSpec (α := α) (a := a) (b := b) (c := c)
          (getIdx (α := α) (xs := dctx) ix)
      vjp := fun _ctx _d δ =>
        TList.single (α := α) (Γ := Γ ++ ss) (s := inS) ix
          (Spec.Tensor.transpose3DLastToFirstSpec (α := α) (a := b) (b := c) (c := a) δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
Transpose a rank-3 tensor by moving the last axis to the first (`(a,b,c) → (c,a,b)`).

PyTorch comparison: `x.permute(2, 0, 1)`.
-/
def transpose3dLastToFirst {α : Type} {Δ : Type} [Zero α] [DecidableEq Shape]
  {Γ : List Shape} {a b c : Nat} (x : Var (.dim a (.dim b (.dim c .scalar)))) :
    MWith α Δ Γ (Var (.dim c (.dim a (.dim b .scalar)))) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let inS : Shape := .dim a (.dim b (.dim c .scalar))
  let outS : Shape := .dim c (.dim a (.dim b .scalar))
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        Spec.Tensor.transpose3DLastToFirstSpec (α := α) (a := a) (b := b) (c := c)
          (getIdx (α := α) (xs := ctx) ix)
      jvp := fun _ctx dctx _d =>
        Spec.Tensor.transpose3DLastToFirstSpec (α := α) (a := a) (b := b) (c := c)
          (getIdx (α := α) (xs := dctx) ix)
      vjp := fun _ctx _d δ =>
        TList.single (α := α) (Γ := Γ ++ ss) (s := inS) ix
          (Spec.Tensor.transpose3DFirstToLastSpec (α := α) (a := c) (b := a) (c := b) δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
Swap the last two axes of a rank-3 tensor (`(a,b,c) → (a,c,b)`).

PyTorch comparison: `x.transpose(1, 2)` for a 3D tensor.
-/
def transpose3dLastTwo {α : Type} {Δ : Type} [Zero α] [DecidableEq Shape]
  {Γ : List Shape} {a b c : Nat} (x : Var (.dim a (.dim b (.dim c .scalar)))) :
    MWith α Δ Γ (Var (.dim a (.dim c (.dim b .scalar)))) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let inS : Shape := .dim a (.dim b (.dim c .scalar))
  let outS : Shape := .dim a (.dim c (.dim b .scalar))
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        Spec.Tensor.transpose3DLastTwoSpec (α := α) (a := a) (b := b) (c := c)
          (getIdx (α := α) (xs := ctx) ix)
      jvp := fun _ctx dctx _d =>
        Spec.Tensor.transpose3DLastTwoSpec (α := α) (a := a) (b := b) (c := c)
          (getIdx (α := α) (xs := dctx) ix)
      vjp := fun _ctx _d δ =>
        TList.single (α := α) (Γ := Γ ++ ss) (s := inS) ix
          (Spec.Tensor.transpose3DLastTwoSpec (α := α) (a := a) (b := c) (c := b) δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
  Swap two adjacent axes at a given nesting `depth`.

  This is the compiled-graph analogue of the eager `Tape.swapAdjacentAtDepth`.
  PyTorch comparison: a `permute` that swaps two neighboring dimensions.
  -/
  def swapAdjacentAtDepth {α : Type} {Δ : Type} [Zero α] [DecidableEq Shape]
    {Γ : List Shape} {s : Shape} (depth : Nat) (x : Var s) :
      MWith α Δ Γ (Var (s.swapAdjacentAtDepth depth)) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let outS : Shape := s.swapAdjacentAtDepth depth
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        Spec.Tensor.swapAtDepthHelper (tensor := getIdx (α := α) (xs := ctx) ix) depth
      jvp := fun _ctx dctx _d =>
        let dx := getIdx (α := α) (xs := dctx) ix
        Spec.Tensor.swapAtDepthHelper (tensor := dx) depth
      vjp := fun _ctx _d δ =>
        let dx' := Spec.Tensor.swapAtDepthHelper (tensor := δ) depth
        let dx : Tensor α s :=
          Tensor.castShape dx' (by simpa [outS] using (Spec.Shape.swapAdjacentAtDepth_involutive s
            depth))
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix dx }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
Broadcast `x : s₁` to a larger shape `s₂` (given a `CanBroadcastTo` witness).

PyTorch comparison: `x.expand(...)` / broadcasting semantics in elementwise ops.
-/
def broadcastTo {α : Type} {Δ : Type} [Inhabited α] [Add α] [Zero α] [DecidableEq Shape]
  {Γ : List Shape} {s₁ s₂ : Shape} (cb : Shape.CanBroadcastTo s₁ s₂) (x : Var s₁) :
  MWith α Δ Γ (Var s₂) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s₂ :=
    { forward := fun ctx _d =>
        Spec.Tensor.broadcastTo (α := α) cb (getIdx (α := α) (xs := ctx) ix)
      jvp := fun _ctx dctx _d =>
        Spec.Tensor.broadcastTo (α := α) cb (getIdx (α := α) (xs := dctx) ix)
      vjp := fun _ctx _d δ =>
        TList.single (α := α) (Γ := Γ ++ ss) (s := s₁) ix
          (Spec.Tensor.reduceFromBroadcastTo (α := α) cb δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s₂) g node

/--
Reduce-sum along a given `axis`.

PyTorch comparison: `torch.sum(x, dim=axis)`.
-/
def reduceSum {α : Type} {Δ : Type} [Add α] [Zero α] [Inhabited α] [DecidableEq Shape]
  {Γ : List Shape} {s : Shape} (axis : Nat)
  [valid : Shape.valid_axis_inst axis s] [wf : Shape.WellFormed s]
  (x : Var s) : MWith α Δ Γ (Var (shapeAfterSum s axis)) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let outS : Shape := shapeAfterSum s axis
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        reduceSumAuto (α := α) (s := s) axis (getIdx (α := α) (xs := ctx) ix)
      jvp := fun _ctx dctx _d =>
        reduceSumAuto (α := α) (s := s) axis (getIdx (α := α) (xs := dctx) ix)
      vjp := fun _ctx _d δ =>
        let cb := shapeAfterSumBroadcastBack (s := s) axis valid wf
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix (Spec.Tensor.broadcastTo (α := α) cb δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
Reduce-mean along a given `axis`.

PyTorch comparison: `torch.mean(x, dim=axis)`.
-/
def reduceMean {α : Type} {Δ : Type} [Context α] [Inhabited α] [DecidableEq Shape]
  {Γ : List Shape} {s : Shape} (axis : Nat)
  [valid : Shape.valid_axis_inst axis s] [wf : Shape.WellFormed s]
  (x : Var s) : MWith α Δ Γ (Var (shapeAfterSum s axis)) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let outS : Shape := shapeAfterSum s axis
  let denomNat :=
    match getDimSize s axis with
    | some n => n
    | none => 1
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        let xv := getIdx (α := α) (xs := ctx) ix
        let h := Shape.proveReducibleAlong axis s valid.proof
        Spec.Tensor.reduceMean (α := α) (s := s) axis xv h
      jvp := fun _ctx dctx _d =>
        let dx := getIdx (α := α) (xs := dctx) ix
        let h := Shape.proveReducibleAlong axis s valid.proof
        Spec.Tensor.reduceMean (α := α) (s := s) axis dx h
      vjp := fun _ctx _d δ =>
        let cb := shapeAfterSumBroadcastBack (s := s) axis valid wf
        let dLdx := Spec.Tensor.broadcastTo (α := α) cb δ
        let dLdx' := scaleSpec (α := α) (s := s) dLdx (1 / (denomNat : α))
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix dLdx' }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
  Gather a single scalar from a vector at a known-in-bounds index.

  PyTorch comparison: `x[i]` for a 1D tensor.
  -/
  def gatherScalar {α : Type} {Δ : Type} [Zero α] [DecidableEq Shape]
    {Γ : List Shape} {n : Nat} (x : Var (.dim n .scalar)) (i : Fin n) : MWith α Δ Γ (Var
      Shape.scalar) := do
    let ⟨ss, g⟩ ← get
    let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
    let node : NodeData α Δ (Γ ++ ss) Shape.scalar :=
      { forward := fun ctx _d =>
          getAtSpec (getIdx (α := α) (xs := ctx) ix) i
        jvp := fun _ctx dctx _d =>
          getAtSpec (getIdx (α := α) (xs := dctx) ix) i
        vjp := fun _ctx _d δ =>
          let gVal : α := Tensor.toScalar δ
          let dx : Tensor α (.dim n .scalar) :=
            Tensor.dim (fun j => Tensor.scalar (if decide (j = i) then gVal else 0))
          TList.single (α := α) (Γ := Γ ++ ss) (s := .dim n .scalar) ix dx }
    push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := Shape.scalar) g node

/--
  Gather a row from a matrix at a known-in-bounds row index.

  PyTorch comparison: `x[i, :]` for a 2D tensor.
  -/
def gatherRow {α : Type} {Δ : Type} [Zero α] [DecidableEq Shape]
  {Γ : List Shape} {rows cols : Nat} (x : Var (.dim rows (.dim cols .scalar))) (i : Fin rows) :
  MWith α Δ Γ (Var (.dim cols .scalar)) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let outS : Shape := .dim cols .scalar
  let inS : Shape := .dim rows (.dim cols .scalar)
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        getAtSpec (getIdx (α := α) (xs := ctx) ix) i
      jvp := fun _ctx dctx _d =>
        getAtSpec (getIdx (α := α) (xs := dctx) ix) i
      vjp := fun _ctx _d δ =>
        let dx : Tensor α inS :=
          Tensor.dim (fun r =>
            if decide (r = i) then
              δ
            else
              fill (0 : α) outS)
        TList.single (α := α) (Γ := Γ ++ ss) (s := inS) ix dx }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
  Gather a scalar from a vector at a runtime `Nat` index.

  If `i` is out of bounds we return `0` and propagate no gradient (matching the forward choice).
  -/
  def gatherScalarNat {α : Type} {Δ : Type} [Zero α] [DecidableEq Shape]
    {Γ : List Shape} {n : Nat} (x : Var (.dim n .scalar)) (i : Nat) :
    MWith α Δ Γ (Var Shape.scalar) := do
    let ⟨ss, g⟩ ← get
    let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
    let node : NodeData α Δ (Γ ++ ss) Shape.scalar :=
      { forward := fun ctx _d =>
          let xv := getIdx (α := α) (xs := ctx) ix
          if h : i < n then
            getAtSpec xv ⟨i, h⟩
          else
            Tensor.scalar 0
        jvp := fun _ctx dctx _d =>
          let dx := getIdx (α := α) (xs := dctx) ix
          if h : i < n then
            getAtSpec dx ⟨i, h⟩
          else
            Tensor.scalar 0
        vjp := fun _ctx _d δ =>
          let gVal : α := Tensor.toScalar δ
          let dx : Tensor α (.dim n .scalar) :=
            Tensor.dim (fun j =>
            if _hi : i < n then
              Tensor.scalar (if decide (j.val = i) then gVal else 0)
            else
              Tensor.scalar 0)
          TList.single (α := α) (Γ := Γ ++ ss) (s := .dim n .scalar) ix dx }
    push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := Shape.scalar) g node

/--
  Gather a vector of length `k` from a length-`n` vector using an index tensor of `Nat`s.

  Out-of-bounds indices yield `0` at the corresponding output position.

  PyTorch comparison: `torch.gather` for 1D inputs, with explicit bounds handling.
  -/
def gatherVecNat {α : Type} {Δ : Type} [Add α] [Zero α] [DecidableEq Shape]
  {Γ : List Shape} {n k : Nat} (x : Var (.dim n .scalar)) (idx : Tensor Nat (.dim k .scalar)) :
  MWith α Δ Γ (Var (.dim k .scalar)) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let outS : Shape := .dim k .scalar
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        let xv := getIdx (α := α) (xs := ctx) ix
        match idx with
        | Tensor.dim f =>
            Tensor.dim (fun j =>
              match f j with
              | Tensor.scalar ij =>
                  if h : ij < n then
                    getAtSpec xv ⟨ij, h⟩
                  else
                    Tensor.scalar 0)
      jvp := fun _ctx dctx _d =>
        let dx := getIdx (α := α) (xs := dctx) ix
        match idx with
        | Tensor.dim f =>
            Tensor.dim (fun j =>
              match f j with
              | Tensor.scalar ij =>
                  if h : ij < n then
                    getAtSpec dx ⟨ij, h⟩
                  else
                    Tensor.scalar 0)
      vjp := fun _ctx _d δ =>
        let dx : Tensor α (.dim n .scalar) :=
          Tensor.dim (fun iFin =>
            let sum : α :=
              (List.finRange k).foldl (fun acc j =>
                let ij :=
                  match idx with
                  | Tensor.dim f =>
                      match f j with
                      | Tensor.scalar v => v
                if _hij : ij < n then
                  if decide (ij = iFin.val) then
                    let gj : α :=
                      match getAtSpec δ j with
                      | Tensor.scalar v => v
                    acc + gj
                  else acc
                else acc
              ) 0
            Tensor.scalar sum)
        TList.single (α := α) (Γ := Γ ++ ss) (s := .dim n .scalar) ix dx }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
  Gather `k` rows from a `(rows×cols)` matrix using an index vector of `Nat`s.

  Out-of-bounds indices yield a zero row.

  PyTorch comparison: `torch.index_select(x, dim=0, index=idx)` with explicit bounds handling.
  -/
def gatherRowsNat {α : Type} {Δ : Type} [Add α] [Zero α] [DecidableEq Shape]
  {Γ : List Shape} {rows cols k : Nat} (x : Var (.dim rows (.dim cols .scalar)))
  (idx : Tensor Nat (.dim k .scalar)) :
  MWith α Δ Γ (Var (.dim k (.dim cols .scalar))) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let outS : Shape := .dim k (.dim cols .scalar)
  let inS : Shape := .dim rows (.dim cols .scalar)
  let rowS : Shape := .dim cols .scalar
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        let xv := getIdx (α := α) (xs := ctx) ix
        match idx with
        | Tensor.dim f =>
            Tensor.dim (fun j =>
              match f j with
              | Tensor.scalar ij =>
                  if h : ij < rows then
                    getAtSpec xv ⟨ij, h⟩
                  else
                    fill (0 : α) rowS)
      jvp := fun _ctx dctx _d =>
        let dx0 := getIdx (α := α) (xs := dctx) ix
        match idx with
        | Tensor.dim f =>
            Tensor.dim (fun j =>
              match f j with
              | Tensor.scalar ij =>
                  if h : ij < rows then
                    getAtSpec dx0 ⟨ij, h⟩
                  else
                    fill (0 : α) rowS)
      vjp := fun _ctx _d δ =>
        let dx : Tensor α inS :=
          Tensor.dim (fun rFin =>
            (List.finRange k).foldl (fun acc j =>
              let ij :=
                match idx with
                | Tensor.dim f =>
                    match f j with
                    | Tensor.scalar v => v
              if _hij : ij < rows then
                if decide (ij = rFin.val) then
                  addSpec acc (getAtSpec δ j)
                else
                  acc
              else
                acc
            ) (fill (0 : α) rowS))
        TList.single (α := α) (Γ := Γ ++ ss) (s := inS) ix dx }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
Scatter-add into a vector at a single in-bounds index.

`scatter_add_vec x v i` adds the scalar `v` into `x[i]`.

PyTorch comparison: `x.index_add_(dim=0, index=[i], source=[v])` (conceptually).
-/
def scatterAddVec {α : Type} {Δ : Type} [Add α] [Zero α] [DecidableEq Shape]
  {Γ : List Shape} {n : Nat} (x : Var (.dim n .scalar)) (v : Var Shape.scalar) (i : Fin n) :
  MWith α Δ Γ (Var (.dim n .scalar)) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let iv ← liftM (mkIdx (_α := α) (Γ := Γ) ss v)
  let outS : Shape := .dim n .scalar
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        let xv := getIdx (α := α) (xs := ctx) ix
        let vv : α := Tensor.toScalar (getIdx (α := α) (xs := ctx) iv)
        let xi : α := Tensor.toScalar (getAtSpec xv i)
        updateSpec xv [i.val] (xi + vv)
      jvp := fun _ctx dctx _d =>
        let dx := getIdx (α := α) (xs := dctx) ix
        let dv : α := Tensor.toScalar (getIdx (α := α) (xs := dctx) iv)
        let dxi : α := Tensor.toScalar (getAtSpec dx i)
        updateSpec dx [i.val] (dxi + dv)
      vjp := fun _ctx _d δ =>
        let dv : Tensor α Shape.scalar := getAtSpec δ i
        TList.add (α := α) (ss := Γ ++ ss)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := outS) ix δ)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := Shape.scalar) iv dv) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
Scatter-add into a matrix at a single in-bounds row index.

`scatter_add_row x v i` adds the row vector `v` into `x[i, :]`.

PyTorch comparison: `x.index_add_(dim=0, index=[i], source=v.unsqueeze(0))` (conceptually).
-/
def scatterAddRow {α : Type} {Δ : Type} [Add α] [Zero α] [DecidableEq Shape]
  {Γ : List Shape} {rows cols : Nat}
  (x : Var (.dim rows (.dim cols .scalar))) (v : Var (.dim cols .scalar)) (i : Fin rows) :
  MWith α Δ Γ (Var (.dim rows (.dim cols .scalar))) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let iv ← liftM (mkIdx (_α := α) (Γ := Γ) ss v)
  let outS : Shape := .dim rows (.dim cols .scalar)
  let rowS : Shape := .dim cols .scalar
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        let xv := getIdx (α := α) (xs := ctx) ix
        let vv := getIdx (α := α) (xs := ctx) iv
        Tensor.dim (fun r =>
          if decide (r = i) then
            addSpec (getAtSpec xv r) vv
          else
            getAtSpec xv r)
      jvp := fun _ctx dctx _d =>
        let dx := getIdx (α := α) (xs := dctx) ix
        let dv := getIdx (α := α) (xs := dctx) iv
        Tensor.dim (fun r =>
          if decide (r = i) then
            addSpec (getAtSpec dx r) dv
          else
            getAtSpec dx r)
      vjp := fun _ctx _d δ =>
        let dv : Tensor α rowS := getAtSpec δ i
        TList.add (α := α) (ss := Γ ++ ss)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := outS) ix δ)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := rowS) iv dv) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
Layer normalization (sequence-first), producing the same shape as the input.

PyTorch comparison: `torch.nn.LayerNorm` / `torch.nn.functional.layer_norm` (modulo exact layout).

Forward-mode status: implemented by `Spec.layerNormJvp`, including parameter tangents for
`gamma` and `beta`.
-/
def layerNorm {α : Type} {Δ : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  [DecidableEq Shape]
  {Γ : List Shape} {seqLen embedDim : Nat}
  (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
  (x : Var (.dim seqLen (.dim embedDim .scalar)))
  (gamma : Var (.dim embedDim .scalar))
  (beta : Var (.dim embedDim .scalar)) :
  MWith α Δ Γ (Var (.dim seqLen (.dim embedDim .scalar))) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let ig ← liftM (mkIdx (_α := α) (Γ := Γ) ss gamma)
  let ib ← liftM (mkIdx (_α := α) (Γ := Γ) ss beta)
  let node : NodeData α Δ (Γ ++ ss) (.dim seqLen (.dim embedDim .scalar)) :=
    { forward := fun ctx _d =>
        Spec.layerNorm (α := α) (seqLen := seqLen) (embedDim := embedDim)
          (x := getIdx (α := α) (xs := ctx) ix)
          (gamma := getIdx (α := α) (xs := ctx) ig)
          (beta := getIdx (α := α) (xs := ctx) ib)
          (h_seq_pos := h_seq_pos) (h_embed_pos := h_embed_pos)
      jvp := fun ctx dctx _d =>
        let xv := getIdx (α := α) (xs := ctx) ix
        let gv := getIdx (α := α) (xs := ctx) ig
        let bv := getIdx (α := α) (xs := ctx) ib
        let dx := getIdx (α := α) (xs := dctx) ix
        let dg := getIdx (α := α) (xs := dctx) ig
        let db := getIdx (α := α) (xs := dctx) ib
        Spec.layerNormJvp (α := α) (seqLen := seqLen) (embedDim := embedDim)
          (h_seq_pos := h_seq_pos) (h_embed_pos := h_embed_pos)
          (x := xv) (tangent := dx) (gamma := gv) (dgamma := dg) (_beta := bv) (dbeta := db)
      vjp := fun ctx _d dLdy =>
        let xv := getIdx (α := α) (xs := ctx) ix
        let gv := getIdx (α := α) (xs := ctx) ig
        let bv := getIdx (α := α) (xs := ctx) ib
        let (dx, dgamma, dbeta) :=
          Spec.layerNormBackward (α := α) (seqLen := seqLen) (embedDim := embedDim)
            (h_seq_pos := h_seq_pos) (h_embed_pos := h_embed_pos)
            (x := xv) (gamma := gv) (_beta := bv) (grad_output := dLdy)
        let z0 :=
          TList.add (α := α) (ss := Γ ++ ss)
            (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim seqLen (.dim embedDim .scalar)) ix dx)
            (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim embedDim .scalar) ig dgamma)
        TList.add (α := α) (ss := Γ ++ ss) z0
          (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim embedDim .scalar) ib dbeta) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := (.dim seqLen (.dim embedDim .scalar))) g node

/--
Batch normalization in channel-first layout (no running statistics; spec-level functional form).

PyTorch comparison: `torch.nn.BatchNorm2d` in NCHW layout (modulo exact semantics/parameters).

Forward-mode status: implemented by `Spec.batchNorm2dJvp`, including parameter tangents for
`gamma` and `beta`.
-/
def batchnormChannelFirst {α : Type} {Δ : Type} [Context α] [DecidableRel ((· > ·) : α → α →
  Prop)] [DecidableEq Shape]
  {Γ : List Shape} {channels height width : Nat}
  (h_c : channels > 0) (h_h : height > 0) (h_w : width > 0)
  (x : Var (.dim channels (.dim height (.dim width .scalar))))
  (gamma : Var (.dim channels .scalar))
  (beta : Var (.dim channels .scalar)) :
  MWith α Δ Γ (Var (.dim channels (.dim height (.dim width .scalar)))) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let ig ← liftM (mkIdx (_α := α) (Γ := Γ) ss gamma)
  let ib ← liftM (mkIdx (_α := α) (Γ := Γ) ss beta)
  let outS : Shape := .dim channels (.dim height (.dim width .scalar))
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        Spec.batchNorm2d (α := α) (channels := channels) (height := height) (width := width)
          (x := getIdx (α := α) (xs := ctx) ix)
          (gamma := getIdx (α := α) (xs := ctx) ig)
          (beta := getIdx (α := α) (xs := ctx) ib)
          (h_c := h_c) (h_h := h_h) (h_w := h_w)
      jvp := fun ctx dctx _d =>
        let xv := getIdx (α := α) (xs := ctx) ix
        let gv := getIdx (α := α) (xs := ctx) ig
        let bv := getIdx (α := α) (xs := ctx) ib
        let dx := getIdx (α := α) (xs := dctx) ix
        let dg := getIdx (α := α) (xs := dctx) ig
        let db := getIdx (α := α) (xs := dctx) ib
        Spec.batchNorm2dJvp (α := α) (channels := channels) (height := height) (width := width)
          (x := xv) (tangent := dx) (gamma := gv) (dgamma := dg) (_beta := bv) (dbeta := db)
          (_h_c := h_c) (_h_h := h_h) (_h_w := h_w)
      vjp := fun ctx _d dLdy =>
        let xv := getIdx (α := α) (xs := ctx) ix
        let gv := getIdx (α := α) (xs := ctx) ig
        let (dx, dgamma, dbeta) :=
          Spec.batchNorm2dBackward (α := α) (channels := channels) (height := height) (width := width)
            (x := xv) (gamma := gv) (grad_output := dLdy)
            (_h_c := h_c) (_h_h := h_h) (_h_w := h_w)
        let z0 :=
          TList.add (α := α) (ss := Γ ++ ss)
            (TList.single (α := α) (Γ := Γ ++ ss) (s := outS) ix dx)
            (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim channels .scalar) ig dgamma)
        TList.add (α := α) (ss := Γ ++ ss) z0
          (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim channels .scalar) ib dbeta) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
Multi-head attention primitive (shape-specialized).

PyTorch comparison: `torch.nn.MultiheadAttention` / scaled dot-product attention.

Forward-mode status: implemented by `Spec.MultiHeadAttentionJvp`, including tangents for the
input and all four projection matrices.
-/
def multiHeadAttention {α : Type} {Δ : Type} [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {Γ : List Shape} {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (wq : Var (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wk : Var (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wv : Var (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wo : Var (.dim (numHeads * headDim) (.dim dModel .scalar)))
  (x : Var (.dim n (.dim dModel .scalar)))
  (mask : Option (Tensor Bool (.dim n (.dim n .scalar))) := none) :
  MWith α Δ Γ (Var (.dim n (.dim dModel .scalar))) := do
  let ⟨ss, g⟩ ← get
  let iwq ← liftM (mkIdx (_α := α) (Γ := Γ) ss wq)
  let iwk ← liftM (mkIdx (_α := α) (Γ := Γ) ss wk)
  let iwv ← liftM (mkIdx (_α := α) (Γ := Γ) ss wv)
  let iwo ← liftM (mkIdx (_α := α) (Γ := Γ) ss wo)
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) (.dim n (.dim dModel .scalar)) :=
    { forward := fun ctx _d =>
        let mha : Spec.MultiHeadAttention α numHeads dModel headDim :=
          { Wq := getIdx (α := α) (xs := ctx) iwq
            Wk := getIdx (α := α) (xs := ctx) iwk
            Wv := getIdx (α := α) (xs := ctx) iwv
            Wo := getIdx (α := α) (xs := ctx) iwo }
        Spec.MultiHeadAttention.forward (α := α) (n := n) (h1 := h1)
          (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
          (mha := mha) (x := getIdx (α := α) (xs := ctx) ix) (mask := mask)
      jvp := fun ctx dctx _d =>
        let mha : Spec.MultiHeadAttention α numHeads dModel headDim :=
          { Wq := getIdx (α := α) (xs := ctx) iwq
            Wk := getIdx (α := α) (xs := ctx) iwk
            Wv := getIdx (α := α) (xs := ctx) iwv
            Wo := getIdx (α := α) (xs := ctx) iwo }
        let dmha : Spec.MultiHeadAttention α numHeads dModel headDim :=
          { Wq := getIdx (α := α) (xs := dctx) iwq
            Wk := getIdx (α := α) (xs := dctx) iwk
            Wv := getIdx (α := α) (xs := dctx) iwv
            Wo := getIdx (α := α) (xs := dctx) iwo }
        Spec.MultiHeadAttentionJvp (α := α) (h1 := h1)
          (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
          (mha := mha) (dmha := dmha)
          (x := getIdx (α := α) (xs := ctx) ix)
          (dx := getIdx (α := α) (xs := dctx) ix)
          (mask := mask)
      vjp := fun ctx _d dLdy =>
        let mha : Spec.MultiHeadAttention α numHeads dModel headDim :=
          { Wq := getIdx (α := α) (xs := ctx) iwq
            Wk := getIdx (α := α) (xs := ctx) iwk
            Wv := getIdx (α := α) (xs := ctx) iwv
            Wo := getIdx (α := α) (xs := ctx) iwo }
        let xv := getIdx (α := α) (xs := ctx) ix
        let (dx, dWq, dWk, dWv, dWo) :=
          Spec.MultiHeadAttentionBackward (α := α) (h1 := h1)
            (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
            (mha := mha) (x := xv) (mask := mask) (grad_output := dLdy)
        let z0 :=
          TList.add (α := α) (ss := Γ ++ ss)
            (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim dModel (.dim (numHeads * headDim)
              .scalar)) iwq dWq)
            (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim dModel (.dim (numHeads * headDim)
              .scalar)) iwk dWk)
        let z1 :=
          TList.add (α := α) (ss := Γ ++ ss) z0
            (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim dModel (.dim (numHeads * headDim)
              .scalar)) iwv dWv)
        let z2 :=
          TList.add (α := α) (ss := Γ ++ ss) z1
            (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim (numHeads * headDim) (.dim dModel
              .scalar)) iwo dWo)
        TList.add (α := α) (ss := Γ ++ ss) z2
          (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim n (.dim dModel .scalar)) ix dx) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := (.dim n (.dim dModel .scalar))) g node

/--
N-D convolution (channels-first) on a single sample tensor (no batch axis).

Conventions:
- input shape is `(inC, spatial...)`,
- kernel shape is `(outC, inC, kernelSpatial...)`,
- bias shape is `(outC)`,
- output spatial sizes use the usual PyTorch-style formula (floor division).

PyTorch comparison: `torch.nn.functional.conv{d}d`, specialized to a single sample.

Forward-mode JVP uses bilinearity:
`d(conv(k,b,x)) = conv(k,0,dx) + conv(dk,db,x)`.
-/
def conv {α : Type} {Δ : Type} [Context α] [DecidableEq Shape]
  {Γ : List Shape} {d inC outC : Nat}
  {kernel stride padding : Vector Nat d}
  {inSpatial : Vector Nat d}
  {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (w : Var (Shape.ofList (outC :: inC :: kernel.toList)))
  (b : Var (.dim outC .scalar))
  (x : Var (Shape.ofList (inC :: inSpatial.toList))) :
  MWith α Δ Γ (Var (Shape.ofList (outC :: (Spec.convOutSpatial inSpatial kernel stride padding).toList))) := do
  have _ := hInC
  have _ := hKernel
  let ⟨ss, g⟩ ← get
  let iw ← liftM (mkIdx (_α := α) (Γ := Γ) ss w)
  let ib ← liftM (mkIdx (_α := α) (Γ := Γ) ss b)
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)

  let outSpatial : Vector Nat d :=
    Spec.convOutSpatial inSpatial kernel stride padding
  let outS : Shape := Shape.ofList (outC :: outSpatial.toList)
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        let wv := getIdx (α := α) (xs := ctx) iw
        let bv := getIdx (α := α) (xs := ctx) ib
        let xv := getIdx (α := α) (xs := ctx) ix
        let layer : Spec.ConvSpec d inC outC kernel stride padding α :=
          { kernel := wv, bias := bv }
        Spec.convSpec (layer := layer) xv
      jvp := fun ctx dctx _d =>
        let wv := getIdx (α := α) (xs := ctx) iw
        let xv := getIdx (α := α) (xs := ctx) ix
        let dW := getIdx (α := α) (xs := dctx) iw
        let dB := getIdx (α := α) (xs := dctx) ib
        let dX := getIdx (α := α) (xs := dctx) ix
        let zeroBias : Tensor α (.dim outC .scalar) := fill (0 : α) (.dim outC .scalar)
        let layerX : Spec.ConvSpec d inC outC kernel stride padding α :=
          { kernel := wv, bias := zeroBias }
        let layerParams : Spec.ConvSpec d inC outC kernel stride padding α :=
          { kernel := dW, bias := dB }
        addSpec (Spec.convSpec (layer := layerX) dX) (Spec.convSpec (layer := layerParams) xv)
      vjp := fun ctx _d dLdy =>
        let wv := getIdx (α := α) (xs := ctx) iw
        let bv := getIdx (α := α) (xs := ctx) ib
        let xv := getIdx (α := α) (xs := ctx) ix
        let layer : Spec.ConvSpec d inC outC kernel stride padding α :=
          { kernel := wv, bias := bv }
        let (dW, dB, dX) := Spec.convBackwardSpec (layer := layer) xv dLdy
        let z0 :=
          TList.add (α := α) (ss := Γ ++ ss)
            (TList.single (α := α) (Γ := Γ ++ ss)
              (s := Shape.ofList (outC :: inC :: kernel.toList)) iw dW)
            (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim outC .scalar) ib dB)
        TList.add (α := α) (ss := Γ ++ ss) z0
          (TList.single (α := α) (Γ := Γ ++ ss)
            (s := Shape.ofList (inC :: inSpatial.toList)) ix dX) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
N-D transpose convolution (channels-first) on a single sample tensor (no batch axis).

Conventions:
- input shape is `(inC, spatial...)`,
- kernel shape is `(inC, outC, kernelSpatial...)` (PyTorch layout),
- bias shape is `(outC)`,
- output spatial sizes use:
  `out[a] = (in[a] - 1) * stride[a] - 2*padding[a] + kernel[a]` (with `output_padding = 0`).

PyTorch comparison: `torch.nn.functional.conv_transpose{d}d`, specialized to a single sample.

Forward-mode JVP uses bilinearity:
`d(convTranspose(k,b,x)) = convTranspose(k,0,dx) + convTranspose(dk,db,x)`.
-/
def convTranspose {α : Type} {Δ : Type} [Context α] [DecidableEq Shape]
  {Γ : List Shape} {d inC outC : Nat}
  {kernel stride padding : Vector Nat d}
  {inSpatial : Vector Nat d}
  {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (w : Var (Shape.ofList (inC :: outC :: kernel.toList)))
  (b : Var (.dim outC .scalar))
  (x : Var (Shape.ofList (inC :: inSpatial.toList))) :
  MWith α Δ Γ (Var (Shape.ofList (outC :: (Spec.convTransposeOutSpatial inSpatial kernel stride padding).toList))) := do
  have _ := hInC
  have _ := hKernel
  let ⟨ss, g⟩ ← get
  let iw ← liftM (mkIdx (_α := α) (Γ := Γ) ss w)
  let ib ← liftM (mkIdx (_α := α) (Γ := Γ) ss b)
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)

  let outSpatial : Vector Nat d :=
    Spec.convTransposeOutSpatial inSpatial kernel stride padding
  let outS : Shape := Shape.ofList (outC :: outSpatial.toList)
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        let wv := getIdx (α := α) (xs := ctx) iw
        let bv := getIdx (α := α) (xs := ctx) ib
        let xv := getIdx (α := α) (xs := ctx) ix
        let layer : Spec.ConvTransposeSpec d inC outC kernel stride padding α :=
          { kernel := wv, bias := bv }
        Spec.convTransposeSpec (layer := layer) xv
      jvp := fun ctx dctx _d =>
        let wv := getIdx (α := α) (xs := ctx) iw
        let xv := getIdx (α := α) (xs := ctx) ix
        let dW := getIdx (α := α) (xs := dctx) iw
        let dB := getIdx (α := α) (xs := dctx) ib
        let dX := getIdx (α := α) (xs := dctx) ix
        let zeroBias : Tensor α (.dim outC .scalar) := fill (0 : α) (.dim outC .scalar)
        let layerX : Spec.ConvTransposeSpec d inC outC kernel stride padding α :=
          { kernel := wv, bias := zeroBias }
        let layerParams : Spec.ConvTransposeSpec d inC outC kernel stride padding α :=
          { kernel := dW, bias := dB }
        addSpec (Spec.convTransposeSpec (layer := layerX) dX)
          (Spec.convTransposeSpec (layer := layerParams) xv)
      vjp := fun ctx _d dLdy =>
        let wv := getIdx (α := α) (xs := ctx) iw
        let bv := getIdx (α := α) (xs := ctx) ib
        let xv := getIdx (α := α) (xs := ctx) ix
        let layer : Spec.ConvTransposeSpec d inC outC kernel stride padding α :=
          { kernel := wv, bias := bv }
        let (dW, dB, dX) := Spec.convTransposeBackwardSpec (layer := layer) xv dLdy
        let z0 :=
          TList.add (α := α) (ss := Γ ++ ss)
            (TList.single (α := α) (Γ := Γ ++ ss)
              (s := Shape.ofList (inC :: outC :: kernel.toList)) iw dW)
            (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim outC .scalar) ib dB)
        TList.add (α := α) (ss := Γ ++ ss) z0
          (TList.single (α := α) (Γ := Γ ++ ss)
            (s := Shape.ofList (inC :: inSpatial.toList)) ix dX) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
2D convolution (channel-first) on a single image tensor.

PyTorch comparison: `torch.nn.functional.conv2d` (without a batch dimension).

Forward-mode JVP uses bilinearity:
`d(conv2d(k,b,x)) = conv2d(k,0,dx) + conv2d(dk,db,x)`.
-/
def conv2d {α : Type} {Δ : Type} [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {Γ : List Shape} {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (kernel : Var (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))))
  (bias : Var (.dim outC .scalar))
  (input : Var (.dim inC (.dim inH (.dim inW .scalar)))) :
  MWith α Δ Γ (Var (.dim outC (.dim ((inH + 2 * padding - kH) / stride + 1) (.dim ((inW + 2 *
    padding - kW) / stride + 1) .scalar)))) := do
  let ⟨ss, g⟩ ← get
  let ik ← liftM (mkIdx (_α := α) (Γ := Γ) ss kernel)
  let ib ← liftM (mkIdx (_α := α) (Γ := Γ) ss bias)
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss input)
  let outH : Nat := (inH + 2 * padding - kH) / stride + 1
  let outW : Nat := (inW + 2 * padding - kW) / stride + 1
  let outS : Shape := .dim outC (.dim outH (.dim outW .scalar))
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        let kern := getIdx (α := α) (xs := ctx) ik
        let bv := getIdx (α := α) (xs := ctx) ib
        let inp := getIdx (α := α) (xs := ctx) ix
        let layer :
            Spec.Conv2DSpec inC outC kH kW stride padding α h1 h2 h3 :=
          { kernel := kern
            bias := bv }
        Spec.conv2dSpec (layer := layer) inp
      jvp := fun ctx dctx _d =>
        let kern := getIdx (α := α) (xs := ctx) ik
        let inp := getIdx (α := α) (xs := ctx) ix
        let dKernel := getIdx (α := α) (xs := dctx) ik
        let dBias := getIdx (α := α) (xs := dctx) ib
        let dInput := getIdx (α := α) (xs := dctx) ix
        let zeroBias : Tensor α (.dim outC .scalar) := fill (0 : α) (.dim outC .scalar)
        let layerX :
            Spec.Conv2DSpec inC outC kH kW stride padding α h1 h2 h3 :=
          { kernel := kern
            bias := zeroBias }
        let layerParams :
            Spec.Conv2DSpec inC outC kH kW stride padding α h1 h2 h3 :=
          { kernel := dKernel
            bias := dBias }
        addSpec (Spec.conv2dSpec (layer := layerX) dInput)
          (Spec.conv2dSpec (layer := layerParams) inp)
      vjp := fun ctx _d dLdy =>
        let kern := getIdx (α := α) (xs := ctx) ik
        let bv := getIdx (α := α) (xs := ctx) ib
        let inp := getIdx (α := α) (xs := ctx) ix
        let layer :
            Spec.Conv2DSpec inC outC kH kW stride padding α h1 h2 h3 :=
          { kernel := kern
            bias := bv }
        let (dKernel, dBias, dInput) := Spec.conv2dBackwardSpec (layer := layer) inp dLdy
        let z0 :=
          TList.add (α := α) (ss := Γ ++ ss)
            (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim outC (.dim inC (.dim kH (.dim kW
              .scalar)))) ik dKernel)
            (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim outC .scalar) ib dBias)
        TList.add (α := α) (ss := Γ ++ ss) z0
          (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim inC (.dim inH (.dim inW .scalar))) ix
            dInput) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
2D transpose convolution (channel-first) on a single image tensor.

PyTorch comparison: `torch.nn.functional.conv_transpose2d` (without a batch dimension).

Forward-mode JVP uses bilinearity:
`d(convTranspose2d(k,b,x)) = convTranspose2d(k,0,dx) + convTranspose2d(dk,db,x)`.
-/
def convTranspose2d {α : Type} {Δ : Type} [Context α]
  [DecidableEq Shape]
  {Γ : List Shape} {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (kernel : Var (.dim inC (.dim outC (.dim kH (.dim kW .scalar)))))
  (bias : Var (.dim outC .scalar))
  (input : Var (.dim inC (.dim inH (.dim inW .scalar)))) :
  MWith α Δ Γ (Var (.dim outC (.dim ((inH - 1) * stride - 2 * padding + kH)
    (.dim ((inW - 1) * stride - 2 * padding + kW) .scalar)))) := do
  have h1' : inC > 0 := Nat.pos_of_ne_zero h1
  let ⟨ss, g⟩ ← get
  let ik ← liftM (mkIdx (_α := α) (Γ := Γ) ss kernel)
  let ib ← liftM (mkIdx (_α := α) (Γ := Γ) ss bias)
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss input)
  let outH : Nat := (inH - 1) * stride - 2 * padding + kH
  let outW : Nat := (inW - 1) * stride - 2 * padding + kW
  let outS : Shape := .dim outC (.dim outH (.dim outW .scalar))
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        let kern := getIdx (α := α) (xs := ctx) ik
        let bv := getIdx (α := α) (xs := ctx) ib
        let inp := getIdx (α := α) (xs := ctx) ix
        let layer :
            Spec.ConvTranspose2DSpec inC outC kH kW stride padding α h1' h2 h3 :=
          { kernel := kern
            bias := bv }
        Spec.convTranspose2dSpec (layer := layer) inp
      jvp := fun ctx dctx _d =>
        let kern := getIdx (α := α) (xs := ctx) ik
        let inp := getIdx (α := α) (xs := ctx) ix
        let dKernel := getIdx (α := α) (xs := dctx) ik
        let dBias := getIdx (α := α) (xs := dctx) ib
        let dInput := getIdx (α := α) (xs := dctx) ix
        let zeroBias : Tensor α (.dim outC .scalar) := fill (0 : α) (.dim outC .scalar)
        let layerX :
            Spec.ConvTranspose2DSpec inC outC kH kW stride padding α h1' h2 h3 :=
          { kernel := kern
            bias := zeroBias }
        let layerParams :
            Spec.ConvTranspose2DSpec inC outC kH kW stride padding α h1' h2 h3 :=
          { kernel := dKernel
            bias := dBias }
        addSpec (Spec.convTranspose2dSpec (layer := layerX) dInput)
          (Spec.convTranspose2dSpec (layer := layerParams) inp)
      vjp := fun ctx _d dLdy =>
        let kern := getIdx (α := α) (xs := ctx) ik
        let bv := getIdx (α := α) (xs := ctx) ib
        let inp := getIdx (α := α) (xs := ctx) ix
        let layer :
            Spec.ConvTranspose2DSpec inC outC kH kW stride padding α h1' h2 h3 :=
          { kernel := kern
            bias := bv }
        let (dKernel, dBias, dInput) := Spec.convTranspose2dBackwardSpec (layer := layer) inp dLdy
        let z0 :=
          TList.add (α := α) (ss := Γ ++ ss)
            (TList.single (α := α) (Γ := Γ ++ ss)
              (s := .dim inC (.dim outC (.dim kH (.dim kW .scalar)))) ik dKernel)
            (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim outC .scalar) ib dBias)
        TList.add (α := α) (ss := Γ ++ ss) z0
          (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim inC (.dim inH (.dim inW .scalar))) ix
            dInput) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

end GraphM
end Compiled
end Autograd
end Runtime
