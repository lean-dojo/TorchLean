/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core
import Mathlib.Algebra.Order.Algebra

/-!
# LinkedSession

Proof-linked imperative Session (eager-style API, proved IR under the hood).

Background:
- `Runtime.Autograd.TorchLean.Session` provides a unified imperative API for training/debugging
  (eager) and verification-friendly execution (compiled).
- `Proofs.Autograd.Algebra.GraphData` is the proved/typed SSA(DAG) IR used by the
  proof-compiled pipeline (`Proofs.Autograd.Algebra.Graph.compileAuxData`), and
  `NN/Proofs/Autograd/Runtime/Link.lean` proves that running the runtime reverse-mode loop on the
  compiled tape matches `GraphData.backpropAllCtx`.

This file provides a *session-style* API that records a `GraphData` (well-typed IR) as you call
ops imperatively, and then runs the standard runtime tape loop on the compiled tape.

Key guarantee (pure theorem, no `IO` reasoning needed):
- If the session snapshot is `(g, x)`, then `Tape.backwardDenseFrom (compileAuxData g x)` equals
  `GraphData.backpropAllCtx g x` (via `backwardDenseFrom_compileAuxData_eq_backpropAllCtx`).

Practical note:
- This session enforces a simple invariant: **all leaf tensors are created before any op node**.
  This matches the standard training pattern (reset → add leaves → forward → backward).
- `const` is available as a graph node, so you can still introduce literal constants mid-graph.
- This is the fully proof-linked variant used by `TorchLean.Session` when `opts.backend :=
  .compiled`.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace Torch

open Spec
open Tensor

namespace Internal

/--
Convenience: turn a `Result α` into `IO α` by throwing `IO.userError` on `.error`.

This mirrors the common pattern in the eager runtime front-end (`Torch.Core`).
-/
abbrev okOrThrow {α : Type} : Runtime.Autograd.Result α → IO α :=
  Runtime.Autograd.okOrThrow

/-- Non-differentiable external environment for the proved graph: a small array of `Nat` inputs. -/
abbrev NatEnv : Type := Array Nat

/-- Internal proof-linked session state (a well-typed `GraphData` plus its leaf values). -/
structure SessionIRState (α : Type) where
  /-- Leaf shapes (inputs/parameters), in creation order. -/
  Γ : List Shape
  /-- Leaf values, aligned with `Γ`. -/
  x : _root_.Proofs.Autograd.Algebra.TList α Γ
  /-- Non-differentiable external inputs (e.g. class labels/indices). -/
  nat : NatEnv
  /-- Internal node shapes, in creation order. -/
  ss : List Shape
  /-- SSA/DAG graph nodes (one per entry in `ss`). -/
  g : _root_.Proofs.Autograd.Algebra.GraphData α NatEnv Γ ss

namespace SessionIRState

/-- Empty session state: no leaves, no nodes, empty nat-environment. -/
def empty {α : Type} : SessionIRState α :=
  { Γ := []
    x := .nil
    nat := #[]
    ss := []
    g := .nil }

end SessionIRState

/--
`SessionIR` is an imperative session that records a `GraphData` (proved IR) as it runs.

It is "eager-style" (you call ops imperatively), but it is proof-linked: the recorded graph can be
compiled and then the runtime tape backward loop is provably equal to `GraphData.backpropAllCtx`.
-/
structure SessionIR (α : Type) where
  /-- Session options shared with the eager front-end. -/
  opts : Options
  /-- Mutable proof-linked graph snapshot. -/
  st : IO.Ref (SessionIRState α)
  /-- Map from graph leaf ids to mutable parameter objects. -/
  paramsByLeaf : IO.Ref (Std.HashMap Nat (AnyParam α))

namespace SessionIR

/--
Create a new proof-linked session.

This allocates `IO.Ref`s for the session snapshot (`SessionIRState`) and the leaf-id→parameter map.
Call `resetTape` to start a new "graph recording" phase.
-/
def new {α : Type} (opts : Options := {}) : IO (SessionIR α) := do
  let st ← IO.mkRef (SessionIRState.empty (α := α))
  let paramsByLeaf ← IO.mkRef (Std.HashMap.emptyWithCapacity)
  pure { opts := opts, st := st, paramsByLeaf := paramsByLeaf }

/--
Reset the session to an empty snapshot.

Important invariant: this session requires that **all leaves are created before any op node**.
`resetTape` is the intended boundary between training steps/forwards.
-/
def resetTape {α : Type} (s : SessionIR α) : IO Unit := do
  s.st.set (SessionIRState.empty (α := α))
  s.paramsByLeaf.set (Std.HashMap.emptyWithCapacity)

/--
Create a mutable parameter object (not yet part of the recorded graph).

To use the parameter in the recorded graph, call `use`, which reads its current value and records
it as a *leaf* in `Γ`.
PyTorch comparison: analogous to creating a `torch.nn.Parameter` and then using it in a forward.
-/
def param {α : Type} (s : SessionIR α) {sh : Shape}
  (init : Tensor α sh) (name : Option String := none) (requiresGrad : Option Bool := none) :
  IO (Param α sh) := do
  let r ← IO.mkRef init
  let cudaValue ← IO.mkRef (none : Option Runtime.Autograd.Cuda.AnyBuffer)
  let hostCurrent ← IO.mkRef true
  pure { name := name
         value := r
         cudaValue := cudaValue
         hostCurrent := hostCurrent
         requiresGrad := requiresGrad.getD s.opts.requiresGradByDefault }

/--
Enforce the session invariant: leaves must be created before any op node.

This keeps the `GraphData` context split `Γ ++ ss` easy to reason about and matches the typical
training pattern: `resetTape → add leaves → forward ops → backward`.
-/
def ensureNoNodes {α : Type} (st : SessionIRState α) : IO Unit := do
  match st.ss with
  | [] => pure ()
  | _ :: _ =>
      throw <| IO.userError
        ("torch(SessionIR): cannot add a new leaf after graph nodes have been " ++
          "created (resetTape first)")

/--
Record a new differentiable leaf tensor in the session context `Γ`.

This is the primitive used by `use` (parameters) and `input` (external inputs).
-/
def addLeaf {α : Type} (s : SessionIR α) {sh : Shape} (v : Tensor α sh) :
    IO (TensorRef α sh) := do
  let st0 ← s.st.get
  ensureNoNodes st0
  let id := st0.Γ.length
  let Γ' := st0.Γ ++ [sh]
  let x' : _root_.Proofs.Autograd.Algebra.TList α Γ' :=
    _root_.Proofs.Autograd.Algebra.TList.snoc (α := α) (ss := st0.Γ) (τ := sh) st0.x v
  -- No nodes yet, so the graph stays `nil`.
  let st1 : SessionIRState α :=
    { Γ := Γ'
      x := x'
      nat := st0.nat
      ss := []
      g := .nil }
  s.st.set st1
  pure { id := id }

/--
Use a `Param` in the recorded graph by reading its current value and recording it as a leaf.

The returned `TensorRef` is the graph handle you pass to subsequent ops. The session also remembers
which leaf-id corresponds to which parameter, so `sgdStepAll` can update parameters after backward.
PyTorch comparison: like referencing a `torch.nn.Parameter` in the forward; the parameter's value
is treated as a leaf for autograd.
-/
def use {α : Type} (s : SessionIR α) {sh : Shape} [DecidableEq Shape]
  (p : Param α sh) : IO (TensorRef α sh) := do
  let v ← p.value.get
  let leaf ← addLeaf (α := α) s (sh := sh) v
  s.paramsByLeaf.modify (fun m => m.insert leaf.id (AnyParam.ofParam p))
  pure leaf

/--
Record an external differentiable input tensor as a leaf.

`name` and `requiresGrad` are accepted for API parity with the eager session, but this proof-linked
session always records the input in `Γ` (a leaf) and uses typing/invariants to determine what
gradients are meaningful.
-/
def input {α : Type} (s : SessionIR α) {sh : Shape} [DecidableEq Shape]
  (v : Tensor α sh) (name : Option String := none) (requiresGrad : Bool := false) :
  IO (TensorRef α sh) := do
  -- `name`/`requiresGrad` are accepted for API parity with the eager Session.
  -- This proof-linked session always records the value as a leaf in `Γ`.
  let _ := name
  let _ := requiresGrad
  addLeaf (α := α) s (sh := sh) v

/--
Record a non-differentiable `Nat` input in the external environment.

This is used for "index-like" inputs (labels, gather indices, etc.) that should not receive
gradients.
PyTorch comparison: like passing an integer tensor / index to an op; indices are not differentiable.
-/
def inputNat {α : Type} (s : SessionIR α) (v : Nat) : IO NatRef := do
  let st0 ← s.st.get
  ensureNoNodes st0
  let id := st0.nat.size
  s.st.set { st0 with nat := st0.nat.push v }
  pure { id := id }

/-- Read a previously recorded `NatRef`. -/
def getNat {α : Type} (s : SessionIR α) (r : NatRef) : IO Nat := do
  let st0 ← s.st.get
  if h : r.id < st0.nat.size then
    pure <| st0.nat[r.id]'h
  else
    throw <| IO.userError "torch(SessionIR): invalid nat id"

/-- Overwrite a previously recorded `NatRef`. -/
def setNat {α : Type} (s : SessionIR α) (r : NatRef) (v : Nat) : IO Unit := do
  let st0 ← s.st.get
  if h : r.id < st0.nat.size then
    let i : Fin st0.nat.size := ⟨r.id, h⟩
    s.st.set { st0 with nat := st0.nat.set i v }
  else
    throw <| IO.userError "torch(SessionIR): invalid nat id"

/--
Convert a small `Tensor Nat (.dim k .scalar)` into an `Array Nat`.

This is used to stage `NatVecRef` inputs into the session nat-environment.
-/
def natVecToArray {k : Nat} (v : Tensor Nat (.dim k .scalar)) : Array Nat :=
  Array.ofFn (fun i : Fin k =>
    match getAtSpec v i with
    | .scalar n => n)

/--
Record a non-differentiable vector of `Nat` inputs.

Returns a `NatVecRef k` which points into the nat-environment. This is useful for "runtime gather"
style ops where indices are supplied externally (and are not differentiable).
-/
def inputNatVec {α : Type} {k : Nat} (s : SessionIR α) (v : Tensor Nat (.dim k .scalar)) : IO
  (NatVecRef k) := do
  let st0 ← s.st.get
  ensureNoNodes st0
  let start := st0.nat.size
  let xsNew := (natVecToArray (k := k) v).foldl (fun acc x => acc.push x) st0.nat
  s.st.set { st0 with nat := xsNew }
  pure { start := start }

/-- Read back the `k`-vector stored at a `NatVecRef k`. -/
def getNatVec {α : Type} {k : Nat} (s : SessionIR α) (r : NatVecRef k) : IO (Tensor Nat (.dim k
  .scalar)) := do
  let st0 ← s.st.get
  if h : r.start + k ≤ st0.nat.size then
    pure <|
      Tensor.dim (fun i =>
        have hi : r.start + i.val < r.start + k := Nat.add_lt_add_left i.is_lt r.start
        have hi' : r.start + i.val < st0.nat.size := lt_of_lt_of_le hi h
        Tensor.scalar (st0.nat[r.start + i.val]'hi'))
  else
    throw <| IO.userError "torch(SessionIR): invalid nat vec ref (out of bounds)"

/-- Overwrite the nat-environment segment referenced by `NatVecRef k`. -/
def setNatVec {α : Type} {k : Nat} (s : SessionIR α) (r : NatVecRef k) (v : Tensor Nat (.dim k
  .scalar)) : IO Unit := do
  let st0 ← s.st.get
  if h : r.start + k ≤ st0.nat.size then
    let xs' :=
      (List.finRange k).foldl (fun acc (i : Fin k) =>
        have hi : r.start + i.val < st0.nat.size := by
          have hlt : r.start + i.val < r.start + k := Nat.add_lt_add_left i.is_lt r.start
          exact lt_of_lt_of_le hlt h
        let vi : Nat :=
          match getAtSpec v i with
          | .scalar n => n
        acc.set! (r.start + i.val) vi
      ) st0.nat
    s.st.set { st0 with nat := xs' }
  else
    throw <| IO.userError "torch(SessionIR): invalid nat vec ref (out of bounds)"

/--
Build a typed index into the current context `Γ ++ ss` from a raw numeric id and expected shape.

This is the main "dynamic check" used by `getValue` (and by a few index-driven nodes): it ensures
that the `Nat` id points to an existing tensor in the session context and that the shape matches.
-/
def mkIdxOrThrow {_α : Type} {Γ ss : List Shape} (id : Nat) (s : Shape) :
    Runtime.Autograd.Result (_root_.Proofs.Autograd.Algebra.Idx (Γ ++ ss) s) := by
    if h : id < (Γ ++ ss).length then
      let fin : Fin (Γ ++ ss).length := ⟨id, h⟩
      let got : Shape := (Γ ++ ss).get fin
      if hg : got = s then
        exact .ok ⟨fin, hg⟩
      else
        exact .error <|
          s!"torch(SessionIR): shape mismatch at id={id}: expected {Shape.pretty s}, got "
            ++ s!"{Shape.pretty got}"
  else
    exact .error s!"torch(SessionIR): invalid id={id} for ctxLen={(Γ ++ ss).length}"

/--
Evaluate the recorded graph and return the value of a `TensorRef`.

This is a pure graph evaluation (`GraphData.eval`) using the recorded leaf values and
nat-environment. It does **not** run the runtime tape or mutate session state.
-/
def getValue {α : Type} (s : SessionIR α) {sh : Shape} [DecidableEq Shape]
  (x : TensorRef α sh) : IO (Tensor α sh) := do
  let st0 ← s.st.get
  -- Evaluate the recorded graph at the recorded leaf values.
  let ctx : _root_.Proofs.Autograd.Algebra.TList α (st0.Γ ++ st0.ss) :=
    _root_.Proofs.Autograd.Algebra.GraphData.eval (α := α) (Δ := NatEnv) (Γ := st0.Γ) (ss := st0.ss)
      st0.g st0.x st0.nat
  let idx ← okOrThrow (mkIdxOrThrow (_α := α) (Γ := st0.Γ) (ss := st0.ss) x.id sh)
  pure (_root_.Proofs.Autograd.Algebra.getIdx (α := α) (xs := ctx) idx)

/-! ## Graph-node ops (implemented by reusing `Compiled.GraphM`) -/

/--
Run a `Compiled.GraphM` computation against the current `(ss, g)` pair.

`Compiled.GraphM` is the builder monad used by the proof-friendly compiled pipeline; reusing it
here ensures this eager-style API records *the same* typed IR that the compiler expects.
-/
def runGraphM {α : Type} {Γ : List Shape} {β : Type}
    (m : Runtime.Autograd.Compiled.GraphM.MWith α NatEnv Γ β)
    (ss : List Shape) (g : _root_.Proofs.Autograd.Algebra.GraphData α NatEnv Γ ss) :
    Runtime.Autograd.Result (β × (Σ ss' : List Shape, _root_.Proofs.Autograd.Algebra.GraphData α
      NatEnv Γ ss')) :=
  StateT.run m ⟨ss, g⟩

/--
Atomically apply a graph-building update to the session snapshot.

This is the central adapter used by each op wrapper below: it reads `s.st`, runs a builder that
returns an updated `SessionIRState`, stores it back into `s.st`, and returns the op result.
-/
def commitGraphM {α : Type} (s : SessionIR α) {β : Type}
    (k :
      ∀ {Γ : List Shape} {ss : List Shape},
        (x : _root_.Proofs.Autograd.Algebra.TList α Γ) →
        (nat : NatEnv) →
        (g : _root_.Proofs.Autograd.Algebra.GraphData α NatEnv Γ ss) →
        Runtime.Autograd.Result (β × SessionIRState α)) :
    IO β := do
  let st0 ← s.st.get
  let r ← okOrThrow (k (Γ := st0.Γ) (ss := st0.ss) st0.x st0.nat st0.g)
  let (b, st1) := r
  s.st.set st1
  pure b

/--
Record a constant tensor.

Subtlety: if no op nodes have been created yet (`ss = []`), we record `const` as a leaf to match
the eager session's leaf-collection behavior. Once op nodes exist, we emit an explicit constant node
so users can introduce literal constants mid-graph.
PyTorch comparison: like `torch.tensor(...)` (a leaf) vs inserting a literal constant into the
graph; constants are treated as non-requires-grad.
-/
def const {α : Type} (s : SessionIR α) {sh : Shape} [Zero α] [DecidableEq Shape]
  (v : Tensor α sh) (name : Option String := none) : IO (TensorRef α sh) := do
  let _ := name
  let st0 ← s.st.get
  match st0.ss with
  | [] =>
      -- Still in the "leaf collection" phase: keep `const` as a leaf for parity with the eager
      -- Session.
      input (α := α) s (sh := sh) v (name := name) (requiresGrad := false)
  | _ :: _ =>
      -- Mid-graph: emit an explicit constant node.
      commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} xv nat g => do
        let (vout, st') ← runGraphM (α := α) (Γ := Γ)
          (Runtime.Autograd.Compiled.GraphM.const (α := α) (Γ := Γ) (s := sh) v)
          ss g
        let ⟨ss', g'⟩ := st'
        let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
        pure ({ id := vout.id }, st1))

/--
Record elementwise addition `a + b`.

PyTorch comparison: `torch.add(a, b)` / the `+` operator.
-/
def add {α : Type} (s : SessionIR α) [Add α] [Zero α] [DecidableEq Shape] {sh : Shape}
  (a b : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} x nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.add (α := α) (Γ := Γ) (s := sh) { id := a.id } { id := b.id
        })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := x, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record elementwise subtraction `a - b`.

PyTorch comparison: `torch.sub(a, b)` / the `-` operator.
-/
def sub {α : Type} (s : SessionIR α) [Sub α] [Add α] [Zero α] [DecidableEq Shape] {sh : Shape}
  (a b : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} x nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.sub (α := α) (Γ := Γ) (s := sh) { id := a.id } { id := b.id
        })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := x, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record elementwise multiplication `a * b`.

PyTorch comparison: `torch.mul(a, b)` / the `*` operator.
-/
def mul {α : Type} (s : SessionIR α) [Mul α] [Add α] [Zero α] [DecidableEq Shape] {sh : Shape}
  (a b : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} x nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.mul (α := α) (Γ := Γ) (s := sh) { id := a.id } { id := b.id
        })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := x, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record scaling by a scalar constant: `x * c`.

PyTorch comparison: like `x * c` (where `c` is a Python scalar).
-/
def scale {α : Type} (s : SessionIR α) [Mul α] [Add α] [Zero α] [DecidableEq Shape] {sh : Shape}
  (x : TensorRef α sh) (c : α) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.scale (α := α) (Γ := Γ) (s := sh) { id := x.id } c)
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record elementwise absolute value.

PyTorch comparison: `torch.abs(x)`.
-/
def abs {α : Type} (s : SessionIR α)
  [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.abs (α := α) (Γ := Γ) (s := sh) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Stop-gradient boundary.

Forward semantics: identity.
Backward semantics: no gradient flows to the input.
PyTorch comparison: `x.detach()`.
-/
def detach {α : Type} (s : SessionIR α) [Context α] [DecidableEq Shape] {sh : Shape}
    (x : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.detach (α := α) (Γ := Γ) (s := sh) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record elementwise square root.

PyTorch comparison: `torch.sqrt(x)`.
-/
def sqrt {α : Type} (s : SessionIR α)
  [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.sqrt (α := α) (Γ := Γ) (s := sh) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record elementwise clamp to the interval `[minVal, maxVal]`.

PyTorch comparison: `torch.clamp(x, min=minVal, max=maxVal)`.
-/
def clamp {α : Type} (s : SessionIR α)
  [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {sh : Shape} (x : TensorRef α sh) (minVal maxVal : α) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.clamp (α := α) (Γ := Γ) (s := sh) { id := x.id } minVal
        maxVal)
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record elementwise maximum of `a` and `b`.

PyTorch comparison: `torch.maximum(a, b)`.
-/
def max {α : Type} (s : SessionIR α)
  [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {sh : Shape} (a b : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} x nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.max (α := α) (Γ := Γ) (s := sh) { id := a.id } { id := b.id
        })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := x, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record elementwise minimum of `a` and `b`.

PyTorch comparison: `torch.minimum(a, b)`.
-/
def min {α : Type} (s : SessionIR α)
  [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {sh : Shape} (a b : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} x nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.min (α := α) (Γ := Γ) (s := sh) { id := a.id } { id := b.id
        })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := x, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record 2D matrix multiplication.

PyTorch comparison: `torch.matmul(a, b)` for 2D tensors.
-/
def matmul {α : Type} (s : SessionIR α) [Context α] [DecidableEq Shape]
  {m n p : Nat}
  (a : TensorRef α (.dim m (.dim n .scalar)))
  (b : TensorRef α (.dim n (.dim p .scalar))) :
  IO (TensorRef α (.dim m (.dim p .scalar))) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim m (.dim p .scalar))) (fun {Γ} {ss} x nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.matmul (α := α) (Γ := Γ) (m := m) (n := n) (p := p) { id :=
        a.id } { id := b.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := x, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record batched matrix multiplication.

PyTorch comparison: `torch.bmm(a, b)` for 3D tensors of shape `(batch, m, n)` and `(batch, n, p)`.
-/
def bmm {α : Type} (s : SessionIR α) [Add α] [Mul α] [Zero α] [DecidableEq Shape]
  {batch m n p : Nat}
  (a : TensorRef α (.dim batch (.dim m (.dim n .scalar))))
  (b : TensorRef α (.dim batch (.dim n (.dim p .scalar)))) :
  IO (TensorRef α (.dim batch (.dim m (.dim p .scalar)))) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim batch (.dim m (.dim p .scalar)))) (fun {Γ} {ss} x
    nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.bmm (α := α) (Γ := Γ) (batch := batch) (m := m) (n := n) (p
        := p) { id := a.id } { id := b.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := x, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Concatenate two 1D vectors along dimension 0.

PyTorch comparison: `torch.cat([a, b], dim=0)` for 1D tensors.
-/
def concatVectors {α : Type} (s : SessionIR α) [Context α] [DecidableEq Shape]
  {n m : Nat}
  (a : TensorRef α (.dim n .scalar))
  (b : TensorRef α (.dim m .scalar)) :
  IO (TensorRef α (.dim (n + m) .scalar)) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim (n + m) .scalar)) (fun {Γ} {ss} x nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.concatVectors (α := α) (Γ := Γ) (n := n) (m := m) { id :=
        a.id } { id := b.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := x, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Concatenate two tensors along dimension 0.

PyTorch comparison: `torch.cat([a, b], dim=0)`.
-/
def concatDim0 {α : Type} (s : SessionIR α) [Context α] [DecidableEq Shape]
  {n m : Nat} {sh : Shape}
  (a : TensorRef α (.dim n sh))
  (b : TensorRef α (.dim m sh)) :
  IO (TensorRef α (.dim (n + m) sh)) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim (n + m) sh)) (fun {Γ} {ss} x nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.concatDim0 (α := α) (Γ := Γ) (n := n) (m := m) (s := sh)
        { id := a.id } { id := b.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := x, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Slice a tensor along dimension 0.

This returns `x[start : start+len]`. The proof argument `h` enforces bounds.
PyTorch comparison: `x[start:start+len]` for tensors with a leading dimension.
-/
def sliceRange0 {α : Type} (s : SessionIR α) [Zero α] [DecidableEq Shape]
  {n : Nat} {sh : Shape}
  (x : TensorRef α (.dim n sh)) (start len : Nat) (h : len + start ≤ n) :
  IO (TensorRef α (.dim len sh)) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim len sh)) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.sliceRange0 (α := α) (Γ := Γ) (n := n) (s := sh) { id :=
        x.id } start len h)
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
N-D max-pooling for channels-first tensors `(C, spatial...)` (no batch axis).

PyTorch comparison: `torch.nn.functional.max_pool1d` / `max_pool2d` / `max_pool3d` depending on the
spatial rank `d`.
-/
def maxPool {α : Type} (s : SessionIR α) [Context α] [DecidableEq Shape]
  {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
  {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (x : TensorRef α (Shape.ofList (C :: inSpatial.toList))) :
  IO (TensorRef α (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))) :=
  commitGraphM (α := α) s
    (β := TensorRef α (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList)))
    (fun {Γ} {ss} xv nat g => do
      let (v, st') ← runGraphM (α := α) (Γ := Γ)
        (Runtime.Autograd.Compiled.GraphM.maxPool (α := α) (Γ := Γ) (d := d) (C := C)
          (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
          (hKernel := hKernel) { id := x.id })
        ss g
      let ⟨ss', g'⟩ := st'
      let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
      pure ({ id := v.id }, st1))

/--
N-D smooth max-pooling (log-sum-exp surrogate) for channels-first tensors `(C, spatial...)`.

This is a differentiable approximation of max-pooling; there is no direct PyTorch primitive.
-/
def smoothMaxPool {α : Type} (s : SessionIR α) [Context α] [DecidableEq Shape]
  {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
  {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (x : TensorRef α (Shape.ofList (C :: inSpatial.toList))) (beta : α) :
  IO (TensorRef α (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))) :=
  commitGraphM (α := α) s
    (β := TensorRef α (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList)))
    (fun {Γ} {ss} xv nat g => do
      let (v, st') ← runGraphM (α := α) (Γ := Γ)
        (Runtime.Autograd.Compiled.GraphM.smoothMaxPool (α := α) (Γ := Γ) (d := d) (C := C)
          (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
          (hKernel := hKernel) { id := x.id } beta)
        ss g
      let ⟨ss', g'⟩ := st'
      let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
      pure ({ id := v.id }, st1))

/--
N-D average-pooling for channels-first tensors `(C, spatial...)` (no batch axis).

PyTorch comparison: `torch.nn.functional.avg_pool1d` / `avg_pool2d` / `avg_pool3d` depending on the
spatial rank `d`.
-/
def avgPool {α : Type} (s : SessionIR α) [Context α] [DecidableEq Shape]
  {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
  (hKernel : ∀ i : Fin d, kernel.get i ≠ 0)
  (x : TensorRef α (Shape.ofList (C :: inSpatial.toList))) :
  IO (TensorRef α (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))) :=
  commitGraphM (α := α) s
    (β := TensorRef α (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList)))
    (fun {Γ} {ss} xv nat g => do
      let (v, st') ← runGraphM (α := α) (Γ := Γ)
        (Runtime.Autograd.Compiled.GraphM.avgPool (α := α) (Γ := Γ) (d := d) (C := C)
          (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
          hKernel { id := x.id })
        ss g
      let ⟨ss', g'⟩ := st'
      let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
      pure ({ id := v.id }, st1))

/--
2D max-pooling for channel-first images.

PyTorch comparison: `torch.nn.functional.max_pool2d` (for NCHW-like layouts, here without batch).
-/
def maxPool2d {α : Type} (s : SessionIR α) [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (x : TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (TensorRef α (.dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1)
    .scalar)))) :=
  commitGraphM (α := α) s
    (β := TensorRef α (.dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1)
      .scalar))))
    (fun {Γ} {ss} xv nat g => do
      let (v, st') ← runGraphM (α := α) (Γ := Γ)
        (Runtime.Autograd.Compiled.GraphM.maxPool2d (α := α) (Γ := Γ)
          (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
          (h1 := h1) (h2 := h2) { id := x.id })
        ss g
      let ⟨ss', g'⟩ := st'
      let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
      pure ({ id := v.id }, st1))

/--
Smooth approximation of max-pooling (softmax pooling) for channel-first images.

This is not a standard PyTorch primitive; conceptually it behaves like applying a softmax over each
pooling window with inverse-temperature `beta` and returning the expected value.
-/
def smoothMaxPool2d {α : Type} (s : SessionIR α) [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (x : TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) (beta : α) :
  IO (TensorRef α (.dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1)
    .scalar)))) :=
  commitGraphM (α := α) s
    (β := TensorRef α (.dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1)
      .scalar))))
    (fun {Γ} {ss} xv nat g => do
      let (v, st') ← runGraphM (α := α) (Γ := Γ)
        (Runtime.Autograd.Compiled.GraphM.smoothMaxPool2d (α := α) (Γ := Γ)
          (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
          (h1 := h1) (h2 := h2) { id := x.id } beta)
        ss g
      let ⟨ss', g'⟩ := st'
      let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
      pure ({ id := v.id }, st1))

/--
2D average-pooling for channel-first images.

PyTorch comparison: `torch.nn.functional.avg_pool2d` (for NCHW-like layouts, here without batch).
-/
def avgPool2d {α : Type} (s : SessionIR α) [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0)
  (x : TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (TensorRef α (.dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1)
    .scalar)))) :=
  commitGraphM (α := α) s
    (β := TensorRef α (.dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1)
      .scalar))))
    (fun {Γ} {ss} xv nat g => do
      let (v, st') ← runGraphM (α := α) (Γ := Γ)
        (Runtime.Autograd.Compiled.GraphM.avgPool2d (α := α) (Γ := Γ)
          (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
          h1 h2 { id := x.id })
        ss g
      let ⟨ss', g'⟩ := st'
      let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
      pure ({ id := v.id }, st1))

/--
Record elementwise ReLU.

PyTorch comparison: `torch.relu(x)` / `torch.nn.functional.relu(x)`.
-/
def relu {α : Type} (s : SessionIR α)
  [Mul α] [Add α] [Zero α] [Max α] [One α] [LT α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.relu (α := α) (Γ := Γ) (s := sh) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Flatten a tensor into a 1D vector of length `Shape.size sh`.

PyTorch comparison: `torch.flatten(x)` (with default `start_dim=0`).
-/
def flatten {α : Type} (s : SessionIR α) [Inhabited α] [Zero α] [DecidableEq Shape] {sh : Shape}
  (x : TensorRef α sh) : IO (TensorRef α (.dim (Shape.size sh) .scalar)) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim (Shape.size sh) .scalar)) (fun {Γ} {ss} xv nat g
    => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.flatten (α := α) (Γ := Γ) (s := sh) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Reshape a tensor while preserving total number of elements.

The proof argument `h` enforces `Shape.size sh1 = Shape.size sh2`.
PyTorch comparison: `torch.reshape(x, new_shape)` / `x.view(new_shape)` (when contiguous).
-/
def reshape {α : Type} (s : SessionIR α) [Inhabited α] [Zero α] [DecidableEq Shape]
  {sh1 sh2 : Shape} (x : TensorRef α sh1) (h : Shape.size sh1 = Shape.size sh2) : IO (TensorRef α
    sh2) :=
  commitGraphM (α := α) s (β := TensorRef α sh2) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.reshape (α := α) (Γ := Γ) (s₁ := sh1) (s₂ := sh2) { id :=
        x.id } h)
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Transpose a 2D matrix (swap the two axes).

PyTorch comparison: `x.t()` for 2D tensors, or `x.transpose(0, 1)`.
-/
def transpose2d {α : Type} (s : SessionIR α) [Zero α] [DecidableEq Shape]
  {m n : Nat} (x : TensorRef α (.dim m (.dim n .scalar))) : IO (TensorRef α (.dim n (.dim m
    .scalar))) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim n (.dim m .scalar))) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.transpose2d (α := α) (Γ := Γ) (m := m) (n := n) { id := x.id
        })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Permute a 3D tensor by moving the first axis to the end: `(a,b,c) → (b,c,a)`.

PyTorch comparison: `x.permute(1,2,0)` for a 3D tensor.
-/
def transpose3dFirstToLast {α : Type} (s : SessionIR α) [Zero α] [DecidableEq Shape]
  {a b c : Nat} (x : TensorRef α (.dim a (.dim b (.dim c .scalar)))) :
  IO (TensorRef α (.dim b (.dim c (.dim a .scalar)))) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim b (.dim c (.dim a .scalar)))) (fun {Γ} {ss} xv nat
    g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.transpose3dFirstToLast (α := α) (Γ := Γ) (a := a) (b :=
        b) (c := c) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Permute a 3D tensor by moving the last axis to the front: `(a,b,c) → (c,a,b)`.

PyTorch comparison: `x.permute(2,0,1)` for a 3D tensor.
-/
def transpose3dLastToFirst {α : Type} (s : SessionIR α) [Zero α] [DecidableEq Shape]
  {a b c : Nat} (x : TensorRef α (.dim a (.dim b (.dim c .scalar)))) :
  IO (TensorRef α (.dim c (.dim a (.dim b .scalar)))) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim c (.dim a (.dim b .scalar)))) (fun {Γ} {ss} xv nat
    g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.transpose3dLastToFirst (α := α) (Γ := Γ) (a := a) (b :=
        b) (c := c) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Swap the last two axes of a 3D tensor: `(a,b,c) → (a,c,b)`.

PyTorch comparison: `x.transpose(1,2)` for a 3D tensor.
-/
def transpose3dLastTwo {α : Type} (s : SessionIR α) [Zero α] [DecidableEq Shape]
  {a b c : Nat} (x : TensorRef α (.dim a (.dim b (.dim c .scalar)))) :
  IO (TensorRef α (.dim a (.dim c (.dim b .scalar)))) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim a (.dim c (.dim b .scalar)))) (fun {Γ} {ss} xv nat
    g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.transpose3dLastTwo (α := α) (Γ := Γ) (a := a) (b := b) (c
        := c) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Swap two adjacent axes at a given `depth` inside the shape.

This is a more general permutation helper used in some shape-manipulating models.
PyTorch comparison: like `x.transpose(dim, dim+1)` for a suitably chosen `dim`.
-/
def swapAdjacentAtDepth {α : Type} (s : SessionIR α) [Context α] [DecidableEq Shape]
  {sh : Shape} (depth : Nat) (x : TensorRef α sh) : IO (TensorRef α (sh.swapAdjacentAtDepth depth))
    :=
  commitGraphM (α := α) s (β := TensorRef α (sh.swapAdjacentAtDepth depth)) (fun {Γ} {ss} xv nat g
    => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.swapAdjacentAtDepth (α := α) (Γ := Γ) (s := sh) depth { id
        := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Broadcast a tensor to a larger shape.

The witness `cb : Shape.CanBroadcastTo sh1 sh2` encodes the broadcasting compatibility proof.
PyTorch comparison: `x.expand(...)` / implicit broadcasting.
-/
def broadcastTo {α : Type} (s : SessionIR α) [Inhabited α] [Add α] [Zero α] [DecidableEq Shape]
  {sh1 sh2 : Shape} (cb : Shape.CanBroadcastTo sh1 sh2) (x : TensorRef α sh1) : IO (TensorRef α sh2)
    :=
  commitGraphM (α := α) s (β := TensorRef α sh2) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.broadcastTo (α := α) (Γ := Γ) (s₁ := sh1) (s₂ := sh2) cb {
        id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Sum-reduce along `axis`.

PyTorch comparison: `torch.sum(x, dim=axis)`.
-/
def reduceSum {α : Type} (s : SessionIR α) [Add α] [Zero α] [Inhabited α] [DecidableEq Shape]
  {sh : Shape} (axis : Nat) [valid : Shape.valid_axis_inst axis sh] [wf : Shape.WellFormed sh]
  (x : TensorRef α sh) : IO (TensorRef α (shapeAfterSum sh axis)) :=
  commitGraphM (α := α) s (β := TensorRef α (shapeAfterSum sh axis)) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.reduceSum (α := α) (Γ := Γ) (s := sh) axis { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Mean-reduce along `axis`.

PyTorch comparison: `torch.mean(x, dim=axis)`.
-/
def reduceMean {α : Type} (s : SessionIR α) [Context α] [Inhabited α] [DecidableEq Shape]
  {sh : Shape} (axis : Nat) [valid : Shape.valid_axis_inst axis sh] [wf : Shape.WellFormed sh]
  (x : TensorRef α sh) : IO (TensorRef α (shapeAfterSum sh axis)) :=
  commitGraphM (α := α) s (β := TensorRef α (shapeAfterSum sh axis)) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.reduceMean (α := α) (Γ := Γ) (s := sh) axis { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Gather a single scalar `x[i]` from a 1D vector, with a compile-time `Fin n` index.

PyTorch comparison: `x[i]` for a 1D tensor.
-/
def gatherScalar {α : Type} (s : SessionIR α) [Zero α] [DecidableEq Shape]
  {n : Nat} (x : TensorRef α (.dim n .scalar)) (i : Fin n) : IO (TensorRef α Shape.scalar) :=
  commitGraphM (α := α) s (β := TensorRef α Shape.scalar) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.gatherScalar (α := α) (Γ := Γ) (n := n) { id := x.id } i)
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Gather a row `x[i]` from a 2D tensor, with a compile-time `Fin rows` index.

PyTorch comparison: `x[i]` for a 2D tensor (row indexing).
-/
def gatherRow {α : Type} (s : SessionIR α) [Zero α] [DecidableEq Shape]
  {rows cols : Nat} (x : TensorRef α (.dim rows (.dim cols .scalar))) (i : Fin rows) :
  IO (TensorRef α (.dim cols .scalar)) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim cols .scalar)) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.gatherRow (α := α) (Γ := Γ) (rows := rows) (cols := cols) {
        id := x.id } i)
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Read a `Nat` from the nat-environment.

Out-of-bounds reads return `0` (total function), which is convenient for modeling "possibly invalid"
indices without throwing.
-/
def natAt (d : NatEnv) (id : Nat) : Nat :=
  match d[id]? with
  | some v => v
  | none => 0

/--
Read a length-`k` vector of `Nat`s starting at `start` from the nat-environment.

Out-of-bounds reads fall back to `0` elementwise via `natAt`.
-/
def natVecAt {k : Nat} (d : NatEnv) (start : Nat) : Tensor Nat (.dim k .scalar) :=
  Tensor.dim (fun i => Tensor.scalar (natAt d (start + i.val)))

/--
Dynamic gather of a scalar from a 1D vector using a runtime `NatRef` index.

Out-of-range indices produce `0` instead of raising.
PyTorch comparison: similar to `x[i]` where `i` is a Python integer, except PyTorch raises on
out-of-range while this definition totalizes the behavior for ease of reasoning.
-/
def gatherScalarRef {α : Type} (s : SessionIR α) [Zero α] [DecidableEq Shape]
  {n : Nat} (x : TensorRef α (.dim n .scalar)) (i : NatRef) : IO (TensorRef α Shape.scalar) :=
  commitGraphM (α := α) s (β := TensorRef α Shape.scalar) (fun {Γ} {ss} xv nat g => do
    let ix ← mkIdxOrThrow (_α := α) (Γ := Γ) (ss := ss) x.id (.dim n .scalar)
    let node : _root_.Proofs.Autograd.Algebra.NodeData α NatEnv (Γ ++ ss) Shape.scalar :=
      { forward := fun ctx d =>
          let xv := _root_.Proofs.Autograd.Algebra.getIdx (α := α) (xs := ctx) ix
          let j := natAt d i.id
          if hj : j < n then
            getAtSpec xv ⟨j, hj⟩
          else
            Tensor.scalar 0
        jvp := fun _ _ _d => fill (0 : α) Shape.scalar
        vjp := fun _ctx d δ =>
          let gVal : α := Tensor.toScalar δ
          let j := natAt d i.id
          if _hj : j < n then
            let dx : Tensor α (.dim n .scalar) :=
              Tensor.dim (fun k => Tensor.scalar (if decide (k.val = j) then gVal else 0))
            _root_.Proofs.Autograd.Algebra.TList.single (α := α) (Γ := Γ ++ ss) (s := .dim n
              .scalar) ix dx
          else
            _root_.Proofs.Autograd.Algebra.TList.zero (α := α) (ss := Γ ++ ss) }
    let outId : Nat := Γ.length + ss.length
    let ss' : List Shape := ss ++ [Shape.scalar]
    let g' : _root_.Proofs.Autograd.Algebra.GraphData α NatEnv Γ ss' := .snoc g node
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := outId }, st1))

/--
Dynamic gather of a row from a 2D tensor using a runtime `NatRef` index.

Out-of-range indices yield a zero row.
PyTorch comparison: similar to `x[i]` for 2D tensors with runtime `i`, but PyTorch raises on
out-of-range whereas this definition is totalized for ease of reasoning.
-/
def gatherRowRef {α : Type} (s : SessionIR α) [Zero α] [DecidableEq Shape]
  {rows cols : Nat} (x : TensorRef α (.dim rows (.dim cols .scalar))) (i : NatRef) :
  IO (TensorRef α (.dim cols .scalar)) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim cols .scalar)) (fun {Γ} {ss} xv nat g => do
    let ix ← mkIdxOrThrow (_α := α) (Γ := Γ) (ss := ss) x.id (.dim rows (.dim cols .scalar))
    let outS : Shape := .dim cols .scalar
    let inS : Shape := .dim rows (.dim cols .scalar)
    let node : _root_.Proofs.Autograd.Algebra.NodeData α NatEnv (Γ ++ ss) outS :=
      { forward := fun ctx d =>
          let xv := _root_.Proofs.Autograd.Algebra.getIdx (α := α) (xs := ctx) ix
          let j := natAt d i.id
          if hj : j < rows then
            getAtSpec xv ⟨j, hj⟩
          else
            fill (0 : α) outS
        jvp := fun _ _ _d => fill (0 : α) outS
        vjp := fun _ctx d δ =>
          let j := natAt d i.id
          let dx : Tensor α inS :=
            if _hj : j < rows then
              Tensor.dim (fun r =>
                if decide (r.val = j) then
                  δ
                else
                  fill (0 : α) outS)
            else
              fill (0 : α) inS
          _root_.Proofs.Autograd.Algebra.TList.single (α := α) (Γ := Γ ++ ss) (s := inS) ix dx }
    let outId : Nat := Γ.length + ss.length
    let ss' : List Shape := ss ++ [outS]
    let g' : _root_.Proofs.Autograd.Algebra.GraphData α NatEnv Γ ss' := .snoc g node
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := outId }, st1))

/--
Dynamic gather of `k` scalars from a 1D tensor using a runtime `NatVecRef k` of indices.

Out-of-range indices yield `0`. In the VJP, gradients are accumulated for repeated indices
(i.e. it behaves like a gather followed by a scatter-add back into the source vector).
PyTorch comparison: related to `torch.gather` / advanced indexing, but with totalized out-of-range
behavior.
-/
def gatherVecRef {α : Type} (s : SessionIR α) [Add α] [Zero α] [DecidableEq Shape]
  {n k : Nat} (x : TensorRef α (.dim n .scalar)) (idx : NatVecRef k) :
  IO (TensorRef α (.dim k .scalar)) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim k .scalar)) (fun {Γ} {ss} xv nat g => do
    let ix ← mkIdxOrThrow (_α := α) (Γ := Γ) (ss := ss) x.id (.dim n .scalar)
    let outS : Shape := .dim k .scalar
    let inS : Shape := .dim n .scalar
    let node : _root_.Proofs.Autograd.Algebra.NodeData α NatEnv (Γ ++ ss) outS :=
      { forward := fun ctx d =>
          let xv := _root_.Proofs.Autograd.Algebra.getIdx (α := α) (xs := ctx) ix
          let idxT := natVecAt (k := k) d idx.start
          match idxT with
          | Tensor.dim f =>
              Tensor.dim (fun j =>
                match f j with
                | Tensor.scalar ij =>
                    if h : ij < n then
                      getAtSpec xv ⟨ij, h⟩
                    else
                      Tensor.scalar 0)
        jvp := fun _ _ _d => fill (0 : α) outS
        vjp := fun _ctx d δ =>
          let idxT := natVecAt (k := k) d idx.start
          let dx : Tensor α inS :=
            Tensor.dim (fun iFin =>
              let sum : α :=
                (List.finRange k).foldl (fun acc j =>
                  let ij :=
                    match getAtSpec idxT j with
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
          _root_.Proofs.Autograd.Algebra.TList.single (α := α) (Γ := Γ ++ ss) (s := inS) ix dx }
    let outId : Nat := Γ.length + ss.length
    let ss' : List Shape := ss ++ [outS]
    let g' : _root_.Proofs.Autograd.Algebra.GraphData α NatEnv Γ ss' := .snoc g node
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := outId }, st1))

/--
Dynamic gather of `k` rows from a 2D tensor using a runtime `NatVecRef k` of row indices.

Out-of-range indices yield zero rows. In the VJP, gradients are accumulated into the selected
rows (scatter-add semantics), including accumulation for repeated indices.
PyTorch comparison: similar to `torch.index_select(x, dim=0, index=...)` or advanced indexing on
the first dimension, but with totalized out-of-range behavior.
-/
def gatherRowsRef {α : Type} (s : SessionIR α) [Add α] [Zero α] [DecidableEq Shape]
  {rows cols k : Nat} (x : TensorRef α (.dim rows (.dim cols .scalar))) (idx : NatVecRef k) :
  IO (TensorRef α (.dim k (.dim cols .scalar))) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim k (.dim cols .scalar))) (fun {Γ} {ss} xv nat g =>
    do
    let ix ← mkIdxOrThrow (_α := α) (Γ := Γ) (ss := ss) x.id (.dim rows (.dim cols .scalar))
    let outS : Shape := .dim k (.dim cols .scalar)
    let inS : Shape := .dim rows (.dim cols .scalar)
    let rowS : Shape := .dim cols .scalar
    let node : _root_.Proofs.Autograd.Algebra.NodeData α NatEnv (Γ ++ ss) outS :=
      { forward := fun ctx d =>
          let xv := _root_.Proofs.Autograd.Algebra.getIdx (α := α) (xs := ctx) ix
          let idxT := natVecAt (k := k) d idx.start
          match idxT with
          | Tensor.dim f =>
              Tensor.dim (fun j =>
                match f j with
                | Tensor.scalar ij =>
                    if h : ij < rows then
                      getAtSpec xv ⟨ij, h⟩
                    else
                      fill (0 : α) rowS)
        jvp := fun _ _ _d => fill (0 : α) outS
        vjp := fun _ctx d δ =>
          let idxT := natVecAt (k := k) d idx.start
          let dx : Tensor α inS :=
            Tensor.dim (fun rFin =>
              let rowGrad : Tensor α rowS :=
                (List.finRange k).foldl (fun acc j =>
                  let ij :=
                    match getAtSpec idxT j with
                    | Tensor.scalar v => v
                  if _hij : ij < rows then
                    if decide (ij = rFin.val) then
                      addSpec acc (getAtSpec δ j)
                    else acc
                  else acc
                ) (fill (0 : α) rowS)
              rowGrad)
          _root_.Proofs.Autograd.Algebra.TList.single (α := α) (Γ := Γ ++ ss) (s := inS) ix dx }
    let outId : Nat := Γ.length + ss.length
    let ss' : List Shape := ss ++ [outS]
    let g' : _root_.Proofs.Autograd.Algebra.GraphData α NatEnv Γ ss' := .snoc g node
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := outId }, st1))

/--
Gather a scalar from a 1D vector using a raw `Nat` index.

PyTorch comparison: like `x[i]` with an integer index, but this operation is recorded into the
proved IR (so it is stable for compilation/verification).
-/
def gatherScalarNat {α : Type} (s : SessionIR α) [Zero α] [DecidableEq Shape]
  {n : Nat} (x : TensorRef α (.dim n .scalar)) (i : Nat) : IO (TensorRef α Shape.scalar) :=
  commitGraphM (α := α) s (β := TensorRef α Shape.scalar) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.gatherScalarNat (α := α) (Γ := Γ) (n := n) { id := x.id }
        i)
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Gather `k` scalars from a 1D vector using an explicit index tensor.

PyTorch comparison: related to `torch.gather` / advanced indexing with an integer index tensor.
-/
def gatherVecNat {α : Type} (s : SessionIR α) [Add α] [Zero α] [DecidableEq Shape]
  {n k : Nat} (x : TensorRef α (.dim n .scalar)) (idx : Tensor Nat (.dim k .scalar)) :
  IO (TensorRef α (.dim k .scalar)) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim k .scalar)) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.gatherVecNat (α := α) (Γ := Γ) (n := n) (k := k) { id :=
        x.id } idx)
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Gather `k` rows from a 2D tensor using an explicit index tensor.

PyTorch comparison: similar to `torch.index_select(x, dim=0, index=...)` or advanced indexing.
-/
def gatherRowsNat {α : Type} (s : SessionIR α) [Add α] [Zero α] [DecidableEq Shape]
  {rows cols k : Nat} (x : TensorRef α (.dim rows (.dim cols .scalar))) (idx : Tensor Nat (.dim k
    .scalar)) :
  IO (TensorRef α (.dim k (.dim cols .scalar))) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim k (.dim cols .scalar))) (fun {Γ} {ss} xv nat g =>
    do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.gatherRowsNat (α := α) (Γ := Γ) (rows := rows) (cols :=
        cols) (k := k) { id := x.id } idx)
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Scatter-add into a vector: return a copy of `x` with `x[i] += v`.

PyTorch comparison: similar to `x.scatter_add_(dim=0, index=..., src=...)` in spirit, but this is
functional (returns a new tensor) and uses a single `Fin n` index.
-/
def scatterAddVec {α : Type} (s : SessionIR α) [Add α] [Zero α] [DecidableEq Shape]
  {n : Nat} (x : TensorRef α (.dim n .scalar)) (v : TensorRef α Shape.scalar) (i : Fin n) :
  IO (TensorRef α (.dim n .scalar)) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim n .scalar)) (fun {Γ} {ss} xv nat g => do
    let (out, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.scatterAddVec (α := α) (Γ := Γ) (n := n) { id := x.id } {
        id := v.id } i)
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := out.id }, st1))

/--
Scatter-add into a matrix row: return a copy of `x` with `x[i, :] += v`.

PyTorch comparison: like adding a row vector into a selected row (functional analogue of an
in-place indexed add).
-/
def scatterAddRow {α : Type} (s : SessionIR α) [Add α] [Zero α] [DecidableEq Shape]
  {rows cols : Nat}
  (x : TensorRef α (.dim rows (.dim cols .scalar))) (v : TensorRef α (.dim cols .scalar)) (i : Fin
    rows) :
  IO (TensorRef α (.dim rows (.dim cols .scalar))) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim rows (.dim cols .scalar))) (fun {Γ} {ss} xv nat g
    => do
    let (out, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.scatterAddRow (α := α) (Γ := Γ) (rows := rows) (cols :=
        cols) { id := x.id } { id := v.id } i)
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := out.id }, st1))

/--
Record elementwise logistic sigmoid.

PyTorch comparison: `torch.sigmoid(x)`.
-/
def sigmoid {α : Type} (s : SessionIR α) [Context α] [Zero α] [DecidableEq Shape] {sh : Shape}
  (x : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.sigmoid (α := α) (Γ := Γ) (s := sh) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record elementwise hyperbolic tangent.

PyTorch comparison: `torch.tanh(x)`.
-/
def tanh {α : Type} (s : SessionIR α) [Context α] [Zero α] [DecidableEq Shape] {sh : Shape}
  (x : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.tanh (α := α) (Γ := Γ) (s := sh) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record softmax (shape-preserving).

PyTorch comparison: `torch.softmax(x, dim=...)`. This helper uses the convention baked into the
underlying `GraphM.softmax` implementation.
-/
def softmax {α : Type} (s : SessionIR α) [Context α] [Zero α] [DecidableEq Shape] {sh : Shape}
  (x : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.softmax (α := α) (Γ := Γ) (s := sh) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record stable log-softmax in the linked compiled session.

This commits a single `GraphM.logSoftmax` node instead of expanding to `softmax` followed by
`log`, so compiled execution keeps the same stable semantics as eager CPU/CUDA.
-/
def logSoftmax {α : Type} (s : SessionIR α) [Context α] [Zero α] [DecidableEq Shape] {sh : Shape}
  (x : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.logSoftmax (α := α) (Γ := Γ) (s := sh) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record elementwise softplus.

PyTorch comparison: `torch.nn.functional.softplus(x)`.
-/
def softplus {α : Type} (s : SessionIR α) [Context α] [Zero α] [DecidableEq Shape] {sh : Shape}
  (x : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.softplus (α := α) (Γ := Γ) (s := sh) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record elementwise exponential.

PyTorch comparison: `torch.exp(x)`.
-/
def exp {α : Type} (s : SessionIR α) [Context α] [Zero α] [DecidableEq Shape] {sh : Shape}
  (x : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.exp (α := α) (Γ := Γ) (s := sh) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record elementwise natural logarithm.

PyTorch comparison: `torch.log(x)`.
-/
def log {α : Type} (s : SessionIR α) [Context α] [Zero α] [DecidableEq Shape] {sh : Shape}
  (x : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.log (α := α) (Γ := Γ) (s := sh) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record elementwise log with epsilon guard.

This is intended for numerically stable losses; it corresponds roughly to `log(max(x, ε))`.
PyTorch comparison: `torch.log(torch.clamp(x, min=ε))`.
-/
def safeLog {α : Type} (s : SessionIR α) [Context α] [Zero α] [DecidableEq Shape] {sh : Shape}
  (x : TensorRef α sh) (ε : α := Numbers.epsilon) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.safeLog (α := α) (Γ := Γ) (s := sh) { id := x.id } (ε :=
        ε))
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Sum-reduce all elements to a scalar.

PyTorch comparison: `x.sum()`.
-/
def sum {α : Type} (s : SessionIR α) [Context α] [DecidableEq Shape] {sh : Shape}
  (x : TensorRef α sh) : IO (TensorRef α Shape.scalar) :=
  commitGraphM (α := α) s (β := TensorRef α Shape.scalar) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.sum (α := α) (Γ := Γ) (s := sh) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record a fully-connected linear layer: `y = w • x + b`.

Type-level shapes enforce `w : (outDim, inDim)`, `b : (outDim,)`, and `x : (inDim,)`.
PyTorch comparison: `torch.nn.functional.linear(x, weight=w, bias=b)` (with the same weight layout).
-/
def linear {α : Type} (s : SessionIR α) [Add α] [Mul α] [Zero α] [DecidableEq Shape]
  {inDim outDim : Nat}
  (w : TensorRef α (.dim outDim (.dim inDim .scalar)))
  (b : TensorRef α (.dim outDim .scalar))
  (x : TensorRef α (.dim inDim .scalar)) : IO (TensorRef α (.dim outDim .scalar)) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim outDim .scalar)) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.linear (α := α) (Γ := Γ)
        (inDim := inDim) (outDim := outDim) { id := w.id } { id := b.id } { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Mean-squared-error loss returning a scalar.

PyTorch comparison: `torch.nn.functional.mse_loss(yhat, target, reduction="mean")`.
-/
def mseLoss {α : Type} (s : SessionIR α)
  [Add α] [Sub α] [Mul α] [Div α] [Zero α] [One α] [Coe Nat α] [DecidableEq Shape]
  {sh : Shape} (yhat target : TensorRef α sh) : IO (TensorRef α Shape.scalar) :=
  commitGraphM (α := α) s (β := TensorRef α Shape.scalar) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.mseLoss (α := α) (Γ := Γ) (s := sh) { id := yhat.id } { id
        := target.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Layer normalization over the trailing embedding dimension.

This variant is specialized to 2D tensors of shape `(seqLen, embedDim)` and expects positive
dimensions for numerical stability and well-formedness.
PyTorch comparison: `torch.nn.LayerNorm(embedDim)` (applied per token), or
`torch.nn.functional.layer_norm`.
-/
def layerNorm {α : Type} (s : SessionIR α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
  (x : TensorRef α (.dim seqLen (.dim embedDim .scalar)))
  (gamma : TensorRef α (.dim embedDim .scalar))
  (beta : TensorRef α (.dim embedDim .scalar)) : IO (TensorRef α (.dim seqLen (.dim embedDim
    .scalar))) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim seqLen (.dim embedDim .scalar))) (fun {Γ} {ss} xv
    nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.layerNorm (α := α) (Γ := Γ)
        (seqLen := seqLen) (embedDim := embedDim) (h_seq_pos := h_seq_pos) (h_embed_pos :=
          h_embed_pos)
        { id := x.id } { id := gamma.id } { id := beta.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Batch normalization for a channel-first image `(C,H,W)` (no batch axis).

`gamma` and `beta` are per-channel scale/shift parameters.
PyTorch comparison: `torch.nn.BatchNorm2d(C)` (conceptually), or `torch.nn.functional.batch_norm`
specialized to a single "batch element" with NCHW layout.
-/
def batchnormChannelFirst {α : Type} (s : SessionIR α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {channels height width : Nat} (h_c : channels > 0) (h_h : height > 0) (h_w : width > 0)
  (x : TensorRef α (.dim channels (.dim height (.dim width .scalar))))
  (gamma : TensorRef α (.dim channels .scalar))
  (beta : TensorRef α (.dim channels .scalar)) :
  IO (TensorRef α (.dim channels (.dim height (.dim width .scalar)))) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim channels (.dim height (.dim width .scalar)))) (fun
    {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.batchnormChannelFirst (α := α) (Γ := Γ)
        (channels := channels) (height := height) (width := width) (h_c := h_c) (h_h := h_h) (h_w :=
          h_w)
        { id := x.id } { id := gamma.id } { id := beta.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
N-D convolution for channels-first tensors `(inC, spatial...)` (no batch axis).

Kernel layout is `(outC, inC, kernelSpatial...)`, bias is `(outC)`.

PyTorch comparison: `torch.nn.functional.conv{d}d` specialized to a single sample.
-/
def conv {α : Type} (s : SessionIR α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {d inC outC : Nat}
  {kernel stride padding : Vector Nat d}
  {inSpatial : Vector Nat d}
  {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (w : TensorRef α (Shape.ofList (outC :: inC :: kernel.toList)))
  (b : TensorRef α (.dim outC .scalar))
  (x : TensorRef α (Shape.ofList (inC :: inSpatial.toList))) :
  IO (TensorRef α
    (Shape.ofList (outC :: (Spec.convOutSpatial inSpatial kernel stride padding).toList))) :=
  commitGraphM (α := α) s
    (β := TensorRef α
      (Shape.ofList (outC :: (Spec.convOutSpatial inSpatial kernel stride padding).toList)))
    (fun {Γ} {ss} xv nat g => do
      let (v, st') ← runGraphM (α := α) (Γ := Γ)
        (Runtime.Autograd.Compiled.GraphM.conv (α := α) (Γ := Γ)
          (d := d) (inC := inC) (outC := outC)
          (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
          (hInC := hInC) (hKernel := hKernel)
          { id := w.id } { id := b.id } { id := x.id })
        ss g
      let ⟨ss', g'⟩ := st'
      let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
      pure ({ id := v.id }, st1))

/--
N-D transpose convolution for channels-first tensors `(inC, spatial...)` (no batch axis).

Kernel layout is `(inC, outC, kernelSpatial...)` (PyTorch convention), bias is `(outC)`.

PyTorch comparison: `torch.nn.functional.conv_transpose{d}d` specialized to a single sample.
-/
def convTranspose {α : Type} (s : SessionIR α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {d inC outC : Nat}
  {kernel stride padding : Vector Nat d}
  {inSpatial : Vector Nat d}
  {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (w : TensorRef α (Shape.ofList (inC :: outC :: kernel.toList)))
  (b : TensorRef α (.dim outC .scalar))
  (x : TensorRef α (Shape.ofList (inC :: inSpatial.toList))) :
  IO (TensorRef α
    (Shape.ofList (outC ::
      (Spec.convTransposeOutSpatial inSpatial kernel stride padding).toList))) :=
  commitGraphM (α := α) s
    (β := TensorRef α
      (Shape.ofList (outC :: (Spec.convTransposeOutSpatial inSpatial kernel stride padding).toList)))
    (fun {Γ} {ss} xv nat g => do
      let (v, st') ← runGraphM (α := α) (Γ := Γ)
        (Runtime.Autograd.Compiled.GraphM.convTranspose (α := α) (Γ := Γ)
          (d := d) (inC := inC) (outC := outC)
          (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
          (hInC := hInC) (hKernel := hKernel)
          { id := w.id } { id := b.id } { id := x.id })
        ss g
      let ⟨ss', g'⟩ := st'
      let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
      pure ({ id := v.id }, st1))

/--
2D convolution for channel-first images `(inC, inH, inW)` (no batch axis).

Type-level shapes fix the kernel layout `(outC, inC, kH, kW)` and output spatial dimensions derived
from `stride` and `padding`.
PyTorch comparison: `torch.nn.functional.conv2d` (conceptually), specialized to a single image.
-/
def conv2d {α : Type} (s : SessionIR α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (kernel : TensorRef α (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))))
  (bias : TensorRef α (.dim outC .scalar))
  (input : TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (TensorRef α (.dim outC (.dim ((inH + 2 * padding - kH) / stride + 1) (.dim ((inW + 2 * padding
    - kW) / stride + 1) .scalar)))) :=
  commitGraphM (α := α) s
    (β := TensorRef α (.dim outC (.dim ((inH + 2 * padding - kH) / stride + 1) (.dim ((inW + 2 *
      padding - kW) / stride + 1) .scalar))))
    (fun {Γ} {ss} xv nat g => do
      let (v, st') ← runGraphM (α := α) (Γ := Γ)
        (Runtime.Autograd.Compiled.GraphM.conv2d (α := α) (Γ := Γ)
          (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
          (inH := inH) (inW := inW) (h1 := h1) (h2 := h2) (h3 := h3)
          { id := kernel.id } { id := bias.id } { id := input.id })
        ss g
      let ⟨ss', g'⟩ := st'
      let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
      pure ({ id := v.id }, st1))

/--
2D transpose convolution for channel-first images `(inC, inH, inW)` (no batch axis).

Kernel layout matches the spec/PyTorch convention `(inC, outC, kH, kW)`.
PyTorch comparison: `torch.nn.functional.conv_transpose2d` specialized to a single image.
-/
def convTranspose2d {α : Type} (s : SessionIR α) [Context α]
  [DecidableEq Shape]
  {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (kernel : TensorRef α (.dim inC (.dim outC (.dim kH (.dim kW .scalar)))))
  (bias : TensorRef α (.dim outC .scalar))
  (input : TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (TensorRef α (.dim outC (.dim ((inH - 1) * stride - 2 * padding + kH)
    (.dim ((inW - 1) * stride - 2 * padding + kW) .scalar)))) :=
  commitGraphM (α := α) s
    (β := TensorRef α (.dim outC (.dim ((inH - 1) * stride - 2 * padding + kH)
      (.dim ((inW - 1) * stride - 2 * padding + kW) .scalar))))
    (fun {Γ} {ss} xv nat g => do
      let (v, st') ← runGraphM (α := α) (Γ := Γ)
        (Runtime.Autograd.Compiled.GraphM.convTranspose2d (α := α) (Γ := Γ)
          (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
          (inH := inH) (inW := inW) (h1 := h1) (h2 := h2) (h3 := h3)
          { id := kernel.id } { id := bias.id } { id := input.id })
        ss g
      let ⟨ss', g'⟩ := st'
      let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
      pure ({ id := v.id }, st1))

/--
Multi-head self-attention.

This is a shape-specialized attention primitive used by some demo transformer-style models:
- input `x` has shape `(n, dModel)`
- `wq`, `wk`, `wv` map `dModel → numHeads*headDim`
- `wo` maps `numHeads*headDim → dModel`
- optional `mask` is a boolean `(n,n)` attention mask

PyTorch comparison: similar to `torch.nn.MultiheadAttention` / scaled dot-product attention, but
encoded in a fully typed IR for compilation/proof linkage.
-/
def multiHeadAttention {α : Type} (s : SessionIR α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (wq : TensorRef α (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wk : TensorRef α (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wv : TensorRef α (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wo : TensorRef α (.dim (numHeads * headDim) (.dim dModel .scalar)))
  (x : TensorRef α (.dim n (.dim dModel .scalar)))
  (mask : Option (Tensor Bool (.dim n (.dim n .scalar))) := none) :
  IO (TensorRef α (.dim n (.dim dModel .scalar))) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim n (.dim dModel .scalar))) (fun {Γ} {ss} xv nat g
    => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.multiHeadAttention (α := α) (Γ := Γ)
        (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim) (h1 := h1)
        { id := wq.id } { id := wk.id } { id := wv.id } { id := wo.id } { id := x.id } (mask :=
          mask))
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/-! ## Backward + SGD (runtime tape loop on the compiled tape) -/

/--
Compile the recorded proved graph into a runtime tape.

This uses `Graph.compileAuxData` (the same compiler used by the proof pipeline) and extracts the
runtime tape component.
-/
def compileTape {α : Type} [DecidableEq Shape]
    (st : SessionIRState α) : Runtime.Autograd.Tape α :=
  (Proofs.Autograd.Algebra.Graph.compileAuxData (α := α) (Δ := NatEnv) (Γ := st.Γ) (ss := st.ss)
    st.g st.x st.nat).1

/--
Run reverse-mode backprop for the whole recorded context and return a dense gradient array.

`seed` is the upstream gradient for `out` (same convention as PyTorch's
  `loss.backward(gradient=...)`).
-/
def backwardDenseAll {α : Type} (s : SessionIR α) [Add α] [Zero α] [DecidableEq Shape]
  {sh : Shape} (out : TensorRef α sh) (seed : Tensor α sh) :
  IO (Array (Runtime.AnyTensor α)) := do
  let st0 ← s.st.get
  let t := compileTape (α := α) (st := st0)
  okOrThrow (Runtime.Autograd.Tape.backwardDenseAll (t := t) (outId := out.id)
    (seed := Runtime.Autograd.AnyTensor.mk seed))

/--
Convenience wrapper for scalar losses: run backward with seed `1`.

PyTorch comparison: `loss.backward()` for a scalar loss.
-/
def backwardScalarDenseAll {α : Type} (s : SessionIR α) [Add α] [Zero α] [One α] [DecidableEq Shape]
  (loss : TensorRef α Shape.scalar) : IO (Array (Runtime.AnyTensor α)) :=
  backwardDenseAll (α := α) s (sh := Shape.scalar) loss (Tensor.scalar (1 : α))

/--
Extract the gradient tensor for a particular `TensorRef` from a dense gradient array.

This is the typed analogue of looking up `grads[x.id]` and casting it to the expected shape.
-/
def grad {α : Type} {sh : Shape} [DecidableEq Shape]
  (grads : Array (Runtime.AnyTensor α)) (x : TensorRef α sh) : IO (Tensor α sh) := do
  let gAny ← match grads[x.id]? with
    | some g => pure g
    | none => throw <| IO.userError "torch(SessionIR): gradient array out of bounds"
    if h : gAny.s = sh then
      pure (Tensor.castShape gAny.t h)
    else
      throw <| IO.userError <|
        s!"torch(SessionIR): grad shape mismatch (expected {Shape.pretty sh}, got "
          ++ s!"{Shape.pretty gAny.s})"

/-! ## Forward-mode: JVP (compiled only) -/

/-- Like `mkIdxOrThrow`, but restricted to leaves `Γ` only. -/
def mkLeafIdxOrThrow {_α : Type} {Γ : List Shape} (id : Nat) (s : Shape) :
    Runtime.Autograd.Result (_root_.Proofs.Autograd.Algebra.Idx Γ s) := by
    if h : id < Γ.length then
      let fin : Fin Γ.length := ⟨id, h⟩
      let got : Shape := Γ.get fin
      if hg : got = s then
        exact .ok ⟨fin, hg⟩
      else
        exact .error <|
          s!"torch(SessionIR): leaf shape mismatch at id={id}: expected {Shape.pretty s}, got "
            ++ s!"{Shape.pretty got}"
  else
    exact .error s!"torch(SessionIR): invalid leaf id={id} for leafLen={Γ.length}"

/--
Convert a dense tangent array (aligned with leaf creation order) into a typed `TList α Γ`.

This is the main adapter needed to call the proved `GraphData.jvpCtx` forward-mode routine.
-/
def dxTListFromAnyArray {α : Type} [Zero α] [DecidableEq Shape]
    (Γ : List Shape) (dxs : Array (Runtime.AnyTensor α)) :
    IO (_root_.Proofs.Autograd.Algebra.TList α Γ) := do
  if _hlen : dxs.size = Γ.length then
    let rec go : (Γ' : List Shape) → (off : Nat) → IO (_root_.Proofs.Autograd.Algebra.TList α Γ')
      | [], _ => pure .nil
      | s :: ss, off => do
          let any ← match dxs[off]? with
            | some v => pure v
            | none => throw <| IO.userError "torch(SessionIR): dx array out of bounds"
            if hs : any.s = s then
              let t : Tensor α s := Tensor.castShape any.t hs
              pure (.cons t (← go ss (off + 1)))
            else
              throw <| IO.userError <|
                s!"torch(SessionIR): dx shape mismatch at idx={off} (expected "
                  ++ s!"{Shape.pretty s}, got "
                  ++ s!"{Shape.pretty any.s})"
    go Γ 0
  else
    throw <| IO.userError
      s!"torch(SessionIR): dx array size mismatch (expected {Γ.length}, got {dxs.size})"

/--
Jacobian-vector product for the current session snapshot.

`dxs` is a dense array of tangents for leaf tensors, aligned with leaf creation order.
-/
def jvpDenseAll {α : Type} (s : SessionIR α) [Zero α] [DecidableEq Shape]
    {sh : Shape} (out : TensorRef α sh) (dxs : Array (Runtime.AnyTensor α)) :
    IO (Tensor α sh) := do
  let st0 ← s.st.get
  let dx ← dxTListFromAnyArray (α := α) (Γ := st0.Γ) dxs
  let dctx : _root_.Proofs.Autograd.Algebra.TList α (st0.Γ ++ st0.ss) :=
    _root_.Proofs.Autograd.Algebra.GraphData.jvpCtx (α := α) (Δ := NatEnv) (Γ := st0.Γ) (ss :=
      st0.ss)
      st0.g st0.x dx st0.nat
  let idx ← okOrThrow (mkIdxOrThrow (_α := α) (Γ := st0.Γ) (ss := st0.ss) out.id sh)
  pure (_root_.Proofs.Autograd.Algebra.getIdx (α := α) (xs := dctx) idx)

/-- JVP for a single leaf: tangent is nonzero only at `x`. -/
def jvpLeaf {α : Type} (s : SessionIR α) [Zero α] [DecidableEq Shape]
    {shOut shX : Shape}
    (out : TensorRef α shOut) (x : TensorRef α shX) (dx : Tensor α shX) :
    IO (Tensor α shOut) := do
  let st0 ← s.st.get
  let idxX ← okOrThrow (mkLeafIdxOrThrow (_α := α) (Γ := st0.Γ) x.id shX)
  let dxAll : _root_.Proofs.Autograd.Algebra.TList α st0.Γ :=
    _root_.Proofs.Autograd.Algebra.TList.single (α := α) (Γ := st0.Γ) (s := shX) idxX dx
  let dxs : Array (Runtime.AnyTensor α) :=
    _root_.Proofs.Autograd.Algebra.TList.toAnyArray (α := α) (ss := st0.Γ) dxAll
  jvpDenseAll (α := α) (sh := shOut) s out dxs

/-- Scalar-loss JVP for a single leaf. -/
def jvpScalarLeaf {α : Type} (s : SessionIR α) [Zero α] [DecidableEq Shape]
    (loss : TensorRef α Shape.scalar) {shX : Shape} (x : TensorRef α shX) (dx : Tensor α shX) :
    IO α := do
  let dl ← jvpLeaf (α := α) s (shOut := Shape.scalar) (shX := shX) loss x dx
  match dl with
  | .scalar a => pure a

/--
Apply an SGD update to all parameters recorded via `use`.

`grads` is expected to be the dense gradient array returned by `backwardDenseAll` /
`backwardScalarDenseAll`. Only entries corresponding to parameters (leaves that were produced by
`use`) are used to update `Param.value`.
PyTorch comparison: like iterating `params` and doing `p.data -= lr * p.grad`.
-/
def sgdStepAll {α : Type} (s : SessionIR α)
  [Sub α] [Mul α] [Add α] [Zero α] [DecidableEq Shape]
  (lr : α) (grads : Array (Runtime.AnyTensor α)) : IO Unit := do
  let m ← s.paramsByLeaf.get
  for (id, p) in m.toList do
    let gAny ← match grads[id]? with
      | some g => pure g
      | none => throw <| IO.userError "torch(SessionIR): gradient array out of bounds during SGD"
    if hs : gAny.s = p.s then
      let pv ← p.get
      if hp : pv.s = p.s then
        let pvT : Tensor α p.s := Tensor.castShape pv.t hp
        let gT : Tensor α p.s := Tensor.castShape gAny.t hs
        let updated : Tensor α p.s :=
          Tensor.materialize <| subSpec pvT (scaleSpec (α := α) (s := p.s) gT lr)
        p.set (Runtime.Autograd.AnyTensor.mk updated)
      else
        throw <| IO.userError "torch(SessionIR): internal param shape mismatch"
    else
      throw <| IO.userError "torch(SessionIR): internal grad shape mismatch during SGD"

/-! ## Pure correctness hook: session snapshot ↔ proved IR backprop -/

/--
Core proof-link: running the runtime reverse-mode loop on the compiled tape equals proved backprop.

This theorem is the "hook" that lets a session-style API be backed by the proved IR:
`compileAuxData` produces a tape, and `Tape.backwardDenseFrom` is shown equal to
`GraphData.backpropAllCtx` (up to the `TList.toAnyArray` representation change).
-/
theorem backwardDenseFrom_compileAuxData_eq_backpropAllCtx
    {α : Type} [DecidableEq Shape] [CommSemiring α]
    (st : SessionIRState α) (seed : _root_.Proofs.Autograd.Algebra.TList α (st.Γ ++ st.ss)) :
    Runtime.Autograd.Tape.backwardDenseFrom
        (t := (Proofs.Autograd.Algebra.Graph.compileAuxData (α := α) (Δ := NatEnv) (Γ := st.Γ) (ss
          := st.ss) st.g st.x st.nat).1)
        (grads0 := _root_.Proofs.Autograd.Algebra.TList.toAnyArray (α := α) (ss := st.Γ ++ st.ss)
          seed)
      =
      .ok
        (_root_.Proofs.Autograd.Algebra.TList.toAnyArray (α := α) (ss := st.Γ ++ st.ss)
          (_root_.Proofs.Autograd.Algebra.GraphData.backpropAllCtx (α := α) (Δ := NatEnv) (Γ :=
            st.Γ) (ss := st.ss) st.g st.x st.nat seed)) := by
  simpa using
    (Proofs.Autograd.Algebra.Graph.backwardDenseFrom_compileAuxData_eq_backpropAllCtx
      (α := α) (Δ := NatEnv) (Γ := st.Γ) (ss := st.ss) st.g st.x st.nat seed)

end SessionIR

end Internal

/-! ## Public re-exports (stable names for docs) -/

-- The proof-linked session lives under `Internal` to keep the surface area small, but we expose
-- a stable public name layer for the blueprint and for downstream users who want the proved hook.

/-- Public alias for the proof-linked session state (internal definition re-export). -/
abbrev SessionIRState (α : Type) : Type := Internal.SessionIRState α

namespace SessionIRState

/-- Empty `SessionIRState` (no parameters/graph recorded yet). -/
abbrev empty {α : Type} : SessionIRState α := Internal.SessionIRState.empty (α := α)

end SessionIRState

/-- Public alias for the proof-linked session object (internal definition re-export). -/
abbrev SessionIR (α : Type) : Type := Internal.SessionIR α

namespace SessionIR

/-- Create a new proof-linked session (records a graph + supports proved backprop hook). -/
abbrev new {α : Type} (opts : Options := {}) : IO (SessionIR α) :=
  Internal.SessionIR.new (α := α) opts

/--
Compute dense gradients for all tracked refs w.r.t. an output tensor and a seed.

This mirrors the "backward with custom seed" pattern in tensor AD systems.
-/
abbrev backwardDenseAll {α : Type} (s : SessionIR α) [Add α] [Zero α] [DecidableEq Shape]
    {sh : Shape} (out : TensorRef α sh) (seed : Tensor α sh) :
    IO (Array (Runtime.AnyTensor α)) :=
  Internal.SessionIR.backwardDenseAll (α := α) (s := s) out seed

/-- Dense gradients for all tracked refs w.r.t. a scalar loss (seed is implicitly `1`). -/
abbrev backwardScalarDenseAll {α : Type} (s : SessionIR α) [Add α] [Zero α] [One α] [DecidableEq
  Shape]
    (loss : TensorRef α Shape.scalar) : IO (Array (Runtime.AnyTensor α)) :=
  Internal.SessionIR.backwardScalarDenseAll (α := α) (s := s) loss

/-- Extract the gradient tensor for a specific ref from a dense gradient array. -/
abbrev grad {α : Type} {sh : Shape} [DecidableEq Shape]
    (grads : Array (Runtime.AnyTensor α)) (x : TensorRef α sh) : IO (Tensor α sh) :=
  Internal.SessionIR.grad (α := α) (grads := grads) x

end SessionIR

/--
Public proof hook: the runtime reverse-mode loop on the compiled tape equals proved IR backprop.

This is a re-export of the internal theorem so downstream users can cite a stable name.
-/
theorem backwardDenseFrom_compileAuxData_eq_backpropAllCtx
    {α : Type} [DecidableEq Shape] [CommSemiring α]
    (st : SessionIRState α) (seed : _root_.Proofs.Autograd.Algebra.TList α (st.Γ ++ st.ss)) :
    Runtime.Autograd.Tape.backwardDenseFrom
        (t := (Proofs.Autograd.Algebra.Graph.compileAuxData (α := α) (Δ := Array Nat) (Γ := st.Γ)
          (ss := st.ss) st.g st.x st.nat).1)
        (grads0 := _root_.Proofs.Autograd.Algebra.TList.toAnyArray (α := α) (ss := st.Γ ++ st.ss)
          seed)
      =
      .ok
        (_root_.Proofs.Autograd.Algebra.TList.toAnyArray (α := α) (ss := st.Γ ++ st.ss)
          (_root_.Proofs.Autograd.Algebra.GraphData.backpropAllCtx (α := α) (Δ := Array Nat) (Γ :=
            st.Γ) (ss := st.ss) st.g st.x st.nat seed)) := by
  simpa [SessionIRState] using
    (Internal.SessionIR.backwardDenseFrom_compileAuxData_eq_backpropAllCtx (α := α) (st := st) seed)

end Torch
end Autograd
end Runtime
