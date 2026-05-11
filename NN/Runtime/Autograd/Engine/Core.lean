/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Context
public import NN.Spec.Layers.Attention
public import NN.Spec.Layers.Conv
public import NN.Spec.Layers.Linear
public import NN.Spec.Layers.Normalization
public import NN.Spec.Layers.Pooling

/-!
# Engine Core

A small dynamic (DAG) autograd engine.

This is intended to be the "runtime" counterpart to `Spec.OpSpec`: instead of manually writing
backward passes end-to-end, you build a tape during a forward pass and then call `backward`.

Design goals:
- Works for arbitrary `Tensor α s` shapes using `Runtime.AnyTensor` packing.
- Supports DAGs (shared subexpressions): gradients are accumulated by summation.
- Keeps the API compact and shape-safe enough for practical use.

Scope boundaries:
- A fully verified analytic calculus proof (`HasFDerivAt` etc.) for all ops. The engine is
  correct *given* the local backward rules used to build nodes, and we add small regression
  checks in `NN/Tests/Runtime`.
- For a PyTorch-style imperative API over this tape, see `NN.Runtime.Autograd.Torch.Core`.

References (PyTorch / background reading):
- PyTorch docs: `torch.autograd` and "Autograd mechanics":
  https://pytorch.org/docs/stable/autograd.html
  https://pytorch.org/docs/stable/notes/autograd.html
- "micrograd" (small autograd engine, useful for intuition):
  https://github.com/karpathy/micrograd
-/

@[expose] public section


namespace Runtime
namespace Autograd

open Spec
open Tensor

/-!
## Core Types

The eager autograd engine is built out of a few small pieces:

* `Result`: a pure error monad used throughout the tape API.
* `AnyTensor`: a shape-erased tensor used to store heterogeneous node values on a single tape.
* `Node`: one recorded computation step, with a local backward (VJP) rule.
* `Tape`: a grow-only array of nodes; reverse-mode traversals walk it in reverse order.
-/

/--
Runtime error monad for the eager autograd engine.

We use plain `Except String` (instead of `IO` exceptions) so the tape constructors remain pure and
easy to test. Front-ends that prefer exceptions can use `okOrThrow`.
-/
abbrev Result (α : Type) := Except String α

/--
Convert an `Autograd.Result` into an `IO` action by throwing `IO.userError` on failure.

This is mainly used by the imperative Torch/TorchLean front-ends to keep their code readable.
-/
def okOrThrow {α : Type} : Result α → IO α
  | .ok a => pure a
  | .error e => throw <| IO.userError e

namespace AnyTensor

/--
Pack a typed tensor as a runtime `AnyTensor`.

This is the primary bridge between the dependently-typed `Tensor α s` world and the dynamic tape,
which stores heterogeneous shapes in a single array.
-/
def mk {α : Type} {s : Shape} (t : Tensor α s) : Runtime.AnyTensor α :=
  { s := s, t := t }

/--
Cast an `AnyTensor` to a specific shape, given an equality proof.

This is used after dynamic shape checks (e.g. `Tape.requireValue`).
-/
def cast {α : Type} {s₂ : Shape} (t : Runtime.AnyTensor α) (h : t.s = s₂) : Tensor α s₂ :=
  Tensor.castShape t.t h

/--
Accumulate two `AnyTensor` values by elementwise addition, with a dynamic shape check.

This is the heart of DAG support: if two different paths contribute gradients to the same parent,
we sum the contributions.
-/
def add {α : Type} [Add α] [DecidableEq Shape]
  (a b : Runtime.AnyTensor α) : Result (Runtime.AnyTensor α) := by
  if h : a.s = b.s then
    let b' : Tensor α a.s := Tensor.castShape b.t h.symm
    exact .ok { s := a.s, t := addSpec a.t b' }
  else
    exact .error "autograd: gradient shape mismatch during accumulation"

end AnyTensor

/--
A tape node representing a single tensor value in the recorded computation graph.

Fields:
- `value`: the forward value (shape-erased).
- `parents`: ids of parent nodes in the tape.
- `backward`: a local VJP rule. Given an upstream cotangent for `value`, it returns a list of
  `(parentId, parentCotangent)` contributions (one per parent, usually).

PyTorch comparison: analogous to an autograd `Function` instance + saved tensors, but here we store
the backward closure directly.
-/
structure Node (α : Type) where
  /-- Optional node name used for debugging and pretty-printing. -/
  name : Option String := none
  /-- Forward value computed at this node (shape-erased). -/
  value : Runtime.AnyTensor α
  /--
  Whether reverse-mode propagation should visit this node.

  If `false`, reverse-mode traversal skips this node and does not accumulate gradients into it.
  -/
  requires_grad : Bool := true
  /-- Parent node ids (dependencies) in the tape. -/
  parents : List Nat := []
  /--
  Local VJP rule for this node.

  Given an upstream cotangent for `value`, return a list of `(parentId, parentCotangent)`
  contributions. If multiple children contribute to the same parent, the engine will sum
  contributions via `AnyTensor.add`.
  -/
  backward : Runtime.AnyTensor α → Result (List (Nat × Runtime.AnyTensor α))

/--
Autograd tape: a grow-only array of nodes.

Node ids are array indices (`Nat`). All ops append exactly one node and return its id.
This makes it easy to implement reverse-mode by traversing ids in reverse order.
-/
structure Tape (α : Type) where
  /--
  Tape nodes in evaluation order.

  Node ids are array indices (`Nat`). Each tape op appends exactly one node and returns its id.
  -/
  nodes : Array (Node α) := #[]

/-!
## Tape Construction

The `Tape` namespace provides *pure* constructors for building a recorded computation graph.
Each op appends exactly one node and returns the updated tape plus the new node id.

If you prefer an implicit tape-threading style, see `NN.Runtime.Autograd.Engine.TapeM`.
-/

namespace Tape

variable {α : Type}

/-- Empty tape (no nodes). -/
def empty : Tape α := {}

/-- Number of nodes stored in the tape. -/
def size (t : Tape α) : Nat := t.nodes.size

/-- Read a node by id (returns `none` if out of bounds). -/
def getNode? (t : Tape α) (id : Nat) : Option (Node α) :=
  t.nodes[id]?

/-- Read just the stored forward value for a node id. -/
def getValue? (t : Tape α) (id : Nat) : Option (Runtime.AnyTensor α) :=
  (t.getNode? id).map (·.value)

/--
Append a node and return its id.

Invariant: the returned id is `t.size`, the pre-append size of the tape.
-/
def addNode (t : Tape α) (node : Node α) : Tape α × Nat :=
  let id := t.nodes.size
  ({ nodes := t.nodes.push node }, id)

/-- `addNode` returns the current tape size as the fresh node id. -/
@[simp] theorem addNode_id (t : Tape α) (node : Node α) :
    (t.addNode node).2 = t.size := by
  simp [addNode, size]

/-- Appending a node increases the tape size by one. -/
@[simp] theorem size_addNode (t : Tape α) (node : Node α) :
    (t.addNode node).1.size = t.size + 1 := by
  simp [addNode, size]

/--
Add a leaf node (no parents).

PyTorch comparison: a tensor that enters the graph as a leaf (e.g. input or parameter value).
-/
def leaf {α : Type} {s : Shape}
  (t : Tape α) (value : Tensor α s) (name : Option String := none) (requires_grad : Bool := true) :
  Tape α × Nat :=
  t.addNode {
    name := name,
    value := AnyTensor.mk value,
    requires_grad := requires_grad,
    parents := [],
    backward := fun _ => .ok []
  }

/--
Read a typed tensor value from a tape node id.

This is the main "dynamic check" boundary in the eager runtime:
- fails if the id is invalid, or
- fails if the stored runtime shape doesn't match the expected dependent shape `s`.
-/
def requireValue {α : Type} [DecidableEq Shape] {s : Shape}
  (t : Tape α) (id : Nat) : Result (Tensor α s) := by
  match t.getValue? id with
  | none => exact .error "autograd: invalid node id"
  | some any =>
    if h : any.s = s then
      exact .ok (Tensor.castShape any.t h)
    else
      exact .error "autograd: shape mismatch"

/--
Read a typed upstream gradient tensor from a runtime `AnyTensor`.

This is the backward analogue of `Tape.requireValue`: it checks that the upstream gradient has the
expected shape `τ` and then performs the dependent cast.
-/
def requireGrad {α : Type} [DecidableEq Shape] {τ : Shape}
    (dLdyAny : Runtime.AnyTensor α) : Result (Tensor α τ) := by
  if h : dLdyAny.s = τ then
    exact .ok (Tensor.castShape dLdyAny.t h)
  else
    exact .error "autograd: upstream gradient shape mismatch"

/--
Generic constructor for unary ops.

You provide:
- `forward : Tensor α σ → Tensor α τ`
- `backward : Tensor α σ → Tensor α τ → Tensor α σ` (a VJP rule; note it may depend on the input)

The returned node stores the forward value and a backward closure that checks the upstream
gradient's shape and returns the parent contribution.
-/
def unary {α : Type} [DecidableEq Shape] {σ τ : Shape}
  (t : Tape α) (opName : String) (xId : Nat)
  (forward : Tensor α σ → Tensor α τ)
  (backward : Tensor α σ → Tensor α τ → Tensor α σ) :
  Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=σ) xId
  let y := forward x
  let node : Node α :=
    { name := some opName
      value := AnyTensor.mk y
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := τ) dLdyAny
        let dLdx : Tensor α σ := backward x dLdy
        pure [(xId, AnyTensor.mk dLdx)]
    }
  pure (t.addNode node)

/--
Flatten a tensor `s` into a 1D vector of length `Shape.size s`.

PyTorch comparison: `torch.flatten(x)` with `start_dim=0`.
-/
def flatten {α : Type} [Inhabited α] [DecidableEq Shape] {s : Shape}
  (t : Tape α) (xId : Nat) : Result (Tape α × Nat) :=
  unary (α := α) (t := t) (σ := s) (τ := .dim (Shape.size s) .scalar)
    "flatten" xId
    (forward := fun x => flattenSpec (α := α) x)
    (backward := fun _x dLdz => unflattenSpec (α := α) s dLdz)

/--
Reshape a tensor while preserving number of elements.

The proof argument `h` enforces `Shape.size s₁ = Shape.size s₂`.
PyTorch comparison: `x.reshape(new_shape)` / `x.view(new_shape)` (when valid).
-/
def reshape {α : Type} [Inhabited α] [DecidableEq Shape] {s₁ s₂ : Shape}
  (t : Tape α) (xId : Nat) (h : Shape.size s₁ = Shape.size s₂) : Result (Tape α × Nat) :=
  unary (α := α) (t := t) (σ := s₁) (τ := s₂)
    "reshape" xId
    (forward := fun x => reshapeSpec (α := α) (s₁ := s₁) (s₂ := s₂) x h)
    (backward := fun _x dLdz => reshapeSpec (α := α) (s₁ := s₂) (s₂ := s₁) dLdz h.symm)

/-- Transpose a 2D matrix. PyTorch: `x.t()` / `x.transpose(0,1)`. -/
def transpose2d {α : Type} [DecidableEq Shape] {m n : Nat}
  (t : Tape α) (xId : Nat) : Result (Tape α × Nat) :=
  unary (α := α) (t := t) (σ := .dim m (.dim n .scalar)) (τ := .dim n (.dim m .scalar))
    "transpose2d" xId
    (forward := fun x => matrixTransposeSpec (α := α) x)
    (backward := fun _x dLdz => matrixTransposeSpec (α := α) dLdz)

/-- Permute a 3D tensor `(a,b,c) → (b,c,a)`. PyTorch: `x.permute(1,2,0)`. -/
def transpose3dFirstToLast {α : Type} [DecidableEq Shape] {a b c : Nat}
  (t : Tape α) (xId : Nat) : Result (Tape α × Nat) :=
  unary (α := α) (t := t)
    (σ := .dim a (.dim b (.dim c .scalar)))
    (τ := .dim b (.dim c (.dim a .scalar)))
    "transpose3d_first_to_last" xId
    (forward := fun x => Spec.Tensor.transpose3DFirstToLastSpec (α := α) (a := a) (b := b) (c
      := c) x)
    (backward := fun _x dLdz =>
      Spec.Tensor.transpose3DLastToFirstSpec (α := α) (a := b) (b := c) (c := a) dLdz)

/-- Permute a 3D tensor `(a,b,c) → (c,a,b)`. PyTorch: `x.permute(2,0,1)`. -/
def transpose3dLastToFirst {α : Type} [DecidableEq Shape] {a b c : Nat}
  (t : Tape α) (xId : Nat) : Result (Tape α × Nat) :=
  unary (α := α) (t := t)
    (σ := .dim a (.dim b (.dim c .scalar)))
    (τ := .dim c (.dim a (.dim b .scalar)))
    "transpose3d_last_to_first" xId
    (forward := fun x => Spec.Tensor.transpose3DLastToFirstSpec (α := α) (a := a) (b := b) (c
      := c) x)
    (backward := fun _x dLdz =>
      Spec.Tensor.transpose3DFirstToLastSpec (α := α) (a := c) (b := a) (c := b) dLdz)

/-- Swap the last two axes of a 3D tensor `(a,b,c) → (a,c,b)`. PyTorch: `x.transpose(1,2)`. -/
def transpose3dLastTwo {α : Type} [DecidableEq Shape] {a b c : Nat}
  (t : Tape α) (xId : Nat) : Result (Tape α × Nat) :=
  unary (α := α) (t := t)
    (σ := .dim a (.dim b (.dim c .scalar)))
    (τ := .dim a (.dim c (.dim b .scalar)))
    "transpose3d_last_two" xId
    (forward := fun x => Spec.Tensor.transpose3DLastTwoSpec (α := α) (a := a) (b := b) (c := c)
      x)
    (backward := fun _x dLdz =>
      Spec.Tensor.transpose3DLastTwoSpec (α := α) (a := a) (b := c) (c := b) dLdz)

/--
Swap adjacent axes at a given depth inside a general `Shape`.

This is a more general analogue of `transpose` operations.
-/
def swapAdjacentAtDepth {α : Type} [DecidableEq Shape] {s : Shape}
  (t : Tape α) (depth : Nat) (xId : Nat) : Result (Tape α × Nat) :=
  unary (α := α) (t := t) (σ := s) (τ := s.swapAdjacentAtDepth depth)
    "swapAdjacentAtDepth" xId
    (forward := fun x => Spec.Tensor.swapAtDepthHelper (tensor := x) depth)
    (backward := fun _x dLdz =>
      let dx' := Spec.Tensor.swapAtDepthHelper (tensor := dLdz) depth
      Tensor.castShape dx' (by simpa using (Spec.Shape.swapAdjacentAtDepth_involutive s depth)))

/--
Broadcast `x : s₁` to `s₂` using a proof `Shape.CanBroadcastTo s₁ s₂`.

PyTorch comparison: implicit broadcasting / `x.expand(...)`.
-/
def broadcastTo {α : Type} [Inhabited α] [Add α] [Zero α] [DecidableEq Shape]
  {s₁ s₂ : Shape} (cb : Shape.CanBroadcastTo s₁ s₂) (t : Tape α) (xId : Nat) :
  Result (Tape α × Nat) :=
  unary (α := α) (t := t) (σ := s₁) (τ := s₂)
    "broadcastTo" xId
    (forward := fun x => Spec.Tensor.broadcastTo (α := α) cb x)
    (backward := fun _x dLdz => Spec.Tensor.reduceFromBroadcastTo (α := α) cb dLdz)

/--
Sum-reduce along `axis`.

PyTorch comparison: `torch.sum(x, dim=axis)`.
-/
def reduceSum {α : Type} [Add α] [Zero α] [Inhabited α] [DecidableEq Shape]
  {s : Shape} (axis : Nat) [valid : Shape.valid_axis_inst axis s] [wf : Shape.WellFormed s]
  (t : Tape α) (xId : Nat) : Result (Tape α × Nat) :=
  unary (α := α) (t := t) (σ := s) (τ := shapeAfterSum s axis)
    s!"reduce_sum(axis={axis})" xId
    (forward := fun x => reduceSumAuto (α := α) (s := s) axis x)
    (backward := fun _x dLdz =>
      let cb := shapeAfterSumBroadcastBack (s := s) axis valid wf
      Spec.Tensor.broadcastTo (α := α) cb dLdz)

/--
Mean-reduce along `axis`.

Backward rule: broadcast the upstream cotangent back to `s` and divide by the reduced dimension.
PyTorch comparison: `torch.mean(x, dim=axis)`.
-/
def reduceMean {α : Type} [Context α] [Inhabited α] [DecidableEq Shape]
  {s : Shape} (axis : Nat) [valid : Shape.valid_axis_inst axis s] [wf : Shape.WellFormed s]
  (t : Tape α) (xId : Nat) : Result (Tape α × Nat) :=
  unary (α := α) (t := t) (σ := s) (τ := shapeAfterSum s axis)
    s!"reduce_mean(axis={axis})" xId
    (forward := fun x =>
      let h := Shape.proveReducibleAlong axis s valid.proof
      Spec.Tensor.reduceMean (α := α) (s := s) axis x h)
    (backward := fun _x dLdz =>
      let cb := shapeAfterSumBroadcastBack (s := s) axis valid wf
      let dLdx := Spec.Tensor.broadcastTo (α := α) cb dLdz
      let denomNat :=
        match getDimSize s axis with
        | some n => n
        | none => 1
      Spec.Tensor.scaleSpec (α := α) (s := s) dLdx (1 / (denomNat : α)))

/--
Gather a scalar from a 1D vector using a compile-time index `Fin n`.

PyTorch comparison: `x[i]` (1D indexing).
-/
def gatherScalar {α : Type} [Zero α] [DecidableEq Shape]
  {n : Nat} (t : Tape α) (xId : Nat) (i : Fin n) : Result (Tape α × Nat) := do
  let x ← requireValue (α := α) (t := t) (s := .dim n .scalar) xId
  let y : Tensor α Shape.scalar := getAtSpec x i
  let node : Node α :=
    { name := some s!"gather_scalar[{i.val}]"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := Shape.scalar) dLdyAny
        let g : α := Tensor.toScalar dLdy
        let dx : Tensor α (.dim n .scalar) :=
          Tensor.dim (fun j => Tensor.scalar (if decide (j = i) then g else 0))
        pure [(xId, AnyTensor.mk dx)]
    }
  pure (t.addNode node)

/--
Gather a row from a 2D matrix using a compile-time index `Fin rows`.

PyTorch comparison: `x[i]` for 2D tensors (row indexing).
-/
def gatherRow {α : Type} [Zero α] [DecidableEq Shape]
  {rows cols : Nat} (t : Tape α) (xId : Nat) (i : Fin rows) : Result (Tape α × Nat) := do
  let x ← requireValue (α := α) (t := t) (s := .dim rows (.dim cols .scalar)) xId
  let y : Tensor α (.dim cols .scalar) := getAtSpec x i
  let node : Node α :=
    { name := some s!"gather_row[{i.val}]"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := .dim cols .scalar) dLdyAny
        let dx : Tensor α (.dim rows (.dim cols .scalar)) :=
          Tensor.dim (fun r =>
            if decide (r = i) then
              dLdy
            else
              fill (0 : α) (.dim cols .scalar))
        pure [(xId, AnyTensor.mk dx)]
    }
  pure (t.addNode node)

/--
Gather a scalar from a 1D vector using a runtime `Nat` index.

Out-of-bounds indices are totalized to return `0`.
PyTorch comparison: `x[i]` would raise on out-of-range; here we return `0` to keep the op total.
-/
def gatherScalarNat {α : Type} [Zero α] [DecidableEq Shape]
  {n : Nat} (t : Tape α) (xId : Nat) (i : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α := α) (t := t) (s := .dim n .scalar) xId
  let y : Tensor α Shape.scalar :=
    if h : i < n then
      getAtSpec x ⟨i, h⟩
    else
      Tensor.scalar 0
  let node : Node α :=
    { name := some s!"gather_scalar_nat[{i}]"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := Shape.scalar) dLdyAny
        let g : α := Tensor.toScalar dLdy
        let dx : Tensor α (.dim n .scalar) :=
          Tensor.dim (fun j =>
            if i < n then
              if decide (j.val = i) then Tensor.scalar g else Tensor.scalar 0
            else
              Tensor.scalar 0)
        pure [(xId, AnyTensor.mk dx)]
    }
  pure (t.addNode node)

/--
Gather `k` scalars from a 1D vector using an explicit index tensor.

Out-of-bounds indices are totalized to `0`. In the backward pass, gradients are accumulated for
repeated indices (scatter-add semantics).
PyTorch comparison: related to `torch.gather` / advanced indexing.
-/
def gatherVecNat {α : Type} [Add α] [Zero α] [DecidableEq Shape]
  {n k : Nat} (t : Tape α) (xId : Nat) (idx : Tensor Nat (.dim k .scalar)) :
  Result (Tape α × Nat) := do
  let x ← requireValue (α := α) (t := t) (s := .dim n .scalar) xId
  let y : Tensor α (.dim k .scalar) :=
    match idx with
    | Tensor.dim f =>
        Tensor.dim (fun j =>
          match f j with
          | Tensor.scalar ij =>
              if h : ij < n then
                getAtSpec x ⟨ij, h⟩
              else
                Tensor.scalar 0)
  let node : Node α :=
    { name := some "gather_vec_nat"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := .dim k .scalar) dLdyAny
        let dx : Tensor α (.dim n .scalar) :=
          Tensor.dim (fun iFin =>
            let sum : α :=
              (List.finRange k).foldl (fun acc j =>
                let ij :=
                  match idx with
                  | Tensor.dim f =>
                      match f j with
                      | Tensor.scalar v => v
                if ij < n then
                  if decide (ij = iFin.val) then
                    let gj : α :=
                      match getAtSpec dLdy j with
                      | Tensor.scalar v => v
                    acc + gj
                  else acc
                else acc
              ) 0
            Tensor.scalar sum)
        pure [(xId, AnyTensor.mk dx)]
    }
  pure (t.addNode node)

/--
Gather `k` rows from a 2D matrix using an explicit index tensor.

Out-of-bounds indices are totalized to zero rows; backward accumulates gradients into selected
rows (scatter-add), including repeated indices.
-/
def gatherRowsNat {α : Type} [Add α] [Zero α] [DecidableEq Shape]
  {rows cols k : Nat} (t : Tape α) (xId : Nat) (idx : Tensor Nat (.dim k .scalar)) :
  Result (Tape α × Nat) := do
  let x ← requireValue (α := α) (t := t) (s := .dim rows (.dim cols .scalar)) xId
  let y : Tensor α (.dim k (.dim cols .scalar)) :=
    match idx with
    | Tensor.dim f =>
        Tensor.dim (fun j =>
          match f j with
          | Tensor.scalar ij =>
              if h : ij < rows then
                getAtSpec x ⟨ij, h⟩
              else
                fill (0 : α) (.dim cols .scalar))
  let node : Node α :=
    { name := some "gather_rows_nat"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := .dim k (.dim cols .scalar)) dLdyAny
        let dx : Tensor α (.dim rows (.dim cols .scalar)) :=
          Tensor.dim (fun rFin =>
            let rowGrad : Tensor α (.dim cols .scalar) :=
              (List.finRange k).foldl (fun acc j =>
                let ij :=
                  match idx with
                  | Tensor.dim f =>
                      match f j with
                      | Tensor.scalar v => v
                if ij < rows then
                  if decide (ij = rFin.val) then
                    addSpec acc (getAtSpec dLdy j)
                  else acc
                else acc
              ) (fill (0 : α) (.dim cols .scalar))
            rowGrad)
        pure [(xId, AnyTensor.mk dx)]
    }
  pure (t.addNode node)

/--
Scatter-add into a vector: return a copy of `x` with `x[i] += v`.

Backward: gradient w.r.t. `x` is the upstream `dL/dy`, and gradient w.r.t. `v` is the gathered
scalar `dL/dy[i]`.
-/
def scatterAddVec {α : Type} [Add α] [Zero α] [DecidableEq Shape]
  {n : Nat} (t : Tape α) (xId vId : Nat) (i : Fin n) : Result (Tape α × Nat) := do
  let x ← requireValue (α := α) (t := t) (s := .dim n .scalar) xId
  let vT ← requireValue (α := α) (t := t) (s := Shape.scalar) vId
  let v : α := Tensor.toScalar vT
  let xiT : Tensor α Shape.scalar := getAtSpec x i
  let xi : α := Tensor.toScalar xiT
  let y : Tensor α (.dim n .scalar) := updateSpec x [i.val] (xi + v)
  let node : Node α :=
    { name := some s!"scatter_add_vec[{i.val}]"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [xId, vId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := .dim n .scalar) dLdyAny
        let dv : Tensor α Shape.scalar := getAtSpec dLdy i
        pure [(xId, AnyTensor.mk dLdy), (vId, AnyTensor.mk dv)]
    }
  pure (t.addNode node)

/--
Scatter-add into a matrix row: return a copy of `x` with `x[i,:] += v`.

Backward: gradient w.r.t. `v` is the gathered row `dL/dy[i,:]`.
-/
def scatterAddRow {α : Type} [Add α] [Zero α] [DecidableEq Shape]
  {rows cols : Nat} (t : Tape α) (xId vId : Nat) (i : Fin rows) : Result (Tape α × Nat) := do
  let x ← requireValue (α := α) (t := t) (s := .dim rows (.dim cols .scalar)) xId
  let v ← requireValue (α := α) (t := t) (s := .dim cols .scalar) vId
  let y : Tensor α (.dim rows (.dim cols .scalar)) :=
    Tensor.dim (fun r =>
      if decide (r = i) then
        addSpec (getAtSpec x r) v
      else
        getAtSpec x r)
  let node : Node α :=
    { name := some s!"scatter_add_row[{i.val}]"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [xId, vId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := .dim rows (.dim cols .scalar)) dLdyAny
        let dv : Tensor α (.dim cols .scalar) := getAtSpec dLdy i
        pure [(xId, AnyTensor.mk dLdy), (vId, AnyTensor.mk dv)]
    }
  pure (t.addNode node)

/-- Elementwise addition. PyTorch: `torch.add` / `+`. -/
def add {α : Type} [Add α] [DecidableEq Shape] {s : Shape}
  (t : Tape α) (aId bId : Nat) : Result (Tape α × Nat) := do
  let a ← requireValue (α:=α) (t:=t) (s:=s) aId
  let b ← requireValue (α:=α) (t:=t) (s:=s) bId
  let y := addSpec a b
  let node : Node α :=
    { name := some "add"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [aId, bId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        pure [(aId, AnyTensor.mk dLdy), (bId, AnyTensor.mk dLdy)]
    }
  pure (t.addNode node)

/-- Elementwise subtraction. PyTorch: `torch.sub` / `-`. -/
def sub {α : Type} [Sub α] [Zero α] [DecidableEq Shape] {s : Shape}
  (t : Tape α) (aId bId : Nat) : Result (Tape α × Nat) := do
  let a ← requireValue (α:=α) (t:=t) (s:=s) aId
  let b ← requireValue (α:=α) (t:=t) (s:=s) bId
  let y := subSpec a b
  let node : Node α :=
    { name := some "sub"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [aId, bId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        let neg_dLdy : Tensor α s := subSpec (fill (0 : α) s) dLdy
        pure [(aId, AnyTensor.mk dLdy), (bId, AnyTensor.mk neg_dLdy)]
    }
  pure (t.addNode node)

/-- Elementwise multiplication. PyTorch: `torch.mul` / `*`. -/
def mul {α : Type} [Mul α] [DecidableEq Shape] {s : Shape}
  (t : Tape α) (aId bId : Nat) : Result (Tape α × Nat) := do
  let a ← requireValue (α:=α) (t:=t) (s:=s) aId
  let b ← requireValue (α:=α) (t:=t) (s:=s) bId
  let y := mulSpec a b
  let node : Node α :=
    { name := some "mul"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [aId, bId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        let da : Tensor α s := mulSpec dLdy b
        let db : Tensor α s := mulSpec dLdy a
        pure [(aId, AnyTensor.mk da), (bId, AnyTensor.mk db)]
    }
  pure (t.addNode node)

/-- Multiply a tensor by a scalar constant. PyTorch: `x * c` for Python scalar `c`. -/
def scale {α : Type} [Mul α] [DecidableEq Shape] {s : Shape}
  (t : Tape α) (xId : Nat) (c : α) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  let y := scaleSpec x c
  let node : Node α :=
    { name := some "scale"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        pure [(xId, AnyTensor.mk (scaleSpec dLdy c))]
    }
  pure (t.addNode node)

/--
Elementwise absolute value.

Backward uses the sign function (`sign_spec`) as a subgradient at `0`.
PyTorch comparison: `torch.abs`.
-/
def abs {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) :=
  unary (α := α) (t := t) (σ := s) (τ := s)
    "abs" xId
    (forward := fun x => absSpec (α := α) (s := s) x)
    (backward := fun x dLdy =>
      let dabs : Tensor α s := signSpec (α := α) (s := s) x
      mulSpec dabs dLdy)

/--
Elementwise square root.

Backward uses `1 / (2 * sqrt(x))` for `x > 0` and `0` otherwise (totalized).
PyTorch comparison: `torch.sqrt`.
-/
def sqrt {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) :=
  unary (α := α) (t := t) (σ := s) (τ := s)
    "sqrt" xId
    (forward := fun x => sqrtSpec (α := α) (s := s) x)
    (backward := fun x dLdy =>
      let dsqrt : Tensor α s :=
        mapSpec (α := α) (s := s) (fun v =>
          if v > 0 then
            (1 : α) / (((2 : Nat) : α) * MathFunctions.sqrt v)
          else
            (0 : α)) x
      mulSpec dsqrt dLdy)

/--
Elementwise clamp to `[minVal, maxVal]`.

Backward multiplies by an indicator of the open interval `(minVal, maxVal)` (zero at boundaries).
PyTorch comparison: `torch.clamp`.
-/
def clamp {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {s : Shape} (t : Tape α) (xId : Nat) (minVal maxVal : α) : Result (Tape α × Nat) :=
  unary (α := α) (t := t) (σ := s) (τ := s)
    "clamp" xId
    (forward := fun x => clampSpec (α := α) (s := s) x minVal maxVal)
    (backward := fun x dLdy =>
      let dclamp : Tensor α s :=
        mapSpec (α := α) (s := s) (fun v =>
          if v > minVal ∧ maxVal > v then (1 : α) else (0 : α)) x
      mulSpec dclamp dLdy)

/--
Elementwise maximum.

Tie-breaking: when `a = b`, the upstream gradient is split evenly (`0.5`) between both inputs.
PyTorch comparison: `torch.maximum`.
-/
def max {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {s : Shape} (t : Tape α) (aId bId : Nat) : Result (Tape α × Nat) := do
  let a ← requireValue (α:=α) (t:=t) (s:=s) aId
  let b ← requireValue (α:=α) (t:=t) (s:=s) bId
  let y := maxSpec (α := α) (s := s) a b
  let node : Node α :=
    { name := some "max"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [aId, bId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        let half : α := (1 : α) / ((2 : Nat) : α)
        let maskA : Tensor α s :=
          map2Spec (α := α) (β := α) (γ := α) (s := s) (fun x y =>
            if x > y then (1 : α) else if y > x then (0 : α) else half) a b
        let maskB : Tensor α s :=
          map2Spec (α := α) (β := α) (γ := α) (s := s) (fun x y =>
            if y > x then (1 : α) else if x > y then (0 : α) else half) a b
        pure [
          (aId, AnyTensor.mk (mulSpec maskA dLdy)),
          (bId, AnyTensor.mk (mulSpec maskB dLdy))
        ]
    }
  pure (t.addNode node)

/--
Elementwise minimum.

Tie-breaking: when `a = b`, the upstream gradient is split evenly (`0.5`) between both inputs.
PyTorch comparison: `torch.minimum`.
-/
def min {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {s : Shape} (t : Tape α) (aId bId : Nat) : Result (Tape α × Nat) := do
  let a ← requireValue (α:=α) (t:=t) (s:=s) aId
  let b ← requireValue (α:=α) (t:=t) (s:=s) bId
  let y := minSpec (α := α) (s := s) a b
  let node : Node α :=
    { name := some "min"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [aId, bId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        let half : α := (1 : α) / ((2 : Nat) : α)
        let maskA : Tensor α s :=
          map2Spec (α := α) (β := α) (γ := α) (s := s) (fun x y =>
            if y > x then (1 : α) else if x > y then (0 : α) else half) a b
        let maskB : Tensor α s :=
          map2Spec (α := α) (β := α) (γ := α) (s := s) (fun x y =>
            if x > y then (1 : α) else if y > x then (0 : α) else half) a b
        pure [
          (aId, AnyTensor.mk (mulSpec maskA dLdy)),
          (bId, AnyTensor.mk (mulSpec maskB dLdy))
        ]
    }
  pure (t.addNode node)

/--
Elementwise ReLU.

PyTorch comparison: `torch.relu(x)` / `torch.nn.functional.relu(x)`.
-/
def relu {α : Type}
  [Mul α] [Zero α] [Max α] [One α] [LT α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  let y := Activation.reluSpec (α:=α) x
  let node : Node α :=
    { name := some "relu"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        let drelu := Activation.reluDerivSpec (α:=α) x
        pure [(xId, AnyTensor.mk (mulSpec drelu dLdy))]
    }
  pure (t.addNode node)

/--
Fully-connected linear layer `y = W x + b` (matvec).

Type-level shapes enforce `W : (outDim, inDim)`, `x : (inDim,)`, `b : (outDim,)`.
PyTorch comparison: `torch.nn.functional.linear`.
-/
def linear {α : Type} [Add α] [Mul α] [Zero α] [DecidableEq Shape]
  {inDim outDim : Nat}
  (t : Tape α) (wId bId xId : Nat) : Result (Tape α × Nat) := do
  let W ← requireValue (α:=α) (t:=t) (s:=.dim outDim (.dim inDim .scalar)) wId
  let b ← requireValue (α:=α) (t:=t) (s:=.dim outDim .scalar) bId
  let x ← requireValue (α:=α) (t:=t) (s:=.dim inDim .scalar) xId
  let layer : Spec.LinearSpec α inDim outDim := { weights := W, bias := b }
  let y := Spec.linearSpec (α:=α) layer x
  let node : Node α :=
    { name := some "linear"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [wId, bId, xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := .dim outDim .scalar) dLdyAny
        let dW := Spec.linearWeightsDerivSpec (α:=α) x dLdy
        let db := Spec.linearBiasDerivSpec (α:=α) (dW) dLdy x
        let dx := Spec.linearInputDerivSpec (α:=α) W dLdy
        pure [
          (wId, AnyTensor.mk dW),
          (bId, AnyTensor.mk db),
          (xId, AnyTensor.mk dx)
        ]
    }
  pure (t.addNode node)

/--
2D matrix multiplication.

PyTorch comparison: `torch.matmul(a, b)` for 2D tensors.
-/
def matmul {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {m n p : Nat} (t : Tape α) (aId bId : Nat) : Result (Tape α × Nat) := do
  let a ← requireValue (α:=α) (t:=t) (s:=.dim m (.dim n .scalar)) aId
  let b ← requireValue (α:=α) (t:=t) (s:=.dim n (.dim p .scalar)) bId
  let y := Spec.matMulSpec a b
  let node : Node α :=
    { name := some "matmul"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [aId, bId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := .dim m (.dim p .scalar)) dLdyAny
        let (dA, dB) := Spec.Tensor.matMulBackwardSpec a b dLdy
        pure [(aId, AnyTensor.mk dA), (bId, AnyTensor.mk dB)]
    }
  pure (t.addNode node)

/--
Batched matrix multiplication.

PyTorch comparison: `torch.bmm(a, b)`.
-/
def bmm {α : Type} [Add α] [Mul α] [Zero α] [DecidableEq Shape]
  {batch m n p : Nat} (t : Tape α) (aId bId : Nat) : Result (Tape α × Nat) := do
  let a ← requireValue (α:=α) (t:=t) (s:=.dim batch (.dim m (.dim n .scalar))) aId
  let b ← requireValue (α:=α) (t:=t) (s:=.dim batch (.dim n (.dim p .scalar))) bId
  let y := Spec.Tensor.bmmSpec (α := α) (batch := batch) (m := m) (n := n) (p := p) a b
  let node : Node α :=
    { name := some "bmm"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [aId, bId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := .dim batch (.dim m (.dim p .scalar))) dLdyAny
        let (dA, dB) := Spec.Tensor.bmmBackwardSpec (α := α) (batch := batch) (m := m) (n := n)
          (p := p) a b dLdy
        pure [(aId, AnyTensor.mk dA), (bId, AnyTensor.mk dB)]
    }
  pure (t.addNode node)

/--
Concatenate two 1D vectors along dimension 0.

PyTorch comparison: `torch.cat([a, b], dim=0)` for vectors.
-/
def concatVectors {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq
  Shape]
  {n m : Nat} (t : Tape α) (aId bId : Nat) : Result (Tape α × Nat) := do
  let a ← requireValue (α:=α) (t:=t) (s:=.dim n .scalar) aId
  let b ← requireValue (α:=α) (t:=t) (s:=.dim m .scalar) bId
  let y := Spec.Tensor.concatVectorsSpec a b
  let node : Node α :=
    { name := some "concat_vectors"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [aId, bId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := .dim (n + m) .scalar) dLdyAny
        let dA := Spec.Tensor.sliceVectorSpec dLdy 0 n (by simp)
        let dB := Spec.Tensor.sliceVectorSpec dLdy n m (by exact Nat.le_refl _)
        pure [(aId, AnyTensor.mk dA), (bId, AnyTensor.mk dB)]
    }
  pure (t.addNode node)

/--
Concatenate two tensors along dimension 0.

PyTorch comparison: `torch.cat([a, b], dim=0)`.
-/
def concatDim0 {α : Type} [DecidableEq Shape]
  {n m : Nat} {s : Shape} (t : Tape α) (aId bId : Nat) : Result (Tape α × Nat) := do
  let a ← requireValue (α := α) (t := t) (s := .dim n s) aId
  let b ← requireValue (α := α) (t := t) (s := .dim m s) bId
  let y := Spec.Tensor.concatDim0Spec (α := α) (n := n) (m := m) (s := s) a b
  let node : Node α :=
    { name := some "concat_dim0"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [aId, bId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := .dim (n + m) s) dLdyAny
        let dA := Spec.sliceRangeSpec (α := α) (n := n + m) (s := s) dLdy 0 n
          (by simp)
        let dB := Spec.sliceRangeSpec (α := α) (n := n + m) (s := s) dLdy n m
          (by simp [Nat.add_comm])
        pure [(aId, AnyTensor.mk dA), (bId, AnyTensor.mk dB)]
    }
  pure (t.addNode node)

/--
Slice along dimension 0: `x[start : start+len]`.

The proof argument `h` enforces bounds.
PyTorch comparison: `x[start:start+len]` on tensors with a leading dimension.
-/
def sliceRange0 {α : Type} [Zero α] [DecidableEq Shape]
  {n : Nat} {s : Shape} (t : Tape α) (xId : Nat) (start len : Nat) (h : len + start ≤ n) :
  Result (Tape α × Nat) :=
  unary (α := α) (t := t) (σ := .dim n s) (τ := .dim len s)
    "slice_range0" xId
    (forward := fun x => Spec.sliceRangeSpec (α := α) (n := n) (s := s) x start len h)
    (backward := fun _x dLdz =>
      Spec.Tensor.sliceRange0BackwardSpec (α := α) (n := n) (s := s) start len h dLdz)

/--
N-D convolution for channels-first tensors `(inC, spatial...)` (no batch axis).

This is the generic counterpart to `conv2d`; `conv2d` is implemented as a specialization with
`d = 2`, scalar stride, and scalar padding.
-/
def conv {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {d inC outC : Nat}
  {kernel stride padding : Vector Nat d}
  {inSpatial : Vector Nat d}
  (t : Tape α) (kernelId biasId inputId : Nat) (name : String := "conv") :
  Result (Tape α × Nat) := do
  let k ← requireValue (α:=α) (t:=t)
    (s:=Shape.ofList (outC :: inC :: kernel.toList)) kernelId
  let b ← requireValue (α:=α) (t:=t) (s:=.dim outC .scalar) biasId
  let x ← requireValue (α:=α) (t:=t)
    (s:=Shape.ofList (inC :: inSpatial.toList)) inputId
  let layer : Spec.ConvSpec d inC outC kernel stride padding α :=
    { kernel := k, bias := b }
  let y := Spec.convSpec (layer := layer) x
  let outSpatial := Spec.convOutSpatial inSpatial kernel stride padding
  let outSh : Shape := Shape.ofList (outC :: outSpatial.toList)
  let node : Node α :=
    { name := some name
      value := AnyTensor.mk y
      requires_grad := true
      parents := [kernelId, biasId, inputId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := outSh) dLdyAny
        let (dK, dB, dX) := Spec.convBackwardSpec (layer := layer) x dLdy
        pure [
          (kernelId, AnyTensor.mk dK),
          (biasId, AnyTensor.mk dB),
          (inputId, AnyTensor.mk dX)
        ]
    }
  pure (t.addNode node)

/--
2D convolution for channel-first images `(inC,inH,inW)` (no batch axis).

PyTorch comparison: `torch.nn.functional.conv2d` specialized to a single image.
-/
def conv2d {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (t : Tape α) (kernelId biasId inputId : Nat) : Result (Tape α × Nat) := by
  let _ := h1
  let _ := h2
  let _ := h3
  exact
    conv (α := α)
      (d := 2)
      (inC := inC)
      (outC := outC)
      (kernel := ⟨#[kH, kW], by simp⟩)
      (stride := ⟨#[stride, stride], by simp⟩)
      (padding := ⟨#[padding, padding], by simp⟩)
      (inSpatial := ⟨#[inH, inW], by simp⟩)
      t kernelId biasId inputId
      (name := "conv2d")

/--
N-D transpose convolution for channels-first tensors `(inC, spatial...)` (no batch axis).

This is the generic counterpart to `conv_transpose2d`.

Kernel layout matches the spec/PyTorch convention `(inC, outC, kernel[0], ..., kernel[d-1])`.

PyTorch comparison: `torch.nn.functional.conv_transpose{d}d` specialized to a single sample
(no batch axis).
-/
def convTranspose {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  [DecidableEq Shape]
  {d inC outC : Nat}
  {kernel stride padding : Vector Nat d}
  {inSpatial : Vector Nat d}
  (t : Tape α) (kernelId biasId inputId : Nat) (name : String := "conv_transpose") :
  Result (Tape α × Nat) := do
  let w ← requireValue (α := α) (t := t)
    (s := Shape.ofList (inC :: outC :: kernel.toList)) kernelId
  let b ← requireValue (α := α) (t := t) (s := .dim outC .scalar) biasId
  let x ← requireValue (α := α) (t := t)
    (s := Shape.ofList (inC :: inSpatial.toList)) inputId

  let layer : Spec.ConvTransposeSpec d inC outC kernel stride padding α :=
    { kernel := w, bias := b }
  let y := Spec.convTransposeSpec (layer := layer) x
  let outSpatial := Spec.convTransposeOutSpatial inSpatial kernel stride padding
  let outSh : Shape := Shape.ofList (outC :: outSpatial.toList)

  let node : Node α :=
    { name := some name
      value := AnyTensor.mk y
      requires_grad := true
      parents := [kernelId, biasId, inputId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := outSh) dLdyAny
        let (dW, dB, dX) := Spec.convTransposeBackwardSpec (layer := layer) x dLdy
        pure [
          (kernelId, AnyTensor.mk dW),
          (biasId, AnyTensor.mk dB),
          (inputId, AnyTensor.mk dX)
        ]
    }
  pure (t.addNode node)

/--
2D transpose convolution for channel-first images `(inC,inH,inW)` (no batch axis).

This is implemented as a specialization of `conv_transpose` with `d = 2`, scalar stride, and
scalar padding.
Kernel layout matches the spec/PyTorch convention `(inC,outC,kH,kW)`.

PyTorch comparison: `torch.nn.functional.conv_transpose2d` specialized to a single image.
-/
def convTranspose2d {α : Type} [Context α] [DecidableEq Shape]
  {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (t : Tape α) (kernelId biasId inputId : Nat) : Result (Tape α × Nat) := by
  let _ := h1
  let _ := h2
  let _ := h3
  exact
    convTranspose (α := α)
      (d := 2)
      (inC := inC)
      (outC := outC)
      (kernel := ⟨#[kH, kW], by simp⟩)
      (stride := ⟨#[stride, stride], by simp⟩)
      (padding := ⟨#[padding, padding], by simp⟩)
      (inSpatial := ⟨#[inH, inW], by simp⟩)
      t kernelId biasId inputId
      (name := "conv_transpose2d")

/--
N-D max pooling for channels-first tensors `(C, spatial...)` (no batch axis).

Padding is symmetric per-axis and uses zeros. To model unpadded pooling, pass `padding := 0` on
every axis.
-/
def maxPool {α : Type} [Context α] [DecidableEq Shape]
  {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
  {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t)
    (s:=Shape.ofList (C :: inSpatial.toList)) xId
  if hStride : (∀ i : Fin d, stride.get i ≠ 0) then
    let layer : Spec.MaxPoolSpec d kernel stride padding hKernel hStride := {}
    let y := Spec.maxPoolSpec (layer := layer) x
    let outSpatial := Spec.poolOutSpatialPad inSpatial kernel stride padding
    let outSh : Shape := Shape.ofList (C :: outSpatial.toList)
    let node : Node α :=
      { name := some "max_pool"
        value := AnyTensor.mk y
        requires_grad := true
        parents := [xId]
        backward := fun dLdyAny => do
          let dLdy ← requireGrad (α := α) (τ := outSh) dLdyAny
          let dx := Spec.maxPoolBackwardSpec (layer := layer) (input := x) (grad_output := dLdy)
          pure [(xId, AnyTensor.mk dx)]
      }
    pure (t.addNode node)
  else
    throw "autograd: max_pool requires stride > 0 on every spatial axis"

/--
N-D average pooling for channels-first tensors `(C, spatial...)` (no batch axis).

Padding is symmetric per-axis and uses zeros; pooling uses `count_include_pad=true` semantics.
-/
def avgPool {α : Type} [Context α] [DecidableEq Shape]
  {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
  (hKernel : ∀ i : Fin d, kernel.get i ≠ 0)
  (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t)
    (s:=Shape.ofList (C :: inSpatial.toList)) xId
  if hStride : (∀ i : Fin d, stride.get i ≠ 0) then
    let layer : Spec.AvgPoolSpec d kernel stride padding hKernel hStride := {}
    let y := Spec.avgPoolSpec (layer := layer) x
    let outSpatial := Spec.poolOutSpatialPad inSpatial kernel stride padding
    let outSh : Shape := Shape.ofList (C :: outSpatial.toList)
    let node : Node α :=
      { name := some "avg_pool"
        value := AnyTensor.mk y
        requires_grad := true
        parents := [xId]
        backward := fun dLdyAny => do
          let dLdy ← requireGrad (α := α) (τ := outSh) dLdyAny
          let dx := Spec.avgPoolBackwardSpec (layer := layer) (grad_output := dLdy)
          pure [(xId, AnyTensor.mk dx)]
      }
    pure (t.addNode node)
  else
    throw "autograd: avg_pool requires stride > 0 on every spatial axis"

/--
N-D smooth max pooling (log-sum-exp surrogate) for channels-first tensors `(C, spatial...)`.
-/
def smoothMaxPool {α : Type} [Context α] [DecidableEq Shape]
  {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
  {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (t : Tape α) (xId : Nat) (beta : α) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t)
    (s:=Shape.ofList (C :: inSpatial.toList)) xId
  if hStride : (∀ i : Fin d, stride.get i ≠ 0) then
    let layer : Spec.MaxPoolSpec d kernel stride padding hKernel hStride := {}
    let y := Spec.smoothMaxPoolSpec (layer := layer) (beta := beta) x
    let outSpatial := Spec.poolOutSpatialPad inSpatial kernel stride padding
    let outSh : Shape := Shape.ofList (C :: outSpatial.toList)
    let node : Node α :=
      { name := some "smooth_max_pool"
        value := AnyTensor.mk y
        requires_grad := true
        parents := [xId]
        backward := fun dLdyAny => do
          let dLdy ← requireGrad (α := α) (τ := outSh) dLdyAny
          let dx :=
            Spec.smoothMaxPoolBackwardSpec (layer := layer) (beta := beta)
              (input := x) (grad_output := dLdy)
          pure [(xId, AnyTensor.mk dx)]
      }
    pure (t.addNode node)
  else
    throw "autograd: smooth_max_pool requires stride > 0 on every spatial axis"

/--
2D max-pooling for channel-first images (no batch axis).

PyTorch comparison: `torch.nn.functional.max_pool2d`.
-/
def maxPool2d {α : Type} [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t)
    (s:=.dim inC (.dim inH (.dim inW .scalar))) xId
  if hStride : stride ≠ 0 then
    let layer : Spec.MaxPool2DSpec kH kW stride h1 h2 hStride := {}
    let y := Spec.maxPool2dMultiSpec (layer := layer) x
    let node : Node α :=
      { name := some "max_pool2d"
        value := AnyTensor.mk y
        requires_grad := true
        parents := [xId]
        backward := fun dLdyAny => do
          let outH := (inH - kH) / stride + 1
          let outW := (inW - kW) / stride + 1
          let dLdy ←
            requireGrad (α := α) (τ := .dim inC (.dim outH (.dim outW .scalar))) dLdyAny
          let dx :=
            Tensor.dim (fun c =>
              Spec.maxPool2dBackwardSpec (_layer := layer)
                (input := getAtSpec x c) (grad_output := getAtSpec dLdy c))
          pure [(xId, AnyTensor.mk dx)]
      }
    pure (t.addNode node)
  else
    throw "autograd: max_pool2d requires stride > 0"

/--
2D max-pooling with padding for channel-first images (no batch axis).

PyTorch comparison: `max_pool2d(..., padding=...)`.
-/
def maxPool2dPad {α : Type} [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride padding : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t)
    (s:=.dim inC (.dim inH (.dim inW .scalar))) xId
  if hStride : stride ≠ 0 then
    let layer : Spec.MaxPool2DSpec kH kW stride h1 h2 hStride := {}
    let y := Spec.maxPool2dMultiSpecPad (layer := layer) (padding := padding) x
    let node : Node α :=
      { name := some "max_pool2d_pad"
        value := AnyTensor.mk y
        requires_grad := true
        parents := [xId]
        backward := fun dLdyAny => do
          let outH := (inH + 2 * padding - kH) / stride + 1
          let outW := (inW + 2 * padding - kW) / stride + 1
          let dLdy ←
            requireGrad (α := α) (τ := .dim inC (.dim outH (.dim outW .scalar))) dLdyAny
          let dx :=
            Spec.maxPool2dMultiBackwardSpecPad (layer := layer) (padding := padding) x dLdy
          pure [(xId, AnyTensor.mk dx)]
      }
    pure (t.addNode node)
  else
    throw "autograd: max_pool2d_pad requires stride > 0"

/--
Smooth approximation of max-pooling (softmax pooling).

This is not a standard PyTorch primitive; it is useful for differentiable relaxations.
-/
def smoothMaxPool2d {α : Type} [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (t : Tape α) (xId : Nat) (beta : α) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t)
    (s:=.dim inC (.dim inH (.dim inW .scalar))) xId
  if hStride : stride ≠ 0 then
    let layer : Spec.MaxPool2DSpec kH kW stride h1 h2 hStride := {}
    let y := Spec.smoothMaxPool2dMultiSpec (layer := layer) (beta := beta) x
    let node : Node α :=
      { name := some "smooth_max_pool2d"
        value := AnyTensor.mk y
        requires_grad := true
        parents := [xId]
        backward := fun dLdyAny => do
          let outH := (inH - kH) / stride + 1
          let outW := (inW - kW) / stride + 1
          let dLdy ←
            requireGrad (α := α) (τ := .dim inC (.dim outH (.dim outW .scalar))) dLdyAny
          let dx :=
            Spec.smoothMaxPool2dMultiBackwardSpec (layer := layer) (beta := beta) x dLdy
          pure [(xId, AnyTensor.mk dx)]
      }
    pure (t.addNode node)
  else
    throw "autograd: smooth_max_pool2d requires stride > 0"

/--
2D average-pooling for channel-first images (no batch axis).

PyTorch comparison: `torch.nn.functional.avg_pool2d`.
-/
def avgPool2d {α : Type} [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0)
  (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t)
    (s:=.dim inC (.dim inH (.dim inW .scalar))) xId
  if hStride : stride ≠ 0 then
    let layer : Spec.AvgPool2DSpec kH kW stride h1 h2 hStride := {}
    let y := Spec.avgPool2dMultiSpec (h1 := h1) (h2 := h2) (layer := layer) x
    let node : Node α :=
      { name := some "avg_pool2d"
        value := AnyTensor.mk y
        requires_grad := true
        parents := [xId]
        backward := fun dLdyAny => do
          let outH := (inH - kH) / stride + 1
          let outW := (inW - kW) / stride + 1
          let dLdy ←
            requireGrad (α := α) (τ := .dim inC (.dim outH (.dim outW .scalar))) dLdyAny
          let dx :=
            Tensor.dim (fun c =>
              Spec.avgPool2dBackwardSpec (α := α) h1 h2 layer (getAtSpec dLdy c))
          pure [(xId, AnyTensor.mk dx)]
      }
    pure (t.addNode node)
  else
    throw "autograd: avg_pool2d requires stride > 0"

/--
2D average-pooling with padding for channel-first images (no batch axis).

PyTorch comparison: `avg_pool2d(..., padding=...)`.
-/
def avgPool2dPad {α : Type} [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride padding : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0)
  (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t)
    (s:=.dim inC (.dim inH (.dim inW .scalar))) xId
  if hStride : stride ≠ 0 then
    let layer : Spec.AvgPool2DSpec kH kW stride h1 h2 hStride := {}
    let y := Spec.avgPool2dMultiSpecPad (h1 := h1) (h2 := h2) (layer := layer) (padding :=
      padding) x
    let node : Node α :=
      { name := some "avg_pool2d_pad"
        value := AnyTensor.mk y
        requires_grad := true
        parents := [xId]
        backward := fun dLdyAny => do
          let outH := (inH + 2 * padding - kH) / stride + 1
          let outW := (inW + 2 * padding - kW) / stride + 1
          let dLdy ←
            requireGrad (α := α) (τ := .dim inC (.dim outH (.dim outW .scalar))) dLdyAny
          let dx :=
            Spec.avgPool2dMultiBackwardSpecPad (h1 := h1) (h2 := h2) (layer := layer)
              (padding := padding) dLdy
          pure [(xId, AnyTensor.mk dx)]
      }
    pure (t.addNode node)
  else
    throw "autograd: avg_pool2d_pad requires stride > 0"

/--
Layer normalization for `(seqLen, embedDim)` tensors.

This records a single node whose backward returns gradients for `x`, `gamma`, and `beta`.
PyTorch comparison: `torch.nn.LayerNorm(embedDim)` (applied per token) / `functional.layer_norm`.
-/
def layerNorm {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
  (t : Tape α) (xId gammaId betaId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=.dim seqLen (.dim embedDim .scalar)) xId
  let gamma ← requireValue (α:=α) (t:=t) (s:=.dim embedDim .scalar) gammaId
  let beta ← requireValue (α:=α) (t:=t) (s:=.dim embedDim .scalar) betaId
  let y := Spec.layerNorm (x := x) (gamma := gamma) (beta := beta) h_seq_pos h_embed_pos
  let node : Node α :=
    { name := some "layer_norm"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [xId, gammaId, betaId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := .dim seqLen (.dim embedDim .scalar)) dLdyAny
        let (dx, dgamma, dbeta) :=
          Spec.layerNormBackward (h_seq_pos := h_seq_pos) (h_embed_pos := h_embed_pos)
            (x := x) (gamma := gamma) (_beta := beta) (grad_output := dLdy)
        pure [
          (xId, AnyTensor.mk dx),
          (gammaId, AnyTensor.mk dgamma),
          (betaId, AnyTensor.mk dbeta)
        ]
    }
  pure (t.addNode node)

/--
Batch normalization for channel-first images `(C,H,W)` (no batch axis).

PyTorch comparison: conceptually `torch.nn.BatchNorm2d(C)` / `functional.batch_norm` on NCHW, but
specialized here to a single image.
-/
def batchnormChannelFirst {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  [DecidableEq Shape]
  {channels height width : Nat}
  (h_c : channels > 0) (h_h : height > 0) (h_w : width > 0)
  (t : Tape α) (xId gammaId betaId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t)
    (s:=.dim channels (.dim height (.dim width .scalar))) xId
  let gamma ← requireValue (α:=α) (t:=t) (s:=.dim channels .scalar) gammaId
  let beta ← requireValue (α:=α) (t:=t) (s:=.dim channels .scalar) betaId
  let y := Spec.batchNorm2d (x := x) (gamma := gamma) (beta := beta) h_c h_h h_w
  let node : Node α :=
    { name := some "batchnorm_channel_first"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [xId, gammaId, betaId]
      backward := fun dLdyAny => do
        let dLdy ←
          requireGrad (α := α) (τ := .dim channels (.dim height (.dim width .scalar))) dLdyAny
        let (dx, dgamma, dbeta) :=
          Spec.batchNorm2dBackward (x := x) (gamma := gamma)
            (grad_output := dLdy) h_c h_h h_w
        pure [
          (xId, AnyTensor.mk dx),
          (gammaId, AnyTensor.mk dgamma),
          (betaId, AnyTensor.mk dbeta)
        ]
    }
  pure (t.addNode node)

/--
Multi-head self-attention.

This is a shape-specialized attention primitive used by transformer-style models. It depends on an
optional boolean `(n,n)` mask and returns the attended output of shape `(n,dModel)`.

PyTorch comparison: similar to `torch.nn.MultiheadAttention` / scaled dot-product attention.
-/
def multiHeadAttention {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq
  Shape]
  {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (t : Tape α) (wqId wkId wvId woId xId : Nat)
  (mask : Option (Tensor Bool (.dim n (.dim n .scalar))) := none) :
  Result (Tape α × Nat) := do
  let wq ← requireValue (α:=α) (t:=t)
    (s:=.dim dModel (.dim (numHeads * headDim) .scalar)) wqId
  let wk ← requireValue (α:=α) (t:=t)
    (s:=.dim dModel (.dim (numHeads * headDim) .scalar)) wkId
  let wv ← requireValue (α:=α) (t:=t)
    (s:=.dim dModel (.dim (numHeads * headDim) .scalar)) wvId
  let wo ← requireValue (α:=α) (t:=t)
    (s:=.dim (numHeads * headDim) (.dim dModel .scalar)) woId
  let x ← requireValue (α:=α) (t:=t) (s:=.dim n (.dim dModel .scalar)) xId
  let mha : Spec.MultiHeadAttention α numHeads dModel headDim :=
    { Wq := wq, Wk := wk, Wv := wv, Wo := wo }
  let y := Spec.MultiHeadAttention.forward (n := n) (h1 := h1) (mha := mha) (x := x) (mask := mask)
  let node : Node α :=
    { name := some "multi_head_attention"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [wqId, wkId, wvId, woId, xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := .dim n (.dim dModel .scalar)) dLdyAny
        let (dx, dWq, dWk, dWv, dWo) :=
          Spec.MultiHeadAttentionBackward (h1 := h1) (mha := mha) (x := x) (mask := mask)
            (grad_output := dLdy)
        pure [
          (xId, AnyTensor.mk dx),
          (wqId, AnyTensor.mk dWq),
          (wkId, AnyTensor.mk dWk),
          (wvId, AnyTensor.mk dWv),
          (woId, AnyTensor.mk dWo)
        ]
    }
  pure (t.addNode node)

/--
 Elementwise logistic sigmoid activation.

 This builds a tape node whose forward pass is `Activation.sigmoid_spec`, and whose backward pass
 multiplies the upstream gradient by `Activation.sigmoid_deriv_spec` (i.e. `σ(x) * (1 - σ(x))`,
 pointwise).

 PyTorch comparison: `torch.sigmoid` / `torch.nn.functional.sigmoid`.
 Reference: https://pytorch.org/docs/stable/generated/torch.sigmoid.html
 -/
def sigmoid {α : Type} [Context α] [DecidableEq Shape]
  {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  let y := Activation.sigmoidSpec (α:=α) x
  let node : Node α :=
    { name := some "sigmoid"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        let dsig := Activation.sigmoidDerivSpec (α := α) x
        pure [(xId, AnyTensor.mk (mulSpec dsig dLdy))]
    }
  pure (t.addNode node)

/--
 Elementwise hyperbolic tangent activation.

 Forward uses `Activation.tanh_spec`; backward uses `Activation.tanh_deriv_spec` (pointwise
 derivative, usually `1 - tanh(x)^2`).

 PyTorch comparison: `torch.tanh`.
 Reference: https://pytorch.org/docs/stable/generated/torch.tanh.html
 -/
def tanh {α : Type} [Context α] [DecidableEq Shape]
  {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  let y := Activation.tanhSpec (α:=α) x
  let node : Node α :=
    { name := some "tanh"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        let dtanh := Activation.tanhDerivSpec (α := α) x
        pure [(xId, AnyTensor.mk (mulSpec dtanh dLdy))]
    }
  pure (t.addNode node)

/--
 Softmax along the last axis (recursing over outer dimensions).

 This matches `Activation.softmax_spec` (which applies softmax to the final dimension and recurses
 over earlier dimensions). The backward pass uses the standard Jacobian-vector product implemented
 by `Activation.softmax_backward_spec`, avoiding materializing an `n×n` Jacobian per slice.

 PyTorch comparison: `torch.softmax(x, dim=-1)`.
 Reference: https://pytorch.org/docs/stable/generated/torch.softmax.html
 -/
def softmax {α : Type} [Context α] [DecidableEq Shape]
  {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  let y := Activation.softmaxSpec (α:=α) x
  let node : Node α :=
    { name := some "softmax"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        let dx := Activation.softmaxBackwardSpec (α := α) (s := s) x dLdy
        pure [(xId, AnyTensor.mk dx)]
    }
  pure (t.addNode node)

/--
Stable log-softmax along the last axis.

Unlike `log (softmax x)`, this uses `Activation.logSoftmaxSpec`, i.e. the max-shifted
`x - max(x) - log(sum(exp(x - max(x))))` formulation.  That matches the numerical contract of
`torch.nn.functional.log_softmax` and is the right primitive for cross-entropy on logits.
-/
def logSoftmax {α : Type} [Context α] [DecidableEq Shape]
  {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  let y := Activation.logSoftmaxSpec (α:=α) x
  let node : Node α :=
    { name := some "log_softmax"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        let dx := Activation.logSoftmaxBackwardSpec (α := α) (s := s) y dLdy
        pure [(xId, AnyTensor.mk dx)]
    }
  pure (t.addNode node)

/--
 Elementwise softplus activation.

 Forward uses `Activation.softplus_spec`; backward uses `Activation.softplus_deriv_spec`.

 PyTorch comparison: `torch.nn.functional.softplus`.
 Reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.softplus.html
 -/
def softplus {α : Type} [Context α] [DecidableEq Shape]
  {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  let y := Activation.softplusSpec (α:=α) x
  let node : Node α :=
    { name := some "softplus"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        let dsoft := Activation.softplusDerivSpec (α := α) x
        pure [(xId, AnyTensor.mk (mulSpec dsoft dLdy))]
    }
  pure (t.addNode node)

/--
 Elementwise exponential.

 Forward uses `exp_spec`; backward multiplies by `exp(x)` (pointwise), i.e. `d/dx exp(x) = exp(x)`.

 PyTorch comparison: `torch.exp`.
 Reference: https://pytorch.org/docs/stable/generated/torch.exp.html
 -/
def exp {α : Type} [Context α] [DecidableEq Shape]
  {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  let y := expSpec (α:=α) x
  let node : Node α :=
    { name := some "exp"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        pure [(xId, AnyTensor.mk (mulSpec (expSpec (α := α) x) dLdy))]
    }
  pure (t.addNode node)

/--
 Elementwise natural logarithm.

 Forward uses `log_spec`; backward multiplies by `1/x` (pointwise), i.e. `d/dx log(x) = 1/x`
 (on its mathematical domain; this runtime does not model NaNs/Infs explicitly).

 PyTorch comparison: `torch.log`.
 Reference: https://pytorch.org/docs/stable/generated/torch.log.html
 -/
def log {α : Type} [Context α] [DecidableEq Shape]
  {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  -- `log` is only defined on positive inputs (and `d/dx log(x) = 1/x` blows up as `x → 0⁺`).
  -- Rather than implicitly relying on backend NaN/Inf behavior, we make the precondition explicit
  -- and ask users to opt into `safe_log` when they want epsilon protection.
  if !(allSpec (α := α) (s := s) (fun v => decide (v > (0 : α))) x) then
    throw "autograd: log: input contains values <= 0 (or NaN); use `safe_log` if you want epsilon protection"
  let y := logSpec (α:=α) x
  let node : Node α :=
    { name := some "log"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        pure [(xId, AnyTensor.mk (mulSpec (invSpec (α := α) x) dLdy))]
    }
  pure (t.addNode node)

/--
 Elementwise reciprocal `x ↦ 1/x`.

 Backward implements `d/dx (x⁻¹) = -(x⁻¹)²` (pointwise).

 PyTorch comparison: `torch.reciprocal`.
 Reference: https://pytorch.org/docs/stable/generated/torch.reciprocal.html
 -/
def inv {α : Type} [Context α] [DecidableEq Shape]
  {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  let y := invSpec (α := α) x
  let node : Node α :=
    { name := some "inv"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        -- d/dx (x⁻¹) = -(x⁻¹)²
        let invx := invSpec (α := α) x
        let invx2 := mulSpec invx invx
        let dx := scaleSpec (α := α) (s := s) (mulSpec dLdy invx2) (-1 : α)
        pure [(xId, AnyTensor.mk dx)]
    }
  pure (t.addNode node)

/--
 Elementwise "safe log" that protects against `log(0)` by adding a small `ε` internally.

 This uses `Activation.safe_log_spec` and `Activation.safe_log_deriv_spec`. The exact behavior is
 controlled by the spec-layer definition; conceptually it is similar to `log(x + ε)` used in
 numerically-stable losses.

 PyTorch comparison: commonly written as `torch.log(x + eps)` in user code (there is no single
 dedicated `torch.safe_log` primitive).
 -/
def safeLog {α : Type} [Context α] [DecidableEq Shape]
  {s : Shape} (t : Tape α) (xId : Nat) (ε : α := Numbers.epsilon) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  let y := Activation.safeLogSpec (α:=α) x ε
  let node : Node α :=
    { name := some "safe_log"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        let dlog := Activation.safeLogDerivSpec (α := α) x ε
        pure [(xId, AnyTensor.mk (mulSpec dlog dLdy))]
    }
  pure (t.addNode node)

/--
 Reduce-sum over all entries, producing a scalar node.

 Backward replicates the upstream scalar gradient to every entry of the input tensor (i.e.
 `d/dx Σ_i x_i = 1` per coordinate).

 PyTorch comparison: `torch.sum(x)` with `dim=None`.
 Reference: https://pytorch.org/docs/stable/generated/torch.sum.html
 -/
def sum {α : Type} [Add α] [Zero α] [DecidableEq Shape]
  {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  let y : Tensor α Shape.scalar := Tensor.scalar (sumSpec (α:=α) x)
  let node : Node α :=
    { name := some "sum"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := Shape.scalar) dLdyAny
        pure [(xId, AnyTensor.mk (replicate (α := α) (s := s) dLdy))]
    }
  pure (t.addNode node)

/--
 Mean-squared error (MSE) scalar loss with `"mean"` reduction over all entries.

 `mse_spec_basic` is the scalar loss `(Σ_i (yhat_i - target_i)^2) / N` where `N = Shape.size s`.
 This matches the default reduction of `torch.nn.functional.mse_loss(..., reduction="mean")`.

 Note: the derivative is defined everywhere in this spec-level setting; we do not model NaNs/Infs.
 -/
def mseSpecBasic {α : Type} [Add α] [Sub α] [Mul α] [Div α] [Zero α] [Coe Nat α]
  {s : Shape} (predicted target : Tensor α s) : α :=
  let diff := subSpec predicted target
  let squared := mulSpec diff diff
  let sum := sumSpec (α:=α) (s:=s) squared
  sum / (Shape.size s : α)

/--
 Gradient of `mse_spec_basic` with respect to `predicted` (same shape as the inputs).

 If `mse = (Σ_i (yhat_i - target_i)^2) / N`, then:
 `∂mse/∂yhat = (2/N) * (yhat - target)`.
 -/
def mseDerivSpecBasic {α : Type} [Add α] [Sub α] [Mul α] [Div α] [Zero α] [One α] [Coe Nat α]
  {s : Shape} (predicted target : Tensor α s) : Tensor α s :=
  let diff := subSpec predicted target
  let two : α := (1 : α) + 1
  scaleSpec (α:=α) (s:=s) diff (two / (Shape.size s : α))

/--
 Tape node for MSE loss with `"mean"` reduction.

 The forward value is a scalar. The backward pass returns gradients for both inputs:
 `dL/dyhat` from `mse_deriv_spec_basic`, and `dL/dtarget = - dL/dyhat`.

 PyTorch comparison: `torch.nn.functional.mse_loss`.
 Reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html
 -/
def mseLoss {α : Type}
  [Add α] [Sub α] [Mul α] [Div α] [Zero α] [One α] [Coe Nat α] [DecidableEq Shape]
  {s : Shape} (t : Tape α) (yhatId targetId : Nat) : Result (Tape α × Nat) := do
  let yhat ← requireValue (α:=α) (t:=t) (s:=s) yhatId
  let target ← requireValue (α:=α) (t:=t) (s:=s) targetId
  let y : Tensor α Shape.scalar := Tensor.scalar (mseSpecBasic (α:=α) (s:=s) yhat target)
  let node : Node α :=
    { name := some "mse_loss"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [yhatId, targetId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := Shape.scalar) dLdyAny
        let g : α := Tensor.toScalar dLdy
        let dYhat :=
          scaleSpec (α:=α) (s:=s) (mseDerivSpecBasic (α:=α) (s:=s) yhat target) g
        let dTarget : Tensor α s := subSpec (fill (0 : α) s) dYhat
        pure [(yhatId, AnyTensor.mk dYhat), (targetId, AnyTensor.mk dTarget)]
    }
  pure (t.addNode node)

/-!
## Backpropagation

Reverse-mode is implemented by traversing node ids in reverse order. Each node’s `backward`
closure produces parent-gradient contributions, which we accumulate by elementwise summation.
-/

/--
 Internal helper: add a single parent gradient contribution into the dense optional gradient array.

 This is where we implement PyTorch-style accumulation for DAGs: if multiple children contribute
 to the same parent id, we sum the contributions.

 The dense array entry is `none` until we first reach a node during reverse traversal.
 -/
def addGradDense
  {α : Type} [Add α] [DecidableEq Shape]
  (t : Tape α) (grads : Array (Option (Runtime.AnyTensor α)))
  (id : Nat) (g : Runtime.AnyTensor α) : Result (Array (Option (Runtime.AnyTensor α))) := do
  let node ← match t.getNode? id with
    | some n => pure n
    | none => throw "autograd: invalid parent id during backward"
  if node.requires_grad = false then
    pure grads
  else if h : g.s = node.value.s then
    let g' : Runtime.AnyTensor α := { s := node.value.s, t := Tensor.castShape g.t h }
    if hid : id < grads.size then
      match grads[id]'hid with
      | none =>
          pure (grads.set id (some g') (h := hid))
      | some existing =>
          let summed ← AnyTensor.add existing g'
          pure (grads.set id (some summed) (h := hid))
    else
      throw "autograd: internal error (gradient array out of bounds)"
  else
    throw "autograd: gradient contribution has wrong shape for parent"

/--
Reverse-mode backpropagation producing a dense array of optional gradients.

- The result array has length `t.nodes.size`.
- Entry `id` is `some g` if the node was reached from `outId` during reverse traversal, otherwise
  `none`.
- When multiple paths contribute to the same node, we sum gradients via `AnyTensor.add`.

This is loosely analogous to PyTorch's autograd engine walking the dynamic graph and accumulating
`.grad` for leaf tensors, but we keep gradients for every node id, not just leaves. That makes the
runtime easier to debug and gives proof-bridge code direct access to intermediate cotangents.

Reference (PyTorch): https://pytorch.org/docs/stable/notes/autograd.html
-/
def backwardDense {α : Type} [Add α] [DecidableEq Shape]
  (t : Tape α) (outId : Nat) (seed : Runtime.AnyTensor α) :
  Result (Array (Option (Runtime.AnyTensor α))) := do
  let outNode ← match t.getNode? outId with
    | some n => pure n
    | none => throw "autograd: invalid output id"
  if h : seed.s = outNode.value.s then
    let seed' : Runtime.AnyTensor α := { s := outNode.value.s, t := Tensor.castShape seed.t h }
    let mut grads : Array (Option (Runtime.AnyTensor α)) := Array.replicate t.nodes.size none
    if hout : outId < grads.size then
      grads := grads.set outId (some seed') (h := hout)
    else
      throw "autograd: invalid output id"
    let ids := (List.range t.nodes.size).reverse
    ids.foldlM (fun acc id => do
      match acc[id]? with
      | none => throw "autograd: internal error (gradient array out of bounds)"
      | some none => pure acc
      | some (some dLdy) =>
        let node ← match t.getNode? id with
          | some n => pure n
          | none => throw "autograd: internal error (node missing)"
        if node.requires_grad = false then
          pure acc
        else
          let contribs ← node.backward dLdy
          contribs.foldlM (fun acc2 (pid, pg) => addGradDense (t:=t) acc2 pid pg) acc
    ) grads
  else
    throw "autograd: seed gradient shape mismatch for output"

/--
Internal helper: like `addGradDense`, but assumes the gradient array is total (no `Option`).

This is used by the proof-friendly variants (`backwardDenseFrom*`, `backwardDenseAll`) that keep
an explicit zero tensor for nodes that do not receive gradients.
-/
def addGradAll
  {α : Type} [Add α] [DecidableEq Shape]
  (t : Tape α) (grads : Array (Runtime.AnyTensor α))
  (id : Nat) (g : Runtime.AnyTensor α) : Result (Array (Runtime.AnyTensor α)) := do
  let node ← match t.getNode? id with
    | some n => pure n
    | none => throw "autograd: invalid parent id during backward"
  if node.requires_grad = false then
    pure grads
  else if h : g.s = node.value.s then
    let g' : Runtime.AnyTensor α := { s := node.value.s, t := Tensor.castShape g.t h }
    match grads[id]? with
    | none => throw "autograd: internal error (gradient array out of bounds)"
      | some existing =>
          if hex : existing.s = node.value.s then
            let existing' : Runtime.AnyTensor α :=
              { s := node.value.s, t := Tensor.castShape existing.t hex }
            let summed ← AnyTensor.add existing' g'
            if hid : id < grads.size then
              pure (grads.set id summed (h := hid))
            else
              throw "autograd: internal error (gradient array out of bounds)"
          else
            throw "autograd: gradient array has wrong shape for node"
  else
    throw "autograd: gradient contribution has wrong shape for parent"

/--
One reverse-mode backprop step at a single node id, updating a total dense gradient array.

Precondition by convention: `acc` has one entry per tape node, and every entry has the matching
node shape. The function checks those conditions dynamically and returns an error if a caller
violates them. This makes it suitable as the small proof-friendly step used by
`backwardDenseFromLoop`.
-/
def backwardDenseFromStep {α : Type} [Add α] [DecidableEq Shape]
  (t : Tape α) (acc : Array (Runtime.AnyTensor α)) (id : Nat) :
  Result (Array (Runtime.AnyTensor α)) := do
  let node ← match t.getNode? id with
    | some n => pure n
    | none => throw "autograd: internal error (node missing)"
  if node.requires_grad = false then
    pure acc
  else
    let dLdyAny ← match acc[id]? with
      | some g => pure g
      | none => throw "autograd: internal error (gradient array out of bounds)"
    if hshape : dLdyAny.s = node.value.s then
      let dLdy : Runtime.AnyTensor α := { s := node.value.s, t := Tensor.castShape dLdyAny.t hshape
        }
      let contribs ← node.backward dLdy
      contribs.foldlM (fun acc2 (pid, pg) => addGradAll (t := t) acc2 pid pg) acc
    else
      throw "autograd: gradient array has wrong shape for node"

/--
Reverse-mode accumulation over the first `n` nodes in reverse order.

The recursion visits node ids `n-1, n-2, ..., 0`. Passing `n = t.nodes.size` therefore traverses the
entire tape. This structurally recursive loop is the one used by proof-linked compiled sessions.
-/
def backwardDenseFromLoop {α : Type} [Add α] [DecidableEq Shape]
  (t : Tape α) : Nat → Array (Runtime.AnyTensor α) → Result (Array (Runtime.AnyTensor α))
  | 0, acc => pure acc
  | n + 1, acc => do
      let acc' ← backwardDenseFromStep (t := t) acc n
      backwardDenseFromLoop (t := t) n acc'

/--
Reverse-mode accumulation starting from an explicit dense gradient array.

This is a proof-friendly variant: it always runs every node (in reverse order) and keeps a
gradient tensor for every node id.
-/
def backwardDenseFrom {α : Type} [Add α] [DecidableEq Shape]
  (t : Tape α) (grads0 : Array (Runtime.AnyTensor α)) :
  Result (Array (Runtime.AnyTensor α)) := do
  if grads0.size = t.nodes.size then
    backwardDenseFromLoop (t := t) t.nodes.size grads0
  else
    throw "autograd: initial dense gradient array has wrong length"

/-- Reverse-mode accumulation that returns a dense gradient array for every node id.

This differs from `backwardDense`: instead of leaving entries as `none` until they are reached,
it initializes a zero gradient tensor for each node. This matches the proof-level tape model
where gradients are explicit (zero for unused nodes).
-/
def backwardDenseAll {α : Type} [Add α] [Zero α] [DecidableEq Shape]
  (t : Tape α) (outId : Nat) (seed : Runtime.AnyTensor α) :
  Result (Array (Runtime.AnyTensor α)) := do
  let outNode ← match t.getNode? outId with
    | some n => pure n
    | none => throw "autograd: invalid output id"
  if h : seed.s = outNode.value.s then
    let seed' : Runtime.AnyTensor α := { s := outNode.value.s, t := Tensor.castShape seed.t h }
    let mut grads : Array (Runtime.AnyTensor α) :=
      t.nodes.map (fun node => AnyTensor.mk (fill (0 : α) node.value.s))
    if hout : outId < grads.size then
      grads := grads.set outId seed' (h := hout)
    else
      throw "autograd: invalid output id"
    backwardDenseFrom (t := t) grads
  else
    throw "autograd: seed gradient shape mismatch for output"

/--
Convert the optional dense gradient array returned by `backwardDense` into a sparse `HashMap`.

Only entries that are present (`some (some g)`) are kept. This is mainly a convenience for callers
that want "just the reached nodes" without carrying `none`s around.
-/
def denseToHashMap {α : Type}
  (grads : Array (Option (Runtime.AnyTensor α))) :
  Std.HashMap Nat (Runtime.AnyTensor α) :=
  (List.range grads.size).foldl (fun acc id =>
    match grads[id]? with
    | some (some g) => acc.insert id g
    | _ => acc
  ) (Std.HashMap.emptyWithCapacity)

/--
Reverse-mode backpropagation returning a `HashMap` of only the nodes that received gradients.

This is a convenience wrapper around `backwardDense` + `denseToHashMap`.
-/
def backward {α : Type} [Add α] [DecidableEq Shape]
  (t : Tape α) (outId : Nat) (seed : Runtime.AnyTensor α) :
  Result (Std.HashMap Nat (Runtime.AnyTensor α)) := do
  let dense ← backwardDense (t := t) outId seed
  pure (denseToHashMap dense)

/--
Backpropagate from a scalar output with seed gradient `1`.

PyTorch analogy: `loss.backward()` when `loss` is a scalar.
-/
def backwardScalar {α : Type} [Add α] [One α] [DecidableEq Shape]
  (t : Tape α) (outId : Nat) : Result (Std.HashMap Nat (Runtime.AnyTensor α)) :=
  backward (t:=t) outId (AnyTensor.mk (Tensor.scalar (1 : α)))

end Tape
end Autograd
end Runtime
