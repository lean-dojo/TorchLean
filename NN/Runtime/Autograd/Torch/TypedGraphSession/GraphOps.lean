/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.TypedGraphSession.Core
public import NN.Runtime.Autograd.TypedGraph.GraphM.Elementwise

/-!
# Typed Graph Session: Basic Graph Operations
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Internal

namespace TypedGraphSession

/-! ## Graph-node operations (implemented by reusing `TypedGraph.GraphM`) -/

/-- Results returned from `commitGraphM` that carry a session reference identity. -/
class StampRefIdentity (β : Type) where
  stamp : RefIdentity → β → β

instance {α : Type} {sh : Shape} : StampRefIdentity (TensorRef α sh) where
  stamp identity ref := { ref with identity? := some identity }

/--
Run a `TypedGraph.GraphM` computation against the current `(ss, g)` pair.

`TypedGraph.GraphM` is the builder monad used by typed graph execution; reusing it
here ensures this eager-style API records the same typed graph used by lowering.
-/
def runGraphM {α : Type} [TorchLean.Storage α] {Γ : List Shape} {β : Type}
    (m : Runtime.Autograd.TypedGraph.GraphM.MWith α NatEnv Γ β)
    (ss : List Shape) (g : Proofs.Autograd.Algebra.GraphData α NatEnv Γ ss) :
    Runtime.Autograd.Result (β × (Σ ss' : List Shape, Proofs.Autograd.Algebra.GraphData α
      NatEnv Γ ss')) := do
  let (value, state) ← StateT.run m ⟨ss, g, #[]⟩
  pure (value, ⟨state.nodeShapes, state.data⟩)

/--
Atomically apply a graph-building update to the session snapshot.

This is the central adapter used by each op wrapper below: it reads `s.state`, runs a builder that
returns an updated `TypedGraphSessionState`, stores it back into `s.state`, and returns the result.
-/
def commitGraphM {α : Type} [TorchLean.Storage α]
    (s : TypedGraphSession α) {β : Type} [StampRefIdentity β]
    (refs : Array (Option RefIdentity) := #[])
    (k :
      ∀ {Γ : List Shape} {ss : List Shape},
        (x : TorchLean.TensorPack α Γ) →
        (nat : NatEnv) →
        (g : Proofs.Autograd.Algebra.GraphData α NatEnv Γ ss) →
        Runtime.Autograd.Result (β × TypedGraphSessionState α)) :
    IO β := do
  s.validateRefIdentities refs
  let st0 ← s.state.get
  let r ← okOrThrow (k (Γ := st0.Γ) (ss := st0.ss) st0.x st0.nat st0.g)
  let (b, st1) := r
  s.setState { st1 with leafMetadata := st0.leafMetadata }
  pure <| StampRefIdentity.stamp (← s.currentRefIdentity) b

/--
Record a constant tensor.

Subtlety: if no op nodes have been created yet (`ss = []`), we record `const` as a leaf to match
the eager session's leaf-collection behavior. Once op nodes exist, we emit an explicit constant node
so users can introduce literal constants mid-graph.
PyTorch comparison: like `torch.tensor(...)` (a leaf) vs inserting a literal constant into the
graph; constants are treated as non-requires-grad.
-/
def const {α : Type} [TorchLean.Storage α]
    (s : TypedGraphSession α) {sh : Shape} [Zero α]
  (v : Tensor α sh) (name : Option String := none) : IO (TensorRef α sh) := do
  let _ := name
  let st0 ← s.state.get
  match st0.ss with
  | [] =>
      -- Still in the "leaf collection" phase: keep `const` as a leaf for parity with the eager
      -- Session.
      input (α := α) s (sh := sh) v (name := name) (requiresGrad := false)
  | _ :: _ =>
      -- Mid-graph: emit an explicit constant node.
      commitGraphM (α := α) s (β := TensorRef α sh) (refs := #[]) (fun {Γ} {ss} xv nat g => do
        let (vout, st') ← runGraphM (α := α) (Γ := Γ)
          (Runtime.Autograd.TypedGraph.GraphM.const (α := α) (Γ := Γ) (s := sh) v)
          ss g
        let ⟨ss', g'⟩ := st'
        let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
        pure ({ id := vout.id }, st1))

/--
Record elementwise addition `a + b`.

PyTorch comparison: `torch.add(a, b)` / the `+` operator.
-/
def add {α : Type} [TorchLean.Storage α]
    (s : TypedGraphSession α) [Add α] [Zero α] {sh : Shape}
  (a b : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (refs := #[a.identity?, b.identity?])
      (fun {Γ} {ss} x nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.add (α := α) (Γ := Γ) (s := sh)
        { id := a.id } { id := b.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := x, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record elementwise subtraction `a - b`.

PyTorch comparison: `torch.sub(a, b)` / the `-` operator.
-/
def sub {α : Type} [TorchLean.Storage α]
    (s : TypedGraphSession α) [Sub α] [Add α] [Zero α] {sh : Shape}
  (a b : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (refs := #[a.identity?, b.identity?])
      (fun {Γ} {ss} x nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.sub (α := α) (Γ := Γ) (s := sh)
        { id := a.id } { id := b.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := x, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record elementwise multiplication `a * b`.

PyTorch comparison: `torch.mul(a, b)` / the `*` operator.
-/
def mul {α : Type} [TorchLean.Storage α]
    (s : TypedGraphSession α) [Mul α] [Add α] [Zero α] {sh : Shape}
  (a b : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (refs := #[a.identity?, b.identity?])
      (fun {Γ} {ss} x nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.mul (α := α) (Γ := Γ) (s := sh)
        { id := a.id } { id := b.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := x, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record scaling by a scalar constant: `x * c`.

PyTorch comparison: like `x * c` (where `c` is a Python scalar).
-/
def scale {α : Type} [TorchLean.Storage α]
    (s : TypedGraphSession α) [Mul α] [Add α] [Zero α] {sh : Shape}
  (x : TensorRef α sh) (c : α) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (refs := #[x.identity?])
      (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.scale (α := α) (Γ := Γ) (s := sh) { id := x.id } c)
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record elementwise absolute value.

PyTorch comparison: `torch.abs(x)`.
-/
def abs {α : Type} [TorchLean.Storage α] (s : TypedGraphSession α)
  [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (refs := #[x.identity?])
      (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.abs (α := α) (Γ := Γ) (s := sh) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Stop-gradient boundary.

Forward semantics: identity.
Backward semantics: no gradient flows to the input.
PyTorch comparison: `x.detach()`.
-/
def detach {α : Type} [TorchLean.Storage α]
    (s : TypedGraphSession α) [Context α] {sh : Shape}
    (x : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (refs := #[x.identity?])
      (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.detach (α := α) (Γ := Γ) (s := sh) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record elementwise square root.

PyTorch comparison: `torch.sqrt(x)`.
-/
def sqrt {α : Type} [TorchLean.Storage α] (s : TypedGraphSession α)
  [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (refs := #[x.identity?])
      (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.sqrt (α := α) (Γ := Γ) (s := sh) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record elementwise clamp to the interval `[minVal, maxVal]`.

PyTorch comparison: `torch.clamp(x, min=minVal, max=maxVal)`.
-/
def clamp {α : Type} [TorchLean.Storage α] (s : TypedGraphSession α)
  [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {sh : Shape} (x : TensorRef α sh) (minVal maxVal : α) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (refs := #[x.identity?])
      (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.clamp (α := α) (Γ := Γ) (s := sh) { id := x.id } minVal
        maxVal)
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record elementwise maximum of `a` and `b`.

PyTorch comparison: `torch.maximum(a, b)`.
-/
def max {α : Type} [TorchLean.Storage α] (s : TypedGraphSession α)
  [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {sh : Shape} (a b : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (refs := #[a.identity?, b.identity?])
      (fun {Γ} {ss} x nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.max (α := α) (Γ := Γ) (s := sh)
        { id := a.id } { id := b.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := x, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record elementwise minimum of `a` and `b`.

PyTorch comparison: `torch.minimum(a, b)`.
-/
def min {α : Type} [TorchLean.Storage α] (s : TypedGraphSession α)
  [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {sh : Shape} (a b : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (refs := #[a.identity?, b.identity?])
      (fun {Γ} {ss} x nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.min (α := α) (Γ := Γ) (s := sh)
        { id := a.id } { id := b.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := x, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/-- Record matrix multiplication with broadcasted batch prefixes. -/
def matmul {α : Type} [TorchLean.Storage α]
    (s : TypedGraphSession α) [Context α]
  {batchA batchB batch : Shape} {m n p : Nat}
  [broadcastA : Shape.BroadcastTo batchA batch]
  [broadcastB : Shape.BroadcastTo batchB batch]
  (a : TensorRef α (batchA.concat [m, n]))
  (b : TensorRef α (batchB.concat [n, p])) :
  IO (TensorRef α (batch.concat [m, p])) :=
  commitGraphM (α := α) s (β := TensorRef α (batch.concat [m, p]))
      (refs := #[a.identity?, b.identity?]) (fun {Γ} {ss} x nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.matmul (α := α) (Γ := Γ)
        (batchA := batchA) (batchB := batchB) (batch := batch)
        (m := m) (n := n) (p := p) { id := a.id } { id := b.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := x, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Concatenate two tensors along dimension 0.

PyTorch comparison: `torch.cat([a, b], dim=0)`.
-/
def concatLeadingAxis {α : Type} [TorchLean.Storage α]
    (s : TypedGraphSession α) [Context α]
  {n m : Nat} {sh : Shape}
  (a : TensorRef α (.dim n sh))
  (b : TensorRef α (.dim m sh)) :
  IO (TensorRef α (.dim (n + m) sh)) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim (n + m) sh))
      (refs := #[a.identity?, b.identity?]) (fun {Γ} {ss} x nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.concatLeadingAxis (α := α) (Γ := Γ) (n := n) (m := m)
        (s := sh) { id := a.id } { id := b.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := x, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Slice a tensor along dimension 0.

This returns `x[start : start+len]`. The proof argument `h` enforces bounds.
PyTorch comparison: `x[start:start+len]` for tensors with a leading dimension.
-/
def sliceLeadingAxisRange {α : Type} [TorchLean.Storage α]
    (s : TypedGraphSession α) [Zero α]
  {n : Nat} {sh : Shape}
  (x : TensorRef α (.dim n sh)) (start len : Nat) (h : start + len ≤ n) :
  IO (TensorRef α (.dim len sh)) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim len sh)) (refs := #[x.identity?])
      (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.sliceLeadingAxisRange (α := α) (Γ := Γ) (n := n) (s := sh)
        { id := x.id } start len h)
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))
end TypedGraphSession

end Internal

end Torch
end Autograd
end Runtime
