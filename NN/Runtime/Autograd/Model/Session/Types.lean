/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Session.Eager
public import NN.Runtime.Autograd.Torch.TypedGraphSession.GraphOps

/-!
Session value types.

The definitions here describe runtime tensor handles, parameter references, and shape-indexed
containers used by the TorchLean session API.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Model

open Spec TorchLean
open TorchLean TorchLean.Tensor

/-- Active execution state owned by a `Session`. -/
inductive SessionState (α : Type) [TorchLean.Storage α] where
  | eager (s : EagerSession α)
  | typedGraph (s : Runtime.Autograd.Torch.Internal.TypedGraphSession α)

/--
Unified session selected by `options.execution`.

This is the recommended "one interface" for:
- training/debugging (eager),
- shape-indexed typed graph recording and execution,
without users having to learn two different Session APIs.
-/
structure Session (α : Type) [TorchLean.Storage α] where
  /-- Scalar, device, and execution options used to create the session. -/
  options : Runtime.Autograd.Torch.Config
  /-- Runtime state for the selected execution mode. -/
  state : SessionState α

namespace Session

/--
Create a new unified session.

The execution mode is selected by `options.execution`: `.eager` builds a dynamic tape, while
`.typedGraph` records a shape-indexed typed SSA graph.
-/
def new {α : Type} [TorchLean.Storage α]
    (options : Runtime.Autograd.Torch.Config := {}) : IO (Session α) := do
  match options.execution with
  | .eager =>
      let s ← EagerSession.new (α := α) (options := options)
      pure { options := options, state := .eager s }
  | .typedGraph =>
      let s ← Runtime.Autograd.Torch.Internal.TypedGraphSession.new (α := α) (options := options)
      pure { options := options, state := .typedGraph s }

/-- Reset the autograd tape or begin a fresh typed-graph recording phase. -/
def resetTape {α : Type} [TorchLean.Storage α] (s : Session α) : IO Unit := do
  match s.state with
  | .eager sess => EagerSession.resetTape (α := α) sess
  | .typedGraph sess => Runtime.Autograd.Torch.Internal.TypedGraphSession.resetTape (α := α) sess

/--
Create a learnable parameter (a leaf tensor) owned by the session.

PyTorch analogue: `torch.nn.Parameter` (conceptually), created inside a module/init and later used
in forward passes.
-/
def param {α : Type} [TorchLean.Storage α] (s : Session α) {sh : Shape}
  (init : Tensor α sh) (name : Option String := none) (requiresGrad : Option Bool := none) :
  IO (Runtime.Autograd.Torch.Param α sh) := do
  match s.state with
  | .eager sess =>
      EagerSession.param (α := α) sess (sh := sh)
        init (name := name) (requiresGrad := requiresGrad)
  | .typedGraph sess =>
      Runtime.Autograd.Torch.Internal.TypedGraphSession.param (α := α) sess (sh := sh)
        init (name := name) (requiresGrad := requiresGrad)

/--
Use a parameter in the current recording phase.

This returns a `TensorRef` that can be passed to operations. Eager execution records the operation
on its tape; typed-graph execution records it in the current typed SSA graph.
-/
def use {α : Type} [TorchLean.Storage α] (s : Session α) {sh : Shape}
  [Runtime.Autograd.Torch.TensorTransfer α]
  (p : Runtime.Autograd.Torch.Param α sh) : IO (Runtime.Autograd.Torch.TensorRef α sh)
    := do
  match s.state with
  | .eager sess => EagerSession.use (α := α) sess (sh := sh) p
  | .typedGraph sess =>
      Runtime.Autograd.Torch.Internal.TypedGraphSession.use (α := α) sess (sh := sh) p

/--
Add an input tensor to the current recording phase.

Inputs are leaf tensors that may or may not require gradients (controlled by `requiresGrad`).
-/
def input {α : Type} [TorchLean.Storage α] (s : Session α) {sh : Shape}
  [Runtime.Autograd.Torch.TensorTransfer α]
  (v : Tensor α sh) (name : Option String := none) (requiresGrad : Bool := false) :
  IO (Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.state with
  | .eager sess =>
      EagerSession.input (α := α) sess (sh := sh) v (name := name) (requiresGrad := requiresGrad)
  | .typedGraph sess =>
      Runtime.Autograd.Torch.Internal.TypedGraphSession.input (α := α) sess (sh := sh)
        v (name := name) (requiresGrad := requiresGrad)

/--
Add a non-differentiable `Nat` input to the session.

This is used for labels/indices (e.g. classification targets, gather indices) without forcing a
numeric embedding into `α`.
-/
def inputNat {α : Type} [TorchLean.Storage α]
    (s : Session α) (v : Nat) : IO (Runtime.Autograd.Torch.NatRef) := do
  match s.state with
  | .eager sess => EagerSession.inputNat (α := α) sess v
  | .typedGraph sess => Runtime.Autograd.Torch.Internal.TypedGraphSession.inputNat (α := α) sess v

/-- Read back a `NatRef` value. -/
def getNat {α : Type} [TorchLean.Storage α]
    (s : Session α) (r : Runtime.Autograd.Torch.NatRef) : IO Nat := do
  match s.state with
  | .eager sess => EagerSession.getNat (α := α) sess r
  | .typedGraph sess => Runtime.Autograd.Torch.Internal.TypedGraphSession.getNat (α := α) sess r

/-- Mutate a `NatRef` value. -/
def setNat {α : Type} [TorchLean.Storage α]
    (s : Session α) (r : Runtime.Autograd.Torch.NatRef) (v : Nat) : IO Unit
  := do
  match s.state with
  | .eager sess => EagerSession.setNat (α := α) sess r v
  | .typedGraph sess => Runtime.Autograd.Torch.Internal.TypedGraphSession.setNat (α := α) sess r v

/-! ### Deterministic RNG state (Session-level) -/

/--
Deterministic RNG state stored inside a session.

We model RNG state explicitly using two non-differentiable leaves:
- `seed`: the current seed value
- `counter`: a monotone counter used to derive fresh keys

PyTorch analogy: explicit `torch.manual_seed` + per-op counter, but represented as explicit state.
-/
structure RngState where
  /-- Random seed. -/
  seed : Runtime.Autograd.Torch.NatRef
  /-- Monotone counter, bumped once per random draw. Seed plus counter is what makes the stream
  reproducible: replaying from the same seed reproduces every draw in order. -/
  counter : Runtime.Autograd.Torch.NatRef

/--
Initialize an `RngState` from a concrete seed.

This allocates two `NatRef`s in the session (`seed` and `counter`) and initializes `counter` to 0.
-/
def initRng {α : Type} [TorchLean.Storage α] (s : Session α) (seed : Nat) : IO RngState := do
  let seedRef ← inputNat (α := α) s seed
  let counterRef ← inputNat (α := α) s 0
  pure { seed := seedRef, counter := counterRef }

/-- Draw a fresh seed from `IO` (best-effort entropy). -/
def freshSeedIO : IO Nat := do
  -- We use `IO.rand` for practicality/ergonomics; this is *not* part of the semantic core.
  -- The semantic model remains seed-threaded deterministic RNG: this just chooses an initial seed.
  IO.rand 0 (Nat.pow 2 63 - 1)

/--
Initialize a deterministic RNG state by sampling an initial seed from `IO`.

This is the recommended "PyTorch-like ergonomics, JAX-like semantics" bridge:
- you get a convenient source of entropy at the boundary,
- but the *core* semantics remains deterministic and replayable given the chosen seed.
-/
def initRngFromIO {α : Type} [TorchLean.Storage α] (s : Session α) : IO RngState := do
  initRng (α := α) s (← freshSeedIO)

/-!
Practical note: in `.typedGraph` execution, the current session implementation
requires that all leaves (tensor inputs/parameters and `NatRef`s) are created before any op nodes.
So for maximum portability, initialize and split RNG states *up-front* before building a graph.
-/

/--
Insert a constant tensor into the current graph.

PyTorch analogy: using a tensor constant/literal in `forward`.
-/
def const {α : Type} [TorchLean.Storage α]
    (s : Session α) {sh : Shape} [Zero α]
  [Runtime.Autograd.Torch.TensorTransfer α]
  (v : Tensor α sh) (name : Option String := none) : IO (Runtime.Autograd.Torch.TensorRef α
    sh) := do
  match s.state with
  | .eager sess => EagerSession.const (α := α) sess (sh := sh) v (name := name)
  | .typedGraph sess =>
      Runtime.Autograd.Torch.Internal.TypedGraphSession.const (α := α) sess (sh := sh) v
        (name := name)

/-- Read the concrete value for a tensor ref (for logging/debugging). -/
def getValue {α : Type} [TorchLean.Storage α]
    (s : Session α) {sh : Shape}
  [Runtime.Autograd.Torch.TensorTransfer α]
  (x : Runtime.Autograd.Torch.TensorRef α sh) : IO (Tensor α sh) := do
  match s.state with
  | .eager sess => EagerSession.getValue (α := α) sess (sh := sh) x
  | .typedGraph sess =>
      Runtime.Autograd.Torch.Internal.TypedGraphSession.getValue (α := α) sess (sh := sh) x

/--
Detach a tensor ref from the graph (stop gradient flow through it).

PyTorch analogy: `x.detach()`.
-/
def detach {α : Type} [TorchLean.Storage α] (s : Session α) [Context α] {sh : Shape}
    [Runtime.Autograd.Torch.TensorTransfer α]
    (x : Runtime.Autograd.Torch.TensorRef α sh) :
    IO (Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.state with
  | .eager sess => EagerSession.detach (α := α) sess (sh := sh) x
  | .typedGraph sess =>
      Runtime.Autograd.Torch.Internal.TypedGraphSession.detach (α := α) sess (sh := sh) x


end Session

end Model
end Autograd
end Runtime
