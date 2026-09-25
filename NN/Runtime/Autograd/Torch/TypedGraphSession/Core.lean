/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Session
public import NN.Runtime.Autograd.TypedGraph.Compiled

/-!
# Typed graph sessions

This session records shape-indexed `GraphData` as operations are called. A backward pass compiles
the recorded graph at its current leaf values, then executes its saved reverse programs.
`TorchLean.Session` selects this implementation when `options.execution := .typedGraph`;
`Runtime.Autograd.Model.Session` provides the shared eager and typed graph interface.

Create all parameter and input leaves before recording the first operation. A training step
therefore resets the session, adds its leaves, runs the forward program, and calls backward.
Constants introduced during the forward program use `const` nodes, so they can appear after
other operations.

The theorem `backwardDenseFrom_lowerGraphDataToTape_eq_backpropAllCtx` identifies the lowered
tape's backward result with `GraphData.backpropAllCtx` for the same graph and leaf values.
It connects two executions of the stored VJP rules. To identify those rules with derivatives of
the forward operations, we also need the local correctness laws carried by
`Proofs.Autograd.Algebra.Node`.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace Torch

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Internal

/-- Non-differentiable external environment for the graph: a small array of `Nat` inputs. -/
abbrev NatEnv : Type := Array Nat

/-- Runtime metadata attached to one typed-graph leaf. -/
structure LeafMetadata where
  /-- Optional name used by tape diagnostics. -/
  name : Option String
  /-- Whether reverse mode accumulates a gradient for this leaf. -/
  requiresGrad : Bool

/-- Internal typed graph state: executable `GraphData` together with its leaf values. -/
structure TypedGraphSessionState (α : Type) [TorchLean.Storage α] where
  /-- Leaf shapes (inputs/parameters), in creation order. -/
  Γ : List Shape
  /-- Leaf values, aligned with `Γ`. -/
  x : TorchLean.TensorPack α Γ
  /-- Runtime metadata aligned with the leaves in `Γ`. -/
  leafMetadata : Array LeafMetadata := #[]
  /-- Non-differentiable external inputs (e.g. class labels/indices). -/
  nat : NatEnv
  /-- Internal node shapes, in creation order. -/
  ss : List Shape
  /-- SSA/DAG graph nodes (one per entry in `ss`). -/
  g : Proofs.Autograd.Algebra.GraphData α NatEnv Γ ss

namespace TypedGraphSessionState

/-- Empty session state: no leaves, no nodes, empty nat-environment. -/
def empty {α : Type} [TorchLean.Storage α] : TypedGraphSessionState α :=
  { Γ := []
    x := .nil
    leafMetadata := #[]
    nat := #[]
    ss := []
    g := .nil }

end TypedGraphSessionState

/--
`TypedGraphSession` is an imperative session that records executable `GraphData` as it runs.

Operations are called imperatively, but the resulting graph is explicit and shape-indexed. After
lowering, the runtime tape backward loop is provably equal to `GraphData.backpropAllCtx`; this is an
implementation-equivalence result, distinct from proving each stored VJP correct.
-/
structure TypedGraphSession (α : Type) [TorchLean.Storage α] where
  /-- Session options shared with the eager front-end. -/
  options : Config
  /-- Mutable executable graph snapshot. -/
  state : IO.Ref (TypedGraphSessionState α)
  /-- Map from graph leaf ids to mutable parameter objects. -/
  parametersByLeaf : IO.Ref (Std.HashMap Nat (AnyParam α))
  /-- Process-unique owner id for session references. -/
  referenceOwner : Nat
  /-- Current recording generation for session references. -/
  referenceGeneration : IO.Ref Nat
  /-- Counter bumped by `setState`, used to key `valueCache`. -/
  stateVersion : IO.Ref Nat
  /-- Evaluated context values for the state version they were computed from. -/
  valueCache : IO.Ref (Option (Nat × Array (Spec.SomeTensor α)))

namespace TypedGraphSession

/--
Create a new typed graph session.

This allocates `IO.Ref`s for the session snapshot (`TypedGraphSessionState`) and the map from leaf
identifiers to parameters. Call `resetTape` to begin a new graph recording phase.
-/
def new {α : Type} [TorchLean.Storage α] (options : Config := {}) : IO (TypedGraphSession α) := do
  unless options.device == .cpu do
    throw <| IO.userError
      s!"typed graph execution currently supports device `cpu`; requested `{options.deviceName}`"
  let state ← IO.mkRef (TypedGraphSessionState.empty (α := α))
  let parametersByLeaf ← IO.mkRef (Std.HashMap.emptyWithCapacity)
  let referenceOwner ← RefIdentity.freshOwner
  let referenceGeneration ← IO.mkRef 0
  let stateVersion ← IO.mkRef 0
  let valueCache ← IO.mkRef none
  pure
    { options, state, parametersByLeaf, referenceOwner, referenceGeneration, stateVersion
      valueCache }

/-- Replace the session snapshot. Every write goes through here so cached values stay current. -/
def setState {α : Type} [TorchLean.Storage α] (s : TypedGraphSession α)
    (st : TypedGraphSessionState α) : IO Unit := do
  s.state.set st
  s.stateVersion.modify (· + 1)
  s.valueCache.set none

/-- Capture the current owner and generation for a newly recorded handle. -/
def currentRefIdentity {α : Type} [TorchLean.Storage α]
    (s : TypedGraphSession α) : IO RefIdentity := do
  pure { owner := s.referenceOwner, generation := ← s.referenceGeneration.get }

/-- Construct a tensor handle owned by the current recording phase. -/
def makeTensorRef {α : Type} [TorchLean.Storage α] {sh : Shape}
    (s : TypedGraphSession α) (id : Nat) :
    IO (TensorRef α sh) := do
  pure { id, identity? := some (← s.currentRefIdentity) }

/-- Construct a non-differentiable handle owned by the current recording phase. -/
def makeNatRef {α : Type} [TorchLean.Storage α]
    (s : TypedGraphSession α) (id : Nat) : IO NatRef := do
  pure { id, identity? := some (← s.currentRefIdentity) }

/-- Validate one tensor handle before using its numeric graph id. -/
def validateTensorRef {α : Type} [TorchLean.Storage α]
    (s : TypedGraphSession α) {sh : Shape}
    (x : TensorRef α sh) : IO Unit := do
  match x.identity? with
  | some identity =>
      identity.validateAgainst s.referenceOwner s.referenceGeneration "tensor reference"
  | none => throw <| IO.userError "torch: tensor reference has no session owner"

/-- Validate tensor handles consumed by one graph operation. -/
def validateRefIdentities {α : Type} [TorchLean.Storage α]
    (s : TypedGraphSession α)
    (identities : Array (Option RefIdentity)) : IO Unit := do
  for identity? in identities do
    match identity? with
    | some identity =>
        identity.validateAgainst s.referenceOwner s.referenceGeneration "tensor reference"
    | none => throw <| IO.userError "torch: tensor reference has no session owner"

/-- Validate one non-differentiable handle before using its environment index. -/
def validateNatRef {α : Type} [TorchLean.Storage α]
    (s : TypedGraphSession α) (x : NatRef) : IO Unit := do
  match x.identity? with
  | some identity =>
      identity.validateAgainst s.referenceOwner s.referenceGeneration "Nat reference"
  | none => throw <| IO.userError "torch: Nat reference has no session owner"

/--
Reset the session to an empty snapshot.

Important invariant: this session requires that **all leaves are created before any op node**.
`resetTape` is the intended boundary between training steps/forwards.
-/
def resetTape {α : Type} [TorchLean.Storage α] (s : TypedGraphSession α) : IO Unit := do
  s.setState (TypedGraphSessionState.empty (α := α))
  s.parametersByLeaf.set (Std.HashMap.emptyWithCapacity)
  s.referenceGeneration.modify (fun generation => generation + 1)

/--
Create a mutable parameter object (not yet part of the recorded graph).

To use the parameter in the recorded graph, call `use`, which reads its current value and records
it as a *leaf* in `Γ`.
PyTorch comparison: analogous to creating a `torch.nn.Parameter` and then using it in a forward.
-/
def param {α : Type} [TorchLean.Storage α] (s : TypedGraphSession α) {sh : Shape}
  (init : Tensor α sh) (name : Option String := none) (requiresGrad : Option Bool := none) :
  IO (Param α sh) :=
  Param.Internal.create init name (requiresGrad.getD s.options.requiresGradByDefault)

/--
Enforce the session invariant: leaves must be created before any op node.

This matches the usual training pattern: `resetTape → add leaves → forward ops → backward`.
-/
def ensureNoNodes {α : Type} [TorchLean.Storage α]
    (st : TypedGraphSessionState α) : IO Unit := do
  match st.ss with
  | [] => pure ()
  | _ :: _ =>
      throw <| IO.userError
        ("torch(TypedGraphSession): cannot add a new leaf after graph nodes have been " ++
          "created (resetTape first)")

/--
Record a new differentiable leaf tensor in the session context `Γ`.

This is the primitive used by `use` (parameters) and `input` (external inputs).
-/
def addLeaf {α : Type} [TorchLean.Storage α]
    (s : TypedGraphSession α) {sh : Shape} (v : Tensor α sh)
    (name : Option String) (requiresGrad : Bool) :
    IO (TensorRef α sh) := do
  let st0 ← s.state.get
  ensureNoNodes st0
  let id := st0.Γ.length
  let Γ' := st0.Γ ++ [sh]
  let x' : TorchLean.TensorPack α Γ' :=
    TorchLean.TensorPack.snoc (α := α) (ss := st0.Γ) (τ := sh) st0.x v
  -- No nodes yet, so the graph stays `nil`.
  let st1 : TypedGraphSessionState α :=
    { Γ := Γ'
      x := x'
      leafMetadata := st0.leafMetadata.push { name, requiresGrad }
      nat := st0.nat
      ss := []
      g := .nil }
  s.setState st1
  s.makeTensorRef id

/--
Use a `Param` in the recorded graph by reading its current value and recording it as a leaf.

The returned `TensorRef` is the graph handle you pass to subsequent ops. The session also remembers
which leaf-id corresponds to which parameter, so `sgdStepAll` can update parameters after backward.
PyTorch comparison: like referencing a `torch.nn.Parameter` in the forward; the parameter's value
is treated as a leaf for autograd.
-/
def use {α : Type} [TorchLean.Storage α] [TensorTransfer α]
    (s : TypedGraphSession α) {sh : Shape}
  (p : Param α sh) : IO (TensorRef α sh) := do
  syncParamCudaToHost p
  let v ← p.value.get
  let leaf ← addLeaf (α := α) s (sh := sh) v p.name
    (s.options.gradEnabled && p.requiresGrad)
  s.parametersByLeaf.modify (fun parameters =>
    parameters.insert leaf.id (AnyParam.ofParam p))
  pure leaf

/--
Record an external input tensor as a leaf.

The input remains part of the typed context whether or not it is differentiable. The
`requiresGrad` flag controls gradient accumulation when the graph is lowered to a runtime tape;
`gradEnabled := false` overrides it for the whole session.
-/
def input {α : Type} [TorchLean.Storage α]
    (s : TypedGraphSession α) {sh : Shape}
  (v : Tensor α sh) (name : Option String := none) (requiresGrad : Bool := false) :
  IO (TensorRef α sh) :=
  addLeaf (α := α) s (sh := sh) v name (s.options.gradEnabled && requiresGrad)

/--
Record a non-differentiable `Nat` input in the external environment.

This is used for "index-like" inputs (labels, gather indices, etc.) that should not receive
gradients.
PyTorch comparison: like passing an integer tensor / index to an op; indices are not differentiable.
-/
def inputNat {α : Type} [TorchLean.Storage α]
    (s : TypedGraphSession α) (v : Nat) : IO NatRef := do
  let st0 ← s.state.get
  ensureNoNodes st0
  let id := st0.nat.size
  s.setState { st0 with nat := st0.nat.push v }
  s.makeNatRef id

/-- Read a previously recorded `NatRef`. -/
def getNat {α : Type} [TorchLean.Storage α]
    (s : TypedGraphSession α) (r : NatRef) : IO Nat := do
  s.validateNatRef r
  let st0 ← s.state.get
  if h : r.id < st0.nat.size then
    pure <| st0.nat[r.id]'h
  else
    throw <| IO.userError "torch(TypedGraphSession): invalid nat id"

/-- Overwrite a previously recorded `NatRef`. -/
def setNat {α : Type} [TorchLean.Storage α]
    (s : TypedGraphSession α) (r : NatRef) (v : Nat) : IO Unit := do
  s.validateNatRef r
  let st0 ← s.state.get
  if h : r.id < st0.nat.size then
    let i : Fin st0.nat.size := ⟨r.id, h⟩
    s.setState { st0 with nat := st0.nat.set i v }
  else
    throw <| IO.userError "torch(TypedGraphSession): invalid nat id"

/--
Build a typed index into the current context `Γ ++ ss` from a raw numeric id and expected shape.

This is the main "dynamic check" used by `getValue` (and by a few index-driven nodes): it ensures
that the `Nat` id points to an existing tensor in the session context and that the shape matches.
-/
def mkIdxOrThrow {_α : Type} {Γ ss : List Shape} (id : Nat) (s : Shape) :
    Runtime.Autograd.Result (Proofs.Idx (Γ ++ ss) s) := by
    if h : id < (Γ ++ ss).length then
      let fin : Fin (Γ ++ ss).length := ⟨id, h⟩
      let got : Shape := (Γ ++ ss).get fin
      if hg : got = s then
        exact .ok ⟨fin, hg⟩
      else
        exact .error <|
          s!"torch(TypedGraphSession): shape mismatch at id={id}: expected {Shape.pretty s}, got "
            ++ s!"{Shape.pretty got}"
  else
    exact .error s!"torch(TypedGraphSession): invalid id={id} for ctxLen={(Γ ++ ss).length}"

/--
Evaluate the recorded graph and return the value of a `TensorRef`.

The first read after a change to the session compiles the graph at the recorded leaf values and
nat-environment and keeps the indexed context values. Later reads of the same snapshot use those
values. It does not run backward.
-/
def getValue {α : Type} [TorchLean.Storage α]
    (s : TypedGraphSession α) {sh : Shape}
  (x : TensorRef α sh) : IO (Tensor α sh) := do
  s.validateTensorRef x
  let version ← s.stateVersion.get
  let st0 ← s.state.get
  let values ← match ← s.valueCache.get with
    | some (cached, values) =>
        if cached == version then pure values
        else evaluate st0 version
    | none => evaluate st0 version
  let _ ← okOrThrow (mkIdxOrThrow (_α := α) (Γ := st0.Γ) (ss := st0.ss) x.id sh)
  match values[x.id]? with
  | some value =>
      if h : value.shape = sh then pure (value.cast h)
      else throw <| IO.userError "torch(TypedGraphSession): cached value has the wrong shape"
  | none => throw <| IO.userError "torch(TypedGraphSession): cached context is too short"
where
  /-- Validate and evaluate the recorded graph, retaining only its value context. -/
  evaluate (st0 : TypedGraphSessionState α) (version : Nat) :
      IO (Array (Spec.SomeTensor α)) := do
    let compiled ← okOrThrow <|
      Runtime.Autograd.TypedGraph.compileChecked st0.g st0.x st0.nat
    let values := compiled.context.values
    s.valueCache.set (some (version, values))
    pure values

end TypedGraphSession

end Internal

end Torch
end Autograd
end Runtime
