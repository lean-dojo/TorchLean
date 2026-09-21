/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.StateIO
public import NN.API.Neural.Builders
public import NN.API.Sample -- shake: keep

/-!
# Runtime Checkpoints

TorchLean examples often want the same simple workflow:

1. train a model for a few steps,
2. save its state, and
3. restore that state before inference or further training.

The implementation delegates binary encoding and device transfers to
`Runtime.Autograd.Model.StateIO`, while this module provides the checked runtime API used by
trainers and data loaders.

## What Is Supported

Both checkpoint formats preserve the model's shape-indexed state layout:

- Native `Float32` modules use exact little-endian binary32 payloads on CPU and CUDA.
- Binary64 `Float` modules use exact binary64 JSON on CPU and binary32 payloads on CUDA.
- Streamed checkpoints write one tensor at a time instead of materializing a second host copy of a
  large device-resident model.

This is enough to checkpoint any TorchLean runtime model implemented as a
`TorchLean.Module.Objective` over native `Float32` or binary64 `Float`, independent of
architecture.

Results of `Trainer.train` use `Checkpoint.State`: `trained.save path` writes the trained state as
`Float` tensors and `trainer.load path data` restores it.

Explicit immutable `nn.State α` checkpoints use `Checkpoint.Encoding α`. Instances preserve exact
bits for `Float`, `Float32`, and configured binary formats, including arbitrary supported widths.
The complex instance stores both coordinates using their component encoding. Loading checks the
format tag as well as the state layout; it never silently casts between scalar formats.
-/

@[expose] public section

namespace TorchLean
namespace Checkpoint

open Spec TorchLean

/-- Checkpoint persistence supported by a tensor element type. -/
class Checkpointable (α : Type) [TorchLean.Storage α] where
  /-- Write every parameter and persistent buffer in order. -/
  write : {shapes : List Shape} →
    Bool → System.FilePath → Runtime.Autograd.Torch.ParamList α shapes → IO Unit
  /-- Validate and restore every parameter and persistent buffer in order. -/
  read : {shapes : List Shape} →
    Bool → System.FilePath → Runtime.Autograd.Torch.ParamList α shapes → IO Unit

/-- Binary64 `Float` checkpoints keep exact binary64 values on CPU and binary32 values on CUDA. -/
instance : Checkpointable Float where
  write := fun {shapes} useCuda path state => do
    if useCuda then
      Runtime.Autograd.Model.StateIO.writeModuleStateFloat32
        Float.toFloat32 (shapes := shapes) path state
    else
      let values ← Runtime.Autograd.Torch.ParamList.valuesSynced (α := Float)
        (ss := shapes) state
      Runtime.Autograd.Model.StateIO.writeStateBits (ss := shapes) path values
  read := fun {shapes} useCuda path state => do
    if ← Runtime.Autograd.Model.StateIO.isModuleStateFloat32 path then
      Runtime.Autograd.Model.StateIO.readModuleStateFloat32Into
        Float32.toFloat path useCuda (shapes := shapes) state
    else
      let result ← Runtime.Autograd.Model.StateIO.readStateBits (ss := shapes) path
      match result with
      | .error message => throw <| IO.userError s!"Checkpoint: load failed for {path}: {message}"
      | .ok values =>
          Runtime.Autograd.Torch.ParamList.setValues (α := Float) (ss := shapes) state values

/-- Native `Float32` checkpoints preserve exact binary32 payloads on both CPU and CUDA. -/
instance : Checkpointable Float32 where
  write := fun {shapes} _ path state =>
    Runtime.Autograd.Model.StateIO.writeModuleStateFloat32
      id (shapes := shapes) path state
  read := fun {shapes} useCuda path state => do
    unless ← Runtime.Autograd.Model.StateIO.isModuleStateFloat32 path do
      throw <| IO.userError
        s!"Checkpoint: {path} is not a native Float32 module checkpoint"
    Runtime.Autograd.Model.StateIO.readModuleStateFloat32Into
      id path useCuda (shapes := shapes) state

/--
Save the current values of a TorchLean runtime module.

The module's element type determines its persistence implementation. Native
`Float32` modules preserve exact binary32 payloads on CPU and CUDA. Binary64
`Float` modules preserve binary64 values on CPU and the runtime's binary32
values on CUDA.
-/
def save {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α] [Checkpointable α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (objective : TorchLean.Module.Objective α β stateShapes inputShapes
      dataInputShapes)
    (path : System.FilePath) : IO Unit := do
  let runtimeObjective := TorchLean.Module.Objective.Internal.runtime objective
  Checkpointable.write runtimeObjective.runtime.usesCuda path runtimeObjective.trainer.state

/--
Load a compatible checkpoint into a module.

The reader checks the format header, state-tensor count, every shape, and every payload length
before accepting the checkpoint.
-/
def load {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α] [Checkpointable α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (objective : TorchLean.Module.Objective α β stateShapes inputShapes
      dataInputShapes)
    (path : System.FilePath) : IO Unit := do
  let runtimeObjective := TorchLean.Module.Objective.Internal.runtime objective
  Checkpointable.read runtimeObjective.runtime.usesCuda path runtimeObjective.trainer.state

namespace Optimizer

/--
Save optimizer state retained by a module's runtime backend.

This currently applies to eager CUDA Adam and AdamW, whose moment buffers live
on the device. The checkpoint also preserves the eager session's random counter, so dropout
continues with the next mask after a restore. The operation fails explicitly when the trainer has no
backend-owned optimizer state instead of writing an incomplete resume
checkpoint.
-/
def save
    {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β] [Context α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (objective : TorchLean.Module.Objective α β stateShapes inputShapes
      dataInputShapes)
    (path : System.FilePath) : IO Unit := do
  let runtimeObjective := TorchLean.Module.Objective.Internal.runtime objective
  match runtimeObjective.trainer.optimizerStateCheckpoint? with
  | some checkpoint => checkpoint.save path
  | none =>
      throw <| IO.userError
        "Checkpoint: selected trainer does not expose backend-owned optimizer state"

/-- Restore optimizer state and its random counter. Legacy files lacking the counter load with
a warning: their moments are recoverable, but they cannot reproduce a stochastic continuation. -/
def load
    {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β] [Context α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (objective : TorchLean.Module.Objective α β stateShapes inputShapes
      dataInputShapes)
    (path : System.FilePath) : IO Unit := do
  let runtimeObjective := TorchLean.Module.Objective.Internal.runtime objective
  match runtimeObjective.trainer.optimizerStateCheckpoint? with
  | some checkpoint => checkpoint.load path
  | none =>
      throw <| IO.userError
        "Checkpoint: selected trainer does not expose backend-owned optimizer state"

end Optimizer

namespace State

/-- Save an immutable state pack in its scalar encoding, with the layout supplied by `model`. -/
def save {σ τ : Shape} {α : Type} [Storage α] [Encoding α]
    (model : nn.Sequential σ τ)
    (state : nn.State α (nn.stateShapes model))
    (path : System.FilePath) : IO Unit :=
  Runtime.Autograd.Model.StateIO.writeStateBits
    (ss := nn.stateShapes model) path
    (nn.State.Internal.toTensorPack state)

/--
Load immutable model state without mutating a runtime module.

The model supplies the exact dependent state layout, so malformed, missing,
extra, or incorrectly shaped tensors are rejected at the boundary.
-/
def load {σ τ : Shape} {α : Type} [Storage α] [Encoding α]
    (model : nn.Sequential σ τ)
    (path : System.FilePath) :
    IO (nn.State α (nn.stateShapes model)) := do
  let stateResult ← Runtime.Autograd.Model.StateIO.readStateBits
    (ss := nn.stateShapes model) path
  match stateResult with
  | Except.error message =>
      throw <| IO.userError s!"Checkpoint: load failed for {path}: {message}"
  | Except.ok state =>
      pure (nn.State.Internal.fromTensorPack state)

end State

end Checkpoint
end TorchLean
