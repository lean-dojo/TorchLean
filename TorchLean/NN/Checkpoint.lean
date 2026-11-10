/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Checkpoint.State
public import TorchLean.NN.Builder.Layer

/-!
# Runtime Checkpoints

The usual workflow in TorchLean examples: train a model for a few steps, save its state, and
restore that state before inference or further training.

`Checkpoint.State.save` and `Checkpoint.State.load` store a shape-indexed `TensorPack α` in the
binary container of `TorchLean.Checkpoint`, which checks the expected layout on loading.
Trainers, sessions and runtime modules all go through this checked API.

## Scalar encodings

`Checkpoint.Encoding α` fixes the format tag and the stored word of each scalar. The instances keep
the exact bits of `Float`, `Float32` and configured binary formats of every supported width, and
the complex instance stores both coordinates in their component encoding. Loading checks the
format tag as well as the state layout, so a checkpoint is never cast silently to another scalar
format.

## Trainers, sessions and modules

`trained.save path` on a result of `Trainer.train` keeps the trained state's scalar encoding, and
`Session.load` restores a checkpoint into a compatible training session. A runtime module on CPU
or CUDA saves with `Checkpoint.State.save (← module.state) path` and restores with
`module.setState (← Checkpoint.State.load shapes path)`. Both directions copy the whole state
through host memory. `Checkpoint.Optimizer` saves backend-owned optimizer state separately.
-/

@[expose] public section

namespace TorchLean
namespace Checkpoint

namespace Optimizer

/--
Save the optimizer state that the module's runtime backend keeps on its side. Only eager CUDA
Adam and AdamW have such state: their moment buffers live on the device. The file also records
the eager session's random counter, so dropout continues with the next mask after a restore.

When the trainer has no backend-owned optimizer state, the call fails instead of writing a resume
checkpoint without the moments.
-/
def save
    {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β] [Context α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (objective : TorchLean.Module.Objective α β stateShapes inputShapes
      dataInputShapes)
    (path : System.FilePath) : IO Unit := do
  let runtimeObjective := Module.Objective.runtime objective
  match runtimeObjective.trainer.optimizerStateCheckpoint? with
  | some checkpoint => checkpoint.save path
  | none =>
      throw <| IO.userError
        "Checkpoint: selected trainer does not expose backend-owned optimizer state"

/--
Restore the optimizer state and random counter written by `save`. A file without the counter loads
its moments with a warning, and a stochastic run cannot resume exactly from it.
-/
def load
    {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β] [Context α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (objective : TorchLean.Module.Objective α β stateShapes inputShapes
      dataInputShapes)
    (path : System.FilePath) : IO Unit := do
  let runtimeObjective := Module.Objective.runtime objective
  match runtimeObjective.trainer.optimizerStateCheckpoint? with
  | some checkpoint => checkpoint.load path
  | none =>
      throw <| IO.userError
        "Checkpoint: selected trainer does not expose backend-owned optimizer state"

end Optimizer

namespace State

/--
Write `state` to `path` atomically, tagged with the scalar format of `α`, each scalar stored as its
`Encoding α` word (PyTorch: `torch.save(model.state_dict(), path)`).
-/
def save {α : Type} [Storage α] [Encoding α] {shapes : List Shape}
    (state : TensorPack α shapes) (path : System.FilePath) : IO Unit :=
  writeStateBits path state

/--
Read a state of layout `shapes` from `path` without touching any runtime module (PyTorch:
`torch.load(path)`). `shapes` is usually `nn.Sequential.stateShapes model`. A different scalar
format, a missing, extra or misshapen tensor, a malformed word and a truncated or overlong file are
all rejected.
-/
def load {α : Type} [Storage α] [Encoding α] (shapes : List Shape)
    (path : System.FilePath) : IO (TensorPack α shapes) := do
  try
    let state ← readStateBits (ss := shapes) path
    pure state
  catch error =>
    throw <| IO.userError s!"Checkpoint: load failed for {path}: {error}"

end State

end Checkpoint
end TorchLean
