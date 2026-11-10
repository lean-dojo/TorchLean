/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Session
public import TorchLean.NN.TensorTransfer

/-!
# Eager Session Lifecycle

Create and reset sessions, and release the intermediates their CUDA tapes own. Parameter snapshots
use Lean reference counting so a reset or update in one session cannot invalidate another session.
-/

public section

namespace TorchLean.Session.Config

/-- Move a GPU request to the CPU, with a diagnostic, when the scalar has no native tape
representation. `create` calls this before it chooses the session's tape. A native carrier keeps the
requested device, so its runtime failures are reported and never turned into a CPU fallback. A
backend profile in the options is validated later against the resulting device. -/
def forScalar {α : Type} [TorchLean.Storage α] [TensorTransfer α]
    (options : Config) : IO Config := do
  if options.device == Backend.Device.cuda &&
      !TensorTransfer.supportsGPU (α := α) then
    IO.eprintln "autograd: this scalar type uses the CPU tape; keeping its type and precision"
    return { options with device := Backend.Device.cpu }
  return options

end TorchLean.Session.Config

namespace TorchLean

namespace Session

/-- Allocate an eager session once `Config.validateForExecution` accepts the options that
`forScalar` returns. The resulting device chooses a CPU or CUDA tape, and the session gets a fresh
owner id. -/
def create {α : Type} [Storage α] [TensorTransfer α]
    (options : Config := {}) (gradEnabled : Bool := true) : IO (Session α) := do
  let options ← options.forScalar (α := α)
  try
    options.validateForExecution
  catch e =>
    throw <| IO.userError s!"torch eager session: {e.toString}"
  let tape ← IO.mkRef (DeviceTape.empty (α := α) (Config.device options))
  let paramsByLeaf ← IO.mkRef (Std.HashMap.emptyWithCapacity)
  let parameterStorageByLeaf ← IO.mkRef (Std.HashMap.emptyWithCapacity)
  let rngCounter ← IO.mkRef 0
  let selectedBackends ← IO.mkRef (#[] : Array Backend.AcceptedKernel)
  let refOwner ← RefIdentity.freshOwner
  let refGeneration ← IO.mkRef 0
  if options.showBackend then
    IO.println "[TorchLean] backend capsules used:"
  pure
    { options := options
      gradEnabled := gradEnabled
      tape := tape
      paramsByLeaf := paramsByLeaf
      parameterStorageByLeaf := parameterStorageByLeaf
      rngCounter := rngCounter
      selectedBackends := selectedBackends
      refOwner := refOwner
      refGeneration := refGeneration }

/-- Release a CUDA allocation. Releasing it again, or finalizing it later, is safe. -/
def releaseCUDABuffer (b : LibTorch.Buffer) : IO Unit :=
  discard <| LibTorch.Buffer.releaseIO b

/-- Force-release a shape-erased CUDA buffer. -/
def releaseCUDASomeBuffer (b : LibTorch.SomeBuffer) : IO Unit :=
  releaseCUDABuffer b.buf

/-- Release the values the tape owns and each node's cleanup buffers. Parameter leaves are left to
reference counting, since other sessions may share their snapshots. -/
private def releaseCUDATapeValues {α : Type} [Storage α]
    (session : Session α) : IO Unit := do
  let tape ← session.cudaTape
  let parameters ← session.paramsByLeaf.get
  for id in [0:tape.nodes.size] do
    match tape.nodes[id]? with
    | none => pure ()
    | some node =>
        if node.ownsValue && !parameters.contains id then
          releaseCUDASomeBuffer node.value
        for buffer in node.cleanup do
          releaseCUDABuffer buffer.buf

/--
Run an action that borrows a sparse CUDA gradient map, then release every buffer in the map.

The action must not retain a gradient buffer after it returns. Cleanup also runs when the action
throws, which keeps optimizer failures from leaking device memory.
-/
def withSparseGrads {β : Type} (xs : LibTorch.Tape.SparseGradMap)
    (action : LibTorch.Tape.SparseGradMap → IO β) : IO β := do
  try
    action xs
  finally
    LibTorch.Tape.releaseSparseGrads xs

/--
Discard the current forward pass and invalidate its handles. The parameter registrations go with
the tape, while the parameters themselves and the random counter are kept.

Owned CUDA intermediates are released before the tape drops its references. A parameter leaf can
hold the current mirror or a value recorded at an earlier `use`, so its snapshot is left to Lean
reference counting and stays valid in any other session that refers to it. Resetting the tape
neither empties LibTorch's allocator cache nor changes optimizer history.
-/
def resetTape {α : Type} [Storage α] (s : Session α) : IO Unit := do
  if s.options.usesCUDA then
    releaseCUDATapeValues s
  s.tape.set (DeviceTape.empty (Config.device s.options))
  s.paramsByLeaf.set (Std.HashMap.emptyWithCapacity)
  s.parameterStorageByLeaf.set (Std.HashMap.emptyWithCapacity)
  s.refGeneration.modify (fun generation => generation + 1)

end Session

end TorchLean
