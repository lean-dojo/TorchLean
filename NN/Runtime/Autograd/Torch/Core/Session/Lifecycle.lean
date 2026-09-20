/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Session.State

/-!
# Eager Session Lifecycle

Create and reset sessions, and release the intermediates their CUDA tapes own. Parameter snapshots
use Lean reference counting so a reset or update in one session cannot invalidate another session.
-/

public section

namespace Runtime.Autograd.Torch.Internal

open Spec TorchLean
namespace EagerSession

/-- Allocate a fresh eager session with an empty tape and empty side tables. -/
def new {α : Type} [Storage α] (options : Config := {}) : IO (EagerSession α) := do
  try
    options.validateForExecution
  catch e =>
    throw <| IO.userError s!"torch eager session: {e.toString}"
  let tape ← IO.mkRef Runtime.Autograd.Tape.empty
  let cudaTape ← IO.mkRef Runtime.Autograd.Cuda.Tape.empty
  let paramsByLeaf ← IO.mkRef (Std.HashMap.emptyWithCapacity)
  let parameterStorageByLeaf ← IO.mkRef (Std.HashMap.emptyWithCapacity)
  let nats ← IO.mkRef #[]
  let rngCounter ← IO.mkRef 0
  let selectedBackends ← IO.mkRef (#[] : Array NN.Backend.AcceptedKernel)
  let refOwner ← RefIdentity.freshOwner
  let refGeneration ← IO.mkRef 0
  if options.showBackend then
    IO.println "[TorchLean] backend capsules used:"
  pure
    { options := options
      tape := tape
      cudaTape := cudaTape
      paramsByLeaf := paramsByLeaf
      parameterStorageByLeaf := parameterStorageByLeaf
      nats := nats
      rngCounter := rngCounter
      selectedBackends := selectedBackends
      refOwner := refOwner
      refGeneration := refGeneration }

/-- Force-free a CUDA buffer allocation; the external finalizer is safe to call twice. -/
def releaseCudaBuffer (b : Runtime.Autograd.Cuda.Buffer) : IO Unit := do
  let released ← Runtime.Autograd.Cuda.Buffer.releaseIO b
  AnyParam.observeCudaCleanupFlag released

/-- Force-release a shape-erased CUDA buffer. -/
def releaseCudaAnyBuffer (b : Runtime.Autograd.Cuda.AnyBuffer) : IO Unit :=
  releaseCudaBuffer b.buf

/-- Device-resident gradients keyed by parameter leaf ids, with each leaf's shape. -/
abbrev CudaGradMap := Std.HashMap Nat Runtime.Autograd.Cuda.AnyBuffer

/-- Release owned intermediates while leaving shared parameter snapshots to reference counting. -/
private def releaseCudaTapeValues {α : Type} [Storage α]
    (session : EagerSession α) : IO Unit := do
  let tape ← session.cudaTape.get
  let parameters ← session.paramsByLeaf.get
  for id in [0:tape.nodes.size] do
    match tape.nodes[id]? with
    | none => pure ()
    | some node =>
        if node.ownsValue && !parameters.contains id then
          releaseCudaAnyBuffer node.value
        for buffer in node.cleanup do
          releaseCudaBuffer buffer

/-- Release a sparse CUDA gradient map after an optimizer has consumed it. -/
def releaseCudaGradMap (xs : CudaGradMap) : IO Unit := do
  for (_id, x) in xs.toList do
    releaseCudaAnyBuffer x

/--
Run an action that borrows a sparse CUDA gradient map, then release every buffer in the map.

The action must not retain a gradient buffer after it returns. Cleanup also runs when the action
throws, which keeps optimizer failures from leaking device memory.
-/
def withCudaGradMap {β : Type} (xs : CudaGradMap) (action : CudaGradMap → IO β) : IO β := do
  try
    action xs
  finally
    releaseCudaGradMap xs

/-- Check that a shape-erased CUDA buffer has the number of elements promised by its shape. -/
def checkCudaAnyBufferSize (where_ : String)
    (x : Runtime.Autograd.Cuda.AnyBuffer) : IO Unit := do
  let expected := Spec.Shape.size x.s
  if _hExpected : expected < UInt32.size then
    let got := Runtime.Autograd.Cuda.Buffer.size x.buf
    let expectedU32 : UInt32 := UInt32.ofNat expected
    if got != expectedU32 then
      throw <| IO.userError
        s!"torch: CUDA buffer size mismatch in {where_} \
           (shape={Shape.pretty x.s}, expected={expected}, got={got.toNat})"
  else
    throw <| IO.userError s!"torch: CUDA tensor too large in {where_}"

/--
Discard the current forward pass and invalidate its handles. Keep parameters and RNG state.

Owned CUDA intermediates are released before the tape drops its references. Parameter leaves can
refer to the current mirror or an older recorded value, so their snapshots follow Lean reference
counting and remain valid in any other session that still uses them. Resetting the tape neither
drains the reuse cache nor changes optimizer history.
-/
def resetTape {α : Type} [Storage α] (s : EagerSession α) : IO Unit := do
  if Config.device s.options == .cuda then
    releaseCudaTapeValues s
  s.tape.set Runtime.Autograd.Tape.empty
  s.cudaTape.set Runtime.Autograd.Cuda.Tape.empty
  s.paramsByLeaf.set (Std.HashMap.emptyWithCapacity)
  s.parameterStorageByLeaf.set (Std.HashMap.emptyWithCapacity)
  s.nats.set #[]
  s.refGeneration.modify (fun generation => generation + 1)

end EagerSession

end Runtime.Autograd.Torch.Internal
