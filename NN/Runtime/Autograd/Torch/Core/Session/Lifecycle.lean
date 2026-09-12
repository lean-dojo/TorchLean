/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Session.State

/-!
# Eager Session Lifecycle

Create and reset sessions, and release the buffers their CUDA tapes own. Parameter mirrors survive
a normal reset; after an optimizer update, only frozen parameters still own their old leaf values.
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

/-- Release the tape's values, preserving parameter mirrors that are still current. -/
private def releaseCudaTapeValues {α : Type} [Storage α] (session : EagerSession α)
    (parametersUpdated : Bool) : IO Unit := do
  let tape ← session.cudaTape.get
  let parameters ← session.paramsByLeaf.get
  for id in [0:tape.nodes.size] do
    match tape.nodes[id]? with
    | none => pure ()
    | some node =>
        let releaseValue := match parameters.get? id with
          | none => true
          | some parameter => parametersUpdated && parameter.requiresGrad
        if node.ownsValue && releaseValue then
          releaseCudaAnyBuffer node.value
        for buffer in node.cleanup do
          releaseCudaBuffer buffer

/--
Release the current CUDA tape's intermediates and backward workspace.

Parameter mirrors stay alive: their `Param` objects own them across forward passes.
-/
def releaseCudaTapeNonParamValues {α : Type} [Storage α] (s : EagerSession α) : IO Unit :=
  releaseCudaTapeValues s false

/--
Release CUDA tape values after an optimizer step.

Trainable parameters now have fresh mirrors, so their old leaf values can go too. Frozen parameters
still use the same mirrors; keep those alive.
-/
def releaseCudaTapeAfterOptimizerStep {α : Type} [Storage α]
    (s : EagerSession α) : IO Unit :=
  releaseCudaTapeValues s true

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

/-- Ask the native allocator to return/free pages after a large CUDA eager step. -/
def collectCudaAllocator : IO Unit := do
  let released := Runtime.Autograd.Cuda.Buffer.collectAllocator true
  AnyParam.observeCudaCleanupFlag released

/-- Discard the current forward pass and invalidate its handles. Keep parameters and RNG state. -/
def resetTape {α : Type} [Storage α] (s : EagerSession α) : IO Unit := do
  if Config.device s.options == .cuda then
    releaseCudaTapeNonParamValues s
  s.tape.set Runtime.Autograd.Tape.empty
  s.cudaTape.set Runtime.Autograd.Cuda.Tape.empty
  s.paramsByLeaf.set (Std.HashMap.emptyWithCapacity)
  s.nats.set #[]
  s.refGeneration.modify (fun generation => generation + 1)

end EagerSession

end Runtime.Autograd.Torch.Internal
