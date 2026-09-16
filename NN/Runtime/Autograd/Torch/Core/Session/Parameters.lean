/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.CudaBridge
public import NN.Runtime.Autograd.Torch.Core.Types

/-!
# Parameter Mirrors

CUDA updates leave the host tensor stale. We only download it at an explicit readback boundary;
writing a host value removes the cached mirror without invalidating recorded CUDA snapshots.
-/

public section

namespace Runtime.Autograd.Torch.Internal

open Spec TorchLean
/--
Synchronize a CUDA-updated parameter back to its host tensor, if needed.

This synchronization point is explicit. Training hot paths keep parameters resident on device;
public readback APIs call this helper before exposing parameter tensors to the Lean side.
-/
def syncParamCudaToHost {α : Type} [Storage α] [TensorTransfer α]
    {sh : Shape}
    (p : Param α sh) : IO Unit := do
  let current ← p.hostCurrent.get
  if current then
    pure ()
  else
    match ← p.cudaValue.get with
    | none =>
        p.hostCurrent.set true
    | some any =>
        let hostValue ← CudaBridge.ofAnyBuffer (α := α) any
        if h : hostValue.shape = sh then
          p.value.set (hostValue.cast h)
          p.hostCurrent.set true
        else
          throw <| IO.userError <|
            s!"torch: CUDA param sync shape mismatch (expected {Shape.pretty sh}, got "
              ++ s!"{Shape.pretty hostValue.shape})"

/-- Store/update the CUDA mirror of a parameter and mark the host tensor stale.

Replacing the mirror drops its previous reference without invalidating recorded snapshots.
The incoming buffer must not remain subject to another scope's explicit release; installing an
owned tape intermediate requires a copy or transfer of that scope's release responsibility.
-/
def setParamCudaValue {α : Type} [Storage α] {sh : Shape} (p : Param α sh)
    (any : Runtime.Autograd.Cuda.AnyBuffer) : IO Unit := do
  if _h : any.s = sh then
    p.cudaValue.set (some { s := sh, buf := any.buf })
    p.hostCurrent.set false
  else
    throw <| IO.userError <|
      s!"torch: CUDA param cache shape mismatch (expected {Shape.pretty sh}, got "
        ++ s!"{Shape.pretty any.s})"

/-- Overwrite a host parameter value and invalidate any stale CUDA mirror. -/
def setParamHostValue {α : Type} [Storage α] {sh : Shape}
    (p : Param α sh) (v : Tensor α sh) : IO Unit := do
  AnyParam.releaseCachedCudaValue p
  p.value.set v
  p.hostCurrent.set true

/-- Read the current CUDA mirror, uploading a current host value when the cache is empty.

The returned buffer is shared with the parameter. A caller may retain it as a recorded snapshot;
replacing the cache later drops only the parameter's reference. A cache with the wrong shape, or
an absent cache paired with a stale host value, has no justified current value to use.
-/
def getParamCudaValue {α : Type} [Storage α] [TensorTransfer α] {sh : Shape}
    (p : Param α sh) : IO Runtime.Autograd.Cuda.AnyBuffer := do
  match ← p.cudaValue.get with
  | some stored =>
      unless stored.s == sh do
        throw <| IO.userError "torch: current CUDA parameter mirror has the wrong shape"
      match Runtime.Autograd.Cuda.AnyBuffer.validate stored with
      | .ok value => pure value
      | .error message => throw <| IO.userError message
  | none =>
      unless ← p.hostCurrent.get do
        throw <| IO.userError
          "torch: CUDA parameter has neither a current mirror nor a current host value"
      let uploaded ← CudaBridge.toAnyBuffer (α := α) (s := sh) (← p.value.get)
      p.cudaValue.set (some uploaded)
      pure uploaded

end Runtime.Autograd.Torch.Internal
