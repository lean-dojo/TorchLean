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
writing a host value releases the old device mirror.
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

/-- Store/update the CUDA mirror of a parameter and mark the host tensor stale. -/
def setParamCudaValue {α : Type} [Storage α] {sh : Shape} (p : Param α sh)
    (any : Runtime.Autograd.Cuda.AnyBuffer) : IO Unit := do
  if _h : any.s = sh then
    AnyParam.releaseCachedCudaValue p
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
  p.cudaValue.set none
  p.hostCurrent.set true

end Runtime.Autograd.Torch.Internal
