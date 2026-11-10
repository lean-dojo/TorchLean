/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.TensorTransfer.CUDA
public import TorchLean.NN.Session.Config

/-!
# Parameter Mirrors

CUDA updates leave the host tensor stale, and it is downloaded only where host code needs the value,
such as a parameter readback or a `use` on a CPU session. Writing a host value drops the cached
mirror, and the CUDA snapshots recorded from it stay valid.
-/

public section

namespace TorchLean

/--
Synchronize a CUDA-updated parameter back to its host tensor when the host value is stale.

Training keeps parameters on the device. Public readback and a CPU session's `use` call this before
Lean code sees the host tensor. A mirror of the wrong shape is an error.
-/
def Param.syncToHost {α : Type} [Storage α] [TensorTransfer α]
    {sh : Shape}
    (p : Param α sh) : IO Unit := do
  let current ← p.hostSynced.get
  if current then
    pure ()
  else
    match ← p.cudaValue.get with
    | none =>
        p.hostSynced.set true
    | some any =>
        let hostValue ← TensorTransfer.ofSomeBuffer (α := α) any
        if h : hostValue.shape = sh then
          p.value.set (hostValue.cast h)
          p.hostSynced.set true
        else
          throw <| IO.userError <|
            s!"torch: CUDA param sync shape mismatch (expected {sh}, got "
              ++ s!"{hostValue.shape})"

/-- Store the CUDA mirror of a parameter and mark the host tensor stale. A buffer of another shape
is an error.

Replacing the mirror drops only its previous reference, so recorded snapshots stay valid. Another
scope must not force-release the incoming buffer afterwards, so an owned tape intermediate needs a
copy, or that scope must hand its release responsibility over.
-/
def Param.setCUDAValue {α : Type} [Storage α] {sh : Shape} (p : Param α sh)
    (any : LibTorch.SomeBuffer) : IO Unit := do
  if _h : any.s = sh then
    p.cudaValue.set (some { s := sh, buf := any.buf })
    p.hostSynced.set false
  else
    throw <| IO.userError <|
      s!"torch: CUDA param cache shape mismatch (expected {sh}, got "
        ++ s!"{any.s})"

/-- Overwrite a host parameter value and drop the cached CUDA mirror that the write makes stale. -/
def Param.setHostValue {α : Type} [Storage α] {sh : Shape}
    (p : Param α sh) (v : Tensor α sh) : IO Unit := do
  SomeParam.releaseCachedCUDAValue p
  p.value.set v
  p.hostSynced.set true

/-- Read the current CUDA mirror. An empty cache is filled by uploading the host value, which must
be current.

The returned buffer is shared with the parameter. A caller may keep it as a recorded snapshot,
because replacing the cache later drops only the parameter's reference. A cached mirror of the
wrong shape is an error.
-/
def Param.getCUDAValue {α : Type} [Storage α] [TensorTransfer α] {sh : Shape}
    (p : Param α sh) : IO LibTorch.SomeBuffer := do
  match ← p.cudaValue.get with
  | some stored =>
      unless stored.s == sh do
        throw <| IO.userError "torch: current CUDA parameter mirror has the wrong shape"
      match LibTorch.SomeBuffer.validate stored with
      | .ok value => pure value
      | .error message => throw <| IO.userError message
  | none =>
      unless ← p.hostSynced.get do
        throw <| IO.userError
          "torch: CUDA parameter has neither a current mirror nor a current host value"
      let uploaded ← TensorTransfer.toSomeBuffer (α := α) (s := sh) (← p.value.get)
      p.cudaValue.set (some uploaded)
      pure uploaded

end TorchLean
