/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Session.Parameters
public import NN.Runtime.Autograd.Torch.Core.Session.References

/-!
# Eager Tensor Recording

Record inputs and parameter leaves, read tensor values, and stop gradient propagation. CUDA
parameter leaves reuse the persistent mirror instead of uploading the host tensor each time.
-/

public section

namespace Runtime.Autograd.Torch.Internal

open Spec TorchLean
namespace EagerSession

/--
Create a mutable parameter object (not yet on the tape).

Call `use` to record its current value as a leaf.
-/
def param {α : Type} [Storage α] (s : EagerSession α) {sh : Shape}
  (init : Tensor α sh) (name : Option String := none) (requiresGrad : Option Bool := none) :
  IO (Param α sh) :=
  Param.Internal.create init name (requiresGrad.getD s.options.requiresGradByDefault)

/--
Read back the concrete tensor value stored at a `TensorRef`.

Reject foreign or stale handles, missing ids, and mismatched shapes. CUDA values are downloaded.
-/
def getValue {α : Type} [Storage α] [TensorTransfer α]
    (s : EagerSession α) {sh : Shape}
  (x : TensorRef α sh) : IO (Tensor α sh) := do
  s.validateTensorRef x
  if Config.device s.options == .cuda then
    let tape ← s.cudaTape.get
    let stored ← match tape.getValue? x.id with
      | some v => pure v
      | none => throw <| IO.userError "torch: invalid tensor id (missing CUDA value)"
    let hostValue ← CudaBridge.ofAnyBuffer (α := α) stored
    if h : hostValue.shape = sh then
      pure (hostValue.cast h)
    else
      throw <| IO.userError <|
        s!"torch: shape mismatch when reading value (expected {Shape.pretty sh}, got "
          ++ s!"{Shape.pretty hostValue.shape})"
  else
    let tape ← s.tape.get
    let stored ← match tape.getValue? x.id with
      | some v => pure v
      | none => throw <| IO.userError "torch: invalid tensor id (missing value)"
    if h : stored.shape = sh then
      pure (stored.cast h)
    else
      throw <| IO.userError <|
        s!"torch: shape mismatch when reading value (expected {Shape.pretty sh}, got "
          ++ s!"{Shape.pretty stored.shape})"

/--
Record an external input tensor as a leaf on the tape.

The session's `gradEnabled` flag gates `requiresGrad`, so inference sessions keep inputs
non-differentiable even when the caller requests gradients.
-/
def input {α : Type} [Storage α] [TensorTransfer α]
    (s : EagerSession α) {sh : Shape}
  (v : Tensor α sh) (name : Option String := none) (requiresGrad : Bool := false) :
  IO (TensorRef α sh) := do
  let requiresGrad := s.options.gradEnabled && requiresGrad
  if Config.device s.options == .cuda then
    let buffer ← CudaBridge.toAnyBuffer (α := α) (s := sh) v
    let tape ← s.cudaTape.get
    let (nextTape, id) := Runtime.Autograd.Cuda.Tape.leaf (t := tape) (value := buffer)
      (name := name) (requiresGrad := requiresGrad)
    s.cudaTape.set nextTape
    s.makeTensorRef id
  else
    let tape ← s.tape.get
    let (nextTape, id) := Runtime.Autograd.Tape.leaf (t := tape) (s := sh) (value := v)
      (name := name) (requiresGrad := requiresGrad)
    s.tape.set nextTape
    s.makeTensorRef id

/-- Record a constant leaf that never receives gradients. -/
def const {α : Type} [Storage α] [TensorTransfer α]
    (s : EagerSession α) {sh : Shape}
    (v : Tensor α sh) (name : Option String := none) : IO (TensorRef α sh) :=
  s.input v name (requiresGrad := false)

/--
Keep the primal value and record a reference that receives no gradient.

CPU carriers with scalar differentiation metadata clear it here as well. Ordinary numeric
tensors and CUDA buffers keep their existing storage; the detached reference borrows that value.
-/
def detach {α : Type} [Storage α] [Context α] [TensorTransfer α]
    (s : EagerSession α) {sh : Shape}
    (x : TensorRef α sh) (name : Option String := none) : IO (TensorRef α sh) := do
  s.validateTensorRef x
  if Config.device s.options == .cuda then
    let tape ← s.cudaTape.get
    let stored ← match tape.getValue? x.id with
      | some v => pure v
      | none => throw <| IO.userError "torch: detach: invalid tensor id (missing CUDA value)"
    if _h : stored.s = sh then
      let stored' : Runtime.Autograd.Cuda.AnyBuffer := { s := sh, buf := stored.buf }
      let node : Runtime.Autograd.Cuda.Node :=
        { name := name
          value := stored'
          ownsValue := false
          requiresGrad := false
          parents := #[x.id]
          backward := fun _ => .ok #[] }
      let (nextTape, id) := Runtime.Autograd.Cuda.Tape.addNode tape node
      s.cudaTape.set nextTape
      s.makeTensorRef id
    else
      throw <| IO.userError <|
        s!"torch: detach: shape mismatch (expected {Shape.pretty sh}, got {Shape.pretty stored.s})"
  else
    let value ← getValue (α := α) s (sh := sh) x
    let tape ← s.tape.get
    let node : Runtime.Autograd.Node α :=
      { name := name
        value := Spec.SomeTensor.ofTensor (Tensor.detachSpec value)
        requiresGrad := false
        parents := #[x.id]
        backward := fun _ => .ok #[] }
    let (nextTape, id) := Runtime.Autograd.Tape.addNode tape node
    s.tape.set nextTape
    s.makeTensorRef id

/--
Use a parameter in the tape by recording its current value as a leaf.

Register the leaf in `paramsByLeaf` for optimizer updates and CUDA cleanup. Keep that registration
when gradients are disabled: cleanup still needs to know that the parameter owns its mirror.
-/
def use {α : Type} [Storage α] [TensorTransfer α]
    (s : EagerSession α) {sh : Shape}
  (p : Param α sh) : IO (TensorRef α sh) := do
  let requiresGrad := s.options.gradEnabled && p.requiresGrad
  let id ←
    if Config.device s.options == .cuda then
      let buffer ←
        match ← p.cudaValue.get with
        | some stored =>
            if _h : stored.s = sh then
              pure ({ s := sh, buf := stored.buf } : Runtime.Autograd.Cuda.AnyBuffer)
            else
              -- A well-formed `Param` has a cached CUDA buffer with the declared shape; if the
              -- cache is inconsistent, re-upload from the host value.
              let v ← p.value.get
              let uploaded ← CudaBridge.toAnyBuffer (α := α) (s := sh) v
              AnyParam.releaseCachedCudaValue p
              p.cudaValue.set (some uploaded)
              p.hostCurrent.set true
              pure uploaded
        | none =>
            let v ← p.value.get
            let uploaded ← CudaBridge.toAnyBuffer (α := α) (s := sh) v
            AnyParam.releaseCachedCudaValue p
            p.cudaValue.set (some uploaded)
            p.hostCurrent.set true
            pure uploaded
      let tape ← s.cudaTape.get
      let (nextTape, id) :=
        Runtime.Autograd.Cuda.Tape.leaf (t := tape) (value := buffer) (name := p.name)
          (requiresGrad := requiresGrad)
      s.cudaTape.set nextTape
      pure id
    else
      syncParamCudaToHost (α := α) (sh := sh) p
      let v ← p.value.get
      let tape ← s.tape.get
      let (nextTape, id) :=
        Runtime.Autograd.Tape.leaf (t := tape) (s := sh)
          (value := v) (name := p.name) (requiresGrad := requiresGrad)
      s.tape.set nextTape
      pure id
  s.paramsByLeaf.modify (fun m => m.insert id (AnyParam.ofParam p))
  s.makeTensorRef id

end EagerSession

end Runtime.Autograd.Torch.Internal
