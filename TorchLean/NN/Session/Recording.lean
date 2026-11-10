/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Session.Parameters
public import TorchLean.NN.Session.References

/-!
# Eager Tensor Recording

Record input, constant and parameter leaves, and read tensor values back. A CUDA parameter leaf
reuses the parameter's persistent mirror, so the host tensor is uploaded only when the cache is
empty.
-/

public section

namespace TorchLean

namespace Session

/--
Read back the concrete tensor value stored at a `TensorRef`.

Foreign or stale handles, missing ids and mismatched shapes are rejected. On a CUDA session the
value is downloaded.
-/
def getValue {α : Type} [Storage α] [TensorTransfer α]
    (s : Session α) {sh : Shape}
  (x : TensorRef α sh) : IO (Tensor α sh) := do
  s.validateTensorRef x
  if s.options.usesCUDA then
    let tape ← s.cudaTape
    let stored ← match tape.getValue? x.id with
      | some v => pure v
      | none => throw <| IO.userError "torch: invalid tensor id (missing CUDA value)"
    let hostValue ← TensorTransfer.ofSomeBuffer (α := α) stored
    if h : hostValue.shape = sh then
      pure (hostValue.cast h)
    else
      throw <| IO.userError <|
        s!"torch: shape mismatch when reading value (expected {sh}, got "
          ++ s!"{hostValue.shape})"
  else
    let tape ← s.cpuTape
    let stored ← match tape.getValue? x.id with
      | some v => pure v
      | none => throw <| IO.userError "torch: invalid tensor id (missing value)"
    if h : stored.shape = sh then
      pure (stored.cast h)
    else
      throw <| IO.userError <|
        s!"torch: shape mismatch when reading value (expected {sh}, got "
          ++ s!"{stored.shape})"

/--
Record an external input tensor as a leaf on the tape.

The session's `gradEnabled` flag gates `requiresGrad`, so inference sessions keep inputs
non-differentiable even when the caller requests gradients.
-/
def input {α : Type} [Storage α] [TensorTransfer α]
    (s : Session α) {sh : Shape}
  (v : Tensor α sh) (name : Option String := none) (requiresGrad : Bool := false) :
  IO (TensorRef α sh) := do
  let requiresGrad := s.gradEnabled && requiresGrad
  if s.options.usesCUDA then
    let buffer ← TensorTransfer.toSomeBuffer (α := α) (s := sh) v
    let id ← s.recordCUDAPure fun tape =>
      Tape.leaf (t := tape) (value := buffer)
        (name := name) (requiresGrad := requiresGrad)
    s.makeTensorRef id
  else
    let id ← s.recordCPUPure fun tape =>
      Tape.leaf (t := tape) (value := .ofTensor v)
        (name := name) (requiresGrad := requiresGrad)
    s.makeTensorRef id

/-- Record a constant leaf that never receives gradients. -/
def const {α : Type} [Storage α] [TensorTransfer α]
    (s : Session α) {sh : Shape}
    (v : Tensor α sh) (name : Option String := none) : IO (TensorRef α sh) :=
  s.input v name (requiresGrad := false)

/--
Record the current value of a parameter as a leaf on the tape.

Every call registers a separate leaf, including calls after the parameter changes. A CUDA leaf
holds the mirror observed at that call, so replacing the parameter leaves an earlier leaf valid,
and a CPU leaf holds the host value, synchronized first if it is stale. The registration is kept
when gradients are disabled too, because tape cleanup must spare these shared snapshots.
-/
def use {α : Type} [Storage α] [TensorTransfer α]
    (s : Session α) {sh : Shape}
  (p : Param α sh) : IO (TensorRef α sh) := do
  let requiresGrad := s.gradEnabled && p.requiresGrad
  let id ←
    if s.options.usesCUDA then
      let buffer ← Param.getCUDAValue p
      s.recordCUDAPure fun tape =>
        tape.addNode
          { name := p.name
            value := buffer
            ownsValue := false
            requiresGrad := requiresGrad
            parents := #[]
            backward := fun _ => .ok #[] }
    else
      Param.syncToHost (α := α) (sh := sh) p
      let v ← p.value.get
      s.recordCPUPure fun tape =>
        Tape.leaf (t := tape)
          (value := .ofTensor v) (name := p.name) (requiresGrad := requiresGrad)
  s.paramsByLeaf.modify (fun m => m.insert id (SomeParam.ofParam p))
  s.parameterStorageByLeaf.modify fun m =>
    m.insert id
      { shape := sh
        value := p.value
        cudaValue := p.cudaValue
        hostSynced := p.hostSynced }
  s.makeTensorRef id

end Session

end TorchLean
