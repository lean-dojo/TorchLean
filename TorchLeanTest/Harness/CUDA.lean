/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.LibTorch.Kernel
public import TorchLeanTest.Harness
public import TorchLean.NN.Tape.CPU

/-!
# CUDA Runtime Test Utils

Helpers for the CUDA kernel-coverage tests. They compare results of the CPU eager tape
(`Tape` over `Float`) with those of the CUDA eager tape
(`Tape LibTorch.SomeBuffer`), whose device buffers the
conversions here create in float32.

The default build does not link LibTorch, and the curated runner skips this GPU suite there. Native
coverage needs a CUDA build and a visible device, and a CUDA build without one fails the run.
-/

@[expose] public section

open TorchLean.Shape (size)

namespace TorchLean.Test
namespace Harness

open TorchLean.Tensor

-- Scalar comparisons go through `assertApprox`. The CUDA tolerances are set at the
-- call sites, and `assertTensorApprox` defaults to `1e-3`.

/-- The sum over every axis of `s`, the registered reduction behind a scalar loss. -/
def sumAll (s : Shape) :
    Op [.value s] [.value (s.reduce (List.finRange s.rank) false)] :=
  .reduce .sum (List.finRange s.rank) false fun h => by rcases h with h | h <;> cases h

/-- Record a registered operator with one value output on a CUDA tape through its entry. -/
def record {ins : List Port} {s : Shape}
    (t : Tape LibTorch.SomeBuffer) (name : String)
    (op : Op ins [.value s]) (args : LibTorch.Args ins) :
    IO (Tape LibTorch.SomeBuffer × Nat) := do
  let (t, result) ← LibTorch.Tape.apply t name op args (pure .float32)
  let (id, ()) ← IO.ofExcept result
  pure (t, id)

/-- Fails unless `a` and `b` have the same size and equal entries as numbers (`-0` equal to `+0`, a
NaN equal to nothing), naming the first index that differs and its bits. -/
def assertFloatArrayEq (msg : String) (a b : FloatArray) : IO Unit := do
  if a.size != b.size then
    throw <| IO.userError s!"{msg}: size mismatch ({a.size} vs {b.size})"
  for i in [:a.size] do
    let x := a.get! i
    let y := b.get! i
    if x != y then
      throw <| IO.userError s!"{msg}[{i}]: got {x} (0x{x.toBits.toBitVec.toHex}), expected {y} \
        (0x{y.toBits.toBitVec.toHex})"

/-- Fails unless `a` and `b` have the same size and finite entries within the absolute tolerance
`tol` of each other. -/
def assertFloatArrayApprox (msg : String) (a b : FloatArray) (tol : Float) : IO Unit := do
  if a.size != b.size then
    throw <| IO.userError s!"{msg}: size mismatch ({a.size} vs {b.size})"
  for i in [:a.size] do
    assertApprox s!"{msg}[{i}]" (a.get! i) (b.get! i) tol

/-- Convert a spec tensor to a CUDA buffer (row-major, cast to float32). -/
def tensorToBuffer {s : Shape} (t : Tensor Float s) : LibTorch.Buffer :=
  LibTorch.Buffer.ofFloatArray (t.to FloatArray)

/-- Convert a spec tensor to a CUDA `SomeBuffer` wrapper. -/
def tensorToSomeBuffer {s : Shape} (t : Tensor Float s) : LibTorch.SomeBuffer :=
  { s := s, buf := tensorToBuffer (s := s) t }

/-- Convert a CUDA buffer back to a spec tensor of shape `s`, failing when the sizes differ. -/
def bufferToTensor {s : Shape} (b : LibTorch.Buffer) : IO (Tensor Float s) := do
  let a := LibTorch.Buffer.toFloatArray b
  match (Tensor.from a).reshape? s with
  | some t => pure t
  | none =>
      throw <| IO.userError
        s!"cuda test: buffer size mismatch (expected {Shape.size s} elements, got {a.size})"

/-- Convert a CUDA `SomeBuffer` back to a spec tensor of shape `s`, failing when its shape or size
differs. -/
def someBufferToTensor {s : Shape} (ab : LibTorch.SomeBuffer) : IO (Tensor
  Float s) := do
  if _h : ab.s = s then
    bufferToTensor (s := s) ab.buf
  else
    throw <| IO.userError "cuda test: SomeBuffer shape mismatch"

/-- Read a typed CPU tape value from an id. -/
def cpuValue {s : Shape} (t : Tape (SomeTensor Float)) (id : Nat) : IO
    (Tensor Float s) :=
  IO.ofExcept (Tape.requireValue (α := Float) (t := t) (s := s) id)

/-- Read a typed CUDA tape value from an id. -/
def cudaValue {s : Shape} (t : Tape LibTorch.SomeBuffer) (id
    : Nat) : IO (Tensor
  Float s) := do
  let b ← IO.ofExcept (LibTorch.Tape.requireValue (t := t) id s)
  bufferToTensor (s := s) b

/-- The gradient of node `id` in the dense gradient array of the CPU tape, failing when `id` is out
of bounds or the gradient's shape is not `s`. -/
def cpuGrad {s : Shape} (grads : Array (SomeTensor Float)) (id : Nat) :
    IO (Tensor Float s) := do
  let g ← match grads[id]? with
    | some g => pure g
    | none => throw <| IO.userError s!"cuda test: gradient id out of bounds: {id}"
  if h : g.shape = s then
    pure (g.cast h)
  else
    throw <| IO.userError s!"cuda test: CPU grad shape mismatch at id {id}"

/-- The gradient of node `id` in the dense gradient array of the CUDA tape, failing when `id` is out
of bounds or the buffer's shape or size is not that of `s`. -/
def cudaGrad {s : Shape} (grads : Array LibTorch.SomeBuffer) (id : Nat) :
    IO (Tensor Float s) := do
  let g ← match grads[id]? with
    | some g => pure g
    | none => throw <| IO.userError s!"cuda test: gradient id out of bounds: {id}"
  someBufferToTensor (s := s) g

/--
Approximate equality for `Tensor Float s`, entrywise after flattening in CUDA row-major order.

The entries must be finite and within the absolute tolerance `tol` of each other.
-/
def assertTensorApprox {s : Shape} (msg : String) (x y : Tensor Float s) (tol : Float := 1e-3) :
    IO Unit := do
  let ax := x.to FloatArray
  let ay := y.to FloatArray
  if ax.size != ay.size then
    throw <| IO.userError s!"{msg}: size mismatch ({ax.size} vs {ay.size})"
  for i in [:ax.size] do
    assertApprox s!"{msg}[{i}]" (ax.get! i) (ay.get! i) tol

end Harness
end TorchLean.Test
