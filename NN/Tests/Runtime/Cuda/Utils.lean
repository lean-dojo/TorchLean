/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Core
public import NN.Runtime.Autograd.Engine.Cuda.Tape
public import NN.Runtime.Autograd.Engine.Cuda.Convert
public import NN.Tests.Utils

/-!
# CUDA Runtime Test Utils

Small helpers for CUDA kernel-coverage tests.

These tests compare:
- CPU eager tape results (`Runtime.Autograd.Tape`, `Float`), against
- CUDA eager tape results (`Runtime.Autograd.Cuda.Tape`, float32 buffers).

When TorchLean is built without CUDA (`lake build` default), the CUDA externs run via CPU stub
implementations, so these tests still run on CI without a GPU.
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace Utils

open Spec TorchLean
open TorchLean TorchLean.Tensor

/-- Convert any `Except String` into `IO` by throwing `IO.userError` on failure. -/
def okOrThrow {α : Type} : Except String α → IO α
  | .ok a => pure a
  | .error e => throw <| IO.userError e

-- Scalar comparisons share `Tests.Utils.assertApprox`; CUDA tolerances are explicit at call sites.

/-- Construct a raw float buffer from an array of values. -/
def floatArray (xs : Array Float) : FloatArray :=
  FloatArray.mk xs

/-- Exact elementwise equality for two raw float buffers, naming the first index that differs. -/
def assertFloatArrayEq (msg : String) (a b : FloatArray) : IO Unit := do
  if a.size != b.size then
    throw <| IO.userError s!"{msg}: size mismatch ({a.size} vs {b.size})"
  for i in [:a.size] do
    let x := a.get! i
    let y := b.get! i
    if x != y then
      throw <| IO.userError s!"{msg}[{i}]: got {x}, expected {y}"

/-- Compare finite buffer elements with an explicit absolute tolerance. -/
def assertFloatArrayApprox (msg : String) (a b : FloatArray) (tol : Float) : IO Unit := do
  if a.size != b.size then
    throw <| IO.userError s!"{msg}: size mismatch ({a.size} vs {b.size})"
  for i in [:a.size] do
    Tests.Utils.assertApprox s!"{msg}[{i}]" (a.get! i) (b.get! i) tol

/-- Convert a spec tensor to a CUDA buffer (row-major, cast to float32). -/
def tensorToBuffer {s : Shape} (t : Tensor Float s) : Runtime.Autograd.Cuda.Buffer :=
  Runtime.Autograd.Cuda.Buffer.ofFloatArray (Runtime.Autograd.Cuda.Convert.flattenFloat (s := s) t)

/-- Convert a spec tensor to a CUDA `AnyBuffer` wrapper. -/
def tensorToAnyBuffer {s : Shape} (t : Tensor Float s) : Runtime.Autograd.Cuda.AnyBuffer :=
  { s := s, buf := tensorToBuffer (s := s) t }

/-- Convert a CUDA buffer back to a spec tensor (checks size matches `s`). -/
def bufferToTensor {s : Shape} (b : Runtime.Autograd.Cuda.Buffer) : IO (Tensor Float s) := do
  let a := Runtime.Autograd.Cuda.Buffer.toFloatArray b
  match Runtime.Autograd.Cuda.Convert.unflattenFloat? (s := s) a with
  | some t => pure t
  | none =>
      throw <| IO.userError
        s!"cuda test: buffer size mismatch (expected {Spec.Shape.size s} elements, got {a.size})"

/-- Convert a CUDA `AnyBuffer` back to a typed spec tensor (checks shape + size). -/
def anyBufferToTensor {s : Shape} (ab : Runtime.Autograd.Cuda.AnyBuffer) : IO (Tensor Float s) := do
  if _h : ab.s = s then
    bufferToTensor (s := s) ab.buf
  else
    throw <| IO.userError "cuda test: AnyBuffer shape mismatch"

/-- Read a typed CPU tape value from an id. -/
def cpuValue {s : Shape} (t : Runtime.Autograd.Tape Float) (id : Nat) : IO (Tensor Float s) :=
  okOrThrow (Runtime.Autograd.Tape.requireValue (α := Float) (t := t) (s := s) id)

/-- Read a typed CUDA tape value from an id. -/
def cudaValue {s : Shape} (t : Runtime.Autograd.Cuda.Tape) (id : Nat) : IO (Tensor Float s) := do
  let b ← okOrThrow (Runtime.Autograd.Cuda.Tape.requireValue (t := t) id s)
  bufferToTensor (s := s) b

/-- Extract a typed gradient tensor from the CPU dense-grad array (with a shape check). -/
def cpuGrad {s : Shape} (grads : Array (Spec.SomeTensor Float)) (id : Nat) :
    IO (Tensor Float s) := do
  let g ← match grads[id]? with
    | some g => pure g
    | none => throw <| IO.userError s!"cuda test: gradient id out of bounds: {id}"
  if h : g.shape = s then
    pure (g.cast h)
  else
    throw <| IO.userError s!"cuda test: CPU grad shape mismatch at id {id}"

/-- Extract a typed gradient tensor from the CUDA dense-grad array (with a shape check). -/
def cudaGrad {s : Shape} (grads : Array Runtime.Autograd.Cuda.AnyBuffer) (id : Nat) :
    IO (Tensor Float s) := do
  let g ← match grads[id]? with
    | some g => pure g
    | none => throw <| IO.userError s!"cuda test: gradient id out of bounds: {id}"
  anyBufferToTensor (s := s) g

/--
Approximate equality for `Tensor Float s` by flattening in CUDA row-major order.

This compares the numeric results rather than the representation.
-/
def assertTensorApprox {s : Shape} (msg : String) (x y : Tensor Float s) (tol : Float := 1e-3) :
    IO Unit := do
  let ax := Runtime.Autograd.Cuda.Convert.flattenFloat (s := s) x
  let ay := Runtime.Autograd.Cuda.Convert.flattenFloat (s := s) y
  if ax.size != ay.size then
    throw <| IO.userError s!"{msg}: size mismatch ({ax.size} vs {ay.size})"
  for i in [:ax.size] do
    Tests.Utils.assertApprox s!"{msg}[{i}]" (ax.get! i) (ay.get! i) tol

end Utils
end Cuda
end Tests
