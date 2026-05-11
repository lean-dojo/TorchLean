/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN

/-!
# Quickstart: Tensor Basics

This is the first stop in the TorchLean examples. It does **not** use sessions, CUDA, or autograd.
It is just about building typed tensors in Lean with a convenient constructor layer.

What it demonstrates:
- 1D and N-D constructors from literal lists (`tensor1d`, `tensorND`),
- the fact that the element type `α` is the “dtype” (e.g. `Float`, `ℚ`, `Int`),
- Float-literal convenience constructors for executable float32 (`tensorF32_*`),
- why we generally do not try to `print` tensors over `ℝ` (noncomputable / too large).

Run:
  `lake exe torchlean quickstart_tensors`
-/

@[expose] public section


namespace NN.Examples.Quickstart.TensorBasics

open Spec

def main (_args : List String) : IO Unit := do
  IO.println "== Quickstart: tensor basics =="

  -- The “dtype” here is just the element type `α`.
  let xF := NN.Tensor.tensor1d (α := Float) [0.1, 0.2, 0.3, 0.4]
  let xQ := NN.Tensor.tensor1d (α := ℚ) [0.1, 0.2, 0.3, 0.4]
  let xI := NN.Tensor.tensor1d (α := Int) [1, 2, 3, 4]

  NN.Tensor.print xF
  NN.Tensor.print xQ
  NN.Tensor.print xI

  -- Convenience: build from Float literals then convert to executable float32 (IEEE32Exec).
  let x32 := NN.Tensor.tensorF321d [0.1, 0.2, 0.3, 0.4]
  NN.Tensor.print x32

  -- N-D tensor using "nested brackets" (like nested Python lists in PyTorch).
  -- This is often the clearest way to see where each element goes.
  let x3 : Spec.Tensor Float (NN.Tensor.shapeOfDims [2, 2, 2]) :=
    tensor! [
      [ [1, 2], [3, 4] ],
      [ [5, 6], [7, 8] ]
    ]
  NN.Tensor.print x3

  -- The lower-level equivalent is `tensorND`: you provide dims + a flat row-major list.
  -- Row-major means the last dimension changes fastest:
  -- the above `x3` is the same as `tensorND [2,2,2] [1,2,3,4,5,6,7,8]`.

  -- Showing the intentional “Real tensors refuse to print” behavior.
  let xR := NN.Tensor.tensor1d (α := ℝ) [0.1, 0.2, 0.3, 0.4]
  try
    NN.Tensor.print xR
  catch e =>
    IO.println s!"Expected failure printing Tensor ℝ: {e}"

end NN.Examples.Quickstart.TensorBasics
