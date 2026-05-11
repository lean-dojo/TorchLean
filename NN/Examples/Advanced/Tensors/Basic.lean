/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import NN.Entrypoint.Widgets
public import NN.Spec.Core.TensorBridge
meta import NN.Spec.Core.TensorBridge

/-!
# Advanced tensor basics

This file is the advanced “tensor basics” map for TorchLean. It covers the core operations people
usually ask about once they understand typed shapes:

- constructing small tensors;
- indexing and slicing through the shape-indexed `Spec.Tensor` representation;
- converting between `Spec.Tensor α s` and array-backed `TensorArray.Tensor α shape`;
- matrix-vector multiplication and elementwise arithmetic;
- manual reshaping when you want the row-major convention to be visible.

The bridge functions themselves live in `NN/Spec/Core/TensorBridge.lean`. This example imports that
library code and only supplies concrete demo values. In other words: this file is a tour, not a
second tensor implementation.

## Why both representations exist

We intentionally keep two representations because they serve different goals:

- The spec representation (`Spec.Tensor`) is a nested function tree (`Fin n → ...`), which is great
  for writing total definitions and proving shape-correctness by structural recursion.
- The array representation (`TensorArray.Tensor`) is compact and practical for serialization and
  numeric kernels (e.g. `matvec` on flat buffers).

The conversion boundary is where you want to be explicit about layout and conventions. TorchLean
uses a *row-major* convention for flattening/unflattening, so that the last axis varies fastest.

## How to read this file

The sections below are small "micro demos" you can copy into other scratch files:

- Convert an array-backed matrix to a `Spec.Tensor` and index into it.
- Do a simple linear `matvec` in `TensorArray`, then convert the output back to `Spec.Tensor`.
- Work with a small batched tensor and slice out samples.

Put your cursor on the `#tensor_view` and `#tensor_stats_view` commands to inspect the values in
the Lean infoview. The notes below also show the closest PyTorch spelling for each operation.

Important distinction:
- Ordinary tutorial definitions use plain `def`; this is the code you should copy into normal
  TorchLean modules.
- A few `meta def ...View` declarations exist only for widgets. The infoview evaluator needs
  interpreter-visible bridge code, while normal compiled modules can use the plain `def`s above
  them.
-/

@[expose] public section


namespace NN.Examples.Advanced.Tensors.Basic

open TensorArray Spec TensorBridge

/-! ## 1) Array-backed matrix → spec tensor (and indexing) -/

/-!
PyTorch analogue:

```python
matrix = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
first_row = matrix[0]
```

TorchLean keeps the shape in the type after the conversion: `matrixSpec` has shape `[2,3]`, and
`firstRowSpec` has shape `[3]`.
-/

/-- A `2×3` matrix stored in row-major order as a `TensorArray.Tensor`. -/
def matrixArray : TensorArray.Tensor Float [2, 3] :=
  TensorArray.ofArray #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0] [2, 3] (by simp)

/-- Convert the matrix into the spec-level representation (`Spec.Tensor`). -/
def matrixSpec : Spec.Tensor Float (listToShape [2, 3]) :=
  toTensor matrixArray

/-- Extract the first row (shape `[3]`) by pattern-matching on the outer dimension. -/
def firstRowSpec : Spec.Tensor Float (listToShape [3]) :=
  match matrixSpec with
  | Spec.Tensor.dim rows => rows ⟨0, by simp⟩

/-- Convert the extracted row back into the array-backed representation. -/
def firstRowAsArray : TensorArray.Tensor Float [3] :=
  toTensorArray firstRowSpec

/-!
Widget lane for this section.

These `meta` declarations deliberately mirror the ordinary definitions above. They are not the
recommended programming style; they simply make the bridge computation executable for ProofWidgets.
-/
meta def matrixSpecView : Spec.Tensor Float (listToShape [2, 3]) :=
  toTensor (TensorArray.ofArray #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0] [2, 3] (by simp))

meta def firstRowSpecView : Spec.Tensor Float (listToShape [3]) :=
  match matrixSpecView with
  | Spec.Tensor.dim rows => rows ⟨0, by simp⟩

#tensor_view matrixSpecView
#tensor_view firstRowSpecView
#tensor_stats_view matrixSpecView

/-! ## 2) A compact linear layer: `matvec` in `TensorArray`, then back to `Spec.Tensor` -/

/-!
PyTorch analogue:

```python
weight = torch.full((4, 3), 0.1)
x = torch.tensor([1., 2., 3.])
y = weight @ x
```

The array-backed `TensorArray.matvec` is the compact runtime-style operation; converting the result
back with `toTensor` makes it usable by spec/proof-facing code and by the tensor widgets.
-/

/-- A `4×3` weight matrix filled with a constant, stored as `TensorArray`. -/
def weightMatrix : TensorArray.Tensor Float [4, 3] :=
  TensorArray.full [4, 3] 0.1

/-- A length-3 input vector built as a `Spec.Tensor` (values 1,2,3). -/
def inputVector : Spec.Tensor Float (listToShape [3]) :=
  Spec.vectorTensor (fun i => Float.ofNat (i.val + 1))

/-- The same input, converted to `TensorArray` so we can call `TensorArray.matvec`. -/
def inputAsArray : TensorArray.Tensor Float [3] :=
  toTensorArray inputVector

/-- Compute `weightMatrix @ inputAsArray` using the array-backed kernel. -/
def linearOutput : TensorArray.Tensor Float [4] :=
  TensorArray.matvec weightMatrix inputAsArray

/-- Convert the output back to `Spec.Tensor` (useful if the rest of the pipeline is spec-level). -/
def outputAsInductive : Spec.Tensor Float (listToShape [4]) :=
  toTensor linearOutput

#tensor_view inputVector

/-! Widget-only mirror for the bridge-backed output. -/
meta def outputAsInductiveView : Spec.Tensor Float (listToShape [4]) :=
  toTensor (TensorArray.matvec (TensorArray.full [4, 3] 0.1)
    (toTensorArray (Spec.vectorTensor (fun i => Float.ofNat (i.val + 1)))))

#tensor_view outputAsInductiveView
#tensor_stats_view outputAsInductiveView

/-! ## 3) A small batch tensor and slicing samples -/

/-!
PyTorch analogue:

```python
batch = torch.arange(1., 13.).reshape(2, 2, 3)
first = batch[0]
second = batch[1]
```

In TorchLean, slicing a `Spec.Tensor` is ordinary dependent pattern matching: the outer
`Spec.Tensor.dim` exposes the `Fin 2 -> Tensor ...` function, and the index proof records that the
sample exists.
-/

/-- A `2×2×3` batch, encoded as a flat array with runtime shape `[2,2,3]`. -/
def batchArray : TensorArray.Tensor Float [2, 2, 3] :=
  TensorArray.ofArray
    #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    [2, 2, 3]
    (by simp)

/-- Convert the whole batch into the spec tensor. -/
def batchSpec : Spec.Tensor Float (listToShape [2, 2, 3]) :=
  toTensor batchArray

/-- Slice out the first sample (shape `[2,3]`). -/
def firstSampleSpec : Spec.Tensor Float (listToShape [2, 3]) :=
  match batchSpec with
  | Spec.Tensor.dim samples => samples ⟨0, by simp⟩

/-- Slice out the second sample (shape `[2,3]`). -/
def secondSampleSpec : Spec.Tensor Float (listToShape [2, 3]) :=
  match batchSpec with
  | Spec.Tensor.dim samples => samples ⟨1, by simp⟩

/-- Convert the first sample back to `TensorArray` (e.g. for a kernel call). -/
def firstSampleAsArray : TensorArray.Tensor Float [2, 3] :=
  toTensorArray firstSampleSpec

/-- Convert the second sample back to `TensorArray`. -/
def secondSampleAsArray : TensorArray.Tensor Float [2, 3] :=
  toTensorArray secondSampleSpec

/-! Widget-only mirrors for the bridge-backed batch tensors. -/
meta def batchSpecView : Spec.Tensor Float (listToShape [2, 2, 3]) :=
  toTensor <|
    TensorArray.ofArray
      #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
      [2, 2, 3]
      (by simp)

meta def firstSampleSpecView : Spec.Tensor Float (listToShape [2, 3]) :=
  match batchSpecView with
  | Spec.Tensor.dim samples => samples ⟨0, by simp⟩

meta def secondSampleSpecView : Spec.Tensor Float (listToShape [2, 3]) :=
  match batchSpecView with
  | Spec.Tensor.dim samples => samples ⟨1, by simp⟩

#tensor_view batchSpecView
#tensor_view firstSampleSpecView
#tensor_view secondSampleSpecView

/-! ## 4) Reshaping example (manual, spec-side) -/

/-!
PyTorch analogue:

```python
flat = torch.tensor([1., 2., 3., 4., 5., 6.])
reshaped = flat.reshape(2, 3)
```

This section intentionally spells out the row-major index arithmetic. Most user code should prefer
library reshape helpers when available, but this makes the layout convention completely explicit.
-/

/-- A flat length-6 vector in `TensorArray`. -/
def flatTensor : TensorArray.Tensor Float [6] :=
  TensorArray.ofArray #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0] [6] (by simp)

/-- Convert it to `Spec.Tensor` (shape `[6]`). -/
def flatAsInductive : Spec.Tensor Float (listToShape [6]) :=
  toTensor flatTensor

/-- Reshape `[6]` into `[2,3]` by explicit index arithmetic (row-major). -/
def reshapedMatrix : Spec.Tensor Float (listToShape [2, 3]) :=
  match flatAsInductive with
  | Spec.Tensor.dim values =>
    Spec.Tensor.dim (fun i =>
      Spec.Tensor.dim (fun j =>
        match values ⟨i.val * 3 + j.val, by
          have h1 : i.val < 2 := i.isLt
          have h2 : j.val < 3 := j.isLt
          -- This is a small bounds proof for `i*3 + j < 6`.
          linarith⟩ with
        | Spec.Tensor.scalar v => Spec.Tensor.scalar v))

/-- Convert the reshaped matrix back to `TensorArray`. -/
def reshapedAsArray : TensorArray.Tensor Float [2, 3] :=
  toTensorArray reshapedMatrix

/-! Widget-only mirror for the manual reshape. -/
meta def reshapedMatrixView : Spec.Tensor Float (listToShape [2, 3]) :=
  let flat : Spec.Tensor Float (listToShape [6]) :=
    toTensor (TensorArray.ofArray #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0] [6] (by simp))
  match flat with
  | Spec.Tensor.dim values =>
    Spec.Tensor.dim (fun i =>
      Spec.Tensor.dim (fun j =>
        match values ⟨i.val * 3 + j.val, by
          have h1 : i.val < 2 := i.isLt
          have h2 : j.val < 3 := j.isLt
          linarith⟩ with
        | Spec.Tensor.scalar v => Spec.Tensor.scalar v))

#tensor_view reshapedMatrixView

/-! ## 5) Elementwise ops across representations -/

/-!
PyTorch analogue:

```python
a = torch.full((2, 2), 2.)
b = torch.full((2, 2), 3.)
c = a + b
```

Here one operand starts array-backed and the other starts spec-backed, so the explicit bridge call
documents where the representation boundary is crossed.
-/

/-- A `2×2` array-backed tensor filled with twos. -/
def tensorA : TensorArray.Tensor Float [2, 2] :=
  TensorArray.full [2, 2] 2.0

/-- A `2×2` spec tensor filled with threes. -/
def tensorB : Spec.Tensor Float (listToShape [2, 2]) :=
  Spec.fill 3.0 (listToShape [2, 2])

/-- Convert `tensorB` so both operands live in the array-backed representation. -/
def tensorBAsArray : TensorArray.Tensor Float [2, 2] :=
  toTensorArray tensorB

/-- Elementwise addition in `TensorArray`. -/
def elementWiseSum : TensorArray.Tensor Float [2, 2] :=
  TensorArray.add tensorA tensorBAsArray

/-- Convert the result back to `Spec.Tensor`. -/
def sumAsInductive : Spec.Tensor Float (listToShape [2, 2]) :=
  toTensor elementWiseSum

#tensor_view tensorB

/-! Widget-only mirror for the bridge-backed elementwise sum. -/
meta def sumAsInductiveView : Spec.Tensor Float (listToShape [2, 2]) :=
  toTensor <|
    TensorArray.add
      (TensorArray.full [2, 2] 2.0)
      (toTensorArray (Spec.fill 3.0 (listToShape [2, 2])))

#tensor_view sumAsInductiveView
#tensor_stats_view sumAsInductiveView

/-! ## 6) "Forward + gradient-shape" worked example -/

/-!
PyTorch analogue:

```python
out = weight @ x
grad_like = 2 * out
```

This is not an autograd example; it is a shape-preserving tensor transformation that looks like a
gradient buffer. Autograd examples live under `NN/Examples/Quickstart/AutogradBasics.lean`.
-/

/-- A compact forward pass: `weights @ input`. -/
def forwardPass (input : TensorArray.Tensor Float [3]) (weights : TensorArray.Tensor Float [4, 3]) :
    TensorArray.Tensor Float [4] :=
  TensorArray.matvec weights input

/-- The same forward pass, returning a spec tensor. -/
def forwardAsInductive (input : TensorArray.Tensor Float [3]) (weights : TensorArray.Tensor Float
  [4, 3]) :
    Spec.Tensor Float (listToShape [4]) :=
  toTensor (forwardPass input weights)

/-- A small "gradient" transformation: elementwise multiply by 2 (same shape, spec-side). -/
def computeGradient (output : Spec.Tensor Float (listToShape [4])) :
    Spec.Tensor Float (listToShape [4]) :=
  match output with
  | Spec.Tensor.dim values =>
    Spec.Tensor.dim (fun i =>
      match values i with
      | Spec.Tensor.scalar v => Spec.Tensor.scalar (2.0 * v))

/-- Convert the example gradient back to `TensorArray`. -/
def gradientAsArray (output : Spec.Tensor Float (listToShape [4])) :
    TensorArray.Tensor Float [4] :=
  toTensorArray (computeGradient output)

def gradientExample : Spec.Tensor Float (listToShape [4]) :=
  computeGradient outputAsInductive

/-! Widget-only mirror for the bridge-backed gradient-like tensor. -/
meta def computeGradientView (output : Spec.Tensor Float (listToShape [4])) :
    Spec.Tensor Float (listToShape [4]) :=
  match output with
  | Spec.Tensor.dim values =>
    Spec.Tensor.dim (fun i =>
      match values i with
      | Spec.Tensor.scalar v => Spec.Tensor.scalar (2.0 * v))

meta def gradientExampleView : Spec.Tensor Float (listToShape [4]) :=
  computeGradientView outputAsInductiveView

#tensor_view gradientExampleView
#tensor_stats_view gradientExampleView

end NN.Examples.Advanced.Tensors.Basic
