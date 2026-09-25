/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.Chain.Syntax
public import NN.Runtime.Autograd.Model.Layers.Activations

/-!
# GraphSpec Spatial Primitives

This file extends the **sequential** GraphSpec core (`NN.GraphSpec.Chain.Primitives`) with
single-input/single-output spatial operations used by convolutional pipelines.

These are not model definitions. They are reusable nodes in the GraphSpec vocabulary:

- `Primitive.conv` wraps the arbitrary-rank convolution specification and runtime operation;
- `Primitive.maxPool` wraps arbitrary-rank max pooling;
- `Primitive.batchNorm` wraps normalization over an arbitrary spatial shape;
- `Primitive.flatten` is the bridge from image-like tensors to vector classifiers.

The corresponding model examples live under `NN.GraphSpec.Models`.

Important scope note:

- These primitives all fit the sequential language `Chain ps σ τ` because they have one input
  tensor and one output tensor (no merging of paths).
- Residual networks require skip connections ($y+x$), which are **multi-input** and require
  **sharing**. For that, use `NN.GraphSpec.DAG`, whose DAG primitive constructors reuse these
  sequential adapters when possible.

Why only these spatial operations?

GraphSpec only exposes an operation once we have both sides of the contract in place:

1. a pure Spec meaning, and
2. an executable TorchLean program meaning.

The general always-available primitives (`linear`, `relu`, `softmax`) live in
`NN.GraphSpec.Chain.Primitives`; this file is the current spatial extension pack. More packs can be
added as we decide which runtime/spec operations should become architecture-level GraphSpec nodes.

## Parameter convention (sequential GraphSpec)

Each primitive has an explicit type-level parameter-shape list `ps : List Shape`.

For example, an N-D convolution is parameterized by:

- `kernel : Tensor α (outC :: inC :: kernelShape)`
- `bias   : Tensor α [outC]`

When you compose graphs with `>>>`, these parameter-shape lists concatenate, giving a typed
interface for model parameters.

## References / citations (informal pointers)

- Convolutional networks: LeCun et al. (1998), “Gradient-based learning applied to document
  recognition”.
- BatchNorm: Ioffe & Szegedy (2015), “Batch Normalization: Accelerating Deep Network Training…”.
-/

@[expose] public section


namespace NN
namespace GraphSpec

open _root_.Spec _root_.TorchLean
open TorchLean.Tensor
open _root_.TorchLean.Tensor

namespace Primitive

/--
Arbitrary-rank convolution on a channels-first tensor with no batch axis.

Inputs:

- parameters `kernel, bias` (in that order),
- input tensor `x : (inChannels, spatial...)`.

Output:

The output has shape `(outChannels, convOutSpatial spatial kernel stride padding...)`.
 -/
def conv
    {d : Nat} (inC outC : Nat)
    (kernel stride padding spatial : TorchLean.Tensor Nat [d]) :
    Primitive
      [Shape.ofList (outC :: inC :: (Tensor.to kernel (List Nat))), [outC]]
      (Shape.ofList (inC :: (Tensor.to spatial (List Nat))))
      (Shape.ofList
        (outC :: (Tensor.to (Spec.convOutSpatial spatial kernel stride padding) (List Nat)))) :=
  { name := s!"conv(rank={d},in={inC},out={outC})"
    specFwd := fun {α} _storage _ctx params x =>
      match params with
      | .cons k (.cons b .nil) =>
          let layer : Spec.ConvSpec d inC outC kernel stride padding α :=
            { kernel := k, bias := b }
          Spec.convSpec (α := α) (inSpatial := spatial) layer x
    program := fun {α} _storage _ctx =>
      fun {m} _ _ =>
        fun k b x =>
          _root_.Runtime.Autograd.Torch.conv (m := m) (α := α)
            (d := d) (inC := inC) (outC := outC)
            (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := spatial)
            k b x
    toLayerM? := some (fun i =>
      let kernelShape := Shape.ofList (outC :: inC :: (Tensor.to kernel (List Nat)))
      let biasShape : Shape := [outC]
      let kernelInit : Tensor Float kernelShape :=
        Runtime.Autograd.Torch.Init.tensor (s := kernelShape)
          (sch := .uniform (-0.1) 0.1) (seed := i)
      let biasInit : Tensor Float biasShape :=
        Runtime.Autograd.Torch.Init.tensor (s := biasShape) (sch := .zeros) (seed := i)
      ⟨ { kind := s!"Conv(rank={d},in={inC},out={outC})"
          stateShapes := [kernelShape, biasShape]
          initState := .cons kernelInit (.cons biasInit .nil)
          runtimeInit := some
            (.cons (Runtime.Autograd.Model.Module.RuntimeInit.FloatInit.ofScheme
              (.uniform (-0.1) 0.1) i) (.cons .zeros .nil))
          requiresGrad := #[true, true]
          validateConfig := do
            if inC = 0 then
              throw "Conv: input channel count must be positive"
            if outC = 0 then
              throw "Conv: output channel count must be positive"
            if spatial.prod = 0 then
              throw "Conv: input spatial dimensions must be positive"
            if !decide (∀ axis : Fin d, kernel.getScalar axis ≠ 0) then
              throw "Conv: kernel size entries must be positive"
            if !decide (∀ axis : Fin d, stride.getScalar axis ≠ 0) then
              throw "Conv: stride entries must be positive"
            if (Spec.convOutSpatial spatial kernel stride padding).prod = 0 then
              throw "Conv: geometry produced an empty spatial grid"
          forward := fun _ {α} _ _ =>
            fun {m} _ _ => fun k b x =>
              _root_.Runtime.Autograd.Torch.conv (m := m) (α := α)
                (d := d) (inC := inC) (outC := outC)
                (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := spatial)
                k b x }
      , by rfl ⟩)
    countsAsLayer := true
  }

/--
Arbitrary-rank max pooling on a channels-first tensor (parameter-free).

Output shapes follow the standard pooling size formulas:

Each spatial axis uses the corresponding kernel, stride, and padding entry.
 -/
def maxPool
    {d : Nat} (channels : Nat)
    (kernel stride padding spatial : TorchLean.Tensor Nat [d])
    {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
    {hStride : ∀ i : Fin d, stride.getScalar i ≠ 0} :
    Primitive []
      (Shape.ofList (channels :: (Tensor.to spatial (List Nat))))
      (Shape.ofList (channels ::
        (Tensor.to (Spec.poolOutSpatialPad spatial kernel stride padding) (List Nat)))) :=
  { name := s!"max_pool(rank={d})"
    specFwd := fun {α} _storage _ctx _params x =>
      let layer : Spec.MaxPoolSpec d kernel stride padding hKernel hStride := {}
      Spec.maxPoolSpec (layer := layer) x
    program := fun {α} _storage _ctx =>
      fun {m} _ _ =>
        fun x =>
          _root_.Runtime.Autograd.Torch.maxPool (m := m) (α := α)
            (d := d) (C := channels) (inSpatial := spatial)
            (kernel := kernel) (stride := stride) (padding := padding)
            x
    toLayerM? := some (fun _i =>
      ⟨ { kind := s!"MaxPool(rank={d})"
          stateShapes := []
          initState := .nil
          requiresGrad := #[]
          forward := fun _ {α} _ _ =>
            fun {m} _ _ => fun x =>
              _root_.Runtime.Autograd.Torch.maxPool (m := m) (α := α)
                (d := d) (C := channels) (inSpatial := spatial)
                (kernel := kernel) (stride := stride) (padding := padding)
                x }
      , by rfl ⟩)
    countsAsLayer := false
  }

/--
Flatten any tensor to a 1D vector (parameter-free).

Output shape is `[Spec.Shape.size s]`, i.e. a vector whose length is the number of
elements of the input shape.

This is a reshape/view operation (no arithmetic), used to connect convolutional features to a
vector-valued classifier head.

PyTorch analogy: `torch.flatten(x)`.
 -/
def flatten (s : Shape) : Primitive [] s [Spec.Shape.size s] :=
  { name := "flatten"
    specFwd := fun {α} _storage _ctx _params x =>
      TorchLean.Tensor.flattenSpec (α := α) (shape := s) x
    program := fun {α} _storage _ctx =>
      fun {m} _ _ =>
        fun x => Runtime.Autograd.Model.flatten (m := m) (α := α) (s := s) x
    toLayerM? := some (fun _i => ⟨Runtime.Autograd.Model.Layers.flatten (s := s), by rfl⟩)
    countsAsLayer := false
  }

/--
BatchNorm over every axis in `spatial`, independently for each channel.

Parameters are the affine vectors `(gamma, beta)`. GraphSpec keeps this primitive stateless;
running statistics belong to the stateful model layer.
-/
def batchNorm (channels : Nat) (spatial : Shape)
    (hWellFormed : (Shape.dim channels spatial).wellFormed) :
    Primitive [[channels], [channels]]
      (.dim channels spatial) (.dim channels spatial) :=
  { name := s!"batch_norm(channels={channels},rank={Shape.rank spatial})"
    specFwd := fun {α} _storage _ctx params x =>
      match params with
      | .cons gamma (.cons beta .nil) =>
          letI : Shape.WellFormed (.dim channels spatial) := ⟨hWellFormed⟩
          Spec.batchNorm (α := α) x gamma beta
    program := fun {α} _storage _ctx =>
      fun {m} _ _ =>
        fun gamma beta x =>
          _root_.Runtime.Autograd.Torch.batchNorm (m := m) (α := α)
            (channels := channels) (sSpatial := spatial) hWellFormed x gamma beta
    toLayerM? := some (fun i =>
      let channelShape : Shape := [channels]
      let gamma : Tensor Float channelShape :=
        Runtime.Autograd.Torch.Init.tensor (s := channelShape) (sch := .ones) (seed := i)
      let beta : Tensor Float channelShape :=
        Runtime.Autograd.Torch.Init.tensor (s := channelShape) (sch := .zeros) (seed := i)
      ⟨ { kind := s!"BatchNorm(channels={channels},rank={Shape.rank spatial})"
          stateShapes := [channelShape, channelShape]
          initState := .cons gamma (.cons beta .nil)
          runtimeInit := some (.cons .ones (.cons .zeros .nil))
          requiresGrad := #[true, true]
          forward := fun _ {α} _ _ =>
            fun {m} _ _ => fun gamma beta x =>
              _root_.Runtime.Autograd.Torch.batchNorm (m := m) (α := α)
                (channels := channels) (sSpatial := spatial) hWellFormed x gamma beta }
      , by rfl ⟩)
    countsAsLayer := true
  }

end Primitive

namespace Chain

/-- Chain constructor for `Primitive.conv`. -/
def conv
    {d : Nat} (inC outC : Nat)
    (kernel stride padding spatial : TorchLean.Tensor Nat [d]) :
    Chain
      [Shape.ofList (outC :: inC :: (Tensor.to kernel (List Nat))), [outC]]
      (Shape.ofList (inC :: (Tensor.to spatial (List Nat))))
      (Shape.ofList
        (outC :: (Tensor.to (Spec.convOutSpatial spatial kernel stride padding) (List Nat)))) :=
  .prim (Primitive.conv (inC := inC) (outC := outC) kernel stride padding spatial)

/-- Chain constructor for `Primitive.maxPool`. -/
def maxPool
    {d : Nat} (channels : Nat)
    (kernel stride padding spatial : TorchLean.Tensor Nat [d])
    {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
    {hStride : ∀ i : Fin d, stride.getScalar i ≠ 0} :
    Chain [] (Shape.ofList (channels :: (Tensor.to spatial (List Nat))))
      (Shape.ofList (channels ::
        (Tensor.to (Spec.poolOutSpatialPad spatial kernel stride padding) (List Nat)))) :=
  .prim (Primitive.maxPool (channels := channels) kernel stride padding spatial
    (hKernel := hKernel) (hStride := hStride))

/-- Chain constructor for `Primitive.flatten`. -/
def flatten (s : Shape) : Chain [] s [Spec.Shape.size s] :=
  .prim (Primitive.flatten s)

/-- Chain constructor for `Primitive.batchNorm`. -/
def batchNorm (channels : Nat) (spatial : Shape)
    (hWellFormed : (Shape.dim channels spatial).wellFormed) :
    Chain [[channels], [channels]]
      (.dim channels spatial) (.dim channels spatial) :=
  .prim (Primitive.batchNorm channels spatial hWellFormed)

end Chain

end GraphSpec
end NN
