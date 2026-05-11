/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.Core
public import NN.Spec.Layers.GlobalPooling
import Mathlib.Algebra.Order.Algebra

/-!
# GraphSpec Vision Primitives

This file extends the **sequential** GraphSpec core (`NN.GraphSpec.Core`) with
single-input/single-output vision operation adapters used by classic CNN pipelines.

These are not model definitions. They are reusable nodes in the GraphSpec vocabulary:

- `Primitive.conv2d` wraps `Spec.conv2dSpec` and `Runtime.Autograd.TorchLean.conv2d`;
- `Primitive.maxPool2d` wraps the corresponding Spec/runtime pooling operation;
- `Primitive.batchnormChw` wraps channel-first BatchNorm;
- `Primitive.globalAvgPool2dChw` wraps channel-wise global average pooling;
- `Primitive.flatten` is the bridge from image-like tensors to vector classifiers.

The corresponding model examples live under `NN.GraphSpec.Models`.

Important scope note:

- These primitives all fit the *chain* graph language `Graph ps σ τ` because they have one input
  tensor and one output tensor (no merging of paths).
- Residual networks require skip connections (`y + x`), which are **multi-input** and require
  **sharing**. For that, use `NN.GraphSpec.DAG`, whose DAG primitive constructors reuse these
  sequential adapters when possible.

Why only these vision ops?

GraphSpec only exposes an operation once we have both sides of the contract in place:

1. a pure Spec meaning, and
2. an executable TorchLean program meaning.

The general always-available primitives (`linear`, `relu`, `softmax`) live in
`NN.GraphSpec.Core`; this file is the current vision extension pack. More packs can be added as
we decide which runtime/spec operations should become architecture-level GraphSpec nodes.

## Parameter convention (sequential GraphSpec)

Each primitive has an explicit type-level parameter-shape list `ps : List Shape`.

For example, convolution is parameterized by:

- `kernel : OIHW outC inC kH kW`
- `bias   : Vec outC`

so the primitive has `ps = [OIHW ..., Vec ...]`. When you compose graphs with `>>>`, these `ps`
lists concatenate, giving a typed “ABI” for model parameters.

## References / citations (informal pointers)

- Convolutional networks: LeCun et al. (1998), “Gradient-based learning applied to document
  recognition”.
- BatchNorm: Ioffe & Szegedy (2015), “Batch Normalization: Accelerating Deep Network Training…”.
- Global average pooling: Lin et al. (2013), “Network In Network”.
-/

@[expose] public section


namespace NN
namespace GraphSpec

open Spec
open Tensor
open NN.Tensor

namespace Primitive

/--
2D convolution on `CHW` tensors (channel-first, no batch).

Inputs:

- parameters `kernel, bias` (in that order),
- input tensor `x : CHW inC inH inW`.

Output:

`CHW outC outH outW` where

`outH = (inH + 2*padding - kH) / stride + 1`  and similarly for `outW`.

This is intentionally close to the underlying Spec/TorchLean op.
PyTorch analogy: `torch.nn.functional.conv2d` on an NCHW tensor, specialized here to CHW.
 -/
def conv2d
    (inC outC kH kW stride padding inH inW : Nat)
    {h_inC : inC ≠ 0} {h_kH : kH ≠ 0} {h_kW : kW ≠ 0} :
    Primitive
      [ NN.Tensor.Shape.OIHW outC inC kH kW, NN.Tensor.Shape.Vec outC ]
      (NN.Tensor.Shape.CHW inC inH inW)
      (NN.Tensor.Shape.CHW outC ((inH + 2 * padding - kH) / stride + 1) ((inW + 2 * padding - kW) /
        stride + 1)) :=
  { name := s!"conv2d(inC={inC},outC={outC},k={kH}x{kW},s={stride},p={padding})"
    specFwd := fun {α} _ctx params x =>
      match params with
      | .cons k (.cons b .nil) =>
          let layer : Spec.Conv2DSpec inC outC kH kW stride padding α h_inC h_kH h_kW :=
            { kernel := k, bias := b }
          Spec.conv2dSpec (α := α) (inH := inH) (inW := inW) layer x
    torchProgram := fun {α} _ctx _deq =>
      fun {m} _ _ =>
        fun k b x =>
          Runtime.Autograd.TorchLean.conv2d (m := m) (α := α)
            (inC := inC) (outC := outC) (kH := kH) (kW := kW)
            (stride := stride) (padding := padding) (inH := inH) (inW := inW)
            (h1 := h_inC) (h2 := h_kH) (h3 := h_kW)
            k b x
    toLayerDefM? := some (fun i =>
      -- Occurrence-indexed seeds, matching the `Primitive.linear` convention.
      ⟨ Runtime.Autograd.TorchLean.NN.conv2d
          (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
          (inH := inH) (inW := inW) (h1 := h_inC) (h2 := h_kH) (h3 := h_kW)
          (seedK := 2 * i) (seedB := 2 * i + 1)
      , by rfl ⟩)
    countsAsLayer := true
  }

/--
MaxPool2D on `CHW` tensors (parameter-free).

Output shapes follow the standard pooling size formulas:

`outH = (inH - kH) / stride + 1` and similarly for `outW`.

PyTorch analogy: `torch.nn.functional.max_pool2d` (with matching `kernel_size` / `stride`).
 -/
def maxPool2d
    (kH kW inH inW inC stride : Nat)
    {h_kH : kH ≠ 0} {h_kW : kW ≠ 0} {hStride : stride ≠ 0} :
    Primitive []
      (NN.Tensor.Shape.CHW inC inH inW)
      (NN.Tensor.Shape.CHW inC ((inH - kH) / stride + 1) ((inW - kW) / stride + 1)) :=
  { name := s!"max_pool2d(k={kH}x{kW},s={stride})"
    specFwd := fun {α} _ctx _params x =>
      let layer : Spec.MaxPool2DSpec kH kW stride h_kH h_kW hStride := {}
      Spec.maxPool2dMultiSpec (layer := layer) x
    torchProgram := fun {α} _ctx _deq =>
      fun {m} _ _ =>
        fun x =>
          Runtime.Autograd.TorchLean.maxPool2d (m := m) (α := α)
            (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
            (h1 := h_kH) (h2 := h_kW) x
    toLayerDefM? := some (fun _i =>
      ⟨ Runtime.Autograd.TorchLean.NN.maxPool2d
          (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
          (h1 := h_kH) (h2 := h_kW)
      , by rfl ⟩)
    countsAsLayer := false
  }

/--
Flatten any tensor to a 1D vector (parameter-free).

Output shape is `.dim (Shape.size s) .scalar`, i.e. a vector whose length is the number of
elements of the input shape.

This is a reshape/view operation (no arithmetic), used to connect convolutional features to a
vector-valued classifier head.

PyTorch analogy: `torch.flatten(x)`.
 -/
def flatten (s : Shape) : Primitive [] s (.dim (Shape.size s) .scalar) :=
  { name := "flatten"
    specFwd := fun {α} _ctx _params x =>
      Spec.Tensor.flattenSpec (α := α) (s := s) x
    torchProgram := fun {α} _ctx _deq =>
      fun {m} _ _ =>
        fun x => Runtime.Autograd.TorchLean.flatten (m := m) (α := α) (s := s) x
    toLayerDefM? := some (fun _i => ⟨Runtime.Autograd.TorchLean.NN.flatten (s := s), by rfl⟩)
    countsAsLayer := false
  }

/--
BatchNorm on `CHW` tensors (channel-first, no batch).

Parameters are `(gamma, beta)` vectors of length `channels`. This models the learnable affine part
of batch normalization.

Note: this op does not carry running mean/variance state inside GraphSpec. If/when we model those,
they will need an explicit state/effect model outside of this pure graph language.

Reference: Ioffe & Szegedy (2015).
 -/
def batchnormChw
    (channels height width : Nat)
    (h_c : channels > 0) (h_h : height > 0) (h_w : width > 0) :
    Primitive
      [ NN.Tensor.Shape.Vec channels, NN.Tensor.Shape.Vec channels ]
      (NN.Tensor.Shape.CHW channels height width)
      (NN.Tensor.Shape.CHW channels height width) :=
  { name := s!"batchnorm_chw(c={channels},h={height},w={width})"
    specFwd := fun {α} _ctx params x =>
      match params with
      | .cons gamma (.cons beta .nil) =>
          Spec.batchNorm2d (α := α)
            (channels := channels) (height := height) (width := width)
            x gamma beta h_c h_h h_w
    torchProgram := fun {α} _ctx _deq =>
      fun {m} _ _ =>
        fun gamma beta x =>
          Runtime.Autograd.TorchLean.batchnormChannelFirst (m := m) (α := α)
            (channels := channels) (height := height) (width := width)
            (h_c := h_c) (h_h := h_h) (h_w := h_w)
            x gamma beta
    toLayerDefM? := some (fun i =>
      ⟨ Runtime.Autograd.TorchLean.NN.batchnormChannelFirst
          (channels := channels) (height := height) (width := width)
          (h_c := h_c) (h_h := h_h) (h_w := h_w)
          (seedGamma := 2 * i) (seedBeta := 2 * i + 1)
      , by rfl ⟩)
    countsAsLayer := true
  }

/--
Global average pooling over spatial dims (`CHW c h w → Vec c`).

This is a common classifier head for CNNs: average each channel over the `h×w` spatial grid.

Reference: Lin et al. (2013), “Network In Network”.
PyTorch analogy: `torch.nn.AdaptiveAvgPool2d((1,1))` followed by flatten.
 -/
def globalAvgPool2dChw
    (c h w : Nat)
    (h_c : c > 0) (h_h : h ≠ 0) (h_w : w ≠ 0) :
    Primitive [] (NN.Tensor.Shape.CHW c h w) (NN.Tensor.Shape.Vec c) :=
  { name := s!"global_avg_pool2d_chw(c={c},h={h},w={w})"
    specFwd := fun {α} _ctx _params x =>
      Spec.globalAvgPool2dFlatSpec (α := α) (inC := c) (inH := h) (inW := w)
        h_h h_w (Spec.GlobalAvgPool2DSpec.mk) x
    torchProgram := fun {α} _ctx _deq =>
      fun {m} _ _ =>
        fun x =>
          Runtime.Autograd.TorchLean.globalAvgPool2dChw (m := m) (α := α)
            (c := c) (h := h) (w := w)
            h_c (Nat.pos_of_ne_zero h_h) (Nat.pos_of_ne_zero h_w) x
    toLayerDefM? := some (fun _i =>
      ⟨ Runtime.Autograd.TorchLean.NN.globalAvgPool2dChw
          (c := c) (h := h) (w := w)
          (h_c_pos := h_c)
          (h_h_pos := Nat.pos_of_ne_zero h_h)
          (h_w_pos := Nat.pos_of_ne_zero h_w)
      , by rfl ⟩)
    countsAsLayer := false
  }

end Primitive

namespace Graph

/-- Graph constructor for `Primitive.conv2d`. -/
def conv2d
    (inC outC kH kW stride padding inH inW : Nat)
    {h_inC : inC ≠ 0} {h_kH : kH ≠ 0} {h_kW : kW ≠ 0} :
    Graph
      [ NN.Tensor.Shape.OIHW outC inC kH kW, NN.Tensor.Shape.Vec outC ]
      (NN.Tensor.Shape.CHW inC inH inW)
      (NN.Tensor.Shape.CHW outC ((inH + 2 * padding - kH) / stride + 1) ((inW + 2 * padding - kW) /
        stride + 1)) :=
  .prim (Primitive.conv2d (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride)
    (padding := padding)
    (inH := inH) (inW := inW) (h_inC := h_inC) (h_kH := h_kH) (h_kW := h_kW))

/-- Graph constructor for `Primitive.max_pool2d`. -/
def maxPool2d
    (kH kW inH inW inC stride : Nat)
    {h_kH : kH ≠ 0} {h_kW : kW ≠ 0} {hStride : stride ≠ 0} :
    Graph [] (NN.Tensor.Shape.CHW inC inH inW)
      (NN.Tensor.Shape.CHW inC ((inH - kH) / stride + 1) ((inW - kW) / stride + 1)) :=
  .prim (Primitive.maxPool2d (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride
    := stride)
    (h_kH := h_kH) (h_kW := h_kW) (hStride := hStride))

/-- Graph constructor for `Primitive.flatten`. -/
def flatten (s : Shape) : Graph [] s (.dim (Shape.size s) .scalar) :=
  .prim (Primitive.flatten s)

/-- Graph constructor for `Primitive.batchnorm_chw`. -/
def batchnormChw
    (channels height width : Nat)
    (h_c : channels > 0) (h_h : height > 0) (h_w : width > 0) :
    Graph [NN.Tensor.Shape.Vec channels, NN.Tensor.Shape.Vec channels]
      (NN.Tensor.Shape.CHW channels height width) (NN.Tensor.Shape.CHW channels height width) :=
  .prim (Primitive.batchnormChw (channels := channels) (height := height) (width := width) h_c h_h
    h_w)

/-- Graph constructor for `Primitive.global_avg_pool2d_chw`. -/
def globalAvgPool2dChw
    (c h w : Nat) (h_c : c > 0) (h_h : h ≠ 0) (h_w : w ≠ 0) :
    Graph [] (NN.Tensor.Shape.CHW c h w) (NN.Tensor.Shape.Vec c) :=
  .prim (Primitive.globalAvgPool2dChw (c := c) (h := h) (w := w) h_c h_h h_w)

end Graph

end GraphSpec
end NN
