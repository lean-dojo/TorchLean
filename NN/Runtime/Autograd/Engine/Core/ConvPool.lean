/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/


module

public import NN.Runtime.Autograd.Engine.Core.Base
public import NN.Tensor.Conversion
public import NN.Spec.Layers.Conv
public import NN.Spec.Layers.Pooling.Spatial

/-!
# Core Tape Convolution and Pooling

This file implements the pure tape nodes for convolution, transposed convolution, and pooling. These
nodes are backend-independent: they record forward values, parents, and backward closures using the
spec-layer definitions before CUDA or typed graph execution enters the picture.
-/

@[expose] public section

namespace Runtime
namespace Autograd

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Tape

/--
N-D convolution for channels-first tensors `(inC, spatial...)` (no batch axis).

The spatial rank and every geometric parameter are encoded by vectors of the same length.
-/
def conv {α : Type} [TorchLean.Storage α] [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {d inC outC : Nat}
  {kernel stride padding : TorchLean.Tensor Nat [d]}
  {inSpatial : TorchLean.Tensor Nat [d]}
  (t : Tape α) (kernelId biasId inputId : Nat) (name : String := "conv") :
  Result (Tape α × Nat) := do
  let k ← requireValue (α:=α) (t:=t)
    (s:=Shape.ofList (outC :: inC :: kernel.to (List Nat))) kernelId
  let b ← requireValue (α:=α) (t:=t) (s:=.dim outC .scalar) biasId
  let x ← requireValue (α:=α) (t:=t)
    (s:=Shape.ofList (inC :: inSpatial.to (List Nat))) inputId
  let layer : Spec.ConvSpec d inC outC kernel stride padding α :=
    { kernel := k, bias := b }
  let y := Spec.convSpec (layer := layer) x
  let outSpatial := Spec.convOutSpatial inSpatial kernel stride padding
  let outSh : Shape := Shape.ofList (outC :: outSpatial.to (List Nat))
  let node : Node α :=
    { name := some name
      value := Spec.SomeTensor.ofTensor y
      requiresGrad :=
        (t.getNode? kernelId).any (·.requiresGrad) ||
        (t.getNode? biasId).any (·.requiresGrad) ||
        (t.getNode? inputId).any (·.requiresGrad)
      parents := #[kernelId, biasId, inputId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := outSh) dLdyAny
        let gradients := Spec.convBackwardSpec (layer := layer) x dLdy
        pure #[
          (kernelId, Spec.SomeTensor.ofTensor gradients.kernelGradient),
          (biasId, Spec.SomeTensor.ofTensor gradients.biasGradient),
          (inputId, Spec.SomeTensor.ofTensor gradients.inputGradient)
        ]
    }
  pure (t.addNode node)

/--
N-D transpose convolution for channels-first tensors `(inC, spatial...)` (no batch axis).

Kernel layout matches the spec/PyTorch convention `(inC, outC, kernel[0], ..., kernel[d-1])`.

PyTorch comparison: `torch.nn.functional.conv_transpose{d}d` specialized to a single sample
(no batch axis).
-/
def convTranspose {α : Type} [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {d inC outC : Nat}
  {kernel stride padding : TorchLean.Tensor Nat [d]}
  {inSpatial : TorchLean.Tensor Nat [d]}
  (t : Tape α) (kernelId biasId inputId : Nat) (name : String := "conv_transpose") :
  Result (Tape α × Nat) := do
  let w ← requireValue (α := α) (t := t)
    (s := Shape.ofList (inC :: outC :: kernel.to (List Nat))) kernelId
  let b ← requireValue (α := α) (t := t) (s := .dim outC .scalar) biasId
  let x ← requireValue (α := α) (t := t)
    (s := Shape.ofList (inC :: inSpatial.to (List Nat))) inputId

  let layer : Spec.ConvTransposeSpec d inC outC kernel stride padding α :=
    { kernel := w, bias := b }
  let y := Spec.convTransposeSpec (layer := layer) x
  let outSpatial := Spec.convTransposeOutSpatial inSpatial kernel stride padding
  let outSh : Shape := Shape.ofList (outC :: outSpatial.to (List Nat))

  let node : Node α :=
    { name := some name
      value := Spec.SomeTensor.ofTensor y
      requiresGrad :=
        (t.getNode? kernelId).any (·.requiresGrad) ||
        (t.getNode? biasId).any (·.requiresGrad) ||
        (t.getNode? inputId).any (·.requiresGrad)
      parents := #[kernelId, biasId, inputId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := outSh) dLdyAny
        let gradients := Spec.convTransposeBackwardSpec (layer := layer) x dLdy
        pure #[
          (kernelId, Spec.SomeTensor.ofTensor gradients.kernelGradient),
          (biasId, Spec.SomeTensor.ofTensor gradients.biasGradient),
          (inputId, Spec.SomeTensor.ofTensor gradients.inputGradient)
        ]
    }
  pure (t.addNode node)

/--
N-D max pooling for channels-first tensors `(C, spatial...)` (no batch axis).

Padding is symmetric per-axis and uses zeros. To model unpadded pooling, pass `padding := 0` on
every axis.
-/
def maxPool {α : Type} [TorchLean.Storage α] [Context α]
  {d C : Nat} {inSpatial kernel stride padding : TorchLean.Tensor Nat [d]}
  (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t)
    (s:=Shape.ofList (C :: inSpatial.to (List Nat))) xId
  if hKernel : (∀ i : Fin d, kernel.getScalar i ≠ 0) then
    if hStride : (∀ i : Fin d, stride.getScalar i ≠ 0) then
      let layer : Spec.MaxPoolSpec d kernel stride padding hKernel hStride := {}
      let y := Spec.maxPoolSpec (layer := layer) x
      let outSpatial := Spec.poolOutSpatialPad inSpatial kernel stride padding
      let outSh : Shape := Shape.ofList (C :: outSpatial.to (List Nat))
      let node : Node α :=
        { name := some "max_pool"
          value := Spec.SomeTensor.ofTensor y
          requiresGrad := (t.getNode? xId).any (·.requiresGrad)
          parents := #[xId]
          backward := fun dLdyAny => do
            let dLdy ← requireGrad (α := α) (τ := outSh) dLdyAny
            let dx :=
              Spec.maxPoolBackwardSpec (layer := layer) (input := x) (gradOutput := dLdy)
            pure #[(xId, Spec.SomeTensor.ofTensor dx)]
        }
      pure (t.addNode node)
    else
      throw "autograd: max_pool requires stride > 0 on every spatial axis"
  else
    throw "autograd: max_pool requires kernel_size > 0 on every spatial axis"

/--
N-D average pooling for channels-first tensors `(C, spatial...)` (no batch axis).

Padding is symmetric per-axis and uses zeros; pooling uses `count_include_pad=true` semantics.
-/
def avgPool {α : Type} [TorchLean.Storage α] [Context α]
  {d C : Nat} {inSpatial kernel stride padding : TorchLean.Tensor Nat [d]}
  (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t)
    (s:=Shape.ofList (C :: inSpatial.to (List Nat))) xId
  if hKernel : (∀ i : Fin d, kernel.getScalar i ≠ 0) then
    if hStride : (∀ i : Fin d, stride.getScalar i ≠ 0) then
      let layer : Spec.AvgPoolSpec d kernel stride padding hKernel hStride := {}
      let y := Spec.avgPoolSpec (layer := layer) x
      let outSpatial := Spec.poolOutSpatialPad inSpatial kernel stride padding
      let outSh : Shape := Shape.ofList (C :: outSpatial.to (List Nat))
      let node : Node α :=
        { name := some "avg_pool"
          value := Spec.SomeTensor.ofTensor y
          requiresGrad := (t.getNode? xId).any (·.requiresGrad)
          parents := #[xId]
          backward := fun dLdyAny => do
            let dLdy ← requireGrad (α := α) (τ := outSh) dLdyAny
            let dx := Spec.avgPoolBackwardSpec (layer := layer) (gradOutput := dLdy)
            pure #[(xId, Spec.SomeTensor.ofTensor dx)]
        }
      pure (t.addNode node)
    else
      throw "autograd: avg_pool requires stride > 0 on every spatial axis"
  else
    throw "autograd: avg_pool requires kernel_size > 0 on every spatial axis"

/--
N-D smooth max pooling (log-sum-exp surrogate) for channels-first tensors `(C, spatial...)`.

The executable tape requires a finite, nonzero `beta`. Finiteness is checked through the scalar
arithmetic contract: finite scalar backends satisfy `beta - beta == 0`, whereas IEEE NaN and
infinity do not. At least one spatial dimension is required, matching the native runtime contract.
-/
def smoothMaxPool {α : Type} [TorchLean.Storage α] [Context α] [DecidableEq α]
  {d C : Nat} {inSpatial kernel stride padding : TorchLean.Tensor Nat [d]}
  (t : Tape α) (xId : Nat) (beta : α) : Result (Tape α × Nat) := do
  if beta == 0 then
    throw "autograd: smooth_max_pool requires finite nonzero beta"
  if hBeta : beta ≠ 0 then
    if !(beta - beta == 0) then
      throw "autograd: smooth_max_pool requires finite nonzero beta"
    if d = 0 then
      throw "autograd: smooth_max_pool requires at least one spatial dimension"
    if hKernel : (∀ i : Fin d, kernel.getScalar i ≠ 0) then
      let x ← requireValue (α:=α) (t:=t)
        (s:=Shape.ofList (C :: inSpatial.to (List Nat))) xId
      if hStride : (∀ i : Fin d, stride.getScalar i ≠ 0) then
        let layer : Spec.MaxPoolSpec d kernel stride padding hKernel hStride := {}
        let y := Spec.smoothMaxPoolSpec (layer := layer) (beta := beta) (hBeta := hBeta) x
        let outSpatial := Spec.poolOutSpatialPad inSpatial kernel stride padding
        let outSh : Shape := Shape.ofList (C :: outSpatial.to (List Nat))
        let node : Node α :=
          { name := some "smooth_max_pool"
            value := Spec.SomeTensor.ofTensor y
            requiresGrad := (t.getNode? xId).any (·.requiresGrad)
            parents := #[xId]
            backward := fun dLdyAny => do
              let dLdy ← requireGrad (α := α) (τ := outSh) dLdyAny
              let dx :=
                Spec.smoothMaxPoolBackwardSpec (layer := layer) (beta := beta) (hBeta := hBeta)
                  (input := x) (gradOutput := dLdy)
              pure #[(xId, Spec.SomeTensor.ofTensor dx)]
          }
        pure (t.addNode node)
      else
        throw "autograd: smooth_max_pool requires stride > 0 on every spatial axis"
    else
      throw "autograd: smooth_max_pool requires kernel_size > 0 on every spatial axis"
  else
    throw "autograd: smooth_max_pool requires finite nonzero beta"
