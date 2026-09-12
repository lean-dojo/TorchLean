/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TypedGraph.GraphM.Core
public import NN.Tensor.Conversion
public import NN.Spec.Layers.Conv

/-!
# GraphM Convolution Ops

Rank-generic convolution and transposed-convolution builders.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace TypedGraph
namespace GraphM

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Proofs.Autograd.Algebra
-- Typed context indices come from `NN.Proofs.Autograd.Tape.Util.Idx`, the one place
-- `Idx` and `getIdx` are defined.
open Proofs (Idx getIdx)

/--
N-dimensional convolution (channels-first) on a single sample tensor.

The input shape is `(inC, spatial...)`, the kernel shape is `(outC, inC, kernelSpatial...)`, and the
bias shape is `(outC)`. The output spatial sizes use the PyTorch-style floor-division formula.

The JVP follows bilinearity:
`d(conv(k,b,x)) = conv(k,0,dx) + conv(dk,db,x)`.
-/
def conv {α : Type} {Δ : Type} [TorchLean.Storage α] [Context α]
  {Γ : List Shape} {d inC outC : Nat}
  {kernel stride padding : TorchLean.Tensor Nat [d]}
  {inSpatial : TorchLean.Tensor Nat [d]}
  (w : Var (Shape.ofList (outC :: inC :: Tensor.to kernel (List Nat))))
  (b : Var (.dim outC .scalar))
  (x : Var (Shape.ofList (inC :: Tensor.to inSpatial (List Nat)))) :
  MWith α Δ Γ (Var (Shape.ofList
    (outC :: Tensor.to (Spec.convOutSpatial inSpatial kernel stride padding) (List Nat)))) := do
  let ⟨ss, g⟩ ← get
  let iw ← liftM (mkIdx (_α := α) (Γ := Γ) ss w)
  let ib ← liftM (mkIdx (_α := α) (Γ := Γ) ss b)
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)

  let outSpatial : TorchLean.Tensor Nat [d] :=
    Spec.convOutSpatial inSpatial kernel stride padding
  let outS : Shape := Shape.ofList (outC :: Tensor.to outSpatial (List Nat))
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        let wv := getIdx (α := α) (xs := ctx) iw
        let bv := getIdx (α := α) (xs := ctx) ib
        let xv := getIdx (α := α) (xs := ctx) ix
        let layer : Spec.ConvSpec d inC outC kernel stride padding α :=
          { kernel := wv, bias := bv }
        Spec.convSpec (layer := layer) xv
      jvp := fun ctx dctx _d =>
        let wv := getIdx (α := α) (xs := ctx) iw
        let xv := getIdx (α := α) (xs := ctx) ix
        let dW := getIdx (α := α) (xs := dctx) iw
        let dB := getIdx (α := α) (xs := dctx) ib
        let dX := getIdx (α := α) (xs := dctx) ix
        let zeroBias : Tensor α [outC] := Tensor.full (.dim outC .scalar) (0 : α)
        let layerX : Spec.ConvSpec d inC outC kernel stride padding α :=
          { kernel := wv, bias := zeroBias }
        let layerParams : Spec.ConvSpec d inC outC kernel stride padding α :=
          { kernel := dW, bias := dB }
        addSpec (Spec.convSpec (layer := layerX) dX) (Spec.convSpec (layer := layerParams) xv)
      vjp := fun ctx _d dLdy =>
        let wv := getIdx (α := α) (xs := ctx) iw
        let bv := getIdx (α := α) (xs := ctx) ib
        let xv := getIdx (α := α) (xs := ctx) ix
        let layer : Spec.ConvSpec d inC outC kernel stride padding α :=
          { kernel := wv, bias := bv }
        let gradients := Spec.convBackwardSpec (layer := layer) xv dLdy
        let z0 :=
          TorchLean.TensorPack.add (α := α) (ss := Γ ++ ss)
            (TensorPack.single (α := α) (Γ := Γ ++ ss)
              (s := Shape.ofList (outC :: inC :: Tensor.to kernel (List Nat))) iw
              gradients.kernelGradient)
            (TensorPack.single (α := α) (Γ := Γ ++ ss) (s := .dim outC .scalar) ib
              gradients.biasGradient)
        TorchLean.TensorPack.add (α := α) (ss := Γ ++ ss) z0
          (TensorPack.single (α := α) (Γ := Γ ++ ss)
            (s := Shape.ofList (inC :: Tensor.to inSpatial (List Nat))) ix
            gradients.inputGradient) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
N-D transpose convolution (channels-first) on a single sample tensor (no batch axis).

Conventions:
- input shape is `(inC, spatial...)`,
- kernel shape is `(inC, outC, kernelSpatial...)` (PyTorch layout),
- bias shape is `(outC)`,
- output spatial sizes use:
  `out[a] = (in[a] - 1) * stride[a] - 2*padding[a] + kernel[a]` (with `output_padding = 0`).

PyTorch comparison: `torch.nn.functional.conv_transpose{d}d`, specialized to a single sample.

Forward-mode JVP uses bilinearity:
`d(convTranspose(k,b,x)) = convTranspose(k,0,dx) + convTranspose(dk,db,x)`.
-/
def convTranspose {α : Type} {Δ : Type} [TorchLean.Storage α] [Context α]
  {Γ : List Shape} {d inC outC : Nat}
  {kernel stride padding : TorchLean.Tensor Nat [d]}
  {inSpatial : TorchLean.Tensor Nat [d]}
  (w : Var (Shape.ofList (inC :: outC :: Tensor.to kernel (List Nat))))
  (b : Var (.dim outC .scalar))
  (x : Var (Shape.ofList (inC :: Tensor.to inSpatial (List Nat)))) :
  MWith α Δ Γ (Var (Shape.ofList
    (outC :: Tensor.to
      (Spec.convTransposeOutSpatial inSpatial kernel stride padding) (List Nat)))) := do
  let ⟨ss, g⟩ ← get
  let iw ← liftM (mkIdx (_α := α) (Γ := Γ) ss w)
  let ib ← liftM (mkIdx (_α := α) (Γ := Γ) ss b)
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)

  let outSpatial : TorchLean.Tensor Nat [d] :=
    Spec.convTransposeOutSpatial inSpatial kernel stride padding
  let outS : Shape := Shape.ofList (outC :: Tensor.to outSpatial (List Nat))
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        let wv := getIdx (α := α) (xs := ctx) iw
        let bv := getIdx (α := α) (xs := ctx) ib
        let xv := getIdx (α := α) (xs := ctx) ix
        let layer : Spec.ConvTransposeSpec d inC outC kernel stride padding α :=
          { kernel := wv, bias := bv }
        Spec.convTransposeSpec (layer := layer) xv
      jvp := fun ctx dctx _d =>
        let wv := getIdx (α := α) (xs := ctx) iw
        let xv := getIdx (α := α) (xs := ctx) ix
        let dW := getIdx (α := α) (xs := dctx) iw
        let dB := getIdx (α := α) (xs := dctx) ib
        let dX := getIdx (α := α) (xs := dctx) ix
        let zeroBias : Tensor α [outC] := Tensor.full (.dim outC .scalar) (0 : α)
        let layerX : Spec.ConvTransposeSpec d inC outC kernel stride padding α :=
          { kernel := wv, bias := zeroBias }
        let layerParams : Spec.ConvTransposeSpec d inC outC kernel stride padding α :=
          { kernel := dW, bias := dB }
        addSpec (Spec.convTransposeSpec (layer := layerX) dX)
          (Spec.convTransposeSpec (layer := layerParams) xv)
      vjp := fun ctx _d dLdy =>
        let wv := getIdx (α := α) (xs := ctx) iw
        let bv := getIdx (α := α) (xs := ctx) ib
        let xv := getIdx (α := α) (xs := ctx) ix
        let layer : Spec.ConvTransposeSpec d inC outC kernel stride padding α :=
          { kernel := wv, bias := bv }
        let gradients := Spec.convTransposeBackwardSpec (layer := layer) xv dLdy
        let z0 :=
          TorchLean.TensorPack.add (α := α) (ss := Γ ++ ss)
            (TensorPack.single (α := α) (Γ := Γ ++ ss)
              (s := Shape.ofList (inC :: outC :: Tensor.to kernel (List Nat))) iw
              gradients.kernelGradient)
            (TensorPack.single (α := α) (Γ := Γ ++ ss) (s := .dim outC .scalar) ib
              gradients.biasGradient)
        TorchLean.TensorPack.add (α := α) (ss := Γ ++ ss) z0
          (TensorPack.single (α := α) (Γ := Γ ++ ss)
            (s := Shape.ofList (inC :: Tensor.to inSpatial (List Nat))) ix
            gradients.inputGradient) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

end GraphM
end TypedGraph
end Autograd
end Runtime
