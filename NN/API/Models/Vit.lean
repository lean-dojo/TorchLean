/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public

/-!
# ViT-Style Model Helpers (API)

This module provides a compact, reusable ViT-style model constructor used by runnable examples.

This is intentionally minimal:
- patch embedding is a strided convolution,
- tokenization is a reshape + axis swap (`N×C×H×W -> N×(H*W)×C`),
- the “transformer” is a single encoder block,
- the head is a simple flatten + linear classifier.

The point is to keep examples readable while still exercising:
Conv2d + tokenization + attention + FFN on both CPU and CUDA eager backends.
-/

@[expose] public section

namespace NN
namespace API

open Spec Tensor

namespace nn
namespace models

/--
Configuration for a small ViT-style classifier.

Shapes:
- input: `N×C×H×W`
- output: `N×outDim`
-/
structure VitConfig where
  batch : Nat
  inC : Nat
  inH : Nat
  inW : Nat
  patchH : Nat
  patchW : Nat
  stride : Nat
  padding : Nat := 0
  dModel : Nat
  outDim : Nat
  numHeads : Nat
  headDim : Nat
  ffnHidden : Nat
deriving Repr

def VitConfig.outH (cfg : VitConfig) : Nat :=
  (cfg.inH + 2 * cfg.padding - cfg.patchH) / cfg.stride + 1

def VitConfig.outW (cfg : VitConfig) : Nat :=
  (cfg.inW + 2 * cfg.padding - cfg.patchW) / cfg.stride + 1

def VitConfig.seqLen (cfg : VitConfig) : Nat :=
  cfg.outH * cfg.outW

def VitConfig.flatDim (cfg : VitConfig) : Nat :=
  -- Keep this in the same “shape-size” form used by `nn.flattenBatch`, so the API-level
  -- constructor typechecks without requiring `simp` reductions on concrete numerals.
  Spec.Shape.size (NN.Tensor.Shape.Mat cfg.seqLen cfg.dModel)

abbrev vitInShape (cfg : VitConfig) : Shape :=
  NN.Tensor.Shape.NCHW cfg.batch cfg.inC cfg.inH cfg.inW

abbrev vitOutShape (cfg : VitConfig) : Shape :=
  NN.Tensor.Shape.Mat cfg.batch cfg.outDim

abbrev vitConvOutShape (cfg : VitConfig) : Shape :=
  NN.Tensor.Shape.NCHW cfg.batch cfg.dModel cfg.outH cfg.outW

abbrev vitTokensShape (cfg : VitConfig) : Shape :=
  shape![cfg.batch, cfg.seqLen, cfg.dModel]

/--
Patch-tokenization adapter: `N×C×H×W -> N×(H*W)×C`.

This is the “low-hanging fruit” to move out of examples: the reshape needs a small size proof.
-/
def nchwToTokens (cfg : VitConfig) : nn.LayerDef (vitConvOutShape cfg) (vitTokensShape cfg) :=
  { paramShapes := []
    initParams := .nil
    paramRequiresGrad := []
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x =>
          (show m (Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) (vitTokensShape cfg)) from do
            let sMid : Shape := shape![cfg.batch, cfg.dModel, cfg.seqLen]
            have hReshape : Shape.size (vitConvOutShape cfg) = Shape.size sMid := by
              simp [vitConvOutShape, NN.Tensor.Shape.NCHW, sMid, _root_.Spec.Shape.size,
                VitConfig.seqLen, VitConfig.outH, VitConfig.outW]
            let xMid ←
              _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
                (s₁ := vitConvOutShape cfg) (s₂ := sMid) x hReshape
            _root_.Runtime.Autograd.Torch.swapAdjacentAtDepth (m := m) (α := α) (s := sMid) 1 xMid)
  }

/--
One-block ViT-style classifier.

This is the constructor used by `torchlean vit`. Keeping it here makes the example a one-liner:
`def mkModel := nn.models.vit1 cfg`.
-/
def vit1 (cfg : VitConfig)
    (h_inC : cfg.inC ≠ 0 := by decide)
    (h_patchH : cfg.patchH ≠ 0 := by decide)
    (h_patchW : cfg.patchW ≠ 0 := by decide)
    (h_seqLen : cfg.seqLen ≠ 0 := by decide)
    (h_dModel : cfg.dModel ≠ 0 := by decide) :
    nn.M (nn.Sequential (vitInShape cfg) (vitOutShape cfg)) :=
  letI : NeZero cfg.inC := ⟨h_inC⟩
  letI : NeZero cfg.patchH := ⟨h_patchH⟩
  letI : NeZero cfg.patchW := ⟨h_patchW⟩
  letI : NeZero cfg.seqLen := ⟨h_seqLen⟩
  letI : NeZero cfg.dModel := ⟨h_dModel⟩
  nn.sequential![
    nn.conv { outC := cfg.dModel, kH := cfg.patchH, kW := cfg.patchW, stride := cfg.stride, padding := cfg.padding },
    nn.lift (nn.of (nchwToTokens cfg)),
    nn.transformerEncoderBlock
      { numHeads := cfg.numHeads
        headDim := cfg.headDim
        ffnHidden := cfg.ffnHidden
        activation := .gelu
        dropout? := none },
    nn.flattenBatch,
    nn.linear cfg.flatDim cfg.outDim (pfx := NN.Tensor.Shape.Vec cfg.batch)
  ]

end models
end nn

end API
end NN
