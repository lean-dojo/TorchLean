/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public

/-!
# ResNet Model Helpers (API)

This module provides a ResNet-style classifier constructor used by runnable examples.

The architecture is small enough to keep the Lean shape arguments readable while still following the
standard ResNet pattern:
- a 3×3 conv stem + BatchNorm + ReLU
- three `resnetBasicBlock`s (one downsampling)
- global average pooling
- linear classifier head

Our goal is to keep the runnable `torchlean resnet` example focused on data loading, optimizer
choice, and reporting. The positivity proofs below are Lean's way of recording that pooling axes
are nonempty; callers should normally just use `nn.models.resnet cfg`.
-/

@[expose] public section

namespace NN
namespace API

open Spec Tensor

namespace nn
namespace models

/-- Configuration for a ResNet-style image classifier. -/
structure ResnetConfig where
  batch : Nat
  inC : Nat
  inH : Nat
  inW : Nat
  stemC : Nat
  numClasses : Nat
deriving Repr

def ResnetConfig.stage2C (cfg : ResnetConfig) : Nat :=
  cfg.stemC * 2

def ResnetConfig.h2 (cfg : ResnetConfig) : Nat :=
  nn.blocks.down2 cfg.inH

def ResnetConfig.w2 (cfg : ResnetConfig) : Nat :=
  nn.blocks.down2 cfg.inW

abbrev resnetInShape (cfg : ResnetConfig) : Shape :=
  NN.Tensor.Shape.NCHW cfg.batch cfg.inC cfg.inH cfg.inW

abbrev resnetOutShape (cfg : ResnetConfig) : Shape :=
  NN.Tensor.Shape.Mat cfg.batch cfg.numClasses

/--
Build a ResNet-style image classifier.

Architecture:
`conv3x3 stem -> batch norm -> ReLU -> basic block -> downsampling basic block -> basic block
-> global average pool -> linear head`.

The explicit proof arguments are optional defaults. For concrete model configs they are solved by
`by decide`; keeping them here hides the proof arguments from runnable examples.
-/
def resnet (cfg : ResnetConfig)
    (h_batch : cfg.batch ≠ 0 := by decide)
    (h_inC : cfg.inC ≠ 0 := by decide)
    (h_inH : cfg.inH ≠ 0 := by decide)
    (h_inW : cfg.inW ≠ 0 := by decide)
    (h_stemC : cfg.stemC ≠ 0 := by decide)
    (h_stage2C : cfg.stage2C ≠ 0 := by decide) :
    nn.M (nn.Sequential (resnetInShape cfg) (resnetOutShape cfg)) := do
  letI : NeZero cfg.batch := ⟨h_batch⟩
  letI : NeZero cfg.inC := ⟨h_inC⟩
  letI : NeZero cfg.inH := ⟨h_inH⟩
  letI : NeZero cfg.inW := ⟨h_inW⟩
  letI : NeZero cfg.stemC := ⟨h_stemC⟩
  letI : NeZero cfg.stage2C := ⟨h_stage2C⟩

  let hb : cfg.batch > 0 := Nat.pos_of_ne_zero (NeZero.ne (n := cfg.batch))
  let hc : cfg.stage2C > 0 := Nat.pos_of_ne_zero (NeZero.ne (n := cfg.stage2C))
  let hh2 : cfg.h2 > 0 := by
    simp [ResnetConfig.h2]
  let hw2 : cfg.w2 > 0 := by
    simp [ResnetConfig.w2]

  let pool : nn.Sequential (NN.Tensor.Shape.Images cfg.batch cfg.stage2C cfg.h2 cfg.w2)
      (NN.Tensor.Shape.Mat cfg.batch cfg.stage2C) :=
    nn.globalAvgPoolNCHW cfg.batch cfg.stage2C cfg.h2 cfg.w2
      (hN := hb) (hC := hc) (hH := hh2) (hW := hw2)

  nn.sequential![
    -- Use the ResNet helper conv so the output shape is definitionally `H×W` (not a conv-formula
    -- expression), while still allocating seeds from the `nn.M` seed stream.
    withSeeds2 (fun seedK seedB =>
      _root_.NN.API.nn.pure.blocks.conv3x3SameImages (n := cfg.batch) (inC := cfg.inC) (outC := cfg.stemC)
        (h := cfg.inH) (w := cfg.inW) (seedK := seedK) (seedB := seedB)
        (kInit := .uniform (-0.1) 0.1)),
    nn.batchNorm,
    nn.relu,
    nn.resnetBasicBlock
      { outC := cfg.stemC, downsample := false, activation := .relu },
    nn.resnetBasicBlock
      { outC := cfg.stage2C, downsample := true, activation := .relu },
    nn.resnetBasicBlock
      { outC := cfg.stage2C, downsample := false, activation := .relu },
    nn.lift pool,
    nn.linear cfg.stage2C cfg.numClasses (pfx := NN.Tensor.Shape.Vec cfg.batch)
  ]

end models
end nn

end API
end NN
