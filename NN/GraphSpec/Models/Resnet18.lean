/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.DAG
import Mathlib.Algebra.Order.Algebra

-- ResNet-18 has a large number of parameters (82 tensors) and our model is fully shape-indexed.
-- Some elaboration steps (mostly around `List.get` + `simp` in parameter indexing) are expensive.
-- We raise the heartbeat limit for this file so `lake build` is robust.
set_option maxHeartbeats 8000000

/-!
# GraphSpec ResNet-18

This file defines a **ResNet-18–style** convolutional network using GraphSpec's general DAG IR,
with BasicBlocks, projection shortcuts, shape-indexed parameters, and TorchLean compilation support.

## Why This Is A DAG Model

Classic ResNet blocks are not purely sequential: the input `x` flows down two paths:

1. a “main” path (Conv → BN → ReLU → Conv → BN),
2. a “skip” path (identity, or a learned projection when shapes change),

and then they are added. In a chain-only representation you either:
- recompute shared values, or
- add special-case “skip” combinators that complicate the core language.

GraphSpec’s DAG IR takes a different approach: it provides a small SSA-like term language
(`Term` + `Args`) that can naturally express sharing. The semantics are:
- `Term.eval`: pure Spec interpreter (math-first).
- `Term.compile`: TorchLean program compilation (executable).

ResNet-18 here is written once, then we get:
- spec-side forward semantics (for proofs / reference),
- a backend-generic TorchLean `Program` (for execution / training).

## Model Scope

This is a “CHW, no batch” variant (C×H×W), matching the rest of the Spec/TorchLean vision layers.
It is faithful to the core ResNet-18 structure:
- stem: 7×7 conv stride 2 padding 3, BN, ReLU, 3×3 maxpool stride 2 padding 1
- stages: [2,2,2,2] BasicBlocks with channel widths [64,128,256,512]
  - first block of stages 2–4 downsamples (stride 2) and uses a 1×1 projection shortcut
- head: global average pool, linear classifier

The state records exactly the metadata needed for parameter allocation:
- Conv bias is included (our Conv2D spec has it), even though many PyTorch ResNets omit it.
- BatchNorm is “train-time” BN with learnable gamma/beta (no running mean/var state).

## Shapes And Type-Level Arithmetic

The main practical challenge is *typing the residual adds*:
- for stride=1 blocks, we need the conv output shape to be exactly `CHW c h w` so we can add it
  to the skip input `x : CHW c h w`.
- for stride=2 blocks, both main-path and skip-path must agree on the downsampled shape.

We solve this by defining a small family of **typed primitives** that cast the “raw” conv/pool
output shapes into a *canonical* downsample formula:

```
down2(h) = (h - 1) / 2 + 1
```

This is the standard stride-2 output formula for kernels with effective receptive field 1 or 3
when you choose padding the usual ResNet way:
- 7×7 s=2 p=3  → outH = (h - 1)/2 + 1
- 3×3 s=2 p=1  → outH = (h - 1)/2 + 1
- 1×1 s=2 p=0  → outH = (h - 1)/2 + 1
- 3×3 maxpool s=2 p=1 → outH = (h - 1)/2 + 1

By enforcing this “canonical” output shape, the residual add becomes definitional/typecheckable
without introducing runtime reshapes.

References / citations:
- He et al. (2016), “Deep Residual Learning for Image Recognition” (ResNet-18, BasicBlock).
- Ioffe & Szegedy (2015), “Batch Normalization…” (BN).
- Lin et al. (2013), “Network In Network” (global average pooling).
-/

@[expose] public section

namespace NN
namespace GraphSpec
namespace Models

open Spec
open Tensor
open NN.Tensor
open NN.GraphSpec.DAG

namespace ResNet18

/-- Convenient local abbreviation for channel-first image tensors (`C × H × W`). -/
abbrev imgShape (c h w : Nat) : Shape := Shape.CHW c h w

/-!
Note on parameter indexing
-------------------------

Inside `model`, the environment is `Γ = params ++ [x]`, so parameters are accessed by index.

We *avoid* clever macros here. In practice, being explicit is faster to debug and (importantly)
more stable under refactors: each use site carries its own proof that the index points at the
expected shape.
-/
/-! ## Canonical stride-2 downsample formula -/

/--
Canonical stride-2 output-size formula used throughout this file.

We write it once and reuse it for the stem, the downsampling residual blocks, and the max-pool so
that all of those paths literally agree on the same type-level height/width expression.
-/
abbrev down2 (h : Nat) : Nat := (h - 1) / 2 + 1

/-- `down2` is always positive. -/
lemma down2_pos (h : Nat) : down2 h > 0 := by
  -- `(h - 1) / 2` is a Nat, so adding 1 is always positive.
  simp [down2]

/-! ## Small typed primitives for ResNet typing -/

/-- Stride-1, padding-1 3×3 conv preserves the spatial size for `h > 0`. -/
lemma conv3_same_out_eq {h : Nat} (hh : h > 0) : ((h + 2 * 1 - 3) / 1 + 1) = h := by
  cases h with
  | zero => cases (Nat.lt_irrefl 0 hh)
  | succ _n => simp

/--
Typed 3×3 convolution (stride 1, padding 1) whose output is cast to the canonical `imgShape c h w`.
-/
def conv3x3Same
    (c h w : Nat)
    (h_c : c ≠ 0) (hh : h > 0) (hw : w > 0) :
    PrimOp
      [ Shape.OIHW c c 3 3, Shape.Vec c, imgShape c h w ]
      (imgShape c h w) :=
  have hShape :
      imgShape c ((h + 2 * 1 - 3) / 1 + 1) ((w + 2 * 1 - 3) / 1 + 1) = imgShape c h w := by
    have hH : h - 1 + 1 = h := by
      cases h with
      | zero => cases (Nat.lt_irrefl 0 hh)
      | succ n => simp
    have hW : w - 1 + 1 = w := by
      cases w with
      | zero => cases (Nat.lt_irrefl 0 hw)
      | succ n => simp
    simpa [imgShape] using And.intro hH hW
  { name := "resnet.conv3x3_same"
    specFwd := fun {α} _ctx xs =>
      match xs with
      | .cons k (.cons b (.cons x .nil)) =>
          let layer : Spec.Conv2DSpec c c 3 3 1 1 α h_c (by decide) (by decide) :=
            { kernel := k, bias := b }
          let y := Spec.conv2dSpec (α := α) (inH := h) (inW := w) layer x
          Tensor.castShape y hShape
    torchProgram := fun {α} _ctx _deq =>
      fun {m} _ _ =>
        fun k b x =>
          (do
            let y ← Runtime.Autograd.TorchLean.conv2d (m := m) (α := α)
              (inC := c) (outC := c) (kH := 3) (kW := 3)
              (stride := 1) (padding := 1) (inH := h) (inW := w)
              (h1 := h_c) (h2 := by decide) (h3 := by decide)
              k b x
            pure <|
              Eq.ndrec
                (motive := fun s => Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) s)
                y
                hShape
          : m (Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) (imgShape c h w)))
  }

/-- Typed 7×7 stem convolution (stride 2, padding 3), cast to the canonical `down2` spatial sizes.
  -/
def conv7x7Down
    (inC outC h w : Nat)
    (h_inC : inC ≠ 0) (_h_outC : outC ≠ 0) :
    PrimOp
      [ Shape.OIHW outC inC 7 7, Shape.Vec outC, imgShape inC h w ]
      (imgShape outC (down2 h) (down2 w)) :=
  have hShape :
      imgShape outC ((h + 2 * 3 - 7) / 2 + 1) ((w + 2 * 3 - 7) / 2 + 1) =
        imgShape outC (down2 h) (down2 w) := by
    simp [imgShape, down2, two_mul, Nat.add_comm]
  { name := "resnet.conv7x7_s2"
    specFwd := fun {α} _ctx xs =>
      match xs with
      | .cons k (.cons b (.cons x .nil)) =>
          let layer : Spec.Conv2DSpec inC outC 7 7 2 3 α h_inC (by decide) (by decide) :=
            { kernel := k, bias := b }
          let y := Spec.conv2dSpec (α := α) (inH := h) (inW := w) layer x
          Tensor.castShape y hShape
    torchProgram := fun {α} _ctx _deq =>
      fun {m} _ _ =>
        fun k b x =>
          (do
            let y ← Runtime.Autograd.TorchLean.conv2d (m := m) (α := α)
              (inC := inC) (outC := outC) (kH := 7) (kW := 7)
              (stride := 2) (padding := 3) (inH := h) (inW := w)
              (h1 := h_inC) (h2 := by decide) (h3 := by decide)
              k b x
            pure <|
              Eq.ndrec
                (motive := fun s => Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) s)
                y
                hShape
          : m (Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) (imgShape outC (down2 h) (down2
            w))))
  }

/-- Typed 3×3 convolution (stride 2, padding 1), cast to the canonical `down2` spatial sizes. -/
def conv3x3Down
    (inC outC h w : Nat)
    (h_inC : inC ≠ 0) (_h_outC : outC ≠ 0) :
    PrimOp
      [ Shape.OIHW outC inC 3 3, Shape.Vec outC, imgShape inC h w ]
      (imgShape outC (down2 h) (down2 w)) :=
  have hShape :
      imgShape outC ((h + 2 * 1 - 3) / 2 + 1) ((w + 2 * 1 - 3) / 2 + 1) =
        imgShape outC (down2 h) (down2 w) := by
    simp [imgShape, down2, Nat.add_comm]
  { name := "resnet.conv3x3_s2"
    specFwd := fun {α} _ctx xs =>
      match xs with
      | .cons k (.cons b (.cons x .nil)) =>
          let layer : Spec.Conv2DSpec inC outC 3 3 2 1 α h_inC (by decide) (by decide) :=
            { kernel := k, bias := b }
          let y := Spec.conv2dSpec (α := α) (inH := h) (inW := w) layer x
          Tensor.castShape y hShape
    torchProgram := fun {α} _ctx _deq =>
      fun {m} _ _ =>
        fun k b x =>
          (do
            let y ← Runtime.Autograd.TorchLean.conv2d (m := m) (α := α)
              (inC := inC) (outC := outC) (kH := 3) (kW := 3)
              (stride := 2) (padding := 1) (inH := h) (inW := w)
              (h1 := h_inC) (h2 := by decide) (h3 := by decide)
              k b x
            pure <|
              Eq.ndrec
                (motive := fun s => Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) s)
                y
                hShape
          : m (Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) (imgShape outC (down2 h) (down2
            w))))
  }

/-- Typed 1×1 projection convolution (stride 2), cast to the canonical `down2` spatial sizes. -/
def conv1x1Down
    (inC outC h w : Nat)
    (h_inC : inC ≠ 0) (_h_outC : outC ≠ 0) :
    PrimOp
      [ Shape.OIHW outC inC 1 1, Shape.Vec outC, imgShape inC h w ]
      (imgShape outC (down2 h) (down2 w)) :=
  have hShape :
      imgShape outC ((h + 2 * 0 - 1) / 2 + 1) ((w + 2 * 0 - 1) / 2 + 1) =
        imgShape outC (down2 h) (down2 w) := by
    simp [imgShape, down2]
  { name := "resnet.conv1x1_s2"
    specFwd := fun {α} _ctx xs =>
      match xs with
      | .cons k (.cons b (.cons x .nil)) =>
          let layer : Spec.Conv2DSpec inC outC 1 1 2 0 α h_inC (by decide) (by decide) :=
            { kernel := k, bias := b }
          let y := Spec.conv2dSpec (α := α) (inH := h) (inW := w) layer x
          Tensor.castShape y hShape
    torchProgram := fun {α} _ctx _deq =>
      fun {m} _ _ =>
        fun k b x =>
          (do
            let y ← Runtime.Autograd.TorchLean.conv2d (m := m) (α := α)
              (inC := inC) (outC := outC) (kH := 1) (kW := 1)
              (stride := 2) (padding := 0) (inH := h) (inW := w)
              (h1 := h_inC) (h2 := by decide) (h3 := by decide)
              k b x
            pure <|
              Eq.ndrec
                (motive := fun s => Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) s)
                y
                hShape
          : m (Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) (imgShape outC (down2 h) (down2
            w))))
  }

/-- Typed 3×3 max-pool (stride 2, padding 1), cast to the canonical `down2` spatial sizes. -/
def maxpool3x3Down
    (c h w : Nat) :
    PrimOp [imgShape c h w] (imgShape c (down2 h) (down2 w)) :=
  have hShape :
      imgShape c ((h + 2 * 1 - 3) / 2 + 1) ((w + 2 * 1 - 3) / 2 + 1) =
        imgShape c (down2 h) (down2 w) := by
    simp [imgShape, down2, Nat.add_comm]
  { name := "resnet.maxpool3x3_s2"
    specFwd := fun {α} _ctx xs =>
      match xs with
      | .cons x .nil =>
          let layer : Spec.MaxPool2DSpec 3 3 2 (by decide) (by decide) (by decide) := {}
          let y :=
            Spec.maxPool2dMultiSpecPad (α := α)
              (kH := 3) (kW := 3) (inH := h) (inW := w) (inC := c) (stride := 2) (padding := 1)
              (h1 := by decide) (h2 := by decide) (hStride := by decide)
              (layer := layer) x
          Tensor.castShape y hShape
    torchProgram := fun {α} _ctx _deq =>
      fun {m} _ _ =>
        fun x =>
          (do
            let y ← Runtime.Autograd.TorchLean.maxPool2dPad (m := m) (α := α)
              (kH := 3) (kW := 3) (inH := h) (inW := w) (inC := c)
              (stride := 2) (padding := 1) (h1 := by decide) (h2 := by decide)
              x
            pure <|
              Eq.ndrec
                (motive := fun s => Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) s)
                y
                hShape
          : m (Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) (imgShape c (down2 h) (down2 w))))
  }

/-! ## Parameter layout -/

/-- Parameter layout for a convolution layer: kernel first, then bias. -/
abbrev convParams (outC inC kH kW : Nat) : List Shape :=
  [ Shape.OIHW outC inC kH kW, Shape.Vec outC ]

/-- Parameter layout for affine batch norm: `(gamma, beta)`. -/
abbrev bnParams (c : Nat) : List Shape :=
  [ Shape.Vec c, Shape.Vec c ]

/--
Parameter ABI for a single BasicBlock.

When `downsample = true`, this includes the projection shortcut `(1×1 conv + BN)` parameters after
the two main-path conv/BN pairs.
-/
abbrev basicBlockParams (inC outC : Nat) (downsample : Bool) : List Shape :=
  convParams outC inC 3 3 ++ bnParams outC ++
  convParams outC outC 3 3 ++ bnParams outC ++
  (if downsample then convParams outC inC 1 1 ++ bnParams outC else [])

/--
Parameter ABI for one ResNet-18 stage.

A stage is two BasicBlocks. The first block downsamples exactly when `inC ≠ outC`; the second
always keeps the same number of channels.
-/
abbrev stageParams (inC outC : Nat) : List Shape :=
  basicBlockParams inC outC (decide (inC ≠ outC)) ++ basicBlockParams outC outC false

/--
Full parameter ABI for the GraphSpec ResNet-18 model.

This list is deliberately written in a flat, explicit order. The model body indexes parameters by
closed numerals, and the flat ABI keeps those index proofs predictable for `simp` and easy to audit.
-/
abbrev params (inC numClasses : Nat) : List Shape :=
  -- Layout (0-based) matches the comment in `model`:
  -- stem (0..3), then stages 1..4, then head (80..81).
  [
    -- stem: conv7×7 (k,b) + BN (gamma,beta)
    Shape.OIHW 64 inC 7 7, Shape.Vec 64, Shape.Vec 64, Shape.Vec 64,

    -- stage1: 2 blocks, 64→64
    Shape.OIHW 64 64 3 3, Shape.Vec 64, Shape.Vec 64, Shape.Vec 64,
    Shape.OIHW 64 64 3 3, Shape.Vec 64, Shape.Vec 64, Shape.Vec 64,
    Shape.OIHW 64 64 3 3, Shape.Vec 64, Shape.Vec 64, Shape.Vec 64,
    Shape.OIHW 64 64 3 3, Shape.Vec 64, Shape.Vec 64, Shape.Vec 64,

    -- stage2: 64→128 (downsample+projection), then 128→128
    Shape.OIHW 128 64 3 3, Shape.Vec 128, Shape.Vec 128, Shape.Vec 128,
    Shape.OIHW 128 128 3 3, Shape.Vec 128, Shape.Vec 128, Shape.Vec 128,
    Shape.OIHW 128 64 1 1, Shape.Vec 128, Shape.Vec 128, Shape.Vec 128,
    Shape.OIHW 128 128 3 3, Shape.Vec 128, Shape.Vec 128, Shape.Vec 128,
    Shape.OIHW 128 128 3 3, Shape.Vec 128, Shape.Vec 128, Shape.Vec 128,

    -- stage3: 128→256 (downsample+projection), then 256→256
    Shape.OIHW 256 128 3 3, Shape.Vec 256, Shape.Vec 256, Shape.Vec 256,
    Shape.OIHW 256 256 3 3, Shape.Vec 256, Shape.Vec 256, Shape.Vec 256,
    Shape.OIHW 256 128 1 1, Shape.Vec 256, Shape.Vec 256, Shape.Vec 256,
    Shape.OIHW 256 256 3 3, Shape.Vec 256, Shape.Vec 256, Shape.Vec 256,
    Shape.OIHW 256 256 3 3, Shape.Vec 256, Shape.Vec 256, Shape.Vec 256,

    -- stage4: 256→512 (downsample+projection), then 512→512
    Shape.OIHW 512 256 3 3, Shape.Vec 512, Shape.Vec 512, Shape.Vec 512,
    Shape.OIHW 512 512 3 3, Shape.Vec 512, Shape.Vec 512, Shape.Vec 512,
    Shape.OIHW 512 256 1 1, Shape.Vec 512, Shape.Vec 512, Shape.Vec 512,
    Shape.OIHW 512 512 3 3, Shape.Vec 512, Shape.Vec 512, Shape.Vec 512,
    Shape.OIHW 512 512 3 3, Shape.Vec 512, Shape.Vec 512, Shape.Vec 512,

    -- head: linear 512→numClasses
    Shape.Mat numClasses 512, Shape.Vec numClasses
  ]

/--
The ResNet-18 GraphSpec ABI contains exactly 82 parameter tensors.

This is a small but useful structural theorem: the executable wrapper, deterministic initializer,
and DAG body all rely on the same flat ABI. Keeping the count as a named theorem makes accidental
parameter-layout edits visible during review instead of hiding them inside a local `simp`.
-/
@[simp] theorem params_length (inC numClasses : Nat) :
    (params inC numClasses).length = 82 := by
  simp [params]

/-! ## Deterministic initialization -/

/-- Deterministically initialize a convolution kernel (uniform) and bias (zeros). -/
def initConv (outC inC kH kW seedK seedB : Nat) :
    Tensor Float (Shape.OIHW outC inC kH kW) × Tensor Float (Shape.Vec outC) :=
  let k : Tensor Float (Shape.OIHW outC inC kH kW) :=
    Runtime.Autograd.Torch.Init.tensor (s := Shape.OIHW outC inC kH kW)
      (sch := .uniform (-0.1) 0.1) (seed := seedK)
  let b : Tensor Float (Shape.Vec outC) :=
    Runtime.Autograd.Torch.Init.tensor (s := Shape.Vec outC) (sch := .zeros) (seed := seedB)
  (k, b)

/-- Deterministically initialize BatchNorm parameters `(gamma, beta)` as `(ones, zeros)`. -/
def initBN (c seedGamma seedBeta : Nat) :
    Tensor Float (Shape.Vec c) × Tensor Float (Shape.Vec c) :=
  let gamma : Tensor Float (Shape.Vec c) :=
    Runtime.Autograd.Torch.Init.tensor (s := Shape.Vec c) (sch := .ones) (seed := seedGamma)
  let beta : Tensor Float (Shape.Vec c) :=
    Runtime.Autograd.Torch.Init.tensor (s := Shape.Vec c) (sch := .zeros) (seed := seedBeta)
  (gamma, beta)

/-- Deterministically initialize linear weights (uniform) and bias (zeros). -/
def initLinear (inDim outDim seedW seedB : Nat) :
    Tensor Float (Shape.Mat outDim inDim) × Tensor Float (Shape.Vec outDim) :=
  let w : Tensor Float (Shape.Mat outDim inDim) :=
    Runtime.Autograd.Torch.Init.tensor (s := Shape.Mat outDim inDim)
      (sch := .uniform (-0.1) 0.1) (seed := seedW)
  let b : Tensor Float (Shape.Vec outDim) :=
    Runtime.Autograd.Torch.Init.tensor (s := Shape.Vec outDim) (sch := .zeros) (seed := seedB)
  (w, b)

/-! ## Model -/

/--
GraphSpec ResNet-18 model.

This is the public entrypoint for the DAG-authored ResNet in this directory. It packages together:

- the full typed parameter ABI (`params inC numClasses`),
- deterministic initialization for all 82 parameter tensors,
- and a DAG body whose pure semantics can be interpreted via `DAG.Model.specFwd` and compiled via
  `DAG.Model.torchProgram`.

The model is channel-first (`CHW`) and batch-free, matching the rest of TorchLean's vision-side
Spec and runtime layers.
-/
def model
    (inC h w numClasses : Nat)
    (h_inC : inC > 0) (_h_h : h > 0) (_h_w : w > 0) (_h_cls : numClasses > 0) :
    Model (ps := params inC numClasses) (ins := [imgShape inC h w]) (τ := Shape.Vec numClasses) :=
  let ps : List Shape := params inC numClasses

  -- Derived spatial sizes along the stem.
  let h1 := down2 h
  let w1 := down2 w
  let h2 := down2 h1
  let w2 := down2 w1

  have h_h1 : h1 > 0 := down2_pos h
  have h_w1 : w1 > 0 := down2_pos w
  have h_h2 : h2 > 0 := down2_pos h1
  have h_w2 : w2 > 0 := down2_pos w1

  -- Stage spatial sizes after successive downsampling blocks (stages 2–4).
  let h3 := down2 h2; let w3 := down2 w2
  let h4 := down2 h3; let w4 := down2 w3
  let h5 := down2 h4; let w5 := down2 w4

  have h_h3 : h3 > 0 := down2_pos h2
  have h_w3 : w3 > 0 := down2_pos w2
  have h_h4 : h4 > 0 := down2_pos h3
  have h_w4 : w4 > 0 := down2_pos w3
  have h_h5 : h5 > 0 := down2_pos h4
  have h_w5 : w5 > 0 := down2_pos w4

  -- Deterministic parameter initialization (seeded; layout matches `params`).
  --
  -- Index map (0-based):
  -- - 0..3:   stem conv7 (k,b) + bn (gamma,beta)
  -- - 4..19:  stage1 (2 blocks, no downsample)
  -- - 20..39: stage2 (block0 downsample, block1 normal)
  -- - 40..59: stage3 (block0 downsample, block1 normal)
  -- - 60..79: stage4 (block0 downsample, block1 normal)
  -- - 80..81: head linear (w,b)
  --
  -- Total: 82 tensors.
  let (k0, b0) := initConv 64 inC 7 7 0 1
  let (g0, bt0) := initBN 64 2 3

  -- stage1 (64→64, no projection)
  let (k4, b5) := initConv 64 64 3 3 4 5
  let (g6, bt7) := initBN 64 6 7
  let (k8, b9) := initConv 64 64 3 3 8 9
  let (g10, bt11) := initBN 64 10 11
  let (k12, b13) := initConv 64 64 3 3 12 13
  let (g14, bt15) := initBN 64 14 15
  let (k16, b17) := initConv 64 64 3 3 16 17
  let (g18, bt19) := initBN 64 18 19

  -- stage2 (64→128, projection in first block)
  let (k20, b21) := initConv 128 64 3 3 20 21
  let (g22, bt23) := initBN 128 22 23
  let (k24, b25) := initConv 128 128 3 3 24 25
  let (g26, bt27) := initBN 128 26 27
  let (kp28, bp29) := initConv 128 64 1 1 28 29
  let (gp30, btp31) := initBN 128 30 31
  let (k32, b33) := initConv 128 128 3 3 32 33
  let (g34, bt35) := initBN 128 34 35
  let (k36, b37) := initConv 128 128 3 3 36 37
  let (g38, bt39) := initBN 128 38 39

  -- stage3 (128→256, projection in first block)
  let (k40, b41) := initConv 256 128 3 3 40 41
  let (g42, bt43) := initBN 256 42 43
  let (k44, b45) := initConv 256 256 3 3 44 45
  let (g46, bt47) := initBN 256 46 47
  let (kp48, bp49) := initConv 256 128 1 1 48 49
  let (gp50, btp51) := initBN 256 50 51
  let (k52, b53) := initConv 256 256 3 3 52 53
  let (g54, bt55) := initBN 256 54 55
  let (k56, b57) := initConv 256 256 3 3 56 57
  let (g58, bt59) := initBN 256 58 59

  -- stage4 (256→512, projection in first block)
  let (k60, b61) := initConv 512 256 3 3 60 61
  let (g62, bt63) := initBN 512 62 63
  let (k64, b65) := initConv 512 512 3 3 64 65
  let (g66, bt67) := initBN 512 66 67
  let (kp68, bp69) := initConv 512 256 1 1 68 69
  let (gp70, btp71) := initBN 512 70 71
  let (k72, b73) := initConv 512 512 3 3 72 73
  let (g74, bt75) := initBN 512 74 75
  let (k76, b77) := initConv 512 512 3 3 76 77
  let (g78, bt79) := initBN 512 78 79

  -- head
  let (w80, b81) := initLinear 512 numClasses 80 81

  let initParams : Runtime.Autograd.Torch.TList Float ps := by
    let append :
        {ss₁ ss₂ : List Shape} →
          Runtime.Autograd.Torch.TList Float ss₁ →
            Runtime.Autograd.Torch.TList Float ss₂ →
              Runtime.Autograd.Torch.TList Float (ss₁ ++ ss₂) :=
      Runtime.Autograd.Torch.Proofs.Autograd.Algebra.TList.append (α := Float)

    let stem : Runtime.Autograd.Torch.TList Float (convParams 64 inC 7 7 ++ bnParams 64) :=
      by
        simpa [convParams, bnParams, List.append_assoc] using
          append (Runtime.Autograd.Torch.tlist2 k0 b0) (Runtime.Autograd.Torch.tlist2 g0 bt0)

    let blk (outC inC : Nat)
        (k1 : Tensor Float (Shape.OIHW outC inC 3 3)) (b1 : Tensor Float (Shape.Vec outC))
        (g1 : Tensor Float (Shape.Vec outC)) (bt1 : Tensor Float (Shape.Vec outC))
        (k2 : Tensor Float (Shape.OIHW outC outC 3 3)) (b2 : Tensor Float (Shape.Vec outC))
        (g2 : Tensor Float (Shape.Vec outC)) (bt2 : Tensor Float (Shape.Vec outC)) :
        Runtime.Autograd.Torch.TList Float (basicBlockParams inC outC false) :=
      by
        simpa [basicBlockParams, convParams, bnParams, List.append_assoc] using
          append (Runtime.Autograd.Torch.tlist4 k1 b1 g1 bt1) (Runtime.Autograd.Torch.tlist4 k2 b2
            g2 bt2)

    let blkDown (outC inC : Nat)
        (k1 : Tensor Float (Shape.OIHW outC inC 3 3)) (b1 : Tensor Float (Shape.Vec outC))
        (g1 : Tensor Float (Shape.Vec outC)) (bt1 : Tensor Float (Shape.Vec outC))
        (k2 : Tensor Float (Shape.OIHW outC outC 3 3)) (b2 : Tensor Float (Shape.Vec outC))
        (g2 : Tensor Float (Shape.Vec outC)) (bt2 : Tensor Float (Shape.Vec outC))
        (kp : Tensor Float (Shape.OIHW outC inC 1 1)) (bp : Tensor Float (Shape.Vec outC))
        (gp : Tensor Float (Shape.Vec outC)) (btp : Tensor Float (Shape.Vec outC)) :
        Runtime.Autograd.Torch.TList Float (basicBlockParams inC outC true) :=
      by
        simpa [basicBlockParams, convParams, bnParams, List.append_assoc] using
          append
            (append (Runtime.Autograd.Torch.tlist4 k1 b1 g1 bt1) (Runtime.Autograd.Torch.tlist4 k2
              b2 g2 bt2))
            (Runtime.Autograd.Torch.tlist4 kp bp gp btp)

    let stage1 : Runtime.Autograd.Torch.TList Float (stageParams 64 64) :=
      append
        (blk 64 64 k4 b5 g6 bt7 k8 b9 g10 bt11)
        (blk 64 64 k12 b13 g14 bt15 k16 b17 g18 bt19)

    let stage2 : Runtime.Autograd.Torch.TList Float (stageParams 64 128) :=
      append
        (blkDown 128 64 k20 b21 g22 bt23 k24 b25 g26 bt27 kp28 bp29 gp30 btp31)
        (blk 128 128 k32 b33 g34 bt35 k36 b37 g38 bt39)

    let stage3 : Runtime.Autograd.Torch.TList Float (stageParams 128 256) :=
      append
        (blkDown 256 128 k40 b41 g42 bt43 k44 b45 g46 bt47 kp48 bp49 gp50 btp51)
        (blk 256 256 k52 b53 g54 bt55 k56 b57 g58 bt59)

    let stage4 : Runtime.Autograd.Torch.TList Float (stageParams 256 512) :=
      append
        (blkDown 512 256 k60 b61 g62 bt63 k64 b65 g66 bt67 kp68 bp69 gp70 btp71)
        (blk 512 512 k72 b73 g74 bt75 k76 b77 g78 bt79)

    let head : Runtime.Autograd.Torch.TList Float [Shape.Mat numClasses 512, Shape.Vec numClasses]
      :=
      Runtime.Autograd.Torch.tlist2 w80 b81

    simpa [ps, params, stageParams, basicBlockParams, convParams, bnParams, List.append_assoc] using
      append stem (append stage1 (append stage2 (append stage3 (append stage4 head))))

  -- Environment Γ = params ++ [x]
  let Γ : List Shape := ps ++ [imgShape inC h w]
  let x : Term Γ (imgShape inC h w) :=
    Term.var (Γ := Γ) ⟨ps.length, by simp [Γ]⟩

  -- Parameter accessor.
  --
  -- The environment is `Γ = ps ++ [x]` (parameters first, then the input). So for any `i` strictly
  -- less than `ps.length`, the index `i` points at a parameter, not at the input.
  --
  -- We intentionally return the *computed* shape `Γ.get …` rather than asking callers to provide a
  -- separate “expected shape” proof. Since `ps` is a flat list literal, `Γ.get` at numeral indices
  -- reduces quickly and unifies with the shapes required by each primitive.
  --
  -- Implementation note: we index parameters with a *closed* bound `i < 82` (ResNet-18 has 82
  -- parameter tensors in this encoding). This avoids tactics like `decide` getting
  -- stuck on goals that mention free variables (`inC`, `numClasses`) even though the bound is
  -- definitionally constant.
  have hps_len : ps.length = 82 := by simp [ps]
  have hps_lt_Γ : ps.length < Γ.length := by simp [Γ]
  let p (i : Nat) (hi : i < 82) :
      Term Γ (Γ.get ⟨i, Nat.lt_trans (by simpa [hps_len] using hi) hps_lt_Γ⟩) :=
    Term.var (Γ := Γ) ⟨i, Nat.lt_trans (by simpa [hps_len] using hi) hps_lt_Γ⟩

  -- Stem
  have h64 : 64 > 0 := by decide
  let stemConv : Term Γ (imgShape 64 h1 w1) :=
    Term.op (Γ := Γ)
      (conv7x7Down (inC := inC) (outC := 64) (h := h) (w := w)
        (h_inC := Nat.ne_of_gt h_inC) (_h_outC := by decide))
      (Args.cons
        (p 0 (by decide))
        (Args.cons
          (p 1 (by decide))
          (Args.cons x (Args.nil))))

  let stemBN : Term Γ (imgShape 64 h1 w1) :=
    Term.op (Γ := Γ)
      (PrimOp.batchnormChw (channels := 64) (height := h1) (width := w1) h64 h_h1 h_w1)
      (Args.cons
        (p 2 (by decide))
        (Args.cons
          (p 3 (by decide))
          (Args.cons stemConv (Args.nil))))

  let stemReLU : Term Γ (imgShape 64 h1 w1) :=
    Term.op (Γ := Γ) (PrimOp.relu (s := imgShape 64 h1 w1)) (Args.cons stemBN (Args.nil))

  let stemPool : Term Γ (imgShape 64 h2 w2) :=
    Term.op (Γ := Γ) (maxpool3x3Down (c := 64) (h := h1) (w := w1)) (Args.cons stemReLU (Args.nil))

  -- BasicBlock (stride 1, identity skip), used in stage1 and as the second block in later stages.
  let basicBlockSame
      (c h w : Nat) (h_c_pos : c > 0) (hh : h > 0) (hw : w > 0)
      (k1 : Term Γ (Shape.OIHW c c 3 3)) (b1 : Term Γ (Shape.Vec c))
      (g1 : Term Γ (Shape.Vec c)) (bt1 : Term Γ (Shape.Vec c))
      (k2 : Term Γ (Shape.OIHW c c 3 3)) (b2 : Term Γ (Shape.Vec c))
      (g2 : Term Γ (Shape.Vec c)) (bt2 : Term Γ (Shape.Vec c))
      (xIn : Term Γ (imgShape c h w)) :
      Term Γ (imgShape c h w) :=
    -- Parameters are supplied explicitly by index at call sites; this helper just threads them
    -- through the block.
    let conv1 :=
      Term.op (Γ := Γ)
        (conv3x3Same (c := c) (h := h) (w := w) (h_c := Nat.ne_of_gt h_c_pos) (hh := hh) (hw :=
          hw))
        (Args.cons k1
          (Args.cons b1
            (Args.cons xIn (Args.nil))))
    let bn1 :=
      Term.op (Γ := Γ)
        (PrimOp.batchnormChw (channels := c) (height := h) (width := w) h_c_pos hh hw)
        (Args.cons g1
          (Args.cons bt1
            (Args.cons conv1 (Args.nil))))
    let relu1 := Term.op (Γ := Γ) (PrimOp.relu (s := imgShape c h w)) (Args.cons bn1 (Args.nil))
    let conv2 :=
      Term.op (Γ := Γ)
        (conv3x3Same (c := c) (h := h) (w := w) (h_c := Nat.ne_of_gt h_c_pos) (hh := hh) (hw :=
          hw))
        (Args.cons k2
          (Args.cons b2
            (Args.cons relu1 (Args.nil))))
    let bn2 :=
      Term.op (Γ := Γ)
        (PrimOp.batchnormChw (channels := c) (height := h) (width := w) h_c_pos hh hw)
        (Args.cons g2
          (Args.cons bt2
            (Args.cons conv2 (Args.nil))))
    let sum := Term.op (Γ := Γ) (PrimOp.add (s := imgShape c h w)) (Args.cons bn2 (Args.cons xIn
      (Args.nil)))
    Term.op (Γ := Γ) (PrimOp.relu (s := imgShape c h w)) (Args.cons sum (Args.nil))

  -- BasicBlock (stride 2, projection skip), used as first block in stages 2–4.
  let basicBlockDown
      (inC outC h w : Nat)
      (h_inC_pos : inC > 0) (h_outC_pos : outC > 0) (hh : h > 0) (hw : w > 0)
      (k1 : Term Γ (Shape.OIHW outC inC 3 3)) (b1 : Term Γ (Shape.Vec outC))
      (g1 : Term Γ (Shape.Vec outC)) (bt1 : Term Γ (Shape.Vec outC))
      (k2 : Term Γ (Shape.OIHW outC outC 3 3)) (b2 : Term Γ (Shape.Vec outC))
      (g2 : Term Γ (Shape.Vec outC)) (bt2 : Term Γ (Shape.Vec outC))
      (kp : Term Γ (Shape.OIHW outC inC 1 1)) (bp : Term Γ (Shape.Vec outC))
      (gp : Term Γ (Shape.Vec outC)) (btp : Term Γ (Shape.Vec outC))
      (xIn : Term Γ (imgShape inC h w)) :
      Term Γ (imgShape outC (down2 h) (down2 w)) :=
    let h' := down2 h
    let w' := down2 w
    have hh' : h' > 0 := down2_pos h
    have hw' : w' > 0 := down2_pos w
    let conv1 :=
      Term.op (Γ := Γ)
        (conv3x3Down (inC := inC) (outC := outC) (h := h) (w := w)
          (h_inC := Nat.ne_of_gt h_inC_pos) (_h_outC := Nat.ne_of_gt h_outC_pos))
        (Args.cons k1
          (Args.cons b1
            (Args.cons xIn (Args.nil))))
    let bn1 :=
      Term.op (Γ := Γ)
        (PrimOp.batchnormChw (channels := outC) (height := h') (width := w') h_outC_pos hh' hw')
        (Args.cons g1
          (Args.cons bt1
            (Args.cons conv1 (Args.nil))))
    let relu1 := Term.op (Γ := Γ) (PrimOp.relu (s := imgShape outC h' w')) (Args.cons bn1
      (Args.nil))
    let conv2 :=
      Term.op (Γ := Γ)
        (conv3x3Same (c := outC) (h := h') (w := w')
          (h_c := Nat.ne_of_gt h_outC_pos) (hh := hh') (hw := hw'))
        (Args.cons k2
          (Args.cons b2
            (Args.cons relu1 (Args.nil))))
    let bn2 :=
      Term.op (Γ := Γ)
        (PrimOp.batchnormChw (channels := outC) (height := h') (width := w') h_outC_pos hh' hw')
        (Args.cons g2
          (Args.cons bt2
            (Args.cons conv2 (Args.nil))))
    let proj :=
      Term.op (Γ := Γ)
        (conv1x1Down (inC := inC) (outC := outC) (h := h) (w := w)
          (h_inC := Nat.ne_of_gt h_inC_pos) (_h_outC := Nat.ne_of_gt h_outC_pos))
        (Args.cons kp
          (Args.cons bp
            (Args.cons xIn (Args.nil))))
    let projBN :=
      Term.op (Γ := Γ)
        (PrimOp.batchnormChw (channels := outC) (height := h') (width := w') h_outC_pos hh' hw')
        (Args.cons gp
          (Args.cons btp
            (Args.cons proj (Args.nil))))
    let sum := Term.op (Γ := Γ) (PrimOp.add (s := imgShape outC h' w')) (Args.cons bn2 (Args.cons
      projBN (Args.nil)))
    Term.op (Γ := Γ) (PrimOp.relu (s := imgShape outC h' w')) (Args.cons sum (Args.nil))

  -- Stage 1: 64, no downsampling.
  let out1 :=
    basicBlockSame 64 h2 w2 h64 h_h2 h_w2
      (p 4 (by decide))
      (p 5 (by decide))
      (p 6 (by decide))
      (p 7 (by decide))
      (p 8 (by decide))
      (p 9 (by decide))
      (p 10 (by decide))
      (p 11 (by decide))
      stemPool

  let out2 :=
    basicBlockSame 64 h2 w2 h64 h_h2 h_w2
      (p 12 (by decide))
      (p 13 (by decide))
      (p 14 (by decide))
      (p 15 (by decide))
      (p 16 (by decide))
      (p 17 (by decide))
      (p 18 (by decide))
      (p 19 (by decide))
      out1

  -- Stage 2: 128, downsample then one normal block.
  have h128 : 128 > 0 := by decide
  let out3 :=
    basicBlockDown 64 128 h2 w2 h64 h128 h_h2 h_w2
      (p 20 (by decide))
      (p 21 (by decide))
      (p 22 (by decide))
      (p 23 (by decide))
      (p 24 (by decide))
      (p 25 (by decide))
      (p 26 (by decide))
      (p 27 (by decide))
      (p 28 (by decide))
      (p 29 (by decide))
      (p 30 (by decide))
      (p 31 (by decide))
      out2

  let out4 :=
    basicBlockSame 128 h3 w3 h128 h_h3 h_w3
      (p 32 (by decide))
      (p 33 (by decide))
      (p 34 (by decide))
      (p 35 (by decide))
      (p 36 (by decide))
      (p 37 (by decide))
      (p 38 (by decide))
      (p 39 (by decide))
      out3

  -- Stage 3: 256
  have h256 : 256 > 0 := by decide
  let out5 :=
    basicBlockDown 128 256 h3 w3 h128 h256 h_h3 h_w3
      (p 40 (by decide))
      (p 41 (by decide))
      (p 42 (by decide))
      (p 43 (by decide))
      (p 44 (by decide))
      (p 45 (by decide))
      (p 46 (by decide))
      (p 47 (by decide))
      (p 48 (by decide))
      (p 49 (by decide))
      (p 50 (by decide))
      (p 51 (by decide))
      out4

  let out6 :=
    basicBlockSame 256 h4 w4 h256 h_h4 h_w4
      (p 52 (by decide))
      (p 53 (by decide))
      (p 54 (by decide))
      (p 55 (by decide))
      (p 56 (by decide))
      (p 57 (by decide))
      (p 58 (by decide))
      (p 59 (by decide))
      out5

  -- Stage 4: 512
  have h512 : 512 > 0 := by decide
  let out7 :=
    basicBlockDown 256 512 h4 w4 h256 h512 h_h4 h_w4
      (p 60 (by decide))
      (p 61 (by decide))
      (p 62 (by decide))
      (p 63 (by decide))
      (p 64 (by decide))
      (p 65 (by decide))
      (p 66 (by decide))
      (p 67 (by decide))
      (p 68 (by decide))
      (p 69 (by decide))
      (p 70 (by decide))
      (p 71 (by decide))
      out6

  let out8 :=
    basicBlockSame 512 h5 w5 h512 h_h5 h_w5
      (p 72 (by decide))
      (p 73 (by decide))
      (p 74 (by decide))
      (p 75 (by decide))
      (p 76 (by decide))
      (p 77 (by decide))
      (p 78 (by decide))
      (p 79 (by decide))
      out7

  -- Head: global avg pool then linear classifier.
  let pooled : Term Γ (Shape.Vec 512) :=
    Term.op (Γ := Γ)
      (PrimOp.globalAvgPool2dChw (c := 512) (h := h5) (w := w5)
        (h_c := h512) (h_h := Nat.ne_of_gt h_h5) (h_w := Nat.ne_of_gt h_w5))
      (Args.cons out8 (Args.nil))

  let logits : Term Γ (Shape.Vec numClasses) :=
    Term.op (Γ := Γ) (PrimOp.linear (inDim := 512) (outDim := numClasses))
      (Args.cons
        (p 80 (by decide))
        (Args.cons
          (p 81 (by decide))
          (Args.cons pooled (Args.nil))))

  { initParams := initParams
    body := logits }

end ResNet18

end Models
end GraphSpec
end NN
