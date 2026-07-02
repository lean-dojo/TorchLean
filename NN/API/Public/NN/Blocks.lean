/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/


module

public import NN.API.Public.NN.VisionLayers
public import NN.API.Public.TensorPack

/-!
Reusable neural-network blocks.

This module defines public block constructors such as residual, convolutional, and MLP-style
compositions built from the public layer-building API.
-/

@[expose] public section

namespace NN
namespace API
namespace nn
namespace pure
namespace blocks


/--
Small set of activation choices for block builders.

PyTorch analogues:
- `relu`    <-> `torch.nn.ReLU`
- `gelu`    <-> `torch.nn.GELU`
- `silu`    <-> `torch.nn.SiLU`
- `tanh`    <-> `torch.nn.Tanh`
- `sigmoid` <-> `torch.nn.Sigmoid`
-/
inductive Activation where
  | relu
  | gelu
  | silu
  | tanh
  | sigmoid
deriving Repr, DecidableEq

/-- Interpret an `Activation` as a TorchLean layer. -/
def activation {s : Spec.Shape} : Activation → Sequential s s
  | .relu => relu (s := s)
  | .gelu => gelu (s := s)
  | .silu => silu (s := s)
  | .tanh => tanh (s := s)
  | .sigmoid => sigmoid (s := s)

/--
MLP (multi-layer perceptron) configuration.

This builder produces a sequential stack of linear layers with activations and optional dropout.

PyTorch analogue: a hand-written `nn.Sequential(Linear(...), ReLU(), ..., Linear(...))`.
-/
structure MLP where
  /-- Hidden layer widths (each entry creates a `Linear -> Activation` stage). -/
  hidden : List Nat := []
  /-- Activation used after each hidden linear layer. -/
  activation : Activation := .relu
  /-- Optional dropout probability after each activation. -/
  dropout? : Option Float := none
  /-- Base seed used to deterministically initialize all linear layers (and dropout if present). -/
  seedBase : Nat := 0

/--
Internal recursion for `mlp`.

This builds the sequential stack stage-by-stage, threading a seed so each linear (and optional
dropout) layer gets a deterministic initialization key.
-/
def mlpGo (act : Activation) (dropout? : Option Float) :
    (inDim : Nat) → (hidden : List Nat) → (outDim : Nat) → (seed : Nat) →
      Sequential (NN.Tensor.Shape.Vec inDim) (NN.Tensor.Shape.Vec outDim)
  | inDim, [], outDim, seed =>
      linear inDim outDim seed (seed + 1)
  | inDim, h :: hs, outDim, seed =>
      let lin : Sequential (NN.Tensor.Shape.Vec inDim) (NN.Tensor.Shape.Vec h) :=
        linear inDim h seed (seed + 1)
      let seed' := seed + 2
      let actLayer : Sequential (NN.Tensor.Shape.Vec h) (NN.Tensor.Shape.Vec h) :=
        activation (s := NN.Tensor.Shape.Vec h) act
      let mid : Sequential (NN.Tensor.Shape.Vec h) (NN.Tensor.Shape.Vec h) × Nat :=
        match dropout? with
        | none => (actLayer, seed')
        | some p =>
            ((seq! actLayer, dropout (s := NN.Tensor.Shape.Vec h) p (seed := seed')), seed' + 1)
      let rest :=
        mlpGo act dropout? h hs outDim mid.snd
      seq! lin, mid.fst, rest

/--
Build an MLP as a sequential stack of linear layers and activations.

This is a small PyTorch-shaped constructor: a typical call looks like:
`API.nn.blocks.mlp 784 10 { hidden := [128, 128], activation := .relu }`.
-/
def mlp (inDim outDim : Nat) (cfg : MLP := {}) :
    Sequential (NN.Tensor.Shape.Vec inDim) (NN.Tensor.Shape.Vec outDim) :=
  mlpGo cfg.activation cfg.dropout? inDim cfg.hidden outDim cfg.seedBase

/--
Conv2d + activation (+ optional dropout) block configuration (CHW layout).

This compact block is used by vision examples before moving to larger curated blocks.
-/
structure Conv2dAct where
  /-- Conv hyperparameters and seeds. -/
  conv : Conv2d
  /-- Activation applied after the convolution. -/
  activation : Activation := .relu
  /-- Optional dropout probability after the activation. -/
  dropout? : Option Float := none
  /-- Seed for dropout RNG (only used when `dropout?` is present). -/
  seedDropout : Nat := 0

/-- `Conv2d -> Activation -> (optional Dropout)` over CHW inputs. -/
def conv2dAct {inC inH inW : Nat} (cfg : Conv2dAct) [NeZero inC] [NeZero cfg.conv.kH] [NeZero
  cfg.conv.kW] :
    Sequential
      (NN.Tensor.Shape.Image inC inH inW)
      (NN.Tensor.Shape.Image cfg.conv.outC
        ((inH + 2 * cfg.conv.padding - cfg.conv.kH) / cfg.conv.stride + 1)
        ((inW + 2 * cfg.conv.padding - cfg.conv.kW) / cfg.conv.stride + 1)) :=
  let core := seq! (conv2dCHW (inC := inC) (inH := inH) (inW := inW) cfg.conv), activation
    cfg.activation
  match cfg.dropout? with
  | none => core
  | some p => seq! core, dropout p (seed := cfg.seedDropout)

/-!
## Vision blocks

These are small, *named-field* building blocks intended for public examples:

- reduce seed/proof noise at call sites,
- keep composition explicit (still `seq!` stacking),
- provide canonical blocks users expect from PyTorch codebases.

They are conservative by design, prioritizing readability and stable typing over maximum coverage.
-/

/--
Configuration for a common vision block:
`Conv2d -> BatchNorm2d -> Activation -> (optional Dropout)`.

This is used by `conv2dNormActCHW` (single-image CHW) and `conv2dNormAct` (batched NCHW).
We keep deterministic seed allocation explicit via `seedBase` so examples stay reproducible.
-/
structure Conv2dNormAct where
  /-- Conv hyperparameters (seeds inside this record are ignored; use `seedBase`). -/
  conv : Conv2d
  /-- Activation after normalization. -/
  activation : Activation := .relu
  /-- Optional dropout applied after the activation. -/
  dropout? : Option Float := none
  /-- Base seed for deterministic init (derived seeds are allocated in a fixed order). -/
  seedBase : Nat := 0

/--
`Conv2d -> BatchNorm -> Activation -> (optional Dropout)`, over a single CHW image (no batch axis).

Seed allocation (relative to `seedBase`):

- `seedBase + 0,1`: conv kernel / bias
- `seedBase + 2..5`: BN gamma / beta / running-mean / running-var
- `seedBase + 6`: dropout
-/
def conv2dNormActCHW {inC inH inW : Nat} (cfg : Conv2dNormAct)
    [NeZero inC] [NeZero cfg.conv.kH] [NeZero cfg.conv.kW] [NeZero cfg.conv.outC] :
    Sequential
      (NN.Tensor.Shape.Image inC inH inW)
      (NN.Tensor.Shape.Image cfg.conv.outC
        ((inH + 2 * cfg.conv.padding - cfg.conv.kH) / cfg.conv.stride + 1)
        ((inW + 2 * cfg.conv.padding - cfg.conv.kW) / cfg.conv.stride + 1)) :=
  let conv : Conv2d :=
    { cfg.conv with seedK := cfg.seedBase, seedB := cfg.seedBase + 1 }
  let outH : Nat := (inH + 2 * conv.padding - conv.kH) / conv.stride + 1
  let outW : Nat := (inW + 2 * conv.padding - conv.kW) / conv.stride + 1
  have hOutH : outH > 0 := by
    -- `outH = _ + 1`
    simp [outH]
  have hOutW : outW > 0 := by
    simp [outW]
  let bn : Sequential (NN.Tensor.Shape.Image conv.outC outH outW) (NN.Tensor.Shape.Image conv.outC
    outH outW) :=
    TorchLean.Layers.batchNormCHW conv.outC outH outW
      (hC := Nat.pos_of_ne_zero (NeZero.ne (n := conv.outC)))
      (hH := hOutH) (hW := hOutW)
      (seedGamma := cfg.seedBase + 2)
      (seedBeta := cfg.seedBase + 3)
      (seedMean := cfg.seedBase + 4)
      (seedVar := cfg.seedBase + 5)
  let act : Sequential (NN.Tensor.Shape.Image conv.outC outH outW) (NN.Tensor.Shape.Image conv.outC
    outH outW) :=
    activation (s := NN.Tensor.Shape.Image conv.outC outH outW) cfg.activation
  let core := seq! (conv2dCHW (inC := inC) (inH := inH) (inW := inW) conv), bn, act
  match cfg.dropout? with
  | none => core
  | some p =>
      let s : Spec.Shape := NN.Tensor.Shape.Image conv.outC outH outW
      seq! core, dropout (s := s) p (seed := cfg.seedBase + 6)

/--
Configuration for `conv2dNormActPool*`: a `Conv2dNormAct` block followed by max-pooling.

This matches the common “conv-bn-act-pool” pattern used in small CNNs.
-/
structure Conv2dNormActPool where
  /-- Conv/BN/activation/dropout block configuration. -/
  block : Conv2dNormAct
  /-- Pooling hyperparameters (defaults to `2×2` stride-2 max pool). -/
  pool : MaxPool2d := { kH := 2, kW := 2, stride := 2 }

/-- `conv2dNormActCHW` followed by `MaxPool2dCHW`. -/
def conv2dNormActPoolCHW {inC inH inW : Nat} (cfg : Conv2dNormActPool)
    [NeZero inC]
    [NeZero cfg.block.conv.kH] [NeZero cfg.block.conv.kW] [NeZero cfg.block.conv.outC]
    [NeZero cfg.pool.kH] [NeZero cfg.pool.kW] :
    Sequential
      (NN.Tensor.Shape.Image inC inH inW)
      (NN.Tensor.Shape.Image cfg.block.conv.outC
        ((((inH + 2 * cfg.block.conv.padding - cfg.block.conv.kH) / cfg.block.conv.stride + 1) -
          cfg.pool.kH) / cfg.pool.stride + 1)
        ((((inW + 2 * cfg.block.conv.padding - cfg.block.conv.kW) / cfg.block.conv.stride + 1) -
          cfg.pool.kW) / cfg.pool.stride + 1)) :=
  let core := conv2dNormActCHW (inC := inC) (inH := inH) (inW := inW) cfg.block
  -- Pool input dims are the conv output dims.
  let outH : Nat := (inH + 2 * cfg.block.conv.padding - cfg.block.conv.kH) / cfg.block.conv.stride +
    1
  let outW : Nat := (inW + 2 * cfg.block.conv.padding - cfg.block.conv.kW) / cfg.block.conv.stride +
    1
  let pool : Sequential (NN.Tensor.Shape.Image cfg.block.conv.outC outH outW)
      (NN.Tensor.Shape.Image cfg.block.conv.outC ((outH - cfg.pool.kH) / cfg.pool.stride + 1) ((outW
        - cfg.pool.kW) / cfg.pool.stride + 1)) :=
    maxPool2dCHW (inC := cfg.block.conv.outC) (inH := outH) (inW := outW) cfg.pool
  seq! core, pool

/--
`Conv2d -> BatchNorm2d -> Activation -> (optional Dropout)`, over batched image tensors (`N×C×H×W`).

This is the public PyTorch-like path: examples should build CNNs directly over batched images.
-/
def conv2dNormAct {n inC inH inW : Nat} (cfg : Conv2dNormAct)
    [NeZero n] [NeZero inC] [NeZero cfg.conv.kH] [NeZero cfg.conv.kW] [NeZero cfg.conv.outC] :
    Sequential
      (NN.Tensor.Shape.Images n inC inH inW)
      (NN.Tensor.Shape.Images n cfg.conv.outC
        ((inH + 2 * cfg.conv.padding - cfg.conv.kH) / cfg.conv.stride + 1)
        ((inW + 2 * cfg.conv.padding - cfg.conv.kW) / cfg.conv.stride + 1)) :=
  let conv : Conv2d :=
    { cfg.conv with seedK := cfg.seedBase, seedB := cfg.seedBase + 1 }
  let outH : Nat := (inH + 2 * conv.padding - conv.kH) / conv.stride + 1
  let outW : Nat := (inW + 2 * conv.padding - conv.kW) / conv.stride + 1
  have hOutH : outH > 0 := by simp [outH]
  have hOutW : outW > 0 := by simp [outW]
  let bn : Sequential (NN.Tensor.Shape.Images n conv.outC outH outW) (NN.Tensor.Shape.Images n
    conv.outC outH outW) :=
    TorchLean.Layers.batchNorm2dNCHW n conv.outC outH outW
      (hN := Nat.pos_of_ne_zero (NeZero.ne (n := n)))
      (hC := Nat.pos_of_ne_zero (NeZero.ne (n := conv.outC)))
      (hH := hOutH) (hW := hOutW)
      (seedGamma := cfg.seedBase + 2)
      (seedBeta := cfg.seedBase + 3)
      (seedMean := cfg.seedBase + 4)
      (seedVar := cfg.seedBase + 5)
  let act : Sequential (NN.Tensor.Shape.Images n conv.outC outH outW) (NN.Tensor.Shape.Images n
    conv.outC outH outW) :=
    activation (s := NN.Tensor.Shape.Images n conv.outC outH outW) cfg.activation
  let core := seq! (conv2d (n := n) (inC := inC) (inH := inH) (inW := inW) conv), bn, act
  match cfg.dropout? with
  | none => core
  | some p =>
      let s : Spec.Shape := NN.Tensor.Shape.Images n conv.outC outH outW
      seq! core, dropout (s := s) p (seed := cfg.seedBase + 6)

/-- `conv2dNormAct` followed by `MaxPool2d`, over batched image tensors. -/
def conv2dNormActPool {n inC inH inW : Nat} (cfg : Conv2dNormActPool)
    [NeZero n]
    [NeZero inC]
    [NeZero cfg.block.conv.kH] [NeZero cfg.block.conv.kW] [NeZero cfg.block.conv.outC]
    [NeZero cfg.pool.kH] [NeZero cfg.pool.kW] :
    Sequential
      (NN.Tensor.Shape.Images n inC inH inW)
      (NN.Tensor.Shape.Images n cfg.block.conv.outC
        ((((inH + 2 * cfg.block.conv.padding - cfg.block.conv.kH) / cfg.block.conv.stride + 1) -
          cfg.pool.kH) / cfg.pool.stride + 1)
        ((((inW + 2 * cfg.block.conv.padding - cfg.block.conv.kW) / cfg.block.conv.stride + 1) -
          cfg.pool.kW) / cfg.pool.stride + 1)) :=
  let core := conv2dNormAct (n := n) (inC := inC) (inH := inH) (inW := inW) cfg.block
  -- Pool input dims are the conv output dims.
  let outH : Nat := (inH + 2 * cfg.block.conv.padding - cfg.block.conv.kH) / cfg.block.conv.stride +
    1
  let outW : Nat := (inW + 2 * cfg.block.conv.padding - cfg.block.conv.kW) / cfg.block.conv.stride +
    1
  let pool : Sequential (NN.Tensor.Shape.Images n cfg.block.conv.outC outH outW)
      (NN.Tensor.Shape.Images n cfg.block.conv.outC ((outH - cfg.pool.kH) / cfg.pool.stride + 1)
        ((outW - cfg.pool.kW) / cfg.pool.stride + 1)) :=
    maxPool2d (n := n) (inC := cfg.block.conv.outC) (inH := outH) (inW := outW) cfg.pool
  seq! core, pool

/--
Residual/skip-connection layer as a single `LayerDef`.

Given `inner : Seq s s`, this builds a layer that computes `x |-> inner(x) + x`.

PyTorch analogue: `x + f(x)` blocks used throughout ResNets and Transformers.
-/
def residualLayer {s : Spec.Shape} (inner : Sequential s s) : LayerDef s s :=
  let ps := TorchLean.NN.Seq.paramShapes inner
  { kind := "Residual"
    paramShapes := ps
    initParams := TorchLean.NN.Seq.initParams inner
    paramRequiresGrad := TorchLean.NN.Seq.paramRequiresGrad inner
    updateBuffers := some (fun mode {α} _ _ ps x =>
      TorchLean.NN.Seq.updateBuffers (α := α) (model := inner) mode ps x)
    forward := fun mode {α} _ _ =>
      fun {m} _ _ =>
        _root_.Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := fun sh => TorchLean.RefTy (m := m) (α := α) sh)
          (ss := ps ++ [s])
          (β := m (TorchLean.RefTy (m := m) (α := α) s))
          (fun args => do
            let (_psRefs, xRef) :=
              _root_.Runtime.Autograd.Torch.RefList.splitAppend1
                (Ref := fun sh => TorchLean.RefTy (m := m) (α := α) sh)
                (ss := ps) (τ := s) args
            let y ←
              _root_.Runtime.Autograd.Torch.CurriedRef.uncurry
                (Ref := fun sh => TorchLean.RefTy (m := m) (α := α) sh)
                (ss := ps ++ [s])
                (β := m (TorchLean.RefTy (m := m) (α := α) s))
                (TorchLean.NN.Seq.programWithMode (mode := mode) (model := inner) (α := α))
                args
            TorchLean.add (m := m) (α := α) (s := s) y xRef)
  }

/-- Lift `residualLayer` into a sequential model. -/
def residual {s : Spec.Shape} (inner : Sequential s s) : Sequential s s :=
  nn.of (residualLayer inner)

/-!
## Branching (skip connections)

`Seq` is linear, but we sometimes want a PyTorch-like `x |-> f(x) + g(x)` block.

We expose this as a single `LayerDef` whose parameter list is `params(f) ++ params(g)` and whose
forward pass runs both programs and adds their outputs.
-/

/--
Combine two sequential branches into a single layer that adds their outputs.

The resulting layer runs both `f` and `g` on the same input `x` and returns `f(x) + g(x)`.
Parameters are concatenated as `params(f) ++ params(g)`.
-/
def addBranchesLayer {σ τ : Spec.Shape} (f g : Sequential σ τ) : LayerDef σ τ :=
  let psF := TorchLean.NN.Seq.paramShapes f
  let psG := TorchLean.NN.Seq.paramShapes g
  { kind := "AddBranches"
    paramShapes := psF ++ psG
    initParams :=
      NN.API.tensorpack.append (α := Float) (ss₁ := psF) (ss₂ := psG)
        (TorchLean.NN.Seq.initParams f) (TorchLean.NN.Seq.initParams g)
    paramRequiresGrad := TorchLean.NN.Seq.paramRequiresGrad f ++ TorchLean.NN.Seq.paramRequiresGrad
      g
    updateBuffers := some (fun mode {α} _ _ ps x => do
      let (psFv, psGv) := NN.API.tensorpack.split (α := α) (ss₁ := psF) (ss₂ := psG) ps
      let psFv' ← TorchLean.NN.Seq.updateBuffers (α := α) (model := f) mode psFv x
      let psGv' ← TorchLean.NN.Seq.updateBuffers (α := α) (model := g) mode psGv x
      pure <| NN.API.tensorpack.append (α := α) (ss₁ := psF) (ss₂ := psG) psFv' psGv'
    )
    forward := fun mode {α} _ _ =>
      fun {m} _ _ =>
        _root_.Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := fun sh => TorchLean.RefTy (m := m) (α := α) sh)
          (ss := psF ++ psG ++ [σ])
          (β := m (TorchLean.RefTy (m := m) (α := α) τ))
          (fun args => do
            let (psAll, xRef) :=
              _root_.Runtime.Autograd.Torch.RefList.splitAppend1
                (Ref := fun sh => TorchLean.RefTy (m := m) (α := α) sh)
                (ss := psF ++ psG) (τ := σ) args
            let (psFrefs, psGrefs) :=
              _root_.Runtime.Autograd.Torch.RefList.split
                (Ref := fun sh => TorchLean.RefTy (m := m) (α := α) sh)
                (ss₁ := psF) (ss₂ := psG) psAll
            let yF ←
              _root_.Runtime.Autograd.Torch.CurriedRef.uncurry
                (Ref := fun sh => TorchLean.RefTy (m := m) (α := α) sh)
                (ss := psF ++ [σ])
                (β := m (TorchLean.RefTy (m := m) (α := α) τ))
                (TorchLean.NN.Seq.programWithMode (mode := mode) (model := f) (α := α))
                (_root_.Runtime.Autograd.Torch.RefList.append psFrefs (.cons xRef .nil))
            let yG ←
              _root_.Runtime.Autograd.Torch.CurriedRef.uncurry
                (Ref := fun sh => TorchLean.RefTy (m := m) (α := α) sh)
                (ss := psG ++ [σ])
                (β := m (TorchLean.RefTy (m := m) (α := α) τ))
                (TorchLean.NN.Seq.programWithMode (mode := mode) (model := g) (α := α))
                (_root_.Runtime.Autograd.Torch.RefList.append psGrefs (.cons xRef .nil))
            TorchLean.add (m := m) (α := α) (s := τ) yF yG)
  }

/--
Combine two models with the same input/output shapes by summing their outputs.

This is a typed residual-add block: `addBranches f g` represents the model `x ↦ f(x) + g(x)`,
and its parameter list is the concatenation of the two branches’ parameter lists.
-/
def addBranches {σ τ : Spec.Shape} (f g : Sequential σ τ) : Sequential σ τ :=
  nn.of (addBranchesLayer f g)

/-!
## ResNet BasicBlock

We provide a *typed* and *composable* ResNet-18 style BasicBlock over CHW tensors.

Key idea: we use a small canonical stride-2 formula `down2` (matching `GraphSpec/Models/resnet18`)
so projection shortcuts typecheck cleanly without leaking Nat arithmetic at call sites.
-/
