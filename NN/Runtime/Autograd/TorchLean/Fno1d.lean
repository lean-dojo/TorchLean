/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.Fft

import Mathlib.Algebra.Order.Algebra

/-!
# FNO1D

1D Fourier Neural Operator (FNO) blocks based on an explicit FFT/IFFT transform.

Important note about TorchLean’s layer architecture:
- TorchLean layers are scalar-polymorphic but **do not change scalar type** mid-model.
- A real-valued FFT (real -> complex) therefore cannot be expressed as a `LayerDef` today.

So this implementation is intended to be instantiated over a complex scalar backend, e.g.:
`--dtype=complex` (see `NN.API.DType`).

Implementation note:
`FFT1D` uses explicit DFT matrices (`matmul` with a constant matrix). This is
correctness-first and keeps the transform differentiable under the existing autograd rules, but it
is not optimized for large `grid`.

Reference:
- Zongyi Li et al., “Fourier Neural Operator for Parametric Partial Differential Equations”, 2020.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace TorchLean

open Spec
open Tensor

namespace NN

namespace FNO1D

/-- Shape abbreviation for a length-`n` vector. -/
abbrev vec (n : Nat) : Shape := .dim n .scalar

/-- Shape abbreviation for an `m×n` matrix. -/
abbrev mat (m n : Nat) : Shape := .dim m (.dim n .scalar)

/--
Activation choice for FNO blocks.

PyTorch analogy: this corresponds to picking `torch.tanh` vs `torch.relu` at the end of a block.
-/
inductive Activation where
  | tanh
  | relu
  deriving DecidableEq, Repr

/-! ## Small shape views -/

/-- Reshape a `grid`-vector into a `grid×1` matrix. -/
def reshapeVecToMat1 (grid : Nat) :
    LayerDef (vec grid) (mat grid 1) :=
  let s₁ : Shape := vec grid
  let s₂ : Shape := mat grid 1
  have h : Shape.size s₁ = Shape.size s₂ := by
    simp [Shape.size, s₁, s₂]
  { paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => TorchLean.reshape (m := m) (α := α) (s₁ := s₁) (s₂ := s₂) x h
  }

/-- Inverse of `reshapeVecToMat1`: view a `grid×1` matrix back as a length-`grid` vector. -/
def reshapeMatToVec1 (grid : Nat) :
    LayerDef (mat grid 1) (vec grid) :=
  let s₁ : Shape := mat grid 1
  let s₂ : Shape := vec grid
  have h : Shape.size s₁ = Shape.size s₂ := by
    simp [Shape.size, s₁, s₂]
  { paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => TorchLean.reshape (m := m) (α := α) (s₁ := s₁) (s₂ := s₂) x h
  }

/-! ## Pointwise affine -/

/-- Pointwise affine map for 2D tensors: `y = x · W + b`, where `b` broadcasts over the first axis. -/
def matAffine
    (grid inC outC : Nat)
    (seedW seedB : Nat := 0) :
    LayerDef (mat grid inC) (mat grid outC) :=
  let WShape : Shape := mat inC outC
  let bShape : Shape := vec outC
  let w0 : Tensor Float WShape :=
    Torch.Init.tensor (s := WShape) (sch := .uniform (-0.1) 0.1) (seed := seedW)
  let b0 : Tensor Float bShape :=
    Torch.Init.tensor (s := bShape) (sch := .zeros) (seed := seedB)
  { paramShapes := [WShape, bShape]
    initParams := Torch.tlist2 w0 b0
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun w b x =>
          (show m (RefTy (m := m) (α := α) (mat grid outC)) from do
            let y ←
              TorchLean.matmul (m := m) (α := α)
                (mDim := grid) (nDim := inC) (pDim := outC) x w
            let bb ←
              TorchLean.broadcastTo (m := m) (α := α)
                (s₁ := bShape) (s₂ := mat grid outC) Shape.BroadcastTo.proof b
            TorchLean.add (m := m) (α := α) (s := mat grid outC) y bb)
  }

/-! ## Spectral convolution -/

/-- 3D weight shape used by mode-wise spectral convolution: `modes × width × width`. -/
abbrev spectralWShape (modes width : Nat) : Shape :=
  .dim modes (.dim width (.dim width .scalar))

/-- Reshape `modes×width` to `modes×1×width` for `bmm` (mode-wise matmul). -/
def reshapeModesMatToBmmIn (modes width : Nat) :
    LayerDef (mat modes width) (.dim modes (.dim 1 (.dim width .scalar))) :=
  let s₁ : Shape := mat modes width
  let s₂ : Shape := .dim modes (.dim 1 (.dim width .scalar))
  have h : Shape.size s₁ = Shape.size s₂ := by
    simp [Shape.size, s₁, s₂]
  { paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => TorchLean.reshape (m := m) (α := α) (s₁ := s₁) (s₂ := s₂) x h
  }

/-- Reshape `modes×1×width` back to `modes×width` after `bmm`. -/
def reshapeBmmOutToModesMat (modes width : Nat) :
    LayerDef (.dim modes (.dim 1 (.dim width .scalar))) (mat modes width) :=
  let s₁ : Shape := .dim modes (.dim 1 (.dim width .scalar))
  let s₂ : Shape := mat modes width
  have h : Shape.size s₁ = Shape.size s₂ := by
    simp [Shape.size, s₁, s₂]
  { paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => TorchLean.reshape (m := m) (α := α) (s₁ := s₁) (s₂ := s₂) x h
  }

/-!
One FNO-style block:

1. `x̂ = FFT(x)` along the grid axis
2. Apply learned complex linear maps to the first `modes` and last `modes` frequency rows
3. Zero out the middle frequencies
4. `y = IFFT(ŷ)`
5. Add a pointwise affine skip connection and apply an activation
-/
def block
    (grid width modes : Nat)
    (activation : Activation := .tanh)
    (seed : Nat := 0)
    (hModes : 2 * modes ≤ grid) :
    LayerDef (mat grid width) (mat grid width) :=
  let wLowShape : Shape := spectralWShape modes width
  let wHighShape : Shape := spectralWShape modes width
  let wSkipShape : Shape := mat width width
  let bSkipShape : Shape := vec width
  let wLow0 : Tensor Float wLowShape :=
    Torch.Init.tensor (s := wLowShape) (sch := .uniform (-0.05) 0.05) (seed := seed)
  let wHigh0 : Tensor Float wHighShape :=
    Torch.Init.tensor (s := wHighShape) (sch := .uniform (-0.05) 0.05) (seed := seed + 1)
  let wSkip0 : Tensor Float wSkipShape :=
    Torch.Init.tensor (s := wSkipShape) (sch := .uniform (-0.05) 0.05) (seed := seed + 2)
  let bSkip0 : Tensor Float bSkipShape :=
    Torch.Init.tensor (s := bSkipShape) (sch := .zeros) (seed := seed + 3)

  have hModes' : modes ≤ grid := by
    -- modes ≤ (modes + modes) = 2*modes ≤ grid
    have hModesAdd : modes + modes ≤ grid := by
      simpa [two_mul] using hModes
    exact le_trans (Nat.le_add_right modes modes) hModesAdd

  let midLen : Nat := grid - 2 * modes

  { paramShapes := [wLowShape, wHighShape, wSkipShape, bSkipShape]
    initParams := Torch.tlist4 wLow0 wHigh0 wSkip0 bSkip0
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun wLow wHigh wSkip bSkip x =>
          (show m (RefTy (m := m) (α := α) (mat grid width)) from do
            let f : Tensor α (mat grid grid) := FFT1D.dftMatrix (α := α) grid
            let fi : Tensor α (mat grid grid) := FFT1D.idftMatrix (α := α) grid
            let fR ← TorchLean.const (m := m) (α := α) (s := mat grid grid) f
            let fiR ← TorchLean.const (m := m) (α := α) (s := mat grid grid) fi

            let xHat ←
              TorchLean.matmul (m := m) (α := α)
                (mDim := grid) (nDim := grid) (pDim := width) fR x

            -- Low frequencies: rows [0, modes)
            let xLow ←
              TorchLean.sliceRange0 (m := m) (α := α)
                (nDim := grid) (s := .dim width .scalar)
                (start := 0) (len := modes) (by simpa using hModes') xHat

            -- High frequencies: rows [grid - modes, grid)
            let xHigh ←
              TorchLean.sliceRange0 (m := m) (α := α)
                (nDim := grid) (s := .dim width .scalar)
                (start := grid - modes) (len := modes)
                (by
                  -- modes + (grid - modes) = grid
                  simp [Nat.add_sub_of_le hModes'] ) xHat

            -- Mode-wise linear maps using `bmm`
            let xLow3 ← (reshapeModesMatToBmmIn (modes := modes) (width := width)).forward .eval
              (α := α) (m := m) xLow
            let yLow3 ←
              TorchLean.bmm (m := m) (α := α)
                (batch := modes) (mDim := 1) (nDim := width) (pDim := width) xLow3 wLow
            let yLow ← (reshapeBmmOutToModesMat (modes := modes) (width := width)).forward .eval
              (α := α) (m := m) yLow3

            let xHigh3 ← (reshapeModesMatToBmmIn (modes := modes) (width := width)).forward .eval
              (α := α) (m := m) xHigh
            let yHigh3 ←
              TorchLean.bmm (m := m) (α := α)
                (batch := modes) (mDim := 1) (nDim := width) (pDim := width) xHigh3 wHigh
            let yHigh ← (reshapeBmmOutToModesMat (modes := modes) (width := width)).forward .eval
              (α := α) (m := m) yHigh3

            -- Zero out the middle frequencies
            let mid0 : Tensor α (mat midLen width) := Spec.fill (0 : α) (mat midLen width)
            let yMid ← TorchLean.const (m := m) (α := α) (s := mat midLen width) mid0

            -- Concatenate [low, mid, high] back to a `grid×width` spectrum
            let yLowMid ←
              TorchLean.concatDim0 (m := m) (α := α)
                (nDim := modes) (mDim := midLen) (s := .dim width .scalar) yLow yMid
            let yHat' ←
              TorchLean.concatDim0 (m := m) (α := α)
                (nDim := modes + midLen) (mDim := modes) (s := .dim width .scalar) yLowMid yHigh
            let yHat : RefTy (m := m) (α := α) (mat grid width) := by
              have hSum : (modes + midLen) + modes = grid := by
                dsimp [midLen]
                -- rearrange to `2*modes + (grid - 2*modes)` and use `a + (b - a) = b`.
                calc
                  (modes + (grid - 2 * modes)) + modes
                      = modes + ((grid - 2 * modes) + modes) := by
                          simp [Nat.add_assoc]
                  _   = modes + (modes + (grid - 2 * modes)) := by
                          simp [Nat.add_comm]
                  _   = (modes + modes) + (grid - 2 * modes) := by
                          simp [Nat.add_assoc]
                  _   = (2 * modes) + (grid - 2 * modes) := by
                          simp [two_mul, Nat.add_assoc]
                  _   = grid := by
                          simpa using (Nat.add_sub_of_le hModes)
              simpa [mat, hSum] using yHat'

            let ySpec ←
              TorchLean.matmul (m := m) (α := α)
                (mDim := grid) (nDim := grid) (pDim := width) fiR yHat

            -- Skip connection in the original (spatial) domain
            let ySkip0 ←
              TorchLean.matmul (m := m) (α := α)
                (mDim := grid) (nDim := width) (pDim := width) x wSkip
            let bSkipB ←
              TorchLean.broadcastTo (m := m) (α := α)
                (s₁ := bSkipShape) (s₂ := mat grid width) Shape.BroadcastTo.proof bSkip
            let ySkip ← TorchLean.add (m := m) (α := α) (s := mat grid width) ySkip0 bSkipB
            let y ← TorchLean.add (m := m) (α := α) (s := mat grid width) ySpec ySkip

            match activation with
            | .tanh => TorchLean.tanh (m := m) (α := α) (s := mat grid width) y
            | .relu => TorchLean.relu (m := m) (α := α) (s := mat grid width) y)
  }

/-! ## Model constructor -/

/-- `Seq.comp` preserves parameter order by list append. -/
theorem paramShapes_comp {σ τ υ : Shape} (f : Seq σ τ) (g : Seq τ υ) :
    Seq.paramShapes (f >>> g) = Seq.paramShapes f ++ Seq.paramShapes g := by
  induction f with
  | id s =>
      simp [Seq.comp, Seq.paramShapes]
  | cons l rest ih =>
      simp [Seq.comp, Seq.paramShapes, ih, List.append_assoc]

/-- The `blocksSeq` helper from `model`, promoted to a named definition for reuse in lemmas. -/
def blocksSeq
    (grid width modes blocks : Nat)
    (activation : Activation := .tanh)
    (seed : Nat := 0)
    (hModes : 2 * modes ≤ grid) :
    Seq (mat grid width) (mat grid width) :=
  match blocks with
  | 0 => .id (mat grid width)
  | Nat.succ k =>
      Seq.cons
        (block (grid := grid) (width := width) (modes := modes) (activation := activation)
          (seed := seed + 10 * k) (hModes := hModes))
        (blocksSeq (grid := grid) (width := width) (modes := modes) (blocks := k)
          (activation := activation) (seed := seed) (hModes := hModes))

/--
Closed-form parameter shapes for `blocks` repetitions of `block`.

This is a convenience for documentation/lemmas: it matches `Seq.paramShapes (blocksSeq ...)`.
-/
def blocksParamShapes (width modes blocks : Nat) : List Shape :=
  let wLow : Shape := spectralWShape modes width
  let wHigh : Shape := spectralWShape modes width
  let wSkip : Shape := mat width width
  let bSkip : Shape := vec width
  let blockShapes : List Shape := [wLow, wHigh, wSkip, bSkip]
  match blocks with
  | 0 => []
  | Nat.succ k => blockShapes ++ blocksParamShapes (width := width) (modes := modes) (blocks := k)

/-- `Seq.paramShapes (blocksSeq ...)` matches the explicit list computed by `blocksParamShapes`. -/
theorem blocksSeq_paramShapes (grid width modes blocks : Nat) (activation : Activation) (seed : Nat)
    (hModes : 2 * modes ≤ grid) :
    Seq.paramShapes (blocksSeq (grid := grid) (width := width) (modes := modes) (blocks := blocks)
      (activation := activation) (seed := seed) (hModes := hModes))
      = blocksParamShapes (width := width) (modes := modes) (blocks := blocks) := by
  induction blocks with
  | zero =>
      simp [blocksSeq, blocksParamShapes, Seq.paramShapes]
  | succ k ih =>
      simp [blocksSeq, blocksParamShapes, Seq.paramShapes, ih, block, spectralWShape, mat, vec]

/--
Construct a small scalar->scalar 1D FNO model:

Input:  vector of length `grid` (interpreted as a 1D field).
Output: vector of length `grid`.
-/
def model
    (grid width modes blocks : Nat)
    (activation : Activation := .tanh)
    (seed : Nat := 0)
    (hModes : 2 * modes ≤ grid) :
    Seq (vec grid) (vec grid) :=
  let reshapeIn := reshapeVecToMat1 (grid := grid)
  let lift :=
    matAffine (grid := grid) (inC := 1) (outC := width) (seedW := seed) (seedB := seed + 1)
  let proj :=
    matAffine (grid := grid) (inC := width) (outC := 1) (seedW := seed + 100) (seedB := seed + 101)
  let reshapeOut := reshapeMatToVec1 (grid := grid)
  Seq.cons reshapeIn <|
    Seq.cons lift <|
      blocksSeq (grid := grid) (width := width) (modes := modes) (blocks := blocks)
        (activation := activation) (seed := seed) (hModes := hModes) >>>
        Seq.cons proj (Seq.cons reshapeOut (.id (vec grid)))

/--
Parameter-shape lemma for `model`.

This can be useful when writing initialization/serialization code and wanting an explicit, stable
shape list (analogous to inspecting parameter tensor shapes in PyTorch).
-/
theorem model_paramShapes (grid width modes blocks : Nat) (activation : Activation := .tanh)
    (seed : Nat := 0) (hModes : 2 * modes ≤ grid) :
    Seq.paramShapes (model (grid := grid) (width := width) (modes := modes) (blocks := blocks)
      (activation := activation) (seed := seed) (hModes := hModes)) =
      [mat 1 width, vec width] ++
        blocksParamShapes (width := width) (modes := modes) (blocks := blocks) ++
        [mat width 1, vec 1] := by
  simp [model, Seq.paramShapes, blocksSeq_paramShapes, paramShapes_comp,
    reshapeVecToMat1, reshapeMatToVec1, matAffine]


namespace Real

/-!
## Real dense-DFT reference path

These definitions live in the same FNO module as the complex-domain reference path. The real path
keeps the scalar type fixed by representing the DFT as explicit cosine/sine matrices and carrying
real and imaginary channels separately. That makes it the portable correctness-first reference used
by examples and by CUDA parity checks.
-/


abbrev vec (n : Nat) : Shape := .dim n .scalar
abbrev mat (m n : Nat) : Shape := .dim m (.dim n .scalar)

/-- Cosine part of the DFT matrix (unnormalized). -/
def dftCosMatrix {α : Type} [Context α] (n : Nat) : Tensor α (mat n n) :=
  Tensor.dim (fun k =>
    Tensor.dim (fun j =>
      let twoPi : α := Numbers.two * MathFunctions.pi
      let ang : α := twoPi * (j.val : α) * (k.val : α) / (n : α)
      Tensor.scalar (MathFunctions.cos ang)))

/-- `-sin` part of the DFT matrix (unnormalized), used for the imaginary component. -/
def dftNegSinMatrix {α : Type} [Context α] (n : Nat) : Tensor α (mat n n) :=
  Tensor.dim (fun k =>
    Tensor.dim (fun j =>
      let twoPi : α := Numbers.two * MathFunctions.pi
      let ang : α := twoPi * (j.val : α) * (k.val : α) / (n : α)
      Tensor.scalar (0 - MathFunctions.sin ang)))

/-- Cosine part of the inverse DFT matrix (normalized by `1/n`). -/
def idftCosMatrix {α : Type} [Context α] (n : Nat) : Tensor α (mat n n) :=
  Tensor.dim (fun j =>
    Tensor.dim (fun k =>
      let twoPi : α := Numbers.two * MathFunctions.pi
      let ang : α := twoPi * (j.val : α) * (k.val : α) / (n : α)
      Tensor.scalar (MathFunctions.cos ang / (n : α))))

/-- Sine part of the inverse DFT matrix (normalized by `1/n`). -/
def idftSinMatrix {α : Type} [Context α] (n : Nat) : Tensor α (mat n n) :=
  Tensor.dim (fun j =>
    Tensor.dim (fun k =>
      let twoPi : α := Numbers.two * MathFunctions.pi
      let ang : α := twoPi * (j.val : α) * (k.val : α) / (n : α)
      Tensor.scalar (MathFunctions.sin ang / (n : α))))

/--
Real-valued spectral convolution block on a `grid×width` tensor.

This is the same high-level algorithm as the complex FNO block, but represented explicitly as:

`DFT(x) = (Re, Im)` computed by `cos/sin` matmuls,
mode-wise complex multiplication using two real weight tensors `(wRe, wIm)`, then
`IDFT(Re,Im)` back to the spatial domain.
-/
def block
    (grid width modes : Nat)
    (seed : Nat := 0)
    (hModes : 2 * modes ≤ grid) :
    LayerDef (mat grid width) (mat grid width) :=
  let wShape : Shape := _root_.Runtime.Autograd.TorchLean.NN.FNO1D.spectralWShape modes width
  let wSkipShape : Shape := mat width width
  let bSkipShape : Shape := vec width
  let w0 (s : Nat) : Tensor Float wShape :=
    _root_.Runtime.Autograd.Torch.Init.tensor (s := wShape) (sch := .uniform (-0.04) 0.04) (seed := seed + s)
  let wSkip0 : Tensor Float wSkipShape :=
    _root_.Runtime.Autograd.Torch.Init.tensor (s := wSkipShape) (sch := .uniform (-0.04) 0.04) (seed := seed + 20)
  let bSkip0 : Tensor Float bSkipShape :=
    _root_.Runtime.Autograd.Torch.Init.tensor (s := bSkipShape) (sch := .zeros) (seed := seed + 21)
  have hModes' : modes ≤ grid := by
    have hModesAdd : modes + modes ≤ grid := by
      simpa [two_mul] using hModes
    exact le_trans (Nat.le_add_right modes modes) hModesAdd
  let midLen : Nat := grid - 2 * modes
  { paramShapes := [wShape, wShape, wShape, wShape, wSkipShape, bSkipShape]
    initParams :=
      .cons (w0 0) <| .cons (w0 1) <| .cons (w0 2) <| .cons (w0 3) <|
        .cons wSkip0 <| .cons bSkip0 .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun wLowRe wLowIm wHighRe wHighIm wSkip bSkip x =>
          (show m (_root_.Runtime.Autograd.TorchLean.NN.Seq.RefT (m := m) (α := α) (mat grid width)) from do
            let fCos ← _root_.Runtime.Autograd.Torch.const (m := m) (α := α) (s := mat grid grid)
              (dftCosMatrix (α := α) grid)
            let fNegSin ← _root_.Runtime.Autograd.Torch.const (m := m) (α := α) (s := mat grid grid)
              (dftNegSinMatrix (α := α) grid)
            let iCos ← _root_.Runtime.Autograd.Torch.const (m := m) (α := α) (s := mat grid grid)
              (idftCosMatrix (α := α) grid)
            let iSin ← _root_.Runtime.Autograd.Torch.const (m := m) (α := α) (s := mat grid grid)
              (idftSinMatrix (α := α) grid)

            let xHatRe ← _root_.Runtime.Autograd.Torch.matmul (m := m) (α := α)
              (mDim := grid) (nDim := grid) (pDim := width) fCos x
            let xHatIm ← _root_.Runtime.Autograd.Torch.matmul (m := m) (α := α)
              (mDim := grid) (nDim := grid) (pDim := width) fNegSin x

            let lowRe ← _root_.Runtime.Autograd.Torch.sliceRange0 (m := m) (α := α)
              (nDim := grid) (s := .dim width .scalar) (start := 0) (len := modes)
              (by simpa using hModes') xHatRe
            let lowIm ← _root_.Runtime.Autograd.Torch.sliceRange0 (m := m) (α := α)
              (nDim := grid) (s := .dim width .scalar) (start := 0) (len := modes)
              (by simpa using hModes') xHatIm
            let highRe ← _root_.Runtime.Autograd.Torch.sliceRange0 (m := m) (α := α)
              (nDim := grid) (s := .dim width .scalar) (start := grid - modes) (len := modes)
              (by simp [Nat.add_sub_of_le hModes']) xHatRe
            let highIm ← _root_.Runtime.Autograd.Torch.sliceRange0 (m := m) (α := α)
              (nDim := grid) (s := .dim width .scalar) (start := grid - modes) (len := modes)
              (by simp [Nat.add_sub_of_le hModes']) xHatIm

            let mulModes
                (xr xi : _root_.Runtime.Autograd.TorchLean.NN.Seq.RefT (m := m) (α := α) (mat modes width))
                (wr wi : _root_.Runtime.Autograd.TorchLean.NN.Seq.RefT (m := m) (α := α)
                  (_root_.Runtime.Autograd.TorchLean.NN.FNO1D.spectralWShape modes width)) := do
              let hIn :
                  Shape.size (mat modes width) = Shape.size (.dim modes (.dim 1 (.dim width .scalar))) := by
                simp [Shape.size]
              let xr3 ← _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
                (s₁ := mat modes width) (s₂ := .dim modes (.dim 1 (.dim width .scalar))) xr hIn
              let xi3 ← _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
                (s₁ := mat modes width) (s₂ := .dim modes (.dim 1 (.dim width .scalar))) xi hIn
              let xrWr ← _root_.Runtime.Autograd.Torch.bmm (m := m) (α := α)
                (batch := modes) (mDim := 1) (nDim := width) (pDim := width) xr3 wr
              let xiWi ← _root_.Runtime.Autograd.Torch.bmm (m := m) (α := α)
                (batch := modes) (mDim := 1) (nDim := width) (pDim := width) xi3 wi
              let xrWi ← _root_.Runtime.Autograd.Torch.bmm (m := m) (α := α)
                (batch := modes) (mDim := 1) (nDim := width) (pDim := width) xr3 wi
              let xiWr ← _root_.Runtime.Autograd.Torch.bmm (m := m) (α := α)
                (batch := modes) (mDim := 1) (nDim := width) (pDim := width) xi3 wr
              let yr3 ← _root_.Runtime.Autograd.Torch.sub (m := m) (α := α)
                (s := .dim modes (.dim 1 (.dim width .scalar))) xrWr xiWi
              let yi3 ← _root_.Runtime.Autograd.Torch.add (m := m) (α := α)
                (s := .dim modes (.dim 1 (.dim width .scalar))) xrWi xiWr
              let hOut :
                  Shape.size (.dim modes (.dim 1 (.dim width .scalar))) = Shape.size (mat modes width) := by
                simp [Shape.size]
              let yr ← _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
                (s₁ := .dim modes (.dim 1 (.dim width .scalar))) (s₂ := mat modes width) yr3 hOut
              let yi ← _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
                (s₁ := .dim modes (.dim 1 (.dim width .scalar))) (s₂ := mat modes width) yi3 hOut
              pure (yr, yi)

            let (yLowRe, yLowIm) := (← mulModes lowRe lowIm wLowRe wLowIm)
            let (yHighRe, yHighIm) := (← mulModes highRe highIm wHighRe wHighIm)

            let mid0 : Tensor α (mat midLen width) := Spec.fill (0 : α) (mat midLen width)
            let yMidRe ← _root_.Runtime.Autograd.Torch.const (m := m) (α := α) (s := mat midLen width) mid0
            let yMidIm ← _root_.Runtime.Autograd.Torch.const (m := m) (α := α) (s := mat midLen width) mid0
            let yLowMidRe ← _root_.Runtime.Autograd.Torch.concatDim0 (m := m) (α := α)
              (nDim := modes) (mDim := midLen) (s := .dim width .scalar) yLowRe yMidRe
            let yLowMidIm ← _root_.Runtime.Autograd.Torch.concatDim0 (m := m) (α := α)
              (nDim := modes) (mDim := midLen) (s := .dim width .scalar) yLowIm yMidIm
            let yHatRe' ← _root_.Runtime.Autograd.Torch.concatDim0 (m := m) (α := α)
              (nDim := modes + midLen) (mDim := modes) (s := .dim width .scalar) yLowMidRe yHighRe
            let yHatIm' ← _root_.Runtime.Autograd.Torch.concatDim0 (m := m) (α := α)
              (nDim := modes + midLen) (mDim := modes) (s := .dim width .scalar) yLowMidIm yHighIm
            let yHatRe : _root_.Runtime.Autograd.TorchLean.NN.Seq.RefT (m := m) (α := α) (mat grid width) := by
              have hSum : (modes + midLen) + modes = grid := by
                dsimp [midLen]
                calc
                  (modes + (grid - 2 * modes)) + modes
                      = modes + ((grid - 2 * modes) + modes) := by simp [Nat.add_assoc]
                  _ = modes + (modes + (grid - 2 * modes)) := by simp [Nat.add_comm]
                  _ = (modes + modes) + (grid - 2 * modes) := by simp [Nat.add_assoc]
                  _ = (2 * modes) + (grid - 2 * modes) := by simp [two_mul, Nat.add_assoc]
                  _ = grid := by simpa using (Nat.add_sub_of_le hModes)
              simpa [mat, hSum] using yHatRe'
            let yHatIm : _root_.Runtime.Autograd.TorchLean.NN.Seq.RefT (m := m) (α := α) (mat grid width) := by
              have hSum : (modes + midLen) + modes = grid := by
                dsimp [midLen]
                calc
                  (modes + (grid - 2 * modes)) + modes
                      = modes + ((grid - 2 * modes) + modes) := by simp [Nat.add_assoc]
                  _ = modes + (modes + (grid - 2 * modes)) := by simp [Nat.add_comm]
                  _ = (modes + modes) + (grid - 2 * modes) := by simp [Nat.add_assoc]
                  _ = (2 * modes) + (grid - 2 * modes) := by simp [two_mul, Nat.add_assoc]
                  _ = grid := by simpa using (Nat.add_sub_of_le hModes)
              simpa [mat, hSum] using yHatIm'

            let yCos ← _root_.Runtime.Autograd.Torch.matmul (m := m) (α := α)
              (mDim := grid) (nDim := grid) (pDim := width) iCos yHatRe
            let ySin ← _root_.Runtime.Autograd.Torch.matmul (m := m) (α := α)
              (mDim := grid) (nDim := grid) (pDim := width) iSin yHatIm
            let ySpec ← _root_.Runtime.Autograd.Torch.sub (m := m) (α := α) (s := mat grid width) yCos ySin

            let ySkip0 ← _root_.Runtime.Autograd.Torch.matmul (m := m) (α := α)
              (mDim := grid) (nDim := width) (pDim := width) x wSkip
            let bSkipB ← _root_.Runtime.Autograd.Torch.broadcastTo (m := m) (α := α)
              (s₁ := bSkipShape) (s₂ := mat grid width) Shape.BroadcastTo.proof bSkip
            let ySkip ← _root_.Runtime.Autograd.Torch.add (m := m) (α := α) (s := mat grid width) ySkip0 bSkipB
            let y ← _root_.Runtime.Autograd.Torch.add (m := m) (α := α) (s := mat grid width) ySpec ySkip
            _root_.Runtime.Autograd.Torch.relu (m := m) (α := α) (s := mat grid width) y)
  }

def blocksSeq (grid width modes blocks : Nat) (seed : Nat) (hModes : 2 * modes ≤ grid) :
    _root_.Runtime.Autograd.TorchLean.NN.Seq (mat grid width) (mat grid width) :=
  match blocks with
  | 0 => .id (mat grid width)
  | Nat.succ k =>
      .cons (block (grid := grid) (width := width) (modes := modes) (seed := seed + 17 * k) (hModes := hModes))
        (blocksSeq (grid := grid) (width := width) (modes := modes) (blocks := k) (seed := seed) (hModes := hModes))

/--
Real-valued (dense-DFT) FNO model constructor.

Input:  length-`grid` vector
Output: length-`grid` vector
-/
def model
    (grid width modes blocks : Nat)
    (seed : Nat := 0)
    (hModes : 2 * modes ≤ grid) :
    Seq (vec grid) (vec grid) :=
  let reshapeIn := _root_.Runtime.Autograd.TorchLean.NN.FNO1D.reshapeVecToMat1 (grid := grid)
  let lift :=
    _root_.Runtime.Autograd.TorchLean.NN.FNO1D.matAffine (grid := grid) (inC := 1) (outC := width)
      (seedW := seed) (seedB := seed + 1)
  let proj :=
    _root_.Runtime.Autograd.TorchLean.NN.FNO1D.matAffine (grid := grid) (inC := width) (outC := 1)
      (seedW := seed + 1000) (seedB := seed + 1001)
  let reshapeOut := _root_.Runtime.Autograd.TorchLean.NN.FNO1D.reshapeMatToVec1 (grid := grid)
  Seq.cons reshapeIn <|
    Seq.cons lift <|
      blocksSeq (grid := grid) (width := width) (modes := modes) (blocks := blocks) (seed := seed) (hModes := hModes) >>>
        Seq.cons proj (Seq.cons reshapeOut (.id (vec grid)))


end Real

end FNO1D

end NN
end TorchLean
end Autograd
end Runtime
