/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Common
public import NN.API.Runtime
public import NN.Runtime.Autograd.Torch.Utils

import Mathlib.Algebra.Order.Algebra

/-!
# Synthetic Samples and Small Datasets

This module provides small deterministic data generators used by examples and tests:
- 2D tabular grids (`grid2`, `linspace`)
- simple regression/classification sample builders
- a small "band image" generator (2D patterns) for CHW examples

These helpers are simple and in-memory. They keep tutorial code focused on models and verification
rather than data-loading infrastructure.
-/

@[expose] public section


namespace NN
namespace API

namespace Samples

open Spec

/-! ## Tabular 2D -/

/-- 2D affine synthetic target function generator (re-exported from the runtime helper module). -/
abbrev affine2 := _root_.Runtime.Autograd.Torch.Samples.affine2

/-- A length-1 float vector tensor. -/
def vec1Float (y : Float) : Spec.Tensor Float (.dim 1 .scalar) :=
  _root_.Runtime.Autograd.Torch.Samples.vec1 y

/-- A length-2 float vector tensor. -/
def vec2Float (x1 x2 : Float) : Spec.Tensor Float (.dim 2 .scalar) :=
  _root_.Runtime.Autograd.Torch.Samples.vec2 x1 x2

/--
Cartesian product of two float vectors (batched tensor of points).

`grid2 xs ys` produces a tensor `X : (m*n, 2)` containing all pairs `(x, y)` with:
- `x` taken from `xs : (m,)`
- `y` taken from `ys : (n,)`

Ordering is row-major: for each `x` in `xs` (outer loop), we sweep all `y` in `ys` (inner loop).

PyTorch analogue: `torch.cartesian_prod(xs, ys)` (up to shape).
-/
def grid2 {m n : Nat}
    (xs : Spec.Tensor Float (.dim m .scalar)) (ys : Spec.Tensor Float (.dim n .scalar)) :
    Spec.Tensor Float (.dim (m * n) (.dim 2 .scalar)) :=
  Spec.Tensor.dim (fun ij =>
    let i : Fin m := ij.divNat (m := m) (n := n)
    let j : Fin n := ij.modNat (m := m) (n := n)
    let x : Float := _root_.Spec.Tensor.toScalar (_root_.Spec.get xs i)
    let y : Float := _root_.Spec.Tensor.toScalar (_root_.Spec.get ys j)
    vec2Float x y)

/--
Linearly spaced points including endpoints.

`linspace lo hi count` returns a vector tensor of shape `(count,)`:
- empty if `count = 0`
- `[lo]` if `count = 1`
- otherwise `count` points from `lo` to `hi` (inclusive).

PyTorch analogue: `torch.linspace`.
-/
def linspace (lo hi : Float) (count : Nat) : Spec.Tensor Float (.dim count .scalar) :=
  match count with
  | 0 => Spec.Tensor.dim (fun i => nomatch i)
  | 1 => Spec.Tensor.dim (fun _ => Spec.Tensor.scalar lo)
  | n + 2 =>
      let denom := Float.ofNat (n + 1)
      Spec.Tensor.dim (fun i =>
        let t := (Float.ofNat i.1) / denom
        Spec.Tensor.scalar (lo + t * (hi - lo)))

/-- Rectangular grid over `[xLo, xHi] x [yLo, yHi]`. -/
def grid2Rect (xLo xHi yLo yHi : Float) (xCount yCount : Nat) :
    Spec.Tensor Float (.dim (xCount * yCount) (.dim 2 .scalar)) :=
  grid2 (linspace xLo xHi xCount) (linspace yLo yHi yCount)

/-- Square grid over `[lo, hi] x [lo, hi]`. -/
def grid2Square (lo hi : Float) (count : Nat) :
    Spec.Tensor Float (.dim (count * count) (.dim 2 .scalar)) :=
  grid2Rect lo hi lo hi count count

/--
Compute 2D→1D regression targets for a batched grid.

Input `X` has shape `(n,2)` and the output `Y` has shape `(n,1)`.
-/
def regression2to1Float {n : Nat} (X : Spec.Tensor Float (.dim n (.dim 2 .scalar)))
    (f : Float → Float → Float) : Spec.Tensor Float (.dim n (.dim 1 .scalar)) :=
  Spec.Tensor.dim (fun i =>
    let x : Spec.Tensor Float (.dim 2 .scalar) := Spec.get X i
    let x1 : Float := _root_.Spec.Tensor.toScalar (_root_.Spec.get x ⟨0, by decide⟩)
    let x2 : Float := _root_.Spec.Tensor.toScalar (_root_.Spec.get x ⟨1, by decide⟩)
    vec1Float (f x1 x2))

/-- Casted version of `vec1Float` under an arbitrary scalar semantics `α`. -/
def vec1 {α : Type} [Context α] (cast : Float → α) (y : Float) : Spec.Tensor α (.dim 1 .scalar) :=
  Common.castTensor cast (vec1Float y)

/-- Casted version of `vec2Float` under an arbitrary scalar semantics `α`. -/
def vec2 {α : Type} [Context α] (cast : Float → α) (x1 x2 : Float) : Spec.Tensor α (.dim 2 .scalar)
  :=
  Common.castTensor cast (vec2Float x1 x2)

/-! ## Image 2D (Synthetic CHW Pixels) -/

namespace Image2D

/--
Which axis a synthetic "band" varies along.

We use this to build small labeled image datasets without depending on any external image format.
-/
inductive Axis
  | row
  | col
  deriving Repr, DecidableEq

/-- Human-readable axis name (useful in example titles / sample names). -/
def Axis.name : Axis → String
  | .row => "horizontal"
  | .col => "vertical"

/--
Render a `CHW c h w` image tensor from a pixel function.

The pixel function is indexed by `Fin` indices that are guaranteed to be in-bounds.
-/
def renderCHWFloat (c h w : Nat) (pixel : Fin c → Fin h → Fin w → Float) :
    Spec.Tensor Float (NN.Tensor.Shape.CHW c h w) :=
  Spec.Tensor.dim (fun ic =>
    Spec.Tensor.dim (fun ir =>
      Spec.Tensor.dim (fun iw =>
        Spec.Tensor.scalar (pixel ic ir iw))))

/--
Binary image tensor renderer in CHW layout.

Pixels satisfying `on c row col` get `onValue`; the rest get `offValue`.
-/
def binaryCHWFloat (c h w : Nat) (on : Fin c → Fin h → Fin w → Bool)
    (onValue : Float := 1.0) (offValue : Float := 0.0) : Spec.Tensor Float (NN.Tensor.Shape.CHW c h
      w) :=
  renderCHWFloat c h w (fun ic ir iw => if on ic ir iw then onValue else offValue)

/--
Render a thick horizontal or vertical band into an `h x w` image tensor.

This is used to create small classification datasets where the label is "horizontal" vs "vertical".

Returns a single-channel `CHW 1 h w` image tensor (channel-first, PyTorch style).
-/
def bandCHWFloat (h w : Nat) (axis : Axis) (offset : Nat) (thickness : Nat := 2)
    (onValue : Float := 1.0) (offValue : Float := 0.0) : Spec.Tensor Float (NN.Tensor.Shape.CHW 1 h
      w) :=
  binaryCHWFloat 1 h w
    (match axis with
    | .row => fun _ r _ => offset ≤ r.1 ∧ r.1 < offset + thickness
    | .col => fun _ _ c => offset ≤ c.1 ∧ c.1 < offset + thickness)
    onValue offValue

/-- Label metadata for a class of synthetic band images. -/
structure BandClass where
  /-- Axis. -/
  axis : Axis
  /-- Label. -/
  label : Nat
  /-- Name. -/
  name : String

/-- Convenience constructor for a "vertical band" class. -/
def verticalClass (label : Nat := 0) (name : String := "vertical") : BandClass :=
  { axis := .col, label := label, name := name }

/-- Convenience constructor for a "horizontal band" class. -/
def horizontalClass (label : Nat := 1) (name : String := "horizontal") : BandClass :=
  { axis := .row, label := label, name := name }

/--
Generate labeled band samples for a list of classes and offsets.

This produces a list of `(x, label)` pairs where `x` is a typed `CHW 1 h w` tensor.
-/
def bandDatasetCHWFloat (h w : Nat) (classes : List BandClass) (offsets : List Nat) (thickness : Nat
  := 2) :
    List (Spec.Tensor Float (NN.Tensor.Shape.CHW 1 h w) × Nat) :=
  classes.foldr
    (fun cls acc =>
      offsets.map (fun offset => (bandCHWFloat h w cls.axis offset thickness, cls.label)) ++ acc)
    []

/--
Like `bandDatasetCHWFloat`, but keep a human-readable name for each sample.

Each entry is `(name, x, label)` where `x` is a typed `CHW 1 h w` tensor.
-/
def namedBandSamplesCHWFloat (h w : Nat) (specs : List (BandClass × Nat)) (thickness : Nat := 2) :
    List (String × Spec.Tensor Float (NN.Tensor.Shape.CHW 1 h w) × Nat) :=
  specs.map (fun (cls, offset) =>
    (s!"{cls.name}-{offset}", bandCHWFloat h w cls.axis offset thickness, cls.label))

end Image2D

/-! ## Labels and Packing -/

/-- One-hot encode a label as a float vector of shape `Vec classes`. -/
def oneHotFloat (classes label : Nat) : Spec.Tensor Float (NN.Tensor.Shape.Vec classes) :=
  NN.Tensor.oneHotNat (α := Float) classes label

/-- Casted version of `oneHotFloat`. -/
def oneHot {α : Type} [Context α] (cast : Float → α) (classes label : Nat) :
    Spec.Tensor α (NN.Tensor.Shape.Vec classes) :=
  Common.castTensor cast (oneHotFloat classes label)

/--
Convert `(x, label)` pairs into `(x, oneHot(label))` pairs.

This is a pure preprocessing step that keeps the data in-memory.
-/
def classification {α : Type} [Context α] {σ : Spec.Shape}
    (cast : Float → α) (classes : Nat) (xs : List (Spec.Tensor Float σ × Nat)) :
    List (Spec.Tensor α σ × Spec.Tensor α (NN.Tensor.Shape.Vec classes)) :=
  xs.map (fun (xF, label) => (Common.castTensor cast xF, oneHot cast classes label))

/--
Pack `(x, y)` tensor pairs into TorchLean supervised `TList` samples.

This is the common sample representation used by the training helpers.
-/
def supervised {α : Type} [Context α] {σ τ : Spec.Shape}
    (cast : Float → α) (xs : List (Spec.Tensor Float σ × Spec.Tensor Float τ)) :
    List (TorchLean.TList α [σ, τ]) :=
  xs.map (fun (xF, yF) =>
    TorchLean.tlist2 (Common.castTensor cast xF) (Common.castTensor cast yF))

/-- Convert `(x, label)` pairs into TorchLean `TList` samples with one-hot targets. -/
def labeled {α : Type} [Context α] {σ : Spec.Shape}
    (cast : Float → α) (classes : Nat) (xs : List (Spec.Tensor Float σ × Nat)) :
    List (TorchLean.TList α [σ, NN.Tensor.Shape.Vec classes]) :=
  (classification (α := α) (σ := σ) cast classes xs).map (fun (x, y) =>
    TorchLean.tlist2 x y)

/-! ## CHW Parsing -/

/--
Interpret a flat list of floats as an image tensor of shape `CHW c h w`.

Fails if `xs.length ≠ c*h*w`.
-/
def imageCHWFloat (c h w : Nat) (xs : List Float) :
    Except String (Spec.Tensor Float (NN.Tensor.Shape.CHW c h w)) := do
  let t ← NN.Tensor.tensorND (α := Float) [c, h, w] xs
  pure (by simpa [NN.Tensor.shapeOfDims, NN.Tensor.Shape.CHW] using t)

/-- Casted version of `imageCHWFloat`. -/
def imageCHW {α : Type} [Context α] (cast : Float → α) (c h w : Nat) (xs : List Float) :
    Except String (Spec.Tensor α (NN.Tensor.Shape.CHW c h w)) := do
  let tF ← imageCHWFloat c h w xs
  pure (Common.castTensor cast tF)

end Samples

end API
end NN
