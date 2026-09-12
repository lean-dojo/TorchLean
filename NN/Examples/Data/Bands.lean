/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API

/-!
# Synthetic Band Dataset

Several TorchLean examples use a compact 4×4 image classification task:

- class `0`: a vertical band
- class `1`: a horizontal band

This domain module owns both the renderer and the canonical 4×4 dataset. Keeping these definitions
out of `TorchLean.Data.Synthetic` prevents a particular image layout from becoming part of
TorchLean's general tensor and sample abstractions.
-/

@[expose] public section

namespace NN.Examples.Data

open TorchLean

namespace Bands

/-! ## Renderer -/

/-- Spatial axis along which a synthetic band varies. -/
inductive Axis
  | row
  | column
  deriving Repr, DecidableEq

/-- Human-readable class name associated with an axis. -/
def Axis.name : Axis → String
  | .row => "horizontal"
  | .column => "vertical"

/--
Render a channel-first image tensor from an in-bounds scalar function.

This shape is part of this particular dataset, not a restriction on TorchLean tensors or models.
-/
def render (channels height width : Nat)
    (value : Fin channels → Fin height → Fin width → Float) :
    Tensor Float [channels, height, width] :=
  Tensor.stack 0 fun channel =>
    Tensor.stack 0 fun row =>
      Tensor.stack 0 fun column =>
        Tensor.full [] (value channel row column)

/-- Render a binary-valued channel-first tensor from a finite predicate. -/
def renderBinary (channels height width : Nat)
    (selected : Fin channels → Fin height → Fin width → Bool)
    (onValue : Float := 1.0) (offValue : Float := 0.0) :
    Tensor Float [channels, height, width] :=
  render channels height width fun channel row column =>
    if selected channel row column then onValue else offValue

/-- Render a single-channel horizontal or vertical band. -/
def renderBand (height width : Nat) (axis : Axis) (offset : Nat) (thickness : Nat := 2)
    (onValue : Float := 1.0) (offValue : Float := 0.0) :
    Tensor Float [1, height, width] :=
  renderBinary 1 height width
    (match axis with
    | .row => fun _ row _ => offset ≤ row.val ∧ row.val < offset + thickness
    | .column => fun _ _ column => offset ≤ column.val ∧ column.val < offset + thickness)
    onValue offValue

/-! ## Classes and datasets -/

/-- Label metadata for one family of synthetic bands. -/
structure Class where
  /-- Axis occupied by the band. -/
  axis : Axis
  /-- Class label for the two-class band task. -/
  label : Fin 2
  /-- Display name used by reports. -/
  name : String

/-- Construct the vertical-band class. -/
def vertical (name : String := "vertical") : Class :=
  { axis := .column, label := 0, name }

/-- Construct the horizontal-band class. -/
def horizontal (name : String := "horizontal") : Class :=
  { axis := .row, label := 1, name }

/-- Generate `(tensor, label)` samples for every class/offset pair. -/
def samples (height width : Nat) (classes : Array Class) (offsets : Array Nat)
    (thickness : Nat := 2) :
    Array (Tensor Float [1, height, width] × Fin 2) :=
  classes.flatMap fun cls =>
    offsets.map fun offset =>
      (renderBand height width cls.axis offset thickness, cls.label)

/-- Canonical label set for the band dataset: vertical ↦ `0`, horizontal ↦ `1`. -/
def classes : Array Class :=
  #[vertical, horizontal]

/-! ### Typed Tensors (Tensor-First) -/

/-- Canonical image shape for the band dataset (single-channel 4×4). -/
abbrev shape : Shape := [1, 4, 4]

/-- Training set samples as a runtime-sized array of fixed-shape tensors. -/
def trainFloat : Array (Tensor Float shape × Fin 2) :=
  samples 4 4 classes #[0, 1, 2]

/-- Small vertical-versus-horizontal dataset with one-hot class targets. -/
def dataset : Trainer.Dataset shape [2] :=
  TorchLean.Data.fromSamples <| trainFloat.map fun (input, label) =>
    { input, target := Tensor.oneHot (α := Float) 2 label }

end Bands
end NN.Examples.Data
