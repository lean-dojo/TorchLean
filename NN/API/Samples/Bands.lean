/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Samples

/-!
# Small Band-Image Dataset (4×4)

Several TorchLean tutorials use a compact 4×4 image classification task:

- class `0`: a vertical band
- class `1`: a horizontal band

This module packages that synthetic dataset in one place so examples can stay focused on models and
training rather than data-loading infrastructure.
-/

@[expose] public section

namespace NN
namespace API
namespace Samples

open Spec

namespace Bands

/-! ### Class Spec -/

/-- Canonical label set for the band dataset: vertical ↦ `0`, horizontal ↦ `1`. -/
def classes : List Image2D.BandClass :=
  [ Image2D.verticalClass 0
  , Image2D.horizontalClass 1
  ]

/-! ### Typed Tensors (Tensor-First) -/

/-- Canonical image shape for the band dataset (single-channel 4×4). -/
abbrev shape : Spec.Shape := NN.Tensor.Shape.CHW 1 4 4

/-- Training set samples: a small list of `(x, label)` pairs. -/
def trainCHWFloat : List (Spec.Tensor Float shape × Nat) :=
  Image2D.bandDatasetCHWFloat 4 4 classes [0, 1, 2]

/-- Probe set for reporting: `(name, x, expectedLabel)` triples. -/
def probesCHWFloat : List (String × Spec.Tensor Float shape × Nat) :=
  Image2D.namedBandSamplesCHWFloat 4 4
    [ (Image2D.verticalClass 0, 1)
    , (Image2D.verticalClass 0, 2)
    , (Image2D.horizontalClass 1, 1)
    , (Image2D.horizontalClass 1, 2)
    ]

/-- Cast `trainCHWFloat` into an arbitrary scalar backend `α`. -/
def trainCHW {α : Type} [Context α] (cast : Float → α) : List (Spec.Tensor α shape × Nat) :=
  trainCHWFloat.map (fun (xF, y) => (Common.castTensor cast xF, y))

/-- Cast `probesCHWFloat` into an arbitrary scalar backend `α`. -/
def probesCHW {α : Type} [Context α] (cast : Float → α)
    (probes : List (String × Spec.Tensor Float shape × Nat) := probesCHWFloat) :
    List (String × Spec.Tensor α shape × Nat) :=
  probes.map (fun (name, xF, y) => (name, Common.castTensor cast xF, y))

end Bands

end Samples
end API
end NN
