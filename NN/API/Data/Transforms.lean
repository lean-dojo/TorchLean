/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Data

/-!
# Dataset and Sample Transforms (Torchvision-Style)

This module provides a small transform library inspired by `torchvision.transforms`:
- composable pure transforms (`Compose`, `Lambda`)
- dataset mapping helpers
- common tensor/sample normalization utilities

## PyTorch Mapping

- `torchvision.transforms`: `https://pytorch.org/vision/stable/transforms.html`
- `torch.utils.data.Dataset` map-style datasets: `https://pytorch.org/docs/stable/data.html`

TorchLean difference: transforms are pure functions over *typed* tensors/samples, so shape mistakes
are caught by the typechecker rather than at runtime.
-/

@[expose] public section


namespace NN
namespace API
namespace Data
namespace Transforms

/--
Torchvision-style transform composition.

Applies transforms left-to-right:

`Compose [f, g, h] x = h (g (f x))`.
-/
def Compose {a : Type} (ts : List (a → a)) : a → a :=
  fun x => ts.foldl (fun acc f => f acc) x

/-- Torchvision-style "Lambda" transform wrapper. -/
def Lambda {a : Type} (f : a → a) : a → a :=
  f

/-- Compose two pure transforms. -/
def compose {a b c : Type} (g : b → c) (f : a → b) : a → c :=
  fun x => g (f x)

/-- Apply a pure transform to every element of a dataset. -/
def onDataset {a b : Type} (f : a → b) (ds : API.Data.Dataset a) : API.Data.Dataset b :=
  _root_.Runtime.Autograd.Train.Dataset.map f ds

/-- Map a tensor elementwise. -/
def mapTensor {α : Type} {s : Spec.Shape} (f : α → α) (x : Spec.Tensor α s) : Spec.Tensor α s :=
  Spec.mapTensor f x

/-- Normalize any tensor elementwise: `(x - mean) / std`. -/
def normalizeTensor {α : Type} [Sub α] [Div α] {s : Spec.Shape} (mean std : α)
    (x : Spec.Tensor α s) : Spec.Tensor α s :=
  mapTensor (fun v => (v - mean) / std) x

/-- Float-literal normalization helper for runtime scalar backends. -/
def normalizeTensorF {α : Type} [API.Runtime.Scalar α] [Sub α] [Div α] {s : Spec.Shape}
    (mean std : Float) (x : Spec.Tensor α s) : Spec.Tensor α s :=
  normalizeTensor (α := α) (s := s) (API.Runtime.ofFloat (α := α) mean) (API.Runtime.ofFloat (α :=
    α) std) x

/-- Transform labels in `(sample, label)` datasets. -/
def mapLabels {a : Type} (f : Nat → Nat) (xs : List (a × Nat)) : List (a × Nat) :=
  xs.map (fun (x, y) => (x, f y))

/-- Transform samples in `(sample, label)` datasets. -/
def mapSamples {a b : Type} (f : a → b) (xs : List (a × Nat)) : List (b × Nat) :=
  xs.map (fun (x, y) => (f x, y))

/-- Apply a sample transform to a labeled dataset. -/
def onSamples {a b : Type} (f : a → b) (ds : API.Data.Dataset (a × Nat)) : API.Data.Dataset (b ×
  Nat) :=
  onDataset (fun (x, y) => (f x, y)) ds

/-- Apply a label transform to a labeled dataset. -/
def onLabels {a : Type} (f : Nat → Nat) (ds : API.Data.Dataset (a × Nat)) : API.Data.Dataset (a ×
  Nat) :=
  onDataset (fun (x, y) => (x, f y)) ds

/-- Transform the input component of a supervised TorchLean sample `TList α [σ, τ]`. -/
def onSupervisedInput {α : Type} {σ τ : Spec.Shape}
    (f : Spec.Tensor α σ → Spec.Tensor α σ) :
    API.sample.Supervised α σ τ → API.sample.Supervised α σ τ :=
  API.sample.mapX (α := α) (σ := σ) (τ := τ) f

/-- Transform the target component of a supervised TorchLean sample `TList α [σ, τ]`. -/
def onSupervisedTarget {α : Type} {σ τ : Spec.Shape}
    (f : Spec.Tensor α τ → Spec.Tensor α τ) :
    API.sample.Supervised α σ τ → API.sample.Supervised α σ τ :=
  API.sample.mapY (α := α) (σ := σ) (τ := τ) f

/-- Apply an input transform over a supervised TorchLean dataset. -/
def onSupervisedDatasetInput {α : Type} {σ τ : Spec.Shape}
    (f : Spec.Tensor α σ → Spec.Tensor α σ)
    (ds : API.Data.Dataset (API.sample.Supervised α σ τ)) :
    API.Data.Dataset (API.sample.Supervised α σ τ) :=
  onDataset (onSupervisedInput (α := α) (σ := σ) (τ := τ) f) ds

end Transforms
end Data
end API
end NN
