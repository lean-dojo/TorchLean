/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public
public import NN.API.Tensor.Views

/-!
# Generative Model Helpers (API)

Small config-style constructors for runnable generative examples.

These are intentionally vector models: examples can flatten images, train the model, and later swap
in convolutional encoders/decoders without changing the command-line/data-loading surface.
-/

@[expose] public section

namespace NN
namespace API

open Spec Tensor

namespace nn
namespace models

/-- Shared dimensions for vector generative examples. -/
structure VectorGenerativeConfig where
  batch : Nat
  dataDim : Nat
  hiddenDim : Nat
  latentDim : Nat
deriving Repr

/-- Convenience constructor for compact vector generative models. -/
def vectorGenerativeConfig (batch dataDim hiddenDim latentDim : Nat) : VectorGenerativeConfig :=
  { batch, dataDim, hiddenDim, latentDim }

/--
Small default config used by the runnable image-vector examples.

The input dimension is intentionally a prefix of a flattened image, not a full image decoder. That
keeps examples fast while still exercising real data, batched training, and generative model
constructors.
-/
def compactImageConfig (batch : Nat := 4) (dataDim : Nat := 128)
    (hiddenDim : Nat := 64) (latentDim : Nat := 32) : VectorGenerativeConfig :=
  vectorGenerativeConfig batch dataDim hiddenDim latentDim

abbrev vectorDataShape (cfg : VectorGenerativeConfig) : Shape :=
  NN.Tensor.Shape.Mat cfg.batch cfg.dataDim

abbrev vectorLatentShape (cfg : VectorGenerativeConfig) : Shape :=
  NN.Tensor.Shape.Mat cfg.batch cfg.latentDim

/--
β-VAE-style supervised output shape.

Rows contain:
- reconstruction, length `dataDim`;
- latent mean proxy, length `latentDim`;
- latent log-variance proxy, length `latentDim`.

The runnable example trains this compact target with MSE, which is a practical path for the
runtime. The formal VAE ELBO/KL objective lives in `NN.Spec.Models.Vae` and
`NN.MLTheory.Generative.Latent.VAE`.
-/
abbrev vectorVaeOutShape (cfg : VectorGenerativeConfig) : Shape :=
  NN.Tensor.Shape.Mat cfg.batch (cfg.dataDim + 2 * cfg.latentDim)

/--
Flatten each sample in a batch and keep the first `cfg.dataDim` entries.

This is useful for quick image-vector experiments: a model can train on a small typed vector view of
an image without every example needing to carry its own flattening proof adapters.
-/
def flattenBatchPrefix {α : Type} [Inhabited α] (cfg : VectorGenerativeConfig) {source : Shape}
    (hData : cfg.dataDim ≤ Shape.size source)
    (x : Spec.Tensor α (.dim cfg.batch source)) : Spec.Tensor α (vectorDataShape cfg) :=
  _root_.NN.API.tensor.flattenBatchPrefix cfg.batch cfg.dataDim hData x

/-- Supervised reconstruction sample: target equals input. -/
def reconstructionSample {α : Type} (cfg : VectorGenerativeConfig)
    (x : Spec.Tensor α (vectorDataShape cfg)) :
    sample.Supervised α (vectorDataShape cfg) (vectorDataShape cfg) :=
  sample.mk x x

/--
Target for compact VAE-style examples.

Rows contain the reconstruction target followed by zero mean/log-variance proxy channels.
-/
def zeroLatentStatsTarget (cfg : VectorGenerativeConfig)
    (x : Spec.Tensor Float (vectorDataShape cfg)) : Spec.Tensor Float (vectorVaeOutShape cfg) :=
  Spec.Tensor.dim (fun bi =>
    let row := Spec.getAtSpec x bi
    Spec.Tensor.dim (fun j =>
      let v :=
        if h : j.val < cfg.dataDim then
          Spec.Tensor.toScalar (Spec.get row ⟨j.val, h⟩)
        else
          0.0
      Spec.Tensor.scalar v))

/-- Supervised compact VAE sample: image reconstruction plus zero latent-stat targets. -/
def vaeSample (cfg : VectorGenerativeConfig)
    (x : Spec.Tensor Float (vectorDataShape cfg)) :
    sample.Supervised Float (vectorDataShape cfg) (vectorVaeOutShape cfg) :=
  sample.mk x (zeroLatentStatsTarget cfg x)

/-- Deterministic matrix-valued pseudo-random tensor in `[lo, hi)`. -/
def vectorNoise (batch dim seed salt : Nat) (lo hi : Float) :
    Spec.Tensor Float (NN.Tensor.Shape.Mat batch dim) :=
  Spec.Tensor.dim (fun bi =>
    Spec.Tensor.dim (fun j =>
      let k := bi.val * dim + j.val
      let raw := (seed * 1103515245 + k * 12345 + salt) % 997
      let u := Float.ofNat raw / 997.0
      Spec.Tensor.scalar (lo + (hi - lo) * u)))

/-- Deterministic latent noise for generator examples. -/
def latentNoise (cfg : VectorGenerativeConfig) (seed : Nat) :
    Spec.Tensor Float (vectorLatentShape cfg) :=
  vectorNoise cfg.batch cfg.latentDim seed 17 (-1.0) 1.0

/-- Deterministic data-shaped noise for discriminator examples. -/
def dataNoise (cfg : VectorGenerativeConfig) (seed : Nat) :
    Spec.Tensor Float (vectorDataShape cfg) :=
  vectorNoise cfg.batch cfg.dataDim seed 91 0.0 1.0

/-- Constant discriminator/critic target. -/
def scoreTarget (cfg : VectorGenerativeConfig) (value : Float) :
    Spec.Tensor Float (NN.Tensor.Shape.Mat cfg.batch 1) :=
  Spec.Tensor.dim (fun _ => Spec.Tensor.dim (fun _ => Spec.Tensor.scalar value))

/-- Target score for real samples. -/
def onesScore (cfg : VectorGenerativeConfig) : Spec.Tensor Float (NN.Tensor.Shape.Mat cfg.batch 1) :=
  scoreTarget cfg 1.0

/-- Target score for generated or noise samples. -/
def zerosScore (cfg : VectorGenerativeConfig) : Spec.Tensor Float (NN.Tensor.Shape.Mat cfg.batch 1) :=
  scoreTarget cfg 0.0

/-- Autoencoder: `x -> hidden -> latent -> hidden -> reconstruction`. -/
def vectorAutoencoder (cfg : VectorGenerativeConfig) :
    nn.M (nn.Sequential (vectorDataShape cfg) (vectorDataShape cfg)) :=
  nn.sequential![
    nn.linear cfg.dataDim cfg.hiddenDim (pfx := NN.Tensor.Shape.Vec cfg.batch),
    nn.relu,
    nn.linear cfg.hiddenDim cfg.latentDim (pfx := NN.Tensor.Shape.Vec cfg.batch),
    nn.relu,
    nn.linear cfg.latentDim cfg.hiddenDim (pfx := NN.Tensor.Shape.Vec cfg.batch),
    nn.relu,
    nn.linear cfg.hiddenDim cfg.dataDim (pfx := NN.Tensor.Shape.Vec cfg.batch),
    nn.sigmoid
  ]

/-- Compact β-VAE-style network producing reconstruction plus latent statistics. -/
def vectorVae (cfg : VectorGenerativeConfig) :
    nn.M (nn.Sequential (vectorDataShape cfg) (vectorVaeOutShape cfg)) :=
  nn.sequential![
    nn.linear cfg.dataDim cfg.hiddenDim (pfx := NN.Tensor.Shape.Vec cfg.batch),
    nn.relu,
    nn.linear cfg.hiddenDim cfg.latentDim (pfx := NN.Tensor.Shape.Vec cfg.batch),
    nn.relu,
    nn.linear cfg.latentDim cfg.hiddenDim (pfx := NN.Tensor.Shape.Vec cfg.batch),
    nn.relu,
    nn.linear cfg.hiddenDim (cfg.dataDim + 2 * cfg.latentDim)
      (pfx := NN.Tensor.Shape.Vec cfg.batch)
  ]

/-- VQ-VAE-style encoder/decoder with a narrow discrete-code proxy bottleneck. -/
def vectorVqVae (cfg : VectorGenerativeConfig) :
    nn.M (nn.Sequential (vectorDataShape cfg) (vectorDataShape cfg)) :=
  nn.sequential![
    nn.linear cfg.dataDim cfg.hiddenDim (pfx := NN.Tensor.Shape.Vec cfg.batch),
    nn.relu,
    nn.linear cfg.hiddenDim cfg.latentDim (pfx := NN.Tensor.Shape.Vec cfg.batch),
    nn.tanh,
    nn.linear cfg.latentDim cfg.hiddenDim (pfx := NN.Tensor.Shape.Vec cfg.batch),
    nn.relu,
    nn.linear cfg.hiddenDim cfg.dataDim (pfx := NN.Tensor.Shape.Vec cfg.batch),
    nn.sigmoid
  ]

/-- Generator `z -> x`. -/
def vectorGanGenerator (cfg : VectorGenerativeConfig) :
    nn.M (nn.Sequential (vectorLatentShape cfg) (vectorDataShape cfg)) :=
  nn.sequential![
    nn.linear cfg.latentDim cfg.hiddenDim (pfx := NN.Tensor.Shape.Vec cfg.batch),
    nn.relu,
    nn.linear cfg.hiddenDim cfg.hiddenDim (pfx := NN.Tensor.Shape.Vec cfg.batch),
    nn.relu,
    nn.linear cfg.hiddenDim cfg.dataDim (pfx := NN.Tensor.Shape.Vec cfg.batch),
    nn.sigmoid
  ]

/-- Discriminator/critic `x -> score`. -/
def vectorGanDiscriminator (cfg : VectorGenerativeConfig) :
    nn.M (nn.Sequential (vectorDataShape cfg) (NN.Tensor.Shape.Mat cfg.batch 1)) :=
  nn.sequential![
    nn.linear cfg.dataDim cfg.hiddenDim (pfx := NN.Tensor.Shape.Vec cfg.batch),
    nn.relu,
    nn.linear cfg.hiddenDim cfg.hiddenDim (pfx := NN.Tensor.Shape.Vec cfg.batch),
    nn.relu,
    nn.linear cfg.hiddenDim 1 (pfx := NN.Tensor.Shape.Vec cfg.batch),
    nn.sigmoid
  ]

end models
end nn

end API
end NN
