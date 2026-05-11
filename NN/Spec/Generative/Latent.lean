/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorOps
public import NN.Spec.Layers.Loss

/-!
# Latent-variable generative helpers

This module contains the shared spec-layer vocabulary for latent generative models:

- continuous latent variables, as used in variational autoencoders (VAEs);
- discrete codebook latents, as used in vector-quantized VAEs (VQ-VAEs); and
- small total scalar/tensor helpers that keep model files focused on architecture.

The definitions are intentionally **model-agnostic**.  A VAE, VQ-VAE, latent diffusion model, or
normalizing-flow model can all reuse these primitives without committing to a particular backbone.

References:
- Kingma and Welling (2014), "Auto-Encoding Variational Bayes" (VAE).
- Rezende, Mohamed, and Wierstra (2014), "Stochastic Backpropagation and Approximate Inference".
- van den Oord, Vinyals, and Kavukcuoglu (2017), "Neural Discrete Representation Learning"
  (VQ-VAE).
-/

@[expose] public section

namespace Generative.Latent

open Spec
open Tensor

variable {α : Type} [Context α]

/-- Elementwise exponential, useful for log-variance parameterizations. -/
def expTensor {s : Shape} (x : Tensor α s) : Tensor α s :=
  mapSpec MathFunctions.exp x

/-- Elementwise `0.5 * x`, written as a tensor helper to make VAE equations readable. -/
def halfTensor {s : Shape} (x : Tensor α s) : Tensor α s :=
  scaleSpec x Numbers.pointfive

/--
Diagonal-Gaussian reparameterization:

`z = μ + exp(0.5 * logσ²) ⊙ ε`.

This is the spec-level form of the VAE reparameterization trick.  The noise `ε` is explicit, so the
function stays pure and deterministic; runtime examples can supply deterministic or random noise.
-/
def reparameterizeDiag {latent : Shape}
    (mu logvar eps : Tensor α latent) : Tensor α latent :=
  let std := expTensor (halfTensor logvar)
  mu + mulSpec std eps

/--
Mean KL term for a diagonal Gaussian posterior against a standard normal prior.

For each latent coordinate:

`KL(N(μ, σ²) || N(0,1)) = 0.5 * (exp(logσ²) + μ² - 1 - logσ²)`.

We return the mean across the latent shape, matching TorchLean's existing loss convention.
-/
def diagonalGaussianKlToStandard
    {latent : Shape} (mu logvar : Tensor α latent) : α :=
  let var := expTensor logvar
  let mu2 := mulSpec mu mu
  let ones := fill (α := α) (1 : α) latent
  let per := var + mu2 - ones - logvar
  Numbers.pointfive * Spec.meanOver (s := latent) (Spec.toScalarSpec per)

/-- A finite codebook for vector-quantized latent models. -/
structure Codebook (α : Type) (numCodes : Nat) (latent : Shape) [Context α] where
  /-- Embedding vector for each code index. -/
  embedding : Fin numCodes → Tensor α latent

/--
Quantize by an explicit code index.

Nearest-neighbor lookup is usually how VQ-VAE chooses this index during execution.  The spec keeps
the index explicit so proofs and verifiers can reason about a fixed code assignment without
depending on an argmin/tie-breaking policy.
-/
def quantizeAt {numCodes : Nat} {latent : Shape}
    (book : Codebook α numCodes latent) (idx : Fin numCodes) : Tensor α latent :=
  book.embedding idx

end Generative.Latent

