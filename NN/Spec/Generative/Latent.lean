/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

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

open Spec TorchLean
open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- Elementwise exponential, useful for log-variance parameterizations. -/
def expTensor {s : Shape} (x : Tensor α s) : Tensor α s :=
  mapSpec MathFunctions.exp x

/-- Elementwise $\tfrac12x$, written as a tensor helper to make VAE equations readable. -/
def halfTensor {s : Shape} (x : Tensor α s) : Tensor α s :=
  scaleSpec x (1 / 2)

/--
Diagonal-Gaussian reparameterization:

$$
z=\mu+\exp\!\left(\tfrac12\log\sigma^2\right)\odot\varepsilon.
$$

This is the spec-level form of the VAE reparameterization trick. The noise $\varepsilon$ is
explicit, so the
function stays pure and deterministic; runtime examples can supply deterministic or random noise.
-/
def reparameterizeDiag {latent : Shape}
    (mu logvar eps : Tensor α latent) : Tensor α latent :=
  let std := expTensor (halfTensor logvar)
  mu + mulSpec std eps

/--
Mean KL term for a diagonal Gaussian posterior against a standard normal prior.

For each latent coordinate:

$$
D_{\mathrm{KL}}\!\left(\mathcal{N}(\mu,\sigma^2)\,\middle\|\,\mathcal{N}(0,1)\right)
=\tfrac12\left(\exp(\log\sigma^2)+\mu^2-1-\log\sigma^2\right).
$$

We return the mean across the latent shape, matching TorchLean's existing loss convention.
-/
def diagonalGaussianKlToStandard
    {latent : Shape} (mu logvar : Tensor α latent) : α :=
  let var := expTensor logvar
  let mu2 := mulSpec mu mu
  let ones := Tensor.full (α := α) latent (1 : α)
  let per := var + mu2 - ones - logvar
  (1 / 2) * Spec.meanOver (s := latent) (Spec.toScalarSpec per)

/-- A finite codebook for vector-quantized latent models. -/
structure Codebook (α : Type) (numCodes : Nat) (latent : Shape) [TorchLean.Storage α]
    [Context α] where
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
