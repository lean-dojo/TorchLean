/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Reduce

/-!
# Latent-variable generative helpers

Shared spec-layer vocabulary for latent generative models:

- continuous latent variables, as used in variational autoencoders (VAEs), and
- discrete codebook latents, as used in vector-quantized VAEs (VQ-VAEs).

The definitions are **model-agnostic**, so a VAE, VQ-VAE, latent diffusion model or
normalizing-flow model can use these primitives with any backbone.

Both Gaussian helpers are written by hand from Kingma and Welling (2014). `VAE.reparameterize` is
the `z = μ + σ ⊙ ε` of their eq. (10), with `σ = exp(½ log σ²)` because the encoder outputs a
log-variance, as their Gaussian MLP does (Appendix C.2). `VAE.meanKLToStandard` is the closed form
of their Appendix B with the sign flipped. The paper sums it over the `J` latent coordinates and
we divide by `J`, so a 20-coordinate latent gets a twentieth of the paper's KL. Over `ℝ`,
`TorchLean.VAE.meanKLToStandard_eq_sum` proves that the stabilized branches and the clamp give
that closed form.

References:
- Diederik P. Kingma and Max Welling (2014), "Auto-Encoding Variational Bayes", ICLR 2014,
  arXiv:1312.6114, eq. (10) and Appendices B and C.2.
- Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra (2014), "Stochastic Backpropagation
  and Approximate Inference in Deep Generative Models", ICML 2014, arXiv:1401.4082.
- Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu (2017), "Neural Discrete
  Representation Learning" (VQ-VAE), NIPS 2017, arXiv:1711.00937.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

/--
Diagonal-Gaussian reparameterization:

$$
z=\mu+\exp\!\left(\tfrac12\log\sigma^2\right)\odot\varepsilon.
$$

It is the spec-level form of the VAE reparameterization trick. The noise $\varepsilon$ is an
argument, so the function is pure and deterministic, and runtime examples can supply fixed or
random noise.
-/
def VAE.reparameterize {latent : Shape}
    (mu logvar eps : Tensor α latent) : Tensor α latent :=
  let std := Tensor.exp (Tensor.scale logvar (1 / 2))
  mu + (std * eps)

/--
Mean KL term for a diagonal Gaussian posterior against a standard normal prior.

For each latent coordinate:

$$
D_{\mathrm{KL}}\!\left(\mathcal{N}(\mu,\sigma^2)\,\middle\|\,\mathcal{N}(0,1)\right)
=\tfrac12\left(\exp(\log\sigma^2)+\mu^2-1-\log\sigma^2\right).
$$

The result is the mean across the latent shape, matching TorchLean's mean-reduction loss convention.
For real-like scalars, the small-log-variance branch avoids subtracting one from a nearly unit
exponential, and a clamp at zero removes negative roundoff from each coordinate's term (it bounds no
error, and a NaN term passes through). Inputs that fail the real-like guard, such as complex values,
use the unclamped textbook expression.
-/
def VAE.meanKLToStandard
    {latent : Shape} (mu logvar : Tensor α latent) : α :=
  let per := Tensor.zipWith (fun mean logVariance =>
    let magnitude := MathFunctions.abs mean
    let varianceMagnitude := MathFunctions.abs logVariance
    -- Complex magnitudes need not equal either signed input, so do not clamp that expression.
    if (magnitude == mean || magnitude == -mean) &&
        (varianceMagnitude == logVariance || varianceMagnitude == -logVariance) then
      let varianceTerm := if varianceMagnitude < 1 then
          -- exp(v) - 1 - v = 2 sinh²(v/2) + sinh(v) - v.
          -- The quadratic term survives when exp(v) rounds to one near the prior.
          let halfSinh := MathFunctions.sinh (logVariance / 2)
          2 * halfSinh * halfSinh + (MathFunctions.sinh logVariance - logVariance)
        else MathFunctions.exp logVariance - 1 - logVariance
      let value := mean * mean + varianceTerm
      if value == value then max 0 value else value
    else
      MathFunctions.exp logVariance + mean * mean - 1 - logVariance) mu logvar
  (1 / 2) * (Tensor.sum per / (Tensor.meanDenominator latent : α))

/-- A finite codebook for vector-quantized latent models. -/
structure Codebook (α : Type) (numCodes : Nat) (latent : Shape) [Storage α] where
  /-- Embedding vector for each code index. -/
  embedding : Fin numCodes → Tensor α latent

/--
Look up the codebook embedding at a given code index.

VQ-VAE usually chooses the index by nearest-neighbor lookup during execution. Taking it as an
argument lets proofs and verifiers reason about a fixed code assignment without an argmin or
tie-breaking policy.
-/
def Codebook.quantizeAt {numCodes : Nat} {latent : Shape}
    (book : Codebook α numCodes latent) (idx : Fin numCodes) : Tensor α latent :=
  book.embedding idx

end TorchLean
