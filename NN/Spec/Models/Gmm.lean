/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Activation
public import NN.Spec.Models.CommonHelpers

/-!
# Gaussian Mixture Model (GMM) (spec model)

This file defines a basic GMM with `nComponents` multivariate Gaussians over `nFeatures`:

- mixing weights `π : nComponents`
- means `μ : nComponents × nFeatures`
- covariances `Σ : nComponents × nFeatures × nFeatures`

`gmm_forward_spec` computes **per-component** log-probabilities for a single input:

`log π_k + log N(x | μ_k, Σ_k)`

PyTorch analogies:

- `torch.distributions.MultivariateNormal` for `N(x | μ, Σ)`,
- `torch.distributions.MixtureSameFamily` for mixture distributions,
- `torch.softmax` for turning per-component log-probabilities into responsibilities.

Implementation note: determinants/inverses are defined via `NN.Spec.Models.CommonHelpers`.
Those are intended for small feature dimensions and proof/reference usage, not high‑performance
clustering on large matrices.

References (background, not required to read the code):

- Dempster, Laird, Rubin (1977), "Maximum Likelihood from Incomplete Data via the EM Algorithm":
  https://www.jstor.org/stable/2984875
- Bishop (2006), "Pattern Recognition and Machine Learning", Chapter 9 (Mixture Models and EM):
  https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/
-/

public section


namespace Spec

open Tensor

variable {α : Type} [Context α]

/-! ## Parameters -/

/-- Parameters of a Gaussian mixture model (GMM). -/
structure GMMSpec (α : Type) (nComponents nFeatures : Nat) where
  /-- Mixing weights `π_k` (typically nonnegative and summing to `1`). -/
  weights : Tensor α (.dim nComponents .scalar)
  /-- Component means `μ_k`. -/
  means : Tensor α (.dim nComponents (.dim nFeatures .scalar))
  /-- Component covariance matrices `Σ_k` (typically symmetric positive definite). -/
  covariances : Tensor α (.dim nComponents (.dim nFeatures (.dim nFeatures .scalar)))

/-- Per-component log-probabilities for a single input.

Given `x : ℝ^d`, each component contributes:

`log π_k - 1/2 * ( (x-μ_k)^T Σ_k^{-1} (x-μ_k) + log det Σ_k + d * log(2π) )`

This is the natural "logit vector" for responsibilities.  If you want posterior probabilities
`P(z=k | x)`, apply `gmm_expectation_spec` (a last-axis softmax).

PyTorch analogy: the returned vector is like per-component `log_prob` values before the final
mixture `logsumexp`.
-/
def gmmForwardSpec {nComponents nFeatures : Nat}
  (m : GMMSpec α nComponents nFeatures)
  (input : Tensor α (.dim nFeatures .scalar)) :
  Tensor α (.dim nComponents .scalar) :=
  match m.weights, m.means, m.covariances with
  | Tensor.dim weights, Tensor.dim means, Tensor.dim covariances =>
    Tensor.dim (fun k =>
      let weight := weights k
      let mean := means k
      let covariance := covariances k

      -- Compute the difference vector (x - μ)
      let diff := subSpec input mean

      -- Compute the log-likelihood for the multivariate Gaussian
      let log_likelihood :=
        -- Determinant/inverse are only well-behaved when the covariance is positive definite.
        -- TorchLean keeps that as an explicit modeling assumption, similar to how PyTorch can
        -- error/NaN if you pass an invalid covariance to `MultivariateNormal`.
        let det := determinantSpec covariance
        let cov_inv := inverseSpec covariance

        -- (x - μ)ᵀ Σ⁻¹ (x - μ)
        let temp := matVecMulSpec cov_inv diff
        let quadratic_form := dotSpec diff temp

        -- 1/2 * (d * log(2π) + log det Σ)
        let d := nFeatures
        let log_2pi := MathFunctions.log (Numbers.two * MathFunctions.pi)
        let log_det := match det with | Tensor.scalar d_val => MathFunctions.log d_val
        let normalization : α :=
          (d : α) / Numbers.two * log_2pi + Numbers.pointfive * log_det

        Tensor.scalar (Numbers.neg_point_five * quadratic_form - normalization)

      -- Add the log weight to get the log-likelihood for this component
      match weight, log_likelihood with
      | Tensor.scalar w, Tensor.scalar ll =>
        let log_weight := MathFunctions.log w
        Tensor.scalar (log_weight + ll)
    )

/-- E-step responsibilities for a single input.

Mathematically:

`γ_k = P(z=k | x) = softmax_k ( log π_k + log N(x | μ_k, Σ_k) )`

PyTorch analogy: `torch.softmax(component_log_probs, dim=-1)` where the logits are the
per-component log-probabilities.
-/
def gmmExpectationSpec {nComponents nFeatures : Nat}
  (m : GMMSpec α nComponents nFeatures)
  (input : Tensor α (.dim nFeatures .scalar))
  (_h : nComponents ≠ 0) :
  Tensor α (.dim nComponents .scalar) :=
  let component_log_probs := gmmForwardSpec m input
  Activation.softmaxVecSpec (α := α) (n := nComponents) component_log_probs

/-- Batched forward pass: apply `gmm_forward_spec` to each sample in a batch. -/
def gmmBatchedForwardSpec {batch nComponents nFeatures : Nat}
  (m : GMMSpec α nComponents nFeatures)
  (input : Tensor α (.dim batch (.dim nFeatures .scalar))) :
  Tensor α (.dim batch (.dim nComponents .scalar)) :=
  match input with
  | Tensor.dim batch_fn =>
    Tensor.dim (fun i => gmmForwardSpec m (batch_fn i))

/-!
## Backward/VJP (for `gmm_forward_spec`)

`gmm_forward_spec` is **vector-valued**: it returns one log-probability per component.

The gradients below are the VJP for that vector function. In particular, responsibilities
`γ = softmax(component_log_probs)` do *not* appear in these formulas by themselves.

Responsibilities show up when you differentiate a **scalar** objective that aggregates components,
like the mixture log-likelihood `logsumexp(component_log_probs)`. In that case, you compute
`dL/d(component_log_probs)` first (which will involve `γ`), then feed that vector into
`gmm_backward_spec`.
-/

/-- Gradient/VJP w.r.t. weights `π` for the output of `gmm_forward_spec`.

For `y_k = log π_k + ...`, we have `∂y_k/∂π_k = 1/π_k`.
-/
def gmmWeightsDerivSpec {nComponents nFeatures : Nat}
  (m : GMMSpec α nComponents nFeatures)
  (grad_output : Tensor α (.dim nComponents .scalar))
  (_h : nComponents ≠ 0) :
  Tensor α (.dim nComponents .scalar) :=
  Tensor.dim (fun k =>
    match get m.weights k, get grad_output k with
    | Tensor.scalar π_k, Tensor.scalar g =>
      Tensor.scalar (g / π_k)
  )

/-- Gradient/VJP w.r.t. means `μ` for the output of `gmm_forward_spec`.

For a single component:

`∂/∂μ log N(x|μ,Σ) = Σ^{-1} (x - μ)`.
-/
def gmmMeansDerivSpec {nComponents nFeatures : Nat}
  (m : GMMSpec α nComponents nFeatures)
  (input : Tensor α (.dim nFeatures .scalar))
  (grad_output : Tensor α (.dim nComponents .scalar))
  (_h : nComponents ≠ 0) :
  Tensor α (.dim nComponents (.dim nFeatures .scalar)) :=
  Tensor.dim (fun k =>
    let mean_k := get m.means k
    let covariance_k := get m.covariances k
    let grad_k := get grad_output k

    -- Compute (x - μ_k)
    let diff := subSpec input mean_k

    -- Compute Σ_k^(-1) * (x - μ_k)
    let cov_inv := inverseSpec covariance_k
    let weighted_diff := matVecMulSpec cov_inv diff

    match grad_k with
    | Tensor.scalar g =>
      scaleSpec weighted_diff g
  )

/-- Gradient/VJP w.r.t. the input `x` for the output of `gmm_forward_spec`.

For one component:

`∂/∂x log N(x|μ,Σ) = - Σ^{-1} (x - μ)`.

We sum the contributions from all components, weighted by the upstream gradient `g_k`.
-/
def gmmInputDerivSpec {nComponents nFeatures : Nat}
  (m : GMMSpec α nComponents nFeatures)
  (input : Tensor α (.dim nFeatures .scalar))
  (grad_output : Tensor α (.dim nComponents .scalar))
  (h : nComponents ≠ 0) :
  Tensor α (.dim nFeatures .scalar) :=
  have inst : Shape.valid_axis_inst 0 (Shape.dim nComponents (.dim nFeatures .scalar)) := by
    apply Shape.validAxisInstZeroAlt h
  let perComponent : Tensor α (.dim nComponents (.dim nFeatures .scalar)) :=
    Tensor.dim (fun k =>
      let mean_k := get m.means k
      let covariance_k := get m.covariances k
      let gk := get grad_output k
      let diff := subSpec input mean_k
      let cov_inv := inverseSpec covariance_k
      let v := matVecMulSpec cov_inv diff
      match gk with
      | Tensor.scalar g =>
          scaleSpec v (Numbers.neg_one * g))
  reduceSumAuto 0 perComponent

/-- Gradient/VJP w.r.t. covariances `Σ` for the output of `gmm_forward_spec`.

For one component:

`∂/∂Σ log N(x|μ,Σ) = 1/2 * ( Σ^{-1} (x-μ)(x-μ)^T Σ^{-1} - Σ^{-1} )`.
-/
def gmmCovariancesDerivSpec {nComponents nFeatures : Nat}
  (m : GMMSpec α nComponents nFeatures)
  (input : Tensor α (.dim nFeatures .scalar))
  (grad_output : Tensor α (.dim nComponents .scalar))
  (_h : nComponents ≠ 0) :
  Tensor α (.dim nComponents (.dim nFeatures (.dim nFeatures .scalar))) :=
  Tensor.dim (fun k =>
    let mean_k := get m.means k
    let covariance_k := get m.covariances k
    let grad_k := get grad_output k

    -- Compute (x - μ_k)
    let diff := subSpec input mean_k

    -- Compute outer product (x - μ_k)(x - μ_k)^T
    let outer_product := outerProductSpec diff diff

    -- Compute Σ_k^(-1)
    let cov_inv := inverseSpec covariance_k

    -- Σ^{-1} (x-μ)(x-μ)^T Σ^{-1} - Σ^{-1}
    let temp1 := matMulSpec cov_inv outer_product
    let temp2 := matMulSpec temp1 cov_inv
    let gradSigma := subSpec temp2 cov_inv

    match grad_k with
    | Tensor.scalar g =>
        scaleSpec gradSigma (Numbers.pointfive * g)
  )

/-- Backward/VJP for `gmm_forward_spec`.

Returns gradients with respect to `(weights, means, covariances, input)`.
-/
def gmmBackwardSpec {nComponents nFeatures : Nat}
  (m : GMMSpec α nComponents nFeatures)
  (input : Tensor α (.dim nFeatures .scalar))
  (grad_output : Tensor α (.dim nComponents .scalar))
  (h : nComponents ≠ 0) :
  (Tensor α (.dim nComponents .scalar) ×
   Tensor α (.dim nComponents (.dim nFeatures .scalar)) ×
   Tensor α (.dim nComponents (.dim nFeatures (.dim nFeatures .scalar))) ×
   Tensor α (.dim nFeatures .scalar)) :=
  let d_weights := gmmWeightsDerivSpec m grad_output h
  let d_means := gmmMeansDerivSpec m input grad_output h
  let d_covariances := gmmCovariancesDerivSpec m input grad_output h
  let d_input := gmmInputDerivSpec m input grad_output h
  (d_weights, d_means, d_covariances, d_input)

/-- Uniform mixture weights (all components have probability `1/nComponents`). -/
private def uniformWeights {nComponents : Nat} : Tensor α (.dim nComponents .scalar) :=
  match nComponents with
  | 0 => Tensor.dim (fun k => nomatch k)
  | Nat.succ _ => Tensor.dim (fun _ => Tensor.scalar (1 / (nComponents : α)))

/-- Default initialization for a GMM.

This is intentionally simple and deterministic:

- uniform weights,
- zero means,
- identity covariances.
-/
def gmmInitSpec {nComponents nFeatures : Nat} :
  GMMSpec α nComponents nFeatures :=
  let weights : Tensor α (.dim nComponents .scalar) := uniformWeights (α := α) (nComponents :=
    nComponents)
  let means : Tensor α (.dim nComponents (.dim nFeatures .scalar)) := Tensor.dim (fun _ =>
    Tensor.dim (fun _ => Tensor.scalar (0 : α)))
  let covariances : Tensor α (.dim nComponents (.dim nFeatures (.dim nFeatures .scalar))) :=
    Tensor.dim (fun _ => identityTensorSpec nFeatures)
  {
    weights := weights,
    means := means,
    covariances := covariances
  }

/--
Numerically stable log-sum-exp reduction: `log (Σ_i exp(log_probs[i]))`.

This is the standard `max + log(sum(exp(x - max)))` trick.
-/
def logSumExpReduce {n : Nat} (log_probs : Tensor α (.dim n .scalar)) (h : n ≠ 0) : α :=
  -- Step 1: Find maximum for numerical stability
  have inst : Shape.valid_axis_inst 0 (Shape.dim n .scalar) := by
    apply Shape.validAxisInstZeroAlt h
  let max_log_prob := reduceMaxAuto 0 log_probs
  have h_shape : shapeAfterSum (Shape.dim n Shape.scalar) 0 = Shape.scalar := by
    simp [shapeAfterSum]
  let max_log_prob' := toScalar (tensorCast (Shape.scalar) h_shape.symm max_log_prob)

  -- Step 2: Compute sum of exp(log_prob - max_log_prob)
  let shifted_probs := Tensor.dim (fun k =>
    match get log_probs k with
    | Tensor.scalar log_prob =>
      Tensor.scalar (MathFunctions.exp (log_prob - max_log_prob'))
  )
  let sum_shifted := sumSpec shifted_probs

  -- Step 3: Compute log(sum_shifted) + max_log_prob
  if sum_shifted > 0 then
    MathFunctions.log sum_shifted + max_log_prob'
  else
    max_log_prob'  -- Fallback if sum is zero

/--
Mixture log-likelihood `log p(x)` computed via log-sum-exp over components.

Mathematically: `log p(x) = log (Σ_k exp(log p(x | z_k) + log π_k))`.
-/
def gmmLogLikelihoodSpec {nComponents nFeatures : Nat}
  (m : GMMSpec α nComponents nFeatures)
  (input : Tensor α (.dim nFeatures .scalar))
  (h : nComponents ≠ 0) :
  α :=
  let component_log_probs := gmmForwardSpec m input
  logSumExpReduce component_log_probs h

/-!
## Classical training: EM for Gaussian mixtures

For a GMM, “training” is typically done with the Expectation–Maximization (EM) algorithm:

- **E-step**: compute responsibilities `r_{ik} = P(z=k | x_i)` for each sample/component.
- **M-step**: update `π, μ, Σ` from the weighted sufficient statistics.

This file already provides `gmm_expectation_spec` (responsibilities for one sample). The helpers
below lift that to a batched dataset and implement a deterministic EM update step.

Numerical notes:
- If a component gets (near) zero total responsibility (`N_k ≈ 0`), we keep that component’s
  parameters unchanged (otherwise we’d divide by zero).
- We add a small diagonal “jitter” (`Numbers.epsilon · I`) to covariances to keep them well-behaved.
-/

/-- Batched responsibilities: apply `gmm_expectation_spec` to each sample. -/
def gmmResponsibilitiesBatchedSpec {nSamples nComponents nFeatures : Nat}
  (m : GMMSpec α nComponents nFeatures)
  (data : Tensor α (.dim nSamples (.dim nFeatures .scalar)))
  (hK : nComponents ≠ 0) :
  Tensor α (.dim nSamples (.dim nComponents .scalar)) :=
  match data with
  | Tensor.dim f =>
      Tensor.dim (fun i => gmmExpectationSpec (α := α) (nComponents := nComponents) (nFeatures :=
        nFeatures) m (f i) hK)

/-- Scalar extraction helper for 2D tensors: `t[i,j]` as an `α`. -/
private def get2D {n m : Nat} (t : Tensor α (.dim n (.dim m .scalar))) (i : Fin n) (j : Fin m) : α
  :=
  match get (get t i) j with
  | Tensor.scalar v => v

/-- Build a vector tensor from a function `Fin n -> α`. -/
private def vecFromFn {n : Nat} (f : Fin n → α) : Tensor α (.dim n .scalar) :=
  Tensor.dim (fun i => Tensor.scalar (f i))

/-- Build a matrix tensor from a function `Fin n -> Fin m -> α`. -/
private def matFromFn {n m : Nat} (f : Fin n → Fin m → α) : Tensor α (.dim n (.dim m .scalar)) :=
  Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (f i j)))

/-- One EM step for a batched dataset. -/
def gmmEmStepSpec {nSamples nComponents nFeatures : Nat}
  (m : GMMSpec α nComponents nFeatures)
  (data : Tensor α (.dim nSamples (.dim nFeatures .scalar)))
  (hK : nComponents ≠ 0) :
  GMMSpec α nComponents nFeatures :=
  if _hN : nSamples = 0 then
    m
  else
    let resp := gmmResponsibilitiesBatchedSpec (α := α) (nSamples := nSamples)
      (nComponents := nComponents) (nFeatures := nFeatures) m data hK

    -- N_k = Σ_i r_{ik}
    let Nk : Tensor α (.dim nComponents .scalar) :=
      vecFromFn (n := nComponents) (fun k =>
        (List.finRange nSamples).foldl (fun acc i =>
          acc + get2D (n := nSamples) (m := nComponents) resp i k
        ) 0)

    -- π_k = N_k / N
    let weights : Tensor α (.dim nComponents .scalar) :=
      let wRaw : Tensor α (.dim nComponents .scalar) :=
        match Nk, m.weights with
        | Tensor.dim f, Tensor.dim wOld =>
          Tensor.dim (fun k =>
            match f k, wOld k with
            | Tensor.scalar nk, Tensor.scalar w =>
              if nk > 0 then Tensor.scalar (nk / (nSamples : α)) else Tensor.scalar w)
        | _, _ => uniformWeights (α := α) (nComponents := nComponents)
      let s := sumSpec wRaw
      if s > 0 then scaleSpec wRaw (1 / s) else uniformWeights (α := α) (nComponents :=
        nComponents)

    -- μ_k = (1/N_k) Σ_i r_{ik} x_i
    let means : Tensor α (.dim nComponents (.dim nFeatures .scalar)) :=
      match Nk, m.means with
      | Tensor.dim NkF, Tensor.dim muOld =>
        Tensor.dim (fun k =>
          match NkF k with
          | Tensor.scalar nk =>
            if nk > 0 then
              Tensor.dim (fun f =>
                Tensor.scalar (
                  (List.finRange nSamples).foldl (fun acc i =>
                    let rik := get2D (n := nSamples) (m := nComponents) resp i k
                    let xi := get data i
                    acc + rik * Tensor.vecGet xi f
                  ) 0 / nk))
            else
              muOld k)
      | _, _ => m.means

    -- Σ_k = (1/N_k) Σ_i r_{ik} (x_i-μ_k)(x_i-μ_k)ᵀ + εI
    let covariances : Tensor α (.dim nComponents (.dim nFeatures (.dim nFeatures .scalar))) :=
      match Nk, means with
      | Tensor.dim NkF, Tensor.dim muF =>
        Tensor.dim (fun k =>
          match NkF k, muF k with
          | Tensor.scalar nk, Tensor.dim muVec =>
            if nk > 0 then
              let μ : Tensor α (.dim nFeatures .scalar) := Tensor.dim muVec
              let base :=
                matFromFn (n := nFeatures) (m := nFeatures) (fun a b =>
                  (List.finRange nSamples).foldl (fun acc i =>
                    let rik := get2D (n := nSamples) (m := nComponents) resp i k
                    let xi := get data i
                    let da := Tensor.vecGet xi a - Tensor.vecGet μ a
                    let db := Tensor.vecGet xi b - Tensor.vecGet μ b
                    acc + rik * da * db
                  ) 0 / nk)
              let jitter := scaleSpec (identityTensorSpec nFeatures) Numbers.epsilon
              addSpec base jitter
            else
              get m.covariances k
          | _, _ => get m.covariances k)
      | _, _ => m.covariances

    { weights := weights, means := means, covariances := covariances }

/-- Total negative log-likelihood of a dataset under the current model. -/
def gmmNegLogLikelihoodBatchedSpec {nSamples nComponents nFeatures : Nat}
  (m : GMMSpec α nComponents nFeatures)
  (data : Tensor α (.dim nSamples (.dim nFeatures .scalar)))
  (hK : nComponents ≠ 0) : α :=
  (List.finRange nSamples).foldl (fun acc i =>
    let xi := get data i
    let ll := gmmLogLikelihoodSpec (α := α) (nComponents := nComponents) (nFeatures := nFeatures)
      m xi hK
    acc - ll
  ) 0

/-- Run `epochs` EM steps (deterministic). -/
def gmmEmTrainSpec {nSamples nComponents nFeatures : Nat}
  (epochs : Nat)
  (m : GMMSpec α nComponents nFeatures)
  (data : Tensor α (.dim nSamples (.dim nFeatures .scalar)))
  (hK : nComponents ≠ 0) :
  GMMSpec α nComponents nFeatures :=
  (List.finRange epochs).foldl (fun cur _ => gmmEmStepSpec (α := α) cur data hK) m

end Spec
