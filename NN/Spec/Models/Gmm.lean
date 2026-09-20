/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Activation
public import NN.Spec.Core.Tensor.Numerics

/-!
# Gaussian Mixture Model (GMM) (spec model)

This file defines a basic GMM with `nComponents` multivariate Gaussians over `nFeatures`:

- mixing weights $\pi$,
- means $\mu$, and
- covariances $\Sigma$.

`gmmForwardSpec` computes per-component log scores for a single input:

$$
\log \pi_k + \log \mathcal{N}(x \mid \mu_k, \Sigma_k).
$$

PyTorch analogies:

- `torch.distributions.MultivariateNormal` for $\mathcal{N}(x \mid \mu, \Sigma)$,
- `torch.distributions.MixtureSameFamily` for mixture distributions,
- `torch.softmax` for turning per-component log-probabilities into responsibilities.

Mixture weights must be positive and normalized within the scalar backend's validation tolerance.
Covariances must be symmetric positive definite. Parameters outside this domain are reported as
`none`. Scores use the stored weights directly: when the weights sum to one, their exponential sum
is a probability density. The default normalization tolerance also applies to exact scalar
backends; callers needing a stricter gate can check `mixtureWeightsValidSpec` with zero tolerance.

Determinants and inverses are defined via `NN.Spec.Core.Tensor.Numerics`; those
definitions are intended for small feature dimensions and proof/reference usage, not
high-performance clustering on large matrices.

References (background, not required to read the code):

- Dempster, Laird, Rubin (1977), "Maximum Likelihood from Incomplete Data via the EM Algorithm":
  https://www.jstor.org/stable/2984875
- Bishop (2006), "Pattern Recognition and Machine Learning", Chapter 9 (Mixture Models and EM):
  https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/

## Implementation status

No API builder implements this model. It is exercised numerically in
`NN/Tests/Runtime/Floats/TorchLeanOpsCheck.lean`; no theorem is proved about it.
-/

public section


open TorchLean

namespace Spec

open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]

/-! ## Parameters -/

/-- Parameters of a Gaussian mixture model (GMM). -/
structure GMMSpec (α : Type) [TorchLean.Storage α]
    (nComponents nFeatures : Nat) where
  /-- Mixing weights $\pi_k$. Model operations require every weight to be strictly positive
  and their compensated total to be within `Context.defaultEpsilon` of one. -/
  weights : Tensor α [nComponents]
  /-- Component means $\mu_k$. -/
  means : Tensor α [nComponents, nFeatures]
  /-- Component covariance matrices $\Sigma_k$ (typically symmetric positive definite). -/
  covariances : Tensor α [nComponents, nFeatures, nFeatures]

/-- The leading `k × k` principal submatrix of a square matrix. -/
private def leadingPrincipalSubmatrix {n : Nat}
    (matrix : Tensor α [n, n]) (k : Nat) (hk : k ≤ n) :
    Tensor α [k, k] :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      Tensor.scalar (get2 matrix (i.castLE hk) (j.castLE hk))))

/-- Whether a matrix is symmetric under the scalar backend's equality operation. -/
def matrixSymmetricSpec {n : Nat}
    (matrix : Tensor α [n, n]) : Bool :=
  (List.finRange n).all (fun i =>
    (List.finRange n).all (fun j => get2 matrix i j == get2 matrix j i))

/--
Executable Sylvester-criterion check for a symmetric positive-definite covariance matrix.

The matrix must be symmetric and every nonempty leading principal minor must be positive. This is
the domain on which the Gaussian density, inverse, and logarithmic determinant used below have
their usual meaning.
-/
def covariancePositiveDefiniteSpec {n : Nat}
    (matrix : Tensor α [n, n]) : Bool :=
  matrixSymmetricSpec matrix &&
    (List.finRange n).all (fun i =>
      let k := i.val + 1
      have hk : k ≤ n := Nat.succ_le_iff.mpr i.isLt
      let leading := leadingPrincipalSubmatrix matrix k hk
      Context.gtBool (Tensor.item (determinantSpec leading)) 0)

/--
Sum the mixture weights while retaining the low-order part lost at each addition.

The correction is carried into the next addition. This matters for mixtures with many components:
adding their small weights directly can accumulate enough rounding error to reject a normalized
mixture. Over exact arithmetic, the correction is zero and this is the ordinary sum.
-/
private def mixtureWeightTotal {n : Nat} (weights : Tensor α [n]) : α :=
  ((List.finRange n).foldl (fun (state : α × α) i =>
    let adjusted := weights.getScalar i - state.2
    let total := state.1 + adjusted
    (total, (total - state.1) - adjusted)) (0, 0)).1

/--
Check that every mixture weight is positive and their total is within `tolerance` of one.

Floating-point division usually cannot represent the exact uniform weight `1 / n`. A check against
exactly one can therefore reject `gmmInitSpec` before its first forward pass. The compensated total
limits error from the reduction, and the tolerance accounts for rounding in the weights themselves.
The admissible tolerance is in `[0, 1)`, and the comparison uses absolute error. The default
`Context.defaultEpsilon` is the backend's configured numerical safeguard. It also applies to exact
scalar backends. Zero requires the computed compensated total to equal one; callers can use this
stricter check before invoking model operations, which use the default tolerance.
-/
def mixtureWeightsValidSpec {n : Nat} (weights : Tensor α [n])
    (tolerance : α := Context.defaultEpsilon) : Bool :=
  let toleranceValid := (tolerance == 0 || Context.gtBool tolerance 0) &&
    Context.gtBool 1 tolerance
  let positive :=
    (List.finRange n).all (fun i => Context.gtBool (weights.getScalar i) 0)
  let discrepancy := MathFunctions.abs (mixtureWeightTotal weights - 1)
  toleranceValid && positive &&
    (discrepancy == tolerance || Context.gtBool tolerance discrepancy)

/--
Check the dimensions, mixing weights, and covariance matrices before evaluating the model.

The model needs at least one component and one feature. Its weights must be positive and satisfy
the configured normalization tolerance, and every covariance must be symmetric positive definite.
Accepted weights are used as stored; this check does not rescale them to sum to exactly one.
-/
def gmmParametersValidSpec {nComponents nFeatures : Nat}
    (m : GMMSpec α nComponents nFeatures) : Bool :=
  nComponents != 0 && nFeatures != 0 &&
    mixtureWeightsValidSpec m.weights &&
    (List.finRange nComponents).all (fun k =>
      covariancePositiveDefiniteSpec (Tensor.unstack m.covariances k))

/-- Per-component log scores for a single input.

Given $x \in \mathbb{R}^d$, each component contributes

$$
\log \pi_k-\frac12\left(
  (x-\mu_k)^\mathsf{T}\Sigma_k^{-1}(x-\mu_k)
  +\log\det\Sigma_k+d\log(2\pi)
\right).
$$

This is the natural "logit vector" for responsibilities.  If you want posterior probabilities
$P(z=k\mid x)$, apply `gmmExpectationSpec` (a last-axis softmax).

PyTorch analogy: the returned vector is like per-component `logProb` values before the final
mixture `logsumexp`.
-/
def gmmForwardSpec {nComponents nFeatures : Nat}
  (m : GMMSpec α nComponents nFeatures)
  (input : Tensor α [nFeatures]) :
  Option (Tensor α [nComponents]) :=
  if gmmParametersValidSpec m then
    sequenceFin (fun k => do
      let mean := Tensor.unstack m.means k
      let covariance := Tensor.unstack m.covariances k
      let diff := subSpec input mean
      let covInv ← inverseSpec? covariance
      let det := Tensor.item (determinantSpec covariance)
      let w := m.weights.getScalar k
      let quadraticForm := dotSpec diff (matVecMulSpec covInv diff)
      let log2pi := MathFunctions.log (2 * MathFunctions.pi)
      let normalization : α :=
        (nFeatures : α) / 2 * log2pi +
          (1 / 2) * MathFunctions.log det
      some (Tensor.scalar
        (MathFunctions.log w + (-1 / 2) * quadraticForm - normalization)))
  else
    none

/-- E-step responsibilities for a single input.

Mathematically:

$$
\gamma_k
  = P(z=k\mid x)
  = \operatorname{softmax}_k\!\left(
      \log \pi_k+\log\mathcal{N}(x\mid\mu_k,\Sigma_k)
    \right).
$$

PyTorch analogy: `torch.softmax(component_log_probs, dim=-1)` where the logits are the
per-component log-probabilities.
-/
def gmmExpectationSpec {nComponents nFeatures : Nat}
  (m : GMMSpec α nComponents nFeatures)
  (input : Tensor α [nFeatures])
  (_h : nComponents ≠ 0) :
  Option (Tensor α [nComponents]) := do
  let componentLogProbs ← gmmForwardSpec m input
  pure (Activation.softmaxVecSpec (α := α) (n := nComponents) componentLogProbs)

/--
Batched forward pass: apply `gmmForwardSpec` to each sample in a batch.

Invalid model parameters return `none`, including for an empty batch.
-/
def gmmBatchedForwardSpec {batch nComponents nFeatures : Nat}
  (m : GMMSpec α nComponents nFeatures)
  (input : Tensor α [batch, nFeatures]) :
  Option (Tensor α [batch, nComponents]) :=
  if gmmParametersValidSpec m then
    sequenceFin (fun i => gmmForwardSpec m (Tensor.unstack input i))
  else
    none

/-!
## Backward/VJP (for `gmmForwardSpec`)

`gmmForwardSpec` is **vector-valued**: it returns one log score per component.

The gradients below are the VJP for that vector function. In particular, responsibilities
$\gamma=\operatorname{softmax}(\text{component log scores})$ do *not* appear in these
formulas by themselves.

Responsibilities show up when you differentiate a **scalar** objective that aggregates components,
like the mixture log-likelihood $\operatorname{logsumexp}(\text{component log scores})$.
In that case, you compute the derivative with respect to the component log scores first
(which will involve $\gamma$), then feed that vector into
`gmmBackwardSpec`.
-/

/-- Gradient/VJP with respect to weights $\pi$ for the output of `gmmForwardSpec`.

For $y_k=\log\pi_k+\cdots$, we have $\partial y_k/\partial\pi_k=1/\pi_k$.
-/
def gmmWeightsDerivSpec {nComponents nFeatures : Nat}
  (m : GMMSpec α nComponents nFeatures)
  (gradOutput : Tensor α [nComponents])
  (_h : nComponents ≠ 0) :
  Option (Tensor α [nComponents]) :=
  if gmmParametersValidSpec m then
    sequenceFin (fun k =>
      some (Tensor.scalar (gradOutput.getScalar k / m.weights.getScalar k)))
  else
    none

/-- Gradient/VJP with respect to means $\mu$ for the output of `gmmForwardSpec`.

For a single component:

$$
\frac{\partial}{\partial\mu}\log\mathcal{N}(x\mid\mu,\Sigma)
=\frac12\left(\Sigma^{-1}+\Sigma^{-\mathsf{T}}\right)(x-\mu).
$$

For a valid symmetric covariance this reduces to the familiar $\Sigma^{-1}(x-\mu)$.
-/
def gmmMeansDerivSpec {nComponents nFeatures : Nat}
  (m : GMMSpec α nComponents nFeatures)
  (input : Tensor α [nFeatures])
  (gradOutput : Tensor α [nComponents])
  (_h : nComponents ≠ 0) :
  Option (Tensor α [nComponents, nFeatures]) :=
  if gmmParametersValidSpec m then
    sequenceFin (fun k => do
      let mean_k := get m.means k
      let covariance_k := get m.covariances k
      let grad_k := get gradOutput k
      let diff := subSpec input mean_k
      let covInv ← inverseSpec? covariance_k
      let covInvT := swapAdjacentAxes covInv 0
      let weightedDiff := scaleSpec
        (addSpec (matVecMulSpec covInv diff) (matVecMulSpec covInvT diff))
        (1 / 2)
      pure (scaleSpec weightedDiff grad_k.item))
  else
    none

/-- Gradient/VJP with respect to the input $x$ for the output of `gmmForwardSpec`.

For one component:

$$
\frac{\partial}{\partial x}\log\mathcal{N}(x\mid\mu,\Sigma)
=-\frac12\left(\Sigma^{-1}+\Sigma^{-\mathsf{T}}\right)(x-\mu).
$$

We sum the contributions from all components, weighted by the upstream gradient $g_k$.
-/
def gmmInputDerivSpec {nComponents nFeatures : Nat}
  (m : GMMSpec α nComponents nFeatures)
  (input : Tensor α [nFeatures])
  (gradOutput : Tensor α [nComponents])
  (h : nComponents ≠ 0) :
  Option (Tensor α [nFeatures]) :=
  if gmmParametersValidSpec m then do
    have inst : Shape.HasNonemptyAxis 0 (Shape.dim nComponents (.dim nFeatures .scalar)) := by
      apply Shape.hasNonemptyAxisZeroOfNe h
    let perComponent ← sequenceFin (fun k => do
        let mean_k := get m.means k
        let covariance_k := get m.covariances k
        let gk := get gradOutput k
        let diff := subSpec input mean_k
        let covInv ← inverseSpec? covariance_k
        let covInvT := swapAdjacentAxes covInv 0
        let v := scaleSpec
          (addSpec (matVecMulSpec covInv diff) (matVecMulSpec covInvT diff))
          (1 / 2)
        pure (scaleSpec v ((-1) * gk.item)))
    pure (reduceSum 0 perComponent inst.proof)
  else
    none

/-- Gradient/VJP with respect to covariances $\Sigma$ for the output of `gmmForwardSpec`.

For one component:

$$
\frac{\partial}{\partial\Sigma}\log\mathcal{N}(x\mid\mu,\Sigma)
=\frac12\left(
  \Sigma^{-\mathsf{T}}(x-\mu)(x-\mu)^\mathsf{T}\Sigma^{-\mathsf{T}}
  -\Sigma^{-\mathsf{T}}
\right).
$$
-/
def gmmCovariancesDerivSpec {nComponents nFeatures : Nat}
  (m : GMMSpec α nComponents nFeatures)
  (input : Tensor α [nFeatures])
  (gradOutput : Tensor α [nComponents])
  (_h : nComponents ≠ 0) :
  Option (Tensor α [nComponents, nFeatures, nFeatures]) :=
  if gmmParametersValidSpec m then
    sequenceFin (fun k => do
      let mean_k := get m.means k
      let covariance_k := get m.covariances k
      let grad_k := get gradOutput k
      let diff := subSpec input mean_k
      let outerProduct := outerProductSpec diff diff
      let covInv ← inverseSpec? covariance_k
      let covInvT := swapAdjacentAxes covInv 0
      let temp1 := matMulSpec covInvT outerProduct
      let temp2 := matMulSpec temp1 covInvT
      let gradSigma := subSpec temp2 covInvT
      pure (scaleSpec gradSigma ((1 / 2) * grad_k.item)))
  else
    none

/-- Gradients for a `GMMSpec`, one field per differentiable component. -/
structure GMMGradients (α : Type) [TorchLean.Storage α] (nComponents nFeatures : Nat) where
  /-- Gradient with respect to the mixture weights. -/
  weightsGradient : Tensor α [nComponents]
  /-- Gradient with respect to the component means. -/
  meansGradient : Tensor α [nComponents, nFeatures]
  /-- Gradient with respect to the component covariance matrices. -/
  covariancesGradient : Tensor α [nComponents, nFeatures, nFeatures]
  /-- Gradient with respect to the observed point. -/
  inputGradient : Tensor α [nFeatures]

/-- Backward/VJP for `gmmForwardSpec`.

Every component gradient needs a covariance inverse, so the whole record is optional: a singular
covariance makes the derivative undefined rather than zero.
-/
def gmmBackwardSpec {nComponents nFeatures : Nat}
  (m : GMMSpec α nComponents nFeatures)
  (input : Tensor α [nFeatures])
  (gradOutput : Tensor α [nComponents])
  (h : nComponents ≠ 0) :
  Option (GMMGradients α nComponents nFeatures) := do
  let dWeights ← gmmWeightsDerivSpec m gradOutput h
  let dMeans ← gmmMeansDerivSpec m input gradOutput h
  let dCovariances ← gmmCovariancesDerivSpec m input gradOutput h
  let dInput ← gmmInputDerivSpec m input gradOutput h
  pure
    { weightsGradient := dWeights
      meansGradient := dMeans
      covariancesGradient := dCovariances
      inputGradient := dInput }

/-- Uniform mixture weights (all components have probability $1/\mathtt{nComponents}$). -/
private def uniformWeights {nComponents : Nat} : Tensor α [nComponents] :=
  match nComponents with
  | 0 => Tensor.dim (fun k => nomatch k)
  | Nat.succ _ => Tensor.dim (fun _ => Tensor.scalar (1 / (nComponents : α)))

/-- Default initialization for a GMM.

This is kept simple and deterministic:

- uniform weights,
- zero means,
- identity covariances.
-/
def gmmInitSpec {nComponents nFeatures : Nat} :
  GMMSpec α nComponents nFeatures :=
  let weights : Tensor α [nComponents] := uniformWeights (α := α) (nComponents :=
    nComponents)
  let means : Tensor α [nComponents, nFeatures] := Tensor.dim (fun _ =>
    Tensor.dim (fun _ => Tensor.scalar (0 : α)))
  let covariances : Tensor α [nComponents, nFeatures, nFeatures] :=
    Tensor.dim (fun _ => identityTensorSpec nFeatures)
  {
    weights := weights,
    means := means,
    covariances := covariances
  }

/--
Numerically stable log-sum-exp reduction:
$\log\!\left(\sum_i \exp(\mathtt{log\_probs}[i])\right)$.

This is the standard
$m+\log\!\left(\sum_i\exp(x_i-m)\right)$ trick, where $m=\max_i x_i$.
-/
def logSumExpReduce {n : Nat} (logProbs : Tensor α [n]) (h : n ≠ 0) : α :=
  -- Step 1: Find maximum for numerical stability
  have inst : Shape.HasNonemptyAxis 0 (Shape.dim n .scalar) := by
    apply Shape.hasNonemptyAxisZeroOfNe h
  let maxLogProb := reduceMax 0 logProbs inst.proof
  have h_shape : shapeAfterSum (Shape.dim n Shape.scalar) 0 = Shape.scalar := by
    simp [shapeAfterSum]
  let max_log_prob' := item (tensorCast (Shape.scalar) h_shape.symm maxLogProb)

  -- Step 2: Compute sum of exp(logProb - maxLogProb)
  let shiftedProbs := mapSpec
    (fun logProb => MathFunctions.exp (logProb - max_log_prob')) logProbs
  let sumShifted := sumSpec shiftedProbs

  -- Step 3: Compute log(sumShifted) + maxLogProb
  if sumShifted > 0 then
    MathFunctions.log sumShifted + max_log_prob'
  else
    max_log_prob'  -- Fallback if sum is zero

/--
Mixture log-likelihood $\log p(x)$ computed via log-sum-exp over components.

Mathematically,
$$
\log p(x)=\log\!\left(
  \sum_k \exp\!\left(\log p(x\mid z_k)+\log\pi_k\right)
\right).
$$

This is the log of a normalized mixture density when the stored weights sum to one.
The default domain check allows `Context.defaultEpsilon` error in that sum; accepted weights
are used directly without renormalization.
-/
def gmmLogLikelihoodSpec {nComponents nFeatures : Nat}
  (m : GMMSpec α nComponents nFeatures)
  (input : Tensor α [nFeatures])
  (h : nComponents ≠ 0) :
  Option α := do
  let componentLogProbs ← gmmForwardSpec m input
  pure (logSumExpReduce componentLogProbs h)

/-!
## Classical training: EM for Gaussian mixtures

For a GMM, “training” is typically done with the Expectation–Maximization (EM) algorithm:

- **E-step**: compute responsibilities $r_{ik}=P(z=k\mid x_i)$ for each sample/component.
- **M-step**: update $\pi$, $\mu$, and $\Sigma$ from the weighted sufficient statistics.

This file already provides `gmmExpectationSpec` (responsibilities for one sample). The helpers
below lift that to a batched dataset and implement a deterministic EM update step.

Numerical notes:
- If a component gets (near) zero total responsibility ($N_k\approx 0$), we keep that component’s
  parameters unchanged (otherwise we’d divide by zero).
- We add a small diagonal “jitter,” $\mathtt{Context.defaultEpsilon}\,I$,
  to covariances to keep them
  well-behaved.
-/

/--
Batched responsibilities: apply `gmmExpectationSpec` to each sample.

Invalid model parameters return `none`, including for an empty dataset.
-/
def gmmResponsibilitiesBatchedSpec {nSamples nComponents nFeatures : Nat}
  (m : GMMSpec α nComponents nFeatures)
  (data : Tensor α [nSamples, nFeatures])
  (hK : nComponents ≠ 0) :
  Option (Tensor α [nSamples, nComponents]) :=
  if gmmParametersValidSpec m then
    sequenceFin (fun i => gmmExpectationSpec (α := α) (nComponents := nComponents)
      (nFeatures := nFeatures) m (Tensor.unstack data i) hK)
  else
    none

/-- Build a vector tensor from a function `Fin n -> α`. -/
private def vecFromFn {n : Nat} (f : Fin n → α) : Tensor α [n] :=
  Tensor.dim (fun i => Tensor.scalar (f i))

/-- Build a matrix tensor from a function `Fin n -> Fin m -> α`. -/
private def matFromFn {n m : Nat} (f : Fin n → Fin m → α) : Tensor α [n, m] :=
  Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (f i j)))

/--
One EM step for a batched dataset.

An empty dataset preserves a valid model; invalid parameters still return `none`.
-/
def gmmEmStepSpec {nSamples nComponents nFeatures : Nat}
  (m : GMMSpec α nComponents nFeatures)
  (data : Tensor α [nSamples, nFeatures])
  (hK : nComponents ≠ 0) :
  Option (GMMSpec α nComponents nFeatures) :=
  if _hN : nSamples = 0 then
    if gmmParametersValidSpec m then some m else none
  else do
    let resp ← gmmResponsibilitiesBatchedSpec (α := α) (nSamples := nSamples)
      (nComponents := nComponents) (nFeatures := nFeatures) m data hK

    -- N_k = Σ_i r_{ik}
    let Nk : Tensor α [nComponents] :=
      vecFromFn (n := nComponents) (fun k =>
        (List.finRange nSamples).foldl (fun acc i =>
          acc + get2 resp i k
        ) 0)

    -- π_k = N_k / N
    let weights : Tensor α [nComponents] :=
      let wRaw : Tensor α [nComponents] :=
        Tensor.dim (fun k =>
          let nk := Nk.getScalar k
          let w := m.weights.getScalar k
          if nk > 0 then Tensor.scalar (nk / (nSamples : α)) else Tensor.scalar w)
      -- Use the same compensated total as validation. A long rounded sum here would
      -- rescale every weight by an inaccurate denominator and invalidate the next EM step.
      let s := mixtureWeightTotal wRaw
      if s > 0 then scaleSpec wRaw (1 / s) else uniformWeights (α := α) (nComponents :=
        nComponents)

    -- μ_k = (1/N_k) Σ_i r_{ik} x_i
    let means : Tensor α [nComponents, nFeatures] :=
      Tensor.dim (fun k =>
        let nk := Nk.getScalar k
        if nk > 0 then
          Tensor.dim (fun f =>
            Tensor.scalar (
              (List.finRange nSamples).foldl (fun acc i =>
                let rik := get2 resp i k
                let xi := Tensor.unstack data i
                acc + rik * Tensor.getScalar xi f
              ) 0 / nk))
        else
          Tensor.unstack m.means k)

    -- Σ_k = (1/N_k) Σ_i r_{ik} (x_i-μ_k)(x_i-μ_k)ᵀ + εI
    let covariances : Tensor α [nComponents, nFeatures, nFeatures] :=
      Tensor.dim (fun k =>
        let nk := Nk.getScalar k
        let μ := Tensor.unstack means k
        if nk > 0 then
          let base :=
            matFromFn (n := nFeatures) (m := nFeatures) (fun row column =>
              -- Both entries of a symmetric pair use the same ordered product and reduction.
              -- Computing the two triangles separately can round them differently, causing the
              -- covariance-domain check to reject the next EM step.
              let (a, b) := if row.val ≤ column.val then (row, column) else (column, row)
              (List.finRange nSamples).foldl (fun acc i =>
                let rik := get2 resp i k
                let xi := Tensor.unstack data i
                let da := Tensor.getScalar xi a - Tensor.getScalar μ a
                let db := Tensor.getScalar xi b - Tensor.getScalar μ b
                acc + rik * da * db
              ) 0 / nk)
          let jitter := scaleSpec (identityTensorSpec nFeatures) Context.defaultEpsilon
          addSpec base jitter
        else
          Tensor.unstack m.covariances k)

    pure { weights := weights, means := means, covariances := covariances }

/--
Sum of negative `gmmLogLikelihoodSpec` scores over the dataset.

An empty dataset has score zero for a valid model; invalid parameters still return `none`.
-/
def gmmNegLogLikelihoodBatchedSpec {nSamples nComponents nFeatures : Nat}
  (m : GMMSpec α nComponents nFeatures)
  (data : Tensor α [nSamples, nFeatures])
  (hK : nComponents ≠ 0) : Option α :=
  if gmmParametersValidSpec m then
    (List.finRange nSamples).foldlM (init := 0) (fun acc i => do
      let xi := get data i
      let ll ← gmmLogLikelihoodSpec (α := α) (nComponents := nComponents)
        (nFeatures := nFeatures) m xi hK
      pure (acc - ll))
  else
    none

/--
Run `epochs` EM steps (deterministic).

Zero epochs preserve a valid model; invalid initial parameters still return `none`.
-/
def gmmEmTrainSpec {nSamples nComponents nFeatures : Nat}
  (epochs : Nat)
  (m : GMMSpec α nComponents nFeatures)
  (data : Tensor α [nSamples, nFeatures])
  (hK : nComponents ≠ 0) :
  Option (GMMSpec α nComponents nFeatures) :=
  if gmmParametersValidSpec m then
    (List.finRange epochs).foldlM (init := m) (fun cur _ =>
      gmmEmStepSpec (α := α) cur data hK)
  else
    none

end Spec
