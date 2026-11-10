/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Layers.Activation
public import TorchLean.Tensor.LinearAlgebra.Algorithms
public import TorchLean.Tensor.Movement
public import TorchLean.Tensor.Layers.Softmax
public import TorchLean.Tensor.Contract

/-!
# Gaussian Mixture Model (GMM) (spec model)

The model has `nComponents` multivariate Gaussians over `nFeatures` features, with

- mixing weights $\pi$,
- means $\mu$, and
- covariances $\Sigma$.

`GMM.forward` computes per-component log scores for a single input:

$$
\log \pi_k + \log \mathcal{N}(x \mid \mu_k, \Sigma_k).
$$

PyTorch analogies:

- `torch.distributions.MultivariateNormal` for $\mathcal{N}(x \mid \mu, \Sigma)$,
- `torch.distributions.MixtureSameFamily` for mixture distributions,
- `torch.softmax` for turning per-component log-probabilities into responsibilities.

Every function that evaluates the model first runs `paramsValid` and returns `none` when
it fails. The weights must be positive with a compensated total within `Context.defaultEpsilon` of
one, and each covariance must pass `GMM.isPositiveDefinite`: symmetry under backend
equality and a positive computed determinant for every leading principal submatrix. Over exact
reals that is Sylvester's criterion, but on a rounded backend a pass does not certify positive
definiteness over the reals. The scores use the stored weights as they are, so the sum of their
exponentials is a probability density in $x$ when the weights sum to one. The default tolerance
applies on exact backends too, and callers who need a stricter gate can run
`GMM.weightsValid` with tolerance zero.

Determinants and inverses come from `TorchLean.Tensor.LinearAlgebra.Algorithms`, where the
determinant is a cofactor expansion. Its cost grows factorially with `nFeatures`, so these specs
suit small feature counts and reference use.

References (background, not required to read the code):

- Dempster, Laird, Rubin (1977), "Maximum Likelihood from Incomplete Data via the EM Algorithm",
  Journal of the Royal Statistical Society, Series B 39(1):
  https://doi.org/10.1111/j.2517-6161.1977.tb01600.x
- Bishop (2006), "Pattern Recognition and Machine Learning", Springer, Chapter 9 (Mixture Models
  and EM), Section 9.2.2 (EM for Gaussian mixtures):
  https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/
- scikit-learn 1.9.1, `sklearn.mixture.GaussianMixture` and `_estimate_gaussian_parameters` in
  `sklearn/mixture/_gaussian_mixture.py`:
  https://scikit-learn.org/1.9/modules/generated/sklearn.mixture.GaussianMixture.html

## Implementation status

No API builder implements this model. The module holds definitions only, with no theorems about
EM convergence or the derivative formulas.
-/

public section


open FloatLib.Numerics (MathFunctions)

namespace TorchLean

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

/-! ## Parameters -/

/-- Parameters of a Gaussian mixture model (GMM). -/
structure GMM (α : Type) [Storage α]
    (nComponents nFeatures : Nat) where
  /-- Mixing weights $\pi_k$. Model operations require every weight to be strictly positive
  and their compensated total to be within `Context.defaultEpsilon` of one. -/
  weights : Tensor α [nComponents]
  /-- Component means $\mu_k$. -/
  means : Tensor α [nComponents, nFeatures]
  /-- Component covariance matrices $\Sigma_k$. Model operations require each one to pass
  `isPositiveDefinite`. -/
  covariances : Tensor α [nComponents, nFeatures, nFeatures]

/-- The leading `k × k` principal submatrix of a square matrix. -/
private def GMM.leadingPrincipalSubmatrix {n : Nat}
    (matrix : Tensor α [n, n]) (k : Nat) (hk : k ≤ n) :
    Tensor α [k, k] :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      Tensor.scalar (get2 matrix (i.castLE hk) (j.castLE hk))))

/-- Whether a matrix is symmetric under the scalar backend's equality operation. -/
def GMM.isSymmetric {n : Nat}
    (matrix : Tensor α [n, n]) : Bool :=
  (List.finRange n).all (fun i =>
    (List.finRange n).all (fun j => get2 matrix i j == get2 matrix j i))

/--
Sylvester's criterion as a Boolean test: the matrix compares symmetric under backend equality, and
the determinant of every leading `k × k` submatrix with `1 ≤ k ≤ n` compares positive.

Over exact real arithmetic a pass means the matrix is positive definite. A rounded backend computes
rounded determinants, so there a pass does not certify positive definiteness over the reals.
-/
def GMM.isPositiveDefinite {n : Nat}
    (matrix : Tensor α [n, n]) : Bool :=
  isSymmetric matrix &&
    (List.finRange n).all (fun i =>
      let k := i.val + 1
      have hk : k ≤ n := Nat.succ_le_iff.mpr i.isLt
      let leading := leadingPrincipalSubmatrix matrix k hk
      Context.gtBool (Tensor.item (determinant leading)) 0)

/--
Kahan (compensated) sum of the mixture weights: the rounding error of each addition is subtracted
from the next weight before that weight is added.

With many components, a plain left-to-right sum of small weights can drift far enough from one to
reject a normalized mixture. Over exact arithmetic the correction is zero and this is the ordinary
sum.
-/
private def GMM.mixtureWeightTotal {n : Nat} (weights : Tensor α [n]) : α :=
  ((List.finRange n).foldl (fun (state : α × α) i =>
    let adjusted := weights.getScalar i - state.2
    let total := state.1 + adjusted
    (total, (total - state.1) - adjusted)) (0, 0)).1

/--
Checks that every mixture weight is positive and that their compensated total is within `tolerance`
of one in absolute error. A `tolerance` outside `[0, 1)` fails the check.

In binary floating point `1 / n` is exact only when `n` is a power of two, so a test against
exactly one could reject `init` before its first forward pass. The compensated total keeps
the reduction error small, and the tolerance absorbs the rounding in the weights themselves. The
default `Context.defaultEpsilon` is the backend's configured numerical safeguard and applies to
exact scalar backends as well. A zero tolerance demands a computed total of exactly one. Model
operations use the default, so callers who want that stricter gate run this check themselves.
-/
def GMM.weightsValid {n : Nat} (weights : Tensor α [n])
    (tolerance : α := Context.defaultEpsilon) : Bool :=
  let toleranceValid := (tolerance == 0 || Context.gtBool tolerance 0) &&
    Context.gtBool 1 tolerance
  let positive :=
    (List.finRange n).all (fun i => Context.gtBool (weights.getScalar i) 0)
  let discrepancy := MathFunctions.abs (mixtureWeightTotal weights - 1)
  toleranceValid && positive &&
    (discrepancy == tolerance || Context.gtBool tolerance discrepancy)

/--
Domain check for GMM parameters: at least one component and one feature, weights that pass
`GMM.weightsValid` at its default tolerance, and covariances that all pass
`isPositiveDefinite`.

Accepted weights are used as stored, without rescaling them to sum to one.
-/
def GMM.paramsValid {nComponents nFeatures : Nat}
    (m : GMM α nComponents nFeatures) : Bool :=
  nComponents != 0 && nFeatures != 0 &&
    weightsValid m.weights &&
    (List.finRange nComponents).all (fun k =>
      isPositiveDefinite (Tensor.unstack m.covariances k))

/-- Per-component log scores for a single input.

Given $x \in \mathbb{R}^d$, each component contributes

$$
\log \pi_k-\frac12\left(
  (x-\mu_k)^\mathsf{T}\Sigma_k^{-1}(x-\mu_k)
  +\log\det\Sigma_k+d\log(2\pi)
\right).
$$

The result is `none` when the parameters fail `GMM.paramsValid`. The scores are the logits
of the responsibilities, and `expectation` takes their softmax to get the posterior
$P(z=k\mid x)$.

PyTorch analogy: the vector that `MixtureSameFamily.log_prob` passes to its final `logsumexp`
(component log-probabilities plus log mixing weights).
-/
def GMM.forward {nComponents nFeatures : Nat}
  (m : GMM α nComponents nFeatures)
  (input : Tensor α [nFeatures]) :
  Option (Tensor α [nComponents]) :=
  if paramsValid m then
    sequenceFin (fun k => do
      let mean := Tensor.unstack m.means k
      let covariance := Tensor.unstack m.covariances k
      let diff := input - mean
      let covInv ← inverse? covariance
      let det := Tensor.item (determinant covariance)
      let w := m.weights.getScalar k
      let quadraticForm := dot diff (Tensor.contract covInv diff _ _ [nFeatures] .matVec)
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
def GMM.expectation {nComponents nFeatures : Nat}
  (m : GMM α nComponents nFeatures)
  (input : Tensor α [nFeatures]) :
  Option (Tensor α [nComponents]) := do
  let componentLogProbs ← forward m input
  pure (Tensor.softmax (α := α) componentLogProbs ⟨0, Nat.one_pos⟩)

/-!
## Backward/VJP (for `GMM.forward`)

`GMM.forward` is vector-valued, with one log score per component, and the gradients below are
vector-Jacobian products (VJPs) of that map for an upstream gradient $g$. Responsibilities
$\gamma=\operatorname{softmax}(\text{component log scores})$ do not appear in them.

Responsibilities enter through a scalar objective that combines the components, such as the
mixture log-likelihood $\operatorname{logsumexp}(\text{component log scores})$. Its gradient with
respect to the scores is $\gamma$, and passing that vector to `GMM.backward` as `gradOutput`
gives the gradient of the log-likelihood.
-/

/-- VJP of `forward` with respect to the mixing weights $\pi$.

Entry $k$ is $g_k/\pi_k$, because $y_k=\log\pi_k+\cdots$ has
$\partial y_k/\partial\pi_k=1/\pi_k$.
-/
def GMM.weightsDeriv {nComponents nFeatures : Nat}
  (m : GMM α nComponents nFeatures)
  (gradOutput : Tensor α [nComponents]) :
  Option (Tensor α [nComponents]) :=
  if paramsValid m then
    some (gradOutput / m.weights)
  else
    none

/-- Mean gradient: the VJP of `forward` with respect to $\mu$, whose row $k$ is $g_k$
times the gradient of component $k$. For a single component:

$$
\frac{\partial}{\partial\mu}\log\mathcal{N}(x\mid\mu,\Sigma)
=\frac12\left(\Sigma^{-1}+\Sigma^{-\mathsf{T}}\right)(x-\mu).
$$

For a symmetric covariance this reduces to the familiar $\Sigma^{-1}(x-\mu)$.
-/
def GMM.meansDeriv {nComponents nFeatures : Nat}
  (m : GMM α nComponents nFeatures)
  (input : Tensor α [nFeatures])
  (gradOutput : Tensor α [nComponents]) :
  Option (Tensor α [nComponents, nFeatures]) :=
  if paramsValid m then
    sequenceFin (fun k => do
      let mean_k := Tensor.unstack m.means k
      let covariance_k := Tensor.unstack m.covariances k
      let grad_k := Tensor.unstack gradOutput k
      let diff := input - mean_k
      let covInv ← inverse? covariance_k
      let covInvT := Tensor.permute covInv
          (Shape.adjacentSwap [nFeatures, nFeatures] 0) [nFeatures, nFeatures]
      let weightedDiff := Tensor.scale
        (Tensor.contract covInv diff _ _ [nFeatures] .matVec +
            Tensor.contract covInvT diff _ _ [nFeatures] .matVec)
        (1 / 2)
      pure (Tensor.scale weightedDiff grad_k.item))
  else
    none

/-- Input gradient: the VJP of `forward` with respect to $x$. For one component:

$$
\frac{\partial}{\partial x}\log\mathcal{N}(x\mid\mu,\Sigma)
=-\frac12\left(\Sigma^{-1}+\Sigma^{-\mathsf{T}}\right)(x-\mu).
$$

The result sums these vectors over the components, weighted by the upstream gradient $g_k$.
-/
def GMM.inputDeriv {nComponents nFeatures : Nat}
  (m : GMM α nComponents nFeatures)
  (input : Tensor α [nFeatures])
  (gradOutput : Tensor α [nComponents]) :
  Option (Tensor α [nFeatures]) :=
  if paramsValid m then do
    let perComponent ← sequenceFin (fun k => do
        let mean_k := Tensor.unstack m.means k
        let covariance_k := Tensor.unstack m.covariances k
        let gk := Tensor.unstack gradOutput k
        let diff := input - mean_k
        let covInv ← inverse? covariance_k
        let covInvT := Tensor.permute covInv
            (Shape.adjacentSwap [nFeatures, nFeatures] 0) [nFeatures, nFeatures]
        let v := Tensor.scale
          (Tensor.contract covInv diff _ _ [nFeatures] .matVec +
              Tensor.contract covInvT diff _ _ [nFeatures] .matVec)
          (1 / 2)
        pure (Tensor.scale v ((-1) * gk.item)))
    pure (castShape (reduce .sum perComponent [⟨0, by simp⟩] false)
        (Shape.reduce_singleton _ _))
  else
    none

/-- Covariance gradient: the VJP of `forward` with respect to $\Sigma$, whose slice $k$ is
$g_k$ times the gradient of component $k$. For one component:

$$
\frac{\partial}{\partial\Sigma}\log\mathcal{N}(x\mid\mu,\Sigma)
=\frac12\left(
  \Sigma^{-\mathsf{T}}(x-\mu)(x-\mu)^\mathsf{T}\Sigma^{-\mathsf{T}}
  -\Sigma^{-\mathsf{T}}
\right).
$$

The formula treats the entries of $\Sigma$ as independent and does not symmetrize the result.
-/
def GMM.covariancesDeriv {nComponents nFeatures : Nat}
  (m : GMM α nComponents nFeatures)
  (input : Tensor α [nFeatures])
  (gradOutput : Tensor α [nComponents]) :
  Option (Tensor α [nComponents, nFeatures, nFeatures]) :=
  if paramsValid m then
    sequenceFin (fun k => do
      let mean_k := Tensor.unstack m.means k
      let covariance_k := Tensor.unstack m.covariances k
      let grad_k := Tensor.unstack gradOutput k
      let diff := input - mean_k
      let outerProduct := Tensor.zipWith (· * ·)
          (Tensor.broadcast diff _ _
            (Shape.broadcastDims_axis [nFeatures, nFeatures] ⟨0, Nat.succ_pos _⟩))
          (Tensor.broadcast diff _ _
            (Shape.broadcastDims_axis [nFeatures, nFeatures] ⟨1, Nat.one_lt_two⟩))
      let covInv ← inverse? covariance_k
      let covInvT := Tensor.permute covInv
          (Shape.adjacentSwap [nFeatures, nFeatures] 0) [nFeatures, nFeatures]
      let temp1 := Tensor.matmul covInvT outerProduct
      let temp2 := Tensor.matmul temp1 covInvT
      let gradSigma := temp2 - covInvT
      pure (Tensor.scale gradSigma ((1 / 2) * grad_k.item)))
  else
    none

/-- Gradients returned by `GMM.backward`: one field per parameter tensor of `GMM` and one
for the input. -/
structure GMM.Grads (α : Type) [Storage α] (nComponents nFeatures : Nat) where
  /-- Gradient with respect to the mixture weights. -/
  weightsGradient : Tensor α [nComponents]
  /-- Gradient with respect to the component means. -/
  meansGradient : Tensor α [nComponents, nFeatures]
  /-- Gradient with respect to the component covariance matrices. -/
  covariancesGradient : Tensor α [nComponents, nFeatures, nFeatures]
  /-- Gradient with respect to the observed point. -/
  inputGradient : Tensor α [nFeatures]

/-- Backward/VJP for `forward`, with all four gradients in one record.

The mean, covariance and input gradients use $\Sigma_k^{-1}$, which a singular covariance does not
have, so the record is optional. It is `none` when the parameters fail `GMM.paramsValid` or
a covariance inverse cannot be computed.
-/
def GMM.backward {nComponents nFeatures : Nat}
  (m : GMM α nComponents nFeatures)
  (input : Tensor α [nFeatures])
  (gradOutput : Tensor α [nComponents]) :
  Option (Grads α nComponents nFeatures) := do
  let dWeights ← weightsDeriv m gradOutput
  let dMeans ← meansDeriv m input gradOutput
  let dCovariances ← covariancesDeriv m input gradOutput
  let dInput ← inputDeriv m input gradOutput
  pure
    { weightsGradient := dWeights
      meansGradient := dMeans
      covariancesGradient := dCovariances
      inputGradient := dInput }

/-- Deterministic default parameters: uniform weights `1 / nComponents`, zero means and identity
covariances.

Every component starts the same, and an EM step keeps identical components identical, so training
from these parameters needs distinct initial means to separate clusters.
-/
def GMM.init {nComponents nFeatures : Nat} :
  GMM α nComponents nFeatures :=
  let weights : Tensor α [nComponents] := Tensor.full [nComponents] (Context.divNat 1 nComponents)
  let means : Tensor α [nComponents, nFeatures] := Tensor.zeros [nComponents, nFeatures]
  let covariances : Tensor α [nComponents, nFeatures, nFeatures] :=
    Tensor.dim (fun _ => Tensor.identity nFeatures)
  {
    weights := weights,
    means := means,
    covariances := covariances
  }

/--
Mixture log-likelihood $\log p(x)$, the `Tensor.logsumexp` of the `GMM.forward` scores.

Mathematically,
$$
\log p(x)=\log\!\left(
  \sum_k \exp\!\left(\log p(x\mid z_k)+\log\pi_k\right)
\right).
$$

This is the log of a normalized mixture density when the stored weights sum to one.
The default domain check allows `Context.defaultEpsilon` error in that sum. Accepted weights
are used directly without renormalization. The sum is `Tensor.logsumexp` over the components. A
mixture without components fails `paramsValid`, so the result is `none`.
-/
def GMM.logLikelihood {nComponents nFeatures : Nat}
  (m : GMM α nComponents nFeatures)
  (input : Tensor α [nFeatures]) :
  Option α := do
  let componentLogProbs ← forward m input
  pure (Tensor.item (castShape
    (Tensor.logsumexp componentLogProbs [⟨0, Nat.zero_lt_one⟩] false)
    (Shape.reduce_singleton _ _)))

/-!
## Classical training: EM for Gaussian mixtures

GMMs are usually trained with the Expectation–Maximization (EM) algorithm:

- **E-step**: compute responsibilities $r_{ik}=P(z=k\mid x_i)$ for each sample/component.
- **M-step**: update $\pi$, $\mu$, and $\Sigma$ from the weighted sufficient statistics.

`GMM.responsibilitiesBatch` applies `GMM.expectation` to every sample, and `GMM.emStep`
builds one deterministic EM update on those responsibilities.

Bishop's Section 9.2.2 has the EM updates for Gaussian mixtures, and `GMM.emStep` is our
hand-written version of them. The M-step computes the counts $N_k$, the weights $N_k/N$ and the
means, then centers the covariances on those updated means, as scikit-learn's `GaussianMixture`
does. On `Float` the jitter is `1e-6`, which is also scikit-learn's default `reg_covar`, so a
weighted variance of 4 comes out as 4.000001. scikit-learn adds ten times the machine epsilon to
every $N_k$ and so re-estimates every component, even one with no responsibility at all.

Numerical notes:
- When a component's total responsibility does not compare strictly positive, its mean, covariance
  and raw weight are kept. All raw weights then pass through the common normalization step, which
  falls back to uniform weights if their total does not compare positive. There is no near-zero
  cutoff.
- For a positive total, the covariance update adds diagonal jitter
  $\mathtt{Context.defaultEpsilon}\,I$. Under rounding the updated covariance can fail
  `isPositiveDefinite` despite the jitter, and the next EM step then returns `none`.
-/

/--
Batched responsibilities: apply `expectation` to each sample.

Invalid model parameters return `none`, including for an empty dataset.
-/
def GMM.responsibilitiesBatch {nSamples nComponents nFeatures : Nat}
  (m : GMM α nComponents nFeatures)
  (data : Tensor α [nSamples, nFeatures]) :
  Option (Tensor α [nSamples, nComponents]) :=
  if paramsValid m then
    sequenceFin (fun i => expectation (α := α) (nComponents := nComponents)
      (nFeatures := nFeatures) m (Tensor.unstack data i))
  else
    none

/--
One EM step for a batched dataset.

An empty dataset returns a valid model unchanged, and invalid parameters give `none`.
-/
def GMM.emStep {nSamples nComponents nFeatures : Nat}
  (m : GMM α nComponents nFeatures)
  (data : Tensor α [nSamples, nFeatures]) :
  Option (GMM α nComponents nFeatures) :=
  if _hN : nSamples = 0 then
    if paramsValid m then some m else none
  else do
    let resp ← responsibilitiesBatch (α := α) (nSamples := nSamples)
      (nComponents := nComponents) (nFeatures := nFeatures) m data

    -- N_k = Σ_i r_{ik}
    let Nk : Tensor α [nComponents] :=
      Tensor.ofFn (fun k =>
        (List.finRange nSamples).foldl (fun acc i =>
          acc + get2 resp i k
        ) 0)

    -- π_k = N_k / N
    let weights : Tensor α [nComponents] :=
      let wRaw : Tensor α [nComponents] :=
        Tensor.dim (fun k =>
          let nk := Nk.getScalar k
          let w := m.weights.getScalar k
          if nk > 0 then Tensor.scalar (Context.divNat nk nSamples) else Tensor.scalar w)
      -- Use the same compensated total as validation. A long rounded sum here would
      -- rescale every weight by an inaccurate denominator and invalidate the next EM step.
      let s := mixtureWeightTotal wRaw
      if s > 0 then Tensor.scale wRaw (1 / s)
      else Tensor.full [nComponents] (Context.divNat 1 nComponents)

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
            Tensor.matrix (m := nFeatures) (n := nFeatures) (fun row column =>
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
          let jitter := Tensor.scale (Tensor.identity nFeatures) Context.defaultEpsilon
          base + jitter
        else
          Tensor.unstack m.covariances k)

    pure { weights := weights, means := means, covariances := covariances }

/--
Sum of negative `logLikelihood` scores over the dataset.

A valid model scores zero on an empty dataset, and invalid parameters give `none`.
-/
def GMM.negLogLikelihoodBatch {nSamples nComponents nFeatures : Nat}
  (m : GMM α nComponents nFeatures)
  (data : Tensor α [nSamples, nFeatures]) : Option α :=
  if paramsValid m then
    (List.finRange nSamples).foldlM (init := 0) (fun acc i => do
      let xi := Tensor.unstack data i
      let ll ← logLikelihood (α := α) (nComponents := nComponents)
        (nFeatures := nFeatures) m xi
      pure (acc - ll))
  else
    none

/--
Runs `epochs` deterministic EM steps from `m`.

Zero epochs return a valid model unchanged, and invalid initial parameters give `none`.
-/
def GMM.emTrain {nSamples nComponents nFeatures : Nat}
  (epochs : Nat)
  (m : GMM α nComponents nFeatures)
  (data : Tensor α [nSamples, nFeatures]) :
  Option (GMM α nComponents nFeatures) :=
  if paramsValid m then
    (List.finRange epochs).foldlM (init := m) (fun cur _ =>
      emStep (α := α) cur data)
  else
    none

end TorchLean
