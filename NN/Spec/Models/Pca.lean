/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Models.CommonHelpers

/-!
# PCA (spec model)

Principal Component Analysis is represented as a linear projection onto learned components,
plus an explicit mean for centering.

This file models only the *transform* (and inverse transform), not the procedure that learns
principal components from data.

PyTorch / ecosystem analogies:

- scikit-learn: `sklearn.decomposition.PCA` (fit + transform)
- PyTorch: `torch.pca_lowrank` or `torch.linalg.svd` (common building blocks)

References (background, not required to read the code):

- Pearson (1901), "On Lines and Planes of Closest Fit to Systems of Points in Space".
  https://doi.org/10.1080/14786440109462720
- Hotelling (1933), "Analysis of a complex of statistical variables into principal components".
  https://doi.org/10.2307/2333955
-/

@[expose] public section


namespace Spec

open Tensor

variable {α : Type} [Context α]

/-- Parameters for PCA as a linear map plus centering.

We store:

- `components : outDim × inDim` (rows are principal directions),
- `mean : inDim` (for centering),
- `explained_variance : outDim` (eigenvalues for the selected components).

This matches the typical PCA API: you can `transform` to `outDim` coordinates and `inverse` back
to `inDim`.
-/
structure PCASpec (α : Type) (inDim outDim : Nat) where
  /-- components. -/
  components : Tensor α (.dim outDim (.dim inDim .scalar))  -- Principal components (outDim × inDim)
  /-- mean. -/
  mean : Tensor α (.dim inDim .scalar)                     -- Data mean for centering
  /-- explained variance. -/
  explained_variance : Tensor α (.dim outDim .scalar)      -- Explained variance ratios

/-- Forward pass: center and project: `y = components · (x - mean)`. -/
def pcaForwardSpec {inDim outDim : Nat}
  (m : PCASpec α inDim outDim)
  (input : Tensor α (.dim inDim .scalar)) :
  Tensor α (.dim outDim .scalar) :=
  -- Center the data: x_centered = x - mean
  let centered := subSpec input m.mean
  -- Project onto principal components: y = components * x_centered
  matVecMulSpec m.components centered

/-- Batched forward pass: apply `pca_forward_spec` to each row. -/
def pcaBatchedForwardSpec {batch inDim outDim : Nat}
  (m : PCASpec α inDim outDim)
  (input : Tensor α (.dim batch (.dim inDim .scalar))) :
  Tensor α (.dim batch (.dim outDim .scalar)) :=
  match input with
  | Tensor.dim batch_fn =>
    Tensor.dim (fun i => pcaForwardSpec m (batch_fn i))

/-- Inverse transform: reconstruct `x ≈ componentsᵀ · y + mean`. -/
def pcaInverseSpec {inDim outDim : Nat}
  (m : PCASpec α inDim outDim)
  (reduced : Tensor α (.dim outDim .scalar)) :
  Tensor α (.dim inDim .scalar) :=
  -- Reconstruct: x_reconstructed = components^T * reduced + mean
  let reconstructed := vecMatMulSpec reduced m.components
  addSpec reconstructed m.mean

/-- VJP contribution for `components`: outer product `dL/dy ⊗ (x - mean)`. -/
def pcaComponentsDerivSpec {inDim outDim : Nat}
  (m : PCASpec α inDim outDim)
  (input : Tensor α (.dim inDim .scalar))
  (grad_output : Tensor α (.dim outDim .scalar)) :
  Tensor α (.dim outDim (.dim inDim .scalar)) :=
  let centered := subSpec input m.mean
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      match grad_output, centered with
      | Tensor.dim g_vals, Tensor.dim x_vals =>
        match g_vals i, x_vals j with
        | Tensor.scalar g, Tensor.scalar x => Tensor.scalar (g * x)
    ))

/-- VJP contribution for `mean`: `dL/dmean = -componentsᵀ · dL/dy`. -/
def pcaMeanDerivSpec {inDim outDim : Nat}
  (m : PCASpec α inDim outDim)
  (grad_output : Tensor α (.dim outDim .scalar)) :
  Tensor α (.dim inDim .scalar) :=
  negSpec (vecMatMulSpec grad_output m.components)

/-- VJP contribution for `input`: `dL/dx = componentsᵀ · dL/dy`. -/
def pcaInputDerivSpec {inDim outDim : Nat}
  (m : PCASpec α inDim outDim)
  (grad_output : Tensor α (.dim outDim .scalar)) :
  Tensor α (.dim inDim .scalar) :=
  vecMatMulSpec grad_output m.components

/-- Full backward pass returning `(dComponents, dMean, dInput)`. -/
def pcaBackwardSpec {inDim outDim : Nat}
  (m : PCASpec α inDim outDim)
  (input : Tensor α (.dim inDim .scalar))
  (grad_output : Tensor α (.dim outDim .scalar)) :
  (Tensor α (.dim outDim (.dim inDim .scalar)) ×
   Tensor α (.dim inDim .scalar) ×
   Tensor α (.dim inDim .scalar)) :=
  let d_components := pcaComponentsDerivSpec m input grad_output
  let d_mean := pcaMeanDerivSpec m grad_output
  let d_input := pcaInputDerivSpec m grad_output
  (d_components, d_mean, d_input)

/-- Fit PCA using the (scaled) covariance matrix and eigendecomposition.

Algorithm:

1. compute the mean and center the data,
2. form the covariance matrix `C = (1/(n-1)) Xᵀ X`,
3. compute eigenpairs of `C`,
4. take the top `nComponents` eigenvectors,
5. orient eigenvectors deterministically (sign convention) so results are reproducible.

Note: this is a spec/reference implementation. In numerical libraries, PCA is often implemented
via SVD for stability and performance.
-/
def pcaFitSpec {nSamples inDim : Nat}
  (data : Tensor α (.dim nSamples (.dim inDim .scalar)))
  (nComponents : Nat) (h1 : 0 < nComponents) (h2 : nComponents ≤ inDim) (h3 : nSamples ≠ 0) :
  PCASpec α inDim nComponents :=
  -- Compute mean
  have inst : Shape.valid_axis_inst 0 (Shape.dim nSamples (Shape.dim inDim Shape.scalar)) := by
    apply Shape.validAxisInstZeroAlt h3
  let mean := reduceMeanAuto 0 inst data

  -- Center the data
  let centered_data := Tensor.dim (fun i => subSpec (get data i) mean)

  -- Compute covariance matrix: C = (1/(n-1)) * X^T * X
  -- Using n-1 for unbiased estimator (Bessel's correction)
  let covariance := matMulSpec (matrixTransposeSpec centered_data) centered_data
  let n_minus_1 := max 1 (nSamples - 1) -- Ensure we don't divide by zero
  let covariance_scaled := scaleSpec covariance (1 / (n_minus_1 : α))

  -- Perform eigendecomposition of covariance matrix
  -- eigendecomp returns (eigenvalues, eigenvectors) where eigenvectors are columns
  let (eigenvalues, eigenvectors) := eigendecompSpec covariance_scaled

  -- Sort eigenvalues and eigenvectors in descending order
  let sorted_indices := argsortDescendingSpec eigenvalues
  let sorted_eigenvalues := gatherSpec eigenvalues sorted_indices
  let sorted_eigenvectors := gatherColumnsSpec eigenvectors sorted_indices

  -- Take the first nComponents eigenvectors as principal components
  -- These are the eigenvectors corresponding to the largest eigenvalues
  let components := sliceColumnsSpec sorted_eigenvectors 0 nComponents h2
  have h4 : Shape.dim inDim (Shape.dim (nComponents - 0) Shape.scalar) = Shape.dim inDim (Shape.dim
    nComponents Shape.scalar) := by
    simp
  let components' := tensorCast (Shape.dim inDim (Shape.dim (nComponents) Shape.scalar)) h4
    components

  -- Extract the explained variance (eigenvalues) for the selected components
  let explained_variance := sliceRangeSpec sorted_eigenvalues 0 nComponents h2

  -- Ensure components are properly oriented (optional: enforce deterministic sign)
  let components_oriented := orientComponentsSpec components' h1
  let components_reshaped := matrixTransposeSpec components_oriented

  {
    components := components_reshaped,
    mean := mean,
    explained_variance := explained_variance
  }

/-- Apply a fitted PCA transform to a batch of samples. -/
def pcaTransformSpec {nSamples inDim outDim : Nat}
  (m : PCASpec α inDim outDim)
  (data : Tensor α (.dim nSamples (.dim inDim .scalar))) :
  Tensor α (.dim nSamples (.dim outDim .scalar)) :=
  match data with
  | Tensor.dim batch_fn =>
    Tensor.dim (fun i => pcaForwardSpec m (batch_fn i))

/-- Reconstruction error: `||x - inverse(transform(x))||_2^2` (sum of squared coordinates).

PyTorch analogy: `torch.sum((x - x_hat) ** 2)`.
-/
def pcaReconstructionErrorSpec {inDim outDim : Nat}
  (m : PCASpec α inDim outDim)
  (input : Tensor α (.dim inDim .scalar)) (h : inDim ≠ 0) :
  α :=
  let reduced := pcaForwardSpec m input
  let reconstructed := pcaInverseSpec m reduced
  let error := subSpec input reconstructed
  let squared_error := squareSpec error
  have inst : Shape.valid_axis_inst 0 (Shape.dim inDim Shape.scalar) := by
    apply Shape.validAxisInstZeroAlt h
  toScalar (reduceSumAuto 0 squared_error)

/-- Explained variance (eigenvalues of the selected components).

If you want the *ratio* (normalized to sum to `1`), you need to divide by the total variance of the
original data; this file keeps just the raw eigenvalues.
-/
def pcaExplainedVarianceRatioSpec {inDim outDim : Nat}
  (m : PCASpec α inDim outDim) :
  Tensor α (.dim outDim .scalar) :=
  m.explained_variance

/-- Cumulative explained variance (prefix sums of `explained_variance`). -/
def pcaCumulativeVarianceSpec {α : Type} [Add α] [Zero α]
    {inDim outDim : Nat} (m : PCASpec α inDim outDim) :
    Tensor α (.dim outDim .scalar) :=
  match m.explained_variance with
  | Tensor.dim f =>
    Tensor.dim (fun i =>
      -- For each position i, sum explained variances from 0 to i
      let rec sum_to_index (j : Nat) (acc : α) : α :=
        if j > i.val then acc
        else
          if h : j < outDim then
            match f ⟨j, h⟩ with
            | Tensor.scalar x => sum_to_index (j + 1) (acc + x)
          else acc
      Tensor.scalar (sum_to_index 0 0)
    )


end Spec
