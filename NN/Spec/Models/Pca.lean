/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Numerics

/-!
# PCA (spec model)

Principal Component Analysis is represented as a linear projection onto learned components,
plus an explicit mean for centering.

The exact model operations are the transform and inverse transform. A separate reference helper
below constructs a one-component approximation with power iteration; its name records that
numerical limitation explicitly.

The model follows the usual centered linear projection used by PCA. The fitting helper is
deliberately narrower: it approximates one leading component by power iteration.

References:

- Pearson (1901), "On Lines and Planes of Closest Fit to Systems of Points in Space".
  https://doi.org/10.1080/14786440109462720
- Hotelling (1933), "Analysis of a complex of statistical variables into principal components".
  https://doi.org/10.2307/2333955

## Implementation status

No API builder implements this model, and no theorem is proved about it. It is a reference
definition only.
-/

@[expose] public section


open TorchLean

namespace Spec

open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- Parameters for PCA as a linear map plus centering.

We store:

- `components : outDim × inDim` (rows are principal directions),
- `mean : inDim` (for centering),
- `explainedVariance : outDim` (eigenvalues for the selected components).

This matches the typical PCA API: you can `transform` to `outDim` coordinates and `inverse` back
to `inDim`.
-/
structure PCASpec (α : Type) [TorchLean.Storage α]
    (inDim outDim : Nat) where
  /-- Principal directions, one row for each output coordinate. -/
  components : Tensor α [outDim, inDim]
  /-- Coordinate-wise sample mean subtracted before projection. -/
  mean : Tensor α [inDim]
  /-- Covariance eigenvalue associated with each selected component. -/
  explainedVariance : Tensor α [outDim]

/-- Forward pass: center and project: `y = components · (x - mean)`. -/
def pcaForwardSpec {inDim outDim : Nat}
  (m : PCASpec α inDim outDim)
  (input : Tensor α [inDim]) :
  Tensor α [outDim] :=
  -- Center the data: x_centered = x - mean
  let centered := subSpec input m.mean
  -- Project onto principal components: y = components * x_centered
  matVecMulSpec m.components centered

/-- Inverse transform: reconstruct `x ≈ componentsᵀ · y + mean`. -/
def pcaInverseSpec {inDim outDim : Nat}
  (m : PCASpec α inDim outDim)
  (reduced : Tensor α [outDim]) :
  Tensor α [inDim] :=
  -- Reconstruct: x_reconstructed = components^T * reduced + mean
  let reconstructed := vecMatMulSpec reduced m.components
  addSpec reconstructed m.mean

/-- VJP contribution for `components`: outer product `dL/dy ⊗ (x - mean)`. -/
def pcaComponentsDerivSpec {inDim outDim : Nat}
  (m : PCASpec α inDim outDim)
  (input : Tensor α [inDim])
  (gradOutput : Tensor α [outDim]) :
  Tensor α [outDim, inDim] :=
  let centered := subSpec input m.mean
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      Tensor.scalar (Tensor.getScalar gradOutput i * Tensor.getScalar centered j)
    ))

/-- VJP contribution for `mean`: `dL/dmean = -componentsᵀ · dL/dy`. -/
def pcaMeanDerivSpec {inDim outDim : Nat}
  (m : PCASpec α inDim outDim)
  (gradOutput : Tensor α [outDim]) :
  Tensor α [inDim] :=
  negSpec (vecMatMulSpec gradOutput m.components)

/-- VJP contribution for `input`: `dL/dx = componentsᵀ · dL/dy`. -/
def pcaInputDerivSpec {inDim outDim : Nat}
  (m : PCASpec α inDim outDim)
  (gradOutput : Tensor α [outDim]) :
  Tensor α [inDim] :=
  vecMatMulSpec gradOutput m.components

/-- Gradients for a `PCASpec` projection. -/
structure PCAGradients (α : Type) [TorchLean.Storage α] (inDim outDim : Nat) where
  /-- Gradient with respect to the component matrix. -/
  componentsGradient : Tensor α [outDim, inDim]
  /-- Gradient with respect to the stored mean. -/
  meanGradient : Tensor α [inDim]
  /-- Gradient with respect to the projected input. -/
  inputGradient : Tensor α [inDim]

/-- Full backward pass for a PCA projection. -/
def pcaBackwardSpec {inDim outDim : Nat}
  (m : PCASpec α inDim outDim)
  (input : Tensor α [inDim])
  (gradOutput : Tensor α [outDim]) :
  PCAGradients α inDim outDim :=
  { componentsGradient := pcaComponentsDerivSpec m input gradOutput
    meanGradient := pcaMeanDerivSpec m gradOutput
    inputGradient := pcaInputDerivSpec m gradOutput }

/-- Approximate the leading PCA component using the scaled covariance matrix and power iteration.

Algorithm:

1. compute the mean and center the data,
2. form the covariance matrix `C = (1/(n-1)) Xᵀ X`,
3. run `iterations` power-iteration steps from the all-ones vector,
4. orient the resulting vector deterministically so results are reproducible.

The output has exactly one component. This is an executable approximation, not a theorem that the
returned vector is the dominant eigenvector. Such a theorem would require spectral hypotheses and
an error analysis. Numerical libraries generally use SVD or a convergent eigensolver for fitting.
-/
def pcaFitLeadingComponentApproxSpec {nSamples inDim : Nat}
  (data : Tensor α [nSamples, inDim])
  (iterations : Nat)
  (hSamples : 1 < nSamples) (hDim : 0 < inDim) :
  PCASpec α inDim 1 :=
  -- Compute mean
  have inst : Shape.HasNonemptyAxis 0 (Shape.dim nSamples (Shape.dim inDim Shape.scalar)) := by
    apply Shape.hasNonemptyAxisZeroOfNe
    intro h
    subst nSamples
    simp at hSamples
  let mean := reduceMean 0 data inst.proof

  -- Center the data
  let centeredData := Tensor.dim (fun i => subSpec (get data i) mean)

  -- Compute covariance matrix: C = (1/(n-1)) * X^T * X
  -- Using n-1 for unbiased estimator (Bessel's correction)
  let covariance := matMulSpec (swapAdjacentAxes centeredData 0) centeredData
  let covarianceScaled := scaleSpec covariance (1 / (nSamples - 1 : α))

  let (eigenvalue, eigenvector) :=
    powerIterationLeadingEigenpairSpec covarianceScaled iterations
  let first := item (get eigenvector ⟨0, hDim⟩)
  let sign : α := if first < 0 then -1 else 1
  let oriented := scaleSpec eigenvector sign

  {
    components := Tensor.dim (fun _ => oriented),
    mean := mean,
    explainedVariance := Tensor.dim (fun _ => Tensor.scalar eigenvalue)
  }

/-- Apply a fitted PCA transform to a batch of samples. -/
def pcaTransformSpec {nSamples inDim outDim : Nat}
  (m : PCASpec α inDim outDim)
  (data : Tensor α [nSamples, inDim]) :
  Tensor α [nSamples, outDim] :=
  Tensor.dim (fun i => pcaForwardSpec m (Tensor.unstack data i))

/-- Reconstruction error: `||x - inverse(transform(x))||_2^2` (sum of squared coordinates).

PyTorch analogy: `torch.sum((x - x_hat) ** 2)`.
-/
def pcaReconstructionErrorSpec {inDim outDim : Nat}
  (m : PCASpec α inDim outDim)
  (input : Tensor α [inDim]) (h : inDim ≠ 0) :
  α :=
  let reduced := pcaForwardSpec m input
  let reconstructed := pcaInverseSpec m reduced
  let error := subSpec input reconstructed
  let squaredError := squareSpec error
  have inst : Shape.HasNonemptyAxis 0 (Shape.dim inDim Shape.scalar) := by
    apply Shape.hasNonemptyAxisZeroOfNe h
  item (reduceSum 0 squaredError inst.proof)

/-- Cumulative explained variance, obtained by prefix-summing `explainedVariance`. -/
def pcaCumulativeExplainedVarianceSpec {α : Type} [Add α] [Zero α]
    [TorchLean.Storage α]
    {inDim outDim : Nat} (m : PCASpec α inDim outDim) :
    Tensor α [outDim] :=
  Tensor.dim (fun i =>
    let entries := (List.finRange outDim).take (i.val + 1)
    Tensor.scalar <| entries.foldl
      (fun acc j => acc + Tensor.getScalar m.explainedVariance j) 0)


end Spec
