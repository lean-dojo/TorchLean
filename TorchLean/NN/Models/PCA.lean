/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.LinearAlgebra.Algorithms
public import TorchLean.Tensor.Layers.Linear

/-!
# PCA (spec model)

Principal Component Analysis is modeled as a centered linear projection: subtract a stored mean,
then multiply by a matrix whose rows are the learned components. The transform and the inverse
transform are exact operations on these parameters. Fitting is approximate, and its name says so:
`fitLeadingComponent` estimates one leading component by power iteration.

The maximum-variance formulation in Bishop's Section 12.1.1 is the definition we coded by hand:
the leading component is a unit eigenvector for the largest eigenvalue of the sample covariance.
scikit-learn's `PCA` gets its components from an SVD of the centered data. We run power iteration
on the covariance, which finds the top direction and nothing else. The divisor is `n - 1`, as in
scikit-learn's `explained_variance_`, so the samples `-1, 0, 1` of one coordinate give variance 1,
where dividing by `n` would give 2/3.

References:

- Pearson (1901), "On Lines and Planes of Closest Fit to Systems of Points in Space", The London,
  Edinburgh, and Dublin Philosophical Magazine and Journal of Science 2(11), 559-572.
  https://doi.org/10.1080/14786440109462720
- Hotelling (1933), "Analysis of a complex of statistical variables into principal components",
  Journal of Educational Psychology 24(6), 417-441. https://doi.org/10.1037/h0071325
- Bishop (2006), "Pattern Recognition and Machine Learning", Springer, Section 12.1.1 (Maximum
  variance formulation). https://doi.org/10.1007/978-0-387-45528-0
- scikit-learn 1.9.1, `sklearn.decomposition.PCA` (SVD solver, `explained_variance_`).
  https://scikit-learn.org/1.9/modules/generated/sklearn.decomposition.PCA.html

## Implementation status

No API builder implements this model, and no theorem is proved about it. It is a reference
definition only.
-/

@[expose] public section


open FloatLib.Numerics (MathFunctions)

namespace TorchLean

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

/-- Parameters of a PCA model: `outDim` principal directions over `inDim` input coordinates, the
mean used for centering and the variance that each direction explains.

The layout follows the usual PCA API. `PCA.forward` (or `PCA.transform` for a batch) maps an
input to `outDim` coordinates, and `PCA.inverse` maps coordinates back to `inDim`.
-/
structure PCA (α : Type) [Storage α]
    (inDim outDim : Nat) where
  /-- Principal directions, one row for each output coordinate. -/
  components : Tensor α [outDim, inDim]
  /-- Coordinate-wise sample mean subtracted before projection. -/
  mean : Tensor α [inDim]
  /-- Covariance eigenvalue associated with each selected component. -/
  explainedVariance : Tensor α [outDim]

/-- Center the input and project it onto the components: `y = components · (x - mean)`. -/
def PCA.forward {inDim outDim : Nat}
  (m : PCA α inDim outDim)
  (input : Tensor α [inDim]) :
  Tensor α [outDim] :=
  let centered := input - m.mean
  Tensor.contract m.components centered _ _ [outDim] .matVec

/-- Inverse transform: reconstruct `x ≈ componentsᵀ · y + mean`. -/
def PCA.inverse {inDim outDim : Nat}
  (m : PCA α inDim outDim)
  (reduced : Tensor α [outDim]) :
  Tensor α [inDim] :=
  let reconstructed := Tensor.contract reduced m.components _ _ [inDim] .vecMat
  reconstructed + m.mean

/-- Gradients of `PCA.forward` with respect to its parameters and its input. -/
structure PCA.Grads (α : Type) [Storage α] (inDim outDim : Nat) where
  /-- Gradient with respect to the component matrix. -/
  componentsGradient : Tensor α [outDim, inDim]
  /-- Gradient with respect to the stored mean. -/
  meanGradient : Tensor α [inDim]
  /-- Gradient with respect to the input before centering. -/
  inputGradient : Tensor α [inDim]

/-- Backward pass of `PCA.forward`. The component gradient is the outer product
`gradOutput ⊗ (x - mean)`, and the input and mean gradients are `componentsᵀ · gradOutput` and its
negation. -/
def PCA.backward {inDim outDim : Nat}
  (m : PCA α inDim outDim)
  (input : Tensor α [inDim])
  (gradOutput : Tensor α [outDim]) :
  Grads α inDim outDim :=
  let centered := input - m.mean
  let gradInput := Tensor.contract gradOutput m.components _ _ [inDim] .vecMat
  { componentsGradient := Linear.weightsDeriv centered gradOutput
    meanGradient := Tensor.neg gradInput
    inputGradient := gradInput }

/-- Approximate one leading PCA component with deterministic multistart power iteration.

The fit centers the samples and uses the covariance `Xᵀ X / (n - 1)`. It runs `iterations` steps
from the normalized all-ones vector and from every coordinate vector, then keeps the result with
the largest Rayleigh quotient. The coordinate starts span the input space, so a dominant
eigenspace cannot be orthogonal to all of them, but convergence depends on the spectral gap and
the iteration count.

The covariance is divided by its largest entry magnitude before iterating, so normalizing an
iterate never squares the covariance's original magnitude, and the eigenvalue is scaled back
afterwards. Forming the covariance itself can overflow. Zero covariance keeps a unit direction
and reports zero variance. Eigenvectors are determined up to sign, so the single output component
is negated when its first coordinate is negative. With `inDim + 1` starts the iteration cost is
cubic in `inDim`.
-/
def PCA.fitLeadingComponent {nSamples inDim : Nat}
  (data : Tensor α [nSamples, inDim])
  (iterations : Nat)
  (hSamples : 1 < nSamples) (hDim : 0 < inDim) :
  PCA α inDim 1 :=
  -- The mean sums over axis 0, which `hSamples` makes nonempty.
  have inst : Shape.HasNonemptyAxis 0 (Shape.dim nSamples (Shape.dim inDim Shape.scalar)) := by
    apply Shape.hasNonemptyAxis_zero_of_ne
    intro h
    subst nSamples
    simp at hSamples
  let mean : Tensor α [inDim] := Tensor.map (Context.divNat · nSamples)
    (castShape (reduce .sum data [⟨0, inst.proof.axisInBounds.proof⟩] false)
      (Shape.reduce_singleton _ _))

  let centeredData := Tensor.dim (fun i => Tensor.unstack data i - mean)

  -- Sample covariance Xᵀ X / (n - 1). The n - 1 (Bessel's correction) makes it unbiased.
  let covariance := Tensor.matmul (Tensor.permute centeredData
      (Shape.adjacentSwap [nSamples, inDim] 0) [inDim, nSamples]) centeredData
  let covarianceScaled := Tensor.scale covariance (Context.divNat 1 (nSamples - 1))

  -- Rescale before taking squared norms, which otherwise lose very small or large covariance.
  let coordinates := List.finRange inDim
  let matrixScale := coordinates.foldl (fun largest i =>
    coordinates.foldl (fun largest j =>
      Max.max largest (MathFunctions.abs (getScalar (Tensor.unstack covarianceScaled i) j)))
        largest) 0
  let (iterationMatrix, restoreScale) := if matrixScale > 0 then
    (Tensor.map (fun value => value / matrixScale) covarianceScaled, matrixScale)
  else (covarianceScaled, 1)
  let initial := powerIteration iterationMatrix iterations
  -- A single fixed start can miss the leading component, even when it finds a nonzero eigenvalue.
  let (scaledEigenvalue, eigenvector) := coordinates.foldl (fun best coordinate =>
    let start := Tensor.ofFn (fun i => if i == coordinate then (1 : α) else 0)
    let candidate :=
      powerIteration iterationMatrix iterations (initial := start)
    if candidate.1 > best.1 then candidate else best) initial
  let eigenvalue := scaledEigenvalue * restoreScale

  let first := item (Tensor.unstack eigenvector ⟨0, hDim⟩)
  let sign : α := if first < 0 then -1 else 1
  let oriented := Tensor.scale eigenvector sign

  {
    components := Tensor.dim (fun _ => oriented),
    mean := mean,
    explainedVariance := Tensor.dim (fun _ => Tensor.scalar eigenvalue)
  }

/-- Apply a fitted PCA transform to a batch of samples. -/
def PCA.transform {nSamples inDim outDim : Nat}
  (m : PCA α inDim outDim)
  (data : Tensor α [nSamples, inDim]) :
  Tensor α [nSamples, outDim] :=
  Tensor.dim (fun i => forward m (Tensor.unstack data i))

/-- Reconstruction error: `||x - inverse(transform(x))||_2^2` (sum of squared coordinates).

PyTorch analogy: `torch.sum((x - x_hat) ** 2)`.
-/
def PCA.reconstructionError {inDim outDim : Nat}
  (m : PCA α inDim outDim)
  (input : Tensor α [inDim]) :
  α :=
  let reduced := forward m input
  let reconstructed := inverse m reduced
  let error := input - reconstructed
  let squaredError := Tensor.square error
  item (castShape (reduce .sum squaredError [⟨0, by simp⟩] false) (Shape.reduce_singleton _ _))

/-- Cumulative explained variance, obtained by prefix-summing `explainedVariance`. -/
def PCA.cumulativeExplainedVariance {α : Type} [Add α] [Zero α]
    [Storage α]
    {inDim outDim : Nat} (m : PCA α inDim outDim) :
    Tensor α [outDim] :=
  (Sequence.mapAccum outDim (0 : α) fun i total =>
    let next := total + Tensor.getScalar m.explainedVariance i
    (next, next)).2


end TorchLean
