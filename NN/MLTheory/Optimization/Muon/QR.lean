/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.Optimization.Muon.Certificates
public import NN.MLTheory.Optimization.Muon.NewtonSchulz
public import NN.Proofs.Tensor.Basic.FactorizationsOrthonormal
public import NN.Proofs.Tensor.Basic.LinearAlgebra

/-!
# QR Muon Backend and Real-Valued Certificates

The real-valued QR orthogonalizer, its positive-pivot condition, and the exact certificates it
supplies to Muon updates.

The entrywise matrix API (`matrix_ext`, `get2_mat_mul_spec`) is stated over `ℝ`, so this module
also holds the real-valued facts about `columnGram` and about one Newton-Schulz step at an exactly
column-orthogonal matrix, which the generic `NN.MLTheory.Optimization.Muon.NewtonSchulz` module
cannot prove.
-/

@[expose] public section

namespace Optim

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Muon

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- QR/Gram-Schmidt orthogonalizer: return the `Q` factor of the fresh matrix buffer. -/
noncomputable def qrOrthogonalizer {m n : Nat} :
    Orthogonalizer ℝ (.dim m (.dim n .scalar)) :=
  { apply := fun buffer => qrQSpec buffer }

/-- Positive pivots of the real-valued QR specification; no native QR claim is made. -/
def HasPositiveQRPivots {m n : Nat} (buffer : MatrixTensor ℝ m n) : Prop :=
  ∀ j : Fin n, 0 < get2 (qrRSpec buffer) j j

/--
Three scaled copies of the same real matrix combine into one scaled copy using the sum of the
coefficients.
-/
theorem add_scaled_three_eq_scale_sum {m n : Nat}
    (Q : MatrixTensor ℝ m n) (a b c : ℝ) :
    addSpec (addSpec (scaleSpec Q a) (scaleSpec Q b)) (scaleSpec Q c) =
      scaleSpec Q (a + b + c) := by
  apply matrix_ext
  intro i j
  simp only [get2_addSpec, get2_scaleSpec]
  ring

/--
If three scaled copies of a matrix are added and the coefficients sum to one, the result is the
original matrix.
-/
theorem add_scaled_three_eq_self_of_coeff_sum_one {m n : Nat}
    (Q : MatrixTensor ℝ m n) (a b c : ℝ) (hsum : a + b + c = 1) :
    addSpec (addSpec (scaleSpec Q a) (scaleSpec Q b)) (scaleSpec Q c) = Q := by
  rw [add_scaled_three_eq_scale_sum, hsum, scaleSpec_one]

/-! ## Column Gram entries -/

/-- Entries of the column Gram matrix are inner products of columns. -/
theorem get2_columnGram {m n : Nat} (Q : MatrixTensor ℝ m n) (i j : Fin n) :
    get2 (columnGram Q) i j = ∑ r : Fin m, get2 Q r i * get2 Q r j := by
  unfold columnGram
  rw [get2_mat_mul_spec]
  exact Finset.sum_congr rfl fun r _ => by rw [get2_matrix_transpose_spec]

/-- Scaling a matrix by `k` scales every column Gram entry by `k * k`. -/
theorem get2_columnGram_scaleSpec {m n : Nat} (Q : MatrixTensor ℝ m n) (k : ℝ) (i j : Fin n) :
    get2 (columnGram (scaleSpec Q k)) i j = get2 (columnGram Q) i j * (k * k) := by
  rw [get2_columnGram, get2_columnGram, Finset.sum_mul]
  refine Finset.sum_congr rfl fun r _ => ?_
  rw [get2_scaleSpec, get2_scaleSpec]
  ring

/-! ## Scaling exactly column-orthogonal matrices -/

/--
Scaling an exact-column-orthogonal real matrix by a scalar whose square is one preserves exact
column Gram.
-/
theorem scale_hasExactColumnGram_of_square_eq_one {m n : Nat}
    (Q : MatrixTensor ℝ m n) (k : ℝ)
    (hgram : HasExactColumnGram Q) (hk : k * k = 1) :
    HasExactColumnGram (scaleSpec Q k) := by
  have hgram' : columnGram Q = identityTensorSpec (α := ℝ) n := hgram
  unfold HasExactColumnGram
  apply matrix_ext
  intro i j
  rw [get2_columnGram_scaleSpec, hgram', hk, mul_one]

/--
Scaling an exact-column-orthogonal real matrix gives an approximate Gram certificate whenever
$|k^2-1|$ is bounded by the requested tolerance.
-/
theorem scale_hasApproxColumnGram_of_exact_column_gram_of_square_error {m n : Nat}
    (Q : MatrixTensor ℝ m n) (k eps : ℝ)
    (hgram : HasExactColumnGram Q)
    (herr : MathFunctions.abs (k * k - 1) ≤ eps)
    (heps : 0 ≤ eps) :
    HasApproxColumnGram eps (scaleSpec Q k) := by
  intro i j
  have hgram' : columnGram Q = identityTensorSpec (α := ℝ) n := hgram
  -- The residual of the scaled matrix is `(k * k - 1)` times the identity entry.
  have hresidual :
      get2 (columnGramResidual (scaleSpec Q k)) i j =
        get2 (identityTensorSpec (α := ℝ) n) i j * (k * k - 1) := by
    unfold columnGramResidual
    rw [get2_subSpec, get2_columnGram_scaleSpec, hgram']
    ring
  rw [hresidual, get2_identityTensorSpec]
  by_cases hij : i = j
  · simpa [hij] using herr
  · simpa [hij, MathFunctions.abs] using heps

/-! ## Newton-Schulz at an exactly column-orthogonal matrix -/

/--
If $Q^\mathsf{T}Q=I$, then one column-oriented Newton-Schulz step returns $(a+b+c)Q$.
-/
theorem newtonSchulzStep_eq_scale_sum_of_exact_column_gram {m n : Nat}
    (coeffs : NewtonSchulzCoeffs ℝ) (Q : MatrixTensor ℝ m n)
    (hgram : HasExactColumnGram Q) :
    newtonSchulzStep coeffs Q = scaleSpec Q (coeffs.a + coeffs.b + coeffs.c) := by
  have hgram' : columnGram Q = identityTensorSpec (α := ℝ) n := hgram
  have hXG : matMulSpec Q (columnGram Q) = Q := by
    rw [hgram']
    exact matMulSpec_identityTensorSpec Q
  have hXG2 : matMulSpec (matMulSpec Q (columnGram Q)) (columnGram Q) = Q := by
    rw [hXG]
    exact hXG
  unfold newtonSchulzStep
  simpa [hXG, hXG2] using add_scaled_three_eq_scale_sum Q coeffs.a coeffs.b coeffs.c

/--
For real coefficients whose sum is one, an exact-column-orthogonal matrix is a fixed point of one
column-oriented Newton-Schulz step.
-/
theorem newtonSchulzFixedPoint_of_exact_column_gram_of_coeff_sum_one {m n : Nat}
    (coeffs : NewtonSchulzCoeffs ℝ) (Q : MatrixTensor ℝ m n)
    (hgram : HasExactColumnGram Q)
    (hsum : coeffs.a + coeffs.b + coeffs.c = 1) :
    NewtonSchulzFixedPoint coeffs Q := by
  unfold NewtonSchulzFixedPoint
  rw [newtonSchulzStep_eq_scale_sum_of_exact_column_gram coeffs Q hgram, hsum, scaleSpec_one]

/--
If $Q^\mathsf{T}Q=I$ and $(a+b+c)^2=1$, then one column-oriented Newton-Schulz step still has exact
column Gram.
-/
theorem newtonSchulzStep_hasExactColumnGram_of_exact_column_gram_of_sum_square_one {m n : Nat}
    (coeffs : NewtonSchulzCoeffs ℝ) (Q : MatrixTensor ℝ m n)
    (hgram : HasExactColumnGram Q)
    (hsquare : (coeffs.a + coeffs.b + coeffs.c) * (coeffs.a + coeffs.b + coeffs.c) = 1) :
    HasExactColumnGram (newtonSchulzStep coeffs Q) := by
  rw [newtonSchulzStep_eq_scale_sum_of_exact_column_gram coeffs Q hgram]
  exact scale_hasExactColumnGram_of_square_eq_one Q (coeffs.a + coeffs.b + coeffs.c) hgram hsquare

/--
If $Q^\mathsf{T}Q=I$ and $|(a+b+c)^2-1|\leq\varepsilon$, then one column-oriented Newton-Schulz
step has entrywise Gram residual bounded by $\varepsilon$.
-/
theorem newtonSchulzStep_hasApproxColumnGram_of_exact_column_gram_of_sum_square_error {m n : Nat}
    (coeffs : NewtonSchulzCoeffs ℝ) (Q : MatrixTensor ℝ m n) (eps : ℝ)
    (hgram : HasExactColumnGram Q)
    (herr :
      MathFunctions.abs
        ((coeffs.a + coeffs.b + coeffs.c) * (coeffs.a + coeffs.b + coeffs.c) - 1) ≤ eps)
    (heps : 0 ≤ eps) :
    HasApproxColumnGram eps (newtonSchulzStep coeffs Q) := by
  rw [newtonSchulzStep_eq_scale_sum_of_exact_column_gram coeffs Q hgram]
  exact scale_hasApproxColumnGram_of_exact_column_gram_of_square_error
    Q (coeffs.a + coeffs.b + coeffs.c) eps hgram herr heps

/--
If the Newton-Schulz coefficients sum to one, exact column Gram is enough to satisfy the exact
fixed-point backend's success predicate.
-/
theorem newtonSchulzFixedPointCheckedExact_success_of_coeff_sum_one {m n : Nat}
    (coeffs : NewtonSchulzCoeffs ℝ) (steps : Nat) (buffer : MatrixTensor ℝ m n)
    (hgram : HasExactColumnGram buffer)
    (hsum : coeffs.a + coeffs.b + coeffs.c = 1) :
    (newtonSchulzFixedPointCheckedExactOrthogonalizer
      (α := ℝ) (m := m) (n := n) coeffs steps).Success buffer :=
  ⟨hgram, newtonSchulzFixedPoint_of_exact_column_gram_of_coeff_sum_one coeffs buffer hgram hsum⟩

/--
For real coefficients with $a+b+c=1$, exact column Gram of the fresh momentum buffer is
enough to certify a Newton-Schulz Muon update exactly.
-/
theorem update_has_exact_certified_step_newtonSchulz_exact_gram_checked {m n : Nat}
    (coeffs : NewtonSchulzCoeffs ℝ) (steps : Nat)
    (learningRate momentum : ℝ) (momentumBuffer parameters gradients : MatrixTensor ℝ m n)
    (hgram :
      HasExactColumnGram
        (update
          ({ learningRate := learningRate, momentum := momentum, momentumBuffer := momentumBuffer,
             orthogonalizer :=
              newtonSchulzOrthogonalizer (α := ℝ) (m := m) (n := n) coeffs steps } :
            State ℝ (.dim m (.dim n .scalar)))
          parameters gradients).optimizerState.momentumBuffer)
    (hsum : coeffs.a + coeffs.b + coeffs.c = 1) :
    ∃ direction : MatrixTensor ℝ m n,
      ExactCertifiedStep
        ({ learningRate := learningRate, momentum := momentum, momentumBuffer := momentumBuffer,
           orthogonalizer :=
            newtonSchulzOrthogonalizer (α := ℝ) (m := m) (n := n) coeffs steps } :
          State ℝ (.dim m (.dim n .scalar)))
        parameters gradients direction := by
  exact exactCertifiedStep_of_checkedBackend
    (backend := newtonSchulzFixedPointCheckedExactOrthogonalizer
      (α := ℝ) (m := m) (n := n) coeffs steps)
    (learningRate := learningRate) (momentum := momentum) (momentumBuffer := momentumBuffer)
    (parameters := parameters) (gradients := gradients)
    (newtonSchulzFixedPointCheckedExact_success_of_coeff_sum_one
      coeffs steps
      (update
        ({ learningRate := learningRate, momentum := momentum, momentumBuffer := momentumBuffer,
           orthogonalizer :=
            newtonSchulzOrthogonalizer (α := ℝ) (m := m) (n := n) coeffs steps } :
          State ℝ (.dim m (.dim n .scalar)))
        parameters gradients).optimizerState.momentumBuffer hgram hsum)

/--
For real coefficients with $a+b+c=1$, exact column Gram of the fresh momentum buffer gives
$Q^\mathsf{T}Q=I$ for the actual Newton-Schulz update direction.
-/
theorem update_newtonSchulz_exact_gram_direction_has_exact_column_gram_checked {m n : Nat}
    (coeffs : NewtonSchulzCoeffs ℝ) (steps : Nat)
    (learningRate momentum : ℝ) (momentumBuffer parameters gradients : MatrixTensor ℝ m n)
    (hgram :
      HasExactColumnGram
        (update
          ({ learningRate := learningRate, momentum := momentum, momentumBuffer := momentumBuffer,
             orthogonalizer :=
              newtonSchulzOrthogonalizer (α := ℝ) (m := m) (n := n) coeffs steps } :
            State ℝ (.dim m (.dim n .scalar)))
          parameters gradients).optimizerState.momentumBuffer)
    (hsum : coeffs.a + coeffs.b + coeffs.c = 1) :
    HasExactColumnGram
      ((newtonSchulzOrthogonalizer (α := ℝ) (m := m) (n := n) coeffs steps).apply
        (update
          ({ learningRate := learningRate, momentum := momentum, momentumBuffer := momentumBuffer,
             orthogonalizer :=
              newtonSchulzOrthogonalizer (α := ℝ) (m := m) (n := n) coeffs steps } :
            State ℝ (.dim m (.dim n .scalar)))
          parameters gradients).optimizerState.momentumBuffer) := by
  exact checkedBackend_updateDirection_hasExactColumnGram
    (backend := newtonSchulzFixedPointCheckedExactOrthogonalizer
      (α := ℝ) (m := m) (n := n) coeffs steps)
    (learningRate := learningRate) (momentum := momentum) (momentumBuffer := momentumBuffer)
    (parameters := parameters) (gradients := gradients)
    (newtonSchulzFixedPointCheckedExact_success_of_coeff_sum_one
      coeffs steps
      (update
        ({ learningRate := learningRate, momentum := momentum, momentumBuffer := momentumBuffer,
           orthogonalizer :=
            newtonSchulzOrthogonalizer (α := ℝ) (m := m) (n := n) coeffs steps } :
          State ℝ (.dim m (.dim n .scalar)))
        parameters gradients).optimizerState.momentumBuffer hgram hsum)

/-! ## QR certificates -/

/--
The QR orthogonalizer satisfies the exact Muon direction contract whenever the QR pivots of the
input buffer are positive.
-/
theorem qrOrthogonalizer_exact_of_positive_pivots {m n : Nat}
    (buffer : MatrixTensor ℝ m n)
    (hpivots : HasPositiveQRPivots buffer) :
    ExactOrthogonalizesBuffer (qrOrthogonalizer (m := m) (n := n)) buffer := by
  change columnGram (qrQSpec buffer) = identityTensorSpec (α := ℝ) n
  apply matrix_ext
  intro i j
  rw [get2_columnGram, get2_identityTensorSpec]
  exact Spec.Factorization.Reconstruction.qrSpec_orthonormal buffer hpivots i j

/-- QR packaged as a checked exact Muon backend. -/
noncomputable def qrCheckedExactOrthogonalizer {m n : Nat} :
    CheckedExactOrthogonalizer ℝ m n :=
  { orthogonalizer := qrOrthogonalizer (m := m) (n := n)
    Success := HasPositiveQRPivots
    certified := fun buffer hpivots =>
      qrOrthogonalizer_exact_of_positive_pivots buffer hpivots }

/--
Concrete QR-backed Muon step theorem: if the fresh momentum buffer has positive QR pivots, the
executable Muon update has a certified exact step.
-/
theorem update_has_exact_certified_step_qr {m n : Nat}
    (learningRate momentum : ℝ) (momentumBuffer parameters gradients : MatrixTensor ℝ m n)
    (hpivots :
      HasPositiveQRPivots
        (update
          ({ learningRate := learningRate, momentum := momentum, momentumBuffer := momentumBuffer,
             orthogonalizer := qrOrthogonalizer (m := m) (n := n) } :
            State ℝ (.dim m (.dim n .scalar)))
          parameters gradients).optimizerState.momentumBuffer) :
    ∃ direction : MatrixTensor ℝ m n,
      ExactCertifiedStep
        ({ learningRate := learningRate, momentum := momentum, momentumBuffer := momentumBuffer,
           orthogonalizer := qrOrthogonalizer (m := m) (n := n) } :
          State ℝ (.dim m (.dim n .scalar)))
        parameters gradients direction := by
  exact exactCertifiedStep_of_checkedBackend
    (backend := qrCheckedExactOrthogonalizer (m := m) (n := n))
    (learningRate := learningRate) (momentum := momentum) (momentumBuffer := momentumBuffer)
    (parameters := parameters) (gradients := gradients)
    hpivots

/--
Concrete QR-backed direction theorem: if the fresh momentum buffer has positive QR pivots, the
actual direction used by the Muon update has column Gram $I$.
-/
theorem update_qr_direction_has_exact_column_gram {m n : Nat}
    (learningRate momentum : ℝ) (momentumBuffer parameters gradients : MatrixTensor ℝ m n)
    (hpivots :
      HasPositiveQRPivots
        (update
          ({ learningRate := learningRate, momentum := momentum, momentumBuffer := momentumBuffer,
             orthogonalizer := qrOrthogonalizer (m := m) (n := n) } :
            State ℝ (.dim m (.dim n .scalar)))
          parameters gradients).optimizerState.momentumBuffer) :
    HasExactColumnGram
      ((qrOrthogonalizer (m := m) (n := n)).apply
        (update
          ({ learningRate := learningRate, momentum := momentum, momentumBuffer := momentumBuffer,
             orthogonalizer := qrOrthogonalizer (m := m) (n := n) } :
            State ℝ (.dim m (.dim n .scalar)))
          parameters gradients).optimizerState.momentumBuffer) := by
  exact checkedBackend_updateDirection_hasExactColumnGram
    (backend := qrCheckedExactOrthogonalizer (m := m) (n := n))
    (learningRate := learningRate) (momentum := momentum) (momentumBuffer := momentumBuffer)
    (parameters := parameters) (gradients := gradients)
    hpivots

end Muon

end Optim
