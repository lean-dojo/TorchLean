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
# QR Muon Backend

The real-valued QR orthogonalizer, its positive-pivot condition, and the exact certificates it
supplies to Muon updates.
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

/-- Entries of the identity matrix over `ℝ`, in the `get2` form the QR proofs consume. -/
theorem get2_identityTensorSpec_real {n : Nat} (i j : Fin n) :
    get2 (identityTensorSpec (α := ℝ) n) i j = if i = j then 1 else 0 := by
  by_cases h : i = j
  · subst j
    simp [identityTensorSpec, get2, Tensor.getScalar, Spec.get, Tensor.unstack,
      Tensor.item]
  · have hval : i.val ≠ j.val := fun hval => h (Fin.ext hval)
    simp [identityTensorSpec, get2, Tensor.getScalar, Spec.get, Tensor.unstack,
      Tensor.item, h, hval]

/-- Entry rule for matrix-shaped tensor addition over $\mathbb{R}$. -/
theorem get2_addSpec_real {m n : Nat} (A B : MatrixTensor ℝ m n) (i : Fin m) (j : Fin n) :
    get2 (addSpec A B) i j = get2 A i j + get2 B i j := by
  simp [addSpec]

/-- Entry rule for matrix-shaped tensor scaling over $\mathbb{R}$. -/
theorem get2_scaleSpec_real {m n : Nat} (A : MatrixTensor ℝ m n) (c : ℝ)
    (i : Fin m) (j : Fin n) :
    get2 (scaleSpec A c) i j = get2 A i j * c := by
  simp [scaleSpec]

/-- Entry rule for matrix-shaped tensor subtraction over $\mathbb{R}$. -/
theorem get2_subSpec_real {m n : Nat} (A B : MatrixTensor ℝ m n) (i : Fin m) (j : Fin n) :
    get2 (subSpec A B) i j = get2 A i j - get2 B i j := by
  simp [subSpec]

/-- Right multiplication by the identity matrix leaves a real matrix unchanged. -/
theorem matMul_right_identity_real {m n : Nat} (A : MatrixTensor ℝ m n) :
    matMulSpec A (identityTensorSpec (α := ℝ) n) = A := by
  classical
  apply matrix_ext
  intro i j
  calc
    get2 (matMulSpec A (identityTensorSpec (α := ℝ) n)) i j
        = ∑ k : Fin n, get2 A i k * get2 (identityTensorSpec (α := ℝ) n) k j := by
          simpa using
            (get2_mat_mul_spec (A := A) (B := identityTensorSpec (α := ℝ) n) (i := i) (j := j))
    _ = ∑ k : Fin n, get2 A i k * (if k = j then 1 else 0) := by
          refine Finset.sum_congr rfl ?_
          intro k _
          rw [get2_identityTensorSpec_real]
    _ = get2 A i j := by
          simp

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
  calc
    get2 (addSpec (addSpec (scaleSpec Q a) (scaleSpec Q b)) (scaleSpec Q c)) i j
        = (get2 Q i j * a + get2 Q i j * b) + get2 Q i j * c := by
          rw [get2_addSpec_real, get2_addSpec_real, get2_scaleSpec_real,
            get2_scaleSpec_real, get2_scaleSpec_real]
    _ = get2 Q i j * (a + b + c) := by
          ring
    _ = get2 (scaleSpec Q (a + b + c)) i j := by
          rw [get2_scaleSpec_real]

/--
If three scaled copies of a matrix are added and the coefficients sum to one, the result is the
original matrix.
-/
theorem add_scaled_three_eq_self_of_coeff_sum_one {m n : Nat}
    (Q : MatrixTensor ℝ m n) (a b c : ℝ) (hsum : a + b + c = 1) :
    addSpec (addSpec (scaleSpec Q a) (scaleSpec Q b)) (scaleSpec Q c) = Q := by
  rw [add_scaled_three_eq_scale_sum]
  apply matrix_ext
  intro i j
  calc
    get2 (scaleSpec Q (a + b + c)) i j = get2 Q i j * (a + b + c) := by
      rw [get2_scaleSpec_real]
    _ = get2 Q i j * 1 := by
          rw [hsum]
    _ = get2 Q i j := by
      ring

/--
Scaling an exact-column-orthogonal real matrix by a scalar whose square is one preserves exact
column Gram.
-/
theorem scale_hasExactColumnGram_of_square_eq_one {m n : Nat}
    (Q : MatrixTensor ℝ m n) (k : ℝ)
    (hgram : HasExactColumnGram Q) (hk : k * k = 1) :
    HasExactColumnGram (scaleSpec Q k) := by
  unfold HasExactColumnGram columnGram
  apply matrix_ext
  intro i j
  have hgram' : matMulSpec (TorchLean.Tensor.swapAdjacentAxes Q 0) Q =
      identityTensorSpec (α := ℝ) n := by
    simpa [HasExactColumnGram, columnGram] using hgram
  have hentry :
      (∑ r : Fin m, get2 Q r i * get2 Q r j) =
        get2 (identityTensorSpec (α := ℝ) n) i j := by
    calc
      (∑ r : Fin m, get2 Q r i * get2 Q r j)
          = get2 (matMulSpec (TorchLean.Tensor.swapAdjacentAxes Q 0) Q) i j := by
            symm
            calc
              get2 (matMulSpec (TorchLean.Tensor.swapAdjacentAxes Q 0) Q) i j
                  = ∑ r : Fin m,
                      get2 (TorchLean.Tensor.swapAdjacentAxes Q 0) i r * get2 Q r j := by
                    simpa using
                      (get2_mat_mul_spec
                        (A := TorchLean.Tensor.swapAdjacentAxes Q 0) (B := Q) (i := i) (j := j))
              _ = ∑ r : Fin m, get2 Q r i * get2 Q r j := by
                    refine Finset.sum_congr rfl ?_
                    intro r _
                    rw [get2_matrix_transpose_spec]
      _ = get2 (identityTensorSpec (α := ℝ) n) i j := by
            exact congrArg (fun M => get2 M i j) hgram'
  calc
    get2
        (matMulSpec (TorchLean.Tensor.swapAdjacentAxes (scaleSpec Q k) 0) (scaleSpec Q k)) i j
        = ∑ r : Fin m,
            get2 (TorchLean.Tensor.swapAdjacentAxes (scaleSpec Q k) 0) i r *
              get2 (scaleSpec Q k) r j := by
          simpa using
            (get2_mat_mul_spec
              (A := TorchLean.Tensor.swapAdjacentAxes (scaleSpec Q k) 0) (B := scaleSpec Q k)
              (i := i) (j := j))
    _ = ∑ r : Fin m, (get2 Q r i * k) * (get2 Q r j * k) := by
          refine Finset.sum_congr rfl ?_
          intro r _
          rw [get2_matrix_transpose_spec, get2_scaleSpec_real, get2_scaleSpec_real]
    _ = ∑ r : Fin m, (get2 Q r i * get2 Q r j) * (k * k) := by
          refine Finset.sum_congr rfl ?_
          intro r _
          ring
    _ = (∑ r : Fin m, get2 Q r i * get2 Q r j) * (k * k) := by
          rw [Finset.sum_mul]
    _ = (∑ r : Fin m, get2 Q r i * get2 Q r j) * 1 := by
          rw [hk]
    _ = get2 (identityTensorSpec (α := ℝ) n) i j := by
          rw [mul_one, hentry]

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
  have hgram' : matMulSpec (TorchLean.Tensor.swapAdjacentAxes Q 0) Q =
      identityTensorSpec (α := ℝ) n := by
    simpa [HasExactColumnGram, columnGram] using hgram
  have hentry :
      (∑ r : Fin m, get2 Q r i * get2 Q r j) =
        get2 (identityTensorSpec (α := ℝ) n) i j := by
    calc
      (∑ r : Fin m, get2 Q r i * get2 Q r j)
          = get2 (matMulSpec (TorchLean.Tensor.swapAdjacentAxes Q 0) Q) i j := by
            symm
            calc
              get2 (matMulSpec (TorchLean.Tensor.swapAdjacentAxes Q 0) Q) i j
                  = ∑ r : Fin m,
                      get2 (TorchLean.Tensor.swapAdjacentAxes Q 0) i r * get2 Q r j := by
                    simpa using
                      (get2_mat_mul_spec
                        (A := TorchLean.Tensor.swapAdjacentAxes Q 0) (B := Q) (i := i) (j := j))
              _ = ∑ r : Fin m, get2 Q r i * get2 Q r j := by
                    refine Finset.sum_congr rfl ?_
                    intro r _
                    rw [get2_matrix_transpose_spec]
      _ = get2 (identityTensorSpec (α := ℝ) n) i j := by
            exact congrArg (fun M => get2 M i j) hgram'
  have hscaledGram :
      get2 (columnGram (scaleSpec Q k)) i j =
        get2 (identityTensorSpec (α := ℝ) n) i j * (k * k) := by
    calc
      get2 (columnGram (scaleSpec Q k)) i j
          = get2
              (matMulSpec (TorchLean.Tensor.swapAdjacentAxes (scaleSpec Q k) 0) (scaleSpec Q k))
              i j := by
            rfl
      _ = ∑ r : Fin m,
            get2 (TorchLean.Tensor.swapAdjacentAxes (scaleSpec Q k) 0) i r *
              get2 (scaleSpec Q k) r j := by
            simpa using
              (get2_mat_mul_spec
                (A := TorchLean.Tensor.swapAdjacentAxes (scaleSpec Q k) 0) (B := scaleSpec Q k)
                (i := i) (j := j))
      _ = ∑ r : Fin m, (get2 Q r i * k) * (get2 Q r j * k) := by
            refine Finset.sum_congr rfl ?_
            intro r _
            rw [get2_matrix_transpose_spec, get2_scaleSpec_real, get2_scaleSpec_real]
      _ = ∑ r : Fin m, (get2 Q r i * get2 Q r j) * (k * k) := by
            refine Finset.sum_congr rfl ?_
            intro r _
            ring
      _ = (∑ r : Fin m, get2 Q r i * get2 Q r j) * (k * k) := by
            rw [Finset.sum_mul]
      _ = get2 (identityTensorSpec (α := ℝ) n) i j * (k * k) := by
            rw [hentry]
  by_cases hij : i = j
  · subst j
    have hdiag : get2 (identityTensorSpec (α := ℝ) n) i i = 1 := by
      simp [get2_identityTensorSpec_real]
    calc
      MathFunctions.abs (get2 (columnGramResidual (scaleSpec Q k)) i i)
          = MathFunctions.abs (k * k - 1) := by
            rw [show columnGramResidual (scaleSpec Q k) =
              subSpec (columnGram (scaleSpec Q k)) (identityTensorSpec n) by rfl]
            rw [get2_subSpec_real, hscaledGram, hdiag]
            ring_nf
      _ ≤ eps := herr
  · have hoff : get2 (identityTensorSpec (α := ℝ) n) i j = 0 := by
      rw [get2_identityTensorSpec_real]
      simp [hij]
    calc
      MathFunctions.abs (get2 (columnGramResidual (scaleSpec Q k)) i j)
          = 0 := by
            rw [show columnGramResidual (scaleSpec Q k) =
              subSpec (columnGram (scaleSpec Q k)) (identityTensorSpec n) by rfl]
            rw [get2_subSpec_real, hscaledGram, hoff]
            simp [MathFunctions.abs]
      _ ≤ eps := heps

/--
If $Q^\mathsf{T}Q=I$, then one column-oriented Newton-Schulz step returns $(a+b+c)Q$.
-/
theorem newtonSchulzStep_eq_scale_sum_of_exact_column_gram {m n : Nat}
    (coeffs : NewtonSchulzCoeffs ℝ) (Q : MatrixTensor ℝ m n)
    (hgram : HasExactColumnGram Q) :
    newtonSchulzStep coeffs Q = scaleSpec Q (coeffs.a + coeffs.b + coeffs.c) := by
  unfold newtonSchulzStep
  have hright : rightGram Q = identityTensorSpec (α := ℝ) n := by
    simpa [HasExactColumnGram, columnGram, rightGram] using hgram
  have hXG : matMulSpec Q (rightGram Q) = Q := by
    rw [hright]
    exact matMul_right_identity_real Q
  have hXG2 : matMulSpec (matMulSpec Q (rightGram Q)) (rightGram Q) = Q := by
    rw [hright]
    rw [matMul_right_identity_real Q]
    exact matMul_right_identity_real Q
  simpa [hXG, hXG2] using
    add_scaled_three_eq_scale_sum Q coeffs.a coeffs.b coeffs.c

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
  rw [newtonSchulzStep_eq_scale_sum_of_exact_column_gram coeffs Q hgram]
  rw [hsum]
  apply matrix_ext
  intro i j
  calc
    get2 (scaleSpec Q 1) i j = get2 Q i j * 1 := by
      rw [get2_scaleSpec_real]
    _ = get2 Q i j := by
      ring

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
      (α := ℝ) (m := m) (n := n) coeffs steps).Success buffer := by
  exact ⟨hgram,
    newtonSchulzFixedPoint_of_exact_column_gram_of_coeff_sum_one coeffs buffer hgram hsum⟩

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

/--
Initialized version: exact column Gram of the first fresh momentum buffer and $a+b+c=1$
certify the first Newton-Schulz Muon step exactly.
-/
theorem init_has_exact_certified_step_newtonSchulz_exact_gram_checked {m n : Nat}
    (coeffs : NewtonSchulzCoeffs ℝ) (steps : Nat)
    (learningRate momentum : ℝ) (parameters gradients : MatrixTensor ℝ m n)
    (hgram :
      HasExactColumnGram
        (update
          (init learningRate momentum
            (newtonSchulzOrthogonalizer (α := ℝ) (m := m) (n := n) coeffs steps)
            parameters)
          parameters gradients).optimizerState.momentumBuffer)
    (hsum : coeffs.a + coeffs.b + coeffs.c = 1) :
    ∃ direction : MatrixTensor ℝ m n,
      ExactCertifiedStep
        (init learningRate momentum
          (newtonSchulzOrthogonalizer (α := ℝ) (m := m) (n := n) coeffs steps)
          parameters)
        parameters gradients direction := by
  exact exactCertifiedStep_of_checkedBackend
    (backend := newtonSchulzFixedPointCheckedExactOrthogonalizer
      (α := ℝ) (m := m) (n := n) coeffs steps)
    (learningRate := learningRate) (momentum := momentum)
    (momentumBuffer := Tensor.full (.dim m (.dim n .scalar)) 0) (parameters := parameters)
    (gradients := gradients)
    (newtonSchulzFixedPointCheckedExact_success_of_coeff_sum_one
      coeffs steps
      (update
        (init learningRate momentum
          (newtonSchulzOrthogonalizer (α := ℝ) (m := m) (n := n) coeffs steps)
          parameters)
        parameters gradients).optimizerState.momentumBuffer hgram hsum)

/--
The QR orthogonalizer satisfies the exact Muon direction contract whenever the executable QR pivots
of the input buffer are positive.
-/
theorem qrOrthogonalizer_exact_of_positive_pivots {m n : Nat}
    (buffer : MatrixTensor ℝ m n)
    (hpivots : HasPositiveQRPivots buffer) :
    ExactOrthogonalizesBuffer (qrOrthogonalizer (m := m) (n := n)) buffer := by
  unfold ExactOrthogonalizesBuffer HasExactColumnGram columnGram qrOrthogonalizer
  apply matrix_ext
  intro i j
  calc
    get2 (matMulSpec (TorchLean.Tensor.swapAdjacentAxes (qrQSpec buffer) 0) (qrQSpec buffer)) i j
        = ∑ k : Fin m, get2 (TorchLean.Tensor.swapAdjacentAxes (qrQSpec buffer) 0) i k *
            get2 (qrQSpec buffer) k j := by
          simpa using
            (get2_mat_mul_spec
              (A := TorchLean.Tensor.swapAdjacentAxes (qrQSpec buffer) 0)
              (B := qrQSpec buffer) (i := i) (j := j))
    _ = ∑ k : Fin m, get2 (qrQSpec buffer) k i * get2 (qrQSpec buffer) k j := by
          refine Finset.sum_congr rfl ?_
          intro k _
          rw [get2_matrix_transpose_spec]
    _ = if i = j then 1 else 0 := by
          exact Spec.Factorization.Reconstruction.qrSpec_orthonormal buffer hpivots i j
    _ = get2 (identityTensorSpec (α := ℝ) n) i j := by
          rw [get2_identityTensorSpec_real]

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

/--
Initialized QR-backed Muon step theorem: if the first fresh momentum buffer has positive QR pivots,
the first initialized Muon update has a certified exact step.
-/
theorem init_has_exact_certified_step_qr {m n : Nat}
    (learningRate momentum : ℝ) (parameters gradients : MatrixTensor ℝ m n)
    (hpivots :
      HasPositiveQRPivots
        (update
          (init learningRate momentum (qrOrthogonalizer (m := m) (n := n)) parameters)
          parameters gradients).optimizerState.momentumBuffer) :
    ∃ direction : MatrixTensor ℝ m n,
      ExactCertifiedStep
        (init learningRate momentum (qrOrthogonalizer (m := m) (n := n)) parameters)
        parameters gradients direction := by
  exact exactCertifiedStep_of_checkedBackend
    (backend := qrCheckedExactOrthogonalizer (m := m) (n := n))
    (learningRate := learningRate) (momentum := momentum)
    (momentumBuffer := Tensor.full (.dim m (.dim n .scalar)) 0) (parameters := parameters)
    (gradients := gradients)
    hpivots

/--
Initialized QR-backed direction theorem: if the first fresh momentum buffer has positive QR pivots,
the first initialized Muon update direction has column Gram $I$.
-/
theorem init_qr_direction_has_exact_column_gram {m n : Nat}
    (learningRate momentum : ℝ) (parameters gradients : MatrixTensor ℝ m n)
    (hpivots :
      HasPositiveQRPivots
        (update
          (init learningRate momentum (qrOrthogonalizer (m := m) (n := n)) parameters)
          parameters gradients).optimizerState.momentumBuffer) :
    HasExactColumnGram
      ((qrOrthogonalizer (m := m) (n := n)).apply
        (update
          (init learningRate momentum (qrOrthogonalizer (m := m) (n := n)) parameters)
          parameters gradients).optimizerState.momentumBuffer) := by
  exact checkedBackend_updateDirection_hasExactColumnGram
    (backend := qrCheckedExactOrthogonalizer (m := m) (n := n))
    (learningRate := learningRate) (momentum := momentum)
    (momentumBuffer := Tensor.full (.dim m (.dim n .scalar)) 0) (parameters := parameters)
    (gradients := gradients)
    hpivots

end Muon

end Optim
