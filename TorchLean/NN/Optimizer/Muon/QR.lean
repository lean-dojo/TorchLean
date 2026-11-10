/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Optimizer.Muon.Certificates
public import TorchLean.NN.Optimizer.Muon.NewtonSchulz
public import TorchLean.Tensor.Factorization.Orthonormal
public import TorchLean.Tensor.LinearAlgebra.Real

/-!
# QR Muon Backend and Real-Valued Certificates

The QR backend returns the `Q` factor of classical Gram-Schmidt over `ℝ`, and positive pivots of the
`R` factor make it an exact Muon backend.

The entrywise matrix API (`ext_get2`, `get2_matmul_eq_sum`) is stated over `ℝ`, so this module
also holds the real-valued facts about `columnGram` and about one Newton-Schulz step at an exactly
column-orthogonal matrix, which the generic `TorchLean.NN.Optimizer.Muon.NewtonSchulz`
module cannot prove.

The backend and its proofs are ours, written by hand from the definitions. Gram-Schmidt meets the
exact contract with a different matrix from the one Newton-Schulz aims at. The buffer with rows
$(1,1)$ and $(0,1)$ is its own $R$, with pivots 1 and 1, so its $Q$ is the identity, while its
polar factor $UV^\mathsf{T}$ is the clockwise rotation by about 26.6 degrees, with rows
$(2/\sqrt5,1/\sqrt5)$ and $(-1/\sqrt5,2/\sqrt5)$. Both are orthogonal, so a backend returning
either one meets `ExactOrthogonalizesBuffer` on that buffer.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean

open Tensor Factorization

namespace Muon

variable {α : Type} [Storage α] [Context α]

/--
QR orthogonalizer: the `Q` factor `(Factorization.qr buffer).1` of classical Gram-Schmidt on the
buffer's columns. It is defined over `ℝ`, so it is noncomputable and serves as a backend for proofs.
-/
noncomputable def QR.orthogonalizer {m n : Nat} :
    Orthogonalizer ℝ (.dim m (.dim n .scalar)) :=
  { apply := fun buffer => (qr buffer).1 }

/--
Positive pivots: every diagonal entry of `(Factorization.qr buffer).2`, the `R` factor of the
real-valued Gram-Schmidt specification, is positive. A native QR kernel is outside the statement.
-/
def QR.HasPositivePivots {m n : Nat} (buffer : MatrixTensor ℝ m n) : Prop :=
  ∀ j : Fin n, 0 < get2 (qr buffer).2 j j

/-- For a real matrix $Q$, $(aQ+bQ)+cQ=(a+b+c)Q$. -/
theorem add_scaled_three_eq_scale_sum {m n : Nat}
    (Q : MatrixTensor ℝ m n) (a b c : ℝ) :
    (Tensor.scale Q a + Tensor.scale Q b) + Tensor.scale Q c =
      Tensor.scale Q (a + b + c) := by
  apply ext_get2
  intro i j
  simp only [get2_add, get2_scale]
  ring

/-! ## Column Gram entries -/

/-- Entries of the column Gram matrix are inner products of columns. -/
theorem get2_columnGram {m n : Nat} (Q : MatrixTensor ℝ m n) (i j : Fin n) :
    get2 (columnGram Q) i j = ∑ r : Fin m, get2 Q r i * get2 Q r j := by
  unfold columnGram
  rw [get2_matmul_eq_sum]
  exact Finset.sum_congr rfl fun r _ => by rw [get2_transpose]

/-- Scaling a matrix by `k` scales every column Gram entry by `k * k`. -/
theorem get2_columnGram_scale {m n : Nat} (Q : MatrixTensor ℝ m n) (k : ℝ) (i j : Fin n) :
    get2 (columnGram (Tensor.scale Q k)) i j = get2 (columnGram Q) i j * (k * k) := by
  rw [get2_columnGram, get2_columnGram, Finset.sum_mul]
  refine Finset.sum_congr rfl fun r _ => ?_
  rw [get2_scale, get2_scale]
  ring

/-! ## Scaling exactly column-orthogonal matrices -/

/-- If $Q^\mathsf{T}Q=I$ and $k^2=1$, then $kQ$ has exact column Gram. -/
theorem HasExactColumnGram.scale {m n : Nat}
    (Q : MatrixTensor ℝ m n) (k : ℝ)
    (hgram : HasExactColumnGram Q) (hk : k * k = 1) :
    HasExactColumnGram (Tensor.scale Q k) := by
  have hgram' : columnGram Q = Tensor.identity (α := ℝ) n := hgram
  unfold HasExactColumnGram
  apply ext_get2
  intro i j
  rw [get2_columnGram_scale, hgram', hk, mul_one]

/--
If $Q^\mathsf{T}Q=I$, $|k^2-1|\leq\varepsilon$ and $0\leq\varepsilon$, then every entry of the Gram
residual of $kQ$ is at most $\varepsilon$ in absolute value.
-/
theorem HasExactColumnGram.approx_scale {m n : Nat}
    (Q : MatrixTensor ℝ m n) (k eps : ℝ)
    (hgram : HasExactColumnGram Q)
    (herr : MathFunctions.abs (k * k - 1) ≤ eps)
    (heps : 0 ≤ eps) :
    HasApproxColumnGram eps (Tensor.scale Q k) := by
  intro i j
  have hgram' : columnGram Q = Tensor.identity (α := ℝ) n := hgram
  -- The residual of the scaled matrix is `(k * k - 1)` times the identity entry.
  have hresidual :
      get2 (columnGramResidual (Tensor.scale Q k)) i j =
        get2 (Tensor.identity (α := ℝ) n) i j * (k * k - 1) := by
    unfold columnGramResidual
    rw [get2_sub, get2_columnGram_scale, hgram']
    ring
  rw [hresidual, get2_identity]
  by_cases hij : i = j
  · simpa [hij] using herr
  · simpa [hij, MathFunctions.abs] using heps

/-! ## Newton-Schulz at an exactly column-orthogonal matrix -/

/--
If $Q^\mathsf{T}Q=I$, then one column-oriented Newton-Schulz step returns $(a+b+c)Q$.
-/
theorem NewtonSchulz.step_eq_scale {m n : Nat}
    (coeffs : Coeffs ℝ) (Q : MatrixTensor ℝ m n)
    (hgram : HasExactColumnGram Q) :
    step coeffs Q = Tensor.scale Q (coeffs.a + coeffs.b + coeffs.c) := by
  have hgram' : columnGram Q = Tensor.identity (α := ℝ) n := hgram
  have hXG : Tensor.matmul Q (columnGram Q) = Q := by
    rw [hgram']
    exact matmul_identity Q
  have hXG2 : Tensor.matmul (Tensor.matmul Q (columnGram Q)) (columnGram Q) = Q := by
    rw [hXG]
    exact hXG
  unfold step
  simpa [hXG, hXG2] using add_scaled_three_eq_scale_sum Q coeffs.a coeffs.b coeffs.c

/--
If $Q^\mathsf{T}Q=I$ and $a+b+c=1$, then $Q$ is a fixed point of one column-oriented Newton-Schulz
step.
-/
theorem NewtonSchulz.fixedPoint_of_sum_eq_one {m n : Nat}
    (coeffs : Coeffs ℝ) (Q : MatrixTensor ℝ m n)
    (hgram : HasExactColumnGram Q)
    (hsum : coeffs.a + coeffs.b + coeffs.c = 1) :
    FixedPoint coeffs Q := by
  unfold FixedPoint
  rw [step_eq_scale coeffs Q hgram, hsum, scale_one]

/--
If $Q^\mathsf{T}Q=I$ and $(a+b+c)^2=1$, then one column-oriented Newton-Schulz step preserves exact
column Gram.
-/
theorem NewtonSchulz.step_hasExactColumnGram {m n : Nat}
    (coeffs : Coeffs ℝ) (Q : MatrixTensor ℝ m n)
    (hgram : HasExactColumnGram Q)
    (hsquare : (coeffs.a + coeffs.b + coeffs.c) * (coeffs.a + coeffs.b + coeffs.c) = 1) :
    HasExactColumnGram (step coeffs Q) := by
  rw [step_eq_scale coeffs Q hgram]
  exact HasExactColumnGram.scale Q (coeffs.a + coeffs.b + coeffs.c) hgram hsquare

/--
If $Q^\mathsf{T}Q=I$, $|(a+b+c)^2-1|\leq\varepsilon$ and $0\leq\varepsilon$, then one
column-oriented Newton-Schulz step has entrywise Gram residual bounded by $\varepsilon$.
-/
theorem NewtonSchulz.step_hasApproxColumnGram {m n : Nat}
    (coeffs : Coeffs ℝ) (Q : MatrixTensor ℝ m n) (eps : ℝ)
    (hgram : HasExactColumnGram Q)
    (herr :
      MathFunctions.abs
        ((coeffs.a + coeffs.b + coeffs.c) * (coeffs.a + coeffs.b + coeffs.c) - 1) ≤ eps)
    (heps : 0 ≤ eps) :
    HasApproxColumnGram eps (step coeffs Q) := by
  rw [step_eq_scale coeffs Q hgram]
  exact HasExactColumnGram.approx_scale
    Q (coeffs.a + coeffs.b + coeffs.c) eps hgram herr heps

/--
For real coefficients with $a+b+c=1$, exact column Gram alone satisfies the success predicate of
`Muon.NewtonSchulz.fixedPointChecked`.
-/
theorem NewtonSchulz.fixedPointChecked_success {m n : Nat}
    (coeffs : Coeffs ℝ) (steps : Nat) (buffer : MatrixTensor ℝ m n)
    (hgram : HasExactColumnGram buffer)
    (hsum : coeffs.a + coeffs.b + coeffs.c = 1) :
    (fixedPointChecked
      (α := ℝ) (m := m) (n := n) coeffs steps).Success buffer :=
  ⟨hgram, fixedPoint_of_sum_eq_one coeffs buffer hgram hsum⟩

/-! ## QR certificates -/

/--
The QR orthogonalizer satisfies the exact Muon direction contract whenever the QR pivots of the
input buffer are positive.
-/
theorem QR.exactOrthogonalizesBuffer {m n : Nat}
    (buffer : MatrixTensor ℝ m n)
    (hpivots : HasPositivePivots buffer) :
    ExactOrthogonalizesBuffer (orthogonalizer (m := m) (n := n)) buffer := by
  change columnGram (qr buffer).1 = Tensor.identity (α := ℝ) n
  apply ext_get2
  intro i j
  rw [get2_columnGram, get2_identity]
  exact Factorization.qrQ_orthonormal buffer hpivots i j

/-- QR packaged as a checked exact Muon backend, with positive pivots as the success predicate. -/
noncomputable def QR.checked {m n : Nat} :
    CheckedExactOrthogonalizer ℝ m n :=
  { orthogonalizer := orthogonalizer (m := m) (n := n)
    Success := HasPositivePivots
    certified := fun buffer hpivots =>
      exactOrthogonalizesBuffer buffer hpivots }

/--
If the fresh momentum buffer has positive QR pivots, the Muon update with the QR backend has an
exact certified step.
-/
theorem QR.exists_exactCertifiedStep {m n : Nat}
    (lr momentum : ℝ) (momentumBuffer parameters gradients : MatrixTensor ℝ m n)
    (hpivots :
      HasPositivePivots
        (update
          ({ lr := lr, momentum := momentum, momentumBuffer := momentumBuffer,
             orthogonalizer := orthogonalizer (m := m) (n := n) } :
            State ℝ (.dim m (.dim n .scalar)))
          parameters gradients).optimizerState.momentumBuffer) :
    ∃ direction : MatrixTensor ℝ m n,
      ExactCertifiedStep
        ({ lr := lr, momentum := momentum, momentumBuffer := momentumBuffer,
           orthogonalizer := orthogonalizer (m := m) (n := n) } :
          State ℝ (.dim m (.dim n .scalar)))
        parameters gradients direction := by
  exact exactCertifiedStep_of_success
    (backend := checked (m := m) (n := n))
    (lr := lr) (momentum := momentum) (momentumBuffer := momentumBuffer)
    (parameters := parameters) (gradients := gradients)
    hpivots

end Muon

end TorchLean
