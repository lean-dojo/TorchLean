/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.Fourier.ZMod
public import Mathlib.LinearAlgebra.Matrix.ConjTranspose
public import TorchLean.Tensor.Layers.RealFFT.Real
import Mathlib.LinearAlgebra.Matrix.SemiringInverse

/-!
# The packed real Fourier matrices and the complex DFT

We compare `TorchLean.RealFFT.forwardMatrix` and `TorchLean.RealFFT.inverseMatrix`, the matrices of
`Tensor.rfft` and `Tensor.irfft`, with Mathlib's discrete Fourier transform `ZMod.dft` on
`ZMod n → ℂ`. Its kernel is the character `ZMod.stdAddChar`, with
`ZMod.stdAddChar j = exp (2πi j / n)`, so `ZMod.stdAddChar (-(j * k))` is the forward root power
$\omega_n^{jk}$ and `ZMod.stdAddChar (j * k)` the inverse one $\zeta_n^{jk}$. Mathlib proves the
inversion of vectors: `ZMod.dft` is a linear equivalence, and `ZMod.invDFT_apply` writes out its
inverse. Everything stated with `ZMod.stdAddChar` takes `[NeZero n]`, the condition under which
the character exists. The root powers in Euler form hold for every `n`, `j` and `k`, with
$\zeta_n$ written as `Complex.exp (2 * Real.pi * Complex.I / n)` and $\omega_n$ as its inverse.

`dftMatrix n` and `idftMatrix n` have these kernels as entries. We prove that they are inverse to
each other and that `idftMatrix n = (1 / n) • (dftMatrix n)ᴴ`, and we write every entry in Euler
form.

At `ℝ`, column `2k` of the forward matrix is the real part of $\omega_n^{tk}$ and column `2k + 1`
its imaginary part, for every stored bin `k ≤ n / 2`. The exact zeros at DC and at the even-length
Nyquist bin are those imaginary parts. Row `2k` of the inverse matrix is the bin's multiplicity
over `n` times the real part of $\zeta_n^{tk}$, and row `2k + 1` the same weight times minus its
imaginary part. These are exact identities over `ℝ` and `ℂ` about the specification matrices.
-/

@[expose] public section

open scoped Matrix

noncomputable section

namespace TorchLean.RealFFT

/-- The DFT matrix of length `n`, indexed by `ZMod n`: entry `(k, j)` is the kernel
`ZMod.stdAddChar (-(j * k))` of `ZMod.dft`, that is $\omega_n^{jk}$ with
$\omega_n=\exp(-2\pi i/n)$. -/
def dftMatrix (n : ℕ) [NeZero n] : Matrix (ZMod n) (ZMod n) ℂ :=
  fun k j => ZMod.stdAddChar (-(j * k))

/-- The inverse DFT matrix of length `n`: entry `(j, k)` is `ZMod.stdAddChar (j * k) / n`, that is
$\zeta_n^{jk}/n$ with $\zeta_n=\exp(2\pi i/n)$. -/
def idftMatrix (n : ℕ) [NeZero n] : Matrix (ZMod n) (ZMod n) ℂ :=
  fun j k => ZMod.stdAddChar (j * k) / n

/-- The inverse DFT matrix is a left inverse of the DFT matrix: $F^{-1}F=I$ over `ℂ`. -/
theorem idft_mul_dft (n : ℕ) [NeZero n] : idftMatrix n * dftMatrix n = 1 := by
  have hn : (n : ℂ) ≠ 0 := Nat.cast_ne_zero.mpr (NeZero.ne n)
  ext i k
  -- Entry `(i, k)` is `n⁻¹ * ∑ j, ψ (j * (i - k))` for the primitive character `ψ`, and the sum
  -- is `n` at `i = k` and `0` otherwise.
  have hentry : (idftMatrix n * dftMatrix n) i k =
      (n : ℂ)⁻¹ * ∑ j : ZMod n, ZMod.stdAddChar (j * (i - k)) := by
    simp only [Matrix.mul_apply, idftMatrix, dftMatrix, Finset.mul_sum]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [div_eq_inv_mul, mul_assoc, ← AddChar.map_add_eq_mul,
      show i * j + -(k * j) = j * (i - k) by ring]
  rw [hentry, AddChar.sum_mulShift _ (ZMod.isPrimitive_stdAddChar n), ZMod.card, Matrix.one_apply]
  by_cases h : i = k
  · simp [h, hn]
  · simp [h, sub_eq_zero]

/-- Conjugation sends the forward kernel to the inverse one. -/
private theorem star_stdAddChar_neg (n : ℕ) [NeZero n] (x : ZMod n) :
    star (ZMod.stdAddChar (-x)) = ZMod.stdAddChar x := by
  rw [ZMod.stdAddChar_apply, ZMod.stdAddChar_apply, AddChar.map_neg_eq_inv,
    Circle.coe_inv_eq_conj, starRingEnd_apply, star_star]

/-- On `ℂ` the inverse DFT matrix is a scaled conjugate transpose of the DFT matrix:
$F^{-1}=n^{-1}F^{\mathrm H}$. -/
theorem idftMatrix_eq_smul_conjTranspose (n : ℕ) [NeZero n] :
    idftMatrix n = (1 / (n : ℂ)) • (dftMatrix n)ᴴ := by
  ext j k
  simp only [idftMatrix, dftMatrix, Matrix.smul_apply, Matrix.conjTranspose_apply, smul_eq_mul,
    star_stdAddChar_neg]
  ring

/-- Orthogonality of the DFT columns: $F^{\mathrm H}F=nI$, so the columns are orthogonal with
squared norm `n`. -/
theorem conjTranspose_dft_mul_dft (n : ℕ) [NeZero n] :
    (dftMatrix n)ᴴ * dftMatrix n = (n : ℂ) • 1 := by
  have hn : (n : ℂ) ≠ 0 := Nat.cast_ne_zero.mpr (NeZero.ne n)
  have h := idft_mul_dft n
  rw [idftMatrix_eq_smul_conjTranspose, Matrix.smul_mul] at h
  rw [← h, smul_smul, mul_one_div_cancel hn, one_smul]

/-- The inverse DFT matrix is also a right inverse: $FF^{-1}=I$. -/
theorem dft_mul_idft (n : ℕ) [NeZero n] : dftMatrix n * idftMatrix n = 1 :=
  -- Square matrices over a commutative semiring are Dedekind-finite.
  mul_eq_one_symm (idft_mul_dft n)

/-- The forward root powers in Euler form: $\omega_n^{jk}=\cos\theta-i\sin\theta$ with
$\theta=2\pi jk/n$, for every `n`. At `n = 0` both sides are `1`. -/
theorem omega_pow_eq_cos_sub_sin (n j k : ℕ) :
    (Complex.exp (2 * Real.pi * Complex.I / n))⁻¹ ^ (j * k) =
      Complex.cos (2 * Real.pi * j * k / n) -
        Complex.I * Complex.sin (2 * Real.pi * j * k / n) := by
  set θ : ℂ := 2 * Real.pi * j * k / n
  have hEuler : Complex.exp (-(θ * Complex.I)) = Complex.cos θ - Complex.I * Complex.sin θ := by
    rw [← neg_mul, Complex.exp_mul_I, Complex.cos_neg, Complex.sin_neg]
    ring
  rw [← Complex.exp_neg, ← Complex.exp_nat_mul, ← hEuler]
  congr 1
  simp only [θ, Nat.cast_mul]
  ring

/-- The inverse root powers in Euler form: $\zeta_n^{jk}=\cos\theta+i\sin\theta$ with
$\theta=2\pi jk/n$, for every `n`. At `n = 0` both sides are `1`. -/
theorem zeta_pow_eq_cos_add_sin (n j k : ℕ) :
    Complex.exp (2 * Real.pi * Complex.I / n) ^ (j * k) =
      Complex.cos (2 * Real.pi * j * k / n) +
        Complex.I * Complex.sin (2 * Real.pi * j * k / n) := by
  set θ : ℂ := 2 * Real.pi * j * k / n
  have hEuler : Complex.exp (θ * Complex.I) = Complex.cos θ + Complex.I * Complex.sin θ := by
    rw [Complex.exp_mul_I]
    ring
  rw [← Complex.exp_nat_mul, ← hEuler]
  congr 1
  simp only [θ, Nat.cast_mul]
  ring

/-- The inverse kernel at `m` is $\zeta_n^m$. -/
private theorem stdAddChar_natCast (n : ℕ) [NeZero n] (m : ℕ) :
    ZMod.stdAddChar (m : ZMod n) = Complex.exp (2 * Real.pi * Complex.I / n) ^ m := by
  rw [ZMod.stdAddChar_apply, ZMod.toCircle_natCast, ← Complex.exp_nat_mul]
  congr 1
  ring

/-- The forward kernel at `m` is $\omega_n^m$. -/
private theorem stdAddChar_neg_natCast (n : ℕ) [NeZero n] (m : ℕ) :
    ZMod.stdAddChar (-(m : ZMod n)) = (Complex.exp (2 * Real.pi * Complex.I / n))⁻¹ ^ m := by
  rw [AddChar.map_neg_eq_inv, stdAddChar_natCast, inv_pow]

/-- Every entry of `dftMatrix n` in Euler form, with the indices read as the naturals `j.val` and
`k.val`. -/
theorem dftMatrix_apply (n : ℕ) [NeZero n] (k j : ZMod n) :
    dftMatrix n k j =
      Complex.cos (2 * Real.pi * j.val * k.val / n) -
        Complex.I * Complex.sin (2 * Real.pi * j.val * k.val / n) := by
  rw [← omega_pow_eq_cos_sub_sin, ← stdAddChar_neg_natCast, Nat.cast_mul, ZMod.natCast_zmod_val,
    ZMod.natCast_zmod_val]
  rfl

/-- Every entry of `idftMatrix n` in Euler form, with the indices read as the naturals `j.val` and
`k.val`. -/
theorem idftMatrix_apply (n : ℕ) [NeZero n] (j k : ZMod n) :
    idftMatrix n j k =
      (Complex.cos (2 * Real.pi * j.val * k.val / n) +
        Complex.I * Complex.sin (2 * Real.pi * j.val * k.val / n)) / n := by
  rw [← zeta_pow_eq_cos_add_sin, ← stdAddChar_natCast, Nat.cast_mul, ZMod.natCast_zmod_val,
    ZMod.natCast_zmod_val]
  rfl

private theorem ofReal_angle (n t k : ℕ) :
    (2 * Real.pi * t * k / n : ℂ) = ((2 * Real.pi * t * k / n : ℝ) : ℂ) := by
  push_cast
  ring

/-- The real part of $\omega_n^{tk}$ is $\cos(2\pi tk/n)$. -/
theorem omega_pow_re (n t k : ℕ) :
    ((Complex.exp (2 * Real.pi * Complex.I / n))⁻¹ ^ (t * k)).re =
      Real.cos (2 * Real.pi * t * k / n) := by
  rw [omega_pow_eq_cos_sub_sin, ofReal_angle, Complex.sub_re, Complex.cos_ofReal_re, Complex.mul_re,
    Complex.I_re, Complex.I_im, ← Complex.ofReal_sin, Complex.ofReal_re, Complex.ofReal_im]
  ring

/-- The imaginary part of $\omega_n^{tk}$ is $-\sin(2\pi tk/n)$. -/
theorem omega_pow_im (n t k : ℕ) :
    ((Complex.exp (2 * Real.pi * Complex.I / n))⁻¹ ^ (t * k)).im =
      -Real.sin (2 * Real.pi * t * k / n) := by
  rw [omega_pow_eq_cos_sub_sin, ofReal_angle, Complex.sub_im, ← Complex.ofReal_cos,
    Complex.ofReal_im, Complex.mul_im, Complex.I_re, Complex.I_im, ← Complex.ofReal_sin,
    Complex.ofReal_re, Complex.ofReal_im]
  ring

/-- The real part of $\zeta_n^{tk}$ is $\cos(2\pi tk/n)$. -/
theorem zeta_pow_re (n t k : ℕ) :
    (Complex.exp (2 * Real.pi * Complex.I / n) ^ (t * k)).re =
      Real.cos (2 * Real.pi * t * k / n) := by
  rw [zeta_pow_eq_cos_add_sin, ofReal_angle, Complex.add_re, Complex.cos_ofReal_re, Complex.mul_re,
    Complex.I_re, Complex.I_im, ← Complex.ofReal_sin, Complex.ofReal_re, Complex.ofReal_im]
  ring

/-- The imaginary part of $\zeta_n^{tk}$ is $\sin(2\pi tk/n)$. -/
theorem zeta_pow_im (n t k : ℕ) :
    (Complex.exp (2 * Real.pi * Complex.I / n) ^ (t * k)).im =
      Real.sin (2 * Real.pi * t * k / n) := by
  rw [zeta_pow_eq_cos_add_sin, ofReal_angle, Complex.add_im, ← Complex.ofReal_cos,
    Complex.ofReal_im, Complex.mul_im, Complex.I_re, Complex.I_im, ← Complex.ofReal_sin,
    Complex.ofReal_re, Complex.ofReal_im]
  ring

/-- At DC and at the even-length Nyquist bin the angle $2\pi tk/n$ is a whole number of half
turns, so its sine vanishes. -/
theorem sin_eq_zero_of_isEndpoint {n t k : ℕ} (ht : t < n)
    (h : RealFFT.isEndpoint n k = true) : Real.sin (2 * Real.pi * t * k / n) = 0 := by
  simp only [RealFFT.isEndpoint, Bool.or_eq_true, beq_iff_eq, Bool.and_eq_true] at h
  rcases h with rfl | ⟨hn, rfl⟩
  · simp
  · obtain ⟨m, rfl⟩ : ∃ m, n = 2 * m := ⟨n / 2, by omega⟩
    have hm : (m : ℝ) ≠ 0 := by
      have : 0 < m := by omega
      exact_mod_cast this.ne'
    have : 2 * Real.pi * t * ((2 * m / 2 : ℕ) : ℝ) / ((2 * m : ℕ) : ℝ) = t * Real.pi := by
      rw [show 2 * m / 2 = m by omega]
      push_cast
      field_simp
    rw [this, Real.sin_nat_mul_pi]

/-- Each entry of the packed forward matrix at `ℝ`: column `2k` is the real part and column
`2k + 1` the imaginary part of the forward kernel `ZMod.stdAddChar (-(t * k))`, at every stored
bin. -/
theorem get2_forwardMatrix (n : ℕ) [NeZero n] (t : Fin n) (p : Fin ((n / 2 + 1) * 2)) :
    Tensor.get2 (RealFFT.forwardMatrix (α := ℝ) n) t p =
      if p.val % 2 = 0 then (ZMod.stdAddChar (-((t.val * (p.val / 2) : ℕ) : ZMod n))).re
      else (ZMod.stdAddChar (-((t.val * (p.val / 2) : ℕ) : ZMod n))).im := by
  rw [RealFFT.forwardMatrix_real]
  simp only [Tensor.get2, Tensor.unstack_dim, Tensor.getScalar_dim,
    beq_iff_eq, stdAddChar_neg_natCast, omega_pow_re, omega_pow_im]
  split_ifs with h1 h2
  · rfl
  · rw [sin_eq_zero_of_isEndpoint t.isLt h2]
    simp
  · rfl

/-- Each entry of the packed inverse matrix at `ℝ`: the bin's multiplicity over `n`, times the real
part of the inverse kernel `ZMod.stdAddChar (t * k)` in row `2k` and minus its imaginary part in
row `2k + 1`. -/
theorem get2_inverseMatrix (n : ℕ) [NeZero n] (p : Fin ((n / 2 + 1) * 2)) (t : Fin n) :
    Tensor.get2 (RealFFT.inverseMatrix (α := ℝ) n) p t =
      (RealFFT.multiplicity n (p.val / 2) : ℝ) / n *
        if p.val % 2 = 0 then (ZMod.stdAddChar ((t.val * (p.val / 2) : ℕ) : ZMod n)).re
        else -(ZMod.stdAddChar ((t.val * (p.val / 2) : ℕ) : ZMod n)).im := by
  rw [RealFFT.inverseMatrix_real]
  simp only [Tensor.get2, Tensor.unstack_dim, Tensor.getScalar_dim,
    beq_iff_eq, stdAddChar_natCast, zeta_pow_re, zeta_pow_im]
  split_ifs with h1 h2
  · rfl
  · rw [sin_eq_zero_of_isEndpoint t.isLt h2]
    simp
  · ring

end TorchLean.RealFFT
