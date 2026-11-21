/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Op.Cert
public import TorchLean.NN.FDeriv.TensorCoordinates
public import TorchLean.Tensor.Euclidean
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.ComplexDeriv
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
public import Mathlib.Analysis.SpecialFunctions.Complex.LogDeriv
public import Mathlib.Analysis.SpecialFunctions.Pow.Deriv

/-!
# The derivative certificates of the complex functions and the complex power

A complex tensor of shape `s` is the real tensor of shape `s ++ [2]` with the real part of the
entry at `c` at `Tensor.pairCoord c 0` and its imaginary part at `Tensor.pairCoord c 1`, read as a
complex number by `pairValue`. A map whose entry at each `c` has the real derivative `L c` has the
derivative `ofPairsCLM L` (`hasFDerivAt_ofPairs`). So a map that applies a complex function `g c`
to the entry at each `c`, where `g c` has the complex derivative `D c` at the entry `c` of `x`, has
as real derivative at `x` the map `pairMulCLM D` that multiplies the entry at `c` by `D c`. Its
adjoint for the Euclidean inner product of the real tensors multiplies the entry at `c` by the
conjugate of `D c` (`pairMulCLM_adjoint`), so a pullback that does so is the adjoint of the
derivative (`pairwise_backward_eq_adjoint`). A map of two tensors whose entry at `c` has the
partial complex derivatives `A c` and `B c` has the derivative `pairMulCLM A` in the first plus
`pairMulCLM B` in the second, and the pullback that multiplies by the conjugates of `A c` and
`B c` is its adjoint (`pairwise₂_backward_eq_adjoint`).

Over the reals `Op.complex f` applies `ComplexFn.value f` to every entry and multiplies the entries
of the cotangent by the conjugates of Mathlib's `deriv` of it. The derivative certificate
(`op_cert`), `complex_backward_eq_adjoint`, holds for every function `f`, at every shape and every
input where `ComplexFn.value f` is complex differentiable at every entry. Each function is
complex differentiable on its set `ComplexFn.domain` (`ComplexFn.differentiableAt_value`), so
`Op.complex f` has the certificate at every input whose entries are in it
(`complex_domain_backward_eq_adjoint`). The set is all of `ℂ` for the six entire functions `exp`,
`expm1`, `sin`, `cos`, `sinh` and `cosh`; for `tan` and `tanh` the points where `cos` and `cosh`
are not zero. For the other ten it is `ℂ` off the branch cut, the points where one expression is in
Mathlib's slit plane `Complex.slitPlane` (off the closed negative real axis): `w` for `log`, `sqrt`
and `rsqrt` (off `(-∞, 0]`), `1 + w` for `log1p` (off `(-∞, -1]`), `1 - w ^ 2` for `asin`, `acos`
and `atanh` (off `(-∞, -1]` and `[1, ∞)`, `ComplexFn.one_sub_sq_mem_slitPlane`), `1 + w ^ 2` for
`atan` and `asinh` (off the points `y i` with `|y| ≥ 1`, `ComplexFn.one_add_sq_mem_slitPlane`) and
`w - 1` for `acosh` (off `(-∞, 1]`). There every logarithm and root in the formula of
`ComplexFn.value` takes its argument in the slit plane: the principal square root of a point of it
has a positive real part, so `i w + (1 - w²) ^ (1 / 2)`, `w + (1 + w²) ^ (1 / 2)` and the sum of
the two roots of `acosh` are in it, and so is `(1 + u) / (1 - u)` where `1 - u ^ 2` is. It is not
proved that the functions are not differentiable on the cuts.

Over the reals `Op.complexPow` takes Mathlib's principal power `z ^ w` of the entries of the base
and the exponent, and multiplies the entries of the cotangent by the conjugates of the partials
`w z ^ (w - 1)` and `z ^ w log z`. It has the certificate in both inputs at every input whose base
entries are in the slit plane (`complexPow_backward_eq_adjoint`, from
`Complex.hasStrictFDerivAt_cpow`), and in the exponent alone at every input where each entry of
the base or of the exponent is not zero (`complexPow_backward_eq_adjoint_right`), which is exact.
-/

@[expose] public section

namespace TorchLean.ComplexFn

open Complex in
/-- A set on which `f` is complex differentiable: all of `ℂ` for `exp`, `expm1`, `sin`, `cos`,
`sinh` and `cosh`; the points where `cos w` and `cosh w` are not zero for `tan` and `tanh`; for the
others `ℂ` off the branch cut, the points where `w` (for `log`, `sqrt` and `rsqrt`), `1 + w` (for
`log1p`), `1 - w ^ 2` (for `asin`, `acos` and `atanh`), `1 + w ^ 2` (for `atan` and `asinh`) or
`w - 1` (for `acosh`) is in the slit plane. -/
def domain : ComplexFn → Set ℂ
  | exp | expm1 | sin | cos | sinh | cosh => Set.univ
  | log | sqrt | rsqrt => slitPlane
  | log1p => {w | 1 + w ∈ slitPlane}
  | tan => {w | Complex.cos w ≠ 0}
  | tanh => {w | Complex.cosh w ≠ 0}
  | asin | acos | atanh => {w | 1 - w ^ 2 ∈ slitPlane}
  | atan | asinh => {w | 1 + w ^ 2 ∈ slitPlane}
  | acosh => {w | w - 1 ∈ slitPlane}

open Complex in
/-- `1 - w ^ 2` is in the slit plane exactly off the half-lines `(-∞, -1]` and `[1, ∞)`. -/
theorem one_sub_sq_mem_slitPlane {w : ℂ} : 1 - w ^ 2 ∈ slitPlane ↔ w.im ≠ 0 ∨ |w.re| < 1 := by
  rw [mem_slitPlane_iff, abs_lt]
  simp only [sub_re, sub_im, one_re, one_im, zero_sub, sq, mul_re, mul_im]
  constructor
  · rintro (h | h)
    · by_cases hi : w.im = 0
      · rw [hi] at h
        exact Or.inr ⟨by nlinarith, by nlinarith⟩
      · exact Or.inl hi
    · exact Or.inl fun hi => h (by rw [hi]; ring)
  · rintro (h | ⟨h₁, h₂⟩)
    · by_cases hr : w.re = 0
      · rw [hr]
        exact Or.inl (by nlinarith [mul_self_nonneg w.im])
      · exact Or.inr fun h' => mul_ne_zero hr h (by linear_combination -h' / 2)
    · have h₃ : 0 < (1 - w.re) * (1 + w.re) := mul_pos (by linarith) (by linarith)
      exact Or.inl (by nlinarith [mul_self_nonneg w.im])

open Complex in
/-- `1 + w ^ 2` is in the slit plane exactly off the points `y i` with `|y| ≥ 1`. -/
theorem one_add_sq_mem_slitPlane {w : ℂ} : 1 + w ^ 2 ∈ slitPlane ↔ w.re ≠ 0 ∨ |w.im| < 1 := by
  simpa [mul_pow] using one_sub_sq_mem_slitPlane (w := w * I)

open Complex in
/-- A denominator is not zero where its quotient is in the slit plane. -/
private theorem denom_ne_zero {a b : ℂ} (h : a / b ∈ slitPlane) : b ≠ 0 := fun hb =>
  slitPlane_ne_zero h (by rw [hb, div_zero])

open Complex in
/-- `(1 + u) / (1 - u)` is in the slit plane where `1 - u ^ 2` is. -/
private theorem one_add_div_mem {u : ℂ} (h : 1 - u ^ 2 ∈ slitPlane) :
    (1 + u) / (1 - u) ∈ slitPlane := by
  have hN : 0 < normSq (1 - u) := normSq_pos.2 fun h0 =>
    slitPlane_ne_zero h (by linear_combination (1 + u) * h0)
  rw [mem_slitPlane_iff, div_re, div_im]
  simp only [add_re, add_im, sub_re, sub_im, one_re, one_im, zero_add, zero_sub]
  by_cases hy : u.im = 0
  · have hx := abs_lt.1 ((one_sub_sq_mem_slitPlane.1 h).resolve_left (not_not.2 hy))
    have h₃ : 0 < (1 + u.re) * (1 - u.re) := mul_pos (by linarith [hx.1]) (by linarith [hx.2])
    rw [hy, ← add_div]
    exact Or.inl (div_pos (by linarith) hN)
  · rw [← sub_div]
    exact Or.inr (div_ne_zero (fun h' => hy (by linear_combination h' / 2)) hN.ne')

open Complex in
/-- The principal square root of a point of the slit plane has a positive real part. -/
private theorem root_re_pos {z : ℂ} (hz : z ∈ slitPlane) : 0 < (z ^ (2⁻¹ : ℂ)).re := by
  rw [cpow_inv_two_re]
  refine Real.sqrt_pos.2 (half_pos ?_)
  rcases mem_slitPlane_iff.1 hz with h | h
  · linarith [norm_nonneg z]
  · linarith [neg_abs_le z.re, abs_re_lt_norm.2 h]

open Complex in
/-- `a + s` is in the slit plane where `s` has a positive real part and `s ^ 2 = 1 + a ^ 2`. -/
private theorem add_root_mem {a s : ℂ} (hs : 0 < s.re) (h : s ^ 2 = 1 + a ^ 2) :
    a + s ∈ slitPlane := by
  rw [mem_slitPlane_iff]
  rcases ne_or_eq (a + s).im 0 with hi | hi
  · exact Or.inr hi
  · left
    have hre := congrArg Complex.re h
    simp only [sq, mul_re, add_re, add_im, one_re] at hre hi ⊢
    have him : a.im = -s.im := by linarith
    rw [him] at hre
    by_contra hn
    rw [not_lt] at hn
    nlinarith [mul_nonneg_of_nonpos_of_nonpos hn (by linarith : a.re - s.re ≤ 0)]

open Complex in
/-- Every function is complex differentiable at every point of its `domain`. -/
theorem differentiableAt_value {f : ComplexFn} {w : ℂ} (hw : w ∈ f.domain) :
    DifferentiableAt ℂ f.value w := by
  cases f
  case exp => exact differentiable_exp w
  case expm1 => exact differentiable_exp.sub_const 1 w
  case sin => exact differentiable_sin w
  case cos => exact differentiable_cos w
  case sinh => exact differentiable_sinh w
  case cosh => exact differentiable_cosh w
  case log => exact differentiableAt_id.clog hw
  case log1p => exact ((differentiableAt_const 1).add differentiableAt_id).clog hw
  case sqrt => exact differentiableAt_id.cpow_const hw
  case rsqrt =>
    exact (differentiableAt_id.cpow_const hw).inv
      ((cpow_ne_zero_iff_of_exponent_ne_zero (inv_ne_zero two_ne_zero)).2 (slitPlane_ne_zero hw))
  case tan => exact differentiableAt_tan.2 hw
  case tanh => exact (differentiable_sinh w).div (differentiable_cosh w) hw
  case asin =>
    have hw : 1 - w ^ 2 ∈ slitPlane := hw
    have hs : I * w + (1 - w ^ 2) ^ (2⁻¹ : ℂ) ∈ slitPlane :=
      add_root_mem (root_re_pos hw) (by rw [cpow_ofNat_inv_pow, mul_pow, I_sq]; ring)
    exact (differentiableAt_const _).mul ((((differentiableAt_const _).mul differentiableAt_id).add
      (((differentiableAt_const _).sub (differentiableAt_id.pow 2)).cpow_const
        (by exact hw))).clog (by exact hs))
  case acos =>
    have hw : 1 - w ^ 2 ∈ slitPlane := hw
    have hs : I * w + (1 - w ^ 2) ^ (2⁻¹ : ℂ) ∈ slitPlane :=
      add_root_mem (root_re_pos hw) (by rw [cpow_ofNat_inv_pow, mul_pow, I_sq]; ring)
    exact (differentiableAt_const _).add ((differentiableAt_const _).mul
      ((((differentiableAt_const _).mul differentiableAt_id).add
        (((differentiableAt_const _).sub (differentiableAt_id.pow 2)).cpow_const
          (by exact hw))).clog (by exact hs)))
  case atan =>
    have hw : (1 + w * I) / (1 - w * I) ∈ slitPlane :=
      one_add_div_mem (by rw [mul_pow, I_sq, mul_neg_one, sub_neg_eq_add]; exact hw)
    show DifferentiableAt ℂ (fun z => -I / 2 * Complex.log ((1 + z * I) / (1 - z * I))) w
    exact (differentiableAt_const _).mul ((((differentiableAt_const _).add
      (differentiableAt_id.mul_const I)).div ((differentiableAt_const _).sub
        (differentiableAt_id.mul_const I)) (by exact denom_ne_zero hw)).clog (by exact hw))
  case asinh =>
    have hw : 1 + w ^ 2 ∈ slitPlane := hw
    have hs : w + (1 + w ^ 2) ^ (2⁻¹ : ℂ) ∈ slitPlane :=
      add_root_mem (root_re_pos hw) (cpow_ofNat_inv_pow _ 2)
    exact (differentiableAt_id.add (((differentiableAt_const _).add
      (differentiableAt_id.pow 2)).cpow_const (by exact hw))).clog (by exact hs)
  case acosh =>
    have hw : w - 1 ∈ slitPlane := hw
    have h₁ : (w + 1) / 2 ∈ slitPlane := by
      rw [mem_slitPlane_iff] at hw ⊢
      simp only [div_ofNat_re, div_ofNat_im, add_re, add_im, sub_re, sub_im, one_re, one_im,
        add_zero, sub_zero] at hw ⊢
      exact hw.imp (fun h => by linarith) (fun h => div_ne_zero h two_ne_zero)
    have h₂ : (w - 1) / 2 ∈ slitPlane := by
      rw [mem_slitPlane_iff] at hw ⊢
      simp only [div_ofNat_re, div_ofNat_im]
      exact hw.imp (fun h => by linarith) (fun h => div_ne_zero h two_ne_zero)
    have hs : ((w + 1) / 2) ^ (2⁻¹ : ℂ) + ((w - 1) / 2) ^ (2⁻¹ : ℂ) ∈ slitPlane :=
      mem_slitPlane_iff.2 (Or.inl (by rw [add_re]; exact add_pos (root_re_pos h₁) (root_re_pos h₂)))
    exact (differentiableAt_const _).mul
      (((((differentiableAt_id.add (differentiableAt_const _)).div_const 2).cpow_const
          (by exact h₁)).add
        (((differentiableAt_id.sub (differentiableAt_const _)).div_const 2).cpow_const
          (by exact h₂))).clog (by exact hs))
  case atanh =>
    have hw : (1 + w) / (1 - w) ∈ slitPlane := one_add_div_mem hw
    exact ((((differentiableAt_const _).add differentiableAt_id).div
      ((differentiableAt_const _).sub differentiableAt_id) (by exact denom_ne_zero hw)).clog
        (by exact hw)).div_const 2

end TorchLean.ComplexFn

namespace TorchLean.Op

open TorchLean.Tensor

variable {s : Shape}

/-- The entry at `c` of a tensor in the layout `s ++ [2]`, as a complex number. -/
def pairValue (y : Tensor ℝ (s ++ [2])) (c : s.Coord) : ℂ :=
  ⟨y (pairCoord c 0), y (pairCoord c 1)⟩

/-- Reading the entry at `c` as a complex number, a real-linear map. -/
noncomputable def pairCLM (c : s.Coord) : Tensor ℝ (s ++ [2]) →L[ℝ] ℂ :=
  (coordinateCLM (pairCoord c 0)).smulRight (1 : ℂ) +
    (coordinateCLM (pairCoord c 1)).smulRight Complex.I

/-- `pairCLM c` reads the entry at `c`. -/
@[simp] theorem pairCLM_apply (c : s.Coord) (y : Tensor ℝ (s ++ [2])) :
    pairCLM c y = pairValue y c := by
  apply Complex.ext <;> simp [pairCLM, pairValue]

/-- The real-linear map into the layout `s ++ [2]` whose entry at each `c` is `L c`. -/
noncomputable def ofPairsCLM {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (L : s.Coord → E →L[ℝ] ℂ) : E →L[ℝ] Tensor ℝ (s ++ [2]) :=
  Tensor.assemble fun e =>
    let parts := Shape.Coord.appendEquiv s [2] e
    (if parts.2.1 = 0 then Complex.reCLM else Complex.imCLM).comp (L parts.1)

/-- The entry at each `c` of `ofPairsCLM L y` is `L c y`. -/
theorem ofPairsCLM_apply {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (L : s.Coord → E →L[ℝ] ℂ) (y : E) :
    ofPairsCLM L y = ofPairs fun c => ((L c y).re, (L c y).im) := by
  apply Tensor.Rep.ext
  intro e
  simp only [ofPairsCLM, Tensor.assemble_apply, ofPairs, Rep.get_ofFn]
  split_ifs <;> simp

/-- Multiplying the entry at each `c` by `D c`, as a real-linear map on the layout `s ++ [2]`. -/
noncomputable def pairMulCLM (D : s.Coord → ℂ) :
    Tensor ℝ (s ++ [2]) →L[ℝ] Tensor ℝ (s ++ [2]) :=
  ofPairsCLM fun c => D c • pairCLM c

/-- `pairMulCLM D` multiplies the entry at each `c` by `D c`. -/
theorem pairMulCLM_apply (D : s.Coord → ℂ) (y : Tensor ℝ (s ++ [2])) :
    pairMulCLM D y = ofPairs fun c => ((D c * pairValue y c).re, (D c * pairValue y c).im) := by
  simp [pairMulCLM, ofPairsCLM_apply]

/-- A coordinate of the layout `s ++ [2]` is a coordinate of `s` and a part. -/
private def pairEquiv (s : Shape) : s.Coord × Fin 2 ≃ (s ++ [2]).Coord :=
  ((Equiv.refl s.Coord).prodCongr (Equiv.prodPUnit (Fin 2)).symm).trans
    (Shape.Coord.appendEquiv s [2]).symm

/-- The inner product of tensors in the layout `s ++ [2]`, summed pair by pair. -/
private theorem inner_eq_sum_pairs (a b : Tensor ℝ (s ++ [2])) :
    inner ℝ a b = ∑ c, (a (pairCoord c 0) * b (pairCoord c 0) +
      a (pairCoord c 1) * b (pairCoord c 1)) := by
  rw [Tensor.inner_eq_sum, ← (pairEquiv s).sum_comp, Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun c _ => ?_
  rw [Fin.sum_univ_two]
  rfl

/-- The adjoint of `pairMulCLM D` multiplies the entry at each `c` by the conjugate of `D c`. -/
theorem pairMulCLM_adjoint (D : s.Coord → ℂ) (δ : Tensor ℝ (s ++ [2])) :
    (pairMulCLM D).adjoint δ = ofPairs fun c =>
      ((D c).re * δ (pairCoord c 0) + (D c).im * δ (pairCoord c 1),
        (D c).re * δ (pairCoord c 1) - (D c).im * δ (pairCoord c 0)) := by
  refine ext_inner_left ℝ fun u => ?_
  rw [ContinuousLinearMap.adjoint_inner_right, inner_eq_sum_pairs, inner_eq_sum_pairs,
    pairMulCLM_apply]
  refine Finset.sum_congr rfl fun c _ => ?_
  have h1 : ((1 : Fin 2) = 0) = False := eq_false (by decide)
  simp only [ofPairs_pairCoord, h1, reduceIte, Complex.mul_re, Complex.mul_im, pairValue]
  ring

/-- A map whose entry at each `c` is `G c`, where `G c` has the real derivative `L c` at `x`, has
the derivative `ofPairsCLM L` at `x`. -/
theorem hasFDerivAt_ofPairs {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (F : E → Tensor ℝ (s ++ [2])) (G : s.Coord → E → ℂ) (L : s.Coord → E →L[ℝ] ℂ) (x : E)
    (hF : ∀ y, F y = ofPairs fun c => ((G c y).re, (G c y).im))
    (hG : ∀ c, HasFDerivAt (G c) (L c) x) : HasFDerivAt F (ofPairsCLM L) x := by
  refine Tensor.hasFDerivAt_of_coordinates F _ x fun e => ?_
  simp only [hF, ofPairs, Rep.get_ofFn]
  split_ifs
  · simpa only [Function.comp_def, Complex.reCLM_apply] using
      Complex.reCLM.hasFDerivAt.comp x (hG _)
  · simpa only [Function.comp_def, Complex.imCLM_apply] using
      Complex.imCLM.hasFDerivAt.comp x (hG _)

/-- Reading the entry at `c` has the derivative `pairCLM c`. -/
theorem pairValue_hasFDerivAt (c : s.Coord) (x : Tensor ℝ (s ++ [2])) :
    HasFDerivAt (fun y => pairValue y c) (pairCLM c) x := by
  have h := (pairCLM c).hasFDerivAt (x := x)
  rwa [show ⇑(pairCLM c) = fun y => pairValue y c from funext (pairCLM_apply c)] at h

/-- If `F` applies `g c` to the entry at each `c`, and `g c` has the complex derivative `D c` at
the entry `c` of `x`, then `pairMulCLM D` is the derivative of `F` at `x`, and a cotangent `v`
whose entry at `c` is the entry of `δ` at `c` times the conjugate of `D c` is the adjoint of that
derivative applied to `δ`. -/
theorem pairwise_backward_eq_adjoint (F : Tensor ℝ (s ++ [2]) → Tensor ℝ (s ++ [2]))
    (g : s.Coord → ℂ → ℂ) (D : s.Coord → ℂ) (x δ v : Tensor ℝ (s ++ [2]))
    (hF : ∀ y, F y = ofPairs fun c => ((g c (pairValue y c)).re, (g c (pairValue y c)).im))
    (hg : ∀ c, HasDerivAt (g c) (D c) (pairValue x c))
    (hv : v = ofPairs fun c =>
      ((D c).re * δ (pairCoord c 0) + (D c).im * δ (pairCoord c 1),
        (D c).re * δ (pairCoord c 1) - (D c).im * δ (pairCoord c 0))) :
    HasFDerivAt F (pairMulCLM D) x ∧ v = (fderiv ℝ F x).adjoint δ := by
  have h : HasFDerivAt F (pairMulCLM D) x := hasFDerivAt_ofPairs F (fun c y => g c (pairValue y c))
    _ x hF fun c => (hg c).comp_hasFDerivAt x (pairValue_hasFDerivAt c x)
  exact ⟨h, by rw [h.fderiv, pairMulCLM_adjoint, hv]⟩

/-- If `F` applies `g` to the entries at each `c` of two tensors, and `g` has the real derivative
`(a, b) ↦ A c a + B c b` at the entries of `x` and `y`, then `F` has the derivative `pairMulCLM A`
in the first tensor plus `pairMulCLM B` in the second at `(x, y)`, and the pair of cotangents
whose entries at `c` are the entry of `δ` times the conjugates of `A c` and of `B c` is the
adjoint of that derivative applied to `δ`. -/
theorem pairwise₂_backward_eq_adjoint
    (F : WithLp 2 (Tensor ℝ (s ++ [2]) × Tensor ℝ (s ++ [2])) → Tensor ℝ (s ++ [2]))
    (g : ℂ × ℂ → ℂ) (A B : s.Coord → ℂ) (x y δ u v : Tensor ℝ (s ++ [2]))
    (hF : ∀ p, F p = ofPairs fun c => ((g (pairValue p.fst c, pairValue p.snd c)).re,
      (g (pairValue p.fst c, pairValue p.snd c)).im))
    (hg : ∀ c, HasFDerivAt g ((A c • ContinuousLinearMap.fst ℂ ℂ ℂ +
      B c • ContinuousLinearMap.snd ℂ ℂ ℂ).restrictScalars ℝ) (pairValue x c, pairValue y c))
    (hu : u = ofPairs fun c =>
      ((A c).re * δ (pairCoord c 0) + (A c).im * δ (pairCoord c 1),
        (A c).re * δ (pairCoord c 1) - (A c).im * δ (pairCoord c 0)))
    (hv : v = ofPairs fun c =>
      ((B c).re * δ (pairCoord c 0) + (B c).im * δ (pairCoord c 1),
        (B c).re * δ (pairCoord c 1) - (B c).im * δ (pairCoord c 0))) :
    HasFDerivAt F ((pairMulCLM A).comp (WithLp.fstL 2 ℝ _ _) +
        (pairMulCLM B).comp (WithLp.sndL 2 ℝ _ _)) (WithLp.toLp 2 (x, y)) ∧
      WithLp.toLp 2 (u, v) = (fderiv ℝ F (WithLp.toLp 2 (x, y))).adjoint δ := by
  have hM : ∀ c, HasFDerivAt (fun p : WithLp 2 (Tensor ℝ (s ++ [2]) × Tensor ℝ (s ++ [2])) =>
      g (pairValue p.fst c, pairValue p.snd c))
      (A c • (pairCLM c).comp (WithLp.fstL 2 ℝ _ _) + B c • (pairCLM c).comp (WithLp.sndL 2 ℝ _ _))
      (WithLp.toLp 2 (x, y)) := fun c => by
    have hP := ((pairValue_hasFDerivAt c x).comp (WithLp.toLp 2 (x, y))
      (WithLp.fstL 2 ℝ _ _).hasFDerivAt).prodMk
      ((pairValue_hasFDerivAt c y).comp (WithLp.toLp 2 (x, y)) (WithLp.sndL 2 ℝ _ _).hasFDerivAt)
    exact ((hg c).comp (WithLp.toLp 2 (x, y)) hP).congr_fderiv
      (ContinuousLinearMap.ext fun p => by simp)
  have hD := hasFDerivAt_ofPairs F _ _ (WithLp.toLp 2 (x, y)) hF hM
  have hE : ofPairsCLM (fun c => A c • (pairCLM c).comp (WithLp.fstL 2 ℝ _ _) +
      B c • (pairCLM c).comp (WithLp.sndL 2 ℝ _ _)) =
      (pairMulCLM A).comp (WithLp.fstL 2 ℝ _ _) + (pairMulCLM B).comp (WithLp.sndL 2 ℝ _ _) := by
    refine ContinuousLinearMap.ext fun p => ?_
    apply Tensor.Rep.ext
    intro e
    simp only [ofPairsCLM, pairMulCLM, Tensor.assemble_apply, _root_.add_apply,
      ContinuousLinearMap.comp_apply, Tensor.add_apply]
    split_ifs <;> simp
  rw [hE] at hD
  refine ⟨hD, ?_⟩
  rw [hD.fderiv]
  have hA := pairMulCLM_adjoint A δ
  have hB := pairMulCLM_adjoint B δ
  rw [← hu] at hA
  rw [← hv] at hB
  refine ext_inner_left ℝ fun p => ?_
  rw [ContinuousLinearMap.adjoint_inner_right, WithLp.prod_inner_apply]
  simp only [_root_.add_apply, ContinuousLinearMap.comp_apply, inner_add_left, WithLp.fstL_apply,
    ← hA, ← hB, ContinuousLinearMap.adjoint_inner_right]
  rfl

/-- The derivative certificate of `Op.complex f`: where `ComplexFn.value f` is complex
differentiable at every entry of `x`, the forward has the derivative `pairMulCLM` of Mathlib's
`deriv` at the entries, and the pullback is its adjoint. -/
@[op_cert TorchLean.Op.complex derivative]
theorem complex_backward_eq_adjoint (f : ComplexFn) (x δ : Tensor ℝ (s ++ [2]))
    (hx : ∀ c, DifferentiableAt ℂ f.value (pairValue x c)) :
    HasFDerivAt (fun y => (((Op.complex f (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (pairMulCLM fun c => deriv f.value (pairValue x c)) x ∧
      (((Op.complex f (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.complex f (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ :=
  pairwise_backward_eq_adjoint _ (fun _ => f.value) _ x δ _ (fun _ => rfl)
    (fun c => (hx c).hasDerivAt) rfl

/-- `Op.complex f` has the certificate at every input whose entries are in `ComplexFn.domain f`. -/
theorem complex_domain_backward_eq_adjoint (f : ComplexFn) (x δ : Tensor ℝ (s ++ [2]))
    (hx : ∀ c, pairValue x c ∈ f.domain) :
    HasFDerivAt (fun y => (((Op.complex f (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (pairMulCLM fun c => deriv f.value (pairValue x c)) x ∧
      (((Op.complex f (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.complex f (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ :=
  complex_backward_eq_adjoint f x δ fun c => ComplexFn.differentiableAt_value (hx c)

/-- The derivative certificate of `Op.complexPow` in the base and the exponent: where every entry
of the base `x` is in the slit plane, off the closed negative real axis, the forward `x ^ y` has
the derivative that multiplies the entries at `c` of the tangents of `x` and `y` by
`y c x c ^ (y c - 1)` and `x c ^ y c log x c` and adds them (`Complex.hasStrictFDerivAt_cpow`),
and the pullback is its adjoint. -/
@[op_cert TorchLean.Op.complexPow derivative 0 1]
theorem complexPow_backward_eq_adjoint (x y δ : Tensor ℝ (s ++ [2]))
    (hx : ∀ c, pairValue x c ∈ Complex.slitPlane) :
    HasFDerivAt (fun p : WithLp 2 (Tensor ℝ (s ++ [2]) × Tensor ℝ (s ++ [2])) =>
        (((Op.complexPow (s := s)).rule? (α := ℝ) |>.get rfl).forward (p.fst, p.snd, ())).1)
      ((pairMulCLM fun c => pairValue y c * pairValue x c ^ (pairValue y c - 1)).comp
          (WithLp.fstL 2 ℝ _ _) +
        (pairMulCLM fun c => pairValue x c ^ pairValue y c * Complex.log (pairValue x c)).comp
          (WithLp.sndL 2 ℝ _ _))
      (WithLp.toLp 2 (x, y)) ∧
    WithLp.toLp 2
        ((((Op.complexPow (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).1,
        (((Op.complexPow (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).2.1) =
      (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ (s ++ [2]) × Tensor ℝ (s ++ [2])) =>
        (((Op.complexPow (s := s)).rule? (α := ℝ) |>.get rfl).forward (p.fst, p.snd, ())).1)
        (WithLp.toLp 2 (x, y))).adjoint δ :=
  pairwise₂_backward_eq_adjoint _ (fun q => q.1 ^ q.2) _ _ x y δ _ _ (fun _ => rfl)
    (fun c => (Complex.hasStrictFDerivAt_cpow' (hx c)).hasFDerivAt.restrictScalars ℝ) rfl rfl

/-- The derivative certificate of `Op.complexPow` in the exponent: where every entry of the base
`x` or of the exponent `y` is not zero, the forward in `y` has the derivative that multiplies the
entry at `c` by `x c ^ y c log x c`, on the negative real axis too (`w ↦ z ^ w` is
`exp (w log z)` for `z ≠ 0`), and the pullback is its adjoint. The set is exact: `w ↦ 0 ^ w` is
`1` at `0` and `0` elsewhere. -/
@[op_cert TorchLean.Op.complexPow derivative 1]
theorem complexPow_backward_eq_adjoint_right (x y δ : Tensor ℝ (s ++ [2]))
    (h : ∀ c, pairValue x c ≠ 0 ∨ pairValue y c ≠ 0) :
    HasFDerivAt (fun y' => (((Op.complexPow (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (x, y', ())).1)
        (pairMulCLM fun c => pairValue x c ^ pairValue y c * Complex.log (pairValue x c)) y ∧
      (((Op.complexPow (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).2.1 =
        (fderiv ℝ (fun y' => (((Op.complexPow (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (x, y', ())).1) y).adjoint δ :=
  pairwise_backward_eq_adjoint _ (fun c w => pairValue x c ^ w) _ y δ _ (fun _ => rfl)
    (fun c => (Complex.hasStrictDerivAt_const_cpow (h c)).hasDerivAt) rfl

end TorchLean.Op
