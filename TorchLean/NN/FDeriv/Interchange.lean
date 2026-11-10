/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.Calculus.FDeriv.Symmetric
public import Mathlib.Analysis.InnerProductSpace.Adjoint

/-!
# Interchanging higher derivatives and pullbacks

For a `C^(n+1)` function, taking `n` directional derivatives of its Fréchet derivative gives the
same linear map as differentiating its fixed `n`-th directional derivative. Over real Hilbert
spaces the adjoints agree as well, and that interchange is the calculus step behind training on
derivatives. The statements live in general normed and Hilbert spaces and involve no tensor
shapes or graphs.
-/

public section

open scoped ContDiff

namespace TorchLean

section Derivatives

variable {𝕜 : Type*} [NontriviallyNormedField 𝕜] [IsRCLikeNormedField 𝕜]
    {E F : Type*} [NormedAddCommGroup E] [NormedSpace 𝕜 E]
    [NormedAddCommGroup F] [NormedSpace 𝕜 F]

/-- For a `C^(n+1)` map, the `n`-th derivative of the directional derivative along `v` equals the
derivative along `v` of the `n`-th derivative in `directions`, which may repeat. The induction
uses the symmetry of second derivatives, so no analyticity or reordering of a power series is
needed. -/
theorem iteratedFDeriv_fderiv_apply {n : Nat} {f : E → F} (hf : ContDiff 𝕜 (n + 1) f)
    (x v : E) (directions : Fin n → E) :
    iteratedFDeriv 𝕜 n (fun y => fderiv 𝕜 f y v) x directions =
      fderiv 𝕜 (fun y => iteratedFDeriv 𝕜 n f y directions) x v := by
  induction n generalizing f with
  | zero => simp only [iteratedFDeriv_zero_apply]
  | succ n ih =>
      let w := directions (Fin.last n)
      have hd : ContDiff 𝕜 (n + 1) (fderiv 𝕜 f) := hf.fderiv_right (by simp)
      have hv : ContDiff 𝕜 (n + 1) (fun y => fderiv 𝕜 f y v) :=
        hd.clm_apply contDiff_const
      have hw : ContDiff 𝕜 (n + 1) (fun y => fderiv 𝕜 f y w) :=
        hd.clm_apply contDiff_const
      have hright : (fun y => iteratedFDeriv 𝕜 (n + 1) f y directions) =
          fun y => iteratedFDeriv 𝕜 n (fun z => fderiv 𝕜 f z w) y
            (Fin.init directions) := by
        funext y
        rw [iteratedFDeriv_succ_apply_right]
        exact (iteratedFDeriv_clm_apply_const_apply hd (by simp)
          (x := y) (m := Fin.init directions) (u := w)).symm
      rw [hright, ← ih hw]
      rw [iteratedFDeriv_succ_apply_right]
      rw [← iteratedFDeriv_clm_apply_const_apply (hv.fderiv_right (m := n) (by simp))
        (by simp)]
      have heq : (fun y => fderiv 𝕜 (fun y => fderiv 𝕜 f y v) y w) =
          fun y => fderiv 𝕜 (fun y => fderiv 𝕜 f y w) y v := by
        have h2 : ContDiff 𝕜 2 f := hf.of_le (by norm_cast; omega)
        funext y
        have hdy : DifferentiableAt 𝕜 (fderiv 𝕜 f) y :=
          (h2.fderiv_right (m := 1) (by norm_num)).differentiable (by norm_num) y
        simpa [fderiv_clm_apply hdy (differentiableAt_const _)] using
          (h2.contDiffAt (x := y)).isSymmSndFDerivAt (by simp) w v
      rw [heq]

/-- Taking the Fréchet derivative commutes with a fixed tuple of higher derivative directions. -/
theorem iteratedFDeriv_fderiv {n : Nat} {f : E → F} (hf : ContDiff 𝕜 (n + 1) f)
    (x : E) (directions : Fin n → E) :
    iteratedFDeriv 𝕜 n (fderiv 𝕜 f) x directions =
      fderiv 𝕜 (fun y => iteratedFDeriv 𝕜 n f y directions) x := by
  ext v
  rw [← iteratedFDeriv_clm_apply_const_apply (hf.fderiv_right (m := n) (by simp))
    (by simp)]
  exact iteratedFDeriv_fderiv_apply hf x v directions

end Derivatives

section Adjoint

variable {E F : Type*} [NormedAddCommGroup E] [NormedAddCommGroup F]
    [InnerProductSpace ℝ E] [InnerProductSpace ℝ F] [CompleteSpace E] [CompleteSpace F]

/-- Higher derivatives of a real pullback equal the pullback of the higher derivative. The
cotangent `seed` and the directions are held constant, and letting them vary would add terms. -/
theorem iteratedFDeriv_adjoint_fderiv {n : Nat} {f : E → F} (hf : ContDiff ℝ (n + 1) f)
    (x : E) (directions : Fin n → E) (seed : F) :
    iteratedFDeriv ℝ n (fun y => (fderiv ℝ f y).adjoint seed) x directions =
      (fderiv ℝ (fun y => iteratedFDeriv ℝ n f y directions) x).adjoint seed := by
  let adj : (E →L[ℝ] F) →L[ℝ] E :=
    (ContinuousLinearMap.apply ℝ E seed).comp
      ContinuousLinearMap.adjoint.toContinuousLinearEquiv.toContinuousLinearMap
  change iteratedFDeriv ℝ n (adj ∘ fderiv ℝ f) x directions =
    adj (fderiv ℝ (fun y => iteratedFDeriv ℝ n f y directions) x)
  rw [adj.iteratedFDeriv_comp_left (hf.fderiv_right (m := n) (by simp)).contDiffAt (by simp)]
  change adj (iteratedFDeriv ℝ n (fderiv ℝ f) x directions) = _
  rw [iteratedFDeriv_fderiv hf]

end Adjoint

end TorchLean
