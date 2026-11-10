/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.FDeriv.PullbackTensor
public import TorchLean.Tensor.Op.Table.Linear
public import TorchLean.Tensor.Op.Cert

/-!
# Derivatives of the real Fourier transforms

We prove that the stored pullbacks of `Op.rfft` and `Op.irfft` are the adjoints of their
derivatives. Both transforms are linear, so each is its own derivative, and the content is the
inner product identity between a transform and its stored pullback, the transposed matrix.

Both transforms contract a fiber of length `n` along the axis `dim` with a matrix of shape
`[n, (n / 2 + 1) * 2]` or its transpose. We pair an entry of the packed spectrum and a time index
with the time-domain entry and the packed frequency index that the contraction multiplies, as an
equivalence `rfftIndexEquiv`, and read both sides of each inner product identity as one sum
over this equivalence.
-/

@[expose] public section

open scoped RealInnerProductSpace

namespace TorchLean

namespace Tensor

variable {s : Shape}

/-- The pairing of an entry of the packed spectrum and a time index with the time-domain entry
and the packed frequency index `2 k + j` that the real Fourier matrices multiply, for the
frequency `k` and the part `j` of the spectrum entry. -/
def rfftIndexEquiv (dim : Fin s.rank) (n : Nat) :
    Shape.Coord (s.set dim (n / 2 + 1) ++ [2]) × Fin n ≃
      Shape.Coord (s.set dim n) × Fin ((n / 2 + 1) * 2) where
  toFun q :=
    let parts := Shape.Coord.appendEquiv (s.set dim (n / 2 + 1)) [2] q.1
    let split := Shape.Coord.setEquiv dim.val (shape := s) (n / 2 + 1) parts.1
    ((Shape.Coord.setEquiv dim.val (shape := s) n).symm (q.2, split.2),
      ⟨2 * split.1.val + parts.2.1.val, by omega⟩)
  invFun q :=
    let split := Shape.Coord.setEquiv dim.val (shape := s) n q.1
    ((Shape.Coord.appendEquiv (s.set dim (n / 2 + 1)) [2]).symm
        ((Shape.Coord.setEquiv dim.val (shape := s) (n / 2 + 1)).symm
          (⟨q.2.val / 2, by have := q.2.isLt; omega⟩, split.2), (⟨q.2.val % 2, by omega⟩, ())),
      split.1)
  left_inv := by
    rintro ⟨c, t⟩
    simp only [Equiv.apply_symm_apply, Prod.mk.injEq, and_true]
    rw [Equiv.symm_apply_eq]
    refine Prod.ext ?_ (Prod.ext (Fin.ext ?_) rfl)
    · rw [Equiv.symm_apply_eq]
      refine Prod.ext (Fin.ext ?_) rfl
      dsimp only
      omega
    · dsimp only
      omega
  right_inv := by
    rintro ⟨c, p⟩
    simp only [Equiv.apply_symm_apply, Prod.mk.eta, Equiv.symm_apply_apply, Prod.mk.injEq,
      true_and, Fin.ext_iff]
    omega

variable (dim : Fin s.rank) (n : Nat)

/-- An entry of `rfft dim n x` is the sum over the time index `t` of the paired time-domain entry
of `x` times the forward matrix at `t` and the paired packed frequency. -/
theorem rfft_apply_eq_sum (x : Tensor ℝ (s.set dim n))
    (c : Shape.Coord (s.set dim (n / 2 + 1) ++ [2])) :
    rfft dim n x c = ∑ t : Fin n, x (rfftIndexEquiv dim n (c, t)).1 *
      RealFFT.forwardMatrix n (t, (rfftIndexEquiv dim n (c, t)).2, ()) := by
  simp only [rfft, Rep.get_ofFn, List.finRange_foldl_add_eq_add_sum, zero_add]
  rfl

/-- An entry of `rfftBackward dim n g` is the sum over the packed frequency `p` of the paired
spectrum entry of `g` times the forward matrix at the paired time index and `p`. -/
theorem rfftBackward_apply_eq_sum (g : Tensor ℝ (s.set dim (n / 2 + 1) ++ [2]))
    (c : Shape.Coord (s.set dim n)) :
    rfftBackward dim n g c = ∑ p : Fin ((n / 2 + 1) * 2),
      g ((rfftIndexEquiv dim n).symm (c, p)).1 *
        RealFFT.forwardMatrix n (((rfftIndexEquiv dim n).symm (c, p)).2, p, ()) := by
  simp only [rfftBackward, Rep.get_ofFn, List.finRange_foldl_add_eq_add_sum, zero_add]
  rfl

/-- An entry of `irfft dim n x` is the sum over the packed frequency `p` of the paired spectrum
entry of `x` times the inverse matrix at `p` and the paired time index. -/
theorem irfft_apply_eq_sum (x : Tensor ℝ (s.set dim (n / 2 + 1) ++ [2]))
    (c : Shape.Coord (s.set dim n)) :
    irfft dim n x c = ∑ p : Fin ((n / 2 + 1) * 2),
      x ((rfftIndexEquiv dim n).symm (c, p)).1 *
        RealFFT.inverseMatrix n (p, ((rfftIndexEquiv dim n).symm (c, p)).2, ()) := by
  simp only [irfft, Rep.get_ofFn, List.finRange_foldl_add_eq_add_sum, zero_add]
  rfl

/-- An entry of `irfftBackward dim n g` is the sum over the time index `t` of the paired
time-domain entry of `g` times the inverse matrix at the paired packed frequency and `t`. -/
theorem irfftBackward_apply_eq_sum (g : Tensor ℝ (s.set dim n))
    (c : Shape.Coord (s.set dim (n / 2 + 1) ++ [2])) :
    irfftBackward dim n g c = ∑ t : Fin n, g (rfftIndexEquiv dim n (c, t)).1 *
      RealFFT.inverseMatrix n ((rfftIndexEquiv dim n (c, t)).2, t, ()) := by
  simp only [irfftBackward, Rep.get_ofFn, List.finRange_foldl_add_eq_add_sum, zero_add]
  rfl

/-- The real Fourier transform along `dim` of length `n`, as a continuous linear map. -/
noncomputable def rfftCLM :
    Tensor ℝ (s.set dim n) →L[ℝ] Tensor ℝ (s.set dim (n / 2 + 1) ++ [2]) :=
  LinearMap.toContinuousLinearMap
    { toFun := rfft dim n
      map_add' := fun x y => Rep.ext fun c => by
        simp only [rfft_apply_eq_sum, Tensor.add_apply, add_mul, Finset.sum_add_distrib]
      map_smul' := fun a x => Rep.ext fun c => by
        simp only [rfft_apply_eq_sum, Rep.smul_apply, RingHom.id_apply, smul_eq_mul,
          Finset.mul_sum, mul_assoc] }

/-- `rfftCLM dim n` applied to `x` is `rfft dim n x`. -/
@[simp] theorem rfftCLM_apply (x : Tensor ℝ (s.set dim n)) :
    rfftCLM dim n x = rfft dim n x :=
  rfl

/-- The inverse real Fourier transform along `dim` with output length `n`, as a continuous linear
map. -/
noncomputable def irfftCLM :
    Tensor ℝ (s.set dim (n / 2 + 1) ++ [2]) →L[ℝ] Tensor ℝ (s.set dim n) :=
  LinearMap.toContinuousLinearMap
    { toFun := irfft dim n
      map_add' := fun x y => Rep.ext fun c => by
        simp only [irfft_apply_eq_sum, Tensor.add_apply, add_mul, Finset.sum_add_distrib]
      map_smul' := fun a x => Rep.ext fun c => by
        simp only [irfft_apply_eq_sum, Rep.smul_apply, RingHom.id_apply, smul_eq_mul,
          Finset.mul_sum, mul_assoc] }

/-- `irfftCLM dim n` applied to `x` is `irfft dim n x`. -/
@[simp] theorem irfftCLM_apply (x : Tensor ℝ (s.set dim (n / 2 + 1) ++ [2])) :
    irfftCLM dim n x = irfft dim n x :=
  rfl

/-- `rfftBackward dim n` is the transpose of `rfft dim n` for the inner product of tensors. -/
theorem inner_rfft (u : Tensor ℝ (s.set dim n))
    (δ : Tensor ℝ (s.set dim (n / 2 + 1) ++ [2])) :
    ⟪rfft dim n u, δ⟫ = ⟪u, rfftBackward dim n δ⟫ := by
  simp only [inner_eq_sum, rfft_apply_eq_sum, rfftBackward_apply_eq_sum, Finset.sum_mul,
    Finset.mul_sum]
  rw [← Fintype.sum_prod_type', ← Fintype.sum_prod_type']
  refine Fintype.sum_equiv (rfftIndexEquiv dim n) _ _ fun q => ?_
  simp only [Prod.mk.eta, Equiv.symm_apply_apply]
  ring

/-- `irfftBackward dim n` is the transpose of `irfft dim n` for the inner product of tensors. -/
theorem inner_irfft (u : Tensor ℝ (s.set dim (n / 2 + 1) ++ [2]))
    (δ : Tensor ℝ (s.set dim n)) :
    ⟪irfft dim n u, δ⟫ = ⟪u, irfftBackward dim n δ⟫ := by
  simp only [inner_eq_sum, irfft_apply_eq_sum, irfftBackward_apply_eq_sum, Finset.sum_mul,
    Finset.mul_sum]
  rw [← Fintype.sum_prod_type', ← Fintype.sum_prod_type']
  refine Fintype.sum_equiv (rfftIndexEquiv dim n).symm _ _ fun q => ?_
  simp only [Prod.mk.eta, Equiv.apply_symm_apply]
  ring

end Tensor

namespace HasPullbackAt

open Tensor

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] {s : Shape} {x : E}

/-- The real Fourier transform along `dim` of length `n` after `f`, pulled back by
`rfftBackward`. -/
@[fun_prop]
theorem rfft (dim : Fin s.rank) (n : Nat) {f : E → Tensor ℝ (s.set dim n)}
    {Df : E →L[ℝ] Tensor ℝ (s.set dim n)} {bf : Tensor ℝ (s.set dim n) → E}
    (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => Tensor.rfft dim n (f y)) ((rfftCLM dim n).comp Df)
      (fun δ => bf (Tensor.rfftBackward dim n δ)) x := by
  simpa only [rfftCLM_apply] using
    clm (rfftCLM dim n) (Tensor.rfftBackward dim n) (inner_rfft dim n) hf

/-- The inverse real Fourier transform along `dim` with output length `n` after `f`, pulled back
by `irfftBackward`. -/
@[fun_prop]
theorem irfft (dim : Fin s.rank) (n : Nat)
    {f : E → Tensor ℝ (s.set dim (n / 2 + 1) ++ [2])}
    {Df : E →L[ℝ] Tensor ℝ (s.set dim (n / 2 + 1) ++ [2])}
    {bf : Tensor ℝ (s.set dim (n / 2 + 1) ++ [2]) → E}
    (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => Tensor.irfft dim n (f y)) ((irfftCLM dim n).comp Df)
      (fun δ => bf (Tensor.irfftBackward dim n δ)) x := by
  simpa only [irfftCLM_apply] using
    clm (irfftCLM dim n) (Tensor.irfftBackward dim n) (inner_irfft dim n) hf

end HasPullbackAt

namespace Op

open Tensor

/-- The forward of `Op.rfft` is differentiable at every input, and its backward is the adjoint of
its derivative. -/
@[op_cert TorchLean.Op.rfft derivative]
theorem rfft_backward_eq_adjoint {s : Shape} (dim : Fin s.rank) (n : Nat) (hn : 0 < n)
    (x : Tensor ℝ (s.set dim n))
    (δ : Tensor ℝ (s.set dim (n / 2 + 1) ++ [2])) :
    DifferentiableAt ℝ (fun x' =>
        (((Op.rfft dim n hn).rule? (α := ℝ) |>.get rfl).forward (x', ())).1) x ∧
      (((Op.rfft dim n hn).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun x' =>
          (((Op.rfft dim n hn).rule? (α := ℝ) |>.get rfl).forward (x', ())).1) x).adjoint δ := by
  simp only [Op.rule?_rfft, Option.get_some]
  apply HasPullbackAt.differentiableAt_and
  · fun_prop
  · rfl

/-- The forward of `Op.irfft` is differentiable at every input, and its backward is the adjoint of
its derivative. -/
@[op_cert TorchLean.Op.irfft derivative]
theorem irfft_backward_eq_adjoint {s : Shape} (dim : Fin s.rank) (n : Nat) (hn : 0 < n)
    (x : Tensor ℝ (s.set dim (n / 2 + 1) ++ [2]))
    (δ : Tensor ℝ (s.set dim n)) :
    DifferentiableAt ℝ (fun x' =>
        (((Op.irfft dim n hn).rule? (α := ℝ) |>.get rfl).forward (x', ())).1) x ∧
      (((Op.irfft dim n hn).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun x' =>
          (((Op.irfft dim n hn).rule? (α := ℝ) |>.get rfl).forward (x', ())).1) x).adjoint δ := by
  simp only [Op.rule?_irfft, Option.get_some]
  apply HasPullbackAt.differentiableAt_and
  · fun_prop
  · rfl

end Op

end TorchLean
