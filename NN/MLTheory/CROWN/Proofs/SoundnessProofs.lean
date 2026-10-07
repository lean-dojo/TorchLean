/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Soundness proofs for CROWN affine bounds and LiRPA verification.
-/

module

public import NN.MLTheory.CROWN.Models.Mlp

/-!
# Soundness of affine relaxations (CROWN / LiRPA)

This file proves basic inequalities and compositional lemmas used by affine-relaxation bound
propagation (LiRPA) methods, in particular the CROWN/DeepPoly family.

This file records the scalar inequalities behind the relaxations:

- soundness of affine images of scalar intervals (`affine_scalar_interval_sound`),
- soundness of the ReLU triangular upper relaxation (`relu_affine_upper_bound_sound`).

For the two-layer MLP's interval bounds, `NN.MLTheory.CROWN.Theorems.bound_ibp_sound`
in `NN.MLTheory.CROWN.Models.Mlp` proves that `boundIbp` encloses every output from the input box.
That theorem concerns IBP, not the separate affine-CROWN calculation.

For certificate-checking theorems over the graph dialect (the form used by TorchLean verification
examples), see:

- `NN.MLTheory.CROWN.Proofs.GraphCrownCertSoundness`
- `NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness`

## References

- CROWN: Zhang et al., *Efficient Neural Network Robustness Certification with General Activation
  Functions*, NeurIPS 2018. (arXiv:1811.00866)
- β-CROWN / α/β-CROWN: Wang et al., *Beta-CROWN: Efficient Bound Propagation with Provable
  Guarantees*, NeurIPS 2021. (arXiv:2103.06624)
- `auto_LiRPA`: https://github.com/Verified-Intelligence/auto_LiRPA
- `alpha-beta-CROWN`: https://github.com/Verified-Intelligence/alpha-beta-CROWN
-/

@[expose] public section


namespace NN.MLTheory.CROWN.Soundness

/--
If $x\in[\ell,h]$, then the affine function $f(x)=ax+b$ satisfies

$$
f(x)\in[\min(a\ell+b,ah+b),\max(a\ell+b,ah+b)].
$$
-/
theorem affine_scalar_interval_sound (a b x lo hi : ℝ)
    (hx : lo ≤ x ∧ x ≤ hi) :
    min (a * lo + b) (a * hi + b) ≤ a * x + b ∧
    a * x + b ≤ max (a * lo + b) (a * hi + b) := by
  by_cases ha : 0 ≤ a
  · exact ⟨(min_le_left _ _).trans
      (add_le_add (mul_le_mul_of_nonneg_left hx.1 ha) le_rfl),
      (add_le_add (mul_le_mul_of_nonneg_left hx.2 ha) le_rfl).trans
        (le_max_right _ _)⟩
  · have ha' : a ≤ 0 := le_of_not_ge ha
    exact ⟨(min_le_right _ _).trans
      (add_le_add (mul_le_mul_of_nonpos_left hx.2 ha') le_rfl),
      (add_le_add (mul_le_mul_of_nonpos_left hx.1 ha') le_rfl).trans
        (le_max_left _ _)⟩

/--
The ReLU upper affine bound is sound. For $\operatorname{ReLU}(z)$ with $z\in[l,u]$,
the triangular relaxation provides an upper bound.

The CROWN relaxation uses:

- If $l\ge 0$: slope $=1$, bias $=0$; ReLU is the identity.
- If $u\le 0$: slope $=0$, bias $=0$; ReLU is zero.
- If $l<0<u$: slope $=\frac{u}{u-l}$ and bias $=-\frac{lu}{u-l}$, the linear upper envelope.
-/
theorem relu_affine_upper_bound_sound (z l u : ℝ) (hz : l ≤ z ∧ z ≤ u) :
    let relu_z := max 0 z
    let slope := if l ≥ 0 then 1 else if u ≤ 0 then 0 else u / (u - l)
    let bias := if l ≥ 0 then 0 else if u ≤ 0 then 0 else -l * u / (u - l)
    relu_z ≤ slope * z + bias := by
  by_cases hl : 0 ≤ l
  · simp [hl, max_eq_right (hl.trans hz.1)]
  · by_cases hu : u ≤ 0
    · simp [hl, hu, max_eq_left (hz.2.trans hu)]
    · simp only [hl, hu, ite_false]
      have hu' : 0 < u := lt_of_not_ge hu
      have hl' : ¬0 < l := not_lt.mpr (le_of_not_ge hl)
      have hbound := Proofs.relu_relax_scalar_upper_real_runtime l u z hz.1 hz.2
      simp only [Runtime.Ops.ReLU.relaxScalar, hu', hl', ite_true, ite_false,
        Activation.Math.reluSpec_eq_max] at hbound
      have hbias : -(u / (u - l)) * l = -l * u / (u - l) := by ring
      rw [max_comm, hbias] at hbound
      exact hbound

end NN.MLTheory.CROWN.Soundness
