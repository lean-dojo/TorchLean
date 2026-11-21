/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Dual.Jet
public import TorchLean.Tensor.Special.Fn

/-!
# Forward-mode special functions

A scalar type with the special functions and their partial derivatives of every order
(`TorchLean.Special`) gives its dual numbers them too. The derivative of orders `k` of a special
function at dual numbers has the derivative of orders `k` at the primals as primal, and as tangent
the sum over the arguments `i` of the derivative of orders `k + Pi.single i 1` at the primals
times the tangent of argument `i`, added from the left (`Special.leftSum`): the chain rule, applied
to the derivative of orders `k`. Order `0` is the function, whose tangent is the sum of its partial
slopes times the input tangents, as for `erf` in one argument. Because the slope of a dual number
is itself a dual number whose tangent carries the next derivatives, nested dual numbers
(`Dual.Nested`) compute every higher derivative. Over the reals the tangent is the derivative of
the order-`k` derivative along the input tangents wherever the function is smooth, where its
partial derivatives commute (`SpecialFn.hasFDerivAt_deriv`), and at order `0` wherever it is
differentiable (`SpecialFn.hasFDerivAt_real`).

For a special function of one argument, `jet_special_comp` proves this over the reals at every
nesting depth: where `f` is smooth at `g x` and `g` is `C^n` at `x`, the order-`k` derivative of `f`
maps the jet of `g` to the jet of `f⁽ᵏ⁾ ∘ g`. With `k = 0`, nested duals compute the exact mixed
derivatives of `f ∘ g`, and seeding a real input (`jet_special`) gives the jet of `f` itself.
`SpecialFn.contDiffAt_real₁` supplies the smoothness on `SpecialFn.smoothDomain` for every member.
-/

public section

open scoped ContDiff Topology

namespace TorchLean.Dual

section Instances

variable {α : Type} [Add α] [Mul α] [Zero α] [Special α]

/-- The partial derivative of orders `k` of each special function, with tangent the sum over the
arguments `i` of the derivative of orders `k + Pi.single i 1` at the primals times the tangent of
argument `i`. -/
instance instSpecial : Special (Dual α) where
  deriv f k x := ⟨Special.deriv f k fun i => (x i).re,
    Special.leftSum fun i => Special.deriv f (k + Pi.single i 1) (fun j => (x j).re) * (x i).du⟩

/-- The primal of a special function of dual numbers is the function at the primals. -/
@[simp] theorem deriv_re {n : ℕ} (f : SpecialFn n) (k : Fin n → ℕ) (x : Fin n → Dual α) :
    (Special.deriv f k x).re = Special.deriv f k fun i => (x i).re :=
  rfl

/-- The tangent of a special function of dual numbers is the sum of its next derivatives in each
argument at the primals times that argument's tangent. -/
@[simp] theorem deriv_du {n : ℕ} (f : SpecialFn n) (k : Fin n → ℕ) (x : Fin n → Dual α) :
    (Special.deriv f k x).du = Special.leftSum fun i =>
      Special.deriv f (k + Pi.single i 1) (fun j => (x j).re) * (x i).du :=
  rfl

end Instances

/-- The special functions at every nesting depth, with the scalar's at depth zero. -/
instance instSpecialNested {α : Type} [Storage α] [Context α] [Special α] (n : Nat) :
    Special (Nested α n) :=
  match n with
  | 0 => inferInstanceAs (Special α)
  | n + 1 => @Dual.instSpecial (Nested α n) _ _ _ (instSpecialNested n)

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- If the real special function `f` of one argument is smooth at `g x` and `g` is `C^n` at `x`,
its derivative of orders `k` maps the jet of `g` to the jet of `f⁽ᵏ⁾ ∘ g`. -/
theorem jet_special_comp {n : Nat} (directions : Fin n → E) (f : SpecialFn 1) (k : Fin 1 → ℕ)
    {g : E → ℝ} {x : E} (hg : ContDiffAt ℝ n g x) (hf : ContDiffAt ℝ ∞ f.real₁ (g x)) :
    jet directions (fun y => Special.deriv f k fun _ => g y) x =
      Special.deriv f k fun _ => jet directions g x := by
  induction n generalizing k g with
  | zero => rfl
  | succ n ih =>
    let v := directions (Fin.last n)
    have hgn : ContDiffAt ℝ n g x := hg.of_le (by simp)
    have hg' : ContDiffAt ℝ n (fun y => fderiv ℝ g y v) x :=
      (hg.fderiv_right (by simp)).clm_apply contDiffAt_const
    have hm : ContDiffAt ℝ 1 (deriv^[k 0] f.real₁) (g x) :=
      (contDiffAt_iterate_deriv hf (k 0)).of_le (by simp)
    have heq : (fun y => fderiv ℝ (fun z => Special.deriv f k fun _ => g z) y v) =ᶠ[𝓝 x]
        fun y => (Special.deriv f (k + Pi.single 0 1) fun _ => g y) * fderiv ℝ g y v := by
      filter_upwards [hg.eventually (by simp),
        hg.continuousAt.tendsto.eventually (hm.eventually (by simp))] with y hgy hmy
      have hd := (SpecialFn.hasDerivAt_deriv f
        (hmy.differentiableAt one_ne_zero)).comp_hasFDerivAt y
          (hgy.differentiableAt (by simp)).hasFDerivAt
      change fderiv ℝ ((fun t => Special.deriv f k fun _ => t) ∘ g) y v = _
      rw [hd.fderiv]
      rfl
    have hm' : ContDiffAt ℝ n (fun y => Special.deriv f (k + Pi.single 0 1) fun _ => g y) x := by
      have h := ((contDiffAt_iterate_deriv hf (k 0 + 1)).of_le (by simp)).comp x hgn
      refine h.congr_of_eventuallyEq (Filter.Eventually.of_forall fun y => ?_)
      simp [SpecialFn.deriv_eq_iterate]
    change Dual.mk _ _ = Dual.mk _ _
    congr 1
    · exact ih _ k hgn hf
    · rw [jet_congr _ heq, jet_mul _ hm' hg', ih _ (k + Pi.single 0 1) hgn hf]
      rfl

/-- Where the real special function `f` of one argument is smooth, applying it to a seeded nested
dual number computes its jet. -/
theorem jet_special {n : Nat} (directions : Fin n → ℝ) (f : SpecialFn 1) {x : ℝ}
    (hf : ContDiffAt ℝ ∞ f.real₁ x) :
    jet directions (fun t => Special.deriv f 0 fun _ => t) x =
      Special.deriv f 0 fun _ => Nested.seed directions x := by
  have h := jet_special_comp directions f 0 (g := fun y => y) contDiffAt_id hf
  rw [jet_id] at h
  exact h

end TorchLean.Dual
