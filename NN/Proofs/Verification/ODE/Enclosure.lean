/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.Calculus.MeanValue

/-!
# ODE Corridor Enclosures

This file formalizes the real-analysis endpoint used by Tanaka and Yatabe,
*Learn and Verify: A Framework for Rigorous Verification of Physics-Informed Neural Networks*,
arXiv:2601.19818.

The paper trains candidate lower/upper PINN corridors and then verifies, using interval arithmetic,
that they are sub- and super-solutions of a scalar ODE. Once those inequalities are certified, the
mathematical argument is independent of the neural-network producer:

1. **Local corridor enclosure.** A solution of the clamped ODE
   `u' = f(t, clampToCorridor uL uU t (u t))` cannot cross the verified corridor
   `[uL(t), uU(t)]`. Inside the corridor, clamping is a no-op, so the same trajectory solves the
   original ODE `u' = f(t,u)`.
2. **Constant-extension enclosure.** The finite-time corridor can be extended after `T` by holding
   both walls constant, provided the paper's sign/monotonicity hypotheses for `f` hold beyond `T`.

The proof is deliberately stated over exact `ℝ` functions. Floating-point or executable backends
must first justify these real hypotheses; `EnclosureBackends.lean` only reuses this theorem through
backend `toReal` views. The trust boundary is at that `toReal` bridge.
The theorems assume a continuous solution with the stated derivative; they do not construct a
solution or prove existence or uniqueness.
-/

@[expose] public section


namespace NN.Proofs.Verification.ODE.Enclosure

open scoped Topology
open Set

/-! ## Corridor clamping -/

/--
The paper's truncation function `𝒯u(t)` (Eq. 16), named here by what it does:
clamp the current value into the certified corridor `[uL(t), uU(t)]`.

If `u` is below the lower wall we feed the vector field `uL(t)`; if it is above the upper wall we
feed `uU(t)`; otherwise we feed `u` itself. The enclosure theorem then proves the clamped solution
never actually needs those emergency branches.
-/
def clampToCorridor (uL uU : ℝ → ℝ) (t : ℝ) (u : ℝ) : ℝ :=
  max (uL t) (min (uU t) u)

/-- If `u` is already within the corridor, clamping is a no-op. -/
theorem clampToCorridor_eq_self {uL uU : ℝ → ℝ} {t u : ℝ}
    (hL : uL t ≤ u) (hU : u ≤ uU t) :
    clampToCorridor uL uU t u = u := by
  simp [clampToCorridor, min_eq_right hU, max_eq_right hL]

/-- If `u` lies strictly above the corridor, clamping snaps to the upper wall. -/
theorem clampToCorridor_eq_upper {uL uU : ℝ → ℝ} {t u : ℝ}
    (hLU : uL t ≤ uU t) (hU : uU t < u) :
    clampToCorridor uL uU t u = uU t := by
  have hm : min (uU t) u = uU t := min_eq_left (le_of_lt hU)
  have hM : max (uL t) (uU t) = uU t := max_eq_right hLU
  simp [clampToCorridor, hm, hM]

/-- If `u` lies strictly below the corridor, clamping snaps to the lower wall. -/
theorem clampToCorridor_eq_lower {uL uU : ℝ → ℝ} {t u : ℝ}
    (hLU : uL t ≤ uU t) (hL : u < uL t) :
    clampToCorridor uL uU t u = uL t := by
  have hU : u ≤ uU t := le_trans (le_of_lt hL) hLU
  have hm : min (uU t) u = u := min_eq_right hU
  have hM : max (uL t) u = uL t := max_eq_left (le_of_lt hL)
  simp [clampToCorridor, hm, hM]

/-! ## Local corridor enclosure -/

/-!
This is the comparison theorem used in the paper's local enclosure result. It is a
“no first crossing” statement: if `u` evolves according to the clamped ODE
`u' = f(t, clampToCorridor uL uU t (u t))`, and `uL,uU` are a sub- and
super-solution pair for the original ODE, then `u` cannot leave the corridor.

We implement this by reducing to mathlib’s 1D *fencing theorem*
`image_le_of_deriv_right_lt_deriv_boundary'` twice:
1. an upper fence `u ≤ uU + ε(1+t)` for any `ε>0`;
2. a lower fence via the same argument applied to `-u`.
-/

/-- Core local corridor theorem:

If `u` solves the clamped ODE (right-derivative form), and `uL,uU` are sub- and
super-solutions for the original ODE, then `u` is trapped between `uL` and `uU` on `[0,T]`.

The solution and its right derivative are hypotheses; no existence or uniqueness claim is made.
-/
theorem localEnclosure_fromClampedDynamics
    {T : ℝ} {f : ℝ → ℝ → ℝ}
    {u uL uU uL' uU' : ℝ → ℝ} {a : ℝ}
    (hu_cont : ContinuousOn u (Icc 0 T))
    (hu_der :
      ∀ t ∈ Ico 0 T,
        HasDerivWithinAt u (f t (clampToCorridor uL uU t (u t))) (Ici t) t)
    (hu0 : u 0 = a)
    (hL_cont : ContinuousOn uL (Icc 0 T))
    (hL_der : ∀ t ∈ Ico 0 T, HasDerivWithinAt uL (uL' t) (Ici t) t)
    (hL_sub : ∀ t ∈ Ico 0 T, uL' t ≤ f t (uL t))
    (hL0 : uL 0 ≤ a)
    (hU_cont : ContinuousOn uU (Icc 0 T))
    (hU_der : ∀ t ∈ Ico 0 T, HasDerivWithinAt uU (uU' t) (Ici t) t)
    (hU_sup : ∀ t ∈ Ico 0 T, f t (uU t) ≤ uU' t)
    (hU0 : a ≤ uU 0)
    (hLU : ∀ t ∈ Icc 0 T, uL t ≤ uU t) :
    ∀ t ∈ Icc 0 T, uL t ≤ u t ∧ u t ≤ uU t := by
  -- Upper enclosure: `u ≤ uU + ε(1+t)` for all ε>0.
  have upper_eps :
      ∀ ε : ℝ, 0 < ε → ∀ t ∈ Icc 0 T, u t ≤ uU t + ε * (1 + t) := by
    intro ε hε
    let B : ℝ → ℝ := fun t => uU t + ε * (1 + t)
    let B' : ℝ → ℝ := fun t => uU' t + ε
    have hB_cont : ContinuousOn B (Icc 0 T) :=
      hU_cont.add (continuousOn_const.mul (continuousOn_const.add continuousOn_id))
    have hB_der : ∀ t ∈ Ico 0 T, HasDerivWithinAt B (B' t) (Ici t) t := by
      intro t ht
      have : HasDerivWithinAt (fun t => ε * (1 + t)) ε (Ici t) t := by
        simpa using ((hasDerivWithinAt_id t (Ici t)).const_add 1).const_mul ε
      change HasDerivWithinAt (uU + fun t : ℝ => ε * (1 + t)) (B' t) (Ici t) t
      simpa [B', add_assoc, add_left_comm, add_comm] using (hU_der t ht).add this
    have h0 : u 0 ≤ B 0 := by
      have : u 0 ≤ uU 0 := by simpa [hu0] using hU0
      have : u 0 ≤ uU 0 + ε := le_trans this (by linarith [hε])
      simpa [B] using this
    have bound :
        ∀ t ∈ Ico 0 T, u t = B t →
          f t (clampToCorridor uL uU t (u t)) < B' t := by
      intro t ht htEq
      have ht1 : 0 < 1 + t := by linarith [ht.1]
      have hLUt : uL t ≤ uU t := hLU t ⟨ht.1, le_of_lt ht.2⟩
      have hUlt : uU t < u t := by
        have hpos : 0 < ε * (1 + t) := mul_pos hε ht1
        have : uU t < B t := by simpa [B] using (lt_add_of_pos_right (uU t) hpos)
        simpa [htEq] using this
      -- The clamp snaps to the upper wall exactly at a hypothetical first upper crossing.
      have htr : clampToCorridor uL uU t (u t) = uU t :=
        clampToCorridor_eq_upper (t := t) (u := u t) hLUt hUlt
      have : f t (clampToCorridor uL uU t (u t)) ≤ uU' t := by
        simpa [htr] using hU_sup t ht
      have : f t (clampToCorridor uL uU t (u t)) < uU' t + ε :=
        lt_of_le_of_lt this (by linarith [hε])
      simpa [B'] using this
    have hu_le : ∀ t ∈ Icc 0 T, u t ≤ B t :=
      image_le_of_deriv_right_lt_deriv_boundary'
        (f := u) (f' := fun t => f t (clampToCorridor uL uU t (u t)))
        (a := (0 : ℝ)) (b := T)
        hu_cont hu_der h0 hB_cont hB_der bound
    intro t ht
    simpa [B, mul_assoc, add_assoc, add_comm, add_left_comm] using hu_le t ht

  -- Lower enclosure: `uL - ε(1+t) ≤ u` for all ε>0 (proved as `-u ≤ -uL + ε(1+t)`).
  have lower_eps :
      ∀ ε : ℝ, 0 < ε → ∀ t ∈ Icc 0 T, uL t - ε * (1 + t) ≤ u t := by
    intro ε hε
    let F : ℝ → ℝ := fun t => -u t
    let F' : ℝ → ℝ := fun t => -(f t (clampToCorridor uL uU t (u t)))
    let B : ℝ → ℝ := fun t => -uL t + ε * (1 + t)
    let B' : ℝ → ℝ := fun t => ε + (-uL' t)
    have hF_cont : ContinuousOn F (Icc 0 T) := hu_cont.neg
    have hF_der : ∀ t ∈ Ico 0 T, HasDerivWithinAt F (F' t) (Ici t) t := by
      intro t ht
      change HasDerivWithinAt (-u) (F' t) (Ici t) t
      simpa [F'] using (hu_der t ht).neg
    have hB_cont : ContinuousOn B (Icc 0 T) :=
      hL_cont.neg.add (continuousOn_const.mul (continuousOn_const.add continuousOn_id))
    have hB_der : ∀ t ∈ Ico 0 T, HasDerivWithinAt B (B' t) (Ici t) t := by
      intro t ht
      have : HasDerivWithinAt (fun t => ε * (1 + t)) ε (Ici t) t := by
        simpa using ((hasDerivWithinAt_id t (Ici t)).const_add 1).const_mul ε
      have hsum :
          HasDerivWithinAt (fun x => ε * (1 + x) + (-uL x)) (ε + (-uL' t)) (Ici t) t := by
        change HasDerivWithinAt ((fun x : ℝ => ε * (1 + x)) + -uL)
          (ε + (-uL' t)) (Ici t) t
        exact this.add (hL_der t ht).neg
      exact hsum.congr_of_mem (s := Ici t)
        (fun x _ => by simp [B, add_comm]) (Set.self_mem_Ici)
    have h0 : F 0 ≤ B 0 := by
      have : uL 0 ≤ u 0 := by simpa [hu0] using hL0
      have : -u 0 ≤ -uL 0 + ε := by
        have : -u 0 ≤ -uL 0 := neg_le_neg this
        linarith [this, hε]
      simpa [F, B, add_assoc, add_left_comm, add_comm] using this
    have bound :
        ∀ t ∈ Ico 0 T, F t = B t → F' t < B' t := by
      intro t ht htEq
      have ht1 : 0 < 1 + t := by linarith [ht.1]
      have hLUt : uL t ≤ uU t := hLU t ⟨ht.1, le_of_lt ht.2⟩
      have hEqU : u t = uL t - ε * (1 + t) := by
        linarith [htEq]
      have hLgt : u t < uL t := by linarith [mul_pos hε ht1]
      have htr : clampToCorridor uL uU t (u t) = uL t :=
        clampToCorridor_eq_lower (t := t) (u := u t) hLUt hLgt
      have huL' : uL' t ≤ f t (clampToCorridor uL uU t (u t)) := by
        simpa [htr] using hL_sub t ht
      have : -(f t (clampToCorridor uL uU t (u t))) ≤ -uL' t := neg_le_neg huL'
      have : -(f t (clampToCorridor uL uU t (u t))) < ε + (-uL' t) :=
        lt_of_le_of_lt this (by linarith [hε])
      simpa [F', B', add_assoc, add_comm, add_left_comm] using this
    have hF_le : ∀ t ∈ Icc 0 T, F t ≤ B t :=
      image_le_of_deriv_right_lt_deriv_boundary'
        (f := F) (f' := F') (a := (0 : ℝ)) (b := T)
        hF_cont hF_der h0 hB_cont hB_der bound
    intro t ht
    have := hF_le t ht
    have : uL t - ε * (1 + t) ≤ u t := by
      linarith [this]
    simpa using this

  -- Let `ε ↓ 0` in the two fences: since `1 + t > 0`, the slack `ε * (1 + t)` sweeps every
  -- positive number.
  intro t ht
  have ht1 : 0 < 1 + t := by linarith [ht.1]
  have hu_le_uU : u t ≤ uU t := by
    refine le_of_forall_pos_le_add fun ε hε => ?_
    have h := upper_eps (ε / (1 + t)) (div_pos hε ht1) t ht
    rwa [div_mul_cancel₀ _ ht1.ne'] at h
  have hu_ge_uL : uL t ≤ u t := by
    refine le_of_forall_pos_le_add fun ε hε => ?_
    have h := lower_eps (ε / (1 + t)) (div_pos hε ht1) t ht
    rw [div_mul_cancel₀ _ ht1.ne'] at h
    linarith
  exact ⟨hu_ge_uL, hu_le_uU⟩

/--
Local unclamping theorem.

Once the local enclosure proves `uL(t) ≤ u(t) ≤ uU(t)`, `clampToCorridor_eq_self` gives
equality of the clamped value and `u(t)`.
A solution of the clamped ODE is therefore enclosed and solves the original ODE on the same
interval. This is the exact real-analysis contract that interval/PINN certificate producers
must establish before TorchLean can claim a verified ODE solve.
-/
theorem localSolutionEnclosed_fromClampedDynamics
    {T : ℝ} {f : ℝ → ℝ → ℝ}
    {u uL uU uL' uU' : ℝ → ℝ} {a : ℝ}
    (hu_cont : ContinuousOn u (Icc 0 T))
    (hu_der :
      ∀ t ∈ Ico 0 T,
        HasDerivWithinAt u (f t (clampToCorridor uL uU t (u t))) (Ici t) t)
    (hu0 : u 0 = a)
    (hL_cont : ContinuousOn uL (Icc 0 T))
    (hL_der : ∀ t ∈ Ico 0 T, HasDerivWithinAt uL (uL' t) (Ici t) t)
    (hL_sub : ∀ t ∈ Ico 0 T, uL' t ≤ f t (uL t))
    (hL0 : uL 0 ≤ a)
    (hU_cont : ContinuousOn uU (Icc 0 T))
    (hU_der : ∀ t ∈ Ico 0 T, HasDerivWithinAt uU (uU' t) (Ici t) t)
    (hU_sup : ∀ t ∈ Ico 0 T, f t (uU t) ≤ uU' t)
    (hU0 : a ≤ uU 0)
    (hLU : ∀ t ∈ Icc 0 T, uL t ≤ uU t) :
    (∀ t ∈ Icc 0 T, uL t ≤ u t ∧ u t ≤ uU t) ∧
      (∀ t ∈ Ico 0 T, HasDerivWithinAt u (f t (u t)) (Ici t) t) := by
  have hEnc :=
    localEnclosure_fromClampedDynamics (T := T) (f := f)
      (u := u) (uL := uL) (uU := uU) (uL' := uL') (uU' := uU') (a := a)
      hu_cont hu_der hu0 hL_cont hL_der hL_sub hL0 hU_cont hU_der hU_sup hU0 hLU
  refine ⟨hEnc, ?_⟩
  intro t ht
  have hLUt : uL t ≤ uU t := hLU t ⟨ht.1, le_of_lt ht.2⟩
  have hLt : uL t ≤ u t := (hEnc t ⟨ht.1, le_of_lt ht.2⟩).1
  have hUt : u t ≤ uU t := (hEnc t ⟨ht.1, le_of_lt ht.2⟩).2
  have htr : clampToCorridor uL uU t (u t) = u t :=
    clampToCorridor_eq_self (t := t) (u := u t) hLt hUt
  simpa [htr] using hu_der t ht

/-! ## Constant-extension corridor -/

/--
Constant extension used by the paper's global-time argument: follow `g` until the verified
time horizon `T`, then freeze it at `g T`.

The function is intentionally right-derivative friendly. At and after `T`, viewed from the
right-neighborhood `Ici t`, the extension is locally constant and has derivative zero.
-/
noncomputable def constantExtensionAfter (T : ℝ) (g : ℝ → ℝ) : ℝ → ℝ :=
  fun t => if t ≤ T then g t else g T

/-- On the left side of the switching time (`t ≤ T`), the extension agrees with `g`. -/
@[simp] theorem constantExtensionAfter_of_le {T : ℝ} {g : ℝ → ℝ} {t : ℝ} (ht : t ≤ T) :
    constantExtensionAfter T g t = g t := by simp [constantExtensionAfter, ht]

/-- On the right side of the switching time (`T < t`), the extension is constant `g T`. -/
@[simp] theorem constantExtensionAfter_of_gt {T : ℝ} {g : ℝ → ℝ} {t : ℝ} (ht : T < t) :
    constantExtensionAfter T g t = g T := by simp [constantExtensionAfter, not_le_of_gt ht]

/-- Derivative of `constantExtensionAfter T g` strictly before `T` matches the derivative of `g`. -/
private theorem hasDerivWithinAt_constantExtensionAfter_before
    {T : ℝ} {g g' : ℝ → ℝ} {t : ℝ} (ht : t < T)
    (hg : HasDerivWithinAt g (g' t) (Ici t) t) :
    HasDerivWithinAt (constantExtensionAfter T g) (g' t) (Ici t) t := by
  have hIio : (Iio T) ∈ 𝓝[Ici t] t := by
    refine (mem_nhdsWithin.2 ?_)
    refine ⟨Iio T, isOpen_Iio, ht, ?_⟩
    intro x hx
    exact hx.1
  have hEq : (constantExtensionAfter T g) =ᶠ[𝓝[Ici t] t] g := by
    refine Filter.eventuallyEq_of_mem hIio ?_
    intro x hx
    have : x ≤ T := le_of_lt hx
    simp [constantExtensionAfter, this]
  exact hg.congr_of_eventuallyEq hEq (by simp [constantExtensionAfter, le_of_lt ht])

/-- Derivative of `constantExtensionAfter T g` at/after `T` is zero in the right-derivative view. -/
private theorem hasDerivWithinAt_constantExtensionAfter_after
    {T : ℝ} {g : ℝ → ℝ} {t : ℝ} (ht : T ≤ t) :
    HasDerivWithinAt (constantExtensionAfter T g) 0 (Ici t) t := by
  have hEq : ∀ x ∈ Ici t, constantExtensionAfter T g x = g T := by
    intro x hx
    have : T < x ∨ T = x := lt_or_eq_of_le (le_trans ht hx)
    cases this with
    | inl hlt =>
        simp [constantExtensionAfter, not_le_of_gt hlt]
    | inr heq =>
        subst heq
        simp [constantExtensionAfter]
  exact (hasDerivWithinAt_const (c := g T) (s := Ici t) (x := t)).congr_of_mem
    (fun x hx => by simp [hEq x hx]) (by simp)

/-- Freezing a function that is continuous on `[0, T]` after time `T` keeps it continuous on any
`[0, τ]`: the two pieces agree at the switching time `T`. -/
private theorem continuousOn_constantExtensionAfter {T τ : ℝ} {g : ℝ → ℝ}
    (hg : ContinuousOn g (Icc 0 T)) :
    ContinuousOn (constantExtensionAfter T g) (Icc 0 τ) := by
  classical
  have h1 : ContinuousOn g (Icc 0 τ ∩ closure {t : ℝ | t ≤ T}) := by
    refine hg.mono fun t ht => ⟨ht.1.1, ?_⟩
    have htcl : t ∈ closure (Iic T) := by simpa [Set.Iic] using ht.2
    have htIic : t ∈ Iic T := by simpa [closure_Iic] using htcl
    simpa [Set.Iic] using htIic
  have hp : ∀ a ∈ Icc 0 τ ∩ frontier {t : ℝ | t ≤ T}, g a = (fun _ => g T) a := by
    intro a ha
    have haFront : a ∈ frontier (Iic T) := by simpa [Set.Iic] using ha.2
    have haT : a ∈ ({T} : Set ℝ) := (frontier_Iic_subset (α := ℝ) T) haFront
    have : a = T := by simpa using haT
    simp [this]
  change ContinuousOn (fun a : ℝ => if a ≤ T then g a else g T) (Icc 0 τ)
  exact ContinuousOn.if (s := Icc 0 τ) (p := fun t : ℝ => t ≤ T) (f := g)
    (g := fun _ => g T) hp h1 continuousOn_const

/-- Constant-extension enclosure theorem:

Assume we have `uL,uU` on `[0,T]` satisfying the local corridor hypotheses, and assume the paper's
extra sign/monotonicity conditions for `f` beyond `T`. Then for any horizon `τ` (the case of
interest being `τ ≥ T`; for `τ ≤ T` the statement reduces to the local one), any solution `u` of
the clamped ODE built from the constant extensions is enclosed on `[0,τ]` and is a genuine solution
of `u' = f(t,u)` on `[0,τ]`.

This is the reusable Lean form of the paper's global-in-time step: after the verified horizon, the
walls stop moving, and the vector field points inward at those frozen walls.
-/
theorem extendedSolutionEnclosed_fromClampedDynamics
    {T τ : ℝ} (hT : 0 ≤ T) {f : ℝ → ℝ → ℝ}
    {u uL uU uL' uU' : ℝ → ℝ} {a : ℝ}
    (hu_cont : ContinuousOn u (Icc 0 τ))
    (hu_der : ∀ t ∈ Ico 0 τ,
      HasDerivWithinAt u
        (f t (clampToCorridor
          (constantExtensionAfter T uL) (constantExtensionAfter T uU) t (u t))) (Ici t) t)
    (hu0 : u 0 = a)
    (hL_cont : ContinuousOn uL (Icc 0 T))
    (hL_der : ∀ t ∈ Ico 0 T, HasDerivWithinAt uL (uL' t) (Ici t) t)
    (hL_sub : ∀ t ∈ Ico 0 T, uL' t ≤ f t (uL t))
    (hL0 : uL 0 ≤ a)
    (hU_cont : ContinuousOn uU (Icc 0 T))
    (hU_der : ∀ t ∈ Ico 0 T, HasDerivWithinAt uU (uU' t) (Ici t) t)
    (hU_sup : ∀ t ∈ Ico 0 T, f t (uU t) ≤ uU' t)
    (hU0 : a ≤ uU 0)
    (hLU : ∀ t ∈ Icc 0 T, uL t ≤ uU t)
    -- Paper’s extra assumptions beyond `T`, phrased for the constant extension.
    (hLower : ∀ t, T < t → 0 ≤ f T (uL T) ∧ f T (uL T) ≤ f t (uL T))
    (hUpper : ∀ t, T < t → f t (uU T) ≤ f T (uU T) ∧ f T (uU T) ≤ 0) :
    (∀ t ∈ Icc 0 τ,
      constantExtensionAfter T uL t ≤ u t ∧ u t ≤ constantExtensionAfter T uU t) ∧
      (∀ t ∈ Ico 0 τ, HasDerivWithinAt u (f t (u t)) (Ici t) t) := by
  let uLext : ℝ → ℝ := constantExtensionAfter T uL
  let uUext : ℝ → ℝ := constantExtensionAfter T uU
  let uLext' : ℝ → ℝ := fun t => if t < T then uL' t else 0
  let uUext' : ℝ → ℝ := fun t => if t < T then uU' t else 0

  have hLext_cont : ContinuousOn uLext (Icc 0 τ) :=
    continuousOn_constantExtensionAfter (τ := τ) hL_cont
  have hUext_cont : ContinuousOn uUext (Icc 0 τ) :=
    continuousOn_constantExtensionAfter (τ := τ) hU_cont

  have hLext_der : ∀ t ∈ Ico 0 τ, HasDerivWithinAt uLext (uLext' t) (Ici t) t := by
    intro t ht
    by_cases htT : t < T
    · have htIco : t ∈ Ico 0 T := ⟨ht.1, htT⟩
      have := hasDerivWithinAt_constantExtensionAfter_before
        (T := T) (g := uL) (g' := uL') htT (hL_der t htIco)
      simpa [uLext, uLext', ite_eq_left htT] using this
    · have htge : T ≤ t := le_of_not_gt htT
      have := hasDerivWithinAt_constantExtensionAfter_after (T := T) (g := uL) (t := t) htge
      simpa [uLext, uLext', ite_eq_right htT] using this

  have hUext_der : ∀ t ∈ Ico 0 τ, HasDerivWithinAt uUext (uUext' t) (Ici t) t := by
    intro t ht
    by_cases htT : t < T
    · have htIco : t ∈ Ico 0 T := ⟨ht.1, htT⟩
      have := hasDerivWithinAt_constantExtensionAfter_before
        (T := T) (g := uU) (g' := uU') htT (hU_der t htIco)
      simpa [uUext, uUext', ite_eq_left htT] using this
    · have htge : T ≤ t := le_of_not_gt htT
      have := hasDerivWithinAt_constantExtensionAfter_after (T := T) (g := uU) (t := t) htge
      simpa [uUext, uUext', ite_eq_right htT] using this

  have hLext_sub : ∀ t ∈ Ico 0 τ, uLext' t ≤ f t (uLext t) := by
    intro t ht
    by_cases htT : t < T
    · have htIco : t ∈ Ico 0 T := ⟨ht.1, htT⟩
      simpa [uLext, uLext', constantExtensionAfter, le_of_lt htT, ite_eq_left htT]
        using hL_sub t htIco
    · have htgt : T < t ∨ T = t := lt_or_eq_of_le (le_of_not_gt htT)
      have hnonneg : 0 ≤ f t (uL T) := by
        cases htgt with
        | inl hlt =>
            exact (hLower t hlt).1.trans (hLower t hlt).2
        | inr heq =>
            -- Use `0 ≤ f(T,uL(T))` from any `t'>T` instance.
            have hTnonneg : 0 ≤ f T (uL T) := (hLower (T + 1) (by linarith)).1
            simpa [heq] using hTnonneg
      have huLext : uLext t = uL T := by
        by_cases htle : t ≤ T
        · have : t = T := le_antisymm htle (le_of_not_gt htT)
          simp [uLext, this]
        · simp [uLext, constantExtensionAfter, htle]
      simpa [uLext, uLext', ite_eq_right htT, huLext] using hnonneg

  have hUext_sup : ∀ t ∈ Ico 0 τ, f t (uUext t) ≤ uUext' t := by
    intro t ht
    by_cases htT : t < T
    · have htIco : t ∈ Ico 0 T := ⟨ht.1, htT⟩
      simpa [uUext, uUext', constantExtensionAfter, le_of_lt htT, ite_eq_left htT]
        using hU_sup t htIco
    · have htge : T ≤ t := le_of_not_gt htT
      have htgt : T < t ∨ T = t := lt_or_eq_of_le htge
      have hnonpos : f t (uU T) ≤ 0 := by
        cases htgt with
        | inl hlt =>
            exact le_trans (hUpper t hlt).1 (hUpper t hlt).2
        | inr heq =>
            have hTnonpos : f T (uU T) ≤ 0 := (hUpper (T + 1) (by linarith)).2
            simpa [heq] using hTnonpos
      have huUext : uUext t = uU T := by
        by_cases htle : t ≤ T
        · have : t = T := le_antisymm htle htge
          simp [uUext, this]
        · simp [uUext, constantExtensionAfter, htle]
      simpa [uUext, uUext', ite_eq_right htT, huUext] using hnonpos

  have hLext0 : uLext 0 ≤ a := by simpa [uLext, constantExtensionAfter, hT] using hL0
  have hUext0 : a ≤ uUext 0 := by simpa [uUext, constantExtensionAfter, hT] using hU0
  have hLUext : ∀ t ∈ Icc 0 τ, uLext t ≤ uUext t := by
    intro t ht
    by_cases htT : t ≤ T
    · have : t ∈ Icc 0 T := ⟨ht.1, htT⟩
      simpa [uLext, uUext, constantExtensionAfter, htT] using hLU t this
    · -- For `t>T`, both are constants.
      have : uL T ≤ uU T := hLU T ⟨hT, le_rfl⟩
      simp [uLext, uUext, constantExtensionAfter, htT, this]

  exact localSolutionEnclosed_fromClampedDynamics (T := τ) (f := f)
    (u := u) (uL := uLext) (uU := uUext) (uL' := uLext') (uU' := uUext') (a := a)
    hu_cont hu_der hu0
    hLext_cont hLext_der hLext_sub hLext0
    hUext_cont hUext_der hUext_sup hUext0 hLUext

end NN.Proofs.Verification.ODE.Enclosure
