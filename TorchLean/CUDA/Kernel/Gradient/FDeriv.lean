/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.CUDA.Kernel.Gradient
public import TorchLean.CUDA.Kernel.Tangent.FDeriv
public import Mathlib.Analysis.Calculus.FDeriv.WithLp
public import Mathlib.Analysis.InnerProductSpace.Adjoint

/-!
# Gradient kernels compute adjoint derivatives

Over the reals, the gradient kernel `Expr.gradient k j n e` computes the adjoint of the
derivative of the outputs of `e` in input `j`. We move the first `m` entries of input `j` with a
vector `θ` of `EuclideanSpace ℝ (Fin m)` and keep the other inputs `x` fixed (`withInput x j θ`),
and we collect the `n` outputs in a vector `F θ` of `EuclideanSpace ℝ (Fin n)`. Suppose every
output evaluation at `θ₀` is regular, and the gradient kernel reads the inputs at `θ₀` and the
cotangent `cot` as operand `k`. Then `F` is differentiable at `θ₀`, and entry `a` of the gradient
kernel is entry `a` of the adjoint of `fderiv ℝ F θ₀` applied to `cot`
(`Expr.gradient_eq_adjoint`).

When we move every input `o < k` at once, its first `m o` entries with block `o` of a vector `θ`
of `EuclideanSpace ℝ (Σ o : Fin k, Fin (m o).toNat)` (`withInputs x θ`), the same hypotheses show
that `F` is differentiable at `θ₀` in all moved entries together, and that the gradient kernel of
each input `j < k` computes block `j` of the adjoint of `fderiv ℝ F θ₀` applied to `cot`
(`Expr.gradient_eq_adjoint_withInputs`).

The proof applies `Expr.hasFDerivAt_evaluate_tangent` to each output, with the coordinate
projections as the derivatives of the moved entries. The seeded tangent of output `i` at address
`a` is the derivative of output `i` applied to the basis vector of entry `a`, and the fold of the
gradient kernel is the inner product with the cotangent that gives the adjoint.
-/

@[expose] public section

open Filter hiding pure_bind
open Topology

namespace TorchLean.Kernel

/-- The inputs `x` with the first `m` entries of input `j` replaced by the vector `θ`. -/
def withInput (x : Nat → UInt64 → ℝ) (j : Nat) {m : UInt64}
    (θ : EuclideanSpace ℝ (Fin m.toNat)) : Nat → UInt64 → ℝ :=
  fun o i => if h : o = j ∧ i.toNat < m.toNat then θ ⟨i.toNat, h.2⟩ else x o i

/-- The inputs `x` with the first `m o` entries of input `o` replaced by block `o` of the vector
`θ`, for every input `o < k`. -/
def withInputs (x : Nat → UInt64 → ℝ) {k : Nat} {m : Fin k → UInt64}
    (θ : EuclideanSpace ℝ (Σ o : Fin k, Fin (m o).toNat)) : Nat → UInt64 → ℝ :=
  fun o i => if ho : o < k then
      if hi : i.toNat < (m ⟨o, ho⟩).toNat then θ ⟨⟨o, ho⟩, ⟨i.toNat, hi⟩⟩ else x o i
    else x o i

namespace Expr

/-- The derivative in `θ` of entry `i` of input `o` of `withInput x j θ`. -/
private noncomputable def withInputDeriv (j : Nat) (m : UInt64) (o : Nat) (i : UInt64) :
    EuclideanSpace ℝ (Fin m.toNat) →L[ℝ] ℝ :=
  if h : o = j ∧ i.toNat < m.toNat then EuclideanSpace.proj ⟨i.toNat, h.2⟩ else 0

private theorem hasFDerivAt_withInput (x : Nat → UInt64 → ℝ) (j : Nat) {m : UInt64}
    (θ₀ : EuclideanSpace ℝ (Fin m.toNat)) (o : Nat) (i : UInt64) :
    HasFDerivAt (fun θ => withInput x j θ o i) (withInputDeriv j m o i) θ₀ := by
  by_cases h : o = j ∧ i.toNat < m.toNat
  · have hf : (fun θ : EuclideanSpace ℝ (Fin m.toNat) => withInput x j θ o i) =
        fun θ => θ ⟨i.toNat, h.2⟩ := funext fun θ => dite_eq_left h
    rw [hf, withInputDeriv, dite_eq_left h]
    exact PiLp.hasFDerivAt_apply 2 θ₀ _
  · have hf : (fun θ : EuclideanSpace ℝ (Fin m.toNat) => withInput x j θ o i) =
        fun _ => x o i := funext fun θ => dite_eq_right h
    rw [hf, withInputDeriv, dite_eq_right h]
    exact hasFDerivAt_const _ _

private theorem withInputDeriv_single (j : Nat) {m : UInt64} (a : Fin m.toNat) (o : Nat)
    (i : UInt64) :
    withInputDeriv j m o i (EuclideanSpace.single a 1) =
      if o = j ∧ i = a.val.toUInt64 then 1 else 0 := by
  have ha : (a.val.toUInt64).toNat = a.val := by
    rw [Nat.toUInt64_eq]
    exact UInt64.toNat_ofNat_of_lt' (lt_trans a.isLt m.toNat_lt_size)
  unfold withInputDeriv
  by_cases h : o = j ∧ i.toNat < m.toNat
  · rw [dite_eq_left h]
    have hiff : i.toNat = a.val ↔ (o = j ∧ i = a.val.toUInt64) :=
      ⟨fun hi => ⟨h.1, UInt64.toNat_inj.mp (hi.trans ha.symm)⟩, fun ⟨_, hi⟩ => hi ▸ ha⟩
    simp only [EuclideanSpace.coe_proj, PiLp.single_apply, Fin.ext_iff]
    exact if_congr hiff rfl rfl
  · rw [dite_eq_right h, _root_.zero_apply, ite_eq_right]
    rintro ⟨rfl, rfl⟩
    exact h ⟨rfl, by rw [ha]; exact a.isLt⟩

/-- The derivative in `θ` of entry `i` of input `o` of `withInputs x θ`. -/
private noncomputable def withInputsDeriv {k : Nat} (m : Fin k → UInt64) (o : Nat) (i : UInt64) :
    EuclideanSpace ℝ (Σ o : Fin k, Fin (m o).toNat) →L[ℝ] ℝ :=
  if ho : o < k then
    if hi : i.toNat < (m ⟨o, ho⟩).toNat then EuclideanSpace.proj ⟨⟨o, ho⟩, ⟨i.toNat, hi⟩⟩ else 0
  else 0

private theorem hasFDerivAt_withInputs (x : Nat → UInt64 → ℝ) {k : Nat} {m : Fin k → UInt64}
    (θ₀ : EuclideanSpace ℝ (Σ o : Fin k, Fin (m o).toNat)) (o : Nat) (i : UInt64) :
    HasFDerivAt (fun θ => withInputs x θ o i) (withInputsDeriv m o i) θ₀ := by
  by_cases ho : o < k
  · by_cases hi : i.toNat < (m ⟨o, ho⟩).toNat
    · have hf : (fun θ : EuclideanSpace ℝ (Σ o : Fin k, Fin (m o).toNat) =>
          withInputs x θ o i) = fun θ => θ ⟨⟨o, ho⟩, ⟨i.toNat, hi⟩⟩ := by
        funext θ
        simp only [withInputs, dite_eq_left ho, dite_eq_left hi]
      rw [hf, withInputsDeriv, dite_eq_left ho, dite_eq_left hi]
      exact PiLp.hasFDerivAt_apply 2 θ₀ _
    · have hf : (fun θ : EuclideanSpace ℝ (Σ o : Fin k, Fin (m o).toNat) =>
          withInputs x θ o i) = fun _ => x o i := by
        funext θ
        simp only [withInputs, dite_eq_left ho, dite_eq_right hi]
      rw [hf, withInputsDeriv, dite_eq_left ho, dite_eq_right hi]
      exact hasFDerivAt_const _ _
  · have hf : (fun θ : EuclideanSpace ℝ (Σ o : Fin k, Fin (m o).toNat) =>
        withInputs x θ o i) = fun _ => x o i := by
      funext θ
      simp only [withInputs, dite_eq_right ho]
    rw [hf, withInputsDeriv, dite_eq_right ho]
    exact hasFDerivAt_const _ _

private theorem withInputsDeriv_single {k : Nat} {m : Fin k → UInt64} (j : Fin k)
    (a : Fin (m j).toNat) (o : Nat) (i : UInt64) :
    withInputsDeriv m o i (EuclideanSpace.single ⟨j, a⟩ 1) =
      if o = j ∧ i = a.val.toUInt64 then 1 else 0 := by
  have ha : (a.val.toUInt64).toNat = a.val := by
    rw [Nat.toUInt64_eq]
    exact UInt64.toNat_ofNat_of_lt' (lt_trans a.isLt (m j).toNat_lt_size)
  unfold withInputsDeriv
  by_cases ho : o < k
  · rw [dite_eq_left ho]
    by_cases hi : i.toNat < (m ⟨o, ho⟩).toNat
    · rw [dite_eq_left hi]
      have hiff : (⟨⟨o, ho⟩, ⟨i.toNat, hi⟩⟩ : Σ o : Fin k, Fin (m o).toNat) = ⟨j, a⟩ ↔
          (o = j ∧ i = a.val.toUInt64) := by
        constructor
        · intro h
          have h₁ : o = j.val := congrArg (fun p : Σ o : Fin k, Fin (m o).toNat => p.1.val) h
          have h₂ : i.toNat = a.val :=
            congrArg (fun p : Σ o : Fin k, Fin (m o).toNat => p.2.val) h
          exact ⟨h₁, UInt64.toNat_inj.mp (h₂.trans ha.symm)⟩
        · rintro ⟨rfl, rfl⟩
          exact Sigma.ext (Fin.ext rfl) (heq_of_eq (Fin.ext ha))
      simp only [EuclideanSpace.coe_proj, PiLp.single_apply]
      exact if_congr hiff rfl rfl
    · rw [dite_eq_right hi, _root_.zero_apply, ite_eq_right]
      rintro ⟨rfl, rfl⟩
      exact hi (by rw [ha]; exact a.isLt)
  · rw [dite_eq_right ho, _root_.zero_apply, ite_eq_right]
    rintro ⟨rfl, -⟩
    exact ho j.isLt

/-- A fold over `p` steps that adds `g r` at step `r` computes, from zero, the sum of `g` over the
first `p` steps. -/
theorem iterate_eq_sum {step : UInt64 → ℝ → Except Error ℝ} {g : Nat → ℝ} :
    ∀ p : Nat, (∀ r < p, ∀ acc, step r.toUInt64 acc = .ok (acc + g r)) →
      iterate step p 0 0 = .ok (∑ r ∈ Finset.range p, g r)
  | 0, _ => by simp only [iterate, Finset.sum_range_zero]
  | p + 1, h => by
      rw [iterate_succ, iterate_eq_sum p fun r hr => h r (by omega), Finset.sum_range_succ]
      refine except_bind_eq_ok.mpr ⟨_, rfl, ?_⟩
      rw [UInt64.zero_add, h p (by omega)]

/-- Over the reals, let the first `m` entries of input `j` of `e` move with `θ`, the other inputs
`x` fixed, and let `F θ` be the vector of the `n` outputs near `θ₀`. If `read` succeeds only below
operand `k`, every output evaluation at `θ₀` is regular and the reader `R` of the gradient kernel
agrees with the inputs at `θ₀` and reads the cotangent `cot` as operand `k`, then `F` is
differentiable at `θ₀` and the gradient kernel computes the adjoint of its derivative at `cot`. -/
theorem gradient_eq_adjoint (e : Expr ℝ [.index] .scalar) {k j : Nat} {m n : UInt64}
    {read : Reader ℝ} (hk : ∀ o i y, read o i = .ok y → o < k) (x : Nat → UInt64 → ℝ)
    {θ₀ : EuclideanSpace ℝ (Fin m.toNat)}
    {F : EuclideanSpace ℝ (Fin m.toNat) → EuclideanSpace ℝ (Fin n.toNat)}
    (hF : ∀ᶠ θ in 𝓝 θ₀, ∀ r : Fin n.toNat, evaluate Arithmetic.ofOps e
      (read.withValues (withInput x j θ)) r.val.toUInt64 = .ok (F θ r))
    (hreg : ∀ r : Fin n.toNat,
      e.Regular (read.withValues (withInput x j θ₀)) (Env.output r.val.toUInt64))
    {R : Reader ℝ} (hR : ∀ o i y, read.withValues (withInput x j θ₀) o i = .ok y → R o i = .ok y)
    {cot : EuclideanSpace ℝ (Fin n.toNat)}
    (hcot : ∀ r : Fin n.toNat, R k r.val.toUInt64 = .ok (cot r)) :
    DifferentiableAt ℝ F θ₀ ∧ ∀ a : Fin m.toNat,
      evaluate Arithmetic.ofOps (e.gradient k j n) R a.val.toUInt64 =
        .ok ((fderiv ℝ F θ₀).adjoint cot a) := by
  have hx : ∀ o i (y : ℝ), read o i = .ok y →
      HasFDerivAt (fun θ => withInput x j θ o i) (withInputDeriv j m o i) θ₀ :=
    fun o i _ _ => hasFDerivAt_withInput x j θ₀ o i
  choose D hD hT using fun r : Fin n.toNat =>
    hasFDerivAt_evaluate_tangent (x := withInput x j) e hk hx (hF.mono fun θ h => h r) (hreg r)
  have hpi : HasFDerivAt (fun θ r => F θ r) (ContinuousLinearMap.pi D) θ₀ := hasFDerivAt_pi.mpr hD
  have htoLp := PiLp.hasFDerivAt_toLp (𝕜 := ℝ) 2 fun r => F θ₀ r
  have hFD : HasFDerivAt F ((PiLp.continuousLinearEquiv 2 ℝ fun _ : Fin n.toNat => ℝ).symm
      |>.toContinuousLinearMap.comp (ContinuousLinearMap.pi D)) θ₀ := (htoLp.comp θ₀ hpi :)
  refine ⟨hFD.differentiableAt, fun a => ?_⟩
  have hseed : ∀ r : Fin n.toNat, evaluate Arithmetic.ofOps (e.tangent k)
      (R.seed k j a.val.toUInt64) r.val.toUInt64 = .ok (D r (EuclideanSpace.single a 1)) := by
    intro r
    refine eval_eq_ok_of_reader _ (fun o i y h => ?_) _ (hT r (EuclideanSpace.single a 1))
    by_cases ho : o < k
    · simp only [Reader.dual, ho, ↓reduceIte] at h
      simp only [Reader.seed, ho, ↓reduceIte]
      exact hR o i y h
    · simp only [Reader.dual, ho, ↓reduceIte] at h
      simp only [Reader.seed, ho, ↓reduceIte]
      obtain ⟨z, -, hy⟩ := Reader.exists_ok_of_withValues h
      subst hy
      have hj : o - k = j ↔ o = k + j := by omega
      simp only [withInputDeriv_single, hj]
  rw [hFD.fderiv, evaluate_gradient, iterate_eq_sum (g := fun r => if h : r < n.toNat then
    cot ⟨r, h⟩ * D ⟨r, h⟩ (EuclideanSpace.single a 1) else 0) n.toNat fun r hr acc => ?_]
  · congr 1
    have hentry (v : EuclideanSpace ℝ (Fin m.toNat)) :
        v a = inner ℝ (EuclideanSpace.single a 1) v := by
      rw [EuclideanSpace.inner_single_left, map_one, one_mul]
    rw [Finset.sum_range, hentry, ContinuousLinearMap.adjoint_inner_right, PiLp.inner_apply]
    refine Finset.sum_congr rfl fun r _ => ?_
    simp [r.isLt]
  · have hc : R k r.toUInt64 = .ok (cot ⟨r, hr⟩) := hcot ⟨r, hr⟩
    have ht : evaluate Arithmetic.ofOps (e.tangent k) (R.seed k j a.val.toUInt64) r.toUInt64 =
        .ok (D ⟨r, hr⟩ (EuclideanSpace.single a 1)) := hseed ⟨r, hr⟩
    simp only [hc, ht, ← except_pure, pure_bind, hr, ↓reduceDIte,
      Arithmetic.ofOps_binary_add, Arithmetic.ofOps_binary_mul]

/-- Over the reals, let the first `m` entries of input `j` of `e` move with `θ`, the other inputs
`x` fixed. If `read` succeeds only below operand `k`, every output evaluation at `θ₀` succeeds and
is regular and `R` agrees with the inputs at `θ₀` and reads the cotangent `cot` as operand `k`,
then near `θ₀` the outputs are the entries of a map `F` that is differentiable at `θ₀`, and the
gradient kernel computes the adjoint of its derivative at `cot`. -/
theorem exists_gradient_eq_adjoint (e : Expr ℝ [.index] .scalar) {k j : Nat} {m n : UInt64}
    {read : Reader ℝ} (hk : ∀ o i y, read o i = .ok y → o < k) (x : Nat → UInt64 → ℝ)
    {θ₀ : EuclideanSpace ℝ (Fin m.toNat)}
    (hy : ∀ r : Fin n.toNat, ∃ y, evaluate Arithmetic.ofOps e
      (read.withValues (withInput x j θ₀)) r.val.toUInt64 = .ok y)
    (hreg : ∀ r : Fin n.toNat,
      e.Regular (read.withValues (withInput x j θ₀)) (Env.output r.val.toUInt64))
    {R : Reader ℝ} (hR : ∀ o i y, read.withValues (withInput x j θ₀) o i = .ok y → R o i = .ok y)
    {cot : EuclideanSpace ℝ (Fin n.toNat)}
    (hcot : ∀ r : Fin n.toNat, R k r.val.toUInt64 = .ok (cot r)) :
    ∃ F : EuclideanSpace ℝ (Fin m.toNat) → EuclideanSpace ℝ (Fin n.toNat),
      (∀ᶠ θ in 𝓝 θ₀, ∀ r : Fin n.toNat, evaluate Arithmetic.ofOps e
        (read.withValues (withInput x j θ)) r.val.toUInt64 = .ok (F θ r)) ∧
      DifferentiableAt ℝ F θ₀ ∧ ∀ a : Fin m.toNat,
        evaluate Arithmetic.ofOps (e.gradient k j n) R a.val.toUInt64 =
          .ok ((fderiv ℝ F θ₀).adjoint cot a) := by
  have hf : ∀ r : Fin n.toNat, ∃ f : EuclideanSpace ℝ (Fin m.toNat) → ℝ, ∀ᶠ θ in 𝓝 θ₀,
      evaluate Arithmetic.ofOps e (read.withValues (withInput x j θ)) r.val.toUInt64 =
        .ok (f θ) := fun r => by
    obtain ⟨y, hy⟩ := hy r
    obtain ⟨f, _, hf, _⟩ := exists_hasFDerivAt_tangent (x := withInput x j) e hk
      (fun o i _ _ => hasFDerivAt_withInput x j θ₀ o i) hy (hreg r)
    exact ⟨f, hf⟩
  choose f hf using hf
  have hF : ∀ᶠ θ in 𝓝 θ₀, ∀ r : Fin n.toNat, evaluate Arithmetic.ofOps e
      (read.withValues (withInput x j θ)) r.val.toUInt64 =
        .ok ((WithLp.toLp 2 fun r => f r θ : EuclideanSpace ℝ (Fin n.toNat)) r) :=
    eventually_all.mpr hf
  exact ⟨fun θ => WithLp.toLp 2 fun r => f r θ, hF,
    gradient_eq_adjoint e hk x (F := fun θ => WithLp.toLp 2 fun r => f r θ) hF hreg hR hcot⟩

/-- Over the reals, let the first `m o` entries of every input `o < k` of `e` move with block `o`
of `θ`, the other entries `x` fixed, and let `F θ` be the vector of the `n` outputs near `θ₀`. If
`read` succeeds only below operand `k`, every output evaluation at `θ₀` is regular and the reader
`R` of the gradient kernels agrees with the inputs at `θ₀` and reads the cotangent `cot` as
operand `k`, then `F` is differentiable at `θ₀` and, for every input `j < k`, the gradient kernel
of input `j` computes block `j` of the adjoint of its derivative at `cot`. -/
theorem gradient_eq_adjoint_withInputs (e : Expr ℝ [.index] .scalar) {k : Nat}
    {m : Fin k → UInt64} {n : UInt64} {read : Reader ℝ}
    (hk : ∀ o i y, read o i = .ok y → o < k) (x : Nat → UInt64 → ℝ)
    {θ₀ : EuclideanSpace ℝ (Σ o : Fin k, Fin (m o).toNat)}
    {F : EuclideanSpace ℝ (Σ o : Fin k, Fin (m o).toNat) → EuclideanSpace ℝ (Fin n.toNat)}
    (hF : ∀ᶠ θ in 𝓝 θ₀, ∀ r : Fin n.toNat, evaluate Arithmetic.ofOps e
      (read.withValues (withInputs x θ)) r.val.toUInt64 = .ok (F θ r))
    (hreg : ∀ r : Fin n.toNat,
      e.Regular (read.withValues (withInputs x θ₀)) (Env.output r.val.toUInt64))
    {R : Reader ℝ}
    (hR : ∀ o i y, read.withValues (withInputs x θ₀) o i = .ok y → R o i = .ok y)
    {cot : EuclideanSpace ℝ (Fin n.toNat)}
    (hcot : ∀ r : Fin n.toNat, R k r.val.toUInt64 = .ok (cot r)) :
    DifferentiableAt ℝ F θ₀ ∧ ∀ (j : Fin k) (a : Fin (m j).toNat),
      evaluate Arithmetic.ofOps (e.gradient k j n) R a.val.toUInt64 =
        .ok ((fderiv ℝ F θ₀).adjoint cot ⟨j, a⟩) := by
  have hx : ∀ o i (y : ℝ), read o i = .ok y →
      HasFDerivAt (fun θ => withInputs x θ o i) (withInputsDeriv m o i) θ₀ :=
    fun o i _ _ => hasFDerivAt_withInputs x θ₀ o i
  choose D hD hT using fun r : Fin n.toNat =>
    hasFDerivAt_evaluate_tangent (x := withInputs x) e hk hx (hF.mono fun θ h => h r) (hreg r)
  have hpi : HasFDerivAt (fun θ r => F θ r) (ContinuousLinearMap.pi D) θ₀ := hasFDerivAt_pi.mpr hD
  have htoLp := PiLp.hasFDerivAt_toLp (𝕜 := ℝ) 2 fun r => F θ₀ r
  have hFD : HasFDerivAt F ((PiLp.continuousLinearEquiv 2 ℝ fun _ : Fin n.toNat => ℝ).symm
      |>.toContinuousLinearMap.comp (ContinuousLinearMap.pi D)) θ₀ := (htoLp.comp θ₀ hpi :)
  refine ⟨hFD.differentiableAt, fun j a => ?_⟩
  have hseed : ∀ r : Fin n.toNat, evaluate Arithmetic.ofOps (e.tangent k)
      (R.seed k j a.val.toUInt64) r.val.toUInt64 =
        .ok (D r (EuclideanSpace.single ⟨j, a⟩ 1)) := by
    intro r
    refine eval_eq_ok_of_reader _ (fun o i y h => ?_) _
      (hT r (EuclideanSpace.single ⟨j, a⟩ 1))
    by_cases ho : o < k
    · simp only [Reader.dual, ho, ↓reduceIte] at h
      simp only [Reader.seed, ho, ↓reduceIte]
      exact hR o i y h
    · simp only [Reader.dual, ho, ↓reduceIte] at h
      simp only [Reader.seed, ho, ↓reduceIte]
      obtain ⟨z, -, hy⟩ := Reader.exists_ok_of_withValues h
      subst hy
      have hj : o - k = j ↔ o = k + j := by omega
      simp only [withInputsDeriv_single, hj]
  rw [hFD.fderiv, evaluate_gradient, iterate_eq_sum (g := fun r => if h : r < n.toNat then
    cot ⟨r, h⟩ * D ⟨r, h⟩ (EuclideanSpace.single ⟨j, a⟩ 1) else 0) n.toNat fun r hr acc => ?_]
  · congr 1
    have hentry (v : EuclideanSpace ℝ (Σ o : Fin k, Fin (m o).toNat)) :
        v ⟨j, a⟩ = inner ℝ (EuclideanSpace.single ⟨j, a⟩ 1) v := by
      rw [EuclideanSpace.inner_single_left, map_one, one_mul]
    rw [Finset.sum_range, hentry, ContinuousLinearMap.adjoint_inner_right, PiLp.inner_apply]
    refine Finset.sum_congr rfl fun r _ => ?_
    simp [r.isLt]
  · have hc : R k r.toUInt64 = .ok (cot ⟨r, hr⟩) := hcot ⟨r, hr⟩
    have ht : evaluate Arithmetic.ofOps (e.tangent k) (R.seed k j a.val.toUInt64) r.toUInt64 =
        .ok (D ⟨r, hr⟩ (EuclideanSpace.single ⟨j, a⟩ 1)) := hseed ⟨r, hr⟩
    simp only [hc, ht, ← except_pure, pure_bind, hr, ↓reduceDIte,
      Arithmetic.ofOps_binary_add, Arithmetic.ofOps_binary_mul]

/-- Over the reals, let the first `m o` entries of every input `o < k` of `e` move with block `o`
of `θ`, the other entries `x` fixed. If `read` succeeds only below operand `k`, every output
evaluation at `θ₀` succeeds and is regular and `R` agrees with the inputs at `θ₀` and reads the
cotangent `cot` as operand `k`, then near `θ₀` the outputs are the entries of a map `F` that is
differentiable at `θ₀`, and the gradient kernel of each input `j < k` computes block `j` of the
adjoint of its derivative at `cot`. -/
theorem exists_gradient_eq_adjoint_withInputs (e : Expr ℝ [.index] .scalar) {k : Nat}
    {m : Fin k → UInt64} {n : UInt64} {read : Reader ℝ}
    (hk : ∀ o i y, read o i = .ok y → o < k) (x : Nat → UInt64 → ℝ)
    {θ₀ : EuclideanSpace ℝ (Σ o : Fin k, Fin (m o).toNat)}
    (hy : ∀ r : Fin n.toNat, ∃ y, evaluate Arithmetic.ofOps e
      (read.withValues (withInputs x θ₀)) r.val.toUInt64 = .ok y)
    (hreg : ∀ r : Fin n.toNat,
      e.Regular (read.withValues (withInputs x θ₀)) (Env.output r.val.toUInt64))
    {R : Reader ℝ}
    (hR : ∀ o i y, read.withValues (withInputs x θ₀) o i = .ok y → R o i = .ok y)
    {cot : EuclideanSpace ℝ (Fin n.toNat)}
    (hcot : ∀ r : Fin n.toNat, R k r.val.toUInt64 = .ok (cot r)) :
    ∃ F : EuclideanSpace ℝ (Σ o : Fin k, Fin (m o).toNat) → EuclideanSpace ℝ (Fin n.toNat),
      (∀ᶠ θ in 𝓝 θ₀, ∀ r : Fin n.toNat, evaluate Arithmetic.ofOps e
        (read.withValues (withInputs x θ)) r.val.toUInt64 = .ok (F θ r)) ∧
      DifferentiableAt ℝ F θ₀ ∧ ∀ (j : Fin k) (a : Fin (m j).toNat),
        evaluate Arithmetic.ofOps (e.gradient k j n) R a.val.toUInt64 =
          .ok ((fderiv ℝ F θ₀).adjoint cot ⟨j, a⟩) := by
  have hf : ∀ r : Fin n.toNat, ∃ f : EuclideanSpace ℝ (Σ o : Fin k, Fin (m o).toNat) → ℝ,
      ∀ᶠ θ in 𝓝 θ₀, evaluate Arithmetic.ofOps e (read.withValues (withInputs x θ))
        r.val.toUInt64 = .ok (f θ) := fun r => by
    obtain ⟨y, hy⟩ := hy r
    obtain ⟨f, _, hf, _⟩ := exists_hasFDerivAt_tangent (x := withInputs x) e hk
      (fun o i _ _ => hasFDerivAt_withInputs x θ₀ o i) hy (hreg r)
    exact ⟨f, hf⟩
  choose f hf using hf
  have hF : ∀ᶠ θ in 𝓝 θ₀, ∀ r : Fin n.toNat, evaluate Arithmetic.ofOps e
      (read.withValues (withInputs x θ)) r.val.toUInt64 =
        .ok ((WithLp.toLp 2 fun r => f r θ : EuclideanSpace ℝ (Fin n.toNat)) r) :=
    eventually_all.mpr hf
  exact ⟨fun θ => WithLp.toLp 2 fun r => f r θ, hF, gradient_eq_adjoint_withInputs e hk x
    (F := fun θ => WithLp.toLp 2 fun r => f r θ) hF hreg hR hcot⟩

/-- Over the reals, a fold from `0` whose step adds to the accumulator the value `g r` of `body`
at step `r`, whatever the accumulator, evaluates to the sum of `g` over the `c` steps of its
count. -/
theorem eval_fold_eq_sum {Γ : List Ty} (read : Reader ℝ) (env : Env ℝ Γ)
    {count : Expr ℝ Γ .index} {c : UInt64} (hc : count.eval Arithmetic.ofOps read env = .ok c)
    {body : Expr ℝ (.scalar :: .index :: Γ) .scalar} {g : Nat → ℝ}
    (hb : ∀ r < c.toNat, ∀ acc : ℝ, body.eval Arithmetic.ofOps read
      (Env.push (t := .scalar) acc (Env.push (t := .index) r.toUInt64 env)) = .ok (g r)) :
    (Expr.fold count (.scalar 0) (.binary .add (.var .zero) body)).eval Arithmetic.ofOps read
      env = .ok (∑ r ∈ Finset.range c.toNat, g r) := by
  have hstep : ∀ r < c.toNat, ∀ acc : ℝ, (Expr.binary .add (.var .zero) body).eval
      Arithmetic.ofOps read (Env.push (t := .scalar) acc (Env.push (t := .index) r.toUInt64 env)) =
        .ok (acc + g r) := by
    intro r hr acc
    simp only [eval, Env.get_push_zero, hb r hr acc, ← except_pure, pure_bind,
      Arithmetic.ofOps_binary_add]
  have h := iterate_eq_sum (step := fun i acc => (Expr.binary .add (.var .zero) body).eval
    Arithmetic.ofOps read (Env.push (t := .scalar) acc (Env.push (t := .index) i env)))
    c.toNat hstep
  exact except_bind_eq_ok.mpr ⟨c, hc, except_bind_eq_ok.mpr ⟨0, rfl, h⟩⟩

/-- Over the reals, the evaluation at output `i` of a fold from `0` whose step adds to the
accumulator the value `g r` of `body` at step `r`, whatever the accumulator, is the sum of `g`
over the `c` steps of its count. -/
theorem evaluate_fold_eq_sum (read : Reader ℝ) (i : UInt64) {count : Expr ℝ [.index] .index}
    {c : UInt64} (hc : count.eval Arithmetic.ofOps read (Env.output i) = .ok c)
    {body : Expr ℝ [.scalar, .index, .index] .scalar} {g : Nat → ℝ}
    (hb : ∀ r < c.toNat, ∀ acc : ℝ, body.eval Arithmetic.ofOps read
      (Env.push (t := .scalar) acc (Env.push (t := .index) r.toUInt64 (Env.output i))) =
        .ok (g r)) :
    evaluate Arithmetic.ofOps (.fold count (.scalar 0) (.binary .add (.var .zero) body)) read i =
      .ok (∑ r ∈ Finset.range c.toNat, g r) :=
  eval_fold_eq_sum read (Env.output i) hc hb

end Expr

end TorchLean.Kernel
