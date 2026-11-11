/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.Calculus.FDeriv.Add
public import Mathlib.Analysis.Calculus.FDeriv.Bilinear
public import Mathlib.Analysis.Calculus.FDeriv.Comp
public import Mathlib.Analysis.Calculus.FDeriv.Linear
public import Mathlib.Analysis.Calculus.FDeriv.Mul
public import Mathlib.Analysis.Calculus.Deriv.Inv
public import Mathlib.Analysis.InnerProductSpace.Adjoint
public import Mathlib.Analysis.InnerProductSpace.ProdL2
public import TorchLean.NN.FDeriv.DerivCert
import Mathlib.Tactic.FunProp

/-!
# Pullbacks of differentiable maps

`HasPullbackAt f D b x` says that `f` has Fréchet derivative `D` at `x` and that `b` is the adjoint
of `D`, in the form `⟪D u, δ⟫ = ⟪u, b δ⟫` for all `u` and `δ`. It is a `fun_prop` property, and
`D` and `b` are ordinary arguments: the rules below give both for the identity, constants,
composition, sums, differences, negation, constant multiples, finite sums, and the projections
and pairing of an L2 product, so `fun_prop` assembles them for a composite of registered maps by
unification. `HasPullbackAt.cert` turns the result into the statement shape of the operator
certificates, the derivative at `x` and a stored pullback equal to its adjoint, and
`HasPullbackAt.differentiableAt_and` into `DerivCert`, the statement `op_cert` checks.
`DerivCert.comp` precomposes a certificate with a map that has a pullback.

Real-valued maps have the product, quotient and constant-multiple rules, a varying real scalar
times a map has the rule `HasPullbackAt.smul`, and `HasPullbackAt.of_hasDerivAt` turns a Mathlib
derivative `HasDerivAt φ φ' a` of a function `ℝ → ℝ` into its pullback `δ ↦ φ' * δ`.

Rules for further maps come from `HasPullbackAt.clm` (a continuous linear map and its transpose)
and `HasPullbackAt.bilin` (a continuous bilinear map and its two transposes).
-/

@[expose] public section

open scoped RealInnerProductSpace
open Filter Topology

namespace TorchLean

variable {E F G H : Type*}
  [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [NormedAddCommGroup F] [InnerProductSpace ℝ F]
  [NormedAddCommGroup G] [InnerProductSpace ℝ G]
  [NormedAddCommGroup H] [InnerProductSpace ℝ H]

/-- `f` has Fréchet derivative `D` at `x`, and `b` is the adjoint of `D`:
`⟪D u, δ⟫ = ⟪u, b δ⟫` for all `u` and `δ`. -/
@[fun_prop]
structure HasPullbackAt (f : E → F) (D : E →L[ℝ] F) (b : F → E) (x : E) : Prop where
  /-- `D` is the Fréchet derivative of `f` at `x`. -/
  hasFDerivAt : HasFDerivAt f D x
  /-- `b` pairs with `D`: `⟪D u, δ⟫ = ⟪u, b δ⟫`. -/
  inner_eq : ∀ u δ, ⟪D u, δ⟫ = ⟪u, b δ⟫

/-- Two points of an L2 product with equal components are equal. -/
theorem withLp_prod_ext {α β : Type*} {x y : WithLp 2 (α × β)} (h₁ : x.fst = y.fst)
    (h₂ : x.snd = y.snd) : x = y :=
  WithLp.ofLp_injective 2 (Prod.ext h₁ h₂)

namespace HasPullbackAt

section Certificates

variable [CompleteSpace E] [CompleteSpace F] {f : E → F} {D D' : E →L[ℝ] F} {b : F → E} {x : E}
  {v : E} {δ : F}

/-- A pullback is the adjoint of the derivative. -/
theorem eq_adjoint (h : HasPullbackAt f D b x) (δ : F) : b δ = D.adjoint δ := by
  apply ext_inner_left ℝ
  intro u
  rw [ContinuousLinearMap.adjoint_inner_right, h.inner_eq]

/-- A derivative and its adjoint form a pullback. -/
theorem of_adjoint (hf : HasFDerivAt f D x) (hb : ∀ δ, b δ = D.adjoint δ) :
    HasPullbackAt f D b x :=
  ⟨hf, fun u δ => by rw [hb, ContinuousLinearMap.adjoint_inner_right]⟩

/-- The certificate shape: `D` is the derivative of `f` at `x`, and a value equal to `b δ` is the
adjoint of the derivative applied to `δ`. -/
theorem cert (h : HasPullbackAt f D b x) (hv : v = b δ) :
    HasFDerivAt f D x ∧ v = (fderiv ℝ f x).adjoint δ :=
  ⟨h.hasFDerivAt, by rw [hv, h.hasFDerivAt.fderiv, h.eq_adjoint]⟩

/-- The certificate shape for a derivative `D` that is shown equal to the synthesized one. -/
theorem cert_of_eq (h : HasPullbackAt f D' b x) (hD : D' = D) (hv : v = b δ) :
    HasFDerivAt f D x ∧ v = (fderiv ℝ f x).adjoint δ :=
  hD ▸ h.cert hv

/-- The certificate shape that leaves the derivative implicit, `DerivCert f x δ v`: `f` is
differentiable at `x`, and a value equal to `b δ` is the adjoint of its derivative applied to
`δ`. -/
theorem differentiableAt_and (h : HasPullbackAt f D b x) (hv : v = b δ) : DerivCert f x δ v :=
  ⟨h.hasFDerivAt.differentiableAt, (h.cert hv).2⟩

/-- A value equal to `b δ` is the adjoint of the derivative of `f` at `x` applied to `δ`. -/
theorem eq_fderiv_adjoint (h : HasPullbackAt f D b x) (hv : v = b δ) :
    v = (fderiv ℝ f x).adjoint δ :=
  (h.cert hv).2

end Certificates

variable {f g : E → F} {D Df Dg : E →L[ℝ] F} {b bf bg : F → E} {x : E}

/-- A pullback is unchanged by replacing the map with one equal to it near `x`. -/
theorem congr_of_eventuallyEq (h : HasPullbackAt f D b x) (hg : g =ᶠ[𝓝 x] f) :
    HasPullbackAt g D b x :=
  ⟨h.hasFDerivAt.congr_of_eventuallyEq hg, h.inner_eq⟩

/-- A pullback is unchanged by replacing the derivative and the pullback with equal ones. -/
theorem congr_deriv {D' : E →L[ℝ] F} {b' : F → E} (h : HasPullbackAt f D b x) (hD : D = D')
    (hb : ∀ δ, b δ = b' δ) : HasPullbackAt f D' b' x :=
  ⟨hD ▸ h.hasFDerivAt, fun u δ => by rw [← hD, ← hb, h.inner_eq]⟩

/-- The identity map is its own derivative and its own pullback. -/
@[fun_prop]
theorem id (x : E) : HasPullbackAt (fun y => y) (ContinuousLinearMap.id ℝ E) (fun δ => δ) x :=
  ⟨hasFDerivAt_id x, fun _ _ => rfl⟩

/-- A constant map has derivative zero and pullback zero. -/
@[fun_prop]
theorem const (c : F) (x : E) : HasPullbackAt (fun _ : E => c) 0 (fun _ => 0) x :=
  ⟨hasFDerivAt_const c x, fun _ _ => by simp⟩

/-- Chain rule: the derivatives compose, and the pullbacks compose in the reverse order. -/
@[fun_prop]
theorem comp {g : F → G} {f : E → F} {Dg : F →L[ℝ] G} {Df : E →L[ℝ] F} {bg : G → F}
    {bf : F → E} {x : E} (hg : HasPullbackAt g Dg bg (f x)) (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun z => g (f z)) (Dg.comp Df) (fun δ => bf (bg δ)) x :=
  ⟨hg.hasFDerivAt.comp x hf.hasFDerivAt, fun u δ => by
    rw [ContinuousLinearMap.comp_apply, hg.inner_eq, hf.inner_eq]⟩

/-- The pullback of a sum is the sum of the pullbacks. -/
@[fun_prop]
theorem add (hf : HasPullbackAt f Df bf x) (hg : HasPullbackAt g Dg bg x) :
    HasPullbackAt (fun y => f y + g y) (Df + Dg) (fun δ => bf δ + bg δ) x :=
  ⟨hf.hasFDerivAt.add hg.hasFDerivAt, fun u δ => by
    rw [_root_.add_apply, inner_add_left, inner_add_right, hf.inner_eq, hg.inner_eq]⟩

/-- The pullback of a difference is the difference of the pullbacks. -/
@[fun_prop]
theorem sub (hf : HasPullbackAt f Df bf x) (hg : HasPullbackAt g Dg bg x) :
    HasPullbackAt (fun y => f y - g y) (Df - Dg) (fun δ => bf δ - bg δ) x :=
  ⟨hf.hasFDerivAt.sub hg.hasFDerivAt, fun u δ => by
    rw [_root_.sub_apply, inner_sub_left, inner_sub_right, hf.inner_eq, hg.inner_eq]⟩

/-- The pullback of a negation is the negated pullback. -/
@[fun_prop]
theorem neg (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => -f y) (-Df) (fun δ => -bf δ) x :=
  ⟨hf.hasFDerivAt.neg, fun u δ => by
    rw [neg_apply, inner_neg_left, inner_neg_right, hf.inner_eq]⟩

/-- The pullback of a constant multiple is the same multiple of the pullback. -/
@[fun_prop]
theorem const_smul (c : ℝ) (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => c • f y) (c • Df) (fun δ => c • bf δ) x :=
  ⟨hf.hasFDerivAt.const_smul c, fun u δ => by
    rw [_root_.smul_apply, real_inner_smul_left, real_inner_smul_right,
      hf.inner_eq]⟩

/-- The pullback of a finite sum over `s` is the sum of the pullbacks of its terms. -/
@[fun_prop]
theorem sum {ι : Type*} (s : Finset ι) {f : ι → E → F} {D : ι → E →L[ℝ] F} {b : ι → F → E}
    (h : ∀ i ∈ s, HasPullbackAt (f i) (D i) (b i) x) :
    HasPullbackAt (fun y => ∑ i ∈ s, f i y) (∑ i ∈ s, D i) (fun δ => ∑ i ∈ s, b i δ) x :=
  ⟨HasFDerivAt.fun_sum fun i hi => (h i hi).hasFDerivAt, fun u δ => by
    rw [_root_.sum_apply, sum_inner, inner_sum]
    exact Finset.sum_congr rfl fun i hi => (h i hi).inner_eq u δ⟩

/-- A continuous linear map `L` with transpose `T` (`⟪L u, δ⟫ = ⟪u, T δ⟫`) applied after `f`:
the derivative is `L ∘ Df`, and the pullback applies `T` first. -/
theorem clm (L : F →L[ℝ] G) (T : G → F) (hT : ∀ u δ, ⟪L u, δ⟫ = ⟪u, T δ⟫)
    (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => L (f y)) (L.comp Df) (fun δ => bf (T δ)) x :=
  ⟨L.hasFDerivAt.comp x hf.hasFDerivAt, fun u δ => by
    rw [ContinuousLinearMap.comp_apply, hT, hf.inner_eq]⟩

/-- A continuous bilinear map `B` with transposes `TL` and `TR` in its two arguments, applied to
`f` and `g`: the product rule, and the pullback is the sum of the two transposed pullbacks. -/
theorem bilin {f : E → F} {g : E → G} {Df : E →L[ℝ] F} {Dg : E →L[ℝ] G} {bf : F → E}
    {bg : G → E} (B : F →L[ℝ] G →L[ℝ] H) (TL : G → H → F) (TR : F → H → G)
    (hL : ∀ u v δ, ⟪B u v, δ⟫ = ⟪u, TL v δ⟫) (hR : ∀ u v δ, ⟪B u v, δ⟫ = ⟪v, TR u δ⟫)
    (hf : HasPullbackAt f Df bf x) (hg : HasPullbackAt g Dg bg x) :
    HasPullbackAt (fun y => B (f y) (g y)) ((B.flip (g x)).comp Df + (B (f x)).comp Dg)
      (fun δ => bf (TL (g x) δ) + bg (TR (f x) δ)) x := by
  refine ⟨(B.hasFDerivAt_of_bilinear hf.hasFDerivAt hg.hasFDerivAt).congr_fderiv ?_,
    fun u δ => ?_⟩
  · ext u
    simp [add_comm]
  · rw [_root_.add_apply, inner_add_left, inner_add_right, ContinuousLinearMap.comp_apply,
      ContinuousLinearMap.comp_apply, ContinuousLinearMap.flip_apply, hL, hf.inner_eq, hR,
      hg.inner_eq]

section Real

variable {f g : E → ℝ} {Df Dg : E →L[ℝ] ℝ} {bf bg : ℝ → E}

/-- A real function with derivative `φ'` at `a` has at `a` the derivative `D = φ' • id` and the
pullback `b = fun δ => φ' * δ`, given as equations so that `fun_prop` uses it as a transition. -/
@[fun_prop]
theorem of_hasDerivAt {φ : ℝ → ℝ} {φ' a : ℝ} {D : ℝ →L[ℝ] ℝ} {b : ℝ → ℝ}
    (hφ : HasDerivAt φ φ' a) (hD : D = φ' • ContinuousLinearMap.id ℝ ℝ)
    (hb : b = fun δ => φ' * δ) : HasPullbackAt φ D b a := by
  subst hD hb
  exact ⟨hφ.hasFDerivAt.congr_fderiv (ContinuousLinearMap.ext fun u => by
      simp only [ContinuousLinearMap.toSpanSingleton_apply, _root_.smul_apply,
        ContinuousLinearMap.id_apply, smul_eq_mul, mul_comm]),
    fun u δ => by
      simp only [_root_.smul_apply, ContinuousLinearMap.id_apply, smul_eq_mul,
        Real.inner_apply]
      ring⟩

/-- The product of two real-valued maps, by the product rule: the cotangent `δ` pulls back as
`bf (δ * g x) + bg (δ * f x)`. -/
@[fun_prop]
theorem mul_real (hf : HasPullbackAt f Df bf x) (hg : HasPullbackAt g Dg bg x) :
    HasPullbackAt (fun y => f y * g y) (f x • Dg + g x • Df)
      (fun δ => bf (δ * g x) + bg (δ * f x)) x :=
  ⟨hf.hasFDerivAt.fun_mul hg.hasFDerivAt, fun u δ => by
    rw [inner_add_right, ← hf.inner_eq, ← hg.inner_eq]
    simp only [_root_.add_apply, _root_.smul_apply, smul_eq_mul, Real.inner_apply]
    ring⟩

/-- A real constant `c` times a real-valued map: the cotangent `δ` pulls back as `bf (c * δ)`. -/
@[fun_prop]
theorem const_mul (c : ℝ) (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => c * f y) (c • Df) (fun δ => bf (c * δ)) x :=
  ⟨hf.hasFDerivAt.const_mul c, fun u δ => by
    rw [← hf.inner_eq]
    simp only [_root_.smul_apply, smul_eq_mul, Real.inner_apply]
    ring⟩

/-- A real-valued map divided by a real constant `c`: the cotangent `δ` pulls back as
`bf (δ / c)`, for every `c` (at `c = 0` both sides are zero). -/
@[fun_prop]
theorem div_const (c : ℝ) (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => f y / c) (c⁻¹ • Df) (fun δ => bf (δ / c)) x :=
  ((const_mul c⁻¹ hf).congr_deriv rfl fun δ => by rw [div_eq_inv_mul]).congr_of_eventuallyEq
    (Eventually.of_forall fun y => div_eq_inv_mul (f y) c)

/-- The quotient of two real-valued maps where the denominator `g x` is nonzero: the cotangent
`δ` pulls back as `bf (δ / g x) + bg (-(δ * f x / g x ^ 2))`. -/
@[fun_prop]
theorem div_real (hg0 : g x ≠ 0) (hf : HasPullbackAt f Df bf x) (hg : HasPullbackAt g Dg bg x) :
    HasPullbackAt (fun y => f y / g y) ((g x)⁻¹ • Df - (f x / g x ^ 2) • Dg)
      (fun δ => bf (δ / g x) + bg (-(δ * f x / g x ^ 2))) x := by
  have hinv := (of_hasDerivAt (hasDerivAt_inv hg0) rfl rfl).comp hg
  refine ((hf.mul_real hinv).congr_deriv ?_ fun δ => ?_).congr_of_eventuallyEq
    (Eventually.of_forall fun y => div_eq_mul_inv (f y) (g y))
  · refine ContinuousLinearMap.ext fun u => ?_
    simp only [_root_.add_apply, _root_.sub_apply, _root_.smul_apply,
      ContinuousLinearMap.comp_apply, ContinuousLinearMap.id_apply, smul_eq_mul]
    field_simp
    ring
  · congr 2; ring

/-- A real-valued map `f` times a map `g` into an inner product space: the cotangent `δ` pulls
back as `bf ⟪g x, δ⟫ + bg (f x • δ)`. -/
@[fun_prop]
theorem smul {f : E → ℝ} {g : E → F} {Df : E →L[ℝ] ℝ} {Dg : E →L[ℝ] F} {bf : ℝ → E}
    {bg : F → E} (hf : HasPullbackAt f Df bf x) (hg : HasPullbackAt g Dg bg x) :
    HasPullbackAt (fun y => f y • g y) (f x • Dg + Df.smulRight (g x))
      (fun δ => bf ⟪g x, δ⟫ + bg (f x • δ)) x :=
  ⟨hf.hasFDerivAt.fun_smul hg.hasFDerivAt, fun u δ => by
    rw [inner_add_right, ← hf.inner_eq, ← hg.inner_eq, _root_.add_apply, inner_add_left,
      _root_.smul_apply, ContinuousLinearMap.smulRight_apply, real_inner_smul_left,
      real_inner_smul_left, real_inner_smul_right, Real.inner_apply]
    ring⟩

end Real

section Prod

variable {f : E → WithLp 2 (F × G)} {D : E →L[ℝ] WithLp 2 (F × G)} {b : WithLp 2 (F × G) → E}

/-- The first projection of an L2 product: its cotangent `δ` pulls back as `(δ, 0)`. -/
@[fun_prop]
theorem fst (h : HasPullbackAt f D b x) :
    HasPullbackAt (fun y => (f y).fst) ((WithLp.fstL 2 ℝ F G).comp D)
      (fun δ => b (WithLp.toLp 2 (δ, 0))) x :=
  ⟨(WithLp.fstL 2 ℝ F G).hasFDerivAt.comp x h.hasFDerivAt, fun u δ => by
    rw [← h.inner_eq]
    simp [WithLp.prod_inner_apply]⟩

/-- The second projection of an L2 product: its cotangent `δ` pulls back as `(0, δ)`. -/
@[fun_prop]
theorem snd (h : HasPullbackAt f D b x) :
    HasPullbackAt (fun y => (f y).snd) ((WithLp.sndL 2 ℝ F G).comp D)
      (fun δ => b (WithLp.toLp 2 (0, δ))) x :=
  ⟨(WithLp.sndL 2 ℝ F G).hasFDerivAt.comp x h.hasFDerivAt, fun u δ => by
    rw [← h.inner_eq]
    simp [WithLp.prod_inner_apply]⟩

/-- Pairing two maps into an L2 product: the pullback adds the pullbacks of the two components. -/
@[fun_prop]
theorem toLp_prod {f : E → F} {g : E → G} {Df : E →L[ℝ] F} {Dg : E →L[ℝ] G} {bf : F → E}
    {bg : G → E} (hf : HasPullbackAt f Df bf x) (hg : HasPullbackAt g Dg bg x) :
    HasPullbackAt (fun y => WithLp.toLp 2 (f y, g y))
      ((WithLp.prodContinuousLinearEquiv 2 ℝ F G).symm.toContinuousLinearMap.comp (Df.prod Dg))
      (fun δ => bf δ.fst + bg δ.snd) x :=
  ⟨((WithLp.prodContinuousLinearEquiv 2 ℝ F G).symm.toContinuousLinearMap.hasFDerivAt.comp x
      (hf.hasFDerivAt.prodMk hg.hasFDerivAt)), fun u δ => by
    simp [WithLp.prod_inner_apply, inner_add_right, hf.inner_eq, hg.inner_eq]⟩

end Prod

end HasPullbackAt

/-- A certificate of `f` at `g y`, precomposed with a map `g` that has the pullback `bg` at `y`,
is a certificate of `f ∘ g` at `y` with the value `bg v`. -/
theorem DerivCert.comp [CompleteSpace E] [CompleteSpace F] [CompleteSpace G] {f : E → F}
    {g : G → E} {Dg : G →L[ℝ] E} {bg : E → G} {y : G} {δ : F} {v : E}
    (h : DerivCert f (g y) δ v) (hg : HasPullbackAt g Dg bg y) :
    DerivCert (f ∘ g) y δ (bg v) := by
  obtain ⟨hf, hv⟩ := h
  have hfg := hf.hasFDerivAt.comp y hg.hasFDerivAt
  refine ⟨hfg.differentiableAt, ?_⟩
  rw [hfg.fderiv, ContinuousLinearMap.adjoint_comp, ContinuousLinearMap.comp_apply, ← hv]
  exact hg.eq_adjoint v

end TorchLean
