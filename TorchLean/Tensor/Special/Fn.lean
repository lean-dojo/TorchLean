/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Elementwise
public import TorchLean.Tensor.Erf
public import TorchLean.Tensor.Special.Bessel
public import TorchLean.Tensor.Special.Gamma
public import TorchLean.Tensor.Special.Zeta
public import Mathlib.Analysis.Calculus.Deriv.Shift
public import Mathlib.Analysis.Calculus.FDeriv.Symmetric
public import Mathlib.Analysis.Calculus.LineDeriv.Basic
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Sinc


/-!
# The special functions

`SpecialFn n` names the special functions of `n` real arguments that TorchLean defines, each after
its `torch.special` function (or `jax.lax`'s, for `betainc`), and `SpecialFn.real` maps each name to
its definition: the error-function family in `TorchLean.Tensor.Erf`, the gamma family and the
regularized incomplete gamma and beta functions in `TorchLean.Tensor.Special.Gamma`, the Bessel
functions in `TorchLean.Tensor.Special.Bessel`, the Hurwitz zeta function in
`TorchLean.Tensor.Special.Zeta` and Mathlib's `Real.sinc` for the spherical Bessel function `j₀`.
There are fourteen functions of one argument, three of two (`igamma`, `igammac` and `zeta`) and
one of three (`betainc`). `Special α` is the optional scalar capability, as `Erf α` is for `erf`: a
scalar type has it when it can evaluate every member and its partial derivatives of every order,
`Special.deriv f k x`, of order `k i` in argument `i`. Order `0` is the function itself; dual
numbers need the higher orders, because the slope of a dual number carries the next derivative in
its tangent.

`ℝ` has an instance here. It takes the derivatives along the axes with Mathlib's `lineDeriv`, the
last argument's first. `SpecialFn.hasFDerivAt_deriv` proves that wherever the function is smooth
the orders `k + Pi.single i 1` are the partial derivatives of order `k`, the components of its
derivative: there the derivatives along two axes commute, by the symmetry of second derivatives
(Schwarz's theorem, Mathlib's `ContDiffAt.isSymmSndFDerivAt`). At order `0` differentiability is
enough (`SpecialFn.hasFDerivAt_real`). For a function of one argument, order `k` is Mathlib's
iterated derivative `deriv^[k 0]` of `SpecialFn.real₁` (`SpecialFn.deriv_eq_iterate`).
`SpecialFn.contDiffAt_real` proves every member smooth on `SpecialFn.smoothDomain`, which for
`igamma` and `igammac` holds the points with both arguments positive, for `zeta` the points
`(s, q)` with `s > 1` and `q > 0`, and for `betainc` the points `(a, b, x)` with `a` and `b`
positive and `x ∈ (0, 1)`. Two functions of the frameworks are orders of these:
`torch.special.polygamma n` and `lax.polygamma` are order `n` of `digamma`
(`SpecialFn.deriv_digamma`), and `lax.igamma_grad_a` is order `(1, 0)` of `igamma`
(`SpecialFn.deriv_igamma`). `Tensor.special f k xs` applies order `k` of `f` at every coordinate of
the tensors `xs i`.

## References

- PyTorch 2.13 documentation, `torch.special`:
  `https://docs.pytorch.org/docs/2.13/special.html`
- JAX documentation, `jax.lax` (`igamma`, `igammac`, `igamma_grad_a`, `polygamma`, `zeta`,
  `betainc`):
  `https://docs.jax.dev/en/latest/jax.lax.html`
-/

@[expose] public section

noncomputable section

namespace TorchLean

open scoped ContDiff

/-- The special functions, each named after its `torch.special` function and indexed by its
number of arguments. -/
inductive SpecialFn : ℕ → Type where
  /-- `torch.special.erfc`. -/
  | erfc : SpecialFn 1
  /-- `torch.special.erfcx`. -/
  | erfcx : SpecialFn 1
  /-- `torch.special.erfinv`. -/
  | erfinv : SpecialFn 1
  /-- `torch.special.ndtr`. -/
  | ndtr : SpecialFn 1
  /-- `torch.special.ndtri`. -/
  | ndtri : SpecialFn 1
  /-- `torch.special.gammaln` (`torch.lgamma`), `log |Γ(x)|` as PyTorch 2.13's docstring says in
  words. Its formula `ln Γ(|input|)` puts the bars inside `Γ`, which is a typo. -/
  | lgamma : SpecialFn 1
  /-- `torch.special.digamma`. -/
  | digamma : SpecialFn 1
  /-- `torch.special.bessel_j0`. -/
  | besselJ0 : SpecialFn 1
  /-- `torch.special.bessel_j1`. -/
  | besselJ1 : SpecialFn 1
  /-- `torch.special.i0`. -/
  | besselI0 : SpecialFn 1
  /-- `torch.special.i1`. -/
  | besselI1 : SpecialFn 1
  /-- `torch.special.i0e`. -/
  | besselI0e : SpecialFn 1
  /-- `torch.special.i1e`. -/
  | besselI1e : SpecialFn 1
  /-- `torch.special.spherical_bessel_j0`. -/
  | sphericalBesselJ0 : SpecialFn 1
  /-- `torch.special.gammainc` (`lax.igamma`), the regularized lower incomplete gamma function
  `P(a, x)`. -/
  | igamma : SpecialFn 2
  /-- `torch.special.gammaincc` (`lax.igammac`), the regularized upper incomplete gamma function
  `Q(a, x)`. -/
  | igammac : SpecialFn 2
  /-- `torch.special.zeta` (`lax.zeta`), the Hurwitz zeta function `ζ(s, q)`. -/
  | zeta : SpecialFn 2
  /-- `lax.betainc`, the regularized incomplete beta function `I_x(a, b)` (DLMF 8.17.1 and
  8.17.2), with arguments `a`, `b` and `x` in that order. PyTorch has no such function. -/
  | betainc : SpecialFn 3
  deriving DecidableEq, Repr

namespace SpecialFn

/-- The real definition of each special function. -/
def real : {n : ℕ} → SpecialFn n → (Fin n → ℝ) → ℝ
  | _, erfc, x => TorchLean.erfc (x 0)
  | _, erfcx, x => TorchLean.erfcx (x 0)
  | _, erfinv, x => TorchLean.erfinv (x 0)
  | _, ndtr, x => TorchLean.ndtr (x 0)
  | _, ndtri, x => TorchLean.ndtri (x 0)
  | _, lgamma, x => TorchLean.lgamma (x 0)
  | _, digamma, x => TorchLean.digamma (x 0)
  | _, besselJ0, x => TorchLean.besselJ0 (x 0)
  | _, besselJ1, x => TorchLean.besselJ1 (x 0)
  | _, besselI0, x => TorchLean.besselI0 (x 0)
  | _, besselI1, x => TorchLean.besselI1 (x 0)
  | _, besselI0e, x => TorchLean.besselI0e (x 0)
  | _, besselI1e, x => TorchLean.besselI1e (x 0)
  | _, sphericalBesselJ0, x => Real.sinc (x 0)
  | _, igamma, x => TorchLean.igamma (x 0) (x 1)
  | _, igammac, x => TorchLean.igammac (x 0) (x 1)
  | _, zeta, x => TorchLean.zeta (x 0) (x 1)
  | _, betainc, x => TorchLean.betainc (x 0) (x 1) (x 2)

/-- A special function of one argument as a function of a real number. -/
def real₁ (f : SpecialFn 1) (t : ℝ) : ℝ :=
  f.real fun _ => t

/-- The `torch.special` name of each function, and `betainc` for `lax.betainc`. -/
def torchName : {n : ℕ} → SpecialFn n → String
  | _, erfc => "erfc"
  | _, erfcx => "erfcx"
  | _, erfinv => "erfinv"
  | _, ndtr => "ndtr"
  | _, ndtri => "ndtri"
  | _, lgamma => "gammaln"
  | _, digamma => "digamma"
  | _, besselJ0 => "bessel_j0"
  | _, besselJ1 => "bessel_j1"
  | _, besselI0 => "i0"
  | _, besselI1 => "i1"
  | _, besselI0e => "i0e"
  | _, besselI1e => "i1e"
  | _, sphericalBesselJ0 => "spherical_bessel_j0"
  | _, igamma => "gammainc"
  | _, igammac => "gammaincc"
  | _, zeta => "zeta"
  | _, betainc => "betainc"

/-- The special functions of `n` arguments. -/
def all : (n : ℕ) → List (SpecialFn n)
  | 1 => [erfc, erfcx, erfinv, ndtr, ndtri, lgamma, digamma, besselJ0, besselJ1, besselI0,
      besselI1, besselI0e, besselI1e, sphericalBesselJ0]
  | 2 => [igamma, igammac, zeta]
  | 3 => [betainc]
  | _ => []

/-- The function of `n` arguments with a given name, the inverse of `torchName`. -/
def ofTorchName? (n : ℕ) (name : String) : Option (SpecialFn n) :=
  (all n).find? (·.torchName == name)

/-- Reading back the name of a function gives the function. -/
@[simp] theorem ofTorchName?_torchName {n : ℕ} (f : SpecialFn n) :
    ofTorchName? n f.torchName = some f := by
  cases f <;> rfl

end SpecialFn

/-- The special functions and their partial derivatives of every order, for scalar types that
have them. -/
class Special (α : Type) where
  /-- The partial derivative of the special function `f` at `x` of order `k i` in argument `i`;
  order `0` is `f` itself. -/
  deriv {n : ℕ} : SpecialFn n → (Fin n → ℕ) → (Fin n → α) → α

/-- The sum of `t 0`, …, `t (n - 1)`, added from the left with no zero: `t 0` for one term,
`(t 0 + t 1) + t 2` for three. The tangents of the special functions add their terms in this
order, the order of the scalar's additions. -/
def Special.leftSum {α : Type} [Add α] [Zero α] : {n : ℕ} → (Fin n → α) → α
  | 0, _ => 0
  | 1, t => t 0
  | n + 2, t => leftSum (fun i : Fin (n + 1) => t i.castSucc) + t (Fin.last (n + 1))

/-- The reals take the derivative of order `k i` along each axis `i` with Mathlib's `lineDeriv`,
the last axis first, and evaluate it at `x`. At order `0` this is `SpecialFn.real`. -/
instance : Special ℝ where
  deriv := fun {n} f k => (List.finRange n).foldr (fun i G =>
    (fun (H : (Fin n → ℝ) → ℝ) (q : Fin n → ℝ) => lineDeriv ℝ H q (Pi.single i 1))^[k i] G)
    f.real

/-- Every iterated derivative of a function smooth at `p` is smooth at `p`. -/
theorem contDiffAt_iterate_deriv {φ : ℝ → ℝ} {p : ℝ} (h : ContDiffAt ℝ ∞ φ p) :
    ∀ k : ℕ, ContDiffAt ℝ ∞ (deriv^[k] φ) p
  | 0 => h
  | k + 1 => by
    rw [Function.iterate_succ_apply']
    exact (contDiffAt_iterate_deriv h k).derivWithin (le_of_eq rfl)

section Along

open Filter Topology

variable {V : Type} [NormedAddCommGroup V] [NormedSpace ℝ V]

/-- The derivative of `G` along the direction `v`, at every point: Mathlib's `lineDeriv`. -/
private def along (v : V) (G : V → ℝ) (q : V) : ℝ := lineDeriv ℝ G q v

private theorem along_eq {G : V → ℝ} {q : V} (h : DifferentiableAt ℝ G q) (v : V) :
    along v G q = fderiv ℝ G q v :=
  h.lineDeriv_eq_fderiv

/-- A function differentiable at `p` has as derivative there the sum of its derivatives along the
vectors `e i` of a basis, each times the coordinate `π i` of that vector. -/
private theorem hasFDerivAt_along {ι : Type} [Fintype ι] {G : V → ℝ} {p : V}
    (h : DifferentiableAt ℝ G p) (e : ι → V) (π : ι → V →L[ℝ] ℝ)
    (he : ∀ v, ∑ i, π i v • e i = v) :
    HasFDerivAt G (∑ i, along (e i) G p • π i) p := by
  refine h.hasFDerivAt.congr_fderiv (ContinuousLinearMap.ext fun v => ?_)
  conv_lhs => rw [← he v, map_sum]
  rw [_root_.sum_apply]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [map_smul, _root_.smul_apply, smul_eq_mul, smul_eq_mul, along_eq h, mul_comm]

private theorem along_congr {G H : V → ℝ} {p : V} (h : G =ᶠ[𝓝 p] H) (v : V) :
    along v G =ᶠ[𝓝 p] along v H :=
  (eventually_eventually_nhds.2 h).mono fun q hq => (show G =ᶠ[𝓝 q] H from hq).lineDeriv_eq

private theorem iterate_along_congr {G H : V → ℝ} {p : V} (h : G =ᶠ[𝓝 p] H) (v : V) :
    ∀ m : ℕ, (along v)^[m] G =ᶠ[𝓝 p] (along v)^[m] H
  | 0 => h
  | m + 1 => by
    rw [Function.iterate_succ_apply', Function.iterate_succ_apply']
    exact along_congr (iterate_along_congr h v m) v

private theorem eventually_differentiableAt {G : V → ℝ} {p : V}
    (h : ContDiffAt ℝ ∞ G p) : ∀ᶠ q in 𝓝 p, DifferentiableAt ℝ G q :=
  ((h.of_le (by simp) : ContDiffAt ℝ 1 G p).eventually (by simp)).mono fun _ hq =>
    hq.differentiableAt (by simp)

private theorem contDiffAt_along {G : V → ℝ} {p : V} (h : ContDiffAt ℝ ∞ G p) (v : V) :
    ContDiffAt ℝ ∞ (along v G) p :=
  contDiffAt_infty.2 fun k => ContDiffAt.congr_of_eventuallyEq
    ((contDiffAt_infty.1 h (k + 1)).fderiv_right_succ.clm_apply contDiffAt_const)
    ((eventually_differentiableAt h).mono fun _ hq => along_eq hq v)

private theorem contDiffAt_iterate_along {G : V → ℝ} {p : V} (h : ContDiffAt ℝ ∞ G p) (v : V) :
    ∀ m : ℕ, ContDiffAt ℝ ∞ ((along v)^[m] G) p
  | 0 => h
  | m + 1 => by
    rw [Function.iterate_succ_apply']
    exact contDiffAt_along (contDiffAt_iterate_along h v m) v

/-- The derivative along `w` of the derivative along a fixed `v` is the second derivative at
`(w, v)`. -/
private theorem fderiv_fderiv_apply {G : V → ℝ} {q : V}
    (h : DifferentiableAt ℝ (fderiv ℝ G) q) (v w : V) :
    fderiv ℝ (fun r => fderiv ℝ G r v) q w = fderiv ℝ (fderiv ℝ G) q w v := by
  rw [fderiv_clm_apply h (differentiableAt_const v)]
  simp

/-- Schwarz's theorem: near a point where a function is smooth, its derivatives along two
directions commute, since its second derivatives are symmetric there
(`ContDiffAt.isSymmSndFDerivAt`). -/
private theorem along_along {G : V → ℝ} {p : V} (h : ContDiffAt ℝ ∞ G p) (v w : V) :
    along w (along v G) =ᶠ[𝓝 p] along v (along w G) := by
  have h₂ : ∀ᶠ q in 𝓝 p, ContDiffAt ℝ 2 G q :=
    (h.of_le (by simp) : ContDiffAt ℝ 2 G p).eventually (by simp)
  filter_upwards [h₂] with q hq
  have hd : ∀ᶠ r in 𝓝 q, DifferentiableAt ℝ G r :=
    (hq.eventually (by simp)).mono fun _ hr => hr.differentiableAt (by simp)
  have hdd : DifferentiableAt ℝ (fderiv ℝ G) q :=
    (hq.fderiv_right (m := 1) (by norm_num)).differentiableAt (by simp)
  have e₁ : along w (along v G) q = fderiv ℝ (fderiv ℝ G) q w v := by
    rw [(along_congr (hd.mono fun _ hr => along_eq hr v) w).eq_of_nhds,
      along_eq (hdd.clm_apply (differentiableAt_const _)), fderiv_fderiv_apply hdd]
  have e₂ : along v (along w G) q = fderiv ℝ (fderiv ℝ G) q v w := by
    rw [(along_congr (hd.mono fun _ hr => along_eq hr w) v).eq_of_nhds,
      along_eq (hdd.clm_apply (differentiableAt_const _)), fderiv_fderiv_apply hdd]
  rw [e₁, e₂]
  exact (hq.isSymmSndFDerivAt (by simp)).eq _ _

/-- The derivative along `w` commutes with every order of the one along `v`, near a point where
the function is smooth. -/
private theorem along_iterate_along {G : V → ℝ} {p : V} (h : ContDiffAt ℝ ∞ G p) (v w : V) :
    ∀ m : ℕ, along w ((along v)^[m] G) =ᶠ[𝓝 p] (along v)^[m] (along w G)
  | 0 => EventuallyEq.rfl
  | m + 1 => by
    rw [Function.iterate_succ_apply', Function.iterate_succ_apply']
    exact (along_along (contDiffAt_iterate_along h v m) v w).trans
      (along_congr (along_iterate_along h v w m) v)

/-- Order `m` along `v` at `q` is the `m`-th derivative of a function `g` that traces `G` on the
line through `q` along `v`. -/
private theorem iterate_along_apply (v : V) (G : V → ℝ) :
    ∀ (m : ℕ) {q : V} {g : ℝ → ℝ} {c : ℝ}, (∀ t, G (q + t • v) = g (c + t)) →
      (along v)^[m] G q = deriv^[m] g c
  | 0, q, g, c, hg => by simpa using hg 0
  | m + 1, q, g, c, hg => by
    rw [Function.iterate_succ_apply', Function.iterate_succ_apply']
    have h : ∀ t, (along v)^[m] G (q + t • v) = deriv^[m] g (c + t) := fun t =>
      iterate_along_apply v G m fun u => by simpa only [add_smul, add_assoc] using hg (t + u)
    show deriv (fun t => (along v)^[m] G (q + t • v)) 0 = _
    simp only [h, deriv_comp_const_add, add_zero]

/-- The derivatives along `e i` of order `k i` for the `i` in `l`, the last innermost. -/
private def alongAll {ι : Type} (e : ι → V) (k : ι → ℕ) (l : List ι) (G : V → ℝ) : V → ℝ :=
  l.foldr (fun i H => (along (e i))^[k i] H) G

private theorem contDiffAt_alongAll {ι : Type} {G : V → ℝ} {p : V} (h : ContDiffAt ℝ ∞ G p)
    (e : ι → V) (k : ι → ℕ) : ∀ l : List ι, ContDiffAt ℝ ∞ (alongAll e k l G) p
  | [] => h
  | i :: l => contDiffAt_iterate_along (contDiffAt_alongAll h e k l) (e i) (k i)

private theorem alongAll_congr {ι : Type} (e : ι → V) {k k' : ι → ℕ} (G : V → ℝ) :
    ∀ l : List ι, (∀ i ∈ l, k i = k' i) → alongAll e k l G = alongAll e k' l G
  | [], _ => rfl
  | i :: l, hk => by
    show (along (e i))^[k i] (alongAll e k l G) = (along (e i))^[k' i] (alongAll e k' l G)
    rw [hk i (List.mem_cons.2 (Or.inl rfl)),
      alongAll_congr e G l fun j hj => hk j (List.mem_cons.2 (Or.inr hj))]

private theorem alongAll_zero {ι : Type} (e : ι → V) (G : V → ℝ) :
    ∀ l : List ι, alongAll e 0 l G = G
  | [] => rfl
  | _ :: l => alongAll_zero e G l

/-- Order `1` along `e i` and `0` along the rest is the derivative along `e i`. -/
private theorem alongAll_single {ι : Type} [DecidableEq ι] (e : ι → V) (G : V → ℝ) {i : ι} :
    ∀ l : List ι, i ∈ l → l.Nodup → alongAll e (Pi.single i 1) l G = along (e i) G
  | [], hi, _ => by simp at hi
  | j :: l, hi, hl => by
    have hl' := List.nodup_cons.1 hl
    by_cases hij : j = i
    · rw [hij] at hl' ⊢
      have h0 : alongAll e (Pi.single i 1) l G = G :=
        (alongAll_congr e G l fun m hm => by
          simp [show m ≠ i from fun h => hl'.1 (h ▸ hm)]).trans (alongAll_zero e G l)
      show (along (e i))^[(Pi.single i 1 : ι → ℕ) i] (alongAll e (Pi.single i 1) l G) = _
      rw [h0, Pi.single_eq_same]
      rfl
    · have hi' : i ∈ l := (List.mem_cons.1 hi).resolve_left (Ne.symm hij)
      show (along (e j))^[(Pi.single i 1 : ι → ℕ) j] (alongAll e (Pi.single i 1) l G) = _
      rw [show (Pi.single i 1 : ι → ℕ) j = 0 by simp [hij]]
      exact alongAll_single e G l hi' hl'.2

/-- Near a point where `G` is smooth, the derivative along `e i` of the derivatives of orders `k`
is the derivatives of orders `k + Pi.single i 1`. Past the first axis this needs the derivatives
along two axes to commute (`along_iterate_along`). -/
private theorem along_alongAll {ι : Type} [DecidableEq ι] {G : V → ℝ} {p : V}
    (h : ContDiffAt ℝ ∞ G p) (e : ι → V) (k : ι → ℕ) {i : ι} :
    ∀ l : List ι, i ∈ l → l.Nodup →
      along (e i) (alongAll e k l G) =ᶠ[𝓝 p] alongAll e (k + Pi.single i 1) l G
  | [], hi, _ => by simp at hi
  | j :: l, hi, hl => by
    have hl' := List.nodup_cons.1 hl
    by_cases hij : j = i
    · rw [hij] at hl' ⊢
      have hk : alongAll e (k + Pi.single i 1) l G = alongAll e k l G :=
        alongAll_congr e G l fun m hm => by
          simp [show m ≠ i from fun h => hl'.1 (h ▸ hm)]
      refine Eventually.of_forall fun q => ?_
      show along (e i) ((along (e i))^[k i] (alongAll e k l G)) q =
        (along (e i))^[(k + Pi.single i 1 : ι → ℕ) i] (alongAll e (k + Pi.single i 1) l G) q
      rw [hk, Pi.add_apply, Pi.single_eq_same, Function.iterate_succ_apply']
    · have hi' : i ∈ l := (List.mem_cons.1 hi).resolve_left (Ne.symm hij)
      have hkj : (k + Pi.single i 1 : ι → ℕ) j = k j := by simp [hij]
      show along (e i) ((along (e j))^[k j] (alongAll e k l G)) =ᶠ[𝓝 p]
        (along (e j))^[(k + Pi.single i 1 : ι → ℕ) j] (alongAll e (k + Pi.single i 1) l G)
      rw [hkj]
      exact (along_iterate_along (contDiffAt_alongAll h e k l) (e j) (e i) (k j)).trans
        (iterate_along_congr (along_alongAll h e k l hi' hl'.2) (e j) (k j))

end Along

namespace SpecialFn

open Filter Topology

/-- Over the reals, order `k` is the derivatives of order `k i` along the axes `i`. -/
private theorem deriv_eq_alongAll {n : ℕ} (f : SpecialFn n) (k : Fin n → ℕ) :
    Special.deriv f k =
      alongAll (fun i => (Pi.single i 1 : Fin n → ℝ)) k (List.finRange n) f.real :=
  rfl

/-- Where the real special function `f` is smooth at `x`, its derivatives of orders
`k + Pi.single i 1` there are the partial derivatives of its derivative of order `k`, the
components of that one's derivative. Past the first argument this needs the derivatives in two
arguments to commute, which is Schwarz's theorem (Mathlib's `ContDiffAt.isSymmSndFDerivAt`). The
derivative theorem of `Op.special`, `TorchLean.Op.special_backward_eq_adjoint`, rests on it. -/
theorem hasFDerivAt_deriv {n : ℕ} (f : SpecialFn n) (k : Fin n → ℕ) {x : Fin n → ℝ}
    (h : ContDiffAt ℝ ∞ f.real x) :
    HasFDerivAt (Special.deriv f k) (∑ i, Special.deriv f (k + Pi.single i 1) x •
      (ContinuousLinearMap.proj i : (Fin n → ℝ) →L[ℝ] ℝ)) x := by
  rw [deriv_eq_alongAll f k]
  refine (hasFDerivAt_along ((contDiffAt_alongAll h _ k _).differentiableAt (by simp)) _
    (fun i => ContinuousLinearMap.proj i) fun v => (pi_eq_sum_univ' v).symm).congr_fderiv ?_
  exact Finset.sum_congr rfl fun i _ => congrArg (· • _)
    (along_alongAll h _ k _ (List.mem_finRange i) (List.nodup_finRange n)).eq_of_nhds

/-- Where the real special function `f` is differentiable at `x`, its derivatives of orders
`Pi.single i 1` there are its partial derivatives, the components of its derivative. This is
`hasFDerivAt_deriv` at order `0`, where differentiability is enough. -/
theorem hasFDerivAt_real {n : ℕ} (f : SpecialFn n) {x : Fin n → ℝ}
    (h : DifferentiableAt ℝ f.real x) :
    HasFDerivAt f.real (∑ i, Special.deriv f (Pi.single i 1) x •
      (ContinuousLinearMap.proj i : (Fin n → ℝ) →L[ℝ] ℝ)) x := by
  refine (hasFDerivAt_along h _ (fun i => ContinuousLinearMap.proj i)
    fun v => (pi_eq_sum_univ' v).symm).congr_fderiv ?_
  exact Finset.sum_congr rfl fun i _ => congrArg (· • _) (congrFun (alongAll_single
    (fun i => (Pi.single i 1 : Fin n → ℝ)) f.real (List.finRange n) (List.mem_finRange i)
    (List.nodup_finRange n)) x).symm

/-- Order `0` of a special function is the function. -/
@[simp] theorem deriv_zero {n : ℕ} (f : SpecialFn n) : Special.deriv f 0 = f.real :=
  alongAll_zero _ f.real (List.finRange n)

/-- Over the reals, order `k` of a special function of one argument is Mathlib's iterated
derivative `deriv^[k 0]` of `real₁`. -/
theorem deriv_eq_iterate (f : SpecialFn 1) (k : Fin 1 → ℕ) (x : Fin 1 → ℝ) :
    Special.deriv f k x = deriv^[k 0] f.real₁ (x 0) := by
  show (along (Pi.single 0 1 : Fin 1 → ℝ))^[k 0] f.real x = _
  exact iterate_along_apply _ f.real (k 0) fun t => congrArg f.real (funext fun j => by
    rw [Subsingleton.elim j 0]
    simp)

/-- Wherever order `k` of a special function of one argument is differentiable, order `k + 1` is
its derivative. -/
theorem hasDerivAt_deriv (f : SpecialFn 1) {k : Fin 1 → ℕ} {t : ℝ}
    (h : DifferentiableAt ℝ (deriv^[k 0] f.real₁) t) :
    HasDerivAt (fun t => Special.deriv f k fun _ => t)
      (Special.deriv f (k + Pi.single 0 1) fun _ => t) t := by
  simp only [deriv_eq_iterate, Pi.add_apply, Pi.single_eq_same, Function.iterate_succ_apply']
  exact h.hasDerivAt

/-- A set on which `f` is smooth: the points whose argument is in `(-1, 1)` for `erfinv`, in
`(0, 1)` for `ndtri`, off the poles `0, -1, -2, …` of `Γ` for `lgamma` and `digamma` and nonzero
for `i0e` and `i1e`, every point for the other functions of one argument, the points with both
arguments positive for `igamma` and `igammac`, the points `(s, q)` with `s > 1` and `q > 0` for
`zeta`, and the points `(a, b, x)` with `a` and `b` positive and `x ∈ (0, 1)` for `betainc`. It
need not be the largest such set. -/
def smoothDomain : {n : ℕ} → SpecialFn n → Set (Fin n → ℝ)
  | _, erfinv => {x | x 0 ∈ Set.Ioo (-1) 1}
  | _, ndtri => {x | x 0 ∈ Set.Ioo 0 1}
  | _, lgamma | _, digamma => {x | ∀ m : ℕ, x 0 ≠ -m}
  | _, besselI0e | _, besselI1e => {x | x 0 ≠ 0}
  | _, erfc | _, erfcx | _, ndtr | _, besselJ0 | _, besselJ1 | _, besselI0 | _, besselI1
  | _, sphericalBesselJ0 => Set.univ
  | _, igamma | _, igammac => {x | 0 < x 0 ∧ 0 < x 1}
  | _, zeta => {x | 1 < x 0 ∧ 0 < x 1}
  | _, betainc => {x | 0 < x 0 ∧ 0 < x 1 ∧ x 2 ∈ Set.Ioo 0 1}

/-- Every special function is smooth on `smoothDomain`. This is the premise of the derivative
certificate of `Op.special`, `TorchLean.Op.special_backward_eq_adjoint`. -/
theorem contDiffAt_real {n : ℕ} {f : SpecialFn n} {x : Fin n → ℝ} (hx : x ∈ f.smoothDomain) :
    ContDiffAt ℝ ∞ f.real x := by
  cases f with
  | erfc => exact contDiff_erfc.contDiffAt.comp x (contDiffAt_apply ℝ (fun _ => ℝ) 0 x)
  | erfcx => exact contDiff_erfcx.contDiffAt.comp x (contDiffAt_apply ℝ (fun _ => ℝ) 0 x)
  | erfinv => exact (contDiffAt_erfinv (y := x 0) hx).comp x (contDiffAt_apply ℝ (fun _ => ℝ) 0 x)
  | ndtr => exact contDiff_ndtr.contDiffAt.comp x (contDiffAt_apply ℝ (fun _ => ℝ) 0 x)
  | ndtri =>
    exact ContDiffAt.comp (g := TorchLean.ndtri) x (contDiffAt_ndtri hx)
      (contDiffAt_apply ℝ (fun _ => ℝ) 0 x)
  | lgamma => exact (contDiffAt_lgamma (x := x 0) hx).comp x (contDiffAt_apply ℝ (fun _ => ℝ) 0 x)
  | digamma => exact (contDiffAt_digamma (x := x 0) hx).comp x (contDiffAt_apply ℝ (fun _ => ℝ) 0 x)
  | besselJ0 => exact contDiff_besselJ0.contDiffAt.comp x (contDiffAt_apply ℝ (fun _ => ℝ) 0 x)
  | besselJ1 => exact contDiff_besselJ1.contDiffAt.comp x (contDiffAt_apply ℝ (fun _ => ℝ) 0 x)
  | besselI0 => exact contDiff_besselI0.contDiffAt.comp x (contDiffAt_apply ℝ (fun _ => ℝ) 0 x)
  | besselI1 => exact contDiff_besselI1.contDiffAt.comp x (contDiffAt_apply ℝ (fun _ => ℝ) 0 x)
  | besselI0e =>
    exact (contDiffAt_besselI0e (x := x 0) hx).comp x (contDiffAt_apply ℝ (fun _ => ℝ) 0 x)
  | besselI1e =>
    exact (contDiffAt_besselI1e (x := x 0) hx).comp x (contDiffAt_apply ℝ (fun _ => ℝ) 0 x)
  | sphericalBesselJ0 => exact contDiff_sinc.contDiffAt.comp x (contDiffAt_apply ℝ (fun _ => ℝ) 0 x)
  | igamma =>
    exact ContDiffAt.comp (g := Function.uncurry TorchLean.igamma) x
      (contDiffOn_igamma.contDiffAt (prod_mem_nhds (Ioi_mem_nhds hx.1) (Ioi_mem_nhds hx.2)))
      ((contDiffAt_apply ℝ (fun _ => ℝ) 0 x).prodMk (contDiffAt_apply ℝ (fun _ => ℝ) 1 x))
  | igammac =>
    exact ContDiffAt.comp (g := Function.uncurry TorchLean.igammac) x
      (contDiffOn_igammac.contDiffAt (prod_mem_nhds (Ioi_mem_nhds hx.1) (Ioi_mem_nhds hx.2)))
      ((contDiffAt_apply ℝ (fun _ => ℝ) 0 x).prodMk (contDiffAt_apply ℝ (fun _ => ℝ) 1 x))
  | zeta =>
    exact ContDiffAt.comp (g := Function.uncurry TorchLean.zeta) x
      (contDiffOn_zeta.contDiffAt (prod_mem_nhds (Ioi_mem_nhds hx.1) (Ioi_mem_nhds hx.2)))
      ((contDiffAt_apply ℝ (fun _ => ℝ) 0 x).prodMk (contDiffAt_apply ℝ (fun _ => ℝ) 1 x))
  | betainc =>
    exact ContDiffAt.comp (g := fun p : ℝ × ℝ × ℝ => TorchLean.betainc p.1 p.2.1 p.2.2) x
      (contDiffOn_betainc.contDiffAt (prod_mem_nhds (Ioi_mem_nhds hx.1)
        (prod_mem_nhds (Ioi_mem_nhds hx.2.1) (Ioo_mem_nhds hx.2.2.1 hx.2.2.2))))
      ((contDiffAt_apply ℝ (fun _ => ℝ) 0 x).prodMk ((contDiffAt_apply ℝ (fun _ => ℝ) 1 x).prodMk
        (contDiffAt_apply ℝ (fun _ => ℝ) 2 x)))

/-- A special function of one argument is smooth at `t` where `smoothDomain` holds the point
`fun _ => t`, the premise of `TorchLean.Dual.jet_special`. -/
theorem contDiffAt_real₁ {f : SpecialFn 1} {t : ℝ} (ht : (fun _ => t) ∈ f.smoothDomain) :
    ContDiffAt ℝ ∞ f.real₁ t :=
  (contDiffAt_real ht).comp t (contDiffAt_pi.2 fun _ => contDiffAt_id)

/-- Over the reals, order `k` of `digamma` is the polygamma function `ψ⁽ᵏ⁾`
(`torch.special.polygamma k`, `lax.polygamma`). -/
theorem deriv_digamma (k : Fin 1 → ℕ) (x : Fin 1 → ℝ) :
    Special.deriv digamma k x = polygamma (k 0) (x 0) := by
  rw [deriv_eq_iterate]
  show deriv^[k 0] TorchLean.digamma (x 0) = iteratedDeriv (k 0) TorchLean.digamma (x 0)
  rw [iteratedDeriv_eq_iterate]

/-- Over the reals, order `(1, 0)` of `igamma` is its partial derivative in the shape `a`,
`igammaGradA` (`lax.igamma_grad_a`). -/
theorem deriv_igamma (x : Fin 2 → ℝ) :
    Special.deriv igamma ![1, 0] x = igammaGradA (x 0) (x 1) := by
  show (along (Pi.single 0 1 : Fin 2 → ℝ))^[1] igamma.real x =
    deriv^[1] (fun a => TorchLean.igamma a (x 1)) (x 0)
  exact iterate_along_apply _ _ 1 fun t => by simp [SpecialFn.real]

end SpecialFn

namespace Tensor

variable {α : Type} [Storage α] {s : Shape}

/-- The partial derivative of order `k` of the special function `f` at the coordinates of the
tensors `xs i` (`torch.special`, `SpecialFn.torchName f`, at order `0`). -/
def special [Special α] {n : ℕ} (f : SpecialFn n) (k : Fin n → ℕ) (xs : Fin n → Tensor α s) :
    Tensor α s :=
  Rep.ofFn fun c => Special.deriv f k fun i => xs i c

/-- Entry `c` of `special f k xs` is `Special.deriv f k` at the entries `c` of the `xs i`. -/
@[simp] theorem special_apply [Special α] {n : ℕ} (f : SpecialFn n) (k : Fin n → ℕ)
    (xs : Fin n → Tensor α s) (c : s.Coord) :
    special f k xs c = Special.deriv f k fun i => xs i c :=
  Rep.get_ofFn _ c

end Tensor

end TorchLean
