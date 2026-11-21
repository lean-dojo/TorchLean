/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Op.Pointwise
public import TorchLean.Tensor.Special.Fn

/-!
# The derivative certificate and the derivative theorems of the special functions

Over the reals `Op.special orders f` is the partial derivative `Special.deriv f orders.get` of
`f.real` at each tuple of entries of its inputs, of order `orders.get i` in input `i`, and its
pullback to input `i` multiplies the cotangent by orders `orders.get + Pi.single i 1`. Where
`f.real` is smooth at every tuple of entries, those are the components of the derivative of order
`orders.get`, since there the derivatives in different arguments commute
(`SpecialFn.hasFDerivAt_deriv`). The pullback is then the adjoint of the derivative of the forward
map, jointly in the inputs, on their L2 product (`special_backward_eq_adjoint_of_contDiffAt`). At
orders all `0` differentiability of `f.real` at every tuple of entries is enough
(`SpecialFn.hasFDerivAt_real`, `special_backward_eq_adjoint_of_differentiableAt`). The theorems
come from one statement that takes the derivative at each entry
(`pointwisePi_backward_eq_adjoint`). For a function of one argument at any order `k`,
differentiability of `deriv^[k] f.real₁` at every entry is enough: the forward then has the slope
map of order `k + 1` as derivative and its pullback is that map's adjoint
(`special₁_backward_eq_adjoint`, from `SpecialFn.hasDerivAt_deriv` and
`pointwise_backward_eq_adjoint`).

The derivative certificate (`op_cert`) is `special_backward_eq_adjoint`. Its premise is that every
tuple of entries lies in `SpecialFn.smoothDomain f`, where `SpecialFn.contDiffAt_real` proves
`f.real` smooth: everywhere for most functions of one argument, off the poles of `Γ` for `lgamma`
and `digamma`, on `(-1, 1)` for `erfinv`, on `(0, 1)` for `ndtri`, away from `0` for `i0e` and
`i1e`, where both arguments are positive for `igamma` and `igammac`, at the points `(s, q)` with
`s > 1` and `q > 0` for `zeta`, and at the points `(a, b, x)` with `a` and `b` positive and
`x ∈ (0, 1)` for `betainc`.
-/

@[expose] public section

namespace TorchLean.Op

open TorchLean.Tensor
open scoped ContDiff

variable {s : Shape}

/-- The pullback of `Op.special orders f` is the adjoint of the derivative of its forward, jointly
in its inputs, wherever orders `orders.get + Pi.single i 1` of `f` are the components of the
derivative of order `orders.get` at every tuple of entries. The derivative theorems of
`Op.special` follow from it. -/
private theorem special_backward_of_hasFDerivAt (orders : List ℕ) (f : SpecialFn orders.length)
    (xs : Values ℝ (orders.map fun _ => .value s)) (δ : Tensor ℝ s)
    (h : ∀ c, HasFDerivAt (Special.deriv f orders.get)
      (∑ i, Special.deriv f (orders.get + Pi.single i 1)
          (fun j => TProd.parts (fun _ => Port.value s) orders xs j c) •
        (ContinuousLinearMap.proj i : (Fin orders.length → ℝ) →L[ℝ] ℝ))
      fun j => TProd.parts (fun _ => Port.value s) orders xs j c) :
    DerivCert (fun p : PiLp 2 (fun _ : Fin orders.length => Tensor ℝ s) =>
        (((Op.special orders f (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (TProd.ofParts (fun _ => Port.value s) orders p.ofLp)).1)
      (WithLp.toLp 2 (TProd.parts (fun _ => Port.value s) orders xs)) δ
      (WithLp.toLp 2 (TProd.parts (fun _ => Port.value s) orders
        (((Op.special orders f (s := s)).rule? (α := ℝ) |>.get rfl).backward xs (δ, ())))) := by
  have hp := pointwisePi_backward_eq_adjoint
    (fun p : PiLp 2 (fun _ : Fin orders.length => Tensor ℝ s) =>
      (((Op.special orders f (s := s)).rule? (α := ℝ) |>.get rfl).forward
        (TProd.ofParts (fun _ => Port.value s) orders p.ofLp)).1)
    (fun _ => Special.deriv f orders.get)
    (fun i c => Special.deriv f (orders.get + Pi.single i 1)
      fun j => TProd.parts (fun _ => Port.value s) orders xs j c)
    (TProd.parts (fun _ => Port.value s) orders xs) δ
    (TProd.parts (fun _ => Port.value s) orders
      (((Op.special orders f (s := s)).rule? (α := ℝ) |>.get rfl).backward xs (δ, ())))
    (fun p c => by
      show Tensor.special f orders.get (TProd.parts (β := Port.Carrier ℝ) (fun _ => Port.value s)
        orders (TProd.ofParts (β := Port.Carrier ℝ) (fun _ => Port.value s) orders p.ofLp)) c = _
      simp only [TProd.parts_ofParts, Tensor.special_apply])
    h
    (fun i c => by
      show TProd.parts (β := Port.Tangent ℝ) (fun _ => Port.value s) orders
        (TProd.ofParts (β := Port.Tangent ℝ) (fun _ => Port.value s) orders fun i =>
          Tensor.Rep.ofFn fun c => δ c * Special.deriv f (orders.get + Pi.single i 1)
            fun j => TProd.parts (fun _ => Port.value s) orders xs j c) i c = _
      simp only [TProd.parts_ofParts, Tensor.Rep.get_ofFn]
      exact mul_comm _ _)
  exact ⟨hp.1.differentiableAt, hp.2⟩

/-- Where the real special function `f` is smooth at every tuple of entries of the inputs, the
forward of `Op.special orders f` is differentiable at the inputs and its pullback of `δ` is the
adjoint of its derivative applied to `δ`, jointly in the inputs. -/
theorem special_backward_eq_adjoint_of_contDiffAt (orders : List ℕ) (f : SpecialFn orders.length)
    (xs : Values ℝ (orders.map fun _ => .value s)) (δ : Tensor ℝ s)
    (h : ∀ c, ContDiffAt ℝ ∞ f.real fun i => TProd.parts (fun _ => Port.value s) orders xs i c) :
    DerivCert (fun p : PiLp 2 (fun _ : Fin orders.length => Tensor ℝ s) =>
        (((Op.special orders f (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (TProd.ofParts (fun _ => Port.value s) orders p.ofLp)).1)
      (WithLp.toLp 2 (TProd.parts (fun _ => Port.value s) orders xs)) δ
      (WithLp.toLp 2 (TProd.parts (fun _ => Port.value s) orders
        (((Op.special orders f (s := s)).rule? (α := ℝ) |>.get rfl).backward xs (δ, ())))) :=
  special_backward_of_hasFDerivAt orders f xs δ fun c =>
    SpecialFn.hasFDerivAt_deriv f orders.get (h c)

/-- The derivative certificate of `Op.special`: where every tuple of entries of the inputs lies in
`SpecialFn.smoothDomain f`, the pullback of `Op.special orders f` is the adjoint of the derivative
of its forward, jointly in the inputs. -/
@[op_cert TorchLean.Op.special derivative]
theorem special_backward_eq_adjoint (orders : List ℕ) (f : SpecialFn orders.length)
    (xs : Values ℝ (orders.map fun _ => .value s)) (δ : Tensor ℝ s)
    (hx : ∀ c, (fun i => TProd.parts (fun _ => Port.value s) orders xs i c) ∈ f.smoothDomain) :
    DerivCert (fun p : PiLp 2 (fun _ : Fin orders.length => Tensor ℝ s) =>
        (((Op.special orders f (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (TProd.ofParts (fun _ => Port.value s) orders p.ofLp)).1)
      (WithLp.toLp 2 (TProd.parts (fun _ => Port.value s) orders xs)) δ
      (WithLp.toLp 2 (TProd.parts (fun _ => Port.value s) orders
        (((Op.special orders f (s := s)).rule? (α := ℝ) |>.get rfl).backward xs (δ, ())))) :=
  special_backward_eq_adjoint_of_contDiffAt orders f xs δ fun c =>
    SpecialFn.contDiffAt_real (hx c)

/-- At orders all `0`, differentiability is enough: where the real special function `f` is
differentiable at every tuple of entries of the inputs, the forward of `Op.special orders f` is
differentiable at the inputs and its pullback of `δ` is the adjoint of its derivative applied to
`δ`, jointly in the inputs. -/
theorem special_backward_eq_adjoint_of_differentiableAt (orders : List ℕ)
    (f : SpecialFn orders.length) (xs : Values ℝ (orders.map fun _ => .value s)) (δ : Tensor ℝ s)
    (h0 : ∀ i, orders.get i = 0)
    (h : ∀ c, DifferentiableAt ℝ f.real
      fun i => TProd.parts (fun _ => Port.value s) orders xs i c) :
    DerivCert (fun p : PiLp 2 (fun _ : Fin orders.length => Tensor ℝ s) =>
        (((Op.special orders f (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (TProd.ofParts (fun _ => Port.value s) orders p.ofLp)).1)
      (WithLp.toLp 2 (TProd.parts (fun _ => Port.value s) orders xs)) δ
      (WithLp.toLp 2 (TProd.parts (fun _ => Port.value s) orders
        (((Op.special orders f (s := s)).rule? (α := ℝ) |>.get rfl).backward xs (δ, ())))) :=
  special_backward_of_hasFDerivAt orders f xs δ fun c => by
    have hk : orders.get = 0 := funext h0
    simpa only [hk, zero_add, SpecialFn.deriv_zero] using SpecialFn.hasFDerivAt_real f (h c)

/-- For a function of one argument, at any order `k`, differentiability of its derivative of
order `k` is enough: where `deriv^[k] f.real₁` is differentiable at every entry of `x`, the
forward of `Op.special [k] f` has the slope map of order `k + 1` at the entries as derivative at
`x`, and its pullback of `δ` is the adjoint of that derivative applied to `δ`. It needs no
smoothness, unlike the certificate, and comes from `SpecialFn.hasDerivAt_deriv`. -/
theorem special₁_backward_eq_adjoint (f : SpecialFn 1) (k : ℕ) (x δ : Tensor ℝ s)
    (hx : ∀ c, DifferentiableAt ℝ (deriv^[k] f.real₁) (x c)) :
    HasFDerivAt (fun y => (((Op.special [k] f (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (slopeCLM fun c => Special.deriv f (fun _ => k + 1) fun _ => x c) x ∧
      (((Op.special [k] f (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.special [k] f (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ := by
  have one (i : Fin 1) : i = 0 := Subsingleton.elim _ _
  have hk : ([k].get + Pi.single 0 1 : Fin 1 → ℕ) = fun _ => k + 1 :=
    funext fun i => by rw [one i]; simp
  have hp (y : Tensor ℝ s) (c : s.Coord) :
      (fun j => TProd.parts (β := Port.Carrier ℝ) (fun _ => Port.value s) [k] (y, ()) j c) =
        fun _ => y c :=
    funext fun j => by rw [one j]; rfl
  refine pointwise_backward_eq_adjoint _ (fun _ t => Special.deriv f (fun _ => k) fun _ => t)
    (fun c => Special.deriv f (fun _ => k + 1) fun _ => x c) x δ _ (fun y c => ?_)
    (fun c => ?_) (fun c => ?_)
  · show Tensor.special (n := 1) f [k].get
      (TProd.parts (β := Port.Carrier ℝ) (fun _ => Port.value s) [k] (y, ())) c = _
    rw [Tensor.special_apply]
    exact congrArg₂ _ (funext fun i => by rw [one i]; rfl) (hp y c)
  · have h := SpecialFn.hasDerivAt_deriv f (k := fun _ => k) (hx c)
    have e : ((fun _ => k) + Pi.single 0 1 : Fin 1 → ℕ) = fun _ => k + 1 :=
      funext fun i => by rw [one i]; simp
    rwa [e] at h
  · show (Tensor.Rep.ofFn fun c => δ c * Special.deriv (n := 1) f ([k].get + Pi.single 0 1)
      fun j => TProd.parts (β := Port.Carrier ℝ) (fun _ => Port.value s) [k] (x, ()) j c) c = _
    rw [Tensor.Rep.get_ofFn, hk]
    exact (mul_comm _ _).trans (congrArg (· * δ c) (congrArg _ (hp x c)))

end TorchLean.Op
