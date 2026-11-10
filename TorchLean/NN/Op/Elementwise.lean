/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tactic.OpDeriv
public import TorchLean.Tensor.Op.Table.Elementwise
public import TorchLean.Tensor.Elementwise.Real

/-!
# Derivatives of interpolation, fused multiply-add, the Euclidean norm, the roots and GELU

Certificates that the stored backward of `Op.lerp`, `Op.addcmul`, `Op.addcdiv`, `Op.hypot`,
`Op.cbrt`, `Op.sqrt`, `Op.rsqrt` and `Op.gelu` is the adjoint of the Fréchet derivative of the real
forward.
Over `ℝ` the first five forwards are `a + w (b - a)`, `x + v a b`, `x + v (a / b)`, `√(a² + b²)`
and `∛x` at every entry. We register these forms with `@[op_deriv_simps]`, and `op_deriv` proves
the certificates from them.

`lerp` and `addcmul` are polynomial, so their certificates hold at every input. `addcdiv` is
differentiable in its three inputs exactly where `v = 0` or every entry of the divisor is nonzero,
in its first input and in its second input alone at every input, and in its divisor alone exactly
where `v = 0` or each coordinate has a nonzero divisor or a zero numerator. Its stored pullbacks
at a zero divisor entry are `δ`, `0` and `0` by Lean's `a / 0 = 0`, an infinity or a NaN in a
binary format, and a `convention` certificate records them at every input. `hypot` is
differentiable exactly where no coordinate has both entries zero. Its stored pullback is zero at
such a coordinate, and a `convention` certificate records the stored pullback at every input.

The real cube root `Context.cbrtReal` has the derivative `1 / (3 ∛x²)` off zero
(`Context.hasDerivAt_cbrtReal`) and none at zero (`Context.not_differentiableAt_cbrtReal_zero`).
So `Op.cbrt` is differentiable exactly where no entry is zero. Its stored pullback over the reals
is zero at a zero entry, which a `convention` certificate records. In a binary format it is
`δ / 0` there, an infinity or a NaN.

`TorchLean.NN.Op.Pointwise` certifies `Op.sqrt` and `Op.gelu` at each policy and formula
separately, and `Op.rsqrt` at positive entries, and the certificates here hold at all of them.
GELU is differentiable at every input. Both real square roots, of `x` and of `max x 0`, are zero
below zero, so they have the slope `1 / (2 √x)` at every nonzero `x`, zero below zero, and no
derivative at zero. Over the reals the stored backward of `Op.sqrt` is `δ / (2 √x)` at every input
and policy, zero at zero, which a `convention` certificate records. The real `(√x)⁻¹` of
`Op.rsqrt` is zero at zero and below, so it has the slope `-(1 / 2) (√x)⁻¹ ^ 3` at every nonzero
`x` and is not continuous at zero, and a `convention` certificate records its stored backward.
-/

@[expose] public section

open Filter Topology

namespace TorchLean

attribute [op_deriv_simps] Context.lerp_real Context.hypot_real

namespace Tensor

variable {s : Shape}

/-- Over the reals, `lerp a b w` is `a + w * (b - a)`. -/
@[op_deriv_simps] theorem lerp_real (a b w : Tensor ℝ s) : lerp a b w = a + w * (b - a) :=
  Rep.ext fun c => by
    simp only [lerp_apply, Context.lerp_real, Tensor.add_apply, Tensor.mul_apply, Rep.hSub_apply]

/-- Over the reals, `addcmul x a b v` is `x + v • (a * b)`. -/
@[op_deriv_simps] theorem addcmul_real (x a b : Tensor ℝ s) (v : ℝ) :
    addcmul x a b v = x + v • (a * b) :=
  Rep.ext fun c => by
    simp only [addcmul_apply, Context.addcmul, Tensor.add_apply, Rep.smul_apply, Tensor.mul_apply,
      smul_eq_mul, mul_assoc]

/-- Over the reals, `addcdiv x a b v` is `x + v • (a / b)`. -/
@[op_deriv_simps] theorem addcdiv_real (x a b : Tensor ℝ s) (v : ℝ) :
    addcdiv x a b v = x + v • (a / b) :=
  Rep.ext fun c => by
    simp only [addcdiv_apply, Context.addcdiv, Tensor.add_apply, Rep.smul_apply, Tensor.div_apply,
      smul_eq_mul]

/-- Over the reals, `hypot a b` is `√(a * a + b * b)` at every entry. -/
@[op_deriv_simps] theorem hypot_real (a b : Tensor ℝ s) :
    hypot a b = map Real.sqrt (a * a + b * b) :=
  Rep.ext fun c => by
    simp only [hypot_apply, Context.hypot_real, Tensor.map_apply, Tensor.add_apply,
      Tensor.mul_apply]

/-- Over the reals, the stored pullbacks of `hypot a b` are `g a / hypot a b` and
`g b / hypot a b`, zero at a coordinate where both entries are zero. -/
@[op_deriv_simps] theorem hypotBackward_real (a b g : Tensor ℝ s) :
    hypotBackward a b g = (g * a / hypot a b, g * b / hypot a b) := by
  have h : ∀ x r u : ℝ, Numeric.hypotBackward x r u = u * x / r := fun x r u => by
    unfold Numeric.hypotBackward
    split_ifs with hr hs
    · rw [beq_iff_eq.mp hr, div_zero, zero_mul]
    · rfl
    · exact absurd rfl hs
  exact Prod.ext
    (Rep.ext fun c => by
      simp only [hypotBackward, zipWith3_apply, h, Tensor.div_apply, Tensor.mul_apply,
        hypot_apply])
    (Rep.ext fun c => by
      simp only [hypotBackward, zipWith3_apply, h, Tensor.div_apply, Tensor.mul_apply,
        hypot_apply])

/-- Over the reals, the entrywise quotient `y / z` is `y` times the entrywise inverse of `z`,
with `0⁻¹ = 0`. -/
theorem div_eq_mul_map_inv (y z : Tensor ℝ s) : y / z = y * map (·⁻¹) z :=
  Rep.ext fun c => by
    simp only [Tensor.div_apply, Tensor.mul_apply, Tensor.map_apply, div_eq_mul_inv]

end Tensor

namespace Context

/-- The cube of the real cube root of `x` is `x`. -/
theorem cbrtReal_pow_three (x : ℝ) : cbrtReal x ^ 3 = x := by
  have h (y : ℝ) (hy : 0 ≤ y) : (y ^ (1 / 3 : ℝ)) ^ 3 = y := by
    rw [one_div]
    exact_mod_cast Real.rpow_inv_natCast_pow hy three_ne_zero
  unfold cbrtReal
  split_ifs with hx
  · rw [Odd.neg_pow (by decide : Odd 3), h _ (by linarith), neg_neg]
  · exact h _ (not_lt.mp hx)

/-- The real cube root of zero is zero. -/
@[simp] theorem cbrtReal_zero : cbrtReal 0 = 0 := by
  simp [cbrtReal]

/-- The real cube root is zero only at zero. -/
theorem cbrtReal_ne_zero {x : ℝ} (hx : x ≠ 0) : cbrtReal x ≠ 0 := fun h =>
  hx (by rw [← cbrtReal_pow_three x, h]; norm_num)

/-- Off zero the real cube root `y` of `x` has the derivative `1 / (3 y²)`. -/
theorem hasDerivAt_cbrtReal {x : ℝ} (hx : x ≠ 0) :
    HasDerivAt cbrtReal (1 / 3 * (1 / (cbrtReal x * cbrtReal x))) x := by
  have hy := cbrtReal_ne_zero hx
  have hcube := cbrtReal_pow_three x
  rcases hx.lt_or_gt with h | h
  · have he : cbrtReal =ᶠ[𝓝 x] fun t => -(-t) ^ (1 / 3 : ℝ) :=
      (eventually_lt_nhds h).mono fun t ht => ite_eq_left ht
    have hd := ((Real.hasDerivAt_rpow_const (p := 1 / 3) (Or.inl (by linarith : -x ≠ 0))).comp x
      (hasDerivAt_neg x)).neg
    refine (hd.congr_of_eventuallyEq he).congr_deriv ?_
    have hc : cbrtReal x = -(-x) ^ (1 / 3 : ℝ) := ite_eq_left h
    rw [Real.rpow_sub (by linarith), Real.rpow_one, ← neg_neg ((-x) ^ (1 / 3 : ℝ)), ← hc]
    have : -x = -(cbrtReal x ^ 3) := by rw [hcube]
    rw [this]
    field_simp
  · have he : cbrtReal =ᶠ[𝓝 x] fun t => t ^ (1 / 3 : ℝ) :=
      (eventually_gt_nhds h).mono fun t ht => ite_eq_right (not_lt.mpr ht.le)
    have hd := Real.hasDerivAt_rpow_const (p := 1 / 3) (Or.inl h.ne')
    refine (hd.congr_of_eventuallyEq he).congr_deriv ?_
    have hc : cbrtReal x = x ^ (1 / 3 : ℝ) := ite_eq_right (not_lt.mpr h.le)
    rw [Real.rpow_sub h, Real.rpow_one, ← hc]
    nth_rewrite 2 [← hcube]
    field_simp

/-- The real cube root has no derivative at zero. -/
theorem not_differentiableAt_cbrtReal_zero : ¬ DifferentiableAt ℝ cbrtReal 0 := fun h => by
  have hid : (fun t => cbrtReal t ^ 3) = id := funext cbrtReal_pow_three
  have h3 := h.hasDerivAt.fun_pow 3
  rw [hid] at h3
  have := h3.unique (hasDerivAt_id 0)
  simp at this

end Context

namespace Op

variable {s : Shape}

/-- The real forward of `Op.lerp` is differentiable in its three inputs at every input, and its
stored backward is the adjoint of the derivative. -/
@[op_cert TorchLean.Op.lerp derivative 0 1 2]
theorem lerp_backward_eq_adjoint (a b w δ : Tensor ℝ s) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ s × WithLp 2 (Tensor ℝ s × Tensor ℝ s)) =>
        (((Op.lerp (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (p.fst, p.snd.fst, p.snd.snd, ())).1)
      (WithLp.toLp 2 (a, WithLp.toLp 2 (b, w))) ∧
    WithLp.toLp 2
        ((((Op.lerp (s := s)).rule? (α := ℝ) |>.get rfl).backward (a, b, w, ()) (δ, ())).1,
        WithLp.toLp 2
          ((((Op.lerp (s := s)).rule? (α := ℝ) |>.get rfl).backward (a, b, w, ()) (δ, ())).2.1,
          (((Op.lerp (s := s)).rule? (α := ℝ) |>.get rfl).backward (a, b, w, ()) (δ, ())).2.2.1)) =
      (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ s × WithLp 2 (Tensor ℝ s × Tensor ℝ s)) =>
        (((Op.lerp (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (p.fst, p.snd.fst, p.snd.snd, ())).1)
        (WithLp.toLp 2 (a, WithLp.toLp 2 (b, w)))).adjoint δ := by
  op_deriv [Tensor.lerpBackward]

/-- The real forward of `Op.addcmul` is differentiable in its three tensor inputs at every input
and every frozen `value`, and its stored backward is the adjoint of the derivative. -/
@[op_cert TorchLean.Op.addcmul derivative 0 1 2]
theorem addcmul_backward_eq_adjoint (x a b : Tensor ℝ s) (value : Tensor ℝ []) (δ : Tensor ℝ s) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ s × WithLp 2 (Tensor ℝ s × Tensor ℝ s)) =>
        (((Op.addcmul (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (p.fst, p.snd.fst, p.snd.snd, value, ())).1)
      (WithLp.toLp 2 (x, WithLp.toLp 2 (a, b))) ∧
    WithLp.toLp 2
        ((((Op.addcmul (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, a, b, value, ())
          (δ, ())).1,
        WithLp.toLp 2
          ((((Op.addcmul (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, a, b, value, ())
            (δ, ())).2.1,
          (((Op.addcmul (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, a, b, value, ())
            (δ, ())).2.2.1)) =
      (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ s × WithLp 2 (Tensor ℝ s × Tensor ℝ s)) =>
        (((Op.addcmul (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (p.fst, p.snd.fst, p.snd.snd, value, ())).1)
        (WithLp.toLp 2 (x, WithLp.toLp 2 (a, b)))).adjoint δ := by
  op_deriv [Tensor.addcmulBackward]

/-- Where `value` is zero or every entry of the divisor `b` is nonzero, the real forward of
`Op.addcdiv` is differentiable in its three tensor inputs and its stored backward is the adjoint
of the derivative. This is the exact set: for a nonzero `value` the entry `x + value (a / b)` is
not continuous in `(a, b)` where the divisor entry is zero. -/
@[op_cert TorchLean.Op.addcdiv derivative 0 1 2]
theorem addcdiv_backward_eq_adjoint (x a b : Tensor ℝ s) (value : Tensor ℝ []) (δ : Tensor ℝ s)
    (h : value.item = 0 ∨ ∀ c, b c ≠ 0) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ s × WithLp 2 (Tensor ℝ s × Tensor ℝ s)) =>
        (((Op.addcdiv (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (p.fst, p.snd.fst, p.snd.snd, value, ())).1)
      (WithLp.toLp 2 (x, WithLp.toLp 2 (a, b))) ∧
    WithLp.toLp 2
        ((((Op.addcdiv (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, a, b, value, ())
          (δ, ())).1,
        WithLp.toLp 2
          ((((Op.addcdiv (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, a, b, value, ())
            (δ, ())).2.1,
          (((Op.addcdiv (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, a, b, value, ())
            (δ, ())).2.2.1)) =
      (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ s × WithLp 2 (Tensor ℝ s × Tensor ℝ s)) =>
        (((Op.addcdiv (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (p.fst, p.snd.fst, p.snd.snd, value, ())).1)
        (WithLp.toLp 2 (x, WithLp.toLp 2 (a, b)))).adjoint δ := by
  rcases h with h | h
  · op_deriv [Tensor.addcdivBackward, h, zero_smul]
  · op_deriv [Tensor.addcdivBackward]

/-- The real forward of `Op.addcdiv` is differentiable in its first input alone at every input,
and its stored pullback to that input is the adjoint of the derivative. At a zero divisor entry
the real forward rests on Lean's `a / 0 = 0`, where the float operator returns an infinity or a
NaN. -/
@[op_cert TorchLean.Op.addcdiv derivative 0]
theorem addcdiv_backward_eq_adjoint_input (x a b : Tensor ℝ s) (value : Tensor ℝ [])
    (δ : Tensor ℝ s) :
    DifferentiableAt ℝ (fun y => (((Op.addcdiv (s := s)).rule? (α := ℝ) |>.get rfl).forward
        (y, a, b, value, ())).1) x ∧
    (((Op.addcdiv (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, a, b, value, ())
        (δ, ())).1 =
      (fderiv ℝ (fun y => (((Op.addcdiv (s := s)).rule? (α := ℝ) |>.get rfl).forward
        (y, a, b, value, ())).1) x).adjoint δ := by
  op_deriv [Tensor.addcdivBackward]

/-- The real forward of `Op.addcdiv` is differentiable in its second input alone at every input,
and its stored pullback to that input is the adjoint of the derivative. At a zero divisor entry
the real forward and pullback rest on Lean's `a / 0 = 0`, where the float operator returns an
infinity or a NaN. -/
@[op_cert TorchLean.Op.addcdiv derivative 1]
theorem addcdiv_backward_eq_adjoint_numerator (x a b : Tensor ℝ s) (value : Tensor ℝ [])
    (δ : Tensor ℝ s) :
    DifferentiableAt ℝ (fun y => (((Op.addcdiv (s := s)).rule? (α := ℝ) |>.get rfl).forward
        (x, y, b, value, ())).1) a ∧
    (((Op.addcdiv (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, a, b, value, ())
        (δ, ())).2.1 =
      (fderiv ℝ (fun y => (((Op.addcdiv (s := s)).rule? (α := ℝ) |>.get rfl).forward
        (x, y, b, value, ())).1) a).adjoint δ := by
  op_deriv [Tensor.addcdivBackward, Tensor.div_eq_mul_map_inv]

/-- Where `value` is zero or each coordinate has a nonzero divisor entry or a zero numerator entry,
the real forward of `Op.addcdiv` is differentiable in its divisor alone, and its stored pullback to
the divisor is the adjoint of the derivative. This is the exact set: for a nonzero `value`, at a
coordinate with a zero divisor entry and a nonzero numerator entry, `x + value (a / t)` is not
continuous in `t`. -/
@[op_cert TorchLean.Op.addcdiv derivative 2]
theorem addcdiv_backward_eq_adjoint_divisor (x a b : Tensor ℝ s) (value : Tensor ℝ [])
    (δ : Tensor ℝ s) (h : value.item = 0 ∨ ∀ c, b c ≠ 0 ∨ a c = 0) :
    DifferentiableAt ℝ (fun y => (((Op.addcdiv (s := s)).rule? (α := ℝ) |>.get rfl).forward
        (x, a, y, value, ())).1) b ∧
    (((Op.addcdiv (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, a, b, value, ())
        (δ, ())).2.2.1 =
      (fderiv ℝ (fun y => (((Op.addcdiv (s := s)).rule? (α := ℝ) |>.get rfl).forward
        (x, a, y, value, ())).1) b).adjoint δ := by
  simp only [Op.rule?_addcdiv, Option.get_some, Tensor.addcdivBackward]
  have hd : ∀ c, HasDerivAt (fun t => x c + value.item * (a c / t))
      (-(value.item * (a c / b c / b c))) (b c) := fun c => by
    by_cases hv : value.item = 0
    · simp only [hv, zero_mul, add_zero, neg_zero]
      exact hasDerivAt_const _ _
    rcases h.resolve_left hv c with hb | ha
    · exact (((hasDerivAt_const (b c) (a c)).div (hasDerivAt_id' (b c)) hb).const_mul
        value.item |>.const_add (x c)).congr_deriv (by ring)
    · simp only [ha, zero_div, mul_zero, add_zero, neg_zero]
      exact hasDerivAt_const _ _
  have hF : ∀ y c, Tensor.addcdiv x a y value.item c = x c + value.item * (a c / y c) :=
    fun y c => by simp only [Tensor.addcdiv_apply, Context.addcdiv]
  refine (HasPullbackAt.pointwise _ _ hF hd (HasPullbackAt.id b)).differentiableAt_and
    (Tensor.Rep.ext fun c => ?_)
  simp only [Tensor.zipWith3_apply, slopeCLM_apply]
  ring

/-- At every real input, entry `c` of the stored pullbacks of `Op.addcdiv` is `δ c`,
`δ c * value / b c` and `-(δ c * value * (a c / b c / b c))`. At a zero divisor entry, where the
forward is not continuous in the divisor for a nonzero `value` and numerator entry, they are
`δ c`, `0` and `0` by Lean's `a / 0 = 0`, and the float operator returns an infinity or a NaN
there. -/
@[op_cert TorchLean.Op.addcdiv convention]
theorem addcdiv_backward_eq (x a b : Tensor ℝ s) (value : Tensor ℝ []) (δ : Tensor ℝ s)
    (c : s.Coord) :
    (((Op.addcdiv (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, a, b, value, ())
        (δ, ())).1 c = δ c ∧
      (((Op.addcdiv (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, a, b, value, ())
        (δ, ())).2.1 c = δ c * value.item / b c ∧
      (((Op.addcdiv (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, a, b, value, ())
        (δ, ())).2.2.1 c = -(δ c * value.item * (a c / b c / b c)) := by
  simp only [Op.rule?_addcdiv, Option.get_some, Tensor.addcdivBackward, Tensor.zipWith_apply,
    Tensor.zipWith3_apply, and_self]

/-- Where no coordinate has both entries zero, the real forward of `Op.hypot` is differentiable in
both inputs and its stored backward is the adjoint of the derivative. This is the exact set: at a
coordinate where both entries are zero, `√(a² + b²)` has no derivative in either entry. -/
@[op_cert TorchLean.Op.hypot derivative 0 1]
theorem hypot_backward_eq_adjoint (a b δ : Tensor ℝ s) (h : ∀ c, a c ≠ 0 ∨ b c ≠ 0) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
        (((Op.hypot (s := s)).rule? (α := ℝ) |>.get rfl).forward (p.fst, p.snd, ())).1)
      (WithLp.toLp 2 (a, b)) ∧
    WithLp.toLp 2
        ((((Op.hypot (s := s)).rule? (α := ℝ) |>.get rfl).backward (a, b, ()) (δ, ())).1,
        (((Op.hypot (s := s)).rule? (α := ℝ) |>.get rfl).backward (a, b, ()) (δ, ())).2.1) =
      (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
        (((Op.hypot (s := s)).rule? (α := ℝ) |>.get rfl).forward (p.fst, p.snd, ())).1)
        (WithLp.toLp 2 (a, b))).adjoint δ := by
  have h' : ∀ c, (a * a + b * b) c ≠ 0 := fun c => by
    simp only [Tensor.add_apply, Tensor.mul_apply]
    rcases h c with h | h
    · exact (add_pos_of_pos_of_nonneg (mul_self_pos.mpr h) (mul_self_nonneg _)).ne'
    · exact (add_pos_of_nonneg_of_pos (mul_self_nonneg _) (mul_self_pos.mpr h)).ne'
  op_deriv

/-- At every input, the stored pullbacks of `Op.hypot` are `δ a / hypot a b` and
`δ b / hypot a b`. At a coordinate where both entries are zero, where `hypot` has no derivative,
both are zero. -/
@[op_cert TorchLean.Op.hypot convention]
theorem hypot_backward_eq (a b δ : Tensor ℝ s) :
    (((Op.hypot (s := s)).rule? (α := ℝ) |>.get rfl).backward (a, b, ()) (δ, ())).1 =
        δ * a / Tensor.hypot a b ∧
      (((Op.hypot (s := s)).rule? (α := ℝ) |>.get rfl).backward (a, b, ()) (δ, ())).2.1 =
        δ * b / Tensor.hypot a b := by
  constructor <;> simp only [Op.rule?_hypot, Option.get_some, Tensor.hypotBackward_real]

/-- Over the reals, the rule of `Op.cbrt` is its table arm at `Context.cbrtReal`. -/
@[op_deriv_simps] theorem rule?_cbrt_real :
    (Op.cbrt (s := s)).rule? (α := ℝ) = some
      { forward := fun xs => (Tensor.map Context.cbrtReal xs.1, ())
        backward := fun xs g => (Tensor.zipWith (fun x seed => seed * (((1 / 3 : Rat) : ℝ) *
          (1 / (Context.cbrtReal x * Context.cbrtReal x)))) xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.zipWith (fun x dx => dx * (((1 / 3 : Rat) : ℝ) *
          (1 / (Context.cbrtReal x * Context.cbrtReal x)))) xs.1 dxs.1, ()) } :=
  Op.rule?_cbrt (α := ℝ) (s := s)

/-- Where no entry is zero, the real forward of `Op.cbrt` is differentiable and its stored
backward `δ / (3 ∛x²)` is the adjoint of the derivative. This is the exact set: the cube root has
no derivative at zero (`Context.not_differentiableAt_cbrtReal_zero`). -/
@[op_cert TorchLean.Op.cbrt derivative]
theorem cbrt_backward_eq_adjoint (x δ : Tensor ℝ s) (hx : ∀ c, x c ≠ 0) :
    DifferentiableAt ℝ (fun y => (((Op.cbrt (s := s)).rule? (α := ℝ) |>.get rfl).forward
        (y, ())).1) x ∧
      (((Op.cbrt (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.cbrt (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ := by
  have hd : ∀ c, HasDerivAt Context.cbrtReal
      (1 / 3 * (1 / (Context.cbrtReal (x c) * Context.cbrtReal (x c)))) (x c) :=
    fun c => Context.hasDerivAt_cbrtReal (hx c)
  op_deriv

/-- At every real input, entry `c` of the stored backward of `Op.cbrt` is `δ c / (3 ∛(x c)²)`. It
is zero where `x c` is zero, where the cube root has no derivative, and in a binary format an
infinity or a NaN there. -/
@[op_cert TorchLean.Op.cbrt convention]
theorem cbrt_backward_eq (x δ : Tensor ℝ s) :
    (((Op.cbrt (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
      Tensor.zipWith (fun t d => d / (3 * (Context.cbrtReal t * Context.cbrtReal t))) x δ := by
  simp only [rule?_cbrt_real, Option.get_some]
  congr 1
  funext t d
  push_cast
  ring

/-- At either formula, the real forward of `Op.gelu` is differentiable at every input and its
stored backward is the adjoint of the derivative. -/
@[op_cert TorchLean.Op.gelu derivative]
theorem gelu_approximate_backward_eq_adjoint (approximate : GeluApproximation)
    (x δ : Tensor ℝ s) :
    DifferentiableAt ℝ (fun y => (((Op.gelu (s := s) approximate).rule? (α := ℝ) |>.get
        (by cases approximate <;> rfl)).forward (y, ())).1) x ∧
      (((Op.gelu (s := s) approximate).rule? (α := ℝ) |>.get
          (by cases approximate <;> rfl)).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.gelu (s := s) approximate).rule? (α := ℝ) |>.get
          (by cases approximate <;> rfl)).forward (y, ())).1) x).adjoint δ := by
  cases approximate
  · exact (gelu_backward_eq_adjoint x δ).imp_left HasFDerivAt.differentiableAt
  · exact (gelu_tanh_backward_eq_adjoint x δ).imp_left HasFDerivAt.differentiableAt

/-- At either policy, where no entry is zero, `Op.sqrt` has the slope map of `1 / (2 √x)` as
derivative, zero below zero where both real forwards vanish since Mathlib sets `√x = 0` for
`x ≤ 0`, and its stored backward is the adjoint of the derivative. This is the exact set, since at
zero the root has no derivative. -/
@[op_cert TorchLean.Op.sqrt derivative]
theorem sqrt_policy_backward_eq_adjoint (policy : Tensor.SqrtPolicy) (x δ : Tensor ℝ s)
    (hx : ∀ c, x c ≠ 0) :
    HasFDerivAt (fun y => (((Op.sqrt (s := s) policy).rule? (α := ℝ) |>.get
          (by cases policy <;> rfl)).forward (y, ())).1)
        (slopeCLM fun c => 1 / (2 * √(x c))) x ∧
      (((Op.sqrt (s := s) policy).rule? (α := ℝ) |>.get (by cases policy <;> rfl)).backward
          (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.sqrt (s := s) policy).rule? (α := ℝ) |>.get
          (by cases policy <;> rfl)).forward (y, ())).1) x).adjoint δ := by
  cases policy
  · exact sqrt_backward_eq_adjoint x δ hx
  have hmax : (fun z : ℝ => √(Max.max z 0)) = Real.sqrt := funext fun z => by
    rcases le_total z 0 with h | h
    · rw [max_eq_right h, Real.sqrt_zero, Real.sqrt_eq_zero_of_nonpos h]
    · rw [max_eq_left h]
  refine pointwise_backward_eq_adjoint _ (fun _ z => √(Max.max z 0)) (fun c => 1 / (2 * √(x c)))
    x δ _ (fun y c => ?_) (fun c => ?_) (fun c => ?_)
  · show (Tensor.sqrt y .clampNegative) c = _
    exact Tensor.sqrt_clampNegative_apply y c
  · show HasDerivAt (fun z : ℝ => √(Max.max z 0)) _ _
    rw [hmax]
    exact Real.hasDerivAt_sqrt (hx c)
  · show (Tensor.zipWith _ x δ) c = _
    rw [Tensor.zipWith_apply]
    show (if x c > 0 then _ else _) = _
    split_ifs with h
    · show δ c / (((2 : ℕ) : ℝ) * √(x c)) = _
      rw [Nat.cast_ofNat]
      ring
    · rw [Real.sqrt_eq_zero_of_nonpos (not_lt.mp h), mul_zero, div_zero, zero_mul]

/-- At every real input and either policy, entry `c` of the stored backward of `Op.sqrt` is
`δ c / (2 √(x c))`. It is zero where `x c` is zero, where the root has no derivative, and where
`x c` is negative, and in a binary format the raw policy gives an infinity at a zero entry and a
NaN at a negative one. -/
@[op_cert TorchLean.Op.sqrt convention]
theorem sqrt_backward_eq (policy : Tensor.SqrtPolicy) (x δ : Tensor ℝ s) :
    (((Op.sqrt (s := s) policy).rule? (α := ℝ) |>.get (by cases policy <;> rfl)).backward
        (x, ()) (δ, ())).1 =
      Tensor.zipWith (fun t d => d / (2 * √t)) x δ := by
  refine Tensor.Rep.ext fun c => ?_
  cases policy
  · show (Tensor.zipWith _ x δ) c = _
    rw [Tensor.zipWith_apply, Tensor.zipWith_apply]
    show δ c / (((2 : ℕ) : ℝ) * |√(x c)|) = _
    rw [abs_of_nonneg (Real.sqrt_nonneg _), Nat.cast_ofNat]
  · show (Tensor.zipWith _ x δ) c = _
    rw [Tensor.zipWith_apply, Tensor.zipWith_apply]
    show (if x c > 0 then _ else _) = _
    split_ifs with h
    · show δ c / (((2 : ℕ) : ℝ) * √(x c)) = _
      rw [Nat.cast_ofNat]
    · rw [Real.sqrt_eq_zero_of_nonpos (not_lt.mp h), mul_zero, div_zero, zero_mul]

/-- Where no entry is zero, `Op.rsqrt` has the slope map of `-(1 / 2) (√x)⁻¹ ^ 3` as derivative,
zero below zero where the real forward `(√x)⁻¹` vanishes since Mathlib sets `√x = 0` for `x ≤ 0`
and `0⁻¹ = 0`, and its stored backward is the adjoint of the derivative. This is the exact set,
since at zero the forward is not continuous. -/
@[op_cert TorchLean.Op.rsqrt derivative]
theorem rsqrt_backward_eq_adjoint_of_ne_zero (x δ : Tensor ℝ s) (hx : ∀ c, x c ≠ 0) :
    HasFDerivAt (fun y => (((Op.rsqrt (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (slopeCLM fun c => -(1 / 2) * (√(x c))⁻¹ ^ 3) x ∧
      (((Op.rsqrt (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.rsqrt (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ := by
  refine pointwise_backward_eq_adjoint _ (fun _ z => (√z)⁻¹) (fun c => -(1 / 2) * (√(x c))⁻¹ ^ 3)
    x δ _ (fun y c => ?_) (fun c => ?_) (fun c => ?_)
  · show (Tensor.rsqrt y) c = _
    rw [Tensor.rsqrt_apply]
    exact one_div _
  · rcases (hx c).lt_or_gt with h | h
    · rw [Real.sqrt_eq_zero_of_nonpos h.le, inv_zero]
      refine ((hasDerivAt_const (x c) (0 : ℝ)).congr_deriv (by norm_num)).congr_of_eventuallyEq ?_
      filter_upwards [eventually_lt_nhds h] with z hz
      rw [Real.sqrt_eq_zero_of_nonpos hz.le, inv_zero]
    · have h' : HasDerivAt (fun z => (√z)⁻¹) (-(1 / (2 * √(x c))) / √(x c) ^ 2) (x c) :=
        (Real.hasDerivAt_sqrt h.ne').inv (Real.sqrt_pos.mpr h).ne'
      exact h'.congr_deriv (by ring)
  · show (Tensor.rsqrtBackward x δ) c = _
    simp only [Tensor.rsqrtBackward, Tensor.zipWith_apply]
    show -((1 / 2 : ℚ) : ℝ) * δ c * (1 / √(x c) * (1 / √(x c)) * (1 / √(x c))) = _
    push_cast
    ring

/-- At every real input, entry `c` of the stored backward of `Op.rsqrt` is
`-(1 / 2) δ c (√(x c))⁻¹ ^ 3`. It is zero where `x c` is zero, where the forward has no
derivative, and where `x c` is negative, and in a binary format an infinity at a zero entry and a
NaN at a negative one. -/
@[op_cert TorchLean.Op.rsqrt convention]
theorem rsqrt_backward_eq (x δ : Tensor ℝ s) :
    (((Op.rsqrt (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
      Tensor.zipWith (fun t d => -(1 / 2) * d * (√t)⁻¹ ^ 3) x δ := by
  refine Tensor.Rep.ext fun c => ?_
  show (Tensor.rsqrtBackward x δ) c = _
  simp only [Tensor.rsqrtBackward, Tensor.zipWith_apply]
  show -((1 / 2 : ℚ) : ℝ) * δ c * (1 / √(x c) * (1 / √(x c)) * (1 / √(x c))) = _
  push_cast
  ring

end Op

end TorchLean
