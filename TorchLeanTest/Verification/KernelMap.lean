/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.CUDA.Kernel.Map
public import TorchLeanTest.Verification.GradientKernel
import TorchLean.CUDA.Kernel.Elab

/-!
# The proved gradient of a user program

We write the entrywise function `x * x + 1` with `Program.of`, once over the configured binary32
format and once over Lean's native `Float32`. We read the gradient kernel of each program as a
real expression, through `ExecFloat.Binary.toReal` for binary32 and through the binary32 value of
each `Float32` for the native program. By `Expr.map_gradient_eq_adjoint`, at every point `θ₀` and
every cotangent `cot` this reading computes the adjoint of the derivative of `θ ↦ θ * θ + 1` at
`cot`, which is `2 θ₀ a * cot a` at entry `a`. By `Expr.hasDerivAt_evaluate_map_tangent`, the
reading of the tangent kernel at an entry with primal value `x` and tangent `dx` computes
`2 x dx`.

The program `quotient` of the kernel README reads `a` from input zero and `b` from input one and
computes `a * b / (a * a + 1)` over `Float32`. By `Expr.map_gradient_eq_adjoint_withInputs`, at
every point and cotangent the real reading of its gradient kernel in input zero computes
`b (1 - a²) / (a² + 1)²` times the cotangent, and in input one `a / (a² + 1)` times the cotangent.
These are the two blocks of the adjoint of the derivative of `(a, b) ↦ a b / (a² + 1)`.

These statements are about real arithmetic. A run of the kernel in binary32 or `Float32` rounds
every operation, and we prove nothing here about that rounding.
-/

@[expose] public section

open Filter hiding pure_bind
open Topology

namespace TorchLean.Test.KernelMap

open Kernel FloatLib.Floats
open GradientKernel (oneInput gradientInputs)
open scoped Kernel

/-- `x * x + 1` at each entry of input zero, written with `Program.of` over the configured
binary32 format. -/
def parabola : Kernel.Program Binary32 :=
  Program.of (fun (x : Binary32) => x * x + 1)

/-- `x * x + 1` at each entry of input zero, written with `Program.of` over Lean's native
`Float32`. -/
def parabolaFloat32 : Kernel.Program Float32 :=
  Program.of (fun (x : Float32) => x * x + 1)

/-- The frontend turns `parabola` into a read of input zero at the output index, bound as a
local, followed by `x * x + 1`. -/
theorem parabola_expression : parabola.expression =
    .letIn (.load 0 (.var .zero))
      (.binary .add (.binary .mul (.var .zero) (.var .zero)) (.scalar 1)) := rfl

/-- The frontend turns `parabolaFloat32` into a read of input zero at the output index, bound as
a local, followed by `x * x + 1`. -/
theorem parabolaFloat32_expression : parabolaFloat32.expression =
    .letIn (.load 0 (.var .zero))
      (.binary .add (.binary .mul (.var .zero) (.var .zero)) (.scalar 1)) := rfl

/-- The map `θ ↦ θ * θ + 1`, entry by entry. -/
noncomputable def parabolaMap {m : UInt64} (θ : EuclideanSpace ℝ (Fin m.toNat)) :
    EuclideanSpace ℝ (Fin m.toNat) :=
  WithLp.toLp 2 fun r => θ r * θ r + 1

/-- The adjoint of the derivative of `parabolaMap` at `θ₀` multiplies entry `a` of the cotangent
by `2 θ₀ a`. -/
theorem adjoint_fderiv_parabolaMap_apply {m : UInt64} (θ₀ cot : EuclideanSpace ℝ (Fin m.toNat))
    (a : Fin m.toNat) : (fderiv ℝ parabolaMap θ₀).adjoint cot a = 2 * θ₀ a * cot a := by
  have hpi := hasFDerivAt_pi.mpr fun r : Fin m.toNat =>
    ((PiLp.hasFDerivAt_apply (𝕜 := ℝ) 2 θ₀ r).mul
      (PiLp.hasFDerivAt_apply (𝕜 := ℝ) 2 θ₀ r)).add_const (1 : ℝ)
  have hFD : HasFDerivAt (parabolaMap (m := m)) _ θ₀ :=
    (PiLp.hasFDerivAt_toLp (𝕜 := ℝ) 2 fun r => θ₀ r * θ₀ r + 1).comp θ₀ hpi
  have hentry (v : EuclideanSpace ℝ (Fin m.toNat)) :
      v a = inner ℝ (EuclideanSpace.single a 1) v := by
    rw [EuclideanSpace.inner_single_left, map_one, one_mul]
  rw [hentry ((fderiv ℝ parabolaMap θ₀).adjoint cot), ContinuousLinearMap.adjoint_inner_right,
    hFD.fderiv, PiLp.inner_apply, Finset.sum_eq_single a]
  · simp only [ContinuousLinearMap.comp_apply, ContinuousLinearMap.coe_pi', add_apply,
      smul_apply, PiLp.proj_apply, PiLp.single_apply, smul_eq_mul, mul_ite, mul_one, mul_zero,
      ContinuousLinearEquiv.coe_coe, PiLp.continuousLinearEquiv_symm_apply, ↓reduceIte,
      RCLike.inner_apply, conj_trivial]
    ring
  · intro b _ hb
    simp [hb]
  · intro h
    exact absurd (Finset.mem_univ a) h

/-- Let `f` read the constants of `α` as reals and keep zero and one, and let `e` read input zero
and compute `x * x + 1`. At every point `θ₀` and cotangent `cot`, `parabolaMap` is differentiable
at `θ₀`, and the real reading of the gradient kernel of `e` computes the adjoint of its derivative
at `cot`, which is `2 θ₀ a * cot a` at entry `a`. -/
theorem evaluate_map_gradient_parabola {α : Type} [OfNat α 0] [OfNat α 1] {f : α → ℝ}
    (hf0 : f 0 = 0) (hf1 : f 1 = 1) {e : Expr α [.index] .scalar}
    (he : e = .letIn (.load 0 (.var .zero))
      (.binary .add (.binary .mul (.var .zero) (.var .zero)) (.scalar 1)))
    (m : UInt64) (θ₀ cot : EuclideanSpace ℝ (Fin m.toNat)) :
    DifferentiableAt ℝ parabolaMap θ₀ ∧ ∀ a : Fin m.toNat,
      evaluate Arithmetic.ofOps ((e.gradient 1 0 m).map f) (gradientInputs θ₀ cot)
          a.val.toUInt64 = .ok ((fderiv ℝ parabolaMap θ₀).adjoint cot a) ∧
        (fderiv ℝ parabolaMap θ₀).adjoint cot a = 2 * θ₀ a * cot a := by
  have hr (r : Fin m.toNat) : (r.val.toUInt64).toNat = r.val := by
    rw [Nat.toUInt64_eq]
    exact UInt64.toNat_ofNat_of_lt' (lt_trans r.isLt m.toNat_lt_size)
  have hlt (r : Fin m.toNat) : r.val.toUInt64 < m := by
    rw [UInt64.lt_iff_toNat_lt, hr]
    exact r.isLt
  have hread (θ : EuclideanSpace ℝ (Fin m.toNat)) (r : Fin m.toNat) :
      (oneInput m).withValues (withInput (fun _ _ => 0) 0 θ) 0 r.val.toUInt64 = .ok (θ r) := by
    have hc : 0 = 0 ∧ (r.val.toUInt64).toNat < m.toNat := ⟨rfl, by rw [hr]; exact r.isLt⟩
    have hf : (⟨(r.val.toUInt64).toNat, hc.2⟩ : Fin m.toNat) = r := Fin.ext (hr r)
    simp only [Reader.withValues, oneInput, hlt r, and_self, ↓reduceIte, Except.map, withInput,
      hc.2, ↓reduceDIte, hf]
  refine (Expr.map_gradient_eq_adjoint hf0 hf1 e (k := 1) (j := 0) (read := oneInput m)
    (fun o i y h => ?_) (fun _ _ => 0) (θ₀ := θ₀) (F := parabolaMap)
    (Eventually.of_forall fun θ r => ?_) (fun r => ?_) (R := gradientInputs θ₀ cot)
    (fun o i y h => ?_) (cot := cot) (fun r => ?_)).imp_right
    fun h a => ⟨h a, adjoint_fderiv_parabolaMap_apply θ₀ cot a⟩
  · unfold oneInput at h
    split_ifs at h with hc
    · rw [hc.1]
      exact Nat.zero_lt_one
  · rw [he]
    simp only [evaluate, Expr.map_letIn, Expr.map_load, Expr.map_var, Expr.map_binary,
      Expr.map_scalar, hf1, Expr.eval, Env.get_output, ← Expr.except_pure, pure_bind, hread θ r,
      Env.get_push_zero, Arithmetic.ofOps_binary_mul, Arithmetic.ofOps_binary_add, parabolaMap,
      PiLp.toLp_apply]
  · rw [he]
    simp [Expr.Regular]
  · obtain ⟨z, hz, rfl⟩ := Reader.exists_ok_of_withValues h
    unfold oneInput at hz
    split_ifs at hz with hc
    obtain ⟨rfl, hi⟩ := hc
    have hi' : i.toNat < m.toNat := UInt64.lt_iff_toNat_lt.mp hi
    simp only [gradientInputs, hi', ↓reduceDIte, ↓reduceIte, withInput, true_and]
  · simp only [gradientInputs, hr, r.isLt, ↓reduceDIte, one_ne_zero, ↓reduceIte]

/-- At every point `θ₀` and cotangent `cot`, the real reading through `ExecFloat.Binary.toReal` of
the gradient kernel of the binary32 program `parabola` computes the adjoint of the derivative of
`parabolaMap` at `cot`, which is `2 θ₀ a * cot a` at entry `a`. -/
theorem parabola_gradient (m : UInt64) (θ₀ cot : EuclideanSpace ℝ (Fin m.toNat)) :
    DifferentiableAt ℝ parabolaMap θ₀ ∧ ∀ a : Fin m.toNat,
      evaluate Arithmetic.ofOps ((parabola.expression.gradient 1 0 m).map ExecFloat.Binary.toReal)
          (gradientInputs θ₀ cot) a.val.toUInt64 = .ok ((fderiv ℝ parabolaMap θ₀).adjoint cot a) ∧
        (fderiv ℝ parabolaMap θ₀).adjoint cot a = 2 * θ₀ a * cot a :=
  evaluate_map_gradient_parabola ExecFloat.Binary.toReal_zero ExecFloat.Binary.toReal_one
    parabola_expression m θ₀ cot

/-- At every point `θ₀` and cotangent `cot`, the real reading through binary32 values of the
gradient kernel of the `Float32` program `parabolaFloat32` computes the adjoint of the derivative
of `parabolaMap` at `cot`, which is `2 θ₀ a * cot a` at entry `a`. -/
theorem parabolaFloat32_gradient (m : UInt64) (θ₀ cot : EuclideanSpace ℝ (Fin m.toNat)) :
    DifferentiableAt ℝ parabolaMap θ₀ ∧ ∀ a : Fin m.toNat,
      evaluate Arithmetic.ofOps ((parabolaFloat32.expression.gradient 1 0 m).map fun x =>
          ExecFloat.Binary.toReal (ExecFloat.Binary.ofFloat32 x : Binary32))
          (gradientInputs θ₀ cot) a.val.toUInt64 = .ok ((fderiv ℝ parabolaMap θ₀).adjoint cot a) ∧
        (fderiv ℝ parabolaMap θ₀).adjoint cot a = 2 * θ₀ a * cot a :=
  evaluate_map_gradient_parabola toReal_ofFloat32_zero toReal_ofFloat32_one
    parabolaFloat32_expression m θ₀ cot

/-- Let `f` read the constants of `α` as reals and keep zero, and let `e` read input zero and
compute `x * x + 1`. At an entry `i` below `m`, the real reading of the tangent kernel of `e` with
primal inputs `x` and tangents `dx` computes `2 * x 0 i * dx 0 i`. -/
theorem evaluate_map_tangent_parabola {α : Type} [OfNat α 0] [OfNat α 1] {f : α → ℝ}
    (hf0 : f 0 = 0) {e : Expr α [.index] .scalar}
    (he : e = .letIn (.load 0 (.var .zero))
      (.binary .add (.binary .mul (.var .zero) (.var .zero)) (.scalar 1)))
    {m i : UInt64} (hi : i < m) (x dx : Nat → UInt64 → ℝ) :
    evaluate Arithmetic.ofOps ((e.tangent 1).map f) ((oneInput m).dual 1 x dx) i =
      .ok (2 * x 0 i * dx 0 i) := by
  have hk (o : Nat) (n : UInt64) (y : ℝ) (h : oneInput m o n = .ok y) : o < 1 := by
    unfold oneInput at h
    split_ifs at h with hc
    rw [hc.1]
    exact Nat.zero_lt_one
  have hread (s : ℝ) : (oneInput m).withValues (fun j n => x j n + s * dx j n) 0 i =
      .ok (x 0 i + s * dx 0 i) := by
    simp only [Reader.withValues, oneInput, hi, and_self, ↓reduceIte, Except.map]
  have hg : ∀ᶠ s in 𝓝 (0 : ℝ), evaluate Arithmetic.ofOps (e.map f)
      ((oneInput m).withValues fun j n => x j n + s * dx j n) i =
        .ok ((x 0 i + s * dx 0 i) * (x 0 i + s * dx 0 i) + f 1) :=
    Eventually.of_forall fun s => by
      rw [he]
      simp only [evaluate, Expr.map_letIn, Expr.map_load, Expr.map_var, Expr.map_binary,
        Expr.map_scalar, Expr.eval, Env.get_output, ← Expr.except_pure, pure_bind, hread s,
        Env.get_push_zero, Arithmetic.ofOps_binary_mul, Arithmetic.ofOps_binary_add]
  have hreg : (e.map f).Regular ((oneInput m).withValues x) (Env.output i) := by
    rw [he]
    simp [Expr.Regular]
  have hline : HasDerivAt (fun s : ℝ => x 0 i + s * dx 0 i) (dx 0 i) 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).mul_const (dx 0 i)).const_add (x 0 i)
  obtain ⟨d, hd, hderiv⟩ := Expr.hasDerivAt_evaluate_map_tangent hf0 e hk x dx hg hreg
  rw [hd, hderiv.unique ((hline.mul hline).add_const (f 1))]
  congr 1
  ring

/-- At an entry `i` below `m`, the real reading through `ExecFloat.Binary.toReal` of the tangent
kernel of the binary32 program `parabola` with primal inputs `x` and tangents `dx` computes
`2 * x 0 i * dx 0 i`. -/
theorem parabola_tangent {m i : UInt64} (hi : i < m) (x dx : Nat → UInt64 → ℝ) :
    evaluate Arithmetic.ofOps ((parabola.expression.tangent 1).map ExecFloat.Binary.toReal)
      ((oneInput m).dual 1 x dx) i = .ok (2 * x 0 i * dx 0 i) :=
  evaluate_map_tangent_parabola ExecFloat.Binary.toReal_zero parabola_expression hi x dx

/-- At an entry `i` below `m`, the real reading through binary32 values of the tangent kernel of
the `Float32` program `parabolaFloat32` with primal inputs `x` and tangents `dx` computes
`2 * x 0 i * dx 0 i`. -/
theorem parabolaFloat32_tangent {m i : UInt64} (hi : i < m) (x dx : Nat → UInt64 → ℝ) :
    evaluate Arithmetic.ofOps ((parabolaFloat32.expression.tangent 1).map fun v =>
        ExecFloat.Binary.toReal (ExecFloat.Binary.ofFloat32 v : Binary32))
      ((oneInput m).dual 1 x dx) i = .ok (2 * x 0 i * dx 0 i) :=
  evaluate_map_tangent_parabola toReal_ofFloat32_zero parabolaFloat32_expression hi x dx

/-- `a * b / (a * a + 1)` at each entry, with `a` read from input zero and `b` from input one,
written with `Program.of` over Lean's native `Float32` as in the kernel README. -/
def quotient : Kernel.Program Float32 :=
  Program.of (fun (read : Kernel.Reader Float32) (i : UInt64) => (do
    let a ← read 0 i
    let b ← read 1 i
    pure (a * b / (a * a + 1)) : Except Kernel.Error Float32))

/-- The frontend turns `quotient` into reads of inputs zero and one at the output index, bound as
locals, followed by `a * b / (a * a + 1)`. -/
theorem quotient_expression : quotient.expression =
    .letIn (.load 0 (.var .zero)) (.letIn (.load 1 (.var (.succ .zero)))
      (.binary .div (.binary .mul (.var (.succ .zero)) (.var .zero))
        (.binary .add (.binary .mul (.var (.succ .zero)) (.var (.succ .zero))) (.scalar 1)))) :=
  rfl

/-- The reads of a program with two inputs of `n` entries each. -/
def twoInputs (n : UInt64) : Reader ℝ :=
  fun o i => if o < 2 ∧ i < n then .ok 0 else .error (.input o)

/-- The reader of the gradient kernels of `quotient`: block `o` of the point `θ` as input `o` for
`o < 2`, and the cotangent `cot` as input two. -/
noncomputable def quotientInputs {n : UInt64} (θ : EuclideanSpace ℝ (Σ _ : Fin 2, Fin n.toNat))
    (cot : EuclideanSpace ℝ (Fin n.toNat)) : Reader ℝ :=
  fun o i =>
    if h : i.toNat < n.toNat then
      if ho : o < 2 then .ok (θ ⟨⟨o, ho⟩, ⟨i.toNat, h⟩⟩)
      else if o = 2 then .ok (cot ⟨i.toNat, h⟩) else .error (.input o)
    else .error (.bounds o i n.toNat)

/-- The map `(a, b) ↦ a * b / (a * a + 1)`, entry by entry, with `a` block zero and `b` block one
of the point. -/
noncomputable def quotientMap {n : UInt64} (θ : EuclideanSpace ℝ (Σ _ : Fin 2, Fin n.toNat)) :
    EuclideanSpace ℝ (Fin n.toNat) :=
  WithLp.toLp 2 fun r => θ ⟨0, r⟩ * θ ⟨1, r⟩ / (θ ⟨0, r⟩ * θ ⟨0, r⟩ + 1)

/-- With `a` and `b` the blocks of `θ₀`, the adjoint of the derivative of `quotientMap` at `θ₀`
sends `cot` to `b (1 - a²) / (a² + 1)² * cot` in block zero and to `a / (a² + 1) * cot` in block
one, entry by entry. -/
theorem adjoint_fderiv_quotientMap_apply {n : UInt64}
    (θ₀ : EuclideanSpace ℝ (Σ _ : Fin 2, Fin n.toNat)) (cot : EuclideanSpace ℝ (Fin n.toNat))
    (a : Fin n.toNat) :
    (fderiv ℝ quotientMap θ₀).adjoint cot ⟨0, a⟩ =
        θ₀ ⟨1, a⟩ * (1 - θ₀ ⟨0, a⟩ ^ 2) / (θ₀ ⟨0, a⟩ ^ 2 + 1) ^ 2 * cot a ∧
      (fderiv ℝ quotientMap θ₀).adjoint cot ⟨1, a⟩ = θ₀ ⟨0, a⟩ / (θ₀ ⟨0, a⟩ ^ 2 + 1) * cot a := by
  have hden (r : Fin n.toNat) : θ₀ ⟨0, r⟩ * θ₀ ⟨0, r⟩ + 1 ≠ 0 :=
    (add_pos_of_nonneg_of_pos (mul_self_nonneg _) one_pos).ne'
  have hp (s : Σ _ : Fin 2, Fin n.toNat) := PiLp.hasFDerivAt_apply (𝕜 := ℝ) 2 θ₀ s
  have hpi := hasFDerivAt_pi.mpr fun r : Fin n.toNat =>
    ((hp ⟨0, r⟩).mul (hp ⟨1, r⟩)).mul ((hasDerivAt_inv (hden r)).comp_hasFDerivAt θ₀
      (((hp ⟨0, r⟩).mul (hp ⟨0, r⟩)).add_const (1 : ℝ)))
  have hFD : HasFDerivAt (quotientMap (n := n)) _ θ₀ :=
    ((PiLp.hasFDerivAt_toLp (𝕜 := ℝ) 2 fun r =>
      θ₀ ⟨0, r⟩ * θ₀ ⟨1, r⟩ * (θ₀ ⟨0, r⟩ * θ₀ ⟨0, r⟩ + 1)⁻¹).comp θ₀ hpi).congr_of_eventuallyEq
      (Eventually.of_forall fun θ => by
        simp only [quotientMap, Function.comp_apply, Pi.mul_apply, div_eq_mul_inv])
  have hentry (j : Fin 2) (v : EuclideanSpace ℝ (Σ _ : Fin 2, Fin n.toNat)) :
      v ⟨j, a⟩ = inner ℝ (EuclideanSpace.single ⟨j, a⟩ 1) v := by
    rw [EuclideanSpace.inner_single_left, map_one, one_mul]
  have hj (j : Fin 2) : (fderiv ℝ quotientMap θ₀).adjoint cot ⟨j, a⟩ =
      (fderiv ℝ quotientMap θ₀ (EuclideanSpace.single ⟨j, a⟩ 1)) a * cot a := by
    rw [hentry j, ContinuousLinearMap.adjoint_inner_right, PiLp.inner_apply,
      Finset.sum_eq_single a]
    · simp only [RCLike.inner_apply, conj_trivial, mul_comm]
    · intro b _ hb
      rw [hFD.fderiv]
      simp only [Fin.isValue, Pi.mul_apply, smul_add, neg_smul, smul_neg, Function.comp_apply,
        ContinuousLinearMap.comp_apply, ContinuousLinearMap.coe_pi', add_apply, neg_apply,
        smul_apply, PiLp.proj_apply, PiLp.single_apply, Sigma.mk.injEq, heq_eq_eq, smul_eq_mul,
        mul_ite, mul_one, mul_zero, ContinuousLinearEquiv.coe_coe,
        PiLp.continuousLinearEquiv_symm_apply, hb, and_false, ↓reduceIte, neg_zero, add_zero,
        RCLike.inner_apply, conj_trivial]
    · intro h
      exact absurd (Finset.mem_univ a) h
  rw [hj 0, hj 1, hFD.fderiv]
  simp only [Fin.isValue, Pi.mul_apply, smul_add, neg_smul, smul_neg, Function.comp_apply,
    ContinuousLinearMap.comp_apply, ContinuousLinearMap.coe_pi', add_apply, neg_apply, smul_apply,
    PiLp.proj_apply, PiLp.single_apply, Sigma.mk.injEq, heq_eq_eq, true_and, smul_eq_mul, mul_ite,
    mul_one, mul_zero, one_ne_zero, false_and, zero_add, ContinuousLinearEquiv.coe_coe,
    PiLp.continuousLinearEquiv_symm_apply, ↓reduceIte, zero_ne_one, neg_zero, add_zero]
  constructor
  · field_simp
    ring
  · field_simp

/-- At every point `θ₀` and cotangent `cot`, `quotientMap` is differentiable at `θ₀`, and the real
reading through binary32 values of the gradient kernel of input `j` of the `Float32` program
`quotient` computes block `j` of the adjoint of its derivative at `cot`. -/
theorem quotient_gradient (n : UInt64) (θ₀ : EuclideanSpace ℝ (Σ _ : Fin 2, Fin n.toNat))
    (cot : EuclideanSpace ℝ (Fin n.toNat)) :
    DifferentiableAt ℝ quotientMap θ₀ ∧ ∀ (j : Fin 2) (a : Fin n.toNat),
      evaluate Arithmetic.ofOps ((quotient.expression.gradient 2 j n).map fun x =>
          ExecFloat.Binary.toReal (ExecFloat.Binary.ofFloat32 x : Binary32))
        (quotientInputs θ₀ cot) a.val.toUInt64 =
          .ok ((fderiv ℝ quotientMap θ₀).adjoint cot ⟨j, a⟩) := by
  have hr (r : Fin n.toNat) : (r.val.toUInt64).toNat = r.val := by
    rw [Nat.toUInt64_eq]
    exact UInt64.toNat_ofNat_of_lt' (lt_trans r.isLt n.toNat_lt_size)
  have hlt (r : Fin n.toNat) : r.val.toUInt64 < n := by
    rw [UInt64.lt_iff_toNat_lt, hr]
    exact r.isLt
  have hread (θ : EuclideanSpace ℝ (Σ _ : Fin 2, Fin n.toNat)) (o : Nat) (ho : o < 2)
      (r : Fin n.toNat) : (twoInputs n).withValues (withInputs (fun _ _ => 0) θ) o
        r.val.toUInt64 = .ok (θ ⟨⟨o, ho⟩, r⟩) := by
    simp only [Reader.withValues, twoInputs, ho, hlt r, and_self, ↓reduceIte, Except.map,
      withInputs, hr, r.isLt, ↓reduceDIte]
  refine Expr.map_gradient_eq_adjoint_withInputs toReal_ofFloat32_zero toReal_ofFloat32_one
    quotient.expression (k := 2) (m := fun _ => n) (read := twoInputs n) (fun o i y h => ?_)
    (fun _ _ => 0) (θ₀ := θ₀) (F := quotientMap) (Eventually.of_forall fun θ r => ?_)
    (fun r => ?_) (R := quotientInputs θ₀ cot) (fun o i y h => ?_) (cot := cot) (fun r => ?_)
  · unfold twoInputs at h
    split_ifs at h with hc
    exact hc.1
  · rw [quotient_expression]
    simp only [evaluate, Expr.map_letIn, Expr.map_load, Expr.map_var, Expr.map_binary,
      Expr.map_scalar, toReal_ofFloat32_one, Expr.eval, Env.get_output, ← Expr.except_pure,
      pure_bind, hread θ 0 two_pos r, hread θ 1 one_lt_two r, Env.get_push_zero,
      Env.get_push_succ, Arithmetic.ofOps_binary_mul, Arithmetic.ofOps_binary_add,
      Arithmetic.ofOps_binary_div, quotientMap, PiLp.toLp_apply, Fin.zero_eta, Fin.mk_one]
  · rw [quotient_expression]
    have hden : θ₀ ⟨0, r⟩ * θ₀ ⟨0, r⟩ + 1 ≠ 0 :=
      (add_pos_of_nonneg_of_pos (mul_self_nonneg _) one_pos).ne'
    simp only [Expr.map_letIn, Expr.map_load, Expr.map_var, Expr.map_binary, Expr.map_scalar,
      toReal_ofFloat32_one, Expr.Regular, Expr.eval, Env.get_output, ← Expr.except_pure,
      pure_bind, hread θ₀ 0 two_pos r, Fin.zero_eta, Fin.isValue, Env.get_push_succ,
      hread θ₀ 1 one_lt_two r, Fin.mk_one, Env.get_push_zero, reduceCtorEq, IsEmpty.forall_iff,
      and_self, implies_true, Arithmetic.ofOps_binary_mul, Arithmetic.ofOps_binary_add,
      forall_const, true_and]
    rintro _ ⟨⟩ _ ⟨⟩ _ - _ ⟨⟩
    exact hden
  · obtain ⟨z, hz, rfl⟩ := Reader.exists_ok_of_withValues h
    unfold twoInputs at hz
    split_ifs at hz with hc
    obtain ⟨ho, hi⟩ := hc
    have hi' : i.toNat < n.toNat := UInt64.lt_iff_toNat_lt.mp hi
    simp only [quotientInputs, hi', ho, ↓reduceDIte, withInputs]
  · simp only [quotientInputs, hr, r.isLt, ↓reduceDIte, lt_self_iff_false, ↓reduceIte]

/-- At every point `θ₀` and cotangent `cot`, with `a` and `b` the blocks of `θ₀`, the real reading
through binary32 values of the gradient kernel of `quotient` computes `b (1 - a²) / (a² + 1)² * cot`
in input zero and `a / (a² + 1) * cot` in input one, entry by entry. -/
theorem quotient_gradient_apply (n : UInt64) (θ₀ : EuclideanSpace ℝ (Σ _ : Fin 2, Fin n.toNat))
    (cot : EuclideanSpace ℝ (Fin n.toNat)) (a : Fin n.toNat) :
    evaluate Arithmetic.ofOps ((quotient.expression.gradient 2 0 n).map fun x =>
          ExecFloat.Binary.toReal (ExecFloat.Binary.ofFloat32 x : Binary32))
        (quotientInputs θ₀ cot) a.val.toUInt64 =
          .ok (θ₀ ⟨1, a⟩ * (1 - θ₀ ⟨0, a⟩ ^ 2) / (θ₀ ⟨0, a⟩ ^ 2 + 1) ^ 2 * cot a) ∧
      evaluate Arithmetic.ofOps ((quotient.expression.gradient 2 1 n).map fun x =>
          ExecFloat.Binary.toReal (ExecFloat.Binary.ofFloat32 x : Binary32))
        (quotientInputs θ₀ cot) a.val.toUInt64 =
          .ok (θ₀ ⟨0, a⟩ / (θ₀ ⟨0, a⟩ ^ 2 + 1) * cot a) := by
  obtain ⟨h0, h1⟩ := adjoint_fderiv_quotientMap_apply θ₀ cot a
  rw [← h0, ← h1]
  exact ⟨(quotient_gradient n θ₀ cot).2 0 a, (quotient_gradient n θ₀ cot).2 1 a⟩

end TorchLean.Test.KernelMap
