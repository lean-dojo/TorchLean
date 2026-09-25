/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Tactic.Measurability.Init
public import NN.MLTheory.CROWN.Runtime.Ops
public import NN.Spec.Core.Context.Real

/-!
# Nonlinear IBP Soundness Lemmas

Monotonicity and Lipschitz facts for the nonlinear graph operations handled by the IBP certificate
soundness theorem.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN

namespace CertSoundness

noncomputable section

/-- Lift a scalar enclosure through a shape-preserving elementwise map. -/
private theorem mapBounds_sound_real {s : Shape}
    (f : ℝ → ℝ) (lower upper : ℝ → ℝ → ℝ)
    (hscalar : ∀ l u v, l ≤ v ∧ v ≤ u → lower l u ≤ f v ∧ f v ≤ upper l u)
    (xB : Box ℝ s) (x : Tensor ℝ s) (hx : Box.contains xB x) :
    Box.contains
      ⟨Tensor.map2Spec lower xB.lo xB.hi, Tensor.map2Spec upper xB.lo xB.hi⟩
      (Tensor.mapSpec f x) := by
  induction s with
  | scalar =>
      simpa only [Box.contains, Tensor.toScalar_map2Spec, Tensor.toScalar_mapSpec]
        using hscalar xB.lo.item xB.hi.item x.item hx
  | dim n inner ih =>
      intro i
      change Box.contains
        ⟨(Tensor.map2Spec lower xB.lo xB.hi).unstack i,
          (Tensor.map2Spec upper xB.lo xB.hi).unstack i⟩
        ((Tensor.mapSpec f x).unstack i)
      rw [show (Tensor.map2Spec lower xB.lo xB.hi).unstack i =
          Tensor.map2Spec lower (xB.lo.unstack i) (xB.hi.unstack i) from
            (TorchLean.Tensor.Internal.Rep.zipWith_unstack lower xB.lo xB.hi i).symm,
        show (Tensor.map2Spec upper xB.lo xB.hi).unstack i =
          Tensor.map2Spec upper (xB.lo.unstack i) (xB.hi.unstack i) from
            (TorchLean.Tensor.Internal.Rep.zipWith_unstack upper xB.lo xB.hi i).symm,
        show (Tensor.mapSpec f x).unstack i = Tensor.mapSpec f (x.unstack i) from
          (TorchLean.Tensor.Internal.Rep.map_unstack f x i).symm]
      exact ih ⟨xB.lo.unstack i, xB.hi.unstack i⟩ (x.unstack i) (hx i)

/-!
### Soundness of `Runtime.Ops.IBP.mapMinmax` for monotone scalar functions

`Runtime.Ops.IBP.sigmoid` and `Runtime.Ops.IBP.tanh` are defined using `mapMinmax`.
If the activation is monotone, then the min/max of the endpoints is a correct enclosure.
-/

theorem map_minmax_sound_real {s : Shape} (f : ℝ → ℝ) (hf : Monotone f)
    (xB : Box ℝ s) (x : Tensor ℝ s)
    (hx : Box.contains (α := ℝ) xB x) :
    Box.contains (α := ℝ) (Runtime.Ops.IBP.mapMinmax f xB)
      (Tensor.mapSpec f x) := by
  refine mapBounds_sound_real f _ _ ?_ xB x hx
  intro l u v hv
  have hflfu : f l ≤ f u := hf (le_trans hv.1 hv.2)
  have hnot : ¬f l > f u := not_lt_of_ge hflfu
  simpa only [ite_eq_right hnot] using And.intro (hf hv.1) (hf hv.2)

/-!
### Soundness of the 1-Lipschitz `sin`/`cos` enclosures

`Runtime.Ops.IBP.sin` / `Runtime.Ops.IBP.cos` use a midpoint enclosure with radius `r=(u-l)/2`,
clamped to `[-1,1]`. This avoids periodic case splits while remaining sound.
-/

theorem sin_lipschitz_real (x y : ℝ) : |Real.sin x - Real.sin y| ≤ |x - y| := by
  have h := Real.sin_sub_sin x y
  calc
    |Real.sin x - Real.sin y|
        = |2 * Real.sin ((x - y) / 2) * Real.cos ((x + y) / 2)| := by
            simp [h, mul_left_comm, mul_comm]
    _ = 2 * |Real.sin ((x - y) / 2)| * |Real.cos ((x + y) / 2)| := by
          simp [abs_mul, mul_left_comm, mul_comm]
    _ ≤ 2 * |(x - y) / 2| * 1 := by
          have hsin : |Real.sin ((x - y) / 2)| ≤ |(x - y) / 2| := by
            simpa using (Real.abs_sin_le_abs (x := (x - y) / 2))
          have hcos : |Real.cos ((x + y) / 2)| ≤ 1 := by
            simpa using Real.abs_cos_le_one ((x + y) / 2)
          -- Multiply the two bounds, keeping track of nonnegativity.
          have h2 : (2 : ℝ) * |Real.sin ((x - y) / 2)| ≤ 2 * |(x - y) / 2| :=
            mul_le_mul_of_nonneg_left hsin (by norm_num)
          have hstep1 :
              (2 * |Real.sin ((x - y) / 2)|) * |Real.cos ((x + y) / 2)|
                ≤ (2 * |(x - y) / 2|) * |Real.cos ((x + y) / 2)| :=
            mul_le_mul_of_nonneg_right h2 (abs_nonneg _)
          have hstep2 :
              (2 * |(x - y) / 2|) * |Real.cos ((x + y) / 2)|
                ≤ (2 * |(x - y) / 2|) * 1 :=
            mul_le_mul_of_nonneg_left hcos (mul_nonneg (by norm_num) (abs_nonneg _))
          -- Reassociate back into `2 * |sin| * |cos|`.
          simpa [mul_assoc, mul_left_comm, mul_comm] using le_trans hstep1 hstep2
    _ = |x - y| := by
          -- `2 * |(x-y)/2| = |x-y|`.
          have htwo : (2 : ℝ) ≠ 0 := by norm_num
          calc
            2 * |(x - y) / 2| * 1 = 2 * (|x - y| / 2) := by
              simp [div_eq_mul_inv, mul_left_comm]
            _ = |x - y| := by nlinarith

/-- Cosine is 1-Lipschitz, proved from the sum-to-product identity.

The interval rules for `sin` and `cos` fall back on this whenever the input interval is too wide for
a monotone branch: a Lipschitz constant of one turns the input width directly into an output width.
-/
theorem cos_lipschitz_real (x y : ℝ) : |Real.cos x - Real.cos y| ≤ |x - y| := by
  have h := Real.cos_sub_cos x y
  calc
    |Real.cos x - Real.cos y|
        = |(-2) * Real.sin ((x + y) / 2) * Real.sin ((x - y) / 2)| := by
            simp [h, mul_assoc]
    _ = 2 * |Real.sin ((x + y) / 2)| * |Real.sin ((x - y) / 2)| := by
          simp [abs_mul, mul_assoc]
    _ ≤ 2 * 1 * |(x - y) / 2| := by
          have hsin1 : |Real.sin ((x + y) / 2)| ≤ 1 := by
            simpa using Real.abs_sin_le_one ((x + y) / 2)
          have hsin2 : |Real.sin ((x - y) / 2)| ≤ |(x - y) / 2| := by
            simpa using (Real.abs_sin_le_abs (x := (x - y) / 2))
          have h2 : (2 : ℝ) * |Real.sin ((x + y) / 2)| ≤ 2 * 1 :=
            mul_le_mul_of_nonneg_left hsin1 (by norm_num)
          have hstep1 :
              (2 * |Real.sin ((x + y) / 2)|) * |Real.sin ((x - y) / 2)|
                ≤ (2 * 1) * |Real.sin ((x - y) / 2)| :=
            mul_le_mul_of_nonneg_right h2 (abs_nonneg _)
          have hstep2 :
              (2 * 1) * |Real.sin ((x - y) / 2)| ≤ (2 * 1) * |(x - y) / 2| :=
            mul_le_mul_of_nonneg_left hsin2 (by norm_num)
          simpa [mul_assoc, mul_left_comm, mul_comm] using le_trans hstep1 hstep2
    _ = |x - y| := by
          have htwo : (2 : ℝ) ≠ 0 := by norm_num
          calc
            2 * 1 * |(x - y) / 2| = 2 * (|x - y| / 2) := by
              simp [div_eq_mul_inv, mul_left_comm, mul_comm]
            _ = |x - y| := by nlinarith

/-- Interval bound propagation through `sin` is sound over `ℝ`.

Each component uses the midpoint Lipschitz enclosure, intersected with `[-1, 1]`.
-/
theorem ibp_sin_sound_real {s : Shape} (xB : Box ℝ s) (x : Tensor ℝ s)
    (hx : Box.contains (α := ℝ) xB x) :
    Box.contains (α := ℝ) (Runtime.Ops.IBP.sin xB)
      (Tensor.mapSpec Real.sin x) := by
  refine mapBounds_sound_real Real.sin _ _ ?_ xB x hx
  intro l u v hv
  let m : ℝ := (l + u) / 2
  let r : ℝ := (u - l) / 2
  have hxm : |v - m| ≤ r := by
    apply abs_le.2
    constructor <;> dsimp [m, r] <;> nlinarith [hv.1, hv.2]
  have hLip : |Real.sin v - Real.sin m| ≤ r :=
    le_trans (sin_lipschitz_real v m) hxm
  have hdiff : -r ≤ Real.sin v - Real.sin m ∧ Real.sin v - Real.sin m ≤ r :=
    abs_le.1 hLip
  have hmidLo : Real.sin m - r ≤ Real.sin v := by linarith [hdiff.1]
  have hmidHi : Real.sin v ≤ Real.sin m + r := by linarith [hdiff.2]
  have hsinRange : (-1 : ℝ) ≤ Real.sin v ∧ Real.sin v ≤ (1 : ℝ) := by
    exact abs_le.1 (by simpa using Real.abs_sin_le_one v)
  have hBounds :
      max (-1 : ℝ) (Real.sin m - r) ≤ Real.sin v ∧
        Real.sin v ≤ min (1 : ℝ) (Real.sin m + r) :=
    ⟨max_le_iff.2 ⟨hsinRange.1, hmidLo⟩,
      le_min_iff.2 ⟨hsinRange.2, hmidHi⟩⟩
  simpa only [MathFunctions.sin, m, r] using hBounds

/-- Interval bound propagation through `cos` is sound over `ℝ`, by the same argument. -/
theorem ibp_cos_sound_real {s : Shape} (xB : Box ℝ s) (x : Tensor ℝ s)
    (hx : Box.contains (α := ℝ) xB x) :
    Box.contains (α := ℝ) (Runtime.Ops.IBP.cos xB)
      (Tensor.mapSpec Real.cos x) := by
  refine mapBounds_sound_real Real.cos _ _ ?_ xB x hx
  intro l u v hv
  let m : ℝ := (l + u) / 2
  let r : ℝ := (u - l) / 2
  have hxm : |v - m| ≤ r := by
    apply abs_le.2
    constructor <;> dsimp [m, r] <;> nlinarith [hv.1, hv.2]
  have hLip : |Real.cos v - Real.cos m| ≤ r :=
    le_trans (cos_lipschitz_real v m) hxm
  have hdiff : -r ≤ Real.cos v - Real.cos m ∧ Real.cos v - Real.cos m ≤ r :=
    abs_le.1 hLip
  have hmidLo : Real.cos m - r ≤ Real.cos v := by linarith [hdiff.1]
  have hmidHi : Real.cos v ≤ Real.cos m + r := by linarith [hdiff.2]
  have hcosRange : (-1 : ℝ) ≤ Real.cos v ∧ Real.cos v ≤ (1 : ℝ) := by
    exact abs_le.1 (by simpa using Real.abs_cos_le_one v)
  have hBounds :
      max (-1 : ℝ) (Real.cos m - r) ≤ Real.cos v ∧
        Real.cos v ≤ min (1 : ℝ) (Real.cos m + r) :=
    ⟨max_le_iff.2 ⟨hcosRange.1, hmidLo⟩,
      le_min_iff.2 ⟨hcosRange.2, hmidHi⟩⟩
  simpa only [MathFunctions.cos, m, r] using hBounds

end

end CertSoundness

end NN.MLTheory.CROWN.Graph
