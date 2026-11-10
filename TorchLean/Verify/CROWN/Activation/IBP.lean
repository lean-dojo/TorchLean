/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
public import TorchLean.Verify.CROWN.Activation
public import TorchLean.Tensor.Context.Real

/-!
# Nonlinear IBP Soundness Lemmas

Monotonicity and Lipschitz facts for the nonlinear graph operations handled by the IBP certificate
soundness theorem. `mapMinmax_sound_real` covers the endpoint min/max rule for monotone
activations, and `sinMidpoint_sound_real` and `cosMidpoint_sound_real` cover the clamped midpoint
rules for `sin` and `cos`.
-/

@[expose] public section

namespace TorchLean.CROWN

open TorchLean.Tensor

noncomputable section

/-- A scalar enclosure of `f` lifts entrywise: applying `lower` and `upper` to the endpoints of `xB`
gives a box that contains `Tensor.map f x`. -/
private theorem Graph.mapBounds_sound_real {s : Shape}
    (f : ℝ → ℝ) (lower upper : ℝ → ℝ → ℝ)
    (hscalar : ∀ l u v, l ≤ v ∧ v ≤ u → lower l u ≤ f v ∧ f v ≤ upper l u)
    (xB : Box ℝ s) (x : Tensor ℝ s) (hx : Box.Contains xB x) :
    Box.Contains
      ⟨Tensor.zipWith lower xB.lo xB.hi, Tensor.zipWith upper xB.lo xB.hi⟩
      (Tensor.map f x) := by
  induction s with
  | scalar =>
      simpa only [Box.Contains, Tensor.toScalar_zipWith, Tensor.item_map]
        using hscalar xB.lo.item xB.hi.item x.item hx
  | dim n inner ih =>
      intro i
      change Box.Contains
        ⟨(Tensor.zipWith lower xB.lo xB.hi).unstack i,
          (Tensor.zipWith upper xB.lo xB.hi).unstack i⟩
        ((Tensor.map f x).unstack i)
      rw [show (Tensor.zipWith lower xB.lo xB.hi).unstack i =
          Tensor.zipWith lower (xB.lo.unstack i) (xB.hi.unstack i) from
            (Tensor.Rep.zipWith_unstack lower xB.lo xB.hi i).symm,
        show (Tensor.zipWith upper xB.lo xB.hi).unstack i =
          Tensor.zipWith upper (xB.lo.unstack i) (xB.hi.unstack i) from
            (Tensor.Rep.zipWith_unstack upper xB.lo xB.hi i).symm,
        show (Tensor.map f x).unstack i = Tensor.map f (x.unstack i) from
          Tensor.unstack_map f x i]
      exact ih ⟨xB.lo.unstack i, xB.hi.unstack i⟩ (x.unstack i) (hx i)

/-!
### Soundness of `IBP.mapMinmax` for monotone scalar functions

`IBP.sigmoidMinmax` and `IBP.tanhMinmax` apply `mapMinmax` to the sigmoid
and tanh specifications. For a monotone activation, the smaller and larger endpoint values enclose
its value on the whole interval.
-/

/-- For a monotone `f`, the box `IBP.mapMinmax f xB` contains `Tensor.map f x` for every `x`
in `xB`. -/
theorem IBP.mapMinmax_sound_real {s : Shape} (f : ℝ → ℝ) (hf : Monotone f)
    (xB : Box ℝ s) (x : Tensor ℝ s)
    (hx : Box.Contains (α := ℝ) xB x) :
    Box.Contains (α := ℝ) (IBP.mapMinmax f xB)
      (Tensor.map f x) := by
  refine Graph.mapBounds_sound_real f _ _ ?_ xB x hx
  intro l u v hv
  have hflfu : f l ≤ f u := hf (le_trans hv.1 hv.2)
  have hnot : ¬f l > f u := not_lt_of_ge hflfu
  simpa only [ite_eq_right hnot] using And.intro (hf hv.1) (hf hv.2)

/-!
### Soundness of the 1-Lipschitz `sin`/`cos` enclosures

`IBP.sinMidpoint` and `IBP.cosMidpoint` enclose the function by its midpoint
value plus or minus the radius `r=(u-l)/2`, clamped to `[-1,1]`. The rule needs no periodic case
split, because the Lipschitz constant one (`Real.abs_sin_sub_sin_le`, `Real.abs_cos_sub_cos_le`)
turns the input half-width into an output half-width.
-/

/-- A unit-range, 1-Lipschitz function is enclosed by its clipped midpoint-radius bounds. -/
private theorem Graph.unit_lipschitz_bounds (f : ℝ → ℝ)
    (hlip : ∀ x y, |f x - f y| ≤ |x - y|)
    (hrange : ∀ x, -1 ≤ f x ∧ f x ≤ 1)
    (l u v : ℝ) (hv : l ≤ v ∧ v ≤ u) :
    max (-1) (f ((l + u) / 2) - (u - l) / 2) ≤ f v ∧
      f v ≤ min 1 (f ((l + u) / 2) + (u - l) / 2) := by
  have hmid : |v - (l + u) / 2| ≤ (u - l) / 2 := by
    apply abs_le.2
    constructor <;> linarith [hv.1, hv.2]
  have hdiff := abs_le.1 ((hlip v ((l + u) / 2)).trans hmid)
  refine ⟨max_le (hrange v).1 ?_, le_min (hrange v).2 ?_⟩ <;> linarith

/-- Over `ℝ`, the box `sinMidpoint xB` contains `sin` of every point of `xB`.

Each entry is the midpoint Lipschitz enclosure intersected with `[-1, 1]`.
-/
theorem Graph.sinMidpoint_sound_real {s : Shape} (xB : Box ℝ s) (x : Tensor ℝ s)
    (hx : Box.Contains (α := ℝ) xB x) :
    Box.Contains (α := ℝ) (IBP.sinMidpoint xB)
      (Tensor.map Real.sin x) := by
  refine mapBounds_sound_real Real.sin _ _ ?_ xB x hx
  intro l u v hv
  exact unit_lipschitz_bounds Real.sin Real.abs_sin_sub_sin_le
    (fun v => ⟨Real.neg_one_le_sin v, Real.sin_le_one v⟩) l u v hv

/-- Over `ℝ`, the box `cosMidpoint xB` contains `cos` of every point of `xB`, by the argument for
`sin`. -/
theorem Graph.cosMidpoint_sound_real {s : Shape} (xB : Box ℝ s) (x : Tensor ℝ s)
    (hx : Box.Contains (α := ℝ) xB x) :
    Box.Contains (α := ℝ) (IBP.cosMidpoint xB)
      (Tensor.map Real.cos x) := by
  refine mapBounds_sound_real Real.cos _ _ ?_ xB x hx
  intro l u v hv
  exact unit_lipschitz_bounds Real.cos Real.abs_cos_sub_cos_le
    (fun v => ⟨Real.neg_one_le_cos v, Real.cos_le_one v⟩) l u v hv

end

end TorchLean.CROWN
