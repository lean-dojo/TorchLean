/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Reduce.Real
public import TorchLean.Tensor.Euclidean
public import Mathlib.Analysis.Calculus.FDeriv.Linear
public import Mathlib.Analysis.InnerProductSpace.Adjoint

/-!
# Inner products over permuted axes

Transporting two real tensors along one shape equality keeps their inner product
(`dot_eqRec`).

For every shape `s` and permutation `σ` of its axes, `Tensor.permute` by `σ` on one side of
`TorchLean.Tensor.Rep.dot` is `Tensor.permute` by `s.unpermute σ h` on the other side
(`TorchLean.Tensor.dot_permute_left`), and permuting both sides by `σ` keeps the inner product
(`TorchLean.Tensor.dot_permute_permute`).
`Tensor.permuteCLM` is `Tensor.permute` as a continuous linear map on real tensors. Its adjoint
permutes by `s.unpermute σ h` (`Tensor.permuteCLM_adjoint_apply`), and it is the derivative of
`Tensor.permute` at every point (`Tensor.hasFDerivAt_permute`, `Tensor.fderiv_permute`).
-/

@[expose] public section

namespace TorchLean.Tensor

open scoped BigOperators
open Rep (dot)

noncomputable section

/-- Transporting both tensors through the same shape equality preserves their inner product. -/
theorem dot_eqRec {s t : Shape} (h : s = t) (x y : Tensor ℝ s) :
    Rep.dot (h ▸ x) (h ▸ y) = Rep.dot x y := by
  cases h
  rfl

end

end TorchLean.Tensor

namespace TorchLean.Tensor

open scoped BigOperators
open Rep (dot)

/-- Permuting the axes of `x` by `σ` on the left of `dot` is permuting the axes of `z` by
`s.unpermute σ h` on the right. -/
theorem dot_permute_left {s out : Shape} (x : Tensor ℝ s) (σ : Equiv.Perm (Fin s.rank))
    (h : out = s.permute σ) (z : Tensor ℝ out) :
    Rep.dot (Tensor.permute x σ out h) z =
      Rep.dot x (Tensor.permute z (s.unpermute σ h) s (s.permute_unpermute σ h).symm) := by
  have key : ∀ {f : out.Coord → s.Coord} {g : s.Coord → out.Coord}, (∀ c, g (f c) = c) →
      (∀ d, f (g d) = d) → ∑ c, x (f c) * z c = ∑ d, x d * z (g d) := fun hfg hgf =>
    Fintype.sum_bijective _ (Function.bijective_iff_has_inverse.mpr ⟨_, hfg, hgf⟩) _ _
      fun c => by rw [hfg]
  simp only [Rep.dot, Tensor.permute_apply]
  subst h
  refine key (fun c => Shape.Coord.ext_index fun k hk => ?_)
    (fun d => Shape.Coord.ext_index fun k hk => ?_) <;>
  · rw [Shape.Coord.index_ofFun, dite_eq_left hk]
    simp only [Fin.val_cast, Shape.Coord.val_get, Shape.Coord.index_reindexEq,
      Shape.Coord.index_ofFun]
    simp [Shape.unpermute, Equiv.permCongr_def]

/-- Permuting the axes of both arguments of `dot` by one `σ` keeps their inner product. -/
theorem dot_permute_permute {s out : Shape} (x y : Tensor ℝ s) (σ : Equiv.Perm (Fin s.rank))
    (h : out = s.permute σ) :
    Rep.dot (Tensor.permute x σ out h) (Tensor.permute y σ out h) = Rep.dot x y := by
  rw [dot_permute_left, Tensor.permute_unpermute]

end TorchLean.Tensor

namespace TorchLean.Tensor

/-- `permute _ σ out hout`, as a continuous linear map on real tensors. -/
noncomputable def permuteCLM {s : Shape} (σ : Equiv.Perm (Fin s.rank)) (out : Shape)
    (hout : out = s.permute σ := by tensor_shape) : Tensor ℝ s →L[ℝ] Tensor ℝ out :=
  LinearMap.toContinuousLinearMap
    { toFun := fun x => permute x σ out hout
      map_add' := fun x y => Rep.ext fun c => by
        simp only [permute_apply, Tensor.add_apply]
      map_smul' := fun a x => Rep.ext fun c => by
        simp only [permute_apply, Rep.smul_apply, RingHom.id_apply] }

/-- `permuteCLM σ out hout` maps `x` to `permute x σ out hout`. -/
@[simp] theorem permuteCLM_apply {s : Shape} (σ : Equiv.Perm (Fin s.rank)) (out : Shape)
    (hout : out = s.permute σ) (x : Tensor ℝ s) :
    permuteCLM σ out hout x = permute x σ out hout :=
  rfl

/-- The adjoint of `permuteCLM σ out hout` maps `z` to
`permute z (s.unpermute σ hout) s (s.permute_unpermute σ hout).symm`. -/
theorem permuteCLM_adjoint_apply {s : Shape} (σ : Equiv.Perm (Fin s.rank)) (out : Shape)
    (hout : out = s.permute σ) (z : Tensor ℝ out) :
    (permuteCLM σ out hout).adjoint z =
      permute z (s.unpermute σ hout) s (s.permute_unpermute σ hout).symm := by
  apply ext_inner_left ℝ
  intro x
  rw [ContinuousLinearMap.adjoint_inner_right, permuteCLM_apply, ← dot_eq_inner,
    ← dot_eq_inner,
    dot_permute_left]

/-- At every real tensor `x`, `permute _ σ out hout` has derivative `permuteCLM σ out hout`. -/
theorem hasFDerivAt_permute {s : Shape} (σ : Equiv.Perm (Fin s.rank)) (out : Shape)
    (hout : out = s.permute σ) (x : Tensor ℝ s) :
    HasFDerivAt (fun y : Tensor ℝ s => permute y σ out hout) (permuteCLM σ out hout) x :=
  (permuteCLM σ out hout).hasFDerivAt

/-- At every real tensor `x`, the derivative of `permute _ σ out hout` is
`permuteCLM σ out hout`. -/
theorem fderiv_permute {s : Shape} (σ : Equiv.Perm (Fin s.rank)) (out : Shape)
    (hout : out = s.permute σ) (x : Tensor ℝ s) :
    fderiv ℝ (fun y : Tensor ℝ s => permute y σ out hout) x = permuteCLM σ out hout :=
  (hasFDerivAt_permute σ out hout x).fderiv

end TorchLean.Tensor
