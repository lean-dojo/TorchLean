/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Algebra
public import TorchLean.Tensor.Context.Rational
public import TorchLean.Verify.CROWN.BoundOps.Lawful
public import TorchLean.Verify.CROWN.FlatBox.Affine

/-!
# Exact rational bound arithmetic and its reflection to the reals

The cast from rational to real tensors (`realTensor`) commutes with the operations of exact bound
propagation. The lemmas cover sums, matrix products, the positive and negative parts of a weight
matrix, ReLU and its relaxations, affine propagation and `IBP.linear`. A bound computed over ℚ thus
casts to the bound that the same computation gives over ℝ, with no floating-point rounding or
transcendental approximation involved. The scoped instance `instBoundOpsRat` uses exact rational
operations for both directed endpoints, and the scoped instance `instLawfulBoundOpsRat` makes ℚ with
these operations a lawful bound dictionary.
-/

@[expose] public section

namespace TorchLean.RationalAlgebraic

open CROWN

/-- Rationals have no NaN, so the IEEE maximum `Context.maximum` is `max`. -/
theorem maximum_eq_max (a b : ℚ) : Context.maximum a b = max a b :=
  Context.maximum_eq_max (fun _ => by simp [Context.isNaN]) (fun _ _ _ => rfl)
    (fun _ _ => Iff.rfl) a b

/-- Rationals have no NaN, so the IEEE minimum `Context.minimum` is `min`. -/
theorem minimum_eq_min (a b : ℚ) : Context.minimum a b = min a b :=
  Context.minimum_eq_min (fun _ => by simp [Context.isNaN]) (fun _ _ _ => rfl)
    (fun _ _ => Iff.rfl) a b

/-- Rational certificate arithmetic is exact, including affine reassociation. -/
scoped instance instBoundOpsRat : BoundOps ℚ := .exact ℚ

/-- The rational arithmetic dictionary satisfies the real enclosure laws. -/
noncomputable scoped instance instLawfulBoundOpsRat : LawfulBoundOps ℚ where
  toReal := fun q => (q : ℝ)
  lt_iff a b := (Rat.cast_lt (K := ℝ)).symm
  toReal_zero := Rat.cast_zero
  toReal_one := Rat.cast_one
  toReal_max a b := Rat.cast_max a b
  toReal_min a b := Rat.cast_min a b
  toReal_maximum a b := by rw [maximum_eq_max]; exact Rat.cast_max a b
  toReal_minimum a b := by rw [minimum_eq_min]; exact Rat.cast_min a b
  toReal_eq_of_beq h := by rw [beq_iff_eq.mp h]
  addDown_le a b := by simp [BoundOps.addDown]
  le_addUp a b := by simp [BoundOps.addUp]
  subDown_le a b := by simp [BoundOps.subDown]
  le_subUp a b := by simp [BoundOps.subUp]
  mulDown_le a b := by simp [BoundOps.mulDown]
  le_mulUp a b := by simp [BoundOps.mulUp]

end TorchLean.RationalAlgebraic

namespace TorchLean.Certificate.Rational

open Tensor
open CROWN Graph
open scoped RationalAlgebraic

/-- Interpret each rational tensor entry as a real number. -/
noncomputable def realTensor {s : Shape} (t : Tensor ℚ s) : Tensor ℝ s :=
  Tensor.map (fun q : ℚ => (q : ℝ)) t

/-- `realTensor` commutes with `Tensor.unstack`. -/
@[simp] theorem realTensor_unstack {n : Nat} {s : Shape} (t : Tensor ℚ (.dim n s))
    (i : Fin n) : Tensor.unstack (realTensor t) i = realTensor (Tensor.unstack t i) := by
  simp [realTensor]

/-- Each entry of `realTensor t` is the cast of the entry of `t`. -/
@[simp] theorem realTensor_getScalar {n : Nat} (t : Tensor ℚ [n]) (i : Fin n) :
    (realTensor t).getScalar i = (t.getScalar i : ℝ) := by
  simp [realTensor]

/-- Each entry of the matrix `realTensor t` is the cast of the entry of `t`. -/
@[simp] theorem realTensor_get2 {m n : Nat} (t : Tensor ℚ [m, n]) (i : Fin m) (j : Fin n) :
    get2 (realTensor t) i j = ((get2 t i j : ℚ) : ℝ) := by
  simp [get2_eq_getScalar_unstack]

/-- `realTensor` of a constant tensor is the constant tensor of the cast. -/
@[simp] theorem realTensor_full {s : Shape} (q : ℚ) :
    realTensor (Tensor.full s q) = Tensor.full s (q : ℝ) := by
  apply Tensor.Rep.ext
  intro i
  simp [realTensor]

/-- `realTensor` commutes with addition. -/
@[simp] theorem realTensor_add {s : Shape} (a b : Tensor ℚ s) :
    realTensor (a + b) = realTensor a + realTensor b := by
  apply Tensor.Rep.ext
  intro i
  simp [realTensor, Tensor.add_def]

/-- `realTensor` commutes with the matrix-vector contraction. -/
@[simp] theorem realTensor_contract_matVec {m n : Nat} (a : Tensor ℚ [m, n]) (b : Tensor ℚ [n]) :
    realTensor (Tensor.contract a b _ _ [m] .matVec) =
        Tensor.contract (realTensor a) (realTensor b) _ _ [m] .matVec := by
  apply Tensor.ext_vector
  intro i
  simp [Tensor.getScalar_contract_matVec_eq_sum]

/-- `realTensor` commutes with `Tensor.matmul`. -/
@[simp] theorem realTensor_matmul {m n p : Nat} (a : Tensor ℚ [m, n]) (b : Tensor ℚ [n, p]) :
    realTensor (Tensor.matmul a b) = Tensor.matmul (realTensor a) (realTensor b) := by
  apply Tensor.Rep.ext
  intro i
  simp only [realTensor, Tensor.map_apply,
    Tensor.matmul_eq_ofFn, Tensor.Rep.get_ofFn]
  conv_rhs => rw [← Rat.cast_zero]
  refine (List.foldl_hom (fun q : ℚ => (q : ℝ)) ?_).symm
  intro acc k
  simp [get2_eq_getScalar_unstack, Tensor.getScalar, Tensor.item]

/-- `realTensor` commutes with the positive part `IBP.matPos` of a weight matrix. -/
@[simp] theorem realTensor_matPos {m n : Nat} (w : Tensor ℚ [m, n]) :
    realTensor (IBP.matPos w) = IBP.matPos (realTensor w) := by
  apply Tensor.Rep.ext
  rintro ⟨i, j, ⟨⟩⟩
  by_cases h : 0 < w (i, j, PUnit.unit) <;>
    simp [h, realTensor, IBP.matPos]

/-- `realTensor` commutes with the negative part `IBP.matNeg` of a weight matrix. -/
@[simp] theorem realTensor_matNeg {m n : Nat} (w : Tensor ℚ [m, n]) :
    realTensor (IBP.matNeg w) = IBP.matNeg (realTensor w) := by
  apply Tensor.Rep.ext
  rintro ⟨i, j, ⟨⟩⟩
  by_cases h : 0 < w (i, j, PUnit.unit) <;>
    simp [h, realTensor, IBP.matNeg]

/-- Interpret both endpoints of a rational box over the reals. -/
noncomputable def realBox {s : Shape} (b : Box ℚ s) : Box ℝ s :=
  ⟨realTensor b.lo, realTensor b.hi⟩

/-- Interpret the coefficients of an exact affine form. -/
noncomputable def realAffine {n m : Nat} (a : Linear ℚ n m) : Linear ℝ n m :=
  ⟨realTensor a.weights, realTensor a.bias⟩

/-- Interpret a pair of exact affine bounds. -/
noncomputable def realAffineBounds (b : FlatAffineBounds ℚ) : FlatAffineBounds ℝ :=
  ⟨b.inDim, b.outDim, realAffine b.loAff, realAffine b.hiAff⟩

/-- Casting the output dimension of an affine form along `m = m` leaves it unchanged. -/
@[simp] theorem castAffineOut_self {α : Type} [Context α] {n m : Nat}
    (a : Linear α n m) (h : m = m) :
    CROWN.Graph.castAffineOut h a = a := by
  rfl

/-- `realAffineBounds` commutes with `CROWN.Graph.propagateLinearBounds`. -/
@[simp] theorem realAffineBounds_linear {n m : Nat}
    (w : Tensor ℚ [m, n]) (bias : Tensor ℚ [m]) (b : FlatAffineBounds ℚ)
    (h : b.outDim = n) :
    realAffineBounds (CROWN.Graph.propagateLinearBounds w bias b h) =
      CROWN.Graph.propagateLinearBounds
        (realTensor w) (realTensor bias) (realAffineBounds b) h := by
  rcases b with ⟨inDim, outDim, lo, hi⟩
  dsimp at h
  subst outDim
  simp [CROWN.Graph.propagateLinearBounds, realAffineBounds, realAffine]
  exact ⟨rfl, rfl⟩

/-- `realAffineBounds` commutes with `CROWN.Graph.boundsConst`. -/
@[simp] theorem realAffineBounds_const (n m : Nat) (lo hi : Tensor ℚ [m]) :
    realAffineBounds (CROWN.Graph.boundsConst n m lo hi) =
      CROWN.Graph.boundsConst n m (realTensor lo) (realTensor hi) := by
  simp [CROWN.Graph.boundsConst, realAffineBounds, realAffine]

/-- `realAffineBounds` maps the identity bounds over `ℚ` to the identity bounds over `ℝ`. -/
@[simp] theorem realAffineBounds_identity (n : Nat) :
    realAffineBounds (CROWN.Graph.boundsIdentity n) =
      CROWN.Graph.boundsIdentity n := by
  have hmatrix :
      realTensor (CROWN.Graph.affIdentity (α := ℚ) n).weights =
        (CROWN.Graph.affIdentity (α := ℝ) n).weights := by
    apply Tensor.Rep.ext
    rintro ⟨i, j, ⟨⟩⟩
    simp [CROWN.Graph.affIdentity, realTensor,
      Tensor.identity, apply_ite]
  have hconstant : realTensor (CROWN.Graph.affIdentity (α := ℚ) n).bias =
      (CROWN.Graph.affIdentity (α := ℝ) n).bias := by
    simp [CROWN.Graph.affIdentity]
  simp only [CROWN.Graph.boundsIdentity, realAffineBounds, realAffine,
    hmatrix, hconstant]

/-- `realTensor` commutes with `Tensor.relu`. -/
@[simp] theorem realTensor_relu {s : Shape} (t : Tensor ℚ s) :
    realTensor (Tensor.relu t) = Tensor.relu (realTensor t) := by
  apply Tensor.Rep.ext
  intro i
  simp [realTensor, Tensor.relu,
    Activation.relu_eq_max, Rat.cast_max]

/-- Interpret the slope and offset of a ReLU relaxation exactly. -/
noncomputable def realRelax (r : CROWN.ReLURelax ℚ) : CROWN.ReLURelax ℝ :=
  ⟨(r.slope : ℝ), (r.bias : ℝ)⟩

/-- `realRelax` commutes with the ReLU relaxation `ReLU.relaxScalar`. -/
@[simp] theorem realRelax_upper (l u : ℚ) :
    realRelax (ReLU.relaxScalar l u) =
      ReLU.relaxScalar (l : ℝ) (u : ℝ) := by
  by_cases hu : 0 < u <;> by_cases hl : 0 < l <;>
    simp [ReLU.relaxScalar, realRelax, hu, hl]

/-- `realRelax` commutes with the lower relaxation `CROWN.Cert.alphaRelaxLowerScalar`. -/
@[simp] theorem realRelax_lower (l u a : ℚ) :
    realRelax (CROWN.Cert.alphaRelaxLowerScalar l u a) =
      CROWN.Cert.alphaRelaxLowerScalar (l : ℝ) (u : ℝ) (a : ℝ) := by
  by_cases hu : 0 < u <;> by_cases hl : 0 < l <;>
    simp [CROWN.Cert.alphaRelaxLowerScalar, realRelax, hu, hl]

/-- `realRelax` commutes entrywise with `ReLU.relaxVector`. -/
@[simp] theorem realRelax_upperVec {n : Nat} (lo hi : Tensor ℚ [n]) :
    Tensor.map realRelax (ReLU.relaxVector lo hi) =
      ReLU.relaxVector (realTensor lo) (realTensor hi) := by
  apply Tensor.ext_vector
  intro i
  simp [ReLU.relaxVector]

/-- `realRelax` commutes entrywise with `CROWN.Cert.alphaRelaxLowerVec`. -/
@[simp] theorem realRelax_lowerVec {n : Nat} (lo hi alpha : Tensor ℚ [n]) :
    Tensor.map realRelax (CROWN.Cert.alphaRelaxLowerVec lo hi alpha) =
      CROWN.Cert.alphaRelaxLowerVec
        (realTensor lo) (realTensor hi) (realTensor alpha) := by
  apply Tensor.ext_vector
  intro i
  simp [CROWN.Cert.alphaRelaxLowerVec]

/-- `realAffine` commutes with `ReLU.propagateAffine`. -/
@[simp] theorem realAffine_propagate {n m : Nat}
    (r : Tensor (CROWN.ReLURelax ℚ) [m]) (a : Linear ℚ n m) :
    realAffine (ReLU.propagateAffine r a) =
      ReLU.propagateAffine (Tensor.map realRelax r) (realAffine a) := by
  rcases a with ⟨w, bias⟩
  apply congrArg₂ (Linear.mk (α := ℝ) (inDim := n) (outDim := m))
  · apply Tensor.Rep.ext
    rintro ⟨i, j, ⟨⟩⟩
    simp [ReLU.propagateAffine, realAffine, realTensor, realRelax,
      get2_eq_getScalar_unstack, Tensor.getScalar, Tensor.item,
      Tensor.scalar]
  · apply Tensor.ext_vector
    intro i
    simp [ReLU.propagateAffine, realAffine, realTensor, realRelax]

private theorem cast_minimum (a b : ℚ) :
    ((Context.minimum a b : ℚ) : ℝ) = Context.minimum (a : ℝ) (b : ℝ) := by
  rw [RationalAlgebraic.minimum_eq_min, Context.minimum_real, Rat.cast_min]

private theorem cast_maximum (a b : ℚ) :
    ((Context.maximum a b : ℚ) : ℝ) = Context.maximum (a : ℝ) (b : ℝ) := by
  rw [RationalAlgebraic.maximum_eq_max, Context.maximum_real, Rat.cast_max]

/-- `realBox` commutes with `IBP.linear`. -/
@[simp] theorem realBox_linear {m n : Nat} (w : Tensor ℚ [m, n])
    (x : Box ℚ [n]) (bias : Box ℚ [m]) :
    realBox (IBP.linear w x bias) =
      IBP.linear (realTensor w) (realBox x) (realBox bias) := by
  unfold IBP.linear realBox
  congr 1
  · apply Tensor.ext_vector
    intro i
    simp only [realTensor_getScalar, getScalar_dim_entry, item_scalar]
    simp only [BoundOps.addDown, Rat.cast_add]
    congr 1
    conv_rhs => rw [← Rat.cast_zero]
    refine (List.foldl_hom (fun q : ℚ => (q : ℝ)) ?_).symm
    intro acc k
    simp only [BoundOps.mulDown, Rat.cast_add, cast_minimum, Rat.cast_mul, realTensor_get2]
  · apply Tensor.ext_vector
    intro i
    simp only [realTensor_getScalar, getScalar_dim_entry, item_scalar]
    simp only [BoundOps.addUp, Rat.cast_add]
    congr 1
    conv_rhs => rw [← Rat.cast_zero]
    refine (List.foldl_hom (fun q : ℚ => (q : ℝ)) ?_).symm
    intro acc k
    simp only [BoundOps.mulUp, Rat.cast_add, cast_maximum, Rat.cast_mul, realTensor_get2]

end TorchLean.Certificate.Rational
