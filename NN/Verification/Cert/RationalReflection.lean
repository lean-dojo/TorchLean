/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Algebra
public import NN.MLTheory.CROWN.Cert.AlphaCROWN
public import NN.MLTheory.CROWN.BoundOps.Lawful
public import NN.Spec.Core.Context.Rational

/-!
# Exact rational reflection for algebraic certificates

These lemmas connect executable rational tensor arithmetic to the existing real-valued tensor
semantics. No floating-point rounding or transcendental approximation is involved.
-/

@[expose] public section

namespace Spec.RationalAlgebraic

open NN.MLTheory.CROWN
open scoped Spec.RationalAlgebraic

/-- Rational certificate arithmetic is exact, including affine reassociation. -/
scoped instance instBoundOpsRat : BoundOps ℚ where
  addDown := (· + ·)
  addUp := (· + ·)
  subDown := (· - ·)
  subUp := (· - ·)
  mulDown := (· * ·)
  mulUp := (· * ·)
  supportsExactAffineReassociation := true

/-- The rational arithmetic dictionary satisfies the real enclosure laws. -/
noncomputable scoped instance instLawfulBoundOpsRat : LawfulBoundOps ℚ where
  toReal := fun q => (q : ℝ)
  lt_iff a b := by exact_mod_cast (Iff.rfl : a < b ↔ a < b)
  toReal_zero := Rat.cast_zero
  toReal_one := Rat.cast_one
  toReal_max a b := Rat.cast_max a b
  toReal_eq_of_beq h := by rw [beq_iff_eq.mp h]
  addDown_le a b := by simp [BoundOps.addDown]
  le_addUp a b := by simp [BoundOps.addUp]
  subDown_le a b := by simp [BoundOps.subDown]
  le_subUp a b := by simp [BoundOps.subUp]
  mulDown_le a b := by simp [BoundOps.mulDown]
  le_mulUp a b := by simp [BoundOps.mulUp]

/-- The rational backend's minimum agrees with its real interpretation. -/
noncomputable scoped instance instLawfulMinBoundOpsRat : LawfulMinBoundOps ℚ where
  toReal_min a b := Rat.cast_min a b

end Spec.RationalAlgebraic

namespace NN.Verification.Cert.RationalReflection

open _root_.Spec TorchLean TorchLean.Tensor
open NN.MLTheory.CROWN NN.MLTheory.CROWN.Graph
open scoped Spec.RationalAlgebraic BigOperators

/-- Interpret each rational tensor entry as a real number. -/
noncomputable def realTensor {s : Shape} (t : Tensor ℚ s) : Tensor ℝ s :=
  Tensor.map (fun q : ℚ => (q : ℝ)) t

@[simp] theorem realTensor_get {n : Nat} {s : Shape} (t : Tensor ℚ (.dim n s))
    (i : Fin n) : get (realTensor t) i = realTensor (get t i) := by
  simp [realTensor, Spec.get]

@[simp] theorem realTensor_getScalar {n : Nat} (t : Tensor ℚ [n]) (i : Fin n) :
    (realTensor t).getScalar i = (t.getScalar i : ℝ) := by
  simp [realTensor]

@[simp] theorem realTensor_get2 {m n : Nat} (t : Tensor ℚ [m, n]) (i : Fin m) (j : Fin n) :
    get2 (realTensor t) i j = ((get2 t i j : ℚ) : ℝ) := by
  simp [get2_eq_getScalar_get]

@[simp] theorem realTensor_fill {s : Shape} (q : ℚ) :
    realTensor (Tensor.full s q) = Tensor.full s (q : ℝ) := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro i
  simp [realTensor, Tensor.map]

@[simp] theorem realTensor_add {s : Shape} (a b : Tensor ℚ s) :
    realTensor (Tensor.addSpec a b) = Tensor.addSpec (realTensor a) (realTensor b) := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro i
  simp [realTensor, Tensor.map, Tensor.addSpec, Tensor.map2Spec]

@[simp] theorem realTensor_matVecMul {m n : Nat} (a : Tensor ℚ [m, n]) (b : Tensor ℚ [n]) :
    realTensor (Spec.matVecMulSpec a b) = Spec.matVecMulSpec (realTensor a) (realTensor b) := by
  apply Tensor.ext_vector
  intro i
  simp [Proofs.TensorAlgebra.getScalar_mat_vec_mul_spec]

private theorem cast_foldl {ι : Type} (xs : List ι)
    (f : ℚ → ι → ℚ) (g : ℝ → ι → ℝ)
    (h : ∀ acc i, (f acc i : ℝ) = g (acc : ℝ) i) (init : ℚ) :
    ((xs.foldl f init : ℚ) : ℝ) = xs.foldl g (init : ℝ) := by
  exact (List.foldl_hom (fun q : ℚ => (q : ℝ)) (fun acc i => (h acc i).symm)).symm

@[simp] theorem realTensor_matMul {m n p : Nat} (a : Tensor ℚ [m, n]) (b : Tensor ℚ [n, p]) :
    realTensor (Spec.matMulSpec a b) = Spec.matMulSpec (realTensor a) (realTensor b) := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro i
  simp only [realTensor, Tensor.map, TorchLean.Tensor.Internal.Rep.map_apply,
    Spec.matMulSpec, TorchLean.Tensor.Internal.Rep.get_ofFn]
  conv_rhs => rw [← Rat.cast_zero]
  apply cast_foldl
  intro acc k
  simp [get2_eq_getScalar_get, Spec.get, Tensor.getScalar, Tensor.item,
    Tensor.unstack]

@[simp] theorem realTensor_matPos {m n : Nat} (w : Tensor ℚ [m, n]) :
    realTensor (IBP.matPos w) = IBP.matPos (realTensor w) := by
  apply TorchLean.Tensor.Internal.Rep.ext
  rintro ⟨i, j, ⟨⟩⟩
  by_cases h : 0 < w (i, j, PUnit.unit) <;>
    simp [h, realTensor, IBP.matPos, get2_eq_getScalar_get, Spec.get, Tensor.unstack,
      Tensor.getScalar, Tensor.item, Tensor.map, Tensor.dim, Tensor.scalar]

@[simp] theorem realTensor_matNeg {m n : Nat} (w : Tensor ℚ [m, n]) :
    realTensor (IBP.matNeg w) = IBP.matNeg (realTensor w) := by
  apply TorchLean.Tensor.Internal.Rep.ext
  rintro ⟨i, j, ⟨⟩⟩
  by_cases h : 0 < w (i, j, PUnit.unit) <;>
    simp [h, realTensor, IBP.matNeg, get2_eq_getScalar_get, Spec.get, Tensor.unstack,
      Tensor.getScalar, Tensor.item, Tensor.map, Tensor.dim, Tensor.scalar]

/-- Interpret both endpoints of a rational box over the reals. -/
noncomputable def realBox {s : Shape} (b : Box ℚ s) : Box ℝ s :=
  ⟨realTensor b.lo, realTensor b.hi⟩

/-- Interpret a dimension-carrying rational box over the reals. -/
noncomputable def realFlatBox (b : FlatBox ℚ) : FlatBox ℝ :=
  ⟨b.dim, realTensor b.lo, realTensor b.hi⟩

/-- Interpret the coefficients of an exact affine form. -/
noncomputable def realAffine {n m : Nat} (a : AffineVec ℚ n m) : AffineVec ℝ n m :=
  ⟨realTensor a.A, realTensor a.c⟩

/-- Interpret a pair of exact affine bounds. -/
noncomputable def realAffineBounds (b : FlatAffineBounds ℚ) : FlatAffineBounds ℝ :=
  ⟨b.inDim, b.outDim, realAffine b.loAff, realAffine b.hiAff⟩

@[simp] theorem castAffineOut_self {α : Type} [Context α] {n m : Nat}
    (a : AffineVec α n m) (h : m = m) :
    NN.MLTheory.CROWN.Graph.castAffineOut h a = a := by
  rfl

@[simp] theorem realAffineBounds_linear {n m : Nat}
    (w : Tensor ℚ [m, n]) (bias : Tensor ℚ [m]) (b : FlatAffineBounds ℚ)
    (h : b.outDim = n) :
    realAffineBounds (NN.MLTheory.CROWN.Cert.linearBoundsFromAffine w bias b h) =
      NN.MLTheory.CROWN.Cert.linearBoundsFromAffine
        (realTensor w) (realTensor bias) (realAffineBounds b) h := by
  rcases b with ⟨inDim, outDim, lo, hi⟩
  dsimp at h
  subst outDim
  simp [NN.MLTheory.CROWN.Cert.linearBoundsFromAffine, realAffineBounds, realAffine]

@[simp] theorem realAffineBounds_const (n m : Nat) (lo hi : Tensor ℚ [m]) :
    realAffineBounds (NN.MLTheory.CROWN.Cert.boundsConst n m lo hi) =
      NN.MLTheory.CROWN.Cert.boundsConst n m (realTensor lo) (realTensor hi) := by
  simp [NN.MLTheory.CROWN.Cert.boundsConst, realAffineBounds, realAffine]

@[simp] theorem realAffineBounds_identity (n : Nat) :
    realAffineBounds (NN.MLTheory.CROWN.Cert.boundsIdentity n) =
      NN.MLTheory.CROWN.Cert.boundsIdentity n := by
  have hmatrix :
      realTensor (NN.MLTheory.CROWN.Cert.affIdentity (α := ℚ) n).A =
        (NN.MLTheory.CROWN.Cert.affIdentity (α := ℝ) n).A := by
    apply TorchLean.Tensor.Internal.Rep.ext
    rintro ⟨i, j, ⟨⟩⟩
    simp [NN.MLTheory.CROWN.Cert.affIdentity, realTensor, Tensor.map, Tensor.dim,
      Tensor.scalar, apply_ite]
  have hconstant : realTensor (NN.MLTheory.CROWN.Cert.affIdentity (α := ℚ) n).c =
      (NN.MLTheory.CROWN.Cert.affIdentity (α := ℝ) n).c := by
    simp [NN.MLTheory.CROWN.Cert.affIdentity]
  simp only [NN.MLTheory.CROWN.Cert.boundsIdentity, realAffineBounds, realAffine,
    hmatrix, hconstant]

@[simp] theorem realTensor_relu {s : Shape} (t : Tensor ℚ s) :
    realTensor (Activation.reluSpec t) = Activation.reluSpec (realTensor t) := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro i
  simp [realTensor, Tensor.map, Activation.reluSpec, Tensor.mapSpec,
    Activation.Math.reluSpec_eq_max, Rat.cast_max]

/-- Interpret the slope and offset of a ReLU relaxation exactly. -/
noncomputable def realRelax (r : Runtime.Ops.ReLURelax ℚ) : Runtime.Ops.ReLURelax ℝ :=
  ⟨(r.slope : ℝ), (r.bias : ℝ)⟩

@[simp] theorem realRelax_upper (l u : ℚ) :
    realRelax (Runtime.Ops.ReLU.relaxScalar l u) =
      Runtime.Ops.ReLU.relaxScalar (l : ℝ) (u : ℝ) := by
  by_cases hu : 0 < u <;> by_cases hl : 0 < l <;>
    simp [Runtime.Ops.ReLU.relaxScalar, realRelax, hu, hl]

@[simp] theorem realRelax_lower (l u a : ℚ) :
    realRelax (NN.MLTheory.CROWN.Cert.alphaRelaxLowerScalar l u a) =
      NN.MLTheory.CROWN.Cert.alphaRelaxLowerScalar (l : ℝ) (u : ℝ) (a : ℝ) := by
  by_cases hu : 0 < u <;> by_cases hl : 0 < l <;>
    simp [NN.MLTheory.CROWN.Cert.alphaRelaxLowerScalar, realRelax, hu, hl]

@[simp] theorem realRelax_upperVec {n : Nat} (lo hi : Tensor ℚ [n]) :
    Tensor.map realRelax (Runtime.Ops.ReLU.relaxVector lo hi) =
      Runtime.Ops.ReLU.relaxVector (realTensor lo) (realTensor hi) := by
  apply Tensor.ext_vector
  intro i
  simp [Runtime.Ops.ReLU.relaxVector]

@[simp] theorem realRelax_lowerVec {n : Nat} (lo hi alpha : Tensor ℚ [n]) :
    Tensor.map realRelax (NN.MLTheory.CROWN.Cert.alphaRelaxLowerVec lo hi alpha) =
      NN.MLTheory.CROWN.Cert.alphaRelaxLowerVec
        (realTensor lo) (realTensor hi) (realTensor alpha) := by
  apply Tensor.ext_vector
  intro i
  simp [NN.MLTheory.CROWN.Cert.alphaRelaxLowerVec]

@[simp] theorem realAffine_propagate {n m : Nat}
    (r : Tensor (Runtime.Ops.ReLURelax ℚ) [m]) (a : AffineVec ℚ n m) :
    realAffine (Runtime.Ops.ReLU.propagateAffine r a) =
      Runtime.Ops.ReLU.propagateAffine (Tensor.map realRelax r) (realAffine a) := by
  rcases a with ⟨w, bias⟩
  apply congrArg₂ (AffineVec.mk (α := ℝ) (inDim := n) (outDim := m))
  · apply TorchLean.Tensor.Internal.Rep.ext
    rintro ⟨i, j, ⟨⟩⟩
    simp [Runtime.Ops.ReLU.propagateAffine, realAffine, realTensor, realRelax,
      get2_eq_getScalar_get, Spec.get, Tensor.unstack, Tensor.getScalar, Tensor.item,
      Tensor.map, Tensor.dim, Tensor.scalar]
  · apply Tensor.ext_vector
    intro i
    simp [Runtime.Ops.ReLU.propagateAffine, realAffine, realTensor, realRelax]

private theorem cast_min2 (a b : ℚ) :
    ((BoundOps.min2 a b : ℚ) : ℝ) = BoundOps.min2 (a : ℝ) (b : ℝ) := by
  by_cases h : b < a <;> simp [BoundOps.min2, h]

private theorem cast_max2 (a b : ℚ) :
    ((BoundOps.max2 a b : ℚ) : ℝ) = BoundOps.max2 (a : ℝ) (b : ℝ) := by
  by_cases h : b < a <;> simp [BoundOps.max2, h]

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
    apply cast_foldl
    intro acc k
    simp only [BoundOps.mulDown, Rat.cast_add, cast_min2, Rat.cast_mul, realTensor_get2]
  · apply Tensor.ext_vector
    intro i
    simp only [realTensor_getScalar, getScalar_dim_entry, item_scalar]
    simp only [BoundOps.addUp, Rat.cast_add]
    congr 1
    conv_rhs => rw [← Rat.cast_zero]
    apply cast_foldl
    intro acc k
    simp only [BoundOps.mulUp, Rat.cast_add, cast_max2, Rat.cast_mul, realTensor_get2]

end NN.Verification.Cert.RationalReflection
