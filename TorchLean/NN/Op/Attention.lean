/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Op.Softmax
public import TorchLean.Tensor.Op.Table.Attention
public import TorchLean.NN.Op.Matmul
public import TorchLean.NN.Attention.HardMask
public import TorchLean.NN.Attention.HardMask.FDeriv
public import TorchLean.NN.Tactic.OpDeriv
public import TorchLean.Tensor.AdjacentSwap.Adjoint

/-!
# Derivatives of the attention operators

We prove that the stored backward of masked softmax and of attention after its score product is
the adjoint of the Fréchet derivative of the real forward, at every constructor argument
(`Op.maskedSoftmax_backward_eq_adjoint`, `Op.attentionFromScores_prod_backward_eq_adjoint`). A
fiber of a hard-masked softmax is the smooth map `HardMaskedSoftmax.weights`, which is zero on a
fiber without an allowed entry, so the certificates need no hypothesis.

The derivative of masked softmax is `fiberwiseCLM` of the row Jacobians
`HardMaskedSoftmax.derivative`. The derivative of `Attention.values` after `Attention.weights`,
jointly in the scores, the value and the multiplier, is `Attention.valuesCLM`, the tangent
`Attention.valuesJVP` transposed by `Attention.valuesBackward` (`Attention.dot_valuesJVP`).
With the `@[fun_prop]` rules `HasPullbackAt.maskedSoftmax` and `HasPullbackAt.attentionValues`,
`op_deriv` proves the certificates. Attention without a multiplier is the same value product with
the multiplier one (`Attention.values_none_eq`) after the scaled score product and the key
transpose (`HasPullbackAt.batchTranspose`). Its certificate at every schedule is in
`TorchLean.Verify.RuntimeApprox.OnlineSoftmax.FDeriv`, since it needs the theorem that over the
reals the online schedule computes the two-pass forward.

For multi-head attention we prove the step of one head: scaled dot-product attention
`Attention.scaledDotProduct` has the derivative `Attention.scaledDotProductCLM` jointly in the
query, the key and the value, with the stored `Attention.scaledDotProductBackward` as its adjoint
(`Attention.hasPullbackAt_scaledDotProduct`). Stacked over the heads, its derivative is
`Attention.headsCLM` with the adjoint `Attention.headsBackward` (`Attention.hasPullbackAt_heads`).
The `@[fun_prop]` rules `HasPullbackAt.splitHeads`, `HasPullbackAt.combineHeads`,
`HasPullbackAt.matmul` and `HasPullbackAt.unstack` give the head split, the head merge, the matrix
products and the slices along the leading axis, and `HasPullbackAt.attentionHeads` gives the
attention of every head. With no leading axes, these rules prove that the stored backward of
multi-head attention is the adjoint of the derivative, jointly in the four projections and the
input (`Op.multiheadAttention_single_backward_eq_adjoint`).

With leading axes, the four projections are shared by the sequences. `HasPullbackAt.stackSlices`
gives the pullback of a map with a shared argument applied to every slice along a leading axis:
it adds the shared components and stacks the slice components. By induction on the leading
shape, `MultiHeadAttention.backwardLeading` with the parameter gradients added by
`MultiHeadAttention.sum` is the pullback of `Tensor.mapLeading` of the forward
(`MultiHeadAttention.hasPullbackAt_mapLeading`). With the `@[fun_prop]` rule
`HasPullbackAt.multiheadAttention` for this map, `op_deriv` proves the certificate at every leading
shape (`Op.multiheadAttention_backward_eq_adjoint`). The `op_deriv_simps` lemmas
`Op.maskedSoftmaxBackward_add` and `Op.maskedSoftmaxBackward_smul` state that the stored masked
softmax pullback is linear in the cotangent.
-/

@[expose] public section

open TorchLean.Shape TorchLean.Tensor
open scoped RealInnerProductSpace

namespace TorchLean

noncomputable section

/-- Over the reals, the hard-masked vector pullback applies `HardMaskedSoftmax.derivative` to the
upstream gradient, also on a vector without an allowed entry. -/
theorem Softmax.getScalarE_hardMaskedPullback {n : Nat} (scores dY : Tensor ℝ [n])
    (mask : Tensor Bool [n]) : getScalarE (hardMaskedPullback scores mask dY) =
      HardMaskedSoftmax.derivative mask (getScalarE scores) (getScalarE dY) := by
  unfold hardMaskedPullback
  rcases hp : hardMaskedPivot? scores mask with _ | pivot
  · have hvec : hardMaskedVec scores mask = Tensor.full _ 0 := by
      unfold hardMaskedVec
      rw [hardMaskedMax?_eq_map, hp]
      rfl
    have hw : HardMaskedSoftmax.weights mask (getScalarE scores) = 0 := by
      rw [← getScalarE_hardMaskedVec, hvec]
      ext i
      simp
    ext i
    simp [hw]
  · cases n with
    | zero => exact Fin.elim0 pivot
    | succ n =>
      have hw (i : Fin (n + 1)) : (hardMaskedVec scores mask).getScalar i =
          HardMaskedSoftmax.weights mask (getScalarE scores) i :=
        getScalar_hardMaskedVec scores mask i
      have hmass : (∑ j, (hardMaskedVec scores mask).getScalar j) = 1 ∨
          ∀ j, (hardMaskedVec scores mask).getScalar j = 0 := by
        by_cases hallowed : ∃ j, mask.getScalar j = true
        · left
          simp_rw [hw]
          change (∑ j, HardMaskedSoftmax.numerator mask (getScalarE scores) j /
            HardMaskedSoftmax.denominator mask (getScalarE scores)) = 1
          rw [← Finset.sum_div]
          exact div_self (ne_of_gt
            (HardMaskedSoftmax.denominator_pos mask (getScalarE scores) hallowed))
        · right
          have hall (j : Fin (n + 1)) : mask.getScalar j = false := by
            cases hj : mask.getScalar j
            · rfl
            · exact (hallowed ⟨j, hj⟩).elim
          intro j
          simp [hw, HardMaskedSoftmax.weights, HardMaskedSoftmax.numerator, hall]
      have hshift : pullback (hardMaskedVec scores mask) dY (dY.getScalar pivot) =
          backwardFromWeights (hardMaskedVec scores mask)
            (dY - Tensor.full _ (dY.getScalar pivot)) := rfl
      dsimp only
      rw [hshift, ← backwardFromWeights_shift _ _ _ hmass,
        HardMaskedSoftmax.backward_eq_derivative]

namespace Op

variable {s : Shape} {n : Nat} {ι : Type}

/-- If `f r` has derivative `L r` at the fiber at `r` of `x` for every `r`, a map `F` that applies
`f r` to the fiber at `r` for every `r` has derivative `fiberwiseCLM e L` at `x`. -/
theorem fiberwise_hasFDerivAt_family (e : s.Coord ≃ Fin n × ι) (F : Tensor ℝ s → Tensor ℝ s)
    (f : ι → Vec n → Vec n) (L : ι → Vec n →L[ℝ] Vec n) (x : Tensor ℝ s)
    (hF : ∀ y c, F y c = f (e c).2 (fiberCLM e (e c).2 y) (e c).1)
    (hf : ∀ r, HasFDerivAt (f r) (L r) (fiberCLM e r x)) : HasFDerivAt F (fiberwiseCLM e L) x := by
  refine Tensor.hasFDerivAt_of_coordinates F _ x fun c => ?_
  have h := (EuclideanSpace.proj (𝕜 := ℝ) (e c).1).hasFDerivAt.comp x
    ((hf (e c).2).comp x (fiberCLM e (e c).2).hasFDerivAt)
  have hfun : (fun y => F y c) =
      (EuclideanSpace.proj (𝕜 := ℝ) (e c).1) ∘ f (e c).2 ∘ (fiberCLM e (e c).2) := by
    funext y
    rw [hF]
    rfl
  rw [hfun]
  exact h

/-- The fiber at `r` of a Boolean mask under the splitting `e`: entry `j` is the entry of the mask
at `e.symm (j, r)`. -/
def maskFiber (e : s.Coord ≃ Fin n × ι) (mask : Tensor Bool s) (r : ι) : Tensor Bool [n] :=
  Tensor.ofFn fun j => mask (e.symm (j, r))

/-- The derivative of masked softmax along `dim` at `x`: on the fiber at `r` it is the row
Jacobian `HardMaskedSoftmax.derivative` of the mask fiber and the score fiber at `r`. -/
def maskedSoftmaxCLM (x : Tensor ℝ s) (mask : Tensor Bool s) (dim : Fin s.rank) :
    Tensor ℝ s →L[ℝ] Tensor ℝ s :=
  fiberwiseCLM (Coord.axisEquiv dim (shape := s)) fun r =>
    HardMaskedSoftmax.derivative (maskFiber (Coord.axisEquiv dim (shape := s)) mask r)
      (fiberCLM (Coord.axisEquiv dim (shape := s)) r x)

/-- The entry of `Tensor.softmax y dim (some mask)` at `c` is the smooth masked weight
`HardMaskedSoftmax.weights` of the mask fiber and the score fiber through `c`. -/
theorem softmax_some_apply_weights (y : Tensor ℝ s) (dim : Fin s.rank) (mask : Tensor Bool s)
    (c : s.Coord) : (Tensor.softmax y dim (some mask)) c =
      HardMaskedSoftmax.weights
        (maskFiber (Coord.axisEquiv dim (shape := s)) mask (Coord.axisEquiv dim (shape := s) c).2)
        (fiberCLM (Coord.axisEquiv dim (shape := s)) (Coord.axisEquiv dim (shape := s) c).2 y)
        (Coord.axisEquiv dim (shape := s) c).1 := by
  rw [Tensor.softmax_mask_apply, ← getScalarE_ofLp, Softmax.getScalarE_hardMaskedVec,
    getScalarE_fiber]
  rfl

/-- The entry of `Tensor.maskedSoftmaxBackward x mask δ dim` at `c` is the row Jacobian
`HardMaskedSoftmax.derivative` at the fibers of the mask and `x` through `c`, applied to the
fiber of `δ` through `c`. -/
theorem maskedSoftmaxBackward_apply_derivative (x δ : Tensor ℝ s) (mask : Tensor Bool s)
    (dim : Fin s.rank) (c : s.Coord) : Tensor.maskedSoftmaxBackward x mask δ dim c =
      HardMaskedSoftmax.derivative
        (maskFiber (Coord.axisEquiv dim (shape := s)) mask (Coord.axisEquiv dim (shape := s) c).2)
        (fiberCLM (Coord.axisEquiv dim (shape := s)) (Coord.axisEquiv dim (shape := s) c).2 x)
        (fiberCLM (Coord.axisEquiv dim (shape := s)) (Coord.axisEquiv dim (shape := s) c).2 δ)
        (Coord.axisEquiv dim (shape := s) c).1 := by
  rw [Tensor.maskedSoftmaxBackward_apply, ← getScalarE_ofLp,
    Softmax.getScalarE_hardMaskedPullback, getScalarE_fiber, getScalarE_fiber]
  rfl

/-- Masked softmax along `dim` has derivative `maskedSoftmaxCLM x mask dim` at every `x`, and the
stored backward `Tensor.maskedSoftmaxBackward` is its adjoint. -/
theorem hasPullbackAt_maskedSoftmax (dim : Fin s.rank) (mask : Tensor Bool s) (x : Tensor ℝ s) :
    HasPullbackAt (fun y => Tensor.softmax y dim (some mask)) (maskedSoftmaxCLM x mask dim)
      (fun δ => Tensor.maskedSoftmaxBackward x mask δ dim) x :=
  HasPullbackAt.of_adjoint
    (fiberwise_hasFDerivAt_family _ _
      (fun r => HardMaskedSoftmax.weights (maskFiber (Coord.axisEquiv dim (shape := s)) mask r))
      _ x (fun y c => softmax_some_apply_weights y dim mask c)
      (fun _ => HardMaskedSoftmax.hasFDerivAt_weights _ _)) (fun δ => (fiberwiseCLM_adjoint _ _
      (fun r => HardMaskedSoftmax.derivative (maskFiber (Coord.axisEquiv dim (shape := s)) mask r)
        (fiberCLM (Coord.axisEquiv dim (shape := s)) r x))
      (fun _ u w => HardMaskedSoftmax.inner_derivative _ _ u w) δ _
      (fun c => maskedSoftmaxBackward_apply_derivative x δ mask dim c)).symm)

end Op

namespace HasPullbackAt

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] {x : E} {s : Shape}
  {f : E → Tensor ℝ s} {Df : E →L[ℝ] Tensor ℝ s} {bf : Tensor ℝ s → E}

/-- Masked softmax along `dim` after `f`: the derivative is `maskedSoftmaxCLM (f x) mask dim`
after `Df`, and the pullback applies the stored masked-softmax backward first. -/
@[fun_prop]
theorem maskedSoftmax (dim : Fin s.rank) (mask : Tensor Bool s) (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => Tensor.softmax (f y) dim (some mask))
      ((Op.maskedSoftmaxCLM (f x) mask dim).comp Df)
      (fun δ => bf (Tensor.maskedSoftmaxBackward (f x) mask δ dim)) x :=
  (Op.hasPullbackAt_maskedSoftmax dim mask (f x)).comp hf

end HasPullbackAt

namespace Op

variable {s : Shape}

/-- At every shape, axis and Boolean mask, masked softmax is differentiable in its scores and its
stored backward is the adjoint of the derivative. A fiber without an allowed entry is the constant
zero map. -/
@[op_cert TorchLean.Op.maskedSoftmax derivative]
theorem maskedSoftmax_backward_eq_adjoint (dim : Fin s.rank) (x : Tensor ℝ s)
    (mask : Tensor Bool s) (δ : Tensor ℝ s) :
    DifferentiableAt ℝ (fun y => (((Op.maskedSoftmax dim).rule? (α := ℝ) |>.get rfl).forward
        (y, mask, ())).1) x ∧
      (((Op.maskedSoftmax dim).rule? (α := ℝ) |>.get rfl).backward (x, mask, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.maskedSoftmax dim).rule? (α := ℝ) |>.get rfl).forward
          (y, mask, ())).1) x).adjoint δ := by
  op_deriv

/-- The stored masked softmax pullback maps a sum of cotangents to the sum of their pullbacks. -/
@[op_deriv_simps] theorem maskedSoftmaxBackward_add (x : Tensor ℝ s) (mask : Tensor Bool s)
    (dim : Fin s.rank) (δ δ' : Tensor ℝ s) : Tensor.maskedSoftmaxBackward x mask (δ + δ') dim =
      Tensor.maskedSoftmaxBackward x mask δ dim + Tensor.maskedSoftmaxBackward x mask δ' dim := by
  simp only [(hasPullbackAt_maskedSoftmax dim mask x).eq_adjoint, map_add]

/-- The stored masked softmax pullback commutes with multiplying the cotangent by `k`. -/
@[op_deriv_simps] theorem maskedSoftmaxBackward_smul (x : Tensor ℝ s) (mask : Tensor Bool s)
    (dim : Fin s.rank) (k : ℝ) (δ : Tensor ℝ s) : Tensor.maskedSoftmaxBackward x mask (k • δ) dim =
      k • Tensor.maskedSoftmaxBackward x mask δ dim := by
  simp only [(hasPullbackAt_maskedSoftmax dim mask x).eq_adjoint, map_smul]

end Op

namespace Attention

variable {batch : Shape} {rows keys width : Nat}

/-- The splitting of a coordinate of score matrices with the leading axes `batch` into its key and
the pair of its batch coordinate and its row. -/
def rowEquiv (batch : Shape) (rows keys : Nat) :
    (batch ++ [rows, keys]).Coord ≃ Fin keys × (batch.Coord × Fin rows) :=
  (Shape.Coord.appendEquiv batch [rows, keys]).trans
    { toFun := fun q => (q.2.2.1, (q.1, q.2.1))
      invFun := fun r => (r.2.1, (r.2.2, (r.1, PUnit.unit)))
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }

/-- Row `i` of the score matrix at the batch coordinate `b`, as a Euclidean vector, is the fiber at
`(b, i)` under `rowEquiv`. -/
theorem getScalarE_unstack_batchSlice (S : Tensor ℝ (batch ++ [rows, keys])) (b : batch.Coord)
    (i : Fin rows) :
    getScalarE ((S.batchSlice b).unstack i) = Op.fiberCLM (rowEquiv batch rows keys) (b, i) S := by
  ext k
  rw [getScalarE_ofLp, Op.fiberCLM_apply, ← get2_eq_getScalar_unstack, get2_eq_apply,
    batchSlice_apply]
  rfl

/-- Over the reals, the entry of the hard-masked weights of the score matrix at `b` in row `i` and
key `k` is `HardMaskedSoftmax.weights` of the mask row and the fiber of `S` at `(b, i)`. -/
theorem hardMasked_batchSlice_apply (S : Tensor ℝ (batch ++ [rows, keys]))
    (mask : Tensor Bool [rows, keys]) (b : batch.Coord) (i : Fin rows) (k : Fin keys) :
    Softmax.hardMasked (S.batchSlice b) mask (i, k, PUnit.unit) =
      HardMaskedSoftmax.weights (mask.unstack i) (Op.fiberCLM (rowEquiv batch rows keys) (b, i) S)
        k := by
  rw [← get2_eq_apply, get2_eq_getScalar_unstack, Softmax.unstack_hardMasked, ← getScalarE_ofLp,
    Softmax.getScalarE_hardMaskedVec, getScalarE_unstack_batchSlice]

/-- Over the reals, the entry of the centered softmax pullback of the hard-masked weights of the
score matrix at `b`, in row `i` and key `k`, is `HardMaskedSoftmax.derivative` of the mask row at
the fiber of `S` at `(b, i)`, applied to the fiber of `dS`. -/
theorem backwardFromWeights_batchSlice_apply (S dS : Tensor ℝ (batch ++ [rows, keys]))
    (mask : Tensor Bool [rows, keys]) (b : batch.Coord) (i : Fin rows) (k : Fin keys) :
    Softmax.backwardFromWeights (Softmax.hardMasked (S.batchSlice b) mask) (dS.batchSlice b)
        (center := true) (i, k, PUnit.unit) =
      HardMaskedSoftmax.derivative (mask.unstack i)
        (Op.fiberCLM (rowEquiv batch rows keys) (b, i) S)
        (Op.fiberCLM (rowEquiv batch rows keys) (b, i) dS) k := by
  rw [← get2_eq_apply, get2_eq_getScalar_unstack,
    Softmax.unstack_backwardFromWeights (center := true), Softmax.unstack_hardMasked,
    ← getScalarE_ofLp, HardMaskedSoftmax.backward_eq_derivative (center := true),
    getScalarE_unstack_batchSlice, getScalarE_unstack_batchSlice]

/-- Over the reals, an entry of `values (weights S mask) V (some M)` is the sum over the keys of the
smooth hard-masked weight, the multiplier entry and the value entry. -/
theorem values_weights_some_apply (S M : Tensor ℝ (batch ++ [rows, keys]))
    (V : Tensor ℝ (batch ++ [keys, width])) (mask : Option (Tensor Bool [rows, keys]))
    (b : batch.Coord) (i : Fin rows) (d : Fin width) : values (weights S mask) V (some M)
        ((Shape.Coord.appendEquiv batch [rows, width]).symm (b, (i, d, PUnit.unit))) =
      ∑ k, HardMaskedSoftmax.weights ((mask.getD (allTrueMask rows keys)).unstack i)
          (Op.fiberCLM (rowEquiv batch rows keys) (b, i) S) k *
        M ((Shape.Coord.appendEquiv batch [rows, keys]).symm (b, (i, k, PUnit.unit))) *
        V ((Shape.Coord.appendEquiv batch [keys, width]).symm (b, (k, d, PUnit.unit))) := by
  rw [values_apply]
  simp only [Equiv.apply_symm_apply, Option.elim_some]
  rw [weights_batchSlice, weightsSlice_eq_hardMasked, ← get2_eq_apply,
    get2_matmul_eq_sum]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [get2_eq_apply, get2_eq_apply, Tensor.mul_apply, hardMasked_batchSlice_apply,
    batchSlice_apply, batchSlice_apply]

/-- Over the reals, an entry of `valuesJVP (weights S mask) V M dS dV dM` is the sum over the keys
of the product rule for the weight, multiplier and value entries, with the weight tangent
`HardMaskedSoftmax.derivative` at the score fiber. -/
theorem valuesJVP_weights_apply (S M dS dM : Tensor ℝ (batch ++ [rows, keys]))
    (V dV : Tensor ℝ (batch ++ [keys, width])) (mask : Option (Tensor Bool [rows, keys]))
    (b : batch.Coord) (i : Fin rows) (d : Fin width) : valuesJVP (weights S mask) V M dS dV dM
        ((Shape.Coord.appendEquiv batch [rows, width]).symm (b, (i, d, PUnit.unit))) =
      ∑ k, ((HardMaskedSoftmax.derivative ((mask.getD (allTrueMask rows keys)).unstack i)
              (Op.fiberCLM (rowEquiv batch rows keys) (b, i) S)
              (Op.fiberCLM (rowEquiv batch rows keys) (b, i) dS) k *
            M ((Shape.Coord.appendEquiv batch [rows, keys]).symm (b, (i, k, PUnit.unit))) +
          HardMaskedSoftmax.weights ((mask.getD (allTrueMask rows keys)).unstack i)
              (Op.fiberCLM (rowEquiv batch rows keys) (b, i) S) k *
            dM ((Shape.Coord.appendEquiv batch [rows, keys]).symm (b, (i, k, PUnit.unit)))) *
          V ((Shape.Coord.appendEquiv batch [keys, width]).symm (b, (k, d, PUnit.unit))) +
        HardMaskedSoftmax.weights ((mask.getD (allTrueMask rows keys)).unstack i)
            (Op.fiberCLM (rowEquiv batch rows keys) (b, i) S) k *
          M ((Shape.Coord.appendEquiv batch [rows, keys]).symm (b, (i, k, PUnit.unit))) *
          dV ((Shape.Coord.appendEquiv batch [keys, width]).symm (b, (k, d, PUnit.unit)))) := by
  rw [valuesJVP_apply]
  simp only [Equiv.apply_symm_apply]
  rw [weights_batchSlice, weightsSlice_eq_hardMasked]
  simp only [valuesJVPSlice, Numeric.QuotientArithmetic.supported_real, Bool.not_false,
    Bool.false_eq_true, ↓reduceIte,
    scaledProductsSlice_two Numeric.QuotientArithmetic.supported_real, Tensor.scale_one]
  rw [← get2_eq_apply, get2_add, get2_matmul_eq_sum, get2_matmul_eq_sum, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun k _ => ?_
  simp only [get2_eq_apply, Tensor.mul_apply, Tensor.add_apply, batchSlice_apply]
  rw [hardMasked_batchSlice_apply, backwardFromWeights_batchSlice_apply]

/-- The value tangent `valuesJVP` at the hard-masked weights of `S` is transposed by the three
components of `valuesBackward`, packed as an L2 product. -/
theorem inner_valuesJVP (S M : Tensor ℝ (batch ++ [rows, keys]))
    (V : Tensor ℝ (batch ++ [keys, width])) (mask : Option (Tensor Bool [rows, keys]))
    (u : WithLp 2 (Tensor ℝ (batch ++ [rows, keys]) ×
      WithLp 2 (Tensor ℝ (batch ++ [keys, width]) × Tensor ℝ (batch ++ [rows, keys]))))
    (δ : Tensor ℝ (batch ++ [rows, width])) :
    ⟪valuesJVP (weights S mask) V M u.fst u.snd.fst u.snd.snd, δ⟫ =
      ⟪u, WithLp.toLp 2 ((valuesBackward (weights S mask) V M δ).1,
        WithLp.toLp 2 ((valuesBackward (weights S mask) V M δ).2.1,
          (valuesBackward (weights S mask) V M δ).2.2))⟫ := by
  have h := dot_valuesJVP S u.fst M u.snd.snd V u.snd.fst δ mask
  simp only [dot_eq_inner] at h
  rw [h, WithLp.prod_inner_apply, WithLp.prod_inner_apply]
  rfl

/-- The derivative of `values (weights S mask) V (some M)` jointly in the scores, the value and the
multiplier: the tangent `valuesJVP` at the hard-masked weights of `S`. -/
def valuesCLM (S : Tensor ℝ (batch ++ [rows, keys])) (V : Tensor ℝ (batch ++ [keys, width]))
    (M : Tensor ℝ (batch ++ [rows, keys])) (mask : Option (Tensor Bool [rows, keys])) :
    WithLp 2 (Tensor ℝ (batch ++ [rows, keys]) ×
        WithLp 2 (Tensor ℝ (batch ++ [keys, width]) × Tensor ℝ (batch ++ [rows, keys]))) →L[ℝ]
      Tensor ℝ (batch ++ [rows, width]) :=
  clmOfTranspose (fun u => valuesJVP (weights S mask) V M u.fst u.snd.fst u.snd.snd)
    (fun δ => WithLp.toLp 2 ((valuesBackward (weights S mask) V M δ).1,
      WithLp.toLp 2 ((valuesBackward (weights S mask) V M δ).2.1,
        (valuesBackward (weights S mask) V M δ).2.2)))
    (inner_valuesJVP S M V mask)

/-- `valuesCLM S V M mask` applied to `u` is `valuesJVP` along the three components of `u`. -/
@[simp] theorem valuesCLM_apply (S : Tensor ℝ (batch ++ [rows, keys]))
    (V : Tensor ℝ (batch ++ [keys, width])) (M : Tensor ℝ (batch ++ [rows, keys]))
    (mask : Option (Tensor Bool [rows, keys]))
    (u : WithLp 2 (Tensor ℝ (batch ++ [rows, keys]) ×
      WithLp 2 (Tensor ℝ (batch ++ [keys, width]) × Tensor ℝ (batch ++ [rows, keys])))) :
    valuesCLM S V M mask u = valuesJVP (weights S mask) V M u.fst u.snd.fst u.snd.snd :=
  rfl

/-- Over the reals, `values (weights S mask) V (some M)` has derivative `valuesCLM S V M mask`
jointly in the scores, the value and the multiplier, at every point and for every mask. -/
theorem hasFDerivAt_values_weights (mask : Option (Tensor Bool [rows, keys]))
    (S M : Tensor ℝ (batch ++ [rows, keys])) (V : Tensor ℝ (batch ++ [keys, width])) :
    HasFDerivAt (fun p : WithLp 2 (Tensor ℝ (batch ++ [rows, keys]) ×
        WithLp 2 (Tensor ℝ (batch ++ [keys, width]) × Tensor ℝ (batch ++ [rows, keys]))) =>
        values (weights p.fst mask) p.snd.fst (some p.snd.snd))
      (valuesCLM S V M mask) (WithLp.toLp 2 (S, WithLp.toLp 2 (V, M))) := by
  have key (c : (batch ++ [rows, width]).Coord) :
      HasFDerivAt (fun p : WithLp 2 (Tensor ℝ (batch ++ [rows, keys]) ×
          WithLp 2 (Tensor ℝ (batch ++ [keys, width]) × Tensor ℝ (batch ++ [rows, keys]))) =>
          values (weights p.fst mask) p.snd.fst (some p.snd.snd) c)
        ((coordinateCLM c).comp (valuesCLM S V M mask))
        (WithLp.toLp 2 (S, WithLp.toLp 2 (V, M))) := by
    obtain ⟨⟨b, i, d, ⟨⟩⟩, rfl⟩ : ∃ q, (Shape.Coord.appendEquiv batch [rows, width]).symm q = c :=
      ⟨_, Equiv.symm_apply_apply _ _⟩
    have hw (k : Fin keys) := (EuclideanSpace.proj (𝕜 := ℝ) k).hasFDerivAt.comp
      (WithLp.toLp 2 (S, WithLp.toLp 2 (V, M)))
      ((HardMaskedSoftmax.hasFDerivAt_weights ((mask.getD (allTrueMask rows keys)).unstack i)
        (Op.fiberCLM (rowEquiv batch rows keys) (b, i) S)).comp
        (WithLp.toLp 2 (S, WithLp.toLp 2 (V, M)))
        ((Op.fiberCLM (rowEquiv batch rows keys) (b, i)).comp
          (WithLp.fstL 2 ℝ (Tensor ℝ (batch ++ [rows, keys]))
            (WithLp 2 (Tensor ℝ (batch ++ [keys, width]) × Tensor ℝ (batch ++ [rows, keys]))))
          ).hasFDerivAt)
    have hm (k : Fin keys) :=
      ((coordinateCLM ((Shape.Coord.appendEquiv batch [rows, keys]).symm
        (b, (i, k, PUnit.unit)))).comp
        ((WithLp.sndL 2 ℝ (Tensor ℝ (batch ++ [keys, width]))
          (Tensor ℝ (batch ++ [rows, keys]))).comp
          (WithLp.sndL 2 ℝ (Tensor ℝ (batch ++ [rows, keys]))
            (WithLp 2 (Tensor ℝ (batch ++ [keys, width]) × Tensor ℝ (batch ++ [rows, keys])))))
        ).hasFDerivAt (x := WithLp.toLp 2 (S, WithLp.toLp 2 (V, M)))
    have hv (k : Fin keys) :=
      ((coordinateCLM ((Shape.Coord.appendEquiv batch [keys, width]).symm
        (b, (k, d, PUnit.unit)))).comp
        ((WithLp.fstL 2 ℝ (Tensor ℝ (batch ++ [keys, width]))
          (Tensor ℝ (batch ++ [rows, keys]))).comp
          (WithLp.sndL 2 ℝ (Tensor ℝ (batch ++ [rows, keys]))
            (WithLp 2 (Tensor ℝ (batch ++ [keys, width]) × Tensor ℝ (batch ++ [rows, keys])))))
        ).hasFDerivAt (x := WithLp.toLp 2 (S, WithLp.toLp 2 (V, M)))
    have hsum := HasFDerivAt.fun_sum (u := Finset.univ) fun k _ =>
      ((hw k).mul (hm k)).mul (hv k)
    have hfun : (fun p : WithLp 2 (Tensor ℝ (batch ++ [rows, keys]) ×
          WithLp 2 (Tensor ℝ (batch ++ [keys, width]) × Tensor ℝ (batch ++ [rows, keys]))) =>
          values (weights p.fst mask) p.snd.fst (some p.snd.snd)
            ((Shape.Coord.appendEquiv batch [rows, width]).symm (b, (i, d, PUnit.unit)))) =
        fun p => ∑ k, HardMaskedSoftmax.weights ((mask.getD (allTrueMask rows keys)).unstack i)
          (Op.fiberCLM (rowEquiv batch rows keys) (b, i) p.fst) k *
          p.snd.snd ((Shape.Coord.appendEquiv batch [rows, keys]).symm (b, (i, k, PUnit.unit))) *
          p.snd.fst ((Shape.Coord.appendEquiv batch [keys, width]).symm (b, (k, d, PUnit.unit))) :=
      funext fun p => values_weights_some_apply p.fst p.snd.snd p.snd.fst mask b i d
    rw [hfun]
    refine hsum.congr_fderiv (ContinuousLinearMap.ext fun u => ?_)
    rw [ContinuousLinearMap.comp_apply, coordinateCLM_apply, valuesCLM_apply,
      valuesJVP_weights_apply, _root_.sum_apply]
    refine Finset.sum_congr rfl fun k _ => ?_
    simp only [_root_.add_apply, _root_.smul_apply, ContinuousLinearMap.comp_apply, smul_eq_mul,
      coordinateCLM_apply, WithLp.fstL_apply, WithLp.sndL_apply, EuclideanSpace.coe_proj,
      Function.comp_apply, Pi.mul_apply, WithLp.toLp_fst, WithLp.toLp_snd]
    ring
  convert Tensor.hasFDerivAt_of_coordinates _ _ _ key using 1
  exact ContinuousLinearMap.ext fun u => Tensor.Rep.ext fun c => by
    rw [assemble_apply]
    rfl

/-- Over the reals, `values (weights S mask) V (some M)` has the stored backward `valuesBackward`
as the adjoint of its derivative, jointly in the scores, the value and the multiplier. -/
theorem hasPullbackAt_values (mask : Option (Tensor Bool [rows, keys]))
    (S M : Tensor ℝ (batch ++ [rows, keys])) (V : Tensor ℝ (batch ++ [keys, width])) :
    HasPullbackAt (fun p : WithLp 2 (Tensor ℝ (batch ++ [rows, keys]) ×
        WithLp 2 (Tensor ℝ (batch ++ [keys, width]) × Tensor ℝ (batch ++ [rows, keys]))) =>
        values (weights p.fst mask) p.snd.fst (some p.snd.snd))
      (valuesCLM S V M mask) (fun δ => WithLp.toLp 2 ((valuesBackward (weights S mask) V M δ).1,
        WithLp.toLp 2 ((valuesBackward (weights S mask) V M δ).2.1,
          (valuesBackward (weights S mask) V M δ).2.2)))
      (WithLp.toLp 2 (S, WithLp.toLp 2 (V, M))) :=
  ⟨hasFDerivAt_values_weights mask S M V, inner_valuesJVP S M V mask⟩

end Attention

namespace HasPullbackAt

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] {x : E}
  {batch : Shape} {rows keys width : Nat}
  {f : E → Tensor ℝ (batch ++ [rows, keys])} {g : E → Tensor ℝ (batch ++ [keys, width])}
  {h : E → Tensor ℝ (batch ++ [rows, keys])}
  {Df : E →L[ℝ] Tensor ℝ (batch ++ [rows, keys])} {Dg : E →L[ℝ] Tensor ℝ (batch ++ [keys, width])}
  {Dh : E →L[ℝ] Tensor ℝ (batch ++ [rows, keys])}
  {bf : Tensor ℝ (batch ++ [rows, keys]) → E} {bg : Tensor ℝ (batch ++ [keys, width]) → E}
  {bh : Tensor ℝ (batch ++ [rows, keys]) → E}

/-- Attention after its score product applied to three maps: the derivative is `valuesCLM` after
the derivatives of the scores, the value and the multiplier, and the pullback applies the three
components of `valuesBackward`. -/
@[fun_prop]
theorem attentionValues (mask : Option (Tensor Bool [rows, keys]))
    (hf : HasPullbackAt f Df bf x) (hg : HasPullbackAt g Dg bg x) (hh : HasPullbackAt h Dh bh x) :
    HasPullbackAt (fun y => Attention.values (Attention.weights (f y) mask) (g y) (some (h y)))
      ((Attention.valuesCLM (f x) (g x) (h x) mask).comp
        ((WithLp.prodContinuousLinearEquiv 2 ℝ _ _).symm.toContinuousLinearMap.comp
          (Df.prod ((WithLp.prodContinuousLinearEquiv 2 ℝ _ _).symm.toContinuousLinearMap.comp
            (Dg.prod Dh)))))
      (fun δ => bf (Attention.valuesBackward (Attention.weights (f x) mask) (g x) (h x) δ).1 +
        (bg (Attention.valuesBackward (Attention.weights (f x) mask) (g x) (h x) δ).2.1 +
          bh (Attention.valuesBackward (Attention.weights (f x) mask) (g x) (h x) δ).2.2)) x :=
  HasPullbackAt.comp (f := fun y => WithLp.toLp 2 (f y, WithLp.toLp 2 (g y, h y)))
    (Attention.hasPullbackAt_values mask (f x) (h x) (g x)) (hf.toLp_prod (hg.toLp_prod hh))

end HasPullbackAt

namespace Op

/-- At every leading shape, mask and matrix sizes, attention after its score product is
differentiable jointly in the scores, the value and the multiplier, and its stored backward is the
adjoint of the derivative. A row without an allowed key has constant zero weights. -/
@[op_cert TorchLean.Op.attentionFromScores derivative 0 1 2]
theorem attentionFromScores_prod_backward_eq_adjoint {batch : Shape} {rows keys width : Nat}
    (mask : Option (Tensor Bool [rows, keys])) (s : Tensor ℝ (batch ++ [rows, keys]))
    (v : Tensor ℝ (batch ++ [keys, width])) (m : Tensor ℝ (batch ++ [rows, keys]))
    (δ : Tensor ℝ (batch ++ [rows, width])) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ (batch ++ [rows, keys]) ×
          WithLp 2 (Tensor ℝ (batch ++ [keys, width]) × Tensor ℝ (batch ++ [rows, keys]))) =>
        (((Op.attentionFromScores (batch := batch) (width := width) mask).rule? (α := ℝ)
          |>.get rfl).forward (p.fst, p.snd.fst, p.snd.snd, ())).1)
      (WithLp.toLp 2 (s, WithLp.toLp 2 (v, m))) ∧
    WithLp.toLp 2
        ((((Op.attentionFromScores (batch := batch) (width := width) mask).rule? (α := ℝ)
          |>.get rfl).backward (s, v, m, ()) (δ, ())).1,
        WithLp.toLp 2
          ((((Op.attentionFromScores (batch := batch) (width := width) mask).rule? (α := ℝ)
            |>.get rfl).backward (s, v, m, ()) (δ, ())).2.1,
          (((Op.attentionFromScores (batch := batch) (width := width) mask).rule? (α := ℝ)
            |>.get rfl).backward (s, v, m, ()) (δ, ())).2.2.1)) =
      (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ (batch ++ [rows, keys]) ×
          WithLp 2 (Tensor ℝ (batch ++ [keys, width]) × Tensor ℝ (batch ++ [rows, keys]))) =>
        (((Op.attentionFromScores (batch := batch) (width := width) mask).rule? (α := ℝ)
          |>.get rfl).forward (p.fst, p.snd.fst, p.snd.snd, ())).1)
        (WithLp.toLp 2 (s, WithLp.toLp 2 (v, m)))).adjoint δ := by
  op_deriv

end Op

namespace HasPullbackAt

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] {x : E} {batch : Shape}
  {m n : Nat} {f : E → Tensor ℝ (batch ++ [m, n])} {Df : E →L[ℝ] Tensor ℝ (batch ++ [m, n])}
  {bf : Tensor ℝ (batch ++ [m, n]) → E}

/-- The transpose of the trailing matrix axes after `f`: the derivative is `permuteCLM` after
`Df`, and the pullback transposes the cotangent back before `bf`. -/
@[fun_prop]
theorem batchTranspose (hout : batch ++ [n, m] =
      (batch ++ [m, n]).permute (Shape.adjacentSwap (batch ++ [m, n]) batch.rank))
    (hf : HasPullbackAt f Df bf x) : HasPullbackAt
      (fun y => Tensor.permute (f y) (Shape.adjacentSwap (batch ++ [m, n]) batch.rank)
        (batch ++ [n, m]) hout)
      ((Tensor.permuteCLM (Shape.adjacentSwap (batch ++ [m, n]) batch.rank) (batch ++ [n, m])
        hout).comp Df)
      (fun δ => bf (Tensor.permute δ (Shape.adjacentSwap (batch ++ [n, m]) batch.rank)
        (batch ++ [m, n]))) x := by
  simpa only [Tensor.permuteCLM_apply] using
    clm (Tensor.permuteCLM (Shape.adjacentSwap (batch ++ [m, n]) batch.rank) (batch ++ [n, m])
        hout)
      (fun δ => Tensor.permute δ (Shape.adjacentSwap (batch ++ [n, m]) batch.rank)
        (batch ++ [m, n]))
      (fun u δ => by
        rw [Tensor.permuteCLM_apply, ← dot_eq_inner, ← dot_eq_inner,
          dot_permute_batch_transpose])
      hf

end HasPullbackAt

namespace Attention

variable {batch : Shape} {rows keys width : Nat}

/-- Over the reals, the value product without a multiplier is the value product with the
multiplier one. -/
theorem values_none_eq (W : Tensor ℝ (batch ++ [rows, keys]))
    (V : Tensor ℝ (batch ++ [keys, width])) :
    values W V none = values W V (some (Tensor.full _ 1)) := by
  have h (b : batch.Coord) :
      W.batchSlice b * (Tensor.full (batch ++ [rows, keys]) (1 : ℝ)).batchSlice b =
        W.batchSlice b :=
    Tensor.Rep.ext fun q => by
      rw [Tensor.mul_apply, batchSlice_apply, batchSlice_apply, Tensor.full_apply, mul_one]
  refine Tensor.Rep.ext fun c => ?_
  simp only [values_apply, Option.elim_none, Option.elim_some, h]

end Attention

namespace HasPullbackAt

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] {x : E} {m n : Nat}
  {f : E → Tensor ℝ [m, n]} {Df : E →L[ℝ] Tensor ℝ [m, n]} {bf : Tensor ℝ [m, n] → E}

/-- The transpose of a matrix-valued map: the derivative is `permuteCLM` after `Df`, and the
pullback transposes the cotangent back before `bf`. -/
theorem matrixTranspose (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => Tensor.permute (f y) (Shape.adjacentSwap [m, n] 0) [n, m])
      ((Tensor.permuteCLM (Shape.adjacentSwap [m, n] 0) [n, m]).comp Df)
      (fun δ => bf (Tensor.permute δ (Shape.adjacentSwap [n, m] 0) [m, n])) x :=
  batchTranspose (batch := .scalar) _ hf

end HasPullbackAt

namespace Attention

section ScaledDotProduct

variable {m n p : Nat}

/-- Over the reals, the scaled matrix product without leading axes is the matrix product scaled
by `c`. -/
theorem scaledMatmul_scalar (A : Tensor ℝ [m, n]) (B : Tensor ℝ [n, p]) (c : ℝ) :
    scaledMatmul (Shape.CanBroadcastTo.refl .scalar) (Shape.CanBroadcastTo.refl .scalar) A B c =
      Tensor.scale (Tensor.matmul A B) c := by
  simp only [scaledMatmul, Tensor.broadcast_self, scaledProducts_one_eq_scale,
    Tensor.matmulCommonBatch_scalar]

/-- Row `i` of a score matrix without leading axes, as a Euclidean vector, is the fiber at
`(PUnit.unit, i)` under `rowEquiv`. -/
theorem getScalarE_unstack_eq_fiber (S : Tensor ℝ [m, n]) (i : Fin m) :
    getScalarE (S.unstack i) = Op.fiberCLM (rowEquiv .scalar m n) (PUnit.unit, i) S := by
  ext k
  rw [getScalarE_ofLp, Op.fiberCLM_apply, ← get2_eq_getScalar_unstack, get2_eq_apply]
  rfl

variable {nQ nK d : Nat} {hQ : nQ ≠ 0} {hK : nK ≠ 0}

/-- Over the reals, entry `(i, j)` of scaled dot-product attention is the sum over the keys of the
smooth hard-masked weight of row `i` of the scaled scores and the value entry, where no mask
counts as the all-true mask. -/
theorem scaledDotProduct_apply (Q : Tensor ℝ [nQ, d]) (K V : Tensor ℝ [nK, d])
    (mask : Option (Tensor Bool [nQ, nK])) (i : Fin nQ) (j : Fin d) :
    scaledDotProduct (⟨Q, K, V, mask⟩ : Inputs ℝ nQ nK d hQ hK) (i, j, PUnit.unit) =
      ∑ k, HardMaskedSoftmax.weights ((mask.getD (allTrueMask nQ nK)).unstack i)
          (Op.fiberCLM (rowEquiv .scalar nQ nK) (PUnit.unit, i)
            (scaledMatmul (Shape.CanBroadcastTo.refl .scalar) (Shape.CanBroadcastTo.refl .scalar)
              Q (Tensor.permute K (Shape.adjacentSwap [nK, d] 0) [d, nK])
              (1 / scaleDenom (α := ℝ) d))) k *
        V (k, j, PUnit.unit) := by
  have hmask : scaledDotProduct (⟨Q, K, V, mask⟩ : Inputs ℝ nQ nK d hQ hK) =
      scaledDotProduct (⟨Q, K, V, some (mask.getD (allTrueMask nQ nK))⟩ :
        Inputs ℝ nQ nK d hQ hK) := by
    cases mask with
    | none => exact (scaledDotProduct_allTrueMask (h1 := hQ) (h2 := hK) ⟨Q, K, V, none⟩).symm
    | some _ => rfl
  rw [hmask, scaledMatmul_scalar]
  simp only [scaledDotProduct, scaledProductsSlice_one Numeric.QuotientArithmetic.supported_real,
    Bool.false_eq_true, ↓reduceIte]
  rw [← get2_eq_apply, get2_matmul_eq_sum]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [get2_eq_apply V, get2_eq_getScalar_unstack, Softmax.unstack_hardMasked, ← getScalarE_ofLp,
    Softmax.getScalarE_hardMaskedVec, getScalarE_unstack_eq_fiber]

/-- Over the reals, entry `(i, j)` of the tangent `scaledDotProductJVP` is the sum over the keys of
the product rule for the weight and value entries, with the weight tangent
`HardMaskedSoftmax.derivative` at row `i` of the scaled scores. -/
theorem scaledDotProductJVP_apply (Q dQ : Tensor ℝ [nQ, d]) (K V dK dV : Tensor ℝ [nK, d])
    (mask : Option (Tensor Bool [nQ, nK])) (i : Fin nQ) (j : Fin d) :
    scaledDotProductJVP (⟨Q, K, V, mask⟩ : Inputs ℝ nQ nK d hQ hK) dQ dK dV (i, j, PUnit.unit) =
      ∑ k, (HardMaskedSoftmax.derivative ((mask.getD (allTrueMask nQ nK)).unstack i)
            (Op.fiberCLM (rowEquiv .scalar nQ nK) (PUnit.unit, i)
              (scaledMatmul (Shape.CanBroadcastTo.refl .scalar) (Shape.CanBroadcastTo.refl .scalar)
                Q (Tensor.permute K (Shape.adjacentSwap [nK, d] 0) [d, nK])
                (1 / scaleDenom (α := ℝ) d)))
            (Op.fiberCLM (rowEquiv .scalar nQ nK) (PUnit.unit, i)
              (scaledMatmul (Shape.CanBroadcastTo.refl .scalar) (Shape.CanBroadcastTo.refl .scalar)
                  dQ (Tensor.permute K (Shape.adjacentSwap [nK, d] 0) [d, nK])
                  (1 / scaleDenom (α := ℝ) d) +
                scaledMatmul (Shape.CanBroadcastTo.refl .scalar) (Shape.CanBroadcastTo.refl .scalar)
                  Q (Tensor.permute dK (Shape.adjacentSwap [nK, d] 0) [d, nK])
                  (1 / scaleDenom (α := ℝ) d))) k *
          V (k, j, PUnit.unit) +
        HardMaskedSoftmax.weights ((mask.getD (allTrueMask nQ nK)).unstack i)
            (Op.fiberCLM (rowEquiv .scalar nQ nK) (PUnit.unit, i)
              (scaledMatmul (Shape.CanBroadcastTo.refl .scalar) (Shape.CanBroadcastTo.refl .scalar)
                Q (Tensor.permute K (Shape.adjacentSwap [nK, d] 0) [d, nK])
                (1 / scaleDenom (α := ℝ) d))) k *
          dV (k, j, PUnit.unit)) := by
  have hmask : scaledDotProductJVP (⟨Q, K, V, mask⟩ : Inputs ℝ nQ nK d hQ hK) dQ dK dV =
      scaledDotProductJVP (⟨Q, K, V, some (mask.getD (allTrueMask nQ nK))⟩ :
        Inputs ℝ nQ nK d hQ hK) dQ dK dV := by
    cases mask with
    | none =>
      exact (scaledDotProductJVP_allTrueMask (h1 := hQ) (h2 := hK) ⟨Q, K, V, none⟩ dQ dK dV).symm
    | some _ => rfl
  have hadd (A B : Tensor ℝ [nQ, nK]) (c : ℝ) :
      Tensor.scale (A + B) c = Tensor.scale A c + Tensor.scale B c :=
    Tensor.ext_get2 fun r k => by simp only [get2_add, get2_scale, add_mul]
  rw [hmask]
  simp only [scaledMatmul_scalar]
  rw [← hadd]
  simp only [scaledDotProductJVP, pullbackProducts, Numeric.QuotientArithmetic.supported_real,
    Bool.not_false, scaledProductsSlice_one Numeric.QuotientArithmetic.supported_real,
    scaledProductsSlice_two Numeric.QuotientArithmetic.supported_real, Bool.false_eq_true,
    ↓reduceIte]
  rw [← get2_eq_apply, get2_add, get2_matmul_eq_sum, get2_matmul_eq_sum, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [get2_eq_apply V, get2_eq_apply dV]
  simp only [get2_eq_getScalar_unstack, Softmax.unstack_backwardFromWeights (center := true),
    Softmax.unstack_hardMasked, ← getScalarE_ofLp,
    HardMaskedSoftmax.backward_eq_derivative (center := true), Softmax.getScalarE_hardMaskedVec,
    getScalarE_unstack_eq_fiber]

/-- The tangent `scaledDotProductJVP` at `ctx` is transposed by the three components of
`scaledDotProductBackward`, packed as an L2 product, for every mask. -/
theorem inner_scaledDotProductJVP (ctx : Inputs ℝ nQ nK d hQ hK)
    (u : WithLp 2 (Tensor ℝ [nQ, d] × WithLp 2 (Tensor ℝ [nK, d] × Tensor ℝ [nK, d])))
    (δ : Tensor ℝ [nQ, d]) :
    ⟪scaledDotProductJVP ctx u.fst u.snd.fst u.snd.snd, δ⟫ =
      ⟪u, WithLp.toLp 2 ((scaledDotProductBackward ctx δ).1,
        WithLp.toLp 2 ((scaledDotProductBackward ctx δ).2.1,
          (scaledDotProductBackward ctx δ).2.2))⟫ := by
  have key (ctx : Inputs ℝ nQ nK d hQ hK) (M : Tensor Bool [nQ, nK]) (hM : ctx.mask = some M) :
      ⟪scaledDotProductJVP ctx u.fst u.snd.fst u.snd.snd, δ⟫ =
        ⟪u, WithLp.toLp 2 ((scaledDotProductBackward ctx δ).1,
          WithLp.toLp 2 ((scaledDotProductBackward ctx δ).2.1,
            (scaledDotProductBackward ctx δ).2.2))⟫ := by
    have h := dot_scaledDotProductJVP ctx M hM u.fst u.snd.fst u.snd.snd δ
    simp only [dot_eq_inner] at h
    rw [h, WithLp.prod_inner_apply, WithLp.prod_inner_apply]
    rfl
  obtain ⟨Q, K, V, mask⟩ := ctx
  cases mask with
  | some M => exact key _ M rfl
  | none =>
    rw [← scaledDotProductJVP_allTrueMask (h1 := hQ) (h2 := hK) ⟨Q, K, V, none⟩,
      ← scaledDotProductBackward_allTrueMask (h1 := hQ) (h2 := hK) ⟨Q, K, V, none⟩]
    exact key _ _ rfl

/-- The derivative of scaled dot-product attention at `ctx` jointly in the query, the key and the
value: the tangent `scaledDotProductJVP` transposed by `scaledDotProductBackward`. -/
def scaledDotProductCLM (ctx : Inputs ℝ nQ nK d hQ hK) :
    WithLp 2 (Tensor ℝ [nQ, d] × WithLp 2 (Tensor ℝ [nK, d] × Tensor ℝ [nK, d])) →L[ℝ]
      Tensor ℝ [nQ, d] :=
  clmOfTranspose (fun u => scaledDotProductJVP ctx u.fst u.snd.fst u.snd.snd)
    (fun δ => WithLp.toLp 2 ((scaledDotProductBackward ctx δ).1,
      WithLp.toLp 2 ((scaledDotProductBackward ctx δ).2.1, (scaledDotProductBackward ctx δ).2.2)))
    (inner_scaledDotProductJVP ctx)

/-- `scaledDotProductCLM ctx` applied to `u` is `scaledDotProductJVP` along the three components
of `u`. -/
@[simp] theorem scaledDotProductCLM_apply (ctx : Inputs ℝ nQ nK d hQ hK)
    (u : WithLp 2 (Tensor ℝ [nQ, d] × WithLp 2 (Tensor ℝ [nK, d] × Tensor ℝ [nK, d]))) :
    scaledDotProductCLM ctx u = scaledDotProductJVP ctx u.fst u.snd.fst u.snd.snd :=
  rfl

/-- Over the reals, scaled dot-product attention has derivative `scaledDotProductCLM` jointly in
the query, the key and the value, at every point and for every mask. -/
theorem hasFDerivAt_scaledDotProduct (mask : Option (Tensor Bool [nQ, nK]))
    (Q : Tensor ℝ [nQ, d]) (K V : Tensor ℝ [nK, d]) :
    HasFDerivAt (fun p : WithLp 2 (Tensor ℝ [nQ, d] ×
        WithLp 2 (Tensor ℝ [nK, d] × Tensor ℝ [nK, d])) =>
        scaledDotProduct (⟨p.fst, p.snd.fst, p.snd.snd, mask⟩ : Inputs ℝ nQ nK d hQ hK))
      (scaledDotProductCLM (⟨Q, K, V, mask⟩ : Inputs ℝ nQ nK d hQ hK))
      (WithLp.toLp 2 (Q, WithLp.toLp 2 (K, V))) := by
  have key (c : Shape.Coord [nQ, d]) :
      HasFDerivAt (fun p : WithLp 2 (Tensor ℝ [nQ, d] ×
          WithLp 2 (Tensor ℝ [nK, d] × Tensor ℝ [nK, d])) =>
          scaledDotProduct (⟨p.fst, p.snd.fst, p.snd.snd, mask⟩ : Inputs ℝ nQ nK d hQ hK) c)
        ((coordinateCLM c).comp
          (scaledDotProductCLM (⟨Q, K, V, mask⟩ : Inputs ℝ nQ nK d hQ hK)))
        (WithLp.toLp 2 (Q, WithLp.toLp 2 (K, V))) := by
    obtain ⟨i, j, ⟨⟩⟩ := c
    have hS := (HasPullbackAt.scaledMatmul (Shape.CanBroadcastTo.refl .scalar)
      (Shape.CanBroadcastTo.refl .scalar) (1 / scaleDenom (α := ℝ) d)
      (HasPullbackAt.fst (HasPullbackAt.id (WithLp.toLp 2 (Q, WithLp.toLp 2 (K, V)))))
      (HasPullbackAt.matrixTranspose (HasPullbackAt.fst (HasPullbackAt.snd
        (HasPullbackAt.id (WithLp.toLp 2 (Q, WithLp.toLp 2 (K, V)))))))).hasFDerivAt
    have hw (k : Fin nK) := (EuclideanSpace.proj (𝕜 := ℝ) k).hasFDerivAt.comp
      (WithLp.toLp 2 (Q, WithLp.toLp 2 (K, V)))
      ((HardMaskedSoftmax.hasFDerivAt_weights ((mask.getD (allTrueMask nQ nK)).unstack i)
        (Op.fiberCLM (rowEquiv .scalar nQ nK) (PUnit.unit, i)
          (scaledMatmul (Shape.CanBroadcastTo.refl .scalar) (Shape.CanBroadcastTo.refl .scalar)
            Q (Tensor.permute K (Shape.adjacentSwap [nK, d] 0) [d, nK])
            (1 / scaleDenom (α := ℝ) d)))).comp
        (WithLp.toLp 2 (Q, WithLp.toLp 2 (K, V)))
        ((Op.fiberCLM (rowEquiv .scalar nQ nK) (PUnit.unit, i)).hasFDerivAt.comp
          (WithLp.toLp 2 (Q, WithLp.toLp 2 (K, V))) hS))
    have hv (k : Fin nK) :=
      ((coordinateCLM ((k, j, PUnit.unit) : Shape.Coord [nK, d])).comp
        ((WithLp.sndL 2 ℝ (Tensor ℝ [nK, d]) (Tensor ℝ [nK, d])).comp
          (WithLp.sndL 2 ℝ (Tensor ℝ [nQ, d])
            (WithLp 2 (Tensor ℝ [nK, d] × Tensor ℝ [nK, d]))))).hasFDerivAt
        (x := WithLp.toLp 2 (Q, WithLp.toLp 2 (K, V)))
    have hsum := HasFDerivAt.fun_sum (u := Finset.univ) fun k _ => (hw k).mul (hv k)
    have hfun : (fun p : WithLp 2 (Tensor ℝ [nQ, d] ×
          WithLp 2 (Tensor ℝ [nK, d] × Tensor ℝ [nK, d])) =>
          scaledDotProduct (⟨p.fst, p.snd.fst, p.snd.snd, mask⟩ : Inputs ℝ nQ nK d hQ hK)
            (i, j, PUnit.unit)) =
        fun p => ∑ k, HardMaskedSoftmax.weights ((mask.getD (allTrueMask nQ nK)).unstack i)
          (Op.fiberCLM (rowEquiv .scalar nQ nK) (PUnit.unit, i)
            (scaledMatmul (Shape.CanBroadcastTo.refl .scalar) (Shape.CanBroadcastTo.refl .scalar)
              p.fst (Tensor.permute p.snd.fst (Shape.adjacentSwap [nK, d] 0) [d, nK])
              (1 / scaleDenom (α := ℝ) d))) k *
          p.snd.snd (k, j, PUnit.unit) :=
      funext fun p => scaledDotProduct_apply p.fst p.snd.fst p.snd.snd mask i j
    rw [hfun]
    refine hsum.congr_fderiv (ContinuousLinearMap.ext fun u => ?_)
    rw [ContinuousLinearMap.comp_apply, coordinateCLM_apply, scaledDotProductCLM_apply,
      scaledDotProductJVP_apply, _root_.sum_apply]
    refine Finset.sum_congr rfl fun k _ => ?_
    simp only [_root_.add_apply, _root_.smul_apply, ContinuousLinearMap.comp_apply, smul_eq_mul,
      coordinateCLM_apply, WithLp.fstL_apply, WithLp.sndL_apply, EuclideanSpace.coe_proj,
      Function.comp_apply, WithLp.toLp_fst, WithLp.toLp_snd, ContinuousLinearMap.id_apply,
      ContinuousLinearMap.flip_apply, scaledMatmulCLM_apply, Tensor.permuteCLM_apply]
    ring
  convert Tensor.hasFDerivAt_of_coordinates _ _ _ key using 1
  exact ContinuousLinearMap.ext fun u => Tensor.Rep.ext fun c => by
    rw [assemble_apply]
    rfl

/-- Over the reals, scaled dot-product attention has the stored backward
`scaledDotProductBackward` as the adjoint of its derivative, jointly in the query, the key and the
value, at every point and for every mask. -/
theorem hasPullbackAt_scaledDotProduct (mask : Option (Tensor Bool [nQ, nK]))
    (Q : Tensor ℝ [nQ, d]) (K V : Tensor ℝ [nK, d]) :
    HasPullbackAt (fun p : WithLp 2 (Tensor ℝ [nQ, d] ×
        WithLp 2 (Tensor ℝ [nK, d] × Tensor ℝ [nK, d])) =>
        scaledDotProduct (⟨p.fst, p.snd.fst, p.snd.snd, mask⟩ : Inputs ℝ nQ nK d hQ hK))
      (scaledDotProductCLM (⟨Q, K, V, mask⟩ : Inputs ℝ nQ nK d hQ hK))
      (fun δ => WithLp.toLp 2
        ((scaledDotProductBackward (⟨Q, K, V, mask⟩ : Inputs ℝ nQ nK d hQ hK) δ).1,
        WithLp.toLp 2
          ((scaledDotProductBackward (⟨Q, K, V, mask⟩ : Inputs ℝ nQ nK d hQ hK) δ).2.1,
          (scaledDotProductBackward (⟨Q, K, V, mask⟩ : Inputs ℝ nQ nK d hQ hK) δ).2.2)))
      (WithLp.toLp 2 (Q, WithLp.toLp 2 (K, V))) :=
  ⟨hasFDerivAt_scaledDotProduct mask Q K V, inner_scaledDotProductJVP _⟩

end ScaledDotProduct

end Attention

namespace Tensor

/-- Exchanging the two leading axes is transposed by exchanging them back. -/
theorem inner_permute_adjacentSwap_zero {m n : Nat} {t : Shape} (x : Tensor ℝ (m :: n :: t))
    (z : Tensor ℝ (n :: m :: t))
    (hout : n :: m :: t = Shape.permute (m :: n :: t) (Shape.adjacentSwap (m :: n :: t) 0))
    (hout' : m :: n :: t = Shape.permute (n :: m :: t) (Shape.adjacentSwap (n :: m :: t) 0)) :
    ⟪Tensor.permute x (Shape.adjacentSwap (m :: n :: t) 0) (n :: m :: t) hout, z⟫ =
      ⟪x, Tensor.permute z (Shape.adjacentSwap (n :: m :: t) 0) (m :: n :: t) hout'⟫ := by
  rw [Tensor.inner_eq_sum, Tensor.inner_eq_sum]
  simp only [permute_adjacentSwap_zero_apply]
  exact Fintype.sum_equiv
    ({ toFun := fun c => (c.2.1, c.1, c.2.2), invFun := fun c => (c.2.1, c.1, c.2.2),
       left_inv := fun _ => rfl, right_inv := fun _ => rfl } :
      Shape.Coord (n :: m :: t) ≃ Shape.Coord (m :: n :: t)) _ _ fun _ => rfl

/-- The slice `i` along the leading axis, as a continuous linear map. -/
def unstackCLM {k : Nat} {s : Shape} (i : Fin k) : Tensor ℝ (k :: s) →L[ℝ] Tensor ℝ s :=
  LinearMap.toContinuousLinearMap
    { toFun := fun x => x.unstack i
      map_add' := fun x y => Rep.ext fun c => by
        simp only [Tensor.unstack_apply, Tensor.add_apply]
      map_smul' := fun a x => Rep.ext fun c => by
        simp only [Tensor.unstack_apply, Rep.smul_apply, RingHom.id_apply] }

/-- `unstackCLM i` applied to `x` is the slice `x.unstack i`. -/
@[simp] theorem unstackCLM_apply {k : Nat} {s : Shape} (i : Fin k) (x : Tensor ℝ (k :: s)) :
    unstackCLM i x = x.unstack i :=
  rfl

/-- The slice `i` along the leading axis is transposed by the stack with `δ` at `i` and zero at
every other index. -/
theorem inner_unstack {k : Nat} {s : Shape} (i : Fin k) (x : Tensor ℝ (k :: s))
    (δ : Tensor ℝ s) :
    ⟪x.unstack i, δ⟫ = ⟪x, Tensor.dim fun j => if j = i then δ else 0⟫ := by
  rw [inner_eq_sum_unstack x, Finset.sum_eq_single i, unstack_dim, ite_eq_left rfl]
  · intro j _ hj
    rw [unstack_dim, ite_eq_right hj, inner_zero_right]
  · intro hi
    exact absurd (Finset.mem_univ i) hi

/-- Without batch axes, the batched matrix product is `Tensor.matmul`. -/
theorem batchMatmul_refl_scalar {m n p : Nat} (A : Tensor ℝ [m, n]) (B : Tensor ℝ [n, p]) :
    batchMatmul (Shape.CanBroadcastTo.refl .scalar) (Shape.CanBroadcastTo.refl .scalar) A B =
      matmul A B := by
  simp only [batchMatmul, broadcast_self, matmulCommonBatch_scalar]

/-- The stack of `m` tensors of shape `s` along a leading axis, as a continuous linear map. -/
def stackCLM (m : Nat) (s : Shape) : (Fin m → Tensor ℝ s) →L[ℝ] Tensor ℝ (m :: s) :=
  LinearMap.toContinuousLinearMap
    { toFun := fun G => Tensor.dim G
      map_add' := fun G H => Rep.ext fun c => by
        simp only [Tensor.dim_apply, Pi.add_apply, Tensor.add_apply]
      map_smul' := fun a G => Rep.ext fun c => by
        simp only [Tensor.dim_apply, Pi.smul_apply, Rep.smul_apply,
          RingHom.id_apply] }

/-- `stackCLM m s` applied to `G` is the stack `Tensor.dim G`. -/
@[simp] theorem stackCLM_apply (m : Nat) (s : Shape) (G : Fin m → Tensor ℝ s) :
    stackCLM m s G = Tensor.dim G :=
  rfl

/-- Adding over `i` the stacks with `G i` at the index `i` and zero at every other index gives the
stack of `G`. -/
theorem sum_dim_ite {m : Nat} {s : Shape} (G : Fin m → Tensor ℝ s) :
    (∑ i, Tensor.dim fun j => if j = i then G i else 0) = Tensor.dim G := by
  rw [← dim_unstack (∑ i, Tensor.dim fun j => if j = i then G i else 0)]
  congr 1
  funext j
  rw [← unstackCLM_apply, _root_.map_sum]
  simp only [unstackCLM_apply, unstack_dim, Fintype.sum_ite_eq]

end Tensor

namespace MultiHeadAttention

variable {n numHeads headDim : Nat}

/-- Splitting the features into heads is transposed by combining the heads. -/
theorem inner_splitHeads (x : Tensor ℝ [n, numHeads * headDim])
    (H : Tensor ℝ [numHeads, n, headDim]) :
    ⟪splitHeads x numHeads headDim rfl, H⟫ = ⟪x, combineHeads H⟫ :=
  (Tensor.inner_permute_adjacentSwap_zero _ H _ (by tensor_shape)).trans
    (Tensor.inner_reshape _ _ _)

/-- Splitting the features of `n` rows into `numHeads` heads of width `headDim`, as a continuous
linear map. -/
def splitHeadsCLM (n numHeads headDim : Nat) :
    Tensor ℝ [n, numHeads * headDim] →L[ℝ] Tensor ℝ [numHeads, n, headDim] :=
  clmOfTranspose (fun x => splitHeads x numHeads headDim rfl) (fun H => combineHeads H)
    inner_splitHeads

/-- `splitHeadsCLM n numHeads headDim` applied to `x` is `splitHeads x numHeads headDim rfl`. -/
@[simp] theorem splitHeadsCLM_apply (x : Tensor ℝ [n, numHeads * headDim]) :
    splitHeadsCLM n numHeads headDim x = splitHeads x numHeads headDim rfl :=
  rfl

/-- Combining the heads is transposed by splitting the features into heads. -/
theorem inner_combineHeads (H : Tensor ℝ [numHeads, n, headDim])
    (x : Tensor ℝ [n, numHeads * headDim]) :
    ⟪combineHeads H, x⟫ = ⟪H, splitHeads x numHeads headDim rfl⟫ := by
  rw [real_inner_comm, ← inner_splitHeads, real_inner_comm]

/-- Combining `numHeads` heads of width `headDim` over `n` rows, as a continuous linear map. -/
def combineHeadsCLM (n numHeads headDim : Nat) :
    Tensor ℝ [numHeads, n, headDim] →L[ℝ] Tensor ℝ [n, numHeads * headDim] :=
  clmOfTranspose (fun H => combineHeads H) (fun x => splitHeads x numHeads headDim rfl)
    inner_combineHeads

/-- `combineHeadsCLM n numHeads headDim` applied to `H` is `combineHeads H`. -/
@[simp] theorem combineHeadsCLM_apply (H : Tensor ℝ [numHeads, n, headDim]) :
    combineHeadsCLM n numHeads headDim H = combineHeads H :=
  rfl

end MultiHeadAttention

namespace HasPullbackAt

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] {x : E}

/-- Splitting the features of a map into heads: the derivative is `splitHeadsCLM` after `Df`, and
the pullback combines the heads first. -/
@[fun_prop]
theorem splitHeads {n numHeads headDim : Nat} {f : E → Tensor ℝ [n, numHeads * headDim]}
    {Df : E →L[ℝ] Tensor ℝ [n, numHeads * headDim]} {bf : Tensor ℝ [n, numHeads * headDim] → E}
    (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => MultiHeadAttention.splitHeads (f y) numHeads headDim rfl)
      ((MultiHeadAttention.splitHeadsCLM n numHeads headDim).comp Df)
      (fun δ => bf (MultiHeadAttention.combineHeads δ)) x := by
  simpa only [MultiHeadAttention.splitHeadsCLM_apply] using
    clm (MultiHeadAttention.splitHeadsCLM n numHeads headDim)
      (fun H => MultiHeadAttention.combineHeads H) MultiHeadAttention.inner_splitHeads hf

/-- Combining the heads of a map: the derivative is `combineHeadsCLM` after `Df`, and the pullback
splits the features into heads first. -/
@[fun_prop]
theorem combineHeads {n numHeads headDim : Nat} {f : E → Tensor ℝ [numHeads, n, headDim]}
    {Df : E →L[ℝ] Tensor ℝ [numHeads, n, headDim]} {bf : Tensor ℝ [numHeads, n, headDim] → E}
    (hf : HasPullbackAt f Df bf x) : HasPullbackAt (fun y => MultiHeadAttention.combineHeads (f y))
      ((MultiHeadAttention.combineHeadsCLM n numHeads headDim).comp Df)
      (fun δ => bf (MultiHeadAttention.splitHeads δ numHeads headDim rfl)) x := by
  simpa only [MultiHeadAttention.combineHeadsCLM_apply] using
    clm (MultiHeadAttention.combineHeadsCLM n numHeads headDim)
      (fun x => MultiHeadAttention.splitHeads x numHeads headDim rfl)
      MultiHeadAttention.inner_combineHeads hf

/-- The matrix product of two maps, by the product rule, pulled back by the two components of
`matmulBackward` at the operands `f x` and `g x`. -/
@[fun_prop]
theorem matmul {m k p : Nat} {f : E → Tensor ℝ [m, k]} {g : E → Tensor ℝ [k, p]}
    {Df : E →L[ℝ] Tensor ℝ [m, k]} {Dg : E →L[ℝ] Tensor ℝ [k, p]} {bf : Tensor ℝ [m, k] → E}
    {bg : Tensor ℝ [k, p] → E} (hf : HasPullbackAt f Df bf x) (hg : HasPullbackAt g Dg bg x) :
    HasPullbackAt (fun y => Tensor.matmul (f y) (g y))
      (((Tensor.batchMatmulCLM (Shape.CanBroadcastTo.refl .scalar)
          (Shape.CanBroadcastTo.refl .scalar)).flip (g x)).comp Df +
        (Tensor.batchMatmulCLM (Shape.CanBroadcastTo.refl .scalar)
          (Shape.CanBroadcastTo.refl .scalar) (f x)).comp Dg)
      (fun δ => bf (Tensor.matmulBackward (Shape.CanBroadcastTo.refl .scalar)
          (Shape.CanBroadcastTo.refl .scalar) (f x) (g x) δ).1 +
        bg (Tensor.matmulBackward (Shape.CanBroadcastTo.refl .scalar)
          (Shape.CanBroadcastTo.refl .scalar) (f x) (g x) δ).2) x := by
  have h :=
    batchMatmul (Shape.CanBroadcastTo.refl .scalar) (Shape.CanBroadcastTo.refl .scalar) hf hg
  simp only [Tensor.batchMatmul_refl_scalar] at h
  exact h

/-- The slice `i` along the leading axis of a map: the derivative is `unstackCLM i` after `Df`,
and the pullback places `δ` at the index `i`. -/
@[fun_prop]
theorem unstack {k : Nat} {s : Shape} {f : E → Tensor ℝ (k :: s)}
    {Df : E →L[ℝ] Tensor ℝ (k :: s)} {bf : Tensor ℝ (k :: s) → E} (i : Fin k)
    (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => (f y).unstack i) ((Tensor.unstackCLM i).comp Df)
      (fun δ => bf (Tensor.dim fun j => if j = i then δ else 0)) x := by
  simpa only [Tensor.unstackCLM_apply] using
    clm (Tensor.unstackCLM i) (fun δ => Tensor.dim fun j => if j = i then δ else 0)
      (Tensor.inner_unstack i) hf

/-- The stack of `m` maps along a leading axis: the derivative stacks their derivatives, and the
pullback adds their pullbacks at the slices of `δ`. -/
theorem stack {m : Nat} {s : Shape} {g : Fin m → E → Tensor ℝ s}
    {D : Fin m → E →L[ℝ] Tensor ℝ s} {b : Fin m → Tensor ℝ s → E}
    (h : ∀ i, HasPullbackAt (g i) (D i) (b i) x) :
    HasPullbackAt (fun y => Tensor.dim fun i => g i y)
      ((Tensor.stackCLM m s).comp (ContinuousLinearMap.pi D))
      (fun δ => ∑ i, b i (δ.unstack i)) x where
  hasFDerivAt := (Tensor.stackCLM m s).hasFDerivAt.comp x
    (hasFDerivAt_pi.2 fun i => (h i).hasFDerivAt)
  inner_eq u δ := by
    show ⟪(Tensor.stackCLM m s).comp (ContinuousLinearMap.pi D) u, δ⟫ =
      ⟪u, ∑ i, b i (δ.unstack i)⟫
    rw [ContinuousLinearMap.comp_apply, Tensor.stackCLM_apply, Tensor.inner_eq_sum_unstack,
      inner_sum]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [Tensor.unstack_dim, ContinuousLinearMap.pi_apply]
    exact (h i).inner_eq u (δ.unstack i)

/-- A map `F` of a shared argument and one slice, applied to every slice along a leading axis:
when `F` has the pullback `b i` at the slice `i`, the stack has the pullback that adds the shared
components of the `b i` and stacks their slice components. -/
theorem stackSlices {Θ : Type*} [NormedAddCommGroup Θ] [InnerProductSpace ℝ Θ] {m : Nat}
    {s t : Shape} {F : WithLp 2 (Θ × Tensor ℝ s) → Tensor ℝ t} {θ : Θ}
    {x : Tensor ℝ (m :: s)} {D : Fin m → WithLp 2 (Θ × Tensor ℝ s) →L[ℝ] Tensor ℝ t}
    {b : Fin m → Tensor ℝ t → WithLp 2 (Θ × Tensor ℝ s)}
    (h : ∀ i, HasPullbackAt F (D i) (b i) (WithLp.toLp 2 (θ, x.unstack i))) :
    HasPullbackAt (fun p : WithLp 2 (Θ × Tensor ℝ (m :: s)) =>
        Tensor.dim fun i => F (WithLp.toLp 2 (p.fst, p.snd.unstack i)))
      (fderiv ℝ (fun p : WithLp 2 (Θ × Tensor ℝ (m :: s)) =>
        Tensor.dim fun i => F (WithLp.toLp 2 (p.fst, p.snd.unstack i))) (WithLp.toLp 2 (θ, x)))
      (fun δ => WithLp.toLp 2 (∑ i, (b i (δ.unstack i)).fst,
        Tensor.dim fun i => (b i (δ.unstack i)).snd)) (WithLp.toLp 2 (θ, x)) := by
  have hid := HasPullbackAt.id (WithLp.toLp 2 (θ, x))
  have hs := HasPullbackAt.stack fun i => HasPullbackAt.comp
    (f := fun p : WithLp 2 (Θ × Tensor ℝ (m :: s)) => WithLp.toLp 2 (p.fst, p.snd.unstack i))
    (x := WithLp.toLp 2 (θ, x)) (h i) (hid.fst.toLp_prod (hid.snd.unstack i))
  refine hs.congr_deriv hs.hasFDerivAt.fderiv.symm fun δ => ?_
  simp only [← WithLp.toLp_add, Prod.mk_add_mk, add_zero, zero_add, ← WithLp.toLp_sum]
  congr 1
  refine Prod.ext ?_ ?_
  · simp only [Prod.fst_sum]
  · simp only [Prod.snd_sum, Tensor.sum_dim_ite]

end HasPullbackAt

namespace Attention

section Heads

variable {numHeads nQ nK d : Nat}

/-- The pullback of scaled dot-product attention on every head: the slice `h` of each component
is the corresponding component of `scaledDotProductBackward` of head `h` at the slice `h` of
`δ`. -/
def headsBackward (hQ : nQ ≠ 0) (hK : nK ≠ 0) (mask : Option (Tensor Bool [nQ, nK]))
    (QH : Tensor ℝ [numHeads, nQ, d]) (KH VH : Tensor ℝ [numHeads, nK, d])
    (δ : Tensor ℝ [numHeads, nQ, d]) : WithLp 2 (Tensor ℝ [numHeads, nQ, d] ×
      WithLp 2 (Tensor ℝ [numHeads, nK, d] × Tensor ℝ [numHeads, nK, d])) :=
  WithLp.toLp 2
    (Tensor.dim fun h => (scaledDotProductBackward
        (⟨QH.unstack h, KH.unstack h, VH.unstack h, mask⟩ : Inputs ℝ nQ nK d hQ hK)
        (δ.unstack h)).1,
      WithLp.toLp 2
        (Tensor.dim fun h => (scaledDotProductBackward
            (⟨QH.unstack h, KH.unstack h, VH.unstack h, mask⟩ : Inputs ℝ nQ nK d hQ hK)
            (δ.unstack h)).2.1,
          Tensor.dim fun h => (scaledDotProductBackward
            (⟨QH.unstack h, KH.unstack h, VH.unstack h, mask⟩ : Inputs ℝ nQ nK d hQ hK)
            (δ.unstack h)).2.2))

/-- The tangent of scaled dot-product attention on every head is transposed by
`headsBackward`. -/
theorem inner_headsJVP (hQ : nQ ≠ 0) (hK : nK ≠ 0) (mask : Option (Tensor Bool [nQ, nK]))
    (QH : Tensor ℝ [numHeads, nQ, d]) (KH VH : Tensor ℝ [numHeads, nK, d])
    (u : WithLp 2 (Tensor ℝ [numHeads, nQ, d] ×
      WithLp 2 (Tensor ℝ [numHeads, nK, d] × Tensor ℝ [numHeads, nK, d])))
    (δ : Tensor ℝ [numHeads, nQ, d]) :
    ⟪Tensor.dim fun h => scaledDotProductJVP
        (⟨QH.unstack h, KH.unstack h, VH.unstack h, mask⟩ : Inputs ℝ nQ nK d hQ hK)
        (u.fst.unstack h) (u.snd.fst.unstack h) (u.snd.snd.unstack h), δ⟫ =
      ⟪u, headsBackward hQ hK mask QH KH VH δ⟫ := by
  rw [headsBackward, WithLp.prod_inner_apply, WithLp.prod_inner_apply]
  dsimp only
  rw [Tensor.inner_eq_sum_unstack _ δ, Tensor.inner_eq_sum_unstack u.ofLp.1,
    Tensor.inner_eq_sum_unstack u.ofLp.2.ofLp.1, Tensor.inner_eq_sum_unstack u.ofLp.2.ofLp.2,
    ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun h _ => ?_
  have e := inner_scaledDotProductJVP
    (⟨QH.unstack h, KH.unstack h, VH.unstack h, mask⟩ : Inputs ℝ nQ nK d hQ hK)
    (WithLp.toLp 2 (u.fst.unstack h, WithLp.toLp 2 (u.snd.fst.unstack h, u.snd.snd.unstack h)))
    (δ.unstack h)
  rw [WithLp.prod_inner_apply, WithLp.prod_inner_apply] at e
  simp only [Tensor.unstack_dim]
  exact e

/-- The derivative of scaled dot-product attention on every head, jointly in the query, key and
value heads: head `h` applies `scaledDotProductJVP` of head `h` to the slices `h`. -/
def headsCLM (hQ : nQ ≠ 0) (hK : nK ≠ 0) (mask : Option (Tensor Bool [nQ, nK]))
    (QH : Tensor ℝ [numHeads, nQ, d]) (KH VH : Tensor ℝ [numHeads, nK, d]) :
    WithLp 2 (Tensor ℝ [numHeads, nQ, d] ×
        WithLp 2 (Tensor ℝ [numHeads, nK, d] × Tensor ℝ [numHeads, nK, d])) →L[ℝ]
      Tensor ℝ [numHeads, nQ, d] :=
  clmOfTranspose (fun u => Tensor.dim fun h => scaledDotProductJVP
      (⟨QH.unstack h, KH.unstack h, VH.unstack h, mask⟩ : Inputs ℝ nQ nK d hQ hK)
      (u.fst.unstack h) (u.snd.fst.unstack h) (u.snd.snd.unstack h))
    (headsBackward hQ hK mask QH KH VH) (inner_headsJVP hQ hK mask QH KH VH)

/-- `headsCLM hQ hK mask QH KH VH` applied to `u` stacks `scaledDotProductJVP` of every head
along the slices of `u`. -/
@[simp] theorem headsCLM_apply (hQ : nQ ≠ 0) (hK : nK ≠ 0) (mask : Option (Tensor Bool [nQ, nK]))
    (QH : Tensor ℝ [numHeads, nQ, d]) (KH VH : Tensor ℝ [numHeads, nK, d])
    (u : WithLp 2 (Tensor ℝ [numHeads, nQ, d] ×
      WithLp 2 (Tensor ℝ [numHeads, nK, d] × Tensor ℝ [numHeads, nK, d]))) :
    headsCLM hQ hK mask QH KH VH u = Tensor.dim fun h => scaledDotProductJVP
      (⟨QH.unstack h, KH.unstack h, VH.unstack h, mask⟩ : Inputs ℝ nQ nK d hQ hK)
      (u.fst.unstack h) (u.snd.fst.unstack h) (u.snd.snd.unstack h) :=
  rfl

/-- Over the reals, scaled dot-product attention on every head has derivative `headsCLM` jointly
in the query, key and value heads, at every point and for every mask. -/
theorem hasFDerivAt_heads (hQ : nQ ≠ 0) (hK : nK ≠ 0) (mask : Option (Tensor Bool [nQ, nK]))
    (QH : Tensor ℝ [numHeads, nQ, d]) (KH VH : Tensor ℝ [numHeads, nK, d]) :
    HasFDerivAt (fun p : WithLp 2 (Tensor ℝ [numHeads, nQ, d] ×
        WithLp 2 (Tensor ℝ [numHeads, nK, d] × Tensor ℝ [numHeads, nK, d])) =>
        Tensor.dim fun h => scaledDotProduct
          (⟨p.fst.unstack h, p.snd.fst.unstack h, p.snd.snd.unstack h, mask⟩ :
            Inputs ℝ nQ nK d hQ hK))
      (headsCLM hQ hK mask QH KH VH) (WithLp.toLp 2 (QH, WithLp.toLp 2 (KH, VH))) := by
  have key (c : Shape.Coord [numHeads, nQ, d]) :
      HasFDerivAt (fun p : WithLp 2 (Tensor ℝ [numHeads, nQ, d] ×
          WithLp 2 (Tensor ℝ [numHeads, nK, d] × Tensor ℝ [numHeads, nK, d])) =>
          (Tensor.dim fun h => scaledDotProduct
            (⟨p.fst.unstack h, p.snd.fst.unstack h, p.snd.snd.unstack h, mask⟩ :
              Inputs ℝ nQ nK d hQ hK)) c)
        ((coordinateCLM c).comp (headsCLM hQ hK mask QH KH VH))
        (WithLp.toLp 2 (QH, WithLp.toLp 2 (KH, VH))) := by
    obtain ⟨h, c⟩ := c
    have hid := HasPullbackAt.id (WithLp.toLp 2 (QH, WithLp.toLp 2 (KH, VH)))
    have hS := (coordinateCLM c).hasFDerivAt.comp (WithLp.toLp 2 (QH, WithLp.toLp 2 (KH, VH)))
      (HasPullbackAt.comp (f := fun y : WithLp 2 (Tensor ℝ [numHeads, nQ, d] ×
          WithLp 2 (Tensor ℝ [numHeads, nK, d] × Tensor ℝ [numHeads, nK, d])) =>
          WithLp.toLp 2 (y.fst.unstack h, WithLp.toLp 2 (y.snd.fst.unstack h, y.snd.snd.unstack h)))
        (hasPullbackAt_scaledDotProduct (hQ := hQ) (hK := hK) mask (QH.unstack h)
          (KH.unstack h) (VH.unstack h))
        ((hid.fst.unstack h).toLp_prod
          ((hid.snd.fst.unstack h).toLp_prod (hid.snd.snd.unstack h)))).hasFDerivAt
    have hfun : (fun p : WithLp 2 (Tensor ℝ [numHeads, nQ, d] ×
          WithLp 2 (Tensor ℝ [numHeads, nK, d] × Tensor ℝ [numHeads, nK, d])) =>
          (Tensor.dim fun h => scaledDotProduct
            (⟨p.fst.unstack h, p.snd.fst.unstack h, p.snd.snd.unstack h, mask⟩ :
              Inputs ℝ nQ nK d hQ hK)) (h, c)) =
        ⇑(coordinateCLM c) ∘ fun p => scaledDotProduct
          (⟨p.fst.unstack h, p.snd.fst.unstack h, p.snd.snd.unstack h, mask⟩ :
            Inputs ℝ nQ nK d hQ hK) := by
      funext p
      rw [Tensor.dim_apply]
      rfl
    rw [hfun]
    refine hS.congr_fderiv (ContinuousLinearMap.ext fun u => ?_)
    simp only [ContinuousLinearMap.comp_apply, coordinateCLM_apply, headsCLM_apply]
    exact (Tensor.dim_apply (fun h => scaledDotProductJVP
      (⟨QH.unstack h, KH.unstack h, VH.unstack h, mask⟩ : Inputs ℝ nQ nK d hQ hK)
      (u.fst.unstack h) (u.snd.fst.unstack h) (u.snd.snd.unstack h)) (h, c)).symm
  convert Tensor.hasFDerivAt_of_coordinates _ _ _ key using 1
  exact ContinuousLinearMap.ext fun u => Tensor.Rep.ext fun c => by
    rw [assemble_apply]
    rfl

/-- Over the reals, scaled dot-product attention on every head has the stored per-head backward
`headsBackward` as the adjoint of its derivative, jointly in the query, key and value heads. -/
theorem hasPullbackAt_heads (hQ : nQ ≠ 0) (hK : nK ≠ 0) (mask : Option (Tensor Bool [nQ, nK]))
    (QH : Tensor ℝ [numHeads, nQ, d]) (KH VH : Tensor ℝ [numHeads, nK, d]) :
    HasPullbackAt (fun p : WithLp 2 (Tensor ℝ [numHeads, nQ, d] ×
        WithLp 2 (Tensor ℝ [numHeads, nK, d] × Tensor ℝ [numHeads, nK, d])) =>
        Tensor.dim fun h => scaledDotProduct
          (⟨p.fst.unstack h, p.snd.fst.unstack h, p.snd.snd.unstack h, mask⟩ :
            Inputs ℝ nQ nK d hQ hK))
      (headsCLM hQ hK mask QH KH VH) (headsBackward hQ hK mask QH KH VH)
      (WithLp.toLp 2 (QH, WithLp.toLp 2 (KH, VH))) :=
  ⟨hasFDerivAt_heads hQ hK mask QH KH VH, inner_headsJVP hQ hK mask QH KH VH⟩

end Heads

end Attention

namespace HasPullbackAt

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] {x : E}

/-- Scaled dot-product attention on every head of three maps: the derivative is `headsCLM` after
the derivatives of the query, key and value heads, and the pullback applies `headsBackward`. -/
@[fun_prop]
theorem attentionHeads {numHeads nQ nK d : Nat} (hQ : nQ ≠ 0) (hK : nK ≠ 0)
    (mask : Option (Tensor Bool [nQ, nK])) {f : E → Tensor ℝ [numHeads, nQ, d]}
    {g k : E → Tensor ℝ [numHeads, nK, d]} {Df : E →L[ℝ] Tensor ℝ [numHeads, nQ, d]}
    {Dg Dk : E →L[ℝ] Tensor ℝ [numHeads, nK, d]} {bf : Tensor ℝ [numHeads, nQ, d] → E}
    {bg bk : Tensor ℝ [numHeads, nK, d] → E} (hf : HasPullbackAt f Df bf x)
    (hg : HasPullbackAt g Dg bg x) (hk : HasPullbackAt k Dk bk x) :
    HasPullbackAt (fun y => Tensor.dim fun h => Attention.scaledDotProduct
        (⟨(f y).unstack h, (g y).unstack h, (k y).unstack h, mask⟩ :
          Attention.Inputs ℝ nQ nK d hQ hK))
      ((Attention.headsCLM hQ hK mask (f x) (g x) (k x)).comp
        ((WithLp.prodContinuousLinearEquiv 2 ℝ _ _).symm.toContinuousLinearMap.comp
          (Df.prod ((WithLp.prodContinuousLinearEquiv 2 ℝ _ _).symm.toContinuousLinearMap.comp
            (Dg.prod Dk)))))
      (fun δ => bf (Attention.headsBackward hQ hK mask (f x) (g x) (k x) δ).fst +
        (bg (Attention.headsBackward hQ hK mask (f x) (g x) (k x) δ).snd.fst +
          bk (Attention.headsBackward hQ hK mask (f x) (g x) (k x) δ).snd.snd)) x :=
  HasPullbackAt.comp (f := fun y => WithLp.toLp 2 (f y, WithLp.toLp 2 (g y, k y))) (x := x)
    (Attention.hasPullbackAt_heads hQ hK mask (f x) (g x) (k x)) (hf.toLp_prod (hg.toLp_prod hk))

end HasPullbackAt

namespace MultiHeadAttention

variable {n numHeads dModel headDim : Nat}

/-- The four projections of multi-head attention as one point of a Euclidean space, in the order
query, key, value, output. -/
abbrev Params (numHeads dModel headDim : Nat) : Type :=
  WithLp 2 (Tensor ℝ [dModel, numHeads * headDim] ×
    WithLp 2 (Tensor ℝ [dModel, numHeads * headDim] ×
      WithLp 2 (Tensor ℝ [dModel, numHeads * headDim] × Tensor ℝ [numHeads * headDim, dModel])))

/-- The projections of `mha` as a point of `Params`. -/
def toParams (mha : MultiHeadAttention ℝ numHeads dModel headDim) :
    Params numHeads dModel headDim :=
  WithLp.toLp 2 (mha.queryWeight, WithLp.toLp 2 (mha.keyWeight,
    WithLp.toLp 2 (mha.valueWeight, mha.outputWeight)))

/-- The point of the fieldwise sum of two records is the sum of their points. -/
theorem toParams_add (a b : MultiHeadAttention ℝ numHeads dModel headDim) :
    toParams (a.add b) = toParams a + toParams b :=
  rfl

/-- The point of `MultiHeadAttention.sum` of a list is the sum of the points of its entries. -/
theorem toParams_sum (mha : MultiHeadAttention ℝ numHeads dModel headDim)
    (L : List (MultiHeadAttention ℝ numHeads dModel headDim)) :
    toParams (mha.sum L) = (L.map toParams).sum := by
  cases L with
  | nil =>
    rw [List.map_nil, List.sum_nil]
    repeat' (first
      | refine withLp_prod_ext ?_ ?_
      | refine Tensor.Rep.ext fun _ => ?_)
    all_goals try simp only [toParams, MultiHeadAttention.sum, WithLp.toLp_fst, WithLp.toLp_snd,
      WithLp.zero_fst, WithLp.zero_snd, Tensor.map_apply, Tensor.Rep.zero_apply]
  | cons first rest =>
    rw [MultiHeadAttention.sum, List.map_cons, List.sum_cons]
    induction rest generalizing first with
    | nil => rw [List.foldl_nil, List.map_nil, List.sum_nil, add_zero]
    | cons a rest ih =>
      rw [List.foldl_cons, ih, toParams_add, List.map_cons, List.sum_cons, add_assoc]

/-- The sum of the points of the entries of `L.flatMap g` adds over `L` the sums of the points of
the entries of `g a`. -/
theorem sum_map_toParams_flatMap {β : Type*} (L : List β)
    (g : β → List (MultiHeadAttention ℝ numHeads dModel headDim)) :
    ((L.flatMap g).map toParams).sum = (L.map fun a => ((g a).map toParams).sum).sum := by
  induction L with
  | nil => rfl
  | cons a L ih =>
    rw [List.flatMap_cons, List.map_append, List.sum_append, ih, List.map_cons, List.sum_cons]

/-- Over the reals, self-attention of one sequence is differentiable jointly in the projections
and the sequence, and the stored backward is its pullback. -/
theorem hasPullbackAt_forward (h1 : n ≠ 0) (mask : Option (Tensor Bool [n, n]))
    (θ : Params numHeads dModel headDim) (x : Tensor ℝ [n, dModel]) :
    HasPullbackAt (fun p : WithLp 2 (Params numHeads dModel headDim × Tensor ℝ [n, dModel]) =>
        forward h1 h1 ⟨p.fst.fst, p.fst.snd.fst, p.fst.snd.snd.fst, p.fst.snd.snd.snd⟩ p.snd
          p.snd mask)
      (fderiv ℝ (fun p : WithLp 2 (Params numHeads dModel headDim × Tensor ℝ [n, dModel]) =>
        forward h1 h1 ⟨p.fst.fst, p.fst.snd.fst, p.fst.snd.snd.fst, p.fst.snd.snd.snd⟩ p.snd
          p.snd mask) (WithLp.toLp 2 (θ, x)))
      (fun δ => WithLp.toLp 2
        (toParams (backward h1 ⟨θ.fst, θ.snd.fst, θ.snd.snd.fst, θ.snd.snd.snd⟩ x mask
          δ).parameters,
          (backward h1 ⟨θ.fst, θ.snd.fst, θ.snd.snd.fst, θ.snd.snd.snd⟩ x mask δ).input))
      (WithLp.toLp 2 (θ, x)) := by
  have hid := HasPullbackAt.id (WithLp.toLp 2 (θ, x))
  have hx := hid.snd
  have hq := (hx.matmul hid.fst.fst).splitHeads (numHeads := numHeads) (headDim := headDim)
  have hk := (hx.matmul hid.fst.snd.fst).splitHeads (numHeads := numHeads) (headDim := headDim)
  have hv := (hx.matmul hid.fst.snd.snd.fst).splitHeads (numHeads := numHeads)
    (headDim := headDim)
  have hout := (HasPullbackAt.attentionHeads h1 h1 mask hq hk hv).combineHeads.matmul
    hid.fst.snd.snd.snd
  refine hout.congr_deriv hout.hasFDerivAt.fderiv.symm fun δ => ?_
  simp only [toParams, ← WithLp.toLp_add, Prod.mk_add_mk, add_zero, zero_add]
  rw [← add_assoc]
  rfl

/-- Over the reals, self-attention of every sequence along the leading axes `batch` is
differentiable jointly in the shared projections and the input, and its pullback is
`backwardLeading` with the parameter gradients of the sequences added. -/
theorem hasPullbackAt_mapLeading (h1 : n ≠ 0) (mask : Option (Tensor Bool [n, n]))
    (batch : Shape) (θ : Params numHeads dModel headDim) (x : Tensor ℝ (batch ++ [n, dModel])) :
    HasPullbackAt
      (fun p : WithLp 2 (Params numHeads dModel headDim × Tensor ℝ (batch ++ [n, dModel])) =>
        Tensor.mapLeading batch (fun z => forward h1 h1
          ⟨p.fst.fst, p.fst.snd.fst, p.fst.snd.snd.fst, p.fst.snd.snd.snd⟩ z z mask) p.snd)
      (fderiv ℝ
        (fun p : WithLp 2 (Params numHeads dModel headDim × Tensor ℝ (batch ++ [n, dModel])) =>
          Tensor.mapLeading batch (fun z => forward h1 h1
            ⟨p.fst.fst, p.fst.snd.fst, p.fst.snd.snd.fst, p.fst.snd.snd.snd⟩ z z mask) p.snd)
        (WithLp.toLp 2 (θ, x)))
      (fun δ => WithLp.toLp 2
        (((backwardLeading h1 ⟨θ.fst, θ.snd.fst, θ.snd.snd.fst, θ.snd.snd.snd⟩ mask batch x
          δ).2.map toParams).sum,
          (backwardLeading h1 ⟨θ.fst, θ.snd.fst, θ.snd.snd.fst, θ.snd.snd.snd⟩ mask batch x
            δ).1))
      (WithLp.toLp 2 (θ, x)) := by
  induction batch with
  | scalar =>
    refine (hasPullbackAt_forward h1 mask θ x).congr_deriv rfl fun δ => ?_
    rw [backwardLeading_scalar]
    simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero]
  | dim m rest ih =>
    refine (HasPullbackAt.stackSlices (F := fun p : WithLp 2 (Params numHeads dModel headDim ×
        Tensor ℝ (rest ++ [n, dModel])) => Tensor.mapLeading rest (fun z => forward h1 h1
          ⟨p.fst.fst, p.fst.snd.fst, p.fst.snd.snd.fst, p.fst.snd.snd.snd⟩ z z mask) p.snd)
        fun i => ih (x.unstack i)).congr_deriv rfl fun δ => ?_
    rw [backwardLeading_dim]
    simp only [WithLp.toLp_fst, WithLp.toLp_snd, sum_map_toParams_flatMap, List.map_ofFn,
      List.sum_ofFn, Function.comp_apply]

end MultiHeadAttention

namespace HasPullbackAt

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] {x : E}

/-- Multi-head self-attention of every sequence along the leading axes `batch`, with the four
projections and the input given by maps: the pullback applies `backwardLeading` and adds the
parameter gradients of the sequences with `MultiHeadAttention.sum`. -/
@[fun_prop]
theorem multiheadAttention {batch : Shape} {n numHeads dModel headDim : Nat}
    [inst : DecidableRel ((· > ·) : ℝ → ℝ → Prop)] (h1 : n ≠ 0)
    (mask : Option (Tensor Bool [n, n])) {fq fk fv : E → Tensor ℝ [dModel, numHeads * headDim]}
    {fo : E → Tensor ℝ [numHeads * headDim, dModel]} {fx : E → Tensor ℝ (batch ++ [n, dModel])}
    {Dq Dk Dv : E →L[ℝ] Tensor ℝ [dModel, numHeads * headDim]}
    {Do : E →L[ℝ] Tensor ℝ [numHeads * headDim, dModel]}
    {Dx : E →L[ℝ] Tensor ℝ (batch ++ [n, dModel])}
    {bq bk bv : Tensor ℝ [dModel, numHeads * headDim] → E}
    {bo : Tensor ℝ [numHeads * headDim, dModel] → E} {bx : Tensor ℝ (batch ++ [n, dModel]) → E}
    (hq : HasPullbackAt fq Dq bq x) (hk : HasPullbackAt fk Dk bk x)
    (hv : HasPullbackAt fv Dv bv x) (ho : HasPullbackAt fo Do bo x)
    (hx : HasPullbackAt fx Dx bx x) :
    HasPullbackAt (fun y => Tensor.mapLeading batch (fun z => MultiHeadAttention.forward h1 h1
        ⟨fq y, fk y, fv y, fo y⟩ z z mask) (fx y))
      (fderiv ℝ (fun y => Tensor.mapLeading batch (fun z => MultiHeadAttention.forward h1 h1
        ⟨fq y, fk y, fv y, fo y⟩ z z mask) (fx y)) x)
      (fun δ =>
        bq (MultiHeadAttention.sum ⟨fq x, fk x, fv x, fo x⟩ (MultiHeadAttention.backwardLeading h1
          ⟨fq x, fk x, fv x, fo x⟩ mask batch (fx x) δ).2).queryWeight +
        (bk (MultiHeadAttention.sum ⟨fq x, fk x, fv x, fo x⟩ (MultiHeadAttention.backwardLeading
          h1 ⟨fq x, fk x, fv x, fo x⟩ mask batch (fx x) δ).2).keyWeight +
        (bv (MultiHeadAttention.sum ⟨fq x, fk x, fv x, fo x⟩ (MultiHeadAttention.backwardLeading
          h1 ⟨fq x, fk x, fv x, fo x⟩ mask batch (fx x) δ).2).valueWeight +
        bo (MultiHeadAttention.sum ⟨fq x, fk x, fv x, fo x⟩ (MultiHeadAttention.backwardLeading
          h1 ⟨fq x, fk x, fv x, fo x⟩ mask batch (fx x) δ).2).outputWeight)) +
        bx (MultiHeadAttention.backwardLeading h1 ⟨fq x, fk x, fv x, fo x⟩ mask batch (fx x)
          δ).1) x := by
  obtain rfl : inst = (by clear inst; exact inferInstance) := Subsingleton.elim _ _
  have h := HasPullbackAt.comp
    (f := fun y => WithLp.toLp 2 (WithLp.toLp 2 (fq y, WithLp.toLp 2 (fk y,
      WithLp.toLp 2 (fv y, fo y))), fx y)) (x := x)
    (MultiHeadAttention.hasPullbackAt_mapLeading h1 mask batch
      (WithLp.toLp 2 (fq x, WithLp.toLp 2 (fk x, WithLp.toLp 2 (fv x, fo x)))) (fx x))
    ((hq.toLp_prod (hk.toLp_prod (hv.toLp_prod ho))).toLp_prod hx)
  refine h.congr_deriv h.hasFDerivAt.fderiv.symm fun δ => ?_
  simp only [WithLp.toLp_fst, WithLp.toLp_snd]
  rw [← MultiHeadAttention.toParams_sum ⟨fq x, fk x, fv x, fo x⟩]
  simp only [MultiHeadAttention.toParams, WithLp.toLp_fst, WithLp.toLp_snd]

end HasPullbackAt

namespace Op

/-- With no leading axes, multi-head attention is differentiable jointly in the four projections
and the input, and its stored backward is the adjoint of the derivative. -/
theorem multiheadAttention_single_backward_eq_adjoint {n numHeads dModel headDim : Nat}
    (h1 : n ≠ 0) (mask : Option (Tensor Bool [n, n]))
    (wq wk wv : Tensor ℝ [dModel, numHeads * headDim])
    (wo : Tensor ℝ [numHeads * headDim, dModel]) (x δ : Tensor ℝ [n, dModel]) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ [dModel, numHeads * headDim] ×
          WithLp 2 (Tensor ℝ [dModel, numHeads * headDim] ×
            WithLp 2 (Tensor ℝ [dModel, numHeads * headDim] ×
              WithLp 2 (Tensor ℝ [numHeads * headDim, dModel] × Tensor ℝ [n, dModel])))) =>
        (((Op.multiheadAttention (batch := []) (numHeads := numHeads) (dModel := dModel)
          (headDim := headDim) h1 mask).rule? (α := ℝ) |>.get rfl).forward
          (p.fst, p.snd.fst, p.snd.snd.fst, p.snd.snd.snd.fst, p.snd.snd.snd.snd, ())).1)
      (WithLp.toLp 2 (wq, WithLp.toLp 2 (wk, WithLp.toLp 2 (wv, WithLp.toLp 2 (wo, x))))) ∧
    WithLp.toLp 2
        ((((Op.multiheadAttention (batch := []) (numHeads := numHeads) (dModel := dModel)
          (headDim := headDim) h1 mask).rule? (α := ℝ) |>.get rfl).backward
          (wq, wk, wv, wo, x, ()) (δ, ())).1,
        WithLp.toLp 2
          ((((Op.multiheadAttention (batch := []) (numHeads := numHeads) (dModel := dModel)
            (headDim := headDim) h1 mask).rule? (α := ℝ) |>.get rfl).backward
            (wq, wk, wv, wo, x, ()) (δ, ())).2.1,
          WithLp.toLp 2
            ((((Op.multiheadAttention (batch := []) (numHeads := numHeads) (dModel := dModel)
              (headDim := headDim) h1 mask).rule? (α := ℝ) |>.get rfl).backward
              (wq, wk, wv, wo, x, ()) (δ, ())).2.2.1,
            WithLp.toLp 2
              ((((Op.multiheadAttention (batch := []) (numHeads := numHeads) (dModel := dModel)
                (headDim := headDim) h1 mask).rule? (α := ℝ) |>.get rfl).backward
                (wq, wk, wv, wo, x, ()) (δ, ())).2.2.2.1,
              (((Op.multiheadAttention (batch := []) (numHeads := numHeads) (dModel := dModel)
                (headDim := headDim) h1 mask).rule? (α := ℝ) |>.get rfl).backward
                (wq, wk, wv, wo, x, ()) (δ, ())).2.2.2.2.1)))) =
      (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ [dModel, numHeads * headDim] ×
          WithLp 2 (Tensor ℝ [dModel, numHeads * headDim] ×
            WithLp 2 (Tensor ℝ [dModel, numHeads * headDim] ×
              WithLp 2 (Tensor ℝ [numHeads * headDim, dModel] × Tensor ℝ [n, dModel])))) =>
        (((Op.multiheadAttention (batch := []) (numHeads := numHeads) (dModel := dModel)
          (headDim := headDim) h1 mask).rule? (α := ℝ) |>.get rfl).forward
          (p.fst, p.snd.fst, p.snd.snd.fst, p.snd.snd.snd.fst, p.snd.snd.snd.snd, ())).1)
        (WithLp.toLp 2 (wq, WithLp.toLp 2 (wk, WithLp.toLp 2 (wv,
          WithLp.toLp 2 (wo, x)))))).adjoint δ := by
  have hid := HasPullbackAt.id
    (WithLp.toLp 2 (wq, WithLp.toLp 2 (wk, WithLp.toLp 2 (wv, WithLp.toLp 2 (wo, x)))))
  have hx := hid.snd.snd.snd.snd
  have hq := (hx.matmul hid.fst).splitHeads (numHeads := numHeads) (headDim := headDim)
  have hk := (hx.matmul hid.snd.fst).splitHeads (numHeads := numHeads) (headDim := headDim)
  have hv := (hx.matmul hid.snd.snd.fst).splitHeads (numHeads := numHeads) (headDim := headDim)
  have hheads := HasPullbackAt.comp
    (f := fun y : WithLp 2 (Tensor ℝ [dModel, numHeads * headDim] ×
        WithLp 2 (Tensor ℝ [dModel, numHeads * headDim] ×
          WithLp 2 (Tensor ℝ [dModel, numHeads * headDim] ×
            WithLp 2 (Tensor ℝ [numHeads * headDim, dModel] × Tensor ℝ [n, dModel])))) =>
      WithLp.toLp 2
        (MultiHeadAttention.splitHeads (Tensor.matmul y.snd.snd.snd.snd y.fst) numHeads headDim rfl,
        WithLp.toLp 2
          (MultiHeadAttention.splitHeads (Tensor.matmul y.snd.snd.snd.snd y.snd.fst) numHeads
            headDim rfl,
          MultiHeadAttention.splitHeads (Tensor.matmul y.snd.snd.snd.snd y.snd.snd.fst) numHeads
            headDim rfl)))
    (x := WithLp.toLp 2 (wq, WithLp.toLp 2 (wk, WithLp.toLp 2 (wv, WithLp.toLp 2 (wo, x)))))
    (Attention.hasPullbackAt_heads h1 h1 mask
      (MultiHeadAttention.splitHeads (Tensor.matmul x wq) numHeads headDim rfl)
      (MultiHeadAttention.splitHeads (Tensor.matmul x wk) numHeads headDim rfl)
      (MultiHeadAttention.splitHeads (Tensor.matmul x wv) numHeads headDim rfl))
    (hq.toLp_prod (hk.toLp_prod hv))
  have hout := hheads.combineHeads.matmul hid.snd.snd.snd.fst
  refine hout.differentiableAt_and ?_
  simp only [← WithLp.toLp_add, Prod.mk_add_mk, add_zero, zero_add]
  rw [← add_assoc]
  rfl

/-- At every leading shape, mask and size, multi-head self-attention is differentiable jointly in
the four projections and the input, and its stored backward is the adjoint of the derivative. -/
@[op_cert TorchLean.Op.multiheadAttention derivative 0 1 2 3 4]
theorem multiheadAttention_backward_eq_adjoint {batch : Shape} {n numHeads dModel headDim : Nat}
    (h1 : n ≠ 0) (mask : Option (Tensor Bool [n, n]))
    (wq wk wv : Tensor ℝ [dModel, numHeads * headDim])
    (wo : Tensor ℝ [numHeads * headDim, dModel]) (x δ : Tensor ℝ (batch ++ [n, dModel])) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ [dModel, numHeads * headDim] ×
          WithLp 2 (Tensor ℝ [dModel, numHeads * headDim] ×
            WithLp 2 (Tensor ℝ [dModel, numHeads * headDim] ×
              WithLp 2 (Tensor ℝ [numHeads * headDim, dModel] ×
                Tensor ℝ (batch ++ [n, dModel]))))) =>
        (((Op.multiheadAttention (batch := batch) (numHeads := numHeads) (dModel := dModel)
          (headDim := headDim) h1 mask).rule? (α := ℝ) |>.get rfl).forward
          (p.fst, p.snd.fst, p.snd.snd.fst, p.snd.snd.snd.fst, p.snd.snd.snd.snd, ())).1)
      (WithLp.toLp 2 (wq, WithLp.toLp 2 (wk, WithLp.toLp 2 (wv, WithLp.toLp 2 (wo, x))))) ∧
    WithLp.toLp 2
        ((((Op.multiheadAttention (batch := batch) (numHeads := numHeads) (dModel := dModel)
          (headDim := headDim) h1 mask).rule? (α := ℝ) |>.get rfl).backward
          (wq, wk, wv, wo, x, ()) (δ, ())).1,
        WithLp.toLp 2
          ((((Op.multiheadAttention (batch := batch) (numHeads := numHeads) (dModel := dModel)
            (headDim := headDim) h1 mask).rule? (α := ℝ) |>.get rfl).backward
            (wq, wk, wv, wo, x, ()) (δ, ())).2.1,
          WithLp.toLp 2
            ((((Op.multiheadAttention (batch := batch) (numHeads := numHeads)
              (dModel := dModel) (headDim := headDim) h1 mask).rule? (α := ℝ) |>.get rfl).backward
              (wq, wk, wv, wo, x, ()) (δ, ())).2.2.1,
            WithLp.toLp 2
              ((((Op.multiheadAttention (batch := batch) (numHeads := numHeads)
                (dModel := dModel) (headDim := headDim) h1 mask).rule? (α := ℝ)
                |>.get rfl).backward (wq, wk, wv, wo, x, ()) (δ, ())).2.2.2.1,
              (((Op.multiheadAttention (batch := batch) (numHeads := numHeads)
                (dModel := dModel) (headDim := headDim) h1 mask).rule? (α := ℝ)
                |>.get rfl).backward (wq, wk, wv, wo, x, ()) (δ, ())).2.2.2.2.1)))) =
      (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ [dModel, numHeads * headDim] ×
          WithLp 2 (Tensor ℝ [dModel, numHeads * headDim] ×
            WithLp 2 (Tensor ℝ [dModel, numHeads * headDim] ×
              WithLp 2 (Tensor ℝ [numHeads * headDim, dModel] ×
                Tensor ℝ (batch ++ [n, dModel]))))) =>
        (((Op.multiheadAttention (batch := batch) (numHeads := numHeads) (dModel := dModel)
          (headDim := headDim) h1 mask).rule? (α := ℝ) |>.get rfl).forward
          (p.fst, p.snd.fst, p.snd.snd.fst, p.snd.snd.snd.fst, p.snd.snd.snd.snd, ())).1)
        (WithLp.toLp 2 (wq, WithLp.toLp 2 (wk, WithLp.toLp 2 (wv,
          WithLp.toLp 2 (wo, x)))))).adjoint δ := by
  op_deriv

end Op

end

end TorchLean
