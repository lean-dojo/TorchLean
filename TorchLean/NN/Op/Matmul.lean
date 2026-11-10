/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Attention.Adjoint
public import TorchLean.NN.FDeriv.PullbackTensor
public import TorchLean.Tensor.Op.Table.Linear
public import TorchLean.NN.Gradients.Linear
public import TorchLean.NN.Op.Certificates
public import TorchLean.NN.Tactic.Init
public import TorchLean.Tensor.Op.Cert
public import TorchLean.Tensor.Operations

/-!
# Pullbacks of matrix products

We prove that the stored backward maps of `Op.matmul`, `Op.scaledMatmul` and `Op.linear` are the
adjoints of the derivatives of their forward maps. The derivative is taken jointly in both operands
of `Op.matmul` and `Op.scaledMatmul`, with the scale of `Op.scaledMatmul` held fixed, and jointly in
the weights, the bias and the input of `Op.linear`. Each product is a continuous bilinear map,
built from two transpose identities that pair the product with a component of the stored backward
map, and `HasPullbackAt.bilin` turns each such map into a `fun_prop` composition rule.

For `Op.matmul` and `Op.scaledMatmul` the transpose identities reduce to the matrix identities
`dot_matmul_right_adjoint` and `dot_matmul_left_adjoint` by induction on the batch prefix,
and to `inner_broadcast` for the broadcast batch prefixes. For `Op.linear` we write the forward map
over a leading shape as the matrix-vector product with the weights at every leading index plus the
bias repeated at every leading index. The transpose in the input maps `inputDeriv` over the leading
shape, the transpose in the weights flattens the leading shape to one axis and applies
`dot_matmul_left_adjoint`, and the transpose of the repeated bias is `biasDerivLeading`. The
certificates then follow from the composition rules by `fun_prop`.

The same construction gives continuous bilinear maps and `fun_prop` rules for the common-batch
product `matmulCommonBatch` and for the general contraction `Tensor.contract`, and the joint
certificate of `Op.contract` in both operands. The `op_deriv_simps` lemmas state that
`matmulBackward`, `Linear.inputDeriv` and `Linear.weightsDeriv` are linear in the cotangent.
-/

@[expose] public section

open scoped RealInnerProductSpace

namespace TorchLean

section Builders

variable {F G H : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F]
  [NormedAddCommGroup G] [InnerProductSpace ℝ G] [NormedAddCommGroup H] [InnerProductSpace ℝ H]

/-- A map `L` with a transpose `T` (`⟪L u, δ⟫ = ⟪u, T δ⟫`) on a finite-dimensional space, as a
continuous linear map. -/
noncomputable def clmOfTranspose [FiniteDimensional ℝ F] (L : F → G) (T : G → F)
    (h : ∀ u δ, ⟪L u, δ⟫ = ⟪u, T δ⟫) : F →L[ℝ] G :=
  let linear : F →ₗ[ℝ] G :=
    { toFun := L
      map_add' := fun u u' => ext_inner_right ℝ fun δ => by
        rw [inner_add_left, h, h, h, inner_add_left]
      map_smul' := fun c u => ext_inner_right ℝ fun δ => by
        rw [RingHom.id_apply, real_inner_smul_left, h, h, real_inner_smul_left] }
  ⟨linear, linear.continuous_of_finiteDimensional⟩

/-- `clmOfTranspose L T h` applied to `u` is `L u`. -/
@[simp] theorem clmOfTranspose_apply [FiniteDimensional ℝ F] (L : F → G) (T : G → F)
    (h : ∀ u δ, ⟪L u, δ⟫ = ⟪u, T δ⟫) (u : F) : clmOfTranspose L T h u = L u :=
  rfl

/-- A map `B` of two arguments with transposes `TL` and `TR` in each of them, on
finite-dimensional spaces, as a continuous bilinear map. -/
noncomputable def clm₂OfTranspose [FiniteDimensional ℝ F] [FiniteDimensional ℝ G]
    (B : F → G → H) (TL : G → H → F) (TR : F → H → G)
    (hL : ∀ u v δ, ⟪B u v, δ⟫ = ⟪u, TL v δ⟫) (hR : ∀ u v δ, ⟪B u v, δ⟫ = ⟪v, TR u δ⟫) :
    F →L[ℝ] G →L[ℝ] H :=
  let linear : F →ₗ[ℝ] G →L[ℝ] H :=
    { toFun := fun u => clmOfTranspose (B u) (TR u) (hR u)
      map_add' := fun u u' => ContinuousLinearMap.ext fun v => ext_inner_right ℝ fun δ => by
        rw [_root_.add_apply, clmOfTranspose_apply, clmOfTranspose_apply, clmOfTranspose_apply,
          inner_add_left, hL, hL, hL, inner_add_left]
      map_smul' := fun c u => ContinuousLinearMap.ext fun v => ext_inner_right ℝ fun δ => by
        rw [RingHom.id_apply, _root_.smul_apply, clmOfTranspose_apply, clmOfTranspose_apply,
          real_inner_smul_left, hL, hL, real_inner_smul_left] }
  ⟨linear, linear.continuous_of_finiteDimensional⟩

/-- `clm₂OfTranspose B TL TR hL hR` applied to `u` and `v` is `B u v`. -/
@[simp] theorem clm₂OfTranspose_apply [FiniteDimensional ℝ F] [FiniteDimensional ℝ G]
    (B : F → G → H) (TL : G → H → F) (TR : F → H → G)
    (hL : ∀ u v δ, ⟪B u v, δ⟫ = ⟪u, TL v δ⟫) (hR : ∀ u v δ, ⟪B u v, δ⟫ = ⟪v, TR u δ⟫) (u : F)
    (v : G) : clm₂OfTranspose B TL TR hL hR u v = B u v :=
  rfl

end Builders

namespace Tensor

variable {batchA batchB batch : Shape} {m n p : Nat}

/-- The inner product of a common-batch matrix product with `C` is the inner product of the left
operand with `C` times the transposed right operand. -/
theorem inner_matmulCommonBatch_left (A : Tensor ℝ (batch ++ [m, n]))
    (B : Tensor ℝ (batch ++ [n, p])) (C : Tensor ℝ (batch ++ [m, p])) :
    ⟪matmulCommonBatch A B, C⟫ =
      ⟪A, matmulCommonBatch C
        (permute B (Shape.adjacentSwap (batch ++ [n, p]) batch.rank) (batch ++ [p, n]))⟫ := by
  rw [← dot_eq_inner, ← dot_eq_inner]
  induction batch with
  | scalar => exact dot_matmul_right_adjoint A B C
  | dim k batch ih =>
    simp only [matmulCommonBatch]
    rw [dot_eq_sum_unstack, dot_eq_sum_unstack]
    refine Finset.sum_congr rfl fun i _ => ?_
    simp only [Tensor.unstack_dim]
    rw [unstack_permute_batch_transpose B _ i]
    exact ih (A.unstack i) (B.unstack i) (C.unstack i)

/-- The inner product of a common-batch matrix product with `C` is the inner product of the right
operand with the transposed left operand times `C`. -/
theorem inner_matmulCommonBatch_right (A : Tensor ℝ (batch ++ [m, n]))
    (B : Tensor ℝ (batch ++ [n, p])) (C : Tensor ℝ (batch ++ [m, p])) :
    ⟪matmulCommonBatch A B, C⟫ =
      ⟪B, matmulCommonBatch
        (permute A (Shape.adjacentSwap (batch ++ [m, n]) batch.rank) (batch ++ [n, m])) C⟫ := by
  rw [← dot_eq_inner, ← dot_eq_inner]
  induction batch with
  | scalar => exact dot_matmul_left_adjoint A B C
  | dim k batch ih =>
    simp only [matmulCommonBatch]
    rw [dot_eq_sum_unstack, dot_eq_sum_unstack]
    refine Finset.sum_congr rfl fun i _ => ?_
    simp only [Tensor.unstack_dim]
    rw [unstack_permute_batch_transpose A _ i]
    exact ih (A.unstack i) (B.unstack i) (C.unstack i)

/-- The common-batch matrix product, as a continuous bilinear map. -/
noncomputable def matmulCommonBatchCLM :
    Tensor ℝ (batch ++ [m, n]) →L[ℝ] Tensor ℝ (batch ++ [n, p]) →L[ℝ]
      Tensor ℝ (batch ++ [m, p]) :=
  clm₂OfTranspose matmulCommonBatch
    (fun B C => matmulCommonBatch C
      (permute B (Shape.adjacentSwap (batch ++ [n, p]) batch.rank) (batch ++ [p, n])))
    (fun A C => matmulCommonBatch
      (permute A (Shape.adjacentSwap (batch ++ [m, n]) batch.rank) (batch ++ [n, m])) C)
    inner_matmulCommonBatch_left inner_matmulCommonBatch_right

/-- `matmulCommonBatchCLM` applied to `A` and `B` is `matmulCommonBatch A B`. -/
@[simp] theorem matmulCommonBatchCLM_apply (A : Tensor ℝ (batch ++ [m, n]))
    (B : Tensor ℝ (batch ++ [n, p])) : matmulCommonBatchCLM A B = matmulCommonBatch A B :=
  rfl

variable (broadcastA : Shape.CanBroadcastTo batchA batch)
  (broadcastB : Shape.CanBroadcastTo batchB batch)

/-- The first component of `matmulBackward` is the transpose of `batchMatmul` in its left
operand: it reads the right operand only, so the left operand `a` it receives is arbitrary. -/
theorem inner_batchMatmul_left (a u : Tensor ℝ (batchA ++ [m, n]))
    (v : Tensor ℝ (batchB ++ [n, p])) (δ : Tensor ℝ (batch ++ [m, p])) :
    ⟪batchMatmul broadcastA broadcastB u v, δ⟫ =
      ⟪u, (matmulBackward broadcastA broadcastB a v δ).1⟫ := by
  simp only [batchMatmul, matmulBackward]
  rw [inner_matmulCommonBatch_left, inner_broadcast]

/-- The second component of `matmulBackward` is the transpose of `batchMatmul` in its right
operand: it reads the left operand only, so the right operand `b` it receives is arbitrary. -/
theorem inner_batchMatmul_right (u : Tensor ℝ (batchA ++ [m, n]))
    (b v : Tensor ℝ (batchB ++ [n, p])) (δ : Tensor ℝ (batch ++ [m, p])) :
    ⟪batchMatmul broadcastA broadcastB u v, δ⟫ =
      ⟪v, (matmulBackward broadcastA broadcastB u b δ).2⟫ := by
  simp only [batchMatmul, matmulBackward]
  rw [inner_matmulCommonBatch_right, inner_broadcast]

/-- The broadcast batched matrix product, as a continuous bilinear map. -/
noncomputable def batchMatmulCLM :
    Tensor ℝ (batchA ++ [m, n]) →L[ℝ] Tensor ℝ (batchB ++ [n, p]) →L[ℝ]
      Tensor ℝ (batch ++ [m, p]) :=
  clm₂OfTranspose (batchMatmul broadcastA broadcastB)
    (fun v δ => (matmulBackward broadcastA broadcastB 0 v δ).1)
    (fun u δ => (matmulBackward broadcastA broadcastB u 0 δ).2)
    (fun u v δ => inner_batchMatmul_left broadcastA broadcastB 0 u v δ)
    (fun u v δ => inner_batchMatmul_right broadcastA broadcastB u 0 v δ)

/-- `batchMatmulCLM` applied to `u` and `v` is `batchMatmul u v`. -/
@[simp] theorem batchMatmulCLM_apply (u : Tensor ℝ (batchA ++ [m, n]))
    (v : Tensor ℝ (batchB ++ [n, p])) :
    batchMatmulCLM broadcastA broadcastB u v = batchMatmul broadcastA broadcastB u v :=
  rfl

section Linearity

variable (a : Tensor ℝ (batchA ++ [m, n])) (b : Tensor ℝ (batchB ++ [n, p]))
  (δ δ' : Tensor ℝ (batch ++ [m, p])) (k : ℝ)

/-- The left component of `matmulBackward` maps a sum of cotangents to the sum of the
components. -/
@[op_deriv_simps] theorem matmulBackward_fst_add :
    (matmulBackward broadcastA broadcastB a b (δ + δ')).1 =
      (matmulBackward broadcastA broadcastB a b δ).1 +
        (matmulBackward broadcastA broadcastB a b δ').1 :=
  ext_inner_left ℝ fun u => by
    simp only [inner_add_right, ← inner_batchMatmul_left broadcastA broadcastB a]

/-- The left component of `matmulBackward` commutes with multiplying the cotangent by `k`. -/
@[op_deriv_simps] theorem matmulBackward_fst_smul :
    (matmulBackward broadcastA broadcastB a b (k • δ)).1 =
      k • (matmulBackward broadcastA broadcastB a b δ).1 :=
  ext_inner_left ℝ fun u => by
    simp only [real_inner_smul_right, ← inner_batchMatmul_left broadcastA broadcastB a]

/-- The left component of `matmulBackward` maps a negated cotangent to the negated component. -/
@[op_deriv_simps] theorem matmulBackward_fst_neg :
    (matmulBackward broadcastA broadcastB a b (-δ)).1 =
      -(matmulBackward broadcastA broadcastB a b δ).1 :=
  ext_inner_left ℝ fun u => by
    simp only [inner_neg_right, ← inner_batchMatmul_left broadcastA broadcastB a]

/-- The left component of `matmulBackward` maps the zero cotangent to zero. -/
@[op_deriv_simps] theorem matmulBackward_fst_zero :
    (matmulBackward broadcastA broadcastB a b 0).1 = 0 :=
  ext_inner_left ℝ fun u => by
    simp only [inner_zero_right, ← inner_batchMatmul_left broadcastA broadcastB a]

/-- The right component of `matmulBackward` maps a sum of cotangents to the sum of the
components. -/
@[op_deriv_simps] theorem matmulBackward_snd_add :
    (matmulBackward broadcastA broadcastB a b (δ + δ')).2 =
      (matmulBackward broadcastA broadcastB a b δ).2 +
        (matmulBackward broadcastA broadcastB a b δ').2 :=
  ext_inner_left ℝ fun v => by
    simp only [inner_add_right, ← inner_batchMatmul_right broadcastA broadcastB a b]

/-- The right component of `matmulBackward` commutes with multiplying the cotangent by `k`. -/
@[op_deriv_simps] theorem matmulBackward_snd_smul :
    (matmulBackward broadcastA broadcastB a b (k • δ)).2 =
      k • (matmulBackward broadcastA broadcastB a b δ).2 :=
  ext_inner_left ℝ fun v => by
    simp only [real_inner_smul_right, ← inner_batchMatmul_right broadcastA broadcastB a b]

/-- The right component of `matmulBackward` maps a negated cotangent to the negated
component. -/
@[op_deriv_simps] theorem matmulBackward_snd_neg :
    (matmulBackward broadcastA broadcastB a b (-δ)).2 =
      -(matmulBackward broadcastA broadcastB a b δ).2 :=
  ext_inner_left ℝ fun v => by
    simp only [inner_neg_right, ← inner_batchMatmul_right broadcastA broadcastB a b]

/-- The right component of `matmulBackward` maps the zero cotangent to zero. -/
@[op_deriv_simps] theorem matmulBackward_snd_zero :
    (matmulBackward broadcastA broadcastB a b 0).2 = 0 :=
  ext_inner_left ℝ fun v => by
    simp only [inner_zero_right, ← inner_batchMatmul_right broadcastA broadcastB a b]

/-- The left component of the two-dimensional `matmulBackward` maps a sum of cotangents to the
sum of the components. -/
@[op_deriv_simps] theorem matmulBackward_refl_fst_add (a : Tensor ℝ [m, n]) (b : Tensor ℝ [n, p])
    (δ δ' : Tensor ℝ [m, p]) :
    (matmulBackward (Shape.CanBroadcastTo.refl .scalar) (Shape.CanBroadcastTo.refl .scalar) a b
        (δ + δ')).1 =
      (matmulBackward (Shape.CanBroadcastTo.refl .scalar) (Shape.CanBroadcastTo.refl .scalar) a b
          δ).1 +
        (matmulBackward (Shape.CanBroadcastTo.refl .scalar) (Shape.CanBroadcastTo.refl .scalar)
          a b δ').1 :=
  matmulBackward_fst_add (batchA := .scalar) (batchB := .scalar) (batch := .scalar) _ _ a b δ δ'

/-- The left component of the two-dimensional `matmulBackward` commutes with multiplying the
cotangent by `k`. -/
@[op_deriv_simps] theorem matmulBackward_refl_fst_smul (a : Tensor ℝ [m, n])
    (b : Tensor ℝ [n, p]) (δ : Tensor ℝ [m, p]) (k : ℝ) :
    (matmulBackward (Shape.CanBroadcastTo.refl .scalar) (Shape.CanBroadcastTo.refl .scalar) a b
        (k • δ)).1 =
      k • (matmulBackward (Shape.CanBroadcastTo.refl .scalar)
        (Shape.CanBroadcastTo.refl .scalar) a b δ).1 :=
  matmulBackward_fst_smul (batchA := .scalar) (batchB := .scalar) (batch := .scalar) _ _ a b δ k

/-- The right component of the two-dimensional `matmulBackward` maps a sum of cotangents to the
sum of the components. -/
@[op_deriv_simps] theorem matmulBackward_refl_snd_add (a : Tensor ℝ [m, n]) (b : Tensor ℝ [n, p])
    (δ δ' : Tensor ℝ [m, p]) :
    (matmulBackward (Shape.CanBroadcastTo.refl .scalar) (Shape.CanBroadcastTo.refl .scalar) a b
        (δ + δ')).2 =
      (matmulBackward (Shape.CanBroadcastTo.refl .scalar) (Shape.CanBroadcastTo.refl .scalar) a b
          δ).2 +
        (matmulBackward (Shape.CanBroadcastTo.refl .scalar) (Shape.CanBroadcastTo.refl .scalar)
          a b δ').2 :=
  matmulBackward_snd_add (batchA := .scalar) (batchB := .scalar) (batch := .scalar) _ _ a b δ δ'

/-- The right component of the two-dimensional `matmulBackward` commutes with multiplying the
cotangent by `k`. -/
@[op_deriv_simps] theorem matmulBackward_refl_snd_smul (a : Tensor ℝ [m, n])
    (b : Tensor ℝ [n, p]) (δ : Tensor ℝ [m, p]) (k : ℝ) :
    (matmulBackward (Shape.CanBroadcastTo.refl .scalar) (Shape.CanBroadcastTo.refl .scalar) a b
        (k • δ)).2 =
      k • (matmulBackward (Shape.CanBroadcastTo.refl .scalar)
        (Shape.CanBroadcastTo.refl .scalar) a b δ).2 :=
  matmulBackward_snd_smul (batchA := .scalar) (batchB := .scalar) (batch := .scalar) _ _ a b δ k

end Linearity

section Contract

variable {s t : Shape} (dims batchDims : List (Fin s.rank × Fin t.rank))
  (h : Shape.ContractValid s t dims batchDims)

/-- Real contraction along `dims` and `batchDims`, as a continuous bilinear map. -/
noncomputable def contractCLM :
    Tensor ℝ s →L[ℝ] Tensor ℝ t →L[ℝ] Tensor ℝ (s.contract t dims batchDims) :=
  clm₂OfTranspose (fun x y => contract x y dims batchDims _ h rfl)
    (fun y g => contractBackwardLeft dims batchDims h g y)
    (fun x g => contractBackwardRight dims batchDims h x g)
    (inner_contract_eq_inner_contractBackwardLeft h)
    (inner_contract_eq_inner_contractBackwardRight h)

/-- `contractCLM dims batchDims h` applied to `x` and `y` is `contract x y dims batchDims`. -/
@[simp] theorem contractCLM_apply (x : Tensor ℝ s) (y : Tensor ℝ t) :
    contractCLM dims batchDims h x y = contract x y dims batchDims _ h rfl :=
  rfl

end Contract

end Tensor

namespace HasPullbackAt

open Tensor

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] {x : E}
  {batchA batchB batch : Shape} {m n p : Nat}
  {f : E → Tensor ℝ (batchA ++ [m, n])} {g : E → Tensor ℝ (batchB ++ [n, p])}
  {Df : E →L[ℝ] Tensor ℝ (batchA ++ [m, n])} {Dg : E →L[ℝ] Tensor ℝ (batchB ++ [n, p])}
  {bf : Tensor ℝ (batchA ++ [m, n]) → E} {bg : Tensor ℝ (batchB ++ [n, p]) → E}

/-- The broadcast batched matrix product of two maps, by the product rule, pulled back by the two
components of `matmulBackward` at the operands `f x` and `g x`. -/
@[fun_prop]
theorem batchMatmul (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch) (hf : HasPullbackAt f Df bf x)
    (hg : HasPullbackAt g Dg bg x) :
    HasPullbackAt (fun y => Tensor.batchMatmul broadcastA broadcastB (f y) (g y))
      (((batchMatmulCLM broadcastA broadcastB).flip (g x)).comp Df +
        (batchMatmulCLM broadcastA broadcastB (f x)).comp Dg)
      (fun δ => bf (matmulBackward broadcastA broadcastB (f x) (g x) δ).1 +
        bg (matmulBackward broadcastA broadcastB (f x) (g x) δ).2) x := by
  simpa only [batchMatmulCLM_apply] using
    bilin (batchMatmulCLM broadcastA broadcastB)
      (fun v δ => (matmulBackward broadcastA broadcastB (f x) v δ).1)
      (fun u δ => (matmulBackward broadcastA broadcastB u (g x) δ).2)
      (fun u v δ => inner_batchMatmul_left broadcastA broadcastB (f x) u v δ)
      (fun u v δ => inner_batchMatmul_right broadcastA broadcastB u (g x) v δ) hf hg

/-- The common-batch matrix product of two maps, by the product rule, pulled back by the products
of the cotangent with the transposed operands `f x` and `g x`. -/
@[fun_prop]
theorem matmulCommonBatch {batch : Shape} {m n p : Nat} {f : E → Tensor ℝ (batch ++ [m, n])}
    {g : E → Tensor ℝ (batch ++ [n, p])} {Df : E →L[ℝ] Tensor ℝ (batch ++ [m, n])}
    {Dg : E →L[ℝ] Tensor ℝ (batch ++ [n, p])} {bf : Tensor ℝ (batch ++ [m, n]) → E}
    {bg : Tensor ℝ (batch ++ [n, p]) → E} (hf : HasPullbackAt f Df bf x)
    (hg : HasPullbackAt g Dg bg x) :
    HasPullbackAt (fun y => Tensor.matmulCommonBatch (f y) (g y))
      ((matmulCommonBatchCLM.flip (g x)).comp Df + (matmulCommonBatchCLM (f x)).comp Dg)
      (fun δ => bf (Tensor.matmulCommonBatch δ
          (permute (g x) (Shape.adjacentSwap (batch ++ [n, p]) batch.rank) (batch ++ [p, n]))) +
        bg (Tensor.matmulCommonBatch
          (permute (f x) (Shape.adjacentSwap (batch ++ [m, n]) batch.rank) (batch ++ [n, m]))
          δ)) x := by
  simpa only [matmulCommonBatchCLM_apply] using
    bilin matmulCommonBatchCLM
      (fun B C => Tensor.matmulCommonBatch C
        (permute B (Shape.adjacentSwap (batch ++ [n, p]) batch.rank) (batch ++ [p, n])))
      (fun A C => Tensor.matmulCommonBatch
        (permute A (Shape.adjacentSwap (batch ++ [m, n]) batch.rank) (batch ++ [n, m])) C)
      inner_matmulCommonBatch_left inner_matmulCommonBatch_right hf hg

/-- Real contraction of two maps, by the product rule, pulled back by `contractBackwardLeft` at
`g x` and `contractBackwardRight` at `f x`. -/
@[fun_prop]
theorem contract {s t : Shape} (dims batchDims : List (Fin s.rank × Fin t.rank))
    (h : Shape.ContractValid s t dims batchDims) {f : E → Tensor ℝ s} {g : E → Tensor ℝ t}
    {Df : E →L[ℝ] Tensor ℝ s} {Dg : E →L[ℝ] Tensor ℝ t} {bf : Tensor ℝ s → E}
    {bg : Tensor ℝ t → E} (hf : HasPullbackAt f Df bf x) (hg : HasPullbackAt g Dg bg x) :
    HasPullbackAt (fun y => Tensor.contract (f y) (g y) dims batchDims _ h rfl)
      (((contractCLM dims batchDims h).flip (g x)).comp Df +
        (contractCLM dims batchDims h (f x)).comp Dg)
      (fun δ => bf (contractBackwardLeft dims batchDims h δ (g x)) +
        bg (contractBackwardRight dims batchDims h (f x) δ)) x := by
  simpa only [contractCLM_apply] using
    bilin (contractCLM dims batchDims h) (fun y g => contractBackwardLeft dims batchDims h g y)
      (fun x g => contractBackwardRight dims batchDims h x g)
      (inner_contract_eq_inner_contractBackwardLeft h)
      (inner_contract_eq_inner_contractBackwardRight h) hf hg

end HasPullbackAt

namespace Attention

open Tensor

/-- Over the reals, a one-group `scaledProducts` is the common-batch matrix product scaled by
`c`. -/
theorem scaledProducts_one_eq_scale {batch : Shape} {m n p : Nat}
    (X : Tensor ℝ (batch ++ [m, n])) (Y : Tensor ℝ (batch ++ [n, p])) (c : ℝ) :
    scaledProducts (fun _ : Fin 1 => X) (fun _ => Y) c =
      Tensor.scale (matmulCommonBatch X Y) c := by
  rw [scaledProductsFast_eq]
  induction batch with
  | scalar =>
    simp only [scaledProductsFast, scaledProductsSlice_one
      Numeric.QuotientArithmetic.supported_real, Bool.false_eq_true, ↓reduceIte,
      matmulCommonBatch_scalar]
  | dim k batch ih =>
    simp only [scaledProductsFast, ih, matmulCommonBatch, Tensor.scale, map_dim]

variable {batchA batchB batch : Shape} {m n p : Nat}
  (broadcastA : Shape.CanBroadcastTo batchA batch)
  (broadcastB : Shape.CanBroadcastTo batchB batch)

/-- The first component of `scaledMatmulBackward` is the transpose of `scaledMatmul` in its left
operand: it reads the right operand only, so the left operand `a` it receives is arbitrary. -/
theorem inner_scaledMatmul_left (a u : Tensor ℝ (batchA ++ [m, n]))
    (v : Tensor ℝ (batchB ++ [n, p])) (δ : Tensor ℝ (batch ++ [m, p])) (c : ℝ) :
    ⟪scaledMatmul broadcastA broadcastB u v c, δ⟫ =
      ⟪u, (scaledMatmulBackward broadcastA broadcastB a v δ c).1⟫ := by
  simp only [scaledMatmul, scaledMatmulBackward, scaledProducts_one_eq_scale]
  rw [← inner_broadcast, ← dot_eq_inner, ← dot_eq_inner, dot_scale_left,
    dot_scale_right, dot_eq_inner, dot_eq_inner, inner_matmulCommonBatch_left]

/-- The second component of `scaledMatmulBackward` is the transpose of `scaledMatmul` in its right
operand: it reads the left operand only, so the right operand `b` it receives is arbitrary. -/
theorem inner_scaledMatmul_right (u : Tensor ℝ (batchA ++ [m, n]))
    (b v : Tensor ℝ (batchB ++ [n, p])) (δ : Tensor ℝ (batch ++ [m, p])) (c : ℝ) :
    ⟪scaledMatmul broadcastA broadcastB u v c, δ⟫ =
      ⟪v, (scaledMatmulBackward broadcastA broadcastB u b δ c).2⟫ := by
  simp only [scaledMatmul, scaledMatmulBackward, scaledProducts_one_eq_scale]
  rw [← inner_broadcast, ← dot_eq_inner, ← dot_eq_inner, dot_scale_left,
    dot_scale_right, dot_eq_inner, dot_eq_inner, inner_matmulCommonBatch_right]

/-- The scaled matrix product with a fixed scale `c`, as a continuous bilinear map. -/
noncomputable def scaledMatmulCLM (c : ℝ) :
    Tensor ℝ (batchA ++ [m, n]) →L[ℝ] Tensor ℝ (batchB ++ [n, p]) →L[ℝ]
      Tensor ℝ (batch ++ [m, p]) :=
  clm₂OfTranspose (fun u v => scaledMatmul broadcastA broadcastB u v c)
    (fun v δ => (scaledMatmulBackward broadcastA broadcastB 0 v δ c).1)
    (fun u δ => (scaledMatmulBackward broadcastA broadcastB u 0 δ c).2)
    (fun u v δ => inner_scaledMatmul_left broadcastA broadcastB 0 u v δ c)
    (fun u v δ => inner_scaledMatmul_right broadcastA broadcastB u 0 v δ c)

/-- `scaledMatmulCLM c` applied to `u` and `v` is `scaledMatmul u v c`. -/
@[simp] theorem scaledMatmulCLM_apply (c : ℝ) (u : Tensor ℝ (batchA ++ [m, n]))
    (v : Tensor ℝ (batchB ++ [n, p])) :
    scaledMatmulCLM broadcastA broadcastB c u v = scaledMatmul broadcastA broadcastB u v c :=
  rfl

end Attention

namespace HasPullbackAt

open Tensor

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] {x : E}
  {batchA batchB batch : Shape} {m n p : Nat}
  {f : E → Tensor ℝ (batchA ++ [m, n])} {g : E → Tensor ℝ (batchB ++ [n, p])}
  {Df : E →L[ℝ] Tensor ℝ (batchA ++ [m, n])} {Dg : E →L[ℝ] Tensor ℝ (batchB ++ [n, p])}
  {bf : Tensor ℝ (batchA ++ [m, n]) → E} {bg : Tensor ℝ (batchB ++ [n, p]) → E}

/-- The scaled matrix product of two maps with a fixed scale `c`, by the product rule, pulled back
by the two components of `scaledMatmulBackward` at the operands `f x` and `g x`. -/
@[fun_prop]
theorem scaledMatmul (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch) (c : ℝ) (hf : HasPullbackAt f Df bf x)
    (hg : HasPullbackAt g Dg bg x) :
    HasPullbackAt (fun y => Attention.scaledMatmul broadcastA broadcastB (f y) (g y) c)
      (((Attention.scaledMatmulCLM broadcastA broadcastB c).flip (g x)).comp Df +
        (Attention.scaledMatmulCLM broadcastA broadcastB c (f x)).comp Dg)
      (fun δ => bf (Attention.scaledMatmulBackward broadcastA broadcastB (f x) (g x) δ c).1 +
        bg (Attention.scaledMatmulBackward broadcastA broadcastB (f x) (g x) δ c).2) x := by
  simpa only [Attention.scaledMatmulCLM_apply] using
    bilin (Attention.scaledMatmulCLM broadcastA broadcastB c)
      (fun v δ => (Attention.scaledMatmulBackward broadcastA broadcastB (f x) v δ c).1)
      (fun u δ => (Attention.scaledMatmulBackward broadcastA broadcastB u (g x) δ c).2)
      (fun u v δ => Attention.inner_scaledMatmul_left broadcastA broadcastB (f x) u v δ c)
      (fun u v δ => Attention.inner_scaledMatmul_right broadcastA broadcastB u (g x) v δ c) hf hg

end HasPullbackAt

namespace Tensor

/-- Mapping `f` over the leading shape `L` is transposed by mapping a transpose `g` of `f` over
`L`. -/
theorem inner_mapLeading {s t : Shape} (L : Shape) {f : Tensor ℝ s → Tensor ℝ t}
    {g : Tensor ℝ t → Tensor ℝ s} (h : ∀ u δ, ⟪f u, δ⟫ = ⟪u, g δ⟫) (x : Tensor ℝ (L ++ s))
    (δ : Tensor ℝ (L ++ t)) : ⟪mapLeading L f x, δ⟫ = ⟪x, mapLeading L g δ⟫ := by
  induction L with
  | scalar => exact h x δ
  | dim k L ih =>
    refine (inner_eq_sum_unstack _ _).trans ((Finset.sum_congr rfl fun i _ => ?_).trans
      (inner_eq_sum_unstack _ _).symm)
    simp only [mapLeading, unstack_dim]
    exact ih (x.unstack i) (δ.unstack i)

end Tensor

namespace Linear

open Tensor

variable {inDim outDim : Nat}

/-- `inputDeriv W` is the transpose of the matrix-vector product with `W` in the vector. -/
theorem inner_matVec_inputDeriv (W : Tensor ℝ [outDim, inDim]) (v : Tensor ℝ [inDim])
    (δ : Tensor ℝ [outDim]) :
    ⟪contract W v _ _ [outDim] .matVec, δ⟫ = ⟪v, inputDeriv W δ⟫ := by
  rw [← dot_eq_inner, ← dot_eq_inner, dot_comm, dot_contract_matVec, dot_comm,
    inputDeriv]

/-- `weightsDeriv v` is the transpose of the matrix-vector product with `v` in the matrix. -/
theorem inner_matVec_weightsDeriv (W : Tensor ℝ [outDim, inDim]) (v : Tensor ℝ [inDim])
    (δ : Tensor ℝ [outDim]) :
    ⟪contract W v _ _ [outDim] .matVec, δ⟫ = ⟪W, weightsDeriv v δ⟫ := by
  rw [← dot_eq_inner, ← dot_eq_inner, dot_vec_eq_sum, dot_mat_eq_sum]
  simp only [getScalar_contract_matVec_eq_sum, weightsDeriv_apply, Finset.sum_mul]
  exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by ring

/-- The matrix-vector product, as a continuous bilinear map in the matrix and the vector. -/
noncomputable def matVecCLM :
    Tensor ℝ [outDim, inDim] →L[ℝ] Tensor ℝ [inDim] →L[ℝ] Tensor ℝ [outDim] :=
  clm₂OfTranspose (fun W v => contract W v _ _ [outDim] .matVec) (fun v δ => weightsDeriv v δ)
    (fun W δ => inputDeriv W δ) inner_matVec_weightsDeriv inner_matVec_inputDeriv

/-- `matVecCLM` applied to `W` and `v` is the matrix-vector product of `W` and `v`. -/
@[simp] theorem matVecCLM_apply (W : Tensor ℝ [outDim, inDim]) (v : Tensor ℝ [inDim]) :
    matVecCLM W v = contract W v _ _ [outDim] .matVec :=
  rfl

/-- Applying `inputDeriv W` to every row of `D` is the matrix product of `D` and `W`. -/
theorem mapLeading_inputDeriv {n : Nat} (W : Tensor ℝ [outDim, inDim])
    (D : Tensor ℝ [n, outDim]) : mapLeading [n] (inputDeriv W) D = matmul D W := by
  refine ext_get2 fun k j => ?_
  rw [get2_matmul_eq_sum, get2_eq_getScalar_unstack, unstack_mapLeading, inputDeriv,
    getScalar_contract_vecMat_eq_sum]
  simp only [get2_eq_getScalar_unstack]

/-- Over a leading shape `L`, `inputDerivLeading L W` is the transpose of the matrix-vector
product with `W` at every leading index, in the input. -/
theorem inner_mapLeading_inputDerivLeading (L : Shape) (W : Tensor ℝ [outDim, inDim])
    (x : Tensor ℝ (L ++ [inDim])) (δ : Tensor ℝ (L ++ [outDim])) :
    ⟪mapLeading L (fun v => contract W v _ _ [outDim] .matVec) x, δ⟫ =
      ⟪x, inputDerivLeading L W δ⟫ :=
  inner_mapLeading L (inner_matVec_inputDeriv W) x δ

/-- Over a leading shape `L`, `weightsDerivLeading L x` is the transpose of the matrix-vector
product with the input `x` at every leading index, in the matrix. -/
theorem inner_mapLeading_weightsDerivLeading (L : Shape) (W : Tensor ℝ [outDim, inDim])
    (x : Tensor ℝ (L ++ [inDim])) (δ : Tensor ℝ (L ++ [outDim])) :
    ⟪mapLeading L (fun v => contract W v _ _ [outDim] .matVec) x, δ⟫ =
      ⟪W, weightsDerivLeading L x δ⟫ := by
  have hx : (L ++ [inDim]).size = Shape.size [L.size, inDim] := by simp [Shape.size_append]
  have hδ : (L ++ [outDim]).size = Shape.size [L.size, outDim] := by simp [Shape.size_append]
  rw [inner_mapLeading_inputDerivLeading, inputDerivLeading,
    mapLeading_eq_reshape L _ δ hδ hx.symm, mapLeading_inputDeriv, real_inner_comm,
    inner_reshape, ← dot_eq_inner, ← dot_eq_inner, dot_matmul_left_adjoint]
  rfl

/-- Over a leading shape `L`, `biasDerivLeading L` is the transpose of mapping the constant
function with value `b` over `L`, as a function of `b`. -/
theorem inner_mapLeading_const_biasDerivLeading (L : Shape) {s : Shape} (z : Tensor ℝ (L ++ s))
    (b : Tensor ℝ [outDim]) (δ : Tensor ℝ (L ++ [outDim])) :
    ⟪mapLeading L (fun _ => b) z, δ⟫ = ⟪b, biasDerivLeading L δ⟫ := by
  have hz : (L ++ s).size = Shape.size (L.size :: s) := by simp [Shape.size_append]
  have hδ : (L ++ [outDim]).size = Shape.size [L.size, outDim] := by simp [Shape.size_append]
  rw [mapLeading_eq_reshape L _ z hz hδ.symm, inner_reshape, inner_eq_sum_unstack]
  simp only [unstack_mapLeading, biasDerivLeading]
  erw [castShape_reduce_sum_zero]
  simp only [inner_eq_sum, Rep.get_ofFn, sum_eq_coord_sum, Tensor.unstack_apply,
    Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun c _ => ?_
  show _ = ∑ i : Fin L.size × PUnit, _
  rw [Fintype.sum_prod_type]
  simp only [ofFn_apply, Finset.univ_unique, Finset.sum_singleton]
  rfl

/-- The matrix-vector product with one matrix at every index of the leading shape `L`, as a
continuous bilinear map in the matrix and the input. -/
noncomputable def matVecLeadingCLM (L : Shape) :
    Tensor ℝ [outDim, inDim] →L[ℝ] Tensor ℝ (L ++ [inDim]) →L[ℝ] Tensor ℝ (L ++ [outDim]) :=
  clm₂OfTranspose (fun W x => mapLeading L (fun v => contract W v _ _ [outDim] .matVec) x)
    (fun x δ => weightsDerivLeading L x δ) (fun W δ => inputDerivLeading L W δ)
    (inner_mapLeading_weightsDerivLeading L) (inner_mapLeading_inputDerivLeading L)

/-- `matVecLeadingCLM L` applied to `W` and `x` is the matrix-vector product with `W` at every
leading index of `x`. -/
@[simp] theorem matVecLeadingCLM_apply (L : Shape) (W : Tensor ℝ [outDim, inDim])
    (x : Tensor ℝ (L ++ [inDim])) :
    matVecLeadingCLM L W x = mapLeading L (fun v => contract W v _ _ [outDim] .matVec) x :=
  rfl

/-- The bias repeated at every index of the leading shape `L`, as a continuous linear map. -/
noncomputable def biasLeadingCLM (L : Shape) :
    Tensor ℝ [outDim] →L[ℝ] Tensor ℝ (L ++ [outDim]) :=
  clmOfTranspose (fun b => mapLeading L (fun _ : Tensor ℝ [] => b) 0) (biasDerivLeading L)
    (inner_mapLeading_const_biasDerivLeading L 0)

/-- `biasLeadingCLM L` applied to `b` is the constant function with value `b` mapped over the
leading shape `L` of the zero tensor. -/
@[simp] theorem biasLeadingCLM_apply (L : Shape) (b : Tensor ℝ [outDim]) :
    biasLeadingCLM L b = mapLeading L (fun _ : Tensor ℝ [] => b) 0 :=
  rfl

/-- The linear layer is the matrix-vector product with the weights plus the bias. -/
theorem forward_eq (W : Tensor ℝ [outDim, inDim]) (b : Tensor ℝ [outDim])
    (v : Tensor ℝ [inDim]) : forward { weights := W, bias := b } v = matVecCLM W v + b := by
  apply Rep.ext
  intro c
  simp only [forward, matVecCLM_apply, Tensor.add_apply]

/-- Over a leading shape `L`, the linear layer is the matrix-vector product with the weights at
every leading index plus the bias repeated at every leading index. -/
theorem forwardLeading_eq (L : Shape) (W : Tensor ℝ [outDim, inDim]) (b : Tensor ℝ [outDim])
    (x : Tensor ℝ (L ++ [inDim])) :
    forwardLeading L { weights := W, bias := b } x =
      matVecLeadingCLM L W x + biasLeadingCLM L b := by
  apply Rep.ext
  intro c
  simp only [forwardLeading, forward, matVecLeadingCLM_apply, biasLeadingCLM_apply,
    mapLeading_apply, Tensor.add_apply]

section Cotangent

variable (W : Tensor ℝ [outDim, inDim]) (v : Tensor ℝ [inDim]) (δ δ' : Tensor ℝ [outDim]) (k : ℝ)

/-- `inputDeriv W` maps a sum of cotangents to the sum of the input gradients. -/
@[op_deriv_simps] theorem inputDeriv_add :
    inputDeriv W (δ + δ') = inputDeriv W δ + inputDeriv W δ' :=
  ext_inner_left ℝ fun u => by simp only [inner_add_right, ← inner_matVec_inputDeriv W]

/-- `inputDeriv W` commutes with multiplying the cotangent by `k`. -/
@[op_deriv_simps] theorem inputDeriv_smul : inputDeriv W (k • δ) = k • inputDeriv W δ :=
  ext_inner_left ℝ fun u => by simp only [real_inner_smul_right, ← inner_matVec_inputDeriv W]

/-- `inputDeriv W` maps a negated cotangent to the negated input gradient. -/
@[op_deriv_simps] theorem inputDeriv_neg : inputDeriv W (-δ) = -inputDeriv W δ :=
  ext_inner_left ℝ fun u => by simp only [inner_neg_right, ← inner_matVec_inputDeriv W]

/-- `inputDeriv W` maps the zero cotangent to zero. -/
@[op_deriv_simps] theorem inputDeriv_zero : inputDeriv W (0 : Tensor ℝ [outDim]) = 0 :=
  ext_inner_left ℝ fun u => by simp only [inner_zero_right, ← inner_matVec_inputDeriv W]

/-- `weightsDeriv v` maps a sum of cotangents to the sum of the weight gradients. -/
@[op_deriv_simps] theorem weightsDeriv_add :
    weightsDeriv v (δ + δ') = weightsDeriv v δ + weightsDeriv v δ' :=
  ext_inner_left ℝ fun U => by simp only [inner_add_right, ← inner_matVec_weightsDeriv U v]

/-- `weightsDeriv v` commutes with multiplying the cotangent by `k`. -/
@[op_deriv_simps] theorem weightsDeriv_smul : weightsDeriv v (k • δ) = k • weightsDeriv v δ :=
  ext_inner_left ℝ fun U => by
    simp only [real_inner_smul_right, ← inner_matVec_weightsDeriv U v]

/-- `weightsDeriv v` maps a negated cotangent to the negated weight gradient. -/
@[op_deriv_simps] theorem weightsDeriv_neg : weightsDeriv v (-δ) = -weightsDeriv v δ :=
  ext_inner_left ℝ fun U => by simp only [inner_neg_right, ← inner_matVec_weightsDeriv U v]

/-- `weightsDeriv v` maps the zero cotangent to zero. -/
@[op_deriv_simps] theorem weightsDeriv_zero : weightsDeriv v (0 : Tensor ℝ [outDim]) = 0 :=
  ext_inner_left ℝ fun U => by simp only [inner_zero_right, ← inner_matVec_weightsDeriv U v]

end Cotangent

end Linear

namespace HasPullbackAt

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] {x : E} {inDim outDim : Nat}
  {fW : E → Tensor ℝ [outDim, inDim]} {fb : E → Tensor ℝ [outDim]}
  {DW : E →L[ℝ] Tensor ℝ [outDim, inDim]} {Db : E →L[ℝ] Tensor ℝ [outDim]}
  {bW : Tensor ℝ [outDim, inDim] → E} {bb : Tensor ℝ [outDim] → E}

/-- The linear layer applied to maps for its weights, bias and input, by the product rule, pulled
back by `weightsDeriv` at the input `fx x`, by `inputDeriv` at the weights `fW x` and by the
identity on the bias. -/
@[fun_prop]
theorem linearForward {fx : E → Tensor ℝ [inDim]} {Dx : E →L[ℝ] Tensor ℝ [inDim]}
    {bx : Tensor ℝ [inDim] → E} (hW : HasPullbackAt fW DW bW x) (hb : HasPullbackAt fb Db bb x)
    (hx : HasPullbackAt fx Dx bx x) :
    HasPullbackAt (fun y => Linear.forward { weights := fW y, bias := fb y } (fx y))
      ((Linear.matVecCLM.flip (fx x)).comp DW + (Linear.matVecCLM (fW x)).comp Dx + Db)
      (fun δ => bW (Linear.weightsDeriv (fx x) δ) + bx (Linear.inputDeriv (fW x) δ) + bb δ) x :=
  ((bilin Linear.matVecCLM (fun v δ => Linear.weightsDeriv v δ)
    (fun W δ => Linear.inputDeriv W δ) Linear.inner_matVec_weightsDeriv
    Linear.inner_matVec_inputDeriv hW hx).add hb).congr_of_eventuallyEq
    (Filter.Eventually.of_forall fun y => Linear.forward_eq (fW y) (fb y) (fx y))

/-- The linear layer over a leading shape `L` applied to maps for its weights, bias and input, by
the product rule, pulled back by `weightsDerivLeading` at the input `fx x`, by `inputDerivLeading`
at the weights `fW x` and by `biasDerivLeading` on the bias. -/
@[fun_prop]
theorem linearForwardLeading (L : Shape) {fx : E → Tensor ℝ (L ++ [inDim])}
    {Dx : E →L[ℝ] Tensor ℝ (L ++ [inDim])} {bx : Tensor ℝ (L ++ [inDim]) → E}
    (hW : HasPullbackAt fW DW bW x) (hb : HasPullbackAt fb Db bb x)
    (hx : HasPullbackAt fx Dx bx x) :
    HasPullbackAt (fun y => Linear.forwardLeading L { weights := fW y, bias := fb y } (fx y))
      (((Linear.matVecLeadingCLM L).flip (fx x)).comp DW +
        (Linear.matVecLeadingCLM L (fW x)).comp Dx + (Linear.biasLeadingCLM L).comp Db)
      (fun δ => bW (Linear.weightsDerivLeading L (fx x) δ) +
        bx (Linear.inputDerivLeading L (fW x) δ) + bb (Linear.biasDerivLeading L δ)) x :=
  ((bilin (Linear.matVecLeadingCLM L) (fun v δ => Linear.weightsDerivLeading L v δ)
    (fun W δ => Linear.inputDerivLeading L W δ) (Linear.inner_mapLeading_weightsDerivLeading L)
    (Linear.inner_mapLeading_inputDerivLeading L) hW hx).add
    (clm (Linear.biasLeadingCLM L) (Linear.biasDerivLeading L)
      (Linear.inner_mapLeading_const_biasDerivLeading L 0) hb)).congr_of_eventuallyEq
    (Filter.Eventually.of_forall fun y => Linear.forwardLeading_eq L (fW y) (fb y) (fx y))

end HasPullbackAt

namespace Op

/-- The forward map of `Op.matmul` is differentiable in both operands jointly, and its stored
pullback is the adjoint of that derivative, for vector or matrix operands and broadcast batch
prefixes. -/
@[op_cert TorchLean.Op.matmul derivative 0 1]
theorem matmul_prod_backward_eq_adjoint {batchA batchB batch : Shape} {m n p : Nat}
    (lv rv : Bool) (hl : lv = true → m = 1) (hr : rv = true → p = 1)
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch)
    (a : Tensor ℝ (Shape.matmulLeft lv batchA m n)) (b : Tensor ℝ (Shape.matmulRight rv batchB n p))
    (δ : Tensor ℝ (Shape.matmulOut lv rv batch m p)) :
    DifferentiableAt ℝ (fun q : WithLp 2 (Tensor ℝ (Shape.matmulLeft lv batchA m n) ×
          Tensor ℝ (Shape.matmulRight rv batchB n p)) =>
        (((Op.matmul lv rv hl hr broadcastA broadcastB).rule? (α := ℝ) |>.get rfl).forward
          (q.fst, q.snd, ())).1)
      (WithLp.toLp 2 (a, b)) ∧
    WithLp.toLp 2
        ((((Op.matmul lv rv hl hr broadcastA broadcastB).rule? (α := ℝ) |>.get rfl).backward
          (a, b, ()) (δ, ())).1,
        (((Op.matmul lv rv hl hr broadcastA broadcastB).rule? (α := ℝ) |>.get rfl).backward
          (a, b, ()) (δ, ())).2.1) =
      (fderiv ℝ (fun q : WithLp 2 (Tensor ℝ (Shape.matmulLeft lv batchA m n) ×
          Tensor ℝ (Shape.matmulRight rv batchB n p)) =>
        (((Op.matmul lv rv hl hr broadcastA broadcastB).rule? (α := ℝ) |>.get rfl).forward
          (q.fst, q.snd, ())).1)
        (WithLp.toLp 2 (a, b))).adjoint δ := by
  simp only [Op.rule?_matmul, Option.get_some]
  apply HasPullbackAt.differentiableAt_and
  · fun_prop
  · refine withLp_prod_ext ?_ ?_ <;> simp

/-- The forward map of `Op.scaledMatmul` with a frozen scale `s` is differentiable in both operands
jointly, and its stored pullback is the adjoint of that derivative, for vector or matrix operands
and broadcast batch prefixes. -/
@[op_cert TorchLean.Op.scaledMatmul derivative 0 1]
theorem scaledMatmul_prod_backward_eq_adjoint {batchA batchB batch : Shape} {m n p : Nat}
    (lv rv : Bool) (hl : lv = true → m = 1) (hr : rv = true → p = 1)
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch)
    (a : Tensor ℝ (Shape.matmulLeft lv batchA m n)) (b : Tensor ℝ (Shape.matmulRight rv batchB n p))
    (s : Tensor ℝ []) (δ : Tensor ℝ (Shape.matmulOut lv rv batch m p)) :
    DifferentiableAt ℝ (fun q : WithLp 2 (Tensor ℝ (Shape.matmulLeft lv batchA m n) ×
          Tensor ℝ (Shape.matmulRight rv batchB n p)) =>
        (((Op.scaledMatmul lv rv hl hr broadcastA broadcastB).rule? (α := ℝ) |>.get rfl).forward
          (q.fst, q.snd, s, ())).1)
      (WithLp.toLp 2 (a, b)) ∧
    WithLp.toLp 2
        ((((Op.scaledMatmul lv rv hl hr broadcastA broadcastB).rule? (α := ℝ) |>.get rfl).backward
          (a, b, s, ()) (δ, ())).1,
        (((Op.scaledMatmul lv rv hl hr broadcastA broadcastB).rule? (α := ℝ) |>.get rfl).backward
          (a, b, s, ()) (δ, ())).2.1) =
      (fderiv ℝ (fun q : WithLp 2 (Tensor ℝ (Shape.matmulLeft lv batchA m n) ×
          Tensor ℝ (Shape.matmulRight rv batchB n p)) =>
        (((Op.scaledMatmul lv rv hl hr broadcastA broadcastB).rule? (α := ℝ) |>.get rfl).forward
          (q.fst, q.snd, s, ())).1)
        (WithLp.toLp 2 (a, b))).adjoint δ := by
  simp only [Op.rule?_scaledMatmul, Option.get_some]
  apply HasPullbackAt.differentiableAt_and
  · fun_prop
  · refine withLp_prod_ext ?_ ?_ <;> simp

/-- The forward map of `Op.linear` is differentiable in the weights, the bias and the input
jointly, and its stored pullback is the adjoint of that derivative, at every leading shape
`lead`. -/
@[op_cert TorchLean.Op.linear derivative 0 1 2]
theorem linear_prod_backward_eq_adjoint {lead : Shape} {inDim outDim : Nat}
    (W : Tensor ℝ [outDim, inDim]) (b : Tensor ℝ [outDim]) (x : Tensor ℝ (lead ++ [inDim]))
    (δ : Tensor ℝ (lead ++ [outDim])) :
    DifferentiableAt ℝ (fun q : WithLp 2 (Tensor ℝ [outDim, inDim] ×
          WithLp 2 (Tensor ℝ [outDim] × Tensor ℝ (lead ++ [inDim]))) =>
        (((Op.linear (lead := lead) (inDim := inDim) (outDim := outDim)).rule? (α := ℝ) |>.get
          rule?_linear_isSome).forward (q.fst, q.snd.fst, q.snd.snd, ())).1)
      (WithLp.toLp 2 (W, WithLp.toLp 2 (b, x))) ∧
    WithLp.toLp 2
        ((((Op.linear (lead := lead) (inDim := inDim) (outDim := outDim)).rule? (α := ℝ) |>.get
          rule?_linear_isSome).backward (W, b, x, ()) (δ, ())).1,
        WithLp.toLp 2
          ((((Op.linear (lead := lead) (inDim := inDim) (outDim := outDim)).rule? (α := ℝ) |>.get
            rule?_linear_isSome).backward (W, b, x, ()) (δ, ())).2.1,
          (((Op.linear (lead := lead) (inDim := inDim) (outDim := outDim)).rule? (α := ℝ) |>.get
            rule?_linear_isSome).backward (W, b, x, ()) (δ, ())).2.2.1)) =
      (fderiv ℝ (fun q : WithLp 2 (Tensor ℝ [outDim, inDim] ×
          WithLp 2 (Tensor ℝ [outDim] × Tensor ℝ (lead ++ [inDim]))) =>
        (((Op.linear (lead := lead) (inDim := inDim) (outDim := outDim)).rule? (α := ℝ) |>.get
          rule?_linear_isSome).forward (q.fst, q.snd.fst, q.snd.snd, ())).1)
        (WithLp.toLp 2 (W, WithLp.toLp 2 (b, x)))).adjoint δ := by
  cases lead with
  | scalar =>
    simp only [Op.rule?_linear, Option.get_some]
    apply HasPullbackAt.differentiableAt_and
    · fun_prop
    · refine withLp_prod_ext ?_ (withLp_prod_ext ?_ ?_) <;> simp
  | dim n rest =>
    simp only [Op.rule?_linear_leading, Option.get_some]
    apply HasPullbackAt.differentiableAt_and
    · fun_prop
    · refine withLp_prod_ext ?_ (withLp_prod_ext ?_ ?_) <;> simp

/-- The forward map of `Op.contract` is differentiable in both operands jointly, and its stored
pullback is the adjoint of that derivative. -/
@[op_cert TorchLean.Op.contract derivative 0 1]
theorem contract_prod_backward_eq_adjoint {s t : Shape}
    (dims batchDims : List (Fin s.rank × Fin t.rank))
    (h : Shape.ContractValid s t dims batchDims) (x : Tensor ℝ s) (y : Tensor ℝ t)
    (δ : Tensor ℝ (s.contract t dims batchDims)) :
    DifferentiableAt ℝ (fun q : WithLp 2 (Tensor ℝ s × Tensor ℝ t) =>
        (((Op.contract dims batchDims h).rule? (α := ℝ) |>.get rfl).forward
          (q.fst, q.snd, ())).1) (WithLp.toLp 2 (x, y)) ∧
    WithLp.toLp 2
        ((((Op.contract dims batchDims h).rule? (α := ℝ) |>.get rfl).backward
          (x, y, ()) (δ, ())).1,
        (((Op.contract dims batchDims h).rule? (α := ℝ) |>.get rfl).backward
          (x, y, ()) (δ, ())).2.1) =
      (fderiv ℝ (fun q : WithLp 2 (Tensor ℝ s × Tensor ℝ t) =>
        (((Op.contract dims batchDims h).rule? (α := ℝ) |>.get rfl).forward
          (q.fst, q.snd, ())).1) (WithLp.toLp 2 (x, y))).adjoint δ := by
  simp only [Op.rule?_contract, Option.get_some]
  apply HasPullbackAt.differentiableAt_and
  · fun_prop
  · refine withLp_prod_ext ?_ ?_ <;> simp

end Op

end TorchLean
