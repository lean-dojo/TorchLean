/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Op.Cert
public import TorchLean.NN.Op.Arms
public import TorchLean.NN.AdjointOp
public import TorchLean.NN.Softmax.Correctness
public import TorchLean.Tensor.Elementwise.Real
public import TorchLean.NN.FDeriv.Reindex
public import TorchLean.NN.FDeriv.TensorCoordinates
public import TorchLean.Tensor.Reduce.Real

/-!
# Registered certificates of the operator table

Correctness certificates of operators in `TorchLean.Op`, stated about their arms and registered
with `op_cert`. A record of the adjointness of a selected tangent map
(`TorchLean.Tape.AdjointOp ℝ`, `algebraic`) is tied to the arm by a theorem saying
that the record's operator has the arm's forward value and pullback, at the shapes the record has.
The linear record fixes the weight and bias and differentiates the input (input `2`). Records are
registered at `ℝ` only: the arm reads its arithmetic from `Context`, which a commutative semiring
does not provide. The derivative certificates of the pointwise and binary
operators, at every shape, are in `TorchLean.NN.Op.Pointwise`. The joint derivative of
`batchNormRunning` in its input, scale and bias is in `TorchLean.NN.Op.BatchNorm`, the joint
derivative of `layerNorm` in `TorchLean.NN.Op.LayerNorm`, and the derivatives of `softmax`
and `logSoftmax` at every shape and axis in `TorchLean.NN.Op.Softmax`.

The derivatives of `pad` and `contract` are stated on the real tensors themselves, with their
Euclidean inner product, at every shape and attribute. `pad` is affine in its input, with the
adjoint of reading back at `padPosition` as derivative, and affine in its padding value, with the
padding mask as derivative. `contract` is linear in each operand. Its pullbacks are the adjoints,
through the bijections that pair an output coordinate and a contracted coordinate with a
coordinate of one operand and a free coordinate of the other.
-/

@[expose] public section

open TorchLean.Shape (rank_eq_length)

namespace TorchLean

open TorchLean.Tensor AdjointOp Tape
open Op (reduce_sum)

/-! ## Records over the reals -/

/-- The operator of the `relu` record has the forward value and pullback of the `Op.relu` arm. -/
@[op_cert TorchLean.Op.relu algebraic]
theorem AdjointOp.relu_op {s : Shape} :
    let rule := (Op.relu (s := s)).rule? (α := ℝ) |>.get rfl
    (AdjointOp.relu (α := ℝ) (s := s)).op =
      ⟨fun x => (rule.forward (x, ())).1, fun x g => (rule.backward (x, ()) (g, ())).1⟩ := rfl

/-- The operator of the `sigmoid` record has the forward value and pullback of the
`Op.sigmoid` arm. -/
@[op_cert TorchLean.Op.sigmoid algebraic]
theorem AdjointOp.sigmoid_op {s : Shape} :
    let rule := (Op.sigmoid (s := s)).rule? (α := ℝ) |>.get rfl
    (sigmoid (s := s)).op =
      ⟨fun x => (rule.forward (x, ())).1, fun x g => (rule.backward (x, ()) (g, ())).1⟩ := rfl

/-- The operator of the `tanh` record has the forward value and pullback of the `Op.tanh` arm. -/
@[op_cert TorchLean.Op.tanh algebraic]
theorem AdjointOp.tanh_op {s : Shape} :
    let rule := (Op.tanh (s := s)).rule? (α := ℝ) |>.get rfl
    (tanh (s := s)).op =
      ⟨fun x => (rule.forward (x, ())).1, fun x g => (rule.backward (x, ()) (g, ())).1⟩ := rfl

/-- The operator of the `exp` record has the forward value and pullback of the `Op.exp` arm. -/
@[op_cert TorchLean.Op.exp algebraic]
theorem AdjointOp.exp_op {s : Shape} :
    let rule := (Op.exp (s := s)).rule? (α := ℝ) |>.get rfl
    (exp (s := s)).op =
      ⟨fun x => (rule.forward (x, ())).1, fun x g => (rule.backward (x, ()) (g, ())).1⟩ := rfl

/-- The record of the total sum is the arm of `Op.reduce` summing over every axis, read through
the cast of `Shape.reduce_finRange`. -/
@[op_cert TorchLean.Op.reduce algebraic]
theorem AdjointOp.sum_op {s : Shape} :
    let rule := (Op.reduce (s := s) .sum (List.finRange s.rank) false
      (fun h => absurd h (by decide))).rule? (α := ℝ) |>.get rfl
    (sum (s := s)).op =
      ⟨fun x => castShape (rule.forward (x, ())).1 (Shape.reduce_finRange s),
        fun x g => (rule.backward (x, ()) (castShape g (Shape.reduce_finRange s).symm, ())).1⟩ := by
  simp only [sum]
  congr 1
  · funext x
    exact ((reduce_sum x).1).symm
  · funext x g
    exact (((reduce_sum x).2 g)).symm

/-- The operator of the `softmax n` record has the forward value and pullback of the `Op.softmax`
arm on axis `0` of `[n]`. -/
@[op_cert TorchLean.Op.softmax algebraic]
theorem AdjointOp.softmax_op (n : Nat) :
    let rule := (Op.softmax (s := [n]) ⟨0, by simp⟩).rule? (α := ℝ) |>.get rfl
    (softmax n).op =
      ⟨fun x => (rule.forward (x, ())).1, fun x g => (rule.backward (x, ()) (g, ())).1⟩ := rfl

/-- The operator of the `logSoftmax n` record has the forward value and pullback of the
`Op.logSoftmax` arm on axis `0` of `[n]`. -/
@[op_cert TorchLean.Op.logSoftmax algebraic]
theorem AdjointOp.logSoftmax_op (n : Nat) :
    let rule := (Op.logSoftmax (s := [n]) ⟨0, by simp⟩).rule? (α := ℝ) |>.get rfl
    (logSoftmax n).op =
      ⟨fun x => (rule.forward (x, ())).1, fun x g => (rule.backward (x, ()) (g, ())).1⟩ := rfl

/-- The linear record's operator is the arm at the empty leading shape with the weight and bias
fixed, on the input. -/
@[op_cert TorchLean.Op.linear algebraic 2]
theorem AdjointOp.linear_op {inDim outDim : Nat} (m : Linear ℝ inDim outDim)
    (x : Tensor ℝ [inDim]) (g : Tensor ℝ [outDim]) :
    (AdjointOp.linear m).op.forward x =
        (((Op.linear (lead := []) (inDim := inDim) (outDim := outDim)).rule? (α := ℝ)
          |>.get rfl).forward
          (m.weights, m.bias, x, ())).1 ∧
      (AdjointOp.linear m).op.backward x g =
        (((Op.linear (lead := []) (inDim := inDim) (outDim := outDim)).rule? (α := ℝ)
          |>.get rfl).backward
          (m.weights, m.bias, x, ()) (g, ())).2.2.1 := ⟨rfl, rfl⟩

/-! ## Conventions -/

/-- The tangent of `detach` is zero by AD convention, although its forward value is the identity
on the reals. The detached input receives no tensor cotangent or tape contribution. -/
@[op_cert TorchLean.Op.detach convention]
theorem Op.detach_jvp_zero {s : Shape} :
    ((Op.detach (s := s)).rule? (α := ℝ) |>.get rfl).jvp? = some fun _ _ => (Tensor.full s 0, ()) :=
  rfl

end TorchLean

/-! ## Padding and contraction -/

namespace TorchLean.Tensor

variable {α : Type} [Storage α] {s : Shape}

/-- `padPosition` is injective: distinct input coordinates land at distinct output coordinates. -/
theorem padPosition_injective (low high interior : Fin s.rank → Nat) :
    Function.Injective (padPosition (s := s) low high interior) := by
  intro c c' h
  apply Shape.Coord.ext_val
  intro d
  have hd := congrArg
    (fun c => (Shape.Coord.get c (d.cast (s.rank_pad low high interior).symm)).val) h
  simp only [padPosition, Shape.Coord.get_ofFun, Fin.cast_cast, Fin.cast_eq_self] at hd
  exact Nat.eq_of_mul_eq_mul_left (Nat.succ_pos _) (Nat.add_left_cancel hd)

/-- An output coordinate of `pad` that is not a `padPosition` holds the padding value. -/
theorem pad_apply_of_forall_ne (x : Tensor α s) (low high interior : Fin s.rank → Nat) (value : α)
    (c : (s.pad low high interior).Coord)
    (hc : ∀ c' : s.Coord, padPosition low high interior c' ≠ c) :
    pad x low high interior value c = value := by
  rw [pad_apply]
  split
  · rename_i hcond
    exfalso
    apply hc (Shape.Coord.ofFun fun d =>
      ⟨((Shape.Coord.get c (d.cast (s.rank_pad low high interior).symm)).val - low d) /
        (interior d + 1), (hcond d).2.2⟩)
    apply Shape.Coord.ext_val
    intro d
    simp only [padPosition, Shape.Coord.get_ofFun, Fin.cast_cast, Fin.cast_eq_self]
    have h1 := (hcond (d.cast (s.rank_pad low high interior))).1
    have h2 := (hcond (d.cast (s.rank_pad low high interior))).2.1
    simp only [Fin.cast_cast, Fin.cast_eq_self] at h1 h2
    rw [Nat.mul_div_cancel' (Nat.dvd_of_mod_eq_zero h2)]
    omega
  · rfl

end TorchLean.Tensor

namespace TorchLean.Op

open Tensor

section Pad

variable {s : Shape} (low high interior : Fin s.rank → Nat)

/-- Over the reals, padding with zero adds each input entry at its `padPosition`: the sum over the
fibers of `padPosition`, the adjoint of reading back through it. -/
theorem pad_zero_eq_sumFibers (y : Tensor ℝ s) :
    Tensor.pad y low high interior 0 =
      Rep.sumFibers (Tensor.padPosition low high interior) y := by
  classical
  apply Tensor.Rep.ext
  intro c
  simp only [Rep.sumFibers, Tensor.Rep.get_ofFn]
  by_cases hc : ∃ c', Tensor.padPosition low high interior c' = c
  · obtain ⟨c', rfl⟩ := hc
    rw [Tensor.pad_padPosition]
    simp only [(Tensor.padPosition_injective low high interior).eq_iff]
    simp
  · push Not at hc
    rw [Tensor.pad_apply_of_forall_ne _ _ _ _ _ _ hc]
    exact (Finset.sum_eq_zero fun c' _ => by simp [hc c']).symm

/-- Padding splits into the padded input with zero padding and the padding of zero. -/
theorem pad_eq_add (y : Tensor ℝ s) (value : ℝ) :
    Tensor.pad y low high interior value =
      Tensor.pad y low high interior 0 + Tensor.pad (0 : Tensor ℝ s) low high interior value := by
  apply Tensor.Rep.ext
  intro c
  simp only [Tensor.add_apply, Tensor.pad_apply]
  split <;> simp

/-- The padding of zero with `value` is `value` times the padding of zero with `1`. -/
theorem pad_zero_eq_smul (value : ℝ) :
    Tensor.pad (0 : Tensor ℝ s) low high interior value =
      value • Tensor.pad (0 : Tensor ℝ s) low high interior 1 := by
  apply Tensor.Rep.ext
  intro c
  simp only [Tensor.Rep.smul_apply, Tensor.pad_apply]
  split <;> simp

/-- With the padding value fixed, the forward of `Op.pad` is affine in its input, with derivative
the adjoint of reading back through `padPosition`. -/
theorem pad_hasFDerivAt (value : Tensor ℝ []) (x : Tensor ℝ s) :
    HasFDerivAt (fun y => (((Op.pad (s := s) low high interior).rule? (α := ℝ) |>.get rfl).forward
        (y, value, ())).1) (Rep.pullCLM (Tensor.padPosition low high interior)).adjoint x := by
  have heq : (fun y => (((Op.pad (s := s) low high interior).rule? (α := ℝ) |>.get rfl).forward
      (y, value, ())).1) =
      fun y => (Rep.pullCLM (Tensor.padPosition low high interior)).adjoint y +
        Tensor.pad (0 : Tensor ℝ s) low high interior value.item := by
    funext y
    rw [Rep.pullCLM_adjoint, ← pad_zero_eq_sumFibers]
    exact pad_eq_add low high interior y value.item
  rw [heq]
  exact ((Rep.pullCLM _).adjoint.hasFDerivAt).add_const _

/-- The pullback of `pad` to its input, the entries at the `padPosition`s, is the adjoint of the
derivative, the padding value fixed. -/
@[op_cert TorchLean.Op.pad derivative 0]
theorem pad_backward_eq_adjoint (value : Tensor ℝ []) (x : Tensor ℝ s)
    (δ : Tensor ℝ (s.pad low high interior)) :
    (((Op.pad (s := s) low high interior).rule? (α := ℝ) |>.get rfl).backward
        (x, value, ()) (δ, ())).1 =
      (fderiv ℝ (fun y => (((Op.pad (s := s) low high interior).rule? (α := ℝ) |>.get rfl).forward
        (y, value, ())).1) x).adjoint δ := by
  rw [(pad_hasFDerivAt low high interior value x).fderiv, ContinuousLinearMap.adjoint_adjoint]
  apply Tensor.Rep.ext
  intro c
  simp only [Rep.pullCLM_apply, Tensor.Rep.pull_apply]
  exact Tensor.padBackward_apply low high interior δ c

/-- The tangent map of `pad` in its padding value: the value times the padding mask. -/
noncomputable def padValueCLM : Tensor ℝ [] →L[ℝ] Tensor ℝ (s.pad low high interior) :=
  (Tensor.coordinateCLM PUnit.unit).smulRight
    (Tensor.pad (0 : Tensor ℝ s) low high interior 1)

/-- With the input fixed, the forward of `Op.pad` is affine in its padding value, with derivative
`padValueCLM`. -/
theorem pad_value_hasFDerivAt (x : Tensor ℝ s) (value : Tensor ℝ []) :
    HasFDerivAt (fun v => (((Op.pad (s := s) low high interior).rule? (α := ℝ) |>.get rfl).forward
        (x, v, ())).1) (padValueCLM low high interior) value := by
  have heq : (fun v => (((Op.pad (s := s) low high interior).rule? (α := ℝ) |>.get rfl).forward
      (x, v, ())).1) =
      fun v => Tensor.pad x low high interior 0 + padValueCLM low high interior v := by
    funext v
    show Tensor.pad x low high interior v.item = _
    rw [pad_eq_add, pad_zero_eq_smul]
    rfl
  rw [heq]
  exact (padValueCLM low high interior).hasFDerivAt.const_add _

/-- The pullback of `pad` to its padding value, the sum of the cotangent over the padding
positions, is the adjoint of the derivative, the input fixed. -/
@[op_cert TorchLean.Op.pad derivative 1]
theorem pad_value_backward_eq_adjoint (x : Tensor ℝ s) (value : Tensor ℝ [])
    (δ : Tensor ℝ (s.pad low high interior)) :
    (((Op.pad (s := s) low high interior).rule? (α := ℝ) |>.get rfl).backward
        (x, value, ()) (δ, ())).2.1 =
      (fderiv ℝ (fun v => (((Op.pad (s := s) low high interior).rule? (α := ℝ) |>.get rfl).forward
        (x, v, ())).1) value).adjoint δ := by
  rw [(pad_value_hasFDerivAt low high interior x value).fderiv]
  apply ext_inner_left ℝ
  intro u
  rw [ContinuousLinearMap.adjoint_inner_right]
  show inner ℝ u (Tensor.scalar (Tensor.sum (Tensor.select
    (Tensor.pad (Tensor.full s false) low high interior true) δ (Tensor.full _ 0)))) = _
  simp only [Tensor.inner_eq_sum, padValueCLM, ContinuousLinearMap.smulRight_apply,
    Tensor.coordinateCLM_apply, Tensor.Rep.smul_apply, smul_eq_mul,
    Fintype.sum_unique]
  show _ * (Tensor.scalar _ : Tensor ℝ []) PUnit.unit = _
  rw [Tensor.scalar_apply, Tensor.sum_eq_coord_sum, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro c _
  rw [Tensor.select_apply, Tensor.full_apply, Tensor.pad_apply, Tensor.pad_apply]
  split <;> simp

end Pad

end TorchLean.Op

namespace TorchLean.Shape

variable {s t : Shape} {dims batchDims : List (Fin s.rank × Fin t.rank)}

/-- The index of `c.restrict l` at the position of a listed axis `a` is the index of `c` along
`a`. -/
theorem Coord.index_restrict_idxOf {s : Shape} (c : s.Coord) {l : List (Fin s.rank)}
    {a : Fin s.rank} (ha : a ∈ l) : (c.restrict l).index (l.idxOf a) = (c.get a).val := by
  rw [Coord.index_restrict c l _ (List.idxOf_lt_length_iff.mpr ha), List.getElem_idxOf]

/-- `contractLeft` reads `c` back at the output coordinate `contractOutputLeft c j` and the
contracted indices of `c`. -/
theorem contractLeft_contractOutputLeft (c : s.Coord)
    (j : Coord (t.eraseAxes ((batchDims ++ dims).map Prod.snd))) :
    s.contractLeft t dims batchDims (s.contractOutputLeft t dims batchDims c j)
      (c.restrict (dims.map Prod.fst)) = c := by
  apply Coord.ext_val
  intro a
  simp only [contractLeft, contractOutputLeft, Equiv.apply_symm_apply, Coord.get_ofFun]
  split_ifs with ha hb
  · exact Coord.index_restrict_idxOf c ha
  · exact Coord.index_restrict_idxOf c hb
  · exact Coord.index_restrict_idxOf c (mem_otherAxes.mpr (by
      simp only [List.map_append, List.mem_append, not_or]
      exact ⟨hb, ha⟩))

/-- The contracted indices of `contractLeft o r` are `r`. -/
theorem restrict_contractLeft (h : ContractValid s t dims batchDims)
    (o : (s.contract t dims batchDims).Coord) (r : Coord ((dims.map Prod.fst).map s.extent)) :
    (s.contractLeft t dims batchDims o r).restrict (dims.map Prod.fst) = r := by
  have hnd : (dims.map Prod.fst).Nodup := by
    have := h.nodup_left
    rw [List.map_append] at this
    exact this.of_append_right
  apply Coord.ext_index
  intro k hk
  have hk' : k < (dims.map Prod.fst).length := by simpa [Shape.rank_eq_length] using hk
  rw [Coord.index_restrict _ _ k hk']
  simp only [contractLeft, Coord.get_ofFun, dite_eq_left (List.getElem_mem hk')]
  rw [hnd.idxOf_getElem]

/-- `contractOutputLeft` recovers `o` from the coordinate `contractLeft o r` and the part of `o`
for the other axes of `t`. -/
theorem contractOutputLeft_contractLeft (h : ContractValid s t dims batchDims)
    (o : (s.contract t dims batchDims).Coord) (r : Coord ((dims.map Prod.fst).map s.extent)) :
    s.contractOutputLeft t dims batchDims (s.contractLeft t dims batchDims o r)
      (Coord.appendEquiv
        ((batchDims.map Prod.fst).map s.extent ++ s.eraseAxes ((batchDims ++ dims).map Prod.fst))
        (t.eraseAxes ((batchDims ++ dims).map Prod.snd)) o).2 = o := by
  have hnd := h.nodup_left
  rw [List.map_append] at hnd
  have hL := hnd.of_append_left
  have hdisj := List.disjoint_of_nodup_append hnd
  refine (Equiv.symm_apply_eq _).mpr (Prod.ext ((Equiv.symm_apply_eq _).mpr (Prod.ext ?_ ?_)) rfl)
  · apply Coord.ext_index
    intro k hk
    have hk' : k < (batchDims.map Prod.fst).length := by simpa [Shape.rank_eq_length] using hk
    have hmem := List.getElem_mem hk'
    have hnot : (batchDims.map Prod.fst)[k] ∉ dims.map Prod.fst := fun h' => hdisj hmem h'
    refine (Coord.index_restrict _ _ k hk').trans ?_
    simp only [contractLeft, Coord.get_ofFun, dite_eq_right hnot, dite_eq_left hmem]
    rw [hL.idxOf_getElem]
    rfl
  · apply Coord.ext_index
    intro k hk
    have hk' : k < (s.otherAxes ((batchDims ++ dims).map Prod.fst)).length := by
      simpa [eraseAxes, Shape.rank_eq_length] using hk
    have hmem := List.getElem_mem hk'
    have hout := mem_otherAxes.mp hmem
    have hb : (s.otherAxes ((batchDims ++ dims).map Prod.fst))[k] ∉ batchDims.map Prod.fst :=
      fun h' => by
        obtain ⟨p, hp, hpa⟩ := List.mem_map.mp h'
        exact hout (List.mem_map.mpr ⟨p, List.mem_append_left _ hp, hpa⟩)
    have hd : (s.otherAxes ((batchDims ++ dims).map Prod.fst))[k] ∉ dims.map Prod.fst :=
      fun h' => by
        obtain ⟨p, hp, hpa⟩ := List.mem_map.mp h'
        exact hout (List.mem_map.mpr ⟨p, List.mem_append_right _ hp, hpa⟩)
    refine (Coord.index_restrict _ _ k hk').trans ?_
    simp only [contractLeft, Coord.get_ofFun, dite_eq_right hb, dite_eq_right hd]
    rw [(nodup_otherAxes _ _).idxOf_getElem]
    rfl

/-- `contractRight` reads `d` back at the output coordinate `contractOutputRight d i` and the
contracted indices of `d`. -/
theorem ContractValid.contractRight_contractOutputRight (h : ContractValid s t dims batchDims)
    (d : t.Coord) (i : Coord (s.eraseAxes ((batchDims ++ dims).map Prod.fst))) :
    h.contractRight (h.contractOutputRight d i)
      (Coord.reindexEq h.dims_extents (d.restrict (dims.map Prod.snd))) = d := by
  apply Coord.ext_val
  intro b
  simp only [ContractValid.contractRight, ContractValid.contractOutputRight,
    Equiv.apply_symm_apply, Coord.get_ofFun]
  split_ifs with hb hc
  · exact (Coord.index_reindexEq _ _ _).trans (Coord.index_restrict_idxOf d hb)
  · exact (Coord.index_reindexEq _ _ _).trans (Coord.index_restrict_idxOf d hc)
  · exact Coord.index_restrict_idxOf d (mem_otherAxes.mpr (by
      simp only [List.map_append, List.mem_append, not_or]
      exact ⟨hc, hb⟩))

/-- The contracted indices of `contractRight o r`, read as a coordinate of the contracted axes
of `s`, are `r`. -/
theorem ContractValid.reindexEq_restrict_contractRight (h : ContractValid s t dims batchDims)
    (o : (s.contract t dims batchDims).Coord) (r : Coord ((dims.map Prod.fst).map s.extent)) :
    Coord.reindexEq h.dims_extents ((h.contractRight o r).restrict (dims.map Prod.snd)) = r := by
  have hnd : (dims.map Prod.snd).Nodup := by
    have := h.nodup_right
    rw [List.map_append] at this
    exact this.of_append_right
  apply Coord.ext_index
  intro k hk
  have hk' : k < (dims.map Prod.snd).length := by simpa [Shape.rank_eq_length] using hk
  rw [Coord.index_reindexEq]
  refine (Coord.index_restrict _ _ k hk').trans ?_
  simp only [ContractValid.contractRight, Coord.get_ofFun, dite_eq_left (List.getElem_mem hk')]
  rw [hnd.idxOf_getElem]

/-- `contractOutputRight` recovers `o` from the coordinate `contractRight o r` and the part of `o`
for the other axes of `s`. -/
theorem ContractValid.contractOutputRight_contractRight (h : ContractValid s t dims batchDims)
    (o : (s.contract t dims batchDims).Coord) (r : Coord ((dims.map Prod.fst).map s.extent)) :
    h.contractOutputRight (h.contractRight o r)
      (Coord.appendEquiv ((batchDims.map Prod.fst).map s.extent)
        (s.eraseAxes ((batchDims ++ dims).map Prod.fst))
        (Coord.appendEquiv
          ((batchDims.map Prod.fst).map s.extent ++ s.eraseAxes ((batchDims ++ dims).map Prod.fst))
          (t.eraseAxes ((batchDims ++ dims).map Prod.snd)) o).1).2 = o := by
  have hnd := h.nodup_right
  rw [List.map_append] at hnd
  have hL := hnd.of_append_left
  have hdisj := List.disjoint_of_nodup_append hnd
  refine (Equiv.symm_apply_eq _).mpr
    (Prod.ext ((Equiv.symm_apply_eq _).mpr (Prod.ext ?_ rfl)) ?_)
  · apply Coord.ext_index
    intro k hk
    have hk' : k < (batchDims.map Prod.snd).length := by simpa [Shape.rank_eq_length] using hk
    have hmem := List.getElem_mem hk'
    have hnot : (batchDims.map Prod.snd)[k] ∉ dims.map Prod.snd := fun h' => hdisj hmem h'
    rw [Coord.index_reindexEq]
    refine (Coord.index_restrict _ _ k hk').trans ?_
    simp only [ContractValid.contractRight, Coord.get_ofFun, dite_eq_right hnot,
      dite_eq_left hmem]
    rw [hL.idxOf_getElem]
    rfl
  · apply Coord.ext_index
    intro k hk
    have hk' : k < (t.otherAxes ((batchDims ++ dims).map Prod.snd)).length := by
      simpa [eraseAxes, Shape.rank_eq_length] using hk
    have hmem := List.getElem_mem hk'
    have hout := mem_otherAxes.mp hmem
    have hb : (t.otherAxes ((batchDims ++ dims).map Prod.snd))[k] ∉ batchDims.map Prod.snd :=
      fun h' => by
        obtain ⟨p, hp, hpa⟩ := List.mem_map.mp h'
        exact hout (List.mem_map.mpr ⟨p, List.mem_append_left _ hp, hpa⟩)
    have hd : (t.otherAxes ((batchDims ++ dims).map Prod.snd))[k] ∉ dims.map Prod.snd :=
      fun h' => by
        obtain ⟨p, hp, hpa⟩ := List.mem_map.mp h'
        exact hout (List.mem_map.mpr ⟨p, List.mem_append_right _ hp, hpa⟩)
    refine (Coord.index_restrict _ _ k hk').trans ?_
    simp only [ContractValid.contractRight, Coord.get_ofFun, dite_eq_right hb, dite_eq_right hd]
    rw [(nodup_otherAxes _ _).idxOf_getElem]
    rfl

/-- The pairs of an output coordinate of `contract` and a contracted coordinate correspond to the
pairs of a coordinate of `s` and a coordinate of the other axes of `t`. -/
def ContractValid.contractLeftEquiv (h : ContractValid s t dims batchDims) :
    (s.contract t dims batchDims).Coord × Coord ((dims.map Prod.fst).map s.extent) ≃
      s.Coord × Coord (t.eraseAxes ((batchDims ++ dims).map Prod.snd)) where
  toFun p := (s.contractLeft t dims batchDims p.1 p.2,
    (Coord.appendEquiv
      ((batchDims.map Prod.fst).map s.extent ++ s.eraseAxes ((batchDims ++ dims).map Prod.fst))
      (t.eraseAxes ((batchDims ++ dims).map Prod.snd)) p.1).2)
  invFun q := (s.contractOutputLeft t dims batchDims q.1 q.2, q.1.restrict (dims.map Prod.fst))
  left_inv p :=
    Prod.ext (contractOutputLeft_contractLeft h p.1 p.2) (restrict_contractLeft h p.1 p.2)
  right_inv q := Prod.ext (contractLeft_contractOutputLeft q.1 q.2) (by
    simp only [contractOutputLeft, Equiv.apply_symm_apply])

/-- The pairs of an output coordinate of `contract` and a contracted coordinate correspond to the
pairs of a coordinate of `t` and a coordinate of the other axes of `s`. -/
def ContractValid.contractRightEquiv (h : ContractValid s t dims batchDims) :
    (s.contract t dims batchDims).Coord × Coord ((dims.map Prod.fst).map s.extent) ≃
      t.Coord × Coord (s.eraseAxes ((batchDims ++ dims).map Prod.fst)) where
  toFun p := (h.contractRight p.1 p.2,
    (Coord.appendEquiv ((batchDims.map Prod.fst).map s.extent)
      (s.eraseAxes ((batchDims ++ dims).map Prod.fst))
      (Coord.appendEquiv
        ((batchDims.map Prod.fst).map s.extent ++ s.eraseAxes ((batchDims ++ dims).map Prod.fst))
        (t.eraseAxes ((batchDims ++ dims).map Prod.snd)) p.1).1).2)
  invFun q := (h.contractOutputRight q.1 q.2,
    Coord.reindexEq h.dims_extents (q.1.restrict (dims.map Prod.snd)))
  left_inv p := Prod.ext (h.contractOutputRight_contractRight p.1 p.2)
    (h.reindexEq_restrict_contractRight p.1 p.2)
  right_inv q := Prod.ext (h.contractRight_contractOutputRight q.1 q.2) (by
    simp only [ContractValid.contractOutputRight, Equiv.apply_symm_apply])

end TorchLean.Shape

namespace TorchLean.Tensor

variable {s t : Shape} {dims batchDims : List (Fin s.rank × Fin t.rank)}

/-- A sum over the flat indices of a shape, each read as its coordinate, is the sum over the
coordinates. -/
theorem sum_unlinearize {M : Type} [AddCommMonoid M] (u : Shape) (f : u.Coord → M) :
    ∑ i : Fin (Shape.size u), f (Shape.Coord.unlinearize i) = ∑ c, f c :=
  (Shape.Coord.equivFin u).symm.sum_comp f

/-- An entry of a real contraction is the sum over the contracted coordinates of the products it
pairs. -/
theorem contract_apply_eq_sum (h : Shape.ContractValid s t dims batchDims) (x : Tensor ℝ s)
    (y : Tensor ℝ t) (o : (s.contract t dims batchDims).Coord) :
    contract x y dims batchDims _ h rfl o =
      ∑ r, x (s.contractLeft t dims batchDims o r) * y (h.contractRight o r) := by
  rw [contract_apply, List.finRange_foldl_add_eq_add_sum, zero_add, Shape.Coord.reindexEq_self]
  exact sum_unlinearize _ fun r => x (s.contractLeft t dims batchDims o r) * y (h.contractRight o r)

/-- An entry of the real pullback of `contract` to `x` is a sum over the other axes of `y`. -/
theorem contractBackwardLeft_apply_eq_sum (h : Shape.ContractValid s t dims batchDims)
    (g : Tensor ℝ (s.contract t dims batchDims)) (y : Tensor ℝ t) (c : s.Coord) :
    contractBackwardLeft dims batchDims h g y c =
      ∑ j, g (s.contractOutputLeft t dims batchDims c j) *
        y (h.contractRight (s.contractOutputLeft t dims batchDims c j)
          (c.restrict (dims.map Prod.fst))) := by
  rw [contractBackwardLeft_apply, List.finRange_foldl_add_eq_add_sum, zero_add]
  exact sum_unlinearize _ fun j => g (s.contractOutputLeft t dims batchDims c j) *
    y (h.contractRight (s.contractOutputLeft t dims batchDims c j) (c.restrict (dims.map Prod.fst)))

/-- An entry of the real pullback of `contract` to `y` is a sum over the other axes of `x`. -/
theorem contractBackwardRight_apply_eq_sum (h : Shape.ContractValid s t dims batchDims)
    (x : Tensor ℝ s) (g : Tensor ℝ (s.contract t dims batchDims)) (d : t.Coord) :
    contractBackwardRight dims batchDims h x g d =
      ∑ i, x (s.contractLeft t dims batchDims (h.contractOutputRight d i)
          (Shape.Coord.reindexEq h.dims_extents (d.restrict (dims.map Prod.snd)))) *
        g (h.contractOutputRight d i) := by
  rw [contractBackwardRight_apply, List.finRange_foldl_add_eq_add_sum, zero_add]
  exact sum_unlinearize _ fun i => x (s.contractLeft t dims batchDims (h.contractOutputRight d i)
    (Shape.Coord.reindexEq h.dims_extents (d.restrict (dims.map Prod.snd)))) *
      g (h.contractOutputRight d i)

/-- The inner product of a real contraction `contract x y` with `g` is the inner product of `x`
with `contractBackwardLeft g y`. -/
theorem inner_contract_eq_inner_contractBackwardLeft (h : Shape.ContractValid s t dims batchDims)
    (x : Tensor ℝ s) (y : Tensor ℝ t) (g : Tensor ℝ (s.contract t dims batchDims)) :
    inner ℝ (contract x y dims batchDims _ h rfl) g =
      inner ℝ x (contractBackwardLeft dims batchDims h g y) := by
  rw [inner_eq_sum, inner_eq_sum]
  simp only [contract_apply_eq_sum h, contractBackwardLeft_apply_eq_sum h, Finset.sum_mul,
    Finset.mul_sum]
  rw [← Fintype.sum_prod_type', ← Fintype.sum_prod_type']
  refine Fintype.sum_equiv h.contractLeftEquiv _ _ ?_
  rintro ⟨o, r⟩
  simp only [Shape.ContractValid.contractLeftEquiv, Equiv.coe_fn_mk]
  rw [Shape.contractOutputLeft_contractLeft h o r, Shape.restrict_contractLeft h o r]
  ring

/-- The inner product of a real contraction `contract x y` with `g` is the inner product of `y`
with `contractBackwardRight x g`. -/
theorem inner_contract_eq_inner_contractBackwardRight (h : Shape.ContractValid s t dims batchDims)
    (x : Tensor ℝ s) (y : Tensor ℝ t) (g : Tensor ℝ (s.contract t dims batchDims)) :
    inner ℝ (contract x y dims batchDims _ h rfl) g =
      inner ℝ y (contractBackwardRight dims batchDims h x g) := by
  rw [inner_eq_sum, inner_eq_sum]
  simp only [contract_apply_eq_sum h, contractBackwardRight_apply_eq_sum h, Finset.sum_mul,
    Finset.mul_sum]
  rw [← Fintype.sum_prod_type', ← Fintype.sum_prod_type']
  refine Fintype.sum_equiv h.contractRightEquiv _ _ ?_
  rintro ⟨o, r⟩
  simp only [Shape.ContractValid.contractRightEquiv, Equiv.coe_fn_mk]
  rw [h.contractOutputRight_contractRight o r, h.reindexEq_restrict_contractRight o r]
  ring

end TorchLean.Tensor

namespace TorchLean.Op

section Contract

variable {s t : Shape} (dims batchDims : List (Fin s.rank × Fin t.rank))
  (h : Shape.ContractValid s t dims batchDims)

/-- Real contraction with the right operand `y` fixed, as a continuous linear map of the left
operand. -/
noncomputable def contractLeftCLM (y : Tensor ℝ t) :
    Tensor ℝ s →L[ℝ] Tensor ℝ (s.contract t dims batchDims) :=
  let linear : Tensor ℝ s →ₗ[ℝ] Tensor ℝ (s.contract t dims batchDims) :=
    { toFun := fun x => Tensor.contract x y dims batchDims _ h rfl
      map_add' := by
        intro x x'
        apply Tensor.Rep.ext
        intro o
        simp only [Tensor.add_apply, Tensor.contract_apply_eq_sum h, add_mul,
          Finset.sum_add_distrib]
      map_smul' := by
        intro a x
        apply Tensor.Rep.ext
        intro o
        simp only [Tensor.Rep.smul_apply, Tensor.contract_apply_eq_sum h, smul_eq_mul,
          RingHom.id_apply, Finset.mul_sum, mul_assoc] }
  ⟨linear, linear.continuous_of_finiteDimensional⟩

/-- Real contraction with the left operand `x` fixed, as a continuous linear map of the right
operand. -/
noncomputable def contractRightCLM (x : Tensor ℝ s) :
    Tensor ℝ t →L[ℝ] Tensor ℝ (s.contract t dims batchDims) :=
  let linear : Tensor ℝ t →ₗ[ℝ] Tensor ℝ (s.contract t dims batchDims) :=
    { toFun := fun y => Tensor.contract x y dims batchDims _ h rfl
      map_add' := by
        intro y y'
        apply Tensor.Rep.ext
        intro o
        simp only [Tensor.add_apply, Tensor.contract_apply_eq_sum h, mul_add,
          Finset.sum_add_distrib]
      map_smul' := by
        intro a y
        apply Tensor.Rep.ext
        intro o
        simp only [Tensor.Rep.smul_apply, Tensor.contract_apply_eq_sum h, smul_eq_mul,
          RingHom.id_apply, Finset.mul_sum, mul_left_comm] }
  ⟨linear, linear.continuous_of_finiteDimensional⟩

/-- The pullback of `contract` to its left operand, `contractBackwardLeft`, is the adjoint of the
derivative, the right operand fixed. -/
@[op_cert TorchLean.Op.contract derivative 0]
theorem contract_backward_eq_adjoint (x : Tensor ℝ s) (y : Tensor ℝ t)
    (δ : Tensor ℝ (s.contract t dims batchDims)) :
    (((Op.contract dims batchDims h).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).1 =
      (fderiv ℝ (fun x' => (((Op.contract dims batchDims h).rule? (α := ℝ) |>.get rfl).forward
        (x', y, ())).1) x).adjoint δ := by
  have hf : (fun x' => (((Op.contract dims batchDims h).rule? (α := ℝ) |>.get rfl).forward
      (x', y, ())).1) = ⇑(contractLeftCLM dims batchDims h y) := rfl
  rw [hf, (contractLeftCLM dims batchDims h y).fderiv]
  apply ext_inner_left ℝ
  intro u
  rw [ContinuousLinearMap.adjoint_inner_right]
  exact (Tensor.inner_contract_eq_inner_contractBackwardLeft h u y δ).symm

/-- The pullback of `contract` to its right operand, `contractBackwardRight`, is the adjoint of
the derivative, the left operand fixed. -/
@[op_cert TorchLean.Op.contract derivative 1]
theorem contract_right_backward_eq_adjoint (x : Tensor ℝ s) (y : Tensor ℝ t)
    (δ : Tensor ℝ (s.contract t dims batchDims)) :
    (((Op.contract dims batchDims h).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).2.1 =
      (fderiv ℝ (fun y' => (((Op.contract dims batchDims h).rule? (α := ℝ) |>.get rfl).forward
        (x, y', ())).1) y).adjoint δ := by
  have hf : (fun y' => (((Op.contract dims batchDims h).rule? (α := ℝ) |>.get rfl).forward
      (x, y', ())).1) = ⇑(contractRightCLM dims batchDims h x) := rfl
  rw [hf, (contractRightCLM dims batchDims h x).fderiv]
  apply ext_inner_left ℝ
  intro u
  rw [ContinuousLinearMap.adjoint_inner_right]
  exact (Tensor.inner_contract_eq_inner_contractBackwardRight h x u δ).symm

end Contract

end TorchLean.Op
