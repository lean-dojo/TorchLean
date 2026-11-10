/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.Op.Exec
public import TorchLean.Verify.CROWN.Op.RunValues
public import TorchLean.Verify.CROWN.Op.Conv
public import TorchLean.Verify.CROWN.FlatBox.Directed
public import TorchLean.Tensor.Reduce.Real
import TorchLean.Verify.CROWN.BoundOps.Interval
public import TorchLean.Verify.CROWN.LayerNorm.Enclosure
public import TorchLean.Verify.CROWN.FlatBox.Transfer
import TorchLean.NN.Attention.CausalMask
import TorchLean.NN.Softmax.HardMask
import TorchLean.NN.Softmax.Axis
import TorchLean.Verify.CROWN.FlatBox.Softplus
import TorchLean.NN.Softmax

/-!
# Enclosures of the rounded IBP boxes at lawful endpoint operations

A box encloses a real flat value when it has the value's width and its endpoints, read as reals,
bound every coordinate (`Encloses`). A value of the scalar is read as a real through `realValue`,
which maps the tensors of value, frozen and detached ports through `toReal` and keeps index,
Boolean and integer tensors, and `RealConsts` and `ConstsReal` say that a real payload holds the
real values of the rounded constants.

The lemmas here are stated at a scalar with lawful endpoint operations (`LawfulBoundOps`), and the
few that also need the nonlinear ones take them as instance arguments. The boxes of pointwise and
binary maps enclose the real images (`encloses_unary`, `encloses_binary`, `binaryBox?_encloses`).
The re-indexings and extrema commute with `toReal` and their real maps are monotone, so their
endpoint boxes enclose their real outputs (`encloses_endpoints`). The box of a sum over one axis
encloses the exact axis sum, and a mean over one axis divides that box by an enclosure of the exact
axis length (`Directed.ibpReduceSumAxis_encloses`,
`Directed.ibpReduceMeanAxis_encloses`, `flat_reduce_sum_axis`, `flat_mean_axis`). The
softmax and masked softmax boxes, within `[0, 1]`, enclose the real probabilities
(`Directed.ibpSoftmaxRange_encloses`,
`Directed.ibpHardMaskedSoftmax_encloses`, `maskedSoftmaxBox?_encloses`). The
directed contraction of a matrix product encloses the real contraction (`contractBox?_encloses`),
a concatenation places the box of each input at that input's coordinates (`encloses_concat`), and
a convolution's box encloses the real convolution of the real kernel and bias (`encloses_conv`).
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)
open TorchLean.Shape.Coord (equivFin equivFin_apply linearize)
open TorchLean.Shape (size_append)
open TorchLean.Shape.Coord (linearize_unlinearize unlinearize)
open TorchLean.Shape.Coord (unlinearize_linearize)
open TorchLean.Shape (rank_eq_length)
open TorchLean.Shape (Coord)

namespace TorchLean.CROWN.Directed

open Tensor
open scoped BigOperators

noncomputable section

open Graph (ibpUnflatten)

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

/-- A shaped endpoint pair encloses a real tensor at every coordinate. -/
def TensorEncloses {s : Shape} (lo hi : Tensor α s) (x : Tensor ℝ s) : Prop :=
  ∀ c : s.Coord, value (lo c) ≤ x c ∧ x c ≤ value (hi c)

/-- Reconstruct a real tensor from its row-major graph coordinates. -/
def realTensor (s : Shape) (f : Nat → ℝ) : Tensor ℝ s :=
  Tensor.reshape (Tensor.ofFn fun i => f i.val) s (Shape.size_vector _)

/-- Read the flat coordinates of a real tensor, extending them by zero out of bounds. -/
def tensorValues {s : Shape} (x : Tensor ℝ s) (i : Nat) : ℝ :=
  getAtOrZero (Tensor.reshape x [_] (Shape.size_vector _).symm) [i]

/-- At an index below `s.size`, `tensorValues x` reads the flattened `x`. -/
@[simp] theorem tensorValues_fin {s : Shape} (x : Tensor ℝ s) (i : Fin s.size) :
    tensorValues x i.val = (Tensor.reshape x [_] (Shape.size_vector _).symm).getScalar i :=
  Tensor.getAtOrZero_eq_getScalar _ i

/-- `realTensor` reads the flat graph value at the row-major position of each coordinate. -/
@[simp] theorem realTensor_apply (s : Shape) (f : Nat → ℝ) (c : s.Coord) :
    realTensor s f c = f (Coord.linearize c).val := by
  unfold realTensor
  rw [← Tensor.getScalar_reshape_toFin, Tensor.reshape_reshape_self, getScalar_ofFn]

/-- An enclosure of a tensor encloses each of its slices along the outer axis. -/
theorem TensorEncloses.unstack {n : Nat} {s : Shape}
    {lo hi : Tensor α (.dim n s)} {x : Tensor ℝ (.dim n s)}
    (h : TensorEncloses lo hi x) (i : Fin n) :
    TensorEncloses (lo.unstack i) (hi.unstack i) (x.unstack i) := by
  intro c
  simpa only [Tensor.unstack_apply] using h (i, c)

/-- Enclosures of the slices along the outer axis enclose the stacked tensor. -/
theorem TensorEncloses.dim {n : Nat} {s : Shape}
    {lo hi : Fin n → Tensor α s} {x : Fin n → Tensor ℝ s}
    (h : ∀ i, TensorEncloses (lo i) (hi i) (x i)) :
    TensorEncloses (Tensor.dim lo) (Tensor.dim hi) (Tensor.dim x) := by
  rintro ⟨i, c⟩
  simpa only [Tensor.dim_apply] using h i c

/-- Reshaping both endpoints and the tensor keeps an enclosure. -/
theorem TensorEncloses.reshape {s t : Shape} {lo hi : Tensor α s} {x : Tensor ℝ s}
    (h : TensorEncloses lo hi x)
    (hsize : Shape.size s = Shape.size t) :
    TensorEncloses (Tensor.Rep.reshape hsize lo)
      (Tensor.Rep.reshape hsize hi) (Tensor.Rep.reshape hsize x) := by
  intro c
  simpa only [Tensor.Rep.reshape_apply_reshapeEquiv] using
    h (Coord.reshapeEquiv hsize c)

/-- Flattening both endpoints and the tensor to vectors keeps an enclosure. -/
theorem TensorEncloses.flatten {s : Shape} {lo hi : Tensor α s} {x : Tensor ℝ s}
    (h : TensorEncloses lo hi x) :
    TensorEncloses (Tensor.reshape lo [_] (Shape.size_vector _).symm) (Tensor.reshape hi [_]
        (Shape.size_vector _).symm) (Tensor.reshape x [_] (Shape.size_vector _).symm) :=
  h.reshape (Shape.size_toList_flatten s)

/-- Reshaping flat endpoints and a flat tensor to the shape `s` keeps an enclosure. -/
theorem TensorEncloses.unflatten {s : Shape} {lo hi : Tensor α [s.size]}
    {x : Tensor ℝ [s.size]} (h : TensorEncloses lo hi x) :
    TensorEncloses (Tensor.reshape lo s (Shape.size_vector _)) (Tensor.reshape hi s
        (Shape.size_vector _))
      (Tensor.reshape x s (Shape.size_vector _)) :=
  h.reshape (Shape.size_toList_flatten s).symm

/-- A vector enclosure is an enclosure of every entry. -/
theorem tensorEncloses_vector_iff {n : Nat} {lo hi : Tensor α [n]} {x : Tensor ℝ [n]} :
    TensorEncloses lo hi x ↔
      ∀ i, value (lo.getScalar i) ≤ x.getScalar i ∧
        x.getScalar i ≤ value (hi.getScalar i) := by
  simp only [TensorEncloses, getScalar_eq_apply, Prod.forall]
  constructor
  · exact fun h i => h i PUnit.unit
  · intro h i c
    cases c
    exact h i

/-- Flattened endpoint bounds recover enclosure of the shaped real input. -/
theorem tensorEncloses_ibpUnflatten {s : Shape} {B : FlatBox α} {f : Nat → ℝ}
    (hd : B.dim = s.size) (hB : RowEncloses B s.size f) :
    TensorEncloses (ibpUnflatten B.dim B.lo hd) (ibpUnflatten B.dim B.hi hd)
      (realTensor s f) := by
  obtain ⟨d, lo, hi⟩ := B
  dsimp only at hd
  subst d
  simp only [ibpUnflatten, castShape_self, realTensor]
  apply TensorEncloses.unflatten
  rw [tensorEncloses_vector_iff]
  simpa only [getScalar_ofFn] using rowEncloses_iff.mp hB

/-- A shaped enclosure gives the flat graph enclosure when the real coordinates agree. -/
theorem TensorEncloses.row {s : Shape} {lo hi : Tensor α s} {x : Tensor ℝ s}
    (h : TensorEncloses lo hi x) {f : Nat → ℝ}
    (hf : ∀ i : Fin s.size,
        f i.val = (Tensor.reshape x [_] (Shape.size_vector _).symm).getScalar i) :
    RowEncloses { dim := s.size, lo := Tensor.reshape lo [_] (Shape.size_vector _).symm,
                  hi := Tensor.reshape hi [_] (Shape.size_vector _).symm }
      s.size f := by
  rw [rowEncloses_iff]
  intro i
  rw [hf i]
  exact tensorEncloses_vector_iff.mp h.flatten i

/-- A shaped enclosure gives the flat graph enclosure of `tensorValues x`. -/
theorem TensorEncloses.tensorValues {s : Shape} {lo hi : Tensor α s} {x : Tensor ℝ s}
    (h : TensorEncloses lo hi x) :
    RowEncloses { dim := s.size, lo := Tensor.reshape lo [_] (Shape.size_vector _).symm,
                  hi := Tensor.reshape hi [_] (Shape.size_vector _).symm }
      s.size (tensorValues x) :=
  h.row (tensorValues_fin x)

/-- A packed directed sum encloses the corresponding exact tensor sum. -/
theorem TensorEncloses.sum {s : Shape} {lo hi : Tensor α s} {x : Tensor ℝ s}
    (h : TensorEncloses lo hi x) :
    value (Tensor.foldl BoundOps.addDown 0 lo) ≤ Tensor.sum x ∧
      Tensor.sum x ≤ value (Tensor.foldl BoundOps.addUp 0 hi) := by
  have hc (i : Fin (Shape.size s)) :
      value (lo.getFlat i) ≤ x.getFlat i ∧ x.getFlat i ≤ value (hi.getFlat i) := by
    simpa only [Tensor.Rep.get, Coord.linearize_unlinearize] using
      h (Coord.unlinearize i)
  simpa only [Tensor.sum_eq_foldl, Tensor.Rep.foldl_eq_fin_foldl,
    Fin.foldl_eq_foldl_finRange, List.finRange_foldl_add_eq_add_sum, zero_add] using
    sum_encloses lo.getFlat hi.getFlat x.getFlat hc

/-- Summing along the outer axis, downward for `lo` and upward for `hi`, keeps an enclosure. -/
theorem TensorEncloses.reduceOuterAxis_sum {s : Shape} {n : Nat}
    {lo hi : Tensor α (.dim n s)} {x : Tensor ℝ (.dim n s)}
    (h : TensorEncloses lo hi x) :
    TensorEncloses
      (Tensor.reduceOuterAxis (fun row => Tensor.foldl BoundOps.addDown 0 row) lo)
      (Tensor.reduceOuterAxis (fun row => Tensor.foldl BoundOps.addUp 0 row) hi)
      (Tensor.reduceOuterAxis Tensor.sum x) := by
  induction s with
  | scalar =>
      intro c
      cases c
      simpa only [Tensor.reduceOuterAxis, Tensor.scalar_apply] using h.sum
  | dim m s ih =>
      apply TensorEncloses.dim
      intro j
      apply ih
      apply TensorEncloses.dim
      intro i
      exact (h.unstack i).unstack j

/-- Summing along `axis`, downward for `lo` and upward for `hi`, keeps an enclosure. -/
theorem TensorEncloses.reduceSum {s : Shape} {lo hi : Tensor α s} {x : Tensor ℝ s}
    (h : TensorEncloses lo hi x) (axis : Nat) :
    TensorEncloses
      (Tensor.reduceDimWith (fun row => Tensor.foldl BoundOps.addDown 0 row) _ axis
          lo)
      (Tensor.reduceDimWith (fun row => Tensor.foldl BoundOps.addUp 0 row) _ axis
          hi)
      (Tensor.reduceDimWith Tensor.sum _ axis x) := by
  induction s generalizing axis with
  | scalar => exact h
  | dim n s ih =>
      cases axis with
      | zero => exact h.reduceOuterAxis_sum
      | succ axis =>
          apply TensorEncloses.dim
          intro i
          exact ih (h.unstack i) axis

open Graph

/-- The exact sum along `axis`, in the runtime's flat output order. -/
def realAxisSum (s : Shape) (axis : Nat) (f : Nat → ℝ) : Nat → ℝ :=
  tensorValues (Tensor.reduceDimWith Tensor.sum _ axis (realTensor s f))

/-- The exact axis sum divided by the selected axis length. -/
def realAxisMean (s : Shape) (axis : Nat) (f : Nat → ℝ) (i : Nat) : ℝ :=
  realAxisSum s axis f i / (s[axis]?.getD 0 : ℝ)

/-- Directed summation encloses the exact sum over the selected axis. -/
theorem ibpReduceSumAxis_encloses {s : Shape} {axis : Nat} {B box : FlatBox α}
    {f : Nat → ℝ} (hB : RowEncloses B s.size f)
    (hbox : ibpReduceSumAxis axis B s = some box) :
    RowEncloses box (Shape.size (s.eraseIdx axis)) (realAxisSum s axis f) := by
  unfold ibpReduceSumAxis at hbox
  split at hbox
  next hd =>
    cases ha : Shape.nonemptyAxis? axis s with
    | none => simp [ha] at hbox
    | some witness =>
        simp only [ha, Option.some.injEq] at hbox
        subst box
        exact ((tensorEncloses_ibpUnflatten hd hB).reduceSum axis).tensorValues
  next => contradiction

/-- `Graph.countBounds n` encloses `n`. -/
theorem countBounds_encloses (n : Nat) :
    value (Graph.countBounds (α := α) n).1 ≤ n ∧
      (n : ℝ) ≤ value (Graph.countBounds (α := α) n).2 := by
  induction n with
  | zero => simp [Graph.countBounds, LawfulBoundOps.toReal_zero (α := α)]
  | succ n ih =>
    simp only [Graph.countBounds, List.range_succ, List.foldl_append, List.foldl_cons,
      List.foldl_nil] at ih ⊢
    push_cast
    have hd := LawfulBoundOps.addDown_le (List.foldl (fun b _ => (BoundOps.addDown b.1 1,
      BoundOps.addUp b.2 1)) ((0 : α), (0 : α)) (List.range n)).1 1
    have hu := LawfulBoundOps.le_addUp (List.foldl (fun b _ => (BoundOps.addDown b.1 1,
      BoundOps.addUp b.2 1)) ((0 : α), (0 : α)) (List.range n)).2 1
    rw [LawfulBoundOps.toReal_one (α := α)] at hd hu
    exact ⟨by linarith [ih.1], by linarith [ih.2]⟩

/-- Dividing each directed axis sum by an enclosing count encloses the exact axis mean. -/
theorem ibpReduceMeanAxis_encloses [NonlinearBoundOps α] [LawfulNonlinearBoundOps α]
    {s : Shape} {axis : Nat} {B box : FlatBox α} {f : Nat → ℝ}
    (hB : RowEncloses B s.size f)
    (hbox : ibpReduceMeanAxis axis B s = some box) :
    RowEncloses box (Shape.size (s.eraseIdx axis)) (realAxisMean s axis f) := by
  unfold ibpReduceMeanAxis at hbox
  obtain ⟨summed, hsum, hbox⟩ := Option.bind_eq_some_iff.mp hbox
  obtain ⟨_, _, hbox⟩ := Option.bind_eq_some_iff.mp hbox
  have hcount := countBounds_encloses (α := α) (s[axis]?.getD 0)
  apply boxUnaryEnclosure?_encloses (F := fun x => x / (s[axis]?.getD 0 : ℝ))
    ?_ (ibpReduceSumAxis_encloses hB hsum) hbox
  intro lo hi outLo outHi x hout hlo hhi
  obtain ⟨_, _, hout⟩ := Option.bind_eq_some_iff.mp hout
  exact LawfulNonlinearBoundOps.divBounds_enclosure
    (finiteBounds?_bind_eq_some hout) hlo hhi hcount.1 hcount.2

omit [LawfulBoundOps α] in
/-- A successful axis sum reduces an axis of positive extent. -/
theorem ibpReduceSumAxis_extent_pos {s : Shape} {axis : Nat} {B box : FlatBox α}
    (hbox : ibpReduceSumAxis axis B s = some box) (hb : axis < s.rank) :
    0 < s.extent ⟨axis, hb⟩ := by
  unfold ibpReduceSumAxis at hbox
  split at hbox
  next =>
    cases ha : Shape.nonemptyAxis? axis s with
    | none => simp [ha] at hbox
    | some witness => exact witness.down.extent_pos hb
  next => contradiction

omit [LawfulBoundOps α] in
/-- A successful axis mean reduces an axis of positive extent. -/
theorem ibpReduceMeanAxis_extent_pos [NonlinearBoundOps α] {s : Shape} {axis : Nat}
    {B box : FlatBox α} (hbox : ibpReduceMeanAxis axis B s = some box) (hb : axis < s.rank) :
    0 < s.extent ⟨axis, hb⟩ := by
  unfold ibpReduceMeanAxis at hbox
  obtain ⟨_, hsum, _⟩ := Option.bind_eq_some_iff.mp hbox
  exact ibpReduceSumAxis_extent_pos hsum hb

end

end TorchLean.CROWN.Directed

namespace TorchLean

open Softmax TorchLean.Tensor
open Vec
open scoped BigOperators

noncomputable section

open CROWN (LawfulBoundOps)
open CROWN.Graph (ibpSoftmaxRange)

/-- Every coordinate of the actual innermost softmax lies in the closed unit interval. -/
theorem Softmax.innermost_mem_Icc {s : Shape} (x : Tensor ℝ s) (c : s.Coord) :
    innermost x c ∈ Set.Icc (0 : ℝ) 1 := by
  induction s with
  | scalar =>
      simp [innermost, Tensor.lastAxisMap, Tensor.scalar]
  | dim n s ih =>
      cases s with
      | scalar =>
          cases n with
          | zero => exact Fin.elim0 c.1
          | succ n =>
              simpa only [innermost, Tensor.lastAxisMap,
                Tensor.getScalar_eq_apply] using
                vec_mem_unitInterval x c.1
      | dim m s =>
          change Tensor.dim
            (fun i => innermost (x.unstack i)) c ∈ _
          simp only [Tensor.dim_apply]
          exact ih (x.unstack c.1) c.2

/-- A singleton innermost row has probability one, independently of its logit. -/
theorem Softmax.innermost_eq_one {s : Shape} (x : Tensor ℝ s)
    (hwidth : Shape.rowWidth s = 1) (c : s.Coord) :
    innermost x c = 1 := by
  rw [innermost_apply_row]
  have h (n : Nat) (hn : n = 1) (v : Vec n) (i : Fin n) : Vec.softmax v i = 1 := by
    subst n
    exact Vec.softmax_fin_one v i
  exact h _ hwidth _ _

/-- The actual real softmax along any valid axis has probability-valued coordinates. -/
theorem CROWN.Directed.softmax_mem_Icc {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (x : Tensor ℝ s) (c : s.Coord) :
    (Tensor.softmax x ⟨axis, Shape.AxisInBounds.proof⟩) c ∈ Set.Icc (0 : ℝ) 1 := by
  rw [Tensor.softmax_apply, Tensor.getScalar_eq_apply]
  exact innermost_mem_Icc (s := [_]) _ _

/-- Softmax on a singleton selected axis is exactly one at every coordinate. -/
theorem CROWN.Directed.softmax_eq_one {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (x : Tensor ℝ s) (hwidth : s[axis]?.getD 0 = 1) (c : s.Coord) :
    (Tensor.softmax x ⟨axis, Shape.AxisInBounds.proof⟩) c = 1 := by
  rw [Tensor.softmax_apply, Tensor.getScalar_eq_apply]
  refine innermost_eq_one (s := [_]) _ (Eq.trans ?_ hwidth) _
  have haxis : axis < s.length := Shape.rank_eq_length s ▸ Shape.AxisInBounds.proof
  rw [List.getElem?_eq_getElem haxis, Option.getD_some]
  rfl

/-- Hard-masked probabilities lie in the unit interval, including an entirely blocked row. -/
theorem Softmax.hardMaskedVec_mem_Icc {n : Nat}
    (x : Tensor ℝ [n]) (allowed : Tensor Bool [n]) (i : Fin n) :
    (hardMaskedVec x allowed).getScalar i ∈ Set.Icc (0 : ℝ) 1 := by
  classical
  cases hi : allowed.getScalar i with
  | false =>
      simp [hardMaskedVec_blocked_eq_zero x allowed i hi]
  | true =>
      rw [getScalar_hardMaskedVec]
      have hn (j : Fin n) :
          0 ≤ HardMaskedSoftmax.numerator allowed (getScalarE x) j := by
        unfold HardMaskedSoftmax.numerator
        split
        · exact (Real.exp_pos _).le
        · exact le_rfl
      have hle :
          HardMaskedSoftmax.numerator allowed (getScalarE x) i ≤
            HardMaskedSoftmax.denominator allowed (getScalarE x) :=
        Finset.single_le_sum (fun j _ => hn j) (Finset.mem_univ i)
      have hp := HardMaskedSoftmax.denominator_pos allowed (getScalarE x) ⟨i, hi⟩
      exact ⟨div_nonneg (hn i) hp.le, (div_le_one hp).mpr hle⟩

/-- The only allowed entry in a hard-mask row has probability one. -/
theorem Softmax.hardMaskedVec_eq_one {n : Nat}
    (x : Tensor ℝ [n]) (allowed : Tensor Bool [n]) (i : Fin n)
    (hi : allowed.getScalar i = true)
    (hunique : ∀ j, allowed.getScalar j = true → j = i) :
    (hardMaskedVec x allowed).getScalar i = 1 := by
  classical
  rw [getScalar_hardMaskedVec]
  have hsum : HardMaskedSoftmax.denominator allowed (getScalarE x) =
      HardMaskedSoftmax.numerator allowed (getScalarE x) i := by
    apply Finset.sum_eq_single i
    · intro j _ hji
      have hj : allowed.getScalar j = false := by
        cases h : allowed.getScalar j
        · rfl
        · exact False.elim (hji (hunique j h))
      simp [HardMaskedSoftmax.numerator, hj]
    · simp
  rw [hsum]
  apply div_self
  simp [HardMaskedSoftmax.numerator, hi, Real.exp_ne_zero]

variable {α : Type} [Storage α] [Context α] [CROWN.BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

/-- The executable range box encloses the actual real softmax of any input tensor. -/
theorem CROWN.Directed.ibpSoftmaxRange_encloses {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (x : Tensor ℝ s) {g : Nat → ℝ}
    (hg : ∀ c : s.Coord,
      g (Coord.linearize c).val = (Tensor.softmax x ⟨axis, Shape.AxisInBounds.proof⟩) c) :
    RowEncloses (ibpSoftmaxRange (α := α) s axis s.size) s.size g := by
  unfold ibpSoftmaxRange
  dsimp only
  split
  · rename_i hwidth
    rw [rowEncloses_iff]
    intro i
    have h := hg (Coord.unlinearize i)
    rw [Coord.linearize_unlinearize,
      softmax_eq_one axis x hwidth] at h
    simp [Tensor.getScalar_full, LawfulBoundOps.toReal_one, h]
  · rw [rowEncloses_iff]
    intro i
    have h := hg (Coord.unlinearize i)
    rw [Coord.linearize_unlinearize] at h
    simpa only [Tensor.getScalar_full, LawfulBoundOps.toReal_zero,
      LawfulBoundOps.toReal_one, h, Set.mem_Icc] using
      softmax_mem_Icc axis x (Coord.unlinearize i)

/-- The actual recursive hard-mask transfer encloses its real specification at every rank. -/
theorem CROWN.Directed.ibpHardMaskedSoftmax_encloses {s : Shape}
    (lo hi : Tensor α s) (x : Tensor ℝ s) (allowed : Tensor Bool s) (c : s.Coord) :
    value ((Graph.ibpHardMaskedSoftmax lo hi allowed).1 c) ≤
        Softmax.hardMasked x allowed c ∧
      Softmax.hardMasked x allowed c ≤
        value ((Graph.ibpHardMaskedSoftmax lo hi allowed).2 c) := by
  induction s with
  | scalar =>
      cases h : allowed.item <;>
        simp [Graph.ibpHardMaskedSoftmax, Softmax.hardMasked,
          Tensor.scalar, h, LawfulBoundOps.toReal_zero, LawfulBoundOps.toReal_one]
  | dim n s ih =>
      cases s with
      | scalar =>
          obtain ⟨i, ⟨⟩⟩ := c
          simp only [Softmax.hardMasked, Graph.ibpHardMaskedSoftmax,
            Tensor.ofFn_apply]
          have hzero :=
            Softmax.hardMaskedVec_blocked_eq_zero x allowed i
          have hone := hardMaskedVec_eq_one x allowed i
          have hrange := hardMaskedVec_mem_Icc x allowed i
          rw [Tensor.getScalar_eq_apply (hardMaskedVec x allowed) i]
            at hzero hone hrange
          cases ha : allowed.getScalar i with
          | false =>
              simp [hzero ha, LawfulBoundOps.toReal_zero]
          | true =>
              simp only [↓reduceIte, LawfulBoundOps.toReal_one]
              split
              · simpa only [LawfulBoundOps.toReal_zero, Set.mem_Icc] using
                  hrange
              · rename_i hother
                have hunique (j : Fin n) (hj : allowed.getScalar j = true) : j = i := by
                  by_contra hji
                  have hany : (List.finRange n).any
                      (fun k => i != k && allowed.getScalar k) = true := by
                    apply List.any_eq_true.mpr
                    exact ⟨j, List.mem_finRange j, by simp [hj, Ne.symm hji]⟩
                  exact hother hany
                simp [hone ha hunique, LawfulBoundOps.toReal_one]
      | dim m s =>
          change value (Tensor.dim (fun i =>
              (Graph.ibpHardMaskedSoftmax (lo.unstack i) (hi.unstack i)
                (allowed.unstack i)).1) c) ≤
              Tensor.dim (fun i => Softmax.hardMasked (x.unstack i)
                (allowed.unstack i)) c ∧
            Tensor.dim (fun i => Softmax.hardMasked (x.unstack i)
              (allowed.unstack i)) c ≤
              value (Tensor.dim (fun i =>
                (Graph.ibpHardMaskedSoftmax (lo.unstack i) (hi.unstack i)
                  (allowed.unstack i)).2) c)
          simp only [Tensor.dim_apply]
          exact ih (lo.unstack c.1) (hi.unstack c.1) (x.unstack c.1)
            (allowed.unstack c.1) c.2

end

end TorchLean

namespace TorchLean.CROWN.Op.Directed

open Tensor
open Graph
open CROWN.Directed
open Linear

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

noncomputable section

/-- A box encloses a real flat tensor: it has the tensor's width, and its endpoints, read as reals,
bound every coordinate. -/
def Encloses (B : FlatBox α) (v : FlatTensor ℝ) : Prop :=
  RowEncloses B v.n fun i => getAtOrZero v.v [i]

/-- The real carrier of a rounded carrier at a port: a tensor of the scalar read through
`toReal`, an index, Boolean or integer tensor unchanged. -/
def realCarrier : (p : Port) → p.Carrier α → p.Carrier ℝ
  | ⟨.value, _⟩, x => Tensor.map value x
  | ⟨.frozen, _⟩, x => Tensor.map value x
  | ⟨.detached, _⟩, x => Tensor.map value x
  | ⟨.index _, _⟩, x => x
  | ⟨.bool, _⟩, x => x
  | ⟨.integer _ _, _⟩, x => x

/-- The real value of a rounded value: its port, with the real carrier. -/
def realValue (v : Op.Value α) : Op.Value ℝ := ⟨v.1, realCarrier v.1 v.2⟩

/-- Reading a real value at a port reads the rounded value and takes the real carrier. -/
theorem read_realValue (p : Port) (v : Op.Value α) :
    p.read (realValue v) = (p.read v).map (realCarrier p) := by
  obtain ⟨⟨e, t⟩, x⟩ := v
  obtain ⟨d, s⟩ := p
  cases d <;> cases e <;>
    simp only [realValue, Port.read, realCarrier] <;>
    first
    | (by_cases h : t = s
       · subst h; simp [pure, Except.pure, Except.map, realCarrier]
       · simp [h, throw, throwThe, MonadExceptOf.throw, Except.map])
    | (simp [throw, throwThe, MonadExceptOf.throw, Except.map]; done)
    | (rename_i b₁ b₂
       by_cases h : b₂ = b₁ ∧ t = s
       · obtain ⟨rfl, rfl⟩ := h; simp [pure, Except.pure, Except.map, realCarrier]
       · have h' : ¬ (⟨.index b₂, t⟩ : Port) = ⟨.index b₁, s⟩ := fun e => h (by cases e; simp)
         simp [h', throw, throwThe, MonadExceptOf.throw, Except.map])
    | (rename_i s₁ w₁ s₂ w₂
       by_cases h : (s₂ = s₁ ∧ w₂ = w₁) ∧ t = s
       · obtain ⟨⟨rfl, rfl⟩, rfl⟩ := h; simp [pure, Except.pure, Except.map, realCarrier]
       · simp [h, throw, throwThe, MonadExceptOf.throw, Except.map])

/-- The real payload holds the real value of each constant of the rounded payload. -/
def RealConsts (Pr : IR.Payload ℝ) (P : IR.Payload α) : Prop :=
  ∀ k, Pr.const? k = (P.const? k).map realValue

/-- At `ℝ` the real value of a value is the value itself. -/
theorem realValue_real (v : Op.Value ℝ) : realValue v = v := by
  obtain ⟨⟨d, s⟩, x⟩ := v
  cases d <;> first | rfl | exact congrArg (Sigma.mk _) (Tensor.Rep.map_id x)

/-- At `ℝ` every payload holds the real values of its own constants. -/
theorem realConsts_self (P : IR.Payload ℝ) : RealConsts P P := fun k => by
  cases P.const? k with
  | none => rfl
  | some v => exact congrArg some (realValue_real v).symm

/-- A pointwise map at a value port: if a box encloses the flat entries of `x` and a transfer
encloses `F` coordinatewise, its box encloses the flat entries of `map f x`, for `f = F`. -/
theorem encloses_map {s : Shape} {x : Tensor ℝ s} {f F : ℝ → ℝ} (hf : ∀ z, f z = F z)
    {Bx B : FlatBox α} (hx : Encloses Bx (Graph.flatOfSome ⟨s, x⟩))
    (htransfer : ∀ {g : Nat → ℝ}, RowEncloses Bx s.size g →
      RowEncloses B s.size fun i => F (g i)) :
    Encloses B (Graph.flatOfSome ⟨s, Tensor.map f x⟩) := by
  refine (htransfer hx).congr fun i => ?_
  simp only [Graph.flatOfSome, Tensor.reshape_vector_map,
    getAtOrZero_eq_getScalar, getScalar_map, hf]

/-- A pointwise binary operation at value ports: if two boxes enclose the flat entries of `x` and
`y` and a transfer encloses `F` coordinatewise, its box encloses the flat entries of
`zipWith f x y`, for `f = F`. -/
theorem encloses_zipWith {s : Shape} {x y : Tensor ℝ s} {f F : ℝ → ℝ → ℝ}
    (hf : ∀ a b, f a b = F a b) {Bx By B : FlatBox α}
    (hx : Encloses Bx (Graph.flatOfSome ⟨s, x⟩))
    (hy : Encloses By (Graph.flatOfSome ⟨s, y⟩))
    (htransfer : ∀ {g h : Nat → ℝ}, RowEncloses Bx s.size g → RowEncloses By s.size h →
      RowEncloses B s.size fun i => F (g i) (h i)) :
    Encloses B (Graph.flatOfSome ⟨s, Tensor.zipWith f x y⟩) := by
  refine (htransfer hx hy).congr fun i => ?_
  simp only [Graph.flatOfSome, Tensor.reshape_vector_zipWith,
    getAtOrZero_eq_getScalar, getScalar_zipWith, hf]

/-- A constant node holds, in the real payload, the real value of its rounded constant. -/
theorem constValue?_real {G : IR.Graph} {P : IR.Payload α} {Pr : IR.Payload ℝ}
    (hP : RealConsts Pr P) (c : Nat) :
    constValue? G Pr c = (constValue? G P c).map realValue := by
  unfold constValue?
  cases hn : G.nodes[c]? with
  | none => rfl
  | some n =>
    cases n with
    | input k p => rfl
    | op o refs => rfl
    | const p =>
      simp only [hP c]
      cases hc : P.const? c with
      | none => rfl
      | some v =>
        simp only [Option.map_some, Option.bind_eq_bind, Option.bind_some, read_realValue]
        cases hr : p.read v with
        | error e => rfl
        | ok x => rfl

/-- Each reference among `refs` to output `0` of a constant node of the rounded payload that reads
a value in the real table `earlier` reads the real value of that constant. -/
def ConstsReal (G : IR.Graph) (P : IR.Payload α)
    (earlier : Array (List (Op.Value ℝ))) (refs : List IR.Ref) : Prop :=
  ∀ r ∈ refs, ∀ w, r.read earlier = .ok w → r.out = 0 → ∀ v, constValue? G P r.node = some v →
    w = realValue v

/-- A reference that names a rounded constant tensor at a value port reads, at that port, the
constant's real tensor. -/
theorem read_constAt? {G : IR.Graph} {P : IR.Payload α}
    {earlier : Array (List (Op.Value ℝ))} {refs : List IR.Ref}
    (hc : ConstsReal G P earlier refs) {r : IR.Ref} (hr : r ∈ refs) {s : Shape}
    {x : Tensor α s} (h : constAt? G P r (.value s) = some x) {w : Op.Value ℝ}
    (hw : r.read earlier = .ok w) {y : Tensor ℝ s} (hy : (Port.value s).read w = .ok y) :
    y = Tensor.map value x := by
  unfold constAt? at h
  by_cases h0 : r.out = 0
  · simp only [h0, ↓reduceIte] at h
    cases hcv : constValue? G P r.node with
    | none => simp [hcv] at h
    | some v =>
      simp only [hcv, Option.bind_eq_bind, Option.bind_some] at h
      obtain rfl := hc r hr w hw h0 v hcv
      rw [read_realValue] at hy
      cases hrv : (Port.value s).read v with
      | error e => simp [hrv, Except.map] at hy
      | ok z =>
        simp only [hrv, Except.toOption, Option.some.injEq] at h
        subst h
        simp only [hrv, Except.map, Except.ok.injEq] at hy
        exact hy.symm
  · simp [h0] at h

/-- A unary operator whose real forward is the pointwise map `f`, with a transfer `T` enclosing
`F = f` coordinatewise: at a node of a run whose reference's box encloses the value it reads, the
transfer's box encloses the node's flat value. -/
theorem encloses_unary {s : Shape} {op : Op [.value s] [.value s]}
    {rule : Op.Rule ℝ [.value s] [.value s]} (hs : op.rule? (α := ℝ) = some rule)
    {f F : ℝ → ℝ} (hforward : ∀ x, (rule.forward (x, ())).1 = Tensor.map f x)
    (hf : ∀ z, f z = F z) {T : FlatBox α → Option (FlatBox α)}
    (hT : ∀ {Bx B : FlatBox α} {g : Nat → ℝ}, RowEncloses Bx s.size g → T Bx = some B →
      RowEncloses B s.size fun i => F (g i))
    {boxes : Array (Option (FlatBox α))} {earlier : Array (List (Op.Value ℝ))}
    {refs : Array IR.Ref} {B : FlatBox α} {outs : List (Op.Value ℝ)}
    (hpe : ∀ r ∈ refs.toList, ∀ (Br : FlatBox α) (w : Op.Value ℝ) (v : FlatTensor ℝ),
      box? boxes r = some Br → r.read earlier = .ok w → flatOf w = some v → Encloses Br v)
    (hc : ∀ r, refs.toList = [r] → (box? boxes r >>= T) = some B)
    (h : op.invoke (refs.toList.mapM (·.read earlier)) = .ok outs) :
    ∀ v, (outs[0]? >>= flatOf) = some v → Encloses B v := by
  obtain ⟨r, w, x, hr, hread, hx, rfl⟩ := invoke_one hs h
  obtain ⟨Br, hBr, hT'⟩ := Option.bind_eq_some_iff.mp (hc r hr)
  rw [hr] at hpe
  intro v hv
  simp only [List.getElem?_cons_zero, Option.bind_eq_bind, Option.bind_some, hforward,
    flatOf_value, Option.some.injEq] at hv
  subst hv
  exact encloses_map hf
    (hpe r (by simp) Br w _ hBr hread ((flatOf_of_read (Or.inl rfl) hx).trans (flatOf_value x)))
    fun hg => hT hg hT'

/-- A binary operator whose real forward is the pointwise `zipWith f`, with a transfer `T`
enclosing `F = f` coordinatewise: at a node of a run whose references' boxes enclose the values they
read, the transfer's box encloses the node's flat value. -/
theorem encloses_binary {s : Shape} {op : Op [.value s, .value s] [.value s]}
    {rule : Op.Rule ℝ [.value s, .value s] [.value s]}
    (hs : op.rule? (α := ℝ) = some rule) {f F : ℝ → ℝ → ℝ}
    (hforward : ∀ x y, (rule.forward (x, y, ())).1 = Tensor.zipWith f x y)
    (hf : ∀ a b, f a b = F a b) {T : FlatBox α → FlatBox α → Option (FlatBox α)}
    (hT : ∀ {Bx By B : FlatBox α} {g h : Nat → ℝ}, RowEncloses Bx s.size g →
      RowEncloses By s.size h → T Bx By = some B → RowEncloses B s.size fun i => F (g i) (h i))
    {boxes : Array (Option (FlatBox α))} {earlier : Array (List (Op.Value ℝ))}
    {refs : Array IR.Ref} {B : FlatBox α} {outs : List (Op.Value ℝ)}
    (hpe : ∀ r ∈ refs.toList, ∀ (Br : FlatBox α) (w : Op.Value ℝ) (v : FlatTensor ℝ),
      box? boxes r = some Br → r.read earlier = .ok w → flatOf w = some v → Encloses Br v)
    (hc : ∀ r₁ r₂, refs.toList = [r₁, r₂] → ∃ B₁ B₂, box? boxes r₁ = some B₁ ∧
      box? boxes r₂ = some B₂ ∧ T B₁ B₂ = some B)
    (h : op.invoke (refs.toList.mapM (·.read earlier)) = .ok outs) :
    ∀ v, (outs[0]? >>= flatOf) = some v → Encloses B v := by
  obtain ⟨r₁, r₂, w₁, w₂, x₁, x₂, hr, hread₁, hread₂, hx₁, hx₂, rfl⟩ := invoke_two hs h
  obtain ⟨B₁, B₂, hB₁, hB₂, hT'⟩ := hc r₁ r₂ hr
  rw [hr] at hpe
  intro v hv
  simp only [List.getElem?_cons_zero, Option.bind_eq_bind, Option.bind_some, hforward,
    flatOf_value, Option.some.injEq] at hv
  subst hv
  exact encloses_zipWith hf
    (hpe r₁ (by simp) B₁ w₁ _ hB₁ hread₁
      ((flatOf_of_read (Or.inl rfl) hx₁).trans (flatOf_value x₁)))
    (hpe r₂ (by simp) B₂ w₂ _ hB₂ hread₂
      ((flatOf_of_read (Or.inl rfl) hx₂).trans (flatOf_value x₂)))
    fun hg hh => hT hg hh hT'

/-- Enclosure at a width is enclosure at an equal width. -/
theorem rowEncloses_width {B : FlatBox α} {n n' : Nat} {f : Nat → ℝ} (h : n = n')
    (H : RowEncloses B n f) : RowEncloses B n' f := h ▸ H

/-- Reading a dimension cast back as a flat vector reads the same entries. -/
theorem getAtOrZero_eqRec {n n' : Nat} (h : n = n') (v : Tensor ℝ [n]) (i : Nat) :
    getAtOrZero (h ▸ v : Tensor ℝ [n']) [i] = getAtOrZero v [i] := by
  subst h
  rfl

omit [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α] in
/-- The flat value of a sum over every axis without `keepdim` has one entry: the sum of the flat
entries of the operand. -/
theorem flat_reduce_sum {s : Shape} (x : Tensor ℝ s)
    (h : (Tensor.ReduceOp.Scalar.seed (op := .sum) (α := ℝ)).isNone →
      ∀ d ∈ List.finRange s.rank, 0 < s.extent d) :
    getAtOrZero (Graph.flatOfSome
        ⟨_, Tensor.reduce .sum x (List.finRange s.rank) false h⟩).v [0] =
      ∑ i : Fin s.size, getAtOrZero (Graph.flatOfSome ⟨s, x⟩).v [i.val] := by
  have hone : (s.reduce (List.finRange s.rank) false).size = 1 := by
    rw [Shape.reduce_finRange]; rfl
  simp only [Graph.flatOfSome]
  rw [show (0 : Nat) = (⟨0, by omega⟩ : Fin (s.reduce (List.finRange s.rank) false).size).val
    from rfl, getAtOrZero_eq_getScalar, Tensor.getScalar_reshape_equivFin,
    Tensor.reduce_sum_finRange_apply,
    show Tensor.foldl (· + ·) 0 x = Tensor.sum x from (Tensor.sum_eq_foldl x).symm,
    Tensor.sum_eq_coord_sum, ← (Coord.equivFin s).sum_comp]
  refine Finset.sum_congr rfl fun c _ => ?_
  rw [Coord.equivFin_apply, getAtOrZero_eq_getScalar, Tensor.getScalar_reshape_toFin]

omit [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α] in
/-- The flat value of a sum over one axis without `keepdim` is the exact axis sum
`realAxisSum` of the flat entries of the operand. -/
theorem flat_reduce_sum_axis {s : Shape} (x : Tensor ℝ s) (d : Fin s.rank)
    (h : (Tensor.ReduceOp.Scalar.seed (op := .sum) (α := ℝ)).isNone → ∀ e ∈ [d], 0 < s.extent e)
    (k : Nat) :
    getAtOrZero (Graph.flatOfSome ⟨_, Tensor.reduce .sum x [d] false h⟩).v [k] =
      realAxisSum s d (fun i => getAtOrZero (Graph.flatOfSome ⟨s, x⟩).v [i]) k := by
  have : Shape.AxisInBounds d.val s := ⟨d.isLt⟩
  have hx : realTensor s (fun i => getAtOrZero (Graph.flatOfSome ⟨s, x⟩).v [i]) = x := by
    apply Tensor.Rep.ext
    intro c
    rw [realTensor_apply]
    exact Tensor.getAtOrZero_reshape_toFin x c
  have hred : Tensor.reduceDimWith Tensor.sum s d x =
      castShape (Tensor.reduce .sum x [d] false h) (Shape.reduce_singleton s d) := by
    apply Tensor.Rep.ext
    intro c
    rw [Tensor.reduceDimWith_apply, castShape_reduce_sum_singleton_apply,
      Tensor.sum_eq_foldl]
  have hcast : ∀ {t u : Shape} (y : Tensor ℝ t) (q : t = u),
      tensorValues (castShape y q) = tensorValues y := by
    intro t u y q
    subst q
    rfl
  rw [realAxisSum, hx, hred, hcast]
  rfl

omit [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α] in
/-- The flat value of a mean over one nonempty axis without `keepdim` is the exact axis mean
`realAxisMean` of the flat entries of the operand. -/
theorem flat_mean_axis {s : Shape} (x : Tensor ℝ s) (d : Fin s.rank) (hpos : 0 < s.extent d)
    (i : Fin (s.reduce [d] false).size) :
    getAtOrZero (Graph.flatOfSome ⟨_, Tensor.mean x [d] false⟩).v [i.val] =
      realAxisMean s d (fun j => getAtOrZero (Graph.flatOfSome ⟨s, x⟩).v [j]) i.val := by
  rw [realAxisMean, ← flat_reduce_sum_axis x d (fun h => absurd h (by decide))]
  simp only [Tensor.mean, Graph.flatOfSome, Tensor.reshape_vector_map,
    getAtOrZero_eq_getScalar, getScalar_map, Shape.selectAxes_singleton]
  congr 2
  have hsz : Shape.size [s.extent d] = s.extent d := by simp
  rw [hsz, ite_eq_right hpos.ne',
    List.getElem?_eq_getElem (d.isLt.trans_eq s.rank_eq_length), Option.getD_some]
  rfl

omit [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α] in
/-- The flat value of a mean squared error has one entry: the mean of the squared differences of
the flat entries of its operands (an empty sum over an empty shape, divided by zero, is zero). -/
theorem flat_meanSquaredError {s : Shape} (y t : Tensor ℝ s) :
    getAtOrZero (Graph.flatOfSome ⟨[], Tensor.scalar (Tensor.meanSquaredError y t)⟩).v
        [0] =
      (∑ i : Fin s.size, (getAtOrZero (Graph.flatOfSome ⟨s, y⟩).v [i.val] -
        getAtOrZero (Graph.flatOfSome ⟨s, t⟩).v [i.val]) ^ 2) / s.size := by
  rw [show getAtOrZero (Graph.flatOfSome
      ⟨[], Tensor.scalar (Tensor.meanSquaredError y t)⟩).v [0] =
      (Tensor.scalar (Tensor.meanSquaredError y t)).item from
    Tensor.getAtOrZero_reshape_toFin _ PUnit.unit, item_scalar,
    Tensor.meanSquaredError_of_isFinite y t (Context.isFinite_real _),
    Tensor.sum_eq_coord_sum, ← (Coord.equivFin s).sum_comp]
  have hsum : ∑ c : s.Coord, ((y - t) * (y - t)) c =
      ∑ c : s.Coord, (getAtOrZero (Graph.flatOfSome ⟨s, y⟩).v
        [((Coord.equivFin s) c).val] -
        getAtOrZero (Graph.flatOfSome ⟨s, t⟩).v
          [((Coord.equivFin s) c).val]) ^ 2 := by
    refine Finset.sum_congr rfl fun c _ => ?_
    simp only [Graph.flatOfSome, Coord.equivFin_apply,
      Tensor.getAtOrZero_reshape_toFin, Tensor.mul_apply, Rep.hSub_apply, pow_two]
  rw [hsum]
  by_cases hs : s.size = 0
  · have : IsEmpty s.Coord := ⟨fun c => (Fin.cast hs (Coord.equivFin s c)).elim0⟩
    rw [Tensor.meanDenominator_of_size_eq_zero hs]
    simp
  · rw [Tensor.meanDenominator_of_size_pos (Nat.pos_of_ne_zero hs), Context.divNat_real]

/-- The endpoints of a box that encloses the flat entries of `x`, read at the shape of `x`,
enclose `x` at every coordinate. -/
theorem encloses_reshape {s : Shape} {Bx : FlatBox α} {x : Tensor ℝ s}
    (hx : Encloses Bx (Graph.flatOfSome ⟨s, x⟩)) (h : Bx.dim = s.size) (c : s.Coord) :
    value (Tensor.reshape Bx.lo s ((Shape.size_vector _).trans h) c) ≤ x c ∧
      x c ≤ value (Tensor.reshape Bx.hi s ((Shape.size_vector _).trans h) c) := by
  obtain ⟨d, lo, hi⟩ := Bx
  simp only at h
  subst h
  have key : ∀ v : Tensor α [s.size], getAtOrZero v [(Coord.linearize c).val] =
      Tensor.reshape v s ((Shape.size_vector _).trans rfl) c := by
    intro v
    conv_lhs => rw [← reshape_reshape_self v ((Shape.size_vector _).trans rfl)
      (Shape.size_vector _).symm]
    exact Tensor.getAtOrZero_reshape_toFin _ c
  have hxc := Tensor.getAtOrZero_reshape_toFin x c
  have hc : value (getAtOrZero lo [(Coord.linearize c).val]) ≤
      getAtOrZero (Tensor.reshape x [s.size] (Shape.size_vector _).symm)
        [(Coord.linearize c).val] ∧
      getAtOrZero (Tensor.reshape x [s.size] (Shape.size_vector _).symm)
        [(Coord.linearize c).val] ≤ value (getAtOrZero hi [(Coord.linearize c).val]) :=
    hx.2 (Coord.linearize c)
  rw [hxc, key, key] at hc
  exact hc

/-- Squaring uses directed endpoint products, with zero as a lower bound across zero. -/
theorem _root_.TorchLean.CROWN.Directed.boxSquare_encloses {x : FlatBox α}
    {n : Nat} {f : Nat → ℝ} (hx : RowEncloses x n f) :
    RowEncloses (boxSquare x) n (fun i => (f i) ^ 2) := by
  obtain ⟨d, lo, hi⟩ := x
  have hd := hx.1
  change d = n at hd
  subst hd
  rw [rowEncloses_iff] at hx
  simp only [boxSquare]
  rw [rowEncloses_iff]
  intro i
  simpa only [getScalar_ofFn, squareBounds, pow_two] using square_encloses (hx i).1 (hx i).2

/-- `scaleBounds c` encloses `x c`. -/
theorem scaleBounds_enclosure (c : α) :
    UnaryEnclosure (fun x => x * value c) (scaleBounds c) := by
  intro lo hi outLo outHi x h hlo hhi
  simp only [scaleBounds, Option.some.injEq] at h
  have h₁ : outLo = (intervalMul lo hi c c).1 := by rw [h]
  have h₂ : outHi = (intervalMul lo hi c c).2 := by rw [h]
  rw [h₁, h₂]
  exact Graph.intervalMul_encloses hlo hhi le_rfl le_rfl

/-- `clampBounds a b` encloses `min (max x a) b`, for any `a` and `b`. -/
theorem clampBounds_enclosure (a b : α) :
    UnaryEnclosure (fun x => min (max x (value a)) (value b)) (clampBounds a b) := by
  intro lo hi outLo outHi x h hlo hhi
  simp only [clampBounds, Option.some.injEq, Prod.mk.injEq] at h
  obtain ⟨rfl, rfl⟩ := h
  simp only [LawfulBoundOps.toReal_minimum, LawfulBoundOps.toReal_maximum]
  exact ⟨min_le_min (max_le_max hlo le_rfl) le_rfl, min_le_min (max_le_max hhi le_rfl) le_rfl⟩

/-- `leakyReLUBounds a` encloses `x` above 0 and `a x` at most 0, for any sign of `a`. -/
theorem leakyReLUBounds_enclosure (a : α) :
    UnaryEnclosure (fun x => if 0 < x then x else value a * x) (leakyReLUBounds a) := by
  intro lo hi outLo outHi x h hlo hhi
  have hpos : ∀ e : α, 0 < e ↔ 0 < value e := fun e => by
    rw [LawfulBoundOps.lt_iff, LawfulBoundOps.toReal_zero]
  unfold leakyReLUBounds at h
  split_ifs at h with h₀ h₁
  · simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    have hx : 0 < x := ((hpos lo).mp h₀).trans_le hlo
    simp only [hx, ↓reduceIte]
    exact ⟨hlo, hhi⟩
  · simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    simp only [LawfulBoundOps.toReal_minimum, LawfulBoundOps.toReal_maximum,
      LawfulBoundOps.toReal_zero]
    have hlo0 : value lo ≤ 0 := not_lt.mp fun h' => h₀ ((hpos lo).mpr h')
    have hhi0 : 0 < value hi := (hpos hi).mp h₁
    have hd := LawfulBoundOps.mulDown_le a lo
    have hu := LawfulBoundOps.le_mulUp a lo
    split_ifs with hx
    · exact ⟨(min_le_right _ _).trans hx.le, hhi.trans (le_max_right _ _)⟩
    · have hx0 : x ≤ 0 := not_lt.mp hx
      rcases le_total 0 (value a) with ha | ha
      · have h₂ : value a * value lo ≤ value a * x := mul_le_mul_of_nonneg_left hlo ha
        have h₃ : value a * x ≤ 0 := mul_nonpos_of_nonneg_of_nonpos ha hx0
        exact ⟨(min_le_left _ _).trans (hd.trans h₂),
          h₃.trans (hhi0.le.trans (le_max_right _ _))⟩
      · have h₂ : value a * x ≤ value a * value lo := mul_le_mul_of_nonpos_left hlo ha
        have h₃ : 0 ≤ value a * x := mul_nonneg_of_nonpos_of_nonpos ha hx0
        exact ⟨(min_le_right _ _).trans h₃, h₂.trans (hu.trans (le_max_left _ _))⟩
  · simp only [Option.some.injEq] at h
    have hx : ¬ 0 < x := fun hx => h₁ ((hpos hi).mpr (hx.trans_le hhi))
    have h₂ : outLo = (intervalMul lo hi a a).1 := by rw [h]
    have h₃ : outHi = (intervalMul lo hi a a).2 := by rw [h]
    simp only [hx, ↓reduceIte, h₂, h₃, mul_comm (value a) x]
    exact Graph.intervalMul_encloses hlo hhi le_rfl le_rfl

/-- `zeroHullBounds` encloses every map whose value lies between its argument and 0. -/
theorem zeroHullBounds_enclosure {f : ℝ → ℝ} (hf : ∀ x, min x 0 ≤ f x ∧ f x ≤ max x 0) :
    UnaryEnclosure f (zeroHullBounds (α := α)) := by
  intro lo hi outLo outHi x h hlo hhi
  simp only [zeroHullBounds, Option.some.injEq, Prod.mk.injEq] at h
  obtain ⟨rfl, rfl⟩ := h
  simp only [LawfulBoundOps.toReal_minimum, LawfulBoundOps.toReal_maximum,
    LawfulBoundOps.toReal_zero]
  exact ⟨(min_le_min hlo le_rfl).trans (hf x).1, (hf x).2.trans (max_le_max hhi le_rfl)⟩

/-- A real `x g` with a gate `g` in `[0, 1]` lies between `x` and 0. -/
theorem mul_gate_mem {x g : ℝ} (h₀ : 0 ≤ g) (h₁ : g ≤ 1) :
    min x 0 ≤ x * g ∧ x * g ≤ max x 0 := by
  rcases le_total 0 x with hx | hx
  · exact ⟨(min_le_right _ _).trans (mul_nonneg hx h₀),
      (mul_le_of_le_one_right hx h₁).trans (le_max_left _ _)⟩
  · exact ⟨(min_le_left _ _).trans (le_mul_of_le_one_right hx h₁),
      (mul_nonpos_of_nonpos_of_nonneg hx h₀).trans (le_max_right _ _)⟩

/-- At `ℝ`, the tanh GELU `x (1 + tanh u) / 2` lies between `x` and 0. -/
theorem _root_.TorchLean.Activation.geluTanh_mem_real (x : ℝ) :
    min x 0 ≤ Activation.geluTanh x ∧ Activation.geluTanh x ≤ max x 0 := by
  have h := mul_gate_mem (x := x)
    (g := (1 + Real.tanh (Activation.geluBeta * (x + Activation.geluKappa * (x * x * x))))
      / 2)
    (by linarith [Real.neg_one_lt_tanh (Activation.geluBeta *
      (x + Activation.geluKappa * (x * x * x)))])
    (by linarith [Real.tanh_lt_one (Activation.geluBeta *
      (x + Activation.geluKappa * (x * x * x)))])
  have he : Activation.geluTanh x = x * ((1 + Real.tanh (Activation.geluBeta *
      (x + Activation.geluKappa * (x * x * x)))) / 2) := by
    simp only [Activation.geluTanh, Context.isInf_real, Bool.false_eq_true, ↓reduceIte,
      MathFunctions.tanh]
    push_cast
    ring
  rw [he]
  exact h

/-- At `ℝ`, the exact GELU `x (1 + erf (x √½)) / 2` lies between `x` and 0. -/
theorem gelu_mem_real (x : ℝ) :
    min x 0 ≤ x * ((1 / 2 : ℚ) : ℝ) * (1 + Erf.erf (x * MathFunctions.sqrt ((1 / 2 : ℚ) : ℝ))) ∧
      x * ((1 / 2 : ℚ) : ℝ) * (1 + Erf.erf (x * MathFunctions.sqrt ((1 / 2 : ℚ) : ℝ))) ≤
        max x 0 := by
  have h := mul_gate_mem (x := x)
    (g := (1 + erf (x * MathFunctions.sqrt ((1 / 2 : ℚ) : ℝ))) / 2)
    (by linarith [neg_one_lt_erf (x * MathFunctions.sqrt ((1 / 2 : ℚ) : ℝ))])
    (by linarith [erf_lt_one (x * MathFunctions.sqrt ((1 / 2 : ℚ) : ℝ))])
  have he : x * ((1 / 2 : ℚ) : ℝ) * (1 + Erf.erf (x * MathFunctions.sqrt ((1 / 2 : ℚ) : ℝ))) =
      x * ((1 + erf (x * MathFunctions.sqrt ((1 / 2 : ℚ) : ℝ))) / 2) := by
    show x * ((1 / 2 : ℚ) : ℝ) * (1 + erf _) = _
    push_cast
    ring
  rw [he]
  exact h

/-- If a box encloses the flat entries of `x`, and bounds that enclose `x` at every coordinate
give bounds enclosing `y` at each output coordinate, `coordwiseBox?` encloses the flat
entries of `y`. -/
theorem coordwiseBox?_encloses {s t : Shape}
    {bound : Tensor α s → Tensor α s → t.Coord → Option (α × α)} {Bx B : FlatBox α}
    {x : Tensor ℝ s} {y : Tensor ℝ t} (hx : Encloses Bx (Graph.flatOfSome ⟨s, x⟩))
    (hy : ∀ lo hi, (∀ c, value (lo c) ≤ x c ∧ x c ≤ value (hi c)) →
      ∀ o b, bound lo hi o = some b → value b.1 ≤ y o ∧ y o ≤ value b.2)
    (h : coordwiseBox? bound Bx = some B) :
    Encloses B (Graph.flatOfSome ⟨t, y⟩) := by
  unfold coordwiseBox? at h
  split at h
  · rename_i hd
    obtain ⟨bounds, hb, h⟩ := Option.bind_eq_some_iff.mp h
    obtain rfl := Option.some.inj h
    refine ⟨rfl, fun (i : Fin t.size) => ?_⟩
    have hi := hy _ _ (encloses_reshape hx hd) _ _
      (Tensor.sequenceFinM_get_of_eq_some hb i)
    have hyi : getAtOrZero (Tensor.reshape y [t.size] (Shape.size_vector _).symm) [i.val] =
        y (Coord.unlinearize i) := by
      conv_lhs => rw [← Coord.linearize_unlinearize i]
      exact Tensor.getAtOrZero_reshape_toFin y _
    show value (getAtOrZero (Tensor.ofFn fun j => (bounds j).1) [i.val]) ≤
        getAtOrZero (Tensor.reshape y [t.size] (Shape.size_vector _).symm) [i.val] ∧
      getAtOrZero (Tensor.reshape y [t.size] (Shape.size_vector _).symm) [i.val] ≤
        value (getAtOrZero (Tensor.ofFn fun j => (bounds j).2) [i.val])
    rw [hyi, getAtOrZero_eq_getScalar, getAtOrZero_eq_getScalar, getScalar_ofFn, getScalar_ofFn]
    exact hi
  · cases h

/-- `poolSumBounds` encloses the window sum of a pooling of `x` when its endpoints
enclose `x` at every coordinate. -/
theorem poolSumBounds_encloses {s : Shape} {dims : List (Fin s.rank)}
    {kernel stride padLow padHigh dilation : List Nat} {ceilMode : Bool} {lo hi : Tensor α s}
    {x : Tensor ℝ s} (hx : ∀ c, value (lo c) ≤ x c ∧ x c ≤ value (hi c))
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) :
    value (poolSumBounds lo hi o).1 ≤
        (List.finRange (Shape.size (s.poolWindow dims kernel))).foldl
          (fun acc i => acc + Tensor.poolInput x o (Coord.unlinearize i)) 0 ∧
      (List.finRange (Shape.size (s.poolWindow dims kernel))).foldl
          (fun acc i => acc + Tensor.poolInput x o (Coord.unlinearize i)) 0 ≤
        value (poolSumBounds lo hi o).2 := by
  have hs := sum_encloses
    (fun i => Tensor.poolInput lo o (Coord.unlinearize i))
    (fun i => Tensor.poolInput hi o (Coord.unlinearize i))
    (fun i => Tensor.poolInput x o (Coord.unlinearize i))
    fun i => by
      unfold Tensor.poolInput
      rcases Tensor.poolSource o (Coord.unlinearize i) with _ | c
      · simp [LawfulBoundOps.toReal_zero (α := α)]
      · exact hx c
  rw [Fin.sum_univ_def, List.sum_eq_foldl, List.foldl_map] at hs
  exact hs

/-- A box of the width of `y` whose endpoints, read at the linear index of every coordinate `c`,
bound `y c` encloses the flat entries of `y`. -/
theorem encloses_of_coord {t : Shape} {B : FlatBox α} {y : Tensor ℝ t} (hd : B.dim = t.size)
    (hy : ∀ c : t.Coord, value (getAtOrZero B.lo [(Coord.linearize c).val]) ≤ y c ∧
      y c ≤ value (getAtOrZero B.hi [(Coord.linearize c).val])) :
    Encloses B (Graph.flatOfSome ⟨t, y⟩) := by
  refine ⟨hd, fun ⟨n, hn⟩ => ?_⟩
  have hn' : n < t.size := hn
  have h := hy (Coord.unlinearize ⟨n, hn'⟩)
  have e := Tensor.getAtOrZero_reshape_toFin y (Coord.unlinearize ⟨n, hn'⟩)
  rw [Coord.linearize_unlinearize] at h e
  simp only [Graph.flatOfSome]
  dsimp only at h e ⊢
  rw [e]
  exact h

/-- The endpoint box of an operator on one value port encloses the operator's real value `F x`
when the box `Bx` encloses `x`, for a rule at `α` that computes the monotone real map `F` of the
real values (`hnat`, `hmono`): the rule on the lower and on the upper endpoint bounds `F x`
coordinatewise. -/
theorem endpointsBox?_encloses {s t : Shape} {op : Op [.value s] [.value t]}
    {rule : Op.Rule α [.value s] [.value t]} {F : Tensor ℝ s → Tensor ℝ t}
    (hs : op.rule? (α := α) = some rule)
    (hnat : ∀ (x : Tensor α s) (c : t.Coord),
      value ((rule.forward (x, ())).1 c) = F (Tensor.map value x) c)
    (hmono : ∀ x y : Tensor ℝ s, (∀ c, x c ≤ y c) → ∀ c, F x c ≤ F y c)
    {Bx B : FlatBox α} {x : Tensor ℝ s}
    (hx : Encloses Bx (Graph.flatOfSome ⟨s, x⟩))
    (hB : endpointsBox? op Bx = some B) :
    Encloses B (Graph.flatOfSome ⟨t, F x⟩) := by
  unfold endpointsBox? at hB
  split at hB
  · rename_i hd
    dsimp only at hB
    split at hB
    · rename_i l u hl hu
      obtain ⟨_, xl, rl, -, hinl, hxl, hsl, -, hlo⟩ := Op.invoke_eq_ok hl
      obtain ⟨_, xu, ru, -, hinu, hxu, hsu, -, hhi⟩ := Op.invoke_eq_ok hu
      rw [hs] at hsl hsu
      obtain rfl := Option.some.inj hsl
      obtain rfl := Option.some.inj hsu
      cases hinl
      cases hinu
      simp only [Values.ofErased, Port.read, bind, Except.bind, pure, Except.pure, dite_true]
        at hxl hxu
      cases hxl
      cases hxu
      change [l] = [⟨_, (rule.forward _).1⟩] at hlo
      change [u] = [⟨_, (rule.forward _).1⟩] at hhi
      simp only [List.cons.injEq, and_true] at hlo hhi
      subst hlo hhi
      simp only [flatOf, Option.bind_eq_bind, Option.bind_some, dite_true,
        Option.some.injEq] at hB
      subst hB
      refine encloses_of_coord rfl fun c => ?_
      dsimp only
      rw [Tensor.getAtOrZero_reshape_toFin, Tensor.getAtOrZero_reshape_toFin, hnat, hnat]
      have hmap : ∀ (z : Tensor α [Bx.dim]) (c' : s.Coord),
          Tensor.map value (Tensor.reshape z s ((Shape.size_vector _).trans hd)) c' =
            value (Tensor.reshape z s ((Shape.size_vector _).trans hd) c') := fun _ _ => by
        simp only [Tensor.map_apply]
      exact ⟨hmono _ _ (fun c' => (hmap _ c').trans_le (encloses_reshape hx hd c').1) c,
        hmono _ _ (fun c' => (encloses_reshape hx hd c').2.trans (hmap _ c').symm.le) c⟩
    · cases hB
  · cases hB

/-- An operator on one value port whose rules at `α` and at `ℝ` agree through the real values
(`hnat`) and whose real forward is monotone (`hmono`): at a node of a run whose reference's box
encloses the value it reads, the endpoint box `endpointsBox?` encloses the node's flat
value. -/
theorem encloses_endpoints {s t : Shape} {op : Op [.value s] [.value t]}
    {rule : Op.Rule α [.value s] [.value t]}
    {ruleR : Op.Rule ℝ [.value s] [.value t]}
    (hs : op.rule? (α := α) = some rule) (hsR : op.rule? (α := ℝ) = some ruleR)
    (hnat : ∀ (x : Tensor α s) (c : t.Coord),
      value ((rule.forward (x, ())).1 c) = (ruleR.forward (Tensor.map value x, ())).1 c)
    (hmono : ∀ x y : Tensor ℝ s, (∀ c, x c ≤ y c) →
      ∀ c, (ruleR.forward (x, ())).1 c ≤ (ruleR.forward (y, ())).1 c)
    {boxes : Array (Option (FlatBox α))} {earlier : Array (List (Op.Value ℝ))}
    {refs : Array IR.Ref} {B : FlatBox α} {outs : List (Op.Value ℝ)}
    (hpe : ∀ r ∈ refs.toList, ∀ (Br : FlatBox α) (w : Op.Value ℝ) (v : FlatTensor ℝ),
      box? boxes r = some Br → r.read earlier = .ok w → flatOf w = some v → Encloses Br v)
    (hc : ∀ r, refs.toList = [r] → (box? boxes r >>= endpointsBox? op) = some B)
    (h : op.invoke (refs.toList.mapM (·.read earlier)) = .ok outs) :
    ∀ v, (outs[0]? >>= flatOf) = some v → Encloses B v := by
  obtain ⟨r, w, x, hr, hread, hx, rfl⟩ := invoke_one hsR h
  obtain ⟨Br, hBr, hB⟩ := Option.bind_eq_some_iff.mp (hc r hr)
  rw [hr] at hpe
  intro v hv
  simp only [List.getElem?_cons_zero, Option.bind_eq_bind, Option.bind_some, flatOf_value,
    Option.some.injEq] at hv
  subst hv
  exact endpointsBox?_encloses hs hnat hmono
    (hpe r (by simp) Br w _ hBr hread ((flatOf_of_read (Or.inl rfl) hx).trans (flatOf_value x))) hB

/-- The maximum's choice between `a` and a candidate `b` denotes the real maximum for every tie
test `tie`: under the order law, two mutually non-greater values denote the same real number. -/
theorem _root_.TorchLean.CROWN.Directed.value_selectedMax {tie : Bool} (a b : α) :
    value (if b > a || (!(a > b) && tie) then b else a) =
      max (value a) (value b) := by
  by_cases hab : b > a
  · simp only [hab, decide_true, Bool.true_or, ↓reduceIte]
    exact (max_eq_right (le_of_lt ((LawfulBoundOps.lt_iff a b).mp hab))).symm
  · by_cases hba : a > b
    · simp only [hab, hba, decide_false, decide_true, Bool.not_true, Bool.false_and,
        Bool.false_or]
      exact (max_eq_left (le_of_lt ((LawfulBoundOps.lt_iff b a).mp hba))).symm
    · have hle : value a ≤ value b := not_lt.mp fun h => hba ((LawfulBoundOps.lt_iff b a).mpr h)
      have hge : value b ≤ value a := not_lt.mp fun h => hab ((LawfulBoundOps.lt_iff a b).mpr h)
      have heq := le_antisymm hle hge
      split <;> simp [heq]

/-- The minimum's choice between `a` and a candidate `b` denotes the real minimum for every tie
test `tie`: under the order law, two mutually non-greater values denote the same real number. -/
theorem value_selectedMin {tie : Bool} (a b : α) :
    value (if a > b || (!(b > a) && tie) then b else a) =
      min (value a) (value b) := by
  by_cases hab : a > b
  · simp only [hab, decide_true, Bool.true_or, ↓reduceIte]
    exact (min_eq_right (le_of_lt ((LawfulBoundOps.lt_iff b a).mp hab))).symm
  · by_cases hba : b > a
    · simp only [hab, hba, decide_false, decide_true, Bool.not_true, Bool.false_and,
        Bool.false_or]
      exact (min_eq_left (le_of_lt ((LawfulBoundOps.lt_iff a b).mp hba))).symm
    · have hle : value a ≤ value b := not_lt.mp fun h => hab ((LawfulBoundOps.lt_iff b a).mpr h)
      have hge : value b ≤ value a := not_lt.mp fun h => hba ((LawfulBoundOps.lt_iff a b).mpr h)
      have heq := le_antisymm hle hge
      split <;> simp [heq]

/-- The coordinates that a maximum (`largest`) or minimum pool selects in two tensors read related
entries, for a relation between the tensors' entries at each coordinate that every choice of
`poolWins` keeps. -/
theorem _root_.TorchLean.Tensor.poolIndices_rel {β γ : Type} [Storage β] [Context β]
    [Storage γ] [Context γ]
    {s : Shape} {R : β → γ → Prop} {largest : Bool} {x : Tensor β s} {y : Tensor γ s}
    {dims : List (Fin s.rank)} {kernel stride padLow padHigh dilation : List Nat}
    {ceilMode : Bool} (hxy : ∀ c, R (x c) (y c))
    (hwins : ∀ a b c d, R a c → R b d →
      R (if poolWins largest a b then a else b) (if poolWins largest c d then c else d))
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) :
    Option.Rel (fun c d => R (x c) (y d))
      (poolIndices largest x dims kernel stride padLow padHigh dilation ceilMode o)
      (poolIndices largest y dims kernel stride padLow padHigh dilation ceilMode o) := by
  unfold poolIndices
  rw [Tensor.Rep.get_ofFn, Tensor.Rep.get_ofFn]
  refine List.foldl_rel (r := Option.Rel fun c d => R (x c) (y d)) .none fun i _ p q hpq => ?_
  cases poolSource o (Coord.unlinearize i) with
  | none => exact hpq
  | some c =>
    cases hpq with
    | none => exact .some (hxy c)
    | @some b d hbd =>
      have h := hwins _ _ _ _ (hxy c) hbd
      dsimp only
      split_ifs at h ⊢ <;> exact .some h

/-- Maximum or minimum pools of two tensors are related entrywise, for a relation between the
tensors' entries at each coordinate that relates the zeros and that every choice of `poolWins`
keeps. -/
theorem _root_.TorchLean.Tensor.pool_rel {β γ : Type} [Storage β] [Context β]
    [Storage γ] [Context γ]
    {s : Shape} {R : β → γ → Prop} {op : PoolOp} (hop : op = .max ∨ op = .min)
    {x : Tensor β s} {y : Tensor γ s} {dims : List (Fin s.rank)}
    {kernel stride padLow padHigh dilation : List Nat} {ceilMode : Bool} (h0 : R 0 0)
    (hxy : ∀ c, R (x c) (y c))
    (hwins : ∀ largest a b c d, R a c → R b d →
      R (if poolWins largest a b then a else b) (if poolWins largest c d then c else d))
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) :
    R (pool op x dims kernel stride padLow padHigh dilation ceilMode o)
      (pool op y dims kernel stride padLow padHigh dilation ceilMode o) := by
  rcases hop with rfl | rfl
  · rw [pool_max_apply, pool_max_apply]
    have h := poolIndices_rel hxy (hwins true) o
    generalize poolIndices true x dims kernel stride padLow padHigh dilation ceilMode o = p at h ⊢
    generalize poolIndices true y dims kernel stride padLow padHigh dilation ceilMode o = q at h ⊢
    cases h <;> assumption
  · rw [pool_min_apply, pool_min_apply]
    have h := poolIndices_rel hxy (hwins false) o
    generalize poolIndices false x dims kernel stride padLow padHigh dilation ceilMode o = p at h ⊢
    generalize poolIndices false y dims kernel stride padLow padHigh dilation ceilMode o = q at h ⊢
    cases h <;> assumption

/-- A maximum or minimum pool commutes with `value` on every tensor: at ties and NaN-like entries
the entry it selects at `α` and the one it selects among the real values may differ, but they
denote the same real number. -/
theorem value_pool {s : Shape} {op : PoolOp} (hop : op = .max ∨ op = .min)
    {dims : List (Fin s.rank)} {kernel stride padLow padHigh dilation : List Nat}
    {ceilMode : Bool} (x : Tensor α s)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) :
    value (pool op x dims kernel stride padLow padHigh dilation ceilMode o) =
      pool op (Tensor.map value x) dims kernel stride padLow padHigh dilation ceilMode o := by
  refine pool_rel (R := fun a r => value a = r) hop LawfulBoundOps.toReal_zero
    (fun c => (Tensor.map_apply _ _ c).symm) ?_ o
  rintro (_ | _) a b _ _ rfl rfl
  · exact (value_selectedMin b a).trans (value_selectedMin (α := ℝ) _ _).symm
  · exact (value_selectedMax b a).trans (value_selectedMax (α := ℝ) _ _).symm

/-- A real maximum or minimum pool is monotone. -/
theorem _root_.TorchLean.Tensor.pool_mono {s : Shape} {op : PoolOp} (hop : op = .max ∨ op = .min)
    {dims : List (Fin s.rank)} {kernel stride padLow padHigh dilation : List Nat}
    {ceilMode : Bool} {x y : Tensor ℝ s} (hxy : ∀ c, x c ≤ y c)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) :
    pool op x dims kernel stride padLow padHigh dilation ceilMode o ≤
      pool op y dims kernel stride padLow padHigh dilation ceilMode o := by
  refine pool_rel hop le_rfl hxy ?_ o
  rintro (_ | _) a b c d hac hbd
  · exact ((value_selectedMin (α := ℝ) b a).trans_le (min_le_min hbd hac)).trans_eq
      (value_selectedMin (α := ℝ) d c).symm
  · exact ((value_selectedMax (α := ℝ) b a).trans_le (max_le_max hbd hac)).trans_eq
      (value_selectedMax (α := ℝ) d c).symm

/-- The box `poolBox?` of a maximum or minimum pool, the pool of each endpoint, encloses
the real pool of every tensor that the input box encloses. -/
theorem poolBox?_encloses [NonlinearBoundOps α] {s : Shape} {op : PoolOp}
    (hop : op = .max ∨ op = .min) {dims : List (Fin s.rank)}
    {kernel stride padLow padHigh dilation : List Nat} {ceilMode : Bool} {Bx B : FlatBox α}
    {x : Tensor ℝ s} (hx : Encloses Bx (Graph.flatOfSome ⟨s, x⟩))
    (hB : poolBox? op dims kernel stride padLow padHigh dilation ceilMode Bx = some B) :
    Encloses B (Graph.flatOfSome
      ⟨_, Tensor.pool op x dims kernel stride padLow padHigh dilation ceilMode⟩) := by
  refine endpointsBox?_encloses
    (Op.rule?_pool op dims kernel stride padLow padHigh dilation ceilMode)
    (F := (Tensor.pool op · dims kernel stride padLow padHigh dilation ceilMode))
    (value_pool hop) (fun _ _ hxy => pool_mono hop hxy) hx ?_
  rcases hop with rfl | rfl <;> exact hB

/-- Directed negation encloses absolute value without assuming that ordinary abs is exact. -/
theorem _root_.TorchLean.CROWN.Directed.abs_encloses
    {lo hi : α} {x : ℝ} (hlo : value lo ≤ x) (hhi : x ≤ value hi) :
    value (if lo < 0 then if 0 < hi then 0 else BoundOps.subDown 0 hi else lo) ≤ |x| ∧
      |x| ≤ value (Context.maximum (BoundOps.subUp 0 lo) hi) := by
  constructor
  · split_ifs
    · simpa only [(LawfulBoundOps.toReal_zero (α := α))] using abs_nonneg x
    · calc
        value (BoundOps.subDown 0 hi) ≤ -value hi := by
          simpa only [(LawfulBoundOps.toReal_zero (α := α)), zero_sub] using
            LawfulBoundOps.subDown_le (0 : α) hi
        _ ≤ -x := neg_le_neg hhi
        _ ≤ |x| := neg_le_abs x
    · exact hlo.trans (le_abs_self x)
  · rw [LawfulBoundOps.toReal_maximum]
    apply abs_le.mpr
    constructor
    · calc
        -max (value (BoundOps.subUp 0 lo)) (value hi) ≤
            -value (BoundOps.subUp 0 lo) := neg_le_neg (le_max_left _ _)
        _ ≤ value lo := by
          have h := LawfulBoundOps.le_subUp (0 : α) lo
          rw [(LawfulBoundOps.toReal_zero (α := α)), zero_sub] at h
          linarith
        _ ≤ x := hlo
    · exact hhi.trans (le_max_right _ _)

/-- The directed absolute-value box encloses the absolute value of every coordinate. -/
theorem _root_.TorchLean.CROWN.Directed.boxAbs_encloses
    {x : FlatBox α} {n : Nat} {f : Nat → ℝ} (hx : RowEncloses x n f) :
    RowEncloses (boxAbs x) n (fun i => |f i|) := by
  obtain ⟨d, lo, hi⟩ := x
  have hd := hx.1
  change d = n at hd
  subst hd
  rw [rowEncloses_iff] at hx
  simp only [boxAbs, Graph.absBound]
  rw [rowEncloses_iff]
  intro i
  simpa only [getScalar_ofFn] using abs_encloses (hx i).1 (hx i).2

/-- Componentwise maximum preserves enclosure in both operands. -/
theorem _root_.TorchLean.CROWN.Directed.boxMaxElem_encloses
    {x y : FlatBox α} {n : Nat} {f g : Nat → ℝ}
    (hx : RowEncloses x n f) (hy : RowEncloses y n g) :
    RowEncloses (boxMaxElem x y) n (fun i => max (f i) (g i)) := by
  obtain ⟨dx, lx, ux⟩ := x
  obtain ⟨dy, ly, uy⟩ := y
  have hdx := hx.1
  have hdy := hy.1
  change dx = n at hdx
  change dy = n at hdy
  subst hdx hdy
  rw [rowEncloses_iff] at hx hy
  simp only [boxMaxElem, dite_true]
  rw [rowEncloses_iff]
  intro i
  simp only [getScalar_zipWith, LawfulBoundOps.toReal_max]
  exact ⟨max_le_max (hx i).1 (hy i).1, max_le_max (hx i).2 (hy i).2⟩

/-- Componentwise minimum preserves enclosure in both operands. -/
theorem _root_.TorchLean.CROWN.Directed.boxMinElem_encloses
    {x y : FlatBox α} {n : Nat} {f g : Nat → ℝ}
    (hx : RowEncloses x n f) (hy : RowEncloses y n g) :
    RowEncloses (boxMinElem x y) n (fun i => min (f i) (g i)) := by
  obtain ⟨dx, lx, ux⟩ := x
  obtain ⟨dy, ly, uy⟩ := y
  have hdx := hx.1
  have hdy := hy.1
  change dx = n at hdx
  change dy = n at hdy
  subst hdx hdy
  rw [rowEncloses_iff] at hx hy
  simp only [boxMinElem, dite_true]
  rw [rowEncloses_iff]
  intro i
  simp only [getScalar_zipWith, LawfulBoundOps.toReal_min]
  exact ⟨min_le_min (hx i).1 (hy i).1, min_le_min (hx i).2 (hy i).2⟩

/-- The coarse softplus interval encloses `log (1 + exp x)` using only directed addition. -/
theorem _root_.TorchLean.CROWN.Directed.softplusBounds_enclosure :
    UnaryEnclosure (α := α) (fun x => Real.log (1 + Real.exp x))
      NonlinearBoundOps.softplusBounds := by
  intro lo hi outLo outHi x hout hxlo hxhi
  simp only [NonlinearBoundOps.softplusBounds, Option.some.injEq, Prod.mk.injEq] at hout
  obtain ⟨rfl, rfl⟩ := hout
  have hsoft := Activation.softplus_envelope_real x
  rw [Activation.softplus_eq_log_one_add_exp] at hsoft
  constructor
  · simpa only [LawfulBoundOps.toReal_maximum, (LawfulBoundOps.toReal_zero (α := α))] using
      (max_le_max hxlo le_rfl).trans hsoft.1
  · calc
      Real.log (1 + Real.exp x) ≤ max x 0 + 1 := hsoft.2
      _ ≤ max (value hi) 0 + 1 := add_le_add (max_le_max hxhi le_rfl) le_rfl
      _ = value (Context.maximum hi 0) + value (1 : α) := by
        rw [LawfulBoundOps.toReal_maximum, (LawfulBoundOps.toReal_zero (α := α)),
          (LawfulBoundOps.toReal_one (α := α))]
      _ ≤ value (BoundOps.addUp (Context.maximum hi 0) 1) :=
        LawfulBoundOps.le_addUp _ _

/-- A reference that names a rounded constant at a port reads, at that port, the constant's real
carrier. -/
theorem read_constAt?_carrier {G : IR.Graph} {P : IR.Payload α}
    {earlier : Array (List (Op.Value ℝ))} {refs : List IR.Ref}
    (hc : ConstsReal G P earlier refs) {r : IR.Ref} (hr : r ∈ refs) {p : Port}
    {x : p.Carrier α} (h : constAt? G P r p = some x) {w : Op.Value ℝ}
    (hw : r.read earlier = .ok w) {y : p.Carrier ℝ} (hy : p.read w = .ok y) :
    y = realCarrier p x := by
  unfold constAt? at h
  by_cases h0 : r.out = 0
  · simp only [h0, ↓reduceIte] at h
    cases hcv : constValue? G P r.node with
    | none => simp [hcv] at h
    | some v =>
      simp only [hcv, Option.bind_eq_bind, Option.bind_some] at h
      obtain rfl := hc r hr w hw h0 v hcv
      rw [read_realValue] at hy
      cases hrv : p.read v with
      | error e => simp [hrv, Except.map] at hy
      | ok z =>
        simp only [hrv, Except.toOption, Option.some.injEq] at h
        subst h
        simp only [hrv, Except.map, Except.ok.injEq] at hy
        exact hy.symm
  · simp [h0] at h

omit [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α] in
/-- A successful read of references at the ports `p :: ps` reads the first reference at `p` and
the others at `ps`. -/
theorem read_cons {p : Port} {ps : List Port} {refs : List IR.Ref}
    {earlier : Array (List (Op.Value ℝ))} {values : List (Op.Value ℝ)}
    {xs : Values ℝ (p :: ps)} (hin : refs.mapM (·.read earlier) = .ok values)
    (hx : Values.ofErased (p :: ps) values = .ok xs) :
    ∃ r rest v vs, refs = r :: rest ∧ r.read earlier = .ok v ∧ p.read v = .ok xs.1 ∧
      rest.mapM (·.read earlier) = .ok vs ∧ Values.ofErased ps vs = .ok xs.2 := by
  match refs, hin with
  | [], hin =>
    simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at hin
    subst hin
    simp [Values.ofErased, throw, throwThe, MonadExceptOf.throw] at hx
  | r :: rest, hin =>
    simp only [List.mapM_cons, bind, Except.bind] at hin
    cases hr : r.read earlier with
    | error e => simp [hr] at hin
    | ok v =>
      cases hm : rest.mapM (·.read earlier) with
      | error e => simp [hr, hm] at hin
      | ok vs =>
        simp only [hr, hm, pure, Except.pure, Except.ok.injEq] at hin
        subst hin
        simp only [Values.ofErased, bind, Except.bind] at hx
        cases hp : p.read v with
        | error e => simp [hp] at hx
        | ok y =>
          simp only [hp] at hx
          cases hrest : Values.ofErased ps vs with
          | error e => simp [hrest] at hx
          | ok ys =>
            simp only [hrest, pure, Except.pure] at hx
            cases hx
            exact ⟨r, rest, v, vs, rfl, hr, hp, hm, hrest⟩

omit [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α] in
/-- A successful read of references at no ports reads no references. -/
theorem read_nil {refs : List IR.Ref} {earlier : Array (List (Op.Value ℝ))}
    {values : List (Op.Value ℝ)} {xs : Values ℝ []}
    (hin : refs.mapM (·.read earlier) = .ok values) (hx : Values.ofErased [] values = .ok xs) :
    refs = [] := by
  match refs, hin with
  | [], _ => rfl
  | r :: rest, hin =>
    simp only [List.mapM_cons, bind, Except.bind] at hin
    cases hr : r.read earlier with
    | error e => simp [hr] at hin
    | ok v =>
      cases hm : rest.mapM (·.read earlier) with
      | error e => simp [hr, hm] at hin
      | ok vs =>
        simp only [hr, hm, pure, Except.pure, Except.ok.injEq] at hin
        subst hin
        simp [Values.ofErased, throw, throwThe, MonadExceptOf.throw] at hx

/-- An operator with a frozen scalar parameter whose real forward is the pointwise map `f c` at
the parameter `c`, with a transfer `T c` enclosing `F (value c) = f (value c)` coordinatewise for
each rounded parameter `c`: at a node of a run whose references' boxes enclose the values they
read, and whose parameter reference names a constant, the transfer's box encloses the node's flat
value. -/
theorem encloses_param {s : Shape} {op : Op [.value s, .frozen []] [.value s]}
    {rule : Op.Rule ℝ [.value s, .frozen []] [.value s]}
    (hs : op.rule? (α := ℝ) = some rule) {f F : ℝ → ℝ → ℝ}
    (hforward : ∀ x c, (rule.forward (x, c, ())).1 = Tensor.map (f (Tensor.item c)) x)
    (hf : ∀ a z, f a z = F a z) {T : α → FlatBox α → Option (FlatBox α)}
    (hT : ∀ {c : α} {Bx B : FlatBox α} {g : Nat → ℝ}, RowEncloses Bx s.size g →
      T c Bx = some B → RowEncloses B s.size fun i => F (value c) (g i))
    {G : IR.Graph} {P : IR.Payload α}
    {boxes : Array (Option (FlatBox α))} {earlier : Array (List (Op.Value ℝ))}
    {refs : Array IR.Ref} {B : FlatBox α} {outs : List (Op.Value ℝ)}
    (hconsts : ConstsReal G P earlier refs.toList)
    (hpe : ∀ r ∈ refs.toList, ∀ (Br : FlatBox α) (w : Op.Value ℝ) (v : FlatTensor ℝ),
      box? boxes r = some Br → r.read earlier = .ok w → flatOf w = some v → Encloses Br v)
    (hc : ∀ rx rc, refs.toList = [rx, rc] → (do
        let c ← constAt? G P rc (.frozen [])
        box? boxes rx >>= T (Tensor.item c)) = some B)
    (h : op.invoke (refs.toList.mapM (·.read earlier)) = .ok outs) :
    ∀ v, (outs[0]? >>= flatOf) = some v → Encloses B v := by
  obtain ⟨rx, rc, wx, wc, x, c, hr, hreadx, hreadc, hx, hcx, rfl⟩ := invoke_two hs h
  have hc' := hc rx rc hr
  simp only [Option.bind_eq_bind] at hc'
  obtain ⟨c', hc'', hc'⟩ := Option.bind_eq_some_iff.mp hc'
  obtain ⟨Bx, hBx, hT'⟩ := Option.bind_eq_some_iff.mp hc'
  rw [hr] at hpe hconsts
  obtain rfl := read_constAt?_carrier hconsts (by simp) hc'' hreadc hcx
  intro v hv
  simp only [List.getElem?_cons_zero, Option.bind_eq_bind, Option.bind_some, hforward,
    flatOf_value, Option.some.injEq] at hv
  subst hv
  simp only [realCarrier, item_map]
  exact encloses_map (hf _) (hpe rx (by simp) Bx wx _ hBx hreadx
    ((flatOf_of_read (Or.inl rfl) hx).trans (flatOf_value x))) fun hg => hT hg hT'

/-- At `ℝ`, clamping to the scalar bounds `a` and `b` is the pointwise `min (max · a) b`. -/
theorem clamp_full_real {s : Shape} (x : Tensor ℝ s) (a b : ℝ) :
    Tensor.clamp x (Tensor.full s a) (Tensor.full s b) .interior =
      Tensor.map (fun z => min (max z a) b) x := by
  apply Tensor.Rep.ext
  intro i
  simp only [Tensor.clamp_interior_apply, Tensor.full_apply, Context.clamp_real,
    Tensor.map_apply]
  split_ifs <;> rfl

/-- Every reference of a list whose boxes all exist names output `0`. -/
theorem out_eq_zero_of_mapM_box? {β : Type} {a : Array (Option β)} :
    ∀ {refs : List IR.Ref} {xs : List β}, refs.mapM (box? a) = some xs → ∀ r ∈ refs, r.out = 0
  | [], _, _, r, hr => by simp at hr
  | r :: rs, xs, h, r', hr' => by
      simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
        Option.some.injEq] at h
      obtain ⟨x, hx, ys, hys, rfl⟩ := h
      rcases List.mem_cons.mp hr' with rfl | hr'
      · unfold box? at hx
        split at hx
        · assumption
        · cases hx
      · exact out_eq_zero_of_mapM_box? hys r' hr'

/-- A checked concatenation of boxes encloses the checked concatenation of real flat values over
the same layout when the box at each position encloses the value at that position. -/
theorem encloses_concat {layout : ConcatLayout} {boxes : Array (FlatBox α)}
    {values : Array (FlatTensor ℝ)} {B : FlatBox α} {v : FlatTensor ℝ}
    (hB : concatFlatBoxes? layout boxes = some B)
    (hv : concatFlatValues? layout values = some v)
    (hparents : ∀ (i : Nat) (Bp : FlatBox α) (vp : FlatTensor ℝ), boxes[i]? = some Bp →
      values[i]? = some vp → Encloses Bp vp) :
    Encloses B v := by
  unfold concatFlatValues? at hv
  split at hv
  · obtain ⟨inputs, hinputs, hv⟩ := Option.bind_eq_some_iff.mp hv
    obtain ⟨hshape, hv⟩ := Option.dite_none_right_eq_some.mp hv
    obtain rfl := Option.some.inj hv
    unfold Encloses
    refine (concatFlatBoxes?_encloses (f := fun p j => getAtOrZero (inputs p).v [j]) hB
      fun p Bp hBp => rowEncloses_width (hshape p)
        (hparents p.val Bp (inputs p) hBp
          (Tensor.sequenceFinM_get_of_eq_some hinputs p))).congr fun i => ?_
    simp only [realConcat, ConcatLayout.concat, Tensor.getAtOrZero_eq_getScalar, getScalar_ofFn]
    rw [← Tensor.getAtOrZero_eq_getScalar, getAtOrZero_eqRec]
  · contradiction

/-- The interval box of `Op.conv` encloses its forward value at the real values of the rounded
kernel and bias, on an input whose flattening the input box encloses. -/
theorem encloses_conv {s k : Shape} {r : Nat} {dims : Shape.ConvDims s k r}
    {stride padLow padHigh dilation : Fin r → Nat} {groups : Nat} {lead x y : Shape}
    {hlead : s.extent (dims.input ⟨0, by omega⟩) = lead.size}
    {hx : x = s.expandAxis (dims.input ⟨0, by omega⟩) lead}
    {hy : y = (s.conv k dims stride padLow padHigh (fun _ => 1) dilation 1).expandAxis
      (dims.output stride padLow padHigh (fun _ => 1) dilation 1 ⟨0, by omega⟩) lead}
    {K : Tensor α k} {b : Tensor α [k.extent (dims.kernel ⟨0, by omega⟩)]} {X B : FlatBox α}
    (hB : convOpBox? dims stride padLow padHigh dilation groups lead K b X = some B)
    (xt : Tensor ℝ x) (hX : Encloses X (flatOfSome ⟨x, xt⟩)) {rule : Op.Rule ℝ _ _}
    (hs : (Op.conv dims stride padLow padHigh dilation groups hlead hx hy).rule? (α := ℝ) =
      some rule) :
    Encloses B
      (flatOfSome ⟨y, (rule.forward (Tensor.map value K, Tensor.map value b, xt, ())).1⟩) := by
  obtain ⟨_, _, _, _, _, _, _, _, _, _, _, h, hB, hv⟩ := convBox?_of_convOpBox? value hB xt hs
  rw [hv]
  have hX' : Encloses X (flatOfSome ⟨_, Tensor.reshape xt _ h⟩) := by
    rw [flatOfSome_reshape]
    exact hX
  exact Directed.convBox?_encloses hB (Tensor.reshape xt _ h) hX'

/-- `cumsumBounds dim lo hi c` encloses entry `c` of the cumulative sum of `x` along
`dim` when `lo` and `hi` enclose `x` at every coordinate. -/
theorem cumsumBounds_encloses {s : Shape} (dim : Fin s.rank) {lo hi : Tensor α s}
    {x : Tensor ℝ s} (hx : ∀ c, value (lo c) ≤ x c ∧ x c ≤ value (hi c)) (c : s.Coord)
    {b : α × α} (h : cumsumBounds dim lo hi c = some b) :
    value b.1 ≤ Tensor.cumsum x dim c ∧ Tensor.cumsum x dim c ≤ value b.2 := by
  obtain rfl := Option.some.inj h
  have hs := sum_encloses
    (fun i : Fin ((Shape.Coord.get c dim).val + 1) =>
      lo (Shape.Coord.update c dim (Fin.castLE (Shape.Coord.get c dim).isLt i)))
    (fun i => hi (Shape.Coord.update c dim (Fin.castLE (Shape.Coord.get c dim).isLt i)))
    (fun i => x (Shape.Coord.update c dim (Fin.castLE (Shape.Coord.get c dim).isLt i)))
    fun i => hx _
  rw [Fin.sum_univ_def, List.sum_eq_foldl, List.foldl_map] at hs
  have hfold : ∀ g : Fin ((Shape.Coord.get c dim).val + 1) → ℝ,
      (List.finRange _).foldl (fun acc i => acc + g i) 0 =
        Tensor.foldl (· + ·) 0 (Tensor.ofFn g) := by
    intro g
    rw [Tensor.ofFn, Rep.ofFn_eq_map_coordinates, Rep.foldl_map,
      ← Fin.foldl_eq_foldl_finRange, fin_foldl_eq_coordinates]
  rw [Tensor.cumsum_apply, ← hfold]
  exact hs

/-- `selectBox?` encloses `select c x y` for every condition `c` when its two boxes
enclose the flat entries of `x` and of `y`. -/
theorem selectBox?_encloses {s : Shape} (c : Tensor Bool s) {x y : Tensor ℝ s}
    {Bx By B : FlatBox α} (hx : Encloses Bx (Graph.flatOfSome ⟨s, x⟩))
    (hy : Encloses By (Graph.flatOfSome ⟨s, y⟩)) (h : selectBox? Bx By = some B) :
    Encloses B (Graph.flatOfSome ⟨s, Tensor.select c x y⟩) := by
  have hread : ∀ (z : Tensor ℝ s) (i : Fin s.size),
      getAtOrZero (Graph.flatOfSome ⟨s, z⟩).v [i.val] = z (Coord.unlinearize i) := by
    intro z i
    show getAtOrZero (Tensor.reshape z [s.size] (Shape.size_vector _).symm) [i.val] = _
    rw [getAtOrZero_eq_getScalar]
    conv_lhs => rw [← Coord.linearize_unlinearize i]
    exact Tensor.getScalar_reshape_toFin z _
  obtain ⟨d₁, lo₁, hi₁⟩ := Bx
  obtain ⟨d₂, lo₂, hi₂⟩ := By
  have hd₁ : d₁ = s.size := hx.1
  have hd₂ : d₂ = s.size := hy.1
  subst hd₁ hd₂
  simp only [selectBox?, dite_true, Option.some.injEq] at h
  subst h
  have hx' := (rowEncloses_iff (α := α)).mp hx
  have hy' := (rowEncloses_iff (α := α)).mp hy
  refine (rowEncloses_iff (α := α)).mpr fun i => ?_
  have hxi := hx' i
  have hyi := hy' i
  rw [hread] at hxi hyi ⊢
  simp only [getScalar_ofFn, Fin.cast_eq_self, LawfulBoundOps.toReal_minimum,
    LawfulBoundOps.toReal_maximum, Tensor.select_apply]
  split_ifs
  · exact ⟨(min_le_left _ _).trans hxi.1, hxi.2.trans (le_max_left _ _)⟩
  · exact ⟨(min_le_right _ _).trans hyi.1, hyi.2.trans (le_max_right _ _)⟩

/-- `padBox?` encloses the padding of `x` with `v` when its two boxes enclose the flat
entries of `x` and of `v`. -/
theorem padBox?_encloses {s : Shape} (low high interior : Fin s.rank → Nat) {x : Tensor ℝ s}
    {v : Tensor ℝ []} {Bx Bv B : FlatBox α} (hx : Encloses Bx (Graph.flatOfSome ⟨s, x⟩))
    (hv : Encloses Bv (Graph.flatOfSome ⟨[], v⟩))
    (h : padBox? low high interior Bx Bv = some B) :
    Encloses B (Graph.flatOfSome
      ⟨s.pad low high interior, Tensor.pad x low high interior v.item⟩) := by
  unfold padBox? at h
  split at h
  · rename_i hdx
    split at h
    · rename_i hdv
      obtain rfl := Option.some.inj h
      refine ⟨rfl, fun (i : Fin (s.pad low high interior).size) => ?_⟩
      have hyi : getAtOrZero (Tensor.reshape (Tensor.pad x low high interior v.item)
          [(s.pad low high interior).size] (Shape.size_vector _).symm) [i.val] =
          Tensor.pad x low high interior v.item (Coord.unlinearize i) := by
        rw [getAtOrZero_eq_getScalar]
        conv_lhs => rw [← Coord.linearize_unlinearize i]
        exact Tensor.getScalar_reshape_toFin _ _
      show value (getAtOrZero (Tensor.ofFn fun j => _) [i.val]) ≤
          getAtOrZero (Tensor.reshape (Tensor.pad x low high interior v.item)
            [(s.pad low high interior).size] (Shape.size_vector _).symm) [i.val] ∧
        getAtOrZero (Tensor.reshape (Tensor.pad x low high interior v.item)
            [(s.pad low high interior).size] (Shape.size_vector _).symm) [i.val] ≤
          value (getAtOrZero (Tensor.ofFn fun j => _) [i.val])
      rw [hyi, getAtOrZero_eq_getScalar, getAtOrZero_eq_getScalar, getScalar_ofFn, getScalar_ofFn]
      simp only [Tensor.pad_apply]
      split_ifs
      · exact encloses_reshape hx hdx _
      · exact encloses_reshape hv hdv PUnit.unit
    · cases h
  · cases h

/-- `gatherBound` encloses every entry of `x` that `gather` can read at `o`, when `lo`
and `hi` enclose `x`. -/
theorem gatherBound_encloses {s t : Shape} {dims : List (Fin s.rank)}
    {batchDims : List (Fin s.rank × Fin t.rank)} (h : Shape.GatherValid s t dims batchDims)
    {x : Tensor ℝ s} {lo hi : Tensor α s} (hx : ∀ c, value (lo c) ≤ x c ∧ x c ≤ value (hi c))
    (p : Einops.AxisTuple s.extent dims)
    (o : (t ++ s.eraseAxes (dims ++ batchDims.map Prod.fst)).Coord) {b : α × α}
    (hb : gatherBound h lo hi o = some b) :
    let parts := Coord.appendEquiv t
      (s.eraseAxes (dims ++ batchDims.map Prod.fst)) o
    value b.1 ≤ x (h.source p parts.1 parts.2) ∧ x (h.source p parts.1 parts.2) ≤ value b.2 := by
  intro parts
  set a₀ := h.source p parts.1 parts.2 with ha₀
  set v := x a₀
  let Q : Option (α × α) → Prop := fun acc =>
    ∃ l u, acc = some (l, u) ∧ value l ≤ v ∧ v ≤ value u
  let f : Option (α × α) → Fin s.size → Option (α × α) := fun acc j =>
    let a : s.Coord := Coord.unlinearize j
    if Coord.linearize (h.source (axisRead dims a) parts.1 parts.2) = j then
      match acc with
      | none => some (lo a, hi a)
      | some (l, u) => some (Context.minimum l (lo a), Context.maximum u (hi a))
    else acc
  have hstep : ∀ acc j, Q acc → Q (f acc j) := by
    rintro acc j ⟨l, u, rfl, hl, hu⟩
    simp only [f]
    split_ifs
    · refine ⟨_, _, rfl, ?_, ?_⟩
      · rw [LawfulBoundOps.toReal_minimum]; exact (min_le_left _ _).trans hl
      · rw [LawfulBoundOps.toReal_maximum]; exact hu.trans (le_max_left _ _)
    · exact ⟨l, u, rfl, hl, hu⟩
  have hhit : ∀ acc, Q (f acc (Coord.linearize a₀)) := by
    intro acc
    have hread : axisRead dims a₀ = p := axisRead_source h p _ _
    have hxa := hx a₀
    simp only [f, Coord.unlinearize_linearize, hread, ← ha₀, ↓reduceIte]
    cases acc with
    | none => exact ⟨_, _, rfl, hxa.1, hxa.2⟩
    | some lu =>
      obtain ⟨l, u⟩ := lu
      refine ⟨_, _, rfl, ?_, ?_⟩
      · rw [LawfulBoundOps.toReal_minimum]; exact (min_le_right _ _).trans hxa.1
      · rw [LawfulBoundOps.toReal_maximum]; exact hxa.2.trans (le_max_right _ _)
  have hmono : ∀ (L : List (Fin s.size)) acc, Q acc → Q (L.foldl f acc) := by
    intro L
    induction L with
    | nil => intro acc hq; exact hq
    | cons j L ih => intro acc hq; exact ih _ (hstep acc j hq)
  have hfold : ∀ (L : List (Fin s.size)) acc, Coord.linearize a₀ ∈ L → Q (L.foldl f acc) := by
    intro L
    induction L with
    | nil => intro acc hm; simp at hm
    | cons j L ih =>
      intro acc hm
      rcases List.mem_cons.mp hm with hj | hm
      · subst hj; exact hmono L _ (hhit acc)
      · exact ih _ hm
  obtain ⟨l, u, he, hl, hu⟩ := hfold (List.finRange s.size) none (List.mem_finRange _)
  have hb' : (List.finRange s.size).foldl f none = some b := hb
  rw [hb'] at he
  obtain rfl := Option.some.inj he
  exact ⟨hl, hu⟩

/-- `gatherBox?` encloses `gather x` at every index tensor when its box encloses `x`. -/
theorem gatherBox?_encloses {s t : Shape} {dims : List (Fin s.rank)}
    {batchDims : List (Fin s.rank × Fin t.rank)} (h : Shape.GatherValid s t dims batchDims)
    {x : Tensor ℝ s} (indices : Tensor (Einops.AxisTuple s.extent dims) t) (out : Shape)
    (hout : out = t ++ s.eraseAxes (dims ++ batchDims.map Prod.fst)) {Bx B : FlatBox α}
    (hx : Encloses Bx (Graph.flatOfSome ⟨s, x⟩)) (hB : gatherBox? h Bx = some B) :
    Encloses B
      (Graph.flatOfSome ⟨out, Tensor.gather x dims batchDims indices out h hout⟩) := by
  subst hout
  refine coordwiseBox?_encloses hx (fun lo hi hlh o b hb => ?_) hB
  rw [Tensor.gather_apply, Shape.Coord.reindexEq_self]
  exact gatherBound_encloses h hlh _ o hb

/-- `rangeBounds m n` encloses every map with values in `[-m, n]`. -/
theorem rangeBounds_enclosure {m n : Nat} {f : ℝ → ℝ} (hf : ∀ x, -(m : ℝ) ≤ f x ∧ f x ≤ n) :
    UnaryEnclosure f (rangeBounds (α := α) m n) := by
  intro lo hi outLo outHi x h _ _
  simp only [rangeBounds, Option.some.injEq, Prod.mk.injEq] at h
  obtain ⟨rfl, rfl⟩ := h
  have hd := LawfulBoundOps.subDown_le (0 : α) (Graph.countBounds (α := α) m).2
  rw [LawfulBoundOps.toReal_zero] at hd
  have hm := (countBounds_encloses (α := α) m).2
  exact ⟨by linarith [(hf x).1], (hf x).2.trans (countBounds_encloses (α := α) n).2⟩

/-- `unitRangeBounds m n` encloses every map with values in `[-m, n]`. -/
theorem unitRangeBounds_enclosure {m n : Nat} {f : ℝ → ℝ}
    (hf : ∀ x, -(m : ℝ) ≤ f x ∧ f x ≤ n) : UnaryEnclosure f (unitRangeBounds (α := α) m n) := by
  intro lo hi outLo outHi x h
  unfold unitRangeBounds at h
  split at h
  · exact rangeBounds_enclosure hf h
  · contradiction

/-- `withinOneBounds` encloses every map whose value lies within 1 of its argument. -/
theorem withinOneBounds_enclosure {f : ℝ → ℝ} (hf : ∀ x, x - 1 ≤ f x ∧ f x ≤ x + 1) :
    UnaryEnclosure f (withinOneBounds (α := α)) := by
  intro lo hi outLo outHi x h hlo hhi
  simp only [withinOneBounds, Option.some.injEq, Prod.mk.injEq] at h
  obtain ⟨rfl, rfl⟩ := h
  have hd := LawfulBoundOps.subDown_le lo (1 : α)
  have hu := LawfulBoundOps.le_addUp hi (1 : α)
  rw [LawfulBoundOps.toReal_one] at hd hu
  exact ⟨by linarith [(hf x).1], by linarith [(hf x).2]⟩

/-- `signBounds` encloses the sign. -/
theorem signBounds_enclosure :
    UnaryEnclosure (Context.sign (α := ℝ)) (signBounds (α := α)) := by
  intro lo hi outLo outHi x h hlo hhi
  rw [Context.sign_real]
  simp only [signBounds, Option.some.injEq, Prod.mk.injEq] at h
  obtain ⟨rfl, rfl⟩ := h
  have hsd := LawfulBoundOps.subDown_le (0 : α) 1
  have hsu := LawfulBoundOps.le_subUp (0 : α) 1
  rw [LawfulBoundOps.toReal_zero, LawfulBoundOps.toReal_one] at hsd hsu
  have hrange : -1 ≤ Real.sign x ∧ Real.sign x ≤ 1 := by
    rcases Real.sign_apply_eq x with h | h | h <;> rw [h] <;> norm_num
  constructor
  · split_ifs with h₁ h₂
    · rw [LawfulBoundOps.lt_iff, LawfulBoundOps.toReal_zero] at h₁
      rw [Real.sign_of_pos (by linarith), LawfulBoundOps.toReal_one]
    · linarith [hrange.1]
    · rw [LawfulBoundOps.lt_iff, LawfulBoundOps.toReal_zero] at h₁ h₂
      rw [LawfulBoundOps.toReal_zero]
      have hx : 0 ≤ x := by linarith [not_lt.mp h₂]
      rcases hx.lt_or_eq with hx | hx
      · rw [Real.sign_of_pos hx]; norm_num
      · rw [← hx, Real.sign_zero]
  · split_ifs with h₁ h₂
    · rw [LawfulBoundOps.lt_iff, LawfulBoundOps.toReal_zero] at h₁
      rw [Real.sign_of_neg (by linarith)]; linarith
    · rw [LawfulBoundOps.toReal_one]; exact hrange.2
    · rw [LawfulBoundOps.lt_iff, LawfulBoundOps.toReal_zero] at h₁ h₂
      rw [LawfulBoundOps.toReal_zero]
      have hx : x ≤ 0 := by linarith [not_lt.mp h₂]
      rcases hx.lt_or_eq with hx | hx
      · rw [Real.sign_of_neg hx]; norm_num
      · rw [hx, Real.sign_zero]

/-- A lawful binary enclosure, applied entry by entry, encloses the pointwise image. -/
theorem binaryBox?_encloses {F : ℝ → ℝ → ℝ} {enclose : α → α → α → α → Option (α × α)}
    (hF : BinaryEnclosure F enclose) {B₁ B₂ B : FlatBox α} {n : Nat} {f g : Nat → ℝ}
    (hx : RowEncloses B₁ n f) (hy : RowEncloses B₂ n g)
    (hb : binaryBox? enclose B₁ B₂ = some B) :
    RowEncloses B n fun i => F (f i) (g i) := by
  obtain ⟨d₁, l₁, u₁⟩ := B₁
  obtain ⟨d₂, l₂, u₂⟩ := B₂
  have h₁ := hx.1
  have h₂ := hy.1
  change d₁ = n at h₁
  change d₂ = n at h₂
  subst h₁
  subst h₂
  rw [rowEncloses_iff] at hx hy
  unfold binaryBox? at hb
  simp only [dite_true] at hb
  cases hs : Tensor.sequenceFinM fun i : Fin d₂ =>
      enclose (l₁.getScalar i) (u₁.getScalar i) (l₂.getScalar (Fin.cast rfl i))
        (u₂.getScalar (Fin.cast rfl i)) with
  | none => simp only [hs, Option.bind_eq_bind, Option.bind_none, reduceCtorEq] at hb
  | some bounds =>
      simp only [hs, Option.bind_eq_bind, Option.bind_some, Option.pure_def,
        Option.some.injEq] at hb
      subst hb
      rw [rowEncloses_iff]
      intro i
      have hi := Tensor.sequenceFinM_get_of_eq_some hs i
      simp only [Fin.cast_eq_self] at hi
      simpa only [getScalar_ofFn] using hF hi (hx i).1 (hx i).2 (hy i).1 (hy i).2

/-- The range `[-4, 4]` encloses the polar angle at `ℝ`. -/
theorem atan2Range_enclosure :
    BinaryEnclosure (InverseTrig.atan2 (α := ℝ))
      (fun aLo aHi _ _ => rangeBounds (α := α) 4 4 aLo aHi) := by
  intro aLo aHi bLo bHi outLo outHi x y h hal hau _ _
  refine rangeBounds_enclosure (α := α) (m := 4) (n := 4) (f := fun z => InverseTrig.atan2 z y)
    (fun z => ?_) h hal hau
  change -((4 : ℕ) : ℝ) ≤ Complex.arg ⟨y, z⟩ ∧ Complex.arg ⟨y, z⟩ ≤ ((4 : ℕ) : ℝ)
  have := Complex.neg_pi_lt_arg ⟨y, z⟩; have := Complex.arg_le_pi ⟨y, z⟩
  have := Real.pi_le_four; push_cast; constructor <;> linarith

/-- `contractBounds` encloses entry `o` of the contraction of `x` and `y` when the
endpoint tensors enclose them. -/
theorem contractBounds_encloses {s t : Shape} {dims batchDims : List (Fin s.rank × Fin t.rank)}
    (h : Shape.ContractValid s t dims batchDims) {x : Tensor ℝ s} {y : Tensor ℝ t}
    {xlo xhi : Tensor α s} {ylo yhi : Tensor α t}
    (hx : ∀ c, value (xlo c) ≤ x c ∧ x c ≤ value (xhi c))
    (hy : ∀ c, value (ylo c) ≤ y c ∧ y c ≤ value (yhi c))
    (o : (s.contract t dims batchDims).Coord) :
    value (contractBounds h xlo xhi ylo yhi o).1 ≤
        Tensor.contract x y dims batchDims _ h rfl o ∧
      Tensor.contract x y dims batchDims _ h rfl o ≤
        value (contractBounds h xlo xhi ylo yhi o).2 := by
  rw [Tensor.contract_apply, Shape.Coord.reindexEq_self]
  unfold contractBounds
  generalize List.finRange (Shape.size ((dims.map Prod.fst).map s.extent)) = L
  have key : ∀ (L : List (Fin (Shape.size ((dims.map Prod.fst).map s.extent))))
      (r : ℝ) (acc : α × α), value acc.1 ≤ r → r ≤ value acc.2 →
      value (L.foldl (fun acc i =>
        let a := s.contractLeft t dims batchDims o (Coord.unlinearize i)
        let b := h.contractRight o (Coord.unlinearize i)
        let p := intervalMul (xlo a) (xhi a) (ylo b) (yhi b)
        (BoundOps.addDown acc.1 p.1, BoundOps.addUp acc.2 p.2)) acc).1 ≤
        L.foldl (fun acc i =>
          acc + x (s.contractLeft t dims batchDims o (Coord.unlinearize i)) *
            y (h.contractRight o (Coord.unlinearize i))) r ∧
      L.foldl (fun acc i =>
          acc + x (s.contractLeft t dims batchDims o (Coord.unlinearize i)) *
            y (h.contractRight o (Coord.unlinearize i))) r ≤
        value (L.foldl (fun acc i =>
          let a := s.contractLeft t dims batchDims o (Coord.unlinearize i)
          let b := h.contractRight o (Coord.unlinearize i)
          let p := intervalMul (xlo a) (xhi a) (ylo b) (yhi b)
          (BoundOps.addDown acc.1 p.1, BoundOps.addUp acc.2 p.2)) acc).2 := by
    intro L
    induction L with
    | nil => intro r acc h₁ h₂; exact ⟨h₁, h₂⟩
    | cons i L ih =>
      intro r acc h₁ h₂
      apply ih
      · have hp := Graph.intervalMul_encloses (α := α)
          (hx (s.contractLeft t dims batchDims o (Coord.unlinearize i))).1
          (hx (s.contractLeft t dims batchDims o (Coord.unlinearize i))).2
          (hy (h.contractRight o (Coord.unlinearize i))).1
          (hy (h.contractRight o (Coord.unlinearize i))).2
        exact (LawfulBoundOps.addDown_le _ _).trans (by linarith [hp.1])
      · have hp := Graph.intervalMul_encloses (α := α)
          (hx (s.contractLeft t dims batchDims o (Coord.unlinearize i))).1
          (hx (s.contractLeft t dims batchDims o (Coord.unlinearize i))).2
          (hy (h.contractRight o (Coord.unlinearize i))).1
          (hy (h.contractRight o (Coord.unlinearize i))).2
        exact (by linarith [hp.2] : _ ≤ _).trans (LawfulBoundOps.le_addUp _ _)
  exact key L 0 (0, 0) (by simp [LawfulBoundOps.toReal_zero])
    (by simp [LawfulBoundOps.toReal_zero])

/-- `contractBox?` encloses the contraction of `x` and `y` when its boxes enclose them. -/
theorem contractBox?_encloses {s t : Shape} {dims batchDims : List (Fin s.rank × Fin t.rank)}
    (h : Shape.ContractValid s t dims batchDims) {x : Tensor ℝ s} {y : Tensor ℝ t}
    {Bx By B : FlatBox α} (hx : Encloses Bx (Graph.flatOfSome ⟨s, x⟩))
    (hy : Encloses By (Graph.flatOfSome ⟨t, y⟩))
    (hB : contractBox? h Bx By = some B) :
    Encloses B (Graph.flatOfSome
      ⟨s.contract t dims batchDims, Tensor.contract x y dims batchDims _ h rfl⟩) := by
  unfold contractBox? at hB
  split at hB
  · rename_i hdy
    refine coordwiseBox?_encloses hx (fun lo hi hlh o b hb => ?_) hB
    obtain rfl := Option.some.inj hb
    exact contractBounds_encloses h hlh (encloses_reshape hy hdy) o
  · cases hB

omit [BoundOps α] [LawfulBoundOps α] in
/-- A box computed from the boxes of two references comes from boxes that both references have. -/
theorem bind_box_pair {boxes : Array (Option (FlatBox α))} {r₁ r₂ : IR.Ref}
    {T : FlatBox α → FlatBox α → Option (FlatBox α)} {B : FlatBox α}
    (h : (do T (← box? boxes r₁) (← box? boxes r₂)) = some B) :
    ∃ B₁ B₂, box? boxes r₁ = some B₁ ∧ box? boxes r₂ = some B₂ ∧ T B₁ B₂ = some B := by
  obtain ⟨B₁, h₁, h⟩ := Option.bind_eq_some_iff.mp h
  obtain ⟨B₂, h₂, h⟩ := Option.bind_eq_some_iff.mp h
  exact ⟨B₁, B₂, h₁, h₂, h⟩

/-- `Graph.absBound lo hi` bounds `|x|` for every `x` in `[lo, hi]`. -/
theorem abs_le_absBound {lo hi : α} {x : ℝ} (hlo : value lo ≤ x) (hhi : x ≤ value hi) :
    |x| ≤ value (Graph.absBound lo hi) := by
  have h := LawfulBoundOps.le_subUp (0 : α) lo
  rw [LawfulBoundOps.toReal_zero] at h
  simp only [Graph.absBound, LawfulBoundOps.toReal_maximum]
  rw [abs_le]
  constructor
  · linarith [le_max_left (value (BoundOps.subUp (0 : α) lo)) (value hi)]
  · linarith [le_max_right (value (BoundOps.subUp (0 : α) lo)) (value hi)]

/-- The box `symBounds m` contains every `z` with `|z| ≤ m`. -/
theorem symBounds_mem {m outLo outHi : α} {z : ℝ}
    (h : symBounds m = some (outLo, outHi)) (hz : |z| ≤ value m) :
    value outLo ≤ z ∧ z ≤ value outHi := by
  simp only [symBounds, Option.some.injEq, Prod.mk.injEq] at h
  obtain ⟨rfl, rfl⟩ := h
  have h := LawfulBoundOps.subDown_le (0 : α) m
  rw [LawfulBoundOps.toReal_zero] at h
  rw [abs_le] at hz
  exact ⟨by linarith, hz.2⟩

/-- At `ℝ`, `x - trunc x` lies between `x` and 0. -/
private theorem sub_truncRound_mem (q : ℝ) :
    min q 0 ≤ q - (FloatLib.Floats.Formats.Flocq.truncRound q : ℝ) ∧
      q - (FloatLib.Floats.Formats.Flocq.truncRound q : ℝ) ≤ max q 0 := by
  unfold FloatLib.Floats.Formats.Flocq.truncRound
  split_ifs with h
  · have h₁ := Int.le_ceil q
    have h₂ : ((⌈q⌉ : ℤ) : ℝ) ≤ 0 := by exact_mod_cast Int.ceil_le.mpr (by exact_mod_cast h.le)
    rw [min_eq_left h.le, max_eq_right h.le]
    constructor <;> linarith
  · push Not at h
    have h₁ := Int.floor_le q
    have h₂ : (0 : ℝ) ≤ (⌊q⌋ : ℝ) := by exact_mod_cast Int.floor_nonneg.mpr h
    rw [min_eq_right h, max_eq_left h]
    constructor <;> linarith

/-- At `ℝ`, `fmod x y` lies between `x` and 0: it has the sign of `x` and at most its size, and
`fmod x 0 = x`. -/
private theorem fmod_mem (x y : ℝ) : min x 0 ≤ Rounding.fmod x y ∧ Rounding.fmod x y ≤ max x 0 := by
  rw [Context.fmod_real]
  rcases eq_or_ne y 0 with rfl | hy
  · simp
  have ht := sub_truncRound_mem (x / y)
  have hx : x = y * (x / y) := by field_simp
  have hr : x - y * (FloatLib.Floats.Formats.Flocq.truncRound (x / y) : ℝ) =
      y * (x / y - (FloatLib.Floats.Formats.Flocq.truncRound (x / y) : ℝ)) := by
    rw [mul_sub, ← hx]
  rw [hr]
  set q := x / y
  set d := q - (FloatLib.Floats.Formats.Flocq.truncRound q : ℝ)
  rcases lt_or_gt_of_ne hy with hy | hy <;> rcases le_or_gt 0 q with hq | hq
  · rw [min_eq_right hq, max_eq_left hq] at ht
    have : y * q ≤ 0 := mul_nonpos_of_nonpos_of_nonneg hy.le hq
    rw [hx, min_eq_left this, max_eq_right this]
    exact ⟨mul_le_mul_of_nonpos_left ht.2 hy.le, mul_nonpos_of_nonpos_of_nonneg hy.le ht.1⟩
  · rw [min_eq_left hq.le, max_eq_right hq.le] at ht
    have : 0 ≤ y * q := mul_nonneg_of_nonpos_of_nonpos hy.le hq.le
    rw [hx, min_eq_right this, max_eq_left this]
    exact ⟨mul_nonneg_of_nonpos_of_nonpos hy.le ht.2, mul_le_mul_of_nonpos_left ht.1 hy.le⟩
  · rw [min_eq_right hq, max_eq_left hq] at ht
    have : 0 ≤ y * q := mul_nonneg hy.le hq
    rw [hx, min_eq_right this, max_eq_left this]
    exact ⟨mul_nonneg hy.le ht.1, mul_le_mul_of_nonneg_left ht.2 hy.le⟩
  · rw [min_eq_left hq.le, max_eq_right hq.le] at ht
    have : y * q ≤ 0 := mul_nonpos_of_nonneg_of_nonpos hy.le hq.le
    rw [hx, min_eq_left this, max_eq_right this]
    exact ⟨mul_le_mul_of_nonneg_left ht.1 hy.le, mul_nonpos_of_nonneg_of_nonpos hy.le ht.2⟩

/-- At `ℝ`, `|remainder x y| ≤ max |x| |y|`: the remainder is `fmod x y`, moved by `y` when their
signs differ. -/
private theorem abs_remainder_le (x y : ℝ) : |Context.remainder x y| ≤ max |x| |y| := by
  have hr := fmod_mem x y
  have hrx : |Rounding.fmod x y| ≤ |x| := by
    rcases le_total 0 x with hx | hx
    · rw [min_eq_right hx, max_eq_left hx] at hr
      rw [abs_of_nonneg hr.1, abs_of_nonneg hx]; exact hr.2
    · rw [min_eq_left hx, max_eq_right hx] at hr
      rw [abs_of_nonpos hr.2, abs_of_nonpos hx]; linarith [hr.1]
  unfold Context.remainder remainderSchedule
  dsimp only
  set r := Rounding.fmod x y
  by_cases h₀ : r = 0
  · rw [h₀]; simp only [beq_self_eq_true, ↓reduceIte]; split_ifs <;> simp
  have hb : (r == 0) = false := by simpa using h₀
  rw [hb]; simp only [Bool.false_eq_true, ↓reduceIte]
  split_ifs with h₁
  · simp only [bne_iff_ne, ne_eq, decide_eq_decide] at h₁
    rcases lt_or_ge r 0 with hr0 | hr0
    · have hy : 0 ≤ y := not_lt.mp fun hy => h₁ ⟨fun _ => hy, fun _ => hr0⟩
      rw [abs_of_nonneg hy, abs_le]
      exact ⟨by linarith [neg_abs_le r, le_max_left |x| y], by linarith [le_max_right |x| y]⟩
    · have hy : y < 0 := by
        by_contra hy
        exact h₁ ⟨fun h => absurd h (not_lt.mpr hr0), fun h => absurd h hy⟩
      rw [abs_of_neg hy, abs_le]
      exact ⟨by linarith [le_max_right |x| (-y)], by linarith [le_abs_self r, le_max_left |x| (-y)]⟩
  · exact hrx.trans (le_max_left _ _)

/-- At `ℝ`, `copysign x y` has the size of `x`. -/
private theorem abs_copysign (x y : ℝ) : |Context.copysign x y| = |x| := by
  change |(if 0 > y then -MathFunctions.abs x else MathFunctions.abs x)| = |x|
  split_ifs <;> simp [abs_neg]

/-- The zero hull of the box of `a` encloses `fmod a b` at `ℝ`. -/
theorem fmodBounds_enclosure :
    BinaryEnclosure (Rounding.fmod (α := ℝ))
      (fun aLo aHi _ _ => zeroHullBounds (α := α) aLo aHi) := by
  intro aLo aHi bLo bHi outLo outHi x y h hal hau _ _
  exact zeroHullBounds_enclosure (α := α) (f := fun z => Rounding.fmod z y) (fun z => fmod_mem z y)
    h hal hau

/-- `[-m, m]`, with `m` bounding both `|a|` and `|b|`, encloses `remainder a b` at `ℝ`. -/
theorem remainderBounds_enclosure :
    BinaryEnclosure (Context.remainder (α := ℝ)) (fun aLo aHi bLo bHi => symBounds (α := α)
      (Context.maximum (Graph.absBound aLo aHi) (Graph.absBound bLo bHi))) := by
  intro aLo aHi bLo bHi outLo outHi x y h hal hau hbl hbu
  refine symBounds_mem h ((abs_remainder_le x y).trans ?_)
  rw [LawfulBoundOps.toReal_maximum]
  exact max_le_max (abs_le_absBound hal hau) (abs_le_absBound hbl hbu)

/-- `nonzeroDivisor? enclose` encloses what `enclose` encloses. -/
theorem nonzeroDivisor?_enclosure {f : ℝ → ℝ → ℝ} {enclose : α → α → α → α → Option (α × α)}
    (h : BinaryEnclosure f enclose) : BinaryEnclosure f (nonzeroDivisor? enclose) := by
  intro aLo aHi bLo bHi outLo outHi x y hb
  unfold nonzeroDivisor? at hb
  split at hb
  · exact h hb
  · contradiction

/-- `[-m, m]`, with `m` bounding `|a|`, encloses `copysign a b` at `ℝ`. -/
theorem copysignBounds_enclosure :
    BinaryEnclosure (Context.copysign (α := ℝ))
      (fun aLo aHi _ _ => symBounds (α := α) (Graph.absBound aLo aHi)) := by
  intro aLo aHi bLo bHi outLo outHi x y h hal hau _ _
  exact symBounds_mem h ((abs_copysign x y).le.trans (abs_le_absBound hal hau))

/-- The flat value of a linear layer whose input extent is empty: the real bias at each leading
row. -/
theorem flat_linear_empty {L : Shape} {n m : Nat} (h0 : n = 0) (W : Tensor α [m, n])
    (b : Tensor α [m]) (x : Tensor ℝ (L ++ [n])) (i : Nat) (hk : i < L.size * m) :
    getAtOrZero (Graph.flatOfSome
        ⟨L ++ [m], Linear.forwardLeading L ⟨Tensor.map value W, Tensor.map value b⟩ x⟩).v [i] =
      value ((linearBiasRows b L.size).getScalar ⟨i, hk⟩) := by
  have hsize : (L ++ [m]).size = L.size * m := by simp [Shape.size_append]
  have hspec : Linear.forward (α := ℝ)
      (⟨Tensor.map value W, Tensor.map value b⟩ : Linear ℝ n m) =
        fun _ => Tensor.map value b :=
    funext (forward_of_inDim_zero h0 _ _)
  simp only [Graph.flatOfSome]
  rw [← getAtOrZero_castDimScalar hsize, Linear.forwardLeading, hspec,
    flatten_mapLeading_const, getAtOrZero_eq_getScalar _ ⟨i, hk⟩]
  simp [linearBiasRows]

/-- The flat value of a linear layer on one leading row: the real affine map of the flat input. -/
theorem flat_linear_one {L : Shape} {n m : Nat} (hone : L.size = 1) (W : Tensor α [m, n])
    (b : Tensor α [m]) (x : Tensor ℝ (L ++ [n])) (i : Fin m) :
    getAtOrZero (Graph.flatOfSome
        ⟨L ++ [m], Linear.forwardLeading L ⟨Tensor.map value W, Tensor.map value b⟩ x⟩).v [i.val] =
      (∑ j : Fin n, value (Tensor.get2 W i j) *
        getAtOrZero (Graph.flatOfSome ⟨L ++ [n], x⟩).v [j.val]) +
        value (b.getScalar i) := by
  have hm : (L ++ [m]).size = m := by simp [Shape.size_append, hone]
  have hn : (L ++ [n]).size = n := by simp [Shape.size_append, hone]
  simp only [Graph.flatOfSome]
  rw [← getAtOrZero_castDimScalar hm, Linear.forwardLeading,
    flatten_mapLeading_of_size_one _ L hone x hm hn, getAtOrZero_eq_getScalar,
    Linear.forward, congrFun (Tensor.getScalar_add _ _) _,
    Tensor.getScalar_contract_matVec_eq_sum]
  simp only [getScalar_map]
  congr 1
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [← getAtOrZero_castDimScalar hn, getAtOrZero_eq_getScalar, get2_map_toReal]

/-- A box encloses the flat value of a reshape of `y` exactly when it encloses the flat value of
`y`. -/
theorem encloses_flat_reshape {s t : Shape} {B : FlatBox α} {y : Tensor ℝ s}
    (h : s.size = t.size) :
    Encloses B (Graph.flatOfSome ⟨t, Tensor.reshape y t h⟩) ↔
      Encloses B (Graph.flatOfSome ⟨s, y⟩) := by
  simp only [Encloses, Graph.flatOfSome, Tensor.reshape_vector_reshape, getAtOrZero_eqRec]
  exact ⟨rowEncloses_width h.symm, rowEncloses_width h⟩

/-- `maskedSoftmaxBox?` encloses the masked softmax along the last axis of any real
tensor, whatever box it is given. -/
theorem maskedSoftmaxBox?_encloses {s : Shape} {dim : Fin s.rank} (hdim : dim.val + 1 = s.rank)
    {allowed : Tensor Bool s} {Bx B : FlatBox α} (x : Tensor ℝ s)
    (h : maskedSoftmaxBox? s allowed Bx = some B) :
    Encloses B (Graph.flatOfSome ⟨s, Tensor.softmax x dim (some allowed)⟩) := by
  unfold maskedSoftmaxBox? at h
  split at h
  · obtain rfl := Option.some.inj h
    rw [show dim = ⟨s.rank - 1, by omega⟩ from Fin.ext (Nat.eq_sub_of_add_eq hdim),
      Tensor.softmax_mask_last]
    refine encloses_of_coord rfl fun c => ?_
    simp only [Tensor.getAtOrZero_reshape_toFin]
    exact ibpHardMaskedSoftmax_encloses _ _ x allowed c
  · cases h

end

end TorchLean.CROWN.Op.Directed
