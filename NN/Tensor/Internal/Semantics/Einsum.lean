/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Check.Einsum
public import Mathlib.Algebra.BigOperators.Fin
public import NN.Tensor.Internal.Representation.Fiber -- shake: keep

/-!
# Algebraic semantics for einsum

A checked einsum ranges over one finite assignment of every logical axis.
Each operand reads that assignment through three coordinate operations:

1. select the labels used by the operand, retaining repeated labels;
2. use the same selected coordinate at every occurrence of a repeated label,
   which implements diagonal indexing; and
3. project resolved dimensions to the operand's physical dimensions, mapping
   singleton dimensions to their unique coordinate.

The operand values are multiplied in source order and contracted coordinates
are accumulated in row-major order. These ordered semantics require only
`Mul`, `Add`, and scalar zero and one, so the same denotation applies to IEEE
floating-point values without asserting false associativity or commutativity
instances. When the scalar operations do form additive and multiplicative
monoids, a theorem identifies the ordered denotation with the usual
`Rep.push` fiber sum. Distributivity enters only in multilinearity
theorems, and commutative multiplication only in the selected-operand
adjoint.

Missing leading ellipsis slots need no special case: an operand simply omits
those logical axes, so its value is constant while the omitted coordinates
vary. This is exactly singleton broadcasting under right-aligned ellipses.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal

open scoped BigOperators

universe u

namespace Check.CheckedEinsum

/-- The heterogeneous family of input tensors accepted by a checked einsum. -/
abbrev InputTensors (checked : CheckedEinsum) (R : Type u)
    [Storage R] :=
  (operand : Fin checked.inputShapes.length) →
    Rep R (checked.inputShapes.get operand)

/-- The output tensor type inferred by a checked einsum. -/
abbrev OutputTensor (checked : CheckedEinsum) (R : Type u)
    [Storage R] :=
  Rep R checked.output

/-- Expanded operand-axis lists and input shapes have the same count. -/
theorem input_axes_length (checked : CheckedEinsum) :
    checked.inputAxes.length = checked.inputShapes.length := by
  simp [inputAxes, einsumInputAxes, checked.input_count]

/-- Expanded logical labels used by one input operand. -/
def operandAxes (checked : CheckedEinsum)
    (operand : Fin checked.inputShapes.length) : List EinsumAxis :=
  checked.inputAxes.get
    ⟨operand.val, by
      rw [checked.input_axes_length]
      exact operand.isLt⟩

/-- Every logical label used by an operand belongs to the global assignment. -/
theorem input_axis_mem_global (checked : CheckedEinsum)
    (operand : Fin checked.inputShapes.length) :
    ∀ ⦃axis⦄, axis ∈ checked.operandAxes operand →
      axis ∈ checked.globalAxes := by
  intro axis hAxis
  have hOperand :
      checked.operandAxes operand ∈ checked.inputAxes :=
    List.get_mem checked.inputAxes
      ⟨operand.val, by
        rw [checked.input_axes_length]
        exact operand.isLt⟩
  have hFlattened : axis ∈ checked.inputAxes.flatten :=
    List.mem_flatten.mpr ⟨checked.operandAxes operand, hOperand, hAxis⟩
  simpa [globalAxes, inputAxes, einsumGlobalAxes] using hFlattened

/--
The physical shape of an operand broadcasts to its resolved logical shape.

This theorem is the proof-level bridge from the checker's per-dimension
certificate to the coordinate map used by the denotation.
-/
theorem operand_shape_broadcastable (checked : CheckedEinsum)
    (operand : Fin checked.inputShapes.length) :
    List.Forall₂
      (fun physicalLength logicalLength =>
        physicalLength = logicalLength ∨ physicalLength = 1)
      (checked.inputShapes.get operand)
      ((checked.operandAxes operand).map checked.axisLength) := by
  have hAxesBound : operand.val < checked.inputAxes.length := by
    rw [checked.input_axes_length]
    exact operand.isLt
  have hDimensions :
      List.Forall₂
        (fun physicalLength axis =>
          physicalLength = checked.axisLength axis ∨ physicalLength = 1)
        (checked.inputShapes.get operand)
        (checked.operandAxes operand) := by
    simpa [operandAxes, inputAxes, axisLength] using
      checked.input_dimensions.get operand.isLt hAxesBound
  rw [List.forall₂_map_right_iff]
  exact hDimensions

/-- Convert a global assignment coordinate to its named logical coordinates. -/
def globalTensorCoordinateEquiv (checked : CheckedEinsum) :
    Coord (checked.globalAxes.map checked.axisLength) ≃
      AxisTuple checked.axisLength checked.globalAxes :=
  AxisTuple.coordEquiv checked.axisLength checked.globalAxes

/-- Convert an output tensor coordinate to its named logical coordinates. -/
def outputTensorCoordinateEquiv (checked : CheckedEinsum) :
    Coord checked.output ≃
      AxisTuple checked.axisLength checked.outputAxes :=
  AxisTuple.coordEquiv checked.axisLength checked.outputAxes

/--
Retain the requested output labels from a complete logical-axis assignment.
-/
def outputCoordinateOfGlobal (checked : CheckedEinsum) :
    Coord (checked.globalAxes.map checked.axisLength) →
      Coord checked.output :=
  fun globalCoordinate =>
    checked.outputTensorCoordinateEquiv.symm <|
      AxisTuple.select checked.output_axis_mem_global <|
        checked.globalTensorCoordinateEquiv globalCoordinate

/--
Read one operand coordinate from a complete logical-axis assignment.

Repeated labels select one shared coordinate before singleton broadcasting,
so this single map covers ordinary indexing, diagonals, and broadcasting.
-/
def inputCoordinateOfGlobal (checked : CheckedEinsum)
    (operand : Fin checked.inputShapes.length) :
    Coord (checked.globalAxes.map checked.axisLength) →
      Coord (checked.inputShapes.get operand) :=
  fun globalCoordinate =>
    Coord.broadcast
      (checked.inputShapes.get operand)
      ((checked.operandAxes operand).map checked.axisLength)
      (checked.operand_shape_broadcastable operand) <|
        (AxisTuple.coordEquiv
          checked.axisLength (checked.operandAxes operand)).symm <|
          AxisTuple.select (checked.input_axis_mem_global operand) <|
            checked.globalTensorCoordinateEquiv globalCoordinate

/--
Every contracted logical axis belongs to the complete global-axis
assignment.
-/
theorem contraction_axis_mem_global (checked : CheckedEinsum) :
    ∀ ⦃axis⦄,
      axis ∈ checked.contractedAxes →
        axis ∈ checked.globalAxes :=
  fun _ hAxis => (List.mem_filter.mp hAxis).1

/--
One einsum contraction-fiber coordinate is exactly one assignment of every
global logical axis absent from the output.

The equivalence is independent of operand count, tensor rank, repeated
labels, broadcasting, and axis lengths. It therefore also covers scalar
outputs, contractions of no axes, and empty fibers caused by zero-length
dimensions.
-/
noncomputable def contractionFiberEquiv (checked : CheckedEinsum)
    (outputCoordinate : Coord checked.output) :
    Fiber checked.outputCoordinateOfGlobal outputCoordinate ≃
      AxisTuple checked.axisLength checked.contractedAxes := by
  classical
  let coordinateFiberEquiv :
      Fiber checked.outputCoordinateOfGlobal outputCoordinate ≃
        Fiber
          (AxisTuple.select checked.output_axis_mem_global)
          (checked.outputTensorCoordinateEquiv outputCoordinate) :=
    { toFun := fun globalCoordinate =>
        ⟨checked.globalTensorCoordinateEquiv globalCoordinate.1, by
          apply checked.outputTensorCoordinateEquiv.symm.injective
          simpa only [outputCoordinateOfGlobal, Equiv.symm_apply_apply] using
            globalCoordinate.2⟩
      invFun := fun globalAxisCoordinate =>
        ⟨checked.globalTensorCoordinateEquiv.symm globalAxisCoordinate.1, by
          simp only [outputCoordinateOfGlobal, Equiv.apply_symm_apply,
            globalAxisCoordinate.2, Equiv.symm_apply_apply]⟩
      left_inv := fun globalCoordinate => by
        apply Subtype.ext
        exact checked.globalTensorCoordinateEquiv.symm_apply_apply
          globalCoordinate.1
      right_inv := fun globalAxisCoordinate => by
        apply Subtype.ext
        exact checked.globalTensorCoordinateEquiv.apply_symm_apply
          globalAxisCoordinate.1 }
  exact coordinateFiberEquiv.trans <|
    AxisTuple.selectFiberEquiv
      checked.output_axes_nodup
      checked.global_axes_nodup
      checked.output_axis_mem_global
      (checked.outputTensorCoordinateEquiv outputCoordinate)

/--
The contraction-fiber equivalence reads the contracted axes from the complete
global logical-axis assignment.
-/
@[simp] theorem contractionFiberEquiv_apply
    (checked : CheckedEinsum)
    (outputCoordinate : Coord checked.output)
    (globalCoordinate :
      Fiber checked.outputCoordinateOfGlobal outputCoordinate) :
    checked.contractionFiberEquiv outputCoordinate globalCoordinate =
      AxisTuple.select checked.contraction_axis_mem_global
        (checked.globalTensorCoordinateEquiv globalCoordinate.1) :=
  rfl

/--
Every output coordinate has one complete assignment for each setting of the
contracted logical axes.

The fiber cardinality is therefore the product of the contracted lengths.
The formula includes scalar outputs, contractions of no axes, and empty
fibers caused by a zero-length contracted axis.
-/
theorem contraction_fiber_card (checked : CheckedEinsum)
    (outputCoordinate : Coord checked.output) :
    Fintype.card
        (Fiber checked.outputCoordinateOfGlobal outputCoordinate) =
      (checked.contractedAxes.map checked.axisLength).prod := by
  classical
  let contractedAxes := checked.contractedAxes
  calc
    Fintype.card
          (Fiber checked.outputCoordinateOfGlobal outputCoordinate) =
        Fintype.card
          (AxisTuple checked.axisLength contractedAxes) :=
      Fintype.card_congr
        (checked.contractionFiberEquiv outputCoordinate)
    _ = Fintype.card
          (Coord (contractedAxes.map checked.axisLength)) :=
      (Fintype.card_congr
        (AxisTuple.coordEquiv checked.axisLength contractedAxes)).symm
    _ = Shape.size (contractedAxes.map checked.axisLength) :=
      Coord.card (contractedAxes.map checked.axisLength)
    _ = (contractedAxes.map checked.axisLength).prod :=
      Shape.size_eq_prod (contractedAxes.map checked.axisLength)
    _ = (checked.contractedAxes.map checked.axisLength).prod := rfl

/--
Reconstruct a complete logical coordinate from retained output coordinates
and one row-major assignment of every contracted axis.

The output axes come first only in this intermediate tuple. Selection restores
the checked global-axis order before converting back to a tensor coordinate.
-/
def reconstructedGlobalCoordinate
    (checked : CheckedEinsum)
    (outputCoordinate : Coord checked.output)
    (contractionCoordinate :
      AxisTuple checked.axisLength checked.contractedAxes) :
    Coord (checked.globalAxes.map checked.axisLength) :=
  checked.globalTensorCoordinateEquiv.symm <|
    AxisTuple.select
      (fun axis hGlobal => by
        by_cases hOutput : axis ∈ checked.outputAxes
        · exact List.mem_append_left _ hOutput
        · exact List.mem_append_right _ <| by
            simp [contractedAxes, hGlobal, hOutput]) <|
      AxisTuple.append checked.outputAxes
        (checked.outputTensorCoordinateEquiv outputCoordinate)
        contractionCoordinate

/-- Reconstructing a global coordinate preserves its supplied output coordinate. -/
@[simp] theorem outputCoordinateOfGlobal_reconstructedGlobalCoordinate
    (checked : CheckedEinsum)
    (outputCoordinate : Coord checked.output)
    (contractionCoordinate :
      AxisTuple checked.axisLength checked.contractedAxes) :
    checked.outputCoordinateOfGlobal
        (checked.reconstructedGlobalCoordinate outputCoordinate
          contractionCoordinate) =
      outputCoordinate := by
  apply checked.outputTensorCoordinateEquiv.injective
  simp only [outputCoordinateOfGlobal, reconstructedGlobalCoordinate,
    Equiv.apply_symm_apply]
  rw [AxisTuple.select_comp]
  exact AxisTuple.select_append_left
    checked.output_axes_nodup
    (checked.outputTensorCoordinateEquiv outputCoordinate)
    contractionCoordinate

/-- Selecting contracted axes from a reconstructed coordinate recovers their values. -/
@[simp] theorem contractionOf_reconstructedGlobalCoordinate
    (checked : CheckedEinsum)
    (outputCoordinate : Coord checked.output)
    (contractionCoordinate :
      AxisTuple checked.axisLength checked.contractedAxes) :
    AxisTuple.select checked.contraction_axis_mem_global
        (checked.globalTensorCoordinateEquiv
          (checked.reconstructedGlobalCoordinate outputCoordinate
            contractionCoordinate)) =
      contractionCoordinate := by
  simp only [reconstructedGlobalCoordinate, Equiv.apply_symm_apply]
  rw [AxisTuple.select_comp]
  exact AxisTuple.select_append_right
    (checked.global_axes_nodup.filter _)
    (fun axis hOutput hContracted => by
      have hAbsent := (List.mem_filter.mp hContracted).2
      have : axis ∉ checked.outputAxes := by
        simpa using hAbsent
      exact this hOutput)
    (checked.outputTensorCoordinateEquiv outputCoordinate)
    contractionCoordinate

end Check.CheckedEinsum

namespace Semantics

open Check

/--
Add values over a multidimensional coordinate space in row-major order.

The initial accumulator is explicit because generated kernels may continue a
partially computed contraction. No associativity or commutativity law is
assumed; the nesting order is part of the denotation.

The fused lowering in `Lowering.Einsum` runs this very function rather than a
private copy of it, so the executable kernel and the denotation can never drift
apart in their traversal order.
-/
def coordinateSum {R : Type u} [Add R] [OfNat R 0] :
    (shape : Shape) → (Coord shape → R) → (initial : R := 0) → R
  | [], values, initial => initial + values PUnit.unit
  | length :: shape, values, initial =>
      Fin.foldl length
        (fun total coordinate =>
          coordinateSum shape
            (fun tailCoordinate => values (coordinate, tailCoordinate))
            total)
        initial

/--
Nested row-major coordinate addition equals the finite sum for lawful addition.

This is the only place the library crosses from the ordered loop to a big
operator; the lowering module reuses it instead of reproving it.
-/
theorem coordinateSum_eq_add_sum {R : Type u} [AddCommMonoid R]
    (shape : Shape) (values : Coord shape → R) (initial : R) :
    coordinateSum shape values initial =
      initial + ∑ coordinate, values coordinate := by
  induction shape generalizing initial with
  | nil =>
      simp only [coordinateSum]
      let unique : Unique (Coord []) := {
        default := PUnit.unit
        uniq := fun coordinate => by
          cases coordinate
          rfl }
      rw [@Fintype.sum_unique R (Coord []) _ _ unique]
  | cons length shape inductionHypothesis =>
      simp only [coordinateSum]
      simp_rw [inductionHypothesis]
      calc
        Fin.foldl length
              (fun total coordinate =>
                total + ∑ tailCoordinate, values (coordinate, tailCoordinate))
              initial =
            initial +
              Fin.foldl length
                (fun total coordinate =>
                  total + ∑ tailCoordinate, values (coordinate, tailCoordinate))
                0 := by
          simpa using
            (Fin.foldl_assoc
              (op := fun left right : R => left + right)
              (f := fun coordinate =>
                ∑ tailCoordinate, values (coordinate, tailCoordinate))
              (a₁ := initial) (a₂ := 0))
        _ = initial +
              ∑ coordinate, ∑ tailCoordinate,
                values (coordinate, tailCoordinate) := by
          rw [Fin.foldl_eq_foldl_finRange, ← List.foldl_map,
            ← List.ofFn_eq_map, ← List.sum_eq_foldl, Fin.sum_ofFn]
        _ = initial + ∑ coordinate, values coordinate := by
          exact congrArg (initial + ·) (Fintype.sum_prod_type values).symm

/--
Changing one entry of a finite ordered product from `leftValue` or
`rightValue` to their sum distributes the complete product.

The induction follows the distinguished operand through the left-to-right
`List.ofFn` order. It uses right distributivity when that operand is first and
left distributivity after passing an earlier operand, so multiplication never
needs to commute.
-/
private theorem prod_ofFn_update_add {R : Type u} [Semiring R] {n : Nat}
    (values : Fin n → R) (operand : Fin n)
    (leftValue rightValue : R) :
    (List.ofFn
          (Function.update values operand (leftValue + rightValue))).prod =
      (List.ofFn (Function.update values operand leftValue)).prod +
        (List.ofFn (Function.update values operand rightValue)).prod := by
  induction n with
  | zero =>
      exact Fin.elim0 operand
  | succ n induction =>
      refine Fin.cases ?_ (fun previousOperand => ?_) operand
      · simp [List.ofFn_succ, Function.update_self, Function.update_of_ne,
          add_mul]
      · simp only [List.ofFn_succ, List.prod_cons]
        have update_succ (replacement : R) :
            (fun remainingOperand =>
                Function.update values previousOperand.succ replacement
                  remainingOperand.succ) =
              Function.update
                (fun remainingOperand => values remainingOperand.succ)
                previousOperand replacement := by
          funext remainingOperand
          by_cases hOperand : remainingOperand = previousOperand
          · subst previousOperand
            simp only [Function.update_self]
          · rw [Function.update_of_ne hOperand]
            apply Function.update_of_ne
            exact fun hEqual => hOperand (Fin.succ_inj.mp hEqual)
        rw [update_succ (leftValue + rightValue), update_succ leftValue,
          update_succ rightValue]
        have hZero : (0 : Fin (n + 1)) ≠ previousOperand.succ :=
          Ne.symm (Fin.succ_ne_zero previousOperand)
        simp only [Function.update_of_ne hZero]
        rw [induction
          (fun remainingOperand => values remainingOperand.succ)
          previousOperand]
        exact mul_add _ _ _

/--
In a commutative product, changing one selected factor is the same as
multiplying that factor by the product obtained after replacing it by one.

This private identity is the algebraic step used by the selected-operand
einsum adjoint. The public statement remains about tensors rather than this
particular finite-product representation.
-/
private theorem prod_ofFn_update_eq_mul_update_one
    {R : Type u} [CommMonoid R] {n : Nat}
    (values : Fin n → R) (operand : Fin n) (replacement : R) :
    (List.ofFn (Function.update values operand replacement)).prod =
      replacement *
        (List.ofFn (Function.update values operand 1)).prod := by
  classical
  simp [List.prod_ofFn, Finset.prod_update_of_mem]

/--
The tensor of ordered operand products over complete logical assignments.

`List.ofFn` enumerates operand indices from left to right, and `List.foldl`
records that exact multiplication order without assuming monoid laws.
-/
def einsumProductTensor {R : Type u} [Storage R] [Mul R] [OfNat R 1]
    (checked : CheckedEinsum) (inputTensors : checked.InputTensors R) :
    Rep R (checked.globalAxes.map checked.axisLength) :=
  Rep.ofFn fun globalCoordinate =>
    (List.ofFn (fun operand =>
      inputTensors operand <|
        checked.inputCoordinateOfGlobal operand globalCoordinate)).foldl
      (· * ·) 1

/--
Reading the product tensor exposes the source-ordered multiplicative fold.
-/
theorem einsumProductTensor_get_ordered {R : Type u}
    [Storage R] [Mul R] [OfNat R 1]
    (checked : CheckedEinsum) (inputTensors : checked.InputTensors R)
    (globalCoordinate : Coord (checked.globalAxes.map checked.axisLength)) :
    (einsumProductTensor checked inputTensors).get globalCoordinate =
      (List.ofFn (fun operand =>
        inputTensors operand <|
          checked.inputCoordinateOfGlobal operand globalCoordinate)).foldl
        (· * ·) 1 :=
  Rep.get_ofFn _ _

/--
For a lawful monoid, the source-ordered fold is the usual list product.
-/
@[simp, grind =] theorem einsumProductTensor_get {R : Type u}
    [Storage R] [Monoid R]
    (checked : CheckedEinsum) (inputTensors : checked.InputTensors R)
    (globalCoordinate : Coord (checked.globalAxes.map checked.axisLength)) :
    (einsumProductTensor checked inputTensors).get globalCoordinate =
      (List.ofFn fun operand =>
        inputTensors operand <|
          checked.inputCoordinateOfGlobal operand globalCoordinate).prod := by
  rw [einsumProductTensor_get_ordered, ← List.prod_eq_foldl]

/--
The ordered-product tensor is additive in any selected input operand.

All other operands remain fixed through `Function.update`. This statement
covers operands of different ranks, repeated labels, singleton broadcasting,
and any position in a positive-arity einsum. In particular, it does not move
the selected operand through its neighbors, so a noncommutative semiring is
sufficient.
-/
@[grind =] theorem einsumProductTensor_update_add {R : Type u}
    [Storage R] [Semiring R]
    (checked : CheckedEinsum)
    (inputTensors : checked.InputTensors R)
    (operand : Fin checked.inputShapes.length)
    (leftTensor rightTensor :
      Rep R (checked.inputShapes.get operand)) :
    einsumProductTensor checked
        (Function.update inputTensors operand (leftTensor + rightTensor)) =
      einsumProductTensor checked
          (Function.update inputTensors operand leftTensor) +
        einsumProductTensor checked
          (Function.update inputTensors operand rightTensor) := by
  ext globalCoordinate
  let inputCoordinate :=
    checked.inputCoordinateOfGlobal operand globalCoordinate
  have sampled_update (replacement :
      Rep R (checked.inputShapes.get operand)) :
      (fun currentOperand =>
          Function.update inputTensors operand replacement currentOperand
            (checked.inputCoordinateOfGlobal currentOperand globalCoordinate)) =
        Function.update
          (fun currentOperand =>
            inputTensors currentOperand
              (checked.inputCoordinateOfGlobal currentOperand globalCoordinate))
          operand (replacement inputCoordinate) := by
    funext currentOperand
    by_cases hOperand : currentOperand = operand
    · subst currentOperand
      simp only [Function.update_self]
      rfl
    · simp only [Function.update_of_ne hOperand]
  simp only [einsumProductTensor_get, Rep.hAdd_apply]
  rw [sampled_update (leftTensor + rightTensor),
    sampled_update leftTensor, sampled_update rightTensor]
  simp only [Rep.hAdd_apply]
  exact prod_ofFn_update_add
    (fun currentOperand =>
      inputTensors currentOperand
        (checked.inputCoordinateOfGlobal currentOperand globalCoordinate))
    operand (leftTensor inputCoordinate) (rightTensor inputCoordinate)

/--
Independent einsum denotation: sum ordered operand products over contraction
coordinates while retaining the output labels in the user's requested order.

Both operand multiplication and contraction addition have explicit
left-to-right orders. This is the semantic contract used for scalar types,
such as IEEE floats, whose operations do not satisfy the algebraic laws
required by unordered finite products and sums.
-/
def denoteEinsum {R : Type u}
    [Storage R] [Add R] [Mul R] [OfNat R 0] [OfNat R 1]
    (checked : CheckedEinsum) (inputTensors : checked.InputTensors R) :
    checked.OutputTensor R :=
  Rep.ofFn fun outputCoordinate =>
    coordinateSum
      (checked.contractedAxes.map checked.axisLength)
      (fun contractionCoordinate =>
        einsumProductTensor checked inputTensors <|
          checked.reconstructedGlobalCoordinate outputCoordinate <|
            AxisTuple.coordEquiv checked.axisLength checked.contractedAxes
              contractionCoordinate)
      0

/--
Evaluate an einsum using any explicit reconstruction of the complete global
coordinate from the retained output coordinate and contracted-axis
coordinates.

The premises state exactly that the reconstruction remains in the requested
output fiber and recovers every supplied contracted-axis tuple. This theorem
is independent of rank and operand count, making it suitable for relating
symbolic einsum expressions to established finite-sum operations.
-/
theorem denoteEinsum_apply_reconstructed {R : Type u}
    [Storage R] [AddCommMonoid R] [Monoid R]
    (checked : CheckedEinsum)
    (inputTensors : checked.InputTensors R)
    (outputCoordinate : Coord checked.output)
    (globalCoordinate :
      AxisTuple checked.axisLength checked.contractedAxes →
        Coord (checked.globalAxes.map checked.axisLength))
    (retainsOutput :
      ∀ contractionCoordinate,
        checked.outputCoordinateOfGlobal
            (globalCoordinate contractionCoordinate) =
          outputCoordinate)
    (recoversContraction :
      ∀ contractionCoordinate,
        AxisTuple.select checked.contraction_axis_mem_global
            (checked.globalTensorCoordinateEquiv
              (globalCoordinate contractionCoordinate)) =
          contractionCoordinate) :
    denoteEinsum checked inputTensors outputCoordinate =
      ∑ contractionCoordinate,
        einsumProductTensor checked inputTensors
          (globalCoordinate contractionCoordinate) := by
  simp only [denoteEinsum, Rep.get_ofFn]
  rw [coordinateSum_eq_add_sum, zero_add]
  let contractionCoordinateEquiv :
      Coord (checked.contractedAxes.map checked.axisLength) ≃
        AxisTuple checked.axisLength checked.contractedAxes :=
    AxisTuple.coordEquiv checked.axisLength checked.contractedAxes
  refine Fintype.sum_equiv
    contractionCoordinateEquiv _ _ ?_
  intro coordinate
  let contractionCoordinate :=
    contractionCoordinateEquiv coordinate
  let reconstructedFiber :
      Fiber checked.outputCoordinateOfGlobal outputCoordinate :=
    ⟨checked.reconstructedGlobalCoordinate outputCoordinate
        contractionCoordinate,
      checked.outputCoordinateOfGlobal_reconstructedGlobalCoordinate
        outputCoordinate contractionCoordinate⟩
  let candidate :
      Fiber checked.outputCoordinateOfGlobal outputCoordinate :=
    ⟨globalCoordinate contractionCoordinate,
      retainsOutput contractionCoordinate⟩
  have reconstructedContraction :
      checked.contractionFiberEquiv outputCoordinate reconstructedFiber =
        contractionCoordinate := by
    rw [checked.contractionFiberEquiv_apply outputCoordinate]
    exact checked.contractionOf_reconstructedGlobalCoordinate
      outputCoordinate contractionCoordinate
  have candidateContraction :
      checked.contractionFiberEquiv outputCoordinate candidate =
        contractionCoordinate := by
    rw [checked.contractionFiberEquiv_apply outputCoordinate]
    simpa only [candidate] using
      recoversContraction contractionCoordinate
  have reconstructedFiber_eq_candidate : reconstructedFiber = candidate := by
    apply
      (checked.contractionFiberEquiv outputCoordinate).injective
    rw [reconstructedContraction, candidateContraction]
  simpa only [reconstructedFiber, candidate, contractionCoordinate] using
    congrArg
      (fun coordinate =>
        einsumProductTensor checked inputTensors coordinate.1)
      reconstructedFiber_eq_candidate

/--
For lawful additive and multiplicative monoids, the ordered denotation is the
usual fiber sum of source-ordered operand products.
-/
theorem denoteEinsum_eq_push {R : Type u}
    [Storage R] [AddCommMonoid R] [Monoid R]
    (checked : CheckedEinsum) (inputTensors : checked.InputTensors R) :
    denoteEinsum checked inputTensors =
      Rep.push checked.outputCoordinateOfGlobal
        (einsumProductTensor checked inputTensors) := by
  ext outputCoordinate
  let fiberEquiv :=
    checked.contractionFiberEquiv outputCoordinate
  rw [denoteEinsum_apply_reconstructed checked inputTensors outputCoordinate
    (fun contractionCoordinate =>
      (fiberEquiv.symm contractionCoordinate).1)
    (fun contractionCoordinate =>
      (fiberEquiv.symm contractionCoordinate).2)
    (fun contractionCoordinate => by
      change checked.contractionFiberEquiv outputCoordinate
          ((checked.contractionFiberEquiv outputCoordinate).symm
            contractionCoordinate) =
        contractionCoordinate
      exact Equiv.apply_symm_apply _ _)]
  rw [Rep.push_apply]
  symm
  refine Fintype.sum_equiv fiberEquiv _ _ ?_
  intro fiberCoordinate
  simpa only using congrArg
    (fun coordinate =>
      einsumProductTensor checked inputTensors coordinate.1)
    (fiberEquiv.symm_apply_apply fiberCoordinate).symm

/--
Pull an output cotangent back to one selected einsum operand.

For each complete logical-axis assignment, the selected operand factor is
replaced by one, leaving the product of all other operands. That product is
multiplied by the output cotangent at the retained output coordinate, then
all contributions are pushed back to the selected operand's physical
coordinate. The push accounts uniformly for contraction, singleton
broadcasting, omitted ellipsis slots, and repeated-label diagonal scatter.

The value stored in `inputTensors operand` is deliberately ignored: an einsum
is multilinear, so its partial derivative with respect to one operand depends
only on the other operands. Commutative multiplication is required to express
the result using the standard tensor pairing with the selected tangent as its
left factor.
-/
def einsumOperandVjp {R : Type u} [Storage R] [CommSemiring R]
    (checked : CheckedEinsum)
    (inputTensors : checked.InputTensors R)
    (operand : Fin checked.inputShapes.length)
    (outputCotangent : checked.OutputTensor R) :
    Rep R (checked.inputShapes.get operand) :=
  Rep.push (checked.inputCoordinateOfGlobal operand) <|
    Rep.ofFn fun globalCoordinate =>
      (List.ofFn
        (Function.update
          (fun currentOperand =>
            inputTensors currentOperand <|
              checked.inputCoordinateOfGlobal currentOperand globalCoordinate)
          operand 1)).prod *
        outputCotangent
          (checked.outputCoordinateOfGlobal globalCoordinate)

/--
Einsum is additive in every individual operand over an arbitrary semiring.

The selected operand may have any checked shape and may occur anywhere in the
heterogeneous input family. The result follows by distributing its ordered
product at each complete logical assignment and then distributing the finite
sum over every contraction fiber.
-/
@[grind =] theorem denoteEinsum_update_add {R : Type u}
    [Storage R] [Semiring R]
    (checked : CheckedEinsum)
    (inputTensors : checked.InputTensors R)
    (operand : Fin checked.inputShapes.length)
    (leftTensor rightTensor :
      Rep R (checked.inputShapes.get operand)) :
    denoteEinsum checked
        (Function.update inputTensors operand (leftTensor + rightTensor)) =
      denoteEinsum checked
          (Function.update inputTensors operand leftTensor) +
        denoteEinsum checked
          (Function.update inputTensors operand rightTensor) := by
  ext outputCoordinate
  rw [denoteEinsum_eq_push, denoteEinsum_eq_push, denoteEinsum_eq_push,
    einsumProductTensor_update_add]
  simp only [Rep.push_apply, Rep.hAdd_apply,
    Finset.sum_add_distrib]

/--
The selected-operand einsum VJP is adjoint to replacing that operand by an
arbitrary tangent tensor.

The statement quantifies over an arbitrary checked operand position and its
dependent physical shape. It therefore covers any operand count,
heterogeneous ranks, repeated-label diagonals, right-aligned ellipses,
singleton broadcasting, scalar outputs, and zero-length axes without
operation-specific cases.
-/
@[grind =] theorem dot_denoteEinsum_update_eq_dot_einsumOperandVjp
    {R : Type u} [Storage R] [CommSemiring R]
    (checked : CheckedEinsum)
    (inputTensors : checked.InputTensors R)
    (operand : Fin checked.inputShapes.length)
    (inputTangent : Rep R (checked.inputShapes.get operand))
    (outputCotangent : checked.OutputTensor R) :
    Rep.dot
        (denoteEinsum checked
          (Function.update inputTensors operand inputTangent))
        outputCotangent =
      Rep.dot inputTangent
        (einsumOperandVjp checked inputTensors operand outputCotangent) := by
  have dot_comm {shape : Shape} (leftTensor rightTensor : Rep R shape) :
      Rep.dot leftTensor rightTensor =
        Rep.dot rightTensor leftTensor := by
    simp only [Rep.dot, mul_comm]
  calc
    Rep.dot
          (denoteEinsum checked
            (Function.update inputTensors operand inputTangent))
          outputCotangent =
      Rep.dot
          (einsumProductTensor checked
            (Function.update inputTensors operand inputTangent))
          (Rep.pull checked.outputCoordinateOfGlobal outputCotangent) := by
      rw [denoteEinsum_eq_push]
      exact
        Rep.dot_push_eq_dot_pull checked.outputCoordinateOfGlobal
          (einsumProductTensor checked
            (Function.update inputTensors operand inputTangent))
          outputCotangent
    _ =
        Rep.dot
          (Rep.pull
            (checked.inputCoordinateOfGlobal operand) inputTangent)
          (Rep.ofFn fun globalCoordinate =>
            (List.ofFn
              (Function.update
                (fun currentOperand =>
                  inputTensors currentOperand <|
                    checked.inputCoordinateOfGlobal currentOperand
                      globalCoordinate)
                operand 1)).prod *
              outputCotangent
                (checked.outputCoordinateOfGlobal globalCoordinate)) := by
      classical
      unfold Rep.dot
      simp only [einsumProductTensor_get]
      apply Finset.sum_congr rfl
      intro globalCoordinate _
      simp only [Rep.pull_apply, Rep.get_ofFn]
      have sampled_update :
          (fun currentOperand =>
              Function.update inputTensors operand inputTangent currentOperand
                (checked.inputCoordinateOfGlobal currentOperand
                  globalCoordinate)) =
            Function.update
              (fun currentOperand =>
                inputTensors currentOperand <|
                  checked.inputCoordinateOfGlobal currentOperand
                    globalCoordinate)
              operand
              (inputTangent <|
                checked.inputCoordinateOfGlobal operand globalCoordinate) := by
        funext currentOperand
        by_cases hOperand : currentOperand = operand
        · subst currentOperand
          simp only [Function.update_self]
        · simp only [Function.update_of_ne hOperand]
      rw [sampled_update, prod_ofFn_update_eq_mul_update_one, mul_assoc]
    _ =
        Rep.dot
          (Rep.ofFn fun globalCoordinate =>
            (List.ofFn
              (Function.update
                (fun currentOperand =>
                  inputTensors currentOperand <|
                    checked.inputCoordinateOfGlobal currentOperand
                      globalCoordinate)
                operand 1)).prod *
              outputCotangent
                (checked.outputCoordinateOfGlobal globalCoordinate))
          (Rep.pull
            (checked.inputCoordinateOfGlobal operand) inputTangent) :=
      dot_comm _ _
    _ =
        Rep.dot
          (Rep.push
            (checked.inputCoordinateOfGlobal operand)
            (Rep.ofFn fun globalCoordinate =>
              (List.ofFn
                (Function.update
                  (fun currentOperand =>
                    inputTensors currentOperand <|
                      checked.inputCoordinateOfGlobal currentOperand
                        globalCoordinate)
                  operand 1)).prod *
                outputCotangent
                  (checked.outputCoordinateOfGlobal globalCoordinate)))
          inputTangent := by
      exact
        (Rep.dot_push_eq_dot_pull
          (checked.inputCoordinateOfGlobal operand)
          (Rep.ofFn fun globalCoordinate =>
            (List.ofFn
              (Function.update
                (fun currentOperand =>
                  inputTensors currentOperand <|
                    checked.inputCoordinateOfGlobal currentOperand
                      globalCoordinate)
                operand 1)).prod *
              outputCotangent
                (checked.outputCoordinateOfGlobal globalCoordinate))
          inputTangent).symm
    _ =
        Rep.dot inputTangent
          (einsumOperandVjp checked inputTensors operand
            outputCotangent) := by
      simpa only [einsumOperandVjp] using
        dot_comm
          (Rep.push
            (checked.inputCoordinateOfGlobal operand)
            (Rep.ofFn fun globalCoordinate =>
              (List.ofFn
                (Function.update
                  (fun currentOperand =>
                    inputTensors currentOperand <|
                      checked.inputCoordinateOfGlobal currentOperand
                        globalCoordinate)
                  operand 1)).prod *
                outputCotangent
                  (checked.outputCoordinateOfGlobal globalCoordinate)))
          inputTangent

/--
Einsum partitions complete logical assignments by their output coordinate,
so summing the output recovers the sum of all ordered operand products.
-/
@[grind =] theorem sum_denoteEinsum {R : Type u}
    [Storage R] [AddCommMonoid R] [Monoid R]
    (checked : CheckedEinsum) (inputTensors : checked.InputTensors R) :
    (∑ outputCoordinate, denoteEinsum checked inputTensors outputCoordinate) =
      ∑ globalCoordinate,
        einsumProductTensor checked inputTensors globalCoordinate :=
  by
    rw [denoteEinsum_eq_push]
    exact Rep.sum_push checked.outputCoordinateOfGlobal
      (einsumProductTensor checked inputTensors)

end Semantics

end TorchLean.Tensor.Internal
