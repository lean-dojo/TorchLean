/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.Check.Einsum
public import TorchLean.Tensor.Rep.Traversal
public import Mathlib.Algebra.BigOperators.Fin
public import TorchLean.Tensor.Rep.Fiber.Mean -- shake: keep
public import TorchLean.Tensor.Rep.Fiber.Product -- shake: keep
public import TorchLean.Tensor.Rep.Fiber.Tie -- shake: keep
import Mathlib.Data.Matrix.Mul
import Mathlib.LinearAlgebra.Multilinear.Basic

/-!
# Algebraic semantics for einsum

A checked einsum ranges over one finite assignment of every logical axis. Each operand reads that
assignment through three coordinate operations:

1. Select the labels the operand uses, keeping repeated labels.
2. Read every occurrence of a repeated label at the same selected coordinate, which gives
   diagonal indexing.
3. Project resolved dimensions to the operand's physical dimensions, sending each singleton
   dimension to its unique coordinate.

Operand values are multiplied in source order, and contracted coordinates are accumulated in
row-major order from zero. Without a contracted axis each output is its product, with no addition.
These ordered semantics need only `Mul`, `Add` and the scalars zero and one, so the same
denotation applies to IEEE floating-point values without asserting associativity or commutativity
instances that floats do not satisfy. When scalar addition forms a commutative monoid,
`denoteEinsum_eq_push` identifies the ordered denotation with the `Rep.push` fiber sum of ordered
products. Distributivity enters only in the multilinearity theorems, and commutative
multiplication only in the selected-operand adjoint.

Missing leading ellipsis slots need no special case. An operand omits those logical axes, so its
value is constant while the omitted coordinates vary, which is singleton broadcasting under
right-aligned ellipses.
-/

@[expose] public section

namespace TorchLean.Einops

open scoped BigOperators
open Tensor (Rep)
open Shape (Coord)

universe u

namespace Einsum

/-- The heterogeneous family of input tensors accepted by a checked einsum. -/
abbrev InputTensors (checked : Einsum) (R : Type u)
    [Storage R] :=
  (operand : Fin checked.inputShapes.length) →
    Rep R (checked.inputShapes.get operand)

/-- The output tensor type inferred by a checked einsum. -/
abbrev OutputTensor (checked : Einsum) (R : Type u)
    [Storage R] :=
  Rep R checked.output

/-- Expanded operand-axis lists and input shapes have the same count. -/
theorem input_axes_length (checked : Einsum) :
    checked.inputAxes.length = checked.inputShapes.length := by
  simp [inputAxes, EinsumPattern.inputAxes, checked.input_count]

/-- Expanded logical labels used by one input operand. -/
def operandAxes (checked : Einsum)
    (operand : Fin checked.inputShapes.length) : List EinsumAxis :=
  checked.inputAxes.get
    ⟨operand.val, by
      rw [checked.input_axes_length]
      exact operand.isLt⟩

/-- Every logical label used by an operand belongs to the global assignment. -/
theorem input_axis_mem_global (checked : Einsum)
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
  simpa [globalAxes, inputAxes, EinsumPattern.globalAxes] using hFlattened

/--
The physical shape of an operand broadcasts to its resolved logical shape.

The checker's per-dimension certificate reaches the coordinate map of the denotation through this
theorem.
-/
theorem operand_shape_broadcastable (checked : Einsum)
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
def globalCoordEquiv (checked : Einsum) :
    Coord (checked.globalAxes.map checked.axisLength) ≃
      AxisTuple checked.axisLength checked.globalAxes :=
  AxisTuple.coordEquiv checked.axisLength checked.globalAxes

/-- Convert an output tensor coordinate to its named logical coordinates. -/
def outputCoordEquiv (checked : Einsum) :
    Coord checked.output ≃
      AxisTuple checked.axisLength checked.outputAxes :=
  AxisTuple.coordEquiv checked.axisLength checked.outputAxes

/--
Retain the requested output labels from a complete logical-axis assignment.
-/
def outputOfGlobal (checked : Einsum) :
    Coord (checked.globalAxes.map checked.axisLength) →
      Coord checked.output :=
  fun globalCoordinate =>
    checked.outputCoordEquiv.symm <|
      AxisTuple.select checked.output_axis_mem_global <|
        checked.globalCoordEquiv globalCoordinate

/--
Read one operand coordinate from a complete logical-axis assignment.

Repeated labels select one shared coordinate before singleton broadcasting,
so this single map covers ordinary indexing, diagonals, and broadcasting.
-/
def inputOfGlobal (checked : Einsum)
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
            checked.globalCoordEquiv globalCoordinate

/--
Every contracted logical axis belongs to the complete global-axis
assignment.
-/
theorem contraction_axis_mem_global (checked : Einsum) :
    ∀ ⦃axis⦄,
      axis ∈ checked.contractedAxes →
        axis ∈ checked.globalAxes :=
  fun _ hAxis => (List.mem_filter.mp hAxis).1

/--
The fiber of an output coordinate is equivalent to the assignments of every global logical axis
absent from the output, which are the contracted axes.

The equivalence holds whatever the operand count, tensor ranks, label repetition, broadcasting and
axis lengths.
-/
noncomputable def contractionFiberEquiv (checked : Einsum)
    (outputCoordinate : Coord checked.output) :
    Fiber checked.outputOfGlobal outputCoordinate ≃
      AxisTuple checked.axisLength checked.contractedAxes :=
  AxisTuple.coordFiberEquiv checked.outputCoordEquiv checked.globalCoordEquiv
    checked.output_axes_nodup checked.global_axes_nodup
    checked.output_axis_mem_global outputCoordinate

/--
The contraction-fiber equivalence reads the contracted axes from the complete
global logical-axis assignment.
-/
@[simp] theorem contractionFiberEquiv_apply
    (checked : Einsum)
    (outputCoordinate : Coord checked.output)
    (globalCoordinate :
      Fiber checked.outputOfGlobal outputCoordinate) :
    checked.contractionFiberEquiv outputCoordinate globalCoordinate =
      AxisTuple.select checked.contraction_axis_mem_global
        (checked.globalCoordEquiv globalCoordinate.1) :=
  rfl

/--
Every output coordinate has one complete assignment for each setting of the contracted logical
axes, so its fiber cardinality is the product of the contracted lengths. The formula includes
scalar outputs, contractions of no axes, and empty fibers caused by a zero-length contracted axis.
-/
theorem contraction_fiber_card (checked : Einsum)
    (outputCoordinate : Coord checked.output) :
    Fintype.card
        (Fiber checked.outputOfGlobal outputCoordinate) =
      (checked.contractedAxes.map checked.axisLength).prod := by
  classical
  let contractedAxes := checked.contractedAxes
  calc
    Fintype.card
          (Fiber checked.outputOfGlobal outputCoordinate) =
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
def globalCoord
    (checked : Einsum)
    (outputCoordinate : Coord checked.output)
    (contractionCoordinate :
      AxisTuple checked.axisLength checked.contractedAxes) :
    Coord (checked.globalAxes.map checked.axisLength) :=
  checked.globalCoordEquiv.symm <|
    AxisTuple.merge checked.outputAxes
      (checked.outputCoordEquiv outputCoordinate) contractionCoordinate

/-- Reconstructing a global coordinate preserves its supplied output coordinate. -/
@[simp] theorem outputOfGlobal_globalCoord
    (checked : Einsum)
    (outputCoordinate : Coord checked.output)
    (contractionCoordinate :
      AxisTuple checked.axisLength checked.contractedAxes) :
    checked.outputOfGlobal
        (checked.globalCoord outputCoordinate
          contractionCoordinate) =
      outputCoordinate := by
  apply checked.outputCoordEquiv.injective
  simp only [outputOfGlobal, globalCoord,
    Equiv.apply_symm_apply]
  exact AxisTuple.select_merge_left
    checked.output_axes_nodup checked.output_axis_mem_global
    (checked.outputCoordEquiv outputCoordinate)
    contractionCoordinate

/-- Selecting contracted axes from a reconstructed coordinate recovers their values. -/
@[simp] theorem select_globalCoord
    (checked : Einsum)
    (outputCoordinate : Coord checked.output)
    (contractionCoordinate :
      AxisTuple checked.axisLength checked.contractedAxes) :
    AxisTuple.select checked.contraction_axis_mem_global
        (checked.globalCoordEquiv
          (checked.globalCoord outputCoordinate
            contractionCoordinate)) =
      contractionCoordinate := by
  simp only [globalCoord, Equiv.apply_symm_apply]
  exact AxisTuple.select_merge_right
    checked.global_axes_nodup
    (checked.outputCoordEquiv outputCoordinate)
    contractionCoordinate

end Einsum

namespace Semantics

/--
For lawful addition, nested row-major coordinate addition from `initial` is `initial` plus the
finite sum.

It is the one place where the ordered loop becomes a big operator, and the lowering module
reuses it.
-/
theorem coordinateSum_eq_add_sum {R : Type u} [AddCommMonoid R]
    (shape : Shape) (values : Coord shape → R) (initial : R) :
    Coord.foldl shape (fun total coordinate => total + values coordinate) initial =
      initial + ∑ coordinate, values coordinate := by
  induction shape generalizing initial with
  | nil =>
      simp only [Coord.foldl]
      let unique : Unique (Coord []) := {
        default := PUnit.unit
        uniq := fun coordinate => by
          cases coordinate
          rfl }
      rw [@Fintype.sum_unique R (Coord []) _ _ unique]
  | cons length shape inductionHypothesis =>
      simp only [Coord.foldl]
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
Contract values over every coordinate of a shape: with no axis the single value itself, and
otherwise their row-major `Coord.foldl` sum seeded with `0`, as `Tensor.contract` sums.

Without a contracted axis nothing is added, so each product keeps every bit (a negative zero stays
negative).
-/
def contraction {R : Type u} [Add R] [OfNat R 0] :
    (shape : Shape) → (Coord shape → R) → R
  | [], values => values PUnit.unit
  | length :: shape, values =>
      Coord.foldl (length :: shape) (fun total coordinate => total + values coordinate) 0

/-- With lawful addition, a contraction is the finite sum of its values. -/
theorem contraction_eq_sum {R : Type u} [AddCommMonoid R]
    (shape : Shape) (values : Coord shape → R) :
    contraction shape values = ∑ coordinate, values coordinate := by
  cases shape with
  | nil =>
      let unique : Unique (Coord []) := {
        default := PUnit.unit
        uniq := fun coordinate => by
          cases coordinate
          rfl }
      rw [@Fintype.sum_unique R (Coord []) _ _ unique]
      rfl
  | cons length shape =>
      rw [contraction, coordinateSum_eq_add_sum, zero_add]

/--
In a commutative monoid, the product with one factor replaced by `replacement` is `replacement`
times the product with that factor replaced by one.

The selected-operand einsum adjoint uses this identity as its algebraic step. It stays private
because the public adjoint theorem is stated about tensors.
-/
private theorem prod_ofFn_update_eq_mul
    {R : Type u} [CommMonoid R] {n : Nat}
    (values : Fin n → R) (operand : Fin n) (replacement : R) :
    (List.ofFn (Function.update values operand replacement)).prod =
      replacement *
        (List.ofFn (Function.update values operand 1)).prod := by
  classical
  simp [List.prod_ofFn, Finset.prod_update_of_mem]

/--
The tensor of ordered operand products over complete logical assignments.

`List.ofFn` enumerates operand indices from left to right, and `List.foldl` from `1` records that
exact multiplication order without assuming monoid laws.
-/
def einsumProduct {R : Type u} [Storage R] [Mul R] [OfNat R 1]
    (checked : Einsum) (inputTensors : checked.InputTensors R) :
    Rep R (checked.globalAxes.map checked.axisLength) :=
  Rep.ofFn fun globalCoordinate =>
    (List.ofFn (fun operand =>
      inputTensors operand <|
        checked.inputOfGlobal operand globalCoordinate)).foldl
      (· * ·) 1

/--
Reading the product tensor exposes the source-ordered multiplicative fold.
-/
theorem einsumProduct_get_ordered {R : Type u}
    [Storage R] [Mul R] [OfNat R 1]
    (checked : Einsum) (inputTensors : checked.InputTensors R)
    (globalCoordinate : Coord (checked.globalAxes.map checked.axisLength)) :
    (einsumProduct checked inputTensors).get globalCoordinate =
      (List.ofFn (fun operand =>
        inputTensors operand <|
          checked.inputOfGlobal operand globalCoordinate)).foldl
        (· * ·) 1 :=
  Rep.get_ofFn _ _

/--
For a lawful monoid, the source-ordered fold is the usual list product.
-/
@[simp, grind =] theorem einsumProduct_get {R : Type u}
    [Storage R] [Monoid R]
    (checked : Einsum) (inputTensors : checked.InputTensors R)
    (globalCoordinate : Coord (checked.globalAxes.map checked.axisLength)) :
    (einsumProduct checked inputTensors).get globalCoordinate =
      (List.ofFn fun operand =>
        inputTensors operand <|
          checked.inputOfGlobal operand globalCoordinate).prod := by
  rw [einsumProduct_get_ordered, ← List.prod_eq_foldl]

/--
The ordered-product tensor is additive in any selected input operand, with every other operand
held fixed by `Function.update`.

The statement covers operands of different ranks, repeated labels, singleton broadcasting and any
position in a positive-arity einsum. The proof never moves the selected operand past its
neighbors, so any semiring suffices, commutative or not.
-/
@[grind =, einops =] theorem einsumProduct_update_add {R : Type u}
    [Storage R] [Semiring R]
    (checked : Einsum)
    (inputTensors : checked.InputTensors R)
    (operand : Fin checked.inputShapes.length)
    (leftTensor rightTensor :
      Rep R (checked.inputShapes.get operand)) :
    einsumProduct checked
        (Function.update inputTensors operand (leftTensor + rightTensor)) =
      einsumProduct checked
          (Function.update inputTensors operand leftTensor) +
        einsumProduct checked
          (Function.update inputTensors operand rightTensor) := by
  ext globalCoordinate
  let inputCoordinate :=
    checked.inputOfGlobal operand globalCoordinate
  have sampled_update (replacement :
      Rep R (checked.inputShapes.get operand)) :
      (fun currentOperand =>
          Function.update inputTensors operand replacement currentOperand
            (checked.inputOfGlobal currentOperand globalCoordinate)) =
        Function.update
          (fun currentOperand =>
            inputTensors currentOperand
              (checked.inputOfGlobal currentOperand globalCoordinate))
          operand (replacement inputCoordinate) := by
    funext currentOperand
    by_cases hOperand : currentOperand = operand
    · subst currentOperand
      simp only [Function.update_self]
      rfl
    · simp only [Function.update_of_ne hOperand]
  simp only [einsumProduct_get, Tensor.add_apply]
  rw [sampled_update (leftTensor + rightTensor),
    sampled_update leftTensor, sampled_update rightTensor]
  simp only [Tensor.add_apply]
  exact (MultilinearMap.mkPiAlgebraFin ℕ _ R).map_update_add
    (fun currentOperand =>
      inputTensors currentOperand
        (checked.inputOfGlobal currentOperand globalCoordinate))
    operand (leftTensor inputCoordinate) (rightTensor inputCoordinate)

/--
Independent einsum denotation: contract ordered operand products over the contracted coordinates
and keep the output labels in the order the user requested.

Operand multiplication and contraction addition both run left to right in a fixed order, and a
pattern without a contracted axis adds nothing. Scalar types such as IEEE floats, whose
operations lack the laws that unordered finite products and sums need, are specified against this
denotation.
-/
def denoteEinsum {R : Type u}
    [Storage R] [Add R] [Mul R] [OfNat R 0] [OfNat R 1]
    (checked : Einsum) (inputTensors : checked.InputTensors R) :
    checked.OutputTensor R :=
  Rep.ofFn fun outputCoordinate =>
    contraction
      (checked.contractedAxes.map checked.axisLength)
      (fun contractionCoordinate =>
        einsumProduct checked inputTensors <|
          checked.globalCoord outputCoordinate <|
            AxisTuple.coordEquiv checked.axisLength checked.contractedAxes
              contractionCoordinate)

/--
For lawful addition, an einsum entry is the finite sum of ordered products over any
reconstruction of the complete global coordinate from the output coordinate and a contracted-axis
tuple.

The premises say that the reconstruction stays in the requested output fiber and recovers every
supplied contracted-axis tuple. Nothing depends on rank or operand count, so the theorem relates
symbolic einsum expressions to established finite-sum operations.
-/
theorem denoteEinsum_apply {R : Type u}
    [Storage R] [AddCommMonoid R] [Mul R] [OfNat R 1]
    (checked : Einsum)
    (inputTensors : checked.InputTensors R)
    (outputCoordinate : Coord checked.output)
    (globalCoordinate :
      AxisTuple checked.axisLength checked.contractedAxes →
        Coord (checked.globalAxes.map checked.axisLength))
    (retainsOutput :
      ∀ contractionCoordinate,
        checked.outputOfGlobal
            (globalCoordinate contractionCoordinate) =
          outputCoordinate)
    (recoversContraction :
      ∀ contractionCoordinate,
        AxisTuple.select checked.contraction_axis_mem_global
            (checked.globalCoordEquiv
              (globalCoordinate contractionCoordinate)) =
          contractionCoordinate) :
    denoteEinsum checked inputTensors outputCoordinate =
      ∑ contractionCoordinate,
        einsumProduct checked inputTensors
          (globalCoordinate contractionCoordinate) := by
  simp only [denoteEinsum, Rep.get_ofFn]
  rw [contraction_eq_sum]
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
      Fiber checked.outputOfGlobal outputCoordinate :=
    ⟨checked.globalCoord outputCoordinate
        contractionCoordinate,
      checked.outputOfGlobal_globalCoord
        outputCoordinate contractionCoordinate⟩
  let candidate :
      Fiber checked.outputOfGlobal outputCoordinate :=
    ⟨globalCoordinate contractionCoordinate,
      retainsOutput contractionCoordinate⟩
  have reconstructedContraction :
      checked.contractionFiberEquiv outputCoordinate reconstructedFiber =
        contractionCoordinate := by
    rw [checked.contractionFiberEquiv_apply outputCoordinate]
    exact checked.select_globalCoord
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
        einsumProduct checked inputTensors coordinate.1)
      reconstructedFiber_eq_candidate

/--
For lawful commutative addition, the ordered denotation is the usual fiber
sum of source-ordered operand products. No multiplication laws are needed.
-/
theorem denoteEinsum_eq_push {R : Type u}
    [Storage R] [AddCommMonoid R] [Mul R] [OfNat R 1]
    (checked : Einsum) (inputTensors : checked.InputTensors R) :
    denoteEinsum checked inputTensors =
      Rep.push checked.outputOfGlobal
        (einsumProduct checked inputTensors) := by
  ext outputCoordinate
  let fiberEquiv :=
    checked.contractionFiberEquiv outputCoordinate
  rw [denoteEinsum_apply checked inputTensors outputCoordinate
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
      einsumProduct checked inputTensors coordinate.1)
    (fiberEquiv.symm_apply_apply fiberCoordinate).symm

/--
Pull an output cotangent back to one selected einsum operand.

For each complete logical-axis assignment, the selected operand factor is replaced by one, leaving
the product of all other operands. That product is multiplied by the output cotangent at the
retained output coordinate, then all contributions are pushed back to the selected operand's
physical coordinate. The push accounts uniformly for contraction, singleton broadcasting, omitted
ellipsis slots, and repeated-label diagonal scatter.

The value stored in `inputTensors operand` is ignored, because an einsum is multilinear: its
partial derivative with respect to one operand depends only on the other operands. Commutative
multiplication is required to express the result through the standard tensor pairing with the
selected tangent as its left factor.
-/
def einsumOperandVJP {R : Type u} [Storage R] [CommSemiring R]
    (checked : Einsum)
    (inputTensors : checked.InputTensors R)
    (operand : Fin checked.inputShapes.length)
    (outputCotangent : checked.OutputTensor R) :
    Rep R (checked.inputShapes.get operand) :=
  Rep.push (checked.inputOfGlobal operand) <|
    Rep.ofFn fun globalCoordinate =>
      (List.ofFn
        (Function.update
          (fun currentOperand =>
            inputTensors currentOperand <|
              checked.inputOfGlobal currentOperand globalCoordinate)
          operand 1)).prod *
        outputCotangent
          (checked.outputOfGlobal globalCoordinate)

/--
Einsum is additive in every individual operand over an arbitrary semiring.

The selected operand may have any checked shape and may occur anywhere in the
heterogeneous input family. The result follows by distributing its ordered
product at each complete logical assignment and then distributing the finite
sum over every contraction fiber.
-/
@[grind =, einops =, einops_simps] theorem denoteEinsum_update_add {R : Type u}
    [Storage R] [Semiring R]
    (checked : Einsum)
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
    einsumProduct_update_add]
  simp only [Rep.push_apply, Tensor.add_apply,
    Finset.sum_add_distrib]

/--
The selected-operand einsum VJP is adjoint to replacing that operand by an arbitrary tangent
tensor.

The statement quantifies over an arbitrary checked operand position and its dependent physical
shape. It therefore covers any operand count, heterogeneous ranks, repeated-label diagonals,
right-aligned ellipses, singleton broadcasting, scalar outputs, and zero-length axes without
operation-specific cases.

The pairing is the bilinear `Rep.dot` over a commutative semiring, so the identity is algebraic,
with no conjugation and no analytic derivative.
-/
@[grind =, einops =, einops_simps] theorem dot_denoteEinsum_update
    {R : Type u} [Storage R] [CommSemiring R]
    (checked : Einsum)
    (inputTensors : checked.InputTensors R)
    (operand : Fin checked.inputShapes.length)
    (inputTangent : Rep R (checked.inputShapes.get operand))
    (outputCotangent : checked.OutputTensor R) :
    Rep.dot
        (denoteEinsum checked
          (Function.update inputTensors operand inputTangent))
        outputCotangent =
      Rep.dot inputTangent
        (einsumOperandVJP checked inputTensors operand outputCotangent) := by
  have dot_comm {shape : Shape} (leftTensor rightTensor : Rep R shape) :
      Rep.dot leftTensor rightTensor =
        Rep.dot rightTensor leftTensor :=
    dotProduct_comm (fun coordinate => leftTensor coordinate)
      (fun coordinate => rightTensor coordinate)
  calc
    Rep.dot
          (denoteEinsum checked
            (Function.update inputTensors operand inputTangent))
          outputCotangent =
      Rep.dot
          (einsumProduct checked
            (Function.update inputTensors operand inputTangent))
          (Rep.pull checked.outputOfGlobal outputCotangent) := by
      rw [denoteEinsum_eq_push]
      exact
        Rep.dot_push_eq_dot_pull checked.outputOfGlobal
          (einsumProduct checked
            (Function.update inputTensors operand inputTangent))
          outputCotangent
    _ =
        Rep.dot
          (Rep.pull
            (checked.inputOfGlobal operand) inputTangent)
          (Rep.ofFn fun globalCoordinate =>
            (List.ofFn
              (Function.update
                (fun currentOperand =>
                  inputTensors currentOperand <|
                    checked.inputOfGlobal currentOperand
                      globalCoordinate)
                operand 1)).prod *
              outputCotangent
                (checked.outputOfGlobal globalCoordinate)) := by
      classical
      unfold Rep.dot
      simp only [einsumProduct_get]
      apply Finset.sum_congr rfl
      intro globalCoordinate _
      simp only [Rep.pull_apply, Rep.get_ofFn]
      have sampled_update :
          (fun currentOperand =>
              Function.update inputTensors operand inputTangent currentOperand
                (checked.inputOfGlobal currentOperand
                  globalCoordinate)) =
            Function.update
              (fun currentOperand =>
                inputTensors currentOperand <|
                  checked.inputOfGlobal currentOperand
                    globalCoordinate)
              operand
              (inputTangent <|
                checked.inputOfGlobal operand globalCoordinate) := by
        funext currentOperand
        by_cases hOperand : currentOperand = operand
        · subst currentOperand
          simp only [Function.update_self]
        · simp only [Function.update_of_ne hOperand]
      rw [sampled_update, prod_ofFn_update_eq_mul, mul_assoc]
    _ =
        Rep.dot
          (Rep.ofFn fun globalCoordinate =>
            (List.ofFn
              (Function.update
                (fun currentOperand =>
                  inputTensors currentOperand <|
                    checked.inputOfGlobal currentOperand
                      globalCoordinate)
                operand 1)).prod *
              outputCotangent
                (checked.outputOfGlobal globalCoordinate))
          (Rep.pull
            (checked.inputOfGlobal operand) inputTangent) :=
      dot_comm _ _
    _ =
        Rep.dot
          (Rep.push
            (checked.inputOfGlobal operand)
            (Rep.ofFn fun globalCoordinate =>
              (List.ofFn
                (Function.update
                  (fun currentOperand =>
                    inputTensors currentOperand <|
                      checked.inputOfGlobal currentOperand
                        globalCoordinate)
                  operand 1)).prod *
                outputCotangent
                  (checked.outputOfGlobal globalCoordinate)))
          inputTangent := by
      exact
        (Rep.dot_push_eq_dot_pull
          (checked.inputOfGlobal operand)
          (Rep.ofFn fun globalCoordinate =>
            (List.ofFn
              (Function.update
                (fun currentOperand =>
                  inputTensors currentOperand <|
                    checked.inputOfGlobal currentOperand
                      globalCoordinate)
                operand 1)).prod *
              outputCotangent
                (checked.outputOfGlobal globalCoordinate))
          inputTangent).symm
    _ =
        Rep.dot inputTangent
          (einsumOperandVJP checked inputTensors operand
            outputCotangent) := by
      simpa only [einsumOperandVJP] using
        dot_comm
          (Rep.push
            (checked.inputOfGlobal operand)
            (Rep.ofFn fun globalCoordinate =>
              (List.ofFn
                (Function.update
                  (fun currentOperand =>
                    inputTensors currentOperand <|
                      checked.inputOfGlobal currentOperand
                        globalCoordinate)
                  operand 1)).prod *
                outputCotangent
                  (checked.outputOfGlobal globalCoordinate)))
          inputTangent

/--
Einsum partitions complete logical assignments by their output coordinate,
so summing the output recovers the sum of all ordered operand products.
-/
@[grind =, einops =, einops_simps] theorem sum_denoteEinsum {R : Type u}
    [Storage R] [AddCommMonoid R] [Mul R] [OfNat R 1]
    (checked : Einsum) (inputTensors : checked.InputTensors R) :
    (∑ outputCoordinate, denoteEinsum checked inputTensors outputCoordinate) =
      ∑ globalCoordinate,
        einsumProduct checked inputTensors globalCoordinate := by
  rw [denoteEinsum_eq_push]
  exact Rep.sum_push checked.outputOfGlobal
    (einsumProduct checked inputTensors)

end Semantics

end TorchLean.Einops
