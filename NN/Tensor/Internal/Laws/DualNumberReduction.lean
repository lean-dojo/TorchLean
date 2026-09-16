/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import Mathlib.Algebra.DualNumber
public import NN.Tensor.Internal.Semantics.Transform.Reduction

/-!
# Dual numbers and product reduction

Forward-mode differentiation of a product reduction is stated here, at both the representation and
the checked-transform level: evaluating the reducer on dual numbers puts the leave-one-out Leibniz
formula in the tangent component, and that formula is adjoint to the zero-aware reverse map.

The results sit apart from `NN.Tensor.Internal.Representation.Fiber.Product` and
`NN.Tensor.Internal.Semantics.Transform.Reduction`, which define the formulas themselves, because
naming `DualNumber` costs mathlib's `TrivSqZeroExt` algebra and everything under it. Nothing in the
tensor runtime or in the layers built on it evaluates a tensor over dual numbers, so the facade
`NN.Tensor.Internal.Laws` leaves this module out and CI imports it directly.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal

open scoped BigOperators

universe u v

namespace Rep

/-- The primal component of a finite dual-number product is the product of primals. -/
private theorem fst_finset_prod_dualNumber
    {R : Type u} [CommSemiring R] {ι : Type*}
    (coordinates : Finset ι)
    (inputTensor inputTangent : ι → R) :
    TrivSqZeroExt.fst
        (coordinates.prod fun coordinate : ι =>
          (TrivSqZeroExt.inl (M := R) (inputTensor coordinate) +
            TrivSqZeroExt.inr (R := R) (inputTangent coordinate) :
            DualNumber R)) =
      coordinates.prod inputTensor := by
  classical
  induction coordinates using Finset.induction_on with
  | empty =>
      simp
  | @insert coordinate coordinates hNotMem inductionHypothesis =>
      simp only [Finset.prod_insert hNotMem, TrivSqZeroExt.fst_mul,
        TrivSqZeroExt.fst_add, TrivSqZeroExt.fst_inl,
        TrivSqZeroExt.fst_inr, add_zero, inductionHypothesis]

/-- The tangent component of a finite product is its sum of leave-one-out products. -/
private theorem snd_finset_prod_dualNumber
    {R : Type u} [CommSemiring R] {ι : Type*} [DecidableEq ι]
    (coordinates : Finset ι)
    (inputTensor inputTangent : ι → R) :
    TrivSqZeroExt.snd
        (coordinates.prod fun coordinate : ι =>
          (TrivSqZeroExt.inl (M := R) (inputTensor coordinate) +
            TrivSqZeroExt.inr (R := R) (inputTangent coordinate) :
            DualNumber R)) =
      ∑ selectedCoordinate ∈ coordinates,
        inputTangent selectedCoordinate *
          (coordinates.erase selectedCoordinate).prod inputTensor := by
  induction coordinates using Finset.induction_on with
  | empty =>
      simp
  | @insert selectedCoordinate coordinates hNotMem inductionHypothesis =>
      simp only [Finset.prod_insert hNotMem, DualNumber.snd_mul,
        TrivSqZeroExt.fst_add, TrivSqZeroExt.snd_add,
        TrivSqZeroExt.fst_inl, TrivSqZeroExt.snd_inl,
        TrivSqZeroExt.fst_inr, TrivSqZeroExt.snd_inr,
        add_zero, zero_add, inductionHypothesis, Finset.sum_insert hNotMem]
      have hPrimalProduct :
          TrivSqZeroExt.fst
              (coordinates.prod fun coordinate : ι =>
                (TrivSqZeroExt.inl (M := R) (inputTensor coordinate) +
                  TrivSqZeroExt.inr (R := R) (inputTangent coordinate) :
                  DualNumber R)) =
            coordinates.prod inputTensor := by
        exact fst_finset_prod_dualNumber coordinates inputTensor inputTangent
      rw [hPrimalProduct, Finset.erase_insert hNotMem]
      have hRemainingCoordinates :
          inputTensor selectedCoordinate *
              ∑ otherCoordinate ∈ coordinates,
                inputTangent otherCoordinate *
                  (coordinates.erase otherCoordinate).prod inputTensor =
            ∑ otherCoordinate ∈ coordinates,
              inputTangent otherCoordinate *
                ((insert selectedCoordinate coordinates).erase
                  otherCoordinate).prod inputTensor := by
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro otherCoordinate hOther
        have hDistinct : selectedCoordinate ≠ otherCoordinate := by
          intro hEqual
          subst otherCoordinate
          exact hNotMem hOther
        rw [Finset.erase_insert_of_ne hDistinct,
          Finset.prod_insert (Finset.notMem_mono
            (Finset.erase_subset _ _) hNotMem)]
        ac_rfl
      rw [hRemainingCoordinates]
      ac_rfl

/--
Product reduction over dual numbers computes `productReduceDifferential` in
its tangent component.

This identifies the explicit Leibniz-rule formula with the first-order part
of the actual forward reducer, including fibers containing zero primal
values.
-/
theorem map_snd_reduce_prod_dualNumber
    {R : Type u} [Storage R] [CommSemiring R] {s t : Shape}
    (f : Coord s → Coord t)
    (inputTensor inputTangent : Rep R s) :
    map TrivSqZeroExt.snd
        (reduce Multiset.prod f <| Rep.ofFn fun inputCoordinate =>
          (TrivSqZeroExt.inl (M := R) (inputTensor inputCoordinate) +
            TrivSqZeroExt.inr (R := R) (inputTangent inputCoordinate) :
            DualNumber R)) =
      productReduceDifferential f inputTensor inputTangent := by
  ext outputCoordinate
  simp only [map_apply, reduce, productReduceDifferential, get_ofFn]
  rw [Fiber.values_prod]
  simp only [get_ofFn]
  exact snd_finset_prod_dualNumber
    Finset.univ
    (fun inputCoordinate : Fiber f outputCoordinate =>
      inputTensor inputCoordinate.1)
    (fun inputCoordinate : Fiber f outputCoordinate =>
      inputTangent inputCoordinate.1)

end Rep

namespace Semantics

open Check

/--
The tangent component of a checked product reduction over dual-number inputs
is the product-reduction differential along the checked coordinate fibers.

Dual-number evaluation ties the derivative formula to the actual
`Multiset.prod` denotation. In particular, the result does not rely on a
nonzero-input assumption or on rewriting the derivative as a quotient.
-/
theorem map_snd_denoteReduce_prod_dualNumber
    {R : Type u} [Storage R] [CommSemiring R]
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .reduce)
    (inputTensor inputTangent : checked.InputTensor R) :
    Rep.map TrivSqZeroExt.snd
        (denoteReduce Multiset.prod checked hKind <|
          Rep.ofFn fun inputCoordinate =>
          (TrivSqZeroExt.inl (M := R) (inputTensor inputCoordinate) +
            TrivSqZeroExt.inr (R := R) (inputTangent inputCoordinate) :
            DualNumber R)) =
      Rep.productReduceDifferential
        (checked.outputCoordinateOfInput <|
          checked.valid.normalization.output_axes_subset_input_of_reduce hKind)
        inputTensor inputTangent :=
  Rep.map_snd_reduce_prod_dualNumber
    (checked.outputCoordinateOfInput <|
      checked.valid.normalization.output_axes_subset_input_of_reduce hKind)
    inputTensor inputTangent

/--
The first-order part of checked product reduction is adjoint to its
zero-aware VJP under the finite tensor pairing.

The theorem applies to every checked reduction pattern and every commutative
semiring. It includes empty reduction fibers and primal tensors containing
arbitrarily many zeros.
-/
theorem dot_map_snd_denoteReduce_prod_dualNumber
    {R : Type u} [Storage R] [CommSemiring R]
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .reduce)
    (inputTensor inputTangent : checked.InputTensor R)
    (outputCotangent : checked.OutputTensor R) :
    Rep.dot
        (Rep.map TrivSqZeroExt.snd
          (denoteReduce Multiset.prod checked hKind <|
            Rep.ofFn fun inputCoordinate =>
            (TrivSqZeroExt.inl (M := R) (inputTensor inputCoordinate) +
              TrivSqZeroExt.inr (R := R) (inputTangent inputCoordinate) :
              DualNumber R)))
        outputCotangent =
      Rep.dot inputTangent
        (Rep.productReduceVjp
          (checked.outputCoordinateOfInput <|
            checked.valid.normalization.output_axes_subset_input_of_reduce hKind)
          inputTensor outputCotangent) := by
  rw [map_snd_denoteReduce_prod_dualNumber]
  exact Rep.dot_productReduceDifferential_eq_dot_productReduceVjp
    (checked.outputCoordinateOfInput <|
      checked.valid.normalization.output_axes_subset_input_of_reduce hKind)
    inputTensor inputTangent outputCotangent

end Semantics

end TorchLean.Tensor.Internal
