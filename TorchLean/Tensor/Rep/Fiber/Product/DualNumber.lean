/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import Mathlib.Algebra.DualNumber
public import TorchLean.Tensor.Einops.Semantics.Transform.Reduction

/-!
# Dual numbers and product reduction

Evaluating product reduction on dual numbers puts the leave-one-out Leibniz formula in the
tangent component, and for a checked reduction that formula is adjoint to the zero-aware VJP
under the tensor pairing.

The proofs need mathlib's `DualNumber` algebra, which the runtime formulas never use. Importing it
in this proof module rather than with those formulas keeps that dependency out of ordinary tensor
programs.
-/

@[expose] public section

namespace TorchLean

open scoped BigOperators

universe u v

namespace Tensor.Rep

open Einops (Fiber)
open Shape (Coord)

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
  change (TrivSqZeroExt.fstHom R R R) _ = _
  rw [map_prod]
  simp only [TrivSqZeroExt.fstHom_apply, TrivSqZeroExt.fst_add,
    TrivSqZeroExt.fst_inl, TrivSqZeroExt.fst_inr, add_zero]

/--
The tangent component of a finite dual-number product is the sum, over coordinates, of each
tangent times the product of the other primals.
-/
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
Product reduction over dual numbers computes `Product.differential` in
its tangent component.

The Leibniz-rule formula is therefore the first-order part of the forward reducer itself,
including on fibers that contain zero primal values.
-/
theorem map_snd_reduce_prod
    {R : Type u} [Storage R] [CommSemiring R] {s t : Shape}
    (f : Coord s → Coord t)
    (inputTensor inputTangent : Rep R s) :
    map TrivSqZeroExt.snd
        (reduce Multiset.prod f <| Rep.ofFn fun inputCoordinate =>
          (TrivSqZeroExt.inl (M := R) (inputTensor inputCoordinate) +
            TrivSqZeroExt.inr (R := R) (inputTangent inputCoordinate) :
            DualNumber R)) =
      Product.differential f inputTensor inputTangent := by
  ext outputCoordinate
  simp only [map_apply, reduce, Product.differential, get_ofFn]
  rw [Fiber.values_prod]
  simp only [get_ofFn]
  exact snd_finset_prod_dualNumber
    Finset.univ
    (fun inputCoordinate : Fiber f outputCoordinate =>
      inputTensor inputCoordinate.1)
    (fun inputCoordinate : Fiber f outputCoordinate =>
      inputTangent inputCoordinate.1)

end Tensor.Rep

namespace Einops.Semantics

open Tensor (Rep)

/--
The tangent component of a checked product reduction over dual-number inputs
is the product-reduction differential along the checked coordinate fibers.

Dual-number evaluation ties the derivative formula to the `Multiset.prod` denotation, so the
result needs no nonzero-input assumption and never writes the derivative as a quotient.
-/
theorem map_snd_denoteReduce_prod
    {R : Type u} [Storage R] [CommSemiring R]
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (inputTensor inputTangent : checked.InputTensor R) :
    Tensor.map TrivSqZeroExt.snd
        (denoteReduce Multiset.prod checked hKind <|
          Rep.ofFn fun inputCoordinate =>
          (TrivSqZeroExt.inl (M := R) (inputTensor inputCoordinate) +
            TrivSqZeroExt.inr (R := R) (inputTangent inputCoordinate) :
            DualNumber R)) =
      Rep.Product.differential
        (checked.outputOfInput <|
          checked.valid.normalization.output_subset_of_reduce hKind)
        inputTensor inputTangent :=
  Rep.map_snd_reduce_prod
    (checked.outputOfInput <|
      checked.valid.normalization.output_subset_of_reduce hKind)
    inputTensor inputTangent

/--
The first-order part of checked product reduction is adjoint to its
zero-aware VJP under the finite tensor pairing.

It holds for every checked reduction pattern and every commutative semiring, including empty
reduction fibers and primal tensors with any number of zeros.
-/
@[einops_simps] theorem dot_map_snd_denoteReduce_prod
    {R : Type u} [Storage R] [CommSemiring R]
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (inputTensor inputTangent : checked.InputTensor R)
    (outputCotangent : checked.OutputTensor R) :
    Rep.dot
        (Tensor.map TrivSqZeroExt.snd
          (denoteReduce Multiset.prod checked hKind <|
            Rep.ofFn fun inputCoordinate =>
            (TrivSqZeroExt.inl (M := R) (inputTensor inputCoordinate) +
              TrivSqZeroExt.inr (R := R) (inputTangent inputCoordinate) :
              DualNumber R)))
        outputCotangent =
      Rep.dot inputTangent
        (Rep.Product.vjp
          (checked.outputOfInput <|
            checked.valid.normalization.output_subset_of_reduce hKind)
          inputTensor outputCotangent) := by
  rw [map_snd_denoteReduce_prod]
  exact Rep.Product.dot_differential_eq_dot_vjp
    (checked.outputOfInput <|
      checked.valid.normalization.output_subset_of_reduce hKind)
    inputTensor inputTangent outputCotangent

end Einops.Semantics

end TorchLean
