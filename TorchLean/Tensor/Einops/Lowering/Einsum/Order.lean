/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Einops.Lowering.Einsum
public import TorchLean.Tensor.Reduce.Order

/-!
# Einsum in a reduction order

`einsumIn order` is `einsum` with the contracted products added in the order `order`, through
`contractionIn order`: with no contracted axis an output entry is its product, as in `einsum`;
otherwise it is `ReduceOrder.fold` from zero over the contracted coordinates in row-major flat
order. The products keep the operand order of `einsumInputProduct`.

- `contractionIn_leftFold`, `einsumIn_leftFold`: in `leftFold` they are `Semantics.contraction`
  and `einsum`, by `coordFoldl_eq_finRange`.
- `contractionIn_eq_contraction`, `einsumIn_eq_einsum`: with an associative and commutative
  addition every order that covers the contracted coordinates gives them, so over `ℝ` every such
  order computes `einsum` (`einsumIn_real`).
-/

@[expose] public section

namespace TorchLean.Einops.Lowering

open Shape (Coord)
open Tensor (ReduceOrder Rep)

/-- The row-major coordinate fold is the fold over the flat indices in order. -/
theorem coordFoldl_eq_finRange {β : Type} (shape : Shape) (step : β → Coord shape → β)
    (init : β) :
    Coord.foldl shape step init =
      (List.finRange (Shape.size shape)).foldl (fun acc i => step acc (Coord.unlinearize i))
        init := by
  rw [← Rep.foldl_coordinates_eq_foldl, Rep.foldl_eq_fin_foldl, Fin.foldl_eq_foldl_finRange]
  simp only [Rep.coordinates, Rep.ofFn, Rep.getFlat_ofFlatFn, id_eq]

/-- `Semantics.contraction` in the order `order`: with no axis the single value, otherwise the
order's fold from zero over the flat coordinates. -/
def contractionIn {R : Type} [Add R] [OfNat R 0] (order : ReduceOrder) :
    (shape : Shape) → (Coord shape → R) → R
  | [], values => values PUnit.unit
  | length :: shape, values =>
    order.fold (Shape.size (length :: shape)) (· + ·) 0 fun i => values (Coord.unlinearize i)

theorem contractionIn_leftFold {R : Type} [Add R] [OfNat R 0] :
    (shape : Shape) → (values : Coord shape → R) →
      contractionIn .leftFold shape values = Semantics.contraction shape values
  | [], _ => rfl
  | length :: shape, values => by
    simp only [contractionIn, Semantics.contraction, ReduceOrder.fold_leftFold,
      coordFoldl_eq_finRange]

theorem contractionIn_eq_contraction {R : Type} [AddCommSemigroup R] [OfNat R 0]
    {order : ReduceOrder} :
    (shape : Shape) → order.Covers (Shape.size shape) → (values : Coord shape → R) →
      contractionIn order shape values = Semantics.contraction shape values
  | [], _, _ => rfl
  | length :: shape, hcov, values => by
    rw [← contractionIn_leftFold]
    exact ReduceOrder.fold_eq_leftFold (β := R) hcov add_assoc add_comm _ _

/-- `einsum` with the contracted products added in the order `order`. -/
def einsumIn {R : Type} [Storage R] [Add R] [Mul R] [OfNat R 0] [OfNat R 1]
    (order : ReduceOrder) (checked : Einsum) (inputTensors : checked.InputTensors R) :
    checked.OutputTensor R :=
  Rep.ofFlatFn fun o =>
    contractionIn order _ (einsumInputProduct checked inputTensors (Coord.unlinearize o))

theorem einsumIn_leftFold {R : Type} [Storage R] [Add R] [Mul R] [OfNat R 0] [OfNat R 1]
    (checked : Einsum) (inputTensors : checked.InputTensors R) :
    einsumIn .leftFold checked inputTensors = einsum checked inputTensors := by
  simp only [einsumIn, einsum]
  congr 1
  funext o
  exact contractionIn_leftFold _ _

theorem einsumIn_eq_einsum {R : Type} [Storage R] [AddCommSemigroup R] [Mul R] [OfNat R 0]
    [OfNat R 1] {order : ReduceOrder} (checked : Einsum) (inputTensors : checked.InputTensors R)
    (hcov : order.Covers (Shape.size (checked.contractedAxes.map checked.axisLength))) :
    einsumIn order checked inputTensors = einsum checked inputTensors := by
  simp only [einsumIn, einsum]
  congr 1
  funext o
  exact contractionIn_eq_contraction _ hcov _

/-- Over `ℝ` every order that covers the contracted coordinates computes `einsum`. -/
theorem einsumIn_real {order : ReduceOrder} (checked : Einsum)
    (inputTensors : checked.InputTensors ℝ)
    (hcov : order.Covers (Shape.size (checked.contractedAxes.map checked.axisLength))) :
    einsumIn order checked inputTensors = einsum checked inputTensors :=
  einsumIn_eq_einsum checked inputTensors hcov

end TorchLean.Einops.Lowering
