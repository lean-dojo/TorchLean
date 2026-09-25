/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import all NN.MLTheory.CROWN.Graph.Engine.Base
import all NN.MLTheory.CROWN.Proofs.DirectedIBPCoordinates
public import NN.MLTheory.CROWN.Proofs.DirectedIBPMaxPool

/-!
# Directed enclosure for average pooling

The executable transfer enumerates each padded window in flat order, encloses its real sum and
count, and divides the resulting intervals. The spatial proof lifts over every leading axis,
including empty leading dimensions.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean TorchLean.Tensor NN.IR
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph.LayerNormDirected
open Spec.Pooling.Internal
open scoped BigOperators

noncomputable section

private theorem avgPoolValue_eq_flat_mean {d : Nat} {spatial : Tensor Nat [d]}
    (x : Tensor ℝ (Shape.ofList spatial.data.toList))
    (out kernel stride padding : List Nat) :
    avgPoolValue x out kernel stride padding =
      (∑ i : Fin (Shape.ofList kernel).size,
        getPaddedAverageInputVal x out
          (flatCoordinates kernel.toArray i.val).toList stride padding) /
        (Shape.ofList kernel).size := by
  unfold avgPoolValue
  rw [foldlIndices_eq_flat_sum (Shape.ofList kernel)]
  simp only [kernelProd, ← List.prod_eq_foldl, Shape.size_eq_prod, Shape.toArray,
    Shape.toList, Shape.ofList]

private theorem avgPoolSpatial_getScalar {d : Nat}
    {kernel stride padding spatial : Tensor Nat [d]}
    {hk : ∀ i : Fin d, kernel.getScalar i ≠ 0}
    {hs : ∀ i : Fin d, stride.getScalar i ≠ 0}
    (layer : AvgPoolSpec d kernel stride padding hk hs)
    (x : Tensor ℝ (Shape.ofList spatial.data.toList))
    (i : Fin (Shape.ofList (poolOutSpatialPad spatial kernel stride padding).data.toList).size) :
    (Tensor.flattenSpec (Spec.avgPoolSpatialSpec layer x)).getScalar i =
      (∑ j : Fin (Shape.ofList kernel.data.toList).size,
        getPaddedAverageInputVal x
          (flatCoordinates (poolOutSpatialPad spatial kernel stride padding).data.toList.toArray
            i.val).toList
          (flatCoordinates kernel.data.toList.toArray j.val).toList
          stride.data.toList padding.data.toList) /
        (Shape.ofList kernel.data.toList).size := by
  rw [getScalar_flatten_coord]
  simp only [Spec.avgPoolSpatialSpec, Tensor.generate, Tensor.Internal.Rep.get_ofFn]
  rw [← flatCoordinates_eq_coord]
  exact avgPoolValue_eq_flat_mean _ _ _ _ _

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]
  [NonlinearBoundOps α] [LawfulNonlinearBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

omit [LawfulBoundOps α] [LawfulNonlinearBoundOps α] in
private theorem directedRowMean?_positive {n : Nat} {bounds : Fin n → α × α}
    {out : α × α} (hout : directedRowMean? bounds = some out) : 0 < n := by
  by_contra hn
  have hz : n = 0 := Nat.eq_zero_of_not_pos hn
  simp only [directedRowMean?, hz, ite_true, reduceCtorEq] at hout

private theorem directedAvgPoolTensor?_encloses
    (config : WindowConfig) (spatial : Tensor Nat [config.spatialRank]) (leading : Shape)
    {hk : ∀ i : Fin config.spatialRank, config.kernel.getScalar i ≠ 0}
    {hs : ∀ i : Fin config.spatialRank, config.stride.getScalar i ≠ 0}
    (layer : AvgPoolSpec config.spatialRank config.kernel config.stride config.padding hk hs)
    {lo hi : Tensor α (leading.concat (Shape.ofList spatial.data.toList))}
    {x : Tensor ℝ (leading.concat (Shape.ofList spatial.data.toList))}
    (h : TensorEncloses lo hi x)
    {bounds : Box α (leading.concat (Shape.ofList
      (poolOutSpatialPad spatial config.kernel config.stride config.padding).data.toList))}
    (hb : directedAvgPoolTensor? config spatial leading lo hi = some bounds) :
    TensorEncloses bounds.lo bounds.hi
      (Tensor.mapLeading leading (Spec.avgPoolSpatialSpec layer) x) := by
  induction leading with
  | scalar =>
      simp only [directedAvgPoolTensor?] at hb
      obtain ⟨entries, hentries, hb⟩ := Option.bind_eq_some_iff.mp hb
      obtain rfl := Option.some.inj hb
      have hflat : TensorEncloses
          (Tensor.ofFn fun i => (entries i).1) (Tensor.ofFn fun i => (entries i).2)
          (Tensor.flattenSpec (Spec.avgPoolSpatialSpec layer x)) := by
        rw [tensorEncloses_vector_iff]
        intro i
        have hi := Internal.traverseFin_eq_some_iff.mp hentries i
        have hmean := directedRowMean?_encloses (directedRowMean?_positive hi) _
          (fun j => getPaddedAverageInputVal x
            (flatCoordinates
              (poolOutSpatialPad spatial config.kernel config.stride config.padding).data.toList.toArray
              i.val).toList
            (flatCoordinates config.kernel.data.toList.toArray j.val).toList
            config.stride.data.toList config.padding.data.toList)
          (fun j => h.paddedAverage _ _ _ _) hi
        simpa only [getScalar_ofFn, avgPoolSpatial_getScalar] using hmean
      simpa only [Tensor.mapLeading, Tensor.unflattenSpec_flattenSpec] using hflat.unflatten
  | dim n leading ih =>
      simp only [directedAvgPoolTensor?] at hb
      obtain ⟨entries, hentries, hb⟩ := Option.bind_eq_some_iff.mp hb
      obtain rfl := Option.some.inj hb
      apply TensorEncloses.dim
      intro i
      exact ih (h.unstack i) (Internal.traverseFin_eq_some_iff.mp hentries i)

/--
Successful directed average pooling encloses the actual real evaluator, for every accepted
window configuration and every collection of leading dimensions.
-/
theorem ibpAvgPool?_encloses
    {config : WindowConfig} {s t : Shape} {B box : FlatBox α}
    {f : Nat → ℝ} {y : Tensor ℝ t}
    (hB : RowEncloses B s.size f)
    (hy : NN.IR.Graph.evalAvgPool config ⟨s, realTensor s f⟩ = .ok ⟨t, y⟩)
    (hbox : ibpAvgPool? config s t B = some box) :
    RowEncloses box t.size (tensorValues y) := by
  unfold ibpAvgPool? at hbox
  unfold NN.IR.Graph.evalAvgPool at hy
  cases hp : OpContracts.planPool "avg_pool" config s with
  | error e =>
      simp only [hp, Except.toOption, Option.bind_eq_bind, Option.bind_none,
        reduceCtorEq] at hbox
  | ok plan =>
      simp only [hp, Except.toOption, Option.bind_eq_bind, Option.bind_some] at hbox
      split at hbox
      · contradiction
      · split at hbox
        next hd =>
          obtain ⟨bounds, hbounds, hbox⟩ := Option.bind_eq_some_iff.mp hbox
          obtain rfl := Option.some.inj hbox
          have hin := (tensorEncloses_ibpUnflatten hd hB).cast plan.concat_eq.symm
          have hout := directedAvgPoolTensor?_encloses config plan.spatial plan.leading
            ({} : AvgPoolSpec config.spatialRank config.kernel config.stride config.padding
              plan.kernelNonzero plan.strideNonzero) hin hbounds
          simp only [hp, Bind.bind, Except.bind, Pure.pure, Except.pure,
            Except.ok.injEq] at hy
          have hsome : SomeTensorEncloses
              ⟨plan.outShape, bounds.lo⟩ ⟨plan.outShape, bounds.hi⟩ ⟨t, y⟩ := by
            rw [← hy]
            exact .mk hout
          cases hsome with
          | mk h => exact h.tensorValues
        next => contradiction

end

end NN.MLTheory.CROWN.Graph.DirectedBackward
