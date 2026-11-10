/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Algebra
public import TorchLean.Tensor.Layers.Conv

/-!
# Convolution Index Arithmetic

The forward and transpose convolution loops describe one integer relation from opposite directions:
on each axis, output index `o`, kernel index `k`, stride `s` and padding `p` meet input index `i`
when `o * s + k = i + p`. The relation is proved once for an arbitrary list of spatial axes, and the
only geometric hypothesis is the standard one that every stride is positive.
-/

@[expose] public section

namespace TorchLean

open TorchLean.Tensor
open scoped BigOperators

/-! ## Finite spatial indices

A spatial index is a coordinate of the shape `dims`, a `Shape.Coord dims`. A tensor is applied
to it as a function, and `Shape.Coord.toList` lists its indices for the runtime loops. -/

/-- On the index list of a coordinate `i`, the total lookup `Tensor.getAtOrZero` reads `x i`.

The lemma connects the two indexing styles of the file: implementations index with unbounded
`List Nat` lists and specifications with bounded coordinates, and the two agree inside the
bounds. -/
@[simp]
theorem Tensor.getAtOrZero_toList {α : Type} [Storage α] [Zero α] (dims : List Nat)
    (x : Tensor α dims) (i : Shape.Coord dims) :
    getAtOrZero x (Shape.Coord.toList dims i) = x i :=
  Tensor.getAtOrZero_coord_toList x i rfl

/--
A total lookup `Tensor.getAtOrZero x indices` is the sum over all coordinates `i` of `x i` when
`indices` is the index list of `i` and zero otherwise.

One formula covers valid indices, padding and malformed or out-of-range lists: an invalid list
matches no coordinate, and the sum is zero.
-/
theorem Tensor.getAtOrZero_eq_sum_indicator {α : Type} [Storage α] [AddCommMonoid α]
    (dims : List Nat) (x : Tensor α dims) (indices : List Nat) :
    getAtOrZero x indices =
      ∑ i : Shape.Coord dims, if indices = Shape.Coord.toList dims i then x i else 0 := by
  induction dims generalizing indices with
  | nil =>
      rw [← Tensor.scalar_item x]
      cases indices <;> simp [Shape.Coord.toList, Tensor.item]
  | cons n ns ih =>
      rw [← Tensor.dim_unstack x]
      rw [Fintype.sum_prod_type]
      cases indices with
      | nil =>
          simp [Shape.Coord.toList]
      | cons j js =>
          by_cases hj : j < n
          · simp only [getAtOrZero_dim_cons, hj, ↓reduceDIte, Tensor.unstack_dim]
            rw [ih (x.unstack ⟨j, hj⟩) js]
            symm
            rw [Finset.sum_eq_single ⟨j, hj⟩]
            · simp [Shape.Coord.toList]
            · intro head _ hne
              have hjne : j ≠ head.val := by
                intro hval
                apply hne
                apply Fin.ext
                simpa using hval.symm
              simp [Shape.Coord.toList, hjne]
            · simp
          · simp only [getAtOrZero_dim_cons, hj, ↓reduceDIte]
            symm
            apply Finset.sum_eq_zero
            intro head _
            have hjne : j ≠ head.val := by
              intro hval
              apply hj
              simp [hval, head.isLt]
            simp [Shape.Coord.toList, hjne]

/-- The executable nested index fold, accumulating `f` by addition, is `init` plus the finite sum
of `f` over the bounded coordinates. -/
theorem Conv.foldlIndices_add {α : Type} [AddCommMonoid α] (dims : List Nat)
    (init : α) (f : List Nat → α) :
    foldlIndices dims init (fun acc i => acc + f i) =
      init + ∑ i : Shape.Coord dims, f (Shape.Coord.toList dims i) := by
  induction dims generalizing init f with
  | nil =>
      change init + f [] = init + ∑ _i : PUnit, f []
      rw [Fintype.sum_unique]
  | cons n ns ih =>
      simp only [foldlIndices]
      simp_rw [ih]
      rw [List.finRange_foldl_add_eq_add_sum]
      simp [Shape.Coord.toList, Fintype.sum_prod_type]

/-- Every stride in a runtime list is positive. -/
def Conv.PositiveStrides (stride : List Nat) : Prop :=
  ∀ s ∈ stride, 0 < s

/-- For positive strides, solving a forward convolution index and then solving backwards recovers
the output index. -/
theorem Conv.mkTransposeInputIdx?_of_forward
    {outIdx kIdx stride padding inIdx : List Nat}
    (hStride : PositiveStrides stride)
    (hForward : mkInputIdx? outIdx kIdx stride padding = some inIdx) :
    mkTransposeInputIdx? inIdx kIdx stride padding = some outIdx := by
  induction outIdx generalizing kIdx stride padding inIdx with
  | nil =>
      cases kIdx <;> cases stride <;> cases padding <;> simp_all [mkInputIdx?, mkTransposeInputIdx?]
  | cons o os ih =>
      cases kIdx with
      | nil => simp [mkInputIdx?] at hForward
      | cons k ks =>
          cases stride with
          | nil => simp [mkInputIdx?] at hForward
          | cons s ss =>
              cases padding with
              | nil => simp [mkInputIdx?] at hForward
              | cons p ps =>
                  have hs : 0 < s := hStride s (by simp)
                  have hss : PositiveStrides ss := by
                    intro t ht
                    exact hStride t (by simp [ht])
                  simp only [mkInputIdx?] at hForward
                  split at hForward
                  · contradiction
                  · split at hForward
                    · contradiction
                    · rename_i rest hRest
                      cases hForward
                      have hTail := ih hss hRest
                      have hp : p ≤ o * s + k := Nat.le_of_not_gt ‹¬o * s + k < p›
                      have hRestore : o * s + k - p + p = o * s + k :=
                        Nat.sub_add_cancel hp
                      simp only [mkTransposeInputIdx?]
                      simp [Nat.ne_of_gt hs, hRestore, hTail]

/-- For positive strides, solving a transpose convolution index and then solving forwards
recovers the input index. -/
theorem Conv.mkInputIdx?_of_transpose
    {inIdx kIdx stride padding outIdx : List Nat}
    (hStride : PositiveStrides stride)
    (hTranspose : mkTransposeInputIdx? inIdx kIdx stride padding = some outIdx) :
    mkInputIdx? outIdx kIdx stride padding = some inIdx := by
  induction inIdx generalizing kIdx stride padding outIdx with
  | nil =>
      cases kIdx <;> cases stride <;> cases padding <;>
        simp_all [mkInputIdx?, mkTransposeInputIdx?]
  | cons i is ih =>
      cases kIdx with
      | nil => simp [mkTransposeInputIdx?] at hTranspose
      | cons k ks =>
          cases stride with
          | nil => simp [mkTransposeInputIdx?] at hTranspose
          | cons s ss =>
              cases padding with
              | nil => simp [mkTransposeInputIdx?] at hTranspose
              | cons p ps =>
                  have hs : 0 < s := hStride s (by simp)
                  have hss : PositiveStrides ss := by
                    intro t ht
                    exact hStride t (by simp [ht])
                  simp only [mkTransposeInputIdx?] at hTranspose
                  simp [Nat.ne_of_gt hs] at hTranspose
                  rcases hTranspose with ⟨hki, hMod, hTranspose⟩
                  cases hRest : mkTransposeInputIdx? is ks ss ps with
                  | none => simp [hRest] at hTranspose
                  | some rest =>
                      simp [hRest] at hTranspose
                      cases hTranspose
                      have hTail := ih hss hRest
                      have hDvd : s ∣ i + p - k := Nat.dvd_of_mod_eq_zero hMod
                      have hDivMul : (i + p - k) / s * s = i + p - k :=
                        Nat.div_mul_cancel hDvd
                      have hIndex : (i + p - k) / s * s + k = i + p := by
                        rw [hDivMul]
                        grind
                      simp only [mkInputIdx?]
                      simp [hIndex, hTail]

/-- The two index solvers define the same relation when strides are positive. -/
theorem Conv.mkInputIdx?_eq_some_iff
    {outIdx kIdx stride padding inIdx : List Nat}
    (hStride : PositiveStrides stride) :
    mkInputIdx? outIdx kIdx stride padding = some inIdx ↔
      mkTransposeInputIdx? inIdx kIdx stride padding = some outIdx :=
  ⟨mkTransposeInputIdx?_of_forward hStride,
    mkInputIdx?_of_transpose hStride⟩

end TorchLean
