/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Algebra
public import NN.Spec.Layers.Conv
public import Mathlib.Data.Fintype.BigOperators

/-!
# Convolution Index Arithmetic

The forward and transpose convolution loops describe the same integer relation from opposite
directions.  This file proves that relation once for an arbitrary list of spatial axes.  The only
geometric hypothesis is the standard one that every stride is positive.
-/

@[expose] public section

namespace Spec.Conv.Internal

open TorchLean TorchLean.Tensor
open scoped BigOperators

/-! ## Finite spatial indices -/

/-- A shape-indexed spatial coordinate, represented without partial list indexing. -/
abbrev MultiIndex : List Nat → Type
  | [] => PUnit
  | n :: ns => Fin n × MultiIndex ns

instance (dims : List Nat) : Fintype (MultiIndex dims) := by
  induction dims with
  | nil => simp only [MultiIndex]; infer_instance
  | cons n ns _ => simp only [MultiIndex]; infer_instance

instance (dims : List Nat) : DecidableEq (MultiIndex dims) := by
  induction dims with
  | nil => simp only [MultiIndex]; infer_instance
  | cons n ns _ => simp only [MultiIndex]; infer_instance

/-- Convert a bounded spatial coordinate to the runtime list representation. -/
def MultiIndex.toList : {dims : List Nat} → MultiIndex dims → List Nat
  | [], _ => []
  | _ :: _, index => index.1.val :: index.2.toList

/-- Read a tensor at a bounded spatial coordinate. -/
def MultiIndex.get {α : Type} [TorchLean.Storage α] : {dims : List Nat} →
    Tensor α (Shape.ofList dims) → MultiIndex dims → α
  | [], tensor, _ => tensor.item
  | _ :: _, tensor, index => index.2.get (tensor.unstack index.1)

/-- Reading a `dim` tensor peels the leading coordinate and recurses into that slice. -/
@[simp]
theorem MultiIndex.get_dim {α : Type} [TorchLean.Storage α] (n : Nat) (dims : List Nat)
    (values : Fin n → Tensor α (Shape.ofList dims))
    (head : Fin n) (tail : MultiIndex dims) :
    MultiIndex.get (dims := n :: dims) (Tensor.dim values) (head, tail) =
      MultiIndex.get (values head) tail := by
  simp [MultiIndex.get]

/-- Reading the sole scalar coordinate below a vector index agrees with the vector view. -/
@[simp]
theorem MultiIndex.get_vector_eq_getScalar {α : Type} [TorchLean.Storage α] {n : Nat}
    (values : Tensor α [n]) (i : Fin n) :
    MultiIndex.get (dims := [n]) values (i, PUnit.unit) = TorchLean.Tensor.getScalar values i := by
  rfl

/-- An index that runs out too early reads zero. -/
@[simp]
theorem getAtOrZero_dim_nil {α : Type} [TorchLean.Storage α] [Zero α]
    (n : Nat) (dims : List Nat)
    (values : Fin n → Tensor α (Shape.ofList dims)) :
    getAtOrZero (Tensor.dim values) [] = 0 := by
  simp

/-- Otherwise the leading coordinate is bounds-checked and the lookup recurses, reading zero when it
falls outside. This total lookup is what lets padding be expressed without a separate case split at
every use site: an out-of-range index simply contributes nothing. -/
theorem getAtOrZero_dim_cons {α : Type} [TorchLean.Storage α] [Zero α]
    (n : Nat) (dims : List Nat)
    (values : Fin n → Tensor α (Shape.ofList dims)) (j : Nat) (js : List Nat) :
    getAtOrZero (Tensor.dim values) (j :: js) =
      if h : j < n then getAtOrZero (values ⟨j, h⟩) js else 0 := by
  simp

/-- Split a finite sum over a nonempty multi-index into its leading coordinate and tail. -/
theorem MultiIndex.sum_cons {α : Type} [AddCommMonoid α] (n : Nat) (dims : List Nat)
    (f : MultiIndex (n :: dims) → α) :
    (∑ i : MultiIndex (n :: dims), f i) =
      ∑ head : Fin n, ∑ tail : MultiIndex dims, f (head, tail) := by
  simpa only [MultiIndex] using
    (Fintype.sum_prod_type (f := f))

/-- Reading a generated tensor applies the generating function to the index's coordinate list. -/
@[simp]
theorem MultiIndex.get_generate {α : Type} [TorchLean.Storage α] (dims : List Nat)
    (f : List Nat → α) (i : MultiIndex dims) :
    i.get (TorchLean.Tensor.generate dims f) = f i.toList := by
  induction dims generalizing f with
  | nil =>
      cases i
      simp [MultiIndex.get, MultiIndex.toList, TorchLean.Tensor.generate, Tensor.item]
      change f [] = f []
      rfl
  | cons n ns inductionHypothesis =>
      rcases i with ⟨head, tail⟩
      have hSlice :
          (TorchLean.Tensor.generate (n :: ns) f).unstack head =
            TorchLean.Tensor.generate ns (fun indices => f (head.val :: indices)) := by
        apply TorchLean.Tensor.Internal.Rep.ext
        intro coordinate
        simp [TorchLean.Tensor.generate, Tensor.unstack, Shape.Coord.toList]
      simp only [MultiIndex.get, MultiIndex.toList, hSlice]
      exact inductionHypothesis (fun indices => f (head.val :: indices)) tail

/-- Bounded-coordinate lookup commutes with pointwise tensor addition. -/
theorem MultiIndex.get_addSpec {α : Type} [TorchLean.Storage α] [Context α] {dims : List Nat}
    (x y : Tensor α (Shape.ofList dims)) (i : MultiIndex dims) :
    i.get (TorchLean.Tensor.addSpec x y) = i.get x + i.get y := by
  induction dims with
  | nil =>
      simp [MultiIndex.get, TorchLean.Tensor.addSpec, TorchLean.Tensor.map2Spec, Tensor.item]
  | cons n dims ih =>
      rcases i with ⟨head, tail⟩
      change tail.get ((TorchLean.Tensor.addSpec x y).unstack head) =
        tail.get (x.unstack head) + tail.get (y.unstack head)
      rw [show (TorchLean.Tensor.addSpec x y).unstack head =
          TorchLean.Tensor.addSpec (x.unstack head) (y.unstack head) by
        exact
          (TorchLean.Tensor.Internal.Rep.zipWith_unstack (· + ·) x y head).symm]
      exact ih (x.unstack head) (y.unstack head) tail

/-- On an in-range index the total lookup agrees with the bounded one.

This is the lemma that connects the two indexing styles in the file: implementations use unbounded
`List Nat` coordinates, specifications use `MultiIndex`, and inside the bounds they coincide. -/
@[simp]
theorem getAtOrZero_toList {α : Type} [TorchLean.Storage α] [Zero α] (dims : List Nat)
    (x : Tensor α (Shape.ofList dims)) (i : MultiIndex dims) :
    getAtOrZero x i.toList = i.get x := by
  induction dims with
  | nil =>
      cases i
      rw [← Tensor.scalar_item x]
      simp [MultiIndex.toList, MultiIndex.get]
  | cons n ns ih =>
      rcases i with ⟨head, tail⟩
      rw [← Tensor.dim_unstack x]
      simp [MultiIndex.toList, MultiIndex.get, head.isLt, ih]

/--
A total lookup is the finite coordinate sum selected by equality of index lists.

This formulation handles valid indices, padding, and malformed or out-of-range lists uniformly.
-/
theorem getAtOrZero_eq_sum_indicator {α : Type} [TorchLean.Storage α] [AddCommMonoid α]
    (dims : List Nat) (x : Tensor α (Shape.ofList dims)) (indices : List Nat) :
    getAtOrZero x indices =
      ∑ i : MultiIndex dims, if indices = i.toList then i.get x else 0 := by
  induction dims generalizing indices with
  | nil =>
      rw [← Tensor.scalar_item x]
      cases indices <;> simp [MultiIndex.toList, MultiIndex.get, one_nsmul]
  | cons n ns ih =>
      rw [← Tensor.dim_unstack x]
      rw [Fintype.sum_prod_type]
      cases indices with
      | nil =>
          simp [MultiIndex.toList]
      | cons j js =>
          by_cases hj : j < n
          · simp only [get_at_or_zero_dim_cons, hj, ↓reduceDIte, Tensor.unstack_dim]
            rw [ih (x.unstack ⟨j, hj⟩) js]
            symm
            rw [Finset.sum_eq_single ⟨j, hj⟩]
            · simp [MultiIndex.toList, MultiIndex.get]
            · intro head _ hne
              have hjne : j ≠ head.val := by
                intro hval
                apply hne
                apply Fin.ext
                simpa using hval.symm
              simp [MultiIndex.toList, hjne]
            · simp
          · simp only [get_at_or_zero_dim_cons, hj, ↓reduceDIte]
            symm
            apply Finset.sum_eq_zero
            intro head _
            have hjne : j ≠ head.val := by
              intro hval
              apply hj
              simp [hval, head.isLt]
            simp [MultiIndex.toList, hjne]

/-- The executable nested index fold is the finite sum over bounded coordinates. -/
theorem foldlIndices_add {α : Type} [AddCommMonoid α] (dims : List Nat)
    (init : α) (f : List Nat → α) :
    foldlIndices dims init (fun acc i => acc + f i) =
      init + ∑ i : MultiIndex dims, f i.toList := by
  induction dims generalizing init f with
  | nil =>
      change init + f [] = init + ∑ _i : PUnit, f []
      rw [Fintype.sum_unique]
  | cons n ns ih =>
      simp only [foldlIndices]
      simp_rw [ih]
      rw [List.finRange_foldl_add_acc]
      simp [MultiIndex, MultiIndex.toList, Fintype.sum_prod_type]

/-- Recursive tensor dot product as a finite sum over bounded multi-indices. -/
theorem dot_eq_sum_get {α : Type} [TorchLean.Storage α] [CommSemiring α] (dims : List Nat)
    (x y : Tensor α (Shape.ofList dims)) :
    Proofs.TensorAlgebra.dot x y = ∑ i : MultiIndex dims, i.get x * i.get y := by
  induction dims with
  | nil =>
      change x.item * y.item = ∑ _i : PUnit, x.item * y.item
      rw [Fintype.sum_unique]
  | cons n ns ih =>
      change
        (List.finRange n).foldl
            (fun acc i => acc + Proofs.TensorAlgebra.dot (x.unstack i) (y.unstack i)) 0 =
          ∑ p : Fin n × MultiIndex ns,
            p.2.get (x.unstack p.1) * p.2.get (y.unstack p.1)
      rw [List.finRange_foldl_add_eq_finset_sum]
      rw [Fintype.sum_prod_type]
      apply Finset.sum_congr rfl
      intro i _
      exact ih (x.unstack i) (y.unstack i)

/-- Every stride in a runtime list is positive. -/
def PositiveStrides (stride : List Nat) : Prop :=
  ∀ s ∈ stride, 0 < s

/-- Solving a forward convolution index and then solving backwards recovers the output index. -/
theorem mkTransposeInputIdx?_of_mkInputIdx?_eq_some
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

/-- Solving a transpose convolution index and then solving forwards recovers the input index. -/
theorem mkInputIdx?_of_mkTransposeInputIdx?_eq_some
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
theorem mkInputIdx?_eq_some_iff
    {outIdx kIdx stride padding inIdx : List Nat}
    (hStride : PositiveStrides stride) :
    mkInputIdx? outIdx kIdx stride padding = some inIdx ↔
      mkTransposeInputIdx? inIdx kIdx stride padding = some outIdx :=
  ⟨mkTransposeInputIdx?_of_mkInputIdx?_eq_some hStride,
    mkInputIdx?_of_mkTransposeInputIdx?_eq_some hStride⟩

end Spec.Conv.Internal
