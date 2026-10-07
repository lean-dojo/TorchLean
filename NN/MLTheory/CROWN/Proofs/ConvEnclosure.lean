/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.Conv
public import NN.MLTheory.CROWN.BoundOps.Lawful

/-!
# Enclosure by the directed convolution folds

The executable convolution transfer bounds each signed product and accumulates those bounds in
the same channel/kernel order as `groupedConvSpec`. These theorems establish its enclosure over
the exact real `BoundOps` instance, including padding, grouped channels, dilation, and arbitrary
leading batch dimensions. The affine corollary uses the existing convolution matrix equality;
the transfer itself continues to use its spatial folds.
-/

public section

namespace NN.MLTheory.CROWN.ConvProof

open Spec TorchLean TorchLean.Tensor Spec.Conv.Internal
open scoped BigOperators

theorem contains_iff_apply {s : Shape} (box : Box ℝ s) (input : Tensor ℝ s) :
    Box.contains box input ↔
      ∀ coordinate : s.Coord,
        box.lo coordinate ≤ input coordinate ∧ input coordinate ≤ box.hi coordinate := by
  induction s with
  | scalar =>
      constructor
      · intro h coordinate
        cases coordinate
        exact h
      · intro h
        exact h PUnit.unit
  | dim n rest ih =>
      constructor
      · intro h coordinate
        rcases coordinate with ⟨head, tail⟩
        simpa only [Tensor.unstack, Internal.Rep.unstack_apply] using
          (ih ⟨box.lo.unstack head, box.hi.unstack head⟩
            (input.unstack head)).mp (h head) tail
      · intro h head
        apply (ih ⟨box.lo.unstack head, box.hi.unstack head⟩
          (input.unstack head)).mpr
        intro tail
        simpa only [Tensor.unstack, Internal.Rep.unstack_apply] using h (head, tail)

/-- Flattening preserves and reflects every component inequality of a real tensor box. -/
theorem box_contains_flatten_iff {s : Shape} (box : Box ℝ s) (input : Tensor ℝ s) :
    Box.contains (flattenBox box) (flattenSpec input) ↔ Box.contains box input := by
  rw [contains_iff_apply, contains_iff_apply]
  constructor
  · intro h coordinate
    have hc := h ((Internal.Rep.reshapeCoordEquiv (size_toList_flatten s)).symm coordinate)
    simpa only [flattenBox, flattenSpec, Internal.Rep.reshape_apply_coordEquiv,
      Equiv.apply_symm_apply] using hc
  · intro h coordinate
    simpa only [flattenBox, flattenSpec, Internal.Rep.reshape_apply_coordEquiv] using
      h (Internal.Rep.reshapeCoordEquiv (size_toList_flatten s) coordinate)

theorem contains_iff_multiIndex (dims : List Nat)
    (box : Box ℝ (Shape.ofList dims)) (input : Tensor ℝ (Shape.ofList dims)) :
    Box.contains box input ↔
      ∀ index : MultiIndex dims,
        index.get box.lo ≤ index.get input ∧ index.get input ≤ index.get box.hi := by
  induction dims with
  | nil =>
      constructor
      · intro h index
        exact h
      · intro h
        exact h PUnit.unit
  | cons n dims ih =>
      constructor
      · intro h index
        exact (ih ⟨box.lo.unstack index.1, box.hi.unstack index.1⟩
          (input.unstack index.1)).mp (h index.1) index.2
      · intro h head
        apply (ih ⟨box.lo.unstack head, box.hi.unstack head⟩
          (input.unstack head)).mpr
        intro tail
        exact h (head, tail)

theorem contains_getAtOrZero {dims : List Nat}
    {box : Box ℝ (Shape.ofList dims)} {input : Tensor ℝ (Shape.ofList dims)}
    (h : Box.contains box input) (indices : List Nat) :
    getAtOrZero box.lo indices ≤ getAtOrZero input indices ∧
      getAtOrZero input indices ≤ getAtOrZero box.hi indices := by
  induction dims generalizing indices with
  | nil =>
      cases indices with
      | nil => simpa only [get_at_or_zero_scalar_nil, Box.contains] using h
      | cons head tail => simp
  | cons n dims ih =>
      cases indices with
      | nil => simp
      | cons head tail =>
          by_cases hhead : head < n
          · simpa only [get_at_or_zero_dim_cons, dite_eq_left hhead] using
              ih (h ⟨head, hhead⟩) tail
          · simp only [get_at_or_zero_dim_cons, dite_eq_right hhead, le_refl, and_self]

/-- The spatial iterator preserves a binary relation independently of rank and empty extents. -/
theorem foldlIndices_rel {A B : Type} (relation : A → B → Prop) (dims : List Nat)
    {left : A → List Nat → A} {right : B → List Nat → B}
    (step : ∀ a b i, relation a b → relation (left a i) (right b i))
    {a : A} {b : B} (initial : relation a b) :
    relation (foldlIndices dims a left) (foldlIndices dims b right) := by
  induction dims generalizing left right a b with
  | nil => exact step a b [] initial
  | cons n dims ih =>
      apply List.foldl_rel initial
      intro head _ a b h
      exact ih (fun a b tail hab => step a b (head.val :: tail) hab) h

private theorem mul_endpoints_enclose {lo hi value weight : ℝ}
    (lower : lo ≤ value) (upper : value ≤ hi) :
    (if weight * lo > weight * hi then weight * hi else weight * lo) ≤ value * weight ∧
      value * weight ≤
        (if weight * lo > weight * hi then weight * lo else weight * hi) := by
  by_cases hw : 0 ≤ weight
  · have hl := mul_le_mul_of_nonneg_left lower hw
    have hu := mul_le_mul_of_nonneg_left upper hw
    have horder : ¬ weight * lo > weight * hi := not_lt_of_ge (hl.trans hu)
    simpa [horder, mul_comm value weight] using And.intro hl hu
  · have hl := mul_le_mul_of_nonpos_left upper (le_of_not_ge hw)
    have hu := mul_le_mul_of_nonpos_left lower (le_of_not_ge hw)
    by_cases horder : weight * lo > weight * hi
    · simpa [horder, mul_comm value weight] using And.intro hl hu
    · have heq : weight * lo = weight * hi :=
        le_antisymm (le_of_not_gt horder) (hl.trans hu)
      simpa [horder, heq, mul_comm value weight] using And.intro hl hu

variable {d inC outC : Nat} {kernel stride padding inSpatial : Tensor Nat [d]}
  (layer : ConvSpec d inC outC kernel stride padding ℝ)
  (dilation paddingAfter : Tensor Nat [d]) (groups : Nat)

/-- The actual directed convolution transfer encloses the grouped convolution specification
over the reals, independently in every leading batch. -/
theorem ibpConv_contains_groupedConv_real (leading : Shape)
    (box : Box ℝ (leading.concat (Shape.ofList (inC :: inSpatial.data.toList))))
    (input : Tensor ℝ (leading.concat (Shape.ofList (inC :: inSpatial.data.toList))))
    (h : Box.contains box input) :
    Box.contains (ibpConv layer dilation paddingAfter groups leading box)
      (Tensor.mapLeading leading
        (groupedConvSpec (inSpatial := inSpatial) (stride := stride)
          (dilation := dilation) (paddingBefore := padding) (paddingAfter := paddingAfter)
          groups layer.kernel layer.bias) input) := by
  induction leading with
  | scalar =>
      apply (contains_iff_multiIndex _ _ _).mpr
      intro index
      rcases index with ⟨outChannel, outIndex⟩
      simp only [ibpConv, ibpConv.mapRows, Tensor.mapLeading, groupedConvSpec,
        groupedConvCoreSpec, convCoreWith, convBiasBroadcastDilatedSpec,
        convBiasBroadcastWith, MultiIndex.get_addSpec, MultiIndex.get_dim,
        MultiIndex.get_generate, Bool.false_eq_true, ↓reduceIte,
        BoundOps.addDown, BoundOps.addUp, BoundOps.mulDown, BoundOps.mulUp]
      constructor
      · apply add_le_add_left
        apply List.foldl_rel le_rfl
        intro localChannel _ lo exactValue hacc
        apply foldlIndices_rel (· ≤ ·) _ _ hacc
        intro lo exactValue kernelIndex hacc
        cases hindex : mkDilatedInputIdx? outIndex.toList kernelIndex
            stride.data.toList dilation.data.toList padding.data.toList with
        | none => simpa [hindex] using hacc
        | some inputIndex =>
            have hx := contains_getAtOrZero h
              (((outChannel.val / (outC / groups)) * (inC / groups) +
                localChannel.val) :: inputIndex)
            simpa only [hindex] using add_le_add hacc
              (mul_endpoints_enclose (weight := getAtOrZero layer.kernel
                (outChannel.val ::
                  ((outChannel.val / (outC / groups)) * (inC / groups) +
                    localChannel.val) :: kernelIndex)) hx.1 hx.2).1
      · apply add_le_add_left
        apply List.foldl_rel le_rfl
        intro localChannel _ exactValue hi hacc
        apply foldlIndices_rel (· ≤ ·) _ _ hacc
        intro exactValue hi kernelIndex hacc
        cases hindex : mkDilatedInputIdx? outIndex.toList kernelIndex
            stride.data.toList dilation.data.toList padding.data.toList with
        | none => simpa [hindex] using hacc
        | some inputIndex =>
            have hx := contains_getAtOrZero h
              (((outChannel.val / (outC / groups)) * (inC / groups) +
                localChannel.val) :: inputIndex)
            simpa only [hindex] using add_le_add hacc
              (mul_endpoints_enclose (weight := getAtOrZero layer.kernel
                (outChannel.val ::
                  ((outChannel.val / (outC / groups)) * (inC / groups) +
                    localChannel.val) :: kernelIndex)) hx.1 hx.2).2
  | dim n rest ih =>
      intro head
      simpa only [ibpConv, ibpConv.mapRows, Tensor.mapLeading, Tensor.unstack_dim] using
        ih ⟨box.lo.unstack head, box.hi.unstack head⟩ (input.unstack head) (h head)

end NN.MLTheory.CROWN.ConvProof
