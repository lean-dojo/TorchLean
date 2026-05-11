/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Shape
public import NN.Spec.Core.Tensor
public import NN.Spec.Core.TensorArray

/-!
# TensorBridge

Computable, row-major conversions between `TensorArray.Tensor` and `Spec.Tensor`.

Why we have this file:

- `Spec.Tensor` is the *spec* representation (functional, shape-indexed).
- `TensorArray.Tensor` is an *array-backed* representation that is convenient for IO and external
  interfaces (CSV/NPY/etc).

This bridge gives explicit, easy-to-audit conversions:

- `flatten` / `unflatten` between `Spec.Tensor` and a flat row-major list,
- plus shape list conversions (`Shape ↔ List Nat`) to connect the typed and runtime views.

## Why “row-major”?

Our spec tensor datatype is a nested `dim` tree: outer dimensions are modeled as functions
`Fin n → ...`. Our `flatten` recursion iterates outer indices first and then recurses inward, so
the **last axis varies fastest** (the conventional row-major / C-order layout for matrices and
images when you think of the outer dimension as “rows”).

We could have picked a different order, but we need one consistent convention for interop (CSV/NPY,
torch tensor serialization, etc.). Row-major matches the contiguous layout assumptions used in many
toolchains, so it’s a practical default.

## Notes for proofs

We don’t want “conversion code” to be a place where meaning quietly changes. So we prove the basic
round-trip facts:

- flattening after unflattening gives you the original list, and
- unflattening after flattening gives you the original tensor.

Those lemmas let us move between the functional spec view and the array/list view without
handwaving.

Example file: `NN.Examples.Advanced.Tensor.TensorBridge`.

References / analogies:
- Row-major ("C-order") is the default contiguous layout in many toolchains (NumPy/PyTorch) and is a
  common convention for serialization. This file fixes a single convention so interop is
  auditably well-defined.
- PyTorch docs (intuition for the operations we mirror here, not the semantics):
  - `torch.flatten`: https://pytorch.org/docs/stable/generated/torch.flatten.html
  - `torch.Tensor.reshape`: https://pytorch.org/docs/stable/generated/torch.Tensor.reshape.html
  - `torch.Tensor.view`: https://pytorch.org/docs/stable/generated/torch.Tensor.view.html
  - `torch.Tensor.contiguous`:
    https://pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html
- NumPy docs (C-order / reshape conventions):
  - `numpy.reshape`: https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
-/

@[expose] public section


namespace TensorBridge

open TensorArray Spec

/-- Convert a `Shape` to a runtime list of dimensions. -/
def shapeToList : Shape → List Nat :=
  Shape.toList

/-- Convert a runtime list of dimensions to a `Shape`. -/
def listToShape : List Nat → Shape :=
  Shape.ofList

/-- Shape conversion is involutive (applying it twice gives the original). -/
theorem shapeToList_listToShape_involutive (s : Shape) :
  listToShape (shapeToList s) = s := by
  simp [shapeToList, listToShape]

/-- List-to-shape conversion is involutive. -/
theorem listToShape_shapeToList_involutive (shape : List Nat) :
  shapeToList (listToShape shape) = shape := by
  simp [shapeToList, listToShape]

/-- Flatten a `Spec.Tensor` of list-shape to row-major list. -/
def flatten {α : Type} : ∀ {shape : List Nat}, Spec.Tensor α (listToShape shape) → List α
  | [], Spec.Tensor.scalar x => [x]
  | n :: ns, Spec.Tensor.dim f =>
    (List.finRange n).flatMap (fun i => flatten (shape := ns) (f i))

/-- Length of `flatten` is exactly the product of dimensions. -/
theorem flatten_length {α : Type} : ∀ {shape : List Nat} (t : Spec.Tensor α (listToShape shape)),
    (flatten t).length = TensorArray.shapeProd shape
  | [], Spec.Tensor.scalar _ => by
      simp [flatten]
  | n :: ns, Spec.Tensor.dim f => by
      let k := TensorArray.shapeProd ns
      have hlen : ∀ i : Fin n, (flatten (shape := ns) (f i)).length = k := by
        intro i
        simpa [k] using (flatten_length (shape := ns) (f i))
      simp [flatten, List.length_flatMap]
      have hmap :
          (List.map (fun i : Fin n => (flatten (shape := ns) (f i)).length) (List.finRange n))
            = (List.map (fun _ : Fin n => k) (List.finRange n)) := by
        apply List.map_congr_left
        intro i hi
        simpa using hlen i
      rw [hmap]
      simp [k]

/-- A block slice of size `k` has length `k` when the full list has length `n*k`. -/
theorem length_take_drop_mul {α : Type} (xs : List α) {n k : Nat}
    (h : xs.length = n * k) (i : Fin n) :
    ((xs.drop (i.val * k)).take k).length = k := by
  have hpos : 0 < n - i.val := Nat.sub_pos_of_lt i.isLt
  have h1 : 1 ≤ n - i.val := Nat.succ_le_of_lt hpos
  have hle_mul : k ≤ (n - i.val) * k := by
    simpa using (Nat.mul_le_mul_right k h1)
  have hle : k ≤ n * k - i.val * k := by
    simpa [Nat.sub_mul] using hle_mul
  have hle2 : k ≤ xs.length - i.val * k := by
    simpa [h] using hle
  have hle3 : k ≤ (xs.drop (i.val * k)).length := by
    simpa [List.length_drop] using hle2
  rw [List.length_take]
  exact Nat.min_eq_left hle3

/-- Unflatten a row-major list into a `Spec.Tensor` of list-shape. -/
def unflatten {α : Type} : ∀ (shape : List Nat) (xs : List α),
    xs.length = TensorArray.shapeProd shape → Spec.Tensor α (listToShape shape)
  | [], xs, h =>
    have hlen : xs.length = 1 := by
      simpa [TensorArray.shapeProd] using h
    have h0 : 0 < xs.length := by
      simp [hlen]
    Spec.Tensor.scalar (xs.get ⟨0, h0⟩)
  | n :: ns, xs, h =>
    let k := TensorArray.shapeProd ns
    have hx : xs.length = n * k := by
      simpa [k] using h
    Spec.Tensor.dim (fun i : Fin n =>
      unflatten ns ((xs.drop (i.val * k)).take k) (by
        simpa [k] using (length_take_drop_mul xs hx i)))

/-- Reconstruct a list by concatenating its `k`-sized blocks. -/
theorem flatMap_drop_take_eq {α : Type} (n k : Nat) (xs : List α) (h : xs.length = n * k) :
    (List.finRange n).flatMap (fun i : Fin n => (xs.drop (i.val * k)).take k) = xs := by
  induction n generalizing xs with
  | zero =>
      have hx : xs = [] := (List.length_eq_zero_iff).1 (by simpa using h)
      simp [hx]
  | succ n ih =>
      have hdrop : (xs.drop k).length = n * k := by
        simp [List.length_drop, h, Nat.succ_mul]
      simp [List.finRange_succ, List.flatMap_map]
      have ih' := ih (xs := xs.drop k) (h := hdrop)
      have hfun :
          (fun a : Fin n => List.take k (List.drop ((↑a + 1) * k) xs))
            = (fun a : Fin n => List.take k (List.drop (↑a * k) (List.drop k xs))) := by
        funext a
        simp [Nat.succ_mul, List.drop_drop, Nat.add_comm]
      rw [hfun]
      rw [ih']
      exact List.take_append_drop k xs

/-- Flatten after unflatten returns the original list. -/
theorem flatten_unflatten {α : Type} :
    ∀ {shape : List Nat} (xs : List α) (h : xs.length = TensorArray.shapeProd shape),
      flatten (unflatten shape xs h) = xs
  | [], xs, h => by
      have hlen : xs.length = 1 := by
        simpa [TensorArray.shapeProd] using h
      rcases (List.length_eq_one_iff).1 hlen with ⟨a, rfl⟩
      simp [unflatten, flatten]
  | n :: ns, xs, h => by
      let k := TensorArray.shapeProd ns
      have hx : xs.length = n * k := by
        simpa [k] using h
      simp [flatten, unflatten, flatten_unflatten (shape := ns)]
      exact flatMap_drop_take_eq n k xs hx

/-- Slice out the `i`-th `k`-block from a flatMap of equal-length blocks. -/
private theorem take_drop_flatMap_finRange_eq {α : Type} :
    ∀ {n : Nat} (k : Nat) (blocks : Fin n → List α)
      (_hlen : ∀ i : Fin n, (blocks i).length = k) (i : Fin n),
      (((List.finRange n).flatMap blocks).drop (i.val * k)).take k = blocks i := by
  intro n
  induction n with
  | zero =>
      intro k blocks _hlen i
      cases i with
      | mk _ isLt => cases isLt
  | succ n ih =>
      intro k blocks _hlen i
      refine Fin.cases ?case0 ?caseSucc i
      ·
        simp [List.finRange_succ, List.flatMap_map]
        have h0 : (blocks 0).length = k := _hlen 0
        have hk : k ≤ (blocks 0).length := by
          simp [h0]
        calc
          List.take k (blocks 0 ++ List.flatMap (fun a : Fin n => blocks a.succ) (List.finRange n))
              = List.take k (blocks 0) := by
                  simpa using
                    (List.take_append_of_le_length
                      (l₁ := blocks 0)
                      (l₂ := List.flatMap (fun a : Fin n => blocks a.succ) (List.finRange n))
                      (i := k)
                      hk)
          _ = blocks 0 := by
              simp [h0]
      · intro j
        simp [List.finRange_succ, List.flatMap_map]
        let rest : List α := List.flatMap (fun a : Fin n => blocks a.succ) (List.finRange n)
        have h0 : (blocks 0).length = k := _hlen 0
        have hk : k ≤ (blocks 0).length := by
          simp [h0]
        have hlen' : ∀ a : Fin n, (blocks a.succ).length = k := by
          intro a; simpa using (_hlen a.succ)
        have ih' := ih k (fun a : Fin n => blocks a.succ) hlen' j
        have hdrop_succ :
            List.drop ((↑j + 1) * k) (blocks 0 ++ rest)
              = List.drop (↑j * k) (List.drop k (blocks 0 ++ rest)) := by
          simp [Nat.succ_mul, List.drop_drop, Nat.add_comm]
        have hdrop0 : List.drop k (blocks 0 ++ rest) = rest := by
          calc
            List.drop k (blocks 0 ++ rest)
                = List.drop k (blocks 0) ++ rest := by
                    simpa using
                      (List.drop_append_of_le_length (l₁ := blocks 0) (l₂ := rest) (i := k) hk)
            _ = [] ++ rest := by
                    simp [h0]
            _ = rest := by simp
        have hdrop : List.drop ((↑j + 1) * k) (blocks 0 ++ rest) = List.drop (↑j * k) rest := by
          simpa [hdrop0] using hdrop_succ
        simpa [rest, hdrop] using ih'

/-- Unflatten after flatten returns the original tensor (for any valid length proof). -/
theorem unflatten_flatten {α : Type} :
    ∀ {shape : List Nat} (t : Spec.Tensor α (listToShape shape))
      {h : (flatten t).length = TensorArray.shapeProd shape},
      unflatten shape (flatten t) h = t
    := by
  intro shape t
  induction shape with
  | nil =>
      cases t with
      | scalar x =>
          intro h
          simp [flatten, unflatten]
  | cons n ns ih =>
      cases t with
      | dim f =>
          intro h
          let k := TensorArray.shapeProd ns
          simp [unflatten, flatten]
          refine (Eq.mpr (Spec.Tensor.dim.injEq _ f) ?_)
          funext i
          have hlenBlocks : ∀ j : Fin n, (flatten (shape := ns) (f j)).length = k := by
            intro j
            simpa [k] using (flatten_length (shape := ns) (f j))
          have hchunkk :
              (((List.finRange n).flatMap (fun j : Fin n => flatten (shape := ns) (f j))).drop
                (i.val * k)).take k
                = flatten (shape := ns) (f i) := by
            simpa using
              (take_drop_flatMap_finRange_eq (k := k)
                (blocks := fun j : Fin n => flatten (shape := ns) (f j))
                hlenBlocks
                i)
          have hchunk :
              (List.take (TensorArray.shapeProd ns)
                    (List.drop (i.val * TensorArray.shapeProd ns)
                      (List.flatMap (fun j : Fin n => flatten (shape := ns) (f j)) (List.finRange
                        n))))
                  = flatten (shape := ns) (f i) := by
            simpa [k] using hchunkk
          simpa [hchunk] using (ih (t := f i))

/-- Convert from `TensorArray.Tensor` to `Spec.Tensor`. -/
def toTensor {α : Type} {shape : List Nat} :
  TensorArray.Tensor α shape → Spec.Tensor α (listToShape shape)
  | t =>
    unflatten shape t.data.toList (by
      calc
        t.data.toList.length = t.data.size := TensorArray.Array.size_toList t.data
        _ = TensorArray.shapeProd shape := by
            simpa [TensorArray.shapeProd] using t.shape_valid)

/-- Convert from `Spec.Tensor` to `TensorArray.Tensor`. -/
def toTensorArray {α : Type} {shape : List Nat} :
  Spec.Tensor α (listToShape shape) → TensorArray.Tensor α shape
  | t =>
    let xs := flatten t
    TensorArray.ofArray xs.toArray shape (by
      calc
        xs.toArray.size = xs.length := by
          simp
        _ = TensorArray.shapeProd shape := by
          simpa using (flatten_length (t := t)))

/-- Converting to tensor and back preserves the original tensor array. -/
theorem to_tensor_array_to_tensor {α : Type} {shape : List Nat} (t : TensorArray.Tensor α shape) :
  toTensorArray (toTensor t) = t := by
  cases t with
  | mk data hv =>
    simp [toTensor, toTensorArray, TensorArray.ofArray, flatten_unflatten, Array.toArray_toList]

/-- Converting to tensor array and back preserves the original tensor. -/
theorem to_tensor_to_tensor_array {α : Type} {shape : List Nat} (t : Spec.Tensor α (listToShape
  shape)) :
  toTensor (toTensorArray t) = t := by
  simp [toTensor, toTensorArray, TensorArray.ofArray, unflatten_flatten]

/-- A definable equivalence between the two tensor representations. -/
def tensorArrayEquivTensor (α : Type) (shape : List Nat) :
  TensorArray.Tensor α shape ≃ Spec.Tensor α (listToShape shape) :=
  Equiv.mk
    (toTensor (α := α) (shape := shape))
    (toTensorArray (α := α) (shape := shape))
    (by intro t; simpa using (to_tensor_array_to_tensor (α := α) (shape := shape) t))
    (by intro t; simpa using (to_tensor_to_tensor_array (α := α) (shape := shape) t))

end TensorBridge
