/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Train.Dataset

/-!
# Tensor loaders

These helpers convert simple in-memory Lean containers into typed tensors and datasets.

This is the canonical runtime layer for shape-checked construction from lists/arrays. The public
API re-exports these helpers from `NN.API.Data`; that is not a second implementation, just the
user-facing namespace. Keeping the implementation here is useful because the CSV/NPY loaders and
training examples can share the same `Result` error model and the same `Dataset` type without
depending on the higher-level API facade.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace Train

open Spec
open Tensor

/-!
## Tensor construction from lists
-/
/--
Build a length-`n` vector from a list, with a length check.

This is the safe list boundary for dependent tensor shapes: callers bring ordinary runtime data,
and the loader either proves the length matches `n` internally or returns a tagged error.
-/
def vectorOfList {a : Type} [Zero a]
  (tag : String) (n : Nat) (xs : List a) : Result (Tensor a (.dim n .scalar)) := by
  if hLen : xs.length = n then
    -- Use `List.get` via a casted `Fin` index: the length check guarantees in-bounds access.
    let f : Fin n → a := fun i =>
      xs.get (Fin.cast hLen.symm i)
    exact .ok (vectorN n f)
  else
    exact .error (tagError tag s!"expected {n} entries, got {xs.length}")

/-- Build a length-`n` vector from an `Array`, using the same checks as `vectorOfList`. -/
def vectorOfArray {a : Type} [Zero a]
  (tag : String) (n : Nat) (xs : Array a) : Result (Tensor a (.dim n .scalar)) :=
  vectorOfList (tag := tag) (n := n) xs.toList

/--
Build an `m×n` matrix from a list of `m` rows, each of length `n`.

Both dimensions are checked before constructing the dependent tensor. If a row has the wrong
length, the error reports the first bad row so data issues are easier to diagnose.
-/
def matrixOfLists {a : Type} [Zero a]
  (tag : String) (m n : Nat) (rows : List (List a)) :
  Result (Tensor a (.dim m (.dim n .scalar))) := by
  if hRows : rows.length = m then
    let colsOk : Bool := rows.all (fun r => decide (r.length = n))
    if hColsOk : colsOk = true then
      -- Helper: elements returned by `List.get` are always members of the list.
      have get_mem {α : Type} (xs : List α) (i : Fin xs.length) : xs.get i ∈ xs := by
        exact List.get_mem xs i

      -- From `all`-success we can recover row-length equalities for every row in `rows`.
      have hAll :
          ∀ row : List a, row ∈ rows → row.length = n := by
        intro row hMem
        have hallBool : rows.all (fun r => decide (r.length = n)) = true := by
          simpa [colsOk] using hColsOk
        have : decide (row.length = n) = true :=
          (List.all_eq_true.mp hallBool) row hMem
        exact of_decide_eq_true this

      let f : Fin m → Fin n → a := fun i j =>
        let i' : Fin rows.length := Fin.cast hRows.symm i
        let row : List a := rows.get i'
        have hRowLen : row.length = n := hAll row (get_mem rows i')
        let j' : Fin row.length := Fin.cast hRowLen.symm j
        row.get j'
      exact .ok (matrixMN m n f)
    else
      -- Try to report the first offending row for a more actionable error message.
      let rec findBad : Nat → List (List a) → Option (Nat × Nat)
        | _i, [] => none
        | i, row :: rest =>
            if row.length = n then
              findBad (i + 1) rest
            else
              some (i + 1, row.length)
      match findBad 0 rows with
      | some (i, len) =>
          exact .error (tagError tag s!"row {i}: expected {n} entries, got {len}")
      | none =>
          exact .error (tagError tag s!"expected each row to have {n} entries")
  else
    exact .error (tagError tag s!"expected {m} rows, got {rows.length}")

/-- Build an `m×n` matrix from an `Array (Array a)`, using the same checks as `matrixOfLists`. -/
def matrixOfArrays {a : Type} [Zero a]
  (tag : String) (m n : Nat) (rows : Array (Array a)) :
  Result (Tensor a (.dim m (.dim n .scalar))) :=
  matrixOfLists (tag := tag) (m := m) (n := n) (rows := rows.toList.map Array.toList)

/-- Array-backed vector loading is definitionally list-backed vector loading after `toList`. -/
@[simp] theorem vectorOfArray_eq_vectorOfList {a : Type} [Zero a]
    (tag : String) (n : Nat) (xs : Array a) :
    vectorOfArray (a := a) (tag := tag) (n := n) xs =
      vectorOfList (tag := tag) (n := n) xs.toList := rfl

/-- Array-backed matrix loading is definitionally list-backed matrix loading after `toList`. -/
@[simp] theorem matrixOfArrays_eq_matrixOfLists {a : Type} [Zero a]
    (tag : String) (m n : Nat) (rows : Array (Array a)) :
    matrixOfArrays (a := a) (tag := tag) (m := m) (n := n) rows =
      matrixOfLists (tag := tag) (m := m) (n := n) (rows := rows.toList.map Array.toList) := rfl

/-!
## Dataset helpers
-/
/-- Zip two lists into a `Dataset` of pairs, with a length check. -/
def datasetOfPairs {a b : Type}
  (tag : String) (xs : List a) (ys : List b) : Result (Dataset (Prod a b)) := by
  if xs.length = ys.length then
    exact .ok (Dataset.ofList (List.zip xs ys))
  else
    exact .error (tagError tag s!"length mismatch: {xs.length} vs {ys.length}")

/-- Convert list-rows into a dataset of length-`n` vectors. -/
def datasetOfListVectors {a : Type} [Zero a]
  (tag : String) (n : Nat) (rows : List (List a)) :
  Result (Dataset (Tensor a (.dim n .scalar))) := do
  let tensors <- rows.mapM (fun row => vectorOfList (tag := tag) (n := n) row)
  pure (Dataset.ofList tensors)

/-- Pair datasets succeed exactly as `List.zip` when the two sides have the same length. -/
@[simp] theorem datasetOfPairs_eq_ok {a b : Type}
    (tag : String) (xs : List a) (ys : List b) (h : xs.length = ys.length) :
    datasetOfPairs (a := a) (b := b) tag xs ys = .ok (Dataset.ofList (List.zip xs ys)) := by
  simp [datasetOfPairs, h]

/-- Pair datasets fail before zipping if the two sides have different lengths. -/
@[simp] theorem datasetOfPairs_eq_error {a b : Type}
    (tag : String) (xs : List a) (ys : List b) (h : xs.length ≠ ys.length) :
    datasetOfPairs (a := a) (b := b) tag xs ys =
      .error (tagError tag s!"length mismatch: {xs.length} vs {ys.length}") := by
  simp [datasetOfPairs, h]

end Train
end Autograd
end Runtime
