/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Algebra.BigOperators.Fin
public import Mathlib.Data.Fintype.Basic
public import Mathlib.Data.Real.Basic

/-!
# Gather Rows / Embedding Lookup

This file proves the small linear-algebra fact behind token embeddings:

* forward gathers rows from an embedding table, and
* reverse mode scatters the upstream row gradients back into the table, summing repeated indices.

The theorem is deliberately stated with `Fin` indices. Runtime APIs often receive `Nat` token IDs
and perform bounds checks or totalization; the clean proof primitive should not mix that IO/error
policy into the VJP theorem. A later runtime bridge can say: if every `Nat` ID is in bounds, the
runtime gather/scatter path agrees with this `Fin`-indexed specification.

PyTorch analogy:

* `torch.nn.Embedding` forward is row gather.
* Its backward wrt the embedding table is scatter-add over token positions.
-/

@[expose] public section

namespace Proofs
namespace Autograd
namespace Embedding

open scoped BigOperators

/-- Matrix-style inner product for finite row/column tables. -/
def matInner {rows cols : Nat} (A B : Fin rows → Fin cols → ℝ) : ℝ :=
  ∑ c : Fin cols, ∑ r : Fin rows, A r c * B r c

/-- Gather rows from a table using in-bounds finite token IDs. -/
def gatherRows {vocab dim k : Nat}
    (table : Fin vocab → Fin dim → ℝ) (idx : Fin k → Fin vocab) :
    Fin k → Fin dim → ℝ :=
  fun j c => table (idx j) c

/--
Scatter-add row cotangents back into an embedding table.

If the same row appears several times in `idx`, its gradient contributions are summed.
-/
def scatterAddRows {vocab dim k : Nat}
    (idx : Fin k → Fin vocab) (dY : Fin k → Fin dim → ℝ) :
    Fin vocab → Fin dim → ℝ :=
  fun r c => ∑ j : Fin k, if idx j = r then dY j c else 0

/--
Gather and scatter-add are adjoint.

This is the local VJP theorem for embedding lookup wrt the embedding table:

`<gatherRows dTable idx, dY> = <dTable, scatterAddRows idx dY>`.
-/
theorem gatherRows_scatterAddRows_adjoint {vocab dim k : Nat}
    (dTable : Fin vocab → Fin dim → ℝ)
    (idx : Fin k → Fin vocab)
    (dY : Fin k → Fin dim → ℝ) :
    matInner (gatherRows dTable idx) dY =
      matInner dTable (scatterAddRows idx dY) := by
  classical
  unfold matInner gatherRows scatterAddRows
  apply Finset.sum_congr rfl
  intro c _
  calc
    ∑ j : Fin k, dTable (idx j) c * dY j c
        =
      ∑ r : Fin vocab, ∑ j ∈ (Finset.univ : Finset (Fin k)) with idx j = r,
        dTable (idx j) c * dY j c := by
          symm
          simpa using
            (Finset.sum_fiberwise (s := (Finset.univ : Finset (Fin k))) (g := idx)
              (f := fun j => dTable (idx j) c * dY j c))
    _ =
      ∑ r : Fin vocab, dTable r c * ∑ j : Fin k, if idx j = r then dY j c else 0 := by
        apply Finset.sum_congr rfl
        intro r _
        rw [Finset.mul_sum]
        simp only [Finset.sum_filter]
        apply Finset.sum_congr rfl
        intro j _
        by_cases h : idx j = r
        · simp [h]
        · simp [h]

end Embedding
end Autograd
end Proofs
