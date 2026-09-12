/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor

/-!
# BugZoo: batch-invariance contracts

Serving systems often batch unrelated requests together. Recent systems discussions and inference
engine bug studies point out that dynamic batching, kernel selection, and reduction order can make
outputs depend on who else happened to be in the batch.

Reference:
- Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference", 2025.
- Liu et al., "A First Look at Bugs in LLM Inference Engines", 2025.

TorchLean cannot prove arbitrary CUDA kernels batch-invariant unless the kernel implementation is
also connected to the spec. This file records the semantic target: if a model is lifted across the
batch axis by applying the same function independently to every row, then selecting one row of the
batched result is exactly the same as evaluating that row alone.
-/

@[expose] public section

namespace NN.Examples.BugZoo.BatchInvariance

open TorchLean

/-- Two requests, each with three features. -/
def requests : Tensor Float [2, 3] := [[1, 2, 3], [4, 5, 6]]

/-- Square each row independently using the public leading-axis map. -/
def squaredRows : Tensor Float [2, 3] :=
  Tensor.mapLeading [2] (fun row => Tensor.mul row row) requests

/-- The second request receives the same result when evaluated on its own. -/
example : squaredRows.unstack 1 =
    Tensor.mul (requests.unstack 1) (requests.unstack 1) := by
  exact Tensor.unstack_mapLeading _ requests 1

/-- Two stages can be mapped separately or composed before batching. -/
example {α : Type} [Storage α] {batch : Nat} {s₁ s₂ s₃ : Spec.Shape}
    (f : Tensor α s₁ → Tensor α s₂) (g : Tensor α s₂ → Tensor α s₃)
    (xs : Tensor α (s₁.prependDim batch)) :
    Tensor.mapLeading [batch] (fun x => g (f x)) xs =
      Tensor.mapLeading [batch] g (Tensor.mapLeading [batch] f xs) :=
  Tensor.mapLeading_comp [batch] f g xs

end NN.Examples.BugZoo.BatchInvariance
