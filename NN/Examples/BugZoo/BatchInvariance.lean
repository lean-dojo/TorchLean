/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape

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

/--
Lift a per-example model across a leading batch axis.

This is the reference semantics for batched runtime behavior. It intentionally contains no
cross-example communication, no dynamic batching heuristic, and no hidden state.
-/
def mapBatch {α : Type} {batch : Nat} {sIn sOut : Spec.Shape}
    (f : Spec.Tensor α sIn → Spec.Tensor α sOut)
    (xs : Spec.Tensor α (.dim batch sIn)) :
    Spec.Tensor α (.dim batch sOut) :=
  match xs with
  | Spec.Tensor.dim rows => Spec.Tensor.dim fun i => f (rows i)

/--
Batch-invariance for the reference batched semantics.

This is the theorem a serving/runtime path should aim to preserve, modulo an explicit floating
point tolerance if it deliberately changes reduction order.
-/
theorem mapBatch_select_eq_single {α : Type} {batch : Nat} {sIn sOut : Spec.Shape}
    (f : Spec.Tensor α sIn → Spec.Tensor α sOut)
    (xs : Spec.Tensor α (.dim batch sIn))
    (i : Fin batch) :
    Spec.getAtSpec (mapBatch f xs) i = f (Spec.getAtSpec xs i) := by
  cases xs
  rfl

/--
Composing two per-example stages before batching is the same as batching each stage in sequence.

This is the clean semantic version of a common deployment expectation: batching is an execution
strategy, not a change to the model.
-/
theorem mapBatch_comp {α : Type} {batch : Nat} {s₁ s₂ s₃ : Spec.Shape}
    (f : Spec.Tensor α s₁ → Spec.Tensor α s₂)
    (g : Spec.Tensor α s₂ → Spec.Tensor α s₃)
    (xs : Spec.Tensor α (.dim batch s₁)) :
    mapBatch (fun x => g (f x)) xs = mapBatch g (mapBatch f xs) := by
  cases xs
  rfl

end NN.Examples.BugZoo.BatchInvariance
