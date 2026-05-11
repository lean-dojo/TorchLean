/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape

/-!
# Sequence utilities (spec layer)

These helpers manipulate *sequence tensors*, typically of shape:

`Tensor α (.dim seqLen (.dim featureDim .scalar))`.

They are intentionally simple (structural recursion on `Tensor.dim`) and are used by the
RNN/LSTM/GRU specs.

Why we have a dedicated "sequence" module:

- In the spec layer we write models the way they appear in papers: "for each timestep t, apply ...".
  These helpers let us express that directly, without constantly unpacking `Tensor.dim`.
- Many proofs about RNN-style models want to reason step-by-step. Definitions that recurse
  structurally on `Tensor` tend to be easier to induct over than definitions that go through arrays
  or ad-hoc lists.
- We deliberately keep this file free of runtime concerns (no `IO`, no mutable state). When we care
  about performance, we use the runtime layer; here we care about "the clean mathematical meaning".

PyTorch analogies:

- A tensor of shape `(seqLen, dim)` is written here as `.dim seqLen (.dim dim .scalar)`.
- `mapSequenceSpec f` is like `torch.vmap(f)` over the leading time dimension (conceptually).
- `reduceSumSequenceSpec` is like `seq.sum(dim=0)` (but we require `seqLen ≠ 0` as evidence).

Naming note:
- This file defines `Spec.concatSequenceSpec` for concatenating along the **feature dimension**
  (inner axis, like `torch.cat(..., dim=1)` for `(seqLen, dim)` tensors).
- `NN.Spec.Core.TensorReductionShape` also defines `Spec.Tensor.concatSequenceSpec` for
  concatenating along the **time dimension** (axis 0).
  The names are similar on purpose (both are "sequence concatenation"), but they are different ops.

References / analogies:
- PyTorch `torch.cat` (shape intuition): https://pytorch.org/docs/stable/generated/torch.cat.html
- PyTorch `torch.flip` (reverse/flip intuition):
  https://pytorch.org/docs/stable/generated/torch.flip.html
-/

@[expose] public section


namespace Spec

variable {α : Type} [Context α]

open Tensor

/-- Map a per‑step function over a sequence (vector input → vector output). -/
def mapSequenceSpec {seqLen inDim outDim : Nat}
  (f : Tensor α (.dim inDim .scalar) → Tensor α (.dim outDim .scalar))
  (seq : Tensor α (.dim seqLen (.dim inDim .scalar))) :
  Tensor α (.dim seqLen (.dim outDim .scalar)) :=
  match seq with
  | Tensor.dim sequence_fn =>
    Tensor.dim (fun i => f (sequence_fn i))

/-- Map a per‑step function over a *scalar* sequence, producing vectors. -/
def mapSequenceVecScalarSpec {seqLen dim : Nat}
  (f : Tensor α .scalar → Tensor α (.dim dim .scalar))
  (seq : Tensor α (.dim seqLen .scalar)) :
  Tensor α (.dim seqLen (.dim dim .scalar)) :=
  match seq with
  | Tensor.dim fn =>
    Tensor.dim (fun i => f (fn i))

/--
Zip two sequences together, then apply a vector x vector function at each step.

We pass `outShape` explicitly so the result type is easy for Lean to elaborate at call sites.
This avoids a lot of "stuck" metavariables in larger model definitions.
-/
def map2SequenceSpec2 {seqLen dim1 dim2 : Nat} (outShape : Shape)
  (f : Tensor α (.dim dim1 .scalar) → Tensor α (.dim dim2 .scalar) → Tensor α outShape)
  (seq1 : Tensor α (.dim seqLen (.dim dim1 .scalar)))
  (seq2 : Tensor α (.dim seqLen (.dim dim2 .scalar))) :
  Tensor α (.dim seqLen outShape) :=
  match seq1, seq2 with
  | Tensor.dim fn1, Tensor.dim fn2 =>
    Tensor.dim (fun i => f (fn1 i) (fn2 i))

/-- Zip a vector sequence and a scalar sequence, then apply a vector×scalar function per step. -/
def map2SequenceVecScalarSpec {seqLen dim : Nat} (outShape : Shape)
  (f : Tensor α (.dim dim .scalar) → Tensor α .scalar → Tensor α outShape)
  (seq1 : Tensor α (.dim seqLen (.dim dim .scalar)))
  (seq2 : Tensor α (.dim seqLen .scalar)) :
  Tensor α (.dim seqLen outShape) :=
  match seq1, seq2 with
  | Tensor.dim fn1, Tensor.dim fn2 =>
    Tensor.dim (fun i => f (fn1 i) (fn2 i))


/--
Reduce a sequence by summing along the time axis (axis = 0).

We ask for `seqLen ≠ 0` because downstream uses often combine this with `mean`-like operations
or want to avoid degenerate "empty sequence" cases in proofs. (PyTorch defines behavior for
zero-length dims, but we prefer making those cases explicit in the spec layer.)
-/
def reduceSumSequenceSpec {seqLen dim : Nat}
  (seq : Tensor α (.dim seqLen (.dim dim .scalar))) (h : seqLen ≠ 0) :
  Tensor α (.dim dim .scalar) :=
  have _ : Shape.valid_axis_inst 0 (Shape.dim seqLen (Shape.dim dim Shape.scalar)) :=
            Shape.validAxisInstZeroAlt h
  reduceSumAuto 0 seq

/-- Like `reduceSumSequenceSpec`, but for 3-D tensors (seqLen × outputSize × hiddenSize). -/
def reduceSumSequenceSpec2 {seqLen outputSize hiddenSize : Nat}
  (seq : Tensor α (.dim seqLen (.dim outputSize (.dim hiddenSize .scalar)))) (h : seqLen ≠ 0) :
  Tensor α (.dim outputSize (.dim hiddenSize .scalar)) :=
  have _ : Shape.valid_axis_inst 0 (Shape.dim seqLen (Shape.dim outputSize (Shape.dim hiddenSize
    Shape.scalar))) :=
            Shape.validAxisInstZeroAlt h
  reduceSumAuto 0 seq

/--
Reverse a sequence along its time axis.

PyTorch analogy: `seq.flip(dims=[0])`.
-/
def reverseSequenceSpec {seqLen dim : Nat}
  (seq : Tensor α (.dim seqLen (.dim dim .scalar))) :
  Tensor α (.dim seqLen (.dim dim .scalar)) :=
  match seq with
  | Tensor.dim sequence_fn =>
    Tensor.dim (fun i => sequence_fn ⟨seqLen - 1 - i.val, by grind⟩)

/--
Concatenate two sequences along the feature dimension.

PyTorch analogy: `torch.cat([seq1, seq2], dim=1)` when shapes are `(seqLen, dim1)` and `(seqLen,
  dim2)`.

Do not confuse this with `Spec.Tensor.concatSequenceSpec` (defined in
`NN.Spec.Core.TensorReductionShape`), which concatenates along the time axis (axis 0).
-/
def concatSequenceSpec {seqLen dim1 dim2 : Nat}
  (seq1 : Tensor α (.dim seqLen (.dim dim1 .scalar)))
  (seq2 : Tensor α (.dim seqLen (.dim dim2 .scalar))) :
  Tensor α (.dim seqLen (.dim (dim1 + dim2) .scalar)) :=
  match seq1, seq2 with
  | Tensor.dim fn1, Tensor.dim fn2 =>
    Tensor.dim (fun i => concatVectorsSpec (fn1 i) (fn2 i))

/--
Reduce a batch of vectors by summing over the batch axis (explicit axis = 0).

This axis-explicit helper takes an `axis` argument plus a proof it equals `0`.
Prefer `reduce_sum_vec2` when the axis is fixed by the surrounding code.
-/
def reduceSumVec {α : Type} [Add α] [Zero α]
  {inDim batch : Nat}
  (axis : Nat)
  (t : Tensor α (Shape.dim batch (Shape.dim inDim .scalar)))
  (_ : axis = 0) :
  Tensor α (Shape.dim inDim .scalar) :=
  match t with
  | Tensor.dim xs =>
      (List.finRange batch).foldl (fun acc i => map2Spec (· + ·) acc (xs i)) (fill 0 (Shape.dim
        inDim .scalar))

/--
Same as `reduce_sum_vec`, but avoids the explicit axis argument.

We define this with a small Nat loop so the recursion is obvious to Lean and doesn't depend on
`List.finRange`. It's the same mathematical operation: sum all batch vectors pointwise.
-/
def reduceSumVec2 {α : Type} [Add α] [Zero α]
    {batch inDim : Nat}
    (t : Tensor α (.dim batch (.dim inDim .scalar))) :
    Tensor α (.dim inDim .scalar) :=
  match t with
  | .dim batchFn =>
    -- fold over Fin batch
    let rec loop (i : Nat) (acc : Tensor α (.dim inDim .scalar)) :=
      if h : i < batch then
        let v := batchFn ⟨i, h⟩
        loop (i+1) (map2Spec (·+·) acc v)
      else acc
    loop 0 (Tensor.dim (fun _ => Tensor.scalar 0))
