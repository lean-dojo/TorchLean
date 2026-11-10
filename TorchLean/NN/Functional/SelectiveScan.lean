/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Program

/-!
# Differentiable diagonal selective scans

Each step computes `h[t] = A[t] * h[t-1] + B[t] * X[t]` with elementwise products, and row `t` of
the result is the state after input `t`. Eager CUDA execution records one native
`Op.selectiveScan` node, whose backward propagates cotangents to the coefficients, the inputs and
the initial state. Every other interpreter, eager CPU execution included, records the recurrence
with ordinary differentiable operations (`SelectiveScan.variableReference`).

The final row can seed another call as its initial state without cutting the gradient. An empty
sequence returns an empty result that does not depend on the initial state.
-/

@[expose] public section

namespace TorchLean.nn.functional

open Ops

variable {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]

namespace SelectiveScan

/--
Reference recurrence with per-token coefficients: the first row of each input updates the initial
state, and the remaining rows recurse from the updated state.
-/
def variableReference {state : Nat} :
    {seqLen : Nat} → Ref (m := m) (α := α) [seqLen, state] → Ref (m := m) (α := α) [seqLen, state] →
      Ref (m := m) (α := α) [seqLen, state] → Ref (m := m) (α := α) [state] →
      m (Ref (m := m) (α := α) [seqLen, state])
  | 0, _, _, _, _ => Ops.const (Tensor.zeros (α := α) [0, state])
  | count + 1, a, b, x, initial => do
      let a0 ← Ops.apply
          (.indexAt ⟨0, Shape.AxisInBounds.proof⟩
            ⟨0, by simpa only [Shape.extent_cons_zero] using Nat.succ_pos count⟩) (a, ())
      let b0 ← Ops.apply
          (.indexAt ⟨0, Shape.AxisInBounds.proof⟩
            ⟨0, by simpa only [Shape.extent_cons_zero] using Nat.succ_pos count⟩) (b, ())
      let x0 ← Ops.apply
          (.indexAt ⟨0, Shape.AxisInBounds.proof⟩
            ⟨0, by simpa only [Shape.extent_cons_zero] using Nat.succ_pos count⟩) (x, ())
      let retained ← Ops.apply .mul (a0, initial, ())
      let written ← Ops.apply .mul (b0, x0, ())
      let next ← Ops.apply .add (retained, written, ())
      let aTail ←
          (Ref.cast
            (Shape.slice_singleton (.dim _ _) ⟨0, by simp⟩ 1 count) <$>
            Ops.apply
              (.slice (s := .dim _ _) [⟨0, by simp⟩] [1] [1 + count] [1]
                (Shape.sliceValid_singleton _ ⟨0, by simp⟩ 1 count
                  (by show 1 + count ≤ count + 1; omega))) (a, ()))
      let bTail ←
          (Ref.cast
            (Shape.slice_singleton (.dim _ _) ⟨0, by simp⟩ 1 count) <$>
            Ops.apply
              (.slice (s := .dim _ _) [⟨0, by simp⟩] [1] [1 + count] [1]
                (Shape.sliceValid_singleton _ ⟨0, by simp⟩ 1 count
                  (by show 1 + count ≤ count + 1; omega))) (b, ()))
      let xTail ←
          (Ref.cast
            (Shape.slice_singleton (.dim _ _) ⟨0, by simp⟩ 1 count) <$>
            Ops.apply
              (.slice (s := .dim _ _) [⟨0, by simp⟩] [1] [1 + count] [1]
                (Shape.sliceValid_singleton _ ⟨0, by simp⟩ 1 count
                  (by show 1 + count ≤ count + 1; omega))) (x, ()))
      let tail ← variableReference aTail bTail xTail next
      let head ← Ops.apply
          (.reshape (t := [1, state]) (by simp)) (next, ())
      let result : Ref (m := m) (α := α) [1 + count, state] ←
        Ops.apply (.cat (s := .dim 0 [state]) ⟨0, by simp⟩ [1, count])
          (head, tail, ())
      pure (by simpa [Nat.one_add] using result)

end SelectiveScan

/--
Diagonal selective scan with independent coefficients for every token.

All three sequence inputs have shape `[seqLen, state]`. Gradients flow through `a`, `b`, `x` and
`initial`, also when the coefficients are computed from the input tokens.
-/
def selectiveScanDiagVar {seqLen state : Nat}
    (a b x : Ref (m := m) (α := α) [seqLen, state]) (initial : Ref (m := m) (α := α) [state]) :
    m (Ref (m := m) (α := α) [seqLen, state]) := do
  if let some native := Ops.applyNative? (m := m) (α := α) then
    if let some result ← native
        (.selectiveScan (s := [seqLen, state]) ⟨0, Nat.zero_lt_two⟩ false false)
        (a, b, x, initial, ()) then
      return result
  SelectiveScan.variableReference a b x initial

/--
Diagonal scan with coefficient vectors `a` and `b` shared across time.

Reverse mode sums the contributions to each shared coefficient over all steps. The native kernel
accumulates them itself, and the reference route gets the sum from the pullback of its broadcast.
-/
def selectiveScanDiag {seqLen state : Nat}
    (a b : Ref (m := m) (α := α) [state]) (x : Ref (m := m) (α := α) [seqLen, state])
    (initial : Ref (m := m) (α := α) [state]) : m (Ref (m := m) (α := α) [seqLen, state]) := do
  if let some native := Ops.applyNative? (m := m) (α := α) then
    if let some result ← native
        (.selectiveScan (s := [seqLen, state]) ⟨0, Nat.zero_lt_two⟩ true true)
        (a, b, x, initial, ()) then
      return result
  let aRows ← Ops.apply
      (.broadcast (t := [seqLen, state]) (Shape.BroadcastTo.proof).dims
        (Shape.BroadcastTo.proof).broadcastDims) (a, ())
  let bRows ← Ops.apply
      (.broadcast (t := [seqLen, state]) (Shape.BroadcastTo.proof).dims
        (Shape.BroadcastTo.proof).broadcastDims) (b, ())
  SelectiveScan.variableReference aRows bRows x initial

end TorchLean.nn.functional
