/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Program
public import NN.Runtime.Autograd.Torch.Core.Functional.Ops

/-!
# Differentiable diagonal selective scans

Each step computes `h[t] = A[t] * h[t-1] + B[t] * X[t]`, with elementwise products.
The result contains every state after its corresponding input. The generic interpreter records
the recurrence using ordinary differentiable operations; the native interpreter records one
scan node and propagates cotangents to coefficients, inputs, and the initial state.

The final row can be passed as the initial state of another call without cutting its gradient.
An empty sequence returns an empty result and has no dependence on the initial state.
-/

@[expose] public section

namespace Runtime.Autograd.Model.F

open Spec TorchLean

variable {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]

namespace SelectiveScan

/-- Reference recurrence with token-dependent coefficients and an explicit initial state. -/
def variableReference {state : Nat} :
    {seqLen : Nat} → RefTy m α [seqLen, state] → RefTy m α [seqLen, state] →
      RefTy m α [seqLen, state] → RefTy m α [state] → m (RefTy m α [seqLen, state])
  | 0, _, _, _, _ => const (Tensor.zeros (α := α) [0, state])
  | count + 1, a, b, x, initial => do
      let a0 ← select 0 a ⟨0, by simpa only [Shape.axisSize_zero] using Nat.succ_pos count⟩
      let b0 ← select 0 b ⟨0, by simpa only [Shape.axisSize_zero] using Nat.succ_pos count⟩
      let x0 ← select 0 x ⟨0, by simpa only [Shape.axisSize_zero] using Nat.succ_pos count⟩
      let retained ← mul a0 initial
      let written ← mul b0 x0
      let next ← add retained written
      let aTail ← sliceLeadingAxisRange 1 count (by omega) a
      let bTail ← sliceLeadingAxisRange 1 count (by omega) b
      let xTail ← sliceLeadingAxisRange 1 count (by omega) x
      let tail ← variableReference aTail bTail xTail next
      let head ← reshape (s₂ := [1, state]) next (by simp [Shape.eraseAxis, Shape.size])
      let result ← concatLeadingAxis head tail
      pure (by simpa [Nat.one_add] using result)

end SelectiveScan

/--
Diagonal selective scan with independent coefficients for every token.

All three sequence inputs have shape `[seqLen, state]`. Gradients flow through `A`, `B`, `X`,
and `initial`, including when the coefficients themselves were computed from the input tokens.
-/
def selectiveScanDiagVar {seqLen state : Nat}
    (a b x : RefTy m α [seqLen, state]) (initial : RefTy m α [state]) :
    m (RefTy m α [seqLen, state]) := do
  if let some native :=
      _root_.Runtime.Autograd.Torch.Ops.selectiveScanDiagVarNative? (m := m) (α := α) then
    if let some result ← native a b x initial then return result
  SelectiveScan.variableReference a b x initial

/--
Diagonal scan with coefficient vectors shared across time.

Reverse mode sums the contributions to each shared coefficient over all steps. The native
kernel performs that accumulation directly; the reference route gets it from broadcasting.
-/
def selectiveScanDiag {seqLen state : Nat}
    (a b : RefTy m α [state]) (x : RefTy m α [seqLen, state])
    (initial : RefTy m α [state]) : m (RefTy m α [seqLen, state]) := do
  if let some native :=
      _root_.Runtime.Autograd.Torch.Ops.selectiveScanDiagNative? (m := m) (α := α) then
    if let some result ← native a b x initial then return result
  let aRows ← broadcastTo (s₂ := [seqLen, state]) Shape.BroadcastTo.proof a
  let bRows ← broadcastTo (s₂ := [seqLen, state]) Shape.BroadcastTo.proof b
  SelectiveScan.variableReference aRows bRows x initial

end Runtime.Autograd.Model.F
