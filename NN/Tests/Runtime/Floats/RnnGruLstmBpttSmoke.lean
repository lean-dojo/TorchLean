/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Gru
public import NN.Spec.Layers.Lstm
public import NN.Spec.Layers.Rnn

/-!
# RnnGruLstmBpttSmoke

 Smoke tests for RNN, GRU, and LSTM BPTT specifications over floats. -/

@[expose] public section

open Spec
open Tensor

namespace Tests
namespace Floats
namespace BPTT

def assertNonzero (name : String) (x : Float) : IO Unit := do
  if Float.abs x ≤ 1e-12 then
    throw <| IO.userError s!"{name} expected nonzero, got {x}"

def run : IO Unit := do
  -- RNN BPTT smoke: gradients are nonzero for a nontrivial sequence.
  let seqLen := 2
  let inputSize := 2
  let hiddenSize := 2

  let rnn : Spec.RNNSpec Float inputSize hiddenSize :=
    { weights :=
        Tensor.dim (fun _ => Tensor.dim (fun _ => Tensor.scalar 0.1))
      bias := Tensor.dim (fun _ => Tensor.scalar 0.01) }

  let inputs : Tensor Float (.dim seqLen (.dim inputSize .scalar)) :=
    Tensor.dim (fun t =>
      Tensor.dim (fun i =>
        Tensor.scalar (0.2 * Float.ofNat (t.val + 1) + 0.1 * Float.ofNat (i.val + 1))))

  let initial_hidden : Tensor Float (.dim hiddenSize .scalar) :=
    Tensor.dim (fun i => Tensor.scalar (0.05 * Float.ofNat (i.val + 1)))

  let hiddens := Spec.rnnSequenceSpec (α := Float) rnn inputs initial_hidden
  let grad_hiddens : Tensor Float (.dim seqLen (.dim hiddenSize .scalar)) :=
    Tensor.dim (fun _ => Tensor.dim (fun _ => Tensor.scalar 1.0))

  let (dW, _db, _dInputs, _dH0) :=
    Spec.rnnSequenceBackwardSpec (α := Float) rnn inputs initial_hidden hiddens grad_hiddens

  assertNonzero "rnn.dW[0,0]" (Float.abs (getAtOrZero dW [0, 0]))

  -- GRU BPTT smoke.
  let gru : Spec.GRUSpec Float inputSize hiddenSize :=
    { reset_weights := Tensor.dim (fun _ => Tensor.dim (fun _ => Tensor.scalar 0.1))
      reset_bias := Tensor.dim (fun _ => Tensor.scalar 0.0)
      update_weights := Tensor.dim (fun _ => Tensor.dim (fun _ => Tensor.scalar 0.2))
      update_bias := Tensor.dim (fun _ => Tensor.scalar 0.0)
      new_weights := Tensor.dim (fun _ => Tensor.dim (fun _ => Tensor.scalar 0.3))
      new_bias := Tensor.dim (fun _ => Tensor.scalar 0.0) }

  let (hidden_states, reset_gates, update_gates, new_candidates, _reset_hiddens) :=
    Spec.gruExtractIntermediateValues (α := Float) (seqLen := seqLen) (inputSize := inputSize)
      (hiddenSize := hiddenSize)
      gru inputs initial_hidden

  let grad_outputs : Tensor Float (.dim seqLen (.dim hiddenSize .scalar)) :=
    Tensor.dim (fun _ => Tensor.dim (fun _ => Tensor.scalar 1.0))

  let (dResetW, _dResetB, _dUpdateW, _dUpdateB, _dNewW, _dNewB, _dInputs, _dH0) :=
    Spec.gruSequenceBackwardFullSpec (α := Float) (seqLen := seqLen) (inputSize := inputSize)
      (hiddenSize := hiddenSize)
      gru inputs hidden_states grad_outputs reset_gates update_gates new_candidates initial_hidden

  assertNonzero "gru.dResetW[0,0]" (Float.abs (getAtOrZero dResetW [0, 0]))

  -- LSTM BPTT smoke.
  let lstm : Spec.LSTMSpec Float inputSize hiddenSize :=
    { forget_weights := Tensor.dim (fun _ => Tensor.dim (fun _ => Tensor.scalar 0.1))
      forget_bias := Tensor.dim (fun _ => Tensor.scalar 0.0)
      input_weights := Tensor.dim (fun _ => Tensor.dim (fun _ => Tensor.scalar 0.2))
      input_bias := Tensor.dim (fun _ => Tensor.scalar 0.0)
      candidate_weights := Tensor.dim (fun _ => Tensor.dim (fun _ => Tensor.scalar 0.3))
      candidate_bias := Tensor.dim (fun _ => Tensor.scalar 0.0)
      output_weights := Tensor.dim (fun _ => Tensor.dim (fun _ => Tensor.scalar 0.4))
      output_bias := Tensor.dim (fun _ => Tensor.scalar 0.0) }

  let initial_state : Spec.LSTMState Float hiddenSize :=
    { hidden := initial_hidden, cell := Tensor.dim (fun _ => Tensor.scalar 0.0) }

  let (dWf, _dbf, _dWi, _dbi, _dWc, _dbc, _dWo, _dbo, _dInputs, _dInit) :=
    Spec.lstmSequenceBackwardSpec (α := Float) (seqLen := seqLen) (inputSize := inputSize)
      (hiddenSize := hiddenSize)
      lstm inputs initial_state grad_hiddens

  assertNonzero "lstm.dWf[0,0]" (Float.abs (getAtOrZero dWf [0, 0]))

  IO.println "bptt_smoke: OK"

end BPTT
end Floats
end Tests
/-!
RNN/GRU/LSTM BPTT smoke tests (floats).

This file exists to catch regressions in sequence-model execution and the parts of the runtime
needed for backprop-through-time style training loops.
-/
