/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Models.Mamba
public import NN.Spec.Models.S4

/-!
# Causality of S4/Mamba-style recurrent blocks

This file proves the sequence-causality property expected of state-space sequence models:
appending future tokens cannot change outputs already emitted for a prefix.

We state the theorem at the list-runner level rather than for a particular CUDA kernel. Runtime
implementations may use chunked or parallel selective scan, but they must refine these spec
runners. Combined with `NN.MLTheory.StateSpace.diagonalSelectiveScan_append`, this gives the
proof-facing contract for Mamba/S4-style causal sequence processing.

References:
* Albert Gu and Tri Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
  COLM 2024.
* Tri Dao and Albert Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms
  Through Structured State Space Duality", ICML 2024.
-/

@[expose] public section

namespace NN.MLTheory.StateSpace

open _root_.Spec
open _root_.Models

variable {α : Type} [Context α]
variable {inputDim stateDim outputDim innerDim convWidth : Nat}

/--
Diagonal S4 prefix causality.

Appending future tokens `ys` cannot change the outputs already produced for prefix `xs`.
-/
theorem diagonalS4_runList_append_outputs_prefix
    (m : DiagonalS4Spec α inputDim stateDim outputDim)
    (h0 : Tensor α (.dim stateDim .scalar))
    (xs ys : List (Tensor α (.dim inputDim .scalar))) :
    (m.runList h0 (xs ++ ys)).2.take xs.length = (m.runList h0 xs).2 := by
  induction xs generalizing h0 with
  | nil =>
      simp
  | cons x rest ih =>
      simp [Models.DiagonalS4Spec.runList_cons, ih]

/--
Compact Mamba prefix causality.

If a sequence `xs` has already been processed, appending future tokens `ys` cannot change the
outputs for `xs`.  This is the recurrent-model analogue of causal attention non-anticipation.
-/
theorem compactMamba_runList_append_outputs_prefix
    (m : MambaBlockSpec α inputDim stateDim outputDim)
    (h0 : Tensor α (.dim stateDim .scalar))
    (xs ys : List (Tensor α (.dim inputDim .scalar))) :
    (m.runList h0 (xs ++ ys)).2.take xs.length = (m.runList h0 xs).2 := by
  induction xs generalizing h0 with
  | nil =>
      simp
  | cons x rest ih =>
      simp [Models.MambaBlockSpec.runList_cons, ih]

/--
Full selective Mamba prefix causality for the internal runner.

The internal runner carries a newest-first causal convolution history.  Even with that extra state,
future input tokens only affect future outputs.
-/
theorem selectiveMamba_runListAux_append_outputs_prefix
    (m : SelectiveMambaBlockSpec α inputDim innerDim stateDim outputDim convWidth)
    (h0 : Tensor α (.dim innerDim (.dim stateDim .scalar)))
    (history : List (Tensor α (.dim innerDim .scalar)))
    (xs ys : List (Tensor α (.dim inputDim .scalar))) :
    (m.runListAux h0 history (xs ++ ys)).2.take xs.length =
      (m.runListAux h0 history xs).2 := by
  induction xs generalizing h0 history with
  | nil =>
      simp
  | cons x rest ih =>
      simp [Models.SelectiveMambaBlockSpec.runListAux, ih]

/--
Full selective Mamba prefix causality for the public runner.

This is the user-facing theorem: extending the input stream preserves all previously produced
outputs.
-/
theorem selectiveMamba_runList_append_outputs_prefix
    (m : SelectiveMambaBlockSpec α inputDim innerDim stateDim outputDim convWidth)
    (h0 : Tensor α (.dim innerDim (.dim stateDim .scalar)))
    (xs ys : List (Tensor α (.dim inputDim .scalar))) :
    (m.runList h0 (xs ++ ys)).2.take xs.length = (m.runList h0 xs).2 := by
  simpa [Models.SelectiveMambaBlockSpec.runList] using
    selectiveMamba_runListAux_append_outputs_prefix (m := m) h0 [] xs ys

end NN.MLTheory.StateSpace
