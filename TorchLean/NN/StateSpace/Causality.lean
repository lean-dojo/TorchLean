/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.StateSpace.Scan
public import TorchLean.NN.Models.Mamba.Block
public import TorchLean.NN.Models.S4

/-!
# Causality of S4/Mamba-style recurrent blocks

A state-space sequence model is causal: appending tokens to a sequence leaves the outputs already
emitted for its prefix unchanged. It holds for the array runners of the diagonal S4 block, the
compact Mamba block and the full selective Mamba block, each by the scan lemma
`scanArray_append_outputs_take`.

A runtime may compute the selective scan in chunks or in parallel, and it must then refine these
spec runners. `TorchLean.StateSpace.scanStates_append` shows how consecutive chunks of each
fiber of `Tensor.selectiveScan` compose. Native kernels and reordered floating-point arithmetic
are outside these statements.

References:
* Albert Gu and Tri Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
  COLM 2024.
* Tri Dao and Albert Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms
  Through Structured State Space Duality", ICML 2024.
-/

@[expose] public section

namespace TorchLean

open Mamba
open StateSpace (scanArray_append_outputs_take)

variable {α : Type} [Storage α] [Context α]
variable {inputDim stateDim outputDim innerDim convWidth : Nat}

/--
Diagonal S4 prefix causality.

Appending future tokens `ys` cannot change the outputs already produced for prefix `xs`.
-/
theorem S4.runArray_causal
    (m : S4 α inputDim stateDim outputDim)
    (h0 : Tensor α [stateDim])
    (xs ys : Array (Tensor α [inputDim])) :
    (m.runArray h0 (xs ++ ys)).2.take xs.size = (m.runArray h0 xs).2 :=
  scanArray_append_outputs_take m.step h0 xs ys

/--
Compact Mamba prefix causality.

Appending future tokens `ys` to a processed sequence `xs` cannot change the outputs for `xs`, the
recurrent counterpart of a causal attention mask.
-/
theorem Mamba.CompactBlock.runArray_causal
    (m : CompactBlock α inputDim stateDim outputDim)
    (h0 : Tensor α [stateDim])
    (xs ys : Array (Tensor α [inputDim])) :
    (m.runArray h0 (xs ++ ys)).2.take xs.size = (m.runArray h0 xs).2 :=
  scanArray_append_outputs_take m.step h0 xs ys

/--
Full selective Mamba prefix causality for the internal runner.

The internal runner also carries a newest-first history of projected tokens for the causal
convolution, and with that extra state future input tokens affect only future outputs.
-/
theorem Mamba.SelectiveBlock.runArrayWithHistory_causal
    (m : SelectiveBlock α inputDim innerDim stateDim outputDim convWidth)
    (h0 : Tensor α [innerDim, stateDim])
    (history : Array (Tensor α [innerDim]))
    (xs ys : Array (Tensor α [inputDim])) :
    (m.runArrayWithHistory h0 history (xs ++ ys)).2.take xs.size =
      (m.runArrayWithHistory h0 history xs).2 :=
  scanArray_append_outputs_take m.stepWithConvolutionHistory (h0, history) xs ys

/--
Full selective Mamba prefix causality for the public runner.

Extending the input stream keeps every output already produced for the prefix. The public runner
is the internal one started from an empty convolution history.
-/
theorem Mamba.SelectiveBlock.runArray_causal
    (m : SelectiveBlock α inputDim innerDim stateDim outputDim convWidth)
    (h0 : Tensor α [innerDim, stateDim])
    (xs ys : Array (Tensor α [inputDim])) :
    (m.runArray h0 (xs ++ ys)).2.take xs.size = (m.runArray h0 xs).2 := by
  simpa [runArray] using
    runArrayWithHistory_causal (m := m) h0 #[] xs ys

end TorchLean
