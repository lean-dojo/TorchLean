/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Models.Sequence.Rnn
public import NN.Examples.Models.Sequence.Lstm
public import NN.Examples.Models.Sequence.SimpleText
public import NN.Examples.Models.Sequence.Transformer
public import NN.Examples.Models.Sequence.Gpt2
public import NN.Examples.Models.Sequence.Gpt2Saved
public import NN.Examples.Models.Sequence.TextGpt2
public import NN.Examples.Models.Sequence.Mamba
public import NN.Examples.Models.Sequence.GptAdder
public import NN.Examples.Models.Sequence.CharGpt

/-!
# Sequence Model Examples

Runnable sequence-model examples, organized by what each file is meant to teach:

The “main” entrypoints most people should look at first:

- `CharGpt` (`torchlean chargpt`): Karpathy-style char-level GPT on a single text file (Tiny Shakespeare).
  This is the simplest end-to-end path: read text, tokenize by characters, train, sample.
- `Gpt2` (`torchlean gpt2`): byte-level GPT-2-style causal Transformer with a small, local-friendly config.
  Use this when you want to see masked self-attention + LayerNorm + FFN wiring, and a save/reload path
  via `Gpt2Saved`.
- `TextGpt2` (`torchlean text_gpt2`): CUDA-only corpus trainer (byte-level by default, optional GPT-2 BPE).
  This is the “serious” trainer interface for bigger text runs.
- `Mamba` (`torchlean mamba`): compact text walkthrough for the Mamba-style model.

Other sequence examples:

- `Rnn` and `Lstm`: compact real-text recurrent smoke tests over the shared `SimpleText` runner.
- `Transformer`: one-block encoder example for attention/norm/FFN wiring.
- `GptAdder`: synthetic algorithmic curriculum (addition), runnable as `torchlean gpt_adder`.

For supervised time-series forecasting with an LSTM, see
`NN.Examples.Models.Supervised.LstmRegression` (`torchlean lstm_regression`).
-/

@[expose] public section
