/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Models.Sequence.Rnn
public import NN.Examples.Models.Sequence.Lstm
public import NN.Examples.Models.Sequence.Transformer
public import NN.Examples.Models.Sequence.Gpt2
public import NN.Examples.Models.Sequence.Gpt2Saved
public import NN.Examples.Models.Sequence.TextGpt2
public import NN.Examples.Models.Sequence.Mamba
public import NN.Examples.Models.Sequence.GptAdder
public import NN.Examples.Models.Sequence.CharGpt

/-!
# Sequence Model Examples

Runnable sequence-model examples, organized by workflow.

Main entrypoints:

- `CharGpt` (`torchlean chargpt`): Karpathy-style char-level GPT on a single text file (Tiny
  Shakespeare). This is the teaching path for character tokenization, with separate smoke and
  lecture presets.
- `Gpt2` (`torchlean gpt2`): byte-level GPT-2-style causal Transformer with a small, local-friendly
  config. Use this when you want to see masked self-attention + LayerNorm + FFN wiring, and a
  save/reload path via `Gpt2Saved`.
- `TextGpt2` (`torchlean text_gpt2`): CUDA-only bounded corpus trainer with byte-level or GPT-2 BPE
  tokenization. Its BPE path projects observed token IDs into a local output vocabulary.
- `Mamba` (`torchlean mamba`): compact text walkthrough for the Mamba-style model.

Other sequence examples:

- `Rnn` and `Lstm`: compact real-text recurrent training checks over the shared corpus-data
  boundary.
- `Transformer`: one-block encoder example for attention/norm/FFN wiring.
- `GptAdder`: synthetic algorithmic curriculum (addition), runnable as `torchlean gpt_adder`.

For supervised time-series forecasting with an LSTM, see
`NN.Examples.Models.Supervised.LstmRegression` (`torchlean lstm_regression`).
-/

@[expose] public section
