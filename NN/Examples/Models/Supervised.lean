/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Models.Supervised.Mlp
public import NN.Examples.Models.Supervised.LstmRegression

/-!
# Supervised Model Examples

Supervised examples for ordinary input/target training tasks.

This folder is for examples whose main structure is a labeled or paired target:

- `Mlp`: tabular supervised regression on the small UCI Auto MPG CSV path.
- `LstmRegression`: real time-series forecasting on UCI household-power windows.

Sequence architectures can still appear here when the task is supervised forecasting. The
`Sequence` folder is reserved for sequence-model behavior itself: RNN/LSTM smoke tests,
Transformer blocks, GPT-style language modeling, Mamba, and synthetic sequence curricula.
-/

@[expose] public section
