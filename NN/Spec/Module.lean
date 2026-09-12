/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Module.Activation
public import NN.Spec.Module.Autoencoder
public import NN.Spec.Module.Conv
public import NN.Spec.Module.DecisionTree
public import NN.Spec.Module.Dropout
public import NN.Spec.Module.Embedding
public import NN.Spec.Module.Flatten
public import NN.Spec.Module.GruModels
public import NN.Spec.Module.Hmm
public import NN.Spec.Module.Linear
public import NN.Spec.Module.LstmModels
public import NN.Spec.Module.Normalization
public import NN.Spec.Module.Pooling
public import NN.Spec.Module.Rnn
public import NN.Spec.Module.RnnModels
public import NN.Spec.Module.Seq2seq
public import NN.Spec.Module.Core

/-!
# Spec modules

This umbrella re-exports shape-indexed modules built from TorchLean's mathematical layer and model
definitions. A `Spec.Module α σ τ` is a pure map from tensors of shape `σ` to tensors of shape
`τ`; `Spec.Module.Chain` composes modules only when adjacent shapes agree.

The `forward` field carries the semantics. Operation names and Python expressions are metadata for
reports and source export, not part of the mathematical definition.
-/

@[expose] public section
