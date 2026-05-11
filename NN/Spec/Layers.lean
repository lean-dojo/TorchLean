/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Activation
public import NN.Spec.Layers.Attention
public import NN.Spec.Layers.Conv
public import NN.Spec.Layers.Dropout
public import NN.Spec.Layers.Embedding
public import NN.Spec.Layers.FlashAttention
public import NN.Spec.Layers.GlobalPooling
public import NN.Spec.Layers.Gnn
public import NN.Spec.Layers.Gru
public import NN.Spec.Layers.Linear
public import NN.Spec.Layers.Loss
public import NN.Spec.Layers.Lstm
public import NN.Spec.Layers.Normalization
public import NN.Spec.Layers.Pooling
public import NN.Spec.Layers.PositionalEncoding
public import NN.Spec.Layers.Rnn
public import NN.Spec.Layers.SelectiveScan
public import NN.Spec.Layers.Utils

/-!
# Spec layers

Umbrella import for layer-level denotational semantics and explicit backward/VJP definitions.

The files under this chapter define what each neural-network layer computes. Higher-level modules,
models, runtime execution, and verification all build on these reference definitions.
-/

@[expose] public section

