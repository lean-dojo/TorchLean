/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Module.Activation
public import NN.Spec.Module.Attention
public import NN.Spec.Module.Autoencoder
public import NN.Spec.Module.Conv
public import NN.Spec.Module.DecisionTree
public import NN.Spec.Module.Dropout
public import NN.Spec.Module.Embedding
public import NN.Spec.Module.Flatten
public import NN.Spec.Module.GlobalPooling
public import NN.Spec.Module.Gmm
public import NN.Spec.Module.Gnn
public import NN.Spec.Module.GradientBoostedTrees
public import NN.Spec.Module.GruModels
public import NN.Spec.Module.Hmm
public import NN.Spec.Module.Linear
public import NN.Spec.Module.LinearRegression
public import NN.Spec.Module.LogisticRegression
public import NN.Spec.Module.LstmModels
public import NN.Spec.Module.Normalization
public import NN.Spec.Module.Pca
public import NN.Spec.Module.Pooling
public import NN.Spec.Module.PositionalEncoding
public import NN.Spec.Module.Resnet
public import NN.Spec.Module.Rnn
public import NN.Spec.Module.RnnModels
public import NN.Spec.Module.Seq2seq
public import NN.Spec.Module.SpecModule
public import NN.Spec.Module.Svm

/-!
# Spec modules

Umbrella import for PyTorch-style `NNModuleSpec` wrappers around pure layer/model specs.

These wrappers add a uniform `forward` interface and lightweight export metadata, while preserving
the underlying spec definitions as the source of mathematical meaning.
-/

@[expose] public section

