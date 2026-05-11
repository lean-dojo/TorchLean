/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Models.Autoencoder
public import NN.Spec.Models.Cnn
public import NN.Spec.Models.CommonHelpers
public import NN.Spec.Models.Gan
public import NN.Spec.Models.Gmm
public import NN.Spec.Models.Gnn
public import NN.Spec.Models.GradientBoostedTrees
public import NN.Spec.Models.Hmm
public import NN.Spec.Models.Hopfield
public import NN.Spec.Models.Knn
public import NN.Spec.Models.LinearRegression
public import NN.Spec.Models.LogisticRegression
public import NN.Spec.Models.Mamba
public import NN.Spec.Models.Mlp
public import NN.Spec.Models.NaiveBayes
public import NN.Spec.Models.Pca
public import NN.Spec.Models.RandomForest
public import NN.Spec.Models.Resnet
public import NN.Spec.Models.S4
public import NN.Spec.Models.Seq2seq
public import NN.Spec.Models.Svm
public import NN.Spec.Models.Transformer
public import NN.Spec.Models.Unet
public import NN.Spec.Models.Vae
public import NN.Spec.Models.Vit
public import NN.Spec.Models.VqVae

/-!
# Spec models

Umbrella import for end-to-end model specifications.

The model files compose the layer specs into reusable architectures and classical ML baselines.
They are still reference definitions: training loops, CUDA kernels, and exporters should agree with
these semantics rather than replacing them.
-/

@[expose] public section

