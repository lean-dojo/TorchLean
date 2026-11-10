# `TorchLean/NN/Models`

TorchLean's models, of two kinds. The spec models are built from the tensors, shapes and scalar
backends of `TorchLean/Tensor` and the layer forward and backward specs of
`TorchLean/Tensor/Layers`. Most are compact mathematical baselines: each spells out a complete
forward pass, and many add a hand-written backward or VJP that doesn't depend on the runtime
autograd engine. The architecture constructors (`nn.cnn`, `nn.resnet`, `nn.vit`, `nn.unet` and
the rest) build executable models from the same typed layer operations. Training loops, CUDA
kernels and exporters should agree with the reference definitions here rather than replace them.

The subfolders `CNN/`, `CausalTransformer/`, `Diffusion/`, `FNO/`, `GAN/`, `Hopfield/`, `Mamba/`,
`Recurrent/`, `Transformer/`, `VAE/` and `VQVAE/` hold the parts and proofs of the model with the
same name.

## File Index

Spec models:

- `MLP.lean`: a two-layer MLP (`Linear → ReLU → Linear`) with its hand-written backward pass.
- `CNN/TwoBlock.lean`: a two-block convolutional network over any number of spatial axes, with
  forward and reverse-mode specs.
- `Transformer.lean`: Transformer encoder and decoder definitions, including an encoder backward.
  `Transformer/` builds the post-norm encoder block, the decoder core, the feed-forward sublayer
  and residual attention as proof-carrying graphs, with their VJP theorems.
- `Seq2Seq.lean`: encoder-decoder RNN baseline with a differentiable one-hot training path.
- `Mamba/Block.lean` and `S4.lean`: state-space sequence models built on `DiagonalSSM`.
- `GCN.lean`: a two-layer graph convolutional network with graph-level mean pooling and a
  hand-written backward.
- `Autoencoder.lean`, `VAE.lean`, `VQVAE.lean`, and `GAN.lean`: generative-model specifications.
  `VAE/Elbo.lean`, `VQVAE/Loss.lean` and `GAN/Loss.lean` prove facts about them, and
  `VQVAE/Autograd.lean` trains a VQ-VAE.
- `Diffusion.lean` and `Diffusion/`: VP schedules, the forward noising process and its Gaussian
  law, DDPM and DDIM samplers, the probability-flow ODE, training losses, sampler theorems, and the
  image DDIM API with its contracts.
- `Hopfield.lean`: Hopfield states, energy, and dynamics. `Hopfield/` holds the single-step energy
  lemma `energy_updateAt_le`, energy along trajectories, and the cyclic-sweep convergence argument.
- `GMM.lean` and `HMM.lean`: mixture and hidden-Markov models with EM-style training.
- `GradientBoostedTrees.lean` and `RandomForest.lean`: tree ensembles.
- `NaiveBayes.lean`, `KNN.lean`, `LinearRegression.lean`, `LogisticRegression.lean`, `SVM.lean`,
  and `PCA.lean`: classical statistical and machine-learning baselines.

Architecture constructors:

- `CNN.lean`, `ResNet.lean`, `ViT.lean` and `UNet.lean`: the `nn.cnn`, `nn.resnet`, `nn.vit` and
  `nn.unet` builders, configured by records such as `nn.ResNet.Config` and `nn.UNet.Config`.
- `Classifier.lean`: `nn.Sequential.classifier`, a linear head on any `nn.Sequential` feature
  model.
- `CausalTransformer.lean` and `CausalTransformer/LanguageModel.lean`: `nn.CausalTransformer.Config`
  for a GPT-style causal Transformer, and `nn.CausalTransformer.indexed`, a complete model over
  bounded token indices.
- `Mamba.lean` and `Mamba/Recurrence.lean`: `nn.Mamba.languageModel`, a selective Mamba-1
  language model, and the Mamba-1 recurrence.
- `Recurrent.lean` and `Recurrent/`: `nn.Recurrent.rnn`, `nn.Recurrent.gru` and
  `nn.Recurrent.lstm` from an `nn.Recurrent.Config`, the Elman cell VJP, recurrent models with
  explicit state, and their correctness under the pure interpreter.
- `FNO.lean` and `FNO/Block.lean`: `nn.fno` and its Fourier layer, over any spatial rank.
- `KAN.lean`: Kolmogorov-Arnold network layers.
- `Generative.lean`: `nn.Generative.autoencoder`, `nn.Generative.generator` and
  `nn.Generative.discriminator`.
- `Diffusion/NoisePredictor.lean`: `nn.NoisePredictor.plain` and `nn.NoisePredictor.residual`.
- `PPO.lean`: `nn.PPO.actor` and `nn.PPO.critic`.
- `MaskedPatchReconstructor.lean`: self-supervised model constructors.

## Adding A New Model

A model added here should let someone else:

1. follow its shapes and dataflow,
2. reuse it as a building block,
3. run it on an executable backend (`Float`, or binary32 as `FloatLib.Floats.ExecFloat.Binary 8 23`
   in software arithmetic on the CPU),
4. if it's trainable, train it through its hand-written backward or VJP without re-deriving
   gradients.

Practical checklist:

1. Pick a file name and namespace
   - Add `TorchLean/NN/Models/<your_model>.lean`.
   - Declare it in `namespace TorchLean` under the model's name (`GCN`, `SVM.Linear`), or extend
     the model's existing namespace.

2. Choose the input convention
   - Describe the semantic axes in the declaration, and keep layouts out of its name.
   - Use a leading shape for independently mapped axes such as batch or ensemble dimensions.
   - Represent every remaining spatial extent by a `Tensor Nat [d]` when the operation is
     rank-polymorphic, so its type records the number of spatial axes.

3. Define a parameter record
   - Use `structure <Model> ... where ...` and store weights/biases as `Tensor α <shape>`.
   - Reuse layer-spec parameter records (`Conv`, `Linear` and the like) where they exist.

4. Implement `forward`
   - Name it `<Model>.forward`, as `MLP.forward` and `Transformer.forward` are.
   - Keep it close to the PyTorch version, with the same operator order and shapes.
   - Keep shape rewrites local, with a comment saying why each is needed.

5. For a trainable model, add a hand-written backward or VJP
   - Name it `<Model>.backward`, or after the loss it differentiates.
   - Reuse the existing backward specs: `Linear.inputDeriv`, `Linear.weightsDeriv`,
     `Conv.backward`, `MultiHeadAttention.backward`, `Transformer.EncoderStack.backward` and the
     like.
   - If it recomputes intermediates, say so once and keep the recomputation structurally aligned
     with the forward.

6. Expose it where it is used
   - Code that uses the model imports `TorchLean.NN.Models.<your_model>` directly.

7. Add evidence in the right layer
   - If the claim is mathematical, add a theorem under `TorchLean/NN`.
   - If the model crosses a runtime or artifact boundary, add a focused runtime check near that
     boundary.
   - If the code is only a usage demonstration, put it under `TorchLeanExamples`.

8. Build the files you touched
   - `scripts/lake.sh build TorchLean.NN.Models.<your_model>`
   - This wrapper holds the checkout lock described in `docs/CONTRIBUTING.md`.

## Common Pitfalls

- Comparisons (`>` / `max` / `argmax`) require decidability: you may need
  `[DecidableRel ((· > ·) : α → α → Prop)]` on the scalar backend.
- Output shapes follow the same arithmetic formulas as PyTorch layer definitions. If a conv should
  preserve `H×W`, add the corresponding equality proof that rewrites the type.
- Write each derivative once: keep one authoritative backward or VJP and call it from the training
  wrappers, as `SVM.lean` does.
- Apply the shape-generic losses (`Tensor.meanSquaredError` and those of
  `TorchLean/Tensor/Layers/Loss.lean`) to the model's predictions, as `SVM.lean`,
  `LinearRegression.lean` and `GradientBoostedTrees.lean` do, and don't give each model its own loss
  catalogue. Their means divide once by the count (`Context.divNat`), and multiplying by a rounded
  reciprocal can give a different floating-point result.
