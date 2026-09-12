# `NN.Spec.Models`

This directory contains model definitions built out of the primitives in:

- `NN/Spec/Core/*` (tensors, shapes, numeric backends),
- `NN/Spec/Layers/*` (layer forward/backward specs),
- `NN/Spec/Module/*` (optional wrappers for composition/export metadata).

Most files here are compact mathematical baselines. They spell out a complete forward pass, and
many include an explicit backward or VJP that can be studied independently of the runtime autograd
engine. Executable architecture constructors, including ResNet, ViT, and U-Net families, live in
`NN/API/Models` and are built from the same typed layer operations.

## How To Navigate

- Neural models: `Mlp.lean`, `Cnn.lean`, `Transformer.lean`, `Seq2seq.lean`, `Mamba.lean`,
  `S4.lean`, and the generative-model files.
- Classical baselines: `LinearRegression.lean`, `LogisticRegression.lean`, `Svm.lean`,
  `Knn.lean`, `NaiveBayes.lean`, `Pca.lean`, `RandomForest.lean`,
  `GradientBoostedTrees.lean`, `Gmm.lean`, and `Hmm.lean`.

## File Index

- `Mlp.lean`: small MLP wiring example (linear layer plus activation).
- `Cnn.lean`: a convolutional baseline with an explicit backward pass.
- `Transformer.lean`: Transformer encoder and decoder definitions, including an encoder backward.
- `Seq2seq.lean`: encoder-decoder RNN baseline with a differentiable one-hot training path.
- `Mamba.lean` and `S4.lean`: state-space sequence models.
- `Gnn.lean`: a compact graph-convolution baseline.
- `Autoencoder.lean`, `Vae.lean`, `VqVae.lean`, and `Gan.lean`: generative-model specifications.
- `Hopfield.lean`: Hopfield states, energy, and dynamics.
- `Gmm.lean` and `Hmm.lean`: mixture and hidden-Markov models with EM-style training.
- `GradientBoostedTrees.lean` and `RandomForest.lean`: tree ensembles.
- `NaiveBayes.lean`, `Knn.lean`, `LinearRegression.lean`, `LogisticRegression.lean`, `Svm.lean`,
  and `Pca.lean`: classical statistical and machine-learning baselines.

## Adding A New Model

When adding a new model, aim to make it clear how someone else can:

1. understand the shapes and dataflow,
2. reuse it as a building block,
3. run it on an executable backend (`Float` or IEEE32Exec),
4. optionally train it (explicit backward/VJP) without re-deriving gradients.

Practical checklist:

1. Pick a file name and namespace
   - Add `NN/Spec/Models/<your_model>.lean`.
   - Use `namespace Models` and open `Spec`/`Tensor` as the other files do.

2. Choose the input convention (and be explicit)
   - Describe semantic axes in the declaration instead of encoding a layout in its name.
   - Use a leading shape for independently mapped axes such as batch or ensemble dimensions.
   - Represent every remaining spatial extent by a `Tensor Nat [d]` when the operation is
     rank-polymorphic. Its type records the number of spatial axes.

3. Define a parameter record
   - Use `structure <Model>Spec ... where ...` and store weights/biases as `Tensor α <shape>`.
   - Prefer reusing layer-spec parameter records when they exist (`ConvSpec`, `LinearSpec`, etc.).

4. Implement `forward`
   - Name it `<Model>Spec.forward`.
   - Keep the forward close to how it would look in PyTorch (same operator order and shapes).
   - If you need shape rewrites, keep them local and comment why they are needed.

5. If you want the model to be trainable, add an explicit backward/VJP
   - Name it `<Model>Spec.backward` or `<Model>Spec.<loss>_grad_*`.
   - Reuse existing backward specs (`linear_backward_spec`, `convBackwardSpec`, attention/encoder backprops, etc.).
   - If you recompute intermediates, say so once (and keep the recomputation structurally aligned with the forward).

6. Hook it into the public spec entrypoint
   - Add an import in the relevant focused umbrella, such as `NN/Spec/Models.lean`.
   - If it should be part of the complete spec import, make sure `NN/Spec.lean` reaches that
     umbrella.
   - If it should be part of ordinary model code, also expose a clean root spelling through
     `NN.lean` or the public model API.

7. Add evidence in the right layer
   - If the claim is mathematical, add a theorem under `NN/Proofs/Models`.
   - If the model crosses a runtime or artifact boundary, add a focused runtime check near that
     boundary.
   - If the code is only a usage demonstration, put it under `NN/Examples`.

8. Build the files you touched
   - `lake build NN.Spec.Models.<your_model>`
   - If you updated the spec entrypoint: `lake build NN.Spec`

## Common Pitfalls

- Comparisons (`>` / `max` / `argmax`) require decidability: you may need
  `[DecidableRel ((· > ·) : α → α → Prop)]` on the scalar backend.
- Output shapes follow the same arithmetic formulas as PyTorch layer definitions. If a conv should
  preserve `H×W`, add the corresponding equality proof that rewrites the type.
- Avoid duplicating derivative logic in two places. Prefer one authoritative backward/VJP and call
  it from training wrappers (as in `Svm.lean`).
