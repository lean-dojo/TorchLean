# `NN.Spec.Models`

This directory contains model definitions built out of the primitives in:

- `NN/Spec/Core/*` (tensors, shapes, numeric backends),
- `NN/Spec/Layers/*` (layer forward/backward specs),
- `NN/Spec/Module/*` (optional wrappers for composition/export metadata).

Most files here are compact baselines: they spell out a complete forward pass, and many also include
an explicit backward/VJP so you can train with gradient descent without relying on a separate
autograd engine.

## How To Navigate

- If you want a neural model: start with `mlp.lean`, `cnn.lean`, `transformer.lean`, `resnet.lean`,
  `vit.lean`, `seq2seq.lean`, or `unet.lean`.
- If you want classical baselines: `linear_regression.lean`, `logistic_regression.lean`, `svm.lean`,
  `knn.lean`, `naive_bayes.lean`, `pca.lean`, `random_forest.lean`, `gradient_boosted_trees.lean`,
  `gmm.lean`, `hmm.lean`.
- If you want shared math helpers: `common_helpers.lean`.

## File Index

- `common_helpers.lean`: shared linear-algebra utilities used by multiple classical models.
- `mlp.lean`: small MLP wiring example (linear layer plus activation).
- `cnn.lean`: a small CNN baseline with an explicit backward pass.
- `transformer.lean`: transformer encoder stack (+ a small decoder layer), and explicit backward for the encoder.
- `vit.lean`: ViT patch embedding + transformer encoder + classifier head, with explicit backward.
- `seq2seq.lean`: encoder decoder RNN baseline with a differentiable one hot training path and gradients.
- `resnet.lean`: ResNet style blocks (basic blocks include backward specs; bottleneck blocks are forward only).
- `unet.lean`: a 2 level U-Net with explicit backward (conv/pool/conv transpose/concat/ReLU).
- `gnn.lean`: a small GCN style model (graph convolution baseline).
- `hopfield.lean`: Hopfield network definitions (states, energy, dynamics) with references.
- `gmm.lean`: Gaussian mixture model (forward, VJP, and EM training).
- `hmm.lean`: HMM likelihood and Baum-Welch (EM) training.
- `gradient_boosted_trees.lean`: tree ensembles (CART style training plus a boosting step).
- `random_forest.lean`: random forest built on top of the tree code.
- `naive_bayes.lean`: multinomial Naive Bayes baseline.
- `knn.lean`: kNN classification/regression baseline.
- `linear_regression.lean`: linear regression baseline (plus a few feature helpers).
- `logistic_regression.lean`: logistic regression baseline (gradient descent on NLL).
- `svm.lean`: linear SVM baseline with explicit objective gradients and a small trainer.
- `pca.lean`: PCA baseline (data centering, covariance, projection).

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
   - For vision we use the single-image convention `(C,H,W)` (no batch axis).
   - For sequences we typically use `(T,D)` (token/time axis first).
   - If you need batching, prefer adding an outer `.dim N` and mapping your single-example forward.

3. Define a parameter record
   - Use `structure <Model>Spec ... where ...` and store weights/biases as `Tensor α <shape>`.
   - Prefer reusing layer-spec parameter records when they exist (`Conv2DSpec`, `LinearSpec`, etc.).

4. Implement `forward`
   - Name it `<Model>Spec.forward`.
   - Keep the forward close to how it would look in PyTorch (same operator order and shapes).
   - If you need shape rewrites, keep them local and comment why they are needed.

5. If you want the model to be trainable, add an explicit backward/VJP
   - Name it `<Model>Spec.backward` or `<Model>Spec.<loss>_grad_*`.
   - Reuse existing backward specs (`linear_backward_spec`, `conv2d_backward_spec`, attention/encoder backprops, etc.).
   - If you recompute intermediates, say so once (and keep the recomputation structurally aligned with the forward).

6. Hook it into the public spec entrypoint
   - Add an import in the relevant focused umbrella, usually `NN/Spec/Models.lean`.
   - If it should be part of the whole spec doorway, make sure `NN/Entrypoint/Spec.lean` reaches that
     umbrella so users can access it through `import NN.Entrypoint.Spec`, `import NN.Library`, or
     `import NN`.

7. Add a quick smoke test (recommended)
   - Put a small compile only / small eval test in `NN/Tests/Runtime/Floats/ModelsSmoke.lean`.
   - Keep it deterministic and small (a few steps of training, or a single forward).

8. Build the files you touched
   - `lake build NN.Spec.Models.<your_model>`
   - If you updated the spec entrypoint: `lake build NN.Entrypoint.Spec`

## Common Pitfalls

- Comparisons (`>` / `max` / `argmax`) require decidability: you may need
  `[DecidableRel ((· > ·) : α → α → Prop)]` on the scalar backend.
- Output shapes follow the same arithmetic formulas as PyTorch layer definitions. If you "expect"
  a conv to preserve `H×W`, you usually need the corresponding equality proof to rewrite the type.
- Avoid duplicating derivative logic in two places. Prefer one authoritative backward/VJP and call
  it from training wrappers (as in `svm.lean`).
