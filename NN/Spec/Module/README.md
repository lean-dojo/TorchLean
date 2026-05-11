# `NN.Spec.Module`

This directory is TorchLean's module wrapper layer for the spec codebase.

Most math lives in `NN/Spec/Layers/*` (pure layer specs) and `NN/Spec/Core/*` (tensors/shapes).
Here we package selected layer specs into a uniform interface that records input/output shapes:

`ModSpec.NNModuleSpec α inShape outShape`

That lets us:

- compose blocks with `ModSpec.SpecChain` (a shape safe `nn.Sequential` style chain), and
- attach lightweight metadata (`kind`, `export_func`) for export and pretty printing tooling.

Files:

- `spec_module.lean`: defines `NNModuleSpec` and `SpecChain` composition/evaluation helpers.
- `activation.lean`: wrappers for common activations (ReLU/sigmoid/tanh/softmax).
- `linear.lean`: linear layer wrappers and small sequence classifier helpers.
- `conv2d.lean`: conv2d wrapper (single image `(C,H,W)` convention).
- `conv_transpose2d.lean`: transpose conv wrapper (single image `(C,H,W)` convention).
- `pooling.lean`: max/avg pooling wrappers (single image `(C,H,W)` convention).
- `normalization.lean`: layer norm wrapper.
- `attention.lean`: scaled dot-product self-attention wrapper.
- `embedding.lean`: one hot embedding wrapper (purely numeric variant).
- `positional_encoding.lean`: learnable positional encoding wrapper.
- `rnn.lean`: RNN/LSTM/GRU wrappers (sequence forward with canonical zero initial state).
- `dropout.lean`: deterministic dropout wrappers (inference-scale and explicit-mask variants).
- `global_pooling.lean`: global avg/max pooling wrappers (flattened `(C)` outputs).
- `gnn.lean`: a GCN style graph layer wrapper.
- `decision_tree.lean`: a small non-neural baseline datatype and evaluator used by tree models.

Note: The `kind`/`export_func` fields are metadata; they are not part of the mathematical meaning.
