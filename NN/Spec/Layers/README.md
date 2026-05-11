# `NN.Spec.Layers`

This directory contains TorchLean's layer definitions: common neural network building blocks
written as pure functions on `Spec.Tensor` (and parameter records).

The emphasis here is on a clear reference definition, with explicit gradient rules for layers used
by reverse-mode training:

- each layer file defines a forward spec (what the layer computes),
- layers used by reverse-mode execution also define a derivative or VJP spec (what gradients mean).

Most models in `NN/Spec/Models/*` are built by composing these layer specs, sometimes through the
`NN/Spec/Module/*` wrappers that provide the `NNModuleSpec` interface for `SpecChain`.

Files:

- `Activation.lean`: scalar activation formulas under `Activation.Math`, tensor activation wrappers,
  real last-axis softmax/log-softmax, and VJP specs.
- `Linear.lean`: fully connected layer spec (`y = W x + b`) and gradients.
- `Attention.lean`: scaled dot product attention and multihead attention with hard mask
  semantics and VJPs.
- `FlashAttention.lean`: FlashAttention style tiling metadata/specs tied back to the same attention
  semantics.
- `Conv.lean`: 1D/2D convolution and transposed convolution specs plus explicit backward rules.
- `Pooling.lean`: max/avg pooling, padded pooling, adaptive pooling, and smooth max pooling
  surrogates, including backward/JVP rules.
- `GlobalPooling.lean`: global avg/max pooling and backward rules.
- `Normalization.lean`: LayerNorm and BatchNorm style utilities with explicit backward specs.
- `Embedding.lean`: one-hot embeddings (`oneHot @ W`) and the corresponding VJP.
- `PositionalEncoding.lean`: learnable/sinusoidal positional encodings and RoPE style rotations.
- `Dropout.lean`: deterministic inference and mask-driven training dropout specs.
- `Loss.lean`: common scalar losses and their derivatives.
- `Gnn.lean`: a compact GCN-style graph layer and backward rules.
- `Rnn.lean`, `Lstm.lean`, `Gru.lean`: recurrent layers and BPTT-style backwards.
- `SelectiveScan.lean`: affine scan primitives used by S4/Mamba style state space models.
- `Utils.lean`: shared image/tensor utilities used by convolution and pooling layers.

If you're looking for the underlying tensor primitives (maps, matmul, reshape, broadcasting),
those live under `NN/Spec/Core/*`.
