# `NN.Spec.Layers`

This directory contains TorchLean's layer definitions: common neural-network building blocks written
as pure functions on spec tensors and parameter records.

The emphasis is a clear reference definition. Runtime code, graph lowering, CUDA kernels, and
verification passes should be able to point back here when they need the mathematical meaning of an
operation. For layers used by reverse-mode training, the spec layer also records derivative or VJP
rules so the backward meaning is visible.

Most models in `NN/Spec/Models/*` compose these layer definitions directly or package them as
shape-indexed `Spec.Module`s and `Spec.Module.Chain`s.

Files:

- `Activation.lean`: scalar activation formulas under `Activation.Math`, tensor activation wrappers,
  axis-indexed softmax/log-softmax and their VJPs, plus final-axis primitives used by row-oriented
  backends.
- `Linear.lean`: fully connected layer spec ($y=Wx+b$) and gradients.
- `Attention.lean`: scaled dot product attention and multihead attention with hard-mask semantics
  (blocked weights are exactly zero, and a fully blocked row evaluates to the zero vector), with
  VJPs.
- `FlashAttention.lean`: FlashAttention style tiling metadata/specs tied back to the same attention
  semantics.
- `Conv.lean`: rank-polymorphic convolution and transposed-convolution specs with explicit
  backward rules, including grouped and dilated convolution in both the block-diagonal dense and
  the PyTorch packed weight layouts.
- `Pooling.lean` and `Pooling/Spatial.lean`: max/avg pooling, padded pooling, adaptive pooling, and
  smooth max pooling surrogates for arbitrary spatial rank, including backward/JVP rules.
- `Normalization.lean`, `Normalization/Core.lean`, `Normalization/BatchNorm.lean`: LayerNorm,
  RMSNorm, and BatchNorm style utilities with explicit backward specs.
- `Embedding.lean`: one-hot embeddings (`oneHot @ W`) and the corresponding VJP.
- `PositionalEncoding.lean`: learnable/sinusoidal positional encodings and RoPE style rotations.
- `Dropout.lean`: deterministic inference and mask-driven training dropout specs.
- `Loss.lean`: common scalar losses and their derivatives.
- `Gnn.lean`: a compact GCN-style graph layer and backward rules.
- `Rnn.lean`, `Lstm.lean`, `Gru.lean`: recurrent layers and BPTT-style backwards.
- `SelectiveScan.lean`: affine scan primitives used by S4/Mamba style state space models.

The underlying tensor primitives (maps, matmul, reshape, broadcasting) live under `NN/Spec/Core/*`.

This folder defines each operation's meaning. Typed graph nodes and ATen implementations should
document which spec operation they implement or approximate.
