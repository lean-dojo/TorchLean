# `NN/Spec`: Specification Layer

This folder is TorchLean's specification layer. It holds the reference definitions of tensors,
operations, layers, models, dynamics, and RL objects that later runtime and verification code point
back to.

The same spec can be instantiated in several scalar worlds:

- `ℝ` for clean mathematical statements;
- `FloatLib.Floats.Formats.Flocq.NF` and its `FP32` specialization for noncomputable
  rounded-real proofs;
- `FloatLib.Floats.ExecFloat.Binary` for CPU software arithmetic at a chosen precision;
- runtime scalar backends where explicit bridges state what is assumed.

Configured precision does not select arbitrary-precision CUDA kernels. See `NN/API/Precision.lean`
for the typed tensor/model entrypoints and their numerical limits.

The practical goal is to avoid a gap between the network we run and the network we reason about:
define the reference behavior once, then make runtime, graph, and verifier layers say how they
connect back to it.

Ordinary model/training code should start from `import NN`. Use `NN.Spec` when a file is
spec-focused and should avoid importing the full public API.

## How To Navigate

- `Core/`
  - `Shape.lean`: type-level tensor shapes, axis utilities, and broadcasting evidence.
  - `Context.lean`: `Context α`, the numeric backend interface for spec code.
  - `Tensor/Core.lean`: proof-facing operations and laws for the canonical `TorchLean.Tensor`.
  - `TensorOps.lean`, `TensorReductionShape.lean`: elementwise ops, reductions, reshapes,
    broadcasts, concat/slice, and axis manipulation.
  - `Complex.lean` and `TensorGrad.lean`: FFT/FNO support and gradient helper specs.
  - `Random.lean`: deterministic `Spec.Random` key and sampling helpers.
- `Layers/`: forward and backward specs for common layers: linear, convolution, attention,
  FlashAttention-style fused attention, normalization, pooling, embeddings, recurrent layers,
  selective scan, dropout, and losses.
- `Autograd/`: spec-level reverse-mode building blocks (`OpSpec`) used by runtime AD wrappers and
  proof files.
- `Module/`: module records that package layer specs with input/output shapes and export metadata.
- `Models/`: model compositions such as MLP, CNN, Transformer, ResNet, ViT, Seq2Seq, UNet, GNN,
  linear/logistic regression, gradient boosted trees, HMM/GMM/PCA, and state-space models.
- `Dynamics/`: pure dynamical-system and state-space recurrence specs (namespace
  `Spec.Dynamics`).
- `Generative/`: diffusion and latent-variable objective specs.
- `Quantization.lean`: the scalar affine quantizer from `NN.Floats` lifted pointwise to tensors.
- `RL/`: Bellman backups, returns, MDPs, Gymnasium-style environment contracts, and GridWorld specs.
- `NN/Examples/`: executable examples that exercise the specs through the public trainer and CLI.

## Terminology

- spec = pure reference definitions in this folder.
- runtime = tape/graph execution, lowering, CUDA paths, and training loops (see `NN/Runtime/*` and
  `NN.Runtime`).
- verification = bound propagation, certificate checking, and artifact replay (see `NN/Verification/*` and
  `NN.Verification`).

## What A Spec Claim Means

A spec definition is the reference object for later layers. Runtime backends, external kernels, and
serialized artifacts connect to it through explicit lowering, checking, or trust-boundary statements.
The usual chain is:

1. define the mathematical behavior here,
2. execute or lower a runtime object elsewhere,
3. state a bridge, checker, test, or theorem connecting the runtime/artifact back to the spec.

Boolean attention masks use the hard-mask meaning: blocked positions contribute exactly zero
softmax numerator. An additive
attention bias is a different operation and must not be used as an approximation to that mask.
Similarly, a real-valued layer spec, an executable `IEEE32Exec` path, and
a CUDA `Float32` kernel are related objects, not interchangeable words.

When adding a new spec, keep the reference behavior small and explicit. Put runtime shortcuts,
foreign-library assumptions, tolerances, and file-format details in the runtime, verification, or
trust-boundary layer that actually owns them.
