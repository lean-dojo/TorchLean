import VersoManual
import VersoBlueprint

open Verso.Genre Manual

#doc (Manual) "Autograd Proofs" =>
%%%
tag := "autograd-proofs"
%%%

Most ML users learn gradients operationally: run the forward pass, call
`loss.backward()`, inspect a few numbers, and move on. That workflow is powerful, and PyTorch and
JAX made it practical at enormous scale. TorchLean keeps the same programmer intuition and puts an
explicit theorem underneath it.

The difference is not that TorchLean "has autograd" while PyTorch and JAX do not. They do, and they
do it very well. The difference is that ordinary framework use usually relies on an implementation
of reverse mode AD, its derivative registrations, compiler rewrites, and native kernels.
TorchLean's proof tree asks a more explicit question:

> For the supported graph fragment, does the reverse pass compute the adjoint of the derivative of
> the forward denotation?

That is the proof architecture in [NN/Proofs/Autograd](https://github.com/lean-dojo/TorchLean/tree/main/NN/Proofs/Autograd/). We built it in layers so the
auditor can stop at the boundary they care about: algebraic tape soundness, Frechet derivative
statements, proofs for particular operators, theorem surfaces for model blocks, and algebra for
training steps.

The proof pipeline is:

```
local op derivative
  -> local JVP/VJP adjointness
  -> graph backprop correctness
  -> Frechet derivative theorem
  -> model block theorem
  -> training step algebra
```

That ordering is important. A theorem about a Transformer sublayer is not proved from scratch; it is
assembled from local derivative rules, graph composition, and the analytic bridge to the derivative
of the denotation.

# From Runtime Confidence To Theorem Obligations

In PyTorch, the default runtime model is roughly:

1. the eager engine records operations into a dynamic tape;
2. each operation has a backward rule, either built in or registered by extension code;
3. the engine traverses the tape in reverse and accumulates gradients;
4. tests, numerical checks, and framework maintenance give confidence that the result is right.

JAX moves the same idea into a functional transformation pipeline: `grad`, `vjp`, `jvp`, `jit`, and
lowering passes cooperate to produce differentiated and compiled programs. See the PyTorch autograd
overview at https://pytorch.org/docs/stable/autograd.html and JAX's autodiff guide at
https://jax.readthedocs.io/en/latest/automatic-differentiation.html for the user facing version of
that workflow.

TorchLean does not claim those systems are careless. We use the same mental model because it is
the right one: local derivative rules plus reverse accumulation. What changes is that the runtime
confidence is split into named theorem obligations.

- A registered backward rule becomes a local JVP/VJP or Fréchet derivative lemma for the op.
- A tape reversal becomes global tape soundness by induction over the graph.
- A scalar loss training step becomes an explicit theorem about the loss seed and parameter update
  algebra.
- A model block such as attention or an RNN cell becomes a packaged theorem with stated hypotheses.

The resulting style is slower to author, but easier to audit. We can say exactly which part is a
Lean theorem and which part is a runtime, compiler, or finite precision agreement statement.

# Tape Soundness: The Algebraic Core

The central algebraic file is
[NN.Proofs.Autograd.Tape.Algebra.Soundness API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/Tape/Algebra/Soundness.lean).
It defines the small tape language used by the rest of the autograd proofs, together with the local
proof data carried by each node.

The important objects are:

- `TList`: heterogeneous tensor context indexed by a list of shapes.
- `Idx`: pointer into a context, carrying the needed proof data.
- `NodeData`: forward, JVP, and VJP data for one local operation.
- `Node`: a node plus the local inner product soundness law.
- `GraphData`: executable snoc list graph data.
- `Graph`: graph whose nodes can share earlier values and carry local proof obligations.
- `Graph.backprop_correct`: global dot product soundness theorem.

The theorem `Graph.backprop_correct` is the first big hinge. Informally, it says that reverse
accumulation is the adjoint of forward sensitivity:

$$`\langle \operatorname{jvp}_G(x, dx), seed\rangle = \langle dx, \operatorname{backprop}_G(x, seed)\rangle`

This is the algebraic essence of reverse mode AD. A forward sensitivity `dx` pushed through the
graph and then paired with an output cotangent gives the same scalar as pairing `dx` with the
cotangent produced by reverse accumulation.

This dot product statement is useful because it is the common language between implementation and
analysis. Runtime engineers recognize it as the VJP law. Mathematicians recognize it as the adjoint
property. Lean can prove it compositionally: each node gives the local adjoint law, and graph
soundness follows by induction over `Graph`.

This is also where the comparison to PyTorch is most direct. PyTorch's engine performs a reverse
walk over a dynamic graph and accumulates cotangents into inputs. TorchLean's algebraic graph does
the same conceptual work, but the graph object carries enough structure for Lean to prove that the
accumulation is sound for every input context in the supported fragment.

# From Dot Products To Frechet Derivatives

The algebraic theorem is intentionally not the last word. A dot product VJP law is useful, but the
guide should also say what function is being differentiated. The
[Frechet derivative bridge](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/Tape/Core/FDeriv.lean) provides that link.

That file vectorizes shaped tensor contexts into Euclidean spaces and connects three views:

- shaped tensors: `Tensor Real s`, `TList Real Gamma`;
- flat Euclidean vectors: `CtxVec Gamma`, `flattenCtx`, `unflattenCtx`;
- analytic derivatives: `HasFDerivAt`, `fderiv`, and `ContinuousLinearMap.adjoint`.

The main theorem is `Graph.backpropVec_eq_adjoint_fderiv`. In plain English:

> If every node in the graph has the stated Frechet derivative, then graph backprop equals the
> adjoint of the Frechet derivative of graph evaluation.

There is also a pointwise version, `Graph.backpropVec_eq_adjoint_fderiv_at`, for hypotheses that
only hold at a particular input. That distinction matters for neural networks. ReLU, normalization,
division, logarithms, and square roots all have domain or nondifferentiability issues. TorchLean
states those conditions explicitly instead of using a blanket "autograd works" slogan. The theorem can demand exactly the
local smoothness or nonzero hypotheses needed by the graph being differentiated.

A useful mental example is
$`forward(x) = softmax(Wx + b)`. The scalar loss supplies an output cotangent
$`seed = dL/dforward`, and the reverse pass returns the input and parameter cotangents
$`dL/dx`, $`dL/dW`, and $`dL/db`.

In a framework, the registered kernels and the engine are expected to compose into the right answer.
In this proof layer, the statement is that the backpropagated cotangent is the adjoint action of the
derivative of the forward denotation. That theorem is the bridge from an executed reverse pass to a
mathematical gradient object.

# Operator Specs: Softmax And LogSoftmax

Softmax and log-softmax are good examples because they look familiar but carry real analytic content.
The two derivative APIs are:

- [NN.Proofs.Autograd.FDeriv.Softmax API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/FDeriv/Softmax.lean)
- [NN.Proofs.Autograd.FDeriv.LogSoftmax API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/FDeriv/LogSoftmax.lean)

The softmax API defines `softmaxVec`, `softmaxDerivCLM`, and `softmaxJvp`. The theorem
`softmaxJvp_eq_deriv` identifies the implemented JVP formula with the derivative formula, and
`hasFDerivAt_softmaxVec` states the Frechet derivative of the vector softmax. The theorem
`inner_softmaxJvp_comm` packages the self adjoint structure of the softmax Jacobian, which is the
reason the VJP can reuse the same formula shape.

The log-softmax API follows the same discipline with `logSoftmaxVec`, `logSoftmaxJvp`, and
`logSoftmaxVjp`. The theorem `logSoftmaxJvp_eq_deriv` gives the derivative formula, while
`inner_logSoftmaxJvp_vjp` states the adjoint relationship between the JVP and the VJP:

For softmax, if

$$`s_i=\frac{e^{x_i}}{\sum_j e^{x_j}},`

then the directional derivative has the familiar form

$$`D\,\operatorname{softmax}(x)[dx]_i
=
s_i\left(dx_i-\sum_j s_j dx_j\right).`

For log-softmax, the formula is even cleaner:

$$`D\,\log\operatorname{softmax}(x)[dx]_i
=
dx_i-\sum_j \operatorname{softmax}(x)_j\,dx_j.`

The theorem proves that the formula used by the reverse rule is the adjoint of the derivative of the
forward function. This is where a framework bug or a mistake in a custom op can otherwise appear:
the program might execute successfully while returning a subtly wrong gradient.

Their comments cite the PyTorch API docs for naming alignment, not as proof sources:

- [PyTorch softmax docs](https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html)
- [PyTorch log-softmax docs](https://pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html)

That separation is deliberate. Documentation tells us what users expect the op to mean; Lean proves
the derivative law for TorchLean's mathematical definition.

# MLP And MSE Gradients: A Concrete Scalar Loss Story

The [MLP/MSE derivative API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/FDeriv/MlpMse.lean)
turns the abstract autograd theorem into a familiar training example: a small MLP followed by mean
squared error.

The definitions are intentionally close to what a reader would write on a whiteboard:

- `affineMat` for an affine layer;
- `mlpVecMat` for a two layer MLP with a hidden nonlinearity;
- `mse` and `mseGrad` for the scalar loss;
- gradient lemmas for `W2`, `b2`, `b1`, `x`, and `W1`.

The important theorem pattern is:

$$`\operatorname{backpropGradient}
=
\left(D\,\operatorname{loss}\right)^{\!*}(1)`

Equivalently, for parameters `θ` and a scalar loss `L(θ)=\ell(f_\theta(x),t)`, the gradient
statement has the shape

$$`\nabla_\theta L(\theta)
=
\left(D_\theta f_\theta(x)\right)^{\!*}\nabla_y \ell(y,t).`

For the last layer, this is clean. For the hidden ReLU layer, the theorem carries hypotheses such
as "the value before activation is nonzero" at the coordinates being differentiated. That is exactly
the kind of condition a runtime usually leaves implicit. PyTorch chooses a subgradient
convention at zero; TorchLean's real analysis statement names the differentiability condition
instead.

The claim should be read precisely: a proof over the reals of the MLP/MSE gradient is not a proof
that an arbitrary float32 training run follows the real gradient exactly. It proves the ideal
mathematical gradient. The finite precision bridge belongs to runtime approximation and to the
[runtime approximation proof API](https://github.com/lean-dojo/TorchLean/tree/main/NN/Proofs/RuntimeApprox/).

# Model Surfaces: Attention, Transformers, And Recurrent Cells

The autograd APIs for model blocks are proof surfaces rather than a claim that every modern model is
fully verified end to end. We built them to show how the algebra scales to the shapes users care
about while keeping boundaries explicit.

Representative theorem surfaces:

- [NN.Proofs.Autograd.Tape.Ops.Attention.ScaledDotProduct API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/Tape/Ops/Attention/ScaledDotProduct.lean)
- [NN.Proofs.Autograd.Tape.Ops.Attention.MultiHeadSelfAttention API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/Tape/Ops/Attention/MultiHeadSelfAttention.lean)
- [NN.Proofs.Autograd.Tape.Ops.Transformer.PostNorm API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/Tape/Ops/Transformer/PostNorm.lean)
- [NN.Proofs.Autograd.Tape.Ops.Transformer.ResidualAttention API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/Tape/Ops/Transformer/ResidualAttention.lean)
- [NN.Proofs.Autograd.Tape.Ops.Transformer.FeedForward API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/Tape/Ops/Transformer/FeedForward.lean)
- [NN.Proofs.Autograd.Tape.Ops.Recurrent.ElmanCell API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/Tape/Ops/Recurrent/ElmanCell.lean)

For attention, the named theorem
`backprop_eq_adjoint_fderiv_scaledDotProduct` states the desired attention theorem directly: the
graph reverse pass for scaled dot product attention agrees with the adjoint derivative of the
forward attention map. Multi head attention and residual attention then package that structure at a
wider interface.

For Transformer post-norm blocks, the post-norm API contains several theorem layers:

- `mhaPostNorm_backpropVec_eq_adjoint_fderiv_at` for residual MHA followed by LayerNorm;
- `seqFfnPostNorm_backpropVec_eq_adjoint_fderiv_at` for residual feed forward followed by LayerNorm;
- `postNorm_backpropVec_eq_adjoint_fderiv_at` for the common post-norm boundary;
- `twoSublayerPostNormBlock_hasFDerivAt` for the analytic composition of two post-norm sublayers;
- named interfaces for residual attention and residual feed-forward post-norm variants.

The recurrent file is intentionally modest. `elmanCell_backpropVec_eq_adjoint_fderiv` proves the
reverse mode theorem for one Elman RNN cell, and `elmanTwoStep_hasFDerivAt` shows the
shape of a short unrolled composition. Full BPTT over an arbitrary sequence length is the next
induction over the unroll. The guide states that boundary explicitly so the current theorem scope is
clear.

# Training Step Algebra

Backprop correctness is about gradients. Training correctness also needs a clean account of how
those gradients are seeded and consumed. That is the role of
[NN.Proofs.Autograd.Training.StepAlgebra API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/Training/StepAlgebra.lean).

There are two small but important pieces.

First, `Graph.scalarLoss_grad_correct` specializes the global graph theorem to scalar losses. The
seed is the scalar cotangent `1`, represented by `seedScalarLoss`. This is the formal version of
$`loss.backward()` seeding $`d loss / d loss = 1`.

Second, `step` defines the algebra behind a simple optimizer update:

$$`\theta_{t+1} = \theta_t - \eta\,\nabla_\theta L(\theta_t)`

The theorem `step_cons` says the head tensor of the parameter list is updated by exactly that
formula, and `step_nil` handles the empty parameter list. These are small theorems, but they keep
the training loop from becoming an opaque execution artifact. A realistic optimizer will add momentum,
Adam statistics, clipping, or weight decay; this file gives the simple algebraic core that those
extensions can refine.

# Autograd Proof Tree

For readers trying to audit an autograd claim, we recommend reading from the bottom up:

1. Start with the [tape algebra soundness API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/Tape/Algebra/Soundness.lean)
   and find `Graph.backprop_correct`.
2. Move to the [Fréchet derivative tape API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/Tape/Core/FDeriv.lean) and find
   `Graph.backpropVec_eq_adjoint_fderiv`.
3. Read a local derivative file such as
   the [softmax derivative API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/FDeriv/Softmax.lean) or
   the [log-softmax derivative API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/FDeriv/LogSoftmax.lean).
4. Read a file for a model block, such as
   the [Transformer post-norm API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/Tape/Ops/Transformer/PostNorm.lean).
5. Finish with the [training step algebra API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/Training/StepAlgebra.lean)
   if the claim is about training rather than only differentiation.

The pattern should feel familiar if you know PyTorch internals or JAX transformations: local rules,
graph traversal, cotangent accumulation, and scalar loss seeding. The proof contribution is that
TorchLean turns those engineering moves into named Lean statements instead of leaving them as a
large opaque execution layer.

# Scope

The autograd proofs are over the mathematical semantics stated by their declarations. They do not, by
themselves, prove all of the following:

- that every TorchLean runtime path uses only proved nodes;
- that CUDA kernels implement those nodes bit for bit;
- that float32 arithmetic equals real arithmetic;
- that all graphs exported from PyTorch or JAX fall inside the proved fragment;
- that every architecture in the model zoo has a full training theorem.

Those are separate bridges. Some live in compiled graph correctness, some in CUDA contracts, and
some in runtime approximation. This modular structure keeps the theorem boundaries explicit. Proved
claims should have theorem names that say what is proved. Claims that cross into runtime execution,
finite precision, or external kernels should keep that boundary visible.
