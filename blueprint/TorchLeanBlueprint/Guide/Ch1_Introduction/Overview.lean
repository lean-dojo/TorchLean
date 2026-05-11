import VersoManual
import VersoBlueprint

open Verso.Genre Manual

#doc (Manual) "What TorchLean Is" =>
%%%
tag := "overview"
%%%

TorchLean is a Lean 4 framework for neural networks whose computations can be run, inspected, and
connected to formal claims. The guiding idea is simple: the model we execute, the graph we inspect,
and the object we mention in a theorem should not be three unrelated artifacts.

This matters because modern ML systems are not just model definitions. They include parameter
files, tokenizers, graph exports, runtime modes, fused kernels, floating point conventions, verifier
certificates, and scientific claims. TorchLean gives these pieces a common place to meet: a typed
Lean development with runnable examples, graph semantics, explicit numerical models, and checkable
artifacts.

PyTorch remains the everyday training ecosystem for many projects. TorchLean focuses on the parts
of an ML workflow that carry formal meaning: the model, the graph, the numerical semantics, the
certificate, and the assumptions around external code. As model code, exported graphs, certificates,
and GPU kernels become easier to generate, the hard question is no longer only whether an artifact
runs. It is whether the artifact still means what we think it means.

The broad user import is [NN root API](https://github.com/lean-dojo/TorchLean/blob/main/NN.lean). Narrower entry points include
[NN.Entrypoint.API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Entrypoint/API.lean),
[NN.Entrypoint.GraphSpec](https://github.com/lean-dojo/TorchLean/blob/main/NN/Entrypoint/GraphSpec.lean),
[NN.Entrypoint.IR](https://github.com/lean-dojo/TorchLean/blob/main/NN/Entrypoint/IR.lean), and
[NN.Entrypoint.Verification](https://github.com/lean-dojo/TorchLean/blob/main/NN/Entrypoint/Verification.lean).

# What TorchLean Covers

At a high level, TorchLean provides three layers.

First, it gives a familiar ML surface: tensors, layers, datasets, optimizers, losses, autograd,
logging, import/export, optional CUDA-backed execution, and examples ranging from small MLPs to
CNNs, ResNets, ViTs, GPT-style models, Mamba-style sequence models, diffusion, operator learning,
reinforcement learning, and scientific ML.

Second, it gives these programs a semantic core. Tensor shapes can appear in types. Architectures
can be lowered to a shared op-tagged graph IR. Scalar meanings can be real valued, executable
float32, proof-side float32, interval-like, or verifier-specific. GraphSpec lets an architecture be
described once and read both as executable structure and as symbolic graph structure.

Third, it provides ways to check claims about the resulting artifacts: graph well-formedness,
shape inference, IBP/CROWN-style bounds, JSON certificate replay, robustness margins, PINN
residuals, ODE enclosures, 3D projection certificates, attention-mask contracts, and floating-point
bridge statements.

These layers do not all have the same proof status. Some are executable infrastructure, some are
checked artifacts, and some are Lean theorems. The guide keeps those roles separate because they
support different kinds of claims.

# The Design Choice

Ordinary ML systems are often optimized around fast experimentation. That is the right default for
many projects, but it can leave important facts implicit: tensor shapes, parameter layouts,
training mode, random state, scalar semantics, and the exact graph a verifier checked. TorchLean
makes a different tradeoff. We accept more typed structure at the boundary so later tools can say
precisely what they consumed and what they checked.

For example, a loss that expects logits and one-hot labels of shape `[batch, classes]` should not
silently receive labels of shape `[batch]`. A matrix multiply whose right-hand dimension is `64`
should not be applied to activations whose last dimension is `128`. In a dynamic runtime, those
errors may appear only after a particular batch reaches a particular kernel, or worse, they may be
hidden by broadcasting. In TorchLean's typed tensor APIs, the shape is part of the tensor type, so
the mismatch is rejected before the example is treated as a runnable model. If the intended design
really is sparse labels, a squeeze, or a reshape, we name that design choice explicitly.

The same discipline applies to semantic claims. A statement such as "this classifier is robust on
this input box" becomes useful only when it says which graph, parameters, scalar interpretation, and
checker result support the claim. Lean gives us a place to write that statement as a checked object,
not as a comment beside a script.

# Three Views Of One Model

At the top level, TorchLean provides a facade with PyTorch vocabulary over a shared internal model.
User code mostly starts from `import NN` and `open NN.API`, then works with familiar concepts:
layers, datasets, optimizers, losses, and training loops.

Under the hood, the same model appears in three representations:

- *Spec layer*: the mathematical meaning of tensors, layers, losses, and model structure.
- *Graph IR*: the op-tagged DAG shared by runtime tooling, widgets, export, and verification.
- *Runtime layer*: eager or compiled execution, autograd, optimizers, logging, and optional CUDA.

This alignment has a simple goal: the model you run, the model you inspect, and the model you state
theorems about are connected by explicit translations rather than by informal correspondence.

# The Object We Keep In View

The object we keep returning to is a typed model. Informally, it has this shape:

$$`\mathrm{Model}_{\theta} :
   \mathrm{Tensor}(\alpha,s_{\mathrm{in}})
   \longrightarrow
   \mathrm{Tensor}(\alpha,s_{\mathrm{out}})`

The runtime may choose `Float`, executable `IEEE32Exec`, or CUDA-backed float32 buffers. A proof may
instantiate the same architecture over real numbers, interval domains, or proof side float32
models. The architecture statement should not change just because the execution substrate
changes.

That gives the whole project a repeated theorem pattern:

$$`\begin{aligned}
&H(M,R)\\
&\Longrightarrow\;
  \forall x,\;
  \mathrm{denote}(R)(x)
  \in
  \mathrm{safeSet}(\mathrm{denote}(M)(x)).
\end{aligned}`

For an exact theorem, `safeSet` is a singleton. For a finite precision or verifier theorem, it may
be an interval, box, affine enclosure, or certified output region.

Different chapters instantiate that pattern in different ways. Autograd statements relate a reverse
pass to the derivative of a denotation. Runtime approximation statements relate executable values to
real specifications. CROWN and certificate checkers establish bound properties of an IR graph. CUDA
chapters name the native boundary when the implementation lives outside Lean.

# Fast When Needed, Explicit When It Matters

TorchLean separates fast execution from proof, but it does not treat them as unrelated worlds. For
prototyping, examples can use the host runtime or optional CUDA-backed float32 paths. For guarantees,
we move to graph denotations, Float32 models, certificate checkers, and Lean theorems. The boundary
is visible: some parts are proved in Lean, and some parts are external systems that must be named
and checked around.

A typical runtime-only bug is a checkpoint whose parameter names load successfully in a script while
one weight has been transposed to match a different convention. The model may still run if the
surrounding dimensions happen to agree, but a verifier might then certify a graph-payload pair that
is not the deployed computation. TorchLean's design pushes parameter payloads, graph lowering, and
shape checks into explicit data so that this kind of agreement can be inspected and, where the
library has the theorem support, proved.

# How The Book Is Organized

The rest of the guide follows the path a model takes through the project:

- The introduction establishes the motivation and the shared mental model.
- Building Models shows the public runnable path.
- Runtime and Interop explains execution, autograd, checkpoints, and PyTorch-facing boundaries.
- Semantics and Graphs fixes the graph and scalar meanings used by later tools.
- Verification and Certificates explains proof obligations, checker outputs, and remaining
  assumptions.
- Examples, Tools, and Conclusion tours the model zoo, widgets, command line workflows, and final
  guarantee structure.
