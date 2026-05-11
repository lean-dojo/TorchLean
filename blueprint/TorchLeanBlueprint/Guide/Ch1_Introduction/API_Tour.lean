import VersoManual
import VersoBlueprint

open Verso.Genre Manual

#doc (Manual) "A Tour of the API" =>
%%%
tag := "api_tour"
%%%

TorchLean's public API is easiest to understand by following one model through the system. At
first, it looks like a small PyTorch-style library embedded in Lean: we write tensors, layers,
losses, optimizers, and training loops. The difference is what TorchLean keeps around as the model
moves through the stack.

A tensor carries its shape. A model carries an input and output contract. A parameter payload is
ordinary data. A lowered graph has operation tags and node ids. A certificate is checked against the
graph rather than treated as a comment beside it.

A typical workflow looks like this:

1. instantiate tensors and a model,
2. build parameters and run a training loop,
3. inspect the run through logs or graphs,
4. import or export a named artifact when Python belongs in the workflow,
5. state a claim about the resulting model and check the artifact that supports it.

The [public API facade](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Public.lean) is the default entry point for this path. The
lower-layer pages become relevant when a chapter asks a more precise question about tensors,
runtime execution, import/export, floating point, or verification.

Here is the rough map:

- When the user writes a tensor literal, TorchLean keeps the scalar type and shape.
- When the user builds a model, TorchLean keeps the architecture and parameter shapes.
- When the user runs training, TorchLean keeps parameter state, optimizer state, random state, and logs.
- When the user lowers a graph, TorchLean keeps operation tags, node ids, parent links, and payloads.
- When the user checks a certificate, TorchLean keeps the checker predicate and scalar semantics.

# Tensors And Models

The first visible difference from ordinary Python model code is that tensor shapes appear in the
Lean type. A value of type `Tensor Float (shape![32, 1])` is not interchangeable with a value of
type `Tensor Float (shape![32])`; a reshape, squeeze, or different loss must be named explicitly.

```
import NN

open NN.Tensor
open NN.API

def logits : Tensor Float (shape![32, 1]) :=
  tensorND! [32, 1] (List.replicate 32 0.0)

def labels : Tensor Float (shape![32]) :=
  tensorND! [32] (List.replicate 32 0.0)

-- A loss expecting matching shapes cannot silently reinterpret `labels`.
```

That strictness is a design choice. Broadcasting is useful when it is intended, but a training loss
with predictions shaped `[batch, 1]` and targets shaped `[batch]` is often a modeling bug. TorchLean
does not try to guess whether the right convention is one-hot encoding, a singleton-dimension
squeeze, a different loss, or a different model head.

The same shape discipline appears at the model level. A small classifier states its input and output
shapes before it ever runs:

```
def classifier : nn.M (nn.Sequential (Shape.Vec 16) (Shape.Vec 4)) :=
  nn.sequential![
    nn.linear 16 32 (pfx := Shape.scalar),
    nn.gelu,
    nn.linear 32 4 (pfx := Shape.scalar)
  ]
```

For tensor constructors and literals, see the [tensor API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Tensor/API.lean). For the model
builder surface used above, start with the [public API facade](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Public.lean).

# Building And Training

In TorchLean, model structure and parameters are separate values. Building a model chooses an
initial parameter payload, but it does not make those parameters hidden fields of a mutable object.

```
def task (seed : Nat) :=
  train.classificationOneHot (nn.build seed classifier)
```

That small separation is what later chapters rely on. A training step can be read as an explicit
state transition:

$$`\mathrm{step} :
  (\theta,\mathrm{optState},\mathrm{rng},\mathrm{mode},x,y)
  \longmapsto
  (\theta',\mathrm{optState}',\mathrm{rng}',\mathrm{log})`

The state may be large, but it is no longer implicit. Parameters, optimizer moments, random seeds,
training/evaluation mode, and metric reports are all ordinary data that can be passed, saved,
loaded, displayed, or mentioned in a theorem statement.

The training surface deliberately looks familiar:

- `train.fitDataset` and `train.fitLoaderWith` cover full fitting,
- `train.stepper` and `train.step` support manual loops familiar from PyTorch,
- `optim.sgd`, `optim.adam`, and `optim.adamw` construct optimizer configurations.

Those names live behind the same public facade, with implementation details in the
[training runtime](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/Train.lean) and the
[runtime optimizer definitions](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/TorchLean/Optim.lean).

# Inspecting Runs

A runnable model should not become opaque after it trains. TorchLean keeps several inspection
surfaces close to the training path:

- structured training logs, persisted as ordinary JSON, are defined by the
  [training log API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Training/Log.lean);
- per-step logger hooks for training loops are defined by the
  [training logger API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/Train/Logging.lean);
- widgets can display tensors, logs, execution traces, and IR graphs through the
  [widget entrypoint](https://github.com/lean-dojo/TorchLean/blob/main/NN/Entrypoint/Widgets.lean).

The purpose is not only convenience. Logs and graphs are also audit artifacts. A loss curve says
which run was performed, while a lowered graph says which operations a verifier or compiler is
about to interpret.

# Graphs, Export, And Import

The public model API is comfortable for writing models. Verification and interop need a more
explicit object: a graph whose nodes carry operation tags, together with a payload store. That is the role of
[NN.IR.Graph](https://github.com/lean-dojo/TorchLean/blob/main/NN/IR/Graph.lean). Its denotation is the semantic reference used by graph
checkers, widgets, compiler bridges, and verifier passes.

The PyTorch boundary follows the same discipline. TorchLean does not claim to import arbitrary
Python object graphs. Instead, the supported path is a small auditable contract:

$$`\text{known architecture family}
\;+\;
\text{named tensor payload}
\;+\;
\text{shape checks}
\quad\leadsto\quad
\text{TorchLean parameters}`

That contract is implemented by the [PyTorch export](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/PyTorch/Export.lean) and
[PyTorch import](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/PyTorch/Import.lean) APIs, with runnable examples in the
[PyTorch interop tutorial](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Interop/PyTorch.lean). Python can remain the right tool
for data preparation or large-scale training, while Lean receives a payload with names, shapes, and
model family assumptions it can check.

# Verification Claims

After lowering, a verifier no longer has to guess which computation is being analyzed. It receives
a graph, a payload, an input region, and a scalar semantics. A robustness claim, for example, has
the form:

$$`\forall x\in B,\quad
\operatorname{margin}(\operatorname{denote}(g,\theta,x)) > 0`

A robustness certificate in TorchLean is a concrete claim about a graph, a parameter payload, an
input region, a scalar semantics, and a checker result. The [verification entrypoint](https://github.com/lean-dojo/TorchLean/blob/main/NN/Entrypoint/Verification.lean)
collects the public verification surface, while the [CROWN graph API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/CROWN/Graph.lean)
shows the graph objects used by the bound propagation chapters.

This is the central API pattern:

$$`\text{runnable model}
\;\longrightarrow\;
\text{explicit graph and payload}
\;\longrightarrow\;
\text{checked artifact}
\;\longrightarrow\;
\text{semantic claim}`

# Failure Modes The API Makes Visible

The API choices above are motivated by concrete failure modes.

Shape mismatch is the simplest one. A target tensor with shape `[batch]` should not silently become a
`[batch, 1]` tensor just because a loss function can broadcast. The type mismatch forces the
training script to state the intended convention.

Hidden runtime state is another. BatchNorm buffers, dropout mode, random seeds, optimizer moments,
tokenizer tables, and cache layouts affect the computation. TorchLean's functional style keeps
these objects in the data path instead of leaving them implicit inside a module instance.

Floating-point semantics also need names. A real valued specification, the proof side `FP32` model,
the executable `IEEE32Exec` model, and native CUDA kernels are related but not identical. The
[floating-point entrypoint](https://github.com/lean-dojo/TorchLean/blob/main/NN/Entrypoint/Floats.lean) exists so a theorem, a verifier claim,
and a runtime run do not quietly use three different meanings of "float32."

Fast kernels are boundaries too. For attention, the mathematical contract is ordinary scaled
dot product attention, while fused FlashAttention style implementations are optimized paths that
must be related back to that contract. The relevant proof statements live near the attention and GPU
chapters; the design principle is simple: a fast path should preserve a slow, readable meaning, or
else it is a different model.

# Reading The Rest Of The Guide

This tour should make the later chapters easier to navigate. *Building Models* explains the public
surface in detail. *Runtime and Interop* shows training, logs, execution modes, and PyTorch
round-trips. *Semantics and Graphs* fixes the graph denotation. *Verification* explains how checked
artifacts become claims about the model.

# References

- Szegedy et al., ["Intriguing properties of neural networks"](https://arxiv.org/abs/1312.6199),
  ICLR 2014.
- Zhang et al., ["Efficient Neural Network Robustness Certification with General Activation
  Functions"](https://arxiv.org/abs/1811.00866), NeurIPS 2018.
- Xu et al., ["Automatic Perturbation Analysis for Scalable Certified Robustness and
  Beyond"](https://arxiv.org/abs/2002.12920), NeurIPS 2020.
- Goldberg, ["What Every Computer Scientist Should Know About Floating-Point Arithmetic"](https://doi.org/10.1145/103162.103163),
  ACM Computing Surveys 1991.
- IEEE 754-2019, *Standard for Floating-Point Arithmetic*.
