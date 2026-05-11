import VersoManual
import VersoBlueprint

open Verso.Genre Manual

#doc (Manual) "TorchLean vs. PyTorch" =>
%%%
tag := "torchlean_vs_pytorch"
%%%

PyTorch and TorchLean should not be compared as if they were trying to solve the same problem.
PyTorch is the right default for broad, high-performance ML engineering. TorchLean asks a narrower
question: what extra structure do we need when a model must be run, inspected, lowered to a graph,
imported or exported, and connected to a formal claim?

# A Fair Comparison

The short version is:

- PyTorch optimizes for breadth, speed, ecosystem integration, and deployment practice.
- TorchLean optimizes for semantic access: typed shapes, explicit parameters, graph denotations,
  numerical meanings, and checker-friendly artifacts.
- A realistic workflow can use both systems.

For the main design dimensions:

- Main goal: PyTorch is a production ML engineering system. TorchLean is proof-aware ML
  infrastructure.
- Tensor shapes: PyTorch treats shapes dynamically. TorchLean often carries shapes in Lean types.
- Parameters: PyTorch modules own mutable state. TorchLean keeps architecture and payload as
  explicit values.
- Autograd: PyTorch provides a mature, broad engine. TorchLean exposes supported gradients through
  definitions that can be related to Lean semantics.
- Graphs: PyTorch graph tools support capture, compilation, and deployment. TorchLean uses a shared
  IR with a Lean denotation.
- Floating point: PyTorch follows backend behavior. TorchLean names real, float32, executable
  IEEE-754, and native-runtime layers separately.

# Modules And Parameters

In PyTorch, a model is usually an `nn.Module`. The module owns its parameters and buffers, and
calling `model(x)` runs `forward` using that internal state. This is the right ergonomic choice for
many training scripts.

```
# PyTorch
model = torch.nn.Sequential(
    torch.nn.Linear(10, 32),
    torch.nn.GELU(),
    torch.nn.Linear(32, 5),
)
out = model(x)
```

TorchLean keeps the authoring style familiar, but separates the model description from the parameter
payload that will be executed, saved, lowered, or verified.

```
-- TorchLean
import NN

open NN.Tensor
open NN.API

def model : nn.M (nn.Sequential (Shape.Vec 10) (Shape.Vec 5)) :=
  nn.sequential![
    nn.linear 10 32 (pfx := Shape.scalar),
    nn.gelu,
    nn.linear 32 5 (pfx := Shape.scalar)
  ]

def built := nn.build 2026 model
```

Informally, the forward pass is a function of both the architecture and the parameters:

$$`\operatorname{forward}(\operatorname{architecture},\theta,x)=y`

That is the first major difference. In PyTorch, `state_dict()` exposes the parameters when needed.
In TorchLean, the parameter bundle is already part of the ordinary data path. This makes it easier
for graph lowering, checkpoint exchange, and theorem statements to refer to the same weights.

# Shapes And Types

PyTorch tensors are dynamically shaped. That flexibility is one of PyTorch's strengths: a script can
build tensors from files, batch them in many ways, and dispatch to highly optimized kernels at
runtime. The cost is that many shape mistakes are discovered only when a particular execution path
hits a mismatched operation, or worse, when broadcasting performs a valid but unintended operation.

TorchLean uses Lean's dependent types for the core tensor APIs. A tensor carries both a scalar type
and a shape:

$$`\operatorname{Tensor}(\alpha,s)`

This does not mean every possible data problem is solved statically. Files can still be malformed,
an imported payload can still be rejected, and dynamic loaders can still fail. The difference is
that once a tensor has entered the typed core, many dimensional contracts are checked before the
runtime kernel is reached.

That tradeoff is intentional. TorchLean gives up some of Python's dynamism in exchange for a
stronger statement about the computations that do elaborate.

# Training And Autograd

A standard PyTorch loop mutates optimizer state and accumulates gradients in tensor fields:

```
for step in range(steps):
    opt.zero_grad()
    pred = model(x)
    loss = mse(pred, y)
    loss.backward()
    opt.step()
```

TorchLean keeps the same conceptual rhythm, but the training state is explicit. Gradients are
returned by the differentiation machinery, optimizer state is passed through the step, and logs are
values that can be rendered or persisted.

$$`(\theta,\mathrm{optState},\mathrm{rng},x,y)
\longmapsto
(\theta',\mathrm{optState}',\mathrm{rng}',\mathrm{report})`

PyTorch's autograd is mature, broad, and deeply optimized. TorchLean's advantage is semantic access:
for supported fragments, the backward pass can be related to a Lean specification and the resulting
artifacts can be inspected inside the same formal environment.

For the public training surface, see the [public API facade](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Public.lean) and the
[training runtime](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/Train.lean).

# Graphs And Verification

PyTorch has several graph-oriented tools, including tracing, FX, `torch.export`, and compiler
pipelines. They are engineering tools for capture, transformation, optimization, and deployment.
They are not, by themselves, Lean theorem statements about a model.

TorchLean's graph path is designed for a different use case. A supported model can be lowered to
[NN.IR.Graph](https://github.com/lean-dojo/TorchLean/blob/main/NN/IR/Graph.lean), a graph whose nodes carry operation tags and a Lean denotation. Verifier
passes and certificate checkers can then talk about that graph directly:

$$`\operatorname{NN.IR.Graph.denoteAll}(g,\theta,x)`

A robustness checker, for example, should not have to trust a training script. It should receive a
graph, a parameter payload, an input region, and a certificate or bound propagation result whose
meaning is defined in Lean.

This is where TorchLean most clearly differs from a PyTorch clone. The goal is not simply to run the
same model syntax. The goal is to keep the runnable workflow connected to a semantic object that
proofs and checkers can cite.

# Import And Export

PyTorch remains valuable at the boundary. It has the ecosystem for large datasets, pretrained
checkpoints, distributed training, debugging tools, and deployment practices. TorchLean therefore
supports interop, but it keeps the contract deliberately small.

The supported round-trip is family based:

1. choose a known architecture family,
2. agree on parameter names, order, and tensor shapes,
3. train or edit the payload in Python,
4. serialize the payload,
5. re-import it into Lean and check the names and shapes again.

The relevant APIs are [PyTorch export](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/PyTorch/Export.lean),
[PyTorch import](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/PyTorch/Import.lean), and the
[round-trip examples](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Interop/PyTorch.lean). A narrow contract is easier to audit
and easier to connect to verification.

# Floating Point And Kernels

PyTorch users normally rely on the behavior of the selected backend: CPU kernels, CUDA kernels,
vendor libraries, compiler fusions, and device-specific floating-point choices. That is appropriate
for practical ML engineering, but it leaves a verification question: which numeric semantics did the
claim use?

TorchLean names the relevant layers separately:

- real valued specifications for clean mathematical statements,
- proof side float32 models such as `FP32`,
- executable IEEE binary32 models such as `IEEE32Exec`,
- native runtime kernels behind explicit assumptions.

This separation does not make native kernels disappear from the trusted base. It makes the boundary
precise. A theorem can say whether it is about real semantics, executable binary32 semantics, or a
native backend related to those semantics by a stated bridge.

# When To Use Which

Use PyTorch when you need the full production ecosystem: broad model coverage, mature GPU kernels,
distributed training, pretrained checkpoints, and standard deployment integrations.

Use TorchLean when the supported model family needs to be a Lean object: when shapes should be part
of the type, when the graph and payload should be inspectable, when imported weights need a checked
contract, or when a verifier result should connect to a formal statement.

A realistic workflow can use both. Train in PyTorch when that is the right engineering choice.
Export a known architecture and payload. Import it into TorchLean. Inspect the graph. Check the
certificate. State the remaining assumptions. TorchLean's contribution is the semantic layer around
that workflow, not a claim that every training job should move into Lean.
