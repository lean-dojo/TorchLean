import VersoManual
import VersoBlueprint

open Verso.Genre Manual

#doc (Manual) "A Running Example" =>
%%%
tag := "running-example"
%%%

The rest of the guide is easier if we keep one small model in view. The model is deliberately
ordinary: a two-layer classifier. Its job is not to be impressive as a model. Its job is to let us
watch the same computation appear as user code, parameters, a graph, a Float32 execution, and a
verification problem.

The question we keep asking is simple:

> What is the same model at this layer?

The path looks like this:

```
tinyClassifier
  -> model builder
  -> parameter payload
  -> graph plus payload
  -> Float32 execution
  -> input box plus verifier bounds
  -> semantic claim
```

Later examples use the same pattern for more interesting objects: causal masks with zero future
attention weight, fused attention specs related to ordinary scaled dot product attention, 3D
projection certificates, PINN residual bounds, neural-controller checks, and CROWN margin
certificates.

# The Public Model

The ordinary entry point is `import NN`. A small multilayer perceptron looks like familiar
model-building code, but the input and output shapes are part of the type. Here the model consumes a
feature vector of length `4` and produces two logits:

```
import NN

open NN.Tensor
open NN.API

def tinyClassifier : nn.M (nn.Sequential (Shape.Vec 4) (Shape.Vec 2)) :=
  nn.sequential![
    nn.linear 4 8 (pfx := Shape.scalar),
    nn.relu,
    nn.linear 8 2 (pfx := Shape.scalar)
  ]

def tinyTask (seed : Nat) :=
  train.classificationOneHot (nn.build seed tinyClassifier)
```

At this stage there is no proof obligation. This is model code. The difference from a Python script
is that the shape contract has already become part of the program. A later theorem, exporter, graph
pass, or checker does not have to rediscover that the model expects four input features and returns
two outputs.

The type is already saying the essential input/output contract:

$$`\operatorname{tinyClassifier} :
\operatorname{Model}\bigl(\operatorname{Tensor}(\alpha,[4]),
                           \operatorname{Tensor}(\alpha,[2])\bigr)`

# The Same Model As Data

When the model is built, TorchLean separates architecture from parameters. That is the first
boundary that matters for verification and interoperation. The architecture says which operations
exist and how they are composed; the parameter bundle says which trained numbers those operations
use.

```
def builtTiny := nn.build 2026 tinyClassifier

-- The structure and the parameter bundle are explicit values.
#check builtTiny
```

PyTorch also has a parameter bundle, usually exposed through `state_dict`. TorchLean makes that idea
central rather than incidental: parameters are passed, saved, loaded, lowered, and checked as data.
That choice is what lets the same trained model move from model authoring to graph inspection to
verification without becoming an untyped blob.

The denotation we keep returning to has this shape:

$$`\operatorname{forward}(architecture,\theta,x) = y`

where the architecture, parameters `θ`, input `x`, and output `y` are all ordinary values that can
be inspected or related by theorems.

# The Same Model As A Graph

The graph chapters explain how TorchLean lowers model code into `NN.IR.Graph`, a DAG whose nodes
carry operation tags and are shared by runtime inspection and verifier code. The graph is not meant
to be pleasant model authoring syntax. It is the object a compiler, widget, or verifier can inspect
node by node.

The discipline is:

- user code should stay readable;
- the lowered graph should stay explicit;
- theorem statements should name the semantics of that graph.

That is why we spend time on both `nn.Sequential` and `NN.IR.Graph`. They are not competing
interfaces. They are two views of the same computation.

For a lowered graph, the semantic question becomes:

$$`\operatorname{NN.IR.Graph.denoteAll}(g,payload,input)`

That expression is the reference meaning that a compiler pass, widget, verifier, or runtime bridge
has to respect. If a pass fuses two operations, changes a layout, or exports a certificate, the
claim is always about preserving or soundly approximating this denotation.

# The Same Model Under Float32

A theorem over real numbers and a float32 execution are not the same statement. TorchLean keeps the
scalar semantics visible:

- `ℝ` is the clean mathematical target;
- `FP32` is the proof friendly model based on rounded reals;
- `IEEE32Exec` is the executable IEEE-754 binary32 model;
- native CUDA kernels are accelerated external code behind an explicit boundary.

The Float32 chapters explain how those views are connected, and where a theorem still depends on an
assumption about the external runtime.

# The Same Model As A Verification Problem

Once the model has a graph and a payload, verification tools can ask bounded questions about it. For
the two-logit classifier above, a typical local robustness question is:

> For every input `x` in this box around a reference example, is logit `0` still greater than logit `1`?

The verifier does not need to trust the training script. It receives explicit objects:

- What output interval follows from this input box?
- Does a margin remain positive after bound propagation?
- Which certificate or JSON artifact was checked, and which values came from an external producer?

The core checker can be read schematically as a small Boolean computation over certified bounds:

```
-- Accept when class 0 is still ahead of class 1.
def marginCertificateOK (logit0Lower logit1Upper : Float) : Bool :=
  logit0Lower > logit1Upper
```

The real verifier carries richer data than two floats, but the shape of the argument is the same:
an external analyzer may propose bounds or certificates, while Lean checks the condition that gives
those artifacts their meaning.

# A Tiny Checked Shape Example

Here is the smallest version of the idea. Two vectors of the same shape can be added. Two tensors of
different shapes cannot be passed to the same typed binary op without an explicit reshape, cast, or
proof.

```
import NN

open NN.Tensor

def a : Tensor Float (shape![2]) :=
  tensor! [1.0, 2.0]

def b : Tensor Float (shape![2]) :=
  tensor! [0.25, -0.5]

def good := Spec.Tensor.addSpec a b

def wrongShape : Tensor Float (shape![2, 1]) :=
  tensor! [[1.0], [2.0]]

-- This is deliberately not accepted:
-- def shapeMismatch := Spec.Tensor.addSpec a wrongShape
```

That tiny example is the design in miniature. TorchLean does not try to guess which broadcast,
reshape, or deployment convention you meant. A real convention should appear as a named operation in
the code, so the model author, exporter, and checker all see the same transformation.

# What To Watch For In The Next Chapters

As the examples get larger, keep asking where each object lives:

- *Spec*: the mathematical meaning of tensors, layers, and losses.
- *Runtime*: eager or compiled execution, gradients, optimizers, logging, and devices.
- *IR*: an op tagged graph that can be inspected and verified.
- *Proofs*: theorems about the spec, the graph, or the verifier output.
- *Trust boundaries*: CUDA kernels, PyTorch exporters, external certificate producers, and datasets.

Is this a tensor in the spec layer? A runtime value? A graph node? A theorem about graph denotation?
A certificate imported from outside Lean? Most TorchLean mistakes become easier to diagnose once
that question is clear.
