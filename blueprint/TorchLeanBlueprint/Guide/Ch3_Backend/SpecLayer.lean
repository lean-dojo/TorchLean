import VersoManual
import VersoBlueprint

open Verso.Genre Manual

#doc (Manual) "The Spec Layer" =>
%%%
tag := "spec-layer"
%%%

The spec layer is where TorchLean says what a model means before asking how it runs. A dense layer
means `W x + b`. A softmax means a particular normalized exponential formula. A dropout training
step means a masked operation with the mask supplied explicitly. A loss means a specific reduction
and numerical convention.

Later layers may execute these definitions, lower them to graphs, approximate them with bounds, or
call faster kernels. The spec layer is the reference point they have to come back to.

TorchLean is written in Lean, which is both a programming language and an interactive theorem
prover. For broader language background, see the official Lean texts
[*Functional Programming in Lean*](https://lean-lang.org/functional_programming_in_lean/),
[*Theorem Proving in Lean 4*](https://lean-lang.org/theorem_proving_in_lean4/), and
[*The Lean Language Reference*](https://lean-lang.org/doc/reference/latest/). The main idea here is
simple: write the mathematical object as an ordinary Lean definition, then make later layers show
which definition they execute, differentiate, lower, or verify.

The rest of this page assumes the tensor and model-building material from the earlier chapters and
focuses on the semantic anchors that later runtime, autograd, graph, and verification statements
cite.

# What This Page Adds

A spec page should answer one question: when a runtime, compiler, autograd rule, or verifier says it
is handling a layer, which Lean definition is it referring to?

The answer should be a spec declaration. A spec declaration fixes the formula, the tensor shapes,
the scalar assumptions, the parameter record, and any conventions such as masks, epsilons,
reductions, or train/eval mode.

This gives each later layer a concrete obligation:

- runtime code must say which spec-level function it implements;
- autograd code must say which spec-level function it differentiates;
- graph compilers must say which spec-level function their graph denotes;
- verifiers must say which spec-level function their bounds or certificates concern.

Most users do not need to write specs by hand. The point is that the project has a named
mathematical object before execution speed, CUDA kernels, certificate formats, or export tools enter
the discussion.

# What A Spec Declaration Adds

A good spec declaration tells the reader five things at once:

- the tensor shapes involved,
- the scalar assumptions needed (`Context α`, `Zero α`, `Max α`, etc.),
- the parameter record, if the operation has trainable parameters,
- the exact mathematical formula,
- and the side conditions that are part of the definition, such as nonzero dimensions, masks, or
  explicit epsilon constants.

Here are representative names:

```
import NN

open NN
open Spec

-- Parameterized layer semantics.
#check LinearSpec
#check linearSpec
#check linearBackwardSpec

-- Pointwise and last-axis activation semantics.
#check Activation.reluSpec
#check Activation.softmaxSpec
#check Activation.logSoftmaxSpec

-- Objective semantics.
#check mseSpec
#check crossEntropyLogitsSpec

-- Explicit stochastic-mode semantics.
#check dropoutMaskedSpec
#check dropoutInferenceSpec
```

This is the level at which a paper statement should say what it means by "the model", "the loss",
"dropout", or "softmax". Runtime code may implement these definitions efficiently; verification
code may overapproximate them; but the spec declaration is where the target computation is named.

# Parameter Records Are Semantic Objects

The public model builder page explains how `nn.linear` allocates parameters. The spec layer explains
what those parameters *mean*. For a dense layer, the semantic object is `LinearSpec α inDim outDim`:
it contains a weight matrix and a bias vector with the shapes required by the formula `W x + b`.

That distinction matters when moving between layers:

- `nn.linear` is a user-facing model builder.
- `LinearSpec` is the mathematical parameter record.
- `linearSpec` is the forward function over that record.
- `linearBackwardSpec` is the reference backward contract for that layer.

The same pattern repeats across convolution, normalization, embedding, recurrent layers, attention,
and model-specific specs. The runtime may store parameters in a typed list or flat payload, but the
spec layer gives those parameters their mathematical roles.

For a dense layer, the roles line up like this:

- `nn.linear` is the user-facing layer builder.
- `LinearSpec` is the parameter record.
- `linearSpec` is the forward mathematical meaning.
- `linearBackwardSpec` is the backward reference rule.
- IR `.linear` is the graph opcode.
- A CUDA GEMM call is a possible runtime implementation for a supported Float32 path.

Those are different objects. They should agree through stated translations or assumptions, not
because their names look similar.

# Losses Are Part Of The Semantics

Training examples often talk about a model and a dataset, but a training claim is incomplete without
the objective. TorchLean keeps losses in the spec vocabulary too.

For example:

- `mseSpec predicted target` is a scalar objective over two tensors of the same shape.
- `crossEntropySpec predicted target` is cross entropy between probability tensors.
- `crossEntropyLogitsSpec logits target` names the stable logits form, with log-softmax inside the
  spec.

This avoids a common ambiguity. In code, "cross entropy" may mean probabilities passed through a
log, logits passed through a stable log-softmax, a mean reduction, a sum reduction, or a version
with ignored labels. A verification or autograd claim cannot just say "cross entropy." It needs the
exact spec.

# Stochastic Behavior Is Made Explicit

The spec layer does not hide randomness in global state. Dropout is the simplest example:

- `dropoutMaskedSpec p mask x` is the training-time mathematical operation once the mask is given.
- `dropoutInferenceSpec p x` is the deterministic inference-time operation.

The mask, seed, or random tensor belongs in the data of the computation. This makes later claims
auditable: a theorem can say whether it is about a fixed mask, an explicit random source, or an
inference-mode model.

# Scalar Semantics

The scalar type `α` is the numerical world in which the model is interpreted. The same architecture
can be read over real numbers for clean mathematics, over `IEEE32Exec` for executable binary32
behavior, over intervals for enclosures, or over `Float` for quick runtime examples.

Most TorchLean spec definitions are scalar-polymorphic: they work for any scalar type `α` that
supports the operations a neural network definition needs.

That interface is packaged as:

`[Context α]`

in [NN.Spec.Core.Context API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/Core/Context.lean).

Informally, `Context α` is "a scalar type on which machine learning expressions make sense":
arithmetic, order/comparison, and the standard functions used in activations and losses (`exp`,
`tanh`, `sqrt`, and so on).

This is what lets us reuse the exact same network code for:

- proofs (`α := ℝ`),
- executable float32 models (`α := IEEE32Exec`),
- interval enclosures (verification backends),
- and small runtime demos (`α := Float`).

## A Tiny Scalar-Polymorphic Definition

This is the smallest useful instance of TorchLean's "write the architecture once; choose the scalar
semantics later" design:

```
import NN

open NN
open Spec

def affine2 {α : Type} [Context α]
    (w11 w12 w21 w22 b1 b2 : α) :
    Spec.Tensor α (Shape.Vec 2) -> Spec.Tensor α (Shape.Vec 2)
  | x =>
      let y1 := w11 * Tensor.toScalar (Spec.get x ⟨0, by decide⟩)
              + w12 * Tensor.toScalar (Spec.get x ⟨1, by decide⟩)
              + b1
      let y2 := w21 * Tensor.toScalar (Spec.get x ⟨0, by decide⟩)
              + w22 * Tensor.toScalar (Spec.get x ⟨1, by decide⟩)
              + b2
      Tensor.dim (fun
        | ⟨0, _⟩ => Tensor.scalar y1
        | ⟨_, _⟩ => Tensor.scalar y2)
```

Not every model needs to be written by hand in this style. The important property is scalar
instantiation: the same definition may later be instantiated at `α := ℝ`, `α := Float`, or
`α := IEEE32Exec` without changing the architecture code.

# Theorem Shapes

Most downstream claims have one of a few recognizable forms. The details differ by model family, but
the target of the statement should be visible:

- a runtime theorem says an executable program returns the same value as a spec definition;
- an autograd theorem says a VJP or gradient routine computes the derivative of a spec definition;
- a compiler theorem says the denotation of a lowered graph equals the spec definition;
- a verifier theorem says a bound or certificate is sound for the spec definition, usually through
  the shared IR denotation.

Informally:

$$`\text{runtime/program output}
\;=\;
\text{Spec.forward(params,input)}`

$$`\text{Graph.denote}(g,payload,input)
\;=\;
\text{Spec.forward(params,input)}`

$$`\text{certificate accepted}
\;\Longrightarrow\;
\text{property of Spec.forward over the stated input set}`

For that reason, the spec layer still matters when the page you are reading is about CUDA, CROWN, or
export. Those layers are meaningful only after they say which spec-level function they are
implementing, differentiating, lowering, or bounding.

# A Practical Reading Habit

When reading a spec file, make three passes:

1. What function is being defined?
2. Which scalar and shape assumptions appear in the type?
3. Which later layer cites this definition?

This habit makes the spec API easier to navigate because most declarations are reusable components
rather than standalone programs. A layer declaration should tell us what it computes; a theorem about
that layer should say which shape, scalar, or state assumptions it needs.

For concrete code, the usual first stops are:

- [Context API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/Core/Context.lean), for the scalar operations a spec may use,
- [Linear layers API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/Layers/Linear.lean), for dense layer semantics,
- [Activation API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/Layers/Activation.lean), for ReLU, sigmoid, tanh, GELU, and related
  operations,
- [Loss API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/Layers/Loss.lean), for scalar objectives used by examples and training
  loops,
- [Dropout API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/Layers/Dropout.lean), for the explicit training/inference split.

# Composition Boundaries

The model-building chapter teaches the user-facing construction syntax. At the spec layer, the
useful question is different: where does each piece of a model become a named mathematical object?

- Layer specs name local formulas such as dense layers, convolutions, activations, normalization,
  pooling, and recurrent steps.
- Model specs compose those formulas into architectures, while keeping parameters explicit.
- Loss specs name the scalar objective used by a training or verification claim.
- Mode-specific specs separate training-time and inference-time behavior when the mathematics
  differs.

That separation is what lets a later theorem be precise. Instead of saying "the runtime matches the
model", the theorem can say exactly which forward spec, which parameters, which loss, and which
mode are involved.

# Semantic Discipline

Spec definitions should avoid hidden context. If an operation depends on a mask, epsilon, mode,
scalar semantics, reduction convention, or parameter record, that dependency should appear in the
definition or its type.

That discipline is what the next layers use. Runtime execution, graph denotation, derivative
statements, and verification certificates can point back to a precise spec-level object instead of
to a prose description of an operation.

# Architecture Specs Beyond The Small Examples

The [model spec API](https://github.com/lean-dojo/TorchLean/tree/main/NN/Spec/Models/) contains denotations for larger architecture families:
residual networks, Vision Transformer style definitions, GPT-style decoders, state-space models,
diffusion objectives, reinforcement-learning interfaces, and scientific-ML examples. The examples
and runtime wrappers may choose different execution paths, but the spec files are where the
mathematical forward maps and objectives are named.

When a demo claims to implement a larger model, a good reading path is:

1. Open the runnable example to see the user-facing workflow.
2. Follow the import to the corresponding spec or GraphSpec model.
3. Identify the forward spec and loss spec.
4. Then read the runtime, autograd, or verification theorem that cites those names.

# What To Read Next

Read *Runtime and Autograd* for executable traces and derivative paths. Read *Graphs and IR* for the
canonical op-tagged DAG. Read *GraphSpec* for architecture objects that have both pure spec
semantics and executable TorchLean programs. Read *Verification* for how bounds and certificates
refer back to the same denotation.

For concrete declarations, start with the [Context API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/Core/Context.lean), then the
[layer semantics API](https://github.com/lean-dojo/TorchLean/tree/main/NN/Spec/Layers/), the [model spec API](https://github.com/lean-dojo/TorchLean/tree/main/NN/Spec/Models/), and the
[autograd spec API](https://github.com/lean-dojo/TorchLean/tree/main/NN/Spec/Autograd/).

# References

- TorchLean paper: https://arxiv.org/abs/2602.22631
- Lean 4 reference manual: https://lean-lang.org/doc/reference/latest/
