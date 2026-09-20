import VersoManual
import NN.API
import NN.Proofs.Autograd.FDeriv.LogSoftmax
import NN.Proofs.Autograd.FDeriv.SoftmaxSpec
import NN.Proofs.Autograd.Runtime.Link.FDeriv
import NN.Proofs.Autograd.Tape.Ops.Attention.SpecBridge
import NN.Proofs.Autograd.Tape.Ops.Conv.FDeriv
import NN.Proofs.Autograd.Tape.Ops.Norm.BatchNorm
import NN.Proofs.Autograd.Tape.Ops.Norm.LayerNorm
import NN.Proofs.Autograd.Tape.Ops.Norm.LayerNormAdjoint
import NN.Proofs.Models.Attention.HardMask
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean

-- The autograd proofs live under a root `Proofs.Autograd` namespace with one sub-namespace per
-- operator family, and Verso keeps displayed code narrow enough to read beside the text. Opening
-- the namespaces here lets every `#check` below name its theorem the short way; a printed signature
-- still spells out each constant in full, so the abbreviation hides nothing.
open Proofs.Autograd
open Proofs.Autograd.Algebra.Graph
open Proofs.Autograd.Attention
open Proofs.Autograd.BatchNorm
open Proofs.Autograd.Conv
open Proofs.Autograd.LayerNorm
open NN.Proofs.Models.Attention

-- Two signatures shown below print wider than this file's 100-column limit, so their `leanOutput`
-- blocks ask for `whitespace := lax` and wrap in the source. The rendered page still shows each
-- message exactly as Lean printed it.
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Autograd Proofs" =>
%%%
tag := "autograd-proofs"
%%%

Calling `loss.backward()` starts a reverse traversal of the operations that produced the loss.
Each local rule pulls an output sensitivity back to the operation's inputs, and the engine adds
contributions when several paths reach the same input. This is the reverse-mode algorithm used by
frameworks such as PyTorch {Informal.citep pytorch2019}[] and surveyed by
{Informal.citet baydin2018}[].

To state what that traversal computes, fix an input to the forward function. Its derivative maps a
small input perturbation, or *tangent*, to a first-order output perturbation. An output *cotangent*
assigns a scalar sensitivity to those output perturbations. Reverse mode pulls that cotangent back
to the inputs. Under the Euclidean inner products used here, this pullback is the adjoint of the
derivative: it is represented by the transposed Jacobian, without requiring the program to build
the Jacobian matrix.

TorchLean proves this connection in layers under
[NN/Proofs/Autograd](https://github.com/lean-dojo/TorchLean/tree/main/NN/Proofs/Autograd/). First,
local forward and reverse rules satisfy an algebraic pairing law. A graph induction extends that
law to reverse accumulation. Analytic proofs then identify the forward rules with derivatives of
the functions the graph evaluates. Runtime links connect this proof graph to an executed tape.

The proof pipeline is:

```
-- Each step connects a different object: a derivative, its
-- adjoint, and an execution.
local op derivative
  -> local JVP/VJP adjointness
  -> graph backprop correctness
  -> Fréchet derivative theorem
  -> model block theorem
  -> training step algebra
```

That ordering is the proof structure. A theorem about a Transformer sublayer is assembled from
local derivative rules, graph composition, and the analytic bridge to the derivative of the
denotation.

# Tensor And Model Gradient Examples

The autograd quickstart gives two instances of the gradient we want to characterize:

```terminal
# Print tensor and model gradients for the losses discussed
# below.
lake exe torchlean quickstart_autograd
```

which prints

```terminal +output
== Differentiate a tensor function ==
mean(x^2) = 4.666667
d/dx       = [0.666667, 1.333333, 2.000000]

== Differentiate a model loss ==
loss     = 0.769887
gradient = [[1, 2]: [[-0.877432, 1.754865]], [1]: [-1.754865]]
```

Both halves can be checked by hand, which is the point of starting here. The first differentiates
$`x\mapsto\frac{1}{3}\sum_i x_i^2` at $`x=(1,2,3)`. The value is $`14/3`, and the gradient is
$`\frac{2}{3}x=(2/3,4/3,2)`, which is what got printed. In PyTorch the same two lines read

```
# Differentiate the matching scalar mean-square loss in the
# comparison API.
import torch
x = torch.tensor([1., 2., 3.], requires_grad=True)
(x ** 2).mean().backward()
x.grad          # tensor([0.6667, 1.3333, 2.0000])
```

The first coordinate can also be expressed as a scalar derivative, holding the other two
coordinates fixed:

```lean
-- Hold the other two squared entries fixed and
-- differentiate one input coordinate.
example (a b c : ℝ) :
    HasDerivAt (fun t : ℝ => (t ^ 2 + b + c) / 3)
      (2 * a / 3) a := by
  have h : HasDerivAt
      (fun t : ℝ => t ^ 2 + b + c) (2 * a) a := by
    simpa using
      ((hasDerivAt_pow 2 a).add_const b).add_const c
  simpa using h.div_const 3
```

The second half of the demo is a single affine layer `y = W x + b` with two inputs, one output, and
squared-error loss, run at `x = [0.5, -1.0]` and `t = [0.25]`. We can recover the weight gradient
from the bias gradient and this input. The bias gradient is $`\partial L/\partial b=2(y-t)`, and the
weight gradient is that same scalar times the input, so

$$`\nabla_W L = (2(y-t))\,x^\top = (-1.754865)\cdot(0.5,-1.0) = (-0.877432,\;1.754865),`

which agrees with the displayed row to its precision, and $`L=(y-t)^2\approx0.769887`.
The rounded printed gradient does not recover every bit of the loss. The chain rule for that layer
is one more `HasDerivAt` proof, this time in a weight rather
than in an input:

```lean
-- Apply the chain rule to the affine prediction and then
-- square the residual.
example (w₁ w₂ b t x₁ x₂ : ℝ) :
    HasDerivAt
      (fun w : ℝ => (w * x₁ + w₂ * x₂ + b - t) ^ 2)
      (2 * (w₁ * x₁ + w₂ * x₂ + b - t) * x₁) w₁ := by
  set y := w₁ * x₁ + w₂ * x₂ + b with hy
  have h : HasDerivAt
      (fun w : ℝ => w * x₁ + w₂ * x₂ + b - t) x₁ w₁ := by
    simpa using
      ((((hasDerivAt_id w₁).mul_const x₁).add_const
        (w₂ * x₂)).add_const b).sub_const t
  have hsq : (fun w : ℝ => (w * x₁ + w₂ * x₂ + b - t) ^ 2)
      = fun w : ℝ =>
        (w * x₁ + w₂ * x₂ + b - t)
          * (w * x₁ + w₂ * x₂ + b - t) := by
    funext w
    ring
  rw [hsq, show 2 * (y - t) * x₁
      = x₁ * (y - t) + (y - t) * x₁ by ring]
  exact h.mul h
```

This proves one component of the gradient for an affine layer. The graph theorem will assemble
such local derivatives and account for parameters used along several paths, where the reverse pass
must add their contributions.

The two scalar proofs unpack the displayed gradients. In the mean-square example, differentiating
one squared entry gives twice that entry and the mean contributes the factor one third. In the
model example, the squared residual contributes twice the residual, while differentiating the
affine prediction with respect to a weight contributes its input feature. The bias contributes
one instead. Thus a common residual factor appears in all parameter gradients, but each weight
also remembers which feature it multiplies. The numerical output is easier to inspect once these
separate factors are visible.

# Autograd Proof Obligations

In PyTorch, the default runtime model is approximately:

1. the eager engine records operations into a dynamic tape;
2. each operation has a backward rule, either built in or registered by extension code;
3. the engine traverses the tape in reverse and accumulates gradients;
4. tests, numerical checks, and framework maintenance give confidence that the result is right.

JAX moves the same idea into a functional transformation pipeline: `grad`, `vjp`, `jvp`, `jit`, and
lowering passes cooperate to produce differentiated programs and typed graphs. See the PyTorch
autograd overview at https://pytorch.org/docs/stable/autograd.html and JAX's autodiff guide at
https://jax.readthedocs.io/en/latest/automatic-differentiation.html for the user facing version of
that workflow.

TorchLean keeps the same operational picture: local derivative rules plus reverse accumulation.
What changes is that runtime confidence is split into named theorem obligations.

- A registered backward rule becomes a local JVP/VJP or Fréchet derivative lemma for the op.
- A tape reversal becomes global tape soundness by induction over the graph.
- A scalar loss training step becomes an explicit theorem about the loss seed and parameter update
  algebra.
- A model block such as attention or an RNN cell becomes a packaged theorem with stated hypotheses.

These obligations allow the derivative proof to be checked separately from representation and
arithmetic agreement. A proof about a real-valued graph becomes a claim about executed gradients
only after those connections have been supplied.

A simple fanout example explains why local formulas are only the beginning. In a graph computing
$`x\,x`, multiplication receives two references to the same variable. Its two local input
cotangents are both needed, and reverse accumulation adds them at that shared variable to obtain
$`2x` for seed one. Keeping only one contribution would produce a tensor of the right shape with
the wrong value. The graph theorem handles this bookkeeping uniformly; the local derivative
certificate explains why each of the two contributions was correct in the first place.

# Tape Soundness: The Algebraic Core

The central algebraic file is
{src "NN/Proofs/Autograd/Tape/Algebra/Soundness.lean"}[NN.Proofs.Autograd.Tape.Algebra.Soundness
API]. It defines the small tape language used by the rest of the autograd proofs, together with the
local proof data carried by each node.

The objects to track are:

- `TensorPack`: a typed tuple of tensors indexed by a list of shapes. It carries model state,
  inputs, and cotangents without introducing another tensor representation.
- `Idx`: pointer into a context, carrying the needed proof data.
- `NodeData`: forward, JVP, and VJP data for one local operation.
- `Node`: a node plus the local inner product soundness law.
- `GraphData`: executable graph data built by appending nodes, each referring to earlier values.
- `Graph`: graph whose nodes can share earlier values and carry local proof obligations.
- `Graph.backprop_correct`: global dot product soundness theorem.

For an input context `x`, let `dx` be a tangent of the same shape and let `seed` be a cotangent
for the context returned by graph evaluation. `Graph.backprop_correct` states that forward
sensitivity and reverse accumulation satisfy the following pairing identity:

$$`\langle \operatorname{jvp}_G(x, dx), seed\rangle = \langle dx, \operatorname{backprop}_G(x,
seed)\rangle`

This identity characterizes the adjoint relationship between the supplied JVP and reverse rules.
A forward sensitivity `dx`
pushed through the graph and then paired with an output cotangent gives the same scalar as pairing
`dx` with the cotangent produced by reverse accumulation.

The dot product statement gives a compositional proof obligation: each node must satisfy its local
adjoint law. Induction over `Graph` then shows that reverse accumulation preserves the pairing
through the whole graph. The analytic theorem below supplies the further fact that the forward
sensitivity is a derivative.

For a linear example, take a graph whose forward map is a $`2\times2` matrix $`A`. Forward
sensitivity
is
$`dx\mapsto A\,dx`, the reverse rule is $`seed\mapsto A^\top seed`, and the adjoint law is the claim
that the two pairings agree:

```lean
-- Expand both inner products to verify the transpose
-- identity for a two-by-two matrix.
example (a b c d x₁ x₂ y₁ y₂ : ℝ) :
    (a * x₁ + b * x₂) * y₁ + (c * x₁ + d * x₂) * y₂
      = x₁ * (a * y₁ + c * y₂)
        + x₂ * (b * y₁ + d * y₂) := by
  ring
```

The left-hand side pushes `x` forward and pairs the result with `y`; the right-hand side pairs `x`
with `y` pulled back. Reading the right-hand side coefficientwise is the clearest explanation of why
a reverse pass is a transpose: the first input cotangent is $`a y_1+c y_2` , using the first
*column* of $`A` .
The second is $`b y_1+d y_2`. `Graph.backprop_correct` is this statement made compositional, and the
induction over `Graph` does the work that `ring` does here.

The comparison to PyTorch is most direct at this point. PyTorch's engine performs a reverse walk
over a dynamic graph and accumulates cotangents into inputs. TorchLean's algebraic graph does the
same conceptual work, but the graph object carries enough structure for Lean to prove that the
accumulation is sound for every input context in the supported fragment.

The two-by-two calculation expresses the same scalar in two orders. On the left, the matrix acts
on the tangent before pairing with the output cotangent. On the right, its transpose acts on that
cotangent before pairing with the original tangent. Expanding the products shows that every
coefficient appears once on each side. This identity is exactly what reverse mode needs to move
a scalar pairing backwards through a node. It also explains why reverse accumulation adds
contributions when several later pairings depend on the same earlier value.

# Adjoint Laws And Fréchet Derivatives

The pairing law alone does not establish that the JVP differentiates the forward function. Two
incorrect rules could still be adjoints of each other. The
{src "NN/Proofs/Autograd/Tape/Core/FDeriv.lean"}[Fréchet derivative bridge] supplies the missing
analytic statement: the proposed linear map approximates the change in the forward function to
first order, with a remainder negligible relative to the input perturbation.

That file vectorizes shaped tensor contexts into Euclidean spaces and connects three views:

- shaped tensors: `Tensor Real s`, `TensorPack Real Gamma`;
- flat Euclidean vectors: `CtxVec Gamma`, `flattenCtx`, `unflattenCtx`;
- analytic derivatives: `HasFDerivAt`, `fderiv`, and `ContinuousLinearMap.adjoint`.

The main theorem is `Graph.backpropVec_eq_adjoint_fderiv`. In plain English:

> If every node in the graph has the stated Fréchet derivative, then graph backprop equals the
> adjoint of the Fréchet derivative of graph evaluation.

There is also a pointwise version, `Graph.backpropVec_eq_adjoint_fderiv_at`, for hypotheses that
only hold at a particular input. Neural networks need this local form because ReLU, normalization,
division, logarithms, and square roots all have domain or nondifferentiability issues. TorchLean
states the local smoothness or nonzero conditions needed by the graph being differentiated.

Adjointness alone cannot identify the derivative of the forward computation. Consider the scalar
forward function $`f(x)=x^2` and imagine assigning zero to both its JVP and VJP. Their pairings
agree for every tangent and cotangent, so the adjoint identity holds. At $`x=1`, however, the
actual derivative is multiplication by two. The missing fact is the connection from the proposed
JVP to the derivative of `f`. Once that fact and adjointness are both established, reverse-mode
correctness follows for this node and can participate in the graph induction.

Flattening the typed context is a mathematical change of representation, not a change in which
variables are differentiated. Each tensor contributes its coordinates to the Euclidean space,
and unflattening recovers the same tensor positions and shapes. This allows Mathlib's derivative
and adjoint APIs to apply to a heterogeneous parameter pack. The ordering must remain consistent
on the tangent and cotangent sides: transposing a matrix representation with one coordinate
order and interpreting its result with another would describe a different pullback. The context
conversion lemmas are what let the graph theorem move between these representations without
leaving that correspondence informal.

## The Adjoint Identity, Numerically

Let `J` denote the Jacobian of the forward map at the chosen input. A numerical check of the
adjoint identity compares two scalar pairings:

$$`\langle J u,\, w\rangle = \langle u,\, J^\top w\rangle`

for every tangent $`u` at the input and every cotangent $`w` at the output. Both sides are things
TorchLean can run. `autograd.jacfwd` pushes forward once per input coordinate and assembles the
Jacobian tensor; `autograd.vjp` runs the reverse pass once against a cotangent and returns
$`J^\top w` directly. The traversals differ but share graph construction, forward operations, and
local derivative
definitions; agreement can therefore miss a shared local-rule error.

The map below is deliberately not diagonal: squaring and then multiplying by a mean couples every
output to every input, so a transpose that got an index wrong would not quietly cancel.

```lean (name := apAdj)
section
open TorchLean
open scoped TorchLean.Tensor

-- f(x) = exp(x) * mean(x^2). The mean makes the Jacobian
-- dense: every output reads every input.
def apAdjMap : autograd.Function [3] [3] := fun x => do
  let squared ← nn.functional.square x
  let scaleFactor ← nn.functional.mean squared
  let curved ← nn.functional.exp x
  nn.functional.mulB (t := [3]) curved scaleFactor

def apAdjPoint : Tensor Float [3] := [0.4, -0.7, 1.1]

-- A tangent and a cotangent. Both are arbitrary, because
-- the identity is not supposed to care which ones we pick.
def apAdjTangent : Tensor Float [3] := [1.0, 0.5, -2.0]

def apAdjCotangent : Tensor Float [3] := [0.25, -1.5, 2.0]

#eval show IO Unit from do
  let jacobian ← autograd.jacfwd apAdjMap apAdjPoint
  let pullback ←
    autograd.vjp apAdjMap apAdjPoint apAdjCotangent
  let pushed : Tensor Float [3] :=
    einsum jacobian, apAdjTangent
      "output input, input -> output"
  let forward := Tensor.dotSpec pushed apAdjCotangent
  -- Reverse side: one backward pass, then one dot product.
  let reverse := Tensor.dotSpec apAdjTangent pullback
  IO.println s!"forward    = {forward}"
  IO.println s!"reverse    = {reverse}"
  IO.println s!"difference = {forward - reverse}"
  IO.println s!"equal      = {forward == reverse}"

end
```
```leanOutput apAdj
forward    = -15.528866
reverse    = -15.528866
difference = 0.000000
equal      = true
```

The computed scalars satisfy `Float` equality for this input and this tangent/cotangent pair.
That exact agreement depends on these particular binary64 computations. The real theorem applies
to every direction under its hypotheses; it does not promise bitwise equality between differently
ordered floating-point sums. A general numerical claim needs local error bounds and a graph-level
approximation theorem, as developed in {ref "runtime-approximation"}[the runtime approximation
chapter]. The run is a useful check of indexing and accumulation, while shared errors in local
rules can still escape it.

The two equal printed pairings test the identity for the particular tangent and cotangent chosen
in the example. Choosing nonuniform values helps expose a transpose or reduction mistake that a
seed of all ones might hide. Nevertheless, the floating-point equality is one observation. The
adjoint theorem quantifies over all tangents and cotangents in the stated spaces, and the analytic
theorem identifies the underlying linear map as a derivative. The experiment is most useful as a
way to understand which scalar quantity those theorems compare.

# Derivative Correctness Of The Lowered Tape

The derivative theorem above is stated for the real analytic graph. The tape-lowering correctness
theorem was originally stated for a more general algebraic graph: its scalar type is abstract, and
each node may read a non-differentiable environment. Those are useful abstractions, but leaving the
two results side by side would not prove that the lowered tape computes the Fréchet derivative.

{src "NN/Proofs/Autograd/Runtime/Link/FDeriv.lean"}[NN.Proofs.Autograd.Runtime.Link.FDeriv]
closes that gap. At scalar type `Real` , fixing the algebraic environment converts nodes to analytic
nodes;
the reverse conversion uses the fixed environment through the bridge. Both node and graph
conversions round-trip, and evaluation, JVP, and
reverse accumulation commute with the conversion.

The algebraic reverse pass returns cotangents for the inputs and every intermediate value. The
analytic theorem needs only the input cotangent. `TensorPack.takeLeft` selects that input prefix,
and `takeLeft_backpropAllCtx` proves that it is exactly the inputs-only reverse pass used by the
derivative theorem.

Both public endpoints below are instantiated over `Real`. Their conclusions connect the dense
tape result to the adjoint derivative, with the fixed environment and differentiability hypothesis
visible in the signature:

```lean (name := agLink)
-- Compare the global derivative premise with its
-- point-local counterpart.
#check @backwardDenseFrom_lowerGraphToTape_adjoint_fderiv
#check @backwardDenseFrom_lowerGraphToTape_adjoint_fderiv_at
```
```leanOutput agLink (whitespace := lax)
@backwardDenseFrom_lowerGraphToTape_adjoint_fderiv : ∀ {Δ : Type} {Γ ss : List Spec.Shape} (g :
  Algebra.Graph Δ Γ ss)
  (x : TorchLean.TensorPack ℝ Γ) (d0 : Δ) (seed : TorchLean.TensorPack ℝ (Γ ++ ss))
  (hg : GraphFDerivCorrect (g.toReal d0)),
  (g.lowerGraphToTape x d0).1.backwardDenseFrom seed.toShapeErasedArray =
      Except.ok (g.backpropAllCtx x d0 seed).toShapeErasedArray ∧
    flattenCtx (Algebra.TensorPack.takeLeft (g.backpropAllCtx x d0 seed)) =
      (ContinuousLinearMap.adjoint (fderiv ℝ (g.toReal d0).evalVec (flattenCtx x)))
        (flattenCtx seed)
```
```leanOutput agLink (whitespace := lax)
@backwardDenseFrom_lowerGraphToTape_adjoint_fderiv_at : ∀ {Δ : Type} {Γ ss : List Spec.Shape} (g
  : Algebra.Graph Δ Γ ss)
  (x : TorchLean.TensorPack ℝ Γ) (d0 : Δ) (seed : TorchLean.TensorPack ℝ (Γ ++ ss))
  (hg : GraphFDerivCorrectAt (g.toReal d0) (flattenCtx x)),
  (g.lowerGraphToTape x d0).1.backwardDenseFrom seed.toShapeErasedArray =
      Except.ok (g.backpropAllCtx x d0 seed).toShapeErasedArray ∧
    flattenCtx (Algebra.TensorPack.takeLeft (g.backpropAllCtx x d0 seed)) =
      (ContinuousLinearMap.adjoint (fderiv ℝ (g.toReal d0).evalVec (flattenCtx x)))
        (flattenCtx seed)
```

`GraphFDerivCorrect` supplies the node derivative rules globally; `GraphFDerivCorrectAt` needs
them only along evaluation at the given input. The latter supports a graph containing operators
whose derivatives exist only on part of their domain. Both hypotheses lead to the same conjunction:
successful reverse execution and identification of its input cotangent with an analytic adjoint.

The result of reverse execution includes cotangents for inputs and intermediate values.
`takeLeft` selects the input prefix, and `flattenCtx` turns that shaped context into the Euclidean
vector used by `fderiv`. These functions specify which part of the runtime result the derivative
claim concerns.

Suppose `g` is an algebraic graph over `Real`, `x` is its typed input context, `d` is its fixed
environment, and `seed` is a cotangent for the full evaluated context `Γ ++ ss`, including the
input prefix and intermediate values. To differentiate a loss attached only to the final output,
its seed is placed in that output slot and the other slots are zero. The environment is held fixed:
a parameter to be differentiated must be included in the input context. The first theorem returns
a conjunction:

1. compiling `g` and running `Tape.backwardDenseFrom` succeeds with the graph's full reverse
   context;
2. the input prefix of that context, after flattening, is
   $`(\operatorname{fderiv}\,\operatorname{eval}(x))^\dagger seed`.

The `_at` theorem asks for differentiability only at `x`. It is the useful form for graphs
containing piecewise-smooth operators, provided the execution point avoids their non-differentiable
or invalid cases.

These are theorems about the exact tape instantiated over `Real`. A Lean `Float` or CUDA run needs
an additional numerical-refinement argument; the rounded-runtime chapter develops that separate
layer rather than folding it into the exact derivative claim.

For the running example, take
$`forward(x) = softmax(Wx + b)`. The scalar loss supplies an output cotangent
$`seed = dL/dforward`, and the reverse pass returns the input and parameter cotangents
$`dL/dx`, $`dL/dW`, and $`dL/db`.

In this example, `x`, `W`, and `b` must all be part of the differentiated input context to obtain
all three cotangents. Composing the scalar loss seed with the adjoint of the forward map gives the
corresponding derivatives of the loss.

The lowered-tape signatures contain both an execution result and a mathematical identification of
that result. Successful execution settles which cotangent pack the evaluator returned; the
adjoint equality settles what that pack means. The context includes the graph inputs and stored
intermediates, so a seed on an intermediate asks a different differentiation question from a seed
only on the final output. For an ordinary scalar loss, the application chooses the seed pack that
selects that loss and puts zero in unrelated entries. The theorem's general seed argument allows
other observations, but does not choose the intended loss on the caller's behalf.

The point-local variant is particularly useful for piecewise operations. It requires the local
derivative conditions at the values reached by this execution, instead of requiring them at every
possible input. That can justify an ordinary derivative inside a fixed ReLU region. It does not
turn the selected slope at a kink into a Fréchet derivative there; the analytic premise still has
to be true at the point where the theorem is applied.

# Operator Specs: Softmax And LogSoftmax

Softmax couples every output to every input through its normalization denominator. Following
that dependence gives us both its derivative and the related log-softmax rule. Their APIs are:

- {src "NN/Proofs/Autograd/FDeriv/Softmax.lean"}[NN.Proofs.Autograd.FDeriv.Softmax API]
- {src "NN/Proofs/Autograd/FDeriv/LogSoftmax.lean"}[NN.Proofs.Autograd.FDeriv.LogSoftmax API]

The softmax API defines `softmaxVec`, `softmaxDerivCLM`, and `softmaxJvp`. The theorem
`softmaxJvp_eq_deriv` identifies the implemented JVP formula with the derivative formula, and
`hasFDerivAt_softmaxVec` states the Fréchet derivative of the vector softmax. The theorem
`inner_softmaxJvp_comm` packages the self adjoint structure of the softmax Jacobian, which is the
reason the VJP can reuse the same formula shape.

The log-softmax API follows the same discipline with `logSoftmaxVec`, `logSoftmaxJvp`, and
`logSoftmaxVjp`. The theorem `logSoftmaxJvp_eq_deriv` gives the derivative formula, while
`inner_logSoftmaxJvp_vjp` states the adjoint relationship between the JVP and the VJP:

For softmax, if

$$`s_i=\frac{e^{x_i}}{\sum_j e^{x_j}},`

each `s_i` is a normalized exponential. Perturbing one input changes its numerator and also the
shared denominator. The quotient rule therefore subtracts the `s`-weighted mean of the input
direction from each coordinate before multiplying by `s_i`:

$$`D\,\operatorname{softmax}(x)[dx]_i
=
s_i\left(dx_i-\sum_j s_j dx_j\right).`

For log-softmax, differentiating the outer logarithm cancels the leading `s_i`. The same
weighted mean remains:

$$`D\,\log\operatorname{softmax}(x)[dx]_i
=
dx_i-\sum_j \operatorname{softmax}(x)_j\,dx_j.`

The two displayed formulas are JVPs. Softmax's Jacobian is symmetric, so its VJP uses the same
formula with the cotangent in place of the tangent. Log-softmax's Jacobian is generally not
symmetric: its VJP subtracts the softmax vector times the sum of the cotangent entries. The
adjointness theorem checks that distinct reverse rule against the displayed directional derivative.

Their comments cite the PyTorch API reference for naming alignment, not as proof sources:

- [PyTorch softmax docs](https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html)
- [PyTorch log-softmax
  docs](https://pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html)

Documentation tells us what users expect the op to mean; Lean proves the derivative law for
TorchLean's mathematical definition.

The difference between the log-softmax JVP and VJP is visible in their sums. The JVP subtracts
the softmax-weighted average of the input tangent from every coordinate. Its adjoint instead
subtracts the softmax weight at that coordinate times the unweighted sum of the output
cotangents. Interchanging these formulas would be an error because the log-softmax Jacobian is
not generally symmetric. For softmax itself, the Jacobian is symmetric, so the analogous formulas
coincide. The adjoint identity establishes which expression belongs on each side without relying
on that special symmetry.

## Operator Derivative Proof Obligations

For a new differentiable operator, we need the same three objects exposed by the softmax files:

```
-- Mathematical forward map.
def forward : Vec n -> Vec m := ...

-- Directional derivative used by the forward-mode view.
def jvp : Vec n -> Vec n -> Vec m := ...

-- Reverse rule used by backprop.
def vjp : Vec n -> Vec m -> Vec n := ...
```

The local theorem should then say two things, and both are short enough to print in full. `Vec n`
abbreviates `EuclideanSpace ℝ (Fin n)`, the flat vector view for which
{Informal.citet mathlib2020}[] supplies `HasFDerivAt` and `ContinuousLinearMap.adjoint`:

```lean (name := sfx)
-- The JVP is the derivative of the forward denotation.
#check @hasFDerivAt_softmaxVec
```

```leanOutput sfx
@hasFDerivAt_softmaxVec : ∀ {n : ℕ} (x : Vec n), HasFDerivAt softmaxVec (softmaxDerivCLM x) x
```

In the first signature, `softmaxDerivCLM x` is the continuous linear map proposed as the
derivative at `x`, and `HasFDerivAt` is the proposition identifying it with the first-order
behavior of `softmaxVec`. The theorem accepts every vector `x`; it does not ask the caller to
prove differentiability again. This is stronger than merely writing an expression involving
`fderiv`, whose notation alone supplies no proof that a proposed local formula is correct.

```lean (name := lsfx)
-- The VJP is adjoint to the JVP.
#check @inner_logSoftmaxJvp_vjp
```

```leanOutput lsfx (whitespace := lax)
@inner_logSoftmaxJvp_vjp : ∀ {n : ℕ} (x dx δ : Vec n),
  inner ℝ (logSoftmaxJvp x dx) δ = inner ℝ dx (logSoftmaxVjp x δ)
```

The second theorem pairs the JVP with an arbitrary output cotangent `δ` and pairs `dx` with the
VJP. Equality for every `dx` and `δ` is the local adjoint law used by the graph induction.

Softmax outputs sum to one. Differentiating that constant sum shows that every output tangent
must sum to zero. At
$`n=2` that is another `ring` call once $`s_1+s_2=1` has been used:

```lean
-- Normalization makes the common-shift contribution cancel
-- from the summed JVP.
example (s₁ s₂ dx₁ dx₂ : ℝ) (h : s₁ + s₂ = 1) :
    s₁ * (dx₁ - (s₁ * dx₁ + s₂ * dx₂))
      + s₂ * (dx₂ - (s₁ * dx₁ + s₂ * dx₂)) = 0 := by
  have hs : s₂ = 1 - s₁ := by linarith
  subst hs
  ring
```

Each coordinate of the tangent is $`s_i(dx_i-\langle s,dx\rangle)`, and the $`s`-weighted average of
$`dx_i-\langle s,dx\rangle` is zero by construction. Because the softmax Jacobian is symmetric, its
VJP also has zero sum.
A backward rule that violates this property cannot be the adjoint of the softmax derivative. The
cotangent itself is not a probability vector; zero sum is an invariant of the rule.

Those two theorems are about the analytic vector functions. The specification layer defines
`Activation.softmaxSpec` on tensors in the numerically stable max-shifted form, so a third file,
`NN.Proofs.Autograd.FDeriv.SoftmaxSpec`, closes the gap: `hasFDerivAt_softmaxSpec_vec` transfers
differentiability to the spec kernel, and `softmaxBackwardSpec_eq_vjp` proves that the spec
backward rule is the vector-Jacobian product of the spec forward map. Log-softmax has the same
pair. Without that bridge, a theorem about `softmaxVec` would say nothing about the kernel the
runtime lowers.

```lean (name := agSpecBridge)
-- Connect the stable specification’s forward function and
-- its actual backward formula.
#check @hasFDerivAt_softmaxSpec_vec
#check @softmaxBackwardSpec_eq_vjp
```
```leanOutput agSpecBridge (whitespace := lax)
@hasFDerivAt_softmaxSpec_vec : ∀ {n : ℕ} (xV : Vec n),
  HasFDerivAt (fun xV => getScalarE (Activation.softmaxSpec 0 (ofFnE xV))) (softmaxDerivCLM xV) xV
```
```leanOutput agSpecBridge (whitespace := lax)
@softmaxBackwardSpec_eq_vjp : ∀ {n : ℕ} (x δ : TorchLean.Tensor ℝ [n]),
  getScalarE (Activation.softmaxBackwardSpec 0 x δ) =
    (vjp (fun xV => getScalarE (Activation.softmaxSpec 0 (ofFnE xV))) (getScalarE x))
      (getScalarE δ)
```

The `0` in `softmaxSpec 0` selects the axis. `getScalarE` and `ofFnE` convert between the tensor
and Euclidean vector representations. The second statement identifies the spec backward kernel
with the VJP of the spec forward kernel, including its max-shifted expression.

Over the reals, subtracting the same maximum from all inputs cancels between numerator and
denominator, leaving the softmax function unchanged. This equality allows the derivative to be
transferred even at inputs where the maximum itself is not differentiable. The statement concerns
the kernel over `ℝ`; rounding in an executable scalar type needs a separate approximation proof.

The two-coordinate cancellation proof gives a useful invariant of the softmax JVP. Since the
probabilities sum to one, the tangent of their sum is zero. A perturbation can redistribute mass
between entries but cannot create total probability mass to first order. Separately, adding the
same scalar to every input score leaves softmax unchanged, so a constant input tangent is
annihilated. These are consequences of the derivative formula and useful checks on its indexing;
neither by itself characterizes the full derivative.

The same pattern now reaches three layer-level specifications. For scaled dot-product attention,
`backpropVec_eq_adjoint_fderiv_scaledDotProductAttention` identifies the tape reverse pass on the
attention graph with the adjoint derivative of `Spec.scaledDotProductAttention`, unmasked and with
the canonical scale; the all-true-mask variant rests on `hardMaskedSoftmaxSpec_allTrueMask`, which
shows the masked and unmasked code paths agree when every position is allowed. For LayerNorm,
`outputCLM_evalVec_layerNormGraph` identifies the graph output with `Spec.layerNorm` and
`layerNormJvp_layerNormBackward_adjoint` shows the spec backward rule is adjoint to the spec JVP in
all three cotangents. For BatchNorm, `hasFDerivAt_batchNorm` and `fderiv_batchNorm_eq_batchNormJvp`
prove, for `0 < ε`, that the spec map is differentiable and that its derivative is exactly
`Spec.batchNormJvp`.

```lean (name := layerLevel)
-- These are distinct bridges: forward equality, analytic
-- derivative, and adjointness.
-- The accompanying prose identifies which obligation each
-- declaration discharges.
#check
  @backpropVec_eq_adjoint_fderiv_scaledDotProductAttention
#check @hardMaskedSoftmaxSpec_allTrueMask
#check @outputCLM_evalVec_layerNormGraph
#check @layerNormJvp_layerNormBackward_adjoint
#check @hasFDerivAt_batchNorm
#check @fderiv_batchNorm_eq_batchNormJvp
```

```leanOutput layerLevel (whitespace := lax)
@hardMaskedSoftmaxSpec_allTrueMask : ∀ {nQ nK : ℕ}
  (scores : TorchLean.Tensor ℝ [nQ, nK]),
  Spec.hardMaskedSoftmaxSpec scores (Spec.allTrueMask nQ nK) =
    Activation.softmaxSpec 1 scores
```


The printed mask lemma quantifies over every real score tensor of the two stated dimensions.
Its equality concerns forward values: an all-true Boolean mask permits every entry, so it produces
the ordinary row softmax. No derivative or native execution appears in that statement.

The attention theorem takes a nonempty sequence dimension, the packed query/key/value input,
and an arbitrary output cotangent. Its conclusion identifies the tape's returned cotangent pack
with the adjoint derivative of the unmasked specification at that input. The LayerNorm forward
bridge instead assumes positive row and column counts and concludes equality of the selected
graph output with the spec output. That equality is meaningful independently of differentiability.
The LayerNorm adjoint theorem then pairs input, scale, and bias tangents with their three returned
cotangents; epsilon is an argument, without a positivity hypothesis in that algebraic identity.

The BatchNorm declarations supply an analytic step explicitly. They require a well-formed input
shape and positive epsilon, differentiate jointly in input, scale, and bias, and identify the
resulting derivative applied to a tangent triple with `Spec.batchNormJvp`. Positive epsilon keeps
the variance-plus-epsilon denominator away from zero. Combined with the separate backward
adjoint identity, these facts justify the analytic reverse rule for that specification.


The mask lemma allows substitution of the unmasked function when every mask entry is true.
Other masks need their own semantic and derivative results, including the convention for a fully
blocked row.

The exact names differ by operator, but the contract should not. A runtime rule is not considered
an autograd theorem merely because the code returns a tensor of the right shape. It needs a
mathematical forward function, a derivative statement, and an adjointness statement that lets the
graph theorem compose the local rule with surrounding nodes.

At a ReLU kink, this distinction matters. A theorem phrased as a Fréchet derivative at a point
must either avoid coordinates where
the pre-activation is zero or state a subgradient convention in a different theorem. The current
real-analysis statements take the first route: the hypothesis says where the derivative exists.

# Runtime Autograd Link

The theorem stack above is mathematical. The runtime autograd engine is another object. TorchLean
therefore keeps a link layer between executed reverse graphs and proof graphs:

- {src "NN/Proofs/Autograd/Runtime/Link/Core.lean"}[runtime/autograd link core]
- {src "NN/Proofs/Autograd/Runtime/Link/BackwardGraph.lean"}[backward graph link]
- {src "NN/Proofs/Autograd/Runtime/Link/Accumulation.lean"}[accumulation invariants]

Those files are about representation agreement. They do not prove a new derivative formula. They
say that the runtime graph, its saved forward values, and its accumulation discipline can be read as
the proof-level graph when the required invariants hold. That is the right granularity for auditing
a new runtime node:

- local derivative theorem for the mathematical operator;
- runtime link theorem that the recorded node carries the same forward/VJP structure;
- finite precision or backend theorem if the runtime scalar path is not the ideal real path.

Write an autograd claim by naming each layer of evidence:

```
-- Runtime transfer needs the same forward function and the
-- same selected reverse rule.
ideal VJP theorem
  + runtime graph/link invariant
  + scalar/runtime approximation bridge
  = claim about executed gradients
```

Without the middle bridge, we only have a theorem about the proof graph. Without the last bridge, we
only have a theorem about ideal arithmetic.

# Convolution And BatchNorm

Convolution and BatchNorm are not represented by separate one-, two-, and three-dimensional proof
families. Their spatial shape is a list, so one theorem covers sequences, images, volumes, and
higher-dimensional grids.

For convolution,
{src "NN/Proofs/Autograd/Tape/Ops/Conv/FDeriv.lean"}[the FDeriv proof]
first identifies the exact Fréchet derivative of the forward operation. It then proves that the
three tensors returned by `convBackwardSpec` are the adjoint action of that derivative: a kernel
gradient, a bias gradient, and an input gradient. Positive stride is the geometric hypothesis that
lets the forward and transpose index equations be inverted without ambiguity.

BatchNorm uses the same discipline in the
{src "NN/Proofs/Autograd/Tape/Ops/Norm/BatchNorm.lean"}[BatchNorm proof].
The forward differential includes the dependence of the batch mean and variance on every value in
the channel. `batchNormJvp_batchNormBackward_adjoint` proves that the implemented reverse rule for
the input, learned scale, and learned bias is adjoint to that full differential.

```lean (name := agConvBn)
-- Separate convolution differentiability from the algebraic
-- adjoint identities.
#check @hasFDerivAt_convForwardVec
#check @convJvpSpec_convBackwardSpec_adjoint
#check @batchNormJvp_batchNormBackward_adjoint
```
```leanOutput agConvBn (whitespace := lax)
@hasFDerivAt_convForwardVec : ∀ {d inC outC : ℕ} {kernel stride padding inSpatial :
  TorchLean.Tensor ℕ [d]}
  (state : ConvState inC outC kernel inSpatial), HasFDerivAt convForwardVec (convDerivative
    state) state
```
```leanOutput agConvBn (whitespace := lax)
@convJvpSpec_convBackwardSpec_adjoint : ∀ {d inC outC : ℕ} {kernel stride padding inSpatial :
  TorchLean.Tensor ℕ [d]}
  (layer tangentLayer : Spec.ConvSpec d inC outC kernel stride padding ℝ)
  (input tangentInput : TorchLean.Tensor ℝ (Spec.Shape.ofList (inC :: inSpatial.to (List ℕ))))
  (gradOutput :
    TorchLean.Tensor ℝ (Spec.Shape.ofList (outC :: (Spec.convOutSpatial inSpatial kernel
      stride padding).to (List ℕ)))),
  Spec.Conv.Internal.PositiveStrides (stride.to (List ℕ)) →
    Proofs.TensorAlgebra.dot (Spec.convJvpSpec layer tangentLayer input tangentInput) gradOutput =
      have gradients := Spec.convBackwardSpec layer input gradOutput;
      Proofs.TensorAlgebra.dot tangentLayer.kernel gradients.kernelGradient +
          Proofs.TensorAlgebra.dot tangentLayer.bias gradients.biasGradient +
        Proofs.TensorAlgebra.dot tangentInput gradients.inputGradient
```
```leanOutput agConvBn (whitespace := lax)
@batchNormJvp_batchNormBackward_adjoint : ∀ {channels : ℕ} {sSpatial : Spec.Shape}
  (x tangent gradOutput : TorchLean.Tensor ℝ (Spec.Shape.dim channels sSpatial))
  (gamma dgamma beta dbeta : TorchLean.Tensor ℝ [channels]) (epsilon : optParam ℝ
    TorchLean.normalizationEpsilon)
  [inst : (Spec.Shape.dim channels sSpatial).WellFormed],
  Spec.dot (Spec.batchNormJvp x tangent gamma dgamma beta dbeta epsilon) gradOutput =
    have backward := Spec.batchNormBackward x gamma gradOutput epsilon;
    Spec.dot tangent backward.inputGradient + Spec.dot dgamma backward.scaleGradient +
      Spec.dot dbeta backward.biasGradient
```

The convolution adjoint statement is the clearest picture in this chapter of what "the backward
pass is a transpose" means once a layer has more than one parameter. On the left is a single inner
product of the forward tangent with the output cotangent. On the right is a sum of three inner
products, one per component that `convBackwardSpec` returns, each pairing that component with the
matching piece of the input tangent. A backward rule that got the kernel gradient right and the bias
gradient wrong would fail this equation, for some choice of input tangent and output cotangent.
Requiring the identity for every choice
prevents a local error from passing merely because one test direction hid it, which is what makes
the statement usable as a specification.

Both adjoint identities name the backward result's components. `ConvGradients` supplies
`kernelGradient`, `biasGradient`, and `inputGradient`; BatchNorm's `NormalizationGradients` supplies
input, scale, and bias gradients. Each field is paired with the tangent for the argument it
differentiates, making the correspondence explicit.

`PositiveStrides` appears as a hypothesis rather than as a field of `ConvSpec` on purpose. A zero
stride makes the indexing argument used by this backward implementation non-invertible. A
totalized linear forward map still has an adjoint, but this theorem does not identify the supplied
backward rule with it without positive strides. BatchNorm's printed adjoint identity is algebraic
and has no positivity premise on `epsilon`. The analytic derivative theorem above separately
requires `0 < ε`. An `optParam` supplies a default when the argument is omitted; it establishes no
connection to the epsilon chosen by a particular runtime call.

The joint convolution derivative has three contributions. Perturbing the kernel while holding
the input fixed gives convolution with the kernel tangent. Perturbing the input while holding the
kernel fixed gives convolution with the input tangent. Perturbing the bias contributes its
broadcast tangent. If kernel and input are perturbed simultaneously, their product also has a
term containing both perturbations; that term is second order and does not belong in the
derivative. The analytic theorem justifies this first-order decomposition. The adjoint identity
then reorganizes its three terms into the corresponding returned gradients.

This also explains why input and parameter gradients cannot be certified by examining only the
output shape. They can share valid tensor dimensions while pairing with the wrong tangent or
omitting one contribution. The printed identity tests the whole differential through arbitrary
tangents and output cotangents. Its indexing hypotheses make that pairing argument apply to the
particular backward definitions used here.

# MLP And MSE Gradients

The
{src "NN/Proofs/Autograd/FDeriv/MlpMse.lean"}[MLP/MSE derivative API] turns the abstract autograd
theorem into a familiar training example: a small MLP followed by mean squared error.

The definitions are close by design to what a reader would write on a whiteboard:

- `affineMat` for an affine layer;
- `mlpVecMat` for a two layer MLP with a hidden nonlinearity;
- `mse` and `mseGrad` for the scalar loss;
- gradient lemmas for `W2`, `b2`, `b1`, `x`, and `W1`.

The theorem pattern is:

$$`\operatorname{backpropGradient}
=
\left(D\,\operatorname{loss}\right)^{\!*}(1)`

Equivalently, for parameters $`\theta` and a scalar loss $`L(\theta)=\ell(f_\theta(x),t)`, the
gradient statement has the shape

$$`\nabla_\theta L(\theta)
=
\left(D_\theta f_\theta(x)\right)^{\!*}\nabla_y \ell(y,t).`

For the last layer, the statement is clean. For the hidden ReLU layer, the theorem carries
hypotheses such as "the value before activation is nonzero" at the coordinates being differentiated.
Runtime systems usually leave that condition implicit. PyTorch chooses a subgradient convention at
zero; TorchLean's real analysis statement names the differentiability condition instead.

Read the claim at that level. The theorem proves the ideal real-valued MLP/MSE gradient. A float32
training run connects to that statement through a finite-precision bridge, which belongs to runtime
approximation and to the
{srcDir "NN/Proofs/RuntimeApprox/"}[runtime approximation proof API].

Avoiding zero hidden preactivations is a sufficient condition for composing the displayed ReLU
derivative rules. It need not be a necessary condition for every complete model to be
differentiable: a later computation could cancel a kink. Such a cancellation would need its own
proof of the composite function. The local-rule approach instead uses conditions that can be
checked at each node and then composed uniformly. This explains both the usefulness and the
limits of the nonzero hypotheses in the MLP result.

# Model Coverage: Attention, Transformers, And Recurrent Cells

The model-block APIs compose these local and graph theorems for selected attention, Transformer,
and recurrent fragments. Each result retains the hypotheses of the operations it uses; connecting
a complete model's runtime remains a separate obligation.

Representative theorem entry points:

- {src "NN/Proofs/Autograd/Tape/Ops/Attention/ScaledDotProduct.lean"}[
  NN.Proofs.Autograd.Tape.Ops.Attention.ScaledDotProduct API]
- {src "NN/Proofs/Autograd/Tape/Ops/Attention/MaskedScaledDotProduct.lean"}[masked
  scaled-dot-product attention API]
- {src "NN/Proofs/Autograd/Tape/Ops/Attention/MaskedMultiHeadSelfAttention.lean"}[masked multi-head
  attention API]
- {src "NN/Proofs/Autograd/Tape/Ops/Attention/MultiHeadSelfAttention.lean"}[
  NN.Proofs.Autograd.Tape.Ops.Attention.MultiHeadSelfAttention API]
- {src "NN/Proofs/Autograd/Tape/Ops/Embedding/GatherRows.lean"}[embedding gather-rows API]
- {src "NN/Proofs/Autograd/Tape/Ops/Norm/LayerNorm.lean"}[LayerNorm API]
- {src "NN/Proofs/Autograd/Tape/Ops/Transformer/PostNorm.lean"}[
  NN.Proofs.Autograd.Tape.Ops.Transformer.PostNorm API]
- {src "NN/Proofs/Autograd/Tape/Ops/Transformer/ResidualAttention.lean"}[
  NN.Proofs.Autograd.Tape.Ops.Transformer.ResidualAttention API]
- {src "NN/Proofs/Autograd/Tape/Ops/Transformer/FeedForward.lean"}[
  NN.Proofs.Autograd.Tape.Ops.Transformer.FeedForward API]
- {src "NN/Proofs/Autograd/Tape/Ops/Transformer/EncoderBlock.lean"}[Transformer encoder-block API]
  and
  {src "NN/Proofs/Autograd/Tape/Ops/Transformer/DecoderBlock.lean"}[decoder-block API]
- {src "NN/Proofs/Autograd/Tape/Ops/Recurrent/ElmanCell.lean"}[
  NN.Proofs.Autograd.Tape.Ops.Recurrent.ElmanCell API]

Loss-node proofs are maintained separately for binary cross entropy with logits, cross entropy,
KL divergence, MSE, and NLL under
{srcDir "NN/Proofs/Autograd/Tape/Nodes/Losses"}[`Tape/Nodes/Losses`].
Coverage is theorem-by-theorem: the existence of an operator proof does not imply that every
runtime registration, backend kernel, mask convention, or whole-model unroll has been connected to
it.

For attention, meaning the scaled dot-product form of {Informal.citet transformer2017}[], the named
theorem
`backprop_eq_adjoint_fderiv_scaledDotProduct` states the desired attention theorem directly: the
graph reverse pass for scaled dot product attention agrees with the adjoint derivative of the
forward attention map. The theorem `backprop_eq_adjoint_fderiv_maskedScaledDotProduct` uses a fixed
finite additive
score bias. It does not prove the boolean hard-mask runtime correct; that transfer needs a separate
hard-mask theorem, including its fully blocked-row convention. Multi-head attention and residual
attention then package that structure at a wider interface.

For Transformer post-norm blocks, the post-norm API contains several theorem layers:

- `mhaPostNorm_backpropVec_eq_adjoint_fderiv_at` for residual MHA followed by LayerNorm;
- `seqFfnPostNorm_backpropVec_eq_adjoint_fderiv_at` for residual feed forward followed by LayerNorm;
- `postNorm_backpropVec_eq_adjoint_fderiv_at` for the common post-norm boundary;
- `twoSublayerPostNormBlock_hasFDerivAt` for the analytic composition of two post-norm sublayers;
- named interfaces for residual attention and residual feed-forward post-norm variants.


The recurrent file has results at three scales. `elmanCell_backpropVec_eq_adjoint_fderiv` proves
the reverse-mode theorem for one Elman RNN cell, and `elmanTwoStep_hasFDerivAt` exhibits a short
composition. `elmanUnroll_hasFDerivAt` proves differentiability for an arbitrary list of transition
builders, assuming `elmanTransitionsDifferentiableAt` along the reached trace. The induction is
therefore present. Connecting a concrete sequence runtime still requires identifying its hidden
state threading, sequence indexing, and parameter uses with those transitions; the unroll theorem
does not itself establish that runtime correspondence.


# Training Step Algebra

Backprop correctness is about gradients. Training correctness also needs a clean account of how
those gradients are seeded and consumed. That role belongs to
{src "NN/Proofs/Autograd/Training/StepAlgebra.lean"}[NN.Proofs.Autograd.Training.StepAlgebra API].

There are two pieces to keep separate.

First, `Graph.scalarLoss_grad_correct` specializes the global graph theorem to scalar losses. The
seed is the scalar cotangent `1`, represented by `seedScalarLoss`. Formally, this is
$`loss.backward()` seeding $`d loss / d loss = 1`.

Second, `step` defines the algebra behind a simple optimizer update:

$$`\theta_{t+1} = \theta_t - \eta\,\nabla_\theta L(\theta_t)`

The theorem `step_cons` says the head tensor of the parameter list is updated by exactly that
formula, and `step_nil` handles the empty parameter list. These are compact theorems, but they keep
the training loop from becoming an opaque execution artifact. A realistic optimizer will add
momentum, Adam statistics, clipping, or weight decay; this file gives the simple algebraic core that
those extensions can refine.

The optimizer theory page extends this idea for larger update records. The autograd page should not
be read as proving optimizer convergence. It proves the gradient side of the handoff:

$$`\text{reverse pass returns }(D_\theta L(\theta))^{\ast} 1.`

Optimization theory then decides what an update using that gradient means under step-size,
smoothness, convexity, or backend-certification hypotheses.

A gradient theorem identifies a local sensitivity of the loss. An optimizer step uses that
sensitivity together with its learning rate and retained state. Correctness of the gradient does
not alone imply that the chosen update decreases the loss: that conclusion depends on additional
properties such as smoothness and step size. Keeping the training-step algebra separate lets the
same proved gradient feed several update rules without silently importing a convergence claim
for any of them.

# Autograd Proof Dependencies

An autograd claim is easiest to audit from the graph theorem down to the local derivative rules and
then back up to training. The proof stack has five levels:

1. The
   {src "NN/Proofs/Autograd/Tape/Algebra/Soundness.lean"}[tape algebra soundness API] contains
   `Graph.backprop_correct`.
2. The
   {src "NN/Proofs/Autograd/Tape/Core/FDeriv.lean"}[Fréchet derivative tape API] contains
   `Graph.backpropVec_eq_adjoint_fderiv`.
3. Local derivative files include the
   {src "NN/Proofs/Autograd/FDeriv/Softmax.lean"}[softmax derivative API] and
   {src "NN/Proofs/Autograd/FDeriv/LogSoftmax.lean"}[log-softmax derivative API].
4. Model-block proofs include the
   {src "NN/Proofs/Autograd/Tape/Ops/Transformer/PostNorm.lean"}[Transformer post-norm API].
5. The
   {src "NN/Proofs/Autograd/Training/StepAlgebra.lean"}[training step algebra API] connects
   differentiation to parameter updates.

For a given model, the dependency chain identifies both the local rules used by reverse mode
and the hypotheses under which those rules are derivatives. The runtime link then determines
whether the executed tape has that same meaning.

The next question is numerical rather than differential. We know how the ideal reverse pass is
assembled from local rules; the runtime-approximation chapter asks how far an executable reverse
pass can drift when those rules run with rounded arithmetic.
