import VersoManual
import TorchLean.NN.LoRA
import TorchLean.NN.Autograd.Function
import TorchLean.NN.Autograd.Frontend
import TorchLean.NN.Autograd.Differential
import TorchLean.NN.Autograd.Complex
import TorchLean.NN.CLI.Training.Command
import TorchLean.NN.Data.Synthetic
import TorchLean.NN.Data.CausalLM
import TorchLean.NN.Models.KAN
import TorchLean.NN.Models.CNN
import TorchLean.NN.Models.ResNet
import TorchLean.NN.Models.UNet
import TorchLean.NN.Models.Recurrent
import TorchLean.NN.Models.CausalTransformer.LanguageModel
import TorchLean.NN.Models.Mamba
import TorchLean.NN.Models.MaskedPatchReconstructor
import TorchLean.NN.Models.Diffusion.NoisePredictor
import TorchLean.NN.Models.FNO
import TorchLean.NN.Models.PPO
import TorchLean.NN.Builder.MoE
import TorchLean.NN.Sequential.Training
import TorchLean.NN.RL.CLI
import TorchLean.NN.RL.PPO.ActorCritic
import TorchLean.NN.SelfSupervised
import TorchLean.NN.Text.Vocabulary
import TorchLean.NN.Text.GPT2BPE
import TorchLean.NN.Text.Generation
import TorchLean.NN.Trainer.Constructor
import TorchLean.NN.Trainer.Train
import TorchLean.NN.Data.Image
import TorchLean.NN.Softmax.Log
import TorchLean.NN.Softmax.Correctness
import TorchLean.NN.Tape.Lowering.FDeriv
import TorchLean.NN.Attention.ScaledDotProduct.FDeriv
import TorchLean.NN.Conv.FDeriv
import TorchLean.NN.Norm.BatchNorm
import TorchLean.NN.Norm.LayerNorm
import TorchLean.NN.Norm.LayerNorm.Adjoint
import TorchLean.NN.Attention.HardMask
import TorchLean.NN.Op.Certificates
import TorchLean.NN.Op.Softmax
import TorchLean.NN.Tactic.Autograd
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles
import TorchLean.NN.Loss
import TorchLean.NN.Session.Random
import TorchLean.Verify.Trainer
import TorchLean.NN.Op.Piecewise
import TorchLean.NN.Tactic.OpDeriv

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean

-- The autograd proofs live under the `TorchLean` namespace with one sub-namespace per
-- operator family, and Verso keeps displayed code narrow enough to read beside the text. Opening
-- the namespaces here lets every `#check` below name its theorem the short way. Printed signatures
-- shorten names the same way where the short form is unambiguous.
open TorchLean.Tensor TorchLean.Vec
open TorchLean.Tape.Graph
open TorchLean.Attention
open TorchLean.BatchNorm
open TorchLean.Conv
open TorchLean.LayerNorm
open TorchLean.Softmax

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
small input perturbation (a *tangent*) to a first-order output perturbation, and an output
*cotangent* assigns a scalar sensitivity to those output perturbations. Reverse mode pulls that
cotangent back to the inputs. Under the Euclidean inner products used here, this pullback is the
adjoint of the derivative, represented by the transposed Jacobian (the program doesn't have to
build the Jacobian matrix).

TorchLean proves this connection in layers under
{srcDir "TorchLean/NN/Tape"}[TorchLean/NN/Tape]. First,
local forward and reverse rules satisfy an algebraic pairing law. A graph induction extends that
law to reverse accumulation. Analytic proofs then identify the forward rules with derivatives of
the functions the graph evaluates. Runtime links connect this proof graph to an executed tape.

Here's the proof pipeline:

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

Two instances of the gradient we want to characterize come from the autograd quickstart:

```terminal
# Print tensor and model gradients for the losses discussed
# below.
scripts/lake.sh exe torchlean quickstart_autograd
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

Both halves can be checked by hand, which is the point of starting here. In the first we
differentiate $`x\mapsto\frac{1}{3}\sum_i x_i^2` at $`x=(1,2,3)`. Its value there is $`14/3`, and
its gradient is $`\frac{2}{3}x=(2/3,4/3,2)`, which is what got printed. In PyTorch the same two
lines read

```
# Differentiate the matching scalar mean-square loss in the
# comparison API.
import torch
x = torch.tensor([1., 2., 3.], requires_grad=True)
(x ** 2).mean().backward()
x.grad          # tensor([0.6667, 1.3333, 2.0000])
```

Holding the other two coordinates fixed, we can also write the first coordinate as a scalar
derivative:

```lean
-- Hold the other two squared entries fixed and
-- differentiate one input coordinate.
example (a b c : ℝ) :
    HasDerivAt (fun t : ℝ => (t ^ 2 + b + c) / 3)
      (2 * a / 3) a := by
  autograd
```

The second half of the demo is a single affine layer `y = W x + b` (two inputs, one output, and
squared-error loss), run at `x = [0.5, -1.0]` and `t = [0.25]`. We can recover the weight gradient
from the bias gradient and this input. Here the bias gradient is $`\partial L/\partial b=2(y-t)`,
and the weight gradient is that same scalar times the input, so

$$`\nabla_W L = (2(y-t))\,x^\top = (-1.754865)\cdot(0.5,-1.0) = (-0.877432,\;1.754865),`

which agrees with the displayed row to its precision, and $`L=(y-t)^2\approx0.769887`. (The rounded
printed gradient doesn't recover every bit of the loss.) For that layer the chain rule is one more
`HasDerivAt` proof, this time in a weight, not an input:

```lean
-- Apply the chain rule to the affine prediction and then
-- square the residual.
example (w₁ w₂ b t x₁ x₂ : ℝ) :
    HasDerivAt
      (fun w : ℝ => (w * x₁ + w₂ * x₂ + b - t) ^ 2)
      (2 * (w₁ * x₁ + w₂ * x₂ + b - t) * x₁) w₁ := by
  autograd
```

`autograd` composes mathlib's derivative rules and checks the resulting formula (it doesn't
estimate a gradient from samples). Import `TorchLean.NN.Tactic.Autograd` to use it, and `autograd?`
shows the proof it found. For a named operation, register a proved derivative rule with
`@[autograd]`, or expose an identity inside expressions with `@[autograd simp]`. Domain conditions
remain proof obligations.

That weight example proves one component of the gradient for an affine layer. The graph theorem
assembles such local derivatives and accounts for parameters used along several paths, where the
reverse pass must add their contributions.

# Autograd Proof Obligations

In PyTorch, the default runtime model is approximately:

1. the eager engine records operations into a dynamic tape
2. each operation has a backward rule, either built in or registered by extension code
3. the engine traverses the tape in reverse and accumulates gradients
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
- A scalar loss training step becomes a theorem about the loss seed and parameter update algebra.
- A model block (such as attention or an RNN cell) becomes a packaged theorem with stated
  hypotheses.

These obligations allow the derivative proof to be checked separately from representation and
arithmetic agreement. A proof about a real-valued graph becomes a claim about executed gradients
only after those connections have been supplied.

A simple fanout example explains why local formulas are only the beginning. In a graph computing
$`x\,x`, multiplication receives two references to the same variable. Its two local input
cotangents are both needed, and reverse accumulation adds them at that shared variable to obtain
$`2x` for seed one. Keeping only one contribution would produce a tensor of the right shape with
the wrong value. The graph theorem handles this bookkeeping uniformly, while the local derivative
certificate explains why each of the two contributions was correct in the first place.

# Tape Soundness: The Algebraic Core

{src "TorchLean/NN/Tape/Graph.lean"}[
TorchLean.NN.Tape.Graph API] is the central algebraic file. It defines the proof-carrying nodes and
graphs `LawfulNode` and `Graph` used by the rest of the autograd proofs and proves their soundness
theorem `Graph.jvpCtx_backpropCtx_adjoint`. Their executable data, `NodeData` and `GraphData`, is
defined in {src "TorchLean/NN/TypedGraph/Data.lean"}[`TypedGraph/Data.lean`] over the shared
`TensorPack` and the context indices `Idx` of
{src "TorchLean/NN/TypedGraph/Idx.lean"}[`TypedGraph/Idx.lean`].

Here are the objects to track:

- `TensorPack`: a typed tuple of tensors indexed by a list of shapes. It carries model state,
  inputs, and cotangents without introducing another tensor representation.
- `Idx`: pointer into a context, carrying the needed proof data.
- `NodeData`: forward, JVP, and VJP data for one local operation.
- `LawfulNode`: a node plus the local inner product soundness law.
- `GraphData`: executable graph data built by appending nodes, each referring to earlier values.
- `Graph`: graph whose nodes can share earlier values and carry local proof obligations.
- `Graph.jvpCtx_backpropCtx_adjoint`: global dot product soundness theorem.

For an input context `x`, let `dx` be a tangent of the same shape and let `seed` be a cotangent
for the context returned by graph evaluation. `Graph.jvpCtx_backpropCtx_adjoint` states that
forward sensitivity and reverse accumulation satisfy the following pairing identity:

$$`\langle \operatorname{jvp}_G(x, dx), seed\rangle = \langle dx, \operatorname{backprop}_G(x,
seed)\rangle`

This identity characterizes the adjoint relationship between the supplied JVP and reverse rules.
A forward sensitivity `dx`
pushed through the graph and then paired with an output cotangent gives the same scalar as pairing
`dx` with the cotangent produced by reverse accumulation.

Stated with dot products, the proof obligation is compositional: each node must satisfy its local
adjoint law. Induction over `Graph` then shows that reverse accumulation preserves the pairing
through the whole graph, and the analytic theorem below supplies the further fact that the forward
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

The left-hand side pushes `x` forward and pairs the result with `y`, while the right-hand side
pairs `x` with `y` pulled back. Reading the right-hand side coefficientwise is the clearest
explanation of why a reverse pass is a transpose: the first input cotangent is $`a y_1+c y_2`
(using the first *column* of $`A`), and the second is $`b y_1+d y_2`.
`Graph.jvpCtx_backpropCtx_adjoint` is this statement made compositional, and the induction over
`Graph` does the work that `ring` does here.

At this point the comparison to PyTorch is most direct. PyTorch's engine walks a dynamic graph in
reverse and accumulates cotangents into inputs. TorchLean's algebraic graph does the same
conceptual work, but its graph object carries enough structure for Lean to prove that the
accumulation is sound for every input context in the supported fragment.

# Adjoint Laws And Fréchet Derivatives

By itself, the pairing law doesn't establish that the JVP differentiates the forward function. Two
incorrect rules could still be adjoints of each other. The
{src "TorchLean/NN/Tape/RealGraph/FDeriv.lean"}[Fréchet derivative bridge] supplies the
missing analytic statement: the proposed linear map approximates the change in the forward function
to first order, with a remainder negligible relative to the input perturbation.

That file vectorizes shaped tensor contexts into Euclidean spaces and connects three views:

- shaped tensors: `Tensor Real s`, `TensorPack Real Gamma`
- flat Euclidean vectors: `CtxVec Gamma`, `TorchLean.TensorPack.toVec`, `TorchLean.TensorPack.ofVec`
- analytic derivatives: `HasFDerivAt`, `fderiv`, and `ContinuousLinearMap.adjoint`.

The main theorem is `TorchLean.Tape.Graph.backpropVec_eq_fderiv`, stated for the algebraic graph
at scalar type `Real` with its environment fixed. In plain English:

> If every node in the graph has the stated Fréchet derivative at the values it receives from an
> input, then graph backprop at that input equals the adjoint of the Fréchet derivative of graph
> evaluation there.

That hypothesis is pointwise (`TorchLean.Tape.RealGraph.FDerivCorrectAt`) because ReLU,
normalization, division, logarithms, and square roots all have domain or nondifferentiability
issues. TorchLean states the local smoothness or nonzero conditions needed by the graph being
differentiated. A graph whose nodes are differentiable everywhere
(`TorchLean.Tape.RealGraph.FDerivCorrect`) meets the hypothesis at every input through
`TorchLean.Tape.RealGraph.FDerivCorrect.at`.

Adjointness alone can't identify the derivative of the forward computation. Take the scalar forward
function $`f(x)=x^2` and imagine assigning zero to both its JVP and VJP: their pairings agree for
every tangent and cotangent, so the adjoint identity holds. At $`x=1`, however, the actual
derivative is multiplication by two. What's missing is the connection from the proposed JVP to the
derivative of `f`. Once that connection and adjointness are both established, reverse-mode
correctness follows for this node and can participate in the graph induction.

Flattening the typed context is a mathematical change of representation, not a change in which
variables are differentiated. Each tensor contributes its coordinates to the Euclidean space, and
unflattening recovers the same tensor positions and shapes. That lets Mathlib's derivative and
adjoint APIs apply to a heterogeneous parameter pack. But the ordering must remain consistent on
the tangent and cotangent sides: transposing a matrix representation with one coordinate order and
interpreting its result with another would describe a different pullback. It's the context
conversion lemmas that let the graph theorem move between these representations without leaving
that correspondence informal.

## The Adjoint Identity, Numerically

Let `J` denote the Jacobian of the forward map at the chosen input. A numerical check of the
adjoint identity compares two scalar pairings:

$$`\langle J u,\, w\rangle = \langle u,\, J^\top w\rangle`

for every tangent $`u` at the input and every cotangent $`w` at the output. TorchLean can run both
sides. `autograd.jacfwd` pushes forward once per input coordinate and assembles the Jacobian
tensor, while `autograd.vjp` runs the reverse pass once against a cotangent and returns
$`J^\top w`. Although the traversals differ, they share graph construction, forward operations, and
local derivative definitions, so agreement can miss a shared local-rule error.

We made the map below non-diagonal on purpose: squaring and then multiplying by a mean couples
every output to every input, so a transpose that got an index wrong wouldn't quietly cancel.

```lean (name := apAdj)
section
open TorchLean

-- f(x) = exp(x) * mean(x^2). The mean makes the Jacobian
-- dense: every output reads every input.
def apAdjMap : autograd.Function [3] [3] := fun x => do
  let squared ← nn.functional.square x
  let scaleFactor ← nn.functional.mean squared
  let curved ← Ops.apply .exp (x, ())
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
  let forward := Tensor.dot pushed apAdjCotangent
  -- Reverse side: one backward pass, then one dot product.
  let reverse := Tensor.dot apAdjTangent pullback
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

For this input and this tangent/cotangent pair, the computed scalars satisfy `Float` equality. That
exact agreement depends on these particular binary64 computations. Over ℝ the theorem applies to
every direction under its hypotheses, but it doesn't promise bitwise equality between differently
ordered floating-point sums. A general numerical claim needs local error bounds and a graph-level
approximation theorem (as developed in {ref "runtime-approximation"}[the runtime approximation
chapter]). As a check of indexing and accumulation the run is useful, while shared errors in local
rules can still escape it.

# Derivative Correctness Of The Lowered Tape

The tape-lowering correctness theorem holds for every algebraic graph: its scalar type is an
abstract commutative semiring, and each node may read a non-differentiable environment. It
identifies the lowered tape's reverse pass with the graph's reverse accumulation, and the
derivative theorem above identifies that accumulation at scalar type `Real` and a fixed
environment with the adjoint derivative.

{src "TorchLean/NN/Tape/Lowering/FDeriv.lean"}[
TorchLean.NN.Tape.Lowering.FDeriv] composes the two for one graph and one
environment.

On the algebraic side, the reverse pass returns cotangents for the inputs and every intermediate
value. The analytic theorem needs only the input cotangent. The left half of `Pack.split` selects
that input prefix, and `takeLeft_backpropAllCtx` proves that it equals the inputs-only reverse pass
used by the derivative theorem.

Over `Real`, the public endpoint below connects the dense tape result to the adjoint derivative in
its conclusion (the fixed environment and the differentiability hypothesis are visible in the
signature):

```lean (name := agLink)
-- The derivative premise is local to the input `x`.
#check @backwardDenseFrom_eq_fderiv
```
```leanOutput agLink (whitespace := lax)
@backwardDenseFrom_eq_fderiv : ∀ {Δ : Type} {Γ ss : List TorchLean.Shape}
  (g : TorchLean.Tape.Graph Δ Γ ss)
  (x : TorchLean.TensorPack ℝ Γ) (d0 : Δ) (seed : TorchLean.TensorPack ℝ (Γ ++ ss))
  (hg : TorchLean.Tape.RealGraph.FDerivCorrectAt g.toData d0 x.toVec),
  (g.toData.toTape x d0).1.backwardDenseFrom seed.erase =
      Except.ok (g.toData.backpropAllCtx x d0 seed).erase ∧
    TorchLean.TensorPack.toVec (TorchLean.Pack.split (g.toData.backpropAllCtx x d0 seed)).1 =
      (ContinuousLinearMap.adjoint (fderiv ℝ (g.toData.evalVec d0) x.toVec))
        seed.toVec
```

`TorchLean.Tape.RealGraph.FDerivCorrectAt` needs the node derivative rules only along evaluation at
the given input, so the graph may contain operators whose derivatives exist only on part of their
domain. A graph whose rules hold everywhere (`TorchLean.Tape.RealGraph.FDerivCorrect`) supplies the
hypothesis at every input as `hg.at (TorchLean.TensorPack.toVec x)`. The conclusion is a
conjunction: successful reverse execution and identification of its input cotangent with an analytic
adjoint.

Suppose `g` is an algebraic graph over `Real`, `x` is its typed input context, `d0` is its fixed
environment, and `seed` is a cotangent for the full evaluated context `Γ ++ ss` (including the input
prefix and intermediate values). To differentiate a loss attached only to the final output, put its
seed in that output slot and zeros in the other slots. Because the environment is held fixed, a
parameter to be differentiated must be included in the input context. The theorem returns a
conjunction:

1. compiling `g` and running `Tape.backwardDenseFrom` succeeds with the graph's full reverse
   context
2. the input prefix of that context, after flattening, is
   $`(\operatorname{fderiv}\,\operatorname{eval}(x))^\dagger seed`.

Because it asks for differentiability only at `x`, the theorem applies to graphs containing
piecewise-smooth operators, provided the execution point avoids their non-differentiable or
invalid cases.

These are theorems about the exact tape instantiated over `Real`. A Lean `Float` or CUDA run needs
an additional numerical-refinement argument, which the rounded-runtime chapter develops as a
separate layer (it isn't folded into the exact derivative claim).

For the running example, take
$`forward(x) = softmax(Wx + b)`. The scalar loss supplies an output cotangent
$`seed = dL/dforward`, and the reverse pass returns the input and parameter cotangents
$`dL/dx`, $`dL/dW`, and $`dL/db`.

In this example, `x`, `W`, and `b` must all be part of the differentiated input context to obtain
all three cotangents. Composing the scalar loss seed with the adjoint of the forward map gives the
corresponding derivatives of the loss.

For piecewise operations the point-local variant is particularly useful: it requires the local
derivative conditions at the values reached by this execution, instead of at every possible input.
That can justify an ordinary derivative inside a fixed ReLU region. But it doesn't turn the
selected slope at a kink into a Fréchet derivative there: the analytic premise still has to be true
at the point where the theorem is applied.

# Operator Specs: Softmax And LogSoftmax

Softmax couples every output to every input through its normalization denominator. Following
that dependence gives us both its derivative and the related log-softmax rule. Their APIs are:

- {src "TorchLean/NN/Softmax/FDeriv.lean"}[
  TorchLean.NN.Softmax.FDeriv API]
- {src "TorchLean/NN/Softmax/Log.lean"}[
  TorchLean.NN.Softmax.Log API]

Softmax's API defines `TorchLean.Vec.softmax`, `softmaxDerivCLM`, and `softmaxJVP`. Its theorem
`softmaxJVP_eq_deriv` identifies the implemented JVP formula with the derivative formula, and
`TorchLean.Vec.hasFDerivAt_softmax` states the Fréchet derivative of the vector softmax. And the
theorem `inner_softmaxJVP_comm` packages the self adjoint structure of the softmax Jacobian, which
is why the VJP can reuse the same formula shape.

Log-softmax's API follows the same discipline with `TorchLean.Vec.logSoftmax`,
`TorchLean.Vec.logSoftmaxJVP`, and `logSoftmaxVJP`. Its theorem `logSoftmaxJVP_eq_deriv` gives the
derivative formula, while `inner_logSoftmaxJVP_vjp` states the adjoint relationship between the JVP
and the VJP:

For softmax, if

$$`s_i=\frac{e^{x_i}}{\sum_j e^{x_j}},`

each `s_i` is a normalized exponential. Perturbing one input changes its numerator and also the
shared denominator, so the quotient rule subtracts the `s`-weighted mean of the input direction
from each coordinate before multiplying by `s_i`:

$$`D\,\operatorname{softmax}(x)[dx]_i
=
s_i\left(dx_i-\sum_j s_j dx_j\right).`

For log-softmax, differentiating the outer logarithm cancels the leading `s_i`, and the same
weighted mean remains:

$$`D\,\log\operatorname{softmax}(x)[dx]_i
=
dx_i-\sum_j \operatorname{softmax}(x)_j\,dx_j.`

Both displayed formulas are JVPs. Softmax's Jacobian is symmetric, so its VJP uses the same formula
with the cotangent in place of the tangent. Log-softmax's Jacobian is generally not symmetric: its
VJP subtracts the softmax vector times the sum of the cotangent entries. That distinct reverse rule
is what the adjointness theorem checks against the displayed directional derivative.

Their comments cite the PyTorch API reference for naming alignment, not as proof sources:

- [PyTorch softmax docs](https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html)
- [PyTorch log-softmax
  docs](https://pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html)

Documentation tells us what users expect the op to mean, while Lean proves the derivative law for
TorchLean's mathematical definition.

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

Then the local theorem should say two things, and both are short enough to print in full. `Vec n`
abbreviates `EuclideanSpace ℝ (Fin n)`, the flat vector view for which
{Informal.citet mathlib2020}[] supplies `HasFDerivAt` and `ContinuousLinearMap.adjoint`:

```lean (name := sfx)
-- The JVP is the derivative of the forward denotation.
#check @TorchLean.Vec.hasFDerivAt_softmax
```

```leanOutput sfx
@TorchLean.Vec.hasFDerivAt_softmax : ∀ {n : ℕ} (x : TorchLean.Vec n),
  HasFDerivAt TorchLean.Vec.softmax x.softmaxDerivCLM x
```

In the first signature, `softmaxDerivCLM x` is the continuous linear map proposed as the derivative
at `x`, and `HasFDerivAt` is the proposition identifying it with the first-order behavior of
`TorchLean.Vec.softmax`. Because the theorem accepts every vector `x`, it doesn't ask the caller to
prove differentiability again. That's stronger than merely writing an expression involving `fderiv`,
whose notation alone supplies no proof that a proposed local formula is correct.

```lean (name := lsfx)
-- The VJP is adjoint to the JVP.
#check @inner_logSoftmaxJVP_vjp
```

```leanOutput lsfx (whitespace := lax)
@inner_logSoftmaxJVP_vjp : ∀ {n : ℕ} (x dx δ : TorchLean.Vec n),
  inner ℝ (x.logSoftmaxJVP dx) δ = inner ℝ dx (x.logSoftmaxVJP δ)
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

Each coordinate of the tangent is $`s_i(dx_i-\langle s,dx\rangle)`, and the $`s`-weighted average
of $`dx_i-\langle s,dx\rangle` is zero by construction. Because the softmax Jacobian is symmetric,
its VJP also has zero sum. A backward rule that violates this property can't be the adjoint of the
softmax derivative. Zero sum is an invariant of the rule (the cotangent itself isn't a probability
vector).

Those two theorems are about the analytic vector functions. We define `Tensor.softmax` in the
specification layer on tensors, in the numerically stable max-shifted form, so a third file
(`TorchLean.NN.Softmax.Correctness`) closes the gap: `TorchLean.Tensor.hasFDerivAt_softmax`
transfers differentiability to the spec kernel, and `softmaxBackward_eq_fderiv` proves that the spec
backward rule is the vector-Jacobian product of the spec forward map. Log-softmax has the analogous
pair, with its backward kernel applied to the saved log-softmax output, not the input. Without that
bridge, a theorem about `TorchLean.Vec.softmax` would say nothing about the kernel the runtime
lowers.

```lean (name := agSpecBridge)
-- Connect the stable specification’s forward function and
-- its actual backward formula.
#check @TorchLean.Tensor.hasFDerivAt_softmax
#check @softmaxBackward_eq_fderiv
```
```leanOutput agSpecBridge (whitespace := lax)
@TorchLean.Tensor.hasFDerivAt_softmax : ∀ {n : ℕ} (xV : TorchLean.Vec n),
  HasFDerivAt (fun xV => ((ofFnE xV).softmax ⟨0, Nat.one_pos⟩).getScalarE) xV.softmaxDerivCLM xV
```
```leanOutput agSpecBridge (whitespace := lax)
@softmaxBackward_eq_fderiv : ∀ {n : ℕ} (x δ : TorchLean.Tensor ℝ [n]),
  (x.softmaxBackward δ ⟨0, Nat.one_pos⟩).getScalarE =
    (ContinuousLinearMap.adjoint (fderiv ℝ (fun xV => ((ofFnE xV).softmax ⟨0,
      Nat.one_pos⟩).getScalarE) x.getScalarE))
      δ.getScalarE
```

In `Tensor.softmax` the last argument selects the axis (axis `0` here). `getScalarE` and `ofFnE`
convert between the tensor and Euclidean vector representations. The second statement identifies
the spec backward kernel with the VJP of the spec forward kernel, including its max-shifted
expression.

Over the reals, subtracting the same maximum from all inputs cancels between numerator and
denominator, leaving the softmax function unchanged, and that equality lets the derivative transfer
even at inputs where the maximum itself isn't differentiable. But the statement concerns the kernel
over `ℝ`, and rounding in an executable scalar type needs a separate approximation proof.

From the two-coordinate cancellation proof we get a useful invariant of the softmax JVP: since the
probabilities sum to one, the tangent of their sum is zero. A perturbation can redistribute mass
between entries but can't create total probability mass to first order. Separately, adding the same
scalar to every input score leaves softmax unchanged, so a constant input tangent is annihilated.
Both are consequences of the derivative formula and useful checks on its indexing, but neither by
itself characterizes the full derivative.

We carry the same pattern to three layer-level specifications. For scaled dot-product attention,
`TorchLean.Attention.backprop_eq_fderiv` identifies the tape reverse pass on the
attention graph with the adjoint derivative of `TorchLean.Attention.scaledDotProduct`, unmasked and
with the canonical scale. Its all-true-mask variant rests on `hardMasked_allTrueMask`, which
shows the masked and unmasked code paths agree when every position is allowed. For LayerNorm,
`outputCLM_evalVec_graph` identifies the graph output with `Tensor.layerNorm`, and
`TorchLean.LayerNorm.jvp_backward_adjoint` shows the spec backward rule is adjoint to the spec JVP
in all three cotangents. For BatchNorm, `TorchLean.BatchNorm.hasFDerivAt_forwardVec` and
`fderiv_forwardVec_eq_jvp` prove, for `0 < ε`, that the spec map is differentiable and that its
derivative equals `TorchLean.BatchNorm.jvp`.

```lean (name := layerLevel)
-- These are distinct bridges: forward equality, analytic
-- derivative, and adjointness.
-- The accompanying prose identifies which obligation each
-- declaration discharges.
#check
  @TorchLean.Attention.backprop_eq_fderiv
#check @hardMasked_allTrueMask
#check @outputCLM_evalVec_graph
#check @TorchLean.LayerNorm.jvp_backward_adjoint
#check @TorchLean.BatchNorm.hasFDerivAt_forwardVec
#check @TorchLean.BatchNorm.fderiv_forwardVec_eq_jvp
```

```leanOutput layerLevel (whitespace := lax)
@hardMasked_allTrueMask : ∀ {nQ nK : ℕ} (scores : TorchLean.Tensor ℝ [nQ, nK]),
  hardMasked scores (allTrueMask nQ nK) = scores.softmax ⟨1, ⋯⟩
```


The printed mask lemma quantifies over every real score tensor of the two stated dimensions.
Its equality concerns forward values: an all-true Boolean mask permits every entry, so it produces
the ordinary row softmax. No derivative or native execution appears in that statement.

For attention, the theorem takes a nonempty sequence dimension, the packed query/key/value input,
and an arbitrary output cotangent, and its conclusion identifies the tape's returned cotangent pack
with the adjoint derivative of the unmasked specification at that input. LayerNorm's forward bridge
instead assumes positive row and column counts and concludes equality of the selected graph output
with the spec output (an equality that's meaningful independently of differentiability).
LayerNorm's adjoint theorem then pairs input, scale, and bias tangents with their three returned
cotangents (epsilon is an argument there, without a positivity hypothesis in that algebraic
identity).

BatchNorm's declarations supply an analytic step. They require a well-formed input shape and
positive epsilon, differentiate jointly in input, scale, and bias, and identify the resulting
derivative applied to a tangent triple with `TorchLean.BatchNorm.jvp`. (Positive epsilon keeps the
variance-plus-epsilon denominator away from zero.) Combined with the separate backward adjoint
identity, these facts justify the analytic reverse rule for that specification.


When every mask entry is true, the mask lemma lets us substitute the unmasked function. Other masks
need their own semantic and derivative results, including the convention for a fully blocked row.

Names differ from operator to operator, but the contract shouldn't. A runtime rule isn't considered
an autograd theorem merely because the code returns a tensor of the right shape. It needs a
mathematical forward function, a derivative statement, and an adjointness statement that lets the
graph theorem compose the local rule with surrounding nodes.

On the operator table these statements are attached to the operator's arm of `Op.rule?` (the rule
the runtimes execute). `@[op_cert Op.<operator> <kind>]` on a theorem registers it as a certificate
of that operator. The attribute checks that the statement mentions the arm's rule, and the kind
says what the theorem claims: `algebraic` (the pullback is adjoint to a selected tangent map, with
no claim that it is a derivative), `derivative` (a Fréchet derivative and its adjoint),
`convention` (a rule fixed by an AD convention, as for `detach`), `prepared` (a compact pullback
returns the dense one) or `roundoff` (a floating-point bound). But the attribute doesn't check that
a statement says what its kind says. In {src "TorchLean/NN/Op/Certificates.lean"}[the
certificate file] we state records over the reals about the arms, each as an equation between the
record's operator and the arm, and register them. And
{src "TorchLean/NN/Op/Softmax.lean"}[the softmax certificates] state the derivatives of
`softmax` and `logSoftmax` at every shape and axis. `#op_certs` lists the certificates of one
operator:

```lean (name := agOpCerts)
-- The certificates of the softmax arm.
#op_certs TorchLean.Op.softmax
```
```leanOutput agOpCerts (whitespace := lax)
TorchLean.Op.softmax: [derivative TorchLean.Op.softmax_backward_eq_adjoint,
  algebraic TorchLean.AdjointOp.softmax_op]
```

Because the record concerns vectors, the algebraic certificate states the arm at rank one along
axis `0`, while the derivative certificate states it at every shape and axis.

At a ReLU kink, this distinction matters. A theorem phrased as a Fréchet derivative at a point must
either avoid coordinates where the pre-activation is zero or state a subgradient convention in a
different theorem. The table does both. The derivative certificate of `relu` assumes that no entry
is zero. Its convention certificate states the stored backward at every input: the cotangent where
the entry is positive and zero elsewhere, which at a kink is the cotangent times the left
derivative.

```lean (name := agReluCerts)
-- The certificates of the ReLU arm.
#op_certs TorchLean.Op.relu
```
```leanOutput agReluCerts (whitespace := lax)
TorchLean.Op.relu: [derivative TorchLean.Op.relu_backward_eq_adjoint,
  algebraic TorchLean.AdjointOp.relu_op,
  convention TorchLean.Op.relu_backward_eq]
```

`copysign`, `fmax`, `fmin` and `atan2` have derivative certificates on the exact set where the
derivative exists. `max` and `min` have theirs off ties, and at a tie their convention certificates
give each input half of the cotangent.

# Proving Backward Passes With `op_deriv`

Over the reals, the operators with a fixed number of inputs (every operator but `cat` and `custom`)
have 166 value inputs, and the gradient of each has a `derivative` certificate in all the value
inputs together. The certificate holds at the points its hypotheses name, such as nonzero divisors
for division. `scripts/checks/Library.lean` fails when one of these gradients has no such
certificate. `cat` has `cat_backward_eq_adjoint`.

The tactic `op_deriv` in {src "TorchLean/NN/Tactic/OpDeriv.lean"}[`OpDeriv.lean`] proves
statements of this form for programs outside the table. `fun_prop` composes the `HasPullbackAt`
rules of the operators, the rule set `OpDeriv` supplies the scalar derivatives, and the tactic
compares the backward in the goal with the composed pullback up to the order of products and the
grouping of sums. Here it checks the backward pass of softplus applied entrywise:

```lean (name := agOpDeriv)
section
open TorchLean

theorem apSoftplusPullback {s : Shape}
    (x δ : Tensor ℝ s) :
    DifferentiableAt ℝ
        (fun y : Tensor ℝ s => Tensor.map
          (fun a => Real.log (1 + Real.exp a)) y) x ∧
      δ * Tensor.map
          (fun a => Real.exp a / (1 + Real.exp a)) x =
        (fderiv ℝ (fun y : Tensor ℝ s => Tensor.map
          (fun a => Real.log (1 + Real.exp a)) y) x).adjoint
          δ := by
  op_deriv

#print axioms apSoftplusPullback

end
```
```leanOutput agOpDeriv (whitespace := lax)
'apSoftplusPullback' depends on axioms: [propext, Classical.choice, Quot.sound]
```

The proof goes through Lean's kernel, so a gap in the tactic can make it fail on a true goal but
never prove a false one. A wrong backward pass fails with the equations the tactic couldn't close,
and some true goals fail too, such as some real powers, `y / y` and `fmax y y`. Like the
certificates, these theorems concern real arithmetic and bound no rounding error. For custom tensor
computations, `Program.gradient` derives a gradient program, and `Expr.gradient_eq_adjoint` proves
over the reals that it computes the adjoint of the derivative where the evaluation is regular.

# Runtime Autograd Link

The theorem stack above is mathematical, and the runtime autograd engine is another object.
TorchLean therefore keeps a link layer between executed reverse graphs and proof graphs:

- {src "TorchLean/NN/Tape/Lowering.lean"}[runtime/autograd link core]
- {src "TorchLean/NN/Tape/Lowering/BackwardGraph.lean"}[backward graph link]
- {src "TorchLean/NN/Tape/Lowering/Accumulation.lean"}[accumulation invariants]

Those files are about representation agreement. They do not prove a new derivative formula. They
say that the runtime graph, its saved forward values, and its accumulation discipline can be read as
the proof-level graph when the required invariants hold.

Every registered operator is recorded on the CPU tape through one function,
`TorchLean.Tape.apply`. That function reads the operator's rule from `Op.rule?`, checks the
attributes and input values, prepares the node with the flags of the parents that need a gradient,
and records it. In {src "TorchLean/NN/Tape/Apply/Correctness.lean"}[the tape-apply proofs] we
state once, for every operator, what a successful call records: the rule's forward value at the
next node id, and a backward that sends the rule's pullback to each input that needs a gradient.
`Tape.apply_backward_eq_filter` states that the recorded backward is the backward of the node
prepared with every input flagged, with the contributions to parents that need no gradient removed.
Then `Tape.apply_backwardDenseFrom` states that replacing the recorded backward by that node's
backward leaves the reverse sweep unchanged. Each way the call can fail is a theorem with its error
(such as `Tape.apply_of_rule?_eq_none` when the scalar has no rule for the operator). Together,
these theorems and a certificate of the arm describe the recorded node.

That is the right granularity for auditing a new runtime node:

- local derivative theorem for the mathematical operator
- runtime link theorem that the recorded node carries the same forward/VJP structure
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

Convolution and BatchNorm aren't represented by separate one-, two-, and three-dimensional proof
families. Convolution takes its kernel, stride, padding, and input extents as `Tensor ℕ [d]`, and
BatchNorm takes a spatial `Shape`, so the same theorems cover sequences, images, volumes, and
higher-dimensional grids.

For convolution, {src "TorchLean/NN/Conv/FDeriv.lean"}[the FDeriv proof]
first identifies the exact Fréchet derivative of the forward operation. It then proves that the
three tensors returned by `Conv.backward` (a kernel gradient, a bias gradient, and an input
gradient) are the adjoint action of that derivative. Positive stride is the geometric hypothesis
that lets the forward and transpose index equations be inverted without ambiguity.

BatchNorm uses the same discipline in the
{src "TorchLean/NN/Norm/BatchNorm.lean"}[BatchNorm proof]. Its forward
differential includes the dependence of the batch mean and variance on every value in the channel,
and `TorchLean.BatchNorm.jvp_backward_adjoint` proves that the implemented reverse rule for the
input, learned scale, and learned bias is adjoint to that full differential.

```lean (name := agConvBn)
-- Separate convolution differentiability from the algebraic
-- adjoint identities.
#check @hasFDerivAt_forward
#check @TorchLean.Conv.jvp_backward_adjoint
#check @TorchLean.BatchNorm.jvp_backward_adjoint
```
```leanOutput agConvBn (whitespace := lax)
@hasFDerivAt_forward : ∀ {d inC outC : ℕ} {kernel stride padding inSpatial :
  TorchLean.Tensor ℕ [d]}
  (state : State inC outC kernel inSpatial),
  HasFDerivAt (fun s => { kernel := s.1.1, bias := s.1.2 }.forward s.2)
    (TorchLean.Conv.derivative state) state
```
```leanOutput agConvBn (whitespace := lax)
@TorchLean.Conv.jvp_backward_adjoint : ∀ {d inC outC : ℕ} {kernel stride padding inSpatial :
  TorchLean.Tensor ℕ [d]}
  (layer tangentLayer : TorchLean.Conv d inC outC kernel stride padding ℝ)
  (input tangentInput : TorchLean.Tensor ℝ (inC :: inSpatial.to (List ℕ)))
  (gradOutput :
    TorchLean.Tensor ℝ
        (outC :: (outSpatialDilated inSpatial kernel stride (ones [d]) padding padding).to (List
          ℕ))),
  PositiveStrides (stride.to (List ℕ)) →
    Rep.dot (layer.jvp tangentLayer input tangentInput) gradOutput =
      have gradients := layer.backward input gradOutput;
      Rep.dot tangentLayer.kernel gradients.kernelGradient + Rep.dot tangentLayer.bias
        gradients.biasGradient +
        Rep.dot tangentInput gradients.inputGradient
```
```leanOutput agConvBn (whitespace := lax)
@TorchLean.BatchNorm.jvp_backward_adjoint : ∀ {channels : ℕ} {sSpatial : TorchLean.Shape}
  (x tangent gradOutput : TorchLean.Tensor ℝ (TorchLean.Shape.dim channels sSpatial))
  (gamma dgamma beta dbeta : TorchLean.Tensor ℝ [channels]) (epsilon : optParam ℝ
    TorchLean.normalizationEpsilon)
  [(TorchLean.Shape.dim channels sSpatial).WellFormed],
  Rep.dot (TorchLean.BatchNorm.jvp x tangent gamma dgamma beta dbeta epsilon) gradOutput =
    have backward := TorchLean.BatchNorm.backward x gamma gradOutput epsilon;
    Rep.dot tangent backward.inputGradient + Rep.dot dgamma backward.scaleGradient + Rep.dot dbeta
      backward.biasGradient
```

For a layer with more than one parameter, the convolution adjoint statement is the clearest picture
here of what "the backward pass is a transpose" means. On the left is a single inner product of the
forward tangent with the output cotangent, and on the right is a sum of three inner products (one
per component that `Conv.backward` returns), each pairing that component with the matching piece
of the input tangent. A backward rule that got the kernel gradient right and the bias gradient
wrong would fail this equation, for some choice of input tangent and output cotangent. Requiring
the identity for every choice prevents a local error from passing merely because one test direction
hid it, which is what makes the statement usable as a specification.

Both adjoint identities name the backward result's components. `Conv.Gradients` supplies
`kernelGradient`, `biasGradient`, and `inputGradient`, and BatchNorm's `Normalization.Gradients`
supplies input, scale, and bias gradients. Each field is paired with the tangent for the argument
it differentiates, which makes the correspondence visible.

We made `PositiveStrides` a hypothesis instead of a field of `TorchLean.Conv` on purpose. A zero
stride makes the indexing argument used by this backward implementation non-invertible. A totalized
linear forward map still has an adjoint, but this theorem doesn't identify the supplied backward
rule with it without positive strides. BatchNorm's printed adjoint identity is algebraic and has no
positivity premise on `epsilon`. Separately, the analytic derivative theorem above requires
`0 < ε`. An `optParam` supplies a default when the argument is omitted, but it establishes no
connection to the epsilon chosen by a particular runtime call.

Three contributions make up the joint convolution derivative. Perturbing the kernel while holding
the input fixed gives convolution with the kernel tangent, and perturbing the input while holding
the kernel fixed gives convolution with the input tangent. Perturbing the bias contributes its
broadcast tangent. If kernel and input are perturbed simultaneously, their product also has a term
containing both perturbations, but that term is second order and doesn't belong in the derivative.
The analytic theorem justifies this first-order decomposition, and the adjoint identity then
reorganizes its three terms into the corresponding returned gradients.

# MLP Gradients

In the {src "TorchLean/NN/FDeriv.lean"}[MLP derivative API], the abstract
autograd theorem becomes a familiar example (a two-layer ReLU network over `ℝ`).

We kept the definitions close to what a reader would write on a whiteboard:

- `affine` for an affine layer on Euclidean vectors
- `mlp` for the network Linear, ReLU, Linear, and `mlpDeriv` for its chain-rule derivative
- `hasFDerivAt_mlp`, which shows that `mlpDeriv` is the Fréchet derivative of `mlp` where no
  hidden pre-activation is zero
- `TorchLean.VJPOp.mlp_backward_eq_fderiv`, which equates the spec-level backward pass with the
  adjoint of that derivative.

Here's the theorem pattern:

$$`\operatorname{backward}(x,\delta)
=
\left(D f(x)\right)^{\!*}\delta`

For parameters $`\theta` and a scalar loss $`L(\theta)=\ell(f_\theta(x),t)`, the chain rule gives
the same shape for the parameter gradient:

$$`\nabla_\theta L(\theta)
=
\left(D_\theta f_\theta(x)\right)^{\!*}\nabla_y \ell(y,t).`

The theorem carries the hypothesis that the value before activation is nonzero at every hidden
coordinate. Runtime systems usually leave that condition implicit. PyTorch chooses a subgradient
convention at zero. TorchLean's real analysis statement names the differentiability condition
instead.

Read the claim at that level: the theorem proves the ideal real-valued MLP gradient. A float32
training run connects to that statement through a finite-precision bridge, which belongs to runtime
approximation and to the {srcDir "TorchLean/Verify/RuntimeApprox/"}[runtime approximation proof
API].

Avoiding zero hidden preactivations is a sufficient condition for composing the displayed ReLU
derivative rules. It need not be a necessary condition for every complete model to be
differentiable (a later computation could cancel a kink), and such a cancellation would need its
own proof of the composite function. Instead, the local-rule approach uses conditions that can be
checked at each node and then composed uniformly. This explains both the usefulness and the limits
of the nonzero hypotheses in the MLP result.

# Model Coverage: Attention, Transformers, And Recurrent Cells

For selected attention, Transformer, and recurrent fragments, the model-block APIs compose these
local and graph theorems. Each result retains the hypotheses of the operations it uses, and
connecting a complete model's runtime remains a separate obligation.

Representative theorem entry points:

- {src "TorchLean/NN/Attention/ScaledDotProduct.lean"}[
  TorchLean.NN.Attention.ScaledDotProduct API]
- {src "TorchLean/NN/Attention/BiasedScaledDotProduct.lean"}[biased
  scaled-dot-product attention API]
- {src "TorchLean/NN/Attention/BiasedMultiHead.lean"}[biased
  multi-head attention API]
- {src "TorchLean/NN/Attention/MultiHead.lean"}[
  TorchLean.NN.Attention.MultiHead API]
- {src "TorchLean/NN/Norm/LayerNorm.lean"}[LayerNorm API]
- {src "TorchLean/NN/Models/Transformer/PostNorm.lean"}[
  TorchLean.NN.Models.Transformer.PostNorm API]
- {src "TorchLean/NN/Models/Transformer/ResidualAttention.lean"}[
  TorchLean.NN.Models.Transformer.ResidualAttention API]
- {src "TorchLean/NN/Models/Transformer/FeedForward.lean"}[
  TorchLean.NN.Models.Transformer.FeedForward API]
- {src "TorchLean/NN/Models/Transformer/EncoderBlock.lean"}[Transformer
  encoder-block API] and
  {src "TorchLean/NN/Models/Transformer/DecoderBlock.lean"}[decoder-block API]
- {src "TorchLean/NN/Models/Recurrent/Elman.lean"}[
  TorchLean.NN.Models.Recurrent.Elman API]

Coverage is theorem-by-theorem: the existence of an operator proof does not imply that every
runtime registration, backend kernel, mask convention, or whole-model unroll has been connected to
it.

For attention, meaning the scaled dot-product form of {Informal.citet transformer2017}[], the graph
`scaledDotProductDGraph` carries its node certificates, so
`TorchLean.Tape.Graph.backpropVec_eq_fderiv` with the bundled proof read at the input,
`(scaledDotProductDGraph c).hg.at xV`, identifies its reverse pass with the adjoint derivative of
the graph's full forward evaluation. The spec bridge above restricts the seed to the output block
and identifies that map with `TorchLean.Attention.scaledDotProduct`. The graph
`biasedScaledDotProductDGraph` uses a fixed finite additive score bias. It does not prove the
boolean hard-mask runtime correct. That transfer needs a separate hard-mask theorem, including its
fully blocked-row convention. The multi-head graph in the `MultiHeadAttention` namespace uses
a direct reshape from `[n, numHeads * headDim]` to `[numHeads, n, headDim]`. The executable
specification instead reshapes to `[n, numHeads, headDim]` and permutes the token and head axes.
These layouts differ in general. The multi-head and dependent residual-attention graph theorems
therefore characterize the declared proof graph. They do not establish its equality with the
executable multi-head attention function.

For Transformer post-norm blocks, the post-norm API contains several theorem layers:

- `mhaPostNorm_backprop_eq_fderiv` for residual MHA followed by LayerNorm
- `seqFFNPostNorm_backprop_eq_fderiv` for residual feed forward followed by LayerNorm
- `TorchLean.LayerNorm.backprop_eq_fderiv` for the common post-norm boundary
- `TorchLean.LayerNorm.graphFDerivAt`, the certificate with which
  `TorchLean.Tape.GraphData.hasFDerivAt_evalVec_comp` composes a differentiable residual map with
  post-norm, once per sublayer
- named interfaces for residual attention and residual feed-forward post-norm variants.


The recurrent file has results at two scales. `RNN.cellDGraph` gives the reverse-mode theorem for
one Elman RNN cell through `TorchLean.Tape.Graph.backpropVec_eq_fderiv` and the bundled proof read
at the input, `(RNN.cellDGraph cell).hg.at xV`. `RNN.differentiableAt_unroll` proves
differentiability for an arbitrary list of transition builders, assuming
`RNN.TransitionsDifferentiableAt` along the reached trace. It gives no derivative formula or
backward pass for the unroll. Connecting a concrete sequence runtime still requires identifying its
hidden state threading, sequence indexing, and parameter uses with those transitions. The unroll
theorem does not itself establish that runtime correspondence.


# Training Step Algebra

Backprop correctness is about gradients. Training correctness also needs a clean account of how
those gradients are seeded and consumed.

There are two pieces to keep separate.

First, scalar losses are an application of `TorchLean.Tape.Graph.jvpCtx_backpropCtx_adjoint` (the
algebraic pairing law) to the seed that is zero on every non-output cotangent and the scalar
cotangent `1` on the loss, corresponding to $`loss.backward()` seeding $`d loss / d loss = 1`.
Identifying the supplied JVP with a derivative still requires the analytic bridge above.

Second, `TorchLean.SGD.update` in the {src "TorchLean/NN/Optimizer/Update.lean"}[optimizer API] is
the simple optimizer update on one parameter tensor:

$$`\theta_{t+1} = \theta_t - \eta\,\nabla_\theta L(\theta_t)`

It returns `parameters - Tensor.scale gradients state.lr`, which is that formula with the learning
rate multiplying each gradient entry on the right. Stating the update as a pure function keeps the
training loop from becoming an opaque execution artifact. A realistic optimizer adds momentum, Adam
statistics, clipping, or weight decay, and the same file defines momentum SGD, Adam, AdamW and the
other updates beside it.

For larger update records, the optimizer theory page extends this idea. This page doesn't prove
optimizer convergence. It proves the gradient side of the handoff:

$$`\text{reverse pass returns }(D_\theta L(\theta))^{\ast} 1.`

Optimization theory then decides what an update using that gradient means under step-size,
smoothness, convexity, or backend-certification hypotheses.

A gradient theorem identifies a local sensitivity of the loss. An optimizer step uses that
sensitivity together with its learning rate and retained state. Correctness of the gradient doesn't
alone imply that the chosen update decreases the loss: that conclusion depends on additional
properties (such as smoothness and step size). Keeping the training-step algebra separate lets the
same proved gradient feed several update rules without silently importing a convergence claim for
any of them.

# Autograd Proof Dependencies

An autograd claim is easiest to audit from the graph theorem down to the local derivative rules and
then back up to training. There are four levels in the proof stack:

1. The
   {src "TorchLean/NN/Tape/Graph.lean"}[tape algebra soundness API]
   contains `TorchLean.Tape.Graph.jvpCtx_backpropCtx_adjoint`.
2. The
   {src "TorchLean/NN/Tape/RealGraph/FDeriv.lean"}[Fréchet derivative tape API] contains
   `TorchLean.Tape.Graph.backpropVec_eq_fderiv`.
3. Local derivative files include the
   {src "TorchLean/NN/Softmax/FDeriv.lean"}[softmax derivative API] and
   {src "TorchLean/NN/Softmax/Log.lean"}[log-softmax derivative API].
4. Model-block proofs include the
   {src "TorchLean/NN/Models/Transformer/PostNorm.lean"}[Transformer post-norm
   API].

Training then consumes the gradient in `TorchLean.SGD.update` and the other updates of the
{src "TorchLean/NN/Optimizer/Update.lean"}[optimizer API].

For a given model, the dependency chain identifies both the local rules used by reverse mode and
the hypotheses under which those rules are derivatives, and the runtime link then determines
whether the executed tape has that same meaning.

So the next question is numerical, not differential. We know how the ideal reverse pass is
assembled from local rules, and the runtime-approximation chapter asks how far an executable
reverse pass can drift when those rules run with rounded arithmetic.
