import VersoManual
import NN.Verification.Builtin.Proved.Correctness.Eval.Main
import NN.IR
import NN.Tensor
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean

-- The live example below builds an IR graph and evaluates it, so this chapter needs the IR
-- namespace, the tensor spec layer, and the verified lowering theorem it quotes.
open TorchLean
open Spec
open NN.IR
open NN.Verification.Builtin.Proved.Correctness

-- Two signatures are printed below at Lean's own line width rather than this file's, so their
-- `leanOutput` blocks are wrapped by hand and compared with whitespace collapsed.
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Proof Systems" =>
%%%
tag := "proof-systems-beyond-bounds"
file := "Proof-Systems-Beyond-Bounds"
%%%

Imagine that lowering accidentally drops the bias from a linear layer. The resulting forward graph
may still be well shaped, execute without an exception, and even receive a convincing interval
certificate. The certificate would then describe the wrong function. A lowering-correctness
relation is what prevents that quiet change of subject.

For the classifier from the previous chapter, this would undermine the entire margin argument.
Suppose we prove that output zero of the lowered graph stays above output one on an input square.
To conclude that the classifier keeps its label, we must know that those graph outputs are the
classifier's scores with its actual trained parameters. Equal tensor dimensions only tell us
that the scores can be compared. They do not tell us that the right function produced them.

This is why a verification library needs more than a bound algorithm. Every translation creates
another place where meaning has to be preserved: a source layer becomes graph nodes, a parameter
record becomes a payload, and graph nodes become executable operations. A proof names the two
interpretations at one such boundary and explains why they agree. We can then compose that
agreement with other results without pretending that one theorem covers the whole stack.

Differential testing has found silent wrong-answer bugs as well as crashes in deep learning
compilers; NNSmith reported 72 new bugs across TVM, TensorRT, ONNXRuntime and PyTorch
{Informal.citep nnsmith2023}[]. For verification, a wrong-answer bug can leave downstream checks
internally consistent while changing the function they describe.

The same pattern reappears elsewhere. IBP and CROWN relate boxes or affine forms to graph values;
autograd relates a reverse rule to the derivative of the forward map; runtime approximation relates
rounded execution to ideal arithmetic. These are different proof systems, but each begins by
naming two objects and the relation that is supposed to connect them.

For a three-node affine-ReLU graph, we can inspect the value table directly, including the
pre-activation that ReLU partly erases. The lowering theorem then says that the graph a verifier
bounds computes its source program, for every input.

# Proof Obligations As Relations

The common structure is relational. Let `S` be a semantic object, `A` an executable or imported
artifact, and `R S A` the proposition that the artifact represents the semantics correctly. A
lowering proof establishes `R` for the executable graph it produces. A certificate checker parses
an untrusted artifact and returns evidence from which `R` follows. A backend contract records `R`
as an assumption when the implementation remains outside the proved fragment.

For a checker

$$`\operatorname{check}:A\to\operatorname{Except}(\mathrm{Error},W),`

the soundness theorem must connect successful checking to the intended relation:

$$`\operatorname{check}(a)=\operatorname{ok}(w)
\quad\Longrightarrow\quad
R(S,\operatorname{decode}(a),w).`

This direction matters. The producer may be Python, CUDA, α,β-CROWN, or a remote solver; none of
those programs must be trusted merely because the checker accepts their output. Trust moves to the
Lean definition of `R`, the checker, and its soundness theorem. When no such theorem exists, the
artifact is evidence or an explicit boundary, not a certificate by vocabulary alone.

The relation `R` is where the intended meaning enters. For lowering, it may be equality of
evaluated value tables. For intervals, it is containment of every semantic value in its box.
For numerical approximation, it is an inequality bounding the distance between two results.
These conclusions allow different substitutions. Equality lets us replace one value by another
without loss. An error bound spends part of a margin. Containment may be sufficient to prove an
inequality even when the exact value is unknown. Choosing the relation comes before choosing
which proof or checker to run.

The idea is old enough to have a name. Proof-carrying code {Informal.citep necula1997}[] made the
same move for machine code: an untrusted producer ships a proof alongside the artifact, and the host
runs a small trusted checker rather than trusting the compiler. For a tensor program, the semantic
object includes shapes, axis conventions, and a scalar interpretation. The relation must preserve
those choices through each translation.

# IR Lowering Correctness

The verifier bounds an `NN.IR.Graph`, so the translation it depends on is the lowering of a
TorchLean program into that graph. For the typed first-order fragment, Lean prints the
preservation theorem as:

```lean (name := psThm)
-- Lowering a typed forward program and running its IR
-- gives the source program's result.
#check @runForwardIR_eq_evalForward
```

```leanOutput psThm (whitespace := lax)
@runForwardIR_eq_evalForward : ∀ {α : Type} [inst : Storage α]
  [inst_1 : Context α] {paramShapes : List Shape}
  {inShape outShape : Shape}
  (p : NN.Verification.Builtin.Proved.ForwardProgram α
    paramShapes inShape outShape)
  (params : TensorPack α paramShapes) (x : Tensor α inShape),
  NN.Verification.Builtin.runForwardIR
      (NN.Verification.Builtin.Proved.lowerForwardProgramToIR
        p params) x =
    NN.Verification.Builtin.Proved.evalForward p params x
```

The declaration lives in the
{src "NN/Verification/Builtin/Proved/Correctness/Eval/Main.lean"}[verified lowering correctness
API].

In plain English:

> Lower a typed forward program and its parameters to an IR graph, and run that graph on any
> input. The result, success or error, is the result of evaluating the source program on the same
> input.

The theorem connects three objects: `evalForward`, the evaluator of the source language;
`lowerForwardProgramToIR`, the lowering function; and `runForwardIR`, which runs the lowered graph
by the IR semantics `Graph.denoteAll`. It has no success premise and no excluded operator. Both
sides are `Except` values, so a failure on one side is the same failure on the other, with the
same message.

The leading `α` is the scalar type. The `Storage α` and `Context α` instances supply the tensor
representation and scalar operations used on both sides. This generality is useful: preservation
of the program structure can be stated without choosing reals or floats. It does not compare
different scalar types. Instantiating the theorem at `Float` compares two Lean interpretations
using that same `Float` context; a real-to-float approximation is a separate relation.

In theorem notation, the statement has the shape:

$$`\forall p\;\theta\;x,\quad
\operatorname{runForwardIR}(\operatorname{lower}(p,\theta),x)
=
\operatorname{evalForward}(p,\theta,x)`

The proof is an induction over the let-chain a source program denotes. Each node's lowering is
matched with the one-node agreement lemma for its operator, and
`denoteAllFrom_lowerForwardLetChain_eq_evalForwardLetChainVals` stitches the steps together. That
lemma compares the full value table, not only the output, which is what the induction needs:
a later node reads earlier values.

A new lowering case must fit this same preservation statement. Its local proof needs to account
for parameter lookup, output shape, and any failure behavior before the recursive argument can
extend the equivalence to the new node.

The example below makes a value table concrete, including a pre-activation that ReLU will partly
erase.

# Affine-ReLU Example

Return to the bias-dropping bug from the opening, and make it concrete. Take the graph

$$`y=\operatorname{ReLU}(Wx+b),\qquad
W=\begin{pmatrix}1&2\\3&4\end{pmatrix},\qquad
b=\begin{pmatrix}0\\-30\end{pmatrix}.`

The bias is large and negative on purpose: it decides whether the second ReLU unit fires at all, so
dropping it is not a small numerical difference, it changes which unit is active.

Three nodes, written out as the IR sees them:

```lean
-- The affine node reads its weights and bias from payload
-- entry one.
def psGraph : Graph :=
  { nodes :=
      #[ { id := 0, parents := #[], kind := .input,
           outShape := [2] }
       , { id := 1, parents := #[0], kind := .linear,
           outShape := [2] }
       , { id := 2, parents := #[1], kind := .relu,
           outShape := [2] } ] }
```

Each node carries its id, its parents, its operation tag, and its declared output shape. The weights
are not in the graph: `linear` nodes read their parameters from a *payload* keyed by node id, which
is what lets the same graph be reused at different scalar types and lets a verifier swap in a
parameter store without rebuilding the structure.

```lean
-- Two payloads share the weight matrix and differ only in
-- the bias.
def psW : Tensor Float [2, 2] := [[1, 2], [3, 4]]

def psPayload (b : Tensor Float [2]) : Payload Float :=
  { linear? := fun id =>
      if id = 1 then
        some { outDim := 2, inDim := 2, W := psW, b := b }
      else
        none }

def psBias : Tensor Float [2] := [0, -30]
def psNoBias : Tensor Float [2] := [0, 0]
def psX : Tensor Float [2] := [1, 1]
```

The IR semantics evaluates node `0` to the input, reads `W` and `b` from the payload at node `1`
and computes $`Wx+b`, and applies ReLU at node `2`. The `denote` entry point evaluates the whole
table and projects one node id, and `expectShape` turns the shape-erased result back into a typed
tensor:

```lean
-- Evaluate the tagged IR and retrieve node two as a
-- length-two vector.
def psOut (b : Tensor Float [2]) :
    Except String (Tensor Float [2]) := do
  let out ←
    Graph.denote (α := Float) (g := psGraph)
      (payload := psPayload b)
      (input := SomeTensor.ofTensor psX)
      (outputId := 2)
  Graph.expectShape (α := Float) (expected := [2]) out
```

Run it with the bias and then with the bias dropped:

```lean (name := psRun)
-- First the intended bias, then a payload without it.
#eval psOut psBias
#eval psOut psNoBias
```

```leanOutput psRun
Except.ok [3.000000, 0.000000]
```

```leanOutput psRun
Except.ok [3.000000, 7.000000]
```

Check the arithmetic by hand. $`Wx=(1+2,\;3+4)^\top=(3,7)^\top`. Adding the bias gives
$`(3,-23)^\top`, and ReLU clamps the second coordinate to zero, so the answer is $`(3,0)^\top`.
Dropping the bias leaves $`(3,7)^\top`, and ReLU changes nothing. The second unit went from dead to
firing at $`7`.

A faulty lowering would produce the second payload from a source that has the first. The result
$`(3,7)` would still have the expected shape and finite entries, so shape and finiteness checks
alone would not reveal the change. The lowering theorem rules it out for every input: what the
graph computes is what the source computes. We changed the payload by hand here; this demonstrates
the effect of a missing bias, not a bug in the lowering.

## Value Tables

The theorem's proof equates value tables, and this graph shows why that is stronger than equating
outputs. Ask for every node instead of node `2`:

```lean
-- Retain all node values so that a final ReLU cannot hide
-- an intermediate difference.
def psTable (b : Tensor Float [2]) :
    Except String (Array String) := do
  let vals ←
    Graph.denoteAll (α := Float) (g := psGraph)
      (payload := psPayload b)
      (input := SomeTensor.ofTensor psX)
  pure (vals.map (fun v => s!"{v.tensor}"))
```

```lean (name := psTables)
-- Input, affine pre-activation, and ReLU output in node
-- order.
#eval psTable psBias
```

```leanOutput psTables (whitespace := lax)
Except.ok #["[1.000000, 1.000000]",
  "[3.000000, -23.000000]",
  "[3.000000, 0.000000]"]
```

The middle entry is the pre-activation $`Wx+b=(3,-23)^\top`; ReLU erases its negative second
coordinate. A pre-activation of $`(3,-100)^\top` would produce the same final output at this input,
so an output comparison here would miss the difference. The table comparison retains it.

The recorded PyTorch comparison computes the same four numbers {Informal.citep pytorch2019}[]:

```
# Compare eager and compiled evaluation with the same
# weights, bias, and input.
import torch
W = torch.tensor([[1., 2.], [3., 4.]])
b = torch.tensor([0., -30.])
x = torch.tensor([1., 1.])
def f(x, b): return torch.relu(x @ W.T + b)
compiled = torch.compile(f)
print("with bias          ", f(x, b).tolist())
print("bias dropped       ", f(x, torch.zeros_like(b)).tolist())
print("torch.compile      ", compiled(x, b).tolist())
print("eager == compiled  ", torch.equal(f(x, b), compiled(x, b)))
```

```
with bias           [3.0, 0.0]
bias dropped        [3.0, 7.0]
torch.compile       [3.0, 0.0]
eager == compiled   True
```

In that recorded comparison, `torch.compile` and eager execution agree on this input. The lowering
theorem makes the corresponding claim about TorchLean's source and IR for every input.

The declared shapes are checked too. Shape inference accepts the graph:

```lean (name := psShapes)
-- Infer every node's shape from its parents and compare
-- it with the declaration.
#eval psGraph.checkShapes
```

```leanOutput psShapes
Except.ok ()
```

`checkShapes_sound` turns that acceptance into a statement about every evaluation: whenever
`denoteAll` succeeds on this graph, each computed value has its node's declared shape. It assumes
the evaluation succeeded; it does not prove that it does.

# Logarithm Domain

A graph with one raw logarithm shows what the IR semantics does outside an operator's domain:

```lean
-- One raw logarithm after the input.
def psLogGraph : Graph :=
  { nodes :=
      #[ { id := 0, parents := #[], kind := .input,
           outShape := [2] }
       , { id := 1, parents := #[0], kind := .log,
           outShape := [2] } ] }

def psLog (x : Tensor Float [2]) :
    Except String (Tensor Float [2]) := do
  let out ←
    Graph.denote (α := Float) (g := psLogGraph)
      (payload := {}) (input := SomeTensor.ofTensor x)
      (outputId := 1)
  Graph.expectShape (α := Float) (expected := [2]) out
```

```lean (name := psLogRuns)
-- A positive input, then one with a negative coordinate.
#eval psLog [1, 2]
#eval psLog [-1, 2]
```

```leanOutput psLogRuns
Except.ok [0.000000, 0.693147]
```

```leanOutput psLogRuns (whitespace := lax)
Except.error "IR eval: log: input contains values <= 0
  (or NaN); use `safe_log`, which is
  log(softplus(x) + eps)"
```

The IR semantics is *partial*: the positive-input domain excludes $`-1`, so the evaluator refuses
and says which operator refused and what to use instead. PyTorch takes the total route and returns
a NaN:

```
# PyTorch returns a tensor containing NaN for the negative
# coordinate.
print(torch.log(torch.tensor([-1., 2.])))
```

```
tensor([nan, 0.6931])
```

Both behaviors are defensible. Mathlib's real logarithm is itself totalized, so neither policy is
simply "the mathematical one", and the protected `safeLogSpec` computes `log(softplus(x) + ε)`, a
different, domain-protected function. What matters for a preservation theorem is that the two
results are observably different: one is an error, the other a successful tensor whose first
coordinate is NaN, and a caller can branch on that. This is why the lowering theorem compares
`Except` results rather than tensors. A lowering that replaced a failure with a conveniently shaped
tensor would change the contract, and the theorem would no longer hold.

# Graph Specification And Runtime Example

The `graphspec` example takes a different path: it converts a typed `GraphSpec.Chain` to an
`nn.Sequential` using `ToSequential.toSeq`, then trains that model. It does not lower to the IR or
instantiate the preservation theorem above.

```terminal
# Run the maintained example that builds, lowers, and trains
# the GraphSpec model.
./scripts/lake.sh exe torchlean graphspec
```

An abridged summary of the seeded example is shown below; this is not a verbatim runtime log:

```
Sequential: [2] -> [1], layers=3, params=13
  [0] Linear(2, 3): [2] -> [3] params=9
  [1] ReLU: [3] -> [3] params=0
  [2] Linear(3, 1): [3] -> [1] params=4
mean_loss(before) = 1.239197
mean_loss(after) = 0.247518
forward: GraphSpec MLP lowered to TorchLean and executed
```

The recorded summary reports a completed execution and a loss decrease on that run. It is runtime
evidence for the GraphSpec-to-sequential path, not an application of the canonical-IR theorem.
To connect this training run to a mathematical model, we need results about the same conversion,
parameters, and runtime used here.

The parameter counts help identify the model in that log. The first linear layer has six
weights and three biases, giving nine parameters. ReLU adds none. The last linear layer has
three weights and one bias, giving four, so the model has thirteen parameters in total.
The printed shapes describe the interfaces between those layers. They are useful checks that
we are discussing the intended architecture before reading the two loss values; the loss values
then describe this run with its particular parameters and data.

# Unsupported Operators And Shape Checks

The axis-operator tutorial uses the shared checked forward-graph evaluator:

```terminal
# Exercise the checked evaluator on the axis-operation
# examples.
./scripts/lake.sh exe torchlean ir_axis_ops
```

For concatenation on the middle axis, it prints the output shape and leading values:

```
[concat_middle_axis] output shape: [2, 8, 4]
```

If shape validation or evaluation fails, the example reports the error.

This distinction matters. The semantic language can describe more
programs than a particular lowering theorem or runtime path currently covers. Returning an array
of the expected shape would not establish correctness for an unsupported operation; the lowering
path must either justify that operation or reject it.

The decidable regression checks in
{src "NN/Tests/IR/ShapeContracts.lean"}[`IR.ShapeContracts`]
check `Infer.nodeOutShape` on malformed convolution, broadcast, and layer-normalization nodes.
They concern shape inference rather than evaluation. The meaning-preservation result for
lowering is `runForwardIR_eq_evalForward`.

# Proof Composition For A Training Step

Consider a claim about one training step on a Float32 backend. No single theorem should be expected
to prove the entire statement. The proof is assembled from relations:

$$`
\begin{aligned}
\text{source model}
&\equiv \text{IR denotation},\\
\text{graph VJP}
&= (D\,\text{forward})^\ast,\\
\text{rounded execution}
&\approx_\varepsilon \text{real execution},\\
\text{optimizer update}
&= \theta-\eta g.
\end{aligned}`

Each line has its own hypotheses and failure modes:

:::table +header
*
  * Relation
  * Typical obligation
*
  * source to IR
  * lowering covers every source constructor used
*
  * VJP to derivative
  * local backward laws and analytic domain conditions
*
  * rounded to real
  * finite values and an explicit error budget
*
  * gradient to update
  * optimizer state and update equation match the intended algorithm
:::

A theorem about the whole workflow composes these relations. A report about a workflow should say
which rows are proved, which were checked for one artifact, and which cross a named backend
boundary.

The optimizer row deserves the same precision as the graph rows. The equation
$`\theta_{\mathrm{new}}=\theta-\eta g` describes a plain gradient step. Momentum and Adam
carry additional state, so their update relation must include that state and its recurrence.
Even for plain gradient descent, the symbol `g` must refer to the gradient of the stated loss
at the stated parameters. An algebraically correct subtraction cannot compensate for a gradient
computed using another reduction convention, parameter order, or forward function.

# Transitivity Of Semantic Relations

Suppose source lowering relates a model `M` to an IR graph `G`, runtime lowering relates `G` to an
executable graph `E`, and a numerical theorem bounds `E_float` against the real denotation of `E`.
The end-to-end argument is ordinary transitivity, but each intermediate term must be the same
mathematical object:

$$`\begin{aligned}
\llbracket M\rrbracket_{\mathbb R}
&=\llbracket G\rrbracket_{\mathbb R},\\
\llbracket G\rrbracket_{\mathbb R}
&=\llbracket E\rrbracket_{\mathbb R},\\
\left\|\llbracket E\rrbracket_{\mathrm{float}}(x)
 -\llbracket E\rrbracket_{\mathbb R}(x)\right\|
&\le\varepsilon(x).
\end{aligned}`

Therefore

$$`\left\|\llbracket E\rrbracket_{\mathrm{float}}(x)
-\llbracket M\rrbracket_{\mathbb R}(x)\right\|
\le\varepsilon(x).`

For the substitution to apply, the intermediate denotations must use the same parameters, masks,
axes, and scalar interpretation. A theorem about a different payload or mask convention cannot
supply the middle equality, even if its input and output tensor shapes match.

The bias example above is the smallest instance of the failure. If lowering drops `b`, then
$`\llbracket G\rrbracket` and $`\llbracket E\rrbracket` are different functions, the second equation
is false, and the chain still *looks* fine because every term in it has the right type. Only an
equation between denotations catches it.

An approximation claim also has a domain. If its error estimate holds only while intermediates
remain finite or inputs stay in a box, those assumptions must survive the substitutions above.
For a classifier, the resulting output-error bounds are charged against the verified score
margin. For a regression model, they may be compared with a requested output tolerance. The same
lowering equality can support both uses, but the numerical budget and final property must be
chosen for the application.

# Related Systems

Executable semantics, proof-producing tools, and differential testing address related parts of
this verification problem.

The K framework {Informal.citep kframework2010}[] takes the same starting point for programming
languages: write one executable semantics, then derive interpreters, verifiers and provers from it
rather than maintaining several definitions that are supposed to agree. Matching logic
{Informal.citep matchinglogic2017}[] is the logic underneath that program, unifying the
model-theoretic and proof-theoretic sides so that reachability claims about a program become
derivations in a fixed proof system. More recently, K's provers emit proof objects checked by an
external kernel {Informal.citep kproofgen2021}[], which is the proof-carrying-code discipline again:
the prover is untrusted, the checker is small.

Comparing that with the present chapter, TorchLean's IR denotation plays the role of the K
semantics: the IR has one executable meaning, and the verifier's lowering into it is a compiler
pass that `runForwardIR_eq_evalForward` verifies. Deriving tools from a shared semantics can
reduce duplicated definitions, but does not remove the need to justify code generation and
runtime execution.

On the testing side, the relevant neighbors are differential and coverage-guided fuzzing
{Informal.citep nnsmith2023}[] {Informal.citep tensorfuzz2019}[], and on the certificate side the
LiRPA family of bound producers {Informal.citep crown2018}[] {Informal.citep autolirpa2020}[]
{Informal.citep betacrown2021}[]. Neither family competes with the lowering theorem; they answer the
other questions in the table above.

# Kinds Of Verification Evidence

The same result may have several kinds of evidence. A runtime example shows that a path executes on
one input, and a regression test guards selected inputs. A checker can validate every field of one
finite artifact. A refinement theorem instead quantifies over all inputs satisfying its hypotheses,
while a backend contract records whatever assumption remains about code outside Lean.

More evidence is welcome, but one kind does not silently become another. A CUDA parity test does
not prove a vendor kernel. A real-arithmetic CROWN theorem does not by itself prove a binary32
margin. A correct local VJP does not prove a complete recurrent network until the composition
theorem covers the unroll.

Autograd correctness, runtime approximation, optimizer laws, learning theory, scientific
certificates, reinforcement learning, and generative models use different relations. The important
question in every case is the same: which semantic object appears on both sides of the theorem?

Further reading on the runtime side of that question: the
[PyTorch autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html) notes and the
[JAX automatic differentiation
guide](https://docs.jax.dev/en/latest/automatic-differentiation.html) describe what the frameworks
guarantee about reverse mode, while
[α,β-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN),
[auto LiRPA](https://github.com/Verified-Intelligence/auto%5FLiRPA) and
[VNN-COMP](https://vnn-comp.github.io/) are where the bound-producing artifacts in the certificate
row come from.
