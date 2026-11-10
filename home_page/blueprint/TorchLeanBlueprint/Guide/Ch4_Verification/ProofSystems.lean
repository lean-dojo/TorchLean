import VersoManual
import TorchLean.Verify.CROWN.Arithmetic
import TorchLean.Tensor.Op.Table.Elementwise
import TorchLean.NN.IR.CustomRule
import TorchLean.NN.IR.Tangent
import TorchLean.NN.IR.Program
import TorchLean.Tensor.Einops.Elab.Einsum
import TorchLean.Tensor.Einops.Elab.Pack
import TorchLean.Tensor.Literal
import TorchLean.Tensor.Einops.Elab.Transform
import TorchLean.Tensor.Tactic.Einops.Report
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles
import TorchLean.Tensor.Op.Pointwise.Real

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean

-- The live example below builds a graph of registered operators and evaluates it, so this
-- chapter needs the operator IR namespace, the tensor spec layer, and the recorder whose
-- lowering theorem it names.
open TorchLean
open IR

-- A value table is printed below at Lean's own line width rather than this file's, so its
-- `leanOutput` block is wrapped by hand and compared with whitespace collapsed.
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Proof Systems" =>
%%%
tag := "proof-systems-beyond-bounds"
file := "Proof-Systems-Beyond-Bounds"
%%%

Imagine that lowering accidentally drops the bias from a linear layer. The resulting forward graph
may still be well shaped, execute without an exception, and even receive a convincing interval
certificate. That certificate would then describe the wrong function. A lowering-correctness
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
compilers: NNSmith reported 72 new bugs across TVM, TensorRT, ONNXRuntime and PyTorch
{Informal.citep nnsmith2023}[]. For verification, a wrong-answer bug can leave downstream checks
internally consistent while changing the function they describe.

The same pattern reappears elsewhere. IBP and CROWN relate boxes or affine forms to graph values,
autograd relates a reverse rule to the derivative of the forward map, and runtime approximation
relates rounded execution to ideal arithmetic. These are different proof systems, but each begins
by naming two objects and the relation that is supposed to connect them.

For a five-node affine-ReLU graph, we can inspect the value table (including the pre-activation
that ReLU partly erases). The lowering theorem then says that the graph a verifier bounds computes
the typed graph that the recorder built, for every input, but no theorem says that the typed graph
computes the source program.

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

This direction matters. Whether the producer is Python, CUDA, α,β-CROWN, or a remote solver, none
of those programs must be trusted merely because the checker accepts their output. Trust moves to
the Lean definition of `R`, the checker, and its soundness theorem. When no such theorem exists,
the artifact is evidence or an explicit boundary, not a certificate by vocabulary alone.

`R` is where the intended meaning enters. For lowering, it may be equality of evaluated value
tables. For intervals, it is containment of every semantic value in its box. For numerical
approximation, it is an inequality bounding the distance between two results. These conclusions
allow different substitutions. Equality lets us replace one value by another without loss. An error
bound spends part of a margin. Containment may be sufficient to prove an inequality even when the
exact value is unknown. Choosing the relation comes before choosing which proof or checker to run.

The idea is old enough to have a name. Proof-carrying code {Informal.citep necula1997}[] made the
same move for machine code: an untrusted producer ships a proof alongside the artifact, and the
host runs a small trusted checker rather than trusting the compiler. For a tensor program, the
semantic object includes shapes, axis conventions, and a scalar interpretation, and the relation
must preserve those choices through each translation.

# IR Lowering Correctness

Since the verifier bounds a graph of registered operators (`IR.Graph`), the translation it
depends on is the lowering of a TorchLean program into that graph. `IR.lowerForward` takes two
steps: it runs the program with a recording interpreter, which builds a typed graph of registered
operators (`IR.OpGraph`), and it builds that typed graph's operator graph. Its second step has a
theorem, {src "TorchLean/NN/IR/OpGraph.lean"}[`IR.OpGraph.eval_toOp`].

In plain English:

> Build the operator graph of a typed graph, with its constants, and run it on any input.
> The run succeeds, and its results are the typed graph's values at the chosen outputs.

This theorem has no premise, and `IR.lowerForward_eq_ok` applies it to the recorded graph. No
theorem covers the first step: nothing proves that the recorded graph computes the source program,
so a claim about a source model still needs its own argument for that step.

We state the theorem for a scalar type `α`. The `Storage α` and `Context α` instances supply the
tensor representation and scalar operations used on both sides, and `Op.Capabilities α` supplies
the optional operations, such as `erf` and the integral roundings, that some registered operators
read. This generality is useful: preservation of the program structure can be stated without
choosing reals or floats. But the theorem doesn't compare different scalar types: instantiating it
at `Float` compares two Lean interpretations using that same `Float` context, while a real-to-float
approximation is a separate relation.

The example below makes a value table concrete, including a pre-activation that ReLU will partly
erase.

# Affine-ReLU Example

Return to the bias-dropping bug from the opening, and make it concrete. Take the graph

$$`y=\operatorname{ReLU}(Wx+b),\qquad
W=\begin{pmatrix}1&2\\3&4\end{pmatrix},\qquad
b=\begin{pmatrix}0\\-30\end{pmatrix}.`

The bias is large and negative on purpose. It decides whether the second ReLU unit fires at all, so
dropping it isn't a small numerical difference: it changes which unit is active.

Five nodes, written out with the graph builder (`Graph.Builder`):

```lean
-- The affine node reads its weights and bias from the
-- constant nodes one and two.
def psGraph : Except String Graph :=
  let (b, x) := ({} : Graph.Builder).input 0 (.value [2])
  let (b, w) := b.const (.value [2, 2])
  let (b, bias) := b.const (.value [2])
  let (b, h) := b.op
    (TorchLean.Op.linear (lead := []) (inDim := 2)
      (outDim := 2)) [w, bias, x]
  let (b, y) := b.op (TorchLean.Op.relu (s := [2])) h
  b.build y.toArray
```

A node's id is its position. Node `0` is the graph input, nodes `1` and `2` are constants at the
ports of the weight matrix and the bias, node `3` is the operator `linear` on the references to
nodes `1`, `2` and `0`, and node `4` is `relu` on node `3`. A port is an element sort and a shape,
so each operator's type fixes the shapes it reads and produces, and `build` returns the graph only
after `Graph.check` accepts it. Weights live outside the graph: a constant node reads its value
from a *payload* keyed by node id, which is what lets the same graph be reused at different scalar
types and with different parameters without rebuilding the structure.

```lean
-- Two payloads share the weight matrix and differ only in
-- the bias.
def psW : Tensor Float [2, 2] := [[1, 2], [3, 4]]

def psPayload (b : Tensor Float [2]) : Payload Float :=
  { const? := fun
      | 1 => some ⟨.value [2, 2], psW⟩
      | 2 => some ⟨.value [2], b⟩
      | _ => none }

def psBias : Tensor Float [2] := [0, -30]
def psNoBias : Tensor Float [2] := [0, 0]
def psX : Tensor Float [2] := [1, 1]
```

`Graph.run` evaluates the nodes in order: node `0` reads the input list, nodes `1` and `2` read `W`
and `b` from the payload, node `3` computes $`Wx+b`, and node `4` applies ReLU. `Graph.eval` runs
the graph and reads its outputs, each a value with its port (`TorchLean.Op.Value`), and matching
the port recovers a typed tensor:

```lean
-- Evaluate the graph and read its one output as a
-- length-two vector.
def psOut (b : Tensor Float [2]) :
    Except String (Tensor Float [2]) := do
  let g ← psGraph
  match ← g.eval (psPayload b) [⟨.value [2], psX⟩] with
  | [⟨⟨.value, [2]⟩, y⟩] => pure y
  | _ => throw "expected one vector of length two"
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

A faulty lowering would produce the second payload from a source that has the first. Its result,
$`(3,7)`, would still have the expected shape and finite entries, so shape and finiteness checks
alone wouldn't reveal the change. But the lowering theorem rules it out for every input: what the
graph computes is what the recorded typed graph computes. We changed the payload by hand here, so
the example demonstrates the effect of a missing bias, not a bug in the lowering.

## Value Tables

The typed graph's theorem is proved by relating whole value tables, each typed value to the output
its reference reads (`IR.OpGraph.run_builder`), and this graph shows why that is stronger than
equating outputs. Ask for every node instead of the output node `4`:

```lean
-- Retain all node values so that a final ReLU cannot hide
-- an intermediate difference.
def psTable (b : Tensor Float [2]) :
    Except String (Array (List String)) := do
  let g ← psGraph
  let vals ← g.run (psPayload b) [⟨.value [2], psX⟩]
  pure (vals.map fun outs => outs.map fun
    | ⟨⟨.value, _⟩, t⟩ => Tensor.pretty t
    | _ => "not a value")
```

```lean (name := psTables)
-- Input, weights, bias, affine pre-activation, and ReLU
-- output, in node order.
#eval psTable psBias
```

```leanOutput psTables (whitespace := lax)
Except.ok #[["[1.000000, 1.000000]"],
  ["[[1.000000, 2.000000], [3.000000, 4.000000]]"],
  ["[0.000000, -30.000000]"], ["[3.000000, -23.000000]"],
  ["[3.000000, 0.000000]"]]
```

Entry `3` is the pre-activation $`Wx+b=(3,-23)^\top`, and ReLU erases its negative second
coordinate. A pre-activation of $`(3,-100)^\top` would produce the same final output at this input,
so an output comparison here would miss the difference. Comparing the tables retains it. Entries
`1` and `2` are the constants as the payload supplies them, so the table also records which bias
the run used.

A recorded PyTorch comparison computes the same four numbers {Informal.citep pytorch2019}[]:

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

In that recorded comparison, `torch.compile` and eager execution agree on this input. For a
recorded typed graph and its IR, the lowering theorem makes the corresponding claim for every
input.

Shapes are checked too. Here the structural check `Graph.check` accepts the graph: every operator
has one reference per input port, each naming an earlier output at a port that input accepts
(`TorchLean.Port.accepts`), every operator passes its attribute check (`TorchLean.Op.check`), and
every graph output names an existing output. `build` returns a graph only after the check accepts
it (`Graph.Builder.check_build`), so checking the built graph again gives the same answer:

```lean (name := psShapes)
-- Check every reference against the ports of the node it
-- names.
#eval psGraph >>= Graph.check
```

```leanOutput psShapes
Except.ok ()
```

`Graph.check_ok_iff` states that the check accepts exactly the well-formed graphs
(`Graph.WellFormed`). The statement about evaluation needs no check: whenever `Graph.run` succeeds,
entry `j` of its table is at the ports of node `j` (`Graph.run_ports`), so each computed value has
the shape its node declares. It assumes the run succeeded but doesn't prove that it does.

# Logarithm Domain

A graph with one raw logarithm shows what the operator `log` does outside the logarithm's domain:

```lean
-- One raw logarithm after the input.
def psLogGraph : Except String Graph :=
  let (b, x) := ({} : Graph.Builder).input 0 (.value [2])
  let (b, y) := b.op (TorchLean.Op.log (s := [2])) [x]
  b.build y.toArray

def psLog (x : Tensor Float [2]) :
    Except String (Tensor Float [2]) := do
  let g ← psLogGraph
  let out ← g.eval ({} : Payload Float) [⟨.value [2], x⟩]
  match out with
  | [⟨⟨.value, [2]⟩, y⟩] => pure y
  | _ => throw "expected one vector of length two"
```

```lean (name := psLogRuns)
-- A positive input, then a negative and a zero coordinate.
#eval psLog [1, 2]
#eval psLog [-1, 2]
#eval psLog [0, 2]
```

```leanOutput psLogRuns
Except.ok [0.000000, 0.693147]
```

```leanOutput psLogRuns
Except.ok [NaN, 0.693147]
```

```leanOutput psLogRuns
Except.ok [-inf, 0.693147]
```

The operator `log` computes `Tensor.log` (`Op.rule?_log`), the scalar type's logarithm at
every input, so over `Float` it returns the IEEE values (a NaN below zero and $`-\infty` at zero).
PyTorch returns the same NaN:

```
# PyTorch returns a tensor containing NaN for the negative
# coordinate.
print(torch.log(torch.tensor([-1., 2.])))
```

```
tensor([nan, 0.6931])
```

Over the reals the same graph evaluates Mathlib's totalized logarithm (with `Real.log 0 = 0` and
`Real.log (-x) = Real.log x`), so neither the real nor the IEEE value is simply "the mathematical
one", and the protected `Activation.safeLog` computes `log(softplus(x) + ε)`, a different,
domain-protected function. Raw IR evaluation can still fail on other nodes (for example on a
payload that doesn't match the node), and a caller can branch on an error where it can't on a
tensor. The lowering theorem compares `Except` values exactly, and the lowered run of a recorded
graph always succeeds.

# Sequential Model And Runtime Example

The `sequential` example takes a different path: it evaluates the MLP of
{ref "sequential"}[the sequential models chapter] under the pure interpreter and then trains it on
the eager runtime. It does not lower to the IR or instantiate the lowering theorem above.

```terminal
# Run the maintained example that evaluates and trains the
# sequential MLP.
./scripts/lake.sh exe torchlean sequential --device cpu
```

For the seeded example, the recorded run prints:

```terminal +output
== Sequential tutorial ==
pure forward (nn.Sequential.eval, Float) = [-0.011884]
model:
Sequential: [2] -> [1], layers=3, params=13, state=13
  [0] Linear(2, 3): [2] -> [3] params=9, state=9 [[3, 2], [3]]
  [1] relu: [3] -> [3] params=0, state=0 []
  [2] Linear(3, 1): [3] -> [1] params=4, state=4 [[1, 3], [1]]
dataset size = 1
mean_loss(before training) = 1.023910
mean_loss(after training) = 0.260259
steps=3 arithmetic=native scalar=Float32 loss=1.023910 -> 0.260259
```

The recorded run reports a completed execution and a loss decrease on that run. It is runtime
evidence for the sequential model on the eager tape, not an application of the canonical-IR
theorem. `TorchLean.Examples.Sequential.MLP.eval_model` covers the pure meaning of the same model.
To connect this training run to a mathematical model, we need results about the same model,
parameters, and runtime used here.

In that log, the parameter counts help identify the model. Its first linear layer has nine
parameters (six weights and three biases). ReLU adds none, and the last linear layer has four
(three weights and one bias), so the model has thirteen parameters in total. The printed shapes
describe the interfaces between those layers. They are useful checks that we're discussing the
intended architecture before reading the two loss values, which then describe this run with its
particular parameters and data.

# Unsupported Operators And Shape Checks

The axis-operator tutorial checks each graph of registered operators, then evaluates it:

```terminal
# Check and evaluate the axis-operation examples.
./scripts/lake.sh exe torchlean ir_axis_ops
```

For concatenation on the middle axis, it prints the output shape and leading values:

```
[cat_middle_dim] output shape: [2, 8, 4]
```

If the structural check or evaluation fails, the example reports the error.

This distinction matters: the semantic language can describe more programs than a particular
lowering theorem or runtime path covers. Returning an array of the expected shape wouldn't
establish correctness for an unsupported operation, so the lowering path must either justify that
operation or reject it.

The structural check `Graph.check` refuses malformed graphs of registered operators: an operator
with too few references, a node reading its own or a later output, and an output naming a missing
node. It concerns the structure of a graph rather than its evaluation. The lowering result is
`IR.lowerForward_eq_ok`.

# Proof Composition For A Training Step

Consider a claim about one training step on a Float32 backend. No single theorem should be expected
to prove the entire statement. Instead, the proof is assembled from relations:

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

The optimizer row deserves the same precision as the graph rows. Its equation
$`\theta_{\mathrm{new}}=\theta-\eta g` describes a plain gradient step. Momentum and Adam carry
additional state, so their update relation must include that state and its recurrence. Even for
plain gradient descent, the symbol `g` must refer to the gradient of the stated loss at the stated
parameters. An algebraically correct subtraction cannot compensate for a gradient computed using
another reduction convention, parameter order, or forward function.

# Transitivity Of Semantic Relations

Suppose source lowering relates a model `M` to an IR graph `G`, runtime lowering relates `G` to an
executable graph `E`, and a numerical theorem bounds `E_float` against the real denotation of `E`.
Then the end-to-end argument is ordinary transitivity, but each intermediate term must be the same
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

For the substitution to apply, the intermediate denotations must agree on parameters, masks, axes,
and scalar interpretation. A theorem about a different payload or mask convention cannot supply the
middle equality, even if its input and output tensor shapes match.

The bias example above is the smallest instance of the failure. If source lowering drops `b`, then
$`\llbracket M\rrbracket` and $`\llbracket G\rrbracket` are different functions, the first equation
is false, and the chain still *looks* fine because every term in it has the right type. Only an
equation between denotations catches it.

An approximation claim also has a domain. If its error estimate holds only while intermediates
remain finite or inputs stay in a box, those assumptions must survive the substitutions above. For
a classifier, the resulting output-error bounds are charged against the verified score margin. For
a regression model, they may be compared with a requested output tolerance. One lowering equality
can support both uses, but the numerical budget and final property must be chosen for the
application.

# Related Systems

Executable semantics, proof-producing tools, and differential testing address related parts of
this verification problem.

For programming languages, the K framework {Informal.citep kframework2010}[] takes the same
starting point: write one executable semantics, then derive interpreters, verifiers and provers
from it instead of maintaining several definitions that are supposed to agree. Matching logic
{Informal.citep matchinglogic2017}[] is the logic underneath that program, unifying the
model-theoretic and proof-theoretic sides so that reachability claims about a program become
derivations in a fixed proof system. In later work, K's provers emit proof objects checked by an
external kernel {Informal.citep kproofgen2021}[], which is the proof-carrying-code discipline
again: the prover is untrusted, the checker is small.

By analogy, TorchLean's IR denotation plays the role of the K semantics: the IR has one executable
meaning, and the verifier's lowering into it is a compiler pass whose second step (from the
recorded typed graph to its operator graph) `IR.OpGraph.eval_toOp` verifies. Deriving tools from
a shared semantics can reduce duplicated definitions, but it doesn't remove the need to justify
code generation and runtime execution.

On the testing side, the relevant neighbors are differential and coverage-guided fuzzing
{Informal.citep nnsmith2023}[] {Informal.citep tensorfuzz2019}[], and on the certificate side the
LiRPA family of bound producers {Informal.citep crown2018}[] {Informal.citep autolirpa2020}[]
{Informal.citep betacrown2021}[]. Neither family competes with the lowering theorem: they answer
the other questions in the table above.

# Kinds Of Verification Evidence

A single result may have several kinds of evidence. A runtime example shows that a path executes on
one input, and a regression test guards selected inputs. A checker can validate every field of one
finite artifact. A refinement theorem instead quantifies over all inputs satisfying its hypotheses,
while a backend contract records whatever assumption remains about code outside Lean.

More evidence is welcome, but one kind does not silently become another. A CUDA parity test does
not prove a vendor kernel. A real-arithmetic CROWN theorem does not by itself prove a binary32
margin. A correct local VJP does not prove a complete recurrent network until the composition
theorem covers the unroll.

Autograd correctness, runtime approximation, optimizer laws, learning theory, scientific
certificates, reinforcement learning, and generative models use different relations. In every case,
though, the important question is the same: which semantic object appears on both sides of the
theorem.

Further reading on the runtime side of that question: the
[PyTorch autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html) notes and the
[JAX automatic differentiation
guide](https://docs.jax.dev/en/latest/automatic-differentiation.html) describe what the frameworks
guarantee about reverse mode, while
[α,β-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN),
[auto LiRPA](https://github.com/Verified-Intelligence/auto%5FLiRPA) and
[VNN-COMP](https://vnn-comp.github.io/) are where the bound-producing artifacts in the certificate
row come from.
