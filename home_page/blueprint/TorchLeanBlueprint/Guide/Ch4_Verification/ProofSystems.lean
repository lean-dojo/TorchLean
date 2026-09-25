import VersoManual
import NN.Runtime.Autograd.IRExec.Correctness.SemanticEquivalence
import NN.IR
import NN.Tensor
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean

-- The live example below builds an IR graph, lowers it, and runs both interpretations, so this
-- chapter needs the IR namespace, the tensor spec layer, and the lowering correctness module. We
-- import the end-to-end theorem file directly: `IRExec.Correctness` deliberately stops short of it
-- so that ordinary runtime builds do not pay for the slowest proof module in the tree.
open TorchLean
open Spec
open NN.IR
open Runtime.Autograd.IRExec

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

For a three-node affine-ReLU graph, we can inspect both value tables directly, including the
pre-activation that ReLU partly erases. The lowering theorem then states when those tables agree
for every input.

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

Here are the three declarations the rest of this chapter uses, printed by Lean. First, the lowering
pass:

```lean (name := psLower)
-- Inspect the lowering function's inputs and its
-- success-or-error result.
#check @lowerToForwardGraph
```

```leanOutput psLower (whitespace := lax)
@lowerToForwardGraph : {α : Type} →
  [inst : Storage α] →
    [inst_1 : Context α] →
      Graph → Payload α → Except String (ForwardGraph α)
```

The `Except String` result distinguishes a successfully lowered graph from an error. The input
IR can describe operations or shapes this lowering path does not support, so the correctness
theorem below is conditional on obtaining a `ForwardGraph`. That success premise also includes
the checks performed by the lowering function.

Second, the fragment condition:

```lean (name := psNoLog)
-- This predicate describes the graph fragment covered by
-- the equality theorem.
#check @NoRawLog
```

```leanOutput psNoLog
NoRawLog : Graph → Prop
```

Third, the theorem that ties the two interpretations together:

```lean (name := psThm)
-- Successful lowering and NoRawLog give equality of value
-- tables for every input.
#check @denoteAll_eq_of_lowerToForwardGraph
```

```leanOutput psThm (whitespace := lax)
@denoteAll_eq_of_lowerToForwardGraph : ∀ {α : Type}
  [inst : Storage α] [inst_1 : Context α] (g : Graph)
  (payload : Payload α) (exec : ForwardGraph α),
  NoRawLog g →
    lowerToForwardGraph g payload = Except.ok exec →
      ∀ (x : Tensor α exec.inShape),
        g.denoteAll payload
            { shape := exec.inShape, tensor := x } =
          Except.ok (exec.denoteAll x)
```

The declaration lives in the
{src "NN/Runtime/Autograd/IRExec/Correctness/SemanticEquivalence.lean"}[IR execution correctness
API].

In plain English:

> If `lowerToForwardGraph` successfully lowers an `NN.IR.Graph` and payload into a shape-indexed
> `ForwardGraph`, and the graph contains no raw `log` node, then evaluating that forward graph on
> any input gives the same value table as the Lean denotational evaluator for the original IR
> graph.

The theorem connects three concrete objects: `NN.IR.Graph.denoteAll`, the reference denotation of
the tagged operation IR; `lowerToForwardGraph`, the lowering function; and `ForwardGraph.denoteAll`,
the forward-only evaluator produced from the IR.

The equation compares a value at every node id. This lets a later proof refer to an intermediate
activation as well as the final output. The input type `Tensor α exec.inShape` supplies the shape
expected by the lowered graph, and the original evaluator receives that same tensor with its shape
attached. Both sides also use the same scalar context and payload.

Read the printed theorem from its two arrows inward. `NoRawLog g →` requires evidence that this
graph meets the fragment restriction. `lowerToForwardGraph ... = Except.ok exec →` requires
evidence that this particular executable graph was produced by lowering this graph and payload.
Only then does `∀ x` introduce every input of the required shape. The final equality is between
the original evaluator's `Except` result and a successful result containing the executable value
table. It establishes successful evaluation as well as equality under those hypotheses.

The leading `α` is the scalar type. The `Storage α` and `Context α` instances supply the tensor
representation and scalar operations used on both sides. This generality is useful: preservation
of the program structure can be stated without choosing reals or floats. It does not compare
different scalar types. Instantiating the theorem at `Float` compares two Lean interpretations
using that same `Float` context; a real-to-float approximation is a separate relation.

In theorem notation, the statement has the shape:

$$`\operatorname{lowerToForwardGraph}(G,P)=\operatorname{ok}(E)
\;\land\; \operatorname{NoRawLog}(G)
\quad\Longrightarrow\quad
\forall x,\;
\operatorname{Graph.denoteAll}(G,P,x)
=
\operatorname{ok}\!\left(\operatorname{ForwardGraph.denoteAll}(E,x)\right)`

For accepted graphs satisfying `NoRawLog`, this theorem establishes the stated equivalence of
the two Lean evaluators. Native execution still needs its own connection to their denotations.

Successful lowering and `NoRawLog` are the two explicit premises. Structural checks are part of
lowering. The extra fragment restriction addresses a domain mismatch for the raw real logarithm.
The IR evaluator rejects a nonpositive input, while the lowered closure applies the
total specification logarithm to every input, so the two can only be compared under a domain fact
the theorem does not carry. A protected logarithm can avoid the domain failure, but a `.log` node
remains outside this syntactic theorem even when its input is provably positive. Covered successful
cases include scalar MSE, concatenation
along any axis, and matrix multiplication or linear layers with any shared leading shape; shapes
the lowering rejects never reach the theorem because `lowerToForwardGraph` returns an error for
them.

A new lowering case must fit this same preservation statement. Its local proof needs to account
for parameter lookup, output shape, and any failure behavior before the recursive argument can
extend the equivalence to the new node.

The proof is large because it recursively mirrors lowering. The workhorse lemma is
`buildFrom_preserves_denotation`: as lowering walks node ids and extends the forward graph, the
IR value table and typed context stay aligned. Each operator branch proves one local
preservation fact, then the recursive theorem stitches the branch into the whole graph.

The current proof is split for auditability:

- The
  {src "NN/Runtime/Autograd/IRExec/Correctness/Common.lean"}[IRExec common API] contains shared
  infrastructure for dynamic values, typed contexts, and finishing a node step.
- The
  {src "NN/Runtime/Autograd/IRExec/Correctness/SemanticEquivalenceCommon.lean"}[semantic equivalence
  common API] contains helper lemmas used by the recursive proof.
- The
  {src "NN/Runtime/Autograd/IRExec/Correctness/SemanticEquivalenceOpCases.lean"}[semantic
  equivalence op cases API] contains the larger named cases such as `.linear` and `.conv`.
- `Correctness/Ops/*` contains smaller branches by op family: activations, constants, elementwise,
  linear algebra, normalization, pooling, permutation, random, reductions, structural ops, and unary
  ops.
- The
  {src "NN/Runtime/Autograd/IRExec/Correctness/SemanticEquivalence.lean"}[semantic equivalence
  theorem API] ties the cases together into `denoteAll_eq_of_lowerToForwardGraph`.

The recursive theorem walks every node kind, so a goal in the middle of it mentions shape
equality, `Except` success and failure paths, and cast proof irrelevance at the same time.
Separating operator families
helps isolate proof failures; adding an operation can still require
changes to its lowering, local preservation lemmas, and the recursive proof.

The example below makes the preserved value table concrete, including a pre-activation that ReLU
will partly erase.

# Affine-ReLU IRExec Example

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

Now the two interpretations. The first one interprets the IR directly: node `0` produces the input,
node `1` reads `W` and `b` from the payload and computes $`Wx+b`, node `2` applies ReLU. The
`denote` entry point evaluates the whole table and projects one node id, and `expectShape` turns the
shape-erased result back into a typed tensor:

```lean
-- Evaluate the tagged IR directly and retrieve node two as
-- a length-two vector.
def psSpecOut (b : Tensor Float [2]) :
    Except String (Tensor Float [2]) := do
  let out ←
    Graph.denote (α := Float) (g := psGraph)
      (payload := psPayload b)
      (input := SomeTensor.ofTensor psX)
      (outputId := 2)
  Graph.expectShape (α := Float) (expected := [2]) out
```

The second lowers the IR to a `ForwardGraph`, whose nodes are typed closures, and runs that:

```lean
-- Lower the same graph and payload, then evaluate its typed
-- forward representation.
def psExecOut (b : Tensor Float [2]) :
    Except String (Tensor Float [2]) := do
  let out ← evaluate psGraph (psPayload b) psX
    ⟨2, by decide⟩
  Graph.expectShape (α := Float) (expected := [2]) out
```

The shared `evaluate` entry point checks the graph and input shape, lowers it, and selects the
requested output. `Graph.expectShape` recovers the `[2]` result needed by this caller. The table
example below exposes the lower-level shape equality when we need every intermediate value.

Run both paths, with the bias and then with the bias dropped:

```lean (name := psRun)
-- First compare the two interpreters with the intended
-- bias.
#eval psSpecOut psBias
#eval psExecOut psBias
-- Then change the payload for both interpreters to expose
-- the bias's effect.
#eval psSpecOut psNoBias
#eval psExecOut psNoBias
```

```leanOutput psRun
Except.ok [3.000000, 0.000000]
```

```leanOutput psRun
Except.ok [3.000000, 0.000000]
```

```leanOutput psRun
Except.ok [3.000000, 7.000000]
```

```leanOutput psRun
Except.ok [3.000000, 7.000000]
```

Check the arithmetic by hand. $`Wx=(1+2,\;3+4)^\top=(3,7)^\top`. Adding the bias gives
$`(3,-23)^\top`, and ReLU clamps the second coordinate to zero, so the answer is $`(3,0)^\top`.
Dropping the bias leaves $`(3,7)^\top`, and ReLU changes nothing. The second unit went from dead to
firing at $`7`.

Both evaluators agree for each supplied payload. Replacing the bias changes both results, as it
should. A faulty lowering would instead mix these cases: the source would retain the bias while
the lowered graph omitted it. The result $`(3,7)` would still have the expected shape and finite
entries, so shape and finiteness checks alone would not reveal that change.

The four outputs are paired in the order of the four commands: reference with bias, lowered
with bias, reference without bias, lowered without bias. `Except.ok` records that evaluation
succeeded; the vector is the returned value. The repeated vectors within each pair support the
comparison for this one input. The difference between the pairs explains what a missing-bias
bug would change. We deliberately changed the payload in both calls here, so this demonstration
does not report an actual bug in the lowering implementation.

## Value-Table Preservation

The conclusion equates value tables, and this graph shows why that is stronger than equating
outputs. Ask both interpretations for every node instead of node `2`:

```lean
-- Retain all node values so that a final ReLU cannot hide
-- an intermediate difference.
def psSpecTable (b : Tensor Float [2]) :
    Except String (Array String) := do
  let vals ←
    Graph.denoteAll (α := Float) (g := psGraph)
      (payload := psPayload b)
      (input := SomeTensor.ofTensor psX)
  pure (vals.map (fun v => s!"{v.tensor}"))

def psExecTable (b : Tensor Float [2]) :
    Except String (Array String) := do
  let eg ← lowerToForwardGraph (α := Float) psGraph
    (psPayload b)
  if h : ([2] : Shape) = eg.inShape then
    let vals :=
      ForwardGraph.denoteAll (α := Float) eg
        (Tensor.castShape (tensor := psX) h)
    pure (vals.map (fun v => s!"{v.tensor}"))
  else
    throw "forward graph: input shape mismatch"
```

```lean (name := psTables)
-- Compare input, affine pre-activation, and ReLU output in
-- node order.
#eval psSpecTable psBias
#eval psExecTable psBias
```

```leanOutput psTables (whitespace := lax)
Except.ok #["[1.000000, 1.000000]",
  "[3.000000, -23.000000]",
  "[3.000000, 0.000000]"]
```

```leanOutput psTables (whitespace := lax)
Except.ok #["[1.000000, 1.000000]",
  "[3.000000, -23.000000]",
  "[3.000000, 0.000000]"]
```

The middle entry is the pre-activation $`Wx+b=(3,-23)^\top`; ReLU erases its negative second
coordinate. A pre-activation of $`(3,-100)^\top` would produce the same final output at this input,
so an output comparison here would miss the difference. The table comparison retains it. This
example concerns one input; it does not show that the two altered programs agree on all inputs,
as a universal output-equivalence theorem would require.

PyTorch computes the same four numbers {Informal.citep pytorch2019}[]:

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

`torch.compile` and eager execution agree on this input. The theorem below compares the two
TorchLean evaluators for every input, conditional on successful lowering and `NoRawLog`.

Instantiate the theorem for this graph. The side condition comes first, and it is a `decide`:

```lean
-- The literal graph contains input, linear, and ReLU nodes,
-- so NoRawLog is decidable.
theorem psNoRawLog : NoRawLog psGraph :=
  noRawLog_of_forall_mem (by decide)

example (exec : ForwardGraph Float)
    (h : lowerToForwardGraph (α := Float) psGraph
      (psPayload psBias) = .ok exec) :
    ∀ x : Tensor Float exec.inShape,
      Graph.denoteAll (α := Float) (g := psGraph)
          (payload := psPayload psBias)
          (input := SomeTensor.mk (α := Float)
            exec.inShape x) =
        .ok (ForwardGraph.denoteAll (α := Float)
          (e := exec) x) :=
  denoteAll_eq_of_lowerToForwardGraph _ _ _ psNoRawLog h
```

The `example` takes the successful lowering as a hypothesis rather than computing it, which is the
right shape for a reusable statement: whatever executable graph lowering produced, its value table
agrees with the IR denotation on all of `Tensor Float exec.inShape`.

`noRawLog_of_forall_mem` converts a fact about entries of `g.nodes` into `NoRawLog`. Internally,
that predicate is stated through `Graph.getNode`, which suits the recursive lowering proof. For
a literal graph, checking the node array is easier: `by decide` verifies that none of its entries
has kind `.log`.

Unlike `#eval`, the theorem block has no tensor result to print. Lean checks that its proof term
has the proposition written after the colon. The unnamed `example` is still checked: its purpose
here is to show how the general theorem specializes to our graph and payload. The variable
`exec` ranges over possible lowered graphs, while `h` ties it to this exact call to the lowering
function. Removing `h` would leave no reason for an arbitrary forward graph to agree with the
reference evaluator.

Notice also what the proof does not need to enumerate. It never lists input vectors, evaluates
the ReLU at a grid of points, or stores expected outputs. Once the graph restriction and successful
lowering are available, the general preservation theorem handles all tensors of the input shape.
The earlier evaluations remain useful for understanding the proposition and the effect of the
bias. Their role is explanatory evidence about concrete values; the quantified conclusion comes
from the theorem application. This is why preserving the hypotheses in the printed statement
matters as much as explaining its final equality.

# Logarithm Domain Restriction

A graph containing one raw logarithm shows why the domain restriction is needed. We can compare
the same two evaluators on positive and negative inputs:

```lean
-- Add a raw logarithm to exhibit a graph excluded by the
-- theorem's premise.
def psLogGraph : Graph :=
  { nodes :=
      #[ { id := 0, parents := #[], kind := .input,
           outShape := [2] }
       , { id := 1, parents := #[0], kind := .log,
           outShape := [2] } ] }

example : ¬ NoRawLog psLogGraph :=
  fun h => h 1 _ rfl rfl
```

The refutation is that short because `NoRawLog` is a statement about node kinds: exhibit node `1`,
whose lookup succeeds by `rfl`, and whose kind is `.log` by `rfl`. So this graph is outside the
theorem's fragment, by proof rather than by convention.

```lean
-- Compare the IR's domain-checking evaluator with the
-- lowered forward evaluator.
def psLogSpec (x : Tensor Float [2]) :
    Except String (Tensor Float [2]) := do
  let out ←
    Graph.denote (α := Float) (g := psLogGraph)
      (payload := {}) (input := SomeTensor.ofTensor x)
      (outputId := 1)
  Graph.expectShape (α := Float) (expected := [2]) out

def psLogExec (x : Tensor Float [2]) :
    Except String (Tensor Float [2]) := do
  let out ← evaluate psLogGraph {} x ⟨1, by decide⟩
  Graph.expectShape (α := Float) (expected := [2]) out
```

On a positive input the two paths agree, so nothing looks wrong:

```lean (name := psLogPos)
-- Both coordinates satisfy the logarithm's positive-input
-- domain.
#eval psLogSpec [1, 2]
#eval psLogExec [1, 2]
```

```leanOutput psLogPos
Except.ok [0.000000, 0.693147]
```

```leanOutput psLogPos
Except.ok [0.000000, 0.693147]
```

Feed in a negative coordinate and they part company:

```lean (name := psLogNeg)
-- A negative coordinate exposes the evaluators' different
-- failure behavior.
#eval psLogSpec [-1, 2]
#eval psLogExec [-1, 2]
```

```leanOutput psLogNeg (whitespace := lax)
Except.error "IR eval: log: input contains values <= 0
  (or NaN); use `safe_log`, which is
  log(softplus(x) + eps)"
```

```leanOutput psLogNeg
Except.ok [NaN, 0.693147]
```

The IR denotation is *partial*: the positive-input domain excludes $`-1`, so the
evaluator refuses and
says which operator refused and what to use instead. The lowered closure is *total*: it applies the
specification logarithm to every input and returns a NaN. Both behaviors are defensible, and neither
is a bug. They are simply not equal, so a theorem claiming they agree for every input would be
false, and a stronger theorem would need a positivity hypothesis or a checked evaluator preserving
the
IR failure behavior.

For comparison, PyTorch takes the total route as well:

```
# PyTorch returns a tensor containing NaN for the negative
# coordinate.
print(torch.log(torch.tensor([-1., 2.])))
```

```
tensor([nan, 0.6931])
```

This compares two internal evaluators. Public `IO` autograd transforms now validate raw-log
preconditions before evaluating nodes. Mathlib's real logarithm is itself totalized, so neither
internal runtime policy is simply "the mathematical one". The protected `safeLogSpec` computes
`log(softplus(x) + ε)`; with positive epsilon it has a different, domain-protected meaning.

The two Lean outputs disagree before we compare their numeric entries. One is an error; the
other is a successful container whose first coordinate is NaN. A caller can branch on that
distinction, so it belongs to the observable behavior that a semantics-preservation theorem
must account for. This is also why replacing a failure with a conveniently shaped tensor would
not repair the proof. It would change the evaluator's contract. A theorem covering raw logarithms
needs either a suitable domain premise or agreement on how domain failures are represented.

## NoRawLog Fragment Restriction

The IR has no separate safe-log node kind. The following example protects a logarithm by clamping
its input against a positive constant. It computes `log(max(x, ε))`, which is a different function
from `safeLogSpec`'s softplus-based formula.

```lean
-- Clamp each coordinate at epsilon before applying the
-- existing raw-log node.
def psEps : Tensor Float [2] := [1e-6, 1e-6]

def psClampedGraph : Graph :=
  { nodes :=
      #[ { id := 0, parents := #[], kind := .input,
           outShape := [2] }
       , { id := 1, parents := #[], kind := .const [2],
           outShape := [2] }
       , { id := 2, parents := #[0, 1], kind := .maxElem,
           outShape := [2] }
       , { id := 3, parents := #[2], kind := .log,
           outShape := [2] } ] }

def psClampPayload : Payload Float :=
  { const? := fun id =>
      if id = 1 then some { n := 2, v := psEps }
      else none }

def psClampedSpec (x : Tensor Float [2]) :
    Except String (Tensor Float [2]) := do
  let out ←
    Graph.denote (α := Float) (g := psClampedGraph)
      (payload := psClampPayload)
      (input := SomeTensor.ofTensor x) (outputId := 3)
  Graph.expectShape (α := Float) (expected := [2]) out

def psClampedExec (x : Tensor Float [2]) :
    Except String (Tensor Float [2]) := do
  let out ← evaluate psClampedGraph psClampPayload x
    ⟨3, by decide⟩
  Graph.expectShape (α := Float) (expected := [2]) out
```

The negative coordinate that split the two interpretations a moment ago now goes through both of
them to the same value:

```lean (name := psClamped)
-- The formerly negative coordinate now reaches log as
-- epsilon.
#eval psClampedSpec [-1, 2]
#eval psClampedExec [-1, 2]
```

```leanOutput psClamped
Except.ok [-13.815511, 0.693147]
```

```leanOutput psClamped
Except.ok [-13.815511, 0.693147]
```

And yet the theorem still does not apply to this graph:

```lean
-- The graph still contains a raw-log tag, regardless of its
-- protected input values.
example : ¬ NoRawLog psClampedGraph :=
  fun h => h 3 _ rfl rfl
```

`NoRawLog` looks at node kinds, not at the values that can reach them, so a `.log` node is excluded
whether or not its parent is clamped. This gives a simple condition with a concrete limitation. A
value-level hypothesis, saying every input reaching a logarithm node is positive, would cover the
clamped graph, but it is a statement about the whole evaluation and a caller could not discharge it
with `by decide`; the syntactic version is decidable for any concrete graph, which is what makes
`psNoRawLog` a one-liner above. The cost is exactly the case seen here: graphs that agree for a
reason the syntax cannot see fall outside the fragment and need their own argument.

Extending the theorem to this clamped graph would require a proof about values reaching its log
node, or an evaluator that preserves the IR's domain checks. Removing `NoRawLog` alone would also
admit the unclamped negative-input example, whose two results differ.

# Graph Specification And Runtime Example

The `graphspec` example constructs a typed model, lowers it, trains it through the runtime, and
reports the object that crossed each layer:

```terminal
# Run the maintained example that builds, lowers, and trains
# the GraphSpec model.
lake exe torchlean graphspec
```

A seeded run includes:

```
Sequential: [2] -> [1], layers=3, params=13
  [0] Linear(2, 3): [2] -> [3] params=9
  [1] ReLU: [3] -> [3] params=0
  [2] Linear(3, 1): [3] -> [1] params=4
mean_loss(before) = 1.239197
mean_loss(after) = 0.247518
forward: GraphSpec MLP lowered to TorchLean and executed
```

The output establishes that this execution completed and that the loss decreased on this run. The
lowering theorem supplies the stronger statement: for every input, if lowering succeeds and the
graph has no raw logarithm, the forward-graph denotation equals the IR denotation. The training
log and lowering theorem answer different questions, and both are useful.

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
lake exe torchlean ir_axis_ops
```

For concatenation on the middle axis, it prints the output shape and leading values:

```
[concat_middle_axis] output shape: [2, 8, 4]
```

If shape validation or lowering fails, the runtime returns an error and the example reports it.

This distinction matters. The semantic language can describe more
programs than a particular lowering theorem or runtime path currently covers. Returning an array
of the expected shape would not establish correctness for an unsupported operation; the lowering
path must either justify that operation or reject it.

The executable negative cases in
{src "NN/Tests/IR/ShapeContracts.lean"}[`IR.ShapeContracts`]
exercise this boundary for incompatible shapes and unsupported contracts. They are regression
checks that rejection remains fail-closed, not semantic lowering theorems. The whole-graph
meaning-preservation result is still
`denoteAll_eq_of_lowerToForwardGraph` with its explicit `NoRawLog` hypothesis.

# Proof Composition For A Training Step

Consider a claim about one training step on a Float32 backend. No single theorem should be expected
to prove the entire statement. The proof is assembled from relations:

$$`
\begin{aligned}
\text{source model}
&\equiv \text{IR denotation},\\
\text{IR denotation}
&\equiv \text{lowered forward-graph denotation},\\
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
  * IR to executable graph
  * graph is well formed, lowering succeeds, and no raw logarithm node is present
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
semantics, `lowerToForwardGraph` is a compiler pass rather than a derived tool, and
`denoteAll_eq_of_lowerToForwardGraph` verifies that separate lowering. Deriving tools from a
shared semantics can reduce duplicated definitions, but does not remove the need to justify code
generation and runtime execution. Here a typed forward graph supports shape-indexed evaluation
and the autograd development, with a separate theorem connecting its meaning to the IR.

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
