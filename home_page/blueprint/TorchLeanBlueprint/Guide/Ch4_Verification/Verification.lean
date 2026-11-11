import VersoManual
import TorchLean.Verify.CROWN.Arithmetic
import TorchLean.NN.IR.Builder
import TorchLean.Verify.Certificate.CROWNNode
import TorchLean.Verify.Certificate.FiniteArtifact.Semantics
-- The shape contract of the matrix product examples below.
import TorchLean.NN.IR.OpContracts
-- The bound theorems quoted below: the IBP certificate, the rounded interval pass, the forward
-- α-CROWN pass and the backward CROWN queries on graphs of registered operators.
import TorchLean.Verify.CROWN.Op.Certify
import TorchLean.Verify.CROWN.Op.Alpha
import TorchLean.Verify.CROWN.Op.BackwardRun
import TorchLean.Verify.CROWN.Op.CertSoundness
-- The worked two-layer example below writes its input corners as tensor literals.
import TorchLean.Tensor.Literal
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles
import TorchLean.NN.Session.Random

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean

-- The theorems quoted in this chapter live in deep namespaces, and Verso keeps displayed
-- code narrow enough to read in a sidebar. Opening the namespaces here lets every `#check`
-- below fit on one line without losing any information: a printed signature always spells
-- out the fully qualified name of everything it mentions.
open TorchLean.CROWN.Op
open TorchLean.Certificate
open TorchLean.CROWN.Graph (EnclosesAtInput AlphaOK)

-- The soundness statements printed below are wider than the 100 columns this file allows, so their
-- expected output is rewrapped by hand and the blocks ask for `whitespace := lax`, which compares
-- the text after collapsing runs of whitespace. The rendered page still shows Lean's own layout.
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

#doc (Manual) "Neural Network Verification" =>
%%%
tag := "verification"
%%%

Consider a classifier with two inputs and two possible labels. Give it the point `[1.5, 0.0]` and
it returns two scores (one for class zero, one for class one), and the larger score determines the
prediction. We can evaluate the model at this point and see which class wins. We can also move the
input a little and evaluate it again. Both computations tell us what happened at the points we
chose, but neither tells us what happens at every nearby point.

That distinction matters when the input is a measurement. Suppose each coordinate may differ by up
to `0.10` from its recorded value. Then the possible inputs form a square around `[1.5, 0.0]`. If
the classifier chooses the same label throughout that square, this particular uncertainty can't
change its decision. But if there's a decision boundary inside the square, an input change within
our stated tolerance may cross it. Small input changes can alter classifier predictions, as
{Informal.citet szegedy2014}[] demonstrated, and evaluating a finite collection of perturbations
leaves open what happens between them.

Neural network verification asks whether a specified property holds for all inputs in a specified
set. Local classification robustness is one such property (others concern output ranges,
monotonicity, or an inequality used by a controller). The model, the input set, and the property
all belong to the statement: verifying one square around one point doesn't establish the same claim
everywhere, and preserving a prediction doesn't show that the prediction is the correct label in
the first place.

I'll start with a classifier because its property is easy to read from the outputs: there are only
two scores to compare, and the same comparison extends to more classes. Let `f` return the
classifier's logits (a logit is a score before a probability normalization, and it may be
negative), let `X` be the input region, and let `y` and `j` be the intended and competing classes.
To show that class `y` stays ahead of class `j` throughout the region, we seek

$$`\forall x\in X,\qquad f_y(x)-f_j(x)>0.`

Subtracting the two logits measures the pairwise margin. A positive lower bound on that margin
would settle the claim, provided it refers to the intended model, input region, and arithmetic.
Each of those choices must be tied to the computation: lowering must preserve the model, the bound
procedure must enclose the graph's values, and accepted certificate data must satisfy the bound
theorem's premises. Rounded execution adds an approximation argument.

For the two-class model, this is one inequality. With ten classes, it's nine inequalities, one
against each competitor. Strict positivity also removes a possible tie: the conclusion doesn't
depend on which class an implementation selects when two scores are equal. What's left is finding a
lower bound that applies to the whole input region and justifying how it was computed. The training
run below gives us a concrete model and margin to follow.

# Classifier Training And Verification

The maintained workflow trains a `2 -> 8 -> 2` ReLU classifier: two input coordinates, eight hidden
units, and two output scores. Its eight training examples separate by the sign of the first
coordinate. Positive first coordinates have class-zero labels, and negative ones have class-one
labels. Training fits that distinction with one-hot cross-entropy and SGD. Verification then fixes
the resulting weights and asks about the model that was actually trained:

```
-- Keep the parameters produced by these 120 training steps.
let trained ← trainer.train dataset { steps := 120 }
-- Ask whether class zero stays strictly ahead throughout
-- the input box.
let report ← trained.verify center
  (radius := 0.10)
  (property := .topLabel 0)
report.printSummary
```

Tensor shapes use the public notation: `[]` for a scalar tensor, `[n]` for a vector, and `[m, n]`
for a matrix. Nothing in the call above exposes recursive shape constructors, graph nodes, flat
boxes, or parameter stores.

Run it with either maintained CPU arithmetic mode:

```terminal
# Run the classifier workflow with its default CPU scalar
# backend.
scripts/lake.sh exe verify -- torchlean-mlp-workflow
# Repeat using the executable IEEE binary32 scalar model.
scripts/lake.sh exe verify -- torchlean-mlp-workflow --arithmetic ieee
```

A recorded run of either command printed these report lines, abridged here with the CROWN line
wrapped. Equal printed values for the two modes don't establish agreement between the backends:

```terminal +output
avg_loss(on samples)=0.010122
prediction at center=[2.016916, -3.001449]
CROWN radius=0.100000 lower=[1.781388, -3.287453] upper=[2.229471, -2.722632]
  label=0 margin=4.504020 certified=true
```

The report lower-bounds class zero by `1.781388` and upper-bounds its only competitor by
`-2.722632`. Their difference is the positive reported margin `4.504020`, so the chosen bound pass
reports `certified=true` for the requested input box. Interpreting that report as a theorem
requires the semantic and arithmetic bridges below.

The loss describes fitting the training examples. A small training loss says nothing by itself
about the entire perturbation region. A single forward evaluation produces the
`prediction at center` line: `2.016916` beats `-3.001449`, so the center receives class zero. The
`lower` and `upper` arrays answer a different question. Their entries describe bounds on each score
as the input varies. To protect class zero, the unfavorable comparison uses its lowest allowed score
and class one's highest allowed score. Comparing the two center scores would miss that variation.

The input region is the L∞ ball of radius `0.10` around the center: each input coordinate may move
independently by at most the radius. In real coordinates the intended region is
`[1.4, 1.6] × [-0.1, 0.1]`. This geometric meaning is part of the verification request, while the
representation and rounding of its endpoints belong to the arithmetic obligations below.

The reported margin subtracts the competitor's upper endpoint from class zero's lower endpoint.
Using the displayed values gives:

```lean (name := marginEval)
-- Use the unfavorable endpoint for each of the two classes.
def certifiedLower : Float := 1.781388

def competitorUpper : Float := -2.722632

-- Reconstruct the margin from the six-decimal values in the
-- report.
#eval certifiedLower - competitorUpper
```

```leanOutput marginEval
4.504020
```

The flag tests the computed margin. Since the six-decimal endpoints above have also been rounded
for display, subtracting their printed values checks the displayed arithmetic, not the exact
internal margin or a theorem about every point in the box.

A proof of the reported property would connect the source model to the lowered graph, show that
the propagated boxes contain its outputs throughout the input region, and use the positive margin
to compare the two logits. For native binary32 execution, the same argument also needs a bound on
how far its outputs can move from those semantic values.

These obligations also localize maintenance. A lowering change affects the source-to-graph
argument, a new CROWN activation affects bound propagation, and a CUDA deployment claim requires
the native arithmetic link in addition to the real-valued theorem.

# Semantic Target And Graph Boundary

TorchLean's verifier lowers a model to a graph of registered operators (`TorchLean.IR.Graph`) and a
payload of its constants (`TorchLean.IR.Payload`), the fields `graph` and `payload` of
`TorchLean.IR.LoweredModel`, and runs its bound passes on that graph. An interval or affine form is
meaningful only relative to the runs of that same graph and payload (`TorchLean.IR.Graph.run`) on
the inputs in that same box. So a lowering theorem is part of a source-model claim.

`TorchLean.IR.lowerForward` records a TorchLean program as a typed graph (`TorchLean.IR.OpGraph`)
and builds its operator graph. No theorem relates the recorded graph to the source program, but the
lowered graph runs as the recorded graph
({src "TorchLean/NN/IR/Lowering.lean"}[`TorchLean.IR.lowerForward_eq_ok`]: on every input the run
succeeds with the typed graph's value at the output).

A smaller fact sits beside it: `TorchLean.IR.Graph.run_ports` proves that the values a successful
run computes at a node are at the ports the node declares (`TorchLean.IR.Node.outs`), so each has
its declared element sort and shape.

Both are Lean semantic statements over an abstract scalar `Context`. Neither says that a PyTorch
module, CUDA kernel, or vendor library agrees with the graph.

When the page builds, Lean checks the following signature against the imported TorchLean modules.
That signature exposes the scalar type and the success conditions needed to apply the result.

```lean (name := loweringThms)
-- The values a successful run computes at a node are at
-- the node's ports.
#check @TorchLean.IR.Graph.run_ports
```
```leanOutput loweringThms (whitespace := lax)
@TorchLean.IR.Graph.run_ports : ∀ {α : Type} [inst : TorchLean.Storage α]
  [inst_1 : TorchLean.Context α] [inst_2 : TorchLean.Op.Capabilities α]
  {g : TorchLean.IR.Graph} {payload : TorchLean.IR.Payload α}
  {inputs : List (TorchLean.Op.Value α)} {values : Array (List (TorchLean.Op.Value α))},
  g.run payload inputs = Except.ok values →
    ∀ {j : ℕ} {n : TorchLean.IR.Node} {vs : List (TorchLean.Op.Value α)},
      g.nodes[j]? = some n → values[j]? = some vs → List.map Sigma.fst vs = n.outs
```

A `#check` prints a declaration's type. It doesn't execute the model or apply the theorem to this
classifier.

In this signature, `∀` means “for every,” `{...}` contains arguments Lean can often infer, and
`[...]` requests an interface instance, such as the scalar operations in `Context α`. An arrow
`P → Q` says that a proof of `P` yields a proof of `Q`. Long namespaces tell Lean which definition
is meant, but the important reading order is the objects, then the hypotheses, then the conclusion.
Here the theorem needs a successful run before it says anything about the ports of a node's values.

The port theorem has a different role. Once a run succeeds, it identifies the port of each
computed value, its element sort and shape, with the port its node declares
(`TorchLean.IR.Node.outs`). It has no premise about the structural check `TorchLean.IR.Graph.check`:
the evaluator returns the values of a successful node at the ports the node declares
(`TorchLean.IR.Node.eval_ports`), whether or not the graph is well formed. It supplies no existence
result for the value table: a successful run is an assumption.

# IBP

Interval bound propagation assigns each node a box. For an affine layer

$$`y=Wx+b,\qquad x\in[\ell,u],`

the usual sign split gives

$$`\ell_y=W^+\ell+W^-u+b,\qquad
u_y=W^+u+W^-\ell+b,`

where $`W^+=\max(W,0)` and $`W^-=\min(W,0)`. Monotone activations transform endpoints: ReLU maps
$`[\ell,u]` to $`[\max(0,\ell),\max(0,u)]`. Elementwise multiplication needs all endpoint products.

For one weighted input, a nonnegative weight preserves the order of the endpoints and a
nonpositive weight reverses it. Splitting the weight into these two parts gives the lower and upper
bounds below. Summing this scalar argument across a row gives the affine-layer rule.

```lean
/-- Every real number is the sum of its positive and
negative parts, written with `max` and `min` so that
both summands have a known sign. -/
theorem weight_eq_max_add_min (w : ℝ) :
    w = max w 0 + min w 0 := by
  rcases le_total w 0 with h | h
  · simp [max_eq_right h, min_eq_left h]
  · simp [max_eq_left h, min_eq_right h]

example (w x l u b : ℝ) (hl : l ≤ x) (hu : x ≤ u) :
    max w 0 * l + min w 0 * u + b ≤ w * x + b := by
  -- Positive weights use the lower input endpoint, and
  -- negative weights use the upper.
  have h1 : max w 0 * l ≤ max w 0 * x :=
    mul_le_mul_of_nonneg_left hl (le_max_right _ _)
  have h2 : min w 0 * u ≤ min w 0 * x :=
    mul_le_mul_of_nonpos_left hu (min_le_right _ _)
  calc max w 0 * l + min w 0 * u + b
      ≤ max w 0 * x + min w 0 * x + b := by
        linarith
    _ = w * x + b := by
        rw [← add_mul, ← weight_eq_max_add_min]

example (w x l u b : ℝ) (hl : l ≤ x) (hu : x ≤ u) :
    w * x + b ≤ max w 0 * u + min w 0 * l + b := by
  -- Reverse the endpoint choices to obtain an upper bound.
  have h1 : max w 0 * x ≤ max w 0 * u :=
    mul_le_mul_of_nonneg_left hu (le_max_right _ _)
  have h2 : min w 0 * x ≤ min w 0 * l :=
    mul_le_mul_of_nonpos_left hl (min_le_right _ _)
  calc w * x + b
      = max w 0 * x + min w 0 * x + b := by
        rw [← add_mul, ← weight_eq_max_add_min]
    _ ≤ max w 0 * u + min w 0 * l + b := by
        linarith
```

The graph proof applies local enclosure arguments like these at each node in topological order. Its
induction maintains enclosure for every earlier value, so a node can use the bounds on its parents.
Tensor dimensions, parameter lookup, and missing values must also agree between the bound pass and
the semantic evaluator. Directed floating arithmetic adds the obligation that rounded endpoints
remain on the enclosing side of the exact result.

The bound passes run on graphs of registered operators (`TorchLean.IR.Graph`). A lowered model's
graph is one (the field `graph` of `TorchLean.IR.LoweredModel`), with its weights and biases as
constant nodes. The generic real soundness theorem is `Op.certLocalOK_encloses`. It requires:

- exact local certificate consistency `CertLocalOK`
- exact local value consistency `SemLocalOK`: the value table is a fixpoint of the graph's node
  evaluation
- seed boxes that enclose the graph inputs

From these premises the theorem concludes that each certificate box encloses the value of its node
in the table. The checker boxes input and constant nodes, the operators of kind `shared` in the
table `Op.ruleKind`, and tanh, sigmoid, sine, and cosine. A node of another operator has no
certified box, so the theorem needs no assumption on the operators or on their order.

`Op.certify` computes the certificate node by node, and on a sorted graph `Op.certify_localOK`
proves that it satisfies `CertLocalOK`. `Op.semLocalOK_iff_run` proves that a value table is a
fixpoint of the node evaluation exactly when it holds the outputs of a successful run, so the
enclosure holds in every successful real run whose graph inputs the seed boxes contain. Neither
`Op.certLocalOK_encloses` nor `Op.semLocalOK_iff_run` has a sorting premise, since the evaluation
of a node reads only the nodes before it.

The executable pass `Op.runIBP` has its own theorem at every scalar with lawful endpoint
operations. `Op.Directed.encloses_of_run` requires `LawfulBoundOps` and `LawfulNonlinearBoundOps`,
a real payload holding the real value of each rounded constant (`RealConsts`), seed boxes that
enclose the graph inputs, and a successful real run. It concludes that every box the pass returns
encloses the flat value of its node in that run. There is no operator restriction: every operator
the pass boxes has a proved rule (`Op.Directed.supported_of_ibp?`). Both the reals and the
rounded-real `FP32` model satisfy these scalar requirements.

Since the run supplies the real values, no equation at a node is assumed: the operators' own
semantics fix the coordinate maps, reductions, normalization formulas, and convolutions that the
boxes must enclose. Supported rules include convolution, binary matrix multiplication, structural
operations, and normalization, and the proof derives their intermediate enclosures by induction. An
operator without an interval rule, invalid shapes, missing constants, and unavailable arithmetic
transfers can leave an entry as `none`, but the theorem establishes enclosure whenever a box is
returned.

Random operators carry their key as an attribute, and the pass bounds them by their range. Lemmas
in {src "TorchLean/NN/Random/Range.lean"}[`Random.Range`] prove that each
uniform value lies in $`[0,1)` and each mask value is zero or one, with keep probabilities zero and
one giving the corresponding constant masks. Agreement with a separate rounded or native random
implementation remains a numerical correspondence question. Model lowering rejects runtime random
draws because their keys count random calls, not IR nodes. Lowering must not silently substitute a
different mask. Deterministic evaluation-mode dropout doesn't draw a mask.

CROWN's backward theorems start from the same run: `Op.graphPoint_of_run` builds their
real point from it, and `Op.equation_of_run` gives every operator node its real
equation. At a convolution with constant kernel and bias, the backward pass applies the transposed
convolution rounded outward, which encloses the real adjoint (`Op.convBack?_sound`).
Structural operations use the same coordinate maps in both directions.

`Op.backwardObjectiveBox?_of_run` in
{src "TorchLean/Verify/CROWN/Op/BackwardRun.lean"}[`BackwardRun`] then connects interval
propagation, directed backward propagation, and final interval evaluation. It requires an interval
table whose boxes enclose the run, such as the boxes of `Op.runIBP` (`Op.runIBP_encloses`),
an objective of the output's width, and an input box enclosing graph input `0`. Every successful
`.ok` query contains the real objective. No intermediate enclosure or node equation is assumed.
`Op.outputBoxCROWN?_of_run` is the same statement for the CROWN output box.

These theorems are about runs of the operator graph. For a lowered model,
`TorchLean.IR.lowerForward_eq_ok` identifies the lowered graph and payload with those that
`TorchLean.IR.OpGraph.toOp` builds from the recorded typed graph, and
`TorchLean.IR.OpGraph.eval_ok_of_admissible` proves that the run of such a graph and payload
succeeds on every input tensor of the input shape.

```lean (name := ibpThms)
-- Locally consistent real boxes enclose the values of
-- a fixpoint of the node evaluation.
#check @TorchLean.CROWN.Op.certLocalOK_encloses
-- The certificate the checker's step computes is locally
-- consistent on a sorted graph.
#check @TorchLean.CROWN.Op.certify_localOK
-- The fixpoints are the outputs of the successful runs.
#check @TorchLean.CROWN.Op.semLocalOK_iff_run
-- The executable interval pass at a lawful scalar.
#check @TorchLean.CROWN.Op.Directed.encloses_of_run
```
```leanOutput ibpThms (whitespace := lax)
certLocalOK_encloses : ∀ (G : TorchLean.IR.Graph) (P : TorchLean.IR.Payload ℝ)
  (seeds : ℕ → Option (TorchLean.CROWN.FlatBox ℝ)) (cert : Array (Option
      (TorchLean.CROWN.FlatBox ℝ)))
  (inputs : List (TorchLean.Op.Value ℝ)) (values : Array (List (TorchLean.Op.Value ℝ))),
  CertLocalOK G P seeds cert →
    SemLocalOK G P inputs values →
      (∀ (id k : ℕ) (p : TorchLean.Port),
          G.nodes[id]? = some (TorchLean.IR.Node.input k p) →
            ∀ (B : TorchLean.CROWN.FlatBox ℝ),
              seeds id = some B →
                ∀ (x : TorchLean.Op.Value ℝ) (y : TorchLean.Port.Carrier ℝ p)
                  (v : TorchLean.CROWN.Graph.FlatTensor ℝ),
                  inputs[k]? = some x → p.read x = Except.ok y → flatOf ⟨p, y⟩ = some v →
                      Directed.Encloses B v) →
        ∀ id < G.nodes.size,
          ∀ (B : TorchLean.CROWN.FlatBox ℝ) (v : TorchLean.CROWN.Graph.FlatTensor ℝ),
            cert[id]! = some B → runVal values id = some v → Directed.Encloses B v
```
```leanOutput ibpThms (whitespace := lax)
@certify_localOK : ∀ {G : TorchLean.IR.Graph} {P : TorchLean.IR.Payload ℝ} {seeds : ℕ → Option
    (TorchLean.CROWN.FlatBox ℝ)},
  TopoSorted G → CertLocalOK G P seeds (certify G P seeds)
```
```leanOutput ibpThms (whitespace := lax)
@semLocalOK_iff_run : ∀ {α : Type} [inst : TorchLean.Storage α] [inst_1 : TorchLean.Context α]
  [inst_2 : TorchLean.Op.Capabilities α] {G : TorchLean.IR.Graph} {P : TorchLean.IR.Payload α}
  {inputs : List (TorchLean.Op.Value α)} {values : Array (List (TorchLean.Op.Value α))},
  SemLocalOK G P inputs values ↔ G.run P inputs = Except.ok values
```
```leanOutput ibpThms (whitespace := lax)
@Directed.encloses_of_run : ∀ {α : Type} [inst : TorchLean.Storage α] [inst_1 : TorchLean.Context α]
  [inst_2 : TorchLean.CROWN.BoundOps α] [inst_3 : TorchLean.CROWN.LawfulBoundOps α]
  [inst_4 : TorchLean.CROWN.NonlinearBoundOps α] [TorchLean.CROWN.LawfulNonlinearBoundOps α]
      (G : TorchLean.IR.Graph)
  (P : TorchLean.IR.Payload α) (Pr : TorchLean.IR.Payload ℝ) (seeds : ℕ → Option
      (TorchLean.CROWN.FlatBox α))
  (inputs : List (TorchLean.Op.Value ℝ)) (values : Array (List (TorchLean.Op.Value ℝ))),
  Directed.RealConsts Pr P →
    (∀ (id k : ℕ) (p : TorchLean.Port),
        G.nodes[id]? = some (TorchLean.IR.Node.input k p) →
          ∀ (B : TorchLean.CROWN.FlatBox α),
            seeds id = some B →
              ∀ (x : TorchLean.Op.Value ℝ) (y : TorchLean.Port.Carrier ℝ p)
                  (v : TorchLean.CROWN.Graph.FlatTensor ℝ),
                inputs[k]? = some x → p.read x = Except.ok y → flatOf ⟨p, y⟩ = some v →
                    Directed.Encloses B v) →
      G.run Pr inputs = Except.ok values →
        ∀ id < G.nodes.size,
          ∀ (B : TorchLean.CROWN.FlatBox α) (v : TorchLean.CROWN.Graph.FlatTensor ℝ),
            (runIBP G P seeds)[id]! = some B → runVal values id = some v → Directed.Encloses B v
```

The enclosure conclusions assume both entries, a box (`cert[id]! = some B`) and a value
(`runVal values id = some v`). These theorems therefore justify an available bound without promising
that the pass produces every bound an application needs.

Each name describes a piece of the graph induction. `CertLocalOK` says that the stored boxes follow
the checker's rules, and `SemLocalOK` says that the stored values follow the graph's node
evaluation. The seed premise starts the induction by placing the actual input values inside the
input boxes. For the classifier, the conclusion we need in the end is enclosure at the output node,
and these hypotheses explain how the proof can reach that node.

At `ℝ` the executable pass and the certificate are also related. On a graph whose every operator
has the same rule in the pass and in the checker (`SharedOnly`: kind `shared` or `none` in
`Op.ruleKind`), `Op.runIBP_eq_certify` proves that the pass computes the certificate, so the
certificate theorems apply to its table. No second arithmetic argument is needed for that
substitution.

## IBP On A Two-Layer Affine Network

These theorems establish enclosure, but a box can be sound and too wide to settle the property. We
can measure that excess width by running the pass on a graph whose exact range we can calculate by
hand.

Consider the map $`x\mapsto (x_0+x_1)-(x_0-x_1)`, written as two affine layers. Composed, it's
exactly $`2x_1`, so on the unit box $`[-1,1]^2` the exact output range is $`[-2,2]`. Since the
bound passes work on graphs, not on layer combinators, we spell the graph out node by node with the
graph builder (`TorchLean.IR.Graph.Builder`) and keep the weights beside it in a payload
(`TorchLean.IR.Payload`), keyed by the constant nodes that read them:

```lean
/-- Two affine layers computing `(x₀+x₁) - (x₀-x₁)`.
Node 3 produces both intermediate coordinates, and node 6
subtracts them. Nodes 1, 2, 4 and 5 hold the weights and
biases. -/
def depGraph : Except String TorchLean.IR.Graph :=
  let (b, x) :=
    ({} : TorchLean.IR.Graph.Builder).input 0 (.value [2])
  let (b, w1) := b.const (.value [2, 2])
  let (b, bias1) := b.const (.value [2])
  let (b, h) := b.op
    (TorchLean.Op.linear (lead := []) (inDim := 2)
      (outDim := 2)) [w1, bias1, x]
  let (b, w2) := b.const (.value [1, 2])
  let (b, bias2) := b.const (.value [1])
  let (b, y) := b.op
    (TorchLean.Op.linear (lead := []) (inDim := 2)
      (outDim := 1)) ([w2, bias2] ++ h)
  b.build y.toArray

/-- The input region: the unit box in the sup norm. -/
def depBox : TorchLean.CROWN.FlatBox Float :=
  { dim := 2, lo := [-1.0, -1.0], hi := [1.0, 1.0] }

/-- The weights and biases of the two layers. -/
def depW1 : TorchLean.Tensor Float [2, 2] :=
  [[1.0, 1.0], [1.0, -1.0]]
def depB1 : TorchLean.Tensor Float [2] := [0.0, 0.0]
def depW2 : TorchLean.Tensor Float [1, 2] :=
  [[1.0, -1.0]]
def depB2 : TorchLean.Tensor Float [1] := [0.0]

/-- The payload: the weights and biases at the constant
nodes that read them. -/
def depPayload : TorchLean.IR.Payload Float :=
  { const? := fun
      | 1 => some ⟨.value [2, 2], depW1⟩
      | 2 => some ⟨.value [2], depB1⟩
      | 4 => some ⟨.value [1, 2], depW2⟩
      | 5 => some ⟨.value [1], depB2⟩
      | _ => none }
```

Both passes read the graph with its payload, and the input box seeds the graph input
(`Op.inputSeeds`). We run interval propagation (`Op.runIBP`) and CROWN (`Op.outputBoxCROWN?`) and
print the box of the graph's one output, node 6:

```lean (name := depRun)
-- Propagate the unit box through the graph. Its one
-- output is node 6.
#eval do
  let g ← IO.ofExcept depGraph
  let boxes :=
    runIBP g depPayload (inputSeeds g [depBox])
  match outputBox? g boxes 0 with
  | .ok b =>
    IO.println
      s!"IBP   lo={b.lo.pretty} hi={b.hi.pretty}"
  | .error e => IO.println s!"IBP failed: {e}"
  match outputBoxCROWN? g depPayload depBox 0 with
  | .ok b =>
    IO.println
      s!"CROWN lo={b.lo.pretty} hi={b.hi.pretty}"
  | .error e => IO.println s!"CROWN failed: {e}"
```
```leanOutput depRun
IBP   lo=[-4.000000] hi=[4.000000]
CROWN lo=[-2.000000] hi=[2.000000]
```

IBP returns twice the exact width. Its box for node 3 is
$`[-2,2]\times[-2,2]`, and each coordinate really does range over $`[-2,2]`. What the box cannot
record is that the two coordinates move together: whenever $`x_0+x_1` is at its maximum, $`x_0-x_1`
is $`0`. The second layer subtracts them, and interval subtraction has to assume the worst case
that the first is at $`2` while the second is at $`-2`, which no input achieves. This is dependency
loss: each interval retains a coordinate's range but loses its relation to the others. Repeating
this loss across layers can make box propagation alone
{Informal.citep gowal2018}[] progressively looser as depth grows.

CROWN {Informal.citep crown2018}[] carries affine forms in the input instead of boxes. Its backward
pass sweeps the output's coefficients back through both layers, so the cancellation happens
symbolically before any interval is evaluated: $`1\cdot(x_0+x_1)-1\cdot(x_0-x_1)` becomes $`2x_1`,
and only then is the box substituted. The result is the exact range. That exactness is a property
of this graph, not of the method: the graph is purely affine, so the affine form is the function.
An unstable ReLU can introduce a strict relaxation, while stable ReLUs may remain exact. How much
CROWN improves over IBP depends on the chosen affine bounds and graph.

This real affine map attains its extrema at box corners. We evaluate the same graph and payload at
the corners in `Float` with `TorchLean.IR.Graph.eval`, which also lets us compare graph evaluation
with the bound pass:

```lean (name := depCorners)
/-- Evaluate `depGraph` at one input and read off the
single output coordinate. -/
def depEval (x : TorchLean.Tensor Float [2]) :
    Except String Float := do
  let g ← depGraph
  match ← g.eval depPayload [⟨.value [2], x⟩] with
  | [⟨⟨.value, [1]⟩, t⟩] =>
    pure (TorchLean.Tensor.getScalar t 0)
  | _ => throw "expected one vector of length one"

#eval do
  let corners : List (TorchLean.Tensor Float [2]) :=
    [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
  for x in corners do
    match depEval x with
    | .ok y => IO.println s!"corner value: {y}"
    | .error e => IO.println s!"failed: {e}"
```
```leanOutput depCorners
corner value: -2.000000
corner value: 2.000000
corner value: -2.000000
corner value: 2.000000
```

So $`\pm 2` are attained and the true range is $`[-2,2]`: CROWN's box is tight here, and IBP's
extra factor of two is pure over-approximation. Neither pass is wrong. Soundness only promises
containment, and a sound bound that is too wide reports "cannot verify" on a network that is in
fact robust. Tighter bounds may cost more computation, and input subdivision lets us explore that
tradeoff while keeping the local interval rules fixed.

## Tensor Geometry In A Bound Pass

The box representation stores one interval per scalar. But the graph records the shape that tells
us how those scalars interact. A matrix product multiplies the matrices on the final two axes,
broadcasting the leading axes. Vectors participate without a separate reshape:

```lean (name := boundMatmulShapes)
example :
    TorchLean.IR.inferMatmulOutShape
      [2, 3, 4, 5] [2, 3, 5, 6] =
        .ok [2, 3, 4, 6] := by decide

example :
    TorchLean.IR.inferMatmulOutShape
      [1, 4, 5] [2, 5, 6] = .ok [2, 4, 6] := by decide

example :
    TorchLean.IR.inferMatmulOutShape
      [5] [2, 5, 6] = .ok [2, 6] := by decide

example :
    TorchLean.IR.inferMatmulOutShape
      [5] [5] = .ok [] := by decide
```

The first example has six independent $`4\times5` by $`5\times6` products. In the second, one left
matrix is reused across two right matrices. In the third, one vector is multiplied by each right
matrix, and the fourth is a dot product with a scalar result. The runtime, interval pass, and
backward pass use the same shape contract and coordinate maps. When an operand is reused by
broadcasting, its gradient accumulates contributions from every use. In the
{src "TorchLean/Verify/CROWN/Op/RunValues.lean"}[runtime proof], `Op.flat_matmul` shows that the
flat value of a matrix product is the flat contraction of its operands' flat values over the
operator's layout.

For a binary product, the backward pass bounds each term by a McCormick plane of the two operand
boxes when the term's coefficient has one sign, and by the interval product of the boxes otherwise.
An operand that is a single point passes its coefficient exactly. Since the planes are relaxations,
this rule can lose correlations that an exact affine transfer would retain.

Concatenation follows the selected dimension, with at least one parent and equal extents on every
other dimension. A parent can have an empty dimension:

```lean (name := boundConcatShape)
example : TorchLean.Op
    [.value [2, 1, 3], .value [2, 0, 3], .value [2, 2, 3]]
    [.value [2, 3, 3]] :=
  TorchLean.Op.cat (s := [2, 0, 3]) ⟨1, by decide⟩ [1, 0, 2]
```

Here each of the two leading coordinates gets one slice from the first parent and two from the
third, and the middle parent contributes no entries. Appending whole flat buffers would mix those
leading coordinates. Instead, the value, interval, and affine transfers share the same coordinate
map, and the backward pass uses it to split output coefficients among parent occurrences. If
$`y=\operatorname{cat}(x,x)` has coefficient slices $`u` and $`v`, both slices reach the same
input, giving coefficient $`u+v`. So repeated parent ids require accumulation.

The certificate checker places the boxes of the parents by this layout too (`Op.catLayout`), and the
certificate induction includes concatenation. In the
{src "TorchLean/Verify/CROWN/Op/RunValues.lean"}[runtime proof] we also show that the flat value of
`Tensor.cat` along any valid dimension is the checked flat concatenation of the flat values of its
inputs over this layout (`Op.flat_cat`).

For LayerNorm on `[2, 2, 2]`, the verifier normalizes the last axis, which gives four independent
rows of two coordinates. If the rows hold $`0,1`, $`2,3`, $`4,5` and $`6,7`, their means are
$`0.5`, $`2.5`, $`4.5` and $`6.5`. Flattening all eight coordinates into one normalization would
give mean $`3.5` and change every output. So the scale and bias carry the last axis's shape `[2]`.
The leading axes must hold at least one row, so an empty batch such as `[0, 2, 2]` is rejected, and
the last axis must hold at least one coordinate.

Softmax value bounds also follow the selected axis. On `[2, 3, 1]`, axes zero and one use
the conservative range $`[0,1]` for each coordinate. Axis two has one entry per softmax row,
so its range is $`[1,1]`. These value rules do not require a directed implementation of
exponential.

Convolution transfers share the IR's validation of channels, groups, stride, dilation, and
asymmetric padding. Any leading batch shape is retained, and each batch coordinate is evaluated
independently. A convolution node in PyTorch's layout reads its kernel as
`(outC, inC / groups, k...)`, keeping for each output channel the input channels of its group
(`TorchLean.Conv.packedKernel`). The backward pass sends coefficients through a convolution by the
transposed convolution over the same configuration, so it never forms the dense matrix below,
which can be much larger than the kernel tensor.

A convolution is still an affine map of its flat input. For real tensors,

$$`\operatorname{vec}(C(x))=A\,\operatorname{vec}(x)+b_{\mathrm{broadcast}}.`

The theorem `convLinearMatrix_spec` in {src "TorchLean/Verify/CROWN/Conv/Affine.lean"}[the
convolution proof] establishes this equality for grouped channels, dilation, asymmetric padding,
and any leading batch shape. The interval pass sums the kernel contributions with directed products
and additions, without constructing this matrix. Its real enclosure theorem is in
{src "TorchLean/Verify/CROWN/Conv/Enclosure.lean"}[the directed convolution proof]. Graph execution
checks the configuration. On graphs of registered operators, `Op.Directed.encloses_conv` proves
that the interval box of `Op.conv` encloses its forward value at the real values of the kernel and
bias, at every lawful scalar and so at `ℝ`, on an input whose flattening the input box encloses.
Its proof goes through `Op.convBox?_of_convOpBox?` in
{src "TorchLean/Verify/CROWN/Op/Conv.lean"}[the operator proof], which identifies the operator's
box with the box of the convolution proofs. Convolution is included in the real certificate
induction and in `Op.Directed.encloses_of_run`. A claim about rounded execution still needs its
numerical correspondence.

Each elementwise interval map in `CROWN.IBP` preserves arbitrary tensor shapes (scalars
and empty tensors included), which extends where the same scalar formulas can be applied but
doesn't change their arithmetic hypotheses. For a theorem about a complete graph, the executable
operator contract and the theorem's premises both have to match the graph we're using.

## Input Subdivision

Another general option is to partition an input box and evaluate the same graph on every piece.
`TorchLean.CROWN.Graph.Refinement.bound enclose box splitBudget` does this for any enclosure
procedure `enclose` without changing any layer transfer, so with interval bound propagation it
applies to any graph IBP supports (mixed architectures included). For example, splitting $`[-1,1]`
at zero tightens the ordinary interval bound on $`\max(x,0)+\max(-x,0)` from approximately $`[0,2]`
to $`[0,1]`.

The budget counts total splits, so at most $`2b+1` IBP calls are made for budget $`b`. Both
children must produce bounds (a failed branch keeps the parent result). Their results are combined
by a hull, then intersected with the parent bound. Since the split boundary is shared, all original
inputs are preserved, including the cut itself. Adjacent floating-point endpoints with no interior
midpoint stay unsplit.

Subdivision can reduce dependency loss across layers, but it doesn't promise improvement
everywhere: a linear transfer may already be tight, and an input-independent activation range may
remain broad. `TorchLean.CROWN.Graph.Refinement.splitAt_covers` proves real input coverage of
each split. Sound output bounds still depend on the underlying transfers, so subdivision doesn't
establish universal soundness for rounded LayerNorm or other operations.

Artifact replay can opt in with
`TorchLean.Certificate.IBP.check G P inputs output path (refinement := some (k, splitBudget))`,
which subdivides the box of graph input `k` of the graph of registered operators `G` (by default
the check doesn't subdivide). This refinement is a graph-level option, separate from the high-level
trainer API.

## Trainer Verification API

`trainer.train` returns a snapshot of the trained parameters. Its `verify` operation uses that
snapshot and accepts a tensor center and named choices such as `radius`, `property`, and
`algorithm`. A result obtained from `session.finish` uses the same snapshot semantics:
further training or loading a checkpoint into the session does not change the model it verifies.

Import `TorchLean.Verify.Trainer` to use the `trained.verify` call shown above. The default method
is `crown`. `ibp` and `alphaCROWN` (fixed-relaxation alpha-CROWN with the ReLU phases IBP proves)
are available for comparison. On a rounded scalar such as binary32, `crown` runs directed CROWN,
rounding coefficients and constants outward, and `alphaCROWN` runs directed CROWN too, so its
report names `CROWN`. To inspect graph nodes and intermediate bounds, import
`TorchLean.Verify.Lowering`.

This high-level verifier belongs to the ordinary binary32 training path. A session opened with
`trainer.openTyped` preserves its chosen scalar in training and prediction but supplies no
verifier, so calling `verify` on its result reports that verification is unavailable. Choosing a
wider training format doesn't by itself supply bound arithmetic or soundness proofs for it.

## Bound Propagation With `auto_LiRPA`

With PyTorch, we can use `auto_LiRPA` {Informal.citep autolirpa2020}[] to propagate bounds through
the model. Its wrapper takes the perturbation region along with the model input:

```
# Wrap the same model with an interface that propagates
# bounds.
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

model = BoundedModule(mlp, torch.empty_like(center))
# Each coordinate may vary by at most 0.10 around the
# center.
ptb = PerturbationLpNorm(norm=float("inf"), eps=0.10)
lb, ub = model.compute_bounds(
    x=(BoundedTensor(center, ptb),), method="alpha-CROWN")
# Compare class zero's lower bound with class one's upper
# bound.
print((lb[0, 0] - ub[0, 1]).item())
```

Both examples ask for bounds over the same kind of perturbation region, then compare one class's
lower bound with another's upper bound. How those endpoints are computed depends on the choice of
bound method and arithmetic. In TorchLean, the real-semantic transfer theorems and local
certificate predicates describe the evidence needed to justify them, but applying those theorems to
the high-level Float report still requires proofs about the produced data and rounded arithmetic
(the trust ledger below records those obligations).

# CROWN And Alpha-CROWN

CROWN {Informal.citep crown2018}[] propagates affine lower and upper forms, not only boxes.
Bounding a network by a convex outer approximation and then optimizing inside it predates CROWN
{Informal.citep wongkolter2018}[], while CROWN obtains bounds with a backward pass. At an uncertain
ReLU with $`\ell<0<u`, the secant upper relaxation is

$$`\operatorname{ReLU}(z)
\le \frac{u}{u-\ell}(z-\ell),`

while a lower relaxation may use a slope $`\alpha` constrained to a valid range. Alpha-CROWN
optimizes these slopes. Alpha-beta-CROWN {Informal.citep betacrown2021}[] also records branch
phases: an active branch uses $`z\ge 0`, and an inactive branch uses $`z\le 0`. Splitting on a
phase can tighten the relaxation, at the cost of additional bound passes.

TorchLean proves the CROWN passes sound on graphs of registered operators. CROWN's forward pass
`Op.alphaStep?` builds affine lower and upper forms in the input node by node: the identity at the
designated input, constants, detach, a linear layer with constant weight and bias, ReLU with α
slopes or with phases, a sum over every axis, and reshape. Matrix products and custom operators get
no bounds, and every other operator gets the constant forms of its own interval box. At `ℝ`,
`Op.Alpha.encloses_of_run` proves that every bound of a table whose present entries the step
replays encloses, at the input point, the flat value of its node in a successful real run. It
assumes that the interval table encloses the run, that the designated input node holds the point,
and that the slopes lie in $`[0,1]` (`AlphaOK`). The phased step rejects phase choices inconsistent
with the current IBP interval. `Op.runAlphaCROWN_encloses` covers the table of the
executable forward pass `Op.runAlphaCROWN`, which takes the default slopes and the phases the
intervals force, so it has no slope premise.

CROWN's backward pass bounds a linear objective of one node by affine forms in the input. It sweeps
interval coefficients from the objective's node to the input, rounding outward, and bounds the
coefficient of a node without a coefficient rule by that node's interval box. At a scalar with
lawful endpoint operations, `Op.outputBoxCROWN?_of_run` proves that a successful CROWN
output box encloses the output of a successful real run, when the payload holds the reals of the
rounded constants and the input box encloses graph input `0`.
`Op.backwardObjectiveBox?_of_run` is the same for an objective query over an interval
table that encloses the run. Full CROWN meets each node's interval box with its affine bounds on
the input box, and `Op.crownBoxes_encloses` and `Op.runCROWN_encloses` prove that
its boxes and affine bounds enclose the run as well.

These theorems take the interval table as an assumption about the run: its boxes enclose the run's
flat values. `Op.Directed.encloses_of_run` and `Op.runIBP_encloses` prove that assumption
for the boxes of `Op.runIBP`.

The executable `Op.outputBoxAlphaCROWN?` seeds IBP with the requested input box. On backends that
support exact affine reassociation, `Op.runAlphaCROWN` takes those intervals, infers every ReLU
phase they already force, and runs the forward pass with them. Unstable phases remain unsplit and
use the default alpha relaxation. Rounded scalar backends instead run the backward pass through
`Op.outputBoxCROWN?`. Nearest-rounded affine coefficients can't be treated as exact-real
enclosures. Both paths are distinct from the external Alpha-Beta-CROWN optimizer's branch-and-bound
search.

```lean (name := crownThms)
-- Every replayed α-CROWN bound encloses the run at the
-- input point.
#check @TorchLean.CROWN.Op.Alpha.encloses_of_run
-- The same for the table of the executable forward pass.
#check @TorchLean.CROWN.Op.runAlphaCROWN_encloses
-- A successful CROWN output box encloses the real output.
#check @TorchLean.CROWN.Op.outputBoxCROWN?_of_run
```
```leanOutput crownThms (whitespace := lax)
Alpha.encloses_of_run : ∀ (G : TorchLean.IR.Graph) (P : TorchLean.IR.Payload ℝ)
  (ibp : Array (Option (TorchLean.CROWN.FlatBox ℝ))) (alpha : Array (Option
      (TorchLean.CROWN.Graph.FlatTensor ℝ)))
  (phases : Array (Option (Array ℤ))) (cert : Array (Option
      (TorchLean.CROWN.Graph.FlatAffineBounds ℝ)))
  (ctx : TorchLean.CROWN.Graph.AffineCtx) (x : TorchLean.Tensor ℝ [ctx.inputDim])
  (inputs : List (TorchLean.Op.Value ℝ)) (values : Array (List (TorchLean.Op.Value ℝ))),
  (∀ (id : ℕ) (b : TorchLean.CROWN.Graph.FlatAffineBounds ℝ),
      id < G.nodes.size → cert[id]! = some b →
          alphaStep? G P ibp alpha phases cert ctx id = some b) →
    (∀ (id : ℕ) (B : TorchLean.CROWN.FlatBox ℝ) (v : TorchLean.CROWN.Graph.FlatTensor ℝ),
        ibp[id]?.join = some B → runVal values id = some v → Directed.Encloses B v) →
      (∀ (k : ℕ) (p : TorchLean.Port) (v : TorchLean.CROWN.Graph.FlatTensor ℝ),
          G.nodes[ctx.inputId]? = some (TorchLean.IR.Node.input k p) →
            runVal values ctx.inputId = some v →
              ∃ (h : v.n = ctx.inputDim), TorchLean.CROWN.Graph.castDimScalar h v.v = x) →
        AlphaOK alpha →
          G.run P inputs = Except.ok values →
            ∀ id < G.nodes.size,
              ∀ (b : TorchLean.CROWN.Graph.FlatAffineBounds ℝ)
                (v : TorchLean.CROWN.Graph.FlatTensor ℝ),
                cert[id]! = some b → runVal values id = some v → EnclosesAtInput ctx x b v
```
```leanOutput crownThms (whitespace := lax)
runAlphaCROWN_encloses : ∀ (G : TorchLean.IR.Graph) (P : TorchLean.IR.Payload ℝ)
  (ctx : TorchLean.CROWN.Graph.AffineCtx) (ibp : Array (Option (TorchLean.CROWN.FlatBox ℝ)))
  (x : TorchLean.Tensor ℝ [ctx.inputDim]) (inputs : List (TorchLean.Op.Value ℝ))
  (values : Array (List (TorchLean.Op.Value ℝ))),
  (∀ (id : ℕ) (B : TorchLean.CROWN.FlatBox ℝ) (v : TorchLean.CROWN.Graph.FlatTensor ℝ),
      ibp[id]?.join = some B → runVal values id = some v → Directed.Encloses B v) →
    (∀ (k : ℕ) (p : TorchLean.Port) (v : TorchLean.CROWN.Graph.FlatTensor ℝ),
        G.nodes[ctx.inputId]? = some (TorchLean.IR.Node.input k p) →
          runVal values ctx.inputId = some v →
            ∃ (h : v.n = ctx.inputDim), TorchLean.CROWN.Graph.castDimScalar h v.v = x) →
      G.run P inputs = Except.ok values →
        ∀ (id : ℕ) (b : TorchLean.CROWN.Graph.FlatAffineBounds ℝ)
          (v : TorchLean.CROWN.Graph.FlatTensor ℝ),
          (runAlphaCROWN G P ctx ibp)[id]! = some b → runVal values id = some v →
              EnclosesAtInput ctx x b v
```
```leanOutput crownThms (whitespace := lax)
@outputBoxCROWN?_of_run : ∀ {α : Type} [inst : TorchLean.Storage α] [inst_1 : TorchLean.Context α]
  [inst_2 : TorchLean.CROWN.BoundOps α] [inst_3 : TorchLean.CROWN.LawfulBoundOps α]
      {G : TorchLean.IR.Graph}
  {P : TorchLean.IR.Payload α} {Pr : TorchLean.IR.Payload ℝ} {inputs : List (TorchLean.Op.Value ℝ)}
  {values : Array (List (TorchLean.Op.Value ℝ))} [inst_4 : TorchLean.CROWN.NonlinearBoundOps α]
  [TorchLean.CROWN.LawfulNonlinearBoundOps α],
  G.run Pr inputs = Except.ok values →
    Directed.RealConsts Pr P →
      ∀ {xB : TorchLean.CROWN.FlatBox α},
        (∀ (x : TorchLean.Op.Value ℝ), inputs[0]? = some x → ∃ t,
            flatOf x = some t ∧ Directed.Encloses xB t) →
          ∀ {i out : ℕ} {ctx : TorchLean.CROWN.Graph.AffineCtx},
            outputContext G xB i = Except.ok (out, ctx) →
              ∀ {t : TorchLean.CROWN.Graph.FlatTensor ℝ},
                runVal values out = some t →
                  ∀ {result : TorchLean.CROWN.FlatBox α},
                    outputBoxCROWN? G P xB i = Except.ok result → Directed.Encloses result t
```

`Op.Alpha.encloses_of_run` holds for any supplied phases, because a phase reaches a ReLU's
relaxation only when it agrees with the interval of the ReLU's input. To assume a phase for an
unstable ReLU, a theorem would also have to restrict the input domain to that branch.

These theorems are separate from the JSON node-certificate replay diagnostic
`CROWNNode.replayDiagnostic`. That `IO (Array String)` function parses finite decimal fields into
FloatLib binary32, recomputes the complete IBP trace from trusted inputs and parameters, replays
each affine node from the affine data it recomputed for the nodes before it, and lists every
difference. A serialized interval may be wider than the recomputed interval but may not move
inward. Affine replay data must match exactly at the binary32 level. When the certificate gives
ReLU phases, the diagnostic also validates their lengths, entries, and consistency with IBP. The
diagnostic is not a safety verdict. The verdict `CROWNNode.check` decides with the exact rational
check of the next section.

The shared JSON boundary validates input regions before replay begins. Endpoint boxes require
finite arrays of the declared dimension with `lo[i] <= hi[i]`, and center-radius boxes also require
a finite nonnegative radius. Incomplete or mixed schemas are rejected. Artifact formats that
prescribe endpoints, including the alpha-beta-CROWN leaf format, request that exact schema and
don't accept the alternate center-radius notation.

`CROWNNode.replayAgrees` is the diagnostic's final agreement test. Its theorem proves that
agreement supplies local consistency for the exact FloatLib binary32 replay function of the
diagnostic.

```lean (name := replayThms)
-- If the certificate agrees with the replay, its stored
-- affine entries satisfy the replay rule.
#check @CROWNNode.replayAgrees_spec
```
```leanOutput replayThms (whitespace := lax)
CROWNNode.replayAgrees_spec : ∀ (cert : CROWNNode) (G : TorchLean.IR.Graph)
  (P :
    TorchLean.IR.Payload
      (ExecFloat.Binary 8 23 FloatFormat.Encoding.ieee (FloatFormat.Encoding.ieee.defaultBias 8)
        CROWNNode.nodeMismatch._proof_1 CROWNNode.nodeMismatch._proof_2
        CROWNNode.nodeMismatch._proof_3 CROWNNode.nodeMismatch._proof_4))
  (authoritativeIbp :
    Array
      (Option
        (TorchLean.CROWN.FlatBox
          (ExecFloat.Binary 8 23 FloatFormat.Encoding.ieee
            (FloatFormat.Encoding.ieee.defaultBias 8)
            CROWNNode.nodeMismatch._proof_1 CROWNNode.nodeMismatch._proof_2
            CROWNNode.nodeMismatch._proof_3 CROWNNode.nodeMismatch._proof_4))))
  (diagnosticsOk : Bool),
  cert.replayAgrees G P authoritativeIbp diagnosticsOk = true →
    cert.crown.size = G.nodes.size ∧
      ∀ id < G.nodes.size,
        cert.crown[id]! = CROWNNode.replayStep G P authoritativeIbp cert cert.crown id
```

The output says: for any certificate, graph, payload, interval trace, and diagnostic flag, *if this
agreement test returns `true`, then the affine table in the certificate agrees with the specified
replay rule*. In the signature, the arrow before the conjunction separates the assumption from the
conclusion. No classifier margin appears in this statement.

Each argument has a concrete job:

:::table +header
*
  * Name in the signature
  * Meaning in the check
*
  * `cert`
  * The proposed certificate, including its affine table `cert.crown` and relaxation data.
*
  * `G`
  * The graph of registered operators whose nodes the table describes.
*
  * `P`
  * The payload: the weights, biases, and other constants used by replay.
*
  * `authoritativeIbp`
  * The interval table given to the agreement test. The IO wrapper constructs it by rerunning IBP.
*
  * `diagnosticsOk`
  * The combined Boolean result of the wrapper's additional checks.
*
  * `replayStep ...`
  * The function that recomputes a node's affine bounds from the table and node index.
:::

`Array (Option (FlatBox (ExecFloat.Binary 8 23)))` describes the interval table's representation.
There is an array slot for each represented node, and a slot may hold a box (`some`) or no box
(`none`). A `FlatBox` stores lower and upper vectors. Their scalar type selects FloatLib binary32,
including its rounding and exceptional values. So a theorem about this table concerns those
executable values, and a real-valued conclusion needs a further refinement argument.

We can unpack the conclusion precisely. Its first conjunct requires the affine table to have the
same length as the graph's node array. By the second, at every valid node index the table entry
equals the result of `replayStep` applied to that table and index. If a producer changes an
affine coefficient, the exact comparison can reject it even when the decimal difference looks
small. Since entries are optional, local consistency alone does not assert that every node has a
bound.

The theorem connects an executable Boolean test to a Lean proposition, which lets a later proof use
the result without reasoning again about every componentwise comparison.

Two details matter when applying the result. The pure theorem accepts the interval trace as an
argument, and calling it `authoritativeIbp` doesn't prove where that trace came from. The wrapper
supplies the recomputation. Likewise, the theorem doesn't interpret the individual diagnostics
behind `diagnosticsOk`: its conclusion is the stated local consistency property. To recover the
classifier claim, we would need a soundness argument saying that these replay rules enclose the
network, bounds at the required output and a positive class margin.

In binary32 the rules need not enclose the network, because each sum, product and quotient rounds
to nearest. After the layers `h = x + 1` and `y = h + 2 ^ -24` at `x = -1`, where `y = 2 ^ -24`,
the replayed upper form of `y` is `x + 1`, which is `0` there, and `CROWNNode.replayDiagnostic`
reports no difference for a certificate holding that form. `CROWNNode.check` rejects it, because
`x + 1` lies below the exact transfer `x + 1 + 2 ^ -24`.

On the real side, `Op.Alpha.encloses_of_run` also takes a replay premise, but over real affine data
and the step at `ℝ`. The binary32 result can't be supplied as that premise on its own.

There are two different universal statements in this chain. Local replay checks every node in a
finite graph. Robustness concerns every input in a region, which contains infinitely many real
points. The transfer theorem is what connects them: each node's rule preserves an enclosure for an
arbitrary input satisfying the assumptions, and graph induction carries that fact to the output.
Checking every node is useful because of that argument. No count of checked entries replaces it,
and the printed `#check` signatures identify the argument without performing it for the displayed
classifier.

## Checking The Supplied Binary32 Bounds Against Real Arithmetic

There is a stronger decision for a vector input followed by a chain of linear and ReLU layers:
`FiniteArtifact.accepts`. It reads a graph of registered operators, the
binary32 weights and biases among its payload constants, an input box, the affine entries, and the
requested output inequalities. The graph must lay out such a chain in a fixed order. The input node
comes first. Each linear layer is its weight constant, its bias constant, and then `Op.linear` on
those two and the output of the previous layer (the node before the two constants). Each ReLU is
`Op.relu` on the previous node, and the last node is the graph's only output. Each finite word has
an exact rational value. We can therefore check whether the supplied coefficients enclose a real
computation without first replacing them with a different certificate.

Here is the rounding issue that this check catches. Put two scalar linear layers in sequence,
each with weight $`a=1+2^{-23}` and zero bias. Their exact composed weight is

$$`a^2=1+2^{-22}+2^{-46}.`

Binary32 rounds that product to $`1+2^{-22}`. An affine upper bound using the rounded coefficient
can agree perfectly with binary32 replay while falling below the real output at a positive input.
On this two-layer graph, the decision rejects the output entry with the rounded coefficient and
accepts the entry rounded outward.

The extra check follows the graph from its input. At each layer it computes an exact rational
transfer from the already checked parent bounds: the supplied lower affine form must lie below that
transfer throughout the input box, and the supplied upper form must lie above it. Then the decision
checks every requested output inequality using those supplied bounds. Strict and non-strict
inequalities are separate choices, so a zero margin can't satisfy a strict request.

`FiniteArtifact.accepts_graph_sound` connects this decision to the graph's evaluator `Graph.eval`,
using the exact real values of its decoded binary32 constants: for every real input inside the
decoded box, the graph evaluates successfully and all requested inequalities hold. And the
statement includes the actual artifact entry at the designated output. The decision reads the
affine entries at the input, at every linear operator and at every ReLU, which are mandatory, and
the slope vector at every ReLU, which is optional (without one it takes slope `1` where the upper
bound exceeds minus the lower bound and `0` elsewhere). A missing affine entry there, a nonfinite
value among the words it reads, empty input or output dimensions, an empty collection of
inequalities, and an output id that doesn't cover the whole chain are rejected. Entries at the
constant nodes and the interval and phase fields aren't read.

This decision covers nonempty vector linear/ReLU chains in that layout, and the node-certificate
verdict `CROWNNode.check` decides with it (`CROWNNode.verdict_ok`). The replay diagnostic above
keeps only its local-consistency conclusion. And the real computation in the stronger theorem
differs from a native floating-point run, so relating a LibTorch result to that real value still
requires an execution-error bound.

# Directed Arithmetic In The Executable Pass

The bound passes don't assume that every scalar type can safely evaluate every interval rule.
`BoundOps` supplies executable lower and upper addition, subtraction, and multiplication, and
`LawfulBoundOps` is the corresponding proof interface: it interprets each endpoint as a real number
and proves that the lower operation is below exact arithmetic and the upper operation is above it.
Sound arithmetic lemmas require this second interface. `NonlinearBoundOps` supplies executable
interval transfers for operations such as division, square root, exponential, logarithm, and layer
normalization. A successful computation isn't itself a theorem: the separate
`LawfulNonlinearBoundOps` class proves that every returned interval encloses the corresponding real
operation, and sound entrypoints must require this class. A transfer returns `none` when the scalar
backend has no finite implementation for that operation.

These classes specify directed endpoint operations and separate scalar soundness laws, but they
don't implement the full IEEE interval-arithmetic standard.

A point interval already shows why the distinction matters. Both entries of $`x=(1,2^{-25})` are
exactly representable in binary32, but their real sum $`1+2^{-25}` rounds to $`1` under nearest
rounding. Using that same rounded value as both endpoints would return $`[1,1]` and exclude the
sum, which is why the engine's reductions accumulate a lower sum downward and an upper sum upward.
A mean also encloses the coordinate count before division, since a large count need not be exactly
representable in the chosen format.

Average pooling uses the same directed mean for each window. Its count includes padded cells only
when `countIncludePad` is set. Otherwise it counts the window cells inside the input.
Input regions round the center-minus-radius downward and center-plus-radius upward. Eval-mode
BatchNorm encloses the stabilized variance, square root, division, scale, and bias separately. Its
running statistics are fixed, but computing a normalization coefficient still involves rounding.
These executable transfers live in {src "TorchLean/Verify/CROWN/Op/Exec.lean"}[the interval pass],
and `Op.Directed.encloses_of_run` covers the pooling and normalization rules among them.

`ℝ` and `FP32` have lawful nonlinear instances, and the `FP32` proofs use its exact-real floor and
ceiling rounding theorems. For FloatLib binary32, the
{src "TorchLean/Verify/CROWN/BoundOps/Binary32.lean"}[executable interval adapter] instead checks
finite, ordered endpoints and uses certified rational interval kernels followed by outward binary32
rounding, and its eight `Binary32.*Bounds_sound` theorems cover division,
exponential, logarithm, square root, sigmoid, tanh, sine, and cosine. Each theorem starts from a
successful transfer and exact finite endpoint decoding, then encloses every real member of the
input interval (including irrational inputs and trigonometric extrema inside the interval).

Exponential and logarithm return finite enclosures when their checks succeed. Invalid domains,
exceptional endpoints, reversed intervals, and overflow can return `none`. Square root clamps the
negative part of an interval crossing zero (matching `Real.sqrt`), and sigmoid and tanh retain
their global codomain fallbacks. `layerNormAbsBound` returns `none`. These finite containment
theorems don't supply a global `LawfulBoundOps` instance for the IEEE carrier (which includes
infinities and NaNs) or an error bound for a native transcendental implementation. Host `Float`
widens basic binary64 arithmetic by one adjacent representable value, but it likewise has no global
`LawfulBoundOps` instance and makes no enclosure claim for the host transcendental library.

CROWN's output query `Op.outputBoxCROWN?` runs the backward pass at every scalar, rounding
coefficients and constants outward. Where a node has an IBP box but no coefficient rule, that box
bounds the active objective without sending coefficients to the node's parents. ReLU, LayerNorm,
softmax, and MSE loss take this fallback, so the backward pass treats every ReLU of an MLP by its
interval box. Binary products use the McCormick planes described above. A few operators (such as
gather, padding, and custom operators) end the sweep, and the query then keeps the output's
interval box. So finishing a CROWN pass doesn't guarantee a tighter result than IBP. The
{src "TorchLean/Verify/CROWN/Op/Backward.lean"}[backward theorems] cover coefficient propagation,
the box fallback, and final interval evaluation at a supplied real point of the graph, while
`Op.outputBoxCROWN?_of_run` and `Op.backwardObjectiveBox?_of_run` take that point
and the node equations from a successful real run. A returned result is sound under these premises,
but unsupported transfers can still prevent success. The α-CROWN relaxation of an unstable ReLU
belongs to the forward pass, which `Op.outputBoxAlphaCROWN?` runs only at scalars with exact
affine reassociation. Relating these bounds to a separate native execution also requires a
runtime-approximation theorem.

# Robustness Margins From Output Bounds

Suppose a sound bound procedure produces, for every `x` in the input box,

$$`f_y(x)\ge L_y,\qquad f_j(x)\le U_j.`

Then $`L_y-U_j>0` proves the pairwise class margin. A multiclass certificate repeats this for every
$`j\ne y`. The arithmetic is elementary, but the substantive obligations are that:

- the bounds enclose the exact graph semantics
- the graph denotes the intended model
- the input box denotes the intended perturbation set
- any rounded or native execution is related to the exact graph.

For a rounded implementation with coordinate errors
$`\lvert f_i^{\mathrm{run}}(x)-f_i(x)\rvert\le\varepsilon_i`, the transferred margin is

$$`f_y^{run}(x)-f_j^{run}(x)
\ge L_y-U_j-\varepsilon_y-\varepsilon_j.`

The right-hand side must remain positive. A real CROWN theorem alone doesn't prove the native
binary32 claim (the FP32 and runtime-approximation sections describe the additional bridge).

Both error terms are subtracted because rounding can move the two scores in opposite unfavorable
directions (down for the chosen class and up for its competitor), which also explains why a very
small positive real margin may be insufficient for a rounded implementation. If the transferred
lower bound is zero or negative, the argument has failed to establish strict dominance, but it
hasn't produced an input where the classifier changes its label. Finding such an input is a
different result from failing to prove that none exists.

# Executable Certificates And Imported Artifacts

TorchLean checks several kinds of artifact, each with a deliberately limited meaning.

A graph numerical certificate records source ranges, derived node ranges and a backend-plan audit.
`generateChecked` reconstructs this data, and `executeIEEE32` performs a bit-level reference replay
while checking each intermediate tensor. A range transfer (`RangeTransfer`) is an executable
function with no semantic soundness field, so a proof-level error trace also requires a separately
constructed `RealEnclosure` (whose fields supply the real denotation and enclosure proof).
See the runtime-approximation section for the complete boundary.

`AbCROWNLeaf.check`, the external alpha-beta-CROWN leaf checker, validates the JSON schema,
finite ordered root and leaf boxes, dimensions, containment of each represented leaf in the root,
and the exported lower-bound witness relative to its threshold. It also checks root coverage using
the closed grid cut by leaf and root endpoints (returning an error above one million cells). But
this executable check has no accompanying soundness theorem, and the lower bounds' relationship to
the network semantics remains a producer obligation.

This yields three distinct uses of "certificate":

- a Lean term containing proof fields
- an artifact accepted by a structural or numerical checker
- an externally produced claim whose semantic validity is assumed.

Proof-carrying code in the sense of {Informal.citet necula1997}[] requires a sound checker for the
property being claimed. Structural validation alone can be useful without establishing that
stronger property. What evidence is available depends on the command's acceptance predicate, not on
the artifact's name.

In the certificate chapter we run the leaf checker, change a witness so that it must fail, and
explain which stronger artifact would be needed to obtain root-region soundness.

## Other Maintained Checkers

Several implemented families in the verifier registry exercise different semantic objects:

- `ibp-cert` replays the interval bounds of a 3 → 4 → 2 ReLU MLP from a JSON certificate and
  accepts it only when the serialized bounds enclose the bounds Lean recomputes.
- `camera-box3d-cert` checks a camera/3D-box artifact. Its
  {src "TorchLeanExamples/Verification/Geometry3D/Box3D.lean"}[`Box3D`] implementation includes
  interval-operation soundness lemmas and `checkCert_sound`, connecting a successful pure check to
  the stated projection, positive-depth, and image/bounding-box guards.
- `vnncomp-mnistfc` parses the supported VNN-COMP-style MNIST-FC suite and runs the in-repo bound
  workflow. It supports that declared suite, not every VNN-LIB/ONNX benchmark.
- `torchlean-mlp-workflow` trains a small classifier and then checks its robustness with CROWN.
- `twostage-torchlean-cegis-van` runs the all-in-Lean Lyapunov workflow of the two-stage chapter,
  from initialization through the final bound check.

`scripts/lake.sh exe verify -- list` is the authoritative command inventory. A successful run
establishes the acceptance predicate or reported computation documented for that command, but it
doesn't merge these heterogeneous formats into one global verification theorem.

# Floating-Point Models And Refinement

The proof float `FP32` is `NF binaryRadix (Model.fexpOf FloatFormat.binary32) nearestEven`: a
rounded-real model with gradual
underflow. Its exponent description has no upper bound, so it does not model overflow, NaN,
infinity, or signed-zero payload behavior. FloatLib binary32 is the executable bit-level model.

FloatLib's `ExecFloat.Binary.toReal_add_eq_roundAt` connects executable addition to the
proof-level rounding model under the explicit hypothesis that the result is finite. The
multiplication theorem `ExecFloat.Binary.toReal_mul_eq_roundAt` and the division and square-root
bridges of {src "TorchLean/Verify/Floats/IEEE754.lean"}[the finite bridge] carry
their own finite-path hypotheses. Layer
and MLP theorems then propagate explicit error budgets. Neither layer is an unstated theorem about
native hardware or a vendor reduction schedule.

The rounded-pass theorems likewise need `LawfulBoundOps` and `LawfulNonlinearBoundOps`, which the
FloatLib IEEE carrier doesn't supply globally. So choosing an IEEE scalar type doesn't discharge
the floating refinement obligations.

# Verification Evidence And Assumptions

A verification report should make the following boundary visible:

:::table +header
*
  * Evidence
  * Established in current source
  * Not established by that evidence
*
  * `IR.lowerForward_eq_ok`
  * the lowered graph and the recorded typed graph agree
  * agreement with the source program or a native runtime
*
  * `Op.certLocalOK_encloses`
  * every box of an accepted real certificate encloses its node's value in a successful real run
    (`Op.semLocalOK_iff_run`)
  * that the certificate has a box at every node, rounded scalars, or the JSON checkers
*
  * `Op.Directed.encloses_of_run`
  * every box `Op.runIBP` returns encloses its node's value in a successful real run, under the
    scalar laws, the real constants, and enclosed inputs
  * that every node returns a box, or that a native execution is that real run
*
  * `Op.backwardObjectiveBox?_of_run`
  * a successful objective query encloses its real value through backward propagation and final
    interval evaluation, under the scalar, run, and input hypotheses
  * total success, improved tightness over IBP, or native runtime equivalence
*
  * `Op.Alpha.encloses_of_run`
  * real α-CROWN enclosure at the input point for a replayed table with enclosing intervals
  * that a produced certificate replays the step and satisfies `AlphaOK`
*
  * CROWN node checker accepts
  * artifact intervals contain the authoritative FloatLib binary32 IBP trace, and affine entries
    exactly match a sequential FloatLib binary32 replay
  * exact-real CROWN soundness
*
  * `abcrown-leaf` checker succeeds
  * represented boxes and witness fields pass structural/numeric checks
  * network-bound provenance, or soundness of the executable coverage check
*
  * numerical range check plus IEEE replay
  * stored trace and one reference execution pass executable checks
  * real semantic enclosure without `RealEnclosure`
*
  * FP32 approximation theorem
  * rounded-real output is within its stated budget
  * native IEEE behavior without finite refinement
:::

For a particular report, select the rows that refer to its graph, scalar type, and checker. The
last column identifies the additional evidence needed before composing them into a deployment
claim.
