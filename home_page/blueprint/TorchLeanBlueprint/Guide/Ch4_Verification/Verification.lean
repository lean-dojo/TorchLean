import VersoManual
import NN.Verification.Builtin.Proved
import NN.Runtime.Autograd.IRExec.Correctness.SemanticEquivalence
import NN.MLTheory.CROWN.Proofs.GraphCertSoundness
import NN.MLTheory.CROWN.Proofs.GraphRunibpEndToEnd
import NN.MLTheory.CROWN.Proofs.GraphCrownCertSoundness
import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness
import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.EndToEnd
import NN.Verification.Cert.CROWNNodeCert
import NN.Verification.Cert.CROWNNodeCertAlphaBeta
import NN.IR.ShapeSoundness
-- The worked two-layer example below writes its input corners as tensor literals.
import NN.Tensor.Internal.Elab.TensorLiteral
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean

-- The theorems quoted in this chapter live in deep namespaces, and Verso keeps displayed
-- code narrow enough to read in a sidebar. Opening the namespaces here lets every `#check`
-- below fit on one line without losing any information: a printed signature always spells
-- out the fully qualified name of everything it mentions.
open NN.Verification.Builtin.Proved.Correctness
open Runtime.Autograd.IRExec
open NN.MLTheory.CROWN.Graph.CertSoundness
open NN.MLTheory.CROWN.Graph.CrownCertSoundness
open NN.MLTheory.CROWN.Graph.AlphaCrownTransferSoundness
open NN.Verification.CROWNNodeCert
open NN.Verification.CROWNNodeCertAlphaBeta

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

Consider a classifier with two inputs and two possible labels. Give it the point `[1.5, 0.0]`,
and it returns one score for class zero and another for class one. The larger score determines the
prediction. We can evaluate the model at this point and see which class wins. We can also move the
input a little and evaluate it again. Both computations tell us what happened at the points we
chose; neither tells us what happens at every nearby point.

That distinction matters when the input is a measurement. Suppose each coordinate may differ by
up to `0.10` from its recorded value. The possible inputs form a square around `[1.5, 0.0]`.
If the classifier chooses the same label throughout that square, this particular uncertainty
cannot change its decision. If there is a decision boundary inside the square, an input change
within our stated tolerance may cross it. Small input changes can alter classifier predictions,
as {Informal.citet szegedy2014}[] demonstrated; evaluating a finite collection of perturbations
leaves open what happens between them.

Neural network verification asks whether a specified property holds for all inputs in a specified
set. Local classification robustness is one such property. Others concern output ranges,
monotonicity, or an inequality used by a controller. The model, the input set, and the property
all belong to the statement: verifying one square around one point does not establish the same
claim everywhere, and preserving a prediction does not show that the prediction is the correct
label in the first place.

I'll start with a classifier because its property is easy to read from the outputs. There are
only two scores to compare, and the same comparison extends directly to more classes. Let `f`
return the classifier's logits, let `X` be the input region, and let `y` and `j` be the intended
and competing classes. A logit is a score before a probability normalization; it may be negative.
To show that class `y` stays ahead of class `j` throughout the region, we seek

$$`\forall x\in X,\qquad f_y(x)-f_j(x)>0.`

The subtraction measures the pairwise margin. A positive lower bound on that margin would settle
the claim, provided it refers to the intended model, input region, and arithmetic. Each of those
choices needs an explicit connection to the computation: lowering must preserve the model, the
bound procedure must enclose the graph's values, and accepted certificate data must satisfy the
bound theorem's premises. Rounded execution adds an approximation argument.

For the two-class model, this is one inequality. With ten classes, it is nine inequalities,
one against each competitor. Strict positivity also removes a possible tie: the conclusion does
not depend on which class an implementation selects when two scores are equal. The rest of the
work is finding a lower bound that applies to the whole input region and justifying how it was
computed. The training run below gives us a concrete model and margin to follow.

# Classifier Training And Verification

The maintained workflow trains a `2 -> 8 -> 2` ReLU classifier: two input coordinates, eight
hidden units, and two output scores. Its eight training examples separate by the sign of the
first coordinate. Positive first coordinates have class-zero labels; negative first coordinates
have class-one labels. Training fits that distinction with one-hot cross-entropy and SGD.
Verification then fixes the resulting weights and asks about the model that was actually trained:

```
-- Keep the parameters produced by these 120 training steps.
let trained ← trainer.train dataset { steps := 120 }
-- Ask whether class zero stays strictly ahead throughout
-- the input box.
let report ← trained.verify center
  (radius := 0.10)
  (norm := .inf)
  (property := .topLabel 0)
report.printSummary
```

Tensor shapes use the public notation: `[]` for a scalar tensor, `[n]` for a vector, and `[m, n]`
for a matrix. The call above does not expose recursive shape constructors, graph nodes, flat boxes,
or parameter stores.

Run it with either maintained CPU arithmetic mode:

```terminal
# Run the classifier workflow with its default CPU scalar
# backend.
lake exe verify -- torchlean-mlp-workflow
# Repeat using the executable IEEE binary32 scalar model.
lake exe verify -- torchlean-mlp-workflow --arithmetic ieee
```

A seeded run on both backends prints the same result:

```terminal +output
mean_loss(before) = 0.978569
mean_loss(after) = 0.010960
prediction at center=[1.885023, -3.156008]
Alpha-Beta-CROWN lower=[1.639539, -3.572599] upper=[2.130507, -2.739417] label=0
  margin=4.378956 certified=true
```

The report lower-bounds class zero by `1.639539` and upper-bounds its only competitor by
`-2.739417`. Their difference is the positive reported margin `4.378956`, so the chosen bound pass
reports `certified=true` for the requested input box. The semantic and arithmetic bridges below
are required before interpreting that report as a theorem.

Read the loss lines as a description of fitting the training examples. Their decrease says
nothing by itself about the entire perturbation region. The `prediction at center` line is a
single forward evaluation: `1.885023` beats `-3.156008`, so the center receives class zero.
The `lower` and `upper` arrays answer a different question. Their entries describe bounds on
each score as the input varies. To protect class zero, the unfavorable comparison uses its
lowest allowed score and class one's highest allowed score. Comparing the two center scores
would miss that variation.

Here `.inf` means that each input coordinate may move independently by at most the radius.
In real coordinates the intended region is `[1.4, 1.6] × [-0.1, 0.1]`. This geometric meaning
is part of the verification request; the representation and rounding of its endpoints still
belong to the arithmetic obligations below.

The reported margin subtracts the competitor's upper endpoint from class zero's lower endpoint.
Using the displayed values gives:

```lean (name := marginEval)
-- Use the unfavorable endpoint for each of the two classes.
def certifiedLower : Float := 1.639539

def competitorUpper : Float := -2.739417

-- Reconstruct the margin from the six-decimal values in the
-- report.
#eval certifiedLower - competitorUpper
```

```leanOutput marginEval
4.378956
```

The flag tests the computed margin. The six-decimal endpoints above have also been rounded for
display, so subtracting their printed values checks the displayed arithmetic, not the exact
internal margin or a theorem about every point in the box.

A proof of the reported property would connect the source model to the lowered graph, show that
the propagated boxes contain its outputs throughout the input region, and use the positive margin
to compare the two logits. For native binary32 execution, the same argument also needs a bound on
how far its outputs can move from those semantic values.

The obligations also localize maintenance. A lowering change affects the source-to-graph argument;
a new CROWN activation affects bound propagation; and a CUDA deployment claim requires the native
arithmetic link in addition to the real-valued theorem.

# Semantic Target And Graph Boundary

The verifier operates on the canonical `NN.IR.Graph`. An interval or affine form is meaningful only
relative to a denotation of that same graph, parameter store, and input box. A lowering theorem is
therefore part of a source-model claim.

TorchLean has two relevant forward correspondences. The typed first-order
{src "NN/Verification/Builtin/Proved.lean"}[proved forward fragment]
lowers `NN.Verification.Builtin.Proved.ForwardProgram` values. Its constructors cover constants,
parameters, arithmetic, ReLU, `exp`, `log`, inverse, matrix products, reshapes and permutations,
softmax along any valid axis, axis-parametrized LayerNorm, linear and convolution layers, and MSE
loss.
`Correctness.lowerForwardProgramToIR_wellFormed` proves structural well-formedness, while
`Correctness.runForwardIR_eq_evalForward` proves equality with the typed program evaluator.

The second correspondence starts from canonical IR rather than the typed source language.
`denoteAll_eq_of_lowerToForwardGraph` proves that a successful lowering to the forward-only
`IRExec.ForwardGraph` preserves denotation for every input, with `NoRawLog` as its only side
condition. A third, smaller fact sits underneath both: `NN.IR.Graph.checkShapes_sound` proves that
the shapes declared in an accepted graph are the shapes its denotation computes.

Both are Lean semantic equalities over an abstract scalar `Context`. They are not statements that a
PyTorch module, CUDA kernel, or vendor library agrees with the graph. General API lowering also
does not inherit the typed-fragment theorem merely because it returns the same IR type.

The following signatures are checked against the imported `NN` modules when the page builds.
They expose the scalar type, accepted fragment, and success conditions needed to apply each result.

```lean (name := loweringThms)
-- The typed source program always lowers to a structurally
-- valid graph.
#check @lowerForwardProgramToIR_wellFormed
-- Its graph evaluation agrees with evaluation of the
-- original typed program.
#check @runForwardIR_eq_evalForward
-- A successful IR-to-forward-graph lowering preserves
-- denotation under NoRawLog.
#check @denoteAll_eq_of_lowerToForwardGraph
-- Successful shape checking agrees with the shapes of
-- successfully computed values.
#check @NN.IR.Graph.checkShapes_sound
```
```leanOutput loweringThms (whitespace := lax)
@lowerForwardProgramToIR_wellFormed : ∀ {α : Type} [inst : TorchLean.Storage α] [inst_1 :
  Context α]
  {paramShapes : List Spec.Shape} {inShape outShape : Spec.Shape}
  (p : NN.Verification.Builtin.Proved.ForwardProgram α paramShapes inShape outShape)
  (params : TorchLean.TensorPack α paramShapes),
  (NN.Verification.Builtin.Proved.lowerForwardProgramToIR p params).graph.wellFormed = true
```
```leanOutput loweringThms (whitespace := lax)
@runForwardIR_eq_evalForward : ∀ {α : Type} [inst : TorchLean.Storage α] [inst_1 : Context α]
  {paramShapes : List Spec.Shape} {inShape outShape : Spec.Shape}
  (p : NN.Verification.Builtin.Proved.ForwardProgram α paramShapes inShape outShape)
  (params : TorchLean.TensorPack α paramShapes) (x : TorchLean.Tensor α inShape),
  NN.Verification.Builtin.runForwardIR (NN.Verification.Builtin.Proved.lowerForwardProgramToIR
    p params) x =
    NN.Verification.Builtin.Proved.evalForward p params x
```
```leanOutput loweringThms (whitespace := lax)
@denoteAll_eq_of_lowerToForwardGraph : ∀ {α : Type} [inst : TorchLean.Storage α] [inst_1 :
  Context α] (g : NN.IR.Graph)
  (payload : NN.IR.Payload α) (exec : ForwardGraph α),
  NoRawLog g →
    lowerToForwardGraph g payload = Except.ok exec →
      ∀ (x : TorchLean.Tensor α exec.inShape),
        g.denoteAll payload { shape := exec.inShape, tensor := x } = Except.ok (exec.denoteAll x)
```
```leanOutput loweringThms (whitespace := lax)
@NN.IR.Graph.checkShapes_sound : ∀ {α : Type} [inst : TorchLean.Storage α] [inst_1 : Context α]
  (g : NN.IR.Graph)
  (payload : NN.IR.Payload α) (input : Spec.SomeTensor α) (vals : Array (Spec.SomeTensor α)),
  g.checkShapes = Except.ok () →
    g.denoteAll payload input = Except.ok vals →
      ∀ (i : ℕ) (hi : i < g.nodes.size) (hiv : i < vals.size), vals[i].shape = g.nodes[i].outShape
```

Each `#check` prints a declaration's type. It does not execute the model or apply the theorem to
this classifier. In the first output, read everything before the final expression as the data
the theorem accepts; the expression ending in `wellFormed = true` is what it establishes. The
second output ends in an equality between two evaluators, both given the same parameters and
input. That equality is the link needed when a bound on one representation is to say something
about the other.

In these signatures, `∀` means “for every,” `{...}` contains arguments Lean can often infer, and
`[...]` requests an interface instance, such as the scalar operations in `Context α`. An arrow
`P → Q` says that a proof of `P` yields a proof of `Q`. Long namespaces tell Lean which definition
is meant; the important reading order is the objects, then the hypotheses, then the conclusion.
For example, the third output needs both `NoRawLog g` and successful lowering before it gives
equality for every input `x`.

The typed-program results need no additional success premise: the program constructors, shapes,
and scalar interfaces already restrict the objects they accept. The IR-to-forward-graph result
instead assumes that lowering returns `Except.ok exec`. It also requires `NoRawLog g`, excluding
all raw-log nodes, even those whose inputs would be positive. The canonical evaluator checks the
logarithm's input domain, while the pure forward graph applies the total scalar operation. Public
IO autograd performs its own domain validation; that does not change this pure theorem's premise.

Shape soundness has a different role. Once shape checking and denotation both succeed, it identifies
the shape of each computed value with the declaration at the corresponding node. It supplies no
existence result for the value table: successful denotation is an assumption.

# IBP

Interval bound propagation assigns each node a box. For an affine layer

$$`y=Wx+b,\qquad x\in[\ell,u],`

the usual sign split gives

$$`\ell_y=W^+\ell+W^-u+b,\qquad
u_y=W^+u+W^-\ell+b,`

where $`W^+=\max(W,0)` and $`W^-=\min(W,0)`. Monotone activations transform endpoints; ReLU maps
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
  -- Positive weights use the lower input endpoint; negative
  -- weights use the upper.
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

The graph proof applies local enclosure arguments like these at each node in topological order.
The induction maintains enclosure for every earlier value, so a node can use the bounds on its
parents. Tensor dimensions, parameter lookup, and missing values must also agree between the bound
pass and the semantic evaluator. Directed floating arithmetic adds the obligation that rounded
endpoints remain on the enclosing side of the exact result.

The generic real soundness theorem is `cert_encloses_semantics`. It requires:

- `TopoSorted g`;
- `Supported g`;
- exact local certificate consistency `CertLocalOK`;
- exact local value consistency `SemLocalOK`;
- `InputsEnclosed`.

It concludes that each available certificate box encloses the matching semantic value. The current
`Supported` predicate contains input, constant, detach, addition, subtraction, elementwise
multiplication, ReLU, linear, matrix multiplication, tanh, sigmoid, softplus, safe logarithm,
sine, and cosine nodes.

The proof-side real evaluator has the stronger end-to-end theorem
`runIBP?_encloses_evalGraphRec`. It proves that the particular real `runIBP?` construction encloses
`evalGraphRec`, under topological order, supported operations, and enclosed inputs.

The executable engine is now covered as well, on a named core. `runIBP_eq_runIBP?` proves that the
engine's `runIBP` computes exactly the proof-side pass on graphs whose nodes are all in
`EngineCore` (input, constant, detach, addition, subtraction, elementwise multiplication, ReLU,
linear, matrix multiplication, softplus, and safe logarithm), provided the semantic-support guard
succeeds and the proof-side pass produced a box at every node (`IBPCovers`).
`runIBP_encloses_evalGraphRec` then gives enclosure
for the executable engine directly. Both are stated over `ℝ`; the transcendental node kinds outside
`EngineCore` and every rounded scalar remain outside these two theorems.

```lean (name := ibpThms)
-- Locally consistent real boxes enclose the corresponding
-- semantic values.
#check @cert_encloses_semantics
-- Instantiate those boxes with the proof-side interval
-- propagation pass.
#check @runIBP?_encloses_evalGraphRec
-- On the supported engine core, the executable real pass
-- equals that construction.
#check @runIBP_eq_runIBP?
-- Transfer enclosure to the executable real pass under its
-- coverage hypotheses.
#check @runIBP_encloses_evalGraphRec
```
```leanOutput ibpThms (whitespace := lax)
cert_encloses_semantics : ∀ (g : NN.MLTheory.CROWN.Graph) (ps :
  NN.MLTheory.CROWN.Graph.ParamStore ℝ)
  (cert : Array (Option (NN.MLTheory.CROWN.FlatBox ℝ))) (inputs : Std.HashMap ℕ Val) (vals :
    Array (Option Val)),
  TopoSorted g →
    Supported g →
      CertLocalOK g ps cert →
        SemLocalOK g ps inputs vals →
          InputsEnclosed g ps inputs →
            ∀ id < g.nodes.size,
              match cert[id]!, vals[id]! with
              | some B, some v => EnclosesBox B v
              | x, x_1 => True
```
```leanOutput ibpThms (whitespace := lax)
runIBP?_encloses_evalGraphRec : ∀ (g : NN.MLTheory.CROWN.Graph) (ps :
  NN.MLTheory.CROWN.Graph.ParamStore ℝ)
  (inputs : Std.HashMap ℕ Val),
  TopoSorted g →
    Supported g →
      InputsEnclosed g ps inputs →
        ∀ id < g.nodes.size,
          match (runIBP? g ps)[id]!, (evalGraphRec g ps inputs)[id]! with
          | some B, some v => EnclosesBox B v
          | x, x_1 => True
```
```leanOutput ibpThms (whitespace := lax)
runIBP_eq_runIBP? : ∀ (g : NN.MLTheory.CROWN.Graph) (ps : NN.MLTheory.CROWN.Graph.ParamStore ℝ),
  TopoSorted g →
    EngineCore g → g.crownGraphSemanticsSupported ps = true → IBPCovers g (runIBP? g ps) →
      g.runIBP ps = runIBP? g ps
```
```leanOutput ibpThms (whitespace := lax)
runIBP_encloses_evalGraphRec : ∀ (g : NN.MLTheory.CROWN.Graph) (ps :
  NN.MLTheory.CROWN.Graph.ParamStore ℝ)
  (inputs : Std.HashMap ℕ Val),
  TopoSorted g →
    EngineCore g →
      IBPCovers g (runIBP? g ps) →
        InputsEnclosed g ps inputs →
          ∀ id < g.nodes.size,
            ∀ (B : NN.MLTheory.CROWN.FlatBox ℝ) (v : Val),
              (g.runIBP ps)[id]! = some B → (evalGraphRec g ps inputs)[id]! = some
                v → EnclosesBox B v
```

The first two conclusions examine a pair of optional entries. Enclosure is asserted only when
both a box and a semantic value are present; the other cases return `True`. Thus these theorems can
justify an available bound without promising that the pass produces every bound an application
needs. `IBPCovers` supplies the separate coverage condition in the executable-engine results.

The names describe the pieces of the graph induction. `TopoSorted` puts a node's parents before
that node. `Supported` restricts the operations to those with a proved transfer rule.
`CertLocalOK` says that the stored boxes follow those rules, and `SemLocalOK` says that the stored
values follow the graph's evaluator. `InputsEnclosed` starts the induction by placing the actual
input values inside the input boxes. For the classifier, the conclusion we ultimately need is
enclosure at the output node; these hypotheses explain how the proof can reach that node.

`runIBP_eq_runIBP?` identifies the executable real pass with the proof-side construction under its
listed guards. Substituting that equality into the proof-side enclosure result gives a theorem
about the engine. No second arithmetic argument is needed for that substitution.

## IBP On A Two-Layer Affine Network

The theorems establish enclosure, but a box can be sound and too wide to settle the property.
We can measure that excess width by running the pass on a graph whose exact range we can
calculate by hand.

Consider the map $`x\mapsto (x_0+x_1)-(x_0-x_1)`, written as two affine layers. Composed, it is
exactly $`2x_1`, so on the unit box $`[-1,1]^2` the exact output range is $`[-2,2]`. The bound
engine works on IR graphs rather than on layer combinators, so the graph is spelled out node by
node, and the weights live beside it in a `ParamStore`:

```lean
/-- Two affine layers computing `(x₀+x₁) - (x₀-x₁)`.
Node 1 produces both intermediate coordinates; node 2
subtracts them. -/
def depGraph : NN.IR.Graph :=
  { nodes :=
      #[ { id := 0, parents := #[]
         , kind := .input, outShape := [2] }
       , { id := 1, parents := #[0]
         , kind := .linear, outShape := [2] }
       , { id := 2, parents := #[1]
         , kind := .linear, outShape := [1] } ] }

/-- The input region: the unit box in the sup norm. -/
def depBox : NN.MLTheory.CROWN.FlatBox Float :=
  { dim := 2, lo := [-1.0, -1.0], hi := [1.0, 1.0] }

/-- Weights and biases for the two layers, keyed by the
node they belong to. -/
def depStore :
    NN.MLTheory.CROWN.Graph.ParamStore Float :=
  { inputBoxes :=
      (Std.HashMap.emptyWithCapacity).insert 0 depBox
    linearWB :=
      (Std.HashMap.emptyWithCapacity)
        |>.insert 1
            { m := 2, n := 2
            , w := [[1.0, 1.0], [1.0, -1.0]]
            , b := [0.0, 0.0] }
        |>.insert 2
            { m := 1, n := 2, w := [[1.0, -1.0]]
            , b := [0.0] } }
```

Now run both bound methods on the same store and print the output box of node 2:

```lean (name := depRun)
-- Both entry points live in `NN.MLTheory.CROWN.Graph`.
-- Opening it for this one command keeps the printed
-- signatures elsewhere in the chapter fully qualified.
open NN.MLTheory.CROWN.Graph in
#eval do
  let boxes := runIBP (α := Float) depGraph depStore
  match boxes[2]! with
  | some b =>
    IO.println
      s!"IBP   lo={Spec.pretty b.lo} hi={Spec.pretty b.hi}"
  | none => IO.println "no IBP box"
  match outputBoxCROWN? (α := Float)
      depGraph depStore depBox 0 2 2 with
  | .ok b =>
    IO.println
      s!"CROWN lo={Spec.pretty b.lo} hi={Spec.pretty b.hi}"
  | .error e => IO.println s!"CROWN failed: {e}"
```
```leanOutput depRun
IBP   lo=[-4.000000] hi=[4.000000]
CROWN lo=[-2.000000] hi=[2.000000]
```

IBP returns twice the exact width. Its box for node 1 is
$`[-2,2]\times[-2,2]`, and each coordinate really does range over $`[-2,2]`. What the box cannot
record is that the two coordinates move together: whenever $`x_0+x_1` is at its maximum, $`x_0-x_1`
is $`0`. The second layer subtracts them, and interval subtraction has to assume the worst case
that the first is at $`2` while the second is at $`-2`, which no input achieves. This is dependency
loss: each interval retains a coordinate's range but loses its relation to the others. Repeating
this loss across layers can make box propagation alone
{Informal.citep gowal2018}[] progressively looser as depth grows.

CROWN {Informal.citep crown2018}[] carries an affine form of each node in terms of the input
instead of a box, so the cancellation happens symbolically before any interval is evaluated:
$`1\cdot(x_0+x_1)-1\cdot(x_0-x_1)` becomes $`2x_1`, and only then is the box substituted. The
result is the exact range. That exactness is a property of this graph, not of the method: the graph
is purely affine, so the affine form is the function. An unstable ReLU can introduce a strict
relaxation; stable ReLUs may remain exact. The amount
of improvement over IBP depends on the chosen affine bounds and graph.

The real affine map attains its extrema at box corners. The following `Float` run checks that the
same node array and corresponding payload produce the expected corner values. It uses
`NN.IR.Graph.denote`, so it also lets us compare graph evaluation with the bound pass:

```lean (name := depCorners)
/-- The same two layers as executable parameters. -/
def depPayload : NN.IR.Payload Float :=
  { linear? := fun id =>
      if id = 1 then
        some { outDim := 2, inDim := 2
             , W := [[1.0, 1.0], [1.0, -1.0]]
             , b := [0.0, 0.0] }
      else if id = 2 then
        some { outDim := 1, inDim := 2
             , W := [[1.0, -1.0]], b := [0.0] }
      else none }

/-- Evaluate `depGraph` at one input and read off the
single output coordinate. -/
def depEval (x : TorchLean.Tensor Float [2]) :
    Except String Float := do
  let out ←
    NN.IR.Graph.denote (α := Float) (g := depGraph)
      (payload := depPayload)
      (input := Spec.SomeTensor.ofTensor x)
      (outputId := 2)
  let vec ←
    NN.IR.Graph.expectShape (α := Float)
      (expected := [1]) out
  pure (TorchLean.Tensor.getScalar vec 0)

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
fact robust. Tighter bounds may cost more computation; input subdivision lets us explore that
tradeoff while keeping the local interval rules fixed.

## Input Subdivision

Another general option is to partition an input box and evaluate the same graph on every piece.
`NN.MLTheory.CROWN.Graph.refinedIBPOutput? g ps inputId outId splitBudget` does this without
changing any layer transfer. It applies to any graph supported by IBP,
including mixed architectures.
For example, splitting $`[-1,1]` at zero tightens the ordinary interval bound on
$`\max(x,0)+\max(-x,0)` from approximately $`[0,2]` to $`[0,1]`.

The budget counts total splits, so at most $`2b+1` IBP calls are made for budget $`b`. Both children
must produce bounds; a failed branch keeps the parent result. Their results are combined by a hull,
then intersected with the parent bound. The shared split boundary preserves all original inputs,
including the cut itself. Adjacent floating-point endpoints with no interior midpoint stay unsplit.

This can reduce dependency loss across layers, but does not promise improvement everywhere. A
linear transfer may already be tight, and an input-independent activation range may remain broad.
The theorem `NN.MLTheory.CROWN.Graph.Refinement.splitAt_covers` proves real input coverage of each
split. Sound output bounds still depend on the underlying transfers; subdivision does not establish
universal soundness for rounded LayerNorm or other operations.

Artifact replay can opt in with
`NN.Verification.IBPCert.check g ps outId path (refinement := some (inputId, splitBudget))`.
The default is unchanged. This is a graph-level option, separate from the high-level trainer API.

## Trainer Verification API

`trainer.train` retains the live trained parameters in its ordinary result. Its one generic
`verify` operation accepts a tensor center and named choices such as `radius`, `norm`, `property`,
and `algorithm`. Nothing about the function name changes when the norm or goal changes.

```
-- This import exposes verification on the ordinary
-- trained-model result.
import NN.API.Verification

-- Verification uses the trained weights retained in this
-- value.
let trained ← trainer.train dataset { steps := 120 }
-- Radius and norm define the input set; topLabel defines
-- the output property.
let report ← trained.verify center
  (radius := 0.10)
  (norm := .inf)
  (property := .topLabel 0)
```

The default method is fixed-relaxation Alpha-Beta-CROWN; `.ibp` and `.crown` are available for
comparison. To inspect graph nodes and intermediate bounds, import
`NN.API.Verification.Lowering`.

## Bound Propagation With `auto_LiRPA`

With PyTorch, we can use `auto_LiRPA` {Informal.citep autolirpa2020}[] to propagate bounds through
the model. The wrapper takes the perturbation region along with the model input:

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
lower bound with another's upper bound. The choice of bound method and arithmetic determines how
those endpoints are computed. In TorchLean, the real-semantic transfer theorems and local
certificate predicates describe the evidence needed to justify them. Applying those theorems to
the high-level Float report still requires proofs about the produced data and rounded arithmetic;
the trust ledger below records those obligations.

# CROWN, Alpha-CROWN, And Alpha-Beta-CROWN

CROWN {Informal.citep crown2018}[] propagates affine lower and upper forms rather than only boxes.
Bounding a network by a convex outer approximation and then optimizing inside it predates
CROWN {Informal.citep wongkolter2018}[]; CROWN obtains bounds with a backward pass.
At an uncertain ReLU with $`\ell<0<u`, the secant upper
relaxation is

$$`\operatorname{ReLU}(z)
\le \frac{u}{u-\ell}(z-\ell),`

while a lower relaxation may use a slope $`\alpha` constrained to a valid range. Alpha-CROWN
optimizes these slopes. Alpha-beta-CROWN {Informal.citep betacrown2021}[] additionally records
branch phases: an active branch uses $`z\ge 0`, and an inactive branch uses $`z\le 0`. Splitting on
a phase can tighten the relaxation, at the cost of additional bound passes.

TorchLean's generic theorem `crown_checker_encloses_semantics` takes an exact
`CrownCertLocalOK` hypothesis and a separate `CrownTransferSound` proof. The local transfer
theorems

- `alphaCrown_transfer_sound`;
- `alphaBetaCrown_transfer_sound`

show that the proposition-level alpha and alpha-beta step functions satisfy
`CrownTransferSound` under their explicit real-semantic and enclosure hypotheses. The alpha-beta
step rejects phase choices inconsistent with the current IBP interval.

The transfer theorems take the IBP boxes as an assumption, `IBPEnclosesVals`. The composed
corollaries discharge it from the IBP soundness theorem. `alphaCrown_cert_encloses_semantics` and
`alphaBetaCrown_cert_encloses_semantics'` conclude enclosure from topological order, supported
operations, locally consistent boxes and values, enclosed inputs, valid slopes, and a certificate
that replays the affine step; `alphaCrown_cert_encloses_evalGraphRec` fixes the boxes to `runIBP?`
and the values to `evalGraphRec`, so only the graph, input, slope, and certificate hypotheses
remain. These are the forms a caller should cite.

The executable `runAlphaBetaCROWN` pass first runs IBP, infers every ReLU phase already forced by
the interval, and rechecks that phase while replaying affine bounds. Unstable phases remain
unsplit and use the default alpha relaxation. This is a useful native fixed-relaxation pass; it is
not the external Alpha-Beta-CROWN optimizer's branch-and-bound search.

```lean (name := crownThms)
-- Combine local certificate consistency with a proof that
-- each transfer encloses values.
#check @crown_checker_encloses_semantics
-- Establish that enclosure property for real alpha-CROWN
-- transfers.
#check @alphaCrown_transfer_sound
-- Include the phase information accepted by the real
-- alpha-beta transfer.
#check @alphaBetaCrown_transfer_sound
-- Compose alpha-CROWN with the real IBP and
-- graph-consistency hypotheses.
#check @alphaCrown_cert_encloses_semantics
-- Fix the interval pass and evaluator, discharging their
-- local-consistency premises.
#check @alphaCrown_cert_encloses_evalGraphRec
-- Obtain the corresponding composed result for alpha-beta
-- certificate data.
#check @alphaBetaCrown_cert_encloses_semantics'
```
```leanOutput crownThms (whitespace := lax)
crown_checker_encloses_semantics : ∀ (g : NN.MLTheory.CROWN.Graph) (ps :
  NN.MLTheory.CROWN.Graph.ParamStore ℝ)
  (step :
    Array (Option (NN.MLTheory.CROWN.Graph.FlatAffineBounds ℝ)) →
      ℕ → Option (NN.MLTheory.CROWN.Graph.FlatAffineBounds ℝ))
  (cert : Array (Option (NN.MLTheory.CROWN.Graph.FlatAffineBounds ℝ))) (inputs : Std.HashMap ℕ
    Val)
  (vals : Array (Option Val)) (ctx : NN.MLTheory.CROWN.Graph.AffineCtx) (x : TorchLean.Tensor
    ℝ [ctx.inputDim]),
  TopoSorted g →
    SemLocalOK g ps inputs vals →
      CrownCertLocalOK g step cert →
        CrownTransferSound g ps inputs vals ctx x step cert →
          ∀ id < g.nodes.size,
            ∀ (b : NN.MLTheory.CROWN.Graph.FlatAffineBounds ℝ) (v : Val),
              cert[id]! = some b → vals[id]! = some v → EnclosesAtInput ctx x b v
```
```leanOutput crownThms (whitespace := lax)
alphaCrown_transfer_sound : ∀ (g : NN.MLTheory.CROWN.Graph) (ps :
  NN.MLTheory.CROWN.Graph.ParamStore ℝ)
  (ibp : Array (Option (NN.MLTheory.CROWN.FlatBox ℝ))) (alpha : Array (Option
    (NN.MLTheory.CROWN.Graph.FlatTensor ℝ)))
  (cert : Array (Option (NN.MLTheory.CROWN.Graph.FlatAffineBounds ℝ))) (inputs : Std.HashMap ℕ
    Val)
  (vals : Array (Option Val)) (ctx : NN.MLTheory.CROWN.Graph.AffineCtx) (x : TorchLean.Tensor
    ℝ [ctx.inputDim]),
  TopoSorted g →
    SemLocalOK g ps inputs vals →
      InputsMatch inputs ctx x →
        IBPEnclosesVals ibp vals →
          AlphaOK alpha → CrownTransferSound g ps inputs vals ctx x (stepAlpha g ps
            ibp alpha ctx) cert
```
```leanOutput crownThms (whitespace := lax)
alphaBetaCrown_transfer_sound : ∀ (g : NN.MLTheory.CROWN.Graph) (ps :
  NN.MLTheory.CROWN.Graph.ParamStore ℝ)
  (ibp : Array (Option (NN.MLTheory.CROWN.FlatBox ℝ))) (alpha : Array (Option
    (NN.MLTheory.CROWN.Graph.FlatTensor ℝ)))
  (beta : Array (Option (Array ℤ))) (cert : Array (Option
    (NN.MLTheory.CROWN.Graph.FlatAffineBounds ℝ)))
  (inputs : Std.HashMap ℕ Val) (vals : Array (Option Val)) (ctx :
    NN.MLTheory.CROWN.Graph.AffineCtx)
  (x : TorchLean.Tensor ℝ [ctx.inputDim]),
  TopoSorted g →
    SemLocalOK g ps inputs vals →
      InputsMatch inputs ctx x →
        IBPEnclosesVals ibp vals →
          AlphaOK alpha → CrownTransferSound g ps inputs vals ctx x (stepAlphaBeta g
            ps ibp alpha beta ctx) cert
```
```leanOutput crownThms (whitespace := lax)
alphaCrown_cert_encloses_semantics : ∀ (g : NN.MLTheory.CROWN.Graph) (ps :
  NN.MLTheory.CROWN.Graph.ParamStore ℝ)
  (ibp : Array (Option (NN.MLTheory.CROWN.FlatBox ℝ))) (alpha : Array (Option
    (NN.MLTheory.CROWN.Graph.FlatTensor ℝ)))
  (cert : Array (Option (NN.MLTheory.CROWN.Graph.FlatAffineBounds ℝ))) (inputs : Std.HashMap ℕ
    Val)
  (vals : Array (Option Val)) (ctx : NN.MLTheory.CROWN.Graph.AffineCtx) (x : TorchLean.Tensor
    ℝ [ctx.inputDim]),
  TopoSorted g →
    Supported g →
      CertLocalOK g ps ibp →
        SemLocalOK g ps inputs vals →
          InputsEnclosed g ps inputs →
            InputsMatch inputs ctx x →
              AlphaOK alpha →
                CrownCertLocalOK g (stepAlpha g ps ibp alpha ctx) cert →
                  ∀ id < g.nodes.size,
                    ∀ (b : NN.MLTheory.CROWN.Graph.FlatAffineBounds ℝ) (v : Val),
                      cert[id]! = some b → vals[id]! = some v → EnclosesAtInput ctx x b v
```
```leanOutput crownThms (whitespace := lax)
alphaCrown_cert_encloses_evalGraphRec : ∀ (g : NN.MLTheory.CROWN.Graph) (ps :
  NN.MLTheory.CROWN.Graph.ParamStore ℝ)
  (alpha : Array (Option (NN.MLTheory.CROWN.Graph.FlatTensor ℝ)))
  (cert : Array (Option (NN.MLTheory.CROWN.Graph.FlatAffineBounds ℝ))) (inputs : Std.HashMap ℕ
    Val)
  (ctx : NN.MLTheory.CROWN.Graph.AffineCtx) (x : TorchLean.Tensor ℝ [ctx.inputDim]),
  TopoSorted g →
    Supported g →
      InputsEnclosed g ps inputs →
        InputsMatch inputs ctx x →
          AlphaOK alpha →
            CrownCertLocalOK g (stepAlpha g ps (runIBP? g ps) alpha ctx) cert →
              ∀ id < g.nodes.size,
                ∀ (b : NN.MLTheory.CROWN.Graph.FlatAffineBounds ℝ) (v : Val),
                  cert[id]! = some b → (evalGraphRec g ps inputs)[id]! = some
                    v → EnclosesAtInput ctx x b v
```
```leanOutput crownThms (whitespace := lax)
alphaBetaCrown_cert_encloses_semantics' : ∀ (g : NN.MLTheory.CROWN.Graph) (ps :
  NN.MLTheory.CROWN.Graph.ParamStore ℝ)
  (ibp : Array (Option (NN.MLTheory.CROWN.FlatBox ℝ))) (alpha : Array (Option
    (NN.MLTheory.CROWN.Graph.FlatTensor ℝ)))
  (beta : Array (Option (Array ℤ))) (cert : Array (Option
    (NN.MLTheory.CROWN.Graph.FlatAffineBounds ℝ)))
  (inputs : Std.HashMap ℕ Val) (vals : Array (Option Val)) (ctx :
    NN.MLTheory.CROWN.Graph.AffineCtx)
  (x : TorchLean.Tensor ℝ [ctx.inputDim]),
  TopoSorted g →
    Supported g →
      CertLocalOK g ps ibp →
        SemLocalOK g ps inputs vals →
          InputsEnclosed g ps inputs →
            InputsMatch inputs ctx x →
              AlphaOK alpha →
                CrownCertLocalOK g (stepAlphaBeta g ps ibp alpha beta ctx) cert →
                  ∀ id < g.nodes.size,
                    ∀ (b : NN.MLTheory.CROWN.Graph.FlatAffineBounds ℝ) (v : Val),
                      cert[id]! = some b → vals[id]! = some v → EnclosesAtInput ctx x b v
```

The two transfer theorems have the same semantic hypotheses. The alpha-beta version adds a branch
vector `beta` and uses `stepAlphaBeta`. It needs no extra domain premise because that step accepts
only phases already justified by the supplied intervals. To assume a phase for an unstable ReLU,
a theorem would also have to restrict the input domain to that branch.

The shared predicate `CrownTransferSound` states the local enclosure obligation. A different
relaxation can use the generic checker theorem once its step function has a proof of that
obligation.

Then compare `alphaCrown_cert_encloses_semantics` with
`alphaCrown_cert_encloses_evalGraphRec`. The second has strictly fewer hypotheses because it stops
being generic: `ibp` is instantiated to `runIBP? g ps` and the values to `evalGraphRec g ps inputs`,
which discharges `CertLocalOK` and `SemLocalOK` outright. That is the version to cite from
application code. The general one is there for a caller who obtained boxes and values some other
way, and paying for that generality means proving two more consistency conditions by hand.

These theorems should not be confused with the JSON node-certificate checkers:

- `checkCROWNNodeCertificate`;
- `checkAlphaBetaCROWNNodeCertificate`.

Those `IO Bool` functions parse finite decimal fields into FloatLib binary32, recompute the
complete IBP
trace from trusted inputs and parameters, and replay affine nodes from previously recomputed affine
data. A serialized interval may be wider than the recomputed interval but may not move inward.
Affine replay data must match exactly at the binary32 level. The alpha-beta checker also validates
branch-vector lengths, entries, and phase consistency.

The shared JSON boundary validates input regions before replay begins. Endpoint boxes require
finite arrays of the declared dimension with `lo[i] <= hi[i]`; center-radius boxes additionally
require a finite nonnegative radius. Incomplete or mixed schemas are rejected. Artifact formats
that prescribe endpoints, including the alpha-beta-CROWN leaf format, request that exact schema
rather than accepting the alternate center-radius notation.

The final decisions are `CROWNNodeCert.certificateAccepts` and
`CROWNNodeCertAlphaBeta.AlphaBetaCROWNNodeCertificate.accepts`. Their soundness theorems prove that
acceptance supplies `CrownCertLocalOK` for the exact FloatLib binary32 replay function used by the
checker.

```lean (name := acceptThms)
-- If the CROWN decision accepts, its stored affine entries
-- satisfy the replay rule.
#check @certificateAccepts_eq_true
-- The alpha-beta decision gives the same kind of result for
-- its phase-aware rule.
#check @AlphaBetaCROWNNodeCertificate.accepts_eq_true
```
```leanOutput acceptThms (whitespace := lax)
certificateAccepts_eq_true : ∀ (cert : NN.Verification.Cert.NodeReplay.CROWNNodeCoreCertificate)
  (g : NN.MLTheory.CROWN.Graph)
  (ps :
    NN.MLTheory.CROWN.Graph.ParamStore
      (ExecFloat.Binary 8 23 FloatFormat.Encoding.ieee (FloatFormat.Encoding.ieee.defaultBias 8)
        checkCROWNNode._proof_1
        checkCROWNNode._proof_2 checkCROWNNode._proof_3 checkCROWNNode._proof_4))
  (authoritativeIbp :
    Array
      (Option
        (NN.MLTheory.CROWN.FlatBox
          (ExecFloat.Binary 8 23 FloatFormat.Encoding.ieee (FloatFormat.Encoding.ieee.defaultBias 8)
            checkCROWNNode._proof_1 checkCROWNNode._proof_2 checkCROWNNode._proof_3
              checkCROWNNode._proof_4))))
  (diagnosticsOk : Bool),
  certificateAccepts cert g ps authoritativeIbp diagnosticsOk = true →
    CrownCertLocalOK g (NN.Verification.CROWNNodeCert.replayStep g ps authoritativeIbp cert)
      cert.crown
```
```leanOutput acceptThms (whitespace := lax)
AlphaBetaCROWNNodeCertificate.accepts_eq_true : ∀ (cert : AlphaBetaCROWNNodeCertificate) (g :
  NN.MLTheory.CROWN.Graph)
  (ps :
    NN.MLTheory.CROWN.Graph.ParamStore
      (ExecFloat.Binary 8 23 FloatFormat.Encoding.ieee (FloatFormat.Encoding.ieee.defaultBias 8)
        AlphaBetaCROWNNodeCertificate._proof_1 AlphaBetaCROWNNodeCertificate._proof_2
        AlphaBetaCROWNNodeCertificate._proof_3 AlphaBetaCROWNNodeCertificate._proof_4))
  (authoritativeIbp :
    Array
      (Option
        (NN.MLTheory.CROWN.FlatBox
          (ExecFloat.Binary 8 23 FloatFormat.Encoding.ieee (FloatFormat.Encoding.ieee.defaultBias 8)
            AlphaBetaCROWNNodeCertificate._proof_1 AlphaBetaCROWNNodeCertificate._proof_2
            AlphaBetaCROWNNodeCertificate._proof_3 AlphaBetaCROWNNodeCertificate._proof_4))))
  (diagnosticsOk : Bool),
  cert.accepts g ps authoritativeIbp diagnosticsOk = true →
    CrownCertLocalOK g (NN.Verification.CROWNNodeCertAlphaBeta.replayStep g ps authoritativeIbp
      cert) cert.crown
```

The first output says: for any certificate, graph, parameter store, interval trace, and diagnostic
flag, *if this decision returns `true`, then the affine table in the certificate agrees with the
specified replay rule*. The arrow before `CrownCertLocalOK` separates the assumption from the
conclusion. There is no classifier margin in this statement yet.

The arguments have concrete jobs:

:::table +header
*
  * Name in the signature
  * Meaning in the check
*
  * `cert`
  * The proposed certificate, including its affine table `cert.crown` and relaxation data.
*
  * `g`
  * The graph whose nodes the table describes.
*
  * `ps`
  * The weights, biases, input boxes, and other parameters used by replay.
*
  * `authoritativeIbp`
  * The interval table supplied to the decision. The IO wrapper constructs it by rerunning IBP.
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
including its rounding and exceptional values. A theorem about this table therefore concerns
those executable values; a real-valued conclusion needs a further refinement argument.

We can unpack `CrownCertLocalOK` precisely. It requires the affine table to have the same length
as the graph's node array. At every valid node index, the table entry must equal the result of
`replayStep` applied to that table and index. If a producer changes an affine coefficient, the
exact comparison can reject it even when the decimal difference looks small. Since entries are
optional, local consistency alone does not assert that every node has a bound.

The second output has the same structure. Its certificate also supplies beta phase information,
so it uses the alpha-beta `replayStep`. The dotted expression `cert.accepts ...` is the decision
applied to that certificate. Both theorems connect an executable Boolean decision to a Lean
proposition, which lets a later proof use the result without reasoning again about every
componentwise comparison.

There are two details to keep when applying either result. The pure theorem accepts the interval
trace as an argument; calling it `authoritativeIbp` does not prove where that trace came from.
The wrapper supplies the recomputation. Likewise, the theorem does not interpret the individual
diagnostics behind `diagnosticsOk`. Its conclusion is the stated local consistency property.
To recover the classifier claim, we still need a soundness argument saying that these replay
rules enclose the network, bounds at the required output, and a positive class margin.

The real theorem `crown_checker_encloses_semantics` also uses a local-consistency predicate, but
over real affine data and a real step function. The binary32 result cannot be supplied directly
as that premise. A real enclosure requires a refinement argument connecting the rounded replay
to the real transfers, as well as the remaining semantic hypotheses.

There are two different universal statements in this chain. Local replay checks every node in a
finite graph. Robustness concerns every input in a region, which contains infinitely many real
points. The transfer theorem is what connects them: each node's rule preserves an enclosure for
an arbitrary input satisfying the assumptions, and graph induction carries that fact to the
output. Checking every node is useful because of that argument. The number of entries checked
does not replace it, and the printed `#check` signatures identify the argument rather than
performing it for the displayed classifier.

# Directed Arithmetic In The Executable Pass

The graph engine does not assume that every scalar type can safely evaluate every interval rule.
`BoundOps` supplies executable lower and upper addition, subtraction, and multiplication.
`LawfulBoundOps` is the corresponding proof interface: it interprets each endpoint as a real
number and proves that the lower operation is below exact arithmetic and the upper operation is
above it. Sound arithmetic lemmas require this second interface. `NonlinearBoundOps` supplies
executable interval transfers for operations such as division, square root, exponential, logarithm,
and layer normalization. A successful computation is not itself a theorem. The separate
`LawfulNonlinearBoundOps` class proves that every returned interval encloses the corresponding real
operation; sound entrypoints must require this class. A transfer returns `none` when the scalar
backend has no finite implementation for that operation.

`ℝ` and `FP32` have lawful nonlinear instances. The `FP32` proofs use its exact-real floor and
ceiling rounding theorems. FloatLib binary32 instead states finite-path soundness in its IEEE
semantics
modules, where overflow, infinities, and NaNs can be handled explicitly. Its directed binary32
division and square root are proved, but executable exponential is not yet proved to enclose real
exponentiation. An IBP node using exponential therefore remains unresolved under that instance. Host
`Float` widens basic binary64 arithmetic by one adjacent representable value, but it has no global
`LawfulBoundOps` instance and makes no enclosure claim for the host transcendental library.
Sigmoid, tanh, sine, and cosine can still use their global codomain bounds when a sharper transfer
is unavailable.

When the IBP pass has a valid nonlinear box but no directed affine relaxation, forward CROWN keeps
the box as a constant affine bound. Objective-dependent backward CROWN carries affine coefficients
through algebraic rewrites. Over the reals those rewrites are exact; over rounded execution they
need a separate runtime-approximation theorem. A printed floating-point backward-CROWN result is
therefore computational evidence until that bridge is supplied.

The first-derivative interval pass is seeded. `runDirectionalDerivative` propagates a point or
interval of input directions, so coordinate vectors give partial-derivative bounds without a
second operator traversal. `runScalarDerivative` is the scalar-input specialization and rejects a
non-scalar input box. Both entrypoints use the same local transfer function; their supported
operators cannot drift apart. The current second-derivative pass remains one-dimensional.

LayerNorm currently leaves its derivative box unresolved. Its real derivative and a rowwise bound
are proved separately, but the interval pass still needs a transfer that uses each row's stored
scale and epsilon. Flattening several rows into one normalization, or enlarging a denominator
before taking its reciprocal, can underestimate the derivative. Certificate consumers report the
missing box; the separate forward evaluation and value bounds remain available.

# Robustness Margins From Output Bounds

Suppose a sound bound procedure produces, for every `x` in the input box,

$$`f_y(x)\ge L_y,\qquad f_j(x)\le U_j.`

Then $`L_y-U_j>0` proves the pairwise class margin. A multiclass certificate repeats this for every
$`j\ne y`. The arithmetic is elementary; the substantive obligations are that:

- the bounds enclose the exact graph semantics;
- the graph denotes the intended model;
- the input box denotes the intended perturbation set;
- any rounded or native execution is related to the exact graph.

For a rounded implementation with coordinate errors
$`\lvert f_i^{\mathrm{run}}(x)-f_i(x)\rvert\le\varepsilon_i`, the transferred margin is

$$`f_y^{run}(x)-f_j^{run}(x)
\ge L_y-U_j-\varepsilon_y-\varepsilon_j.`

The right-hand side must remain positive. A real CROWN theorem alone does not prove the native
binary32 claim; the FP32 and runtime-approximation sections describe the additional bridge.

Both error terms are subtracted because rounding can move the two scores in opposite unfavorable
directions: down for the chosen class and up for its competitor. This also explains why a very
small positive real margin may be insufficient for a rounded implementation. If the transferred
lower bound is zero or negative, the argument has failed to establish strict dominance. It has
not produced an input where the classifier changes its label. Finding such an input is a
different result from failing to prove that none exists.

# Executable Certificates And Imported Artifacts

TorchLean currently checks several kinds of artifact, each with a deliberately limited meaning.

The graph numerical certificate records source ranges, derived node ranges, a registry identity,
and a backend-plan audit. `generateChecked` reconstructs this data, and `executeIEEE32` performs a
bit-level reference replay while checking each intermediate tensor. A `GraphRangeContract`
contains an executable `derive` function but no semantic soundness field. A proof-level error trace
therefore also requires a separately constructed `ProvedRealEnclosure`, whose fields supply the
real denotation and enclosure proof. See the runtime-approximation section for the complete
boundary.

The external alpha-beta-CROWN leaf checker `checkAbCrownLeafArtifact` validates the JSON schema,
finite ordered root and leaf boxes, dimensions, containment of each represented leaf in the root,
and the exported lower-bound witness relative to its threshold. It does not prove that the lower
bound came from the network semantics, and it does not prove that the represented leaves cover the
root box. Those are producer obligations.

This yields three distinct uses of "certificate":

- a Lean term containing proof fields;
- an artifact accepted by a structural or numerical checker;
- an externally produced claim whose semantic validity is assumed.

Proof-carrying code in the sense of {Informal.citet necula1997}[] requires a sound checker for the
property being claimed. Structural validation alone can be useful without establishing that
stronger property. The command's acceptance predicate, rather than the artifact's name, determines
what evidence is available.

The certificate chapter runs the leaf checker, changes a witness so that it must fail, and explains
which stronger artifact would be needed to obtain root-region soundness.

## Other Maintained Checkers

The verifier registry includes several implemented families that exercise different semantic
objects:

- `lirpa-mlp`, `lirpa-cnn`, `lirpa-attention`, `lirpa-gru`, and `lirpa-encoder` replay their
  corresponding finite LiRPA JSON formats. Acceptance is specific to each format and supported
  operator fragment.
- `camera-box3d-cert` checks a camera/3D-box artifact. Its
  [`Box3D`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/Geometry3D/Box3D.lean)
  implementation includes interval-operation soundness lemmas and `checkCert_sound`, connecting a
  successful pure check to the stated projection, positive-depth, and image/bounding-box guards.
- `vnncomp-mnistfc` parses the supported VNN-COMP-style MNIST-FC suite and runs the in-repo bound
  workflow. This is support for that declared suite, not for every VNN-LIB/ONNX benchmark.
- `digits` and `digits-train-certify` run the prepared sklearn-digits robustness paths; one checks
  supplied weights and the other trains before compiling and reporting bounds.
- the three `twostage-*` commands implement the Lyapunov workflows described in the two-stage
  chapter, with distinct external-producer, hybrid, and all-in-Lean boundaries.

`lake exe verify -- list` is the authoritative command inventory. A successful run establishes the
acceptance predicate or reported computation documented for that command; it does not merge these
heterogeneous formats into one global verification theorem.

# Floating-Point Models And Refinement

The proof float `FP32` is `NF binaryRadix fexp32 rnd32`: a rounded-real model with gradual
underflow. Its exponent description has no upper bound, so it does not model overflow, NaN,
infinity, or signed-zero payload behavior. FloatLib binary32 is the executable bit-level model.

The {src "NN/Floats/IEEEExec/Bridge/Finite.lean"}[finite addition theorem] connects executable
addition to the proof-level rounding model under the explicit hypothesis that the result is finite.
Corresponding
multiplication, division, and square-root bridges carry their own finite-path hypotheses. Layer
and MLP theorems then propagate explicit error budgets. Neither layer is an unstated theorem about
native hardware or a vendor reduction schedule.

The generic IEEE32 CROWN theorem likewise leaves the node evaluator and
`CrownTransferSound` proof to its caller. Choosing an IEEE scalar type does not discharge the
floating refinement obligations.

# Verification Evidence And Assumptions

A verification report should make the following boundary visible:

:::table +header
*
  * Evidence
  * Established in current source
  * Not established by that evidence
*
  * `Correctness.runForwardIR_eq_evalForward`
  * typed proved-program and IR evaluator agree
  * arbitrary frontend or native runtime agreement
*
  * `runIBP?_encloses_evalGraphRec`
  * proof-side real IBP encloses proof-side real semantics
  * every executable IBP implementation
*
  * `runIBP_encloses_evalGraphRec`
  * the executable real engine encloses the semantics on `EngineCore` graphs with full coverage
  * transcendental node kinds, rounded scalars, or the JSON checkers
*
  * `alphaCrown_cert_encloses_evalGraphRec`
  * α-CROWN enclosure with the IBP hypothesis discharged
  * that a produced certificate satisfies `CrownCertLocalOK` and `AlphaOK`
*
  * `cert_encloses_semantics`
  * enclosure from exact local IBP and semantic hypotheses
  * that a JSON/Float checker supplied those hypotheses
*
  * `alphaCrown_transfer_sound` / `alphaBetaCrown_transfer_sound`
  * exact real local affine transfer soundness
  * approximate artifact-checker acceptance
*
  * CROWN node checker returns `true`
  * artifact intervals contain the authoritative FloatLib binary32 IBP trace and affine entries
  exactly
    match a sequential FloatLib binary32 replay
  * exact-real CROWN soundness
*
  * alpha-beta leaf checker succeeds
  * represented boxes and witness fields pass structural/numeric checks
  * network-bound provenance or root coverage
*
  * numerical range check plus IEEE replay
  * stored trace and one reference execution pass executable checks
  * real semantic enclosure without `ProvedRealEnclosure`
*
  * FP32 approximation theorem
  * rounded-real output is within its stated budget
  * native IEEE behavior without finite refinement
:::

For a particular report, select the rows that refer to its graph, scalar type, and checker. The
last column identifies the additional evidence needed before composing them into a deployment
claim.

# References

IEEE 1788-2015 specifies the interval-arithmetic conventions followed by the `BoundOps` classes.
