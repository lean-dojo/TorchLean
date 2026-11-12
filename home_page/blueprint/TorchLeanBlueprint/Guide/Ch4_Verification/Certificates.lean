import VersoManual
import TorchLean.Verify.Certificate.AbCROWNLeaf
import TorchLean.Verify.Certificate.IBP
import TorchLean.Verify.Certificate.IBPNode
import TorchLean.Verify.Certificate.CROWNNode
-- The parse layer demonstrated below lives here rather than in the checker file.
import TorchLean.Verify.JSON
-- The interval pass exercised under "IBP On A Two-Layer Graph" needs the tensor literal notation,
-- the builder and evaluator of graphs of registered operators, and the interval pass with its
-- soundness theorem.
import TorchLean.Tensor.Literal
import TorchLean.NN.IR.Builder
import TorchLean.Verify.CROWN.Op.ExecCert
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean

-- The checkers quoted below sit in a handful of sibling namespaces under `TorchLean.Certificate`,
-- and Verso keeps displayed code narrow enough to read beside the text. Opening the namespaces
-- here lets each `#check` fit on one line, and the printed signatures use those open namespaces.
open TorchLean
open Certificate
open Numeric (approxEq)
open Tensor
open TorchLean (Tensor)
-- Only the box and decimal readers are opened from `TorchLean.Json`.
open Json.BoxRegion (parseEndpoints)
open Json (parseDecimalRatArray)
-- The leaf checker compares exact rationals, whose `Context` instance is scoped here.
open scoped RationalAlgebraic
-- The soundness theorem of the interval pass below prints its operator-graph names relative to
-- this namespace.
open CROWN.Op

-- One signature below prints wider than this file's 100-column limit, so its `leanOutput` block
-- asks for `whitespace := lax` and is wrapped in the source. The rendered page still shows the
-- message exactly as Lean printed it.
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

#doc (Manual) "Verification Certificates" =>
%%%
tag := "certificates"
%%%

A branch-and-bound verifier explores subdomains, chooses relaxation parameters, and computes
bounds. A certificate records evidence from that search for a separate checker to examine. So the
producer can use expensive heuristics without putting all of them inside the trusted argument,
provided checker acceptance implies the desired property. This separation is also central to the
proof-carrying-code approach of {Informal.citet necula1997}[].

To connect this to the classifier, imagine that one bound over its whole input square is too loose
to show a positive class margin. A producer can split the square and try again on each piece. One
piece may be easy (every ReLU has a fixed phase there), while another may require more splits or a
better relaxation. Whatever evidence comes out at the end should let a reader recover both what was
established on each piece and how those pieces account for the original square.

A search's history can be much larger than the useful evidence. Failed choices of relaxation
parameters need not appear in a certificate if a checker can justify the final bounds without them.
That's the attraction of separating production from checking: search can change while the meaning
of acceptance stays fixed. But it makes the artifact's contents important. A file that records only
a claimed margin can't support the same checks as a file that also records the network,
intermediate bounds, and a derivation of that margin.

How much the checker can establish depends on the evidence. A list of boxes and claimed lower
bounds permits consistency checks on those fields. Proving that the lower bounds enclose the
network requires a graph and justification for the bound transfers, and proving a property over the
original input region also requires coverage by the represented boxes. With the bundled leaf
artifact we can inspect the consistency checks, and its missing evidence identifies what a
root-region proof would still need.

TorchLean doesn't vendor the Two-Stage / α,β-CROWN repository {Informal.citep betacrown2021}[], and
its core Lean build doesn't require that Python environment. Generating fresh α,β-CROWN leaf
artifacts requires external verifier output plus TorchLean's conversion helper.

During a branch-and-bound verification run, an instrumented external verifier can expose terminal
subdomains. TorchLean's helper converts that terminal-domain data into a JSON *leaf artifact*
containing the represented subdomains. TorchLean can parse that JSON and validate several
properties entirely inside Lean: every leaf box lies inside the declared root input region, every
leaf satisfies the exported local prune test ($`\exists i,\;lb_i>threshold_i` in the exported
fields), and the document is internally consistent (dimensions, array lengths, and cross-references
line up). And the leaf boxes must cover the root.

These checks focus on the JSON artifact itself: the boxes nest correctly and cover the root, all
leaves satisfy the stated prune rule, and the fields fit together. The numeric bound propagation
that produced each `lb` belongs to the external producer unless a separate recompute-and-compare
certificate path is added.

Here's the concrete path:

- α,β-CROWN performs branch and bound outside Lean
- the producer exports or exposes terminal leaf domains
- TorchLean's converter writes those domains in `abcrown_leaf_artifact_v0_1.json`
- TorchLean parses the JSON
- Lean checks the structural predicate for each leaf
- the checker accepts or rejects the artifact.

Here, `verified` or `pruned` means that a represented leaf passes the producer's exported local
test. It doesn't mean that Lean has proved the neural-network property on the root box.

# Bundled Leaf Checker Example

We bundle a sample artifact with one two-dimensional leaf. Its box equals the declared root, its
exported lower bound is `1.0`, and its threshold is `0.0`.

```terminal
# Check the bundled artifact against the leaf format's
# structural predicates.
scripts/lake.sh exe verify -- abcrown-leaf
```

Lean reports:

```terminal +output
[artifact] Checked 1 leaves: ok=1, bad=0
[artifact] consistent: the leaves cover the root and every leaf clears its threshold.
[artifact] The lower bounds are the producer's claims; TorchLean did not recompute them.
```

That report counts represented leaves: `ok=1` means that the one parsed leaf passed containment and
witness checks, and `bad=0` means that none failed those checks. There's no count of network nodes
in the summary (this entry point doesn't read or evaluate a network). We can change one field to
see which of these checks rejects the artifact.

To see what was actually checked, make a temporary copy whose threshold is larger than the
exported lower bound:

```terminal
# Raise the threshold and keep the redundant margin
# consistent with that edit.
jq '.leaves[0].threshold=[2.0] | .leaves[0].witness_margin=-1.0' \
  TorchLeanExamples/Verification/AbCROWN/sample_abcrown_leaf_artifact_v0_1.json \
  > /tmp/torchlean_bad_leaf.json

scripts/lake.sh exe verify -- abcrown-leaf /tmp/torchlean_bad_leaf.json
```

This time the verify command exits unsuccessfully:

```terminal +output
[artifact] leaf 0 rejected: witness index 0 does not satisfy lb > threshold
[artifact] Checked 1 leaves: ok=0, bad=1
error: Artifact failed checks for 1 leaves
```

After the edit, the JSON still parses, but its witness doesn't satisfy the prune inequality. We can
isolate that failure from two others: moving a leaf outside the root and choosing an index beyond
the bound vector.

A leaf is accepted when its box lies inside the root, its dimensions are coherent, and some
exported lower bound exceeds its threshold. An optional witness index restricts the comparison to
that coordinate, and an optional margin must accompany a valid index and agree with its difference.
In symbols, the prune test itself has the form:

$$`\exists i,\qquad lb_i>threshold_i.`

We transcribe the bundled leaf from
{src "TorchLeanExamples/Verification/AbCROWN/sample_abcrown_leaf_artifact_v0_1.json"}[the sample
file] as exact rationals, the values the checker reads from its decimals. The leaf box equals the
root, so in this particular artifact the single leaf covers the whole region. The checker inspects
containment, the threshold comparison, and the recorded margin separately.

```lean (name := leafData)
-- The bundled leaf occupies the entire declared
-- two-dimensional root box.
def rootLo : Tensor ℚ [2] := [-1.0, -1.0]
def rootHi : Tensor ℚ [2] := [1.0, 1.0]

def leafLo : Tensor ℚ [2] := [-1.0, -1.0]
def leafHi : Tensor ℚ [2] := [1.0, 1.0]

-- These are the producer's bound and threshold for its one
-- property expression.
def leafLb : Tensor ℚ [1] := [1.0]
def leafThr : Tensor ℚ [1] := [0.0]
```

Nesting is `boxWithin`, which compares equally shaped tensors coordinatewise. The JSON decoder
rejects length mismatches before constructing these tensors:

```lean (name := within)
-- Equal root and leaf endpoints satisfy coordinatewise
-- containment.
#eval boxWithin rootLo rootHi leafLo leafHi
```

```leanOutput within
true
```

The prune test is `refutesThresholdAt`. It uses the witness index recorded in the artifact and
doesn't search for a coordinate that happens to work, so coordinate `3` can't rescue a leaf
claiming coordinate `0`:

```lean (name := prune)
-- The named witness is coordinate zero, where 1.0 is
-- strictly greater than 0.0.
#eval refutesThresholdAt leafLb leafThr 0
```

```leanOutput prune
true
```

Last comes the bookkeeping field. `witness_margin` is supposed to be $`lb_i-threshold_i`, and the
checker recomputes it and compares:

```lean (name := margin)
-- Recompute the producer's recorded difference at the
-- witness coordinate.
#eval approxEq
  (leafLb.getScalar 0 - leafThr.getScalar 0) 1.0
```

```leanOutput margin
true
```

The prune comparison uses the plain order of the rationals, without a tolerance, so it establishes
a strict inequality between the decimals in the file. For the redundant `witness_margin` field,
`approxEq` checks at absolute tolerance `1 / 1000000`, allowing a small discrepancy between the
producer's printed subtraction and the checker's recomputation. Passing that margin check can't
compensate for a failed prune inequality.

Each of the three `true` results answers a separate question: the first concerns the input region,
the second compares a bound with a threshold, and the third checks a redundant subtraction. The
vectors also live in different spaces: `leafLo` and `leafHi` have two coordinates because the input
has dimension two, while `leafLb` and `leafThr` have one coordinate because this artifact records
one property expression. A property's coordinate need not be a raw output class. It may already
represent a linear combination of logits, such as a chosen class margin.

Here are values on either side of the margin tolerance:

```lean (name := marginTol)
#eval (approxEq (1.0 : ℚ) 1.000001,
  approxEq (1.0 : ℚ) 1.0000011)
```

```leanOutput marginTol
(true, false)
```

The first difference is exactly the absolute tolerance, which `approxEq` accepts, and the second
exceeds it. This is not a relative parts-per-million comparison. The checker reads a margin as the
exact rational of its decimal, so a margin such as the string `"Infinity"` is rejected before any
comparison.

Run the tampered copy made above, whose threshold was raised to `2.0`:

```lean (name := tampered)
-- Keep the bound at 1.0 while moving its required threshold
-- to 2.0.
def tamperedThr : Tensor ℚ [1] := [2.0]

#eval refutesThresholdAt leafLb tamperedThr 0
```

```leanOutput tampered
false
```

The `jq` edit also moved `witness_margin` to `-1.0`, which keeps the bookkeeping field consistent
with the tampered threshold:

```lean (name := tamperedMargin)
-- A correct subtraction can coexist with a failed strict
-- prune inequality.
#eval approxEq
  (leafLb.getScalar 0 - tamperedThr.getScalar 0) (-1.0)
```

```leanOutput tamperedMargin
true
```

So the failed leaf count comes from the prune test. Updating the margin along with the threshold
isolates that predicate: the fields remain mutually consistent while the claimed positive witness
fails.

Equality at the threshold would fail too. The predicate asks for `lb_i > threshold_i`, so a stored
bound of zero against a zero threshold supplies no strict witness. This matters when interpreting a
small margin: the schema's tolerance belongs only to the redundant `witness_margin` comparison. It
doesn't relax the property threshold or let a slightly negative bound count as positive. A consumer
that silently reused the bookkeeping tolerance for pruning would be checking a different statement.

## Leaf Acceptance Predicates

For a parsed leaf supplying both witness fields, the acceptance flag is the conjunction of these
three checks:

```lean (name := verdicts)
def leafVerdict (lo hi : Tensor ℚ [2])
    (lb thr : Tensor ℚ [1])
    (idx : Nat) (margin : ℚ) : String :=
  let within := boxWithin rootLo rootHi lo hi
  let pruned := refutesThresholdAt lb thr idx
  let bookkeeping :=
    if h : idx < 1 then
      approxEq (lb.getScalar ⟨idx, h⟩ -
        thr.getScalar ⟨idx, h⟩) margin
    else false
  let ok := within && pruned && bookkeeping
  s!"within={within} pruned={pruned} " ++
    s!"margin={bookkeeping} accept={ok}"

#eval do
  -- the bundled leaf
  IO.println <| leafVerdict
    [-1.0, -1.0] [1.0, 1.0] [1.0] [0.0] 0 1.0
  -- threshold raised past the lower bound
  IO.println <| leafVerdict
    [-1.0, -1.0] [1.0, 1.0] [1.0] [2.0] 0 (-1.0)
  -- one corner pushed outside the root box
  IO.println <| leafVerdict
    [-1.0, -1.0] [1.5, 1.0] [1.0] [0.0] 0 1.0
  -- witness index past the end of both arrays
  IO.println <| leafVerdict
    [-1.0, -1.0] [1.0, 1.0] [1.0] [0.0] 3 1.0
```

```leanOutput verdicts (whitespace := lax)
within=true pruned=true margin=true accept=true
within=true pruned=false margin=true accept=false
within=false pruned=true margin=true accept=false
within=true pruned=false margin=false accept=false
```

The bundled leaf passes all three checks. Raising the threshold fails the prune test, and pushing a
corner to `1.5` fails containment while leaving the prune test true. An out-of-range witness index
fails both the prune and margin checks, since neither can retrieve the named coordinate.

All of these verdicts concern fields from the artifact. None of them connects a lower bound from
the artifact to the network's outputs.

## Artifact Parsing And Validation

The checker reads the box endpoints with `parseEndpoints`, checking their lengths and order,
and the bounds with `parseDecimalRatArray`, all as the exact rationals their decimals denote. A
parse failure reports which field could not be accepted:

```lean (name := parseLayer)
-- Build small JSON documents that isolate one parser
-- condition at a time.
def leafText (lo hi lb : String) : String :=
  "{\"lo\": " ++ lo ++ ", \"hi\": " ++ hi ++
    ", \"lb\": " ++ lb ++ ", \"threshold\": [0.0]}"

-- Report parsing and field-validation failures before any
-- prune comparison.
def tryLeaf (label text : String) : IO Unit := do
  match DecimalJson.parse text with
  | .error e => IO.println s!"{label}: not JSON ({e})"
  | .ok value =>
    try
      let box ←
        IO.ofExcept (parseEndpoints "leaf" value)
      let lb ← IO.ofExcept do
        let lbValue ← value.field "lb" "leaf"
        parseDecimalRatArray "leaf.lb" lbValue
      IO.println s!"{label}: dim={box.dim}, lb={lb}"
    catch e =>
      IO.println s!"{label}: rejected -- {e}"

#eval do
  tryLeaf "bundled leaf"
    (leafText "[-1.0, -1.0]" "[1.0, 1.0]" "[1.0]")
  tryLeaf "lb at infinity"
    (leafText "[-1.0, -1.0]" "[1.0, 1.0]" "[\"Infinity\"]")
  tryLeaf "lo above hi"
    (leafText "[1.0, -1.0]" "[-1.0, 1.0]" "[1.0]")
  tryLeaf "center/eps box"
    ("{\"center\": [0.0, 0.0], \"eps\": 1.0, " ++
      "\"lb\": [1.0], \"threshold\": [0.0]}")
```

```leanOutput parseLayer (whitespace := lax)
bundled leaf: dim=2, lb=#[1]
lb at infinity: rejected -- leaf.lb[0]: expected a number
lo above hi: rejected -- leaf: invalid interval at coordinate 0:
  [1, -1]
center/eps box: rejected -- leaf: expected endpoint fields
  (`lo`, `hi`), not `center` or `eps`
```

These failures concern the representation. The reader accepts only JSON numbers, or strings holding
one, as bounds, so $`+\infty` cannot enter the prune comparison as a claimed lower bound. It also
rejects a box with reversed endpoints, and `boxWithin` checks endpoint order as well.

The region parser accepts endpoint boxes. A center-and-radius description using $`\epsilon`
must first be converted by the exporter. Keeping that conversion explicit makes it possible to
check which region the resulting endpoints describe.

## Explicit Witness Indices

An explicit index makes the producer's chosen witness part of the claim. Consider a leaf with
bound `0.0` at coordinate `0` and bound `1.0` at coordinate `1`, both compared with threshold `0.0`:

```lean (name := searchVsWitness)
-- Searching may use coordinate one, whose lower bound
-- clears its threshold.
#eval refutesThreshold
  ([0.0, 1.0] : Tensor ℚ [2]) [0.0, 0.0]
-- A recorded witness at coordinate zero must pass at that
-- exact coordinate.
#eval refutesThresholdAt
  ([0.0, 1.0] : Tensor ℚ [2]) [0.0, 0.0] 0
```

```leanOutput searchVsWitness
true
```

```leanOutput searchVsWitness
false
```

Here one leaf gets two verdicts. The searching form goes looking, finds coordinate `1`, and
accepts. Told to look at coordinate `0`, the witnessed form finds $`0.0>0.0` false and says so.
Neither answer is a bug, because they answer different questions. "Is some coordinate above its
threshold" is a question about the leaf, and "is the coordinate the producer named above its
threshold" is a question about the producer's claim. Asking the second question detects a recorded
wrong witness. When the index is omitted, which the schema permits, the checker uses the searching
form.

Leaf nesting has the form:

$$`B_\ell\subseteq B_{\mathrm{root}}.`

At the artifact level, the checker also requires coverage:

$$`B_{\mathrm{root}}\subseteq\bigcup_\ell B_\ell.`

For a semantic margin property, the target shape is:

$$`\forall x\in B_\ell,\qquad c^\top f(x)\ge threshold.`

TorchLean's leaf checker doesn't establish that quantified statement. It accepts when the exported
boxes and witness fields are coherent, every represented leaf passes the finite comparison
$`lb_i>threshold_i`, and the leaves cover the root. Missing lower-bound provenance still prevents a
semantic root-region proof. In this fragment, the certificate is structural.

Containment and coverage point in opposite directions. Containment prevents a leaf from making
claims outside the named problem. Coverage prevents parts of that problem from being omitted. For
example, two leaves can both lie inside a square while leaving an unrepresented strip between them.
Every local check may pass, yet no leaf says anything about an input in that strip. TorchLean's
artifact-level coverage check rejects this missing strip even when both local checks pass.

The existential prune rule also needs to be read in the producer's property convention. If an
unsafe specification is a conjunction of inequalities, falsifying one required inequality can rule
out that unsafe case, which explains the shape “there exists a coordinate” in this format. That
shape doesn't mean that beating any one competitor establishes multiclass robustness. Encoding the
intended property expressions and their thresholds correctly is the producer's job, because this
leaf checker sees the supplied arrays, not the logical construction that gave them meaning.

# Bound Computation And Pruning In Python

An external verifier already compares lower bounds with thresholds when deciding whether to
prune a domain. For example, `auto_LiRPA` exposes α-CROWN bounds through `compute_bounds`
{Informal.citep autolirpa2020}[]. If the model's outputs are the property expressions compared
with `threshold`, the leaf format's pruning condition is:

```
# Compute bounds on the property expressions represented by
# this bounded model.
lb, ub = bounded_model.compute_bounds(x=(bounded_x,), method="alpha-CROWN")
prunable = (lb - threshold > 0).any(dim=1)
```

Each row passes when at least one lower bound exceeds its threshold. This call computes bounds for
the supplied domain, while the full α,β-CROWN verifier also manages branching and split
constraints. For each exported terminal domain, the artifact records the resulting bounds and
thresholds.

Repeating the threshold comparison can detect a malformed witness or a serialization error, but it
can't establish that propagation produced a sound `lb` (a wrong bound may still exceed its
threshold).

An independent Lean implementation checks containment, dimensions, witness indices, margins, and
prune comparisons after export. These checks can catch disagreements in the recorded data, although
producer and checker may still share a misunderstood convention. At `v0.1`, neither the network nor
its bound transfers are present in the artifact, so the checker can't recompute the lower bound.

When the producer supplies `witness_idx`, the checker validates that coordinate without searching
for a replacement. Omitting it requests the existential check. In both modes, after all leaves
pass, `leavesCoverRoot` checks coverage on the closed grid cut out by the endpoints (skipping
coordinates spanned fully by every leaf), and it returns an error if the grid exceeds one million
cells.

# Certificate Checking And Soundness

For an exported lower bound, the leaf artifact's check is $`lb_i>threshold_i`. A stronger artifact
would also check that $`lb_i` is a sound lower bound for the graph on the leaf.

There are three progressively stronger designs:

- *Structural checking:* the artifact is self-consistent and each exported witness passes its
  stated arithmetic test, with the leaves covering the root. This is what `abcrown-leaf` provides.
- *Recompute and compare:* the artifact contains a network and enough node data for Lean to
  reproduce the bound calculation. TorchLean's interval node checker recomputes the complete trace
  with FloatLib binary32, and interval entries must contain the authoritative trace. The CROWN
  replay diagnostic lists where affine entries differ from a binary32 replay, which is not a
  verdict.
- *Proof-backed soundness:* checker acceptance supplies the exact hypotheses of a theorem that
  encloses the graph semantics. This requires a proved local transfer for every supported
  operator, plus a lowering correspondence and any required floating-point bridge.

All three levels can share one producer workflow, but they support different claims.

For a concrete review, start with the conclusion you want. If the question is whether an exporter
preserved its witness index and margin, the leaf artifact contains the necessary fields. If the
question is whether an affine coefficient came from the stated network, a replay format must
identify the graph and parameters used to compute it. If the question is whether the resulting
bound holds for every real input in a leaf, a soundness theorem must connect the transfer rules
to that real-valued network. These questions determine which evidence is missing more precisely
than calling every JSON file a proof.

## IBP On A Two-Layer Graph

We can illustrate the propagation underlying the second level separately from JSON replay. Here's a
graph small enough to fit on the page: two inputs, one hidden layer of width two, one output. We
chose the weights so the composition is easy to name, and the input box is the unit square.

```lean (name := ctIbp)
-- Input node, two-coordinate affine layer, and
-- one-coordinate affine output. Nodes 1, 2, 4 and 5 are
-- the weights and biases.
def ctGraph : Except String IR.Graph :=
  let (b, x) :=
    ({} : IR.Graph.Builder).input 0 (.value [2])
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

-- The weights of both layers, at the constant nodes that
-- read them.
def ctW1 : TorchLean.Tensor Float [2, 2] :=
  [[1.0, 1.0], [1.0, -1.0]]
def ctB1 : TorchLean.Tensor Float [2] := [0.0, 0.0]
def ctW2 : TorchLean.Tensor Float [1, 2] := [[1.0, 1.0]]
def ctB2 : TorchLean.Tensor Float [1] := [0.0]

def ctPayload : IR.Payload Float :=
  { const? := fun
      | 1 => some ⟨.value [2, 2], ctW1⟩
      | 2 => some ⟨.value [2], ctB1⟩
      | 4 => some ⟨.value [1, 2], ctW2⟩
      | 5 => some ⟨.value [1], ctB2⟩
      | _ => none }

-- Both hidden coordinates depend on the same input square.
def ctBox : CROWN.FlatBox Float :=
  { dim := 2, lo := [-1.0, -1.0], hi := [1.0, 1.0] }

-- Propagate the square and read the boxes by node id.
def ctBoxes :
    Array (Option (CROWN.FlatBox Float)) :=
  match ctGraph with
  | .ok g => runIBP g ctPayload (inputSeeds g [ctBox])
  | .error _ => #[]

def ctShow
    (b : Option (CROWN.FlatBox Float)) :
    String :=
  match b with
  | none => "no box"
  | some B => s!"{repr B.lo} .. {repr B.hi}"

-- Inspect the hidden-node box first, then the output-node
-- box.
#eval ctShow ctBoxes[3]!
#eval ctShow ctBoxes[6]!
```
```leanOutput ctIbp
"[-2.000000, -2.000000] .. [2.000000, 2.000000]"
```
```leanOutput ctIbp
"[-4.000000] .. [4.000000]"
```

Weights and biases sit in the payload at the constant nodes 1, 2, 4 and 5. The pass seeds the input
node with the box and applies each operation's transfer rule in order (the hidden layer is node 3
and the output node 6). Here the hidden box has exact coordinate ranges. Since the first hidden
coordinate is $`x_0+x_1` and the second is $`x_0-x_1`, both really do range over $`[-2,2]` on the
unit square. The output box `[-4, 4]` is sound for the real affine calculation, but loose.

## Loss Of Correlation In Interval Bounds

This network computes $`(x_0+x_1)+(x_0-x_1)`, which is $`2x_0`. Its true range on the unit square
is $`[-2,2]`. Evaluate the same graph at the four corners and the values come back:

```lean (name := ctCorners)
-- Evaluate with the same graph and payload used to
-- construct the interval boxes.
def ctRun (x : TorchLean.Tensor Float [2]) :
    Except String Float := do
  let g ← ctGraph
  match ← g.eval ctPayload [⟨.value [2], x⟩] with
  | [⟨⟨.value, [1]⟩, t⟩] =>
    pure (TorchLean.Tensor.getScalar t 0)
  | _ => throw "expected one vector of length one"

#eval [ctRun [1.0, 1.0], ctRun [1.0, -1.0],
  ctRun [-1.0, 1.0], ctRun [-1.0, -1.0]]
```
```leanOutput ctCorners (whitespace := lax)
[Except.ok 2.000000, Except.ok 2.000000, Except.ok (-2.000000), Except.ok (-2.000000)]
```

Because the function is linear, its extremes are at the corners, and they're $`\pm2`. IBP reported
$`\pm4`: a factor of two too loose on a network with two layers and no activation. That's because
the second layer treats its two inputs as independent intervals when they're in fact constrained by
their shared inputs, and the cancellation of $`x_1` is invisible to a rule that only sees
endpoints. This is the dependency problem. It's why linear relaxation methods such as CROWN
{Informal.citep crown2018}[] keep a symbolic affine form instead of a pair of numbers, and why
IBP-trained networks {Informal.citep gowal2018}[] are trained to make interval bounds tight rather
than expecting them to be tight by default.

Containment checks preserve the direction of an enclosure. If the authoritative recomputation is
enclosed by the supplied interval, the supplied interval may be looser and can still be valid. But
an imported interval that's narrower needs additional justification: agreement with part of the
recomputed range is insufficient. That's why the node checker allows widening of interval side data
but rejects shrinking. It doesn't certify that an accepted interval is tight.

## IBP Soundness Theorem

A recomputed interval becomes a semantic enclosure when the pass has a soundness theorem and its
hypotheses are satisfied. At `ℝ` the interval pass has the following theorem:

```lean (name := ctThm)
-- Inspect the hypotheses that turn real interval
-- propagation into enclosure.
#check @CROWN.Op.runIBP_encloses_of_run
```
```leanOutput ctThm (whitespace := lax)
@runIBP_encloses_of_run : ∀ {G : IR.Graph} {P : IR.Payload ℝ}
  (seeds : ℕ → Option (CROWN.FlatBox ℝ)) (inputs : List (Op.Value ℝ))
  (values : Array (List (Op.Value ℝ))),
  NoBackendEnclosure G →
    (∀ (id k : ℕ) (p : Port),
        G.nodes[id]? = some (IR.Node.input k p) →
          ∀ (B : CROWN.FlatBox ℝ),
            seeds id = some B →
              ∀ (x : Op.Value ℝ) (y : Port.Carrier ℝ p)
                  (v : CROWN.Graph.FlatTensor ℝ),
                inputs[k]? = some x → p.read x = Except.ok y → flatOf ⟨p, y⟩ = some v →
                    Directed.Encloses B v) →
      G.run P inputs = Except.ok values →
        ∀ id < G.nodes.size,
          ∀ (B : CROWN.FlatBox ℝ),
            (certify G P seeds)[id]! = some B →
              (runIBP G P seeds)[id]! = some B ∧
                ∀ (v : CROWN.Graph.FlatTensor ℝ), runVal values id = some v →
                    Directed.Encloses B v
```

To apply the theorem, we must establish each premise. `NoBackendEnclosure G` excludes tanh,
sigmoid, sine, and cosine (their rules in the pass are the backend's enclosures, not the
certificate checker's). Such an operator doesn't make the theorem false, it makes it inapplicable.
The seed premise says the seed boxes really do contain the inputs of the run. To claim a property
over a chosen input region, that premise must be established throughout that region. And the run
itself must succeed: `G.run P inputs` returns the values of every node. There is no ordering
premise, because a successful run reads every reference from an earlier node.

With those hypotheses, at each node where the certificate `Op.certify G P seeds` has a box, the
pass returns the same box, and it encloses the node's value in the run. A node without a
certificate box gets no claim, so the theorem is no promise that every node is bounded. Its
statement is over $`ℝ`. Our demonstration above used `Float`, and applying the real theorem to a
rounded pass requires an arithmetic bridge. Backends with `LawfulBoundOps` and
`LawfulNonlinearBoundOps` have the full forward theorem `Op.Directed.encloses_of_run` (the
rounded-real `FP32` model satisfies these scalar requirements). That theorem covers every operator
the pass boxes, given a real payload that holds the real value of each rounded constant and
enclosed inputs, and establishes enclosure for every returned box without assuming intermediate
enclosures. For an objective of the output's width and an input box enclosing graph input `0`,
`Op.backwardObjectiveBox?_of_run` carries these enclosures through the backward sweep and
final interval evaluation, taking the node equations from the same run. Its successful result
encloses the real objective, but neither theorem promises a result for every graph or a particular
bound tightness.

Host `Float` has no such lawful arithmetic instance. Using FloatLib binary32 in the node checkers
fixes the reference binary32 operations for replay, but doesn't by itself provide that bridge. The
{ref "fp32-soundness"}[floating-point soundness chapter] develops the relevant distinction.

The {ref "certificates"}[leaf artifact described here] carries no graph or transfer evidence, so
this theorem can't be applied to it directly. The CROWN replay diagnostic has a theorem,
`CROWNNode.replayAgrees_spec`, establishing local consistency of its binary32 replay, but no real
enclosure follows from it. The CROWN node verdict `CROWNNode.check` decides with
`FiniteArtifact.accepts` instead, and `FiniteArtifact.accepts_graph_sound` proves its requested
output inequalities for every real input in the box, on linear/ReLU chains.

The last lines of the output are worth reading literally. Choose a valid node `id` and a box `B`.
If the certificate contains `some B` at that node, the theorem concludes that the interval pass
contains `some B` there too and that `Directed.Encloses B v` holds for every value `v` the run has
at that node. Here `some` is an `Option` constructor: it records that the result exists at that
position. The final proposition says that the box has the value's width and that every coordinate
of the value lies between the corresponding lower and upper endpoints, but it says nothing about
how close those endpoints are to the true extrema, which is why the loose `[-4, 4]` result can
still be a valid enclosure of this affine example.

# Lean Entry Points

Checker signatures expose which checkers receive a network and which inspect only the artifact. The
leaf entry point takes a file path:

```lean (name := leafEntry)
-- This checker receives a path, and the leaf format
-- supplies no network graph.
#check @AbCROWNLeaf.check
```

```leanOutput leafEntry
AbCROWNLeaf.check : String → IO Report
```

Each checker returns a `Report`: the lines that report the check and the reason for a rejection,
`none` when the checker accepted the artifact (`Report.ok`). The checkers print nothing, so another
Lean workflow can inspect the report. The `verify` CLI prints the lines and raises `IO.userError`
with the reason, so a leaf that fails gives an unsuccessful shell exit. Reading and parsing the file
may still raise an error before any report exists.

```lean (name := nodeEntries)
-- Check one host-Float output box, with an optional
-- input-subdivision request.
#check @IBP.check
-- Replay every supplied interval node of an operator graph
-- with binary32 constants.
#check @IBPNode.check
-- Decide a CROWN certificate on exact rational transfers.
#check @CROWNNode.check
-- List where the affine CROWN transcript differs from the
-- binary32 replay.
#check @CROWNNode.replayDiagnostic
-- The parsed CROWN certificate, with optional ReLU phases.
#check @Certificate.CROWNNode
```

```leanOutput nodeEntries (whitespace := lax)
@IBP.check : IR.Graph →
  IR.Payload Float → List (CROWN.FlatBox Float) → ℕ → String →
  optParam (Option (ℕ × ℕ)) none → IO Report
```

```leanOutput nodeEntries (whitespace := lax)
IBPNode.check : IR.Graph →
  IR.Payload
      (ExecFloat.Binary 8 23 FloatFormat.Encoding.ieee (FloatFormat.Encoding.ieee.defaultBias 8)
        IBPNode.read._proof_1 IBPNode.read._proof_2 IBPNode.read._proof_3
        IBPNode.read._proof_4) →
    List
        (CROWN.FlatBox
          (ExecFloat.Binary 8 23 FloatFormat.Encoding.ieee
            (FloatFormat.Encoding.ieee.defaultBias 8)
            IBPNode.read._proof_1 IBPNode.read._proof_2 IBPNode.read._proof_3
            IBPNode.read._proof_4)) →
      String → IO Report
```

```leanOutput nodeEntries (whitespace := lax)
CROWNNode.check : IR.Graph →
  IR.Payload
      (ExecFloat.Binary 8 23 FloatFormat.Encoding.ieee (FloatFormat.Encoding.ieee.defaultBias 8)
        CROWNNode.nodeMismatch._proof_1 CROWNNode.nodeMismatch._proof_2
        CROWNNode.nodeMismatch._proof_3 CROWNNode.nodeMismatch._proof_4) →
    CROWN.FlatBox
        (ExecFloat.Binary 8 23 FloatFormat.Encoding.ieee (FloatFormat.Encoding.ieee.defaultBias 8)
          CROWNNode.nodeMismatch._proof_1 CROWNNode.nodeMismatch._proof_2
          CROWNNode.nodeMismatch._proof_3 CROWNNode.nodeMismatch._proof_4) →
      FiniteArtifact.Query → String → IO Report
```

```leanOutput nodeEntries (whitespace := lax)
CROWNNode.replayDiagnostic : IR.Graph →
  IR.Payload
      (ExecFloat.Binary 8 23 FloatFormat.Encoding.ieee (FloatFormat.Encoding.ieee.defaultBias 8)
        CROWNNode.nodeMismatch._proof_1 CROWNNode.nodeMismatch._proof_2
        CROWNNode.nodeMismatch._proof_3 CROWNNode.nodeMismatch._proof_4) →
    List
        (CROWN.FlatBox
          (ExecFloat.Binary 8 23 FloatFormat.Encoding.ieee
            (FloatFormat.Encoding.ieee.defaultBias 8)
            CROWNNode.nodeMismatch._proof_1 CROWNNode.nodeMismatch._proof_2
            CROWNNode.nodeMismatch._proof_3 CROWNNode.nodeMismatch._proof_4)) →
      String → IO (Array String)
```

```leanOutput nodeEntries
CROWNNode : Type
```

All three checkers receive a graph and its parameters, so they can recompute bounds for a
supplied network. `IBPNode.check` and `CROWNNode.replayDiagnostic` take a graph of registered
operators, its payload of `ExecFloat.Binary 8 23` constants and one input box per graph input, and
compare against a specified binary32 execution. A producer using different rounding or reduction
rules may disagree with that replay. `CROWNNode.check` takes the box of graph input `0` and the
requested output inequalities, a `FiniteArtifact.Query`, and compares nothing with a replay.

Both node formats index their arrays by operator-node id: entry `i` belongs to node `i` of the
operator graph, including the constant nodes of parameters and masks. In `ibp` and `crown`, a node
whose output `0` holds scalar values (every numeric constant included) must have an entry, and a
node whose output `0` holds Booleans, indices or integers (such as an attention mask) must have
`null`. `alpha` and `phases` are optional arrays: an absent one is `null` at every node, and a
`null` entry gives its node no α vector or no phases. Every constant node must also have its
constant in the payload. Input boxes are given in graph input order (one per graph input index),
not by node id.

The checkers read each box endpoint and affine coefficient as its rounded binary32 value, not as
its real value: the decimal is converted once as in IEEE 754-2019 section 5.12 (outward for a box
endpoint and to nearest for a coefficient), and it overflows as section 7.4 says for that rounding
direction. To nearest, a magnitude of at least `2 ^ 128 - 2 ^ 103` (halfway between the largest
finite value and `2 ^ 128`) is the infinity of its sign, and a smaller one is finite. So `1e39` is
`+∞` as an upper endpoint or a coefficient and the largest finite value as a lower endpoint (and
`-1e39` symmetrically). For the infinities, the writers `NodeReplay.flatBoxValue` and
`NodeReplay.affineBoundsValue?` emit `±1e39`. The replay diagnostic reports a `crown` entry that
differs from Lean's replay in any bit. Matrix entries must be finite, and an offset may be the
infinity that bounds nothing on its side (`-∞` in `loC` and `+∞` in `hiC`), but the other infinity
is refused. `CROWNNode.check` refuses an infinite coefficient in every entry it decodes: those of
the input node and of the `Op.linear` and `Op.relu` nodes.

`IBP.check` instead uses host `Float` and takes the index of a graph output. Its artifact
stores the box of that output, not a trace for every node.

Here `String → IO Report` means “given a path, perform an IO action that returns a report.” An
exception signals a file that the checker cannot read in its format, and a report with a rejection
signals an artifact that fails the check. Signatures of the node checkers end in the same
`IO Report`, and the replay diagnostic ends in `IO (Array String)`, its list of differences. On its
own line, `CROWNNode : Type` describes a data type, not a checker or a theorem. In `IBP.check`,
`optParam ... none` displays a defaulted argument (callers may omit the optional refinement
request). If you read the result type and arguments first, the long namespace prefixes are much
less distracting.

`IBPNode.check` and `CROWNNode.replayDiagnostic` differ only in their results, and no signature
encodes an artifact kind. These node schemas don't require a distinguishing `format` field: the
selected checker parses its fields and recomputes its own rules. So callers must choose the
intended checker, and shared or extra JSON fields don't establish that choice.

Use `abcrown-leaf` when the artifact is a branch-and-bound leaf summary. `IBP.check` checks a
compact Float output-bound artifact against a supplied graph of registered operators, its payload
and its input boxes. `IBPNode` instead replays per-node interval data with FloatLib binary32.
`CROWNNode` adds affine CROWN data and optional ReLU phase tags already justified by the IBP
intervals. Its diagnostic replays α-CROWN bounds, with no β multipliers or branch splits, and its
verdict checks the affine data against exact rational transfers. These formats are related, but
they aren't interchangeable transcripts.

When reporting an accepted artifact, name the checker and the condition it established:

- `abcrown-leaf` checks a structural leaf artifact.
- `IBPNode.check` recomputes the authoritative binary32 interval trace and requires the imported
  node boxes to contain it.
- `CROWNNode.check` accepts exactly when `FiniteArtifact.accepts` does (`CROWNNode.verdict_ok`).
  That decision checks the affine bounds of a CROWN certificate on a linear/ReLU chain against
  exact rational transfers, and `FiniteArtifact.accepts_graph_sound` proves the requested output
  inequalities for every real input in the box.
- `CROWNNode.replayDiagnostic` is not a verdict. It recomputes α-CROWN transfers for the supported
  CROWN fragment, validates any IBP-stable phase tags, and lists where the affine transcript
  differs from the binary32 replay. Parsed interval side data may widen the recomputed boxes but
  may never shrink them. Decimal endpoints are first rounded outward to binary32. Lean's replay
  rounds to nearest, so an empty list does not show that the affine bounds enclose the graph.
- graph soundness theorems apply only when the certificate format and graph fragment supply the
  hypotheses those theorems demand.

# File Format: `abcrown_leaf_artifact_v0_1.json`

Top-level object:

```
{
  "format": "abcrown_leaf_artifact_v0_1",
  "input_dim": 2,
  "root": { "lo": [-4.8, -10.8], "hi": [4.8, 10.8] },
  "leaves": [
    {
      "lo": [...],
      "hi": [...],
      "lb": [...],
      "threshold": [...],
      "witness_idx": 0,
      "witness_margin": 0.123
    }
  ]
}
```

Semantics:

- `root` describes the input box being verified.
- For real verification runs, pass the original input-property box as `root`. If the raw dump
  doesn't contain a root, the exporter can infer the componentwise leaf envelope as a structural
  fallback, but that fallback is only the envelope of the represented leaves.
- Each `leaf` is a sub-box of `root`, and together the leaves must cover `root`.
- The root and every leaf must contain finite coordinates ordered coordinatewise ($`lo_i\le hi_i`),
  and the leaf array must be nonempty.
- `lb` and `threshold` are the lower bounds and thresholds reported by the external producer for
  that leaf at the moment it was pruned or verified.
- The exported prune test passes iff $`\exists i,\;lb_i>threshold_i`, which is a comparison of
  producer-supplied fields, not proof of a network bound.
- `witness_idx` and `witness_margin` are an optional convenience witness for the check above:
  $`witness\_margin=lb_{witness\_idx}-threshold_{witness\_idx}`. When `witness_idx` is present, the
  checker validates that exact coordinate and doesn't search for a different witness. When
  `witness_margin` is present, it must accompany the index and agree with the recomputed margin up
  to the schema tolerance.

We deliberately left a neural-network graph, α slopes, ReLU phases, and per-node affine forms out
of the schema. So it's a *leaf artifact*, not a full proof certificate. It records enough to check
the terminal-domain bookkeeping exported by the producer, but it doesn't replay the producer's
bound propagation.

The listing uses `...` where each producer supplies its own arrays, so it's a schema illustration,
not a file to paste into a JSON parser (the complete bundled sample is the executable example). In
particular, `input_dim` governs the endpoint lengths, while each leaf's `lb` and `threshold` must
have matching lengths of their own (that second length describes the number of property expressions
and can differ from `input_dim`). The format tag selects this interpretation of the fields, but it
doesn't attach a soundness proof to their numeric contents.

# Producer Artifact Conversion

The accepted format and checker statement are defined here, while the producer workflow uses
`scripts/verification/abcrown/export_leaf_artifact.py` to convert terminal domains into this schema
(either from the command line or from inside the producer process).

The exporter accepts several field names used by external producers. This example supplies the
root endpoints as `x_L` and `x_U`, terminal domains under `domains`, and lower bounds as
`lower_bounds` with thresholds under `rhs`:

```
{
  "x_L": [-1.0, -1.0],
  "x_U": [1.0, 1.0],
  "domains": [
    { "x_L": [-1.0, -1.0], "x_U": [0.0, 1.0], "lower_bounds": [0.42], "rhs": [0.0] },
    { "x_L": [0.0, -1.0], "x_U": [1.0, 1.0], "lower_bounds": [1.37], "rhs": [0.0] }
  ]
}
```

Converting and checking it in one command:

```terminal +output
$ python3 scripts/verification/abcrown/export_leaf_artifact.py \
    --input raw_dump.json --out exported.json --check
Wrote TorchLean alpha-beta-CROWN-style leaf artifact to exported.json
[artifact] Checked 2 leaves: ok=2, bad=0
[artifact] consistent: the leaves cover the root and every leaf clears its threshold.
[artifact] The lower bounds are the producer's claims; TorchLean did not recompute them.
```

Two details of that conversion matter to a producer. The exporter reads the root box from the top
level instead of inferring it from the leaves, so the checked containment claim is about the input
property the verifier was given and not about the envelope of whatever leaves happened to be
exported. (The envelope is only a fallback when the dump carries no root.) And the witness index
is computed, not requested: the exporter fills in `witness_idx` from the bounds it receives and
leaves out the optional `witness_margin`, so the checker recomputes each margin (here `0.42` and
`1.37`) exactly from the file's decimals. A margin added by hand is compared with that
recomputation at the schema tolerance, so a wrong one is caught as a disagreement rather than
believed.

A dump with one leaf and no list is accepted too, as is JSON Lines with one domain per line, since
producers stream terminal domains as they are closed.

Run `scripts/lake.sh exe verify -- list` to see the other registered certificate and workflow
checkers, including the IBP certificate (`ibp-cert`), PINN, spline, ODE and TorchLean-to-IR
robustness paths.
The neural-network verification chapter gives the theorem chain needed for a semantic robustness
result, while the two-stage chapter shows where an external producer enters that chain.

# References

For terminal subdomains and pruning, the branch-and-bound framework of {Informal.citet bunel2020}[]
provides context. The β-CROWN, LiRPA, and proof-carrying-code references above describe the bound
methods and the producer/checker separation.

The [α,β-CROWN project](https://github.com/Verified-Intelligence/alpha-beta-CROWN) contains the
producer implementation whose terminal-domain output the exporter reads.
