import VersoManual
import NN.Verification.Cert.AbCrownLeafCert
import NN.Verification.Cert.IBPCert
import NN.Verification.Cert.IBPNodeCert
import NN.Verification.Cert.CROWNNodeCert
import NN.Verification.Cert.CROWNNodeCertAlphaBeta
-- The parse layer demonstrated below lives here rather than in the checker file.
import NN.Verification.Util.Json
-- The interval pass exercised under "Three Levels Of Checking" needs the tensor literal
-- notation, the IR evaluator, and the CROWN graph engine together with its soundness proof.
import NN.Tensor.Internal.Elab.TensorLiteral
import NN.IR.Semantics
import NN.MLTheory.CROWN.Proofs.GraphRunibpEndToEnd
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean

-- The checkers quoted below sit in a handful of sibling namespaces under `NN.Verification`, and
-- Verso keeps displayed code narrow enough to read beside the text. Opening the namespaces here
-- lets each `#check` fit on one line; the printed signatures still spell out every name in full.
open NN.Verification
open NN.Verification.CROWNNodeCertAlphaBeta
open NN.Verification.Util (approxEq)
open NN.Verification.Util.Tensor
open TorchLean (Tensor)
-- Two named imports rather than the whole `Json` namespace: the JSON helpers share short
-- names like `expectString` with the checker's own vocabulary, and only these two are used.
open NN.Verification.Json (fromExcept parseEndpointBoxRegion)

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
bounds. A certificate records evidence from that search for a separate checker to examine. The
producer can use expensive heuristics without putting all of those heuristics inside the trusted
argument, provided checker acceptance implies the desired property. This separation is also central
to the proof-carrying-code approach of {Informal.citet necula1997}[].

To connect this to the classifier, imagine that one bound over its whole input square is too
loose to show a positive class margin. A producer can split the square and try again on each
piece. One piece may be easy because every ReLU has a fixed phase there; another may require
more splits or a better relaxation. The final evidence should let a reader recover both what
was established on each piece and how those pieces account for the original square.

The search history can be much larger than the useful evidence. Failed choices of relaxation
parameters need not appear in a certificate if a checker can justify the final bounds directly.
That is the attraction of separating production from checking: search can change while the
meaning of acceptance stays fixed. But it makes the artifact's contents important. A file that
records only a claimed margin cannot support the same checks as a file that also records the
network, intermediate bounds, and a derivation of that margin.

The evidence determines how much the checker can establish. A list of boxes and claimed lower
bounds permits consistency checks on those fields. Proving that the lower bounds enclose the
network requires a graph and justification for the bound transfers. Proving a property over the
original input region also requires coverage by the represented boxes. The bundled leaf artifact
lets us inspect the consistency checks directly; its missing evidence identifies what a
root-region proof would still need.

TorchLean does not vendor the Two-Stage / α,β-CROWN repository {Informal.citep betacrown2021}[].
The core Lean build does not require that Python environment. Generating fresh α,β-CROWN leaf
artifacts requires external verifier output plus TorchLean's conversion helper.

During a branch-and-bound verification run, an instrumented external verifier can expose terminal
subdomains. TorchLean's helper converts that terminal-domain data into a small JSON *leaf artifact*
for each represented subdomain. TorchLean can parse that JSON and validate several
properties entirely inside Lean: every leaf box lies inside the declared root input region; every
leaf marked as verified satisfies the exported local prune test
($`\exists i,\;lb_i>threshold_i` in the exported fields); and the document is internally consistent
(dimensions, array lengths, and cross-references line up).

These checks focus on the JSON artifact itself: boxes nest correctly, verified leaves satisfy the
stated prune rule, and the fields fit together. The numeric bound propagation that produced each
`lb` belongs to the external producer unless a separate recompute-and-compare certificate path is
added.

The concrete path is:

- α,β-CROWN performs branch and bound outside Lean;
- the producer exports or exposes terminal leaf domains;
- TorchLean's converter writes those domains in `abcrown_leaf_artifact_v0_1.json`;
- TorchLean parses the JSON;
- Lean checks the structural predicate for each leaf;
- the checker accepts or rejects the artifact.

Here, `verified` or `pruned` means that a represented leaf passes the producer's exported local
test. It does not yet mean that Lean has proved the neural-network property on the root box.

# Bundled Leaf Checker Example

The sample artifact contains one two-dimensional leaf. Its box equals the declared root, its
exported lower bound is `1.0`, and its threshold is `0.0`.

```terminal
# Check the bundled artifact against the leaf format's
# structural predicates.
lake exe verify -- abcrown-leaf
```

Lean reports:

```terminal +output
[artifact] Checked 1 leaves: ok=1, bad=0
```

This counts represented leaves. `ok=1` means that the one parsed leaf passed containment and
witness checks; `bad=0` means that none failed those checks. The summary has no count of network
nodes because this entry point does not read or evaluate a network. We can change one field
to see exactly which of these checks rejects the artifact.

To see what was actually checked, make a temporary copy whose threshold is larger than the
exported lower bound:

```terminal
# Raise the threshold and keep the redundant margin
# consistent with that edit.
jq '.leaves[0].threshold=[2.0] | .leaves[0].witness_margin=-1.0' \
  NN/Examples/Verification/AbCrown/sample_abcrown_leaf_artifact_v0_1.json \
  > /tmp/torchlean_bad_leaf.json

lake exe verify -- abcrown-leaf /tmp/torchlean_bad_leaf.json
```

The command exits unsuccessfully:

```terminal +output
[artifact] leaf 0 rejected: witness index 0 does not satisfy lb > threshold
[artifact] Checked 1 leaves: ok=0, bad=1
error: Artifact failed checks for 1 leaves
```

The edited JSON still parses, but its witness no longer satisfies the prune inequality. Below,
we isolate that failure from two others: moving a leaf outside the root and choosing an index beyond
the bound vector.

A leaf is accepted when its box lies inside the root, its dimensions are coherent, and some
exported lower bound exceeds its threshold. An optional witness index restricts the comparison to
that coordinate; an optional margin must accompany a valid index and agree with its difference. The
prune test
itself has the form:

$$`\exists i,\qquad lb_i>threshold_i.`

The following tensors transcribe the bundled leaf from
{src "NN/Examples/Verification/AbCrown/sample_abcrown_leaf_artifact_v0_1.json"}[the sample file].
The leaf box equals the root, so in this particular artifact the single leaf covers the whole
region. The checker will inspect containment, the threshold comparison, and the recorded margin
separately.

```lean (name := leafData)
-- The bundled leaf occupies the entire declared
-- two-dimensional root box.
def rootLo : Tensor Float [2] := [-1.0, -1.0]
def rootHi : Tensor Float [2] := [1.0, 1.0]

def leafLo : Tensor Float [2] := [-1.0, -1.0]
def leafHi : Tensor Float [2] := [1.0, 1.0]

-- These are the producer's bound and threshold for its one
-- property expression.
def leafLb : Tensor Float [1] := [1.0]
def leafThr : Tensor Float [1] := [0.0]
```

Nesting is `boxWithin`. It compares equally shaped tensors coordinatewise. The JSON decoder rejects
length mismatches before constructing these tensors:

```lean (name := within)
-- Equal root and leaf endpoints satisfy coordinatewise
-- containment.
#eval boxWithin rootLo rootHi leafLo leafHi
```

```leanOutput within
true
```

The prune test is `refutesThresholdAt`. It uses the witness index recorded in the artifact rather
than searching for a coordinate that happens to work, so a leaf claiming coordinate `0` cannot be
rescued by coordinate `3`:

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

The prune comparison uses `Float` ordering directly, without a tolerance. It establishes a strict
inequality between the stored binary64 values. The redundant `witness_margin` field is checked with
`approxEq` at absolute tolerance `1e-6`, allowing a small discrepancy between the producer's printed
subtraction and the checker's recomputation. Passing that margin check cannot compensate for a
failed prune inequality.

The three `true` results answer three separate questions. The first concerns the input region,
the second compares a bound with a threshold, and the third checks a redundant subtraction.
The vectors also live in different spaces: `leafLo` and `leafHi` have two coordinates because
the input has dimension two, while `leafLb` and `leafThr` have one coordinate because this
artifact records one property expression. A property's coordinate need not be a raw output
class. It may already represent a linear combination of logits, such as a chosen class margin.

Here are values on either side of the margin tolerance:

```lean (name := marginTol)
#eval (approxEq 1.0 1.000001, approxEq 1.0 1.0000011)
-- A margin that is not a finite number is never
-- believed, whichever side it appears on.
#eval (approxEq 1.0 (1.0 / 0.0),
  approxEq (0.0 / 0.0) (0.0 / 0.0))
```

```leanOutput marginTol
(true, false)
```

```leanOutput marginTol
(false, false)
```

After binary64 conversion, the two decimal differences fall on opposite sides of the absolute
tolerance. This is not a relative parts-per-million comparison. `approxEq` also requires finite
operands before comparing their difference, so an infinite margin is rejected explicitly.

Now run the tampered copy from the previous section, whose threshold was raised to `2.0`:

```lean (name := tampered)
-- Keep the bound at 1.0 while moving its required threshold
-- to 2.0.
def tamperedThr : Tensor Float [1] := [2.0]

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

The failed leaf count therefore comes from the prune test. Updating the margin along with the
threshold isolates that predicate: the fields remain mutually consistent while the claimed
positive witness fails.

Equality at the threshold would fail too. The predicate asks for `lb_i > threshold_i`, so a
stored bound of zero against a zero threshold supplies no strict witness. This matters when
interpreting a small margin: the schema's tolerance belongs only to the redundant
`witness_margin` comparison. It does not relax the property threshold or let a slightly negative
bound count as positive. A consumer that silently reused the bookkeeping tolerance for pruning
would be checking a different statement.

The failed threshold example also shows why diagnostics should report the field that failed.
The exporter need not regenerate endpoint boxes when the boxes still pass containment. Nor does
a consistent subtraction rescue a threshold that the lower bound does not clear. Reading the
three flags separately lets us distinguish an inconsistent document from an insufficient
exported witness, even before asking whether that witness came from a sound network bound.

## Leaf Acceptance Predicates

For a parsed leaf supplying both witness fields, the acceptance flag is the conjunction of these
three checks:

```lean (name := verdicts)
def leafVerdict (lo hi : Tensor Float [2])
    (lb thr : Tensor Float [1])
    (idx : Nat) (margin : Float) : String :=
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

The bundled leaf passes all three checks. Raising the threshold fails the prune test; pushing
a corner to `1.5` fails containment while leaving the prune test true. An out-of-range witness index
fails both the prune and margin checks, since neither can retrieve the named coordinate.

All of these verdicts concern fields from the artifact. None yet connects its lower bound to the
network's outputs.

## Artifact Parsing And Validation

The rows above already contain shaped tensors of `Float` values. Before constructing those
tensors, the parser checks the JSON arrays for finite values, lengths, and endpoint order.
`parseEndpointBoxRegion` reads the box, and `expectFieldFiniteFloatArray` reads the bounds. A parse
failure reports which field could not be accepted:

```lean (name := parseLayer)
-- Build small JSON documents that isolate one parser
-- condition at a time.
def leafText (lo hi lb : String) : String :=
  "{\"lo\": " ++ lo ++ ", \"hi\": " ++ hi ++
    ", \"lb\": " ++ lb ++ ", \"threshold\": [0.0]}"

-- Report parsing and field-validation failures before any
-- prune comparison.
def tryLeaf (label text : String) : IO Unit := do
  match Lean.Json.parse text with
  | .error e => IO.println s!"{label}: not JSON ({e})"
  | .ok value =>
    try
      let box ←
        fromExcept (parseEndpointBoxRegion "leaf" value)
      let lb ←
        NN.Verification.Json.expectFieldFiniteFloatArray
          value "lb" "leaf"
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
bundled leaf: dim=2, lb=#[1.000000]
lb at infinity: rejected -- leaf.lb[0]: expected finite float
lo above hi: rejected -- leaf: invalid interval at coordinate 0:
  [1.000000, -1.000000]
center/eps box: rejected -- leaf: expected endpoint fields
  (`lo`, `hi`), not `center` or `eps`
```

These failures concern the representation. The parser rejects an infinite bound, so $`+\infty`
cannot enter the prune comparison as a claimed lower bound. It also rejects a box with reversed
endpoints; `boxWithin` checks endpoint order as well.

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
  ([0.0, 1.0] : Tensor Float [2]) [0.0, 0.0]
-- A recorded witness at coordinate zero must pass at that
-- exact coordinate.
#eval refutesThresholdAt
  ([0.0, 1.0] : Tensor Float [2]) [0.0, 0.0] 0
```

```leanOutput searchVsWitness
true
```

```leanOutput searchVsWitness
false
```

The same leaf, two verdicts. The searching form goes looking, finds coordinate `1`, and accepts.
The witnessed form is told to look at coordinate `0`, finds $`0.0>0.0` false, and says so. Neither
answer is a bug; they answer different questions. "Is some coordinate above its threshold" is a
question about the leaf, and "is the coordinate the producer named above its threshold" is a
question about the producer's claim. The second detects a recorded wrong witness. The schema permits
an omitted index, in which
case the checker uses the searching form.


Those checks are about exported numbers. They do not by themselves prove that the exported `lb`
values are lower bounds of the neural network. That stronger claim needs either recomputation in
Lean or a proof-backed certificate whose local transfer rules Lean can check.

Leaf nesting has the form:

$$`B_\ell\subseteq B_{\mathrm{root}}.`

A stronger branch certificate would also check coverage:

$$`B_{\mathrm{root}}\subseteq\bigcup_\ell B_\ell.`

For a semantic margin property, the target shape is:

$$`\forall x\in B_\ell,\qquad c^\top f(x)\ge threshold.`

The current checker does not establish that quantified statement. It accepts when the exported
boxes and witness fields are coherent and every represented leaf passes the finite comparison
$`lb_i>threshold_i`. Even if the lower-bound provenance were added, turning the leaves into a
root-region proof would still require separately checked coverage. In this fragment, the
certificate is structural.

Containment and coverage point in opposite directions. Containment prevents a leaf from making
claims outside the named problem. Coverage prevents parts of that problem from being omitted.
For example, two leaves can both lie inside a square while leaving an unrepresented strip
between them. Every local check may pass, yet no leaf says anything about an input in that strip.
Checking more carefully inside the two existing leaves cannot repair the missing region.

The existential prune rule also needs to be read in the producer's property convention.
If an unsafe specification is a conjunction of inequalities, falsifying one required inequality
can rule out that unsafe case. That explains the shape “there exists a coordinate” in this
format. It does not mean that beating any one competitor establishes multiclass robustness.
The producer must encode the intended property expressions and their thresholds correctly; this
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

Each row passes when at least one lower bound exceeds its threshold. This call computes bounds
for the supplied domain; the full α,β-CROWN verifier also manages branching and split constraints.
The artifact records the resulting bounds and thresholds for the terminal domains it exports.

Repeating the threshold comparison can detect a malformed witness or a serialization error, but
it cannot establish that propagation produced a sound `lb`. A wrong bound may still exceed its
threshold.

An independent Lean implementation checks containment, dimensions, witness indices, margins, and
prune comparisons after export. These checks can catch disagreements in the recorded data, although
the producer and checker may still share a misunderstood convention. At `v0.1`, neither the network
nor its bound transfers are present in the artifact, so the checker cannot recompute the lower
bound.

When the producer supplies `witness_idx`, the checker validates that coordinate rather than
searching for a replacement. Omitting it requests the existential check. Neither mode checks
coverage, so a dropped leaf can go undetected while all remaining leaves pass.

# Certificate Checking And Soundness

The `v0.1` format intentionally stops at structural checking. Lean checks that every represented
leaf lies inside the root box, that dimensions and arrays agree, that numeric fields are finite,
and that every leaf's witness satisfies $`\exists i,\;lb_i>threshold_i`.

The current artifact checks $`lb_i>threshold_i` for an exported lower bound. A stronger artifact
would also check that $`lb_i` is a sound lower bound for the graph on the leaf.

There are three progressively stronger designs:

- *Structural checking:* the artifact is self-consistent and each exported witness passes its
  stated arithmetic test. This is what `abcrown-leaf` provides.
- *Recompute and compare:* the artifact contains a network and enough node data for Lean to
  reproduce the bound calculation. TorchLean's node-certificate checkers recompute the complete
  trace with FloatLib binary32: interval entries must contain the authoritative trace, while affine
  replay entries must agree exactly at the binary32 level.
- *Proof-backed soundness:* checker acceptance supplies the exact hypotheses of a theorem that
  encloses the graph semantics. This requires a proved local transfer for every supported
  operator, plus a lowering correspondence and any required floating-point bridge.

The levels can share one producer workflow, but they support different claims.

For a concrete review, start with the conclusion you want. If the question is whether an exporter
preserved its witness index and margin, the leaf artifact contains the necessary fields. If the
question is whether an affine coefficient came from the stated network, a replay format must
identify the graph and parameters used to compute it. If the question is whether the resulting
bound holds for every real input in a leaf, a soundness theorem must connect the transfer rules
to that real-valued network. These questions determine which evidence is missing more precisely
than calling every JSON file a proof.

## IBP On A Two-Layer Graph

The propagation underlying the second level can be illustrated separately from JSON replay. Here is
a graph small
enough to fit on the page: two inputs, one hidden layer of width two, one output. The weights are
chosen so the composition is easy to name, and the input box is the unit square.

```lean (name := ctIbp)
-- Input node, two-coordinate affine layer, and
-- one-coordinate affine output.
def ctGraph : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := #[], kind := .input,
        outShape := [2] },
      { id := 1, parents := #[0], kind := .linear,
        outShape := [2] },
      { id := 2, parents := #[1], kind := .linear,
        outShape := [1] }] }

-- Both hidden coordinates depend on the same input square.
def ctStore : NN.MLTheory.CROWN.Graph.ParamStore Float :=
  { inputBoxes := (Std.HashMap.emptyWithCapacity).insert 0
      { dim := 2, lo := [-1.0, -1.0], hi := [1.0, 1.0] }
    linearWB := (Std.HashMap.emptyWithCapacity)
      |>.insert 1
          { m := 2, n := 2,
            w := [[1.0, 1.0], [1.0, -1.0]],
            b := [0.0, 0.0] }
      |>.insert 2
          { m := 1, n := 2,
            w := [[1.0, 1.0]], b := [0.0] } }

def ctBoxes :
    Array (Option (NN.MLTheory.CROWN.FlatBox Float)) :=
  NN.MLTheory.CROWN.Graph.runIBP (α := Float)
    ctGraph ctStore

def ctShow
    (b : Option (NN.MLTheory.CROWN.FlatBox Float)) :
    String :=
  match b with
  | none => "no box"
  | some B => s!"{repr B.lo} .. {repr B.hi}"

-- Inspect the hidden-node box first, then the output-node
-- box.
#eval ctShow ctBoxes[1]!
#eval ctShow ctBoxes[2]!
```
```leanOutput ctIbp
"[-2.000000, -2.000000] .. [2.000000, 2.000000]"
```
```leanOutput ctIbp
"[-4.000000] .. [4.000000]"
```

The pass seeds the input node with a box and applies each operation's transfer rule in order.
Here the hidden box has exact coordinate ranges. The first hidden
coordinate is $`x_0+x_1` and the second is $`x_0-x_1`, and both really do range over $`[-2,2]` on
the
unit square. The output box `[-4, 4]` is sound for the real affine calculation, but loose.

## Loss Of Correlation In Interval Bounds

The output of this network is $`(x_0+x_1)+(x_0-x_1)`, which is $`2x_0`. Its true range on the unit
square is $`[-2,2]`. Evaluate the same graph at the four corners and the values come back:

```lean (name := ctCorners)
-- Evaluate with the same weights used to construct the
-- interval boxes.
def ctPayload : NN.IR.Payload Float :=
  { linear? := fun id =>
      if id = 1 then
        some { outDim := 2, inDim := 2,
               W := [[1.0, 1.0], [1.0, -1.0]],
               b := [0.0, 0.0] }
      else if id = 2 then
        some { outDim := 1, inDim := 2,
               W := [[1.0, 1.0]], b := [0.0] }
      else none }

def ctRun (x : TorchLean.Tensor Float [2]) :
    Except String Float := do
  let out ← NN.IR.Graph.denote (α := Float) (g := ctGraph)
    (payload := ctPayload)
    (input := Spec.SomeTensor.ofTensor x) (outputId := 2)
  let t ← NN.IR.Graph.expectShape (expected := [1]) out
  pure t[0]

#eval [ctRun [1.0, 1.0], ctRun [1.0, -1.0],
  ctRun [-1.0, 1.0], ctRun [-1.0, -1.0]]
```
```leanOutput ctCorners (whitespace := lax)
[Except.ok 2.000000, Except.ok 2.000000, Except.ok (-2.000000), Except.ok (-2.000000)]
```

The function is linear, so the corners are where the extremes are, and the extremes are $`\pm2`. IBP
reported $`\pm4`: a factor of two too loose on a network with two layers and no activation. The
reason is that the second layer treats its two inputs as independent intervals when they are in fact
constrained by their shared inputs, and the cancellation of $`x_1` is invisible to a rule that only
sees
endpoints.
This is the dependency problem, and it is why linear relaxation methods such as CROWN
{Informal.citep crown2018}[] keep a symbolic affine form instead of a pair of numbers, and why
IBP-trained networks {Informal.citep gowal2018}[] are trained to make interval bounds tight rather
than expecting them to be tight by default.

Containment checks preserve the direction of an enclosure. If the authoritative recomputation
is enclosed by the supplied interval, the supplied interval may be looser and can still be valid.
An imported interval that is narrower needs additional justification; agreement with part of the
recomputed range is insufficient. This explains why the node checker allows widening of interval
side data but rejects shrinking. It does not certify that an accepted interval is tight.

## IBP Soundness Theorem

A recomputed interval becomes a semantic enclosure when the pass has a soundness theorem and its
hypotheses are satisfied. The real-valued interval pass has the following theorem:

```lean (name := ctThm)
-- Inspect the hypotheses that turn real interval
-- propagation into enclosure.
open NN.MLTheory.CROWN.Graph.CertSoundness in
#check @runIBP_encloses_evalGraphRec
```
```leanOutput ctThm (whitespace := lax)
runIBP_encloses_evalGraphRec : ∀ (g : NN.MLTheory.CROWN.Graph)
  (ps : NN.MLTheory.CROWN.Graph.ParamStore ℝ)
  (inputs : Std.HashMap ℕ Val),
  TopoSorted g →
    EngineCore g →
      IBPCovers g (runIBP? g ps) →
        InputsEnclosed g ps inputs →
          ∀ id < g.nodes.size,
            ∀ (B : NN.MLTheory.CROWN.FlatBox ℝ) (v : Val),
              (g.runIBP ps)[id]! = some B →
                (evalGraphRec g ps inputs)[id]! = some v →
                  EnclosesBox B v
```

To apply the theorem, we must establish each premise. `TopoSorted g` says parents come before
children, so a single left-to-right pass is a valid
schedule. `EngineCore g` restricts the graph to the operator fragment whose transfer rules have
proved local soundness; an unsupported operator does not make the theorem false, it makes it
inapplicable. `IBPCovers` says the proof-side pass produced a box at every node, which rules out the
vacuous reading where a missing box satisfies everything. `InputsEnclosed` says the seed boxes
really
do contain the inputs being evaluated. To claim a property over a chosen input region, that
premise must be established throughout that region.

With those hypotheses, each available engine box encloses the corresponding semantic value.
The statement is over $`ℝ`. The demonstration above used `Float`; applying the real theorem to a
rounded pass requires an arithmetic bridge. Using FloatLib binary32 in the node checkers fixes the
reference binary32 operations for replay, but does not by itself provide that bridge. The
{ref "fp32-soundness"}[floating-point soundness chapter] develops the relevant distinction.

The {ref "certificates"}[leaf artifact described here] carries no graph or transfer evidence, so
this theorem cannot be applied to it
directly. The CROWN and alpha-beta checkers do have acceptance theorems establishing local
consistency of their binary32 replay. A real enclosure still needs the real transfer hypotheses and
a refinement argument from that replay to the real semantics.

The last lines of the output are worth reading literally. Choose a valid node `id`, a box `B`,
and a value `v`. If the interval pass contains `some B` at that node and the evaluator contains
`some v`, the theorem concludes `EnclosesBox B v`. The word `some` is an `Option` constructor:
it records that the result exists at that position. The final proposition says every coordinate
of the value lies between the corresponding lower and upper endpoints. It says nothing about
how close those endpoints are to the true extrema, which is why the loose `[-4, 4]` result can
still be a valid enclosure of this affine example.

# Lean Entry Points

The signatures expose which checkers receive a network and which inspect only the artifact. The
leaf entry point takes a file path:

```lean (name := leafEntry)
-- This checker receives a path; the leaf format supplies no
-- network graph.
#check @Cert.AbCrownLeafCert.checkAbCrownLeafArtifact
```

```leanOutput leafEntry
Cert.AbCrownLeafCert.checkAbCrownLeafArtifact : String → IO Unit
```

The leaf CLI prints counts and raises `IO.userError` if any leaf fails, producing an unsuccessful
shell exit. The node checkers return a Boolean verdict that another Lean workflow can inspect.
Their file-reading and parsing stages may still raise errors; an `IO Bool` result does not remove
those failure modes.

```lean (name := nodeEntries)
-- Check one host-Float output box, with an optional
-- input-subdivision request.
#check @IBPCert.check
-- Replay every supplied interval node using the binary32
-- parameter store.
#check @IBPNodeCert.checkIBPNodeCertificate
-- Add exact comparisons for the affine CROWN transcript.
#check @CROWNNodeCert.checkCROWNNodeCertificate
-- This is the parsed alpha-beta certificate's data type.
#check @AlphaBetaCROWNNodeCertificate
-- Read and check an alpha-beta transcript for a supplied
-- graph and parameters.
#check @checkAlphaBetaCROWNNodeCertificate
```

```leanOutput nodeEntries (whitespace := lax)
@IBPCert.check : NN.MLTheory.CROWN.Graph →
  NN.MLTheory.CROWN.Graph.ParamStore Float → ℕ → String →
  optParam (Option (ℕ × ℕ)) none → IO Bool
```

```leanOutput nodeEntries (whitespace := lax)
IBPNodeCert.checkIBPNodeCertificate : NN.MLTheory.CROWN.Graph →
  NN.MLTheory.CROWN.Graph.ParamStore
      (ExecFloat.Binary 8 23 FloatFormat.Encoding.ieee (FloatFormat.Encoding.ieee.defaultBias 8)
        IBPNodeCert.readIBPNodeCertificate._proof_1 IBPNodeCert.readIBPNodeCertificate._proof_2
        IBPNodeCert.readIBPNodeCertificate._proof_3 IBPNodeCert.readIBPNodeCertificate._proof_4) →
    String → IO Bool
```

```leanOutput nodeEntries (whitespace := lax)
CROWNNodeCert.checkCROWNNodeCertificate : NN.MLTheory.CROWN.Graph →
  NN.MLTheory.CROWN.Graph.ParamStore
      (ExecFloat.Binary 8 23 FloatFormat.Encoding.ieee (FloatFormat.Encoding.ieee.defaultBias 8)
        CROWNNodeCert.checkCROWNNode._proof_1 CROWNNodeCert.checkCROWNNode._proof_2
        CROWNNodeCert.checkCROWNNode._proof_3 CROWNNodeCert.checkCROWNNode._proof_4) →
    String → IO Bool
```

```leanOutput nodeEntries
AlphaBetaCROWNNodeCertificate : Type
```

```leanOutput nodeEntries (whitespace := lax)
checkAlphaBetaCROWNNodeCertificate : NN.MLTheory.CROWN.Graph →
  NN.MLTheory.CROWN.Graph.ParamStore
      (ExecFloat.Binary 8 23 FloatFormat.Encoding.ieee (FloatFormat.Encoding.ieee.defaultBias 8)
        AlphaBetaCROWNNodeCertificate._proof_1 AlphaBetaCROWNNodeCertificate._proof_2
        AlphaBetaCROWNNodeCertificate._proof_3 AlphaBetaCROWNNodeCertificate._proof_4) →
    String → IO Bool
```

All four node/output checkers receive a graph and parameter store, so they can recompute bounds
for a supplied network. `IBPNodeCert`, `CROWNNodeCert`, and the alpha-beta variant use
`ParamStore (ExecFloat.Binary 8 23)` and compare against a specified binary32 execution. A
producer using
different rounding or reduction rules may disagree with that replay.

`IBPCert.check` instead uses host `Float` and takes an output-node identifier. Its artifact stores
an output box at that node, rather than a trace for every node.

Here `String → IO Unit` means “given a path, perform an IO action with no returned value.”
Success is communicated by completion and the printed summary; an exception signals failure.
The node signatures instead end in `IO Bool`, so successful IO returns a Boolean verdict.
The standalone `AlphaBetaCROWNNodeCertificate : Type` line describes a data type, not a checker
or a theorem. In `IBPCert.check`, `optParam ... none` displays a defaulted argument: callers
may omit the optional refinement request. Reading the result type and arguments first makes
the long namespace prefixes much less distracting.

`checkAlphaBetaCROWNNodeCertificate` has the same type as `checkCROWNNodeCertificate`. Neither
signature encodes an artifact kind. These node schemas do not require a distinguishing
`format` field: the selected checker parses its fields and recomputes its own rules. Callers must
choose the intended checker; shared or extra JSON fields do not establish that choice.

Use `abcrown-leaf` when the artifact is a branch-and-bound leaf summary. `IBPCert.check` checks a compact
Float output-bound artifact against a supplied graph and parameter store. `IBPNodeCert` instead
replays per-node interval data with FloatLib binary32. `CROWNNodeCert` adds affine CROWN data, and
the
alpha-beta variant adds the corresponding relaxation parameters. These formats are related, but
they are not interchangeable transcripts.

When reporting an accepted artifact, name the checker and the condition it established:

- `abcrown-leaf` checks a structural leaf artifact.
- `checkIBPNodeCertificate` recomputes the authoritative binary32 interval trace and requires the
  imported node boxes to contain it.
- `checkCROWNNodeCertificate` checks the additional affine transcript for the supported CROWN
  fragment.
- `checkAlphaBetaCROWNNodeCertificate` checks per-node α,β-CROWN transfer data by recomputation and
  exact binary32 transcript comparison; its interval side data may widen the recomputed boxes but
  may never shrink them.
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
- For real verification runs, pass the original input-property box as `root`. If the raw dump does
  not contain a root, the exporter can infer the componentwise leaf envelope as a structural
  fallback, but that fallback is only the envelope of the represented leaves.
- Each `leaf` is a sub-box of `root`.
- The root and every leaf must contain finite coordinates ordered coordinatewise
  ($`lo_i\le hi_i`), and the leaf array must be nonempty.
- `lb` and `threshold` are the lower bounds and thresholds reported by the external producer for
  that leaf at the moment it was pruned or verified.
- A leaf is considered "verified" iff $`\exists i,\;lb_i>threshold_i`.
  (This matches how `complete_verifier/input_split/branching_domains.py` filters out verified
  domains.)
- `witness_idx` and `witness_margin` are a convenience witness for the check above:
  $`witness\_margin=lb_{witness\_idx}-threshold_{witness\_idx}`.
  When `witness_idx` is present, the checker validates that exact coordinate rather than searching
  for a different witness. When `witness_margin` is present, it must accompany the index and agree
  with the recomputed margin up to the schema tolerance.

The schema deliberately does not contain a neural-network graph, α slopes, β phases, or per-node
affine forms. It is therefore a *leaf artifact*, not a full proof certificate.
The artifact records enough to check the terminal-domain bookkeeping exported by the producer; it
does not replay the producer's bound propagation.

The listing uses `...` where each producer supplies its own arrays, so it is a schema illustration
rather than a file to paste into a JSON parser. The complete bundled sample is the executable
example. In particular, `input_dim` governs the endpoint lengths, while each leaf's `lb` and
`threshold` must have matching lengths of their own. The latter length describes the number of
property expressions and can differ from `input_dim`. The format tag selects this interpretation
of the fields; it does not attach a soundness proof to their numeric contents.

# Producer Artifact Conversion

The accepted format and checker statement are defined here. The producer workflow uses
`scripts/verification/abcrown/export_leaf_artifact.py` to convert terminal domains into this schema,
either from the command line or from inside the producer process.

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
```

Two details of that conversion matter to a producer. The exporter reads the root box from the top
level rather than inferring it from the leaves, so the checked containment claim is about the input
property the verifier was given and not about the envelope of whatever leaves happened to be
exported; the envelope is only a fallback when the dump carries no root. And the witness fields are
computed rather than requested: the exporter fills in `witness_idx` and the matching
`witness_margin` (here `0.42` and `1.37`) from the bounds it was handed, which is why a hand-edited
margin is caught as a disagreement rather than believed.

A dump with one leaf and no list is accepted too, as is JSON Lines with one domain per line, since
producers stream terminal domains as they are closed.

# Leaf Checker CLI

Use the unified `verify` CLI tool `abcrown-leaf` to check the converted JSON artifact against
TorchLean's structural leaf predicate.

Example:

```terminal
# With no path argument, this command checks the maintained
# sample file.
lake exe verify -- abcrown-leaf
```

With no path, the command uses the bundled sample. With a path, it checks that artifact instead.
Run `lake exe verify -- list` to see the other registered certificate and workflow checkers,
including LiRPA, PINN, spline, logit-margin, and TorchLean-to-IR robustness paths.

The leaf command answers whether the exported leaf document satisfies
the `v0.1` structural contract. The neural-network verification chapter gives the theorem chain
needed for a semantic robustness result, while the two-stage chapter shows where an external
producer enters that chain.

# References

The branch-and-bound framework of {Informal.citet bunel2020}[] provides context for terminal
subdomains and pruning. The β-CROWN, LiRPA, and proof-carrying-code references above describe the
bound methods and the producer/checker separation.

The [α,β-CROWN project](https://github.com/Verified-Intelligence/alpha-beta-CROWN) contains the
producer implementation whose terminal-domain output the exporter reads.
