import VersoManual
import FloatLib
import NN.MLTheory.CROWN.Lyapunov.Verification
import NN.Tensor
-- The coverage experiment below calls the same array predicates the leaf checker calls, so it
-- imports them rather than restating them. If one of those predicates changes meaning, this
-- chapter's output changes with it.
import NN.Verification.Util.Tensor
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open Spec
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Lyapunov
-- Tensor predicates for the coverage experiment.
open NN.Verification.Util.Tensor
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

#doc (Manual) "Two-Stage Verification" =>
%%%
tag := "twostage"
file := "Two-Stage-Verification-Workflows"
%%%

A two-stage workflow lets an external verifier search for bounds, then passes evidence to Lean
for checking. The producer may split boxes, optimize relaxations, and use GPU kernels. The checker
needs enough evidence to justify the result without reproducing every search decision. Which
conclusion it can justify depends on what the artifact records.

Return to a classifier whose label should stay fixed in an input square. A producer may spend
many passes trying to lower-bound the difference between two scores. It can adjust an affine
relaxation, split the square, or choose the next region to examine. A consumer receiving the
final result needs a more stable description: which model and square were checked, what bound
is claimed on each part, and what evidence justifies those bounds. Repeating every failed
search attempt would add work without helping establish the final claim.

The distinction is useful even when both stages run on the same machine. Search code is organized
around finding useful evidence; checking code is organized around the conditions that make that
evidence valid. A change to the search heuristic can leave those conditions unchanged. Conversely,
changing the model, input region, or interpretation of the scores changes the claim, even if
the JSON still has the same field names.

The producer is allowed to be complicated:

```
trained model + input property
  -> external verifier
  -> branch-and-bound leaves and claimed lower bounds
```

The consumer should be narrow:

```
JSON artifact
  -> finite parser
  -> schema and local predicate checks
  -> accept or reject
```

The producer/checker separation follows the proof-carrying-code approach of
{Informal.citet necula1997}[]. TorchLean's paths implement different parts of that idea. The leaf
workflow checks a summary of terminal domains; the controller workflows also run bound propagation.
The examples below follow the data from producer to consumer and identify the evidence available at
each step.

# External Search And Certificate Checking

A search procedure chooses where to spend computation. α,β-CROWN's branch-and-bound method
{Informal.citep betacrown2021}[], for example, refines relaxations and splits domains until it can
prune them or must continue searching. A sound certificate checker would allow those search choices
to remain untrusted: any accepted evidence would have to imply the property regardless of how it
was found.

For a leaf, the semantic claim is that a margin bound holds at every point of its sub-box. This is
a quantified statement about a fixed network, even though a finite artifact may carry evidence for
it. Checking only the exported endpoints and margins does not establish that statement.

TorchLean's leaf format currently checks a narrower set of facts: dimensions, containment, witness
indices, and comparisons among exported numbers. It can detect inconsistent fields. A consistently
misapplied sign or axis convention can still pass, because the artifact has no network or transfer
rules against which to check that convention.

The phrase *checked by Lean* covers everything from parsing a JSON file to replaying every bound
computation and deriving a theorem. TorchLean's leaf checker sits near the parsing end. The
{ref "certificates"}[certificate chapter] derives its predicates line by line; this chapter starts
one step earlier, at the producer, and follows the data outward.

# Producer Artifact Export

The external verifier's terminal-domain data must be converted to TorchLean's schema. The adapter
{src "scripts/verification/abcrown/export_leaf_artifact.py"}[`export_leaf_artifact.py`] does the
translation. It accepts the field names external tools actually use, normalizes them, computes a
witness coordinate, and writes the checked format.

The bundled raw dump is written in α,β-CROWN's vocabulary, not TorchLean's:

```
{
  "root": {
    "lo": [-1.0, -1.0],
    "hi": [1.0, 1.0]
  },
  "domains": [
    {
      "x_L": [-1.0, -1.0],
      "x_U": [1.0, 1.0],
      "lower_bounds": [1.0],
      "thresholds": [0.0]
    }
  ]
}
```

Run the whole producer-to-checker path on it:

```terminal
# Normalize the bundled terminal-domain dump and check the
# resulting leaf document.
python3 scripts/verification/abcrown/export_leaf_artifact.py \
  --input NN/Examples/Verification/AbCrown/example_raw_leaf_dump.json \
  --out /tmp/torchlean-abcrown-artifact.json \
  --check
```

```terminal +output
Wrote TorchLean alpha-beta-CROWN-style leaf artifact to /tmp/torchlean-abcrown-artifact.json
[artifact] Checked 1 leaves: ok=1, bad=0
```

The second line comes from Lean: `--check` shells out to
`lake exe verify -- abcrown-leaf` on the file it just wrote. The exported artifact is

```
{
  "format": "abcrown_leaf_artifact_v0_1",
  "input_dim": 2,
  "leaves": [
    {
      "hi": [1.0, 1.0],
      "lb": [1.0],
      "lo": [-1.0, -1.0],
      "threshold": [0.0],
      "witness_idx": 0,
      "witness_margin": 1.0
    }
  ],
  "root": {
    "hi": [1.0, 1.0],
    "lo": [-1.0, -1.0]
  }
}
```

The writer sorts keys and puts one array element per line, so the file on disk is taller than the
listing above, which is reflowed to fit the page. The field names and their order are the real
ones. Sorted keys are not a stylistic preference: two runs of the exporter on the same input produce
byte-identical files, which makes the artifact diffable in review and hashable in a report.

The adapter first normalizes field names. A leaf's lower box may arrive as `lo`, `input_lo`,
`x_L`, `x_l`, `domain_lo`, or `lower`; lower bounds may use `lb`, `lower_bound`, `lower_bounds`,
`output_lb`, or `margin_lb`. `_get_any` selects the first present field from its alias list and
raises `ArtifactExportError` if none is present. It does not infer whether the producer used a
field with the intended meaning or reject conflicting aliases merely because both are present.
The Lean parser receives the normalized spelling.

*The witness is computed, not trusted.* The adapter picks the coordinate with the largest positive
margin `lb[i] - threshold[i]` and records both the index and the margin. When the index is present,
the checker validates that coordinate. The schema also permits an
omitted index, in which case the checker searches for a positive margin. Neither route validates
the underlying network bound.

For the bundled domain, the witness calculation is `1.0 - 0.0 = 1.0`, so index zero is the only
possible choice. `input_dim = 2` comes from the two endpoint coordinates, while the bound and
threshold arrays have length one. This tells us that the domain is two-dimensional and carries
one property expression. The converter can establish those lengths and arithmetic relationships
from the supplied data. It cannot infer what network expression the producer placed in that
one-element bound vector.

*A leaf with no positive margin is rejected at export time.* Feed the adapter a domain whose lower
bound sits below its threshold:

```
{
  "root": { "lo": [-1.0, -1.0], "hi": [1.0, 1.0] },
  "domains": [
    {
      "x_L": [-1.0, -1.0], "x_U": [1.0, 1.0],
      "lower_bounds": [-0.25], "thresholds": [0.0]
    }
  ]
}
```

```terminal +output
$ python3 scripts/verification/abcrown/export_leaf_artifact.py \
    --input /tmp/torchlean-raw-nowitness.json \
    --out /tmp/torchlean-nowitness.json
error: leaf: no verified witness found; expected some coordinate with lb > threshold
$ echo $?
2
```

The export fails before writing the requested artifact. This format represents leaves with
positive witnesses; a domain without one cannot be encoded as verified. For a root-region claim,
such a failure cannot be repaired by simply dropping that domain. Its region still needs coverage
and a valid bound.

For instrumenting a verifier in place, the same script exposes the function directly:

```
# Call the adapter after collecting terminal leaves from the
# external search.
from scripts.verification.abcrown.export_leaf_artifact import \
    write_abcrown_leaf_artifact

write_abcrown_leaf_artifact(
    root_lo=original_property_lo,
    root_hi=original_property_hi,
    leaves=terminal_verified_leaves,
    out_path="leaf_artifact.json",
)
```

The call belongs after the search has collected its terminal leaves. TorchLean does not vendor
α,β-CROWN or the neural-controller repositories: their Python, CUDA, and solver dependencies stay
in their own environments, and the core Lean build needs only the exported JSON.

# Root Boxes And Leaf Coverage

The root endpoints identify the property region. Omitting them lets the adapter construct an
envelope from the leaves, which changes what the containment check refers to.

Here is a two-leaf dump that splits the unit square down the middle, with no root recorded:

```
{
  "domains": [
    { "x_L": [-1.0, -1.0], "x_U": [0.0, 1.0],
      "lower_bounds": [0.5], "thresholds": [0.0] },
    { "x_L": [0.0, -1.0], "x_U": [1.0, 1.0],
      "lower_bounds": [1.25], "thresholds": [0.0] }
  ]
}
```

```terminal +output
$ python3 scripts/verification/abcrown/export_leaf_artifact.py \
    --input /tmp/torchlean-raw-2leaf.json \
    --out /tmp/torchlean-2leaf.json --check
Wrote TorchLean alpha-beta-CROWN-style leaf artifact to /tmp/torchlean-2leaf.json
[artifact] Checked 2 leaves: ok=2, bad=0
```

The adapter derives the following root from the leaves:

```
"root": { "lo": [-1.0, -1.0], "hi": [1.0, 1.0] }
```

Each leaf lies inside this envelope by construction. The check establishes containment in that
derived box, but the box supplies no independent evidence of the original property region. A run
record must retain the original root if the claim is about that region.

Now supply a root twice as large as the region the leaves actually cover:

```terminal +output
$ python3 scripts/verification/abcrown/export_leaf_artifact.py \
    --input /tmp/torchlean-raw-2leaf.json \
    --out /tmp/torchlean-2leaf-wide.json \
    --root-lo=-2,-2 --root-hi=2,2 --check
Wrote TorchLean alpha-beta-CROWN-style leaf artifact to /tmp/torchlean-2leaf-wide.json
[artifact] Checked 2 leaves: ok=2, bad=0
```

The leaves cover $`[-1,1]^2`. The declared property region is $`[-2,2]^2`. The checker accepts, and
it is right to: every check it advertises still holds. It never claimed the leaves *cover* the
root. Coverage is a union condition over all leaves, the current checker does not compute the union
of the listed boxes, and the schema does not
record the producer's search tree.

An accepted `abcrown-leaf` artifact establishes that *each listed leaf is a well-formed sub-box
of the declared root, and each carries a witness that passes its own arithmetic test.* Whether the
listed leaves exhaust the property region is a claim the producer makes in prose, and the checker
does not see it.

The wide-root example separates two questions that often get compressed into “the boxes look
right.” Each leaf is a valid subdomain of the declared root. But a point such as `(1.5, 0.0)`
belongs to that root and to neither represented leaf. Even perfect network bounds on the
leaves would say nothing about that point. A root-region theorem must be able to take an
arbitrary root input, locate a leaf containing it, and apply the leaf's bound. The coverage
evidence supplies exactly that middle step.

Coverage can also fail when the root is correct. This next dump records the intended root
$`[-1,1]^2` and two leaves, leaving a strip in the middle that neither leaf covers. This is the
kind of gap a lost branch-and-bound subtree could leave.

```
{
  "root": {
    "lo": [-1.0, -1.0],
    "hi": [1.0, 1.0]
  },
  "domains": [
    { "x_L": [-1.0, -1.0], "x_U": [0.0, 1.0],
      "lower_bounds": [0.5], "thresholds": [0.0] },
    { "x_L": [0.5, -1.0], "x_U": [1.0, 1.0],
      "lower_bounds": [1.25], "thresholds": [0.0] }
  ]
}
```

```terminal +output
$ python3 scripts/verification/abcrown/export_leaf_artifact.py \
    --input /tmp/torchlean-raw-hole.json \
    --out /tmp/torchlean-hole.json --check
Wrote TorchLean alpha-beta-CROWN-style leaf artifact to /tmp/torchlean-hole.json
[artifact] Checked 2 leaves: ok=2, bad=0
```

Both leaves pass, yet neither covers a point whose first coordinate lies strictly between `0.0`
and `0.5`. We can check acceptance and the missing coverage side by side, using the same two
predicates the leaf checker uses:

```lean (name := tsCoverage)
/-- One entry of the artifact's `leaves` array, carrying
only the fields the checker reads. -/
structure ArtifactLeaf where
  lo : TorchLean.Tensor Float [2]
  hi : TorchLean.Tensor Float [2]
  lb : TorchLean.Tensor Float [1]
  thr : TorchLean.Tensor Float [1]
  idx : Nat

def holeRootLo : TorchLean.Tensor Float [2] := [-1.0, -1.0]
def holeRootHi : TorchLean.Tensor Float [2] := [1.0, 1.0]

def holeLeaves : List ArtifactLeaf :=
  [ { lo := [-1.0, -1.0], hi := [0.0, 1.0]
      lb := [0.5], thr := [0.0], idx := 0 },
    { lo := [0.5, -1.0], hi := [1.0, 1.0]
      lb := [1.25], thr := [0.0], idx := 0 } ]

-- Exactly what the checker asks of each leaf: nesting in
-- the root, and a witness that beats its threshold.
#eval holeLeaves.map fun L =>
  (boxWithin holeRootLo holeRootHi L.lo L.hi,
   refutesThresholdAt L.lb L.thr L.idx)

/-- Pointwise membership in a closed box. -/
def inBox (lo hi p : TorchLean.Tensor Float [2]) : Bool :=
  boundsOrdered lo p && boundsOrdered p hi

-- A point of the declared root that no leaf contains.
#eval
  let p : TorchLean.Tensor Float [2] := [0.25, 0.0]
  (inBox holeRootLo holeRootHi p,
   holeLeaves.any fun L => inBox L.lo L.hi p)
```
```leanOutput tsCoverage (whitespace := lax)
[(true, true), (true, true)]
```
```leanOutput tsCoverage (whitespace := lax)
(true, false)
```

Both leaves pass both checks, and `(0.25, 0.0)` is inside the root and inside no leaf. The checker
has checked containment and the witness comparisons, but the artifact contains no partition
evidence. The result `ok=2, bad=0` therefore leaves the property on $`[-1,1]^2` unresolved, even
if we grant the claimed bound on each leaf.

Root coverage requires a separate union-of-boxes check or evidence of a complete partition.
Split-tree data could supply that evidence directly. With only finite endpoints, coverage can also
be decided geometrically, but in several dimensions sorting by a single coordinate is insufficient.
The current leaf checker performs neither form of coverage check.

## Interval Coverage Checking

For these two leaves, a one-dimensional calculation suffices. Both span the full height of the
root, so
the only question is whether their first coordinates cover $`[-1,1]`. Sort the segments by left
endpoint and sweep:

```lean (name := tsSweep)
/-- Check whether closed segments cover `[lo, hi]`.
Reject nonfinite endpoints and reversed intervals. -/
def covers1D (lo hi : Float)
    (segs : List (Float × Float)) : Bool :=
  if !(lo.isFinite && hi.isFinite && decide (lo ≤ hi) &&
      segs.all fun (a, b) =>
        a.isFinite && b.isFinite && decide (a ≤ b)) then
    false
  else
    let sorted :=
      segs.mergeSort fun (a, _) (b, _) => decide (a ≤ b)
    let rec go (reach : Float) :
        List (Float × Float) → Bool
      | .nil => false
      | (a, b) :: rest =>
        if decide (b < reach) then go reach rest
        else if decide (a ≤ reach) then
          if decide (hi ≤ b) then true else go b rest
        else false
    go lo sorted

-- The first coordinate of each leaf of the hole example.
#eval covers1D (-1.0) 1.0
  (holeLeaves.map fun L =>
    (L.lo.getScalar 0, L.hi.getScalar 0))

-- The same two leaves with the split point recorded right.
#eval covers1D (-1.0) 1.0 [(-1.0, 0.0), (0.0, 1.0)]

-- Extra segments cannot undo coverage.
-- A singleton still needs a segment.
#eval (covers1D 0 1 [(0, 1), (2, 3)],
  covers1D 0 0 [], covers1D 0 0 [(-1, 1)])
```
```leanOutput tsSweep
false
```
```leanOutput tsSweep
true
```
```leanOutput tsSweep
(true, false, true)
```

After the input checks and sorting, `reach` starts at the target's left endpoint. Segments ending
before it cannot extend coverage and are skipped. The first segment that reaches it must also
start at or before it; otherwise there is a gap. Each accepted segment then extends the covered
prefix, and reaching `hi` completes the check immediately. Returning `false` on an empty list
also ensures that a singleton target needs a segment containing its point.

The first run finds the missing strip, while the corrected split covers the interval. The final
three cases check coverage with an extra disjoint segment and both outcomes for a singleton.
These executable checks illustrate the sweep; a verified coverage checker also needs a theorem
relating its Boolean result to membership in the union of the closed intervals.

The tuple output follows the order of the final three calls. The first is `true` because
`[0, 1]` already covers the target; the extra `[2, 3]` does not undo that fact. The second is
`false` because an empty collection covers no point, including the singleton target `{0}`.
The third is `true` because `[-1, 1]` contains that singleton. Closed endpoints matter at an
ordinary split too: `[-1, 0]` and `[0, 1]` both contain the cut at zero, leaving no uncovered
boundary between them.

For boxes split along several axes, even a correct interval checker cannot decide coverage from
one projection. A multidimensional sweep has to track coverage of cross-sections, involving
$`d-1` dimensional geometry. A split tree offers a different approach: verify each split and show
that every branch ends in a represented leaf.

Finite floating-point endpoints represent exact dyadic coordinates. An exact union test can
therefore detect a real gap between two endpoints even if it is only one ULP wide; that width
depends
on the endpoints' magnitude and is not uniformly $`2^{-24}`. Such a gap cannot be dismissed in a
universal robustness claim. Recording split coordinates and values in a tree would give the checker
a direct partition witness and avoid reconstructing the union geometrically. Neither form of
coverage evidence is checked by the current leaf workflow.

# Artifact Rejection Tests

The {ref "certificates"}[certificate chapter] evaluates the checker's predicates one at a time
inside Lean. Here we do the coarser thing a continuous-integration job does: run the process and
read its exit status. Start from the good artifact above and change one field.

Push a leaf coordinate outside the root, `hi[1] = 2.0`:

```terminal +output
$ lake exe verify -- abcrown-leaf /tmp/torchlean-escaped.json
[artifact] leaf 0 rejected: box escapes the root region
[artifact] Checked 1 leaves: ok=0, bad=1
error: Artifact failed checks for 1 leaves
$ echo $?
1
```

Leave the arithmetic alone and inflate only the bookkeeping field, `witness_margin = 5.0` where the
real margin is `1.0`:

```terminal +output
$ lake exe verify -- abcrown-leaf /tmp/torchlean-margin.json
[artifact] leaf 0 rejected: witness_margin 5.000000 disagrees with lb - threshold = 1.000000
[artifact] Checked 1 leaves: ok=0, bad=1
error: Artifact failed checks for 1 leaves
$ echo $?
1
```

Point the witness past the end of the bound vector, `witness_idx = 3` in a one-element `lb`:

```terminal +output
$ lake exe verify -- abcrown-leaf /tmp/torchlean-idx.json
[artifact] leaf 0 rejected: witness index 3 does not satisfy lb > threshold; witness_margin
  1.000000 names no witness index in range
[artifact] Checked 1 leaves: ok=0, bad=1
error: Artifact failed checks for 1 leaves
$ echo $?
1
```

The diagnostics distinguish containment, prune, and margin failures. An out-of-range index also
prevents margin recomputation, so that edit can fail more than one check. The CLI prints the leaf
counts before raising an error, giving the caller both a failing exit status and a location to
inspect.

The margin-only edit illustrates why a redundant field can be useful. The lower bound still
exceeds its threshold, but the recorded subtraction disagrees with those arrays. That detects an
inconsistent artifact without asserting that the underlying network is unsafe.

The exit statuses distinguish process completion from acceptance. A caller using these examples
in a script should read the nonzero status as failure of the requested artifact check, then use
the diagnostic to identify the field. Status `1` here comes from the Lean checker rejecting
the edited artifact; status `2` in the earlier example comes from the exporter finding no
positive witness. Neither status is a counterexample to the underlying network property.
They identify failures at different points in this data path.

# Neural Controllers

The leaf workflow exports and checks terminal-domain data. The Lyapunov workflows add training
and bound propagation, with different choices about where those computations run. They learn
a Lyapunov function and a controller together, then seek a certificate
for the pair, following {Informal.citet neurallyapunov2019}[].

A controller adds a time-dependent meaning to the output. Its action changes the system's next
state, and a candidate Lyapunov function assigns a scalar value to each state. To reason about
decay, we compare that value with its derivative along the system's dynamics. The network may
represent the controller, the candidate function, or both; the desired inequality is about the
closed-loop system assembled from them.

This is related to the classifier margin argument. In both cases, a universal statement about
many inputs is reduced to signs of sound bounds. A classifier needs the chosen score difference
to remain positive. A Lyapunov argument needs appropriate positivity and decrease conditions,
along with the other hypotheses of the stability result being used. The sign calculation can
be short once the bounds are available; establishing what those bounds refer to is the harder
part.

- {src "NN/MLTheory/CROWN/Lyapunov/TwoStage/PipelineIPythonOnly.lean"}[
  `twostage-pythononly-certgen`] calls the external CROWN producer and writes a Lean module. The
  generated numbers enter as a `LyapunovCert`, and Lean proves their sign conditions. The theorem
  transferring those signs to the Lyapunov functions still assumes `LyapunovCert.ValidFor`, because
  this path does not
  replay the external verifier. Running it needs a `.pth` checkpoint and the producer's Python
  environment, so it is the one runner this chapter does not execute.
- {src "NN/MLTheory/CROWN/Lyapunov/TwoStage/PipelineIIHybrid.lean"}[`twostage-hybrid-van-stage2`]
  treats PyTorch as an untrusted initializer. Stage one trains in float32 and exports the
  parameters as raw bit patterns; stage two loads those bits into FloatLib binary32, refines, and
  runs
  the IBP and CROWN box check inside Lean.
- {src "NN/MLTheory/CROWN/Lyapunov/TwoStage/PipelineIIIAllInLean.lean"}[
  `twostage-torchlean-cegis-van`] does initialization, sampled training, PGD-style candidate
  search, refinement, and the final bound check in Lean, with no external stage one at all.

These runner transcripts predate the FloatLib migration; their source and toolchain identities
are recorded below. The all-in-Lean runner takes no arguments:

```terminal +output
$ lake exe verify -- twostage-torchlean-cegis-van
== TwoStage TorchLean CEGIS workflow (IEEE32Exec) ==
width=100 stage1Steps=1 stage2Rounds=1 pgdSteps=1
[stage1] step 0: loss=0.020687
[stage2] round 0: loss=0.144132
Stage 2 check: IBP + CROWN on the scalar loss over a small box
lowered IR nodes: 67
[IBP] scalar loss box dim=1
[CROWN] loss lo = [-0.000000]
[CROWN] loss hi = [241.477783]
```

The hybrid runner re-exports its stage-one weights when asked:

```terminal +output
$ lake exe verify -- twostage-hybrid-van-stage2 --stage1
[stage1] running PyTorch exporter (width=500 steps=10) → _external/van_stage1_w500_bits.json
[stage1] step=0 loss=0.102408
[stage1] step=5 loss=0.490164
Wrote _external/van_stage1_w500_bits.json (width=500)
== TwoStage Hybrid workflow: Stage1=PyTorch (bits), Stage2=TorchLean (IEEE32Exec) ==
weights=_external/van_stage1_w500_bits.json width=500 stage2Rounds=1 candidates=1 pgdSteps=1
[stage2] round 0: lossBefore=0.009320 lossAfterPGD=0.009654
[stage2] PGD counterexample candidates=1 (positive-loss=1)
Stage 2 check: IBP + CROWN on the scalar loss over a small box
lowered IR nodes: 67
[IBP] scalar loss box dim=1
[CROWN] loss lo = [-0.000000]
[CROWN] loss hi = [1017.059021]
```

Both commands exit successfully, meaning their workflows completed. Their reported bounds do
not certify the requested Lyapunov conditions.

The objective adds two ReLU hinge penalties: one for the lower bound on the Lyapunov value,
and one for its required decrease along the dynamics. Each penalty is the positive part of a
constraint violation. In the exact real interpretation, their nonnegative sum is zero precisely
when both encoded inequalities hold at that point. To establish those conditions throughout a box,
one needs a sound upper bound of zero on the real penalty. A rounded zero alone would need an
arithmetic argument as well. The printed lower bounds near zero do not establish this, and the
upper bounds of about 241 and 1017 leave the question unresolved.

Read the transcript in that order. The stage-one and stage-two losses are sampled training or
search measurements. `lowered IR nodes: 67` identifies the size of the resulting operation
graph, not the number of scalar weights. `scalar loss box dim=1` says the verifier is bounding
one penalty value. The final interval then bounds that penalty according to the selected pass.
A lower endpoint near zero is unsurprising for a sum of nonnegative hinges; the useful
certificate would need control of the upper endpoint throughout the region.

These runs demonstrate lowering and in-repository IBP and CROWN bound computation
{Informal.citep gowal2018}[]{Informal.citep crown2018}[]. An upper bound above zero does not by
itself provide a violating point or a stability certificate.

The hybrid log's “counterexample candidates” are search outputs to inspect. A candidate becomes
evidence of a violated condition only after evaluating the intended property with the required
semantics and confirming the violation. The positive-loss count reports the selected executable
penalty at the sampled candidates. It does not explain the upper bound on all other states,
and it is not a proof about trajectories of the dynamical system.

The short stage counts in these transcripts also belong to their interpretation. They exercise
the workflow with the printed initialization and refinement settings. The outputs do not claim
that training converged or that a longer run would obtain a useful bound. The remaining question
is concrete: can the selected model and verifier produce an upper bound small enough to establish
the encoded condition, together with the semantic and arithmetic evidence needed to trust it?

The two transcripts also record different model widths:

:::table +header
*
  * runner
  * hidden width
  * IR nodes
  * CROWN loss upper bound
*
  * `twostage-torchlean-cegis-van`
  * 100
  * 67
  * 241.48
*
  * `twostage-hybrid-van-stage2`
  * 500
  * 67
  * 1017.06
:::

The runs differ in weights, initialization, training, and width, so their upper bounds do not
isolate the effect of any one change. Without a matching lower estimate of the true maximum, the
numbers alone also do not tell us how loose the bounds are. Branching and refined relaxations, as
in α,β-CROWN {Informal.citep betacrown2021}[], can improve inconclusive bounds; `auto_LiRPA`
{Informal.citep autolirpa2020}[] supplies bound propagation across more general graphs. A semantic
claim still requires the checker-to-model theorem and its arithmetic assumptions.

## Float32 Parameter Bit Patterns

To check the parameters produced by stage one, stage two must load the same values. The export
records float32 bit patterns as decimal-encoded `uint32` values:

```
{
  "format": "uint32-bits-as-decimal-strings",
  "dtype": "float32",
  "width": 500,
  "wC": ["3156207770", "1061300877"],
  ...
}
```

In Python those two integers decode with `struct`:

```
>>> # Reinterpret each stored 32-bit pattern as a float,
>>> # preserving its encoding.
>>> import struct
>>> [struct.unpack('<f', struct.pack('<I', b))[0]
...  for b in (3156207770, 1061300877)]
[-0.009760046377778053, 0.7584617733955383]
```

The Lean side consumes the same integers through `ExecFloat.Binary.ofBits32`. This preserves the
stored encoding without parsing a decimal approximation:

```lean (name := tsBits)
-- Decode the exact stored bit patterns; printing toFloat
-- rounds only their display.
#eval (ExecFloat.Binary.toFloat32
  (ExecFloat.Binary.ofBits32 3156207770)).toFloat
#eval (ExecFloat.Binary.toFloat32
  (ExecFloat.Binary.ofBits32 1061300877)).toFloat
```

```leanOutput tsBits (whitespace := lax)
-0.009760
```

```leanOutput tsBits (whitespace := lax)
0.758462
```

The displayed decimals agree to the shown precision, and `ofBits` preserves the supplied 32-bit
encoding without a decimal-to-float conversion. This establishes how parameter data enter stage
two. Finiteness, model identity, and the arithmetic used after loading remain separate obligations.
A decimal export could also be exact if it used a round-trip representation; the raw-bit format
makes that representation choice explicit. The {ref "fp32-soundness"}[float32 soundness chapter]
explains the further connection between the bit-level and rounded-real models.

# Coordinate Selection In Verifier Lowering

An earlier lowering path rejected the coordinate selection needed by both runners. Its error
was:

```terminal +output
Stage 2 check: IBP + CROWN on the scalar loss over a small box
error: TorchLean→IR: select is outside the verifier IR fragment
```

The stage-two loss reads coordinates out of the state vector, `x₁` and `x₂` for the van der Pol
dynamics, with `Model.select`. The lowering in
{src "NN/Verification/Builtin/Lowering/Builder.lean"}[`Lowering/Builder.lean`] refused it, grouped
with `indexSelect` and `scatterAdd` under the heading of gather and scatter operations that the
verifier fragment does not accept.

The current builder handles `select` on the leading axis by reading the supplied `Fin` index
while constructing the graph. The index is fixed for that graph; having type `Fin` alone does not
mean a value must be known at elaboration time. Selecting one coordinate becomes a constant one-hot
projection: take a leading slice of length one, then reshape to remove that axis.

`indexSelect` and `scatterAdd` take tensor-valued indices and remain outside this lowering fragment.
That is a limit of the implemented translation and transfer rules, not a claim that indexed
operations cannot be bounded. Selection on an inner axis is also rejected by this builder. The
leading-axis case is sufficient for the state coordinates used by these two loss programs.

# Lyapunov Certificate Validity

The pipeline that generates a Lean module hands Lean a `LyapunovCert`: four numbers, plus a region.

The finite numbers cannot, on their own, imply anything about a real-valued dynamical system. The
missing ingredient has a name in TorchLean, and it is a `Prop`:

```lean (name := tsValidFor)
-- ValidFor is the proposition that these bounds enclose
-- these particular functions.
#check @LyapunovCert.ValidFor
-- Converting the numeric record constructs certificate data
-- without proving validity.
#check @RealCert.toCert
```

```leanOutput tsValidFor (whitespace := lax)
@LyapunovCert.ValidFor : {α : Type} →
  [inst : Storage α] → [inst_1 : Context α] → {n : ℕ} → LyapunovCert α n → NeuralLyapunov α n → Prop
```

```leanOutput tsValidFor (whitespace := lax)
@RealCert.toCert : {n : ℕ} → RealCert n → LyapunovCert ℝ n
```

`ValidFor cert lyap` states enclosure of the supplied Lyapunov value and orbital-derivative
functions throughout the region. The four numeric fields alone cannot establish it. A producer
must supply checkable evidence for that proposition, or the generated theorem must retain it as a
hypothesis.

In the first output, `LyapunovCert α n → NeuralLyapunov α n → Prop` says that `ValidFor` takes
certificate data and a pair of functions and produces a proposition. That proposition needs a
proof. The second output ends in `LyapunovCert ℝ n`, a data type: `toCert` repackages the real
endpoints into the canonical record. The types already expose the distinction between
constructing bounds and proving that they apply to the supplied functions.

For a concrete instance, take the one-dimensional candidate $`V(x)=x^2` on $`[1,2]`, with dynamics
$`x'=-x` and orbital derivative $`-2x^2`. `RealCert`
carries the numbers, and `RealCert.toCert` builds the boxed region from them:

```lean (name := tsCertDefs)
-- Proposed ranges for x squared and minus twice x squared
-- on the interval [1, 2].
noncomputable def tsRealCert : RealCert 1 :=
  { vLower := 1
    vUpper := 4
    derivativeLower := -8
    derivativeUpper := -2
    regionLower := fun _ => 1
    regionUpper := fun _ => 2 }

noncomputable def tsCert : LyapunovCert ℝ 1 :=
  tsRealCert.toCert

noncomputable def tsLyap : NeuralLyapunov ℝ 1 :=
  { value := fun x => (x.unstack 0).item ^ 2
    orbitalDerivative := fun x =>
      -2 * (x.unstack 0).item ^ 2 }
```

The example can establish that evidence directly. The enclosure has two halves: on the region, `V`
lies in $`[1,4]`, and its orbital derivative lies in $`[-8,-2]`:

```lean (name := tsValidProof)
-- Prove both enclosure fields from membership in the stated
-- one-dimensional region.
theorem tsValid : tsCert.ValidFor tsLyap := by
  constructor
  -- The candidate value lies between its stored lower and
  -- upper bounds.
  · intro x hx
    have h := hx 0
    simp [Box.contains, tsCert, tsRealCert,
      RealCert.toCert] at h
    obtain ⟨hLower, hUpper⟩ := h
    refine ⟨?_, ?_⟩
    · simp only [tsLyap, tsCert, tsRealCert,
        RealCert.toCert]
      nlinarith [sq_nonneg ((x.unstack 0).item)]
    · simp only [tsLyap, tsCert, tsRealCert,
        RealCert.toCert]
      nlinarith [hLower, hUpper]
  -- The supplied orbital derivative lies between its stored
  -- bounds as well.
  · intro x hx
    have h := hx 0
    simp [Box.contains, tsCert, tsRealCert,
      RealCert.toCert] at h
    obtain ⟨hLower, hUpper⟩ := h
    constructor <;>
      simp only [tsLyap, tsCert, tsRealCert,
        RealCert.toCert] <;>
      nlinarith [sq_nonneg ((x.unstack 0).item)]
```

The region gives $`1\le x\le2`, hence $`1\le x^2\le4` and
$`-8\le-2x^2\le-2`. The supplied orbital derivative is consistent with the scalar dynamics
$`x'=-x`: the derivative of $`V(x)=x^2` is $`2x`, and $`(2x)(-x)=-2x^2`.
`NeuralLyapunov` itself stores the value and orbital-derivative functions independently, so a
controller application must establish that connection as well. This example proves sign and
range conditions on $`[1,2]`; it does not prove equilibrium stability or invariance of that region.

With the proof in hand, the analytic conclusion follows from the theorem that consumes it:

```lean (name := tsConditionsProof)
-- Combine the proved enclosures with the strict signs of
-- their endpoints.
theorem tsConditions :
    (∀ x, tsCert.region.contains x →
      tsLyap.value x > 0) ∧
    (∀ x, tsCert.region.contains x →
      tsLyap.orbitalDerivative x < 0) :=
  Real.lyapunov_conditions tsLyap tsCert tsValid
    (by norm_num [tsCert, tsRealCert, RealCert.toCert])
    (by norm_num [tsCert, tsRealCert, RealCert.toCert])
```

The two `norm_num` calls are the sign conditions on the numbers, `vLower > 0` and
`derivativeUpper < 0`. In the numeric-only generated-module path, these are the obligations that
can be discharged from the imported numbers. Spelling the theorem out at $`n=1` names the three
hypotheses; the
`example` compiles only if the real theorem has exactly this shape:

```lean (name := tsLyapThm)
-- State the same implication with validity and endpoint
-- signs as explicit hypotheses.
example (lyap : NeuralLyapunov ℝ 1)
    (cert : LyapunovCert ℝ 1)
    (enclosure : cert.ValidFor lyap)
    (positiveValue : cert.vLower > 0)
    (negativeDerivative : cert.derivativeUpper < 0) :
    (∀ x : Tensor ℝ [1], cert.region.contains x →
        lyap.value x > 0) ∧
    (∀ x : Tensor ℝ [1], cert.region.contains x →
        lyap.orbitalDerivative x < 0) :=
  Real.lyapunov_conditions lyap cert enclosure
    positiveValue negativeDerivative
```

The hypothesis `enclosure` supplies `ValidFor` on the region. The next two hypotheses compare
real certificate endpoints with zero. A generated module that leaves `ValidFor` as a
hypothesis proves a conditional sign result. It does not establish that the external verifier's
bounds apply to the intended dynamical system.

The proof `tsValid` shows what discharging that hypothesis looks like in a small case.
Membership in the one-dimensional region gives `1 ≤ x ≤ 2`. The first branch derives the
range of `x²`; the second derives the range after multiplication by `-2`, which reverses the
endpoint order. `tsConditions` then uses the lower value bound `1 > 0` and the upper derivative
bound `-2 < 0`. Its short proof is possible because the enclosure argument has already been
done. The code blocks defining these theorems produce no numeric output: successful checking
means Lean accepted the proof terms for their stated propositions.

In a controller workflow the producer searches for a policy $`u_\theta` and a candidate $`V`, aiming
at

$$`V(x)\geq 0,\qquad
\nabla V(x)\cdot f(x,u_\theta(x))
\leq-\alpha\|x\|^2.`

Exporting terminal boxes with positive numeric margins does not prove those inequalities. To close
the gap, an artifact would have to identify the model and parameter hash, the state-space root
region, the exact dynamics and scalar semantics, every partition leaf, replayable bounds for `V`
and its Lie derivative, and the coverage and boundary conditions. TorchLean's other `RealCert`-style
and IR-based checkers show stronger replay patterns, but each command should be read on its own
terms rather than credited by association with the name of an external solver.

# Recording A Run

A useful run record pins down everything the numbers depend on. The transcripts above were
recorded before the FloatLib migration, with the following identities. Their recorded toolchain
remains 4.33.0; the current checkout requires 4.34.0 and pinned FloatLib. Rebuilding the current
examples is a separate validation from preserving this record:

:::table +header
*
  * field
  * value
*
  * TorchLean commit
  * `12f5c65`
*
  * Lean toolchain
  * `leanprover/lean4:v4.33.0`
*
  * producer
  * bundled raw dump, no external solver invoked
*
  * raw dump
  * `NN/Examples/Verification/AbCrown/example_raw_leaf_dump.json`
*
  * artifact
  * `/tmp/torchlean-abcrown-artifact.json`
*
  * checker
  * `lake exe verify -- abcrown-leaf <artifact>`
*
  * output
  * `[artifact] Checked 1 leaves: ok=1, bad=0`
:::

For a real α,β-CROWN run the same table would also carry the verifier repository and commit, the
Python and solver versions, the model checkpoint hash, the property file, and the dtype, device, and
numerical flags. A lower-bound vector has no stable meaning once
the model, the property, or the output-margin convention has changed, and none of those changes are
visible in the artifact.

# Workflow Limits And Proof Obligations

The leaf workflow checks that exported data satisfy its structural contract. Its current limits
are concrete:

- *No coverage.* Leaves need not exhaust the root, as the wide-root run above shows.
- *No replay.* The lower bounds are read, not recomputed against the network.
- *No soundness lemma for the predicates.* `boxWithin`, `refutesThreshold`, and
  `refutesThresholdAt` in {src "NN/Verification/Util/Tensor.lean"}[`Util/Tensor.lean`] are `Bool`
  functions with no theorem tying them to a `Prop`-level statement about box inclusion or margin
  refutation. They compute the right comparisons, and the {ref "certificates"}[certificate chapter]
  evaluates them on both sides of every boundary, but a reader who wants a proof rather than a test
  will not find one yet.

The Lyapunov runners additionally compute bounds on a graph lowered from the trained TorchLean
program. The reported upper bounds do not settle the penalty condition. A completed controller
argument needs useful bounds, coverage of the intended region, and proofs connecting the selected
checker and arithmetic to the Lyapunov hypotheses.
