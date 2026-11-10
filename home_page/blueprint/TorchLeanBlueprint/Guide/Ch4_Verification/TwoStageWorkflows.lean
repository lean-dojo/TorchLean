import VersoManual
import FloatLib
import TorchLeanExamples.Verification.Lyapunov.SignConditions
import TorchLean.Tensor.Einops.Elab.Einsum
import TorchLean.Tensor.Einops.Elab.Pack
import TorchLean.Tensor.Literal
import TorchLean.Tensor.Einops.Elab.Transform
import TorchLean.Tensor.Tactic.Einops.Report
import TorchLean.Tensor.LinearAlgebra
-- The coverage experiment below calls the same array predicates the leaf checker calls, so it
-- imports them rather than restating them. If one of those predicates changes meaning, this
-- chapter's output changes with it.
import TorchLean.Verify.Tensor
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles
import TorchLean.Tensor.Constructors

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open CROWN
open Examples.Lyapunov
-- Tensor predicates for the coverage experiment.
open Tensor
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

#doc (Manual) "Two-Stage Verification" =>
%%%
tag := "twostage"
file := "Two-Stage-Verification-Workflows"
%%%

A two-stage workflow lets an external verifier search for bounds, then passes evidence to Lean for
checking. Its producer may split boxes, optimize relaxations, and use GPU kernels, while the
checker needs enough evidence to justify the result without reproducing every search decision.
Which conclusion it can justify depends on what the artifact records.

Return to a classifier whose label should stay fixed in an input square. A producer may spend many
passes trying to lower-bound the difference between two scores (adjusting an affine relaxation,
splitting the square, or choosing the next region to examine). A consumer receiving the final
result needs a more stable description: which model and square were checked, what bound is claimed
on each part, and what evidence justifies those bounds. Repeating every failed search attempt would
add work without helping establish the final claim.

Keeping the two stages apart is useful even when they run on the same machine. Search code is
organized around finding useful evidence, while checking code is organized around the conditions
that make that evidence valid. So a change to the search heuristic can leave those conditions
unchanged. But changing the model, input region, or interpretation of the scores changes the claim,
even if the JSON still has the same field names.

A producer is allowed to be complicated:

```
trained model + input property
  -> external verifier
  -> branch-and-bound leaves and claimed lower bounds
```

A consumer should be narrow:

```
JSON artifact
  -> finite parser
  -> schema and local predicate checks
  -> accept or reject
```

This producer/checker separation follows the proof-carrying-code approach of
{Informal.citet necula1997}[]. TorchLean's paths implement different parts of that idea: the leaf
workflow checks a summary of terminal domains, while the controller workflows also run bound
propagation. The examples below follow the data from producer to consumer and identify the evidence
available at each step.

# External Search And Certificate Checking

A search procedure chooses where to spend computation. α,β-CROWN's branch-and-bound method
{Informal.citep betacrown2021}[], for example, refines relaxations and splits domains until it can
prune them or must continue searching. A sound certificate checker would allow those search choices
to remain untrusted: any accepted evidence would have to imply the property regardless of how it
was found.

For a leaf, the semantic claim is that a margin bound holds at every point of its sub-box. That's a
quantified statement about a fixed network, even though a finite artifact may carry evidence for
it. Checking only the exported endpoints and margins doesn't establish that statement.

TorchLean's leaf format checks a narrower set of facts (dimensions, containment, root coverage,
witness indices, and comparisons among exported numbers). So it can detect inconsistent fields. But
a consistently misapplied sign or axis convention can still pass, because the artifact has no
network or transfer rules against which to check that convention.

*Checked by Lean* can mean anything from parsing a JSON file to replaying every bound computation
and deriving a theorem, and TorchLean's leaf checker sits near the parsing end. Where the
{ref "certificates"}[certificate chapter] derives its predicates line by line, this chapter starts
one step earlier (at the producer) and follows the data outward.

# Producer Artifact Export

Terminal-domain data from the external verifier must be converted to TorchLean's schema. That's the
job of the adapter
{src "scripts/verification/abcrown/export_leaf_artifact.py"}[`export_leaf_artifact.py`], which
accepts the field names external tools actually use, normalizes them, computes a witness
coordinate, and writes the checked format.

Our bundled raw dump uses α,β-CROWN's vocabulary, not TorchLean's:

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
  --input TorchLeanExamples/Verification/AbCROWN/example_raw_leaf_dump.json \
  --out /tmp/torchlean-abcrown-artifact.json \
  --check
```

Passing `--check` also runs the leaf checker (`scripts/lake.sh exe verify -- abcrown-leaf`) on the
file the exporter just wrote. Here's the exported artifact:

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
      "witness_idx": 0
    }
  ],
  "root": {
    "hi": [1.0, 1.0],
    "lo": [-1.0, -1.0]
  }
}
```

On disk, the file is taller than the listing above (which is reflowed to fit the page), because the
writer sorts keys and puts one array element per line. Field names and their order are the real
ones. Sorting the keys isn't a stylistic preference: two runs of the exporter on the same input
produce byte-identical files, which makes the artifact diffable in review and hashable in a report.

First, the adapter normalizes field names. A leaf's lower box may arrive as `lo`, `input_lo`,
`x_L`, `x_l`, `domain_lo` or `lower`, its upper box as `hi`, `input_hi`, `x_U`, `x_u`, `domain_hi`
or `upper`, its lower bounds as `lb`, `lower_bound`, `lower_bounds`, `output_lb` or `margin_lb`,
and its threshold as `threshold`, `thresholds`, `rhs` or `unsafe_threshold`, with a top-level
default for leaves that give none. The domain list and the root box have spellings of their own. A
record that lacks a field it needs raises `ArtifactExportError`, and so does a record that gives
two spellings of one field different values, with an error naming both keys. Two spellings with
equal values are read as one. The adapter doesn't check that a producer used a field with the
meaning its name suggests. Lean's parser receives the normalized spelling.

*The witness is computed, not trusted.* The adapter picks the coordinate with the largest positive
margin `lb[i] - threshold[i]` and records its index. It leaves out the optional `witness_margin`,
so the checker recomputes the margin exactly from the file's decimals. When the index is present,
the checker validates that coordinate. If the index is omitted (the schema permits that too), the
checker searches for a positive margin. Neither route validates the underlying network bound.

A `witness_margin` added by hand must come with a witness index, and the checker compares it with
the recomputed margin using an absolute tolerance of `1e-6`. That's bookkeeping agreement, not
exact equality. But the strict `lb[i] > threshold[i]` test itself uses no tolerance.

For the bundled domain, the witness calculation is `1.0 - 0.0 = 1.0`, so index zero is the only
possible choice. `input_dim = 2` comes from the two endpoint coordinates, while the bound and
threshold arrays have length one. So the domain is two-dimensional and carries one property
expression. From the supplied data, the converter can establish those lengths and arithmetic
relationships, but it can't infer what network expression the producer placed in that one-element
bound vector.

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

Save that input as `/tmp/torchlean-raw-nowitness.json`, then run:

```terminal
python3 scripts/verification/abcrown/export_leaf_artifact.py \
  --input /tmp/torchlean-raw-nowitness.json \
  --out /tmp/torchlean-nowitness.json
```

This time the exporter fails before writing the requested artifact. Leaves in this format carry
positive witnesses, so a domain without one can't be encoded in it. For a root-region claim, you
can't repair such a failure by dropping that domain: its region still needs coverage and a valid
bound.

To instrument a verifier in place, use the function the same script exposes:

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

Place the call after the search has collected its terminal leaves. TorchLean doesn't vendor
α,β-CROWN or the neural-controller repositories: their Python, CUDA, and solver dependencies stay
in their own environments, and the core Lean build needs only the exported JSON.

# Root Boxes And Leaf Coverage

Root endpoints identify the property region. If you omit them, the adapter constructs an envelope
from the leaves, which changes what the containment check refers to.

Here's a two-leaf dump that splits the unit square down the middle, with no root recorded:

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

Save it as `/tmp/torchlean-raw-2leaf.json`:

```terminal
python3 scripts/verification/abcrown/export_leaf_artifact.py \
  --input /tmp/torchlean-raw-2leaf.json \
  --out /tmp/torchlean-2leaf.json --check
```

From the leaves, the adapter derives this root:

```
"root": { "lo": [-1.0, -1.0], "hi": [1.0, 1.0] }
```

Each leaf lies inside this envelope by construction. So the check establishes containment in that
derived box, but the box supplies no independent evidence of the original property region. If the
claim is about that region, a run record must retain the original root.

Supply a root twice as large as the region the leaves actually cover:

```terminal
python3 scripts/verification/abcrown/export_leaf_artifact.py \
  --input /tmp/torchlean-raw-2leaf.json \
  --out /tmp/torchlean-2leaf-wide.json \
  --root-lo=-2,-2 --root-hi=2,2 --check
```

Together, the leaves cover $`[-1,1]^2`. The declared property region is $`[-2,2]^2`. Every leaf
passes containment, but the checker rejects the artifact because the leaves don't cover the root.
Coverage is a union condition over all leaves, and the checker uses the supplied endpoints to check
it without requiring the producer's search tree.

An accepted `abcrown-leaf` artifact passes checks that *each listed leaf is a well-formed sub-box
of the declared root, the leaves cover that root, and each carries a witness that passes its own
arithmetic test.* But the lower bounds themselves remain the producer's claims: coverage doesn't
establish their validity for the network.

This wide-root example separates two questions that often get compressed into “the boxes look
right.” Each leaf is a valid subdomain of the declared root. But a point such as `(1.5, 0.0)`
belongs to that root and to neither represented leaf, and even perfect network bounds on the leaves
would say nothing about that point. A root-region theorem must be able to take an arbitrary root
input, locate a leaf containing it, and apply the leaf's bound. Coverage evidence supplies that
middle step.

Coverage can also fail when the root is correct. This next dump records the intended root
$`[-1,1]^2` and two leaves, leaving a strip in the middle that neither leaf covers (the kind of gap
a lost branch-and-bound subtree could leave).

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

Save the dump as `/tmp/torchlean-raw-hole.json`:

```terminal
python3 scripts/verification/abcrown/export_leaf_artifact.py \
  --input /tmp/torchlean-raw-hole.json \
  --out /tmp/torchlean-hole.json --check
```

Both leaves pass the local containment and witness checks, yet neither covers a point whose first
coordinate lies strictly between `0.0` and `0.5`. So the CLI's separate coverage check rejects this
artifact. Using the same two local predicates the leaf checker uses, we can inspect why local
checks alone are insufficient:

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

Both leaves pass both local checks, and `(0.25, 0.0)` is inside the root and inside no leaf. Our
experiment stops before the CLI's coverage check. It shows why passing the local checks alone
leaves the property on $`[-1,1]^2` unresolved, even if we grant the claimed bound on each leaf.

Root coverage requires a separate union-of-boxes check or evidence of a complete partition.
`leavesCoverRoot` collects the sorted distinct endpoints on each axis, as exact rationals, and
checks the resulting closed grid cells: every cell must lie in at least one leaf (axes on which
every leaf spans the root need no subdivision). It refuses a grid exceeding one million cells, so a
large artifact can require another representation or checking strategy. In several dimensions,
sorting by a single coordinate would be insufficient.

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
before it can't extend coverage and are skipped. The first segment that reaches it must also start
at or before it, or else there's a gap. Each accepted segment then extends the covered prefix, and
reaching `hi` completes the check immediately. Returning `false` on an empty list also ensures that
a singleton target needs a segment containing its point.

The first run finds the missing strip, while the corrected split covers the interval. In the final
three cases, we check coverage with an extra disjoint segment and both outcomes for a singleton.
These executable checks illustrate the sweep, but a verified coverage checker also needs a theorem
relating its Boolean result to membership in the union of the closed intervals.

Closed endpoints matter at a split: `[-1, 0]` and `[0, 1]` both contain the cut at zero, leaving
no uncovered boundary between them.

For boxes split along several axes, even a correct interval checker can't decide coverage from one
projection. A multidimensional sweep has to track coverage of cross-sections (which involves $`d-1`
dimensional geometry). A split tree offers a different approach: verify each split and show that
every branch ends in a represented leaf.

The checker reads each endpoint as the exact rational its decimal denotes, so the union test cannot
miss a gap, however narrow. The decimals `0.1` and `0.10000000000000001` round to the same binary64
value, but leaves `[0, 0.1]` and `[0.10000000000000001, 1]` miss the points between them. Such a
gap can't be dismissed in a universal robustness claim. Recording split coordinates and values in a
tree could instead give the checker a direct partition witness and avoid reconstructing the union
geometrically, but TorchLean's leaf workflow uses the endpoint grid and doesn't consume such a
tree.

# Artifact Rejection Tests

In the {ref "certificates"}[certificate chapter], we evaluate the checker's predicates one at a
time inside Lean. To exercise the command-line path as well, start from the good artifact above,
change or add one field in a separate copy, and check that the process rejects it.

Push a leaf coordinate outside the root (`hi[1] = 2.0`) and save the copy as
`/tmp/torchlean-escaped.json`:

```terminal
scripts/lake.sh exe verify -- abcrown-leaf /tmp/torchlean-escaped.json
```

Leave the arithmetic alone and add only a wrong bookkeeping field (`witness_margin = 5.0` where the
real margin is `1.0`). Save this copy as `/tmp/torchlean-margin.json`:

```terminal
scripts/lake.sh exe verify -- abcrown-leaf /tmp/torchlean-margin.json
```

Point the witness past the end of the bound vector (`witness_idx = 3` in a one-element `lb`). Save
this copy as `/tmp/torchlean-idx.json`:

```terminal
scripts/lake.sh exe verify -- abcrown-leaf /tmp/torchlean-idx.json
```

The diagnostics distinguish containment, prune, and margin failures. With a `witness_margin`
present, an out-of-range index also leaves no margin to recompute, so that edit fails two checks.
Before raising an error, the CLI prints the leaf counts, which gives the caller both a failing exit
status and a location to inspect.

Inflating only the margin shows why a redundant field can be useful. The lower bound still exceeds
its threshold, but the recorded subtraction disagrees with those arrays, so the checker detects an
inconsistent artifact without asserting that the underlying network is unsafe.

If you use these examples in a script, treat a nonzero status as failure of the requested artifact
check, then use the diagnostic to identify the field. Rejection by the exporter or checker
identifies inconsistent or insufficient artifact data, but it doesn't provide a counterexample to
the underlying network property.

# Neural Controllers

The leaf workflow exports and checks terminal-domain data, while the Lyapunov workflow adds
training and bound propagation, all in Lean. Following {Informal.citet neurallyapunov2019}[], it
learns a Lyapunov function and a controller together, then seeks a certificate for the pair.

A controller adds a time-dependent meaning to the output. Its action changes the system's next
state, and a candidate Lyapunov function assigns a scalar value to each state. To reason about
decay, we compare that value with its derivative along the system's dynamics. The network may
represent the controller, the candidate function, or both, but the desired inequality is about the
closed-loop system assembled from them.

This is related to the classifier margin argument: in both cases, a universal statement about many
inputs is reduced to signs of sound bounds. A classifier needs the chosen score difference to
remain positive, while a Lyapunov argument needs appropriate positivity and decrease conditions,
along with the other hypotheses of the stability result being used. Once the bounds are available,
the sign calculation can be short. Establishing what those bounds refer to is the harder part.

- {src "TorchLeanExamples/Verification/Lyapunov/AllInLean.lean"}[
  `twostage-torchlean-cegis-van`] does initialization, sampled training, PGD-style candidate
  search, refinement, and the final bound check in Lean, with no external stage one at all.

Run the all-in-Lean workflow with its default settings:

```terminal
scripts/lake.sh exe verify -- twostage-torchlean-cegis-van
```

Completing the workflow doesn't by itself establish the requested Lyapunov conditions. What you can
conclude depends on what the computed bounds prove about the penalty.

The objective adds two ReLU hinge penalties: one for the lower bound on the Lyapunov value, and one
for its required decrease along the dynamics. Each penalty is the positive part of a constraint
violation. In the exact real interpretation, their nonnegative sum is zero precisely when both
encoded inequalities hold at that point. So to establish those conditions throughout a box, you
need a sound upper bound of zero on the real penalty. A rounded zero alone needs an arithmetic
argument as well. A lower endpoint near zero can't establish the condition: the useful bound must
control the upper endpoint throughout the region.

In the reports, sampled training and search losses are kept apart from bounds on the whole box. The
lowered IR node count describes the operation graph, while tensor shapes determine how many scalar
weights those operations use. The final IBP and CROWN intervals concern the scalar penalty over the
selected box {Informal.citep gowal2018}[]{Informal.citep crown2018}[]. An upper bound above zero
leaves the encoded condition unresolved, but it doesn't by itself provide a violating point.

In the log, each `[stage2]` line reports the penalty at the state that round's PGD search found.
That state is a candidate to inspect: it becomes evidence of a violated condition only after
evaluating the intended property with the required semantics and confirming the violation. A
positive value there doesn't explain the upper bound on all other states, and it isn't a proof
about trajectories of the dynamical system.

To compare verifier bounds, hold the model, parameters, input region, and scalar semantics fixed,
because changing the network changes the function whose maximum is being bounded. Even for a fixed
function, an upper bound alone doesn't reveal its own looseness (a matching lower estimate of the
maximum helps measure that gap). Branching and refined relaxations (as in α,β-CROWN
{Informal.citep betacrown2021}[]) can improve inconclusive bounds, and `auto_LiRPA`
{Informal.citep autolirpa2020}[] supplies bound propagation across more general graphs. But a
semantic claim still requires the checker-to-model theorem and its arithmetic assumptions.

# Coordinate Selection In Verifier Lowering

In stage two, the loss reads coordinates out of the state vector (`x₁` and `x₂` for the van der Pol
dynamics) with `Op.indexAt` through `Ops.apply`. The lowering in
{src "TorchLean/NN/IR/Record.lean"}[`IR/Record.lean`] handles `indexAt` along any valid axis, and
the supplied `Fin` index is fixed when we construct the graph. Selection emits a native slice of
length one along that axis, followed by a reshape that removes the singleton dimension.

The graph copies the selected entries, preserving their floating-point words, including NaNs,
infinities and signed zeros (`Op.rule?_slice`, `Tensor.slice_apply`). Interval propagation applies
the slice to both endpoints of its input's box. When that box encloses the input's value, the
result encloses the slice's value (`Op.ibp?_encloses`). The backward CROWN rule `Op.directedRule`
in {src "TorchLean/Verify/CROWN/Op/ExecCROWN.lean"}[`Op/ExecCROWN.lean`] scatters the coefficient
of a slice along one axis with step 1 back to its source positions, with zero at the other
positions, and discharges the coefficient of any other slice against the slice's interval box. Both
cases are sound (`Op.directedRule_sound`). The forward α-CROWN step gives a slice the constant
forms of its interval box. At `ℝ` they enclose the slice's value whenever the box does
(`Op.alpha?_encloses`).

`Op.gather` and `Op.scatter` take tensor-valued indices and remain outside this lowering fragment.
That's a limit of the implemented translation and transfer rules, not a claim that indexed
operations can't be bounded.

# Lyapunov Certificate Validity

A Lyapunov certificate `Cert` is four numbers plus a region.

On their own, those finite numbers can't imply anything about a real-valued dynamical system. The
missing ingredient has a name in TorchLean, and it's a `Prop`:

```lean (name := tsValidFor)
-- ValidFor is the proposition that these bounds enclose
-- these particular functions.
#check @Cert.ValidFor
-- Converting the numeric record constructs certificate data
-- without proving validity.
#check @RealCert.toCert
```

```leanOutput tsValidFor (whitespace := lax)
@Cert.ValidFor : {α : Type} →
  [inst : Storage α] → [inst_1 : Context α] → {n : ℕ} → Cert α n → Candidate α n → Prop
```

```leanOutput tsValidFor (whitespace := lax)
@RealCert.toCert : {n : ℕ} → RealCert n → Cert ℝ n
```

`ValidFor cert lyap` states enclosure of the supplied Lyapunov value and orbital-derivative
functions throughout the region, and the four numeric fields alone can't establish it. So a
theorem must either prove that proposition or keep it as a hypothesis.

In the first output, `Cert α n → Candidate α n → Prop` says that `ValidFor` takes
certificate data and a pair of functions and produces a proposition. That proposition needs a
proof. By contrast, the second output ends in `Cert ℝ n`, a data type (`toCert` repackages
the real endpoints into the canonical record). So the types already expose the distinction between
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

noncomputable def tsCert : Cert ℝ 1 :=
  tsRealCert.toCert

noncomputable def tsLyap : Candidate ℝ 1 :=
  { value := fun x => (x.unstack 0).item ^ 2
    orbitalDerivative := fun x =>
      -2 * (x.unstack 0).item ^ 2 }
```

In this example, we can establish that evidence with a proof. The enclosure has two halves: on the
region, `V` lies in $`[1,4]`, and its orbital derivative lies in $`[-8,-2]`:

```lean (name := tsValidProof)
-- Prove both enclosure fields from membership in the stated
-- one-dimensional region.
theorem tsValid : tsCert.ValidFor tsLyap := by
  constructor
  -- The candidate value lies between its stored lower and
  -- upper bounds.
  · intro x hx
    have h := hx 0
    simp [Box.Contains, tsCert, tsRealCert,
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
    simp [Box.Contains, tsCert, tsRealCert,
      RealCert.toCert] at h
    obtain ⟨hLower, hUpper⟩ := h
    constructor <;>
      simp only [tsLyap, tsCert, tsRealCert,
        RealCert.toCert] <;>
      nlinarith [sq_nonneg ((x.unstack 0).item)]
```

On the region, $`1\le x\le2`, hence $`1\le x^2\le4` and $`-8\le-2x^2\le-2`. And the supplied
orbital derivative is consistent with the scalar dynamics $`x'=-x`: the derivative of $`V(x)=x^2`
is $`2x`, and $`(2x)(-x)=-2x^2`. `Candidate` itself stores the value and orbital-derivative
functions independently, so a controller application must establish that connection as well. This
example proves sign and range conditions on $`[1,2]`, but it doesn't prove equilibrium stability or
invariance of that region.

With the proof in hand, the analytic conclusion follows from the theorem that consumes it:

```lean (name := tsConditionsProof)
-- Combine the proved enclosures with the strict signs of
-- their endpoints.
theorem tsConditions :
    (∀ x, tsCert.region.Contains x →
      tsLyap.value x > 0) ∧
    (∀ x, tsCert.region.Contains x →
      tsLyap.orbitalDerivative x < 0) :=
  Cert.sign_conditions tsLyap tsCert tsValid
    (by norm_num [tsCert, tsRealCert, RealCert.toCert])
    (by norm_num [tsCert, tsRealCert, RealCert.toCert])
```

The two `norm_num` calls are the sign conditions on the numbers (`vLower > 0` and
`derivativeUpper < 0`). These are the obligations that the numbers alone discharge. Spelling the
theorem out at $`n=1` names the three hypotheses, and the `example` compiles only if the real
theorem has exactly this shape:

```lean (name := tsLyapThm)
-- State the same implication with validity and endpoint
-- signs as explicit hypotheses.
example (lyap : Candidate ℝ 1)
    (cert : Cert ℝ 1)
    (enclosure : cert.ValidFor lyap)
    (positiveValue : cert.vLower > 0)
    (negativeDerivative : cert.derivativeUpper < 0) :
    (∀ x : Tensor ℝ [1], cert.region.Contains x →
        lyap.value x > 0) ∧
    (∀ x : Tensor ℝ [1], cert.region.Contains x →
        lyap.orbitalDerivative x < 0) :=
  Cert.sign_conditions lyap cert enclosure
    positiveValue negativeDerivative
```

Here the hypothesis `enclosure` supplies `ValidFor` on the region, and the next two hypotheses
compare real certificate endpoints with zero. A theorem that leaves `ValidFor` as a hypothesis
proves a conditional sign result. It does not show that any computed bounds apply to the intended
dynamical system.

In a controller workflow the producer searches for a policy $`u_\theta` and a candidate $`V`, aiming
at

$$`V(x)\geq 0,\qquad
\nabla V(x)\cdot f(x,u_\theta(x))
\leq-\alpha\|x\|^2.`

Exporting terminal boxes with positive numeric margins does not prove those inequalities. To close
the gap, an artifact would have to identify the model and parameter hash, the state-space root
region, the exact dynamics and scalar semantics, every partition leaf, replayable bounds for `V`
and its Lie derivative, and the coverage and boundary conditions. TorchLean's other
`RealCert`-style and IR-based checkers show stronger replay patterns, but each command should be
read on its own terms, without crediting it by association with the name of an external solver.

# Recording A Run

A useful run record lets another person recover the claim and repeat the check. Keep the raw
producer dump, the normalized artifact, the exact commands and their outputs, and the source and
dependency revisions used for that run. The bundled dump above exercises the exporter and checker.
Evaluating an external solver also requires its repository revision, Python and solver environment,
model checkpoint hash, and property file.

Record the dtype, device, numerical settings, original input region, and meaning of each output
margin alongside those files. A lower-bound vector changes meaning when the model, property, or
margin convention changes. The leaf format doesn't identify all of those inputs, so a record
outside the artifact must supply them. Reproducing the numbers still leaves the semantic proof
obligations described below.

# Workflow Limits And Proof Obligations

The leaf workflow checks that exported data satisfy its structural contract. Its limits are
concrete:

- *Bounded coverage checking.* The endpoint-grid check rejects missing regions, as in the wide-root
  and hole examples above. It refuses grids with more than one million cells.
- *No replay.* The lower bounds are read, not recomputed against the network.
- *No soundness lemma for the predicates.* `boxWithin`, `refutesThreshold`, and
  `refutesThresholdAt` in {src "TorchLean/Verify/Tensor.lean"}[`Verify/Tensor.lean`] are `Bool`
  functions with no theorem tying them to a `Prop`-level statement about box inclusion or margin
  refutation. The endpoint-grid coverage check likewise has no theorem connecting acceptance to
  set-theoretic coverage. The local predicates compute the stated comparisons, and the
  {ref "certificates"}[certificate chapter] tests them on both sides of every boundary, but no
  proof covers them.

The Lyapunov runner also computes bounds on a graph lowered from the trained TorchLean program. A
completed controller argument needs an upper bound establishing the penalty condition, coverage of
the intended region, and proofs connecting the selected checker and arithmetic to the Lyapunov
hypotheses.
