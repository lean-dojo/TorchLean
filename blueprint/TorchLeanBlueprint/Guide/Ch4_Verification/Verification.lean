import VersoManual
import VersoBlueprint

open Verso.Genre Manual

#doc (Manual) "Verification" =>
%%%
tag := "verification"
%%%

Verification in TorchLean is not a single button. It is a family of claims about explicit objects.
A graph can be well formed. A compiler can preserve denotation. A bound propagation pass can
enclose all outputs over an input box. A certificate can be replayed. An autograd pass can compute
the adjoint derivative. A Float32 theorem can transfer a real-valued margin to finite precision.

The first question is always:

> What object is being verified?

The usual object is an [NN.IR.Graph](https://github.com/lean-dojo/TorchLean/blob/main/NN/IR/Graph.lean)
plus a parameter payload and an input region. Other pages in this section use tensor programs,
tapes, datasets, trajectories, residual functions, and imported JSON certificates. The object
matters because the theorem can only talk about the artifact it names.

The common path is:

- start from a TorchLean API program, GraphSpec model, or imported artifact;
- lower or translate it to a semantic object such as `NN.IR.Graph` plus a `ParamStore`;
- run a checker such as IBP, CROWN, a certificate parser, an ODE corridor checker, or a scientific
  ML checker;
- interpret the accepted result through a theorem about the named object.

A few names come up over and over in what follows:

- `IBP`: interval bound propagation, where each node carries an interval box `[lo, hi]`.
- `CROWN`: a family of backward linear-relaxation methods that propagate affine bounds.
- `LiRPA`: linear relaxation-based perturbation analysis, the broader family that includes IBP and
  CROWN methods.
- `IEEE32Exec`: TorchLean's executable IEEE-754 binary32 kernel in Lean.
- `FP32`: TorchLean's float32 model used in proofs, instantiated as rounded real semantics.

# A Map Of Claim Types

TorchLean's verification support is easiest to read as a table of objects and support mechanisms:

- *Shape well formedness*: the object is `NN.IR.Graph`; the support is an executable structural
  checker for node ids, parent ids, op arities, payloads, and output shapes.
- *Compiler alignment*: the object is an IR graph and a compiled runtime graph; the support is a
  Lean theorem relating their denotations for a supported fragment.
- *IBP enclosure*: the object is a graph, payload, and input box; the support is a checker plus a
  soundness theorem for interval propagation.
- *CROWN certificate*: the object is affine lower and upper forms over a graph; the support is a
  replay checker or theorem fragment for the supported operators.
- *Autograd correctness*: the object is a tape or graph; the support is a Lean theorem saying
  backprop computes the adjoint derivative.
- *Runtime approximation*: the object is a runtime tensor and a spec tensor; the support is a
  tolerance theorem.
- *External α,β-CROWN artifact*: the object is a JSON leaf certificate; the support is structural
  checking today, with stronger recomputation certificates as the natural next level.
- *Scientific ML certificate*: the object is an ODE tube, PINN residual certificate, spline
  certificate, or controller artifact; the support is a checker plus a theorem pattern for that
  mathematical domain.

The useful habit is to name the workflow and claim level. "Verified" should come with a noun:
verified shape check, verified IR compilation theorem, verified IBP enclosure, verified imported
certificate, or verified float32 transfer theorem.

# Three Theorem Shapes

The details differ by subsystem, but several theorem shapes appear repeatedly.

For robustness, the target is usually a margin statement:

$$`\forall x\in B,\qquad f_y(x)-\max_{k\ne y} f_k(x) > 0`

For a bound propagation pass, the soundness theorem has the form:

$$`x\in B
\quad\Longrightarrow\quad
\llbracket G\rrbracket(x)\in \gamma(\mathcal A(G,B))`

Here `G` is the graph, `B` is the input region, `\mathcal A` is the abstract interpreter or
relaxation pass, and `\gamma` converts the abstract object back into a concrete set of possible
outputs.

For a certificate checker, the theorem shape is:

$$`\operatorname{Check}(G,C)=\texttt{true}
\quad\Longrightarrow\quad
\operatorname{Sound}(G,C)`

This is the pattern that makes external producers useful. A producer can search for `C`; Lean checks
the finite object and applies the theorem attached to the checker.

# The Main Idea

The same graph serves several roles at once. A model is compiled to an explicit symbolic IR graph
(`NN.IR.Graph`); verifier passes such as interval bounds, affine bounds, and certificate checks run
on that graph; and theorems talk about the graph denotation rather than about an opaque execution
trace.

That gives us a useful vocabulary for the rest of the manual:

- `proved` means Lean has a theorem about a semantic object defined in Lean.
- `checked` means Lean recomputes or structurally validates an imported artifact.
- `assumed` means a runtime or external producer hypothesis is named instead of folded silently into
  the verified claim.

## The Claim Ladder

A good way to read TorchLean verification output is as a ladder:

1. *The file runs.* This catches ordinary integration failures.
2. *The graph is well formed and well shaped.* The symbolic object is structurally meaningful.
3. *A checker accepts an artifact.* Lean has parsed and checked a concrete object.
4. *A theorem applies to the supported fragment.* The accepted object implies a mathematical
   statement about the graph denotation.
5. *A scalar bridge applies.* The theorem can be related to the runtime scalar path being claimed.

The ladder is meant to prevent category mistakes. A CUDA training run supplies evidence about a
runtime path. A graph soundness theorem supplies a mathematical statement about an IR denotation.
An imported certificate checker supplies a statement about the artifact it accepts. Strong claims
say which rungs were used.

This is the bridge between a PyTorch style API and verification mathematics that can be stated and
cited precisely.

# Shared Semantic Target

TorchLean is not just a verifier with a convenient front end. Runtime compilation and verification
share one graph semantic target, the scalar semantics are explicit enough that floating point
assumptions can be named rather than hand-waved, and certificate checking is separated into
recompute-and-compare checks versus weaker structural checks of imported artifacts.

That perspective is why the examples cover several application families instead of a single minimal
robustness case:

- robustness and certified margins,
- PINN residual bounds,
- ODE enclosure and dynamical-system reasoning,
- controller and Lyapunov two-stage workflows,
- and benchmark oriented exported artifact checks such as VNN-COMP suites.

# External Producers, Python, Julia, And FFI

Verification workflows often need tools outside Lean. A Python verifier may optimize CROWN slopes,
a Julia script may generate a scientific certificate, a PyTorch checkpoint may provide parameters,
and a CUDA kernel may produce runtime values. TorchLean treats those systems as producers: they do
search, training, optimization, or execution, and Lean checks the artifacts that have a stated
contract.

The repository uses two patterns:

- *Process wrappers*: [external process helpers](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/External/Process.lean) and
  [Julia helpers](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/External/Julia.lean) resolve commands, run tools, capture JSON,
  and fail loudly when the external command is unavailable or malformed.
- *Artifact checkers*: verification modules parse exported JSON, checkpoint-like parameter stores,
  ODE corridors, PINN residual certificates, CROWN node certificates, or α,β-CROWN leaf artifacts
  and check a precise predicate inside Lean.

That gives a clean division of labor. Python, Julia, CUDA, and external verifiers can search,
train, optimize, and generate candidate artifacts. Lean checks the artifacts that have a specified
format and theorem-backed meaning. If a workflow imports an external answer without recomputation
or a proof-backed checker, the guide records the producer hypothesis as part of the claim.

# From Bug Case Studies To Verification Obligations

The BugZoo examples are not separate from verification. They are small versions of the same pattern:
turn an informal failure mode into a precise obligation.

For example:

- An attention-mask bug becomes a theorem about masked softmax weights.
- A KV-cache bug becomes an append invariant over the final cache slot.
- A tokenizer/config bug becomes an in-range token contract using `Fin vocabSize`.
- A compiler wrong-code bug becomes a source-target semantic preservation obligation.
- A float deployment bug becomes a bridge obligation from runtime float32 behavior to
  `IEEE32Exec` or `FP32`.
- A CUDA fused kernel bug becomes a question of whether the native kernel refines the Lean spec.

That last item is exactly how to read FlashAttention in TorchLean. The spec theorem
`flashAttention_eq_scaledDotProductAttention` says the fused proof object denotes ordinary scaled
dot product attention. The native CUDA symbols are then tested and isolated behind the FFI boundary.
So the verification claim is layered:

1. the fused spec equals SDPA;
2. the graph/compiler path uses the stated attention op;
3. verifier passes reason about the supported graph fragment;
4. the native CUDA kernel remains an external implementation that must be tested or separately
   proved against the spec.

This is our recurring habit: every strong claim should say which layer discharged it.

## Reproducing The Geometry3D Detector Check

The Geometry3D workflow is the same producer/checker pattern with a real 3D vision model. WildDet3D
is the producer: it emits a camera matrix, a 3D object box, and a 2D detection box. Lean is
the checker: it parses the exported JSON, recomputes the projection, checks positive depth and image
bounds, and proves via `checkCert_sound` that an accepted certificate satisfies the
`Verified3DBox` predicate.

The light real-image path uses DETR plus Depth Anything V2:

```
python3 scripts/verification/regenerate_assets.py --group geometry3d-real --run
lake exe verify -- camera-box3d-cert _external/geometry3d/realworld/coco_cats_depth_box.json
```

The heavier actual-3D-detector path uses Ai2's
[`allenai/WildDet3D`](https://huggingface.co/allenai/WildDet3D) model/Space:

```
python3 -m pip install -r scripts/verification/geometry3d/requirements-wilddet3d.txt
python3 -m pip install --no-deps utils3d
python3 scripts/verification/regenerate_assets.py --group geometry3d-wilddet3d --run
lake exe verify -- camera-box3d-cert _external/geometry3d/wilddet3d/wilddet3d_cat_box3d_cert.json
```

The WildDet3D group produces both an accepted projected-footprint certificate and a rejected strict
model-bbox diagnostic. This is the important distinction: the detector can be useful, while its
2D/3D glue claim is still checked after export instead of accepted because the tensor came from a
large model.

![WildDet3D model bbox versus projected 3D footprint](Guide/Assets/bug-zoo/geometry3d-wilddet3d-bbox-diagnostic.png)

# LiRPA Family (The Unifying Pattern)

Most practical neural network verifiers in the IBP/CROWN family are instances of LiRPA
(Linear Relaxation-based Perturbation Analysis). The pattern is familiar: start with an input
uncertainty set, usually a box or an `l∞` ball; propagate an overapproximation of each node's
value through the graph; and rely on induction over node order to turn local rules into a global
guarantee.

TorchLean makes that pattern visible in the module structure. The shared object is `NN.IR.Graph`,
the shared semantics is `NN.IR.Graph.denote` / `NN.IR.Graph.denoteAll`, and the verifier passes
produce `PropState` artifacts such as IBP boxes and affine forms over that same IR.

This section focuses on the implemented IBP and CROWN affine layer, plus the way the
repository keeps `proved`, `checked`, and `assumed` distinct.

# IBP vs CROWN (What These Passes Actually Compute)

TorchLean's core bound propagation passes follow the LiRPA family of methods, in which the verifier
tracks an overapproximation through the graph instead of symbolically expanding every branch.

## IBP (Interval Bound Propagation)

IBP is the simplest sound pass:

- Each node carries an interval box `[lo, hi]` for each scalar component of the node's tensor.
- Each op has a local "interval transformer" that maps parent boxes to an output box.

Informally, IBP says:

> if each parent value lies inside its parent box, then the node value lies inside the box produced
> by the IBP transformer for that op.

This makes IBP cheap and robust, but also conservative. Over sufficiently deep nonlinear
composition, intervals can widen quickly.

The key linear layer rule is sign splitting. For a row `y_i = sum_j W_ij x_j + b_i`, positive
weights use the lower endpoint for the lower bound and the upper endpoint for the upper bound,
while negative weights do the opposite. In a two input example with `x1 in [0, 1]`,
`x2 in [-1, 2]`, and `y = 3*x1 - 2*x2 + 0.5`, IBP gives `lower = -3.5` and `upper = 5.5`.

For ReLU, IBP uses monotonicity: if `y in [l, u]`, then `ReLU y` lies in
`[max 0 l, max 0 u]`.

That is sound and extremely simple. The limitation is that after this step the verifier has forgotten
which input directions made `y` large or small; it keeps only the interval endpoints.

## CROWN / DeepPoly (Affine Bounds)

CROWN passes carry *affine* upper and lower bounds with respect to a designated input, plus
(typically) the IBP box as a fallback. Affine bounds are tighter than pure intervals, but:

- they are more expensive,
- they need per op linear relaxations (especially for nonlinearities),
- and they often use IBP to seed slopes or guard cases.

TorchLean stores these objects in flattened form for efficiency and reuse across operators. In
spirit, that design is closest to the DeepPoly/ERAN style of verifier.

The important difference from IBP is that CROWN keeps a symbolic linear envelope instead of only a
box. A node is not merely "somewhere between `lo` and `hi`"; it is bounded by affine functions of
the original input.

For linear layers, we reuse the same sign-splitting idea, but we split against the parent affine
lower and upper forms rather than against raw interval endpoints. The
[AlphaCROWN certificate API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/CROWN/Cert/AlphaCROWN.lean) contains this transfer
rule, especially the
`linearBoundsFromAffine` path.

For ReLU, the interesting case is the unstable interval `l < 0 < u`. The exact function is bent, so
we enclose it with lines. The standard upper line has slope `u / (u - l)` and passes through
`(l, 0)` and `(u, u)`. The lower line can be either `0` or `y`. For example, if `y in [-1, 2]`,
the standard CROWN upper line has slope `2/3`.

The lower line can be either `0` or `y`, both sound but useful in different downstream objectives.
That small choice is the seed of α-CROWN.

# A More Concrete Example: Certifying A Margin

A typical robustness property is not "bound every logit independently" but "prove one logit stays
above the others." If class `0` is supposed to win against class `1`, the verifier works with the
margin `logit_0 x - logit_1 x` and tries to prove that it is positive for every perturbed input.

IBP can bound `logit_0` and `logit_1` separately. For example, if `logit_0 in [0.8, 1.4]` and
`logit_1 in [0.2, 1.0]`, the safe margin lower bound is `0.8 - 1.0 = -0.2`, so IBP cannot certify the property even
though the two logits may be strongly correlated.

CROWN instead can backpropagate the linear objective `logit_0 - logit_1` through the network and
bound that margin directly. If the affine margin lower bound evaluates to `0.15` on the input box,
then every input in the perturbation box keeps class `0` ahead of class `1` by at least `0.15`.

This is why CROWN methods matter: they can preserve correlations that interval methods
erase.

# Worked Example: TorchLean, IR, and IBP (Tiny MLP)

The simplest verification pipeline in TorchLean is easy to state: define a model with the
TorchLean API, compile its forward pass to the canonical IR with operation tags (`NN.IR.Graph`), seed an
input uncertainty set, run a verifier pass such as IBP, and then read off the output bounds at the
output node id.

The runnable demo is
[TorchLean IBP](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Verification/TorchLean/TorchLeanIBP.lean), and the
matching CLI entrypoint is `lake exe verify -- torchlean-ibp`.

Below is the essential core of that file, trimmed to the minimum needed to follow the pipeline:

```
import NN
import NN.Verification.TorchLean.Compile

open _root_.Spec
open Tensor
open NN.Tensor
open NN.API

open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN

def model : nn.Sequential (Shape.Vec 2) (Shape.Vec 1) :=
  nn.blocks.mlp 2 1 { hidden := [3], activation := .relu, seedBase := 0 }

def paramShapes : List Shape := nn.paramShapes model

def runOnce {α : Type} [Semantics.Scalar α] [Runtime.Scalar α] [DecidableEq Shape] [ToString α] : IO Unit := do
  -- Parameters in the order expected by `paramShapes`.
  let params : tlist.TList α paramShapes :=
    tlist!
      (tensorND! [3, 2] [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
      (tensorND! [3] [0.1, 0.2, 0.3]),
      (tensorND! [1, 3] [0.7, 0.8, 0.9]),
      (tensorND! [1] [0.4])

  let compiled ←
    match NN.Verification.TorchLean.compileForward1
          (α := α) (paramShapes := paramShapes) (inShape := Shape.Vec 2) (outShape := Shape.Vec 1)
          (nn.program (model := model) (α := α)) params with
    | .ok c => pure c
    | .error e => throw <| IO.userError e

  -- Seed an input box x in [x0 - eps, x0 + eps].
  let x0 : Tensor α (Shape.Vec 2) := tensorND! [2] [0.5, 0.8]
  let eps : α := Runtime.ofFloat 0.1
  let rad : Tensor α (Shape.Vec 2) := Spec.fill eps (Shape.Vec 2)
  let xB : FlatBox α := { dim := 2, lo := Tensor.sub_spec x0 rad, hi := Tensor.add_spec x0 rad }

  -- Run IBP on the compiled IR + parameter store.
  let ps : ParamStore α :=
    { compiled.ps with inputBoxes := compiled.ps.inputBoxes.insert compiled.inputId xB }
  let boxes := runIBP (α := α) compiled.graph ps

  -- Read the output bounds.
  let some outB := boxes[compiled.outputId]! | throw <| IO.userError "IBP produced no output box"
  IO.println s!"output lo = {pretty outB.lo}"
  IO.println s!"output hi = {pretty outB.hi}"
```

Two TorchLean invariants are visible immediately:

- The verifier consumes the same `NN.IR.Graph` the runtime can also interpret (one semantic target).
- Parameters live in an explicit `ParamStore` keyed by node id, so verification never guesses weights.

The key takeaway from the code sample is:

- `compileForward1` makes the graph and its payload explicit,
- `runIBP` reads that graph directly,
- and the output bounds are only meaningful because both steps share the same semantics.

Smallest complete verifier command:

```
lake exe verify -- torchlean-ibp
```

# α-CROWN and α,β-CROWN (Where They Fit)

In the broader ecosystem, CROWN is usually paired with stronger linear relaxations for nonlinear
ops and with a backward pass that tightens the bound for a particular output margin. TorchLean has
the same general shape, but the implementation is split between a Lean LiRPA engine
for a curated op set ([NN/MLTheory/CROWN](https://github.com/lean-dojo/TorchLean/tree/main/NN/MLTheory/CROWN/)) and checkers that compare Lean
recomputation to JSON artifacts exported from Python tooling
([NN/Verification/Cert](https://github.com/lean-dojo/TorchLean/tree/main/NN/Verification/Cert/)).

Here is the practical meaning of the Greek letters:

- `α` chooses the lower relaxation slope for unstable ReLUs.
- `β` carries branch/split information, usually ReLU phase information, from a branch-and-bound
  verifier.

For an unstable ReLU with pre-activation `y in [l, u]` and `l < 0 < u`, the lower bound can choose
between the two obvious sound lines, `ReLU y >= 0` and `ReLU y >= y`. α-CROWN generalizes this as
a slope choice `ReLU y >= alpha * y` with `0 <= alpha <= 1`.

Different `alpha` values give different final margin bounds. The external α-CROWN optimizer searches
for good α values. TorchLean's Lean transfer rule does not pretend to have run that optimizer;
it checks the result for a fixed set of α values and proves/checks that the transfer rule is sound
when the α values are in range.

β-CROWN adds the branch-and-bound side. A ReLU that was unstable in the root box might become known
on a branch:

- `-1`: inactive phase, meaning `y <= 0`, so the ReLU transfer is exact with `ReLU(y) = 0`.
- `0`: unstable or unsplit phase, so the checker uses the CROWN or α-CROWN relaxation.
- `1`: active phase, meaning `0 <= y`, so the ReLU transfer is exact with `ReLU(y) = y`.

TorchLean's current α,β checker uses exactly that narrow interpretation. In the
[α,β-CROWN certificate API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/CROWN/Cert/AlphaBetaCROWN.lean), β is represented as a
per node vector with entries `-1`, `0`, or `1`. The checker verifies that the requested phase agrees
with the provided IBP pre-activation bounds:

- active requires `0 <= lo`,
- inactive requires `hi <= 0`,
- unstable imposes no extra phase condition.

Then the ReLU transfer becomes exact for active/inactive neurons and falls back to the relaxation
for unstable neurons.

That boundary is explicit. The full external search loop that chooses branches, optimizes α values,
and prunes the tree is still outside the formal boundary. The imported artifact is checkable at the
boundary we can state precisely: dimensions, graph ids, per node affine bounds, α vectors,
β phase vectors, and Lean recomputation of the local transfer rule.

There are two checker levels:

- The [CROWN node certificate API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/Cert/CROWNNodeCert.lean) checks per node
  α-CROWN affine bounds by recomputing the Lean transfer step.
- The [α,β CROWN node certificate API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/Cert/CROWNNodeCertAlphaBeta.lean)
  does the same with optional β phase vectors.

The α,β-CROWN two-stage workflow is developed in the *Certificates* and *Two-Stage Workflows*
chapters. Here, the emphasis stays on the shared IR and on what "certificate check" means inside
TorchLean.

# More Worked Examples

The fastest way to understand the verifier stack is to watch a few tiny networks go through the
same logic used for larger graphs. These examples are intentionally small; their value is that every
production-scale claim is made of these same local steps plus induction over the graph.

## Example 1: IBP certifies a one-layer margin

Suppose the input is one-dimensional: `x in [0.9, 1.1]`, `logit0 x = 2*x`, and
`logit1 x = x + 0.4`.

IBP gives:

- `logit0 in [1.8, 2.2]`,
- `logit1 in [1.3, 1.5]`.

The margin lower bound is `1.8 - 1.5 = 0.3`.

So the verifier can certify class `0` over the whole input box. This is the pleasant case: the
property is simple enough that interval endpoints are already decisive.

## Example 2: IBP can lose a true correlation

Now use a different pair of logits: `x in [0, 1]`, `logit0 x = x`, and
`logit1 x = x - 0.1`.

Mathematically, class `0` wins everywhere because the margin simplifies to `0.1`.

But if IBP bounds the logits separately, it gets:

- `logit0 in [0, 1]`,
- `logit1 in [-0.1, 0.9]`,
- margin lower bound `0 - 0.9 = -0.9`.

That does not prove the property. Nothing is wrong with IBP; it is sound. The interval abstraction
has discarded the fact that the same `x` appears in both logits. CROWN can keep the affine
dependency and bound the margin directly:

For the margin objective, CROWN keeps the cancellation `x - (x - 0.1) = 0.1`.

So CROWN certifies the property with margin `0.1`. Affine forms retain this dependency information;
endpoint boxes alone do not.

## Example 3: ReLU relaxation at an unstable neuron

Consider `x in [0, 1]`, `y = x - 0.5`, and `z = ReLU y`.

IBP first gives:

- `y in [-0.5, 0.5]`,
- `z in [0, 0.5]`.

The ReLU is unstable because `y` may be negative or positive. CROWN stores linear envelopes. The
standard upper line over `[-0.5, 0.5]` is `z <= 0.5 * (y + 0.5)`. Sound lower
choices include `z >= 0` and `z >= y`. The α-CROWN lower relaxation chooses an
interpolating lower slope `z >= alpha * y` with `0 <= alpha <= 1`.

If we only need a lower bound on `z`, `alpha = 0` is often strongest on this box because it gives
`z >= 0`. But downstream layers can flip signs or combine neurons, so the best α value is
objective-dependent. That is why an external α optimizer can improve the final margin even though
every local α choice remains a simple line.

## Example 4: β splitting turns an unstable ReLU into exact cases

The same ReLU can become easy after a split. Again take `x in [0, 1]`,
`y = x - 0.5`, and `z = ReLU y`.

At the root box, `y in [-0.5, 0.5]`, so β must be `0` (unstable). Now split the input domain:

- left branch: `x in [0, 0.5]`, so `y in [-0.5, 0]`;
- right branch: `x in [0.5, 1]`, so `y in [0, 0.5]`.

On the left branch, β may say inactive (`beta = -1`) and the exact transfer is `z = 0`. On the
right branch, β may say active (`beta = 1`) and the exact transfer is `z = y`.

This is the branch-and-bound intuition behind β-CROWN. A loose triangle relaxation is not the end of
the argument; the input space can be split until ambiguous ReLUs become exact on each leaf.
TorchLean's current checker does not accept the split just because a JSON file says so. It checks
the β phase against the branch's IBP pre-activation bounds, then applies the exact active/inactive
transfer when the phase is consistent.

## Example 5: What a node certificate is proving locally

For a ReLU node certificate, the checker is essentially asking:

Given parent affine bounds, a parent IBP box, an optional alpha slope, an optional beta phase, and
claimed output affine bounds, Lean asks whether the local transfer rule recomputes the same output
affine bounds.

If yes, that node's certificate is accepted. If every node is accepted in topological order, and the
graph is in the supported semantic fragment, the proof declarations can lift those local checks to a global
enclosure theorem. That is the heart of the design: small local checks, then a graph soundness
argument.

# Pointers To Deeper Chapters

The verification tree is broad, so this section names the areas that need their own careful reading
path rather than being compressed into one verifier slogan:

- `CROWN` as objective dependent backward reasoning, not only forward node by node affine bounds.
- The difference between a per node recompute certificate and a branch and bound leaf artifact.
- Which ops have proofs today versus only executable/debuggable support today.
- How derivative bounds support PINN and ODE workflows.
- How `BoundOps` and directed rounding relate to real valued enclosures and float32 execution.
- How imported PyTorch / VNN-COMP / α,β-CROWN artifacts become Lean objects instead of
  remaining opaque blobs.
- How the proof declarations connect to the executable checkers: local transfer soundness, topo-order
  induction, and final property extraction.

The next sections cover the first few of those more concretely; the certificate and two-stage
workflows cover the external artifact side.

# Application Families In The Current Repository

The verification tree covers several different application styles. They all reuse the same graph
ideas, but they answer different scientific questions.

## 1. Robustness and margin certification

This is the most familiar verifier workflow for many ML readers:

- seed an input perturbation region,
- propagate bounds through the network,
- check that the true class margin stays nonnegative.

Relevant tools and examples:

- `torchlean-robustness`
- `digits`
- `margin-cert`
- `torchlean-ibp`

This is the natural place to connect TorchLean to the adversarial-robustness literature and to the
IBP / CROWN / LiRPA papers already cited below.

## 2. PINN residual bounds

The repository includes a meaningful PINN slice rather than only generic verifier infrastructure.
The PINN code answers questions of the form:

> Over a spatial or spatio-temporal box, how large can the PDE residual become, given bounds on
> the network output and its derivatives?

The implementation is organized around the [PINN CLI](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/PINN/CLI.lean), the
[PINN certificate checker](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/PINN/Certificate.lean), the
[PINN dataset checker](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/PINN/DatasetCheck.lean), and the commands
`pinn-cli`, `pinn-cert`, and `pinn-dataset-check`.

This is one of the clearest places where TorchLean's derivative-aware graph propagation is doing
something more specialized than standard image-classifier robustness.

## 3. ODE enclosure verification

The ODE stack answers a different question again:

> Does a neural network satisfy subsolution / supersolution style inequalities for a differential
> equation over a region?

Useful declarations and tools:

- [ODE API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/ODE.lean)
- [ODE verifier API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/ODE/Verify.lean)
- CLI tool: `ode`

This is also one of the places where the proof layer and the numerical layer meet most directly:
the executable checker runs over an explicit scalar backend, while the theorem interpretation
depends on the selected floating-point semantics and runtime agreement.

## 4. External benchmark and ecosystem workflows

Not every useful workflow in TorchLean starts from a model written natively in Lean.

Relevant tools and examples:

- `abcrown-leaf`
- `vnncomp-mnistfc`
- [NN/Examples/Verification/VNNComp](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/Verification/VNNComp/)

These workflows matter because they show how TorchLean interfaces with existing verifier ecosystems
without treating "imported JSON" as already a theorem.

## 5. Controller / Lyapunov two-stage workflows

The Lyapunov and controller workflows are still more research-flavored than the smallest public demos,
but they are important for understanding the paper's broader claim that the stack applies to
dynamical systems as well as to classifier margins.

Useful declarations:

- [Lyapunov oracle boundary](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/CROWN/Lyapunov/Oracle.lean)
- [Lyapunov verification theorems](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/CROWN/Lyapunov/Verification.lean)
- [two stage Lyapunov runner](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/CROWN/Lyapunov/TwoStage/Run.lean)

These declarations expose oracle or certificate assumptions directly instead of hiding them inside a
monolithic external script.

# Core Data Structures (What The Verifier Actually Consumes)

This section answers a simple question: what data do the verifier passes actually consume?

## `NN.IR.Graph` (Structure Only)

- `Graph` is a DAG of `Node`s.
- Nodes store `parents`, an op tag `OpKind`, and a declared `outShape`.
- TorchLean intentionally keeps *values/parameters out of the graph*, because different backends want
  different parameter stores.

## `ParamStore` (Values, Weights, and Seed Input Boxes)

`ParamStore` is a minimal, backend-friendly store of payloads keyed by node id:

- `inputBoxes`: node id mapped to `FlatBox` seed regions for designated input nodes
- `constVals`: node id mapped to constant tensors
- `linearWB`: node id mapped to linear parameters `(W,b)`
- `matmulW`: node id mapped to bias-free matrix multiply parameters
- `conv2dCfg`: node id mapped to typed convolution payloads

Verifiers never guess parameters; they look them up in a store produced by a compiler or provided by
the user.

The reason for the flattened representation is practical: it is easy to update node by node, easy
to compare in tests, and easy to inspect when a bound behaves unexpectedly.

## `PropState` (Per-Node Bound Results)

`PropState` is the per node bound workspace used while debugging passes:

- `states[i] : NodeState` stores:
  - `shape : Shape` (unflattened tensor shape),
  - `ibp? : Option FlatBox` (interval bounds if computed),
  - `aff? : Option FlatAffine` (affine form if computed).

Running a pass typically yields arrays of optional boxes or affine forms, which are then packaged
into a `PropState` for reporting.

For debugging, the infoview widgets `#crown_view` and `#bounds_tightness_view` show quickly whether a
pass populated the expected state.

```
#check NN.IR.Graph.checkWellFormed
#check NN.IR.Graph.checkShapes
#check NN.IR.Graph.checkInferredShapes
#check NN.IR.Check.wellFormed_iff
#check NN.IR.Check.wellShaped_iff
#check NN.Verification.TorchLean.Proved.compileForward1_wellFormed
#check NN.Verification.TorchLean.Proved.evalCompiledForward1_eq_evalForward1
```

# Supported Proved Fragment (Today)

The authoritative statement of the current proved IBP fragment appears at the top of
[NN.MLTheory.CROWN.Proofs.GraphCertSoundness API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/CROWN/Proofs/GraphCertSoundness.lean).

At present, the soundness development is centered on the core graph dialect used by the bundled
TorchLean verification demos:

- `.input`, `.const`, `.detach`
- `.add`, `.sub`, `.mul_elem`, `.relu`
- `.linear`, `.matmul` in the verifier dialect where weights live in `ParamStore`
- `.tanh`, `.sigmoid`, `.sin`, `.cos`

This list matters because it distinguishes a bound that is merely executable from one that is
backed by the proof layer. Demos that use operators outside that fragment can still be run
and inspected, but their outputs should not be treated as proof claims without extending the
soundness development.

The clean way to write this in a paper or note is:

- cite the proved IBP fragment from the graph certificate soundness theorem when the claim is
  limited to the list above, or
- cite the executable verifier implementation when the claim is outside the current theorem scope.

# Certificates (Two Different Meanings)

In TorchLean documentation, the word "certificate" is used in two related but importantly different
senses.

## Recompute and compare certificates (strong checking, in Lean)

These are JSON artifacts that record some intermediate (or final) bound results, and then Lean
recomputes the same step and accepts the JSON only if it matches within a tolerance (to account for
decimal serialization of floats).

This pattern lets Lean re-run the same arithmetic and reject JSON that does not match:

- the JSON is externally produced,
- Lean recomputation is the source of truth,
- and the checker fails loudly on mismatch, often printing both bounds for debugging.

Relevant declarations:

- [IBP certificate checks](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/Cert/IBPCert.lean)
- [per node IBP checks](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/Cert/IBPNodeCert.lean)
- [per node α-CROWN checks](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/Cert/CROWNNodeCert.lean)
- [per node α,β-CROWN checks](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/Cert/CROWNNodeCertAlphaBeta.lean)

For CROWN node certificates, the imported JSON has four conceptual parts:

```
ctx    : which input node the affine forms are written against
ibp    : interval boxes used to choose/validate nonlinear relaxations
crown  : claimed affine lower/upper forms for each node
alpha  : optional ReLU lower-slope choices
beta   : optional ReLU phase choices, encoded as -1 / 0 / 1
```

The checker does not accept the claimed `crown` arrays blindly. For each node, it:

1. reads the parent affine bounds already checked earlier in topological order,
2. reads the parameters from `ParamStore`,
3. recomputes the Lean transfer rule for that node,
4. compares the recomputed affine bounds with the JSON bounds under an explicit tolerance,
5. rejects the artifact if shapes, dimensions, phases, or numeric values disagree.

A tiny ReLU example shows the flavor. Suppose a ReLU node has pre-activation IBP box
`[-1, 2]`.

- If the certificate says `beta = 1` (active), Lean rejects it, because active would require
  `0 <= lo`, but `lo = -1`.
- If the certificate says `beta = -1` (inactive), Lean rejects it, because inactive would require
  `hi <= 0`, but `hi = 2`.
- If the certificate says `beta = 0`, Lean treats the ReLU as unstable and uses the CROWN/α-CROWN
  relaxation.

If another branch has box `[0.3, 2]`, then `beta = 1` is consistent and the transfer can use the
exact active rule `ReLU(y) = y`. If a branch has box `[-2, -0.1]`, then `beta = -1` is consistent
and the transfer can use the exact inactive rule `ReLU(y) = 0`.

## Certificate Strengths

Different certificate formats support different claims.

- A recompute and compare certificate says Lean can independently check a claimed bound artifact.
- A structural certificate says Lean can verify shape and consistency conditions for an artifact.
- A proof-backed certificate exports enough local evidence for Lean to connect acceptance to the
  graph semantics.

## Structural leaf artifacts

Some external workflows export "leaf artifacts" that are meaningful only relative to the verifier's
internal semantics, for example a branch-and-bound leaf condition that says a subdomain may be
pruned because a lower bound exceeds a threshold.

Lean can still parse these artifacts and check their declared structure. If the artifact does not
include the computation that produced the bounds, the claim should record the external producer
hypothesis for those bounds.

The [α,β-CROWN leaf checker](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/Cert/AbCrownLeafCert.lean) handles that boundary
and is exposed through `lake exe verify -- abcrown-leaf`.

# Runnable verification workflows today

All bundled verification workflows are registered in one CLI dispatcher, `verify`.
See *CLI Entry Points* for the exact invocation shape and for the tool-listing convention.

First run: `torchlean-ibp`.

What these runs do, roughly:

1. Build or import a TorchLean model.
2. Compile the forward pass to `NN.IR.Graph` plus a `ParamStore` (weights, consts, and seed boxes).
3. Run IBP and/or CROWN propagation over the IR graph.
4. Report bounds and optionally check a simple postcondition.

For the "Two-Stage" path, which covers α,β-CROWN integration, consult the *Certificates* and
*Two-Stage Workflows* chapters for the external artifact workflow.

# How To Run The Curated Verification Examples

All of the following are registered under the unified verification CLI; see
the [verification CLI API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/CLI.lean) for the authoritative list and defaults:

- TorchLean, IR, and IBP demo: `torchlean-ibp`
- TorchLean, IR, and IBP plus CROWN demo (ops like softmax and `mse_loss`): `torchlean-crown-ops`
- TorchLean "tiny transformer" demo: `torchlean-transformer-ibp`
- Margin / robustness demos: `torchlean-robustness`, `digits`, `margin-cert`
- PINN workflows: `pinn-cert`, `pinn-cli`, `pinn-dataset-check`
- IBP/LiRPA certificate checkers: `lirpa-mlp`, `lirpa-cnn`, `lirpa-attention`, `lirpa-gru`, `lirpa-encoder`
- External artifact structural checks: `abcrown-leaf`
- ODE enclosure verification: `ode`
- Geometry and spline certificates: `camera-box3d-cert`, `spline-cert`
- End-to-end TorchLean MLP workflow: `torchlean-mlp-workflow`
- Benchmark-style exported-artifact suite: `vnncomp-mnistfc`

## Practical reading order

Starting from a runtime demo, the suggested order is:

1. `Graphs and IR`
2. `Floating-Point Semantics`
3. return here once the terms above are familiar

Starting from a certificate artifact, the suggested order is:

1. `Certificates`
2. `Two-Stage Workflows`
3. the [certificate checker API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/Cert.lean)

Many tools accept an explicit JSON path as the first argument; the `-- list` output shows the
default path.

# Reference Map

## IR and compilation

- Canonical graph with operation tags: [IR graph API](https://github.com/lean-dojo/TorchLean/blob/main/NN/IR/Graph.lean)
- TorchLean forward compiler to IR: [compiler API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/TorchLean/Compile.lean)
- IR evaluation helpers: [TorchLean correctness API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/TorchLean/Correctness.lean)
- Proved forward fragment: [proved compiler fragment](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/TorchLean/Proved.lean)

The "Graphs and IR" guide gives the full picture of the different graph representations and how
they relate.

## Bound propagation (IBP/CROWN)

- Graph payloads and passes: [CROWN graph API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/CROWN/Graph.lean)
- Operators and transfer rules: [CROWN operators source tree](https://github.com/lean-dojo/TorchLean/tree/main/NN/MLTheory/CROWN/Operators/)
- Model-specific helpers: [CROWN models source tree](https://github.com/lean-dojo/TorchLean/tree/main/NN/MLTheory/CROWN/Models/)
- Certificate-side definitions: [CROWN cert source tree](https://github.com/lean-dojo/TorchLean/tree/main/NN/MLTheory/CROWN/Cert/)
- Soundness proofs: [CROWN proof source tree](https://github.com/lean-dojo/TorchLean/tree/main/NN/MLTheory/CROWN/Proofs/)

## CLI registry

- Unified entrypoint: [verification CLI API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/CLI.lean)
  - This is what `lake exe verify` runs.

## Certificate checkers (JSON compare)

- [certificate checker API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/Cert.lean): checks externally produced JSON artifacts by
  Lean recomputation with explicit tolerance comparisons.

## Curated examples

- [NN/Examples/Verification](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/Verification/)
  - Small bundled fixtures and tutorial checkers registered into the `verify` CLI.

## PINN and ODE application code

- PINN helpers and CLIs: [PINN API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/PINN.lean)
- ODE support: [ODE API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/ODE.lean) and
  [ODE proof source tree](https://github.com/lean-dojo/TorchLean/tree/main/NN/Proofs/Verification/ODE/)

## Lyapunov / controller workflows

- [Lyapunov oracle boundary](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/CROWN/Lyapunov/Oracle.lean) and
  [two stage Lyapunov runner](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/CROWN/Lyapunov/TwoStage/Run.lean): oracle style
  and two stage controller / Lyapunov developments.

# Trust Boundaries (What Is Executable vs Proved vs Assumed)

TorchLean marks three different kinds of claim explicitly:

## 1) Executable in Lean (Fast Path)

- IR evaluation ([NN.IR.Semantics API](https://github.com/lean-dojo/TorchLean/blob/main/NN/IR/Semantics.lean)) is executable.
- IBP/CROWN passes are executable (compute boxes/affine forms).
- Many demos run on an executable float backend (`IEEE32Exec` by default).

This tier is excellent for debugging and for checkable artifacts, but execution alone does not
establish why a bound is sound.

## 2) Proved Soundness (For a Supported Fragment)

TorchLean has a soundness proof development for a supported subset of the graph dialect:

- Proof entrypoint: [graph certificate soundness API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/CROWN/Proofs/GraphCertSoundness.lean)

Informal theorem shape:

> If every node in the graph is in the supported fragment and each local IBP transformer encloses
> the op semantics, then the propagated boxes enclose the denotation of the whole graph (by
> induction over node ids / topo order).

This is the standard "local soundness + induction over a DAG" structure that scales as more ops
are added.

Important caveat:
Some operator enclosures, especially for transcendentals, are implemented as heuristics
and are explicitly marked as not enclosure sound. For a proof quality enclosure layer, prefer the
operators covered by the soundness theorem and treat the rest as work in progress.

## 3) Imported Artifacts (Externally Produced Certificates)

When a certificate produced by an external verifier such as α,β-CROWN is imported, Lean can still:

- parse it,
- check it is *structurally consistent* (dimensions, ids, declared property form),
- and connect it to downstream theorems with the stated producer assumptions.

Lean replays α,β-CROWN only when the computation is exported in a checkable form and verified inside
Lean.

TorchLean is explicit about what is proved in Lean and what is checked as an external artifact:

- If a verifier recomputes bounds inside Lean over an executable scalar backend, then the remaining
  runtime agreement is that the scalar backend matches its intended semantics.
- With an imported external certificate, Lean can check structural consistency; validating the bound
  computation requires the computation itself to be exported and checked as well.

The certificate guide documents the current minimal JSON leaf certificate format (v0.1) used in the
Two-Stage path.

# Informal theorem shapes (for citations)

This section stays informal. It records the main guarantee shapes that papers and talks often
need, with pointers to the precise Lean theorems.

## Compiler-to-IR Alignment (Proved Fragment)

Lean source: [TorchLean compile proof fragment](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/TorchLean/Proved.lean)

In words (forward fragment only):

> Compiling a forward program to IR preserves meaning:
> evaluating the compiled IR graph equals evaluating the forward program's spec evaluator.

In code, look for the compile-forward theorem in the `Correctness` namespace. Its name starts with
`evalCompiledForward1` and ends with `eq_evalForward1`.

## IBP Soundness Over the Graph Dialect (Supported Subset)

Lean source: [graph certificate soundness](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/CROWN/Proofs/GraphCertSoundness.lean)

In words:

> If `runIBP` produces boxes for all nodes in a supported graph, then for any input `x` in the
> seeded input box, the true denotation of every node lies within its IBP box.

This is the main "interval bounds cover true values" theorem surface for the proved subset. The Lean source
to cite is [graph certificate soundness](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/CROWN/Proofs/GraphCertSoundness.lean).

## Informal soundness statement (the one-liner)

For a supported graph fragment, the central IBP soundness claim has this informal shape:

Assume `g : NN.IR.Graph` is well formed and uses only supported ops, `ps : ParamStore` provides the
required payloads and input seed box, every local transformer used by `runIBP` encloses the
corresponding IR op semantics, and the scalar backend matches its intended meaning. If
`st := runIBP g ps` produces per node interval boxes, then every input in the seed box evaluates to
values that lie inside the corresponding `ibpBox st i` for each node `i`.

The actual proof in the repo is a standard "local soundness + induction over node ids" argument
implemented in
[NN.MLTheory.CROWN.Proofs.GraphCertSoundness API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/CROWN/Proofs/GraphCertSoundness.lean).

## Exact theorem names worth citing

Theorem names that are often cited first:

```
#check NN.MLTheory.CROWN.Graph.CertSoundness.cert_encloses_semantics
#check NN.MLTheory.CROWN.Graph.CrownCertSoundness.crown_checker_encloses_semantics
#check NN.MLTheory.CROWN.Graph.AlphaCrownTransferSoundness.alphaCrown_transfer_sound
#check NN.MLTheory.CROWN.Graph.AlphaCrownTransferSoundness.alphaBetaCrown_transfer_sound
```

Interpreted as a progression:

- `cert_encloses_semantics` is the IBP enclosure theorem for the supported graph dialect;
- `crown_checker_encloses_semantics` is the corresponding CROWN checker theorem;
- `alphaCrown_transfer_sound` and `alphaBetaCrown_transfer_sound` are the transfer-soundness
  theorems for connecting external α-CROWN / α,β-CROWN style artifacts to the Lean semantics.

## Certificate Checking (External Artifact Boundary)

Reference declarations:
- [certificate checkers](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/Cert.lean)
- [verification CLI](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/CLI.lean)

In words:

> If Lean accepts a leaf certificate artifact, then the leaf's stated prune/verified condition
> holds under the certificate's declared semantics and any explicit oracle assumptions.

This is the intended shape for integrating external runs without treating them as already proved.

Two theorems that line up compilation and evaluation:

```
#check NN.Verification.TorchLean.Proved.compileForward1_wellFormed
#check NN.Verification.TorchLean.Proved.evalCompiledForward1_eq_evalForward1
```

Related manual pages: *Graphs and IR* (IR definition), *Floating-Point Semantics* (scalar backends),
*Certificates* and *Two-Stage Workflows* (α,β-CROWN artifacts).

# References

Bound propagation / LiRPA family:

- IBP: Gowal et al., "On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models" (ICCV 2019). https://arxiv.org/abs/1810.12715
- CROWN: Zhang et al., "Efficient Neural Network Robustness Certification with General Activation Functions" (NeurIPS 2018). https://arxiv.org/abs/1811.00866
- DeepPoly / ERAN-style abstract interpretation: Singh et al., "An Abstract Domain for Certifying Neural Networks" (POPL 2019). https://arxiv.org/abs/1812.02648
- LiRPA on general computational graphs: Xu et al., "Automatic Perturbation Analysis for Scalable Certified Robustness and Beyond" (NeurIPS 2020). https://arxiv.org/abs/2002.12920
- β-CROWN / α,β-CROWN: Wang et al., "Beta-CROWN: Efficient Bound Propagation with Per-neuron Split Constraints ..." (NeurIPS 2021). https://arxiv.org/abs/2103.06624

Related robustness-verification background:

- Wong and Kolter, "Provable defenses against adversarial examples via the convex outer adversarial polytope" (ICML 2018). https://arxiv.org/abs/1711.00851

Ecosystem tool TorchLean interoperates with:

- α,β-CROWN codebase: https://github.com/Verified-Intelligence/alpha-beta-CROWN

Application and benchmark references:

- PINNs:
  Raissi, Perdikaris, and Karniadakis, "Physics-informed neural networks" (JCP 2019).
  https://arxiv.org/abs/1711.10561
- ODE enclosure workflow inspiration:
  Tanaka and Yatabe, learn-and-verify style ODE enclosures, arXiv:2601.19818.
  https://arxiv.org/abs/2601.19818
- VNN-COMP ecosystem:
  competition pages and benchmark suite material.
  https://sites.google.com/view/vnncomp
