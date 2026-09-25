import VersoManual
import NN.API
import NN.Spec.Core.Context.Real
import NN.Proofs.Gradients.Activation
import FloatLib
import FloatLib.Floats.Formats.Flocq.Theory.Error.Directed
import NN.MLTheory.CROWN.Core
import NN.Runtime.PyTorch.Import.Core
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open FloatLib.Numerics
open FloatLib.Floats.Formats.Flocq
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

#doc (Manual) "Conclusion" =>
%%%
tag := "conclusion"
file := "Where-The-Pieces-Meet"
%%%

Return to the regression model. After training, its parameter tuple $`\theta` contains weight
matrices
$`W_1,W_2` and bias vectors $`b_1,b_2`:

$$`
F_\theta(x)
=
W_2\operatorname{ReLU}(W_1x+b_1)+b_2.
`

We can save those parameters, evaluate the network, or ask Lean for a bound on its output.
For that bound to describe the saved model's predictions, it must use the same weights and input
domain. We must also connect the arithmetic in the theorem to the arithmetic used in execution:
the formula can denote a real-valued function, a FloatLib binary32 computation, or a native
runtime calculation.

Following a prediction back to the model gives us a sequence of objects to check:

```
typed model
  -> initialized parameters
  -> executable graph and derivatives
  -> explicit scalar and backend semantics
  -> verifier input
  -> checked proposition
```

Shapes constrain composition, parameter artifacts identify the function being evaluated, and
numerical semantics determine how operations round. At the last step, a checker soundness theorem
states what follows from acceptance. I'll use a shifted ReLU to make these obligations small
enough to inspect together.

# Verification Evidence

Over the reals, the shifted ReLU below is flat at 1 for inputs at most 1 and grows with slope 2
above that point.
It lets us compare a universal theorem, an evaluation, a membership check, and the theorem that
relates acceptance to membership on the same function.

```lean (name := ccDef)
-- Keep one expression fixed while choosing its scalar
-- semantics at each use.
/-- The closing example: a shifted ReLU and a readout. -/
def ccScalar {α : Type} [TorchLean.Storage α] [Context α]
    (x : α) : α :=
  2 * max 0 (x - 1) + 1
```

The definition is generic in the scalar type, as in {ref "twostage"}[the two-stage chapter]. At
$`\mathbb R` it denotes a mathematical function; at `Float` it can execute on the host; at
FloatLib binary32 it executes within a binary32 model. `Context` supplies the operations in each
case.
Sharing the definition keeps the expression aligned, while the scalar instance determines the
meaning of its arithmetic.

## Real-Valued Output Bound

Over the reals, every output is at least 1:

```lean (name := ccThm)
-- Prove a lower bound for every real input from the
-- nonnegativity of the ReLU term.
/-- The readout never drops below its bias. -/
theorem ccLowerBound (x : ℝ) : 1 ≤ ccScalar x := by
  have h : (0 : ℝ) ≤ max 0 (x - 1) := le_max_left _ _
  simp only [ccScalar]
  linarith
```

The `have` records that the maximum is nonnegative. Multiplying by 2 preserves that inequality,
and adding 1 gives the bound. After unfolding `ccScalar`, `linarith` checks the remaining linear
inequality.

Because `x` is arbitrary, the result covers both branches and the threshold itself. The kink
causes no problem for this lower bound: we never asked for a derivative there. The arithmetic in
the statement is exact real arithmetic; relating it to an execution will require a numerical
argument.

## Float Evaluation

```lean (name := ccEvalHigh)
-- Above the threshold one, the shifted input is positive
-- and the readout grows linearly.
#eval ccScalar (3.5 : Float)
```

```leanOutput ccEvalHigh (whitespace := lax)
6.000000
```

```lean (name := ccEvalLow)
-- Below the threshold one, ReLU contributes zero and only
-- the readout bias remains.
#eval ccScalar (0.5 : Float)
```

```leanOutput ccEvalLow (whitespace := lax)
1.000000
```

At 3.5, the shifted input is 2.5, giving $`2\cdot 2.5+1=6`. At $`0.5`, the shifted input is
$`-0.5`, so ReLU returns zero and the output is 1. These evaluations exercise both branches at
particular host `Float` inputs; they do not extend the real-valued theorem to every floating-point
input.

## Box Membership Checks

Give an output box and ask whether one computed value lies inside it. This checks point membership;
it does not establish an enclosure for all inputs in an input region:

```lean (name := ccBoxDef)
-- This is a window for one output coordinate, not a region
-- of possible model inputs.
open NN.MLTheory.CROWN in
/-- An output window the value at `3.5` should satisfy. -/
def ccBox : Box Float [1] :=
  { lo := Tensor.from #[5.5], hi := Tensor.from #[6.5] }
```

```lean (name := ccBoxTrue)
-- Check that the computed output six satisfies both
-- endpoint comparisons.
open NN.MLTheory.CROWN in
#eval ccBox.containsDecBool
  (Tensor.from #[ccScalar (3.5 : Float)])
```

```leanOutput ccBoxTrue (whitespace := lax)
true
```

```lean (name := ccBoxFalse)
-- Keep the box fixed and change the input so its computed
-- output falls below the window.
open NN.MLTheory.CROWN in
#eval ccBox.containsDecBool
  (Tensor.from #[ccScalar (0.5 : Float)])
```

```leanOutput ccBoxFalse (whitespace := lax)
false
```

The first result, 6, lies in $`[5.5,6.5]`; the second, 1, does not. The second result still
satisfies the real lower bound proved above. We asked a stricter question when we chose this
output window.
Testing both catches a checker
that returns `true` unconditionally on these cases. Larger certificate workflows also test
malformed evidence; see {ref "certificates"}[the certificate chapter].

## Checker Soundness

To derive a proposition from acceptance, we need the checker's soundness theorem:

```lean (name := ccSound)
-- Expose the hypotheses needed to turn an acceptance
-- equality into logical containment.
open NN.MLTheory.CROWN in
#check @Box.containsDecBool_sound
```

```leanOutput ccSound (whitespace := lax)
@Box.containsDecBool_sound : ∀ {α : Type} [inst : Storage α]
  [inst_1 : Context α]
  [inst_2 : DecidableRel fun x1 x2 => x1 ≤ x2] {s : Shape}
  (b : Box α s) (x : Tensor α s),
  b.containsDecBool x = true → b.contains x
```

Read `α` as the scalar representation and `s` as the common tensor shape. `Storage` and
`Context` supply the required tensor and scalar operations; `DecidableRel` supplies a decision
procedure for the scalar relation `≤`. The explicit arguments are the box `b` and the particular
tensor `x`. Finally, the arrow takes evidence that the Boolean check equals `true` and returns
`b.contains x`: every coordinate lies between its matching lower and upper entries. No graph,
network, or input region occurs in this statement. Applying it to a network output does not add
a universal quantifier over that network's inputs.

A printed `true` records the runtime observation. To apply this theorem inside Lean, we must
supply a kernel-checked proof of the acceptance equality.

The pinned Lean version includes logical models of core `Float` arithmetic. For example, this
comparison can be established inside the kernel:

```lean (name := ccFloatModel)
-- Prove a scalar comparison using logical Float operations;
-- no packed tensor is traversed.
example : ((0.1 + 0.2 : Float) == 0.3) = false := by
  decide
```

Here `decide` resolves the closed scalar proposition using Lean's logical definitions. The
`ccBox` expression also traverses packed tensor storage, and `decide` does not close its acceptance
equality with the current instances. We have observed acceptance for that example but have not
constructed the witness needed by the theorem. Agreement between the logical scalar model and
its native overrides is a further runtime obligation.

## Evidence Scope

:::table +header
*
  * evidence
  * covers
  * does not cover
*
  * theorem at `ℝ`
  * every input, exactly
  * rounding, kernels, devices, data
*
  * `#eval` at `Float`
  * one input, on this host
  * every other input; other arithmetics
*
  * checker returning `true`
  * observed membership of the supplied value in the supplied box
  * an enclosure over an input region; a formal acceptance proof
*
  * soundness theorem
  * every accepting run of that checker
  * whether the checker was run on the object you care about
:::

A certificate proof needs both the soundness theorem and a proof that the particular certificate
was accepted. A runtime comparison additionally needs to identify the artifact and implementation
that produced the observed result. {ref "verification"}[The verification overview] and
{ref "proof-systems-beyond-bounds"}[Proof Systems] develop these connections for
larger examples.

# Example Commands

These commands exercise the tensor, differentiation, graph, arithmetic, and certificate interfaces
used in that chain:

```terminal
# Compare tensor shapes and scalar representations.
lake exe torchlean quickstart_tensors
# Differentiate a tensor function and a model loss.
lake exe torchlean quickstart_autograd
# Train the small model from a stated seed and update
# budget.
lake exe torchlean quickstart_mlp \
  --device cpu --steps 200 --seed 2026
# Exercise eager execution of a graph-based model on CPU.
lake exe torchlean graphspec --device cpu --execution eager
# Compare evaluation and bounds attached to one operation
# graph.
lake exe torchlean one_semantic_universe
# Compare native Float32 values and derivatives with the
# executable reference.
lake exe torchlean float32_semantics
# Replay numerical evidence, including cases the checker
# must reject.
lake exe torchlean numerical_certificate
```

The first two stay close to concrete tensors. `quickstart_tensors` prints the same small array under
four element types, which is the `Context` genericity from the previous section made visible at the
command line:

```terminal +output
== Quickstart: tensor basics ==
Rank-zero tensor: 0.500000
Float tensor:    [0.100000, 0.200000, 0.300000, 0.400000]
Rational tensor: [(1 : Rat)/10, (1 : Rat)/5, (3 : Rat)/10,
                  (2 : Rat)/5]
Integer tensor:  [1, 2, 3, 4]
Float32 cast:    [0.100000, 0.200000, 0.300000, 0.400000]
[two longer rank-3 lines elided]
```

The rational entry `(1 : Rat)/10` is exactly one tenth. The `Float` entry is the nearest binary64
value to one tenth, and the `Float32 cast` entry is the nearest binary32 value. The two floating
entries both print `0.100000`, although they represent different numbers; the rational row displays
its exact fraction.

`quickstart_autograd` then differentiates, first a tensor function and then a model loss:

```terminal +output
== Differentiate a tensor function ==
mean(x^2) = 4.666667
d/dx       = [0.666667, 1.333333, 2.000000]

== Differentiate a model loss ==
loss     = 0.769887
gradient = [[1, 2]: [[-0.877432, 1.754865]],
            [1]: [-1.754865]]
```

The first gradient has one entry per input coordinate. The model-loss gradient has a weight row
and a bias, matching the parameters with respect to which we differentiated. The two loss values
belong to different objectives and cannot be compared as measures of model quality.

For $`x=(1,2,3)` the mean of squares is $`\tfrac{14}{3}=4.\overline{6}` and its gradient is
$`\tfrac{2}{3}x`, which is what the second line prints. The model gradient is tagged by parameter
shape, `[1, 2]` for the weight and `[1]` for the bias, because a gradient in this repository is a
shaped object rather than a flat vector; {ref "autograd-walkthrough"}[the autograd walkthrough]
takes that structure apart.

`graphspec` and `one_semantic_universe` then show why lowering matters: the same operation graph can
be evaluated for values or interpreted for bounds.

Run the graph and arithmetic examples directly:

```terminal
lake exe torchlean one_semantic_universe
lake exe torchlean float32_semantics
```

The first evaluates a graph and checks sampled values against an interval. A final $`\tanh`
activation has the general range bound $`[-1,1]`; a useful input-specific enclosure may be tighter.
Sampled membership checks do not establish enclosure for every input, and printing a soundness
theorem's name does not supply an acceptance proof.

The second compares native `Float32` with FloatLib binary32, including forward values, parameter
gradients, and input gradients. Those checks reach both sides of the autograd calculation. Their
scope is the graph and inputs they run; {ref "fp32-soundness"}[the FP32 soundness chapter] develops
the additional argument needed to relate native operations to the reference.

`numerical_certificate` exercises a graph-level checker and negative cases containing malformed
evidence. Its output records which cases accepted or rejected; the associated soundness statement
identifies the proposition obtainable from a formal acceptance witness.

# Validation Commands

The small demonstrations above explain individual ideas. Before relying on a larger change, use the
validation command that reaches the relevant boundary:

```terminal
# Compile and run the curated suite against the CPU CUDA stub.
lake build nn_tests_suite
lake exe nn_tests_suite

# Build with the CUDA-enabled LibTorch SDK, then execute on a device.
export TORCHLEAN_LIBTORCH_HOME=/path/to/libtorch
scripts/lake.sh -R -K cuda=true build nn_tests_suite
CUDA_VISIBLE_DEVICES=0 TORCHLEAN_REQUIRE_CUDA=1 \
  scripts/lake.sh -K cuda=true env ./.lake/build/bin/nn_tests_suite

# Check native kernels for memory, race, and synchronization defects.
scripts/checks/cuda_sanitize_tests.sh \
  --all-tools --cuda-home /usr/local/cuda --skip-build

# Replay the default checked-artifact suite.
lake exe verify -- all

# Check conventions and rebuild the documentation site.
lake lint
scripts/docs/build_site.sh
```

A successful CUDA-enabled build establishes that the selected native sources compile and link.
The runtime test also needs a visible CUDA device to exercise GPU execution. A theorem build checks
Lean declarations, while certificate replay exercises parsers, policy gates, and stored artifacts.
The documentation build checks that its elaborated examples and imports remain valid. Compute
Sanitizer checks the tested binaries with `memcheck`, `racecheck`, `initcheck`, and `synccheck`;
its results are specific to the executions it observes.

Read the coverage of these commands as carefully as their results. `verify -- all` runs ten
sections, while `verify -- list` registers twenty-three tools, so a green `all` leaves thirteen
registered tools untouched, including every `torchlean-*` workflow and the two-stage Lyapunov
pipelines. That is a deliberate choice about runtime, not a claim that the rest passed. The same
applies inside a section: `margin-report` finishes with

```terminal +output
[margin report] examples=360
[margin report] nominal_ok=349 (requires 'pred' in examples)
[margin report] positive_margin=318
```

and exits zero. The report contains 318 positive margins among 360 entries. That count is distinct
from the tool's internal consistency checks, and a successful exit does not establish that all 360
entries have a positive verified margin.

The three counts have different predicates. `examples` counts the entries inspected;
`nominal_ok` depends on the available prediction field; `positive_margin` counts entries meeting
the reported margin condition. A reader cannot infer from these totals alone which individual
entries satisfy both conditions, or why an entry lacks one of them. In particular, exit status
zero means the reporting workflow completed its own checks. It does not replace the application
criterion that every required example, or every point in a specified region, must have a positive
margin.

# Verification Claims

Suppose a colleague sends you a result that says:

> The trained model is robust on a box of inputs.

To assess the claim, the artifact needs to identify:

1. *Which model?* Identify architecture, parameter artifact, and graph payload.
2. *Which input box?* Give lower and upper tensors with checked shapes.
3. *Which robustness property?* State the output inequality or class-margin condition.
4. *Which scalar semantics?* Exact real, finite FP32, executable IEEE32, or a native runtime.
5. *Which method?* IBP, CROWN, α,β-CROWN artifact replay, branch-and-bound, or another checker.
6. *Which theorem?* Name the proposition obtained when the checker accepts.
7. *Which boundary remains trusted?* Identify unproved assumptions about artifact identity, parsing,
   the backend, compiler, or hardware. A soundly checked certificate need not trust the search that
   proposed it.

For example, a certificate could establish that graph `G`, with parameter artifact `P`, has
positive class margin `Q` throughout $`[\mathrm{lo},\mathrm{hi}]` under exact real semantics.
The evidence would include the checker's acceptance proof and the theorem deriving that margin
from acceptance. Deploying a float implementation adds a separate obligation to preserve the
margin after numerical error. Changing `P`, the input box, or the implementation changes which
parts of this argument must be repeated.

For the shifted ReLU, the observed computation is {lean}`(ccScalar : Float → Float)` at input
$`3.5`. The returned value 6 passes `Box.containsDecBool` for {lean}`ccBox`, whose bounds are
$`5.5\le y\le 6.5`. `Box.containsDecBool_sound` would derive membership from a proof of that
acceptance equality in the logical scalar model. This page records the host evaluation but does
not supply that proof for `ccBox`. Connecting native evaluation to the logical model, and
checking that both refer to the same value and box, are additional obligations. The check is
point membership, not a bound over an input region.

## Printed Precision

A six-decimal rendering can hide differences in the stored values:

```lean (name := ccSum)
-- Observe how the default decimal formatter displays the
-- computed sum.
#eval (0.1 + 0.2 : Float)
```

```leanOutput ccSum (whitespace := lax)
0.300000
```

```lean (name := ccSumEq)
-- Compare the stored Float values instead of comparing
-- their displayed decimal strings.
#eval (0.1 + 0.2 : Float) == 0.3
```

```leanOutput ccSumEq (whitespace := lax)
false
```

```lean (name := ccSumGap)
-- Scale the small nonzero difference so the default
-- formatter can expose it.
#eval ((0.1 + 0.2 : Float) - 0.3) * 1e17
```

```leanOutput ccSumGap (whitespace := lax)
5.551115
```

The printed sum matches the printed literal, but the equality check detects a gap of about
$`5.55\times 10^{-17}`, one ULP at that magnitude {Informal.citep goldberg1991}[]. Agreement at
six decimal places therefore establishes only that displayed precision. Equality tests and
numerical differences expose more information, as in {ref "floats"}[the floats chapter].

# Proof Modules

Much of the argument can live entirely in Lean. Shapes and layer composition, mathematical operator
specifications, graph well-formedness, autograd rules, generic rounding, finite FP32 mathematics,
executable IEEE32 reference algorithms, optimizer laws, and checker soundness are all stated as
named definitions or theorems with explicit hypotheses.

To find the statement behind a particular operation, use the corresponding source directory:

:::table +header
*
  * directory
  * what to inspect
*
  * `NN/Proofs`
  * autograd soundness, RL, runtime approximation, analytic gradients
*
  * `NN/Floats`
  * rounded-real binary32 analysis, finite-operation bridges, and real-scale quantization
*
  * FloatLib dependency
  * generic rounding theory, configured executable scalars, and scalar interval arithmetic
*
  * `NN/MLTheory`
  * bound propagation, learning theory, and optimizer mathematics
*
  * `NN/Tensor`
  * shape algebra and tensor representation identities
*
  * `NN/Spec`
  * operator definitions and their local semantic laws
*
  * `NN/IR`
  * graph invariants, shape soundness, and semantic interfaces
*
  * `NN/Verification`
  * artifact checkers and acceptance implications
*
  * `NN/Runtime and NN/API`
  * execution and public interfaces, connected to selected proved models
:::

Start with the declaration supporting your claim and follow its imports and hypotheses. The
existence of many nearby theorems does not prove a runtime implementation, and a thin API module can
still expose a well-supported operation by delegating to a proved specification.

## Theorem Hypotheses

The following declarations illustrate three different sets of hypotheses.

Derivative rules use mathlib's `HasDerivAt`, which states differentiability with a specified
derivative at a point {Informal.citep mathlib2020}[]:

```lean (name := ccRelu)
-- Read the input restriction and the function, derivative
-- value, and point in HasDerivAt.
open Proofs in
#check @relu_deriv_correct
```

```leanOutput ccRelu (whitespace := lax)
relu_deriv_correct : ∀ (x : ℝ), x ≠ 0 →
  HasDerivAt Activation.Math.reluSpec
    (Activation.Math.reluDerivSpec x) x
```

`HasDerivAt` takes three arguments: the function being differentiated, its proposed derivative
value, and the point where that value is claimed. Here they are `reluSpec`, `reluDerivSpec x`,
and `x`. The theorem identifies the derivative specification with the ordinary real derivative
at every nonzero point. For negative inputs that value is zero; for positive inputs it is one.
The condition `x ≠ 0` is an input to the theorem, not a result produced by it. A caller must
establish that condition before using the derivative conclusion.

The hypothesis $`x\neq 0` excludes ReLU's kink, where the left and right derivatives differ.
PyTorch selects 0 at zero for its backward rule {Informal.citep pytorch2019}[]. That convention
can be specified and checked, but it is not an ordinary derivative of ReLU at zero.

FloatLib proves rounding facts generically in the radix and the exponent function. TorchLean
instantiates those facts for its numerical models:

```lean (name := ccRound)
-- Inspect the format and rounding assumptions of the
-- generic one-ULP error bound.
#check @round_abs_error_le_ulp
```

```leanOutput ccRound (whitespace := lax)
@round_abs_error_le_ulp : ∀ {β : Radix} {fexp : ℤ → ℤ} [inst : ValidExp fexp] (rnd : ℝ → ℤ)
  [ValidRnd rnd] (x : ℝ),
  |FloatLib.Floats.Formats.Flocq.round rnd x - x| ≤ ulp β fexp x
```

The radix `β` and exponent function `fexp` determine the representable-number system, while
`rnd` chooses how a real significand is rounded to an integer. The validity instances constrain
those choices; they are not arbitrary functions for which the inequality is assumed to work.
The conclusion bounds the absolute difference between one rounded value and its real input by
one unit in the last place at that input. This theorem permits general valid rounding, so it does
not claim the sharper half-ULP bound associated with nearest rounding. Nor is it by itself an
IEEE overflow or whole-network error theorem.

`ValidExp` supplies laws for the exponent function; `ValidRnd` requires the integer rounder to
be monotone and to fix every integer. Choosing a new format means supplying those laws, after
which the same error theorem applies. This separation of formats and rounding is developed in
{Informal.citet flocq2011}[] and {Informal.citet boldo2015}[];
{ref "floats"}[the floating-point chapter] works through FloatLib's definitions.

The checker bridge has a different hypothesis: a Boolean acceptance equality. Expensive search
can propose evidence outside the kernel, while a sound checker and a kernel-checked acceptance
witness establish the resulting proposition {Informal.citep necula1997}[].

## Runtime Boundaries

Real training crosses boundaries the Lean kernel cannot inspect directly:

:::table +header
*
  * Lean-side object
  * Boundary needed to connect it to a run
*
  * typed tensor or model
  * dataset loader, parameter artifact, and preprocessing
*
  * operation graph semantics
  * model lowering, importer, or compiler
*
  * FloatLib binary32 reference operation
  * CPU/CUDA kernel, compiler, driver, and hardware conformance
*
  * spectral or matrix operation contract
  * cuFFT, cuBLAS, LibTorch, or another selected provider
*
  * checker acceptance theorem
  * acceptance proof, artifact identity, and conformance of the executed checker
*
  * PINN or neural-operator proposition
  * equation encoding, simulator, sampling, and dataset provenance
:::

Kernel capsules record provider, device, shape and layout contracts, numerical policy, derivative
implementation responsibility, and evidence. Artifact parsers and policy gates reject malformed or
inadmissible input. Those mechanisms make the remaining trust visible; wrapping a native library or
Python script in a Lean function would not make it proved.

# Numerical Guarantees

For a numerical claim, choose the representation that retains the information the claim needs:

```
FloatLib.Floats.Formats.Flocq
  generic radix, format, and rounding mathematics; FloatRep and NF

FP32
  binary32-precision, gradual-underflow rounded-real semantics with no upper exponent cutoff

FloatLib.Floats.ExecFloat.Binary e f
  configured executable binary representation and operations
  8 23 gives binary32; 15 112 gives binary128; valid custom widths use the same API
```

The generic theory supports reasoning about spacing and rounding across formats
{Informal.citep flocq2011}[]. Configured values also retain encodings and exceptional values.
To connect either account to a CPU or CUDA run, we need the selected provider's contract.

Our closing function makes the distinction concrete. Evaluate it at binary64 and at the executable
binary32 semantics, and report the difference in units of $`10^{-9}`:

```lean (name := ccSem)
-- Compare the whole binary64 path with input conversion and
-- evaluation in binary32.
/-- The readout at binary64, at binary32, and the gap. -/
def ccSemantics (x : Float) : Float × Float × Float :=
  let binary64 := ccScalar x
  let f32 := (ExecFloat.Binary.toFloat32
    (ccScalar
      (ExecFloat.Binary.ofFloat32 x.toFloat32))).toFloat
  (binary64, f32, (f32 - binary64) * 1000000000)

#eval ccSemantics 1.1
```

```leanOutput ccSem (whitespace := lax)
(1.200000, 1.200000, 47.683716)
```

The tuple records the binary64 answer, the binary32 answer converted back for display, and the
signed difference multiplied by a billion. The binary32 path first converts the input `x`, then
evaluates the function in its chosen arithmetic. Its discrepancy therefore includes input
representation as well as subsequent rounding. It cannot be assigned entirely to the final
addition. Converting the binary32 result to binary64 exposes its value without restoring digits
that the narrower path discarded. The positive last component says this path's answer is slightly
larger for the particular input 1.1.

Both displayed values are `1.200000`, but their difference is about $`4.768\times 10^{-8}`,
roughly four tenths of a binary32 ULP near $`1.2`. The six-decimal display has rounded away the
difference {Informal.citep goldberg1991}[]; comparing these strings would miss it.

FloatLib makes the representation and each arithmetic operation explicit
{Informal.citep boldo2015}[]. The checked example above therefore identifies a particular numerical
computation, whose relation to a native implementation still needs to be established.

Let $`f` be the target function, $`F_{\mathbb R}` the ideal real-valued network, and
$`F_{\mathrm{runtime}}` its runtime implementation. Interpreting both outputs in a common real
space lets us split the total error:

$$`
|F_{\mathrm{runtime}}(x)-f(x)|
\leq
|F_{\mathrm{runtime}}(x)-F_{\mathbb R}(x)|
+
|F_{\mathbb R}(x)-f(x)|.
`

The first term is numerical implementation error. Rounding theorems at the `FP32` or
FloatLib binary32 level can contribute to its bound once the runtime is connected to that formal
semantics. The second is model approximation error, addressed by the real-valued theory in
*Approximation Theory*. Bounds on both terms must use compatible input domains and the same
parameters to produce an end-to-end bound against $`f`.

# Configuration Validation

Shape and configuration checks act at different stages. A typed expression can fail during
elaboration, an external tensor can fail during parsing, and a command can reject an unsupported
flag before starting its computation.

The first is elaboration. A shape error in a matrix product is not a runtime failure; the file does
not compile. Here is a two by three tensor multiplied by itself:

```lean (name := ccMM) +error
-- A two-by-three matrix cannot multiply itself: the
-- contracted dimensions disagree.
def ccA : Tensor Float [2, 3] :=
  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

def ccMismatch : Tensor Float [2, 2] :=
  Tensor.matmul ccA ccA
```

```leanOutput ccMM (whitespace := lax)
Application type mismatch: The last
  ccA
argument has type
  Tensor Float [2, 3]
but is expected to have type
  Tensor Float [3, 2]
in the application
  ccA.matmul ccA
```

The inner matrix dimensions are 3 and 2, so multiplication is ill-typed. The elaborator reports
the incompatible shapes before evaluation.

The second is parsing at a boundary. Shapes crossing from outside cannot be checked by the
elaborator, because the payload is a string when the program is already running. The PyTorch
`state_dict` importer therefore returns an option and we turn it into a message:

```lean (name := ccParseDef)
-- Parse JSON and validate both axis lengths before treating
-- the payload as a 2-by-2 tensor.
/-- Read a JSON payload as a two by two tensor. -/
def ccParse (payload : String) : String :=
  match (Lean.Json.parse payload).toOption.bind
      (Import.PyTorch.parseTensor [2, 2]) with
  | some t => toString t
  | none => "refused: shape does not match [2, 2]"

#eval ccParse "[[1.0, 2.0], [3.0, 4.0]]"
```

```leanOutput ccParseDef (whitespace := lax)
"[[1.000000, 2.000000], [3.000000, 4.000000]]"
```

The wrapper first asks whether the string is valid JSON, then asks whether that JSON has the
nested numeric-array structure required by shape `[2, 2]`. A successful parse produces an actual
tensor whose shape is known to Lean. The wrapper returns its formatted contents only to make the
example easy to inspect. Its failure branch combines malformed JSON and an incompatible tensor
payload into one message, so that message should not be treated as a precise diagnosis of every
possible input error. A production loader can retain more detail from each stage.

A row of the wrong width is refused rather than truncated or zero-padded:

```lean (name := ccParseBad)
-- A rectangular payload can still have the wrong extents
-- for the requested tensor.
#eval ccParse "[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]"
```

```leanOutput ccParseBad (whitespace := lax)
"refused: shape does not match [2, 2]"
```

So is a JSON object where an array was expected:

```lean (name := ccParseObj)
-- Valid JSON syntax is insufficient when the expected
-- tensor encoding is nested arrays.
#eval ccParse "{\"weight\": 1.0}"
```

```leanOutput ccParseObj (whitespace := lax)
"refused: shape does not match [2, 2]"
```

The rejected matrix has two rows but three entries per row, so it fails the inner extent check.
The object containing a `weight` key is syntactically valid JSON but fails a different requirement:
this parser expects the tensor itself, not a state-dictionary wrapper. A caller handling such a
wrapper must select the intended field before asking for a tensor. These checks establish the
shape and encoding of the imported value. They do not establish that its numbers are the intended
checkpoint or that the external model uses the same weight convention.

This wrapper gives the same message for a wrong row width and a wrong JSON kind because both
become `none`. The importer's `Except String` variants retain more detail, allowing callers to
distinguish missing keys, invalid JSON kinds, and incompatible shapes.

The third is the command line. Flags are parsed strictly, so an unknown flag stops the example
rather than being ignored:

```terminal +output
$ lake exe torchlean quickstart_tensors --show-backend
error: quickstart_tensors: unexpected arguments: [--show-backend]
```

Rejecting unknown flags prevents a requested device or arithmetic mode from being silently
ignored. This message identifies the unexpected arguments; the command's help lists supported
options.

# Backend Contracts

Large models rely on specialized matrix multiplication, convolution, attention, FFT, and
communication libraries. TorchLean can specify those operations and dispatch to a provider, with
an explicit contract for the values and derivatives that provider returns.

The graph fixes the computation, while the provider implements its numerical operations:

```
one semantic operation graph
  -> several admissible kernel providers
  -> explicit contract and evidence per boundary
```

TorchLean records graph structure, shapes, loss, optimizer meaning, and proof statements while a
provider supplies a fast value or a local VJP. The assurance level ranges from a proved internal
implementation to a checked or explicitly trusted external kernel, and the capsule says which.

Examples that dispatch backend operations can report their contracts with `--show-backend`:

```terminal +output
$ lake exe torchlean quickstart_mlp --steps 3 --show-backend
== Quickstart: simple MLP training (seed=0, steps=3) ==
...
  matmul: reference.matmul provider=reference trust=checked
      vjp=torchlean-tape reduction=fixed-left
    shape: shape safety for matmul; guarded at runtime by portable
      runtime shape checks
    layout: canonical-tensor layout compatibility for matmul;
      guarded at runtime by typed tensor layout
    value: matmul forward refines its TorchLean semantics; covered
      by test suite NN.Tests.Runtime.Floats.Suite
    vjp: matmul torchlean-tape VJP refines its TorchLean semantics;
      covered by test suite NN.Tests.Runtime.Floats.Suite
```

Long lines are wrapped here; the tool prints one line per claim. `provider=reference` identifies
the selected implementation, and `trust=checked` records test evidence. `reduction=fixed-left`
specifies the summation order, which can affect floating-point results. The shape claim cites
runtime checks, while the layout claim cites the typed tensor layout. The value and VJP claims
name `NN.Tests.Runtime.Floats.Suite` as their evidence; these entries do not name refinement
theorems.

The capsule makes it possible to inspect evidence per operation. Strengthening one entry from
test coverage to a proof requires a theorem for the stated provider, semantics, and numerical
policy.

The flag belongs to examples that perform backend dispatch. For example,
`quickstart_tensors --show-backend` rejects it because that example does no dispatch. An unsupported
flag is a usage error, not evidence that a different backend ran.

# Scientific ML

A PINN or neural operator usually participates in a larger chain:

```
equation and domain
  -> discretization or simulator
  -> dataset
  -> model and training
  -> prediction artifact
  -> residual, invariant, or error certificate
  -> Lean checker and theorem
```

The neural network is only one part. Boundary conditions, quadrature, sampling coverage, simulator
accuracy, and interpolation between grid points can dominate the final claim. A residual that is
small at the collocation points says nothing about the points in between unless something else
supplies that step, and in the Fourier neural operator case the discretization is part of the
operator rather than an implementation detail {Informal.citep fno2021}[].

A checker soundness theorem can connect an accepted artifact to a precise scientific claim.
The external search or training process need not be repeated to validate a sufficient certificate
{Informal.citep necula1997}[]. The resulting guarantee still depends on the encoded equation,
domain, and assumptions matching the scientific problem.

# Operation Development

When adding an operation, define its tensor shapes and scalar semantics, then implement the
runtime forward computation and derivative rule. Add an IR representation if the operation must
participate in export, lowering, or graph verification.

Each provider needs a declared layout, numerical policy, and evidence level. Reusable semantic
facts belong in theorems; rejection behavior also needs executable checks. A bad shape, nonfinite
parameter, unsupported policy, or malformed artifact should fail for the documented reason.
Recorded negative cases make that behavior reviewable.

Running a shared model through each claimed path checks whether its definition, derivative rule,
lowering case, and dispatch agree on concrete inputs. Property-based and structure-aware fuzzing
explore a wider range of such interactions {Informal.citep tensorfuzz2019}[]
{Informal.citep nnsmith2023}[]. These executions complement the declarations checked by Lean;
a small example suite does not establish the same coverage as a fuzzing campaign.

Proofs establish propositions about formal objects under their stated hypotheses. Tests exercise
wiring, FFI, builds, command lines, documentation, and platform behavior that those propositions
may not cover. Validation should include the paths an application actually uses.

# Application Requirements

Start from the property your application needs, then follow its hypotheses through the model,
arithmetic, and selected runtime. The earlier chapters give the relevant contracts beside the
operations and checkers that use them. For example, a classifier needs a positive verified margin;
an ODE candidate needs successful certificate replay over its stated domain. A training loss alone
does not establish either property.

Use the public tensor API for numerical values and select a runtime with an implemented provider.
Run the corresponding verification workflow and backend checks for your deployment. The command
registry identifies which workflows `verify -- all` includes; additional workflows can be invoked
individually.

# Model Review Exercise

Choose one example from the model collection and write down:

```
input and output shapes
parameter shapes
loss
data source and preprocessing
arithmetic
execution mode
device and selected providers
forward and backward implementation responsibility
available theorem or checker
remaining trusted assumptions
```

For an example that supports backend reporting, compare this record with `--show-backend`.
The report identifies provider, trust level, reduction order, and evidence for dispatched
operations. Loss definition, data provenance, and application assumptions must come from the model
and workflow artifacts.

Compare the shifted ReLU in this chapter with a training workflow such as `chargpt` or
`fno1d_burgers`. The former has a theorem over every real input but no training artifact; the
latter adds parameter state, data, optimization, and backend choices. Identify which component
theorems apply and which claimed connections still rely on tests or assumptions.

For a result built with TorchLean {Informal.citep torchlean2026}[], record the exact model and
artifact, the proposition established, and the evidence connecting that proposition to the
selected runtime. Those identifiers let a later change to parameters, preprocessing, or backend
be traced to the guarantees that need to be checked again.
