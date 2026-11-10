import VersoManual
import TorchLean.NN.LoRA
import TorchLean.NN.Autograd.Function
import TorchLean.NN.Autograd.Frontend
import TorchLean.NN.Autograd.Differential
import TorchLean.NN.Autograd.Complex
import TorchLean.NN.CLI.Training.Command
import TorchLean.NN.Data.Synthetic
import TorchLean.NN.Data.CausalLM
import TorchLean.NN.Models.KAN
import TorchLean.NN.Models.CNN
import TorchLean.NN.Models.ResNet
import TorchLean.NN.Models.UNet
import TorchLean.NN.Models.Recurrent
import TorchLean.NN.Models.CausalTransformer.LanguageModel
import TorchLean.NN.Models.Mamba
import TorchLean.NN.Models.MaskedPatchReconstructor
import TorchLean.NN.Models.Diffusion.NoisePredictor
import TorchLean.NN.Models.FNO
import TorchLean.NN.Models.PPO
import TorchLean.NN.Builder.MoE
import TorchLean.NN.Sequential.Training
import TorchLean.NN.RL.CLI
import TorchLean.NN.RL.PPO.ActorCritic
import TorchLean.NN.SelfSupervised
import TorchLean.NN.Text.Vocabulary
import TorchLean.NN.Text.GPT2BPE
import TorchLean.NN.Text.Generation
import TorchLean.NN.Trainer.Constructor
import TorchLean.NN.Trainer.Train
import TorchLean.NN.Data.Image
import TorchLean.NN.Optimizer.AdamW
import TorchLean.NN.Optimizer.TensorOptimizer
import TorchLean.NN.Optimizer.GD
import TorchLean.NN.Optimizer.GD.Convergence
import TorchLean.NN.Optimizer.GD.StrongConvexity
import TorchLeanExamples.DeepDives.Muon
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles
import TorchLean.NN.Loss
import TorchLean.NN.Session.Random
import TorchLean.Verify.Trainer

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean

open TorchLean
open TensorOptimizer
open Muon NewtonSchulz QR
open GD FirstOrderStrongConvex

-- `lax` whitespace on the expected-output blocks lets a long printed signature be
-- rewrapped to fit this file's line budget without changing what is being asserted.
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Optimization Theory" =>
%%%
tag := "optimization-theory"
%%%

Training code executes updates. Optimization theory explains when those updates should make
progress.

A TorchLean training run may call SGD or Adam, but a convergence theorem can't come from the name
of the optimizer alone: it needs an ideal update map, assumptions on the objective or gradient, and
a step size condition.

Even two updates that print the same parameter can differ: Adam's epsilon changes its recurrence
by less than the six-decimal display reveals. We can expose that difference on a single parameter
before asking which identities hold for arbitrary states.

# The Optimization Contract

TorchLean's optimization contract has three layers:

- *Runtime update*: the concrete operation that writes new values into parameter tensors, such as
  one SGD, momentum, or Adam-style step.
- *Ideal update*: the mathematical map the runtime update is intended to approximate, for example
  $`x\mapsto x-\eta g(x)`.
- *Convergence theorem*: the conditional theorem saying that iterating the ideal map makes progress
  under assumptions such as strong monotonicity, Lipschitzness, and a safe step size.

A decreasing loss curve describes one run. A contraction theorem compares the update at arbitrary
pairs of states, so it needs more information than that curve supplies.

# First Order Updates

We start with a first order optimizer state: parameters, gradients, an optional buffer, and a time
counter. Its executable equation belongs in the definition of the update itself, and TorchLean
doesn't add a theorem merely to restate each record field after reduction. The theory layer begins
when two distinct updates are related.

## Scalar Optimizer Comparison

Start a single parameter at $`1`, supply the constant gradient $`1` at every step, and use a
learning rate of $`0.1`. Holding the gradient fixed isolates the effect of the update rule and its
state: any difference between the runs comes from momentum, adaptive normalization, or decay.

```lean (name := otRuns)
-- Feed three identical gradients through each optimizer
-- while retaining its state.
def otParams : Tensor Float [1] := [1.0]

def otGrad : Tensor Float [1] := [1.0]

/-- Three steps of `opt` from `otParams`, handing the same
gradient to the optimizer every time. -/
def otRun (opt : TensorOptimizer Float) :
    Tensor Float [1] :=
  (TensorOptimizer.runSteps opt
      { optimizerState := opt.init otParams,
        parameters := otParams }
      #[otGrad, otGrad, otGrad]).parameters

#eval otRun (TensorOptimizer.sgd 0.1)

#eval otRun (TensorOptimizer.momentumSGD 0.1 0.9)

#eval otRun (TensorOptimizer.adam 0.1 0.9 0.999 1e-8)

#eval otRun (TensorOptimizer.adamw 0.1 0.0 0.9 0.999 1e-8)

#eval otRun (TensorOptimizer.adamw 0.1 0.5 0.9 0.999 1e-8)
```

```leanOutput otRuns (whitespace := lax)
[0.700000]
```

```leanOutput otRuns (whitespace := lax)
[0.439000]
```

```leanOutput otRuns (whitespace := lax)
[0.700000]
```

```leanOutput otRuns (whitespace := lax)
[0.700000]
```

```leanOutput otRuns (whitespace := lax)
[0.572125]
```

You can check each number without a computer: plain SGD subtracts $`0.1` three times, giving
$`0.7`. Momentum accumulates the buffer $`1`, then $`1.9`, then $`2.71`, and subtracting $`0.1`
times each of those from $`1` gives $`0.439`. Adam divides the gradient by the square root of its
bias-corrected second moment plus epsilon. With this constant gradient the update is close to
$`0.1`, so the result rounds to $`0.7` at six decimals (epsilon keeps it from being exact). AdamW
at zero weight decay is the same run again, while AdamW at weight decay $`0.5` differs from Adam
because it subtracts $`\eta\lambda p` from the parameter on top of the Adam step.

Because of the buffers, the parameter alone isn't enough to resume these runs. At the third
momentum step the update uses the accumulated value $`2.71`, although the supplied gradient is
still $`1`. Restarting momentum from its zero buffer at the same parameter would instead subtract
$`0.1`. Adam carries two buffers and a step count because bias correction depends on how many
updates those buffers have seen. `otRun` passes the entire `optimizerState` forward through the
array, so the comparison includes that history.

For the decay example, ignoring epsilon's small correction gives the recurrence
$`p_{t+1}=0.95p_t-0.1`, whose first three values are $`0.85`, $`0.7075`, and $`0.572125`. Each
step's shrinkage acts on the current parameter: it isn't one subtraction of $`0.05` repeated
independently of the parameter.

We recorded the PyTorch comparison in double precision to match Lean's `Float` format
{Informal.citep pytorch2019}[]:

```
SGD                     [0.7000000000000001]
SGD momentum 0.9        [0.43900000000000006]
Adam                    [0.7000000030000006]
AdamW weight_decay=0    [0.7000000030000006]
AdamW weight_decay=0.5  [0.5721250028525007]
Adam  weight_decay=0.5  [0.7003815249719783]
```

The first five lines agree at the displayed precision, but the sixth uses coupled decay, which
`TensorOptimizer.adam` doesn't expose. PyTorch's `Adam(weight_decay=0.5)` adds $`\lambda p` to the
gradient before the moment estimates see it, so the decay is then divided by the adaptive
denominator along with everything else. Three steps of coupled decay leave the parameter at
$`0.7003815`, whereas decoupled decay reaches $`0.5721250`. Adam's adaptive normalization changes
how an L2 gradient penalty affects the step. Here the public tensor Adam API exposes no
weight-decay argument, while AdamW exposes separate decoupled decay {Informal.citep adamw2019}[].

TorchLean shows six decimals, and $`0.7000000030` rounds to `0.700000` at that precision. To see
the difference the display hides, the checks below compare TorchLean Adam with its own zero-decay
AdamW and measure the residual from `0.7` (they don't compare PyTorch bits):

```lean (name := otAdamEq)
-- Compare full values to expose the Adam difference hidden
-- by six-decimal printing.
def otAdamResult : Tensor Float [1] :=
  otRun (TensorOptimizer.adam 0.1 0.9 0.999 1e-8)

def otAdamwResult : Tensor Float [1] :=
  otRun (TensorOptimizer.adamw 0.1 0.0 0.9 0.999 1e-8)

#eval otAdamResult == otAdamwResult

#eval otAdamResult[0] == 0.7

#eval (otAdamResult[0] - 0.7) * 1000000000.0
```

```leanOutput otAdamEq (whitespace := lax)
true
```

```leanOutput otAdamEq (whitespace := lax)
false
```

```leanOutput otAdamEq (whitespace := lax)
3.000001
```

Adam and AdamW at zero decay compare equal numerically on this input. Neither is equal to $`0.7`:
the residual is about $`3\times 10^{-9}` (consistent with the scale of the PyTorch residual shown
above). In the denominator, $`\varepsilon = 10^{-8}` causes the leading difference: even in exact
arithmetic, each Adam step is slightly smaller than the SGD step. Floating arithmetic adds its own
rounding error. So the discrepancy from `0.7` isn't solely a rounding residual.

## Adam And Zero-Decay AdamW

`update_params_eq_adam` (in the `AdamW` namespace) is the
real-valued counterpart of the Adam/zero-decay AdamW agreement. It quantifies over the state and
parameters for a single update. Extending the equality across steps also requires relating the
evolving optimizer states.

```lean (name := otAdamwThm)
-- Compare one AdamW step with the corresponding Adam state
-- when decay is zero.
open AdamW in
#check @update_params_eq_adam
```

```leanOutput otAdamwThm (whitespace := lax)
@update_params_eq_adam : ∀ {s : Shape} (state : AdamW.State ℝ s)
  (parameters gradients : Tensor ℝ s), state.weightDecay = 0 → (AdamW.update state parameters
  gradients).parameters = (Adam.update state.toState parameters gradients).parameters
```

`state.toState` is the Adam state that the AdamW state extends: the same hyperparameters, moments,
step counter and bias-correction powers, without the decay field. That tells us which two updates
are compared: same parameters, same gradients, same moment history, same hyperparameters, with
AdamW's decay field set to zero. With the braces in `{s : Shape}`, Lean can infer the tensor shape
from those arguments. Since the statement quantifies over `state`, it applies to an intermediate
state as well as an initialized one, provided the zero-decay equality is available.

Here the scalar type is `ℝ`, and the proof simplifies the decay term using algebraic laws such as
`x * 0 = 0`, which fail for IEEE infinities and NaNs. In floating point, the run above checks
agreement on one finite input, but the real theorem doesn't establish it for arbitrary floating
states. That gap between the two claims gets its own bounds in the chapter on
{ref "runtime-approximation"}[runtime approximation].

The conclusion is about `.parameters`, not equality of the whole state or the loss curve of a
training run. Generalization would require a separate statistical theorem.

## Optimizer State And Update Laws

For tensor optimizers with richer state, the
{src "TorchLean/NN/Optimizer/TensorOptimizer.lean"}[optimizer laws API] exposes the same
pattern. An optimizer is a record with a state type, an initializer, and an update, so "which
optimizer" is a value that theorems can quantify over, not a string in a config file:

```lean (name := otLaws)
-- Inspect state-preserving stream composition and the
-- generic recurrence bridge.
#check @TensorOptimizer.sgd
#check @TensorOptimizer.adamw
#check @TensorOptimizer.runSteps_append
#check @StepLaw
#check @StepLaw.runSteps_eq
```

```leanOutput otLaws (whitespace := lax)
@sgd : {α : Type} → [inst : Storage α] → [inst_1 : Context α] →
  α → TensorOptimizer α
```

```leanOutput otLaws (whitespace := lax)
@adamw : {α : Type} → [inst : Storage α] → [inst_1 : Context α] → [DecidableRel
  fun x1 x2 => x1 > x2] → α → α → α → α → α → TensorOptimizer α
```

```leanOutput otLaws (whitespace := lax)
@runSteps_append : ∀ {α : Type} [inst : Storage α] [inst_1 : Context α] (opt :
  TensorOptimizer α) {s : Shape} (current : OptimizerStep α s (opt.State s)) (left right :
  Array (Tensor α s)), opt.runSteps current (left ++ right) = opt.runSteps (opt.runSteps current
  left) right
```

```leanOutput otLaws (whitespace := lax)
@StepLaw : {α : Type} → [inst : Storage α] → [inst_1 : Context α] → TensorOptimizer α → Type
```

```leanOutput otLaws (whitespace := lax)
@StepLaw.runSteps_eq : ∀ {α : Type} [inst : Storage α] [inst_1 : Context α] {opt :
  TensorOptimizer α} (law : opt.StepLaw) {s : Shape} (current : OptimizerStep α s (opt.State s))
  (gradients : Array (Tensor α s)), law.runSteps current gradients = opt.runSteps current
  gradients
```

`TensorOptimizer.adamw` takes five scalar arguments in order: learning rate, weight decay, `beta1`,
`beta2`, and epsilon. Their types are identical, so you need the calls above to identify which
setting each experiment uses.

`TensorOptimizer.runSteps_append` is the associativity of running a stream of gradients, and it
applies to every packaged optimizer at once, because the optimizer is an argument. It is the lemma
that lets a proof about an epoch be assembled from proofs about batches.

On the right of `runSteps_append`, the nested call matters. Its initial value is the complete
result of running `left` (including the updated moment buffers and counter), and reinitializing the
optimizer before `right` would describe a different computation. Gradient order is preserved too:
the theorem permits splitting a stream, not shuffling it. In a trainer that recomputes gradients
from the current parameters, you also have to show that the two executions produce the same
gradient stream. That stream is an input to the append law, which doesn't generate it from a loss
function.

`StepLaw` is reserved for the stronger situation in which a separately stated mathematical
recurrence is proved equal to an executable update, and its generic run theorem then lifts that
agreement over a stream of gradients. Concrete optimizer equations live in
`TorchLean.NN.Optimizer.Update`, while optimizer-specific theory files establish algebraic
invariants and comparisons that aren't restatements of those definitions.

GaLore-style projected SGD follows the same rule, and its executable projection interface lives in
{src "TorchLean/NN/Optimizer/Update.lean"}[TorchLean.NN.Optimizer.Update]. Choosing the identity
projector recovers the ordinary SGD parameter update by definition, and
`GaLore.update_params_eq_sgd` states that equality at every scalar type. An arbitrary learned
low-rank projector needs its own contract.

# Optimizer Extension Points: Muon And GaLore-Style Updates

Modern optimizer work often mixes a familiar base update with a specialized backend. TorchLean
models that mix: its runtime layer gives the executable update equation, and its theory layer
states what has to be true of the backend output before the update can be cited in a proof.

Muon is represented as momentum followed by an orthogonalization backend. One step first updates the
momentum buffer

$$`m_{t+1}=\beta m_t+g_t,`

then asks a backend for the direction used in the parameter update. With the identity backend,
Muon's parameter update is exactly momentum SGD. With a certified matrix backend, the claim is about
the actual direction returned by that backend: exact column orthogonality

$$`Q^\top Q = I`

or an approximate Gram-residual bound

$$`\max_{i,j}|(Q^\top Q-I)_{ij}| \le \varepsilon.`

Those two predicates, and the residual matrix that separates them, are the whole vocabulary:

```lean (name := otGramApi)
-- The Gram matrix records column lengths and pairwise
-- column inner products.
#check @HasExactColumnGram
#check @HasApproxColumnGram
#check @columnGramResidual
```

```leanOutput otGramApi (whitespace := lax)
@HasExactColumnGram : {α : Type} → [inst : Storage α] → [Context α] → {m n : ℕ} → MatrixTensor α
  m n → Prop
```

```leanOutput otGramApi (whitespace := lax)
@HasApproxColumnGram : {α : Type} → [inst : Storage α] → [Context α] → {m n : ℕ} → α →
  MatrixTensor α m n → Prop
```

```leanOutput otGramApi (whitespace := lax)
@columnGramResidual : {α : Type} → [inst : Storage α] → [Context α] → {m n : ℕ} → MatrixTensor α
  m n → MatrixTensor α n n
```

`MatrixTensor α m n` is a rank-two tensor with `m` rows and `n` columns. `columnGramResidual Q` is
$`Q^\top Q-I`, so `HasExactColumnGram` says that residual is zero and `HasApproxColumnGram eps`
says every entry of it is within `eps`.

Each diagonal entry of $`Q^\top Q` is a column's squared length, and each off-diagonal entry is the
dot product of two distinct columns. So an approximate entrywise Gram certificate controls two
things at once: the error in unit length and the failure of pairwise orthogonality. It doesn't
choose a preferred direction relative to the objective gradient. For a Muon update, this is useful
geometric information about the backend's matrix, but an objective-descent theorem would still need
to connect that direction to the loss. Shapes matter too: $`m` rows and $`n` orthonormal columns
require enough ambient coordinates to accommodate those columns.

## Newton-Schulz Iteration

TorchLean's Newton-Schulz orthogonalizer forms the right Gram matrix $`G=Q^\top Q` and evaluates
$`aQ+b\,QG+c\,(QG)G` repeatedly. Over the reals this is the same odd polynomial as the left-Gram
form, but the two evaluation orders need not agree in floating arithmetic. Compare the Muon
coefficient set below with the classical cubic $`\tfrac32 Q-\tfrac12 QQ^\top Q`. For exact Gram
preservation, the library's theorem
`step_hasExactColumnGram` needs $`(a+b+c)^2=1`,
so start by asking whether each coefficient set even qualifies:

```lean (name := otMuonSums)
-- An exactly orthonormal input is rescaled by the sum of
-- these coefficients.
def otQuintic : Coeffs Float :=
  { a := 3.4445, b := -4.7750, c := 2.0315 }

def otCubic : Coeffs Float :=
  { a := 1.5, b := -0.5, c := 0.0 }

#eval (otQuintic.a + otQuintic.b + otQuintic.c,
       otCubic.a + otCubic.b + otCubic.c)
```

```leanOutput otMuonSums (whitespace := lax)
(0.701000, 1.000000)
```

Only the cubic coefficients satisfy that condition. When the columns are already orthonormal, the
Gram matrix is the identity, so one polynomial step scales the whole matrix by the sum of the
coefficients and its Gram matrix by the square of that sum. That explains the theorem's hypothesis
and why it doesn't apply to the displayed quintic coefficients.

Next, watch the residual on the matrix below, which has been divided by its Frobenius norm (Muon
does the same to its momentum buffer before orthogonalizing):

```lean (name := otCubicRun)
-- Track the Gram residual after several cubic iterations,
-- then test exact zero.
def otRaw : Tensor Float [2, 2] := [[1.0, 0.5], [0.0, 1.0]]

def otScaled : Tensor Float [2, 2] :=
  [[0.6666666666666666, 0.3333333333333333],
   [0.0, 0.6666666666666666]]

/-- Largest absolute entry of `columnGramResidual`, the
entrywise maximum that `HasApproxColumnGram` bounds. -/
def otResidualMax {m n : ℕ}
    (Q : Tensor Float [m, n]) : Float :=
  (columnGramResidual Q).foldl
    (fun acc x =>
      if acc.isNaN || x.isNaN then 0.0 / 0.0
      else max acc x.abs) 0.0

#eval otResidualMax otScaled

#eval otResidualMax (iter otCubic 1 otScaled)

#eval otResidualMax (iter otCubic 3 otScaled)

#eval otResidualMax (iter otCubic 6 otScaled)

#eval columnGram (iter otCubic 10 otScaled)

#eval otResidualMax
        (iter otCubic 10 otScaled) == 0.0
```

```leanOutput otCubicRun (whitespace := lax)
0.555556
```

```leanOutput otCubicRun (whitespace := lax)
0.330590
```

```leanOutput otCubicRun (whitespace := lax)
0.022991
```

```leanOutput otCubicRun (whitespace := lax)
0.000000
```

```leanOutput otCubicRun (whitespace := lax)
[[1.000000, -0.000000], [-0.000000, 1.000000]]
```

```leanOutput otCubicRun (whitespace := lax)
false
```

In that run the residual falls from $`0.556` to $`0.331` to $`0.023`, and by six steps it prints as
zero. At ten steps the Gram matrix prints as the identity (negative zeros in the off-diagonal
included).

But the final equality test returns `false`: the computed residual is nonzero, but too small for
the six-decimal display. An approximate certificate would need a chosen `eps` and a proof that
every residual entry is within it, and printed output alone doesn't supply that proof or an exact
Gram certificate. Nor does this finite trace establish what every later iterate does: depending on
the input, floating iterations can settle, cycle, or reach an exact representable solution. That's
the same lesson the {ref "floats"}[floating point chapter] draws from Flocq's treatment of rounding
{Informal.citep flocq2011}[]: a decimal rendering can conceal the difference between a small error
and zero.

On an unnormalized matrix, the quintic coefficients do something worse than converge slowly:

```lean (name := otQuinticRun)
-- Compare the quintic iteration on the raw matrix and its
-- normalized version.
#eval otResidualMax (iter otQuintic 1 otRaw)

#eval otResidualMax (iter otQuintic 3 otRaw)

#eval otResidualMax (iter otQuintic 5 otScaled)
```

```leanOutput otQuinticRun (whitespace := lax)
0.567997
```

```leanOutput otQuinticRun (whitespace := lax)
6182.917801
```

```leanOutput otQuinticRun (whitespace := lax)
0.318189
```

Starting from `otRaw` (whose largest singular value exceeds one), the residual grows from $`0.568`
to $`6182.9` in two more steps, in the displayed finite run. On the normalized input the residual
after five steps is about $`0.32`, an observation that establishes neither eventual convergence nor
stalling. Since the polynomial doesn't control arbitrary singular values, its input scaling
matters.

A scalar picture explains the sensitivity to normalization. Along a singular direction with
singular value $`s`, the matrix polynomial applies $`s\mapsto as+bs^3+cs^5` in real arithmetic. For
the cubic coefficients, this is $`s\mapsto(3s-s^3)/2` (with a fixed point at $`s=1`). For large
$`s`, the highest-degree term can dominate instead of moving the value toward one. Dividing by the
Frobenius norm limits the initial singular values, but that alone isn't a proof of convergence for
an arbitrary coefficient choice. The Gram-preservation theorem above answers the narrower algebraic
question of what one step does when the Gram matrix is already exactly the identity.

`NewtonSchulz.orthogonalizer` leaves normalization to the caller. A residual-checked backend then
checks the direction it actually returns. Input normalization and output certification are
separate obligations.

## Muon Backend Contracts

In {src "TorchLean/NN/Optimizer/Muon.lean"}[the Muon contract file] and the
{srcDir "TorchLean/NN/Optimizer/Muon"}[Muon theory directory], these cases are packaged as exact,
approximate, and checked-backend contracts. QR-backed directions give an exact
path under positive-pivot hypotheses. This QR backend is a noncomputable real-valued specification,
and a floating-point QR kernel would need its own bridge. Newton-Schulz-style directions give a
residual-checked approximate path, together with fixed-point exact statements when the iteration
has reached the corresponding algebraic condition.

These theorems consume checked-backend and QR hypotheses:

```lean
-- These consumers require evidence for the backend used by
-- the actual update.
#check @exactCertifiedStep_of_success
#check @approxCertifiedStep_of_success
#check @exists_exactCertifiedStep
```

The next theorem derives a Gram bound from the success predicate of a
`CheckedApproxOrthogonalizer`, whose contract connects success to that bound:

```lean (name := otChecked)
-- Success must be established for the freshly updated
-- momentum buffer.
#check @update_hasApproxColumnGram
```

```leanOutput otChecked (whitespace := lax)
@update_hasApproxColumnGram : ∀ {α : Type} [inst : Storage α] [inst_1 :
  Context α] {m n : ℕ} {eps : α} (backend : CheckedApproxOrthogonalizer α m n eps) (lr
  momentum : α) (momentumBuffer parameters gradients : MatrixTensor α m n), backend.Success
  (Muon.update { lr := lr, momentum := momentum, momentumBuffer :=
  momentumBuffer, orthogonalizer := backend.orthogonalizer } parameters
  gradients).optimizerState.momentumBuffer → HasApproxColumnGram eps
  (backend.orthogonalizer.apply (Muon.update { lr := lr, momentum := momentum,
  momentumBuffer := momentumBuffer, orthogonalizer := backend.orthogonalizer } parameters
  gradients).optimizerState.momentumBuffer)
```

`backend.Success` appears to the left of the arrow, applied to the *post-momentum* buffer (the
matrix the orthogonalizer actually receives). A backend may be fast, randomized, iterative, or
external, and Lean only uses it as an orthogonalizing step after that predicate has been proved for
that buffer (using the backend's contract). Executing the update alone doesn't discharge the
success premise.

The consumer side of the boundary is in
{src "TorchLeanExamples/DeepDives/Muon.lean"}[
TorchLeanExamples.DeepDives.Muon], where the certificates are combined with the
update equation to say something about a parameter step. Start with its `Concrete` namespace: the
direction is the column $`Q=(3/5,4/5)^\mathsf{T}`, whose Gram condition reduces to $`9/25+16/25=1`.
The column $`(1,1)^\mathsf{T}` is rejected because its squared length is two. With zero momentum,
parameters $`(2,3)^\mathsf{T}`, and learning rate $`1/10`, the certified update is
$`(97/50,73/25)^\mathsf{T}`:

```lean
-- The unit column supplies both a Gram certificate and a
-- concrete parameter step.
open Examples.Muon.Concrete in
example :
    ExactCertifiedStep state parameters direction
      direction :=
  step_certified

open Examples.Muon.Concrete in
example :
    (Muon.update state parameters direction).parameters
      = column (97 / 50) (73 / 25) :=
  updated_parameters_eq
```

That concrete case supplies the whole certificate without an unproved backend hypothesis. Its
identity backend works because the chosen buffer already has a unit column, but it doesn't
orthogonalize arbitrary gradients or establish convergence. Build the tutorial with
`scripts/lake.sh build TorchLeanExamples.DeepDives.Muon` and inspect
`Concrete.step_certified` in the Infoview. For general QR and Newton–Schulz backends, the later
wrappers explain how to consume certificates:

```lean (name := otQrCert)
-- Read the shared direction witness, then inspect the
-- theorem’s logical dependencies.
open Examples.Muon in
#check @qr_step_exact_gram

open Examples.Muon in
#print axioms qr_step_exact_gram
```

```leanOutput otQrCert (whitespace := lax)
@qr_step_exact_gram : ∀ {m n : ℕ} (lr momentum : ℝ)
  (momentumBuffer parameters gradients : Tensor ℝ [m, n]), HasPositivePivots (Muon.update {
  lr := lr, momentum := momentum, momentumBuffer := momentumBuffer,
  orthogonalizer := QR.orthogonalizer } parameters gradients).optimizerState.momentumBuffer → ∃
  direction, HasExactColumnGram direction ∧ (Muon.update { lr := lr, momentum :=
  momentum, momentumBuffer := momentumBuffer, orthogonalizer := QR.orthogonalizer } parameters
  gradients).parameters = parameters - direction.scale lr
```

```leanOutput otQrCert (whitespace := lax)
'TorchLean.Examples.Muon.qr_step_exact_gram' depends on
  axioms: [propext, Classical.choice, Quot.sound]
```

In the QR theorem, the conclusion supplies a direction with orthonormal columns and identifies the
updated parameters as the old parameters minus the learning rate times that direction.
`HasPositivePivots`, its hypothesis, is the admissibility condition for the QR path, and it's
stated on the momentum buffer, not on the raw gradient. Its axiom list is the usual classical trio
and nothing library-specific: the proof uses `Classical.choice`, propositional extensionality, and
quotient soundness.

In `∃ direction`, the existential binds one matrix shared by both conjuncts. A caller receives
orthonormality of that matrix and the equation showing that it was used in the parameter step.
Keeping the same witness in both places rules out proving a Gram fact about an unrelated matrix.

## GaLore Projection Contracts

GaLore-style code is different. GaLore is a gradient-projection strategy, not a single optimizer
name, and its runtime object is a projector/lift pair around a base update:

$$`p_{t+1}=p_t-\eta\,\mathrm{lift}(\mathrm{project}(g_t)).`

With the identity projector, the projected parameter update is ordinary SGD by definition. A
low-rank projector or refresh policy can optimize memory and matrix structure, but it has to state
its own projection contract instead of being hidden inside the word "optimizer."

You can see this naming in the trainer API. Standard trainer configs use names such as `optim.sgd`,
`optim.adamW`, and `optim.adaDelta`. Runtime-level extension points use more specific names such as
`optim.muon` and `optim.galore.sgd`, because those calls must record their backend or
projection as part of the mathematical object.

# Gradient Descent As A Contractive Map

TorchLean's core convergence theorem studies the ideal deterministic update
`step eta g x = x - eta * g x`.

In mathematical notation:

$$`x_{t+1}=x_t-\eta g(x_t).`

Before reading the general theorem, run the recurrence for

$$`f(x)=x^2,\qquad \nabla f(x)=2x,\qquad \eta=\frac14.`

We chose this example so that the three constants the abstract theorem asks for are all known
exactly. Its gradient $`x\mapsto 2x` is strongly monotone with $`\mu=2` and Lipschitz with $`L=2`,
so the contraction factor $`q=1-2\eta\mu+\eta^2L^2` can be computed, not estimated. We use
rationals so that every printed iterate is exact:

```lean (name := gdDemo)
-- Use rational arithmetic to display the contraction factor
-- and iterates exactly.
def demoGrad (x : ℚ) : ℚ := 2 * x

def demoStep (x : ℚ) : ℚ :=
  x - (1 / 4) * demoGrad x

-- The three constants the general theorem quantifies over,
-- instantiated for this one gradient map.
def demoEta : ℚ := 1 / 4
def demoMu : ℚ := 2
def demoLip : ℚ := 2

def demoQ : ℚ :=
  1 - 2 * demoEta * demoMu + demoEta ^ 2 * demoLip ^ 2

/-- The first `k + 1` iterates of `demoStep` from `x`. -/
def demoTrace (x : ℚ) (k : ℕ) : Tensor ℚ [k + 1] :=
  Tensor.ofFn fun i => (demoStep^[i.val]) x

#eval demoQ

#eval (demoEta * demoLip ^ 2, 2 * demoMu)

#eval demoTrace 4 4

#eval Tensor.map (fun x => x ^ 2) (demoTrace 4 4)

#eval (demoStep (demoStep 4)) ^ 2
        == demoQ ^ 2 * (4 : ℚ) ^ 2

example (x : ℚ) : demoStep x = x / 2 := by
  simp [demoStep, demoGrad]
  ring
```

```leanOutput gdDemo (whitespace := lax)
1 / 4
```

```leanOutput gdDemo (whitespace := lax)
(1, 4)
```

```leanOutput gdDemo (whitespace := lax)
[4, 2, 1, (1 : Rat)/2, (1 : Rat)/4]
```

```leanOutput gdDemo (whitespace := lax)
[16, 4, 1, (1 : Rat)/4, (1 : Rat)/16]
```

```leanOutput gdDemo (whitespace := lax)
true
```

Here the contraction factor is $`1/4`, and the step size condition $`\eta L^2<2\mu` reads $`1<4`
(the second output), so this step size is inside the safe range with room to spare. Successive
iterates are `4, 2, 1, 1/2, 1/4`, halving as the `example` at the end of the block proves they
must, and their squared distances to the minimizer are `16, 4, 1, 1/4, 1/16`, each one a factor of
$`1/4` below the last.

Finally, the fifth output compares the run with the library theorem's bound,
$`\|x_k-x^\star\|^2\le q^k\|x_0-x^\star\|^2`. On this example the two sides are *equal* at $`k=2`
(the rational equality test checks it). And the symbolic example proves that every step halves the
iterate. So the rate is attained, which rules out a uniform strict improvement at these particular
values $`\mu=L=2` and $`\eta=1/4`, but attainment here doesn't show that the formula is optimal for
every $`\mu<L`.

For the general theorem, assumptions on an abstract gradient map replace this one-dimensional
calculation, and the exponent $`k` survives the generalization.

If `g` is strongly monotone with parameter $`\mu`, Lipschitz with parameter $`L`, and the step size
$`\eta` is in the safe range, then one step is contractive:

> the squared distance between `step eta g x` and `step eta g y` is at most $`q` times the squared
> distance between $`x` and $`y`, and the distance itself has factor $`\sqrt q`.

Strong monotonicity is the predicate `StrongMonotone mu g` in the
{src "TorchLean/NN/Optimizer/GD.lean"}[linear convergence API].
Informally, it says that the inner product of $`g(x)-g(y)` with $`x-y` dominates
$`\mu\lVert x-y\rVert^2`.

`E` doesn't have to be a scalar. `NormedAddCommGroup E` supplies vector addition, subtraction, and
a norm, while `InnerProductSpace ℝ E` supplies the real inner product used in the cross term. A
finite collection of real model parameters can be viewed in such a space once its chosen
representation and norm are connected to this interface. `L : NNReal` carries nonnegativity in its
type (`↑L` is the coercion to a real number for the formula for `q`). Strong monotonicity is a
global comparison of two inputs of `g`, so checking a positive gradient at one point wouldn't
establish it.

Written out, the two analytic hypotheses read:

$$`\langle g(x)-g(y),x-y\rangle\ge \mu\|x-y\|^2`

and

$$`\|g(x)-g(y)\|\le L\|x-y\|.`

Here are the signatures of the one-step bound, its iteration, and the step-size conditions:

```lean (name := otGdThms)
-- Follow the one-step bound through iteration and the
-- explicit step-size conditions.
#check @StrongMonotone
#check @step_norm_sq_le
#check @dist_sq_iterate_le_of_q_nonneg
#check @q_lt_one_of_mul_sq_lt
#check @dist_sq_iterate_le_of_step_size
#print axioms GD.dist_sq_iterate_le_of_step_size
```

```leanOutput otGdThms (whitespace := lax)
@StrongMonotone : {E : Type} → [inst : NormedAddCommGroup E] → [InnerProductSpace ℝ E] → ℝ → (E
  → E) → Prop
```

```leanOutput otGdThms (whitespace := lax)
@step_norm_sq_le : ∀ {E : Type} [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  (η μ : ℝ), 0 ≤ η → ∀ {L : NNReal} (g : E → E), StrongMonotone μ g → LipschitzWith L g → ∀ (x y
  : E), ‖GD.step η g x - GD.step η g y‖ ^ 2 ≤ (1 - 2 * η * μ + η ^ 2 * ↑L ^ 2) * ‖x - y‖ ^ 2
```

```leanOutput otGdThms (whitespace := lax)
@dist_sq_iterate_le_of_q_nonneg : ∀ {E : Type} [inst : NormedAddCommGroup E] [inst_1 :
  InnerProductSpace ℝ E] (η μ : ℝ), 0 ≤ η → ∀ {L : NNReal} (g : E → E), StrongMonotone μ g →
  LipschitzWith L g → ∀ {xStar x : E}, g xStar = 0 → 0 ≤ q η μ L → ∀ (k : ℕ),
  ‖(GD.step η g)^[k] x - xStar‖ ^ 2 ≤ q η μ L ^ k * ‖x - xStar‖ ^ 2
```

```leanOutput otGdThms (whitespace := lax)
q_lt_one_of_mul_sq_lt : ∀ (η μ : ℝ) (L : NNReal), 0 < η → η * ↑L ^ 2 < 2 * μ → q η μ L < 1
```

```leanOutput otGdThms (whitespace := lax)
@dist_sq_iterate_le_of_step_size : ∀ {E : Type} [inst : NormedAddCommGroup E] [inst_1 :
  InnerProductSpace ℝ E] (η μ : ℝ) {L : NNReal} (g : E → E), StrongMonotone μ g → LipschitzWith
  L g → ∀ {xStar x : E}, g xStar = 0 → 0 ≤ μ → μ ≤ ↑L → 0 < η → η * ↑L ^ 2 < 2 * μ → ∀ (k : ℕ),
  ‖(GD.step η g)^[k] x - xStar‖ ^ 2 ≤ q η μ L ^ k * ‖x - xStar‖ ^ 2 ∧ q η μ L < 1
```

```leanOutput otGdThms (whitespace := lax)
'TorchLean.GD.dist_sq_iterate_le_of_step_size' depends on axioms: [propext, Classical.choice,
  Quot.sound]
```

`step_norm_sq_le` is the one-step inequality, with the factor $`1-2\eta\mu+\eta^2L^2` written out
in the conclusion (not abbreviated). To obtain it, expand the squared norm of the difference
between two updates: the expansion has the original squared distance, a cross term with coefficient
minus twice the step size, and the squared gradient difference. Strong monotonicity bounds the
cross term from above because its coefficient is nonpositive, and Lipschitzness bounds the squared
gradient difference. Combining the three coefficients gives the displayed factor. Then the
{src "TorchLean/NN/Optimizer/GD/Convergence.lean"}[strongly convex gradient descent
API] iterates it with `dist_sq_iterate_le_of_q_nonneg`:

> If the contraction factor $`q(\eta,\mu,L)` is nonnegative and strictly below one, then after
> $`k` gradient descent steps the squared distance to the root is at most $`q^k` times the initial
> squared distance.

For the bound itself, $`q\ge 0` is enough. When $`q<1`, its right-hand side tends to zero, and
`tendsto_iterate_of_q_lt_one` proves that the iterates converge to the supplied root.

In that theorem the factor $`q^k` is in the conclusion, not only in the prose, and `(step η g)^[k]`
is Mathlib's function iteration, so "after $`k` steps" is literal {Informal.citep mathlib2020}[].
Through the hypothesis `g xStar = 0`, the caller supplies a root. The theorem neither proves that a
root exists nor requires `g` to have been identified as the gradient of an objective.

In the axiom report, Lean lists the logical dependencies of this proof. `propext` identifies
logically equivalent propositions, `Classical.choice` supplies classical choice, and `Quot.sound`
supports reasoning with quotients. The report names these foundations, not an added assumption that
gradient descent converges. The mathematical conditions on the update map remain premises stated in
the theorem signature.

In the iteration proof, the root equation identifies a fixed point:
$`T_\eta(x^\star)=x^\star-\eta g(x^\star)=x^\star`. At step zero, the estimate is an equality
because function iteration is the identity and $`q^0=1`. For the inductive step, the one-step
inequality multiplies the previous squared-error bound by `q`, whose nonnegativity preserves the
inequality. Repeating this argument produces the exponent. That's why `0 ≤ q` is listed separately
from `q < 1`: one condition supports induction, while the other makes the geometric bound decrease
with the number of steps.

Two companions turn the abstract condition on $`q` into a condition on the step size.
`q_lt_one_of_mul_sq_lt` shows $`q<1` whenever $`0<\eta` and $`\eta L^2<2\mu`, and
`dist_sq_iterate_le_of_step_size` packages the geometric bound together with $`q<1` under
$`0\le\mu\le L` and that step-size inequality, so a caller checks numbers, not an inequality about
$`q`. Those are the numbers the demo above checked: $`\eta L^2=1` against $`2\mu=4`.

We prove the scalar specialization separately as `ScalarGD.abs_step_sub_target_lt` (the version
closest to the rational demo above):

```lean
-- The scalar result isolates the error multiplier for a
-- quadratic objective.
#check @ScalarGD.abs_step_sub_target_lt
```

Mathematically, the result is the standard smooth/strongly-monotone contraction argument found in
convex optimization texts such as Nesterov's [*Introductory Lectures on Convex
Optimization*](https://link.springer.com/book/10.1007/978-1-4419-8853-9). TorchLean attaches this
familiar rate to the exact update map and assumptions used by its training theorems.

# Smoothness And Strong Convexity

Most papers state convergence using smoothness and strong convexity of an objective `f`, not
strong monotonicity of an abstract gradient map. TorchLean keeps both vocabularies and proves the
bridge between them.

Informally, first order strong convexity says that `f y` lies above the tangent model at $`x` plus
a quadratic term with coefficient $`\mu/2`.

In the {src "TorchLean/NN/Optimizer/GD/StrongConvexity.lean"}[smooth strong
convex bridge API], that objective level statement becomes a gradient map statement. The theorem to
recognize is `strongMonotone_gradient`: under the first order strong
convexity hypothesis, the gradient is strongly monotone. That bridge lets a training theorem move
from "the loss has these analytic assumptions" to "the update map is contractive."

```lean (name := otBridge)
-- Convert the objective’s first-order inequality into
-- strong monotonicity.
#check @FirstOrderStrongConvex
#check @strongMonotone_gradient
```

```leanOutput otBridge (whitespace := lax)
@FirstOrderStrongConvex : {E : Type} → [inst : NormedAddCommGroup E] → [InnerProductSpace ℝ E] →
  [CompleteSpace E] → ℝ → (E → ℝ) → Prop
```

```leanOutput otBridge (whitespace := lax)
@strongMonotone_gradient : ∀ {E : Type} [inst : NormedAddCommGroup E]
  [inst_1 : InnerProductSpace ℝ E] [inst_2 : CompleteSpace E] (μ : ℝ) {f : E → ℝ},
  FirstOrderStrongConvex μ f → StrongMonotone μ fun x => gradient f x
```

In the bridge theorem, the conclusion is `StrongMonotone μ fun x => gradient f x`, which is
precisely the hypothesis `step_norm_sq_le` wants (with the same $`\mu`). `FirstOrderStrongConvex`
carries an extra `[CompleteSpace E]` instance because `gradient` is Mathlib's
Fréchet-derivative-based gradient, which needs the inner product space to be complete for the Riesz
identification to be available. The proof applies first-order strong convexity in both directions
and adds the two inequalities: the objective values cancel, leaving the required lower bound on the
gradient inner product. Applying the convergence theorem still requires a Lipschitz bound, a root,
and the step-size hypotheses.

That first-order inequality itself mentions Mathlib's `gradient f x`. To obtain it from a usual
convex-analysis specification, the same source provides a pointwise bridge from `StrongConvexOn` on
the whole space and differentiability at the base point, and differentiability everywhere supplies
the global first-order hypothesis. This supplies the mathematical justification for treating the
named gradient as the tangent term, but not the separate Lipschitz bound.

Autograd theorems usually speak about derivatives of a loss, while convergence theorems often speak
about a gradient map, and that's what makes the bridge useful. The theorem connects those two
vocabularies without treating "gradient" as an informal word.

# Verified Training

TorchLean's autograd theorems explain why the gradient path computes the intended derivative.
Runtime approximation theorems explain how close a rounded update is to the ideal update.
Optimization theory is where we state what that update means as an algorithm.

For example, a theorem about a full training run can be read as a composition of three facts:

- an autograd theorem saying the gradient is the adjoint derivative of the loss
- a runtime approximation theorem saying the executable update is close to the ideal update
- an optimization theorem saying the ideal update contracts under smoothness and strong convexity
  hypotheses.

Those hypotheses matter. The optimization layer avoids turning a loss curve into a convergence
claim: convexity, smoothness, strong monotonicity, and step size conditions remain visible in the
theorem statement.

The examples distinguish algorithmic effects from numerical ones: the real AdamW theorem compares
two update rules, while the Adam result's difference from $`0.7` (about $`3\times10^{-9}`) comes
primarily from epsilon in the recurrence. Separately, the Newton-Schulz residual prints as zero
while its equality test returns false. A runtime error theorem must compare an execution with the
same ideal recurrence, including its epsilon and backend choices.

There's also a state boundary between these equations and a native trainer. Runtime Adam-family
optimizers maintain moment buffers and a step counter for each parameter, while checkpoints must
restore those states with the same parameter identity and ordering. The pure `TensorOptimizer` and
`StepLaw` theorems prove the recurrence they state, but by themselves they don't prove allocation,
aliasing, checkpoint restoration, or the mutable runtime state machine. An end-to-end optimizer
claim needs a bridge showing that the trainer's stored state implements the same recurrence.

For a rounded implementation, the next useful equation is a one-step comparison with the ideal map,
such as $`\|\widehat T(x)-T(x)\|\le e`. Combined with a distance contraction factor $`r=\sqrt q`,
the triangle inequality gives an error recurrence of the form $`d_{t+1}\le e+r d_t` when those
bounds apply at the relevant iterates. That recurrence explains why repeated small execution errors
need to be accumulated through training, not checked only at the final step, and it also identifies
the necessary inputs to such an argument: a uniform or stepwise numerical error bound and an ideal
contraction theorem for the same update.

# Training Theorem Assumptions

A theorem about a particular training run must identify the gradient, scalar semantics, step-size
condition, and any backend contract (such as a Muon orthogonalizer certificate or a projector law).
The results here supply selected components, while the trainer must satisfy their premises and
connect its mutable state to the stated recurrence.

# Sources

Files behind this chapter:

- {src "TorchLean/NN/Optimizer/AdamW.lean"}[AdamW.lean], the zero-decay
  AdamW/Adam agreement theorem.
- {src "TorchLean/NN/Optimizer/Update.lean"}[Update.lean], the executable optimizer update
  equations.
- {src "TorchLean/NN/Optimizer/TensorOptimizer.lean"}[TensorOptimizer.lean],
  `TensorOptimizer`, the packaged optimizers used in the run above, and `StepLaw`.
- {src "TorchLean/NN/Optimizer/Muon.lean"}[Optimizer/Muon.lean], the Gram predicates and the
  checked-backend contracts.
- {srcDir "TorchLean/NN/Optimizer/Muon"}[the Muon directory], the Newton-Schulz iteration, the QR
  backend, and the step certificates.
- {src "TorchLeanExamples/DeepDives/Muon.lean"}[Muon.lean], the consumer
  theorems that combine a certificate with the update equation.
- {src "TorchLean/NN/Optimizer/GD.lean"}[GD.lean],
  `StrongMonotone`, `GD.step`, and the one-step contraction.
- {src "TorchLean/NN/Optimizer/GD/Convergence.lean"}[Convergence.lean], the
  contraction factor `q`, the iterated geometric bounds, and the step-size packaging.
- {src "TorchLean/NN/Optimizer/GD/StrongConvexity.lean"}[
  StrongConvexity.lean], first order strong convexity and the bridge to strong
  monotonicity.

# References

The gradient-descent proof follows the contraction argument in Yurii Nesterov's
[*Introductory Lectures on Convex
Optimization*](https://link.springer.com/book/10.1007/978-1-4419-8853-9), Springer 2004.
