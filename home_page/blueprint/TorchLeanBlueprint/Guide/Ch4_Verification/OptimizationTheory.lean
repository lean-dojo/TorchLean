import VersoManual
import NN.API
import NN.MLTheory.Optimization.FirstOrder
import NN.MLTheory.Optimization.OptimizerLaws
import NN.MLTheory.Optimization.Muon
import NN.MLTheory.Optimization.GDLinearConvergence
import NN.MLTheory.Optimization.StronglyConvexGD
import NN.MLTheory.Optimization.SmoothStrongConvexBridge
import NN.Examples.Optimization.MuonCertificates
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean

-- Opening these namespaces is what lets the `#check` lines below stay narrow enough
-- to read. Nothing is hidden by it: a printed signature always names its constants in
-- full, so the module a theorem came from is still visible in the output.
open TorchLean
open Optim
open Optim.Muon
open Optim.GD

-- `lax` whitespace on the expected-output blocks lets a long printed signature be
-- rewrapped to fit this file's line budget without changing what is being asserted.
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Optimization Theory" =>
%%%
tag := "optimization-theory"
%%%

Training code executes updates. Optimization theory explains when those updates should make
progress.

A TorchLean training run may call SGD or Adam, but a convergence theorem cannot come from the name
of the optimizer alone. It needs an ideal update map, assumptions on the objective or gradient, and
a step size condition.

Even two updates that print the same parameter can differ: Adam's epsilon changes its recurrence
by less than the six-decimal display reveals. We can expose that difference on a single parameter
before asking which identities hold for arbitrary states.

# The Optimization Contract

The optimization contract has three layers:

- *Runtime update*: the concrete operation that writes new values into parameter tensors, such as
  one SGD, momentum, or Adam-style step.
- *Ideal update*: the mathematical map the runtime update is intended to approximate, for example
  $`x\mapsto x-\eta g(x)`.
- *Convergence theorem*: the conditional theorem saying that iterating the ideal map makes progress
  under assumptions such as strong monotonicity, Lipschitzness, and a safe step size.

A decreasing loss curve describes one run. A contraction theorem compares the update at arbitrary
pairs of states, so it needs more information than that curve supplies.

# First Order Updates

The first object is a first order optimizer state: parameters, gradients, an optional buffer, and a
time counter. Its executable equation belongs in the definition of the update itself. TorchLean does
not add a theorem merely to restate each record field after reduction. The theory layer begins when
two distinct updates are related.

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

Each number is checkable without a computer. Plain SGD subtracts $`0.1` three times, giving
$`0.7`. Momentum accumulates the buffer $`1`, then $`1.9`, then $`2.71`, and subtracting
$`0.1` times each of those from $`1` gives $`0.439`. Adam divides the gradient by the square
root of its bias-corrected second moment plus epsilon. With this constant gradient the update is
close to $`0.1`, so the result rounds to $`0.7` at six decimals; epsilon keeps it from being exact.
AdamW at zero weight decay
is the same run again. AdamW at weight decay $`0.5` differs from Adam because it
subtracts $`\eta\lambda p` from the parameter on top of the Adam step.

The buffers explain why the parameter alone is insufficient to resume these runs. At the third
momentum step the update uses the accumulated value $`2.71`, although the supplied gradient is
still $`1`. Restarting momentum from its zero buffer at the same parameter would instead subtract
$`0.1`. Adam carries two buffers and a step count because bias correction depends on how many
updates those buffers have seen. `otRun` passes the entire `optimizerState` forward through the
array, so the comparison includes that history.

For the decay example, ignoring epsilon's small correction gives the recurrence
$`p_{t+1}=0.95p_t-0.1`. Its first three values are $`0.85`, $`0.7075`, and $`0.572125`.
The shrinkage acts on the current parameter each time; it is not one subtraction of $`0.05`
repeated independently of the parameter.

The corresponding PyTorch experiment uses double precision to match Lean's `Float` format
{Informal.citep pytorch2019}[]:

```
SGD                     [0.7000000000000001]
SGD momentum 0.9        [0.43900000000000006]
Adam                    [0.7000000030000006]
AdamW weight_decay=0    [0.7000000030000006]
AdamW weight_decay=0.5  [0.5721250028525007]
Adam  weight_decay=0.5  [0.7003815249719783]
```

The first five lines agree at the displayed precision. The sixth uses coupled decay, which
`TensorOptimizer.adam` does not expose. PyTorch's `Adam(weight_decay=0.5)` adds $`\lambda p` to
the gradient before the moment
estimates see it, so the decay is then divided by the adaptive denominator along with everything
else. Three steps of coupled decay leave the parameter at $`0.7003815`, whereas decoupled decay
reaches
$`0.5721250`. The adaptive normalization changes how an L2 gradient penalty affects the step.
The public tensor Adam API here exposes no weight-decay argument; AdamW exposes
separate decoupled decay {Informal.citep adamw2019}[].

TorchLean shows six decimals, and $`0.7000000030` rounds to `0.700000` at that precision.
To examine the difference hidden by the display, the following checks compare TorchLean Adam
with its own zero-decay AdamW and measure the residual from `0.7`; they do not compare PyTorch bits:

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

Adam and AdamW at zero decay compare equal numerically on this input. Neither is
equal to $`0.7`: the residual is about $`3\times 10^{-9}`, consistent with the scale of the PyTorch
residual shown above.
The leading difference comes from $`\varepsilon = 10^{-8}` in the denominator: even in exact
arithmetic, each Adam step is slightly smaller than the SGD step. Floating arithmetic adds its
own rounding error. The discrepancy from `0.7` is therefore not solely a rounding residual.

## Adam And Zero-Decay AdamW

The real-valued counterpart of the Adam/zero-decay AdamW agreement is
`update_weight_decay_zero_parameters_eq_adam_real`, in the `Optim.AdamW` namespace. It quantifies
over the state and parameters for a single update. Extending the
equality across steps also requires relating the evolving optimizer states.

```lean (name := otAdamwThm)
-- Compare one AdamW step with the corresponding Adam state
-- when decay is zero.
open Optim.AdamW in
#check @update_weight_decay_zero_parameters_eq_adam_real
```

```leanOutput otAdamwThm (whitespace := lax)
@update_weight_decay_zero_parameters_eq_adam_real : ∀ {s : Shape} (state : AdamW.State ℝ s)
  (parameters gradients : Tensor ℝ s), state.weightDecay = 0 → (AdamW.update state parameters
  gradients).parameters = (Adam.update { learningRate := state.learningRate, beta1 :=
  state.beta1, beta2 := state.beta2, epsilon := state.epsilon, firstMoment := state.firstMoment,
  secondMoment := state.secondMoment, stepCount := state.stepCount } parameters
  gradients).parameters
```

The large record in this signature is the Adam state corresponding to the supplied AdamW state.
Every moment field and the step counter are copied explicitly. This tells us exactly which two
updates are compared: same parameters, same gradients, same moment history, same hyperparameters,
with AdamW's decay field set to zero. The braces in `{s : Shape}` allow Lean to infer the tensor
shape from those arguments. Since the statement quantifies over `state`, it applies to an
intermediate state as well as an initialized one, provided the zero-decay equality is available.

The scalar type is `ℝ`. The proof simplifies the decay term using algebraic laws such as
`x * 0 = 0`, which fail for IEEE infinities and NaNs. The floating run above checks agreement on
one finite input; the real theorem does not establish it for arbitrary floating states. The
chapter on
{ref "runtime-approximation"}[runtime approximation] is where the gap between those two claims gets
its own bounds.

The conclusion is about the `.parameters` field, not the state as a whole, and not the loss curve
of a training run. The moment buffers are equal too, but that takes its own theorem, and
generalization would require a separate statistical theorem with additional assumptions.

The hypothesis `state.weightDecay = 0` lets a caller apply the result to any state for which it
can prove that field is zero.

## Optimizer State And Update Laws

For tensor optimizers with richer state, the
{src "NN/MLTheory/Optimization/OptimizerLaws.lean"}[optimizer laws API] exposes the same pattern.
An optimizer is a record with a state type, an initializer, and an update, so "which optimizer" is
a value that theorems can quantify over rather than a string in a config file:

```lean (name := otLaws)
-- Inspect state-preserving stream composition and the
-- generic recurrence bridge.
#check @TensorOptimizer.sgd
#check @TensorOptimizer.adamw
#check @TensorOptimizer.runSteps_append
#check @StepSpec
#check @StepSpec.runSteps_eq_optimizer_runSteps
```

```leanOutput otLaws (whitespace := lax)
@TensorOptimizer.sgd : {α : Type} → [inst : Storage α] → [inst_1 : Context α] → [DecidableRel
  fun x1 x2 => x1 > x2] → α → TensorOptimizer α
```

```leanOutput otLaws (whitespace := lax)
@TensorOptimizer.adamw : {α : Type} → [inst : Storage α] → [inst_1 : Context α] → [DecidableRel
  fun x1 x2 => x1 > x2] → α → α → α → α → α → TensorOptimizer α
```

```leanOutput otLaws (whitespace := lax)
@TensorOptimizer.runSteps_append : ∀ {α : Type} [inst : Storage α] [inst_1 : Context α] (opt :
  TensorOptimizer α) {s : Shape} (current : Step α s (opt.State s)) (left right : Array (Tensor
  α s)), opt.runSteps current (left ++ right) = opt.runSteps (opt.runSteps current left) right
```

```leanOutput otLaws (whitespace := lax)
@StepSpec : {α : Type} → [inst : Storage α] → [inst_1 : Context α] → TensorOptimizer α → Type
```

```leanOutput otLaws (whitespace := lax)
@StepSpec.runSteps_eq_optimizer_runSteps : ∀ {α : Type} [inst : Storage α] [inst_1 : Context α]
  {opt : TensorOptimizer α} (law : StepSpec opt) {s : Shape} (current : Step α s (opt.State s))
  (gradients : Array (Tensor α s)), law.runSteps current gradients = opt.runSteps current
  gradients
```

`TensorOptimizer.adamw` takes five scalar arguments in order: learning rate, weight decay,
`beta1`, `beta2`, and epsilon. Their types are identical, so the explicit calls above are needed
to identify which setting each experiment uses.

`TensorOptimizer.runSteps_append` is the associativity of running a stream of gradients, and it
applies to every packaged optimizer at once, because the optimizer is an argument. It is the lemma
that lets a proof about an epoch be assembled from proofs about batches.

The nested call on the right of `runSteps_append` matters. Its initial value is the complete
result of running `left`, including the new moment buffers and counter. Reinitializing the
optimizer before `right` would describe a different computation. The theorem also preserves the
order of gradients; it permits splitting a stream, not shuffling it. In a trainer that recomputes
gradients from the current parameters, one must additionally show that the two executions produce
the same gradient stream. The append law itself takes that stream as input rather than generating
it from a loss function.

`StepSpec` is reserved for the stronger situation in which a separately stated mathematical
recurrence is proved equal to an executable update; its generic run theorem then lifts that
agreement over a stream of gradients. Concrete optimizer equations remain in
`NN.Runtime.Optim.Optimizers`, while optimizer-specific theory files establish algebraic invariants
and comparisons that are not restatements of those definitions.

GaLore-style projected SGD follows the same rule. The executable projection interface and the
identity-projection agreement theorem live together in
{src "NN/Runtime/Optim/Optimizers.lean"}[NN.Runtime.Optim.Optimizers].
The theorem says that choosing the identity projector recovers ordinary SGD; it does not claim that
an arbitrary learned low-rank projector preserves the update.

# Optimizer Extension Points: Muon And GaLore-Style Updates

Modern optimizer work often mixes a familiar base update with a specialized backend. TorchLean
models that explicitly. The runtime layer gives the executable update equation. The theory layer
states what has to be true of the backend output before the update can be cited in a proof.

Muon is represented as momentum followed by an orthogonalization backend. One step first updates the
momentum buffer

$$`m_{t+1}=\beta m_t+g_t,`

then asks a backend for the direction used in the parameter update. With the identity backend,
Muon's parameter update is exactly momentum SGD. With a certified matrix backend, the claim is about
the actual direction returned by that backend: exact column orthogonality

$$`Q^\top Q = I`

or an approximate Gram-residual bound

$$`\|Q^\top Q-I\|_\infty \le \varepsilon.`

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

Each diagonal entry of $`Q^\top Q` is a column's squared length, and each off-diagonal entry
is the dot product of two distinct columns. Thus an approximate entrywise Gram certificate
controls two things at once: the error in unit length and the failure of pairwise orthogonality.
It does not choose a preferred direction relative to the objective gradient. For a Muon update,
this is useful geometric information about the backend's matrix, but an objective-descent theorem
would still need to connect that direction to the loss. The shapes also matter: $`m` rows and
$`n` orthonormal columns require enough ambient coordinates to accommodate those columns.

## Newton-Schulz Iteration

The Newton-Schulz orthogonalizer forms the right Gram matrix $`G=Q^\top Q` and evaluates
$`aQ+b\,QG+c\,(QG)G` repeatedly. Over the reals this is the same odd polynomial as the left-Gram
form; the two evaluation orders need not agree in floating arithmetic. Compare the Muon
coefficient set below with the classical cubic $`\tfrac32 Q-\tfrac12 QQ^\top Q`.
The exact fixed-point theorem
in the library, `newtonSchulzStep_hasExactColumnGram_of_exact_column_gram_of_sum_square_one`, needs
$`(a+b+c)^2=1`, so start by asking whether each coefficient set even qualifies:

```lean (name := otMuonSums)
-- An exactly orthonormal input is rescaled by the sum of
-- these coefficients.
def otQuintic : NewtonSchulzCoeffs Float :=
  { a := 3.4445, b := -4.7750, c := 2.0315 }

def otCubic : NewtonSchulzCoeffs Float :=
  { a := 1.5, b := -0.5, c := 0.0 }

#eval (otQuintic.a + otQuintic.b + otQuintic.c,
       otCubic.a + otCubic.b + otCubic.c)
```

```leanOutput otMuonSums (whitespace := lax)
(0.701000, 1.000000)
```

Only the cubic coefficients satisfy that condition. When the columns are already orthonormal,
the Gram matrix is the identity, so one polynomial step scales the whole matrix by the sum of
the coefficients. Its new Gram matrix is scaled by the square of that sum. This explains the
theorem's hypothesis and why it does not apply to the displayed quintic coefficients.

Now watch the residual. The matrix below has been divided by its Frobenius norm, which is what
Muon does to its momentum buffer before orthogonalizing:

```lean (name := otCubicRun)
-- Track the Gram residual after several cubic iterations,
-- then test exact zero.
def otRaw : Tensor Float [2, 2] := [[1.0, 0.5], [0.0, 1.0]]

def otScaled : Tensor Float [2, 2] :=
  [[0.6666666666666666, 0.3333333333333333],
   [0.0, 0.6666666666666666]]

/-- Largest absolute entry of `columnGramResidual`, the
$`\|\cdot\|_\infty` that `HasApproxColumnGram` bounds. -/
def otResidualMax {m n : ℕ}
    (Q : Tensor Float [m, n]) : Float :=
  (columnGramResidual Q).foldl
    (fun acc x =>
      if acc.isNaN || x.isNaN then 0.0 / 0.0
      else max acc x.abs) 0.0

#eval otResidualMax otScaled

#eval otResidualMax (newtonSchulzIter otCubic 1 otScaled)

#eval otResidualMax (newtonSchulzIter otCubic 3 otScaled)

#eval otResidualMax (newtonSchulzIter otCubic 6 otScaled)

#eval columnGram (newtonSchulzIter otCubic 10 otScaled)

#eval otResidualMax
        (newtonSchulzIter otCubic 10 otScaled) == 0.0
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

The residual falls from $`0.556` to $`0.331` to $`0.023`, and by six steps it prints as zero.
The Gram matrix at ten steps prints as the identity, negative zeros in the off-diagonal included.

The final equality test returns `false`: the computed residual is nonzero, but too small for the
six-decimal display. An approximate certificate would need an explicit `eps` and a proof that
every residual entry is within it. The output alone does not supply that proof or an exact Gram
certificate. This finite trace does not establish
what
every later iterate does; floating iterations can settle, cycle, or reach an exact representable
solution depending on the input. This is the same lesson the {ref "floats"}[floating point chapter]
draws from Flocq's
treatment of rounding {Informal.citep flocq2011}[]: a decimal rendering can conceal the difference
between a small error and zero.

The quintic coefficients on an unnormalized matrix do something worse than converge slowly:

```lean (name := otQuinticRun)
-- Compare the quintic iteration on the raw matrix and its
-- normalized version.
#eval otResidualMax (newtonSchulzIter otQuintic 1 otRaw)

#eval otResidualMax (newtonSchulzIter otQuintic 3 otRaw)

#eval otResidualMax (newtonSchulzIter otQuintic 5 otScaled)
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

Starting from `otRaw`, whose largest singular value exceeds one, the residual grows from $`0.568`
to $`6182.9` in two more steps, in the displayed finite run. On the normalized input the residual
after five steps is about
$`0.32`; that observation establishes neither eventual convergence nor stalling. The polynomial
does not control arbitrary singular values, so its input scaling matters.

The scalar picture explains the sensitivity to normalization. Along a singular direction with
singular value $`s`, the matrix polynomial applies $`s\mapsto as+bs^3+cs^5` in real
arithmetic. For the cubic coefficients, this is $`s\mapsto(3s-s^3)/2`, with a fixed point
at $`s=1`. For large $`s`, the highest-degree term can dominate instead of moving the value
toward one. Dividing by the Frobenius norm limits the initial singular values, but that alone
is not a proof of convergence for an arbitrary coefficient choice. The fixed-point theorem
above answers the narrower algebraic question of what one step does when the Gram matrix is
already exactly the identity.

`newtonSchulzOrthogonalizer` leaves normalization to the caller. A residual-checked backend then
checks the direction it actually returns. Input normalization and output certification are
separate obligations.

## Muon Backend Contracts

The
{src "NN/MLTheory/Optimization/Muon.lean"}[Muon theory file] packages these cases as exact,
approximate, and checked-backend contracts. QR-backed directions give an exact path under
positive-pivot hypotheses. Newton-Schulz-style directions give a residual-checked approximate path,
together with fixed-point exact statements when the iteration has reached the corresponding
algebraic condition.

The following theorems consume checked-backend and QR hypotheses:

```lean
-- These consumers require evidence for the backend used by
-- the actual update.
#check @exactCertifiedStep_of_checkedBackend
#check @approxCertifiedStep_of_checkedBackend
#check @update_has_exact_certified_step_qr
```

The next theorem derives a Gram bound from the success predicate of a
`CheckedApproxOrthogonalizer`, whose contract connects success to that bound:

```lean (name := otChecked)
-- Success must be established for the freshly updated
-- momentum buffer.
#check @checkedBackend_updateDirection_hasApproxColumnGram
```

```leanOutput otChecked (whitespace := lax)
@checkedBackend_updateDirection_hasApproxColumnGram : ∀ {α : Type} [inst : Storage α] [inst_1 :
  Context α] {m n : ℕ} {eps : α} (backend : CheckedApproxOrthogonalizer α m n eps) (learningRate
  momentum : α) (momentumBuffer parameters gradients : MatrixTensor α m n), backend.Success
  (update { learningRate := learningRate, momentum := momentum, momentumBuffer :=
  momentumBuffer, orthogonalizer := backend.orthogonalizer } parameters
  gradients).optimizerState.momentumBuffer → HasApproxColumnGram eps
  (backend.orthogonalizer.apply (update { learningRate := learningRate, momentum := momentum,
  momentumBuffer := momentumBuffer, orthogonalizer := backend.orthogonalizer } parameters
  gradients).optimizerState.momentumBuffer)
```

`backend.Success` appears to the left of the arrow, applied to the *post-momentum* buffer, which is
the matrix the orthogonalizer will actually be handed. A backend may be fast, randomized,
iterative, or external. Lean only uses it as an orthogonalizing step after that predicate has been
proved for that buffer, using the backend's contract. Executing the update alone does not
discharge the success premise.

Read the long conclusion from its innermost expression outward. `update ...` first constructs
the new momentum state. `.optimizerState.momentumBuffer` selects the matrix sent to the backend,
and `backend.orthogonalizer.apply` computes its direction. `HasApproxColumnGram eps` is then a
property of that direction. The success premise uses the same updated buffer, so a certificate
for yesterday's buffer cannot be substituted without another argument. This matching of the
certificate input with the actual backend input is the essential content of a checked-step
interface.

The consumer side of the boundary is in
{src "NN/Examples/Optimization/MuonCertificates.lean"}[NN.Examples.Optimization.MuonCertificates],
where the certificates are combined with the update equation to say something about a parameter
step. Start with its `Concrete` namespace: the direction is the column
$`Q=(3/5,4/5)^\mathsf{T}`, whose Gram condition reduces to $`9/25+16/25=1`. The column
$`(1,1)^\mathsf{T}` is rejected because its squared length is two. With zero momentum, parameters
$`(2,3)^\mathsf{T}`, and learning rate $`1/10`, the certified update is
$`(97/50,73/25)^\mathsf{T}`:

```lean
-- The unit column supplies both a Gram certificate and a
-- concrete parameter step.
open NN.Examples.Optimization.MuonCertificates.Concrete in
example :
    ExactCertifiedStep state parameters direction
      direction :=
  step_certified

open NN.Examples.Optimization.MuonCertificates.Concrete in
open Optim.Muon in
example :
    (update state parameters direction).parameters
      = column (97 / 50) (73 / 25) :=
  updated_parameters_eq
```

This case supplies the whole certificate without an unproved backend hypothesis. Its identity
backend works because the chosen buffer already has a unit column; it does not orthogonalize
arbitrary gradients or establish convergence. Build the tutorial with
`lake build NN.Examples.Optimization` and inspect `Concrete.step_certified` in the Infoview.
The later wrappers explain how to consume certificates for general QR and Newton–Schulz backends:

```lean (name := otQrCert)
-- Read the shared direction witness, then inspect the
-- theorem’s logical dependencies.
open NN.Examples.Optimization.MuonCertificates in
#check @qr_update_step_direction_has_exact_gram

open NN.Examples.Optimization.MuonCertificates in
#print axioms qr_update_step_direction_has_exact_gram
```

```leanOutput otQrCert (whitespace := lax)
@qr_update_step_direction_has_exact_gram : ∀ {m n : ℕ} (learningRate momentum : ℝ)
  (momentumBuffer parameters gradients : Tensor ℝ [m, n]), HasPositiveQRPivots (update {
  learningRate := learningRate, momentum := momentum, momentumBuffer := momentumBuffer,
  orthogonalizer := qrOrthogonalizer } parameters gradients).optimizerState.momentumBuffer → ∃
  direction, HasExactColumnGram direction ∧ (update { learningRate := learningRate, momentum :=
  momentum, momentumBuffer := momentumBuffer, orthogonalizer := qrOrthogonalizer } parameters
  gradients).parameters = parameters.subSpec (direction.scaleSpec learningRate)
```

```leanOutput otQrCert (whitespace := lax)
'NN.Examples.Optimization.MuonCertificates.qr_update_step_direction_has_exact_gram' depends on
  axioms: [propext, Classical.choice, Quot.sound]
```

The conclusion supplies a direction with orthonormal columns and identifies the new parameters
as the old parameters minus the learning rate times that direction. The
hypothesis `HasPositiveQRPivots` is the admissibility condition for the QR path, and it is on the
momentum buffer, not on the raw gradient. The axiom list is the usual classical trio and nothing
library-specific: this proof uses `Classical.choice`, propositional extensionality, and quotient
soundness.

The existential `∃ direction` binds one matrix shared by both conjuncts. A caller receives
orthonormality of that matrix and the equation showing that it was used in the parameter step.
Keeping the same witness in both places rules out proving a Gram fact about an unrelated matrix.
In the concrete column example, the parameter equation is simply
$`2-(1/10)(3/5)=97/50` and $`3-(1/10)(4/5)=73/25`. The certificate explains the direction's
geometry; the subtraction equation connects that geometry to the two parameters that change.

## GaLore Projection Contracts

GaLore-style code is different. GaLore is a gradient-projection strategy, not a single optimizer
name. The runtime object is a projector/lift pair around a base update:

$$`p_{t+1}=p_t-\eta\,\mathrm{lift}(\mathrm{project}(g_t)).`

The current checked baseline says that if the projector is the identity, projected SGD is ordinary
SGD. A future low-rank projector or refresh policy can optimize memory and matrix structure, but it
has to state its own projection contract instead of being hidden inside the word "optimizer."

This naming is reflected in the trainer API. Standard trainer configs use names such as
`optim.sgd`, `optim.adamW`, and `optim.adaDelta`. Runtime-level extension points use more explicit
names such as `optim.muon.optimizer` and `optim.galore.sgd`, because those calls must record
their backend or projection as part of the mathematical object.

# Gradient Descent As A Contractive Map

The core convergence theorem studies the ideal deterministic update
`step eta g x = x - eta * g x`.

In mathematical notation:

$$`x_{t+1}=x_t-\eta g(x_t).`

Before reading the general theorem, run the recurrence for

$$`f(x)=x^2,\qquad \nabla f(x)=2x,\qquad \eta=\frac14.`

This example is chosen so that the three constants the abstract theorem asks for are all known
exactly. The gradient $`x\mapsto 2x` is strongly monotone with $`\mu=2` and Lipschitz with
$`L=2`, so the contraction factor $`q=1-2\eta\mu+\eta^2L^2` can be computed rather than estimated.
The demonstration uses rationals so that every printed iterate is exact:

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

#eval Tensor.mapSpec (fun x => x ^ 2) (demoTrace 4 4)

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

The contraction factor is $`1/4`. The step size condition $`\eta L^2<2\mu` reads $`1<4`, which is
the second output, so this step size is inside the safe range with room to spare. The iterates are
`4, 2, 1, 1/2, 1/4`, halving as the `example` at the end of the block proves they must. The squared
distances to the minimizer are `16, 4, 1, 1/4, 1/16`, each one a factor of $`1/4` below the last.

The fifth output compares the run with the library theorem's bound,
$`\|x_k-x^\star\|^2\le q^k\|x_0-x^\star\|^2`. On this example the two sides are *equal* at
$`k=2`, as checked by the rational equality test. The symbolic example also proves that every
step halves the iterate. Thus the rate is attained, ruling out a uniform strict
improvement at these particular values $`\mu=L=2` and
$`\eta=1/4`; it does not show that the formula is optimal for every $`\mu<L`.

The general theorem replaces this one-dimensional calculation by assumptions on an abstract
gradient map, and the exponent $`k` survives the generalization.

If `g` is strongly monotone with parameter $`\mu`, Lipschitz with parameter $`L`, and the step size
$`\eta` is in the safe range, then one step is contractive:

> the squared distance between `step eta g x` and `step eta g y` is at most $`q` times the squared
> distance between $`x` and $`y`; the distance itself has factor $`\sqrt q`.

The
{src "NN/MLTheory/Optimization/GDLinearConvergence.lean"}[linear convergence API] names the
predicate `StrongMonotone mu g`. Informally, it says that the inner product of $`g(x)-g(y)` with
$`x-y` dominates $`\mu\lVert x-y\rVert^2`.

The space `E` need not be a scalar. `NormedAddCommGroup E` supplies vector addition, subtraction,
and a norm; `InnerProductSpace ℝ E` supplies the real inner product used in the cross term.
A finite collection of real model parameters can be viewed in such a space once its chosen
representation and norm are connected to this interface. `L : NNReal` carries nonnegativity
in its type, and `↑L` is the coercion to a real number for the formula for `q`. Strong monotonicity
is a global comparison of two inputs of `g`, so checking a positive gradient at one point would
not establish it.

The two analytic hypotheses can be read as:

$$`\langle g(x)-g(y),x-y\rangle\ge \mu\|x-y\|^2`

and

$$`\|g(x)-g(y)\|\le L\|x-y\|.`

The following signatures show the one-step bound, its iteration, and the step-size conditions:

```lean (name := otGdThms)
-- Follow the one-step bound through iteration and the
-- explicit step-size conditions.
#check @StrongMonotone
#check @step_norm_sq_le
#check @dist_sq_iterate_le_of_q_lt_one
#check @q_lt_one_of_mul_sq_lt
#check @dist_sq_iterate_le_of_step_size
#print axioms Optim.GD.dist_sq_iterate_le_of_step_size
```

```leanOutput otGdThms (whitespace := lax)
@StrongMonotone : {E : Type} → [inst : NormedAddCommGroup E] → [InnerProductSpace ℝ E] → ℝ → (E
  → E) → Prop
```

```leanOutput otGdThms (whitespace := lax)
@step_norm_sq_le : ∀ {E : Type} [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  (η μ : ℝ), 0 ≤ η → ∀ {L : NNReal} (g : E → E), StrongMonotone μ g → LipschitzWith L g → ∀ (x y
  : E), ‖step η g x - step η g y‖ ^ 2 ≤ (1 - 2 * η * μ + η ^ 2 * ↑L ^ 2) * ‖x - y‖ ^ 2
```

```leanOutput otGdThms (whitespace := lax)
@dist_sq_iterate_le_of_q_lt_one : ∀ {E : Type} [inst : NormedAddCommGroup E] [inst_1 :
  InnerProductSpace ℝ E] (η μ : ℝ), 0 ≤ η → ∀ {L : NNReal} (g : E → E), StrongMonotone μ g →
  LipschitzWith L g → ∀ {xStar x : E}, g xStar = 0 → 0 ≤ q η μ L → q η μ L < 1 → ∀ (k : ℕ),
  ‖(step η g)^[k] x - xStar‖ ^ 2 ≤ q η μ L ^ k * ‖x - xStar‖ ^ 2
```

```leanOutput otGdThms (whitespace := lax)
q_lt_one_of_mul_sq_lt : ∀ (η μ : ℝ) (L : NNReal), 0 < η → η * ↑L ^ 2 < 2 * μ → q η μ L < 1
```

```leanOutput otGdThms (whitespace := lax)
@dist_sq_iterate_le_of_step_size : ∀ {E : Type} [inst : NormedAddCommGroup E] [inst_1 :
  InnerProductSpace ℝ E] (η μ : ℝ) {L : NNReal} (g : E → E), StrongMonotone μ g → LipschitzWith
  L g → ∀ {xStar x : E}, g xStar = 0 → 0 ≤ μ → μ ≤ ↑L → 0 < η → η * ↑L ^ 2 < 2 * μ → ∀ (k : ℕ),
  ‖(step η g)^[k] x - xStar‖ ^ 2 ≤ q η μ L ^ k * ‖x - xStar‖ ^ 2 ∧ q η μ L < 1
```

```leanOutput otGdThms (whitespace := lax)
'Optim.GD.dist_sq_iterate_le_of_step_size' depends on axioms: [propext, Classical.choice,
  Quot.sound]
```

`step_norm_sq_le` is the one-step inequality, with the factor $`1-2\eta\mu+\eta^2L^2` written out
in the conclusion rather than abbreviated. To obtain it, expand the squared norm of the
difference between two updates. The expansion has the original squared distance, a cross term
with coefficient minus twice the step size, and the squared gradient difference. Strong
monotonicity bounds the cross term from above because its coefficient is nonpositive.
Lipschitzness bounds the squared gradient difference. Combining the three coefficients gives
the displayed factor. The
{src "NN/MLTheory/Optimization/StronglyConvexGD.lean"}[strongly convex gradient descent API] then
iterates it. `dist_sq_iterate_le_of_q_lt_one` is the statement readers should remember:

> If the contraction factor $`q(\eta,\mu,L)` is nonnegative and strictly below one, then after
> $`k` gradient descent steps the squared distance to the root is at most $`q^k` times the initial
> squared distance.

The factor $`q^k` is in the conclusion, not only in the prose, and `(step η g)^[k]` is Mathlib's
function iteration, so "after $`k` steps" is literal {Informal.citep mathlib2020}[]. Note also
`g xStar = 0`: the caller supplies a root. The theorem neither proves that a root exists nor
requires `g` to have been identified as the gradient of an objective.

The axiom report describes the logical dependencies of this proof. `propext` identifies logically
equivalent propositions, `Classical.choice` supplies classical choice, and `Quot.sound` supports
reasoning with quotients. The report names these foundations rather than a new assumption that
gradient descent converges. The mathematical conditions on the update map remain the explicit
premises in the theorem signature.

The iteration proof uses the root equation to identify a fixed point:
$`T_\eta(x^\star)=x^\star-\eta g(x^\star)=x^\star`. At step zero, the estimate is
an equality because function iteration is the identity and $`q^0=1`. At the next step, the
one-step inequality multiplies the previous squared-error bound by `q`; its nonnegativity
preserves the inequality. Repeating this argument produces the exponent. This is why `0 ≤ q`
is listed separately from `q < 1`: one condition supports induction, while the other makes the
geometric bound decrease with the number of steps.

Two companions turn the abstract condition on $`q` into a condition on the step size.
`q_lt_one_of_mul_sq_lt` shows $`q<1` whenever $`0<\eta` and $`\eta L^2<2\mu`, and
`dist_sq_iterate_le_of_step_size` packages the geometric bound together with $`q<1` under
$`0\le\mu\le L` and that step-size inequality, so a caller checks numbers rather than an inequality
about $`q`. The demo above checked exactly those numbers: $`\eta L^2=1` against $`2\mu=4`.

The contraction and convergence shapes are:

$$`\|T_\eta(x)-T_\eta(y)\|^2\le q\|x-y\|^2`

with a typical monotone/Lipschitz factor

$$`q=1-2\eta\mu+\eta^2L^2,`

and then

$$`\|x_t-x^\star\|^2\le q^t\|x_0-x^\star\|^2.`

The scalar specialization is proved separately as `ScalarGD.error_abs_contract_real`, which is the
version closest to the rational demo above:

```lean
-- The scalar result isolates the error multiplier for a
-- quadratic objective.
#check @ScalarGD.error_abs_contract_real
```

The result is the standard smooth/strongly-monotone contraction argument found in convex
optimization texts such as Nesterov's
[*Introductory Lectures on Convex
Optimization*](https://link.springer.com/book/10.1007/978-1-4419-8853-9). TorchLean attaches this
familiar rate to the exact update map and assumptions used by its training theorems.

# Smoothness And Strong Convexity

Most papers state convergence using smoothness and strong convexity of an objective `f`, not
strong monotonicity of an abstract gradient map. TorchLean keeps both vocabularies and proves the
bridge between them.

The informal first order strong convexity condition says that `f y` lies above the tangent model at
$`x` plus a quadratic term with coefficient $`\mu/2`.

The
{src "NN/MLTheory/Optimization/SmoothStrongConvexBridge.lean"}[smooth strong convex bridge API]
turns that objective level statement into a gradient map statement. The theorem to recognize is
`strongMonotone_gradient_of_firstOrderStrongConvex`: under the first order strong convexity
hypothesis, the gradient is strongly monotone. That bridge lets a training theorem move from "the
loss has these analytic assumptions" to "the update map is contractive."

```lean (name := otBridge)
-- Convert the objective’s first-order inequality into
-- strong monotonicity.
#check @FirstOrderStrongConvex
#check @strongMonotone_gradient_of_firstOrderStrongConvex
```

```leanOutput otBridge (whitespace := lax)
@FirstOrderStrongConvex : {E : Type} → [inst : NormedAddCommGroup E] → [InnerProductSpace ℝ E] →
  [CompleteSpace E] → ℝ → (E → ℝ) → Prop
```

```leanOutput otBridge (whitespace := lax)
@strongMonotone_gradient_of_firstOrderStrongConvex : ∀ {E : Type} [inst : NormedAddCommGroup E]
  [inst_1 : InnerProductSpace ℝ E] [inst_2 : CompleteSpace E] (μ : ℝ) {f : E → ℝ},
  FirstOrderStrongConvex μ f → StrongMonotone μ fun x => gradient f x
```

The conclusion is `StrongMonotone μ fun x => gradient f x`, which is precisely the hypothesis
`step_norm_sq_le` wants, with the same $`\mu`. The extra `[CompleteSpace E]` instance on
`FirstOrderStrongConvex` is there because `gradient` is Mathlib's Fréchet-derivative-based
gradient, which needs the inner product space to be complete for the Riesz identification to be
available. The proof applies first-order strong convexity in both directions and adds the two
inequalities: the objective values cancel, leaving the required lower bound on the gradient
inner product. Applying the convergence theorem still requires a Lipschitz bound, a root, and
the step-size hypotheses.

The first-order inequality itself mentions Mathlib's `gradient f x`. To obtain it from a usual
convex-analysis specification, the same source provides a bridge from `StrongConvexOn` on the
whole space together with differentiability. This supplies the mathematical justification for
treating the named gradient as the tangent term. Once first-order inequalities are available at
both $`x` and $`y`, adding them cancels $`f(x)` and $`f(y)`, and the two quadratic terms add to
$`\mu\|x-y\|^2`. Neither that cancellation nor strong monotonicity supplies the separate
upper bound on gradient differences required by `LipschitzWith`.

The bridge is useful because autograd theorems usually speak about derivatives of a loss, while
convergence theorems often speak about a gradient map. The theorem connects those two vocabularies
without treating "gradient" as an informal word.

# Verified Training

The autograd theorems explain why the gradient path computes the intended derivative. Runtime
approximation theorems explain how close a rounded update is to the ideal update. Optimization
theory is where we state what that update means as an algorithm.

For example, a theorem about a full training run can be read as a composition of three facts:

- an autograd theorem saying the gradient is the adjoint derivative of the loss;
- a runtime approximation theorem saying the executable update is close to the ideal update;
- an optimization theorem saying the ideal update contracts under smoothness and strong convexity
  hypotheses.

Those hypotheses matter. The optimization layer avoids turning a loss curve into a convergence
claim: convexity, smoothness, strong monotonicity, and step size conditions remain visible in the
theorem statement.

The examples distinguish algorithmic effects from numerical ones. The real AdamW theorem compares
two update rules; the Adam result's difference from $`0.7`, about $`3\times10^{-9}`, comes
primarily from epsilon in the recurrence. Separately, the Newton-Schulz residual prints as zero
while its equality test returns false. A runtime error theorem must compare an execution with
the same ideal recurrence, including its epsilon and backend choices.

There is also a state boundary between these equations and a native trainer. Runtime Adam-family
optimizers maintain moment buffers and a step counter for each parameter, while checkpoints must
restore those states with the same parameter identity and ordering. The pure `TensorOptimizer` and
`StepSpec` theorems prove the recurrence they state; they do not by themselves prove allocation,
aliasing, checkpoint restoration, or the mutable runtime state machine. An end-to-end optimizer
claim needs a bridge showing that the trainer's stored state implements the same recurrence.

For a rounded implementation, the next useful equation is a one-step comparison with the ideal
map, such as $`\|\widehat T(x)-T(x)\|\le e`. Combined with a distance contraction factor
$`r=\sqrt q`, the triangle inequality gives an error recurrence of the form
$`d_{t+1}\le e+r d_t` when those bounds apply at the relevant iterates. This explains why
repeated small execution errors need to be accumulated through training rather than checked only
at the final step. It also identifies the necessary inputs to such an argument: a uniform or
stepwise numerical error bound and an ideal contraction theorem for the same update.

# Training Theorem Assumptions

A theorem about a particular training run must identify the gradient, scalar semantics,
step-size condition, and any backend contract such as a Muon orthogonalizer certificate or a
projector law. The results here supply selected components; the trainer must satisfy their
premises and connect its mutable state to the stated recurrence.

# Sources

The files behind this chapter:

- {src "NN/MLTheory/Optimization/FirstOrder.lean"}[FirstOrder.lean], the Adam and AdamW update
  equations and the zero-decay agreement theorem.
- {src "NN/MLTheory/Optimization/OptimizerLaws.lean"}[OptimizerLaws.lean], `TensorOptimizer`, the
  packaged optimizers used in the run above, and `StepSpec`.
- {srcDir "NN/MLTheory/Optimization/Muon"}[the Muon directory], the Gram predicates,
  the Newton-Schulz iteration, and the checked-backend contracts.
- {src "NN/Examples/Optimization/MuonCertificates.lean"}[MuonCertificates.lean], the consumer
  theorems that combine a certificate with the update equation.
- {src "NN/MLTheory/Optimization/GDLinearConvergence.lean"}[GDLinearConvergence.lean],
  `StrongMonotone`, `q`, and the one-step contraction.
- {src "NN/MLTheory/Optimization/StronglyConvexGD.lean"}[StronglyConvexGD.lean], the iterated
  geometric bounds and the step-size packaging.
- {src "NN/MLTheory/Optimization/SmoothStrongConvexBridge.lean"}[SmoothStrongConvexBridge.lean],
  first order strong convexity and the bridge to strong monotonicity.

# References

The gradient-descent proof follows the contraction argument in Yurii Nesterov's
[*Introductory Lectures on Convex
Optimization*](https://link.springer.com/book/10.1007/978-1-4419-8853-9), Springer 2004.
