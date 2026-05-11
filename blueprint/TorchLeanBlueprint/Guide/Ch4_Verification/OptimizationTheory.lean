import VersoManual
import VersoBlueprint

open Verso.Genre Manual

#doc (Manual) "Optimization Theory" =>
%%%
tag := "optimization-theory"
%%%

Training code executes updates. Optimization theory explains when those updates should make
progress.

A TorchLean training run may call SGD or Adam, but a convergence theorem cannot come from the name
of the optimizer alone. It needs an ideal update map, assumptions on the objective or gradient, and
a step size condition.

The runtime can execute SGD, Adam style updates, or PPO losses. The optimization theory material
names the ideal update and the assumptions under which a theorem is allowed to say "this step moves
closer." That is the useful comparison with PyTorch: PyTorch faithfully runs an optimizer step;
TorchLean can also state the mathematical contract that would justify calling the step contractive
or convergent.

# The Optimization Contract

The optimization contract has three layers:

- *Runtime update*: the concrete operation that writes new values into parameter tensors, such as
  one SGD, momentum, or Adam style step.
- *Ideal update*: the mathematical map the runtime update is intended to approximate, for example
  `x ↦ x - η g x`.
- *Convergence theorem*: the conditional theorem saying that iterating the ideal map makes progress
  under assumptions such as strong monotonicity, Lipschitzness, and a safe step size.

Keeping these layers separate stops a common overclaim. A decreasing loss curve is evidence that an
optimizer is doing something useful; it is not itself a proof that the update map is contractive.

# First Order Updates

The first object is a first order optimizer state: parameters, gradients, an optional buffer, and a
time counter. The simplest theorem says that the named update exposes exactly the pieces we expect:
the returned record contains the next parameters, the next optimizer buffer, and a time counter
equal to `state.time + 1`. In Lean this is intentionally small algebra.

The [FirstOrder optimization API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Optimization/FirstOrder.lean) contains
the corresponding projection theorems: `update_params_eq`, `update_buffer_eq`, and
`update_time_eq`. These do not prove convergence. They make the update record transparent so later
proofs can cite the actual algebra instead of relying on a prose description of the optimizer.

Low rank optimizer facts follow the same pattern. The
[low rank optimization API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Optimization/LowRank.lean) records equations such as
"with the identity projection, projected SGD is ordinary SGD" and "identity low rank state agrees
with the corresponding momentum update." These are algebraic compatibility facts, not global
optimality claims for every low rank method.

# Gradient Descent As A Contractive Map

The core convergence theorem is not "SGD always converges." The theorem studies the ideal update
`step eta g x = x - eta * g x`.

In mathematical notation:

$$`x_{t+1}=x_t-\eta g(x_t).`

If `g` is strongly monotone with parameter $`\mu`, Lipschitz with parameter $`L`, and the step size
$`\eta` is in the safe range, then one step is contractive:

> the distance between `step eta g x` and `step eta g y` is at most `q` times the distance between
> `x` and `y`.

The [linear convergence API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Optimization/GDLinearConvergence.lean) names the
important predicate `StrongMonotone mu g`. Informally, it says that the inner product of
`g x - g y` with `x - y` dominates `mu * ||x - y||^2`.

The two analytic hypotheses can be read as:

$$`\langle g(x)-g(y),x-y\rangle\ge \mu\|x-y\|^2`

and

$$`\|g(x)-g(y)\|\le L\|x-y\|.`

and proves the one step inequality `step_norm_sq_le`. The
[strongly convex gradient descent API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Optimization/StronglyConvexGD.lean) then
iterates the inequality. Its theorem `dist_sq_iterate_le_of_q_lt_one` is the statement readers
should remember:

> If the contraction factor `q η μ L` is nonnegative and strictly below one, then repeated gradient
> descent steps shrink the squared distance to the reference point geometrically.

That is the real mathematical content. The Lean theorem is deliberately conditional because those
conditions are exactly what experiments cannot infer from a loss curve.

The contraction and convergence shapes are:

$$`\|T_\eta(x)-T_\eta(y)\|^2\le q\|x-y\|^2`

with a typical monotone/Lipschitz factor

$$`q=1-2\eta\mu+\eta^2L^2,`

and then

$$`\|x_t-x^\star\|^2\le q^t\|x_0-x^\star\|^2.`

# Smoothness, Strong Convexity, And The Bridge

Most papers state convergence using smoothness and strong convexity of an objective `f`, not
strong monotonicity of an abstract gradient map. TorchLean keeps both vocabularies and proves the
bridge between them.

The informal first order strong convexity condition says that `f y` lies above the tangent model at
`x` plus a quadratic term with coefficient `mu / 2`.

The [smooth strong convex bridge API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Optimization/SmoothStrongConvexBridge.lean)
turns that objective level statement into a gradient map statement. The theorem to recognize is
`strongMonotone_gradient_of_firstOrderStrongConvex`: under the first order strong convexity
hypothesis, the gradient is strongly monotone. This is the bridge that lets a training theorem move
from "the loss has these analytic assumptions" to "the update map is contractive."

# How This Connects To Verified Training

Autograd correctness can tell us that the gradient path computes the intended derivative. Runtime
approximation can tell us how close a rounded update is to the ideal update. Optimization theory is
where we state what that update means as an algorithm.

For example, an end-to-end training theorem can be read as a composition of three facts:

- an autograd theorem saying the gradient is the adjoint derivative of the loss;
- a runtime approximation theorem saying the executable update is close to the ideal update;
- an optimization theorem saying the ideal update contracts under smoothness and strong convexity
  hypotheses.

Those hypotheses matter. The optimization layer avoids turning a loss curve into a convergence
claim: convexity, smoothness, strong monotonicity, and step size conditions remain visible in the
theorem statement.

# What We Claim

TorchLean formalizes selected first order optimization facts and their assumptions, so future
training theorems can cite named Lean objects rather than prose folklore. It does not prove that
every training run converges. The classical lineage is the convex optimization tradition:
smoothness, strong convexity, gradient descent, and contraction arguments. The Lean declarations let us use
that tradition without hiding its assumptions behind a training loop.
