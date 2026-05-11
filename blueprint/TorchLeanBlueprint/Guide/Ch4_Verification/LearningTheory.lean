import VersoManual
import VersoBlueprint

open Verso.Genre Manual

#doc (Manual) "Learning Theory" =>
%%%
tag := "learning-theory"
%%%

Learning theory belongs in TorchLean because many ML claims become mathematical before they become
experimental. A training script can compute a loss curve, run an attack, add a private optimizer, or
fit a ridge regression model. The guarantee, however, usually lives in a predicate: privacy,
robustness, stability, convergence, or a finite precision bridge from an ideal theorem to executable
arithmetic.

This page is not about checking one certificate. It is about naming the predicates that appear in
learning theory papers and making them usable inside the same codebase as models, runtimes, and
verification artifacts.

We formalized this layer so those predicates have names inside Lean. Runtime diagnostics still
matter, but they are not silently upgraded into theorems. The recurring pattern is:

- start with a mathematical predicate such as privacy, robustness, or stability;
- compute a runtime diagnostic or artifact when that is useful evidence;
- add a bridge theorem when the artifact is meant to support a formal claim.

This is the same discipline used elsewhere in TorchLean. Specifications say what the claim means,
runtime code computes evidence or artifacts, and proofs connect checked hypotheses to the theorem
being cited.

# Core Definitions And Checked Claims

The learning theory material is organized around five concrete objects:

- *Randomized mechanism*: a map from inputs to probability measures over outputs; Lean states
  `(ε, δ)` privacy, pure privacy, monotonicity in `δ`, and post processing.
- *Robustness predicate*: a classifier is stable on a perturbation ball; Lean names tensor norms,
  local balls, Lipschitz predicates, and certified robustness.
- *Algorithmic stability*: replacing one training example changes the learned loss only slightly;
  Lean gives typed datasets, `replaceAt`, `removeAt`, learning maps, and loss change bounds.
- *Dynamical stability*: trajectories stay bounded or converge under stated hypotheses; Lean names
  Lyapunov, ISS, BIBO, incremental, practical, and finite time predicates.
- *Ridge regression case study*: a one dimensional strongly regularized ERM theorem, with a real
  stability theorem plus an `IEEE32Exec` execution bridge.

The API links in the sections below point to the exact declarations, but the guide should be read
from left to right: first the definition, then the theorem shape, then the runtime boundary.

# Differential Privacy

The core privacy definitions live in
[NN.MLTheory.LearningTheory.DifferentialPrivacy.Core API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/LearningTheory/DifferentialPrivacy/Core.lean).
The central abstraction is a randomized mechanism: a function from inputs of type `alpha` to
probability measures over outputs of type `beta`.

An adjacency relation `Adj : α → α → Prop` says which inputs are neighboring datasets or
neighboring records. The definition of `(ε, δ)`-DP is the standard event inequality: for every
adjacent pair and measurable event `S`, the probability of `M a` landing in `S` is at most
`exp ε` times the corresponding probability for `M a'`, plus `δ`.

The definition returns a `ProbabilityMeasure`. This keeps it general enough for discrete mechanisms,
continuous mechanisms, and randomized training procedures.
The mechanism does not have to be a particular optimizer or sampler; the definition says what any
privacy proof must establish.

In symbols, the definition has the usual event form:

$$`\forall D\sim D',\;\forall S,\qquad
\Pr[M(D)\in S]\le e^\varepsilon \Pr[M(D')\in S]+\delta.`

The file also proves two closure facts we expect to reuse in larger developments:

- `differentialPrivacy_mono_delta`: a mechanism that is private for `δ₁` is also private for any looser
  `δ₂ ≥ δ₁`.
- `differentialPrivacy_postprocess`: measurable post processing preserves DP.

That second theorem is the practical bridge to mainstream stacks. In ordinary ML code, we often
train a private model and then pass it through exporters, evaluators, dashboards, or downstream
selection logic. The DP theorem says the post processing step does not need to inspect the private
input again. TorchLean states that as a Lean theorem rather than relying on a comment.

The theorem shape is:

$$`M\ \text{is}\ (\varepsilon,\delta)\text{-DP}
\quad\Longrightarrow\quad
f\circ M\ \text{is}\ (\varepsilon,\delta)\text{-DP}.`

# Robustness: Spec Versus Runtime

Robustness is deliberately split into two files:

- [Robustness.Spec](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/LearningTheory/Robustness/Spec.lean) defines the mathematical
  predicates.
- [Robustness.Runtime](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/LearningTheory/Robustness/Runtime.lean) defines executable
  `Float` diagnostics.

The spec side works over TorchLean tensors without committing to one runtime scalar. It defines:

- `tensorLinfNorm` and `tensorL2Norm`,
- `tensorDistance`,
- closed tensor balls,
- global and local Lipschitz continuity,
- adversarial robustness at a point,
- certified robustness for classifiers,
- uniform robustness over a finite dataset,
- contraction mappings,
- local sensitivity ratios.

This is the vocabulary used by proof developments. For example, a certified robustness claim should
say that the classifier is constant on an `ε` ball, not merely that a search attack failed to find an
adversarial point.

A typical local robustness predicate has the form:

$$`\operatorname{Robust}(f,x,y,\varepsilon)
\Longleftrightarrow
\forall x',\;\|x'-x\|_\infty\le \varepsilon
\Rightarrow
\operatorname*{argmax} f(x')=y.`

The runtime layer specializes norms and distances to `Float` and provides empirical helpers such as
Lipschitz ratios from finite samples and deterministic perturbation sampling. We built that layer because
engineers need fast diagnostics, but its documentation is explicit: a sampled maximum is not a
global certificate. It is evidence, a debugging aid, or a counterexample search tool.

That distinction is one of the main differences from mainstream ML evaluation scripts. A typical
robustness notebook might compute "max observed ratio" and report it as if it were a property of
the model. TorchLean names it as an empirical runtime quantity unless a separate proof connects it
to a certified bound.

# Algorithmic Stability

The [algorithmic stability API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/LearningTheory/Stability/Core.lean) uses one
central representation choice: a dataset of size `n` is a `Spec.Vec n Z`, so its shape is part of
the type, not an untyped list whose length must be remembered separately. That lets stability definitions quantify over replace one and remove one
perturbations while keeping the sample size in the type.

The core file defines:

- coordinate access for datasets,
- `replaceAt` and `removeAt`,
- deterministic learning maps `Dataset n Z → H`,
- losses over the reals,
- empirical error,
- true population error under a probability measure,
- standard stability predicates over algorithms and losses.

In mainstream ML code, "replace one example and retrain" is usually an experiment. In TorchLean, it
is also a formal operation with a type. A theorem about replace one stability can refer to
`replaceAt S i z'` directly and inherit the fact that the modified object is still a dataset of the
same size.

The theorem shape is the standard uniform stability inequality:

$$`\forall S,S^{(i)},z,\qquad
|\ell(A(S),z)-\ell(A(S^{(i)}),z)|\le \beta.`

# Dynamical Stability

The stability entrypoint also imports
[NN.MLTheory.LearningTheory.Stability.Dynamics API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/LearningTheory/Stability/Dynamics.lean).
This is for recurrence systems, such as `x_{t+1} = f x_t`, and for systems driven by inputs, where
the next state depends on an input sequence. The spec file
[Dynamics.Spec](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/LearningTheory/Stability/Dynamics/Spec.lean) names predicates
such as Lyapunov stability, asymptotic stability, exponential stability, input to state stability,
BIBO stability, incremental stability, practical stability, finite time stability, and data/model
stability. The runtime file
[Dynamics.Runtime](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/LearningTheory/Stability/Dynamics/Runtime.lean) provides
`Float` diagnostics for concrete systems.

This matters for neural networks because not every learning theory question is a static supervised
learning theorem. Recurrent models, samplers, controllers, RL policies interacting with
state, and learned dynamical systems all need language for trajectories. Again, the runtime side is
empirical unless a theorem connects the observed diagnostic to the spec predicate.

# Ridge Regression As A Worked Theorem

The most concrete learning theory development is
[NN.MLTheory.LearningTheory.Stability.RidgeRegression1D.Real API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/LearningTheory/Stability/RidgeRegression1D/Real.lean).
It proves a replace one uniform stability bound for one dimensional ridge regression with squared
loss under bounded inputs.

The theorem follows the classical strongly convex ERM argument, but the file does it in a deliberately
small setting so the proof is inspectable:

1. Each example is a bounded pair `(x, y)` with `|x| <= X` and `|y| <= Y`.
2. The closed form fit is
   $`\hat w(S) = \frac{\sum_i x_i y_i}{\sum_i x_i^2 + \lambda N}`.
3. Replacing one example changes the numerator and denominator by controlled finite sums.
4. The proof bounds the difference between the two fitted weights.
5. A difference of squares argument converts the weight change bound into a loss change bound.

The final stability statement has the same form as the general predicate, now with an explicit
bound depending on the sample size, regularization, and data bounds:

$$`|\ell(\hat w(S),z)-\ell(\hat w(S'),z)|
\le
\beta(N,\lambda,X,Y).`

The worked theorem is deliberately small enough to inspect. The estimator is not a foreign function.
The dataset is the same `Dataset` representation from the stability core. The boundedness
assumptions are carried by types and hypotheses. The final statement is a Lean theorem, not a prose
claim next to a Python implementation.

# Runtime And Spec Splits

The learning theory tree intentionally repeats a pattern: a spec predicate or theorem is kept
separate from the executable runtime diagnostic, and an optional float32 or artifact bridge connects
them only when the hypotheses have been stated.

For robustness, the split is `Robustness.Spec` versus `Robustness.Runtime`.

For dynamical stability, the split is `Stability.Dynamics.Spec` versus
`Stability.Dynamics.Runtime`.

For ridge regression, the ideal theorem is in `RidgeRegression1D.Real`, while the executable
float32 development is in
[RidgeRegression1D/IEEE32Exec](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/LearningTheory/Stability/RidgeRegression1D/IEEE32Exec.lean).
The IEEE32Exec side implements the algorithm with TorchLean's executable binary32 model and states
the bridge to a proof level round after each primitive semantics under explicit finiteness
conditions.

That is the same numerics philosophy used elsewhere in TorchLean. A theorem over the reals is not
automatically a native float theorem. We first prove the clean mathematical result, then separately
state what finite precision execution means and which hypotheses are needed to connect it.

# A Concrete Comparison To Mainstream Stacks

Here is the practical difference we are aiming for:

- If a script says an optimizer is differentially private because it used a DP library, the
  TorchLean guidepost is to define a mechanism and prove the DP event inequality or import a
  checked theorem.
- If no attack found an adversarial example, that is runtime evidence unless it implies
  `isCertifiedRobust`.
- If a model seems stable when retrained, the formal guidepost is a replace one stability predicate
  over `Dataset n Z`.
- If a dynamical system stayed bounded in simulation, the formal guidepost is a BIBO, ISS,
  Lyapunov, or related stability predicate.
- If the theorem is over the reals but the code uses float32, the guidepost is an
  `IEEE32Exec`/FP32 bridge with explicit finite path hypotheses.

This does not make TorchLean automatically prove every learning theory theorem. It makes the boundary
between theorem, checker, diagnostic, and assumption harder to blur.

# Suggested Path Through The Theory

For readers coming in fresh, we recommend this first pass:

1. Read the [differential privacy core API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/LearningTheory/DifferentialPrivacy/Core.lean)
   for the smallest complete example of a reusable measure theoretic definition plus closure
   lemmas.
2. Read the [robustness spec API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/LearningTheory/Robustness/Spec.lean) beside
   the [robustness runtime API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/LearningTheory/Robustness/Runtime.lean) to see the
   spec/runtime split.
3. Read the [stability core API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/LearningTheory/Stability/Core.lean) for datasets,
   replace one perturbations, learning maps, and loss/error definitions.
4. Read the [ridge regression real theorem API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/LearningTheory/Stability/RidgeRegression1D/Real.lean)
   for a complete theorem development with explicit constants.
5. Read the [ridge regression IEEE32Exec API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/LearningTheory/Stability/RidgeRegression1D/IEEE32Exec.lean)
   when you want the finite precision bridge.

The common voice across this layer is intentional: we built small, named definitions first, then
attached runtime checks and proof obligations to them. That is how TorchLean lets ML engineering
and formal learning theory live in the same codebase without treating them as the same activity.
