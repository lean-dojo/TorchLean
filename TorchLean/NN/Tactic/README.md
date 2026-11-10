# Derivative and convergence tactics

`autograd` proves derivative formulas and derivative certificates, `op_deriv` proves that a
backward pass is the adjoint of the derivative, and `converges` proves limits and error bounds of
iterations. They build on results of the NN layer: `autograd` and `op_deriv` on the derivative
proofs in `TorchLean/NN`, and `converges` on the gradient-descent bounds in
`TorchLean/NN/Optimizer`. The tactics that need only the tensor library
live in [`TorchLean/Tensor/Tactic`](../../Tensor/Tactic/README.md).

- `Autograd.lean`: the `autograd` and `autograd?` tactics and their tensor-node and graph rules.
- `Autograd/Scalar.lean`: the `@[autograd]` and `@[autograd simp]` attributes and scalar rules.
- `Converges.lean`: the `converges` and `converges?` tactics and the `@[converges]` attribute.
- `Init.lean`: the Aesop rule sets `AutogradDeriv`, `Autograd`, `Converges` and `OpDeriv`, and the
  simp sets `autograd_simps` and `op_deriv_simps`.
- `OpDeriv.lean`: the `op_deriv` tactic, which proves that a backward pass is the adjoint of the
  derivative of its forward map.

## Derivatives

`autograd` proves a derivative formula using mathlib's rules. For example:

```lean
import TorchLean.NN.Tactic.Autograd

example (x : ℝ) : HasDerivAt (fun y : ℝ => Real.exp (y * y))
    (2 * x * Real.exp (x * x)) x := by
  autograd
```

`HasDerivAt f d x` says that `f` has derivative `d` at `x`. The tactic composes the derivative
rules, then checks that their result equals the derivative stated in the goal. Arithmetic rules keep
mathlib's scalar generality, and supported complex functions use mathlib's complex derivatives. A
complex derivative is not the gradient of a real loss on complex parameters.

For your own named function, prove its rule once and mark that theorem `@[autograd]`. Registered
rules can have hypotheses: for logarithms and reciprocals, for instance, the input must be
nonzero. The tactic fails if it cannot discharge those conditions. It does not invent a derivative
at a ReLU kink or turn a runtime implementation into a theorem.

`autograd?` shows the proof script it found. If a function is unsupported, unfold its definition
or supply a proved rule. Ordinary `apply`, `simp` and mathlib lemmas work alongside the tactic, and
there is no separate expression language to learn.

For rewriting *inside* an expression, use `@[autograd simp]`. For example, a function built from
existing operations can expose its defining equation to `autograd` without adding it to ordinary
`simp`:

```lean
open FloatLib.Numerics (MathFunctions)

def weightedExp {α : Type} [Mul α] [MathFunctions α] (x : α) : α :=
  x * MathFunctions.exp x

@[autograd simp] theorem weightedExp_eq {α : Type} [Mul α] [MathFunctions α] (x : α) :
    weightedExp x = x * MathFunctions.exp x := rfl
```

Such a rule works inside higher-order scalar expressions, tensor maps and reductions. Unfolding does
not discharge domain conditions: an unfolded reciprocal keeps its nonzero condition.
`@[local autograd simp]` limits a registration to its section. The registered rewrites form the simp
set `autograd_simps`, and `simp only [autograd_simps]` applies them alone for inspection.

A primitive whose implementation should stay hidden can register its proved jet identities with
`@[autograd simp]` and its smoothness theorem with mathlib's `@[fun_prop]`. As with the built-in
operations, simplifier matching may need both a plain scalar form and a composed-input form of
each rule. Orient jet rules from the mathematical jet to the runtime expression, and keep domain
conditions in the hypotheses. A first-derivative theorem alone does not certify an arbitrary
higher-order implementation.

The same tactic can prove a tensor node's derivative certificate:

```lean
open TorchLean TorchLean.Tape

noncomputable example {Γ : List Shape} {s : Shape} (input : Idx Γ s) :
    RealNode.FDerivCorrect
      (RealNode.elemwise input (fun x => x * x) (fun x => 2 * x)).toNodeData () := by
  autograd
```

Here `Γ` lists the shapes of the saved tensors, `input` selects one of them and `s` is any tensor
shape. The certificate is about the node's data `toNodeData`, and the trailing `()` is the node's
environment, of type `Unit` for library nodes.
`RealNode.FDerivCorrect` connects the node's forward function and implemented derivative to
mathlib's Fréchet derivative. The tactic covers every coordinate at once with the elementwise
lifting theorem, and it assembles registered node certificates into graph certificates. For a node
built by `Tape.LawfulNode.ofArm` from an operator arm, it applies the derivative certificate
registered for that operator with `op_cert`.

The [autograd deep dive](../../../TorchLeanExamples/DeepDives/AutogradTransforms.lean) composes this
operation twice and proves that the checked reverse pass returns the adjoint derivative. These are
first-order graph results over exact reals, and they do not identify floating-point execution with
real arithmetic.

## Backward passes

`op_deriv` proves that a backward pass is the adjoint of the derivative of its forward map. The
goal has the form of an operator certificate:

```lean
import TorchLean.NN.Tactic.OpDeriv

open TorchLean

example {s : Shape} (x δ : Tensor ℝ s) :
    DifferentiableAt ℝ (fun y : Tensor ℝ s => Tensor.map (fun a => Real.log (1 + Real.exp a)) y)
        x ∧
      δ * Tensor.map (fun a => Real.exp a / (1 + Real.exp a)) x =
        (fderiv ℝ (fun y : Tensor ℝ s => Tensor.map (fun a => Real.log (1 + Real.exp a)) y)
          x).adjoint δ := by
  op_deriv
```

The tactic works with `HasPullbackAt f D b x` from `TorchLean/NN/FDeriv/Pullback.lean`: `f` has
the derivative `D` at `x`, and `b` is its adjoint, `⟪D u, δ⟫ = ⟪u, b δ⟫`. Mathlib's `fun_prop`
builds `D` and `b` from the `@[fun_prop]` rules for composition, sums, products, quotients, the
components of an `L²` pair, entrywise maps and the certified operators. `op_deriv` then shows that
the stated backward pass is `b`. A continuous linear map gets its rule from `HasPullbackAt.clm` and
its transpose, and a continuous bilinear map from `HasPullbackAt.bilin` and its two transposes.
The scalar derivatives inside an entrywise map come from the Aesop rule set `OpDeriv`, which holds
mathlib's `HasDerivAt` lemmas and the derivatives of the activations and pointwise operators of
the library, and then from the rule set `AutogradDeriv` of `autograd`. Side conditions, such as
the nonzero argument of the logarithm above, come from the hypotheses, `ne_of_gt`, `ne_of_lt` and
`positivity`, which also sees that a sum of squares such as `Tensor.sum (x * x)` is nonnegative.
The condition `a ≠ 0 ∨ 1 ≤ p` of a real power `a ^ p` comes from either side, with `norm_num` for
`1 ≤ p`. The rules of the library's functions are in the modules of their operators, so
`op_deriv` uses the rule of `Tensor.mean` only where `TorchLean.NN.Op.Reduce` is imported.

`fun_prop` matches the rules at reducible and instance transparency, so the rule of another
function with the same head fails at once, and neither function is unfolded. A definition with no
rule is unfolded only when it is given in brackets: `op_deriv [e₁, e₂]` also rewrites with the
lemmas or unfolds the definitions `e₁` and `e₂` in the forward map as well as in the backward
pass, and a failing call names the definitions applied to the variable that have no rule. A `let`
introduced with `intro` is unfolded, also one whose value is an operator's table entry.
`@[op_deriv_simps]` registers a rewrite for every call, such as the real form of an operator's
table entry, and `@[aesop safe apply (rule_sets := [OpDeriv])]` a scalar derivative.

The tactic fails on a wrong backward pass, and on the square root, `relu` or `a ^ (1 / 2)`
without the hypothesis that no entry is zero. `TorchLeanTest/Verification/OpDeriv.lean` proves
with it 61 theorems about programs outside the operator table, among them the backward passes of
a dense layer with `Linear.backward`, of a two-layer network in its four parameters, of masked
attention with `matmulBackward` and `maskedSoftmaxBackward`, of cross-entropy through `logSoftmax`
and of an ordered weighted sum with `sortBackward`. Each proof takes at most 14,600 heartbeats of
the default 200,000, each rejection at most 5,400, and the file compiles in 17 s. Most derivative
certificates in `TorchLean/NN/Op` come from the same rules, through `op_deriv` or `fun_prop`.

## Arbitrary derivative order

Nested dual numbers attach a separate infinitesimal to each direction, and the coefficient
containing all of them is the mixed derivative. `TorchLean.NN.Dual.Jet` proves this
against mathlib's `iteratedFDeriv` and connects addition, multiplication, negation, constants,
linear input coordinates, subtraction, `exp`, `sin`, `cos`, sigmoid, `tanh`, `sinh` and `cosh` to
the runtime operations at every finite depth.

Logarithms, square roots and reciprocals have local rules in
`TorchLean.NN.Dual.Domain`, and the quotient rule `jet_div` is in
`TorchLean.NN.Dual.Jet`. They require only `ContDiffAt` (smoothness near the point
where you differentiate), and the denominator, logarithm argument or square-root argument must be
nonzero at that point. The arithmetic and activation rules also accept local smoothness, so these
operations compose. On negative inputs the real-logarithm theorem follows mathlib's convention,
while a native floating-point logarithm returns NaN there.

```lean
open TorchLean
open FloatLib.Numerics (MathFunctions)

example (n : Nat) (directions : Fin n → ℝ) (x : ℝ) (hx : x ≠ 0) :
    Dual.Nested.tangent (MathFunctions.log (Dual.Nested.seed directions x)) =
      iteratedFDeriv ℝ n Real.log x directions := by
  autograd
```

The proof uses the hypothesis `hx`: without it, `autograd` refuses this arbitrary-order logarithm
claim. The same rule works inside expressions such as `exp(log x)` and `log(a x) / b x`, given the
corresponding nonzero hypotheses.

Square roots compose in the same way. For a smoothed absolute value, the runtime expression
`sqrt(z * z + ε)` agrees with every order of mathlib's `Real.sqrt (x * x + ε)` wherever
`x * x + ε ≠ 0`, which a positive `ε` guarantees. The exact-real rule also covers negative
square-root arguments: mathlib and the exact-real runtime both return zero there, so all
positive-order derivatives vanish locally. A native floating-point square root returns NaN on a
negative input, and the rule assumes no smoothness at zero. Tensor maps and reductions reuse the
same rule.

For instance, this proves all orders of `exp(x*x)` at once:

```lean
open TorchLean
open FloatLib.Numerics (MathFunctions)

example (n : Nat) (directions : Fin n → ℝ) (x : ℝ) :
    Dual.Nested.tangent (MathFunctions.exp
      (Dual.Nested.seed directions x * Dual.Nested.seed directions x)) =
      iteratedFDeriv ℝ n (fun y : ℝ => Real.exp (y * y)) x directions := by
  autograd
```

The operation theorems need only `n` derivatives, and their input space can be any real normed
space. Directions may repeat and need not be coordinate vectors. The rules apply to ordinary Lean
functions, with no second expression language or polynomial-only evaluator.

Smooth activations compose in the same way. With
`MathFunctions.tanh (MathFunctions.sinh (z * z) - MathFunctions.cosh z)` in place of the expression
above, where `z` is the seeded input, the right side uses the corresponding real expression inside
`iteratedFDeriv`. The `tanh` rule reuses its first-derivative proof and proves that
`(1 - tanh(x) * tanh(x)) * dx` is correct at every level of dual nesting. It applies to tensor
coordinates and graph pullbacks as well as scalars, and it isn't tied to a particular network or
PDE.

The same rules lift to tensors of any shape:

```lean
open TorchLean
open FloatLib.Numerics (MathFunctions)

example {shape : Shape} (n : Nat) (directions : Fin n → Tensor ℝ shape)
    (x : Tensor ℝ shape) :
    Dual.Nested.tangentTensor
      (Tensor.map (fun y => MathFunctions.exp (y * y)) (Dual.Nested.seedTensor directions x)) =
      iteratedFDeriv ℝ n (fun y => Tensor.map (fun z => Real.exp (z * z)) y) x directions := by
  autograd
```

`Dual.Nested α n` is defined in `TorchLean/NN/Dual/Nested.lean` and works
over any scalar context. The model's `derivative` and `derivativeVJP` use the same tensor seeds
and coefficient extraction. The proof above is over exact reals: continuous-linear coordinate maps
connect the result to the Fréchet derivative in the tensor's Euclidean norm. Fixed coordinate
reindexing is supported too, including repeated reads, so input and output shapes need not agree.

Sums, dot products, and matrix multiplication use the same higher-order rules. For example,
the squared norm reduces to a dot product, with both operands depending on the input:

```lean
example {shape : Shape} (n : Nat) (directions : Fin n → Tensor ℝ shape)
    (x : Tensor ℝ shape) :
    Dual.Nested.tangent (Tensor.dot (Dual.Nested.seedTensor directions x)
      (Dual.Nested.seedTensor directions x)) =
      iteratedFDeriv ℝ n (fun y => Tensor.dot y y) x directions := by
  autograd
```

For a custom reduction, use an ordinary `List.foldl`. The sum and product rules keep the order you
supply, allow repeated indices and empty lists, and differentiate the initial accumulator as well
as the entries. Products need no assumption that their factors are nonzero. `Dual.jet_foldl`
extends this to other smooth update rules with a proved pointwise jet law. The traversal must be
fixed while differentiating. These results do not justify an input-dependent sorting or selection
step, and they do not equate different floating-point reduction orders.

Local domain conditions also work for tensor-valued outputs and their reductions. For example,
the sum of elementwise logarithms needs each input entry to be nonzero at the evaluation point:

```lean
example {shape : TorchLean.Shape} (n : Nat) (directions : Fin n → Tensor ℝ shape)
    (x : Tensor ℝ shape) (hx : ∀ i, x i ≠ 0) :
    Dual.Nested.tangent
      (Tensor.sum (Tensor.map MathFunctions.log (Dual.Nested.seedTensor directions x))) =
      iteratedFDeriv ℝ n (fun y => Tensor.sum (Tensor.map Real.log y)) x directions := by
  autograd
```

This example proves the iterated derivative of the whole tensor function, which is more than
separate derivatives of its entries. The same local rules cover binary elementwise maps, dot
products, matrix products and fixed coordinate selections. A selection needs domain conditions
only for the coordinates it reads. For `List.foldl`, membership in the supplied order is available
while proving the update, so unvisited entries need no nonzero assumption.

The tactic carries proved tensor equalities through surrounding calculations with mathlib's
congruence rules, so sums and dot products can share the same scalar operation proofs. These
examples certify the tensor arithmetic. Recorded-graph execution needs the graph certificates
described below.

`TorchLean.NN.Tape.Lowering.HigherOrder` extends the result to recorded graphs. Each
node must preserve the input derivatives, and `autograd` can discharge that obligation for
supported elementwise operations. The graph theorem propagates the result through all saved
intermediates and identifies any selected output with `iteratedFDeriv`. It works on the executable
`GraphData`, so there is no second graph representation. Fixed model state can be combined with a
seeded input tensor.

The deep dive also records `square` followed by `exp` with the runtime's graph builder. It proves
that recording succeeds and that executing the resulting nested-dual graph gives the iterated
derivative of its real execution, for every shape and finite order. So the example checks the
recorder as well as the tensor arithmetic.

`TorchLean.NN.Tape.Lowering.HigherOrderReverse` covers reverse accumulation. It tracks saved
activations, sparse input gradients and contributions added at shared inputs, and the output seed
can itself depend on the input. `autograd` assembles these laws for addition, multiplication and
exponential pullbacks from the same scalar jet rules.

There are two distinct obligations: a pullback must be the correct first derivative, and its
nested-dual implementation must preserve higher derivatives.
`TorchLean.NN.TypedGraph.WithData.HigherOrder` combines both certificates and proves that
extracting the reverse result gives mathlib's iterated derivative of the adjoint Fréchet
derivative. A backward routine that is equally wrong over reals
and dual numbers can preserve jets while failing the first-order obligation. The same module
proves `WithData.tangent_vjp` for the runtime's `vjp` method, assuming that the real and
nested runs record the same node shapes and select the same output.

`WithData.tangent_vjpChecked_eq_fderiv` extends the result to successful checked
execution, including tape lowering and typed gradient recovery. The bookkeeping proof preserves
addition order and requires no semiring laws from the scalar backend.

For training on derivatives, we also need to know that differentiating the reverse pass gives
the reverse pass of the derivative. `FDeriv.Interchange` proves this for every finite order under
`ContDiff ℝ (n + 1) f`. It uses mathlib's symmetry of second derivatives and induction, without
requiring analyticity. For real Hilbert spaces:

```lean
example {E F : Type*} [NormedAddCommGroup E] [NormedAddCommGroup F]
    [InnerProductSpace ℝ E] [InnerProductSpace ℝ F] [CompleteSpace E] [CompleteSpace F]
    {n : Nat} (f : E → F) (hf : ContDiff ℝ (n + 1) f)
    (x : E) (directions : Fin n → E) (seed : F) :
    iteratedFDeriv ℝ n (fun y => (fderiv ℝ f y).adjoint seed) x directions =
      (fderiv ℝ (fun y => iteratedFDeriv ℝ n f y directions) x).adjoint seed := by
  autograd
```

Here `seed` is the output cotangent, which picks the output combination to differentiate. The
directions and seed stay fixed while the input varies. A varying seed would add terms, so the
tactic rejects that substitution. The interchange theorem without adjoints also holds over complex
normed spaces. The adjoint theorem is stated over reals. It covers a real loss on a complex space
viewed as a real space, and the loss need not be complex-analytic.

`WithData.tangent_vjpChecked_iteratedFDeriv` connects this identity to a successful
checked nested reverse execution. Its directions can involve any context entries, including
parameters, and its conclusion is stated for the selected native tensor gradient. The first-order
certificate, the higher-order jet laws, smoothness and successful execution are all hypotheses.

For the public forward transform, `TorchLean.autograd.model.derivative_eq` in
`TorchLean.NN.Autograd.Model.Reverse` connects the actual IO call to `iteratedFDeriv`, given
successful lowering and the graph's jet certificate: the call throws the first failed node check
of `TypedGraph.lowerToTapeChecked` and otherwise returns the derivative. It accounts for fixed
state and the whole direction list.

`TorchLean.autograd.model.derivativeVJP_eq` in `Model.Reverse` covers the public reverse
transform. It proves the complete IO return value, including validation, graph execution, nested
coefficient extraction and the separation into state and input gradients. The directions have zero
parameter components, but the final adjoint differentiates with respect to the parameters too,
which is what lets a coordinate derivative contribute to a PINN training gradient. The seed
correspondence is proved for arbitrary state and input shapes, so the caller doesn't supply it as
an assumption. `autograd` can apply the theorem when the lowering result, checked execution,
smoothness and graph certificates are available.

Arbitrary model recording and the higher derivatives of general batched contractions are not
certified automatically. `op_deriv` proves the first-order pullbacks of these contractions with the
rule `HasPullbackAt.contract` of `TorchLean/NN/Op/Matmul.lean`. Other runtime operations, such as
variable-exponent powers, have no higher-order rules, and these results do not cover floating-point
rounding.

The deep dive's `squareModel_derivative` proves a complete public `model.derivative` call for
every shape and order, with no assumed lowering result. Its proof covers model validation and
graph recording before using `autograd` for the tensor arithmetic. The recording proof keeps the
scalar backend abstract, and the derivative statement specializes to exact reals.

`TorchLean.NN.Sequential.Composition` provides execution equations for ordinary
sequential models. `nn.Sequential.forwardState_comp` preserves state layout and the order of forward
calls and buffer updates in both training and evaluation mode, for any lawful execution monad. It
does not assume that numerical operations are associative or that effects commute. These equations
let model proofs reason about layers before specializing to the graph recorder or eager backend.

## Convergence

`converges` uses proved bounds for contractive fixed-point iterations and gradient descent.
It can prove a limit or a supported finite-iteration error bound. For example:

```lean
import TorchLean.NN.Tactic.Converges

open Filter TorchLean.GD
open scoped Topology

example {E : Type} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (η μ : ℝ) {L : NNReal} (g : E → E)
    (hmono : StrongMonotone μ g) (hlip : LipschitzWith L g)
    (x xStar : E) (hroot : g xStar = 0)
    (hμ : 0 ≤ μ) (hμL : μ ≤ (L : ℝ)) (hη : 0 < η)
    (hstep : η * (L : ℝ) ^ 2 < 2 * μ) :
    Tendsto (fun n => (step η g)^[n] x) atTop (𝓝 xStar) := by
  converges
```

Here `g` is an operator on any real inner product space. Strong monotonicity and a Lipschitz
bound control how a step changes the distance to the root, and the step-size condition makes that
distance shrink. The tactic applies the geometric error bound, then
`TorchLean.GD.tendsto_iterate_of_q_lt_one` turns the bound into convergence. Because the root is
supplied, the result needs no completeness assumption and no proof that a minimizer exists.

For a contractive map on a complete metric space, the limit comes from mathlib's fixed-point
theorem, and the tactic supports its a priori and a posteriori error bounds as well. Use
`converges?` to get the proof script, and `@[converges]` to register another proved rule.
Missing regularity or step-size hypotheses make the tactic fail. It does not infer strong
convexity from a neural-network definition or prove convergence of arbitrary SGD, Adam or
training runs.
