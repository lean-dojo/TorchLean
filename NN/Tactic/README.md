# Proof tactics

Import `NN.Tactic` for the collection, or the individual module you need. The full `NN` import
also includes them. `NN.API` does not import the full tactic collection.

## Where metaprogramming belongs

Not every syntax extension is a proof tactic:

| Form | What it does | Implementation |
| --- | --- | --- |
| `autograd`, `converges`, `verify`, `einops`, `except_cases` | Construct or simplify proofs. | `NN/Tactic/` |
| `#compare` | Run differential tests on supplied inputs; no proof is produced. | `NN/Testing/` |
| `einsum`, `rearrange`, tensor literals | Elaborate expressions into shaped tensor programs. | `NN/Tensor/Internal/Elab/` |
| `nn.Sequential!`, `nn.compose!` | Expand model-building syntax into ordinary composition. | `NN/API/Macros.lean` |
| `#tensor_view` and other widget commands | Display data in the editor. | `NN/Widgets/` |

`einops?` is an inspection tactic, kept alongside `einops` in `NN/Tactic/Einops/Report.lean`.
It explains a tensor transformation without closing the goal. `einsum` itself is a term elaborator,
not a proof tactic: it checks the pattern and shapes and emits the tensor computation.

Reusable proof tactics belong here. File-local helpers may stay beside the lemmas they use;
moving those helpers would split a proof across files without making them more reusable.
The external CROWN producer/inspection workflow stays with its Python integration under
`NN/MLTheory/CROWN/Tactics/`; it loads reports rather than proving their claims.

## Derivatives

`autograd` proves a derivative formula using mathlib's rules. For example:

```lean
import NN.Tactic.Autograd

example (x : ℝ) : HasDerivAt (fun y : ℝ => Real.exp (y * y))
    (2 * x * Real.exp (x * x)) x := by
  autograd
```

`HasDerivAt f d x` means that `f` has derivative `d` at `x`. The tactic first composes the
derivative rules, then checks that their result equals your formula. Arithmetic rules retain
mathlib's scalar generality; supported complex functions use mathlib's complex derivatives.
A complex derivative is not the same thing as the gradient of a real loss on complex parameters.

For a new named function, prove its rule once and mark that theorem `@[autograd]`. Registered
rules can have hypotheses: for logarithms and reciprocals, for instance, the input must be
nonzero. The tactic fails if it cannot discharge those conditions. It does not invent a derivative
at a ReLU kink or turn a runtime implementation into a theorem.

Use `autograd?` to see the proof script it found. If a function is unsupported, unfold its
definition or supply a proved rule. Ordinary `apply`, `simp`, and mathlib lemmas remain available;
there is no separate expression language to learn.

For rewriting *inside* an expression, use `@[autograd simp]`. For example, a function built from
existing operations can expose its defining equation to `autograd` without adding it to ordinary
`simp`:

```lean
def weightedExp {α : Type} [Mul α] [MathFunctions α] (x : α) : α :=
  x * MathFunctions.exp x

@[autograd simp] theorem weightedExp_eq {α : Type} [Mul α] [MathFunctions α] (x : α) :
    weightedExp x = x * MathFunctions.exp x := rfl
```

This rule works inside higher-order scalar expressions, tensor maps, and reductions. Existing
domain conditions still have to be proved; unfolding a reciprocal does not remove its nonzero
condition. `@[local autograd simp]` limits a registration to its section. The underlying named
simp set is `autograd_simps`, so `simp only [autograd_simps]` is available for inspecting rewrites.

A new primitive whose implementation should stay hidden can instead register its proved jet
identities with `@[autograd simp]` and its smoothness theorem with mathlib's `@[fun_prop]`.
As with the built-in operations, direct scalar and composed-input forms may both be needed for
simplifier matching. Orient jet rules from the mathematical jet to the runtime expression, and
keep domain conditions in the hypotheses. Registering a first-derivative theorem alone does not
certify an arbitrary higher-order implementation.

The same tactic can prove a tensor node's derivative certificate:

```lean
open Spec Proofs Proofs.Autograd

noncomputable example {Γ : List Shape} {s : Shape} (input : Idx Γ s) :
    NodeFDerivCorrect
      (TapeNodes.elemwise input (fun x => x * x) (fun x => 2 * x)) := by
  autograd
```

Here `Γ` describes the saved tensors and `input` selects one of them. `s` is any tensor shape.
`NodeFDerivCorrect` connects the node's forward function and implemented derivative to mathlib's
Fréchet derivative. The tactic uses the existing elementwise lifting theorem rather than proving
each coordinate separately. It also assembles registered node certificates for explicit graphs.

The [autograd deep dive](../Examples/DeepDives/AutogradTransforms.lean) composes this operation
twice and proves the checked reverse pass returns the adjoint derivative. These are exact-real,
first-order graph results; they do not identify floating-point execution with real arithmetic.

## Arbitrary derivative order

Nested dual numbers attach a separate infinitesimal to each direction. The coefficient containing
all of them is the mixed derivative. `NN.Proofs.Autograd.Dual` proves this against mathlib's
`iteratedFDeriv` and connects addition, multiplication, negation, constants, linear input
coordinates, subtraction, `exp`, `sin`, `cos`, `tanh`, `sinh`, and `cosh` to the existing runtime
operations at every finite depth.

Division, logarithms, and square roots have local rules in `NN.Proofs.Autograd.Dual.Domain`.
They require
`ContDiffAt` (smoothness near the point where you differentiate), not smoothness everywhere.
The denominator, logarithm argument, or square-root argument must be nonzero at that point.
The arithmetic and activation rules also accept local smoothness, so these operations can be composed.
The real-logarithm theorem follows mathlib's convention on negative inputs; it does not claim
that a native floating-point logarithm returns a real value for a negative argument.

```lean
open Runtime.Autograd.Model

example (n : Nat) (directions : Fin n → ℝ) (x : ℝ) (hx : x ≠ 0) :
    Dual.Nested.tangent (MathFunctions.log (Dual.Nested.seed directions x)) =
      iteratedFDeriv ℝ n Real.log x directions := by
  autograd
```

The condition `hx` is used by the proof, not silently assumed. Without it, `autograd` refuses
this arbitrary-order logarithm claim. The same rule works inside expressions such as
`exp(log x)` and `log(a x) / b x`, with the corresponding nonzero hypotheses.

Square roots compose in the same way. For a smoothed absolute value, the runtime expression
`sqrt(z * z + ε)` agrees with every order of mathlib's `Real.sqrt (x * x + ε)` wherever
`x * x + ε ≠ 0`. A positive `ε` ensures this condition. The exact-real rule also covers negative
square-root arguments: mathlib and the exact-real runtime both return zero there, so all positive-order
derivatives vanish locally. This is not the behavior of a native floating-point square root on a
negative input, and no smoothness at zero is assumed. Tensor maps and reductions reuse the same rule.

For instance, this proves all orders of `exp(x*x)` at once:

```lean
open Runtime.Autograd.Model

example (n : Nat) (directions : Fin n → ℝ) (x : ℝ) :
    Dual.Nested.tangent (MathFunctions.exp
      (Dual.Nested.seed directions x * Dual.Nested.seed directions x)) =
      iteratedFDeriv ℝ n (fun y : ℝ => Real.exp (y * y)) x directions := by
  autograd
```

`n` is a variable, not a tested upper limit. The operation theorems require only `n` derivatives,
and their input space can be any real normed space. Directions may repeat and need not be
coordinate vectors. The rules apply to ordinary Lean functions; there is no second expression
language or polynomial-only evaluator.

Smooth activations compose in the same way. For example, replacing the expression above with
`MathFunctions.tanh (MathFunctions.sinh (z * z) - MathFunctions.cosh z)`, where `z` is the seeded
input, uses the corresponding real expression in `iteratedFDeriv`. The `tanh` rule reuses its
first-derivative proof and proves that `(1 - tanh(x) * tanh(x)) * dx` remains correct under every
level of dual nesting. It applies to tensor coordinates and graph pullbacks as well as scalars;
it is not restricted to a particular network or PDE.

The same rules lift to tensors of any shape:

```lean
open TorchLean

example {shape : Shape} (n : Nat) (directions : Fin n → Tensor ℝ shape)
    (x : Tensor ℝ shape) :
    Dual.Nested.tangentTensor
      (Tensor.map (fun y => MathFunctions.exp (y * y)) (Dual.Nested.seedTensor directions x)) =
      iteratedFDeriv ℝ n (fun y => Tensor.map (fun z => Real.exp (z * z)) y) x directions := by
  autograd
```

`Dual.Nested α n` now lives in the runtime and accepts any scalar context. The model's
`derivative` and `derivativeVjp` use these same tensor seeds and coefficient extraction. The
proof above is over exact reals: continuous-linear coordinate maps connect the result to the
Fréchet derivative in the tensor's Euclidean norm. Fixed coordinate reindexing is supported too,
including repeated reads, so input and output shapes need not agree.

Sums, dot products, and matrix multiplication use the same higher-order rules. For example,
the squared norm reduces to a dot product, with both operands depending on the input:

```lean
example {shape : Shape} (n : Nat) (directions : Fin n → Tensor ℝ shape)
    (x : Tensor ℝ shape) :
    Dual.Nested.tangent (Tensor.dotSpec (Dual.Nested.seedTensor directions x)
      (Dual.Nested.seedTensor directions x)) =
      iteratedFDeriv ℝ n (fun y => Tensor.dotSpec y y) x directions := by
  autograd
```

For a custom reduction, use an ordinary `List.foldl`. The sum and product rules preserve the
order you supply, allow repeated indices and empty lists, and differentiate the initial
accumulator as well as the entries. Products need no assumption that their factors are nonzero.
`Dual.jet_foldl` extends this to other smooth update rules with a proved jet law. The traversal
must be fixed while differentiating; these results do not justify an input-dependent sorting or
selection step, or equate different floating-point reduction orders.

Local domain conditions also work for tensor-valued outputs and their reductions. For example,
the sum of elementwise logarithms needs each input entry to be nonzero at the evaluation point:

```lean
example {shape : Spec.Shape} (n : Nat) (directions : Fin n → Tensor ℝ shape)
    (x : Tensor ℝ shape) (hx : ∀ i, x i ≠ 0) :
    Dual.Nested.tangent
      (Tensor.sumSpec (Tensor.map MathFunctions.log (Dual.Nested.seedTensor directions x))) =
      iteratedFDeriv ℝ n (fun y => Tensor.sumSpec (Tensor.map Real.log y)) x directions := by
  autograd
```

This proves the whole tensor function's iterated derivative, not just separate derivatives of
its entries. The same local rules cover binary elementwise maps, dot products, matrix products,
and fixed coordinate selections. A selection only needs domain conditions for the coordinates
it reads. For `List.foldl`, membership in the supplied order is available while proving the
update, so unvisited entries need no nonzero assumption.

The tactic carries proved tensor equalities through surrounding calculations using mathlib's
congruence rules. Sums and dot products can therefore share the same scalar operation proofs.
These examples certify the tensor arithmetic directly; recorded-graph execution still requires
the graph certificates described below.

`NN.Proofs.Autograd.Runtime.Link.HigherOrder` extends this result to recorded graphs. Each node
must preserve the input derivatives; `autograd` can discharge this obligation for supported
elementwise operations. The graph theorem propagates the result through all saved intermediates
and identifies any selected output with `iteratedFDeriv`. It uses the existing executable
`GraphData`, not a second graph representation. Fixed model state can be combined with a seeded
input tensor.

The deep dive also records `square` followed by `exp` with the runtime's graph builder. It proves
that recording succeeds and that executing the resulting nested-dual graph gives the iterated
derivative of its real execution, for every shape and finite order. This example checks the
recorder as well as the tensor arithmetic.

Reverse accumulation is covered by `Runtime.Link.HigherOrderReverse`. It tracks saved
activations, sparse input gradients, and contributions added at shared inputs. The output seed
can itself depend on the input. `autograd` assembles these laws for addition, multiplication,
and exponential pullbacks from the same scalar jet rules.

There are two distinct obligations: a pullback must be the correct first derivative, and its
nested-dual implementation must preserve higher derivatives. `Runtime.Link.HigherOrderFDeriv`
combines both certificates and proves that extracting the reverse result gives mathlib's
iterated derivative of the adjoint Fréchet derivative. A backward routine that is equally wrong
over reals and dual numbers cannot meet that first-order obligation merely by preserving jets.
The same module provides `TypedGraphWithData.tangent_vjp` for the runtime's `vjpWithSeed` method;
it checks that the real and nested runs record matching node shapes and select the same output.

`TypedGraphWithData.tangent_vjpChecked_adjoint_fderiv` extends the result to successful checked
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

Here `seed` is the output cotangent: it specifies which output combination to differentiate.
The directions and seed stay fixed while the input varies. A varying seed would add terms, so
the tactic rejects that substitution. The derivative-interchange theorem without adjoints also
works over complex normed spaces. The adjoint theorem is stated over reals, including for a real
loss on a complex space viewed as a real space; it does not assert a complex-analytic loss.

`TypedGraphWithData.tangent_vjpChecked_iteratedFDeriv` connects this identity to a successful
checked nested reverse execution. Its directions can involve any context entries, including
parameters, and its conclusion returns the selected native tensor gradient. The first-order
certificate, higher-order jet laws, smoothness, and successful execution are all explicit.

For the public forward transform, `TorchLean.autograd.model.derivative_eq` in
`NN.Proofs.Autograd.Model` connects the actual IO call to `iteratedFDeriv`, given successful
lowering and the graph's jet certificate. It accounts for fixed state and the whole direction
list.

`TorchLean.autograd.model.derivativeVjp_eq` in `Model.Reverse` covers the public reverse transform
as well. It proves the complete IO return value, including validation, graph execution, nested
coefficient extraction, and the split into state and input gradients. The directions have zero
parameter components, but the final adjoint differentiates with respect to parameters too. This
is the distinction that lets a coordinate derivative contribute to a PINN training gradient.
The seed correspondence is proved for arbitrary state and input shapes; it is not another
assumption the caller has to supply. `autograd` can apply the theorem when the lowering result,
checked execution, smoothness, and graph certificates are available.

Automatic certification of arbitrary model recording and general batched contractions remain
to be connected. Other runtime operations, such as variable-exponent powers, still need
higher-order rules; floating-point rounding remains separate.

The deep dive's `squareModel_derivative` proves a complete public `model.derivative` call for
every shape and order, without an assumed lowering result. It checks model validation and graph
recording before using `autograd` for the tensor arithmetic. The recording proof keeps the
scalar backend abstract; the derivative statement specializes to exact reals.

`NN.Proofs.Autograd.Model.Composition` provides execution equations for ordinary sequential
models. `Seq.forwardState_comp` preserves state layout and the order of forward calls and buffer
updates in both training and evaluation mode, for any lawful execution monad. It does not assume
that numerical operations are associative or that effects commute. These equations let model
proofs reason about layers before specializing to the graph recorder or eager backend.

## Verification

`verify` looks for a soundness theorem for the requested property and proves its remaining
conditions. For a CROWN query:

```lean
import NN.Verification.Cert.CROWNQuery

example {n m : Nat} (q : NN.Verification.CROWNQuery.Query n m)
    (checked : q.check = true) : q.Safe := by
  verify
```

For a small concrete query, `verify` can establish the check by kernel reduction as well. The
conclusion quantifies over all real inputs in the query's box; it is not a finite sample test.
The same tactic consumes checked Muon backend evidence to prove an update certificate. Such a
certificate states the direction and update equations, not convergence or improved loss.

Register another checker's proved soundness theorem with `@[verify]`. `verify?` shows the rules
and checks used. The tactic runs no external solver, trusts no success flag without a soundness
theorem, and fails if an obligation is unresolved. It does not optimize or mutate the program.

### Lowering a program

Import `NN.Tactic.Verify.Lowering` to use `verify` for graph lowering or a checked einsum plan.
We keep this import separate because the end-to-end graph proof is a substantial dependency.
For a graph, the tactic needs evidence that lowering succeeded and that the graph has no raw
logarithm nodes. That restriction matters: the source evaluator rejects nonpositive raw-log
inputs, whereas the executable graph totalizes them.

```lean
import NN.Tactic.Verify.Lowering

open TorchLean Runtime.Autograd.IRExec

example {α : Type} [Storage α] [Context α]
    (graph : NN.IR.Graph) (payload : NN.IR.Payload α) (exec : ForwardGraph α)
    (hNoRawLog : NoRawLog graph)
    (hLowered : lowerToForwardGraph graph payload = .ok exec)
    (input : Tensor α exec.inShape) :
    NN.IR.Graph.denoteAll (g := graph) (payload := payload)
        (input := Spec.SomeTensor.mk exec.inShape input) =
      .ok (ForwardGraph.denoteAll (e := exec) input) := by
  verify
```

This is an equality of Lean semantics for every input, not a comparison of a few runs. For
einsum, the registered theorem relates the checked plan to an independent tensor denotation.
It does not say that every input string expresses the operation its author intended, or that
native code and GPU kernels have been verified.

## Convergence

`converges` uses proved bounds for contractive fixed-point iterations and gradient descent.
It can prove a limit or a supported finite-iteration error bound. For example:

```lean
import NN.Tactic.Converges

open Filter Optim.GD
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
bound control how a step changes the distance to the root. The step-size condition makes that
distance shrink. The tactic applies the existing geometric error bound, then the theorem
`Optim.GD.tendsto_iterate_of_q_lt_one` turns the bound into convergence. The root is supplied,
so this result needs no completeness assumption or claim that a minimizer exists.

For a contractive map on a complete metric space, mathlib's fixed-point theorem supplies the
limit instead. The tactic also supports its a priori and a posteriori error bounds. Use
`converges?` to get the proof script, and `@[converges]` to register another proved rule.
Missing regularity or step-size hypotheses make it fail. It does not infer strong convexity
from a neural-network definition or prove convergence of arbitrary SGD, Adam, or training runs.

## Comparing implementations

For executable code, an independent reference is often a useful first check:

```lean
import NN.Testing.Command

#compare (fun n : Nat => n + n) with (fun n => 2 * n) on #[0, 1, 7]
-- Compared 3 cases (test only; not a proof).
```

The command reports the first failing input and both outputs, and rejects an empty corpus.
`using relation` selects a comparison instead of Boolean equality. For floating-point work,
choose deliberately between numerical equality, bit identity, and a tolerance. NaNs do not
silently count as equal, unless your chosen comparison says they do. In a `module` file, imported
runtime functions used by this editor command need `meta import`, just as they do for `#eval`.

For CPU/GPU calls or other effectful implementations, import `NN.Testing.Compare` and call
`NN.Testing.compareOn cases candidate reference agrees` in a native executable. Both functions
take the same input type, but their result types can differ: the comparison relates the two.
Supply reproducible inputs, and reset any shared state before each side.
Exceptions fail the test; two failing implementations do not count as agreement. The helper
uses no external solver or hidden random generator.

Use temporary scratch files for development sweeps and remove them after validation. Keep a
permanent runtime check only when it covers a specific gap in the proofs and existing tests.
For einsum, the existing tensor tests cover matrix products, empty dimensions, and floating-point
cancellation that detects reordered reductions.

Independent proof validation is a different task. A tool such as Lean Comparator can check a
software-correctness theorem against a separately fixed specification, just as it can check a
mathematical theorem. It cannot establish that the specification matches the intended software
merely by accepting a proof. `#compare` is not a replacement for that proof checker.

## Tensor identities

The existing `einops` and `einops?` tactics live under `NN/Tactic/Einops`. Import `NN.Tactic.Einops`
for both, or `NN.Tactic.Einops.Proof` for proof automation without the report renderer.
`NN.Tensor` still exposes them for tensor users. Their behavior is unchanged; only their source
location has moved. The tensor compiler, semantics, and lowering laws remain in the tensor library.

## Successful validation

`NN.Tactic.Except` provides `except_cases`. Given a successful computation, it extracts a
successful intermediate step and eliminates the error branch:

```lean
example {ε α β : Type} (step : Except ε α) (next : α → Except ε β) (result : β)
    (success : (step >>= next) = .ok result) : ∃ value, step = .ok value := by
  except_cases h : step using success with value =>
    exact ⟨value, rfl⟩
```

Unfold the enclosing computation in the success hypothesis first if needed. This tactic is used
in RL validation and checked autograd execution; it does not depend on a particular error type.
