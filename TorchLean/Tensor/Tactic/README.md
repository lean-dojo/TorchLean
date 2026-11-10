# Proof tactics

This directory holds TorchLean's tensor proof tactics: `verify` for properties backed by a checker's
soundness theorem, `einops` for tensor identities, `tensor_shape` for the output-shape equation of a
general tensor operator and `except_cases` for successful `Except` computations. Import the module
that defines the tactic you use, such as `TorchLean.Tensor.Tactic.Verify` or
`TorchLean.Tensor.Tactic.TensorShape`. `Init.lean` declares the Aesop rule set `Verify` that
`verify` uses and `instantiateOuterLets`, the expression helper of the einops tactics.

`autograd` for derivatives and `converges` for limits and error bounds of iterations live in
[`TorchLean/NN/Tactic`](../../NN/Tactic/README.md).

## Where metaprogramming belongs

Not every syntax extension is a proof tactic:

| Form | What it does | Implementation |
| --- | --- | --- |
| `autograd`, `converges` | Construct or simplify proofs. | `TorchLean/NN/Tactic/` |
| `verify`, `einops`, `except_cases` | Construct or simplify proofs. | `TorchLean/Tensor/Tactic/` |
| `einsum`, `rearrange`, tensor literals | Elaborate into tensor programs. | `TorchLean/Tensor/` |
| `nn.Sequential!`, `nn.compose!` | Expand into ordinary composition. | `TorchLean/NN/Sequential/` |
| `#tensor_view` and other widget commands | Display data in the editor. | `TorchLean/NN/Widget/` |

`einops?` (`TorchLean/Tensor/Tactic/Einops/Report.lean`) is an inspection tactic: it explains a
tensor transformation without closing the goal. `einsum` is a term elaborator: it checks the pattern
and shapes and emits the tensor computation.

Reusable proof tactics belong here. File-local helpers can stay beside the lemmas they use, since
moving them would split a proof across files without making them more reusable.

## Verification

`verify` looks for a soundness theorem for the requested property and proves its remaining
conditions. For a CROWN query:

```lean
import TorchLean.Verify.Certificate.CROWNQuery

example {n m : Nat} (q : TorchLean.Certificate.CROWNQuery.Query n m)
    (checked : q.check = true) : q.Safe := by
  verify
```

For a small concrete query, `verify` can also establish the check by kernel reduction. The
conclusion quantifies over all real inputs in the query's box, so it is not a finite sample test.
The same tactic consumes checked Muon backend evidence to prove an update certificate. Such a
certificate states the direction and update equations, not convergence or improved loss.

Register another checker's proved soundness theorem with `@[verify]`. `verify?` shows the rules
and checks used. The tactic runs no external solver, trusts no success flag without a soundness
theorem, and fails if an obligation is unresolved. It does not optimize or mutate the program.

### Lowering a program

Import `TorchLean.Tensor.Tactic.Verify.Lowering` to use `verify` for a checked einsum plan. The
registered theorems relate the executable plan to an independent tensor denotation:

```lean
import TorchLean.Tensor.Tactic.Verify.Lowering

open TorchLean TorchLean.Einops

example {R : Type} [Storage R] [Add R] [Mul R] [OfNat R 0] [OfNat R 1]
    (checked : Einops.Einsum) (inputs : checked.InputTensors R) :
    Lowering.einsum checked inputs = Semantics.denoteEinsum checked inputs := by
  verify
```

The theorem proves that the two Lean semantics agree on every input. It does not show that an einsum
string expresses the operation its author intended, or verify native code and GPU kernels.

## Comparing implementations

For executable code, compare against an independent reference in a temporary `#eval` or native
executable. Decide up front between numerical equality, bit identity and a tolerance. Use
reproducible inputs and reset shared state before each implementation runs. Such checks produce
no proofs.

Use temporary scratch files for development sweeps and remove them after validation. Keep a
permanent runtime check only when it covers a gap that the proofs and the tests leave.

Independent proof validation is a different task. A tool such as Lean Comparator can check a
software-correctness theorem against a separately fixed specification, just as it can check a
mathematical theorem. It cannot establish that the specification matches the intended software
merely by accepting a proof. Executable comparisons do not replace that proof checker.

## Tensor identities

The `einops` and `einops?` tactics are in `TorchLean/Tensor/Tactic/Einops.lean` and
`TorchLean/Tensor/Tactic/Einops/`. Import `TorchLean.Tensor.Tactic.Einops.Report` for both, or
`TorchLean.Tensor.Tactic.Einops` for the proof automation without the report renderer. `einops`
works on the first unsolved goal, and its initial normalization may leave the goal unchanged or
close it. `einops?` inspects the current target. The tensor compiler, semantics and lowering laws
are in the tensor library.

`contract`, `cat`, `gather` and `permute` take their output shape as an argument, with a proof that
it equals the shape they compute. `tensor_shape` (`TorchLean/Tensor/Tactic/TensorShape.lean`) is the
default proof: `rfl` when both sides compute to the same shape, then
`simp only [tensor_shape_simps]`, then `decide`. A lemma tagged `@[tensor_shape_simps]` rewrites a
computed shape to its normal form.

## Successful validation

`TorchLean.Tensor.Tactic.Except` provides `except_cases`. Given a successful computation, it
extracts a successful intermediate step and eliminates the error branch:

```lean
example {ε α β : Type} (step : Except ε α) (next : α → Except ε β) (result : β)
    (success : (step >>= next) = .ok result) : ∃ value, step = .ok value := by
  except_cases h : step using success with value =>
    exact ⟨value, rfl⟩
```

If needed, unfold the enclosing computation in the success hypothesis first. RL validation and
checked autograd execution use `except_cases`, and it works with any error type.
