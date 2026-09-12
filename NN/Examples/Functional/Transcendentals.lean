/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.Proofs.Autograd.FDeriv.Elementwise

/-!
# Differentiate three scalar functions

Run `lake exe torchlean transcendentals` to compare autograd with hand-computed derivatives:

* `exp x` has derivative `exp x`;
* `3 * x + 1` has derivative `3`;
* `exp (-2 * x)` has derivative `-2 * exp (-2 * x)` by the chain rule.

All three are evaluated at `x = 0.5` using `Float` and absolute tolerance `1e-6`. Each check also
rejects a named wrong answer. In particular, the last one distinguishes a missing minus sign from
rounding error. Every check should print `PASS` or `PASS-NEG`; a mismatch exits with an error.
No dataset or GPU is needed.

`expProofSurface` and `expBackward_eq_adjoint_fderiv` give the separate real-arithmetic theorem.
The executable comparisons test the Float tape on these inputs; they do not prove its native
implementation or establish a derivative at every input. Read `expNegativeTwoFn`, then `gradAt`,
then `checkAll` for the application flow.
-/

@[expose] public section

namespace NN.Examples.Functional.Transcendentals

open TorchLean.Tensor
open TorchLean

/-! ## Proof objects and runtime checks -/

noncomputable section

/--
The theorem-backed real-valued exp op used by the proof layer.

This is the actual proof layer object: it packages the forward op, its JVP, a Fréchet-derivative
candidate, and the theorem that the JVP is the true derivative. The runtime checks below exercise
the executable Float tape; this declaration points to the corresponding real-valued theorem.
-/
def expProofSurface : Proofs.Autograd.OpSpecFDerivCorrect 1 1 :=
  Proofs.Autograd.OpSpecFDerivCorrect.exp (n := 1)

/--
For scalar exp over `ℝ`, the proved backward rule is the adjoint of the Fréchet derivative.

This is the theorem-level statement that the executable regression check is meant to complement.
-/
theorem expBackward_eq_adjoint_fderiv
    (x δ : Tensor ℝ [1]) :
    Proofs.Autograd.getScalarE (expProofSurface.correct.op.backward x δ) =
      Proofs.Autograd.vjp expProofSurface.forwardVec (Proofs.Autograd.getScalarE x)
        (Proofs.Autograd.getScalarE δ) :=
  Proofs.Autograd.OpSpecFDerivCorrect.backward_eq_adjoint_fderiv expProofSurface x δ

end

/-! ## Functions under test (written once; gradients come from autograd) -/

/-- $f(x)=e^x$. -/
def expFn : autograd.Function [] [] :=
  fun x => nn.functional.exp x

/-- $f(x)=e^{-2x}$; its derivative needs both the factor two and the minus sign. -/
def expNegativeTwoFn : autograd.Function [] [] :=
  fun x => do
    let u ← nn.functional.scale x (-2)
    nn.functional.exp u

/-- $f(x)=3x+1$ via the scalar-affine op. -/
def affineFn : autograd.Function [] [] :=
  fun x => nn.functional.affine x 3 1

/-! ## Float checks -/

/-- Absolute-tolerance float compare. -/
def approx (a b : Float) (tol : Float := 1e-6) : Bool := (a - b).abs ≤ tol

/-- Positive control: `name`'s autograd gradient is approximately equal to the expected value;
throws on mismatch. -/
def expectGrad (name : String) (got expected : Float) (tol : Float := 1e-6) : IO Unit :=
  if approx got expected tol then
    IO.println s!"[PASS] {name}: grad = {got} ≈ {expected}"
  else
    throw <| IO.userError s!"[FAIL] {name}: grad = {got}, expected {expected}"

/-- Negative control: the gradient must *not* equal `wrong`; throws if it does. -/
def expectNot (name : String) (got wrong : Float) (tol : Float := 1e-6) : IO Unit :=
  if approx got wrong tol then
    throw <| IO.userError s!"[FAIL-NEG] {name}: grad = {got} wrongly matched {wrong}"
  else
    IO.println s!"[PASS-NEG] {name}: grad = {got} ≠ {wrong} (test discriminates)"

/-- Differentiate a scalar→scalar `Fn` at a Float point, returning the gradient. -/
def gradAt (f : autograd.Function [] []) (x0 : Float) :
    IO Float := do
  let x := Tensor.full [] x0
  let g ← autograd.grad (σ := []) (α := Float) f x
  pure g.item

/--
Compare each derivative with its analytic value and reject a plausible wrong value.

The positive checks establish agreement at the sampled point; the negative controls show that
those points and tolerances distinguish the particular mistakes named below.
-/
def checkAll : IO Unit := do
  -- exp:  d/dx eˣ = eˣ
  let ge ← gradAt expFn 0.5
  expectGrad "exp"   ge (Float.exp 0.5)
  expectNot  "exp≠1" ge 1.0                       -- a constant-1 gradient would be caught

  -- affine:  d/dx (3x+1) = 3
  let ga ← gradAt affineFn 0.5
  expectGrad "affine(3x+1)"   ga 3.0
  expectNot  "affine≠1"       ga 1.0              -- the slope is 3, not 1

  -- exp(-2x):  d/dx e^{-2x} = -2·e^{-2x}
  let gn ← gradAt expNegativeTwoFn 0.5
  expectGrad "exp(-2x)"      gn ((-2.0) * Float.exp (-1.0))
  -- A positive derivative would have the wrong sign for this decreasing function.
  expectNot  "exp(-2x) sign" gn (( 2.0) * Float.exp (-1.0))

  IO.println "[transcendentals] all positive + negative controls passed ✓"

/-- Command-line help for the transcendental autograd checks. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean transcendental autograd checks"
    , ""
    , "Usage:"
    , "  lake exe torchlean transcendentals"
    , ""
    , "Checks derivatives of exp(x), 3x+1, and exp(-2x) at x=0.5 (Float tolerance 1e-6)."
    , "PASS-NEG means a deliberately wrong derivative was rejected. No data or GPU needed."
    ]

/-- Entry point. -/
def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println usage
    return
  CLI.requireNoArgs "transcendentals" args
  checkAll

end NN.Examples.Functional.Transcendentals
