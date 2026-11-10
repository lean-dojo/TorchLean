/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Autograd.Function
public import TorchLean.Tensor.Op.Table.Elementwise
public import TorchLean.NN.RL.PPO.ActorCritic
public import TorchLean.NN.Sequential.Composition
public import TorchLean.NN.Autograd.Model.Reverse
public import TorchLean.NN.Tape.DGraph.Checked
public import TorchLean.NN.Tactic.Autograd
import Batteries.Lean.LawfulMonad
import all Init.System.IO
public import TorchLean.NN.Data.Source
public import TorchLean.NN.Builder
public import TorchLean.NN.Training.Reporting
public import TorchLeanExamples.Quickstart.Autograd

/-!
# Autograd Transforms

The differentiation tools that the first autograd tutorial leaves out:

- `jacrev` and `hessian` for tensor functions,
- `jvp` and `hvp` for model state,
- `Loss.detach` to stop a gradient,
- repeated and mixed input derivatives of a smooth model,
- a custom operation's proof composed with checked first-order differentiation,
- derivatives of every order of a recorded graph,
- a complete model derivative call, lowering included.

The demo prints the first four, and `fourthPower_vjp`, `expSquareM_derivative` and
`squareModel_derivative` prove the other three. The Hessian example reuses `meanSquare` from the
`quickstart_autograd` example (`TorchLeanExamples.Quickstart.Autograd`). Run the demo with:

```bash
scripts/lake.sh exe torchlean autograd_transforms
```
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean.Examples.AutogradTransforms

/-- Subcommand name, used in argument-error messages. -/
def exeName : String := "autograd_transforms"

/--
Componentwise squaring, whose Jacobian is the diagonal matrix with entries `2 * x`, so the printed
rows are easy to check by hand.
-/
def square {shape : Shape} : autograd.Function shape shape :=
  fun x => nn.functional.square x

/-- One linear layer `2 -> 3`, seeded deterministically so the printed numbers are reproducible. -/
def model : nn.Sequential [2] [3] :=
  nn.build 0 (nn.linear 2 3)

/--
Print the Jacobian of `square` and the Hessian of `meanSquare` at `[0.5, -1.2]`, a JVP, an HVP and
a detached gradient of the MSE loss of `model`, and two input derivatives of `tanh` at zero.
-/
def runDemo : IO Unit := do
  let x : Tensor Float [2] := [0.5, -1.2]

  let jacobian ← autograd.jacrev square x
  IO.println s!"Jacobian of x^2: {reprStr jacobian}"

  let hessian ← autograd.hessian Quickstart.Autograd.meanSquare x
  IO.println s!"Hessian of mean(x^2): {reprStr hessian}"

  let state : autograd.model.State model Float :=
    nn.initialState model
  let target : Tensor Float [3] := [0.7, 0.1, -0.5]
  let direction : autograd.model.State model Float :=
    .fill 0.1
  let mse : autograd.model.Loss [3] [3] := autograd.model.Loss.mse

  let directionalDerivative ←
    autograd.model.jvp model mse state x target direction
  IO.println s!"loss directional derivative = {directionalDerivative}"

  let curvature ←
    autograd.model.hvp model mse state x target direction
  IO.println s!"loss Hessian-vector product = {reprStr curvature}"

  let detached : autograd.model.Loss [3] [3] :=
    autograd.model.Loss.detach mse
  let stoppedGradient ← autograd.model.grad model detached state x target
  IO.println s!"state gradient after detaching the model output = {reprStr stoppedGradient}"

  -- `dx` and `dy` perturb the input, while `direction` perturbs the parameters of `model`.
  let field : nn.Sequential [2] [2] := nn.build 0 (nn.op .tanh)
  let origin : Tensor Float [2] := [0, 0]
  let dx : Tensor Float [2] := [1, 0]
  let dy : Tensor Float [2] := [0, 1]
  let mixed ← autograd.model.derivative field .nil origin [dx, dy]
  let third ← autograd.model.derivative field .nil origin [dx, dx, dx]
  IO.println s!"mixed input derivative of tanh at zero: {reprStr mixed} (expected [0, 0])"
  IO.println s!"third input derivative along [1, 0]: {reprStr third} (expected [-2, 0])"

/-- Help text for the demo, which takes no flags. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean autograd transforms"
    , ""
    , "Usage:"
    , "  scripts/lake.sh exe torchlean autograd_transforms"
    ]

/-- Entry point: print the help text for `--help` or `-h`, reject any other argument, and run the
demo. -/
def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println usage
    return
  CLI.requireNoArgs exeName args
  runDemo

section

open Tape TensorPack Tensor

noncomputable section

/-- Add elementwise squaring to a proved graph, for any tensor shape.

The `autograd` tactic composes mathlib's scalar rules and lifts them to tensors, and `snoc` carries
the certificate into the composed graph, so no shape-specific proof is needed.
-/
def snocSquare {Γ ss : List Shape} {s : Shape} (graph : DGraph Γ ss)
    (input : Idx (Γ ++ ss) s) : DGraph Γ (ss ++ [s]) :=
  graph.snoc (RealNode.elemwise input (fun x => x * x) (fun x => 2 * x))
    (by autograd)

/-- Squaring twice reuses the proof of `snocSquare`, so the fourth power needs no backward rule of
its own. -/
def fourthPower (s : Shape) : DGraph [s] [s, s] :=
  let first := snocSquare DGraph.nil (s := s) ⟨⟨0, by simp⟩, rfl⟩
  snocSquare first ⟨⟨1, by simp⟩, rfl⟩

/-- Index of the second square, the graph's output, with the input and the first square kept as
saved graph values. -/
def fourthPowerOutput (s : Shape) : Idx ([s] ++ [s, s]) s := ⟨⟨2, by simp⟩, rfl⟩

/-- If the checked reverse pass of `fourthPower` succeeds, its gradient is the adjoint of the
forward function's derivative applied to the seed.

The statement is about the exact-real graph. The same arithmetic in floating point needs a separate
rounding-error argument, and higher derivatives need more than this first-order proof.
-/
theorem fourthPower_vjp (s : Shape) (inputs gradient : TensorPack ℝ [s]) (seed value : Tensor ℝ s)
    (checked : ((fourthPower s).toTypedGraph (fourthPowerOutput s)).vjpChecked inputs () seed =
      .ok (gradient, value)) :
    TensorPack.toVec gradient =
      (fderiv ℝ (fun x => Tensor.toVec (((fourthPower s).toTypedGraph (fourthPowerOutput s)).forward
        (TensorPack.ofVec x) ())) (TensorPack.toVec inputs)).adjoint (Tensor.toVec seed) :=
  (fourthPower s).vjpChecked_eq_fderiv (fourthPowerOutput s) inputs seed (gradient, value) checked

end

end

open nn

/-- Apply `exp(x*x)` to a tensor, using the caller's scalar backend and shape. -/
def expSquare {α : Type} [Storage α] [Context α] {shape : Shape}
    (x : Tensor α shape) : Tensor α shape :=
  Tensor.map (fun y => MathFunctions.exp (y * y)) x

/-- For every order `n`, shape and tuple of `n` directions, nested-dual evaluation of `expSquare`
returns the `n`-th iterated Fréchet derivative of the entrywise map `exp (y * y)` along them.

`autograd.model.derivative` uses the same seed and extraction functions, `Dual.Nested.seedTensor`
and `Dual.Nested.tangentTensor`. The theorem covers the tensor computation between them, and model
lowering needs a correctness proof of its own.
-/
theorem expSquare_derivative {shape : Shape} (n : Nat)
    (directions : Fin n → Tensor ℝ shape) (x : Tensor ℝ shape) :
    Dual.Nested.tangentTensor (expSquare (Dual.Nested.seedTensor directions x)) =
      iteratedFDeriv ℝ n (fun y => Tensor.map (fun z => Real.exp (z * z)) y) x directions := by
  unfold expSquare
  autograd

open TypedGraph WithData Ops

/-- Record squaring and exponentiation as two graph operations, for any scalar backend. -/
def expSquareM {α : Type} [Storage α] [Context α] (shape : Shape) :
    GraphM.M α [shape] (GraphM.Var shape) := do
  let x ← GraphM.arg 0 shape
  let squared ← GraphM.apply .mul (x, x, ())
  GraphM.apply .exp (squared, ())

/-- Recording `expSquareM` succeeds at any scalar backend, and the recorded graph maps `x` to
`Tensor.exp (x * x)`.

The proof keeps the backend abstract because recording checks only shapes and indices, so it never
unfolds real or nested-dual arithmetic.
-/
theorem expSquareM_eval {α : Type} [Storage α] [Context α] {shape : Shape}
    (x : Tensor α shape) :
    (lower (expSquareM (α := α) shape)).map
      (fun graph => graph.forward (Pack.singleton x) ()) =
      .ok (Tensor.exp (x * x)) := by
  simp only [lower, expSquareM, GraphM.arg,
    GraphM.run, GraphM.empty]
  simp
  -- Each recording pushes the node of its arm's rule (`Op.rule?_mul`, `Op.rule?_exp`).
  erw [GraphM.apply_run (α := α) (Δ := Unit) (Γ := [shape]) (Op.mul (s := shape))
    ({ id := 0 }, { id := 0 }, ()) ⟨[], .nil, #[], 0⟩ (rule := _) rfl Op.rule?_mul rfl rfl
    (xs := (⟨⟨0, by simp⟩, rfl⟩, ⟨⟨0, by simp⟩, rfl⟩, ()))
    (by simp [GraphM.resolve, GraphM.mkIdx, GraphM.ctxLen]; rfl)]
  simp only [bind, Except.bind]
  erw [GraphM.apply_run (α := α) (Δ := Unit) (Γ := [shape]) (Op.exp (s := shape))
    ({ id := [shape].length + [].length }, ()) ⟨[] ++ [shape], .snoc .nil (GraphM.opNode _ _ _ _),
      #[], 0⟩ (rule := _) rfl Op.rule?_exp rfl rfl (xs := (⟨⟨1, by simp⟩, rfl⟩, ()))
    (by simp [GraphM.resolve, GraphM.mkIdx, GraphM.ctxLen])]
  simp [GraphM.mkIdx, GraphM.ctxLen, Except.map, Functor.map]
  -- The output node's forward map (`GraphM.apply_forward`) reads the first node's.
  rfl

/-- Any graph the recorder returns maps `x` to `Tensor.exp (x * x)`. -/
private theorem expSquareM_forward {α : Type} [Storage α] [Context α] {shape : Shape}
    (graph : TypedGraph α [shape] shape)
    (recorded : lower (expSquareM (α := α) shape) = .ok graph)
    (x : Tensor α shape) :
    graph.forward (Pack.singleton x) () = Tensor.exp (x * x) := by
  simpa only [recorded, Except.map, Except.ok.injEq] using expSquareM_eval x

/-- Nested-dual execution of the recorded graph computes every mixed derivative of its real run.

Here the recorder and graph execution are part of the statement, while `expSquare_derivative` is
about the tensor function alone. General model lowering and the reverse pass are not covered.
-/
theorem expSquareM_derivative {shape : Shape} (n : Nat)
    (real : TypedGraph ℝ [shape] shape)
    (nested : TypedGraph (Dual.Nested ℝ n) [shape] shape)
    (hr : lower (expSquareM (α := ℝ) shape) = .ok real)
    (hn : lower (expSquareM (α := Dual.Nested ℝ n) shape) = .ok nested)
    (directions : Fin n → Tensor ℝ shape) (x : Tensor ℝ shape) :
    Dual.Nested.tangentTensor
      (nested.forward (Pack.singleton (Dual.Nested.seedTensor directions x)) ()) =
      iteratedFDeriv ℝ n (fun y => real.forward (Pack.singleton y) ()) x directions := by
  simp only [expSquareM_forward _ hn, expSquareM_forward _ hr,
    Tensor.exp, Tensor.mul_def, funext MathFunctions.exp_real]
  autograd

/-- A parameter-free model made of `nn.Layer.square`, the square layer of executable networks. -/
def squareModel (shape : Shape) : nn.Sequential shape shape :=
  nn.Sequential.fromLayer Layer.square

private theorem squareModel_forward {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [LawfulMonad m] [Ops m α] (shape : Shape) :
    (nn.Sequential.forward (squareModel shape) (α := α) (m := m) :
      Ref (m := m) (α := α) shape → m (Ref (m := m) (α := α) shape)) = functional.square := by
  funext x
  change nn.Sequential.forwardState (nn.Sequential.fromLayer Layer.square) .eval
    (Pack.nil.append .nil) x = functional.square x
  exact nn.Sequential.forwardState_fromLayer_eval Layer.square .nil x

private def squareM {α : Type} [Storage α] [Context α] (shape : Shape) :
    GraphM.M α [shape] (GraphM.Var shape) :=
  GraphM.apply .mul ({ id := 0 }, { id := 0 }, ())

private theorem squareModel_lower {α : Type} [Storage α] [Context α] {shape : Shape} :
    nn.Sequential.lowerToTypedGraph (squareModel shape) (α := α) =
      IO.ofExcept (lower (squareM (α := α) shape)) := by
  unfold nn.Sequential.lowerToTypedGraph
  have hvalid : nn.Sequential.validate (squareModel shape) = .ok () := by
    simp [squareModel, nn.Sequential.fromLayer, nn.Sequential.validate, Layer.validate,
      Layer.square, validateStatic, Init.Plan.validate]
    rfl
  rw [hvalid]
  simp only [Autodiff.lowerToTypedGraph, squareModel_forward]
  rfl

private theorem squareM_checked {α : Type} [Storage α] [Context α] {shape : Shape}
    (x : Tensor α shape) :
    (lower (squareM (α := α) shape)).bind (fun graph =>
      (TypedGraph.lowerToTapeChecked graph.data (Pack.singleton x) ()).map
        (fun result => TensorPack.getIdx result.2 graph.output)) = .ok (x * x) := by
  simp only [lower, squareM, GraphM.run,
    GraphM.empty]
  -- The recording pushes the node of `Op.rule?_mul`'s rule.
  rw [GraphM.apply_run (α := α) (Δ := Unit) (Γ := [shape]) (Op.mul (s := shape))
    ({ id := 0 }, { id := 0 }, ()) ⟨[], .nil, #[], 0⟩ (rule := _) rfl Op.rule?_mul rfl rfl
    (xs := (⟨⟨0, by simp⟩, rfl⟩, ⟨⟨0, by simp⟩, rfl⟩, ()))
    (by simp [GraphM.resolve, GraphM.mkIdx, GraphM.ctxLen]; rfl)]
  simp [GraphM.mkIdx, GraphM.ctxLen, bind, Except.bind, Except.map]
  rfl

/-- For every shape and order, `autograd.model.derivative` on `squareModel` returns the iterated
Fréchet derivative of `y * y` along the slices `directions[i]`.

No lowering result is assumed: the proof goes through model validation, graph recording and
execution before `autograd` handles the arithmetic. Order zero, with an empty direction axis, is
included.
-/
theorem squareModel_derivative {shape : Shape} {order : Nat} (x : Tensor ℝ shape)
    (directions : Tensor ℝ (Shape.dim order shape)) :
    autograd.model.derivative (squareModel shape) .nil x directions =
      pure (iteratedFDeriv ℝ order (fun y => y * y) x
        (fun i : Fin order => directions[i])) := by
  simp only [autograd.model.derivative, squareModel_lower]
  have hcheck := squareM_checked (Dual.Nested.seedTensor
    (fun i : Fin order => directions[i]) x)
  cases h : lower (squareM (α := Dual.Nested ℝ order) shape) with
  | error err => simp [h, Except.bind] at hcheck
  | ok graph =>
      simp only [h, Except.bind] at hcheck
      cases hc : TypedGraph.lowerToTapeChecked graph.data
          (Pack.singleton (Dual.Nested.seedTensor (fun i : Fin order => directions[i]) x)) () with
      | error err => simp [hc, Except.map] at hcheck
      | ok result =>
          simp only [hc, Except.map, Except.ok.injEq] at hcheck
          -- Normalize the empty state before applying IO laws to the recorded graph's type.
          dsimp only [squareModel, nn.Sequential.fromLayer, nn.Sequential.stateShapes,
            Layer.square]
          dsimp only [IO.ofExcept]
          change (do
            let g ← (pure graph : IO _)
            let (_, context) ← IO.ofExcept (TypedGraph.lowerToTapeChecked g.data
              (Pack.singleton (Dual.Nested.seedTensor (fun i : Fin order => directions[i]) x))
              ())
            pure (Dual.Nested.tangentTensor (TensorPack.getIdx context g.output))) = _
          simp only [pure_bind, hc]
          dsimp only [IO.ofExcept]
          simp only [pure_bind]
          congr 1
          rw [hcheck]
          simp only [Tensor.mul_def]
          autograd

end TorchLean.Examples.AutogradTransforms
