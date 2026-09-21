/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.Proofs.Autograd.Model
public import NN.Proofs.Autograd.Runtime.Link.GraphComposition
public import NN.Tactic.Autograd
import Batteries.Lean.LawfulMonad

/-!
# Autograd Transforms

Advanced differentiation operations are kept out of the first autograd tutorial:

- `jacrev` and `hessian` for tensor functions;
- `jvp` and `hvp` for model state;
- `Loss.detach` for an explicit gradient stop;
- repeated and mixed input derivatives of a smooth model;
- composing a custom operation's proof with checked first-order differentiation;
- arbitrary-order derivatives of a recorded graph;
- a complete model derivative call, including successful lowering.

Build this module after `NN.Examples.Quickstart.AutogradBasics`.

Run:

```bash
scripts/lake.sh exe torchlean autograd_transforms
```
-/

@[expose] public section

namespace NN.Examples.DeepDives.AutogradTransforms

open TorchLean

/-- Subcommand name, used in the usage text and in argument-error messages. -/
def exeName : String := "autograd_transforms"

/--
Componentwise squaring, whose Jacobian is the diagonal matrix `2x`. Small enough that the printed
rows can be checked by hand.
-/
def square {shape : Shape} : autograd.Function shape shape :=
  fun x => nn.functional.square x

/-- Mean of the squares: a scalar-valued function, so it has a Hessian to compute. -/
def meanSquare {shape : Shape} : autograd.Function shape [] :=
  fun x => do
    let squared ← nn.functional.square x
    nn.functional.mean squared

/-- One linear layer `2 -> 3`, seeded deterministically so the printed numbers are reproducible. -/
def model : nn.Sequential [2] [3] :=
  nn.build 0 (nn.linear 2 3)

/-- Run higher-order and directional differentiation examples. -/
def runDemo : IO Unit := do
  let x : Tensor Float [2] := [0.5, -1.2]

  let jacobian ← autograd.jacrev square x
  IO.println s!"Jacobian of x^2: {reprStr jacobian}"

  let hessian ← autograd.hessian meanSquare x
  IO.println s!"Hessian of mean(x^2): {reprStr hessian}"

  let state : autograd.model.State model Float :=
    autograd.model.initialState model
  let target : Tensor Float [3] := [0.7, 0.1, -0.5]
  let direction : autograd.model.State model Float :=
    autograd.model.fullState model 0.1
  let mse : autograd.model.Loss [3] [3] := autograd.model.Loss.meanSquaredError

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

  -- These directions perturb inputs, unlike the parameter direction passed to `jvp` above.
  let field : nn.Sequential [2] [2] := nn.build 0 nn.tanh
  let origin : Tensor Float [2] := [0, 0]
  let dx : Tensor Float [2] := [1, 0]
  let dy : Tensor Float [2] := [0, 1]
  let mixed ← autograd.model.derivative field nn.State.empty origin [dx, dy]
  let third ← autograd.model.derivative field nn.State.empty origin [dx, dx, dx]
  IO.println s!"mixed input derivative of tanh at zero: {reprStr mixed} (expected [0, 0])"
  IO.println s!"third input derivative along [1, 0]: {reprStr third} (expected [-2, 0])"

/-- Help text; the demo takes no flags. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean autograd transforms"
    , ""
    , "Usage:"
    , "  scripts/lake.sh exe torchlean autograd_transforms"
    ]

/-- Entry point. -/
def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println usage
    return
  CLI.requireNoArgs exeName args
  runDemo

namespace Correctness

open Proofs Proofs.Autograd

noncomputable section

/-- Add elementwise squaring to a proved graph, for any tensor shape.

`autograd` composes mathlib's scalar rules and lifts them to tensors; `snoc` carries the
certificate into the composed graph. No shape-specific proof is needed.
-/
def square {Γ ss : List Shape} {s : Shape} (graph : DGraph Γ ss)
    (input : Idx (Γ ++ ss) s) : DGraph Γ (ss ++ [s]) :=
  graph.snoc (TapeNodes.elemwise input (fun x => x * x) (fun x => 2 * x))
    (by autograd)

/-- Squaring twice reuses the same primitive proof; no new backward rule is needed. -/
def fourthPower (s : Shape) : DGraph [s] [s, s] :=
  let first := square DGraph.nil (s := s) ⟨⟨0, by simp⟩, rfl⟩
  square first ⟨⟨1, by simp⟩, rfl⟩

/-- The second square is the output; the input and first square remain saved graph values. -/
def output (s : Shape) : Idx ([s] ++ [s, s]) s := ⟨⟨2, by simp⟩, rfl⟩

/-- A successful checked reverse pass computes the adjoint of the forward function's derivative.

This theorem is about the exact-real graph. Running the same arithmetic in floating point needs
a separate rounding-error argument, and higher derivatives need more than this first-order proof.
-/
theorem fourthPower_vjp (s : Shape) (inputs gradient : TensorPack ℝ [s]) (seed value : Tensor ℝ s)
    (checked : ((fourthPower s).toTypedGraph (output s)).vjpChecked inputs () seed =
      .ok (gradient, value)) :
    flattenCtx gradient =
      (fderiv ℝ (fun x => tensorToVec (((fourthPower s).toTypedGraph (output s)).forward
        (unflattenCtx x))) (flattenCtx inputs)).adjoint (tensorToVec seed) :=
  (fourthPower s).vjpChecked_adjoint_fderiv (output s) inputs seed (gradient, value) checked

end

end Correctness

namespace HigherOrder

open Runtime.Autograd.Model

/-- Apply `exp(x*x)` to a tensor, using the caller's scalar backend and shape. -/
def expSquare {α : Type} [Storage α] [Context α] {shape : Shape}
    (x : Tensor α shape) : Tensor α shape :=
  Tensor.map (fun y => MathFunctions.exp (y * y)) x

/-- Every derivative order, tensor shape, and direction tuple share one proof.

The seed and extraction functions are also used by `autograd.model.derivative`. This theorem
checks the tensor computation between them; model lowering requires its own correctness evidence.
-/
theorem expSquare_derivative {shape : Shape} (n : Nat)
    (directions : Fin n → Tensor ℝ shape) (x : Tensor ℝ shape) :
    Dual.Nested.tangentTensor (expSquare (Dual.Nested.seedTensor directions x)) =
      iteratedFDeriv ℝ n (fun y => Tensor.map (fun z => Real.exp (z * z)) y) x directions := by
  unfold expSquare
  autograd

open Runtime.Autograd.TypedGraph Runtime.Autograd.Torch

/-- Record squaring and exponentiation as two graph operations, for any scalar backend. -/
def expSquareProgram {α : Type} [Storage α] [Context α] (shape : Spec.Shape) :
    GraphM.M α [shape] (GraphM.Var shape) := do
  let x ← GraphM.arg 0 shape
  let squared ← GraphM.square x
  GraphM.exp squared

/-- Recording succeeds and the resulting graph evaluates the two requested operations.

The proof keeps the scalar backend abstract: recording checks shapes and indices, not numerical
values. This also prevents the proof from expanding real or nested-dual arithmetic internals.
-/
theorem expSquareProgram_eval {α : Type} [Storage α] [Context α] {shape : Spec.Shape}
    (x : Tensor α shape) :
    (lowerToTypedGraph (expSquareProgram (α := α) shape)).map
      (fun graph => graph.forward (TensorPack.singleton x)) =
      .ok (Tensor.expSpec (Tensor.mulSpec x x)) := by
  simp only [lowerToTypedGraph, lowerToTypedGraphWithData, expSquareProgram, GraphM.arg,
    GraphM.square, GraphM.mul, GraphM.exp, GraphM.push, GraphM.emptyWith, GraphM.mkIdx,
    GraphM.ctxLen]
  simp
  dsimp only [Bind.bind, Except.bind, Functor.map, Except.map]
  simp
  rfl

/-- The graph returned by the recorder has the specified forward function. -/
private theorem expSquareProgram_forward {α : Type} [Storage α] [Context α] {shape : Spec.Shape}
    (graph : TypedGraph α [shape] shape)
    (recorded : lowerToTypedGraph (expSquareProgram (α := α) shape) = .ok graph)
    (x : Tensor α shape) :
    graph.forward (TensorPack.singleton x) = Tensor.expSpec (Tensor.mulSpec x x) := by
  simpa only [recorded, Except.map, Except.ok.injEq] using expSquareProgram_eval x

/-- Nested-dual execution of the recorded graph computes every mixed derivative of its real run.

Unlike the tensor-only example above, this statement includes the actual recorder and graph
execution. It does not assert correctness of arbitrary model lowering or the reverse pass.
-/
theorem expSquareProgram_derivative {shape : Spec.Shape} (n : Nat)
    (real : TypedGraph ℝ [shape] shape)
    (nested : TypedGraph (Dual.Nested ℝ n) [shape] shape)
    (hr : lowerToTypedGraph (expSquareProgram (α := ℝ) shape) = .ok real)
    (hn : lowerToTypedGraph (expSquareProgram (α := Dual.Nested ℝ n) shape) = .ok nested)
    (directions : Fin n → Tensor ℝ shape) (x : Tensor ℝ shape) :
    Dual.Nested.tangentTensor
      (nested.forward (TensorPack.singleton (Dual.Nested.seedTensor directions x))) =
      iteratedFDeriv ℝ n (fun y => real.forward (TensorPack.singleton y)) x directions := by
  simp only [expSquareProgram_forward _ hn, expSquareProgram_forward _ hr,
    Tensor.expSpec, Tensor.mulSpec, Tensor.mapSpec, funext Proofs.mathfunc_exp_eq_rexp]
  autograd

/-- A parameter-free model using the same square layer as executable neural networks. -/
def squareModel (shape : Spec.Shape) : nn.Sequential shape shape :=
  Layers.Seq.fromLayer Layers.square

private theorem squareModel_forward {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [LawfulMonad m] [Ops m α] (shape : Spec.Shape) :
    (Layers.Seq.forward (squareModel shape) (α := α) (m := m) :
      RefTy m α shape → m (RefTy m α shape)) = F.square := by
  funext x
  change Layers.Seq.forwardState (Layers.Seq.fromLayer Layers.square) .eval
    (RefList.nil.append .nil) x = F.square x
  exact Layers.Seq.forwardState_fromLayer_eval Layers.square .nil x

private def squareProgram {α : Type} [Storage α] [Context α] (shape : Spec.Shape) :
    GraphM.M α [shape] (GraphM.Var shape) :=
  GraphM.square { id := 0 }

private theorem squareModel_lower {α : Type} [Storage α] [Context α] {shape : Spec.Shape} :
    nn.lowerToTypedGraph (squareModel shape) (α := α) =
      Autodiff.Impl.okOrThrow (lowerToTypedGraph (squareProgram (α := α) shape)) := by
  unfold nn.lowerToTypedGraph
  have hvalid : nn.validate (squareModel shape) = .ok () := by
    simp [squareModel, Layers.Seq.fromLayer, Layers.Seq.validate, Layers.Layer.validate,
      Layers.square, Module.RuntimeInit.Plan.validate]
    rfl
  rw [hvalid]
  simp only [Autodiff.lowerToTypedGraph, squareModel_forward]
  rfl

private theorem squareProgram_eval {α : Type} [Storage α] [Context α] {shape : Spec.Shape}
    (x : Tensor α shape) :
    (lowerToTypedGraph (squareProgram (α := α) shape)).map
      (fun graph => graph.forward (TensorPack.singleton x)) = .ok (Tensor.mulSpec x x) := by
  simp only [lowerToTypedGraph, lowerToTypedGraphWithData, squareProgram,
    GraphM.square, GraphM.mul, GraphM.push, GraphM.emptyWith, GraphM.mkIdx, GraphM.ctxLen]
  simp
  dsimp only [Bind.bind, Except.bind, Functor.map, Except.map]
  simp
  rfl

/-- The public IO transform returns the exact mixed derivative, for every shape and order.

There is no assumed lowering result: the proof checks model validation, graph recording, and
execution before applying `autograd` to the arithmetic. The empty direction list is included.
-/
theorem squareModel_derivative {shape : Spec.Shape} (x : Tensor ℝ shape)
    (directions : List (Tensor ℝ shape)) :
    autograd.model.derivative (squareModel shape) nn.State.empty x directions =
      pure (iteratedFDeriv ℝ directions.length (fun y => Tensor.mulSpec y y) x
        (fun i : Fin directions.length => directions[i])) := by
  simp only [autograd.model.derivative, squareModel_lower]
  have heval := squareProgram_eval (Dual.Nested.seedTensor
    (fun i : Fin directions.length => directions[i]) x)
  cases h : lowerToTypedGraph (squareProgram (α := Dual.Nested ℝ directions.length) shape) with
  | error err => simp [h, Except.map] at heval
  | ok graph =>
      simp only [h, Except.map, Except.ok.injEq] at heval
      dsimp only [Autodiff.Impl.okOrThrow]
      -- Normalize the empty state before applying IO laws to the recorded graph's type.
      dsimp only [squareModel, Layers.Seq.fromLayer, Layers.Seq.stateShapes, Layers.square]
      simp only [nn.TypedGraphModel.forward_empty]
      change (do
        let g ← (pure graph : IO _)
        pure (Dual.Nested.tangentTensor (g.forward (TensorPack.singleton
          (Dual.Nested.seedTensor (fun i : Fin directions.length => directions[i]) x))))) = _
      simp only [pure_bind]
      congr 1
      rw [heval]
      simp only [Tensor.mulSpec]
      autograd

end HigherOrder

end NN.Examples.DeepDives.AutogradTransforms
