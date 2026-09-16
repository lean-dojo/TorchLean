/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.Graph.LinkAutogradAlgebra
public import NN.Proofs.RuntimeApprox.NF.BackwardOps.Main
public import NN.Proofs.RuntimeApprox.Optimizer

/-!
# NF End-To-End GraphData Bridge

End-to-end runtime→spec bridge for NF graphs executed as `GraphData`.

`NN.Proofs.RuntimeApprox.NF` provides per-op NF approximation lemmas and composes them over
`RevGraph` via `RevGraph.eval_approx` and `NFBackend.backprop_approx`.

This file links those results to the executable SSA/DAG form used by typed graph execution:
`Proofs.Autograd.Algebra.GraphData`. A `GraphData` stores forward, JVP, and VJP functions; it does
not carry the derivative laws required by the proof-carrying `Graph` type.

In other words, this is where the abstract approximation graph model meets the executable graph
interpreter used elsewhere in TorchLean.
-/

@[expose] public section


namespace Proofs
namespace RuntimeApprox
namespace NFBackend

open Spec TorchLean
open TorchLean TorchLean.Tensor
open NN.MLTheory.Robustness.Spec

open LinkAutogradAlgebra
open Proofs.Autograd.Algebra

noncomputable section

open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq
open Proofs.RuntimeRoundingApprox

variable {β : Radix} {fexp : ℤ → ℤ} [ValidExp fexp]
variable {rnd : ℝ → ℤ} [ValidRndToNearest rnd]

local notation "R" => NF β fexp rnd

omit [ValidExp fexp] [ValidRndToNearest rnd] in
/--
Executable forward-pass soundness for an NF `RevGraph` erased to `GraphData`.

The theorem says that evaluating the executable `GraphData` forward interpreter gives the same
runtime context covered by `RevGraph.eval_approx`, so the abstract graph approximation theorem
applies to the executable representation.
-/
theorem eval_approx_graphData {Γ : List Shape} {ss : List Shape}
    (g : RevGraph (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ ss) :
    ∀ (xS : TorchLean.TensorPack SpecScalar Γ) (xR : TorchLean.TensorPack R Γ) (epsIn : EList Γ),
      approxCtx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR epsIn →
      approxCtx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (RevGraph.evalSpec g xS)
        (GraphData.eval (α := R) (Δ := Unit) (Γ := Γ) (ss := ss)
          (g := LinkAutogradAlgebra.RevGraph.toGraphData (α := R)
            (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) (Γ := Γ) (ss := ss) g)
          xR ())
        (RevGraph.evalBounds g epsIn xR) := by
  intro xS xR epsIn hx
  have h := RevGraph.eval_approx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Γ := Γ) (ss := ss) g xS xR epsIn hx
  have hEq :
      GraphData.eval (α := R) (Δ := Unit) (Γ := Γ) (ss := ss)
            (g := LinkAutogradAlgebra.RevGraph.toGraphData (α := R)
              (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) (Γ := Γ) (ss := ss) g)
            xR ()
        =
      RevGraph.evalRuntime (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) (Γ := Γ)
        (ss := ss) g xR :=
    LinkAutogradAlgebra.RevGraph.evalRuntime_of_toGraphData (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) (Γ := Γ) (ss := ss) g xR
  simpa [hEq] using h

/--
Executable backward-pass soundness for an NF `RevGraph` erased to `GraphData`.

Given approximate inputs and approximate seed cotangents, executable `GraphData.backpropCtx`
approximates the real-spec reverse-mode result with the bound computed by the NF backend.
-/
theorem backprop_approx_graphData {Γ : List Shape} {ss : List Shape}
    (g : RevGraph (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ ss) :
    ∀ (xS : TorchLean.TensorPack SpecScalar Γ) (xR : TorchLean.TensorPack R Γ) (epsIn : EList Γ)
      (seedS : TorchLean.TensorPack SpecScalar (Γ ++ ss))
      (seedR : TorchLean.TensorPack R (Γ ++ ss)) (epsSeed : EList (Γ ++ ss)),
      approxCtx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR epsIn →
      approxCtx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) seedS seedR epsSeed
        →
        approxCtx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (RevGraph.backpropSpec g xS seedS)
          (GraphData.backpropCtx (α := R) (Δ := Unit) (Γ := Γ) (ss := ss)
            (g := LinkAutogradAlgebra.RevGraph.toGraphData (α := R)
              (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) (Γ := Γ) (ss := ss) g)
            xR () seedR)
          (RevGraph.backpropBounds g epsIn xR epsSeed seedR (ctxAddBound (β := β) (fexp := fexp)
            (rnd := rnd))) := by
  intro xS xR epsIn seedS seedR epsSeed hx hseed
  have h := backprop_approx (β := β) (fexp := fexp) (rnd := rnd) (Γ := Γ) (ss := ss) g
      xS xR epsIn seedS seedR epsSeed hx hseed
  have hEq :
      GraphData.backpropCtx (α := R) (Δ := Unit) (Γ := Γ) (ss := ss)
            (g := LinkAutogradAlgebra.RevGraph.toGraphData (α := R)
              (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) (Γ := Γ) (ss := ss) g)
            xR () seedR
        =
      RevGraph.backpropRuntime (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) (Γ
        := Γ) (ss := ss) g xR seedR :=
    LinkAutogradAlgebra.RevGraph.backpropRuntime_of_toGraphData (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) (Γ := Γ) (ss := ss) g xR seedR
  simpa [hEq] using h

/-- Extract one typed gradient and its numerical error from executable reverse mode.

This is the common bridge used by every optimizer theorem below. Keeping the projection here avoids
repeating a proof that the `i`th entry of an approximated heterogeneous context is itself
approximated. -/
theorem backprop_gradient_approx_graphData {Γ : List Shape} {ss : List Shape}
    (g : RevGraph (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ ss)
    (i : Fin Γ.length)
    (xS : TorchLean.TensorPack SpecScalar Γ) (xR : TorchLean.TensorPack R Γ) (epsIn : EList Γ)
    (seedS : TorchLean.TensorPack SpecScalar (Γ ++ ss)) (seedR : TorchLean.TensorPack R (Γ ++ ss))
    (epsSeed : EList (Γ ++ ss))
    (hx : approxCtx (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR epsIn)
    (hseed : approxCtx (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) seedS seedR epsSeed) :
    approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (TorchLean.TensorPack.get (RevGraph.backpropSpec g xS seedS) i)
      (TorchLean.TensorPack.get
        (GraphData.backpropCtx (α := R) (Δ := Unit) (Γ := Γ) (ss := ss)
          (g := LinkAutogradAlgebra.RevGraph.toGraphData (α := R)
            (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
            (Γ := Γ) (ss := ss) g) xR () seedR) i)
      (EList.get
        (RevGraph.backpropBounds g epsIn xR epsSeed seedR
          (ctxAddBound (β := β) (fexp := fexp) (rnd := rnd))) i) := by
  exact approxCtx_get
    (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
    (backprop_approx_graphData (β := β) (fexp := fexp) (rnd := rnd)
      g xS xR epsIn seedS seedR epsSeed hx hseed) i

/-!
## A typed backward-and-update step

The seed context represents the cotangent supplied by a loss. The theorem below takes one gradient
from the executable reverse pass and applies an arbitrary proved numerical optimizer contract to
the corresponding parameter tensor. It is deliberately indexed by `i`: models with many parameter
tensors apply the same theorem to each entry of the typed parameter context.
-/

/-- Executable reverse mode followed by any valid numerical optimizer contract is sound.

The theorem is shape-polymorphic and optimizer-polymorphic. `assumptionsHold` is trivial for
globally sound updates such as SGD and records domain conditions for updates such as AdamW whose
square root and division must stay away from singularities. No optimizer needs a separate graph
theorem. Models with several parameter tensors instantiate this theorem at each typed index. -/
theorem backprop_optimizer_update_approx_graphData {Γ : List Shape} {ss : List Shape}
    (g : RevGraph (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ ss)
    (i : Fin Γ.length)
    (contract : Proofs.RuntimeApprox.Optimizer.NumericalStepContract R
      (toSpec (β := β) (fexp := fexp) (rnd := rnd)))
    (xS : TorchLean.TensorPack SpecScalar Γ) (xR : TorchLean.TensorPack R Γ) (epsIn : EList Γ)
    (seedS : TorchLean.TensorPack SpecScalar (Γ ++ ss)) (seedR : TorchLean.TensorPack R (Γ ++ ss))
    (epsSeed : EList (Γ ++ ss))
    (paramsS : Tensor ℝ (Γ.get i)) (paramsR : Tensor R (Γ.get i)) (paramsError : ℝ)
    (stateS : contract.ExactState (Γ.get i))
    (stateR : contract.RuntimeState (Γ.get i))
    (stateError : contract.StateError (Γ.get i))
    (assumptions : contract.StepAssumptions (Γ.get i))
    (hx : approxCtx (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR epsIn)
    (hseed : approxCtx (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) seedS seedR epsSeed)
    (hparams : approxTensor (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      paramsS paramsR paramsError)
    (hstate : contract.stateApprox stateS stateR stateError)
    (hAssumptions :
      let exactGradients := RevGraph.backpropSpec g xS seedS
      let runtimeGradients := GraphData.backpropCtx (α := R) (Δ := Unit) (Γ := Γ) (ss := ss)
        (g := LinkAutogradAlgebra.RevGraph.toGraphData (α := R)
          (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (Γ := Γ) (ss := ss) g) xR () seedR
      let gradientError := EList.get
        (RevGraph.backpropBounds g epsIn xR epsSeed seedR
          (ctxAddBound (β := β) (fexp := fexp) (rnd := rnd))) i
      contract.assumptionsHold stateS stateR stateError paramsS paramsR paramsError
        (TorchLean.TensorPack.get exactGradients i)
        (TorchLean.TensorPack.get runtimeGradients i)
        gradientError assumptions) :
    let exactGradients := RevGraph.backpropSpec g xS seedS
    let runtimeGradients := GraphData.backpropCtx (α := R) (Δ := Unit) (Γ := Γ) (ss := ss)
      (g := LinkAutogradAlgebra.RevGraph.toGraphData (α := R)
        (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Γ := Γ) (ss := ss) g) xR () seedR
    let gradientError := EList.get
      (RevGraph.backpropBounds g epsIn xR epsSeed seedR
        (ctxAddBound (β := β) (fexp := fexp) (rnd := rnd))) i
    let nextError := contract.nextError stateError paramsError gradientError
      stateR paramsR (TorchLean.TensorPack.get runtimeGradients i) assumptions
    contract.stateApprox
        (contract.updateExact
          stateS paramsS (TorchLean.TensorPack.get exactGradients i)).optimizerState
        (contract.updateRuntime
          stateR paramsR (TorchLean.TensorPack.get runtimeGradients i)).optimizerState
        nextError.optimizerStateError ∧
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (contract.updateExact
          stateS paramsS (TorchLean.TensorPack.get exactGradients i)).parameters
        (contract.updateRuntime
          stateR paramsR (TorchLean.TensorPack.get runtimeGradients i)).parameters
        nextError.parameterError := by
  dsimp only
  exact contract.updateApprox stateS stateR stateError paramsS paramsR paramsError _ _ _ assumptions
    hstate hparams
    (backprop_gradient_approx_graphData (β := β) (fexp := fexp) (rnd := rnd)
      g i xS xR epsIn seedS seedR epsSeed hx hseed) hAssumptions

/-! ## Architecture-independent reporting -/

/-- Proof-free numerical summary of one forward/backward/update step.

The arrays retain typed-context order. A UI may attach architecture-specific display names after
lowering, but propagation itself does not inspect whether the graph came from an MLP, CNN,
transformer, or another model family.
-/
structure TrainingStepTrace where
  optimizerName : String
  parameterIndex : Nat
  forwardErrors : Array ℝ
  backwardErrors : Array ℝ
  gradientError : ℝ
  parameterError : ℝ
  optimizerStateErrors : Array (String × ℝ)
  assumptions : Array (String × ℝ)

/-- Compute the report consumed by CLI, file, or InfoView front ends.

This function is intentionally proof-free: `backprop_optimizer_update_approx_graphData` is the
theorem establishing its interpretation when the input, seed, parameter, optimizer-state, and
step assumptions hold.
-/
def trainingStepTrace {Γ : List Shape} {ss : List Shape}
    (g : RevGraph (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ ss)
    (i : Fin Γ.length)
    (contract : Proofs.RuntimeApprox.Optimizer.NumericalStepContract R
      (toSpec (β := β) (fexp := fexp) (rnd := rnd)))
    (xR : TorchLean.TensorPack R Γ) (epsIn : EList Γ)
    (seedR : TorchLean.TensorPack R (Γ ++ ss)) (epsSeed : EList (Γ ++ ss))
    (paramsR : Tensor R (Γ.get i)) (paramsError : ℝ)
    (stateR : contract.RuntimeState (Γ.get i))
    (stateError : contract.StateError (Γ.get i))
    (assumptions : contract.StepAssumptions (Γ.get i)) : TrainingStepTrace :=
  let forwardErrors := RevGraph.evalBounds g epsIn xR
  let backwardErrors := RevGraph.backpropBounds g epsIn xR epsSeed seedR
    (ctxAddBound (β := β) (fexp := fexp) (rnd := rnd))
  let runtimeGradients := GraphData.backpropCtx (α := R) (Δ := Unit) (Γ := Γ) (ss := ss)
    (g := LinkAutogradAlgebra.RevGraph.toGraphData (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Γ := Γ) (ss := ss) g) xR () seedR
  let gradientError := EList.get backwardErrors i
  let nextError := contract.nextError stateError paramsError gradientError
    stateR paramsR (TorchLean.TensorPack.get runtimeGradients i) assumptions
  { optimizerName := contract.name
    parameterIndex := i.val
    forwardErrors := forwardErrors.toArray
    backwardErrors := backwardErrors.toArray
    gradientError
    parameterError := nextError.parameterError
    optimizerStateErrors := contract.stateErrorReport nextError.optimizerStateError
    assumptions := contract.assumptionReport assumptions }

end
end NFBackend
end RuntimeApprox
end Proofs
