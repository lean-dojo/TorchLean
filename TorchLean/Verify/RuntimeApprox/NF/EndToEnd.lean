/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.Graph.Autograd
public import TorchLean.Verify.RuntimeApprox.NF.Backprop
public import TorchLean.Verify.RuntimeApprox.OptimizerContract

/-!
# NF End-To-End GraphData Bridge

End-to-end runtime→spec bridge for NF graphs executed as `GraphData`.

The modules under `TorchLean/Verify/RuntimeApprox/NF/` provide per-op NF approximation lemmas and
compose them over `RevGraph` via `RevGraph.eval_approx` and `NF.backprop_approx`. The
theorems here carry those results to `TorchLean.Tape.GraphData`, the executable SSA/DAG
form used by typed graph execution. A `GraphData` stores forward, JVP, and VJP functions but none of
the derivative laws required by the proof-carrying `Graph` type.

The bridge covers forward evaluation and reverse accumulation. The erased JVP field repeats the
forward map, so forward-mode differentiation is outside its scope. These NF definitions are
mathematical computations over real-valued rounding models, and native CPU/GPU execution and
serialization need their own guarantees.
-/

@[expose] public section

namespace TorchLean
namespace NF

open TorchLean.Tensor

open Tape

noncomputable section

open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq

variable {β : Radix} {fexp : ℤ → ℤ} [ValidExp fexp]
variable {rnd : ℝ → ℤ} [ValidRndToNearest rnd]

local notation "R" => NF β fexp rnd

omit [ValidExp fexp] [ValidRndToNearest rnd] in
/--
Executable forward-pass soundness for an NF `RevGraph` erased to `GraphData`.

The `GraphData` forward interpreter computes the same runtime context as `RevGraph.evalRuntime`,
so the bound of `RevGraph.eval_approx` carries over to the executable representation.
-/
theorem eval_approx_graphData {Γ : List Shape} {ss : List Shape}
    (g : RevGraph (α := R) (toSpec := NF.toReal) Γ ss) :
    ∀ (xS : TensorPack ℝ Γ) (xR : TensorPack R Γ) (epsIn : EList Γ),
      ApproxCtx (α := R) (toSpec := NF.toReal) xS xR epsIn →
      ApproxCtx (α := R) (toSpec := NF.toReal)
        (RevGraph.evalReal g xS)
        (GraphData.eval (α := R) (Δ := Unit) (Γ := Γ) (ss := ss)
          (g := RevGraph.toGraphData (α := R)
            (toSpec := NF.toReal) (Γ := Γ) (ss := ss) g)
          xR ())
        (RevGraph.evalBounds g epsIn xR) := by
  intro xS xR epsIn hx
  have h := RevGraph.eval_approx (α := R) (toSpec := NF.toReal)
      (Γ := Γ) (ss := ss) g xS xR epsIn hx
  have hEq :
      GraphData.eval (α := R) (Δ := Unit) (Γ := Γ) (ss := ss)
            (g := RevGraph.toGraphData (α := R)
              (toSpec := NF.toReal) (Γ := Γ) (ss := ss) g)
            xR ()
        =
      RevGraph.evalRuntime (α := R) (toSpec := NF.toReal) (Γ := Γ)
        (ss := ss) g xR :=
    RevGraph.eval_toGraphData (α := R)
      (toSpec := NF.toReal) (Γ := Γ) (ss := ss) g xR
  simpa [hEq] using h

/--
Executable backward-pass soundness for an NF `RevGraph` erased to `GraphData`.

Given approximate inputs and approximate seed cotangents, executable `GraphData.backpropCtx`
approximates the real-spec reverse-mode result with the bound computed by the NF backend.
-/
theorem backprop_approx_graphData {Γ : List Shape} {ss : List Shape}
    (g : RevGraph (α := R) (toSpec := NF.toReal) Γ ss) :
    ∀ (xS : TensorPack ℝ Γ) (xR : TensorPack R Γ) (epsIn : EList Γ)
      (seedS : TensorPack ℝ (Γ ++ ss))
      (seedR : TensorPack R (Γ ++ ss)) (epsSeed : EList (Γ ++ ss)),
      ApproxCtx (α := R) (toSpec := NF.toReal) xS xR epsIn →
      ApproxCtx (α := R) (toSpec := NF.toReal) seedS seedR epsSeed
        →
        ApproxCtx (α := R) (toSpec := NF.toReal)
          (RevGraph.backpropReal g xS seedS)
          (GraphData.backpropCtx (α := R) (Δ := Unit) (Γ := Γ) (ss := ss)
            (g := RevGraph.toGraphData (α := R)
              (toSpec := NF.toReal) (Γ := Γ) (ss := ss) g)
            xR () seedR)
          (RevGraph.backpropBounds g epsIn xR epsSeed seedR (ctxAddBound (β := β) (fexp := fexp)
            (rnd := rnd))) := by
  intro xS xR epsIn seedS seedR epsSeed hx hseed
  have h := backprop_approx (β := β) (fexp := fexp) (rnd := rnd) (Γ := Γ) (ss := ss) g
      xS xR epsIn seedS seedR epsSeed hx hseed
  have hEq :
      GraphData.backpropCtx (α := R) (Δ := Unit) (Γ := Γ) (ss := ss)
            (g := RevGraph.toGraphData (α := R)
              (toSpec := NF.toReal) (Γ := Γ) (ss := ss) g)
            xR () seedR
        =
      RevGraph.backpropRuntime (α := R) (toSpec := NF.toReal) (Γ
        := Γ) (ss := ss) g xR seedR :=
    RevGraph.backpropCtx_toGraphData (α := R)
      (toSpec := NF.toReal) (Γ := Γ) (ss := ss) g xR seedR
  simpa [hEq] using h

/-- Extract one typed gradient and its numerical error from executable reverse mode.

The optimizer theorem below takes its gradient bound from this projection, which shows that entry
`i` of an approximated heterogeneous context is itself approximated. -/
theorem gradient_approx_graphData {Γ : List Shape} {ss : List Shape}
    (g : RevGraph (α := R)
      (toSpec := NF.toReal) Γ ss)
    (i : Fin Γ.length)
    (xS : TensorPack ℝ Γ) (xR : TensorPack R Γ) (epsIn : EList Γ)
    (seedS : TensorPack ℝ (Γ ++ ss)) (seedR : TensorPack R (Γ ++ ss))
    (epsSeed : EList (Γ ++ ss))
    (hx : ApproxCtx (α := R)
      (toSpec := NF.toReal) xS xR epsIn)
    (hseed : ApproxCtx (α := R)
      (toSpec := NF.toReal) seedS seedR epsSeed) :
    ApproxTensor (α := R) (toSpec := NF.toReal)
      (Pack.get (RevGraph.backpropReal g xS seedS) i)
      (Pack.get
        (GraphData.backpropCtx (α := R) (Δ := Unit) (Γ := Γ) (ss := ss)
          (g := RevGraph.toGraphData (α := R)
            (toSpec := NF.toReal)
            (Γ := Γ) (ss := ss) g) xR () seedR) i)
      (Pack.get
        (RevGraph.backpropBounds g epsIn xR epsSeed seedR
          (ctxAddBound (β := β) (fexp := fexp) (rnd := rnd))) i) :=
  ApproxCtx.get
    (α := R) (toSpec := NF.toReal)
    (backprop_approx_graphData (β := β) (fexp := fexp) (rnd := rnd)
      g xS xR epsIn seedS seedR epsSeed hx hseed) i

/-!
## A typed backward-and-update step

The seed context represents the cotangent supplied by a loss. The theorem below takes one gradient
from the reverse pass and applies a proved numerical optimizer contract to parameter tensors of
the matching shape. Those tensors need not equal context entry `i`, so a caller who claims an
update at the differentiated parameter proves that identification. Models with several parameter
tensors apply the theorem once per index.
-/

/-- Executable reverse mode followed by any numerical optimizer contract is sound: the updated
parameters and optimizer state lie within the contract's `nextError` budgets.

The theorem is polymorphic in the shape and the optimizer, so no optimizer needs its own graph
theorem. `assumptionsHold` is trivial for globally sound updates such as SGD and records domain
conditions for updates such as AdamW, whose square root and division must stay away from
singularities. -/
theorem update_approx_graphData {Γ : List Shape} {ss : List Shape}
    (g : RevGraph (α := R)
      (toSpec := NF.toReal) Γ ss)
    (i : Fin Γ.length)
    (contract : OptimizerContract R
      (NF.toReal))
    (xS : TensorPack ℝ Γ) (xR : TensorPack R Γ) (epsIn : EList Γ)
    (seedS : TensorPack ℝ (Γ ++ ss)) (seedR : TensorPack R (Γ ++ ss))
    (epsSeed : EList (Γ ++ ss))
    (paramsS : Tensor ℝ (Γ.get i)) (paramsR : Tensor R (Γ.get i)) (paramsError : ℝ)
    (stateS : contract.ExactState (Γ.get i))
    (stateR : contract.RuntimeState (Γ.get i))
    (stateError : contract.StateError (Γ.get i))
    (assumptions : contract.StepAssumptions (Γ.get i))
    (hx : ApproxCtx (α := R)
      (toSpec := NF.toReal) xS xR epsIn)
    (hseed : ApproxCtx (α := R)
      (toSpec := NF.toReal) seedS seedR epsSeed)
    (hparams : ApproxTensor (α := R)
      (toSpec := NF.toReal)
      paramsS paramsR paramsError)
    (hstate : contract.stateApprox stateS stateR stateError)
    (hAssumptions :
      let exactGradients := RevGraph.backpropReal g xS seedS
      let runtimeGradients := GraphData.backpropCtx (α := R) (Δ := Unit) (Γ := Γ) (ss := ss)
        (g := RevGraph.toGraphData (α := R)
          (toSpec := NF.toReal)
          (Γ := Γ) (ss := ss) g) xR () seedR
      let gradientError := Pack.get
        (RevGraph.backpropBounds g epsIn xR epsSeed seedR
          (ctxAddBound (β := β) (fexp := fexp) (rnd := rnd))) i
      contract.assumptionsHold stateS stateR stateError paramsS paramsR paramsError
        (Pack.get exactGradients i)
        (Pack.get runtimeGradients i)
        gradientError assumptions) :
    let exactGradients := RevGraph.backpropReal g xS seedS
    let runtimeGradients := GraphData.backpropCtx (α := R) (Δ := Unit) (Γ := Γ) (ss := ss)
      (g := RevGraph.toGraphData (α := R)
        (toSpec := NF.toReal)
        (Γ := Γ) (ss := ss) g) xR () seedR
    let gradientError := Pack.get
      (RevGraph.backpropBounds g epsIn xR epsSeed seedR
        (ctxAddBound (β := β) (fexp := fexp) (rnd := rnd))) i
    let nextError := contract.nextError stateError paramsError gradientError
      stateR paramsR (Pack.get runtimeGradients i) assumptions
    contract.stateApprox
        (contract.updateExact
          stateS paramsS (Pack.get exactGradients i)).optimizerState
        (contract.updateRuntime
          stateR paramsR (Pack.get runtimeGradients i)).optimizerState
        nextError.optimizerStateError ∧
      ApproxTensor (α := R) (toSpec := NF.toReal)
        (contract.updateExact
          stateS paramsS (Pack.get exactGradients i)).parameters
        (contract.updateRuntime
          stateR paramsR (Pack.get runtimeGradients i)).parameters
        nextError.parameterError := by
  dsimp only
  exact contract.update_sound stateS stateR stateError paramsS paramsR paramsError _ _ _ assumptions
    hstate hparams
    (gradient_approx_graphData (β := β) (fexp := fexp) (rnd := rnd)
      g i xS xR epsIn seedS seedR epsSeed hx hseed) hAssumptions

/-! ## Architecture-independent reporting -/

/-- Proof-free numerical summary of one forward/backward/update step.

The arrays retain typed-context order, and `parameterIndex` is the position of the selected
parameter in the input context `Γ`.
-/
structure TrainingStepTrace where
  /-- The name of the optimizer. -/
  optimizerName : String
  /-- The position of the selected parameter in `Γ`. -/
  parameterIndex : Nat
  /-- The forward error bound of each value, in context order. -/
  forwardErrors : Array ℝ
  /-- The backward error bound of each value, in context order. -/
  backwardErrors : Array ℝ
  /-- The error bound of the gradient of the selected parameter. -/
  gradientError : ℝ
  /-- The error bound of the updated parameter. -/
  parameterError : ℝ
  /-- The error bounds of the optimizer state, as the named numbers of `stateErrorReport`. -/
  optimizerStateErrors : Array (String × ℝ)
  /-- The step assumptions, as the named numbers of `assumptionReport`. -/
  assumptions : Array (String × ℝ)

/-- Assemble a mathematical report of the propagated error budgets.

The report carries no proofs. The theorems above give its fields their meaning, and
`update_approx_graphData` covers the update when the input, seed, parameter,
optimizer-state, and step assumptions hold. The real-valued fields and NF calculations are
noncomputable, so showing the numbers in a CLI or a serialized artifact needs a separate
executable representation.
-/
def traceTrainingStep {Γ : List Shape} {ss : List Shape}
    (g : RevGraph (α := R)
      (toSpec := NF.toReal) Γ ss)
    (i : Fin Γ.length)
    (contract : OptimizerContract R
      (NF.toReal))
    (xR : TensorPack R Γ) (epsIn : EList Γ)
    (seedR : TensorPack R (Γ ++ ss)) (epsSeed : EList (Γ ++ ss))
    (paramsR : Tensor R (Γ.get i)) (paramsError : ℝ)
    (stateR : contract.RuntimeState (Γ.get i))
    (stateError : contract.StateError (Γ.get i))
    (assumptions : contract.StepAssumptions (Γ.get i)) : TrainingStepTrace :=
  let forwardErrors := RevGraph.evalBounds g epsIn xR
  let backwardErrors := RevGraph.backpropBounds g epsIn xR epsSeed seedR
    (ctxAddBound (β := β) (fexp := fexp) (rnd := rnd))
  let runtimeGradients := GraphData.backpropCtx (α := R) (Δ := Unit) (Γ := Γ) (ss := ss)
    (g := RevGraph.toGraphData (α := R)
      (toSpec := NF.toReal)
      (Γ := Γ) (ss := ss) g) xR () seedR
  let gradientError := Pack.get backwardErrors i
  let nextError := contract.nextError stateError paramsError gradientError
    stateR paramsR (Pack.get runtimeGradients i) assumptions
  { optimizerName := contract.name
    parameterIndex := i.val
    forwardErrors := forwardErrors.toArray
    backwardErrors := backwardErrors.toArray
    gradientError
    parameterError := nextError.parameterError
    optimizerStateErrors := contract.stateErrorReport nextError.optimizerStateError
    assumptions := contract.assumptionReport assumptions }

end
end NF
end TorchLean
