/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Module.Execution
public import NN.Runtime.RL.Algorithms.PolicyGradient
public import NN.Runtime.RL.PolicyGradient.Autograd
public import NN.Tests.Runtime.Cuda.Softmax

/-!
# CUDA Coverage: PPO Log-Probabilities

PPO records old log-probabilities during collection with `actionLogSoftmax`, which evaluates
`logSoftmaxVecSpec`. The CUDA log-softmax computes the same max-shifted formula, but in float32.
These checks pin down both facts:

- the CUDA log-softmax agrees with `logSoftmaxVecSpec` to float32 rounding, including a row whose
  chosen action has probability near `e^-60`;
- at identical parameters, the CUDA PPO objective with unit advantages is exactly 1 when the old
  log-probabilities come from the CUDA forward, and within float32 rounding of 1 when they come
  from the CPU formula in `Float`.
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace PPORatio

open Spec TorchLean
open TorchLean TorchLean.Tensor

def rows : Array (Array Float) := #[#[0.0, 1.0, -0.5], #[0.0, 30.0, -30.0]]

def chosen : Array (Fin 3) := #[⟨1, by decide⟩, ⟨2, by decide⟩]

def logits : Tensor Float [2, 3] :=
  (Tensor.ofFn fun i : Fin 6 => (rows[i.val / 3]!)[i.val % 3]!).reshape [2, 3] (by decide)

def actions : Tensor Float [2, 3] :=
  (Tensor.ofFn fun i : Fin 6 =>
    if chosen[i.val / 3]!.val == i.val % 3 then (1 : Float) else 0).reshape [2, 3] (by decide)

def row (i : Nat) : Tensor Float [3] := Tensor.ofFn fun j : Fin 3 => (rows[i]!)[j.val]!

/-- Old log-probabilities as PPO collection records them on the CPU. -/
def cpuOldLogProb : Tensor Float [2] := Tensor.ofFn fun i =>
  Runtime.RL.PolicyGradient.actionLogSoftmax (α := Float) (row i.val) chosen[i.val]!

/-- Row-wise `logSoftmaxVecSpec` in `Float`. -/
def referenceLogSoftmax : Tensor Float [2, 3] :=
  (Tensor.ofFn fun i : Fin 6 =>
    Tensor.getScalar (Activation.logSoftmaxVecSpec (α := Float) (row (i.val / 3)))
      ⟨i.val % 3, Nat.mod_lt _ (by decide)⟩).reshape [2, 3] (by decide)

/-- Mean clipped PPO objective, with the logits as the only state tensor. -/
def ppoObjective : Module.ObjectiveDefinition Unit [[2, 3]] [[2, 3], [2], [2]] :=
  { initState := .cons logits .nil
    loss := fun {_α} _ _ => fun {_m} _ _ => fun newLogits actionOneHot oldLogProb advantage => (do
      let objective ← Runtime.RL.PolicyGradient.Autograd.ppoClippedObjective (m := _m) (α := _α)
        (batch := 2) (nActions := 3) newLogits actionOneHot oldLogProb advantage
      Runtime.Autograd.Model.F.mean (m := _m) (α := _α) (s := .dim 2 .scalar) objective :
      _m (Runtime.Autograd.Model.RefTy _m _α [])) }

/-- Evaluate the objective on CUDA for the given old log-probabilities and unit advantages. -/
def cudaObjective (oldLogProb : Tensor Float [2]) : IO Float := do
  let objective ← Module.instantiate ppoObjective
    { execution := .eager, device := .cuda } (α := Float)
  let inputs := Arguments.Internal.fromTensorPack <|
    .cons actions (.cons oldLogProb (.cons (Tensor.full [2] (1 : Float)) .nil))
  pure (← objective.loss inputs Arguments.empty).item

def run : IO Unit := do
  IO.println "=== CUDA coverage: PPO log-probabilities ==="
  let (cudaLogSoftmax, _) ← Softmax.evalLogSoftmax .cuda logits (Tensor.full [2, 3] 0.0)
  -- Relative float32 rounding of values up to 60 in magnitude.
  Utils.assertTensorApprox (s := [2, 3]) "CUDA log_softmax against logSoftmaxVecSpec"
    cudaLogSoftmax referenceLogSoftmax (tol := 1e-5)
  let cudaOldLogProb : Tensor Float [2] := Tensor.ofFn fun i =>
    Tensor.getScalar (Tensor.unstack cudaLogSoftmax ⟨i.val, i.isLt⟩) chosen[i.val]!
  let sameDevice ← cudaObjective cudaOldLogProb
  unless sameDevice == 1 do
    throw <| IO.userError
      s!"CUDA PPO objective with CUDA old log-probabilities should be 1, got {sameDevice}"
  let crossDevice ← cudaObjective cpuOldLogProb
  unless Float.abs (crossDevice - 1) <= 1e-5 do
    throw <| IO.userError
      s!"CUDA PPO objective with CPU old log-probabilities should be near 1, got {crossDevice}"
  let logSoftmaxError := (List.finRange 6).foldl (init := 0.0) fun acc i =>
    let index : Fin 6 := i
    let flat := fun (t : Tensor Float [2, 3]) =>
      Tensor.getScalar (Tensor.unstack t ⟨index.val / 3, by omega⟩)
        ⟨index.val % 3, Nat.mod_lt _ (by decide)⟩
    Max.max acc (Float.abs (flat cudaLogSoftmax - flat referenceLogSoftmax))
  let nano := fun x : Float => s!"{x * 1e9}e-9"
  IO.println s!"CUDA log_softmax max error {nano logSoftmaxError}; PPO objective minus 1: \
    {nano (sameDevice - 1)} (CUDA old log-probs), {nano (crossDevice - 1)} (CPU old log-probs)"

end PPORatio
end Cuda
end Tests
