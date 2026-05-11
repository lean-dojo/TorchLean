/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.PPO.Rollout
public import NN.Runtime.Training.Log

-- We compute GAE/returns in a widget (meta) context, so the RL core must be available to meta code.
public meta import NN.Runtime.RL.Core
public meta import NN.Floats.IEEEExec.Exec32

public meta import NN.Widgets.Runtime.Training
public meta import ProofWidgets.Component.HtmlDisplay
public meta import ProofWidgets.Demos.Macro

/-!
# PPO Rollout Viewer

This module provides a small infoview widget for visualizing PPO rollouts as curves:

- `reward_t` (environment rewards),
- `return_t` (lambda-returns computed from GAE),
- `advantage_t` (GAE(λ) advantages).

Implementation note: we intentionally reuse TorchLean's generic training-log widget
(`NN.Widgets.Runtime.Training.trainLogHtml`) so we do not duplicate plotting/sparkline code.

References:
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
  (2015): https://arxiv.org/abs/1506.02438
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017):
  https://arxiv.org/abs/1707.06347

## Main definitions

- `ToVizFloat`: compact conversion class for plotting different scalar backends.
- `ppoRolloutTrainLog`: converts rollout tensors into `TrainLog` series.
- `ppoRolloutHtml`: delegates to the generic training viewer.
- `#ppo_rollout_view`: command form for quick rollout inspection.

## Implementation notes

- Reusing `trainLogHtml` keeps one plotting surface for many
  widget frontends.
- We keep backend conversion explicit (`ToVizFloat`) so adding a new scalar type is obvious and
  local.
- We use the exact RL-core GAE/return routines so visualized values match training semantics.

## References

- [ProofWidgets](https://github.com/leanprover-community/ProofWidgets4)
- [Lean community documentation style](https://leanprover-community.github.io/contribute/doc.html)

## Tags

ppo, gae, rollout, reinforcement-learning, visualization
-/

namespace NN.Widgets

public meta section

open scoped ProofWidgets.Jsx

open _root_.Spec
open _root_.Spec.Tensor

open _root_.Runtime.Training

namespace RL
namespace PPO

/--
Lossy conversion from a scalar backend `α` to Lean's `Float` for visualization.

This is intentionally a small widget-only typeclass. If you define your own scalar backend for RL,
add an instance here (or locally in your project) to enable `#ppo_rollout_view`.
-/
class ToVizFloat (α : Type) where
  toVizFloat : α → Float

instance : ToVizFloat Float := ⟨fun x => x⟩
instance : ToVizFloat TorchLean.Floats.IEEE754.IEEE32Exec :=
  ⟨TorchLean.Floats.IEEE754.IEEE32Exec.toFloat⟩

/--
Convert a length-`n` scalar vector tensor to an `Array Float` for plotting.

We pattern-match on the tensor representation to avoid runtime indexing overhead.
-/
private def vecToFloatArray {α : Type} [ToVizFloat α] {n : Nat} :
    Tensor α (.dim n .scalar) → Array Float
  | .dim f =>
      Array.ofFn (fun i : Fin n =>
        match f i with
        | .scalar x => ToVizFloat.toVizFloat x)

/-- Build a `TrainLog` containing reward/return/advantage curves for a fixed-horizon PPO rollout. -/
def ppoRolloutTrainLog {α : Type} [Context α] [ToVizFloat α] [DecidableEq Shape]
    {obsShape : Shape} {nActions horizon : Nat}
    (gamma lam : α)
    (r : _root_.Runtime.RL.PPO.Rollout α obsShape nActions horizon) :
    TrainLog :=
  let steps := r.steps

  let rewardsArr : Array α := steps.map (fun st => st.reward)
  let donesArr : Array Bool := steps.map (fun st => st.done)
  let valuesArr : Array α := steps.map (fun st => st.value)
  let nextValuesArr : Array α := steps.map (fun st => st.nextValue)

  let hRewards : horizon = rewardsArr.size := by
    have : rewardsArr.size = horizon := by
      simpa [rewardsArr, Array.size_map] using r.steps_size_eq_horizon
    simpa using this.symm
  let hDones : horizon = donesArr.size := by
    have : donesArr.size = horizon := by
      simpa [donesArr, Array.size_map] using r.steps_size_eq_horizon
    simpa using this.symm
  let hValues : horizon = valuesArr.size := by
    have : valuesArr.size = horizon := by
      simpa [valuesArr, Array.size_map] using r.steps_size_eq_horizon
    simpa using this.symm
  let hNextValues : horizon = nextValuesArr.size := by
    have : nextValuesArr.size = horizon := by
      simpa [nextValuesArr, Array.size_map] using r.steps_size_eq_horizon
    simpa using this.symm

  let rewards : Tensor α (.dim horizon .scalar) :=
    Tensor.ofArray1D (α := α) (n := horizon) rewardsArr hRewards
  let dones : Tensor Bool (.dim horizon .scalar) :=
    Tensor.ofArray1D (α := Bool) (n := horizon) donesArr hDones
  let values : Tensor α (.dim horizon .scalar) :=
    Tensor.ofArray1D (α := α) (n := horizon) valuesArr hValues
  let nextValues : Tensor α (.dim horizon .scalar) :=
    Tensor.ofArray1D (α := α) (n := horizon) nextValuesArr hNextValues

  let advRaw :=
    _root_.Runtime.RL.Core.generalizedAdvantageEstimationVec (α := α) (n := horizon)
      gamma lam rewards values nextValues dones
  let returns :=
    _root_.Runtime.RL.Core.returnsFromAdvantagesVec (α := α) (n := horizon) advRaw values

  let rewardsF : Array Float := rewardsArr.map ToVizFloat.toVizFloat
  let returnsF : Array Float := vecToFloatArray (α := α) (n := horizon) returns
  let advF : Array Float := vecToFloatArray (α := α) (n := horizon) advRaw

  let notes : Array String :=
    #[
      s!"gamma={ToVizFloat.toVizFloat gamma}",
      s!"lambda={ToVizFloat.toVizFloat lam}"
    ]
  let series : Array Series :=
    #[
      { name := "reward", values := rewardsF, color := "#f28e2b" },
      { name := "return", values := returnsF, color := "#4e79a7" },
      { name := "advantage", values := advF, color := "#e15759" }
    ]
  { title := "PPO rollout", series := series, notes := notes }

/-- Render a PPO rollout viewer as infoview HTML (reward/return/advantage curves + table). -/
def ppoRolloutHtml {α : Type} [Context α] [ToVizFloat α] [DecidableEq Shape]
    {obsShape : Shape} {nActions horizon : Nat}
    (gamma lam : α)
    (r : _root_.Runtime.RL.PPO.Rollout α obsShape nActions horizon) :
    ProofWidgets.Html :=
  trainLogHtml (ppoRolloutTrainLog (α := α) (obsShape := obsShape) (nActions := nActions)
    (horizon := horizon) gamma lam r)

/-!
## Commands
-/

syntax (name := ppoRolloutViewCmd) "#ppo_rollout_view " term ", " term ", " term : command

macro "#ppo_rollout_view " gamma:term ", " lam:term ", " r:term : command =>
  Lean.TSyntax.mkInfoCanonical <$> `(#html (ppoRolloutHtml $gamma $lam $r))

end PPO
end RL

end
end NN.Widgets
