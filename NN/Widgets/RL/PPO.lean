/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.PPO.Rollout
public import NN.Runtime.Training.Log
-- We compute GAE/returns in a widget (meta) context, so the RL core must be available to meta code.
public meta import NN.Runtime.RL.PPO.Rollout
public meta import NN.Widgets.Runtime.Training
public import NN.Widgets.Runtime.Training
import ProofWidgets.Component.HtmlDisplay

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
- `#ppo_rollout_view`: command form for inspecting rollout summaries.
-/

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

namespace NN.Widgets

public meta section

open scoped ProofWidgets.Jsx

open Spec TorchLean
open TorchLean.Tensor

open Runtime.Training

namespace RL
namespace PPO

/--
Lossy conversion from a scalar backend `α` to Lean's `Float` for visualization.

This is a small widget-only typeclass. If you define your own scalar backend for RL, add an
instance here (or locally in your project) to enable `#ppo_rollout_view`.
-/
class ToVizFloat (α : Type) where
  toVizFloat : α → Float

instance : ToVizFloat Float := ⟨fun x => x⟩
instance : ToVizFloat (ExecFloat.Binary 8 23) :=
  ⟨(fun x => ExecFloat.Binary.toFloat (ExecFloat.Binary.ofModel (Model.cast FloatFormat.binary32
    FloatFormat.binary64 (ExecFloat.Binary.toModel x))))⟩

/--
Convert a length-`n` scalar tensor to an `Array Float` for plotting.

Each entry is converted with the plotting scalar interface.
-/
private def tensorToFloatArray {α : Type} [TorchLean.Storage α] [ToVizFloat α] {n : Nat}
    (tensor : Tensor α [n]) : Array Float :=
  Array.ofFn fun i : Fin n => ToVizFloat.toVizFloat (tensor.getScalar i)

/-- Build a `TrainLog` containing reward/return/advantage curves for a fixed-horizon PPO rollout. -/
def ppoRolloutTrainLog {α : Type} [TorchLean.Storage α] [Context α] [ToVizFloat α]
    {obsShape : Shape} {nActions horizon : Nat}
    (gamma lam : α)
    (r : Runtime.RL.PPO.Rollout α obsShape nActions horizon) :
    TrainLog :=
  let stepAt (index : Fin horizon) :=
    r.steps[index.val]'(by simp [r.steps_size_eq_horizon])
  let rewards : Tensor α [horizon] := Tensor.ofFn (fun i => (stepAt i).reward)
  let values : Tensor α [horizon] := Tensor.ofFn (fun i => (stepAt i).value)

  let advRaw := r.generalizedAdvantages gamma lam
  let returns :=
    Runtime.RL.Core.returnsFromAdvantages (α := α) (n := horizon) advRaw values

  let rewardsF : Array Float := tensorToFloatArray rewards
  let returnsF : Array Float := tensorToFloatArray (α := α) (n := horizon) returns
  let advF : Array Float := tensorToFloatArray (α := α) (n := horizon) advRaw

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
def ppoRolloutHtml {α : Type} [TorchLean.Storage α] [Context α] [ToVizFloat α]
    {obsShape : Shape} {nActions horizon : Nat}
    (gamma lam : α)
    (r : Runtime.RL.PPO.Rollout α obsShape nActions horizon) :
    ProofWidgets.Html :=
  trainLogHtml (ppoRolloutTrainLog (α := α) (obsShape := obsShape) (nActions := nActions)
    (horizon := horizon) gamma lam r)

/-!
## Commands
-/

syntax (name := ppoRolloutViewCmd) "#ppo_rollout_view " term ", " term ", " term : command

macro "#ppo_rollout_view " gamma:term ", " lam:term ", " r:term : command =>
  UI.canonicalCommand <$> `(#html (ppoRolloutHtml $gamma $lam $r))

end PPO
end RL

end
end NN.Widgets
