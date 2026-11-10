/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL.PPO.Rollout
public import TorchLean.NN.Training
-- The widget computes GAE and returns in meta code, so the RL core is also a meta import.
public meta import TorchLean.NN.RL.PPO.Rollout
public meta import TorchLean.NN.Widget.Training
public import TorchLean.NN.Widget.Training
import ProofWidgets.Component.HtmlDisplay

/-!
# PPO Rollout Viewer

`#ppo_rollout_view gamma, lam, r` plots three curves over the steps of a PPO rollout `r`:

- `reward_t` (environment rewards),
- `return_t` (lambda-returns, the GAE advantages plus the value estimates),
- `advantage_t` (GAE(λ) advantages).

The curves use the shared training-log viewer, and this module only prepares the rollout series,
with `gamma` and `lambda` as notes. Entries reach `Float` through `ElementCast α Float`, which a
custom scalar backend provides.

References:
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
  (2015): https://arxiv.org/abs/1506.02438
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017):
  https://arxiv.org/abs/1707.06347

## Main definitions

- `rolloutTrainLog`: converts rollout tensors into `TrainLog` series.
- `rolloutHtml`: delegates to the generic training viewer.
- `#ppo_rollout_view`: command form for inspecting rollout summaries.
-/

open FloatLib.Floats (ExecFloat)

namespace TorchLean.Widget

public meta section

open scoped ProofWidgets.Jsx

open TorchLean.Tensor

open Training

namespace RL
namespace PPO

open FloatLib.Floats.Formats.BinaryInterchange in
/-- A configured value in its default code as Lean's binary64 `Float`, by
`ExecFloat.Binary.toFloat`: rounded to nearest-even where binary64 cannot hold it. -/
instance configuredBinaryFloatCast {format : FloatFormat} {plan : Configured.StoragePlan format} :
    ElementCast (ExecFloat (Configured.Family format (Configured.Code plan) plan)) Float :=
  ⟨ExecFloat.Binary.toFloat⟩

/--
Convert a length-`n` scalar tensor to an `Array Float` for plotting.

Each entry is converted with `ElementCast α Float`.
-/
private def tensorToFloatArray {α : Type} [Storage α] [ElementCast α Float] {n : Nat}
    (tensor : Tensor α [n]) : Array Float :=
  Array.ofFn fun i : Fin n => ElementCast.cast (tensor.getScalar i)

/-- Build a `TrainLog` with reward, return and advantage series for a fixed-horizon PPO rollout,
with the GAE(λ) advantages computed from the discount `gamma` and the parameter `lam`. -/
def rolloutTrainLog {α : Type} [Storage α] [Context α] [ElementCast α Float]
    {obsShape : Shape} {nActions horizon : Nat}
    (gamma lam : α)
    (r : RL.PPO.Rollout α obsShape nActions horizon) :
    TrainLog :=
  let stepAt (index : Fin horizon) :=
    r.steps[index.val]'(by simp [r.steps_size_eq_horizon])
  let rewards : Tensor α [horizon] := Tensor.ofFn (fun i => (stepAt i).reward)
  let values : Tensor α [horizon] := Tensor.ofFn (fun i => (stepAt i).value)

  let advRaw := r.generalizedAdvantages gamma lam
  let returns :=
    RL.returnsFromAdvantages (α := α) (n := horizon) advRaw values

  let rewardsF : Array Float := tensorToFloatArray rewards
  let returnsF : Array Float := tensorToFloatArray (α := α) (n := horizon) returns
  let advF : Array Float := tensorToFloatArray (α := α) (n := horizon) advRaw

  let notes : Array String :=
    #[
      s!"gamma={(ElementCast.cast gamma : Float)}",
      s!"lambda={(ElementCast.cast lam : Float)}"
    ]
  let series : Array Series :=
    #[
      { name := "reward", values := rewardsF, color := "#f28e2b" },
      { name := "return", values := returnsF, color := "#4e79a7" },
      { name := "advantage", values := advF, color := "#e15759" }
    ]
  { title := "PPO rollout", series := series, notes := notes }

/-- Render a PPO rollout in the training-log viewer: reward, return and advantage curves and a
table of the last steps. -/
def rolloutHtml {α : Type} [Storage α] [Context α] [ElementCast α Float]
    {obsShape : Shape} {nActions horizon : Nat}
    (gamma lam : α)
    (r : RL.PPO.Rollout α obsShape nActions horizon) :
    ProofWidgets.Html :=
  trainLogHtml (rolloutTrainLog (α := α) (obsShape := obsShape) (nActions := nActions)
    (horizon := horizon) gamma lam r)

/-!
## Commands
-/

/-- `#ppo_rollout_view gamma, lam, r` shows the `rolloutHtml` view of the PPO rollout `r` with
discount `gamma` and GAE parameter `lam`. -/
macro "#ppo_rollout_view " gamma:term ", " lam:term ", " r:term : command =>
  UI.canonicalCommand <$> `(#html (rolloutHtml $gamma $lam $r))

end PPO
end RL

end
end TorchLean.Widget
