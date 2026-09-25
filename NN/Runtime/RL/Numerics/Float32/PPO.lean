/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Numerics.Float32.Returns

/-!
# Checked Float32 PPO Objective Helpers

PPO is usually run with ordinary host floats, but these helpers make the scalar objective pieces
executable under the explicit `ExecFloat.Binary 8 23` model and reject non-finite intermediates.
They are useful for regression tests, debugging numerically fragile runs, and connecting runtime
checks to proof layer finite hypotheses.

Reference: Schulman et al., "Proximal Policy Optimization Algorithms" (2017).
-/

@[expose] public section

namespace Runtime
namespace RL
namespace Numerics
namespace Float32

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Spec.RL
open FloatLib.Floats (ExecFloat)


/--
Checked importance ratio `exp(newLogProb - oldLogProb)`, specialized to `ExecFloat.Binary 8 23`.

This is the float32-semantics variant of `Runtime.RL.PolicyGradient.importanceRatio`.
-/
def importanceRatioChecked (newLogProb oldLogProb : ExecFloat.Binary 8 23) :
    Except String (ExecFloat.Binary 8 23) := do
  let diff ← checkedSub "importanceRatio/sub(newLogProb,oldLogProb)" newLogProb oldLogProb
  checkedExp "importanceRatio/exp(diff)" diff

/--
Checked PPO clipped surrogate objective from a precomputed importance ratio:

`min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)`.

This avoids re-doing the softmax/log-prob computation when you already have ratios.

Reference:
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017):
  https://arxiv.org/abs/1707.06347
-/
def ppoClippedObjectiveFromRatioChecked
    (ratio advantage clipEps : ExecFloat.Binary 8 23) :
    Except String (ExecFloat.Binary 8 23) := do
  let one : ExecFloat.Binary 8 23 := (1 : ExecFloat.Binary 8 23)
  let lo ← checkedSub "ppoClip/sub(1,eps)" one clipEps
  let hi ← checkedAdd "ppoClip/add(1,eps)" one clipEps
  let clippedLo ← checkedMax "ppoClip/max(lo,ratio)" lo ratio
  let clippedRatio ← checkedMin "ppoClip/min(hi,clippedLo)" hi clippedLo
  let unclipped ← checkedMul "ppoClip/mul(ratio,advantage)" ratio advantage
  let clipped ← checkedMul "ppoClip/mul(clippedRatio,advantage)" clippedRatio advantage
  checkedMin "ppoClip/min(unclipped,clipped)" unclipped clipped


end Float32
end Numerics
end RL
end Runtime
