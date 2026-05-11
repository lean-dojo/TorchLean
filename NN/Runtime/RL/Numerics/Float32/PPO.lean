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
executable under the explicit `IEEE32Exec` model and reject non-finite intermediates. They are useful
for regression tests, debugging numerically fragile runs, and connecting runtime checks to proof-side
finite hypotheses.

Reference: Schulman et al., "Proximal Policy Optimization Algorithms" (2017).
-/

@[expose] public section

namespace Runtime
namespace RL
namespace Numerics
namespace Float32

open Spec
open Tensor
open Spec.RL

open TorchLean.Floats
open TorchLean.Floats.IEEE754

/--
Checked importance ratio `exp(newLogProb - oldLogProb)`, specialized to `IEEE32Exec`.

This is the float32-semantics variant of `Runtime.RL.PolicyGradient.importanceRatio`.
-/
def importanceRatioIEEE32ExecChecked (newLogProb oldLogProb : Float32Exec) :
    Except String Float32Exec := do
  let diff ← checkedSub "importanceRatio/sub(newLogProb,oldLogProb)" newLogProb oldLogProb
  checkedExp "importanceRatio/exp(diff)" diff

/--
Checked PPO clipped surrogate objective from a precomputed importance ratio:

`min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)`.

This avoids re-doing the softmax/log-prob computation when you already have ratios.

Reference:
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017): https://arxiv.org/abs/1707.06347
-/
def ppoClippedObjectiveFromRatioIEEE32ExecChecked
    (ratio advantage clipEps : Float32Exec) :
    Except String Float32Exec := do
  let one : Float32Exec := (1 : Float32Exec)
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
