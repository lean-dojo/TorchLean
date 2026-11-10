/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL.Numerics.Returns

/-!
# Checked PPO Objective Helpers

PPO usually runs on host floats. These helpers compute the scalar pieces of its objective in a
configured FloatLib binary format, binary32 `ExecFloat.Binary 8 23` among them, and reject every
non-finite intermediate. We use them to debug numerically fragile runs and to connect runtime
checks to the finiteness hypotheses of the proof layer.

We wrote both by hand. `ppoClippedObjective` is the checked form of `RL.PPO.objective`, one summand
of $L^{CLIP}$ from Section 3 of the PPO paper, with its clamp spelled out as `max` against
`1 - clipEps` and then `min` against `1 + clipEps`.

## Main declarations

* `importanceRatio`: the checked importance ratio `exp(newLogProb - oldLogProb)`.
* `ppoClippedObjective`: the checked clipped surrogate objective.

Reference: Schulman et al., "Proximal Policy Optimization Algorithms" (2017), arXiv:1707.06347,
Section 3, eqs. (6) and (7).
-/

@[expose] public section

namespace TorchLean
namespace RL
namespace Checked

open TorchLean.Tensor
open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/--
The importance ratio `exp(newLogProb - oldLogProb)` with the difference and the exponential
checked, the checked form of `TorchLean.RL.PolicyGradient.ratio`. In binary32 it fails once the
log-ratio passes about 88.72, where the exponential overflows, while a log-ratio of -104 rounds to
0 and passes.
-/
def importanceRatio
    [ExecFloat.Backend.PolicyFor (Configured.Family format code plan)]
    [ExecFloat.Sub (Configured.Family format code plan)]
    (newLogProb oldLogProb : Value) :
    Except String Value := do
  let diff ← sub "importanceRatio/sub(newLogProb,oldLogProb)" newLogProb oldLogProb
  exp "importanceRatio/exp(diff)" diff

/--
Checked PPO clipped surrogate objective from a precomputed importance ratio:

`min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)`.

Callers that already have the ratios skip the softmax and log-probability computation. Every
intermediate, from `1 - ε` to the final minimum, is checked for finiteness.

Reference:
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017), eq. (7):
  https://arxiv.org/abs/1707.06347
-/
def ppoClippedObjective
    [ExecFloat.Backend.PolicyFor (Configured.Family format code plan)]
    [ExecFloat.Add (Configured.Family format code plan)]
    [ExecFloat.Mul (Configured.Family format code plan)]
    [ExecFloat.Sub (Configured.Family format code plan)]
    (ratio advantage clipEps : Value) :
    Except String Value := do
  let one : Value := (1 : Value)
  let lo ← sub "ppoClip/sub(1,eps)" one clipEps
  let hi ← add "ppoClip/add(1,eps)" one clipEps
  let clippedLo ← max "ppoClip/max(lo,ratio)" lo ratio
  let clippedRatio ← min "ppoClip/min(hi,clippedLo)" hi clippedLo
  let unclipped ← mul "ppoClip/mul(ratio,advantage)" ratio advantage
  let clipped ← mul "ppoClip/mul(clippedRatio,advantage)" clippedRatio advantage
  min "ppoClip/min(unclipped,clipped)" unclipped clipped


end Checked
end RL
end TorchLean
