/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Models.Attention.CausalMask
public import Mathlib.Analysis.SpecialFunctions.Log.ERealExp

/-!
# BugZoo: attention-mask semantics

Attention code has its own failure modes: mask polarity, head reshaping, Q/K/V layout, and KV-cache
mismatches are easy to get wrong and hard to notice from accuracy tests alone. This file focuses on the
mask part, because TorchLean already has a precise theorem stack for it.

Here is the bug-shaped PyTorch pattern we want to rule out:

```python
# Wrong for causal attention: the polarity is flipped, so future tokens are allowed.
scores = q @ k.transpose(-2, -1) / math.sqrt(d)
mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
weights = torch.softmax(scores.masked_fill(mask == False, -torch.inf), dim=-1)
```

The intended PyTorch version uses true negative infinity on blocked future entries:

```python
future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
weights = torch.softmax(scores.masked_fill(future, -torch.inf), dim=-1)
assert weights[i, j] == 0.0 for all j > i
```

Lean's ordinary `ℝ` does not contain a literal `-∞`, but mathlib does provide extended reals
`EReal`, where `⊥` is negative infinity and `EReal.exp ⊥ = 0`. We record that exact `-∞` fact
first. TorchLean's ordinary tensor softmax then uses the computationally convenient equivalent:
blocked logits get zero numerator before normalization. Both views lead to the exact theorem below.

References:
- Vaswani et al., “Attention Is All You Need”, NeurIPS 2017.
  https://arxiv.org/abs/1706.03762
- PyTorch `scaled_dot_product_attention` documentation, for the runtime-style mask interface:
  https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- PyTorch issue #99282, where `MultiheadAttention(is_causal=True)` was reported ignored when
  `need_weights=True`:
  https://github.com/pytorch/pytorch/issues/99282
- PyTorch issue #160064, where fully masked attention heads were reported to produce NaNs when
  attention weights were requested:
  https://github.com/pytorch/pytorch/issues/160064
-/

@[expose] public section

namespace NN.Examples.BugZoo.AttentionMask

open Spec
open Spec.Tensor

/-- Exact extended-real masked logit: allowed entries keep their real score, blocked entries are
literal `-∞` (`⊥ : EReal`). -/
noncomputable def exactMaskedLogit (score : ℝ) (allowed : Bool) : EReal :=
  if allowed then (score : EReal) else ⊥

/-- Blocking a logit really means assigning `-∞` in the extended-real presentation. -/
@[simp] theorem exactMaskedLogit_blocked (score : ℝ) :
    exactMaskedLogit score false = (⊥ : EReal) := by
  rfl

/-- The key `-∞` softmax fact: `exp(-∞) = 0`. -/
@[simp] theorem exactMaskedLogit_blocked_exp_zero (score : ℝ) :
    EReal.exp (exactMaskedLogit score false) = 0 := by
  simp [exactMaskedLogit]

/-- Exact extended-real causal masking of one score-matrix coordinate. -/
noncomputable def exactCausalMaskedScore {n : Nat}
    (scores : Spec.Tensor ℝ (.dim n (.dim n .scalar))) (i j : Fin n) : EReal :=
  exactMaskedLogit (Spec.get2 scores i j) (Spec.get2 (Spec.causalMask n) i j)

/--
For a strict-future position, exact causal masking assigns literal `-∞`.

This is the formal version of the PyTorch operation
`scores.masked_fill(future, -torch.inf)` at one matrix coordinate.
-/
theorem exactCausalMaskedScore_future_eq_bot
    {n : Nat}
    (scores : Spec.Tensor ℝ (.dim n (.dim n .scalar)))
    (i j : Fin n) (hij : i.val < j.val) :
    exactCausalMaskedScore scores i j = (⊥ : EReal) := by
  simp [exactCausalMaskedScore, exactMaskedLogit,
    NN.Proofs.Models.Attention.causalMask_blocks_future i j hij]

/--
Therefore, the strict-future numerator is exactly zero.

This is why TorchLean's attention spec writes this zero numerator directly.
-/
theorem exactCausalMaskedScore_future_exp_zero
    {n : Nat}
    (scores : Spec.Tensor ℝ (.dim n (.dim n .scalar)))
    (i j : Fin n) (hij : i.val < j.val) :
    EReal.exp (exactCausalMaskedScore scores i j) = 0 := by
  simp [exactCausalMaskedScore_future_eq_bot scores i j hij]

/--
True-`-∞` causal attention gets the exact zero-weight theorem.

This is the statement we want for formal output-causality arguments: every strict-future key has
zero attention mass for the current query row. In TorchLean this is represented by
`hardMaskedSoftmaxSpec`, not by a finite real sentinel treated as `-∞`.
-/
theorem trueInfinityMask_future_attention_weight_zero
    {n : Nat}
    (scores : Spec.Tensor ℝ (.dim n (.dim n .scalar)))
    (i j : Fin n) (hij : i.val < j.val) :
    Spec.get2 (Spec.hardMaskedSoftmaxSpec scores (Spec.causalMask n)) i j = 0 :=
  NN.Proofs.Models.Attention.hardMaskedSoftmaxSpec_causal_future_zero scores i j hij

end NN.Examples.BugZoo.AttentionMask
