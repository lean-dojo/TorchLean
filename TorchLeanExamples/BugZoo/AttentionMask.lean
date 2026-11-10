/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Attention.CausalMask
public import Mathlib.Analysis.SpecialFunctions.Log.ERealExp

/-!
# BugZoo: attention-mask semantics

Attention code has its own failure modes. Mask polarity, head reshaping, Q/K/V layout and KV-cache
mismatches are easy to get wrong and hard to notice from accuracy tests alone. The file takes the
mask here, because TorchLean's attention proofs already state what a causal mask does.

Here is the bug-shaped PyTorch pattern to rule out:

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
assert torch.all(weights[future] == 0.0)
```

Lean's `ℝ` has no $-\infty$. Mathlib's extended reals `EReal` have one, `⊥`, with `EReal.exp ⊥ = 0`,
and the file records that fact first. TorchLean's tensor softmax uses an equivalent that is easier
to compute with: a blocked logit gets a zero numerator before normalization. Both views lead to
`TorchLean.Softmax.hardMasked_causalMask_eq_zero`, restated below.

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

namespace TorchLean.Examples.BugZoo.AttentionMask

open TorchLean.Tensor

/-- The masked logit in `EReal`: an allowed entry keeps its real score, and a blocked entry is
$-\infty$ (`⊥ : EReal`). -/
noncomputable def logit (score : ℝ) (allowed : Bool) : EReal :=
  if allowed then (score : EReal) else ⊥

/-- A blocked logit is `⊥`, the extended-real $-\infty$. -/
@[simp] theorem logit_false (score : ℝ) :
    logit score false = (⊥ : EReal) := by
  rfl

/-- The exponential of a blocked logit is zero, the softmax fact $\exp(-\infty)=0$. -/
@[simp] theorem exp_logit_false (score : ℝ) :
    EReal.exp (logit score false) = 0 := by
  simp [logit]

/-- Entry `(i, j)` of the causally masked score matrix, as an extended real. -/
noncomputable def causalScore {n : Nat}
    (scores : Tensor ℝ [n, n]) (i j : Fin n) : EReal :=
  logit scores[(i, j)] (Attention.causalMask n)[(i, j)]

/--
A strict-future entry (`i < j`) of the causally masked scores is $-\infty$.

At one matrix coordinate this is the PyTorch operation `scores.masked_fill(future, -torch.inf)`.
-/
theorem causalScore_eq_bot
    {n : Nat}
    (scores : Tensor ℝ [n, n])
    (i j : Fin n) (hij : i.val < j.val) :
    causalScore scores i j = (⊥ : EReal) := by
  change logit scores[(i, j)] (Tensor.get2 (Attention.causalMask n) i j) = ⊥
  simp [logit, Attention.get2_causalMask_eq_false_of_lt i j hij]

/--
The exponential of a strict-future masked score is zero, so that entry adds nothing to the
softmax numerator.

TorchLean's attention spec writes this numerator as a literal `0`.
-/
theorem exp_causalScore
    {n : Nat}
    (scores : Tensor ℝ [n, n])
    (i j : Fin n) (hij : i.val < j.val) :
    EReal.exp (causalScore scores i j) = 0 := by
  simp [causalScore_eq_bot scores i j hij]

/-- Under the causal mask, `Softmax.hardMasked` gives every strict-future key zero attention
weight. TorchLean represents true $-\infty$ masking this way rather than by a finite sentinel. -/
example {n : Nat} (scores : Tensor ℝ [n, n]) (i j : Fin n) (hij : i.val < j.val) :
    (Softmax.hardMasked scores (Attention.causalMask n))[(i, j)] = 0 :=
  Softmax.hardMasked_causalMask_eq_zero scores i j hij

end TorchLean.Examples.BugZoo.AttentionMask
