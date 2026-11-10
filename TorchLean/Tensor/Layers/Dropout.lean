/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Elementwise

/-!
# Dropout

Inverted dropout with drop probability `p` keeps an entry with probability `1 - p` and divides it
by `1 - p`. The keep mask is an argument, so the specification is deterministic: `Op.dropout`
derives it from a uniform sample `u` on `[0, 1]` as `u ≥ p` (rounding in binary32 can produce the
upper endpoint `1`). For a uniform `u` the mask keeps an entry with probability exactly `1 - p`,
every entry at `p = 0` (even where `u = 0`) and none at a NaN `p`.

- Evaluation mode is the identity.
- `Dropout.masked p mask x` is training mode with the Boolean keep mask `mask`. A kept entry is
  `x / (1 - p)`, one division by the difference `1 - p`, which a floating-point scalar forms
  exactly for `1/2 ≤ p ≤ 1` and rounds once otherwise. A dropped entry is `x * 0`, a NaN at an
  infinite or NaN `x` and `-0` at a negative one. Every entry is `+0` when `p ≥ 1`, so `p = 1`
  keeps nothing. A NaN `p` is not at least one, so the entries the mask keeps are NaN, and the
  layers refuse a NaN `p` and a `p` outside `[0, 1]`.
- The backward map sends the cotangent `g` to `Dropout.masked p mask g`. Neither `p` nor the mask
  receives a gradient.

PyTorch's training-mode dropout multiplies a kept entry by the rounded scale `1 / (1 - p)`, which
adds a second rounding. In evaluation mode both are the identity.

We wrote this by hand from Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks
from Overfitting", JMLR 15(56):1929-1958, 2014. Their `p` is the probability of keeping a unit, and
Sec. 10 notes that scaling the kept activations by `1/p` in training is equivalent to scaling the
weights by `p` at test time, once the learning rate and the weight initialization are scaled to
match. Our `p` is the drop probability of `torch.nn.Dropout` (PyTorch 2.8 docs), so the training
scale is `1 / (1 - p)` and evaluation changes nothing.
-/

@[expose] public section


namespace TorchLean

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

/-- Training-mode dropout with drop probability `p` and the Boolean keep mask `mask`. A kept entry
is `x / (1 - p)`, one division, and a dropped entry is `x * 0`. When `p ≥ 1` the result is the
constant `+0` tensor, so an infinite or NaN `x` gives no NaN there. -/
def Dropout.masked {s : Shape} (p : α) (mask : Tensor Bool s) (x : Tensor α s) : Tensor α s :=
  if Context.geBool p 1 then Tensor.full s 0 else
    zipWith (fun xi keep => if keep then xi / ((1 : α) - p) else xi * 0) x mask

end TorchLean
