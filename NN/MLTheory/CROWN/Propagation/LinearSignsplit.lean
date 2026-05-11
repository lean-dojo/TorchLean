/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Core

/-!
# Sign-splitting linear bounds (IBP helper)

For a linear layer `y = W x + b` with input interval `x ∈ [lo, hi]`,
we can compute output bounds using the standard sign-splitting trick:

- `W⁺ = max(W, 0)` (elementwise)
- `W⁻ = min(W, 0)` (elementwise)

Then:
- `y_lo = W⁺·lo + W⁻·hi + b_lo`
- `y_hi = W⁺·hi + W⁻·lo + b_hi`

This is algebraically equivalent to the per-weight min/max rule, but it makes it
easy to cache `W⁺/W⁻` and matches the common LiRPA/CROWN implementation style.
-/

@[expose] public section


namespace NN.MLTheory.CROWN

open _root_.Spec
open _root_.Spec.Tensor

variable {α : Type} [Context α]

namespace IBP

/-- Positive part of a weight matrix: `W⁺ = max(W, 0)` (elementwise). -/
def matPos {m n : Nat}
    (W : Tensor α (.dim m (.dim n .scalar))) : Tensor α (.dim m (.dim n .scalar)) :=
  match W with
  | .dim rows =>
      Tensor.dim (fun i =>
        match rows i with
        | .dim cols =>
            Tensor.dim (fun j =>
              match cols j with
              | .scalar w => Tensor.scalar (if w > 0 then w else 0)))

/-- Negative part of a weight matrix: `W⁻ = min(W, 0)` (elementwise). -/
def matNeg {m n : Nat}
    (W : Tensor α (.dim m (.dim n .scalar))) : Tensor α (.dim m (.dim n .scalar)) :=
  match W with
  | .dim rows =>
      Tensor.dim (fun i =>
        match rows i with
        | .dim cols =>
            Tensor.dim (fun j =>
              match cols j with
              | .scalar w => Tensor.scalar (if w > 0 then 0 else w)))

/-- Linear IBP bounds computed via sign-splitting (`W⁺/W⁻`). -/
def linearSignSplit {m n : Nat}
    (W : Tensor α (.dim m (.dim n .scalar)))
    (xB : Box α (.dim n .scalar))
    (bB : Box α (.dim m .scalar)) : Box α (.dim m .scalar) :=
  let Wpos := matPos (α := α) (m := m) (n := n) W
  let Wneg := matNeg (α := α) (m := m) (n := n) W
  { lo :=
      Tensor.addSpec
        (Tensor.addSpec
          (Spec.matVecMulSpec (α := α) Wpos xB.lo)
          (Spec.matVecMulSpec (α := α) Wneg xB.hi))
        bB.lo
    hi :=
      Tensor.addSpec
        (Tensor.addSpec
          (Spec.matVecMulSpec (α := α) Wpos xB.hi)
          (Spec.matVecMulSpec (α := α) Wneg xB.lo))
        bB.hi }

end IBP

end NN.MLTheory.CROWN
