/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor
public import NN.Proofs

/-!
# Quickstart: Proving Small TorchLean Facts

Many TorchLean guarantees are ordinary Lean theorems. These examples cover:

- compile-time guarantees from shape-indexed tensor types, and
- ordinary mathematical lemmas about the public API.

The deeper proof libraries live under `NN.Proofs.*`, `NN.Verification.*`, and `NN.MLTheory.*`.
-/

@[expose] public section

namespace NN.Examples.Quickstart.Proofs

open TorchLean

/--
A tensor's shape is part of its type.

If this definition compiles, Lean has already checked that the literal has exactly two entries and
therefore has type `Tensor Float [2]`. The commented shape mismatch below is the kind of bug Lean
catches before runtime:

```lean
-- def badTensor : Tensor Float [3] := [1.0, 2.0]
```
-/
def twoTensor : Tensor Float [2] :=
  [1.0, 2.0]

/-- ReLU fixes every nonnegative real number. -/
theorem relu_eq_self_of_nonnegative (x : ℝ) (hx : 0 ≤ x) :
    Activation.Math.reluSpec x = x := by
  simpa only [Activation.Math.reluSpec_eq_max] using max_eq_left hx

/-- ReLU clamps nonpositive real inputs to zero. -/
theorem relu_eq_zero_of_nonpositive (x : ℝ) (hx : x ≤ 0) :
    Activation.Math.reluSpec x = 0 := by
  simpa only [Activation.Math.reluSpec_eq_max] using max_eq_right hx

example : Activation.Math.reluSpec (3 : ℝ) = 3 := by
  exact relu_eq_self_of_nonnegative 3 (by norm_num)

example : Activation.Math.reluSpec (-2 : ℝ) = 0 := by
  exact relu_eq_zero_of_nonpositive (-2) (by norm_num)

end NN.Examples.Quickstart.Proofs
