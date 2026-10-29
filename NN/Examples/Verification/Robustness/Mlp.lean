/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor
public import NN.Verification.Cert.CROWNQuery
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.Linarith

/-!
# Proving an MLP keeps its predicted class

We build a two-layer ReLU network with one input and two output scores. Its hidden values are
`relu(x)` and `relu(-x)`, and its scores are `1 + relu(x)` and `relu(-x)`.

We ask CROWN to prove that the first score exceeds the second by at least `3/4` whenever
`|x| ≤ 1/4`. Both ReLUs change branch inside this interval. The theorem covers every real input
in the interval; it does not rely on sampling inputs.

The weights are defined once. `model.forward` runs those layers with the input's scalar type.
For example, in the editor:

```lean
#eval model.forward ([0.25] : Tensor Float [1]) -- scores: 1.25 and 0
#eval model.forward ([-1 / 4] : Tensor ℚ [1]) -- scores: 1 and 1/4
```

`query` uses the same network to compute rational bounds. The existing `Query.check_sound`
theorem turns a successful check into the requested inequality over real numbers.
`Network.realTensor_forward` proves that exact rational execution agrees with that real model.
Bounding floating-point execution additionally needs rounding error bounds.
-/

@[expose] public section

namespace NN.Examples.Verification.Robustness.Mlp

open TorchLean
open NN.Verification.Cert.CROWNQuery
open NN.Verification.Cert.RationalReflection

open Network in
/-- A `1 → 2 → 2` MLP: linear layer, ReLU, then linear layer. -/
def network : Network 1 2 :=
  comp (linear { weights := [[1], [-1]], bias := [0, 0] })
    -- These zeros select CROWN's lower-bound slopes; the activation is ordinary ReLU.
    (comp (relu [0, 0])
      (linear { weights := [[1, 0], [0, 1]], bias := [1, 0] }))

/-- Run the same layers and weights with the input's scalar type, or use `ℝ` in a theorem. -/
def model {α : Type} [Storage α] [Add α] [Mul α] [Zero α] [Max α] [BEq α]
    [RatCast α] : Spec.Module.Chain α [1] [2] := network.model

/-- Require `score₁ - score₀ + 3/4 ≤ 0` throughout the input interval `[-1/4, 1/4]`. -/
def query : Query 1 2 where
  network := network
  input := { lo := [-1 / 4], hi := [1 / 4] }
  numConstraints := 1
  inequalities := {
    weights := Tensor.matrix fun _ j => if j = 0 then -1 else 1
    bias := Tensor.full [1] (3 / 4) }
  strict := false

/-- Every input within `1/4` of zero keeps class zero ahead by a margin of at least `3/4`. -/
theorem robust (x : Tensor ℝ [1]) (hx : |x.getScalar 0| ≤ 1 / 4) :
    (3 : ℝ) / 4 ≤ (model.forward x).getScalar 0 - (model.forward x).getScalar 1 := by
  -- Lean checks CROWN's rational computation, then applies its general soundness theorem.
  have h := query.check_sound (by decide +kernel) x (by
    intro i
    fin_cases i
    simpa [query, neg_div] using (abs_le.mp hx)) ⟨0, by decide⟩
  -- Unpack the query's affine inequality into the two output scores.
  have hmargin : -(model.forward x).getScalar 0 + (model.forward x).getScalar 1 +
      (3 : ℝ) / 4 ≤ 0 := by
    simpa [query, model, Tensor.addSpec, Proofs.TensorAlgebra.getScalar_mat_vec_mul_spec,
      Fin.sum_univ_two, Tensor.matrix, Spec.get2] using h
  linarith

end NN.Examples.Verification.Robustness.Mlp
