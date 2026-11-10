/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import TorchLean.Tensor.Einops.Elab.Einsum
public meta import TorchLean.Tensor.Einops.Elab.Pack
public meta import TorchLean.Tensor.Literal
public meta import TorchLean.Tensor.Einops.Elab.Transform
public meta import TorchLean.Tensor.Tactic.Einops.Report
public import TorchLean.Verify.Certificate.CROWNQuery

/-!
# Proving an MLP keeps its predicted class

We build a two-layer ReLU network with one input and two output scores. Its hidden values are
`relu(x)` and `relu(-x)`, and its scores are `1 + relu(x)` and `relu(-x)`.

We ask CROWN to prove that the first score exceeds the second by at least `3/4` whenever
`|x| ≤ 1/4`. Both ReLUs change branch inside this interval. The theorem covers every real input
in the interval, which no finite set of sampled inputs can do.

The weights are defined once. `network.toCertificate` holds its layers without the slopes, and
`network.toCertificate.model.eval network.toCertificate.params` runs those layers with the
input's scalar type. For example, in the editor:

```lean
-- scores: 1.25 and 0
#eval network.toCertificate.model.eval network.toCertificate.params ([0.25] : Tensor Float [1])
-- scores: 1 and 1/4
open scoped TorchLean.RationalAlgebraic in
#eval network.toCertificate.model.eval network.toCertificate.params ([-1 / 4] : Tensor ℚ [1])
```

`query` uses the same network to compute rational bounds. `Query.check_sound` turns a successful
check into the requested inequality over real numbers.
`Network.realTensor_forward` proves that exact rational execution agrees with that real model.
Bounding floating-point execution also needs rounding error bounds.
-/

@[expose] public section

namespace TorchLean.Examples.RobustMLP

open Certificate.CROWNQuery
open CROWN.Graph (encloses_iff_getScalar)

open Network in
/-- A `1 → 2 → 2` MLP: linear layer, ReLU, then linear layer. -/
def network : Network 1 2 :=
  comp (linear { weights := [[1], [-1]], bias := [0, 0] })
    -- The zeros are CROWN's lower-bound slopes, and the forward pass is a plain ReLU.
    (comp (relu [0, 0])
      (linear { weights := [[1, 0], [0, 1]], bias := [1, 0] }))

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
    (3 : ℝ) / 4 ≤ (network.toCertificate.model.eval network.toCertificate.params x).getScalar 0 -
      (network.toCertificate.model.eval network.toCertificate.params x).getScalar 1 := by
  -- Lean checks CROWN's rational computation, then applies its general soundness theorem.
  have h := query.check_sound (by decide +kernel) x ((encloses_iff_getScalar _ _ x).mpr (by
    intro i
    fin_cases i
    simpa [query, neg_div] using (abs_le.mp hx))) ⟨0, by decide⟩
  -- Unpack the query's affine inequality into the two output scores.
  have hmargin : -(network.toCertificate.model.eval network.toCertificate.params x).getScalar 0 +
      (network.toCertificate.model.eval network.toCertificate.params x).getScalar 1 +
        (3 : ℝ) / 4 ≤ 0 := by
    erw [Tensor.add_def] at h
    simpa [query, Tensor.getScalar_contract_matVec_eq_sum,
      Fin.sum_univ_two, Tensor.matrix, Tensor.get2] using h
  linarith

end TorchLean.Examples.RobustMLP
