/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.MLP
public import TorchLean.Verify.RuntimeApprox.NF.MLP

/-!
# CROWN/IBP boxes for rounded execution

The CROWN/IBP development proves that a real network output lies in a box. This module widens the
box by a forward-error bound, so that the widened box contains the rounded output as well.

The widening lemma `CROWN.Box.contains_inflate_of_approx` holds for every runtime scalar type `α`
and interpretation `toSpec : α → ℝ`. If the real output lies in `B` and the runtime output
is within `eps` of it in the infinity norm, the interpreted runtime output lies in
`CROWN.Box.inflate B eps`. `reluTwoLayerMLP_ibp_sound` applies it to the IBP box of the two-layer
ReLU MLP over `NF β fexp rnd`, widened by `ibpReLUTwoLayerMLPErrorBudget`.
-/

@[expose] public section

namespace TorchLean

open TorchLean.Tensor

noncomputable section

/-! ## Margins of a real interval -/

/-- If `l ≤ y ≤ u` and `|z - y| ≤ eps`, then `l - eps ≤ z ≤ u + eps`. -/
theorem Roundoff.inflate_bounds_of_abs_sub_le {l u y z eps : ℝ} (hy : l ≤ y ∧ y ≤ u)
    (happrox : |z - y| ≤ eps) : l - eps ≤ z ∧ z ≤ u + eps := by
  obtain ⟨hupper, hlower⟩ := abs_sub_le_iff.1 happrox
  exact ⟨by linarith [hy.1], by linarith [hy.2]⟩

/-- If `y ≤ t - eps` and `|z - y| ≤ eps`, then `z ≤ t`. -/
theorem Roundoff.le_of_real_le_sub_margin {y t z eps : ℝ} (h : y ≤ t - eps)
    (happrox : |z - y| ≤ eps) :
    z ≤ t := by
  linarith [(abs_sub_le_iff.1 happrox).1]

/-- If `y ≥ t + eps` and `|z - y| ≤ eps`, then `z ≥ t`. -/
theorem Roundoff.ge_of_real_ge_add_margin {y t z eps : ℝ} (h : y ≥ t + eps)
    (happrox : |z - y| ≤ eps) :
    z ≥ t := by
  linarith [(abs_sub_le_iff.1 happrox).2]

/-! ## Widening a box by an approximation bound -/

open CROWN TwoLayerMLP

/-- The box `B` widened by `eps` in every component, from `B.lo - eps` to `B.hi + eps`. -/
def CROWN.Box.inflate {s : Shape} (B : Box ℝ s) (eps : ℝ) : Box ℝ s :=
  { lo := B.lo - Tensor.full (α := ℝ) s eps
  , hi := B.hi + Tensor.full (α := ℝ) s eps }

/-- If the real tensor `yS` lies in `B` (`hy`) and the runtime tensor `yR` approximates it within
`eps` (`happrox`), the interpretation of `yR` lies in `inflate B eps`. -/
theorem CROWN.Box.contains_inflate_of_approx {α : Type} [Storage α] {toSpec : α → ℝ}
    {s : Shape} {B : Box ℝ s} {yS : Tensor ℝ s} {yR : Tensor α s} {eps : ℝ}
    (hy : Box.Contains (α := ℝ) B yS)
    (happrox : ApproxTensor (α := α) (toSpec := toSpec) yS yR eps) :
    Box.Contains (α := ℝ) (inflate (B := B) eps)
      (toReal (α := α) (toSpec := toSpec) yR) := by
  induction s with
  | scalar =>
      have habs : |toSpec yR.item - yS.item| ≤ eps :=
        (approxTensor_scalar_item_iff (α := α) (toSpec := toSpec)).1 happrox
      have hinterval := Roundoff.inflate_bounds_of_abs_sub_le (hy := hy) habs
      simpa [inflate, Box.Contains, toReal,
        Tensor.sub_def, Tensor.add_def, Tensor.item,
        Tensor.map_apply,
        Tensor.zipWith_apply,
        Tensor.full_apply] using hinterval
  | dim n s ih =>
      intro i
      have hrec :=
        ih (B := { lo := B.lo.unstack i, hi := B.hi.unstack i })
          (yS := yS.unstack i) (yR := yR.unstack i)
          (hy := hy i)
          (happrox := ApproxTensor.unstack (α := α) (toSpec := toSpec) happrox i)
      have hfill :
          Tensor.unstack
              (Tensor.full (α := ℝ) (.dim n s) eps) i =
            Tensor.full (α := ℝ) s eps := by
        apply Tensor.Rep.ext
        intro coordinate
        simp
      change Box.Contains (α := ℝ)
        { lo := (Tensor.zipWith (· - ·) B.lo
            (Tensor.full (.dim n s) eps)).unstack i
          hi := (Tensor.zipWith (· + ·) B.hi
            (Tensor.full (.dim n s) eps)).unstack i }
        ((Tensor.map toSpec yR).unstack i)
      rw [Tensor.unstack_map,
        ← Tensor.Rep.zipWith_unstack,
        ← Tensor.Rep.zipWith_unstack, hfill]
      exact hrec

/-! ## The two-layer ReLU MLP -/

namespace NF

open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq

variable {β : Radix} {fexp : ℤ → ℤ} [ValidExp fexp]
variable {rnd : ℝ → ℤ} [ValidRndToNearest rnd]

local notation "R" => NF β fexp rnd

/-- The forward-error bound by which `reluTwoLayerMLP_ibp_sound` widens the real IBP box
of a two-layer ReLU MLP. -/
def ibpReLUTwoLayerMLPErrorBudget {inDim hidDim outDim : Nat}
    (netR : TwoLayerMLP R inDim hidDim outDim)
    (xR : Tensor R [inDim]) (eW1 eb1 eW2 eb2 ex : ℝ) : ℝ :=
  let l1R : Linear R inDim hidDim :=
    { weights := netR.hiddenWeight, bias := netR.hiddenBias }
  let l2R : Linear R hidDim outDim :=
    { weights := netR.outputWeight, bias := netR.outputBias }
  reluTwoLayerMLPErrorBudget eW1 eb1 eW2 eb2 ex l1R l2R xR

/-- Over the reals the CROWN forward pass is `Linear → max · 0 → Linear`, by
`Activation.relu_eq_max`. -/
private theorem crown_forward_eq_linear_relu {inDim hidDim outDim : Nat}
    (net : TwoLayerMLP ℝ inDim hidDim outDim)
    (x : Tensor ℝ [inDim]) :
    forward (α := ℝ) net x =
      (let hiddenLayer : Linear ℝ inDim hidDim :=
          { weights := net.hiddenWeight, bias := net.hiddenBias }
       let outputLayer : Linear ℝ hidDim outDim :=
          { weights := net.outputWeight, bias := net.outputBias }
       Linear.forward (α := ℝ) outputLayer
         (Tensor.map (fun value : ℝ => max value 0)
           (Linear.forward (α := ℝ) hiddenLayer x))) := by
  have hrelu : (Activation.relu : ℝ → ℝ) = (fun value => max value 0) :=
    funext fun value => Activation.relu_eq_max value
  simp only [forward, Tensor.relu, hrelu]

/-- If `xS` lies in `xB` (`hxB`) and the runtime parameters and input are within `eW1`, …, `ex` of
the real ones (`hW1`, …, `hx`), the rounded output of `netR` lies in the IBP box of `netS`,
widened by `ibpReLUTwoLayerMLPErrorBudget`. -/
theorem reluTwoLayerMLP_ibp_sound {inDim hidDim outDim : Nat}
    (netS : TwoLayerMLP ℝ inDim hidDim outDim)
    (netR : TwoLayerMLP R inDim hidDim outDim)
    (xB : Box ℝ (.dim inDim .scalar))
    (xS : Tensor ℝ [inDim])
    (xR : Tensor R [inDim])
    {eW1 eb1 eW2 eb2 ex : ℝ}
    (hW1 : ApproxTensor (α := R) (toSpec := NF.toReal)
      netS.hiddenWeight netR.hiddenWeight eW1)
    (hb1 : ApproxTensor (α := R) (toSpec := NF.toReal)
      netS.hiddenBias netR.hiddenBias eb1)
    (hW2 : ApproxTensor (α := R) (toSpec := NF.toReal)
      netS.outputWeight netR.outputWeight eW2)
    (hb2 : ApproxTensor (α := R) (toSpec := NF.toReal)
      netS.outputBias netR.outputBias eb2)
    (hx : ApproxTensor (α := R) (toSpec := NF.toReal)
      xS xR ex)
    (hxB : Box.Contains (α := ℝ) xB xS) :
    Box.Contains (α := ℝ)
      (Box.inflate (B := boundIBP (α := ℝ) netS xB)
        (ibpReLUTwoLayerMLPErrorBudget netR xR eW1 eb1 eW2 eb2 ex))
      (Tensor.toReal (α := R) (toSpec := NF.toReal)
        (let l1R : Linear R inDim hidDim :=
            { weights := netR.hiddenWeight, bias := netR.hiddenBias }
         let l2R : Linear R hidDim outDim :=
            { weights := netR.outputWeight, bias := netR.outputBias }
         let z1R := Linear.forward (α := R) l1R xR
         let a1R := Tensor.map (reluR (β := β) (fexp := fexp) (rnd := rnd)) z1R
         Linear.forward (α := R) l2R a1R)) := by
  have hyS : Box.Contains (α := ℝ) (boundIBP (α := ℝ) netS xB) (forward (α := ℝ) netS xS) :=
    TwoLayerMLP.boundIBP_sound (net := netS) (xB := xB) (x := xS) hxB
  let l1S : Linear ℝ inDim hidDim :=
    { weights := netS.hiddenWeight, bias := netS.hiddenBias }
  let l2S : Linear ℝ hidDim outDim :=
    { weights := netS.outputWeight, bias := netS.outputBias }
  let l1R : Linear R inDim hidDim :=
    { weights := netR.hiddenWeight, bias := netR.hiddenBias }
  let l2R : Linear R hidDim outDim :=
    { weights := netR.outputWeight, bias := netR.outputBias }
  have hOut := approxTensor_reluTwoLayerMLP
    (L0S := l1S) (L1S := l2S) (L0R := l1R) (L1R := l2R)
    (xS := xS) (xR := xR)
    (e0W := eW1) (e0b := eb1) (e1W := eW2) (e1b := eb2) (ex := ex)
    hW1 hb1 hW2 hb2 hx
  refine Box.contains_inflate_of_approx (α := R)
    (toSpec := NF.toReal)
    (B := boundIBP (α := ℝ) netS xB) (yS := forward (α := ℝ) netS xS)
    (yR :=
      (let z1R := Linear.forward (α := R) l1R xR
       let a1R := Tensor.map (reluR (β := β) (fexp := fexp) (rnd := rnd)) z1R
       Linear.forward (α := R) l2R a1R))
    (eps := ibpReLUTwoLayerMLPErrorBudget netR xR eW1 eb1 eW2 eb2 ex) hyS ?_
  -- With `change`, Lean would find the ReLU equation by reduction and unfold the tensor
  -- computations around it.
  rw [crown_forward_eq_linear_relu]
  exact hOut

end NF

end
end TorchLean
