/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Models.Mlp
public import NN.Proofs.RuntimeApprox.FP32.MLP

/-!
# FP32 CROWN/IBP Integration

The CROWN/IBP development (`NN/MLTheory/CROWN/*`) proves *real-valued* enclosure theorems of the
form “the network output lies in this box”.

This module combines those real enclosures with float32 forward-error bounds (expressed using
`approxTensor`) to obtain a **float32-sound** enclosure:
we inflate the real box by the explicit forward error budget.

This is the key separation of concerns:
- CROWN/IBP proves a real-valued semantic enclosure.
- RuntimeApprox/FP32 proves the rounded runtime output is close to that real-valued output.
- This file composes the two by uniformly widening the box.

"Float32" and the `_fp32` suffix here mean the rounded-real model
`TorchLean.Floats.FP32 := NF binaryRadix fexp32 rnd32`. These theorems say nothing about Lean's
`Float32` type or the bit-level `IEEE32Exec` model; that bridge lives in
`NN/Floats/IEEEExec/Bridge/FP32`.
-/

@[expose] public section


namespace NN.Proofs.RuntimeApprox.FP32

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor

open _root_.Proofs
open _root_.Proofs.RuntimeApprox
open _root_.Proofs.RuntimeApprox.NFBackend
open TorchLean.Floats

noncomputable section

/-! ## Scalar Margin Lemmas -/

/--
If a real value `y` lies in `[l, u]` and a runtime value `yR` is within `eps` of `y`, then the
interpreted runtime value lies in the widened interval `[l - eps, u + eps]`.

This is the scalar heart of the FP32/CROWN bridge.
-/
theorem interval_contains_inflate_of_abs_error {l u y : ℝ} {yR : R} {eps : ℝ}
    (hy : l ≤ y ∧ y ≤ u)
    (happrox : abs (toSpec yR - y) ≤ eps) :
    (l - eps ≤ toSpec yR) ∧ (toSpec yR ≤ u + eps) := by
  constructor
  · have : toSpec yR ≥ y - eps := by
      have h' := abs_sub_le_iff.1 (by simpa [abs_sub_comm] using happrox)
      -- `y - toSpec yR ≤ eps` rearranges to `toSpec yR ≥ y - eps`.
      linarith
    linarith [hy.1, this]
  · have : toSpec yR ≤ y + eps := by
      have h' := abs_sub_le_iff.1 (by simpa using happrox)
      linarith
    linarith [hy.2, this]

/--
One-sided upper-margin rule.

If the real value is at least `eps` below threshold `t`, then any FP32 value within `eps` is still
below `t`.
-/
theorem fp32_le_of_real_le_sub_margin {y t : ℝ} {yR : R} {eps : ℝ}
    (h : y ≤ t - eps)
    (happrox : abs (toSpec yR - y) ≤ eps) :
    toSpec yR ≤ t := by
  have h' := abs_sub_le_iff.1 (by simpa using happrox)
  have : toSpec yR ≤ y + eps := by linarith [h'.1]
  linarith

/--
One-sided lower-margin rule.

If the real value is at least `eps` above threshold `t`, then any FP32 value within `eps` is still
above `t`.
-/
theorem fp32_ge_of_real_ge_add_margin {y t : ℝ} {yR : R} {eps : ℝ}
    (h : y ≥ t + eps)
    (happrox : abs (toSpec yR - y) ≤ eps) :
    toSpec yR ≥ t := by
  have h' := abs_sub_le_iff.1 (by simpa [abs_sub_comm] using happrox)
  have : toSpec yR ≥ y - eps := by linarith [h'.2]
  linarith

/-! ## Inflating Real Boxes To Cover FP32 Execution -/

open NN.MLTheory.CROWN

/--
Uniformly widen a real-valued `CROWN.Box` by `eps` in every component.

The lower face moves down by `eps`; the upper face moves up by `eps`. This is kept simple
and conservative, matching an `L∞`-style output error bound.
-/
noncomputable def inflateBoxUniform {s : Shape} (B : Box ℝ s) (eps : ℝ) : Box ℝ s :=
  { lo := Tensor.subSpec B.lo (Tensor.full (α := ℝ) s eps)
  , hi := Tensor.addSpec B.hi (Tensor.full (α := ℝ) s eps) }

/--
Tensor version of `interval_contains_inflate_of_abs_error`.

If the real-spec output `yS` is inside a real CROWN/IBP box `B`, and the FP32 runtime output `yR`
approximates `yS` within uniform `eps`, then the interpreted FP32 output is inside the uniformly
widened box.
-/
theorem box_contains_inflateUniform_of_approx {s : Shape}
    {B : Box ℝ s} {yS : Tensor ℝ s} {yR : Tensor R s} {eps : ℝ}
    (hy : Box.contains (α := ℝ) B yS)
    (happrox : approxTensor (α := R) (toSpec := toSpec) yS yR eps) :
    Box.contains (α := ℝ) (inflateBoxUniform (B := B) eps)
      (tensorToSpec (α := R) (toSpec := toSpec) yR) := by
  induction s with
  | scalar =>
      have habs : abs (toSpec yR.item - yS.item) ≤ eps :=
        (approxTensor_scalar_iff (α := R) (toSpec := toSpec)
          (x := yS.item) (xR := yR.item) (eps := eps)).1 (by simpa using happrox)
      have hinterval := interval_contains_inflate_of_abs_error (hy := hy) habs
      simpa [inflateBoxUniform, Box.contains, tensorToSpec, TorchLean.Tensor.map,
        Tensor.full, Tensor.subSpec, Tensor.addSpec, Tensor.map2Spec, Tensor.item,
        TorchLean.Tensor.Internal.Rep.map_apply,
        TorchLean.Tensor.Internal.Rep.zipWith_apply,
        TorchLean.Tensor.Internal.Rep.const_apply] using hinterval
  | dim n s ih =>
      intro i
      have hrec :=
        ih (B := { lo := B.lo.unstack i, hi := B.hi.unstack i })
          (yS := yS.unstack i) (yR := yR.unstack i)
          (hy := hy i)
          (happrox := approxTensor_dim_get (α := R) (toSpec := toSpec) happrox i)
      have hfill :
          TorchLean.Tensor.Internal.Rep.unstack
              (Tensor.full (α := ℝ) (.dim n s) eps) i =
            Tensor.full (α := ℝ) s eps := by
        apply TorchLean.Tensor.Internal.Rep.ext
        intro coordinate
        simp [Tensor.full]
      change Box.contains (α := ℝ)
        { lo := (TorchLean.Tensor.Internal.Rep.zipWith (· - ·) B.lo
            (Tensor.full (.dim n s) eps)).unstack i
          hi := (TorchLean.Tensor.Internal.Rep.zipWith (· + ·) B.hi
            (Tensor.full (.dim n s) eps)).unstack i }
        ((TorchLean.Tensor.Internal.Rep.map toSpec yR).unstack i)
      rw [← TorchLean.Tensor.Internal.Rep.map_unstack,
        ← TorchLean.Tensor.Internal.Rep.zipWith_unstack,
        ← TorchLean.Tensor.Internal.Rep.zipWith_unstack, hfill]
      exact hrec

/-- Forward-error budget used to inflate the real IBP box for a two-layer ReLU network. -/
def ibpReluTwoLayerErrorBudget {inDim hidDim outDim : Nat}
    (netR : NN.MLTheory.CROWN.TwoLayerMLP R inDim hidDim outDim)
    (xR : Tensor R [inDim]) (eW1 eb1 eW2 eb2 ex : ℝ) : ℝ :=
  let l1R : Spec.LinearSpec R inDim hidDim :=
    { weights := netR.hiddenWeight, bias := netR.hiddenBias }
  let l2R : Spec.LinearSpec R hidDim outDim :=
    { weights := netR.outputWeight, bias := netR.outputBias }
  reluTwoLayerMlpErrorBudget eW1 eb1 eW2 eb2 ex l1R l2R xR

/--
Express the real CROWN forward pass in the form used by the FP32 approximation theorem.

The scalar ReLU has an explicit zero branch so that dual numbers clear their tangents at the
kink. Over the reals, `reluSpec_eq_max` identifies that branch with the usual maximum. Rewrite
the scalar function here, before it is placed inside the output enclosure and error budget.
-/
private theorem crown_forward_eq_linear_relu {inDim hidDim outDim : Nat}
    (net : NN.MLTheory.CROWN.TwoLayerMLP ℝ inDim hidDim outDim)
    (x : Tensor ℝ [inDim]) :
    NN.MLTheory.CROWN.forward (α := ℝ) net x =
      (let hiddenLayer : Spec.LinearSpec ℝ inDim hidDim :=
          { weights := net.hiddenWeight, bias := net.hiddenBias }
       let outputLayer : Spec.LinearSpec ℝ hidDim outDim :=
          { weights := net.outputWeight, bias := net.outputBias }
       Spec.linearSpec (α := ℝ) outputLayer
         (mapSpec (fun value : ℝ => max value 0)
           (Spec.linearSpec (α := ℝ) hiddenLayer x))) := by
  have hrelu : (Activation.Math.reluSpec : ℝ → ℝ) = (fun value => max value 0) :=
    funext fun value => Activation.Math.reluSpec_eq_max value
  simp only [NN.MLTheory.CROWN.forward, Activation.reluSpec, hrelu]

/--
Float32-sound IBP for a 2-layer ReLU MLP, via uniform output-box inflation:

1. Use the real-spec IBP theorem `NN.MLTheory.CROWN.Theorems.bound_ibp_sound`.
2. Use `approxTensor` to bound FP32 forward error (`approxTensor_reluTwoLayerMlp_fp32`).
3. Inflate the real IBP box by that error bound.

The result is a real-valued interval that is guaranteed to contain the `FP32` execution result.
-/
theorem ibpBound_contains_reluTwoLayerMlp_fp32 {inDim hidDim outDim : Nat}
    (netS : NN.MLTheory.CROWN.TwoLayerMLP ℝ inDim hidDim outDim)
    (netR : NN.MLTheory.CROWN.TwoLayerMLP R inDim hidDim outDim)
    (xB : NN.MLTheory.CROWN.Box ℝ (.dim inDim .scalar))
    (xS : Tensor ℝ [inDim])
    (xR : Tensor R [inDim])
    {eW1 eb1 eW2 eb2 ex : ℝ}
    (hW1 : approxTensor (α := R) (toSpec := toSpec) netS.hiddenWeight netR.hiddenWeight eW1)
    (hb1 : approxTensor (α := R) (toSpec := toSpec) netS.hiddenBias netR.hiddenBias eb1)
    (hW2 : approxTensor (α := R) (toSpec := toSpec) netS.outputWeight netR.outputWeight eW2)
    (hb2 : approxTensor (α := R) (toSpec := toSpec) netS.outputBias netR.outputBias eb2)
    (hx : approxTensor (α := R) (toSpec := toSpec) xS xR ex)
    (hxB : NN.MLTheory.CROWN.Box.contains (α := ℝ) xB xS) :
    NN.MLTheory.CROWN.Box.contains (α := ℝ)
      (inflateBoxUniform (B := NN.MLTheory.CROWN.boundIbp (α := ℝ) netS xB)
        (ibpReluTwoLayerErrorBudget netR xR eW1 eb1 eW2 eb2 ex))
      (tensorToSpec (α := R) (toSpec := toSpec)
        (let l1R : Spec.LinearSpec R inDim hidDim :=
            { weights := netR.hiddenWeight, bias := netR.hiddenBias }
         let l2R : Spec.LinearSpec R hidDim outDim :=
            { weights := netR.outputWeight, bias := netR.outputBias }
         let z1R := Spec.linearSpec (α := R) l1R xR
         let a1R := mapSpec (reluR (β := β) (fexp := fexp) (rnd := rnd)) z1R
         Spec.linearSpec (α := R) l2R a1R)) := by
  -- Real IBP box contains the real forward output.
  have hyS :
      NN.MLTheory.CROWN.Box.contains (α := ℝ)
        (NN.MLTheory.CROWN.boundIbp (α := ℝ) netS xB)
        (NN.MLTheory.CROWN.forward (α := ℝ) netS xS) :=
    NN.MLTheory.CROWN.Theorems.bound_ibp_sound (net := netS) (xB := xB) (x := xS) hxB

  -- FP32 forward is close to the real forward with the named propagated budget.
  let l1S : Spec.LinearSpec ℝ inDim hidDim :=
    { weights := netS.hiddenWeight, bias := netS.hiddenBias }
  let l2S : Spec.LinearSpec ℝ hidDim outDim :=
    { weights := netS.outputWeight, bias := netS.outputBias }
  let l1R : Spec.LinearSpec R inDim hidDim :=
    { weights := netR.hiddenWeight, bias := netR.hiddenBias }
  let l2R : Spec.LinearSpec R hidDim outDim :=
    { weights := netR.outputWeight, bias := netR.outputBias }

  let epsOut := ibpReluTwoLayerErrorBudget netR xR eW1 eb1 eW2 eb2 ex
  have hOut := approxTensor_reluTwoLayerMlp_fp32
    (L0S := l1S) (L1S := l2S) (L0R := l1R) (L1R := l2R)
    (xS := xS) (xR := xR)
    (e0W := eW1) (e0b := eb1) (e1W := eW2) (e1b := eb2) (ex := ex)
    hW1 hb1 hW2 hb2 hx

  -- Inflate the real box to cover the interpreted FP32 output.
  refine box_contains_inflateUniform_of_approx (B := NN.MLTheory.CROWN.boundIbp (α := ℝ) netS xB)
    (yS := NN.MLTheory.CROWN.forward (α := ℝ) netS xS)
    (yR :=
      (let z1R := Spec.linearSpec (α := R) l1R xR
       let a1R := mapSpec (reluR (β := β) (fexp := fexp) (rnd := rnd)) z1R
       Spec.linearSpec (α := R) l2R a1R))
    (eps := epsOut) hyS ?_
  -- The scalar equality is propositional; asking `change` to find it by reduction would
  -- unfold the surrounding tensor computations. Once rewritten, the layer records and
  -- named budget agree definitionally with the existing approximation theorem.
  rw [crown_forward_eq_linear_relu]
  exact hOut

end

end NN.Proofs.RuntimeApprox.FP32
