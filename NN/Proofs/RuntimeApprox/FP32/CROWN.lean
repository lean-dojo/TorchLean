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
`approxT`) to obtain a **float32-sound** enclosure:
we inflate the real box by the explicit forward error budget.

This is the key separation of concerns:
- CROWN/IBP proves a real-valued semantic enclosure.
- RuntimeApprox/FP32 proves the rounded runtime output is close to that real-valued output.
- This file composes the two by uniformly widening the box.
-/

@[expose] public section


namespace NN.Proofs.RuntimeApprox.FP32

open _root_.Spec
open _root_.Spec.Tensor

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

The lower face moves down by `eps`; the upper face moves up by `eps`. This is intentionally simple
and conservative, matching an `L∞`-style output error bound.
-/
noncomputable def inflateBoxUniform {s : Shape} (B : Box ℝ s) (eps : ℝ) : Box ℝ s :=
  { lo := Tensor.subSpec B.lo (Spec.fill (α := ℝ) eps s)
  , hi := Tensor.addSpec B.hi (Spec.fill (α := ℝ) eps s) }

/--
Tensor version of `interval_contains_inflate_of_abs_error`.

If the real-spec output `yS` is inside a real CROWN/IBP box `B`, and the FP32 runtime output `yR`
approximates `yS` within uniform `eps`, then the interpreted FP32 output is inside the uniformly
widened box.
-/
theorem box_contains_inflateUniform_of_approx {s : Shape}
    {B : Box ℝ s} {yS : Tensor ℝ s} {yR : Tensor R s} {eps : ℝ}
    (hy : Box.contains (α := ℝ) B yS)
    (happrox : approxT (α := R) (toSpec := toSpec) yS yR eps) :
    Box.contains (α := ℝ) (inflateBoxUniform (B := B) eps)
      (tensorToSpec (α := R) (toSpec := toSpec) yR) := by
  induction s with
  | scalar =>
      cases B with
      | mk lo hi =>
          cases lo with
          | scalar l =>
              cases hi with
              | scalar u =>
                  cases yS with
                  | scalar y =>
                      cases yR with
                      | scalar yR =>
                          have habs : abs (toSpec yR - y) ≤ eps :=
                            (approxT_scalar_iff (α := R) (toSpec := toSpec)
                              (x := y) (xR := yR) (eps := eps)).1 happrox
                          have hinterval := interval_contains_inflate_of_abs_error (hy := hy) habs
                          simpa [inflateBoxUniform, Box.contains, tensorToSpec, Spec.mapTensor,
                            Spec.fill,
                            Tensor.subSpec, Tensor.addSpec, Tensor.map2Spec] using hinterval
  | dim n s ih =>
      cases B with
      | mk lo hi =>
          cases lo with
          | dim loF =>
              cases hi with
              | dim hiF =>
                  cases yS with
                  | dim ySf =>
                      cases yR with
                      | dim yRf =>
                          intro i
                          have hy_i : Box.contains (α := ℝ) { lo := loF i, hi := hiF i } (ySf i) :=
                            hy i
                          have happrox_i :
                              approxT (α := R) (toSpec := toSpec) (ySf i) (yRf i) eps := by
                            simpa using
                              (approxT_dim_get (α := R) (toSpec := toSpec)
                                (xS := Tensor.dim ySf) (xR := Tensor.dim yRf) (eps := eps) happrox
                                  i)
                          have hrec :=
                            ih (B := { lo := loF i, hi := hiF i }) (yS := ySf i) (yR := yRf i)
                              (hy := hy_i) (happrox := happrox_i)
                          simpa [inflateBoxUniform, Box.contains, tensorToSpec, Spec.mapTensor,
                            Spec.fill,
                            Tensor.subSpec, Tensor.addSpec, Tensor.map2Spec] using hrec

/--
Float32-sound IBP for a 2-layer ReLU MLP, via uniform output-box inflation:

1. Use the real-spec IBP theorem `NN.MLTheory.CROWN.Theorems.bound_ibp_sound`.
2. Use `approxT` to bound FP32 forward error (`approxT_reluMlp2_fp32`).
3. Inflate the real IBP box by that error bound.

The result is a real-valued interval that is guaranteed to contain the `FP32` execution result.
-/
theorem ibpBound_contains_reluMlp2_fp32 {inDim hidDim outDim : Nat}
    (netS : NN.MLTheory.CROWN.MLP2 ℝ inDim hidDim outDim)
    (netR : NN.MLTheory.CROWN.MLP2 R inDim hidDim outDim)
    (xB : NN.MLTheory.CROWN.Box ℝ (.dim inDim .scalar))
    (xS : Tensor ℝ (.dim inDim .scalar))
    (xR : Tensor R (.dim inDim .scalar))
    {eW1 eb1 eW2 eb2 ex : ℝ}
    (hW1 : approxT (α := R) (toSpec := toSpec) netS.W1 netR.W1 eW1)
    (hb1 : approxT (α := R) (toSpec := toSpec) netS.b1 netR.b1 eb1)
    (hW2 : approxT (α := R) (toSpec := toSpec) netS.W2 netR.W2 eW2)
    (hb2 : approxT (α := R) (toSpec := toSpec) netS.b2 netR.b2 eb2)
    (hx : approxT (α := R) (toSpec := toSpec) xS xR ex)
    (hxB : NN.MLTheory.CROWN.Box.contains (α := ℝ) xB xS) :
    ∃ epsOut : ℝ,
      NN.MLTheory.CROWN.Box.contains (α := ℝ)
        (inflateBoxUniform (B := NN.MLTheory.CROWN.boundIbp (α := ℝ) netS xB) epsOut)
        (tensorToSpec (α := R) (toSpec := toSpec)
          (let l1R : Spec.LinearSpec R inDim hidDim := { weights := netR.W1, bias := netR.b1 }
           let l2R : Spec.LinearSpec R hidDim outDim := { weights := netR.W2, bias := netR.b2 }
           let z1R := Spec.linearSpec (α := R) l1R xR
           let a1R := mapSpec (reluR (β := β) (fexp := fexp) (rnd := rnd)) z1R
           Spec.linearSpec (α := R) l2R a1R)) := by
  -- Real IBP box contains the real forward output.
  have hyS :
      NN.MLTheory.CROWN.Box.contains (α := ℝ)
        (NN.MLTheory.CROWN.boundIbp (α := ℝ) netS xB)
        (NN.MLTheory.CROWN.forward (α := ℝ) netS xS) :=
    NN.MLTheory.CROWN.Theorems.bound_ibp_sound (net := netS) (xB := xB) (x := xS) hxB

  -- FP32 forward is close to the real forward, with some propagated `epsOut`.
  let l1S : Spec.LinearSpec ℝ inDim hidDim := { weights := netS.W1, bias := netS.b1 }
  let l2S : Spec.LinearSpec ℝ hidDim outDim := { weights := netS.W2, bias := netS.b2 }
  let l1R : Spec.LinearSpec R inDim hidDim := { weights := netR.W1, bias := netR.b1 }
  let l2R : Spec.LinearSpec R hidDim outDim := { weights := netR.W2, bias := netR.b2 }

  rcases approxT_reluMlp2_fp32
    (L0S := l1S) (L1S := l2S) (L0R := l1R) (L1R := l2R)
    (xS := xS) (xR := xR)
    (e0W := eW1) (e0b := eb1) (e1W := eW2) (e1b := eb2) (ex := ex)
    hW1 hb1 hW2 hb2 hx with ⟨epsOut, hOut⟩
  refine ⟨epsOut, ?_⟩

  -- Inflate the real box to cover the interpreted FP32 output.
  refine box_contains_inflateUniform_of_approx (B := NN.MLTheory.CROWN.boundIbp (α := ℝ) netS xB)
    (yS := NN.MLTheory.CROWN.forward (α := ℝ) netS xS)
    (yR :=
      (let z1R := Spec.linearSpec (α := R) l1R xR
       let a1R := mapSpec (reluR (β := β) (fexp := fexp) (rnd := rnd)) z1R
       Spec.linearSpec (α := R) l2R a1R))
    (eps := epsOut) hyS ?_
  -- Match the MLP approximation conclusion to the `CROWN.forward`/`reluR` expressions.
  simpa [NN.MLTheory.CROWN.forward, l1S, l2S, l1R, l2R] using hOut

end

end NN.Proofs.RuntimeApprox.FP32
