/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.BackwardOps
public import NN.Proofs.RuntimeApprox.NF.ConvForward
public import NN.Proofs.RuntimeApprox.NF.Utils
public import NN.Spec.Layers.Utils

/-!
# Conv2D Backward Approximation

NF (rounded) backend: Conv2D backward (VJP) runtime→spec approximation.

This file proves soundness of explicit bounds for the three Conv2D gradients computed by
`Spec.conv2d_backward_spec`:
- kernel gradient
- bias gradient
- input gradient

The file is intentionally explicit because each gradient has a different nested-indexing pattern.
The important public objects are the tensor-level bounds (`conv2d*BoundTensor`), the approximation
theorems (`approxT_conv2d_*_deriv_spec`), and `conv2dRevNode`, which packages Conv2D as a `RevNode`
so it composes via `RevGraph.backprop_approx`.

PyTorch analogue: these are the VJP/gradient computations produced by Autograd for Conv2D.
https://pytorch.org/docs/stable/autograd.html
https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html

## Map of this file
- Shared padding/read lemmas (to relate the padded-input branches to the original `approxT`
  hypothesis).
- Bias gradient bounds: `conv2dBiasPointBound`, `approx_conv2d_bias_point`, and tensor-lifted
  bound.
- Kernel gradient bounds: `conv2dKernelPointBound`, `approx_conv2d_kernel_point`, and
  tensor-lifted bound.
- Input gradient bounds: `conv2dInputPointBound`, `approx_conv2d_input_point`, and tensor-lifted
  bound.
- `conv2dRevNode`: packaging as a `RevNode` so the bound composes inside larger graphs.

## References
- Dumoulin & Visin, *A guide to convolution arithmetic for deep learning* (indexing/stride/padding
  conventions).
- Goodfellow, Bengio, Courville, *Deep Learning* (MIT Press, 2016), convolution/backprop background.
- Baydin et al., *Automatic Differentiation in Machine Learning: a Survey* (JMLR 2018) (VJP
  framing).
-/

@[expose] public section


namespace Proofs
namespace RuntimeApprox

open Spec
open Tensor
open NN.MLTheory.Robustness.Spec

noncomputable section

namespace NFBackend

open TorchLean.Floats

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]
variable {rnd : ℝ → ℤ} [NeuralValidRndToNearest rnd]

local notation "R" => TorchLean.Floats.NF β fexp rnd

set_option maxHeartbeats 12000000

-- ---------------------------------------------------------------------------
-- Fold helper: turn `acc + foldl (· + f _) 0` into a fold with init `acc`.
-- ---------------------------------------------------------------------------

private lemma specFold5_eq_threadFold5
    {α : Type} [AddMonoid α]
    {outC out_h out_w kH kW : Nat}
    (term : Fin outC → Fin out_h → Fin out_w → Fin kH → Fin kW → α) :
    let specFold : α :=
      (List.finRange outC).foldl (fun acc out_ch =>
          acc +
            (List.finRange out_h).foldl (fun acc out_i =>
                acc +
                  (List.finRange out_w).foldl (fun acc out_j =>
                      acc +
                        (List.finRange kH).foldl (fun acc di =>
                            acc +
                              (List.finRange kW).foldl (fun acc dj =>
                                  acc + term out_ch out_i out_j di dj)
                                0)
                          0)
                    0)
              0)
        0
    let threadFold : α :=
      (List.finRange outC).foldl (fun accC out_ch =>
          (List.finRange out_h).foldl (fun accH out_i =>
              (List.finRange out_w).foldl (fun accW out_j =>
                  (List.finRange kH).foldl (fun accKH di =>
                      (List.finRange kW).foldl (fun accKW dj =>
                          accKW + term out_ch out_i out_j di dj)
                        accKH)
                    accW)
                accH)
            accC)
        0
    specFold = threadFold := by
  intro specFold threadFold
  classical
  -- Auxiliary "spec-sum" functions at each nested level.
  let sumKW0 : Fin outC → Fin out_h → Fin out_w → Fin kH → α :=
    fun out_ch out_i out_j di =>
      (List.finRange kW).foldl (fun acc dj => acc + term out_ch out_i out_j di dj) 0
  let sumKH0 : Fin outC → Fin out_h → Fin out_w → α :=
    fun out_ch out_i out_j =>
      (List.finRange kH).foldl (fun acc di => acc + sumKW0 out_ch out_i out_j di) 0
  let sumW0 : Fin outC → Fin out_h → α :=
    fun out_ch out_i =>
      (List.finRange out_w).foldl (fun acc out_j => acc + sumKH0 out_ch out_i out_j) 0
  let sumH0 : Fin outC → α :=
    fun out_ch =>
      (List.finRange out_h).foldl (fun acc out_i => acc + sumW0 out_ch out_i) 0

  -- Rewrite the spec fold into the threaded fold by repeatedly using `add_foldl_add0`.
  have hC :
      (List.finRange outC).foldl (fun acc out_ch => acc + sumH0 out_ch) 0 =
        (List.finRange outC).foldl (fun accC out_ch =>
            (List.finRange out_h).foldl (fun accH out_i =>
                (List.finRange out_w).foldl (fun accW out_j =>
                    (List.finRange kH).foldl (fun accKH di =>
                        (List.finRange kW).foldl (fun accKW dj =>
                            accKW + term out_ch out_i out_j di dj)
                          accKH)
                      accW)
                  accH)
              accC)
          0 := by
    refine foldl_congr (l := List.finRange outC)
      (f := fun acc out_ch => acc + sumH0 out_ch)
      (g := fun accC out_ch =>
        (List.finRange out_h).foldl (fun accH out_i =>
            (List.finRange out_w).foldl (fun accW out_j =>
                (List.finRange kH).foldl (fun accKH di =>
                    (List.finRange kW).foldl (fun accKW dj =>
                        accKW + term out_ch out_i out_j di dj)
                      accKH)
                  accW)
              accH)
          accC)
      (init := (0 : α)) ?_
    intro accC out_ch
    -- Push `accC` into the out_h fold, then convert the body similarly at deeper levels.
    have hPushH :
        accC + sumH0 out_ch =
          (List.finRange out_h).foldl (fun accH out_i => accH + sumW0 out_ch out_i) accC := by
      -- `sumH0 out_ch` is a fold from 0, so use `add_foldl_add0`.
      simpa [sumH0, add_assoc] using
        (List.add_foldl_add0 (l := List.finRange out_h) (f := fun out_i => sumW0 out_ch out_i) (acc
          := accC))
    -- Now rewrite the out_h fold body to the threaded out_w fold.
    have hH :
        (List.finRange out_h).foldl (fun accH out_i => accH + sumW0 out_ch out_i) accC =
          (List.finRange out_h).foldl (fun accH out_i =>
              (List.finRange out_w).foldl (fun accW out_j =>
                  (List.finRange kH).foldl (fun accKH di =>
                      (List.finRange kW).foldl (fun accKW dj =>
                          accKW + term out_ch out_i out_j di dj)
                        accKH)
                    accW)
                accH)
            accC := by
      refine foldl_congr (l := List.finRange out_h)
        (f := fun accH out_i => accH + sumW0 out_ch out_i)
        (g := fun accH out_i =>
          (List.finRange out_w).foldl (fun accW out_j =>
              (List.finRange kH).foldl (fun accKH di =>
                  (List.finRange kW).foldl (fun accKW dj =>
                      accKW + term out_ch out_i out_j di dj)
                    accKH)
                accW)
            accH)
        (init := accC) ?_
      intro accH out_i
      have hPushW :
          accH + sumW0 out_ch out_i =
            (List.finRange out_w).foldl (fun accW out_j => accW + sumKH0 out_ch out_i out_j) accH :=
              by
        simpa [sumW0, add_assoc] using
          (List.add_foldl_add0 (l := List.finRange out_w) (f := fun out_j => sumKH0 out_ch out_i
            out_j) (acc := accH))
      -- Convert the out_w fold body to the threaded kH/kW fold.
      have hW :
          (List.finRange out_w).foldl (fun accW out_j => accW + sumKH0 out_ch out_i out_j) accH =
            (List.finRange out_w).foldl (fun accW out_j =>
                (List.finRange kH).foldl (fun accKH di =>
                    (List.finRange kW).foldl (fun accKW dj =>
                        accKW + term out_ch out_i out_j di dj)
                      accKH)
                  accW)
              accH := by
        refine foldl_congr (l := List.finRange out_w)
          (f := fun accW out_j => accW + sumKH0 out_ch out_i out_j)
          (g := fun accW out_j =>
            (List.finRange kH).foldl (fun accKH di =>
                (List.finRange kW).foldl (fun accKW dj =>
                    accKW + term out_ch out_i out_j di dj)
                  accKH)
              accW)
          (init := accH) ?_
        intro accW out_j
        have hPushKH :
            accW + sumKH0 out_ch out_i out_j =
              (List.finRange kH).foldl (fun accKH di => accKH + sumKW0 out_ch out_i out_j di) accW
                := by
          simpa [sumKH0, add_assoc] using
            (List.add_foldl_add0 (l := List.finRange kH) (f := fun di => sumKW0 out_ch out_i out_j
              di) (acc := accW))
        have hKH :
            (List.finRange kH).foldl (fun accKH di => accKH + sumKW0 out_ch out_i out_j di) accW =
              (List.finRange kH).foldl (fun accKH di =>
                  (List.finRange kW).foldl (fun accKW dj =>
                      accKW + term out_ch out_i out_j di dj)
                    accKH)
                accW := by
          refine foldl_congr (l := List.finRange kH)
            (f := fun accKH di => accKH + sumKW0 out_ch out_i out_j di)
            (g := fun accKH di =>
              (List.finRange kW).foldl (fun accKW dj => accKW + term out_ch out_i out_j di dj)
                accKH)
            (init := accW) ?_
          intro accKH di
          -- Push `accKH` into the kW fold.
          simpa [sumKW0, add_assoc] using
            (List.add_foldl_add0 (l := List.finRange kW) (f := fun dj => term out_ch out_i out_j di
              dj) (acc := accKH))
        exact hPushKH.trans hKH
      exact hPushW.trans hW
    exact hPushH.trans hH

  simpa [specFold, threadFold, sumH0, sumW0, sumKH0, sumKW0, add_assoc] using hC

-- ---------------------------------------------------------------------------
-- Component selection lemmas (match-based indexing ↔ `get_at_or_zero`)
-- ---------------------------------------------------------------------------

@[simp] private lemma get_at_or_zero_tensor_cast {α : Type} [Zero α] {s t : Shape} (h : s = t)
    (x : Tensor α s) (idx : List Nat) :
    getAtOrZero (Tensor.castShape (t := x) h) idx = getAtOrZero x idx := by
  cases h
  rfl

/-- Padded-input helper used by the Conv2D spec: cast when `padding = 0`, otherwise `padMultiChannel`. -/
private def paddedInput {α : Type} [Context α] {inC inH inW padding : Nat}
    (img : Spec.MultiChannelImage inC inH inW α) :
    Spec.MultiChannelImage inC (inH + 2 * padding) (inW + 2 * padding) α :=
  if h4 : padding = 0 then
    tensorCast
      (Shape.dim inC
        (Shape.dim (inH + 2 * padding) (Shape.dim (inW + 2 * padding) Shape.scalar)))
      (by simp; rw [h4])
      img
  else
    Spec.padMultiChannel img padding

private lemma get_at_or_zero_paddedInput
    {α : Type} [Context α] {inC inH inW padding : Nat}
    (img : Spec.MultiChannelImage inC inH inW α) (c : Fin inC) (p q : Nat) :
    getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) img)
        [c.val, p, q]
      =
    (if _h : p < padding ∨ q < padding then
        (0 : α)
      else
        getAtOrZero img [c.val, p - padding, q - padding]) := by
  classical
  by_cases h0 : padding = 0
  · subst h0
    simp [paddedInput]
  · simpa [paddedInput, h0] using
      (Spec.get_at_or_zero_pad_multi_channel (α := α) (img := img) (c := c) (p := p) (q := q)
        (padding := padding))

private lemma mkInputIdx_match_eq_paddedInput
    {α : Type} [Context α] {inC inH inW stride padding : Nat}
    (img : Spec.MultiChannelImage inC inH inW α) (c : Fin inC)
    (oi di oj dj : Nat) :
    (match Spec.Private.mkInputIdx? [oi, oj] [di, dj] [stride, stride] [padding, padding] with
      | none => (0 : α)
      | some inIdx => getAtOrZero img (c.val :: inIdx))
      =
    getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) img)
        [c.val, oi * stride + di, oj * stride + dj] := by
  classical
  -- Compare both sides via the explicit `paddedInput` read formula.
  rw [get_at_or_zero_paddedInput (img := img) (c := c) (p := oi * stride + di) (q := oj * stride + dj)]
  by_cases h0 : oi * stride + di < padding
  · simp [Spec.Private.mkInputIdx?, h0]
  · by_cases h1 : oj * stride + dj < padding
    · simp [Spec.Private.mkInputIdx?, h0, h1]
    · simp [Spec.Private.mkInputIdx?, h0, h1]

private lemma conv2dKernelFoldRead_eq_paddedFold
    {α : Type} [Context α] {inC outC inH inW outH outW stride padding : Nat}
    (input : Spec.MultiChannelImage inC inH inW α)
    (grad : Spec.MultiChannelImage outC outH outW α)
    (out_ch : Fin outC) (in_ch : Fin inC) (di dj : Nat) :
    (List.finRange outH).foldl (fun acc i =>
        (List.finRange outW).foldl (fun acc j =>
          acc +
            (match Spec.Private.mkInputIdx? [i.val, j.val] [di, dj] [stride, stride]
                [padding, padding] with
              | none => 0
              | some inIdx => getAtOrZero input (in_ch.val :: inIdx)) *
              getAtOrZero grad [out_ch.val, i.val, j.val]) acc) 0 =
      (List.finRange outH).foldl (fun acc i =>
        (List.finRange outW).foldl (fun acc j =>
          acc +
            getAtOrZero
                (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding)
                  input)
                [in_ch.val, i.val * stride + di, j.val * stride + dj] *
              getAtOrZero grad [out_ch.val, i.val, j.val]) acc) 0 := by
  refine foldl_congr (l := List.finRange outH)
    (f := fun acc i =>
      (List.finRange outW).foldl (fun acc j =>
        acc +
          (match Spec.Private.mkInputIdx? [i.val, j.val] [di, dj] [stride, stride]
              [padding, padding] with
            | none => 0
            | some inIdx => getAtOrZero input (in_ch.val :: inIdx)) *
            getAtOrZero grad [out_ch.val, i.val, j.val]) acc)
    (g := fun acc i =>
      (List.finRange outW).foldl (fun acc j =>
        acc +
          getAtOrZero
              (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding)
                input)
              [in_ch.val, i.val * stride + di, j.val * stride + dj] *
            getAtOrZero grad [out_ch.val, i.val, j.val]) acc)
    (init := (0 : α)) ?_
  intro acc i
  refine foldl_congr (l := List.finRange outW)
    (f := fun acc j =>
      acc +
        (match Spec.Private.mkInputIdx? [i.val, j.val] [di, dj] [stride, stride]
            [padding, padding] with
          | none => 0
          | some inIdx => getAtOrZero input (in_ch.val :: inIdx)) *
          getAtOrZero grad [out_ch.val, i.val, j.val])
    (g := fun acc j =>
      acc +
        getAtOrZero
            (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding)
              input)
            [in_ch.val, i.val * stride + di, j.val * stride + dj] *
          getAtOrZero grad [out_ch.val, i.val, j.val])
    (init := acc) ?_
  intro acc j
  have hRead :=
    mkInputIdx_match_eq_paddedInput (stride := stride) (padding := padding)
      (img := input) (c := in_ch) (oi := i.val) (di := di) (oj := j.val) (dj := dj)
  simpa using congrArg (fun x => acc + x * getAtOrZero grad [out_ch.val, i.val, j.val]) hRead

lemma entry_eq_scalar_get_at_or_zero1
    {α : Type} [Zero α] {n : Nat}
    (t : Tensor α (.dim n .scalar)) (i : Fin n) :
    (match t with
    | .dim f => f i) = Tensor.scalar (getAtOrZero t [i.val]) := by
  cases t with
  | dim f =>
      have hi : i.val < n := i.isLt
      cases h1 : f i with
      | scalar v =>
          simp [hi, h1]

lemma entry_eq_scalar_get_at_or_zero4
    {α : Type} [Zero α] {n1 n2 n3 n4 : Nat}
    (t : Tensor α (.dim n1 (.dim n2 (.dim n3 (.dim n4 .scalar)))))
    (i1 : Fin n1) (i2 : Fin n2) (i3 : Fin n3) (i4 : Fin n4) :
    (match
      match
        match
          match t with
          | .dim f => f i1 with
        | .dim g => g i2 with
      | .dim h => h i3 with
    | .dim k => k i4) =
      Tensor.scalar (getAtOrZero t [i1.val, i2.val, i3.val, i4.val]) := by
  cases t with
  | dim f =>
      have hi1 : i1.val < n1 := i1.isLt
      cases h1 : f i1 with
      | dim g =>
          have hi2 : i2.val < n2 := i2.isLt
          cases h2 : g i2 with
          | dim h =>
              have hi3 : i3.val < n3 := i3.isLt
              cases h3 : h i3 with
              | dim k =>
                  have hi4 : i4.val < n4 := i4.isLt
                  cases h4 : k i4 with
                  | scalar v =>
                      simp [hi1, hi2, hi3, hi4, h1, h2, h3, h4]

-- ---------------------------------------------------------------------------
-- Conv2D bias gradient: pointwise bound
-- ---------------------------------------------------------------------------

/--
Pointwise error bound for the Conv2D **bias** gradient (NF runtime vs spec).

This bound is a replay of the bias-gradient summation with per-term error `epsδ` coming from the
`grad_output` approximation hypothesis.
-/
def conv2dBiasPointBound
    {outC kH kW stride padding inH inW : Nat}
    (δR : Spec.MultiChannelImage outC (conv2dOutH inH kH stride padding) (conv2dOutW inW kW stride
      padding) R)
    (epsδ : ℝ)
    (out_ch : Fin outC) : ℝ :=
  let out_h := conv2dOutH inH kH stride padding
  let out_w := conv2dOutW inW kW stride padding
  let idxs : List (Fin out_h × Fin out_w) :=
    (List.finRange out_h).flatMap (fun i => (List.finRange out_w).map (fun j => (i, j)))
  let termR : (Fin out_h × Fin out_w) → R := fun t =>
    getAtOrZero δR [out_ch.val, t.1.val, t.2.val]
  let epsTerm : (Fin out_h × Fin out_w) → ℝ := fun _ => epsδ
  (foldAddState (β := β) (fexp := fexp) (rnd := rnd) idxs termR epsTerm).2

/--
Soundness of the Conv2D **bias**-gradient pointwise bound.

Given `approxT` for `grad_output`, this shows the spec bias-gradient entry is approximated by the
NF runtime entry within `conv2dBiasPointBound`.
-/
theorem approx_conv2d_bias_point
    {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    {kernelS : Tensor ℝ (.dim outC (.dim inC (.dim kH (.dim kW .scalar))))}
    {kernelR : Tensor R (.dim outC (.dim inC (.dim kH (.dim kW .scalar))))}
    {biasS : Tensor ℝ (.dim outC .scalar)} {biasR : Tensor R (.dim outC .scalar)}
    {inputS : Spec.MultiChannelImage inC inH inW ℝ}
    {inputR : Spec.MultiChannelImage inC inH inW R}
    {δS : Spec.MultiChannelImage outC (conv2dOutH inH kH stride padding) (conv2dOutW inW kW stride
      padding) ℝ}
    {δR : Spec.MultiChannelImage outC (conv2dOutH inH kH stride padding) (conv2dOutW inW kW stride
      padding) R}
    {epsδ : ℝ}
    (hδ : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) δS δR epsδ)
    (out_ch : Fin outC) :
    let layerS : Spec.Conv2DSpec inC outC kH kW stride padding ℝ h1 h2 h3 :=
      { kernel := kernelS, bias := biasS }
    let layerR : Spec.Conv2DSpec inC outC kH kW stride padding R h1 h2 h3 :=
      { kernel := kernelR, bias := biasR }
    abs
        (toSpec (β := β) (fexp := fexp) (rnd := rnd)
            (getAtOrZero (Spec.conv2dBiasDerivSpec (α := R) (layer := layerR) (input :=
              inputR) (grad_output := δR))
              [out_ch.val]) -
            getAtOrZero (Spec.conv2dBiasDerivSpec (α := ℝ) (layer := layerS) (input :=
              inputS) (grad_output := δS))
                [out_ch.val]) ≤
        conv2dBiasPointBound (β := β) (fexp := fexp) (rnd := rnd)
          (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding) (inH := inH)
            (inW := inW)
          δR epsδ out_ch := by
  intro layerS layerR
  classical
  let out_h := conv2dOutH inH kH stride padding
  let out_w := conv2dOutW inW kW stride padding
  let idxs : List (Fin out_h × Fin out_w) :=
    (List.finRange out_h).flatMap (fun i => (List.finRange out_w).map (fun j => (i, j)))
  let termR : (Fin out_h × Fin out_w) → R := fun t =>
    getAtOrZero δR [out_ch.val, t.1.val, t.2.val]
  let termS : (Fin out_h × Fin out_w) → ℝ := fun t =>
    getAtOrZero δS [out_ch.val, t.1.val, t.2.val]
  let epsTerm : (Fin out_h × Fin out_w) → ℝ := fun _ => epsδ
  let sumR : R := idxs.foldl (fun acc t => acc + termR t) 0
  let sumS : ℝ := idxs.foldl (fun acc t => acc + termS t) 0
  let sumEps : ℝ := (foldAddState (β := β) (fexp := fexp) (rnd := rnd) idxs termR epsTerm).2

  have hTermIdx : ∀ t ∈ idxs, abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (termR t) - termS t)
    ≤ epsTerm t := by
    intro t _ht
    rcases t with ⟨i, j⟩
    simpa [termR, termS, epsTerm] using
      (approx_get_at_or_zero (β := β) (fexp := fexp) (rnd := rnd)
        (s := .dim outC (.dim out_h (.dim out_w .scalar)))
        (xS := δS) (xR := δR) (eps := epsδ) hδ [out_ch.val, i.val, j.val])

  have hSum :
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) sumR - sumS) ≤ sumEps := by
    simpa [sumR, sumS, sumEps] using
      (approx_fold_add (β := β) (fexp := fexp) (rnd := rnd)
        (l := idxs) (termS := termS) (termR := termR) (epsTerm := epsTerm) hTermIdx)

  have hsumR_nested :
      (List.finRange out_h).foldl (fun acc i =>
          (List.finRange out_w).foldl (fun acc j => acc + termR (i, j)) acc) 0 = sumR := by
    let fR : R → (Fin out_h × Fin out_w) → R := fun acc t => acc + termR t
    have hW :
        ∀ (acc : R) (i : Fin out_h),
          List.foldl fR acc ((List.finRange out_w).map (fun j => (i, j))) =
            (List.finRange out_w).foldl (fun acc j => acc + termR (i, j)) acc := by
      intro acc i
      exact
        (List.foldl_map (f := fun j => (i, j)) (g := fR) (l := List.finRange out_w) (init := acc))
    have hC :
        idxs.foldl fR (0 : R) =
          (List.finRange out_h).foldl (fun acc i =>
            List.foldl fR acc ((List.finRange out_w).map (fun j => (i, j)))) 0 := by
      simpa [idxs] using
        (foldl_flatMap (l := List.finRange out_h) (g := fun i => (List.finRange out_w).map (fun j =>
          (i, j)))
          (f := fR) (init := (0 : R)))
    have hC' :
        idxs.foldl fR 0 =
          (List.finRange out_h).foldl (fun acc i => (List.finRange out_w).foldl (fun acc j => acc +
            termR (i, j)) acc) 0 := by
      have :=
        foldl_congr (l := List.finRange out_h)
          (f := fun acc i => List.foldl fR acc ((List.finRange out_w).map (fun j => (i, j))))
          (g := fun acc i => (List.finRange out_w).foldl (fun acc j => acc + termR (i, j)) acc)
          (init := (0 : R)) (by intro acc i; simpa using (hW acc i))
      simpa [hC] using this
    simpa [sumR, fR] using hC'.symm

  have hsumS_nested :
      (List.finRange out_h).foldl (fun acc i =>
          (List.finRange out_w).foldl (fun acc j => acc + termS (i, j)) acc) 0 = sumS := by
    -- identical proof over `ℝ`
    let fS : ℝ → (Fin out_h × Fin out_w) → ℝ := fun acc t => acc + termS t
    have hW :
        ∀ (acc : ℝ) (i : Fin out_h),
          List.foldl fS acc ((List.finRange out_w).map (fun j => (i, j))) =
            (List.finRange out_w).foldl (fun acc j => acc + termS (i, j)) acc := by
      intro acc i
      exact
        (List.foldl_map (f := fun j => (i, j)) (g := fS) (l := List.finRange out_w) (init := acc))
    have hC :
        idxs.foldl fS (0 : ℝ) =
          (List.finRange out_h).foldl (fun acc i =>
            List.foldl fS acc ((List.finRange out_w).map (fun j => (i, j)))) 0 := by
      simpa [idxs] using
        (foldl_flatMap (l := List.finRange out_h) (g := fun i => (List.finRange out_w).map (fun j =>
          (i, j)))
          (f := fS) (init := (0 : ℝ)))
    have hC' :
        idxs.foldl fS 0 =
          (List.finRange out_h).foldl (fun acc i => (List.finRange out_w).foldl (fun acc j => acc +
            termS (i, j)) acc) 0 := by
      have :=
        foldl_congr (l := List.finRange out_h)
          (f := fun acc i => List.foldl fS acc ((List.finRange out_w).map (fun j => (i, j))))
          (g := fun acc i => (List.finRange out_w).foldl (fun acc j => acc + termS (i, j)) acc)
          (init := (0 : ℝ)) (by intro acc i; simpa using (hW acc i))
      simpa [hC] using this
    simpa [sumS, fS] using hC'.symm

  have houtR :
      getAtOrZero (Spec.conv2dBiasDerivSpec (α := R) (layer := layerR) (input := inputR)
        (grad_output := δR))
        [out_ch.val] = sumR := by
    have hFoldR :
        (List.finRange out_h).foldl (fun acc i =>
            (List.finRange out_w).foldl (fun acc j =>
                acc + getAtOrZero δR [out_ch.val, i.val, j.val]) acc) 0 = sumR := by
      have hNested :
          (List.finRange out_h).foldl (fun acc i =>
              (List.finRange out_w).foldl (fun acc j =>
                  acc + getAtOrZero δR [out_ch.val, i.val, j.val]) acc) 0 =
            (List.finRange out_h).foldl (fun acc i =>
              (List.finRange out_w).foldl (fun acc j => acc + termR (i, j)) acc) 0 := by
        refine foldl_congr (l := List.finRange out_h)
          (f := fun acc i =>
            (List.finRange out_w).foldl (fun acc j => acc + getAtOrZero δR [out_ch.val, i.val,
              j.val]) acc)
          (g := fun acc i =>
            (List.finRange out_w).foldl (fun acc j => acc + termR (i, j)) acc)
          (init := (0 : R)) ?_
        intro acc i
        refine foldl_congr (l := List.finRange out_w)
          (f := fun acc j => acc + getAtOrZero δR [out_ch.val, i.val, j.val])
          (g := fun acc j => acc + termR (i, j))
          (init := acc) ?_
        intro acc j
        simp [termR]
      simpa [hNested] using hsumR_nested
    have hFoldR' :
        (List.finRange ((inH + 2 * padding - kH) / stride + 1)).foldl (fun acc i =>
            (List.finRange ((inW + 2 * padding - kW) / stride + 1)).foldl (fun acc j =>
                acc + getAtOrZero δR [out_ch.val, i.val, j.val]) acc) 0 = sumR := by
      simpa [out_h, out_w, conv2dOutH, conv2dOutW] using hFoldR
    simpa [Spec.conv2dBiasDerivSpec, Spec.convBiasDerivSpec, Spec.Private.foldlIndices,
      Spec.Private.foldlIndices.go, Spec.convOutSpatial, Spec.convOutDim, Vector.get,
      Vector.toList_ofFn, out_ch.isLt, sumR] using hFoldR'

  have houtS :
      getAtOrZero (Spec.conv2dBiasDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
        (grad_output := δS))
        [out_ch.val] = sumS := by
    have hFoldS :
        (List.finRange out_h).foldl (fun acc i =>
            (List.finRange out_w).foldl (fun acc j =>
                acc + getAtOrZero δS [out_ch.val, i.val, j.val]) acc) 0 = sumS := by
      have hNested :
          (List.finRange out_h).foldl (fun acc i =>
              (List.finRange out_w).foldl (fun acc j =>
                  acc + getAtOrZero δS [out_ch.val, i.val, j.val]) acc) 0 =
            (List.finRange out_h).foldl (fun acc i =>
              (List.finRange out_w).foldl (fun acc j => acc + termS (i, j)) acc) 0 := by
        refine foldl_congr (l := List.finRange out_h)
          (f := fun acc i =>
            (List.finRange out_w).foldl (fun acc j => acc + getAtOrZero δS [out_ch.val, i.val,
              j.val]) acc)
          (g := fun acc i =>
            (List.finRange out_w).foldl (fun acc j => acc + termS (i, j)) acc)
          (init := (0 : ℝ)) ?_
        intro acc i
        refine foldl_congr (l := List.finRange out_w)
          (f := fun acc j => acc + getAtOrZero δS [out_ch.val, i.val, j.val])
          (g := fun acc j => acc + termS (i, j))
          (init := acc) ?_
        intro acc j
        simp [termS]
      simpa [hNested] using hsumS_nested
    have hFoldS' :
        (List.finRange ((inH + 2 * padding - kH) / stride + 1)).foldl (fun acc i =>
            (List.finRange ((inW + 2 * padding - kW) / stride + 1)).foldl (fun acc j =>
                acc + getAtOrZero δS [out_ch.val, i.val, j.val]) acc) 0 = sumS := by
      simpa [out_h, out_w, conv2dOutH, conv2dOutW] using hFoldS
    simpa [Spec.conv2dBiasDerivSpec, Spec.convBiasDerivSpec, Spec.Private.foldlIndices,
      Spec.Private.foldlIndices.go, Spec.convOutSpatial, Spec.convOutDim, Vector.get,
      Vector.toList_ofFn, out_ch.isLt, sumS] using hFoldS'

  have hFinal :
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd)
              (getAtOrZero (Spec.conv2dBiasDerivSpec (α := R) (layer := layerR) (input :=
                inputR) (grad_output := δR))
                [out_ch.val]) -
            getAtOrZero (Spec.conv2dBiasDerivSpec (α := ℝ) (layer := layerS) (input :=
              inputS) (grad_output := δS))
              [out_ch.val]) ≤ sumEps := by
    simpa [houtR, houtS] using hSum

  simpa [conv2dBiasPointBound, out_h, out_w, idxs, termR, epsTerm, sumEps] using hFinal

/--
Tensor-shaped bias-gradient bound.

This packages `conv2dBiasPointBound` into a `Tensor` so later `approxT` statements can use
`linfNorm` to obtain a single scalar error bound.
-/
def conv2dBiasBoundTensor
    {outC kH kW stride padding inH inW : Nat}
    (δR : Spec.MultiChannelImage outC (conv2dOutH inH kH stride padding) (conv2dOutW inW kW stride
      padding) R)
    (epsδ : ℝ) : Tensor ℝ (.dim outC .scalar) :=
  Tensor.dim (fun out_ch =>
    Tensor.scalar <| abs <|
      conv2dBiasPointBound (β := β) (fexp := fexp) (rnd := rnd)
        (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding) (inH := inH)
          (inW := inW)
        δR epsδ out_ch)

-- ---------------------------------------------------------------------------
-- Conv2D kernel gradient: pointwise bound
-- ---------------------------------------------------------------------------

/--
Pointwise error bound for the Conv2D **kernel** gradient (NF runtime vs spec).

The kernel-gradient entry accumulates products of padded input values and upstream gradients.
The bound is a replay of this accumulation with per-term errors derived from `epsX` and `epsδ`.
-/
def conv2dKernelPointBound
    {inC outC kH kW stride padding inH inW : Nat}
    (inputR : Spec.MultiChannelImage inC inH inW R)
    (δR : Spec.MultiChannelImage outC (conv2dOutH inH kH stride padding) (conv2dOutW inW kW stride
      padding) R)
    (epsX epsδ : ℝ)
    (out_ch : Fin outC) (in_ch : Fin inC) (di : Fin kH) (dj : Fin kW) : ℝ :=
  let out_h := conv2dOutH inH kH stride padding
  let out_w := conv2dOutW inW kW stride padding
  let padded_inputR :=
    if h4 : padding = 0 then
      tensorCast
        (Shape.dim inC (Shape.dim (inH + 2 * padding) (Shape.dim (inW + 2 * padding) .scalar)))
        (by simp; rw [h4])
        inputR
    else
      Spec.padMultiChannel inputR padding
  let idxs : List (Fin out_h × Fin out_w) :=
    (List.finRange out_h).flatMap (fun i => (List.finRange out_w).map (fun j => (i, j)))
  let termR : (Fin out_h × Fin out_w) → R := fun t =>
    let i := t.1
    let j := t.2
    let input_val := getAtOrZero padded_inputR [in_ch.val, i.val * stride + di.val, j.val *
      stride + dj.val]
    let grad_val := getAtOrZero δR [out_ch.val, i.val, j.val]
    input_val * grad_val
  let epsTerm : (Fin out_h × Fin out_w) → ℝ := fun t =>
    let i := t.1
    let j := t.2
    let input_val := getAtOrZero padded_inputR [in_ch.val, i.val * stride + di.val, j.val *
      stride + dj.val]
    let grad_val := getAtOrZero δR [out_ch.val, i.val, j.val]
    mulEps (β := β) (fexp := fexp) (rnd := rnd) input_val grad_val epsX epsδ
  (foldAddState (β := β) (fexp := fexp) (rnd := rnd) idxs termR epsTerm).2

/--
Soundness of the Conv2D **kernel**-gradient pointwise bound.

Given `approxT` hypotheses for the input and upstream gradient (`grad_output`), this shows the spec
kernel-gradient entry is approximated by the NF runtime entry within `conv2dKernelPointBound`.
-/
theorem approx_conv2d_kernel_point
    {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    {kernelS : Tensor ℝ (.dim outC (.dim inC (.dim kH (.dim kW .scalar))))}
    {kernelR : Tensor R (.dim outC (.dim inC (.dim kH (.dim kW .scalar))))}
    {biasS : Tensor ℝ (.dim outC .scalar)} {biasR : Tensor R (.dim outC .scalar)}
    {inputS : Spec.MultiChannelImage inC inH inW ℝ}
    {inputR : Spec.MultiChannelImage inC inH inW R}
    {δS : Spec.MultiChannelImage outC (conv2dOutH inH kH stride padding) (conv2dOutW inW kW stride
      padding) ℝ}
    {δR : Spec.MultiChannelImage outC (conv2dOutH inH kH stride padding) (conv2dOutW inW kW stride
      padding) R}
    {epsX epsδ : ℝ}
    (hX : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) inputS inputR
      epsX)
    (hδ : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) δS δR epsδ)
    (out_ch : Fin outC) (in_ch : Fin inC) (di : Fin kH) (dj : Fin kW) :
    let layerS : Spec.Conv2DSpec inC outC kH kW stride padding ℝ h1 h2 h3 :=
      { kernel := kernelS, bias := biasS }
    let layerR : Spec.Conv2DSpec inC outC kH kW stride padding R h1 h2 h3 :=
      { kernel := kernelR, bias := biasR }
    abs
        (toSpec (β := β) (fexp := fexp) (rnd := rnd)
            (getAtOrZero
              (Spec.conv2dKernelDerivSpec (α := R) (layer := layerR) (input := inputR)
                (grad_output := δR))
              [out_ch.val, in_ch.val, di.val, dj.val]) -
    getAtOrZero
      (Spec.conv2dKernelDerivSpec (α := ℝ) (layer := layerS) (input := inputS) (grad_output :=
        δS))
      [out_ch.val, in_ch.val, di.val, dj.val]) ≤
      conv2dKernelPointBound (β := β) (fexp := fexp) (rnd := rnd)
        (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
          (inH := inH) (inW := inW)
        inputR δR epsX epsδ out_ch in_ch di dj := by
  intro layerS layerR
  classical
  let out_h := conv2dOutH inH kH stride padding
  let out_w := conv2dOutW inW kW stride padding
  let paddedR :=
    if h4 : padding = 0 then
      tensorCast
        (Shape.dim inC (Shape.dim (inH + 2 * padding) (Shape.dim (inW + 2 * padding) .scalar)))
        (by simp; rw [h4])
        inputR
    else
      Spec.padMultiChannel inputR padding
  let paddedS :=
    if h4 : padding = 0 then
      tensorCast
        (Shape.dim inC (Shape.dim (inH + 2 * padding) (Shape.dim (inW + 2 * padding) .scalar)))
        (by simp; rw [h4])
        inputS
    else
      Spec.padMultiChannel inputS padding
  let idxs : List (Fin out_h × Fin out_w) :=
    (List.finRange out_h).flatMap (fun i => (List.finRange out_w).map (fun j => (i, j)))
  let termR : (Fin out_h × Fin out_w) → R := fun t =>
    let i := t.1; let j := t.2
    let input_val := getAtOrZero paddedR [in_ch.val, i.val * stride + di.val, j.val * stride +
      dj.val]
    let grad_val := getAtOrZero δR [out_ch.val, i.val, j.val]
    input_val * grad_val
  let termS : (Fin out_h × Fin out_w) → ℝ := fun t =>
    let i := t.1; let j := t.2
    let input_val := getAtOrZero paddedS [in_ch.val, i.val * stride + di.val, j.val * stride +
      dj.val]
    let grad_val := getAtOrZero δS [out_ch.val, i.val, j.val]
    input_val * grad_val
  let epsTerm : (Fin out_h × Fin out_w) → ℝ := fun t =>
    let i := t.1; let j := t.2
    let input_val := getAtOrZero paddedR [in_ch.val, i.val * stride + di.val, j.val * stride +
      dj.val]
    let grad_val := getAtOrZero δR [out_ch.val, i.val, j.val]
    mulEps (β := β) (fexp := fexp) (rnd := rnd) input_val grad_val epsX epsδ
  let sumR : R := idxs.foldl (fun acc t => acc + termR t) 0
  let sumS : ℝ := idxs.foldl (fun acc t => acc + termS t) 0
  let sumEps : ℝ := (foldAddState (β := β) (fexp := fexp) (rnd := rnd) idxs termR epsTerm).2

  have hInputVal :
      ∀ (p q : Nat),
        abs
            (toSpec (β := β) (fexp := fexp) (rnd := rnd) (getAtOrZero paddedR [in_ch.val, p, q])
              -
              getAtOrZero paddedS [in_ch.val, p, q]) ≤ epsX := by
    intro p q
    simpa [paddedR, paddedS] using
      (approx_padded_input_read (β := β) (fexp := fexp) (rnd := rnd)
        (inC := inC) (inH := inH) (inW := inW) (padding := padding) (xS := inputS) (xR := inputR)
          (epsX := epsX) hX in_ch p q)

  have hGradVal :
      ∀ (i : Fin out_h) (j : Fin out_w),
        abs
            (toSpec (β := β) (fexp := fexp) (rnd := rnd) (getAtOrZero δR [out_ch.val, i.val,
              j.val]) -
              getAtOrZero δS [out_ch.val, i.val, j.val]) ≤ epsδ := by
    intro i j
    simpa using
      (approx_get_at_or_zero (β := β) (fexp := fexp) (rnd := rnd)
        (s := .dim outC (.dim out_h (.dim out_w .scalar)))
        (xS := δS) (xR := δR) (eps := epsδ) hδ [out_ch.val, i.val, j.val])

  have hTermIdx :
      ∀ t ∈ idxs, abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (termR t) - termS t) ≤ epsTerm t
        := by
    intro t _ht
    rcases t with ⟨i, j⟩
    have hx := hInputVal (i.val * stride + di.val) (j.val * stride + dj.val)
    have hy := hGradVal i j
    have :=
      approx_mul_nf (β := β) (fexp := fexp) (rnd := rnd)
        (x := getAtOrZero paddedS [in_ch.val, i.val * stride + di.val, j.val * stride + dj.val])
        (y := getAtOrZero δS [out_ch.val, i.val, j.val])
        (xR := getAtOrZero paddedR [in_ch.val, i.val * stride + di.val, j.val * stride + dj.val])
        (yR := getAtOrZero δR [out_ch.val, i.val, j.val])
        (epsx := epsX) (epsy := epsδ) hx hy
    simpa [termR, termS, epsTerm, mulEps] using this

  have hSum :
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) sumR - sumS) ≤ sumEps := by
    simpa [sumR, sumS, sumEps] using
      (approx_fold_add (β := β) (fexp := fexp) (rnd := rnd)
        (l := idxs) (termS := termS) (termR := termR) (epsTerm := epsTerm) hTermIdx)

  have hsumR_nested :
      (List.finRange out_h).foldl (fun acc i =>
          (List.finRange out_w).foldl (fun acc j => acc + termR (i, j)) acc) 0 = sumR := by
    let fR : R → (Fin out_h × Fin out_w) → R := fun acc t => acc + termR t
    have hW :
        ∀ (acc : R) (i : Fin out_h),
          List.foldl fR acc ((List.finRange out_w).map (fun j => (i, j))) =
            (List.finRange out_w).foldl (fun acc j => acc + termR (i, j)) acc := by
      intro acc i
      exact
        (List.foldl_map (f := fun j => (i, j)) (g := fR) (l := List.finRange out_w) (init := acc))
    have hC :
        idxs.foldl fR (0 : R) =
          (List.finRange out_h).foldl (fun acc i =>
            List.foldl fR acc ((List.finRange out_w).map (fun j => (i, j)))) 0 := by
      simpa [idxs] using
        (foldl_flatMap (l := List.finRange out_h) (g := fun i => (List.finRange out_w).map (fun j =>
          (i, j)))
          (f := fR) (init := (0 : R)))
    have hC' :
        idxs.foldl fR 0 =
          (List.finRange out_h).foldl (fun acc i => (List.finRange out_w).foldl (fun acc j => acc +
            termR (i, j)) acc) 0 := by
      have :=
        foldl_congr (l := List.finRange out_h)
          (f := fun acc i => List.foldl fR acc ((List.finRange out_w).map (fun j => (i, j))))
          (g := fun acc i => (List.finRange out_w).foldl (fun acc j => acc + termR (i, j)) acc)
          (init := (0 : R)) (by intro acc i; simpa using (hW acc i))
      simpa [hC] using this
    simpa [sumR, fR] using hC'.symm

  have hsumS_nested :
      (List.finRange out_h).foldl (fun acc i =>
          (List.finRange out_w).foldl (fun acc j => acc + termS (i, j)) acc) 0 = sumS := by
    let fS : ℝ → (Fin out_h × Fin out_w) → ℝ := fun acc t => acc + termS t
    have hW :
        ∀ (acc : ℝ) (i : Fin out_h),
          List.foldl fS acc ((List.finRange out_w).map (fun j => (i, j))) =
            (List.finRange out_w).foldl (fun acc j => acc + termS (i, j)) acc := by
      intro acc i
      exact
        (List.foldl_map (f := fun j => (i, j)) (g := fS) (l := List.finRange out_w) (init := acc))
    have hC :
        idxs.foldl fS (0 : ℝ) =
          (List.finRange out_h).foldl (fun acc i =>
            List.foldl fS acc ((List.finRange out_w).map (fun j => (i, j)))) 0 := by
      simpa [idxs] using
        (foldl_flatMap (l := List.finRange out_h) (g := fun i => (List.finRange out_w).map (fun j =>
          (i, j)))
          (f := fS) (init := (0 : ℝ)))
    have hC' :
        idxs.foldl fS 0 =
          (List.finRange out_h).foldl (fun acc i => (List.finRange out_w).foldl (fun acc j => acc +
            termS (i, j)) acc) 0 := by
      have :=
        foldl_congr (l := List.finRange out_h)
          (f := fun acc i => List.foldl fS acc ((List.finRange out_w).map (fun j => (i, j))))
          (g := fun acc i => (List.finRange out_w).foldl (fun acc j => acc + termS (i, j)) acc)
          (init := (0 : ℝ)) (by intro acc i; simpa using (hW acc i))
      simpa [hC] using this
    simpa [sumS, fS] using hC'.symm

  have hFoldR :
      (List.finRange out_h).foldl (fun acc i =>
          (List.finRange out_w).foldl (fun acc j =>
              acc +
                getAtOrZero paddedR [in_ch.val, i.val * stride + di.val, j.val * stride + dj.val]
                  *
                  getAtOrZero δR [out_ch.val, i.val, j.val]) acc) 0 = sumR := by
    have hNested :
        (List.finRange out_h).foldl (fun acc i =>
            (List.finRange out_w).foldl (fun acc j =>
                acc +
                  getAtOrZero paddedR [in_ch.val, i.val * stride + di.val, j.val * stride +
                    dj.val] *
                    getAtOrZero δR [out_ch.val, i.val, j.val]) acc) 0 =
          (List.finRange out_h).foldl (fun acc i =>
            (List.finRange out_w).foldl (fun acc j => acc + termR (i, j)) acc) 0 := by
      refine foldl_congr (l := List.finRange out_h)
        (f := fun acc i =>
          (List.finRange out_w).foldl (fun acc j =>
              acc +
                getAtOrZero paddedR [in_ch.val, i.val * stride + di.val, j.val * stride + dj.val]
                  *
                  getAtOrZero δR [out_ch.val, i.val, j.val]) acc)
        (g := fun acc i =>
          (List.finRange out_w).foldl (fun acc j => acc + termR (i, j)) acc)
        (init := (0 : R)) ?_
      intro acc i
      refine foldl_congr (l := List.finRange out_w)
        (f := fun acc j =>
          acc +
            getAtOrZero paddedR [in_ch.val, i.val * stride + di.val, j.val * stride + dj.val] *
              getAtOrZero δR [out_ch.val, i.val, j.val])
        (g := fun acc j => acc + termR (i, j))
        (init := acc) ?_
      intro acc j
      simp [termR]
    simpa [hNested] using hsumR_nested

  have hFoldS :
      (List.finRange out_h).foldl (fun acc i =>
          (List.finRange out_w).foldl (fun acc j =>
              acc +
                getAtOrZero paddedS [in_ch.val, i.val * stride + di.val, j.val * stride + dj.val]
                  *
                  getAtOrZero δS [out_ch.val, i.val, j.val]) acc) 0 = sumS := by
    have hNested :
        (List.finRange out_h).foldl (fun acc i =>
            (List.finRange out_w).foldl (fun acc j =>
                acc +
                  getAtOrZero paddedS [in_ch.val, i.val * stride + di.val, j.val * stride +
                    dj.val] *
                    getAtOrZero δS [out_ch.val, i.val, j.val]) acc) 0 =
          (List.finRange out_h).foldl (fun acc i =>
            (List.finRange out_w).foldl (fun acc j => acc + termS (i, j)) acc) 0 := by
      refine foldl_congr (l := List.finRange out_h)
        (f := fun acc i =>
          (List.finRange out_w).foldl (fun acc j =>
              acc +
                getAtOrZero paddedS [in_ch.val, i.val * stride + di.val, j.val * stride + dj.val]
                  *
                  getAtOrZero δS [out_ch.val, i.val, j.val]) acc)
        (g := fun acc i =>
          (List.finRange out_w).foldl (fun acc j => acc + termS (i, j)) acc)
        (init := (0 : ℝ)) ?_
      intro acc i
      refine foldl_congr (l := List.finRange out_w)
        (f := fun acc j =>
          acc +
            getAtOrZero paddedS [in_ch.val, i.val * stride + di.val, j.val * stride + dj.val] *
              getAtOrZero δS [out_ch.val, i.val, j.val])
        (g := fun acc j => acc + termS (i, j))
        (init := acc) ?_
      intro acc j
      simp [termS]
    simpa [hNested] using hsumS_nested

  have houtR :
      getAtOrZero (Spec.conv2dKernelDerivSpec (α := R) (layer := layerR) (input := inputR)
        (grad_output := δR))
          [out_ch.val, in_ch.val, di.val, dj.val] = sumR := by
    have hFoldR' :
        (List.finRange ((inH + 2 * padding - kH) / stride + 1)).foldl (fun acc i =>
            (List.finRange ((inW + 2 * padding - kW) / stride + 1)).foldl (fun acc j =>
                acc +
                  getAtOrZero
                      (if h4 : padding = 0 then
                        tensorCast
                          (Shape.dim inC (Shape.dim (inH + 2 * padding) (Shape.dim (inW + 2 *
                            padding) Shape.scalar)))
                          (by simp; rw [h4])
                          inputR
                      else padMultiChannel inputR padding)
                      [in_ch.val, i.val * stride + di.val, j.val * stride + dj.val] *
                    getAtOrZero δR [out_ch.val, i.val, j.val]) acc) 0 = sumR := by
      simpa [out_h, out_w, conv2dOutH, conv2dOutW, paddedR] using hFoldR
    have hGet :
        getAtOrZero (Spec.conv2dKernelDerivSpec (α := R) (layer := layerR) (input := inputR)
          (grad_output := δR))
            [out_ch.val, in_ch.val, di.val, dj.val] =
          (List.finRange ((inH + 2 * padding - kH) / stride + 1)).foldl (fun acc i =>
              (List.finRange ((inW + 2 * padding - kW) / stride + 1)).foldl (fun acc j =>
                  acc +
                    getAtOrZero
                        (if h4 : padding = 0 then
                          tensorCast
                            (Shape.dim inC (Shape.dim (inH + 2 * padding) (Shape.dim (inW + 2 *
                              padding) Shape.scalar)))
                            (by simp; rw [h4])
                            inputR
                        else padMultiChannel inputR padding)
                        [in_ch.val, i.val * stride + di.val, j.val * stride + dj.val] *
                        getAtOrZero δR [out_ch.val, i.val, j.val]) acc) 0 := by
        simpa [Spec.conv2dKernelDerivSpec, conv2dOutH, conv2dOutW, paddedInput,
          tensor_cast_eq_cast_shape, get_at_or_zero_tensor_cast, out_ch.isLt, in_ch.isLt,
          di.isLt, dj.isLt] using
          (conv2dKernelFoldRead_eq_paddedFold (input := inputR) (grad := δR)
            (out_ch := out_ch) (in_ch := in_ch) (di := di.val) (dj := dj.val)
            (stride := stride) (padding := padding))
    exact hGet.trans hFoldR'

  have houtS :
      getAtOrZero (Spec.conv2dKernelDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
        (grad_output := δS))
          [out_ch.val, in_ch.val, di.val, dj.val] = sumS := by
    have hFoldS' :
        (List.finRange ((inH + 2 * padding - kH) / stride + 1)).foldl (fun acc i =>
            (List.finRange ((inW + 2 * padding - kW) / stride + 1)).foldl (fun acc j =>
                acc +
                  getAtOrZero
                      (if h4 : padding = 0 then
                        tensorCast
                          (Shape.dim inC (Shape.dim (inH + 2 * padding) (Shape.dim (inW + 2 *
                            padding) Shape.scalar)))
                          (by simp; rw [h4])
                          inputS
                      else padMultiChannel inputS padding)
                      [in_ch.val, i.val * stride + di.val, j.val * stride + dj.val] *
                    getAtOrZero δS [out_ch.val, i.val, j.val]) acc) 0 = sumS := by
      simpa [out_h, out_w, conv2dOutH, conv2dOutW, paddedS] using hFoldS
    have hGet :
        getAtOrZero (Spec.conv2dKernelDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
          (grad_output := δS))
            [out_ch.val, in_ch.val, di.val, dj.val] =
          (List.finRange ((inH + 2 * padding - kH) / stride + 1)).foldl (fun acc i =>
              (List.finRange ((inW + 2 * padding - kW) / stride + 1)).foldl (fun acc j =>
                  acc +
                    getAtOrZero
                        (if h4 : padding = 0 then
                          tensorCast
                            (Shape.dim inC (Shape.dim (inH + 2 * padding) (Shape.dim (inW + 2 *
                              padding) Shape.scalar)))
                            (by simp; rw [h4])
                            inputS
                        else padMultiChannel inputS padding)
                        [in_ch.val, i.val * stride + di.val, j.val * stride + dj.val] *
                        getAtOrZero δS [out_ch.val, i.val, j.val]) acc) 0 := by
        simpa [Spec.conv2dKernelDerivSpec, conv2dOutH, conv2dOutW, paddedInput,
          tensor_cast_eq_cast_shape, get_at_or_zero_tensor_cast, out_ch.isLt, in_ch.isLt,
          di.isLt, dj.isLt] using
          (conv2dKernelFoldRead_eq_paddedFold (input := inputS) (grad := δS)
            (out_ch := out_ch) (in_ch := in_ch) (di := di.val) (dj := dj.val)
            (stride := stride) (padding := padding))
    exact hGet.trans hFoldS'

  have hFinal :
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd)
              (getAtOrZero
                (Spec.conv2dKernelDerivSpec (α := R) (layer := layerR) (input := inputR)
                  (grad_output := δR))
                [out_ch.val, in_ch.val, di.val, dj.val]) -
            getAtOrZero
              (Spec.conv2dKernelDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
                (grad_output := δS))
              [out_ch.val, in_ch.val, di.val, dj.val]) ≤ sumEps := by
    simpa [houtR, houtS] using hSum

  simpa [conv2dKernelPointBound, out_h, out_w, idxs, termR, epsTerm, sumEps, paddedR] using
    hFinal

/--
Tensor-shaped kernel-gradient bound.

This packages `conv2dKernelPointBound` into the full 4D kernel-tensor shape so later `approxT`
lemmas can use a single global bound via `linfNorm`.
-/
def conv2dKernelBoundTensor
    {inC outC kH kW stride padding inH inW : Nat}
    (inputR : Spec.MultiChannelImage inC inH inW R)
    (δR : Spec.MultiChannelImage outC (conv2dOutH inH kH stride padding) (conv2dOutW inW kW stride
      padding) R)
    (epsX epsδ : ℝ) :
    Tensor ℝ (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))) :=
  Tensor.dim (fun out_ch =>
    Tensor.dim (fun in_ch =>
      Tensor.dim (fun di =>
        Tensor.dim (fun dj =>
          Tensor.scalar <| abs <|
            conv2dKernelPointBound (β := β) (fexp := fexp) (rnd := rnd)
              (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding :=
                padding) (inH := inH) (inW := inW)
              inputR δR epsX epsδ out_ch in_ch di dj))))

-- ---------------------------------------------------------------------------
-- Tensor-level backward bounds (kernel + bias)
-- ---------------------------------------------------------------------------

/--
Tensor-level `approxT` bound for the Conv2D **bias** gradient.

This lifts `approx_conv2d_bias_point` entrywise and packages the error into
`linfNorm (conv2dBiasBoundTensor ...)`.
-/
theorem approxT_conv2d_bias_deriv_spec
    {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    {kernelS : Tensor ℝ (.dim outC (.dim inC (.dim kH (.dim kW .scalar))))}
    {kernelR : Tensor R (.dim outC (.dim inC (.dim kH (.dim kW .scalar))))}
    {biasS : Tensor ℝ (.dim outC .scalar)} {biasR : Tensor R (.dim outC .scalar)}
    {inputS : Spec.MultiChannelImage inC inH inW ℝ}
    {inputR : Spec.MultiChannelImage inC inH inW R}
    {δS : Spec.MultiChannelImage outC (conv2dOutH inH kH stride padding) (conv2dOutW inW kW stride
      padding) ℝ}
    {δR : Spec.MultiChannelImage outC (conv2dOutH inH kH stride padding) (conv2dOutW inW kW stride
      padding) R}
    {epsδ : ℝ}
    (hδ : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) δS δR epsδ) :
    let layerS : Spec.Conv2DSpec inC outC kH kW stride padding ℝ h1 h2 h3 :=
      { kernel := kernelS, bias := biasS }
    let layerR : Spec.Conv2DSpec inC outC kH kW stride padding R h1 h2 h3 :=
      { kernel := kernelR, bias := biasR }
    let outS := Spec.conv2dBiasDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
      (grad_output := δS)
    let outR := Spec.conv2dBiasDerivSpec (α := R) (layer := layerR) (input := inputR)
      (grad_output := δR)
    let bT :=
      conv2dBiasBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
        (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding) (inH := inH)
            (inW := inW)
          δR epsδ
    approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) outS outR (linfNorm
      bT) := by
  intro layerS layerR outS outR bT
  classical
  have hε : 0 ≤ linfNorm bT := linf_norm_nonneg (t := bT)
  refine approxT_dim_of_forall (β := β) (fexp := fexp) (rnd := rnd) (n := outC) (s := .scalar)
    (xS := outS) (xR := outR) (eps := linfNorm bT) hε ?_
  intro oc
  have hpt :=
    approx_conv2d_bias_point (β := β) (fexp := fexp) (rnd := rnd)
      (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding) (inH
        := inH) (inW := inW)
      (h1 := h1) (h2 := h2) (h3 := h3)
      (kernelS := kernelS) (kernelR := kernelR) (biasS := biasS) (biasR := biasR)
      (inputS := inputS) (inputR := inputR) (δS := δS) (δR := δR) (epsδ := epsδ) hδ oc
  have hEntry :
      abs (conv2dBiasPointBound (β := β) (fexp := fexp) (rnd := rnd)
          (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding) (inH := inH)
            (inW := inW)
          δR epsδ oc) ≤ linfNorm bT := by
    have hcoord := (linf_norm_le_get_dim (t := bT) oc)
    let bound :=
      conv2dBiasPointBound (β := β) (fexp := fexp) (rnd := rnd)
        (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding) (inH := inH)
          (inW := inW)
        δR epsδ oc
    have hdouble' : (MathFunctions.abs (abs bound) : ℝ) ≤ linfNorm bT := by
      simpa [bound, bT, conv2dBiasBoundTensor, linfNorm, RuntimeApprox.linfNorm,
        tensorLinfNorm] using hcoord
    have habs : (MathFunctions.abs (abs bound) : ℝ) = abs (abs bound) := by
      rfl
    have hdouble : abs (abs bound) ≤ linfNorm bT := by
      simpa [habs] using hdouble'
    simpa [bound, abs_abs] using hdouble

  have hscalar :
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd)
              (getAtOrZero outR [oc.val]) -
            getAtOrZero outS [oc.val]) ≤ linfNorm bT :=
    le_trans (le_trans hpt (le_abs_self _)) hEntry

  let entryS : Tensor ℝ .scalar := (match outS with | .dim f => f oc)
  let entryR : Tensor R .scalar := (match outR with | .dim f => f oc)
  change
    approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) entryS entryR
      (linfNorm bT)
  have hEntryS :
      entryS = Tensor.scalar (getAtOrZero outS [oc.val]) := by
    simpa [entryS] using (entry_eq_scalar_get_at_or_zero1 (t := outS) oc)
  have hEntryR :
      entryR = Tensor.scalar (getAtOrZero outR [oc.val]) := by
    simpa [entryR] using (entry_eq_scalar_get_at_or_zero1 (t := outR) oc)
  have happ :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Tensor.scalar (getAtOrZero outS [oc.val]))
        (Tensor.scalar (getAtOrZero outR [oc.val]))
        (linfNorm bT) :=
    (approxT_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (x := getAtOrZero outS [oc.val]) (xR := getAtOrZero outR [oc.val]) (eps := linfNorm
          bT)).2 (by
          simpa using hscalar)
  simpa [hEntryS, hEntryR] using happ

/--
Tensor-level `approxT` bound for the Conv2D **kernel** gradient.

This lifts `approx_conv2d_kernel_point` entrywise and packages the error into
`linfNorm (conv2dKernelBoundTensor ...)`.
-/
theorem approxT_conv2d_kernel_deriv_spec
    {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    {kernelS : Tensor ℝ (.dim outC (.dim inC (.dim kH (.dim kW .scalar))))}
    {kernelR : Tensor R (.dim outC (.dim inC (.dim kH (.dim kW .scalar))))}
    {biasS : Tensor ℝ (.dim outC .scalar)} {biasR : Tensor R (.dim outC .scalar)}
    {inputS : Spec.MultiChannelImage inC inH inW ℝ}
    {inputR : Spec.MultiChannelImage inC inH inW R}
    {δS : Spec.MultiChannelImage outC (conv2dOutH inH kH stride padding) (conv2dOutW inW kW stride
      padding) ℝ}
    {δR : Spec.MultiChannelImage outC (conv2dOutH inH kH stride padding) (conv2dOutW inW kW stride
      padding) R}
    {epsX epsδ : ℝ}
    (hX : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) inputS inputR
      epsX)
    (hδ : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) δS δR epsδ) :
    let layerS : Spec.Conv2DSpec inC outC kH kW stride padding ℝ h1 h2 h3 :=
      { kernel := kernelS, bias := biasS }
    let layerR : Spec.Conv2DSpec inC outC kH kW stride padding R h1 h2 h3 :=
      { kernel := kernelR, bias := biasR }
    let outS := Spec.conv2dKernelDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
      (grad_output := δS)
    let outR := Spec.conv2dKernelDerivSpec (α := R) (layer := layerR) (input := inputR)
      (grad_output := δR)
    let bT :=
        conv2dKernelBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
            (inH := inH) (inW := inW)
          inputR δR epsX epsδ
    approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) outS outR (linfNorm
      bT) := by
  intro layerS layerR outS outR bT
  classical
  have hε : 0 ≤ linfNorm bT := linf_norm_nonneg (t := bT)
  refine approxT_dim_of_forall (β := β) (fexp := fexp) (rnd := rnd) (n := outC)
    (s := .dim inC (.dim kH (.dim kW .scalar)))
    (xS := outS) (xR := outR) (eps := linfNorm bT) hε ?_
  intro oc
  refine approxT_dim_of_forall (β := β) (fexp := fexp) (rnd := rnd) (n := inC)
    (s := .dim kH (.dim kW .scalar))
    (xS := (match outS with | .dim f => f oc)) (xR := (match outR with | .dim f => f oc))
    (eps := linfNorm bT) hε ?_
  intro ic
  refine approxT_dim_of_forall (β := β) (fexp := fexp) (rnd := rnd) (n := kH)
    (s := .dim kW .scalar)
    (xS := (match (match outS with | .dim f => f oc) with | .dim g => g ic))
    (xR := (match (match outR with | .dim f => f oc) with | .dim g => g ic))
    (eps := linfNorm bT) hε ?_
  intro di
  refine approxT_dim_of_forall (β := β) (fexp := fexp) (rnd := rnd) (n := kW)
    (s := .scalar)
    (xS :=
      (match
        match
          match outS with
          | .dim f => f oc
        with
        | .dim g => g ic
      with
      | .dim h => h di))
    (xR :=
      (match
        match
          match outR with
          | .dim f => f oc
        with
        | .dim g => g ic
      with
      | .dim h => h di))
    (eps := linfNorm bT) hε ?_
  intro dj

  have hpt :=
    approx_conv2d_kernel_point (β := β) (fexp := fexp) (rnd := rnd)
      (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding) (inH
        := inH) (inW := inW)
      (h1 := h1) (h2 := h2) (h3 := h3)
      (kernelS := kernelS) (kernelR := kernelR) (biasS := biasS) (biasR := biasR)
      (inputS := inputS) (inputR := inputR) (δS := δS) (δR := δR)
      (epsX := epsX) (epsδ := epsδ) hX hδ oc ic di dj

  have hEntry :
      abs (conv2dKernelPointBound (β := β) (fexp := fexp) (rnd := rnd)
          (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
            (inH := inH) (inW := inW)
          inputR δR epsX epsδ oc ic di dj) ≤ linfNorm bT := by
    let bToc := match bT with | .dim f => f oc
    let bTocic := match bToc with | .dim f => f ic
    let bTocicdi := match bTocic with | .dim f => f di
    let bTocicdidj := match bTocicdi with | .dim f => f dj
    have h0 : linfNorm bToc ≤ linfNorm bT := by
      simpa [bToc] using (linf_norm_le_get_dim (t := bT) oc)
    have h1' : linfNorm bTocic ≤ linfNorm bToc := by
      simpa [bTocic] using (linf_norm_le_get_dim (t := bToc) ic)
    have h2' : linfNorm bTocicdi ≤ linfNorm bTocic := by
      simpa [bTocicdi] using (linf_norm_le_get_dim (t := bTocic) di)
    have h3' : linfNorm bTocicdidj ≤ linfNorm bTocicdi := by
      simpa [bTocicdidj] using (linf_norm_le_get_dim (t := bTocicdi) dj)
    have hchain : linfNorm bTocicdidj ≤ linfNorm bT :=
      le_trans (le_trans (le_trans h3' h2') h1') h0
    let bound :=
      conv2dKernelPointBound (β := β) (fexp := fexp) (rnd := rnd)
        (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
          (inH := inH) (inW := inW)
        inputR δR epsX epsδ oc ic di dj
    have hdouble' : (MathFunctions.abs (abs bound) : ℝ) ≤ linfNorm bT := by
      simpa [bound, bT, bToc, bTocic, bTocicdi, bTocicdidj, conv2dKernelBoundTensor, linfNorm,
        RuntimeApprox.linfNorm, tensorLinfNorm] using hchain
    have habs : (MathFunctions.abs (abs bound) : ℝ) = abs (abs bound) := by
      rfl
    have hdouble : abs (abs bound) ≤ linfNorm bT := by
      simpa [habs] using hdouble'
    simpa [bound, abs_abs] using hdouble

  have hscalar :
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd)
              (getAtOrZero outR [oc.val, ic.val, di.val, dj.val]) -
            getAtOrZero outS [oc.val, ic.val, di.val, dj.val]) ≤ linfNorm bT :=
    le_trans (le_trans hpt (le_abs_self _)) hEntry

  let entryS : Tensor ℝ .scalar :=
    (match
      match
        match
          match outS with
          | .dim f => f oc with
        | .dim g => g ic with
      | .dim h => h di with
    | .dim k => k dj)
  let entryR : Tensor R .scalar :=
    (match
      match
        match
          match outR with
          | .dim f => f oc with
        | .dim g => g ic with
      | .dim h => h di with
    | .dim k => k dj)

  change
    approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) entryS entryR
      (linfNorm bT)
  have hEntryS :
      entryS = Tensor.scalar (getAtOrZero outS [oc.val, ic.val, di.val, dj.val]) := by
    simpa [entryS] using (entry_eq_scalar_get_at_or_zero4 (t := outS) oc ic di dj)
  have hEntryR :
      entryR = Tensor.scalar (getAtOrZero outR [oc.val, ic.val, di.val, dj.val]) := by
    simpa [entryR] using (entry_eq_scalar_get_at_or_zero4 (t := outR) oc ic di dj)

  have happ :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Tensor.scalar (getAtOrZero outS [oc.val, ic.val, di.val, dj.val]))
        (Tensor.scalar (getAtOrZero outR [oc.val, ic.val, di.val, dj.val]))
        (linfNorm bT) :=
    (approxT_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (x := getAtOrZero outS [oc.val, ic.val, di.val, dj.val])
        (xR := getAtOrZero outR [oc.val, ic.val, di.val, dj.val])
        (eps := linfNorm bT)).2 (by
          simpa using hscalar)

  simpa [hEntryS, hEntryR] using happ

-- ---------------------------------------------------------------------------
-- Conv2D input gradient: pointwise bound
-- ---------------------------------------------------------------------------

/--
Pointwise error bound for the Conv2D **input** gradient (NF runtime vs spec).

The input-gradient entry accumulates contributions from all output channels and spatial positions
that “hit” the given input coordinate under the stride/padding relation. The bound is a replay of
that accumulation with per-term errors derived from `epsK` and `epsδ`.
-/
def conv2dInputPointBound
    {inC outC kH kW stride padding inH inW : Nat}
    (kernelR : Tensor R (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))))
    (δR : Spec.MultiChannelImage outC (conv2dOutH inH kH stride padding) (conv2dOutW inW kW stride
      padding) R)
    (epsK epsδ : ℝ)
    (in_ch : Fin inC) (i : Fin inH) (j : Fin inW) : ℝ :=
  let out_h := conv2dOutH inH kH stride padding
  let out_w := conv2dOutW inW kW stride padding
  let idxs : List (Fin outC × Fin out_h × Fin out_w × Fin kH × Fin kW) :=
    (List.finRange outC).flatMap (fun out_ch =>
      (List.finRange out_h).flatMap (fun out_i =>
        (List.finRange out_w).flatMap (fun out_j =>
          (List.finRange kH).flatMap (fun di =>
            (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj))))))
  let termR : (Fin outC × Fin out_h × Fin out_w × Fin kH × Fin kW) → R := fun t =>
    let out_ch := t.1
    let out_i := t.2.1
    let out_j := t.2.2.1
    let di := t.2.2.2.1
    let dj := t.2.2.2.2
      if _ :
          (out_i.val * stride + di.val = i.val + padding) ∧
          (out_j.val * stride + dj.val = j.val + padding) then
      let grad_val := getAtOrZero δR [out_ch.val, out_i.val, out_j.val]
      let kernel_val := getAtOrZero kernelR [out_ch.val, in_ch.val, di.val, dj.val]
      grad_val * kernel_val
    else
      0
  let epsTerm : (Fin outC × Fin out_h × Fin out_w × Fin kH × Fin kW) → ℝ := fun t =>
    let out_ch := t.1
    let out_i := t.2.1
    let out_j := t.2.2.1
    let di := t.2.2.2.1
    let dj := t.2.2.2.2
      if _ :
          (out_i.val * stride + di.val = i.val + padding) ∧
          (out_j.val * stride + dj.val = j.val + padding) then
      let grad_val := getAtOrZero δR [out_ch.val, out_i.val, out_j.val]
      let kernel_val := getAtOrZero kernelR [out_ch.val, in_ch.val, di.val, dj.val]
      mulEps (β := β) (fexp := fexp) (rnd := rnd) grad_val kernel_val epsδ epsK
    else
      0
  (foldAddState (β := β) (fexp := fexp) (rnd := rnd) idxs termR epsTerm).2

/--
Tensor-shaped input-gradient bound.

This packages `conv2dInputPointBound` into the full input image shape so later `approxT` lemmas
can use a single global bound via `linfNorm`.
-/
def conv2dInputBoundTensor
    {inC outC kH kW stride padding inH inW : Nat}
    (kernelR : Tensor R (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))))
    (δR : Spec.MultiChannelImage outC (conv2dOutH inH kH stride padding) (conv2dOutW inW kW stride
      padding) R)
    (epsK epsδ : ℝ) :
    Spec.MultiChannelImage inC inH inW ℝ :=
  Tensor.dim (fun in_ch =>
    Tensor.dim (fun i =>
      Tensor.dim (fun j =>
        Tensor.scalar <| abs <|
          conv2dInputPointBound (β := β) (fexp := fexp) (rnd := rnd)
            (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding :=
              padding) (inH := inH) (inW := inW)
            kernelR δR epsK epsδ in_ch i j)))

/--
Soundness of the Conv2D **input**-gradient pointwise bound.

Given `approxT` hypotheses for the kernel and upstream gradient (`grad_output`), this shows the spec
input-gradient entry is approximated by the NF runtime entry within `conv2dInputPointBound`.
-/
theorem approx_conv2d_input_point
    {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    {kernelS : Tensor ℝ (.dim outC (.dim inC (.dim kH (.dim kW .scalar))))}
    {kernelR : Tensor R (.dim outC (.dim inC (.dim kH (.dim kW .scalar))))}
    {biasS : Tensor ℝ (.dim outC .scalar)} {biasR : Tensor R (.dim outC .scalar)}
    {inputS : Spec.MultiChannelImage inC inH inW ℝ}
    {inputR : Spec.MultiChannelImage inC inH inW R}
    {δS : Spec.MultiChannelImage outC (conv2dOutH inH kH stride padding) (conv2dOutW inW kW stride
      padding) ℝ}
    {δR : Spec.MultiChannelImage outC (conv2dOutH inH kH stride padding) (conv2dOutW inW kW stride
      padding) R}
    {epsK epsδ : ℝ}
    (hK : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) kernelS kernelR
      epsK)
    (hδ : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) δS δR epsδ)
    (in_ch : Fin inC) (i : Fin inH) (j : Fin inW) :
    let layerS : Spec.Conv2DSpec inC outC kH kW stride padding ℝ h1 h2 h3 :=
      { kernel := kernelS, bias := biasS }
    let layerR : Spec.Conv2DSpec inC outC kH kW stride padding R h1 h2 h3 :=
      { kernel := kernelR, bias := biasR }
    abs
        (toSpec (β := β) (fexp := fexp) (rnd := rnd)
            (getAtOrZero
              (Spec.conv2dInputDerivSpec (α := R) (layer := layerR) (input := inputR)
                (grad_output := δR))
              [in_ch.val, i.val, j.val]) -
    getAtOrZero
      (Spec.conv2dInputDerivSpec (α := ℝ) (layer := layerS) (input := inputS) (grad_output :=
        δS))
      [in_ch.val, i.val, j.val]) ≤
      conv2dInputPointBound (β := β) (fexp := fexp) (rnd := rnd)
        (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
          (inH := inH) (inW := inW)
        kernelR δR epsK epsδ in_ch i j := by
  intro layerS layerR
  classical
  let out_h := conv2dOutH inH kH stride padding
  let out_w := conv2dOutW inW kW stride padding
  let idxs : List (Fin outC × Fin out_h × Fin out_w × Fin kH × Fin kW) :=
    (List.finRange outC).flatMap (fun out_ch =>
      (List.finRange out_h).flatMap (fun out_i =>
        (List.finRange out_w).flatMap (fun out_j =>
          (List.finRange kH).flatMap (fun di =>
            (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj))))))
  let termR : (Fin outC × Fin out_h × Fin out_w × Fin kH × Fin kW) → R := fun t =>
    let out_ch := t.1
    let out_i := t.2.1
    let out_j := t.2.2.1
    let di := t.2.2.2.1
    let dj := t.2.2.2.2
    if h :
        (out_i.val * stride + di.val = i.val + padding) ∧
        (out_j.val * stride + dj.val = j.val + padding) then
      let grad_val := getAtOrZero δR [out_ch.val, out_i.val, out_j.val]
      let kernel_val := getAtOrZero kernelR [out_ch.val, in_ch.val, di.val, dj.val]
      grad_val * kernel_val
    else
      0
  let termS : (Fin outC × Fin out_h × Fin out_w × Fin kH × Fin kW) → ℝ := fun t =>
    let out_ch := t.1
    let out_i := t.2.1
    let out_j := t.2.2.1
    let di := t.2.2.2.1
    let dj := t.2.2.2.2
    if h :
        (out_i.val * stride + di.val = i.val + padding) ∧
        (out_j.val * stride + dj.val = j.val + padding) then
      let grad_val := getAtOrZero δS [out_ch.val, out_i.val, out_j.val]
      let kernel_val := getAtOrZero kernelS [out_ch.val, in_ch.val, di.val, dj.val]
      grad_val * kernel_val
    else
      0
  let epsTerm : (Fin outC × Fin out_h × Fin out_w × Fin kH × Fin kW) → ℝ := fun t =>
    let out_ch := t.1
    let out_i := t.2.1
    let out_j := t.2.2.1
    let di := t.2.2.2.1
    let dj := t.2.2.2.2
    if h :
        (out_i.val * stride + di.val = i.val + padding) ∧
        (out_j.val * stride + dj.val = j.val + padding) then
      let grad_val := getAtOrZero δR [out_ch.val, out_i.val, out_j.val]
      let kernel_val := getAtOrZero kernelR [out_ch.val, in_ch.val, di.val, dj.val]
      mulEps (β := β) (fexp := fexp) (rnd := rnd) grad_val kernel_val epsδ epsK
    else
      0
  let sumR : R := idxs.foldl (fun acc t => acc + termR t) 0
  let sumS : ℝ := idxs.foldl (fun acc t => acc + termS t) 0
  let sumEps : ℝ := (foldAddState (β := β) (fexp := fexp) (rnd := rnd) idxs termR epsTerm).2

  have hGradVal :
      ∀ (out_ch : Fin outC) (out_i : Fin out_h) (out_j : Fin out_w),
        abs
            (toSpec (β := β) (fexp := fexp) (rnd := rnd) (getAtOrZero δR [out_ch.val, out_i.val,
              out_j.val]) -
              getAtOrZero δS [out_ch.val, out_i.val, out_j.val]) ≤ epsδ := by
    intro out_ch out_i out_j
    simpa using
      (approx_get_at_or_zero (β := β) (fexp := fexp) (rnd := rnd)
        (s := .dim outC (.dim out_h (.dim out_w .scalar)))
        (xS := δS) (xR := δR) (eps := epsδ) hδ [out_ch.val, out_i.val, out_j.val])

  have hKernelVal :
      ∀ (out_ch : Fin outC) (di : Fin kH) (dj : Fin kW),
        abs
            (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                (getAtOrZero kernelR [out_ch.val, in_ch.val, di.val, dj.val]) -
              getAtOrZero kernelS [out_ch.val, in_ch.val, di.val, dj.val]) ≤ epsK := by
    intro out_ch di dj
    simpa using
      (approx_get_at_or_zero (β := β) (fexp := fexp) (rnd := rnd)
        (s := .dim outC (.dim inC (.dim kH (.dim kW .scalar))))
        (xS := kernelS) (xR := kernelR) (eps := epsK) hK [out_ch.val, in_ch.val, di.val, dj.val])

  have hTermIdx :
      ∀ t ∈ idxs, abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (termR t) - termS t) ≤ epsTerm t
        := by
    intro t _ht
    rcases t with ⟨out_ch, out_i, out_j, di, dj⟩
    by_cases h :
        (out_i.val * stride + di.val = i.val + padding) ∧
        (out_j.val * stride + dj.val = j.val + padding)
    · have hx := hGradVal out_ch out_i out_j
      have hy := hKernelVal out_ch di dj
      have hmul :
          abs
              (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                  (getAtOrZero δR [out_ch.val, out_i.val, out_j.val] *
                    getAtOrZero kernelR [out_ch.val, in_ch.val, di.val, dj.val]) -
                getAtOrZero δS [out_ch.val, out_i.val, out_j.val] *
                  getAtOrZero kernelS [out_ch.val, in_ch.val, di.val, dj.val]) ≤
            mulEps (β := β) (fexp := fexp) (rnd := rnd)
              (getAtOrZero δR [out_ch.val, out_i.val, out_j.val])
              (getAtOrZero kernelR [out_ch.val, in_ch.val, di.val, dj.val]) epsδ epsK :=
        approx_mul_nf (β := β) (fexp := fexp) (rnd := rnd)
          (x := getAtOrZero δS [out_ch.val, out_i.val, out_j.val])
          (y := getAtOrZero kernelS [out_ch.val, in_ch.val, di.val, dj.val])
          (xR := getAtOrZero δR [out_ch.val, out_i.val, out_j.val])
          (yR := getAtOrZero kernelR [out_ch.val, in_ch.val, di.val, dj.val])
          (epsx := epsδ) (epsy := epsK) hx hy
      simpa [termR, termS, epsTerm, h] using hmul
    · simp [termR, termS, epsTerm, h]

  have hSum :
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) sumR - sumS) ≤ sumEps := by
    simpa [sumR, sumS, sumEps] using
      (approx_fold_add (β := β) (fexp := fexp) (rnd := rnd)
        (l := idxs) (termS := termS) (termR := termR) (epsTerm := epsTerm) hTermIdx)

  -- Rewrite the `Spec.conv2dInputDerivSpec` nested fold into the flattened `idxs` fold (`sumR/sumS`).
  have hsumR_nested :
      (List.finRange outC).foldl (fun acc out_ch =>
          (List.finRange out_h).foldl (fun acc out_i =>
              (List.finRange out_w).foldl (fun acc out_j =>
                  (List.finRange kH).foldl (fun acc di =>
                      (List.finRange kW).foldl (fun acc dj => acc + termR (out_ch, out_i, out_j, di,
                        dj)) acc) acc) acc) acc) 0 =
        sumR := by
    let fR : R → (Fin outC × Fin out_h × Fin out_w × Fin kH × Fin kW) → R := fun acc t => acc +
      termR t
    have hC0 :
        idxs.foldl fR (0 : R) =
          (List.finRange outC).foldl (fun acc out_ch =>
            List.foldl fR acc
              ((List.finRange out_h).flatMap (fun out_i =>
                (List.finRange out_w).flatMap (fun out_j =>
                  (List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj))))))) (0 : R)
                      := by
      simpa [idxs] using
        (foldl_flatMap (l := List.finRange outC)
          (g := fun out_ch =>
            (List.finRange out_h).flatMap (fun out_i =>
              (List.finRange out_w).flatMap (fun out_j =>
                (List.finRange kH).flatMap (fun di =>
                  (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj))))))
          (f := fR) (init := (0 : R)))
    have hOut :
        ∀ (acc : R) (out_ch : Fin outC),
          List.foldl fR acc
              ((List.finRange out_h).flatMap (fun out_i =>
                (List.finRange out_w).flatMap (fun out_j =>
                  (List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj))))))
            =
          (List.finRange out_h).foldl (fun acc out_i =>
              (List.finRange out_w).foldl (fun acc out_j =>
                  (List.finRange kH).foldl (fun acc di =>
                      (List.finRange kW).foldl (fun acc dj => acc + termR (out_ch, out_i, out_j, di,
                        dj)) acc) acc) acc) acc := by
      intro acc out_ch
      -- fold over `out_h`
      have hH' :
          List.foldl fR acc
              ((List.finRange out_h).flatMap (fun out_i =>
                (List.finRange out_w).flatMap (fun out_j =>
                  (List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj))))))
            =
          (List.finRange out_h).foldl (fun acc out_i =>
              List.foldl fR acc
                ((List.finRange out_w).flatMap (fun out_j =>
                  (List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj)))))) acc := by
        simpa using
          (foldl_flatMap (l := List.finRange out_h)
            (g := fun out_i =>
              (List.finRange out_w).flatMap (fun out_j =>
                (List.finRange kH).flatMap (fun di =>
                  (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj)))))
            (f := fR) (init := acc))
      -- fold over `out_w`/`kH`/`kW`
      have hH :
          ∀ (acc : R) (out_i : Fin out_h),
            List.foldl fR acc
                ((List.finRange out_w).flatMap (fun out_j =>
                  (List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj)))))
              =
            (List.finRange out_w).foldl (fun acc out_j =>
                (List.finRange kH).foldl (fun acc di =>
                    (List.finRange kW).foldl (fun acc dj => acc + termR (out_ch, out_i, out_j, di,
                      dj)) acc) acc) acc := by
        intro acc out_i
        have hW' :
            List.foldl fR acc
                ((List.finRange out_w).flatMap (fun out_j =>
                  (List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj)))))
              =
            (List.finRange out_w).foldl (fun acc out_j =>
                List.foldl fR acc
                  ((List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj))))) acc := by
          simpa using
            (foldl_flatMap (l := List.finRange out_w)
              (g := fun out_j =>
                (List.finRange kH).flatMap (fun di =>
                  (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj))))
              (f := fR) (init := acc))
        have hW :
            ∀ (acc : R) (out_j : Fin out_w),
              List.foldl fR acc
                  ((List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj))))
                =
              (List.finRange kH).foldl (fun acc di =>
                (List.finRange kW).foldl (fun acc dj => acc + termR (out_ch, out_i, out_j, di, dj))
                  acc) acc :=
        by
          intro acc out_j
          have hK' :
              List.foldl fR acc
                  ((List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj))))
                =
              (List.finRange kH).foldl (fun acc di =>
                  List.foldl fR acc ((List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di,
                    dj)))) acc :=
          by
            simpa using
              (foldl_flatMap (l := List.finRange kH)
                (g := fun di => (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj)))
                (f := fR) (init := acc))
          have hWk :
              ∀ (acc : R) (di : Fin kH),
                List.foldl fR acc ((List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di,
                  dj))) =
                  (List.finRange kW).foldl (fun acc dj => acc + termR (out_ch, out_i, out_j, di,
                    dj)) acc :=
          by
            intro acc di
            exact
              (List.foldl_map (f := fun dj => (out_ch, out_i, out_j, di, dj)) (g := fR) (l :=
                List.finRange kW) (init := acc))
          simpa [hK'] using
            (foldl_congr (l := List.finRange kH)
              (f := fun acc di => List.foldl fR acc ((List.finRange kW).map (fun dj => (out_ch,
                out_i, out_j, di, dj))))
              (g := fun acc di => (List.finRange kW).foldl (fun acc dj => acc + termR (out_ch,
                out_i, out_j, di, dj)) acc)
                (init := acc) (by intro acc di; simpa using (hWk acc di)))
        simpa [hW'] using
          (foldl_congr (l := List.finRange out_w)
            (f := fun acc out_j =>
              List.foldl fR acc
                ((List.finRange kH).flatMap (fun di => (List.finRange kW).map (fun dj => (out_ch,
                  out_i, out_j, di, dj)))))
            (g := fun acc out_j =>
              (List.finRange kH).foldl (fun acc di =>
                  (List.finRange kW).foldl (fun acc dj => acc + termR (out_ch, out_i, out_j, di,
                    dj)) acc) acc)
            (init := acc) (by intro acc out_j; simpa using (hW acc out_j)))
      simpa [hH'] using
        (foldl_congr (l := List.finRange out_h)
          (f := fun acc out_i =>
            List.foldl fR acc
              ((List.finRange out_w).flatMap (fun out_j =>
                (List.finRange kH).flatMap (fun di => (List.finRange kW).map (fun dj => (out_ch,
                  out_i, out_j, di, dj))))))
          (g := fun acc out_i =>
            (List.finRange out_w).foldl (fun acc out_j =>
                (List.finRange kH).foldl (fun acc di =>
                    (List.finRange kW).foldl (fun acc dj => acc + termR (out_ch, out_i, out_j, di,
                      dj)) acc) acc) acc)
          (init := acc) (by intro acc out_i; simpa using (hH acc out_i)))
    have hC0' :
        idxs.foldl fR 0 =
          (List.finRange outC).foldl (fun acc out_ch =>
            (List.finRange out_h).foldl (fun acc out_i =>
                (List.finRange out_w).foldl (fun acc out_j =>
                    (List.finRange kH).foldl (fun acc di =>
                        (List.finRange kW).foldl (fun acc dj => acc + termR (out_ch, out_i, out_j,
                          di, dj)) acc) acc) acc) acc) 0 := by
      have :=
        foldl_congr (l := List.finRange outC)
          (f := fun acc out_ch =>
            List.foldl fR acc
              ((List.finRange out_h).flatMap (fun out_i =>
                (List.finRange out_w).flatMap (fun out_j =>
                  (List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj)))))))
          (g := fun acc out_ch =>
            (List.finRange out_h).foldl (fun acc out_i =>
                (List.finRange out_w).foldl (fun acc out_j =>
                    (List.finRange kH).foldl (fun acc di =>
                        (List.finRange kW).foldl (fun acc dj => acc + termR (out_ch, out_i, out_j,
                          di, dj)) acc) acc) acc) acc)
          (init := (0 : R)) (by intro acc out_ch; simpa using (hOut acc out_ch))
      simpa [hC0] using this
    simpa [sumR, fR] using hC0'.symm

  have hsumS_nested :
      (List.finRange outC).foldl (fun acc out_ch =>
          (List.finRange out_h).foldl (fun acc out_i =>
              (List.finRange out_w).foldl (fun acc out_j =>
                  (List.finRange kH).foldl (fun acc di =>
                      (List.finRange kW).foldl (fun acc dj => acc + termS (out_ch, out_i, out_j, di,
                        dj)) acc) acc) acc) acc) 0 =
        sumS := by
    -- same proof as `hsumR_nested`, over `ℝ`
    let fS : ℝ → (Fin outC × Fin out_h × Fin out_w × Fin kH × Fin kW) → ℝ := fun acc t => acc +
      termS t
    have hC0 :
        idxs.foldl fS (0 : ℝ) =
          (List.finRange outC).foldl (fun acc out_ch =>
            List.foldl fS acc
              ((List.finRange out_h).flatMap (fun out_i =>
                (List.finRange out_w).flatMap (fun out_j =>
                  (List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj))))))) (0 : ℝ)
                      := by
      simpa [idxs] using
        (foldl_flatMap (l := List.finRange outC)
          (g := fun out_ch =>
            (List.finRange out_h).flatMap (fun out_i =>
              (List.finRange out_w).flatMap (fun out_j =>
                (List.finRange kH).flatMap (fun di =>
                  (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj))))))
          (f := fS) (init := (0 : ℝ)))
    have hOut :
        ∀ (acc : ℝ) (out_ch : Fin outC),
          List.foldl fS acc
              ((List.finRange out_h).flatMap (fun out_i =>
                (List.finRange out_w).flatMap (fun out_j =>
                  (List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj))))))
            =
          (List.finRange out_h).foldl (fun acc out_i =>
              (List.finRange out_w).foldl (fun acc out_j =>
                  (List.finRange kH).foldl (fun acc di =>
                      (List.finRange kW).foldl (fun acc dj => acc + termS (out_ch, out_i, out_j, di,
                        dj)) acc) acc) acc) acc := by
      intro acc out_ch
      have hH' :
          List.foldl fS acc
              ((List.finRange out_h).flatMap (fun out_i =>
                (List.finRange out_w).flatMap (fun out_j =>
                  (List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj))))))
            =
          (List.finRange out_h).foldl (fun acc out_i =>
              List.foldl fS acc
                ((List.finRange out_w).flatMap (fun out_j =>
                  (List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj)))))) acc := by
        simpa using
          (foldl_flatMap (l := List.finRange out_h)
            (g := fun out_i =>
              (List.finRange out_w).flatMap (fun out_j =>
                (List.finRange kH).flatMap (fun di =>
                  (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj)))))
            (f := fS) (init := acc))
      have hH :
          ∀ (acc : ℝ) (out_i : Fin out_h),
            List.foldl fS acc
                ((List.finRange out_w).flatMap (fun out_j =>
                  (List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj)))))
              =
            (List.finRange out_w).foldl (fun acc out_j =>
                (List.finRange kH).foldl (fun acc di =>
                    (List.finRange kW).foldl (fun acc dj => acc + termS (out_ch, out_i, out_j, di,
                      dj)) acc) acc) acc := by
        intro acc out_i
        have hW' :
            List.foldl fS acc
                ((List.finRange out_w).flatMap (fun out_j =>
                  (List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj)))))
              =
            (List.finRange out_w).foldl (fun acc out_j =>
                List.foldl fS acc
                  ((List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj))))) acc := by
          simpa using
            (foldl_flatMap (l := List.finRange out_w)
              (g := fun out_j =>
                (List.finRange kH).flatMap (fun di =>
                  (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj))))
              (f := fS) (init := acc))
        have hW :
            ∀ (acc : ℝ) (out_j : Fin out_w),
              List.foldl fS acc
                  ((List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj))))
                =
              (List.finRange kH).foldl (fun acc di =>
                  (List.finRange kW).foldl (fun acc dj => acc + termS (out_ch, out_i, out_j, di,
                    dj)) acc) acc := by
          intro acc out_j
          have hK' :
              List.foldl fS acc
                  ((List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj))))
                =
              (List.finRange kH).foldl (fun acc di =>
                  List.foldl fS acc ((List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di,
                    dj)))) acc := by
            simpa using
              (foldl_flatMap (l := List.finRange kH)
                (g := fun di => (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj)))
                (f := fS) (init := acc))
          have hWk :
              ∀ (acc : ℝ) (di : Fin kH),
                List.foldl fS acc ((List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di,
                  dj))) =
                  (List.finRange kW).foldl (fun acc dj => acc + termS (out_ch, out_i, out_j, di,
                    dj)) acc := by
            intro acc di
            exact
              (List.foldl_map (f := fun dj => (out_ch, out_i, out_j, di, dj)) (g := fS) (l :=
                List.finRange kW) (init := acc))
          simpa [hK'] using
            (foldl_congr (l := List.finRange kH)
              (f := fun acc di => List.foldl fS acc ((List.finRange kW).map (fun dj => (out_ch,
                out_i, out_j, di, dj))))
              (g := fun acc di => (List.finRange kW).foldl (fun acc dj => acc + termS (out_ch,
                out_i, out_j, di, dj)) acc)
              (init := acc) (by intro acc di; simpa using (hWk acc di)))
        simpa [hW'] using
          (foldl_congr (l := List.finRange out_w)
            (f := fun acc out_j =>
              List.foldl fS acc
                ((List.finRange kH).flatMap (fun di => (List.finRange kW).map (fun dj => (out_ch,
                  out_i, out_j, di, dj)))))
            (g := fun acc out_j =>
              (List.finRange kH).foldl (fun acc di =>
                  (List.finRange kW).foldl (fun acc dj => acc + termS (out_ch, out_i, out_j, di,
                    dj)) acc) acc)
            (init := acc) (by intro acc out_j; simpa using (hW acc out_j)))
      simpa [hH'] using
        (foldl_congr (l := List.finRange out_h)
          (f := fun acc out_i =>
            List.foldl fS acc
              ((List.finRange out_w).flatMap (fun out_j =>
                (List.finRange kH).flatMap (fun di => (List.finRange kW).map (fun dj => (out_ch,
                  out_i, out_j, di, dj))))))
          (g := fun acc out_i =>
            (List.finRange out_w).foldl (fun acc out_j =>
                (List.finRange kH).foldl (fun acc di =>
                    (List.finRange kW).foldl (fun acc dj => acc + termS (out_ch, out_i, out_j, di,
                      dj)) acc) acc) acc)
          (init := acc) (by intro acc out_i; simpa using (hH acc out_i)))
    have hC0' :
        idxs.foldl fS 0 =
          (List.finRange outC).foldl (fun acc out_ch =>
            (List.finRange out_h).foldl (fun acc out_i =>
                (List.finRange out_w).foldl (fun acc out_j =>
                    (List.finRange kH).foldl (fun acc di =>
                        (List.finRange kW).foldl (fun acc dj => acc + termS (out_ch, out_i, out_j,
                          di, dj)) acc) acc) acc) acc) 0 := by
      have :=
        foldl_congr (l := List.finRange outC)
          (f := fun acc out_ch =>
            List.foldl fS acc
              ((List.finRange out_h).flatMap (fun out_i =>
                (List.finRange out_w).flatMap (fun out_j =>
                  (List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj)))))))
          (g := fun acc out_ch =>
            (List.finRange out_h).foldl (fun acc out_i =>
                (List.finRange out_w).foldl (fun acc out_j =>
                    (List.finRange kH).foldl (fun acc di =>
                        (List.finRange kW).foldl (fun acc dj => acc + termS (out_ch, out_i, out_j,
                          di, dj)) acc) acc) acc) acc)
          (init := (0 : ℝ)) (by intro acc out_ch; simpa using (hOut acc out_ch))
      simpa [hC0] using this
    simpa [sumS, fS] using hC0'.symm

  have houtR :
      getAtOrZero (Spec.conv2dInputDerivSpec (α := R) (layer := layerR) (input := inputR)
        (grad_output := δR))
          [in_ch.val, i.val, j.val] = sumR := by
    let foldIfR : R :=
      (List.finRange outC).foldl (fun accC out_ch =>
          (List.finRange out_h).foldl (fun accH out_i =>
              (List.finRange out_w).foldl (fun accW out_j =>
                  (List.finRange kH).foldl (fun accKH di =>
                      (List.finRange kW).foldl (fun accKW dj =>
                          accKW +
                            if (out_i.val * stride + di.val = i.val + padding) ∧
                                (out_j.val * stride + dj.val = j.val + padding) then
                              getAtOrZero δR [out_ch.val, out_i.val, out_j.val] *
                                getAtOrZero kernelR [out_ch.val, in_ch.val, di.val, dj.val]
                            else 0)
                        accKH)
                    accW)
                accH)
            accC)
        (0 : R)
    have hGet :
        getAtOrZero
            (Spec.conv2dInputDerivSpec (α := R) (layer := layerR) (input := inputR) (grad_output
              := δR))
            [in_ch.val, i.val, j.val] = foldIfR := by
      have hEntry :=
        entry_eq_scalar_get_at_or_zero3
          (t := Spec.conv2dInputDerivSpec (α := R) (layer := layerR) (input := inputR)
            (grad_output := δR))
          in_ch i j
      have hSpecEntry :
          (match
              match
                match
                  Spec.conv2dInputDerivSpec (α := R) (layer := layerR) (input := inputR)
                    (grad_output := δR) with
                | Tensor.dim f => f in_ch with
              | Tensor.dim f => f i with
            | Tensor.dim f => f j) =
            Tensor.scalar foldIfR := by
        dsimp [Spec.conv2dInputDerivSpec, foldIfR, layerR, out_h, out_w, conv2dOutH,
          conv2dOutW]
        rfl
      have hTensor :
          Tensor.scalar foldIfR =
            Tensor.scalar
              (getAtOrZero
                (Spec.conv2dInputDerivSpec (α := R) (layer := layerR) (input := inputR)
                  (grad_output := δR))
                [in_ch.val, i.val, j.val]) := by
        exact hSpecEntry.symm.trans hEntry
      have :
          foldIfR =
            getAtOrZero
              (Spec.conv2dInputDerivSpec (α := R) (layer := layerR) (input := inputR)
                (grad_output := δR))
              [in_ch.val, i.val, j.val] := by
        simpa [Tensor.scalar.injEq] using hTensor
      simpa using this.symm
    have hFold : foldIfR = sumR := by
      simpa [foldIfR, termR] using hsumR_nested
    exact hGet.trans hFold

  have houtS :
      getAtOrZero (Spec.conv2dInputDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
        (grad_output := δS))
          [in_ch.val, i.val, j.val] = sumS := by
    let foldIfS : ℝ :=
      (List.finRange outC).foldl (fun accC out_ch =>
          (List.finRange out_h).foldl (fun accH out_i =>
              (List.finRange out_w).foldl (fun accW out_j =>
                  (List.finRange kH).foldl (fun accKH di =>
                      (List.finRange kW).foldl (fun accKW dj =>
                          accKW +
                            if (out_i.val * stride + di.val = i.val + padding) ∧
                                (out_j.val * stride + dj.val = j.val + padding) then
                              getAtOrZero δS [out_ch.val, out_i.val, out_j.val] *
                                getAtOrZero kernelS [out_ch.val, in_ch.val, di.val, dj.val]
                            else 0)
                        accKH)
                    accW)
                accH)
            accC)
        0
    have hGet :
        getAtOrZero
            (Spec.conv2dInputDerivSpec (α := ℝ) (layer := layerS) (input := inputS) (grad_output
              := δS))
            [in_ch.val, i.val, j.val] = foldIfS := by
      have hEntry :=
        entry_eq_scalar_get_at_or_zero3
          (t := Spec.conv2dInputDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
            (grad_output := δS))
          in_ch i j
      have hSpecEntry :
          (match
              match
                match
                  Spec.conv2dInputDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
                    (grad_output := δS) with
                | Tensor.dim f => f in_ch with
              | Tensor.dim f => f i with
            | Tensor.dim f => f j) =
            Tensor.scalar foldIfS := by
        dsimp [Spec.conv2dInputDerivSpec, foldIfS, layerS, out_h, out_w, conv2dOutH,
          conv2dOutW]
        rfl
      have hTensor :
          Tensor.scalar foldIfS =
            Tensor.scalar
              (getAtOrZero
                (Spec.conv2dInputDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
                  (grad_output := δS))
                [in_ch.val, i.val, j.val]) := by
        exact hSpecEntry.symm.trans hEntry
      have :
          foldIfS =
            getAtOrZero
              (Spec.conv2dInputDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
                (grad_output := δS))
              [in_ch.val, i.val, j.val] := by
        simpa [Tensor.scalar.injEq] using hTensor
      simpa using this.symm
    have hFold : foldIfS = sumS := by
      simpa [foldIfS, termS] using hsumS_nested
    exact hGet.trans hFold

  have hFinal :
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd)
              (getAtOrZero
                (Spec.conv2dInputDerivSpec (α := R) (layer := layerR) (input := inputR)
                  (grad_output := δR))
                [in_ch.val, i.val, j.val]) -
            getAtOrZero
              (Spec.conv2dInputDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
                (grad_output := δS))
              [in_ch.val, i.val, j.val]) ≤ sumEps := by
    simpa [houtR, houtS] using hSum

  have hSumEps :
      sumEps =
        conv2dInputPointBound (β := β) (fexp := fexp) (rnd := rnd)
          (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
            (inH := inH)
          (inW := inW) kernelR δR epsK epsδ in_ch i j := by
    simp [sumEps, conv2dInputPointBound, out_h, out_w, idxs, termR, epsTerm]
  simpa [hSumEps] using hFinal

-- ---------------------------------------------------------------------------
-- Tensor-level backward bound (input gradient)
-- ---------------------------------------------------------------------------

/--
Tensor-level `approxT` bound for the Conv2D **input** gradient.

This lifts `approx_conv2d_input_point` entrywise and packages the error into
`linfNorm (conv2dInputBoundTensor ...)`.
-/
theorem approxT_conv2d_input_deriv_spec
    {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    {kernelS : Tensor ℝ (.dim outC (.dim inC (.dim kH (.dim kW .scalar))))}
    {kernelR : Tensor R (.dim outC (.dim inC (.dim kH (.dim kW .scalar))))}
    {biasS : Tensor ℝ (.dim outC .scalar)} {biasR : Tensor R (.dim outC .scalar)}
    {inputS : Spec.MultiChannelImage inC inH inW ℝ}
    {inputR : Spec.MultiChannelImage inC inH inW R}
    {δS : Spec.MultiChannelImage outC (conv2dOutH inH kH stride padding) (conv2dOutW inW kW stride
      padding) ℝ}
    {δR : Spec.MultiChannelImage outC (conv2dOutH inH kH stride padding) (conv2dOutW inW kW stride
      padding) R}
    {epsK epsδ : ℝ}
    (hK : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) kernelS kernelR
      epsK)
    (hδ : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) δS δR epsδ) :
    let layerS : Spec.Conv2DSpec inC outC kH kW stride padding ℝ h1 h2 h3 :=
      { kernel := kernelS, bias := biasS }
    let layerR : Spec.Conv2DSpec inC outC kH kW stride padding R h1 h2 h3 :=
      { kernel := kernelR, bias := biasR }
    let outS := Spec.conv2dInputDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
      (grad_output := δS)
    let outR := Spec.conv2dInputDerivSpec (α := R) (layer := layerR) (input := inputR)
      (grad_output := δR)
    let bT :=
        conv2dInputBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
            (inH := inH) (inW := inW)
          kernelR δR epsK epsδ
    approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) outS outR (linfNorm
      bT) := by
  intro layerS layerR outS outR bT
  classical
  have hε : 0 ≤ linfNorm bT := linf_norm_nonneg (t := bT)
  refine approxT_dim_of_forall (β := β) (fexp := fexp) (rnd := rnd) (n := inC)
    (s := .dim inH (.dim inW .scalar))
    (xS := outS) (xR := outR) (eps := linfNorm bT) hε ?_
  intro ic
  refine approxT_dim_of_forall (β := β) (fexp := fexp) (rnd := rnd) (n := inH)
    (s := .dim inW .scalar)
    (xS := (match outS with | .dim f => f ic))
    (xR := (match outR with | .dim f => f ic))
    (eps := linfNorm bT) hε ?_
  intro ii
  refine approxT_dim_of_forall (β := β) (fexp := fexp) (rnd := rnd) (n := inW)
    (s := .scalar)
    (xS := (match (match outS with | .dim f => f ic) with | .dim g => g ii))
    (xR := (match (match outR with | .dim f => f ic) with | .dim g => g ii))
    (eps := linfNorm bT) hε ?_
  intro jj

  have hpt :=
    approx_conv2d_input_point (β := β) (fexp := fexp) (rnd := rnd)
      (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding) (inH
        := inH) (inW := inW)
      (h1 := h1) (h2 := h2) (h3 := h3)
      (kernelS := kernelS) (kernelR := kernelR) (biasS := biasS) (biasR := biasR)
      (inputS := inputS) (inputR := inputR) (δS := δS) (δR := δR)
      (epsK := epsK) (epsδ := epsδ) hK hδ ic ii jj

  have hEntry :
      abs (conv2dInputPointBound (β := β) (fexp := fexp) (rnd := rnd)
          (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
            (inH := inH) (inW := inW)
          kernelR δR epsK epsδ ic ii jj) ≤ linfNorm bT := by
    let bTic := match bT with | .dim f => f ic
    let bTicii := match bTic with | .dim f => f ii
    let bTiciijj := match bTicii with | .dim f => f jj
    have h0 : linfNorm bTic ≤ linfNorm bT := by
      simpa [bTic] using (linf_norm_le_get_dim (t := bT) ic)
    have h1' : linfNorm bTicii ≤ linfNorm bTic := by
      simpa [bTicii] using (linf_norm_le_get_dim (t := bTic) ii)
    have h2' : linfNorm bTiciijj ≤ linfNorm bTicii := by
      simpa [bTiciijj] using (linf_norm_le_get_dim (t := bTicii) jj)
    have hchain : linfNorm bTiciijj ≤ linfNorm bT :=
      le_trans (le_trans h2' h1') h0
    let bound :=
      conv2dInputPointBound (β := β) (fexp := fexp) (rnd := rnd)
        (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
          (inH := inH) (inW := inW)
        kernelR δR epsK epsδ ic ii jj
    have hdouble' : (MathFunctions.abs (abs bound) : ℝ) ≤ linfNorm bT := by
      simpa [bound, bT, bTic, bTicii, bTiciijj, conv2dInputBoundTensor, linfNorm,
        RuntimeApprox.linfNorm,
        tensorLinfNorm] using hchain
    have habs : (MathFunctions.abs (abs bound) : ℝ) = abs (abs bound) := by
      rfl
    have hdouble : abs (abs bound) ≤ linfNorm bT := by
      simpa [habs] using hdouble'
    simpa [bound, abs_abs] using hdouble

  have hscalar :
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd)
              (getAtOrZero outR [ic.val, ii.val, jj.val]) -
            getAtOrZero outS [ic.val, ii.val, jj.val]) ≤ linfNorm bT :=
    le_trans (le_trans hpt (le_abs_self _)) hEntry

  let entryS : Tensor ℝ .scalar :=
    (match match match outS with
      | .dim f => f ic with
    | .dim g => g ii with
    | .dim h => h jj)
  let entryR : Tensor R .scalar :=
    (match match match outR with
      | .dim f => f ic with
    | .dim g => g ii with
    | .dim h => h jj)
  change
    approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) entryS entryR
      (linfNorm bT)
  have hEntryS :
      entryS = Tensor.scalar (getAtOrZero outS [ic.val, ii.val, jj.val]) := by
    simpa [entryS] using (entry_eq_scalar_get_at_or_zero3 (t := outS) ic ii jj)
  have hEntryR :
      entryR = Tensor.scalar (getAtOrZero outR [ic.val, ii.val, jj.val]) := by
    simpa [entryR] using (entry_eq_scalar_get_at_or_zero3 (t := outR) ic ii jj)
  have happ :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Tensor.scalar (getAtOrZero outS [ic.val, ii.val, jj.val]))
        (Tensor.scalar (getAtOrZero outR [ic.val, ii.val, jj.val]))
        (linfNorm bT) :=
    (approxT_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (x := getAtOrZero outS [ic.val, ii.val, jj.val]) (xR := getAtOrZero outR [ic.val,
          ii.val, jj.val])
        (eps := linfNorm bT)).2 (by
          simpa using hscalar)
  simpa [hEntryS, hEntryR] using happ

-- ---------------------------------------------------------------------------
-- `RevNode` packaging for Conv2D.
-- ---------------------------------------------------------------------------

lemma idx_i_ne_of_shape_ne {Γ : List Shape} {s₁ s₂ : Shape} (a : Idx Γ s₁) (b : Idx Γ s₂) (hs : s₁ ≠
  s₂) :
    a.i ≠ b.i := by
  intro hEq
  apply hs
  exact idx_shape_eq_of_i_eq (a := a) (b := b) hEq

/--
Package Conv2D forward + backward (VJP) as a `RevNode` for `RevGraph.backprop_approx`.

The node uses the forward bound from `conv2d_forward` and the three gradient bounds proved in this
file (kernel/bias/input) to provide a compositional reverse-mode approximation theorem.
-/
def conv2dRevNode
    {Γ : List Shape}
    {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    (kernelIdx : Idx Γ (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))))
    (biasIdx : Idx Γ (.dim outC .scalar))
    (inputIdx : Idx Γ (.dim inC (.dim inH (.dim inW .scalar)))) :
    RevNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ
      (.dim outC (.dim (conv2dOutH inH kH stride padding) (.dim (conv2dOutW inW kW stride padding)
        .scalar))) :=
by
  classical
  have hShapeKB :
      (Shape.dim outC (Shape.dim inC (Shape.dim kH (Shape.dim kW Shape.scalar)))) ≠
        (Shape.dim outC Shape.scalar) := by
    intro h
    injection h with _ hRest
    cases hRest
  have hShapeKX :
      (Shape.dim outC (Shape.dim inC (Shape.dim kH (Shape.dim kW Shape.scalar)))) ≠
        (Shape.dim inC (Shape.dim inH (Shape.dim inW Shape.scalar))) := by
    intro h
    injection h with _ hRest1
    injection hRest1 with _ hRest2
    injection hRest2 with _ hRest3
    cases hRest3
  have hShapeBX :
      (Shape.dim outC Shape.scalar) ≠ (Shape.dim inC (Shape.dim inH (Shape.dim inW Shape.scalar)))
        := by
    intro h
    injection h with _ hRest
    cases hRest
  have hab : kernelIdx.i ≠ biasIdx.i :=
    idx_i_ne_of_shape_ne (a := kernelIdx) (b := biasIdx) hShapeKB
  have hac : kernelIdx.i ≠ inputIdx.i :=
    idx_i_ne_of_shape_ne (a := kernelIdx) (b := inputIdx) hShapeKX
  have hbc : biasIdx.i ≠ inputIdx.i :=
    idx_i_ne_of_shape_ne (a := biasIdx) (b := inputIdx) hShapeBX
  refine
    { toFwdNode := conv2dNode (β := β) (fexp := fexp) (rnd := rnd)
        (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
          (inH := inH) (inW := inW)
        (h1 := h1) (h2 := h2) (h3 := h3)
        kernelIdx biasIdx inputIdx
      vjpSpec := fun ctx δ =>
        let kernelS := getIdx (α := SpecScalar) ctx kernelIdx
        let biasS := getIdx (α := SpecScalar) ctx biasIdx
        let inputS := getIdx (α := SpecScalar) ctx inputIdx
        let layerS : Spec.Conv2DSpec inC outC kH kW stride padding ℝ h1 h2 h3 :=
          { kernel := kernelS, bias := biasS }
        let dK := Spec.conv2dKernelDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
          (grad_output := δ)
        let dB := Spec.conv2dBiasDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
          (grad_output := δ)
        let dX := Spec.conv2dInputDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
          (grad_output := δ)
        TList.set3IdxNe (α := SpecScalar) (Γ := Γ) (s₁ := (.dim outC (.dim inC (.dim kH (.dim kW
          .scalar)))))
          (s₂ := (.dim outC .scalar)) (s₃ := (.dim inC (.dim inH (.dim inW .scalar))))
          kernelIdx dK biasIdx dB inputIdx dX hab hac hbc
      vjpRuntime := fun ctx δ =>
        let kernelR := getIdx (α := R) ctx kernelIdx
        let biasR := getIdx (α := R) ctx biasIdx
        let inputR := getIdx (α := R) ctx inputIdx
        let layerR : Spec.Conv2DSpec inC outC kH kW stride padding R h1 h2 h3 :=
          { kernel := kernelR, bias := biasR }
        let dK := Spec.conv2dKernelDerivSpec (α := R) (layer := layerR) (input := inputR)
          (grad_output := δ)
        let dB := Spec.conv2dBiasDerivSpec (α := R) (layer := layerR) (input := inputR)
          (grad_output := δ)
        let dX := Spec.conv2dInputDerivSpec (α := R) (layer := layerR) (input := inputR)
          (grad_output := δ)
        TList.set3IdxNe (α := R) (Γ := Γ) (s₁ := (.dim outC (.dim inC (.dim kH (.dim kW
          .scalar)))))
          (s₂ := (.dim outC .scalar)) (s₃ := (.dim inC (.dim inH (.dim inW .scalar))))
          kernelIdx dK biasIdx dB inputIdx dX hab hac hbc
      vjpBound := fun epsCtx ctxR epsδ δR =>
        let kernelR := getIdx (α := R) ctxR kernelIdx
        let inputR := getIdx (α := R) ctxR inputIdx
        let epsK := getIdxEps (Γ := Γ) (s := (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))))
          epsCtx kernelIdx
        let epsX := getIdxEps (Γ := Γ) (s := (.dim inC (.dim inH (.dim inW .scalar)))) epsCtx
          inputIdx
          let epsDK :=
            linfNorm
              (conv2dKernelBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding :=
                  padding) (inH := inH) (inW := inW)
                inputR δR epsX epsδ)
          let epsDB :=
            linfNorm
              (conv2dBiasBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding) (inH :=
                  inH) (inW := inW)
                δR epsδ)
          let epsDX :=
            linfNorm
              (conv2dInputBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding :=
                  padding) (inH := inH) (inW := inW)
                kernelR δR epsK epsδ)
        EList.set3IdxNe (Γ := Γ) (s₁ := (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))))
          (s₂ := (.dim outC .scalar)) (s₃ := (.dim inC (.dim inH (.dim inW .scalar))))
          kernelIdx epsDK biasIdx epsDB inputIdx epsDX hab hac hbc
      vjpSound := ?_ }
  intro ctxS ctxR epsCtx δS δR epsδ hctx hδ
  have hK :=
    approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx kernelIdx
  have hB :=
    approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx biasIdx
  have hX :=
    approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx inputIdx
  -- Unpack the context tensors once, to share with both spec and runtime sides.
  let kernelS := getIdx (α := SpecScalar) ctxS kernelIdx
  let kernelR := getIdx (α := R) ctxR kernelIdx
  let biasS := getIdx (α := SpecScalar) ctxS biasIdx
  let biasR := getIdx (α := R) ctxR biasIdx
  let inputS := getIdx (α := SpecScalar) ctxS inputIdx
  let inputR := getIdx (α := R) ctxR inputIdx
  let layerS : Spec.Conv2DSpec inC outC kH kW stride padding ℝ h1 h2 h3 :=
    { kernel := kernelS, bias := biasS }
  let layerR : Spec.Conv2DSpec inC outC kH kW stride padding R h1 h2 h3 :=
    { kernel := kernelR, bias := biasR }
  let epsK := getIdxEps (Γ := Γ) (s := (.dim outC (.dim inC (.dim kH (.dim kW .scalar))))) epsCtx
    kernelIdx
  let epsX := getIdxEps (Γ := Γ) (s := (.dim inC (.dim inH (.dim inW .scalar)))) epsCtx inputIdx
  have hBias :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Spec.conv2dBiasDerivSpec (α := ℝ) (layer := layerS) (input := inputS) (grad_output :=
          δS))
      (Spec.conv2dBiasDerivSpec (α := R) (layer := layerR) (input := inputR) (grad_output :=
        δR))
      (linfNorm
        (conv2dBiasBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding) (inH :=
                inH) (inW := inW)
              δR epsδ)) := by
    -- `bias_deriv` depends only on `δ`, but we keep the same `layer`/`input` arguments.
    simpa [kernelS, kernelR, biasS, biasR, inputS, inputR, layerS, layerR] using
      (approxT_conv2d_bias_deriv_spec (β := β) (fexp := fexp) (rnd := rnd)
        (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
          (inH := inH) (inW := inW)
        (h1 := h1) (h2 := h2) (h3 := h3)
        (kernelS := kernelS) (kernelR := kernelR) (biasS := biasS) (biasR := biasR)
        (inputS := inputS) (inputR := inputR) (δS := δS) (δR := δR) (epsδ := epsδ) hδ)
  have hKernel :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Spec.conv2dKernelDerivSpec (α := ℝ) (layer := layerS) (input := inputS) (grad_output :=
          δS))
      (Spec.conv2dKernelDerivSpec (α := R) (layer := layerR) (input := inputR) (grad_output :=
        δR))
      (linfNorm
        (conv2dKernelBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
            (inH := inH) (inW := inW)
          inputR δR epsX epsδ)) := by
    have hX' : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) inputS
      inputR epsX := by
      simpa [epsX] using hX
    simpa [kernelS, kernelR, biasS, biasR, inputS, inputR, layerS, layerR] using
      (approxT_conv2d_kernel_deriv_spec (β := β) (fexp := fexp) (rnd := rnd)
        (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
          (inH := inH) (inW := inW)
        (h1 := h1) (h2 := h2) (h3 := h3)
        (kernelS := kernelS) (kernelR := kernelR) (biasS := biasS) (biasR := biasR)
        (inputS := inputS) (inputR := inputR) (δS := δS) (δR := δR) (epsX := epsX) (epsδ := epsδ)
        hX' hδ)
  have hInput :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Spec.conv2dInputDerivSpec (α := ℝ) (layer := layerS) (input := inputS) (grad_output :=
          δS))
      (Spec.conv2dInputDerivSpec (α := R) (layer := layerR) (input := inputR) (grad_output :=
        δR))
      (linfNorm
        (conv2dInputBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
            (inH := inH) (inW := inW)
          kernelR δR epsK epsδ)) := by
    have hK' : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) kernelS
      kernelR epsK := by
      simpa [epsK] using hK
    simpa [kernelS, kernelR, biasS, biasR, inputS, inputR, layerS, layerR] using
      (approxT_conv2d_input_deriv_spec (β := β) (fexp := fexp) (rnd := rnd)
        (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
          (inH := inH) (inW := inW)
        (h1 := h1) (h2 := h2) (h3 := h3)
        (kernelS := kernelS) (kernelR := kernelR) (biasS := biasS) (biasR := biasR)
        (inputS := inputS) (inputR := inputR) (δS := δS) (δR := δR) (epsK := epsK) (epsδ := epsδ)
        hK' hδ)
  have hctx' :=
    approxCtx_set3Idx_ne (β := β) (fexp := fexp) (rnd := rnd)
      (Γ := Γ)
      (s₁ := (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))))
      (s₂ := (.dim outC .scalar))
      (s₃ := (.dim inC (.dim inH (.dim inW .scalar))))
      kernelIdx biasIdx inputIdx
      (t₁S := Spec.conv2dKernelDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
        (grad_output := δS))
      (t₁R := Spec.conv2dKernelDerivSpec (α := R) (layer := layerR) (input := inputR)
        (grad_output := δR))
        (eps₁ :=
          linfNorm
            (conv2dKernelBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding :=
                padding) (inH := inH) (inW := inW)
              inputR δR epsX epsδ))
      (t₂S := Spec.conv2dBiasDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
        (grad_output := δS))
      (t₂R := Spec.conv2dBiasDerivSpec (α := R) (layer := layerR) (input := inputR)
        (grad_output := δR))
        (eps₂ :=
          linfNorm
            (conv2dBiasBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding) (inH :=
                inH) (inW := inW)
              δR epsδ))
      (t₃S := Spec.conv2dInputDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
        (grad_output := δS))
      (t₃R := Spec.conv2dInputDerivSpec (α := R) (layer := layerR) (input := inputR)
        (grad_output := δR))
        (eps₃ :=
          linfNorm
            (conv2dInputBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding :=
                padding) (inH := inH) (inW := inW)
              kernelR δR epsK epsδ))
      hKernel hBias hInput hab hac hbc
  -- Goal matches the `set3Idx_ne` packaging in `vjpSpec/vjpRuntime/vjpBound`.
  simpa [kernelS, kernelR, biasS, biasR, inputS, inputR, layerS, layerR, epsK, epsX] using hctx'

end NFBackend

end
end RuntimeApprox
end Proofs
