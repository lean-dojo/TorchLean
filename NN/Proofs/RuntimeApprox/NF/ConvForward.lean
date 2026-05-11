/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.Graph.ForwardApprox
public import NN.Proofs.RuntimeApprox.NF.Conv
public import NN.Proofs.RuntimeApprox.NF.Utils
public import NN.Spec.Layers.Utils

/-!
# Conv2D Forward Approximation

NF (rounded) backend: Conv2D forward runtime→spec approximation.

This file proves soundness of the `NFBackend.conv2dPointBound`/`conv2dBoundTensor` bounds and
packages Conv2D as a `FwdNode` so it composes via `FwdGraph.eval_approx`.

PyTorch analogue: a forward Conv2D op (typically `torch.nn.functional.conv2d`) plus the standard
“stack over channels/spatial positions” tensor semantics.
https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html

## Map of this file
- Small fold/indexing lemmas (`foldl_flatMap`, `entry_eq_scalar_get_at_or_zero3`) used to align the
  spec definition of convolution with the bound-generating replay in the runtime proof.
- `approx_conv2d_point`: elementwise forward error bound for a single `(out_ch, i, j)` output.
- `approxT_conv2d_spec`: tensor-level `approxT` statement obtained by lifting the pointwise bound.
- `conv2dNode`: packaging as a `FwdNode` so it composes inside larger graphs.

## References
- Dumoulin & Visin, *A guide to convolution arithmetic for deep learning* (indexing/stride/padding
  conventions).
- Goodfellow, Bengio, Courville, *Deep Learning* (MIT Press, 2016), Chapter on convolutional
  networks.
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

set_option maxHeartbeats 8000000

-- ---------------------------------------------------------------------------
-- Small list-fold lemma: `foldl` over `flatMap` corresponds to nested `foldl`.
-- ---------------------------------------------------------------------------

lemma foldl_flatMap {α β γ : Type} (l : List α) (g : α → List β) (f : γ → β → γ) (init : γ) :
    (l.flatMap g).foldl f init = l.foldl (fun acc a => (g a).foldl f acc) init := by
  induction l generalizing init with
  | nil =>
      simp
  | cons a tl ih =>
      simp [List.flatMap_cons, List.foldl_append, ih]

private lemma foldl_finRange3_eq_flat_foldl
    {γ : Type} [Zero γ] [Add γ] {inC kH kW : Nat} (term : Fin inC × Fin kH × Fin kW → γ) :
    (List.finRange inC).foldl (fun acc in_ch =>
          (List.finRange kH).foldl (fun acc di =>
                (List.finRange kW).foldl (fun acc dj => acc + term (in_ch, di, dj)) acc)
              acc)
        0
      =
      ((List.finRange inC).flatMap (fun in_ch =>
            (List.finRange kH).flatMap (fun di =>
              (List.finRange kW).map (fun dj => (in_ch, di, dj))))).foldl
        (fun acc t => acc + term t) 0 := by
  classical
  let idxs : List (Fin inC × Fin kH × Fin kW) :=
    (List.finRange inC).flatMap (fun in_ch =>
      (List.finRange kH).flatMap (fun di =>
        (List.finRange kW).map (fun dj => (in_ch, di, dj))))
  let f : γ → (Fin inC × Fin kH × Fin kW) → γ := fun acc t => acc + term t

  have hC :
      idxs.foldl f (0 : γ) =
        (List.finRange inC).foldl (fun acc in_ch =>
            ((List.finRange kH).flatMap (fun di =>
                  (List.finRange kW).map (fun dj => (in_ch, di, dj)))).foldl f acc)
          0 := by
    simpa [idxs] using
      (foldl_flatMap (l := List.finRange inC)
        (g := fun in_ch =>
          (List.finRange kH).flatMap (fun di =>
            (List.finRange kW).map (fun dj => (in_ch, di, dj))))
        (f := f) (init := (0 : γ)))

  have hH :
      ∀ (acc : γ) (in_ch : Fin inC),
        ((List.finRange kH).flatMap (fun di =>
              (List.finRange kW).map (fun dj => (in_ch, di, dj)))).foldl f acc
          =
          (List.finRange kH).foldl (fun acc di =>
                (List.finRange kW).foldl (fun acc dj => acc + term (in_ch, di, dj)) acc)
              acc := by
    intro acc in_ch
    have hH' :
        ((List.finRange kH).flatMap (fun di =>
              (List.finRange kW).map (fun dj => (in_ch, di, dj)))).foldl f acc
          =
          (List.finRange kH).foldl (fun acc di =>
              ((List.finRange kW).map (fun dj => (in_ch, di, dj))).foldl f acc)
            acc := by
      simpa using
        (foldl_flatMap (l := List.finRange kH)
          (g := fun di => (List.finRange kW).map (fun dj => (in_ch, di, dj)))
          (f := f) (init := acc))
    have hW :
        ∀ (acc : γ) (di : Fin kH),
          ((List.finRange kW).map (fun dj => (in_ch, di, dj))).foldl f acc
            =
            (List.finRange kW).foldl (fun acc dj => acc + term (in_ch, di, dj)) acc := by
      intro acc di
      simp [f, List.foldl_map]
    have :
        (List.finRange kH).foldl (fun acc di =>
            ((List.finRange kW).map (fun dj => (in_ch, di, dj))).foldl f acc)
          acc
          =
          (List.finRange kH).foldl (fun acc di =>
                (List.finRange kW).foldl (fun acc dj => acc + term (in_ch, di, dj)) acc)
              acc := by
      refine
        foldl_congr (l := List.finRange kH)
          (f := fun acc di =>
            ((List.finRange kW).map (fun dj => (in_ch, di, dj))).foldl f acc)
          (g := fun acc di =>
            (List.finRange kW).foldl (fun acc dj => acc + term (in_ch, di, dj)) acc)
          (init := acc) ?_
      intro acc di
      simpa using (hW acc di)
    exact hH'.trans this

  have hC' :
      idxs.foldl f 0 =
        (List.finRange inC).foldl (fun acc in_ch =>
              (List.finRange kH).foldl (fun acc di =>
                    (List.finRange kW).foldl (fun acc dj => acc + term (in_ch, di, dj)) acc)
                  acc)
            0 := by
    have :=
      foldl_congr (l := List.finRange inC)
        (f := fun acc in_ch =>
          ((List.finRange kH).flatMap (fun di =>
              (List.finRange kW).map (fun dj => (in_ch, di, dj)))).foldl f acc)
        (g := fun acc in_ch =>
          (List.finRange kH).foldl (fun acc di =>
                (List.finRange kW).foldl (fun acc dj => acc + term (in_ch, di, dj)) acc)
              acc)
        (init := (0 : γ)) (by
          intro acc in_ch
          simpa using (hH acc in_ch))
    simpa [hC] using this

  simpa [idxs, f] using hC'.symm

-- ---------------------------------------------------------------------------
-- Component selection: relate 3D `Fin` indexing (via `match`) to `get_at_or_zero`.
-- ---------------------------------------------------------------------------

lemma entry_eq_scalar_get_at_or_zero3
    {α : Type} [Zero α] {n1 n2 n3 : Nat}
    (t : Tensor α (.dim n1 (.dim n2 (.dim n3 .scalar))))
    (i1 : Fin n1) (i2 : Fin n2) (i3 : Fin n3) :
    (match
      match
        match t with
        | .dim f => f i1 with
      | .dim f => f i2 with
    | .dim f => f i3) =
      Tensor.scalar (getAtOrZero t [i1.val, i2.val, i3.val]) := by
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
              | scalar v =>
                  simp [get_at_or_zero_dim_cons, get_at_or_zero_scalar_nil, hi1, hi2, hi3, h1, h2,
                    h3]

-- ---------------------------------------------------------------------------
-- Padding reads: relate `pad_multi_channel` branches to the original input approximation.
-- ---------------------------------------------------------------------------

private lemma mkInputIdx?_2d
    (out_i out_j di dj stride padding : Nat) :
    Spec.Private.mkInputIdx? [out_i, out_j] [di, dj] [stride, stride] [padding, padding] =
      if _ : out_i * stride + di < padding ∨ out_j * stride + dj < padding then
        none
      else
        some [out_i * stride + di - padding, out_j * stride + dj - padding] := by
  by_cases hq0 : out_i * stride + di < padding
  · simp [Spec.Private.mkInputIdx?, hq0]
  · by_cases hq1 : out_j * stride + dj < padding
    · simp [Spec.Private.mkInputIdx?, hq0, hq1]
    · have hOr : ¬(out_i * stride + di < padding ∨ out_j * stride + dj < padding) := by
        intro h
        cases h with
        | inl h => exact hq0 h
        | inr h => exact hq1 h
      simp [Spec.Private.mkInputIdx?, hq0, hq1]

private lemma conv_input_val_eq_padded
    {α : Type} [Context α]
    {inC inH inW stride padding : Nat}
    (img : Spec.MultiChannelImage inC inH inW α)
    (c : Fin inC) (out_i out_j di dj : Nat) :
    (match Spec.Private.mkInputIdx? [out_i, out_j] [di, dj] [stride, stride] [padding, padding] with
      | none => (0 : α)
      | some inIdx => getAtOrZero img (c.val :: inIdx))
      =
    getAtOrZero
        (if h4 : padding = 0 then
          tensorCast
            (Shape.dim inC
              (Shape.dim (inH + 2 * padding) (Shape.dim (inW + 2 * padding) Shape.scalar)))
            (by simp; rw [h4]) img
        else
          Spec.padMultiChannel img padding)
        [c.val, out_i * stride + di, out_j * stride + dj] := by
  classical
  by_cases h4 : padding = 0
  · subst h4
    simp [Spec.Private.mkInputIdx?]
  · have hpad :=
      Spec.get_at_or_zero_pad_multi_channel (α := α) (img := img) (c := c)
        (p := out_i * stride + di) (q := out_j * stride + dj) (padding := padding)
    by_cases ht : out_i * stride + di < padding ∨ out_j * stride + dj < padding
    · -- left/top padding: both sides are `0`
      have :
          Spec.Private.mkInputIdx? [out_i, out_j] [di, dj] [stride, stride] [padding, padding] =
            none := by
        simp [mkInputIdx?_2d, ht]
      simp [h4, this, hpad, ht]
    · -- core region: both sides read the original tensor at shifted indices
      have :
          Spec.Private.mkInputIdx? [out_i, out_j] [di, dj] [stride, stride] [padding, padding] =
            some [out_i * stride + di - padding, out_j * stride + dj - padding] := by
        simp [mkInputIdx?_2d, ht]
      simp [h4, this, hpad, ht]

  private lemma conv2dSpec_getAtOrZero_point
      {α : Type} [Context α]
      {inC outC kH kW stride padding inH inW : Nat}
      {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
      (layer : Spec.Conv2DSpec inC outC kH kW stride padding α h1 h2 h3)
      (input : Spec.MultiChannelImage inC inH inW α)
      (out_ch : Fin outC)
      (i : Fin (conv2dOutH inH kH stride padding))
      (j : Fin (conv2dOutW inW kW stride padding)) :
    getAtOrZero (Spec.conv2dSpec (α := α) (layer := layer) (input := input))
        [out_ch.val, i.val, j.val] =
      (List.finRange inC).foldl (fun acc (in_ch : Fin inC) =>
          (List.finRange kH).foldl (fun acc (di : Fin kH) =>
              (List.finRange kW).foldl (fun acc (dj : Fin kW) =>
                  acc +
                    (match Spec.Private.mkInputIdx? [i.val, j.val] [di.val, dj.val] [stride, stride]
                      [padding, padding] with
                    | none => (0 : α)
                    | some inIdx => getAtOrZero input (in_ch.val :: inIdx)) *
                      getAtOrZero layer.kernel [out_ch.val, in_ch.val, di.val, dj.val])
                acc)
            acc)
          0 +
          getAtOrZero layer.bias [out_ch.val] := by
    classical
    unfold Spec.conv2dSpec
    -- `getAtOrZero` reduces via in-bounds checks. With `outH/outW = (...) / stride + 1`, simp will
    -- rewrite `i < outH` into `i ≤ outH - 1`, so we provide those `≤` proofs explicitly.
    have hi : i.val ≤ (inH + 2 * padding - kH) / stride := by
      have hi' : i.val < (inH + 2 * padding - kH) / stride + 1 := by
        simpa [conv2dOutH] using i.isLt
      exact Nat.le_of_lt_succ hi'
    have hj : j.val ≤ (inW + 2 * padding - kW) / stride := by
      have hj' : j.val < (inW + 2 * padding - kW) / stride + 1 := by
        simpa [conv2dOutW] using j.isLt
      exact Nat.le_of_lt_succ hj'
    simp [conv2dOutH, conv2dOutW, Spec.get_at_or_zero_dim_cons, Spec.get_at_or_zero_scalar_nil,
      out_ch.isLt, hi, hj]
    rfl

-- ---------------------------------------------------------------------------
-- Conv2D forward: pointwise soundness for `conv2dPointBound`.
-- ---------------------------------------------------------------------------

/--
Pointwise forward soundness for Conv2D in the rounded `NF` backend.

Given spec/runtime approximations for the kernel, bias, and input, this bounds the absolute error
of a single output entry of `Spec.conv2dSpec` by `conv2dPointBound` (a replay-style bound
constructed from local per-term bounds).
-/
theorem approx_conv2d_point
    {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    {kernelS : Tensor ℝ (.dim outC (.dim inC (.dim kH (.dim kW .scalar))))}
    {kernelR : Tensor R (.dim outC (.dim inC (.dim kH (.dim kW .scalar))))}
    {biasS : Tensor ℝ (.dim outC .scalar)} {biasR : Tensor R (.dim outC .scalar)}
    {inputS : Spec.MultiChannelImage inC inH inW ℝ}
    {inputR : Spec.MultiChannelImage inC inH inW R}
    {epsK epsB epsX : ℝ}
    (hK : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) kernelS kernelR
      epsK)
    (hB : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) biasS biasR epsB)
    (hX : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) inputS inputR
      epsX)
    (out_ch : Fin outC) (i : Fin (conv2dOutH inH kH stride padding)) (j : Fin (conv2dOutW inW kW
      stride padding)) :
    let layerS : Spec.Conv2DSpec inC outC kH kW stride padding ℝ h1 h2 h3 :=
      { kernel := kernelS, bias := biasS }
    let layerR : Spec.Conv2DSpec inC outC kH kW stride padding R h1 h2 h3 :=
      { kernel := kernelR, bias := biasR }
    abs
        (toSpec (β := β) (fexp := fexp) (rnd := rnd)
            (getAtOrZero (Spec.conv2dSpec (α := R) (layer := layerR) (input := inputR))
              [out_ch.val, i.val, j.val]) -
          getAtOrZero (Spec.conv2dSpec (α := ℝ) (layer := layerS) (input := inputS))
              [out_ch.val, i.val, j.val]) ≤
      conv2dPointBound (β := β) (fexp := fexp) (rnd := rnd) (layerR := layerR) (inputR := inputR)
        (epsK := epsK) (epsB := epsB) (epsX := epsX) out_ch i j := by
  intro layerS layerR
  classical

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

  let idxs : List (Fin inC × Fin kH × Fin kW) :=
    (List.finRange inC).flatMap (fun in_ch =>
      (List.finRange kH).flatMap (fun di =>
        (List.finRange kW).map (fun dj => (in_ch, di, dj))))

  let termR : (Fin inC × Fin kH × Fin kW) → R := fun t =>
    let in_ch : Fin inC := t.1
    let di : Fin kH := t.2.1
    let dj : Fin kW := t.2.2
    let input_val := getAtOrZero paddedR [in_ch.val, i.val * stride + di.val, j.val * stride +
      dj.val]
    let kernel_val := getAtOrZero layerR.kernel [out_ch.val, in_ch.val, di.val, dj.val]
    input_val * kernel_val
  let termS : (Fin inC × Fin kH × Fin kW) → ℝ := fun t =>
    let in_ch : Fin inC := t.1
    let di : Fin kH := t.2.1
    let dj : Fin kW := t.2.2
    let input_val := getAtOrZero paddedS [in_ch.val, i.val * stride + di.val, j.val * stride +
      dj.val]
    let kernel_val := getAtOrZero layerS.kernel [out_ch.val, in_ch.val, di.val, dj.val]
    input_val * kernel_val
  let epsTerm : (Fin inC × Fin kH × Fin kW) → ℝ := fun t =>
    let in_ch : Fin inC := t.1
    let di : Fin kH := t.2.1
    let dj : Fin kW := t.2.2
    let input_val := getAtOrZero paddedR [in_ch.val, i.val * stride + di.val, j.val * stride +
      dj.val]
    let kernel_val := getAtOrZero layerR.kernel [out_ch.val, in_ch.val, di.val, dj.val]
    mulEps (β := β) (fexp := fexp) (rnd := rnd) input_val kernel_val epsX epsK

  let sumR : R := idxs.foldl (fun acc t => acc + termR t) 0
  let sumS : ℝ := idxs.foldl (fun acc t => acc + termS t) 0
  let sumEps : ℝ := (foldAddState (β := β) (fexp := fexp) (rnd := rnd) idxs termR epsTerm).2

  have hBiasVal :
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd) (getAtOrZero layerR.bias [out_ch.val]) -
            getAtOrZero layerS.bias [out_ch.val]) ≤ epsB := by
    simpa using
      (approx_get_at_or_zero (β := β) (fexp := fexp) (rnd := rnd)
        (s := .dim outC .scalar) (xS := biasS) (xR := biasR) (eps := epsB) hB [out_ch.val])

  have hKernelVal :
      ∀ (in_ch : Fin inC) (di : Fin kH) (dj : Fin kW),
        abs
            (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                (getAtOrZero layerR.kernel [out_ch.val, in_ch.val, di.val, dj.val]) -
              getAtOrZero layerS.kernel [out_ch.val, in_ch.val, di.val, dj.val]) ≤ epsK := by
    intro in_ch di dj
    simpa using
      (approx_get_at_or_zero (β := β) (fexp := fexp) (rnd := rnd)
        (s := .dim outC (.dim inC (.dim kH (.dim kW .scalar))))
        (xS := kernelS) (xR := kernelR) (eps := epsK) hK
        [out_ch.val, in_ch.val, di.val, dj.val])

  have hInputVal :
      ∀ (in_ch : Fin inC) (p q : Nat),
        abs
            (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                (getAtOrZero paddedR [in_ch.val, p, q]) -
              getAtOrZero paddedS [in_ch.val, p, q]) ≤ epsX := by
    intro in_ch p q
    simpa [paddedR, paddedS] using
      (approx_padded_input_read (β := β) (fexp := fexp) (rnd := rnd)
        (inC := inC) (inH := inH) (inW := inW) (padding := padding)
        (xS := inputS) (xR := inputR) (epsX := epsX) hX in_ch p q)

  have hTermIdx :
      ∀ t ∈ idxs, abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (termR t) - termS t) ≤ epsTerm t
        := by
    intro t _ht
    rcases t with ⟨in_ch, di, dj⟩
    have hx := hInputVal in_ch (i.val * stride + di.val) (j.val * stride + dj.val)
    have hk := hKernelVal in_ch di dj
    have := approx_mul_nf (β := β) (fexp := fexp) (rnd := rnd)
      (x := getAtOrZero paddedS [in_ch.val, i.val * stride + di.val, j.val * stride + dj.val])
      (y := getAtOrZero layerS.kernel [out_ch.val, in_ch.val, di.val, dj.val])
      (xR := getAtOrZero paddedR [in_ch.val, i.val * stride + di.val, j.val * stride + dj.val])
      (yR := getAtOrZero layerR.kernel [out_ch.val, in_ch.val, di.val, dj.val])
      (epsx := epsX) (epsy := epsK) hx hk
    simpa [termR, termS, epsTerm, mulEps] using this

  have hSum :
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) sumR - sumS) ≤ sumEps := by
    simpa [sumR, sumS, sumEps] using
      (approx_fold_add (β := β) (fexp := fexp) (rnd := rnd) (l := idxs)
        (termS := termS) (termR := termR) (epsTerm := epsTerm) hTermIdx)

  have hFinal :
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd) (sumR + getAtOrZero layerR.bias
            [out_ch.val]) -
            (sumS + getAtOrZero layerS.bias [out_ch.val])) ≤
        addEps (β := β) (fexp := fexp) (rnd := rnd) sumR (getAtOrZero layerR.bias [out_ch.val])
          sumEps epsB := by
    have := approx_add_nf (β := β) (fexp := fexp) (rnd := rnd) (x := sumS) (y := getAtOrZero
      layerS.bias [out_ch.val])
      (xR := sumR) (yR := getAtOrZero layerR.bias [out_ch.val]) (epsx := sumEps) (epsy := epsB)
        hSum hBiasVal
    simpa [addEps] using this

  have hsumR_nested :
      (List.finRange inC).foldl (fun acc in_ch =>
          (List.finRange kH).foldl (fun acc di =>
              (List.finRange kW).foldl (fun acc dj => acc + termR (in_ch, di, dj)) acc) acc) 0
        = sumR := by
    simpa [sumR, idxs] using (foldl_finRange3_eq_flat_foldl (term := termR))

  have houtR :
      getAtOrZero (Spec.conv2dSpec (α := R) (layer := layerR) (input := inputR)) [out_ch.val,
        i.val, j.val] =
        sumR + getAtOrZero layerR.bias [out_ch.val] := by
    have hNestedMulR :
        (List.finRange inC).foldl (fun acc (in_ch : Fin inC) =>
            (List.finRange kH).foldl (fun acc (di : Fin kH) =>
                (List.finRange kW).foldl (fun acc (dj : Fin kW) =>
                    acc +
                      getAtOrZero paddedR [↑in_ch, ↑i * stride + ↑di, ↑j * stride + ↑dj] *
                        getAtOrZero layerR.kernel [↑out_ch, ↑in_ch, ↑di, ↑dj])
                  acc)
              acc)
          0
          =
        (List.finRange inC).foldl (fun acc (in_ch : Fin inC) =>
            (List.finRange kH).foldl (fun acc (di : Fin kH) =>
                (List.finRange kW).foldl (fun acc (dj : Fin kW) => acc + termR (in_ch, di, dj)) acc)
              acc)
          0 := by
      refine
        foldl_congr (l := List.finRange inC)
          (f := fun acc (in_ch : Fin inC) =>
            (List.finRange kH).foldl (fun acc (di : Fin kH) =>
                (List.finRange kW).foldl (fun acc (dj : Fin kW) =>
                    acc +
                      getAtOrZero paddedR [↑in_ch, ↑i * stride + ↑di, ↑j * stride + ↑dj] *
                        getAtOrZero layerR.kernel [↑out_ch, ↑in_ch, ↑di, ↑dj])
                  acc)
              acc)
          (g := fun acc (in_ch : Fin inC) =>
            (List.finRange kH).foldl (fun acc (di : Fin kH) =>
                (List.finRange kW).foldl (fun acc (dj : Fin kW) => acc + termR (in_ch, di, dj)) acc)
              acc)
          (init := (0 : R)) ?_
      intro acc in_ch
      refine
        foldl_congr (l := List.finRange kH)
          (f := fun acc (di : Fin kH) =>
            (List.finRange kW).foldl (fun acc (dj : Fin kW) =>
                acc +
                  getAtOrZero paddedR [↑in_ch, ↑i * stride + ↑di, ↑j * stride + ↑dj] *
                    getAtOrZero layerR.kernel [↑out_ch, ↑in_ch, ↑di, ↑dj])
              acc)
          (g := fun acc (di : Fin kH) =>
            (List.finRange kW).foldl (fun acc (dj : Fin kW) => acc + termR (in_ch, di, dj)) acc)
          (init := acc) ?_
      intro acc di
      refine
        foldl_congr (l := List.finRange kW)
          (f := fun acc (dj : Fin kW) =>
            acc +
              getAtOrZero paddedR [↑in_ch, ↑i * stride + ↑di, ↑j * stride + ↑dj] *
                getAtOrZero layerR.kernel [↑out_ch, ↑in_ch, ↑di, ↑dj])
          (g := fun acc (dj : Fin kW) => acc + termR (in_ch, di, dj))
          (init := acc) ?_
      intro acc dj
      simp [termR]

    have hFoldMulR :
        (List.finRange inC).foldl (fun acc (in_ch : Fin inC) =>
            (List.finRange kH).foldl (fun acc (di : Fin kH) =>
                (List.finRange kW).foldl (fun acc (dj : Fin kW) =>
                    acc +
                      getAtOrZero paddedR [↑in_ch, ↑i * stride + ↑di, ↑j * stride + ↑dj] *
                        getAtOrZero layerR.kernel [↑out_ch, ↑in_ch, ↑di, ↑dj])
                  acc)
              acc)
          0
          = sumR := by
      calc
        _ =
            (List.finRange inC).foldl (fun acc (in_ch : Fin inC) =>
                (List.finRange kH).foldl (fun acc (di : Fin kH) =>
                    (List.finRange kW).foldl (fun acc (dj : Fin kW) => acc + termR (in_ch, di, dj))
                      acc)
                  acc)
              0 := hNestedMulR
        _ = sumR := hsumR_nested

    -- `conv2dSpec` defines its padded input via an `if`; rewrite those reads to our `paddedR`
    -- abbreviation so we can reuse `hFoldMulR` without reintroducing proof-term noise.
    have convInputVal_eq_paddedR (in_ch : Fin inC) (di : Fin kH) (dj : Fin kW) :
        (match Spec.Private.mkInputIdx? [i.val, j.val] [di.val, dj.val] [stride, stride]
              [padding, padding] with
          | none => (0 : R)
          | some inIdx => getAtOrZero inputR (in_ch.val :: inIdx))
          =
        getAtOrZero paddedR [in_ch.val, i.val * stride + di.val, j.val * stride + dj.val] := by
      simpa [paddedR, -tensor_cast_eq_cast_shape] using
        (conv_input_val_eq_padded (img := inputR) (c := in_ch) (out_i := i.val) (out_j := j.val)
          (di := di.val) (dj := dj.val) (stride := stride) (padding := padding))

    have houtR0 := by
      simpa [convInputVal_eq_paddedR, -tensor_cast_eq_cast_shape] using
        (conv2dSpec_getAtOrZero_point (layer := layerR) (input := inputR) out_ch i j)
    refine houtR0.trans ?_
    simpa using congrArg (fun x => x + getAtOrZero layerR.bias [↑out_ch]) hFoldMulR
  have houtS :
      getAtOrZero (Spec.conv2dSpec (α := ℝ) (layer := layerS) (input := inputS)) [out_ch.val,
        i.val, j.val] =
      sumS + getAtOrZero layerS.bias [out_ch.val] := by
    have hsumS_nested :
        (List.finRange inC).foldl (fun acc in_ch =>
            (List.finRange kH).foldl (fun acc di =>
                (List.finRange kW).foldl (fun acc dj => acc + termS (in_ch, di, dj)) acc) acc) 0
          = sumS := by
      simpa [sumS, idxs] using (foldl_finRange3_eq_flat_foldl (term := termS))
    have hNestedMul :
        (List.finRange inC).foldl (fun acc (in_ch : Fin inC) =>
            (List.finRange kH).foldl (fun acc (di : Fin kH) =>
                (List.finRange kW).foldl (fun acc (dj : Fin kW) =>
                    acc +
                      getAtOrZero paddedS [↑in_ch, ↑i * stride + ↑di, ↑j * stride + ↑dj] *
                        getAtOrZero layerS.kernel [↑out_ch, ↑in_ch, ↑di, ↑dj])
                  acc)
              acc)
          0
          =
        (List.finRange inC).foldl (fun acc (in_ch : Fin inC) =>
            (List.finRange kH).foldl (fun acc (di : Fin kH) =>
                (List.finRange kW).foldl (fun acc (dj : Fin kW) => acc + termS (in_ch, di, dj)) acc)
              acc)
          0 := by
      refine
        foldl_congr (l := List.finRange inC)
          (f := fun acc (in_ch : Fin inC) =>
            (List.finRange kH).foldl (fun acc (di : Fin kH) =>
                (List.finRange kW).foldl (fun acc (dj : Fin kW) =>
                    acc +
                      getAtOrZero paddedS [↑in_ch, ↑i * stride + ↑di, ↑j * stride + ↑dj] *
                        getAtOrZero layerS.kernel [↑out_ch, ↑in_ch, ↑di, ↑dj])
                  acc)
              acc)
          (g := fun acc (in_ch : Fin inC) =>
            (List.finRange kH).foldl (fun acc (di : Fin kH) =>
                (List.finRange kW).foldl (fun acc (dj : Fin kW) => acc + termS (in_ch, di, dj)) acc)
              acc)
          (init := (0 : ℝ)) ?_
      intro acc in_ch
      refine
        foldl_congr (l := List.finRange kH)
          (f := fun acc (di : Fin kH) =>
            (List.finRange kW).foldl (fun acc (dj : Fin kW) =>
                acc +
                  getAtOrZero paddedS [↑in_ch, ↑i * stride + ↑di, ↑j * stride + ↑dj] *
                    getAtOrZero layerS.kernel [↑out_ch, ↑in_ch, ↑di, ↑dj])
              acc)
          (g := fun acc (di : Fin kH) =>
            (List.finRange kW).foldl (fun acc (dj : Fin kW) => acc + termS (in_ch, di, dj)) acc)
          (init := acc) ?_
      intro acc di
      refine
        foldl_congr (l := List.finRange kW)
          (f := fun acc (dj : Fin kW) =>
            acc +
              getAtOrZero paddedS [↑in_ch, ↑i * stride + ↑di, ↑j * stride + ↑dj] *
                getAtOrZero layerS.kernel [↑out_ch, ↑in_ch, ↑di, ↑dj])
          (g := fun acc (dj : Fin kW) => acc + termS (in_ch, di, dj))
          (init := acc) ?_
      intro acc dj
      simp [termS]

    have hFoldMul :
        (List.finRange inC).foldl (fun acc (in_ch : Fin inC) =>
            (List.finRange kH).foldl (fun acc (di : Fin kH) =>
                (List.finRange kW).foldl (fun acc (dj : Fin kW) =>
                    acc +
                      getAtOrZero paddedS [↑in_ch, ↑i * stride + ↑di, ↑j * stride + ↑dj] *
                        getAtOrZero layerS.kernel [↑out_ch, ↑in_ch, ↑di, ↑dj])
                  acc)
              acc)
          0
          = sumS := by
      calc
        _ = (List.finRange inC).foldl (fun acc (in_ch : Fin inC) =>
              (List.finRange kH).foldl (fun acc (di : Fin kH) =>
                  (List.finRange kW).foldl (fun acc (dj : Fin kW) => acc + termS (in_ch, di, dj))
                    acc)
                acc)
            0 := hNestedMul
        _ = sumS := hsumS_nested

    have convInputVal_eq_paddedS (in_ch : Fin inC) (di : Fin kH) (dj : Fin kW) :
        (match Spec.Private.mkInputIdx? [i.val, j.val] [di.val, dj.val] [stride, stride]
              [padding, padding] with
          | none => (0 : ℝ)
          | some inIdx => getAtOrZero inputS (in_ch.val :: inIdx))
          =
        getAtOrZero paddedS [in_ch.val, i.val * stride + di.val, j.val * stride + dj.val] := by
      simpa [paddedS, -tensor_cast_eq_cast_shape] using
        (conv_input_val_eq_padded (img := inputS) (c := in_ch) (out_i := i.val) (out_j := j.val)
          (di := di.val) (dj := dj.val) (stride := stride) (padding := padding))

    have houtS0 := by
      simpa [convInputVal_eq_paddedS, -tensor_cast_eq_cast_shape] using
        (conv2dSpec_getAtOrZero_point (layer := layerS) (input := inputS) out_ch i j)
    refine houtS0.trans ?_
    simpa using congrArg (fun x => x + getAtOrZero layerS.bias [↑out_ch]) hFoldMul

  have hFinal' :
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd)
              (getAtOrZero (Spec.conv2dSpec (α := R) (layer := layerR) (input := inputR))
                [out_ch.val, i.val, j.val]) -
            getAtOrZero (Spec.conv2dSpec (α := ℝ) (layer := layerS) (input := inputS))
              [out_ch.val, i.val, j.val]) ≤
        addEps (β := β) (fexp := fexp) (rnd := rnd) sumR (getAtOrZero layerR.bias [out_ch.val])
          sumEps epsB := by
    simpa [houtR, houtS, add_assoc, add_comm, add_left_comm] using hFinal

  simpa [conv2dPointBound, paddedR, idxs, termR, epsTerm, sumR, sumEps, addEps] using hFinal'

-- ---------------------------------------------------------------------------
-- Tensor-level forward bound: `Spec.conv2dSpec` is approximated within `linfNorm (conv2dBoundTensor ...)`.
-- ---------------------------------------------------------------------------

/--
Tensor-level forward approximation for Conv2D (`approxT`).

This lifts `approx_conv2d_point` to an `approxT` statement for the full output tensor, with a
global error `linfNorm (conv2dBoundTensor ...)` that upper-bounds all pointwise error entries.
-/
theorem approxT_conv2d_spec
    {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    {kernelS : Tensor ℝ (.dim outC (.dim inC (.dim kH (.dim kW .scalar))))}
    {kernelR : Tensor R (.dim outC (.dim inC (.dim kH (.dim kW .scalar))))}
    {biasS : Tensor ℝ (.dim outC .scalar)} {biasR : Tensor R (.dim outC .scalar)}
    {inputS : Spec.MultiChannelImage inC inH inW ℝ}
    {inputR : Spec.MultiChannelImage inC inH inW R}
    {epsK epsB epsX : ℝ}
    (hK : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) kernelS kernelR
      epsK)
    (hB : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) biasS biasR epsB)
    (hX : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) inputS inputR
      epsX) :
    let layerS : Spec.Conv2DSpec inC outC kH kW stride padding ℝ h1 h2 h3 :=
      { kernel := kernelS, bias := biasS }
    let layerR : Spec.Conv2DSpec inC outC kH kW stride padding R h1 h2 h3 :=
      { kernel := kernelR, bias := biasR }
    let outS := Spec.conv2dSpec (α := ℝ) (layer := layerS) (input := inputS)
    let outR := Spec.conv2dSpec (α := R) (layer := layerR) (input := inputR)
    let bT :=
      conv2dBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (layerR := layerR) (inputR := inputR)
        (epsK := epsK) (epsB := epsB) (epsX := epsX)
    approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) outS outR (linfNorm
      bT) := by
  intro layerS layerR outS outR bT
  classical
  have hε : 0 ≤ linfNorm bT := linf_norm_nonneg (t := bT)
  refine approxT_dim_of_forall (β := β) (fexp := fexp) (rnd := rnd) (n := outC)
    (s := .dim (conv2dOutH inH kH stride padding) (.dim (conv2dOutW inW kW stride padding)
      .scalar))
    (xS := outS) (xR := outR) (eps := linfNorm bT) hε ?_
  intro oc
  refine approxT_dim_of_forall (β := β) (fexp := fexp) (rnd := rnd) (n := conv2dOutH inH kH stride
    padding)
    (s := .dim (conv2dOutW inW kW stride padding) .scalar)
    (eps := linfNorm bT) hε ?_
  intro oi
  refine approxT_dim_of_forall (β := β) (fexp := fexp) (rnd := rnd) (n := conv2dOutW inW kW stride
    padding)
    (s := .scalar)
    (eps := linfNorm bT) hε ?_
  intro oj

  have hpt :=
    approx_conv2d_point (β := β) (fexp := fexp) (rnd := rnd)
      (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding) (inH
        := inH) (inW := inW)
      (h1 := h1) (h2 := h2) (h3 := h3)
      (kernelS := kernelS) (kernelR := kernelR) (biasS := biasS) (biasR := biasR) (inputS := inputS)
        (inputR := inputR)
      (epsK := epsK) (epsB := epsB) (epsX := epsX) hK hB hX oc oi oj

  have hEntry :
      abs (conv2dPointBound (β := β) (fexp := fexp) (rnd := rnd) (layerR := layerR) (inputR :=
        inputR)
        (epsK := epsK) (epsB := epsB) (epsX := epsX) oc oi oj) ≤ linfNorm bT := by
    let bToc :=
      match bT with
      | .dim f => f oc
    let bTocoi :=
      match bToc with
      | .dim f => f oi
    let bTocoiW :=
      match bTocoi with
      | .dim f => f oj
    have h0 : linfNorm bToc ≤ linfNorm bT := by
      simpa [bToc] using (linf_norm_le_get_dim (t := bT) oc)
    have h1' : linfNorm bTocoi ≤ linfNorm bToc := by
      simpa [bTocoi] using (linf_norm_le_get_dim (t := bToc) oi)
    have h2' : linfNorm bTocoiW ≤ linfNorm bTocoi := by
      simpa [bTocoiW] using (linf_norm_le_get_dim (t := bTocoi) oj)
    have hchain : linfNorm bTocoiW ≤ linfNorm bT := le_trans (le_trans h2' h1') h0
    have hdouble' :
        (MathFunctions.abs (abs (conv2dPointBound (β := β) (fexp := fexp) (rnd := rnd)
              (layerR := layerR) (inputR := inputR) (epsK := epsK) (epsB := epsB) (epsX := epsX) oc
                oi oj)) : ℝ)
          ≤ linfNorm bT := by
      simpa [bT, bToc, bTocoi, bTocoiW, conv2dBoundTensor, linfNorm, RuntimeApprox.linfNorm,
        tensorLinfNorm] using
        hchain
    have habs :
        (MathFunctions.abs (abs (conv2dPointBound (β := β) (fexp := fexp) (rnd := rnd)
              (layerR := layerR) (inputR := inputR) (epsK := epsK) (epsB := epsB) (epsX := epsX) oc
                oi oj)) : ℝ)
          =
        abs (abs (conv2dPointBound (β := β) (fexp := fexp) (rnd := rnd)
              (layerR := layerR) (inputR := inputR) (epsK := epsK) (epsB := epsB) (epsX := epsX) oc
                oi oj)) := by
      rfl
    have hdouble :
        abs (abs (conv2dPointBound (β := β) (fexp := fexp) (rnd := rnd)
              (layerR := layerR) (inputR := inputR) (epsK := epsK) (epsB := epsB) (epsX := epsX) oc
                oi oj)) ≤
          linfNorm bT := by
      simpa [habs] using hdouble'
    simpa [abs_abs] using hdouble

  have hscalar :
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd)
              (getAtOrZero outR [oc.val, oi.val, oj.val]) -
            getAtOrZero outS [oc.val, oi.val, oj.val]) ≤ linfNorm bT :=
    le_trans (le_trans hpt (le_abs_self _)) hEntry

  let outSentry :=
    (match
      match
        match outS with
        | .dim f => f oc with
      | .dim f => f oi with
    | .dim f => f oj)
  let outRentry :=
    (match
      match
        match outR with
        | .dim f => f oc with
      | .dim f => f oi with
    | .dim f => f oj)

  have hEntryS :
      outSentry = Tensor.scalar (getAtOrZero outS [oc.val, oi.val, oj.val]) := by
    dsimp [outSentry]
    exact entry_eq_scalar_get_at_or_zero3 (t := outS) oc oi oj
  have hEntryR :
      outRentry = Tensor.scalar (getAtOrZero outR [oc.val, oi.val, oj.val]) := by
    dsimp [outRentry]
    exact entry_eq_scalar_get_at_or_zero3 (t := outR) oc oi oj

  have happ :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Tensor.scalar (getAtOrZero outS [oc.val, oi.val, oj.val]))
        (Tensor.scalar (getAtOrZero outR [oc.val, oi.val, oj.val]))
        (linfNorm bT) :=
    (approxT_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (x := getAtOrZero outS [oc.val, oi.val, oj.val])
        (xR := getAtOrZero outR [oc.val, oi.val, oj.val])
        (eps := linfNorm bT)).2 (by
          simpa using hscalar)

  have happ' :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        outSentry outRentry (linfNorm bT) := by
    rw [hEntryS, hEntryR]
    exact happ

  simpa [outSentry, outRentry] using happ'

-- ---------------------------------------------------------------------------
-- `FwdNode` packaging for Conv2D.
-- ---------------------------------------------------------------------------

/--
Package Conv2D as a `FwdNode` for use in `FwdGraph.eval_approx`.

The node reads kernel/bias/input from the typed context `Γ`, runs the spec/runtime forward passes,
and uses `approxT_conv2d_spec` as its soundness proof.
-/
def conv2dNode
    {Γ : List Shape}
    {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    (kernelIdx : Idx Γ (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))))
    (biasIdx : Idx Γ (.dim outC .scalar))
    (inputIdx : Idx Γ (.dim inC (.dim inH (.dim inW .scalar)))) :
    FwdNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ
      (.dim outC (.dim (conv2dOutH inH kH stride padding) (.dim (conv2dOutW inW kW stride padding)
        .scalar))) :=
by
  classical
  refine
    { forwardSpec := fun ctx =>
        let kernelS := getIdx (α := SpecScalar) ctx kernelIdx
        let biasS := getIdx (α := SpecScalar) ctx biasIdx
        let inputS := getIdx (α := SpecScalar) ctx inputIdx
        let layerS : Spec.Conv2DSpec inC outC kH kW stride padding ℝ h1 h2 h3 :=
          { kernel := kernelS, bias := biasS }
        Spec.conv2dSpec (α := ℝ) (layer := layerS) (input := inputS)
      forwardRuntime := fun ctx =>
        let kernelR := getIdx (α := R) ctx kernelIdx
        let biasR := getIdx (α := R) ctx biasIdx
        let inputR := getIdx (α := R) ctx inputIdx
        let layerR : Spec.Conv2DSpec inC outC kH kW stride padding R h1 h2 h3 :=
          { kernel := kernelR, bias := biasR }
        Spec.conv2dSpec (α := R) (layer := layerR) (input := inputR)
      bound := fun epsCtx ctxR =>
        let kernelR := getIdx (α := R) ctxR kernelIdx
        let biasR := getIdx (α := R) ctxR biasIdx
        let inputR := getIdx (α := R) ctxR inputIdx
        let epsK := getIdxEps (Γ := Γ) (s := (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))))
          epsCtx kernelIdx
        let epsB := getIdxEps (Γ := Γ) (s := (.dim outC .scalar)) epsCtx biasIdx
        let epsX := getIdxEps (Γ := Γ) (s := (.dim inC (.dim inH (.dim inW .scalar)))) epsCtx
          inputIdx
        let layerR : Spec.Conv2DSpec inC outC kH kW stride padding R h1 h2 h3 :=
          { kernel := kernelR, bias := biasR }
        linfNorm
          (conv2dBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (layerR := layerR) (inputR := inputR) (epsK := epsK) (epsB := epsB) (epsX := epsX))
      sound := by
        intro ctxS ctxR epsCtx hctx
        have hK := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          hctx kernelIdx
        have hB := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          hctx biasIdx
        have hX := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          hctx inputIdx
        simpa using
          (approxT_conv2d_spec (β := β) (fexp := fexp) (rnd := rnd)
            (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding :=
              padding) (inH := inH) (inW := inW)
            (h1 := h1) (h2 := h2) (h3 := h3)
            (kernelS := getIdx (α := SpecScalar) ctxS kernelIdx)
            (kernelR := getIdx (α := R) ctxR kernelIdx)
            (biasS := getIdx (α := SpecScalar) ctxS biasIdx)
            (biasR := getIdx (α := R) ctxR biasIdx)
            (inputS := getIdx (α := SpecScalar) ctxS inputIdx)
            (inputR := getIdx (α := R) ctxR inputIdx)
            (epsK := getIdxEps (Γ := Γ) (s := (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))))
              epsCtx kernelIdx)
            (epsB := getIdxEps (Γ := Γ) (s := (.dim outC .scalar)) epsCtx biasIdx)
            (epsX := getIdxEps (Γ := Γ) (s := (.dim inC (.dim inH (.dim inW .scalar)))) epsCtx
              inputIdx)
            hK hB hX) }

end NFBackend

end
end RuntimeApprox
end Proofs
