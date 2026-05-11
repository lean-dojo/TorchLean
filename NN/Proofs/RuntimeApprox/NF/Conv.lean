/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.BackwardOps
public import NN.Proofs.RuntimeApprox.NF.Ops
public import NN.Proofs.RuntimeApprox.NF.Utils
public import NN.Spec.Layers.Conv

/-!
# Conv2D Shared Bounds

Shared NF backend utilities for Conv2D forward/backward runtime-to-spec approximation.

This file contains the common arithmetic error updates and tensor-shaped Conv2D bound builders used
by both `ConvForward` and `ConvBackward`. The node constructors live in the direction-specific
files; the shared definitions stay here so the forward and VJP proofs do not duplicate the same
fold/error algebra.

The key idea is to treat Conv2D as a pure spec definition (`Spec.conv2d_spec` /
`Spec.conv2d_backward_spec`) instantiated at:
- spec scalars: `ℝ`
- runtime scalars: `NF β fexp rnd` (rounding after each primitive op)

and then prove explicit bounds of the form:

  `toSpec (runtime_result) ≈ spec_result`

compatible with the tape/DAG composition theorems in:
- `NN.Proofs.RuntimeApprox.Graph.ForwardApprox`
- `NN.Proofs.RuntimeApprox.Graph.BackwardApprox`

## PyTorch correspondence / citations
Our image tensor shape is `C × H × W` (no batch dimension), and the kernel shape is
`outC × inC × kH × kW`. This matches the per-example core of PyTorch’s `conv2d` (which normally
expects `N × C × H × W` inputs and supports extra features like groups).
https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
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
open Proofs.RuntimeRoundingApprox

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]
variable {rnd : ℝ → ℤ} [NeuralValidRndToNearest rnd]

local notation "R" => TorchLean.Floats.NF β fexp rnd

/-! ## Small helper bounds for list folds -/

/-- Absolute-error update for one rounded addition in the NF backend. -/
def addEps (accR termR : R) (epsAcc epsTerm : ℝ) : ℝ :=
  epsAcc + epsTerm +
    neuralUlp β fexp
      (toSpec (β := β) (fexp := fexp) (rnd := rnd) accR +
        toSpec (β := β) (fexp := fexp) (rnd := rnd) termR)
      TrainingPhase.forward / 2

/-- Absolute-error update for one rounded multiplication in the NF backend. -/
def mulEps (xR yR : R) (epsx epsy : ℝ) : ℝ :=
  let xhat := toSpec (β := β) (fexp := fexp) (rnd := rnd) xR
  let yhat := toSpec (β := β) (fexp := fexp) (rnd := rnd) yR
  (abs xhat + epsx) * epsy + (abs yhat + epsy) * epsx +
    neuralUlp β fexp (xhat * yhat) TrainingPhase.forward / 2

/-- Fold a rounded sum while tracking an absolute-error budget, starting from an explicit state. -/
def foldAddStateFrom {ι : Type} (l : List ι) (termR : ι → R) (epsTerm : ι → ℝ) (st0 : R × ℝ) : R ×
  ℝ :=
  l.foldl
    (fun st i =>
      let accR := st.1
      let epsAcc := st.2
      let tR := termR i
      let eT := epsTerm i
      (accR + tR, addEps (β := β) (fexp := fexp) (rnd := rnd) accR tR epsAcc eT))
    st0

/-- `foldAddStateFrom` started at `(0,0)`. -/
def foldAddState {ι : Type} (l : List ι) (termR : ι → R) (epsTerm : ι → ℝ) : R × ℝ :=
  foldAddStateFrom (β := β) (fexp := fexp) (rnd := rnd) l termR epsTerm ((0 : R), (0 : ℝ))

lemma approx_fold_add_state {ι : Type} (l : List ι)
    (termS : ι → ℝ) (termR : ι → R) (epsTerm : ι → ℝ) :
    ∀ (accS : ℝ) (st : R × ℝ),
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) st.1 - accS) ≤ st.2 →
      (∀ i ∈ l, abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (termR i) - termS i) ≤ epsTerm i) →
        abs
            (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                ((l.foldl (fun acc i => acc + termR i) st.1)) -
              (l.foldl (fun acc i => acc + termS i) accS)) ≤
          (foldAddStateFrom (β := β) (fexp := fexp) (rnd := rnd) l termR epsTerm st).2 := by
  intro accS st hAcc hTerm
  induction l generalizing accS st with
  | nil =>
      simp at hAcc ⊢
      simpa using hAcc
  | cons hd tl ih =>
      have hHd : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (termR hd) - termS hd) ≤ epsTerm
        hd :=
        hTerm hd (by simp)
      have hNext :
          abs
              (toSpec (β := β) (fexp := fexp) (rnd := rnd) (st.1 + termR hd) - (accS + termS hd)) ≤
            addEps (β := β) (fexp := fexp) (rnd := rnd) st.1 (termR hd) st.2 (epsTerm hd) := by
        simpa [addEps, add_assoc, add_left_comm, add_comm] using
          (approx_add_nf (β := β) (fexp := fexp) (rnd := rnd)
            (x := accS) (y := termS hd) (xR := st.1) (yR := termR hd)
            (epsx := st.2) (epsy := epsTerm hd) hAcc hHd)
      -- apply IH on the tail, with the updated accumulator/state
      have hTail :
          abs
              (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                  (tl.foldl (fun acc i => acc + termR i) (st.1 + termR hd)) -
                tl.foldl (fun acc i => acc + termS i) (accS + termS hd)) ≤
            (foldAddStateFrom (β := β) (fexp := fexp) (rnd := rnd) tl termR epsTerm (st.1 + termR
              hd,
              addEps (β := β) (fexp := fexp) (rnd := rnd) st.1 (termR hd) st.2 (epsTerm hd))).2 :=
                by
        refine ih (accS := accS + termS hd)
          (st := (st.1 + termR hd, addEps (β := β) (fexp := fexp) (rnd := rnd) st.1 (termR hd) st.2
            (epsTerm hd)))
          hNext ?_
        intro i hi
        exact hTerm i (by simp [hi])
      -- rewrite the unfolded state fold on the full list
      simpa [foldAddStateFrom, List.foldl, add_assoc, add_left_comm, add_comm] using hTail

-- Convenience corollary: start the fold from 0.
lemma approx_fold_add {ι : Type} (l : List ι)
    (termS : ι → ℝ) (termR : ι → R) (epsTerm : ι → ℝ)
    (hTerm : ∀ i ∈ l, abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (termR i) - termS i) ≤
      epsTerm i) :
    abs
        (toSpec (β := β) (fexp := fexp) (rnd := rnd)
            (l.foldl (fun acc i => acc + termR i) (0 : R)) -
          l.foldl (fun acc i => acc + termS i) (0 : ℝ)) ≤
      (foldAddState (β := β) (fexp := fexp) (rnd := rnd) l termR epsTerm).2 := by
  have h0 : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (0 : R) - (0 : ℝ)) ≤ (0 : ℝ) := by
    simp [toSpec_zero (β := β) (fexp := fexp) (rnd := rnd)]
  simpa [foldAddState, foldAddStateFrom] using
    (approx_fold_add_state (β := β) (fexp := fexp) (rnd := rnd) l termS termR epsTerm (accS := (0 :
      ℝ))
      (st := ((0 : R), (0 : ℝ))) h0 hTerm)

/-! ## Component access: `getAtOrZero` respects `approxT` -/

lemma approx_get_at_or_zero {s : Shape} :
    ∀ {xS : SpecTensor s} {xR : Tensor R s} {eps : ℝ} (_hx :
        approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps)
      (idx : List Nat),
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (getAtOrZero xR idx) - getAtOrZero xS
        idx) ≤ eps := by
  intro xS xR eps _hx idx
  classical
  induction s generalizing idx with
  | scalar =>
      cases xS with
      | scalar x =>
          cases xR with
          | scalar xR =>
              cases idx with
              | nil =>
                  simpa using
                    (approxT_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                      rnd))
                      (x := x) (xR := xR) (eps := eps)).1 _hx
              | cons i is =>
                  have heps : 0 ≤ eps := approxT_eps_nonneg (β := β) (fexp := fexp) (rnd := rnd) _hx
                  -- Both sides are `0` for out-of-shape indices.
                  simpa [get_at_or_zero_scalar_cons, toSpec_zero (β := β) (fexp := fexp) (rnd :=
                    rnd)] using heps
  | dim n s ih =>
      cases xS with
      | dim xSf =>
          cases xR with
          | dim xRf =>
              cases idx with
              | nil =>
                  have heps : 0 ≤ eps := approxT_eps_nonneg (β := β) (fexp := fexp) (rnd := rnd) _hx
                  -- Out-of-shape: both reads are `0`.
                  simpa [get_at_or_zero_dim_nil, toSpec_zero (β := β) (fexp := fexp) (rnd := rnd)]
                    using heps
              | cons i is =>
                  by_cases hi : i < n
                  · -- In-bounds: reduce to the IH on the selected slice.
                    let fi : Fin n := ⟨i, hi⟩
                    have hx_i :
                        approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                          (xSf fi) (xRf fi) eps := by
                      -- `approxT_dim_get` gives component-wise approximation with the same `eps`.
                      have := approxT_dim_get (α := R) (toSpec := toSpec (β := β) (fexp := fexp)
                        (rnd := rnd))
                        (xS := Tensor.dim xSf) (xR := Tensor.dim xRf) (eps := eps) _hx fi
                      simpa using this
                    simpa [get_at_or_zero_dim_cons, hi] using ih (xS := xSf fi) (xR := xRf fi) hx_i
                      is
                  · -- Out-of-bounds: both reads are `0`.
                    have heps : 0 ≤ eps := approxT_eps_nonneg (β := β) (fexp := fexp) (rnd := rnd)
                      _hx
                    have hn : ¬ i < n := hi
                    simpa [get_at_or_zero_dim_cons, hn, toSpec_zero (β := β) (fexp := fexp) (rnd :=
                      rnd)] using heps

/-! ## Padding reads -/

/--
Approximation bound for reading padded inputs.

Both the forward and backward Conv2D approximation proofs reduce to reading a padded input tensor
at indices `(c,p,q)` and comparing the runtime read (in `R`) against the spec read (in `ℝ`).
-/
lemma approx_padded_input_read
    {inC inH inW padding : Nat}
    {xS : Spec.MultiChannelImage inC inH inW ℝ}
    {xR : Spec.MultiChannelImage inC inH inW R}
    {epsX : ℝ}
    (hx : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR epsX)
    (c : Fin inC) (p q : Nat) :
    abs
        (toSpec (β := β) (fexp := fexp) (rnd := rnd)
            (getAtOrZero
              (if h4 : padding = 0 then
                tensorCast
                  (Shape.dim inC (Shape.dim (inH + 2 * padding) (Shape.dim (inW + 2 * padding)
                    .scalar)))
                  (by simp; rw [h4])
                  xR
              else
                Spec.padMultiChannel xR padding)
              [c.val, p, q]) -
          getAtOrZero
            (if h4 : padding = 0 then
              tensorCast
                (Shape.dim inC (Shape.dim (inH + 2 * padding) (Shape.dim (inW + 2 * padding)
                  .scalar)))
                (by simp; rw [h4])
                xS
            else
              Spec.padMultiChannel xS padding)
            [c.val, p, q]) ≤ epsX := by
  classical
  by_cases h4 : padding = 0
  · subst h4
    simpa using
      (approx_get_at_or_zero (β := β) (fexp := fexp) (rnd := rnd)
        (s := Shape.dim inC (Shape.dim inH (Shape.dim inW .scalar))) (xS := xS) (xR := xR) (eps :=
          epsX) hx
        [c.val, p, q])
  · have hR :=
      Spec.get_at_or_zero_pad_multi_channel (α := R) (img := xR) (c := c) (p := p) (q := q) (padding
        := padding)
    have hS :=
      Spec.get_at_or_zero_pad_multi_channel (α := ℝ) (img := xS) (c := c) (p := p) (q := q) (padding
        := padding)
    by_cases ht : p < padding ∨ q < padding
    · have heps : 0 ≤ epsX := approxT_eps_nonneg (β := β) (fexp := fexp) (rnd := rnd) hx
      simpa [h4, ht, hR, hS] using heps
    · have hcore :=
        approx_get_at_or_zero (β := β) (fexp := fexp) (rnd := rnd)
          (s := Shape.dim inC (Shape.dim inH (Shape.dim inW .scalar))) (xS := xS) (xR := xR)
          (eps := epsX) hx [c.val, p - padding, q - padding]
      have ht' : ¬(p < padding ∨ q < padding) := ht
      have hp : padding ≤ p := by
        have : ¬ p < padding := by
          intro hp
          exact ht' (Or.inl hp)
        exact Nat.le_of_not_gt this
      have hq : padding ≤ q := by
        have : ¬ q < padding := by
          intro hq
          exact ht' (Or.inr hq)
        exact Nat.le_of_not_gt this
      simpa [h4, ht, hR, hS, Nat.sub_add_cancel hp, Nat.sub_add_cancel hq] using hcore

/-! ## Conv2D forward: pointwise error bound for one output scalar -/

/-- Output height for a 2D convolution. -/
def conv2dOutH (inH kH stride padding : Nat) : Nat :=
  (inH + 2 * padding - kH) / stride + 1

/-- Output width for a 2D convolution. -/
def conv2dOutW (inW kW stride padding : Nat) : Nat :=
  (inW + 2 * padding - kW) / stride + 1

/--
Pointwise absolute-error bound for one Conv2D output scalar in the NF backend.

Implementation note:
we replay the same fold structure as `Spec.conv2d_spec`, but track an explicit absolute-error
budget alongside the runtime accumulator. This keeps the bound aligned with the runtime addition
order (which matters for worst-case rounding error).
-/
def conv2dPointBound
    {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    (layerR : Spec.Conv2DSpec inC outC kH kW stride padding R h1 h2 h3)
    (inputR : Spec.MultiChannelImage inC inH inW R)
    (epsK epsB epsX : ℝ)
    (out_ch : Fin outC) (i : Fin (conv2dOutH inH kH stride padding)) (j : Fin (conv2dOutW inW kW
      stride padding)) :
    ℝ :=
  let padded_inputR :=
    if h4 : padding = 0 then
      tensorCast
        (Shape.dim inC (Shape.dim (inH + 2 * padding) (Shape.dim (inW + 2 * padding) .scalar)))
        (by simp; rw [h4])
        inputR
    else
      Spec.padMultiChannel inputR padding
  let idxs : List (Fin inC × Fin kH × Fin kW) :=
    (List.finRange inC).flatMap (fun in_ch =>
      (List.finRange kH).flatMap (fun di =>
        (List.finRange kW).map (fun dj => (in_ch, di, dj))))
  let termR : (Fin inC × Fin kH × Fin kW) → R := fun t =>
    let in_ch : Fin inC := t.1
    let di : Fin kH := t.2.1
    let dj : Fin kW := t.2.2
    let input_val := getAtOrZero padded_inputR [in_ch.val, i.val * stride + di.val, j.val *
      stride + dj.val]
    let kernel_val := getAtOrZero layerR.kernel [out_ch.val, in_ch.val, di.val, dj.val]
    input_val * kernel_val
  let epsTerm : (Fin inC × Fin kH × Fin kW) → ℝ := fun t =>
    let in_ch : Fin inC := t.1
    let di : Fin kH := t.2.1
    let dj : Fin kW := t.2.2
    let input_val := getAtOrZero padded_inputR [in_ch.val, i.val * stride + di.val, j.val *
      stride + dj.val]
    let kernel_val := getAtOrZero layerR.kernel [out_ch.val, in_ch.val, di.val, dj.val]
    mulEps (β := β) (fexp := fexp) (rnd := rnd) input_val kernel_val epsX epsK
  let sumR : R := idxs.foldl (fun acc t => acc + termR t) 0
  let sumEps : ℝ := (foldAddState (β := β) (fexp := fexp) (rnd := rnd) idxs termR epsTerm).2
  -- Add bias.
  let bias_val := getAtOrZero layerR.bias [out_ch.val]
  addEps (β := β) (fexp := fexp) (rnd := rnd) sumR bias_val sumEps epsB

/-- Tensor of pointwise Conv2D absolute-error bounds for the full output image. -/
def conv2dBoundTensor
    {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    (layerR : Spec.Conv2DSpec inC outC kH kW stride padding R h1 h2 h3)
    (inputR : Spec.MultiChannelImage inC inH inW R)
    (epsK epsB epsX : ℝ) :
    Spec.MultiChannelImage outC (conv2dOutH inH kH stride padding) (conv2dOutW inW kW stride
      padding) ℝ :=
  Tensor.dim (fun out_ch =>
    Tensor.dim (fun i =>
      Tensor.dim (fun j =>
        Tensor.scalar <|
          abs <|
            conv2dPointBound (β := β) (fexp := fexp) (rnd := rnd)
              (layerR := layerR) (inputR := inputR) (epsK := epsK) (epsB := epsB) (epsX := epsX)
                out_ch i j)))

/-!
`conv2dPointBound` and `conv2dBoundTensor` provide the *data* needed for a Conv2D
runtime→spec approximation lemma, but the full end-to-end `FwdNode`/`RevNode` instances are
intentionally deferred.

Reason: a direct proof that these bounds are sound for the current `Spec.conv2d_spec` definition
is non-trivial and requires careful performance engineering (similar to the Conv2D `fderiv` file).
The design intent is to keep Conv2D as a "big op" with its own proof module, instead of slowing
down the core NF numeric library.
-/

end NFBackend

end
end RuntimeApprox
end Proofs
