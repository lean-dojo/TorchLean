/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Core
public import NN.API.TList
public import NN.API.Tensor.Views
public import NN.MLTheory.SelfSupervised.PredictiveView

/-!
# Self-Supervised Training API

Self-supervised learning is primarily a **training objective and data-view** interface, not a special
kind of layer.

This module is the public, model-independent SSL surface:

- it turns ordinary typed tensors into supervised training samples whose targets are derived from
  the input itself;
- it exposes deterministic masks that line up with the finite-mask theory in
  `NN.MLTheory.API`;
- it stays independent of any particular encoder, so the same SSL sample/objective helpers can be
  used with an MLP, CNN, ViT, ResNet, or a custom `nn.Sequential`.

Architecture constructors, when useful, live under `NN.API.Models.*`.  For example, a compact
vector autoencoder is convenient for runnable CIFAR smoke tests, but the MAE idea itself belongs
here: create a masked view of a tensor and reconstruct the original content.
-/

@[expose] public section

namespace NN
namespace API

open Spec Tensor

namespace ssl

/-! ## Compact MAE-style masked reconstruction -/

/--
The hidden-coordinate mask used by compact MAE training.

`true` means "this feature coordinate is hidden from the encoder."  The type is the finite-mask
type used in the ML-theory files, specialized to the feature axis of a `batch × dataDim` matrix.
-/
def vectorMaeHiddenMask (dataDim period offset : Nat) :
    NN.MLTheory.SelfSupervised.Mask dataDim :=
  fun j =>
    if period = 0 then
      false
    else
      decide (j.val % period = offset % period)

private theorem vectorMaeHiddenMask_selected_iff (dataDim period offset : Nat) (j : Fin dataDim) :
    NN.MLTheory.SelfSupervised.selected (vectorMaeHiddenMask dataDim period offset) j ↔
      period ≠ 0 ∧ j.val % period = offset % period := by
  by_cases hp : period = 0
  · simp [NN.MLTheory.SelfSupervised.selected, vectorMaeHiddenMask, hp]
  · simp [NN.MLTheory.SelfSupervised.selected, vectorMaeHiddenMask, hp]

/--
Feature-level deterministic mask for MAE samples over a `batch × dataDim` matrix.

Every coordinate whose index is congruent to `offset` modulo `period` is hidden by setting it to
zero. This is intentionally deterministic so examples and tests are reproducible.
-/
def vectorMaeMask (batch dataDim period offset : Nat)
    (x : Spec.Tensor Float (NN.Tensor.Shape.Mat batch dataDim)) :
    Spec.Tensor Float (NN.Tensor.Shape.Mat batch dataDim) :=
  Spec.Tensor.dim (fun bi =>
    let row := Spec.getAtSpec x bi
    Spec.Tensor.dim (fun j =>
      let v := Spec.Tensor.toScalar (Spec.get row j)
      let keep :=
        if period = 0 then
          true
        else
          j.val % period ≠ offset % period
      Spec.Tensor.scalar (if keep then v else 0.0)))

/--
Coordinate-level bridge from the executable tensor mask to the finite mask used in the
self-supervised theory files.

For every batch row and feature coordinate, `vectorMaeMask` returns exactly zero on hidden
coordinates and the original tensor value on visible coordinates.
-/
theorem vectorMaeMask_get_eq_if_selected_hidden (batch dataDim period offset : Nat)
    (x : Spec.Tensor Float (NN.Tensor.Shape.Mat batch dataDim))
    (bi : Fin batch) (j : Fin dataDim) :
    Spec.Tensor.toScalar (Spec.get (Spec.get (vectorMaeMask batch dataDim period offset x) bi) j) =
      if vectorMaeHiddenMask dataDim period offset j then
        0.0
      else
        Spec.Tensor.toScalar (Spec.get (Spec.get x bi) j) := by
  by_cases hp : period = 0
  · cases x with
    | dim rows =>
        simp [vectorMaeMask, vectorMaeHiddenMask, Spec.get, Spec.getAtSpec, Spec.Tensor.toScalar, hp]
  · by_cases hj : j.val % period = offset % period
    · cases x with
      | dim rows =>
          simp [vectorMaeMask, vectorMaeHiddenMask, Spec.get, Spec.getAtSpec, Spec.Tensor.toScalar,
            hp, hj]
    · cases x with
      | dim rows =>
          simp [vectorMaeMask, vectorMaeHiddenMask, Spec.get, Spec.getAtSpec, Spec.Tensor.toScalar,
            hp, hj]

/-- Hidden feature coordinates are exactly zero after applying `vectorMaeMask`. -/
theorem vectorMaeMask_hidden_get_eq_zero (batch dataDim period offset : Nat)
    (x : Spec.Tensor Float (NN.Tensor.Shape.Mat batch dataDim))
    (bi : Fin batch) (j : Fin dataDim)
    (h : NN.MLTheory.SelfSupervised.selected (vectorMaeHiddenMask dataDim period offset) j) :
    Spec.Tensor.toScalar (Spec.get (Spec.get (vectorMaeMask batch dataDim period offset x) bi) j) =
      0.0 := by
  have hbool : vectorMaeHiddenMask dataDim period offset j = true := by
    simpa [NN.MLTheory.SelfSupervised.selected] using h
  simpa [hbool] using
    vectorMaeMask_get_eq_if_selected_hidden batch dataDim period offset x bi j

/-- Visible feature coordinates are preserved by `vectorMaeMask`. -/
theorem vectorMaeMask_visible_get_eq_input (batch dataDim period offset : Nat)
    (x : Spec.Tensor Float (NN.Tensor.Shape.Mat batch dataDim))
    (bi : Fin batch) (j : Fin dataDim)
    (h : ¬ NN.MLTheory.SelfSupervised.selected (vectorMaeHiddenMask dataDim period offset) j) :
    Spec.Tensor.toScalar (Spec.get (Spec.get (vectorMaeMask batch dataDim period offset x) bi) j) =
      Spec.Tensor.toScalar (Spec.get (Spec.get x bi) j) := by
  have hbool : vectorMaeHiddenMask dataDim period offset j = false := by
    cases hm : vectorMaeHiddenMask dataDim period offset j <;>
      simp [NN.MLTheory.SelfSupervised.selected, hm] at h ⊢
  simpa [hbool] using
    vectorMaeMask_get_eq_if_selected_hidden batch dataDim period offset x bi j

/--
Build a compact MAE training sample from a vector batch.

The model sees the masked vector and reconstructs the original vector.  This is represented using
TorchLean's existing supervised sample type because the "label" is derived from the input.
-/
def vectorMaeSample (batch dataDim period offset : Nat)
    (x : Spec.Tensor Float (NN.Tensor.Shape.Mat batch dataDim)) :
    sample.Supervised Float (NN.Tensor.Shape.Mat batch dataDim) (NN.Tensor.Shape.Mat batch dataDim) :=
  sample.mk (vectorMaeMask batch dataDim period offset x) x

/--
The executable vector MAE training input is exactly the masked tensor.

This is the whole-tensor statement behind the coordinate theorems below. When the runtime training
loop calls `sample.x`, it receives this tensor and no other preprocessing is hidden in the sample
wrapper.
-/
theorem vectorMaeSample_input_eq_mask (batch dataDim period offset : Nat)
    (x : Spec.Tensor Float (NN.Tensor.Shape.Mat batch dataDim)) :
    sample.x (vectorMaeSample batch dataDim period offset x) =
      vectorMaeMask batch dataDim period offset x := by
  simp [vectorMaeSample]

/--
The executable vector MAE training target is exactly the original tensor.

Together with `vectorMaeSample_input_eq_mask`, this says the fixed-sample training call compares a
model output against the unmasked source tensor.
-/
theorem vectorMaeSample_target_eq_source (batch dataDim period offset : Nat)
    (x : Spec.Tensor Float (NN.Tensor.Shape.Mat batch dataDim)) :
    sample.y (vectorMaeSample batch dataDim period offset x) = x := by
  simp [vectorMaeSample]

/-! ## Tensor-to-theory bridge for predictive-view SSL -/

/--
The finite hidden-index list induced by the executable vector MAE mask.

This is the serialization of the masked coordinate set used by the finite MAE/predictive-view
objective. The tensor API uses the Boolean mask directly; the theory objective sums over a list.
-/
def vectorMaeSelectedIdxs (dataDim period offset : Nat) : List (Fin dataDim) :=
  (List.finRange dataDim).filter (fun j => vectorMaeHiddenMask dataDim period offset j)

/-- Extract one runtime tensor row as the finite patch batch used by the SSL theory layer. -/
def matrixRowAsPatchBatch (batch n : Nat)
    (x : Spec.Tensor Float (NN.Tensor.Shape.Mat batch n)) (bi : Fin batch) :
    NN.MLTheory.SelfSupervised.PatchBatch n Float :=
  fun j => Spec.Tensor.toScalar (Spec.get (Spec.get x bi) j)

/-- Extract one runtime prediction row as a finite prediction function. -/
def matrixRowAsPrediction (batch n : Nat)
    (yhat : Spec.Tensor Float (NN.Tensor.Shape.Mat batch n)) (bi : Fin batch) :
    Fin n → Float :=
  fun j => Spec.Tensor.toScalar (Spec.get (Spec.get yhat bi) j)

/--
The tensor MAE sample keeps the original row as the finite theory target.

The target row is exactly the patch batch appearing in the finite MAE/predictive-view objective.
-/
theorem vectorMaeSample_target_row_eq_source_row (batch dataDim period offset : Nat)
    (x : Spec.Tensor Float (NN.Tensor.Shape.Mat batch dataDim))
    (bi : Fin batch) :
    matrixRowAsPatchBatch batch dataDim (sample.y (vectorMaeSample batch dataDim period offset x)) bi =
      matrixRowAsPatchBatch batch dataDim x bi := by
  funext j
  simp [matrixRowAsPatchBatch, vectorMaeSample]

/--
The tensor MAE sample input has zero at every finite hidden coordinate.

This is the executable masking invariant seen by the model before the theory objective asks it to
predict those original target coordinates back.
-/
theorem vectorMaeSample_input_hidden_get_eq_zero (batch dataDim period offset : Nat)
    (x : Spec.Tensor Float (NN.Tensor.Shape.Mat batch dataDim))
    (bi : Fin batch) (j : Fin dataDim)
    (h : NN.MLTheory.SelfSupervised.selected (vectorMaeHiddenMask dataDim period offset) j) :
    Spec.Tensor.toScalar
        (Spec.get (Spec.get (sample.x (vectorMaeSample batch dataDim period offset x)) bi) j) = 0.0 := by
  simpa [vectorMaeSample] using
    vectorMaeMask_hidden_get_eq_zero batch dataDim period offset x bi j h

/--
A single row of the executable vector MAE path instantiates the finite predictive-view contract.

`yhat` is the model output tensor. After extracting row `bi`, the finite objective is precisely the
MAE masked reconstruction loss over the selected hidden coordinates. This is the key bridge from
`Spec.Tensor` implementation data to the SSL objective algebra.
-/
def vectorMaeRowPredictiveContract (batch dataDim period offset : Nat)
    (x yhat : Spec.Tensor Float (NN.Tensor.Shape.Mat batch dataDim))
    (bi : Fin batch) (patchLoss : Float → Float → Nat) :
    NN.MLTheory.SelfSupervised.PredictiveViewContract dataDim Unit Float Float Float :=
  NN.MLTheory.SelfSupervised.maeAsPredictiveViewContract
    (vectorMaeSelectedIdxs dataDim period offset)
    (matrixRowAsPatchBatch batch dataDim x bi)
    (matrixRowAsPrediction batch dataDim yhat bi)
    patchLoss

/--
The extracted tensor-row predictive-view objective is exactly the finite MAE loss.

This is the formal version of the implementation diagram:

`Spec.Tensor` batch row → hidden-coordinate mask → model prediction row → masked reconstruction
objective.
-/
theorem vectorMaeRow_predictive_objective_eq_maeLoss
    (batch dataDim period offset : Nat)
    (x yhat : Spec.Tensor Float (NN.Tensor.Shape.Mat batch dataDim))
    (bi : Fin batch) (patchLoss : Float → Float → Nat) :
    NN.MLTheory.SelfSupervised.predictiveViewObjective
        (vectorMaeRowPredictiveContract batch dataDim period offset x yhat bi patchLoss) =
      NN.MLTheory.SelfSupervised.maeLoss
        (vectorMaeSelectedIdxs dataDim period offset)
        (matrixRowAsPatchBatch batch dataDim x bi)
        (matrixRowAsPrediction batch dataDim yhat bi)
        patchLoss := by
  exact NN.MLTheory.SelfSupervised.mae_is_predictive_view_objective
    (vectorMaeSelectedIdxs dataDim period offset)
    (matrixRowAsPatchBatch batch dataDim x bi)
    (matrixRowAsPrediction batch dataDim yhat bi)
    patchLoss

/--
Build a compact MAE sample from any batched tensor source.

The source can be an image tensor, spectrogram tensor, token-feature tensor, etc.  This helper
chooses a flattened prefix of each row, masks that prefix, and reconstructs the original prefix.
A full ViT/patch MAE can replace this prefix projection with a patch embedding while keeping the
same training idea.
-/
def tensorPrefixMaeSample {source : Shape} (batch dataDim : Nat)
    (hData : dataDim ≤ Shape.size source) (period offset : Nat)
    (x : Spec.Tensor Float (.dim batch source)) :
    sample.Supervised Float (NN.Tensor.Shape.Mat batch dataDim) (NN.Tensor.Shape.Mat batch dataDim) :=
  vectorMaeSample batch dataDim period offset (_root_.NN.API.tensor.flattenBatchPrefix batch dataDim hData x)

/-! ## Image patch masking -/

/--
Boolean predicate for the deterministic image-patch mask.

The predicate is deliberately phrased at the *pixel* coordinate level rather than at an abstract
patch-id level. This gives the BugZoo examples a coordinate-level contract: for any concrete `NCHW`
tensor coordinate, Lean can say whether this exact scalar is hidden from the model input.

`true` means the pixel belongs to a hidden patch. Degenerate mask parameters (`period = 0`,
`patchH = 0`, or `patchW = 0`) hide nothing, matching `imagePatchMask`.
-/
def imagePatchHidden (height width patchH patchW period offset : Nat)
    (row : Fin height) (col : Fin width) : Bool :=
  if period = 0 ∨ patchH = 0 ∨ patchW = 0 then
    false
  else
    let patchesPerRow := (width + patchW - 1) / patchW
    let patchIndex := (row.val / patchH) * patchesPerRow + (col.val / patchW)
    decide (patchIndex % period = offset % period)

/--
Scalar access helper for `NCHW` image tensors.

This compact definition keeps the certification statements readable. Instead of exposing four nested
`Spec.get` calls everywhere, the image-pipeline theorems can talk about the scalar at batch/channel/
row/column coordinates directly.
-/
def imagePixel (batch c height width : Nat)
    (x : Spec.Tensor Float (NN.Tensor.Shape.Images batch c height width))
    (bi : Fin batch) (ch : Fin c) (row : Fin height) (col : Fin width) : Float :=
  Spec.Tensor.toScalar (Spec.get (Spec.get (Spec.get (Spec.get x bi) ch) row) col)

/--
Patch-level deterministic mask for batched `NCHW` image tensors.

The image remains an image tensor.  We divide pixel coordinates by `patchH` and `patchW` to obtain a
patch-grid index, then hide one patch index class modulo `period`.  If `period = 0` or either patch
dimension is zero, the mask is the identity.
-/
def imagePatchMask (batch c h w patchH patchW period offset : Nat)
    (x : Spec.Tensor Float (NN.Tensor.Shape.Images batch c h w)) :
    Spec.Tensor Float (NN.Tensor.Shape.Images batch c h w) :=
  Spec.Tensor.dim (fun bi =>
    let img := Spec.getAtSpec x bi
    Spec.Tensor.dim (fun ch =>
      let plane := Spec.getAtSpec img ch
      Spec.Tensor.dim (fun row =>
        let line := Spec.getAtSpec plane row
        Spec.Tensor.dim (fun col =>
          let v := Spec.Tensor.toScalar (Spec.get line col)
          let keep :=
            !imagePatchHidden h w patchH patchW period offset row col
          Spec.Tensor.scalar (if keep then v else 0.0)))))

/--
Coordinate-level behavior of the executable image MAE mask.

This is the main implementation certificate for image masking: every scalar pixel in the output
masked image is either the original scalar (visible patch) or exactly zero (hidden patch), according
to the explicit finite predicate `imagePatchHidden`.
-/
theorem imagePatchMask_pixel_eq_if_hidden
    (batch c h w patchH patchW period offset : Nat)
    (x : Spec.Tensor Float (NN.Tensor.Shape.Images batch c h w))
    (bi : Fin batch) (ch : Fin c) (row : Fin h) (col : Fin w) :
    imagePixel batch c h w (imagePatchMask batch c h w patchH patchW period offset x)
        bi ch row col =
      if imagePatchHidden h w patchH patchW period offset row col then
        0.0
      else
        imagePixel batch c h w x bi ch row col := by
  cases x with
  | dim imgs =>
      by_cases hidden : imagePatchHidden h w patchH patchW period offset row col
      · simp [imagePixel, imagePatchMask, Spec.get, Spec.getAtSpec, Spec.Tensor.toScalar, hidden]
      · simp [imagePixel, imagePatchMask, Spec.get, Spec.getAtSpec, Spec.Tensor.toScalar, hidden]

/--
Hidden image patches are unavailable to the model input.

This is the no-target-leakage half of the MAE contract: if a pixel belongs to a hidden patch, then
the tensor handed to the encoder contains zero at that coordinate.
-/
theorem imagePatchMask_hidden_pixel_eq_zero
    (batch c h w patchH patchW period offset : Nat)
    (x : Spec.Tensor Float (NN.Tensor.Shape.Images batch c h w))
    (bi : Fin batch) (ch : Fin c) (row : Fin h) (col : Fin w)
    (hHidden : imagePatchHidden h w patchH patchW period offset row col = true) :
    imagePixel batch c h w (imagePatchMask batch c h w patchH patchW period offset x)
      bi ch row col = 0.0 := by
  simpa [hHidden] using
    imagePatchMask_pixel_eq_if_hidden batch c h w patchH patchW period offset x bi ch row col

/--
Visible image patches are copied through unchanged.

Together with `imagePatchMask_hidden_pixel_eq_zero`, this says the mask does not perturb visible
context pixels. That matters for implementation debugging: a bad patch-index formula would show up
as a failure of this exact coordinate theorem.
-/
theorem imagePatchMask_visible_pixel_eq_input
    (batch c h w patchH patchW period offset : Nat)
    (x : Spec.Tensor Float (NN.Tensor.Shape.Images batch c h w))
    (bi : Fin batch) (ch : Fin c) (row : Fin h) (col : Fin w)
    (hVisible : imagePatchHidden h w patchH patchW period offset row col = false) :
    imagePixel batch c h w (imagePatchMask batch c h w patchH patchW period offset x)
        bi ch row col =
      imagePixel batch c h w x bi ch row col := by
  simpa [hVisible] using
    imagePatchMask_pixel_eq_if_hidden batch c h w patchH patchW period offset x bi ch row col

/--
Build a MAE-style image reconstruction sample from a batched image tensor.

Input: the same image tensor with deterministic patches zeroed out.
Target: a flattened prefix of the original image tensor.

The target is flattened because the current trainable decoder heads in `NN.API.Models` produce
batched matrices.  The source tensor itself remains a real image tensor, so the encoder can be a
CNN/ViT/image model rather than an MLP over a pre-flattened input.
-/
def imagePatchMaeSample (batch c h w reconDim patchH patchW period offset : Nat)
    (hRecon : reconDim ≤ Shape.size (NN.Tensor.Shape.Image c h w))
    (x : Spec.Tensor Float (NN.Tensor.Shape.Images batch c h w)) :
    sample.Supervised Float
      (NN.Tensor.Shape.Images batch c h w)
      (NN.Tensor.Shape.Mat batch reconDim) :=
  sample.mk
    (imagePatchMask batch c h w patchH patchW period offset x)
    (_root_.NN.API.tensor.flattenBatchPrefix batch reconDim hRecon x)

/--
The actual tensor passed to the model by image MAE training is exactly the masked image tensor.

This is a whole-sample/runtime statement: `TrainFixed.curveFloat` forwards its module on
`sample.x`, and for `imagePatchMaeSample` that input is definitionally `imagePatchMask ... x`.
-/
theorem imagePatchMaeSample_input_eq_imagePatchMask
    (batch c h w reconDim patchH patchW period offset : Nat)
    (hRecon : reconDim ≤ Shape.size (NN.Tensor.Shape.Image c h w))
    (x : Spec.Tensor Float (NN.Tensor.Shape.Images batch c h w)) :
    sample.x (imagePatchMaeSample batch c h w reconDim patchH patchW period offset hRecon x) =
      imagePatchMask batch c h w patchH patchW period offset x := by
  simp [imagePatchMaeSample]

/-! ## Certified image MAE pipeline -/

/--
The image MAE sample input hides every pixel selected by `imagePatchHidden`.

This theorem is intentionally stated against `sample.x`, not just `imagePatchMask`. It certifies the
actual value handed to any downstream model constructor/training loop using `imagePatchMaeSample`.
For the paper-level claim, this is the "no hidden target leakage into the encoder input" invariant.
-/
theorem imagePatchMaeSample_input_hidden_pixel_eq_zero
    (batch c h w reconDim patchH patchW period offset : Nat)
    (hRecon : reconDim ≤ Shape.size (NN.Tensor.Shape.Image c h w))
    (x : Spec.Tensor Float (NN.Tensor.Shape.Images batch c h w))
    (bi : Fin batch) (ch : Fin c) (row : Fin h) (col : Fin w)
    (hHidden : imagePatchHidden h w patchH patchW period offset row col = true) :
    imagePixel batch c h w
        (sample.x (imagePatchMaeSample batch c h w reconDim patchH patchW period offset hRecon x))
        bi ch row col = 0.0 := by
  simpa [imagePatchMaeSample] using
    imagePatchMask_hidden_pixel_eq_zero batch c h w patchH patchW period offset x
      bi ch row col hHidden

/--
Visible pixels in the image MAE sample input are exactly the original image pixels.

This is the companion invariant to the hidden-pixel theorem: the SSL view transformation only
removes selected patches. It does not accidentally corrupt the visible context.
-/
theorem imagePatchMaeSample_input_visible_pixel_eq_source
    (batch c h w reconDim patchH patchW period offset : Nat)
    (hRecon : reconDim ≤ Shape.size (NN.Tensor.Shape.Image c h w))
    (x : Spec.Tensor Float (NN.Tensor.Shape.Images batch c h w))
    (bi : Fin batch) (ch : Fin c) (row : Fin h) (col : Fin w)
    (hVisible : imagePatchHidden h w patchH patchW period offset row col = false) :
    imagePixel batch c h w
        (sample.x (imagePatchMaeSample batch c h w reconDim patchH patchW period offset hRecon x))
        bi ch row col =
      imagePixel batch c h w x bi ch row col := by
  simpa [imagePatchMaeSample] using
    imagePatchMask_visible_pixel_eq_input batch c h w patchH patchW period offset x
      bi ch row col hVisible

/--
The target tensor of `imagePatchMaeSample` is the original image flattened to the decoder target
prefix.

This pins down the target side of the runnable MAE example: the target is not a label loaded from
elsewhere and not the masked image. It is the original image tensor, flattened and truncated to
`reconDim`.
-/
theorem imagePatchMaeSample_target_eq_flattenBatchPrefix
    (batch c h w reconDim patchH patchW period offset : Nat)
    (hRecon : reconDim ≤ Shape.size (NN.Tensor.Shape.Image c h w))
    (x : Spec.Tensor Float (NN.Tensor.Shape.Images batch c h w)) :
    sample.y (imagePatchMaeSample batch c h w reconDim patchH patchW period offset hRecon x) =
      _root_.NN.API.tensor.flattenBatchPrefix batch reconDim hRecon x := by
  simp [imagePatchMaeSample]

/--
For the current executable image MAE head, the finite objective indexes every flattened decoder
coordinate.

The masking theorem above certifies which *input* pixels are hidden. This decoder target is a flat
reconstruction prefix, so the objective side is a finite list of reconstruction coordinates. Patch
token decoders can swap this list for masked patch-token indices while reusing the same
predictive-view bridge.
-/
def imageMaeReconstructionIdxs (reconDim : Nat) : List (Fin reconDim) :=
  List.finRange reconDim

/--
The target row of the image MAE sample is the finite target batch used by the predictive-view
objective.

This theorem is intentionally narrow: the theorem-level target is definitionally the same tensor row
that the runtime supervised-training path passes to MSE.
-/
theorem imagePatchMaeSample_target_row_eq_flattened_source_row
    (batch c h w reconDim patchH patchW period offset : Nat)
    (hRecon : reconDim ≤ Shape.size (NN.Tensor.Shape.Image c h w))
    (x : Spec.Tensor Float (NN.Tensor.Shape.Images batch c h w))
    (bi : Fin batch) :
    matrixRowAsPatchBatch batch reconDim
        (sample.y (imagePatchMaeSample batch c h w reconDim patchH patchW period offset hRecon x))
        bi =
      matrixRowAsPatchBatch batch reconDim
        (_root_.NN.API.tensor.flattenBatchPrefix batch reconDim hRecon x)
        bi := by
  funext j
  simp [matrixRowAsPatchBatch, imagePatchMaeSample]

/--
One row of the runnable image MAE pipeline as a finite predictive-view contract.

`yhat` is the model output matrix, for example the output of `vitMaskedAutoencoder`. The contract's
target comes from `sample.y (imagePatchMaeSample ...)`, so this definition connects the model's
runtime tensor output directly to the finite SSL objective algebra.
-/
def imagePatchMaeRowPredictiveContract
    (batch c h w reconDim patchH patchW period offset : Nat)
    (hRecon : reconDim ≤ Shape.size (NN.Tensor.Shape.Image c h w))
    (x : Spec.Tensor Float (NN.Tensor.Shape.Images batch c h w))
    (yhat : Spec.Tensor Float (NN.Tensor.Shape.Mat batch reconDim))
    (bi : Fin batch) (patchLoss : Float → Float → Nat) :
    NN.MLTheory.SelfSupervised.PredictiveViewContract reconDim Unit Float Float Float :=
  NN.MLTheory.SelfSupervised.maeAsPredictiveViewContract
    (imageMaeReconstructionIdxs reconDim)
    (matrixRowAsPatchBatch batch reconDim
      (sample.y (imagePatchMaeSample batch c h w reconDim patchH patchW period offset hRecon x))
      bi)
    (matrixRowAsPrediction batch reconDim yhat bi)
    patchLoss

/--
The image MAE tensor-row contract is exactly a finite MAE reconstruction loss.

This is the implementation-to-objective theorem for the runnable image path:

1. build an SSL sample from an image tensor;
2. feed `sample.x` to an image model;
3. view the output row as finite predictions; and
4. compute the predictive-view objective, which is exactly `maeLoss` over the flattened target row.
-/
theorem imagePatchMaeRow_predictive_objective_eq_maeLoss
    (batch c h w reconDim patchH patchW period offset : Nat)
    (hRecon : reconDim ≤ Shape.size (NN.Tensor.Shape.Image c h w))
    (x : Spec.Tensor Float (NN.Tensor.Shape.Images batch c h w))
    (yhat : Spec.Tensor Float (NN.Tensor.Shape.Mat batch reconDim))
    (bi : Fin batch) (patchLoss : Float → Float → Nat) :
    NN.MLTheory.SelfSupervised.predictiveViewObjective
        (imagePatchMaeRowPredictiveContract batch c h w reconDim patchH patchW period offset
          hRecon x yhat bi patchLoss) =
      NN.MLTheory.SelfSupervised.maeLoss
        (imageMaeReconstructionIdxs reconDim)
        (matrixRowAsPatchBatch batch reconDim
          (sample.y (imagePatchMaeSample batch c h w reconDim patchH patchW period offset
            hRecon x))
          bi)
        (matrixRowAsPrediction batch reconDim yhat bi)
        patchLoss := by
  exact NN.MLTheory.SelfSupervised.mae_is_predictive_view_objective
    (imageMaeReconstructionIdxs reconDim)
    (matrixRowAsPatchBatch batch reconDim
      (sample.y (imagePatchMaeSample batch c h w reconDim patchH patchW period offset hRecon x))
      bi)
    (matrixRowAsPrediction batch reconDim yhat bi)
    patchLoss

end ssl

end API
end NN
