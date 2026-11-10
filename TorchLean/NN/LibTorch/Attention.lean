/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.LibTorch.Activation
public import TorchLean.CUDA.LibTorch.Convert
public import TorchLean.Tensor.ConfiguredBinary
public import TorchLean.Tensor.Layers.Attention

/-!
# CUDA Tape Operations: Attention
-/

@[expose] public section

namespace TorchLean
namespace LibTorch

open TorchLean.Tensor

namespace Buffer

private def attentionDimensions (batch n d : UInt32) (scale : Float) (dtype : Dtype) :
    Except String (UInt32 × UInt32 × UInt32) := do
  let effectiveScale := if dtype == .float32 then scale.toFloat32.toFloat else scale
  if !effectiveScale.isFinite then
    throw "attention: scale must remain finite in the buffer dtype"
  let some rows := Bytes.toUInt32? (batch.toNat * n.toNat) | throw SomeBuffer.tooLarge
  let some values := Bytes.toUInt32? (rows.toNat * d.toNat) | throw SomeBuffer.tooLarge
  let some scores := Bytes.toUInt32? (rows.toNat * n.toNat) | throw SomeBuffer.tooLarge
  pure (rows, values, scores)

private def requireAttentionSize (label : String) (buffer : Buffer) (expected : UInt32)
    (dtype : Dtype) :
    Except String Unit := do
  if buffer.size != expected then
    throw s!"attention: {label} buffer size mismatch (expected {expected}, got {buffer.size})"
  unless buffer.dtype == dtype do
    throw s!"attention: {label} buffer dtype mismatch"

private def matrixFromArray {α : Type} [Storage α] (encode : Float → α) (values : FloatArray)
    (offset rows columns : Nat) (transpose : Bool := false) : Tensor α [rows, columns] :=
  Tensor.dim fun row => Tensor.dim fun column =>
    Tensor.scalar <| encode <| values.get! <| offset +
      (if transpose then column.val * rows + row.val else row.val * columns + column.val)

private def replayBmm {α : Type} [Storage α] [Context α]
    (encode : Float → α) (decode : α → Float) (left right ordinary : FloatArray)
    (batch m n p : Nat) (transposeA transposeB : Bool) (scale : Float) : FloatArray := Id.run do
  let ops := (inferInstance : Numeric.QuotientArithmetic α)
  let scaleCoefficient := (inferInstance : Numeric.QuotientArithmetic Float).encode scale
  let mut output := FloatArray.emptyWithCapacity (batch * m * p)
  for group in [:batch] do
    let a := matrixFromArray encode left (group * m * n) m n transposeA
    let b := matrixFromArray encode right (group * n * p) n p transposeB
    for row in List.finRange m do
      for column in List.finRange p do
        let previous := ordinary.get! (group * m * p + row.val * p + column.val)
        let candidate := do
          let coefficient ← scaleCoefficient
          let products ← Attention.productCoefficients?
            (fun _ : Fin 1 => a) (fun _ => b) (1 : α) row column
          -- The buffer ABI supplies a Float64 scale even when its operands are Float32.
          let result := Numeric.QuotientCoefficients.map (· * coefficient.primal) products
          let rounded ← ops.decode result
          pure <| if result.primal == 0 && encode previous == 0 then
            ops.copyPrimal (encode previous) rounded else rounded
        output := output.push (candidate.map decode |>.getD previous)
  return output

/-- Scaled batched product, replayed from the original matrices when native scaling loses range.

One device flag gates the host transfer. A flagged call downloads its original operands and reuses
the reference contraction's finite-coefficient replay, in the buffer's actual precision. The guard
is conservative: zero contractions, including exact orthogonality, can also take this CPU path.
Nonfinite inputs retain the native result. -/
@[no_expose] def scaledProduct (left right : Buffer) (batch m n p : UInt32)
    (scale : Float) (transposeA transposeB : Bool := false) : Buffer :=
  let (ordinary, recover) := scaledBmm left right batch m n p
    (if transposeA then 1 else 0) (if transposeB then 1 else 0) scale
  if !recover then ordinary
  else
    let a := toFloatArray left
    let b := toFloatArray right
    let previous := toFloatArray ordinary
    let values := match Buffer.dtype left with
      | .float32 => replayBmm Float.toFloat32 Float32.toFloat a b previous
          batch.toNat m.toNat n.toNat p.toNat transposeA transposeB scale
      | .float64 => replayBmm id id a b previous
          batch.toNat m.toNat n.toNat p.toNat transposeA transposeB scale
      | .encoded => previous
    releaseThen ordinary <| ofFloatArray values (Buffer.dtype left)

private def replaySoftmaxProduct {α : Type} [Storage α] [Context α]
    (encode : Float → α) (decode : α → Float)
    (weights gradient value ordinary : FloatArray) (batch rows keys width : Nat)
    (multiplier : Option FloatArray := none) : FloatArray :=
  Id.run do
    let mut output := FloatArray.emptyWithCapacity (batch * rows * keys)
    for group in [:batch] do
      let p := matrixFromArray encode weights (group * rows * keys) rows keys
      let g := matrixFromArray encode gradient (group * rows * width) rows width
      let v := matrixFromArray encode value (group * keys * width) width keys true
      let previous := matrixFromArray encode ordinary (group * rows * keys) rows keys
      let factor := multiplier.map fun values =>
        matrixFromArray encode values (group * rows * keys) rows keys
      for row in List.finRange rows do
        let coefficients := match factor with
          | none => Attention.productCoefficients? (fun _ : Fin 1 => g) (fun _ => v) 1 row
          | some factor => Attention.weightedProductCoefficients? g v factor row
        let recovered := (Attention.replayPullbackCoefficientsRow? (p.unstack row)
          (previous.unstack row) coefficients)
          |>.getD (previous.unstack row)
        for column in List.finRange keys do
          output := output.push (decode (recovered.getScalar column))
    return output

private def replayWeightedProducts {α : Type} [Storage α] [Context α]
    (encode : Float → α) (decode : α → Float)
    (weights gradient value ordinary : FloatArray) (batch rows keys width : Nat) : FloatArray :=
  Id.run do
    let ops := (inferInstance : Numeric.QuotientArithmetic α)
    let mut output := FloatArray.emptyWithCapacity (batch * rows * keys)
    for group in [:batch] do
      let p := matrixFromArray encode weights (group * rows * keys) rows keys
      let g := matrixFromArray encode gradient (group * rows * width) rows width
      let v := matrixFromArray encode value (group * keys * width) width keys true
      for row in List.finRange rows do
        for column in List.finRange keys do
          let previous := ordinary.get! (group * rows * keys + row.val * keys + column.val)
          let recovered := do
            let coefficient ← Attention.weightedProductCoefficients? g v p row column
            let candidate ← ops.decode coefficient
            pure <| if coefficient.primal == 0 && encode previous == 0 then
              ops.copyPrimal (encode previous) candidate else candidate
          output := output.push (recovered.map decode |>.getD previous)
    return output

private def replayValueProducts {α : Type} [Storage α] [Context α]
    (encode : Float → α) (decode : α → Float)
    (weights multiplier gradient ordinary : FloatArray) (batch rows keys width : Nat) :
    FloatArray := Id.run do
  let ops := (inferInstance : Numeric.QuotientArithmetic α)
  let mut output := FloatArray.emptyWithCapacity (batch * keys * width)
  for group in [:batch] do
    let p := matrixFromArray encode weights (group * rows * keys) rows keys
    let m := matrixFromArray encode multiplier (group * rows * keys) rows keys
    let g := matrixFromArray encode gradient (group * rows * width) rows width
    for key in List.finRange keys do
      for column in List.finRange width do
        let previous := ordinary.get! (group * keys * width + key.val * width + column.val)
        let recovered := do
          let coefficient ← Attention.weightedValueCoefficients? p m g key column
          let candidate ← ops.decode coefficient
          pure <| if coefficient.primal == 0 && encode previous == 0 then
            ops.copyPrimal (encode previous) candidate else candidate
        output := output.push (recovered.map decode |>.getD previous)
  return output

/-- Whether `buffer` has an infinite or NaN entry: an entry times zero is a NaN exactly at such an
entry (`∞ * 0` and `NaN * 0` are NaNs, a finite entry gives a zero), so their sum is a NaN exactly
then. -/
private def hasNonfinite (buffer : Buffer) : Bool :=
  if buffer.size == 0 then false else
    let zeros := Buffer.scale buffer 0.0
    let total := releaseThen zeros <| reduceSumByRow zeros 1 buffer.size
    let observed := toFloatArray total
    releaseThen total <| (observed.get! 0).isNaN

/-- Apply the centered softmax VJP without discarding the original matrix cotangent.

The centered pullback multiplies every entry of `gradient @ valueᵀ` by its saved probability, as
`Attention.pullbackProducts` does, so a zero probability against a nonfinite entry gives NaN.
Contraction range loss or a nonfinite centered result triggers recovery, and any other call
returns the ordinary GPU result. Recovery reconstructs a flagged finite contraction through the
pullback from the saved probabilities, without first rounding `gradient @ valueᵀ` to infinity.
Nonfinite original operands or saved probabilities retain the ordinary native buffer. -/
@[no_expose] def softmaxProductBackward (weights gradient value : Buffer)
    (batch rows keys width : UInt32) : Buffer :=
  let (dp, recover) := scaledBmm gradient value batch rows width keys 0 1 1
  let ordinary := releaseThen dp <| Tape.rowSoftmaxBackward weights dp (batch * rows) keys
  let nonfinite := hasNonfinite ordinary
  if !recover && !nonfinite then ordinary
  else
    let probabilities := toFloatArray weights
    if !probabilities.toList.all Float.isFinite then ordinary
    else
      let g := toFloatArray gradient
      let v := toFloatArray value
      if !g.toList.all Float.isFinite || !v.toList.all Float.isFinite then ordinary
      else
        let previous := toFloatArray ordinary
        let recovered := match Buffer.dtype weights with
          | .float32 => replaySoftmaxProduct Float.toFloat32 Float32.toFloat
              probabilities g v previous batch.toNat rows.toNat keys.toNat width.toNat
          | .float64 => replaySoftmaxProduct id id
              probabilities g v previous batch.toNat rows.toNat keys.toNat width.toNat
          | .encoded => previous
        releaseThen ordinary <| ofFloatArray recovered (Buffer.dtype weights)

/-- The normalized product's local pullback, including its differentiable multiplier.
Recovery uses the actual saved probabilities and retains the original contraction through the
multiplier. The scalar-product guard also checks the pointwise factors, requiring a device-flag
synchronization in addition to the output-finiteness checks. Fully blocked rows suppress their
upstream values, as in `Attention.valuesBackwardSlice`. Elsewhere any nonfinite original keeps the
three native results with their NaN payloads. -/
@[no_expose] def attentionValuesBackward (weights value multiplier gradient : Buffer)
    (batch rows keys width : UInt32) : Buffer × Buffer × Buffer :=
  let active := reduceSumByRow weights (batch * rows) keys
  let activeColumns := broadcastVecToCols active (batch * rows) width
  let gradient := releaseThen activeColumns <| Buffer.mask gradient activeColumns
  let (dp, rangeLoss) := scaledBmm gradient value batch rows width keys 0 1 1
  let activeScores := releaseThen active <| broadcastVecToCols active (batch * rows) keys
  let dp := releaseThen dp <| releaseThen activeScores <| Buffer.mask dp activeScores
  let (productProbe, multiplierLoss) := scaledBmm weights multiplier weights.size 1 1 1 0 0 1
  let mixed := releaseThen productProbe <| Buffer.mul weights multiplier
  let dv := releaseThen mixed <|
    scaledProduct mixed gradient batch keys rows width 1 (transposeA := true)
  let allowed := Buffer.mask dp weights
  let dm := Buffer.mul dp weights
  let weighted := Buffer.mul allowed multiplier
  let ds := releaseThen weighted <| Tape.rowSoftmaxBackward weights weighted (batch * rows) keys
  let recover := rangeLoss || multiplierLoss || hasNonfinite ds || hasNonfinite dm ||
    hasNonfinite dv
  releaseThen gradient <| releaseThen dp <| releaseThen allowed <|
    if !recover then (ds, dv, dm) else
      let p := toFloatArray weights
      let g := toFloatArray gradient
      let v := toFloatArray value
      let factor := toFloatArray multiplier
      if !#[p, g, v, factor].all (fun values => values.toList.all Float.isFinite) then (ds, dv, dm)
      else
        let dsValues := toFloatArray ds
        let dmValues := toFloatArray dm
        let dvValues := toFloatArray dv
        let result := match Buffer.dtype weights with
          | .float32 =>
              (replaySoftmaxProduct Float.toFloat32 Float32.toFloat p g v dsValues
                batch.toNat rows.toNat keys.toNat width.toNat (some factor),
               replayWeightedProducts Float.toFloat32 Float32.toFloat p g v dmValues
                batch.toNat rows.toNat keys.toNat width.toNat,
               replayValueProducts Float.toFloat32 Float32.toFloat p factor g dvValues
                batch.toNat rows.toNat keys.toNat width.toNat)
          | .float64 =>
              (replaySoftmaxProduct id id p g v dsValues
                batch.toNat rows.toNat keys.toNat width.toNat (some factor),
               replayWeightedProducts id id p g v dmValues
                batch.toNat rows.toNat keys.toNat width.toNat,
               replayValueProducts id id p factor g dvValues
                batch.toNat rows.toNat keys.toNat width.toNat)
          | .encoded => (dsValues, dmValues, dvValues)
        (releaseThen ds <| ofFloatArray result.1 (Buffer.dtype weights),
          releaseThen dv <| ofFloatArray result.2.2 (Buffer.dtype weights),
          releaseThen dm <| ofFloatArray result.2.1 (Buffer.dtype weights))

/--
Compose scaled dot-product attention from numerical buffer operations.

Inputs have shape `(batch, n, d)`, and an optional `(batch, n, n)` mask marks allowed entries.
Returns the output and saved probabilities. The caller owns both buffers and keeps Q/K/V and
probabilities alive until backward. Blocked probabilities and the probabilities of a fully blocked
row are zero. The output multiplies them by `V` as `Attention.scaledDotProduct` does, so a zero
probability against a nonfinite value gives NaN.
-/
@[no_expose] def attentionForward (Q K V : Buffer) (mask : Option Buffer)
    (batch n d : UInt32) (scale : Float) : Except String (Buffer × Buffer) := do
  let (rows, values, scores) ← attentionDimensions batch n d scale (Buffer.dtype Q)
  for (label, buffer) in #[("Q", Q), ("K", K), ("V", V)] do
    requireAttentionSize label buffer values (Buffer.dtype Q)
  if let some allowed := mask then
    requireAttentionSize "mask" allowed scores (Buffer.dtype Q)
  -- Only an empty batch returns early: at `d = 0` every score is zero, and an unmasked row of
  -- probabilities is `1 / n`.
  if rows == 0 then
    return (zeros values (Buffer.dtype Q), zeros scores (Buffer.dtype Q))
  let logits := scaledProduct Q K batch n d n scale (transposeB := true)
  let probabilities := match mask with
    | some allowed =>
        releaseThen logits <| hardMaskedSoftmaxByRow logits allowed rows n
    | none =>
        let result := Tape.rowSoftmaxForward logits rows n
        releaseThen logits <| result.releaseWorkspaceThen result.value
  let output := bmm probabilities V batch n n d
  pure (output, probabilities)

/--
Attention's local VJP, composed in Lean from the saved probabilities and numerical primitives.

`dP = dOut Vᵀ`. For each row, choose a maximal-probability coordinate and subtract its
`dP` value: `centered = dP - anchor`, `dS = P * (centered - sum(P * centered))`.
For normalized exact-real probabilities this is the usual softmax VJP. Rounded probabilities
need not sum exactly to one, so the centered and uncentered computations can differ.
`dQ = scale * dS K`, `dK = scale * dSᵀ Q`, and `dV = Pᵀ dOut`.
All arguments are borrowed. As in `Attention.scaledDotProductBackward`, the products and the softmax
pullback multiply zero probabilities and score cotangents by entries of `dOut`, `V`, `K` and `Q`,
so a fully blocked row gives NaN gradients at nonfinite entries.

The contraction check can replay finite `dOut` and `V` through the complete centered pullback
using the actual saved probabilities, which can recover a finite score cotangent even when `dP`
cannot be represented on its own. Native operations, transfers, and codecs are trust boundaries.
-/
@[no_expose] def attentionBackward (Q K V probabilities dOut : Buffer)
    (batch n d : UInt32) (scale : Float) : Except String (Buffer × Buffer × Buffer) := do
  let (_, values, scores) ← attentionDimensions batch n d scale (Buffer.dtype Q)
  for (label, buffer) in #[("Q", Q), ("K", K), ("V", V), ("dOut", dOut)] do
    requireAttentionSize label buffer values (Buffer.dtype Q)
  requireAttentionSize "probabilities" probabilities scores (Buffer.dtype Q)
  if values == 0 then
    return (zeros values (Buffer.dtype Q), zeros values (Buffer.dtype Q),
      zeros values (Buffer.dtype Q))
  let ds := softmaxProductBackward probabilities dOut V batch n n d
  let dq := scaledProduct ds K batch n n d scale
  let dk := releaseThen ds <| scaledProduct ds Q batch n n d scale (transposeA := true)
  let dv := bmm probabilities dOut batch n n d (transposeA := 1)
  pure (dq, dk, dv)

end Buffer

end LibTorch
end TorchLean
