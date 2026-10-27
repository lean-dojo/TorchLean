/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.LibTorch.Ops.Core
public import NN.Runtime.Autograd.Engine.LibTorch.Convert
public import NN.Spec.Core.FloatInstances
public import NN.Spec.Layers.Attention

/-!
# CUDA Tape Operations: Attention
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace LibTorch

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Buffer

private def attentionDimensions (batch n d : UInt32) (scale : Float) (dtype : Dtype) :
    Except String (UInt32 × UInt32 × UInt32) := do
  let effectiveScale := if dtype == .float32 then scale.toFloat32.toFloat else scale
  if !effectiveScale.isFinite then
    throw "attention: scale must remain finite in the buffer dtype"
  let rows ← SomeBuffer.natToU32Checked (batch.toNat * n.toNat)
  let values ← SomeBuffer.natToU32Checked (rows.toNat * d.toNat)
  let scores ← SomeBuffer.natToU32Checked (rows.toNat * n.toNat)
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
          let products ← Spec.Internal.productCoefficients?
            (fun _ : Fin 1 => a) (fun _ => b) (1 : α) row column
          -- The buffer ABI supplies a Float64 scale even when its operands are Float32.
          let result := Numeric.QuotientCoefficients.map (· * coefficient.primal) products
          let rounded ← ops.decode result
          pure <| if result.primal == 0 && encode previous == 0 then
            ops.copyPrimal (encode previous) rounded else rounded
        output := output.push (candidate.map decode |>.getD previous)
  return output

/-- Keep the original matrices available when scaling a native contraction loses range.

One device flag gates the host transfer. A flagged call downloads its original operands and reuses
the reference contraction's finite-coefficient replay, in the buffer's actual precision. The guard
is conservative: zero contractions, including exact orthogonality, can also take this CPU path.
Nonfinite inputs retain the native result. This is range recovery, not a proof of ATen. -/
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
          | none => Spec.Internal.productCoefficients? (fun _ : Fin 1 => g) (fun _ => v) 1 row
          | some factor => Spec.Internal.weightedProductCoefficients? g v factor row
        let recovered := (Spec.Internal.replayPullbackCoefficientsRow? (p.unstack row)
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
            let coefficient ← Spec.Internal.weightedProductCoefficients? g v p row column
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
          let coefficient ← Spec.Internal.weightedValueCoefficients? p m g key column
          let candidate ← ops.decode coefficient
          pure <| if coefficient.primal == 0 && encode previous == 0 then
            ops.copyPrimal (encode previous) candidate else candidate
        output := output.push (recovered.map decode |>.getD previous)
  return output

private def hasNonfinite (buffer : Buffer) : Bool :=
  if buffer.size == 0 then false else
    let magnitude := Buffer.abs buffer
    let largest := releaseThen magnitude <| reduceMaxByRow magnitude 1 buffer.size
    let observed := toFloatArray largest
    releaseThen largest <| !((observed.get! 0).isFinite)

/-- Apply the centered softmax VJP without discarding the original matrix cotangent.

Contraction range loss or a nonfinite centered result triggers recovery; otherwise the ordinary
GPU operations are retained. A flagged finite contraction is reconstructed through the pullback
using the saved probabilities, rather than first rounding `gradient @ valueᵀ` to infinity.
Nonfinite original operands or saved probabilities retain the ordinary native buffer. -/
@[no_expose] def softmaxProductBackward (weights gradient value : Buffer)
    (batch rows keys width : UInt32) : Buffer :=
  let (dp, recover) := scaledBmm gradient value batch rows width keys 0 1 1
  let allowed := releaseThen dp <| Buffer.mask dp weights
  let ordinary := releaseThen allowed <| Tape.rowSoftmaxBwd weights allowed (batch * rows) keys
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
multiplier. The existing scalar-product guard also checks the pointwise factors, requiring a
device-flag synchronization in addition to the output-finiteness checks. Fully blocked rows suppress
their upstream values. Elsewhere nonfinite originals keep native results rather than round-trip
NaN payloads. -/
@[no_expose] def softmaxMatmulBackward (weights value multiplier gradient : Buffer)
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
  let ds := releaseThen weighted <| Tape.rowSoftmaxBwd weights weighted (batch * rows) keys
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

Inputs have shape `(batch, n, d)`; an optional `(batch, n, n)` mask marks allowed entries.
Returns the output and saved probabilities. The caller owns both buffers and keeps Q/K/V and
probabilities alive until backward. Blocked probabilities and fully blocked rows are zero.
-/
@[no_expose] def attentionForward (Q K V : Buffer) (mask : Option Buffer)
    (batch n d : UInt32) (scale : Float) : Except String (Buffer × Buffer) := do
  let (rows, values, scores) ← attentionDimensions batch n d scale (Buffer.dtype Q)
  for (label, buffer) in #[("Q", Q), ("K", K), ("V", V)] do
    requireAttentionSize label buffer values (Buffer.dtype Q)
  if let some allowed := mask then
    requireAttentionSize "mask" allowed scores (Buffer.dtype Q)
  if values == 0 then
    return (zeros values (Buffer.dtype Q), zeros scores (Buffer.dtype Q))
  let logits := scaledProduct Q K batch n d n scale (transposeB := true)
  let probabilities := match mask with
    | some allowed =>
        releaseThen logits <| hardMaskedSoftmaxByRow logits allowed rows n
    | none =>
        let result := Tape.rowSoftmaxForward logits rows n
        releaseThen logits <| result.releaseWorkspaceThen result.value
  let output := bmm probabilities V batch n n d
  let output := match mask with
    | none => output
    | some _ =>
        -- Zero probabilities do not suppress a nonfinite V through multiplication alone.
        let active := reduceSumByRow probabilities rows n
        let activeB := releaseThen active <| broadcastVecToCols active rows d
        releaseThen activeB <| releaseThen output <| Buffer.mask output activeB
  pure (output, probabilities)

/--
Attention's local VJP, composed in Lean from the saved probabilities and numerical primitives.

`dP = dOut Vᵀ`. For each row, choose a maximal-probability coordinate and subtract its
`dP` value: `centered = dP - anchor`, `dS = P * (centered - sum(P * centered))`.
For normalized exact-real probabilities this is the usual softmax VJP; rounded probabilities
need not sum exactly to one, so the centered and uncentered computations can differ.
`dQ = scale * dS K`, `dK = scale * dSᵀ Q`, and `dV = Pᵀ dOut`.
All arguments are borrowed. Fully blocked rows contribute zero even for nonfinite cotangents.

The contraction check can replay finite `dOut` and `V` through the complete centered pullback
using the actual saved probabilities. This can recover a finite score cotangent even when `dP`
cannot be represented on its own. Native operations, transfers, and codecs remain trust boundaries.
-/
@[no_expose] def attentionBackward (Q K V probabilities dOut : Buffer)
    (batch n d : UInt32) (scale : Float) : Except String (Buffer × Buffer × Buffer) := do
  let (rows, values, scores) ← attentionDimensions batch n d scale (Buffer.dtype Q)
  for (label, buffer) in #[("Q", Q), ("K", K), ("V", V), ("dOut", dOut)] do
    requireAttentionSize label buffer values (Buffer.dtype Q)
  requireAttentionSize "probabilities" probabilities scores (Buffer.dtype Q)
  if values == 0 then
    return (zeros values (Buffer.dtype Q), zeros values (Buffer.dtype Q),
      zeros values (Buffer.dtype Q))
  let active := reduceSumByRow probabilities rows n
  let activeB := releaseThen active <| broadcastVecToCols active rows d
  let grad := Buffer.mask dOut activeB
  -- A zero dS does not suppress a nonfinite query through multiplication alone.
  let maskedQ := Buffer.mask Q activeB
  let ds := softmaxProductBackward probabilities grad V batch n n d
  let dq := scaledProduct ds K batch n n d scale
  let dk := releaseThen maskedQ <| releaseThen ds <|
    scaledProduct ds maskedQ batch n n d scale (transposeA := true)
  let dv := releaseThen grad <| bmmLeftTranspose probabilities grad batch n n d
  -- Nonfinite keys can produce 0 * infinity in a fully blocked row's dQ.
  let dq := releaseThen activeB <| releaseThen dq <| Buffer.mask dq activeB
  pure (dq, dk, dv)

end Buffer

namespace Tape

/-- Record a row-normalized matrix product on the existing native tape.
Only saved probabilities are workspace; input buffers remain borrowed tape parents. -/
def softmaxMatmul {batch : Shape} {rows keys width : Nat} (t : Tape SomeBuffer)
    (scoresId valueId multiplierId : Nat)
    (mask : Option (Tensor Bool [rows, keys]) := none) : Result (Tape SomeBuffer × Nat) := do
  let b ← SomeBuffer.natToU32Checked batch.size
  let r ← SomeBuffer.natToU32Checked rows
  let k ← SomeBuffer.natToU32Checked keys
  let w ← SomeBuffer.natToU32Checked width
  let rowCount ← SomeBuffer.natToU32Checked (batch.size * rows)
  let scoreCount ← SomeBuffer.natToU32Checked (batch.size * rows * keys)
  let _ ← SomeBuffer.natToU32Checked (batch.size * keys * width)
  let outputCount ← SomeBuffer.natToU32Checked (batch.size * rows * width)
  let scoreShape := batch.concat [rows, keys]
  let valueShape := batch.concat [keys, width]
  let outputShape := batch.concat [rows, width]
  let scores ← requireValue (t := t) scoresId scoreShape
  let value ← requireValue (t := t) valueId valueShape
  let multiplier ← requireValue (t := t) multiplierId scoreShape
  unless scores.dtype == value.dtype && scores.dtype == multiplier.dtype do
    throw "softmaxMatmul: inputs must have the same buffer dtype"
  let probabilities := if scoreCount == 0 then Buffer.zeros scoreCount scores.dtype else
    match mask with
    | none =>
        let normalized := rowSoftmaxForward scores rowCount k
        normalized.releaseWorkspaceThen normalized.value
    | some mask =>
        let source := Buffer.ofFloatArray (Convert.flattenBoolMask mask) scores.dtype
        let allowed := Buffer.broadcastTo source #[rows, keys] #[batch.size, rows, keys] #[1, 2]
        Buffer.releaseThen source <| Buffer.releaseThen allowed <|
          Buffer.hardMaskedSoftmaxByRow scores allowed rowCount k
  let mixed := Buffer.mul probabilities multiplier
  let output := Buffer.releaseThen mixed <|
    if outputCount == 0 then Buffer.zeros outputCount scores.dtype else
      Buffer.bmm mixed value b r k w
  let node : Node SomeBuffer :=
    { name := some "softmax_matmul"
      value := { s := outputShape, buf := output }
      requiresGrad := #[scoresId, valueId, multiplierId].any fun id =>
        (t.getNode? id).any (·.requiresGrad)
      parents := #[scoresId, valueId, multiplierId]
      cleanup := #[SomeBuffer.ofWorkspace probabilities]
      backward := fun seed => do
        let gradient ← requireGrad seed outputShape
        let (ds, dv, dm) := Buffer.softmaxMatmulBackward probabilities value multiplier
          gradient.buf b r k w
        pure #[(scoresId, { s := scoreShape, buf := ds }),
          (valueId, { s := valueShape, buf := dv }),
          (multiplierId, { s := scoreShape, buf := dm })] }
  pure (t.addNode node)

/-!
## Multi-head self-attention

Forward structure matches `Spec.MultiHeadAttention.forward`:
1. `Q = x @ Wq`, `K = x @ Wk`, `V = x @ Wv`
2. reshape to heads `(numHeads, n, headDim)`
3. attention per head (batched): `softmax(Q Kᵀ / sqrt(headDim)) @ V`
4. combine heads, then output projection `@ Wo`

Masking:
- Blocked entries contribute zero softmax numerator, and fully blocked rows return zero.
- Lean composes forward and backward and retains probabilities on its tape.
- LibTorch executes the numerical primitives without recording an autograd graph.
- The host `Tensor Bool` mask is copied to the device.
-/

namespace Internal

/-- Attention on a fixed input shape, selected by the optional-batch public adapter. -/
def attention
  {n numHeads dModel headDim : Nat} (_hSeq : n ≠ 0)
  (batch : Nat) (_hBatch : batch ≠ 0) (inputShape : Shape)
  (t : Tape SomeBuffer) (wqId wkId wvId woId xId : Nat)
  (mask : Option (Tensor Bool [n, n]) := none) :
  IO (Result (Tape SomeBuffer × Nat)) := (do
  let one32 : UInt32 := 1
  let depth1 : UInt32 := 1
  let n32 ← ExceptT.mk (pure <| SomeBuffer.natToU32Checked n)
  let dModel32 ← ExceptT.mk (pure <| SomeBuffer.natToU32Checked dModel)
  let head32 ← ExceptT.mk (pure <| SomeBuffer.natToU32Checked headDim)
  let projDim : Nat := numHeads * headDim
  let proj32 ← ExceptT.mk (pure <| SomeBuffer.natToU32Checked projDim)
  let rows : Nat := batch * n
  let rows32 ← ExceptT.mk (pure <| SomeBuffer.natToU32Checked rows)
  let batchHeads : Nat := batch * numHeads
  let batchHeads32 ← ExceptT.mk (pure <| SomeBuffer.natToU32Checked batchHeads)
  let wq ← ExceptT.mk (pure <| requireValue (t := t) wqId (.dim dModel (.dim projDim .scalar)))
  let wk ← ExceptT.mk (pure <| requireValue (t := t) wkId (.dim dModel (.dim projDim .scalar)))
  let wv ← ExceptT.mk (pure <| requireValue (t := t) wvId (.dim dModel (.dim projDim .scalar)))
  let wo ← ExceptT.mk (pure <| requireValue (t := t) woId (.dim projDim (.dim dModel .scalar)))
  let x ← ExceptT.mk (pure <| requireValue (t := t) xId inputShape)
  -- Flatten the leading batch into the row axis for the shared projections.
  let Q := Buffer.bmm x wq one32 rows32 dModel32 proj32
  let K := Buffer.bmm x wk one32 rows32 dModel32 proj32
  let V := Buffer.bmm x wv one32 rows32 dModel32 proj32
  -- Split heads:
  --   `(batch,n,projDim)` views as `(batch,n,numHeads,headDim)`, then swaps to
  --   `(batch,numHeads,n,headDim)`. The first two axes are folded into the BMM batch axis.
  let dimsView : Array Nat := #[batch, n, numHeads, headDim]
  let dimsHead : Array Nat := #[batch, numHeads, n, headDim]
  let Qh := Buffer.releaseThen Q <| Buffer.swapAdjacentAtDepth Q dimsView depth1
  let Kh := Buffer.releaseThen K <| Buffer.swapAdjacentAtDepth K dimsView depth1
  let Vh := Buffer.releaseThen V <| Buffer.swapAdjacentAtDepth V dimsView depth1
  let scaleDenom : Float := if headDim = 0 then 1.0 else Float.sqrt (Float.ofNat headDim)
  let scale : Float := 1.0 / scaleDenom
  -- Optional mask: `mask[i,j]=true` means allowed.
  let maskB : Option Buffer :=
    match mask with
    | none => none
    | some m =>
        let mF := Buffer.ofFloatArray
          (Convert.flattenBoolMask (s := .dim n (.dim n .scalar)) m) (Buffer.dtype Qh)
        let inDims : Array Nat := #[n, n]
        let outDims : Array Nat := #[batchHeads, n, n]
        let axisMap : Array Nat := #[0, 1, 2]
        let maskB := Buffer.broadcastTo mF inDims outDims axisMap
        some (Buffer.releaseThen mF maskB)
  let result ← liftM <| IO.lazyPure fun _ =>
    Buffer.attentionForward Qh Kh Vh maskB batchHeads32 n32 head32 scale
  for buffer in maskB.toArray do
    discard <| liftM (Buffer.releaseIO buffer)
  let (outHeads, probabilities) ← match result with
    | .ok output => pure output
    | .error message =>
        for buffer in #[Qh, Kh, Vh] do
          discard <| liftM (Buffer.releaseIO buffer)
        throw message
  -- Combine heads and fold the leading axes back into `rows` for the output projection.
  let concat := Buffer.releaseThen outHeads <|
    Buffer.swapAdjacentAtDepth outHeads dimsHead depth1
  let y := Buffer.bmm concat wo one32 rows32 proj32 dModel32
  let node : Node SomeBuffer :=
    { name := some "attention"
      value := { s := inputShape, buf := y }
      requiresGrad := (t.getNode? wqId).any (·.requiresGrad) ||
        (t.getNode? wkId).any (·.requiresGrad) ||
        (t.getNode? wvId).any (·.requiresGrad) ||
        (t.getNode? woId).any (·.requiresGrad) ||
        (t.getNode? xId).any (·.requiresGrad)
      parents := #[wqId, wkId, wvId, woId, xId]
      cleanup := #[Qh, Kh, Vh, probabilities, concat].map SomeBuffer.ofWorkspace
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny inputShape
        -- Backprop through output projection: y = concat @ wo
        let dConcat :=
          Buffer.bmmRightTranspose dLdy.buf wo one32 rows32 dModel32 proj32
        let dWo :=
          Buffer.bmmLeftTranspose concat dLdy.buf one32 proj32 rows32 dModel32
        let dOutHeads := Buffer.swapAdjacentAtDepth dConcat dimsView depth1
        let (dQh, dKh, dVh) ←
          match Buffer.attentionBackward Qh Kh Vh probabilities dOutHeads
              batchHeads32 n32 head32 scale with
          | .ok (dq, dk, dv) =>
              pure (Buffer.releaseThen dOutHeads dq, dk, dv)
          | .error message => .error message
        -- Undo the head permutation and view each projection gradient as `(rows, projDim)`.
        let dQ := Buffer.releaseThen dQh <| Buffer.swapAdjacentAtDepth dQh dimsHead depth1
        let dK := Buffer.releaseThen dKh <| Buffer.swapAdjacentAtDepth dKh dimsHead depth1
        let dV := Buffer.releaseThen dVh <| Buffer.swapAdjacentAtDepth dVh dimsHead depth1
        -- Backprop projections Q = x @ wq etc.
        let dxQ := Buffer.bmmRightTranspose dQ wq one32 rows32 proj32 dModel32
        let dxK := Buffer.bmmRightTranspose dK wk one32 rows32 proj32 dModel32
        let dxV := Buffer.bmmRightTranspose dV wv one32 rows32 proj32 dModel32
        let dxQK := Buffer.add dxQ dxK
        let dxRaw := Buffer.add dxQK dxV
        let dx := Buffer.releaseThen dxQ <| Buffer.releaseThen dxK <|
          Buffer.releaseThen dxV <| Buffer.releaseThen dxQK dxRaw
        let dWq := Buffer.bmmLeftTranspose x dQ one32 dModel32 rows32 proj32
        let dWk := Buffer.bmmLeftTranspose x dK one32 dModel32 rows32 proj32
        let dWv := Buffer.bmmLeftTranspose x dV one32 dModel32 rows32 proj32
        let dWv := Buffer.releaseThen dConcat <| Buffer.releaseThen dQ <|
          Buffer.releaseThen dK <| Buffer.releaseThen dV dWv
        pure #[
          (xId,  { s := inputShape, buf := dx }),
          (wqId, { s := .dim dModel (.dim projDim .scalar), buf := dWq }),
          (wkId, { s := .dim dModel (.dim projDim .scalar), buf := dWk }),
          (wvId, { s := .dim dModel (.dim projDim .scalar), buf := dWv }),
          (woId, { s := .dim projDim (.dim dModel .scalar), buf := dWo })
        ] }
  pure (t.addNode node) : ExceptT String IO (Tape SomeBuffer × Nat)).run

end Internal

/--
Self-attention with shared projection weights and an optional leading batch dimension.

Without `batch`, the input is `[n, dModel]`; `batch := some b` selects `[b, n, dModel]`.
The head count is `numHeads`, including one for single-head attention.
-/
def attention
  {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (t : Tape SomeBuffer) (wqId wkId wvId woId xId : Nat)
  (mask : Option (Tensor Bool [n, n]) := none)
  (batch : Option Nat := none) (hBatch : batch.getD 1 ≠ 0 := by decide) :
  IO (Result (Tape SomeBuffer × Nat)) :=
  let shape := match batch with
    | none => Shape.dim n (.dim dModel .scalar)
    | some b => Shape.dim b (.dim n (.dim dModel .scalar))
  Internal.attention
    (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
    h1 (batch.getD 1) hBatch shape t wqId wkId wvId woId xId mask

end Tape

end LibTorch
end Autograd
end Runtime
