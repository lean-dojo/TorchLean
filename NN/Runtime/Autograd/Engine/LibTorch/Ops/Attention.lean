/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.LibTorch.Ops.Core
public import NN.Runtime.Autograd.Engine.LibTorch.Convert

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
  let rows ← AnyBuffer.natToU32Checked (batch.toNat * n.toNat)
  let values ← AnyBuffer.natToU32Checked (rows.toNat * d.toNat)
  let scores ← AnyBuffer.natToU32Checked (rows.toNat * n.toNat)
  pure (rows, values, scores)

private def requireAttentionSize (label : String) (buffer : Buffer) (expected : UInt32)
    (dtype : Dtype) :
    Except String Unit := do
  if buffer.size != expected then
    throw s!"attention: {label} buffer size mismatch (expected {expected}, got {buffer.size})"
  unless buffer.dtype == dtype do
    throw s!"attention: {label} buffer dtype mismatch"

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
  -- Shrinking each operand avoids overflow in an otherwise finite scaled dot product.
  -- For growing scales, scale the product instead: enlarging K can overflow even when Q is zero.
  -- As in math SDPA, split the host scale before rounding its factors in the buffer dtype.
  let logits := if scale.abs < 1.0 then
      let factor := scale.abs.sqrt
      let scaledQ := Buffer.scale Q (if scale < 0.0 then -factor else factor)
      let scaledK := Buffer.scale K factor
      releaseThen scaledQ <| releaseThen scaledK <|
        bmmRightTranspose scaledQ scaledK batch n d n
    else
      let products := bmmRightTranspose Q K batch n d n
      releaseThen products <| Buffer.scale products scale
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

`dP = dOut Vᵀ`, `dS = P * (dP - sum(P * dP))`,
`dQ = scale * dS K`, `dK = scale * dSᵀ Q`, and `dV = Pᵀ dOut`.
All arguments are borrowed. Fully blocked rows contribute zero even for nonfinite cotangents.
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
  let dp := bmmRightTranspose grad V batch n d n
  let allowedDP := releaseThen dp <| Buffer.mask dp probabilities
  let ds := releaseThen allowedDP <|
    Tape.rowSoftmaxBwd probabilities allowedDP rows n
  let (dq, dk) := if scale.abs < 1.0 then
      let factor := scale.abs.sqrt
      let queryFactor := if scale < 0.0 then -factor else factor
      let scaledQ := releaseThen maskedQ <| Buffer.scale maskedQ queryFactor
      let scaledK := Buffer.scale K factor
      let dq := releaseThen scaledK <| bmm ds scaledK batch n n d
      let dk := releaseThen scaledQ <| bmmLeftTranspose ds scaledQ batch n n d
      (releaseThen dq <| Buffer.scale dq queryFactor,
        releaseThen ds <| releaseThen dk <| Buffer.scale dk factor)
    else
      let dq := bmm ds K batch n n d
      let dk := releaseThen maskedQ <| bmmLeftTranspose ds maskedQ batch n n d
      (releaseThen dq <| Buffer.scale dq scale,
        releaseThen ds <| releaseThen dk <| Buffer.scale dk scale)
  let dv := releaseThen grad <| bmmLeftTranspose probabilities grad batch n n d
  -- Nonfinite keys can produce 0 * infinity in a fully blocked row's dQ.
  let dq := releaseThen activeB <| releaseThen dq <| Buffer.mask dq activeB
  pure (dq, dk, dv)

end Buffer

namespace Tape

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
  (t : Tape) (wqId wkId wvId woId xId : Nat)
  (mask : Option (Tensor Bool [n, n]) := none) :
  IO (Result (Tape × Nat)) := (do
  let one32 : UInt32 := 1
  let depth1 : UInt32 := 1
  let n32 ← ExceptT.mk (pure <| AnyBuffer.natToU32Checked n)
  let dModel32 ← ExceptT.mk (pure <| AnyBuffer.natToU32Checked dModel)
  let head32 ← ExceptT.mk (pure <| AnyBuffer.natToU32Checked headDim)
  let projDim : Nat := numHeads * headDim
  let proj32 ← ExceptT.mk (pure <| AnyBuffer.natToU32Checked projDim)
  let rows : Nat := batch * n
  let rows32 ← ExceptT.mk (pure <| AnyBuffer.natToU32Checked rows)
  let batchHeads : Nat := batch * numHeads
  let batchHeads32 ← ExceptT.mk (pure <| AnyBuffer.natToU32Checked batchHeads)
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
  let node : Node :=
    { name := some "attention"
      value := { s := inputShape, buf := y }
      requiresGrad := (t.getNode? wqId).any (·.requiresGrad) ||
        (t.getNode? wkId).any (·.requiresGrad) ||
        (t.getNode? wvId).any (·.requiresGrad) ||
        (t.getNode? woId).any (·.requiresGrad) ||
        (t.getNode? xId).any (·.requiresGrad)
      parents := #[wqId, wkId, wvId, woId, xId]
      cleanup := #[Qh, Kh, Vh, probabilities, concat]
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
  pure (t.addNode node) : ExceptT String IO (Tape × Nat)).run

end Internal

/--
Self-attention with shared projection weights and an optional leading batch dimension.

Without `batch`, the input is `[n, dModel]`; `batch := some b` selects `[b, n, dModel]`.
The head count is `numHeads`, including one for single-head attention.
-/
def attention
  {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (t : Tape) (wqId wkId wvId woId xId : Nat)
  (mask : Option (Tensor Bool [n, n]) := none)
  (batch : Option Nat := none) (hBatch : batch.getD 1 ≠ 0 := by decide) :
  IO (Result (Tape × Nat)) :=
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
