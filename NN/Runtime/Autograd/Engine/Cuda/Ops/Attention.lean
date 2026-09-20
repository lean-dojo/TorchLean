/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Attention
public import NN.Runtime.Autograd.Engine.Cuda.Ops.Core
public import Mathlib.Algebra.GroupWithZero.Nat
public import NN.Runtime.Autograd.Engine.Cuda.Convert

/-!
# CUDA Tape Operations: Attention
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Cuda

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Tape

/-!
## Multi-head self-attention

Forward structure matches `Spec.MultiHeadAttention.forward`:
1. `Q = x @ Wq`, `K = x @ Wk`, `V = x @ Wv`
2. reshape to heads `(numHeads, n, headDim)`
3. attention per head (batched): `softmax(Q Kᵀ / sqrt(headDim)) @ V`
4. combine heads, then output projection `@ Wo`

Masking:
- Every provider uses hard-mask semantics: blocked entries contribute zero softmax numerator.
- The selected capsule determines the forward provider and VJP owner. Native CUDA uses its fused
  VJP; the TorchLean and LibTorch-forward routes record the same TorchLean local VJP.
- This incurs a host-to-device copy for the mask (since the mask is a host `Tensor Bool`).
-/

namespace Internal

/-- Shared implementation behind the single-sample and batched attention entrypoints. -/
def multiHeadAttention
  {n numHeads dModel headDim : Nat} (_hSeq : n ≠ 0)
  (batch : Nat) (_hBatch : batch ≠ 0) (inputShape outputShape : Shape) (nodeName : String)
  (t : Tape) (wqId wkId wvId woId xId : Nat)
  (mask : Option (Tensor Bool [n, n]) := none)
  (attentionCapsule : NN.Backend.KernelCapsule := NN.Backend.Attention.torchLeanComposed) :
  IO (Result (Tape × Nat)) := (do
  ExceptT.mk (pure <|
    attentionCapsule.validateEagerRequest .scaledDotProductAttention .cuda
  )
  match attentionCapsule.provider, attentionCapsule.vjpMode with
  | .nativeCuda, .backendVJP => pure ()
  | .torchLean, .torchLeanTape => pure ()
  | .libTorch, .torchLeanTape => pure ()
  | provider, vjpMode =>
      throw <|
        s!"attention capsule `{attentionCapsule.name}` has no eager executor for " ++
          s!"provider `{reprStr provider}` with VJP mode `{reprStr vjpMode}`"
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
  -- Optional mask: `mask[i,j]=true` means allowed for every attention provider.
  let (maskB, hasMask) : Buffer × UInt32 :=
    match mask with
    | none => (Buffer.zeros 0, 0)
    | some m =>
        let mF := Buffer.ofFloatArray (Convert.flattenBoolMask (s := .dim n (.dim n .scalar)) m)
        let inDims : Array Nat := #[n, n]
        let outDims : Array Nat := #[batchHeads, n, n]
        let axisMap : Array Nat := #[0, 1, 2]
        let maskB := Buffer.broadcastTo mF inDims outDims axisMap
        (Buffer.releaseThen mF maskB, 1)
  let outHeads ← match attentionCapsule.provider with
    | .nativeCuda =>
        pure <| Buffer.flashAttentionFwd
          Qh Kh Vh maskB hasMask batchHeads32 n32 head32 scale
    | .libTorch =>
        Buffer.libTorchSDPAFwd Qh Kh Vh maskB hasMask batchHeads32 n32 head32 scale
    | .torchLean =>
      let scores := Buffer.bmmRightTranspose Qh Kh batchHeads32 n32 head32 n32
      let scaled0 := Buffer.scale scores scale
      let rowsFold32 ← ExceptT.mk (pure <| AnyBuffer.natToU32Checked (batchHeads * n))
      let attnOwned : Buffer.WithWorkspace :=
        match mask with
        | none => rowSoftmaxForward scaled0 rowsFold32 n32
        | some _ => rowHardMaskedSoftmaxForward scaled0 maskB rowsFold32 n32
      let outHeadsRaw := Buffer.bmm attnOwned.value Vh batchHeads32 n32 n32 head32
      let outHeads := Buffer.releaseThen scores <| Buffer.releaseThen scaled0 <|
        Buffer.releaseThen attnOwned.value <| attnOwned.releaseWorkspaceThen outHeadsRaw
      pure outHeads
    | provider =>
        throw s!"attention provider `{reprStr provider}` is not implemented by the CUDA tape"
  -- Combine heads and fold the leading axes back into `rows` for the output projection.
  let swapped := Buffer.swapAdjacentAtDepth outHeads dimsHead depth1
  let concat := swapped
  let y := Buffer.bmm concat wo one32 rows32 proj32 dModel32
  let node : Node :=
    { name := some nodeName
      value := { s := outputShape, buf := y }
      requiresGrad := true
      parents := #[wqId, wkId, wvId, woId, xId]
      cleanup := #[Qh, Kh, Vh, maskB, outHeads, swapped]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outputShape
        -- Backprop through output projection: y = concat @ wo
        let dConcat :=
          Buffer.bmmRightTranspose dLdy.buf wo one32 rows32 dModel32 proj32
        let dWo :=
          Buffer.bmmLeftTranspose concat dLdy.buf one32 proj32 rows32 dModel32
        let dSwapped := dConcat
        let dOutHeads := Buffer.swapAdjacentAtDepth dSwapped dimsView depth1
        let (dQh, dKh, dVh) ←
          if attentionCapsule.provider == .nativeCuda then
            let (dQh, dKh, dVh) :=
              Buffer.flashAttentionBwd Qh Kh Vh maskB dOutHeads hasMask
                batchHeads32 n32 head32 scale
            -- The fused VJP reads `dOutHeads` but returns three fresh gradient buffers. Thread the
            -- release through one returned buffer so every attention step retires this activation.
            pure (Buffer.releaseThen dOutHeads dQh, dKh, dVh)
          else
            let scores := Buffer.bmmRightTranspose Qh Kh batchHeads32 n32 head32 n32
            let scaled0 := Buffer.scale scores scale
            let rowsFold32 ← AnyBuffer.natToU32Checked (batchHeads * n)
            let attnOwned :=
              match mask with
              | none => rowSoftmaxForward scaled0 rowsFold32 n32
              | some _ => rowHardMaskedSoftmaxForward scaled0 maskB rowsFold32 n32
            let dAttn :=
              Buffer.bmmRightTranspose dOutHeads Vh batchHeads32 n32 head32 n32
            let dVh := Buffer.releaseThen dOutHeads <|
              Buffer.bmmLeftTranspose
                attnOwned.value dOutHeads batchHeads32 n32 n32 head32
            let dScaled := rowSoftmaxBwd attnOwned.value dAttn rowsFold32 n32
            let dScoresMasked := Buffer.scale dScaled scale
            let dScores := dScoresMasked
            let dQh := Buffer.bmm dScores Kh batchHeads32 n32 n32 head32
            let dKh :=
              Buffer.bmmLeftTranspose dScores Qh batchHeads32 n32 n32 head32
            let dQh := Buffer.releaseThen scores <| Buffer.releaseThen scaled0 <|
                Buffer.releaseThen attnOwned.value <| Buffer.releaseThen dAttn <|
                  Buffer.releaseThen dScaled <| Buffer.releaseThen dScoresMasked <|
                    attnOwned.releaseWorkspaceThen dQh
            pure (dQh, dKh, dVh)
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
Single-sample multi-head self-attention.

This is a thin shape wrapper around the batch-aware implementation so both paths use the same
TorchLean forward and VJP code.
-/
def multiHeadAttention
  {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (t : Tape) (wqId wkId wvId woId xId : Nat)
  (mask : Option (Tensor Bool [n, n]) := none)
  (attentionCapsule : NN.Backend.KernelCapsule := NN.Backend.Attention.torchLeanComposed) :
  IO (Result (Tape × Nat)) :=
  Internal.multiHeadAttention
    (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
    h1 1 (by simp) (.dim n (.dim dModel .scalar)) (.dim n (.dim dModel .scalar))
    "multi_head_attention" t wqId wkId wvId woId xId mask attentionCapsule

/--
Batch-aware multi-head self-attention.

The leading batch is folded into the projection row axis and the `(batch, head)` pair becomes the
BMM batch axis. TorchLean still records one typed node and computes the complete local VJP,
including accumulation of the shared projection-weight gradients over every sample.
-/
def batchedMultiHeadAttention
  {batch n numHeads dModel headDim : Nat} (hBatch : batch ≠ 0) (h1 : n ≠ 0)
  (t : Tape) (wqId wkId wvId woId xId : Nat)
  (mask : Option (Tensor Bool [n, n]) := none)
  (attentionCapsule : NN.Backend.KernelCapsule := NN.Backend.Attention.torchLeanComposed) :
  IO (Result (Tape × Nat)) := by
  exact Internal.multiHeadAttention
    (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
    h1 batch hBatch
    (.dim batch (.dim n (.dim dModel .scalar)))
    (.dim batch (.dim n (.dim dModel .scalar)))
    "batched_multi_head_attention" t wqId wkId wvId woId xId mask attentionCapsule

end Tape

end Cuda
end Autograd
end Runtime
