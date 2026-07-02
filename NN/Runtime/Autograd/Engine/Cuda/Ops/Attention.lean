/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Ops.NormSoftmax

/-!
# CUDA Tape Operations: Attention
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Cuda

open Spec
open Tensor

namespace Tape

/-!
## Multi-head self-attention

Forward structure matches `Spec.MultiHeadAttention.forward`:
1. `Q = x @ Wq`, `K = x @ Wk`, `V = x @ Wv`
2. reshape to heads `(numHeads, n, headDim)`
3. attention per head (batched): `softmax(Q Kᵀ / sqrt(headDim)) @ V`
4. combine heads, then output projection `@ Wo`

Masking:
- If `mask` is provided, we upload it as a float32 `{0,1}` matrix and apply hard-mask semantics:
  blocked entries contribute zero softmax numerator, and their score gradients are zero.
- This incurs a host-to-device copy for the mask (since the mask is a host `Tensor Bool`).
-/

def multiHeadAttention
  {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (t : Tape) (wqId wkId wvId woId xId : Nat)
  (mask : Option (Tensor Bool (.dim n (.dim n .scalar))) := none)
  (useFlash : Bool := true) :
  Result (Tape × Nat) := do
  have _ := h1
  let one32 : UInt32 := 1
  let depth0 : UInt32 := 0
  let depth1 : UInt32 := 1
  let n32 ← u32 n
  let h32 ← u32 numHeads
  let dModel32 ← u32 dModel
  let head32 ← u32 headDim
  let projDim : Nat := numHeads * headDim
  let proj32 ← u32 projDim
  let wq ← requireValue (t := t) wqId (.dim dModel (.dim projDim .scalar))
  let wk ← requireValue (t := t) wkId (.dim dModel (.dim projDim .scalar))
  let wv ← requireValue (t := t) wvId (.dim dModel (.dim projDim .scalar))
  let wo ← requireValue (t := t) woId (.dim projDim (.dim dModel .scalar))
  let x ← requireValue (t := t) xId (.dim n (.dim dModel .scalar))
  -- Projections: (n,dModel) @ (dModel,projDim) -> (n,projDim)
  let Q := Buffer.bmm x wq one32 n32 dModel32 proj32
  let K := Buffer.bmm x wk one32 n32 dModel32 proj32
  let V := Buffer.bmm x wv one32 n32 dModel32 proj32
  -- Split heads:
  --   (n, projDim) views as (n, numHeads, headDim) with the same underlying layout,
  --   then we swap to (numHeads, n, headDim) so each head is the batch axis for `bmm`.
  let dimsView : Array Nat := #[n, numHeads, headDim]
  let dimsHead : Array Nat := #[numHeads, n, headDim]
  let Qh := Buffer.swapAdjacentAtDepth Q dimsView depth0  -- (numHeads,n,headDim)
  let Kh := Buffer.swapAdjacentAtDepth K dimsView depth0  -- (numHeads,n,headDim)
  let Vh := Buffer.swapAdjacentAtDepth V dimsView depth0  -- (numHeads,n,headDim)
  let scale : Float := 1.0 / Float.sqrt (Float.ofNat headDim)
  -- Optional mask: `mask[i,j]=true` means allowed; false entries contribute zero numerator.
  let (maskB, hasMask) : Buffer × UInt32 :=
    match mask with
    | none => (Buffer.zeros 0, 0)
    | some m =>
        let mF := Buffer.ofFloatArray (Convert.flattenBoolMask (s := .dim n (.dim n .scalar)) m)
        let inDims : Array Nat := #[n, n]
        let outDims : Array Nat := #[numHeads, n, n]
        let axisMap : Array Nat := #[0, 1, 2]
        let maskB := Buffer.broadcastTo mF inDims outDims axisMap
        (Buffer.releaseThen mF maskB, 1)
  -- Fused native attention over split heads. This replaces the composed
  -- `scores -> mask -> softmax -> bmm` path while keeping the same spec contract.
  let (outHeads, attentionWorkspace) ←
    if useFlash then
      let outHeads := Buffer.flashAttentionFwd Qh Kh Vh maskB hasMask h32 n32 head32 scale
      pure (outHeads, ([] : List Buffer))
    else
      let KhT := Buffer.swapAdjacentAtDepth Kh dimsHead depth1
      let scores := Buffer.bmm Qh KhT h32 n32 head32 n32
      let scaled0 := Buffer.scale scores scale
      let rowsFold32 ← u32 (numHeads * n)
      let attnOwned :=
        match mask with
        | none => rowSoftmaxForward scaled0 rowsFold32 n32
        | some _ => rowHardMaskedSoftmaxForward scaled0 maskB rowsFold32 n32
      let outHeads := Buffer.bmm attnOwned.value Vh h32 n32 n32 head32
      pure (outHeads, [KhT, scores, scaled0, attnOwned.value] ++
        attnOwned.workspace)
  -- combine heads: swap to (n,numHeads,headDim), then reshape to (n,projDim)
  let swapped := Buffer.swapAdjacentAtDepth outHeads dimsHead depth0  -- (n,numHeads,headDim)
  let concat := swapped  -- view as (n,projDim)
  -- output projection: (n,projDim) @ (projDim,dModel) -> (n,dModel)
  let y := Buffer.bmm concat wo one32 n32 proj32 dModel32
  let outShape : Shape := .dim n (.dim dModel .scalar)
  let node : Node :=
    { name := some "multi_head_attention"
      value := { s := outShape, buf := y }
      requires_grad := true
      parents := [wqId, wkId, wvId, woId, xId]
      cleanup := [Q, K, V, Qh, Kh, Vh, maskB, outHeads, swapped] ++ attentionWorkspace
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        -- Backprop through output projection: y = concat @ wo
        let woT := Buffer.transpose2d wo proj32 dModel32
        let dConcat := Buffer.releaseThen woT <|
          Buffer.bmm dLdy.buf woT one32 n32 dModel32 proj32
        let concatT := Buffer.transpose2d concat n32 proj32
        let dWo := Buffer.releaseThen concatT <|
          Buffer.bmm concatT dLdy.buf one32 proj32 n32 dModel32
        -- Backprop combine-heads: reshape/view then swap back
        let dSwapped := dConcat -- view (n,numHeads,headDim)
        let dOutHeads := Buffer.swapAdjacentAtDepth dSwapped dimsView depth0 -- (numHeads,n,headDim)
        let (dQh, dKh, dVh) ←
          if useFlash then
            -- Fused VJP for the native attention primitive. The kernels recompute the row-wise
            -- softmax summaries instead of materializing/storing the full attention matrix.
            let dQh := Buffer.flashAttentionBwdQ Qh Kh Vh maskB dOutHeads hasMask h32 n32 head32 scale
            let dKh := Buffer.flashAttentionBwdK Qh Kh Vh maskB dOutHeads hasMask h32 n32 head32 scale
            let dVh := Buffer.releaseThen dOutHeads <|
              Buffer.flashAttentionBwdV Qh Kh Vh maskB dOutHeads hasMask h32 n32 head32 scale
            pure (dQh, dKh, dVh)
          else
            let dimsAttn : Array Nat := #[numHeads, n, n]
            let KhT := Buffer.swapAdjacentAtDepth Kh dimsHead depth1
            let scores := Buffer.bmm Qh KhT h32 n32 head32 n32
            let scaled0 := Buffer.scale scores scale
            let rowsFold32 ← u32 (numHeads * n)
            let attnOwned :=
              match mask with
              | none => rowSoftmaxForward scaled0 rowsFold32 n32
              | some _ => rowHardMaskedSoftmaxForward scaled0 maskB rowsFold32 n32
            let VhT := Buffer.swapAdjacentAtDepth Vh dimsHead depth1
            let dAttn := Buffer.releaseThen VhT <|
              Buffer.bmm dOutHeads VhT h32 n32 head32 n32
            let attnT := Buffer.swapAdjacentAtDepth attnOwned.value dimsAttn depth1
            let dVh := Buffer.releaseThen attnT <|
              Buffer.releaseThen dOutHeads <|
                Buffer.bmm attnT dOutHeads h32 n32 n32 head32
            let dScaled := rowSoftmaxBwd attnOwned.value dAttn rowsFold32 n32
            let dScoresMasked := Buffer.scale dScaled scale
            let dScores := dScoresMasked
            let dQh := Buffer.bmm dScores Kh h32 n32 n32 head32
            let dScoresT := Buffer.swapAdjacentAtDepth dScores dimsAttn depth1
            let dKh := Buffer.releaseThen dScoresT <|
              Buffer.bmm dScoresT Qh h32 n32 n32 head32
            let dQh := Buffer.releaseThen KhT <| Buffer.releaseThen scores <|
              Buffer.releaseThen scaled0 <|
                Buffer.releaseThen attnOwned.value <| Buffer.releaseThen dAttn <|
                  Buffer.releaseThen dScaled <| Buffer.releaseThen dScoresMasked <|
                    attnOwned.releaseWorkspaceThen dQh
            pure (dQh, dKh, dVh)
        -- Backprop split-head permutations: swap back to (n,numHeads,headDim), then view as (n,projDim).
        let dQ := Buffer.releaseThen dQh <| Buffer.swapAdjacentAtDepth dQh dimsHead depth0
        let dK := Buffer.releaseThen dKh <| Buffer.swapAdjacentAtDepth dKh dimsHead depth0
        let dV := Buffer.releaseThen dVh <| Buffer.swapAdjacentAtDepth dVh dimsHead depth0
        -- Backprop projections Q = x @ wq etc.
        let wqT := Buffer.transpose2d wq dModel32 proj32
        let wkT := Buffer.transpose2d wk dModel32 proj32
        let wvT := Buffer.transpose2d wv dModel32 proj32
        let dxQ := Buffer.releaseThen wqT <| Buffer.bmm dQ wqT one32 n32 proj32 dModel32
        let dxK := Buffer.releaseThen wkT <| Buffer.bmm dK wkT one32 n32 proj32 dModel32
        let dxV := Buffer.releaseThen wvT <| Buffer.bmm dV wvT one32 n32 proj32 dModel32
        let dxQK := Buffer.add dxQ dxK
        let dxRaw := Buffer.add dxQK dxV
        let dx := Buffer.releaseThen dxQ <| Buffer.releaseThen dxK <|
          Buffer.releaseThen dxV <| Buffer.releaseThen dxQK dxRaw
        let xT := Buffer.transpose2d x n32 dModel32
        let dWq := Buffer.bmm xT dQ one32 dModel32 n32 proj32
        let dWk := Buffer.bmm xT dK one32 dModel32 n32 proj32
        let dWv := Buffer.releaseThen xT <| Buffer.bmm xT dV one32 dModel32 n32 proj32
        let dWv := Buffer.releaseThen dConcat <| Buffer.releaseThen dQ <|
          Buffer.releaseThen dK <| Buffer.releaseThen dV dWv
        pure [
          (xId,  { s := .dim n (.dim dModel .scalar), buf := dx }),
          (wqId, { s := .dim dModel (.dim projDim .scalar), buf := dWq }),
          (wkId, { s := .dim dModel (.dim projDim .scalar), buf := dWk }),
          (wvId, { s := .dim dModel (.dim projDim .scalar), buf := dWv }),
          (woId, { s := .dim projDim (.dim dModel .scalar), buf := dWo })
        ] }
  pure (t.addNode node)

end Tape

end Cuda
end Autograd
end Runtime
