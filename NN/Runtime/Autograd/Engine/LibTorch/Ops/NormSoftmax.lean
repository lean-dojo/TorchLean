/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.LibTorch.Ops.Core
public import NN.Spec.Core.Context

/-!
# CUDA Tape Operations: Normalization and Row Softmax
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace LibTorch

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Tape

/-!
## Normalization
-/

/--
LayerNorm over the last dimension for `(seqLen, embedDim)` buffers.

The tape records one normalization operation and keeps TorchLean's usual VJP. The native
primitive evaluates the forward formula and that VJP with LibTorch tensor operations; intermediate
tensors remain internal to the primitive.

`epsilon` is added to the variance before taking the square root. Backward reuses the normalized
input and inverse standard deviation saved by forward, so both passes use the caller's value.
-/
@[inline]
def layerNorm {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
  (t : Tape) (xId gammaId betaId : Nat)
  (epsilon : Float := TorchLean.normalizationEpsilon) : Result (Tape × Nat) := do
  have _ := h_seq_pos
  have _ := h_embed_pos
  let rows32 ← AnyBuffer.natToU32Checked seqLen
  let cols32 ← AnyBuffer.natToU32Checked embedDim
  let x ← requireValue (t := t) xId (.dim seqLen (.dim embedDim .scalar))
  let gamma ← requireValue (t := t) gammaId (.dim embedDim .scalar)
  let beta ← requireValue (t := t) betaId (.dim embedDim .scalar)
  let invCols : Float := 1.0 / Float.ofNat embedDim
  let (y, xHat, invStd) :=
    Buffer.layerNormFwd x gamma beta rows32 cols32 invCols epsilon
  let outShape : Shape := .dim seqLen (.dim embedDim .scalar)
  let node : Node :=
    { name := some "layer_norm"
      value := { s := outShape, buf := y }
      requiresGrad := (t.getNode? xId).any (·.requiresGrad) ||
        (t.getNode? gammaId).any (·.requiresGrad) ||
        (t.getNode? betaId).any (·.requiresGrad)
      parents := #[xId, gammaId, betaId]
      cleanup := #[xHat, invStd]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        let (dx, dGamma, dBeta) :=
          Buffer.layerNormBwd dLdy.buf xHat invStd gamma rows32 cols32
            (Float.ofNat embedDim) invCols
        pure #[
          (xId, { s := outShape, buf := dx }),
          (gammaId, { s := .dim embedDim .scalar, buf := dGamma }),
          (betaId, { s := .dim embedDim .scalar, buf := dBeta })
        ] }
  pure (t.addNode node)

/--
Batch normalization over every axis after the channel axis.

The spatial shape is folded to one contiguous dimension for the CUDA reduction. This is a view of
the storage layout, not a rank-specific implementation.
-/
@[inline] def batchNorm {channels : Nat} {spatial : Shape}
    (hWellFormed : (Shape.dim channels spatial).wellFormed)
    (t : Tape) (xId gammaId betaId : Nat)
    (epsilon : Float := TorchLean.normalizationEpsilon) : Result (Tape × Nat) := do
  have _hChannels : channels > 0 := hWellFormed.1
  have _hSpatial : Shape.size spatial > 0 :=
    Shape.size_pos_of_well_formed hWellFormed.2
  let rows32 ← AnyBuffer.natToU32Checked channels
  let cols : Nat := Shape.size spatial
  let cols32 ← AnyBuffer.natToU32Checked cols
  let xShape : Shape := .dim channels spatial
  let x ← requireValue (t := t) xId xShape
  let gamma ← requireValue (t := t) gammaId (.dim channels .scalar)
  let beta ← requireValue (t := t) betaId (.dim channels .scalar)
  -- Treat as a zero-copy (channels, cols) view; the underlying layout already matches.
  let sum1 := Buffer.reduceSumByRow x rows32 cols32
  let invCols : Float := 1.0 / Float.ofNat cols
  let mean := Buffer.scale sum1 invCols
  let meanB := Buffer.broadcastVecToCols mean rows32 cols32
  let centered := Buffer.sub x meanB
  let centered2 := Buffer.mul centered centered
  let varSum := Buffer.reduceSumByRow centered2 rows32 cols32
  let var := Buffer.scale varSum invCols
  let epsVec := Buffer.full rows32 epsilon
  let varEps := Buffer.add var epsVec
  let std := Buffer.sqrt varEps
  let stdB := Buffer.broadcastVecToCols std rows32 cols32
  let xHat := Buffer.div centered stdB
  let gammaB := Buffer.broadcastVecToCols gamma rows32 cols32
  let betaB := Buffer.broadcastVecToCols beta rows32 cols32
  let xHatGamma := Buffer.mul xHat gammaB
  let y := Buffer.releaseManyThen
    #[sum1, mean, meanB, centered, centered2, varSum, var, epsVec, varEps,
      stdB, betaB, xHatGamma] (Buffer.add xHatGamma betaB)
  let node : Node :=
    { name := some "batch_norm"
      value := { s := xShape, buf := y }
      requiresGrad := (t.getNode? xId).any (·.requiresGrad) ||
        (t.getNode? gammaId).any (·.requiresGrad) ||
        (t.getNode? betaId).any (·.requiresGrad)
      parents := #[xId, gammaId, betaId]
      cleanup := #[std, xHat, gammaB]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny xShape
        -- dBeta / dGamma sum over spatial dimension (axis=1 of the folded matrix).
        let dBeta := Buffer.reduceSumByRow dLdy.buf rows32 cols32
        let dGammaPointwise := Buffer.mul dLdy.buf xHat
        let dGamma := Buffer.releaseThen dGammaPointwise <|
          Buffer.reduceSumByRow dGammaPointwise rows32 cols32
        -- dX
        let dXhat := Buffer.mul dLdy.buf gammaB
        let sumDXhat := Buffer.reduceSumByRow dXhat rows32 cols32
        let dXhatXhat := Buffer.mul dXhat xHat
        let sumDXhatXhat := Buffer.releaseThen dXhatXhat <|
          Buffer.reduceSumByRow dXhatXhat rows32 cols32
        let sum1B := Buffer.broadcastVecToCols sumDXhat rows32 cols32
        let sum2B := Buffer.broadcastVecToCols sumDXhatXhat rows32 cols32
        let scaledDXhat := Buffer.scale dXhat (Float.ofNat cols)
        let centeredDXhat := Buffer.sub scaledDXhat sum1B
        let xHatSum2 := Buffer.mul xHat sum2B
        let term := Buffer.sub centeredDXhat xHatSum2
        let invStd := Buffer.inv std
        let invStdB := Buffer.broadcastVecToCols invStd rows32 cols32
        let termInv := Buffer.mul term invStdB
        let dxRaw := Buffer.scale termInv invCols
        let dx :=
          Buffer.releaseThen dXhat <| Buffer.releaseThen sumDXhat <|
            Buffer.releaseThen sumDXhatXhat <| Buffer.releaseThen sum1B <|
              Buffer.releaseThen sum2B <| Buffer.releaseThen scaledDXhat <|
                Buffer.releaseThen centeredDXhat <| Buffer.releaseThen xHatSum2 <|
                  Buffer.releaseThen term <| Buffer.releaseThen invStd <|
                    Buffer.releaseThen invStdB <| Buffer.releaseThen termInv dxRaw
        pure #[
          (xId, { s := xShape, buf := dx }),
          (gammaId, { s := .dim channels .scalar, buf := dGamma }),
          (betaId, { s := .dim channels .scalar, buf := dBeta })
        ] }
  pure (t.addNode node)

/-!
## Softmax (last axis, row folding)

All leading dimensions are folded into one row axis; normalization acts independently on the
last axis at any rank. Empty tensors retain their shape. A scalar softmax is one, and a scalar
log-softmax is zero; both have zero cotangent.
-/

/--
Record a row operation whose VJP uses the saved output.

The scalar forward function runs only after input validation. Empty tensors are copied; for
non-scalar, nonempty tensors, row-dimension validation precedes input lookup.
-/
@[inline] def Internal.rowOpLast {s : Shape} (t : Tape) (opName : String) (xId : Nat)
    (scalarForward : Unit → Buffer)
    (forward : Buffer → UInt32 → UInt32 → Buffer.WithWorkspace)
    (backward : Buffer → Buffer → UInt32 → UInt32 → Buffer) : Result (Tape × Nat) := do
  -- With no coordinates, both the result and its cotangent have the same empty shape.
  if Shape.size s == 0 then
    return ← unary t opName xId s s Buffer.copy (fun _ gradient => Buffer.copy gradient)
  match s with
  | .scalar =>
      let _x ← requireValue (t := t) xId Shape.scalar
      let one32 : UInt32 := 1
      let y := scalarForward ()
      let node : Node :=
        { name := some opName
          value := { s := Shape.scalar, buf := y }
          requiresGrad := (t.getNode? xId).any (·.requiresGrad)
          parents := #[xId]
          backward := fun dLdyAny => do
            let _ ← requireGrad dLdyAny Shape.scalar
            let dx := Buffer.zeros one32
            pure #[(xId, { s := Shape.scalar, buf := dx })] }
      pure (t.addNode node)
  | _ =>
      let (rows32, cols32) ← foldRowsColsLastAxis s
      let x ← requireValue (t := t) xId s
      let yOwned := forward x rows32 cols32
      let y := yOwned.releaseWorkspaceThen yOwned.value
      let node : Node :=
        { name := some opName
          value := { s := s, buf := y }
          requiresGrad := (t.getNode? xId).any (·.requiresGrad)
          parents := #[xId]
          backward := fun dLdyAny => do
            let dLdy ← requireGrad dLdyAny s
            let dx := backward y dLdy.buf rows32 cols32
            pure #[(xId, { s := s, buf := dx })] }
      pure (t.addNode node)

/-- Record a last-axis softmax on the tape, returning the extended tape and the new node id. -/
@[inline] def softmaxLast {s : Shape} (t : Tape) (xId : Nat) : Result (Tape × Nat) :=
  Internal.rowOpLast (s := s) t "softmax" xId (fun () => Buffer.full 1 1.0)
    rowSoftmaxForward rowSoftmaxBwd

/-- Stable log-softmax along the last axis, implemented directly on CUDA buffers. -/
@[inline] def logSoftmaxLast {s : Shape} (t : Tape) (xId : Nat) : Result (Tape × Nat) :=
  Internal.rowOpLast (s := s) t "log_softmax" xId (fun () => Buffer.zeros 1)
    rowLogSoftmaxForward rowLogSoftmaxBwd

end Tape

end LibTorch
end Autograd
end Runtime
