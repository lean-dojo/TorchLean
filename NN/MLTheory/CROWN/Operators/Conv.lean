/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Conv
public import NN.Tensor.Conversion
public import NN.MLTheory.CROWN.Core
public import NN.Spec.Core.TensorReductionShape.ShapeChange

/-!
# Convolution Bounds

CROWN-IBP bounds and an exact flattened affine map for arbitrary-dimensional convolution.

Design notes:
- We flatten the input and convolution output when constructing the exact affine map. This reuses
  `AffineVec` without assigning a special semantic meaning to any spatial axis.
- The convolution operator is explicitly materialized as a matrix whose rows
  correspond to output positions and columns to input positions. The verifier stays deterministic
  for the tensor sizes targeted by the CROWN operator layer.
-/

@[expose] public section

namespace NN.MLTheory.CROWN

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- Flatten a `Box` to a rank-one box by flattening both endpoints. -/
def flattenBox {s : Shape} (B : Box α s) : Box α (.dim (Spec.Shape.size s) .scalar) :=
  { lo := Tensor.flattenSpec B.lo, hi := Tensor.flattenSpec B.hi }

/-- Interval propagation for grouped convolution with independent leading batches.

Products, accumulation and the final bias addition round outwards. Each row follows the
channel/kernel fold order of the convolution specification.
-/
def ibpConv [BoundOps α]
    {d inC outC : Nat} {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (layer : Spec.ConvSpec d inC outC kernel stride padding α)
    (dilation paddingAfter : TorchLean.Tensor Nat [d]) (groups : Nat)
    (leading : Shape)
    (xB : Box α (leading.concat (Shape.ofList (inC :: Tensor.to inSpatial (List Nat))))) :
    Box α (leading.concat (Shape.ofList (outC ::
      Tensor.to (Spec.convOutSpatialDilated inSpatial kernel stride dilation padding paddingAfter)
        (List Nat)))) :=
  let outSpatial :=
    Spec.convOutSpatialDilated inSpatial kernel stride dilation padding paddingAfter
  let row := fun (input : Box α (Shape.ofList (inC :: Tensor.to inSpatial (List Nat)))) =>
    let endpoint := fun (lower : Bool) =>
      Tensor.dim (fun outChannel =>
        TorchLean.Tensor.generate (Tensor.to outSpatial (List Nat)) (fun outIdx =>
          let total :=
            (List.finRange (inC / groups)).foldl (fun acc localChannel =>
              let inChannel := (outChannel.val / (outC / groups)) * (inC / groups) +
                localChannel.val
              Spec.Conv.Internal.foldlIndices (Tensor.to kernel (List Nat)) acc
                (fun acc kernelIdx =>
                  match Spec.Conv.Internal.mkDilatedInputIdx? outIdx kernelIdx
                      (Tensor.to stride (List Nat)) (Tensor.to dilation (List Nat))
                      (Tensor.to padding (List Nat)) with
                  | none => acc
                  | some inputIdx =>
                      let lo := getAtOrZero input.lo (inChannel :: inputIdx)
                      let hi := getAtOrZero input.hi (inChannel :: inputIdx)
                      let weight := getAtOrZero layer.kernel
                        (outChannel.val :: inChannel :: kernelIdx)
                      let multiply :=
                        if lower then BoundOps.mulDown else BoundOps.mulUp
                      let pLo := multiply weight lo
                      let pHi := multiply weight hi
                      let bound :=
                        if lower then
                          if pLo > pHi then pHi else pLo
                        else if pLo > pHi then pLo else pHi
                      if lower then BoundOps.addDown acc bound
                      else BoundOps.addUp acc bound)) 0
          if lower then BoundOps.addDown total (getAtOrZero layer.bias [outChannel.val])
          else BoundOps.addUp total (getAtOrZero layer.bias [outChannel.val])))
    { lo := endpoint true, hi := endpoint false : Box α
        (Shape.ofList (outC :: Tensor.to outSpatial (List Nat))) }
  let rec mapRows : (batch : Shape) →
      Box α (batch.concat (Shape.ofList (inC :: Tensor.to inSpatial (List Nat)))) →
      Box α (batch.concat (Shape.ofList (outC :: Tensor.to outSpatial (List Nat))))
    | .scalar, input => row input
    | .dim n rest, input =>
        let rows := fun i : Fin n =>
          mapRows rest ⟨input.lo.unstack i, input.hi.unstack i⟩
        ⟨Tensor.dim fun i => (rows i).lo, Tensor.dim fun i => (rows i).hi⟩
  mapRows leading xB

/-- Explicit matrix for grouped convolution, block diagonal over the leading batch shape. -/
def convLinearMatrix
    {d inC outC : Nat} {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (layer : Spec.ConvSpec d inC outC kernel stride padding α)
    (dilation paddingAfter : TorchLean.Tensor Nat [d]) (groups : Nat) (leading : Shape) :
    let inShape := leading.concat (Shape.ofList (inC :: Tensor.to inSpatial (List Nat)))
    let outShape := leading.concat (Shape.ofList (outC ::
      Tensor.to (Spec.convOutSpatialDilated inSpatial kernel stride dilation padding paddingAfter)
        (List Nat)))
    Tensor α [outShape.size, inShape.size] :=
  let inShape := leading.concat (Shape.ofList (inC :: Tensor.to inSpatial (List Nat)))
  let outSpatial :=
    Spec.convOutSpatialDilated inSpatial kernel stride dilation padding paddingAfter
  let outShape := leading.concat (Shape.ofList (outC :: Tensor.to outSpatial (List Nat)))
  Tensor.dim (fun row =>
    let outCoordinates := Shape.Coord.toList outShape (Shape.Coord.unlinearize row)
    let outBatch := outCoordinates.take leading.rank
    let outSample := outCoordinates.drop leading.rank
    let outChannel := outSample.headD 0
    let outIdx := outSample.drop 1
    let groupStart := (outChannel / (outC / groups)) * (inC / groups)
    Tensor.dim (fun column =>
      let inCoordinates := Shape.Coord.toList inShape (Shape.Coord.unlinearize column)
      let inBatch := inCoordinates.take leading.rank
      let inSample := inCoordinates.drop leading.rank
      let inChannel := inSample.headD 0
      let inputIdx := inSample.drop 1
      let coefficient :=
        if outBatch != inBatch || decide (inChannel < groupStart) ||
            decide (groupStart + inC / groups ≤ inChannel) then
          0
        else
          Spec.Conv.Internal.foldlIndices (Tensor.to kernel (List Nat)) 0 (fun acc kernelIdx =>
            if Spec.Conv.Internal.mkDilatedInputIdx? outIdx kernelIdx
                (Tensor.to stride (List Nat)) (Tensor.to dilation (List Nat))
                (Tensor.to padding (List Nat)) = some inputIdx then
              acc + getAtOrZero layer.kernel (outChannel :: inChannel :: kernelIdx)
            else
              acc)
      Tensor.scalar coefficient))

/-- Flattened broadcast of a convolution bias over spatial and leading batch coordinates. -/
def convBiasBroadcast
    {d outC : Nat} {outSpatial : TorchLean.Tensor Nat [d]}
    (bias : Tensor α [outC]) (leading : Shape := .scalar) :
    let outShape := leading.concat (Shape.ofList (outC :: Tensor.to outSpatial (List Nat)))
    Tensor α [outShape.size] :=
  let outShape := leading.concat (Shape.ofList (outC :: Tensor.to outSpatial (List Nat)))
  Tensor.dim (fun row =>
    let coordinates := Shape.Coord.toList outShape (Shape.Coord.unlinearize row)
    let outChannel := (coordinates.drop leading.rank).headD 0
    Tensor.scalar (getAtOrZero bias [outChannel]))

end NN.MLTheory.CROWN
