/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Layers.Conv
public import TorchLean.Tensor.Conversion
public import TorchLean.Verify.CROWN
public import TorchLean.Tensor.ShapeChange

/-!
# Convolution Bounds

CROWN-IBP bounds and an exact flattened affine map for arbitrary-dimensional convolution, grouped
and over a leading batch shape. `ibpConv` propagates a box, rounding outward through `BoundOps`. The
affine map flattens the input and the output, so it reuses `TorchLean.Linear` and gives no spatial
axis a special meaning. `convLinearMatrix` materializes the convolution as a matrix with a row per
output position and a column per input position, and `convBiasBroadcast` spreads the bias over the
flattened output. The verifier stays deterministic at the tensor sizes the CROWN operator layer
targets.
-/

@[expose] public section

open TorchLean.Shape.Coord (unlinearize)
open TorchLean.Shape (size)

namespace TorchLean.CROWN

open Tensor
open Shape (Coord)

variable {α : Type} [Storage α] [Context α]

/-- Flatten a `Box` to a rank-one box by flattening both endpoints. -/
def flattenBox {s : Shape} (B : Box α s) : Box α (.dim (Shape.size s) .scalar) :=
  { lo := Tensor.reshape B.lo [_] (Shape.size_vector _).symm,
    hi := Tensor.reshape B.hi [_] (Shape.size_vector _).symm }

/-- Interval propagation for grouped convolution with independent leading batches.

Products, accumulation and the final bias addition round outwards. Each row follows the
channel/kernel fold order of the convolution specification. Every weight gets the rule of
`IBP.linear`, the smaller of its two directed products with the endpoints at the lower end and the
larger at the upper end, and `CROWN.ibpConv_sound` proves the box at a lawful scalar.
-/
def ibpConv [BoundOps α]
    {d inC outC : Nat} {kernel stride padding inSpatial : Tensor Nat [d]}
    (layer : Conv d inC outC kernel stride padding α)
    (dilation paddingAfter : Tensor Nat [d]) (groups : Nat)
    (leading : Shape)
    (xB : Box α (leading ++ (inC :: Tensor.to inSpatial (List Nat)))) :
    Box α (leading ++ (outC ::
      Tensor.to (Conv.outSpatialDilated inSpatial kernel stride dilation padding paddingAfter)
        (List Nat))) :=
  let outSpatial :=
    Conv.outSpatialDilated inSpatial kernel stride dilation padding paddingAfter
  let row := fun (input : Box α (inC :: Tensor.to inSpatial (List Nat))) =>
    let endpoint := fun (lower : Bool) =>
      Tensor.dim (fun outChannel =>
        Tensor.generate (Tensor.to outSpatial (List Nat)) (fun outIdx =>
          let total :=
            (List.finRange (inC / groups)).foldl (fun acc localChannel =>
              let inChannel := (outChannel.val / (outC / groups)) * (inC / groups) +
                localChannel.val
              Conv.foldlIndices (Tensor.to kernel (List Nat)) acc
                (fun acc kernelIdx =>
                  match Conv.mkDilatedInputIdx? outIdx kernelIdx
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
                        if lower then Context.minimum pLo pHi else Context.maximum pLo pHi
                      if lower then BoundOps.addDown acc bound
                      else BoundOps.addUp acc bound)) 0
          if lower then BoundOps.addDown total (getAtOrZero layer.bias [outChannel.val])
          else BoundOps.addUp total (getAtOrZero layer.bias [outChannel.val])))
    { lo := endpoint true, hi := endpoint false : Box α
        (outC :: Tensor.to outSpatial (List Nat)) }
  let rec mapRows : (batch : Shape) →
      Box α (batch ++ (inC :: Tensor.to inSpatial (List Nat))) →
      Box α (batch ++ (outC :: Tensor.to outSpatial (List Nat)))
    | .scalar, input => row input
    | .dim n rest, input =>
        let rows := fun i : Fin n =>
          mapRows rest ⟨input.lo.unstack i, input.hi.unstack i⟩
        ⟨Tensor.dim fun i => (rows i).lo, Tensor.dim fun i => (rows i).hi⟩
  mapRows leading xB

/-- The coefficient of a convolution between one spatial output position and one input position.

With positive dilation at most one kernel coordinate reaches the input position, and its weight is
read as is, so the coefficient keeps the weight's value even off a floating-point grid. When some
dilation is zero, several kernel coordinates can reach the same input position, and the coefficient
is the sum of their weights.
-/
def convKernelCoefficient (kernel outIdx inputIdx stride dilation padding : List Nat)
    (weight : List Nat → α) : α :=
  if dilation.all (fun d => decide (0 < d)) then
    let shape : Shape := kernel
    match (List.finRange shape.size).find? (fun k =>
        decide (Conv.mkDilatedInputIdx? outIdx
          (Shape.Coord.toList shape (Coord.unlinearize k)) stride dilation padding =
            some inputIdx)) with
    | none => 0
    | some k => weight (Shape.Coord.toList shape (Coord.unlinearize k))
  else
    Conv.foldlIndices kernel 0 (fun acc kernelIdx =>
      if Conv.mkDilatedInputIdx? outIdx kernelIdx stride dilation padding =
          some inputIdx then
        acc + weight kernelIdx
      else acc)

/-- The matrix of grouped convolution on flattened tensors, with a row per output position and a
column per input position. It is block diagonal over the leading batch shape and zero outside the
input channels of the output channel's group. -/
def convLinearMatrix
    {d inC outC : Nat} {kernel stride padding inSpatial : Tensor Nat [d]}
    (layer : Conv d inC outC kernel stride padding α)
    (dilation paddingAfter : Tensor Nat [d]) (groups : Nat) (leading : Shape) :
    let inShape := leading ++ (inC :: Tensor.to inSpatial (List Nat))
    let outShape := leading ++ (outC ::
      Tensor.to (Conv.outSpatialDilated inSpatial kernel stride dilation padding paddingAfter)
        (List Nat))
    Tensor α [outShape.size, inShape.size] :=
  let inShape := leading ++ (inC :: Tensor.to inSpatial (List Nat))
  let outSpatial :=
    Conv.outSpatialDilated inSpatial kernel stride dilation padding paddingAfter
  let outShape := leading ++ (outC :: Tensor.to outSpatial (List Nat))
  Tensor.dim (fun row =>
    let outCoordinates := Shape.Coord.toList outShape (Coord.unlinearize row)
    let outBatch := outCoordinates.take leading.rank
    let outSample := outCoordinates.drop leading.rank
    let outChannel := outSample.headD 0
    let outIdx := outSample.drop 1
    let groupStart := (outChannel / (outC / groups)) * (inC / groups)
    Tensor.dim (fun column =>
      let inCoordinates := Shape.Coord.toList inShape (Coord.unlinearize column)
      let inBatch := inCoordinates.take leading.rank
      let inSample := inCoordinates.drop leading.rank
      let inChannel := inSample.headD 0
      let inputIdx := inSample.drop 1
      let coefficient :=
        if outBatch != inBatch || decide (inChannel < groupStart) ||
            decide (groupStart + inC / groups ≤ inChannel) then
          0
        else
          convKernelCoefficient (Tensor.to kernel (List Nat)) outIdx inputIdx
            (Tensor.to stride (List Nat)) (Tensor.to dilation (List Nat))
            (Tensor.to padding (List Nat))
            (fun kernelIdx =>
              getAtOrZero layer.kernel (outChannel :: inChannel :: kernelIdx))
      Tensor.scalar coefficient))

/-- Flattened broadcast of a convolution bias over spatial and leading batch coordinates. -/
def convBiasBroadcast
    {d outC : Nat} {outSpatial : Tensor Nat [d]}
    (bias : Tensor α [outC]) (leading : Shape) :
    let outShape := leading ++ (outC :: Tensor.to outSpatial (List Nat))
    Tensor α [outShape.size] :=
  let outShape := leading ++ (outC :: Tensor.to outSpatial (List Nat))
  Tensor.dim (fun row =>
    let coordinates := Shape.Coord.toList outShape (Coord.unlinearize row)
    let outChannel := (coordinates.drop leading.rank).headD 0
    Tensor.scalar (getAtOrZero bias [outChannel]))

end TorchLean.CROWN
