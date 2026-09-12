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

/-- Decode a row-major flat index into one coordinate per dimension. -/
def decodeFlatIndex : List Nat → Nat → List Nat
  | [], _ => []
  | n :: ns, idx =>
      let stride := ns.prod
      let coordinate := if stride = 0 then 0 else (idx / stride) % n
      coordinate :: decodeFlatIndex ns (if stride = 0 then 0 else idx % stride)

/-- Interval propagation for an arbitrary-dimensional convolution. -/
def ibpConv
    {d inC outC : Nat} {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (layer : Spec.ConvSpec d inC outC kernel stride padding α)
    (xB : Box α (Shape.ofList (inC :: Tensor.to inSpatial (List Nat)))) :
    Box α (Shape.ofList (outC ::
      Tensor.to (Spec.convOutSpatial inSpatial kernel stride padding) (List Nat))) :=
  let outSpatial := Spec.convOutSpatial inSpatial kernel stride padding
  let endpoint := fun (lower : Bool) =>
    Tensor.dim (fun outChannel =>
      TorchLean.Tensor.generate (Tensor.to outSpatial (List Nat)) (fun outIdx =>
        let total :=
          (List.finRange inC).foldl (fun acc inChannel =>
            Spec.Conv.Internal.foldlIndices (Tensor.to kernel (List Nat)) acc
              (fun acc kernelIdx =>
              match Spec.Conv.Internal.mkInputIdx? outIdx kernelIdx
                  (Tensor.to stride (List Nat)) (Tensor.to padding (List Nat)) with
              | none => acc
              | some inputIdx =>
                  let lo := getAtOrZero xB.lo (inChannel.val :: inputIdx)
                  let hi := getAtOrZero xB.hi (inChannel.val :: inputIdx)
                  let weight := getAtOrZero layer.kernel
                    (outChannel.val :: inChannel.val :: kernelIdx)
                  let pLo := weight * lo
                  let pHi := weight * hi
                  let bound :=
                    if lower then
                      if pLo > pHi then pHi else pLo
                    else if pLo > pHi then pLo else pHi
                  acc + bound)) 0
        total + getAtOrZero layer.bias [outChannel.val]))
  { lo := endpoint true, hi := endpoint false }

/-- Explicit flattened linear operator for an arbitrary-dimensional convolution. -/
def convLinearMatrix
    {d inC outC : Nat} {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (layer : Spec.ConvSpec d inC outC kernel stride padding α) :
    let inShape := Shape.ofList (inC :: Tensor.to inSpatial (List Nat))
    let outShape := Shape.ofList (outC ::
      Tensor.to (Spec.convOutSpatial inSpatial kernel stride padding) (List Nat))
    Tensor α [outShape.size, inShape.size] :=
  let inDims := inC :: Tensor.to inSpatial (List Nat)
  let outSpatial := Spec.convOutSpatial inSpatial kernel stride padding
  let outDims := outC :: Tensor.to outSpatial (List Nat)
  Tensor.dim (fun row =>
    let outCoordinates := decodeFlatIndex outDims row.val
    let outChannel := outCoordinates.headD 0
    let outIdx := outCoordinates.drop 1
    Tensor.dim (fun column =>
      let inCoordinates := decodeFlatIndex inDims column.val
      let inChannel := inCoordinates.headD 0
      let inputIdx := inCoordinates.drop 1
      let coefficient :=
        Spec.Conv.Internal.foldlIndices (Tensor.to kernel (List Nat)) 0 (fun acc kernelIdx =>
          if Spec.Conv.Internal.matchesInputPos outIdx kernelIdx
              (Tensor.to stride (List Nat)) (Tensor.to padding (List Nat))
              inputIdx then
            acc + getAtOrZero layer.kernel (outChannel :: inChannel :: kernelIdx)
          else
            acc)
      Tensor.scalar coefficient))

/-- Flattened broadcast of a convolution bias over every output spatial position. -/
def convBiasBroadcast
    {d outC : Nat} {outSpatial : TorchLean.Tensor Nat [d]}
    (bias : Tensor α [outC]) :
    let outShape := Shape.ofList (outC :: Tensor.to outSpatial (List Nat))
    Tensor α [outShape.size] :=
  let outDims := outC :: Tensor.to outSpatial (List Nat)
  Tensor.dim (fun row =>
    let outChannel := (decodeFlatIndex outDims row.val).headD 0
    Tensor.scalar (getAtOrZero bias [outChannel]))

end NN.MLTheory.CROWN
