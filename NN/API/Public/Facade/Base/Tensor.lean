/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.NN
public import NN.API.Public.TensorPack
public import NN.API.Public.Seeded
public import NN.API.Public.Autograd
public import NN.API.Data
public import NN.API.Data.Transforms
public import NN.API.Runtime
public import NN.API.Models
public import NN.API.Public.NN.Transformer
public import NN.API.RL
public import NN.API.Rand
public import NN.API.Samples.Bands
public import NN.API.Text.Bpe
public import NN.MLTheory.CROWN.Flatbox
public import NN.MLTheory.CROWN.Graph
public import NN.Verification.TorchLean.Compile
public import NN.API.Public.Facade.Base.Runtime

/-!
# TorchLean Tensor Names

Ops, random, shape, tensor, and semantic names exposed by the `NN` umbrella.
-/

@[expose] public section

namespace TorchLean

namespace Ops

/-!
Low-level executable ops for verification and compiler-facing examples.

Most model code should use `nn.*` and `Trainer.*`. Use `Ops.*` when writing an explicit
TorchLean executable program directly, for example before compiling a hand-built fragment to
`NN.IR.Graph`.
-/

export NN.API.TorchLean
  (RefTy const add sub mul scale abs sqrt clamp max min
   broadcastTo reshape transpose2d reduceSum reduceMean
   gatherScalar gatherRow gatherScalarNat gatherVecNat gatherRowsNat scatterAddVec scatterAddRow
   matmul concatVectors
   maxPool2d maxPool2dPad smoothMaxPool2d avgPool2d avgPool2dPad
   relu silu gelu sigmoid tanh softmax softplus exp log inv safeLog logSoftmax
   globalAvgPool2dChw globalAvgPool2dNchw
   sum flatten linear mseLoss layerNorm batchnormChannelFirst multiHeadAttention conv2d)

end Ops

namespace rand

export NN.API.rand
  (keyOf uniform mask uniformND maskND randND
   nextSeed nextSeedGlobal nextSeedsGlobal)

end rand

namespace Shape

@[inherit_doc Spec.Shape.scalar]
abbrev scalar := Spec.Shape.scalar

@[inherit_doc Spec.Shape.size]
abbrev size := Spec.Shape.size

@[inherit_doc Spec.Shape.dim]
abbrev batch := Spec.Shape.dim

@[inherit_doc NN.Tensor.Shape.Vec]
abbrev vec := NN.Tensor.Shape.Vec

@[inherit_doc NN.Tensor.Shape.Mat]
abbrev mat := NN.Tensor.Shape.Mat

@[inherit_doc NN.Tensor.Shape.CHW]
abbrev image := NN.Tensor.Shape.CHW

@[inherit_doc NN.Tensor.Shape.NCHW]
abbrev images := NN.Tensor.Shape.NCHW

@[inherit_doc NN.Tensor.Shape.NCHW]
abbrev nchw := NN.Tensor.Shape.NCHW

/-- Number of scalar features in a CHW image shape. -/
abbrev imageSize (channels height width : Nat) : Nat :=
  Shape.size (image channels height width)

@[inherit_doc NN.Tensor.shapeOfDims]
abbrev ofDims := NN.Tensor.shapeOfDims

@[inherit_doc Spec.Shape.toList]
abbrev toList := Spec.Shape.toList

end Shape

namespace Tensor

@[inherit_doc Spec.Tensor]
abbrev T := Spec.Tensor

@[inherit_doc NN.Tensor.tensor1d]
abbrev vector (α : Type := Float) (xs : List α) :
    Tensor.T α (.dim xs.length .scalar) :=
  NN.Tensor.tensor1d (α := α) xs

@[inherit_doc Spec.fromList1d]
abbrev fromList1d {α : Type} (xs : List α) :
    Tensor.T α (.dim xs.length .scalar) :=
  Spec.fromList1d xs

@[inherit_doc Spec.fromList2d]
abbrev fromList2d {α : Type} [Inhabited α] (xss : List (List α)) :
    Option (Tensor.T α (.dim xss.length (.dim (if xss.isEmpty then 0 else xss.head!.length) .scalar))) :=
  Spec.fromList2d xss

@[inherit_doc NN.Tensor.tensorND]
abbrev ofList {α : Type} (dims : List Nat) (xs : List α) :
    Except String (Tensor.T α (Shape.ofDims dims)) :=
  NN.Tensor.tensorND (α := α) dims xs

@[inherit_doc NN.Tensor.tensorF321d]
abbrev float32Vector (xs : List Float) :
    Tensor.T TorchLean.Floats.IEEE754.IEEE32Exec (.dim xs.length .scalar) :=
  NN.Tensor.tensorF321d xs

@[inherit_doc Spec.fill]
abbrev fill {α : Type} (value : α) (s : Shape) : Tensor.T α s :=
  Spec.fill value s

@[inherit_doc Spec.mapTensor]
abbrev map {α β : Type} {s : Shape} (f : α → β) (x : Tensor.T α s) : Tensor.T β s :=
  Spec.mapTensor f x

@[inherit_doc Spec.Tensor.vecGet]
abbrev vecGet {α : Type} {n : Nat} (x : Tensor.T α (.dim n .scalar)) (i : Fin n) : α :=
  Spec.Tensor.vecGet x i

@[inherit_doc NN.Tensor.print]
abbrev print {α : Type} [NN.Tensor.DTypeName α] [NN.Tensor.TensorPrintable α]
    {s : Shape} (t : Tensor.T α s) : IO Unit :=
  NN.Tensor.print t

@[inherit_doc Spec.pretty]
abbrev pretty {α : Type} [ToString α] {s : Shape} (t : Tensor.T α s) : String :=
  Spec.pretty t

@[inherit_doc Spec.Tensor.toScalar]
abbrev toScalar {α : Type} (t : Tensor.T α .scalar) : α :=
  Spec.Tensor.toScalar t

@[inherit_doc NN.API.Common.castTensor]
abbrev castFloat {α : Type} (cast : Float → α) {s : Shape} (t : Tensor.T Float s) :
    Tensor.T α s :=
  NN.API.Common.castTensor cast t

/--
Repeat one tensor across a fixed batch axis.

Use this for classifier demos whose checked model consumes a whole batch, while the example wants to
inspect one ordinary input.
-/
def repeatBatch {α : Type} {s : Shape} (batch : Nat) (x : Tensor.T α s) :
    Tensor.T α (.dim batch s) :=
  Spec.Tensor.dim (fun _ => x)

/--
Convert a runtime tensor back to a `Float` tensor inside `IO`.

Trainer prediction handles use this so examples can train under executable IEEE32 or another scalar
backend, then inspect ordinary `Float` tensors afterward.
-/
def toFloatIO {α : Type} [Runtime.TensorScalar α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α] :
    ∀ {s : Shape}, Tensor.T α s → IO (Tensor.T Float s)
  | .scalar, .scalar x => do
      pure <| .scalar (← _root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv.toFloat
        (α := α) x)
  | .dim n s, .dim f => do
      let xs ← (List.finRange n).mapM (fun i => toFloatIO (s := s) (f i))
      match n, xs with
      | 0, _ => pure <| .dim (fun i => nomatch i.2)
      | _ + 1, [] =>
          throw <| IO.userError "Tensor.toFloatIO: internal error: nonempty dimension produced no entries"
      | _ + 1, h :: _ => pure <| .dim (fun i => xs.getD i.val h)

export NN.API.Common
  (tensorF tensorFGen tensorFGen! tensorFGenShape!)

end Tensor

namespace Semantics

@[inherit_doc NN.API.Semantics.relu]
abbrev relu {α : Type} [Zero α] [Max α] (x : α) : α :=
  NN.API.Semantics.relu x

end Semantics

/-!
## Public Namespaces

These are the names users usually type for model construction, tensor utilities, training, runtime
selection, and verification examples. The definitions below forward to the checked implementation;
the semantics are not copied or forked.
-/

end TorchLean
