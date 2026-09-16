/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Elab.Native.Pointwise

/-!
# Element conversion and mixed-type tensor arithmetic

TorchLean tensors remain homogeneous: every individual tensor has one element
type and one corresponding physical storage representation. This module
provides explicit element conversion and an extensible promotion relation for
operations whose inputs have different element types.

The built-in promotion order is

```text
UInt8 < Nat < Int < Rat < Float32 < Float
```

Moving from exact values to an IEEE type rounds to nearest with ties to even.
Rational conversion rounds the exact quotient, avoiding intermediate numerator or
denominator overflow. User-defined element types can participate by providing
`ElementCast` and `ElementPromotion` instances.
-/

@[expose] public section

namespace TorchLean

universe u v

/--
An explicit element conversion used by `Tensor.cast`.

Instances need not be lossless. In particular, conversions to `Float32` or
`Float` use Lean's ordinary IEEE rounding behavior.
-/
class ElementCast (α : Type u) (β : Type v) where
  /-- Convert one source element to the target element type. -/
  cast : α → β

/--
The common element type and two conversions selected for a mixed operation.

The result is an `outParam`, so typeclass search determines it from the
ordered input pair before looking for arithmetic or storage instances.
Library instances are symmetric; custom instances should normally provide
both operand orders.
-/
class ElementPromotion (α β : Type) (γ : outParam Type) where
  /-- Convert the left element to the common type. -/
  left : α → γ
  /-- Convert the right element to the common type. -/
  right : β → γ

namespace ElementCast

/-- Apply the registered element conversion. -/
@[inline] def apply {α : Type u} {β : Type v} [conversion : ElementCast α β]
    (value : α) : β :=
  conversion.cast value

end ElementCast

namespace ElementPromotion

/-- Build a promotion from two explicit conversions to one result type. -/
@[instance_reducible] def ofCasts (α β γ : Type) [leftCast : ElementCast α γ]
    [rightCast : ElementCast β γ] : ElementPromotion α β γ where
  left := leftCast.cast
  right := rightCast.cast

end ElementPromotion

/-- Every element type converts to itself. -/
instance {α : Type u} : ElementCast α α where
  cast := id

instance : ElementCast UInt8 Nat where
  cast := UInt8.toNat

instance : ElementCast UInt8 Int where
  cast value := Int.ofNat value.toNat

instance : ElementCast UInt8 Rat where
  cast value := value.toNat

instance : ElementCast UInt8 Float32 where
  cast := UInt8.toFloat32

instance : ElementCast UInt8 Float where
  cast := UInt8.toFloat

instance : ElementCast Nat Int where
  cast := Int.ofNat

instance : ElementCast Nat Rat where
  cast value := value

instance : ElementCast Nat Float32 where
  cast := Float32.ofNat

instance : ElementCast Nat Float where
  cast := Float.ofNat

instance : ElementCast Int Rat where
  cast value := value

instance : ElementCast Int Float32 where
  cast := Float32.ofInt

instance : ElementCast Int Float where
  cast := Float.ofInt

/-- Round an exact rational through Lean's quotient and residual-bit model. -/
def Rat.roundToFormat (format : Float.Model.Format) (value : Rat) :
    Float.Model.UnpackedFloat :=
  if value.num = 0 then
    .zero .positive
  else
    let (mantissa, exponent, accuracy) :=
      Float.Model.UnpackedFloat.divCore format value.num.natAbs 0 value.den 0
    Float.Model.UnpackedFloat.roundWithAccuracy format
      (if value.num < 0 then .negative else .positive) mantissa exponent accuracy

/-- Convert an exact rational to binary32, rounding once to nearest with ties to even.

Small exactly representable integers use native division. Larger values use Lean's float model
before packing, so a finite ratio near one does not become `∞ / ∞`. -/
def Rat.toFloat32 (value : Rat) : Float32 :=
  if value.num.natAbs ≤ 2^24 && value.den ≤ 2^24 then
    Float32.ofInt value.num / Float32.ofNat value.den
  else
    Float32.ofModel (Float32.Model.pack (Rat.roundToFormat .binary32 value))

/-- Convert an exact rational to binary64, rounding once to nearest with ties to even. -/
def Rat.toFloat (value : Rat) : Float :=
  if value.num.natAbs ≤ 2^53 && value.den ≤ 2^53 then
    Float.ofInt value.num / Float.ofNat value.den
  else
    Float.ofModel (Float.Model.pack (Rat.roundToFormat .binary64 value))

instance : ElementCast Rat Float32 where
  cast := Rat.toFloat32

instance : ElementCast Rat Float where
  cast := Rat.toFloat

instance : ElementCast Float32 Float where
  cast := Float32.toFloat

/-- Explicit binary64-to-binary32 conversion using Lean's IEEE rounding. -/
instance : ElementCast Float Float32 where
  cast := Float.toFloat32

instance {α : Type} : ElementPromotion α α α :=
  ElementPromotion.ofCasts α α α

instance : ElementPromotion UInt8 Nat Nat :=
  ElementPromotion.ofCasts UInt8 Nat Nat

instance : ElementPromotion Nat UInt8 Nat :=
  ElementPromotion.ofCasts Nat UInt8 Nat

instance : ElementPromotion UInt8 Int Int :=
  ElementPromotion.ofCasts UInt8 Int Int

instance : ElementPromotion Int UInt8 Int :=
  ElementPromotion.ofCasts Int UInt8 Int

instance : ElementPromotion UInt8 Rat Rat :=
  ElementPromotion.ofCasts UInt8 Rat Rat

instance : ElementPromotion Rat UInt8 Rat :=
  ElementPromotion.ofCasts Rat UInt8 Rat

instance : ElementPromotion UInt8 Float32 Float32 :=
  ElementPromotion.ofCasts UInt8 Float32 Float32

instance : ElementPromotion Float32 UInt8 Float32 :=
  ElementPromotion.ofCasts Float32 UInt8 Float32

instance : ElementPromotion UInt8 Float Float :=
  ElementPromotion.ofCasts UInt8 Float Float

instance : ElementPromotion Float UInt8 Float :=
  ElementPromotion.ofCasts Float UInt8 Float

instance : ElementPromotion Nat Int Int :=
  ElementPromotion.ofCasts Nat Int Int

instance : ElementPromotion Int Nat Int :=
  ElementPromotion.ofCasts Int Nat Int

instance : ElementPromotion Nat Rat Rat :=
  ElementPromotion.ofCasts Nat Rat Rat

instance : ElementPromotion Rat Nat Rat :=
  ElementPromotion.ofCasts Rat Nat Rat

instance : ElementPromotion Nat Float32 Float32 :=
  ElementPromotion.ofCasts Nat Float32 Float32

instance : ElementPromotion Float32 Nat Float32 :=
  ElementPromotion.ofCasts Float32 Nat Float32

instance : ElementPromotion Nat Float Float :=
  ElementPromotion.ofCasts Nat Float Float

instance : ElementPromotion Float Nat Float :=
  ElementPromotion.ofCasts Float Nat Float

instance : ElementPromotion Int Rat Rat :=
  ElementPromotion.ofCasts Int Rat Rat

instance : ElementPromotion Rat Int Rat :=
  ElementPromotion.ofCasts Rat Int Rat

instance : ElementPromotion Int Float32 Float32 :=
  ElementPromotion.ofCasts Int Float32 Float32

instance : ElementPromotion Float32 Int Float32 :=
  ElementPromotion.ofCasts Float32 Int Float32

instance : ElementPromotion Int Float Float :=
  ElementPromotion.ofCasts Int Float Float

instance : ElementPromotion Float Int Float :=
  ElementPromotion.ofCasts Float Int Float

instance : ElementPromotion Rat Float32 Float32 :=
  ElementPromotion.ofCasts Rat Float32 Float32

instance : ElementPromotion Float32 Rat Float32 :=
  ElementPromotion.ofCasts Float32 Rat Float32

instance : ElementPromotion Rat Float Float :=
  ElementPromotion.ofCasts Rat Float Float

instance : ElementPromotion Float Rat Float :=
  ElementPromotion.ofCasts Float Rat Float

instance : ElementPromotion Float32 Float Float :=
  ElementPromotion.ofCasts Float32 Float Float

instance : ElementPromotion Float Float32 Float :=
  ElementPromotion.ofCasts Float Float32 Float

namespace Tensor.Internal

/--
Execution strategy for pointwise tensor addition.

The low-priority instance preserves the fully generic mixed-type operation.
Packed element types can provide a higher-priority implementation while proving
the same coordinate semantics.
-/
class PointwiseAdd (α β : Type) (γ : outParam Type)
    [Storage α] [Storage β] [Storage γ] where
  /-- Scalar operation implemented by this tensor kernel. -/
  scalar : α → β → γ
  /-- Build the pointwise sum. -/
  apply {shape : Shape} :
    Rep α shape → Rep β shape → Rep γ shape
  /-- The tensor implementation agrees with its scalar operation. -/
  apply_at {shape : Shape}
    (left : Rep α shape) (right : Rep β shape)
    (coordinate : Coord shape) :
    apply left right coordinate =
      scalar (left coordinate) (right coordinate)

namespace PointwiseAdd

/-- Use the selected pointwise addition strategy. -/
@[inline] def run {α β γ : Type}
    [Storage α] [Storage β] [Storage γ]
    [operation : PointwiseAdd α β γ]
    {shape : Shape} (left : Rep α shape) (right : Rep β shape) :
    Rep γ shape :=
  operation.apply left right

end PointwiseAdd

/-- Execution strategy for pointwise tensor subtraction. -/
class PointwiseSub (α β : Type) (γ : outParam Type)
    [Storage α] [Storage β] [Storage γ] where
  /-- Scalar operation implemented by this tensor kernel. -/
  scalar : α → β → γ
  /-- Build the pointwise difference. -/
  apply {shape : Shape} :
    Rep α shape → Rep β shape → Rep γ shape
  /-- The tensor implementation agrees with its scalar operation. -/
  apply_at {shape : Shape}
    (left : Rep α shape) (right : Rep β shape)
    (coordinate : Coord shape) :
    apply left right coordinate =
      scalar (left coordinate) (right coordinate)

namespace PointwiseSub

/-- Use the selected pointwise subtraction strategy. -/
@[inline] def run {α β γ : Type}
    [Storage α] [Storage β] [Storage γ]
    [operation : PointwiseSub α β γ]
    {shape : Shape} (left : Rep α shape) (right : Rep β shape) :
    Rep γ shape :=
  operation.apply left right

end PointwiseSub

/-- Execution strategy for pointwise tensor multiplication. -/
class PointwiseMul (α β : Type) (γ : outParam Type)
    [Storage α] [Storage β] [Storage γ] where
  /-- Scalar operation implemented by this tensor kernel. -/
  scalar : α → β → γ
  /-- Build the pointwise product. -/
  apply {shape : Shape} :
    Rep α shape → Rep β shape → Rep γ shape
  /-- The tensor implementation agrees with its scalar operation. -/
  apply_at {shape : Shape}
    (left : Rep α shape) (right : Rep β shape)
    (coordinate : Coord shape) :
    apply left right coordinate =
      scalar (left coordinate) (right coordinate)

namespace PointwiseMul

/-- Use the selected pointwise multiplication strategy. -/
@[inline] def run {α β γ : Type}
    [Storage α] [Storage β] [Storage γ]
    [operation : PointwiseMul α β γ]
    {shape : Shape} (left : Rep α shape) (right : Rep β shape) :
    Rep γ shape :=
  operation.apply left right

end PointwiseMul

/-- Execution strategy for pointwise tensor division. -/
class PointwiseDiv (α β : Type) (γ : outParam Type)
    [Storage α] [Storage β] [Storage γ] where
  /-- Scalar operation implemented by this tensor kernel. -/
  scalar : α → β → γ
  /-- Build the pointwise quotient. -/
  apply {shape : Shape} :
    Rep α shape → Rep β shape → Rep γ shape
  /-- The tensor implementation agrees with its scalar operation. -/
  apply_at {shape : Shape}
    (left : Rep α shape) (right : Rep β shape)
    (coordinate : Coord shape) :
    apply left right coordinate =
      scalar (left coordinate) (right coordinate)

namespace PointwiseDiv

/-- Use the selected pointwise division strategy. -/
@[inline] def run {α β γ : Type}
    [Storage α] [Storage β] [Storage γ]
    [operation : PointwiseDiv α β γ]
    {shape : Shape} (left : Rep α shape) (right : Rep β shape) :
    Rep γ shape :=
  operation.apply left right

end PointwiseDiv

namespace Rep

/-- Generic one-pass fallback for every supported scalar promotion. -/
instance (priority := low) {α β γ : Type}
    [Storage α] [Storage β]
    [promotion : ElementPromotion α β γ] [Storage γ] [Add γ] :
    PointwiseAdd α β γ where
  scalar x y := promotion.left x + promotion.right y
  apply left right :=
    zipWith
      (fun x y => promotion.left x + promotion.right y)
      left right
  apply_at left right coordinate := by
    simp

/-- Packed Float addition uses one exact-size native output loop. -/
instance : PointwiseAdd Float Float Float where
  scalar := (· + ·)
  apply := Tensor.Internal.Elab.Impl.nativeFloatAdd
  apply_at left right coordinate := by
    exact Tensor.Internal.Elab.Impl.nativeFloatAdd_apply
      left right coordinate

/-- Packed byte-to-float promotion is fused with packed Float addition. -/
instance : PointwiseAdd UInt8 Float Float where
  scalar x y := x.toFloat + y
  apply := Tensor.Internal.Elab.Impl.nativeUInt8FloatAdd
  apply_at left right coordinate := by
    exact Tensor.Internal.Elab.Impl.nativeUInt8FloatAdd_apply
      left right coordinate

/-- Packed Float addition is fused with packed byte-to-float promotion. -/
instance : PointwiseAdd Float UInt8 Float where
  scalar x y := x + y.toFloat
  apply := Tensor.Internal.Elab.Impl.nativeFloatUInt8Add
  apply_at left right coordinate := by
    exact Tensor.Internal.Elab.Impl.nativeFloatUInt8Add_apply
      left right coordinate

/-- Generic one-pass fallback for every supported subtraction promotion. -/
instance (priority := low) {α β γ : Type}
    [Storage α] [Storage β]
    [promotion : ElementPromotion α β γ] [Storage γ] [Sub γ] :
    PointwiseSub α β γ where
  scalar x y := promotion.left x - promotion.right y
  apply left right :=
    zipWith
      (fun x y => promotion.left x - promotion.right y)
      left right
  apply_at left right coordinate := by
    simp

/-- Packed Float subtraction uses one exact-size native output loop. -/
instance : PointwiseSub Float Float Float where
  scalar := (· - ·)
  apply := Tensor.Internal.Elab.Impl.nativeFloatSub
  apply_at left right coordinate := by
    exact Tensor.Internal.Elab.Impl.nativeFloatSub_apply
      left right coordinate

/-- Generic one-pass fallback for every supported multiplication promotion. -/
instance (priority := low) {α β γ : Type}
    [Storage α] [Storage β]
    [promotion : ElementPromotion α β γ] [Storage γ] [Mul γ] :
    PointwiseMul α β γ where
  scalar x y := promotion.left x * promotion.right y
  apply left right :=
    zipWith
      (fun x y => promotion.left x * promotion.right y)
      left right
  apply_at left right coordinate := by
    simp

/-- Packed Float multiplication uses one exact-size native output loop. -/
instance : PointwiseMul Float Float Float where
  scalar := (· * ·)
  apply := Tensor.Internal.Elab.Impl.nativeFloatMul
  apply_at left right coordinate := by
    exact Tensor.Internal.Elab.Impl.nativeFloatMul_apply
      left right coordinate

/-- Generic one-pass fallback for every supported division promotion. -/
instance (priority := low) {α β γ : Type}
    [Storage α] [Storage β]
    [promotion : ElementPromotion α β γ] [Storage γ] [Div γ] :
    PointwiseDiv α β γ where
  scalar x y := promotion.left x / promotion.right y
  apply left right :=
    zipWith
      (fun x y => promotion.left x / promotion.right y)
      left right
  apply_at left right coordinate := by
    simp

/-- Packed Float division uses one exact-size native output loop. -/
instance : PointwiseDiv Float Float Float where
  scalar := (· / ·)
  apply := Tensor.Internal.Elab.Impl.nativeFloatDiv
  apply_at left right coordinate := by
    exact Tensor.Internal.Elab.Impl.nativeFloatDiv_apply
      left right coordinate

/--
Convert every element while preserving the tensor's static shape.

The target `Storage` instance selects the target physical buffer, so a
cast to `Float` writes a `FloatArray` and a cast to `UInt8` writes a
`ByteArray`.
-/
def cast {α : Type u} [Storage α] {shape : Shape}
    (tensor : Rep α shape) (target : Type v) [Storage target]
    [ElementCast α target] : Rep target shape :=
  tensor.map (β := target) (ElementCast.apply (β := target))

/-- Reading a cast tensor converts the element at the same coordinate. -/
@[simp, grind =] theorem cast_apply
    {α : Type u} [Storage α] {shape : Shape}
    (tensor : Rep α shape) (target : Type v) [Storage target]
    [ElementCast α target] (coordinate : Coord shape) :
    tensor.cast target coordinate = ElementCast.apply (tensor coordinate) := by
  simp [cast]

/--
Add two tensors pointwise, promoting their element types when necessary.

Both conversions and the addition occur in one output-building pass.
-/
def add {α β γ : Type} [Storage α] [Storage β] [Storage γ]
    [PointwiseAdd α β γ] {shape : Shape}
    (left : Rep α shape) (right : Rep β shape) :
    Rep γ shape :=
  PointwiseAdd.run left right

/-- Coordinate semantics of pointwise tensor addition. -/
@[simp, grind =] theorem add_apply
    {α β γ : Type} [Storage α] [Storage β] [Storage γ]
    [operation : PointwiseAdd α β γ] {shape : Shape}
    (left : Rep α shape) (right : Rep β shape)
    (coordinate : Coord shape) :
    add left right coordinate =
      operation.scalar (left coordinate) (right coordinate) := by
  exact operation.apply_at left right coordinate

/-- Tensor addition automatically selects the common element type. -/
instance {α β γ : Type} [Storage α] [Storage β] [Storage γ]
    [PointwiseAdd α β γ]
    {shape : Shape} :
    HAdd (Rep α shape) (Rep β shape) (Rep γ shape) where
  hAdd := add

/--
Subtract two tensors pointwise, promoting their element types when necessary.

Both conversions and the subtraction occur in one output-building pass.
-/
def sub {α β γ : Type} [Storage α] [Storage β] [Storage γ]
    [PointwiseSub α β γ] {shape : Shape}
    (left : Rep α shape) (right : Rep β shape) :
    Rep γ shape :=
  PointwiseSub.run left right

/-- Coordinate semantics of pointwise tensor subtraction. -/
@[simp, grind =] theorem sub_apply
    {α β γ : Type} [Storage α] [Storage β] [Storage γ]
    [operation : PointwiseSub α β γ] {shape : Shape}
    (left : Rep α shape) (right : Rep β shape)
    (coordinate : Coord shape) :
    sub left right coordinate =
      operation.scalar (left coordinate) (right coordinate) := by
  exact operation.apply_at left right coordinate

/-- Tensor subtraction automatically selects the common element type. -/
instance {α β γ : Type} [Storage α] [Storage β] [Storage γ]
    [PointwiseSub α β γ] {shape : Shape} :
    HSub (Rep α shape) (Rep β shape) (Rep γ shape) where
  hSub := sub

/--
Multiply two tensors pointwise, promoting their element types when necessary.

Both conversions and the multiplication occur in one output-building pass.
-/
def mul {α β γ : Type} [Storage α] [Storage β] [Storage γ]
    [PointwiseMul α β γ] {shape : Shape}
    (left : Rep α shape) (right : Rep β shape) :
    Rep γ shape :=
  PointwiseMul.run left right

/-- Coordinate semantics of pointwise tensor multiplication. -/
@[simp, grind =] theorem mul_apply
    {α β γ : Type} [Storage α] [Storage β] [Storage γ]
    [operation : PointwiseMul α β γ] {shape : Shape}
    (left : Rep α shape) (right : Rep β shape)
    (coordinate : Coord shape) :
    mul left right coordinate =
      operation.scalar (left coordinate) (right coordinate) := by
  exact operation.apply_at left right coordinate

/-- Tensor multiplication automatically selects the common element type. -/
instance {α β γ : Type} [Storage α] [Storage β] [Storage γ]
    [PointwiseMul α β γ] {shape : Shape} :
    HMul (Rep α shape) (Rep β shape) (Rep γ shape) where
  hMul := mul

/--
Divide two tensors pointwise, promoting their element types when necessary.

Both conversions and the division occur in one output-building pass.
-/
def div {α β γ : Type} [Storage α] [Storage β] [Storage γ]
    [PointwiseDiv α β γ] {shape : Shape}
    (left : Rep α shape) (right : Rep β shape) :
    Rep γ shape :=
  PointwiseDiv.run left right

/-- Coordinate semantics of pointwise tensor division. -/
@[simp, grind =] theorem div_apply
    {α β γ : Type} [Storage α] [Storage β] [Storage γ]
    [operation : PointwiseDiv α β γ] {shape : Shape}
    (left : Rep α shape) (right : Rep β shape)
    (coordinate : Coord shape) :
    div left right coordinate =
      operation.scalar (left coordinate) (right coordinate) := by
  exact operation.apply_at left right coordinate

/-- Tensor division automatically selects the common element type. -/
instance {α β γ : Type} [Storage α] [Storage β] [Storage γ]
    [PointwiseDiv α β γ] {shape : Shape} :
    HDiv (Rep α shape) (Rep β shape) (Rep γ shape) where
  hDiv := div

end Rep
end Tensor.Internal

end TorchLean
