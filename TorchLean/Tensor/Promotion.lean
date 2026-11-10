/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Numeric.Quotient
public import TorchLean.Tensor.Native.Pointwise

/-!
# Element conversion and mixed-type tensor arithmetic

Every TorchLean tensor is homogeneous, with one element type and one physical
storage representation. Operations on tensors of different element types go
through element conversion (`ElementCast`) and an extensible promotion
relation (`ElementPromotion`).

The built-in promotion order is

```text
UInt8 < Nat < Int < Rat < Float32 < Float
```

Converting an exact value to an IEEE type rounds to nearest with ties to even.
A rational rounds as its exact quotient, so a numerator or denominator beyond
the float range does not overflow first. A natural reaches `Float` or `Float32`
through its `NatCast` and a rational through `QuotientArithmetic.round`.
User-defined element types join in by providing `ElementCast` and
`ElementPromotion` instances.
-/

@[expose] public section

namespace TorchLean

universe u v

/--
Element conversion from `α` to `β`, used by `Tensor.cast`.

Instances may lose information. The library's conversions to `Float32` and
`Float` round to nearest with ties to even.
-/
class ElementCast (α : Type u) (β : Type v) where
  /-- Convert one source element to the target element type. -/
  cast : α → β

/--
The common element type and two conversions selected for a mixed operation.

The result is an `outParam`, so typeclass search determines it from the
ordered input pair before looking for arithmetic or storage instances.
Library instances come in both operand orders, and custom instances should
normally do the same.
-/
class ElementPromotion (α β : Type) (γ : outParam Type) where
  /-- Convert the left element to the common type. -/
  left : α → γ
  /-- Convert the right element to the common type. -/
  right : β → γ

namespace ElementPromotion

/-- Build a promotion from two `ElementCast` conversions into one result type. -/
@[instance_reducible] def ofCasts (α β γ : Type) [leftCast : ElementCast α γ]
    [rightCast : ElementCast β γ] : ElementPromotion α β γ where
  left := leftCast.cast
  right := rightCast.cast

end ElementPromotion

/-- Every element type converts to itself. -/
instance {α : Type u} : ElementCast α α where
  cast := id

/-- A byte converts to the natural with the same value. -/
instance : ElementCast UInt8 Nat where
  cast := UInt8.toNat

/-- A byte converts to the integer with the same value. -/
instance : ElementCast UInt8 Int where
  cast value := Int.ofNat value.toNat

/-- A byte converts to the rational with the same value. -/
instance : ElementCast UInt8 Rat where
  cast value := value.toNat

/-- A byte converts exactly to `Float32`. -/
instance : ElementCast UInt8 Float32 where
  cast := UInt8.toFloat32

/-- A byte converts exactly to `Float`. -/
instance : ElementCast UInt8 Float where
  cast := UInt8.toFloat

/-- A natural converts to the same integer. -/
instance : ElementCast Nat Int where
  cast := Int.ofNat

/-- A natural converts to the same rational. -/
instance : ElementCast Nat Rat where
  cast value := value

/-- Convert a natural to a host float with its `NatCast`. -/
instance {β : Type} [Numeric.HostFloat β] [NatCast β] : ElementCast Nat β where
  cast := Nat.cast

/-- An integer converts to the same rational. -/
instance : ElementCast Int Rat where
  cast value := value

/-- An integer converts to `Float32` with `Float32.ofInt`. -/
instance : ElementCast Int Float32 where
  cast := Float32.ofInt

/-- An integer converts to `Float` with `Float.ofInt`. -/
instance : ElementCast Int Float where
  cast := Float.ofInt

/-- Convert an exact rational to a host float with `QuotientArithmetic.round`. -/
instance {β : Type} [Numeric.HostFloat β] [NatCast β] [Neg β] [Div β] : ElementCast Rat β where
  cast := Numeric.QuotientArithmetic.round

/-- `Float32` widens exactly to `Float`. -/
instance : ElementCast Float32 Float where
  cast := Float32.toFloat

/-- Explicit binary64-to-binary32 conversion using Lean's IEEE rounding. -/
instance : ElementCast Float Float32 where
  cast := Float.toFloat32

/-- Operands of one type promote to that type. -/
instance {α : Type} : ElementPromotion α α α :=
  ElementPromotion.ofCasts α α α

/-- `UInt8` and `Nat` operands promote to `Nat`. -/
instance : ElementPromotion UInt8 Nat Nat :=
  ElementPromotion.ofCasts UInt8 Nat Nat

/-- `Nat` and `UInt8` operands promote to `Nat`. -/
instance : ElementPromotion Nat UInt8 Nat :=
  ElementPromotion.ofCasts Nat UInt8 Nat

/-- `UInt8` and `Int` operands promote to `Int`. -/
instance : ElementPromotion UInt8 Int Int :=
  ElementPromotion.ofCasts UInt8 Int Int

/-- `Int` and `UInt8` operands promote to `Int`. -/
instance : ElementPromotion Int UInt8 Int :=
  ElementPromotion.ofCasts Int UInt8 Int

/-- `UInt8` and `Rat` operands promote to `Rat`. -/
instance : ElementPromotion UInt8 Rat Rat :=
  ElementPromotion.ofCasts UInt8 Rat Rat

/-- `Rat` and `UInt8` operands promote to `Rat`. -/
instance : ElementPromotion Rat UInt8 Rat :=
  ElementPromotion.ofCasts Rat UInt8 Rat

/-- `UInt8` and `Float32` operands promote to `Float32`. -/
instance : ElementPromotion UInt8 Float32 Float32 :=
  ElementPromotion.ofCasts UInt8 Float32 Float32

/-- `Float32` and `UInt8` operands promote to `Float32`. -/
instance : ElementPromotion Float32 UInt8 Float32 :=
  ElementPromotion.ofCasts Float32 UInt8 Float32

/-- `UInt8` and `Float` operands promote to `Float`. -/
instance : ElementPromotion UInt8 Float Float :=
  ElementPromotion.ofCasts UInt8 Float Float

/-- `Float` and `UInt8` operands promote to `Float`. -/
instance : ElementPromotion Float UInt8 Float :=
  ElementPromotion.ofCasts Float UInt8 Float

/-- `Nat` and `Int` operands promote to `Int`. -/
instance : ElementPromotion Nat Int Int :=
  ElementPromotion.ofCasts Nat Int Int

/-- `Int` and `Nat` operands promote to `Int`. -/
instance : ElementPromotion Int Nat Int :=
  ElementPromotion.ofCasts Int Nat Int

/-- `Nat` and `Rat` operands promote to `Rat`. -/
instance : ElementPromotion Nat Rat Rat :=
  ElementPromotion.ofCasts Nat Rat Rat

/-- `Rat` and `Nat` operands promote to `Rat`. -/
instance : ElementPromotion Rat Nat Rat :=
  ElementPromotion.ofCasts Rat Nat Rat

/-- `Nat` and `Float32` operands promote to `Float32`. -/
instance : ElementPromotion Nat Float32 Float32 :=
  ElementPromotion.ofCasts Nat Float32 Float32

/-- `Float32` and `Nat` operands promote to `Float32`. -/
instance : ElementPromotion Float32 Nat Float32 :=
  ElementPromotion.ofCasts Float32 Nat Float32

/-- `Nat` and `Float` operands promote to `Float`. -/
instance : ElementPromotion Nat Float Float :=
  ElementPromotion.ofCasts Nat Float Float

/-- `Float` and `Nat` operands promote to `Float`. -/
instance : ElementPromotion Float Nat Float :=
  ElementPromotion.ofCasts Float Nat Float

/-- `Int` and `Rat` operands promote to `Rat`. -/
instance : ElementPromotion Int Rat Rat :=
  ElementPromotion.ofCasts Int Rat Rat

/-- `Rat` and `Int` operands promote to `Rat`. -/
instance : ElementPromotion Rat Int Rat :=
  ElementPromotion.ofCasts Rat Int Rat

/-- `Int` and `Float32` operands promote to `Float32`. -/
instance : ElementPromotion Int Float32 Float32 :=
  ElementPromotion.ofCasts Int Float32 Float32

/-- `Float32` and `Int` operands promote to `Float32`. -/
instance : ElementPromotion Float32 Int Float32 :=
  ElementPromotion.ofCasts Float32 Int Float32

/-- `Int` and `Float` operands promote to `Float`. -/
instance : ElementPromotion Int Float Float :=
  ElementPromotion.ofCasts Int Float Float

/-- `Float` and `Int` operands promote to `Float`. -/
instance : ElementPromotion Float Int Float :=
  ElementPromotion.ofCasts Float Int Float

/-- `Rat` and `Float32` operands promote to `Float32`. -/
instance : ElementPromotion Rat Float32 Float32 :=
  ElementPromotion.ofCasts Rat Float32 Float32

/-- `Float32` and `Rat` operands promote to `Float32`. -/
instance : ElementPromotion Float32 Rat Float32 :=
  ElementPromotion.ofCasts Float32 Rat Float32

/-- `Rat` and `Float` operands promote to `Float`. -/
instance : ElementPromotion Rat Float Float :=
  ElementPromotion.ofCasts Rat Float Float

/-- `Float` and `Rat` operands promote to `Float`. -/
instance : ElementPromotion Float Rat Float :=
  ElementPromotion.ofCasts Float Rat Float

/-- `Float32` and `Float` operands promote to `Float`. -/
instance : ElementPromotion Float32 Float Float :=
  ElementPromotion.ofCasts Float32 Float Float

/-- `Float` and `Float32` operands promote to `Float`. -/
instance : ElementPromotion Float Float32 Float :=
  ElementPromotion.ofCasts Float Float32 Float

namespace Tensor

open Shape (Coord)

/--
Execution strategy for the pointwise tensor operation `op`.

A low-priority instance serves every promotion whose common type has a
`BinaryOp.Scalar op` instance. Packed `Float` tensors, alone or with a byte tensor on either side,
use native kernels with the same coordinate semantics.
-/
class Pointwise (op : BinaryOp) (α β : Type) (γ : outParam Type)
    [Storage α] [Storage β] [Storage γ] where
  /-- Scalar operation implemented by this tensor kernel. -/
  scalar : α → β → γ
  /-- Build the pointwise result. -/
  apply {shape : Shape} :
    Rep α shape → Rep β shape → Rep γ shape
  /-- The tensor implementation agrees with its scalar operation. -/
  apply_at {shape : Shape}
    (left : Rep α shape) (right : Rep β shape)
    (coordinate : Coord shape) :
    apply left right coordinate =
      scalar (left coordinate) (right coordinate)

attribute [simp, grind =] Pointwise.apply_at

namespace Rep

/-- Generic one-pass fallback for every supported scalar promotion. -/
instance (priority := low) {op : BinaryOp} {α β γ : Type}
    [Storage α] [Storage β]
    [promotion : ElementPromotion α β γ] [Storage γ] [BinaryOp.Scalar op γ] :
    Pointwise op α β γ where
  scalar x y := BinaryOp.Scalar.eval op (promotion.left x) (promotion.right y)
  apply left right :=
    zipWith
      (fun x y => BinaryOp.Scalar.eval op (promotion.left x) (promotion.right y))
      left right
  apply_at left right coordinate := by
    simp

/-- Packed Float operations use one exact-size native output loop. -/
instance {op : BinaryOp} : Pointwise op Float Float Float where
  scalar := Native.floatBinary op
  apply := Native.binary op
  apply_at left right coordinate := by
    exact Native.binary_apply op
      left right coordinate

/-- Packed bytes are converted to `Float` inside one native output loop with packed Floats. -/
instance {op : BinaryOp} : Pointwise op UInt8 Float Float where
  scalar x y := Native.floatBinary op x.toFloat y
  apply := Native.promoteLeft op
  apply_at left right coordinate := by
    exact Native.promoteLeft_apply op
      left right coordinate

/-- Packed Floats meet packed bytes converted to `Float` inside one native output loop. -/
instance {op : BinaryOp} : Pointwise op Float UInt8 Float where
  scalar x y := Native.floatBinary op x y.toFloat
  apply := Native.promoteRight op
  apply_at left right coordinate := by
    exact Native.promoteRight_apply op
      left right coordinate

/--
Convert every entry to the element type `target` through `ElementCast`, keeping the shape.

The `Storage target` instance chooses the physical buffer of the result, so a cast to `Float`
writes a `FloatArray` and a cast to `UInt8` writes a `ByteArray`.
-/
def _root_.TorchLean.Tensor.cast {α : Type u} [Storage α] {shape : Shape}
    (tensor : Tensor α shape) (target : Type v) [Storage target]
    [ElementCast α target] : Tensor target shape :=
  tensor.map (β := target) (ElementCast.cast (β := target))

/-- Reading a cast tensor converts the element at the same coordinate. -/
@[simp, grind =] theorem cast_apply
    {α : Type u} [Storage α] {shape : Shape}
    (tensor : Rep α shape) (target : Type v) [Storage target]
    [ElementCast α target] (coordinate : Coord shape) :
    cast tensor target coordinate = ElementCast.cast (tensor coordinate) := by
  simp [cast]

/-- Tensor addition automatically selects the common element type. -/
instance (priority := low) {α β γ : Type} [Storage α] [Storage β] [Storage γ]
    [Pointwise .add α β γ] {shape : Shape} :
    HAdd (Rep α shape) (Rep β shape) (Rep γ shape) where
  hAdd := Pointwise.apply (op := .add)

/-- Tensor subtraction automatically selects the common element type. -/
instance (priority := low) {α β γ : Type} [Storage α] [Storage β] [Storage γ]
    [Pointwise .sub α β γ] {shape : Shape} :
    HSub (Rep α shape) (Rep β shape) (Rep γ shape) where
  hSub := Pointwise.apply (op := .sub)

/-- Tensor multiplication automatically selects the common element type. -/
instance (priority := low) {α β γ : Type} [Storage α] [Storage β] [Storage γ]
    [Pointwise .mul α β γ] {shape : Shape} :
    HMul (Rep α shape) (Rep β shape) (Rep γ shape) where
  hMul := Pointwise.apply (op := .mul)

/-- Tensor division automatically selects the common element type. -/
instance (priority := low) {α β γ : Type} [Storage α] [Storage β] [Storage γ]
    [Pointwise .div α β γ] {shape : Shape} :
    HDiv (Rep α shape) (Rep β shape) (Rep γ shape) where
  hDiv := Pointwise.apply (op := .div)

end Rep
end Tensor

end TorchLean
