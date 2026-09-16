/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

-- This module supplies public namespace exports used by downstream consumers. Import shaking
-- cannot see those downstream lookups, so keep the marked imports.
module -- shake: keep-downstream

public import NN.Spec.Core.FloatInstances -- shake: keep
public import NN.Spec.Core.Complex -- shake: keep
public import NN.Tensor.Conversion -- shake: keep

/-!
# Public Tensor Constructors

This module owns the `TorchLean.Tensor` construction surface: literals, flat-index generation,
and bounded-index constructors. The constant constructors `full`, `zeros`, and `ones` live with the
specification-layer constructors so that spec code can use them too. Import `NN.Tensor` for the
complete public tensor API.

`NN.Spec.Core.FloatInstances` supplies FloatLib's configured binary scalar types and their
TorchLean contexts, so the same constructors accept both native and configured precisions.
-/

@[expose] public section

namespace TorchLean

export Spec (Shape)

namespace Shape

export Spec.Shape
  (AxisInBounds EndsWith appendDim appendDim_appendDim_eq_concat appendDim_eq_concat axisSize concat
   concat_appendDim concat_assoc getDim insertAxis insertAxis_zero ofArray ofList
   prependDim pretty rank rank_prependDim replaceAxis replaceLast size size_appendDim size_concat
   size_prependDim toList)

end Shape

namespace Tensor

/--
Lift an integer literal into a rank-zero tensor.

The expected tensor type fixes both the scalar representation and the empty
shape, so users can write `1 : Tensor α []` without an explicit constructor.
-/
instance scalarOfNat {α : Type} [TorchLean.Storage α] {n : Nat} [OfNat α n] :
    OfNat (Tensor α (Shape.ofList [])) n where
  ofNat := TorchLean.Tensor.scalar (OfNat.ofNat n)

/--
Lift a decimal or scientific literal into a rank-zero tensor.

This is representation-polymorphic: any executable or proof-oriented scalar
type with `OfScientific` receives the same ordinary literal syntax.
-/
instance scalarOfScientific {α : Type} [TorchLean.Storage α] [OfScientific α] :
    OfScientific (Tensor α (Shape.ofList [])) where
  ofScientific mantissa exponentSign decimalExponent :=
    TorchLean.Tensor.scalar <|
      OfScientific.ofScientific mantissa exponentSign decimalExponent

/-- One-hot vector of length `n`, with a single `1` at index `k`. -/
def oneHot {α : Type} [TorchLean.Storage α] [Zero α] [One α]
    (n : Nat) (k : Fin n) : Tensor α [n] :=
  TorchLean.Tensor.dim fun i => TorchLean.Tensor.scalar (if decide (i = k) then (1 : α) else 0)

/-- One-hot encode every bounded index along a new final axis. -/
def oneHotIndices {α : Type} [TorchLean.Storage α] [Zero α] [One α] (n : Nat) :
    {s : Shape} → Tensor (Fin n) s → Tensor α (s.appendDim n)
  | .scalar, tensor => oneHot (α := α) n tensor.item
  | .dim _ _, tensor =>
      TorchLean.Tensor.dim fun i =>
        oneHotIndices (α := α) n (TorchLean.Tensor.unstack tensor i)

namespace IndexValidation

/-- Decide whether every tensor entry is smaller than `n`. -/
def indicesInRangeDecidable (n : Nat) :
    {s : Shape} → (x : Tensor Nat s) →
      Decidable (TorchLean.Tensor.Forall (fun k => k < n) x)
  | .scalar, _ => by
      unfold TorchLean.Tensor.Forall
      infer_instance
  | .dim _ _, tensor =>
      letI : DecidablePred
          (fun i => TorchLean.Tensor.Forall (fun k => k < n) (TorchLean.Tensor.unstack tensor i)) :=
        fun i => indicesInRangeDecidable n (TorchLean.Tensor.unstack tensor i)
      by
        unfold TorchLean.Tensor.Forall
        exact Fintype.decidableForallFintype

/-- Attach a proved scalar bound to every entry of a tensor. -/
def boundIndices (n : Nat) :
    {s : Shape} → (x : Tensor Nat s) →
      TorchLean.Tensor.Forall (fun k => k < n) x → Tensor (Fin n) s
  | .scalar, tensor, h => TorchLean.Tensor.scalar ⟨tensor.item, h⟩
  | .dim _ _, tensor, h =>
      TorchLean.Tensor.dim fun i =>
        boundIndices n (TorchLean.Tensor.unstack tensor i) (h i)

end IndexValidation

/-- Validate every natural-number entry and return a tensor of bounded indices. -/
def checkIndices (n : Nat) {s : Shape} (x : Tensor Nat s) :
    Except String (Tensor (Fin n) s) :=
  letI := IndexValidation.indicesInRangeDecidable n x
  if h : TorchLean.Tensor.Forall (fun k => k < n) x then
    .ok (IndexValidation.boundIndices n x h)
  else
    .error s!"tensor contains an index outside the valid range [0, {n})"

/-- Generate tensor entries from contiguous row-major flat indices.

The buffer is filled directly from the index function; no intermediate array is built. -/
def generateFlat {α : Type} [TorchLean.Storage α]
    (shape : Shape) (f : Nat → α) :
    Tensor α shape :=
  TorchLean.Tensor.Internal.Rep.ofFlatFn fun index => f index.val

/-- Construct a tensor by running one action per row-major coordinate, in increasing flat order.

The result retains the requested shape. A failed action stops construction according to the
supplied monad.
-/
def generateFlatM {m : Type → Type} [Monad m] {α : Type} [TorchLean.Storage α]
    (shape : Shape) (f : Fin shape.size → m α) : m (Tensor α shape) :=
  TorchLean.Tensor.Internal.Rep.ofFlatFnM fun index =>
    f (index.cast (Spec.Shape.internalSize_eq shape))

/-- Run one action per leading slice and stack its result, preserving the common trailing shape. -/
def stackLeadingM {m : Type → Type} [Monad m] {α : Type} [TorchLean.Storage α]
    {count : Nat} {shape : Shape} (components : Fin count → m (Tensor α shape)) :
    m (Tensor α (shape.prependDim count)) :=
  TorchLean.Tensor.Internal.Rep.stackM components

/-- Apply an effectful scalar operation in row-major order, retaining the tensor shape. -/
def mapM {m : Type → Type} [Monad m] {α β : Type}
    [TorchLean.Storage α] [TorchLean.Storage β] {shape : Shape}
    (f : α → m β) (tensor : Tensor α shape) : m (Tensor β shape) :=
  generateFlatM shape fun index => f (TorchLean.Tensor.Internal.Rep.getFlat tensor
    (index.cast (Spec.Shape.internalSize_eq shape).symm))

/-- Left-to-right scan with one output per input entry, excluding the initial accumulator. -/
def scanl {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β] {n : Nat}
    (step : β → α → β) (initial : β) (values : Tensor α [n]) : Tensor β [n] :=
  let build : StateM β (Tensor β [n]) :=
    generateFlatM [n] fun flat => do
      let index : Fin n := ⟨flat.val, by simpa [Spec.Shape.size] using flat.isLt⟩
      let next := step (← MonadState.get) values[index]
      MonadState.set next
      pure next
  (build.run initial).1

/-- Effectful right-to-left scan, with one output per input entry.

The terminal accumulator is excluded. Effects run from the last input to the first; an error
stops the scan according to the supplied monad.
-/
def scanrM {m : Type → Type} [Monad m] {α β : Type}
    [TorchLean.Storage α] [TorchLean.Storage β] {n : Nat}
    (step : α → β → m β) (initial : β) (values : Tensor α [n]) : m (Tensor β [n]) := do
  let build : StateT β m (Tensor β [n]) :=
    generateFlatM [n] fun flat => do
      let index : Fin n := ⟨flat.val, by simpa [Spec.Shape.size] using flat.isLt⟩
      let next ← liftM (step values[index.rev] (← MonadState.get))
      MonadState.set next
      pure next
  let (reversed, _) ← build.run initial
  pure (TorchLean.Tensor.ofFn fun index => reversed[index.rev])

/-- Right-to-left scan with one output per input entry, excluding the terminal accumulator. -/
def scanr {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β] {n : Nat}
    (step : α → β → β) (initial : β) (values : Tensor α [n]) : Tensor β [n] :=
  Id.run (scanrM (fun value accumulator => pure (step value accumulator)) initial values)

/--
Take a fixed-width tensor window, padding entries past the end.

`offset` and `length` determine the result without exposing bounded-index construction. This is the
general in-memory constructor used by token, byte, and minibatch window helpers.
-/
def window {α : Type} [TorchLean.Storage α] {count : Nat}
    (values : Tensor α [count]) (length : Nat) (offset : Nat := 0) (pad : α) :
    Tensor α [length] :=
  generateFlat [length] fun i =>
    if h : offset + i < count then values[(⟨offset + i, h⟩ : Fin count)] else pad

end Tensor
end TorchLean
