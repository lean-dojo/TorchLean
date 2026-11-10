/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

-- Downstream modules reach the marked imports through this module. Import shaking cannot see
-- those downstream uses, so keep the marked imports.
module -- shake: keep-downstream

public import TorchLean.Tensor.ConfiguredBinary -- shake: keep
public import TorchLean.Tensor.Complex -- shake: keep
public import TorchLean.Tensor.Conversion -- shake: keep

/-!
# Public Tensor Constructors

Constructors of `TorchLean.Tensor` values: scalar literals, one-hot vectors, validated index
tensors, flat-index generation, scans and windows. The constant constructors `full`, `zeros` and
`ones` live with the specification-layer constructors so that spec code can use them too.

`TorchLean.Tensor.ConfiguredBinary` supplies FloatLib's configured binary scalar types and
their TorchLean contexts, so the same constructors accept both native and configured precisions.
-/

@[expose] public section

namespace TorchLean

namespace Tensor

/--
Lift a natural-number literal into a rank-zero tensor.

The expected tensor type fixes both the scalar representation and the empty shape, so
`1 : Tensor α []` needs no constructor call.
-/
instance scalarOfNat {α : Type} [TorchLean.Storage α] {n : Nat} [OfNat α n] :
    OfNat (Tensor α []) n where
  ofNat := TorchLean.Tensor.scalar (OfNat.ofNat n)

/--
Lift a decimal or scientific literal into a rank-zero tensor.

Any scalar type with `OfScientific`, executable or proof-oriented, gets the same literal syntax.
-/
instance scalarOfScientific {α : Type} [TorchLean.Storage α] [OfScientific α] :
    OfScientific (Tensor α []) where
  ofScientific mantissa exponentSign decimalExponent :=
    TorchLean.Tensor.scalar <|
      OfScientific.ofScientific mantissa exponentSign decimalExponent

/-- One-hot vector of length `n`, with a single `1` at index `k`. -/
def oneHot {α : Type} [TorchLean.Storage α] [Zero α] [One α]
    (n : Nat) (k : Fin n) : Tensor α [n] :=
  TorchLean.Tensor.dim fun i => TorchLean.Tensor.scalar (if decide (i = k) then (1 : α) else 0)

/-- One-hot encode every bounded index along an appended final axis of length `n`. -/
def oneHotIndices {α : Type} [TorchLean.Storage α] [Zero α] [One α] (n : Nat) :
    {s : Shape} → Tensor (Fin n) s → Tensor α (s ++ [n])
  | .scalar, tensor => oneHot (α := α) n tensor.item
  | .dim _ _, tensor =>
      TorchLean.Tensor.dim fun i =>
        oneHotIndices (α := α) n (TorchLean.Tensor.unstack tensor i)

/-- Decide whether every tensor entry is smaller than `n`. -/
def decidableForallLt (n : Nat) :
    {s : Shape} → (x : Tensor Nat s) →
      Decidable (TorchLean.Tensor.Forall (fun k => k < n) x)
  | .scalar, _ => by
      unfold TorchLean.Tensor.Forall
      infer_instance
  | .dim _ _, tensor =>
      letI : DecidablePred
          (fun i => TorchLean.Tensor.Forall (fun k => k < n) (TorchLean.Tensor.unstack tensor i)) :=
        fun i => decidableForallLt n (TorchLean.Tensor.unstack tensor i)
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

/-- Check that every entry is below `n` and return the tensor as bounded indices, or an error
naming the valid range `[0, n)`. -/
def checkIndices (n : Nat) {s : Shape} (x : Tensor Nat s) :
    Except String (Tensor (Fin n) s) :=
  letI := decidableForallLt n x
  if h : TorchLean.Tensor.Forall (fun k => k < n) x then
    .ok (boundIndices n x h)
  else
    .error s!"tensor contains an index outside the valid range [0, {n})"

/-- Build a tensor whose entry at row-major flat index `i` is `f i`, filling the buffer from `f`
with no intermediate array. -/
def generateFlat {α : Type} [TorchLean.Storage α]
    (shape : Shape) (f : Nat → α) :
    Tensor α shape :=
  Tensor.Rep.ofFlatFn fun index => f index.val

/-- Run each slice's action once, in index order, then stack the results along a new leading
axis with `Tensor.dim`. -/
def stackLeadingM {m : Type → Type} [Monad m] {α : Type} [TorchLean.Storage α]
    {count : Nat} {shape : Shape} (components : Fin count → m (Tensor α shape)) :
    m (Tensor α (Shape.dim count shape)) := do
  let values ← Tensor.sequenceFinM components
  pure (TorchLean.Tensor.dim values)

/-- Apply an effectful scalar function to every entry in row-major order. -/
def mapM {m : Type → Type} [Monad m] {α β : Type}
    [TorchLean.Storage α] [TorchLean.Storage β] {shape : Shape}
    (f : α → m β) (tensor : Tensor α shape) : m (Tensor β shape) :=
  generateFlatM shape fun index => f (Tensor.Rep.getFlat tensor index)

/-- Effectful right-to-left scan with one output per input entry. Entry `i` is `step values[i]`
applied to entry `i + 1`, starting from `initial`, which is not itself an output.

Effects run from the last input to the first, and a failing step stops the scan as `m` dictates.
-/
def scanrM {m : Type → Type} [Monad m] {α β : Type}
    [TorchLean.Storage α] [TorchLean.Storage β] {n : Nat}
    (step : α → β → m β) (initial : β) (values : Tensor α [n]) : m (Tensor β [n]) := do
  let build : StateT β m (Tensor β [n]) :=
    generateFlatM [n] fun flat => do
      let index : Fin n := ⟨flat.val, by simpa using flat.isLt⟩
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
The `length` entries of `values` starting at `offset`, with `pad` past the end.

Callers pass plain `offset` and `length` values and never build bounded indices. The token, byte
and minibatch window helpers are built on it.
-/
def window {α : Type} [TorchLean.Storage α] {count : Nat}
    (values : Tensor α [count]) (length : Nat) (offset : Nat := 0) (pad : α) :
    Tensor α [length] :=
  generateFlat [length] fun i =>
    if h : offset + i < count then values[(⟨offset + i, h⟩ : Fin count)] else pad

end Tensor
end TorchLean
