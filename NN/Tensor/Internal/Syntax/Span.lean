/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

/-!
# Source locations

TorchLean.Tensor.Internal records source locations as half-open ranges measured in Unicode
scalar values. A span stores an offset and a length, so malformed ranges
cannot be represented.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal.Syntax

universe u v

/-- A half-open source range, measured in Unicode scalar values. -/
structure Span where
  /-- Number of Unicode scalar values preceding the range. -/
  offset : Nat
  /-- Number of Unicode scalar values contained in the range. -/
  length : Nat
deriving Repr, DecidableEq, BEq

namespace Span

/-- The first offset after a source span. -/
def stop (span : Span) : Nat :=
  span.offset + span.length

/-- The empty span at `offset`. -/
def point (offset : Nat) : Span :=
  ⟨offset, 0⟩

/-- The half-open range from `start` to `stop`. -/
def between (start stop : Nat) : Span :=
  ⟨start, stop - start⟩

/-- The smallest span containing two spans that occur in source order. -/
def join (left right : Span) : Span :=
  between left.offset right.stop

/-- The endpoint of an explicitly constructed span is offset plus length. -/
@[simp] theorem stop_mk (offset length : Nat) :
    (Span.mk offset length).stop = offset + length := rfl

/-- A point span ends at its starting offset. -/
@[simp] theorem stop_point (offset : Nat) : (point offset).stop = offset := by
  simp [point, stop]

/-- The start of a span never lies after its endpoint. -/
theorem offset_le_stop (span : Span) : span.offset ≤ span.stop := by
  simp [stop]

end Span

/-- A value together with its location in source text. -/
structure Located (α : Type u) where
  /-- Parsed value. -/
  value : α
  /-- Source range from which the value was parsed. -/
  span : Span
deriving Repr, DecidableEq

namespace Located

/-- Apply a function without changing a value's source location. -/
def map {α : Type u} {β : Type v} (f : α → β) (value : Located α) : Located β :=
  ⟨f value.value, value.span⟩

/-- Mapping a located value applies the function to its payload. -/
@[simp] theorem map_value {α : Type u} {β : Type v} (f : α → β) (value : Located α) :
    (value.map f).value = f value.value := rfl

/-- Mapping a located value preserves its source span. -/
@[simp] theorem map_span {α : Type u} {β : Type v} (f : α → β) (value : Located α) :
    (value.map f).span = value.span := rfl

end Located

end TorchLean.Tensor.Internal.Syntax
