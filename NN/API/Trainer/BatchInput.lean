/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# Inputs for Optional Batching

The input precedes the defaulted `batch` flag in trainer methods. These two instances give
sample literals their expected type once the flag is selected, without coercing or copying data.
-/

@[expose] public section

namespace TorchLean.Trainer.Internal

/-- The selected batch flag determines the exact input type. -/
class BatchInput (single many : Type) (batch : Bool) (Input : outParam Type) : Prop where
  /-- Input types must agree; batching never converts an arbitrary input. -/
  type_eq : Input = if batch then many else single

instance {single many : Type} : BatchInput single many false single := ⟨rfl⟩

instance {single many : Type} : BatchInput single many true many := ⟨rfl⟩

end TorchLean.Trainer.Internal
