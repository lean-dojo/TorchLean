/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# Inputs for Optional Batching

Training and loss functions take their input before the defaulted `batch` flag. The `BatchInput`
instances, one per flag value, fix the input type once the flag is known, so a sample literal gets
its expected type with no coercion or copy.
-/

@[expose] public section

namespace TorchLean

/-- The `batch` flag determines the input type: `single` when it is `false`, `many` when `true`. -/
class BatchInput (single many : Type) (batch : Bool) (Input : outParam Type) : Prop where
  /-- The input type is the one the flag selects. Callers transport the input along this equation,
  so batching never converts an arbitrary input. -/
  type_eq : Input = if batch then many else single

instance {single many : Type} : BatchInput single many false single := ⟨rfl⟩

instance {single many : Type} : BatchInput single many true many := ⟨rfl⟩

end TorchLean
