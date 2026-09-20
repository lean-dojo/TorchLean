/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Arguments

/-!
# Supervised Samples

Typed input-target records used by datasets and trainers.

Application code works with the named `input` and `target` fields. The runtime conversion to its
heterogeneous graph-argument representation is confined to `Sample.Internal`.
-/

@[expose] public section

namespace TorchLean.Sample

/-- A supervised input-target sample with both shapes tracked statically. -/
structure Supervised (α : Type) [TorchLean.Storage α]
    (σ τ : Spec.Shape) where
  /-- Model input. -/
  input : Tensor α σ
  /-- Expected model output. -/
  target : Tensor α τ
deriving Repr

/-- A fixed-size minibatch whose tensors share the leading dimension `n`. -/
abbrev Batch (α : Type) [TorchLean.Storage α]
    (n : Nat) (σ τ : Spec.Shape) :=
  Supervised α (σ.prependDim n) (τ.prependDim n)

/-- Map the input tensor, optionally changing its shape. -/
def mapInput {α : Type} [TorchLean.Storage α]
    {σ σ' τ : Spec.Shape}
    (f : Tensor α σ → Tensor α σ')
    (sample : Supervised α σ τ) :
    Supervised α σ' τ :=
  { input := f sample.input, target := sample.target }

/-- Map the target tensor, optionally changing its shape. -/
def mapTarget {α : Type} [TorchLean.Storage α]
    {σ τ τ' : Spec.Shape}
    (f : Tensor α τ → Tensor α τ')
    (sample : Supervised α σ τ) :
    Supervised α σ τ' :=
  { input := sample.input, target := f sample.target }

/-- Map both tensors, optionally changing their element type and shapes. -/
def map {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β]
    {σ τ σ' τ' : Spec.Shape}
    (mapInput : Tensor α σ → Tensor β σ')
    (mapTarget : Tensor α τ → Tensor β τ')
    (sample : Supervised α σ τ) :
    Supervised β σ' τ' :=
  { input := mapInput sample.input, target := mapTarget sample.target }

namespace Internal

/-- Convert a supervised record to the generic argument representation used by graph runtimes. -/
def arguments {α : Type} [TorchLean.Storage α]
    {σ τ : Spec.Shape}
    (sample : Supervised α σ τ) :
    TorchLean.Arguments α [σ, τ] :=
  TorchLean.Arguments.empty
    |>.push sample.input
    |>.push sample.target

end Internal

end TorchLean.Sample
