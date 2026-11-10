/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Pack

/-!
# Supervised Samples

`Sample.Supervised α σ τ` is a typed input-target record, the `(input, target)` pair a PyTorch
dataset yields, with both shapes in the type. Datasets and trainers pass samples in this form,
and `Sample.Batch` is the same record with a leading batch dimension. `mapInput`, `mapTarget` and
`map` transform the tensors.

Application code works with the named `input` and `target` fields. The runtime conversion to its
heterogeneous graph-argument representation is `Sample.arguments`.
-/

@[expose] public section

namespace TorchLean.Sample

/-- A supervised input-target sample with both shapes tracked statically. -/
structure Supervised (α : Type) [TorchLean.Storage α]
    (σ τ : Shape) where
  /-- Model input. -/
  input : Tensor α σ
  /-- Expected model output. -/
  target : Tensor α τ
deriving Repr

/-- A fixed-size minibatch whose tensors share the leading dimension `n`. -/
abbrev Batch (α : Type) [TorchLean.Storage α]
    (n : Nat) (σ τ : Shape) :=
  Supervised α (Shape.dim n σ) (Shape.dim n τ)

/-- Map the input tensor, optionally changing its shape. -/
def mapInput {α : Type} [TorchLean.Storage α]
    {σ σ' τ : Shape}
    (f : Tensor α σ → Tensor α σ')
    (sample : Supervised α σ τ) :
    Supervised α σ' τ :=
  { input := f sample.input, target := sample.target }

/-- Map the target tensor, optionally changing its shape. -/
def mapTarget {α : Type} [TorchLean.Storage α]
    {σ τ τ' : Shape}
    (f : Tensor α τ → Tensor α τ')
    (sample : Supervised α σ τ) :
    Supervised α σ τ' :=
  { input := sample.input, target := f sample.target }

/-- Map both tensors, optionally changing their element type and shapes. -/
def map {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β]
    {σ τ σ' τ' : Shape}
    (mapInput : Tensor α σ → Tensor β σ')
    (mapTarget : Tensor α τ → Tensor β τ')
    (sample : Supervised α σ τ) :
    Supervised β σ' τ' :=
  { input := mapInput sample.input, target := mapTarget sample.target }

/-- The sample as the two-tensor `TensorPack` (input, then target) that graph runtimes take. -/
def arguments {α : Type} [TorchLean.Storage α]
    {σ τ : Shape}
    (sample : Supervised α σ τ) :
    TorchLean.TensorPack α [σ, τ] :=
  .pair sample.input sample.target

end TorchLean.Sample
