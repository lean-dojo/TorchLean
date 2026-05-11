/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Dropout
public import NN.Spec.Module.SpecModule

/-!
# Dropout as `NNModuleSpec`s (deterministic spec variants)

PyTorch's dropout is stochastic during training and becomes identity during evaluation.
In the spec layer we often want a deterministic, pure meaning that can be composed into models
and used in proofs without introducing randomness.

This file wraps the deterministic dropout specs from `NN/Spec/Layers/Dropout.lean` as
  `NNModuleSpec`s
so they can be used in `SpecChain` pipelines and carry export metadata.

Two variants are provided:

- `DropoutInferenceModuleSpec p`: a deterministic "scale by keep probability" variant.
- `DropoutMaskedModuleSpec p mask`: a deterministic "training-style" dropout that takes the mask
  explicitly (useful when you want to model a particular dropout pattern).
-/

@[expose] public section


namespace Spec

open Tensor
open ModSpec

variable {α : Type} [Context α]

/-- Deterministic inference-time dropout wrapper.

This is a pure scaling op (`x ↦ (1 - p) * x`). It is a deterministic surrogate used in some spec
models rather than PyTorch's running training/eval state.
-/
def DropoutInferenceModuleSpec {s : Shape} (p : α) : NNModuleSpec α s s :=
{ forward := fun x => dropoutInferenceSpec (α := α) (s := s) p x
  kind := "DropoutInference"
  export_func := {
    -- Keep this as a plain string (no `ToString α` requirement).
    toPyTorch := "DropoutInference(p=...)"
    dimensions := (0, 0)
  } }

/-- Deterministic masked dropout wrapper (mask is captured as data).

This matches the usual training-time dropout structure with the mask made explicit instead of
sampled. The forward uses the same scaling and epsilon-protection as `dropout_masked_spec`.
-/
def DropoutMaskedModuleSpec {s : Shape} (p : α) (mask : Tensor Bool s) : NNModuleSpec α s s :=
{ forward := fun x => dropoutMaskedSpec (α := α) (s := s) p mask x
  kind := "DropoutMasked"
  export_func := {
    -- Keep this as a plain string (no `ToString α` requirement).
    toPyTorch := "DropoutMasked(p=..., mask=...)"
    dimensions := (0, 0)
  } }

end Spec
