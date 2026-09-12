/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Dropout
public import NN.Spec.Module.Core

/-!
# Dropout as `Spec.Module`s (deterministic spec variants)

PyTorch's dropout is stochastic during training and becomes identity during evaluation.
In the spec layer we often want a deterministic, pure meaning that can be composed into models
and used in proofs without introducing randomness.

This file wraps the deterministic dropout specs from `NN/Spec/Layers/Dropout.lean` as
  `Spec.Module`s
so they can be used in `Spec.Module.Chain` pipelines and carry export metadata.

Two variants are provided:

- `Spec.Module.dropoutInference p`: evaluation-mode dropout, hence the identity map.
- `Spec.Module.dropoutMasked p mask`: a deterministic "training-style" dropout that takes the mask
  explicitly (useful when you want to model a particular dropout pattern).
-/

@[expose] public section


namespace Spec.Module

open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- Evaluation-mode dropout wrapper. The configured training probability is retained as module
metadata, while the forward map is the identity. -/
def dropoutInference {s : Shape} (p : α) : Spec.Module α s s :=
  { forward := fun x => dropoutInferenceSpec (α := α) (s := s) p x
    kind := "DropoutInference"
    -- Keep this as a plain string so the specification does not require `ToString α`.
    pythonExpr := "DropoutInference(p=...)" }

/-- Deterministic masked dropout wrapper (mask is captured as data).

This matches the usual training-time dropout structure with the mask made explicit instead of
sampled. The forward uses the same scaling and epsilon-protection as `dropoutMaskedSpec`.
-/
def dropoutMasked {s : Shape} (p : α) (mask : Tensor Bool s) : Spec.Module α s s :=
  { forward := fun x => dropoutMaskedSpec (α := α) (s := s) p mask x
    kind := "DropoutMasked"
    -- Keep this as a plain string so the specification does not require `ToString α`.
    pythonExpr := "DropoutMasked(p=..., mask=...)" }

end Spec.Module
