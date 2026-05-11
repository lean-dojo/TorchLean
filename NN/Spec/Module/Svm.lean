/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Models.Svm
public import NN.Spec.Module.Linear

/-!
# SVM as an `NNModuleSpec`

The SVM spec model includes a gradient-descent baseline and prediction helpers.
This file adds the `NNModuleSpec` wrapper so it can be composed/exported in the module system.

References:
- For the actual SVM objective/gradients and classic SVM citations (Cortes-Vapnik, Vapnik),
  see `NN.Spec.Models.Svm`:
  `NN/Spec/Models/Svm.lean`.
- PyTorch analogy for the forward map: a linear score `X @ w + b` is the same shape-level role as
  `torch.nn.Linear(p, 1)` (no activation).
-/

@[expose] public section


namespace Spec

open Tensor
open ModSpec

variable {α : Type} [Context α]

/-- Linear SVM wrapped as an `NNModuleSpec` (single-output linear layer). -/
def svmModule {p : ℕ} (model : SVM p 0 α) :
  NNModuleSpec α (.dim p .scalar) (.dim 1 .scalar) :=
  let weightMatrix : Tensor α (.dim 1 (.dim p .scalar)) :=
    Tensor.dim (fun _ => model.weights)
  let biasVec : Tensor α (.dim 1 .scalar) :=
    Tensor.dim (fun _ => Tensor.scalar model.bias)
  let lspec : Spec.LinearSpec α p 1 :=
    { weights := weightMatrix, bias := biasVec }
  Spec.LinearModuleSpec (α := α) lspec

/-- `SpecChain` wrapper for linear SVM. -/
def svmChain {p : ℕ} (model : SVM p 0 α) :
  SpecChain α (.dim p .scalar) (.dim 1 .scalar) :=
  SpecChain.single (svmModule (α := α) model)

end Spec
