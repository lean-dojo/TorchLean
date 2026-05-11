/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Linear
public import NN.Spec.Models.LogisticRegression
public import NN.Spec.Module.SpecModule

/-!
# Logistic regression as an `NNModuleSpec`

The model spec provides a standalone gradient-descent baseline and prediction helpers.
This file adds the `NNModuleSpec` wrapper (Linear + Sigmoid) for composition and export.
-/

@[expose] public section


namespace Spec

open Tensor
open ModSpec
open Activation

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-- Logistic regression wrapped as `NNModuleSpec` (linear + sigmoid). -/
def logisticRegressionModule {p : ℕ} (model : LogisticRegression p 0 α) :
  NNModuleSpec α (.dim p .scalar) (.dim 1 .scalar) :=
  let weightMatrix : Tensor α (.dim 1 (.dim p .scalar)) :=
    Tensor.dim (fun _ => model.weights)
  let biasVec : Tensor α (.dim 1 .scalar) :=
    Tensor.dim (fun _ => Tensor.scalar model.intercept)
  let lspec : Spec.LinearSpec α p 1 :=
    { weights := weightMatrix, bias := biasVec }
  {
    forward := fun x =>
      Activation.sigmoidSpec (Spec.linearSpec (α := α) lspec x),
    kind := "LogisticRegression",
    export_func := {
      toPyTorch := s!"nn.Sequential(nn.Linear({p}, 1), nn.Sigmoid())",
      dimensions := (p, 1)
    }
  }

/-- `SpecChain` wrapper for logistic regression (useful for composition). -/
def logisticRegressionChain {p : ℕ} (model : LogisticRegression p 0 α) :
  SpecChain α (.dim p .scalar) (.dim 1 .scalar) :=
  SpecChain.single (logisticRegressionModule (α := α) model)

end Spec

