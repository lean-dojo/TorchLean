/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Sample
public import NN.API.Arithmetic
public import NN.Data.SampleStream

/-!
# Trainer Datasets

`Trainer.Dataset input target` is the public supervised-data interface. It delays both
sample construction and conversion into the selected arithmetic representation until training
begins. Consequently the same dataset can be used with every executable arithmetic implementation
supported by the trainer.

The materialized value is a finite `Data.SampleStream`: samples are requested by index and need not
be stored eagerly. Dataset constructors live in `NN.API.Data.Training`.
-/

@[expose] public section

namespace TorchLean.Trainer

/-- Supervised data with statically known input and target shapes. -/
structure Dataset (input target : Spec.Shape) where
  /-- Materialize the finite sample stream after the trainer selects its runtime arithmetic. -/
  materialize :
    {α : Type} →
    [TorchLean.Storage α] →
    [Context α] →
    [Runtime.FromFloat α] →
    IO (Data.SampleStream (Sample.Supervised α input target))

end TorchLean.Trainer
