/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Sample
public import TorchLean.NN.Arithmetic
public import TorchLean.NN.Data.SampleStream

/-!
# Trainer Datasets

`Trainer.Dataset input target` is the supervised-data interface of the trainer. Building the
samples and converting them into the selected arithmetic both wait until training begins, so one
dataset serves every executable arithmetic the trainer supports.

The materialized value is a finite `Data.SampleStream`, which produces samples by index without
storing them all. Dataset constructors live in `TorchLean.NN.Data.Dataset`.
-/

@[expose] public section

namespace TorchLean.Trainer

/-- Supervised data with statically known input and target shapes. -/
structure Dataset (input target : Shape) where
  /-- Materialize the finite sample stream after the trainer selects its runtime arithmetic. -/
  materialize :
    {α : Type} →
    [TorchLean.Storage α] →
    [Context α] →
    [FromFloat α] →
    IO (Data.SampleStream (Sample.Supervised α input target))

end TorchLean.Trainer
