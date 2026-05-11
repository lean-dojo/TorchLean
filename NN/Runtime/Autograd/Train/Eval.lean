/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Train.Core
public import NN.Runtime.Autograd.Train.Dataset
public import NN.Runtime.Autograd.Train.Trainer

/-!
# Evaluation helpers

These utilities aggregate per-sample or per-batch `StepReport`s into a single
mean report. Metrics are matched by name and position.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace Train
namespace Eval

/-!
## Metric aggregation
-/
/--
Add two metric lists pointwise.

We require that names match (same metric in the same position). This keeps aggregation honest and
avoids silently averaging unrelated quantities.
-/
def addMetrics {a : Type} [Add a]
  (tag : String) (xs ys : List (Metric a)) : Result (List (Metric a)) :=
  match xs, ys with
  | [], [] => .ok []
  | m1 :: ms1, m2 :: ms2 =>
      if m1.name = m2.name then
        match addMetrics (tag := tag) ms1 ms2 with
        | .ok rest => .ok ({ name := m1.name, value := m1.value + m2.value } :: rest)
        | .error e => .error e
      else
        .error (tagError tag s!"metric name mismatch: {m1.name} vs {m2.name}")
  | _, _ =>
      .error (tagError tag "metric length mismatch")

/-- Multiply every metric value by a scalar (used for weighted batch averaging). -/
def scaleMetrics {a : Type} [Mul a] [Coe Nat a]
  (count : Nat) (metrics : List (Metric a)) : List (Metric a) :=
  metrics.map (fun m => { name := m.name, value := m.value * (count : a) })

/-!
## Report sums (for weighted aggregation)
-/
/--
An accumulator for averaging `StepReport`s.

Instead of keeping a list of all reports and reducing at the end, we maintain:
- `count`: how many samples contributed,
- `lossSum`: the sum of losses (optionally weighted by batch size),
- `metricsSum`: a pointwise sum of named metrics.

This is the same idea as computing streaming averages in a typical PyTorch evaluation loop.
-/
structure ReportSum (a : Type) where
  /-- Number of samples represented by this accumulator. -/
  count : Nat
  /-- Sum of losses, already weighted by sample count for batch reports. -/
  lossSum : a
  /-- Pointwise sum of metrics; names must stay aligned across additions. -/
  metricsSum : List (Metric a)

namespace ReportSum

/-- Start an accumulator from a single-sample report. -/
def ofReport {a : Type} (r : StepReport a) : ReportSum a :=
  { count := 1, lossSum := r.loss, metricsSum := r.metrics }

/--
Start an accumulator from a batch report, weighted by the number of samples in the batch.

This is the appropriate constructor when `evalBatch` returns *means* over the batch, but we want
the final mean to weight by the number of items in each batch.
-/
def ofBatch {a : Type} [Mul a] [Coe Nat a]
  (count : Nat) (r : StepReport a) : ReportSum a :=
  { count := count
    lossSum := r.loss * (count : a)
    metricsSum := scaleMetrics (count := count) r.metrics }

/-- Combine two accumulators (failing if metric names/lengths mismatch). -/
def add {a : Type} [Add a]
  (tag : String) (acc next : ReportSum a) : Result (ReportSum a) := do
  let metrics ← addMetrics (tag := tag) acc.metricsSum next.metricsSum
  pure { count := acc.count + next.count
         lossSum := acc.lossSum + next.lossSum
         metricsSum := metrics }

/-- Convert an accumulator to a mean `StepReport`. -/
def mean {a : Type} [Div a] [Coe Nat a] (s : ReportSum a) : StepReport a :=
  let denom : a := (s.count : a)
  { loss := s.lossSum / denom
    metrics := s.metricsSum.map (fun m => { name := m.name, value := m.value / denom }) }

end ReportSum

/-!
## Dataset evaluation
-/
/--
Evaluate a list of samples and average their reports.

This is the “for sample in dataset: compute report; take mean” pattern.
-/
def evalList {sample a : Type}
  [Add a] [Div a] [Coe Nat a]
  (tag : String) (xs : List sample) (evalSample : sample -> Result (StepReport a)) :
  Result (StepReport a) := do
  match xs with
  | [] => .error (tagError tag "empty dataset")
  | x0 :: xs => do
      let r0 <- evalSample x0
      let acc0 := ReportSum.ofReport r0
      let acc <- xs.foldlM (init := acc0) (fun acc x => do
        let r <- evalSample x
        ReportSum.add (tag := tag) acc (ReportSum.ofReport r))
      pure (ReportSum.mean acc)

/-- Evaluate a `Dataset` by converting to a list and calling `evalList`. -/
def evalDataset {sample a : Type}
  [Add a] [Div a] [Coe Nat a]
  (tag : String) (ds : Dataset sample) (evalSample : sample -> Result (StepReport a)) :
  Result (StepReport a) :=
  evalList (tag := tag) ds.toList evalSample

/--
Evaluate a list of non-empty batches and compute a weighted mean report.

Each batch contributes proportionally to its length (so small last-batches do not distort the
average).
-/
def evalBatches {sample a : Type}
  [Add a] [Mul a] [Div a] [Coe Nat a]
  (tag : String) (batches : List (List sample))
  (evalBatch : List sample -> Result (StepReport a)) :
  Result (StepReport a) := do
  match batches with
  | [] => .error (tagError tag "empty batch list")
  | b0 :: bs => do
      if b0.isEmpty then
        .error (tagError tag "empty batch")
      else
        let r0 <- evalBatch b0
        let acc0 := ReportSum.ofBatch (count := b0.length) r0
        let acc <- bs.foldlM (init := acc0) (fun acc b => do
          if b.isEmpty then
            .error (tagError tag "empty batch")
          else
            let r <- evalBatch b
            let sum := ReportSum.ofBatch (count := b.length) r
            ReportSum.add (tag := tag) acc sum)
        pure (ReportSum.mean acc)

/-- Batch a dataset and then call `evalBatches`. -/
def evalDatasetBatches {sample a : Type}
  [Add a] [Mul a] [Div a] [Coe Nat a]
  (tag : String) (batchSize : Nat) (ds : Dataset sample)
  (evalBatch : List sample -> Result (StepReport a)) :
  Result (StepReport a) := do
  let batches <- Dataset.batches (tag := tag) batchSize ds
  evalBatches (tag := tag) batches evalBatch

end Eval
end Train
end Autograd
end Runtime
