/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Train.Core
public import NN.Data.SampleStream
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
Add two metric arrays pointwise.

Names must match, so unrelated quantities are not silently averaged.
-/
def addMetrics {a : Type} [Add a]
  (tag : String) (xs ys : Array (Metric a)) : Result (Array (Metric a)) := do
  if xs.size != ys.size then
    throw (tagError tag "metric length mismatch")
  let mut result := Array.emptyWithCapacity xs.size
  for i in [:xs.size] do
    match xs[i]?, ys[i]? with
    | some left, some right =>
        if left.name != right.name then
          throw (tagError tag s!"metric name mismatch: {left.name} vs {right.name}")
        result := result.push { name := left.name, value := left.value + right.value }
    | _, _ =>
        throw (tagError tag "metric length changed during aggregation")
  pure result

/-!
## Report sums (for weighted aggregation)
-/
/--
An accumulator for averaging `StepReport`s.

Instead of retaining every report and reducing at the end, we maintain:
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
  metricsSum : Array (Metric a)

namespace ReportSum

/-- Start an accumulator from a single-sample report. -/
def ofReport {a : Type} (r : StepReport a) : ReportSum a :=
  { count := 1, lossSum := r.loss, metricsSum := r.metrics }

/-- Combine two accumulators (failing if metric names/lengths mismatch). -/
def add {a : Type} [Add a]
  (tag : String) (acc next : ReportSum a) : Result (ReportSum a) := do
  let metrics ← addMetrics (tag := tag) acc.metricsSum next.metricsSum
  pure { count := acc.count + next.count
         lossSum := acc.lossSum + next.lossSum
         metricsSum := metrics }

/-- Convert an accumulator to a mean `StepReport`. -/
def mean {a : Type} [Div a] [NatCast a] (s : ReportSum a) : StepReport a :=
  let denom : a := (s.count : a)
  { loss := s.lossSum / denom
    metrics := s.metricsSum.map (fun m => { name := m.name, value := m.value / denom }) }

end ReportSum

/-!
## Sample-stream evaluation
-/
/--
Evaluate an array of samples and average their reports.

This is the “for sample in dataset: compute report; take mean” pattern.
-/
def evalArray {sample a : Type}
  [Add a] [Div a] [NatCast a]
  (tag : String) (xs : Array sample) (evalSample : sample -> Result (StepReport a)) :
  Result (StepReport a) := do
  match xs[0]? with
  | none => .error (tagError tag "empty dataset")
  | some x0 => do
      let r0 <- evalSample x0
      let acc0 := ReportSum.ofReport r0
      let acc <- (xs.drop 1).foldlM (init := acc0) (fun acc x => do
        let r <- evalSample x
        ReportSum.add (tag := tag) acc (ReportSum.ofReport r))
      pure (ReportSum.mean acc)

/-- Evaluate a finite sample stream in index order. -/
def evalDataset {sample a : Type}
  [Add a] [Div a] [NatCast a]
  (tag : String) (ds : TorchLean.Data.SampleStream sample)
  (evalSample : sample -> Result (StepReport a)) :
  Result (StepReport a) :=
  evalArray (tag := tag) ds.toArray evalSample

end Eval
end Train
end Autograd
end Runtime
