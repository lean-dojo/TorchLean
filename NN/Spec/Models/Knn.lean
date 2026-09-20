/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Numerics

/-!
# k‑Nearest Neighbors (kNN) (spec model)

This file provides a small kNN classifier/regressor baseline:

- a dataset is an array of `(featureTensor, label)` pairs,
- prediction is based on the `k` closest points under a chosen distance.

Equal distances are ordered by the observation's position in the dataset. All classifiers resolve
vote ties by the closest observation whose label has the maximum count. Weighted regression averages
only the exact matches when any of the selected neighbors has distance zero.

References:

- Cover and Hart (1967), "Nearest Neighbor Pattern Classification":
  https://ieeexplore.ieee.org/document/1053964

PyTorch / sklearn analogies:

- In the Python ecosystem this is closest to `sklearn.neighbors.KNeighborsClassifier` /
  `sklearn.neighbors.KNeighborsRegressor`.
- kNN is not typically an `nn.Module` in PyTorch, but it is a common baseline for "classic ML"
  comparisons and reference checks.

## Implementation status

This is a standalone reference specification with targeted executable tests. No API builder or
model-correctness theorem is provided here.
-/

public section


open TorchLean

namespace Spec
open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]

/-! ## Model container -/

/-- A small kNN model container (parameters + stored dataset).

This is a *lazy* model: inference consults the stored `dataset` at query time, rather than learning
weights.
-/
structure KNN (α : Type) [TorchLean.Storage α] (β : Type) (n : Nat) where
  /-- Number of neighbors to consult. -/
  k : Nat
  /-- Training data: feature vectors paired with labels/targets. -/
  dataset : Array (Tensor α [n] × β)

/-! ## Neighbor selection -/

/-!
The key technical detail here is deterministic tie-breaking: when two points are at exactly the
same distance, we prefer the earlier point in the dataset. This makes evaluation stable and keeps
formal reasoning about the classifier simpler.
-/

/-- Select neighbors by distance and dataset order, retaining each computed distance.

Weighted regression uses these same distances for its weights, so each observation's distance is
evaluated once. The dataset index makes equal-distance selection independent of sort stability.
-/
private def nearestWithDistances {β : Type} {n : Nat}
  (distanceFn : Tensor α [n] → Tensor α [n] → α)
  (knn : KNN α β n) (input : Tensor α [n]) :
  Array ((Tensor α [n] × β) × α) :=
  if knn.dataset.isEmpty || knn.k == 0 then #[]
  else
    let withDistances := knn.dataset.mapIdx (fun idx point =>
      (point, distanceFn input point.1, idx))
    let sorted :=
      withDistances.qsort (fun a b =>
        let da := a.2.1
        let db := b.2.1
        let ia := a.2.2
        let ib := b.2.2
        da < db ∨ (¬ (db < da) ∧ ia < ib))
    (sorted.extract 0 (min knn.k sorted.size)).map (fun triple => (triple.1, triple.2.1))

/-- Find the `k` nearest neighbors under Euclidean distance.

PyTorch/sklearn analogy: Euclidean `L2` distance is the default for many baseline kNN examples.
-/
def findKNearest (α β : Type) (n : ℕ)
  [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  (knn : KNN α β n) (input : Tensor α [n]) :
  Array (Tensor α [n] × β) :=
  (nearestWithDistances euclideanDistanceSpec knn input).map (·.1)

/-- Find the `k` nearest neighbors under a user-provided distance function.

Notes:
- The distance value is only used for *ranking* neighbors. It does not need to satisfy metric
  axioms, but it should be consistent with "smaller means closer".
- We keep deterministic tie-breaking via dataset order.
-/
def findKNearestWithDistance (α β : Type) (n : ℕ)
  [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  (distanceFn : Tensor α [n] → Tensor α [n] → α)
  (knn : KNN α β n) (input : Tensor α [n]) :
  Array (Tensor α [n] × β) :=
  (nearestWithDistances distanceFn knn input).map (·.1)

/-! ## Classification -/

/-- Count label frequencies in an array (hash-map implementation). -/
private def voteCounts {β : Type} [BEq β] [Hashable β] (labels : Array β) : Std.HashMap β Nat :=
  labels.foldl (fun acc label =>
    let currentCount := acc[label]? |>.getD 0
    acc.insert label (currentCount + 1)
  ) Std.HashMap.emptyWithCapacity

/-- Select a label by total count, retaining its first occurrence when counts are equal.

`labels` is already in neighbor order. Looking up counts while traversing that array makes the
winner depend on distance and dataset order, rather than on a map's traversal order. The count
returned with the label is also used by `classifyWithConfidence`.
-/
private def firstMajority {β : Type} (labels : Array β) (count : β → Nat) : Option (β × Nat) :=
  labels.foldl (fun best label =>
    let votes := count label
    match best with
    | none => some (label, votes)
    | some (_, bestVotes) => if votes > bestVotes then some (label, votes) else best) none

/-- Count with a hash map and select the winner in neighbor order. -/
private def majorityWithCount {β : Type} [BEq β] [Hashable β]
    (labels : Array β) : Option (β × Nat) :=
  let counts := voteCounts labels
  firstMajority labels (fun label => counts[label]?.getD 0)

/-- Majority vote among the neighbors.

Among labels with the same maximum count, choose the one whose closest observation comes first.
Equal-distance observations retain dataset order. The nearest observation itself need not belong
to a winning class: with ordered labels `[A, B, C, C, B]`, the winner is `B`.
An empty neighborhood, including `k = 0`, returns `default`.
-/
def classify (α β : Type) (n : ℕ)
  [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  [BEq β] [Hashable β] [Inhabited β]
  (knn : KNN α β n) (input : Tensor α [n]) : β :=
  let neighbors := findKNearest α β n knn input
  let labels := neighbors.map (fun (_, label) => label)
  ((majorityWithCount labels).map (·.1)).getD default

/-- Classification using an ordered tree map for label counts.

This uses the same neighbor-order tie rule as `classify`, with label equality determined by
`compare`. The map supplies counts; its key order does not select the winner. Empty neighborhoods
return `none`.
-/
def classifyTreeMap (α β : Type) (n : ℕ)
  [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  [Ord β] [Inhabited β]
  (knn : KNN α β n) (input : Tensor α [n]) : Option β :=
  let neighbors := findKNearest α β n knn input
  let labels := neighbors.map (fun (_, label) => label)
  if labels.isEmpty then none
  else
    let labelCounts := labels.foldl (fun (acc : Std.TreeMap β Nat compare) label =>
      acc.insert label (acc.getD label 0 + 1)
    ) Std.TreeMap.empty
    (firstMajority labels (fun label => labelCounts.getD label 0)).map (·.1)

/-! ## Regression -/

/-- Average targets in their original order, dividing before summation if the sum overflows.

The ordinary sum preserves cancellation and the derivatives of zero-valued targets. When finite
targets overflow that sum, dividing each target by the count first avoids the intermediate
overflow. Explicit infinities and NaNs retain the ordinary sum's arithmetic.
-/
private def meanTargets (values : Array α) : α :=
  if values.isEmpty then 0
  else
    let total := values.foldl (· + ·) 0
    let count : α := values.size
    -- Self-subtraction distinguishes finite IEEE values from infinities and NaNs. For dual
    -- scalars, equality inspects the primal; this branch does not discard tangent components.
    if total - total == 0 || !(values.all (fun value => value - value == 0)) then
      total / count
    else
      values.foldl (fun acc value => acc + value / count) 0

/-- Unweighted kNN regression: average of the neighbor targets. -/
def predict (α : Type) (n : ℕ)
  [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  (knn : KNN α α n) (input : Tensor α [n]) : α :=
  let neighbors := findKNearest α α n knn input
  meanTargets (neighbors.map (fun (_, value) => value))

/-- Weighted kNN regression using inverse-distance weights.

Select neighbors using the same `(distance, dataset position)` order as `findKNearest`. If one or
more selected distances equal zero, return the arithmetic mean of just those targets. Positive
distances then have no influence, however small they are. Otherwise use weights `w_i = 1 / d_i`.
If finite positive distances and finite targets overflow the ordinary accumulation, divide all
weights by the largest weight, giving `d_min / d_i`. These weights lie between zero and one.
If their weighted sum still overflows, divide each weight by the total before multiplying by its
target. No target-dependent scale is introduced.
When more than `k` observations match exactly, dataset order determines which `k` are averaged.
An empty neighborhood or a zero total weight returns `0`.
-/
def predictWeighted (α : Type) (n : ℕ)
  [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  (knn : KNN α α n) (input : Tensor α [n]) : α :=
  let neighbors := nearestWithDistances euclideanDistanceSpec knn input
  let neighborsWithDist := neighbors.map (fun ((_, value), dist) => (value, dist))
  match neighborsWithDist[0]? with
  | none => 0
  | some (_, nearestDistance) =>
    let exactMatches := neighborsWithDist.filter (fun (_, dist) => dist == 0)
    if !exactMatches.isEmpty then
      meanTargets (exactMatches.map (·.1))
    else
      let (weightedSum, totalWeight) : α × α :=
        neighborsWithDist.foldl (fun (sum, total) (value, dist) =>
          let weight := 1 / dist
          (sum + weight * value, total + weight)) (0, 0)
      let ordinary := if totalWeight == 0 then 0 else weightedSum / totalWeight
      let finiteInputs := neighborsWithDist.all (fun (value, dist) =>
        value - value == 0 && dist - dist == 0 && Context.gtBool dist 0)
      if (weightedSum - weightedSum == 0 && totalWeight - totalWeight == 0) ||
          !finiteInputs then
        ordinary
      else
        -- The first neighbor supplies the common weight scale. Preserve the original target
        -- arithmetic so zero primals and cancellation retain their derivative information.
        let weightedTargets := neighborsWithDist.map (fun (value, dist) =>
          (value, nearestDistance / dist))
        let (sum, total) : α × α :=
          weightedTargets.foldl (fun (sum, total) (value, weight) =>
            (sum + weight * value, total + weight)) (0, 0)
        if total == 0 then 0
        else if sum - sum == 0 then sum / total
        else
          weightedTargets.foldl (fun acc (value, weight) =>
            acc + (weight / total) * value) 0

/-! ## Helpers -/

/-- Constructor helper (explicit arguments keep elaboration simple in examples). -/
def KNN.fromData (α β : Type) (n : ℕ) (k : Nat)
    [TorchLean.Storage α]
    (data : Array (Tensor α [n] × β)) : KNN α β n :=
  { k := k, dataset := data }

/-- Batch regression: map `predict` over an array of inputs. -/
def batchPredict (α : Type) (n : ℕ)
  [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  (knn : KNN α α n) (inputs : Array (Tensor α [n])) : Array α :=
  inputs.map (predict α n knn)

/-- Batch classification: map `classify` over an array of inputs. -/
def batchClassify (α β : Type) (n : ℕ)
  [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  [Hashable β] [Inhabited β] [BEq β]
  (knn : KNN α β n) (inputs : Array (Tensor α [n])) : Array β :=
  inputs.map (classify α β n knn)

/-- Classify with an explicit distance function and the same vote tie rule as `classify`. -/
def classifyWithDistance (α β : Type) (n : ℕ)
  [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  [BEq β] [Hashable β] [Inhabited β]
  (distanceFn : Tensor α [n] → Tensor α [n] → α)
  (knn : KNN α β n) (input : Tensor α [n]) : β :=
  let neighbors := findKNearestWithDistance α β n distanceFn knn input
  let labels := neighbors.map (fun (_, label) => label)
  ((majorityWithCount labels).map (·.1)).getD default

/-- Classify and return the winning label's fraction of the selected neighbors.

The winner uses the same tie rule as `classify`. The denominator is the actual number of neighbors,
`min k dataset.size`, so requesting more neighbors than exist does not lower the score.
An empty neighborhood returns `(default, 0)`.
-/
def classifyWithConfidence (α β : Type) (n : ℕ)
  [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  [BEq β] [Hashable β] [Inhabited β]
  (knn : KNN α β n) (input : Tensor α [n]) : (β × α) :=
  let neighbors := findKNearest α β n knn input
  let labels := neighbors.map (fun (_, label) => label)
  match majorityWithCount labels with
  | none => (default, 0)
  | some (label, votes) => (label, (votes : α) / (neighbors.size : α))

end Spec
