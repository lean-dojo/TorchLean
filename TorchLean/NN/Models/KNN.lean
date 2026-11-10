/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.LinearAlgebra.Algorithms

/-!
# k‑Nearest Neighbors (kNN) (spec model)

A small kNN baseline for classification and regression:

- a dataset is an array of `(featureTensor, label)` pairs,
- prediction is based on the `k` closest points under a chosen distance.

Neighbors are ordered by distance, a NaN distance after every other, and equal distances, or two
NaNs, by the observation's position in the dataset. All classifiers resolve vote ties by the
closest observation whose label has the maximum count. Weighted regression averages
only the exact matches when any of the selected neighbors has distance zero.

`classify` with `k = 1` is the nearest neighbor rule of Cover and Hart (Section II), coded by
hand, and a larger `k` takes the majority vote of the nearest `k` neighbors (Section III). Their
footnote 1 lets a tie for the nearest neighbor go to the most popular category among the tied
points. Ours goes by dataset position: three points at one distance, labelled `A`, `B`, `B` in
dataset order, give `A` for `k = 1`, and that modified rule gives `B`. scikit-learn's
`KNeighborsClassifier` docs warn that tied distances make the result depend on the order of the
training data, and here that order is the dataset position.

References:

- Cover and Hart (1967), "Nearest Neighbor Pattern Classification", IEEE Transactions on
  Information Theory 13(1), 21-27, doi:10.1109/TIT.1967.1053964:
  https://ieeexplore.ieee.org/document/1053964
- scikit-learn 1.9.1, the `KNeighborsClassifier` and `KNeighborsRegressor` docs, and
  `_get_weights` in `sklearn/neighbors/_base.py`.
  https://scikit-learn.org/1.9/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

PyTorch / sklearn analogies:

- The closest Python counterparts are `sklearn.neighbors.KNeighborsClassifier` and
  `sklearn.neighbors.KNeighborsRegressor`.
- PyTorch code rarely wraps kNN in an `nn.Module`, and kNN serves as a common baseline for
  "classic ML" comparisons and reference checks.

## Implementation status

The module is a standalone reference specification, with no API builder and no theorems about the
model. The guide runs a small classification example.
-/

public section


namespace TorchLean
open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]

/-! ## Model container -/

/-- A kNN model: the neighbor count `k` and the stored dataset.

kNN is a *lazy* model with no learned weights: every query reads the whole stored `dataset`.
-/
structure KNN (α : Type) [Storage α] (β : Type) (n : Nat) where
  /-- Number of neighbors to consult. -/
  k : Nat
  /-- Training data: feature vectors paired with labels/targets. -/
  dataset : Array (Tensor α [n] × β)

/-! ## Neighbor selection -/

/-!
Neighbors are ranked by the key `(Context.isNaN d, d, index)`, so a NaN distance comes after
every other distance, and equal distances, or two NaN distances, go to the earlier dataset
position. The selected neighbors depend only on the dataset and the query. That keeps evaluation
reproducible and formal reasoning about the classifier simple.
-/

/-- Select the first `k` observations under the key `(Context.isNaN d, d, index)`, retaining each
computed distance.

Weighted regression uses these same distances for its weights, so each observation's distance is
evaluated once. The dataset index orders equal distances, so the selection does not depend on the
sort's stability.
-/
private def KNN.neighborDistances {β : Type} {n : Nat}
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
        let na := Context.isNaN da
        let nb := Context.isNaN db
        (na = false ∧ nb = true) ∨ (na = nb ∧ (da < db ∨ (¬ (db < da) ∧ ia < ib))))
    (sorted.extract 0 (min knn.k sorted.size)).map (fun triple => (triple.1, triple.2.1))

/-- The `k` nearest neighbors under a caller-supplied distance function, with ties broken by
dataset order.

Notes:
- The distance value is only used for *ranking* neighbors. It does not need to satisfy metric
  axioms, but it should be consistent with "smaller means closer".
- Equal distances are ordered by dataset position, and a NaN distance ranks after every other.
-/
def KNN.nearestWithDistance (α β : Type) (n : ℕ)
  [Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  (distanceFn : Tensor α [n] → Tensor α [n] → α)
  (knn : KNN α β n) (input : Tensor α [n]) :
  Array (Tensor α [n] × β) :=
  (neighborDistances distanceFn knn input).map (·.1)

/-- The `k` nearest neighbors under the Euclidean distance `Tensor.euclideanDistance`.

PyTorch/sklearn analogy: Euclidean `L2` distance is the default in many baseline kNN examples, and
the sklearn kNN estimators default to the Minkowski metric with `p = 2`, which is the same distance.
-/
def KNN.nearest (α β : Type) (n : ℕ)
  [Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  (knn : KNN α β n) (input : Tensor α [n]) :
  Array (Tensor α [n] × β) :=
  nearestWithDistance α β n euclideanDistance knn input

/-! ## Classification -/

/-- Count label frequencies in an array (hash-map implementation). -/
private def KNN.voteCounts {β : Type} [BEq β] [Hashable β] (labels : Array β) : Std.HashMap β Nat :=
  labels.foldl (fun acc label =>
    let currentCount := acc[label]? |>.getD 0
    acc.insert label (currentCount + 1)
  ) Std.HashMap.emptyWithCapacity

/-- The label with the largest `count`, paired with that count, or `none` for an empty array.
Among labels with equal counts, the one that appears first in `labels` wins.

`labels` is in neighbor order, so traversing it makes the winner depend only on distance and
dataset order, whatever order a count map keeps its keys in. `classifyWithConfidence` uses the
returned count.
-/
private def KNN.firstMajority {β : Type} (labels : Array β) (count : β → Nat) : Option (β × Nat) :=
  labels.foldl (fun best label =>
    let votes := count label
    match best with
    | none => some (label, votes)
    | some (_, bestVotes) => if votes > bestVotes then some (label, votes) else best) none

/-- Count with a hash map and select the winner in neighbor order. -/
private def KNN.majorityWithCount {β : Type} [BEq β] [Hashable β]
    (labels : Array β) : Option (β × Nat) :=
  let counts := voteCounts labels
  firstMajority labels (fun label => counts[label]?.getD 0)

/-- Majority-vote classification under a caller-supplied distance function, with the vote tie rule
of `classify`. -/
def KNN.classifyWithDistance (α β : Type) (n : ℕ)
  [Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  [BEq β] [Hashable β] [Inhabited β]
  (distanceFn : Tensor α [n] → Tensor α [n] → α)
  (knn : KNN α β n) (input : Tensor α [n]) : β :=
  let neighbors := nearestWithDistance α β n distanceFn knn input
  let labels := neighbors.map (fun (_, label) => label)
  ((majorityWithCount labels).map (·.1)).getD default

/-- Majority vote among the neighbors.

Among labels with the same maximum count, choose the one whose closest observation comes first.
Equal-distance observations retain dataset order. The nearest observation itself need not belong
to a winning class: with ordered labels `[A, B, C, C, B]`, the winner is `B`.
An empty neighborhood, including `k = 0`, returns `default`.
-/
def KNN.classify (α β : Type) (n : ℕ)
  [Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  [BEq β] [Hashable β] [Inhabited β]
  (knn : KNN α β n) (input : Tensor α [n]) : β :=
  classifyWithDistance α β n euclideanDistance knn input

/-- Classification that counts labels in an ordered tree map.

The winner follows the neighbor-order tie rule of `classify`, with labels identified by `compare`,
and the map only supplies counts. Labels need only an `Ord β` instance, and an empty neighborhood
gives `none`.
-/
def KNN.classifyTreeMap (α β : Type) (n : ℕ)
  [Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  [Ord β]
  (knn : KNN α β n) (input : Tensor α [n]) : Option β :=
  let neighbors := nearest α β n knn input
  let labels := neighbors.map (fun (_, label) => label)
  if labels.isEmpty then none
  else
    let labelCounts := labels.foldl (fun (acc : Std.TreeMap β Nat compare) label =>
      acc.insert label (acc.getD label 0 + 1)
    ) Std.TreeMap.empty
    (firstMajority labels (fun label => labelCounts.getD label 0)).map (·.1)

/-! ## Regression -/

/-- Average of the targets in their original order, dividing each target by the count before
summing when the ordinary sum overflows.

The ordinary sum preserves cancellation and the derivatives of zero-valued targets, so it is used
whenever it is finite. When finite targets overflow it, dividing each by the count first avoids the
overflow. Targets that are infinite or NaN keep the ordinary sum's arithmetic. An empty array gives
`0`.
-/
private def KNN.meanTargets (values : Array α) : α :=
  if values.isEmpty then 0
  else
    let total := values.foldl (· + ·) 0
    -- `x - x == 0` holds for finite IEEE values and fails for infinities and NaNs. For dual
    -- scalars `==` compares primals, and the branch taken keeps the tangent components.
    if total - total == 0 || !(values.all (fun value => value - value == 0)) then
      Context.divNat total values.size
    else
      values.foldl (fun acc value => acc + Context.divNat value values.size) 0

/-- Unweighted kNN regression: the mean of the neighbor targets, or `0` for an empty
neighborhood. -/
def KNN.predict (α : Type) (n : ℕ)
  [Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  (knn : KNN α α n) (input : Tensor α [n]) : α :=
  let neighbors := nearest α α n knn input
  meanTargets (neighbors.map (fun (_, value) => value))

/-- Weighted kNN regression with inverse-distance weights `w_i = 1 / d_i`.

Neighbors are selected in the `(distance, dataset position)` order of `nearest`. If any
selected distance equals zero, the result is the arithmetic mean of the targets at distance zero,
and positive distances have no influence, however small they are. When more than `k` observations
match exactly, dataset order determines which `k` are averaged.

If finite positive distances and finite targets overflow the ordinary accumulation, every weight
is divided by the largest one, giving weights `d_min / d_i` between zero and one. If that weighted
sum overflows too, each weight is divided by the total before multiplying its target. No
target-dependent scale is introduced. An empty neighborhood or a zero total weight returns `0`.
These are the weights of scikit-learn's `weights="distance"`, whose `_get_weights` also gives
zero-distance points weight one and every other point weight zero.
-/
def KNN.predictWeighted (α : Type) (n : ℕ)
  [Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  (knn : KNN α α n) (input : Tensor α [n]) : α :=
  let neighbors := neighborDistances euclideanDistance knn input
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
        -- The first neighbor is the nearest, so its distance is `d_min`. Targets stay unscaled,
        -- which keeps the derivative information of zero primals and cancellation.
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

/-- Classify and return the winning label's fraction of the selected neighbors.

The winner uses the same tie rule as `classify`. The denominator is the actual number of neighbors,
`min k dataset.size`, so requesting more neighbors than exist does not lower the score.
An empty neighborhood returns `(default, 0)`.
-/
def KNN.classifyWithConfidence (α β : Type) (n : ℕ)
  [Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  [BEq β] [Hashable β] [Inhabited β]
  (knn : KNN α β n) (input : Tensor α [n]) : (β × α) :=
  let neighbors := nearest α β n knn input
  let labels := neighbors.map (fun (_, label) => label)
  match majorityWithCount labels with
  | none => (default, 0)
  | some (label, votes) => (label, (((votes : ℚ) / (neighbors.size : ℚ) : ℚ) : α))

end TorchLean
