/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Models.GradientBoostedTrees
public import NN.Spec.Module.DecisionTree
import NN.Spec.Core.Random

/-!
# Random Forest

Forests aggregate tree predictions by majority vote for classification or by an arithmetic mean
for regression. `Forest` stores symbolic decision trees; `Numeric` fits typed CART trees to tensors.

Numeric training draws a bootstrap sample of rows for each tree, with replacement. An explicit seed
makes those draws reproducible. A separate stream selects candidate features without replacement at
each split when `maxFeatures` is smaller than the input width. The default considers every feature.

These are standalone reference models with targeted executable tests. No API builder or
model-correctness theorem is provided here.
-/

public section


namespace RandomForest
open DecisionTree

/--
A forest is just an array of trees. The count is a runtime value, not a type index, because nothing
in the aggregation depends on how many trees there are.

The container leaves training and label semantics to its caller. Numeric training below has a
separate typed representation whose feature count and depth are recorded in each tree's type.
-/
structure Forest (α : Type) where
  /-- The trees, in the order they were grown. -/
  trees : Array (DecisionTree α)

/--
Predict by evaluating every tree and folding the results with a caller-supplied aggregation.

The aggregation is a parameter rather than a field of `Forest` because the same grown forest answers
a classification question with `majorityVote` and a regression question with `average`, and a spec
should not force a choice the caller has not made yet.
-/
def predict {α : Type} (forest : Forest α) (decisionFn : String → Bool)
    (aggregateFn : Array α → α) : α :=
  aggregateFn (forest.trees.map fun tree => evaluate tree decisionFn)

/--
Majority vote: the most frequent prediction, or `none` for an empty forest.

Ties go to the smallest label under `Ord`. This symbolic helper uses label order; the numeric
classification forest below instead keeps the first tied label in tree order.
-/
def majorityVote {α : Type} [Ord α] (predictions : Array α) : Option α :=
  if predictions.isEmpty then
    none
  else
    -- Count frequencies using an ordered map so results are deterministic.
    let grouped := predictions.foldl
      (fun acc pred =>
        acc.insert pred (acc.getD pred 0 + 1))
      (Std.TreeMap.empty : Std.TreeMap α Nat compare)

    -- Pick the element with highest frequency.
    grouped.foldl
      (fun best label count =>
        match best with
        | none => some (label, count)
        | some (_, bestCount) => if count > bestCount then some (label, count) else best)
      none
      |>.map (·.1)

/--
Arithmetic mean of the predictions, and `0` for an empty forest.

The empty-array result matches the convention used by `regressionForestForwardSpec`.
-/
def average {α : Type} [Zero α] [Add α] [Div α] [NatCast α] (predictions : Array α) : α :=
  if predictions.isEmpty then
    0
  else
    predictions.foldl (fun sum pred => sum + pred) 0 / (predictions.size : α)

/-!
## Numeric random forest (spec baseline)

The `Forest` above wraps the symbolic `DecisionTree` from `NN.Spec.Module.DecisionTree`, where
splits are keyed by `String` feature names and an external `decisionFn : String → Bool` decides the
branch. That is handy for examples, but it is not something we can “train” without providing
feature-value semantics.

For a more classical baseline, we also provide a *numeric* random forest built on the
`Spec.DecisionTreeSpec` representation used by `NN/Spec/Models/GradientBoostedTrees.lean`:

- features are indexed by `Nat`
- splits compare a feature value to a threshold
- regression splits minimize the sum of squared errors and classification splits minimize
  weighted Gini impurity
- each tree receives `batch` row draws from the original dataset, with replacement
- `maxFeatures` controls a fresh feature subset at each split; its default is all features

Both samplers use `Spec.Random`'s SplitMix64 generator. Row draws and feature draws have separate
streams, so changing the feature budget does not change a tree's bootstrap sample.
-/

namespace Numeric

open Spec TorchLean

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- Draw `batch` row indices with replacement for one tree.

Each draw chooses from the full original batch. Repeated indices repeat both the observation and
its target during training; rows that were not drawn are absent from that tree. `treeIndex` selects
the tree's stream, and increasing the forest size preserves the samples of its existing prefix.
`hBatch` rules out drawing from an empty dataset.
-/
def bootstrapIndices {batch : Nat} (seed treeIndex : Nat) (hBatch : batch ≠ 0) :
    Array (Fin batch) :=
  let key := Spec.Random.keyOf seed (2 * treeIndex)
  (Array.finRange batch).map fun i =>
    ⟨Spec.Random.sampleNat key i.val batch, Nat.mod_lt _ (Nat.pos_of_ne_zero hBatch)⟩

/-- Select candidate features without replacement for a single node.

A partial Fisher-Yates shuffle chooses `min maxFeatures nFeatures` distinct indices. We sort the
selected indices afterwards so equally good splits keep the lower feature index, independently of
the order in which it was drawn. A zero budget yields no candidates and therefore a leaf.

The root has `nodeIndex = 0`; its children have indices `1` and `2`, with children of node `i`
numbered `2 * i + 1` and `2 * i + 2`. Each node's draw depends only on its seed, tree and position.
-/
def sampleFeatures (seed treeIndex nodeIndex nFeatures maxFeatures : Nat) :
    Array (Fin nFeatures) :=
  let treeKey := Spec.Random.keyOf seed (2 * treeIndex + 1)
  let key := Spec.Random.keyOf treeKey.toNat nodeIndex
  let count := min maxFeatures nFeatures
  let selected := Id.run do
    let mut indices := Array.finRange nFeatures
    for i in [:count] do
      let offset := Spec.Random.sampleNat key i (nFeatures - i)
      indices := indices.swapIfInBounds i (i + offset)
    return indices.extract 0 count
  selected.qsort (fun a b => a.val < b.val)

/-- Project a row onto the sampled features before using the shared CART split search. -/
private def projectFeatures {nFeatures : Nat} (features : Array (Fin nFeatures))
    (x : Tensor α [nFeatures]) : Tensor α [features.size] :=
  Tensor.ofFn fun i => x.getScalar (features[i])

/-- Grow a regression tree with a fresh candidate-feature subset at each node.

The shared CART search scores projected rows. After choosing a split, we partition the original
rows so the children can draw from every original feature again.
-/
private def fitRegressionNode {nFeatures : Nat} (seed treeIndex maxFeatures : Nat) :
    (depth : Nat) → Nat → Array (Spec.RegressionExample (α := α) nFeatures) →
      Spec.TreeNode α nFeatures depth
  | 0, _, xs => .leaf (Spec.leafValue xs)
  | depth + 1, nodeIndex, xs =>
    let features := sampleFeatures seed treeIndex nodeIndex nFeatures maxFeatures
    let projected : Array (Spec.RegressionExample (α := α) features.size) :=
      xs.map fun ex => { x := projectFeatures features ex.x, y := ex.y }
    match Spec.bestSplit projected with
    | none => .leaf (Spec.leafValue xs)
    | some (feature, threshold, _, _, score) =>
      let parentScore := Spec.GradientBoostedTrees.Internal.sse
        (Spec.GradientBoostedTrees.Internal.targets xs)
      if Context.gtBool parentScore score then
        let originalFeature := features[feature]
        let (left, right) := Spec.GradientBoostedTrees.Internal.partitionBySplit
          originalFeature threshold xs
        .split originalFeature threshold
          (fitRegressionNode seed treeIndex maxFeatures depth (2 * nodeIndex + 1) left)
          (fitRegressionNode seed treeIndex maxFeatures depth (2 * nodeIndex + 2) right)
      else
        .leaf (Spec.leafValue xs)

/-- A regression random forest: an ensemble of regression trees averaged at inference time. -/
structure RegressionForestSpec (α : Type) (nTrees maxDepth nFeatures : Nat) where
  /-- The regression trees in the ensemble, stored as a fixed-length tensor. -/
  trees : Tensor (Spec.DecisionTreeSpec α nFeatures maxDepth) [nTrees]

/-- Forward pass: average tree predictions.

This corresponds to `RandomForestRegressor.predict` (mean over tree outputs).
-/
def regressionForestForwardSpec {nTrees maxDepth nFeatures : Nat}
  (model : RegressionForestSpec α nTrees maxDepth nFeatures)
  (x : Tensor α [nFeatures]) : Tensor α .scalar :=
  if _h0 : nTrees = 0 then
    Tensor.scalar 0
  else
    let rec go (i : Nat) (acc : α) : α :=
      if h : i < nTrees then
        let t := model.trees.getScalar ⟨i, h⟩
        let yi := Spec.decisionTreeForwardSpec (α := α) (maxDepth := maxDepth) (nFeatures :=
          nFeatures) t x
        go (i + 1) (acc + yi)
      else
        acc / (nTrees : α)
    Tensor.scalar (go 0 0)

/--
Fit a regression forest with seeded row sampling and greedy CART splits.

Each tree receives `batch` paired row/target draws with replacement. At each node, search the
observed thresholds of at most `maxFeatures` sampled features and split only when SSE decreases.
Ties keep the lower feature index, then the first observed threshold in bootstrap row order.

The default seed is `0`, and the default feature budget is the full input width. A smaller budget
samples a fresh subset at every node; `0` gives leaf-only trees. Repeating a call with the same
data, seed and scalar backend reproduces its forest.
-/
def regressionForestFitRegressionMseSpec {batch nTrees maxDepth nFeatures : Nat}
  (x : Tensor α [batch, nFeatures])
  (y : Tensor α [batch])
  (hBatch : batch ≠ 0) (seed : Nat := 0) (maxFeatures : Nat := nFeatures) :
  RegressionForestSpec α nTrees maxDepth nFeatures :=
  let trees : Tensor (Spec.DecisionTreeSpec α nFeatures maxDepth) [nTrees] :=
    Tensor.ofFn fun k =>
      let examples : Array (Spec.RegressionExample (α := α) nFeatures) :=
        (bootstrapIndices seed k.val hBatch).map fun i => { x := x.get i, y := y.getScalar i }
      let root :=
        if nFeatures ≤ maxFeatures then
          Spec.fitRegressionNode maxDepth examples
        else
          fitRegressionNode seed k.val maxFeatures maxDepth 0 examples
      { root := root }
  { trees := trees }

/-!
### Classification forest (Gini)

This mirrors the regression forest, but uses the classifier-tree type from
`NN/Spec/Models/GradientBoostedTrees.lean` so leaf values can be arbitrary labels (`β`).
-/

/-- Count how many times label `lbl` appears in a fixed-size prediction tensor. -/
private def countEq {β : Type} [TorchLean.Storage β] [DecidableEq β]
    {n : Nat} (lbl : β)
    (ys : Tensor β [n]) : Nat :=
  (Array.finRange n).foldl
    (fun acc i => if ys.getScalar i = lbl then acc + 1 else acc) 0

/-- Deterministic majority label of a fixed-size prediction tensor.

Tie-breaking: we keep the first label that attains the maximal count.
-/
private def majorityLabel {β : Type} [TorchLean.Storage β]
    [DecidableEq β] [Inhabited β] {n : Nat}
    (ys : Tensor β [n]) : β :=
  if hn : n = 0 then
    default
  else
    let first : Fin n := ⟨0, Nat.pos_of_ne_zero hn⟩
    (Array.finRange n).foldl (fun best i =>
      let label := ys.getScalar i
      if countEq label ys > countEq best ys then label else best) (ys.getScalar first)

/-- Grow a classifier using the shared Gini split search on each sampled feature subset. -/
private def fitClassificationNode {β : Type} [DecidableEq β] [Inhabited β]
    {nFeatures : Nat} (seed treeIndex maxFeatures : Nat) :
    (depth : Nat) → Nat → Array (Spec.ClassificationExample (α := α) nFeatures β) →
      Spec.ClassifierTreeNode α β nFeatures depth
  | 0, _, xs => .leaf (Spec.majorityLabel (Spec.classTargets xs))
  | depth + 1, nodeIndex, xs =>
    let features := sampleFeatures seed treeIndex nodeIndex nFeatures maxFeatures
    let projected : Array (Spec.ClassificationExample (α := α) features.size β) :=
      xs.map fun ex => { x := projectFeatures features ex.x, y := ex.y }
    match Spec.bestSplitC projected with
    | none => .leaf (Spec.majorityLabel (Spec.classTargets xs))
    | some (feature, threshold, _, _, score) =>
      let parentScore := Spec.giniWeighted (α := α) (Spec.classTargets xs)
      if Context.gtBool parentScore score then
        let originalFeature := features[feature]
        let (left, right) := Spec.partitionBySplitC originalFeature threshold xs
        .split originalFeature threshold
          (fitClassificationNode seed treeIndex maxFeatures depth (2 * nodeIndex + 1) left)
          (fitClassificationNode seed treeIndex maxFeatures depth (2 * nodeIndex + 2) right)
      else
        .leaf (Spec.majorityLabel (Spec.classTargets xs))

/-- A classification random forest: an ensemble of classifier trees (majority vote). -/
structure ClassificationForestSpec (α β : Type) (nTrees maxDepth nFeatures : Nat) where
  /-- The classifier trees in the ensemble, stored as a fixed-length tensor. -/
  trees : Tensor (Spec.DecisionTreeClassifierSpec α β nFeatures maxDepth) [nTrees]

/-- Predict by majority vote across trees, keeping the first tied label in tree order.

An empty forest returns `default`.
-/
def classificationForestPredictSpec {β : Type} [TorchLean.Storage β]
  [DecidableEq β] [Inhabited β]
  {nTrees maxDepth nFeatures : Nat}
  (model : ClassificationForestSpec α β nTrees maxDepth nFeatures)
  (x : Tensor α [nFeatures]) : β :=
  if _h0 : nTrees = 0 then
    default
  else
    let preds : Tensor β [nTrees] :=
      Tensor.ofFn (fun i =>
        let t := model.trees.getScalar i
        Spec.decisionTreeClassifyForwardSpecN (α := α) (β := β)
          (maxDepth := maxDepth) (nFeatures := nFeatures) t x)
    majorityLabel (β := β) preds

/-- Fit a classification forest with seeded bootstrap samples and Gini-based CART trees.

Sampling and feature/threshold ties follow `regressionForestFitRegressionMseSpec`. Leaf votes keep
the first tied label in that node's bootstrap row order. The default seed is `0`; all features are
candidates unless `maxFeatures` requests a smaller subset at each node.
-/
def classificationForestFitClassificationGiniSpec {β : Type}
  [TorchLean.Storage β] [DecidableEq β] [Inhabited β]
  {batch nTrees maxDepth nFeatures : Nat}
  (x : Tensor α [batch, nFeatures])
  (y : Tensor β [batch])
  (hBatch : batch ≠ 0) (seed : Nat := 0) (maxFeatures : Nat := nFeatures) :
  ClassificationForestSpec α β nTrees maxDepth nFeatures :=
  let trees : Tensor (Spec.DecisionTreeClassifierSpec α β nFeatures maxDepth) [nTrees] :=
    Tensor.ofFn fun k =>
      let examples : Array (Spec.ClassificationExample (α := α) nFeatures β) :=
        (bootstrapIndices seed k.val hBatch).map fun i => { x := x.get i, y := y.getScalar i }
      let root :=
        if nFeatures ≤ maxFeatures then
          Spec.fitClassificationNode maxDepth examples
        else
          fitClassificationNode seed k.val maxFeatures maxDepth 0 examples
      { root := root }
  { trees := trees }

end Numeric

end RandomForest
