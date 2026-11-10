/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Random.Permutation
public import TorchLean.NN.Models.GradientBoostedTrees
import TorchLean.Tensor.Random

/-!
# Random Forest

Forests aggregate tree predictions by majority vote for classification or by an arithmetic mean
for regression. `Regressor.fit` and `Classifier.fit` fit typed CART trees to tensors.

Training draws a bootstrap sample of rows for each tree, with replacement. An explicit seed
makes those draws reproducible: tree `t` samples its rows under the key
`TorchLean.Random.keyOf seed (2 * t)`. A separate stream, under
`TorchLean.Random.keyOf seed (2 * t + 1)`, selects candidate features without replacement at each
split when `maxFeatures` is smaller than the input width. The default considers every feature.

We wrote both forests by hand from Breiman (2001). The vote is his Definition 1.1, where each tree
casts a unit vote, and the regression mean is his average over the trees. scikit-learn's
`RandomForestClassifier` averages the trees' class probabilities. Two trees that say class 0 with
leaf probability 0.6 and one that says class 1 with probability 1 give class 0 here and class 1
there, 0.4 against 0.6. Breiman's Forest-RI draws a small group of features at each node, so
with the default budget of every feature our forest is bagging. He grows unpruned trees to
maximum size. Ours stop at `maxDepth`, or earlier when no split lowers the impurity.

These are standalone reference models, with no API builder and no model-correctness theorem.

References:

- Breiman (2001), "Random Forests", Machine Learning 45(1), 5-32, Definition 1.1 and Forest-RI.
  https://doi.org/10.1023/A:1010933404324
- scikit-learn 1.9.1 user guide, Section 1.11.2.1 (Random Forests).
  https://scikit-learn.org/1.9/modules/ensemble.html
-/

public section


namespace TorchLean.RandomForest

variable {α : Type} [Storage α] [Context α]

/-- Draw `batch` row indices with replacement for one tree.

Each draw chooses from the full original batch. A repeated index repeats both the observation and
its target during training, and undrawn rows are absent from that tree. `treeIndex` selects the
tree's stream, so growing the forest keeps the samples of the trees it already has. `hBatch` rules
out drawing from an empty dataset.
-/
def bootstrapIndices {batch : Nat} (seed treeIndex : Nat) (hBatch : batch ≠ 0) :
    Array (Fin batch) :=
  let key := Random.keyOf seed (2 * treeIndex)
  (Array.finRange batch).map fun i =>
    ⟨Random.sampleNat key i.val batch, Nat.mod_lt _ (Nat.pos_of_ne_zero hBatch)⟩

/-- Select candidate features without replacement for a single node.

A partial Fisher-Yates shuffle chooses `min maxFeatures nFeatures` distinct indices. The selected
indices are then sorted, so equally good splits keep the lower feature index whatever the draw
order. A zero budget yields no candidates and therefore a leaf.

Nodes are numbered heap-style: the root has `nodeIndex = 0`, and the children of node `i` are
`2 * i + 1` and `2 * i + 2`. Each node's draw depends only on its seed, tree and position.
-/
def sampleFeatures (seed treeIndex nodeIndex nFeatures maxFeatures : Nat) :
    Array (Fin nFeatures) :=
  let treeKey := Random.keyOf seed (2 * treeIndex + 1)
  let key := Random.keyOf treeKey.toNat nodeIndex
  (Random.permutation key nFeatures maxFeatures).qsort (fun a b => a.val < b.val)

/-- A regression random forest: an ensemble of regression trees averaged at inference time. -/
structure Regressor (α : Type) (nTrees maxDepth nFeatures : Nat) where
  /-- The regression trees in the ensemble, stored as a fixed-length tensor. -/
  trees : Tensor (DecisionTree α nFeatures maxDepth) [nTrees]

/-- Forward pass: the mean of the tree predictions, as in scikit-learn's
`RandomForestRegressor.predict`. An empty forest predicts `0`. -/
def Regressor.forward {nTrees maxDepth nFeatures : Nat}
  (model : Regressor α nTrees maxDepth nFeatures)
  (x : Tensor α [nFeatures]) : Tensor α .scalar :=
  if _h0 : nTrees = 0 then
    Tensor.scalar 0
  else
    let rec go (i : Nat) (acc : α) : α :=
      if h : i < nTrees then
        let t := model.trees.getScalar ⟨i, h⟩
        let yi := DecisionTree.forward (α := α) (maxDepth := maxDepth) (nFeatures :=
          nFeatures) t x
        go (i + 1) (acc + yi)
      else
        Context.divNat acc nTrees
    Tensor.scalar (go 0 0)

/--
Fit a regression forest with seeded row sampling and greedy CART splits.

Each tree receives `batch` paired row/target draws with replacement. At each node, the search
covers the observed thresholds of at most `maxFeatures` sampled features and splits only when the
SSE decreases. Ties keep the lower feature index, then the first observed threshold in bootstrap
row order.

The default seed is `0`, and the default feature budget is the full input width. A smaller budget
samples a fresh subset at every node, and a budget of `0` gives leaf-only trees. Repeating a call
with the same data, seed and scalar backend reproduces its forest.
-/
def Regressor.fit {batch nTrees maxDepth nFeatures : Nat}
  (x : Tensor α [batch, nFeatures])
  (y : Tensor α [batch])
  (hBatch : batch ≠ 0) (seed : Nat := 0) (maxFeatures : Nat := nFeatures) :
  Regressor α nTrees maxDepth nFeatures :=
  let trees : Tensor (DecisionTree α nFeatures maxDepth) [nTrees] :=
    Tensor.ofFn fun k =>
      let examples : Array (DecisionTree.RegressionExample (α := α) nFeatures) :=
        (bootstrapIndices seed k.val hBatch).map fun i => { x := x.unstack i, y := y.getScalar i }
      let root :=
        if nFeatures ≤ maxFeatures then
          DecisionTree.fitRegressionNode maxDepth examples
        else
          DecisionTree.fitRegressionNode maxDepth examples
            (sampleFeatures seed k.val · nFeatures maxFeatures)
      { root := root }
  { trees := trees }

/-!
### Classification forest (Gini)

The classification forest mirrors the regression forest with the classifier-tree type of
`TorchLean/NN/Models/GradientBoostedTrees.lean`, whose leaves can hold arbitrary labels of type
`β`.
-/

/-- A classification random forest: an ensemble of classifier trees (majority vote). -/
structure Classifier (α β : Type) (nTrees maxDepth nFeatures : Nat) where
  /-- The classifier trees in the ensemble, stored as a fixed-length tensor. -/
  trees : Tensor (DecisionTree.Classifier α β nFeatures maxDepth) [nTrees]

/-- Predict by majority vote across trees, keeping the first tied label in tree order.

An empty forest returns `default`.
-/
def Classifier.predict {β : Type} [Storage β]
  [DecidableEq β] [Inhabited β]
  {nTrees maxDepth nFeatures : Nat}
  (model : Classifier α β nTrees maxDepth nFeatures)
  (x : Tensor α [nFeatures]) : β :=
  if _h0 : nTrees = 0 then
    default
  else
    let preds : Tensor β [nTrees] :=
      Tensor.ofFn (fun i =>
        let t := model.trees.getScalar i
        DecisionTree.Classifier.forward (α := α) (β := β)
          (maxDepth := maxDepth) (nFeatures := nFeatures) t x)
    DecisionTree.Classifier.majorityLabel (Array.ofFn preds.getScalar)

/-- Fit a classification forest with seeded bootstrap samples and Gini-based CART trees.

Sampling, defaults and feature/threshold tie-breaking follow `Regressor.fit`. Leaf
votes keep the first tied label in the node's bootstrap row order.
-/
def Classifier.fit {β : Type}
  [Storage β] [DecidableEq β] [Inhabited β]
  {batch nTrees maxDepth nFeatures : Nat}
  (x : Tensor α [batch, nFeatures])
  (y : Tensor β [batch])
  (hBatch : batch ≠ 0) (seed : Nat := 0) (maxFeatures : Nat := nFeatures) :
  Classifier α β nTrees maxDepth nFeatures :=
  let trees : Tensor (DecisionTree.Classifier α β nFeatures maxDepth) [nTrees] :=
    Tensor.ofFn fun k =>
      let examples : Array (DecisionTree.Classifier.Example (α := α) nFeatures β) :=
        (bootstrapIndices seed k.val hBatch).map fun i => { x := x.unstack i, y := y.getScalar i }
      let root :=
        if nFeatures ≤ maxFeatures then
          DecisionTree.Classifier.fitNode maxDepth examples
        else
          DecisionTree.Classifier.fitNode maxDepth examples
            (sampleFeatures seed k.val · nFeatures maxFeatures)
      { root := root }
  { trees := trees }

end TorchLean.RandomForest
