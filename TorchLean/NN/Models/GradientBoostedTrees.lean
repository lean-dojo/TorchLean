/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.ConcatSlice
public import TorchLean.Tensor.Sequence
public import TorchLean.Tensor.Reduce

/-!
# Gradient boosted trees (spec model)

A reference specification of gradient boosting with decision trees: typed regression and
classification trees, greedy CART fitting and an ensemble that grows by one tree per step. The
code is plain recursion and folds that are easy to reason about, and it is not tuned for speed.

References (classical):
- CART: Breiman, Friedman, Olshen, Stone, "Classification and Regression Trees", Wadsworth,
  Belmont, CA, 1984.
- Gradient boosting: Friedman, "Greedy Function Approximation: A Gradient Boosting Machine",
  Annals of Statistics 29(5), 1189-1232, 2001. https://doi.org/10.1214/aos/1013203451
- XGBoost: Chen and Guestrin, "XGBoost: A Scalable Tree Boosting System", Proc. 22nd ACM SIGKDD
  (KDD 2016), 785-794, split gain Eq. (7) in arXiv:1603.02754v3.
  https://doi.org/10.1145/2939672.2939785
- LightGBM: Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree", Advances
  in Neural Information Processing Systems 30, 2017, Section 2.1 and Algorithm 1 (histogram
  split finding).
- scikit-learn 1.9.1 docs, `GradientBoostingRegressor` and `DecisionTreeClassifier`.
  https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html

Squared-error boosting fits trees to `target - prediction`, the negative gradient of the shared
loss `Tensor.meanSquaredError` (`TorchLean.Tensor.meanSquaredErrorBackward`) up to the factor
`2 / batch`.

`GradientBoostedTrees.trainStepFit` is the least-squares case of Friedman (2001): fit a regression
tree to the residuals, then add it times `lr`, the factor scikit-learn's `GradientBoostingRegressor`
calls `learning_rate` and defaults to 0.1. The trees are CART, typed in by hand like the loop. Our
base prediction is whatever the caller stores in `initialPrediction`, and storing the target mean
gives scikit-learn's default `init` for squared error. We cite XGBoost and LightGBM for
comparison: these trees use neither second-order split gains nor histogram bins.

## Implementation status

No API builder implements this model, and no theorem is proved about it.
`TorchLean/NN/Models/RandomForest.lean` reuses its decision trees.
-/

@[expose] public section


namespace TorchLean

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

/-!
## Tree representation

A decision tree is a small inductive type:

- `leaf value` stores the prediction for that leaf
- `split feature threshold left right` branches on a single feature

Trees of this type are easy to evaluate and to fit with the greedy CART-style procedure below.

The scalar interface `Context α` provides a decidable `>` (`Context.decidableGT`) but no decidable
`<` on every backend, so a split uses the rule

`goRight := (x_feature > threshold)`

and goes left otherwise. For a total order that sends $x_{\mathrm{feature}}\leq\mathrm{threshold}$
left. Under IEEE 754 a NaN feature value goes left as well, because `NaN > threshold` is false.
-/

/--
A regression-tree node whose index `depth` bounds the number of splits on any path below it.

`leaf value` predicts `value`. `split feature threshold left right` sends an input right when
`x_feature > threshold` and left otherwise, and its children have an index one lower.
-/
inductive DecisionTree.Node (α : Type) (nFeatures : Nat) : Nat → Type where
  | leaf {depth : Nat} (value : α) : Node α nFeatures depth
  | split {depth : Nat} (feature : Fin nFeatures) (threshold : α)
      (left right : Node α nFeatures depth) : Node α nFeatures (depth + 1)
deriving Inhabited

/--
Decision-tree specification whose type records its feature count and maximum split depth.

Each split consumes one unit of the depth index, so a value of this type cannot contain a path
with more than `maxDepth` splits. Every feature index is a `Fin nFeatures`, so feature access is
always in range.
-/
structure DecisionTree (α : Type) (nFeatures maxDepth : Nat) where
  /-- Root node with at most `maxDepth` splits on any path. -/
  root : DecisionTree.Node α nFeatures maxDepth
deriving Inhabited

/--
Gradient boosted regression ensemble: `nTrees` trees in a tensor, the shrinkage factor `lr` and a
constant base prediction.
-/
structure GradientBoostedTrees (α : Type) (nFeatures nTrees maxDepth : Nat) where
  /-- The boosted regression trees, applied in order and summed. -/
  trees : Tensor (DecisionTree α nFeatures maxDepth) [nTrees]
  /-- The shrinkage factor multiplying each tree's contribution. -/
  lr : α
  /-- The constant base prediction that the trees correct. -/
  initialPrediction : α

/-!
## Forward pass

Every forward pass takes inputs of `nFeatures` features. The depth `maxDepth` bounds the splits on
any root-to-leaf path and is independent of the feature count.
-/

/-- Forward pass for a single decision tree on an input vector of `nFeatures` features. -/
def DecisionTree.forward {maxDepth nFeatures : Nat}
  (tree : DecisionTree α nFeatures maxDepth)
  (input : Tensor α [nFeatures]) : α :=
  let rec
    /-- The value of the leaf that `input` reaches from `node`, going right exactly when the tested
    feature exceeds the threshold. -/
    traverse {depth : Nat} (node : Node α nFeatures depth) : α :=
      match node with
      | Node.leaf value => value
      | Node.split feature threshold left right =>
        let featureValue := Tensor.item (Tensor.unstack input feature)
        if decide (featureValue > threshold) then traverse right else traverse left
  traverse tree.root

/--
Forward pass for a gradient boosted ensemble on a single input.

Starting at `initialPrediction`, the trees are visited in index order with the update
`acc := acc + lr * tree_i(x)`. Shrinkage multiplies each tree's output before its addition, and
under rounding the result can differ from `initialPrediction + lr * Σᵢ tree_i(x)`.
-/
def GradientBoostedTrees.forward {nTrees maxDepth nFeatures : Nat}
  (model : GradientBoostedTrees α nFeatures nTrees maxDepth)
  (input : Tensor α [nFeatures]) :
  Tensor α .scalar :=
  let rec
    /-- Add `model.lr` times the prediction of each tree from index `i` on to `acc`, in index
    order. -/
    accumulate (i : Nat) (acc : α) : α :=
      if h : i < nTrees then
        let tree := model.trees.getScalar ⟨i, h⟩
        let treePrediction := DecisionTree.forward (α := α) (maxDepth := maxDepth)
          (nFeatures := nFeatures) tree input
        accumulate (i + 1) (acc + model.lr * treePrediction)
      else acc
  let ensemblePrediction := accumulate 0 model.initialPrediction
  Tensor.scalar ensemblePrediction

/-!
## Classical training: CART-style regression trees (MSE)

Trees serve as baselines and reference points. The neural models have reverse-mode derivatives,
while trees get a classical fitting routine that uses no derivatives.

The regression fit is a small greedy CART procedure:

- choose the split `(feature, threshold)` that minimizes
  `SSE(left) + SSE(right)` (sum of squared errors around each side’s mean)
- recurse until the depth runs out, every candidate split leaves one side empty, or the best split
  does not lower the parent's SSE

The fit is deterministic:
- thresholds are the observed feature values
- ties keep the first best split, scanning features in index order and samples in dataset order
-/

/-- A single regression training example: feature vector `x` and scalar target `y`. -/
structure DecisionTree.RegressionExample (nFeatures : Nat) where
  /-- The feature vector. -/
  x : Tensor α [nFeatures]
  /-- The scalar regression target. -/
  y : α

namespace DecisionTree

/-- Mean of an array, the sum divided once by the count (`Context.divNat`), or `0` for an empty
array. -/
def meanOrZero (xs : Array α) : α :=
  if xs.isEmpty then
    0
  else
    -- `Array.sum` is a right fold: this one adds left to right, starting from zero.
    Context.divNat (xs.foldl (fun acc x => acc + x) 0) xs.size

/-- Sum of squared deviations from the mean (SSE). -/
def sse (ys : Array α) : α :=
  if ys.isEmpty then
    0
  else
    let μ := meanOrZero ys
    ys.foldl (fun acc y =>
      let d := y - μ
      acc + d * d) 0

/-- Partition samples into `(left, right)` for a given split, where `x` reads the features of a
sample: a sample goes right exactly when its value at `feature` is greater than `threshold`. -/
def partitionBySplit {ε : Type} {nFeatures : Nat} (x : ε → Tensor α [nFeatures])
    (feature : Fin nFeatures) (threshold : α) (xs : Array ε) : Array ε × Array ε :=
  xs.partition (fun ex => !decide ((x ex).getScalar feature > threshold))

/-- Extract the regression targets from an array of examples. -/
def targets {nFeatures : Nat} (xs : Array (RegressionExample (α := α) nFeatures)) : Array α :=
  xs.map (fun ex => ex.y)

/--
Score a candidate split by the sum of `impurity` over its two sides, and return the score with
the `(left, right)` partition, or `none` when all samples go to one side.
-/
def splitScore {ε : Type} {nFeatures : Nat} (x : ε → Tensor α [nFeatures])
    (impurity : Array ε → α) (feature : Fin nFeatures) (threshold : α) (xs : Array ε) :
    Option (α × Array ε × Array ε) :=
  let (l, r) := partitionBySplit x feature threshold xs
  -- Disallow degenerate splits: a tree should not split when one side is empty.
  if l.isEmpty || r.isEmpty then
    none
  else
    some (impurity l + impurity r, l, r)

end DecisionTree

open DecisionTree

/--
Find the best split `(feature, threshold, left, right, score)` of `xs`, where `x` reads the
features of a sample and `impurity` scores each side. The search tries each of `features` in order
(by default every feature) with each observed value as threshold, skips degenerate splits, and
replaces the best so far only when a candidate's total score is strictly lower.
-/
def DecisionTree.bestSplit {ε : Type} {nFeatures : Nat} (x : ε → Tensor α [nFeatures])
    (impurity : Array ε → α) (xs : Array ε)
    (features : Array (Fin nFeatures) := Array.finRange nFeatures) :
    Option (Fin nFeatures × α × Array ε × Array ε × α) :=
  features.foldl (fun best feature =>
    let thresholds : Array α := xs.map (fun ex => Tensor.getScalar (x ex) feature)
    thresholds.foldl (fun best threshold =>
      match splitScore x impurity feature threshold xs with
      | none => best
      | some (score, l, r) =>
        match best with
        | none => some (feature, threshold, l, r, score)
        | some (bestF, bestT, bestL, bestR, bestScore) =>
          if Context.gtBool bestScore score then
            some (feature, threshold, l, r, score)
          else
            some (bestF, bestT, bestL, bestR, bestScore)
    ) best
  ) none

/-- Leaf prediction value for regression: the mean target. -/
def DecisionTree.leafValue {nFeatures : Nat}
    (xs : Array (RegressionExample (α := α) nFeatures)) : α :=
  meanOrZero (targets xs)

/--
Fit a regression tree with at most `depth` splits on any path by greedy CART splitting. Node `i`
searches the features `features i` (by default every feature) and splits only when its best split
lowers the SSE of its samples. The root is node `0`, and node `i` has children `2 * i + 1` and
`2 * i + 2`.
-/
def DecisionTree.fitRegressionNode {nFeatures : Nat} (depth : Nat)
    (xs : Array (RegressionExample (α := α) nFeatures))
    (features : Nat → Array (Fin nFeatures) := fun _ => Array.finRange nFeatures) :
    Node α nFeatures depth :=
  go depth 0 xs
where
  /-- The subtree with at most `d` splits on any path at node `node`. -/
  go : (d : Nat) → Nat → Array (RegressionExample (α := α) nFeatures) → Node α nFeatures d
    | 0, _, xs => .leaf (leafValue xs)
    | d + 1, node, xs =>
      -- `none` means every candidate split leaves one side empty: stop at a leaf.
      match bestSplit (fun ex : RegressionExample (α := α) nFeatures => ex.x)
          (fun s => sse (targets s)) xs (features node) with
      | none => .leaf (leafValue xs)
      | some (feature, threshold, l, r, score) =>
        let parentScore := sse (targets xs)
        -- Split only when that strictly lowers the SSE.
        if Context.gtBool parentScore score then
          .split feature threshold (go d (2 * node + 1) l) (go d (2 * node + 2) r)
        else
          .leaf (leafValue xs)

/-- Fit a regression decision tree from a batched dataset. -/
def DecisionTree.fitRegression {batch maxDepth nFeatures : Nat}
  (x : Tensor α [batch, nFeatures])
  (y : Tensor α [batch]) :
  DecisionTree α nFeatures maxDepth :=
  let examples : Array (RegressionExample (α := α) nFeatures) :=
    (Array.finRange batch).map (fun i =>
      { x := Tensor.unstack x i, y := Tensor.item (Tensor.unstack y i) })
  { root := fitRegressionNode (α := α) (nFeatures := nFeatures) maxDepth examples }

/-!
## Classical training: CART-style classification trees (Gini impurity)

A classification leaf predicts a label (a `String` or a `Nat`, say), while split thresholds are
numeric. The classifier tree type therefore takes the label type `β` as a separate parameter, so
labels need not be values of the numeric scalar type `α`.

Training mirrors the regression case:
- enumerate candidate thresholds from observed feature values
- pick the split that minimizes weighted Gini impurity
- recurse until the depth runs out, no split leaves both sides nonempty, or the best split does not
  lower the weighted impurity
- predict the majority class at a leaf, breaking ties by first occurrence

PyTorch / sklearn analogies:

- The tree is closest in spirit to `sklearn.tree.DecisionTreeClassifier` with `criterion="gini"`,
  written as a small pure spec.
- The boosting step (adding trees one at a time) follows the high-level idea of
  `sklearn.ensemble.GradientBoostingRegressor` with squared-error loss, without sklearn’s
  engineering details (regularization knobs, histogram binning, etc.).
- scikit-learn weights each side's impurity by its share of the node's samples. Our SSE sums and
  `giniWeighted` scale that by the node size, so splits rank the same way. Its default
  `min_impurity_decrease=0.0` accepts a split that leaves the impurity unchanged, where ours makes
  a leaf. On the XOR points `(0,0) ↦ 0`, `(0,1) ↦ 1`, `(1,0) ↦ 1`, `(1,1) ↦ 0` every first split
  keeps the weighted Gini at `2`, so we fit one leaf that predicts `0`.
-/

/-- A classifier tree node: numeric splits, label-valued leaves. -/
inductive DecisionTree.Classifier.Node (α β : Type) (nFeatures : Nat) : Nat → Type where
  | leaf {depth : Nat} (label : β) : Node α β nFeatures depth
  | split {depth : Nat} (feature : Fin nFeatures) (threshold : α)
      (left right : Node α β nFeatures depth) :
      Node α β nFeatures (depth + 1)
deriving Inhabited

/-- Classification decision tree with at most `maxDepth` splits on any path: numeric splits on `α`
and leaves labelled in `β`. -/
structure DecisionTree.Classifier (α β : Type) (nFeatures maxDepth : Nat) where
  /-- Root node with bounded depth and valid feature indices. -/
  root : Classifier.Node α β nFeatures maxDepth
deriving Inhabited

/-- Forward pass for a classifier decision tree on an input vector of `nFeatures` features.

A split sends inputs with $x_{\mathrm{feature}}>\mathrm{threshold}$ right and the rest left, the
same rule as `DecisionTree.forward`. The test needs only the decidable `>` of `Context α`.
-/
def DecisionTree.Classifier.forward {β : Type} {maxDepth nFeatures : Nat}
  (tree : Classifier α β nFeatures maxDepth)
  (input : Tensor α [nFeatures]) : β :=
  let rec
    /-- The label of the leaf that `input` reaches from `node`. -/
    traverse {depth : Nat} (node : Node α β nFeatures depth) : β :=
      match node with
      | .leaf lbl => lbl
      | .split feature threshold left right =>
        let featureValue := Tensor.item (Tensor.unstack input feature)
        if decide (featureValue > threshold) then traverse right else traverse left
  traverse tree.root

/-- A single classification training example: feature vector `x` and label `y`. -/
structure DecisionTree.Classifier.Example (nFeatures : Nat) (β : Type) where
  /-- The feature vector. -/
  x : Tensor α [nFeatures]
  /-- The class label. -/
  y : β

/-- Most frequent label in `ys`, with ties going to the label that occurs first in `ys` and
`default` for an empty array. -/
def DecisionTree.Classifier.majorityLabel {β : Type} [DecidableEq β] [Inhabited β]
    (ys : Array β) : β :=
  match ys[0]? with
  | none => default
  | some first =>
    ys.toList.eraseDups.foldl (fun best lbl =>
      let cBest := ys.count best
      let cLbl := ys.count lbl
      if cLbl > cBest then lbl else best
    ) first

namespace DecisionTree.Classifier

/-- Gini impurity of a multiset of labels, `0` when it is empty.

`gini(ys) = 1 - Σ_c p(c)^2` where `p(c)` is the empirical class frequency.

It is the standard CART impurity. The value is computed exactly from the class counts `c` of the `n`
labels, as the rational `(n² - Σ_c c²) / n²`, and converted with the scalar's rational cast, which
rounds once in the binary floating-point formats. Subtracting rounded squared frequencies from one
would cancel when one class holds almost every label: the impurity is then about `2 / n`, while the
rounding error of the sum is about the unit roundoff. `TorchLean.Examples.Cancellation.gini_real`
proves the value is `1 - Σ_c p(c)^2` over the reals, and `binary32_gini` shows that 299 labels `0`
and one `1` give an impurity within one unit of roundoff in binary32.
-/
def gini {β : Type} [DecidableEq β] (ys : Array β) : α :=
  if ys.isEmpty then
    0
  else
    let n := ys.size
    let squares := ys.toList.eraseDups.foldl (fun total label => total + ys.count label ^ 2) 0
    (((n ^ 2 - squares : ℤ) / (n ^ 2 : ℤ) : ℚ) : α)

end DecisionTree.Classifier

/-- Weighted Gini impurity `|ys| * gini(ys)`, `0` when `ys` is empty. As in `gini`, it is computed
exactly from the class counts `c` of the `n` labels, as the rational `(n² - Σ_c c²) / n`, and
converted with the scalar's rational cast, so no count is cast to the scalar first. -/
def DecisionTree.Classifier.giniWeighted {β : Type} [DecidableEq β] (ys : Array β) : α :=
  if ys.isEmpty then
    0
  else
    let n := ys.size
    let squares := ys.toList.eraseDups.foldl (fun total label => total + ys.count label ^ 2) 0
    (((n ^ 2 - squares : ℤ) / (n : ℤ) : ℚ) : α)

/-- Extract the labels from an array of classification examples. -/
def DecisionTree.Classifier.targets {nFeatures : Nat} {β : Type}
    (xs : Array (Example (α := α) nFeatures
  β)) : Array β :=
  xs.map (fun ex => ex.y)

/--
Fit a classification tree with at most `depth` splits on any path by greedy CART splitting. Node
`i` searches the features `features i` (by default every feature) and splits only when its best
split lowers the weighted Gini impurity of its samples. Nodes are numbered as in
`fitRegressionNode`, and each leaf predicts the majority label of its samples.
-/
def DecisionTree.Classifier.fitNode {nFeatures : Nat} {β : Type} [DecidableEq β] [Inhabited β]
    (depth : Nat) (xs : Array (Example (α := α) nFeatures β))
    (features : Nat → Array (Fin nFeatures) := fun _ => Array.finRange nFeatures) :
    Node α β nFeatures depth :=
  go depth 0 xs
where
  /-- The subtree with at most `d` splits on any path at node `node`. -/
  go : (d : Nat) → Nat → Array (Example (α := α) nFeatures β) →
      Node α β nFeatures d
    | 0, _, xs => .leaf (majorityLabel (targets xs))
    | d + 1, node, xs =>
      match bestSplit (fun ex : Example (α := α) nFeatures β => ex.x)
          (fun s => giniWeighted (α := α) (targets s)) xs (features node) with
      | none => .leaf (majorityLabel (targets xs))
      | some (feature, threshold, l, r, score) =>
        let parentScore := giniWeighted (α := α) (targets xs)
        if Context.gtBool parentScore score then
          .split feature threshold (go d (2 * node + 1) l) (go d (2 * node + 2) r)
        else
          .leaf (majorityLabel (targets xs))

/--
Fit a classification decision tree (CART-style) using Gini impurity.

The label vector has exactly one element per input row. Its length is part of the type, so fitting
cannot silently discard labels or invent missing ones.
-/
def DecisionTree.Classifier.fit {β : Type} [Storage β]
    [DecidableEq β] [Inhabited β]
  {batch maxDepth nFeatures : Nat}
  (x : Tensor α [batch, nFeatures])
  (y : Tensor β [batch]) :
  Classifier α β nFeatures maxDepth :=
  let examples : Array (Example (α := α) nFeatures β) :=
    (Array.finRange batch).map (fun i =>
      { x := Tensor.unstack x i, y := y.getScalar i })
  { root := fitNode (β := β) (nFeatures := nFeatures) maxDepth examples }

/--
Squared-error residuals `target - prediction` of the current ensemble, one per row: the targets
for the next tree.
-/
def GradientBoostedTrees.residuals {batch nTrees maxDepth nFeatures : Nat}
  (model : GradientBoostedTrees α nFeatures nTrees maxDepth)
  (input : Tensor α [batch, nFeatures])
  (target : Tensor α [batch]) :
  Tensor α [batch] :=
  let predictions : Tensor α [batch] :=
    Tensor.mapLeading [batch] (forward model) input
  target - predictions

/--
One gradient-boosting "add a tree" step, given a pre-fit `newTree`.

Returns the `Tensor.meanSquaredError` of the current model on the batch, paired with the model
that has `newTree` appended. `trainStepFit` fits that tree to the
residuals first.
-/
def GradientBoostedTrees.trainStep {batch nTrees maxDepth nFeatures : Nat}
  (model : GradientBoostedTrees α nFeatures nTrees maxDepth)
  (input : Tensor α [batch, nFeatures])
  (target : Tensor α [batch])
  (newTree : DecisionTree α nFeatures maxDepth) :
  (Tensor α .scalar × GradientBoostedTrees α nFeatures (nTrees + 1) maxDepth) :=
  let predictions : Tensor α [batch] :=
    Tensor.mapLeading [batch] (forward model) input
  let loss := Tensor.scalar (Tensor.meanSquaredError predictions target)
  -- Append `newTree` as the last tree.
  let newTrees := Tensor.cat (s := [0]) ⟨0, by simp⟩ ![nTrees, 1]
      (Fin.cons model.trees (Fin.cons (Tensor.dim (fun _ => Tensor.scalar newTree)) finZeroElim))
      [nTrees + 1]
  let updatedModel := {
    model with
    trees := newTrees
  }
  (loss, updatedModel)

/-!
### Gradient boosting: a "fit-one-more-tree" step

A boosting step that fits its own tree keeps the baseline self-contained and deterministic: the
tree comes from `DecisionTree.fitRegression` on the current residuals.
-/

/-- Fits a regression tree to the current residuals and appends it with
`GradientBoostedTrees.trainStep`, which also returns the loss before the update. -/
def GradientBoostedTrees.trainStepFit {batch nTrees maxDepth nFeatures : Nat}
  (model : GradientBoostedTrees α nFeatures nTrees maxDepth)
  (input : Tensor α [batch, nFeatures])
  (target : Tensor α [batch]) :
  (Tensor α .scalar × GradientBoostedTrees α nFeatures (nTrees + 1) maxDepth) :=
  let residuals := GradientBoostedTrees.residuals model input target
  let newTree :=
    fitRegression (α := α) (batch := batch) (maxDepth := maxDepth)
      (nFeatures := nFeatures) input residuals
  trainStep model input target newTree

namespace DecisionTree

/--
Adds one to the `feature` entry of a length-`nFeatures` counter vector, for the split-count
feature importance.
-/
def incrFeature {nFeatures : Nat} (acc : Tensor α [nFeatures])
    (feature : Fin nFeatures) :
  Tensor α [nFeatures] :=
  Tensor.dim (fun i =>
    let v := acc.getScalar i
    if i = feature then Tensor.scalar (v + (1 : α)) else Tensor.scalar v)

end DecisionTree

/--
Adds to `acc`, for each feature index, the number of split nodes in the tree that test it.
-/
def DecisionTree.Node.featureCounts {nFeatures depth : Nat} :
    Node α nFeatures depth → Tensor α [nFeatures] →
      Tensor α [nFeatures]
  | leaf _v, acc => acc
  | split featureIdx _threshold left right, acc =>
      let acc' := incrFeature (α := α) (nFeatures := nFeatures) acc featureIdx
      let accL := featureCounts (nFeatures := nFeatures) left acc'
      featureCounts (nFeatures := nFeatures) right accL

/-- Split-count feature importance for an ensemble: the share of all split nodes that test each
feature.

The counts are scaled by one over their total, so they sum to one up to rounding, and an ensemble
without splits gives all zeros. Gain-based importance, as in XGBoost or LightGBM, weights each
split by its loss reduction and can rank features differently.
-/
def GradientBoostedTrees.featureImportance {nTrees maxDepth nFeatures : Nat}
  (model : GradientBoostedTrees α nFeatures nTrees maxDepth) :
  Tensor α [nFeatures] :=
  let rec
    /-- Add the split counts of each tree from index `i` on to `acc`, in index order. -/
    accumulate (i : Nat) (acc : Tensor α [nFeatures]) : Tensor α [nFeatures] :=
      if h : i < nTrees then
        let tree := model.trees.getScalar ⟨i, h⟩
        let acc' := Node.featureCounts (α := α) (nFeatures := nFeatures) tree.root acc
        accumulate (i + 1) acc'
      else acc
  let counts := accumulate 0 (Tensor.full (.dim nFeatures .scalar) 0)
  let total : α := Tensor.sum counts
  if Context.gtBool total 0 then
    Tensor.scale counts (1 / total)
  else
    counts

end TorchLean
