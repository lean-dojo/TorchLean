/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape.ConcatSlice
public import NN.Spec.Core.Sequence
public import NN.Spec.Core.TensorReductionShape.Reductions

/-!
# Gradient boosted trees (spec model)

This is a math/reference specification of gradient boosting using decision trees.

Important caveat:
- Many computations here are written in a straightforward, proof-friendly style rather than as a
  tuned implementation.

References (classical):
- CART: Breiman, Friedman, Olshen, Stone, "Classification and Regression Trees", 1984.
- Gradient boosting: Friedman, "Greedy Function Approximation: A Gradient Boosting Machine", 2001.
- XGBoost: Chen and Guestrin, "XGBoost: A Scalable Tree Boosting System", 2016.
- LightGBM: Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree", 2017.

Squared-error boosting fits trees to `target - prediction`. The shared mean-squared-error gradient
is `mseLossGradSpec` in `NN/Spec/Models/LinearRegression.lean`.

## Implementation status

No API builder implements this model. `NN/Spec/Models/RandomForest.lean` reuses its decision
trees; no theorem is proved about it.
-/

@[expose] public section


open TorchLean

namespace Spec

open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]

/-!
## Tree representation

We represent a decision tree as a small inductive datatype:

- `leaf value` stores the prediction for that leaf
- `split feature threshold left right` branches on a single feature

This is kept compact: it is easy to interpret (forward pass) and easy to fit with a
simple greedy CART-style algorithm (implemented below).

Note on comparisons:
The spec layer’s scalar interface (`Context α`) gives us a decidable `>` (via
  `Context.decidableGT`)
but does not promise a decidable `<` for every backend. To stay portable, the tree uses the rule:

`goRight := (x_feature > threshold)`

and goes left otherwise (equivalently, $x_{\mathrm{feature}}\leq\mathrm{threshold}$ for the usual
numeric orders).
-/

/--
A regression-tree node for the typed GBDT specification.

- `leaf value` stores the prediction for that leaf.
- `split feature threshold left right` branches on a single feature using the rule
  `goRight := (x_feature > threshold)`.
-/
inductive TreeNode (α : Type) (nFeatures : Nat) : Nat → Type where
  | leaf {depth : Nat} (value : α) : TreeNode α nFeatures depth
  | split {depth : Nat} (feature : Fin nFeatures) (threshold : α)
      (left right : TreeNode α nFeatures depth) : TreeNode α nFeatures (depth + 1)
deriving Inhabited

/--
Decision-tree specification whose type records its feature count and maximum split depth.

Each split consumes one unit of the depth index, so a value of this type cannot contain a path
deeper than `maxDepth`. Its feature index is a `Fin nFeatures`, ruling out invalid feature access.
-/
structure DecisionTreeSpec (α : Type) (nFeatures maxDepth : Nat) where
  /-- Root node with at most `maxDepth` splits on any path. -/
  root : TreeNode α nFeatures maxDepth
deriving Inhabited

/--
Gradient boosted tree ensemble (regression-style) specification.

The model stores an explicit tensor of trees, a shrinkage parameter, and an initial prediction.
-/
structure GradientBoostedTreesSpec (α : Type) (nFeatures nTrees maxDepth : Nat) where
  /-- The boosted regression trees, applied in order and summed. -/
  trees : Tensor (DecisionTreeSpec α nFeatures maxDepth) [nTrees]
  /-- The shrinkage factor multiplying each tree's contribution. -/
  learningRate : α
  /-- The constant base prediction that the trees correct. -/
  initialPrediction : α

/-!
## Forward pass

All forward passes in this file are explicit about the feature dimension `nFeatures`. Tree depth
(`maxDepth`) limits *how many splits* a tree may contain; it has nothing to do with how many input
features exist.
-/

/-- Forward pass for a single decision tree on an input vector of `nFeatures` features. -/
def decisionTreeForwardSpec {maxDepth nFeatures : Nat}
  (tree : DecisionTreeSpec α nFeatures maxDepth)
  (input : Tensor α [nFeatures]) : α :=
  let rec traverse {depth : Nat} (node : TreeNode α nFeatures depth) : α :=
    match node with
    | TreeNode.leaf value => value
    | TreeNode.split feature threshold left right =>
      let featureValue := Tensor.item (get input feature)
      if decide (featureValue > threshold) then traverse right else traverse left
  traverse tree.root

/-- Apply a decision tree independently at every index of a leading shape. -/
def decisionTreeForwardLeadingSpec (leading : Shape) {maxDepth nFeatures : Nat}
  (tree : DecisionTreeSpec α nFeatures maxDepth)
  (input : Tensor α (leading.concat [nFeatures])) :
  Tensor α (leading.concat .scalar) :=
  Tensor.mapLeading leading
    (fun x => Tensor.scalar (decisionTreeForwardSpec tree x)) input

/--
Forward pass for a gradient boosted ensemble on a single input.

Starting at `initialPrediction`, visit trees in index order and update
`acc := acc + learningRate * tree_i(x)`. Shrinkage is applied before each addition; rounded
arithmetic does not in general permit factoring it out of the sum.
-/
def gradientBoostedTreesForwardSpec {nTrees maxDepth nFeatures : Nat}
  (model : GradientBoostedTreesSpec α nFeatures nTrees maxDepth)
  (input : Tensor α [nFeatures]) :
  Tensor α .scalar :=
  let rec accumulate_trees (i : Nat) (acc : α) : α :=
    if h : i < nTrees then
      let tree := model.trees.getScalar ⟨i, h⟩
      let treePrediction := decisionTreeForwardSpec (α := α) (maxDepth := maxDepth)
        (nFeatures := nFeatures) tree input
      accumulate_trees (i + 1) (acc + model.learningRate * treePrediction)
    else acc
  let ensemblePrediction := accumulate_trees 0 model.initialPrediction
  Tensor.scalar ensemblePrediction

/-- Apply a gradient-boosted ensemble independently at every index of a leading shape. -/
def gradientBoostedTreesForwardLeadingSpec (leading : Shape)
  {nTrees maxDepth nFeatures : Nat}
  (model : GradientBoostedTreesSpec α nFeatures nTrees maxDepth)
  (input : Tensor α (leading.concat [nFeatures])) :
  Tensor α (leading.concat .scalar) :=
  Tensor.mapLeading leading (gradientBoostedTreesForwardSpec model) input

/-!
## Classical training: CART-style regression trees (MSE)

Tree-based models here are intended as **baselines** and **reference points**. For neural
models we implement reverse-mode explicitly; for trees we instead provide a classical (non-gradient)
training routine.

The code below implements a small greedy CART-like procedure for *regression*:

- choose the split `(feature, threshold)` that minimizes
  `SSE(left) + SSE(right)` (sum of squared errors around each side’s mean)
- recurse until depth runs out or the split becomes degenerate

This is deterministic by construction:
- thresholds are chosen from the observed feature values
- ties are broken by the “first best” encountered during folding

This implementation prioritizes clarity and determinism over performance.
-/

/-- A single regression training example: feature vector `x` and scalar target `y`. -/
structure RegressionExample (nFeatures : Nat) where
  /-- The feature vector. -/
  x : Tensor α [nFeatures]
  /-- The scalar regression target. -/
  y : α

namespace GradientBoostedTrees.Internal

/-- Mean of an array, with `0` as a convenient default for an empty sample. -/
def meanOrZero (xs : Array α) : α :=
  if xs.isEmpty then
    0
  else
    -- `Array.sum` is a right fold; retain the zero-seeded left-to-right arithmetic here.
    xs.foldl (fun acc x => acc + x) 0 / (xs.size : α)

/-- Sum of squared deviations from the mean (SSE). -/
def sse (ys : Array α) : α :=
  if ys.isEmpty then
    0
  else
    let μ := meanOrZero ys
    ys.foldl (fun acc y =>
      let d := y - μ
      acc + d * d) 0

/-- Partition samples into `(left, right)` for a given split. -/
def partitionBySplit {nFeatures : Nat}
  (feature : Fin nFeatures) (threshold : α) (xs : Array (RegressionExample (α := α) nFeatures)) :
  Array (RegressionExample (α := α) nFeatures) × Array (RegressionExample (α := α) nFeatures) :=
  xs.partition (fun ex => !decide (ex.x.getScalar feature > threshold))

/-- Extract the regression targets from an array of examples. -/
def targets {nFeatures : Nat} (xs : Array (RegressionExample (α := α) nFeatures)) : Array α :=
  xs.map (fun ex => ex.y)

/--
Score a candidate split by sum of squared errors (SSE).

Returns `none` for degenerate splits (all samples go to one side).
-/
def splitScore {nFeatures : Nat}
  (feature : Fin nFeatures) (threshold : α)
  (xs : Array (RegressionExample (α := α) nFeatures)) :
  Option (α × Array (RegressionExample (α := α) nFeatures) ×
    Array (RegressionExample (α := α) nFeatures)) :=
  let (l, r) := partitionBySplit (α := α) feature threshold xs
  -- Disallow degenerate splits: a tree should not split when one side is empty.
  if l.isEmpty || r.isEmpty then
    none
  else
    some (sse (targets l) + sse (targets r), l, r)

end GradientBoostedTrees.Internal

open GradientBoostedTrees.Internal

/-- Find the best split `(feature, threshold)` by exhaustive search over observed thresholds. -/
def bestSplit {nFeatures : Nat}
  (xs : Array (RegressionExample (α := α) nFeatures)) :
  Option (Fin nFeatures × α × Array (RegressionExample (α := α) nFeatures) ×
    Array (RegressionExample (α := α) nFeatures) × α) :=
  (Array.finRange nFeatures).foldl (fun best feature =>
    let thresholds : Array α := xs.map (fun ex => Tensor.getScalar ex.x feature)
    thresholds.foldl (fun best threshold =>
      match splitScore (α := α) feature threshold xs with
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
def leafValue {nFeatures : Nat} (xs : Array (RegressionExample (α := α) nFeatures)) : α :=
  meanOrZero (targets xs)

/--
Fit a regression tree by greedy CART-style splitting (MSE/SSE), with a depth budget.

`depthLeft` counts how many splits we are still allowed to make.
-/
def fitRegressionNode {nFeatures : Nat} :
    (depth : Nat) → Array (RegressionExample (α := α) nFeatures) → TreeNode α nFeatures depth
  | 0, xs => TreeNode.leaf (leafValue (α := α) xs)
  | (d+1), xs =>
    -- If there’s no meaningful split, stop at a leaf.
    match bestSplit (α := α) (nFeatures := nFeatures) xs with
    | none => TreeNode.leaf (leafValue (α := α) xs)
    | some (feature, threshold, l, r, score) =>
      let parentScore := sse (targets xs)
      -- Only split if it actually improves SSE.
      if Context.gtBool parentScore score then
        TreeNode.split feature threshold
          (fitRegressionNode d l)
          (fitRegressionNode d r)
      else
        TreeNode.leaf (leafValue (α := α) xs)

/-- Fit a regression decision tree from a batched dataset. -/
def decisionTreeFitRegressionMseSpec {batch maxDepth nFeatures : Nat}
  (x : Tensor α [batch, nFeatures])
  (y : Tensor α [batch]) :
  DecisionTreeSpec α nFeatures maxDepth :=
  let examples : Array (RegressionExample (α := α) nFeatures) :=
    (Array.finRange batch).map (fun i =>
      { x := get x i, y := Tensor.item (get y i) })
  { root := fitRegressionNode (α := α) (nFeatures := nFeatures) maxDepth examples }

/-!
## Classical training: CART-style classification trees (Gini impurity)

For classification we often want the leaf prediction to be a *label* (e.g. `String` or `Nat`),
while split thresholds remain numeric. To avoid forcing labels into the numeric scalar type `α`,
we define a separate classifier tree type parameterized by the label type `β`.

The training algorithm mirrors the regression case:
- enumerate candidate thresholds from observed feature values
- pick the split that minimizes weighted Gini impurity
- recurse until depth runs out or no improvement is possible
- leaf prediction is the majority class (deterministic tie-breaking via first occurrence)

Why `β` is separate from `α`:

- Splits compare numeric features (`α`), so they need ordering/decidable comparison.
- Leaf predictions are usually discrete (`β`), and we do not want to pretend that labels form a
  numeric scalar domain.

PyTorch / sklearn analogies:

- This is closest in spirit to `sklearn.tree.DecisionTreeClassifier` with `criterion="gini"`,
  expressed as a small pure spec.
- The *boosting* semantics (adding many trees sequentially) matches the high-level idea of
  `sklearn.ensemble.GradientBoostingClassifier`, but TorchLean does not try to reproduce all of
  sklearn’s engineering details (regularization knobs, histogram binning, etc.).
-/

/-- A classifier tree node: numeric splits, label-valued leaves. -/
inductive ClassifierTreeNode (α β : Type) (nFeatures : Nat) : Nat → Type where
  | leaf {depth : Nat} (label : β) : ClassifierTreeNode α β nFeatures depth
  | split {depth : Nat} (feature : Fin nFeatures) (threshold : α)
      (left right : ClassifierTreeNode α β nFeatures depth) :
      ClassifierTreeNode α β nFeatures (depth + 1)
deriving Inhabited

/-- Specification wrapper for a classification decision tree (numeric splits, label-valued leaves).
  -/
structure DecisionTreeClassifierSpec (α β : Type) (nFeatures maxDepth : Nat) where
  /-- Root node with bounded depth and valid feature indices. -/
  root : ClassifierTreeNode α β nFeatures maxDepth
deriving Inhabited

/-- Forward pass for a classifier decision tree on an input vector of `nFeatures` features.

Branching convention:

- go right iff `(x[feature] > threshold)`,
- otherwise go left.

This mirrors the common convention
$x_{\mathrm{feature}}\leq\mathrm{threshold}$ goes left and
$x_{\mathrm{feature}}>\mathrm{threshold}$ goes right, but avoids needing a decidable `<` for every
`Context α` backend.
-/
def decisionTreeClassifyForwardSpec {β : Type} {maxDepth nFeatures : Nat}
  (tree : DecisionTreeClassifierSpec α β nFeatures maxDepth)
  (input : Tensor α [nFeatures]) : β :=
  let rec traverse {depth : Nat} (node : ClassifierTreeNode α β nFeatures depth) : β :=
    match node with
    | .leaf lbl => lbl
    | .split feature threshold left right =>
      let featureValue := Tensor.item (get input feature)
      if decide (featureValue > threshold) then traverse right else traverse left
  traverse tree.root

/-- A single classification training example: feature vector `x` and label `y`. -/
structure ClassificationExample (nFeatures : Nat) (β : Type) where
  /-- The feature vector. -/
  x : Tensor α [nFeatures]
  /-- The class label. -/
  y : β

namespace GradientBoostedTrees.Internal

/-- Count how many times `lbl` appears in `ys`. -/
def countEq {β : Type} [DecidableEq β] (lbl : β) (ys : Array β) : Nat :=
  ys.countP (fun y => decide (y = lbl))

/-- Remove repeated labels while preserving their first-occurrence order. -/
def distinct {β : Type} [DecidableEq β] (ys : Array β) : Array β :=
  ys.foldl (fun labels y =>
    if labels.any (fun label => decide (label = y)) then labels else labels.push y) #[]

end GradientBoostedTrees.Internal

/-- Majority label with deterministic tie-breaking.

If there is a tie, we keep the earlier winner from the fold. This is intentional: it avoids
non-determinism and keeps the spec stable across backends.
-/
def majorityLabel {β : Type} [DecidableEq β] [Inhabited β] (ys : Array β) : β :=
  match ys[0]? with
  | none => default
  | some first =>
    let labels := GradientBoostedTrees.Internal.distinct ys
    labels.foldl (fun best lbl =>
      let cBest := countEq best ys
      let cLbl := countEq lbl ys
      if cLbl > cBest then lbl else best
    ) first

namespace GradientBoostedTrees.Internal

/-- Gini impurity of a multiset of labels.

`gini(ys) = 1 - Σ_c p(c)^2` where `p(c)` is the empirical class frequency.

This is the standard CART impurity used by many tree classifiers.
-/
def gini {β : Type} [DecidableEq β] (ys : Array β) : α :=
  if ys.isEmpty then
    0
  else
    let n : α := (ys.size : α)
    let labels := distinct ys
    let sumSq :=
      labels.foldl (fun acc lbl =>
        let c : α := (countEq lbl ys : Nat)
        let p := c / n
        acc + (p * p)
      ) 0
    (1 : α) - sumSq

end GradientBoostedTrees.Internal

/-- Weighted Gini impurity: `|ys| * gini(ys)`. -/
def giniWeighted {β : Type} [DecidableEq β] (ys : Array β) : α :=
  (ys.size : α) * gini ys

/-- Extract the labels from an array of classification examples. -/
def classTargets {nFeatures : Nat} {β : Type} (xs : Array (ClassificationExample (α := α) nFeatures
  β)) : Array β :=
  xs.map (fun ex => ex.y)

/-- Partition classification samples into `(left, right)` for a candidate split. -/
def partitionBySplitC {nFeatures : Nat} {β : Type}
  (feature : Fin nFeatures) (threshold : α)
  (xs : Array (ClassificationExample (α := α) nFeatures β)) :
  Array (ClassificationExample (α := α) nFeatures β) ×
    Array (ClassificationExample (α := α) nFeatures β) :=
  xs.partition (fun ex => !decide (ex.x.getScalar feature > threshold))

namespace GradientBoostedTrees.Internal

/--
Score a candidate classification split `(feature, threshold)` by weighted Gini impurity.

Returns `none` when the split is degenerate (one side is empty); otherwise returns the score and
the `(left, right)` partitions.
-/
def splitScoreC {nFeatures : Nat} {β : Type} [DecidableEq β]
  (feature : Fin nFeatures) (threshold : α)
  (xs : Array (ClassificationExample (α := α) nFeatures β)) :
  Option (α × Array (ClassificationExample (α := α) nFeatures β) ×
    Array (ClassificationExample (α := α) nFeatures β)) :=
  let (l, r) := partitionBySplitC feature threshold xs
  if l.isEmpty || r.isEmpty then
    none
  else
    let yl := classTargets l
    let yr := classTargets r
    some (giniWeighted yl + giniWeighted yr, l, r)

end GradientBoostedTrees.Internal

/--
Find the best classification split `(feature, threshold)` by exhaustive search.

Thresholds are drawn from the observed feature values in the dataset.
-/
def bestSplitC {nFeatures : Nat} {β : Type} [DecidableEq β]
  (xs : Array (ClassificationExample (α := α) nFeatures β)) :
  Option (Fin nFeatures × α × Array (ClassificationExample (α := α) nFeatures β) × Array
    (ClassificationExample (α := α) nFeatures β) × α) :=
  (Array.finRange nFeatures).foldl (fun best feature =>
    let thresholds : Array α := xs.map (fun ex => Tensor.getScalar ex.x feature)
    thresholds.foldl (fun best threshold =>
      match splitScoreC (β := β) feature threshold xs with
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

/--
Fit a classification tree node by greedy CART-style splitting (Gini impurity).

`depthLeft` counts how many more splits we are allowed to make.
-/
def fitClassificationNode {nFeatures : Nat} {β : Type} [DecidableEq β] [Inhabited β] :
    (depth : Nat) → Array (ClassificationExample (α := α) nFeatures β) →
      ClassifierTreeNode α β nFeatures depth
  | 0, xs => .leaf (majorityLabel (β := β) (classTargets xs))
  | (d+1), xs =>
    match bestSplitC (β := β) (nFeatures := nFeatures) xs with
    | none => .leaf (majorityLabel (β := β) (classTargets xs))
    | some (feature, threshold, l, r, score) =>
      let parentScore := giniWeighted (β := β) (classTargets xs)
      if Context.gtBool parentScore score then
        .split feature threshold
          (fitClassificationNode (β := β) d l)
          (fitClassificationNode (β := β) d r)
      else
        .leaf (majorityLabel (β := β) (classTargets xs))

/--
Fit a classification decision tree (CART-style) using Gini impurity.

The label vector has exactly one element per input row. Its length is part of the type, so fitting
cannot silently discard labels or invent missing ones.
-/
def decisionTreeFitClassificationGiniSpec {β : Type} [TorchLean.Storage β]
    [DecidableEq β] [Inhabited β]
  {batch maxDepth nFeatures : Nat}
  (x : Tensor α [batch, nFeatures])
  (y : Tensor β [batch]) :
  DecisionTreeClassifierSpec α β nFeatures maxDepth :=
  let examples : Array (ClassificationExample (α := α) nFeatures β) :=
    (Array.finRange batch).map (fun i =>
      { x := get x i, y := y.getScalar i })
  { root := fitClassificationNode (β := β) (nFeatures := nFeatures) maxDepth examples }

-- Mean Squared Error loss for regression
/-- Mean squared error (MSE) loss for regression, reduced to a scalar by averaging over the batch.
  -/
def gbtMseLossSpec {batch nTrees maxDepth nFeatures : Nat}
  (model : GradientBoostedTreesSpec α nFeatures nTrees maxDepth)
  (input : Tensor α [batch, nFeatures])
  (target : Tensor α [batch]) (h : batch ≠ 0) :
  Tensor α .scalar :=
  let predictions := gradientBoostedTreesForwardLeadingSpec (.dim batch .scalar) model input
  let errors := subSpec predictions target
  let squaredErrors := squareSpec errors
  have inst : Shape.HasNonemptyAxis 0 (Shape.dim batch Shape.scalar) := by
    apply Shape.hasNonemptyAxisZeroOfNe h
  let mse := reduceSum 0 squaredErrors inst.proof
  scaleSpec mse (1 / (batch : α))

/--
Residual computation for gradient boosting.

For squared-error regression, the residual is `target - prediction`.
-/
def computeResidualsSpec {batch nTrees maxDepth nFeatures : Nat}
  (model : GradientBoostedTreesSpec α nFeatures nTrees maxDepth)
  (input : Tensor α [batch, nFeatures])
  (target : Tensor α [batch]) :
  Tensor α [batch] :=
  let predictions := gradientBoostedTreesForwardLeadingSpec (.dim batch .scalar) model input
  subSpec target predictions

/--
One gradient-boosting "add a tree" step, given a pre-fit `newTree`.

This returns the loss before the update and the model with `newTree` appended.
`gradientBoostedTreesTrainStepFitSpec` also fits the new tree to the current residuals.
-/
def gradientBoostedTreesTrainStepSpec {batch nTrees maxDepth nFeatures : Nat}
  (model : GradientBoostedTreesSpec α nFeatures nTrees maxDepth)
  (input : Tensor α [batch, nFeatures])
  (target : Tensor α [batch])
  (newTree : DecisionTreeSpec α nFeatures maxDepth)
  (h : batch ≠ 0) :
  (Tensor α .scalar × GradientBoostedTreesSpec α nFeatures (nTrees + 1) maxDepth) :=
  -- Compute loss
  let loss := gbtMseLossSpec model input target h
  -- Add new tree to ensemble
  let newTrees := concatAxisSpec .scalar model.trees (Tensor.dim (fun _ => Tensor.scalar newTree))
  let updatedModel := {
    model with
    trees := newTrees
  }
  (loss, updatedModel)

/-!
### Gradient boosting: a "fit-one-more-tree" step

The original `gradientBoostedTreesTrainStepSpec` expects a pre-fit `newTree`. For a more
complete baseline, we also provide a deterministic step that *fits* that tree to the residuals.
-/

/-- Fit a new tree to residuals and append it to the ensemble. -/
def gradientBoostedTreesTrainStepFitSpec {batch nTrees maxDepth nFeatures : Nat}
  (model : GradientBoostedTreesSpec α nFeatures nTrees maxDepth)
  (input : Tensor α [batch, nFeatures])
  (target : Tensor α [batch])
  (h : batch ≠ 0) :
  (Tensor α .scalar × GradientBoostedTreesSpec α nFeatures (nTrees + 1) maxDepth) :=
  let residuals := computeResidualsSpec model input target
  let newTree :=
    decisionTreeFitRegressionMseSpec (α := α) (batch := batch) (maxDepth := maxDepth)
      (nFeatures := nFeatures) input residuals
  gradientBoostedTreesTrainStepSpec model input target newTree h

namespace GradientBoostedTrees.Internal

/--
Increment a single feature counter by 1 inside a length-`nFeatures` vector.

This is used by the split-count feature-importance computation below.
-/
def incrFeature {nFeatures : Nat} (acc : Tensor α [nFeatures])
    (feature : Fin nFeatures) :
  Tensor α [nFeatures] :=
  Tensor.dim (fun i =>
    let v := acc.getScalar i
    if i = feature then Tensor.scalar (v + (1 : α)) else Tensor.scalar v)

end GradientBoostedTrees.Internal

/--
Count how many times each feature index appears in split nodes of a tree.

This mirrors a very common "split count" importance heuristic.
-/
def treeFeatureCounts {nFeatures depth : Nat} :
    TreeNode α nFeatures depth → Tensor α [nFeatures] →
      Tensor α [nFeatures]
  | TreeNode.leaf _v, acc => acc
  | TreeNode.split featureIdx _threshold left right, acc =>
      let acc' := incrFeature (α := α) (nFeatures := nFeatures) acc featureIdx
      let accL := treeFeatureCounts (nFeatures := nFeatures) left acc'
      treeFeatureCounts (nFeatures := nFeatures) right accL

/-- Simple split-count feature importance for an ensemble.

This mirrors the common "how often was a feature used in a split?" heuristic.  It is *not* the
same as gain-based importance in XGBoost/LightGBM, but it is deterministic and easy to interpret.
-/
def computeFeatureImportanceSpec {nTrees maxDepth nFeatures : Nat}
  (model : GradientBoostedTreesSpec α nFeatures nTrees maxDepth) :
  Tensor α [nFeatures] :=
  let rec accumulate_importance (i : Nat) (acc : Tensor α [nFeatures]) : Tensor α [nFeatures] :=
    if h : i < nTrees then
      let tree := model.trees.getScalar ⟨i, h⟩
      let acc' := treeFeatureCounts (α := α) (nFeatures := nFeatures) tree.root acc
      accumulate_importance (i + 1) acc'
    else acc
  let counts := accumulate_importance 0 (Tensor.full (.dim nFeatures .scalar) 0)
  let total : α := sumSpec counts
  if Context.gtBool total 0 then
    scaleSpec counts (1 / total)
  else
    counts

end Spec
