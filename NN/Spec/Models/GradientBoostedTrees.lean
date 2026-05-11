/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape

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
-/

@[expose] public section


namespace Spec

open Tensor

variable {α : Type} [Context α]

/-!
## Tree representation

We represent a decision tree as a small inductive datatype:

- `leaf value` stores the prediction for that leaf
- `split feature threshold left right` branches on a single feature

This is deliberately compact: it is easy to interpret (forward pass) and easy to fit with a
simple greedy CART-style algorithm (implemented below).

Note on comparisons:
The spec layer’s scalar interface (`Context α`) gives us a decidable `>` (via
  `Context.decidable_gt`)
but does not promise a decidable `<` for every backend. To stay portable, the tree uses the rule:

`goRight := (x_feature > threshold)`

and goes left otherwise (which matches “≤ threshold” for the usual numeric orders).
-/

/--
A regression-tree node for the typed GBDT specification.

- `leaf value` stores the prediction for that leaf.
- `split feature threshold left right` branches on a single feature using the rule
  `goRight := (x_feature > threshold)`.

This keeps the representation small and easy to interpret.
-/
inductive TreeNode (α : Type) where
  | leaf (value : α) : TreeNode α
  | split (feature : Nat) (threshold : α) (left right : TreeNode α) : TreeNode α
deriving Inhabited

/--
Decision-tree spec wrapper with an explicit max-depth budget.

The `max_depth` field is redundant (it matches the type index) but is convenient in downstream
code that wants a value-level knob.
-/
structure DecisionTreeSpec (α : Type) (maxDepth : Nat) where
  /-- root. -/
  root : TreeNode α
  /-- max depth. -/
  max_depth : Nat := maxDepth
deriving Inhabited

/--
Gradient boosted tree ensemble (regression-style) specification.

We keep the model as an explicit tensor of trees plus a shrinkage parameter `learning_rate` and an
`initial_prediction` bias term.
-/
structure GradientBoostedTreesSpec (α : Type) (nTrees : Nat) (maxDepth : Nat) where
  /-- trees. -/
  trees : Tensor (DecisionTreeSpec α maxDepth) (.dim nTrees .scalar)
  /-- learning rate. -/
  learning_rate : α
  /-- initial prediction. -/
  initial_prediction : α

/-!
## Forward pass

All forward passes in this file are explicit about the feature dimension `nFeatures`. Tree depth
(`maxDepth`) limits *how many splits* a tree may contain; it has nothing to do with how many input
features exist.
-/

/-- Forward pass for a single decision tree on an input vector of `nFeatures` features. -/
def decisionTreeForwardSpecN {maxDepth nFeatures : Nat}
  (tree : DecisionTreeSpec α maxDepth)
  (input : Tensor α (.dim nFeatures .scalar)) : α :=
  let rec traverse (node : TreeNode α) : α :=
    match node with
    | TreeNode.leaf value => value
    | TreeNode.split feature_idx threshold left right =>
      match input with
      | Tensor.dim values =>
        if h : feature_idx < nFeatures then
          match values ⟨feature_idx, h⟩ with
          | Tensor.scalar feature_value =>
            if decide (feature_value > threshold) then
              traverse right
            else
              traverse left
        else
          -- Out-of-range feature index: treat as a missing feature and fall back to the left
          -- branch.
          traverse left
  traverse tree.root

/-- Batched forward pass for a single decision tree. -/
def decisionTreeBatchedForwardSpecN {batch maxDepth nFeatures : Nat}
  (tree : DecisionTreeSpec α maxDepth)
  (input : Tensor α (.dim batch (.dim nFeatures .scalar))) :
  Tensor α (.dim batch .scalar) :=
  match input with
  | Tensor.dim batch_fn =>
    Tensor.dim (fun i => Tensor.scalar (decisionTreeForwardSpecN (α := α) (maxDepth := maxDepth)
      (nFeatures := nFeatures) tree (batch_fn i)))

/--
Forward pass for a gradient boosted ensemble on a single input.

This simply accumulates `initial_prediction + learning_rate * sum(tree_i(x))`.
-/
def gradientBoostedTreesForwardSpec {nTrees maxDepth nFeatures : Nat}
  (model : GradientBoostedTreesSpec α nTrees maxDepth)
  (input : Tensor α (.dim nFeatures .scalar)) :
  Tensor α .scalar :=
  let rec accumulate_trees (i : Nat) (acc : α) : α :=
    if h : i < nTrees then
      match model.trees with
      | Tensor.dim trees =>
        match trees ⟨i, h⟩ with
        | Tensor.scalar tree =>
          let tree_prediction := decisionTreeForwardSpecN (α := α) (maxDepth := maxDepth)
            (nFeatures := nFeatures) tree input
          accumulate_trees (i + 1) (acc + model.learning_rate * tree_prediction)
    else acc
  let ensemble_prediction := accumulate_trees 0 model.initial_prediction
  Tensor.scalar ensemble_prediction

/-- Batched forward pass for a gradient boosted ensemble. -/
def gradientBoostedTreesBatchedForwardSpec {batch nTrees maxDepth nFeatures : Nat}
  (model : GradientBoostedTreesSpec α nTrees maxDepth)
  (input : Tensor α (.dim batch (.dim nFeatures .scalar))) :
  Tensor α (.dim batch .scalar) :=
  match input with
  | Tensor.dim batch_fn =>
    Tensor.dim (fun i => gradientBoostedTreesForwardSpec model (batch_fn i))

/--
"Gradient" w.r.t. a tree's prediction.

Decision trees are piecewise-constant in the inputs, so we do not attempt to define meaningful
derivatives through their internal decisions here. For boosting, this convention makes the intended
dataflow explicit: gradient information is used to fit subsequent trees, not to differentiate
through split predicates.
-/
def treePredictionGradSpec {maxDepth nFeatures : Nat}
  (_tree : DecisionTreeSpec α maxDepth)
  (_input : Tensor α (.dim nFeatures .scalar))
  (grad_output : α) :
  α :=
  -- For decision trees, the gradient is simply the gradient of the output
  -- since trees are piecewise constant functions
  grad_output

/--
Approximate gradient w.r.t. input features for a tree.

In this spec we return `0` gradients (trees are treated as non-differentiable).
-/
def treeInputGradSpec {maxDepth nFeatures : Nat}
  (_tree : DecisionTreeSpec α maxDepth)
  (_input : Tensor α (.dim nFeatures .scalar))
  (_grad_output : α) :
  Tensor α (.dim nFeatures .scalar) :=
  -- For decision trees, gradients w.r.t. inputs are typically zero
  -- since trees are piecewise constant. We return zero gradients.
  fill 0 (.dim nFeatures .scalar)

/--
Zero input-gradient convention for the ensemble.

This file treats boosted trees as a classical model: we do not backpropagate through tree
structure. Instead, residuals/gradients are used to fit *new* trees. This helper is intentionally
not wired into an `OpSpec`; callers should not mistake it for a differentiable surrogate.
-/
def gradientBoostedTreesZeroInputGradForNondiffTrees {nTrees maxDepth nFeatures : Nat}
  (_model : GradientBoostedTreesSpec α nTrees maxDepth)
  (_input : Tensor α (.dim nFeatures .scalar))
  (_grad_output : α) :
  Tensor α (.dim nFeatures .scalar) :=
  -- For gradient boosting, we typically don't backprop through the trees
  -- Instead, we use the gradients to fit new trees
  fill 0 (.dim nFeatures .scalar)

/-!
## Classical training: CART-style regression trees (MSE)

Tree-based models here are intended as **baselines** and **reference points**. For neural
models we implement reverse-mode explicitly; for trees we instead provide a classical (non-gradient)
training routine.

This section implements a small greedy CART-like procedure for *regression*:

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
  /-- x. -/
  x : Tensor α (.dim nFeatures .scalar)
  /-- y. -/
  y : α

namespace GradientBoostedTrees.Internal

/-- Sum all elements of a list. -/
def listSum (xs : List α) : α :=
  xs.foldl (fun acc x => acc + x) 0

/-- Mean of a list, with `0` as a convenient default for the empty list. -/
def meanOrZero (xs : List α) : α :=
  if xs.isEmpty then
    0
  else
    (listSum xs) / (xs.length : α)

/-- Sum of squared deviations from the mean (SSE). -/
def sse (ys : List α) : α :=
  if ys.isEmpty then
    0
  else
    let μ := meanOrZero ys
    ys.foldl (fun acc y =>
      let d := y - μ
      acc + d * d) 0

/-- Decide whether a sample goes to the *right* branch for a `(feature, threshold)` split. -/
def goesRight {nFeatures : Nat} (feature : Fin nFeatures) (threshold : α)
  (ex : RegressionExample (α := α) nFeatures) : Bool :=
  match ex.x with
  | Tensor.dim values =>
    match values feature with
    | Tensor.scalar v => decide (v > threshold)

/-- Partition samples into `(left, right)` for a given split. -/
def partitionBySplit {nFeatures : Nat}
  (feature : Fin nFeatures) (threshold : α) (xs : List (RegressionExample (α := α) nFeatures)) :
  List (RegressionExample (α := α) nFeatures) × List (RegressionExample (α := α) nFeatures) :=
  xs.partition (fun ex => !(goesRight feature threshold ex))

/-- Extract the regression targets from a list of examples. -/
def targets {nFeatures : Nat} (xs : List (RegressionExample (α := α) nFeatures)) : List α :=
  xs.map (fun ex => ex.y)

/--
Score a candidate split by sum of squared errors (SSE).

Returns `none` for degenerate splits (all samples go to one side).
-/
def splitScore {nFeatures : Nat}
  (feature : Fin nFeatures) (threshold : α)
  (xs : List (RegressionExample (α := α) nFeatures)) : Option (α × List (RegressionExample (α := α)
    nFeatures) × List (RegressionExample (α := α) nFeatures)) :=
  let (l, r) := partitionBySplit (α := α) feature threshold xs
  -- Disallow degenerate splits (one side empty). This keeps trees from “splitting” without
  -- learning.
  if l.isEmpty || r.isEmpty then
    none
  else
    some (sse (targets l) + sse (targets r), l, r)

end GradientBoostedTrees.Internal

open GradientBoostedTrees.Internal

/-- Find the best split `(feature, threshold)` by exhaustive search over observed thresholds. -/
def bestSplit {nFeatures : Nat}
  (xs : List (RegressionExample (α := α) nFeatures)) :
  Option (Fin nFeatures × α × List (RegressionExample (α := α) nFeatures) × List (RegressionExample
    (α := α) nFeatures) × α) :=
  (List.finRange nFeatures).foldl (fun best feature =>
    let thresholds : List α := xs.map (fun ex => Tensor.vecGet ex.x feature)
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
def leafValue {nFeatures : Nat} (xs : List (RegressionExample (α := α) nFeatures)) : α :=
  meanOrZero (targets xs)

/--
Fit a regression tree by greedy CART-style splitting (MSE/SSE), with a depth budget.

`depthLeft` counts how many splits we are still allowed to make.
-/
def fitRegressionNode {nFeatures : Nat} :
    Nat → List (RegressionExample (α := α) nFeatures) → TreeNode α
  | 0, xs => TreeNode.leaf (leafValue (α := α) xs)
  | (d+1), xs =>
    -- If there’s no meaningful split, stop at a leaf.
    match bestSplit (α := α) (nFeatures := nFeatures) xs with
    | none => TreeNode.leaf (leafValue (α := α) xs)
    | some (feature, threshold, l, r, score) =>
      let parentScore := sse (targets xs)
      -- Only split if it actually improves SSE.
      if Context.gtBool parentScore score then
        TreeNode.split feature.val threshold
          (fitRegressionNode d l)
          (fitRegressionNode d r)
      else
        TreeNode.leaf (leafValue (α := α) xs)

/-- Fit a regression decision tree from a batched dataset. -/
def decisionTreeFitRegressionMseSpec {batch maxDepth nFeatures : Nat}
  (x : Tensor α (.dim batch (.dim nFeatures .scalar)))
  (y : Tensor α (.dim batch .scalar)) :
  DecisionTreeSpec α maxDepth :=
  let examples : List (RegressionExample (α := α) nFeatures) :=
    (List.finRange batch).map (fun i =>
      { x := get x i, y := Tensor.toScalar (get y i) })
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
inductive ClassifierTreeNode (α β : Type) where
  | leaf (label : β) : ClassifierTreeNode α β
  | split (feature : Nat) (threshold : α) (left right : ClassifierTreeNode α β) : ClassifierTreeNode
    α β
deriving Inhabited

/-- Specification wrapper for a classification decision tree (numeric splits, label-valued leaves).
  -/
structure DecisionTreeClassifierSpec (α β : Type) (maxDepth : Nat) where
  /-- root. -/
  root : ClassifierTreeNode α β
  /-- max depth. -/
  max_depth : Nat := maxDepth
deriving Inhabited

/-- Forward pass for a classifier decision tree on an input vector of `nFeatures` features.

Branching convention:

- go right iff `(x[feature] > threshold)`,
- otherwise go left.

This mirrors a common "≤ goes left / > goes right" convention, but avoids needing a decidable `<`
for every `Context α` backend.
-/
def decisionTreeClassifyForwardSpecN {β : Type} {maxDepth nFeatures : Nat}
  (tree : DecisionTreeClassifierSpec α β maxDepth)
  (input : Tensor α (.dim nFeatures .scalar)) : β :=
  let rec traverse (node : ClassifierTreeNode α β) : β :=
    match node with
    | .leaf lbl => lbl
    | .split feature_idx threshold left right =>
      match input with
      | Tensor.dim values =>
        if h : feature_idx < nFeatures then
          match values ⟨feature_idx, h⟩ with
          | Tensor.scalar feature_value =>
            if decide (feature_value > threshold) then
              traverse right
            else
              traverse left
        else
          -- Out-of-range feature index: treat as a missing feature and fall back to the left
          -- branch.
          traverse left
  traverse tree.root

/-- A single classification training example: feature vector `x` and label `y`. -/
structure ClassificationExample (nFeatures : Nat) (β : Type) where
  /-- x. -/
  x : Tensor α (.dim nFeatures .scalar)
  /-- y. -/
  y : β

namespace GradientBoostedTrees.Internal

/-- Count how many times `lbl` appears in `ys`. -/
def countEq {β : Type} [DecidableEq β] (lbl : β) (ys : List β) : Nat :=
  ys.foldl (fun acc y => if y = lbl then acc + 1 else acc) 0

end GradientBoostedTrees.Internal

/-- Majority label with deterministic tie-breaking.

If there is a tie, we keep the earlier winner from the fold. This is intentional: it avoids
non-determinism and keeps the spec stable across backends.
-/
def majorityLabel {β : Type} [DecidableEq β] [Inhabited β] (ys : List β) : β :=
  match ys with
  | [] => default
  | _ =>
    let labels := ys.eraseDups
    labels.foldl (fun best lbl =>
      let cBest := countEq best ys
      let cLbl := countEq lbl ys
      if cLbl > cBest then lbl else best
    ) (labels.headD default)

namespace GradientBoostedTrees.Internal

/-- Gini impurity of a multiset of labels.

`gini(ys) = 1 - Σ_c p(c)^2` where `p(c)` is the empirical class frequency.

This is the standard CART impurity used by many tree classifiers.
-/
def gini {β : Type} [DecidableEq β] (ys : List β) : α :=
  if ys.isEmpty then
    0
  else
    let n : α := (ys.length : α)
    let labels := ys.eraseDups
    let sumSq :=
      labels.foldl (fun acc lbl =>
        let c : α := (countEq lbl ys : Nat)
        let p := c / n
        acc + (p * p)
      ) 0
    (1 : α) - sumSq

end GradientBoostedTrees.Internal

/-- Weighted Gini impurity: `|ys| * gini(ys)`. -/
def giniWeighted {β : Type} [DecidableEq β] (ys : List β) : α :=
  (ys.length : α) * gini ys

/-- Extract the labels from a list of classification examples. -/
def classTargets {nFeatures : Nat} {β : Type} (xs : List (ClassificationExample (α := α) nFeatures
  β)) : List β :=
  xs.map (fun ex => ex.y)

/-- Decide whether a classification sample goes right for a `(feature, threshold)` split. -/
def goesRightC {nFeatures : Nat} {β : Type} (feature : Fin nFeatures) (threshold : α)
  (ex : ClassificationExample (α := α) nFeatures β) : Bool :=
  match ex.x with
  | Tensor.dim values =>
    match values feature with
    | Tensor.scalar v => decide (v > threshold)

/-- Partition classification samples into `(left, right)` for a candidate split. -/
def partitionBySplitC {nFeatures : Nat} {β : Type}
  (feature : Fin nFeatures) (threshold : α) (xs : List (ClassificationExample (α := α) nFeatures β))
    :
  List (ClassificationExample (α := α) nFeatures β) × List (ClassificationExample (α := α) nFeatures
    β) :=
  xs.partition (fun ex => !(goesRightC feature threshold ex))

namespace GradientBoostedTrees.Internal

/--
Score a candidate classification split `(feature, threshold)` by weighted Gini impurity.

Returns `none` when the split is degenerate (one side is empty); otherwise returns the score and
the `(left, right)` partitions.
-/
def splitScoreC {nFeatures : Nat} {β : Type} [DecidableEq β]
  (feature : Fin nFeatures) (threshold : α)
  (xs : List (ClassificationExample (α := α) nFeatures β)) :
  Option (α × List (ClassificationExample (α := α) nFeatures β) × List (ClassificationExample (α :=
    α) nFeatures β)) :=
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
  (xs : List (ClassificationExample (α := α) nFeatures β)) :
  Option (Fin nFeatures × α × List (ClassificationExample (α := α) nFeatures β) × List
    (ClassificationExample (α := α) nFeatures β) × α) :=
  (List.finRange nFeatures).foldl (fun best feature =>
    let thresholds : List α := xs.map (fun ex => Tensor.vecGet ex.x feature)
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
    Nat → List (ClassificationExample (α := α) nFeatures β) → ClassifierTreeNode α β
  | 0, xs => .leaf (majorityLabel (β := β) (classTargets xs))
  | (d+1), xs =>
    match bestSplitC (β := β) (nFeatures := nFeatures) xs with
    | none => .leaf (majorityLabel (β := β) (classTargets xs))
    | some (feature, threshold, l, r, score) =>
      let parentScore := giniWeighted (β := β) (classTargets xs)
      if Context.gtBool parentScore score then
        .split feature.val threshold
          (fitClassificationNode (β := β) d l)
          (fitClassificationNode (β := β) d r)
      else
        .leaf (majorityLabel (β := β) (classTargets xs))

/--
Fit a classification decision tree (CART-style) using Gini impurity.

Labels are supplied as a list of length `batch`.

- If `y` is shorter than `batch`, missing entries use `default` (so the function remains total).
- If `y` is longer than `batch`, extra labels are ignored.

This “list-based labels” API is meant for demos and small experiments. If you have labels already
as a tensor or an index type, it is usually better to convert them to a list explicitly at the
boundary of your program so the conversion is visible.
-/
def decisionTreeFitClassificationGiniListSpec {β : Type} [DecidableEq β] [Inhabited β]
  {batch maxDepth nFeatures : Nat}
  (x : Tensor α (.dim batch (.dim nFeatures .scalar)))
  (y : List β) :
  DecisionTreeClassifierSpec α β maxDepth :=
  let examples : List (ClassificationExample (α := α) nFeatures β) :=
    (List.finRange batch).map (fun i =>
      { x := get x i, y := y.getD i.val default })
  { root := fitClassificationNode (β := β) (nFeatures := nFeatures) maxDepth examples }

-- Mean Squared Error loss for regression
/-- Mean squared error (MSE) loss for regression, reduced to a scalar by averaging over the batch.
  -/
def gbtMseLossSpec {batch nTrees maxDepth nFeatures : Nat}
  (model : GradientBoostedTreesSpec α nTrees maxDepth)
  (input : Tensor α (.dim batch (.dim nFeatures .scalar)))
  (target : Tensor α (.dim batch .scalar)) (h : batch ≠ 0) :
  Tensor α .scalar :=
  let predictions := gradientBoostedTreesBatchedForwardSpec model input
  let errors := subSpec predictions target
  let squared_errors := squareSpec errors
  have inst : Shape.valid_axis_inst 0 (Shape.dim batch Shape.scalar) := by
    apply Shape.validAxisInstZeroAlt h
  let mse := reduceSumAuto 0 squared_errors
  scaleSpec mse (1 / (batch : α))

/--
Binary cross-entropy loss for classification (with a sigmoid), reduced to a scalar by averaging.

This is a direct probability-space loss helper. For numerically sensitive classification pipelines,
prefer a stable "BCE with logits" implementation in the runtime/training layer.
-/
def gbtBinaryCrossentropyLossSpec {batch nTrees maxDepth nFeatures : Nat}
  (model : GradientBoostedTreesSpec α nTrees maxDepth)
  (input : Tensor α (.dim batch (.dim nFeatures .scalar)))
  (target : Tensor α (.dim batch .scalar)) (h : batch ≠ 0) :
  Tensor α .scalar :=
  let predictions := gradientBoostedTreesBatchedForwardSpec model input
  let sigmoid_preds := mapSpec (fun x => 1 / (1 + MathFunctions.exp (-x))) predictions
  let log_preds := mapSpec MathFunctions.log sigmoid_preds
  let log_one_minus_preds := mapSpec (fun x => MathFunctions.log (1 - x)) sigmoid_preds
  let loss1 := mulSpec target log_preds
  let loss2 := mulSpec (subSpec (broadcastLike target (Tensor.scalar (1 : α))) target)
    log_one_minus_preds
  let total_loss := subSpec loss1 loss2
  have inst : Shape.valid_axis_inst 0 (Shape.dim batch Shape.scalar) := by
    apply Shape.validAxisInstZeroAlt h
  let mse := reduceSumAuto 0 total_loss
  negSpec (scaleSpec mse (1 / (batch : α)))

/-- Gradient of MSE loss w.r.t. predictions (elementwise). -/
def gbtMseGradSpec {batch : Nat}
  (predictions : Tensor α (.dim batch .scalar))
  (target : Tensor α (.dim batch .scalar)) :
  Tensor α (.dim batch .scalar) :=
  let errors := subSpec predictions target
  scaleSpec errors (Numbers.two / (batch : α))

/-- Gradient of sigmoid binary cross-entropy loss w.r.t. predictions (elementwise). -/
def gbtBinaryCrossentropyGradSpec {batch : Nat}
  (predictions : Tensor α (.dim batch .scalar))
  (target : Tensor α (.dim batch .scalar)) :
  Tensor α (.dim batch .scalar) :=
  let sigmoid_preds := mapSpec (fun x => 1 / (1 + MathFunctions.exp (-x))) predictions
  subSpec sigmoid_preds target

/--
Residual computation for gradient boosting.

For squared-error regression, the residual is `target - prediction`.
-/
def computeResidualsSpec {batch nTrees maxDepth nFeatures : Nat}
  (model : GradientBoostedTreesSpec α nTrees maxDepth)
  (input : Tensor α (.dim batch (.dim nFeatures .scalar)))
  (target : Tensor α (.dim batch .scalar)) :
  Tensor α (.dim batch .scalar) :=
  let predictions := gradientBoostedTreesBatchedForwardSpec model input
  subSpec target predictions

/--
One gradient-boosting "add a tree" step, given a pre-fit `new_tree`.

This returns the current loss and the updated model with `new_tree` appended.
The residuals computed here are illustrative; the "fit a tree to residuals" variant below is
usually the more self-contained baseline.
-/
def gradientBoostedTreesTrainStepSpec {batch nTrees maxDepth nFeatures : Nat}
  (model : GradientBoostedTreesSpec α nTrees maxDepth)
  (input : Tensor α (.dim batch (.dim nFeatures .scalar)))
  (target : Tensor α (.dim batch .scalar))
  (new_tree : DecisionTreeSpec α maxDepth)
  (h : batch ≠ 0) :
  (Tensor α .scalar × GradientBoostedTreesSpec α (nTrees + 1) maxDepth) :=
  -- Compute current predictions
  let _predictions := gradientBoostedTreesBatchedForwardSpec model input
  -- Compute loss
  let loss := gbtMseLossSpec model input target h
  -- Compute residuals (negative gradients)
  let _residuals := computeResidualsSpec model input target
  -- Add new tree to ensemble
  let new_trees := concatVectorsSpec model.trees (Tensor.dim (fun _ => Tensor.scalar new_tree))
  let updated_model := {
    model with
    trees := new_trees
  }
  (loss, updated_model)

/-!
### Gradient boosting: a "fit-one-more-tree" step

The original `gradient_boosted_trees_train_step_spec` expects a pre-fit `new_tree`. For a more
complete baseline, we also provide a deterministic step that *fits* that tree to the residuals.
-/

/-- Fit a new tree to residuals and append it to the ensemble. -/
def gradientBoostedTreesTrainStepFitSpec {batch nTrees maxDepth nFeatures : Nat}
  (model : GradientBoostedTreesSpec α nTrees maxDepth)
  (input : Tensor α (.dim batch (.dim nFeatures .scalar)))
  (target : Tensor α (.dim batch .scalar))
  (h : batch ≠ 0) :
  (Tensor α .scalar × GradientBoostedTreesSpec α (nTrees + 1) maxDepth) :=
  let loss := gbtMseLossSpec model input target h
  let residuals := computeResidualsSpec model input target
  let newTree :=
    decisionTreeFitRegressionMseSpec (α := α) (batch := batch) (maxDepth := maxDepth)
      (nFeatures := nFeatures) input residuals
  let new_trees := concatVectorsSpec model.trees (Tensor.dim (fun _ => Tensor.scalar newTree))
  let updated_model := { model with trees := new_trees }
  (loss, updated_model)

namespace GradientBoostedTrees.Internal

/--
Increment a single feature counter by 1 inside a length-`nFeatures` vector.

This is used by the split-count feature-importance computation below.
-/
def incrFeature {nFeatures : Nat} (acc : Tensor α (.dim nFeatures .scalar)) (featureIdx : Nat) :
  Tensor α (.dim nFeatures .scalar) :=
  Tensor.dim (fun i =>
    match get acc i with
    | Tensor.scalar v =>
        if i.val = featureIdx then Tensor.scalar (v + (1 : α)) else Tensor.scalar v)

end GradientBoostedTrees.Internal

/--
Count how many times each feature index appears in split nodes of a tree.

This mirrors a very common "split count" importance heuristic.
-/
def treeFeatureCounts {nFeatures : Nat} : TreeNode α → Tensor α (.dim nFeatures .scalar) → Tensor
  α (.dim nFeatures .scalar)
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
  (model : GradientBoostedTreesSpec α nTrees maxDepth) :
  Tensor α (.dim nFeatures .scalar) :=
  let rec accumulate_importance (i : Nat) (acc : Tensor α (.dim nFeatures .scalar)) : Tensor α (.dim
    nFeatures .scalar) :=
    if h : i < nTrees then
      match model.trees with
      | Tensor.dim trees =>
        match trees ⟨i, h⟩ with
        | Tensor.scalar tree =>
          let acc' := treeFeatureCounts (α := α) (nFeatures := nFeatures) tree.root acc
          accumulate_importance (i + 1) acc'
    else acc
  let counts := accumulate_importance 0 (fill 0 (.dim nFeatures .scalar))
  let total : α := sumSpec counts
  if Context.gtBool total 0 then
    scaleSpec counts (1 / total)
  else
    counts

/--
Coefficient of determination (R^2) for regression.

This uses the standard formula `1 - ss_res / ss_tot`, written as `(ss_tot - ss_res) / ss_tot`
to avoid an explicit `1 - ...` when working in an abstract scalar context.
-/
def gbtRSquaredSpec {batch nTrees maxDepth nFeatures : Nat}
  (model : GradientBoostedTreesSpec α nTrees maxDepth)
  (input : Tensor α (.dim batch (.dim nFeatures .scalar)))
  (target : Tensor α (.dim batch .scalar)) (h : batch ≠ 0) :
  Tensor α .scalar :=
  let predictions := gradientBoostedTreesBatchedForwardSpec model input
  have inst : Shape.valid_axis_inst 0 (Shape.dim batch Shape.scalar) := by
    apply Shape.validAxisInstZeroAlt h
  let target_mean := reduceMeanAuto 0 inst target
  let target_mean_broadcast := broadcastLike target target_mean
  let ss_res := reduceSumAuto 0 (squareSpec (subSpec predictions target))
  let ss_tot := reduceSumAuto 0 (squareSpec (subSpec target target_mean_broadcast))
  -- Correct R-squared formula: (ss_tot - ss_res) / ss_tot
  divSpec (subSpec ss_tot ss_res) ss_tot

/-- Mean absolute error (MAE) for regression. -/
def gbtMaeSpec {batch nTrees maxDepth nFeatures : Nat}
  (model : GradientBoostedTreesSpec α nTrees maxDepth)
  (input : Tensor α (.dim batch (.dim nFeatures .scalar)))
  (target : Tensor α (.dim batch .scalar)) (h : batch ≠ 0) :
  Tensor α .scalar :=
  let predictions := gradientBoostedTreesBatchedForwardSpec model input
  let errors := absSpec (subSpec predictions target)
  have inst : Shape.valid_axis_inst 0 (Shape.dim batch Shape.scalar) := by
    apply Shape.validAxisInstZeroAlt h
  reduceMeanAuto 0 inst errors

/-- Root mean squared error (RMSE) for regression. -/
def gbtRmseSpec {batch nTrees maxDepth nFeatures : Nat}
  (model : GradientBoostedTreesSpec α nTrees maxDepth)
  (input : Tensor α (.dim batch (.dim nFeatures .scalar)))
  (target : Tensor α (.dim batch .scalar)) (h : batch ≠ 0) :
  Tensor α .scalar :=
  let predictions := gradientBoostedTreesBatchedForwardSpec model input
  let errors := subSpec predictions target
  let squared_errors := squareSpec errors
  have inst : Shape.valid_axis_inst 0 (Shape.dim batch Shape.scalar) := by
    apply Shape.validAxisInstZeroAlt h
  let mse := reduceMeanAuto 0 inst squared_errors
  sqrtSpec mse

/--
Loss-margin early-stopping predicate for gradient boosting.

This compares a training loss and validation loss with a margin `min_delta`.
The caller is responsible for tracking the patience counter; this predicate only checks one
train/validation loss pair.
-/
def earlyStoppingCheckSpec {batch nTrees maxDepth nFeatures : Nat}
  (model : GradientBoostedTreesSpec α nTrees maxDepth)
  (input : Tensor α (.dim batch (.dim nFeatures .scalar)))
  (target : Tensor α (.dim batch .scalar))
  (validation_input : Tensor α (.dim batch (.dim nFeatures .scalar)))
  (validation_target : Tensor α (.dim batch .scalar))
  (h : batch ≠ 0)
  (_patience : Nat)
  (min_delta : α) :
  Bool :=
  let train_loss := Tensor.toScalar (gbtMseLossSpec model input target h)
  let val_loss := Tensor.toScalar (gbtMseLossSpec model validation_input validation_target h)
  -- Single-step loss-margin check; patience is tracked by the caller.
  Context.gtBool (train_loss + min_delta) val_loss

/-- Adjust the ensemble learning rate (shrinkage) while keeping the same trees. -/
def adjustLearningRateSpec {_nTrees maxDepth : Nat}
  (model : GradientBoostedTreesSpec α _nTrees maxDepth)
  (new_rate : α) :
  GradientBoostedTreesSpec α _nTrees maxDepth :=
  { model with learning_rate := new_rate }

/--
Deterministic prefix selection used as a proof-friendly stand-in for stochastic subsampling.

Real stochastic GBDT implementations sample rows using randomness. This helper instead takes the
first `newBatch` rows and uses `h_new_batch` to make that access total, so it is deterministic and
does not silently pad with zeros.
-/
def prefixSubsampleDataSpec {batch newBatch nFeatures : Nat}
  (input : Tensor α (.dim batch (.dim nFeatures .scalar)))
  (target : Tensor α (.dim batch .scalar))
  (_subsample_ratio : α)
  (_h_ratio : _subsample_ratio > 0 ∧ _subsample_ratio ≤ 1)
  (h_new_batch : newBatch ≤ batch) :
  (Tensor α (.dim newBatch (.dim nFeatures .scalar)) × Tensor α (.dim newBatch .scalar)) :=
  let subsampled_input := Tensor.dim (fun i =>
    have h : i.val < batch := Nat.lt_of_lt_of_le i.isLt h_new_batch
    get input ⟨i.val, h⟩)
  let subsampled_target := Tensor.dim (fun i =>
    have h : i.val < batch := Nat.lt_of_lt_of_le i.isLt h_new_batch
    get target ⟨i.val, h⟩)
  (subsampled_input, subsampled_target)

/--
XGBoost-style squared-error proxy with an L2-shaped scalar penalty.

This objective is a typed loss for an already-materialized ensemble. It is not a full XGBoost
split-gain objective; tree-builder policies such as histogram binning and split search are
represented elsewhere by the tree-fitting routines.
-/
def xgboostSquaredErrorObjectiveSpec {batch nTrees maxDepth nFeatures : Nat}
  (model : GradientBoostedTreesSpec α nTrees maxDepth)
  (input : Tensor α (.dim batch (.dim nFeatures .scalar)))
  (target : Tensor α (.dim batch .scalar))
  (h : batch ≠ 0)
  (lambda : α)
  (_gamma : α) :
  Tensor α .scalar :=
  let _predictions := gradientBoostedTreesBatchedForwardSpec model input
  let mse := Tensor.toScalar (gbtMseLossSpec model input target h)
  -- Compact scalar proxy for an L2-style ensemble penalty.
  let regularization := lambda * mse
  Tensor.scalar (mse + regularization)

/--
LightGBM-style squared-error proxy with L1/L2-shaped scalar penalties.

This objective is deterministic because it operates on a fixed ensemble and batch. It records the
loss shape used by examples rather than the full LightGBM histogram/split objective.
-/
def lightgbmSquaredErrorObjectiveSpec {batch nTrees maxDepth nFeatures : Nat}
  (model : GradientBoostedTreesSpec α nTrees maxDepth)
  (input : Tensor α (.dim batch (.dim nFeatures .scalar)))
  (target : Tensor α (.dim batch .scalar))
  (lambda_l1 : α)
  (lambda_l2 : α)
  (h : batch ≠ 0) :
  Tensor α .scalar :=
  let _predictions := gradientBoostedTreesBatchedForwardSpec model input
  let mse := Tensor.toScalar (gbtMseLossSpec model input target h)
  -- Compact scalar proxy for L1/L2-style ensemble penalties.
  let regularization := lambda_l1 * mse + lambda_l2 * mse
  Tensor.scalar (mse + regularization)

end Spec
