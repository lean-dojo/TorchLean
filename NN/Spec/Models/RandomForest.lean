/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Models.GradientBoostedTrees
public import NN.Spec.Module.DecisionTree

/-!
# Random Forest

This file defines a lightweight random-forest wrapper around a list of decision trees from
`NN.Spec.Module.DecisionTree`, plus a couple of standard aggregation strategies:

- classification: majority vote
- regression: average

This is a small reference implementation intended to be:

- deterministic (no RNG hidden inside the definitions),
- easy to read and audit,
- suitable for examples and baseline comparisons.

Ecosystem analogies:

- scikit-learn: `RandomForestClassifier` / `RandomForestRegressor`
- PyTorch: there is no built-in random forest in `torch.nn`; forests are typically handled via
  scikit-learn or dedicated libraries.
-/

public section


namespace random_forest
open DecisionTree

-- Random Forest structure
/-- Random-forest container: a list of trees.

Aggregation (vote/average/etc.) is handled by `predict` so the container stays generic.
-/
structure RandomForest (α : Type) where
  /-- Forest members. We keep a `List` for simplicity. -/
  trees : List (DecisionTree α)

/--
Function to predict using all trees in the forest and combine results with a given aggregation
  function.
For classification, this would typically be a majority vote.
For regression, this would be an average.
-/
def predict {a : Type} (forest : RandomForest a) (decisionFn : String → Bool) (aggregateFn : List a
  → a) : a :=
  let predictions := forest.trees.map (fun tree => evaluate tree decisionFn)
  aggregateFn predictions

/--
For classification tasks, majority vote aggregation function.
Returns the most frequent element in the list of predictions.
Works with any type that supports ordering.
-/
def majorityVote {a : Type} [Ord a] [Inhabited a] (predictions : List a) : Option a :=
  if predictions.isEmpty then
    none
  else
    -- Count frequencies using an ordered map so results are deterministic.
    let grouped := predictions.foldl
      (fun acc pred =>
        acc.insert pred (acc.findD pred 0 + 1))
      (Lean.RBMap.empty : Lean.RBMap a Nat compare)

    -- Pick the element with highest frequency.
    let groupedList := grouped.toList
    some (groupedList.foldl
      (fun best cur => if cur.2 > best.2 then cur else best)
      (groupedList.headD (default, 0))
      |>.1)

/--
For regression tasks, average aggregation function.
-/
def average {α : Type} [Zero α] [Add α] [Div α] [Coe Nat α] (predictions : List α) : α :=
  if predictions.isEmpty then
    0
  else
    predictions.foldl (fun sum pred => sum + pred) 0 / (predictions.length : α)

/-!
## Numeric random forest (spec baseline)

The `RandomForest` above wraps the symbolic `DecisionTree` from `NN.Spec.Module.DecisionTree`, where
splits are keyed by `String` feature names and an external `decisionFn : String → Bool` decides the
branch. That is handy for demos, but it is not something we can “train” without providing
feature-value semantics.

For a more classical baseline, we also provide a *numeric* random forest built on the
`Spec.DecisionTreeSpec` representation used by `gradient_boosted_trees.lean`:

- features are indexed by `Nat`
- splits compare a feature value to a threshold
- training uses a deterministic greedy CART-style routine (MSE) and a deterministic “bootstrap”
  resampling scheme (rotation) so it stays pure and reproducible.

This is still a spec/reference implementation: correctness and readability come first.
-/

namespace Numeric

open Spec

variable {α : Type} [Context α]

/-- A regression random forest: an ensemble of regression trees averaged at inference time. -/
structure RegressionForestSpec (α : Type) (nTrees maxDepth nFeatures : Nat) where
  /-- trees. -/
  trees : Tensor (Spec.DecisionTreeSpec α maxDepth) (.dim nTrees .scalar)

/-- Forward pass: average tree predictions.

This corresponds to `RandomForestRegressor.predict` (mean over tree outputs).
-/
def regressionForestForwardSpec {nTrees maxDepth nFeatures : Nat}
  (model : RegressionForestSpec α nTrees maxDepth nFeatures)
  (x : Tensor α (.dim nFeatures .scalar)) : Tensor α .scalar :=
  if _h0 : nTrees = 0 then
    Tensor.scalar 0
  else
    let rec go (i : Nat) (acc : α) : α :=
      if h : i < nTrees then
        match model.trees with
        | Tensor.dim trees =>
          match trees ⟨i, h⟩ with
          | Tensor.scalar t =>
            let yi := Spec.decisionTreeForwardSpecN (α := α) (maxDepth := maxDepth) (nFeatures :=
              nFeatures) t x
            go (i + 1) (acc + yi)
      else
        acc / (nTrees : α)
    Tensor.scalar (go 0 0)

/--
Fit a regression random forest by training each tree on a deterministic “bootstrap” resample.

Why rotate instead of RNG-based bootstrapping?
- it keeps the whole spec deterministic and pure (important for proofs/verification)
- it still exercises the same API shape as bagging: each tree sees a different multiset

If you want true randomness, treat this as the spec and implement an executable wrapper that
supplies a randomized index mapping.
-/
def regressionForestFitRegressionMseSpec {batch nTrees maxDepth nFeatures : Nat}
  (x : Tensor α (.dim batch (.dim nFeatures .scalar)))
  (y : Tensor α (.dim batch .scalar))
  (hBatch : batch ≠ 0) :
  RegressionForestSpec α nTrees maxDepth nFeatures :=
  let makeBootstrapX (k : Fin nTrees) : Tensor α (.dim batch (.dim nFeatures .scalar)) :=
    Tensor.dim (fun i =>
      let j : Fin batch :=
        ⟨(i.val + k.val) % batch, by
          exact Nat.mod_lt _ (Nat.pos_of_ne_zero hBatch)⟩
      get x j)
  let makeBootstrapY (k : Fin nTrees) : Tensor α (.dim batch .scalar) :=
    Tensor.dim (fun i =>
      let j : Fin batch :=
        ⟨(i.val + k.val) % batch, by
          exact Nat.mod_lt _ (Nat.pos_of_ne_zero hBatch)⟩
      get y j)
  let trees : Tensor (Spec.DecisionTreeSpec α maxDepth) (.dim nTrees .scalar) :=
    Tensor.dim (fun k => Tensor.scalar (Spec.decisionTreeFitRegressionMseSpec (α := α)
      (batch := batch) (maxDepth := maxDepth) (nFeatures := nFeatures)
      (makeBootstrapX k) (makeBootstrapY k)))
  { trees := trees }

/-!
### Classification forest (Gini)

This mirrors the regression forest, but uses the classifier-tree type from
`NN/Spec/Models/GradientBoostedTrees.lean` so leaf values can be arbitrary labels (`β`).
-/

/-- Count how many times label `lbl` appears in `ys`. -/
private def countEq {β : Type} [DecidableEq β] (lbl : β) (ys : List β) : Nat :=
  ys.foldl (fun acc y => if y = lbl then acc + 1 else acc) 0

/-- Deterministic majority label of a list.

Tie-breaking: we keep the first label (in the `eraseDups` order) that attains the maximal count.
-/
private def majorityLabel {β : Type} [DecidableEq β] [Inhabited β] (ys : List β) : β :=
  match ys with
  | [] => default
  | _ =>
    let labels := ys.eraseDups
    labels.foldl (fun best lbl =>
      let cBest := countEq best ys
      let cLbl := countEq lbl ys
      if cLbl > cBest then lbl else best
    ) (labels.headD default)

/-- A classification random forest: an ensemble of classifier trees (majority vote). -/
structure ClassificationForestSpec (α β : Type) (nTrees maxDepth nFeatures : Nat) where
  /-- trees. -/
  trees : Tensor (Spec.DecisionTreeClassifierSpec α β maxDepth) (.dim nTrees .scalar)

/-- Predict by majority vote across trees. -/
def classificationForestPredictSpec {β : Type} [DecidableEq β] [Inhabited β]
  {nTrees maxDepth nFeatures : Nat}
  (model : ClassificationForestSpec α β nTrees maxDepth nFeatures)
  (x : Tensor α (.dim nFeatures .scalar)) : β :=
  if _h0 : nTrees = 0 then
    default
  else
    let preds : List β :=
      (List.finRange nTrees).map (fun i =>
        match get model.trees i with
        | Tensor.scalar t =>
          Spec.decisionTreeClassifyForwardSpecN (α := α) (β := β)
            (maxDepth := maxDepth) (nFeatures := nFeatures) t x)
    majorityLabel (β := β) preds

/-- Fit a classification forest using deterministic “rotation bootstraps” and Gini-based CART trees.
  -/
def classificationForestFitClassificationGiniSpec {β : Type} [DecidableEq β] [Inhabited β]
  {batch nTrees maxDepth nFeatures : Nat}
  (x : Tensor α (.dim batch (.dim nFeatures .scalar)))
  (y : List β)
  (hBatch : batch ≠ 0) :
  ClassificationForestSpec α β nTrees maxDepth nFeatures :=
  let makeBootstrapX (k : Fin nTrees) : Tensor α (.dim batch (.dim nFeatures .scalar)) :=
    Tensor.dim (fun i =>
      let j : Fin batch :=
        ⟨(i.val + k.val) % batch, by
          exact Nat.mod_lt _ (Nat.pos_of_ne_zero hBatch)⟩
      get x j)
  let makeBootstrapY (k : Fin nTrees) : List β :=
    (List.finRange batch).map (fun i =>
      y.getD ((i.val + k.val) % batch) default)
  let trees : Tensor (Spec.DecisionTreeClassifierSpec α β maxDepth) (.dim nTrees .scalar) :=
    Tensor.dim (fun k => Tensor.scalar (Spec.decisionTreeFitClassificationGiniListSpec (α :=
      α) (β := β)
      (batch := batch) (maxDepth := maxDepth) (nFeatures := nFeatures)
      (makeBootstrapX k) (makeBootstrapY k)))
  { trees := trees }

end Numeric

end random_forest
