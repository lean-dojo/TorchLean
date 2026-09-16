/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Context
import Std.Data.HashMap.AdditionalOperations
public import NN.Tensor.Internal.Representation.Storage

/-!
# Multinomial Naive Bayes

This module gives a pure multinomial Naive Bayes classifier over `String` features and labels,
using Lean's `HashMap` for the fitted count tables. It is not a tensor-indexed neural model like
most of `NN/Spec/Models/*`; it is a non-neural baseline that keeps the training and prediction
semantics explicit.

Probabilities are computed in log space (via `MathFunctions.log`) to avoid underflow.

Ecosystem note:
PyTorch does not provide a Naive Bayes classifier in `torch.nn`; the closest ecosystem analogue is
scikit-learn’s `MultinomialNB`.

## What "training" means here

Naive Bayes is a *counting* model: training is just collecting label and feature counts from the
dataset. The API keeps fitting and inference separate, so examples can show exactly where counts
are learned and where predictions are made.

The API exposes an explicit `fit` step that produces a `Model`, plus:
- `predict` for inference using the fitted counts
- `negLogLikelihood` as a standard training objective (useful for evaluation/comparison)

## Implementation status

No neural API builder implements this model, and no model-correctness theorem is proved here.
-/

public section


open Std

namespace NaiveBayes

/-- Inputs for which the fitted count tables do not define the requested posterior. -/
inductive Error where
  /-- Fitting needs at least one labeled example. Feature bags themselves may be empty. -/
  | emptyTrainingData
  /-- A manually constructed model has no class labels. -/
  | noLabels
  /-- Evaluation requested a class outside the fitted label support. -/
  | unknownLabel (label : String)
  /-- Inference supplied a token outside the fitted vocabulary. -/
  | unknownFeature (feature : String)
  deriving Repr, BEq

-- A training example is a bag of features (multiset) and a label
/-- One training example: a bag-of-words feature multiset and a class label. -/
structure Example where
  /-- The feature tokens of the example; duplicates are kept so it acts as a multiset. -/
  features : Array String  -- multiset-like: allows duplicates
  /-- The class label of the example. -/
  label : String
deriving Repr

-- Count occurrences of each label.
/-- Count how many times each label appears in the dataset. -/
private def countLabels (data : Array Example) : HashMap String Nat :=
  data.foldl (fun acc ex =>
    acc.insert ex.label (acc.getD ex.label 0 + 1)
  ) {}

-- Count occurrences of each feature per label.
/-- Count how many times each feature appears *within each label*. -/
private def countFeaturesPerLabel (data : Array Example) : HashMap String (HashMap String Nat) :=
  data.foldl (fun acc ex =>
    let labelMap := acc.getD ex.label {}
    let updated := ex.features.foldl (fun m f =>
      m.insert f (m.getD f 0 + 1)
    ) labelMap
    acc.insert ex.label updated
  ) {}

-- Total number of features seen for each label.
/-- Total number of feature occurrences per label (sum of the per-feature counts). -/
private def totalFeatureCounts (counts : HashMap String (HashMap String Nat)) : HashMap String Nat
  :=
  counts.map (fun _ fmap => fmap.fold (fun acc _ v => acc + v) 0)

-- Get all distinct features in the dataset.
/-- Collect the vocabulary as an array of distinct feature strings. -/
private def distinctFeatures (data : Array Example) : Array String :=
  let seen : HashMap String Unit := data.foldl (fun seen ex =>
    ex.features.foldl (fun seen feature => seen.insert feature ()) seen) {}
  seen.keysArray

/-!
## Fitted model

`Model` stores the counts and some precomputed bookkeeping derived from the dataset.
Nothing here depends on the scalar type `α`; we only need `α` when we turn counts into smoothed
probabilities (log-space scores).
-/

/--
Fitted multinomial Naive Bayes model.

This stores raw counts plus a little derived bookkeeping (`labels`, `vocab`, `totalExamples`).
Scoring functions turn these counts into Laplace-smoothed log probabilities on demand.
-/
structure Model where
  /-- Number of training examples seen for each label. -/
  labelCounts : HashMap String Nat
  /-- For each label, the number of occurrences of each feature token. -/
  featureCounts : HashMap String (HashMap String Nat)
  /-- For each label, the total number of feature tokens across its examples. -/
  totalCounts : HashMap String Nat
  /-- The distinct labels seen during fitting. -/
  labels : Array String
  /-- The distinct feature tokens seen during fitting. -/
  vocab : Array String
  /-- The total number of training examples. -/
  totalExamples : Nat

/-- Collect counts from a nonempty training set.

An empty bag of words is legal: its prediction depends only on the class priors. An empty
training set is different, since it supplies no classes and therefore defines no posterior.
-/
def fit (data : Array Example) : Except Error Model :=
  if data.isEmpty then .error .emptyTrainingData else
  let labelCounts := countLabels data
  let featureCounts := countFeaturesPerLabel data
  let totalCounts := totalFeatureCounts featureCounts
  let labels := labelCounts.keysArray
  let vocab := distinctFeatures data
  .ok { labelCounts, featureCounts, totalCounts, labels, vocab, totalExamples := data.size }

/-- Vocabulary size (number of distinct features). -/
private def vocabularySize (m : Model) : Nat := m.vocab.size
/-- Number of distinct labels. -/
private def nLabels (m : Model) : Nat := m.labels.size

/-!
## Scoring and prediction

We use the standard multinomial NB scoring rule (with Laplace smoothing):

- prior: `(count(label)+1) / (N + nLabels)`
- conditional: `(count(feature,label)+1) / (totalFeatures(label) + vocabularySize)`

Scores are in log space. For prediction we only need relative ordering.
-/

/-- Log prior probability `log P(lbl)` with Laplace smoothing. -/
private def logPrior {α : Type} [TorchLean.Storage α] [Context α] (m : Model) (lbl : String) : α :=
  MathFunctions.log (((m.labelCounts.getD lbl 0 + 1) : α) /
    ((m.totalExamples + nLabels m) : α))

/-- Log conditional probability `log P(f | lbl)` with Laplace smoothing. -/
private def logCond {α : Type} [TorchLean.Storage α] [Context α] (m : Model) (lbl : String)
    (f : String) : α :=
  let countF := m.featureCounts.getD lbl {} |>.getD f 0
  let totalF := m.totalCounts.getD lbl 0
  MathFunctions.log (((countF + 1) : α) / ((totalF + vocabularySize m) : α))

/-- Unnormalized log score `log P(lbl) + Σ log P(f|lbl)` for a bag of features. -/
private def scoreUnchecked {α : Type} [TorchLean.Storage α] [Context α]
    (m : Model) (input : Array String)
    (lbl : String) : α :=
  let prior := logPrior (α := α) m lbl
  -- A repeated word contributes its count times the same log probability. Count first,
  -- then add that contribution once, so a long document does not accumulate one rounding
  -- error for every occurrence. Distinct words retain their first-occurrence order.
  let counts : HashMap String Nat := input.foldl (fun acc f =>
    acc.insert f (acc.getD f 0 + 1)) {}
  let (_, cond) := input.foldl (fun (seen, sum) f =>
    if seen.contains f then (seen, sum)
    else (seen.insert f (), sum + (counts.getD f 0 : α) * logCond (α := α) m lbl f))
    (({} : HashMap String Unit), (0 : α))
  prior + cond

private def predictUnchecked
  (m : Model)
  (input : Array String)
  (α : Type) [TorchLean.Storage α] [Context α] : String :=
  match m.labels[0]? with
  | none => ""
  | some lbl0 =>
      let initScore := scoreUnchecked (α := α) m input lbl0
      m.labels.foldl (fun (bestLbl, bestScore) lbl =>
        let sc := scoreUnchecked (α := α) m input lbl
        if Context.gtBool sc bestScore then (lbl, sc) else (bestLbl, bestScore)
      ) (lbl0, initScore) |>.fst

/-- Check the fixed vocabulary before converting counts into probabilities.

Unknown tokens are rejected explicitly. They cannot be assigned a smoothed probability while
leaving the vocabulary size unchanged, since that would use a different event space.
-/
private def validateInput (m : Model) (input : Array String) : Except Error Unit := do
  if m.labels.isEmpty then throw .noLabels
  for feature in input do
    unless m.vocab.contains feature do throw (.unknownFeature feature)

/-- Compute a log score for a fitted class and a bag of known feature tokens. -/
def score {α : Type} [TorchLean.Storage α] [Context α]
    (m : Model) (input : Array String) (label : String) : Except Error α := do
  validateInput m input
  unless m.labels.contains label do throw (.unknownLabel label)
  return scoreUnchecked m input label

/-- Predict a fitted label, reporting an unfitted model or an unknown token explicitly.

The empty string remains a valid class label; it is never used to signal failure.
-/
def predict (m : Model) (input : Array String)
    (α : Type) [TorchLean.Storage α] [Context α] : Except Error String := do
  validateInput m input
  return predictUnchecked m input α

/-!
## Training objective (negative log-likelihood)

This is the standard objective used to evaluate NB models:

`- Σ log P(y_i | x_i)`

Even though we don't optimize it with gradients (NB training is closed-form counting), having this
objective is useful for:
- checking improvements (smoothing choices, feature engineering)
- comparing NB against other baselines
- unit tests / runtime checks
-/

/-- Sum an array by left-folding with `+` (used by `logSumExp`). -/
private def arraySum {α : Type} [Add α] [Zero α] (xs : Array α) : α :=
  xs.foldl (fun acc x => acc + x) 0

/-- Numerically stable `log (sum_i exp xs[i])`. -/
private def logSumExp {α : Type} [TorchLean.Storage α] [Context α] (xs : Array α) : α :=
  -- Numerically-stable log-sum-exp:
  --   log Σ exp(x_i) = m + log Σ exp(x_i - m), where m = max_i x_i.
  match xs[0]? with
  | none => MathFunctions.log (0 : α)
  | some x0 =>
      let m :=
        xs.foldl (fun cur x => if x > cur then x else cur) x0
      let s := arraySum (xs.map (fun x => MathFunctions.exp (x - m)))
      m + MathFunctions.log s

/-- Negative log-likelihood over fitted classes and known feature tokens.

Every target label belongs to the same class support used in the normalization. Unknown
targets are rejected, rather than producing a value that can be negative. Empty evaluation
data has loss zero for a model with at least one fitted class.
-/
def negLogLikelihood {α : Type} [TorchLean.Storage α] [Context α] (m : Model)
    (data : Array Example) : Except Error α := do
  if m.labels.isEmpty then throw .noLabels
  let mut loss := 0
  for ex in data do
    validateInput m ex.features
    unless m.labels.contains ex.label do throw (.unknownLabel ex.label)
    let logits := m.labels.map (fun lbl => scoreUnchecked (α := α) m ex.features lbl)
    let trueLogit := scoreUnchecked (α := α) m ex.features ex.label
    -- Normalize relative to the true class before log-sum-exp. Adding a large common
    -- document score and subtracting it again would erase a small posterior loss.
    let relativeLogits := logits.map (fun logit => logit - trueLogit)
    loss := loss + logSumExp (α := α) relativeLogits
  return loss

end NaiveBayes
