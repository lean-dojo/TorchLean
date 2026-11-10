/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Context
public import TorchLean.Tensor.Layers.Softmax
public import Std.Data.HashMap

/-!
# Multinomial Naive Bayes

A pure multinomial Naive Bayes classifier over `String` features and labels, using Lean's
`HashMap` for the fitted count tables. It is a non-neural baseline that spells out what training
and prediction compute. Its data are strings and hash maps, without the tensor shapes used by most
of `TorchLean/NN/Models/*`.

Probabilities are computed in log space (via `MathFunctions.log`) to avoid underflow.

Ecosystem note:
PyTorch has no Naive Bayes classifier in `torch.nn`. The closest ecosystem analogue is
scikit-learn’s `MultinomialNB`.

References:

- scikit-learn 1.9.1 user guide, Section 1.9.2 (Multinomial Naive Bayes), and
  `_update_class_log_prior` in `sklearn/naive_bayes.py`.
  https://scikit-learn.org/1.9/modules/naive_bayes.html

## What "training" means here

Naive Bayes is a *counting* model: training collects label and feature counts from the dataset.
The API keeps fitting and inference separate, so examples can show where counts are learned and
where predictions are made.

The API has a `fit` step that produces a `NaiveBayes`, plus:
- `predict` for inference using the fitted counts
- `negLogLikelihood` as a standard training objective (useful for evaluation/comparison)

## Implementation status

No neural API builder implements this model, and no model-correctness theorem is proved here.
-/

public section


open Std
open FloatLib.Numerics (MathFunctions)

namespace TorchLean.NaiveBayes

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

/-- One training example: a bag-of-words feature multiset and a class label. -/
structure Example where
  /-- The feature tokens of the example, kept with duplicates so the array acts as a multiset. -/
  features : Array String
  /-- The class label of the example. -/
  label : String
deriving Repr

/-- Count how many times each label appears in the dataset. -/
private def countLabels (data : Array Example) : HashMap String Nat :=
  data.foldl (fun acc ex =>
    acc.insert ex.label (acc.getD ex.label 0 + 1)
  ) {}

/-- Count how many times each feature appears *within each label*. -/
private def countFeaturesPerLabel (data : Array Example) : HashMap String (HashMap String Nat) :=
  data.foldl (fun acc ex =>
    let labelMap := acc.getD ex.label {}
    let updated := ex.features.foldl (fun m f =>
      m.insert f (m.getD f 0 + 1)
    ) labelMap
    acc.insert ex.label updated
  ) {}

/-- Total number of feature occurrences per label (sum of the per-feature counts). -/
private def totalFeatureCounts (counts : HashMap String (HashMap String Nat)) : HashMap String Nat
  :=
  counts.map (fun _ fmap => fmap.fold (fun acc _ v => acc + v) 0)

/-- Collect the vocabulary as an array of distinct feature strings. -/
private def distinctFeatures (data : Array Example) : Array String :=
  let seen : HashMap String Unit := data.foldl (fun seen ex =>
    ex.features.foldl (fun seen feature => seen.insert feature ()) seen) {}
  seen.keysArray

/-!
## Fitted model

`NaiveBayes` stores the counts and some precomputed bookkeeping derived from the dataset. None of it
depends on the scalar type `α`, which enters only when scoring turns counts into smoothed
probabilities (log-space scores).
-/

end TorchLean.NaiveBayes

/--
Fitted multinomial Naive Bayes model.

It stores raw counts plus a little derived bookkeeping (`labels`, `vocab`, `totalExamples`).
Scoring functions turn these counts into Laplace-smoothed log probabilities on demand.
-/
structure TorchLean.NaiveBayes where
  /-- Number of training examples seen for each label. -/
  labelCounts : HashMap String Nat
  /-- For each label, the number of occurrences of each feature token. -/
  featureCounts : HashMap String (HashMap String Nat)
  /-- For each label, the total number of feature tokens across its examples. -/
  totalCounts : HashMap String Nat
  /-- The distinct labels seen during fitting. Their order breaks ties in `predict`. -/
  labels : Array String
  /-- The distinct feature tokens seen during fitting. -/
  vocab : Array String
  /-- The total number of training examples. -/
  totalExamples : Nat

namespace TorchLean.NaiveBayes

/-- Collect counts from a nonempty training set.

An empty bag of words is legal: its prediction depends only on the class priors. An empty
training set is different, since it supplies no classes and therefore defines no posterior.
The order of `labels` is the key order of the label-count hash map.
-/
def fit (data : Array Example) : Except Error NaiveBayes :=
  if data.isEmpty then .error .emptyTrainingData else
  let labelCounts := countLabels data
  let featureCounts := countFeaturesPerLabel data
  let totalCounts := totalFeatureCounts featureCounts
  let labels := labelCounts.keysArray
  let vocab := distinctFeatures data
  .ok { labelCounts, featureCounts, totalCounts, labels, vocab, totalExamples := data.size }

/-!
## Scoring and prediction

The scoring rule is the standard multinomial NB rule with Laplace smoothing:

- prior: `(count(label)+1) / (N + nLabels)`
- conditional: `(count(feature,label)+1) / (totalFeatures(label) + vocabularySize)`

Scores are in log space, and prediction needs only their relative order.

We wrote the scorer by hand. The conditional is scikit-learn's estimate
`θ̂_yi = (N_yi + α) / (N_y + α n)` at its default `alpha = 1.0`, with one vocabulary of size `n`
shared by every label. The prior is where we differ: scikit-learn fits the plain class frequency,
and we add one to every label count, so 3 of 10 examples under 2 labels give `4 / 12` here and
`3 / 10` there.
-/

/-- Log prior probability `log P(lbl)` with Laplace smoothing. -/
private def logPrior {α : Type} [Context α] (m : NaiveBayes) (lbl : String) : α :=
  MathFunctions.log (((m.labelCounts.getD lbl 0 + 1) : α) /
    ((m.totalExamples + m.labels.size) : α))

/-- Log conditional probability `log P(f | lbl)` with Laplace smoothing. -/
private def logCond {α : Type} [Context α] (m : NaiveBayes) (lbl : String)
    (f : String) : α :=
  let countF := m.featureCounts.getD lbl {} |>.getD f 0
  let totalF := m.totalCounts.getD lbl 0
  MathFunctions.log (((countF + 1) : α) / ((totalF + m.vocab.size) : α))

/-- Unnormalized log score `log P(lbl) + Σ log P(f|lbl)` for a bag of features. -/
private def scoreUnchecked {α : Type} [Context α]
    (m : NaiveBayes) (input : Array String)
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
  (m : NaiveBayes)
  (input : Array String)
  (α : Type) [Context α] : String :=
  match m.labels[0]? with
  | none => ""
  | some lbl0 =>
      let initScore := scoreUnchecked (α := α) m input lbl0
      m.labels.foldl (fun (bestLbl, bestScore) lbl =>
        let sc := scoreUnchecked (α := α) m input lbl
        if Context.gtBool sc bestScore then (lbl, sc) else (bestLbl, bestScore)
      ) (lbl0, initScore) |>.fst

/-- Check that the model has labels and that every input token is in its fixed vocabulary, before
counts become probabilities.

An unknown token gives `unknownFeature`. A smoothed probability for it would need a larger
vocabulary size, and with it a different event space.
-/
private def validateInput (m : NaiveBayes) (input : Array String) : Except Error Unit := do
  if m.labels.isEmpty then throw .noLabels
  for feature in input do
    unless m.vocab.contains feature do throw (.unknownFeature feature)

/-- Compute a log score for a fitted class and a bag of known feature tokens. -/
def score {α : Type} [Context α]
    (m : NaiveBayes) (input : Array String) (label : String) : Except Error α := do
  validateInput m input
  unless m.labels.contains label do throw (.unknownLabel label)
  return scoreUnchecked m input label

/-- Predict the fitted label with the highest score, reporting a model without labels or an
unknown token as an `Error`.

When several labels share the highest score, the first of them in the model's `labels` wins. The
empty string is a valid class label and never signals failure.
-/
def predict (m : NaiveBayes) (input : Array String)
    (α : Type) [Context α] : Except Error String := do
  validateInput m input
  return predictUnchecked m input α

/-!
## Training objective (negative log-likelihood)

The standard objective for evaluating NB models is

`- Σ log P(y_i | x_i)`

NB training is closed-form counting and never optimizes it with gradients, but the objective is
useful for:
- checking improvements (smoothing choices, feature engineering)
- comparing NB against other baselines
- unit tests / runtime checks
-/

/-- Negative log-likelihood over fitted classes and known feature tokens.

Each example contributes `-log P(y | x)`, the log-sum-exp of its class scores taken relative to the
score of its target. Every target must belong to the class support used in that normalization, and
an unknown target gives `unknownLabel`, since its loss could be negative. Empty evaluation data has
loss zero for a model with at least one fitted class.
-/
def negLogLikelihood {α : Type} [Context α] (m : NaiveBayes)
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
    let fiber : Tensor α [relativeLogits.size] :=
      Tensor.ofFn fun i => relativeLogits[i]
    loss := loss + Tensor.item (Tensor.castShape
      (Tensor.logsumexp fiber [⟨0, Nat.zero_lt_one⟩] false)
      (Shape.reduce_singleton _ _))
  return loss

end TorchLean.NaiveBayes
