/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# Decision Trees (Small Spec-Only Baseline)

This file is small on purpose: it provides a pure decision tree datatype and an evaluator.
It is a standalone baseline and does not use the tensor/module APIs.

Why include it here?

- It serves as a non-neural "sanity check" baseline for experiments (e.g. mixed pipelines).
- The datatype is small enough to be convenient in proofs or examples without pulling in tensor
  machinery.

If you're thinking in the Python ecosystem: this is not an `nn.Module`-style component. It is
closer to a symbolic scikit-learn style decision tree, except we keep it fully pure and explicit.

References / analogies:
- Breiman, Friedman, Olshen, Stone, "Classification and Regression Trees" (CART), 1984.
- Quinlan, "Induction of Decision Trees" (ID3), 1986; and "C4.5: Programs for Machine Learning",
  1993.
- scikit-learn user guide (for intuition, not semantics):
  https://scikit-learn.org/stable/modules/tree.html
-/

@[expose] public section

/-- A small decision tree datatype (spec-only baseline).

`node feature left right` branches on `decisionFn feature`.
-/
inductive DecisionTree (α : Type) : Type
| leaf (value : α) : DecisionTree α
| node (feature : String) (left right : DecisionTree α) : DecisionTree α

open DecisionTree

-- Function to traverse the decision tree given a feature evaluation function
/-- Evaluate a decision tree using a Boolean predicate for each feature name. -/
def evaluate {α : Type} (tree : DecisionTree α) (decisionFn : String → Bool) : α :=
  match tree with
  | leaf value      => value
  | node feature left right =>
      if decisionFn feature then evaluate left decisionFn
      else evaluate right decisionFn
