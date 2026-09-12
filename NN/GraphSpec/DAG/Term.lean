/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.DAG.Primitives.Core
public import NN.GraphSpec.DAG.Semantics

/-!
# DAG Term Combinators

Reusable typed term constructors whose semantics are stated independently of graph syntax.
-/

@[expose] public section

namespace NN
namespace GraphSpec
namespace DAG

open Spec TorchLean
open TorchLean.Tensor

namespace Term


/-- Add a list of same-shaped terms, starting from the all-zero tensor. -/
def sum {Γ : List Shape} (s : Shape) (terms : List (Term Γ s)) : Term Γ s :=
  terms.foldl
    (fun total term => Term.op (PrimOp.add s) (.cons total (.cons term .nil)))
    (Term.op (PrimOp.zero s) .nil)

/-- Pure evaluation commutes with `Term.sum`. -/
theorem eval_sum {Γ : List Shape} {s : Shape} {α : Type} [TorchLean.Storage α] [Context α]
    (env : TorchLean.TensorPack α Γ) (terms : List (Term Γ s)) :
    Term.eval env (sum s terms) =
      terms.foldl (fun total term => TorchLean.Tensor.addSpec total (Term.eval env term))
        (Tensor.full s 0) := by
  unfold sum
  let stepTerm : Term Γ s → Term Γ s → Term Γ s := fun total term =>
    Term.op (PrimOp.add s) (.cons total (.cons term .nil))
  let stepValue : TorchLean.Tensor α s → Term Γ s → TorchLean.Tensor α s :=
    fun total term =>
    TorchLean.Tensor.addSpec total (Term.eval env term)
  have eval_foldl : ∀ (xs : List (Term Γ s)) (initial : Term Γ s),
      Term.eval env (xs.foldl stepTerm initial) =
        xs.foldl stepValue (Term.eval env initial) := by
    intro xs
    induction xs with
    | nil => intro initial; rfl
    | cons term rest ih =>
        intro initial
        simp only [List.foldl_cons]
        rw [ih]
        rfl
  simpa [stepTerm, stepValue, Term.eval, Term.evalArgs, PrimOp.zero] using
    eval_foldl terms (Term.op (PrimOp.zero s) .nil)



end Term

end DAG
end GraphSpec
end NN
