/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Builtin.Proved.Correctness.Eval.Core

/-!
# Concat IR Evaluation

Local semantics for IR concat.  The evaluator keeps the generic-axis implementation in the shared
`Graph.evalConcat` helper, which moves the requested axis to the front, folds
`Tensor.concatAxisSpec`, and moves the result back. `LeadingAxisConcat.Input` packages an
input with its leading dimension, `LeadingAxisConcat.fold` specifies nonempty list concatenation,
and `evalAt_concat_leadingAxis_eq` proves end-to-end graph evaluation correct for every arity of at
least two.
-/

@[expose] public section

namespace NN.Verification.Builtin.Proved

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor
open NN.IR

namespace Correctness

namespace IRStep

/-- Local IR semantics for binary concat, pinned to the shared generic concat interpreter. -/
theorem evalAt_concat_binary_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {s₁ s₂ out : Shape} (axis : Nat)
    (lhs : Tensor α s₁) (rhs : Tensor α s₂) :
    Graph.evalAt (α := α)
        (g := binaryGraphOut (.concat axis) s₁ s₂ out)
        (payload := {})
        (input := Spec.SomeTensor.mk (α := α) s₁ lhs)
        (vals := #[
          Spec.SomeTensor.mk (α := α) s₁ lhs,
          Spec.SomeTensor.mk (α := α) s₂ rhs
        ]) (i := 2)
      =
      (Graph.evalConcat (α := α) 2 (binaryNodeOut (.concat axis) out) axis
          #[Spec.SomeTensor.mk (α := α) s₁ lhs, Spec.SomeTensor.mk (α := α) s₂ rhs]).bind
        (Graph.normalizeNodeOutput (α := α) 2 (binaryNodeOut (.concat axis) out)) := by
  simp [Graph.evalAt, Graph.evalNode, Graph.normalizeNodeOutput, binaryGraphOut, binaryNodeOut,
    Graph.getNode, Graph.getNode?,
    Graph.normalizeNodeOutput, Bind.bind, Except.bind, Pure.pure, Except.pure]

/--
Successful binary concat evaluation, once the shared concat interpreter has produced a value with
the node's declared output shape.
-/
theorem evalAt_concat_binary_ok
    {α : Type} [TorchLean.Storage α] [Context α]
    {s₁ s₂ out : Shape} (axis : Nat)
    (lhs : Tensor α s₁) (rhs : Tensor α s₂) (y : Tensor α out)
    (hConcat :
      Graph.evalConcat (α := α) 2 (binaryNodeOut (.concat axis) out) axis
          #[Spec.SomeTensor.mk (α := α) s₁ lhs, Spec.SomeTensor.mk (α := α) s₂ rhs] =
        .ok (Spec.SomeTensor.mk (α := α) out y)) :
    Graph.evalAt (α := α)
        (g := binaryGraphOut (.concat axis) s₁ s₂ out)
        (payload := {})
        (input := Spec.SomeTensor.mk (α := α) s₁ lhs)
        (vals := #[
          Spec.SomeTensor.mk (α := α) s₁ lhs,
          Spec.SomeTensor.mk (α := α) s₂ rhs
        ]) (i := 2)
      =
      .ok (Spec.SomeTensor.mk (α := α) out y) := by
  rw [evalAt_concat_binary_eq]
  rw [hConcat]
  simp [Graph.normalizeNodeOutput, binaryNodeOut, Except.bind, Pure.pure, Except.pure]

/-- Binary concat evaluation rejects the node whenever the shared concat interpreter rejects it. -/
theorem evalAt_concat_binary_error
    {α : Type} [TorchLean.Storage α] [Context α]
    {s₁ s₂ out : Shape} (axis : Nat)
    (lhs : Tensor α s₁) (rhs : Tensor α s₂) (msg : String)
    (hConcat :
      Graph.evalConcat (α := α) 2 (binaryNodeOut (.concat axis) out) axis
          #[Spec.SomeTensor.mk (α := α) s₁ lhs, Spec.SomeTensor.mk (α := α) s₂ rhs] =
        .error msg) :
    Graph.evalAt (α := α)
        (g := binaryGraphOut (.concat axis) s₁ s₂ out)
        (payload := {})
        (input := Spec.SomeTensor.mk (α := α) s₁ lhs)
        (vals := #[
          Spec.SomeTensor.mk (α := α) s₁ lhs,
          Spec.SomeTensor.mk (α := α) s₂ rhs
        ]) (i := 2)
      =
      .error msg := by
  rw [evalAt_concat_binary_eq]
  rw [hConcat]
  rfl

namespace LeadingAxisConcat

/-- Shapes with a common tail and the supplied leading dimensions. -/
def shapes (rest : Shape) (sizes : List Nat) : List Shape :=
  sizes.map (fun size => .dim size rest)

/-- Sum a nonempty sequence of leading dimensions from left to right. -/
def totalSize (head : Nat) (tail : List Nat) : Nat :=
  tail.foldl (· + ·) head

/-- A tensor whose leading dimension is existentially quantified while its tail shape stays
fixed. -/
abbrev Input (α : Type) [TorchLean.Storage α] [Context α] (rest : Shape) :=
  Sigma fun size => Tensor α (.dim size rest)

/-- Erase the leading-axis witness to the dynamic value consumed by the IR evaluator. -/
def Input.toSomeTensor {α : Type} [TorchLean.Storage α] [Context α] {rest : Shape}
    (input : Input α rest) : Spec.SomeTensor α :=
  Spec.SomeTensor.mk (α := α) (.dim input.1 rest) input.2

/-- Concatenate a nonempty sequence of compatible leading-axis inputs from left to right. -/
def fold {α : Type} [TorchLean.Storage α] [Context α] {rest : Shape}
    (head : Input α rest) (tail : List (Input α rest)) : Input α rest :=
  tail.foldl
    (fun acc next =>
      ⟨acc.1 + next.1,
        Tensor.concatAxisSpec (α := α) .scalar acc.2 next.2⟩)
    head

/-- The leading dimension of a concat fold is the left-associated sum of its input dimensions. -/
theorem fold_size {α : Type} [TorchLean.Storage α] [Context α] {rest : Shape}
    (head : Input α rest) (tail : List (Input α rest)) :
    (fold head tail).1 = (tail.map Sigma.fst).foldl (· + ·) head.1 := by
  induction tail generalizing head with
  | nil => rfl
  | cons next tail ih =>
      unfold fold
      rw [List.foldl_cons, List.map_cons, List.foldl_cons]
      simpa only [fold] using
        (ih (head :=
          ⟨head.1 + next.1,
            Tensor.concatAxisSpec (α := α) .scalar head.2 next.2⟩))

end LeadingAxisConcat

/-- Packaging and then decoding a typed leading-axis input preserves it exactly. -/
@[simp] theorem expectLeadingAxisInput_toSomeTensor
    {α : Type} [TorchLean.Storage α] [Context α]
    {rest : Shape} (i : Nat) (input : LeadingAxisConcat.Input α rest) :
    Graph.expectLeadingAxisInput (α := α) i rest input.toSomeTensor = .ok input := by
  rcases input with ⟨size, tensor⟩
  simp [Graph.expectLeadingAxisInput, LeadingAxisConcat.Input.toSomeTensor, Pure.pure, Except.pure]

/-- Decoding a list of typed leading-axis inputs after packaging preserves the whole list. -/
@[simp] theorem mapM_expectLeadingAxisInput_toSomeTensor
    {α : Type} [TorchLean.Storage α] [Context α]
    {rest : Shape} (i : Nat) (inputs : List (LeadingAxisConcat.Input α rest)) :
    (inputs.map LeadingAxisConcat.Input.toSomeTensor).mapM
        (Graph.expectLeadingAxisInput (α := α) i rest) =
      .ok inputs := by
  induction inputs with
  | nil => rfl
  | cons input inputs ih =>
      rw [List.map_cons, List.mapM_cons, expectLeadingAxisInput_toSomeTensor, ih]
      rfl

/--
The dynamic leading-axis evaluator agrees with the typed concat fold for every nonempty input
list. This single list-indexed result subsumes the former pair, triple, and quadruple theorems.
-/
theorem evalConcatLeadingAxisFold_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {rest : Shape} (i : Nat)
    (head : LeadingAxisConcat.Input α rest)
    (tail : List (LeadingAxisConcat.Input α rest)) :
    let result := LeadingAxisConcat.fold head tail
    Graph.evalConcatLeadingAxisFold (α := α) i result.1 rest
        ((head :: tail).map LeadingAxisConcat.Input.toSomeTensor).toArray
      = .ok (LeadingAxisConcat.Input.toSomeTensor result) := by
  unfold Graph.evalConcatLeadingAxisFold
  rw [List.mapM_toArray,
    mapM_expectLeadingAxisInput_toSomeTensor (α := α) i (head :: tail)]
  simp only [Functor.map, Except.map]
  simp only [Bind.bind, Except.bind]
  have hExtract : (head :: tail).toArray.extract 1 = tail.toArray := by
    simp
  rw [hExtract]
  simp only [List.getElem?_toArray, List.getElem?_cons_zero]
  rw [List.foldl_toArray]
  simp [LeadingAxisConcat.fold, LeadingAxisConcat.Input.toSomeTensor,
    Pure.pure, Except.pure]
  congr

/-- Leading-axis shape inference sums every input's leading dimension for any arity of at least
two. -/
theorem inferConcatOutShape_leadingAxis_eq
    {rest : Shape} (first second : Nat) (tail : List Nat) :
    OpContracts.inferConcatOutShape 0
        (LeadingAxisConcat.shapes rest (first :: second :: tail)).toArray =
      .ok (.dim (LeadingAxisConcat.totalSize first (second :: tail)) rest) := by
  have hSame (dimensions : List Nat) (index : Nat) (hindex : index ≠ 0) :
      OpContracts.mergeConcatDims 0 index dimensions dimensions = .ok dimensions := by
    induction dimensions generalizing index with
    | nil => rfl
    | cons dimension dimensions ih =>
        simp [OpContracts.mergeConcatDims, hindex, ih (index := index + 1) (by simp)]
  have hMerge (left right : Nat) :
      OpContracts.mergeConcatDims 0 0 (left :: rest.toList) (right :: rest.toList) =
        .ok ((left + right) :: rest.toList) := by
    simp [OpContracts.mergeConcatDims, hSame rest.toList 1 (by simp)]
  have hFold (sizes : List Nat) (total : Nat) :
      OpContracts.foldConcatDims 0 (total :: rest.toList)
          (LeadingAxisConcat.shapes rest sizes) =
        .ok (sizes.foldl (· + ·) total :: rest.toList) := by
    induction sizes generalizing total with
    | nil => rfl
    | cons size sizes ih =>
        simp only [LeadingAxisConcat.shapes, List.map_cons, OpContracts.foldConcatDims]
        change (do
          let dimensions ← OpContracts.mergeConcatDims 0 0
            (total :: rest.toList) (size :: rest.toList)
          OpContracts.foldConcatDims 0 dimensions
            (sizes.map (fun size => Shape.dim size rest))) = _
        rw [hMerge total size]
        simp only [Bind.bind, Except.bind]
        simpa only [LeadingAxisConcat.shapes, List.foldl_cons] using ih (total + size)
  have hAxis : OpContracts.checkAxisValid 0 (.dim first rest) = .ok () := by
    simp [OpContracts.checkAxisValid, Spec.Shape.rank]
    rfl
  have hFoldAll := hFold (second :: tail) first
  simp only [LeadingAxisConcat.shapes, List.map_cons] at hFoldAll
  simp only [OpContracts.inferConcatOutShape, LeadingAxisConcat.shapes, List.map_cons,
    hAxis, Bind.bind, Except.bind,
    Pure.pure, Except.pure]
  rw [show (Shape.dim first rest).toList = first :: rest.toList from rfl]
  rw [hFoldAll]
  simp [LeadingAxisConcat.totalSize, List.foldl_cons]

/-- The shared concat interpreter agrees with the typed fold for every arity of at least two. -/
theorem evalConcat_leadingAxis_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {rest : Shape} (i : Nat)
    (first second : LeadingAxisConcat.Input α rest)
    (tail : List (LeadingAxisConcat.Input α rest)) :
    let inputs := first :: second :: tail
    let result := LeadingAxisConcat.fold first (second :: tail)
    let parentShapes :=
      (inputs.map fun input => Shape.dim input.1 rest).toArray
    Graph.evalConcat (α := α) i
        (variadicNodeOut (.concat 0) parentShapes (.dim result.1 rest)) 0
        (inputs.map LeadingAxisConcat.Input.toSomeTensor).toArray =
      .ok result.toSomeTensor := by
  dsimp only
  have hInfer :
      OpContracts.inferConcatOutShape 0
          ((first :: second :: tail).map fun input => Shape.dim input.1 rest).toArray =
        .ok (.dim (LeadingAxisConcat.fold first (second :: tail)).1 rest) := by
    have hTailShapes :
        tail.map (fun input => Shape.dim input.1 rest) =
          (tail.map Sigma.fst).map (fun size => Shape.dim size rest) := by
      induction tail with
      | nil => rfl
      | cons input tail ih => simp only [List.map_cons, ih]
    simp only [List.map_cons]
    rw [hTailShapes]
    simpa only [LeadingAxisConcat.shapes, LeadingAxisConcat.totalSize,
      LeadingAxisConcat.fold_size, List.map_cons] using
      (inferConcatOutShape_leadingAxis_eq (rest := rest) first.1 second.1
        (tail.map Sigma.fst))
  have hSame :
      (Shape.dim (LeadingAxisConcat.fold first (second :: tail)).1 rest !=
        Shape.dim (LeadingAxisConcat.fold first (second :: tail)).1 rest) = false :=
    shapeBNe_refl _
  have hShapes :
      ((first :: second :: tail).map LeadingAxisConcat.Input.toSomeTensor).map
          Spec.SomeTensor.shape =
        (first :: second :: tail).map (fun input => Shape.dim input.1 rest) := by
    simp [LeadingAxisConcat.Input.toSomeTensor]
  have hShapesArray :
      ((first :: second :: tail).map LeadingAxisConcat.Input.toSomeTensor).toArray.map
          Spec.SomeTensor.shape =
        ((first :: second :: tail).map (fun input => Shape.dim input.1 rest)).toArray := by
    simpa using congrArg List.toArray hShapes
  unfold Graph.evalConcat
  rw [hShapesArray, hInfer]
  simp only [Bind.bind, Except.bind, Pure.pure, Except.pure, variadicNodeOut]
  rw [hSame]
  simp only [Bool.false_eq_true, ↓reduceIte]
  exact evalConcatLeadingAxisFold_eq (α := α) i first (second :: tail)

/--
End-to-end local IR semantics for leading-axis concat with any number of inputs greater than one.
The graph, parent values, inferred output dimension, and tensor result are all derived from the same
shape-erased input list.
-/
theorem evalAt_concat_leadingAxis_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {rest : Shape}
    (first second : LeadingAxisConcat.Input α rest)
    (tail : List (LeadingAxisConcat.Input α rest)) :
    let inputs := first :: second :: tail
    let result := LeadingAxisConcat.fold first (second :: tail)
    let parentShapes :=
      (inputs.map fun input => Shape.dim input.1 rest).toArray
    let values := (inputs.map LeadingAxisConcat.Input.toSomeTensor).toArray
    Graph.evalAt (α := α)
        (g := variadicGraphOut (.concat 0) parentShapes (.dim result.1 rest))
        (payload := {}) (input := first.toSomeTensor) (vals := values)
        (i := parentShapes.size) =
      .ok result.toSomeTensor := by
  dsimp only
  let inputs := first :: second :: tail
  let parentShapes := (inputs.map fun input => Shape.dim input.1 rest).toArray
  let values := (inputs.map LeadingAxisConcat.Input.toSomeTensor).toArray
  have hSize : values.size = parentShapes.size := by
    simp [values, parentShapes, inputs]
  have hParents :
      (Array.range parentShapes.size).mapM
          (Graph.getParentValue values parentShapes.size
            (variadicNodeOut (.concat 0) parentShapes
              (.dim (LeadingAxisConcat.fold first (second :: tail)).1 rest))) =
        .ok values := by
    rw [← hSize]
    apply rangeArray_mapM_eq_of_getElem_eq
    intro parentId hParentId
    simp [Graph.getParentValue, hParentId, Pure.pure, Except.pure]
  change Graph.evalAt (α := α)
      (g := variadicGraphOut (.concat 0) parentShapes
        (.dim (LeadingAxisConcat.fold first (second :: tail)).1 rest))
      (payload := {}) (input := first.toSomeTensor) (vals := values)
      (i := parentShapes.size) = _
  unfold Graph.evalAt
  rw [variadicGraphOut_getNode]
  simp only [Bind.bind, Except.bind]
  simp only [Graph.evalNode, Graph.evalNodeRaw, variadicNodeOut]
  simp only [variadicNodeOut] at hParents
  rw [hParents]
  simp only [Bind.bind, Except.bind]
  have hConcat :
      Graph.evalConcat (α := α) parentShapes.size
          { id := parentShapes.size
            parents := Array.range parentShapes.size
            kind := .concat 0
            outShape := .dim (LeadingAxisConcat.fold first (second :: tail)).1 rest }
          0 (inputs.map LeadingAxisConcat.Input.toSomeTensor).toArray =
        .ok (LeadingAxisConcat.fold first (second :: tail)).toSomeTensor := by
    simpa [inputs, parentShapes, variadicNodeOut] using
      (evalConcat_leadingAxis_eq (α := α) parentShapes.size first second tail)
  rw [hConcat]
  change Graph.normalizeNodeOutput (α := α) parentShapes.size
      (variadicNodeOut (.concat 0) parentShapes
        (.dim (LeadingAxisConcat.fold first (second :: tail)).1 rest))
      (LeadingAxisConcat.fold first (second :: tail)).toSomeTensor = _
  exact Graph.normalizeNodeOutput_declared _ _ _

end IRStep

end Correctness

end NN.Verification.Builtin.Proved
