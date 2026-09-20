/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Builtin.Proved.Correctness.WellFormed

/-!
# Lowered Forward Evaluation: Shared Invariants
-/

@[expose] public section

namespace NN.Verification.Builtin.Proved

open Spec TorchLean
open TorchLean.Tensor
open NN.IR

namespace Correctness

open NN.Verification.Builtin
-- Typed context indices come from `NN.Proofs.Autograd.Tape.Util.Idx`, the one place
-- `Idx` and `getIdx` are defined.
open Proofs (Idx getIdx)

namespace IRStep

/-- Reflexivity for the structural shape equality used by IR runtime guards. -/
theorem shapeBEq_refl (s : Shape) : (s == s) = true :=
  beq_self_eq_true s

/-- Reflexivity for the structural shape inequality used by IR runtime guards. -/
theorem shapeBNe_refl (s : Shape) : (s != s) = false := by
  simp [bne]

end IRStep

/-! ### Lowering correctness (forward fragment) -/

namespace IRStep

/-! ### Local graph constructors for evaluator lemmas -/

/-- A unary node with parent `0` and an explicit output shape. -/
def unaryNodeOut (kind : OpKind) (outShape : Shape) : NN.IR.Node :=
  { id := 1, parents := #[0], kind := kind, outShape := outShape }

/-- A two-node graph for a unary op with explicit input and output shapes. -/
def unaryGraphOut (kind : OpKind) (inShape outShape : Shape) : Graph :=
  { nodes := #[
      { id := 0, parents := #[], kind := .input, outShape := inShape },
      unaryNodeOut kind outShape
    ] }

/-- A unary node whose input and output share the same shape. -/
def unaryNode (kind : OpKind) (s : Shape) : NN.IR.Node :=
  { id := 1, parents := #[0], kind := kind, outShape := s }

/-- A two-node graph for a unary op whose input and output share the same shape. -/
def unaryGraph (kind : OpKind) (s : Shape) : Graph :=
  { nodes := #[
      { id := 0, parents := #[], kind := .input, outShape := s },
      unaryNode kind s
    ] }

/-- A binary node with parents `0` and `1` and an explicit output shape. -/
def binaryNodeOut (kind : OpKind) (outShape : Shape) : NN.IR.Node :=
  { id := 2, parents := #[0, 1], kind := kind, outShape := outShape }

/-- A three-node graph for a binary op with explicit parent and output shapes. -/
def binaryGraphOut (kind : OpKind) (leftShape rightShape outShape : Shape) : Graph :=
  { nodes := #[
      { id := 0, parents := #[], kind := .input, outShape := leftShape },
      { id := 1, parents := #[], kind := .input, outShape := rightShape },
      binaryNodeOut kind outShape
    ] }

/-- A binary node whose inputs and output share the same shape. -/
def binaryNode (kind : OpKind) (s : Shape) : NN.IR.Node :=
  { id := 2, parents := #[0, 1], kind := kind, outShape := s }

/-- A three-node graph for a binary op whose inputs and output share the same shape. -/
def binaryGraph (kind : OpKind) (s : Shape) : Graph :=
  { nodes := #[
      { id := 0, parents := #[], kind := .input, outShape := s },
      { id := 1, parents := #[], kind := .input, outShape := s },
      binaryNode kind s
    ] }

/-- A node consuming every preceding entry of a shape array, in order. -/
def variadicNodeOut (kind : OpKind) (parentShapes : Array Shape) (outShape : Shape) : NN.IR.Node :=
  { id := parentShapes.size
    parents := Array.range parentShapes.size
    kind := kind
    outShape := outShape }

/--
Graph fixture for an arbitrary-arity evaluator theorem: one input node per parent shape followed by
a node whose parent ids are the complete preceding range.
-/
def variadicGraphOut (kind : OpKind) (parentShapes : Array Shape) (outShape : Shape) : Graph :=
  { nodes := (parentShapes.mapIdx fun i shape =>
      { id := i, parents := #[], kind := .input, outShape := shape }).push
      (variadicNodeOut kind parentShapes outShape) }

/-- A failure-aware lookup reconstructs an array when it succeeds at every valid index. -/
theorem range_mapM_eq_toList_of_getElem_eq {β : Type} (values : Array β)
    (get : Nat → Except String β)
    (hget : ∀ i (hi : i < values.size), get i = Except.ok values[i]) :
    (List.range values.size).mapM get = Except.ok values.toList := by
  have hLookups :
      (List.range values.size).map get = values.toList.map Except.ok := by
    apply List.ext_getElem
    · simp
    · intro i hRange hValues
      have hi : i < values.size := by simpa using hRange
      simp [hget i hi, Array.getElem_toList]
  have aux : ∀ (ids : List Nat) (result : List β),
      ids.map get = result.map Except.ok → ids.mapM get = Except.ok result := by
    intro ids
    induction ids with
    | nil =>
        intro result hResult
        cases result with
        | nil => rfl
        | cons _ _ => simp at hResult
    | cons i ids ih =>
        intro result hResult
        cases result with
        | nil => simp at hResult
        | cons value result =>
            simp only [List.map_cons, List.cons.injEq] at hResult
            rw [List.mapM_cons, hResult.1, ih result hResult.2]
            rfl
  exact aux (List.range values.size) values.toList hLookups

/-- A failure-aware lookup reconstructs an array when it succeeds at every valid index. -/
theorem rangeArray_mapM_eq_of_getElem_eq {β : Type} (values : Array β)
    (get : Nat → Except String β)
    (hget : ∀ i (hi : i < values.size), get i = Except.ok values[i]) :
    (Array.range values.size).mapM get = Except.ok values := by
  rw [← List.toArray_range, List.mapM_toArray,
    range_mapM_eq_toList_of_getElem_eq values get hget]
  simp

/-- The final node of a variadic evaluator fixture is its variadic operation node. -/
@[simp] theorem variadicGraphOut_getNode
    (kind : OpKind) (parentShapes : Array Shape) (outShape : Shape) :
    (variadicGraphOut kind parentShapes outShape).getNode parentShapes.size =
      .ok (variadicNodeOut kind parentShapes outShape) := by
  let inputNodes : Array NN.IR.Node := parentShapes.mapIdx fun i shape =>
    { id := i, parents := #[], kind := OpKind.input, outShape := shape }
  have hSize : inputNodes.size = parentShapes.size := by simp [inputNodes]
  change ({ nodes := inputNodes.push (variadicNodeOut kind parentShapes outShape) } : Graph).getNode
      parentShapes.size = .ok (variadicNodeOut kind parentShapes outShape)
  rw [← hSize]
  simp [Graph.getNode, Graph.getNode?, variadicNodeOut,
    Bind.bind, Except.bind, Pure.pure, Except.pure]
  exact hSize.symm

end IRStep

/-- Evaluate a typed forward let-chain while accumulating every intermediate dynamic value. -/
def evalForwardLetChainVals
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (g : ForwardLetChain α paramShapes inShape ss out)
    (params : TorchLean.TensorPack α paramShapes)
    (vals : Array (Spec.SomeTensor α)) : Except String (Array (Spec.SomeTensor α)) :=
  match g with
  | .ret _y =>
      pure vals
  | .let1 (ss := ss) (mid := mid) (out := out) node gNext => do
      let vOut ←
        evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out := mid)
          node params vals
      evalForwardLetChainVals (α := α) (paramShapes := paramShapes) (inShape := inShape)
        (ss := ss ++ [mid]) (out := out) gNext params (vals.push vOut)

/-- `Graph.expectShape` returns the stored tensor when the dynamic shape tag matches. -/
theorem expectShape_eq_ok
    {α : Type} [TorchLean.Storage α] [Context α]
    {expected : Shape} (v : Spec.SomeTensor α) (h : v.shape = expected) :
    NN.IR.Graph.expectShape (α := α) (expected := expected) v =
      Except.ok (v.cast h) := by
  cases h
  simp [NN.IR.Graph.expectShape, Spec.SomeTensor.cast, Pure.pure, Except.pure]

/-- A successful shape check certifies the shape stored by the packed tensor. -/
theorem shape_eq_of_expectShape_eq_ok
    {α : Type} [TorchLean.Storage α] [Context α]
    {expected : Shape} {v : Spec.SomeTensor α} {t : Tensor α expected}
    (h : NN.IR.Graph.expectShape (α := α) (expected := expected) v = Except.ok t) :
    v.shape = expected := by
  by_contra hShape
  simp [NN.IR.Graph.expectShape, hShape] at h

/-- `getVal` returns the indexed tensor when the runtime value carries the expected shape tag. -/
theorem getVal_eq_ok
    {α : Type} [TorchLean.Storage α] [Context α]
    {inShape : Shape} {ss : List Shape} {expected : Shape}
    (vals : Array (Spec.SomeTensor α)) (idx : Idx (Ctx inShape ss) expected)
    (v : Spec.SomeTensor α) (hSome : vals[idx.id]? = some v)
    (h : v.shape = expected) :
    getVal (α := α) (inShape := inShape) (ss := ss) (s := expected) vals idx =
      Except.ok (v.cast h) := by
  cases h
  simp [getVal, getValue?, hSome, Spec.SomeTensor.cast, Bind.bind, Except.bind,
    Pure.pure, Except.pure]

/-- Shape lookup through `shapesOfVals` agrees with looking up the dynamic value first. -/
theorem shapesOfVals_get?_eq
    {α : Type} [TorchLean.Storage α] [Context α] (vals : Array (Spec.SomeTensor α)) (i : Nat) :
    (shapesOfVals (α := α) vals)[i]? = (vals[i]?).map (fun v => v.1) := by
  -- Avoid `simp` loops on `Array.getElem?_eq_toList_get?'`.
  have hToList : vals.toList[i]? = vals[i]? := by
    simp
  simp [shapesOfVals, List.getElem?_map, hToList]

/-- `shapesOfVals` has one entry per runtime value. -/
@[simp] theorem shapesOfVals_length
    {α : Type} [TorchLean.Storage α] [Context α] (vals : Array (Spec.SomeTensor α)) :
    (shapesOfVals (α := α) vals).length = vals.size := by
  simp [shapesOfVals]

/-- A shape-context invariant proves that a typed index is in bounds for the value array. -/
theorem index_lt_of_shapesOfVals_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {inShape : Shape} {ss : List Shape} {s : Shape}
    (vals : Array (Spec.SomeTensor α)) (idx : Idx (Ctx inShape ss) s)
    (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss) : idx.id < vals.size := by
  have hLen : vals.size = (Ctx inShape ss).length := by
    simpa [shapesOfVals_length] using congrArg List.length hShapes
  have hiΓ : idx.id < (Ctx inShape ss).length := idx_id_lt_length (x := idx)
  simpa [hLen] using hiΓ

/-- The packed runtime value selected by a typed index and a matching shape context. -/
def packedAt
    {α : Type} [TorchLean.Storage α] [Context α]
    {inShape : Shape} {ss : List Shape} {s : Shape}
    (vals : Array (Spec.SomeTensor α)) (idx : Idx (Ctx inShape ss) s)
    (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss) : Spec.SomeTensor α :=
  vals[idx.id]'(index_lt_of_shapesOfVals_eq vals idx hShapes)

/-- Safe array lookup returns the packed value selected by `packedAt`. -/
theorem getElem?_eq_some_packedAt
    {α : Type} [TorchLean.Storage α] [Context α]
    {inShape : Shape} {ss : List Shape} {s : Shape}
    (vals : Array (Spec.SomeTensor α)) (idx : Idx (Ctx inShape ss) s)
    (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss) :
    vals[idx.id]? = some (packedAt vals idx hShapes) := by
  simp [packedAt, index_lt_of_shapesOfVals_eq vals idx hShapes]

/-- The packed value selected by a typed index carries the statically expected shape. -/
@[simp] theorem packedAt_shape
    {α : Type} [TorchLean.Storage α] [Context α]
    {inShape : Shape} {ss : List Shape} {s : Shape}
    (vals : Array (Spec.SomeTensor α)) (idx : Idx (Ctx inShape ss) s)
    (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss) :
    (packedAt vals idx hShapes).shape = s := by
  classical
  have hLen : vals.size = (Ctx inShape ss).length := by
    simpa [shapesOfVals_length] using congrArg List.length hShapes
  have hiΓ : idx.id < (Ctx inShape ss).length := idx_id_lt_length (x := idx)
  have hFin : (⟨idx.id, hiΓ⟩ : Fin (Ctx inShape ss).length) = idx.i := by
    apply Fin.ext
    rfl
  have hGetElem : (Ctx inShape ss)[idx.id]'hiΓ = s := by
    -- `l[i]'h` is definitional `l.get ⟨i,h⟩`.
    simpa [Idx.id, List.get, hFin] using idx.h
  have hΓOpt : (Ctx inShape ss)[idx.id]? = some s := by
    have hSome : (Ctx inShape ss)[idx.id]? = some ((Ctx inShape ss)[idx.id]'hiΓ) := by
      simp
    simp [hSome, hGetElem]
  have hShapesAt :
      (shapesOfVals (α := α) vals)[idx.id]? = some s := by
    have hEq : (shapesOfVals (α := α) vals)[idx.id]? = (Ctx inShape ss)[idx.id]? :=
      congrArg (fun l => l[idx.id]?) hShapes
    exact Eq.trans hEq hΓOpt
  -- Convert to an Option statement about the Array lookup.
  have hAt :
      (vals[idx.id]?).map (fun v => v.1) = some s := by
    simpa [shapesOfVals_get?_eq] using hShapesAt
  have hSome : vals[idx.id]? = some (packedAt vals idx hShapes) :=
    getElem?_eq_some_packedAt vals idx hShapes
  -- Extract the shape from the mapped option.
  simpa [hSome] using hAt

/-- The typed tensor selected by an index into a runtime context with the expected shapes. -/
def tensorAt
    {α : Type} [TorchLean.Storage α] [Context α]
    {inShape : Shape} {ss : List Shape} {s : Shape}
    (vals : Array (Spec.SomeTensor α)) (idx : Idx (Ctx inShape ss) s)
    (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss) : Tensor α s :=
  (packedAt vals idx hShapes).cast (packedAt_shape vals idx hShapes)

/-- Packing the typed tensor recovered from a well-shaped context returns the original value. -/
theorem ofTensor_tensorAt_eq_packedAt
    {α : Type} [TorchLean.Storage α] [Context α]
    {inShape : Shape} {ss : List Shape} {s : Shape}
    (vals : Array (Spec.SomeTensor α)) (idx : Idx (Ctx inShape ss) s)
    (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss) :
    Spec.SomeTensor.ofTensor (tensorAt vals idx hShapes) = packedAt vals idx hShapes := by
  let value := packedAt vals idx hShapes
  have hShape : value.shape = s := packedAt_shape vals idx hShapes
  change Spec.SomeTensor.ofTensor (value.cast hShape) = value
  exact Spec.SomeTensor.ofTensor_cast value hShape

/-- Shape checking succeeds for the typed tensor extracted from a well-shaped context. -/
theorem expectShape_packedAt_eq_ok
    {α : Type} [TorchLean.Storage α] [Context α]
    {inShape : Shape} {ss : List Shape} {s : Shape}
    (vals : Array (Spec.SomeTensor α)) (idx : Idx (Ctx inShape ss) s)
    (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss) :
    NN.IR.Graph.expectShape (α := α) (expected := s) (packedAt vals idx hShapes) =
      Except.ok (tensorAt vals idx hShapes) := by
  simpa [tensorAt] using
    expectShape_eq_ok (expected := s) (packedAt vals idx hShapes)
      (packedAt_shape vals idx hShapes)

/--
`getVal` succeeds from a well-shaped executable context.

This is the proof layer form of `getVal_eq_ok`: callers use the semantic invariant
`shapesOfVals vals = Ctx inShape ss`, and the lemma derives the array-bounds fact internally.
-/
theorem getVal_eq_ok_of_shapesOfVals_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {inShape : Shape} {ss : List Shape} {expected : Shape}
    (vals : Array (Spec.SomeTensor α)) (idx : Idx (Ctx inShape ss) expected)
    (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss) :
    getVal (α := α) (inShape := inShape) (ss := ss) (s := expected) vals idx =
      Except.ok (tensorAt vals idx hShapes) :=
  getVal_eq_ok vals idx (packedAt vals idx hShapes)
    (getElem?_eq_some_packedAt vals idx hShapes) (packedAt_shape vals idx hShapes)

end Correctness

end NN.Verification.Builtin.Proved
