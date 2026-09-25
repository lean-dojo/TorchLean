/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphConcatTensor
public import NN.Tensor.Internal.Laws.Sequence
public import NN.Verification.Builtin.Proved.Correctness.Eval.Concat
public import NN.Proofs.Tensor.Basic.Core

/-!
# Flattening concatenated tensor families

The CROWN layout and shaped tensor concatenation use the same parent-occurrence coordinate
bijection. Flattening transports that bijection through the row-major coordinate equivalences.
The proofs allow empty slices and repeated values.
-/

public section

namespace NN.MLTheory.CROWN.Graph

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor
open NN.Verification.Builtin.Proved.Correctness.IRStep

attribute [local simp] Bind.bind Pure.pure Except.bind Except.pure NN.IR.throw_eq_error
  Option.bind Except.toOption Shape.toList Shape.ofList Shape.dim

private theorem leadingAxisConcat_assoc {α : Type} [TorchLean.Storage α] [Context α]
    {trailing : Shape} (a b c : LeadingAxisConcat.Input α trailing) :
    (⟨(a.1 + b.1) + c.1,
      Tensor.concatAxisSpec [] (Tensor.concatAxisSpec [] a.2 b.2) c.2⟩ :
        LeadingAxisConcat.Input α trailing) =
      ⟨a.1 + (b.1 + c.1),
        Tensor.concatAxisSpec [] a.2 (Tensor.concatAxisSpec [] b.2 c.2)⟩ := by
  apply Sigma.ext (Nat.add_assoc a.1 b.1 c.1)
  have h := heq_of_eq (Tensor.Internal.Rep.concatenateAxis_assoc [] trailing a.2 b.2 c.2)
  simp only [Tensor.Internal.Rep.castShape, eqRec_heq_iff] at h
  simpa only [Tensor.concatAxisSpec, Tensor.Internal.Rep.castShape, Shape.toList,
    eqRec_heq_iff] using h

private theorem leadingAxisConcat_fold_cons {α : Type} [TorchLean.Storage α] [Context α]
    {trailing : Shape} (head next : LeadingAxisConcat.Input α trailing)
    (tail : List (LeadingAxisConcat.Input α trailing)) :
    LeadingAxisConcat.fold head (next :: tail) =
      ⟨head.1 + (LeadingAxisConcat.fold next tail).1,
        Tensor.concatAxisSpec [] head.2 (LeadingAxisConcat.fold next tail).2⟩ := by
  let op (a b : LeadingAxisConcat.Input α trailing) : LeadingAxisConcat.Input α trailing :=
    ⟨a.1 + b.1, Tensor.concatAxisSpec [] a.2 b.2⟩
  let : Std.Associative op := ⟨leadingAxisConcat_assoc⟩
  exact List.foldl_assoc (op := op)

/-- The existing typed left fold implements the independent family coordinate map. -/
theorem leadingAxisConcat_fold_ofFn {α : Type} [TorchLean.Storage α] [Context α]
    (trailing : Shape) (first : Nat) (lengths : List Nat)
    (values : (parent : Fin (first :: lengths).length) →
      Tensor α ((first :: lengths).get parent :: trailing)) :
    LeadingAxisConcat.fold ⟨first, values 0⟩
        (List.ofFn fun parent : Fin lengths.length =>
          (⟨lengths.get parent, values parent.succ⟩ : LeadingAxisConcat.Input α trailing)) =
      ⟨(first :: lengths).sum,
        Tensor.Internal.Rep.concatenateAxes [] trailing (first :: lengths) values⟩ := by
  induction lengths generalizing first with
  | nil =>
      simp [LeadingAxisConcat.fold, Tensor.Internal.Rep.concatenateAxes_singleton]
  | cons next lengths ih =>
      rw [List.ofFn_succ, leadingAxisConcat_fold_cons]
      erw [ih next (fun parent => values parent.succ)]
      simp [Tensor.Internal.Rep.concatenateAxes_cons, Tensor.concatAxisSpec,
        Tensor.Internal.Rep.castShape]

/-- The IR's leading-axis evaluator returns the family concatenation for every nonempty family. -/
theorem evalConcatLeadingAxisFold_family {α : Type} [TorchLean.Storage α] [Context α]
    (id : Nat) (trailing : Shape) (first : Nat) (lengths : List Nat)
    (values : (parent : Fin (first :: lengths).length) →
      Tensor α ((first :: lengths).get parent :: trailing)) :
    NN.IR.Graph.evalConcatLeadingAxisFold id (first :: lengths).sum trailing
        (Array.ofFn fun parent =>
          SomeTensor.mk ((first :: lengths).get parent :: trailing) (values parent)) =
      .ok (SomeTensor.mk ((first :: lengths).sum :: trailing)
        (Tensor.Internal.Rep.concatenateAxes [] trailing (first :: lengths) values)) := by
  have h := evalConcatLeadingAxisFold_eq id
    (⟨first, values 0⟩ : LeadingAxisConcat.Input α trailing)
    (List.ofFn fun parent : Fin lengths.length =>
      (⟨lengths.get parent, values parent.succ⟩ : LeadingAxisConcat.Input α trailing))
  rw [leadingAxisConcat_fold_ofFn] at h
  dsimp only at h
  have hparents :
      ((⟨first, values 0⟩ : LeadingAxisConcat.Input α trailing) ::
        List.ofFn (fun parent : Fin lengths.length =>
          (⟨lengths.get parent, values parent.succ⟩ : LeadingAxisConcat.Input α trailing))) =
        List.ofFn (fun parent : Fin (first :: lengths).length =>
          (⟨(first :: lengths).get parent, values parent⟩ :
            LeadingAxisConcat.Input α trailing)) := by
    rw [List.ofFn_succ]
    rfl
  rw [hparents] at h
  simpa only [List.map_ofFn, List.toArray_ofFn, LeadingAxisConcat.Input.toSomeTensor,
    Function.comp_def, Shape.dim] using h

/-- Shape agreement recovers a typed family from an actual array of runtime values. -/
theorem exists_tensor_family_of_shapes {α : Type} [TorchLean.Storage α]
    {count : Nat} (shapes : Fin count → Shape) (parents : Array (SomeTensor α))
    (h : parents.map SomeTensor.shape = Array.ofFn shapes) :
    ∃ values : (parent : Fin count) → Tensor α (shapes parent),
      parents = Array.ofFn (fun parent => SomeTensor.ofTensor (values parent)) := by
  have hsize : parents.size = count := by
    simpa using congrArg Array.size h
  let value (parent : Fin count) : SomeTensor α :=
    parents[parent.val]'(by simp [hsize])
  have hshape (parent : Fin count) : (value parent).shape = shapes parent := by
    have hget := congrArg (fun entries : Array Shape => entries[parent.val]!) h
    simpa [value, hsize, parent.isLt] using hget
  refine ⟨fun parent => (value parent).cast (hshape parent), ?_⟩
  apply Array.ext (by simp [hsize])
  intro index hleft hright
  rw [Array.getElem_ofFn]
  exact (SomeTensor.ofTensor_cast (value ⟨index, by simpa [hsize] using hleft⟩) _).symm

private theorem mergeConcatDims_pastAxis (axis index : Nat) (dimensions : List Nat)
    (hindex : axis < index) :
    NN.IR.OpContracts.mergeConcatDims axis index dimensions dimensions = .ok dimensions := by
  induction dimensions generalizing index with
  | nil => rfl
  | cons dimension dimensions ih =>
      simp [NN.IR.OpContracts.mergeConcatDims, Nat.ne_of_gt hindex,
        ih (index + 1) (by omega)]

private theorem mergeConcatDims_pastAxis_ok (axis index : Nat)
    (expected actual output : List Nat) (hindex : axis < index)
    (h : NN.IR.OpContracts.mergeConcatDims axis index expected actual = .ok output) :
    actual = expected ∧ output = expected := by
  induction expected generalizing index actual output with
  | nil =>
      cases actual with
      | nil => simpa [NN.IR.OpContracts.mergeConcatDims] using h.symm
      | cons dimension dimensions => simp [NN.IR.OpContracts.mergeConcatDims] at h
  | cons dimension dimensions ih =>
      cases actual with
      | nil => simp [NN.IR.OpContracts.mergeConcatDims] at h
      | cons other others =>
          cases htail : NN.IR.OpContracts.mergeConcatDims axis (index + 1)
              dimensions others with
          | error message =>
              simp [NN.IR.OpContracts.mergeConcatDims, htail] at h
          | ok tail =>
              obtain ⟨rfl, rfl⟩ := ih (index + 1) others tail (by omega) htail
              by_cases heq : other = dimension
              · subst other
                simpa [NN.IR.OpContracts.mergeConcatDims, htail,
                  Nat.ne_of_gt hindex] using h.symm
              · simp [NN.IR.OpContracts.mergeConcatDims, htail,
                  Nat.ne_of_gt hindex, heq] at h

private theorem mergeConcatDims_layout (leading trailing : List Nat)
    (index left right : Nat) :
    NN.IR.OpContracts.mergeConcatDims (index + leading.length) index
        (leading ++ left :: trailing) (leading ++ right :: trailing) =
      .ok (leading ++ (left + right) :: trailing) := by
  induction leading generalizing index with
  | nil =>
      simp [NN.IR.OpContracts.mergeConcatDims,
        mergeConcatDims_pastAxis index (index + 1) trailing (by omega)]
  | cons dimension leading ih =>
      rw [show index + (dimension :: leading).length = (index + 1) + leading.length by
        simp only [List.length_cons]
        omega]
      simp [NN.IR.OpContracts.mergeConcatDims, ih (index + 1),
        show index ≠ (index + 1) + leading.length by omega]

private theorem mergeConcatDims_layout_ok (leading trailing : List Nat)
    (index left : Nat) (actual output : List Nat)
    (h : NN.IR.OpContracts.mergeConcatDims (index + leading.length) index
      (leading ++ left :: trailing) actual = .ok output) :
    ∃ right, actual = leading ++ right :: trailing ∧
      output = leading ++ (left + right) :: trailing := by
  induction leading generalizing index actual output with
  | nil =>
      cases actual with
      | nil => simp [NN.IR.OpContracts.mergeConcatDims] at h
      | cons right rest =>
          cases htail : NN.IR.OpContracts.mergeConcatDims index (index + 1)
              trailing rest with
          | error message =>
              simp [NN.IR.OpContracts.mergeConcatDims, htail] at h
          | ok tail =>
              obtain ⟨rfl, rfl⟩ :=
                mergeConcatDims_pastAxis_ok index (index + 1) trailing rest tail (by omega) htail
              exact ⟨right, rfl, by
                simpa [NN.IR.OpContracts.mergeConcatDims, htail] using h.symm⟩
  | cons dimension leading ih =>
      cases actual with
      | nil => simp [NN.IR.OpContracts.mergeConcatDims] at h
      | cons other others =>
          have haxis : index + (dimension :: leading).length =
              (index + 1) + leading.length := by
            simp only [List.length_cons]
            omega
          rw [haxis] at h
          cases htail : NN.IR.OpContracts.mergeConcatDims ((index + 1) + leading.length)
              (index + 1) (leading ++ left :: trailing) others with
          | error message =>
              simp [NN.IR.OpContracts.mergeConcatDims, htail] at h
          | ok tail =>
              obtain ⟨right, rfl, rfl⟩ := ih (index + 1) others tail htail
              by_cases heq : other = dimension
              · subst other
                refine ⟨right, rfl, ?_⟩
                simpa [NN.IR.OpContracts.mergeConcatDims, htail,
                  show index ≠ (index + 1) + leading.length by omega] using h.symm
              · simp [NN.IR.OpContracts.mergeConcatDims, htail, heq,
                  show index ≠ (index + 1) + leading.length by omega] at h

private theorem foldConcatDims_layout (leading trailing lengths : List Nat) (initial : Nat) :
    NN.IR.OpContracts.foldConcatDims leading.length (leading ++ initial :: trailing)
        (lengths.map fun length => leading ++ length :: trailing) =
      .ok (leading ++ (initial + lengths.sum) :: trailing) := by
  induction lengths generalizing initial with
  | nil => rfl
  | cons length lengths ih =>
      have hmerge := mergeConcatDims_layout leading trailing 0 initial length
      simp only [Nat.zero_add] at hmerge
      simp [NN.IR.OpContracts.foldConcatDims, hmerge, ih, Nat.add_assoc]

private theorem foldConcatDims_layout_ok (leading trailing : List Nat)
    (initial : Nat) (shapes : List Shape) (output : List Nat)
    (h : NN.IR.OpContracts.foldConcatDims leading.length
      (leading ++ initial :: trailing) shapes = .ok output) :
    ∃ lengths : List Nat, shapes = lengths.map (fun length => leading ++ length :: trailing) ∧
      output = leading ++ (initial + lengths.sum) :: trailing := by
  induction shapes generalizing initial output with
  | nil =>
      refine ⟨[], rfl, ?_⟩
      simpa [NN.IR.OpContracts.foldConcatDims] using h.symm
  | cons shape shapes ih =>
      cases hmerge : NN.IR.OpContracts.mergeConcatDims leading.length 0
          (leading ++ initial :: trailing) shape with
      | error message => simp [NN.IR.OpContracts.foldConcatDims, hmerge] at h
      | ok dimensions =>
          have hmerge' : NN.IR.OpContracts.mergeConcatDims (0 + leading.length) 0
              (leading ++ initial :: trailing) shape = .ok dimensions := by
            simpa only [Nat.zero_add] using hmerge
          obtain ⟨length, rfl, rfl⟩ :=
            mergeConcatDims_layout_ok leading trailing 0 initial shape dimensions hmerge'
          have htail : NN.IR.OpContracts.foldConcatDims leading.length
              (leading ++ (initial + length) :: trailing) shapes = .ok output := by
            simpa [NN.IR.OpContracts.foldConcatDims, hmerge] using h
          obtain ⟨lengths, rfl, houtput⟩ := ih (initial + length) output htail
          exact ⟨length :: lengths, rfl, by simpa [Nat.add_assoc] using houtput⟩

/-- IR shape inference accepts a concat family along any axis, including empty segments. -/
theorem inferConcatOutShape_layout (leading trailing : List Nat)
    (first second : Nat) (lengths : List Nat) :
    NN.IR.OpContracts.inferConcatOutShape leading.length
        ((first :: second :: lengths).map fun length => leading ++ length :: trailing).toArray =
      .ok (leading ++ (first + second + lengths.sum) :: trailing) := by
  have haxis :
      NN.IR.OpContracts.checkAxisValid leading.length (leading ++ first :: trailing) = .ok () := by
    simp [NN.IR.OpContracts.checkAxisValid, Shape.rank_eq_length]
  have hfold := foldConcatDims_layout leading trailing (second :: lengths) first
  simp only [List.map_cons] at hfold
  simp [NN.IR.OpContracts.inferConcatOutShape, haxis, hfold, Nat.add_assoc]

/-- Successful IR concat inference determines a common leading shape and trailing shape. -/
theorem inferConcatOutShape_family (axis : Nat) (parents : Array Shape) (output : Shape)
    (h : NN.IR.OpContracts.inferConcatOutShape axis parents = .ok output) :
    ∃ (leading trailing : List Nat) (first second : Nat) (lengths : List Nat),
      axis = leading.length ∧
      parents =
        ((first :: second :: lengths).map fun length => leading ++ length :: trailing).toArray ∧
      output = leading ++ (first + second + lengths.sum) :: trailing := by
  rcases parents with ⟨parents⟩
  cases parents with
  | nil => simp [NN.IR.OpContracts.inferConcatOutShape] at h
  | cons first parents =>
      cases parents with
      | nil => simp [NN.IR.OpContracts.inferConcatOutShape] at h
      | cons second parents =>
          change (NN.IR.OpContracts.checkAxisValid axis first >>= fun _ =>
            NN.IR.OpContracts.foldConcatDims axis first (second :: parents) >>= fun dimensions =>
              pure (Shape.ofList dimensions)) = .ok output at h
          cases hvalid : NN.IR.OpContracts.checkAxisValid axis first with
          | error message => simp [hvalid] at h
          | ok checked =>
              have haxis : axis < first.length := by
                unfold NN.IR.OpContracts.checkAxisValid at hvalid
                split at hvalid
                · simpa [Shape.rank_eq_length] using ‹axis < first.rank›
                · cases hvalid
              have hfirst :
                  first = first.take axis ++ first[axis] :: first.drop (axis + 1) := by
                rw [← List.drop_eq_getElem_cons haxis, List.take_append_drop]
              have hleading : (first.take axis).length = axis := by
                simp [List.length_take, Nat.min_eq_left haxis.le]
              cases hfold : NN.IR.OpContracts.foldConcatDims axis first (second :: parents) with
              | error message => simp [hvalid, hfold] at h
              | ok dimensions =>
                  have hout : dimensions = output := by simpa [hvalid, hfold] using h
                  subst dimensions
                  have hfold' : NN.IR.OpContracts.foldConcatDims (first.take axis).length
                      (first.take axis ++ first[axis] :: first.drop (axis + 1))
                      (second :: parents) = .ok output := by
                    simpa only [hleading, ← hfirst] using hfold
                  obtain ⟨sizes, hsizes, houtput⟩ := foldConcatDims_layout_ok
                    (first.take axis) (first.drop (axis + 1)) first[axis]
                    (second :: parents) output hfold'
                  cases sizes with
                  | nil => simp at hsizes
                  | cons length lengths =>
                      obtain ⟨hsecond, hparents⟩ := List.cons.inj hsizes
                      refine ⟨first.take axis, first.drop (axis + 1), first[axis],
                        length, lengths, hleading.symm, ?_, ?_⟩
                      · simp only [List.map_cons, ← hfirst, ← hsecond, ← hparents]
                      · simpa [Nat.add_assoc] using houtput

/-- The checked layout recovers every shaped concat family accepted by the IR contract. -/
theorem concatLayout?_family (leading trailing : List Nat)
    (first second : Nat) (lengths : List Nat) :
    concatLayout? leading.length
        ((first :: second :: lengths).map fun length => leading ++ length :: trailing).toArray
        (leading ++ (first + second + lengths.sum) :: trailing) =
      some { leading := leading, trailing := trailing, lengths := first :: second :: lengths } := by
  unfold concatLayout?
  rw [inferConcatOutShape_layout]
  simp [ConcatLayout.outputShape,
    List.drop_append, List.map_map, Function.comp_def, Nat.add_assoc,
    List.drop_eq_nil_of_le (by omega : leading.length ≤ leading.length + 1)]
  unfold ConcatLayout.parentShape
  simp [List.map_map, Function.comp_def, List.drop_append,
    List.drop_eq_nil_of_le (by omega : leading.length ≤ leading.length + 1)]
  change (first :: second :: lengths).map (fun length => leading ++ length :: trailing) = _
  apply List.ext_getElem
  · simp
  · intro index hleft hright
    simp only [List.getElem_map, List.getElem_finRange]
    rfl

/-- Leading-axis concat evaluates to the same family tensor for any arity of at least two. -/
theorem evalConcat_zero_family {α : Type} [TorchLean.Storage α] [Context α]
    (id : Nat) (node : Node) (trailing : Shape) (first second : Nat) (lengths : List Nat)
    (values : (parent : Fin (first :: second :: lengths).length) →
      Tensor α ((first :: second :: lengths).get parent :: trailing))
    (hout : node.outShape = (first + second + lengths.sum) :: trailing) :
    NN.IR.Graph.evalConcat id node 0
        (Array.ofFn fun parent => SomeTensor.ofTensor (values parent)) =
      .ok (SomeTensor.ofTensor
        (Tensor.Internal.Rep.concatenateAxes [] trailing (first :: second :: lengths) values)) := by
  have hshapes :
      (Array.ofFn fun parent => SomeTensor.ofTensor (values parent)).map SomeTensor.shape =
        ((first :: second :: lengths).map fun length => length :: trailing).toArray := by
    rw [Array.map_ofFn, ← List.toArray_ofFn]
    congr 1
    simpa only [List.map_ofFn, Function.comp_def, List.get_eq_getElem,
      SomeTensor.ofTensor, SomeTensor.shape] using
      congrArg (List.map fun length => length :: trailing)
        (List.ofFn_getElem (xs := first :: second :: lengths))
  have hinfer := inferConcatOutShape_layout [] trailing first second lengths
  simp only [List.nil_append, List.length_nil] at hinfer
  unfold NN.IR.Graph.evalConcat
  rw [hshapes, hinfer]
  simpa [hout, Nat.add_assoc] using
    evalConcatLeadingAxisFold_family id trailing first (second :: lengths) values

/-- A scalar in the flattened tensor is read at its row-major shaped coordinate. -/
theorem getScalar_flattenSpec {α : Type} [TorchLean.Storage α]
    {shape : Shape} (tensor : Tensor α shape) (index : Fin shape.size) :
    Tensor.getScalar (Tensor.flattenSpec tensor) index =
      tensor (((Tensor.Internal.Coord.equivFin shape).trans
        (finCongr (Shape.internalSize_eq shape))).symm index) := by
  obtain ⟨c, rfl⟩ := ((Tensor.Internal.Coord.equivFin shape).trans
    (finCongr (Shape.internalSize_eq shape))).surjective index
  rw [Equiv.symm_apply_apply]
  exact Spec.getScalar_flattenSpec_linearize tensor c

namespace ConcatLayout

/-- Flattening shaped concatenation gives exactly the CROWN layout's flat concatenation. -/
theorem flatten_concatenateAxes {α : Type} [TorchLean.Storage α] (layout : ConcatLayout)
    (values : (parent : Fin layout.lengths.length) → Tensor α (layout.parentShape parent)) :
    Tensor.flattenSpec
        (Tensor.Internal.Rep.concatenateAxes layout.leading layout.trailing layout.lengths
          values) =
      layout.concat (fun parent => Tensor.flattenSpec (values parent)) := by
  apply Tensor.ext_vector
  intro index
  rw [concat]
  erw [Tensor.getScalar_ofFn]
  simp only [getScalar_flattenSpec,
    Tensor.Internal.Rep.concatenateAxes, Tensor.Internal.Rep.get_ofFn]
  let parentEquiv (parent : Fin layout.lengths.length) :=
    (Tensor.Internal.Coord.equivFin (layout.parentShape parent)).trans
      (finCongr (Shape.internalSize_eq (layout.parentShape parent)))
  let source :=
    (Tensor.Internal.Rep.concatenateAxesCoordinateEquiv
      layout.leading layout.trailing layout.lengths).symm
      (((Tensor.Internal.Coord.equivFin layout.outputShape).trans
        (finCongr (Shape.internalSize_eq layout.outputShape))).symm index)
  change values source.1 source.2 =
    values source.1 ((parentEquiv source.1).symm (parentEquiv source.1 source.2))
  exact (congrArg (values source.1)
    ((parentEquiv source.1).symm_apply_apply source.2)).symm

end ConcatLayout

/-- Checked flat concatenation accepts every shaped family with the declared layout. -/
theorem concatFlatValues?_flatten {α : Type} [TorchLean.Storage α] [Context α]
    (layout : ConcatLayout)
    (values : (parent : Fin layout.lengths.length) → Tensor α (layout.parentShape parent)) :
    concatFlatValues? layout
        (Array.ofFn fun parent =>
          { n := (layout.parentShape parent).size, v := Tensor.flattenSpec (values parent) }) =
      some
        { n := layout.outputShape.size
          v := Tensor.flattenSpec
            (Tensor.Internal.Rep.concatenateAxes layout.leading layout.trailing layout.lengths
              values) } := by
  have hsequence (f : Fin layout.lengths.length → FlatTensor α) :
      Tensor.Internal.sequenceFinM (fun parent => some (f parent)) = some f := by
    exact Tensor.Internal.sequenceFinM_pure f
  simp only [concatFlatValues?, Array.size_ofFn, beq_self_eq_true,
    ↓reduceIte, Array.getElem?_ofFn, Fin.isLt]
  erw [hsequence]
  simp [← ConcatLayout.flatten_concatenateAxes]
  rfl

end NN.MLTheory.CROWN.Graph
