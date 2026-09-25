/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphConcatBridge
public import NN.MLTheory.CROWN.Proofs.GraphConcatPermutation
public import NN.MLTheory.CROWN.Proofs.GraphConcatInversePermutation

/-!
# Runtime concatenation through axis permutations

Successful runtime traversal recovers a typed family. The leading-axis fold then supplies the
family concatenation between the front and back coordinate permutations.
-/

public section

namespace NN.MLTheory.CROWN.Graph

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor.Internal (Rep)
open NN.Verification.Builtin.Proved.Correctness.IRStep

attribute [local simp] Bind.bind Pure.pure Except.bind Except.pure NN.IR.throw_eq_error
  Option.bind Shape.toList Shape.ofList Shape.dim

private theorem bind_eq_ok {α β : Type} {action : Except String α}
    {next : α → Except String β} {result : β}
    (h : (action >>= next) = .ok result) :
    ∃ value, action = .ok value ∧ next value = .ok result := by
  cases action with
  | error message => cases h
  | ok value => exact ⟨value, rfl, h⟩

private theorem array_mapM_ofFn_of_ok {α β : Type} {count : Nat}
    (values : Fin count → α) (results : Fin count → β) (f : α → Except String β)
    (h : ∀ index result, f (values index) = .ok result → result = results index)
    {output : Array β} (heval : (Array.ofFn values).mapM f = .ok output) :
    output = Array.ofFn results := by
  have hlist (indices : List (Fin count)) (output : List β)
      (heval : indices.mapM (f ∘ values) = .ok output) :
      output = indices.map results := by
    induction indices generalizing output with
    | nil => simpa using heval.symm
    | cons index indices ih =>
        rw [List.mapM_cons] at heval
        obtain ⟨value, hvalue, hnext⟩ := bind_eq_ok heval
        obtain ⟨tail, htail, hresult⟩ := bind_eq_ok hnext
        have houtput : value :: tail = output := Except.ok.inj hresult
        rw [← houtput, h index value hvalue, ih tail htail]
        rfl
  have hvalues : Array.ofFn values = (Array.ofFn id).map values := by
    rw [Array.map_ofFn]
    rfl
  rw [hvalues, Array.mapM_map, Array.mapM_eq_mapM_toList] at heval
  cases htraverse : (List.ofFn (id : Fin count → Fin count)).mapM (f ∘ values) with
  | error message => simp [htraverse] at heval
  | ok collected =>
      have houtput : collected.toArray = output := by simpa [htraverse] using heval
      rw [← houtput, hlist _ collected htraverse]
      simp only [List.map_ofFn, List.toArray_ofFn, Function.comp_id]

private theorem mapM_range'_eq_some {α : Type} (f : Nat → Option α)
    (start : Nat) (values : List α)
    (h : ∀ index, index < values.length → f (start + index) = values[index]?) :
    (List.range' start values.length).mapM f = some values := by
  induction values generalizing start with
  | nil => rfl
  | cons value values ih =>
      have hfirst : f start = some value := by simpa using h 0 (by simp)
      have htail : ∀ index, index < values.length →
          f (start + 1 + index) = values[index]? := by
        intro index hi
        simpa [Nat.add_assoc, Nat.add_comm 1] using h (index + 1) (by simpa using hi)
      simp only [List.length_cons, List.range'_succ, List.mapM_cons, hfirst]
      rw [ih (start + 1) htail]
      rfl

private theorem permuteShape_front_of_some (leading trailing : Shape) (length : Nat)
    (result : Shape)
    (h : Shape.permute? (leading ++ length :: trailing)
        (leading.length ::
          (List.range leading.length ++ List.range' (leading.length + 1) trailing.length)) =
      some result) :
    result = length :: (leading ++ trailing) := by
  have hleading :
      (List.range leading.length).mapM
          (fun index => (leading ++ length :: trailing)[index]?) = some leading := by
    rw [List.range_eq_range']
    apply mapM_range'_eq_some
    intro index hi
    simpa using List.getElem?_append_left (l₂ := length :: trailing) hi
  have htrailing :
      (List.range' (leading.length + 1) trailing.length).mapM
          (fun index => (leading ++ length :: trailing)[index]?) = some trailing := by
    apply mapM_range'_eq_some
    intro index _
    rw [List.getElem?_append_right (by omega)]
    rw [show leading.length + 1 + index - leading.length = index + 1 by omega]
    rfl
  have hfirst : (leading ++ length :: trailing)[leading.length]? = some length := by
    rw [List.getElem?_append_right (Nat.le_refl _)]
    simp
  have hrank :
      (leading.length ::
          (List.range leading.length ++
            List.range' (leading.length + 1) trailing.length)).length =
        (leading ++ length :: trailing).rank := by
    simp only [List.length_cons, List.length_append, List.length_range, List.length_range',
      Shape.rank_eq_length]
    omega
  simp only [Shape.permute?, hrank, bne_self_eq_false, Bool.false_eq_true, ↓reduceIte] at h
  split at h
  · cases h
  · simpa [List.mapM_cons, List.mapM_append, hfirst, hleading, htrailing] using h.symm

private theorem leadingAxis_fold_ofFn {α : Type} [TorchLean.Storage α] [Context α]
    (trailing : Shape) (first : Nat) (lengths : List Nat)
    (values : (parent : Fin (first :: lengths).length) →
      Tensor α ((first :: lengths).get parent :: trailing)) :
    ((Array.ofFn fun parent : Fin (first :: lengths).length =>
        (⟨(first :: lengths).get parent, values parent⟩ :
          LeadingAxisConcat.Input α trailing)).extract 1).foldl
        (β := LeadingAxisConcat.Input α trailing)
        (fun acc next => ⟨acc.1 + next.1, Tensor.concatAxisSpec Shape.scalar acc.2 next.2⟩)
        (Array.ofFn fun parent : Fin (first :: lengths).length =>
          (⟨(first :: lengths).get parent, values parent⟩ :
            LeadingAxisConcat.Input α trailing))[0] =
      ⟨(first :: lengths).sum, Rep.concatenateAxes [] trailing (first :: lengths) values⟩ := by
  simp only [Array.getElem_ofFn]
  change Array.foldl _ (⟨first, values 0⟩ : LeadingAxisConcat.Input α trailing) _ = _
  rw [← List.toArray_ofFn, List.ofFn_succ]
  erw [show ((⟨first, values 0⟩ : LeadingAxisConcat.Input α trailing) ::
      List.ofFn (fun parent : Fin lengths.length =>
        (⟨lengths.get parent, values parent.succ⟩ :
          LeadingAxisConcat.Input α trailing))).toArray.extract 1 =
      (List.ofFn (fun parent : Fin lengths.length =>
        (⟨lengths.get parent, values parent.succ⟩ :
          LeadingAxisConcat.Input α trailing))).toArray by simp [List.take_of_length_le]]
  rw [List.foldl_toArray]
  exact leadingAxisConcat_fold_ofFn trailing first lengths values

private theorem evalConcat_family_of_permutations
    {α : Type} [TorchLean.Storage α] [Context α]
    (id : Nat) (node : Node) (leading trailing : Shape)
    (first second : Nat) (lengths : List Nat)
    (values : (parent : Fin (first :: second :: lengths).length) →
      Tensor α (leading ++ (first :: second :: lengths).get parent :: trailing))
    (result : SomeTensor α)
    (move : (length : Nat) → Tensor α (leading ++ length :: trailing) →
      Tensor α (length :: (leading ++ trailing)))
    (hfront : ∀ length tensor output,
      NN.IR.Graph.permuteSomeTensor (SomeTensor.ofTensor tensor)
          (leading.length ::
            (List.range leading.length ++
              List.range' (leading.length + 1) trailing.length)).toArray = .ok output →
        output = SomeTensor.ofTensor (move length tensor))
    (hback : ∀ output,
      NN.IR.Graph.permuteSomeTensor
          (SomeTensor.ofTensor (move (first :: second :: lengths).sum
            (Rep.concatenateAxes leading trailing (first :: second :: lengths) values)))
          (List.range' 1 leading.length ++ [0] ++
            List.range' (leading.length + 1) trailing.length).toArray = .ok output →
        output = SomeTensor.ofTensor
          (Rep.concatenateAxes leading trailing (first :: second :: lengths) values))
    (hfamily : move (first :: second :: lengths).sum
        (Rep.concatenateAxes leading trailing (first :: second :: lengths) values) =
      Rep.concatenateAxes [] (leading ++ trailing) (first :: second :: lengths)
        (fun parent => move _ (values parent)))
    (hperm : NN.IR.OpContracts.permMoveAxisToFront leading.length
        (leading ++ (first :: second :: lengths).sum :: trailing) =
      .ok (leading.length ::
        (List.range leading.length ++ List.range' (leading.length + 1) trailing.length)).toArray)
    (hinverse : NN.IR.OpContracts.inversePerm
        (leading.length ::
          (List.range leading.length ++
            List.range' (leading.length + 1) trailing.length)).toArray =
      .ok (List.range' 1 leading.length ++ [0] ++
        List.range' (leading.length + 1) trailing.length).toArray)
    (haxis : leading.length ≠ 0)
    (hout : node.outShape = leading ++ (first :: second :: lengths).sum :: trailing)
    (heval : NN.IR.Graph.evalConcat id node leading.length
        (Array.ofFn fun parent => SomeTensor.ofTensor (values parent)) = .ok result) :
    result = SomeTensor.ofTensor
      (Rep.concatenateAxes leading trailing (first :: second :: lengths) values) := by
  have hshapes :
      (Array.ofFn fun parent => SomeTensor.ofTensor (values parent)).map SomeTensor.shape =
        ((first :: second :: lengths).map
          fun length => leading ++ length :: trailing).toArray := by
    rw [Array.map_ofFn, ← List.toArray_ofFn]
    congr 1
    simpa only [List.map_ofFn, Function.comp_def, List.get_eq_getElem,
      SomeTensor.ofTensor, SomeTensor.shape] using
      congrArg (List.map fun length => leading ++ length :: trailing)
        (List.ofFn_getElem (xs := first :: second :: lengths))
  have hinfer := inferConcatOutShape_layout leading trailing first second lengths
  have hinfer' :
      NN.IR.OpContracts.inferConcatOutShape leading.length
          ((first :: second :: lengths).map
            fun length => leading ++ length :: trailing).toArray =
        .ok (leading ++ (first :: second :: lengths).sum :: trailing) := by
    simpa [Nat.add_assoc] using hinfer
  unfold NN.IR.Graph.evalConcat at heval
  rw [hshapes, hinfer', hout] at heval
  simp only [Bind.bind, Pure.pure, Except.bind, Except.pure, bne_self_eq_false,
    Bool.false_eq_true, ↓reduceIte, haxis] at heval
  rw [hperm] at heval
  dsimp only at heval
  rw [hinverse] at heval
  dsimp only at heval
  cases hshape : Shape.permute? (leading ++ (first :: second :: lengths).sum :: trailing)
      (leading.length ::
        (List.range leading.length ++
          List.range' (leading.length + 1) trailing.length)) with
  | none =>
      simp only [hshape, NN.IR.throw_eq_error] at heval
      cases heval
  | some frontShape =>
      have hfrontShape := permuteShape_front_of_some leading trailing
        (first :: second :: lengths).sum frontShape hshape
      subst frontShape
      simp only [hshape] at heval
      obtain ⟨frontParents, hfrontParents, hcontinue⟩ := bind_eq_ok heval
      have hparents : frontParents = Array.ofFn
          (fun parent => SomeTensor.ofTensor (move _ (values parent))) := by
        apply array_mapM_ofFn_of_ok _ _ _ _ hfrontParents
        intro parent output hread
        cases hpermute : NN.IR.Graph.permuteSomeTensor
            (SomeTensor.ofTensor (values parent))
            (leading.length ::
              (List.range leading.length ++
                List.range' (leading.length + 1) trailing.length)).toArray with
        | error message =>
            simp only [hpermute, NN.IR.throw_eq_error] at hread
            cases hread
        | ok permuted =>
            have houtput : permuted = output := by
              simpa only [hpermute, Except.ok.injEq] using hread
            subst output
            exact hfront _ _ _ hpermute
      rw [hparents] at hcontinue
      obtain ⟨sigmas, hsigmas, hfold⟩ := bind_eq_ok hcontinue
      have hsigmas' : sigmas = Array.ofFn
          (fun parent : Fin (first :: second :: lengths).length =>
            (⟨(first :: second :: lengths).get parent, move _ (values parent)⟩ :
              LeadingAxisConcat.Input α (leading ++ trailing))) := by
        apply array_mapM_ofFn_of_ok _ _ _ _ hsigmas
        intro parent output hread
        simpa [SomeTensor.ofTensor] using hread.symm
      rw [hsigmas'] at hfold
      rw [Array.getElem?_eq_getElem (by simp)] at hfold
      dsimp only at hfold
      have hfoldResult := leadingAxis_fold_ofFn (leading ++ trailing) first (second :: lengths)
        (fun parent => move _ (values parent))
      rw [← hfamily] at hfoldResult
      rw [hfoldResult] at hfold
      simp only [↓reduceDIte] at hfold
      cases hpermute : NN.IR.Graph.permuteSomeTensor
          (SomeTensor.ofTensor (move (first :: second :: lengths).sum
            (Rep.concatenateAxes leading trailing (first :: second :: lengths) values)))
          (List.range' 1 leading.length ++ [0] ++
            List.range' (leading.length + 1) trailing.length).toArray with
      | error message =>
          erw [hpermute] at hfold
          simp only [NN.IR.throw_eq_error] at hfold
          cases hfold
      | ok output =>
          have houtput := hback output hpermute
          erw [hpermute] at hfold
          rw [houtput] at hfold
          simpa [NN.IR.Graph.expectShape, SomeTensor.ofTensor] using hfold.symm

/-- Successful concat of a typed family has the independent tensor concatenation as its value. -/
theorem evalConcat_family {α : Type} [TorchLean.Storage α] [Context α]
    (id : Nat) (node : Node) (leading trailing : Shape)
    (first second : Nat) (lengths : List Nat)
    (values : (parent : Fin (first :: second :: lengths).length) →
      Tensor α (leading ++ (first :: second :: lengths).get parent :: trailing))
    (result : SomeTensor α)
    (hout : node.outShape = leading ++ (first :: second :: lengths).sum :: trailing)
    (heval : NN.IR.Graph.evalConcat id node leading.length
        (Array.ofFn fun parent => SomeTensor.ofTensor (values parent)) = .ok result) :
    result = SomeTensor.ofTensor
      (Rep.concatenateAxes leading trailing (first :: second :: lengths) values) := by
  by_cases haxis : leading.length = 0
  · have hleading : leading = [] := List.length_eq_zero_iff.mp haxis
    subst leading
    have hresult := evalConcat_zero_family id node trailing first second lengths values
      (by simpa [Nat.add_assoc] using hout)
    exact Except.ok.inj (heval.symm.trans hresult)
  · exact evalConcat_family_of_permutations id node leading trailing first second lengths
      values result (fun _ tensor => concatAxisToFront leading trailing tensor)
      (fun _ tensor output => permuteSomeTensor_front_of_ok leading trailing tensor output)
      (fun output heval => permuteSomeTensor_back_of_ok leading trailing
        (Rep.concatenateAxes leading trailing (first :: second :: lengths) values) output
        (by simpa only [List.append_assoc, List.singleton_append] using heval))
      (concatAxisToFront_concatenateAxes leading trailing (first :: second :: lengths) values)
      (permMoveAxisToFront_append leading trailing (first :: second :: lengths).sum)
      (inversePerm_front leading.length trailing.length) haxis hout heval

/-- Actual runtime concat determines its checked layout and tensor family at every valid axis. -/
theorem evalConcat_family_of_ok {α : Type} [TorchLean.Storage α] [Context α]
    (id : Nat) (node : Node) (axis : Nat) (parents : Array (SomeTensor α))
    (result : SomeTensor α)
    (heval : NN.IR.Graph.evalConcat id node axis parents = .ok result) :
    ∃ (layout : ConcatLayout)
      (values : (parent : Fin layout.lengths.length) → Tensor α (layout.parentShape parent)),
      concatLayout? axis (parents.map SomeTensor.shape) node.outShape = some layout ∧
      parents = Array.ofFn (fun parent => SomeTensor.ofTensor (values parent)) ∧
      result = SomeTensor.ofTensor
        (Rep.concatenateAxes layout.leading layout.trailing layout.lengths values) := by
  cases hinfer : NN.IR.OpContracts.inferConcatOutShape axis (parents.map SomeTensor.shape) with
  | error message => simp [NN.IR.Graph.evalConcat, hinfer] at heval
  | ok expected =>
      have hout : expected = node.outShape := by
        by_contra hne
        simp [NN.IR.Graph.evalConcat, hinfer, hne] at heval
      obtain ⟨leading, trailing, first, second, lengths, haxis, hshapes, houtput⟩ :=
        inferConcatOutShape_family axis (parents.map SomeTensor.shape) expected hinfer
      let layout : ConcatLayout :=
        { leading := leading, trailing := trailing, lengths := first :: second :: lengths }
      have hfamily : parents.map SomeTensor.shape = Array.ofFn layout.parentShape := by
        rw [hshapes, ← List.toArray_ofFn]
        congr 1
        symm
        change (List.ofFn fun parent : Fin (first :: second :: lengths).length =>
          leading ++ (first :: second :: lengths).get parent :: trailing) = _
        simpa only [List.map_ofFn, Function.comp_def, List.get_eq_getElem] using
          congrArg (List.map fun length => leading ++ length :: trailing)
            (List.ofFn_getElem (xs := first :: second :: lengths))
      obtain ⟨values, hvalues⟩ := exists_tensor_family_of_shapes layout.parentShape parents hfamily
      refine ⟨layout, values, ?_, hvalues, ?_⟩
      · rw [haxis, hshapes, ← hout, houtput]
        exact concatLayout?_family leading trailing first second lengths
      · have hnode : node.outShape = leading ++ (first :: second :: lengths).sum :: trailing := by
          simpa [Nat.add_assoc] using hout.symm.trans houtput
        rw [haxis, hvalues] at heval
        exact evalConcat_family id node leading trailing first second lengths values result
          hnode heval

end NN.MLTheory.CROWN.Graph
