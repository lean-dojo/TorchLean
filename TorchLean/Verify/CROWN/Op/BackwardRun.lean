/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.Op.BackwardProduct

/-!
# The CROWN queries on a run of an operator graph

The soundness theorems of the backward passes in `CROWN.Op` and `Op.BackwardProduct` hold at a real
point of the graph (`GraphPoint`) whose operator nodes satisfy their real equations (`Equation`,
`ProductEquation`). Here all three come from a successful real run on a payload holding the reals of
the rounded constants (`Op.Directed.RealConsts`). The point is the run's table of flat values
(`runVals`), read by `dimsOf` and `pointOf`. A node that evaluates reads every reference
(`read_of_eval`), so the run orders the graph (`topoSorted_of_run`), and a reference whose value has
a flat tensor names output `0` (`out_eq_zero`): every later output of an operator is an index tensor
(`index_of_outs`).

`equation_of_eval` and `productEquation_of_eval` give the real equations of a node from its
evaluation, one lemma per clause: `detach_eq`, `reshape_eq`, `add_eq`, `sub_eq`, `linear_eq`,
`sum_eq`, `slice_eq`, `cat_eq`, `permute_eq`, `conv_eq`, `mul_eq`, `matmul_eq` and
`scaledMatmul_eq`. `graphPoint_of_run` is the point for any table whose boxes enclose the run, such
as the boxes of `Op.runIBP` (`runIBP_encloses`). The CROWN queries then hold on the run with no
premise on the point or the equations: `affineBounds?_of_run`, `nodeBounds?_of_run`,
`backwardObjectiveBox?_of_run` and `outputBoxCROWN?_of_run`.

Full CROWN holds on the run too. A box of `crownStep` encloses its node's flat value when the boxes
of the nodes before it, the supplied boxes and the seed boxes enclose the run and `xB` encloses the
flat value of the designated input, an input node (`crownStep_encloses`): the interval step encloses
it (`Op.Directed.step?_encloses`), and so do its meets with the supplied box and with the node's
affine bounds evaluated on `xB` (`evalCROWNOutputBox?_encloses`). So every box of `crownBoxes`
encloses the run (`crownBoxes_encloses`, node by node through `tabulate_induction`), and when the
input's flat value has the width of the affine forms, every affine bound of `runCROWN` bounds its
node's flat value at it (`runCROWN_encloses`, with `boundsIdentity_encloses` and
`tighterRows_encloses`).
-/

@[expose] public section

open TorchLean.Shape.Coord (linearize linearize_unlinearize unlinearize)
open TorchLean.Shape (size_append)
open TorchLean.Shape (size)

namespace TorchLean.CROWN.Op

open Tensor
open Graph
open CROWN.Directed

noncomputable section

open Shape (Coord)

/-! ## The point of a table of flat values -/

/-- The width of the flat value at node `id` of a table, zero without one. -/
def dimsOf (vals : Array (Option (FlatTensor ℝ))) (id : Nat) : Nat :=
  ((vals[id]?).join.map (·.n)).getD 0

/-- Coordinate `i` of the flat value at node `id` of a table, zero without one. -/
def pointOf (vals : Array (Option (FlatTensor ℝ))) (id i : Nat) : ℝ :=
  ((vals[id]?).join.map fun t => getAtOrZero t.v [i]).getD 0

/-- At a node with flat value `t`, the width of the table's point is the width of `t`. -/
theorem dimsOf_eq {vals : Array (Option (FlatTensor ℝ))} {id : Nat} {t : FlatTensor ℝ}
    (h : (vals[id]?).join = some t) : dimsOf vals id = t.n := by
  simp [dimsOf, h]

/-- At a node with flat value `t`, the coordinates of the table's point are those of `t`. -/
theorem pointOf_eq {vals : Array (Option (FlatTensor ℝ))} {id : Nat} {t : FlatTensor ℝ}
    (h : (vals[id]?).join = some t) : pointOf vals id = fun i => getAtOrZero t.v [i] := by
  funext i
  simp [pointOf, h]

/-! ## Reads in a run -/

/-- Every output of an operator after its first is an index tensor: the positions that `sort`,
`topk`, `cummax` and `cummin` return beside their values. -/
theorem index_of_outs {ins outs : List Port} (op : Op ins outs) {k : Nat} {q : Port}
    (h : outs[k + 1]? = some q) : ∃ b, q.kind = .index b := by
  cases op <;> rcases k with _ | k <;> simp at h <;> (subst h; exact ⟨_, rfl⟩)

/-- A value with a flat tensor is not an index tensor. -/
theorem kind_ne_index {w : Op.Value ℝ} {t : FlatTensor ℝ} (h : flatOf w = some t)
    (b : List Nat) : w.1.kind ≠ .index b := by
  obtain ⟨⟨d, s⟩, x⟩ := w
  intro hd
  simp only at hd
  subst hd
  simp [flatOf] at h

/-- In a successful run, a reference that reads a value with a flat tensor names output `0`. -/
theorem out_eq_zero {G : IR.Graph} {P : IR.Payload ℝ} {inputs : List (Op.Value ℝ)}
    {values : Array (List (Op.Value ℝ))} (hrun : G.run P inputs = .ok values)
    {pre : Array (List (Op.Value ℝ))} {j : Nat}
    (hagree : ∀ k, k < j → pre[k]? = values[k]?) (hsize : pre.size = j) {r : IR.Ref}
    {w : Op.Value ℝ} (hr : r.read pre = .ok w) {t : FlatTensor ℝ}
    (ht : flatOf w = some t) :
    r.out = 0 := by
  have hv := read_of_agree hagree hsize hr
  unfold IR.Ref.read at hv
  obtain ⟨vs, hvs, hw⟩ : ∃ vs, values[r.node]? = some vs ∧ vs[r.out]? = some w := by
    cases hn : values[r.node]? with
    | none => simp [hn, throw, throwThe, MonadExceptOf.throw] at hv
    | some vs =>
      cases hw : vs[r.out]? with
      | none => simp [hn, hw, throw, throwThe, MonadExceptOf.throw] at hv
      | some w' =>
        simp [hn, hw, pure, Except.pure] at hv
        exact ⟨vs, rfl, by rw [hw, hv]⟩
  obtain ⟨node, hnode⟩ : ∃ node, G.nodes[r.node]? = some node :=
    ⟨_, Array.getElem?_eq_getElem (by
      rw [← IR.Graph.run_size hrun]; exact (Array.getElem?_eq_some_iff.mp hvs).1)⟩
  have hports := IR.Graph.run_ports hrun hnode hvs
  by_contra h0
  obtain ⟨k, hk⟩ : ∃ k, r.out = k + 1 := ⟨r.out - 1, by omega⟩
  have hq : node.outs[k + 1]? = some w.1 := by
    rw [← hports, List.getElem?_map, ← hk, hw]
    rfl
  cases node with
  | input _ p => simp [IR.Node.outs] at hq
  | const p => simp [IR.Node.outs] at hq
  | op o refs =>
    obtain ⟨ins, outs, op⟩ := o
    obtain ⟨b, hb⟩ := index_of_outs op hq
    exact kind_ne_index ht b hb

/-- In a successful run, `box?` finds in the run's table, for each reference a node reads, the
flat tensor of the value the reference reads, and none when that value has no flat tensor. -/
theorem box?_runVals_of_run {G : IR.Graph} {P : IR.Payload ℝ}
    {inputs : List (Op.Value ℝ)} {values : Array (List (Op.Value ℝ))}
    (hrun : G.run P inputs = .ok values) {pre : Array (List (Op.Value ℝ))} {j : Nat}
    (hagree : ∀ k, k < j → pre[k]? = values[k]?) (hsize : pre.size = j) {r : IR.Ref}
    {w : Op.Value ℝ} (hr : r.read pre = .ok w) : box? (runVals values) r = flatOf w := by
  by_cases h0 : r.out = 0
  · exact box?_runVals h0 (read_of_agree hagree hsize hr)
  · cases hf : flatOf w with
    | none => simp [box?, h0]
    | some t => exact absurd (out_eq_zero hrun hagree hsize hr hf) h0

/-! ## Slice coordinates -/

/-- Unlinearizing one index in two equal shapes gives coordinates with the same indices. -/
theorem _root_.TorchLean.Shape.Coord.index_ofFin_congr {t u : Shape} (h : t = u) {i : Fin t.size}
    {j : Fin u.size} (hij : i.val = j.val) (k : Nat) :
    Shape.Coord.index (Coord.unlinearize i) k =
      Shape.Coord.index (Coord.unlinearize j) k := by
  subst h
  rw [Fin.ext hij]

/-- A slice along one axis with step one reads its input at the flat index of its slice
layout. -/
theorem flat_slice {s : Shape} {axes : List (Fin s.rank)} {starts stops steps : List Nat}
    (hv : Shape.SliceValid s axes starts stops steps) (x : Tensor ℝ s) {axis start stop : Nat}
    {layout : SliceLayout} (hax : axes.map Fin.val = [axis]) (hstart : starts = [start])
    (hstop : stops = [stop]) (hstep : steps = [1])
    (hl : sliceLayout? axis start (stop - start) s (s.slice axes starts stops steps) =
      some layout) (i : Fin layout.outputShape.size) :
    getAtOrZero (flatOfSome ⟨_, Tensor.slice x axes starts stops steps hv⟩).v [i.val] =
      getAtOrZero (flatOfSome ⟨s, x⟩).v [(layout.flatIndex i).val] := by
  subst hstart hstop hstep
  obtain ⟨a, rfl, rfl⟩ : ∃ a, axes = [a] ∧ a.val = axis := by
    rcases axes with _ | ⟨a, _ | ⟨b, l⟩⟩ <;> simp at hax
    exact ⟨a, rfl, hax⟩
  simp only [sliceLayout?, Option.bind_eq_bind, Option.bind_eq_some_iff] at hl
  obtain ⟨⟨hAxis⟩, -, hl⟩ := hl
  split at hl
  · rename_i hRange
    split at hl
    · rename_i hshape
      obtain rfl := Option.some.inj hl
      have hi : i.val < (s.slice [a] [start] [stop] [1]).size :=
        lt_of_lt_of_eq i.isLt (congrArg Shape.size hshape)
      have hlin : i.val = (Coord.linearize
          (Coord.unlinearize ⟨i.val, hi⟩ : (s.slice [a] [start] [stop] [1]).Coord)).val := by
        rw [Coord.linearize_unlinearize]
      simp only [flatOfSome, SliceLayout.flatIndex]
      rw [hlin, Tensor.getAtOrZero_reshape_toFin, Tensor.getAtOrZero_reshape_toFin,
        Tensor.slice_apply]
      congr 1
      apply Shape.Coord.ext_index
      intro k hk
      rw [Shape.Coord.index_ofFun, dite_eq_left hk, index_sliceAxisRangeCoordinate]
      dsimp only
      rw [Shape.Coord.val_get, Fin.val_cast, Coord.index_ofFin_congr hshape.symm rfl k]
      by_cases hka : k = a.val
      · simp [hka]
      · have ha : (⟨k, hk⟩ : Fin s.rank) ≠ a := fun e => hka (by rw [← e])
        simp [ha, hka]
    · cases hl
  · cases hl

/-! ## The real equations of a node -/

section Clauses

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

variable {G : IR.Graph} {P : IR.Payload α} {Pr : IR.Payload ℝ}
  {inputs : List (Op.Value ℝ)} {earlier : Array (List (Op.Value ℝ))}
  {vals : Array (Option (FlatTensor ℝ))} {id : Nat} {refs : Array IR.Ref}
  {outs : List (Op.Value ℝ)}

/-- At each index below the size of `s`, the flat value of `zipWith f x y` is `f` of the flat
entries of `x` and `y`. -/
theorem getAtOrZero_flat_zipWith {s : Shape} (f : ℝ → ℝ → ℝ) (x y : Tensor ℝ s) {i : Nat}
    (hi : i < s.size) :
    getAtOrZero (flatOfSome ⟨s, Tensor.zipWith f x y⟩).v [i] =
      f (getAtOrZero (flatOfSome ⟨s, x⟩).v [i]) (getAtOrZero (flatOfSome ⟨s, y⟩).v [i]) := by
  simp only [flatOfSome, reshape_vector_zipWith]
  rw [getAtOrZero_eq_getScalar _ ⟨i, hi⟩, getAtOrZero_eq_getScalar _ ⟨i, hi⟩,
    getAtOrZero_eq_getScalar _ ⟨i, hi⟩, getScalar_zipWith]

/-- A detach copies the flat value it reads. -/
theorem detach_eq {s : Shape}
    (h : (IR.Node.op ⟨_, _, Op.detach (s := s)⟩ refs).eval Pr inputs earlier id =
      .ok outs)
    (hvals : ∀ r ∈ refs.toList, ∀ w, r.read earlier = .ok w → box? vals r = flatOf w)
    (hself : (vals[id]?).join = outs[0]? >>= flatOf) :
    Equation G P (dimsOf vals) (pointOf vals) id ⟨_, _, .detach (s := s)⟩ refs.toList := by
  obtain ⟨r, w, x, hr, hread, hx, rfl⟩ := invoke_one Op.rule?_detach h
  have hb : box? vals r = some (flatOfSome ⟨s, x⟩) :=
    (hvals r (by simp [hr]) w hread).trans (flatOf_of_read (Or.inr (Or.inr rfl)) hx)
  have hs : (vals[id]?).join = some (flatOfSome ⟨s, x⟩) := hself
  rw [hr]
  exact ⟨(dimsOf_eq (join_of_box? hb)).trans (dimsOf_eq hs).symm, fun i => by
    rw [pointOf_eq hs, pointOf_eq (join_of_box? hb)]⟩

/-- A reshape keeps the flat value it reads. -/
theorem reshape_eq {s t : Shape} {hst : s.size = t.size}
    (h : (IR.Node.op ⟨_, _, Op.reshape hst⟩ refs).eval Pr inputs earlier id =
      .ok outs)
    (hvals : ∀ r ∈ refs.toList, ∀ w, r.read earlier = .ok w → box? vals r = flatOf w)
    (hself : (vals[id]?).join = outs[0]? >>= flatOf) :
    Equation G P (dimsOf vals) (pointOf vals) id ⟨_, _, .reshape hst⟩ refs.toList := by
  obtain ⟨r, w, x, hr, hread, hx, rfl⟩ := invoke_one (Op.rule?_reshape hst) h
  have hb : box? vals r = some (flatOfSome ⟨s, x⟩) :=
    (hvals r (by simp [hr]) w hread).trans
      ((flatOf_of_read (Or.inl rfl) hx).trans (flatOf_value x))
  have hs : (vals[id]?).join = some (flatOfSome ⟨t, Tensor.reshape x t hst⟩) := hself
  rw [hr]
  refine ⟨(dimsOf_eq (join_of_box? hb)).trans (hst.trans (dimsOf_eq hs).symm), fun i => ?_⟩
  rw [pointOf_eq hs, pointOf_eq (join_of_box? hb)]
  simp only [flatOfSome, Tensor.reshape_vector_reshape, Op.Directed.getAtOrZero_eqRec]

/-- A pointwise binary operator whose real forward is `zipWith f` reads two references and
computes `f` of their flat entries. -/
theorem binary_eq {s : Shape} {op : Op [.value s, .value s] [.value s]}
    {rule : Op.Rule ℝ [.value s, .value s] [.value s]}
    (hs : op.rule? (α := ℝ) = some rule) {f : ℝ → ℝ → ℝ}
    (hforward : ∀ x y, (rule.forward (x, y, ())).1 = Tensor.zipWith f x y)
    (h : op.invoke (refs.toList.mapM (·.read earlier)) = .ok outs)
    (hvals : ∀ r ∈ refs.toList, ∀ w, r.read earlier = .ok w → box? vals r = flatOf w)
    (hself : (vals[id]?).join = outs[0]? >>= flatOf) :
    ∃ r₁ r₂, refs.toList = [r₁, r₂] ∧
      BinaryEquation (dimsOf vals) (pointOf vals) id r₁.node r₂.node f := by
  obtain ⟨r₁, r₂, w₁, w₂, x₁, x₂, hr, hread₁, hread₂, hx₁, hx₂, rfl⟩ := invoke_two hs h
  have hb₁ : box? vals r₁ = some (flatOfSome ⟨s, x₁⟩) :=
    (hvals r₁ (by simp [hr]) w₁ hread₁).trans
      ((flatOf_of_read (Or.inl rfl) hx₁).trans (flatOf_value x₁))
  have hb₂ : box? vals r₂ = some (flatOfSome ⟨s, x₂⟩) :=
    (hvals r₂ (by simp [hr]) w₂ hread₂).trans
      ((flatOf_of_read (Or.inl rfl) hx₂).trans (flatOf_value x₂))
  have hs' : (vals[id]?).join = some (flatOfSome ⟨s, Tensor.zipWith f x₁ x₂⟩) := by
    rw [hself, ← hforward]
    rfl
  refine ⟨r₁, r₂, hr, (dimsOf_eq (join_of_box? hb₁)).trans (dimsOf_eq hs').symm,
    (dimsOf_eq (join_of_box? hb₂)).trans (dimsOf_eq hs').symm, fun i => ?_⟩
  rw [pointOf_eq hs', pointOf_eq (join_of_box? hb₁), pointOf_eq (join_of_box? hb₂)]
  exact getAtOrZero_flat_zipWith f x₁ x₂ (i.isLt.trans_eq (dimsOf_eq hs'))

/-- A sum of two references adds their flat entries. -/
theorem add_eq {s : Shape}
    (h : (IR.Node.op ⟨_, _, Op.add (s := s)⟩ refs).eval Pr inputs earlier id =
      .ok outs)
    (hvals : ∀ r ∈ refs.toList, ∀ w, r.read earlier = .ok w → box? vals r = flatOf w)
    (hself : (vals[id]?).join = outs[0]? >>= flatOf) :
    Equation G P (dimsOf vals) (pointOf vals) id ⟨_, _, .add (s := s)⟩ refs.toList := by
  obtain ⟨r₁, r₂, hr, he⟩ := binary_eq Op.rule?_add (fun _ _ => rfl) h hvals hself
  rw [hr]
  exact he

/-- A difference of two references subtracts their flat entries. -/
theorem sub_eq {s : Shape}
    (h : (IR.Node.op ⟨_, _, Op.sub (s := s)⟩ refs).eval Pr inputs earlier id =
      .ok outs)
    (hvals : ∀ r ∈ refs.toList, ∀ w, r.read earlier = .ok w → box? vals r = flatOf w)
    (hself : (vals[id]?).join = outs[0]? >>= flatOf) :
    Equation G P (dimsOf vals) (pointOf vals) id ⟨_, _, .sub (s := s)⟩ refs.toList := by
  obtain ⟨r₁, r₂, hr, he⟩ := binary_eq Op.rule?_sub (fun _ _ => rfl) h hvals hself
  rw [hr]
  exact he

/-- An entrywise product of two references multiplies their flat entries. -/
theorem mul_eq {s : Shape}
    (h : (IR.Node.op ⟨_, _, Op.mul (s := s)⟩ refs).eval Pr inputs earlier id =
      .ok outs)
    (hvals : ∀ r ∈ refs.toList, ∀ w, r.read earlier = .ok w → box? vals r = flatOf w)
    (hself : (vals[id]?).join = outs[0]? >>= flatOf) :
    ProductEquation (dimsOf vals) (pointOf vals) id ⟨_, _, .mul (s := s)⟩ refs.toList := by
  obtain ⟨r₁, r₂, hr, he⟩ := binary_eq Op.rule?_mul (fun _ _ => rfl) h hvals hself
  rw [hr]
  exact he

/-- A linear layer reads the reals of its constant weight and bias: its flat value has the width
of its output shape, its input the width of its input shape, and on one leading row it is the
real affine map of its input's flat value. -/
theorem linear_eq {L : Shape} {n m : Nat}
    (h : (IR.Node.op ⟨_, _, Op.linear (lead := L) (inDim := n) (outDim := m)⟩
      refs).eval Pr inputs earlier id = .ok outs)
    (hconsts : Op.Directed.ConstsReal G P earlier refs.toList)
    (hvals : ∀ r ∈ refs.toList, ∀ w, r.read earlier = .ok w → box? vals r = flatOf w)
    (hself : (vals[id]?).join = outs[0]? >>= flatOf) :
    Equation G P (dimsOf vals) (pointOf vals) id
      ⟨_, _, .linear (lead := L) (inDim := n) (outDim := m)⟩ refs.toList := by
  obtain ⟨rW, rb, rx, vW, vb, vx, Wr, br, x, hr, hreadW, hreadb, hreadx, hxW, hxb, hxx, rfl⟩ :=
    invoke_three (Option.some_get Op.rule?_linear_isSome).symm h
  have hx : (vals[rx.node]?).join = some (flatOfSome ⟨_, x⟩) := join_of_box?
    ((hvals rx (by simp [hr]) vx hreadx).trans
      ((flatOf_of_read (Or.inl rfl) hxx).trans (flatOf_value x)))
  rw [hr] at hconsts ⊢
  intro W b hW hb
  obtain rfl := Op.Directed.read_constAt? hconsts (by simp) hW hreadW hxW
  obtain rfl := Op.Directed.read_constAt? hconsts (by simp) hb hreadb hxb
  have hs : (vals[id]?).join = some (flatOfSome ⟨L ++ [m],
      Linear.forwardLeading L ⟨Tensor.map value W, Tensor.map value b⟩ x⟩) := by
    rw [hself, Op.rule?_linear_forward]
    rfl
  refine ⟨dimsOf_eq hx, dimsOf_eq hs, fun hone => ?_⟩
  have hm : (L ++ [m]).size = m := by simp [Shape.size_append, hone]
  have hn : (L ++ [n]).size = n := by simp [Shape.size_append, hone]
  refine ⟨(dimsOf_eq hs).trans hm, (dimsOf_eq hx).trans hn, fun i => ?_⟩
  rw [pointOf_eq hs, pointOf_eq hx]
  exact Op.Directed.flat_linear_one hone W b x i

/-- A sum over every axis in order without `keepdim` is the sum of the flat entries it reads. -/
theorem sum_eq {s : Shape} {axes : List (Fin s.rank)}
    {hop : Tensor.ReduceOp.sum = .max ∨ Tensor.ReduceOp.sum = .min → ∀ d ∈ axes, 0 < s.extent d}
    (h : (IR.Node.op ⟨_, _, Op.reduce .sum axes false hop⟩ refs).eval Pr inputs
      earlier id = .ok outs)
    (hvals : ∀ r ∈ refs.toList, ∀ w, r.read earlier = .ok w → box? vals r = flatOf w)
    (hself : (vals[id]?).join = outs[0]? >>= flatOf) :
    Equation G P (dimsOf vals) (pointOf vals) id ⟨_, _, .reduce .sum axes false hop⟩
      refs.toList := by
  obtain ⟨r, w, x, hr, hread, hx, rfl⟩ :=
    invoke_one (Op.rule?_reduce .sum axes false hop) h
  have hb : (vals[r.node]?).join = some (flatOfSome ⟨s, x⟩) := join_of_box?
    ((hvals r (by simp [hr]) w hread).trans
      ((flatOf_of_read (Or.inl rfl) hx).trans (flatOf_value x)))
  rw [hr]
  intro hall
  obtain rfl : axes = List.finRange s.rank :=
    List.map_injective_iff.mpr Fin.val_injective
      (hall.trans (List.map_coe_finRange_eq_range).symm)
  have hs : (vals[id]?).join = some (flatOfSome
      ⟨_, Tensor.reduce .sum x (List.finRange s.rank) false fun h _ _ => by cases h⟩) := hself
  rw [pointOf_eq hs, pointOf_eq hb, dimsOf_eq hb]
  exact Op.Directed.flat_reduce_sum x _

/-- At each output position, the flat matrix product is the sum over the contraction of the
products of the operand entries its layout pairs. -/
theorem getAtOrZero_matmulFlat (D : IR.MatmulDims) {nl nr : Nat}
    (a : Tensor ℝ [nl]) (b : Tensor ℝ [nr]) {o : Nat} (ho : o < D.outShape.size) :
    getAtOrZero (IR.MatmulDims.matmulFlat D a b) [o] =
      ∑ t : Fin D.inner,
        getAtOrZero a [D.leftIndex o t.val] * getAtOrZero b [D.rightIndex o t.val] := by
  have hs : ((List.range D.inner).map fun t =>
      getAtOrZero a [D.leftIndex o t] * getAtOrZero b [D.rightIndex o t]).sum =
      ∑ t : Fin D.inner,
        getAtOrZero a [D.leftIndex o t.val] * getAtOrZero b [D.rightIndex o t.val] := by
    rw [Fin.sum_univ_def, ← List.map_coe_finRange_eq_range, List.map_map]; rfl
  rw [List.sum_eq_foldl, List.foldl_map] at hs
  rw [getAtOrZero_eq_getScalar _ ⟨o, ho⟩]
  simp only [IR.MatmulDims.matmulFlat, Tensor.getScalar_ofFn]
  exact hs

/-- A matrix product reads two references and is the contraction of their flat entries over its
layout. -/
theorem matmul_eq {bA bB b : Shape} {m n p : Nat} {lv rv : Bool} {hl : lv = true → m = 1}
    {hr : rv = true → p = 1} {hA : Shape.CanBroadcastTo bA b} {hB : Shape.CanBroadcastTo bB b}
    (h : (IR.Node.op ⟨_, _, Op.matmul (batchA := bA) (batchB := bB) (batch := b)
      (m := m) (n := n) (p := p) lv rv hl hr hA hB⟩ refs).eval Pr inputs earlier id = .ok outs)
    (hvals : ∀ r ∈ refs.toList, ∀ w, r.read earlier = .ok w → box? vals r = flatOf w)
    (hself : (vals[id]?).join = outs[0]? >>= flatOf) :
    ProductEquation (dimsOf vals) (pointOf vals) id ⟨_, _, .matmul (n := n) lv rv hl hr hA hB⟩
      refs.toList := by
  obtain ⟨r₁, r₂, w₁, w₂, x₁, x₂, hrefs, hread₁, hread₂, hx₁, hx₂, rfl⟩ :=
    invoke_two (Op.rule?_matmul lv rv hl hr hA hB) h
  have h₁ : (vals[r₁.node]?).join = some (flatOfSome ⟨_, x₁⟩) := join_of_box?
    ((hvals r₁ (by simp [hrefs]) w₁ hread₁).trans
      ((flatOf_of_read (Or.inl rfl) hx₁).trans (flatOf_value x₁)))
  have h₂ : (vals[r₂.node]?).join = some (flatOfSome ⟨_, x₂⟩) := join_of_box?
    ((hvals r₂ (by simp [hrefs]) w₂ hread₂).trans
      ((flatOf_of_read (Or.inl rfl) hx₂).trans (flatOf_value x₂)))
  have hs := hself.trans (congrArg some (flat_matmul hl hr hA hB x₁ x₂))
  rw [hrefs]
  intro o
  rw [pointOf_eq hs, pointOf_eq h₁, pointOf_eq h₂]
  exact getAtOrZero_matmulFlat _ _ _ (o.isLt.trans_eq (dimsOf_eq hs))

/-- A scaled matrix product reads two operands and a scale: the contraction of the operands' flat
entries over its layout times the scale's entry. -/
theorem scaledMatmul_eq {bA bB b : Shape} {m n p : Nat} {lv rv : Bool}
    {hl : lv = true → m = 1} {hr : rv = true → p = 1} {hA : Shape.CanBroadcastTo bA b}
    {hB : Shape.CanBroadcastTo bB b}
    (h : (IR.Node.op ⟨_, _, Op.scaledMatmul (batchA := bA) (batchB := bB) (batch := b)
      (m := m) (n := n) (p := p) lv rv hl hr hA hB⟩ refs).eval Pr inputs earlier id = .ok outs)
    (hvals : ∀ r ∈ refs.toList, ∀ w, r.read earlier = .ok w → box? vals r = flatOf w)
    (hself : (vals[id]?).join = outs[0]? >>= flatOf) :
    ProductEquation (dimsOf vals) (pointOf vals) id
      ⟨_, _, .scaledMatmul (n := n) lv rv hl hr hA hB⟩ refs.toList := by
  obtain ⟨r₁, r₂, r₃, w₁, w₂, w₃, x₁, x₂, x₃, hrefs, hread₁, hread₂, hread₃, hx₁, hx₂, hx₃,
    rfl⟩ := invoke_three (Op.rule?_scaledMatmul lv rv hl hr hA hB) h
  have h₁ : (vals[r₁.node]?).join = some (flatOfSome ⟨_, x₁⟩) := join_of_box?
    ((hvals r₁ (by simp [hrefs]) w₁ hread₁).trans
      ((flatOf_of_read (Or.inl rfl) hx₁).trans (flatOf_value x₁)))
  have h₂ : (vals[r₂.node]?).join = some (flatOfSome ⟨_, x₂⟩) := join_of_box?
    ((hvals r₂ (by simp [hrefs]) w₂ hread₂).trans
      ((flatOf_of_read (Or.inl rfl) hx₂).trans (flatOf_value x₂)))
  have h₃ : (vals[r₃.node]?).join = some (flatOfSome ⟨[], x₃⟩) := join_of_box?
    ((hvals r₃ (by simp [hrefs]) w₃ hread₃).trans (flatOf_of_read (Or.inr (Or.inl rfl)) hx₃))
  have hs := hself.trans (congrArg some (flat_scaledMatmul hl hr hA hB x₁ x₂ x₃))
  rw [hrefs]
  intro o
  have ho := o.isLt.trans_eq (dimsOf_eq hs)
  rw [pointOf_eq hs, pointOf_eq h₁, pointOf_eq h₂, pointOf_eq h₃]
  dsimp only
  rw [getAtOrZero_eq_getScalar _ ⟨o.val, ho⟩, getScalar_map, ← getAtOrZero_eq_getScalar,
    getAtOrZero_matmulFlat _ _ _ ho]

/-- A checked concatenation of flat values holds coordinate `i` of the value of block `j` at the
position `layout.flatEquiv ⟨j, i⟩`. -/
theorem concatFlatValues?_flatEquiv {layout : ConcatLayout} {values : Array (FlatTensor ℝ)}
    {v : FlatTensor ℝ}
    (h : concatFlatValues? layout values = some v) (j : Fin layout.lengths.length)
    (i : Fin (layout.parentShape j).size) {vj : FlatTensor ℝ} (hj : values[j.val]? = some vj) :
    getAtOrZero v.v [(layout.flatEquiv ⟨j, i⟩).val] = getAtOrZero vj.v [i.val] := by
  unfold concatFlatValues? at h
  split at h
  · obtain ⟨inputs, hinputs, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨hshape, h⟩ := Option.dite_none_right_eq_some.mp h
    obtain rfl := Option.some.inj h
    have hv := Tensor.sequenceFinM_get_of_eq_some hinputs j
    rw [hj] at hv
    obtain rfl := Option.some.inj hv
    simp only [ConcatLayout.concat, getAtOrZero_eq_getScalar, getScalar_ofFn]
    rw [Equiv.symm_apply_apply, ← getAtOrZero_eq_getScalar, Op.Directed.getAtOrZero_eqRec]
  · contradiction

/-- A concatenation holds, at the position of coordinate `i` of block `j` in its layout,
coordinate `i` of its reference `j`. -/
theorem cat_eq {s : Shape} {dim : Fin s.rank} {sizes : List Nat}
    (h : (IR.Node.op ⟨_, _, Op.cat dim sizes⟩ refs).eval Pr inputs earlier id =
      .ok outs)
    (hvals : ∀ r ∈ refs.toList, ∀ w, r.read earlier = .ok w → box? vals r = flatOf w)
    (hself : (vals[id]?).join = outs[0]? >>= flatOf) :
    Equation G P (dimsOf vals) (pointOf vals) id ⟨_, _, .cat dim sizes⟩ refs.toList := by
  obtain ⟨values, xs, rule, -, hin, hx, hs, -, rfl⟩ := Op.invoke_eq_ok h
  rw [Op.rule?_cat] at hs
  cases hs
  have hfs := mapM_box?_cat dim sizes hin hx hvals
  have hcat := flat_cat dim sizes (Op.catElementParts dim sizes xs)
    (s.set dim sizes.sum) (by simp)
  intro _ j i
  have hj : (List.ofFn fun k =>
      flatOfSome ⟨_, Op.catElementParts dim sizes xs k⟩)[j.val]? =
      some (flatOfSome ⟨_, Op.catElementParts dim sizes xs j⟩) := by
    simp only [List.getElem?_ofFn, Option.dite_none_right_eq_some]
    exact ⟨j.isLt, rfl⟩
  obtain ⟨r, hrj, hbr⟩ := mapM_getElem? hfs j.val _ hj
  obtain ⟨hlt, hget⟩ := List.getElem?_eq_some_iff.mp hrj
  have hnode : (refs.toList.map (·.node)).toArray[j.val]! = r.node := by
    rw [getElem!_pos _ j.val (by simpa using hlt)]
    simp [hget]
  have hself' : (vals[id]?).join = some (flatOfSome ⟨_, Tensor.cat dim sizes.get
      (Op.catElementParts dim sizes xs) (s.set dim sizes.sum)
      (by simp)⟩) := hself
  rw [hnode, pointOf_eq hself', pointOf_eq (join_of_box? hbr)]
  exact concatFlatValues?_flatEquiv hcat j i (by rw [List.getElem?_toArray]; exact hj)

/-- A permutation of the axes reads a reference of the width of its input shape, and at each
coordinate its flat value is the permutation of any tensor that holds the reference's flat
value. -/
theorem permute_eq {s out : Shape} {σ : Equiv.Perm (Fin s.rank)} {hout : out = s.permute σ}
    (h : (IR.Node.op ⟨_, _, Op.permute σ out hout⟩ refs).eval Pr inputs earlier id =
      .ok outs)
    (hvals : ∀ r ∈ refs.toList, ∀ w, r.read earlier = .ok w → box? vals r = flatOf w)
    (hself : (vals[id]?).join = outs[0]? >>= flatOf) :
    Equation G P (dimsOf vals) (pointOf vals) id ⟨_, _, .permute σ out hout⟩ refs.toList := by
  obtain ⟨r, w, x₀, hr, hread, hx, rfl⟩ := invoke_one (Op.rule?_permute σ out hout) h
  have hb : (vals[r.node]?).join = some (flatOfSome ⟨s, x₀⟩) := join_of_box?
    ((hvals r (by simp [hr]) w hread).trans
      ((flatOf_of_read (Or.inl rfl) hx).trans (flatOf_value x₀)))
  have hs : (vals[id]?).join = some (flatOfSome ⟨out, Tensor.permute x₀ σ out hout⟩) := hself
  rw [hr]
  refine ⟨dimsOf_eq hb, fun x hxv c => ?_⟩
  have hxx : x = x₀ := by
    apply Tensor.Rep.ext
    intro c'
    rw [hxv c', pointOf_eq hb]
    exact Tensor.getAtOrZero_reshape_toFin x₀ c'
  subst hxx
  rw [pointOf_eq hs]
  exact Tensor.getAtOrZero_reshape_toFin _ c

/-- A convolution reads the reals of its constant kernel and bias: its flat value and its input's
have the sizes of their shapes, and at each coordinate it is the operator's rule at `ℝ` on the reals
of the kernel and the bias and on any tensor that holds its input's coordinates. -/
theorem conv_eq {s k : Shape} {r : Nat} {cd : Shape.ConvDims s k r}
    {stride padLow padHigh dilation : Fin r → Nat} {groups : Nat} {lead xs ys : Shape}
    {hlead : s.extent (cd.input ⟨0, by omega⟩) = lead.size}
    {hx : xs = s.expandAxis (cd.input ⟨0, by omega⟩) lead}
    {hy : ys = (s.conv k cd stride padLow padHigh (fun _ => 1) dilation 1).expandAxis
      (cd.output stride padLow padHigh (fun _ => 1) dilation 1 ⟨0, by omega⟩) lead}
    (h : (IR.Node.op ⟨_, _, Op.conv cd stride padLow padHigh dilation groups hlead hx
      hy⟩ refs).eval Pr inputs earlier id = .ok outs)
    (hconsts : Op.Directed.ConstsReal G P earlier refs.toList)
    (hvals : ∀ r ∈ refs.toList, ∀ w, r.read earlier = .ok w → box? vals r = flatOf w)
    (hself : (vals[id]?).join = outs[0]? >>= flatOf) :
    Equation G P (dimsOf vals) (pointOf vals) id
      ⟨_, _, .conv cd stride padLow padHigh dilation groups hlead hx hy⟩ refs.toList := by
  obtain ⟨rule₀, hspec⟩ : ∃ rule₀, (Op.conv cd stride padLow padHigh dilation groups hlead
      hx hy).rule? (α := ℝ) = some rule₀ :=
    ⟨_, Op.rule?_conv cd stride padLow padHigh dilation groups hlead hx hy⟩
  obtain ⟨rK, rb, rx, wK, wb, wx, Kr, br, x₀, hr, hreadK, hreadb, hreadx, hxK, hxb, hxx, rfl⟩ :=
    invoke_three hspec h
  have hb : (vals[rx.node]?).join = some (flatOfSome ⟨xs, x₀⟩) := join_of_box?
    ((hvals rx (by simp [hr]) wx hreadx).trans
      ((flatOf_of_read (Or.inl rfl) hxx).trans (flatOf_value x₀)))
  rw [hr] at hconsts ⊢
  intro K b hK hbias
  obtain rfl := Op.Directed.read_constAt? hconsts (by simp) hK hreadK hxK
  obtain rfl := Op.Directed.read_constAt? hconsts (by simp) hbias hreadb hxb
  have hs : (vals[id]?).join = some (flatOfSome ⟨ys,
      (rule₀.forward (Tensor.map value K, Tensor.map value b, x₀, ())).1⟩) := hself
  refine ⟨dimsOf_eq hb, dimsOf_eq hs, fun rule hrule x hxv c => ?_⟩
  obtain rfl := Option.some.inj (hrule.symm.trans hspec)
  have hxx : x = x₀ := by
    apply Tensor.Rep.ext
    intro c'
    rw [hxv c', pointOf_eq hb]
    exact Tensor.getAtOrZero_reshape_toFin x₀ c'
  subst hxx
  rw [pointOf_eq hs]
  exact Tensor.getAtOrZero_reshape_toFin _ c

/-- A slice along one axis with step one reads its reference at the flat indices of its slice
layout. -/
theorem slice_eq {s : Shape} {axes : List (Fin s.rank)} {starts stops steps : List Nat}
    {hv : Shape.SliceValid s axes starts stops steps}
    (h : (IR.Node.op ⟨_, _, Op.slice axes starts stops steps hv⟩ refs).eval Pr
      inputs earlier id = .ok outs)
    (hvals : ∀ r ∈ refs.toList, ∀ w, r.read earlier = .ok w → box? vals r = flatOf w)
    (hself : (vals[id]?).join = outs[0]? >>= flatOf) :
    Equation G P (dimsOf vals) (pointOf vals) id ⟨_, _, .slice axes starts stops steps hv⟩
      refs.toList := by
  obtain ⟨r, w, x, hr, hread, hx, rfl⟩ :=
    invoke_one (Op.rule?_slice axes starts stops steps hv) h
  have hb : (vals[r.node]?).join = some (flatOfSome ⟨s, x⟩) := join_of_box?
    ((hvals r (by simp [hr]) w hread).trans
      ((flatOf_of_read (Or.inl rfl) hx).trans (flatOf_value x)))
  have hs : (vals[id]?).join =
      some (flatOfSome ⟨_, Tensor.slice x axes starts stops steps hv⟩) := hself
  rw [hr]
  intro axis start stop layout hax hstart hstop hstep hl i
  rw [pointOf_eq hs, pointOf_eq hb]
  exact flat_slice hv x hax hstart hstop hstep hl i

/-- At an operator node that evaluates, the real equations of `Equation` hold at the point of
any table that holds the flat value of the node's first output and of every value its references
read, when each reference to a rounded constant reads that constant's real value. -/
theorem equation_of_eval {o : IR.Operator}
    (h : (IR.Node.op o refs).eval Pr inputs earlier id = .ok outs)
    (hconsts : Op.Directed.ConstsReal G P earlier refs.toList)
    (hvals : ∀ r ∈ refs.toList, ∀ w, r.read earlier = .ok w → box? vals r = flatOf w)
    (hself : (vals[id]?).join = outs[0]? >>= flatOf) :
    Equation G P (dimsOf vals) (pointOf vals) id o refs.toList := by
  obtain ⟨ins, outs', op⟩ := o
  cases op <;> (try cases ‹Op.NumericElement›) <;> (try cases ‹Op.Element›)
  case detach => exact detach_eq h hvals hself
  case reshape => exact reshape_eq h hvals hself
  case add => exact add_eq h hvals hself
  case sub => exact sub_eq h hvals hself
  case linear => exact linear_eq h hconsts hvals hself
  case «reduce» rop _ kd _ =>
    cases rop <;> cases kd <;> first | exact sum_eq h hvals hself | exact True.intro
  case slice => exact slice_eq h hvals hself
  case cat => exact cat_eq h hvals hself
  case permute => exact permute_eq h hvals hself
  case conv => exact conv_eq h hconsts hvals hself
  all_goals exact True.intro

/-- At an operator node that evaluates, the real equations of `ProductEquation` hold at the point
of any table that holds the flat value of the node's first output and of every value its
references read. -/
theorem productEquation_of_eval {o : IR.Operator}
    (h : (IR.Node.op o refs).eval Pr inputs earlier id = .ok outs)
    (hvals : ∀ r ∈ refs.toList, ∀ w, r.read earlier = .ok w → box? vals r = flatOf w)
    (hself : (vals[id]?).join = outs[0]? >>= flatOf) :
    ProductEquation (dimsOf vals) (pointOf vals) id o refs.toList := by
  obtain ⟨ins, outs', op⟩ := o
  cases op <;> (try cases ‹Op.NumericElement›) <;> (try cases ‹Op.Element›)
  case mul => exact mul_eq h hvals hself
  case matmul => exact matmul_eq h hvals hself
  case scaledMatmul => exact scaledMatmul_eq h hvals hself
  all_goals exact True.intro

end Clauses

/-! ## The point of a run -/

section Run

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

variable {G : IR.Graph} {P : IR.Payload α} {Pr : IR.Payload ℝ}
  {inputs : List (Op.Value ℝ)} {values : Array (List (Op.Value ℝ))}

/-- Every reference of an operator node that evaluates reads a value. -/
theorem read_of_eval {earlier : Array (List (Op.Value ℝ))} {id : Nat}
    {o : IR.Operator} {refs : Array IR.Ref} {outs : List (Op.Value ℝ)}
    (h : (IR.Node.op o refs).eval Pr inputs earlier id = .ok outs) :
    ∀ r ∈ refs, ∃ w, r.read earlier = .ok w := by
  have hlist : ∀ {β : Type} {f : IR.Ref → Except String β} {ys : List β},
      (∀ r y, f r = .ok y → ∃ w, r.read earlier = .ok w) → refs.toList.mapM f = .ok ys →
        ∀ r ∈ refs, ∃ w, r.read earlier = .ok w :=
    fun hf hm r hr => read_of_mapM hf hm r (Array.mem_toList_iff.mpr hr)
  simp only [IR.Node.eval] at h
  unfold IR.evalOp at h
  split at h
  · rename_i name shapes output
    cases hc : IR.runCustom Pr earlier id refs name shapes output with
    | error e => simp [hc, bind, Except.bind] at h
    | ok t =>
      unfold IR.runCustom at hc
      obtain ⟨ts, hm⟩ : ∃ ts, refs.mapM (fun r => do IR.someTensor? (← r.read earlier)) =
          .ok ts := by
        cases hm : refs.mapM (fun r => do IR.someTensor? (← r.read earlier)) with
        | ok ts => exact ⟨ts, rfl⟩
        | error e =>
          simp only [bind, Except.bind] at hm
          by_cases hs : refs.size = shapes.length <;> cases hp : Pr.custom? id <;>
            simp [hs, hp, hm, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at hc
      rw [Array.mapM_eq_mapM_toList] at hm
      cases hl : refs.toList.mapM (fun r => do IR.someTensor? (← r.read earlier)) with
      | error e => simp [hl, Functor.map, Except.map] at hm
      | ok l =>
        refine hlist (fun r y hy => ?_) hl
        cases hr : r.read earlier with
        | error e => simp [hr, bind, Except.bind] at hy
        | ok w => exact ⟨w, rfl⟩
  · obtain ⟨vs, -, -, -, hin, -⟩ := Op.invoke_eq_ok h
    exact hlist (fun r y hy => ⟨y, hy⟩) hin

/-- A successful run reads, at every operator node, only earlier nodes. -/
theorem topoSorted_of_run (hrun : G.run Pr inputs = .ok values) : TopoSorted G := by
  intro id _ o refs hn r hr
  obtain ⟨pre, outs, hsize, -, heval, -⟩ := IR.Graph.run_step hrun hn
  obtain ⟨w, hw⟩ := read_of_eval heval r hr
  exact hsize ▸ IR.Ref.read_lt hw

/-- In a successful real run on a payload holding the reals of the rounded constants, the run's
point at a constant node whose rounded value has a flat tensor `c` has the width of `c` and the
reals of its entries. -/
theorem const_eq_of_run (hrun : G.run Pr inputs = .ok values) (hP : Op.Directed.RealConsts Pr P)
    {id : Nat} {c : FlatTensor α} (hc : (constValue? G P id >>= flatOf) = some c) :
    dimsOf (runVals values) id = c.n ∧
      ∀ i : Fin c.n, pointOf (runVals values) id i.val = value (getAtOrZero c.v [i.val]) := by
  obtain ⟨u, hu, hfu⟩ := Option.bind_eq_some_iff.mp hc
  obtain ⟨p, hn⟩ : ∃ p, G.nodes[id]? = some (.const p) := by
    unfold constValue? at hu
    split at hu
    · exact ⟨_, by assumption⟩
    · cases hu
  obtain ⟨pre, outs, -, -, heval, hout⟩ := IR.Graph.run_step hrun hn
  obtain ⟨w, hw, rfl⟩ := const_eval hn heval
  rw [Op.Directed.constValue?_real hP, hu, Option.map_some] at hw
  obtain rfl : w = Op.Directed.realValue u := (Option.some.inj hw).symm
  have ht : ((runVals values)[id]?).join = some ⟨c.n, Tensor.map value c.v⟩ := by
    rw [join_runVals]
    simp [runVal, hout, Op.Directed.flatOf_realValue, hfu]
  refine ⟨dimsOf_eq ht, fun i => ?_⟩
  rw [pointOf_eq ht]
  dsimp only
  rw [getAtOrZero_eq_getScalar _ i, getScalar_map, ← getAtOrZero_eq_getScalar]

/-- The flat values of a successful real run on a payload holding the reals of the rounded
constants are a real point of the graph (`GraphPoint`) for any table whose boxes enclose the run,
when the designated input is an input node whose flat value has the width of the affine forms. -/
theorem graphPoint_of_run (hrun : G.run Pr inputs = .ok values)
    (hP : Op.Directed.RealConsts Pr P) {ibp : Array (Option (FlatBox α))}
    (hibp : ∀ id B, (ibp[id]?).join = some B →
      ∃ t, runVal values id = some t ∧ Op.Directed.Encloses B t)
    {ctx : AffineCtx} (hin : ∃ k p, G.nodes[ctx.inputId]? = some (.input k p))
    (hdim : dimsOf (runVals values) ctx.inputId = ctx.inputDim) :
    GraphPoint G P ibp ctx (dimsOf (runVals values)) (pointOf (runVals values)) where
  input_node := hin
  input_dim := hdim
  topo := topoSorted_of_run hrun
  ibp_encloses id B hB := by
    obtain ⟨t, ht, hBt⟩ := hibp id B hB
    rw [← join_runVals] at ht
    rw [dimsOf_eq ht, pointOf_eq ht]
    exact hBt
  const_eq _ _ hc := const_eq_of_run hrun hP hc

/-- In a successful real run on a payload holding the reals of the rounded constants, every
operator node satisfies its real equations of `Equation` at the run's point. -/
theorem equation_of_run (hrun : G.run Pr inputs = .ok values)
    (hP : Op.Directed.RealConsts Pr P) {id : Nat} {o : IR.Operator} {refs : Array IR.Ref}
    (hn : G.nodes[id]? = some (.op o refs)) :
    Equation G P (dimsOf (runVals values)) (pointOf (runVals values)) id o refs.toList := by
  obtain ⟨pre, outs, hsize, hagree, heval, hout⟩ := IR.Graph.run_step hrun hn
  exact equation_of_eval heval (Op.Directed.constsReal_of_run hP hrun hagree hsize refs.toList)
    (fun r _ w hr => box?_runVals_of_run hrun hagree hsize hr)
    (by rw [join_runVals]; simp [runVal, hout])

/-- In a successful real run, every operator node satisfies its real equations of
`ProductEquation` at the run's point. -/
theorem productEquation_of_run (hrun : G.run Pr inputs = .ok values) {id : Nat}
    {o : IR.Operator} {refs : Array IR.Ref} (hn : G.nodes[id]? = some (.op o refs)) :
    ProductEquation (dimsOf (runVals values)) (pointOf (runVals values)) id o refs.toList := by
  obtain ⟨pre, outs, hsize, hagree, heval, hout⟩ := IR.Graph.run_step hrun hn
  exact productEquation_of_eval heval (fun r _ w hr => box?_runVals_of_run hrun hagree hsize hr)
    (by rw [join_runVals]; simp [runVal, hout])

/-- A port reads a value only at the value's own port, or as a frozen or detached port of a value
tensor of its shape. -/
theorem port_of_read {p : Port} {x : Op.Value ℝ} {y : p.Carrier ℝ}
    (h : p.read x = .ok y) :
    x.1 = p ∨ x.1 = .value p.shape ∧ (p.kind = .frozen ∨ p.kind = .detached) := by
  unfold Port.read at h
  split at h
  · split at h
    · rename_i hst
      subst hst
      exact Or.inr ⟨rfl, Or.inl rfl⟩
    · simp [throw, throwThe, MonadExceptOf.throw] at h
  · split at h
    · rename_i hst
      subst hst
      exact Or.inr ⟨rfl, Or.inr rfl⟩
    · simp [throw, throwThe, MonadExceptOf.throw] at h
  · split at h
    · exact Or.inl ‹_›
    · simp [throw, throwThe, MonadExceptOf.throw] at h

/-- A port reads a value with a flat tensor as a tensor with the same flat value. -/
theorem flatOf_read {p : Port} {x : Op.Value ℝ} {y : p.Carrier ℝ} {t : FlatTensor ℝ}
    (ht : flatOf x = some t) (h : p.read x = .ok y) : flatOf ⟨p, y⟩ = some t := by
  refine (flatOf_of_read ?_ h).symm.trans ht
  obtain ⟨⟨e, u⟩, z⟩ := x
  rcases port_of_read h with hq | ⟨-, hp⟩
  · subst hq
    cases e <;> simp_all [flatOf]
  · exact Or.inr hp

/-- A flat value has the size of its port's shape. -/
theorem n_of_flatOf {w : Op.Value ℝ} {t : FlatTensor ℝ} (h : flatOf w = some t) :
    t.n = w.1.shape.size := by
  obtain ⟨⟨d, s⟩, x⟩ := w
  cases d <;> simp [flatOf] at h <;> (subst h; rfl)

/-- In a successful run, a node with a flat value has that value's width as `outSize?`. -/
theorem outSize?_of_run (hrun : G.run Pr inputs = .ok values) {id : Nat} {t : FlatTensor ℝ}
    (h : runVal values id = some t) : outSize? G id = some t.n := by
  simp only [runVal, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨w, ⟨outs, hout, hw⟩, hf⟩ := h
  obtain ⟨n, hn⟩ : ∃ n, G.nodes[id]? = some n :=
    ⟨_, Array.getElem?_eq_getElem (by
      rw [← IR.Graph.run_size hrun]; exact (Array.getElem?_eq_some_iff.mp hout).1)⟩
  have hhead : n.outs[0]? = some w.1 := by
    rw [← IR.Graph.run_ports hrun hn hout, List.getElem?_map, hw]
    rfl
  rw [n_of_flatOf hf]
  unfold outSize?
  rw [hn]
  cases n with
  | input _ p => simp_all [IR.Node.outs]
  | const p => simp_all [IR.Node.outs]
  | op o refs =>
    obtain ⟨ins, outs', op⟩ := o
    simp only [IR.Node.outs] at hhead
    simp [List.head?_eq_getElem?, hhead]

/-- The node that `inputId? G k` finds is an input node of graph input `k`. -/
theorem node_of_inputId? {k id : Nat} (h : inputId? G k = some id) :
    ∃ p, G.nodes[id]? = some (.input k p) := by
  unfold inputId? at h
  have hp := List.find?_some h
  split at hp
  · rename_i j p hn
    exact ⟨p, by rw [hn, beq_iff_eq.mp hp]⟩
  · cases hp

omit [BoundOps α] [LawfulBoundOps α] in
/-- The designated input of an output context is a node of graph input `0`, and the affine forms
have the width of the box. -/
theorem input_of_outputContext {xB : FlatBox α} {i out : Nat} {ctx : AffineCtx}
    (h : outputContext G xB i = .ok (out, ctx)) :
    (∃ p, G.nodes[ctx.inputId]? = some (.input 0 p)) ∧ ctx.inputDim = xB.dim := by
  unfold outputContext at h
  cases hr : G.outputs[i]? with
  | none => simp [hr, throw, throwThe, MonadExceptOf.throw] at h
  | some r =>
    by_cases h0 : r.out = 0
    · cases hid : inputId? G 0 with
      | none => simp [hr, hid, h0, throw, throwThe, MonadExceptOf.throw] at h
      | some id =>
        simp [hr, hid, h0, pure, Except.pure] at h
        obtain ⟨-, rfl⟩ := h
        exact ⟨node_of_inputId? hid, rfl⟩
    · simp [hr, h0, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h

/-- The seed boxes `inputSeeds G [xB]` enclose the flat values of the input nodes of a successful
run whose graph input `0` is a real tensor that `xB` encloses. -/
theorem inputSeeds_encloses (hrun : G.run Pr inputs = .ok values) {xB : FlatBox α}
    (hx : ∀ x, inputs[0]? = some x → ∃ t, flatOf x = some t ∧ Op.Directed.Encloses xB t) :
    ∀ id k p B, G.nodes[id]? = some (.input k p) → inputSeeds G [xB] id = some B →
      ∃ t, runVal values id = some t ∧ Op.Directed.Encloses B t := by
  intro id k p B hn hB
  simp only [inputSeeds, hn] at hB
  obtain ⟨rfl, rfl⟩ : k = 0 ∧ xB = B := by
    rcases k with _ | k <;> simp at hB
    exact ⟨rfl, hB⟩
  obtain ⟨pre, outs, -, -, heval, hout⟩ := IR.Graph.run_step hrun hn
  obtain ⟨x, y, hx0, hy, rfl⟩ := input_eval heval
  obtain ⟨t, ht, hBt⟩ := hx x hx0
  exact ⟨t, by simp [runVal, hout, flatOf_read ht hy], hBt⟩

end Run

/-! ## The CROWN queries on a run -/

section Queries

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

variable {G : IR.Graph} {P : IR.Payload α} {Pr : IR.Payload ℝ}
  {inputs : List (Op.Value ℝ)} {values : Array (List (Op.Value ℝ))}

omit [LawfulBoundOps α] in
/-- An operator that `ibp?` boxes has one output, a value tensor. -/
theorem outs_of_ibp? [NonlinearBoundOps α] [Op.Capabilities α]
    {boxes : Array (Option (FlatBox α))} {ins outs : List Port} {op : Op ins outs}
    {refs : List IR.Ref} {B : FlatBox α} (h : ibp? G P boxes ⟨ins, outs, op⟩ refs = some B) :
    ∃ t, outs = [.value t] := by
  cases op <;> (try cases ‹Op.NumericElement›) <;> (try cases ‹Op.Element›) <;>
    (try cases ‹DivRounding›) <;>
    first | exact ⟨_, rfl⟩ | exact (Option.some_ne_none B (h.symm.trans rfl)).elim

/-- In a successful real run on a payload holding the reals of the rounded constants, a node that
`step?` at `ibp?` boxes has a flat value. An input node with a seed box is assumed to have one
(`hseeds`). -/
theorem runVal_of_step?_ibp? [NonlinearBoundOps α] [Op.Capabilities α]
    (hrun : G.run Pr inputs = .ok values) (hP : Op.Directed.RealConsts Pr P)
    {seeds : Nat → Option (FlatBox α)} {tbl : Array (Option (FlatBox α))} {id : Nat}
    {B : FlatBox α} (h : step? G P (ibp? G P) seeds tbl id = some B)
    (hseeds : ∀ k p, G.nodes[id]? = some (.input k p) → seeds id = some B →
      ∃ t, runVal values id = some t) :
    ∃ t, runVal values id = some t := by
  unfold step? at h
  replace h := (Option.filter_eq_some_iff.mp h).1
  cases hn : G.nodes[id]? with
  | none => simp [hn] at h
  | some n =>
    rw [hn] at h
    obtain ⟨pre, outs, -, -, heval, hout⟩ := IR.Graph.run_step hrun hn
    cases n with
    | input k p => exact hseeds k p hn h
    | const p =>
      obtain ⟨c, hc, -⟩ := Option.map_eq_some_iff.mp h
      obtain ⟨u, hu, hfu⟩ := Option.bind_eq_some_iff.mp hc
      obtain ⟨w, hw, rfl⟩ := const_eval hn heval
      rw [Op.Directed.constValue?_real hP, hu, Option.map_some] at hw
      obtain rfl : w = Op.Directed.realValue u := (Option.some.inj hw).symm
      rw [runVal_eq hout]
      simp [Op.Directed.flatOf_realValue, hfu]
    | op o refs =>
      simp only at h
      split at h
      · obtain ⟨ins, outs', op⟩ := o
        obtain ⟨q, rfl⟩ := outs_of_ibp? h
        have hports := IR.Graph.run_ports hrun hn hout
        simp only [IR.Node.outs] at hports
        obtain ⟨w, rfl, hw⟩ := List.map_eq_singleton_iff.mp hports
        obtain ⟨⟨d, s⟩, x⟩ := w
        obtain ⟨rfl, rfl⟩ : d = .value ∧ s = q := by simpa [Port.value] using hw
        rw [runVal_eq hout]
        exact ⟨_, rfl⟩
      · cases h

/-- Seed boxes that enclose the flat values of the input nodes of a successful run enclose the
graph inputs those nodes read, the input premise of `Op.Directed.encloses_of_run`. -/
theorem seedsRead_of_run (hrun : G.run Pr inputs = .ok values)
    {seeds : Nat → Option (FlatBox α)}
    (hseeds : ∀ id k p B, G.nodes[id]? = some (.input k p) → seeds id = some B →
      ∃ t, runVal values id = some t ∧ Op.Directed.Encloses B t) :
    ∀ id k p, G.nodes[id]? = some (.input k p) → ∀ B, seeds id = some B →
      ∀ x y v, inputs[k]? = some x → p.read x = .ok y → flatOf ⟨p, y⟩ = some v →
        Op.Directed.Encloses B v := by
  intro id k p hn B hB x y v hx hy hv
  obtain ⟨t, ht, hBt⟩ := hseeds id k p B hn hB
  obtain ⟨pre, outs, -, -, heval, hout⟩ := IR.Graph.run_step hrun hn
  obtain ⟨x', y', hx', hy', rfl⟩ := input_eval heval
  rw [hx, Option.some.injEq] at hx'
  subst hx'
  rw [hy, Except.ok.injEq] at hy'
  subst hy'
  rw [show runVal values id = some v by simp [runVal, hout, hv], Option.some.injEq] at ht
  subst ht
  exact hBt

/-- Every node that `Op.runIBP` boxes on the seeds `inputSeeds G [xB]` has a flat value in a
successful real run on a payload holding the reals of the rounded constants, and its box encloses
that value. The run's graph input `0` must be a real tensor that `xB` encloses. -/
theorem runIBP_encloses [NonlinearBoundOps α] [LawfulNonlinearBoundOps α]
    (hrun : G.run Pr inputs = .ok values)
    (hP : Op.Directed.RealConsts Pr P) {xB : FlatBox α}
    (hx : ∀ x, inputs[0]? = some x → ∃ t, flatOf x = some t ∧ Op.Directed.Encloses xB t)
    (id : Nat) (B : FlatBox α) (hB : ((runIBP G P (inputSeeds G [xB]))[id]?).join = some B) :
    ∃ t, runVal values id = some t ∧ Op.Directed.Encloses B t := by
  have hseeds := inputSeeds_encloses hrun hx
  cases hj : (runIBP G P (inputSeeds G [xB]))[id]? with
  | none => simp [hj] at hB
  | some e =>
    simp only [hj, Option.join_some] at hB
    subst hB
    have hlt := (Array.getElem?_eq_some_iff.mp hj).1
    have hid : id < G.nodes.size := by simpa [runIBP, size_tabulate] using hlt
    have hstep : step? G P (ibp? G P) (inputSeeds G [xB])
        (tabulate (step? G P (ibp? G P) (inputSeeds G [xB])) id) id = some B := by
      have h := hj
      unfold runIBP at h
      rw [tabulate_getElem?_self hid] at h
      exact Option.some.inj h
    obtain ⟨t, ht⟩ := runVal_of_step?_ibp? hrun hP hstep fun k p hn hs =>
      (hseeds id k p B hn hs).imp fun _ h => h.1
    refine ⟨t, ht, Op.Directed.encloses_of_run G P Pr _ inputs values hP
      (seedsRead_of_run hrun hseeds) hrun id hid B t ?_ ht⟩
    rw [getElem!_pos _ id hlt]
    exact Option.some.inj ((Array.getElem?_eq_getElem hlt).symm.trans hj)

/-- Over any table whose boxes enclose a successful real run, `affineBounds?` succeeds only when `C`
has one column per coordinate of the flat value `t` of node `out`, and its rows then bound the rows
of `C · t` at the flat value `x` of the designated input. The payload must hold the reals of the
rounded constants, and the designated input must be an input node whose flat value has the width
of the affine forms. -/
theorem affineBounds?_of_run [Op.Capabilities α] (hrun : G.run Pr inputs = .ok values)
    (hP : Op.Directed.RealConsts Pr P)
    {ibp : Array (Option (FlatBox α))}
    (hibp : ∀ id B, (ibp[id]?).join = some B →
      ∃ t, runVal values id = some t ∧ Op.Directed.Encloses B t)
    {ctx : AffineCtx} (hin : ∃ k p, G.nodes[ctx.inputId]? = some (.input k p)) {x : FlatTensor ℝ}
    (hx : runVal values ctx.inputId = some x) (hxd : x.n = ctx.inputDim) {out : Nat}
    {t : FlatTensor ℝ} (ht : runVal values out = some t) {k n : Nat} (C : Tensor α [k, n])
    {bounds : FlatAffineBounds α} (h : affineBounds? G P ctx ibp out C = some bounds) :
    bounds.inDim = ctx.inputDim ∧ bounds.outDim = k ∧ n = t.n ∧
      AffineRowsEnclose bounds (fun i => getAtOrZero x.v [i])
        (fun r => Directed.dot n (fun j => value (getAtOrZero C [r, j]))
          fun j => getAtOrZero t.v [j]) := by
  have hx' := (join_runVals values ctx.inputId).trans hx
  have ht' := (join_runVals values out).trans ht
  have hd : outSize? G out = some (dimsOf (runVals values) out) := by
    rw [dimsOf_eq ht']
    exact outSize?_of_run hrun ht
  have hb := affineBounds?_encloses (graphPoint_of_run hrun hP hibp hin ((dimsOf_eq hx').trans hxd))
    (fun k o refs hn => equation_of_run hrun hP hn)
    (fun k o refs hn => productEquation_of_run hrun hn) hd C h
  rwa [dimsOf_eq ht', pointOf_eq ht', pointOf_eq hx'] at hb

/-- The rows of `nodeBounds?` over any table whose boxes enclose a successful real run enclose the
flat value `t` of node `out` at the flat value `x` of the designated input, under the premises of
`affineBounds?_of_run`. -/
theorem nodeBounds?_of_run [Op.Capabilities α] (hrun : G.run Pr inputs = .ok values)
    (hP : Op.Directed.RealConsts Pr P)
    {ibp : Array (Option (FlatBox α))}
    (hibp : ∀ id B, (ibp[id]?).join = some B →
      ∃ t, runVal values id = some t ∧ Op.Directed.Encloses B t)
    {ctx : AffineCtx} (hin : ∃ k p, G.nodes[ctx.inputId]? = some (.input k p)) {x : FlatTensor ℝ}
    (hx : runVal values ctx.inputId = some x) (hxd : x.n = ctx.inputDim) {out : Nat}
    {t : FlatTensor ℝ} (ht : runVal values out = some t) {bounds : FlatAffineBounds α}
    (h : nodeBounds? G P ctx ibp out = some bounds) :
    bounds.inDim = ctx.inputDim ∧ bounds.outDim = t.n ∧
      AffineRowsEnclose bounds (fun i => getAtOrZero x.v [i]) fun i => getAtOrZero t.v [i] := by
  have hx' := (join_runVals values ctx.inputId).trans hx
  have ht' := (join_runVals values out).trans ht
  have hd : outSize? G out = some (dimsOf (runVals values) out) := by
    rw [dimsOf_eq ht']
    exact outSize?_of_run hrun ht
  have hb := nodeBounds?_encloses (graphPoint_of_run hrun hP hibp hin ((dimsOf_eq hx').trans hxd))
    (fun k o refs hn => equation_of_run hrun hP hn)
    (fun k o refs hn => productEquation_of_run hrun hn) hd h
  rwa [dimsOf_eq ht', pointOf_eq ht', pointOf_eq hx'] at hb

/-- Over any table whose boxes enclose a successful real run, the box of `backwardObjectiveBox?`
at graph output `i` encloses the objective `obj · t`, where `t` is the flat value of the output's
node and `obj` has its width. The payload must hold the reals of the rounded constants, and graph
input `0` must be a real tensor that `xB` encloses. -/
theorem backwardObjectiveBox?_of_run [Op.Capabilities α] [NonlinearBoundOps α]
    (hrun : G.run Pr inputs = .ok values) (hP : Op.Directed.RealConsts Pr P)
    {ibp : Array (Option (FlatBox α))}
    (hibp : ∀ id B, (ibp[id]?).join = some B →
      ∃ t, runVal values id = some t ∧ Op.Directed.Encloses B t)
    {xB : FlatBox α}
    (hxB : ∀ x, inputs[0]? = some x → ∃ t, flatOf x = some t ∧ Op.Directed.Encloses xB t)
    {i out : Nat} {ctx : AffineCtx} (hctx : outputContext G xB i = .ok (out, ctx))
    {t : FlatTensor ℝ} (ht : runVal values out = some t) {obj : FlatTensor α} (hobj : obj.n = t.n)
    {result : FlatBox α} (h : backwardObjectiveBox? G P ibp xB i obj = .ok result) :
    RowEncloses result 1 fun _ => Directed.dot t.n
      (fun j => value (getAtOrZero obj.v [j])) fun j => getAtOrZero t.v [j] := by
  obtain ⟨⟨p, hp⟩, hdim⟩ := input_of_outputContext hctx
  obtain ⟨x, hx, hxBx⟩ :=
    inputSeeds_encloses hrun hxB ctx.inputId 0 p xB hp (by simp [inputSeeds, hp])
  have hx' := (join_runVals values ctx.inputId).trans hx
  have ht' := (join_runVals values out).trans ht
  have hxd : x.n = ctx.inputDim := hxBx.1.symm.trans hdim.symm
  have hb := backwardObjectiveBox?_encloses hctx
    (graphPoint_of_run hrun hP hibp ⟨0, p, hp⟩ ((dimsOf_eq hx').trans hxd))
    (fun k o refs hn => equation_of_run hrun hP hn)
    (fun k o refs hn => productEquation_of_run hrun hn)
    (by rw [pointOf_eq hx', ← hxd]; exact hxBx) (hobj.trans (dimsOf_eq ht').symm) h
  simpa only [objectiveValue, dimsOf_eq ht', pointOf_eq ht'] using hb

/-- The CROWN box of graph output `i` encloses the flat value `t` of its node `out` in a successful
real run, when the payload holds the reals of the rounded constants and graph input `0` is a real
tensor that `xB` encloses. -/
theorem outputBoxCROWN?_of_run [NonlinearBoundOps α] [LawfulNonlinearBoundOps α]
    (hrun : G.run Pr inputs = .ok values)
    (hP : Op.Directed.RealConsts Pr P)
    {xB : FlatBox α}
    (hxB : ∀ x, inputs[0]? = some x → ∃ t, flatOf x = some t ∧ Op.Directed.Encloses xB t)
    {i out : Nat} {ctx : AffineCtx} (hctx : outputContext G xB i = .ok (out, ctx))
    {t : FlatTensor ℝ} (ht : runVal values out = some t) {result : FlatBox α}
    (h : outputBoxCROWN? G P xB i = .ok result) :
    Op.Directed.Encloses result t := by
  obtain ⟨⟨p, hp⟩, hdim⟩ := input_of_outputContext hctx
  obtain ⟨x, hx, hxBx⟩ :=
    inputSeeds_encloses hrun hxB ctx.inputId 0 p xB hp (by simp [inputSeeds, hp])
  have hx' := (join_runVals values ctx.inputId).trans hx
  have ht' := (join_runVals values out).trans ht
  have hxd : x.n = ctx.inputDim := hxBx.1.symm.trans hdim.symm
  have hb := outputBoxCROWN?_encloses hctx
    (graphPoint_of_run hrun hP (runIBP_encloses hrun hP hxB) ⟨0, p, hp⟩
      ((dimsOf_eq hx').trans hxd))
    (fun k o refs hn => equation_of_run hrun hP hn)
    (fun k o refs hn => productEquation_of_run hrun hn)
    (by rw [dimsOf_eq ht']; exact outSize?_of_run hrun ht)
    (by rw [pointOf_eq hx', ← hxd]; exact hxBx) h
  rwa [dimsOf_eq ht', pointOf_eq ht'] at hb

end Queries

/-! ## Full CROWN on a run -/

section Full

/-- Every entry of `tabulate step m` has the property `Q` of its node when, at each node `k`, the
step has it whenever every entry of `tabulate step k` has it. -/
theorem tabulate_induction {β : Type} {step : Array (Option β) → Nat → Option β}
    (Q : Nat → β → Prop)
    (h : ∀ k, (∀ j b, ((tabulate step k)[j]?).join = some b → Q j b) →
      ∀ b, step (tabulate step k) k = some b → Q k b) :
    ∀ m j b, ((tabulate step m)[j]?).join = some b → Q j b
  | 0, j, b, hb => by simp [tabulate] at hb
  | m + 1, j, b, hb => by
    simp only [tabulate, Array.getElem?_push, size_tabulate] at hb
    split at hb
    · rename_i hj
      subst hj
      exact h j (tabulate_induction Q h j) b hb
    · exact tabulate_induction Q h m j b hb

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

variable {G : IR.Graph} {P : IR.Payload α} {Pr : IR.Payload ℝ}
  {inputs : List (Op.Value ℝ)} {values : Array (List (Op.Value ℝ))}

/-- The rows of the identity bounds enclose their input at every point. -/
theorem boundsIdentity_encloses (n : Nat) (x : Nat → ℝ) :
    AffineRowsEnclose (boundsIdentity (α := α) n) x x := by
  have h : ∀ i : Fin n,
      (∑ j : Fin n, value (Tensor.get2 (affIdentity (α := α) n).weights i j) * x j.val) +
        value ((affIdentity (α := α) n).bias.getScalar i) = x i.val := by
    intro i
    simp [affIdentity, get2_identity, apply_ite (f := value),
      LawfulBoundOps.toReal_one, LawfulBoundOps.toReal_zero]
  intro i
  exact ⟨(h i).le, (h i).ge⟩

/-- The bounds `tighterRows` picks row by row from two bounds enclose what both
enclose. -/
theorem tighterRows_encloses (xB : FlatBox α) {a b : FlatAffineBounds α} {x y : Nat → ℝ}
    (ha : AffineRowsEnclose a x y) (hb : AffineRowsEnclose b x y) :
    AffineRowsEnclose (tighterRows xB a b) x y := by
  unfold tighterRows
  split
  · rename_i h
    obtain ⟨na, ma, la, ua⟩ := a
    obtain ⟨nb, mb, lb, ub⟩ := b
    obtain ⟨h₁, h₂, h₃⟩ := h
    dsimp only at h₁ h₂ h₃
    subst h₁ h₃
    subst h₂
    have hm : ∀ {m k : Nat} (f : Fin m → Fin k → α) (i : Fin m) (j : Fin k),
        Tensor.get2 (Tensor.matrix f) i j = f i j := fun f i j => by
      simp [Tensor.matrix, Tensor.get2]
    intro i
    have hai := ha i
    have hbi := hb i
    dsimp only at hai hbi ⊢
    simp only [hm, getScalar_ofFn]
    constructor
    · split_ifs
      · simpa only [getAtOrZero_eq_get2, Tensor.getAtOrZero_eq_getScalar] using hbi.1
      · exact hai.1
    · split_ifs
      · simpa only [getAtOrZero_eq_get2, Tensor.getAtOrZero_eq_getScalar] using hbi.2
      · exact hai.2
  · exact ha

/-- When `evalCROWNOutputBox?` evaluates the bound at index `out` on the box `xB`, `xB` has the
evaluation's input width and the result encloses what the bound's rows enclose at each point that
`xB` encloses. -/
theorem evalCROWNOutputBox?_encloses {bounds : Array (Option (FlatAffineBounds α))} {out : Nat}
    {b : FlatAffineBounds α} (hb : bounds[out]? = some (some b)) {xB : FlatBox α} {n : Nat}
    {box : FlatBox α} (h : evalCROWNOutputBox? bounds xB out n = .ok box) :
    xB.dim = n ∧ ∀ x y, RowEncloses xB n x → AffineRowsEnclose b x y →
      RowEncloses box b.outDim y := by
  unfold evalCROWNOutputBox? at h
  simp only [hb, pure, Except.pure, bind, Except.bind] at h
  split at h
  · rename_i hin
    split at h
    · rename_i hxB
      obtain rfl := Except.ok.inj h
      exact ⟨hxB, fun x y hx hy => affineBoundsEval_encloses b xB x y (by rw [hin]; exact hx) hy⟩
    · simp [throw, throwThe, MonadExceptOf.throw] at h
  · simp [throw, throwThe, MonadExceptOf.throw] at h

/-- In a successful real run on a payload holding the reals of the rounded constants, a node `id`
that `crownStep` boxes from a table `tbl` of the nodes before it has a flat value, and the box
encloses it. The boxes of `tbl`, of the supplied table `ibp` and of the seeds at the input nodes
must enclose the run, and `xB` must enclose the flat value of the designated input, which is an
input node. -/
theorem crownStep_encloses [NonlinearBoundOps α] [LawfulNonlinearBoundOps α]
    (hrun : G.run Pr inputs = .ok values) (hP : Op.Directed.RealConsts Pr P)
    {seeds : Nat → Option (FlatBox α)}
    (hseeds : ∀ id k p B, G.nodes[id]? = some (.input k p) → seeds id = some B →
      ∃ t, runVal values id = some t ∧ Op.Directed.Encloses B t)
    {ibp : Array (Option (FlatBox α))}
    (hibp : ∀ id B, (ibp[id]?).join = some B →
      ∃ t, runVal values id = some t ∧ Op.Directed.Encloses B t)
    {ctx : AffineCtx} (hin : ∃ k p, G.nodes[ctx.inputId]? = some (.input k p)) {xB : FlatBox α}
    {x : FlatTensor ℝ} (hx : runVal values ctx.inputId = some x) (hxB : Op.Directed.Encloses xB x)
    {tbl : Array (Option (FlatBox α))}
    (htbl : ∀ id B, (tbl[id]?).join = some B →
      ∃ t, runVal values id = some t ∧ Op.Directed.Encloses B t)
    {id : Nat} (hsize : tbl.size = id) {B : FlatBox α}
    (h : crownStep G P ctx seeds ibp xB tbl id = some B) :
    ∃ t, runVal values id = some t ∧ Op.Directed.Encloses B t := by
  unfold crownStep at h
  obtain ⟨B₀, h₀, h⟩ := Option.bind_eq_some_iff.mp h
  obtain ⟨t, ht⟩ := runVal_of_step?_ibp? hrun hP h₀ fun k p hn hs =>
    (hseeds id k p B₀ hn hs).imp fun _ h => h.1
  have hB₀ : Op.Directed.Encloses B₀ t :=
    Op.Directed.step?_encloses G P Pr Op.ibp?_encloses seeds inputs values hP
      (seedsRead_of_run hrun hseeds) hrun tbl id (fun o refs hn r hr hlt Br u hBr hu => by
        obtain ⟨u', hu', hBu⟩ := htbl r.node Br (join_of_box? hBr)
        rw [← join_runVals, join_of_box? hu, Option.some.injEq] at hu'
        exact hu' ▸ hBu) B₀ t h₀ ht
  -- The node's affine bounds over the table that holds a box `B₁` of it, evaluated on `xB`.
  have key : ∀ B₁ : FlatBox α, Op.Directed.Encloses B₁ t → ∀ box,
      ((affineBounds? G P ctx (tbl.push (some B₁)) id
        (Tensor.matrix fun (i j : Fin B₁.dim) => if i = j then (1 : α) else 0)).bind
          fun b => (evalCROWNOutputBox? #[some b] xB 0 ctx.inputDim).toOption) = some box →
      Op.Directed.Encloses (meetBox box B₁) t := by
    intro B₁ hB₁ box hbox
    obtain ⟨b, hb, hbox⟩ := Option.bind_eq_some_iff.mp hbox
    obtain ⟨hd, hE⟩ : xB.dim = ctx.inputDim ∧ ∀ x y, RowEncloses xB ctx.inputDim x →
        AffineRowsEnclose b x y → RowEncloses box b.outDim y := by
      cases he : evalCROWNOutputBox? #[some b] xB 0 ctx.inputDim with
      | error e => simp [he, Except.toOption] at hbox
      | ok box' =>
        simp only [he, Except.toOption, Option.some.injEq] at hbox
        subst hbox
        exact evalCROWNOutputBox?_encloses rfl he
    have hxd : x.n = ctx.inputDim := hxB.1.symm.trans hd
    have hpush : ∀ j S, ((tbl.push (some B₁))[j]?).join = some S →
        ∃ t, runVal values j = some t ∧ Op.Directed.Encloses S t := by
      intro j S hS
      rw [Array.getElem?_push] at hS
      split at hS
      · rename_i hj
        simp only [Option.join_some, Option.some.injEq] at hS
        subst hS
        exact ⟨t, by rw [hj, hsize]; exact ht, hB₁⟩
      · exact htbl j S hS
    have hdim : outSize? G id = some B₁.dim :=
      (outSize?_of_run hrun ht).trans (congrArg some hB₁.1.symm)
    have hN : nodeBounds? G P ctx (tbl.push (some B₁)) id = some b := by
      simp only [nodeBounds?, hdim, Option.bind_eq_bind, Option.bind_some, hb]
      rfl
    obtain ⟨-, hbout, hrows⟩ := nodeBounds?_of_run hrun hP hpush hin hx hxd ht hN
    have hbox := hE _ _ (by rw [← hxd]; exact hxB) hrows
    rw [hbout] at hbox
    exact meetBox_encloses hbox hB₁
  refine ⟨t, ht, ?_⟩
  have hmeet : ∀ S, (ibp[id]?).join = some S → Op.Directed.Encloses (meetBox S B₀) t := by
    intro S hS
    obtain ⟨t', ht', hSt⟩ := hibp id S hS
    rw [ht, Option.some.injEq] at ht'
    exact meetBox_encloses (ht' ▸ hSt) hB₀
  dsimp only at h
  split at h
  · split at h
    · rename_i box hbox
      obtain rfl := Option.some.inj h
      refine key _ ?_ box hbox
      split <;> first | exact hB₀ | exact hmeet _ ‹_›
    · obtain rfl := Option.some.inj h
      split <;> first | exact hB₀ | exact hmeet _ ‹_›
  · obtain rfl := Option.some.inj h
    split <;> first | exact hB₀ | exact hmeet _ ‹_›

/-- In a successful real run on a payload holding the reals of the rounded constants, every node
that `crownBoxes` boxes has a flat value, and its box encloses it. The premises are those of
`crownStep_encloses` without the table `tbl`, which `crownBoxes` builds node by node. -/
theorem crownBoxes_encloses [NonlinearBoundOps α] [LawfulNonlinearBoundOps α]
    (hrun : G.run Pr inputs = .ok values) (hP : Op.Directed.RealConsts Pr P)
    {seeds : Nat → Option (FlatBox α)}
    (hseeds : ∀ id k p B, G.nodes[id]? = some (.input k p) → seeds id = some B →
      ∃ t, runVal values id = some t ∧ Op.Directed.Encloses B t)
    {ibp : Array (Option (FlatBox α))}
    (hibp : ∀ id B, (ibp[id]?).join = some B →
      ∃ t, runVal values id = some t ∧ Op.Directed.Encloses B t)
    {ctx : AffineCtx} (hin : ∃ k p, G.nodes[ctx.inputId]? = some (.input k p)) {xB : FlatBox α}
    {x : FlatTensor ℝ} (hx : runVal values ctx.inputId = some x) (hxB : Op.Directed.Encloses xB x)
    (id : Nat) (B : FlatBox α) (hB : ((crownBoxes G P ctx seeds ibp xB)[id]?).join = some B) :
    ∃ t, runVal values id = some t ∧ Op.Directed.Encloses B t :=
  tabulate_induction (step := crownStep G P ctx seeds ibp xB)
    (fun j b => ∃ t, runVal values j = some t ∧ Op.Directed.Encloses b t)
    (fun k hk _ hb => crownStep_encloses hrun hP hseeds hibp hin hx hxB hk
      (size_tabulate _ k) hb) G.nodes.size id B hB

/-- Every affine bound that `runCROWN` gives node `id` bounds the node's flat value `t` at the flat
value `x` of the designated input, and has the input width of the affine forms and the width of
`t`. The premises are those of `crownBoxes_encloses`, with `x` of the width of the affine forms. -/
theorem runCROWN_encloses [NonlinearBoundOps α] [LawfulNonlinearBoundOps α]
    (hrun : G.run Pr inputs = .ok values) (hP : Op.Directed.RealConsts Pr P)
    {seeds : Nat → Option (FlatBox α)}
    (hseeds : ∀ id k p B, G.nodes[id]? = some (.input k p) → seeds id = some B →
      ∃ t, runVal values id = some t ∧ Op.Directed.Encloses B t)
    {ibp : Array (Option (FlatBox α))}
    (hibp : ∀ id B, (ibp[id]?).join = some B →
      ∃ t, runVal values id = some t ∧ Op.Directed.Encloses B t)
    {ctx : AffineCtx} (hin : ∃ k p, G.nodes[ctx.inputId]? = some (.input k p)) {xB : FlatBox α}
    {x : FlatTensor ℝ} (hx : runVal values ctx.inputId = some x) (hxB : Op.Directed.Encloses xB x)
    (hxd : x.n = ctx.inputDim) {id : Nat} {t : FlatTensor ℝ} (ht : runVal values id = some t)
    {bounds : FlatAffineBounds α} (h : ((runCROWN G P ctx seeds ibp xB)[id]?).join = some bounds) :
    bounds.inDim = ctx.inputDim ∧ bounds.outDim = t.n ∧
      AffineRowsEnclose bounds (fun i => getAtOrZero x.v [i]) fun i => getAtOrZero t.v [i] := by
  have hboxes := crownBoxes_encloses hrun hP hseeds hibp hin hx hxB
  have hlt : id < G.nodes.size := by
    rw [← IR.Graph.run_size hrun]
    by_contra hge
    rw [runVal, Array.getElem?_eq_none (by omega)] at ht
    cases ht
  unfold runCROWN at h
  simp only [Array.getElem?_map, Array.getElem?_range, hlt, ↓reduceIte, Option.map_some,
    Option.join_some] at h
  split at h
  · split at h
    · rename_i hid
      obtain rfl := Option.some.inj h
      rw [hid, hx, Option.some.injEq] at ht
      subst ht
      exact ⟨rfl, hxd.symm, boundsIdentity_encloses _ _⟩
    · obtain ⟨B, hB, rfl⟩ := Option.map_eq_some_iff.mp h
      obtain ⟨t', ht', hBt⟩ := hboxes id B hB
      rw [ht, Option.some.injEq] at ht'
      subst ht'
      exact ⟨rfl, hBt.1, boundsConst_encloses _ B _ _ (by rw [hBt.1]; exact hBt)⟩
  · obtain ⟨v, hv, rfl⟩ := Option.map_eq_some_iff.mp h
    have ht' := (join_runVals values id).trans ht
    obtain ⟨hn, hc⟩ := const_eq_of_run hrun hP hv
    rw [dimsOf_eq ht'] at hn
    refine ⟨rfl, hn.symm, boundsConst_encloses _ ⟨v.n, v.v, v.v⟩ _ _ ⟨rfl, fun i => ?_⟩⟩
    have hi := hc i
    rw [pointOf_eq ht'] at hi
    exact ⟨hi.ge, hi.le⟩
  · split at h
    · rename_i b B hb hB
      obtain rfl := Option.some.inj h
      obtain ⟨hbin, hbout, hrows⟩ := nodeBounds?_of_run hrun hP hboxes hin hx hxd ht hb
      obtain ⟨t', ht', hBt⟩ := hboxes id B hB
      rw [ht, Option.some.injEq] at ht'
      subst ht'
      have hd : ∀ c, (tighterRows xB b c).inDim = b.inDim ∧
          (tighterRows xB b c).outDim = b.outDim := fun c => by
        unfold tighterRows
        split <;> exact ⟨rfl, rfl⟩
      exact ⟨(hd _).1.trans hbin, (hd _).2.trans hbout, tighterRows_encloses xB hrows
        (boundsConst_encloses _ B _ _ (by rw [hBt.1]; exact hBt))⟩
    · exact nodeBounds?_of_run hrun hP hboxes hin hx hxd ht h
  · cases h

end Full

end

end TorchLean.CROWN.Op
