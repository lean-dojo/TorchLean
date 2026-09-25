/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.DirectedIBPBasic

/-!
# Soundness of the rounded forward IBP pass

`runIBP` computes one interval box per node with directed endpoint arithmetic. This file proves
that, for any backend with `LawfulBoundOps` and `LawfulNonlinearBoundOps`, every box it returns
encloses the real value of its node, provided the real point satisfies the node equations of
`NodeEquation` and lies in the seeded input boxes.

The proof covers the node kinds accepted by `ibpForwardSupportedNode`: inputs, constants, copies,
`add`, `sub`, `mulElem`, `relu`, `linear`, unary `matmul`, `sum`, and the pointwise `exp`, `log`,
`sqrt`, `inv`, `tanh`, `sigmoid`, `sin`, and `cos`. On such graphs `GraphPoint.ofRunIBP` builds the
`GraphPoint` that the backward theorems take, so their IBP hypothesis is discharged. The
corollaries `runCROWNBackwardObjective_encloses_runIBP` and `directedNodeBounds_encloses_runIBP`
state the resulting end-to-end guarantees.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean NN.IR
open TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph.Internal
open NN.MLTheory.CROWN.IntervalLemmas (intervalMul_encloses value_min2 value_max2)
open scoped BigOperators

noncomputable section

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)


section Step

variable [NonlinearBoundOps α] [LawfulNonlinearBoundOps α]

/-- Node kinds whose forward IBP transfer is proved sound in this file. Binary `matmul` is not
included because `NodeEquation` gives it no equation. -/
def ibpForwardSupportedNode (node : Node) : Bool :=
  match node.kind with
  | .input | .const _ | .detach | .reshape .. | .flatten ..
  | .add | .sub | .mulElem | .relu | .linear | .sum
  | .exp | .log | .sqrt | .inv | .tanh | .sigmoid | .sin | .cos => true
  | .matmul => node.parents.size == 1
  | _ => false

/-- Every node of the graph has a kind covered by the forward IBP soundness theorem. -/
def ibpForwardSupported (nodes : Array Node) : Bool :=
  nodes.all ibpForwardSupportedNode

/-- The real point lies in every seeded input box. -/
def InputsInBoxes (nodes : Array Node) (ps : ParamStore α) (dims : Nat → Nat)
    (v : Nat → Nat → ℝ) : Prop :=
  ∀ id, id < nodes.size → nodes[id]!.kind = .input →
    ∀ box, ps.inputBoxes[id]? = some box → RowEncloses box (dims id) (v id)

/--
One IBP transfer is sound: if the parent boxes enclose the parent values, the box computed for a
supported node encloses its value.

`boxes` is the array the transfer reads and `ibp` the array named by the node equations; they
agree at every parent.
-/
theorem ibpStepNodeAt?_encloses
    {nodes : Array Node} {ps : ParamStore α} {boxes ibp : Array (Option (FlatBox α))}
    {dims : Nat → Nat} {v : Nat → Nat → ℝ} {id : Nat}
    (hsupported : ibpForwardSupportedNode nodes[id]! = true)
    (hinput : nodes[id]!.kind = .input →
      ∀ box, ps.inputBoxes[id]? = some box → RowEncloses box (dims id) (v id))
    (heq : NodeEquation nodes ps ibp dims v id)
    (hagree : ∀ p ∈ nodes[id]!.parents, (boxes[p]?).join = ibp[p]!)
    (henc : ∀ p ∈ nodes[id]!.parents, ∀ box, ibp[p]! = some box →
      RowEncloses box (dims p) (v p))
    {box : FlatBox α} (hstep : ibpStepNodeAt? nodes ps boxes id nodes[id]! = some box) :
    RowEncloses box (dims id) (v id) := by
  have hget : ∀ p ∈ nodes[id]!.parents, ∀ B, (boxes[p]?).join = some B →
      RowEncloses B (dims p) (v p) := fun p hp B hB =>
    henc p hp B ((hagree p hp).symm.trans hB)
  have hunary {p : Nat} (hp : unaryParent? nodes[id]!.parents = some p) :=
    hget p (mem_of_unaryParent?_eq_some hp)
  have hleft {p q : Nat} (hpq : binaryParents? nodes[id]!.parents = some (p, q)) :=
    hget p (fst_mem_of_binaryParents?_eq_some hpq)
  have hright {p q : Nat} (hpq : binaryParents? nodes[id]!.parents = some (p, q)) :=
    hget q (snd_mem_of_binaryParents?_eq_some hpq)
  unfold NodeEquation at heq
  rw [ibpStepNodeAt?] at hstep
  cases hk : (nodes[id]!).kind <;>
    simp only [ibpForwardSupportedNode, hk, Bool.false_eq_true, beq_iff_eq] at hsupported <;>
    simp only [hk] at heq hinput <;>
    rw [hk] at hstep <;> dsimp only at hstep
  case input => exact hinput trivial box hstep
  case const shape =>
    cases hc : ps.constVals[id]? with
    | none => simp [hc] at hstep
    | some stored =>
        simp only [hc, Option.bind_eq_bind, Option.bind_some, Option.some.injEq] at hstep
        subst hstep
        obtain ⟨hd, hv⟩ := heq stored hc
        rw [hd, RowEncloses]
        exact ⟨rfl, fun i => by simp only [read_fin, hv i, le_refl, and_self]⟩
  case detach | reshape | flatten =>
    cases hp : unaryParent? nodes[id]!.parents with
    | none => simp [hp] at hstep
    | some p =>
        simp only [hp, Option.bind_eq_bind, Option.bind_some] at hstep
        obtain ⟨hd, hv⟩ := heq p hp
        have hB := hunary hp box hstep
        rw [hd] at hB
        exact hB.congr hv
  case add =>
    obtain ⟨p, q, X, Y, hpq, hX, hY, hbox⟩ := binary_lookup
      (k := fun x y => if x.dim = y.dim then some (boxAdd x y) else none) hstep
    split at hbox
    · obtain rfl := Option.some.inj hbox
      obtain ⟨hdp, hdq, hv⟩ := heq p q hpq
      have hX' := hleft hpq X hX
      have hY' := hright hpq Y hY
      rw [hdp] at hX'
      rw [hdq] at hY'
      exact (boxAdd_encloses hX' hY').congr hv
    · simp at hbox
  case sub =>
    obtain ⟨p, q, X, Y, hpq, hX, hY, hbox⟩ := binary_lookup
      (k := fun x y => if x.dim = y.dim then some (boxSub x y) else none) hstep
    split at hbox
    · obtain rfl := Option.some.inj hbox
      obtain ⟨hdp, hdq, hv⟩ := heq p q hpq
      have hX' := hleft hpq X hX
      have hY' := hright hpq Y hY
      rw [hdp] at hX'
      rw [hdq] at hY'
      exact (boxSub_encloses hX' hY').congr hv
    · simp at hbox
  case mulElem =>
    obtain ⟨p, q, X, Y, hpq, hX, hY, hbox⟩ := binary_lookup (k := boxMulElem) hstep
    obtain ⟨hdp, hdq, hv⟩ := heq p q hpq
    have hX' := hleft hpq X hX
    have hY' := hright hpq Y hY
    rw [hdp] at hX'
    rw [hdq] at hY'
    exact (boxMulElem_encloses hX' hY' hbox).congr hv
  case relu =>
    obtain ⟨p, B, hp, hB, hbox⟩ := unary_lookup (k := fun B => some (boxRelu B)) hstep
    exact unary_step (transfer := fun B => some (boxRelu B))
      (fun hB hb => by obtain rfl := Option.some.inj hb; exact boxRelu_encloses hB)
      (heq p hp) (hunary hp B hB) hbox
  case linear =>
    obtain ⟨p, B, hp, hB, hbox⟩ := unary_lookup (k := ibpLinear id ps) hstep
    cases hc : ps.linearWB[id]? with
    | none => simp [ibpLinear, hc] at hbox
    | some config =>
        simp only [ibpLinear, hc] at hbox
        obtain ⟨hm, hn, hv⟩ := heq p hp config hc
        have hB' := hunary hp B hB
        rw [hn] at hB'
        rw [hm]
        exact ibpLinearParams_encloses hB' hv hbox
  case matmul =>
    obtain ⟨p, hpar⟩ := Array.size_eq_one_iff.mp hsupported
    have hp : unaryParent? nodes[id]!.parents = some p := by
      simp [unaryParent?, hpar]
    rw [hpar] at hstep
    change ((boxes[p]?).join >>= ibpMatmul id ps) = some box at hstep
    cases hB : (boxes[p]?).join with
    | none => simp [hB] at hstep
    | some B =>
        simp only [hB, Option.bind_eq_bind, Option.bind_some] at hstep
        cases hc : ps.matmulW[id]? with
        | none => simp [ibpMatmul, hc] at hstep
        | some config =>
            obtain ⟨hm, hn, hv⟩ := heq p hp config hc
            have hB' := hunary hp B hB
            rw [hn] at hB'
            rw [hm]
            exact ibpMatmul_encloses hc hB' hv hstep
  case sum =>
    obtain ⟨p, B, hp, hB, hbox⟩ := unary_lookup (k := fun B => some (boxSum B)) hstep
    obtain rfl := Option.some.inj hbox
    obtain ⟨h1, hdp, hv⟩ := heq p hp B
      ((hagree p (mem_of_unaryParent?_eq_some hp)).symm.trans hB)
    have hB' := hunary hp B hB
    rw [hdp] at hB' hv
    rw [h1]
    exact boxSum_encloses hB' hv
  case exp =>
    obtain ⟨p, B, hp, hB, hbox⟩ :=
      unary_lookup (k := boxUnaryEnclosure? NonlinearBoundOps.expBounds) hstep
    exact unary_step (fun hB hb => boxUnaryEnclosure?_encloses
      LawfulNonlinearBoundOps.expBounds_enclosure hB hb) (heq p hp) (hunary hp B hB) hbox
  case tanh =>
    obtain ⟨p, B, hp, hB, hbox⟩ :=
      unary_lookup (k := boxUnaryEnclosure? NonlinearBoundOps.tanhBounds) hstep
    exact unary_step (fun hB hb => boxUnaryEnclosure?_encloses
      LawfulNonlinearBoundOps.tanhBounds_enclosure hB hb) (heq p hp) (hunary hp B hB) hbox
  case sigmoid =>
    obtain ⟨p, B, hp, hB, hbox⟩ :=
      unary_lookup (k := boxUnaryEnclosure? NonlinearBoundOps.sigmoidBounds) hstep
    exact unary_step (fun hB hb => boxUnaryEnclosure?_encloses
      LawfulNonlinearBoundOps.sigmoidBounds_enclosure hB hb) (heq p hp) (hunary hp B hB) hbox
  case sin =>
    obtain ⟨p, B, hp, hB, hbox⟩ :=
      unary_lookup (k := boxUnaryEnclosure? NonlinearBoundOps.sinBounds) hstep
    exact unary_step (fun hB hb => boxUnaryEnclosure?_encloses
      LawfulNonlinearBoundOps.sinBounds_enclosure hB hb) (heq p hp) (hunary hp B hB) hbox
  case cos =>
    obtain ⟨p, B, hp, hB, hbox⟩ :=
      unary_lookup (k := boxUnaryEnclosure? NonlinearBoundOps.cosBounds) hstep
    exact unary_step (fun hB hb => boxUnaryEnclosure?_encloses
      LawfulNonlinearBoundOps.cosBounds_enclosure hB hb) (heq p hp) (hunary hp B hB) hbox
  case sqrt =>
    obtain ⟨p, B, hp, hB, hbox⟩ := unary_lookup (k := boxSqrt?) hstep
    exact unary_step (transfer := boxSqrt?) (fun hB hb => boxUnaryEnclosure?_encloses
      LawfulNonlinearBoundOps.sqrtBounds_enclosure hB hb) (heq p hp) (hunary hp B hB) hbox
  case inv =>
    obtain ⟨p, B, hp, hB, hbox⟩ := unary_lookup (k := boxInv?) hstep
    exact unary_step (transfer := boxInv?) (fun hB hb => boxUnaryEnclosure?_encloses
      unaryEnclosure_inv hB hb) (heq p hp) (hunary hp B hB) hbox
  case log =>
    obtain ⟨p, B, hp, hB, hbox⟩ := unary_lookup (k := fun input =>
      if (List.finRange input.dim).all
          (fun i => decide (0 < Tensor.item (getDimScalarFn (α := α) input.lo i))) then
        boxUnaryEnclosure? NonlinearBoundOps.logBounds input
      else none) hstep
    split at hbox
    · exact unary_step (fun hB hb => boxUnaryEnclosure?_encloses
        LawfulNonlinearBoundOps.logBounds_enclosure hB hb) (heq p hp) (hunary hp B hB) hbox
    · simp at hbox

/-! ## The whole pass -/

omit [LawfulBoundOps α] [LawfulNonlinearBoundOps α] in
private theorem getElem!_eq_join {β : Type} (a : Array (Option β)) (i : Nat) :
    a[i]! = (a[i]?).join := by
  by_cases h : i < a.size
  · simp [h]
  · simp [h]
    rfl

/-- The array after the first `k` steps of the `runIBP` fold. -/
private def ibpFoldPrefix (nodes : Array Node) (ps : ParamStore α) :
    Nat → Array (Option (FlatBox α))
  | 0 => Array.replicate nodes.size none
  | k + 1 => propagateIBPNode nodes ps (ibpFoldPrefix nodes ps k) k

omit [LawfulBoundOps α] [LawfulNonlinearBoundOps α] in
private theorem ibpFoldPrefix_size (nodes : Array Node) (ps : ParamStore α) (k : Nat) :
    (ibpFoldPrefix nodes ps k).size = nodes.size := by
  induction k with
  | zero => simp [ibpFoldPrefix]
  | succ k ih =>
      simp only [ibpFoldPrefix, propagateIBPNode, propagateIBPNodeAt]
      split <;> simp [ih]

omit [LawfulBoundOps α] [LawfulNonlinearBoundOps α] in
private theorem ibpFoldPrefix_succ_ne (nodes : Array Node) (ps : ParamStore α) {k i : Nat}
    (hi : i ≠ k) : (ibpFoldPrefix nodes ps (k + 1))[i]? = (ibpFoldPrefix nodes ps k)[i]? := by
  simp only [ibpFoldPrefix, propagateIBPNode, propagateIBPNodeAt]
  split <;> simp [Ne.symm hi]

omit [LawfulBoundOps α] [LawfulNonlinearBoundOps α] in
private theorem ibpFoldPrefix_succ_self (nodes : Array Node) (ps : ParamStore α) {k : Nat}
    (hk : k < nodes.size) :
    (ibpFoldPrefix nodes ps (k + 1))[k]? =
      some (ibpStepNodeAt? nodes ps (ibpFoldPrefix nodes ps k) k nodes[k]!) := by
  have hsize := ibpFoldPrefix_size nodes ps k
  simp only [ibpFoldPrefix, propagateIBPNode, propagateIBPNodeAt, Array.getElem?_eq_getElem hk,
    getElem!_pos nodes k hk]
  simp [hsize, hk]

omit [LawfulBoundOps α] [LawfulNonlinearBoundOps α] in
private theorem ibpFoldPrefix_stable (nodes : Array Node) (ps : ParamStore α) {k i : Nat}
    (hi : i < k) : ∀ m, (ibpFoldPrefix nodes ps (k + m))[i]? = (ibpFoldPrefix nodes ps k)[i]?
  | 0 => rfl
  | m + 1 => by
      rw [← Nat.add_assoc, ibpFoldPrefix_succ_ne nodes ps (by omega),
        ibpFoldPrefix_stable nodes ps hi m]

omit [LawfulBoundOps α] [LawfulNonlinearBoundOps α] in
private theorem runIBP_eq_ibpFoldPrefix (g : Graph) (ps : ParamStore α)
    (hguard : crownGraphSemanticsSupported (α := α) g ps = true) :
    runIBP g ps = ibpFoldPrefix g.nodes ps g.nodes.size := by
  simp only [runIBP, hguard, ite_true]
  have h : (do
      let a ← List.finRange g.nodes.size
      pure (a : Nat)) = List.range g.nodes.size := by
    simp only [bind, pure, ← List.map_eq_flatMap]
    exact List.map_coe_finRange_eq_range
  rw [h]
  have hfold : ∀ k, (List.range k).foldl (fun acc i => propagateIBPNode g.nodes ps acc i)
      (Array.replicate g.nodes.size none) = ibpFoldPrefix g.nodes ps k := by
    intro k
    induction k with
    | zero => rfl
    | succ k ih =>
        rw [List.range_succ, List.foldl_append, ih]
        rfl
  exact hfold g.nodes.size

omit [LawfulBoundOps α] [LawfulNonlinearBoundOps α] in
/-- Entry `id` of the final pass is the step computed from the prefix before `id`. -/
private theorem ibpFoldPrefix_final_self (nodes : Array Node) (ps : ParamStore α) {id : Nat}
    (hid : id < nodes.size) :
    (ibpFoldPrefix nodes ps nodes.size)[id]! =
      ibpStepNodeAt? nodes ps (ibpFoldPrefix nodes ps id) id nodes[id]! := by
  have h := ibpFoldPrefix_stable nodes ps (Nat.lt_succ_self id) (nodes.size - (id + 1))
  rw [show id + 1 + (nodes.size - (id + 1)) = nodes.size by omega,
    ibpFoldPrefix_succ_self nodes ps hid] at h
  rw [getElem!_eq_join, h, Option.join_some]

omit [LawfulBoundOps α] [LawfulNonlinearBoundOps α] in
/-- Entries before `k` are final once the fold has passed them. -/
private theorem ibpFoldPrefix_final_lt (nodes : Array Node) (ps : ParamStore α) {k p : Nat}
    (hk : k ≤ nodes.size) (hp : p < k) :
    ((ibpFoldPrefix nodes ps k)[p]?).join = (ibpFoldPrefix nodes ps nodes.size)[p]! := by
  have h := ibpFoldPrefix_stable nodes ps hp (nodes.size - k)
  rw [show k + (nodes.size - k) = nodes.size by omega] at h
  rw [getElem!_eq_join, h]

omit [LawfulNonlinearBoundOps α] in
/-- A sound node transfer lifts to the complete topologically ordered forward pass.

The array read by a transfer is only a prefix. Its parent entries agree with the final pass
because later iterations do not overwrite earlier nodes. -/
theorem runIBP_encloses_of_step (g : Graph) (ps : ParamStore α)
    {dims : Nat → Nat} {v : Nat → Nat → ℝ}
    (hparent : ∀ id, id < g.nodes.size → ∀ p ∈ g.nodes[id]!.parents, p < id)
    (htransfer : ∀ id, id < g.nodes.size → ∀ boxes : Array (Option (FlatBox α)),
      (∀ p ∈ g.nodes[id]!.parents, (boxes[p]?).join = (runIBP g ps)[p]!) →
      (∀ p ∈ g.nodes[id]!.parents, ∀ box, (runIBP g ps)[p]! = some box →
        RowEncloses box (dims p) (v p)) →
      ∀ box, ibpStepNodeAt? g.nodes ps boxes id g.nodes[id]! = some box →
        RowEncloses box (dims id) (v id)) :
    ∀ id, id < g.nodes.size → ∀ box, (runIBP g ps)[id]! = some box →
      RowEncloses box (dims id) (v id) := by
  cases hguard : crownGraphSemanticsSupported (α := α) g ps with
  | false =>
      intro id hid box hbox
      simp [runIBP, hguard, hid] at hbox
  | true =>
      have hrun := runIBP_eq_ibpFoldPrefix g ps hguard
      rw [hrun] at htransfer ⊢
      intro id
      induction id using Nat.strong_induction_on with
      | _ id ih =>
          intro hid box hbox
          rw [ibpFoldPrefix_final_self g.nodes ps hid] at hbox
          exact htransfer id hid _
            (fun p hp => ibpFoldPrefix_final_lt g.nodes ps (Nat.le_of_lt hid)
              (hparent id hid p hp))
            (fun p hp => ih p (hparent id hid p hp)
              (lt_trans (hparent id hid p hp) hid)) box hbox

/--
Soundness of the rounded forward IBP pass. Every box that `runIBP` returns encloses the value of
its node, when the parents precede their consumers, every node kind is covered by
`ibpForwardSupportedNode`, the input boxes contain the real inputs, and the real point satisfies
the node equations.

If the semantic guard of `runIBP` rejects the graph, no box is produced and the statement holds
trivially.
-/
theorem runIBP_encloses (g : Graph) (ps : ParamStore α) {dims : Nat → Nat} {v : Nat → Nat → ℝ}
    (hparent : ∀ id, id < g.nodes.size → ∀ p ∈ g.nodes[id]!.parents, p < id)
    (hsupported : ibpForwardSupported g.nodes = true)
    (hinputs : InputsInBoxes g.nodes ps dims v)
    (hequation : ∀ id, id < g.nodes.size → NodeEquation g.nodes ps (runIBP g ps) dims v id) :
    ∀ id, id < g.nodes.size → ∀ box, (runIBP g ps)[id]! = some box →
      RowEncloses box (dims id) (v id) := by
  apply runIBP_encloses_of_step g ps hparent
  intro id hid boxes hagree henc box hstep
  have hnode : ibpForwardSupportedNode g.nodes[id]! = true := by
    rw [getElem!_pos g.nodes id hid]
    exact Array.all_eq_true.mp hsupported id hid
  exact ibpStepNodeAt?_encloses hnode (hinputs id hid) (hequation id hid)
    hagree henc hstep


/-- The `GraphPoint` of a rounded `runIBP` pass. Its `ibp_encloses` field is `runIBP_encloses`. -/
theorem GraphPoint.ofRunIBP {g : Graph} {ps : ParamStore α} {ctx : AffineCtx}
    {dims : Nat → Nat} {v : Nat → Nat → ℝ}
    (input_lt : ctx.inputId < g.nodes.size)
    (input_dim : dims ctx.inputId = ctx.inputDim)
    (input_kind : g.nodes[ctx.inputId]!.kind = .input)
    (node_id : ∀ id, id < g.nodes.size → g.nodes[id]!.id = id)
    (parent_lt : ∀ id, id < g.nodes.size → ∀ p ∈ g.nodes[id]!.parents, p < id)
    (hsupported : ibpForwardSupported g.nodes = true)
    (hinputs : InputsInBoxes g.nodes ps dims v)
    (equation : ∀ id, id < g.nodes.size → NodeEquation g.nodes ps (runIBP g ps) dims v id) :
    GraphPoint g.nodes ps (runIBP g ps) ctx dims v where
  input_lt := input_lt
  input_dim := input_dim
  input_kind := input_kind
  node_id := node_id
  parent_lt := parent_lt
  ibp_encloses := runIBP_encloses g ps parent_lt hsupported hinputs equation
  equation := equation

/-- The public rounded objective, run on the boxes of `runIBP`, encloses the real objective.
No IBP soundness is assumed: it comes from `runIBP_encloses`. -/
theorem runCROWNBackwardObjective_encloses_runIBP
    (hrounded : BoundOps.supportsExactAffineReassociation (α := α) = false)
    {g : Graph} {ps : ParamStore α} {ctx : AffineCtx} {dims : Nat → Nat} {v : Nat → Nat → ℝ}
    (input_lt : ctx.inputId < g.nodes.size)
    (input_dim : dims ctx.inputId = ctx.inputDim)
    (input_kind : g.nodes[ctx.inputId]!.kind = .input)
    (node_id : ∀ id, id < g.nodes.size → g.nodes[id]!.id = id)
    (parent_lt : ∀ id, id < g.nodes.size → ∀ p ∈ g.nodes[id]!.parents, p < id)
    (hsupported : ibpForwardSupported g.nodes = true)
    (hinputs : InputsInBoxes g.nodes ps dims v)
    (equation : ∀ id, id < g.nodes.size → NodeEquation g.nodes ps (runIBP g ps) dims v id)
    (output : Nat) (houtput : output < g.nodes.size) (obj : FlatTensor α)
    (hdim : obj.n = dims output) {bounds : FlatAffineBounds α}
    (hresult : runCROWNBackwardObjective g ps ctx (runIBP g ps) output obj = some bounds) :
    bounds.inDim = ctx.inputDim ∧ bounds.outDim = 1 ∧
      AffineRowsEnclose bounds (v ctx.inputId)
        (fun _ => dot (dims output) (fun i => value (getAtOrZero obj.v [i])) (v output)) :=
  runCROWNBackwardObjective_encloses hrounded
    (GraphPoint.ofRunIBP input_lt input_dim input_kind node_id parent_lt hsupported hinputs
      equation) output houtput obj hdim hresult

/-- The nodewise rounded CROWN bounds, run on the boxes of `runIBP`, enclose every coordinate of
the output node. No IBP soundness is assumed. -/
theorem directedNodeBounds_encloses_runIBP
    {g : Graph} {ps : ParamStore α} {ctx : AffineCtx} {dims : Nat → Nat} {v : Nat → Nat → ℝ}
    (input_lt : ctx.inputId < g.nodes.size)
    (input_dim : dims ctx.inputId = ctx.inputDim)
    (input_kind : g.nodes[ctx.inputId]!.kind = .input)
    (node_id : ∀ id, id < g.nodes.size → g.nodes[id]!.id = id)
    (parent_lt : ∀ id, id < g.nodes.size → ∀ p ∈ g.nodes[id]!.parents, p < id)
    (hsupported : ibpForwardSupported g.nodes = true)
    (hinputs : InputsInBoxes g.nodes ps dims v)
    (equation : ∀ id, id < g.nodes.size → NodeEquation g.nodes ps (runIBP g ps) dims v id)
    (output : Nat) (houtput : output < g.nodes.size)
    (hdim : dims output = g.nodes[output]!.outShape.size)
    {bounds : FlatAffineBounds α}
    (hresult : directedNodeBounds? g ps ctx (runIBP g ps) output = some bounds) :
    bounds.inDim = ctx.inputDim ∧ bounds.outDim = dims output ∧
      AffineRowsEnclose bounds (v ctx.inputId) (v output) :=
  directedNodeBounds_encloses
    (GraphPoint.ofRunIBP input_lt input_dim input_kind node_id parent_lt hsupported hinputs
      equation) output houtput hdim hresult

end Step

end

end NN.MLTheory.CROWN.Graph.DirectedBackward
