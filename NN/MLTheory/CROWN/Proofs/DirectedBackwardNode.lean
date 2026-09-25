/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.DirectedBackwardConcat

/-!
# Soundness of the directed node dispatcher

Every successful branch restores the objective removed at the frontier. Missing data and
unsupported active transfers retain the engine's failure behavior.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean NN.IR
open TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph.Internal
open scoped BigOperators

noncomputable section

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

private theorem linear_restores (hzero : value (0 : α) = 0)
    {dims : Nat → Nat} {v : Nat → Nat → ℝ} {input k id p : Nat}
    {st : DirectedBackwardState α} {z : ℝ} {aB : FlatBox α} {a : Nat → ℝ}
    (h : Represents dims v input k st (z - dot (dims id) a (v id)))
    (hp : p < st.coeffs.size) (hlive : p ∈ pending input k)
    (ha : RowEncloses aB (dims id) a)
    {m n : Nat} (W : Tensor α [m, n]) (b : Tensor α [m])
    (heq : LinearEquation dims v id p W b) :
    let result :=
      match directedBackwardLinear aB W b with
      | some (aX, c) => addDirectedConstant (addDirectedCoeff st p aX) c.1 c.2
      | none => st.fail
    result.failed = true ∨ Represents dims v input k result z := by
  cases hr : directedBackwardLinear aB W b with
  | none => exact Or.inl rfl
  | some result =>
      rcases result with ⟨aX, lo, hi⟩
      right
      have ha' : RowEncloses aB m a := by simpa only [heq.1] using ha
      have hresult := represents_linear hzero h p hp hlive heq.2.1 ha' W b heq.2.2 hr
      simpa only [← heq.1, sub_add_cancel] using hresult.1

private theorem sum_restores
    {dims : Nat → Nat} {v : Nat → Nat → ℝ} {input k id p : Nat}
    {st : DirectedBackwardState α} {z : ℝ} {aB : FlatBox α} {a : Nat → ℝ}
    (h : Represents dims v input k st (z - dot (dims id) a (v id)))
    (hp : p < st.coeffs.size) (hlive : p ∈ pending input k)
    (ha : RowEncloses aB (dims id) a) (n : Nat)
    (hdy : dims id = 1) (hdp : dims p = n)
    (hy : v id 0 = ∑ i : Fin (dims p), v p i.val) :
    let result :=
      if hdim : aB.dim = 1 then
        let lo := castDimScalar hdim aB.lo
        let hi := castDimScalar hdim aB.hi
        addDirectedCoeff st p
          { dim := n
            lo := Tensor.full (α := α) (.dim n .scalar) (getAtOrZero lo [0])
            hi := Tensor.full (α := α) (.dim n .scalar) (getAtOrZero hi [0]) }
      else st.fail
    result.failed = true ∨ Represents dims v input k result z := by
  obtain ⟨adim, alo, ahi⟩ := aB
  have hadim := ha.1.trans hdy
  change adim = 1 at hadim
  subst adim
  simp only [↓reduceDIte, castDimScalar_self]
  right
  have ha' : RowEncloses { dim := 1, lo := alo, hi := ahi } 1 a := by
    simpa only [hdy] using ha
  have hb := broadcast_encloses (n := n) alo ahi (a 0)
    (by
      have hscalar := ha'.2 (0 : Fin 1)
      change value (getAtOrZero alo [0]) ≤ a 0 ∧
        a 0 ≤ value (getAtOrZero ahi [0]) at hscalar
      rw [show getAtOrZero alo [0] = alo.getScalar 0 from read_fin alo 0,
        show getAtOrZero ahi [0] = ahi.getScalar 0 from read_fin ahi 0] at hscalar
      exact hscalar)
  have hb' : RowEncloses
      { dim := n
        lo := Tensor.full (α := α) (.dim n .scalar) (getAtOrZero alo [0])
        hi := Tensor.full (α := α) (.dim n .scalar) (getAtOrZero ahi [0]) }
      (dims p) (fun _ => a 0) := by simpa only [hdp] using hb
  have hd : dot (dims id) a (v id) = dot (dims p) (fun _ => a 0) (v p) := by
    rw [hdy]
    simp only [dot, Fin.sum_univ_one, Fin.val_zero, hy, Finset.mul_sum]
  have hr := represents_add h p hp hlive _ (fun _ => a 0) hb'
  simpa only [← hd, sub_add_cancel] using hr

/-- Every executable directed node preserves the sweep invariant for a real graph point. -/
theorem backwardNode_preserves (hzero : value (0 : α) = 0)
    {nodes : Array Node} {ps : ParamStore α} {ibp : Array (Option (FlatBox α))}
    {ctx : AffineCtx} {dims : Nat → Nat} {v : Nat → Nat → ℝ}
    (point : GraphPoint nodes ps ibp ctx dims v)
    (k : Nat) (hk : k < nodes.size) (st : DirectedBackwardState α) (z : ℝ)
    (hinv : SweepInvariant dims v ctx.inputId nodes.size (k + 1) z st) :
    SweepInvariant dims v ctx.inputId nodes.size k z
      (directedBackwardNode nodes ps ibp ctx st k) := by
  rcases hinv with hfailed | ⟨hsize, hrep⟩
  · exact Or.inl (backwardNode_failed nodes ps ibp ctx st k hfailed)
  by_cases hfailed : st.failed = true
  · exact Or.inl (backwardNode_failed nodes ps ibp ctx st k hfailed)
  have hksize : k < st.coeffs.size := by simpa only [hsize] using hk
  suffices hpost :
      (directedBackwardNode nodes ps ibp ctx st k).failed = true ∨
        Represents dims v ctx.inputId k (directedBackwardNode nodes ps ibp ctx st k) z by
    rcases hpost with hf | hr
    · exact Or.inl hf
    · exact Or.inr ⟨(backwardNode_size nodes ps ibp ctx st k).trans hsize, hr⟩
  by_cases hinput : k = ctx.inputId
  · subst k
    have hnode : directedBackwardNode nodes ps ibp ctx st ctx.inputId = st := by
      cases he : st.coeffs[ctx.inputId]! <;>
        simp [directedBackwardNode, hfailed, he, point.input_kind,
          point.node_id _ point.input_lt]
    rw [hnode]
    right
    obtain ⟨f, c, hs, hz⟩ := hrep
    exact ⟨f, c, hs, by simpa only [frontier_input] using hz⟩
  cases hentry : st.coeffs[k]! with
  | none =>
      simp only [directedBackwardNode, hfailed, hentry, Bool.false_eq_true, ↓reduceIte]
      exact Or.inr (represents_drop_none hrep hinput hksize hentry)
  | some aB =>
      obtain ⟨a, ha, hbase⟩ := represents_drop_some hrep hinput hksize hentry
      have hparent (p : Nat) (hp : p ∈ nodes[k]!.parents) :
          p < st.coeffs.size ∧ p ∈ pending ctx.inputId k := by
        have hpk := point.parent_lt k hk p hp
        exact ⟨by omega, by simp [pending, hpk]⟩
      have hconsume :
          let result := match ibp[k]! with
            | some box => consumeDirectedObjective st aB box
            | none => st.fail
          result.failed = true ∨ Represents dims v ctx.inputId k result z := by
        cases hb : ibp[k]! with
        | none => exact Or.inl rfl
        | some box =>
            simpa only [sub_add_cancel] using
              represents_consume hzero hbase ha (point.ibp_encloses k hk box hb)
      have hfail : st.fail.failed = true ∨ Represents dims v ctx.inputId k st.fail z :=
        Or.inl rfl
      have heq := point.equation k hk
      simp only [directedBackwardNode, hfailed, hentry, Bool.false_eq_true, ↓reduceIte]
      cases hkind : nodes[k]!.kind <;> dsimp only
      all_goals try exact hconsume
      case input =>
        simp only [point.node_id k hk, hinput, ↓reduceIte]
        exact hconsume
      case const shape =>
        simp only [NodeEquation, hkind] at heq
        cases hs : ps.constVals[k]? with
        | none => exact hfail
        | some stored =>
            have hsEq := heq stored hs
            have hb : RowEncloses (pointCoeffBox stored) (dims k) (v k) := by
              refine ⟨hsEq.1.symm, ?_⟩
              rw [hsEq.1]
              intro i
              simp only [pointCoeffBox, read_fin, hsEq.2 i, le_refl, and_self]
            simpa only [sub_add_cancel] using represents_consume hzero hbase ha hb
      case detach | reshape _ _ | flatten _ =>
        simp only [NodeEquation, hkind] at heq
        cases hp : unaryParent? nodes[k]!.parents with
        | none => exact hfail
        | some p =>
            have hp' := hparent p (mem_of_unaryParent?_eq_some hp)
            right
            simpa only [sub_add_cancel] using
              represents_copy hbase hp'.1 hp'.2 ha (heq p hp)
      case add =>
        simp only [NodeEquation, hkind] at heq
        cases hpq : binaryParents? nodes[k]!.parents with
        | none => exact hfail
        | some pq =>
            rcases pq with ⟨p, q⟩
            have hp := hparent p (fst_mem_of_binaryParents?_eq_some hpq)
            have hq := hparent q (snd_mem_of_binaryParents?_eq_some hpq)
            obtain ⟨hdp, hdq, hy⟩ := heq p q hpq
            right
            simpa only [sub_add_cancel] using
              represents_addParents hbase hp.1 hq.1 hp.2 hq.2 ha hdp hdq hy
      case sub =>
        simp only [NodeEquation, hkind] at heq
        cases hpq : binaryParents? nodes[k]!.parents with
        | none => exact hfail
        | some pq =>
            rcases pq with ⟨p, q⟩
            have hp := hparent p (fst_mem_of_binaryParents?_eq_some hpq)
            have hq := hparent q (snd_mem_of_binaryParents?_eq_some hpq)
            obtain ⟨hdp, hdq, hy⟩ := heq p q hpq
            right
            simpa only [sub_add_cancel] using
              represents_subParents hzero hbase hp.1 hq.1 hp.2 hq.2 ha hdp hdq hy
      case linear =>
        simp only [NodeEquation, hkind] at heq
        cases hp : unaryParent? nodes[k]!.parents with
        | none => exact hfail
        | some p =>
            cases hc : ps.linearWB[k]? with
            | none => exact hfail
            | some config =>
                have hp' := hparent p (mem_of_unaryParent?_eq_some hp)
                exact linear_restores hzero hbase hp'.1 hp'.2 ha
                  config.w config.b (heq p hp config hc)
      case matmul =>
        simp only [NodeEquation, hkind] at heq
        cases hp : unaryParent? nodes[k]!.parents with
        | none => exact hconsume
        | some p =>
            cases hc : ps.matmulW[k]? with
            | none => exact hfail
            | some config =>
                have hp' := hparent p (mem_of_unaryParent?_eq_some hp)
                exact linear_restores hzero hbase hp'.1 hp'.2 ha config.w
                  (Tensor.full (α := α) (.dim config.m .scalar) 0) (heq p hp config hc)
      case conv configuration =>
        simp only [NodeEquation, hkind] at heq
        cases hp : unaryParent? nodes[k]!.parents with
        | none => exact hfail
        | some p =>
            dsimp only
            cases hc : ps.convCfg[k]? with
            | none => exact hfail
            | some config =>
                cases hn : nodes[p]? with
                | none => exact hfail
                | some parent =>
                    dsimp only
                    cases hl : planConvTransfer? configuration config
                        parent.outShape nodes[k]!.outShape with
                    | none => exact hfail
                    | some leading =>
                        have hp' := hparent p (mem_of_unaryParent?_eq_some hp)
                        exact linear_restores hzero hbase hp'.1 hp'.2 ha
                          (affOfConv (α := α) config leading).A
                          (affOfConv (α := α) config leading).c
                          (heq p hp config hc parent hn leading hl)
      case sum =>
        simp only [NodeEquation, hkind] at heq
        cases hp : unaryParent? nodes[k]!.parents with
        | none => exact hfail
        | some p =>
            dsimp only
            cases hb : ibp[p]! with
            | none => exact hfail
            | some box =>
                have hp' := hparent p (mem_of_unaryParent?_eq_some hp)
                obtain ⟨hdy, hdp, hy⟩ := heq p hp box hb
                exact sum_restores hbase hp'.1 hp'.2 ha box.dim hdy hdp hy
      case permute perm =>
        simp only [NodeEquation, hkind] at heq
        cases hp : unaryParent? nodes[k]!.parents with
        | none => exact hfail
        | some p =>
            cases hr : directedAxisPermutation? nodes[k]!.outShape perm aB with
            | none => exact hfail
            | some aX =>
                have hp' := hparent p (mem_of_unaryParent?_eq_some hp)
                obtain ⟨a', ha', hd⟩ :=
                  axisPermutation_encloses ha nodes[k]!.outShape perm (heq p hp) hr
                right
                simpa only [← hd, sub_add_cancel] using
                  represents_add hbase p hp'.1 hp'.2 aX a' ha'
      case transpose axis₁ axis₂ =>
        simp only [NodeEquation, hkind] at heq
        cases hp : unaryParent? nodes[k]!.parents with
        | none => exact hfail
        | some p =>
            cases ht : (OpContracts.transposePerm nodes[p]!.outShape.rank axis₁ axis₂).toOption with
            | none =>
                simpa only [ht, Option.bind_eq_bind, Option.bind_none] using hfail
            | some perm =>
                simp only [ht, Option.bind_eq_bind, Option.bind_some]
                cases hr : directedAxisPermutation? nodes[k]!.outShape perm aB with
                | none => exact hfail
                | some aX =>
                    have hp' := hparent p (mem_of_unaryParent?_eq_some hp)
                    obtain ⟨a', ha', hd⟩ :=
                      axisPermutation_encloses ha nodes[k]!.outShape perm (heq p hp perm ht) hr
                    right
                    simpa only [← hd, sub_add_cancel] using
                      represents_add hbase p hp'.1 hp'.2 aX a' ha'
      case concat axis =>
        simp only [NodeEquation, hkind] at heq
        cases hl : concatBackwardLayout? nodes ibp nodes[k]! axis with
        | none =>
            simpa only [hl, Option.bind_eq_bind, Option.bind_none] using hfail
        | some layout =>
            simp only [Option.bind_eq_bind, Option.bind_some]
            cases hc : splitDirectedCoeff layout aB with
            | none =>
                simpa only [hc, Option.bind_none] using hfail
            | some coefficients =>
                simp only [Option.bind_some, Option.pure_def]
                obtain ⟨hdy, hparents, hdims, hy⟩ := heq layout hl
                have ha' : RowEncloses aB layout.outputShape.size a := by
                  simpa only [hdy] using ha
                have hr := represents_concat hbase layout nodes[k]!.parents hparents
                  (fun p hp => (hparent p hp).1) (fun p hp => (hparent p hp).2)
                  hdims ha' hy hc
                right
                simpa only [← hdy, sub_add_cancel] using hr
      case layernorm axis =>
        split
        · exact hfail
        · exact hconsume

end

end NN.MLTheory.CROWN.Graph.DirectedBackward
