/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.FlatBox.Affine.Real
public import TorchLean.Verify.CROWN.Op.Directed
public import TorchLean.Verify.CROWN.Op.ExecCROWN

/-!
# Soundness of the forward α-CROWN pass on operator graphs

At `ℝ`, every affine bound the forward pass gives a node encloses, at the input point `x`, the
flat value of that node in a successful real run, when the interval table encloses the run's flat
values, an input node at `ctx.inputId` has flat value `x` and the supplied slopes lie in `[0, 1]`.
`alpha?_encloses` is the rule of one operator node: detach, a linear layer with constant weight
and bias (one leading row or an empty input axis), ReLU with slopes or with phases
(`reluBounds?_encloses`), a sum over every axis in order without `keepdim`, reshape, and the
constant forms of the node's own box for every other operator but `.matmul` and `.custom`, which
have no bounds. `transfer_of_run` is the step at one node, `encloses_of_run` covers every table
whose entries the step replays, and `runAlphaCROWN_encloses` the engine's own table. The phases
enter the step only at a ReLU node (`alphaStep?_phases`). `alphaStep?_relu_inv` says what a
successful ReLU step computes, and `reluBounds?_phase_inv` does the same for `reluBounds?` with
phases. `flat_relu_of_eval` and `flat_linear_of_eval` give the flat values of a ReLU node and of a
linear layer node.

The range `[0, 1]` (`AlphaOK`) is the one α-CROWN puts on its slopes (Xu et al. 2021,
arXiv:2011.13824, Eq. (5) and (9)), and nothing in the engine clips a slope into it.
`runAlphaCROWN_encloses` needs no slope premise: the engine supplies no slopes and the defaults lie
in the range (`defaultAlphaVec_range`). The certificate reader `parseAlphaVec?`
(`TorchLean.Verify.Certificate.NodeReplay`) refuses a slope outside it, and
`TorchLeanTest.Verification.CertificatePreconditions` checks that the reader takes `[0.0, 0.75]` and
refuses `[2.0, 0.5]` and `[1e999, 0.5]`.
-/

@[expose] public section

open TorchLean.Shape (size_append size_cons size_nil)

namespace TorchLean.CROWN.Op

open Tensor
open Graph
open Linear
open Cert

noncomputable section

/-- Flat values of one width with the same entries are equal. -/
theorem Alpha.flat_ext {v v' : FlatTensor ℝ} (hn : v.n = v'.n)
    (he : ∀ i, i < v.n → getAtOrZero v.v [i] = getAtOrZero v'.v [i]) : v = v' := by
  obtain ⟨n, t⟩ := v
  obtain ⟨n', t'⟩ := v'
  simp only at hn
  subst hn
  congr
  ext i
  rw [← getAtOrZero_eq_getScalar, ← getAtOrZero_eq_getScalar]
  exact he i i.isLt

/-- What a successful `reluBounds?` with phases computes when the α table satisfies
`CROWN.Graph.AlphaOK`. The bounds `xin` have the width of the box `preB`, and `phaseRelaxVec?`
accepts the phases `nodePhases` of node `id` on `preB` with a slope vector in `[0, 1]`, the supplied
one or the default. The lower form of the result `b` is the lower relaxation applied to the lower
form of `xin`, and likewise for the upper forms. -/
theorem reluBounds?_phase_inv {alpha : Array (Option (FlatTensor ℝ))}
    {phases : Array (Option (Array Int))} {id : Nat} {xin : FlatAffineBounds ℝ} {preB : FlatBox ℝ}
    {b : FlatAffineBounds ℝ} {nodePhases : Array Int} (halpha : AlphaOK alpha)
    (hph : Cert.getPhases? phases id = some nodePhases)
    (h : reluBounds? alpha phases id xin preB = some b) :
    ∃ (hout : xin.outDim = preB.dim) (αt : Tensor ℝ [preB.dim])
      (relaxLo relaxHi : Tensor (ReLURelax ℝ) [preB.dim]),
      (∀ i : Fin preB.dim, (0 : ℝ) ≤ getScalar αt i ∧ getScalar αt i ≤ 1) ∧
      Cert.phaseRelaxVec? preB.lo preB.hi αt nodePhases = some (relaxLo, relaxHi) ∧
      b = { inDim := xin.inDim, outDim := preB.dim
            loAff := ReLU.propagateAffine relaxLo (castAffineOut hout xin.loAff)
            hiAff := ReLU.propagateAffine relaxHi (castAffineOut hout xin.hiAff) } := by
  unfold reluBounds? at h
  split at h
  · rename_i hout
    rw [hph] at h
    simp only at h
    split at h
    · rename_i αt hαt
      have hr : ∀ i : Fin preB.dim, (0 : ℝ) ≤ getScalar αt i ∧ getScalar αt i ≤ 1 := by
        split at hαt
        · rename_i αv hαv
          split at hαt
          · rename_i hα
            obtain rfl := Option.some.inj hαt
            exact castDimScalar_unit_range hα αv.v (getAlpha?_unit_range halpha hαv)
          · cases hαt
        · obtain rfl := Option.some.inj hαt
          exact defaultAlphaVec_range preB.lo preB.hi
      cases hrel : Cert.phaseRelaxVec? preB.lo preB.hi αt nodePhases with
      | none => simp [hrel] at h
      | some lh =>
        obtain ⟨lo, hi⟩ := lh
        simp only [hrel, Option.map_some, Option.some.injEq] at h
        exact ⟨hout, αt, lo, hi, hr, hrel, h.symm⟩
    · cases h
  · cases h

/-- The ReLU rule of the forward pass: when the reference's bounds enclose its flat value at `x`
and its box encloses that value, the bounds `reluBounds?` returns enclose the value's ReLU at `x`,
with or without phases, for slopes in `[0, 1]`. -/
theorem reluBounds?_encloses {alpha : Array (Option (FlatTensor ℝ))}
    {phases : Array (Option (Array Int))} {id : Nat} {ctx : AffineCtx}
    {x : Tensor ℝ [ctx.inputDim]} {xin : FlatAffineBounds ℝ} {preB : FlatBox ℝ}
    {b : FlatAffineBounds ℝ} {vp : FlatTensor ℝ} (halpha : AlphaOK alpha)
    (hpar : EnclosesAtInput ctx x xin vp) (hpb : Directed.Encloses preB vp)
    (h : reluBounds? alpha phases id xin preB = some b) :
    EnclosesAtInput ctx x b { n := vp.n, v := Tensor.relu (α := ℝ) vp.v } := by
  cases hph : Cert.getPhases? phases id with
  | some nodePhases =>
    obtain ⟨hout, αt, lo, hi, hr, hrel, rfl⟩ := reluBounds?_phase_inv halpha hph h
    exact enclosesAtInput_relu_phase ctx x xin _ preB hout αt nodePhases lo hi hr hrel hpar hpb
  | none =>
    unfold reluBounds? at h
    split at h
    · rename_i hout
      rw [hph] at h
      obtain rfl := Option.some.inj h
      refine enclosesAtInput_relu_propagate _ ?_ hout hpar hpb
      split
      · exact selected_alpha_range halpha (by assumption)
      · exact defaultAlphaVec_range preB.lo preB.hi
    · cases h

/-- The phases enter the forward step only at a ReLU node: phase tables that agree at node `id`
when it is a ReLU node give the same step there. -/
theorem alphaStep?_phases {G : IR.Graph} {P : IR.Payload ℝ}
    {ibp : Array (Option (FlatBox ℝ))} {alpha : Array (Option (FlatTensor ℝ))}
    {phases phases' : Array (Option (Array Int))} {cert : Array (Option (FlatAffineBounds ℝ))}
    {ctx : AffineCtx} {id : Nat}
    (h : ∀ (s : Shape) (refs : Array IR.Ref),
      G.nodes[id]? = some (.op ⟨_, _, Op.relu (s := s)⟩ refs) →
        Cert.getPhases? phases id = Cert.getPhases? phases' id) :
    alphaStep? G P ibp alpha phases cert ctx id = alphaStep? G P ibp alpha phases' cert ctx id := by
  unfold alphaStep?
  split
  · rfl
  · rfl
  · next o refs hn =>
    split
    · obtain ⟨ins, outs, op⟩ := o
      cases op <;> (try cases ‹Op.NumericElement›) <;> (try cases ‹Op.Element›)
      case relu s =>
        have hr : reluBounds? alpha phases id = reluBounds? alpha phases' id := by
          funext xin preB
          unfold reluBounds?
          rw [h s refs hn]
        unfold alpha?
        rw [hr]
      case «reduce» s rop dims keepdim hop =>
        cases rop <;> cases keepdim <;>
          rcases refs.toList with _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _⟩⟩⟩⟩ <;> rfl
      all_goals
        rcases refs.toList with _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _⟩⟩⟩⟩ <;> rfl
    · rfl
  · rfl

/-- What a successful forward step computes at a ReLU node whose only reference is `r`: `r` names
output `0`, `cert` holds bounds `xin` at the node of `r`, `ibp` gives `r` the box `preB`, and
`reluBounds?` returns the step's bounds from `xin` and `preB`. -/
theorem alphaStep?_relu_inv {G : IR.Graph} {P : IR.Payload ℝ}
    {ibp : Array (Option (FlatBox ℝ))} {alpha : Array (Option (FlatTensor ℝ))}
    {phases : Array (Option (Array Int))} {cert : Array (Option (FlatAffineBounds ℝ))}
    {ctx : AffineCtx} {id : Nat} {s : Shape} {refs : Array IR.Ref} {r : IR.Ref}
    {b : FlatAffineBounds ℝ}
    (hn : G.nodes[id]? = some (.op ⟨_, _, Op.relu (s := s)⟩ refs))
    (hr : refs.toList = [r]) (h : alphaStep? G P ibp alpha phases cert ctx id = some b) :
    r.out = 0 ∧ ∃ xin preB, Cert.getAff? cert r.node = some xin ∧ box? ibp r = some preB ∧
      reluBounds? alpha phases id xin preB = some b := by
  unfold alphaStep? at h
  rw [hn] at h
  simp only at h
  split at h
  · rename_i hall
    rw [hr] at h
    simp only [alpha?, Option.bind_eq_bind] at h
    obtain ⟨xin, hxin, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨preB, hpre, h⟩ := Option.bind_eq_some_iff.mp h
    have hr0 : r.out = 0 := by
      have hm : r ∈ refs := by simp [← Array.mem_toList_iff, hr]
      obtain ⟨i, hi, rfl⟩ := Array.getElem_of_mem hm
      simpa using Array.all_eq_true.mp hall i hi
    exact ⟨hr0, xin, preB, hxin, hpre, h⟩
  · cases h

/-- The flat value of a ReLU node: the ReLU of the flat value its one reference reads. -/
theorem Alpha.flat_relu_of_eval {P : IR.Payload ℝ} {inputs : List (Op.Value ℝ)}
    {earlier : Array (List (Op.Value ℝ))} {id : Nat} {s : Shape}
    {refs : Array IR.Ref} {outs : List (Op.Value ℝ)}
    (h : (IR.Node.op ⟨_, _, Op.relu (s := s)⟩ refs).eval P inputs earlier id =
      .ok outs) :
    ∃ r w vp, refs.toList = [r] ∧ r.read earlier = .ok w ∧ flatOf w = some vp ∧
      (outs[0]? >>= flatOf) = some { n := vp.n, v := Tensor.relu (α := ℝ) vp.v } := by
  obtain ⟨r, w, xt, hr, hread, hxt, rfl⟩ := invoke_one Op.rule?_relu h
  refine ⟨r, w, _, hr, hread, (flatOf_of_read (Or.inl rfl) hxt).trans (flatOf_value xt), ?_⟩
  simp [flatOf, flatOfSome, Tensor.relu, reshape_vector_map]

/-- The flat value of a linear layer node whose weight and bias references name the constants `W`
and `b` and read them (`Directed.ConstsReal`): its input reference reads the flat value of a
tensor `xt`, and its first output has the flat value of `Linear.forwardLeading L ⟨W, b⟩ xt`. -/
theorem Alpha.flat_linear_of_eval {G : IR.Graph} {P : IR.Payload ℝ}
    {inputs : List (Op.Value ℝ)} {earlier : Array (List (Op.Value ℝ))}
    {id : Nat} {L : Shape} {n m : Nat} {refs : Array IR.Ref} {rW rb rx : IR.Ref}
    {W : Tensor ℝ [m, n]} {b : Tensor ℝ [m]} {outs : List (Op.Value ℝ)}
    (hconsts : Directed.ConstsReal G P earlier [rW, rb]) (hr : refs.toList = [rW, rb, rx])
    (hW : constAt? G P rW (.value [m, n]) = some W) (hb : constAt? G P rb (.value [m]) = some b)
    (h : (IR.Node.op ⟨_, _, Op.linear (lead := L) (inDim := n) (outDim := m)⟩
      refs).eval P inputs earlier id = .ok outs) :
    ∃ w xt, rx.read earlier = .ok w ∧ flatOf w = some (flatOfSome ⟨L ++ [n], xt⟩) ∧
      (outs[0]? >>= flatOf) =
        some (flatOfSome ⟨L ++ [m], Linear.forwardLeading L ⟨W, b⟩ xt⟩) := by
  obtain ⟨rW', rb', rx', vW, vb, vx, Wr, br, xt, hr', hreadW, hreadb, hreadx, hxW, hxb, hxx, rfl⟩ :=
    invoke_three (Option.some_get Op.rule?_linear_isSome).symm h
  rw [hr, List.cons.injEq, List.cons.injEq, List.cons.injEq] at hr'
  obtain ⟨rfl, rfl, rfl, -⟩ := hr'
  have hWr : Wr = W := (Directed.read_constAt? hconsts (by simp) hW hreadW hxW).trans
    (Tensor.Rep.map_id W)
  have hbr : br = b := (Directed.read_constAt? hconsts (by simp) hb hreadb hxb).trans
    (Tensor.Rep.map_id b)
  subst hWr hbr
  refine ⟨vx, xt, hreadx, (flatOf_of_read (Or.inl rfl) hxx).trans (flatOf_value xt), ?_⟩
  rw [Op.rule?_linear_forward]
  simp only [List.getElem?_cons_zero, Option.bind_eq_bind, Option.bind_some, flatOf_value]

/-- The rule of `alpha?` at one operator node: when the node's operator evaluates on the values
its references read, the weight and bias references of a linear layer read their constants
(`Directed.ConstsReal`), the bounds of each reference enclose the flat value it reads at `x`, its
box encloses that value, and the node's own box encloses the node's flat value, every affine bound
the rule returns encloses the flat value of the node's first output at `x`, for slopes in
`[0, 1]`. -/
theorem alpha?_encloses {G : IR.Graph} {P : IR.Payload ℝ}
    {ibp : Array (Option (FlatBox ℝ))} {alpha : Array (Option (FlatTensor ℝ))}
    {phases : Array (Option (Array Int))} {cert : Array (Option (FlatAffineBounds ℝ))}
    {ctx : AffineCtx} {x : Tensor ℝ [ctx.inputDim]}
    {inputs : List (Op.Value ℝ)} {earlier : Array (List (Op.Value ℝ))}
    {id : Nat} {o : IR.Operator} {refs : Array IR.Ref} {b : FlatAffineBounds ℝ}
    {outs : List (Op.Value ℝ)}
    (hconsts : ∀ (L : Shape) (n m : Nat) (rW rb rx : IR.Ref),
      o = ⟨_, _, Op.linear (lead := L) (inDim := n) (outDim := m)⟩ →
      refs.toList = [rW, rb, rx] → Directed.ConstsReal G P earlier [rW, rb])
    (hcert : ∀ r ∈ refs.toList, ∀ (br : FlatAffineBounds ℝ) (w : Op.Value ℝ)
        (v : FlatTensor ℝ),
      Cert.getAff? cert r.node = some br → r.read earlier = .ok w → flatOf w = some v →
        EnclosesAtInput ctx x br v)
    (hbox : ∀ r ∈ refs.toList, ∀ (B : FlatBox ℝ) (w : Op.Value ℝ) (v : FlatTensor ℝ),
      box? ibp r = some B → r.read earlier = .ok w → flatOf w = some v → Directed.Encloses B v)
    (hself : ∀ (B : FlatBox ℝ) (v : FlatTensor ℝ), (ibp[id]?).join = some B →
      (outs[0]? >>= flatOf) = some v → Directed.Encloses B v)
    (halpha : AlphaOK alpha)
    (hc : alpha? G P ibp alpha phases cert ctx id o refs.toList = some b)
    (h : (IR.Node.op o refs).eval P inputs earlier id = .ok outs) :
    ∀ v, (outs[0]? >>= flatOf) = some v → EnclosesAtInput ctx x b v := by
  intro v hv
  have hfallback : boxBounds? ibp ctx id = some b → EnclosesAtInput ctx x b v := by
    intro hB
    unfold boxBounds? at hB
    obtain ⟨B, hB', rfl⟩ := Option.map_eq_some_iff.mp hB
    exact enclosesAtInput_boundsConst (hself B v hB' hv)
  obtain ⟨ins, outs', op⟩ := o
  cases op <;> (try cases ‹Op.NumericElement›) <;> (try cases ‹Op.Element›)
  all_goals try (simp only [alpha?] at hc; exact hfallback hc)
  case matmul => simp [alpha?] at hc
  case custom => simp [alpha?] at hc
  case detach s =>
    obtain ⟨r, w, xt, hr, hread, hxt, rfl⟩ := invoke_one Op.rule?_detach h
    have hrm : r ∈ refs.toList := by rw [hr]; simp
    rw [hr] at hc
    simp only [alpha?] at hc
    simp only [List.getElem?_cons_zero, Option.bind_eq_bind, Option.bind_some] at hv
    exact hcert r hrm b w v hc hread ((flatOf_of_read (Or.inr (Or.inr rfl)) hxt).trans hv)
  case reshape s t hst =>
    obtain ⟨r, w, xt, hr, hread, hxt, rfl⟩ := invoke_one (Op.rule?_reshape hst) h
    have hrm : r ∈ refs.toList := by rw [hr]; simp
    rw [hr] at hc
    simp only [alpha?, Option.bind_eq_bind] at hc
    obtain ⟨xin, hxin, hc⟩ := Option.bind_eq_some_iff.mp hc
    have hpar := hcert r hrm xin w _ hxin hread
      ((flatOf_of_read (Or.inl rfl) hxt).trans (flatOf_value xt))
    split at hc
    · rename_i hout
      obtain rfl := Option.some.inj hc
      simp only [List.getElem?_cons_zero, Option.bind_eq_bind, Option.bind_some, flatOf_value,
        Option.some.injEq] at hv
      subst hv
      have hvout : (flatOfSome ⟨s, xt⟩).n = t.size := hst
      have henc := enclosesAtInput_castOut ctx x xin _ hout hvout hpar
      convert henc using 1
      refine Alpha.flat_ext ?_ fun i hi => ?_
      · rfl
      · rw [getAtOrZero_castDimScalar hvout]
        simp only [Graph.flatOfSome, Tensor.reshape_vector_reshape]
        exact Directed.getAtOrZero_eqRec _ _ _
    · cases hc
  case relu s =>
    obtain ⟨r, w, vp, hr, hread, hflat, hvr⟩ := Alpha.flat_relu_of_eval h
    have hrm : r ∈ refs.toList := by rw [hr]; simp
    rw [hvr] at hv
    obtain rfl := Option.some.inj hv
    rw [hr] at hc
    simp only [alpha?, Option.bind_eq_bind] at hc
    obtain ⟨xin, hxin, hc⟩ := Option.bind_eq_some_iff.mp hc
    obtain ⟨preB, hpre, hc⟩ := Option.bind_eq_some_iff.mp hc
    exact reluBounds?_encloses halpha (hcert r hrm xin w _ hxin hread hflat)
      (hbox r hrm preB w _ hpre hread hflat) hc
  case linear L n m =>
    obtain ⟨rW, rb, rx, _, _, _, _, _, _, hr, -⟩ :=
      invoke_three (Option.some_get Op.rule?_linear_isSome).symm h
    rw [hr] at hc
    simp only [alpha?, Option.bind_eq_bind] at hc
    obtain ⟨W, hW, hc⟩ := Option.bind_eq_some_iff.mp hc
    obtain ⟨bv, hb, hc⟩ := Option.bind_eq_some_iff.mp hc
    obtain ⟨xin, hxin, hc⟩ := Option.bind_eq_some_iff.mp hc
    obtain ⟨vx, xt, hreadx, hflat, hvout⟩ :=
      Alpha.flat_linear_of_eval (hconsts L n m rW rb rx rfl hr) hr hW hb h
    have hpar := hcert rx (by simp [hr]) xin vx _ hxin hreadx hflat
    rw [hvout] at hv
    obtain rfl := Option.some.inj hv
    split at hc
    · rename_i hout
      have hdim : xin.outDim = (L ++ [n]).size := hpar.2.1
      have hcond : (flatOfSome ⟨L ++ [n], xt⟩).n = n := hdim.symm.trans hout
      by_cases h0 : n = 0
      · simp only [h0, ↓reduceIte, Option.some.injEq] at hc
        subst hc
        have hsize : (L ++ [m]).size = L.size * m := by simp [Shape.size_append]
        have hspec : Linear.forward (α := ℝ) (⟨W, bv⟩ : Linear ℝ n m) = fun _ => bv :=
          funext (forward_of_inDim_zero h0 _ _)
        have hval : flatOfSome ⟨L ++ [m], Linear.forwardLeading L ⟨W, bv⟩ xt⟩ =
            { n := L.size * m, v := Graph.linearBiasRows bv L.size } :=
          flatTensor_eq_of_cast hsize _ _ (by
            rw [Linear.forwardLeading, hspec]
            exact flatten_mapLeading_const bv L xt hsize)
        rw [hval]
        exact enclosesAtInput_boundsConst
          (B0 := { dim := _, lo := Graph.linearBiasRows bv _, hi := Graph.linearBiasRows bv _ })
          ⟨rfl, fun i => ⟨le_rfl, le_rfl⟩⟩
      · simp only [h0, ↓reduceIte, Option.some.injEq] at hc
        subst hc
        have hone : L.size = 1 := by
          have h1 : (L ++ [n]).size = n := hcond
          simp only [Shape.size_append, Shape.size_nil, Shape.size_cons] at h1
          exact (Nat.mul_eq_right (Nat.pos_of_ne_zero h0).ne').mp (by simpa using h1)
        have hm : (L ++ [m]).size = m := by simp [Shape.size_append, hone]
        convert enclosesAtInput_linear W bv hout hcond hpar using 1
        exact flatTensor_eq_of_cast hm _ _ (flatten_mapLeading_of_size_one _ L hone xt hm hcond)
    · cases hc
  case «reduce» s rop dims keepdim hop =>
    obtain ⟨r, w, xt, hr, hread, hxt, rfl⟩ :=
      invoke_one (Op.rule?_reduce rop dims keepdim hop) h
    have hrm : r ∈ refs.toList := by rw [hr]; simp
    rw [hr] at hc
    cases rop <;> cases keepdim <;> simp only [alpha?] at hc
    all_goals try exact hfallback hc
    split at hc
    · rename_i hdims
      obtain rfl : dims = List.finRange s.rank :=
        List.map_injective_iff.mpr Fin.val_injective
          (hdims.trans (List.map_coe_finRange_eq_range).symm)
      simp only [Option.bind_eq_bind] at hc
      obtain ⟨xin, hxin, hc⟩ := Option.bind_eq_some_iff.mp hc
      obtain rfl := Option.some.inj hc
      have hflat : flatOf w = some (flatOfSome ⟨s, xt⟩) :=
        (flatOf_of_read (Or.inl rfl) hxt).trans (flatOf_value xt)
      have hpar := hcert r hrm xin w _ hxin hread hflat
      have hvIn : (flatOfSome ⟨s, xt⟩).n = xin.outDim := hpar.2.1.symm
      have henc := enclosesAtInput_linear
        (Tensor.full (α := ℝ) (.dim 1 (.dim xin.outDim .scalar)) 1)
        (Tensor.full (α := ℝ) (.dim 1 .scalar) 0) rfl hvIn hpar
      simp only [List.getElem?_cons_zero, Option.bind_eq_bind, Option.bind_some, flatOf_value,
        Option.some.injEq] at hv
      subst hv
      convert henc using 1
      have hone : (s.reduce (List.finRange s.rank) false).size = 1 := by
        rw [Shape.reduce_finRange]; rfl
      refine Alpha.flat_ext hone fun i hi => ?_
      obtain rfl : i = 0 := by
        have : i < (s.reduce (List.finRange s.rank) false).size := hi
        omega
      rw [Directed.flat_reduce_sum, forward_zero_bias,
        show (0 : Nat) = (⟨0, Nat.one_pos⟩ : Fin 1).val from rfl, getAtOrZero_eq_getScalar,
        Tensor.getScalar_contract_matVec_eq_sum]
      have hterm : ∀ k : Fin xin.outDim,
          get2 (Tensor.full (α := ℝ) (.dim 1 (.dim xin.outDim .scalar)) 1) ⟨0, Nat.one_pos⟩ k *
            getScalar (castDimScalar (α := ℝ) hvIn (flatOfSome ⟨s, xt⟩).v) k =
          getAtOrZero (flatOfSome ⟨s, xt⟩).v [k.val] := by
        intro k
        rw [get2_full, one_mul, ← getAtOrZero_eq_getScalar, getAtOrZero_castDimScalar]
      rw [Finset.sum_congr rfl fun k _ => hterm k]
      exact Fin.sum_congr' (fun k : Fin xin.outDim => getAtOrZero (flatOfSome ⟨s, xt⟩).v [k.val])
        hvIn
    · exact hfallback hc


/-- A reference to output `0` that reads `w` in a run's outputs names the flat value of `w`. -/
theorem Alpha.runVal_of_read {values : Array (List (Op.Value ℝ))} {r : IR.Ref}
    {w : Op.Value ℝ} (h0 : r.out = 0) (h : r.read values = .ok w) :
    runVal values r.node = flatOf w := by
  rw [← box?_runVals h0 h]
  simp only [box?, h0, ↓reduceIte, runVals, runVal, Array.getElem?_map]
  cases values[r.node]? <;> rfl

/-- The affine bounds at the designated input are the identity: they enclose at `x` the input's
flat value when it is `x`. -/
theorem Alpha.enclosesAtInput_identity {ctx : AffineCtx} {x : Tensor ℝ [ctx.inputDim]}
    {v : FlatTensor ℝ} (h : v.n = ctx.inputDim) (hv : castDimScalar (α := ℝ) h v.v = x) :
    EnclosesAtInput ctx x (Graph.boundsIdentity (α := ℝ) ctx.inputDim) v := by
  subst hv
  obtain ⟨n, w⟩ := v
  change n = ctx.inputDim at h
  subst h
  -- Along `rfl` both casts are the identity, so the box is the point box of `w`.
  refine ⟨rfl, ?_⟩
  exact (congrArg (fun B => Directed.RowEncloses B ctx.inputDim fun i => getAtOrZero w [i])
    (boundsEvalAt_boundsIdentity w)).mpr ⟨rfl, fun _ => ⟨le_rfl, le_rfl⟩⟩

/-- The step of `encloses_of_run` at one node of a successful real run: when the interval table
encloses the run's flat values, an input node at `ctx.inputId` has flat value `x`, the slopes lie in
`[0, 1]`, and the bounds `cert` gives the references of an operator node enclose their flat values
at `x`, every bound `alphaStep?` returns encloses the node's flat value at `x`. -/
theorem Alpha.transfer_of_run (G : IR.Graph) (P : IR.Payload ℝ)
    (ibp : Array (Option (FlatBox ℝ))) (alpha : Array (Option (FlatTensor ℝ)))
    (phases : Array (Option (Array Int))) (cert : Array (Option (FlatAffineBounds ℝ)))
    (ctx : AffineCtx) (x : Tensor ℝ [ctx.inputDim])
    (inputs : List (Op.Value ℝ)) (values : Array (List (Op.Value ℝ)))
    (hibp : ∀ id (B : FlatBox ℝ) (v : FlatTensor ℝ), (ibp[id]?).join = some B →
      runVal values id = some v → Directed.Encloses B v)
    (hx : ∀ k p v, G.nodes[ctx.inputId]? = some (.input k p) → runVal values ctx.inputId = some v →
      ∃ h : v.n = ctx.inputDim, castDimScalar (α := ℝ) h v.v = x)
    (halpha : AlphaOK alpha) (hrun : G.run P inputs = .ok values) :
    ∀ id, id < G.nodes.size →
      (∀ o refs, G.nodes[id]? = some (.op o refs) → ∀ r ∈ refs.toList, r.node < id →
        ∀ (br : FlatAffineBounds ℝ) (v : FlatTensor ℝ), Cert.getAff? cert r.node = some br →
          runVal values r.node = some v → EnclosesAtInput ctx x br v) →
      ∀ (b : FlatAffineBounds ℝ) (v : FlatTensor ℝ),
        alphaStep? G P ibp alpha phases cert ctx id = some b →
        runVal values id = some v → EnclosesAtInput ctx x b v := by
  intro id hid hparents b v hb hv
  obtain ⟨n, hn⟩ : ∃ n, G.nodes[id]? = some n := ⟨_, Array.getElem?_eq_getElem hid⟩
  obtain ⟨pre, outs, hsize, hagree, heval, hout⟩ := IR.Graph.run_step hrun hn
  have hv' : (outs[0]? >>= flatOf) = some v := by simpa [runVal, hout] using hv
  unfold alphaStep? at hb
  rw [hn] at hb
  cases n with
  | input k p =>
      simp only at hb
      split at hb
      · rename_i hin
        obtain rfl := Option.some.inj hb
        subst hin
        obtain ⟨h, hxv⟩ := hx k p v hn hv
        exact enclosesAtInput_identity h hxv
      · cases hb
  | const p =>
      obtain ⟨w, hw, rfl⟩ := const_eval hn heval
      simp only [hw, Option.bind_eq_bind, Option.bind_some] at hb
      simp only [List.getElem?_cons_zero, Option.bind_eq_bind, Option.bind_some] at hv'
      rw [hv', Option.map_some] at hb
      obtain rfl := Option.some.inj hb
      exact enclosesAtInput_boundsConst (B0 := { dim := v.n, lo := v.v, hi := v.v })
        ⟨rfl, fun i => ⟨le_rfl, le_rfl⟩⟩
  | op o refs =>
      simp only at hb
      split at hb
      · rename_i hall
        have h0 : ∀ r ∈ refs.toList, r.out = 0 := by
          intro r hr
          have hr' : r ∈ refs := by simpa using hr
          obtain ⟨i, hi, rfl⟩ := Array.getElem_of_mem hr'
          simpa using Array.all_eq_true.mp hall i hi
        have hread : ∀ r ∈ refs.toList, ∀ w, r.read pre = .ok w → runVal values r.node = flatOf w :=
          fun r hr w hw => runVal_of_read (h0 r hr) (read_of_agree hagree hsize hw)
        refine alpha?_encloses
          (fun _ _ _ rW rb _ _ _ =>
            Directed.constsReal_of_run (Directed.realConsts_self P) hrun hagree hsize [rW, rb])
          (fun r hr br w vr hbr hw hvr => ?_) (fun r hr B w vr hB hw hvr => ?_)
          (fun B vr hB hvr => hibp id B vr hB (by simpa [runVal, hout] using hvr)) halpha hb heval
          v hv'
        · exact hparents o refs hn r hr (hsize ▸ IR.Ref.read_lt hw) br vr hbr
            ((hread r hr w hw).trans hvr)
        · have hB' : (ibp[r.node]?).join = some B := by simpa [box?, h0 r hr] using hB
          exact hibp r.node B vr hB' ((hread r hr w hw).trans hvr)
      · cases hb

/-- Every bound of a table whose present entries `alphaStep?` replays encloses at `x` the flat
value of its node in a successful real run, when the interval table encloses the run's flat
values, an input node at `ctx.inputId` has flat value `x` and the slopes lie in `[0, 1]`. -/
theorem Alpha.encloses_of_run (G : IR.Graph) (P : IR.Payload ℝ)
    (ibp : Array (Option (FlatBox ℝ))) (alpha : Array (Option (FlatTensor ℝ)))
    (phases : Array (Option (Array Int))) (cert : Array (Option (FlatAffineBounds ℝ)))
    (ctx : AffineCtx) (x : Tensor ℝ [ctx.inputDim])
    (inputs : List (Op.Value ℝ)) (values : Array (List (Op.Value ℝ)))
    (hcert : ∀ id (b : FlatAffineBounds ℝ), id < G.nodes.size → cert[id]! = some b →
      alphaStep? G P ibp alpha phases cert ctx id = some b)
    (hibp : ∀ id (B : FlatBox ℝ) (v : FlatTensor ℝ), (ibp[id]?).join = some B →
      runVal values id = some v → Directed.Encloses B v)
    (hx : ∀ k p v, G.nodes[ctx.inputId]? = some (.input k p) → runVal values ctx.inputId = some v →
      ∃ h : v.n = ctx.inputDim, castDimScalar (α := ℝ) h v.v = x)
    (halpha : AlphaOK alpha) (hrun : G.run P inputs = .ok values) :
    ∀ id, id < G.nodes.size → ∀ (b : FlatAffineBounds ℝ) (v : FlatTensor ℝ),
      cert[id]! = some b → runVal values id = some v → EnclosesAtInput ctx x b v := by
  intro id
  induction id using Nat.strong_induction_on with
  | _ k ih =>
    intro hk b v hb hv
    refine transfer_of_run G P ibp alpha phases cert ctx x inputs values hibp hx halpha hrun k hk
      (fun o refs hn r hr hlt br vr hbr hvr => ?_) b v (hcert k b hk hb) hv
    exact ih r.node hlt (lt_trans hlt hk) br vr (getElem!_of_getAff?_eq_some hbr) hvr

/-- Every bound of the forward pass `runAlphaCROWN` encloses at `x` the flat value of its node in
a successful real run, when the interval table encloses the run's flat values and an input node at
`ctx.inputId` has flat value `x`. -/
theorem runAlphaCROWN_encloses (G : IR.Graph) (P : IR.Payload ℝ) (ctx : AffineCtx)
    (ibp : Array (Option (FlatBox ℝ))) (x : Tensor ℝ [ctx.inputDim])
    (inputs : List (Op.Value ℝ)) (values : Array (List (Op.Value ℝ)))
    (hibp : ∀ id (B : FlatBox ℝ) (v : FlatTensor ℝ), (ibp[id]?).join = some B →
      runVal values id = some v → Directed.Encloses B v)
    (hx : ∀ k p v, G.nodes[ctx.inputId]? = some (.input k p) → runVal values ctx.inputId = some v →
      ∃ h : v.n = ctx.inputDim, castDimScalar (α := ℝ) h v.v = x)
    (hrun : G.run P inputs = .ok values) :
    ∀ id (b : FlatAffineBounds ℝ) (v : FlatTensor ℝ), (runAlphaCROWN G P ctx ibp)[id]! = some b →
      runVal values id = some v → EnclosesAtInput ctx x b v := by
  have halpha : AlphaOK (Array.replicate G.nodes.size (none : Option (FlatTensor ℝ))) := by
    intro id hid
    rw [getElem!_pos _ id hid, Array.getElem_replicate]
    trivial
  let step := fun cert id => alphaStep? G P ibp (Array.replicate G.nodes.size none)
    (inferredPhases G ibp) cert ctx id
  change ∀ id (b : FlatAffineBounds ℝ) (v : FlatTensor ℝ),
    (tabulate step G.nodes.size)[id]! = some b →
    runVal values id = some v → EnclosesAtInput ctx x b v
  intro id
  induction id using Nat.strong_induction_on with
  | _ k ih =>
    intro b v hb hv
    by_cases hk : k < G.nodes.size
    · have hself := tabulate_getElem?_self (step := step) hk
      rw [getElem!_def, hself] at hb
      refine Alpha.transfer_of_run G P ibp _ _ (tabulate step k) ctx x inputs values hibp
        hx halpha hrun
        k hk (fun o refs hn r hr hlt br vr hbr hvr => ?_) b v hb hv
      have hlt' := tabulate_getElem?_of_lt (step := step) (j := r.node) hlt (le_of_lt hk)
      have hentry : (tabulate step G.nodes.size)[r.node]! = some br := by
        rw [getElem!_def, hlt', ← getElem!_def]
        exact getElem!_of_getAff?_eq_some hbr
      exact ih r.node hlt br vr hentry hvr
    · rw [getElem!_neg _ k (by rw [size_tabulate]; omega)] at hb
      cases hb

end

end TorchLean.CROWN.Op
