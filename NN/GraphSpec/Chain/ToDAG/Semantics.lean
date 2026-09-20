/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.Chain.ToDAG.Core
public import NN.GraphSpec.Chain.Semantics
public import NN.GraphSpec.DAG.Semantics
/-!
# Pure semantics of chain-to-DAG conversion

`LowerToDAG.Chain.eval_toDAGTerm` proves that the structural conversion preserves the direct
`Interp.spec` interpretation for every sequential chain, including custom primitives.

The induction keeps arbitrary parameter prefixes, suffixes, and temporary SSA values in the
ambient environment. Parameter lookup and environment reassociation are proved independently of
primitive arithmetic. Since the embedding reuses `Primitive.specFwd`, no agreement hypothesis
between a primitive's `specFwd` and its executable `program` is needed here. Such an agreement is
still required to transfer this pure result to program execution or a native backend.
-/

@[expose] public section

open Spec TorchLean
namespace NN.GraphSpec.LowerToDAG
variable {α : Type} [Storage α]

private theorem get_append_left_heq {ss ts : List Shape}
    (xs : TensorPack α ss) (ys : TensorPack α ts) (i : Fin ss.length)
    (j : Fin (ss ++ ts).length) (h : j.val = i.val) :
    HEq (TensorPack.get (TensorPack.append xs ys) j) (TensorPack.get xs i) := by
  induction ss with
  | nil => exact Fin.elim0 i
  | cons s ss ih =>
    cases xs with
    | cons x xs =>
      cases i with
      | mk i hi =>
        cases j with
        | mk j hj =>
          dsimp at h
          subst j
          cases i with
          | zero => rfl
          | succ i =>
              exact ih xs ⟨i, Nat.lt_of_succ_lt_succ hi⟩
                ⟨i, Nat.lt_of_succ_lt_succ hj⟩ rfl

private theorem get_append_right_heq {ss ts : List Shape}
    (xs : TensorPack α ss) (ys : TensorPack α ts) (i : Fin ts.length)
    (j : Fin (ss ++ ts).length) (h : j.val = ss.length + i.val) :
    HEq (TensorPack.get (TensorPack.append xs ys) j) (TensorPack.get ys i) := by
  induction ss with
  | nil =>
    cases xs
    have : j = i := Fin.ext (by simpa using h)
    subst j
    rfl
  | cons s ss ih =>
    cases xs with
    | cons x xs =>
      cases j with
      | mk j hj =>
        cases j with
        | zero => simp only [List.length_cons] at h; omega
        | succ j =>
          exact ih xs ⟨j, Nat.lt_of_succ_lt_succ hj⟩ (by
            change j = ss.length + i.val
            change j + 1 = ss.length + 1 + i.val at h
            omega)

variable [Context α]

private theorem eval_castTerm_heq {Γ : List Shape} {s t : Shape}
    (env : TensorPack α Γ) (h : s = t) (term : DAG.Term Γ s) :
    HEq (DAG.Term.eval env (castTerm h term)) (DAG.Term.eval env term) := by
  cases h
  rfl

private theorem eval_mkParamTerm {pre ps post extra : List Shape}
    (a : TensorPack α pre) (b : TensorPack α ps)
    (c : TensorPack α post) (d : TensorPack α extra) (i : Fin ps.length) :
    DAG.Term.eval (TensorPack.append (TensorPack.append (TensorPack.append a b) c) d)
      (mkParamTerm (pre := pre) (post := post) (extra := extra) i) =
      TensorPack.get b i := by
  unfold mkParamTerm
  apply eq_of_heq
  apply HEq.trans (eval_castTerm_heq _ _ _)
  simp only [DAG.Term.eval, DAG.Env.tget_ofFin]
  let j : Fin (pre ++ ps).length := ⟨pre.length + i.val, by simp⟩
  let k : Fin ((pre ++ ps) ++ post).length := ⟨j.val, by simp only [j, List.length_append]; omega⟩
  exact (get_append_left_heq _ d k _ rfl).trans
    ((get_append_left_heq _ c j k rfl).trans (get_append_right_heq a b i j rfl))
private theorem eval_argsOfFn {Γ ps : List Shape} (env : TensorPack α Γ)
    (params : TensorPack α ps) (f : ∀ i : Fin ps.length, DAG.Term Γ (ps.get i))
    (hf : ∀ i, DAG.Term.eval env (f i) = TensorPack.get params i) :
    DAG.Term.evalArgs env (argsOfFn ps f) = params := by
  induction ps with
  | nil => cases params; rfl
  | cons shape ps ih =>
    cases params with
    | cons value rest =>
      simp only [argsOfFn, DAG.Term.evalArgs]
      congr 1
      · exact hf ⟨0, by simp⟩
      · apply ih rest
        intro i
        exact hf ⟨i.val + 1, Nat.succ_lt_succ i.isLt⟩

private theorem eval_append1 {Γ ps : List Shape} {σ : Shape}
    (env : TensorPack α Γ) (args : DAG.Args Γ ps) (x : DAG.Term Γ σ) :
    DAG.Term.evalArgs env (Args.append1 args x) =
      TensorPack.append (DAG.Term.evalArgs env args) (.cons (DAG.Term.eval env x) .nil) := by
  induction ps with
  | nil => cases args; rfl
  | cons shape ps ih =>
    cases args with
    | cons value rest =>
      simp only [Args.append1, DAG.Term.evalArgs, TensorPack.append, ih rest]

private theorem eval_primCall {pre ps post extra : List Shape} {σ τ : Shape}
    (a : TensorPack α pre) (b : TensorPack α ps)
    (c : TensorPack α post) (d : TensorPack α extra)
    (p : Primitive ps σ τ) (x : DAG.Term ((pre ++ ps ++ post) ++ extra) σ) :
    let env := TensorPack.append (TensorPack.append (TensorPack.append a b) c) d
    DAG.Term.eval env (primCall p x) = p.specFwd b (DAG.Term.eval env x) := by
  dsimp only
  unfold primCall
  dsimp only
  simp only [eq_mpr_eq_cast, eq_mp_eq_cast, cast_cast, cast_eq]
  rw [DAG.Term.eval_op, eval_append1, eval_argsOfFn _ b _ (eval_mkParamTerm a b c d)]
  simp only [Primitive.toDAGPrimOp, TensorPack.split_append]

private theorem eval_castEnvTerm {Γ Δ : List Shape} {σ : Shape}
    (env : TensorPack α Δ) (h : Γ = Δ) (term : DAG.Term Γ σ) :
    DAG.Term.eval env (castEnvTerm h term) = DAG.Term.eval (TensorPack.cast h.symm env) term := by
  cases h
  rfl

omit [Context α] in
private theorem pack_cast_cons {s : Shape} {ss ts : List Shape} (h : ss = ts)
    (x : Tensor α s) (xs : TensorPack α ss) :
    TensorPack.cast (congrArg (List.cons s) h) (.cons x xs) =
      .cons x (TensorPack.cast h xs) := by
  cases h
  rfl

omit [Context α] in
private theorem pack_cast_assoc {ss ts us : List Shape}
    (xs : TensorPack α ss) (ys : TensorPack α ts) (zs : TensorPack α us) :
    TensorPack.cast (List.append_assoc ss ts us)
      (TensorPack.append (TensorPack.append xs ys) zs) =
      TensorPack.append xs (TensorPack.append ys zs) := by
  induction ss with
  | nil => cases xs; rfl
  | cons s ss ih =>
    cases xs with
    | cons x xs =>
      change TensorPack.cast (congrArg (List.cons s) (List.append_assoc ss ts us))
        (.cons x ((xs.append ys).append zs)) = .cons x (xs.append (ys.append zs))
      rw [pack_cast_cons (List.append_assoc ss ts us), ih xs]

omit [Context α] in
private theorem pack_append_split {ps qs : List Shape} (params : TensorPack α (ps ++ qs)) :
    (TensorPack.split params).1.append (TensorPack.split params).2 = params := by
  induction ps with
  | nil => rfl
  | cons s ps ih =>
    cases params with
    | cons x xs =>
      simpa only [TensorPack.split, TensorPack.append] using
        congrArg (TensorPack.cons x) (ih xs)

omit [Context α] in
private theorem pack_cast_eq_of_heq {ss ts : List Shape} (h : ss = ts)
    {xs : TensorPack α ss} {ys : TensorPack α ts} (hxy : HEq xs ys) :
    TensorPack.cast h xs = ys := by
  cases h
  exact eq_of_heq hxy

omit [Context α] in
private theorem pack_append_heq {ss ts us vs : List Shape}
    (h : ss = us) (h' : ts = vs)
    {xs : TensorPack α ss} {ys : TensorPack α ts}
    {xs' : TensorPack α us} {ys' : TensorPack α vs}
    (hx : HEq xs xs') (hy : HEq ys ys') :
    HEq (xs.append ys) (xs'.append ys') := by
  cases h
  cases h'
  cases eq_of_heq hx
  cases eq_of_heq hy
  rfl

omit [Context α] in
private theorem pack_assoc_heq {ss ts us : List Shape}
    (xs : TensorPack α ss) (ys : TensorPack α ts) (zs : TensorPack α us) :
    HEq ((xs.append ys).append zs) (xs.append (ys.append zs)) := by
  exact (cast_heq _ _).symm.trans (heq_of_eq (pack_cast_assoc xs ys zs))

omit [Context α] in
private theorem pack_four_heq {ss ts us vs : List Shape}
    (a : TensorPack α ss) (b : TensorPack α ts)
    (c : TensorPack α us) (d : TensorPack α vs) :
    HEq ((a.append (b.append c)).append d) ((a.append b).append (c.append d)) := by
  apply (pack_assoc_heq a (b.append c) d).trans
  apply HEq.trans (pack_append_heq rfl (List.append_assoc ts us vs)
    (HEq.rfl : HEq a a) (pack_assoc_heq b c d))
  exact (pack_assoc_heq a b (c.append d)).symm

private theorem eval_lastOfFin {Γ : List Shape} {s : Shape}
    (env : TensorPack α Γ) (value : Tensor α s)
    (i : Fin (Γ ++ [s]).length) (hi : i.val = Γ.length)
    (h : (Γ ++ [s]).get i = s) :
    DAG.Term.eval (env.append (.cons value .nil))
      (castTerm h (DAG.Term.var (DAG.Var.ofFin i))) = value := by
  apply eq_of_heq
  apply (eval_castTerm_heq _ _ _).trans
  simp only [DAG.Term.eval, DAG.Env.tget_ofFin]
  exact get_append_right_heq env (.cons value .nil) ⟨0, by simp⟩ i (by simpa using hi)

/-- The parameter segment is preserved even inside an ambient SSA environment. -/
private theorem eval_toTerm {ps : List Shape} {σ τ : Shape} (g : Chain ps σ τ) :
    ∀ {pre post extra : List Shape} (a : TensorPack α pre) (b : TensorPack α ps)
      (c : TensorPack α post) (d : TensorPack α extra)
      (x : DAG.Term ((pre ++ ps ++ post) ++ extra) σ),
      DAG.Term.eval (((a.append b).append c).append d) (toTerm g x) =
        Interp.spec g b (DAG.Term.eval (((a.append b).append c).append d) x) := by
  induction g with
  | id s =>
    intro pre post extra a b c d x
    simp only [toTerm, Interp.spec, eq_mpr_eq_cast, eq_mp_eq_cast, cast_cast, cast_eq]
  | prim p =>
    intro pre post extra a b c d x
    simp only [toTerm, Interp.spec, eq_mpr_eq_cast, eq_mp_eq_cast, cast_cast, cast_eq]
    exact eval_primCall a b c d p x
  | @seq ps₁ ps₂ σ middle τ g₁ g₂ ih₁ ih₂ =>
    intro pre post extra a b c d x
    have hb := pack_append_split b
    generalize hleft : (TensorPack.split b).1 = b₁ at hb
    generalize hright : (TensorPack.split b).2 = b₂ at hb
    subst b
    simp only [toTerm, eq_mpr_eq_cast, eq_mp_eq_cast, cast_cast, cast_eq]
    simp only [DAG.Term.eval, eval_castEnvTerm, Interp.spec, TensorPack.split_append]
    -- The first stage reads its own parameter prefix; the second stage sees the bound result.
    have env₁ : TensorPack.cast (by simp [List.append_assoc])
        (((a.append (b₁.append b₂)).append c).append d) =
        (((a.append b₁).append (b₂.append c)).append d) := by
      apply pack_cast_eq_of_heq
      exact pack_append_heq (by simp [List.append_assoc]) rfl (pack_four_heq a b₁ b₂ c) HEq.rfl
    rw [env₁, ih₁]
    rw [eval_castEnvTerm]
    have env₁back : TensorPack.cast (by simp [List.append_assoc])
        (((a.append b₁).append (b₂.append c)).append d) =
        (((a.append (b₁.append b₂)).append c).append d) := by
      apply pack_cast_eq_of_heq
      exact (pack_append_heq (by simp [List.append_assoc]) rfl
        (pack_four_heq a b₁ b₂ c) (HEq.rfl : HEq d d)).symm
    rw [env₁back]
    -- Reassociate the environment without changing the order of parameters or temporary values.
    let v := Interp.spec g₁ b₁ (DAG.Term.eval (((a.append (b₁.append b₂)).append c).append d) x)
    have henv₂ : HEq
        ((((a.append (b₁.append b₂)).append c).append d).append (.cons v .nil))
        ((((a.append b₁).append b₂).append c).append (d.append (.cons v .nil))) := by
      have hab := (pack_assoc_heq a b₁ b₂).symm
      have habc := pack_append_heq (by simp [List.append_assoc]) rfl hab (HEq.rfl : HEq c c)
      have habcd := pack_append_heq (by simp [List.append_assoc]) rfl habc (HEq.rfl : HEq d d)
      exact (pack_append_heq (by simp [List.append_assoc]) rfl habcd
        (HEq.rfl : HEq (.cons v .nil : TensorPack α [middle]) (.cons v .nil))).trans
        (pack_assoc_heq (((a.append b₁).append b₂).append c) d (.cons v .nil))
    rw [pack_cast_eq_of_heq _ henv₂, ih₂, eval_castEnvTerm,
      pack_cast_eq_of_heq _ henv₂.symm, eval_lastOfFin _ _ _ rfl]

private theorem eval_transport {Γ Δ : List Shape} {s : Shape}
    (h : Γ = Δ) (env : TensorPack α Δ) (term : DAG.Term Γ s) :
    DAG.Term.eval env (cast (congrArg (fun shapes => DAG.Term shapes s) h) term) =
      DAG.Term.eval (TensorPack.cast h.symm env) term := by
  cases h
  rfl

omit [Context α] in
private theorem pack_append_nil_heq {ps : List Shape} (params : TensorPack α ps) :
    HEq (params.append (.nil : TensorPack α [])) params := by
  induction ps with
  | nil => cases params; rfl
  | cons s ps ih =>
    cases params with
    | cons value rest =>
      apply HEq.trans (show HEq (TensorPack.cons value (rest.append .nil))
        (TensorPack.cast (congrArg (List.cons s) (List.append_nil ps)).symm
          (.cons value rest)) from ?_)
      · exact cast_heq _ _
      · apply HEq.symm
        apply HEq.trans (heq_of_eq (pack_cast_cons (List.append_nil ps).symm value rest))
        have hr := pack_cast_eq_of_heq (List.append_nil ps).symm (ih rest).symm
        rw [hr]

/-- Converting any sequential chain to a DAG preserves its pure tensor semantics. The embedding
uses each primitive's existing `specFwd`; no claim about its executable `program` is needed. -/
theorem Chain.eval_toDAGTerm {ps : List Shape} {σ τ : Shape} (g : Chain ps σ τ)
    (params : TensorPack α ps) (input : Tensor α σ) :
    DAG.Term.eval (params.append (.cons input .nil)) (Chain.toDAGTerm g) =
      Interp.spec g params input := by
  unfold Chain.toDAGTerm
  simp only [eq_mp_eq_cast]
  rw [eval_transport (by simp)]
  have henv : TensorPack.cast (by simp) (params.append (.cons input .nil)) =
      (((TensorPack.nil.append params).append TensorPack.nil).append (.cons input .nil)) := by
    apply pack_cast_eq_of_heq
    exact pack_append_heq (by simp) rfl (pack_append_nil_heq params).symm HEq.rfl
  rw [henv, eval_toTerm]
  apply congrArg (Interp.spec g params)
  rw [eval_lastOfFin _ _ _ (by simp)]

end NN.GraphSpec.LowerToDAG
