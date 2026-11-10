/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.Op.CertSoundness
public import TorchLean.Verify.CROWN.Op.Certify

/-!
# Interval bound propagation at ℝ and the certificate

At `ℝ`, on a graph without tanh, sigmoid, sine and cosine (`NoBackendEnclosure`), the engine's
table has every box of the certificate (`runIBP_of_certify`). The engine's rules for those four
(`BackendEnclosure`) are the backend's enclosures, while the certificate rule `cert?` keeps
endpoint min-max and clipped midpoints. For every other operator `cert?` is the shared rule
`shared?` (`cert?_eq_shared?`, in `TorchLean.Verify.CROWN.Op.Cert`), and `shared?` keeps each box
it gives on every table that extends the one it read (`shared?_mono`, `Extends`). On a graph
without those four, each box of the certificate is the engine's box at its node and encloses the
node's flat value in every successful real run whose inputs the seed boxes enclose
(`runIBP_encloses_of_run`).

The engine's table is in general not a certificate (`CertLocalOK` is equality with the certificate
step at every node): it has boxes where the certificate has none, at reshape, softmax or layer
norm and their consumers. On a graph whose every operator has kind `shared` or `none`
(`SharedOnly`), the two tables are equal (`runIBP_eq_certify`), so on a sorted such graph the
engine's table is a certificate (`runIBP_localOK`). Apart from `shared?_mono`, which holds at any
scalar, nothing here is about a rounded scalar.
-/

@[expose] public section

namespace TorchLean.CROWN.Op


noncomputable section

/-- No operator node of the graph is tanh, sigmoid, sine or cosine. -/
def NoBackendEnclosure (G : IR.Graph) : Prop :=
  ∀ (id : Nat) (o : IR.Operator) (refs : Array IR.Ref),
    G.nodes[id]? = some (.op o refs) → ¬ BackendEnclosure o

/-- `c'` has every box of `c`, at the same node. -/
def Extends {β : Type} (c c' : Array (Option β)) : Prop :=
  ∀ (j : Nat) (x : β), c[j]? = some (some x) → c'[j]? = some (some x)

/-- An extension `c'` of `c` gives a reference `r` the box that `c` gives it. -/
theorem Extends.box?_eq_some {β : Type} {c c' : Array (Option β)} (h : Extends c c') {r : IR.Ref}
    {x : β} (hr : box? c r = some x) : box? c' r = some x := by
  unfold box? at hr ⊢
  split at hr
  · rename_i h0
    simp only [h0, ↓reduceIte]
    cases hc : c[r.node]? with
    | none => simp [hc] at hr
    | some y =>
      simp only [hc, Option.join_some] at hr
      subst hr
      simp [h r.node _ hc]
  · cases hr

/-- Reading the boxes of a list of references from an extending table keeps every read box. -/
theorem Extends.mapM_box?_eq_some {β : Type} {c c' : Array (Option β)} (h : Extends c c') :
    ∀ {refs : List IR.Ref} {xs : List β}, refs.mapM (box? c) = some xs →
      refs.mapM (box? c') = some xs
  | [], xs, hx => hx
  | r :: rs, xs, hx => by
      simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff,
        Option.pure_def, Option.some.injEq] at hx ⊢
      obtain ⟨x, hr, ys, hys, rfl⟩ := hx
      exact ⟨x, h.box?_eq_some hr, ys, h.mapM_box?_eq_some hys, rfl⟩

/-- Wherever `shared?` gives operator `o` a box on the table `c`, it gives the same box on every
table that extends `c`. -/
theorem shared?_mono {α : Type} [Storage α] [Context α] [BoundOps α] [NonlinearBoundOps α]
    {G : IR.Graph} {P : IR.Payload α} {c c' : Array (Option (FlatBox α))}
    (hc : Extends c c') {o : IR.Operator} {refs : List IR.Ref} {B : FlatBox α}
    (h : shared? G P c o refs = some B) : shared? G P c' o refs = some B := by
  unfold shared? at h
  split at h
  case h_13 | h_14 | h_15 => cases h
  case h_16 =>
    unfold catBox? at h
    split at h
    · simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at h
      obtain ⟨xs, hxs, hB⟩ := h
      simp only [shared?, catBox?, Option.bind_eq_bind, Option.bind_eq_some_iff]
      exact ⟨xs, hc.mapM_box?_eq_some hxs, hB⟩
    all_goals cases h
  case h_1 =>
    simp only [shared?]
    exact hc.box?_eq_some h
  case h_2 =>
    split at h
    · rename_i h₁ h₂
      simp only [shared?, hc.box?_eq_some h₁, hc.box?_eq_some h₂]
      exact h
    · cases h
  case h_3 =>
    split at h
    · rename_i h₁ h₂
      simp only [shared?, hc.box?_eq_some h₁, hc.box?_eq_some h₂]
      exact h
    · cases h
  case h_4 =>
    split at h
    · rename_i h₁ h₂
      simp only [shared?, hc.box?_eq_some h₁, hc.box?_eq_some h₂]
      exact h
    · cases h
  case h_5 =>
    simp only [Option.map_eq_some_iff] at h
    obtain ⟨X, hX, rfl⟩ := h
    simp only [shared?, hc.box?_eq_some hX, Option.map_some]
  case h_6 =>
    simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at h
    obtain ⟨X, hX, hB⟩ := h
    simp only [shared?, hc.box?_eq_some hX]
    exact hB
  case h_7 =>
    simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at h
    obtain ⟨X, hX, hB⟩ := h
    simp only [shared?, hc.box?_eq_some hX]
    exact hB
  case h_8 =>
    split at h
    · rename_i h₁ h₂
      simp only [shared?, hc.box?_eq_some h₁, hc.box?_eq_some h₂]
      exact h
    · cases h
  case h_9 =>
    simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at h
    obtain ⟨W, hW, b, hb, X, hX, hB⟩ := h
    simp only [shared?, Option.bind_eq_bind, Option.bind_eq_some_iff]
    exact ⟨W, hW, b, hb, X, hc.box?_eq_some hX, hB⟩
  case h_10 =>
    split at h
    · rename_i h₁ h₂
      simp only [shared?, hc.box?_eq_some h₁, hc.box?_eq_some h₂]
      exact h
    · cases h
  case h_11 =>
    split at h
    · rename_i h₁ h₂ h₃
      simp only [shared?, hc.box?_eq_some h₁, hc.box?_eq_some h₂, hc.box?_eq_some h₃]
      exact h
    · cases h
  case h_12 =>
    simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at h
    obtain ⟨W, hW, b, hb, X, hX, hB⟩ := h
    simp only [shared?, Option.bind_eq_bind, Option.bind_eq_some_iff]
    exact ⟨W, hW, b, hb, X, hc.box?_eq_some hX, hB⟩

variable {G : IR.Graph} {P : IR.Payload ℝ}

/-- Wherever the certificate step gives node `id` a box, the engine's step at ℝ gives the same box
on every table that extends the table the certificate step reads, on a graph without tanh,
sigmoid, sine and cosine. -/
theorem step?_ibp?_of_step?_cert? {seeds : Nat → Option (FlatBox ℝ)}
    (hG : NoBackendEnclosure G)
    {c c' : Array (Option (FlatBox ℝ))} (hc : Extends c c') {id : Nat} {B : FlatBox ℝ}
    (h : step? G P (cert? G P) seeds c id = some B) :
    step? G P (ibp? (α := ℝ) G P) seeds c' id = some B := by
  unfold step? at h ⊢
  obtain ⟨h, hp⟩ := Option.filter_eq_some_iff.mp h
  refine Option.filter_eq_some_iff.mpr ⟨?_, hp⟩
  cases hn : G.nodes[id]? with
  | none => simp [hn] at h
  | some n =>
    rw [hn] at h
    cases n with
    | input _ _ => exact h
    | const _ => exact h
    | op o refs =>
      simp only at h ⊢
      split at h
      · rename_i hall
        rw [cert?_eq_shared? (hG id o refs hn)] at h
        simp only [hall, ↓reduceIte, ibp?, shared?_mono hc h]
      · cases h

/-- Tables computed node by node: when each step of `g` gives every box the step of `f` gives, on
every table that extends the one `f` read, the table of `g` has every box of the table of `f`. -/
theorem tabulate_extends {β : Type} {f g : Array (Option β) → Nat → Option β}
    (hstep : ∀ (c c' : Array (Option β)) (id : Nat) (x : β), Extends c c' → f c id = some x →
      g c' id = some x) :
    ∀ k, Extends (tabulate f k) (tabulate g k)
  | 0 => fun j x h => by simp [tabulate] at h
  | k + 1 => fun j x h => by
      have ih := tabulate_extends hstep k
      by_cases hj : j < k
      · rw [tabulate_getElem?_of_lt hj (Nat.le_succ k)] at h ⊢
        exact ih j x h
      · by_cases hjk : j = k
        · subst hjk
          rw [tabulate_getElem?_self (Nat.lt_succ_self j), Option.some.injEq] at h ⊢
          exact hstep _ _ j x ih h
        · rw [Array.getElem?_eq_none (by rw [size_tabulate]; omega)] at h
          cases h

/-- At ℝ, on a graph without tanh, sigmoid, sine and cosine, the engine computes the certificate's
box wherever the certificate has one. -/
theorem runIBP_of_certify (seeds : Nat → Option (FlatBox ℝ))
    (hG : NoBackendEnclosure G) :
    Extends (certify G P seeds) (runIBP (α := ℝ) G P seeds) := by
  unfold certify runIBP
  exact tabulate_extends (f := step? G P (cert? G P) seeds)
    (g := step? G P (ibp? (α := ℝ) G P) seeds)
    (fun _ _ id _ hc h => step?_ibp?_of_step?_cert? (id := id) hG hc h) G.nodes.size

/-- At ℝ, on a graph without tanh, sigmoid, sine and cosine, every box `B` of the certificate is
the engine's box at its node. `B` encloses the node's flat value in every successful real run
whose inputs the seed boxes enclose. -/
theorem runIBP_encloses_of_run (seeds : Nat → Option (FlatBox ℝ))
    (inputs : List (Op.Value ℝ)) (values : Array (List (Op.Value ℝ)))
    (hG : NoBackendEnclosure G)
    (hin : ∀ id k p, G.nodes[id]? = some (.input k p) → ∀ B, seeds id = some B →
      ∀ x y v, inputs[k]? = some x → p.read x = .ok y → flatOf ⟨p, y⟩ = some v →
        Directed.Encloses B v)
    (hrun : G.run P inputs = .ok values) :
    ∀ id, id < G.nodes.size → ∀ (B : FlatBox ℝ), (certify G P seeds)[id]! = some B →
      (runIBP (α := ℝ) G P seeds)[id]! = some B ∧
        ∀ v, runVal values id = some v → Directed.Encloses B v := by
  intro id hid B hB
  have hc : (certify G P seeds)[id]? = some (some B) := by
    rw [Array.getElem?_eq_getElem (by simp [certify, size_tabulate, hid])]
    rw [getElem!_pos _ id (by simp [certify, size_tabulate, hid])] at hB
    rw [hB]
  have he := runIBP_of_certify seeds hG id B hc
  have hB' : (runIBP (α := ℝ) G P seeds)[id]! = some B := by
    rw [getElem!_pos _ id (by simp [runIBP, size_tabulate, hid])]
    rw [Array.getElem?_eq_getElem (by simp [runIBP, size_tabulate, hid])] at he
    exact Option.some.inj he
  exact ⟨hB', fun v hv => Directed.encloses_of_step G P P cert?_encloses seeds inputs values
    (Directed.realConsts_self P) hin hrun (certify G P seeds)
    (fun _ hj => by unfold certify; exact ⟨_, tabulate_getElem! hj⟩) id hid B v hB hv⟩

/-- The operators of kind `shared` or `none` in `ruleKind`, at which the engine's rule is the
certificate rule `cert?` (`ibp?_eq_cert?`). At kind `none` neither rule gives a box. -/
def SharedRule (o : IR.Operator) : Prop :=
  ruleKind o = .shared ∨ ruleKind o = .none

/-- At every operator node of the graph the engine's rule is the certificate rule (`SharedRule`). -/
def SharedOnly (G : IR.Graph) : Prop :=
  ∀ (id : Nat) (o : IR.Operator) (refs : Array IR.Ref),
    G.nodes[id]? = some (.op o refs) → SharedRule o

/-- Every table extends itself. -/
theorem Extends.refl {β : Type} (c : Array (Option β)) : Extends c c := fun _ _ h => h

/-- An operator of kind `shared` or `none` is not of kind `backend` (`BackendEnclosure`). -/
theorem SharedRule.not_backendEnclosure {o : IR.Operator} (h : SharedRule o) :
    ¬ BackendEnclosure o := by
  rcases h with h | h <;> simp only [BackendEnclosure, h, reduceCtorEq, not_false_eq_true]

/-- The engine's other rules give no box to an operator of kind `shared` or `none`. -/
theorem directed?_of_sharedRule {c : Array (Option (FlatBox ℝ))} {o : IR.Operator}
    (refs : List IR.Ref) (h : SharedRule o) : directed? (α := ℝ) G P c o refs = none := by
  rcases h with h | h
  · rcases o with ⟨_, _, op⟩
    cases op <;> (try cases ‹Op.NumericElement›) <;> (try cases ‹Op.Element›) <;>
      simp only [ruleKind.eq_def, ite_eq_iff, reduceCtorEq, and_false, or_self] at h <;>
      rcases refs with _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _⟩⟩⟩⟩ <;> rfl
  · -- At kind `none` the engine gives no box, so neither does its directed rule.
    have hn : ibp? (α := ℝ) G P c o refs = none :=
      Option.eq_none_iff_forall_ne_some.mpr fun B hB => Directed.supported_of_ibp? hB h
    unfold ibp? at hn
    split at hn
    · cases hn
    · exact hn

/-- At ℝ the engine's rule for an operator of kind `shared` or `none` is the certificate rule. -/
theorem ibp?_eq_cert? {c : Array (Option (FlatBox ℝ))} {o : IR.Operator}
    {refs : List IR.Ref} (h : SharedRule o) :
    ibp? (α := ℝ) G P c o refs = cert? G P c o refs := by
  rw [cert?_eq_shared? (SharedRule.not_backendEnclosure h)]
  unfold ibp?
  cases shared? (α := ℝ) G P c o refs with
  | some B => rfl
  | none => exact directed?_of_sharedRule refs h

/-- On a graph whose every operator has kind `shared` or `none` (`SharedOnly`), the engine's step
at ℝ is the certificate step. -/
theorem step?_ibp?_eq_step?_cert? {seeds : Nat → Option (FlatBox ℝ)} (hG : SharedOnly G)
    (c : Array (Option (FlatBox ℝ))) (id : Nat) :
    step? G P (ibp? (α := ℝ) G P) seeds c id = step? G P (cert? G P) seeds c id := by
  unfold step?
  refine congrArg (Option.filter _) ?_
  cases hn : G.nodes[id]? with
  | none => rfl
  | some n =>
    cases n with
    | input _ _ => rfl
    | const _ => rfl
    | op o refs =>
      simp only
      split
      · exact ibp?_eq_cert? (hG id o refs hn)
      · rfl

/-- On a graph whose every operator has kind `shared` or `none` (`SharedOnly`), the engine's table
at ℝ is the certificate. -/
theorem runIBP_eq_certify (seeds : Nat → Option (FlatBox ℝ)) (hG : SharedOnly G) :
    runIBP (α := ℝ) G P seeds = certify G P seeds := by
  unfold runIBP certify
  congr 1
  funext c id
  exact step?_ibp?_eq_step?_cert? hG c id

/-- On a sorted graph whose every operator has kind `shared` or `none` (`SharedOnly`), the engine's
table at ℝ is a certificate (`CertLocalOK`). -/
theorem runIBP_localOK (seeds : Nat → Option (FlatBox ℝ)) (htopo : TopoSorted G)
    (hG : SharedOnly G) : CertLocalOK G P seeds (runIBP (α := ℝ) G P seeds) := by
  rw [runIBP_eq_certify seeds hG]
  exact certify_localOK htopo

end

end TorchLean.CROWN.Op
