/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.IR.Flow
public import Mathlib.Order.Filter.Finite

/-!
# Flows in a language that follows another

A flow can run in a second language `Ld` on states that stand for states of a first language
`L`: for tangents, `Ld`'s values carry derivatives and its blocks run forward-mode rules; for
roundoff, its values are rounded and carry error bounds. `Flow.eval` gives every flow node its rule
in `Ld`: `cond` runs the branch that `Ld`'s index selects, `scan n body` runs `body` `n` times, and
`whileLoop` runs `body` while `Ld`'s test holds. `Lang.dual L run` is the language whose blocks
run `run` on pairs of a value of `L` and an annotation, and whose indices and tests read the value,
so it takes the branches and runs the iterations that the values select.

The theorems relate the two languages along a filter `l` on parameters `θ`, through a relation
`D v w` between a value family `v` (a value for each `θ`) and a value `w` of `Ld`. A state family
tracks a state of `Ld` (`Tracks`) when, eventually along `l`, it is a list of value families
related one by one. `Ld` follows `L` (`Follows`) when, wherever a block succeeds in `Ld` on a
tracked state, it succeeds in `L` eventually along `l` with a family that tracks `Ld`'s result
(`Lifts`), and where `Ld` reads an index or a test, `L` reads the same one eventually along `l`.

Proved for every pair of languages, filter, relation, flow, fuel and state:
- `Follows.eval`: if `Ld` follows `L`, then wherever a flow succeeds in `Ld` on a tracked state,
  it succeeds in `L` eventually along `l` with a family that tracks `Ld`'s result;
- `Follows.cond`: `cond` in `Ld` runs the branch its index selects, and eventually along `l` the
  `cond` in `L` equals that branch;
- `Follows.scan`: eventually along `l`, `scan n body` in `L` equals `body` unrolled `n` times
  (`eval_scan`), with a result that `Ld`'s tracks;
- `Follows.whileLoop`: a `whileLoop` that succeeds in `Ld` is `body` unrolled `k` times there,
  for some `k ≤ fuel`, and eventually along `l` the `whileLoop` in `L` and `body` unrolled `k`
  times return one result, which tracks `Ld`'s (the statement does not say that `L`'s test fails
  after `k` iterations);
- `lifts_mapM`: a block that maps the state to a list of outputs follows its dual where each
  output does.

`Flow.Reals` has two instances: tangents (`Reals.dual_follows`) and roundoff
(`Reals.rounded_follows`).
-/

@[expose] public section

namespace TorchLean.IR.Flow

section Follows

variable {ι β V W : Type} {l : Filter ι} {D : (ι → V) → W → Prop}

/-- The state family `F` tracks the state `t` along `l` through `D`: eventually along `l`, `F` is a
list of value families, each related by `D` to the value of `t` at its position. -/
def Tracks (l : Filter ι) (D : (ι → V) → W → Prop) (F : ι → List V) (t : List W) : Prop :=
  ∃ vs : List (ι → V), (∀ᶠ θ in l, F θ = vs.map (· θ)) ∧ List.Forall₂ D vs t

/-- `g` follows `f` along `l` through `D`: wherever `g` succeeds on a state that a family `F`
tracks, `f` succeeds on `F θ` eventually along `l`, with a family that tracks `g`'s result. -/
def Lifts (l : Filter ι) (D : (ι → V) → W → Prop) (f : List V → Except String (List V))
    (g : List W → Except String (List W)) : Prop :=
  ∀ ⦃F t t'⦄, Tracks l D F t → g t = .ok t' →
    ∃ G, (∀ᶠ θ in l, f (F θ) = .ok (G θ)) ∧ Tracks l D G t'

/-- A family of states that are lists of value families tracks the state they are related to. -/
theorem tracks_map {vs : List (ι → V)} {t : List W} (h : List.Forall₂ D vs t) :
    Tracks l D (fun θ => vs.map (· θ)) t :=
  ⟨vs, .of_forall fun _ => rfl, h⟩

/-- `pure` follows `pure`. -/
theorem Lifts.pure : Lifts l D Pure.pure Pure.pure := by
  intro F t t' hF he
  obtain rfl : t = t' := Except.ok.inj he
  exact ⟨F, .of_forall fun _ => rfl, hF⟩

/-- `g` then `g'` follows `f` then `f'` where `g` follows `f` and `g'` follows `f'`. -/
theorem Lifts.bind {f f' : List V → Except String (List V)} {g g' : List W → Except String (List W)}
    (hf : Lifts l D f g) (hf' : Lifts l D f' g') :
    Lifts l D (fun s => f s >>= f') (fun t => g t >>= g') := by
  intro F t t' hF he
  obtain ⟨t₁, h₁, h₂⟩ := bind_eq_ok.1 he
  obtain ⟨G₁, hG₁, hT₁⟩ := hf hF h₁
  obtain ⟨G, hG, hT⟩ := hf' hT₁ h₂
  exact ⟨G, (hG₁.and hG).mono fun θ ⟨e₁, e₂⟩ => bind_eq_ok.2 ⟨_, e₁, e₂⟩, hT⟩

/-- `g` `n` times follows `f` `n` times where `g` follows `f`. -/
theorem Lifts.iterate {f : List V → Except String (List V)} {g : List W → Except String (List W)}
    (h : Lifts l D f g) : ∀ n, Lifts l D (Flow.iterate f n) (Flow.iterate g n)
  | 0 => by
      intro F t t' hF he
      simp only [Flow.iterate] at he ⊢
      exact Lifts.pure hF he
  | n + 1 => by
      intro F t t' hF he
      simp only [Flow.iterate] at he ⊢
      exact h.bind (Lifts.iterate h n) hF he

/-- A block that maps the state to the values of a list of outputs, by `ev` in `L` and `du` in
`Ld`, follows where each output does: where `du` succeeds on a state related to a list of value
families, `ev` succeeds on it eventually along `l` with a related family. -/
theorem lifts_mapM {O : Type} {ev : List V → O → Except String V}
    {du : List W → O → Except String W}
    (h : ∀ ⦃vs : List (ι → V)⦄ ⦃t : List W⦄, List.Forall₂ D vs t → ∀ ⦃o w⦄, du t o = .ok w →
      ∃ g, (∀ᶠ θ in l, ev (vs.map (· θ)) o = .ok (g θ)) ∧ D g w) (b : List O) :
    Lifts l D (fun s => b.mapM (ev s)) (fun t => b.mapM (du t)) := by
  intro F t t' hF he
  change b.mapM (du t) = .ok t' at he
  show ∃ G, (∀ᶠ θ in l, b.mapM (ev (F θ)) = .ok (G θ)) ∧ Tracks l D G t'
  obtain ⟨vs, hvs, hD⟩ := hF
  induction b generalizing t' with
  | nil =>
      simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at he
      subst he
      exact ⟨fun _ => [], .of_forall fun θ => by simp [pure, Except.pure], tracks_map .nil⟩
  | cons o b ih =>
      rw [List.mapM_cons] at he
      obtain ⟨w, hw, he⟩ := bind_eq_ok.1 he
      obtain ⟨ws, hws, he⟩ := bind_eq_ok.1 he
      simp only [pure, Except.pure, Except.ok.injEq] at he
      subst he
      obtain ⟨g, hg, hn⟩ := h hD hw
      obtain ⟨G, hG, gs, hgs, hT⟩ := ih hws
      refine ⟨fun θ => g θ :: gs.map (· θ), ((hvs.and hg).and (hG.and hgs)).mono ?_,
        tracks_map (.cons hn hT)⟩
      rintro θ ⟨⟨e₁, e₂⟩, e₃, e₄⟩
      rw [e₁, List.mapM_cons, e₂]
      rw [e₁, e₄] at e₃
      simp [e₃, bind, Except.bind, pure, Except.pure]

/-- The dual language `Ld` follows `L` along `l` through `D`: where the dual of a block succeeds on
a tracked state, the block follows it (`Lifts`), and where `Ld` reads an index or a test of a dual
value, `L` reads the same one of a related value family eventually along `l`. -/
structure Follows (L : Lang (Except String) β V) (Ld : Lang (Except String) β W) (l : Filter ι)
    (D : (ι → V) → W → Prop) : Prop where
  /-- Each block's dual follows the block. -/
  run : ∀ b, Lifts l D (L.run b) (Ld.run b)
  /-- An index of a dual value is the index of a related family eventually. -/
  index : ∀ ⦃v w i⦄, D v w → Ld.index w = .ok i → ∀ᶠ θ in l, L.index (v θ) = .ok i
  /-- A test of a dual value is the test of a related family eventually. -/
  test : ∀ ⦃v w c⦄, D v w → Ld.test w = .ok c → ∀ᶠ θ in l, L.test (v θ) = .ok c

variable {L : Lang (Except String) β V} {Ld : Lang (Except String) β W}

/-- Where a loop test's dual holds or fails on a tracked state, the test holds or fails on the
family eventually. -/
theorem Follows.holds (h : Follows L Ld l D) {fuel : Nat} {test : Flow β}
    (htest : Lifts l D (eval L fuel test) (eval Ld fuel test)) {F : ι → List V} {t : List W}
    {c : Bool} (hF : Tracks l D F t) (he : Flow.holds Ld fuel test t = .ok c) :
    ∀ᶠ θ in l, Flow.holds L fuel test (F θ) = .ok c := by
  obtain ⟨u, hu, hc⟩ := bind_eq_ok.1 he
  obtain ⟨G, hG, vs, hvs, hD⟩ := htest hF hu
  rcases u with _ | ⟨w, _ | ⟨w', u⟩⟩
  · simp [holdsOf, throw, throwThe, MonadExceptOf.throw] at hc
  · rcases hD with _ | ⟨hv, _ | _⟩
    have hc : Ld.test w = .ok c := by simpa [holdsOf] using hc
    refine ((hG.and hvs).and (h.test hv hc)).mono fun θ ⟨⟨e₁, e₂⟩, e₃⟩ => ?_
    exact bind_eq_ok.2 ⟨_, e₁, by simpa [e₂, holdsOf] using e₃⟩
  · simp [holdsOf, throw, throwThe, MonadExceptOf.throw] at hc

/-- Where the dual of a `whileLoop` succeeds on a tracked state, for test and body flows whose duals
follow them, it is the dual of `body` unrolled `k` times for some `k ≤ fuel`, and eventually along
`l` the `whileLoop` and the unrolled flow return one family that tracks the dual's result. -/
theorem Follows.loop (h : Follows L Ld l D) {fuel : Nat} {test body : Flow β}
    (htest : Lifts l D (eval L fuel test) (eval Ld fuel test))
    (hbody : Lifts l D (eval L fuel body) (eval Ld fuel body)) {F : ι → List V} {t t' : List W}
    (hF : Tracks l D F t) (he : eval Ld fuel (.whileLoop test body) t = .ok t') :
    ∃ k ≤ fuel, eval Ld fuel (body.unroll k) t = .ok t' ∧ ∃ G,
      (∀ᶠ θ in l, eval L fuel (.whileLoop test body) (F θ) = .ok (G θ) ∧
        eval L fuel (body.unroll k) (F θ) = .ok (G θ)) ∧ Tracks l D G t' := by
  obtain ⟨k, hk, ht, hf, hu⟩ := eval_whileLoop_ok Ld he
  have unroll : ∀ j, Lifts l D (eval L fuel (body.unroll j)) (eval Ld fuel (body.unroll j)) := by
    intro j
    rw [show eval L fuel (body.unroll j) = Flow.iterate (eval L fuel body) j from
        funext fun s => (iterate_eval L fuel body j s).symm,
      show eval Ld fuel (body.unroll j) = Flow.iterate (eval Ld fuel body) j from
        funext fun s => (iterate_eval Ld fuel body j s).symm]
    exact hbody.iterate j
  have after : ∀ j c, (eval Ld fuel (body.unroll j) t >>= Flow.holds Ld fuel test) = .ok c →
      ∀ᶠ θ in l, (eval L fuel (body.unroll j) (F θ) >>= Flow.holds L fuel test) = .ok c := by
    intro j c hj
    obtain ⟨u, hu, hc⟩ := bind_eq_ok.1 hj
    obtain ⟨G, hG, hT⟩ := unroll j hF hu
    exact (hG.and (h.holds htest hT hc)).mono fun θ ⟨e₁, e₂⟩ => bind_eq_ok.2 ⟨_, e₁, e₂⟩
  have before : ∀ᶠ θ in l, ∀ j : Fin k,
      (eval L fuel (body.unroll j) (F θ) >>= Flow.holds L fuel test) = .ok true :=
    Filter.eventually_all.2 fun j => after j true (ht j j.2)
  obtain ⟨G, hG, hT⟩ := unroll k hF hu
  refine ⟨k, hk, hu, G, ((before.and (after k false hf)).and hG).mono ?_, hT⟩
  rintro θ ⟨⟨e₁, e₂⟩, e₃⟩
  exact ⟨(eval_whileLoop L hk (fun j hj => e₁ ⟨j, hj⟩) e₂).trans e₃, e₃⟩

mutual

/-- If `Ld` follows `L` along `l`, the dual of every flow follows the flow. -/
theorem Follows.eval (h : Follows L Ld l D) (fuel : Nat) :
    ∀ f : Flow β, Lifts l D (Flow.eval L fuel f) (Flow.eval Ld fuel f)
  | .block b => by
      intro F t t' hF he
      simp only [Flow.eval] at he ⊢
      exact h.run b hF he
  | .skip => by
      intro F t t' hF he
      simp only [Flow.eval] at he ⊢
      exact Lifts.pure hF he
  | .seq f g => by
      intro F t t' hF he
      simp only [Flow.eval] at he ⊢
      exact (h.eval fuel f).bind (h.eval fuel g) hF he
  | .cond fs => by
      intro F t t' hF he
      rcases t with _ | ⟨w, t⟩
      · simp [Flow.eval, throw, throwThe, MonadExceptOf.throw] at he
      simp only [Flow.eval] at he
      obtain ⟨i, hi, hb⟩ := bind_eq_ok.1 he
      obtain ⟨vs, hvs, hD⟩ := hF
      rcases hD with _ | ⟨hv, hD⟩
      obtain ⟨G, hG, hT⟩ := h.evalBranch fuel fs _ (tracks_map hD) hb
      refine ⟨G, ((hvs.and (h.index hv hi)).and hG).mono fun θ ⟨⟨e₁, e₂⟩, e₃⟩ => ?_, hT⟩
      rw [e₁]
      simpa [Flow.eval, e₂, bind, Except.bind] using e₃
  | .scan n body => by
      intro F t t' hF he
      simp only [Flow.eval] at he ⊢
      exact (h.eval fuel body).iterate n hF he
  | .whileLoop test body => by
      intro F t t' hF he
      obtain ⟨_, _, _, G, hG, hT⟩ := h.loop (h.eval fuel test) (h.eval fuel body) hF he
      exact ⟨G, hG.mono fun _ e => e.1, hT⟩

/-- If `Ld` follows `L` along `l`, the dual of every branch of a list of flows follows the
branch. -/
theorem Follows.evalBranch (h : Follows L Ld l D) (fuel : Nat) :
    ∀ (fs : List (Flow β)) (k : Nat),
      Lifts l D (Flow.evalBranch L fuel fs k) (Flow.evalBranch Ld fuel fs k)
  | [], _ => by
      intro F t t' hF he
      simp [Flow.evalBranch, throw, throwThe, MonadExceptOf.throw] at he
  | f :: _, 0 => by
      intro F t t' hF he
      simp only [Flow.evalBranch] at he ⊢
      exact h.eval fuel f hF he
  | _ :: fs, k + 1 => by
      intro F t t' hF he
      simp only [Flow.evalBranch] at he ⊢
      exact h.evalBranch fuel fs k hF he

end

/-- Where the dual of `cond fs` succeeds on a tracked state `w :: t`, the dual index `i` selects
branch `f` of `fs`, the dual of `cond` is the dual of `f` on `t`, and eventually along `l` the
`cond` and `f` on the rest of the state return one family that tracks the dual's result. -/
theorem Follows.cond (h : Follows L Ld l D) {fuel : Nat} {fs : List (Flow β)} {F : ι → List V}
    {w : W} {t t' : List W} (hF : Tracks l D F (w :: t))
    (he : Flow.eval Ld fuel (.cond fs) (w :: t) = .ok t') :
    ∃ i f, Ld.index w = .ok i ∧ fs[branchOf fs.length i]? = some f ∧
      Flow.eval Ld fuel f t = .ok t' ∧ ∃ G,
        (∀ᶠ θ in l, Flow.eval L fuel (.cond fs) (F θ) = .ok (G θ) ∧
          Flow.eval L fuel f (F θ).tail = .ok (G θ)) ∧ Tracks l D G t' := by
  simp only [Flow.eval] at he
  obtain ⟨i, hi, hb⟩ := bind_eq_ok.1 he
  rw [evalBranch_eq] at hb
  obtain ⟨vs, hvs, hD⟩ := hF
  rcases hD with _ | ⟨hv, hD⟩
  cases hf : fs[branchOf fs.length i]? with
  | none => simp [hf, throw, throwThe, MonadExceptOf.throw] at hb
  | some f =>
      rw [hf] at hb
      obtain ⟨G, hG, hT⟩ := h.eval fuel f (tracks_map hD) hb
      refine ⟨i, f, hi, hf, hb, G, ((hvs.and (h.index hv hi)).and hG).mono ?_, hT⟩
      rintro θ ⟨⟨e₁, e₂⟩, e₃⟩
      rw [e₁]
      refine ⟨?_, e₃⟩
      simpa [Flow.eval, e₂, bind, Except.bind, evalBranch_eq, hf] using e₃

/-- Where the dual of `scan n body` succeeds on a tracked state, eventually along `l` the scan and
`body` unrolled `n` times return one family that tracks the dual's result. -/
theorem Follows.scan (h : Follows L Ld l D) {fuel n : Nat} {body : Flow β} {F : ι → List V}
    {t t' : List W} (hF : Tracks l D F t) (he : Flow.eval Ld fuel (.scan n body) t = .ok t') :
    ∃ G, (∀ᶠ θ in l, Flow.eval L fuel (.scan n body) (F θ) = .ok (G θ) ∧
      Flow.eval L fuel (body.unroll n) (F θ) = .ok (G θ)) ∧ Tracks l D G t' := by
  obtain ⟨G, hG, hT⟩ := h.eval fuel _ hF he
  exact ⟨G, hG.mono fun θ e => ⟨e, (eval_scan L fuel n body _).symm.trans e⟩, hT⟩

/-- Where the dual of a `whileLoop` succeeds on a tracked state, it is the dual of `body` unrolled
`k` times for some `k ≤ fuel`, and eventually along `l` the `whileLoop` and the unrolled flow
return one family that tracks the dual's result. -/
theorem Follows.whileLoop (h : Follows L Ld l D) {fuel : Nat} {test body : Flow β}
    {F : ι → List V} {t t' : List W} (hF : Tracks l D F t)
    (he : Flow.eval Ld fuel (.whileLoop test body) t = .ok t') :
    ∃ k ≤ fuel, Flow.eval Ld fuel (body.unroll k) t = .ok t' ∧ ∃ G,
      (∀ᶠ θ in l, Flow.eval L fuel (.whileLoop test body) (F θ) = .ok (G θ) ∧
        Flow.eval L fuel (body.unroll k) (F θ) = .ok (G θ)) ∧ Tracks l D G t' :=
  h.loop (h.eval fuel test) (h.eval fuel body) hF he

end Follows

/-- The dual of the language `L` with annotations `T`: a value is a pair of a value of `L` and an
annotation, such as a tangent or an error bound, a block runs `run`, and an index or a test reads
the value and ignores the annotation. -/
def Lang.dual {m : Type → Type} {β V T : Type} (L : Lang m β V)
    (run : β → List (V × T) → m (List (V × T))) : Lang m β (V × T) where
  run := run
  index w := L.index w.1
  test w := L.test w.1

end TorchLean.IR.Flow
