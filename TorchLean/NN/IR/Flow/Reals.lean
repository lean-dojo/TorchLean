/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.IR.Flow.Follows
public import Mathlib.Analysis.Calculus.FDeriv.Mul
public import Mathlib.Topology.Order.OrderClosed

/-!
# Flows of real arithmetic: tangents and roundoff

`Flow.Reals.lang` is a language of real arithmetic. A value is a real or a Boolean, and a block maps
the state to a list of outputs, each a real term (the real at a position, a constant, a sum, a
product or a negation) or a comparison `a < b` of two terms. A branch index and a loop test are
Booleans. `follows_of` proves that a dual of `lang` that maps the state to the duals of the outputs
follows `lang` (`Flow.Follows`) wherever each output's dual follows its value, so `Follows.eval`,
`cond`, `scan` and `whileLoop` apply to its two instances.

Tangents: `dual E` runs each term with a derivative in `E →L[ℝ] ℝ` by the sum, product and
negation rules, and refuses a comparison of two equal reals. `dual_follows θ₀` proves, for every
real normed space `E` and point `θ₀`, that `dual E` follows `lang` near `θ₀` through `Near θ₀`. So
for every flow and fuel, wherever the dual's run succeeds on values with derivatives at `θ₀`, the
flow near `θ₀` takes the dual's branches, each of its loops returns what its body unrolled as often
as the dual's loop runs returns, and it returns reals with the dual's values and derivatives at
`θ₀`: the derivative of the selected branch for `cond`, and of the body unrolled for `scan` and a
terminating `whileLoop`.

Roundoff: `rounded rnd err` runs each term on computed values with error bounds. A constant, a sum
and a product are rounded by `rnd`, and their bound adds the rounding error bound `err` of the exact
result to the error that the operands' bounds propagate to it. A negation is exact, and a comparison
is refused where its operands are within their bounds of each other. `rounded_follows` proves, for
every `rnd` and `err` with `|rnd z - z| ≤ err z` for every `z`, and every type `ι`, that
`rounded rnd err` follows `lang` along `⊤` through `Within`. So for every flow and fuel, wherever
the rounded run succeeds, every family of exact states within its input bounds takes the rounded
run's branches, each loop on it returns what its body unrolled as often as the rounded loop runs
returns, and it returns reals within its result bounds: per branch for `cond`, and per iteration
for `scan` and a terminating `whileLoop`, whose rounded run is that of the body unrolled, its
bounds growing with each iteration. FloatLib's `roundAt fmt` is such an `rnd`, with
`err z = |roundAt fmt z - z|` or the standard model's `unitRoundoffAt fmt * |z| + underflowErrorAt
fmt` (`abs_roundAt_sub_le_unitRoundoffAt`). It rounds to nearest-even on the format's grid with no
largest exponent, as IEEE 754-2019 rounds an operation's exact result where it does not overflow.
-/

@[expose] public section

open Topology

namespace TorchLean.IR.Flow.Reals

/-- A real term over a state of reals and Booleans: the real at a position of the state, a
constant, a sum, a product or a negation. -/
inductive Term where
  /-- The real at position `i`. -/
  | var (i : Nat)
  /-- The constant `c`. -/
  | const (c : ℝ)
  /-- `a + b`. -/
  | add (a b : Term)
  /-- `a * b`. -/
  | mul (a b : Term)
  /-- `-a`. -/
  | neg (a : Term)

/-- An output of a block: a real term, or whether one real term is less than another. -/
inductive Out where
  /-- The real `e`. -/
  | real (e : Term)
  /-- The Boolean `a < b`. -/
  | lt (a b : Term)

noncomputable section

/-- The value of a term on a state, or an error where it reads a position that holds no real. -/
def Term.eval (s : List (ℝ ⊕ Bool)) : Term → Except String ℝ
  | .var i => match s[i]? with
    | some (.inl x) => pure x
    | _ => throw s!"no real at {i}"
  | .const c => pure c
  | .add a b => do pure ((← a.eval s) + (← b.eval s))
  | .mul a b => do pure ((← a.eval s) * (← b.eval s))
  | .neg a => do pure (-(← a.eval s))

/-- The value of an output on a state. -/
def Out.eval (s : List (ℝ ⊕ Bool)) : Out → Except String (ℝ ⊕ Bool)
  | .real e => do pure (.inl (← e.eval s))
  | .lt a b => do pure (.inr (decide ((← a.eval s) < (← b.eval s))))

/-- The language: a block maps the state to the values of its outputs, a branch index is a Boolean
(`false` 0, `true` 1), as `lax.cond` reads it, and a loop test is a Boolean. -/
def lang : Lang (Except String) (List Out) (ℝ ⊕ Bool) where
  run b s := b.mapM (Out.eval s)
  index
    | .inr c => pure (if c then 1 else 0)
    | .inl _ => throw "cond: a real index"
  test
    | .inr c => pure c
    | .inl _ => throw "while: a real test"

/-- A dual of `lang` that maps the state to the duals `du` of the outputs follows `lang` along `l`
through `D` where each output's dual follows its value and `D` relates a Boolean only to a family
that is that Boolean eventually along `l`. -/
theorem follows_of {ι T : Type} {l : Filter ι} {D : (ι → ℝ ⊕ Bool) → (ℝ ⊕ Bool) × T → Prop}
    {du : List ((ℝ ⊕ Bool) × T) → Out → Except String ((ℝ ⊕ Bool) × T)}
    (hout : ∀ ⦃vs : List (ι → ℝ ⊕ Bool)⦄ ⦃t⦄, List.Forall₂ D vs t → ∀ ⦃o w⦄, du t o = .ok w →
      ∃ g, (∀ᶠ θ in l, o.eval (vs.map (· θ)) = .ok (g θ)) ∧ D g w)
    (hbool : ∀ ⦃v c a⦄, D v (.inr c, a) → ∀ᶠ θ in l, v θ = .inr c) :
    Follows lang (lang.dual fun b t => b.mapM (du t)) l D where
  run := lifts_mapM hout
  index v w i hv hi := by
    rcases w with ⟨_ | c, a⟩
    · simp [Lang.dual, lang, throw, throwThe, MonadExceptOf.throw] at hi
    · exact (hbool hv).mono fun θ e => by simpa [e, Lang.dual, lang] using hi
  test v w c hv hc := by
    rcases w with ⟨_ | c', a⟩
    · simp [Lang.dual, lang, throw, throwThe, MonadExceptOf.throw] at hc
    · exact (hbool hv).mono fun θ e => by simpa [e, Lang.dual, lang] using hc

section Tangents

variable {E : Type} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- The value and derivative of a term on a state of values and derivatives, by the sum and
product rules. -/
def Term.dual (t : List ((ℝ ⊕ Bool) × (E →L[ℝ] ℝ))) : Term → Except String (ℝ × (E →L[ℝ] ℝ))
  | .var i => match t[i]? with
    | some (.inl x, d) => pure (x, d)
    | _ => throw s!"no real at {i}"
  | .const c => pure (c, 0)
  | .add a b => do
      let p ← a.dual t
      let q ← b.dual t
      pure (p.1 + q.1, p.2 + q.2)
  | .mul a b => do
      let p ← a.dual t
      let q ← b.dual t
      pure (p.1 * q.1, p.1 • q.2 + q.1 • p.2)
  | .neg a => do
      let p ← a.dual t
      pure (-p.1, -p.2)

/-- The dual of an output: a real term's value and derivative, or a comparison with derivative
zero, which fails where its two reals are equal. -/
def Out.dual (t : List ((ℝ ⊕ Bool) × (E →L[ℝ] ℝ))) : Out → Except String ((ℝ ⊕ Bool) × (E →L[ℝ] ℝ))
  | .real e => do
      let p ← e.dual t
      pure (.inl p.1, p.2)
  | .lt a b => do
      let p ← a.dual t
      let q ← b.dual t
      if p.1 = q.1 then throw "lt: equal operands" else pure (.inr (decide (p.1 < q.1)), 0)

variable (E) in
/-- The dual of `lang`, with derivatives in `E →L[ℝ] ℝ`: a block maps the state to the duals of
its outputs. -/
def dual : Lang (Except String) (List Out) ((ℝ ⊕ Bool) × (E →L[ℝ] ℝ)) :=
  lang.dual fun b t => b.mapM (Out.dual t)

/-- The value family `v` near `θ₀` is the dual value `w`: for `w = (x, d)` with `x` real, `v` is a
real family with value `x` and derivative `d` at `θ₀`, and for `w` a Boolean `c`, `v` is `c` near
`θ₀`. -/
def Near (θ₀ : E) (v : E → ℝ ⊕ Bool) : (ℝ ⊕ Bool) × (E →L[ℝ] ℝ) → Prop
  | (.inl x, d) => ∃ f : E → ℝ, (∀ᶠ θ in 𝓝 θ₀, v θ = .inl (f θ)) ∧ f θ₀ = x ∧ HasFDerivAt f d θ₀
  | (.inr c, _) => ∀ᶠ θ in 𝓝 θ₀, v θ = .inr c

variable {θ₀ : E} {vs : List (E → ℝ ⊕ Bool)} {t : List ((ℝ ⊕ Bool) × (E →L[ℝ] ℝ))}

/-- Where the dual of a term succeeds on a state related to `vs`, the term on `vs` near `θ₀` is a
real family with the dual's value and derivative at `θ₀`. -/
theorem Term.near (hD : List.Forall₂ (Near θ₀) vs t) :
    ∀ (e : Term) {x : ℝ} {d : E →L[ℝ] ℝ}, e.dual t = .ok (x, d) →
      ∃ f : E → ℝ, (∀ᶠ θ in 𝓝 θ₀, e.eval (vs.map (· θ)) = .ok (f θ)) ∧ f θ₀ = x ∧
        HasFDerivAt f d θ₀
  | .var i, x, d, h => by
      have hi := forall₂_getElem? hD i
      cases ht : t[i]? with
      | none => simp [Term.dual, ht, throw, throwThe, MonadExceptOf.throw] at h
      | some w =>
          rcases w with ⟨_ | c, d'⟩
          · simp only [Term.dual, ht, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
            obtain ⟨rfl, rfl⟩ := h
            rw [ht] at hi
            generalize hv : vs[i]? = ov at hi
            cases hi with
            | some hw =>
                obtain ⟨f, hf, hx, hd⟩ := hw
                refine ⟨f, hf.mono fun θ e => ?_, hx, hd⟩
                simp [Term.eval, List.getElem?_map, hv, e, pure, Except.pure]
          · simp [Term.dual, ht, throw, throwThe, MonadExceptOf.throw] at h
  | .const c, x, d, h => by
      simp only [Term.dual, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      exact ⟨fun _ => c, .of_forall fun _ => rfl, rfl, hasFDerivAt_const c θ₀⟩
  | .add a b, x, d, h => by
      obtain ⟨⟨x₁, d₁⟩, h₁, h⟩ := bind_eq_ok.1 h
      obtain ⟨⟨x₂, d₂⟩, h₂, h⟩ := bind_eq_ok.1 h
      simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      obtain ⟨f₁, hf₁, rfl, hd₁⟩ := Term.near hD a h₁
      obtain ⟨f₂, hf₂, rfl, hd₂⟩ := Term.near hD b h₂
      refine ⟨fun θ => f₁ θ + f₂ θ, (hf₁.and hf₂).mono fun θ ⟨e₁, e₂⟩ => ?_, rfl,
        hd₁.fun_add hd₂⟩
      simp [Term.eval, e₁, e₂, bind, Except.bind, pure, Except.pure]
  | .mul a b, x, d, h => by
      obtain ⟨⟨x₁, d₁⟩, h₁, h⟩ := bind_eq_ok.1 h
      obtain ⟨⟨x₂, d₂⟩, h₂, h⟩ := bind_eq_ok.1 h
      simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      obtain ⟨f₁, hf₁, rfl, hd₁⟩ := Term.near hD a h₁
      obtain ⟨f₂, hf₂, rfl, hd₂⟩ := Term.near hD b h₂
      refine ⟨fun θ => f₁ θ * f₂ θ, (hf₁.and hf₂).mono fun θ ⟨e₁, e₂⟩ => ?_, rfl,
        hd₁.fun_mul hd₂⟩
      simp [Term.eval, e₁, e₂, bind, Except.bind, pure, Except.pure]
  | .neg a, x, d, h => by
      obtain ⟨⟨x₁, d₁⟩, h₁, h⟩ := bind_eq_ok.1 h
      simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      obtain ⟨f, hf, rfl, hd⟩ := Term.near hD a h₁
      refine ⟨fun θ => -f θ, hf.mono fun θ e => ?_, rfl, hd.fun_neg⟩
      simp [Term.eval, e, bind, Except.bind, pure, Except.pure]

/-- Where the dual of an output succeeds on a state related to `vs`, the output on `vs` near `θ₀`
is a family related to the dual's result. -/
theorem Out.near (hD : List.Forall₂ (Near θ₀) vs t) {o : Out}
    {w : (ℝ ⊕ Bool) × (E →L[ℝ] ℝ)} (h : o.dual t = .ok w) :
    ∃ g : E → ℝ ⊕ Bool, (∀ᶠ θ in 𝓝 θ₀, o.eval (vs.map (· θ)) = .ok (g θ)) ∧ Near θ₀ g w := by
  cases o with
  | real e =>
      obtain ⟨⟨x, d⟩, he, h⟩ := bind_eq_ok.1 h
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      obtain ⟨f, hf, hx, hd⟩ := Term.near hD e he
      refine ⟨fun θ => .inl (f θ), hf.mono fun θ e => ?_, f, .of_forall fun _ => rfl, hx, hd⟩
      simp [Out.eval, e, bind, Except.bind, pure, Except.pure]
  | lt a b =>
      obtain ⟨⟨x, d⟩, ha, h⟩ := bind_eq_ok.1 h
      obtain ⟨⟨y, e⟩, hb, h⟩ := bind_eq_ok.1 h
      by_cases hxy : x = y
      · simp [hxy, throw, throwThe, MonadExceptOf.throw] at h
      simp only [hxy, ↓reduceIte, pure, Except.pure, Except.ok.injEq] at h
      subst h
      obtain ⟨f₁, hf₁, rfl, hd₁⟩ := Term.near hD a ha
      obtain ⟨f₂, hf₂, rfl, hd₂⟩ := Term.near hD b hb
      have hc : ∀ᶠ θ in 𝓝 θ₀, decide (f₁ θ < f₂ θ) = decide (f₁ θ₀ < f₂ θ₀) := by
        rcases lt_or_gt_of_ne hxy with hlt | hlt
        · exact (hd₁.continuousAt.eventually_lt hd₂.continuousAt hlt).mono fun θ e => by
            simp [e, hlt]
        · exact (hd₂.continuousAt.eventually_lt hd₁.continuousAt hlt).mono fun θ e => by
            simp [lt_asymm e, lt_asymm hlt]
      refine ⟨fun _ => .inr (decide (f₁ θ₀ < f₂ θ₀)), (hf₁.and (hf₂.and hc)).mono ?_,
        .of_forall fun _ => rfl⟩
      rintro θ ⟨e₁, e₂, e₃⟩
      simp [Out.eval, e₁, e₂, e₃, bind, Except.bind, pure, Except.pure]

/-- At every point `θ₀` of every real normed space `E`, the dual of `lang` with derivatives in
`E →L[ℝ] ℝ` follows `lang` near `θ₀` through `Near θ₀`. -/
theorem dual_follows (θ₀ : E) : Follows lang (dual E) (𝓝 θ₀) (Near θ₀) :=
  follows_of (fun _ _ hD _ _ h => Out.near hD h) fun _ _ _ h => h

end Tangents

section Roundoff

variable (rnd err : ℝ → ℝ)

/-- The computed value and error bound of a term on a state of computed values and bounds: a
constant, a sum and a product are rounded by `rnd`, and their bound adds `err` of the exact result
to the error the operands' bounds propagate to it; a negation is exact. -/
def Term.round (t : List ((ℝ ⊕ Bool) × ℝ)) : Term → Except String (ℝ × ℝ)
  | .var i => match t[i]? with
    | some (.inl x, e) => pure (x, e)
    | _ => throw s!"no real at {i}"
  | .const c => pure (rnd c, err c)
  | .add a b => do
      let p ← a.round t
      let q ← b.round t
      pure (rnd (p.1 + q.1), p.2 + q.2 + err (p.1 + q.1))
  | .mul a b => do
      let p ← a.round t
      let q ← b.round t
      pure (rnd (p.1 * q.1), |p.1| * q.2 + |q.1| * p.2 + p.2 * q.2 + err (p.1 * q.1))
  | .neg a => do
      let p ← a.round t
      pure (-p.1, p.2)

/-- The rounded output: a real term's computed value and bound, or a comparison with bound zero,
which fails where its two reals are within their bounds of each other. -/
def Out.round (t : List ((ℝ ⊕ Bool) × ℝ)) : Out → Except String ((ℝ ⊕ Bool) × ℝ)
  | .real e => do
      let p ← e.round rnd err t
      pure (.inl p.1, p.2)
  | .lt a b => do
      let p ← a.round rnd err t
      let q ← b.round rnd err t
      if |p.1 - q.1| ≤ p.2 + q.2 then throw "lt: operands within their error bounds"
      else pure (.inr (decide (p.1 < q.1)), 0)

/-- The rounded `lang`, with rounding `rnd` and rounding error bound `err`: a block maps the state
to the rounded outputs. -/
def rounded : Lang (Except String) (List Out) ((ℝ ⊕ Bool) × ℝ) :=
  lang.dual fun b t => b.mapM (Out.round rnd err t)

variable {ι : Type}

/-- The family `v` of exact values is within the bound of the computed value `w`: for `w = (x, e)`
with `x` real, every value of `v` is a real within `e` of `x`, and for `w` a Boolean `c`, every
value of `v` is `c`. -/
def Within (v : ι → ℝ ⊕ Bool) : (ℝ ⊕ Bool) × ℝ → Prop
  | (.inl x, e) => ∃ f : ι → ℝ, (∀ θ, v θ = .inl (f θ)) ∧ ∀ θ, |f θ - x| ≤ e
  | (.inr c, _) => ∀ θ, v θ = .inr c

variable {rnd err} {vs : List (ι → ℝ ⊕ Bool)} {t : List ((ℝ ⊕ Bool) × ℝ)}

/-- Where the rounded term succeeds on a state whose bounds `vs` is within, the term on `vs` is a
real family within the result's bound of its computed value. -/
theorem Term.within (h : ∀ z, |rnd z - z| ≤ err z) (hD : List.Forall₂ Within vs t) :
    ∀ (e : Term) {x b : ℝ}, e.round rnd err t = .ok (x, b) →
      ∃ f : ι → ℝ, (∀ θ, e.eval (vs.map (· θ)) = .ok (f θ)) ∧ ∀ θ, |f θ - x| ≤ b
  | .var i, x, b, he => by
      have hi := forall₂_getElem? hD i
      cases ht : t[i]? with
      | none => simp [Term.round, ht, throw, throwThe, MonadExceptOf.throw] at he
      | some w =>
          rcases w with ⟨_ | c, b'⟩
          · simp only [Term.round, ht, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at he
            obtain ⟨rfl, rfl⟩ := he
            rw [ht] at hi
            generalize hv : vs[i]? = ov at hi
            cases hi with
            | some hw =>
                obtain ⟨f, hf, hb⟩ := hw
                refine ⟨f, fun θ => ?_, hb⟩
                simp [Term.eval, List.getElem?_map, hv, hf θ, pure, Except.pure]
          · simp [Term.round, ht, throw, throwThe, MonadExceptOf.throw] at he
  | .const c, x, b, he => by
      simp only [Term.round, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at he
      obtain ⟨rfl, rfl⟩ := he
      exact ⟨fun _ => c, fun _ => rfl, fun _ => by rw [abs_sub_comm]; exact h c⟩
  | .add a a', x, b, he => by
      obtain ⟨⟨x₁, b₁⟩, h₁, he⟩ := bind_eq_ok.1 he
      obtain ⟨⟨x₂, b₂⟩, h₂, he⟩ := bind_eq_ok.1 he
      simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at he
      obtain ⟨rfl, rfl⟩ := he
      obtain ⟨f₁, hf₁, hb₁⟩ := Term.within h hD a h₁
      obtain ⟨f₂, hf₂, hb₂⟩ := Term.within h hD a' h₂
      refine ⟨fun θ => f₁ θ + f₂ θ, fun θ => ?_, fun θ => ?_⟩
      · simp [Term.eval, hf₁ θ, hf₂ θ, bind, Except.bind, pure, Except.pure]
      rw [show f₁ θ + f₂ θ - rnd (x₁ + x₂) = (f₁ θ - x₁) + (f₂ θ - x₂) + (x₁ + x₂ - rnd (x₁ + x₂))
        by ring]
      have := h (x₁ + x₂)
      rw [abs_sub_comm] at this
      linarith [abs_add_three (f₁ θ - x₁) (f₂ θ - x₂) (x₁ + x₂ - rnd (x₁ + x₂)), hb₁ θ, hb₂ θ]
  | .mul a a', x, b, he => by
      obtain ⟨⟨x₁, b₁⟩, h₁, he⟩ := bind_eq_ok.1 he
      obtain ⟨⟨x₂, b₂⟩, h₂, he⟩ := bind_eq_ok.1 he
      simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at he
      obtain ⟨rfl, rfl⟩ := he
      obtain ⟨f₁, hf₁, hb₁⟩ := Term.within h hD a h₁
      obtain ⟨f₂, hf₂, hb₂⟩ := Term.within h hD a' h₂
      refine ⟨fun θ => f₁ θ * f₂ θ, fun θ => ?_, fun θ => ?_⟩
      · simp [Term.eval, hf₁ θ, hf₂ θ, bind, Except.bind, pure, Except.pure]
      rw [show f₁ θ * f₂ θ - rnd (x₁ * x₂) = x₁ * (f₂ θ - x₂) + x₂ * (f₁ θ - x₁) +
        (f₁ θ - x₁) * (f₂ θ - x₂) + (x₁ * x₂ - rnd (x₁ * x₂)) by ring]
      have e₁ := abs_add_le (x₁ * (f₂ θ - x₂) + x₂ * (f₁ θ - x₁) + (f₁ θ - x₁) * (f₂ θ - x₂))
        (x₁ * x₂ - rnd (x₁ * x₂))
      have e₂ := abs_add_three (x₁ * (f₂ θ - x₂)) (x₂ * (f₁ θ - x₁)) ((f₁ θ - x₁) * (f₂ θ - x₂))
      simp only [abs_mul] at e₂
      have e₃ := mul_le_mul_of_nonneg_left (hb₂ θ) (abs_nonneg x₁)
      have e₄ := mul_le_mul_of_nonneg_left (hb₁ θ) (abs_nonneg x₂)
      have e₅ := mul_le_mul (hb₁ θ) (hb₂ θ) (abs_nonneg _) ((abs_nonneg _).trans (hb₁ θ))
      have e₆ := h (x₁ * x₂)
      rw [abs_sub_comm] at e₆
      linarith
  | .neg a, x, b, he => by
      obtain ⟨⟨x₁, b₁⟩, h₁, he⟩ := bind_eq_ok.1 he
      simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at he
      obtain ⟨rfl, rfl⟩ := he
      obtain ⟨f, hf, hb⟩ := Term.within h hD a h₁
      refine ⟨fun θ => -f θ, fun θ => ?_, fun θ => ?_⟩
      · simp [Term.eval, hf θ, bind, Except.bind, pure, Except.pure]
      rw [show -f θ - -x₁ = -(f θ - x₁) by ring, abs_neg]
      exact hb θ

/-- Where the rounded output succeeds on a state whose bounds `vs` is within, the output on `vs` is
a family within the result. -/
theorem Out.within (h : ∀ z, |rnd z - z| ≤ err z) (hD : List.Forall₂ Within vs t) {o : Out}
    {w : (ℝ ⊕ Bool) × ℝ} (ho : o.round rnd err t = .ok w) :
    ∃ g : ι → ℝ ⊕ Bool, (∀ θ, o.eval (vs.map (· θ)) = .ok (g θ)) ∧ Within g w := by
  cases o with
  | real e =>
      obtain ⟨⟨x, b⟩, he, ho⟩ := bind_eq_ok.1 ho
      simp only [pure, Except.pure, Except.ok.injEq] at ho
      subst ho
      obtain ⟨f, hf, hb⟩ := Term.within h hD e he
      refine ⟨fun θ => .inl (f θ), fun θ => ?_, f, fun _ => rfl, hb⟩
      simp [Out.eval, hf θ, bind, Except.bind, pure, Except.pure]
  | lt a a' =>
      obtain ⟨⟨x₁, b₁⟩, h₁, ho⟩ := bind_eq_ok.1 ho
      obtain ⟨⟨x₂, b₂⟩, h₂, ho⟩ := bind_eq_ok.1 ho
      by_cases hm : |x₁ - x₂| ≤ b₁ + b₂
      · simp [hm, throw, throwThe, MonadExceptOf.throw] at ho
      simp only [hm, ↓reduceIte, pure, Except.pure, Except.ok.injEq] at ho
      subst ho
      obtain ⟨f₁, hf₁, hb₁⟩ := Term.within h hD a h₁
      obtain ⟨f₂, hf₂, hb₂⟩ := Term.within h hD a' h₂
      refine ⟨fun _ => .inr (decide (x₁ < x₂)), fun θ => ?_, fun _ => rfl⟩
      obtain ⟨e₁, e₂⟩ := abs_le.1 (hb₁ θ)
      obtain ⟨e₃, e₄⟩ := abs_le.1 (hb₂ θ)
      have hc : decide (f₁ θ < f₂ θ) = decide (x₁ < x₂) := by
        rcases lt_abs.1 (not_le.1 hm) with hm | hm <;>
          exact decide_eq_decide.2 ⟨fun _ => by linarith, fun _ => by linarith⟩
      simp [Out.eval, hf₁ θ, hf₂ θ, hc, bind, Except.bind, pure, Except.pure]

variable (ι) in
/-- For every rounding `rnd` with error at most `err`, the rounded `lang` follows `lang` along `⊤`
on every type `ι` through `Within`. -/
theorem rounded_follows (h : ∀ z, |rnd z - z| ≤ err z) :
    Follows lang (rounded rnd err) (⊤ : Filter ι) Within :=
  follows_of (fun _ _ hD _ _ ho => (Out.within h hD ho).imp fun _ ⟨hg, hw⟩ =>
    ⟨.of_forall hg, hw⟩) fun _ _ _ hv => .of_forall hv

end Roundoff

end

end TorchLean.IR.Flow.Reals
