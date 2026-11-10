/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.CUDA.Kernel.Program

/-!
# Tangent kernels

`Expr.tangent k e` is the forward-mode derivative of a kernel expression `e` that reads `k` inputs.
It is an expression of the same language, so it runs on the CPU with `Expr.eval` and on the GPU
through the same CUDA emitter. It reads the primal inputs as operands `0` to `k - 1` and the
tangent of input `j` as operand `k + j`, at the same addresses.

A tangent expression lives in the scope `dualScope Γ`, where every scalar variable of `Γ` becomes
two adjacent variables, its tangent and then its primal. `Expr.primal` recomputes a value in that
scope. A `let` binds the tangent of its value next to the primal, a conditional keeps the primal
predicate, and the tangent of a product or a quotient is the product or quotient rule in kernel
arithmetic. A fold carries the tangent accumulator. When the tangent of the fold body reads the
primal accumulator, as for a product or for a `let` of a value computed from the accumulator, each
step recomputes the primal accumulator with a fold over the earlier steps, so the tangent fold
takes time quadratic in the count. A fold whose body adds a term to its accumulator, as
`acc + x * y` for a dot product, keeps its linear cost.

`Expr.eval_primal` shows that the primal translation computes the value of the expression, for any
arithmetic, wherever the evaluation of the expression succeeds. The theorem that the tangent
computes the real derivative is in `TorchLean.CUDA.Kernel.Tangent.FDeriv`.
-/

@[expose] public section

namespace TorchLean.Kernel

/-- The scope of a tangent expression: every scalar variable of `Γ` becomes its tangent followed
by its primal, and index and predicate variables stay single. -/
def dualScope : List Ty → List Ty
  | [] => []
  | .scalar :: Γ => .scalar :: .scalar :: dualScope Γ
  | .index :: Γ => .index :: dualScope Γ
  | .predicate :: Γ => .predicate :: dualScope Γ

namespace Var

/-- The variable of `dualScope Γ` that holds the value of `v`. -/
def primal : {Γ : List Ty} → {t : Ty} → Var Γ t → Var (dualScope Γ) t
  | .scalar :: _, _, .zero => .succ .zero
  | .index :: _, _, .zero => .zero
  | .predicate :: _, _, .zero => .zero
  | .scalar :: _, _, .succ v => .succ (.succ v.primal)
  | .index :: _, _, .succ v => .succ v.primal
  | .predicate :: _, _, .succ v => .succ v.primal

/-- The variable of `dualScope Γ` that holds the tangent of the scalar variable `v`. -/
def tangent : {Γ : List Ty} → Var Γ .scalar → Var (dualScope Γ) .scalar
  | .scalar :: _, .zero => .zero
  | .scalar :: _, .succ v => .succ (.succ v.tangent)
  | .index :: _, .succ v => .succ v.tangent
  | .predicate :: _, .succ v => .succ v.tangent

/-- Whether two variables of a scope, possibly of different types, are at the same position. -/
def same : {Γ : List Ty} → {s t : Ty} → Var Γ s → Var Γ t → Bool
  | _, _, _, .zero, .zero => true
  | _, _, _, .zero, .succ _ => false
  | _, _, _, .succ _, .zero => false
  | _, _, _, .succ v, .succ w => same v w

end Var

namespace Expr

variable {α : Type}

/-- Whether the expression reads the variable `v`. -/
def reads : {Γ : List Ty} → {s t : Ty} → Var Γ s → Expr α Γ t → Bool
  | _, _, _, _, .scalar _ => false
  | _, _, _, _, .index _ => false
  | _, _, _, _, .predicate _ => false
  | _, _, _, v, .var w => v.same w
  | _, _, _, v, .load _ i => reads v i
  | _, _, _, v, .binary _ x y => reads v x || reads v y
  | _, _, _, v, .neg x => reads v x
  | _, _, _, v, .indexBinary _ x y => reads v x || reads v y
  | _, _, _, v, .compare _ x y => reads v x || reads v y
  | _, _, _, v, .indexCompare _ x y => reads v x || reads v y
  | _, _, _, v, .cond p yes no => reads v p || reads v yes || reads v no
  | _, _, _, v, .letIn value body => reads v value || reads v.succ body
  | _, _, _, v, .fold count initial body =>
      reads v count || reads v initial || reads v.succ.succ body

/-- Bind a value in a dual scope: a scalar `value` together with its `tangent` on top, a value of
another type alone. -/
def bindDual {Γ : List Ty} {t : Ty} : (s : Ty) → Expr α (dualScope Γ) s →
    Expr α (s :: dualScope Γ) s → Expr α (dualScope (s :: Γ)) t → Expr α (dualScope Γ) t
  | .scalar, value, tangent, body => .letIn value (.letIn tangent body)
  | .index, value, _, body => .letIn value body
  | .predicate, value, _, body => .letIn value body

/-- The expression in the dual scope, computing its value from the primal variables. -/
def primal : {Γ : List Ty} → {t : Ty} → Expr α Γ t → Expr α (dualScope Γ) t
  | _, _, .scalar x => .scalar x
  | _, _, .index i => .index i
  | _, _, .predicate p => .predicate p
  | _, _, .var v => .var v.primal
  | _, _, .load j i => .load j i.primal
  | _, _, .binary op x y => .binary op x.primal y.primal
  | _, _, .neg x => .neg x.primal
  | _, _, .indexBinary op x y => .indexBinary op x.primal y.primal
  | _, _, .compare op x y => .compare op x.primal y.primal
  | _, _, .indexCompare op x y => .indexCompare op x.primal y.primal
  | _, _, .cond p yes no => .cond p.primal yes.primal no.primal
  | _, _, @letIn _ _ s _ value body => bindDual s value.primal (.var .zero) body.primal
  | _, _, .fold count initial body =>
      .fold count.primal initial.primal (.letIn (.var .zero) body.primal)

/-- The step of a tangent fold whose body has tangent `body'` (accumulator tangent, accumulator,
index, then the dual scope). If `body'` reads the primal accumulator, the step first recomputes it
by a fold of `body` over the earlier steps from `initial`, and otherwise it binds the accumulator
tangent in its place. -/
def foldTangent {Γ : List Ty} (initial : Expr α (dualScope Γ) .scalar)
    (body : Expr α (.scalar :: .scalar :: .index :: dualScope Γ) .scalar)
    (body' : Expr α (.scalar :: .scalar :: .index :: dualScope Γ) .scalar) :
    Expr α (.scalar :: .index :: dualScope Γ) .scalar :=
  if body'.reads (Var.succ (s := .scalar) (Var.zero (t := .scalar))) then
    .letIn
      (.fold (.var (.succ .zero)) (initial.rename fun v => .succ (.succ v))
        (.letIn (.var .zero) (body.rename
          (Renaming.lift (Renaming.lift (Renaming.lift fun v => .succ (.succ v)))))))
      (.letIn (.var (.succ .zero)) (body'.rename (Renaming.lift (Renaming.lift fun v => .succ v))))
  else .letIn (.var .zero) body'

variable [OfNat α 0]

/-- The forward-mode derivative of an expression that reads `k` inputs, in the dual scope: the
tangent of a scalar expression, and the value of an index or predicate expression. The tangent of
input `j` is read as operand `k + j`. -/
def tangent (k : Nat) : {Γ : List Ty} → {t : Ty} → Expr α Γ t → Expr α (dualScope Γ) t
  | _, _, .scalar _ => .scalar 0
  | _, _, .index i => .index i
  | _, _, .predicate p => .predicate p
  | _, .scalar, .var v => .var v.tangent
  | _, .index, .var v => .var v.primal
  | _, .predicate, .var v => .var v.primal
  | _, _, .load j i => .load (k + j) i.primal
  | _, _, .binary .add x y => .binary .add (x.tangent k) (y.tangent k)
  | _, _, .binary .sub x y => .binary .sub (x.tangent k) (y.tangent k)
  | _, _, .binary .mul x y =>
      .binary .add (.binary .mul (x.tangent k) y.primal) (.binary .mul x.primal (y.tangent k))
  | _, _, .binary .div x y =>
      .binary .div
        (.binary .sub (x.tangent k) (.binary .mul (.binary .div x.primal y.primal) (y.tangent k)))
        y.primal
  | _, _, .neg x => .neg (x.tangent k)
  | _, _, .indexBinary op x y => .indexBinary op x.primal y.primal
  | _, _, .compare op x y => .compare op x.primal y.primal
  | _, _, .indexCompare op x y => .indexCompare op x.primal y.primal
  | _, _, .cond p yes no => .cond p.primal (yes.tangent k) (no.tangent k)
  | _, _, @letIn _ _ s _ value body =>
      bindDual s value.primal ((value.tangent k).rename fun v => .succ v) (body.tangent k)
  | _, _, .fold count initial body =>
      .fold count.primal (initial.tangent k)
        (foldTangent initial.primal body.primal (body.tangent k))

end Expr

/-- The forward-mode derivative of a program that reads `k` inputs: it reads the primal inputs as
operands `0` to `k - 1` and the tangent of input `j` as operand `k + j`, computes the tangent of
each output entry, and has the evaluation of the tangent expression as its reference. The count
`k` is not checked, and it must exceed every operand that the program loads. -/
def Program.tangent {α : Type} [OfNat α 0] (program : Program α) (k : Nat) : Program α where
  scalar := program.scalar
  arithmetic := program.arithmetic
  reference := evaluate program.arithmetic (program.expression.tangent k)
  expression := program.expression.tangent k
  correct _ _ := rfl
  native := program.native

/-! ## The primal translation -/

namespace Expr

variable {α : Type}

/-- A successful evaluation `m >>= f` is a successful `m` followed by a successful `f`. -/
theorem except_bind_eq_ok {β γ : Type} {m : Except Error β} {f : β → Except Error γ} {y : γ} :
    m >>= f = .ok y ↔ ∃ a, m = .ok a ∧ f a = .ok y := by
  cases m with
  | error e => simp [Bind.bind, Except.bind]
  | ok a => simp [Bind.bind, Except.bind]

/-- A fold whose steps succeed wherever the steps of a successful fold succeed, with the same
results, succeeds with the same result. -/
theorem iterate_eq_ok_of_imp {β : Type} {f g : UInt64 → β → Except Error β}
    (h : ∀ i a b, f i a = .ok b → g i a = .ok b) :
    ∀ (n : Nat) (i : UInt64) (a : β) {y : β}, iterate f n i a = .ok y → iterate g n i a = .ok y
  | 0, _, _, _, hy => hy
  | n + 1, i, a, y, hy => by
      simp only [iterate] at hy ⊢
      obtain ⟨b, hb, hy⟩ := except_bind_eq_ok.mp hy
      rw [h i a b hb]
      exact iterate_eq_ok_of_imp h n (i + 1) b hy

/-- Under a successful evaluation, the primal translation computes the same value in a dual
environment whose primal variables hold the values of the environment, with a reader that agrees
with the original reader wherever the original read succeeds. -/
theorem eval_primal (arithmetic : Arithmetic α) {read dread : Reader α}
    (hread : ∀ j i y, read j i = .ok y → dread j i = .ok y) {Γ : List Ty} {t : Ty}
    (e : Expr α Γ t) {env : Env α Γ} {denv : Env α (dualScope Γ)}
    (henv : ∀ {s : Ty} (v : Var Γ s), denv.get v.primal = env.get v) {y : t.Value α}
    (h : e.eval arithmetic read env = .ok y) : e.primal.eval arithmetic dread denv = .ok y := by
  induction e with
  | scalar | index | predicate => exact h
  | var v => simpa only [eval, primal, henv] using h
  | load j i ih =>
      simp only [eval, primal] at h ⊢
      obtain ⟨a, ha, h⟩ := except_bind_eq_ok.mp h
      rw [ih henv ha]
      exact hread j a y h
  | binary op x y hx hy =>
      simp only [eval, primal] at h ⊢
      obtain ⟨a, ha, h⟩ := except_bind_eq_ok.mp h
      obtain ⟨b, hb, h⟩ := except_bind_eq_ok.mp h
      rw [hx henv ha, hy henv hb]
      exact h
  | neg x hx =>
      simp only [eval, primal] at h ⊢
      obtain ⟨a, ha, h⟩ := except_bind_eq_ok.mp h
      rw [hx henv ha]
      exact h
  | indexBinary op x y hx hy =>
      simp only [eval, primal] at h ⊢
      obtain ⟨a, ha, h⟩ := except_bind_eq_ok.mp h
      obtain ⟨b, hb, h⟩ := except_bind_eq_ok.mp h
      rw [hx henv ha, hy henv hb]
      exact h
  | compare op x y hx hy =>
      simp only [eval, primal] at h ⊢
      obtain ⟨a, ha, h⟩ := except_bind_eq_ok.mp h
      obtain ⟨b, hb, h⟩ := except_bind_eq_ok.mp h
      rw [hx henv ha, hy henv hb]
      exact h
  | indexCompare op x y hx hy =>
      simp only [eval, primal] at h ⊢
      obtain ⟨a, ha, h⟩ := except_bind_eq_ok.mp h
      obtain ⟨b, hb, h⟩ := except_bind_eq_ok.mp h
      rw [hx henv ha, hy henv hb]
      exact h
  | cond p yes no hp hyes hno =>
      simp only [eval, primal] at h ⊢
      obtain ⟨b, hb, h⟩ := except_bind_eq_ok.mp h
      rw [hp henv hb]
      cases b
      · exact hno henv h
      · exact hyes henv h
  | @letIn _ s _ value body hv hb =>
      simp only [eval] at h
      obtain ⟨a, ha, h⟩ := except_bind_eq_ok.mp h
      have hv' := hv henv ha
      cases s with
      | scalar =>
          simp only [primal, bindDual, eval, Env.get_push_zero]
          refine except_bind_eq_ok.mpr ⟨a, hv', except_bind_eq_ok.mpr ⟨a, rfl, ?_⟩⟩
          refine hb (env := env.push a) (denv := (denv.push a).push a) ?_ h
          intro s' v
          cases v with
          | zero => rfl
          | succ v => exact henv v
      | index =>
          simp only [primal, bindDual, eval]
          refine except_bind_eq_ok.mpr ⟨a, hv', ?_⟩
          refine hb (env := env.push a) (denv := denv.push a) ?_ h
          intro s' v
          cases v with
          | zero => rfl
          | succ v => exact henv v
      | predicate =>
          simp only [primal, bindDual, eval]
          refine except_bind_eq_ok.mpr ⟨a, hv', ?_⟩
          refine hb (env := env.push a) (denv := denv.push a) ?_ h
          intro s' v
          cases v with
          | zero => rfl
          | succ v => exact henv v
  | fold count initial body hc hi hb =>
      simp only [eval] at h
      obtain ⟨c, hcount, h⟩ := except_bind_eq_ok.mp h
      obtain ⟨a, ha, h⟩ := except_bind_eq_ok.mp h
      simp only [primal, eval]
      refine except_bind_eq_ok.mpr ⟨c, hc henv hcount, except_bind_eq_ok.mpr ⟨a, hi henv ha, ?_⟩⟩
      refine iterate_eq_ok_of_imp (fun i acc b hstep => ?_) c.toNat 0 a h
      refine except_bind_eq_ok.mpr ⟨acc, rfl, ?_⟩
      refine hb (env := (env.push i).push acc) (denv := ((denv.push i).push acc).push acc) ?_ hstep
      intro s' v
      cases v with
      | zero => rfl
      | succ v =>
          cases v with
          | zero => rfl
          | succ v => exact henv v

end Expr

end TorchLean.Kernel
