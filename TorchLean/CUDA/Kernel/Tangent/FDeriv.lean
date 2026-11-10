/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.CUDA.Kernel.Tangent
public import Mathlib.Analysis.Calculus.Deriv.Inv
public import Mathlib.Analysis.Calculus.FDeriv.Mul

/-!
# Tangent kernels compute derivatives

Over the reals, the tangent expression `Expr.tangent k e` computes the Fréchet derivative of the
value of `e` as a function of its inputs. We move the input values with a parameter `θ` of a
normed space: `read.withValues (x θ)` reads `x θ j i` wherever `read` succeeds and fails wherever
`read` fails. Suppose the evaluation succeeds at `θ₀` and every input entry `x θ j i` has the
derivative `x' j i` there. Then the value of `e` has a derivative `D` at `θ₀`, and the tangent
expression computes `D u` with the dual reader that holds the primal inputs `x θ₀` and the
tangents `x' j i u` (`Expr.exists_hasFDerivAt_tangent`).

The hypothesis `Expr.Regular` asks that the evaluation at `θ₀` divides by no zero and compares no
two equal scalars unless both are constants (`Expr.IsConstant`). Then the evaluation near `θ₀`
takes the same branches and the same numbers of fold steps, so near `θ₀` the value is built from
input reads by the four arithmetic operations. The condition is sufficient and not necessary:
`max x x` has a derivative although its comparison is a tie. It also rejects a tie whose outcome
does not depend on the inputs, such as `acc < 0` at the first step of a fold that starts from `0`,
or `mask[i] == 1` on an input `mask` that does not move. With one more case at comparisons, the
induction below also allows a comparison whose operands agree near `θ₀` as functions of `θ`. We do
not state that condition, since `Expr.Regular` sees the evaluation at `θ₀` only.

The proof is one induction over the expression. It carries, for every scalar variable in scope,
the derivative of its value, and for every index or predicate variable, its value, which does not
change near `θ₀`. A scalar `let` and a fold step pass the derivative of their value to their body.
`Expr.hasDerivAt_evaluate_tangent` states the result on a line `x + s dx` through the inputs,
where the tangent inputs are `dx`.
-/

@[expose] public section

open Filter hiding pure_bind
open Topology

namespace TorchLean.Kernel

/-- The reader that reads `x j i` wherever `read` succeeds and fails wherever `read` fails. -/
def Reader.withValues {α : Type} (read : Reader α) (x : Nat → UInt64 → α) : Reader α :=
  fun j i => (read j i).map fun _ => x j i

/-- The reader of the tangent of a program that reads `k` inputs through `read`: operand `j < k`
reads the primal values `x`, and operand `k + j` reads the tangents `dx` of input `j`, wherever
`read` succeeds. -/
def Reader.dual {α : Type} (read : Reader α) (k : Nat) (x dx : Nat → UInt64 → α) : Reader α :=
  fun j i => if j < k then read.withValues x j i else read.withValues dx (j - k) i

namespace Reader

variable {α : Type} {read : Reader α}

/-- Where `read` succeeds, `read.withValues x` reads `x`. -/
theorem withValues_of_ok {j : Nat} {i : UInt64} {z : α} (h : read j i = .ok z)
    (x : Nat → UInt64 → α) : read.withValues x j i = .ok (x j i) := by
  simp only [withValues, h, Except.map]

/-- `read.withValues x` succeeds only where `read` succeeds, and then it reads `x`. -/
theorem exists_ok_of_withValues {x : Nat → UInt64 → α} {j : Nat} {i : UInt64} {y : α}
    (h : read.withValues x j i = .ok y) : ∃ z, read j i = .ok z ∧ x j i = y := by
  unfold withValues at h
  cases hr : read j i with
  | error e => simp only [hr, Except.map, reduceCtorEq] at h
  | ok z =>
      simp only [hr, Except.map, Except.ok.injEq] at h
      exact ⟨z, rfl, h⟩

/-- If `read` only succeeds below operand `k`, the dual reader reads the primal values wherever
`read.withValues x` succeeds. -/
theorem dual_eq_ok {k : Nat} (hk : ∀ j i y, read j i = .ok y → j < k)
    (x dx : Nat → UInt64 → α) {j : Nat} {i : UInt64} {y : α}
    (h : read.withValues x j i = .ok y) : read.dual k x dx j i = .ok y := by
  obtain ⟨z, hz, -⟩ := exists_ok_of_withValues h
  simp only [dual, hk j i z hz, ↓reduceIte]
  exact h

/-- The dual reader reads the tangent of input `j` as operand `k + j`. -/
theorem dual_add (k : Nat) (x dx : Nat → UInt64 → α) (j : Nat) (i : UInt64) :
    read.dual k x dx (k + j) i = read.withValues dx j i := by
  simp only [dual, Nat.not_lt.mpr (Nat.le_add_right k j), ↓reduceIte, Nat.add_sub_cancel_left]

end Reader

namespace Arithmetic

variable {α : Type} [Add α] [Sub α] [Mul α] [Div α] [Neg α] [BEq α] [LT α] [LE α]
  [DecidableLT α] [DecidableLE α]

/-- `ofOps` adds with the type's `+`. -/
@[simp] theorem ofOps_binary_add (x y : α) : (ofOps : Arithmetic α).binary .add x y = x + y :=
  rfl

/-- `ofOps` subtracts with the type's `-`. -/
@[simp] theorem ofOps_binary_sub (x y : α) : (ofOps : Arithmetic α).binary .sub x y = x - y :=
  rfl

/-- `ofOps` multiplies with the type's `*`. -/
@[simp] theorem ofOps_binary_mul (x y : α) : (ofOps : Arithmetic α).binary .mul x y = x * y :=
  rfl

/-- `ofOps` divides with the type's `/`. -/
@[simp] theorem ofOps_binary_div (x y : α) : (ofOps : Arithmetic α).binary .div x y = x / y :=
  rfl

/-- `ofOps` negates with the type's `-`. -/
@[simp] theorem ofOps_neg (x : α) : (ofOps : Arithmetic α).neg x = -x := rfl

end Arithmetic

/-- Pulling an environment back along the shift past its two newest variables drops them. -/
@[simp] theorem Env.pull_succ_succ_push {α : Type} {Γ : List Ty} {s s' : Ty} (a : s.Value α)
    (b : s'.Value α) (env : Env α Γ) :
    Env.pull (fun v => Var.succ (Var.succ v)) ((env.push a).push b) = env := rfl

namespace Expr

/-- `e` is built from scalar constants by the arithmetic operations and negation: it loads no
input and reads no variable. -/
def IsConstant : {Γ : List Ty} → {t : Ty} → Expr ℝ Γ t → Prop
  | _, _, .scalar _ => True
  | _, _, .binary _ x y => x.IsConstant ∧ y.IsConstant
  | _, _, .neg x => x.IsConstant
  | _, _, _ => False

/-- A constant expression has the same evaluation with every reader and in every environment. -/
theorem IsConstant.eval_eq {Γ : List Ty} {t : Ty} {e : Expr ℝ Γ t} (h : e.IsConstant)
    (read read' : Reader ℝ) (env env' : Env ℝ Γ) :
    e.eval Arithmetic.ofOps read env = e.eval Arithmetic.ofOps read' env' := by
  induction e with
  | scalar => rfl
  | binary op x y hx hy => simp only [eval, hx h.1 env env', hy h.2 env env']
  | neg x hx => simp only [eval, hx h env env']
  | _ => exact h.elim

/-- At every step that the evaluation of `e` at `env` with `read` reaches, every scalar division
has a nonzero divisor and every scalar comparison has unequal operands or two constant operands. A
zero divisor or a tie of operands that are not both constant is rejected even when the result does
not depend on the inputs nearby, as for `0 / x` at `x = 0` or `acc < 0` at the first step of a fold
from `0`. -/
def Regular (read : Reader ℝ) : {Γ : List Ty} → {t : Ty} → Expr ℝ Γ t → Env ℝ Γ → Prop
  | _, _, .scalar _, _ => True
  | _, _, .index _, _ => True
  | _, _, .predicate _, _ => True
  | _, _, .var _, _ => True
  | _, _, .load _ i, env => i.Regular read env
  | _, _, .binary op x y, env => x.Regular read env ∧
      ∀ a, x.eval Arithmetic.ofOps read env = .ok a → y.Regular read env ∧
        (op = .div → ∀ b, y.eval Arithmetic.ofOps read env = .ok b → b ≠ 0)
  | _, _, .neg x, env => x.Regular read env
  | _, _, .indexBinary _ x y, env => x.Regular read env ∧
      ∀ a, x.eval Arithmetic.ofOps read env = .ok a → y.Regular read env
  | _, _, .compare _ x y, env => x.Regular read env ∧
      ∀ a, x.eval Arithmetic.ofOps read env = .ok a → y.Regular read env ∧
        ∀ b, y.eval Arithmetic.ofOps read env = .ok b → a ≠ b ∨ (x.IsConstant ∧ y.IsConstant)
  | _, _, .indexCompare _ x y, env => x.Regular read env ∧
      ∀ a, x.eval Arithmetic.ofOps read env = .ok a → y.Regular read env
  | _, _, .cond p yes no, env => p.Regular read env ∧
      (p.eval Arithmetic.ofOps read env = .ok true → yes.Regular read env) ∧
      (p.eval Arithmetic.ofOps read env = .ok false → no.Regular read env)
  | _, _, .letIn value body, env => value.Regular read env ∧
      ∀ v, value.eval Arithmetic.ofOps read env = .ok v → body.Regular read (env.push v)
  | _, _, .fold count initial body, env => count.Regular read env ∧
      ∀ c, count.eval Arithmetic.ofOps read env = .ok c → initial.Regular read env ∧
        ∀ a, initial.eval Arithmetic.ofOps read env = .ok a → ∀ n < c.toNat, ∀ acc,
        iterate (fun i acc => body.eval Arithmetic.ofOps read
          (Env.push (t := .scalar) acc (Env.push (t := .index) i env))) n 0 a = .ok acc →
        body.Regular read (Env.push (t := .scalar) acc (Env.push (t := .index) n.toUInt64 env))

/-- Returning a value in `Except` is the successful result. -/
@[simp] theorem except_pure {β : Type} (a : β) : (pure a : Except Error β) = .ok a := rfl

/-- A successful fold of `m` steps has successful prefixes of every length `n ≤ m`. -/
theorem iterate_exists_ok_of_le {β : Type} {f : UInt64 → β → Except Error β} {m n : Nat}
    {i : UInt64} {a y : β} (h : iterate f m i a = .ok y) (hn : n ≤ m) :
    ∃ z, iterate f n i a = .ok z := by
  induction m generalizing n i a with
  | zero =>
      rw [Nat.le_zero.mp hn]
      exact ⟨a, rfl⟩
  | succ m ih =>
      cases n with
      | zero => exact ⟨a, rfl⟩
      | succ n =>
          simp only [iterate] at h ⊢
          obtain ⟨b, hb, h⟩ := except_bind_eq_ok.mp h
          obtain ⟨z, hz⟩ := ih h (Nat.le_of_succ_le_succ hn)
          exact ⟨z, except_bind_eq_ok.mpr ⟨b, hb, hz⟩⟩

variable {α : Type}

/-- The evaluation of an expression that does not read the variable `v` does not depend on the
value of `v`. -/
theorem eval_congr_of_reads (arithmetic : Arithmetic α) (read : Reader α) {Γ : List Ty} {t : Ty}
    (e : Expr α Γ t) {s : Ty} (v : Var Γ s) (hv : e.reads v = false) {env env' : Env α Γ}
    (h : ∀ {s' : Ty} (w : Var Γ s'), v.same w = false → env.get w = env'.get w) :
    e.eval arithmetic read env = e.eval arithmetic read env' := by
  induction e with
  | scalar | index | predicate => rfl
  | var w => simp only [eval, h w hv]
  | load j i ih => simp only [eval, ih v hv h]
  | binary op x y hx hy =>
      simp only [reads, Bool.or_eq_false_iff] at hv
      simp only [eval, hx v hv.1 h, hy v hv.2 h]
  | neg x hx => simp only [eval, hx v hv h]
  | indexBinary op x y hx hy =>
      simp only [reads, Bool.or_eq_false_iff] at hv
      simp only [eval, hx v hv.1 h, hy v hv.2 h]
  | compare op x y hx hy =>
      simp only [reads, Bool.or_eq_false_iff] at hv
      simp only [eval, hx v hv.1 h, hy v hv.2 h]
  | indexCompare op x y hx hy =>
      simp only [reads, Bool.or_eq_false_iff] at hv
      simp only [eval, hx v hv.1 h, hy v hv.2 h]
  | cond p yes no hp hyes hno =>
      simp only [reads, Bool.or_eq_false_iff] at hv
      simp only [eval, hp v hv.1.1 h, hyes v hv.1.2 h, hno v hv.2 h]
  | letIn value body hval hbody =>
      simp only [reads, Bool.or_eq_false_iff] at hv
      simp only [eval, hval v hv.1 h]
      congr 1
      funext a
      refine hbody v.succ hv.2 fun w hw => ?_
      cases w with
      | zero => rfl
      | succ w => exact h w hw
  | fold count initial body hc hi hb =>
      simp only [reads, Bool.or_eq_false_iff] at hv
      simp only [eval, hc v hv.1.1 h, hi v hv.1.2 h]
      congr 1
      funext c
      congr 1
      funext a
      apply iterate_congr
      intro i acc
      refine hb v.succ.succ hv.2 fun w hw => ?_
      cases w with
      | zero => rfl
      | succ w =>
          cases w with
          | zero => rfl
          | succ w => exact h w hw

/-- In a fold step of the dual scope, the primal variables hold the values of the step of the
original scope, given that they do so in the enclosing scopes. -/
theorem primal_push_push {Γ : List Ty} {env : Env α Γ} {denv : Env α (dualScope Γ)}
    (h : ∀ {s : Ty} (v : Var Γ s), denv.get v.primal = env.get v) (i : UInt64) (acc d : α) :
    ∀ {s : Ty} (v : Var (.scalar :: .index :: Γ) s),
      (Env.push (t := .scalar) d (Env.push (t := .scalar) acc (Env.push (t := .index) i denv)) :
        Env α (dualScope (.scalar :: .index :: Γ))).get v.primal =
      (Env.push (t := .scalar) acc (Env.push (t := .index) i env)).get v
  | _, .zero => rfl
  | _, .succ .zero => rfl
  | _, .succ (.succ v) => h v

/-- A step of a tangent fold evaluates the tangent of the body with the primal accumulator, which
is the value of the primal fold over the earlier steps. -/
theorem eval_foldTangent (arithmetic : Arithmetic α) (read : Reader α) {Γ : List Ty}
    (initial : Expr α (dualScope Γ) .scalar)
    (body body' : Expr α (.scalar :: .scalar :: .index :: dualScope Γ) .scalar)
    {denv : Env α (dualScope Γ)} {n : Nat} (hn : n.toUInt64.toNat = n) {a g d : α}
    (ha : initial.eval arithmetic read denv = .ok a)
    (hg : iterate (fun i acc => body.eval arithmetic read (Env.push (t := .scalar) acc
      (Env.push (t := .scalar) acc (Env.push (t := .index) i denv)))) n 0 a = .ok g) :
    (foldTangent initial body body').eval arithmetic read
        (Env.push (t := .scalar) d (Env.push (t := .index) n.toUInt64 denv)) =
      body'.eval arithmetic read (Env.push (t := .scalar) d (Env.push (t := .scalar) g
        (Env.push (t := .index) n.toUInt64 denv))) := by
  unfold foldTangent
  split_ifs with hr
  · simp only [eval, Env.get_push_zero, Env.get_push_succ, ← except_pure, pure_bind, eval_rename,
      Env.pull_push, Env.pull_succ_push, Env.pull_succ_succ_push, ha, hn, hg]
  · simp only [eval, Env.get_push_zero, ← except_pure, pure_bind]
    refine eval_congr_of_reads arithmetic read body' (Var.succ Var.zero)
      (Bool.eq_false_iff.mpr hr) fun w hw => ?_
    cases w with
    | zero => rfl
    | succ w =>
        cases w with
        | zero => simp only [Var.same, Bool.true_eq_false] at hw
        | succ w => rfl

/-- On the reals, a comparison has the same result at two pairs in the same strict order. -/
theorem compare_eq_of_lt {a b c d : ℝ} (h : (a < b ∧ c < d) ∨ (b < a ∧ d < c)) (op : Compare) :
    (Arithmetic.ofOps : Arithmetic ℝ).compare op a b =
      (Arithmetic.ofOps : Arithmetic ℝ).compare op c d := by
  rcases h with ⟨h1, h2⟩ | ⟨h1, h2⟩ <;> cases op <;>
    simp [Arithmetic.ofOps, h1, h2, h1.ne, h2.ne, h1.le, h2.le, h1.ne', h2.ne', not_lt_of_gt h1,
      not_lt_of_gt h2, not_le_of_gt h1, not_le_of_gt h2]

variable {P : Type} [NormedAddCommGroup P] [NormedSpace ℝ P] {θ₀ : P}

/-- The local behaviour at `θ₀` of an evaluation `F` with tangent evaluation `T`: a scalar
evaluation succeeds near `θ₀` with a value that has a derivative `D` at `θ₀` and `T u` is `D u`,
and an index or a predicate evaluation is constant near `θ₀` and `T u` is that constant. -/
private def Good (θ₀ : P) : (t : Ty) → (P → Except Error (t.Value ℝ)) →
    (P → Except Error (t.Value ℝ)) → Prop
  | .scalar, F, T => ∃ (g : P → ℝ) (D : P →L[ℝ] ℝ), (∀ᶠ θ in 𝓝 θ₀, F θ = .ok (g θ)) ∧
      HasFDerivAt g D θ₀ ∧ ∀ u, T u = .ok (D u)
  | .index, F, T => ∃ y, (∀ᶠ θ in 𝓝 θ₀, F θ = .ok y) ∧ ∀ u, T u = .ok y
  | .predicate, F, T => ∃ y, (∀ᶠ θ in 𝓝 θ₀, F θ = .ok y) ∧ ∀ u, T u = .ok y

/-- `Good` depends on the evaluation only near `θ₀` and on the tangent evaluation only through
its values. -/
private theorem Good.congr {t : Ty} {F F' T T' : P → Except Error (t.Value ℝ)}
    (h : Good θ₀ t F T) (hF : ∀ᶠ θ in 𝓝 θ₀, F' θ = F θ) (hT : ∀ u, T' u = T u) :
    Good θ₀ t F' T' := by
  cases t with
  | scalar =>
      obtain ⟨g, D, hg, hD, hTD⟩ := h
      exact ⟨g, D, (hF.and hg).mono fun θ h => h.1.trans h.2, hD, fun u => (hT u).trans (hTD u)⟩
  | index =>
      obtain ⟨y, hy, hTy⟩ := h
      exact ⟨y, (hF.and hy).mono fun θ h => h.1.trans h.2, fun u => (hT u).trans (hTy u)⟩
  | predicate =>
      obtain ⟨y, hy, hTy⟩ := h
      exact ⟨y, (hF.and hy).mono fun θ h => h.1.trans h.2, fun u => (hT u).trans (hTy u)⟩

/-- The environments of the induction. `env θ` is the environment at the parameter `θ`, and
`denv u` is the dual environment of the direction `u`: its primal variables hold the values at
`θ₀`, its tangent variables the derivatives at `θ₀` applied to `u`, and the index and predicate
variables do not depend on `θ`. -/
private structure Tracks (θ₀ : P) {Γ : List Ty} (env : P → Env ℝ Γ)
    (denv : P → Env ℝ (dualScope Γ)) : Prop where
  primal : ∀ u {s : Ty} (v : Var Γ s), (denv u).get v.primal = (env θ₀).get v
  tangent : ∀ v : Var Γ .scalar, ∃ D : P →L[ℝ] ℝ,
    HasFDerivAt (fun θ => (env θ).get v) D θ₀ ∧ ∀ u, (denv u).get v.tangent = D u
  index : ∀ (v : Var Γ .index) θ, (env θ).get v = (env θ₀).get v
  predicate : ∀ (v : Var Γ .predicate) θ, (env θ).get v = (env θ₀).get v

/-- The induction behind `Expr.exists_hasFDerivAt_tangent`, over the expression and the
environments of `Tracks`. -/
private theorem good {read : Reader ℝ} {k : Nat} (hk : ∀ j i y, read j i = .ok y → j < k)
    {x : P → Nat → UInt64 → ℝ} {x' : Nat → UInt64 → P →L[ℝ] ℝ}
    (hx : ∀ j i y, read j i = .ok y → HasFDerivAt (fun θ => x θ j i) (x' j i) θ₀)
    {Γ : List Ty} {t : Ty} (e : Expr ℝ Γ t) {env : P → Env ℝ Γ}
    {denv : P → Env ℝ (dualScope Γ)} (htr : Tracks θ₀ env denv)
    (hok : ∃ y, e.eval Arithmetic.ofOps (read.withValues (x θ₀)) (env θ₀) = .ok y)
    (hreg : e.Regular (read.withValues (x θ₀)) (env θ₀)) :
    Good θ₀ t (fun θ => e.eval Arithmetic.ofOps (read.withValues (x θ)) (env θ))
      (fun u => (e.tangent k).eval Arithmetic.ofOps (read.dual k (x θ₀) fun j i => x' j i u)
        (denv u)) := by
  have hread : ∀ (u : P) j i y, read.withValues (x θ₀) j i = .ok y →
      read.dual k (x θ₀) (fun j i => x' j i u) j i = .ok y :=
    fun u j i y h => Reader.dual_eq_ok hk _ _ h
  induction e with
  | scalar c =>
      refine ⟨fun _ => c, 0, Eventually.of_forall fun θ => rfl, hasFDerivAt_const c θ₀,
        fun u => ?_⟩
      simp only [tangent, eval, _root_.zero_apply]
  | index c => exact ⟨c, Eventually.of_forall fun θ => rfl, fun u => rfl⟩
  | predicate c => exact ⟨c, Eventually.of_forall fun θ => rfl, fun u => rfl⟩
  | @var _ t v =>
      cases t with
      | scalar =>
          obtain ⟨D, hD, hDu⟩ := htr.tangent v
          exact ⟨fun θ => (env θ).get v, D, Eventually.of_forall fun θ => rfl, hD,
            fun u => by simp only [tangent, eval, hDu]⟩
      | index =>
          exact ⟨(env θ₀).get v, Eventually.of_forall fun θ => by simp only [eval, htr.index v θ],
            fun u => by simp only [tangent, eval, htr.primal u v]⟩
      | predicate =>
          exact ⟨(env θ₀).get v,
            Eventually.of_forall fun θ => by simp only [eval, htr.predicate v θ],
            fun u => by simp only [tangent, eval, htr.primal u v]⟩
  | load j i ih =>
      obtain ⟨y, hy⟩ := hok
      simp only [eval] at hy
      obtain ⟨a, ha, hy⟩ := except_bind_eq_ok.mp hy
      obtain ⟨b, hb, -⟩ := ih htr ⟨a, ha⟩ hreg
      rw [Except.ok.inj (ha.symm.trans hb.self_of_nhds)] at hy
      obtain ⟨z, hz, -⟩ := Reader.exists_ok_of_withValues hy
      refine ⟨fun θ => x θ j b, x' j b, hb.mono fun θ hθ => ?_, hx j b z hz, fun u => ?_⟩
      · simp only [eval, hθ, ← except_pure, pure_bind, Reader.withValues_of_ok hz]
      · simp only [tangent, eval, eval_primal _ (hread u) i (htr.primal u) hb.self_of_nhds,
          ← except_pure, pure_bind, Reader.dual_add, Reader.withValues_of_ok hz]
  | binary op e₁ e₂ ih₁ ih₂ =>
      obtain ⟨y, hy⟩ := hok
      simp only [eval] at hy
      obtain ⟨a, ha, hy⟩ := except_bind_eq_ok.mp hy
      obtain ⟨b, hb, -⟩ := except_bind_eq_ok.mp hy
      obtain ⟨hreg₁, hreg₂⟩ := hreg
      obtain ⟨hreg₂, hdiv⟩ := hreg₂ a ha
      obtain ⟨g₁, D₁, hF₁, hg₁, hT₁⟩ := ih₁ htr ⟨a, ha⟩ hreg₁
      obtain ⟨g₂, D₂, hF₂, hg₂, hT₂⟩ := ih₂ htr ⟨b, hb⟩ hreg₂
      have hp₁ : ∀ u, e₁.primal.eval Arithmetic.ofOps (read.dual k (x θ₀) fun j i => x' j i u)
          (denv u) = .ok (g₁ θ₀) :=
        fun u => eval_primal _ (hread u) e₁ (htr.primal u) hF₁.self_of_nhds
      have hp₂ : ∀ u, e₂.primal.eval Arithmetic.ofOps (read.dual k (x θ₀) fun j i => x' j i u)
          (denv u) = .ok (g₂ θ₀) :=
        fun u => eval_primal _ (hread u) e₂ (htr.primal u) hF₂.self_of_nhds
      have hF : ∀ᶠ θ in 𝓝 θ₀, (binary op e₁ e₂).eval Arithmetic.ofOps (read.withValues (x θ))
          (env θ) = .ok ((Arithmetic.ofOps : Arithmetic ℝ).binary op (g₁ θ) (g₂ θ)) :=
        (hF₁.and hF₂).mono fun θ h => by simp only [eval, h.1, h.2, ← except_pure, pure_bind]
      cases op with
      | add =>
          refine ⟨fun θ => g₁ θ + g₂ θ, D₁ + D₂, hF, hg₁.add hg₂, fun u => ?_⟩
          simp only [tangent, eval, hT₁ u, hT₂ u, ← except_pure, pure_bind,
            Arithmetic.ofOps_binary_add, _root_.add_apply]
      | sub =>
          refine ⟨fun θ => g₁ θ - g₂ θ, D₁ - D₂, hF, hg₁.sub hg₂, fun u => ?_⟩
          simp only [tangent, eval, hT₁ u, hT₂ u, ← except_pure, pure_bind,
            Arithmetic.ofOps_binary_sub, _root_.sub_apply]
      | mul =>
          refine ⟨fun θ => g₁ θ * g₂ θ, g₁ θ₀ • D₂ + g₂ θ₀ • D₁, hF, hg₁.fun_mul hg₂,
            fun u => ?_⟩
          simp only [tangent, eval, hT₁ u, hT₂ u, hp₁ u, hp₂ u, ← except_pure, pure_bind,
            Arithmetic.ofOps_binary_add, Arithmetic.ofOps_binary_mul,
            _root_.add_apply, _root_.smul_apply, smul_eq_mul]
          congr 1
          ring
      | div =>
          have hne : g₂ θ₀ ≠ 0 := hdiv rfl (g₂ θ₀) hF₂.self_of_nhds
          have hinv : HasFDerivAt (fun θ => (g₂ θ)⁻¹)
              ((ContinuousLinearMap.toSpanSingleton ℝ (-(g₂ θ₀ ^ 2)⁻¹)).comp D₂) θ₀ :=
            (hasFDerivAt_inv hne).comp θ₀ hg₂
          refine ⟨fun θ => g₁ θ * (g₂ θ)⁻¹, _,
            hF.mono fun θ h => h.trans (by rw [Arithmetic.ofOps_binary_div, div_eq_mul_inv]),
            hg₁.fun_mul hinv, fun u => ?_⟩
          simp only [tangent, eval, hT₁ u, hT₂ u, hp₁ u, hp₂ u, ← except_pure, pure_bind,
            Arithmetic.ofOps_binary_sub, Arithmetic.ofOps_binary_mul,
            Arithmetic.ofOps_binary_div, _root_.add_apply,
            _root_.smul_apply, ContinuousLinearMap.comp_apply,
            ContinuousLinearMap.toSpanSingleton_apply, smul_eq_mul]
          congr 1
          field_simp
          ring
  | neg e₁ ih₁ =>
      obtain ⟨y, hy⟩ := hok
      simp only [eval] at hy
      obtain ⟨a, ha, -⟩ := except_bind_eq_ok.mp hy
      obtain ⟨g₁, D₁, hF₁, hg₁, hT₁⟩ := ih₁ htr ⟨a, ha⟩ hreg
      refine ⟨fun θ => -g₁ θ, -D₁,
        hF₁.mono fun θ h => by simp only [eval, h, ← except_pure, pure_bind,
          Arithmetic.ofOps_neg], hg₁.neg, fun u => ?_⟩
      simp only [tangent, eval, hT₁ u, ← except_pure, pure_bind, Arithmetic.ofOps_neg,
        _root_.neg_apply]
  | indexBinary op e₁ e₂ ih₁ ih₂ =>
      obtain ⟨y, hy⟩ := hok
      simp only [eval] at hy
      obtain ⟨a, ha, hy⟩ := except_bind_eq_ok.mp hy
      obtain ⟨b, hb, -⟩ := except_bind_eq_ok.mp hy
      obtain ⟨a', hF₁, -⟩ := ih₁ htr ⟨a, ha⟩ hreg.1
      obtain ⟨b', hF₂, -⟩ := ih₂ htr ⟨b, hb⟩ (hreg.2 a ha)
      have hF : ∀ᶠ θ in 𝓝 θ₀, (indexBinary op e₁ e₂).eval Arithmetic.ofOps
          (read.withValues (x θ)) (env θ) = .ok (op.eval a' b') :=
        (hF₁.and hF₂).mono fun θ h => by simp only [eval, h.1, h.2, ← except_pure, pure_bind]
      exact ⟨_, hF, fun u =>
        eval_primal _ (hread u) (indexBinary op e₁ e₂) (htr.primal u) hF.self_of_nhds⟩
  | compare op e₁ e₂ ih₁ ih₂ =>
      obtain ⟨y, hy⟩ := hok
      simp only [eval] at hy
      obtain ⟨a, ha, hy⟩ := except_bind_eq_ok.mp hy
      obtain ⟨b, hb, -⟩ := except_bind_eq_ok.mp hy
      obtain ⟨hreg₁, hreg₂⟩ := hreg
      obtain ⟨hreg₂, hne⟩ := hreg₂ a ha
      by_cases hcst : e₁.IsConstant ∧ e₂.IsConstant
      · have hF : ∀ᶠ θ in 𝓝 θ₀, (compare op e₁ e₂).eval Arithmetic.ofOps
            (read.withValues (x θ)) (env θ) =
              .ok ((Arithmetic.ofOps : Arithmetic ℝ).compare op a b) :=
          Eventually.of_forall fun θ => by
            have h₁ := (hcst.1.eval_eq (read.withValues (x θ)) (read.withValues (x θ₀)) (env θ)
              (env θ₀)).trans ha
            have h₂ := (hcst.2.eval_eq (read.withValues (x θ)) (read.withValues (x θ₀)) (env θ)
              (env θ₀)).trans hb
            simp only [eval, h₁, h₂, ← except_pure, pure_bind]
        exact ⟨_, hF, fun u =>
          eval_primal _ (hread u) (compare op e₁ e₂) (htr.primal u) hF.self_of_nhds⟩
      replace hne : ∀ b, e₂.eval Arithmetic.ofOps (read.withValues (x θ₀)) (env θ₀) = .ok b →
          a ≠ b := fun b hb => (hne b hb).resolve_right hcst
      obtain ⟨g₁, D₁, hF₁, hg₁, -⟩ := ih₁ htr ⟨a, ha⟩ hreg₁
      obtain ⟨g₂, D₂, hF₂, hg₂, -⟩ := ih₂ htr ⟨b, hb⟩ hreg₂
      have hne' : g₁ θ₀ ≠ g₂ θ₀ := by
        rw [← Except.ok.inj (ha.symm.trans hF₁.self_of_nhds)]
        exact hne _ hF₂.self_of_nhds
      have hord : ∀ᶠ θ in 𝓝 θ₀, (g₁ θ < g₂ θ ∧ g₁ θ₀ < g₂ θ₀) ∨ (g₂ θ < g₁ θ ∧ g₂ θ₀ < g₁ θ₀) := by
        rcases hne'.lt_or_gt with h | h
        · exact (hg₁.continuousAt.eventually_lt hg₂.continuousAt h).mono fun θ hθ => .inl ⟨hθ, h⟩
        · exact (hg₂.continuousAt.eventually_lt hg₁.continuousAt h).mono fun θ hθ => .inr ⟨hθ, h⟩
      have hF : ∀ᶠ θ in 𝓝 θ₀, (compare op e₁ e₂).eval Arithmetic.ofOps (read.withValues (x θ))
          (env θ) = .ok ((Arithmetic.ofOps : Arithmetic ℝ).compare op (g₁ θ₀) (g₂ θ₀)) :=
        ((hF₁.and hF₂).and hord).mono fun θ h => by
          simp only [eval, h.1.1, h.1.2, ← except_pure, pure_bind, compare_eq_of_lt h.2 op]
      exact ⟨_, hF, fun u =>
        eval_primal _ (hread u) (compare op e₁ e₂) (htr.primal u) hF.self_of_nhds⟩
  | indexCompare op e₁ e₂ ih₁ ih₂ =>
      obtain ⟨y, hy⟩ := hok
      simp only [eval] at hy
      obtain ⟨a, ha, hy⟩ := except_bind_eq_ok.mp hy
      obtain ⟨b, hb, -⟩ := except_bind_eq_ok.mp hy
      obtain ⟨a', hF₁, -⟩ := ih₁ htr ⟨a, ha⟩ hreg.1
      obtain ⟨b', hF₂, -⟩ := ih₂ htr ⟨b, hb⟩ (hreg.2 a ha)
      have hF : ∀ᶠ θ in 𝓝 θ₀, (indexCompare op e₁ e₂).eval Arithmetic.ofOps
          (read.withValues (x θ)) (env θ) = .ok (op.index a' b') :=
        (hF₁.and hF₂).mono fun θ h => by simp only [eval, h.1, h.2, ← except_pure, pure_bind]
      exact ⟨_, hF, fun u =>
        eval_primal _ (hread u) (indexCompare op e₁ e₂) (htr.primal u) hF.self_of_nhds⟩
  | cond p yes no ihp ihy ihn =>
      obtain ⟨y, hy⟩ := hok
      simp only [eval] at hy
      obtain ⟨c, hc, hy⟩ := except_bind_eq_ok.mp hy
      obtain ⟨hregp, hregy, hregn⟩ := hreg
      obtain ⟨c', hFp, -⟩ := ihp htr ⟨c, hc⟩ hregp
      have hec := Except.ok.inj (hc.symm.trans hFp.self_of_nhds)
      rw [hec] at hc hy
      have hpp : ∀ u, p.primal.eval Arithmetic.ofOps (read.dual k (x θ₀) fun j i => x' j i u)
          (denv u) = .ok c' :=
        fun u => eval_primal _ (hread u) p (htr.primal u) hc
      cases c' with
      | true =>
          exact (ihy htr ⟨y, hy⟩ (hregy hc)).congr
            (hFp.mono fun θ h => by simp only [eval, h, ← except_pure, pure_bind, Bool.cond_true])
            fun u => by simp only [tangent, eval, hpp u, ← except_pure, pure_bind, Bool.cond_true]
      | false =>
          exact (ihn htr ⟨y, hy⟩ (hregn hc)).congr
            (hFp.mono fun θ h => by simp only [eval, h, ← except_pure, pure_bind, Bool.cond_false])
            fun u => by simp only [tangent, eval, hpp u, ← except_pure, pure_bind, Bool.cond_false]
  | @letIn _ s _ value body ihv ihb =>
      obtain ⟨y, hy⟩ := hok
      simp only [eval] at hy
      obtain ⟨a, ha, hy⟩ := except_bind_eq_ok.mp hy
      obtain ⟨hregv, hregb⟩ := hreg
      cases s with
      | scalar =>
          obtain ⟨gv, Dv, hFv, hgv, hTv⟩ := ihv htr ⟨a, ha⟩ hregv
          have hea := Except.ok.inj (ha.symm.trans hFv.self_of_nhds)
          rw [hea] at ha hy
          have htr' : Tracks θ₀ (fun θ => (env θ).push (t := .scalar) (gv θ))
              (fun u => ((denv u).push (t := .scalar) (gv θ₀)).push (t := .scalar) (Dv u)) :=
            { primal := fun u {_} v => by
                cases v with
                | zero => rfl
                | succ v => exact htr.primal u v
              tangent := fun v => by
                cases v with
                | zero => exact ⟨Dv, hgv, fun u => rfl⟩
                | succ v => exact htr.tangent v
              index := fun v θ => by
                cases v with
                | succ v => exact htr.index v θ
              predicate := fun v θ => by
                cases v with
                | succ v => exact htr.predicate v θ }
          refine (ihb htr' ⟨y, hy⟩ (hregb _ ha)).congr
            (hFv.mono fun θ h => by simp only [eval, h, ← except_pure, pure_bind]) fun u => ?_
          simp only [tangent, bindDual, eval, eval_primal _ (hread u) value (htr.primal u) ha,
            ← except_pure, pure_bind, eval_rename, Env.pull_succ_push, hTv u]
          rfl
      | index =>
          obtain ⟨yv, hFv, -⟩ := ihv htr ⟨a, ha⟩ hregv
          have hea := Except.ok.inj (ha.symm.trans hFv.self_of_nhds)
          rw [hea] at ha hy
          have htr' : Tracks θ₀ (fun θ => (env θ).push (t := .index) yv)
              (fun u => (denv u).push (t := .index) yv) :=
            { primal := fun u {_} v => by
                cases v with
                | zero => rfl
                | succ v => exact htr.primal u v
              tangent := fun v => by
                cases v with
                | succ v => exact htr.tangent v
              index := fun v θ => by
                cases v with
                | zero => rfl
                | succ v => exact htr.index v θ
              predicate := fun v θ => by
                cases v with
                | succ v => exact htr.predicate v θ }
          refine (ihb htr' ⟨y, hy⟩ (hregb _ ha)).congr
            (hFv.mono fun θ h => by simp only [eval, h, ← except_pure, pure_bind]) fun u => ?_
          simp only [tangent, bindDual, eval, eval_primal _ (hread u) value (htr.primal u) ha,
            ← except_pure, pure_bind]
          rfl
      | predicate =>
          obtain ⟨yv, hFv, -⟩ := ihv htr ⟨a, ha⟩ hregv
          have hea := Except.ok.inj (ha.symm.trans hFv.self_of_nhds)
          rw [hea] at ha hy
          have htr' : Tracks θ₀ (fun θ => (env θ).push (t := .predicate) yv)
              (fun u => (denv u).push (t := .predicate) yv) :=
            { primal := fun u {_} v => by
                cases v with
                | zero => rfl
                | succ v => exact htr.primal u v
              tangent := fun v => by
                cases v with
                | succ v => exact htr.tangent v
              index := fun v θ => by
                cases v with
                | succ v => exact htr.index v θ
              predicate := fun v θ => by
                cases v with
                | zero => rfl
                | succ v => exact htr.predicate v θ }
          refine (ihb htr' ⟨y, hy⟩ (hregb _ ha)).congr
            (hFv.mono fun θ h => by simp only [eval, h, ← except_pure, pure_bind]) fun u => ?_
          simp only [tangent, bindDual, eval, eval_primal _ (hread u) value (htr.primal u) ha,
            ← except_pure, pure_bind]
          rfl
  | fold count initial body ihc ihi ihb =>
      obtain ⟨y, hy⟩ := hok
      simp only [eval] at hy
      obtain ⟨c, hc, hy⟩ := except_bind_eq_ok.mp hy
      obtain ⟨a, ha, hy⟩ := except_bind_eq_ok.mp hy
      obtain ⟨hregc, hregi⟩ := hreg
      obtain ⟨hregi, hregb⟩ := hregi c hc
      obtain ⟨c', hFc, -⟩ := ihc htr ⟨c, hc⟩ hregc
      obtain ⟨g₀, D₀, hF₀, hg₀, hT₀⟩ := ihi htr ⟨a, ha⟩ hregi
      have hec := Except.ok.inj (hc.symm.trans hFc.self_of_nhds)
      rw [hec] at hc hy hregb
      have hea := Except.ok.inj (ha.symm.trans hF₀.self_of_nhds)
      rw [hea] at ha hy
      have hi₀ : ∀ u, initial.primal.eval Arithmetic.ofOps
          (read.dual k (x θ₀) fun j i => x' j i u) (denv u) = .ok (g₀ θ₀) :=
        fun u => eval_primal _ (hread u) initial (htr.primal u) ha
      have claim : ∀ n, n ≤ c'.toNat → ∃ (g : P → ℝ) (D : P →L[ℝ] ℝ),
          (∀ᶠ θ in 𝓝 θ₀, iterate (fun i acc => body.eval Arithmetic.ofOps
            (read.withValues (x θ))
            (Env.push (t := .scalar) acc (Env.push (t := .index) i (env θ)))) n 0 (g₀ θ) =
              .ok (g θ)) ∧
          HasFDerivAt g D θ₀ ∧
          ∀ u, iterate (fun i acc => (foldTangent initial.primal body.primal
            (body.tangent k)).eval Arithmetic.ofOps (read.dual k (x θ₀) fun j i => x' j i u)
            (Env.push (t := .scalar) acc (Env.push (t := .index) i (denv u)))) n 0 (D₀ u) =
              .ok (D u) := by
        intro n
        induction n with
        | zero => exact fun _ => ⟨g₀, D₀, Eventually.of_forall fun θ => rfl, hg₀, fun u => rfl⟩
        | succ n ihn =>
            intro hn
            obtain ⟨g, D, hF, hg, hT⟩ := ihn (Nat.le_of_succ_le hn)
            have hlt : n < c'.toNat := hn
            have hn' : n.toUInt64.toNat = n := by
              rw [Nat.toUInt64_eq]
              exact UInt64.toNat_ofNat_of_lt' (lt_trans hlt c'.toNat_lt_size)
            obtain ⟨z, hz⟩ := iterate_exists_ok_of_le hy hn
            rw [iterate_succ, hF.self_of_nhds, UInt64.zero_add] at hz
            obtain ⟨_, hw, hz⟩ := except_bind_eq_ok.mp hz
            cases hw
            have htr₂ : Tracks θ₀
                (fun θ => Env.push (t := .scalar) (g θ)
                  (Env.push (t := .index) n.toUInt64 (env θ)))
                (fun u => Env.push (t := .scalar) (D u) (Env.push (t := .scalar) (g θ₀)
                  (Env.push (t := .index) n.toUInt64 (denv u)))) :=
              { primal := fun u {_} v => primal_push_push (htr.primal u) n.toUInt64 (g θ₀) (D u) v
                tangent := fun v => by
                  cases v with
                  | zero => exact ⟨D, hg, fun u => rfl⟩
                  | succ v =>
                      cases v with
                      | succ v => exact htr.tangent v
                index := fun v θ => by
                  cases v with
                  | succ v =>
                      cases v with
                      | zero => rfl
                      | succ v => exact htr.index v θ
                predicate := fun v θ => by
                  cases v with
                  | succ v =>
                      cases v with
                      | succ v => exact htr.predicate v θ }
            obtain ⟨g', D', hF', hg', hT'⟩ :=
              ihb htr₂ ⟨z, hz⟩ (hregb (g₀ θ₀) ha n hlt (g θ₀) hF.self_of_nhds)
            refine ⟨g', D', (hF.and hF').mono fun θ h => ?_, hg', fun u => ?_⟩
            · rw [iterate_succ, h.1, UInt64.zero_add]
              exact except_bind_eq_ok.mpr ⟨g θ, rfl, h.2⟩
            · rw [iterate_succ, hT u, UInt64.zero_add]
              refine except_bind_eq_ok.mpr ⟨D u, rfl, ?_⟩
              rw [← hT' u]
              exact eval_foldTangent _ _ initial.primal body.primal (body.tangent k) hn' (hi₀ u)
                (iterate_eq_ok_of_imp (fun i acc b hb => eval_primal _ (hread u) body
                  (primal_push_push (htr.primal u) i acc acc) hb) n 0 (g₀ θ₀) hF.self_of_nhds)
      obtain ⟨g, D, hF, hg, hT⟩ := claim c'.toNat le_rfl
      refine ⟨g, D, (hFc.and (hF₀.and hF)).mono fun θ h => ?_, hg, fun u => ?_⟩
      · simp only [eval]
        exact except_bind_eq_ok.mpr ⟨c', h.1, except_bind_eq_ok.mpr ⟨g₀ θ, h.2.1, h.2.2⟩⟩
      · simp only [tangent, eval]
        exact except_bind_eq_ok.mpr ⟨c', eval_primal _ (hread u) count (htr.primal u) hc,
          except_bind_eq_ok.mpr ⟨D₀ u, hT₀ u, hT u⟩⟩

/-- Over the reals, suppose the evaluation of `e` with the inputs `x θ₀` succeeds and is regular,
`read` succeeds only below operand `k`, and each input entry `x θ j i` has the derivative
`x' j i` at `θ₀`. Then the value of `e` with the inputs `x θ` succeeds near `θ₀` and has a
derivative `D` at `θ₀`, and the tangent expression computes `D u` from the primal inputs `x θ₀`
and the input tangents `x' j i u`. -/
theorem exists_hasFDerivAt_tangent (e : Expr ℝ [.index] .scalar) {k : Nat} {read : Reader ℝ}
    (hk : ∀ j i y, read j i = .ok y → j < k) {x : P → Nat → UInt64 → ℝ}
    {x' : Nat → UInt64 → P →L[ℝ] ℝ}
    (hx : ∀ j i y, read j i = .ok y → HasFDerivAt (fun θ => x θ j i) (x' j i) θ₀)
    {index : UInt64} {y : ℝ}
    (hy : evaluate Arithmetic.ofOps e (read.withValues (x θ₀)) index = .ok y)
    (hreg : e.Regular (read.withValues (x θ₀)) (Env.output index)) :
    ∃ (f : P → ℝ) (D : P →L[ℝ] ℝ),
      (∀ᶠ θ in 𝓝 θ₀, evaluate Arithmetic.ofOps e (read.withValues (x θ)) index = .ok (f θ)) ∧
      HasFDerivAt f D θ₀ ∧
      ∀ u, evaluate Arithmetic.ofOps (e.tangent k) (read.dual k (x θ₀) fun j i => x' j i u)
        index = .ok (D u) := by
  have htr : Tracks θ₀ (fun _ => Env.output index)
      (fun _ => (Env.output index : Env ℝ (dualScope [.index]))) :=
    { primal := fun u {_} v => by
        cases v with
        | zero => rfl
        | succ v => cases v
      tangent := fun v => by
        cases v with
        | succ v => cases v
      index := fun v θ => rfl
      predicate := fun v θ => by
        cases v with
        | succ v => cases v }
  obtain ⟨f, D, hf, hD, hT⟩ := good hk hx e htr ⟨y, hy⟩ hreg
  exact ⟨f, D, hf, hD, hT⟩

/-- Over the reals, if the value of `e` with the inputs `x θ` is `f θ` near `θ₀`, the evaluation
at `θ₀` is regular, `read` succeeds only below operand `k` and each input entry `x θ j i` has the
derivative `x' j i` at `θ₀`, then `f` has a derivative `D` at `θ₀` and the tangent expression
computes `D u` from the primal inputs `x θ₀` and the input tangents `x' j i u`. -/
theorem hasFDerivAt_evaluate_tangent (e : Expr ℝ [.index] .scalar) {k : Nat} {read : Reader ℝ}
    (hk : ∀ j i y, read j i = .ok y → j < k) {x : P → Nat → UInt64 → ℝ}
    {x' : Nat → UInt64 → P →L[ℝ] ℝ}
    (hx : ∀ j i y, read j i = .ok y → HasFDerivAt (fun θ => x θ j i) (x' j i) θ₀)
    {index : UInt64} {f : P → ℝ}
    (hf : ∀ᶠ θ in 𝓝 θ₀, evaluate Arithmetic.ofOps e (read.withValues (x θ)) index = .ok (f θ))
    (hreg : e.Regular (read.withValues (x θ₀)) (Env.output index)) :
    ∃ D : P →L[ℝ] ℝ, HasFDerivAt f D θ₀ ∧
      ∀ u, evaluate Arithmetic.ofOps (e.tangent k) (read.dual k (x θ₀) fun j i => x' j i u)
        index = .ok (D u) := by
  obtain ⟨f', D, hf', hD, hT⟩ := exists_hasFDerivAt_tangent e hk hx hf.self_of_nhds hreg
  exact ⟨D, hD.congr_of_eventuallyEq
    ((hf.and hf').mono fun θ h => Except.ok.inj (h.1.symm.trans h.2)), hT⟩

/-- Over the reals, the tangent expression run with the primal inputs `x` and the tangent inputs
`dx` computes the derivative at `0` of the value `f s` of `e` on the line of inputs `x + s dx`,
where the evaluation at `x` is regular and `read` succeeds only below operand `k`. -/
theorem hasDerivAt_evaluate_tangent (e : Expr ℝ [.index] .scalar) {k : Nat} {read : Reader ℝ}
    (hk : ∀ j i y, read j i = .ok y → j < k) (x dx : Nat → UInt64 → ℝ) {index : UInt64}
    {f : ℝ → ℝ}
    (hf : ∀ᶠ s in 𝓝 0, evaluate Arithmetic.ofOps e
      (read.withValues fun j i => x j i + s * dx j i) index = .ok (f s))
    (hreg : e.Regular (read.withValues x) (Env.output index)) :
    ∃ d, evaluate Arithmetic.ofOps (e.tangent k) (read.dual k x dx) index = .ok d ∧
      HasDerivAt f d 0 := by
  have hx : ∀ j i (y : ℝ), read j i = .ok y → HasFDerivAt (fun s : ℝ => x j i + s * dx j i)
      (ContinuousLinearMap.toSpanSingleton ℝ (dx j i)) 0 := fun j i _ _ =>
    hasFDerivAt_iff_hasDerivAt.mpr (by
      rw [ContinuousLinearMap.toSpanSingleton_apply_one]
      exact (((hasDerivAt_id' (0 : ℝ)).mul_const (dx j i)).const_add (x j i)).congr_deriv
        (one_mul _))
  obtain ⟨D, hD, hT⟩ := hasFDerivAt_evaluate_tangent e hk hx hf
    (by simpa only [zero_mul, add_zero] using hreg)
  refine ⟨D 1, ?_, hD.hasDerivAt⟩
  simpa only [zero_mul, add_zero, ContinuousLinearMap.toSpanSingleton_apply_one] using hT 1

end Expr

end TorchLean.Kernel
