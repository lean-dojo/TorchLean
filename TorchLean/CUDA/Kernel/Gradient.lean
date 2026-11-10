/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.CUDA.Kernel.Tangent

/-!
# Gradient kernels

`Expr.gradient k j n e` is the reverse-mode derivative of a kernel expression `e` with `n`
outputs that reads `k` inputs, for input `j`. It reads the primal inputs as operands `0` to
`k - 1` and the cotangent of the outputs as operand `k`. Entry `a` of its result is the sum over
the outputs `i < n` of the cotangent of output `i` times the derivative of output `i` in entry `a`
of input `j`. It is an expression of the kernel language, so it runs on the CPU with `Expr.eval`
and on the GPU through the same CUDA emitter.

The derivative in one entry is a seeded tangent. `Expr.seed` replaces the tangent reads of a
tangent expression by constants: the tangent of input `j` is one at address `a` and zero
elsewhere, and the tangents of the other inputs are zero. `Expr.eval_seed` shows that the seeded
expression computes the expression with these tangent inputs (`Reader.seed`), and
`Expr.evaluate_gradient` writes the gradient entry as a fold over the outputs. Each entry runs the
tangent of every output, so a gradient entry costs `n` tangent evaluations, and the gradient of an
entrywise map of `n` entries costs `n²`.

In a binary format the tangent of an output can be a NaN at every seeded address, through a
product such as `0 * ∞`, even where the output is finite, as for `x / (x * x)` when `x * x`
overflows. One such output makes every entry of the gradient a NaN, while in the tangent kernel the
NaN stays at that output.

The theorem that the gradient kernel computes the adjoint of the real derivative is in
`TorchLean.CUDA.Kernel.Gradient.FDeriv`.
-/

@[expose] public section

namespace TorchLean.Kernel

variable {α : Type}

/-- The reader of the seeded tangent inputs: operands below `k` read through `read`, operand
`k + j` reads one at address `a` and zero elsewhere, and the other operands read zero. -/
def Reader.seed [OfNat α 0] [OfNat α 1] (read : Reader α) (k j : Nat) (a : UInt64) : Reader α :=
  fun o i => if o < k then read o i else .ok (if o = k + j ∧ i = a then 1 else 0)

/-- The renaming of a tangent expression of the output scope into the step of a fold inside a
kernel: the output variable becomes the fold index. -/
def Renaming.foldIndex : Renaming (dualScope [.index]) [.scalar, .index, .index]
  | _, .zero => .succ .zero
  | _, .succ v => nomatch v

/-- Pulling the environment of a fold step back along `Renaming.foldIndex` gives the output
environment of the fold index. -/
@[simp] theorem Env.pull_foldIndex (env : Env α [.index]) (i : UInt64) (acc : α) :
    Env.pull Renaming.foldIndex (Env.push (t := .scalar) acc (Env.push (t := .index) i env)) =
      (Env.output i : Env α (dualScope [.index])) := by
  apply congrArg (@Env.mk α [.index])
  funext t v
  cases v with
  | zero => rfl
  | succ v => cases v

namespace Expr

/-- A successful evaluation gives the same result with a reader that agrees with the original
reader wherever the original read succeeds. -/
theorem eval_eq_ok_of_reader (arithmetic : Arithmetic α) {read read' : Reader α}
    (hread : ∀ j i y, read j i = .ok y → read' j i = .ok y) {Γ : List Ty} {t : Ty}
    (e : Expr α Γ t) {env : Env α Γ} {y : t.Value α}
    (h : e.eval arithmetic read env = .ok y) : e.eval arithmetic read' env = .ok y := by
  induction e with
  | scalar | index | predicate | var => exact h
  | load j i ih =>
      simp only [eval] at h ⊢
      obtain ⟨a, ha, h⟩ := except_bind_eq_ok.mp h
      rw [ih ha]
      exact hread j a y h
  | binary op x y hx hy =>
      simp only [eval] at h ⊢
      obtain ⟨a, ha, h⟩ := except_bind_eq_ok.mp h
      obtain ⟨b, hb, h⟩ := except_bind_eq_ok.mp h
      rw [hx ha, hy hb]
      exact h
  | neg x hx =>
      simp only [eval] at h ⊢
      obtain ⟨a, ha, h⟩ := except_bind_eq_ok.mp h
      rw [hx ha]
      exact h
  | indexBinary op x y hx hy =>
      simp only [eval] at h ⊢
      obtain ⟨a, ha, h⟩ := except_bind_eq_ok.mp h
      obtain ⟨b, hb, h⟩ := except_bind_eq_ok.mp h
      rw [hx ha, hy hb]
      exact h
  | compare op x y hx hy =>
      simp only [eval] at h ⊢
      obtain ⟨a, ha, h⟩ := except_bind_eq_ok.mp h
      obtain ⟨b, hb, h⟩ := except_bind_eq_ok.mp h
      rw [hx ha, hy hb]
      exact h
  | indexCompare op x y hx hy =>
      simp only [eval] at h ⊢
      obtain ⟨a, ha, h⟩ := except_bind_eq_ok.mp h
      obtain ⟨b, hb, h⟩ := except_bind_eq_ok.mp h
      rw [hx ha, hy hb]
      exact h
  | cond p yes no hp hyes hno =>
      simp only [eval] at h ⊢
      obtain ⟨b, hb, h⟩ := except_bind_eq_ok.mp h
      rw [hp hb]
      cases b
      · exact hno h
      · exact hyes h
  | letIn value body hv hb =>
      simp only [eval] at h ⊢
      obtain ⟨a, ha, h⟩ := except_bind_eq_ok.mp h
      exact except_bind_eq_ok.mpr ⟨a, hv ha, hb h⟩
  | fold count initial body hc hi hb =>
      simp only [eval] at h ⊢
      obtain ⟨c, hcount, h⟩ := except_bind_eq_ok.mp h
      obtain ⟨a, ha, h⟩ := except_bind_eq_ok.mp h
      exact except_bind_eq_ok.mpr ⟨c, hc hcount, except_bind_eq_ok.mpr ⟨a, hi ha,
        iterate_eq_ok_of_imp (fun i acc b hstep => hb hstep) c.toNat 0 a h⟩⟩

variable [OfNat α 0] [OfNat α 1]

/-- The expression with the seeded tangent inputs of `Reader.seed` at the address that `a` holds: a
read of operand `k + j` becomes one at that address and zero elsewhere, and a read of another
operand from `k` on becomes zero. The index of a replaced read is still evaluated, so the seeded
expression fails where the index fails. -/
def seed (k j : Nat) : {Γ : List Ty} → {t : Ty} → Var Γ .index → Expr α Γ t → Expr α Γ t
  | _, _, _, .scalar c => .scalar c
  | _, _, _, .index c => .index c
  | _, _, _, .predicate c => .predicate c
  | _, _, _, .var v => .var v
  | _, _, a, .load o i =>
      if o < k then .load o (seed k j a i)
      else if o = k + j then
        .cond (.indexCompare .eq (seed k j a i) (.var a)) (.scalar 1) (.scalar 0)
      else .letIn (seed k j a i) (.scalar 0)
  | _, _, a, .binary op x y => .binary op (seed k j a x) (seed k j a y)
  | _, _, a, .neg x => .neg (seed k j a x)
  | _, _, a, .indexBinary op x y => .indexBinary op (seed k j a x) (seed k j a y)
  | _, _, a, .compare op x y => .compare op (seed k j a x) (seed k j a y)
  | _, _, a, .indexCompare op x y => .indexCompare op (seed k j a x) (seed k j a y)
  | _, _, a, .cond p yes no => .cond (seed k j a p) (seed k j a yes) (seed k j a no)
  | _, _, a, .letIn value body => .letIn (seed k j a value) (seed k j a.succ body)
  | _, _, a, .fold count initial body =>
      .fold (seed k j a count) (seed k j a initial) (seed k j a.succ.succ body)

/-- The reverse-mode derivative, for input `j`, of an expression with `n` outputs that reads `k`
inputs. Entry `a` sums over the outputs `i < n` the cotangent of output `i`, read as operand `k`,
times the tangent of output `i` in the direction of entry `a` of input `j`. -/
def gradient (k j : Nat) (n : UInt64) (e : Expr α [.index] .scalar) : Expr α [.index] .scalar :=
  .fold (.index n) (.scalar 0)
    (.binary .add (.var .zero) (.binary .mul (.load k (.var (.succ .zero)))
      (((e.tangent k).rename Renaming.foldIndex).seed k j (.succ (.succ .zero)))))

/-- The seeded expression computes the expression with the seeded reader, at the address that `a`
holds. -/
theorem eval_seed (arithmetic : Arithmetic α) (read : Reader α) (k j : Nat) {Γ : List Ty}
    {t : Ty} (a : Var Γ .index) (e : Expr α Γ t) (env : Env α Γ) :
    (e.seed k j a).eval arithmetic read env =
      e.eval arithmetic (read.seed k j (env.get a)) env := by
  induction e with
  | scalar | index | predicate | var => rfl
  | load o i ih =>
      simp only [seed]
      split_ifs with h₁ h₂
      · simp only [eval, ih, Reader.seed, h₁, ↓reduceIte]
      · simp only [eval, ih]
        cases i.eval arithmetic (read.seed k j (env.get a)) env with
        | error e => rfl
        | ok b =>
            by_cases hb : b = env.get a
            · simp [Reader.seed, h₂, hb, Compare.index, Bind.bind, Except.bind, pure, Except.pure]
            · simp [Reader.seed, h₂, hb, Compare.index, Bind.bind, Except.bind, pure, Except.pure]
      · simp only [eval, ih]
        cases i.eval arithmetic (read.seed k j (env.get a)) env with
        | error e => rfl
        | ok b => simp [Reader.seed, h₁, h₂]
  | binary op x y hx hy => simp only [seed, eval, hx, hy]
  | neg x hx => simp only [seed, eval, hx]
  | indexBinary op x y hx hy => simp only [seed, eval, hx, hy]
  | compare op x y hx hy => simp only [seed, eval, hx, hy]
  | indexCompare op x y hx hy => simp only [seed, eval, hx, hy]
  | cond p yes no hp hy hn => simp only [seed, eval, hp, hy, hn]
  | letIn value body hv hb => simp only [seed, eval, hv, hb, Env.get_push_succ]
  | fold count initial body hc hi hb => simp only [seed, eval, hc, hi, hb, Env.get_push_succ]

/-- Entry `a` of the gradient kernel folds over the outputs `i < n`, adding the cotangent of
output `i` times the tangent of output `i` with the seeded reader of address `a`. -/
theorem evaluate_gradient (arithmetic : Arithmetic α) (read : Reader α) (k j : Nat) (n : UInt64)
    (e : Expr α [.index] .scalar) (a : UInt64) :
    evaluate arithmetic (e.gradient k j n) read a =
      iterate (fun i acc => do
        let c ← read k i
        let t ← evaluate arithmetic (e.tangent k) (read.seed k j a) i
        pure (arithmetic.binary .add acc (arithmetic.binary .mul c t))) n.toNat 0 0 := by
  simp only [evaluate, gradient, eval, eval_seed, eval_rename, Env.get_push_succ,
    Env.get_push_zero, Env.get_output, Env.pull_foldIndex, bind_assoc, pure_bind]
  rfl

end Expr

/-- The reverse-mode derivative, for input `j`, of a program with `n` outputs that reads `k`
inputs: it reads the primal inputs as operands `0` to `k - 1` and the cotangent of the outputs as
operand `k`, computes the gradient of input `j`, and has the evaluation of the gradient expression
as its reference. Neither count is checked: `k` must exceed every operand that the program loads,
and only the first `n` cotangent entries enter the sum, so a longer cotangent is cut to `n`. -/
def Program.gradient [OfNat α 0] [OfNat α 1] (program : Program α) (k j : Nat) (n : UInt64) :
    Program α where
  scalar := program.scalar
  arithmetic := program.arithmetic
  reference := evaluate program.arithmetic (program.expression.gradient k j n)
  expression := program.expression.gradient k j n
  correct _ _ := rfl
  native := program.native

end TorchLean.Kernel
