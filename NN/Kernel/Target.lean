/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Kernel.Expr

/-!
# Structured kernel target

Primitive operands are values, not computations. Bindings therefore fix the order of reads and
arithmetic before CUDA rendering. Branches and loops retain their own blocks, so lowering cannot
hoist a failing read out of a branch or reassociate a sequential accumulation.

The target evaluator specifies these blocks independently of the source evaluator. Its lowering
theorem is about this representation, not NVIDIA's compiler or execution of rendered text.
-/

@[expose] public section

namespace NN.Kernel.Target

namespace Internal

/-- Incrementing below an unsigned loop bound cannot wrap, even at the largest bound. -/
theorem increment_toNat {i count : UInt64} (h : i < count) :
    (i + 1).toNat = i.toNat + 1 := by
  rw [UInt64.toNat_add, UInt64.toNat_one, Nat.mod_eq_of_lt]
  have hi := UInt64.lt_iff_toNat_lt.mp h
  have hc := UInt64.toNat_lt count
  omega

end Internal

/-- An unsigned counter loop, stopping before `count` and at the first failed step.

The strict guard ensures that incrementing the counter cannot wrap, even when `count` is the
largest representable index. Starting at or beyond the bound leaves the accumulator unchanged.
-/
def loop {α : Type} (step : UInt64 → α → Except Error α) (count i : UInt64)
    (acc : α) : Except Error α :=
  if h : i < count then do
    let acc ← step i acc
    loop step count (i + 1) acc
  else .ok acc
termination_by count.toNat - i.toNat
decreasing_by
  have hi := UInt64.lt_iff_toNat_lt.mp h
  rw [Internal.increment_toNat h]
  omega

/-- The unsigned guard executes exactly the remaining sequential steps, including their errors. -/
theorem loop_eq_iterate {α : Type} (step : UInt64 → α → Except Error α)
    (count i : UInt64) (acc : α) :
    loop step count i acc = iterate step (count.toNat - i.toNat) i acc := by
  generalize hn : count.toNat - i.toNat = n
  induction n generalizing i acc with
  | zero =>
      have h : ¬ i < count := by
        simp only [UInt64.lt_iff_toNat_lt]
        omega
      rw [loop, dite_eq_right h, iterate]
  | succ n ih =>
      have h : i < count := by
        simp only [UInt64.lt_iff_toNat_lt]
        omega
      have hnext : count.toNat - (i + 1).toNat = n := by
        rw [Internal.increment_toNat h]
        omega
      rw [loop, dite_eq_left h, iterate]
      congr 1
      funext value
      exact ih (i + 1) value hnext

/-- Constants and typed local variables, with no effects or arithmetic. -/
inductive Atom (α : Type) : List Ty → Ty → Type where
  | scalar {Γ : List Ty} (value : α) : Atom α Γ .scalar
  | index {Γ : List Ty} (value : UInt64) : Atom α Γ .index
  | predicate {Γ : List Ty} (value : Bool) : Atom α Γ .predicate
  | var {Γ : List Ty} {t : Ty} (v : Var Γ t) : Atom α Γ t

/-- One primitive operation whose operands have already been evaluated. -/
inductive Primitive (α : Type) : List Ty → Ty → Type where
  | load {Γ : List Ty} (operand : Nat) (index : Atom α Γ .index) : Primitive α Γ .scalar
  | binary {Γ : List Ty} (op : ScalarOp) (x y : Atom α Γ .scalar) : Primitive α Γ .scalar
  | neg {Γ : List Ty} (x : Atom α Γ .scalar) : Primitive α Γ .scalar
  | indexBinary {Γ : List Ty} (op : IndexOp) (x y : Atom α Γ .index) : Primitive α Γ .index
  | compare {Γ : List Ty} (op : Compare) (x y : Atom α Γ .scalar) : Primitive α Γ .predicate
  | indexCompare {Γ : List Ty} (op : Compare) (x y : Atom α Γ .index) : Primitive α Γ .predicate

/-- Scoped statement blocks. A binding finishes its value block before entering its body. -/
inductive Block (α : Type) : List Ty → Ty → Type where
  | result {Γ : List Ty} {t : Ty} (value : Atom α Γ t) : Block α Γ t
  | compute {Γ : List Ty} {t : Ty} (operation : Primitive α Γ t) : Block α Γ t
  | bind {Γ : List Ty} {s t : Ty} (value : Block α Γ s)
      (body : Block α (s :: Γ) t) : Block α Γ t
  | branch {Γ : List Ty} {t : Ty} (p : Atom α Γ .predicate)
      (yes no : Block α Γ t) : Block α Γ t
  | loop {Γ : List Ty} (count : Atom α Γ .index) (initial : Atom α Γ .scalar)
      (body : Block α (.scalar :: .index :: Γ) .scalar) : Block α Γ .scalar

/-- Read an atomic value from the current environment. -/
def Atom.eval {α : Type} {Γ : List Ty} {t : Ty} (env : Env α Γ) :
    Atom α Γ t → t.Value α
  | .scalar x => x
  | .index i => i
  | .predicate p => p
  | .var v => env.get v

/-- Perform one operation, retaining the reader's error rather than supplying a default value. -/
def Primitive.eval {α : Type} (arithmetic : Arithmetic α) (read : Reader α)
    {Γ : List Ty} {t : Ty} (env : Env α Γ) :
    Primitive α Γ t → Except Error (t.Value α)
  | .load operand i => read operand (i.eval env)
  | .binary op x y => .ok (arithmetic.binary op (x.eval env) (y.eval env))
  | .neg x => .ok (arithmetic.neg (x.eval env))
  | .indexBinary op x y => .ok (op.eval (x.eval env) (y.eval env))
  | .compare op x y => .ok (arithmetic.compare op (x.eval env) (y.eval env))
  | .indexCompare op x y => .ok (op.index (x.eval env) (y.eval env))

/-- Sequential block semantics, with lazy branches and increasing-index loop accumulation. -/
def Block.eval {α : Type} (arithmetic : Arithmetic α) (read : Reader α)
    {Γ : List Ty} {t : Ty} (env : Env α Γ) : Block α Γ t → Except Error (t.Value α)
  | .result value => .ok (value.eval env)
  | .compute operation => operation.eval arithmetic read env
  | .bind value body => do
      let value ← value.eval arithmetic read env
      body.eval arithmetic read (env.push value)
  | .branch p yes no =>
      match p.eval env with
      | true => yes.eval arithmetic read env
      | false => no.eval arithmetic read env
  | .loop count initial body =>
      Target.loop (fun i acc => body.eval arithmetic read
        (Env.push (t := .scalar) acc (Env.push (t := .index) i env)))
        (count.eval env) 0 (initial.eval env)

/-- Move an atom's variable addresses under a new scope. -/
def Atom.rename {α : Type} {Γ Δ : List Ty} {t : Ty} (ρ : Renaming Γ Δ) :
    Atom α Γ t → Atom α Δ t
  | .scalar x => .scalar x
  | .index i => .index i
  | .predicate p => .predicate p
  | .var v => .var (ρ v)

/-- Rename only the already-evaluated operands of a primitive. -/
def Primitive.rename {α : Type} {Γ Δ : List Ty} {t : Ty} (ρ : Renaming Γ Δ) :
    Primitive α Γ t → Primitive α Δ t
  | .load operand i => .load operand (i.rename ρ)
  | .binary op x y => .binary op (x.rename ρ) (y.rename ρ)
  | .neg x => .neg (x.rename ρ)
  | .indexBinary op x y => .indexBinary op (x.rename ρ) (y.rename ρ)
  | .compare op x y => .compare op (x.rename ρ) (y.rename ρ)
  | .indexCompare op x y => .indexCompare op (x.rename ρ) (y.rename ρ)

/-- Rename a block without changing its sequencing or control-flow regions. -/
def Block.rename {α : Type} {Γ Δ : List Ty} {t : Ty} (ρ : Renaming Γ Δ) :
    Block α Γ t → Block α Δ t
  | .result value => .result (value.rename ρ)
  | .compute operation => .compute (operation.rename ρ)
  | .bind value body => .bind (value.rename ρ) (body.rename ρ.lift)
  | .branch p yes no => .branch (p.rename ρ) (yes.rename ρ) (no.rename ρ)
  | .loop count initial body => .loop (count.rename ρ) (initial.rename ρ)
      (body.rename (Renaming.lift (s := .scalar) (Renaming.lift (s := .index) ρ)))

/-- Renaming an atom is equivalent to looking up its original address in a pulled environment. -/
theorem Atom.eval_rename {α : Type} {Γ Δ : List Ty} {t : Ty}
    (value : Atom α Γ t) (ρ : Renaming Γ Δ) (env : Env α Δ) :
    (value.rename ρ).eval env = value.eval (env.pull ρ) := by
  cases value <;> rfl

/-- Renaming cannot change primitive results or failed reads. -/
theorem Primitive.eval_rename {α : Type} {Γ Δ : List Ty} {t : Ty}
    (operation : Primitive α Γ t) (arithmetic : Arithmetic α) (read : Reader α)
    (ρ : Renaming Γ Δ) (env : Env α Δ) :
    (operation.rename ρ).eval arithmetic read env =
      operation.eval arithmetic read (env.pull ρ) := by
  cases operation <;> simp only [rename, eval, Atom.eval_rename]

/-- Renaming scoped statements preserves both values and errors. -/
theorem Block.eval_rename {α : Type} {Γ : List Ty} {t : Ty} (block : Block α Γ t)
    (arithmetic : Arithmetic α) (read : Reader α) {Δ : List Ty}
    (ρ : Renaming Γ Δ) (env : Env α Δ) :
    (block.rename ρ).eval arithmetic read env = block.eval arithmetic read (env.pull ρ) := by
  induction block generalizing Δ with
  | result value => simp only [rename, eval, Atom.eval_rename]
  | compute operation => simp only [rename, eval, Primitive.eval_rename]
  | bind value body hv hb => simp only [rename, eval, hv, hb, Env.pull_push]
  | branch p yes no hy hn => simp only [rename, eval, Atom.eval_rename, hy, hn]
  | loop count initial body hb =>
      simp only [rename, eval, Atom.eval_rename, loop_eq_iterate,
        UInt64.toNat_zero, Nat.sub_zero]
      apply iterate_congr
      intro i acc
      rw [hb, Env.pull_push, Env.pull_push]

/-- Lower source expressions to blocks with atomic primitive operands. -/
def lower {α : Type} {Γ : List Ty} {t : Ty} : Expr α Γ t → Block α Γ t
  | .scalar x => .result (.scalar x)
  | .index i => .result (.index i)
  | .predicate p => .result (.predicate p)
  | .var v => .result (.var v)
  | .load operand i => .bind (lower i) (.compute (.load operand (.var .zero)))
  | .binary op x y => .bind (lower x) (.bind ((lower y).rename Var.succ)
      (.compute (.binary op (.var (.succ .zero)) (.var .zero))))
  | .neg x => .bind (lower x) (.compute (.neg (.var .zero)))
  | .indexBinary op x y => .bind (lower x) (.bind ((lower y).rename Var.succ)
      (.compute (.indexBinary op (.var (.succ .zero)) (.var .zero))))
  | .compare op x y => .bind (lower x) (.bind ((lower y).rename Var.succ)
      (.compute (.compare op (.var (.succ .zero)) (.var .zero))))
  | .indexCompare op x y => .bind (lower x) (.bind ((lower y).rename Var.succ)
      (.compute (.indexCompare op (.var (.succ .zero)) (.var .zero))))
  | .cond p yes no => .bind (lower p)
      (.branch (.var .zero) ((lower yes).rename Var.succ) ((lower no).rename Var.succ))
  | .letIn value body => .bind (lower value) (lower body)
  | .fold count initial body => .bind (lower count)
      (.bind ((lower initial).rename Var.succ)
        (.loop (.var (.succ .zero)) (.var .zero)
          ((lower body).rename
            (Renaming.lift (s := .scalar) (Renaming.lift (s := .index)
              (fun v => .succ (.succ v)))))))

private theorem pull_skip {α : Type} {Γ : List Ty} {s : Ty}
    (env : Env α Γ) (value : s.Value α) :
    (env.push value).pull Var.succ = env := rfl

/-- Lowering preserves results and failures without scalar associativity or field assumptions. -/
theorem eval_lower {α : Type} {Γ : List Ty} {t : Ty} (expr : Expr α Γ t)
    (arithmetic : Arithmetic α) (read : Reader α) (env : Env α Γ) :
    (lower expr).eval arithmetic read env = expr.eval arithmetic read env := by
  induction expr with
  | scalar | index | predicate | var => rfl
  | load operand i ih =>
      simp only [lower, Block.eval, Primitive.eval, Atom.eval, Expr.eval, ih, Env.push]
  | binary op x y hx hy =>
      simp only [lower, Block.eval, Block.eval_rename, Primitive.eval, Atom.eval,
        Expr.eval, hx, hy, Env.pull, Env.push]
      rfl
  | neg x hx =>
      simp only [lower, Block.eval, Primitive.eval, Atom.eval, Expr.eval, hx, Env.push]
      rfl
  | indexBinary op x y hx hy =>
      simp only [lower, Block.eval, Block.eval_rename, Primitive.eval, Atom.eval,
        Expr.eval, hx, hy, Env.pull, Env.push]
      rfl
  | compare op x y hx hy =>
      simp only [lower, Block.eval, Block.eval_rename, Primitive.eval, Atom.eval,
        Expr.eval, hx, hy, Env.pull, Env.push]
      rfl
  | indexCompare op x y hx hy =>
      simp only [lower, Block.eval, Block.eval_rename, Primitive.eval, Atom.eval,
        Expr.eval, hx, hy, Env.pull, Env.push]
      rfl
  | cond p yes no hp hy hn =>
      simp only [lower, Block.eval, Block.eval_rename, Atom.eval, Expr.eval,
        hp, hy, hn, pull_skip, _root_.cond]
      simp only [Env.push]
      cases p.eval arithmetic read env with
      | error error => rfl
      | ok value => cases value <;> rfl
  | letIn value body hv hb => simp only [lower, Block.eval, Expr.eval, hv, hb]
  | fold count initial body hc hi hb =>
      simp only [lower, Block.eval, Block.eval_rename, Atom.eval, Expr.eval,
        hc, hi, hb, Env.pull_push, pull_skip, loop_eq_iterate,
        UInt64.toNat_zero, Nat.sub_zero]
      rfl

end NN.Kernel.Target
