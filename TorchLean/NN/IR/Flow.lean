/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.IR.Program

/-!
# Branches and loops

`IR.Flow β` is a program of blocks `β` (straight-line programs, such as an IR graph with its
payload) with branches and loops. Its state is a list of values, and each node maps a state to a
state:

- `block b` runs `b` on the state, and `skip` returns it;
- `seq f g` runs `f`, then `g` on its result;
- `cond fs` reads the state's first value as an index `i` and runs branch `i` of `fs` on the rest of
  the state, and no other branch. An index that is negative or past the end selects the last branch,
  as StableHLO's `case` does (`branchOf`);
- `scan n body` runs `body` `n` times, each time on the state the previous iteration returned: a
  loop with carried state whose trip count is fixed in the node. A scanned operand or a stacked
  output is a value of the state that the body reads or writes at a counter it carries;
- `whileLoop test body` runs `body` while `test`, run on the state, returns one value that holds.

`Flow.Lang m` gives the meaning of the blocks and values in a monad `m` with string errors: how a
block runs, the branch index a value holds and whether a value holds. `Flow.eval L fuel` is the
semantics, and it is total: a `whileLoop` runs its body at most `fuel` times each time it is
entered and fails if its test still holds then, and the other nodes recurse on their parts.
`Flow.graphs α` is the language of IR graphs over the scalar `α`, run by `Graph.eval` in
`Except String`. `Flow.programs R` is the language of typed programs under the runtime `R`, run by
`Program.eval` in `StateT ConversionStatus (Except String)`, which merges the conversion flags of
every block it runs, a loop's tests included. `blocks` lists a flow's blocks in order and `mapM`
replaces each, and `mapM_blocks` proves that each block `mapM g` returns is `g` of one of them.

Proved for every language, flow, fuel and state, `eval_scan` in every monad and the others in
`Except String`:
- `eval_cond`: where every branch is defined on the operands, `cond` equals evaluating every branch
  and then selecting one (`condAll`). Where an unselected branch fails, `cond` still returns the
  selected branch's state;
- `eval_scan`: `scan n body` equals `body` unrolled `n` times (`unroll`), for every `n`;
- `eval_whileLoop`: a `whileLoop` whose test holds before each of the first `k` iterations and not
  after them, with `k ≤ fuel`, equals `body` unrolled `k` times, for every `k`;
- `eval_whileLoop_ok`: a `whileLoop` that succeeds is such a loop for some `k ≤ fuel`.

The JAX importer writes flows for a `cond` or `while` that the capture does not decide and that
gives a value the computation returns, and for a loop past the importer's count of iterations. A
`scan` whose body holds such a flow is one `scan` where every iteration would be imported alike,
except in the cases the IR README lists, and is unrolled otherwise. The PyTorch importer writes
flows for a `torch.cond` on a predicate not known at capture, a `while_loop` and a `map` whose body
does not draw, each where it gives a value the graph returns. `Flow.Follows` relates a flow's runs
in two languages, and `Flow.Reals` gives flows of real arithmetic derivatives and roundoff bounds.
-/

@[expose] public section

namespace TorchLean.IR

open FloatLib.Floats.ExecFloat (ConversionStatus)

/-- A program of blocks `β` with branches and loops, on a state of values. -/
inductive Flow (β : Type) where
  /-- The block `b` on the state. -/
  | block (b : β)
  /-- The state unchanged. -/
  | skip
  /-- `first`, then `second` on its result. -/
  | seq (first second : Flow β)
  /-- Branch `i` on the rest of the state, `i` the index of its first value (`branchOf`). -/
  | cond (branches : List (Flow β))
  /-- `body` `length` times, each time on the previous state. -/
  | scan (length : Nat) (body : Flow β)
  /-- `body` while `test` on the state returns one value that holds. -/
  | whileLoop (test body : Flow β)
  deriving Inhabited

/-- The meaning of a flow's blocks and values in the monad `m`: how a block runs, the branch
index a value holds and whether a value holds. -/
structure Flow.Lang (m : Type → Type) (β V : Type) where
  /-- The state after a block. -/
  run : β → List V → m (List V)
  /-- The branch index a value holds. -/
  index : V → m Int
  /-- Whether a value holds, for a loop's test. -/
  test : V → m Bool

namespace Flow

variable {m : Type → Type} [Monad m] [MonadExcept String m] {β γ V σ : Type}

/-- `step` `n` times from `s`, or the first error. -/
def iterate (step : σ → m σ) : Nat → σ → m σ
  | 0, s => pure s
  | n + 1, s => do iterate step n (← step s)

/-- `step` while `test` holds, at most `fuel` times, or an error where `test` still holds after
`fuel` steps. -/
def runWhile (test : σ → m Bool) (step : σ → m σ) : Nat → σ → m σ
  | 0, s => do if ← test s then throw "while: out of fuel" else pure s
  | fuel + 1, s => do if ← test s then runWhile test step fuel (← step s) else pure s

/-- The branch that index `i` selects among `n`: `i` for `0 ≤ i < n`, otherwise the last. -/
def branchOf (n : Nat) (i : Int) : Nat :=
  if 0 ≤ i ∧ i < n then i.toNat else n - 1

/-- Whether the outputs of a loop's test, which must be one value, hold. -/
def holdsOf (L : Lang m β V) : List V → m Bool
  | [v] => L.test v
  | vs => throw s!"while: the test returns {vs.length} values"

/-- Entry `k` of `outs`. -/
def pick (outs : List (List V)) (k : Nat) : m (List V) :=
  match outs[k]? with
  | some out => pure out
  | none => throw "cond: no branches"

section Semantics

variable (L : Lang m β V)

mutual

/-- The state after the flow on `s`, each `whileLoop` running its body at most `fuel` times. -/
def eval (fuel : Nat) : Flow β → List V → m (List V)
  | .block b, s => L.run b s
  | .skip, s => pure s
  | .seq f g, s => do eval fuel g (← eval fuel f s)
  | .cond _, [] => throw "cond: no index"
  | .cond fs, v :: s => do evalBranch fuel fs (branchOf fs.length (← L.index v)) s
  | .scan n body, s => iterate (eval fuel body) n s
  | .whileLoop test body, s =>
      runWhile (fun s => do holdsOf L (← eval fuel test s)) (eval fuel body) fuel s

/-- Branch `k` of `fs` on `s`. -/
def evalBranch (fuel : Nat) : List (Flow β) → Nat → List V → m (List V)
  | [], _, _ => throw "cond: no branches"
  | f :: _, 0, s => eval fuel f s
  | _ :: fs, k + 1, s => evalBranch fuel fs k s

end

/-- The IR's `cond` before this node: every branch on the operands, then the outputs of the one
the index selects. -/
def condAll (fuel : Nat) (fs : List (Flow β)) : List V → m (List V)
  | [] => throw "cond: no index"
  | v :: s => do
      let k := branchOf fs.length (← L.index v)
      pick (← fs.mapM (eval L fuel · s)) k

/-- Whether `test` holds on `s`. -/
def holds (fuel : Nat) (test : Flow β) (s : List V) : m Bool := do
  holdsOf L (← eval L fuel test s)

/-- `body` `n` times in sequence: the unrolled loop. -/
def unroll (body : Flow β) : Nat → Flow β
  | 0 => .skip
  | n + 1 => .seq body (unroll body n)

theorem iterate_eval (fuel : Nat) (body : Flow β) :
    ∀ (n : Nat) (s : List V), iterate (eval L fuel body) n s = eval L fuel (body.unroll n) s
  | 0, s => by simp [iterate, unroll, eval]
  | n + 1, s => by
      simp only [iterate, unroll, eval]
      congr 1
      funext t
      exact iterate_eval fuel body n t

/-- Branch `k` of `fs` is the flow at position `k` of `fs`, or an error where there is none. -/
theorem evalBranch_eq (fuel : Nat) :
    ∀ (fs : List (Flow β)) (k : Nat) (s : List V), evalBranch L fuel fs k s =
      match fs[k]? with
      | some f => eval L fuel f s
      | none => throw "cond: no branches"
  | [], _, _ => by simp only [evalBranch, List.getElem?_nil]
  | _ :: _, 0, _ => by simp only [evalBranch, List.getElem?_cons_zero]
  | _ :: fs, k + 1, s => by
      simp only [evalBranch, List.getElem?_cons_succ]
      exact evalBranch_eq fuel fs k s

/-- `scan n body` equals `body` unrolled `n` times. -/
theorem eval_scan (fuel n : Nat) (body : Flow β) (s : List V) :
    eval L fuel (.scan n body) s = eval L fuel (body.unroll n) s := by
  simp only [eval]
  exact iterate_eval L fuel body n s

end Semantics

mutual

/-- The blocks of a flow, in order. -/
def blocks : Flow β → List β
  | .block b => [b]
  | .skip => []
  | .seq f g => blocks f ++ blocks g
  | .cond fs => branchBlocks fs
  | .scan _ body => blocks body
  | .whileLoop test body => blocks test ++ blocks body

/-- The blocks of a list of flows, in order. -/
def branchBlocks : List (Flow β) → List β
  | [] => []
  | f :: fs => blocks f ++ branchBlocks fs

end

omit [MonadExcept String m] in
mutual

/-- The flow with each block `b` replaced by `g b`, the blocks run in order. -/
def mapM (g : β → m γ) : Flow β → m (Flow γ)
  | .block b => do pure (.block (← g b))
  | .skip => pure .skip
  | .seq f h => do pure (.seq (← mapM g f) (← mapM g h))
  | .cond fs => do pure (.cond (← mapMBranches g fs))
  | .scan n body => do pure (.scan n (← mapM g body))
  | .whileLoop test body => do pure (.whileLoop (← mapM g test) (← mapM g body))

/-- `mapM` on each flow of a list. -/
def mapMBranches (g : β → m γ) : List (Flow β) → m (List (Flow γ))
  | [] => pure []
  | f :: fs => do pure ((← mapM g f) :: (← mapMBranches g fs))

end

section Except

variable (L : Lang (Except String) β V)

theorem evalBranch_eq_pick {fuel : Nat} {s : List V} {fs : List (Flow β)}
    {outs : List (List V)} (h : List.Forall₂ (fun f out => eval L fuel f s = .ok out) fs outs) :
    ∀ k, evalBranch L fuel fs k s = pick outs k := by
  induction h with
  | nil => intro k; simp [evalBranch, pick]
  | cons hf _ ih =>
      intro k
      cases k with
      | zero => simp [evalBranch, pick, hf, pure, Except.pure]
      | succ k => simpa [evalBranch, pick] using ih k

/-- Where every branch is defined on the operands `s`, `cond` equals evaluating every branch and
then selecting one. -/
theorem eval_cond {fuel : Nat} {fs : List (Flow β)} {v : V} {s : List V} {outs : List (List V)}
    (h : fs.mapM (eval L fuel · s) = .ok outs) :
    eval L fuel (.cond fs) (v :: s) = condAll L fuel fs (v :: s) := by
  have hb := evalBranch_eq_pick L (List.mapM_eq_ok_iff_forall₂.1 h)
  cases hi : L.index v <;> simp [eval, condAll, hi, h, hb, bind, Except.bind]

/-- A loop whose test holds before each of the first `k` steps and not after them, with
`k ≤ fuel`, is `k` steps. -/
theorem runWhile_eq_iterate {test : σ → Except String Bool} {step : σ → Except String σ} :
    ∀ {k fuel : Nat} {s : σ}, k ≤ fuel →
      (∀ j < k, (iterate step j s >>= test) = .ok true) →
      (iterate step k s >>= test) = .ok false →
      runWhile test step fuel s = iterate step k s
  | 0, fuel, s, _, _, hf => by
      simp only [iterate] at hf ⊢
      cases fuel <;> simp_all [runWhile, pure, Except.pure, bind, Except.bind]
  | k + 1, 0, _, hk, _, _ => absurd hk (by omega)
  | k + 1, fuel + 1, s, hk, ht, hf => by
      have h0 := ht 0 (by omega)
      simp only [iterate] at h0
      obtain ⟨t, hs⟩ : ∃ t, step s = .ok t := by
        cases hs : step s with
        | error e => simp [iterate, hs, bind, Except.bind] at hf
        | ok t => exact ⟨t, rfl⟩
      have ih := runWhile_eq_iterate (k := k) (fuel := fuel) (s := t) (by omega)
        (fun j hj => by simpa [iterate, hs, bind, Except.bind] using ht (j + 1) (by omega))
        (by simpa [iterate, hs, bind, Except.bind] using hf)
      simp_all [runWhile, iterate, pure, Except.pure, bind, Except.bind]

/-- A `whileLoop` whose test holds before each of the first `k` iterations and not after them,
with `k ≤ fuel`, equals `body` unrolled `k` times. -/
theorem eval_whileLoop {fuel k : Nat} {test body : Flow β} {s : List V} (hk : k ≤ fuel)
    (ht : ∀ j < k, (eval L fuel (body.unroll j) s >>= holds L fuel test) = .ok true)
    (hf : (eval L fuel (body.unroll k) s >>= holds L fuel test) = .ok false) :
    eval L fuel (.whileLoop test body) s = eval L fuel (body.unroll k) s := by
  simp only [eval]
  rw [← iterate_eval]
  refine runWhile_eq_iterate hk (fun j hj => ?_) ?_
  · rw [iterate_eval]
    exact ht j hj
  · rw [iterate_eval]
    exact hf

/-- A loop that succeeds is `k` steps for some `k ≤ fuel`, its test holding before each of them
and not after them. -/
theorem runWhile_ok {test : σ → Except String Bool} {step : σ → Except String σ} :
    ∀ {fuel : Nat} {s t : σ}, runWhile test step fuel s = .ok t →
      ∃ k ≤ fuel, (∀ j < k, (iterate step j s >>= test) = .ok true) ∧
        (iterate step k s >>= test) = .ok false ∧ iterate step k s = .ok t
  | fuel, s, t, h => by
      have stop : test s = .ok false → s = t →
          ∃ k ≤ fuel, (∀ j < k, (iterate step j s >>= test) = .ok true) ∧
            (iterate step k s >>= test) = .ok false ∧ iterate step k s = .ok t :=
        fun hc hs => ⟨0, Nat.zero_le _, fun _ hj => absurd hj (Nat.not_lt_zero _),
          by simpa [iterate, pure, Except.pure, bind, Except.bind] using hc,
          by simp [iterate, hs, pure, Except.pure]⟩
      cases fuel with
      | zero =>
          obtain ⟨c, hc, h⟩ := bind_eq_ok.1 h
          cases c
          · exact stop hc (by simpa [pure, Except.pure] using h)
          · simp [throw, throwThe, MonadExceptOf.throw] at h
      | succ fuel =>
          obtain ⟨c, hc, h⟩ := bind_eq_ok.1 h
          cases c
          · exact stop hc (by simpa [pure, Except.pure] using h)
          · simp only [ite_true] at h
            obtain ⟨s', hs', h⟩ := bind_eq_ok.1 h
            obtain ⟨k, hk, ht, hf, he⟩ := runWhile_ok h
            refine ⟨k + 1, by omega, fun j hj => ?_, ?_, ?_⟩
            · cases j with
              | zero => simpa [iterate, pure, Except.pure, bind, Except.bind] using hc
              | succ j => simpa [iterate, hs', bind, Except.bind] using ht j (by omega)
            · simpa [iterate, hs', bind, Except.bind] using hf
            · simpa [iterate, hs', bind, Except.bind] using he

/-- A `whileLoop` that succeeds equals `body` unrolled `k` times for some `k ≤ fuel`, its test
holding before each of the `k` iterations and not after them. -/
theorem eval_whileLoop_ok {fuel : Nat} {test body : Flow β} {s t : List V}
    (h : eval L fuel (.whileLoop test body) s = .ok t) :
    ∃ k ≤ fuel, (∀ j < k, (eval L fuel (body.unroll j) s >>= holds L fuel test) = .ok true) ∧
      (eval L fuel (body.unroll k) s >>= holds L fuel test) = .ok false ∧
      eval L fuel (body.unroll k) s = .ok t := by
  simp only [eval] at h
  obtain ⟨k, hk, ht, hf, he⟩ := runWhile_ok h
  refine ⟨k, hk, fun j hj => ?_, ?_, ?_⟩
  · rw [← iterate_eval]
    exact ht j hj
  · rw [← iterate_eval]
    exact hf
  · rw [← iterate_eval]
    exact he

mutual

/-- Each block of a flow that `mapM g` returns is `g` of a block of the flow. -/
theorem mapM_blocks {g : β → Except String γ} :
    ∀ {f : Flow β} {f' : Flow γ}, f.mapM g = .ok f' → ∀ c ∈ f'.blocks, ∃ b ∈ f.blocks, g b = .ok c
  | .block b, f', h, c, hc => by
      obtain ⟨b', hb, h⟩ := bind_eq_ok.1 h
      cases h
      simp only [blocks, List.mem_singleton] at hc
      exact ⟨b, by simp [blocks], hc ▸ hb⟩
  | .skip, f', h, c, hc => by
      cases h
      simp [blocks] at hc
  | .seq f₁ f₂, f', h, c, hc => by
      obtain ⟨g₁, h₁, h⟩ := bind_eq_ok.1 h
      obtain ⟨g₂, h₂, h⟩ := bind_eq_ok.1 h
      cases h
      simp only [blocks, List.mem_append] at hc ⊢
      rcases hc with hc | hc
      · obtain ⟨b, hb, e⟩ := mapM_blocks h₁ c hc
        exact ⟨b, .inl hb, e⟩
      · obtain ⟨b, hb, e⟩ := mapM_blocks h₂ c hc
        exact ⟨b, .inr hb, e⟩
  | .cond fs, f', h, c, hc => by
      obtain ⟨gs, hs, h⟩ := bind_eq_ok.1 h
      cases h
      exact mapMBranches_blocks hs c hc
  | .scan n body, f', h, c, hc => by
      obtain ⟨g₁, h₁, h⟩ := bind_eq_ok.1 h
      cases h
      exact mapM_blocks h₁ c hc
  | .whileLoop test body, f', h, c, hc => by
      obtain ⟨g₁, h₁, h⟩ := bind_eq_ok.1 h
      obtain ⟨g₂, h₂, h⟩ := bind_eq_ok.1 h
      cases h
      simp only [blocks, List.mem_append] at hc ⊢
      rcases hc with hc | hc
      · obtain ⟨b, hb, e⟩ := mapM_blocks h₁ c hc
        exact ⟨b, .inl hb, e⟩
      · obtain ⟨b, hb, e⟩ := mapM_blocks h₂ c hc
        exact ⟨b, .inr hb, e⟩

/-- Each block of the flows that `mapMBranches g` returns is `g` of a block of the flows. -/
theorem mapMBranches_blocks {g : β → Except String γ} :
    ∀ {fs : List (Flow β)} {fs' : List (Flow γ)}, mapMBranches g fs = .ok fs' →
      ∀ c ∈ branchBlocks fs', ∃ b ∈ branchBlocks fs, g b = .ok c
  | [], fs', h, c, hc => by
      cases h
      simp [branchBlocks] at hc
  | f :: fs, fs', h, c, hc => by
      obtain ⟨g₁, h₁, h⟩ := bind_eq_ok.1 h
      obtain ⟨g₂, h₂, h⟩ := bind_eq_ok.1 h
      cases h
      simp only [branchBlocks, List.mem_append] at hc ⊢
      rcases hc with hc | hc
      · obtain ⟨b, hb, e⟩ := mapM_blocks h₁ c hc
        exact ⟨b, .inl hb, e⟩
      · obtain ⟨b, hb, e⟩ := mapMBranches_blocks h₂ c hc
        exact ⟨b, .inr hb, e⟩

end

end Except

section Graphs

variable {α : Type} [Storage α]

/-- The branch index of a scalar value: an index, an integer word (two's complement when signed),
or a Boolean (`false` 0, `true` 1), as `lax.switch` and `lax.cond` read theirs. -/
def toIndex : Op.Value α → Except String Int
  | ⟨⟨.index [_], .scalar⟩, x⟩ => pure (Fin.val (Tensor.item x))
  | ⟨⟨.integer true _, .scalar⟩, x⟩ => pure (BitVec.toInt (Tensor.item x))
  | ⟨⟨.integer false _, .scalar⟩, x⟩ => pure (BitVec.toNat (Tensor.item x))
  | ⟨⟨.bool, .scalar⟩, x⟩ => pure (if Tensor.item x then 1 else 0)
  | ⟨p, _⟩ => throw s!"cond: a branch index at {repr p}"

/-- Whether a scalar Boolean value holds, as `lax.while_loop`'s test returns it. -/
def toBool : Op.Value α → Except String Bool
  | ⟨⟨.bool, .scalar⟩, x⟩ => pure (Tensor.item x)
  | ⟨p, _⟩ => throw s!"while: a test value at {repr p}"

/-- The flows of IR graphs over the scalar `α`: a block is a graph with its payload, run by
`Graph.eval`; indices are read by `toIndex` and tests by `toBool`. -/
def graphs (α : Type) [Storage α] [Context α] [Op.Capabilities α] :
    Lang (Except String) (Graph × Payload α) (Op.Value α) where
  run b := b.1.eval b.2
  index := toIndex
  test := toBool

end Graphs

/-- The flows of typed programs under `R`: a block is a program with its payload, run by
`Program.eval` on the state as its arguments, its conversion flags merged into the state of the
monad. A block's program must have one input per value of the state, and its outputs are the new
state. Indices are read by `toIndex` and tests by `toBool`. -/
def programs (R : Program.Machine) :
    Lang (StateT ConversionStatus (Except String)) (Program × Program.Payload R)
      (Program.Value R) where
  run b s := do
    let (outs, raised) ← StateT.lift (b.1.eval R b.2 s.toArray)
    modify (·.merge raised)
    pure outs
  index v := letI := R.storageAt v.typed.scalar; StateT.lift (toIndex ⟨v.typed.port, v.tensor⟩)
  test v := letI := R.storageAt v.typed.scalar; StateT.lift (toBool ⟨v.typed.port, v.tensor⟩)

end Flow

end TorchLean.IR
