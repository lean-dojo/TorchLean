/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLeanExamples.Tree.Hylos
public import TorchLeanTest.Tree.Fold

/-!
# Hylomorphisms

The checks run the hylomorphisms of `TorchLeanExamples.Tree.Hylos` and two of their own from
batches of roots, and compare every result with an independent computation or with the reference
function by its full encoding. `walk` follows a list stored in an input, so the scalar slot of a
call is read from the input when the call is made, and `probe` reads an input in its guard, so
the number of calls of a row can fail.

On the host the call trees of the Fibonacci numbers up to 20 have `2 * fib (n + 1) - 1` rows, and
the expansion succeeds at a capacity of exactly their total and gives up one row below it. The
schedule gives the numbers of a loop over pairs, the sums by halves of ranges of an input, and
the composite midpoint rule, in binary64, binary32 and a 12-bit format. In binary64 the sums by
halves of small integers and the midpoint rule over the unit interval are exact, so they must be
the exact sums and `(4 ^ (d + 1) - 1) / 3 / 4 ^ (d + 1)`. A range past the end of the input fails
with its first missing entry. A walk past the end of its list and a probe past the end of its
input make the expansion give up, and the results are then those of the reference. An empty batch
expands to no levels.

On the device `Expansion.expandDevice` must give the levels of `Expansion.expand` at the capacity
`min capacity Expansion.deviceCapacity` bit for bit, or give up with it, `Hylo.runDevice` must give
the results of the host schedule or give up with it, and `Hylo.run` must give the reference
results, failures included, in binary64, binary32, binary16 and configured 12-bit, 17-bit, 65-bit
and 256-bit formats, and in binary64 for a walk of 3000 steps. Roots without the slots of the
hylomorphism are refused. Numbers of calls of `2 ^ 31 - 1`, `2 ^ 31`, `2 ^ 63` and `2 ^ 64 - 1`,
and two of `2 ^ 62`, make the device give up as the host does, and an expansion without index
slots gives the host levels.

Without LibTorch `Hylo.run` must give the reference results on the host.
-/

@[expose] public section

namespace TorchLean.Test.Tree.Hylos

open Kernel Examples.Tree Arenas Folds

variable {α : Type}

/-- The value after `d` steps of a walk over input `0` from entry `i` with the value `v`: a step
reads entry `i` and goes on from entry `i + 1` with that entry as the value. -/
def walk (read : Reader α) : Nat → UInt64 → α → Except Error α
  | 0, _, v => .ok v
  | d + 1, i, _ => do walk read d (i + 1) (← read 0 i)

namespace Walk

/-- The number of steps `d` left to a call. -/
def steps : Expr α (Slots 2 1) .index :=
  .var (Slots.indexVar 0)

/-- The entry `i` that a call reads. -/
def entry : Expr α (Slots 2 1) .index :=
  .var (Slots.indexVar 1)

/-- The reference: `walk` from the slots of the call. -/
def reference : Reader α → Env α (Slots 2 1) → Except Error α :=
  fun read env => walk read (env.get (Slots.indexVar 0)).toNat (env.get (Slots.indexVar 1))
    (env.get (Slots.scalarVar 0))

/-- Without steps left a call makes no calls and gives its value. -/
def done : Expr α (Slots 2 1) .predicate × Hylo.Branch α 2 1 :=
  (.indexCompare .eq steps (.index 0), ⟨[], .var (Row.scalarVar 0)⟩)

/-- With steps left a call makes one call, from the next entry with the entry it reads as the
value, and gives its result. -/
def step : Hylo.Branch α 2 1 :=
  ⟨[⟨fun k => if k = 0 then .indexBinary .sub steps (.index 1)
        else .indexBinary .add entry (.index 1), fun _ => .load 0 entry⟩],
    .load 1 (.var .zero)⟩

/-- The certificate of the branches of `walk`. -/
theorem correct (A : Arithmetic α) :
    Hylo.Branches.Correct 2 1 A 1 reference (fun _ _ => True) [done] step := by
  refine ⟨rfl, ⟨by simp [UInt64.size], ?_⟩, ⟨by simp [step, UInt64.size], ?_⟩⟩
  · rintro read R index value first - ⟨-, h⟩ - -
    simp only [steps, Expr.eval, Slots.get_env_indexVar, Fin.val_zero, Compare.index,
      Bind.bind, Except.bind, Pure.pure, Except.pure, Except.ok.injEq, beq_iff_eq] at h
    simp [reference, Expr.eval, h, walk]
  · rintro read R index value first - ⟨-, h⟩ hfirst hcall
    simp only [steps, Expr.eval, Slots.get_env_indexVar, Fin.val_zero, Compare.index,
      Bind.bind, Except.bind, Pure.pure, Except.pure, Except.ok.injEq, beq_eq_false_iff_ne] at h
    have hk : 1 ≤ (index 0).toNat :=
      Nat.pos_of_ne_zero fun h0 => h (UInt64.toNat_inj.1 (by simpa using h0))
    obtain ⟨i0, v0, hi0, hv0, e0⟩ := hcall 0 (by simp [step])
    have a0 := hi0 0
    have b0 := hi0 1
    have c0 := hv0 0
    simp only [step, steps, entry, List.getElem_cons_zero, Expr.eval, Slots.get_env_indexVar,
      Fin.val_zero, Fin.val_one, Bind.bind, Except.bind, Pure.pure, Except.pure, IndexOp.eval,
      Except.ok.injEq, Fin.isValue, one_ne_zero, ↓reduceIte] at a0 b0 c0
    simp only [reference, Slots.get_env_indexVar, Slots.get_env_scalarVar, Fin.val_zero,
      Fin.val_one, Nat.add_zero] at e0
    simp only [step, reference, Expr.eval, Row.get_env_zero, Slots.get_env_indexVar,
      Slots.get_env_scalarVar, Fin.val_zero, Fin.val_one, Bind.bind, Except.bind]
    rw [e0, ← a0, ← b0, UInt64.toNat_sub_of_le _ _ (by simpa [UInt64.le_iff_toNat_le] using hk)]
    obtain ⟨k, hk'⟩ : ∃ k, (index 0).toNat = k + 1 := ⟨(index 0).toNat - 1, by omega⟩
    simp [hk', walk, c0, Bind.bind, Except.bind]

end Walk

/-- The hylomorphism of `walk` in the device arithmetic of `α`: a call holds the steps left and
the entry it reads in its index slots and its value in its scalar slot. -/
def walkHylo [Scalar α] : Hylo α 2 1 where
  scalar := inferInstance
  arithmetic := Scalar.arithmetic
  inputs := 1
  expr := Hylo.stepOf [Walk.done] Walk.step
  native := rfl
  reference := Walk.reference
  calls := Hylo.callsOf [Walk.done] Walk.step
  argIndex := Hylo.argIndexOf [Walk.done] Walk.step
  argValue := Hylo.argValueOf (α := α) [Walk.done] Walk.step
  correct := Hylo.correct_branches (Walk.correct Scalar.arithmetic)

/-- Whether entry `i` of input `0` is below `zero`: `one` when it is and `zero` otherwise. -/
def probe (A : Arithmetic α) (zero one : α) (read : Reader α) (i : UInt64) : Except Error α := do
  let x ← read 0 i
  return bif A.compare .lt x zero then one else zero

namespace Probe

/-- The reference: `probe` at the index slot of the call. -/
def reference (A : Arithmetic α) (zero one : α) : Reader α → Env α (Slots 1 0) → Except Error α :=
  fun read env => probe A zero one read (env.get (Slots.indexVar 0))

/-- A call whose entry is below `zero` makes no calls and gives `one`. -/
def below (zero one : α) : Expr α (Slots 1 0) .predicate × Hylo.Branch α 1 0 :=
  (.compare .lt (.load 0 (.var (Slots.indexVar 0))) (.scalar zero), ⟨[], .scalar one⟩)

/-- Any other call makes no calls and gives `zero`. -/
def above (zero : α) : Hylo.Branch α 1 0 :=
  ⟨[], .scalar zero⟩

/-- The certificate of the branches of `probe`. -/
theorem correct (A : Arithmetic α) (zero one : α) :
    Hylo.Branches.Correct 1 0 A 1 (reference A zero one) (fun _ _ => True) [below zero one]
      (above zero) := by
  refine ⟨rfl, ⟨by simp [UInt64.size], ?_⟩, ⟨by simp [above, UInt64.size], ?_⟩⟩
  · rintro read R index value first - ⟨-, h⟩ - -
    simp only [Expr.eval, Slots.get_env_indexVar, Fin.val_zero, Bind.bind, Except.bind,
      Pure.pure, Except.pure] at h
    simp only [reference, probe, Expr.eval, Slots.get_env_indexVar, Fin.val_zero, Bind.bind,
      Except.bind, Pure.pure, Except.pure]
    cases hx : read 0 (index 0) with
    | error e => simp [hx] at h
    | ok x => simp_all
  · rintro read R index value first - ⟨-, h⟩ - -
    simp only [Expr.eval, Slots.get_env_indexVar, Fin.val_zero, Bind.bind, Except.bind,
      Pure.pure, Except.pure] at h
    simp only [above, reference, probe, Expr.eval, Slots.get_env_indexVar, Fin.val_zero,
      Bind.bind, Except.bind, Pure.pure, Except.pure]
    cases hx : read 0 (index 0) with
    | error e => simp [hx] at h
    | ok x => simp_all

end Probe

/-- The hylomorphism of `probe` in the device arithmetic of `α`: a call holds its entry in its
index slot and makes no calls, but its number of calls reads the entry. -/
def probeHylo [Scalar α] (zero one : α) : Hylo α 1 0 where
  scalar := inferInstance
  arithmetic := Scalar.arithmetic
  inputs := 1
  expr := Hylo.stepOf [Probe.below zero one] (Probe.above zero)
  native := rfl
  reference := Probe.reference Scalar.arithmetic zero one
  calls := Hylo.callsOf [Probe.below zero one] (Probe.above zero)
  argIndex := Hylo.argIndexOf [Probe.below zero one] (Probe.above zero)
  argValue := Hylo.argValueOf (α := α) [Probe.below zero one] (Probe.above zero)
  correct := Hylo.correct_branches (Probe.correct Scalar.arithmetic zero one)

/-- The roots `count` rows with the index slots `index` and the scalar slots `scalar`. -/
def roots (count : Nat) (index : Array (Array UInt64)) (scalar : Array (Array α)) :
    Arena.Level α :=
  ⟨Array.replicate count 0, index, scalar⟩

/-- The Fibonacci numbers in arithmetic `A` from `zero` and `one`, by a loop over pairs. -/
def fibLoop (A : Arithmetic α) (zero one : α) (n : Nat) : α :=
  (Nat.fold n (fun _ _ p => (p.2, A.binary .add p.2 p.1)) (zero, one)).1

/-- The number of rows of the call tree of `fib n`: `2 * fib (n + 1) - 1`. -/
def fibCalls (n : Nat) : Nat :=
  2 * (Nat.fold n (fun _ _ p => (p.2, p.1 + p.2)) (1, 1)).1 - 1

/-- Whether two lists of levels have the same tags, index slots and scalar slots, the scalars by
their encodings. -/
def sameLevels [Scalar α] (xs ys : List (Arena.Level α)) : Bool :=
  xs.length == ys.length && (xs.zip ys).all fun (x, y) =>
    x.tag == y.tag && x.index == y.index && x.scalar.size == y.scalar.size &&
      (x.scalar.zip y.scalar).all fun (u, v) => allSame (u.map .ok) (v.map .ok)

/-- The cases of the checks for scalars of one type, as functions of a check of a hylomorphism
from roots with inputs at a capacity, whether it expands, and the expected results. -/
structure Cases (α : Type) where
  /-- Checks a hylomorphism. -/
  check : {m s : Nat} → String → Hylo α m s → Nat → Arena.Level α → Array (Array α) → Bool →
    Array (Except Error α) → IO Unit

/-- The checks for scalars of one type, through `c`: the Fibonacci numbers up to 20 at the
capacity of their call trees and one row below it, an empty batch, sums by halves of ranges of an
input and past its end, the midpoint rule, walks and probes inside and past their inputs. -/
@[noinline, nospecialize] def checkAll [Scalar α] [Inhabited α] (label : String) (c : Cases α) :
    IO Unit := do
  let A : Arithmetic α := Scalar.arithmetic
  let zero : α := Numeric.Word.ofNatBits 0
  let (z, o) : α × α := (moderate 1, moderate 2)
  let ns := (List.range 21).toArray
  let total := (ns.map fibCalls).foldl (· + ·) 0
  let fibRoots := roots ns.size #[ns.map (·.toUInt64)] #[]
  let fibs := ns.map fun n => .ok (fibLoop A z o n)
  c.check s!"{label} fib" (fibHylo z o) total fibRoots #[] true fibs
  c.check s!"{label} fib below capacity" (fibHylo z o) (total - 1) fibRoots #[] false fibs
  c.check s!"{label} fib 20 at 2 ^ 40" (fibHylo z o) (2 ^ 40) (roots 1 #[#[20]] #[]) #[] true
    #[.ok (fibLoop A z o 20)]
  c.check s!"{label} empty batch" (fibHylo z o) 0 (roots 0 #[#[]] #[]) #[] true #[]
  let n := 10000
  let xs : Array α := (Array.range n).map fun k => moderate (k + 77).toUInt64
  let read := Reader.ofArrays #[xs]
  let ranges : Array (Nat × Nat) :=
    #[(0, n), (0, 1), (5, 5), (9, 3), (1, n - 1), (17, 1000), (n - 1, n)]
  let rangeRoots (rs : Array (Nat × Nat)) : Arena.Level α :=
    roots rs.size #[rs.map (·.1.toUInt64), rs.map (·.2.toUInt64)] #[]
  c.check s!"{label} sums by halves" (pairwiseHylo zero) (2 ^ 20) (rangeRoots ranges) #[xs] true
    (ranges.map fun (lo, hi) => pairwise A zero read lo hi)
  c.check s!"{label} sum past the end" (pairwiseHylo zero) (2 ^ 20) (rangeRoots #[(n - 2, n + 3)])
    #[xs] true #[.error (.bounds 0 n.toUInt64 n)]
  let depths := (List.range 11).toArray
  let ends : Array α := depths.map fun d => moderate (d + 300).toUInt64
  let widths : Array α := depths.map fun d => moderate (d + 400).toUInt64
  let two : α := moderate 5
  c.check s!"{label} midpoint rule" (midpointHylo two) (2 ^ 20)
    (roots depths.size #[depths.map (·.toUInt64)] #[ends, widths]) #[] true
    (depths.map fun d => .ok (midpoint A two d ends[d]! widths[d]!))
  let list : Array α := (Array.range 300).map fun k => moderate (k + 500).toUInt64
  let walks : Array (Nat × Nat) := #[(0, 5), (3, 0), (300, 0), (1, 299), (150, 100)]
  let walkRoots (ws : Array (Nat × Nat)) : Arena.Level α :=
    roots ws.size #[ws.map (·.1.toUInt64), ws.map (·.2.toUInt64)] #[ws.map fun _ => z]
  let readList := Reader.ofArrays #[list]
  c.check s!"{label} walks" walkHylo (2 ^ 20) (walkRoots walks) #[list] true
    (walks.map fun (d, i) => walk readList d i.toUInt64 z)
  c.check s!"{label} walk past the end" walkHylo (2 ^ 20) (walkRoots #[(3, 1), (5, 298)])
    #[list] false #[walk readList 3 1 z, .error (.bounds 0 300 300)]
  let entries : Array Nat := #[0, 1, 2, 299]
  let probeRoots (es : Array Nat) : Arena.Level α := roots es.size #[es.map (·.toUInt64)] #[]
  c.check s!"{label} probes" (probeHylo zero o) 4 (probeRoots entries) #[list] true
    (entries.map fun i => probe A zero o readList i.toUInt64)
  c.check s!"{label} probe past the end" (probeHylo zero o) 4 (probeRoots #[3, 300]) #[list] false
    #[probe A zero o readList 3, .error (.bounds 0 300 300)]

/-- The host check of a hylomorphism: the expansion succeeds when `expands` holds and gives up
otherwise, the schedule then gives `expected`, and the results are `expected`. -/
@[noinline, nospecialize] def checkHost [Inhabited α] {m s : Nat} (label : String)
    (H : Hylo α m s) (capacity : Nat) (r : Arena.Level α) (inputs : Array (Array α))
    (expands : Bool) (expected : Array (Except Error α)) : IO Unit :=
  letI := H.scalar
  do
    let read := Reader.ofArrays inputs
    let levels := H.expand read capacity r
    ensure (levels.isSome == expands)
      s!"{label}: the expansion {if expands then "gave up" else "finished"}"
    if expands && expected.all (· matches .ok _) then
      match H.schedule read capacity r with
      | some vs =>
        ensure (allSame (vs.map .ok) expected)
          s!"{label}: the schedule differs from the expected results"
      | none => throw <| IO.userError s!"{label}: the schedule failed"
    ensure (allSame (H.results read capacity r) expected)
      s!"{label}: the results differ from the expected results"

/-- The host checks: the call trees of the Fibonacci numbers, exact sums by halves and midpoint
rules in binary64, and the cases of `checkAll` in three formats. -/
def run : IO Unit := do
  let ns := List.range 21
  for n in ns do
    match (fibHylo (0.0 : Float) 1.0).expand failing (2 ^ 20) (roots 1 #[#[n.toUInt64]] #[]) with
    | some levels =>
      ensure ((levels.map (·.tag.size)).sum == fibCalls n)
        s!"fib {n}: the call tree does not have {fibCalls n} rows"
    | none => throw <| IO.userError s!"fib {n}: the expansion gave up"
  let n := 100000
  let xs : Array Float := (Array.range n).map fun k => (k % 7).toFloat
  let ranges : Array (Nat × Nat) := #[(0, n), (3, 3), (1, 2), (12345, 67890), (n - 7, n)]
  let exact := ranges.map fun (lo, hi) =>
    .ok ((List.range (hi - lo)).foldl (fun acc k => acc + (lo + k) % 7) 0).toFloat
  checkHost "binary64 exact sums by halves" (pairwiseHylo 0.0) (2 ^ 20)
    (roots ranges.size #[ranges.map (·.1.toUInt64), ranges.map (·.2.toUInt64)] #[]) #[xs] true
    exact
  let depths := (List.range 13).toArray
  checkHost "binary64 exact midpoint rules" (midpointHylo 2.0) (2 ^ 20)
    (roots depths.size #[depths.map (·.toUInt64)]
      #[depths.map fun _ => 0.0, depths.map fun _ => 1.0]) #[] true
    (depths.map fun d => .ok (((4 ^ (d + 1) - 1) / 3).toFloat / (4 ^ (d + 1)).toFloat))
  checkAll (α := Float) "binary64" ⟨checkHost⟩
  checkAll (α := Float32) "binary32" ⟨checkHost⟩
  checkAll (α := FloatLib.Floats.ExecFloat.Binary 5 6) "12-bit" ⟨checkHost⟩
  IO.println "  tree hylomorphisms: call trees, sums by halves, midpoint rules, walks and probes \
    passed on the host"

/-- The device check of a hylomorphism: `Expansion.expandDevice` must give the levels of
`Expansion.expand` at the capacity `min capacity Expansion.deviceCapacity`, and expand when
`expands` holds, `Hylo.runDevice` must give the results of the host schedule or give up with it,
and `Hylo.run` must give `expected`. -/
@[noinline, nospecialize] def checkDevice [Inhabited α] {m s : Nat} (label : String)
    (H : Hylo α m s) (capacity : Nat) (r : Arena.Level α) (inputs : Array (Array α))
    (expands : Bool) (expected : Array (Except Error α)) : IO Unit :=
  letI := H.scalar
  do
    let read := Reader.ofArrays inputs
    let bound := min capacity Expansion.deviceCapacity
    let host := H.expand read bound r
    ensure (host.isSome == expands)
      s!"{label}: the host expansion {if expands then "gave up" else "finished"}"
    let device ← H.expandDevice capacity r inputs
    ensure (device.isSome == expands)
      s!"{label}: the device expansion {if expands then "gave up" else "finished"}"
    if let (some hs, some ds) := (host, device) then
      ensure (sameLevels hs ds) s!"{label}: the device levels differ from the host expansion"
    let schedule := H.schedule read bound r
    let results ← H.runDevice capacity r inputs
    ensure (results.isSome == schedule.isSome)
      s!"{label}: the device {if schedule.isSome then "gave up" else "finished"}"
    if let (some vs, some ws) := (results, schedule) then
      ensure (allSame (vs.map .ok) (ws.map .ok))
        s!"{label}: the device results differ from the host schedule"
    ensure (allSame (← H.run capacity r inputs) expected)
      s!"{label}: the results of run differ from the expected results"

/-- The check of `Hylo.run` alone: it must give `expected`. -/
@[noinline, nospecialize] def checkRun [Inhabited α] {m s : Nat} (label : String)
    (H : Hylo α m s) (capacity : Nat) (r : Arena.Level α) (inputs : Array (Array α)) (_ : Bool)
    (expected : Array (Except Error α)) : IO Unit :=
  letI := H.scalar
  do
    ensure (allSame (← H.run capacity r inputs) expected)
      s!"{label}: the results of run differ from the expected results"

/-- The checks of builds without LibTorch: `Hylo.run` gives the expected results on the host in
the cases of `checkAll`, in binary64 and a 12-bit format. -/
def runUnlinked : IO Unit := do
  checkAll (α := Float) "binary64" ⟨checkRun⟩
  checkAll (α := FloatLib.Floats.ExecFloat.Binary 5 6) "12-bit" ⟨checkRun⟩
  IO.println "  tree hylomorphisms: run computes the results on the host without LibTorch"

/-- An expansion over one index slot whose rows make as many calls as their slot holds, every
call with the slot `0`, so that a root of `c` makes one level of `c` rows without calls. -/
def burst [Scalar α] : Expansion α 1 0 where
  scalar := inferInstance
  arithmetic := Scalar.arithmetic
  inputs := 0
  calls := .var (Slots.indexVar 0)
  argIndex _ := .index 0
  argValue i := i.elim0
  native := rfl

/-- An expansion over one scalar slot in binary64, without index slots: a row whose slot is at
least `1` makes two calls with half of it, and a row with a smaller slot makes none. -/
def halving : Expansion Float 0 1 where
  scalar := inferInstance
  arithmetic := Scalar.arithmetic
  inputs := 0
  calls := .cond (.compare .lt (.var (Slots.scalarVar 0)) (.scalar 1.0)) (.index 0) (.index 2)
  argIndex i := i.elim0
  argValue _ := .binary .mul (.var (Row.scalarVar 0)) (.scalar 0.5)
  native := rfl

/-- The device check of an expansion without inputs: `Expansion.expandDevice` must give the levels
of `Expansion.expand` at the capacity `min capacity Expansion.deviceCapacity` bit for bit, and
expand when `expands` holds. -/
@[noinline, nospecialize] def checkExpansion [Inhabited α] {m s : Nat} (label : String)
    (E : Expansion α m s) (capacity : Nat) (r : Arena.Level α) (expands : Bool) : IO Unit :=
  letI := E.scalar
  do
    let host := E.expand (Reader.ofArrays #[]) (min capacity Expansion.deviceCapacity) r
    ensure (host.isSome == expands)
      s!"{label}: the host expansion {if expands then "gave up" else "finished"}"
    let device ← E.expandDevice capacity r #[]
    ensure (device.isSome == expands)
      s!"{label}: the device expansion {if expands then "gave up" else "finished"}"
    if let (some hs, some ds) := (host, device) then
      ensure (sameLevels hs ds) s!"{label}: the device levels differ from the host expansion"

/-- The device checks: roots without the slots of the hylomorphism are refused, numbers of calls
near `2 ^ 31`, `2 ^ 63` and `2 ^ 64` give up, an expansion without index slots, exact sums by
halves and midpoint rules in binary64, a walk of 3000 steps, and the cases of `checkAll` in seven
formats. -/
def runCUDA : IO Unit := do
  ensureFails "roots of other slots" "slots of the expansion"
    ((fibHylo (0.0 : Float) 1.0).expandDevice 10 (roots 2 #[#[1]] #[]) #[])
  ensureFails "roots with scalar slots" "slots of the expansion"
    ((fibHylo (0.0 : Float) 1.0).expandDevice 10 (roots 1 #[#[1]] #[#[1.0]]) #[])
  let bursts : List (String × Array Nat × Bool) :=
    [("2 ^ 63 calls", #[2 ^ 63], false), ("2 ^ 64 - 1 calls", #[2 ^ 64 - 1], false),
      ("2 ^ 31 calls", #[2 ^ 31], false), ("2 ^ 31 - 1 calls", #[2 ^ 31 - 1], false),
      ("two roots of 2 ^ 62 calls", #[2 ^ 62, 2 ^ 62], false), ("3 calls", #[3], true),
      ("no calls", #[0], true)]
  for (name, counts, expands) in bursts do
    checkExpansion s!"binary64 {name}" (burst (α := Float)) (2 ^ 40)
      (roots counts.size #[counts.map (·.toUInt64)] #[]) expands
  checkExpansion "binary64 halving without index slots" halving (2 ^ 20)
    (roots 4 #[] #[#[8.0, 0.25, 3.0, 1.0]]) true
  let n := 1000000
  let xs : Array Float := (Array.range n).map fun k => (k % 7).toFloat
  let ranges : Array (Nat × Nat) := #[(0, n), (3, 3), (1, 2), (12345, 678901), (n - 7, n)]
  let exact := ranges.map fun (lo, hi) =>
    .ok ((List.range (hi - lo)).foldl (fun acc k => acc + (lo + k) % 7) 0).toFloat
  checkDevice "binary64 exact sums by halves" (pairwiseHylo 0.0) (2 ^ 23)
    (roots ranges.size #[ranges.map (·.1.toUInt64), ranges.map (·.2.toUInt64)] #[]) #[xs] true
    exact
  let depths := (List.range 17).toArray
  checkDevice "binary64 exact midpoint rules" (midpointHylo 2.0) (2 ^ 20)
    (roots depths.size #[depths.map (·.toUInt64)]
      #[depths.map fun _ => 0.0, depths.map fun _ => 1.0]) #[] true
    (depths.map fun d => .ok (((4 ^ (d + 1) - 1) / 3).toFloat / (4 ^ (d + 1)).toFloat))
  let list : Array Float := (Array.range 3000).map (·.toFloat)
  checkDevice "binary64 walk of 3000 steps" walkHylo (2 ^ 20) (roots 1 #[#[3000], #[0]] #[#[0.5]])
    #[list] true #[.ok 2999.0]
  checkAll (α := Float) "binary64" ⟨checkDevice⟩
  checkAll (α := Float32) "binary32" ⟨checkDevice⟩
  checkAll (α := FloatLib.Floats.ExecFloat.Binary 5 10) "binary16" ⟨checkDevice⟩
  checkAll (α := FloatLib.Floats.ExecFloat.Binary 5 6) "12-bit" ⟨checkDevice⟩
  checkAll (α := FloatLib.Floats.ExecFloat.Binary 5 11) "17-bit" ⟨checkDevice⟩
  checkAll (α := FloatLib.Floats.ExecFloat.Binary 11 53) "65-bit" ⟨checkDevice⟩
  checkAll (α := FloatLib.Floats.ExecFloat.Binary 15 240) "256-bit" ⟨checkDevice⟩
  IO.println "  tree hylomorphisms: device expansions match the host levels bit for bit, and \
    device results the host schedule"

end TorchLean.Test.Tree.Hylos
