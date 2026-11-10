/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLeanTest.Tree.Unfold

/-!
# Scalars held as their encodings

The checks hold binary64 and binary32 values as `FloatBits` and `Float32Bits` and compare their
scalars by full encoding, NaNs included. On the host the special encodings of the format and
pseudorandom words come back from `Numeric.Word.ofNatBits` unchanged, while as `Float` and
`Float32` all the NaN encodings come back as one. Binary and rose trees of words go through their
arenas, columns and blocks bit for bit. The balanced trees of ranges of an input of these words,
built by `Unfold.build`, hold every input word bit for bit. Decoding an arena at the edge with
`Arena.map` gives the arena of the decoded tree, and folds over arenas of words, decoded at the
edge, give the folds over the decoded arenas bit for bit. The arena, fold, hylo and unfold checks
of the other modules run on words too.

On the device, arenas of words come back from `Arena.upload` and `Arena.Device.download` byte for
byte, the arenas that `Unfold.buildDevice` builds from an input of these words are the host builds
bit for bit, payloads included, and the device checks of the other modules run on words.
-/

@[expose] public section

namespace TorchLean.Test.Tree.Words

open Kernel Examples.Tree Arenas Folds Hylos Unfolds

variable {α β : Type}

/-- Whether two results are the same failure or scalars with the same full encoding. -/
def sameBits [Numeric.Word α] : Except Error α → Except Error α → Bool
  | .ok x, .ok y => bits x == bits y
  | .error e, .error e' => e == e'
  | _, _ => false

/-- Whether two lists of levels have the same tags, index slots and scalar slots, every scalar by
its full encoding. -/
def exactLevels [Numeric.Word α] (xs ys : List (Arena.Level α)) : Bool :=
  xs.length == ys.length && (xs.zip ys).all fun (x, y) =>
    x.tag == y.tag && x.index == y.index &&
      x.scalar.map (·.map bits) == y.scalar.map (·.map bits)

/-- The words of the checks in a format with `e` exponent and `f` fraction bits: its special
encodings and pseudorandom words. -/
def words (e f : Nat) : List Nat :=
  specials e f ++ (List.range 1000).map fun k => (BTree.mix k.toUInt64).toNat % 2 ^ (1 + e + f)

/-- Checks that the scalars of type `α` hold every word of `words`, and that the scalars of type
`native`, Lean's own type of the format, give all the NaN words of `specials` one encoding. -/
@[noinline, nospecialize] def checkHeld [Scalar α] (native : Type) [Numeric.Word native]
    (label : String) : IO Unit := do
  let (e, f) := fieldWidths (Scalar.precision α)
  let ws := words e f
  ensure ((ws.map fun w => bits (Numeric.Word.ofNatBits w : α)) == ws)
    s!"{label}: a word changed"
  let nans := (specials e f).filter fun w => w / 2 ^ f % 2 ^ e == 2 ^ e - 1 && w % 2 ^ f != 0
  ensure (nans.length == 4) s!"{label}: the special encodings hold {nans.length} NaNs"
  ensure ((nans.map fun w => bits (Numeric.Word.ofNatBits w : native)).eraseDups.length == 1)
    s!"{label}: Lean's type of the format holds two NaN encodings"

/-- Checks that decoding with `decode` at the edge commutes with the arenas and the folds:
the arena of a tree of words mapped by `decode` decodes to the tree mapped by `decode`, and the
sums of binary and rose trees and the values of terms over the arenas of words, decoded, are the
folds over the decoded arenas, bit for bit. -/
@[noinline, nospecialize] def checkEdge [Scalar α] [Inhabited α] [Scalar β] [Inhabited β]
    (label : String) (decode : α → β) : IO Unit := do
  let (e, f) := fieldWidths (Scalar.precision α)
  let xs : List α := (words e f).map Numeric.Word.ofNatBits
  for (name, t) in [("specials", spine xs), ("random", BTree.random scalarOf 5000 23)] do
    ensure (decodeB ((arenaB t).map decode) == some ((t.map decode).map bits))
      s!"{label} {name}: the arena decoded at the edge is not the arena of the decoded tree"
  let zero : α := Numeric.Word.ofNatBits 0
  let zero' : β := Numeric.Word.ofNatBits 0
  for (name, t) in btrees (α := α) ++ [("words", spine xs)] do
    let a := arenaB t
    ensure (sameBits (((BTree.sumFold zero).eval failing a).map decode)
        ((BTree.sumFold zero').eval failing (a.map decode)))
      s!"{label} {name}: the decoded sum differs from the sum of the decoded tree"
  for (name, t) in roses (α := α) do
    let a := arenaR t
    ensure (sameBits ((Rose.sumFold.eval failing a).map decode)
        (Rose.sumFold.eval failing (a.map decode)))
      s!"{label} rose {name}: the decoded sum differs from the sum of the decoded tree"
  let env := environment (α := α)
  for (name, t) in terms (α := α) do
    let a := arenaT t
    ensure (sameBits ((Term.evalFold.eval (Reader.ofArrays #[env]) a).map decode)
        (Term.evalFold.eval (Reader.ofArrays #[env.map decode]) (a.map decode)))
      s!"{label} term {name}: the decoded value differs from the value of the decoded term"

/-- Checks the balanced trees of ranges of an input of the words of `words`: the arena of
`Unfold.build` is the arena of the trees of the input words, every word bit for bit, and with
`device` so is the arena of `Unfold.buildDevice`. -/
@[noinline, nospecialize] def checkUnfoldWords [Scalar α] [Inhabited α] (device : Bool)
    (label : String) : IO Unit := do
  let (e, f) := fieldWidths (Scalar.precision α)
  let xs : Array α := ((words e f).map Numeric.Word.ofNatBits).toArray
  let n := xs.size
  let ranges : Array (Nat × Nat) := #[(0, n), (0, 13), (4, 9), (n - 1, n)]
  let r : Arena.Level α :=
    roots ranges.size #[ranges.map (·.1.toUInt64), ranges.map (·.2.toUInt64)] #[]
  let trees := ranges.toList.map fun (lo, hi) => balanced ((xs.toList.drop lo).take (hi - lo))
  let expected : Arena α :=
    Arena.ofValues (σ := BTree.sig) (T := fun _ => BTree α) (trees.map (⟨0, ·⟩))
  let some a := ofRangeUnfold.build (Reader.ofArrays #[xs]) (2 ^ 20) r
    | throw <| IO.userError s!"{label}: the build failed"
  ensure (exactLevels a.levels.toList expected.levels.toList)
    s!"{label}: the built arena is not the arena of the input words"
  if device then
    let some d ← ofRangeUnfold.buildDevice (2 ^ 20) r #[xs]
      | throw <| IO.userError s!"{label}: the device build failed"
    ensure (exactLevels (← d.download).levels.toList a.levels.toList)
      s!"{label}: the device arena is not the host build"

/-- The host checks: the words held in both formats, the arenas, the unfolds and the edge, and the
arena, fold, hylo and unfold checks of the other modules on words. -/
def run : IO Unit := do
  checkHeld (α := FloatBits) Float "binary64 words"
  checkHeld (α := Float32Bits) Float32 "binary32 words"
  checkHostTrees (α := FloatBits) "binary64 words"
  checkHostTrees (α := Float32Bits) "binary32 words"
  checkUnfoldWords (α := FloatBits) false "binary64 words"
  checkUnfoldWords (α := Float32Bits) false "binary32 words"
  checkEdge "binary64 words" FloatBits.toFloat
  checkEdge "binary32 words" Float32Bits.toFloat32
  checkTrees (α := FloatBits) "binary64 words"
  checkTrees (α := Float32Bits) "binary32 words"
  Hylos.checkAll (α := FloatBits) "binary64 words" ⟨Hylos.checkHost⟩
  Hylos.checkAll (α := Float32Bits) "binary32 words" ⟨Hylos.checkHost⟩
  Unfolds.checkAll (α := FloatBits) false "binary64 words"
  Unfolds.checkAll (α := Float32Bits) false "binary32 words"
  IO.println "  tree words: binary64 and binary32 encodings held bit for bit on the host, NaN \
    payloads included"

/-- The device checks: the arenas and the unfolds of words bit for bit, and the device checks of
arenas, folds, hylos and unfolds of the other modules on words. -/
def runCUDA : IO Unit := do
  checkDeviceTrees (α := FloatBits) "binary64 words"
  checkDeviceTrees (α := Float32Bits) "binary32 words"
  checkUnfoldWords (α := FloatBits) true "binary64 words"
  checkUnfoldWords (α := Float32Bits) true "binary32 words"
  checkDeviceFolds (α := FloatBits) "binary64 words"
  checkDeviceFolds (α := Float32Bits) "binary32 words"
  Hylos.checkAll (α := FloatBits) "binary64 words" ⟨Hylos.checkDevice⟩
  Hylos.checkAll (α := Float32Bits) "binary32 words" ⟨Hylos.checkDevice⟩
  Unfolds.checkAll (α := FloatBits) true "binary64 words"
  Unfolds.checkAll (α := Float32Bits) true "binary32 words"
  IO.println "  tree words: device arenas and unfolds of binary64 and binary32 encodings match \
    the host bit for bit, NaN payloads included"

end TorchLean.Test.Tree.Words
