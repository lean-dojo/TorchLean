/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLeanTest.Tree.Arena

/-!
# Derived signatures on the host and the device

Three types get their `Sig.Inductive` instances from `deriving`: binary trees of keys with no
scalars, and a mutual block of programs and terms with index fields, scalar fields and lists. The
checks encode values and decode them from their arenas, columns and blocks, and after a device
round trip. A program or a term is compared by the blocks of the value it decodes to, which
compares every scalar by its full encoding, and it must not decode as a value of the other sort.
The arenas of keys have `Unit` scalars and go through binary64 storage by `Arena.map`.
-/

@[expose] public section

namespace TorchLean.Test.Tree.Derived

open Kernel Examples.Tree Arenas
open FloatLib.Floats

/-- Binary trees of keys with no scalars: a leaf, or a node with two subtrees and a key. -/
inductive Keys where
  | leaf
  | node (l : Keys) (key : UInt64) (r : Keys)
  deriving BEq, Sig.Inductive

mutual
/-- Programs of a small loop language: an assignment of a term to a variable, a block of
programs, and a loop that runs its body a number of times. -/
inductive Prog (α : Type) where
  | assign (v : UInt64) (e : Term α)
  | block (ps : List (Prog α))
  | loop (n : UInt64) (body : Prog α)
  deriving Sig.Inductive

/-- Terms of the loop language: constants, variables, sums and products. -/
inductive Term (α : Type) where
  | const (x : α)
  | var (v : UInt64)
  | add (a b : Term α)
  | mul (a b : Term α)
  deriving Sig.Inductive
end

namespace Keys

/-- The tree with `k` inserted as in a binary search tree. -/
def insert (k : UInt64) : Keys → Keys
  | .leaf => .node .leaf k .leaf
  | .node l x r => if k < x then .node (l.insert k) x r else .node l x (r.insert k)

/-- The binary search tree of some keys, inserted in order. -/
def ofArray (ks : Array UInt64) : Keys := ks.foldl (fun t k => t.insert k) .leaf

end Keys

variable {α : Type}

/-- The arena of a tree of keys, with `Unit` scalars. -/
def arenaK (t : Keys) : Arena Unit :=
  Arena.encode (σ := Keys.sig) (T := fun _ => Keys) (s := 0) t

/-- The tree of keys that an arena decodes to. -/
def decodeK (a : Arena Unit) : Option Keys := a.decode (σ := Keys.sig) (fun _ => Keys) 0

/-- The arena of a tree of keys in binary64 storage. -/
def arenaK64 (t : Keys) : Arena Float := (arenaK t).map fun _ => 0

/-- The tree of keys that an arena in binary64 storage decodes to. -/
def decodeK64 (a : Arena Float) : Option Keys := decodeK (a.map fun _ => ())

/-- The trees of keys of the checks: the empty tree, a tree of extreme keys, a chain of sorted
keys and a pseudorandom search tree. -/
def keyTrees : List (String × Keys) :=
  [("leaf", .leaf),
    ("extremes", Keys.ofArray #[0x8000000000000000, 0, 0xFFFFFFFFFFFFFFFF, 1, 0x7FFFFFFFFFFFFFFF]),
    ("chain", Keys.ofArray ((Array.range 500).map Nat.toUInt64)),
    ("random", Keys.ofArray ((Array.range 20000).map fun i => BTree.mix i.toUInt64))]

/-- A term with the shape of a binary tree: a leaf is a variable, a node with two leaves is a
constant, and the other nodes are sums and products, chosen by the words of `seed`. -/
def termOf (seed : UInt64) : BTree α → Term α
  | .leaf => .var seed
  | .node .leaf x .leaf => .const x
  | .node l x r =>
    if seed % 2 == 0 then .add (termOf (BTree.mix seed) l) (termOf (BTree.mix (seed + 1)) r)
    else .mul (termOf (BTree.mix seed) l) (.add (.const x) (termOf (BTree.mix (seed + 1)) r))

mutual
/-- A program with the shape of a rose tree of words: a leaf assigns a term to a variable, and
a node is a block of the programs of its children, in a loop for one word in three. -/
def progOf (term : UInt64 → Term α) : Rose UInt64 → Prog α
  | .node w cs =>
    if cs.isEmpty then .assign w (term w)
    else if w % 3 == 0 then .loop w (.block (progsOf term cs))
    else .block (progsOf term cs)

/-- The programs of a list of rose trees of words. -/
def progsOf (term : UInt64 → Term α) : List (Rose UInt64) → List (Prog α)
  | [] => []
  | c :: cs => progOf term c :: progsOf term cs
end

/-- The arena of a value of sort `s` of the block of programs and terms. -/
def arenaP [Inhabited α] (s : Fin Prog.sig.sorts) (v : Prog.SigFamily (α := α) s) : Arena α :=
  Arena.encode (σ := Prog.sig) (T := Prog.SigFamily) (s := s) v

/-- The blocks of the columns of an arena of signature `σ`: the words block and the scalar
block. -/
def blocks [Scalar α] (σ : Sig) (a : Arena α) : Array UInt8 × Array UInt8 :=
  let c := a.columns σ
  (c.words.data, (c.scalars (Scalar.precision α).bytes).data)

/-- The blocks of the arena of the value of sort `s` that an arena decodes to. Decoding the arena
of a value gives the value, so equal blocks mean equal values. -/
def reblocks [Scalar α] [Inhabited α] (s : Fin Prog.sig.sorts) (a : Arena α) :
    Option (Array UInt8 × Array UInt8) :=
  (a.decode (σ := Prog.sig) Prog.SigFamily s).map fun v => blocks Prog.sig (arenaP s v)

/-- The values of the checks, each with its sort: pseudorandom programs, the empty block, extreme
words, and terms alone, which are values of the second sort. -/
def values [Numeric.Word α] : List (String × Σ s, Prog.SigFamily (α := α) s) :=
  let term (w : UInt64) : Term α := termOf w (BTree.random scalarOf (w % 16).toNat w)
  [("programs", ⟨0, progOf term (Rose.random id 3000 5)⟩),
    ("empty block", ⟨0, Prog.block []⟩),
    ("extreme words", ⟨0, Prog.loop 0xFFFFFFFFFFFFFFFF
      (.assign 0x8000000000000000 (.add (.var 0xFFFFFFFFFFFFFFFF) (.const (scalarOf 1))))⟩),
    ("terms", ⟨1, termOf 3 (BTree.random scalarOf 3000 9)⟩),
    ("constant", ⟨1, Term.const (scalarOf 2)⟩)]

/-- Checks the programs and terms with scalars of one type through `check`, the host or the
device round trip of `Arenas`, and that no value decodes as a value of the other sort. -/
@[noinline, nospecialize] def checkValues [Scalar α] [Inhabited α] (label : String)
    (check : String → Sig → Arena α → (Arena α → Option (Array UInt8 × Array UInt8)) →
      Option (Array UInt8 × Array UInt8) → IO Unit) : IO Unit := do
  for (name, ⟨s, v⟩) in values (α := α) do
    let a := arenaP s v
    check s!"{label} {name}" Prog.sig a (reblocks s) (some (blocks Prog.sig a))
    let other : Fin Prog.sig.sorts := if s = 0 then 1 else 0
    ensure (a.decode (σ := Prog.sig) Prog.SigFamily other).isNone
      s!"{label} {name}: the value decodes as a value of the other sort"

/-- The host checks: trees of keys decode to themselves with `Unit` scalars and through binary64
blocks, and programs and terms in four formats. -/
def run : IO Unit := do
  ensure (Keys.sig.scalarSlots == 0) "keys: the signature has scalar slots"
  for (name, t) in keyTrees do
    ensure (decodeK (arenaK t) == some t) s!"keys {name}: the arena does not decode to the tree"
    checkHost s!"keys {name}" Keys.sig (arenaK64 t) decodeK64 (some t)
  checkValues (α := Float) "binary64" checkHost
  checkValues (α := Float32) "binary32" checkHost
  checkValues (α := ExecFloat.Binary 5 10) "binary16" checkHost
  checkValues (α := ExecFloat.Binary 5 6) "12-bit" checkHost
  IO.println "  derived signatures: keys, programs and terms passed on the host"

/-- The device checks: trees of keys in binary64 storage, and programs and terms in four
formats, come back byte for byte and decode to themselves. -/
def runCUDA : IO Unit := do
  for (name, t) in keyTrees do
    checkDevice s!"keys {name}" Keys.sig (arenaK64 t) decodeK64 (some t)
  checkValues (α := Float) "binary64" checkDevice
  checkValues (α := Float32) "binary32" checkDevice
  checkValues (α := ExecFloat.Binary 5 10) "binary16" checkDevice
  checkValues (α := ExecFloat.Binary 5 6) "12-bit" checkDevice
  IO.println "  derived signatures: device round trips of keys, programs and terms passed"

/-- The checks of builds without LibTorch: an upload of keys fails with the runtime's message. -/
def runUnlinked : IO Unit := do
  ensureFails "keys upload without LibTorch" "built without LibTorch"
    (Arena.upload Keys.sig (arenaK64 (Keys.ofArray #[3, 1, 2])))
  IO.println "  derived signatures: uploads refused without LibTorch"

end TorchLean.Test.Tree.Derived
