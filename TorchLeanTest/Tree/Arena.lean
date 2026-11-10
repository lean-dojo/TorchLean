/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import FloatLib.Floats
public import TorchLean.CUDA.Tree.Device
public import TorchLeanExamples.Tree.Types

/-!
# Tree arenas on the host and the device

The host checks encode binary and rose trees, take the device columns and the blocks of
little-endian words of their arenas, read them back and decode them, comparing every scalar by its
full encoding. Besides pseudorandom encodings, each format gets trees of its special encodings:
both zeros, both infinities, quiet and signaling NaNs with payloads, and the smallest and largest
subnormals and normals. Lean's `Float` and `Float32` hold a single NaN, so for them the payloads
are checked in raw device blocks. The blocks of one binary tree are compared with blocks worked
out by hand. Arenas that break the layout in one place must not decode, a batch of values on level
zero decodes root by root, and the empty batch has no root.

The device checks upload arenas with binary64, binary32, binary16 and configured 12-bit, 17-bit,
65-bit and 256-bit scalars, download them and compare the blocks byte for byte, both as
`Arena.Device.download` reads them and as the runtime returns them, and the decoded trees by
encoding. The runtime must refuse a handle read with another scalar storage, width or descriptor
(binary16 read as bfloat16 among them), blocks that do not match their descriptor and sizes that
overflow. Builds without LibTorch check that an upload fails
with the runtime's message.
-/

@[expose] public section

namespace TorchLean.Test.Tree.Arenas

open Kernel Examples.Tree
open FloatLib.Floats

variable {α : Type}

/-- The full encoding of a scalar. -/
def bits [Numeric.Word α] (x : α) : Nat := (Numeric.Word.toBits x).toNat

/-- The scalar whose encoding is made of pseudorandom words from `w`, enough to fill its width. -/
def scalarOf [Numeric.Word α] (w : UInt64) : α := Id.run do
  let mut word := 0
  let mut state := w
  for j in [:Numeric.Word.width α / 64 + 1] do
    word := word + state.toNat * 2 ^ (64 * j)
    state := BTree.mix state
  return Numeric.Word.ofNatBits word

/-- The exponent and fraction widths of a precision. -/
def fieldWidths : Precision → Nat × Nat
  | .native .binary32 => (8, 23)
  | .native .binary64 => (11, 52)
  | .binary f => (f.expWidth, f.fracWidth)

/-- The special encodings of a binary format with `e` exponent and `f` fraction bits: both zeros,
both infinities, a quiet NaN, a signaling NaN, a NaN with every fraction bit set, a negative NaN
with a payload, the smallest and largest subnormals and normals, and the negative smallest
subnormal. -/
def specials (e f : Nat) : List Nat :=
  let sign := 2 ^ (e + f)
  let inf := (2 ^ e - 1) * 2 ^ f
  [0, sign, inf, sign + inf, inf + 2 ^ (f - 1), inf + 1, inf + 2 ^ f - 1,
    sign + inf + 2 ^ (f - 1) + 1, 1, 2 ^ f - 1, 2 ^ f, inf - 1, sign + 1]

/-- The binary tree with the scalars `xs` down its left spine. -/
def spine (xs : List α) : BTree α := xs.foldr (fun x t => .node t x .leaf) .leaf

/-- The arena of a binary tree. -/
def arenaB [Inhabited α] (t : BTree α) : Arena α :=
  Arena.encode (σ := BTree.sig) (T := fun _ => BTree α) (s := 0) t

/-- The arena of a rose tree. -/
def arenaR [Inhabited α] (t : Rose α) : Arena α :=
  Arena.encode (σ := Rose.sig) (T := fun _ => Rose α) (s := 0) t

/-- The arena of a batch of binary trees, one per row of level zero. -/
def batchB [Inhabited α] (ts : List (BTree α)) : Arena α :=
  Arena.ofValues (σ := BTree.sig) (T := fun _ => BTree α) (ts.map (⟨0, ·⟩))

/-- The binary tree that an arena decodes to, with each scalar replaced by its encoding. -/
def decodeB [Numeric.Word α] (a : Arena α) : Option (BTree Nat) :=
  (a.decode (σ := BTree.sig) (fun _ => BTree α) 0).map (·.map bits)

/-- The rose tree that an arena decodes to, with each scalar replaced by its encoding. -/
def decodeR [Numeric.Word α] (a : Arena α) : Option (Rose Nat) :=
  (a.decode (σ := Rose.sig) (fun _ => Rose α) 0).map (·.map bits)

/-- The arena read back from the blocks of the columns of signature `σ`, with scalar words of
`width` bytes, as `Arena.decode_ofBlocks_encode` reads them. -/
def throughBlocks [Numeric.Word α] (σ : Sig) (c : Arena.Columns α) (width : Nat) : Arena α :=
  Arena.ofColumns (Arena.Columns.ofBlocks c.sizes c.tag.size σ.indexSlots σ.scalarSlots width
    c.words (c.scalars width))

/-- Fails with `message` unless `ok` holds. -/
def ensure (ok : Bool) (message : String) : IO Unit :=
  unless ok do throw <| IO.userError message

/-- Checks that an arena of signature `σ` is wide and that it, its columns and the blocks of its
columns all decode, through `decode`, to `expected`. -/
@[noinline, nospecialize] def checkHost {β : Type} [BEq β] [Scalar α] (label : String) (σ : Sig)
    (a : Arena α) (decode : Arena α → Option β) (expected : Option β) : IO Unit := do
  ensure (decide (a.Wide σ)) s!"{label}: the arena is not wide"
  ensure (decode a == expected) s!"{label}: the arena does not decode to the tree"
  let c := a.columns σ
  ensure (decode (Arena.ofColumns c) == expected)
    s!"{label}: the columns do not decode to the tree"
  ensure (decode (throughBlocks σ c (Scalar.precision α).bytes) == expected)
    s!"{label}: the blocks do not decode to the tree"

/-- The little-endian bytes of words of `width` bytes. -/
def bytesLE (width : Nat) (words : Array Nat) : ByteArray :=
  ⟨words.flatMap fun w => (Array.range width).map fun j => (w / 256 ^ j % 256).toUInt8⟩

/-- Checks the blocks of a binary tree with four levels against blocks worked out by hand. -/
def checkLayout : IO Unit := do
  let t : BTree Float :=
    .node (.node .leaf 1.5 .leaf) 2.0 (.node (.node .leaf 3.25 .leaf) 4.0 .leaf)
  let c := (arenaB t).columns BTree.sig
  ensure (c.sizes == #[1, 2, 4, 2]) s!"layout: level sizes {c.sizes}"
  -- Tags, then the left and right child rows, one entry per row of levels 0 to 3.
  let words := #[1, 1, 1, 0, 0, 1, 0, 0, 0] ++ #[0, 0, 2, 0, 0, 0, 0, 0, 0] ++
    #[1, 1, 3, 0, 0, 1, 0, 0, 0]
  ensure (c.words.data == (bytesLE 8 words).data) "layout: the words block differs"
  let scalars := #[2.0, 1.5, 4.0, 0.0, 0.0, 3.25, 0.0, 0.0, 0.0].map fun x : Float => x.toBits.toNat
  ensure ((c.scalars 8).data == (bytesLE 8 scalars).data) "layout: the scalar block differs"

/-- A level of a binary64 arena with the given tag, index and scalar columns. -/
def level (tag : Array UInt64) (index : Array (Array UInt64)) (scalar : Array (Array Float)) :
    Arena.Level Float :=
  ⟨tag, index, scalar⟩

/-- Checks arenas that break the layout in one place: none decodes. Rows that share a child, and
lists that end exactly at the end of the next level, do decode. Decoding needs one unit of fuel
per level. -/
def checkMalformed : IO Unit := do
  let leaves := level #[0, 0] #[#[0, 0], #[0, 0]] #[#[0, 0]]
  let badB : List (String × Arena Float) := [
    ("no levels", ⟨#[]⟩),
    ("tag past the constructors", ⟨#[level #[2] #[#[0], #[0]] #[#[0]]]⟩),
    ("largest tag", ⟨#[level #[0xFFFFFFFFFFFFFFFF] #[#[0], #[0]] #[#[0]]]⟩),
    ("children on a missing level", ⟨#[level #[1] #[#[0], #[1]] #[#[1.5]]]⟩),
    ("child past the next level", ⟨#[level #[1] #[#[0], #[2]] #[#[1.5]], leaves]⟩),
    ("no scalar column", ⟨#[level #[1] #[#[0], #[1]] #[], leaves]⟩),
    ("short scalar column", ⟨#[level #[1] #[#[0], #[1]] #[#[]], leaves]⟩),
    ("short index column", ⟨#[level #[1] #[#[0], #[]] #[#[1.5]], leaves]⟩)]
  for (name, a) in badB do
    ensure (decodeB a).isNone s!"malformed: an arena with {name} decodes"
  let badR : List (String × Arena Float) := [
    ("a list past the next level", ⟨#[level #[0] #[#[0], #[3]] #[#[1]], leaves]⟩),
    ("an empty list past the next level", ⟨#[level #[0] #[#[5], #[0]] #[#[1]]]⟩),
    ("a list that wraps around",
      ⟨#[level #[0] #[#[0xFFFFFFFFFFFFFFFF], #[2]] #[#[1]], leaves]⟩)]
  for (name, a) in badR do
    ensure (decodeR a).isNone s!"malformed: an arena with {name} decodes"
  let b := bits (1.5 : Float)
  let shared : Arena Float := ⟨#[level #[1] #[#[0], #[0]] #[#[1.5]],
    level #[1] #[#[0], #[1]] #[#[2.5]], leaves]⟩
  let child : BTree Nat := .node .leaf (bits (2.5 : Float)) .leaf
  ensure (decodeB shared == some (.node child b child)) "malformed: a shared child does not decode"
  let z := bits (0.0 : Float)
  ensure (decodeR (⟨#[level #[0] #[#[0], #[2]] #[#[1.5]], leaves]⟩ : Arena Float) ==
    some (.node b [.node z [], .node z []])) "malformed: a list to the end does not decode"
  ensure (decodeR (⟨#[level #[0] #[#[0], #[0]] #[#[1.5]]]⟩ : Arena Float) == some (.node b []))
    "malformed: an empty list at the end does not decode"
  let a := arenaB (BTree.caterpillar (1.0 : Float) 5)
  ensure (a.decodeAt (σ := BTree.sig) (fun _ => BTree Float) (a.levels.size - 1) 0 0 0).isNone
    "malformed: a value decodes with fuel for one level less than it has"

/-- Checks a batch of binary trees on level zero, root by root, and the empty batch. -/
def checkBatch : IO Unit := do
  let ts : List (BTree Float) :=
    [.leaf, BTree.perfect 2.5 3, BTree.random scalarOf 300 23, spine [1.0, 2.0]]
  let a := batchB ts
  ensure (decide (a.Wide BTree.sig)) "batch: the arena is not wide"
  let root (k : Nat) := (a.decodeAt (σ := BTree.sig) (fun _ => BTree Float) a.levels.size 0 k
    0).map (·.map bits)
  for (t, k) in ts.zipIdx do
    ensure (root k == some (t.map bits)) s!"batch: root {k} does not decode to its tree"
  ensure (root ts.length).isNone "batch: the row after the last root decodes"
  checkHost "empty batch" BTree.sig (batchB ([] : List (BTree Float))) decodeB none
  ensure ((batchB ([] : List (BTree Float))).levels.size == 0) "empty batch: it has levels"

/-- The host checks for scalars of one type: pseudorandom and special encodings in binary and
rose trees. A configured format must hold every special encoding. -/
@[noinline, nospecialize] def checkHostTrees [Scalar α] [Inhabited α] (label : String) :
    IO Unit := do
  let (e, f) := fieldWidths (Scalar.precision α)
  let xs : List α := (specials e f).map Numeric.Word.ofNatBits
  if Scalar.precision α matches .binary _ then
    ensure (xs.map bits == specials e f) s!"{label}: the scalar type changes a special encoding"
  let x : α := scalarOf 7
  for (name, t) in [("leaf", BTree.leaf), ("node", .node .leaf x .leaf),
      ("perfect", BTree.perfect x 10), ("caterpillar", BTree.caterpillar x 300),
      ("random", BTree.random scalarOf 5000 11), ("specials", spine xs)] do
    checkHost s!"{label} {name}" BTree.sig (arenaB t) decodeB (some (t.map bits))
  for (name, t) in [("single", Rose.node x []), ("random", Rose.random scalarOf 5000 13),
      ("wide", Rose.node x ((List.range 300).map fun i => .node (scalarOf i.toUInt64) [])),
      ("specials", Rose.node x (xs.map (Rose.node · [])))] do
    checkHost s!"{label} rose {name}" Rose.sig (arenaR t) decodeR (some (t.map bits))

/-- The host checks: the hand-worked layout, malformed arenas, batches, and the round trips for
each scalar type. -/
def run : IO Unit := do
  checkLayout
  checkMalformed
  checkBatch
  checkHostTrees (α := Float) "binary64"
  checkHostTrees (α := Float32) "binary32"
  checkHostTrees (α := ExecFloat.Binary 5 10) "binary16"
  checkHostTrees (α := ExecFloat.Binary 5 6) "12-bit"
  checkHostTrees (α := ExecFloat.Binary 5 11) "17-bit"
  checkHostTrees (α := ExecFloat.Binary 11 53) "65-bit"
  checkHostTrees (α := ExecFloat.Binary 15 240) "256-bit"
  IO.println "  tree arenas: layout, malformed arenas, batches and round trips passed on the host"

/-- Checks that an action fails with an error containing `expected`. -/
def ensureFails {β : Type} (label expected : String) (action : IO β) : IO Unit := do
  let failed ← try discard action; pure none catch e => pure (some e.toString)
  match failed with
  | none => throw <| IO.userError s!"{label}: succeeded"
  | some e => ensure (e.contains expected) s!"{label}: failed with {e}"

/-- Checks the refusals that happen before the runtime is called. -/
def checkRefusals : IO Unit := do
  let narrow : Arena Float := ⟨#[{ tag := #[1], index := #[#[0]], scalar := #[] }]⟩
  ensureFails "narrow arena" "full size" (Arena.upload BTree.sig narrow)
  let wrong : Scalar Float := { (inferInstance : Scalar Float) with precision := .native .binary32 }
  ensureFails "scalars wider than storage" "wider"
    (@Arena.upload Float BTree.sig wrong (arenaB (BTree.perfect 1.0 3)))

/-- Checks that a device round trip of an arena of signature `σ` returns the same blocks, from
the runtime and through `Arena.Device.download`, and an arena that decodes to `expected`. -/
@[noinline, nospecialize] def checkDevice {β : Type} [BEq β] [Scalar α] (label : String) (σ : Sig)
    (a : Arena α) (decode : Arena α → Option β) (expected : Option β) : IO Unit := do
  let p := Scalar.precision α
  let width := p.bytes
  let c := a.columns σ
  let d ← Arena.upload σ a
  let words ← d.handle.words p.tag p.descriptor width.toUInt64
  let scalars ← d.handle.scalars p.tag p.descriptor width.toUInt64
  ensure (words.data == c.words.data) s!"{label}: the runtime returned another words block"
  ensure (scalars.data == (c.scalars width).data)
    s!"{label}: the runtime returned another scalar block"
  let back ← d.download
  let c' := back.columns σ
  ensure (c'.sizes == c.sizes) s!"{label}: level sizes changed on the device"
  ensure (c'.words.data == c.words.data) s!"{label}: the words block changed on the device"
  ensure ((c'.scalars width).data == (c.scalars width).data)
    s!"{label}: the scalar block changed on the device"
  ensure (decode back == expected) s!"{label}: the downloaded arena does not decode"

/-- The device checks for scalars of one type: pseudorandom and special encodings in binary and
rose trees. -/
@[noinline, nospecialize] def checkDeviceTrees [Scalar α] [Inhabited α] (label : String) :
    IO Unit := do
  let (e, f) := fieldWidths (Scalar.precision α)
  let xs : List α := (specials e f).map Numeric.Word.ofNatBits
  for (name, t) in [("leaf", BTree.leaf), ("random", BTree.random (scalarOf (α := α)) 20000 17),
      ("caterpillar", BTree.caterpillar (scalarOf 5) 200), ("specials", spine xs)] do
    checkDevice s!"{label} {name}" BTree.sig (arenaB t) decodeB (some (t.map bits))
  for (name, t) in [("random", Rose.random (scalarOf (α := α)) 20000 19),
      ("specials", Rose.node (scalarOf 3) (xs.map (Rose.node · [])))] do
    checkDevice s!"{label} rose {name}" Rose.sig (arenaR t) decodeR (some (t.map bits))

/-- Checks that the runtime refuses a handle read with another scalar storage, width or
descriptor, while it reads a handle with its own. -/
def checkHandles : IO Unit := do
  let p := Scalar.precision Float
  let d ← Arena.upload BTree.sig (arenaB (BTree.perfect (2.5 : Float) 4))
  let reads : List (String × UInt32 × String × UInt64) := [("binary32", 0, "binary32", 4),
    ("another width", 1, "binary64", 4), ("another descriptor", 1, "binary32", 8),
    ("byte storage", 2, p.descriptor, 8)]
  for (name, format, descriptor, width) in reads do
    ensureFails s!"binary64 read as {name}" "another layout"
      (d.handle.words format descriptor width)
    ensureFails s!"binary64 scalars read as {name}" "another layout"
      (d.handle.scalars format descriptor width)
  let half ← Arena.upload BTree.sig (arenaB (BTree.perfect (scalarOf 3 : ExecFloat.Binary 5 10) 4))
  let bfloat := Scalar.precision (ExecFloat.Binary 8 7)
  let halfPrecision := Scalar.precision (ExecFloat.Binary 5 10)
  ensure (bfloat.tag == halfPrecision.tag && bfloat.bytes == halfPrecision.bytes)
    "handles: bfloat16 and binary16 have different storage"
  ensureFails "binary16 read as bfloat16" "another layout"
    (half.handle.scalars bfloat.tag bfloat.descriptor 2)

/-- Checks that the runtime refuses uploads whose blocks do not match their description, whose
sizes overflow or whose scalar layout it cannot store, and that it keeps empty blocks. -/
def checkUploads : IO Unit := do
  let p := Scalar.precision Float
  let zeros (n : Nat) : ByteArray := ⟨Array.replicate n 0⟩
  -- One row with two index slots and one binary64 slot takes 24 and 8 bytes.
  let up := Arena.Handle.ofBlocks p.tag p.descriptor
  discard <| up 8 1 2 1 (zeros 24) (zeros 8)
  let uploads : List (String × String × IO Arena.Handle) := [
    ("a short words block", "words block", up 8 1 2 1 (zeros 16) (zeros 8)),
    ("a long scalar block", "scalar block", up 8 1 2 1 (zeros 24) (zeros 16)),
    ("too many rows", "overflow", up 8 (2 ^ 62) 3 1 (zeros 0) (zeros 0)),
    ("too many index slots", "overflow", up 8 1 0xFFFFFFFFFFFFFFFF 1 (zeros 0) (zeros 8)),
    ("an unknown storage", "invalid scalar layout",
      Arena.Handle.ofBlocks 3 p.descriptor 8 1 2 1 (zeros 24) (zeros 8)),
    ("empty scalar words", "invalid scalar layout",
      Arena.Handle.ofBlocks 2 p.descriptor 0 1 2 1 (zeros 24) (zeros 0)),
    ("binary64 in four bytes", "native scalar width", up 4 1 2 1 (zeros 24) (zeros 4))]
  for (name, expected, upload) in uploads do
    ensureFails s!"upload of {name}" expected upload
  let empty ← up 8 0 2 1 (zeros 0) (zeros 0)
  let words ← empty.words p.tag p.descriptor 8
  let scalars ← empty.scalars p.tag p.descriptor 8
  ensure (words.size == 0 && scalars.size == 0) "uploads: empty blocks came back non-empty"

/-- Checks that the runtime keeps every special encoding of the native formats, NaN payloads
included, which Lean's `Float` and `Float32` do not hold. -/
def checkNativeSpecials : IO Unit := do
  for p in [Scalar.precision Float, Scalar.precision Float32] do
    let (e, f) := fieldWidths p
    let xs := specials e f
    let scalars := bytesLE p.bytes xs.toArray
    let words : ByteArray := ⟨Array.replicate (8 * xs.length) 0⟩
    let h ← Arena.Handle.ofBlocks p.tag p.descriptor p.bytes.toUInt64 xs.length.toUInt64 0 1
      words scalars
    let back ← h.scalars p.tag p.descriptor p.bytes.toUInt64
    ensure (back.data == scalars.data) s!"{p.descriptor}: the runtime changed a special encoding"

/-- The device checks: the refusals before the runtime and in it, the special encodings of the
native formats, round trips for each scalar type, and the empty batch. -/
def runCUDA : IO Unit := do
  checkRefusals
  checkHandles
  checkUploads
  checkNativeSpecials
  checkDeviceTrees (α := Float) "binary64"
  checkDeviceTrees (α := Float32) "binary32"
  checkDeviceTrees (α := ExecFloat.Binary 5 10) "binary16"
  checkDeviceTrees (α := ExecFloat.Binary 5 6) "12-bit"
  checkDeviceTrees (α := ExecFloat.Binary 5 11) "17-bit"
  checkDeviceTrees (α := ExecFloat.Binary 11 53) "65-bit"
  checkDeviceTrees (α := ExecFloat.Binary 15 240) "256-bit"
  checkDevice "empty batch" BTree.sig (batchB ([] : List (BTree Float))) decodeB none
  IO.println "  tree arenas: device round trips are byte for byte, bad reads and uploads refused"

/-- The checks of builds without LibTorch: the refusals before the runtime, and an upload that
fails with the runtime's message. -/
def runUnlinked : IO Unit := do
  checkRefusals
  ensureFails "upload without LibTorch" "built without LibTorch"
    (Arena.upload BTree.sig (arenaB (BTree.perfect (1.0 : Float) 3)))
  IO.println "  tree arenas: uploads refused without LibTorch"

end TorchLean.Test.Tree.Arenas
