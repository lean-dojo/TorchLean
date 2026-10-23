/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Data.IO.Npy

/-!
# NPY loader tests

Regression checks for the `.npy` payload decoders. The files are synthesised in memory, so the
expected values are known exactly; the file-backed readers are exercised on the same bytes written
to a scratch fixture.

The checks cover three properties:

- every decode path agrees bit for bit, not approximately. The decoders reinterpret stored bits
  rather than computing, so equality is the honest test and a tolerance would hide exactly the
  transcription bugs this is meant to catch;
- the values `==` on `Float` cannot see are checked from their bit patterns: both signed zeros,
  both infinities and the extremes of either format exactly, and NaNs as the canonical NaN that
  Lean's `Float.ofBits` makes of every NaN pattern, sign and payload included;
- the concurrent leading-axis path agrees with the sequential one, and agrees with itself across
  repeated runs. Blocks are decoded on separate tasks and reassembled by index, so a
  scheduling-order dependence would show up as a block permutation.
-/

@[expose] public section

namespace NN.Tests.Data.IO.Npy

open TorchLean.Data.IO
open TorchLean.Data.IO.Internal

def expect (tag : String) (ok : Bool) : IO Unit := do
  unless ok do
    throw <| IO.userError s!"npy loader check failed: {tag}"

/-! ## Synthetic `.npy` construction -/

/-- Append a little-endian `UInt64`. -/
def pushU64LE (ba : ByteArray) (w : UInt64) : ByteArray := Id.run do
  let mut out := ba
  for k in [0:8] do
    out := out.push (UInt8.ofNat ((w >>> (UInt64.ofNat (8 * k))) &&& 0xff).toNat)
  pure out

/-- Append a little-endian `UInt32`. -/
def pushU32LE (ba : ByteArray) (w : UInt32) : ByteArray := Id.run do
  let mut out := ba
  for k in [0:4] do
    out := out.push (UInt8.ofNat ((w >>> (UInt32.ofNat (8 * k))) &&& 0xff).toNat)
  pure out

/-- Build a version-1 `.npy` header block, padded to the 64-byte alignment NumPy uses. -/
def header (descr : String) (shape : List Nat) (fortran : Bool) : ByteArray := Id.run do
  let dims :=
    match shape with
    | [] => "()"
    | [d] => s!"({d},)"
    | _ => "(" ++ String.intercalate ", " (shape.map toString) ++ ")"
  let order := if fortran then "True" else "False"
  let dict := s!"\{'descr': '{descr}', 'fortran_order': {order}, 'shape': {dims}, }"
  -- magic (6) + version (2) + length (2) + dict + '\n', padded to a multiple of 64
  let unpadded := 10 + dict.length + 1
  let pad := (64 - unpadded % 64) % 64
  let body := dict ++ String.ofList (List.replicate pad ' ') ++ "\n"
  let mut ba := ByteArray.empty
  ba := ba.push 0x93
  for c in "NUMPY".toList do ba := ba.push (UInt8.ofNat c.toNat)
  ba := ba.push 1
  ba := ba.push 0
  ba := ba.push (UInt8.ofNat (body.length % 256))
  ba := ba.push (UInt8.ofNat (body.length / 256))
  for c in body.toList do ba := ba.push (UInt8.ofNat c.toNat)
  pure ba

/-- Deterministic test values spanning several exponents. -/
def sample (i : Nat) : Float :=
  let x := Float.ofNat (i % 9973) + 0.5
  x * Float.exp (Float.ofNat (i % 17) - 8.0) - 3.25

/-- A `<f8` file holding `sample 0 … sample (count-1)` in C order. -/
def buildF8 (shape : List Nat) : ByteArray := Id.run do
  let count := shape.foldl (fun acc n => acc * n) 1
  let mut ba := header "<f8" shape false
  for i in [0:count] do
    ba := pushU64LE ba (sample i).toBits
  pure ba

/-- A `<f4` file holding the `Float32` roundings of the same samples. -/
def buildF4 (shape : List Nat) : ByteArray := Id.run do
  let count := shape.foldl (fun acc n => acc * n) 1
  let mut ba := header "<f4" shape false
  for i in [0:count] do
    ba := pushU32LE ba (sample i).toFloat32.toBits
  pure ba

/-! ## Comparisons -/

/-- Bit-for-bit equality. `==` on `Float` is the wrong test here: it identifies `0.0` with
`-0.0` and makes every `NaN` unequal to itself, so it would both miss sign-bit transcription
errors and reject a correctly decoded `NaN`. -/
def bitsEq (a b : Float) : Bool := a.toBits == b.toBits

def faBitsEq (a b : FloatArray) : Bool := Id.run do
  if a.size != b.size then
    return false
  let mut ok := true
  for i in [0:a.size] do
    unless bitsEq (a[i]!) (b[i]!) do
      ok := false
  pure ok

def ofExcept {α : Type} (tag : String) : Except String α → IO α
  | .ok a => pure a
  | .error e => throw <| IO.userError s!"npy loader check failed: {tag}: {e}"

def isError {α : Type} : Except String α → Bool
  | .ok _ => false
  | .error _ => true

/-- Flatten decoded blocks back into one payload, for comparison against `parseNpy`. -/
def concatBlocks (blocks : Array FloatArray) : FloatArray := Id.run do
  let total := blocks.foldl (fun acc b => acc + b.size) 0
  let mut out := FloatArray.emptyWithCapacity total
  for b in blocks do
    for i in [0:b.size] do
      out := out.push (b[i]!)
  pure out

/-! ## Tests -/

/-- Every element survives the round trip exactly, for both supported dtypes. -/
def testRoundTrip : IO Unit := do
  let shape := [3, 5, 7]
  let count := 105
  let d8 ← ofExcept "f8 parse" (parseNpy "t" (buildF8 shape))
  expect "f8 dtype" (d8.dtype == "<f8")
  expect "f8 shape" (d8.shape.toList == shape)
  expect "f8 size" (d8.values.size == count)
  for i in [0:count] do
    expect s!"f8 element {i}" (bitsEq (d8.values[i]!) (sample i))
  let d4 ← ofExcept "f4 parse" (parseNpy "t" (buildF4 shape))
  expect "f4 size" (d4.values.size == count)
  for i in [0:count] do
    expect s!"f4 element {i}" (bitsEq (d4.values[i]!) (sample i).toFloat32.toFloat)
  let scalar ← ofExcept "scalar parse" (parseNpy "t" (buildF8 []))
  expect "scalar shape" scalar.shape.isEmpty
  expect "scalar value" (scalar.values.size == 1 && bitsEq (scalar.values[0]!) (sample 0))

/-! ## Special values -/

/-- A `<f8` file holding exactly these bit patterns, in C order. -/
def buildF8Bits (shape : List Nat) (bits : Array UInt64) : ByteArray := Id.run do
  let mut ba := header "<f8" shape false
  for w in bits do
    ba := pushU64LE ba w
  pure ba

/-- A `<f4` file holding exactly these bit patterns, in C order. -/
def buildF4Bits (shape : List Nat) (bits : Array UInt32) : ByteArray := Id.run do
  let mut ba := header "<f4" shape false
  for w in bits do
    ba := pushU32LE ba w
  pure ba

/-- The `<f8` patterns a decoder can get wrong without `==` on `Float` noticing: both signed
zeros, both infinities, quiet NaNs of either sign, NaNs carrying a payload, and the largest
finite and smallest subnormal magnitudes of either sign. Twelve, so the block path cuts them
into three blocks of four. -/
def specialF8 : Array UInt64 :=
  #[0x0000000000000000, 0x8000000000000000,   -- +0.0, -0.0
    0x7ff0000000000000, 0xfff0000000000000,   -- +inf, -inf
    0x7ff8000000000000, 0xfff8000000000000,   -- quiet NaN, negative quiet NaN
    0x7ff00000deadbeef, 0xfff00000deadbeef,   -- NaN with a payload, either sign
    0x7fefffffffffffff, 0xffefffffffffffff,   -- largest finite, either sign
    0x0000000000000001, 0x8000000000000001]   -- smallest subnormal, either sign

/-- The same twelve patterns in `<f4`. -/
def specialF4 : Array UInt32 :=
  #[0x00000000, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00000, 0xffc00000,
    0x7f80beef, 0xff80beef, 0x7f7fffff, 0xff7fffff, 0x00000001, 0x80000001]

/-- The sign bit of a value, read from its bits so that it is defined for `-0.0`. -/
def signBit (x : Float) : Bool := x.toBits >>> 63 == 1

/-- The canonical quiet NaN: what Lean's `Float.ofBits` returns for every NaN bit pattern. -/
def canonicalNaN : UInt64 := 0x7ff8000000000000

/-- Whether entry `i` of `specialF8` and `specialF4` is a NaN pattern. -/
def isNaNPattern (i : Nat) : Bool := 4 ≤ i && i < 8

/-- The patterns `==` on `Float` cannot see survive every decode path, with one exception the
runtime imposes: a NaN's sign and payload. `Float.ofBits` and `Float32.ofBits` return the
canonical quiet NaN for every NaN pattern, so no decoder over `Float` can keep those bits; what
the decoders preserve, and what is checked here, is that a NaN decodes as a NaN. The canonical
result is pinned so that a change in the runtime shows up here rather than downstream.

For `<f8` the stored bits are the decoded bits, except for the four NaNs. For `<f4` the decoded
value is the stored `Float32` widened to `Float`, so the bits are checked against that widening,
and what the widening must preserve is checked on the decoded values themselves: the sign of each
zero, the NaN-ness of each NaN, the exact infinities, and the magnitude of the extremes. -/
def testSpecialValues (fixtureDirectory : System.FilePath) : IO Unit := do
  let shape := [3, 4]
  let n := specialF8.size
  let bs8 := buildF8Bits shape specialF8
  let bs4 := buildF4Bits shape specialF4
  -- `<f8`: exact bits, flat and by blocks, in memory and from a file.
  let d8 ← ofExcept "special f8 parse" (parseNpy "t" bs8)
  expect "special f8 size" (d8.values.size == n)
  for i in [0:n] do
    if isNaNPattern i then
      expect s!"special f8 element {i} is the canonical NaN"
        ((d8.values[i]!).isNaN && (d8.values[i]!).toBits == canonicalNaN)
    else
      expect s!"special f8 element {i}" ((d8.values[i]!).toBits == specialF8[i]!)
  let b8 ← ofExcept "special f8 blocks" (parseNpyBlocks "t" bs8)
  expect "special f8 blocks = flat" (faBitsEq (concatBlocks b8.blocks) d8.values)
  let path8 := fixtureDirectory / "nn_data_io_npy_special_f8.npy"
  IO.FS.writeBinFile path8 bs8
  let f8 ← ofExcept "special f8 file" (← readNpy path8)
  expect "special f8 readNpy = parseNpy" (faBitsEq f8.values d8.values)
  let fb8 ← ofExcept "special f8 file blocks" (← readNpyBlocks path8)
  expect "special f8 file blocks = flat" (faBitsEq (concatBlocks fb8.blocks) d8.values)
  -- `<f4`: the widening, flat and by blocks, in memory and from a file.
  let d4 ← ofExcept "special f4 parse" (parseNpy "t" bs4)
  expect "special f4 size" (d4.values.size == n)
  for i in [0:n] do
    expect s!"special f4 element {i}"
      (bitsEq (d4.values[i]!) (Float32.ofBits specialF4[i]!).toFloat)
  let b4 ← ofExcept "special f4 blocks" (parseNpyBlocks "t" bs4)
  expect "special f4 blocks = flat" (faBitsEq (concatBlocks b4.blocks) d4.values)
  let path4 := fixtureDirectory / "nn_data_io_npy_special_f4.npy"
  IO.FS.writeBinFile path4 bs4
  let f4 ← ofExcept "special f4 file" (← readNpy path4)
  expect "special f4 readNpy = parseNpy" (faBitsEq f4.values d4.values)
  let fb4 ← ofExcept "special f4 file blocks" (← readNpyBlocks path4)
  expect "special f4 file blocks = flat" (faBitsEq (concatBlocks fb4.blocks) d4.values)
  -- What the widening must preserve, checked on the decoded values themselves.
  let v := d4.values
  expect "f4 +0.0 is zero with the sign clear" (v[0]! == 0.0 && !signBit v[0]!)
  expect "f4 -0.0 is zero with the sign set" (v[1]! == 0.0 && signBit v[1]!)
  expect "f4 +inf" ((v[2]!).toBits == 0x7ff0000000000000)
  expect "f4 -inf" ((v[3]!).toBits == 0xfff0000000000000)
  for i in [4:8] do
    expect s!"f4 element {i} is the canonical NaN, whatever its sign or payload"
      ((v[i]!).isNaN && (v[i]!).toBits == canonicalNaN)
  expect "f4 largest finite" ((v[8]!).isFinite && v[8]! > 3.4028e38 && v[8]! < 3.40283e38)
  expect "f4 negative largest finite" ((v[9]!).isFinite && v[9]! < -3.4028e38)
  expect "f4 smallest subnormal" (v[10]! > 0.0 && v[10]! < 1.5e-45)
  expect "f4 negative smallest subnormal" (v[11]! < 0.0 && v[11]! > -1.5e-45)

/-- The leading-axis blocks are exactly the payload, cut at the leading-axis stride. -/
def testBlocksMatchFlat (shape : List Nat) (tag : String) : IO Unit := do
  let bs := buildF8 shape
  let flat ← ofExcept s!"{tag} flat" (parseNpy "t" bs)
  let blk ← ofExcept s!"{tag} blocks" (parseNpyBlocks "t" bs)
  let d0 := shape.headD 0
  let blockSize := (shape.drop 1).foldl (fun acc n => acc * n) 1
  expect s!"{tag} block count" (blk.blocks.size == d0)
  expect s!"{tag} block sizes" (blk.blocks.all (fun b => b.size == blockSize))
  expect s!"{tag} blocks = flat" (faBitsEq (concatBlocks blk.blocks) flat.values)

/-- Every task count decodes the same payload, bit for bit.

This is the property that makes the task count a free choice: it may be tuned for a machine
without changing what the loader returns. A block permutation, an off-by-one in the block
grouping, or a torn write would all break it. The sweep deliberately includes counts above
the block count and above the core count, which `resolveTaskCount` clamps. -/
def testParallelismAgrees : IO Unit := do
  -- Keep all eight blocks large enough to exercise eight tasks, not just two.
  let shape := [8, 257, 257]
  let count := shape.foldl (fun acc n => acc * n) 1
  expect "test payload exercises eight tasks" (resolveTaskCount (.tasks 8) 8 count == 8)
  let bs := buildF8 shape
  let flat ← ofExcept "parallel flat" (parseNpy "t" bs)
  let seq ← ofExcept "sequential blocks" (parseNpyBlocks "t" bs .sequential)
  expect "sequential = flat" (faBitsEq (concatBlocks seq.blocks) flat.values)
  for n in [0, 1, 2, 3, 4, 5, 7, 8, 9, 16, 64] do
    let got ← ofExcept s!"tasks {n}" (parseNpyBlocks "t" bs (.tasks n))
    expect s!"tasks {n} block count" (got.blocks.size == seq.blocks.size)
    for b in [0:seq.blocks.size] do
      expect s!"tasks {n} block {b}" (faBitsEq (got.blocks[b]!) (seq.blocks[b]!))
  let auto ← ofExcept "auto" (parseNpyBlocks "t" bs .auto)
  expect "auto block count" (auto.blocks.size == seq.blocks.size)
  for b in [0:seq.blocks.size] do
    expect s!"auto block {b}" (faBitsEq (auto.blocks[b]!) (seq.blocks[b]!))

/-- Repeated concurrent decodes of one payload agree, so the result does not depend on the
order in which tasks happen to finish. -/
def testConcurrentDeterminism : IO Unit := do
  let shape := [8, 257, 257]
  let bs := buildF8 shape
  let first ← ofExcept "concurrent first" (parseNpyBlocks "t" bs .auto)
  for r in [0:8] do
    let again ← ofExcept s!"concurrent repeat {r}"
      (parseNpyBlocks s!"concurrent repeat {r}" bs .auto)
    expect s!"concurrent repeat {r} block count" (again.blocks.size == first.blocks.size)
    for b in [0:first.blocks.size] do
      expect s!"concurrent repeat {r} block {b}"
        (faBitsEq (again.blocks[b]!) (first.blocks[b]!))

/-- The task count respects its three bounds: the request, the block count, and the payload
size. In particular a small payload stays on the calling thread whatever is requested. -/
def testTaskCountBounds : IO Unit := do
  let big := 1 <<< 20
  expect "sequential is one task" (resolveTaskCount .sequential 64 big == 1)
  expect "never more tasks than blocks" (resolveTaskCount (.tasks 64) 8 big == 8)
  expect "honours a request below the block count" (resolveTaskCount (.tasks 4) 64 big == 4)
  expect "small payload stays on one thread" (resolveTaskCount (.tasks 64) 64 1024 == 1)
  expect "task count is at least one" (resolveTaskCount (.tasks 0) 64 big == 1)
  expect "auto is bounded by the block count" (resolveTaskCount .auto 2 big <= 2)

/-- Fortran-order payloads are still reordered into C order by `parseNpy`. -/
def testFortranOrder : IO Unit := do
  let shape := [2, 3]
  -- Fortran storage of [[a b c], [d e f]] is a d b e c f
  let cOrder : Array Float := #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  let mut ba := header "<f8" shape true
  for v in #[1.0, 4.0, 2.0, 5.0, 3.0, 6.0] do
    ba := pushU64LE ba (Float.toBits v)
  let d ← ofExcept "fortran parse" (parseNpy "t" ba)
  expect "fortran flag cleared" (d.fortran == false)
  for i in [0:6] do
    expect s!"fortran element {i}" (bitsEq (d.values[i]!) (cOrder[i]!))

/-- Malformed and unsupported files are rejected rather than silently decoded. -/
def testRejections : IO Unit := do
  let good := buildF8 [4, 4]
  expect "rejects bad magic" (isError (parseNpy "t" (good.set! 1 0x00)))
  expect "rejects truncated payload" (isError (parseNpy "t" (good.extract 0 (good.size - 8))))
  expect "rejects truncated blocks"
    (isError (parseNpyBlocks "t" (good.extract 0 (good.size - 8))))
  expect "rejects unsupported dtype" (isError (parseNpy "t" (header "<i4" [4] false)))
  expect "rejects fortran order for blocks"
    (isError (parseNpyBlocks "t" (header "<f8" [2, 2] true)))
  expect "rejects scalar shape for blocks" (isError (parseNpyBlocks "t" (buildF8 [])))

/-- The file-backed readers return the same bits as the in-memory parsers, and reject a file whose
header is unfit for block loading before reading its payload. -/
def testFileReaders (fixtureDirectory : System.FilePath) : IO Unit := do
  let path := fixtureDirectory / "nn_data_io_npy.npy"
  let bs := buildF8 [6, 5, 7]
  IO.FS.writeBinFile path bs
  let flat ← ofExcept "file flat" (← readNpy path)
  let parsed ← ofExcept "memory flat" (parseNpy "t" bs)
  expect "readNpy = parseNpy" (faBitsEq flat.values parsed.values)
  let fromFile ← ofExcept "file blocks" (← readNpyBlocks path)
  let fromMemory ← ofExcept "memory blocks" (parseNpyBlocks "t" bs)
  expect "file block count" (fromFile.blocks.size == fromMemory.blocks.size)
  for b in [0:fromMemory.blocks.size] do
    expect s!"file block {b}" (faBitsEq (fromFile.blocks[b]!) (fromMemory.blocks[b]!))
  IO.FS.writeBinFile path (bs.extract 0 (bs.size - 8))
  expect "file rejects truncated blocks" (isError (← readNpyBlocks path))
  -- The header alone is written: a payload-free Fortran-order file must fail on the storage order,
  -- which shows the check runs before the payload read rather than as a truncation error.
  IO.FS.writeBinFile path (header "<f8" [2, 2] true)
  match ← readNpyBlocks path with
  | .ok _ => expect "file rejects fortran order for blocks" false
  | .error message =>
      expect "fortran rejection names the storage order" ((message.splitOn "C-order").length > 1)
  IO.FS.writeBinFile path (header "<f8" [] false)
  match ← readNpyBlocks path with
  | .ok _ => expect "file rejects scalar shape for blocks" false
  | .error message =>
      expect "scalar rejection names the leading axis"
        ((message.splitOn "leading axis").length > 1)

/-- Prefix readers decode only the requested C-order rows, even when later rows are absent. -/
def testPrefixReaders (fixtureDirectory : System.FilePath) : IO Unit := do
  let path := fixtureDirectory / "nn_data_io_npy_prefix.npy"
  for bytes in #[buildF8 [6, 5], buildF4 [6, 5]] do
    let full ← ofExcept "prefix reference" (parseNpy "t" bytes)
    let hdr ← ofExcept "prefix header" (parseNpyHeaderMeta "t" bytes)
    let prefixOnly := bytes.extract 0 (hdr.dataStart + 10 * hdr.elementBytes)
    let selected ← ofExcept "memory prefix" (parseNpyPrefix "t" #[2, 5] prefixOnly)
    expect "prefix shape" (selected.shape == #[2, 5])
    expect "prefix size" (selected.values.size == 10)
    for i in [:10] do
      expect s!"prefix element {i}" (bitsEq selected.values[i]! full.values[i]!)
    expect "full reader rejects missing later rows" (isError (parseNpy "t" prefixOnly))
    expect "prefix rejects wrong trailing shape"
      (isError (parseNpyPrefix "t" #[2, 4] bytes))
    expect "prefix rejects too many rows" (isError (parseNpyPrefix "t" #[7, 5] bytes))
    expect "prefix rejects missing requested data"
      (isError (parseNpyPrefix "t" #[2, 5] (prefixOnly.extract 0 (prefixOnly.size - 1))))
    IO.FS.writeBinFile path prefixOnly
    let fromFile ← ofExcept "file prefix" (← readNpyPrefix path #[2, 5])
    expect "file prefix shape" (fromFile.shape == selected.shape)
    expect "file prefix = memory prefix" (faBitsEq fromFile.values selected.values)

def run : IO Unit := do
  IO.println "-- npy loader --"
  let fixtureDirectory : System.FilePath := ".lake/build/tmp"
  IO.FS.createDirAll fixtureDirectory
  testRoundTrip
  testSpecialValues fixtureDirectory
  testBlocksMatchFlat [4, 6] "2d"
  testBlocksMatchFlat [3, 5, 7] "3d"
  testBlocksMatchFlat [5] "1d"
  testBlocksMatchFlat [3, 0] "empty blocks"
  testBlocksMatchFlat [0, 3] "no blocks"
  testParallelismAgrees
  testConcurrentDeterminism
  testTaskCountBounds
  testFortranOrder
  testRejections
  testFileReaders fixtureDirectory
  testPrefixReaders fixtureDirectory
  IO.println "npy loader: ok"

end NN.Tests.Data.IO.Npy
