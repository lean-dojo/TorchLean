/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Data.IO.Npy

/-!
# NPY loader tests

Regression checks for the `.npy` payload decoders. The files are synthesised in memory, so
the suite needs no fixtures on disk and the expected values are known exactly.

Two properties carry the weight here:

- every decode path agrees **bit for bit**, not approximately. The decoders reinterpret
  stored bits rather than computing, so equality is the honest test and a tolerance would
  hide exactly the transcription bugs this is meant to catch;
- the concurrent leading-axis path agrees with the sequential one, and agrees with itself
  across repeated runs. Blocks are decoded on separate tasks and reassembled by index, so a
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
    | [d] => s!"({d},)"
    | _ => "(" ++ String.intercalate ", " (shape.map toString) ++ ",)"
  let dict := s!"\{'descr': '{descr}', 'fortran_order': {if fortran then "True" else "False"}, 'shape': {dims}, }"
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

/-- A deterministic value for element `i`, spread across exponents so the byte pattern of
each element differs in every byte. -/
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
    unless bitsEq (a.get! i) (b.get! i) do
      ok := false
  pure ok

def ofExcept {α : Type} (tag : String) : Except String α → IO α
  | .ok a => pure a
  | .error e => throw <| IO.userError s!"npy loader check failed: {tag}: {e}"

/-- Flatten decoded blocks back into one payload, for comparison against `parseNpy`. -/
def concatBlocks (blocks : Array FloatArray) : FloatArray := Id.run do
  let total := blocks.foldl (fun acc b => acc + b.size) 0
  let mut out := FloatArray.emptyWithCapacity total
  for b in blocks do
    for i in [0:b.size] do
      out := out.push (b.get! i)
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
    expect s!"f8 element {i}" (bitsEq (d8.values.get! i) (sample i))
  let d4 ← ofExcept "f4 parse" (parseNpy "t" (buildF4 shape))
  expect "f4 size" (d4.values.size == count)
  for i in [0:count] do
    expect s!"f4 element {i}" (bitsEq (d4.values.get! i) (sample i).toFloat32.toFloat)

/-- The leading-axis blocks are exactly the payload, cut at the leading-axis stride. -/
def testBlocksMatchFlat (shape : List Nat) (tag : String) : IO Unit := do
  let bs := buildF8 shape
  let flat ← ofExcept s!"{tag} flat" (parseNpy "t" bs)
  let blk ← ofExcept s!"{tag} blocks" (parseNpyLeadingAxisBlocks "t" bs)
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
  -- large enough that `resolveTaskCount` is not clamped to 1 by `minElementsPerTask`
  let shape := [8, 137, 131]
  let count := shape.foldl (fun acc n => acc * n) 1
  expect "test payload exceeds the sequential threshold" (count / minElementsPerTask >= 2)
  let bs := buildF8 shape
  let flat ← ofExcept "parallel flat" (parseNpy "t" bs)
  let seq ← ofExcept "sequential blocks" (parseNpyLeadingAxisBlocks "t" bs .sequential)
  expect "sequential = flat" (faBitsEq (concatBlocks seq.blocks) flat.values)
  for n in [1, 2, 3, 4, 5, 7, 8, 9, 16, 64] do
    let got ← ofExcept s!"tasks {n}" (parseNpyLeadingAxisBlocks "t" bs (.tasks n))
    expect s!"tasks {n} block count" (got.blocks.size == seq.blocks.size)
    for b in [0:seq.blocks.size] do
      expect s!"tasks {n} block {b}" (faBitsEq (got.blocks[b]!) (seq.blocks[b]!))
  let auto ← ofExcept "auto" (parseNpyLeadingAxisBlocks "t" bs .auto)
  expect "auto block count" (auto.blocks.size == seq.blocks.size)
  for b in [0:seq.blocks.size] do
    expect s!"auto block {b}" (faBitsEq (auto.blocks[b]!) (seq.blocks[b]!))

/-- Repeated concurrent decodes of one payload agree, so the result does not depend on the
order in which tasks happen to finish. -/
def testConcurrentDeterminism : IO Unit := do
  let shape := [8, 137, 131]
  let bs := buildF8 shape
  let first ← ofExcept "concurrent first" (parseNpyLeadingAxisBlocks "t" bs .auto)
  for r in [0:8] do
    let again ← ofExcept s!"concurrent repeat {r}" (parseNpyLeadingAxisBlocks "t" bs .auto)
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
    expect s!"fortran element {i}" (bitsEq (d.values.get! i) (cOrder[i]!))

/-- Malformed and unsupported files are rejected rather than silently decoded. -/
def testRejections : IO Unit := do
  let good := buildF8 [4, 4]
  let isData : Except String NpyData → Bool := fun r => match r with | .error _ => false | .ok _ => true
  let isBlocks : Except String NpyBlocks → Bool := fun r => match r with | .error _ => false | .ok _ => true
  expect "rejects bad magic"
    (!isData (parseNpy "t" (good.set! 1 0x00)))
  expect "rejects truncated payload"
    (!isData (parseNpy "t" (good.extract 0 (good.size - 8))))
  expect "rejects unsupported dtype"
    (!isData (parseNpy "t" (header "<i4" [4] false)))
  expect "rejects fortran order for blocks"
    (!isBlocks (parseNpyLeadingAxisBlocks "t" (header "<f8" [2, 2] true)))
  expect "rejects scalar shape for blocks"
    (!isBlocks (parseNpyLeadingAxisBlocks "t" (buildF8 [])))

def run : IO Unit := do
  IO.println "-- npy loader --"
  testRoundTrip
  testBlocksMatchFlat [4, 6] "2d"
  testBlocksMatchFlat [3, 5, 7] "3d"
  testBlocksMatchFlat [5] "1d"
  testParallelismAgrees
  testConcurrentDeterminism
  testTaskCountBounds
  testFortranOrder
  testRejections
  IO.println "npy loader: ok"

end NN.Tests.Data.IO.Npy
