/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.LibTorch.Buffer
public import NN.Runtime.Autograd.Engine.LibTorch.MX

/-!
# CUDA Kernel Coverage: OCP MX tensors

Compares the native MX operations with FloatLib's `MX.Standard`, the reference they implement.

- Decoding: every code of every profile, at the scale codes 0, 1, 126, 127, 128, 254 and 255,
  lane for lane against `Block.decodeLane`, bit for bit (zero signs included) where the value lies
  in binary32's finite range, `±∞` beyond it, NaN for NaN.
- Conversion: blocks of each profile's finite values and of the midpoints between neighbours
  (every rounding tie, half the smallest subnormal), values past the largest finite one and the
  FP8 OVF bounds, all-zero blocks, blocks whose exponent clamps at `-127`, non-finite lanes, and
  pseudo-random binary32 blocks, against `quantize` at §6.3's scales and `quantizeAt` at given
  scales, under SAT and OVF for FP8, code for code and scale for scale.
- Each refusal beside an input of the same shape that passes, an empty tensor, and the handle's
  finalizer. The native side's own refusal of a count over `UInt32`'s range is unreachable here:
  `Shape.native` refuses such a shape before any native call.

A block whose §6.3 exponent passes 127 cannot be built from binary32 input: `⌊log₂⌋` of a binary32
value is at most 127 and every profile's largest power-of-two exponent is at least 0. A given
scale of 255 covers the NaN block instead.
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace MX

open Runtime.Autograd.LibTorch
open Runtime.Autograd.LibTorch.MX (Shape Tensor)
open FloatLib.Floats.Formats.OCP.MX.Standard
open FloatLib.Floats.Formats.OCP.MX (E8M0)
open FloatLib.Numerics

def profiles : List Profile := [.e5m2, .e4m3, .e3m2, .e2m3, .e2m1, .int8]

def infinity : Float := 1.0 / 0.0
def nan : Float := 0.0 / 0.0

/-- The binary64 value of a dyadic rational whose numerator and denominator are exact in
binary64: every value these checks build. -/
def exactFloat (q : Rat) : Float := Float.ofInt q.num / Float.ofNat q.den

/-- Equal as IEEE values, the sign of zero included, any NaN equal to any NaN. -/
def sameFloat (a b : Float) : Bool :=
  (a.isNaN && b.isNaN) || (a == b && (a.toBits >>> 63) == (b.toBits >>> 63))

/-- The binary32 result decoding must give for a FloatLib lane value. -/
def expectedLane : NumericalValue SignedRat → Float
  | .finite x =>
    if x.value = 0 then (if x.negative then -0.0 else 0.0)
    else if |x.value| ≤ Runtime.Autograd.LibTorch.MX.binary32Max then exactFloat x.value
    else if x.value < 0 then -infinity else infinity
  | .infinity negative => if negative then -infinity else infinity
  | .exceptional _ => nan

/-- Codes in the layout the native side stores: E2M1 two to a byte, element 0 low. -/
def pack (p : Profile) (codes : Array Nat) : ByteArray :=
  if p = .e2m1 then
    ⟨(Array.range (codes.size / 2)).map fun i =>
      (codes[2 * i]! + 16 * codes[2 * i + 1]!).toUInt8⟩
  else ⟨codes.map (·.toUInt8)⟩

def shapeOf (p : Profile) (blocks : Nat) : Shape :=
  ⟨p, blocks, 32, by decide, by decide⟩

/-- Block `b` of `codes` at one scale; the codes come in whole blocks of 32. -/
def blockOf (p : Profile) (scale : Nat) (codes : Array Nat) (b : Nat) : Option (Block p) :=
  Block.ofArray? (E8M0.ofNatBits scale)
    ((codes.extract (32 * b) (32 * b + 32)).map (BitVec.ofNat p.width ·))

def fail (message : String) : IO Unit := throw <| IO.userError s!"MX: {message}"

/-! ## Decoding -/

def checkDecode (p : Profile) : IO Unit := do
  let count := 2 ^ p.width
  let codes : Array Nat := (Array.range (max count 32)).map (· % count)
  let blocks := codes.size / 32
  for scale in [0, 1, 126, 127, 128, 254, 255] do
    let t ← Runtime.Autograd.LibTorch.MX.upload (shapeOf p blocks) (pack p codes)
      ⟨Array.replicate blocks scale.toUInt8⟩
    let actual ← Buffer.toFloatArrayIO (← t.decode)
    unless actual.size = codes.size do fail s!"{repr p} decode size {actual.size}"
    for i in [0:codes.size] do
      let some block := blockOf p scale codes (i / 32) | fail s!"{repr p}: block {i / 32} short"
      let expected := expectedLane (block.decodeLane ⟨i % 32, by omega⟩)
      unless sameFloat actual[i]! expected do
        fail s!"{repr p} code {codes[i]!} at scale {scale}: {actual[i]!} \
          (bits {actual[i]!.toBits}), FloatLib {expected} (bits {expected.toBits})"

/-! ## Conversion -/

/-- One lane of a conversion input. -/
inductive Lane where
  | value (q : Rat)
  | negativeZero
  | infinite (negative : Bool)
  | notANumber
  deriving Inhabited

def Lane.float : Lane → Float
  | .value q => exactFloat q
  | .negativeZero => -0.0
  | .infinite negative => if negative then -infinity else infinity
  | .notANumber => nan

def Lane.numerical : Lane → NumericalValue SignedRat
  | .value q => .finite (SignedRat.ofRat q)
  | .negativeZero => .finite SignedRat.negZero
  | .infinite negative => .infinity negative
  | .notANumber => .exceptional .nan

/-- FloatLib's block for 32 lanes: `quantize` at §6.3's scale, or `quantizeAt` at a given one,
and the NaN block for a non-finite lane at either; none for a block of another size. -/
def reference (p : Profile) (lanes : Array Lane) (overflow : OverflowMode)
    (scale : Option Nat) : Option (Block p) :=
  let numerical := lanes.map Lane.numerical
  if h : numerical.size = 32 then
    match scale with
    | none => some (quantize p ⟨numerical, h⟩ overflow)
    | some s =>
      if numerical.all NumericalValue.isFinite then
        some (quantizeAt p (E8M0.ofNatBits s)
          ⟨numerical.map fun v => v.finite?.getD 0, by simp [h]⟩ overflow)
      else some (Block.nan p)
  else none

/-- The first index where two byte arrays differ, or 0. -/
def firstDifference (a b : ByteArray) : Nat :=
  ((Array.range a.size).find? fun i => a[i]! != b[i]!).getD 0

def checkConvert (label : String) (p : Profile) (shape : Shape) (blocks : Array (Array Lane))
    (overflow : OverflowMode) (scales : Option (Array Nat)) : IO Unit := do
  let input ← Buffer.ofFloatArrayIO ⟨blocks.flatMap fun b => b.map Lane.float⟩
  let t ← Runtime.Autograd.LibTorch.MX.quantize input shape overflow
    (scales.map fun given => ⟨given.map (·.toUInt8)⟩)
  let (codes, actualScales) ← t.download
  let expected := (Array.range blocks.size).filterMap fun b =>
    reference p blocks[b]! overflow (scales.map (·[b]!))
  unless expected.size = blocks.size do fail s!"{label} {repr p}: a block is not 32 lanes"
  let expectedCodes := pack p (expected.flatMap fun block => block.values.toArray.map (·.toNat))
  let expectedScales : ByteArray := ⟨expected.map fun block => block.scale.toNat.toUInt8⟩
  unless actualScales.data = expectedScales.data do
    let b := firstDifference actualScales expectedScales
    fail s!"{label} {repr p}: block {b} scale is {actualScales[b]!}, \
      FloatLib's {expectedScales[b]!}"
  unless codes.data = expectedCodes.data do
    let i := firstDifference codes expectedCodes
    fail s!"{label} {repr p}: code byte {i} is {codes[i]!}, FloatLib's {expectedCodes[i]!}"

/-- Lanes in blocks of 32, each led by `anchor`, which fixes the block's §6.3 scale; the last
block is filled with zeros. -/
def chunk (anchor : Lane) (lanes : Array Lane) : Array (Array Lane) :=
  (Array.range ((lanes.size + 30) / 31)).map fun b =>
    (Array.range 32).map fun j =>
      if j = 0 then anchor else (lanes[31 * b + j - 1]?).getD (.value 0)

def finiteValues (p : Profile) : Array Rat :=
  ((Element.candidates p).map (·.2) |>.eraseDups.mergeSort (· ≤ ·)).toArray

/-- Each finite value, each midpoint between neighbours, and values past the largest finite one,
all within the binade `2^maxPowerExponent` fixes; and the FP8 OVF bounds with their neighbours. -/
def boundaryLanes (p : Profile) : Array Lane :=
  let values := finiteValues p
  let midpoints := (Array.range (values.size - 1)).map fun i => (values[i]! + values[i + 1]!) / 2
  let top := values[values.size - 1]!
  let step := top - values[values.size - 2]!
  let past := #[top + step / 2, top + step / 4, top + step].filter
    (· < 2 ^ (p.maxPowerExponent + 1))
  let bounds : Array Rat := match p with
    | .e4m3 => #[464, 464 + 1 / 64, 480]
    | .e5m2 => #[61440, 61440 - 1024, 61440 + 1024]
    | _ => #[]
  let extra := (past ++ bounds).flatMap fun q => #[q, -q]
  (values ++ midpoints ++ extra).map .value |>.push .negativeZero

/-- Pseudo-random finite binary32 values, from a linear congruential sequence of bit patterns. -/
def binary32Of (bits : Nat) : Rat :=
  let exponent : Nat := (bits >>> 23) % 255
  let fraction : Nat := bits % 2 ^ 23
  let magnitude : Rat :=
    if exponent = 0 then (fraction : Rat) * 2 ^ (-149 : Int)
    else ((2 ^ 23 + fraction : Nat) : Rat) * 2 ^ ((exponent : Int) - 150)
  if (bits >>> 31) % 2 = 1 then -magnitude else magnitude

def randomBlocks (seed count : Nat) : Array (Array Lane) := Id.run do
  let mut state := seed
  let mut result := #[]
  for b in [0:count] do
    -- Each block draws from a window of binades, so blocks differ in scale and in spread.
    let base := (b * 37) % 200
    let mut lanes : Array Lane := #[]
    for _ in [0:32] do
      state := (state * 6364136223846793005 + 1442695040888963407) % 2 ^ 64
      let bits := state >>> 32
      let exponent := base + (bits >>> 23) % 24
      lanes := lanes.push (.value (binary32Of (((bits >>> 31) % 2) * 2 ^ 31 +
        exponent * 2 ^ 23 + bits % 2 ^ 23)))
    result := result.push lanes
  return result

def specialBlocks : Array (Array Lane) :=
  let fill (f : Nat → Lane) : Array Lane := (Array.range 32).map f
  #[ fill fun _ => .value 0,
     fill fun j => if j % 2 = 0 then .negativeZero else .value 0,
     fill fun j => .value ((j + 1 : Nat) * 2 ^ (-149 : Int)),
     fill fun j => .value (if j = 3 then 2 ^ (-140 : Int) else -(2 ^ (-147 : Int))),
     fill fun j => if j = 5 then .infinite false else .value 1,
     fill fun j => if j = 31 then .infinite true else .value (-3),
     fill fun j => if j = 0 then .notANumber else .value (1 / 2),
     fill fun j => .value ((2 ^ 24 - 1 : Nat) * 2 ^ (104 : Int) * (if j = 7 then -1 else 1)) ]

def checkConversions (p : Profile) : IO Unit := do
  let anchor := Lane.value (2 ^ p.maxPowerExponent)
  let listed := chunk anchor (boundaryLanes p) ++ specialBlocks ++ randomBlocks (p.width + 17) 48
  -- An even number of blocks, so the same lanes also run as rows of two blocks.
  let blocks := if listed.size % 2 = 0 then listed else listed.push (Array.replicate 32 (.value 0))
  let narrow := shapeOf p blocks.size
  let wide : Shape := ⟨p, blocks.size / 2, 64, by decide, by decide⟩
  let modes : List OverflowMode :=
    if p = .e4m3 ∨ p = .e5m2 then [.saturate, .native] else [.saturate]
  for mode in modes do
    let label := if mode = .native then "OVF" else "SAT"
    checkConvert s!"{label} at §6.3 scales" p narrow blocks mode none
    checkConvert s!"{label} at §6.3 scales, rows of 64" p wide blocks mode none
    let given := (Array.range blocks.size).map fun b => #[127, 120, 135, 0, 1, 254, 255][b % 7]!
    checkConvert s!"{label} at given scales" p narrow blocks mode (some given)

/-! ## Refusals, empty tensors and ownership -/

def expectRefused {α : Type} (label : String) (action : IO α) : IO Unit := do
  match ← action.toBaseIO with
  | .ok _ => fail s!"{label}: accepted"
  | .error _ => pure ()

def expectAccepted {α : Type} (label : String) (action : IO α) : IO Unit := do
  match ← action.toBaseIO with
  | .ok _ => pure ()
  | .error error => fail s!"{label}: refused: {error}"

def checkRefusals : IO Unit := do
  let up := Runtime.Autograd.LibTorch.MX.upload
  let fp6 := shapeOf .e3m2 1
  let zeros32 : ByteArray := ⟨Array.replicate 32 0⟩
  expectAccepted "FP6 code 0x3F" (up fp6 ⟨(Array.replicate 31 0).push 0x3F⟩ ⟨#[127]⟩)
  expectRefused "FP6 code 0x40" (up fp6 ⟨(Array.replicate 31 0).push 0x40⟩ ⟨#[127]⟩)
  expectRefused "FP6 code 0x80" (up (shapeOf .e2m3 1) ⟨(Array.replicate 31 0).push 0x80⟩ ⟨#[127]⟩)
  expectAccepted "E2M1 16 code bytes" (up (shapeOf .e2m1 1) ⟨Array.replicate 16 0⟩ ⟨#[127]⟩)
  expectRefused "E2M1 15 code bytes" (up (shapeOf .e2m1 1) ⟨Array.replicate 15 0⟩ ⟨#[127]⟩)
  expectRefused "E4M3 31 code bytes" (up (shapeOf .e4m3 1) ⟨Array.replicate 31 0⟩ ⟨#[127]⟩)
  expectRefused "E4M3 two scale bytes for one block" (up (shapeOf .e4m3 1) zeros32 ⟨#[127, 127]⟩)
  expectAccepted "E4M3 one scale byte" (up (shapeOf .e4m3 1) zeros32 ⟨#[127]⟩)
  -- A shape of 2^32 elements is refused before anything is allocated; 2^32 - 32 passes that check.
  let huge : Shape := ⟨.e4m3, 2 ^ 27, 32, by decide, by decide⟩
  expectRefused "2^32 elements" (huge.native)
  let largest : Shape := ⟨.e4m3, 2 ^ 27 - 1, 32, by decide, by decide⟩
  expectAccepted "2^32 - 32 elements" largest.native
  -- Conversion takes a binary32 buffer of the shape's size.
  let ones ← Buffer.ofFloatArrayIO ⟨Array.replicate 32 1.0⟩
  expectAccepted "binary32 input" (Runtime.Autograd.LibTorch.MX.quantize ones (shapeOf .e4m3 1))
  let doubles ← Buffer.ofFloatArrayIO ⟨Array.replicate 32 1.0⟩ .float64
  expectRefused "binary64 input" (Runtime.Autograd.LibTorch.MX.quantize doubles (shapeOf .e4m3 1))
  expectRefused "input of 32 elements for 64"
    (Runtime.Autograd.LibTorch.MX.quantize ones (shapeOf .e4m3 2))
  expectRefused "two given scales for one block"
    (Runtime.Autograd.LibTorch.MX.quantize ones (shapeOf .e4m3 1) .saturate (some ⟨#[127, 127]⟩))
  -- A handle used under another shape is refused at decoding.
  let t ← up (shapeOf .e4m3 1) zeros32 ⟨#[127]⟩
  expectAccepted "decode under its own shape" t.decode
  expectRefused "decode under another profile" ({ t with shape := shapeOf .int8 1 } : Tensor).decode
  let row ← up ⟨.e4m3, 1, 64, by decide, by decide⟩ ⟨Array.replicate 64 0⟩ ⟨#[127, 127]⟩
  expectAccepted "decode one row of 64 as one row of 64" row.decode
  expectRefused "decode one row of 64 as two rows of 32"
    ({ row with shape := shapeOf .e4m3 2 } : Tensor).decode
  -- An empty tensor uploads, decodes and converts.
  let empty : Shape := ⟨.e2m1, 0, 32, by decide, by decide⟩
  let e ← up empty .empty .empty
  let decoded ← Buffer.toFloatArrayIO (← e.decode)
  unless decoded.size = 0 do fail "empty tensor decodes to elements"
  let back ← Runtime.Autograd.LibTorch.MX.quantize (← Buffer.ofFloatArrayIO .empty) empty
  let (codes, scales) ← back.download
  unless codes.size = 0 ∧ scales.size = 0 do fail "empty tensor converts to bytes"

/-- Upload and download every profile's codes; the handles die when this returns. -/
def roundTrips : IO Unit := do
  for p in profiles do
    let count := 2 ^ p.width
    let codes : Array Nat := (Array.range (max count 32)).map (· % count)
    let blocks := codes.size / 32
    let scales : ByteArray := ⟨(Array.range blocks).map fun b => (b * 53 % 256).toUInt8⟩
    let t ← Runtime.Autograd.LibTorch.MX.upload (shapeOf p blocks) (pack p codes) scales
    let (back, backScales) ← t.download
    unless back.data = (pack p codes).data ∧ backScales.data = scales.data do
      fail s!"{repr p}: download differs from upload"

def checkOwnership : IO Unit := do
  let before ← Buffer.memory
  roundTrips
  let after ← Buffer.memory
  unless after.wrapperLiveCount = before.wrapperLiveCount ∧ after.liveBytes = before.liveBytes do
    fail s!"handles outlive their uses: {before.wrapperLiveCount} wrappers and \
      {before.liveBytes} bytes before, {after.wrapperLiveCount} and {after.liveBytes} after"

/-- In a build without LibTorch, uploading reports the missing library. Decoding, downloading and
converting need an MX handle or a buffer, which such a build cannot create. -/
def checkNotLinked : IO Unit := do
  let upload := Runtime.Autograd.LibTorch.MX.upload (shapeOf .e4m3 1) ⟨Array.replicate 32 0⟩
    ⟨#[127]⟩
  match ← upload.toBaseIO with
  | .ok _ => fail "a build without LibTorch uploaded an MX tensor"
  | .error error =>
    unless ((toString error).splitOn "built without LibTorch").length > 1 do
      fail s!"upload without LibTorch failed otherwise: {error}"
  IO.println "MX without LibTorch: upload reports the missing library: ok"

def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: OCP MX ==="
  checkOwnership
  for p in profiles do
    checkDecode p
  IO.println "MX decoding: every code of every profile at seven scales: ok"
  for p in profiles do
    checkConversions p
  IO.println "MX conversion: boundaries, special and random blocks, SAT and OVF, given scales: ok"
  checkRefusals
  IO.println "MX refusals, empty tensors and handle ownership: ok"

end MX
end Cuda
end Tests
