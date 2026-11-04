/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

LibTorch FFI: tensors in the OCP Microscaling (MX) 1.0 formats on the CUDA device.
-/

module

public import FloatLib.Floats.Formats.OCP.MX.Standard.Runtime
public import NN.Runtime.Autograd.Engine.LibTorch.Trusted
public import Mathlib.Algebra.Order.Field.Power
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Positivity

/-!
# OCP MX tensors

Shapes, byte layout and the exactness fact behind decoding for tensors stored in the six concrete
formats of the OCP *Microscaling Formats (MX) Specification* 1.0 (E5M2, E4M3, E3M2, E2M3, E2M1 and
INT8). FloatLib's `FloatLib.Floats.Formats.OCP.MX.Standard` is their semantics: a block is 32
elements sharing one E8M0 scale.

A tensor here is a row-major matrix of `rows × columns` elements, blocked by 32 consecutive
elements along each row, so `columns` is a positive multiple of 32 (the specification pads a
shorter row, §6.2). Codes are stored one to a byte, except E2M1, two to a byte with element 0 in
the low four bits; an FP6 code occupies a byte's low six bits. Scales take one byte per block.

Decoding to binary32 multiplies each element's binary32 value by the block's power of two. The
theorem `decodeLane_binary32` shows that a finite lane value within binary32's finite range is a
binary32 value: it is `m · 2^e` with a significand of at most 8 bits and `e ≥ -143`, inside
binary32's 24 bits and minimal exponent `-149`. An IEEE multiplication returns its exact product
whenever that product is a binary32 value, so the decoded lane is exact there; the CUDA check
confirms it bit for bit. Beyond binary32's finite range the product overflows to `±∞`; FloatLib
keeps the exact rational there.

The CUDA operations (`upload`, `download`, `decode`, `quantize`) take each profile's values from
FloatLib, computed here once: each code's value for decoding, the sorted finite values with their
codes for rounding, the FP8 overflow bound (`overflowBound`) and words, and the largest
power-of-two exponent. The native side implements the remaining rules of FloatLib itself: the E8M0
scale's value, the scale selection of §6.3, the NaN block and the order of ties; the CUDA check
(`NN/Tests/Runtime/Cuda/MX.lean`) is what ties those to FloatLib.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace LibTorch
namespace MX

open FloatLib.Floats.Formats.OCP.MX.Standard
open FloatLib.Numerics (SignedRat)

/-- A row-major `rows × columns` matrix of MX elements of one profile, blocked by 32 along each
row. A shape whose column count is zero or not a multiple of 32 cannot be built. -/
structure Shape where
  /-- The element encoding of every block. -/
  profile : Profile
  /-- The number of rows; zero gives an empty tensor. -/
  rows : Nat
  /-- The number of elements per row. -/
  columns : Nat
  columns_pos : 0 < columns
  columns_dvd : 32 ∣ columns

namespace Shape

/-- The number of elements. -/
def elements (s : Shape) : Nat := s.rows * s.columns

/-- The number of blocks, one scale byte each. -/
def blocks (s : Shape) : Nat := s.elements / 32

/-- The bytes holding the element codes: two E2M1 codes to a byte, one code of any other
profile. -/
def codeBytes (s : Shape) : Nat :=
  match s.profile with
  | .e2m1 => s.elements / 2
  | _ => s.elements

theorem dvd_elements (s : Shape) : 32 ∣ s.elements :=
  Nat.dvd_trans s.columns_dvd (Nat.dvd_mul_left _ _)

/-- The scales cover the elements exactly: no element is left without a block. -/
theorem blocks_mul (s : Shape) : s.blocks * 32 = s.elements :=
  Nat.div_mul_cancel s.dvd_elements

/-- The E2M1 codes fill their bytes exactly: no half byte is left over. -/
theorem codeBytes_mul_of_e2m1 (s : Shape) (h : s.profile = .e2m1) :
    s.codeBytes * 2 = s.elements := by
  have : 2 ∣ s.elements := Nat.dvd_trans (by decide) s.dvd_elements
  simp [codeBytes, h, Nat.div_mul_cancel this]

/-- Any other profile takes one byte per code. -/
theorem codeBytes_of_ne_e2m1 (s : Shape) (h : s.profile ≠ .e2m1) :
    s.codeBytes = s.elements := by
  unfold codeBytes
  split <;> simp_all

end Shape

/-- binary32's largest finite value, `(2 - 2^-23) · 2^127`. -/
def binary32Max : Rat := (2 ^ 24 - 1) * 2 ^ (104 : Int)

/-- The finite binary32 values: a significand of at most 24 bits, an exponent of at least the
smallest subnormal's, `-149`, and a magnitude at most the largest finite value. -/
def IsBinary32 (x : Rat) : Prop :=
  (∃ m e : Int, x = m * 2 ^ e ∧ m.natAbs < 2 ^ 24 ∧ -149 ≤ e) ∧ |x| ≤ binary32Max

/-- The largest `e` in `[-16, 16]` with `q / 2^e` an integer, paired with that integer. -/
def elementWitness (q : Rat) : Int × Int :=
  match ((List.range 33).map (fun i : Nat => (16 : Int) - (i : Int))).find?
      (fun e => (q / 2 ^ e).den == 1) with
  | some e => ((q / 2 ^ e).num, e)
  | none => (0, 0)

/-- Whether a decoded element, if finite, is `m · 2^e` with `|m| < 2^8` and `e ≥ -16`, by the
witness `elementWitness`. -/
def elementFits (q? : Option Rat) : Bool :=
  match q? with
  | none => true
  | some q =>
    q == ((elementWitness q).1 : Rat) * 2 ^ (elementWitness q).2 &&
      decide ((elementWitness q).1.natAbs < 2 ^ 8) && decide (-16 ≤ (elementWitness q).2)

theorem elementFits_all (p : Profile) (w : Element p) : elementFits w.toRat? = true := by
  have key : ∀ n : Fin (2 ^ p.width),
      elementFits (Element.toRat? (profile := p) (BitVec.ofNat p.width n)) = true := by
    cases p <;> decide +kernel
  simpa [BitVec.ofNat_toNat] using key ⟨w.toNat, w.isLt⟩

/-- Every finite element of every profile is `m · 2^e` with `|m| < 2^8` and `e ≥ -16`. -/
theorem element_dyadic {p : Profile} (w : Element p) {q : Rat} (h : w.toRat? = some q) :
    ∃ m e : Int, q = m * 2 ^ e ∧ m.natAbs < 2 ^ 8 ∧ -16 ≤ e := by
  have fits := elementFits_all p w
  simp only [h, elementFits, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at fits
  exact ⟨_, _, fits.1.1, fits.1.2, fits.2⟩

/-- Decoding is exact in binary32: a finite lane value of any block, whose magnitude lies in
binary32's finite range, is a binary32 value. -/
theorem decodeLane_binary32 {p : Profile} (b : Block p) (lane : Fin 32) {v : SignedRat}
    (h : (b.decodeLane lane).finite? = some v) (range : |v.value| ≤ binary32Max) :
    IsBinary32 v.value := by
  refine ⟨?_, range⟩
  unfold Block.decodeLane at h
  cases hs : b.scale.exponent? with
  | none => simp [hs, FloatLib.Numerics.NumericalValue.finite?] at h
  | some s =>
    have s_ge : -127 ≤ s := by
      unfold FloatLib.Floats.Formats.OCP.MX.E8M0.exponent? at hs
      split at hs <;> simp_all [FloatLib.Floats.Formats.OCP.MX.E8M0.bias]
      omega
    cases hd : Element.decode b.values[lane.val] with
    | finite u =>
      simp [hs, hd, FloatLib.Numerics.NumericalValue.map,
        FloatLib.Numerics.NumericalValue.finite?] at h
      subst h
      have hq : b.values[lane.val].toRat? = some u.value := by
        simp [Element.toRat?, hd, FloatLib.Numerics.NumericalValue.finite?]
      obtain ⟨m, e, hm, hbits, he⟩ := element_dyadic _ hq
      refine ⟨m, e + s, ?_, by omega, by omega⟩
      simp only [scaleFinite, SignedRat.value_mul, SignedRat.value_ofRat, hm]
      rw [zpow_add₀ (by norm_num : (2 : Rat) ≠ 0)]
      ring
    | infinity sign => simp [hs, hd, FloatLib.Numerics.NumericalValue.map,
        FloatLib.Numerics.NumericalValue.finite?] at h
    | exceptional e => simp [hs, hd, FloatLib.Numerics.NumericalValue.map,
        FloatLib.Numerics.NumericalValue.finite?] at h

/-- One bit finer fails: `2^-150`, half binary32's smallest subnormal, is no binary32 value, so
the bound `-149` in `IsBinary32` is the one binary32 has and the theorem above is no vacuous
statement about every rational. -/
theorem not_isBinary32_half_smallest : ¬ IsBinary32 (2 ^ (-150 : Int)) := by
  rintro ⟨⟨m, e, h, -, he⟩, -⟩
  have hm : (m : Rat) = 2 ^ (-150 - e) := by
    rw [zpow_sub₀ (by norm_num : (2 : Rat) ≠ 0), h, mul_div_assoc,
      div_self (zpow_ne_zero _ (by norm_num : (2 : Rat) ≠ 0)), mul_one]
  have pos : (0 : Rat) < m := by rw [hm]; positivity
  have lt : (m : Rat) < 1 := by
    rw [hm]
    exact zpow_lt_one_of_neg₀ (by norm_num) (by omega)
  have : (0 : Int) < m := by exact_mod_cast pos
  have : m < (1 : Int) := by exact_mod_cast lt
  omega

/-! ## Tables for the native side -/

/-- A finite element value as binary64, exact by `element_dyadic`. -/
def elementFloat (value : SignedRat) : Float :=
  if value.value = 0 then (if value.negative then -0.0 else 0.0)
  else Float.scaleB (Float.ofInt (elementWitness value.value).1) (elementWitness value.value).2

/-- The value of each code of a profile, by code: `±∞` and NaN for the non-finite words. -/
def decodeTable (p : Profile) : FloatArray :=
  ⟨(Array.range (2 ^ p.width)).map fun n =>
    match Element.decode (profile := p) (BitVec.ofNat p.width n) with
    | .finite value => elementFloat value
    | .infinity negative => if negative then -(1.0 / 0.0) else 1.0 / 0.0
    | .exceptional _ => 0.0 / 0.0⟩

/-- The rounding candidates of a profile: its distinct finite values in increasing order, and for
each the code chosen for an input with a clear sign bit and for one with a set sign bit. Only zero
has two codes, in the floating profiles; the other values have one, used for both. -/
structure RoundingTable where
  values : FloatArray
  positive : ByteArray
  negative : ByteArray

def roundingTable (p : Profile) : RoundingTable :=
  let candidates := Element.candidates p
  let values := ((candidates.map (·.2)).eraseDups).mergeSort (· ≤ ·)
  let codes (value : Rat) := (candidates.filter (·.2 = value)).map (·.1)
  let pick (value : Rat) (negative : Bool) : UInt8 :=
    let found := codes value
    ((found.find? (Element.negative · = negative)).getD (found.headD 0)).toNat.toUInt8
  { values := ⟨(values.map fun value => elementFloat (SignedRat.ofRat value)).toArray⟩
    positive := ⟨(values.map (pick · false)).toArray⟩
    negative := ⟨(values.map (pick · true)).toArray⟩ }

/-- The FP8 OVF bound: an input whose magnitude passes it rounds out of range, strictly or
inclusively. Other profiles have none. -/
def overflowBound : Profile → Option (Rat × Bool)
  | .e4m3 => some (464, false)
  | .e5m2 => some (61440, true)
  | _ => none

/-- `overflowBound` is FloatLib's `roundsOut`. -/
theorem roundsOut_eq_overflowBound (p : Profile) (x : Rat) :
    Element.roundsOut p x = match overflowBound p with
      | some (bound, true) => decide (bound ≤ |x|)
      | some (bound, false) => decide (bound < |x|)
      | none => false := by
  cases p <;> rfl

/-- The profile's index in the native ABI. -/
def profileIndex : Profile → UInt8
  | .e5m2 => 0 | .e4m3 => 1 | .e3m2 => 2 | .e2m3 => 3 | .e2m1 => 4 | .int8 => 5

/-- Everything the native side needs of a profile, computed once per profile. -/
structure Tables where
  decode : FloatArray
  rounding : RoundingTable
  /-- The OVF bound, 0 when the profile has none, and whether it is inclusive. -/
  bound : Float
  inclusive : Bool
  /-- `Element.overflow` for a clear and a set sign bit. -/
  overflowPositive : UInt8
  overflowNegative : UInt8

def computeTables (p : Profile) : Tables :=
  let (bound, inclusive) := match overflowBound p with
    | some (bound, inclusive) => (elementFloat (SignedRat.ofRat bound), inclusive)
    | none => (0.0, false)
  { decode := decodeTable p, rounding := roundingTable p, bound, inclusive
    overflowPositive := (Element.overflow p false).toNat.toUInt8
    overflowNegative := (Element.overflow p true).toNat.toUInt8 }

/-- The six profiles' tables, built once at initialization. -/
def allTables : Array Tables :=
  #[.e5m2, .e4m3, .e3m2, .e2m3, .e2m1, .int8].map computeTables

def tables (p : Profile) : Tables :=
  allTables[(profileIndex p).toNat]?.getD (computeTables p)

/-! ## The CUDA tensor -/

/-- Opaque handle to an MX tensor's codes and scales in CUDA memory, with the profile and counts
it was created with (`csrc/libtorch/torchlean.cpp`). Builds without `-K cuda=true` cannot create
one. -/
opaque HandleImpl : NonemptyType.{0}

def Handle : Type := HandleImpl.val

instance : Nonempty Handle := HandleImpl.property

/-- An MX tensor in CUDA memory: its shape and its storage. -/
structure Tensor where
  shape : Shape
  handle : Handle

@[never_extract, extern "torchlean_mx_upload_io"]
private opaque uploadRaw (profile : UInt8) (rows columns : UInt32) (codes scales : @& ByteArray) :
    IO Handle

@[never_extract, extern "torchlean_mx_download_io"]
private opaque downloadRaw (handle : @& Handle) : IO (ByteArray × ByteArray)

@[never_extract, extern "torchlean_mx_decode_io"]
private opaque decodeRaw (handle : @& Handle) (profile : UInt8) (rows columns : UInt32)
    (table : @& FloatArray) : IO Buffer

@[never_extract, extern "torchlean_mx_quantize_io"]
private opaque quantizeRaw (input : @& Buffer) (profile : UInt8) (rows columns : UInt32)
    (maxPower : UInt8) (values : @& FloatArray) (positive negative : @& ByteArray) (bound : Float)
    (inclusive : UInt8)
    (overflowPositive overflowNegative : UInt8) (native : UInt8) (callerScales : UInt8)
    (scales : @& ByteArray) : IO Handle

/-- The counts as the native ABI takes them, or an error naming the limit. -/
def Shape.native (s : Shape) : IO (UInt32 × UInt32) := do
  unless s.elements < 2 ^ 32 do
    throw <| IO.userError s!"MX: {s.elements} elements exceed a buffer's UInt32 size"
  pure (s.rows.toUInt32, s.columns.toUInt32)

/-- Upload codes and scales laid out as the module header states. The byte lengths must be
`Shape.codeBytes` and `Shape.blocks`, and an FP6 code must leave its byte's top two bits clear;
anything else is an `IO` error. -/
@[no_expose] def upload (s : Shape) (codes scales : ByteArray) : IO Tensor := do
  let (rows, columns) ← s.native
  unless codes.size = s.codeBytes do
    throw <| IO.userError s!"MX: {codes.size} code bytes, the shape needs {s.codeBytes}"
  unless scales.size = s.blocks do
    throw <| IO.userError s!"MX: {scales.size} scale bytes, the shape needs {s.blocks}"
  pure ⟨s, ← uploadRaw (profileIndex s.profile) rows columns codes scales⟩

/-- The codes and scales, in the layout `upload` takes. -/
@[no_expose] def Tensor.download (t : Tensor) : IO (ByteArray × ByteArray) :=
  downloadRaw t.handle

/-- Decode to a binary32 buffer of `rows · columns` elements in row-major order, lane by lane as
FloatLib's `Block.decodeLane`, exact within binary32's finite range (`decodeLane_binary32`). It
widens each code to a 64-bit index on the device, so its peak device memory is several times the
size of the codes. -/
@[no_expose] def Tensor.decode (t : Tensor) : IO Buffer := do
  let (rows, columns) ← t.shape.native
  decodeRaw t.handle (profileIndex t.shape.profile) rows columns (tables t.shape.profile).decode

/-- Convert a binary32 buffer of `rows · columns` elements, block by block as FloatLib's
`quantize`: at the scales §6.3 selects, or at `scales` (one E8M0 byte per block) as
`quantizeAt`, with `overflow` choosing SAT or OVF for FP8. A block with a non-finite lane, or a
given scale of 255 (E8M0's NaN), becomes the NaN block. The conversion works in binary64 on the
device and holds several temporaries the size of the input at once, binary64 values and 64-bit
indices, so its peak device memory is several times the binary32 input's. -/
@[no_expose] def quantize (input : Buffer) (s : Shape)
    (overflow : FloatLib.Numerics.OverflowMode := .saturate)
    (scales : Option ByteArray := none) : IO Tensor := do
  let (rows, columns) ← s.native
  if let some given := scales then
    unless given.size = s.blocks do
      throw <| IO.userError s!"MX: {given.size} scale bytes, the shape needs {s.blocks}"
  let t := tables s.profile
  let handle ← quantizeRaw input (profileIndex s.profile) rows columns
    s.profile.maxPowerExponent.toNat.toUInt8 t.rounding.values
    t.rounding.positive t.rounding.negative t.bound (if t.inclusive then 1 else 0)
    t.overflowPositive t.overflowNegative (if overflow = .native then 1 else 0)
    (if scales.isSome then 1 else 0) (scales.getD ByteArray.empty)
  pure ⟨s, handle⟩

end MX
end LibTorch
end Autograd
end Runtime
