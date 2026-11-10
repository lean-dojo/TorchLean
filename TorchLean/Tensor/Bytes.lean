/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# Little-endian bytes

The binary formats of TorchLean store fixed-width words little-endian, least significant byte
first: NPY payloads, model and optimizer checkpoints, and the scalar words that custom CUDA
computations exchange with the device and write as byte literals into generated source. A word of
`count` bytes holds a natural number below `256 ^ count`.

`pushWordLE` and `wordFromLE` encode and decode a word of any width. `uint64LE` and `uint32LE`
read one 8- or 4-byte word without allocating, for payload loops over a byte range the caller has
already checked. `toUInt32?` is the checked conversion from `Nat` to `UInt32`, for the element
counts and dimensions of the CUDA buffer API.

All five are proved correct here. `pushWordLE` appends exactly the `count` low base-256 digits of
the word (`size_pushWordLE`, `getElem!_pushWordLE_of_lt`, `getElem!_pushWordLE_size_add`), and
`wordFromLE` reads the base-256 numeral of its bytes (`wordFromLE_succ`, `wordFromLE_lt`) and
inverts `pushWordLE` (`wordFromLE_pushWordLE`). The fixed-width readers equal `wordFromLE`
(`toNat_uint64LE`, `toNat_uint32LE`), and `toUInt32?` succeeds exactly below `2 ^ 32`
(`toUInt32?_eq_some_iff`, `toUInt32?_eq_none_iff`).
-/

@[expose] public section

namespace TorchLean
namespace Bytes

/-- Append the low `count` bytes of `word` to `out` in little-endian order. -/
def pushWordLE (out : ByteArray) (count word : Nat) : ByteArray := Id.run do
  let mut out := out
  let mut value := word
  for _ in [0:count] do
    out := out.push (UInt8.ofNat (value % 256))
    value := value / 256
  return out

/-- The little-endian word of `count` bytes that starts at byte `offset`. A byte past the end of
`bytes` reads as `0`, so callers check the byte range first. -/
def wordFromLE (bytes : ByteArray) (offset count : Nat) : Nat := Id.run do
  let mut value := 0
  let mut place := 1
  for i in [0:count] do
    value := value + bytes[offset + i]!.toNat * place
    place := place * 256
  return value

/-- The word left in `pushWordLE`'s loop after `n` steps. -/
private theorem pushWordLE_loop_snd (out : ByteArray) (word s n : Nat) :
    (List.foldl (fun b (_ : Nat) => (b.1.push (UInt8.ofNat (b.2 % 256)), b.2 / 256)) (out, word)
      (List.range' s n)).2 = word / 256 ^ n := by
  induction n generalizing out word s with
  | zero => simp
  | succ n ih => rw [List.range'_succ, List.foldl_cons, ih, Nat.div_div_eq_div_mul, Nat.pow_succ']

/-- The place value left in `wordFromLE`'s loop after `n` steps. -/
private theorem wordFromLE_loop_snd (bytes : ByteArray) (offset s n v p : Nat) :
    (List.foldl (fun b a => (b.1 + bytes[offset + a]!.toNat * b.2, b.2 * 256)) (v, p)
      (List.range' s n)).2 = p * 256 ^ n := by
  induction n generalizing v p s with
  | zero => simp
  | succ n ih => rw [List.range'_succ, List.foldl_cons, ih, Nat.pow_succ', Nat.mul_assoc]

/-- Zero bytes leave `out` unchanged. -/
@[simp] theorem pushWordLE_zero (out : ByteArray) (word : Nat) : pushWordLE out 0 word = out := by
  simp [pushWordLE]

/-- The encoding of `count + 1` bytes appends digit `count` of `word` to that of `count` bytes. -/
theorem pushWordLE_succ (out : ByteArray) (count word : Nat) :
    pushWordLE out (count + 1) word =
      (pushWordLE out count word).push (UInt8.ofNat (word / 256 ^ count % 256)) := by
  simp only [pushWordLE, Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size,
    List.forIn_pure_yield_eq_foldl, Id.run_bind, Id.run_pure, Nat.sub_zero, Nat.add_sub_cancel,
    Nat.div_one]
  rw [List.range'_concat, List.foldl_append, List.foldl_cons, List.foldl_nil, pushWordLE_loop_snd]

/-- `pushWordLE` appends exactly `count` bytes. -/
@[simp] theorem size_pushWordLE (out : ByteArray) (count word : Nat) :
    (pushWordLE out count word).size = out.size + count := by
  induction count with
  | zero => simp
  | succ n ih => simp [pushWordLE_succ, ih, Nat.add_assoc]

/-- `pushWordLE` keeps the bytes already in `out`. -/
theorem getElem!_pushWordLE_of_lt {out : ByteArray} {i : Nat} (count word : Nat)
    (h : i < out.size) : (pushWordLE out count word)[i]! = out[i]! := by
  induction count with
  | zero => simp
  | succ n ih =>
    rw [pushWordLE_succ, ByteArray.getElem!_push_lt _ _ _ (by rw [size_pushWordLE]; omega), ih]

/-- Appended byte `k` is digit `k` of `word` in base 256. -/
theorem getElem!_pushWordLE_size_add (out : ByteArray) {count k : Nat} (word : Nat)
    (hk : k < count) :
    (pushWordLE out count word)[out.size + k]! = UInt8.ofNat (word / 256 ^ k % 256) := by
  induction count with
  | zero => omega
  | succ n ih =>
    have hs := size_pushWordLE out n word
    rw [pushWordLE_succ]
    by_cases h : k < n
    · rw [ByteArray.getElem!_push_lt _ _ _ (by omega), ih h]
    · obtain rfl : k = n := by omega
      rw [← hs, ByteArray.getElem!_push_eq]

/-- The word of no bytes is `0`. -/
@[simp] theorem wordFromLE_zero (bytes : ByteArray) (offset : Nat) :
    wordFromLE bytes offset 0 = 0 := by
  simp [wordFromLE]

/-- Byte `offset + count` is the base-256 digit of place `256 ^ count`. -/
theorem wordFromLE_succ (bytes : ByteArray) (offset count : Nat) :
    wordFromLE bytes offset (count + 1) =
      wordFromLE bytes offset count + bytes[offset + count]!.toNat * 256 ^ count := by
  simp only [wordFromLE, Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size,
    List.forIn_pure_yield_eq_foldl, Id.run_bind, Id.run_pure, Nat.sub_zero, Nat.add_sub_cancel,
    Nat.div_one]
  rw [List.range'_concat, List.foldl_append, List.foldl_cons, List.foldl_nil, wordFromLE_loop_snd]
  simp

/-- `wordFromLE` reads only the `count` bytes from `offset`. -/
theorem wordFromLE_congr {a b : ByteArray} {offset count : Nat}
    (h : ∀ i < count, a[offset + i]! = b[offset + i]!) :
    wordFromLE a offset count = wordFromLE b offset count := by
  induction count with
  | zero => simp
  | succ n ih =>
    rw [wordFromLE_succ, wordFromLE_succ, ih (fun i hi => h i (by omega)), h n (by omega)]

/-- A word of `count` bytes is below `256 ^ count`. -/
theorem wordFromLE_lt (bytes : ByteArray) (offset count : Nat) :
    wordFromLE bytes offset count < 256 ^ count := by
  induction count with
  | zero => simp
  | succ n ih =>
    have hb : bytes[offset + n]!.toNat * 256 ^ n ≤ 255 * 256 ^ n :=
      Nat.mul_le_mul_right _ (by have := bytes[offset + n]!.toNat_lt; omega)
    rw [wordFromLE_succ, Nat.pow_succ]
    omega

/-- Decoding the bytes `pushWordLE` appends gives `word` back, reduced to `count` bytes. -/
@[simp] theorem wordFromLE_pushWordLE (out : ByteArray) (count word : Nat) :
    wordFromLE (pushWordLE out count word) out.size count = word % 256 ^ count := by
  induction count with
  | zero => simp [Nat.mod_one]
  | succ n ih =>
    have hs := size_pushWordLE out n word
    rw [pushWordLE_succ, wordFromLE_succ,
      wordFromLE_congr (b := pushWordLE out n word)
        (fun i hi => ByteArray.getElem!_push_lt _ _ _ (by omega)), ih, ← hs,
      ByteArray.getElem!_push_eq, UInt8.toNat_ofNat', Nat.mod_pow_succ (b := 256) (k := n)]
    simp only [Nat.reducePow, Nat.mod_mod]
    rw [Nat.mul_comm]

/-- The little-endian `UInt64` at byte offset `o`, read without allocating. A byte past the end of
`bs` reads as `0`, so callers check the byte range first. -/
@[inline] def uint64LE (bs : ByteArray) (o : Nat) : UInt64 :=
  (bs.get! o).toUInt64
    ||| ((bs.get! (o + 1)).toUInt64 <<< 8)
    ||| ((bs.get! (o + 2)).toUInt64 <<< 16)
    ||| ((bs.get! (o + 3)).toUInt64 <<< 24)
    ||| ((bs.get! (o + 4)).toUInt64 <<< 32)
    ||| ((bs.get! (o + 5)).toUInt64 <<< 40)
    ||| ((bs.get! (o + 6)).toUInt64 <<< 48)
    ||| ((bs.get! (o + 7)).toUInt64 <<< 56)

/-- The little-endian `UInt32` at byte offset `o`, with the same precondition as `uint64LE`. -/
@[inline] def uint32LE (bs : ByteArray) (o : Nat) : UInt32 :=
  (bs.get! o).toUInt32
    ||| ((bs.get! (o + 1)).toUInt32 <<< 8)
    ||| ((bs.get! (o + 2)).toUInt32 <<< 16)
    ||| ((bs.get! (o + 3)).toUInt32 <<< 24)

/-- `ByteArray.get!` is `getElem!`, the form the `ByteArray` lemmas of core state. -/
private theorem get!_eq_getElem! (bs : ByteArray) (i : Nat) : bs.get! i = bs[i]! := by
  cases bs with
  | mk data =>
    by_cases h : i < data.size
    · rw [getElem!_pos (⟨data⟩ : ByteArray) i h]
      simp only [ByteArray.get!, getElem!_pos data i h]
      rfl
    · rw [getElem!_neg (⟨data⟩ : ByteArray) i h]
      simp only [ByteArray.get!, getElem!_neg data i h]

/-- `x ||| y * 2 ^ k` is a sum when the bits of `x` lie below `k`. -/
private theorem or_mul_two_pow {x : Nat} (y k : Nat) (hx : x < 2 ^ k) :
    x ||| y * 2 ^ k = x + y * 2 ^ k := by
  rw [Nat.or_comm, Nat.mul_comm, ← Nat.two_pow_add_eq_or_of_lt hx, Nat.add_comm]

/-- `uint64LE` reads the word `wordFromLE` reads at width 8. -/
theorem toNat_uint64LE (bs : ByteArray) (o : Nat) : (uint64LE bs o).toNat = wordFromLE bs o 8 := by
  have h0 := bs[o]!.toNat_lt
  have h1 := bs[o + 1]!.toNat_lt
  have h2 := bs[o + 2]!.toNat_lt
  have h3 := bs[o + 3]!.toNat_lt
  have h4 := bs[o + 4]!.toNat_lt
  have h5 := bs[o + 5]!.toNat_lt
  have h6 := bs[o + 6]!.toNat_lt
  have h7 := bs[o + 7]!.toNat_lt
  simp only [uint64LE, wordFromLE_succ, wordFromLE_zero, UInt64.toNat_or, UInt64.toNat_shiftLeft,
    UInt8.toNat_toUInt64, get!_eq_getElem!, Nat.shiftLeft_eq, UInt64.reduceToNat, Nat.reduceMod,
    Nat.add_zero]
  generalize bs[o]!.toNat = b0 at *
  generalize bs[o + 1]!.toNat = b1 at *
  generalize bs[o + 2]!.toNat = b2 at *
  generalize bs[o + 3]!.toNat = b3 at *
  generalize bs[o + 4]!.toNat = b4 at *
  generalize bs[o + 5]!.toNat = b5 at *
  generalize bs[o + 6]!.toNat = b6 at *
  generalize bs[o + 7]!.toNat = b7 at *
  rw [Nat.mod_eq_of_lt (by omega : b1 * 2 ^ 8 < 2 ^ 64),
    Nat.mod_eq_of_lt (by omega : b2 * 2 ^ 16 < 2 ^ 64),
    Nat.mod_eq_of_lt (by omega : b3 * 2 ^ 24 < 2 ^ 64),
    Nat.mod_eq_of_lt (by omega : b4 * 2 ^ 32 < 2 ^ 64),
    Nat.mod_eq_of_lt (by omega : b5 * 2 ^ 40 < 2 ^ 64),
    Nat.mod_eq_of_lt (by omega : b6 * 2 ^ 48 < 2 ^ 64),
    Nat.mod_eq_of_lt (by omega : b7 * 2 ^ 56 < 2 ^ 64),
    or_mul_two_pow _ 8 (by omega), or_mul_two_pow _ 16 (by omega), or_mul_two_pow _ 24 (by omega),
    or_mul_two_pow _ 32 (by omega), or_mul_two_pow _ 40 (by omega), or_mul_two_pow _ 48 (by omega),
    or_mul_two_pow _ 56 (by omega)]
  omega

/-- `uint32LE` reads the word `wordFromLE` reads at width 4. -/
theorem toNat_uint32LE (bs : ByteArray) (o : Nat) : (uint32LE bs o).toNat = wordFromLE bs o 4 := by
  have h0 := bs[o]!.toNat_lt
  have h1 := bs[o + 1]!.toNat_lt
  have h2 := bs[o + 2]!.toNat_lt
  have h3 := bs[o + 3]!.toNat_lt
  simp only [uint32LE, wordFromLE_succ, wordFromLE_zero, UInt32.toNat_or, UInt32.toNat_shiftLeft,
    UInt8.toNat_toUInt32, get!_eq_getElem!, Nat.shiftLeft_eq, UInt32.reduceToNat, Nat.reduceMod,
    Nat.add_zero]
  generalize bs[o]!.toNat = b0 at *
  generalize bs[o + 1]!.toNat = b1 at *
  generalize bs[o + 2]!.toNat = b2 at *
  generalize bs[o + 3]!.toNat = b3 at *
  rw [Nat.mod_eq_of_lt (by omega : b1 * 2 ^ 8 < 2 ^ 32),
    Nat.mod_eq_of_lt (by omega : b2 * 2 ^ 16 < 2 ^ 32),
    Nat.mod_eq_of_lt (by omega : b3 * 2 ^ 24 < 2 ^ 32),
    or_mul_two_pow _ 8 (by omega), or_mul_two_pow _ 16 (by omega), or_mul_two_pow _ 24 (by omega)]
  omega

/-- `n` as a `UInt32`, or `none` when `n` does not fit in 32 bits. -/
def toUInt32? (n : Nat) : Option UInt32 :=
  if h : n < UInt32.size then some (UInt32.ofNatLT n h) else none

/-- `toUInt32?` returns the `UInt32` whose value is `n`. -/
theorem toUInt32?_eq_some_iff {n : Nat} {x : UInt32} : toUInt32? n = some x ↔ x.toNat = n := by
  unfold toUInt32?
  split
  · simp only [Option.some.injEq]
    constructor
    · rintro rfl
      exact UInt32.toNat_ofNatLT
    · intro hx
      apply UInt32.toNat_inj.mp
      rw [UInt32.toNat_ofNatLT, hx]
  · simp only [reduceCtorEq, false_iff]
    intro hx
    have := x.toNat_lt
    simp [UInt32.size] at *
    omega

/-- `toUInt32?` fails exactly when `n` needs more than 32 bits. -/
theorem toUInt32?_eq_none_iff {n : Nat} : toUInt32? n = none ↔ UInt32.size ≤ n := by
  unfold toUInt32?
  split <;> simp <;> omega

end Bytes
end TorchLean
