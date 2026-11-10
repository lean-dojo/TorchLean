/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.CUDA.Tree.Arena
public import TorchLean.CUDA.Tree.Word

/-!
# Arenas in device memory

An arena travels to the device as two blocks of bytes. The words block holds the tag column and
then each index column as little-endian `UInt64` words, and the scalar block holds each scalar
column as complete words of the scalar's precision. `Arena.Columns.ofBlocks` reads the blocks
back, and `Arena.Columns.ofBlocks_words_scalars` proves that it returns the columns we packed.
With `Arena.ofColumns_columns` and `Arena.decode_encode`, a value that we encode, pack and read back
decodes to itself (`Arena.decode_ofBlocks_encode`). A value of a type without scalar fields goes
through the blocks of any scalar type by `Arena.map` (`Arena.decode_map_ofBlocks_encode`).
Lean's `Float` and `Float32` hold a single NaN, so arenas of them carry no NaN payloads. Arenas
of `FloatBits` and `Float32Bits` hold the encodings themselves, and the same round trips return
every word, payloads included.

`Arena.upload` and `Arena.Device.download` move the blocks through three `@[extern]` entry points
of `csrc/libtorch/tree.cpp`, `Arena.Handle.ofBlocks`, `Arena.Handle.words` and
`Arena.Handle.scalars`. They are trusted native code, and the tests check that a round trip
through the device returns the same bytes. Builds without LibTorch link stubs that fail with an
`IO` error. The type of a device arena records the `Scalar` instance its scalars were uploaded
with, so a download reads them back with the same encoding. The native handle also keeps the
storage and the full description of the precision, and refuses to be read with any other.
-/

@[expose] public section

namespace TorchLean

namespace Bytes

variable {β : Type}

/-- Append `f x` for each `x` of `xs` to `out`, as little-endian words of `width` bytes. -/
def pushWords (out : ByteArray) (width : Nat) (f : β → Nat) (xs : Array β) : ByteArray :=
  xs.foldl (fun b x => pushWordLE b width (f x)) out

private theorem pushWords_eq (out : ByteArray) (width : Nat) (f : β → Nat) (xs : Array β) :
    pushWords out width f xs = xs.toList.foldl (fun b x => pushWordLE b width (f x)) out :=
  (Array.foldl_toList ..).symm

private theorem size_foldl (width : Nat) (f : β → Nat) :
    ∀ (l : List β) (out : ByteArray),
      (l.foldl (fun b x => pushWordLE b width (f x)) out).size = out.size + width * l.length
  | [], out => by simp
  | x :: l, out => by
    rw [List.foldl_cons, size_foldl width f l, size_pushWordLE, List.length_cons, Nat.mul_succ]
    omega

private theorem getElem!_foldl_of_lt (width : Nat) (f : β → Nat) {i : Nat} :
    ∀ (l : List β) {out : ByteArray}, i < out.size →
      (l.foldl (fun b x => pushWordLE b width (f x)) out)[i]! = out[i]!
  | [], _, _ => rfl
  | x :: l, out, h => by
    rw [List.foldl_cons, getElem!_foldl_of_lt width f l (by rw [size_pushWordLE]; omega),
      getElem!_pushWordLE_of_lt _ _ h]

private theorem wordFromLE_foldl (width : Nat) (f : β → Nat) :
    ∀ (l : List β) (out : ByteArray) (i : Nat) (h : i < l.length),
      wordFromLE (l.foldl (fun b x => pushWordLE b width (f x)) out) (out.size + width * i)
        width = f l[i] % 256 ^ width
  | [], _, _, h => absurd h (Nat.not_lt_zero _)
  | x :: l, out, 0, _ => by
    rw [List.foldl_cons, Nat.mul_zero, Nat.add_zero,
      wordFromLE_congr (b := pushWordLE out width (f x)) (fun j hj =>
        getElem!_foldl_of_lt width f l (by rw [size_pushWordLE]; omega)),
      wordFromLE_pushWordLE]
    rfl
  | x :: l, out, i + 1, h => by
    have := wordFromLE_foldl width f l (pushWordLE out width (f x)) i
      (by simp only [List.length_cons] at h; omega)
    rw [size_pushWordLE] at this
    rw [List.foldl_cons, show out.size + width * (i + 1) = out.size + width + width * i by
      rw [Nat.mul_succ]; omega, this]
    rfl

/-- `pushWords` appends `width` bytes per element. -/
@[simp] theorem size_pushWords (out : ByteArray) (width : Nat) (f : β → Nat) (xs : Array β) :
    (pushWords out width f xs).size = out.size + width * xs.size := by
  rw [pushWords_eq, size_foldl, Array.length_toList]

/-- `pushWords` keeps the bytes already in `out`. -/
theorem getElem!_pushWords_of_lt {out : ByteArray} {i : Nat} (width : Nat) (f : β → Nat)
    (xs : Array β) (h : i < out.size) : (pushWords out width f xs)[i]! = out[i]! := by
  rw [pushWords_eq]
  exact getElem!_foldl_of_lt width f _ h

/-- Word `i` of the words that `pushWords` appends is `f xs[i]`, reduced to `width` bytes. -/
theorem wordFromLE_pushWords (out : ByteArray) (width : Nat) (f : β → Nat) (xs : Array β)
    {i : Nat} (h : i < xs.size) :
    wordFromLE (pushWords out width f xs) (out.size + width * i) width = f xs[i] % 256 ^ width := by
  rw [pushWords_eq, wordFromLE_foldl width f xs.toList out i (by simpa using h),
    Array.getElem_toList]

/-- Appending arrays one after another with `pushWords` keeps the bytes already in `out`. -/
theorem getElem!_foldl_pushWords_of_lt {width : Nat} {f : β → Nat} {i : Nat} :
    ∀ (xss : List (Array β)) {out : ByteArray}, i < out.size →
      (xss.foldl (fun out xs => pushWords out width f xs) out)[i]! = out[i]!
  | [], _, _ => rfl
  | xs :: xss, out, h => by
    rw [List.foldl_cons, getElem!_foldl_pushWords_of_lt xss (by rw [size_pushWords]; omega),
      getElem!_pushWords_of_lt _ _ _ h]

/-- Appending arrays one after another with `pushWords` keeps the words already in `out`. -/
theorem wordFromLE_foldl_pushWords_of_le {width : Nat} {f : β → Nat} (xss : List (Array β))
    {out : ByteArray} {offset count : Nat} (h : offset + count ≤ out.size) :
    wordFromLE (xss.foldl (fun out xs => pushWords out width f xs) out) offset count =
      wordFromLE out offset count :=
  wordFromLE_congr fun _ hi => getElem!_foldl_pushWords_of_lt xss (by omega)

/-- In arrays of `n` elements each, appended one after another with `pushWords`, element `k` of
array `c` is word `c * n + k`. -/
theorem wordFromLE_foldl_pushWords {width n : Nat} {f : β → Nat} :
    ∀ (xss : List (Array β)) (out : ByteArray) (hxss : ∀ xs ∈ xss, xs.size = n) {c k : Nat}
      (hc : c < xss.length) (hk : k < n),
      wordFromLE (xss.foldl (fun out xs => pushWords out width f xs) out)
          (out.size + width * (c * n + k)) width =
        f (xss[c][k]'(by rw [hxss _ (List.getElem_mem hc)]; exact hk)) % 256 ^ width
  | [], _, _, _, _, hc, _ => absurd hc (Nat.not_lt_zero _)
  | xs :: xss, out, hxss, 0, k, _, hk => by
    have hxs : xs.size = n := hxss xs List.mem_cons_self
    have hle : width * k + width ≤ width * n := by
      rw [← Nat.mul_succ]
      exact Nat.mul_le_mul_left _ hk
    rw [List.foldl_cons, Nat.zero_mul, Nat.zero_add,
      wordFromLE_congr (b := pushWords out width f xs) (fun j hj =>
        getElem!_foldl_pushWords_of_lt xss (by rw [size_pushWords, hxs]; omega)),
      wordFromLE_pushWords out width f xs (by omega)]
    rfl
  | xs :: xss, out, hxss, c + 1, k, hc, hk => by
    have hxs : xs.size = n := hxss xs List.mem_cons_self
    have := wordFromLE_foldl_pushWords (width := width) (f := f) xss (pushWords out width f xs)
      (fun ys hys => hxss ys (List.mem_cons_of_mem _ hys)) (c := c)
      (by simp only [List.length_cons] at hc; omega) hk
    rw [size_pushWords, hxs] at this
    rw [List.foldl_cons, show out.size + width * ((c + 1) * n + k) =
      out.size + width * n + width * (c * n + k) by
      simp only [Nat.add_mul, Nat.one_mul, Nat.mul_add]; omega, this]
    rfl

end Bytes

namespace Kernel

/-- The storage of scalars as the device runtime names it: `0` for binary32, `1` for binary64
and `2` for the bytes of a configured binary format. -/
def Precision.tag : Precision → UInt32
  | .native .binary32 => 0
  | .native .binary64 => 1
  | .binary _ => 2

/-- The name of an exceptional-value encoding of configured binary formats. -/
def encodingName : FloatLib.Floats.Formats.BinaryInterchange.FloatFormat.Encoding → String
  | .ieee => "ieee"
  | .finiteMaxNaN => "finiteMaxNaN"
  | .finiteUnsignedZero => "finiteUnsignedZero"
  | .finite => "finite"

/-- A full description of a precision, which the device runtime compares on every read: the
native format, or every field of a configured binary format. -/
def Precision.descriptor : Precision → String
  | .native .binary32 => "binary32"
  | .native .binary64 => "binary64"
  | .binary f =>
    s!"binary exp {f.expWidth} frac {f.fracWidth} bias {f.exponentBias} {encodingName f.encoding}"

namespace Arena

namespace Columns

variable {α : Type}

/-- The words block of columns: the tag column and then each index column, as little-endian
`UInt64` words. -/
def words (c : Columns α) : ByteArray :=
  c.index.foldl (fun out col => Bytes.pushWords out 8 UInt64.toNat col)
    (Bytes.pushWords .empty 8 UInt64.toNat c.tag)

/-- The scalar block of columns: each scalar column as complete words of `width` bytes. -/
def scalars [Numeric.Word α] (width : Nat) (c : Columns α) : ByteArray :=
  c.scalar.foldl
    (fun out col => Bytes.pushWords out width (fun x => (Numeric.Word.toBits x).toNat) col) .empty

/-- Columns of `count` rows read from a words block and a block of `width`-byte scalar words, with
`indexSlots` index columns, `scalarSlots` scalar columns and the level sizes `sizes`. The blocks
should hold `8 * (indexSlots + 1) * count` and `width * scalarSlots * count` bytes, as
`Device.download` checks: a byte past the end of a block reads as zero, after a panic. -/
def ofBlocks [Numeric.Word α] (sizes : Array Nat) (count indexSlots scalarSlots width : Nat)
    (words scalars : ByteArray) : Columns α where
  sizes := sizes
  tag := Array.ofFn fun k : Fin count => (Bytes.wordFromLE words (8 * k) 8).toUInt64
  index := Array.ofFn fun c : Fin indexSlots => Array.ofFn fun k : Fin count =>
    (Bytes.wordFromLE words (8 * ((c + 1) * count + k)) 8).toUInt64
  scalar := Array.ofFn fun c : Fin scalarSlots => Array.ofFn fun k : Fin count =>
    Numeric.Word.ofNatBits (Bytes.wordFromLE scalars (width * (c * count + k)) width)

private theorem toUInt64_toNat_mod (x : UInt64) : (x.toNat % 256 ^ 8).toUInt64 = x := by
  rw [Nat.mod_eq_of_lt (by simpa using x.toNat_lt), Nat.toUInt64_eq, UInt64.ofNat_toNat]

private theorem ofNatBits_toNat_mod [Numeric.Word α] {width : Nat}
    (hwidth : Numeric.Word.width α ≤ 8 * width) (x : α) :
    Numeric.Word.ofNatBits ((Numeric.Word.toBits x).toNat % 256 ^ width) = x := by
  have hlt : (Numeric.Word.toBits x).toNat < 256 ^ width :=
    Nat.lt_of_lt_of_le (BitVec.isLt _) (by
      rw [show (256 : Nat) = 2 ^ 8 by rfl, ← Nat.pow_mul]
      exact Nat.pow_le_pow_right (by decide) hwidth)
  rw [Nat.mod_eq_of_lt hlt, Numeric.Word.ofNatBits_eq, BitVec.ofNat_toNat, BitVec.setWidth_eq,
    Numeric.Word.ofBits_toBits]

/-- Reading back the blocks of columns whose columns all have `count` rows gives the columns,
when the scalar words fit in `width` bytes. -/
theorem ofBlocks_words_scalars [Numeric.Word α] {c : Columns α} {count width : Nat}
    (htag : c.tag.size = count) (hindex : ∀ col ∈ c.index, col.size = count)
    (hscalar : ∀ col ∈ c.scalar, col.size = count) (hwidth : Numeric.Word.width α ≤ 8 * width) :
    ofBlocks c.sizes count c.index.size c.scalar.size width c.words (c.scalars width) = c := by
  obtain ⟨sizes, tag, index, scalar⟩ := c
  simp only at htag hindex hscalar
  have hfirst : (Bytes.pushWords .empty 8 UInt64.toNat tag).size = 8 * count := by
    simp [htag]
  simp only [ofBlocks, words, scalars, Columns.mk.injEq, true_and]
  refine ⟨?_, ?_, ?_⟩
  · refine Array.ext (by simp [htag]) fun k _ hk => ?_
    have hkc : k < count := htag ▸ hk
    have hw : Bytes.wordFromLE (index.foldl (fun out col => Bytes.pushWords out 8 UInt64.toNat col)
        (Bytes.pushWords .empty 8 UInt64.toNat tag)) (8 * k) 8 = tag[k].toNat % 256 ^ 8 := by
      rw [← Array.foldl_toList, Bytes.wordFromLE_foldl_pushWords_of_le _ (by rw [hfirst]; omega)]
      have := Bytes.wordFromLE_pushWords .empty 8 UInt64.toNat tag hk
      rwa [ByteArray.size_empty, Nat.zero_add] at this
    simp only [Array.getElem_ofFn, hw, toUInt64_toNat_mod]
  · refine Array.ext (by simp) fun c _ hc => ?_
    have hcol : index[c].size = count := hindex _ (Array.getElem_mem hc)
    simp only [Array.getElem_ofFn]
    refine Array.ext (by simp [hcol]) fun k _ hk => ?_
    have hw := Bytes.wordFromLE_foldl_pushWords (width := 8) (f := UInt64.toNat) index.toList
      (Bytes.pushWords .empty 8 UInt64.toNat tag)
      (fun xs hxs => hindex xs (Array.mem_toList_iff.1 hxs)) (c := c) (k := k)
      (by simpa using hc) (hcol ▸ hk)
    rw [hfirst, Array.foldl_toList] at hw
    have hoff : 8 * ((c + 1) * count + k) = 8 * count + 8 * (c * count + k) := by
      simp only [Nat.add_mul, Nat.one_mul, Nat.mul_add]; omega
    simp only [Array.getElem_ofFn, hoff, hw, Array.getElem_toList, toUInt64_toNat_mod]
  · refine Array.ext (by simp) fun c _ hc => ?_
    have hcol : scalar[c].size = count := hscalar _ (Array.getElem_mem hc)
    simp only [Array.getElem_ofFn]
    refine Array.ext (by simp [hcol]) fun k _ hk => ?_
    have hw := Bytes.wordFromLE_foldl_pushWords (width := width)
      (f := fun x : α => (Numeric.Word.toBits x).toNat) scalar.toList .empty
      (fun xs hxs => hscalar xs (Array.mem_toList_iff.1 hxs)) (c := c) (k := k)
      (by simpa using hc) (hcol ▸ hk)
    rw [Array.foldl_toList, ByteArray.size_empty, Nat.zero_add] at hw
    simp only [Array.getElem_ofFn, hw, Array.getElem_toList,
      ofNatBits_toNat_mod hwidth]

end Columns

/-- A value that we encode, pack into blocks and read back decodes to itself, if it has fewer
than `2 ^ 64` nodes and the scalar words fit in `width` bytes. These are the reads that
`Device.download` makes of the blocks that `upload` sends. -/
theorem decode_ofBlocks_encode {σ : Sig} {α : Type} {T : Fin σ.sorts → Type} [σ.Inductive α T]
    [Numeric.Word α] [Inhabited α] {s : Fin σ.sorts} (t : T s)
    (ht : Sig.Inductive.count t < UInt64.size) {width : Nat}
    (hwidth : Numeric.Word.width α ≤ 8 * width) :
    let c := (encode t : Arena α).columns σ
    (ofColumns (Columns.ofBlocks c.sizes c.tag.size σ.indexSlots σ.scalarSlots width c.words
      (c.scalars width))).decode T s = some t := by
  intro c
  have hw : (encode t : Arena α).Wide σ := wide_ofValues _
  have hi : c.index.size = σ.indexSlots := Array.size_ofFn
  have hs : c.scalar.size = σ.scalarSlots := Array.size_ofFn
  rw [← hi, ← hs, Columns.ofBlocks_words_scalars rfl hw.size_of_mem_index hw.size_of_mem_scalar
    hwidth, ofColumns_columns hw, decode_encode t ht]

/-- A value of a type without scalar fields that we encode, map to scalars of type `β`, pack
into blocks, read back and map back to `Unit` decodes to itself. These are the reads of a device
round trip of `(encode t).map f`. -/
theorem decode_map_ofBlocks_encode {σ : Sig} {T : Fin σ.sorts → Type} [σ.Inductive Unit T]
    {β : Type} [Numeric.Word β] {s : Fin σ.sorts} (t : T s)
    (ht : Sig.Inductive.count t < UInt64.size) (f : Unit → β) {width : Nat}
    (hwidth : Numeric.Word.width β ≤ 8 * width) :
    let c := ((encode t : Arena Unit).map f).columns σ
    ((ofColumns (Columns.ofBlocks (α := β) c.sizes c.tag.size σ.indexSlots σ.scalarSlots width
      c.words (c.scalars width))).map fun _ => ()).decode T s = some t := by
  intro c
  have hw : ((encode t : Arena Unit).map f).Wide σ := (wide_ofValues _).map f
  have hi : c.index.size = σ.indexSlots := Array.size_ofFn
  have hs : c.scalar.size = σ.scalarSlots := Array.size_ofFn
  rw [← hi, ← hs, Columns.ofBlocks_words_scalars rfl hw.size_of_mem_index hw.size_of_mem_scalar
    hwidth, ofColumns_columns hw, map_map_unit, decode_encode t ht]

/-- The native handle of arena columns in device memory, the `torchlean_tree` of
`csrc/libtorch/tree.cpp`. Builds without LibTorch cannot create one. -/
opaque HandlePointed : NonemptyType.{0}

/-- Lean type of the native handles of device arenas. Only `Arena.Handle.ofBlocks` creates one. -/
def Handle : Type := HandlePointed.type

instance : Nonempty Handle := HandlePointed.property

/-- A native handle holding the blocks `words` and `scalars` of `count` rows with `indexSlots` index
and `scalarSlots` scalar columns, in scalar storage `format` with words of `width` bytes of the
precision `descriptor`. The runtime refuses blocks whose sizes do not match this description. -/
@[never_extract, extern "torchlean_tree_upload"]
opaque Handle.ofBlocks (format : UInt32) (descriptor : @& String)
    (width count indexSlots scalarSlots : UInt64) (words scalars : @& ByteArray) : IO Handle

/-- The words block of a native handle. The runtime refuses to read it with another scalar
storage `format`, precision `descriptor` or word width `width` than the handle was made with. -/
@[never_extract, extern "torchlean_tree_words"]
opaque Handle.words (handle : @& Handle) (format : UInt32) (descriptor : @& String)
    (width : UInt64) : IO ByteArray

/-- The scalar block of a native handle, read with the same checks as `Arena.Handle.words`. -/
@[never_extract, extern "torchlean_tree_scalars"]
opaque Handle.scalars (handle : @& Handle) (format : UInt32) (descriptor : @& String)
    (width : UInt64) : IO ByteArray

/-- An arena whose columns live in device memory, with scalars encoded by the `Scalar α` instance
of its type: the native handle, its level sizes, its number of rows and its numbers of index and
scalar columns. -/
structure Device (α : Type) [Scalar α] where
  private mk ::
  /-- The native handle of the columns. -/
  handle : Handle
  /-- The number of rows of each level. -/
  sizes : Array Nat
  /-- The number of rows of all levels together. -/
  count : Nat
  /-- The number of index columns. -/
  indexSlots : Nat
  /-- The number of scalar columns. -/
  scalarSlots : Nat

variable {α : Type}

/-- The device arena of a native handle whose levels have the sizes `sizes`, with a tag,
`indexSlots` index slots and `scalarSlots` scalar slots per row, as `Expansion.buildHandle` builds
them. -/
@[no_expose] def Device.ofHandle [Scalar α] (handle : Handle) (sizes : Array Nat)
    (indexSlots scalarSlots : Nat) : Device α :=
  ⟨handle, sizes, sizes.foldl (· + ·) 0, indexSlots, scalarSlots⟩

/-- Copy the columns of a wide arena of signature `σ` to the device, with scalars encoded by
`Scalar α`. The device must have arithmetic for that precision, and its words must hold the
scalars' words. Reading the blocks back gives the arena (`Arena.decode_ofBlocks_encode`). -/
@[no_expose] def upload (σ : Sig) [Scalar α] (a : Arena α) : IO (Device α) := do
  unless decide (a.Wide σ) do
    throw <| IO.userError "tree upload: the arena's columns do not all have their full size"
  let precision := Scalar.precision α
  unless precision.supportsGPU do
    throw <| IO.userError "tree upload: the device has no arithmetic for the scalar format"
  let width := precision.bytes
  unless Numeric.Word.width α ≤ 8 * width do
    throw <| IO.userError "tree upload: the scalar words are wider than their storage"
  let c := a.columns σ
  let count := c.tag.size
  let handle ← Handle.ofBlocks precision.tag precision.descriptor width.toUInt64 count.toUInt64
    σ.indexSlots.toUInt64 σ.scalarSlots.toUInt64 c.words (c.scalars width)
  return ⟨handle, c.sizes, count, σ.indexSlots, σ.scalarSlots⟩

/-- Copy a device arena back to the host, decoding its scalars with the instance they were
encoded with. When the device returns the blocks that `upload` sent, the arena of a value decodes
to that value (`Arena.decode_ofBlocks_encode`). -/
@[no_expose] def Device.download [Scalar α] (d : Device α) : IO (Arena α) := do
  let precision := Scalar.precision α
  let width := precision.bytes
  let words ← d.handle.words precision.tag precision.descriptor width.toUInt64
  let scalars ← d.handle.scalars precision.tag precision.descriptor width.toUInt64
  unless words.size = 8 * (d.indexSlots + 1) * d.count ∧
      scalars.size = width * d.scalarSlots * d.count do
    throw <| IO.userError "tree download: the device returned blocks of the wrong size"
  return ofColumns (Columns.ofBlocks d.sizes d.count d.indexSlots d.scalarSlots width words
    scalars)

end Arena

end Kernel

end TorchLean
