/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Data.IO.Parsing

/-!
# NPY Loader

This module implements the small, explicit `.npy` subset that TorchLean's native training
examples use:

- NumPy format versions 1 and 2;
- little-endian `float32` and `float64` payloads (`<f4`, `<f8`);
- C-order arrays directly, and Fortran-order arrays converted to C-order at load time;
- deterministic conversion to a flat `Array Float` in C order.

The loader stays narrow. It is a runtime bridge for trusted experiment artifacts, not
a general NumPy parser and not part of the formal tensor semantics. Tensor construction happens in
`NN.API.Data.Sources`, which keeps this file-format boundary independent of model training.

Reference:
- NumPy `.npy` format documentation:
  https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
-/

@[expose] public section

namespace TorchLean
namespace Data
namespace IO

open Internal

/--
In-memory representation of a loaded `.npy` file in TorchLean's supported subset.

`values` is always flattened in C-order. If the source file declares `fortran_order = True`, we
reorder the payload during parsing and store `fortran := false` in the returned value so downstream
tensor loaders never have to reason about storage order.
-/
structure NpyData where
  /-- Dtype string as stored in the header, for example `"<f4"` or `"<f8"`. -/
  dtype : String
  /-- Logical array shape as stored in the header. -/
  shape : Array Nat
  /-- Whether the returned flat payload is still Fortran-ordered. This loader returns `false`. -/
  fortran : Bool
  /-- Flattened numeric payload, converted to Lean `Float` values. -/
  values : Array Float

/--
Validated NPY metadata, without reading or allocating the numeric payload.

Dimensions and the element count fit an addressable `Array Float`. Payload availability is checked
separately by full or prefix readers; a valid header alone does not certify a complete file.
-/
structure NpyHeader where
  /-- Supported dtype descriptor, either `"<f4"` or `"<f8"`. -/
  dtype : String
  /-- Logical array shape from the header. -/
  shape : Array Nat
  /-- Whether the on-disk payload is Fortran-ordered. -/
  fortran : Bool
  /-- Byte offset where the numeric payload begins. -/
  dataStart : Nat
  /-- Number of bytes occupied by one on-disk element. -/
  elementBytes : Nat
  /-- Product of the validated dimensions; scalar arrays contain one element. -/
  elementCount : Nat

namespace Internal

/--
Prefix products of a shape array.

For a shape `[d₀, d₁, d₂]`, this returns `[1, d₀, d₀*d₁]`, which are exactly the
Fortran-order strides. We use these strides to convert Fortran storage into TorchLean's ordinary
C-order flattening convention.
-/
def prefixProducts (shape : Array Nat) : Array Nat := Id.run do
  let mut products := Array.mkEmpty shape.size
  let mut acc := 1
  for d in shape do
    products := products.push acc
    acc := acc * d
  return products

/--
Convert a linear C-order index to the corresponding linear Fortran-order index.

Both indices describe the same multi-dimensional coordinate. The difference is only how the
coordinate is flattened into a one-dimensional payload.
-/
def idxFortranOfCIdx (shape : Array Nat) (idxC : Nat) : Nat := Id.run do
  let strides := prefixProducts shape
  let mut idx := idxC
  let mut result := 0
  for offset in [0:shape.size] do
    let axis := shape.size - 1 - offset
    let dim := shape[axis]!
    let stride := strides[axis]!
    result := result + (idx % dim) * stride
    idx := idx / dim
  return result

/-- Reorder a Fortran-ordered flat array into C-order, rejecting an inconsistent payload. -/
def reorderFortranToC
    (tag : String) (shape : Array Nat) (raw : Array Float) : Except String (Array Float) := do
  let count := shape.foldl (fun acc n => acc * n) 1
  unless raw.size = count do
    throw <| formatError tag s!"shape requires {count} elements, got {raw.size}"
  let values ← (Array.range count).mapM fun i =>
    let idxF := idxFortranOfCIdx shape i
    match raw[idxF]? with
    | some value => pure value
    | none => throw <| formatError tag <|
        s!"Fortran-order index {idxF} is outside the {raw.size}-element payload"
  pure values

/-- Safe `ByteArray` indexing. -/
def byteAt? (bs : ByteArray) (i : Nat) : Option UInt8 :=
  if h : i < bs.size then
    some (bs.get i h)
  else
    none

/-- Read a little-endian `UInt16` at byte offset `i`, returning `none` on out-of-bounds input. -/
def readUInt16LE (bs : ByteArray) (i : Nat) : Option Nat :=
  match byteAt? bs i, byteAt? bs (i + 1) with
  | some b0, some b1 =>
      let n := b0.toNat + b1.toNat * 256
      some n
  | _, _ => none

/-- Read a little-endian `UInt32` at byte offset `i`, returning `none` on out-of-bounds input. -/
def readUInt32LE (bs : ByteArray) (i : Nat) : Option UInt32 :=
  match byteAt? bs i, byteAt? bs (i + 1), byteAt? bs (i + 2), byteAt? bs (i + 3) with
  | some b0, some b1, some b2, some b3 =>
      let w0 := b0.toUInt32
      let w1 := b1.toUInt32 <<< (UInt32.ofNat 8)
      let w2 := b2.toUInt32 <<< (UInt32.ofNat 16)
      let w3 := b3.toUInt32 <<< (UInt32.ofNat 24)
      some (w0 + w1 + w2 + w3)
  | _, _, _, _ => none

/-- Read a little-endian `UInt64` at byte offset `i`, returning `none` on out-of-bounds input. -/
def readUInt64LE (bs : ByteArray) (i : Nat) : Option UInt64 :=
  match byteAt? bs i, byteAt? bs (i + 1), byteAt? bs (i + 2), byteAt? bs (i + 3),
        byteAt? bs (i + 4), byteAt? bs (i + 5), byteAt? bs (i + 6), byteAt? bs (i + 7) with
  | some b0, some b1, some b2, some b3, some b4, some b5, some b6, some b7 =>
      let w0 := b0.toUInt64
      let w1 := b1.toUInt64 <<< (UInt64.ofNat 8)
      let w2 := b2.toUInt64 <<< (UInt64.ofNat 16)
      let w3 := b3.toUInt64 <<< (UInt64.ofNat 24)
      let w4 := b4.toUInt64 <<< (UInt64.ofNat 32)
      let w5 := b5.toUInt64 <<< (UInt64.ofNat 40)
      let w6 := b6.toUInt64 <<< (UInt64.ofNat 48)
      let w7 := b7.toUInt64 <<< (UInt64.ofNat 56)
      some (w0 + w1 + w2 + w3 + w4 + w5 + w6 + w7)
  | _, _, _, _, _, _, _, _ => none

/-- Parse a shape tuple like `(3, 4)` or `(3,)` from a NumPy header fragment. -/
def parseShapeValue (s : String) : Option (Array Nat) := do
  match (s.trimAsciiStart).toString.toList with
  | '(' :: rest =>
      let (inside, remaining) := rest.span (fun c => c != ')')
      match remaining with
      | [] => none
      | _ :: tail =>
          if !isHeaderValueEnd tail then return ← none
          let contents := (String.ofList inside).trimAscii.toString
          if contents.isEmpty then return #[]
          let parts := contents.splitOn ","
          if parts.length = 1 then return ← none
          let dims := if contents.endsWith "," then parts.dropLast else parts
          let values ← dims.mapM parseNatValue
          return values.toArray
  | _ => none

/--
Parse the NumPy header dictionary.

We only need three standard fields: `descr`, `fortran_order`, and `shape`. The header format is a
Python-literal dictionary padded to an alignment boundary, so this parser stays field-oriented
rather than trying to become a full Python parser.
-/
def parseHeader (tag : String) (hdr : String) :
  Except String (String × Bool × Array Nat) :=
  let descrOpt := findField hdr "descr" >>= parseQuotedValue
  let fortranOpt := findField hdr "fortran_order" >>= parseBoolValue
  let shapeOpt := findField hdr "shape" >>= parseShapeValue
  match descrOpt, fortranOpt, shapeOpt with
  | some descr, some fortran, some shape => .ok (descr, fortran, shape)
  | _, _, _ => .error (formatError tag "failed to parse NPY header")

/-- Bound header allocation before interpreting an untrusted version-2 header length. -/
def maxNpyHeaderBytes : Nat := 1024 * 1024

/-- Validate the magic and version, returning the header offset and its bounded byte length. -/
def npyHeaderLayout (tag : String) (bs : ByteArray) : Except String (Nat × Nat) := do
  if bs.size < 10 then
    .error (formatError tag "file too small")
  else
    let magicOk :=
      ((byteAt? bs 0).map (fun b => b.toNat == 0x93) |>.getD false)
      && ((byteAt? bs 1).map (fun b => b.toNat == ('N' : Char).toNat) |>.getD false)
      && ((byteAt? bs 2).map (fun b => b.toNat == ('U' : Char).toNat) |>.getD false)
      && ((byteAt? bs 3).map (fun b => b.toNat == ('M' : Char).toNat) |>.getD false)
      && ((byteAt? bs 4).map (fun b => b.toNat == ('P' : Char).toNat) |>.getD false)
      && ((byteAt? bs 5).map (fun b => b.toNat == ('Y' : Char).toNat) |>.getD false)
    if !magicOk then
      .error (formatError tag "invalid NPY magic header")
    else
      let major := (byteAt? bs 6).map (fun b => b.toNat) |>.getD 0
      let minor := (byteAt? bs 7).map (fun b => b.toNat) |>.getD 0
      if !(major = 1 || major = 2) || minor != 0 then
        .error (formatError tag s!"unsupported NPY version: {major}.{minor}")
      else
        let headerLenOpt :=
          if major = 1 then
            readUInt16LE bs 8
          else
            readUInt32LE bs 8 |>.map UInt32.toNat
        match headerLenOpt with
        | none => .error (formatError tag "invalid NPY header length")
        | some headerLen =>
            let headerStart := if major = 1 then 10 else 12
            if headerLen > maxNpyHeaderBytes then
              .error (formatError tag s!"NPY header exceeds {maxNpyHeaderBytes} bytes")
            else
              .ok (headerStart, headerLen)

/-- Byte width for the dtypes supported by TorchLean's NPY loader. -/
def npyElementBytes (tag descr : String) : Except String Nat :=
  if descr = "<f8" then
    .ok 8
  else if descr = "<f4" then
    .ok 4
  else
    .error (formatError tag s!"unsupported dtype: {descr}")

/-- Bound each dimension and the shape product before allocating a decoded `Array Float`. -/
def npyElementCount (tag : String) (shape : Array Nat) : Except String Nat := do
  let limit := (USize.size - 1) / 8
  for dim in shape do
    if dim > limit then
      throw <| formatError tag "NPY dimension exceeds addressable array size"
  if shape.contains 0 then return 0
  let mut count := 1
  for dim in shape do
    if count > limit / dim then
      throw <| formatError tag "NPY shape exceeds addressable array size"
    count := count * dim
  return count

/-- Parse and validate the metadata shared by in-memory and file-backed NPY readers. -/
def parseNpyHeaderMeta (tag : String) (bs : ByteArray) : Except String NpyHeader := do
  let (headerStart, headerLen) ← npyHeaderLayout tag bs
  let headerEnd := headerStart + headerLen
  if headerEnd > bs.size then
    throw <| formatError tag "NPY header out of bounds"
  let headerBytes := bs.extract headerStart headerEnd
  let some headerStr := String.fromUTF8? headerBytes
    | throw <| formatError tag "invalid NPY header encoding"
  let (dtype, fortran, shape) ← parseHeader tag headerStr
  let elementBytes ← npyElementBytes tag dtype
  let elementCount ← npyElementCount tag shape
  return { dtype, fortran, shape, dataStart := headerEnd, elementBytes, elementCount }

/-- Read one supported numeric element from an NPY payload. -/
def readNpyElement (tag descr : String) (bs : ByteArray) (off : Nat) : Except String Float :=
  if descr = "<f8" then
    match readUInt64LE bs off with
    | some w => .ok (Float.ofBits w)
    | none => .error (formatError tag "invalid float64 data")
  else if descr = "<f4" then
    match readUInt32LE bs off with
    | some w => .ok (Float32.toFloat (Float32.ofBits w))
    | none => .error (formatError tag "invalid float32 data")
  else
    .error (formatError tag s!"unsupported dtype: {descr}")

/-- Validate a C-order leading prefix and retain only its requested shape and element count. -/
def npyPrefixHeader (tag : String) (hdr : NpyHeader) (expectedShape : Array Nat) :
    Except String NpyHeader := do
  if hdr.fortran then
    throw <| formatError tag "prefix row loading requires C-order NPY arrays"
  match expectedShape[0]?, hdr.shape[0]? with
  | some expectedN, some actualN =>
      let expectedTail := expectedShape.extract 1 expectedShape.size
      let actualTail := hdr.shape.extract 1 hdr.shape.size
      if actualTail != expectedTail then
        throw <| formatError tag
          s!"shape mismatch: expected trailing dims {expectedTail}, got {actualTail}"
      if actualN < expectedN then
        throw <| formatError tag s!"expected at least {expectedN} rows, got {actualN}"
      let elementCount ← npyElementCount tag expectedShape
      return { hdr with shape := expectedShape, elementCount }
  | none, none =>
      throw <| formatError tag
        "prefix row loading requires a leading axis; scalar arrays have none"
  | _, _ =>
      throw <| formatError tag s!"shape mismatch: expected {expectedShape}, got {hdr.shape}"

/-- Decode a validated full or prefix payload, checking its byte range before reserving values. -/
def decodeNpyPayload (tag : String) (hdr : NpyHeader) (bs : ByteArray) :
    Except String NpyData := do
  if hdr.dataStart + hdr.elementCount * hdr.elementBytes > bs.size then
    throw <| formatError tag "NPY data truncated"
  let mut raw : Array Float := Array.mkEmpty hdr.elementCount
  for i in [0:hdr.elementCount] do
    let value ← readNpyElement tag hdr.dtype bs (hdr.dataStart + i * hdr.elementBytes)
    raw := raw.push value
  let values ← if hdr.fortran then reorderFortranToC tag hdr.shape raw else pure raw
  return { dtype := hdr.dtype, shape := hdr.shape, fortran := false, values }

/--
Read exactly the requested bytes using bounded allocations, rejecting an early end of file.

The buffer grows only with bytes actually read. An untrusted element count never becomes the
capacity of a read buffer before the corresponding file contents have been observed.
-/
partial def readNpyBytes (handle : _root_.IO.FS.Handle) (count : Nat) :
    _root_.IO (Except String ByteArray) := do
  let rec loop (remaining : Nat) (bytes : ByteArray) :
      _root_.IO (Except String ByteArray) := do
    if remaining = 0 then return .ok bytes
    let chunk ← handle.read (min remaining 65536).toUSize
    if chunk.isEmpty then return .error (formatError "npy" "NPY data truncated")
    loop (remaining - chunk.size) (bytes ++ chunk)
  loop count ByteArray.empty

/-- Read only the bounded NPY preamble and header, leaving the handle at the numeric payload. -/
def readNpyHeaderFromHandle (handle : _root_.IO.FS.Handle) :
    _root_.IO (Except String NpyHeader) := ExceptT.run do
  let initial ← ExceptT.mk (readNpyBytes handle 10)
  let preamble ←
    if byteAt? initial 6 == some 2 then
      let extra ← ExceptT.mk (readNpyBytes handle 2)
      pure (initial ++ extra)
    else pure initial
  let (_, headerLen) ← npyHeaderLayout "npy" preamble
  let header ← ExceptT.mk (readNpyBytes handle headerLen)
  return ← parseNpyHeaderMeta "npy" (preamble ++ header)

/-- Read and decode the validated payload at the handle's current position. -/
def readNpyPayload (handle : _root_.IO.FS.Handle) (hdr : NpyHeader) :
    _root_.IO (Except String NpyData) := ExceptT.run do
  let bytes ← ExceptT.mk (readNpyBytes handle (hdr.elementCount * hdr.elementBytes))
  return ← decodeNpyPayload "npy" { hdr with dataStart := 0 } bytes

end Internal

/--
Parse the bytes of a `.npy` file into `NpyData`.

The parser rejects unsupported dtypes, missing or malformed required header fields, and truncated
payloads. Header parsing recognizes the three required fields; it is not a general Python parser.
That makes loader failures explicit at the trust boundary instead of silently producing tensors with
the wrong shape or partial data.
-/
def parseNpy (tag : String) (bs : ByteArray) : Except String NpyData := do
  let hdr ← parseNpyHeaderMeta tag bs
  decodeNpyPayload tag hdr bs

/--
Parse only the requested leading rows of a C-order `.npy` array.

This supports large exported tensors kept on disk while a run uses only the first `n` rows. The rank
and trailing dimensions must match exactly; only the leading axis may be larger than requested.

The implementation shares header and dtype parsing with `parseNpy`, then decodes only the requested
prefix. This avoids building a full `Array Float` when a command asks for a small leading slice of a
real image or sequence dataset. Only the requested range must be present; unused trailing rows are
not checked for truncation.

Why C-order only?  In row-major NPY files, the first `n` rows are physically contiguous, so the
prefix is exactly the first `n * trailingSize` elements.  In Fortran-order files the same logical
prefix is interleaved across the payload, so prefix decoding would be unsound.  Rather than
silently returning bad rows, we reject Fortran-order prefix loading and ask callers to convert the
array to C-order first.
-/
def parseNpyLeadingAxisPrefix
    (tag : String) (expectedShape : Array Nat) (bs : ByteArray) : Except String NpyData := do
  let hdr ← parseNpyHeaderMeta tag bs
  let selected ← npyPrefixHeader tag hdr expectedShape
  decodeNpyPayload tag selected bs

/--
Read validated metadata without reading the payload.

Supports version 1 and 2 headers up to one MiB. Dtype, storage-order syntax, dimensions, and shape
products are validated; payload completeness belongs to `readNpy` or `readNpyLeadingAxisPrefix`.
-/
def readNpyHeader (path : System.FilePath) : IO (Except String NpyHeader) :=
  _root_.IO.FS.withFile path .read readNpyHeaderFromHandle

/-- Read the full declared NPY payload, rejecting truncation and converting Fortran to C order. -/
def readNpy (path : System.FilePath) : IO (Except String NpyData) := do
  _root_.IO.FS.withFile path .read fun handle => ExceptT.run do
    let hdr ← ExceptT.mk (readNpyHeaderFromHandle handle)
    ExceptT.mk (readNpyPayload handle hdr)

/--
Read only the header and requested leading rows of a C-order `.npy` file.

Unrequested rows are neither read nor decoded. The requested range must be complete, and rank and
trailing dimensions must match exactly. Reads use bounded chunks so a malformed claimed size cannot
trigger a correspondingly large allocation before any bytes have been read.
-/
def readNpyLeadingAxisPrefix
    (path : System.FilePath) (expectedShape : Array Nat) : IO (Except String NpyData) := do
  _root_.IO.FS.withFile path .read fun handle => ExceptT.run do
    let hdr ← ExceptT.mk (readNpyHeaderFromHandle handle)
    let selected ← npyPrefixHeader "npy" hdr expectedShape
    ExceptT.mk (readNpyPayload handle selected)

end IO
end Data
end TorchLean
