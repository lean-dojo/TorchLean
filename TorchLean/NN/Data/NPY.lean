/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Batteries.Data.Array.Basic
public import TorchLean.NN.Data.Parsing
public import TorchLean.Tensor.Bytes
import TorchLean.NN.Data.Monad

/-!
# NPY Loader

The `.npy` subset that TorchLean's native training examples use:

- NumPy format versions 1.0 and 2.0,
- little-endian `float32` and `float64` payloads (`<f4` and `<f8`),
- C-order arrays, and Fortran-order arrays converted to C order when loaded whole,
- decoding to one flat `FloatArray` in C order (`NPY.read`, `NPY.parse`), to the requested leading
  rows (`readPrefix`, `parsePrefix`) or to one `FloatArray` per index along the leading axis
  (`readBlocks`, `parseBlocks`).

The loader is a runtime bridge for trusted experiment artifacts, narrow on purpose and outside the
formal tensor semantics. Tensor construction happens in `TorchLean.NN.Data.Source`, which keeps
this file-format boundary independent of model training.

Reference:
- NumPy `.npy` format documentation:
  https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
-/

@[expose] public section

namespace TorchLean
namespace Data
namespace NPY

/--
In-memory representation of a loaded `.npy` file in TorchLean's supported subset.

`values` is always flattened in C-order: if the source file declares `fortran_order = True`, the
payload is reordered during parsing, so downstream tensor loaders never reason about storage order.
-/
structure Contents where
  /-- Dtype string as stored in the header, for example `"<f4"` or `"<f8"`. -/
  dtype : String
  /-- Logical array shape as stored in the header. -/
  shape : Array Nat
  /-- Flattened numeric payload, converted to Lean `Float` values.

  The payload is bulk numeric data, so it is a `FloatArray`, which stores the doubles unboxed: a
  large file costs one machine word per element, where an `Array Float` would add a heap cell to
  each, and `Tensor.from` wraps the array without copying. -/
  values : FloatArray

/--
A `.npy` payload decoded as one `FloatArray` per index along the leading axis.

`blocks.size` is the leading dimension and every block holds the product of the trailing
dimensions, so `shape = #[d₀] ++ tail` gives `blocks.size = d₀` and
`blocks[i]!.size = tail.foldl (· * ·) 1`.
-/
structure Blocks where
  /-- Dtype string as stored in the header, for example `"<f4"` or `"<f8"`. -/
  dtype : String
  /-- Logical array shape as stored in the header. -/
  shape : Array Nat
  /-- One decoded block per index along the leading axis, in order. -/
  blocks : Array FloatArray

/--
Validated NPY metadata, without reading or allocating the numeric payload.

Dimensions and the element count fit an addressable payload. The full, prefix and block readers
check separately that the payload is present, since a truncated file can carry a valid header.
-/
structure Header where
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
  /-- Product of the validated dimensions, which is 1 for a scalar array. -/
  elementCount : Nat

/-- How a leading-axis block decode is spread across tasks. -/
inductive Parallelism where
  /-- Decode every block on the calling thread. -/
  | sequential
  /-- Spread the blocks across at most `n` tasks. -/
  | tasks (n : Nat)
  /-- Spread the blocks across the platform's hardware concurrency. -/
  | auto
  deriving Repr, BEq

/--
Convert a linear C-order index to the corresponding linear Fortran-order index.

Both indices describe the same multi-dimensional coordinate. The difference is only how the
coordinate is flattened into a one-dimensional payload.
-/
def idxFortranOfCIdx (shape : Array Nat) (idxC : Nat) : Nat := Id.run do
  -- The Fortran-order strides of `[d₀, d₁, d₂]` are `[1, d₀, d₀ * d₁]`.
  let strides : Array Nat := (shape.scanl (· * ·) 1).pop
  let mut idx := idxC
  let mut result := 0
  for offset in [0:shape.size] do
    let axis := shape.size - 1 - offset
    let dim := shape[axis]!
    let stride := strides[axis]!
    result := result + (idx % dim) * stride
    idx := idx / dim
  return result

/--
Reorder a Fortran-ordered flat payload into C order, rejecting an inconsistent payload.

The output is an unboxed `FloatArray`. The payload size must equal the shape's element count, and
an index past its end is an error, so a malformed file is reported and never padded with zeros.
-/
def reorderFortranToC
    (tag : String) (shape : Array Nat) (raw : FloatArray) : Except String FloatArray := do
  let count := shape.foldl (fun acc n => acc * n) 1
  unless raw.size = count do
    throw <| formatError tag s!"shape requires {count} elements, got {raw.size}"
  let mut out := FloatArray.emptyWithCapacity count
  for i in [0:count] do
    let idxF := idxFortranOfCIdx shape i
    if idxF < raw.size then
      out := out.push (raw[idxF]!)
    else
      throw <| formatError tag <|
        s!"Fortran-order index {idxF} is outside the {raw.size}-element payload"
  pure out

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
Parse the NumPy header dictionary into its `descr`, `fortran_order` and `shape` values.

The header is a Python-literal dictionary padded to an alignment boundary. Each of the three fields
is found by its key and parsed on its own, and the rest of the dictionary is ignored. A missing or
malformed field fails the whole header.
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
def maxHeaderBytes : Nat := 1024 * 1024

/-- Validate the magic and version, returning the header offset and its bounded byte length. -/
def headerLayout (tag : String) (bs : ByteArray) : Except String (Nat × Nat) := do
  if bs.size < 10 then
    .error (formatError tag "file too small")
  else
    let magicOk :=
      (bs[0]?.map (fun b => b.toNat == 0x93) |>.getD false)
      && (bs[1]?.map (fun b => b.toNat == ('N' : Char).toNat) |>.getD false)
      && (bs[2]?.map (fun b => b.toNat == ('U' : Char).toNat) |>.getD false)
      && (bs[3]?.map (fun b => b.toNat == ('M' : Char).toNat) |>.getD false)
      && (bs[4]?.map (fun b => b.toNat == ('P' : Char).toNat) |>.getD false)
      && (bs[5]?.map (fun b => b.toNat == ('Y' : Char).toNat) |>.getD false)
    if !magicOk then
      .error (formatError tag "invalid NPY magic header")
    else
      let major : Nat := bs[6]?.map (fun b => b.toNat) |>.getD 0
      let minor : Nat := bs[7]?.map (fun b => b.toNat) |>.getD 0
      if !(major = 1 || major = 2) || minor != 0 then
        .error (formatError tag s!"unsupported NPY version: {major}.{minor}")
      else
        let lengthBytes := if major = 1 then 2 else 4
        if bs.size < 8 + lengthBytes then
          .error (formatError tag "invalid NPY header length")
        else
          let headerLen := Bytes.wordFromLE bs 8 lengthBytes
          if headerLen > maxHeaderBytes then
            .error (formatError tag s!"NPY header exceeds {maxHeaderBytes} bytes")
          else
            .ok (8 + lengthBytes, headerLen)

/-- Byte width for the dtypes supported by TorchLean's NPY loader. -/
def elementBytes (tag descr : String) : Except String Nat :=
  if descr = "<f8" then
    .ok 8
  else if descr = "<f4" then
    .ok 4
  else
    .error (formatError tag s!"unsupported dtype: {descr}")

/-- Bound each dimension and the shape product before allocating a decoded payload. -/
def elementCount (tag : String) (shape : Array Nat) : Except String Nat := do
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
def parseHeaderMeta (tag : String) (bs : ByteArray) : Except String Header := do
  let (headerStart, headerLen) ← headerLayout tag bs
  let headerEnd := headerStart + headerLen
  if headerEnd > bs.size then
    throw <| formatError tag "NPY header out of bounds"
  let headerBytes := bs.extract headerStart headerEnd
  let some headerStr := String.fromUTF8? headerBytes
    | throw <| formatError tag "invalid NPY header encoding"
  let (dtype, fortran, shape) ← parseHeader tag headerStr
  let elementBytes ← NPY.elementBytes tag dtype
  let elementCount ← NPY.elementCount tag shape
  return { dtype, fortran, shape, dataStart := headerEnd, elementBytes, elementCount }

/-!
### Payload decoding

Checked indexing `bs[i]?` is the right tool at the file boundary, where a malformed header has to
be reported rather than guessed at. Inside a payload loop it is the wrong one: an `Option` per
byte and an `Except` per element each cost a heap cell. Decoding an array that way pays about ten
allocations per element and runs an order of magnitude below the rate at which the same loop can
fill a `FloatArray`.

The functions below decode a payload whose byte range and dtype the caller has already checked.
The dtype is matched once per file rather than once per element, `Bytes.uint64LE` and
`Bytes.uint32LE` read each word without an intermediate `Option`, and the result is unboxed.

The decoders reinterpret stored bits, with the one exception the runtime imposes: `Float.ofBits`
and `Float32.ofBits` return the canonical quiet NaN for every NaN pattern, so a NaN decodes as a
NaN but without its sign or payload. Signed zeros, subnormals, infinities and every finite value
decode to exactly their stored bits.
-/

/-- Decode the half-open element range `[lo, hi)` of a `<f8` payload. -/
def decodeRangeF8 (bs : ByteArray) (dataStart lo hi : Nat) : FloatArray := Id.run do
  let mut out := FloatArray.emptyWithCapacity (hi - lo)
  for i in [lo:hi] do
    out := out.push (Float.ofBits (Bytes.uint64LE bs (dataStart + i * 8)))
  pure out

/-- Decode the half-open element range `[lo, hi)` of a `<f4` payload, widening to `Float`. -/
def decodeRangeF4 (bs : ByteArray) (dataStart lo hi : Nat) : FloatArray := Id.run do
  let mut out := FloatArray.emptyWithCapacity (hi - lo)
  for i in [lo:hi] do
    out := out.push (Float32.toFloat (Float32.ofBits (Bytes.uint32LE bs (dataStart + i * 4))))
  pure out

/-- The range decoder for a supported dtype, selected once per file.

Returning the decoder keeps the dtype test out of the element loop and lets one selection serve
many ranges. -/
def rangeDecoder (tag dtype : String) (bs : ByteArray) (dataStart : Nat) :
    Except String (Nat → Nat → FloatArray) :=
  if dtype = "<f8" then
    .ok (decodeRangeF8 bs dataStart)
  else if dtype = "<f4" then
    .ok (decodeRangeF4 bs dataStart)
  else
    .error (formatError tag s!"unsupported dtype: {dtype}")

/-- Element count below which spreading a decode across tasks does not pay.

A task costs a scheduling round trip that only a reasonably large slice earns back, so this
bounds the task count from above by `totalElements / minElementsPerTask`. -/
def minElementsPerTask : Nat := 1 <<< 16

/-- Resolve a `Parallelism` request against a payload into an actual task count, at least 1.

Three bounds apply, and the smallest wins:

- The caller's request, or the platform's hardware concurrency for `auto`.
- The number of blocks. A block is the smallest piece this can hand back whole, so raising
  the task count past `nBlocks` would mean cutting a block in half and copying the halves
  back together afterwards, and that copy costs more than the extra task saves. The
  leading dimension is therefore a hard ceiling on this decode's parallelism.
- `totalElements / minElementsPerTask`, so small payloads stay on one thread.
-/
def resolveTaskCount (p : Parallelism) (nBlocks totalElements : Nat) : Nat :=
  let requested :=
    match p with
    | .sequential => 1
    | .tasks n => n
    | .auto => (System.Platform.Internal.getHardwareConcurrency ()).toNat
  max 1 (min (min (max 1 requested) nBlocks) (totalElements / minElementsPerTask))

/-- Validate a C-order leading prefix and retain only its requested shape and element count. -/
def prefixHeader (tag : String) (hdr : Header) (expectedShape : Array Nat) :
    Except String Header := do
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
      let elementCount ← NPY.elementCount tag expectedShape
      return { hdr with shape := expectedShape, elementCount }
  | none, none =>
      throw <| formatError tag
        "prefix row loading requires a leading axis; scalar arrays have none"
  | _, _ =>
      throw <| formatError tag s!"shape mismatch: expected {expectedShape}, got {hdr.shape}"

/--
Validate that a header describes C-order leading-axis blocks, returning the block count and the
element count per block.

Fortran-order and scalar arrays are rejected here, before any payload is read: a Fortran-order
file interleaves every leading-axis block across the payload, so neither the contiguity nor the
independence of the blocks holds, and a scalar has no leading axis to cut along.
-/
def blocksLayout (tag : String) (hdr : Header) : Except String (Nat × Nat) := do
  if hdr.fortran then
    throw <| formatError tag "leading-axis block loading requires C-order NPY arrays"
  match hdr.shape[0]? with
  | some d0 =>
      let tail := hdr.shape.extract 1 hdr.shape.size
      return (d0, tail.foldl (fun acc n => acc * n) 1)
  | none =>
      throw <| formatError tag
        "leading-axis block loading requires a leading axis; scalar arrays have none"

/-- Decode a validated full or prefix payload, checking its byte range before reserving values. -/
def decodePayload (tag : String) (hdr : Header) (bs : ByteArray) :
    Except String Contents := do
  if hdr.dataStart + hdr.elementCount * hdr.elementBytes > bs.size then
    throw <| formatError tag "NPY data truncated"
  let decode ← rangeDecoder tag hdr.dtype bs hdr.dataStart
  let raw := decode 0 hdr.elementCount
  let values ← if hdr.fortran then reorderFortranToC tag hdr.shape raw else pure raw
  return { dtype := hdr.dtype, shape := hdr.shape, values }

/--
Decode a validated C-order payload as one block per leading-axis index, checking its byte range
before reserving values.

For a C-order array of shape `(d₀, d₁, …, dₖ)` each leading-axis block is physically contiguous,
so the `d₀` blocks are independent and therefore support either sequential or concurrent decoding.
`parallelism` sets how many tasks that uses, within the bounds `resolveTaskCount` applies. One
bound is `d₀`, so a file with a small leading axis is decoded with at most `d₀` tasks however many
cores are free.
-/
def decodeBlocks (tag : String) (hdr : Header) (bs : ByteArray)
    (parallelism : Parallelism) : Except String Blocks := do
  let (d0, blockSize) ← blocksLayout tag hdr
  if hdr.dataStart + hdr.elementCount * hdr.elementBytes > bs.size then
    throw <| formatError tag "NPY data truncated"
  let decode ← rangeDecoder tag hdr.dtype bs hdr.dataStart
  let block : Nat → FloatArray := fun b => decode (b * blockSize) ((b + 1) * blockSize)
  let nTasks := resolveTaskCount parallelism d0 hdr.elementCount
  let blocks :=
    if nTasks ≤ 1 then
      (Array.range d0).map block
    else
      -- Each task decodes a run of consecutive blocks, since the task count follows the machine
      -- and the block count follows the file.
      --
      -- `Task.spawn` takes the work as a `Unit → α` closure, which is what keeps the decode
      -- inside the task body. A pure expression handed to an `IO`-level spawn can be floated out
      -- onto the spawning thread, and the fan-out then runs sequentially at the single-threaded
      -- rate.
      let perTask := (d0 + nTasks - 1) / nTasks
      let groups : Array (Task (Array FloatArray)) :=
        (Array.range nTasks).map fun t =>
          let lo := t * perTask
          let hi := min d0 (lo + perTask)
          Task.spawn (fun _ => (Array.range (hi - lo)).map (fun k => block (lo + k)))
      -- Regrouping moves one pointer per block, so it carries none of the per-element cost that
      -- joining decoded payloads would.
      groups.foldl (fun acc g => acc ++ g.get) (Array.emptyWithCapacity d0)
  return { dtype := hdr.dtype, shape := hdr.shape, blocks }

/--
Read exactly the requested bytes using bounded allocations, rejecting an early end of file.

The buffer grows only with bytes actually read. An untrusted element count never becomes the
capacity of a read buffer before the corresponding file contents have been observed.
-/
def readBytes (handle : _root_.IO.FS.Handle) (count : Nat) :
    _root_.IO (Except String ByteArray) := do
  let bytes ← readUpTo handle count
  if bytes.size < count then return .error (formatError "npy" "NPY data truncated")
  return .ok bytes

/-- When `readBytes handle count` succeeds, it returns exactly `count` bytes. -/
theorem readBytes_size {handle : _root_.IO.FS.Handle} {count : Nat} {bytes : ByteArray}
    (h : MonadAttach.CanReturn (readBytes handle count) (.ok bytes)) : bytes.size = count := by
  obtain ⟨read, hRead, h⟩ := LawfulMonadAttach.canReturn_bind_imp' h
  have hLe := readUpTo_size_le hRead
  split at h
  · cases LawfulMonadAttach.eq_of_canReturn_pure h
  · cases LawfulMonadAttach.eq_of_canReturn_pure h
    omega

/-- Read only the bounded NPY preamble and header, leaving the handle at the numeric payload. -/
def readHeaderFromHandle (handle : _root_.IO.FS.Handle) :
    _root_.IO (Except String Header) := ExceptT.run do
  let initial ← ExceptT.mk (readBytes handle 10)
  let preamble ←
    if initial[6]? == some 2 then
      let extra ← ExceptT.mk (readBytes handle 2)
      pure (initial ++ extra)
    else pure initial
  let (_, headerLen) ← headerLayout "npy" preamble
  let header ← ExceptT.mk (readBytes handle headerLen)
  return ← parseHeaderMeta "npy" (preamble ++ header)

/-- Read and decode the validated payload at the handle's current position. -/
def readPayload (handle : _root_.IO.FS.Handle) (hdr : Header) :
    _root_.IO (Except String Contents) := ExceptT.run do
  let bytes ← ExceptT.mk (readBytes handle (hdr.elementCount * hdr.elementBytes))
  return ← decodePayload "npy" { hdr with dataStart := 0 } bytes

/--
Parse the bytes of a `.npy` file into `Contents`, converting a Fortran-order payload to C order.

The parser rejects unsupported versions and dtypes, missing or malformed required header fields,
oversized shapes and truncated payloads. Header parsing reads the three required fields alone, and
a bad file fails at the trust boundary before it can become a tensor with the wrong shape or
partial data.
-/
def parse (tag : String) (bs : ByteArray) : Except String Contents := do
  let hdr ← parseHeaderMeta tag bs
  decodePayload tag hdr bs

/--
Parse only the requested leading rows of a C-order `.npy` array.

Large exported tensors can stay on disk while a run uses the first `n` rows. The rank and trailing
dimensions must match `expectedShape`, and the file's leading axis must be at least as long as the
requested one.

Header and dtype parsing are shared with `parse`, and only the requested prefix is decoded, so a
command that asks for a small leading slice of an image or sequence dataset never builds the full
payload. The requested range must be present, and unused trailing rows are not checked for
truncation.

C order is required because the first `n` rows of a row-major file are physically contiguous, so
the prefix is the first `n * trailingSize` elements. In a Fortran-order file the same logical
prefix is interleaved across the payload, and reading it as a prefix would return wrong rows.
Prefix loading therefore rejects Fortran-order arrays, and callers convert them to C order first.
-/
def parsePrefix
    (tag : String) (expectedShape : Array Nat) (bs : ByteArray) : Except String Contents := do
  let hdr ← parseHeaderMeta tag bs
  let selected ← prefixHeader tag hdr expectedShape
  decodePayload tag selected bs

/--
Parse a C-order `.npy` file into one `FloatArray` per index along the leading axis.

Use this when the consumer wants the blocks apart anyway: one channel, one band, one batch
element at a time. The blocks are decoded concurrently under the bounds `resolveTaskCount`
describes, and the result does not depend on the task count.

`parse` serves a consumer that wants one flat payload, and it decodes on one thread by design.
A concurrent flat decode would fill several separate arrays and then copy them into one contiguous
result. That final copy walks every element a second time, costs several times as much as the
concurrent decode itself, and holds the pieces and the result in memory at once. Splitting the
work pays only when the pieces are what the caller wants.

Fortran-order and scalar arrays are rejected, as `blocksLayout` explains.
-/
def parseBlocks (tag : String) (bs : ByteArray)
    (parallelism : Parallelism := .auto) : Except String Blocks := do
  let hdr ← parseHeaderMeta tag bs
  decodeBlocks tag hdr bs parallelism

/--
Read validated metadata without reading the payload.

Version 1.0 and 2.0 headers up to one MiB are supported. Dtype, storage-order syntax, dimensions
and shape products are validated, and `read`, `readPrefix` and `readBlocks` check that
the payload is complete.
-/
def readHeader (path : System.FilePath) : IO (Except String Header) :=
  _root_.IO.FS.withFile path .read readHeaderFromHandle

/-- Read the full declared NPY payload, rejecting truncation and converting Fortran to C order. -/
def read (path : System.FilePath) : IO (Except String Contents) := do
  _root_.IO.FS.withFile path .read fun handle => ExceptT.run do
    let hdr ← ExceptT.mk (readHeaderFromHandle handle)
    ExceptT.mk (readPayload handle hdr)

/--
Read only the header and requested leading rows of a C-order `.npy` file.

Unrequested rows are neither read nor decoded. The requested range must be complete, and rank and
trailing dimensions must match exactly. Reads use bounded chunks so a malformed claimed size cannot
trigger a correspondingly large allocation before any bytes have been read.
-/
def readPrefix
    (path : System.FilePath) (expectedShape : Array Nat) : IO (Except String Contents) := do
  _root_.IO.FS.withFile path .read fun handle => ExceptT.run do
    let hdr ← ExceptT.mk (readHeaderFromHandle handle)
    let selected ← prefixHeader "npy" hdr expectedShape
    ExceptT.mk (readPayload handle selected)

/--
Read a C-order `.npy` file as one `FloatArray` per index along the leading axis.

The header is validated for block loading before the payload is read, so a Fortran-order or scalar
file is rejected without reading its payload. The payload itself is read with the same bounded
chunks as `read`, and truncation is rejected.
-/
def readBlocks (path : System.FilePath)
    (parallelism : Parallelism := .auto) : IO (Except String Blocks) := do
  _root_.IO.FS.withFile path .read fun handle => ExceptT.run do
    let hdr ← ExceptT.mk (readHeaderFromHandle handle)
    let _ ← blocksLayout "npy" hdr
    let bytes ← ExceptT.mk (readBytes handle (hdr.elementCount * hdr.elementBytes))
    return ← decodeBlocks "npy" { hdr with dataStart := 0 } bytes parallelism

end NPY
end Data
end TorchLean
