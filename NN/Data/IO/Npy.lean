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
- deterministic conversion to a flat `FloatArray` in C order.

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
  /-- Flattened numeric payload, converted to Lean `Float` values.

  This is a `FloatArray` rather than an `Array Float` because the payload is bulk numeric
  data: `FloatArray` stores the doubles unboxed, so a large file costs one machine word per
  element instead of one word plus one heap cell. -/
  values : FloatArray

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

/-- Reorder a Fortran-ordered flat array into C-order, rejecting an inconsistent payload.

The payload is a `FloatArray` for the reason `NpyData.values` gives — the doubles stay unboxed —
and an out-of-range index is an error rather than a `0.0` fill, so a malformed file is reported
instead of silently reshaped. Both properties are load-bearing and neither implies the other. -/
def reorderFortranToC
    (tag : String) (shape : Array Nat) (raw : FloatArray) : Except String FloatArray := do
  let count := shape.foldl (fun acc n => acc * n) 1
  let mut out := FloatArray.emptyWithCapacity count
  for i in [0:count] do
    let idxF := idxFortranOfCIdx shape i
    if idxF < raw.size then
      out := out.push (raw.get! idxF)
    else
      throw <| formatError tag <|
        s!"Fortran-order index {idxF} is outside the {raw.size}-element payload"
  pure out

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
def parseShapeValue (s : String) : Option (Array Nat) :=
  let cs := dropUntil (fun c => c = '(') (s.trimAsciiStart).toString.toList
  match cs with
  | '(' :: rest =>
      let (inside, _) := takeUntilChar ')' rest
      let parts := (String.ofList inside).splitOn ","
      let dims := parts.map (fun p => (p.trimAscii).toString) |>.filter (fun x => x != "")
      let rec parseAll (xs : List String) (acc : Array Nat) : Option (Array Nat) :=
        match xs with
        | [] => some acc
        | x :: xs =>
            match parseNatValue x with
            | some n => parseAll xs (acc.push n)
            | none => none
      parseAll dims #[]
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

/-- Parsed `.npy` metadata needed before reading the numeric payload. -/
structure NpyHeaderMeta where
  /-- Dtype descriptor from the header, for example `"<f4"` or `"<f8"`. -/
  descr : String
  /-- Whether the on-disk payload is Fortran-ordered. -/
  fortran : Bool
  /-- Logical array shape from the header. -/
  shape : Array Nat
  /-- Byte offset where the numeric payload begins. -/
  dataStart : Nat

/-- Read and validate the NumPy magic/version/header block shared by all NPY loaders. -/
def parseNpyHeaderMeta (tag : String) (bs : ByteArray) : Except String NpyHeaderMeta := do
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
      if !(major = 1 || major = 2) then
        .error (formatError tag s!"unsupported NPY version: {major}")
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
            let headerEnd := headerStart + headerLen
            if headerEnd > bs.size then
              .error (formatError tag "NPY header out of bounds")
            else
              let headerBytes := bs.extract headerStart headerEnd
              let headerStr :=
                match String.fromUTF8? headerBytes with
                | some s => s
                | none => ""
              let (descr, fortran, shape) <- parseHeader tag headerStr
              .ok { descr := descr, fortran := fortran, shape := shape, dataStart := headerEnd }

/-- Byte width for the dtypes supported by TorchLean's NPY loader. -/
def npyElementBytes (tag descr : String) : Except String Nat :=
  if descr = "<f8" then
    .ok 8
  else if descr = "<f4" then
    .ok 4
  else
    .error (formatError tag s!"unsupported dtype: {descr}")

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

/-!
### Payload decoding

`byteAt?` and `readNpyElement` are the right shape at the file boundary, where a malformed
file has to be reported rather than guessed at. Inside the payload loop they are the wrong
shape: `Option` per byte and `Except` per element each cost a heap cell, so decoding an
array pays about ten allocations per element and runs an order of magnitude below the rate
at which the same loop can fill a `FloatArray`.

The functions below decode a payload whose bounds and dtype the caller has already checked.
The dtype is matched once per file rather than once per element, the bytes are read without
an intermediate `Option`, and the result is unboxed.
-/

/-- Little-endian `UInt64` at byte offset `o`. Out-of-range bytes read as `0`; callers check
the payload length first, so that fallback is unreachable for accepted files. -/
@[inline] def uint64LE (bs : ByteArray) (o : Nat) : UInt64 :=
  (bs.get! o).toUInt64
    ||| ((bs.get! (o + 1)).toUInt64 <<< 8)
    ||| ((bs.get! (o + 2)).toUInt64 <<< 16)
    ||| ((bs.get! (o + 3)).toUInt64 <<< 24)
    ||| ((bs.get! (o + 4)).toUInt64 <<< 32)
    ||| ((bs.get! (o + 5)).toUInt64 <<< 40)
    ||| ((bs.get! (o + 6)).toUInt64 <<< 48)
    ||| ((bs.get! (o + 7)).toUInt64 <<< 56)

/-- Little-endian `UInt32` at byte offset `o`, with the same precondition as `uint64LE`. -/
@[inline] def uint32LE (bs : ByteArray) (o : Nat) : UInt32 :=
  (bs.get! o).toUInt32
    ||| ((bs.get! (o + 1)).toUInt32 <<< 8)
    ||| ((bs.get! (o + 2)).toUInt32 <<< 16)
    ||| ((bs.get! (o + 3)).toUInt32 <<< 24)

/-- Decode the half-open element range `[lo, hi)` of a `<f8` payload. -/
def decodeRangeF8 (bs : ByteArray) (dataStart lo hi : Nat) : FloatArray := Id.run do
  let mut out := FloatArray.emptyWithCapacity (hi - lo)
  for i in [lo:hi] do
    out := out.push (Float.ofBits (uint64LE bs (dataStart + i * 8)))
  pure out

/-- Decode the half-open element range `[lo, hi)` of a `<f4` payload, widening to `Float`. -/
def decodeRangeF4 (bs : ByteArray) (dataStart lo hi : Nat) : FloatArray := Id.run do
  let mut out := FloatArray.emptyWithCapacity (hi - lo)
  for i in [lo:hi] do
    out := out.push (Float32.toFloat (Float32.ofBits (uint32LE bs (dataStart + i * 4))))
  pure out

/-- The range decoder for a supported dtype, selected once per file.

Returning the decoder rather than the decoded payload is what keeps the dtype test out of
the element loop, and it lets one selection serve many ranges. -/
def rangeDecoder (tag descr : String) (bs : ByteArray) (dataStart : Nat) :
    Except String (Nat → Nat → FloatArray) :=
  if descr = "<f8" then
    .ok (decodeRangeF8 bs dataStart)
  else if descr = "<f4" then
    .ok (decodeRangeF4 bs dataStart)
  else
    .error (formatError tag s!"unsupported dtype: {descr}")

/-- Element count below which spreading a decode across tasks does not pay.

A task costs a scheduling round trip that only a reasonably large slice earns back, so this
bounds the task count from below by `totalElements / minElementsPerTask`. -/
def minElementsPerTask : Nat := 1 <<< 16

/-- How a leading-axis decode is spread across tasks. -/
inductive Parallelism where
  /-- Decode every block on the calling thread. -/
  | sequential
  /-- Spread the blocks across at most `n` tasks. -/
  | tasks (n : Nat)
  /-- Spread the blocks across the platform's hardware concurrency. -/
  | auto
  deriving Repr, BEq

/-- Resolve a `Parallelism` request against a payload into an actual task count.

Three bounds apply, and the smallest wins:

- the caller's request, or the platform's hardware concurrency for `auto`;
- the number of blocks. A block is the smallest piece this can hand back whole, so raising
  the task count past `nBlocks` would mean cutting a block in half and copying the halves
  back together afterwards — and that copy costs more than the extra task saves. The
  leading dimension is therefore a hard ceiling on this decode's parallelism;
- `totalElements / minElementsPerTask`, so small payloads stay on one thread.
-/
def resolveTaskCount (p : Parallelism) (nBlocks totalElements : Nat) : Nat :=
  let requested :=
    match p with
    | .sequential => 1
    | .tasks n => n
    | .auto => (System.Platform.Internal.getHardwareConcurrency ()).toNat
  max 1 (min (min (max 1 requested) nBlocks) (totalElements / minElementsPerTask))

end Internal

/--
Parse the bytes of a `.npy` file into `NpyData`.

The parser rejects unsupported dtypes, malformed headers, and truncated payloads.
That makes loader failures explicit at the trust boundary instead of silently producing tensors with
the wrong shape or partial data.
-/
def parseNpy (tag : String) (bs : ByteArray) : Except String NpyData := do
  let hdr <- parseNpyHeaderMeta tag bs
  let bytesPer <- npyElementBytes tag hdr.descr
  let count := hdr.shape.foldl (fun acc n => acc * n) 1
  let dataBytes := count * bytesPer
  if hdr.dataStart + dataBytes > bs.size then
    .error (formatError tag "NPY data truncated")
  else
    let decode ← rangeDecoder tag hdr.descr bs hdr.dataStart
    let raw := decode 0 count
    let values ← if hdr.fortran then reorderFortranToC tag hdr.shape raw else pure raw
    .ok { dtype := hdr.descr, shape := hdr.shape, fortran := false, values := values }

/--
Parse only the requested leading rows of a C-order `.npy` array.

This supports large exported tensors kept on disk while a run uses only the first `n` rows. The rank
and trailing dimensions must match exactly; only the leading axis may be larger than requested.

The implementation shares header and dtype parsing with `parseNpy`, then decodes only the requested
prefix. This avoids building a full `Array Float` when a command asks for a small leading slice of a
real image or sequence dataset.

Why C-order only?  In row-major NPY files, the first `n` rows are physically contiguous, so the
prefix is exactly the first `n * trailingSize` elements.  In Fortran-order files the same logical
prefix is interleaved across the payload, so prefix decoding would be unsound.  Rather than
silently returning bad rows, we reject Fortran-order prefix loading and ask callers to convert the
array to C-order first.
-/
def parseNpyLeadingAxisPrefix
    (tag : String) (expectedShape : Array Nat) (bs : ByteArray) : Except String NpyData := do
  let hdr <- parseNpyHeaderMeta tag bs
  let bytesPer <- npyElementBytes tag hdr.descr
  if hdr.fortran then
    .error (formatError tag "prefix row loading requires C-order NPY arrays")
  else
    match expectedShape[0]?, hdr.shape[0]? with
    | some expectedN, some actualN =>
        let expectedTail := expectedShape.extract 1 expectedShape.size
        let actualTail := hdr.shape.extract 1 hdr.shape.size
        if actualTail != expectedTail then
          .error (formatError tag
            s!"shape mismatch: expected trailing dims {expectedTail}, got {actualTail}")
        else if actualN < expectedN then
          .error (formatError tag s!"expected at least {expectedN} rows, got {actualN}")
        else
          let expectedCount := expectedShape.foldl (fun acc n => acc * n) 1
          let actualCount := hdr.shape.foldl (fun acc n => acc * n) 1
          if hdr.dataStart + actualCount * bytesPer > bs.size then
            .error (formatError tag "NPY data truncated")
          else if hdr.dataStart + expectedCount * bytesPer > bs.size then
            .error (formatError tag "NPY prefix data truncated")
          else
            let decode ← rangeDecoder tag hdr.descr bs hdr.dataStart
            .ok { dtype := hdr.descr, shape := expectedShape, fortran := false,
                  values := decode 0 expectedCount }
    | none, none =>
        .ok { dtype := hdr.descr, shape := #[], fortran := false,
              values := FloatArray.emptyWithCapacity 0 }
    | _, _ =>
        .error (formatError tag s!"shape mismatch: expected {expectedShape}, got {hdr.shape}")

/--
A `.npy` payload decoded as one `FloatArray` per index along the leading axis.

`blocks.size` is the leading dimension and every block holds the product of the trailing
dimensions, so `shape = #[d₀] ++ tail` gives `blocks.size = d₀` and
`blocks[i]!.size = tail.foldl (· * ·) 1`.
-/
structure NpyBlocks where
  /-- Dtype string as stored in the header, for example `"<f4"` or `"<f8"`. -/
  dtype : String
  /-- Logical array shape as stored in the header. -/
  shape : Array Nat
  /-- One decoded block per index along the leading axis, in order. -/
  blocks : Array FloatArray

/--
Parse a C-order `.npy` file into one `FloatArray` per index along the leading axis.

For a C-order array of shape `(d₀, d₁, …, dₖ)` each leading-axis block is physically
contiguous, so the `d₀` blocks are independent and therefore support either sequential or
concurrent decoding.

`parallelism` chooses how many tasks that uses; see `resolveTaskCount` for the bounds that
apply to the request. `d₀` is a ceiling on it, so a file with a small leading axis is decoded
with at most that many tasks however many cores are free.

Use this when the consumer wants the blocks apart anyway — one channel, one band, one batch
element at a time.

Use `parseNpy` when the consumer wants a single flat payload instead. That path decodes on
one thread by design. Decoding it concurrently would mean filling several separate arrays
and then copying all of them into one contiguous result, and that final copy walks every
element a second time: it costs several times the concurrent decode it was supposed to
speed up, and it holds the pieces and the result in memory simultaneously. Splitting the
work is only worth it when the pieces are what the caller wanted.

Fortran-order files are rejected. Their leading-axis blocks are interleaved across the
payload rather than contiguous, so neither the independence nor the contiguity holds.
-/
def parseNpyLeadingAxisBlocks (tag : String) (bs : ByteArray)
    (parallelism : Parallelism := .auto) : Except String NpyBlocks := do
  let hdr <- parseNpyHeaderMeta tag bs
  let bytesPer <- npyElementBytes tag hdr.descr
  if hdr.fortran then
    .error (formatError tag "leading-axis block loading requires C-order NPY arrays")
  else
    match hdr.shape[0]? with
    | none => .error (formatError tag "leading-axis block loading requires a non-scalar array")
    | some d0 =>
        let tail := hdr.shape.extract 1 hdr.shape.size
        let blockSize := tail.foldl (fun acc n => acc * n) 1
        let count := d0 * blockSize
        if hdr.dataStart + count * bytesPer > bs.size then
          .error (formatError tag "NPY data truncated")
        else
          let decode <- rangeDecoder tag hdr.descr bs hdr.dataStart
          let block : Nat -> FloatArray := fun b => decode (b * blockSize) ((b + 1) * blockSize)
          let nTasks := resolveTaskCount parallelism d0 count
          let blocks :=
            if nTasks <= 1 then
              (Array.range d0).map block
            else
              -- Blocks are grouped, not handed out one per task: the task count follows the
              -- machine and the block count follows the file, and those are independent.
              --
              -- `Task.spawn` takes the work as a `Unit → α` closure, which is what keeps the
              -- decode inside the task body. Handing a pure expression to an `IO`-level spawn
              -- instead lets it be floated out onto the spawning thread, and the fan-out then
              -- runs sequentially at exactly the single-threaded rate.
              let perTask := (d0 + nTasks - 1) / nTasks
              let groups : Array (Task (Array FloatArray)) :=
                (Array.range nTasks).map fun t =>
                  let lo := t * perTask
                  let hi := min d0 (lo + perTask)
                  Task.spawn (fun _ => (Array.range (hi - lo)).map (fun k => block (lo + k)))
              -- Regrouping moves one pointer per block, not one double per element, so it
              -- carries none of the cost that joining decoded payloads would.
              groups.foldl (fun acc g => acc ++ g.get) (Array.emptyWithCapacity d0)
          .ok { dtype := hdr.descr, shape := hdr.shape, blocks := blocks }

/-- Read a `.npy` file from disk and parse it as `NpyBlocks`. -/
def readNpyLeadingAxisBlocks (path : System.FilePath)
    (parallelism : Parallelism := .auto) : IO (Except String NpyBlocks) := do
  let bs <- IO.FS.readBinFile path
  pure (parseNpyLeadingAxisBlocks (tag := "npy") bs parallelism)

/-- Read a `.npy` file from disk and parse it as `NpyData`. -/
def readNpy (path : System.FilePath) : IO (Except String NpyData) := do
  let bs <- IO.FS.readBinFile path
  pure (parseNpy (tag := "npy") bs)

/--
Read a `.npy` file but decode only the requested leading rows.

This is the file-system wrapper around `parseNpyLeadingAxisPrefix`.  It still reads the file bytes into
memory, but it avoids building a full `Array Float` for rows the run did not ask to use.  The
public `API.Data` layer uses this when a dataset source says "load the first `n` examples" from a
larger exported NPY tensor.
-/
def readNpyLeadingAxisPrefix
    (path : System.FilePath) (expectedShape : Array Nat) : IO (Except String NpyData) := do
  let bs <- IO.FS.readBinFile path
  pure (parseNpyLeadingAxisPrefix (tag := "npy") expectedShape bs)

end IO
end Data
end TorchLean
