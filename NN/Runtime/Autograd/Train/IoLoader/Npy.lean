/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Train.IoLoader.Common

/-!
# NPY loaders for typed training tensors

This module implements the small, explicit `.npy` subset that TorchLean's native training
examples need:

- NumPy format versions 1 and 2;
- little-endian `float32` and `float64` payloads (`<f4`, `<f8`);
- C-order arrays directly, and Fortran-order arrays converted to C-order at load time;
- typed 1D and 2D tensor views for vectors and matrices.

The loader deliberately stays narrow. It is a runtime bridge for trusted experiment artifacts, not
a general NumPy parser and not part of the formal tensor semantics. Keeping it here, under
`Runtime.Autograd.Train`, makes that boundary visible while still giving examples a convenient path
from Python-produced arrays into TorchLean tensors.

Reference:
- NumPy `.npy` format documentation:
  https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Train

open Spec
open Tensor
open IoLoader.Internal

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
  shape : List Nat
  /-- Whether the returned flat payload is still Fortran-ordered. This loader returns `false`. -/
  fortran : Bool
  /-- Flattened numeric payload, converted to Lean `Float` values. -/
  values : Array Float

namespace IoLoader.Internal

/--
Prefix products of a shape list.

For a shape `[d₀, d₁, d₂]`, this returns `[1, d₀, d₀*d₁]`, which are exactly the
Fortran-order strides. We use these strides to convert Fortran storage into TorchLean's ordinary
C-order flattening convention.
-/
def prefixProducts (shape : List Nat) : List Nat :=
  let rec go (acc : Nat) : List Nat → List Nat
    | [] => []
    | d :: ds => acc :: go (acc * d) ds
  go 1 shape

/--
Convert a linear C-order index to the corresponding linear Fortran-order index.

Both indices describe the same multi-dimensional coordinate. The difference is only how the
coordinate is flattened into a one-dimensional payload.
-/
def idxFortranOfCIdx (shape : List Nat) (idxC : Nat) : Nat :=
  let dimsRev := shape.reverse
  let stridesRev := (prefixProducts shape).reverse
  let rec go (dims strides : List Nat) (idx : Nat) (acc : Nat) : Nat :=
    match dims, strides with
    | [], [] => acc
    | d :: ds, s :: ss =>
        let coord := idx % d
        let idx' := idx / d
        go ds ss idx' (acc + coord * s)
    | _, _ => acc
  go dimsRev stridesRev idxC 0

/--
Reorder a Fortran-ordered flat array into C-order.

The function is total and defensive: if the file payload is malformed and an index is missing, the
missing element is filled with `0.0`. The parser checks payload length before calling this function,
so that fallback should not happen for accepted files.
-/
def reorderFortranToC (shape : List Nat) (raw : Array Float) : Array Float :=
  let count := shape.foldl (fun acc n => acc * n) 1
  Array.ofFn (n := count) (fun (i : Fin count) =>
    let idxF := idxFortranOfCIdx shape i.val
    (raw[idxF]?).getD 0.0)

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
def parseShapeValue (s : String) : Option (List Nat) :=
  let cs := dropUntil (fun c => c = '(') (s.trimAsciiStart).toString.toList
  match cs with
  | '(' :: rest =>
      let (inside, _) := takeUntilChar ')' rest
      let parts := (String.ofList inside).splitOn ","
      let dims := parts.map (fun p => (p.trimAscii).toString) |>.filter (fun x => x != "")
      let rec parseAll (xs : List String) (acc : List Nat) : Option (List Nat) :=
        match xs with
        | [] => some acc.reverse
        | x :: xs =>
            match parseNatValue x with
            | some n => parseAll xs (n :: acc)
            | none => none
      parseAll dims []
  | _ => none

/--
Parse the NumPy header dictionary.

We only need three standard fields: `descr`, `fortran_order`, and `shape`. The header format is a
Python-literal dictionary padded to an alignment boundary; this parser is intentionally
field-oriented rather than a full Python parser.
-/
def parseHeader (tag : String) (hdr : String) :
  Result (String × Bool × List Nat) :=
  let descrOpt := findField hdr "descr" >>= parseQuotedValue
  let fortranOpt := findField hdr "fortran_order" >>= parseBoolValue
  let shapeOpt := findField hdr "shape" >>= parseShapeValue
  match descrOpt, fortranOpt, shapeOpt with
  | some descr, some fortran, some shape => .ok (descr, fortran, shape)
  | _, _, _ => .error (tagError tag "failed to parse NPY header")

end IoLoader.Internal

/--
Parse the bytes of a `.npy` file into `NpyData`.

The parser rejects unsupported dtypes, malformed headers, and truncated payloads.
That makes loader failures explicit at the trust boundary instead of silently producing tensors with
the wrong shape or partial data.
-/
def parseNpy (tag : String) (bs : ByteArray) : Result NpyData := do
  if bs.size < 10 then
    .error (tagError tag "file too small")
  else
    let magicOk :=
      ((byteAt? bs 0).map (fun b => b.toNat == 0x93) |>.getD false)
      && ((byteAt? bs 1).map (fun b => b.toNat == ('N' : Char).toNat) |>.getD false)
      && ((byteAt? bs 2).map (fun b => b.toNat == ('U' : Char).toNat) |>.getD false)
      && ((byteAt? bs 3).map (fun b => b.toNat == ('M' : Char).toNat) |>.getD false)
      && ((byteAt? bs 4).map (fun b => b.toNat == ('P' : Char).toNat) |>.getD false)
      && ((byteAt? bs 5).map (fun b => b.toNat == ('Y' : Char).toNat) |>.getD false)
    if !magicOk then
      .error (tagError tag "invalid NPY magic header")
    else
      let major := (byteAt? bs 6).map (fun b => b.toNat) |>.getD 0
      if !(major = 1 || major = 2) then
        .error (tagError tag s!"unsupported NPY version: {major}")
      else
        let headerLenOpt :=
          if major = 1 then
            readUInt16LE bs 8
          else
            readUInt32LE bs 8 |>.map UInt32.toNat
        match headerLenOpt with
        | none => .error (tagError tag "invalid NPY header length")
        | some headerLen =>
            let headerStart := if major = 1 then 10 else 12
            let headerEnd := headerStart + headerLen
            if headerEnd > bs.size then
              .error (tagError tag "NPY header out of bounds")
            else
              let headerBytes := bs.extract headerStart headerEnd
              let headerStr :=
                match String.fromUTF8? headerBytes with
                | some s => s
                | none => ""
              let (descr, fortran, shape) <- parseHeader tag headerStr
              let (bytesPer, readElem) :=
                if descr = "<f8" then
                  (8, fun off =>
                    match readUInt64LE bs off with
                    | some w => .ok (Float.ofBits w)
                    | none => .error (tagError tag "invalid float64 data"))
                else if descr = "<f4" then
                  (4, fun off =>
                    match readUInt32LE bs off with
                    | some w => .ok (Float32.toFloat (Float32.ofBits w))
                    | none => .error (tagError tag "invalid float32 data"))
                else
                  (0, fun _ => .error (tagError tag s!"unsupported dtype: {descr}"))
              if bytesPer = 0 then
                .error (tagError tag s!"unsupported dtype: {descr}")
              else
                let count := shape.foldl (fun acc n => acc * n) 1
                let dataStart := headerEnd
                let dataBytes := count * bytesPer
                if dataStart + dataBytes > bs.size then
                  .error (tagError tag "NPY data truncated")
                else
                  let mut raw : Array Float := Array.mkEmpty count
                  for i in [0:count] do
                    let v ← readElem (dataStart + i * bytesPer)
                    raw := raw.push v
                  let values := if fortran then reorderFortranToC shape raw else raw
                  .ok { dtype := descr, shape := shape, fortran := false, values := values }

/--
Parse only the requested leading rows of a C-order `.npy` array.

This supports the common tutorial workflow where a large exported tensor is kept on disk but a run
uses only the first `n` rows. The rank and trailing dimensions must match exactly; only dim 0 may be
larger than requested.

The implementation intentionally repeats the small NPY header checks instead of calling `parseNpy`
and slicing afterwards.  `parseNpy` decodes the entire data payload; that is fine for small examples
but wasteful when a command asks for a quick prefix of a real image or sequence dataset.  Here we
read the header, validate that the file layout is compatible with the requested type-level shape,
and then decode exactly `expectedShape.product` elements.

Why C-order only?  In row-major NPY files, the first `n` rows are physically contiguous, so the
prefix is exactly the first `n * trailingSize` elements.  In Fortran-order files the same logical
prefix is interleaved across the payload, so a cheap prefix decode would be wrong.  Rather than
silently returning bad rows, we reject Fortran-order prefix loading and ask callers to convert the
array to C-order first.
-/
def parseNpyPrefixDim0 (tag : String) (expectedShape : List Nat) (bs : ByteArray) : Result NpyData := do
  if bs.size < 10 then
    .error (tagError tag "file too small")
  else
    let magicOk :=
      ((byteAt? bs 0).map (fun b => b.toNat == 0x93) |>.getD false)
      && ((byteAt? bs 1).map (fun b => b.toNat == ('N' : Char).toNat) |>.getD false)
      && ((byteAt? bs 2).map (fun b => b.toNat == ('U' : Char).toNat) |>.getD false)
      && ((byteAt? bs 3).map (fun b => b.toNat == ('M' : Char).toNat) |>.getD false)
      && ((byteAt? bs 4).map (fun b => b.toNat == ('P' : Char).toNat) |>.getD false)
      && ((byteAt? bs 5).map (fun b => b.toNat == ('Y' : Char).toNat) |>.getD false)
    if !magicOk then
      .error (tagError tag "invalid NPY magic header")
    else
      let major := (byteAt? bs 6).map (fun b => b.toNat) |>.getD 0
      if !(major = 1 || major = 2) then
        .error (tagError tag s!"unsupported NPY version: {major}")
      else
        let headerLenOpt :=
          if major = 1 then
            readUInt16LE bs 8
          else
            readUInt32LE bs 8 |>.map UInt32.toNat
        match headerLenOpt with
        | none => .error (tagError tag "invalid NPY header length")
        | some headerLen =>
            let headerStart := if major = 1 then 10 else 12
            let headerEnd := headerStart + headerLen
            if headerEnd > bs.size then
              .error (tagError tag "NPY header out of bounds")
            else
              let headerBytes := bs.extract headerStart headerEnd
              let headerStr :=
                match String.fromUTF8? headerBytes with
                | some s => s
                | none => ""
              let (descr, fortran, shape) <- parseHeader tag headerStr
              let (bytesPer, readElem) :=
                if descr = "<f8" then
                  (8, fun off =>
                    match readUInt64LE bs off with
                    | some w => .ok (Float.ofBits w)
                    | none => .error (tagError tag "invalid float64 data"))
                else if descr = "<f4" then
                  (4, fun off =>
                    match readUInt32LE bs off with
                    | some w => .ok (Float32.toFloat (Float32.ofBits w))
                    | none => .error (tagError tag "invalid float32 data"))
                else
                  (0, fun _ => .error (tagError tag s!"unsupported dtype: {descr}"))
              if bytesPer = 0 then
                .error (tagError tag s!"unsupported dtype: {descr}")
              else if fortran then
                .error (tagError tag "prefix row loading requires C-order NPY arrays")
              else
                match expectedShape, shape with
                | expectedN :: expectedTail, actualN :: actualTail =>
                    if actualTail != expectedTail then
                      .error (tagError tag
                        s!"shape mismatch: expected trailing dims {expectedTail}, got {actualTail}")
                    else if actualN < expectedN then
                      .error (tagError tag s!"expected at least {expectedN} rows, got {actualN}")
                    else
                      let expectedCount := expectedShape.foldl (fun acc n => acc * n) 1
                      let actualCount := shape.foldl (fun acc n => acc * n) 1
                      let dataStart := headerEnd
                      if dataStart + actualCount * bytesPer > bs.size then
                        .error (tagError tag "NPY data truncated")
                      else if dataStart + expectedCount * bytesPer > bs.size then
                        .error (tagError tag "NPY prefix data truncated")
                      else
                        let mut raw : Array Float := Array.mkEmpty expectedCount
                        for i in [0:expectedCount] do
                          let v ← readElem (dataStart + i * bytesPer)
                          raw := raw.push v
                        .ok { dtype := descr, shape := expectedShape, fortran := false, values := raw }
                | [], [] =>
                    .ok { dtype := descr, shape := [], fortran := false, values := #[] }
                | _, _ =>
                    .error (tagError tag s!"shape mismatch: expected {expectedShape}, got {shape}")

/-- Read a `.npy` file from disk and parse it as `NpyData`. -/
def readNpy (path : System.FilePath) : IO (Result NpyData) := do
  let bs <- IO.FS.readBinFile path
  pure (parseNpy (tag := "npy") bs)

/--
Read a `.npy` file but decode only the requested leading rows.

This is the file-system wrapper around `parseNpyPrefixDim0`.  It still reads the file bytes into
memory, but it avoids building a full `Array Float` for rows the run did not ask to use.  The
public `API.Data` layer uses this when a dataset source says "load the first `n` examples" from a
larger exported NPY tensor.
-/
def readNpyPrefixDim0 (path : System.FilePath) (expectedShape : List Nat) : IO (Result NpyData) := do
  let bs <- IO.FS.readBinFile path
  pure (parseNpyPrefixDim0 (tag := "npy") expectedShape bs)

/--
Read a 1D `.npy` file as a typed TorchLean vector tensor.

The shape check is part of the loader contract: files with the wrong logical size are rejected
instead of being reshaped implicitly.
-/
def readNpyVector (path : System.FilePath) (n : Nat) :
  IO (Result (Tensor Float (.dim n .scalar))) := do
  let res <- readNpy path
  match res with
  | .error e => pure (.error e)
  | .ok data =>
      if data.shape = [n] then
        let f : Fin n -> Float := fun i => (data.values[i.val]?).getD 0.0
        pure (.ok (vectorN n f))
      else
        pure (.error (tagError "npy" "shape mismatch for vector"))

/--
Read a 2D `.npy` file as a typed TorchLean matrix tensor.

The returned matrix uses the same row-major indexing convention as the rest of the runtime tensor
helpers.
-/
def readNpyMatrix (path : System.FilePath) (m n : Nat) :
  IO (Result (Tensor Float (.dim m (.dim n .scalar)))) := do
  let res <- readNpy path
  match res with
  | .error e => pure (.error e)
  | .ok data =>
      if data.shape = [m, n] then
        let f : Fin m -> Fin n -> Float := fun i j =>
          let idx := i.val * n + j.val
          (data.values[idx]?).getD 0.0
        pure (.ok (matrixMN m n f))
      else
        pure (.error (tagError "npy" "shape mismatch for matrix"))

end Train
end Autograd
end Runtime
