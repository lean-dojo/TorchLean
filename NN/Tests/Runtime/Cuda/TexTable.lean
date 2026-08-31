/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Buffer
public import NN.Runtime.Autograd.Engine.Cuda.TexTable
public import NN.Tests.Runtime.Cuda.Utils

/-!
# CUDA Kernel Coverage: Lookup-Table Textures

Validates `Runtime.Autograd.Cuda.TexTable` against an executable `Float32` reference:

- **point mode** must match the reference **bit-for-bit** in both builds (the CPU stub and the
  CUDA kernel evaluate the same clamp/floor/lerp in float32 with FMA contraction blocked);
- **hardware mode** is compared by tolerance (`2⁻⁸` of the local sample gap): CUDA's texture unit
  uses a 9-bit fixed-point lerp weight whose rounding is unspecified;
- **integer-node fetches** must be bit-exact in *both* modes (the lerp weight is exactly 0 there) —
  this probe catches any texel-center (`±0.5`) coordinate-convention error;
- edge behavior: clamping below/above the abscissa range, layer-index clamping, `width = 1`,
  and empty coordinate buffers.
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace TexTable

open Runtime.Autograd.Cuda

/-- Float32 clamp mirroring the native kernels' `fminf(fmaxf(u, 0), wmax)`. -/
def clampF32 (x lo hi : Float32) : Float32 :=
  if x < lo then lo else if x > hi then hi else x

/--
Executable reference for `TexTable.fetch`, computed in Lean core `Float32` (IEEE binary32, the
same arithmetic as the native kernels). Mirrors the CPU stub statement-for-statement; in point
mode the CUDA kernel is bit-identical by construction, in hardware mode the CUDA texture unit may
round the 9-bit weight differently (hence tolerance).
-/
def refFetch (tab : Array Float32) (width layers : Nat) (hw : Bool)
    (u layer : Float32) : Float32 :=
  let wmax : Float32 := Float32.ofNat (width - 1)
  let uc := clampF32 u 0.0 wmax
  let lF := clampF32 (layer + 0.5) 0.0 (Float32.ofNat (layers - 1))
  let L := lF.toFloat.toUInt64.toNat
  let j := uc.floor
  let f := uc - j
  let f := if hw then Float32.floor (f * 256.0 + 0.5) / 256.0 else f
  let j0 := j.toFloat.toUInt64.toNat
  let j1 := if j0 + 1 < width then j0 + 1 else width - 1
  let a := tab[L * width + j0]!
  let b := tab[L * width + j1]!
  a + f * (b - a)

/-- The float64 sample grid used by every test below (3 layers × 7 samples, non-trivial values). -/
def sampleData (width layers : Nat) : FloatArray := Id.run do
  let mut a := FloatArray.emptyWithCapacity (width * layers)
  for l in [0:layers] do
    for i in [0:width] do
      let x := Float.ofNat i
      let y := Float.ofNat l
      a := a.push (0.17 * x - 0.031 * x * x + 0.4 * y + 0.05)
  return a

/-- Run one fetch through the native path and the reference, returning both as `Float32`. -/
def runFetch (t : TexTable) (tab32 : Array Float32) (width layers : Nat) (hw : Bool)
    (coords layerIdx : FloatArray) : IO (Array Float32 × Array Float32) := do
  let out := Buffer.toFloatArray
    (Runtime.Autograd.Cuda.TexTable.fetch t (Buffer.ofFloatArray coords)
      (Buffer.ofFloatArray layerIdx))
  let mut native : Array Float32 := #[]
  let mut refv : Array Float32 := #[]
  for i in [0:out.size] do
    native := native.push (out[i]!).toFloat32
    refv := refv.push
      (refFetch tab32 width layers hw (coords[i]!).toFloat32 (layerIdx[i]!).toFloat32)
  return (native, refv)

/-- Assert bitwise equality of native and reference results. -/
def assertBits (msg : String) (native refv : Array Float32) : IO Unit := do
  for i in [0:native.size] do
    let x := native[i]!
    let y := refv[i]!
    unless x.toBits == y.toBits do
      throw <| IO.userError
        s!"{msg}[{i}]: native {x} (bits {x.toBits}) ≠ reference {y} (bits {y.toBits})"

/-- Assert tolerance agreement of native and reference results. -/
def assertTol (msg : String) (native refv : Array Float32) (tol : Float) : IO Unit := do
  for i in [0:native.size] do
    let x := (native[i]!).toFloat
    let y := (refv[i]!).toFloat
    Utils.assertApprox s!"{msg}[{i}]" x y (tol := tol)

def widthN : Nat := 7
def layersN : Nat := 3

/-- Interior, boundary, and out-of-range grid coordinates across all layers. -/
def probeCoords : FloatArray :=
  FloatArray.mk #[0.0, 0.25, 1.0, 2.5, 3.75, 5.999, 6.0, -1.5, 9.75, 4.125, 2.875, 0.001]

def probeLayers : FloatArray :=
  FloatArray.mk #[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]

def runPointMode : IO Unit := do
  IO.println "== textable point mode (bit-exact) =="
  let data := sampleData widthN layersN
  let tab32 := (Array.range data.size).map fun i => (data[i]!).toFloat32
  let t := Runtime.Autograd.Cuda.TexTable.ofFloatArray data widthN layersN (hwFilter := false)
  unless Runtime.Autograd.Cuda.TexTable.width t == widthN &&
      Runtime.Autograd.Cuda.TexTable.layers t == layersN &&
      Runtime.Autograd.Cuda.TexTable.filterHw t == false do
    throw <| IO.userError "textable point: metadata mismatch"
  let (native, refv) ← runFetch t tab32 widthN layersN false probeCoords probeLayers
  assertBits "textable point" native refv

def runHardwareMode : IO Unit := do
  IO.println "== textable hardware mode (tolerance) =="
  let data := sampleData widthN layersN
  let tab32 := (Array.range data.size).map fun i => (data[i]!).toFloat32
  let t := Runtime.Autograd.Cuda.TexTable.ofFloatArray data widthN layersN (hwFilter := true)
  unless Runtime.Autograd.Cuda.TexTable.filterHw t == true do
    throw <| IO.userError "textable hw: metadata mismatch"
  let (native, refv) ← runFetch t tab32 widthN layersN true probeCoords probeLayers
  -- Weight quantization bounds the error by 2⁻⁸ · max |Δ sample| (≈ 0.17 here), plus slack for
  -- the unspecified hardware weight rounding.
  assertTol "textable hw" native refv (tol := 2.0e-3)

/-- Integer-node fetches return stored samples bit-exactly in both modes (texel-center probe). -/
def runIntegerNodes : IO Unit := do
  IO.println "== textable integer nodes (bit-exact, both modes) =="
  let data := sampleData widthN layersN
  let mut coords : FloatArray := FloatArray.emptyWithCapacity (widthN * layersN)
  let mut lidx : FloatArray := FloatArray.emptyWithCapacity (widthN * layersN)
  for l in [0:layersN] do
    for i in [0:widthN] do
      coords := coords.push (Float.ofNat i)
      lidx := lidx.push (Float.ofNat l)
  for hw in [false, true] do
    let t := Runtime.Autograd.Cuda.TexTable.ofFloatArray data widthN layersN (hwFilter := hw)
    let out := Buffer.toFloatArray
      (Runtime.Autograd.Cuda.TexTable.fetch t (Buffer.ofFloatArray coords)
        (Buffer.ofFloatArray lidx))
    for i in [0:out.size] do
      let got := (out[i]!).toFloat32
      let want := (data[i]!).toFloat32
      unless got.toBits == want.toBits do
        throw <| IO.userError
          s!"textable integer node (hw={hw})[{i}]: got {got}, want stored sample {want}"

def runEdges : IO Unit := do
  IO.println "== textable edges (width=1, layer clamp, empty) =="
  -- width = 1: every coordinate hits the single sample.
  let one := FloatArray.mk #[0.75, -0.75]
  let t1 := Runtime.Autograd.Cuda.TexTable.ofFloatArray one 1 2 (hwFilter := false)
  let outs := Buffer.toFloatArray
    (Runtime.Autograd.Cuda.TexTable.fetch t1
      (Buffer.ofFloatArray (FloatArray.mk #[0.0, 3.5, -2.0]))
      (Buffer.ofFloatArray (FloatArray.mk #[0, 0, 1])))
  let expect : Array Float := #[0.75, 0.75, -0.75]
  for i in [0:outs.size] do
    unless ((outs[i]!).toFloat32).toBits == ((expect[i]!).toFloat32).toBits do
      throw <| IO.userError s!"textable width=1[{i}]: got {outs[i]!}, want {expect[i]!}"
  -- layer index out of range clamps to the last layer.
  let outc := Buffer.toFloatArray
    (Runtime.Autograd.Cuda.TexTable.fetch t1
      (Buffer.ofFloatArray (FloatArray.mk #[0.0]))
      (Buffer.ofFloatArray (FloatArray.mk #[7.0])))
  unless ((outc[0]!).toFloat32).toBits == ((-0.75 : Float).toFloat32).toBits do
    throw <| IO.userError s!"textable layer clamp: got {outc[0]!}, want -0.75"
  -- empty coordinate buffer yields an empty result.
  let oute := Buffer.toFloatArray
    (Runtime.Autograd.Cuda.TexTable.fetch t1
      (Buffer.ofFloatArray (FloatArray.mk #[]))
      (Buffer.ofFloatArray (FloatArray.mk #[])))
  unless oute.size == 0 do
    throw <| IO.userError s!"textable empty: expected empty result, got size {oute.size}"

/-- Unified TexTable test entrypoint. -/
def run : IO Unit := do
  runPointMode
  runHardwareMode
  runIntegerNodes
  runEdges

end TexTable
end Cuda
end Tests
