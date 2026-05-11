/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Core
public import NN.Runtime.Autograd.Engine.Cuda.Ops
public import NN.Entrypoint.Tensor
public import NN.Tests.Runtime.Cuda.Utils
import Batteries.Data.Vector.Lemmas

/-!
# CUDA Kernel Coverage: Conv2D + Pooling

Compares CPU eager tape vs CUDA eager tape for:
- `conv2d`
- `max_pool2d`
- `smooth_max_pool2d`
- `avg_pool2d`

All cases are single-image, channels-first, small shapes.
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace ConvPool

open Spec
open Tensor
open Runtime.Autograd

abbrev inC : Nat := 1
abbrev outC : Nat := 1
abbrev kH : Nat := 2
abbrev kW : Nat := 2
abbrev stride : Nat := 1
abbrev padding : Nat := 0
abbrev inH : Nat := 3
abbrev inW : Nat := 3

def hInC : inC ≠ 0 := by decide
def hKH : kH ≠ 0 := by decide
def hKW : kW ≠ 0 := by decide

def outH : Nat := (inH + 2 * padding - kH) / stride + 1
def outW : Nat := (inW + 2 * padding - kW) / stride + 1

def kernel : Tensor Float (shape![outC, inC, kH, kW]) :=
  tensorND! [outC, inC, kH, kW] [0.2, -0.1, 0.3, 0.4]

def bias : Tensor Float (shape![outC]) :=
  tensorND! [outC] [0.05]

def input : Tensor Float (shape![inC, inH, inW]) :=
  tensorND! [inC, inH, inW] [
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0
  ]

/-!
## N-D smoke cases (d = 3)

These exercise the new "ND" ConvPool CUDA entrypoints (`conv`/`max_pool`/`avg_pool`/`smooth_max_pool`)
which accept per-axis parameters.
-/

abbrev d3 : Nat := 3
abbrev inD0 : Nat := 3
abbrev inD1 : Nat := 3
abbrev inD2 : Nat := 3

abbrev k0 : Nat := 2
abbrev k1 : Nat := 2
abbrev k2 : Nat := 2

def inSpatial3 : Vector Nat d3 :=
  #v[inD0, inD1, inD2]

def kernel3V : Vector Nat d3 :=
  #v[k0, k1, k2]

def stride3V : Vector Nat d3 :=
  #v[1, 1, 1]

def padding3V : Vector Nat d3 :=
  #v[0, 0, 0]

def hKernel3V : ∀ i : Fin d3, kernel3V.get i ≠ 0 := by
  intro i
  fin_cases i <;> simp [kernel3V, Vector.get]

def outSpatial3 : Vector Nat d3 :=
  Spec.convOutSpatial inSpatial3 kernel3V stride3V padding3V

def outShape3 : Shape :=
  Shape.ofList (outC :: outSpatial3.toList)

def kernel3 : Tensor Float (Shape.ofList (outC :: inC :: kernel3V.toList)) :=
  tensorND! [outC, inC, k0, k1, k2] [
    0.2, -0.1,
    0.3, 0.4,
    -0.25, 0.15,
    0.05, -0.35
  ]

def input3 : Tensor Float (Shape.ofList (inC :: inSpatial3.toList)) :=
  tensorND! [inC, inD0, inD1, inD2] [
    1.0,  2.0,  3.0,
    4.0,  5.0,  6.0,
    7.0,  8.0,  9.0,

    10.0, 11.0, 12.0,
    13.0, 14.0, 15.0,
    16.0, 17.0, 18.0,

    19.0, 20.0, 21.0,
    22.0, 23.0, 24.0,
    25.0, 26.0, 27.0
  ]

def runConv3 : IO Unit := do
  IO.println "== conv (d=3) =="

  -- CPU tape
  let t0 : Tape Float := Tape.empty
  let (t1, kId) := Tape.leaf (t := t0) kernel3 (name := some "kernel")
  let (t2, bId) := Tape.leaf (t := t1) bias (name := some "bias")
  let (t3, xId) := Tape.leaf (t := t2) input3 (name := some "input")
  let (t4, yId) ← Utils.okOrThrow
    (Tape.conv (α := Float) (t := t3)
      (d := d3) (inC := inC) (outC := outC)
      (kernel := kernel3V) (stride := stride3V) (padding := padding3V)
      (inSpatial := inSpatial3)
      kId bId xId (name := "conv[d=3]"))
  let yCpu ← Utils.cpuValue (s := outShape3) t4 yId
  let seedCpu : Runtime.AnyTensor Float := AnyTensor.mk (fill (1.0 : Float) outShape3)
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t4) yId seedCpu)
  let dKCpu ← Utils.cpuGrad (s := Shape.ofList (outC :: inC :: kernel3V.toList)) gradsCpu kId
  let dBCpu ← Utils.cpuGrad (s := shape![outC]) gradsCpu bId
  let dXCpu ← Utils.cpuGrad (s := Shape.ofList (inC :: inSpatial3.toList)) gradsCpu xId

  -- CUDA tape
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, kIdc) :=
    Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer kernel3) (name := some "kernel")
  let (t2c, bIdc) :=
    Runtime.Autograd.Cuda.Tape.leaf (t := t1c) (Utils.tensorToAnyBuffer bias) (name := some "bias")
  let (t3c, xIdc) :=
    Runtime.Autograd.Cuda.Tape.leaf (t := t2c) (Utils.tensorToAnyBuffer input3) (name := some "input")
  let (t4c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.conv (t := t3c)
      (d := d3) (inC := inC) (outC := outC)
      (kernel := kernel3V) (stride := stride3V) (padding := padding3V)
      (inSpatial := inSpatial3)
      kIdc bIdc xIdc (hInC := hInC) (hKernel := hKernel3V))
  let yCuda ← Utils.cudaValue (s := outShape3) t4c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := outShape3
      buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Shape.size outShape3)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t4c) yIdc seedCuda)
  let dKCuda ← Utils.cudaGrad (s := Shape.ofList (outC :: inC :: kernel3V.toList)) gradsCuda kIdc
  let dBCuda ← Utils.cudaGrad (s := shape![outC]) gradsCuda bIdc
  let dXCuda ← Utils.cudaGrad (s := Shape.ofList (inC :: inSpatial3.toList)) gradsCuda xIdc

  Utils.assertTensorApprox (s := outShape3) "conv[d=3] forward" yCuda yCpu (tol := 1e-2)
  Utils.assertTensorApprox (s := Shape.ofList (outC :: inC :: kernel3V.toList))
    "conv[d=3] dKernel" dKCuda dKCpu (tol := 1e-2)
  Utils.assertTensorApprox (s := shape![outC]) "conv[d=3] dBias" dBCuda dBCpu (tol := 1e-2)
  Utils.assertTensorApprox (s := Shape.ofList (inC :: inSpatial3.toList))
    "conv[d=3] dInput" dXCuda dXCpu (tol := 1e-2)

def runMaxPool3 : IO Unit := do
  IO.println "== max_pool (d=3) =="

  let outSpatial3 := Spec.poolOutSpatialPad inSpatial3 kernel3V stride3V padding3V
  let yShape : Shape := Shape.ofList (inC :: outSpatial3.toList)

  -- CPU
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) input3 (name := some "input")
  let (t2, yId) ← Utils.okOrThrow
    (Tape.maxPool (α := Float) (t := t1)
      (d := d3) (C := inC)
      (inSpatial := inSpatial3) (kernel := kernel3V) (stride := stride3V) (padding := padding3V)
      (hKernel := hKernel3V) xId)
  let yCpu ← Utils.cpuValue (s := yShape) t2 yId
  let seedCpu : Runtime.AnyTensor Float := AnyTensor.mk (fill (1.0 : Float) yShape)
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := Shape.ofList (inC :: inSpatial3.toList)) gradsCpu xId

  -- CUDA
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer input3)
    (name := some "input")
  let (t2c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.maxPool (t := t1c)
      (d := d3) (C := inC)
      (inSpatial := inSpatial3) (kernel := kernel3V) (stride := stride3V) (padding := padding3V)
      (hKernel := hKernel3V) xIdc)
  let yCuda ← Utils.cudaValue (s := yShape) t2c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := yShape
      buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Shape.size yShape)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t2c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := Shape.ofList (inC :: inSpatial3.toList)) gradsCuda xIdc

  Utils.assertTensorApprox (s := yShape) "max_pool[d=3] forward" yCuda yCpu (tol := 1e-6)
  Utils.assertTensorApprox (s := Shape.ofList (inC :: inSpatial3.toList))
    "max_pool[d=3] dx" dxCuda dxCpu (tol := 1e-6)

def runSmoothMaxPool3 : IO Unit := do
  IO.println "== smooth_max_pool (d=3) =="

  let outSpatial3 := Spec.poolOutSpatialPad inSpatial3 kernel3V stride3V padding3V
  let yShape : Shape := Shape.ofList (inC :: outSpatial3.toList)
  let beta : Float := 0.5

  -- CPU
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) input3 (name := some "input")
  let (t2, yId) ← Utils.okOrThrow
    (Tape.smoothMaxPool (α := Float) (t := t1)
      (d := d3) (C := inC)
      (inSpatial := inSpatial3) (kernel := kernel3V) (stride := stride3V) (padding := padding3V)
      (hKernel := hKernel3V) xId beta)
  let yCpu ← Utils.cpuValue (s := yShape) t2 yId
  let seedCpu : Runtime.AnyTensor Float := AnyTensor.mk (fill (1.0 : Float) yShape)
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := Shape.ofList (inC :: inSpatial3.toList)) gradsCpu xId

  -- CUDA
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer input3)
    (name := some "input")
  let (t2c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.smoothMaxPool (t := t1c)
      (d := d3) (C := inC)
      (inSpatial := inSpatial3) (kernel := kernel3V) (stride := stride3V) (padding := padding3V)
      (hKernel := hKernel3V) xIdc beta)
  let yCuda ← Utils.cudaValue (s := yShape) t2c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := yShape
      buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Shape.size yShape)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t2c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := Shape.ofList (inC :: inSpatial3.toList)) gradsCuda xIdc

  Utils.assertTensorApprox (s := yShape) "smooth_max_pool[d=3] forward" yCuda yCpu (tol := 1e-2)
  Utils.assertTensorApprox (s := Shape.ofList (inC :: inSpatial3.toList))
    "smooth_max_pool[d=3] dx" dxCuda dxCpu (tol := 1e-2)

def runAvgPool3 : IO Unit := do
  IO.println "== avg_pool (d=3) =="

  let outSpatial3 := Spec.poolOutSpatialPad inSpatial3 kernel3V stride3V padding3V
  let yShape : Shape := Shape.ofList (inC :: outSpatial3.toList)

  -- CPU
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) input3 (name := some "input")
  let (t2, yId) ← Utils.okOrThrow
    (Tape.avgPool (α := Float) (t := t1)
      (d := d3) (C := inC)
      (inSpatial := inSpatial3) (kernel := kernel3V) (stride := stride3V) (padding := padding3V)
      (hKernel := hKernel3V) xId)
  let yCpu ← Utils.cpuValue (s := yShape) t2 yId
  let seedCpu : Runtime.AnyTensor Float := AnyTensor.mk (fill (1.0 : Float) yShape)
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := Shape.ofList (inC :: inSpatial3.toList)) gradsCpu xId

  -- CUDA
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer input3)
    (name := some "input")
  let (t2c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.avgPool (t := t1c)
      (d := d3) (C := inC)
      (inSpatial := inSpatial3) (kernel := kernel3V) (stride := stride3V) (padding := padding3V)
      (hKernel := hKernel3V) xIdc)
  let yCuda ← Utils.cudaValue (s := yShape) t2c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := yShape
      buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Shape.size yShape)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t2c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := Shape.ofList (inC :: inSpatial3.toList)) gradsCuda xIdc

  Utils.assertTensorApprox (s := yShape) "avg_pool[d=3] forward" yCuda yCpu (tol := 1e-2)
  Utils.assertTensorApprox (s := Shape.ofList (inC :: inSpatial3.toList))
    "avg_pool[d=3] dx" dxCuda dxCpu (tol := 1e-2)

def runConv2d : IO Unit := do
  IO.println "== conv2d =="

  let yShape : Shape := shape![outC, outH, outW]

  -- CPU tape
  let t0 : Tape Float := Tape.empty
  let (t1, kId) := Tape.leaf (t := t0) kernel (name := some "kernel")
  let (t2, bId) := Tape.leaf (t := t1) bias (name := some "bias")
  let (t3, xId) := Tape.leaf (t := t2) input (name := some "input")
  let (t4, yId) ← Utils.okOrThrow
    (Tape.conv2d (α := Float) (t := t3)
      (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
      (inH := inH) (inW := inW) (h1 := hInC) (h2 := hKH) (h3 := hKW)
      kId bId xId)
  let yCpu ← Utils.cpuValue (s := yShape) t4 yId
  let seedCpu : Runtime.AnyTensor Float := AnyTensor.mk (fill (1.0 : Float) yShape)
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t4) yId seedCpu)
  let dKCpu ← Utils.cpuGrad (s := shape![outC, inC, kH, kW]) gradsCpu kId
  let dBCpu ← Utils.cpuGrad (s := shape![outC]) gradsCpu bId
  let dXCpu ← Utils.cpuGrad (s := shape![inC, inH, inW]) gradsCpu xId

  -- CUDA tape
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, kIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer kernel)
    (name := some "kernel")
  let (t2c, bIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t1c) (Utils.tensorToAnyBuffer bias)
    (name := some "bias")
  let (t3c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t2c) (Utils.tensorToAnyBuffer input)
    (name := some "input")
  let (t4c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.conv2d (t := t3c)
      (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
      (inH := inH) (inW := inW) (h1 := hInC) (h2 := hKH) (h3 := hKW)
      kIdc bIdc xIdc)
  let yCuda ← Utils.cudaValue (s := yShape) t4c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := yShape, buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Shape.size yShape)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t4c) yIdc seedCuda)
  let dKCuda ← Utils.cudaGrad (s := shape![outC, inC, kH, kW]) gradsCuda kIdc
  let dBCuda ← Utils.cudaGrad (s := shape![outC]) gradsCuda bIdc
  let dXCuda ← Utils.cudaGrad (s := shape![inC, inH, inW]) gradsCuda xIdc

  Utils.assertTensorApprox (s := yShape) "conv2d forward" yCuda yCpu (tol := 5e-3)
  Utils.assertTensorApprox (s := shape![outC, inC, kH, kW])
    "conv2d dKernel" dKCuda dKCpu (tol := 5e-3)
  Utils.assertTensorApprox (s := shape![outC]) "conv2d dBias" dBCuda dBCpu (tol := 5e-3)
  Utils.assertTensorApprox (s := shape![inC, inH, inW])
    "conv2d dInput" dXCuda dXCpu (tol := 5e-3)

def runMaxPool : IO Unit := do
  IO.println "== max_pool2d =="
  let yShape : Shape := shape![inC, outH, outW]

  -- CPU
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) input (name := some "input")
  let (t2, yId) ← Utils.okOrThrow
    (Tape.maxPool2d (α := Float) (t := t1)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      (h1 := hKH) (h2 := hKW) xId)
  let yCpu ← Utils.cpuValue (s := yShape) t2 yId
  let seedCpu : Runtime.AnyTensor Float := AnyTensor.mk (fill (1.0 : Float) yShape)
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := shape![inC, inH, inW]) gradsCpu xId

  -- CUDA
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer input)
    (name := some "input")
  let (t2c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.maxPool2d (t := t1c)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      (h1 := hKH) (h2 := hKW) xIdc)
  let yCuda ← Utils.cudaValue (s := yShape) t2c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := yShape, buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Shape.size yShape)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t2c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := shape![inC, inH, inW]) gradsCuda xIdc

  Utils.assertTensorApprox (s := yShape) "max_pool2d forward" yCuda yCpu (tol := 1e-6)
  Utils.assertTensorApprox (s := shape![inC, inH, inW]) "max_pool2d dx" dxCuda dxCpu (tol := 1e-6)

def runMaxPoolPadNegative : IO Unit := do
  IO.println "== max_pool2d padding negative inputs =="

  let x : Tensor Float (shape![1, 1, 1]) :=
    tensorND! [1, 1, 1] [-3.0]
  let yShape : Shape := shape![1, 2, 2]
  let expectedY : Tensor Float (shape![1, 2, 2]) :=
    tensorND! [1, 2, 2] [-3.0, -3.0, -3.0, -3.0]
  let expectedDx : Tensor Float (shape![1, 1, 1]) :=
    tensorND! [1, 1, 1] [4.0]

  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) x (name := some "input")
  let (t2, yId) ← Utils.okOrThrow
    (Tape.maxPool2dPad (α := Float) (t := t1)
      (kH := 2) (kW := 2) (inH := 1) (inW := 1) (inC := 1) (stride := 1) (padding := 1)
      (h1 := by decide) (h2 := by decide) xId)
  let yCpu ← Utils.cpuValue (s := yShape) t2 yId
  let seedCpu : Runtime.AnyTensor Float := AnyTensor.mk (fill (1.0 : Float) yShape)
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := shape![1, 1, 1]) gradsCpu xId

  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer x)
    (name := some "input")
  let (t2c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.maxPool2dPad (t := t1c)
      (kH := 2) (kW := 2) (inH := 1) (inW := 1) (inC := 1) (stride := 1) (padding := 1)
      (h1 := by decide) (h2 := by decide) xIdc)
  let yCuda ← Utils.cudaValue (s := yShape) t2c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := yShape, buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Shape.size yShape)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t2c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := shape![1, 1, 1]) gradsCuda xIdc

  Utils.assertTensorApprox (s := yShape) "max_pool2d_pad negative CPU expected" yCpu expectedY (tol := 1e-6)
  Utils.assertTensorApprox (s := yShape) "max_pool2d_pad negative CUDA expected" yCuda expectedY (tol := 1e-6)
  Utils.assertTensorApprox (s := shape![1, 1, 1])
    "max_pool2d_pad negative CPU dx" dxCpu expectedDx (tol := 1e-6)
  Utils.assertTensorApprox (s := shape![1, 1, 1])
    "max_pool2d_pad negative CUDA dx" dxCuda expectedDx (tol := 1e-6)

def runMaxPool3PadNegative : IO Unit := do
  IO.println "== max_pool (d=3) padding negative inputs =="

  let inSpatial : Vector Nat 3 := #v[1, 1, 1]
  let kernel : Vector Nat 3 := #v[2, 2, 2]
  let stride : Vector Nat 3 := #v[1, 1, 1]
  let padding : Vector Nat 3 := #v[1, 1, 1]
  let hKernel : ∀ i : Fin 3, kernel.get i ≠ 0 := by
    intro i
    fin_cases i <;> simp [kernel, Vector.get]
  let yShape : Shape := Shape.ofList [1, 2, 2, 2]
  let x : Tensor Float (Shape.ofList [1, 1, 1, 1]) :=
    tensorND! [1, 1, 1, 1] [-3.0]
  let expectedY : Tensor Float (Shape.ofList [1, 2, 2, 2]) :=
    tensorND! [1, 2, 2, 2] [-3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0]
  let expectedDx : Tensor Float (Shape.ofList [1, 1, 1, 1]) :=
    tensorND! [1, 1, 1, 1] [8.0]

  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) x (name := some "input")
  let (t2, yId) ← Utils.okOrThrow
    (Tape.maxPool (α := Float) (t := t1)
      (d := 3) (C := 1)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      (hKernel := hKernel) xId)
  let yCpu ← Utils.cpuValue (s := yShape) t2 yId
  let seedCpu : Runtime.AnyTensor Float := AnyTensor.mk (fill (1.0 : Float) yShape)
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := Shape.ofList [1, 1, 1, 1]) gradsCpu xId

  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer x)
    (name := some "input")
  let (t2c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.maxPool (t := t1c)
      (d := 3) (C := 1)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      (hKernel := hKernel) xIdc)
  let yCuda ← Utils.cudaValue (s := yShape) t2c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := yShape, buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Shape.size yShape)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t2c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := Shape.ofList [1, 1, 1, 1]) gradsCuda xIdc

  Utils.assertTensorApprox (s := yShape) "max_pool[d=3] pad negative CPU expected" yCpu expectedY
    (tol := 1e-6)
  Utils.assertTensorApprox (s := yShape) "max_pool[d=3] pad negative CUDA expected" yCuda expectedY
    (tol := 1e-6)
  Utils.assertTensorApprox (s := Shape.ofList [1, 1, 1, 1])
    "max_pool[d=3] pad negative CPU dx" dxCpu expectedDx (tol := 1e-6)
  Utils.assertTensorApprox (s := Shape.ofList [1, 1, 1, 1])
    "max_pool[d=3] pad negative CUDA dx" dxCuda expectedDx (tol := 1e-6)

def runSmoothMaxPool : IO Unit := do
  IO.println "== smooth_max_pool2d =="
  let yShape : Shape := shape![inC, outH, outW]
  let beta : Float := 0.5

  -- CPU
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) input (name := some "input")
  let (t2, yId) ← Utils.okOrThrow
    (Tape.smoothMaxPool2d (α := Float) (t := t1)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      (h1 := hKH) (h2 := hKW) xId beta)
  let yCpu ← Utils.cpuValue (s := yShape) t2 yId
  let seedCpu : Runtime.AnyTensor Float := AnyTensor.mk (fill (1.0 : Float) yShape)
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := shape![inC, inH, inW]) gradsCpu xId

  -- CUDA
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer input)
    (name := some "input")
  let (t2c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.smoothMaxPool2d (t := t1c)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      (h1 := hKH) (h2 := hKW) xIdc beta)
  let yCuda ← Utils.cudaValue (s := yShape) t2c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := yShape, buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Shape.size yShape)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t2c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := shape![inC, inH, inW]) gradsCuda xIdc

  Utils.assertTensorApprox (s := yShape) "smooth_max_pool2d forward" yCuda yCpu (tol := 5e-3)
  Utils.assertTensorApprox (s := shape![inC, inH, inW])
    "smooth_max_pool2d dx" dxCuda dxCpu (tol := 5e-3)

def runAvgPool : IO Unit := do
  IO.println "== avg_pool2d =="
  let yShape : Shape := shape![inC, outH, outW]

  -- CPU
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) input (name := some "input")
  let (t2, yId) ← Utils.okOrThrow
    (Tape.avgPool2d (α := Float) (t := t1)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      (h1 := hKH) (h2 := hKW) xId)
  let yCpu ← Utils.cpuValue (s := yShape) t2 yId
  let seedCpu : Runtime.AnyTensor Float := AnyTensor.mk (fill (1.0 : Float) yShape)
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := shape![inC, inH, inW]) gradsCpu xId

  -- CUDA
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer input)
    (name := some "input")
  let (t2c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.avgPool2d (t := t1c)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      (h1 := hKH) (h2 := hKW) xIdc)
  let yCuda ← Utils.cudaValue (s := yShape) t2c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := yShape, buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Shape.size yShape)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t2c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := shape![inC, inH, inW]) gradsCuda xIdc

  Utils.assertTensorApprox (s := yShape) "avg_pool2d forward" yCuda yCpu (tol := 5e-3)
  Utils.assertTensorApprox (s := shape![inC, inH, inW]) "avg_pool2d dx" dxCuda dxCpu (tol := 5e-3)

def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: conv2d + pooling ==="
  runConv2d
  runConv3
  runMaxPool
  runMaxPoolPadNegative
  runMaxPool3
  runMaxPool3PadNegative
  runSmoothMaxPool
  runSmoothMaxPool3
  runAvgPool
  runAvgPool3

end ConvPool
end Cuda
end Tests
