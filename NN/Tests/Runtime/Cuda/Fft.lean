/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.LibTorch.Ops
public import NN.Runtime.Autograd.Model.Functional.Spectral
public import NN.Runtime.Autograd.Torch.Core.BackwardOptim
public import NN.Runtime.Autograd.Torch.Core.Trainer.EagerOps
public import NN.Tests.Runtime.Cuda.Utils

/-!
# CUDA Kernel Coverage: Real FFT

Low-level coverage for the packed real FFT buffer primitives:

- `Buffer.rfft1dPacked`: `(batch, n)` real float32 rows to `(batch, n/2+1, 2)`,
- `Buffer.irfft1dPacked`: packed half-spectrum back to normalized real rows.

The CUDA backend calls LibTorch's `at::fft_rfft` and `at::fft_irfft`. Spectral convolution checks
exercise the public Lean composition, compare its values and pullbacks with the dense reference,
and check finite differences across odd/even grids, channel mixing, and empty retained spectra.
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace Fft

open Runtime.Autograd.LibTorch
open Runtime.Autograd
open Spec TorchLean

open Tests.Cuda.Utils (floatArray assertFloatArrayApprox)

def Internal.dotFloatArray (a b : FloatArray) : IO Float := do
  unless a.size == b.size do
    throw <| IO.userError "spectral finite-difference loss: output/seed length mismatch"
  let mut acc := 0.0
  for i in [:a.size] do
    acc := acc + a.get! i * b.get! i
  pure acc

def perturbArray (xs : Array Float) (i : Nat) (delta : Float) : Array Float :=
  match xs[i]? with
  | none => xs
  | some x => xs.setIfInBounds i (x + delta)

/-- Evaluate the public spectral composition and optionally all three seeded pullbacks. -/
def evaluateSpectralConv (grid width modes : Nat) (x wRe wIm dY : Array Float)
    (device : NN.Backend.Device := .cuda) (path : Model.F.SpectralPath := .automatic)
    (backward : Bool := true) : IO (FloatArray × FloatArray × FloatArray × FloatArray) := do
  if valid : 0 < grid ∧ 0 < width ∧ modes ≤ grid / 2 + 1 then
    unless x.size == grid * width && dY.size == grid * width &&
        wRe.size == modes * width * width && wIm.size == modes * width * width do
      throw <| IO.userError "spectral test input shape mismatch"
    let session ← Torch.Internal.EagerSession.new (α := Float) { device, execution := .eager }
    try
      let xRef ← session.input
        (Tensor.generateFlat [grid, width] fun i => x[i]!) (requiresGrad := backward)
      let realRef ← session.input
        (Tensor.generateFlat [modes, width, width] fun i => wRe[i]!) (requiresGrad := backward)
      let imagRef ← session.input
        (Tensor.generateFlat [modes, width, width] fun i => wIm[i]!) (requiresGrad := backward)
      let outputRef ← (Model.F.spectralConv (α := Float) (m := Torch.Internal.EagerM Float)
        valid.1 valid.2.1 valid.2.2 xRef realRef imagRef path) session
      let output := floatArray ((← session.getValue outputRef).to (Array Float))
      if device == .cuda then
        let tape ← session.cudaTape.get
        let forwardCount := (tape.nodes.filter fun node => node.name == some "rfft1d").size
        let inverseCount := (tape.nodes.filter fun node => node.name == some "irfft1d").size
        let expected := if path == .automatic then 1 else 0
        unless forwardCount == expected && inverseCount == expected do
          throw <| IO.userError "spectral composition selected incorrect FFT primitives"
      if !backward then return (output, FloatArray.empty, FloatArray.empty, FloatArray.empty)
      let gradients ← session.backwardDenseAll outputRef
        (Tensor.generateFlat [grid, width] fun i => dY[i]!)
      let dx ← Torch.Internal.EagerSession.grad gradients xRef
      let dReal ← Torch.Internal.EagerSession.grad gradients realRef
      let dImag ← Torch.Internal.EagerSession.grad gradients imagRef
      return (output, floatArray (dx.to (Array Float)),
        floatArray (dReal.to (Array Float)), floatArray (dImag.to (Array Float)))
    finally session.resetTape
  else
    throw <| IO.userError "spectral test dimensions must be positive and modes within the spectrum"

def spectralConvLoss
    (x wRe wIm dY : Array Float) (grid width modes : Nat) : IO Float := do
  let (y, _, _, _) ← evaluateSpectralConv grid width modes x wRe wIm dY (backward := false)
  Internal.dotFloatArray y (floatArray dY)

def assertFiniteDiff
    (msg : String) (analytic : FloatArray) (idx : Nat) (fd : Float) (tol : Float) : IO Unit := do
  Utils.assertApprox s!"{msg}[{idx}]" (analytic.get! idx) fd tol

def runKnownSpectrum : IO Unit := do
  IO.println "== rfft1d packed known spectra =="

  -- Two rows, n=4. The second row catches the sign convention:
  -- DFT([1,2,3,4]) at k=1 is `-2 + 2i` for the `exp(-2*pi*i*k*t/n)` convention used by cuFFT.
  let x := Buffer.ofFloatArray (floatArray #[
    1.0, 0.0, 0.0, 0.0,
    1.0, 2.0, 3.0, 4.0
  ])
  let got := Buffer.toFloatArray (Buffer.rfft1dPacked x 2 4)
  let expected := floatArray #[
    1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
    10.0, 0.0, -2.0, 2.0, -2.0, 0.0
  ]
  assertFloatArrayApprox "rfft1dPacked known" got expected (tol := 1e-4)

def runRoundtripEvenOdd : IO Unit := do
  IO.println "== rfft1d/irfft1d packed roundtrip =="

  -- Even and odd lengths exercise different Nyquist-bin handling. The adapter requests
  -- backward normalization from ATen's inverse, which includes the `1/n` factor.
  let even := floatArray #[
    0.25, -0.50, 1.00, 0.75, -1.25, 0.50, 0.125, -0.875,
    -0.30, 0.20, 0.90, -0.10, 0.45, -0.65, 1.10, -0.95
  ]
  let evenBuf := Buffer.ofFloatArray even
  let evenBack := Buffer.toFloatArray (Buffer.irfft1dPacked (Buffer.rfft1dPacked evenBuf 2 8) 2 8)
  assertFloatArrayApprox "rfft/irfft roundtrip even" evenBack even (tol := 2e-4)

  let odd := floatArray #[
    0.10, 0.30, -0.20, 0.70, -0.40,
    -0.60, 0.80, 0.15, -0.25, 0.55
  ]
  let oddBuf := Buffer.ofFloatArray odd
  let oddBack := Buffer.toFloatArray (Buffer.irfft1dPacked (Buffer.rfft1dPacked oddBuf 2 5) 2 5)
  assertFloatArrayApprox "rfft/irfft roundtrip odd" oddBack odd (tol := 2e-4)

def runSpectralConvIdentity : IO Unit := do
  IO.println "== spectral convolution identity/full-spectrum check =="

  -- With all retained RFFT bins and identity channel weights, the spectral convolution is
  -- exactly `irfft(rfft(x))`, so it should return the input up to float32/cuFFT roundoff.
  let x := #[
    0.25, -0.50,
    1.00, 0.75,
    -1.25, 0.50,
    0.125, -0.875
  ]
  let wRe := #[
    1.0, 0.0, 0.0, 1.0,
    1.0, 0.0, 0.0, 1.0,
    1.0, 0.0, 0.0, 1.0
  ]
  let wIm := Array.replicate 12 0.0
  let (got, _, _, _) ← evaluateSpectralConv 4 2 3 x wRe wIm x (backward := false)
  assertFloatArrayApprox "spectral convolution identity" got (floatArray x) (tol := 3e-4)

/-- The real FFT adjoint must neither amplify its spectrum into overflow nor normalize tiny
results to zero. Exact powers of two make both range boundaries independent of decimal rounding. -/
def runAdjointRange : IO Unit := do
  IO.println "== real FFT adjoint range =="
  let before ← Buffer.allocatorStats
  for dc in #[Float.ofNat (2 ^ 126), 1.0 / Float.ofNat (2 ^ 149)] do
    let gradient ← Buffer.ofFloatArrayIO <| floatArray #[dc, 0.0, 0.0, 0.0, 0.0, 0.0]
    let result ← IO.lazyPure fun _ => Buffer.rfft1dAdjoint gradient 1 4
    assertFloatArrayApprox "rfft adjoint preserves finite DC"
      (← Buffer.toFloatArrayIO result) (floatArray #[dc, dc, dc, dc]) (tol := 0.0)
    for buffer in #[gradient, result] do
      discard <| Buffer.releaseIO buffer
  -- Exercise the same large cotangent through the public spectral composition and Lean tape.
  let large := Float.ofNat (2 ^ 126)
  let (output, dx, _, _) ← evaluateSpectralConv 4 1 1
    #[0.0, 0.0, 0.0, 0.0] #[large] #[0.0] #[1.0, 1.0, 1.0, 1.0]
  assertFloatArrayApprox "spectral zero forward" output (floatArray #[0.0, 0.0, 0.0, 0.0]) 0.0
  assertFloatArrayApprox "spectral finite large dX" dx
    (floatArray #[large, large, large, large]) 0.0
  let after ← Buffer.allocatorStats
  unless after.liveBytes == before.liveBytes do
    throw <| IO.userError "FFT range checks retained tensor payloads"

def checkSpectralConvFiniteDiff (grid width modes : Nat)
    (x wRe wIm dY : Array Float) : IO Unit := do
  -- This validates the composed pullback against the scalar pairing
  --   L(x,w) = sum(spectralConv(x,w) * dY).
  -- The half-spectrum adjoint has subtle `2/n` factors for interior frequencies, so this test
  -- checks numerical gradients as well as native-versus-dense parity.
  let eps := 1e-2
  let tol := 2e-2

  let before ← Buffer.allocatorStats
  let (actual, dX, dWRe, dWIm) ← evaluateSpectralConv grid width modes x wRe wIm dY
  let after ← Buffer.allocatorStats
  unless after.liveBytes == before.liveBytes do
    throw <| IO.userError "spectral composition retained tensor payloads after session reset"
  unless dX.size == x.size && dWRe.size == wRe.size && dWIm.size == wIm.size do
    throw <| IO.userError "spectral backward returned incorrect gradient sizes"
  for device in [NN.Backend.Device.cpu, NN.Backend.Device.cuda] do
    let (expected, refDX, refDWRe, refDWIm) ←
      evaluateSpectralConv grid width modes x wRe wIm dY device .denseReference
    assertFloatArrayApprox "spectral forward dense parity" actual expected (tol := 3e-4)
    assertFloatArrayApprox "spectral dX dense parity" dX refDX (tol := 3e-4)
    assertFloatArrayApprox "spectral dWRe dense parity" dWRe refDWRe (tol := 3e-4)
    assertFloatArrayApprox "spectral dWIm dense parity" dWIm refDWIm (tol := 3e-4)

  for i in [:x.size] do
    let lp ← spectralConvLoss (perturbArray x i eps) wRe wIm dY grid width modes
    let lm ← spectralConvLoss (perturbArray x i (-eps)) wRe wIm dY grid width modes
    assertFiniteDiff "spectral convolution dX" dX i ((lp - lm) / (2.0 * eps)) tol

  for i in [:wRe.size] do
    let lp ← spectralConvLoss x (perturbArray wRe i eps) wIm dY grid width modes
    let lm ← spectralConvLoss x (perturbArray wRe i (-eps)) wIm dY grid width modes
    assertFiniteDiff "spectral convolution dWRe" dWRe i ((lp - lm) / (2.0 * eps)) tol

  for i in [:wIm.size] do
    let lp ← spectralConvLoss x wRe (perturbArray wIm i eps) dY grid width modes
    let lm ← spectralConvLoss x wRe (perturbArray wIm i (-eps)) dY grid width modes
    assertFiniteDiff "spectral convolution dWIm" dWIm i ((lp - lm) / (2.0 * eps)) tol

def runSpectralConvFiniteDiff : IO Unit := do
  IO.println "== spectral convolution backward finite differences and payload lifetime =="
  checkSpectralConvFiniteDiff 4 1 3
    #[0.20, -0.40, 0.70, 1.10] #[0.75, -0.30, 0.20]
    #[0.00, 0.45, 0.00] #[1.00, -0.50, 0.25, 0.75]
  -- Width two detects channel transposition; odd grids distinguish the final bin from Nyquist.
  for (grid, modes) in #[(1, 1), (4, 0), (4, 2), (4, 3), (5, 0), (5, 2), (5, 3)] do
    let values := fun (count phase : Nat) =>
      (Array.range count).map fun i => Float.ofNat ((i * 7 + phase) % 17) / 10.0 - 0.8
    IO.println s!"  grid={grid}, width=2, modes={modes}"
    checkSpectralConvFiniteDiff grid 2 modes
      (values (grid * 2) 1) (values (modes * 4) 3) (values (modes * 4) 5)
      (values (grid * 2) 9)

def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: real FFT ==="
  runKnownSpectrum
  runRoundtripEvenOdd
  runSpectralConvIdentity
  runAdjointRange
  runSpectralConvFiniteDiff

end Fft
end Cuda
end Tests
