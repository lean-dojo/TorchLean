/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Layered 1-D lookup-table textures (float32).

Implementation:
- CUDA: `csrc/cuda/textures/torchlean_cuda_textable.cu`
- CPU stub (default `lake build`): `csrc/cuda/textures/torchlean_cuda_textable_stub.c`
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Trusted
public import NN.Runtime.Autograd.Engine.Cuda.Buffer

@[expose] public section

namespace Runtime
namespace Autograd
namespace Cuda

namespace TexTable

/-!
# Lookup-table textures

A `TexTable` holds a small, immutable, *layered* 1-D table of float32 samples over a uniform
abscissa — tabulated transfer functions, calibration curves, and similar tabulated 1-D functions
whose analytic form is expensive or unavailable (common in signal processing and radar/optical
remote sensing). On the CUDA build it is backed by a layered `cudaArray` bound to a texture
object, so fetches go through the GPU's dedicated texture cache; the default build uses a
host-memory parity stub.

## Coordinate convention

`fetch` evaluates the table by piecewise-linear interpolation at *grid-space* coordinates
`u ∈ [0, width − 1]` (unnormalized; clamped at both ends), in the layer selected by an
integral-valued float index (clamped to `[0, layers − 1]`; layers are **not** interpolated
across). With `i = ⌊clamp(u, 0, width−1)⌋` and `f = u − i`, the result is
`table[layer][i] + f · (table[layer][i+1] − table[layer][i])` — the uniform-grid analogue of
`np.interp`. The native kernels add any texel-center offset internally; callers never add `0.5`.

## Filter modes (fixed at construction)

- `hwFilter := false` (**point mode**, default): two point fetches plus an explicit float32 lerp.
  Bit-reproducible — the CUDA build and the CPU stub return identical float32 results.
- `hwFilter := true` (**hardware mode**): the texture unit interpolates in hardware at zero ALU
  cost. CUDA specifies a 9-bit fixed-point lerp weight (8 fractional bits), so results carry a
  weight-quantization error of at most `2⁻⁸ · |table[i+1] − table[i]|` per fetch and must be
  compared by tolerance; the stub emulates the quantized weight.

Tables are forward-only values: fetches record no tape and define no gradient.
-/

/--
Build a table from `layers * width` samples in layer-major order (row `l` occupies
`data[l*width ..< (l+1)*width]`), narrowed to float32. Panics unless `width ≥ 1`, `layers ≥ 1`,
and `data.size = layers * width`.
-/
@[extern "torchlean_cuda_textable_make"]
opaque ofFloatArray (data : @& FloatArray) (width layers : @& Nat) (hwFilter : Bool := false) :
  TexTable

/-- Samples per layer. -/
@[extern "torchlean_cuda_textable_width"]
opaque width (t : @& TexTable) : Nat

/-- Number of layers. -/
@[extern "torchlean_cuda_textable_layers"]
opaque layers (t : @& TexTable) : Nat

/-- `true` iff the table was built with hardware filtering (`hwFilter := true`). -/
@[extern "torchlean_cuda_textable_filter_hw"]
opaque filterHw (t : @& TexTable) : Bool

/--
Evaluate the table at per-element grid coordinates `coords` (clamped to `[0, width − 1]`) in the
layers selected by `layerIdx` (integral-valued float32 indices, clamped to `[0, layers − 1]`).
`coords` and `layerIdx` must have equal sizes; the result is a fresh buffer of that size.
-/
@[extern "torchlean_cuda_textable_fetch"]
opaque fetch (t : @& TexTable) (coords layerIdx : @& Buffer) : Buffer

end TexTable

end Cuda
end Autograd
end Runtime
