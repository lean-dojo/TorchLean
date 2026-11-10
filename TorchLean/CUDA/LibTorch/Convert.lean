/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Datatype

/-!
# Boolean masks for the LibTorch bridge

The LibTorch bridge reads a buffer in row-major order, the last axis fastest. A `Float` tensor
already is such a buffer, which `Tensor.to tensor FloatArray` exposes and `Tensor.reshape?` reads
back at a shape. A Boolean mask is not, so `Convert.flattenBoolMask` writes it with `1.0` for `true`
and `0.0` for `false`.
-/

@[expose] public section

open TorchLean.Shape (size)

namespace TorchLean
namespace LibTorch

namespace Convert

/-- Flatten a `TorchLean.Tensor Bool s` mask into a row-major `FloatArray`, with `1.0` for `true`
and `0.0` for `false`. -/
def flattenBoolMask {s : Shape} (mask : Tensor Bool s) : FloatArray :=
  Tensor.foldl (fun acc value => acc.push (if value then 1.0 else 0.0))
    (FloatArray.emptyWithCapacity (Shape.size s)) mask

end Convert

end LibTorch
end TorchLean
