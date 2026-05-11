/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Conv
public import NN.Spec.Module.SpecModule

/-!
# Convolution module wrappers

This file exposes conv specs as `NNModuleSpec`s.

Compatibility wrappers for `Conv2D` and `ConvTranspose2D` are consolidated here with their public
names.
-/

@[expose] public section

namespace Spec

open Tensor
open ModSpec

/-!
## Conv2D
-/

/-- Wrap `conv2d_spec` as an `NNModuleSpec`, with the output shape computed in the type. -/
def Conv2DModuleSpec {α : Type} [Context α] {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (m : Conv2DSpec inC outC kH kW stride padding α h1 h2 h3) :
  NNModuleSpec α
    (.dim inC (.dim inH (.dim inW .scalar)))
    (.dim outC (.dim ((inH + 2 * padding - kH) / stride + 1) (.dim ((inW + 2 * padding - kW) /
      stride + 1) .scalar))) :=
{ forward := fun x => conv2dSpec m x, kind := "Conv2D", export_func := {
  toPyTorch :=
    s!"nn.Conv2d({inC}, {outC}, kernel_size=({kH}, {kW}), stride={stride}, padding={padding})",
  dimensions := (inC, outC)
} }

/-!
## ConvTranspose2D
-/

/-- ConvTranspose2D wrapper as an `NNModuleSpec` (output shape encoded at the type level). -/
def ConvTranspose2DModuleSpec {α : Type} [Context α]
  {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC > 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (m : ConvTranspose2DSpec inC outC kH kW stride padding α h1 h2 h3) :
  NNModuleSpec α
    (.dim inC (.dim inH (.dim inW .scalar)))
    (.dim outC
      (.dim ((inH - 1) * stride - 2 * padding + kH)
        (.dim ((inW - 1) * stride - 2 * padding + kW) .scalar))) :=
{ forward := fun x => convTranspose2dSpec (inC := inC) (outC := outC)
    (kH := kH) (kW := kW) (stride := stride) (padding := padding)
    (inH := inH) (inW := inW) m x
  kind := "ConvTranspose2D"
  export_func := {
    toPyTorch :=
      s!"nn.ConvTranspose2d({inC}, {outC}, kernel_size=({kH}, {kW}), " ++
        s!"stride={stride}, padding={padding})"
    dimensions := (inC, outC)
  } }

end Spec
