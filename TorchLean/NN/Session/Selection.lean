/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

-- Downstream modules reach the marked imports through this module. Import shaking cannot see
-- those downstream uses, so keep the marked imports.
module -- shake: keep-downstream

public import TorchLean.NN.Backend.Report -- shake: keep
public import TorchLean.NN.Program -- shake: keep
public import TorchLean.NN.Arithmetic -- shake: keep
public import TorchLean.NN.TensorTransfer -- shake: keep
public import TorchLean.NN.Session.Config

/-!
# Runtime Selection

The public names for choosing and inspecting how a program runs. `TorchLean.Session.Config` holds
the device, seed and backend profile, `Backend.Device.parseSelector` reads a device selector, and
`Session.Config.planReport` reports the backend-contract plan for a list of operators.
The element arithmetic comes from the imported `Arithmetic`.
-/

@[expose] public section

namespace TorchLean

/-!
## Devices and backend contracts
-/

namespace Backend.Device

/--
Parse a device selector such as `cpu`, `cuda` or `gpu`. `auto` selects the portable CPU runtime,
and any other value must name one of the `Backend.Device` targets.
-/
def parseSelector (value : String) : Except String Device :=
  if value == "auto" then pure .cpu else Backend.Device.parse value

end Backend.Device

namespace Session.Config

/--
Plan `ops` under the backend-contract profile that `config` selects and return the report as
text. Pass `gradEnabled := false` to plan for inference, where no operation needs a VJP.
-/
def planReport (config : Config) (ops : Array (Σ ins outs, TorchLean.Op ins outs))
    (gradEnabled : Bool := true) : Except String String := do
  let profile ← config.effectiveBackendProfile gradEnabled
  profile.planReport ops

end Session.Config

end TorchLean
