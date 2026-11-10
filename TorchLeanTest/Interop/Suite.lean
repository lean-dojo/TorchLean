/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLeanTest.Verification.CertificatePreconditions
public import TorchLeanTest.Floats.DirectedConversion
public import TorchLeanTest.Interop.IRPyTorch
public import TorchLeanTest.Interop.PyTorchRoundtrip
public import TorchLeanTest.Interop.RL
public import TorchLeanTest.CUDA.SessionRefIdentity
public import TorchLeanTest.Interop.ONNX

/-!
# Float runtime suite

`run` calls the float runtime checks against outside references: the readers of certificate JSON,
the native binary32 conversion, IR and round-trip parity with PyTorch, strict JSON integers at the
RL boundary, session references on CUDA and the ONNX bridge.
-/

@[expose] public section

namespace TorchLean.Test
namespace Floats

/-- Unified Float test entrypoint (called by `TorchLeanTest/Suite.lean`). -/
def run : IO Unit := do
  CertificatePreconditions.run
  DirectedConversion.run
  IRPyTorch.run
  PyTorchRoundtrip.run
  RL.run
  SessionRefIdentity.run
  ONNX.run

end Floats
end TorchLean.Test
