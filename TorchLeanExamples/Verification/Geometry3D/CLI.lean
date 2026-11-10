/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLeanExamples.Verification.Geometry3D.Box3D
public import TorchLean.NN.CLI.Parser

/-!
# Geometry3D CLI wrapper

`camera-box3d-cert` is the runnable entrypoint of the camera-box verifier in
`Examples.Geometry3D`, kept next to it. Real-world artifacts come from the
`export_*_box3d_cert.py` scripts in this directory. The default path is a fixed sample
certificate, so the CLI runs without network access, a GPU or model downloads.
-/

@[expose] public section

namespace TorchLean.Examples.Geometry3D

/-- Default sample certificate used when no path is passed to `scripts/lake.sh exe verify`. -/
def defaultCertPath : String :=
  "TorchLeanExamples/Verification/Geometry3D/check_box3d_camera_cert.json"

/--
CLI entrypoint used by the unified verification dispatcher. The certificate path comes from
`--cert` or the first positional argument, defaulting to `defaultCertPath`, and any other argument
is an error.
-/
def main (args : List String) : IO Unit := do
  let args := CLI.normalizePathFlag args "cert" defaultCertPath
  let (path, args) ←
    match CLI.takeFlagValue args "cert" (default := defaultCertPath) with
    | .ok result => pure result
    | .error e => throw <| IO.userError e
  IO.ofExcept <| CLI.checkNoArgs args
  checkOrThrow path

end TorchLean.Examples.Geometry3D
