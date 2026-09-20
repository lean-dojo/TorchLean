/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor

/-!
# Image Artifacts

Write normalized channel-first RGB tensors as dependency-free PPM files. Finite channel values are
clamped to [0,1] before conversion to bytes; nonfinite channels are rejected.
-/

@[expose] public section

namespace TorchLean.Data.Image

/-- Write the first RGB image as an ASCII PPM artifact, rejecting empty spatial dimensions. -/
def writeFirstRgbPpm {batch c h w : Nat}
    (path : System.FilePath)
    (x : Tensor Float [batch, c, h, w]) : IO Unit := do
  if batch = 0 then
    throw <| IO.userError "RGB PPM export requires a nonempty batch"
  if c < 3 then
    throw <| IO.userError "RGB PPM export requires at least 3 channels"
  if h = 0 || w = 0 then
    throw <| IO.userError "RGB PPM export requires positive image dimensions"
  if let some parent := path.parent then
    IO.FS.createDirAll parent
  let clamp01 (v : Float) : Float :=
    if v < 0.0 then 0.0 else if v > 1.0 then 1.0 else v
  let toByte (v : Float) : Nat :=
    let v01 := clamp01 v
    Nat.min 255 ((v01 * 255.0).toUInt64.toNat)
  let hOut ← IO.FS.Handle.mk path IO.FS.Mode.write
  hOut.putStr s!"P3\n{w} {h}\n255\n"
  let getPx (channel row column : Nat) : IO Float :=
    match x.at? #[0, channel, row, column] with
    | some value =>
        if value.isFinite then pure value
        else throw <| IO.userError "RGB PPM export requires finite channel values"
    | none =>
        throw <| IO.userError
          s!"RGB PPM export encountered invalid coordinate [0,{channel},{row},{column}]"
  for row in [0:h] do
    for column in [0:w] do
      let red ← getPx 0 row column
      let green ← getPx 1 row column
      let blue ← getPx 2 row column
      hOut.putStr
        s!"{toByte red} {toByte green} {toByte blue}\n"

end TorchLean.Data.Image
