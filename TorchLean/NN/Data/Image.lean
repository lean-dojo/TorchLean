/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Operations

/-!
# Image Artifacts

Write normalized channel-first RGB tensors as dependency-free PPM files. Finite channel values are
clamped to [0,1] and rounded to the nearest byte, and a nonfinite value is rejected before the file
is touched.
-/

@[expose] public section

namespace TorchLean.Data.Image

/--
Write the first image of the batch (channels 0, 1 and 2) as an ASCII PPM file. An empty batch,
fewer than three channels or an empty spatial dimension raises an error.
-/
def writeFirstPPM {batch c h w : Nat}
    (path : System.FilePath)
    (x : Tensor Float [batch, c, h, w]) : IO Unit := do
  if batch = 0 then
    throw <| IO.userError "RGB PPM export requires a nonempty batch"
  if c < 3 then
    throw <| IO.userError "RGB PPM export requires at least 3 channels"
  if h = 0 || w = 0 then
    throw <| IO.userError "RGB PPM export requires positive image dimensions"
  let clamp01 (v : Float) : Float :=
    if v < 0.0 then 0.0 else if v > 1.0 then 1.0 else v
  -- One correctly rounded product, then C `round` (ties away from zero) to the nearest byte.
  let toByte (v : Float) : UInt8 := (clamp01 v * 255.0).round.toUInt8
  let getPx (channel row column : Nat) : Except String Float :=
    match x.at? #[0, channel, row, column] with
    | some value =>
        if value.isFinite then pure value
        else throw "RGB PPM export requires finite channel values"
    | none =>
        throw s!"RGB PPM export encountered invalid coordinate [0,{channel},{row},{column}]"
  let body : Except String String := do
    let mut text := s!"P3\n{w} {h}\n255\n"
    for row in [0:h] do
      for column in [0:w] do
        let red ← getPx 0 row column
        let green ← getPx 1 row column
        let blue ← getPx 2 row column
        text := text ++ s!"{toByte red} {toByte green} {toByte blue}\n"
    pure text
  match body with
  | .error message => throw <| IO.userError message
  | .ok text =>
      if let some parent := path.parent then
        IO.FS.createDirAll parent
      IO.FS.writeFile path text

end TorchLean.Data.Image
