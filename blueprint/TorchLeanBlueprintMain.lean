import VersoManual
import VersoBlueprint.PreviewManifest
import TorchLeanBlueprint.Guide

open Verso Doc
open Verso.Genre Manual

def main (args : List String) : IO UInt32 :=
  Informal.PreviewManifest.manualMainWithSharedPreviewManifest
    (%doc TorchLeanBlueprint.Guide)
    args
    (extensionImpls := by exact extension_impls%)
