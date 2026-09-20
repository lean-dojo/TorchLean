import VersoManual
import VersoBlueprint.PreviewManifest
import TorchLeanBlueprint.Guide

open Verso Doc
open Verso.Genre Manual

/-- Keep numbered navigation at chapter and page level; read the rest within the page. -/
private partial def readingLayout (part : Part Manual) (depth : Nat := 0) : Part Manual :=
  let metadata := part.metadata.getD {}
  -- The book is depth 0, chapters are depth 1, and pages such as 2.1 are depth 2.
  -- Deeper headings keep their anchors, but add neither numbers nor separate pages.
  let metadata := if depth >= 3 then { metadata with number := false } else metadata
  let metadata := if depth >= 2 then
      { metadata with htmlSplit := .never, htmlToc := false }
    else metadata
  { part with
    metadata := some metadata
    subParts := part.subParts.map fun child => readingLayout child (depth + 1) }

def main (args : List String) : IO UInt32 :=
  Informal.PreviewManifest.blueprintMainWithPreviewData
    (readingLayout (%doc TorchLeanBlueprint.Guide))
    args
    (extensionImpls := by exact extension_impls%)
