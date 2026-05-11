/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN
public import NN.API.Models.Generative
public import NN.Examples.Data.RealPaths

/-!
# Shared Real-Data Helpers for Model Examples

`NN/Examples/Models/*` should exercise real data paths. This file centralizes the shared parts:

- loading a prepared CIFAR-10 NPY minibatch,
- reading a local text corpus, and
- printing the same "how to prepare data" hint everywhere.

The data files are prepared by `scripts/datasets/download_example_data.py`; examples report missing inputs
explicitly instead of silently falling back to synthetic tensors.
-/

@[expose] public section

open Spec
open Tensor
open NN.API

namespace NN.Examples.Models.RealData

def cifarChannels : Nat := 3
def cifarHeight : Nat := 32
def cifarWidth : Nat := 32
def cifarClasses : Nat := 10
def defaultCifarRows : Nat := 200

def imagenet64Channels : Nat := 3
def imagenet64Height : Nat := 64
def imagenet64Width : Nat := 64
def imagenet64Classes : Nat := 1000
def defaultImageNet64Rows : Nat := 200

instance : NeZero cifarChannels := ⟨by decide⟩
instance : NeZero cifarHeight := ⟨by decide⟩
instance : NeZero cifarWidth := ⟨by decide⟩
instance : NeZero cifarClasses := ⟨by decide⟩
instance : NeZero imagenet64Channels := ⟨by decide⟩
instance : NeZero imagenet64Height := ⟨by decide⟩
instance : NeZero imagenet64Width := ⟨by decide⟩
instance : NeZero imagenet64Classes := ⟨by decide⟩

abbrev CifarImage : Shape :=
  Shape.Image cifarChannels cifarHeight cifarWidth

abbrev CifarTarget : Shape :=
  Shape.Vec cifarClasses

/-- ImageNet-style converted image shape used by the higher-resolution diffusion example. -/
abbrev ImageNet64Image : Shape :=
  Shape.Image imagenet64Channels imagenet64Height imagenet64Width

/--
One-hot target shape for ImageNet-style folders.

The diffusion example ignores labels, but reusing `Data.LabeledSource` keeps the data path identical
to the supervised examples and lets class-directory conversion catch malformed labels early.
-/
abbrev ImageNet64Target : Shape :=
  Shape.Vec imagenet64Classes

def missingCifarHint : String :=
  "Prepare real CIFAR-10 arrays with:\n" ++
  "  python3 scripts/datasets/download_example_data.py --cifar10\n" ++
  "Then rerun the model command."

def missingImageNet64Hint : String :=
  "Prepare an ImageNet-style 64x64 subset with:\n" ++
  "  python3 scripts/datasets/torchlean_data_convert.py image-folder \\\n" ++
  "    --input /path/to/imagenet/train \\\n" ++
  "    --x-output data/real/imagenet64/imagenet64_train_X.npy \\\n" ++
  "    --y-output data/real/imagenet64/imagenet64_train_y.npy \\\n" ++
  "    --height 64 --width 64 --labels-from-dirs --limit 2000\n" ++
  "TorchLean does not redistribute ImageNet; this path expects a local image-folder dataset."

def missingTextHint : String :=
  "Prepare the text corpus with:\n" ++
  "  python3 scripts/datasets/download_example_data.py --tiny-shakespeare\n" ++
  "or pass --data-file PATH."

def parseCifarFlags (args : List String) :
    Except String (System.FilePath × System.FilePath × Nat × Nat × List String) := do
  let (seed, args) ← CLI.takeSeed args 0
  let (nRows?, args) ← CLI.takeNatFlagOnce args "n-total"
  let (x?, args) ← CLI.takePathFlagOnce args "x"
  let (y?, args) ← CLI.takePathFlagOnce args "y"
  pure (x?.getD _root_.NN.Examples.Data.RealPaths.cifar10TrainX,
    y?.getD _root_.NN.Examples.Data.RealPaths.cifar10TrainY,
    nRows?.getD defaultCifarRows,
    seed,
    args)

/--
Parse the shared flags for an ImageNet-style 64x64 NPY dataset.

The expected input is produced by `scripts/datasets/torchlean_data_convert.py image-folder`; that converter
handles JPEG/PNG decoding, RGB conversion, resizing, class-directory labels, and the final NCHW
layout. Lean then reads only the simple `.npy` tensors.
-/
def parseImageNet64Flags (args : List String) :
    Except String (System.FilePath × System.FilePath × Nat × Nat × List String) := do
  let (seed, args) ← CLI.takeSeed args 0
  let (nRows?, args) ← CLI.takeNatFlagOnce args "n-total"
  let (x?, args) ← CLI.takePathFlagOnce args "x"
  let (y?, args) ← CLI.takePathFlagOnce args "y"
  pure (x?.getD _root_.NN.Examples.Data.RealPaths.imagenet64TrainX,
    y?.getD _root_.NN.Examples.Data.RealPaths.imagenet64TrainY,
    nRows?.getD defaultImageNet64Rows,
    seed,
    args)

def loadCifarLoader {α : Type} [Semantics.Scalar α] [Runtime.Scalar α]
    (exeName : String) (batch nRows seed : Nat) (xPath yPath : System.FilePath) :
    IO (Data.BatchLoader α batch CifarImage CifarTarget) := do
  unless (← xPath.pathExists) do
    throw <| IO.userError s!"{exeName}: missing CIFAR-10 images: {xPath}\n{missingCifarHint}"
  unless (← yPath.pathExists) do
    throw <| IO.userError s!"{exeName}: missing CIFAR-10 labels: {yPath}\n{missingCifarHint}"
  let src := Data.LabeledSource.ofPaths .npy xPath yPath nRows
    [cifarChannels, cifarHeight, cifarWidth] cifarClasses
  let dsE ← src.load (α := α)
  let ds ←
    match dsE with
    | .ok ds => pure ds
    | .error msg =>
        let hint :=
          s!"{exeName}: failed to load CIFAR-10 arrays for --n-total {nRows}.\n" ++
          s!"{msg}\n" ++
          "If your local .npy files contain fewer rows, pass --n-total with that row count; " ++
          "to regenerate the default 200-row slice, run:\n" ++
          "  python3 scripts/datasets/download_example_data.py --cifar10"
        throw <| IO.userError hint
  -- Return the general typed minibatch loader, not just one batch.  Model examples can either take
  -- the first batch for a small fixed-sample check or pass the whole loader to
  -- `train.fitModuleLoaderWith` for real multi-epoch training over shuffled batches.
  let dl := Data.batchLoader ds batch (shuffle := true) (seed := seed) (dropLast := true)
  pure dl

def loadCifarBatch {α : Type} [Semantics.Scalar α] [Runtime.Scalar α]
    (exeName : String) (batch nRows seed : Nat) (xPath yPath : System.FilePath) :
    IO (sample.Batch α batch CifarImage CifarTarget) := do
  let dl ← loadCifarLoader (α := α) exeName batch nRows seed xPath yPath
  let (_dl', batches) ← Common.orThrow exeName <| Data.BatchLoader.epoch exeName dl
  match batches with
  | b :: _ => pure b
  | [] =>
      throw <| IO.userError
        s!"{exeName}: no full CIFAR-10 minibatch available (batch={batch}, rows={Data.size dl.raw.dataset})"

/--
Load a user-prepared ImageNet-style `64x64` minibatch.

This is intentionally a `.npy` reader, not a JPEG reader.  The Python converter is the trust boundary
for filesystem image decoding and resizing; this Lean path checks the resulting tensor shape and class
range before handing the batch to examples.
-/
def loadImageNet64Loader {α : Type} [Semantics.Scalar α] [Runtime.Scalar α]
    (exeName : String) (batch nRows seed : Nat) (xPath yPath : System.FilePath) :
    IO (Data.BatchLoader α batch ImageNet64Image ImageNet64Target) := do
  unless (← xPath.pathExists) do
    throw <| IO.userError s!"{exeName}: missing ImageNet64 images: {xPath}\n{missingImageNet64Hint}"
  unless (← yPath.pathExists) do
    throw <| IO.userError s!"{exeName}: missing ImageNet64 labels: {yPath}\n{missingImageNet64Hint}"
  let src := Data.LabeledSource.ofPaths .npy xPath yPath nRows
    [imagenet64Channels, imagenet64Height, imagenet64Width] imagenet64Classes
  let dsE ← src.load (α := α)
  let ds ←
    match dsE with
    | .ok ds => pure ds
    | .error msg =>
        let hint :=
          s!"{exeName}: failed to load ImageNet64 arrays for --n-total {nRows}.\n" ++
          s!"{msg}\n" ++
          "If your local .npy files contain fewer rows, pass --n-total with that row count; " ++
          "to create ImageNet64 arrays, run the image-folder converter described in the error hint."
        throw <| IO.userError hint
  -- Same convention as CIFAR: this is the reusable loader for all full-dataset loops.  The
  -- `loadImageNet64Batch` wrapper below is only for examples that intentionally want one batch.
  let dl := Data.batchLoader ds batch (shuffle := true) (seed := seed) (dropLast := true)
  pure dl

def loadImageNet64Batch {α : Type} [Semantics.Scalar α] [Runtime.Scalar α]
    (exeName : String) (batch nRows seed : Nat) (xPath yPath : System.FilePath) :
    IO (sample.Batch α batch ImageNet64Image ImageNet64Target) := do
  let dl ← loadImageNet64Loader (α := α) exeName batch nRows seed xPath yPath
  let (_dl', batches) ← Common.orThrow exeName <| Data.BatchLoader.epoch exeName dl
  match batches with
  | b :: _ => pure b
  | [] =>
      throw <| IO.userError
        s!"{exeName}: no full ImageNet64 minibatch available (batch={batch}, rows={Data.size dl.raw.dataset})"

/--
Load a CIFAR minibatch and expose it as a compact flattened vector batch.

The file paths and download hints remain in `NN.Examples`, but the flattening logic itself lives in
`NN.API.Models.Generative` so users can reuse it with their own image tensors.
-/
def loadCifarVectorBatch (cfg : nn.models.VectorGenerativeConfig)
    (hData : cfg.dataDim ≤ Shape.size CifarImage)
    (exeName : String) (xPath yPath : System.FilePath) (nRows seed : Nat) :
    IO (Tensor Float (nn.models.vectorDataShape cfg)) := do
  let batchSample ← loadCifarBatch (α := Float) exeName cfg.batch nRows seed xPath yPath
  pure (nn.models.flattenBatchPrefix cfg hData (sample.x batchSample))

def parseTextFlags (args : List String) :
    Except String (System.FilePath × List String) := do
  let tiny := args.contains "--tiny-shakespeare"
  let args := args.filter (fun a => a != "--tiny-shakespeare")
  let (path?, args) ← CLI.takePathFlagOnce args "data-file"
  pure (path?.getD (if tiny then _root_.NN.Examples.Data.RealPaths.tinyShakespeare
    else _root_.NN.Examples.Data.RealPaths.tinyShakespeare), args)

def readTextCorpus (exeName : String) (path : System.FilePath) : IO String := do
  unless (← path.pathExists) do
    throw <| IO.userError s!"{exeName}: missing text corpus: {path}\n{missingTextHint}"
  let text ← IO.FS.readFile path
  if text.isEmpty then
    throw <| IO.userError s!"{exeName}: empty text corpus: {path}"
  pure text

def textCausalSample {α : Type} [Semantics.Scalar α] [Runtime.Scalar α]
    (seqLen vocab : Nat) (input : String) :
    sample.Supervised α (Shape.Mat seqLen vocab) (Shape.Mat seqLen vocab) :=
  let bytes := input.toUTF8
  let toks := (NN.API.text.byteTokenWindow bytes (seqLen + 1)).map (fun b => b % vocab)
  let (xF, yF) := NN.API.text.causalLmXYOneHotMatFloat seqLen vocab toks
  let x : Tensor α (Shape.Mat seqLen vocab) := Common.castTensor Runtime.ofFloat xF
  let y : Tensor α (Shape.Mat seqLen vocab) := Common.castTensor Runtime.ofFloat yF
  sample.mk x y

end NN.Examples.Models.RealData
