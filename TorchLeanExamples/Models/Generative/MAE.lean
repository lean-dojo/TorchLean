/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Run:
  python3 scripts/datasets/download_example_data.py --cifar10
  scripts/lake.sh -Kcuda=true exe torchlean mae --device cuda --steps 1 --n-total 1
-/

module

public import TorchLeanExamples.Models.RealData
public import TorchLean.NN.Models.MaskedPatchReconstructor
public import TorchLean.NN.SelfSupervised
public import TorchLeanExamples.Models.Training

/-!
# Masked Autoencoder CIFAR Example

A compact ViT-MAE-style model trained on real CIFAR-10 images. The data path has five steps:

1. Load the CIFAR-10 `.npy` arrays through `Data`.
2. Crop a typed image batch of shape `[batch, channels, height, width]` to `4×4`.
3. Zero one deterministic `2×2` patch with `SelfSupervised.BlockMAE.sample`.
4. Run a ViT encoder over the patch tokens, the zeroed patch included.
5. Train a decoder head to reconstruct the original image vector.

The model is one transformer encoder block followed by a linear pixel decoder. Small as it is, it
exercises image patch masking, patch embedding, transformer tokens and reconstruction of the
original image.
-/

@[expose] public section

open TorchLean.Trainer.Objective
open TorchLean.Shape (size)

namespace TorchLean.Examples.MAE

/-- Command name used in error messages and CLI output. -/
def exeName : String := "mae"

/-- Default JSON loss-curve path for this command. -/
def defaultLogPath : System.FilePath := Training.artifactPath "mae"

/-- CIFAR minibatch size used by the typed MAE command. -/
def batchSize : Nat := 1

/-- Number of CIFAR image channels. -/
def inputChannels : Nat := RealData.cifarChannels

/-- Cropped CIFAR image height for the compact runnable example. -/
def cropHeight : Nat := 4

/-- Cropped CIFAR image width for the compact runnable example. -/
def cropWidth : Nat := 4

/-- Patch height for the image-to-token projection. -/
def patchHeight : Nat := 2

/-- Patch width for the image-to-token projection. -/
def patchWidth : Nat := 2

/-- Patch stride, equal to the patch size so that patches do not overlap. -/
def stride : Nat := 2

/-- Zero padding around the image before patch extraction. -/
def padding : Nat := 0

/-- Width of each patch token after projection into the encoder stream. -/
def modelWidth : Nat := 4

/-- Number of self-attention heads in the compact ViT encoder. -/
def attentionHeads : Nat := 2

/--
Per-head attention width, chosen so that `attentionHeads * attentionHeadWidth = modelWidth`.
-/
def attentionHeadWidth : Nat := 2

/-- Hidden width of the feed-forward block inside the encoder. -/
def feedForwardWidth : Nat := 8

/-- Number of flattened crop values predicted by the decoder head. -/
def reconstructionWidth : Nat :=
  Shape.size [inputChannels, cropHeight, cropWidth]

/--
Small ViT-MAE configuration.

The command crops CIFAR images to `4×4`, cuts each crop into four `2×2` patches and reconstructs
all 48 values of the flattened crop. Four patch positions keep the run quick and are enough for
masking and attention to act across patches.
-/
abbrev modelConfig : nn.MaskedPatchReconstructor.Config 2 :=
  { encoder :=
      { inputChannels := inputChannels
        spatial := [cropHeight, cropWidth]
        patchEmbedding :=
          { outChannels := modelWidth
            kernelSize := [patchHeight, patchWidth]
            stride := [stride, stride]
            padding := [padding, padding] }
        headCount := attentionHeads
        headWidth := attentionHeadWidth
        feedForwardWidth := feedForwardWidth }
    outputWidth := reconstructionWidth }

/--
Mask period: a patch is hidden when its row-major index is congruent to `maskOffset` modulo 4,
which picks one of the four patches of a crop.

The mask zeros the whole patch region of the image tensor before patch embedding.
-/
def maskPeriod : Nat := 4

/--
Phase of the deterministic patch mask. Offset `0` hides the top-left patch, and offsets `1` to `3`
select the other three.
-/
def maskOffset : Nat := 0

/--
Per-axis block policy of the input mask and the reconstruction loss: `none` keeps all channels in
one block, and the two spatial axes are cut into `2×2` patches.
-/
def maskBlocks : Tensor (Option Nat) [3] :=
  [none, some patchHeight, some patchWidth]

/-- Batch shape used by this training example. -/
abbrev batch : Shape := [batchSize]

/-- Input shape: a batch of cropped CIFAR images. -/
abbrev input := modelConfig.encoder.inputShape batch

/-- Output shape: flattened image reconstruction. -/
abbrev output := modelConfig.outputShape batch

/--
Construct the trainable model from the public self-supervised model API.

The example only picks the configuration, loads the data and trains it.
-/
def model : nn.Builder (nn.Sequential input output) :=
  nn.maskedPatchReconstructor modelConfig batch

/--
Turn a cropped CIFAR batch into an MAE training sample.

The input stays an image tensor with some patches zeroed out. The target is the original image
flattened to a vector because the decoder head predicts a batched matrix.
-/
def sample
    (b : Sample.Supervised Float
      [batchSize, modelConfig.encoder.inputChannels, cropHeight, cropWidth]
      [batchSize, RealData.cifarClasses]) :
    Except String (Sample.Supervised Float input output) := do
  let sample ←
    SelfSupervised.BlockMAE.sample [batchSize]
      (dataShape := [modelConfig.encoder.inputChannels, cropHeight, cropWidth])
      modelConfig.outputWidth maskBlocks maskPeriod maskOffset b.input
  pure <| by
    simpa [input, output,
      nn.MaskedPatchReconstructor.Config.outputShape,
      nn.ViT.EncoderConfig.inputShape, batch, modelConfig] using sample

/--
Reconstruction weights `1/k` on the `k` hidden coordinates of a sample and zero elsewhere,
repeated across the batch.

Here `k = 12` (one `2×2` patch in three channels), and with the singleton batch the weights sum
to one.
-/
def lossWeights {α : Type} [Storage α] [Zero α] [RatCast α] :
    Tensor α output :=
  Tensor.repeatAxis 0 batchSize <|
    SelfSupervised.BlockMAE.reconstructionWeights
      (dataShape := [inputChannels, cropHeight, cropWidth]) maskBlocks maskPeriod maskOffset

/-- Mean squared reconstruction error over hidden coordinates only. -/
def loss {α : Type}
    [Storage α] [Context α] :
    Program α [output, output] ([] : Shape) :=
  fun {m} _ _ prediction target =>
    show m (Ops.Ref (m := m) (α := α) ([] : Shape)) from do
      let weights ← Ops.const
        (m := m) (α := α) (s := output) (lossWeights (α := α))
      Loss.mseWeighted (m := m) (α := α) (s := output) prediction target weights

/--
One-sample dataset for masked-image reconstruction, built from the first CIFAR batch.

The batch is read as `Float`, cropped and masked, and `Data.defer` casts it to the runtime's scalar
type when the trainer materializes the dataset.
-/
def data (flags : Training.Options NPY.Options) (seed : Nat) :
    Trainer.Dataset input output :=
  Data.defer do
    let sampleBatch ←
      RealData.loadCifarBatch exeName batchSize flags.data.nRows seed
        flags.data.xPath flags.data.yPath
    let cropped ← CLI.orThrow exeName <|
      RealData.cropCifarBatch
        batchSize cropHeight cropWidth sampleBatch
    CLI.orThrow exeName (sample cropped)

/--
Train the compact MAE model with the public `Trainer` and Adam.

Before training it rejects a mask that hides no coordinate and checks that both CIFAR files exist.
-/
def train (runtime : Session.Config) (flags : Training.Options NPY.Options) :
    IO (Trainer.Result input output) := do
  let mask := SelfSupervised.BlockMAE.hiddenMask
    (dataShape := [inputChannels, cropHeight, cropWidth]) maskBlocks maskPeriod maskOffset
  if (mask.map fun hidden => if hidden then (1 : Nat) else 0).sum == 0 then
    throw <| IO.userError s!"{exeName}: the configured mask hides no reconstruction coordinates"
  Data.requirePairedFiles exeName
    "CIFAR-10 images" flags.data.xPath
    "CIFAR-10 labels" flags.data.yPath
    RealData.missingCifarHint
  let trainer :=
    Trainer.create model <|
      { runtime with
          optimizer := optim.adam { lr := flags.training.lr }
          objective := custom loss }
  trainer.train
    (data flags runtime.seed)
    (flags.training.trainOptions
      (logTitle := "MAE CIFAR masked reconstruction")
      (logNotes := RealData.trainingNotes "cifar10" batchSize flags
        #[s!"maskPeriod={maskPeriod}", s!"maskOffset={maskOffset}"]))

/--
CLI entrypoint.

Useful flags:
- `--device cuda` runs the public trainer on the CUDA runtime.
- `--steps <n>` sets the number of optimization steps (default 10).
- `--x <path> --y <path>` select custom CIFAR-style `.npy` arrays.
- `--log <path>` writes the standard TorchLean training log JSON.
-/
def main (args : List String) : IO UInt32 :=
  NPY.train exeName args
    (fun rest => Training.Options.parse exeName rest defaultLogPath 10 1e-3
      (parseData := RealData.parseCifar))
    (Training.banner exeName "CIFAR masked reconstruction")
    train (fun result => result.printSummary)

end TorchLean.Examples.MAE
