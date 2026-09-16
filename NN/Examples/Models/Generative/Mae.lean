/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Run:
  python3 scripts/datasets/download_example_data.py --cifar10
  lake -R -K cuda=true exe torchlean mae --device cuda --steps 1 --n-total 1
-/

module

public import NN.API
public import NN.Examples.Models.Common.RealData

/-!
# Masked Autoencoder CIFAR Example

This is the compact ViT-MAE-style training path in TorchLean.

The data path is explicit:

1. load real CIFAR-10 `.npy` arrays through `Data`;
2. take a typed image batch with shape `[batch, channels, height, width]`;
3. hide deterministic spatial blocks with `ssl.BlockMAE.sample`;
4. run a ViT encoder over patch tokens;
5. train a decoder head to reconstruct the original image vector.

The architecture uses one transformer encoder block and a linear pixel decoder rather than a large
asymmetric MAE decoder. It exercises image patch masking, patch embedding, transformer tokens, and
reconstruction of the original image.
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Generative.Mae

/-- Command name used in error messages and CLI output. -/
def exeName : String := "mae"

/-- Default JSON loss-curve path for this command. -/
def defaultLogPath : System.FilePath := Support.trainLogPath "mae"

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

/-- Patch stride; equal to patch size here, so patches do not overlap. -/
def stride : Nat := 2

/-- Zero padding around the image before patch extraction. -/
def padding : Nat := 0

/-- Width of each patch token after projection into the encoder stream. -/
def modelWidth : Nat := 4

/-- Number of self-attention heads in the compact ViT encoder. -/
def attentionHeads : Nat := 2

/--
Per-head attention width;
`attentionHeads * attentionHeadWidth = modelWidth`.
-/
def attentionHeadWidth : Nat := 2

/-- Hidden width of the feed-forward block inside the encoder. -/
def feedForwardWidth : Nat := 8

/-- Number of flattened crop values predicted by the decoder head. -/
def reconstructionWidth : Nat :=
  Shape.size [inputChannels, cropHeight, cropWidth]

/--
Small ViT-MAE configuration.

The command crops CIFAR images to `4×4`, divides them into four `2×2` patches, and reconstructs the
entire flattened crop. This keeps the command quick while making masking and attention genuinely
operate across multiple patch positions.
-/
abbrev modelConfig : nn.models.ViT.MaskedPatchReconstructor.Config 2 :=
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
    reconstructionWidth := reconstructionWidth }

/--
Hide one patch-index class every four patch positions.

The image remains an image tensor; the mask zeros whole patch regions before patch embedding.
-/
def maskPeriod : Nat := 4

/-- Phase of the deterministic patch mask. Changing this selects a different patch-index class. -/
def maskOffset : Nat := 0

/-- Per-axis block policy used by both the input mask and the reconstruction loss. -/
def maskBlocks : Tensor (Option Nat) [3] :=
  [none, some patchHeight, some patchWidth]

/-- Batch shape used by this training example. -/
abbrev batch : Shape := [batchSize]

/-- Input shape: a real batched CIFAR image tensor. -/
abbrev input := modelConfig.encoder.input batch

/-- Output shape: flattened image reconstruction. -/
abbrev output := modelConfig.output batch

/--
Construct the trainable model.

The architecture lives in the public self-supervised model API; this example only chooses a config,
loads data, and trains it.
-/
def model : nn.Builder (nn.Sequential input output) :=
  nn.models.ViT.maskedPatchReconstructor modelConfig batch

/--
Turn a typed CIFAR image batch into the compact MAE training sample.

The input stays an image tensor with some patches zeroed out. The target is the original image
flattened to a vector because the current decoder head predicts a batched matrix.
-/
def maskedAutoencoderSample
    (b : Sample.Supervised Float
      [batchSize, modelConfig.encoder.inputChannels, cropHeight, cropWidth]
      [batchSize, RealData.cifarClasses]) :
    Except String (Sample.Supervised Float input output) := do
  let sample ←
    ssl.BlockMAE.sample [batchSize]
      (dataShape := [modelConfig.encoder.inputChannels, cropHeight, cropWidth])
      modelConfig.reconstructionWidth maskBlocks maskPeriod maskOffset b.input
  pure <| by
    simpa [input, output,
      nn.models.ViT.MaskedPatchReconstructor.Config.output,
      nn.models.ViT.EncoderConfig.input, batch, modelConfig] using sample

/--
Normalized reconstruction weights repeated across the batch.

Only coordinates hidden in the model input have nonzero weight, and those weights sum to one across
each batch because this compact example uses a singleton batch.
-/
def lossWeights {α : Type} [Storage α] [Context α] : Tensor α output :=
  Tensor.repeatAxis 0 batchSize <|
    ssl.BlockMAE.reconstructionWeights
      (dataShape := [inputChannels, cropHeight, cropWidth]) maskBlocks maskPeriod maskOffset

/-- Mean squared reconstruction error over hidden coordinates only. -/
def hiddenReconstructionLoss {α : Type}
    [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Runtime.Ops (m := m) (α := α)]
    (prediction target : Runtime.ValueRef (m := m) (α := α) output) :
    m (Runtime.ValueRef (m := m) (α := α) ([] : Shape)) := do
  let weights ← TorchLean.Runtime.const
    (m := m) (α := α) (s := output) (lossWeights (α := α))
  Loss.mseWeighted (m := m) (α := α) (s := output) prediction target weights

/-- Runtime-polymorphic form of `hiddenReconstructionLoss` for `Trainer`. -/
def hiddenReconstructionLossProgram {α : Type}
    [Storage α] [Context α] :
    Runtime.Program α [output, output] ([] : Shape) :=
  fun {m} _ _ =>
    fun prediction target =>
      hiddenReconstructionLoss (m := m) (α := α) prediction target

/--
Public singleton dataset for masked-image reconstruction on one real CIFAR batch.

Like the compact vector generative examples, the sample itself is loaded as `Float` from the real
data boundary, then cast into the runtime-selected arithmetic representation by the public dataset
constructor.
-/
def data (flags : RealData.CifarModelTrainFlags) : Trainer.Dataset input output :=
  Data.defer do
    let sampleBatch ←
      RealData.loadCifarBatch exeName batchSize flags.data.nRows flags.data.seed
        flags.data.xPath flags.data.yPath
    let cropped ← CLI.orThrow exeName <|
      RealData.cropCifarBatch
        batchSize cropHeight cropWidth sampleBatch
    CLI.orThrow exeName (maskedAutoencoderSample cropped)

/-- Train the compact MAE model with the public `Trainer` surface. -/
def train (runtime : Runtime.Config) (flags : RealData.CifarModelTrainFlags) :
    IO (Trainer.Result input output) := do
  let mask := ssl.BlockMAE.hiddenMask
    (dataShape := [inputChannels, cropHeight, cropWidth]) maskBlocks maskPeriod maskOffset
  if (mask.map fun hidden => if hidden then (1 : Nat) else 0).sum == 0 then
    throw <| IO.userError s!"{exeName}: the configured mask hides no reconstruction coordinates"
  Data.requirePairedFiles exeName
    "CIFAR-10 images" flags.data.xPath
    "CIFAR-10 labels" flags.data.yPath
    RealData.missingCifarHint
  let trainer :=
    Trainer.new model <|
      Trainer.RunConfig.forObjective
        (Trainer.RunConfig.fromRuntime runtime
          { optimizer := optim.adam { learningRate := flags.training.learningRate } })
        (.custom hiddenReconstructionLossProgram)
        (seed := flags.data.seed)
  trainer.train
    (data flags)
    (flags.training.trainOptions
      (logTitle := "MAE CIFAR masked reconstruction")
      (logNotes := RealData.cifarClassifierNotes batchSize flags
        #[s!"maskPeriod={maskPeriod}", s!"maskOffset={maskOffset}"]))

/--
CLI entrypoint.

Useful flags:
- `--device cuda` runs the public trainer on the CUDA runtime.
- `--steps <n>` controls optimization steps.
- `--x <path> --y <path>` selects custom CIFAR-style `.npy` arrays.
- `--log <path>` writes the standard TorchLean training log JSON.
-/
def main (args : List String) : IO UInt32 :=
  TrainCommand.regressionNpy exeName args
    (fun rest => RealData.CifarModelTrainFlags.parse exeName rest defaultLogPath 10 1e-3)
    (Support.bannerWithDevice exeName "CIFAR masked reconstruction")
    train

end NN.Examples.Models.Generative.Mae
