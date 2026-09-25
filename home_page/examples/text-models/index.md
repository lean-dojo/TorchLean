---
title: Text Models Walkthrough
---

The text-model examples read a corpus, build next-token samples, train, save and reload parameters,
and generate a continuation. Tokenization, causal windows, parameter shapes, generation settings,
and logs remain explicit along that path.

## Data: One Explicit Text File

The text examples use a single UTF-8 text file. In the default setup, the
download script places Tiny Shakespeare under `data/real/text/`, and the Lean example fails loudly
if the file is missing.

Run the data step once:

```bash
python3 scripts/datasets/download_example_data.py --tiny-shakespeare
```

The function that turns “flags and paths” into an actual corpus is shared across examples. In the
GPT-2 example, `takeInputText` uses `text.Corpus.takeUtf8Input` to support a “use this known
corpus” flag or an explicit `--data-file` path.

## Tokenization: Bytes First, BPE When Requested

The main tutorial path tokenizes bytes directly. `text.Tokenizer.byte` maps every UTF-8 byte to
one token id in `[0, 256)`, so its `vocabularySize` is fixed. That choice is practical:

- there is no external BPE model file to keep in sync,
- there is no “which tokenizer version did you mean?” ambiguity,
- it makes boundary mistakes easy to spot (and easy to turn into case studies).

The runnable GPT example uses all 256 byte ids. Training and generation therefore share the same
vocabulary, and generated ids can be decoded by the byte tokenizer without a lossy modulo mapping.
The model width and context length keep the example small.

The larger `text_gpt2` command can also use GPT-2 BPE files:

```bash
lake -R -K cuda=true exe torchlean text_gpt2 --device cuda \
  --data-file data/real/text/tinystories_valid.txt \
  --bpe-vocab data/real/gpt2/vocab.json \
  --bpe-merges data/real/gpt2/merges.txt \
  --allow-small-data --steps 1 --generate 0
```

That path is still a TorchLean training example with randomly initialized weights. The BPE files
define the tokenizer boundary, while the Lean command projects the observed BPE ids into a compact
local vocabulary for a runnable example. The tokenizer choice is visible
in the command and in the shapes, rather than being an implicit cache dependency.

## Supervised Examples: Next-Token Prediction As Tensors

The training data is represented directly as typed supervised samples. The examples build explicit
`Sample.Supervised` values whose shapes say what they are.

For GPT-2, the sample is a one-hot matrix for causal language modeling:

```lean
abbrev input : Shape := [batchSize, contextLength, vocabularySize]
abbrev output : Shape := input
```

The function `Data.CausalLM.oneHotSample` converts a token tensor whose final axis has length
$\mathtt{seqLen}+1$ into $(x,y)$: input tokens and the same window shifted by one position as the
target. The leading shape determines whether the sample is batched.

The Lean sample constructor is explicit:

```lean
def batchSampleFromTokenIds (idsByBatch : Tensor Nat [batchSize, contextLength + 1]) :
    Sample.Supervised Float input output :=
  Data.CausalLM.oneHotSample (α := Float) [batchSize] contextLength vocabularySize
    (idsByBatch.map byteIndex)
```

The tutorial keeps the dataloader convention visible: a supervised example is a pair of typed
tensors. `Gpt2.samplesFromCorpus` byte-encodes the corpus once and retains the token tensor in the
sample stream's closure. Each requested sample takes windows from that tensor, then constructs
the one-hot inputs and shifted targets. The stream does not re-encode the complete text for
every sample. Its window offsets and space-byte padding still determine which tokens each
sample contains.

<a id="gpt-2"></a>

## GPT-2: A Small Causal Transformer

`NN.Examples.Models.Sequence.Gpt2` wires up a miniature causal Transformer from reusable layers
(`nn.models.CausalTransformer.oneHot`). The default configuration is compact enough for a local run,
but it is still large enough to learn short structure from Tiny Shakespeare.

The model declaration is a normal Lean value:

```lean
def model : nn.Builder (nn.Sequential input output) :=
  nn.models.CausalTransformer.oneHot
    { sequenceLength := contextLength
      vocabularySize := vocabularySize
      headCount := attentionHeads
      headWidth := attentionHeadWidth
      feedForwardWidth := feedForwardWidth
      layerCount := transformerLayers }
    (batchShape := [batchSize])
```

The configuration describes the Transformer itself. The `batchShape` argument supplies the leading
batch shape for this particular training run; the same constructor also accepts an unbatched
sequence or several leading collection axes.

The training loop stays on the same public API used by the simpler quickstarts:

```lean
let run := Trainer.RunConfig.fromRuntime runtime
  { optimizer := optim.adam { learningRate := options.training.learningRate } }
let objective : Trainer.Objective output := .oneHotCrossEntropy 2
let trainer := Trainer.new model (Trainer.RunConfig.forObjective run objective)
let trained ← trainer.train (Data.fromSamples samples)
  { steps := options.training.steps
    samplesPerStep := options.training.batchSize
    loadCheckpoint? := options.checkpoint.loadCheckpoint?
    saveCheckpoint? := options.checkpoint.saveCheckpoint? }
trained.printSummary
```

`Trainer.RunConfig.fromRuntime` turns the parsed `--device`, `--arithmetic`, and `--execution`
flags into a run configuration, and `forObjective` attaches the loss. `TrainOptions.steps` is
required; `samplesPerStep` is the number of dataset items whose gradients are averaged before one
update. The surrounding code builds a bank of token windows from the corpus, reports before/after
predictions, saves checkpoints when requested, and samples text from the trained prediction closure.

Sampling is also explicit: the example computes logits, applies temperature and top-k filtering, and
chooses the next token. If you’ve ever written a small “Karpathy-style” sampler, the code will look
familiar.

Try the short CUDA run:

```bash
lake -R -K cuda=true exe torchlean gpt2 --device cuda --tiny-shakespeare \
  --steps 300 --windows 32 --lr 0.001 --prompt "ROMEO:" --generate 220 \
  --temperature 0.85 --top-k 24 --repeat-penalty 1.25 --repeat-window 24 \
  --sample-seed 11 --log data/examples/gpt2_trainlog.json
```

For a tiny runtime check, keep the same CUDA path and shrink the workload:

```bash
lake -R -K cuda=true exe torchlean gpt2 --device cuda --tiny-shakespeare --steps 1 --windows 1 --generate 0
```

## Saving and Reloading A Model

TorchLean’s checkpoint format stores the model's shape-indexed state and round-trips exact
IEEE-754 bit patterns through JSON.

`gpt2_saved` is a separate example because it loads a state pack, checks that the shapes
match the model architecture, and runs sampling without touching an optimizer.

The GPT-2 command writes a checkpoint through `--save-checkpoint`; the inference example reloads
the same shape-indexed state before sampling. If the shape list no longer matches
the model, loading fails before the weights are used.

<a id="mamba"></a>

## Mamba: State-Space Text In The Same Runtime

`NN.Examples.Models.Sequence.Mamba` also runs on byte tokens, but swaps attention for a
compact state-space block. The contrast is sequence modeling without the quadratic attention path,
using the same autograd and the same logging style.

Algorithmically, the example replaces “attend over all previous tokens” with a learned recurrent
state update. Each token updates a compact state, and the model projects the resulting sequence
states to next-token logits. The surrounding training interface stays the same: corpus windows in,
logits out, cross-entropy loss, optimizer step, JSON log.

The Mamba example has the same tutorial shape:

```lean
abbrev modelConfig : nn.models.Mamba.Config :=
  { vocabularySize := vocabularySize
    modelWidths := [stateWidth] }

abbrev input : Shape := modelConfig.inputShape contextLength
abbrev output : Shape := modelConfig.outputShape contextLength

def model : nn.Builder (nn.Sequential input output) :=
  nn.models.Mamba.languageModel modelConfig contextLength
```

and the training body is still an ordinary trainer call:

```lean
let run := Trainer.RunConfig.fromRuntime runtime
  { optimizer := optim.adam { learningRate := options.training.learningRate } }
let trainer := Trainer.new model <|
  Trainer.RunConfig.forObjective run (.oneHotCrossEntropy 1)
let trained ← trainer.train (Data.fromSamples samples) (options.training.trainOptions)
trained.printSummary
```

Run it with:

```bash
lake -R -K cuda=true exe torchlean mamba --device cuda --tiny-shakespeare \
  --steps 1 --windows 1 --generate 0
```

## Inspecting Runs In The Lean Editor

For interactive inspection, open `NN.Examples.Models.Sequence.Gpt2` or
`NN.Examples.Models.Sequence.Mamba` in VS Code with the Lean Infoview enabled. The relevant widgets
can display saved training logs, tensor summaries, inferred shapes, and debug traces next to the
Lean source.

A concrete starting point is the training-log widget documented near the top of
`NN.Examples.Models.Sequence.Gpt2`; it renders a saved JSON loss log in the Infoview.

Source entry points:

- [`NN.Examples.Models.Sequence.Gpt2`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Sequence/Gpt2.lean)
- [`NN.Examples.Models.Sequence.Gpt2Saved`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Sequence/Gpt2Saved.lean)
- [`NN.Examples.Models.Sequence.Mamba`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Sequence/Mamba.lean)
- [Three End-to-End Case Studies]({{ '/blueprint/Examples-and-Applications/Three-End-to-End-Case-Studies/' | relative_url }})
