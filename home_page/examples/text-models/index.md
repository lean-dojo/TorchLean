---
title: Text Models Walkthrough
---

In this walkthrough, we'll train a small GPT-style transformer and a Mamba model on text.
We'll start with a text file, turn it into next-token prediction samples, and pass those samples
to TorchLean's trainer. With the GPT example, we'll also save a checkpoint, reload it, and
generate text from a prompt.

Run the shell commands from the repository root. The CUDA examples need a compatible LibTorch
SDK and GPU; follow the [installation guide]({{ '/installation/' | relative_url }}) first.

## Load the text

Let's start with Tiny Shakespeare, stored as a single UTF-8 text file. The download script puts
it under `data/real/text/`. Run this once before training; the Lean example reports an error if
the file is missing:

```bash
python3 scripts/datasets/download_example_data.py --tiny-shakespeare
```

To use your own text, pass `--data-file path/to/file.txt` instead of `--tiny-shakespeare`.
The GPT-2 example reads either choice through `text.Corpus.takeUtf8Input`.

## Tokenization: Bytes First, BPE When Requested

We'll use byte tokens first. `text.Tokenizer.byte` maps every UTF-8 byte to
one token id in `[0, 256)`. This gives us a fixed vocabulary of 256 entries without downloading
a separate tokenizer.

The runnable GPT example uses all 256 byte ids. Training and generation therefore share the same
vocabulary, and generated ids can be decoded by the byte tokenizer without a lossy modulo mapping.
The model width and context length keep the example small.

The `text_gpt2` command can also use GPT-2 BPE files. Supply a matching `vocab.json` and
`merges.txt` under `data/real/gpt2/`; the dataset downloader does not install tokenizer files.
The command below reuses the Tiny Shakespeare corpus downloaded above:

```bash
scripts/lake.sh -Kcuda=true exe torchlean text_gpt2 --device cuda \
  --data-file data/real/text/tiny_shakespeare.txt \
  --bpe-vocab data/real/gpt2/vocab.json \
  --bpe-merges data/real/gpt2/merges.txt \
  --allow-small-data --steps 1 --generate 0
```

This command trains from randomly initialized weights. The BPE files define how text is split into
tokens. The Lean command maps the observed BPE ids into a compact
local vocabulary of at most 512 entries; token ids outside that vocabulary map to entry zero.
It does not load pretrained GPT-2 weights or use the full GPT-2 output vocabulary.

## Supervised Examples: Next-Token Prediction As Tensors

Now we need to pair each input with the tokens we want the model to predict. We'll store the
input and target tensors together in a `Sample.Supervised` value.

For GPT-2, the sample is a one-hot matrix for causal language modeling:

```lean
abbrev input : Shape := [batchSize, contextLength, vocabularySize]
abbrev output : Shape := input
```

The function `Data.CausalLM.oneHotSample` converts a token tensor whose final axis has length
$\mathtt{seqLen}+1$ into $(x,y)$: input tokens and the same window shifted by one position as the
target. The leading shape determines whether the sample is batched.

We construct a sample with:

```lean
def batchSampleFromTokenIds (idsByBatch : Tensor Nat [batchSize, contextLength + 1]) :
    Sample.Supervised Float input output :=
  Data.CausalLM.oneHotSample (α := Float) [batchSize] contextLength vocabularySize
    (idsByBatch.map byteIndex)
```

`Gpt2.samplesFromCorpus` byte-encodes the corpus once and retains the token tensor in the
sample stream's closure. Each requested sample takes windows from that tensor, then constructs
the one-hot inputs and shifted targets. The stream does not re-encode the complete text for
every sample. Its window offsets and space-byte padding still determine which tokens each
sample contains.

<a id="gpt-2"></a>

## GPT-2: A Small Causal Transformer

We'll build a small causal transformer with `nn.models.CausalTransformer.oneHot`.
The example uses the GPT-style architecture, not pretrained GPT-2 weights.

We can build the transformer with this Lean definition:

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

With the model defined, we can train it using the same API as the quickstarts. These excerpts
use the constants and options declared in the linked example source:

```lean
let run := Trainer.RunConfig.fromRuntime runtime
  { optimizer := optim.adam { learningRate := options.training.learningRate } }
let objective : Trainer.Objective output := Trainer.Objective.oneHotCrossEntropy 2
let trainer := Trainer.new model <|
  Trainer.RunConfig.forObjective run objective (seed := runtime.seed)
let trained ← trainer.train (Data.fromStream samples)
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

For generation, we pass the model's score callback to `text.generate`. It keeps the context
window, applies the repetition penalty, and appends the sampled tokens to the prompt.
`text.next` selects a single token using the same options. For lower-level experiments,
`text.greedy?`, `text.topK`, and `text.sample?` work directly with a tensor of scores.
The `?` means selection can fail—for example, if every allowed score is NaN.

Try the short CUDA run:

```bash
scripts/lake.sh -Kcuda=true exe torchlean gpt2 --device cuda --tiny-shakespeare \
  --steps 300 --windows 32 --lr 0.001 --prompt "ROMEO:" --generate 220 \
  --temperature 0.85 --top-k 24 --repeat-penalty 1.25 --repeat-window 24 \
  --sample-seed 11 --log data/examples/gpt2_trainlog.json
```

For a tiny runtime check, keep the same CUDA path and shrink the workload:

```bash
scripts/lake.sh -Kcuda=true exe torchlean gpt2 --device cuda --tiny-shakespeare \
  --steps 1 --windows 1 --generate 0
```

## Saving and Reloading A Model

Once we've trained the model, we can save its state for another run. TorchLean's checkpoint
format stores the shape-indexed state in JSON and preserves the exact IEEE-754 bit patterns.

`gpt2_saved` is a separate example because it loads a state pack, checks that the shapes
match the model architecture, and runs sampling without touching an optimizer.

The GPT-2 command writes a checkpoint through `--save-checkpoint`; the inference example reloads
the same shape-indexed state before sampling. If the shape list no longer matches
the model, loading fails before the weights are used.

<a id="mamba"></a>

## Mamba: State-Space Text In The Same Runtime

Next, let's try a different sequence model. `NN.Examples.Models.Sequence.Mamba` uses byte tokens
with a compact state-space block in place of attention. We can keep the same trainer, automatic
differentiation, and JSON loss logs.

Each token updates a recurrent state instead of attending to every previous token.
The model turns those states into next-token scores. We'll train it with the same cross-entropy
loss and save the same kind of loss log as the transformer.

We'll configure one block with width `modelWidth`:

```lean
abbrev modelConfig : nn.models.Mamba.Config :=
  { vocabularySize := vocabularySize
    modelWidths := [modelWidth] }

abbrev input : Shape := modelConfig.inputShape contextLength
abbrev output : Shape := modelConfig.outputShape contextLength

def model : nn.Builder (nn.Sequential input output) :=
  nn.models.Mamba.languageModel modelConfig contextLength
```

Then we pass it to the trainer:

```lean
let run := Trainer.RunConfig.fromRuntime runtime
  { optimizer := optim.adam { learningRate := options.training.learningRate } }
let trainer := Trainer.new model <|
  Trainer.RunConfig.forObjective run (.oneHotCrossEntropy 1) (seed := runtime.seed)
let trained ← trainer.train (Data.fromStream samples) (options.training.trainOptions)
trained.printSummary
```

Run it with:

```bash
scripts/lake.sh -Kcuda=true exe torchlean mamba --device cuda --tiny-shakespeare \
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
