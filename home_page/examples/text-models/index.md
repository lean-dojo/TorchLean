---
title: Text Models Walkthrough
---

In this walkthrough, we'll train a small GPT-style transformer and a Mamba model on text.
We'll start with a text file, turn it into next-token prediction samples, and pass those samples
to TorchLean's trainer. With the GPT example, we'll also save a checkpoint, reload it, and
generate text from a prompt.

Run the shell commands from the repository root. Since the CUDA examples need a compatible LibTorch
SDK and GPU, follow the [installation guide]({{ '/installation/' | relative_url }}) first.

## Load the text

Let's start with Tiny Shakespeare, stored as a single UTF-8 text file, which the download script
puts under `data/real/text/`. Run this once before training (the Lean example reports an error if
the file is missing):

```bash
python3 scripts/datasets/download_example_data.py --tiny-shakespeare
```

To use your own text, pass `--data-file path/to/file.txt` instead of `--tiny-shakespeare`. Either
way, the GPT-2 example reads the text through `Text.Corpus.takeUTF8Input`.

## Tokenization: Bytes First, BPE When Requested

We'll use byte tokens first. `Text.Tokenizer.byte` maps every UTF-8 byte to one token id in
`[0, 256)`, which gives us a fixed vocabulary of 256 entries without downloading a separate
tokenizer.

Our runnable GPT example uses all 256 byte ids, so training and generation share the same
vocabulary, and the byte tokenizer can decode generated ids without a lossy modulo mapping. The
model width and context length keep the example small.

The `text_gpt2` command can also use GPT-2 BPE files. Supply a matching `vocab.json` and
`merges.txt` under `data/real/gpt2/` (the dataset downloader doesn't install tokenizer files). Here
we reuse the Tiny Shakespeare corpus downloaded above:

```bash
scripts/lake.sh -Kcuda=true exe torchlean text_gpt2 --device cuda \
  --data-file data/real/text/tiny_shakespeare.txt \
  --bpe-vocab data/real/gpt2/vocab.json \
  --bpe-merges data/real/gpt2/merges.txt \
  --allow-small-data --steps 1 --generate 0
```

This command trains from randomly initialized weights. The BPE files define how text is split into
tokens, and the Lean command maps the observed BPE ids into a compact local vocabulary of at most
512 entries (token ids outside that vocabulary map to entry zero). It doesn't load pretrained GPT-2
weights or use the full GPT-2 output vocabulary.

## Supervised Examples: Next-Token Prediction As Tensors

Now we need to pair each input with the tokens we want the model to predict. We'll store the
input and target tensors together in a `Sample.Supervised` value.

For GPT-2, the sample is a one-hot matrix for causal language modeling:

```lean
abbrev input : Shape := [batchSize, contextLength, vocabularySize]
abbrev output : Shape := input
```

`Data.CausalLM.oneHotSample` converts a token tensor whose final axis has length
$\mathtt{seqLen}+1$ into $(x,y)$: input tokens and the same window shifted by one position as the
target. Whether the sample is batched depends on the leading shape.

We construct a sample with:

```lean
def batchSampleFromTokenIds (idsByBatch : Tensor Nat [batchSize, contextLength + 1]) :
    Sample.Supervised Float input output :=
  Data.CausalLM.oneHotSample (α := Float) [batchSize] contextLength vocabularySize
    (idsByBatch.map byteIndex)
```

`GPT2.samplesFromCorpus` byte-encodes the corpus once and retains the token tensor in the sample
stream's closure. Each requested sample takes windows from that tensor, then constructs the one-hot
inputs and shifted targets. So the stream doesn't re-encode the complete text for every sample,
though its window offsets and space-byte padding determine which tokens each sample contains.

<a id="gpt-2"></a>

## GPT-2: A Small Causal Transformer

We'll build a small causal transformer with `nn.CausalTransformer.oneHot`. Our example uses
the GPT-style architecture, not pretrained GPT-2 weights.

We can build the transformer with this Lean definition:

```lean
def model : nn.Builder (nn.Sequential input output) :=
  nn.CausalTransformer.oneHot
    { sequenceLength := contextLength
      vocabularySize := vocabularySize
      headCount := attentionHeads
      headWidth := attentionHeadWidth
      feedForwardWidth := feedForwardWidth
      layerCount := transformerLayers }
    (batchShape := [batchSize])
```

The configuration describes the Transformer itself. For this particular training run, `batchShape`
supplies the leading batch shape, but the same constructor also accepts an unbatched sequence or
several leading collection axes.

With the model defined, we can train it using the same API as the quickstarts (these excerpts use
the constants and options declared in the linked example source):

```lean
let trainer := Trainer.create model <|
  { runtime with
      optimizer := optim.adam { lr := options.training.lr }
      objective := oneHotCrossEntropy 2 }
let trained ← trainer.train (Data.fromStream samples)
  { steps := options.training.steps
    samplesPerStep := options.training.batchSize
    loadCheckpoint? := options.checkpoint.loadCheckpoint?
    saveCheckpoint? := options.checkpoint.saveCheckpoint? }
trained.printSummary
```

Starting from the parsed `--device` and `--seed` flags, the record update adds the optimizer and
the loss. `TrainOptions.steps` is required, and `samplesPerStep` is the number of dataset items
whose gradients are averaged before one update. In the surrounding code, we build a bank of token
windows from the corpus, report before/after predictions, save checkpoints when requested, and
sample text from the trained prediction closure.

For generation, we pass the model's score callback to `Text.generate`, which keeps the context
window, applies the repetition penalty, and appends the sampled tokens to the prompt. `Text.next`
selects a single token using the same options. For lower-level experiments, `Text.greedy?`,
`Text.topK`, and `Text.sample?` work on a tensor of scores (the `?` means selection can fail, for
example if every allowed score is NaN).

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

`gpt2_saved` loads a state pack, checks its checkpoint format and tensor shapes,
and samples without opening an optimizer. Those checks do not establish architecture identity.

With `--save-checkpoint`, the GPT-2 command writes a checkpoint, and the inference example reloads
the same shape-indexed state before sampling. If the shape list doesn't match the model, loading
fails before the weights are used.

<a id="mamba"></a>

## Mamba: State-Space Text In The Same Runtime

Next, let's try a different sequence model. `TorchLeanExamples.Models.Sequence.Mamba` uses byte
tokens with a compact state-space block in place of attention, and we can keep the same trainer,
automatic differentiation, and JSON loss logs.

Each token updates a recurrent state instead of attending to every previous token, and the model
turns those states into next-token scores. We'll train it with the same cross-entropy loss and save
the same kind of loss log as the transformer.

We'll configure one block with width `modelWidth`:

```lean
abbrev modelConfig : nn.Mamba.Config :=
  { vocabularySize := vocabularySize
    modelWidths := [modelWidth] }

abbrev input : Shape := modelConfig.inputShape contextLength
abbrev output : Shape := modelConfig.outputShape contextLength

def model : nn.Builder (nn.Sequential input output) :=
  nn.Mamba.languageModel modelConfig contextLength
```

Then we pass it to the trainer:

```lean
let trainer := Trainer.create model <|
  { runtime with
      optimizer := optim.adam { lr := options.training.lr }
      objective := oneHotCrossEntropy 1 }
let trained ← trainer.train (Data.fromStream samples) (options.training.trainOptions)
trained.printSummary
```

Run it with:

```bash
scripts/lake.sh -Kcuda=true exe torchlean mamba --device cuda --tiny-shakespeare \
  --steps 1 --windows 1 --generate 0
```

## Inspecting Runs In The Lean Editor

For interactive inspection, open `TorchLeanExamples.Models.Sequence.GPT2` or
`TorchLeanExamples.Models.Sequence.Mamba` in VS Code with the Lean Infoview enabled. The relevant
widgets can display saved training logs, tensor summaries, inferred shapes, and debug traces next
to the Lean source.

A concrete starting point is the training-log widget, which renders a saved JSON loss log in the
Infoview (it's documented near the top of `TorchLeanExamples.Models.Sequence.GPT2`).

Source entry points:

- [`TorchLeanExamples.Models.Sequence.GPT2`](https://github.com/lean-dojo/TorchLean/blob/main/TorchLeanExamples/Models/Sequence/GPT2.lean)
- [`TorchLeanExamples.Models.Sequence.GPT2Saved`](https://github.com/lean-dojo/TorchLean/blob/main/TorchLeanExamples/Models/Sequence/GPT2Saved.lean)
- [`TorchLeanExamples.Models.Sequence.Mamba`](https://github.com/lean-dojo/TorchLean/blob/main/TorchLeanExamples/Models/Sequence/Mamba.lean)
- [Three End-to-End Case Studies]({{ '/blueprint/Examples-and-Applications/Three-End-to-End-Case-Studies/' | relative_url }})
