---
title: Text Models Walkthrough
usemathjax: true
---

The text-model examples run from corpus to continuation. The models are small, but the workflow is
complete: read a text file, build next-token training examples, run a training step, save
parameters, reload them, and sample from the saved model.

The useful part is visibility. Tokenization, sequence length, causal windows, parameter shapes,
generation settings, and saved logs all appear as artifacts the reader can inspect.

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

## Tokenization: Bytes on Purpose

These examples tokenize bytes directly. Every UTF-8 byte is a token, so the vocabulary is fixed
and tiny (256). That choice is practical:

- there is no external BPE model file to keep in sync,
- there is no “which tokenizer version did you mean?” ambiguity,
- it makes boundary mistakes easy to spot (and easy to turn into case studies).

The code-level version is:

```lean
def vocab : Nat := text.Tokenizer.byte.vocabSize
```

That line is from `NN.Examples.Models.Sequence.Gpt2`. It means the tutorial model is not silently
depending on a tokenizer JSON, BPE merge table, or Hugging Face cache entry. The byte tokenizer is
small enough that the input and target tensors can be written down directly.

## Supervised Examples: Next-Token Prediction As Tensors

The training data is represented directly as typed supervised samples. The examples build explicit
`SupervisedSample` values whose shapes say what they are.

For GPT-2, the sample is a one-hot matrix for causal language modeling:

```lean
abbrev σ : Shape := shape![batch, seqLen, vocab]
abbrev τ : Shape := σ
```

The function `Data.causalLmOneHotSample` converts a `(seqLen + 1)` token window into `(x, y)`:
input tokens and the same window shifted by one position as the target. The trainer consumes those
typed tensors directly.

The Lean sample constructor is explicit:

```lean
def mkSampleFromTokenIds (toks : List Nat) : SupervisedSample Float σ τ :=
  Data.causalLmOneHotSample (α := Float) batch seqLen vocab toks (padId := 32)
```

The tutorial value is the absence of hidden dataloader convention: a supervised example is
literally a pair of typed tensors.

<a id="gpt-2"></a>

## GPT-2: A Small Causal Transformer

`NN.Examples.Models.Sequence.Gpt2` wires up a miniature causal transformer from reusable layers
(`nn.models.CausalTransformerOneHot`). The default configuration is compact enough for a local run,
but it is still large enough to learn short structure from Tiny Shakespeare.

The model declaration is a normal Lean value:

```lean
def model : nn.M (nn.Sequential σ τ) :=
  nn.models.CausalTransformerOneHot
    { batch := batch
      seqLen := seqLen
      vocab := vocab
      numHeads := numHeads
      headDim := headDim
      ffnHidden := ffnHidden
      layers := layers }
```

The training loop stays on the same public surface used by the simpler quickstarts:

```lean
let trainer := Trainer.new model <|
  Trainer.Config.fromRunConfig run (.crossEntropy)
let trained ← trainer.train data train.options probes
trained.printSummary
```

The surrounding code builds a bank of token windows from the corpus, reports before/after
predictions, saves checkpoints when requested, and samples text from the trained prediction closure.

Sampling is also explicit: the example computes logits, applies temperature and top-k filtering, and
chooses the next token. If you’ve ever written a small “Karpathy-style” sampler, the code will look
familiar.

Try the short CUDA run:

```bash
lake exe -K cuda=true torchlean gpt2 --cuda --fast-kernels --tiny-shakespeare \
  --steps 300 --windows 32 --lr 0.001 --prompt "ROMEO:" --generate 220 \
  --temperature 0.85 --top-k 24 --repeat-penalty 1.25 --repeat-window 24 \
  --sample-seed 11 --log data/model_zoo/gpt2_trainlog.json
```

For a tiny runtime check, keep the same CUDA path and shrink the workload:

```bash
lake exe -K cuda=true torchlean gpt2 --cuda --tiny-shakespeare --steps 1 --windows 1 --generate 0
```

## Saving and Reloading Parameters

TorchLean’s checkpoint format is kept simple: a model’s parameters are a shape-indexed
pack, and the save/load operations round-trip exact IEEE-754 bit patterns through JSON.

That is why `gpt2_saved` is a separate example: it loads a parameter pack, checks that the shapes
match the model architecture, and runs sampling without touching an optimizer.

The GPT-2 command exposes parameter export through `--save-params`; the saved-parameter example
reloads the same shape-indexed parameter pack before sampling. If the shape list no longer matches
the model, loading fails before the weights are used.

<a id="mamba"></a>

## Mamba: State-Space Text In The Same Runtime

`NN.Examples.Models.Sequence.Mamba` also runs on byte tokens, but swaps attention for a
compact state-space block. It is useful as a contrast: sequence modeling without the quadratic
attention path, using the same autograd and the same logging style.

That contrast is useful because it shows that the text pipeline is not tied to attention.

Algorithmically, the example replaces “attend over all previous tokens” with a learned recurrent
state update. Each token updates a compact state, and the model projects the resulting sequence
states to next-token logits. The surrounding training interface stays the same: corpus windows in,
logits out, cross-entropy loss, optimizer step, JSON log.

The Mamba example has the same tutorial shape:

```lean
abbrev σ : Shape := nn.models.mambaTokenMat cfg seqLen
abbrev τ : Shape := nn.models.mambaLogitMat cfg seqLen

def model : nn.M (nn.Sequential σ τ) :=
  nn.models.MambaTextLM cfg seqLen
```

and the training body is still an ordinary trainer call:

```lean
let trainer := Trainer.new model <|
  Trainer.Config.fromRunConfig run (.crossEntropy)
let trained ← trainer.train data train.options probes
trained.printSummary
```

Run it with:

```bash
lake exe -K cuda=true torchlean mamba --cuda --fast-kernels --tiny-shakespeare \
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

- [`NN.Examples.Models.Sequence.Gpt2`]({{ '/docs/NN/Examples/Models/Sequence/Gpt2.html' | relative_url }})
- [`NN.Examples.Models.Sequence.Gpt2Saved`]({{ '/docs/NN/Examples/Models/Sequence/Gpt2Saved.html' | relative_url }})
- [`NN.Examples.Models.Sequence.Mamba`]({{ '/docs/NN/Examples/Models/Sequence/Mamba.html' | relative_url }})
- [Model Zoo Deep Dive]({{ '/blueprint/Examples-and-Applications/Model-Zoo-Deep-Dive/' | relative_url }})
