---
title: Text Models Walkthrough
usemathjax: true
---

This page follows the text-model examples from corpus to continuation. The models are small, but
the workflow is complete: read a text file, build next-token training examples, run a training step,
save parameters, reload them, and sample from the saved model.

The useful part is visibility. Tokenization, sequence length, causal windows, parameter shapes,
generation settings, and saved logs all appear as artifacts the reader can inspect.

## Data: One Explicit Text File

The text demos are intentionally built around a single UTF-8 text file. In the default setup, the
download script places Tiny Shakespeare under `data/real/text/`, and the Lean example fails loudly
if the file is missing.

Run the data step once:

```bash
python3 scripts/datasets/download_example_data.py --tiny-shakespeare
```

The helper that turns “flags and paths” into an actual corpus is shared across examples. In the GPT-2
example, `takeInputText` uses `text.Corpus.takeUtf8Input` to support a “use this known corpus” flag
or an explicit `--data-file` path.

## Tokenization: Bytes on Purpose

These examples tokenize bytes directly. Every UTF-8 byte is a token, so the vocabulary is fixed
and tiny (256). That choice is practical:

- there is no external BPE model file to keep in sync,
- there is no “which tokenizer version did you mean?” ambiguity,
- it makes boundary mistakes easy to spot (and easy to turn into case studies).

In code, this shows up as:

```lean
def vocab : Nat := text.Tokenizer.byte.vocabSize
```

That line is from `NN.Examples.Models.Sequence.Gpt2`. It means the tutorial model is not silently
depending on a tokenizer JSON, BPE merge table, or Hugging Face cache entry. The byte tokenizer is
small enough that the input and target tensors can be written down directly.

## Supervised Examples: Next-Token Prediction As Tensors

The training data is represented directly as typed supervised samples. The examples build explicit
`sample.Supervised` values whose shapes say what they are.

For GPT-2, the sample is a one-hot matrix for causal language modeling:

```lean
abbrev σ : Shape := shape![batch, seqLen, vocab]
abbrev τ : Shape := σ
```

The helper `text.causalLmXYOneHotMatFloat` converts a `(seqLen + 1)` token window into `(x, y)`:
input tokens and the same window shifted by one position as the target. The rest of the training
loop consumes `sample.x` and `sample.y`.

The Lean sample constructor is explicit:

```lean
def mkSampleFromTokenIds {α : Type} [Semantics.Scalar α] [Runtime.Scalar α]
    (toks : List Nat) : API.sample.Supervised α σ τ :=
  let (x2DF, y2DF) :=
    text.causalLmXYOneHotMatFloat (seqLen := seqLen) (vocab := vocab) toks
      (padId := 32)
  let x2D : Tensor α (NN.Tensor.Shape.Mat seqLen vocab) :=
    Common.castTensor Runtime.ofFloat x2DF
  let y2D : Tensor α (NN.Tensor.Shape.Mat seqLen vocab) :=
    Common.castTensor Runtime.ofFloat y2DF
  let x : Tensor α σ := Tensor.dim (fun _bi => x2D)
  let y : Tensor α τ := Tensor.dim (fun _bi => y2D)
  API.sample.mk x y
```

This is why the example is useful as a tutorial: there is no hidden dataloader convention. A
supervised example is literally a pair of typed tensors.

<a id="gpt-2"></a>

## GPT-2: A Small Causal Transformer

`NN.Examples.Models.Sequence.Gpt2` wires up a miniature causal transformer from reusable layers
(`nn.models.causalTransformerOneHot`). The default configuration is compact enough for a local run,
but it is still large enough to learn short structure from Tiny Shakespeare.

The model declaration is a normal Lean value:

```lean
def mkModel : nn.M (nn.Sequential σ τ) :=
  nn.models.causalTransformerOneHot
    { batch := batch
      seqLen := seqLen
      vocab := vocab
      numHeads := numHeads
      headDim := headDim
      ffnHidden := ffnHidden
      layers := layers }
```

The training loop is plain:

- build a bank of token windows from the corpus,
- evaluate a cross-entropy loss on one batch,
- take an optimizer step (Adam by default),
- periodically sample text to see what the model is doing.

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

For a smoke test without CUDA:

```bash
lake exe torchlean gpt2 --tiny-shakespeare --steps 1 --windows 2 --generate 32
```

## Saving and Reloading Parameters

TorchLean’s checkpoint format is intentionally simple: a model’s parameters are a shape-indexed
pack, and the save/load helpers round-trip exact IEEE-754 bit patterns through JSON.

That is why `gpt2_saved` is a separate example: it loads a parameter pack, checks that the shapes
match the model architecture, and runs sampling without touching an optimizer.

The generic API helper for this is:

- `NN.API.TorchLean.ParamIO.saveModuleParamsBits`
- `NN.API.TorchLean.ParamIO.loadModuleParamsBits`

The GPT-2 example calls the same helpers directly:

```lean
TorchLean.ParamIO.saveModuleParamsBits
  (paramShapes := nn.paramShapes model)
  (inputShapes := [σ, τ])
  m path
```

and the saved-parameter example reloads the same shape-indexed parameter pack before sampling. If
the shape list no longer matches the model, loading fails before the weights are used.

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

def mkModel : nn.M (nn.Sequential σ τ) :=
  nn.models.mambaTextLm cfg seqLen
```

and the training entrypoint is still just a runner around `trainOnText`:

```lean
def main (args : List String) : IO UInt32 := do
  TorchLean.Module.run exeName args
    (.float (fun opts rest => do
      let (path, rest) ← Common.orThrow exeName <| RealData.parseTextFlags rest
      let (train, rest) ← Common.orThrow exeName <| parseTrainOptions rest
      Common.orThrow exeName <| CLI.requireNoArgs rest
      let input ← RealData.readTextCorpus exeName path
      let (L0, L1) ← trainOnText opts input train
      Common.writeBeforeAfterLossLogTo train.log "Mamba text training"
        train.steps L0 L1))
```

Run it with:

```bash
lake exe -K cuda=true torchlean mamba --cuda --fast-kernels --tiny-shakespeare \
  --steps 200 --windows 128 --prompt "ROMEO:" --generate 180 \
  --temperature 0.8 --top-k 16 --sample-seed 11
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
