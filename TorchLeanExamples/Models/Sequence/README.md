# Sequence Examples

Here are the runnable sequence models: recurrent networks, Transformer blocks, causal language
models, a Mamba-style model, and a small arithmetic curriculum. They are the model examples that
handle tokenization, sequence length, causal masking, recurrent state, cache-like behavior, and
generation options.

## Main Entry Points

- `Recurrent.lean` (`scripts/lake.sh exe torchlean rnn ...` and `torchlean lstm ...`): byte-window
  text training with a vanilla RNN or an LSTM cell and a time-distributed head.
- `Transformer.lean` (`scripts/lake.sh exe torchlean transformer ...`): a compact causal Transformer
  trained on shifted, bucketed byte windows.
- `CharGPT.lean` (`scripts/lake.sh exe torchlean chargpt ...`): character-level GPT training with a
  two-update smoke preset and a full Tiny Shakespeare lecture preset.
- `GPT2.lean` (`scripts/lake.sh exe torchlean gpt2 ...`): a compact byte-level GPT-2-style causal
  Transformer with save/reload support.
- `GPT2Saved.lean` (`scripts/lake.sh exe torchlean gpt2_saved ...`): load weights saved by `gpt2`
  and sample.
- `TextGPT2.lean` (`scripts/lake.sh exe torchlean text_gpt2 ...`): CUDA corpus trainer with
  byte-level tokens or GPT-2 BPE.
- `GPTAdder.lean` (`scripts/lake.sh exe torchlean gpt_adder ...`): a synthetic addition curriculum
  for next-token models.
- `Mamba.lean` (`scripts/lake.sh exe torchlean mamba ...`): compact text training for the
  Mamba-style state model.

## Bounds

- `GPT2BlockBound.lean`: a theorem file, not a runnable command. It evaluates the transformer
  block bound of `TorchLean/Verify/RuntimeApprox/TransformerBlock.lean` under the nearest rounding
  of `TorchLean/Verify/RuntimeApprox/TransformerBlock/Nearest.lean` at GPT-2 small in `binary32`.

## Why There Are Several GPT Paths

They test different pieces of the stack.

- `chargpt` is the smallest path, a single file that shows the data flow and keeps a fast
  sequence-model check in the runner.
- `gpt2` is the compact byte-level causal Transformer, with save and reload through `gpt2_saved`.
- `text_gpt2` trains on larger corpora with raw byte tokens or GPT-2 BPE tokenizer files and is
  meant for CUDA runs.
- `gpt_adder` checks sequence models, optimizers, and generation plumbing on a tiny arithmetic
  next-token curriculum.

## Token And Artifact Boundaries

Sequence models depend on conventions that are easy to hide. The docs and command output should
make these visible:

- tokenizer choice: byte tokens, character tokens, or GPT-2 BPE files
- sequence length and batch shape
- causal mask semantics
- shifted target construction for next-token prediction
- generation settings such as temperature, top-k, repeat penalty, and seed
- saved parameter paths for reloadable runs

The generating byte models (`gpt2`, `text_gpt2`, and `mamba`) keep all 256 UTF-8 byte ids. The
BPE path loads GPT-2's `vocab.json` and `merges.txt` from `--bpe-vocab` and `--bpe-merges`, then
uses the shared `Text.VocabularyProjection` to keep the first observed ids in its 512-entry output
vocabulary. Local id zero is the unknown-token slot. Saved parameters are shape-checked before
sampling, so their tensor dimensions must agree with the model's vocabulary and configuration.

`chargpt` uses `nn.CausalTransformer.indexed`, and its batches hold bounded
`Tensor (Fin vocab) [batch, seqLen]` token IDs. A tokenizer may first produce
`Tensor Nat [batch, seqLen]`, and `Tensor.checkIndices` validates that boundary before the model
runs. The embedding layer gathers table rows, and repeated IDs scatter-add into the same
weight-gradient row, so no one-hot token vectors are allocated. The `transformer`, `gpt2`,
`text_gpt2`, and `gpt_adder` commands use `nn.CausalTransformer.oneHot` with
floating-point one-hot inputs.

The training-only bucketed demonstrations and generating models use these conventions:

| Command | Context length | Byte-token convention |
| --- | --- | --- |
| `rnn`, `lstm` | 4 | byte modulo 8 |
| `transformer` | 4 | byte modulo 4 |
| `gpt2`, `text_gpt2` byte mode | 4 | all 256 byte ids |
| `mamba` | 4 | all 256 byte ids |

The RNN, LSTM, and Transformer buckets demonstrate training on shifted windows. Character GPT
derives its alphabet from the corpus. The generating examples share one tensor sampling loop for
context cropping, padding, repetition penalties, and token selection, and tokenizer arrays appear
only at the text serialization boundary. Corpus schedules are indexed streams, so training never
allocates one-hot tensors for all future updates in advance.

`GPT2.samplesFromCorpus` encodes the corpus once and keeps its token tensor in the sample stream.
Each request selects a window from that tensor and builds its one-hot input and shifted target.
Tokenization happens once per stream, and window selection, padding, and target construction
happen for each requested sample.

Useful commands:

```bash
scripts/lake.sh exe torchlean rnn --device cpu --steps 1
scripts/lake.sh -Kcuda=true exe torchlean transformer --device cuda --tiny-shakespeare --steps 1
scripts/lake.sh -Kcuda=true exe torchlean gpt2 --device cuda --steps 10 --generate 0
scripts/lake.sh -Kcuda=true exe torchlean text_gpt2 --device cuda \
  --data-file data/real/text/tinystories_valid.txt --allow-small-data --steps 1 --generate 0
scripts/lake.sh -Kcuda=true exe torchlean mamba --device cuda \
  --tiny-shakespeare --steps 1 --windows 1 --generate 0
```

For a save/reload check:

```bash
scripts/lake.sh -Kcuda=true exe torchlean gpt2 --device cuda --tiny-shakespeare \
  --steps 1 --windows 1 --generate 0 --save-checkpoint /tmp/gpt2.state

scripts/lake.sh -Kcuda=true exe torchlean gpt2_saved --device cuda \
  --checkpoint /tmp/gpt2.state --prompt "ROMEO:" --generate 16
```

## What To Inspect

| Artifact | Why it matters |
| --- | --- |
| Training log JSON | Shows loss over steps and records run metadata. |
| State checkpoint | Tests the shape-indexed parameter boundary used by reload/sampling workflows. |
| Generated text | Qualitative preview of the runtime path. |
| Tokenizer files | Define the vocabulary boundary for BPE models. |
| Lean source shapes | Show batch, sequence, vocabulary, and logits dimensions at the type level. |

Sequence models also raise verification questions: causal masks, KV-cache position contracts,
tokenizer boundaries, and batch invariance. Those contracts live in the spec and proof layers and
in `TorchLeanExamples/BugZoo`. The commands here are runtime producers and regression examples, so
cite theorem statements from the files that own the contracts.

Time-series forecasting with an LSTM layer lives in
`TorchLeanExamples/Models/Supervised/LSTMRegression.lean`, because its task is supervised
forecasting from windows.
