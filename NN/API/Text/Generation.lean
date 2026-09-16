/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Text.Options
public import Batteries.Data.BinomialHeap.Basic

/-!
# Text Generation

Score filtering, top-k sampling, logit extraction, decoding, and causal masks used by language-model
examples.
-/

@[expose] public section

namespace TorchLean
namespace text

open Spec TorchLean.Tensor

namespace Internal

/-- A valid token id together with its unscaled logit. -/
structure ScoredToken (vocabularySize : Nat) where
  /-- Token id in the model vocabulary. -/
  token : Fin vocabularySize
  /-- Logit, before temperature scaling. Selection excludes NaNs. -/
  score : Float

/-- Lower scores rank worse; equal scores prefer the smaller token id. -/
def worseOrEqual {vocabularySize : Nat}
    (left right : ScoredToken vocabularySize) : Bool :=
  left.score < right.score || (left.score == right.score && left.token.val ≥ right.token.val)

/-- Read one allowed, non-NaN logit without building an index tensor. -/
def scoredTokenAt? {vocabularySize : Nat} (scores : Tensor Float [vocabularySize])
    (allowToken : Fin vocabularySize → Bool) (index : Nat) :
    Option (ScoredToken vocabularySize) :=
  if h : index < vocabularySize then
    let token : Fin vocabularySize := ⟨index, h⟩
    let score := scores[token]
    if allowToken token && !score.isNaN then some { token, score } else none
  else none

/-- Keep only the best `k` logits in a bounded heap, then return them in descending order. -/
def topCandidates {vocabularySize : Nat} (scores : Tensor Float [vocabularySize]) (k : Nat)
    (allowToken : Fin vocabularySize → Bool) : Array (ScoredToken vocabularySize) := Id.run do
  if k = 0 then return #[]
  let mut heap : Batteries.BinomialHeap (ScoredToken vocabularySize) worseOrEqual := .empty
  let mut count := 0
  for index in [:vocabularySize] do
    if let some candidate := scoredTokenAt? scores allowToken index then
      if count < k then
        heap := heap.insert candidate
        count := count + 1
      else if let some worst := heap.head? then
        if !worseOrEqual candidate worst then
          heap := heap.tail.insert candidate
  return heap.toArray.reverse

/-- Find the best indexed candidate in one pass, breaking ties by the smaller token id. -/
def bestCandidate? {vocabularySize : Nat} (count : Nat)
    (candidateAt? : Nat → Option (ScoredToken vocabularySize)) :
    Option (ScoredToken vocabularySize) := Id.run do
  let mut best := none
  for index in [:count] do
    if let some candidate := candidateAt? index then
      if best.all (fun previous => !worseOrEqual candidate previous) then
        best := some candidate
  return best

end Internal

/--
Return up to `k` allowed, non-NaN score indices, largest first.

The result has `min k vocabularySize` slots; unused trailing slots contain `none`. Equal scores,
including signed zeros and infinities, prefer smaller token ids. A bounded heap uses
`O(vocabularySize * log(k + 1))` time and `O(min k vocabularySize)` auxiliary storage.

The optional predicate filters token ids without replacing their scores by a finite sentinel. This
matters when model logits are unbounded: a disallowed token must never become selectable merely
because every allowed score is smaller than an arbitrary masking constant.

Example:
```lean
-- Highest scores first. `allowToken` filters rather than masking, so a banned token cannot become
-- selectable just because the masking constant happened to sit above every real score.
def best (scores : Tensor Float [256]) : Tensor (Option (Fin 256)) [8] :=
  text.topKTokens scores 8 (allowToken := fun token => token.val < 128)
```
-/
def topKTokens
    {vocabularySize : Nat}
    (scores : Tensor Float [vocabularySize])
    (k : Nat)
    (allowToken : Fin vocabularySize → Bool := fun _ => true) :
    Tensor (Option (Fin vocabularySize)) [min k vocabularySize] :=
  let candidates := Internal.topCandidates scores (min k vocabularySize) allowToken
  Tensor.ofFn fun index => candidates[index.val]?.map (·.token)

/--
Greedy `argmax`, or `none` when no allowed non-NaN token exists. Equal scores prefer smaller token
ids. This scans the vocabulary once with constant auxiliary storage.

Example:
```lean
-- `none` means no allowed non-NaN score exists, which is a failure worth seeing rather than a
-- silent fall back to token zero.
def next (scores : Tensor Float [256]) : Option (Fin 256) :=
  text.greedyToken? scores
```
-/
def greedyToken?
    {vocabularySize : Nat}
    (scores : Tensor Float [vocabularySize])
    (allowToken : Fin vocabularySize → Bool := fun _ => true) :
    Option (Fin vocabularySize) :=
  (Internal.bestCandidate? vocabularySize (Internal.scoredTokenAt? scores allowToken)).map (·.token)

namespace Internal

/--
Apply a repetition penalty by subtracting
$\mathrm{repeatPenalty}\,\mathrm{count}(\mathrm{token})$ for tokens
appearing in `recent`.

This is a local sampling heuristic; it is not the same as the presence or frequency penalties used
by hosted APIs, but it gives examples a deterministic way to discourage immediate repetition.

Internal on purpose: `chooseNextToken` applies it for you, and calling it out of order (after the
softmax rather than on the logits) would silently change the sampling distribution.
-/
def penalizeRepeats
    {vocabularySize recentCount : Nat}
    (scores : Tensor Float [vocabularySize])
    (recentTokens : Tensor Nat [recentCount])
    (repeatPenalty : Float) :
    Tensor Float [vocabularySize] :=
  if repeatPenalty <= 0.0 then
    scores
  else
    let counts := recentTokens.foldl
      (fun (counts : Std.HashMap Nat Nat) token =>
        counts.insert token ((counts[token]?).getD 0 + 1)) {}
    Tensor.ofFn fun token =>
      scores[token] - repeatPenalty * Float.ofNat ((counts[token.val]?).getD 0)

end Internal

/--
True for byte tokens that a terminal can print: the printable ASCII range plus newline.

Named with the `is` prefix that Lean core uses for `Char.isAlpha` and friends, so that reading
`if isPrintableAscii token` at a call site tells you a `Bool` comes back.
-/
def isPrintableAscii (token : Nat) : Bool :=
  token = 10 || (32 ≤ token && token ≤ 126)

namespace Internal

/-- Escape one byte token for display inside a quoted string. Used only by `formatByteTokens`. -/
def escapeByteToken (token : Nat) : String :=
  if token = 10 then "\\n"
  else if token = 9 then "\\t"
  else if token = 34 then "\\\""
  else if token = 92 then "\\\\"
  else if 32 ≤ token && token ≤ 126 then
    String.singleton (Char.ofNat token)
  else
    let hex : Array Char := #['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']
    let highDigit := (token / 16) % 16
    let lowDigit := token % 16
    "\\x" ++ String.singleton (hex.getD highDigit '0') ++
      String.singleton (hex.getD lowDigit '0')

end Internal

/--
Escape byte ids as a one-line quoted display string.

Example:
```lean
-- Generated bytes are not always valid UTF-8, so display escapes them instead of guessing:
-- `#[104, 105, 10]` prints as `"hi\n"`.
def display {n : Nat} (tokens : Tensor Nat [n]) : String := text.formatByteTokens tokens
```
-/
def formatByteTokens {n : Nat} (tokens : Tensor Nat [n]) : String :=
  tokens.foldl (fun result token => result ++ Internal.escapeByteToken token) "\"" ++ "\""

namespace Internal

/-- Stable softmax weight relative to a finite maximum and a positive finite temperature. -/
def samplingWeight (score maximum temperature : Float) : Float :=
  let difference := score - maximum
  -- Opposite finite extremes can overflow during subtraction. Scale those first instead.
  let exponent :=
    if difference.isInf && score.isFinite then score / temperature - maximum / temperature
    else difference / temperature
  MathFunctions.exp exponent

/-- Sample an indexed candidate sequence, storing weights once and never sorting the sequence. -/
def sampleCandidates? {vocabularySize : Nat} (count : Nat)
    (candidateAt? : Nat → Option (ScoredToken vocabularySize))
    (temperature : Float) (seed counter : Nat) : Option (Fin vocabularySize) := Id.run do
  let some maximum := bestCandidate? count candidateAt? | return none
  if maximum.score.isInf then return some maximum.token
  let mut weights : Array Float := Array.emptyWithCapacity count
  let mut total := 0.0
  for index in [:count] do
    let weight := match candidateAt? index with
      | some candidate => samplingWeight candidate.score maximum.score temperature
      | none => 0.0
    weights := weights.push weight
    total := total + weight
  let key := Spec.Random.keyOf seed counter
  let denom : Nat := (2 : Nat) ^ 32
  let u := Float.ofNat (Spec.Random.sampleNat key 0 denom) / Float.ofNat denom
  let target := u * total
  let mut cumulative := 0.0
  for index in [:count] do
    cumulative := cumulative + weights[index]!
    if target < cumulative then
      return (candidateAt? index).map (·.token)
  return some maximum.token

end Internal

/--
Sample an allowed token id using a positive finite temperature and top-k sampling.

NaNs and disallowed tokens are excluded. Ties at the cutoff prefer smaller token ids. If the
maximum is infinite, return its smallest token id directly (also when every allowed score is
negative infinity), avoiding undefined softmax normalization.

`topK = 0` or `topK ≥ vocabularySize` samples the full vocabulary in token-id order in linear time.
A smaller positive cutoff samples the bounded heap's output in descending score order. Randomness
is deterministic given the scores, filter, temperature, cutoff, seed, and counter; the two traversal
orders can produce different tokens for the same random draw.
-/
def sampleTopKToken? {vocabularySize : Nat} (scores : Tensor Float [vocabularySize])
    (temperature : Float) (topK seed counter : Nat)
    (allowToken : Fin vocabularySize → Bool := fun _ => true) :
    Option (Fin vocabularySize) :=
  if !temperature.isFinite || temperature <= 0.0 then none
  else if topK = 0 || vocabularySize ≤ topK then
    Internal.sampleCandidates? vocabularySize
      (Internal.scoredTokenAt? scores allowToken) temperature seed counter
  else
    let candidates := Internal.topCandidates scores topK allowToken
    Internal.sampleCandidates? candidates.size (fun index => candidates[index]?)
      temperature seed counter

/--
Select the next token, rejecting an empty allow-list, a non-finite or negative repetition penalty,
or an invalid sampling temperature. Greedy decoding (`topK = 1`) ignores temperature.

Example:
```lean
-- One decoding policy shared by every text example: repeat penalty first, then greedy or top-k
-- sampling. `counter` keeps each step's randomness distinct while staying reproducible.
def next (scores : Tensor Float [256])
    (options : text.GenerationOptions) (step : Nat) :
    Except String (Fin 256) :=
  text.chooseNextToken scores options (counter := step) (recentTokens := Tensor.full [0] 0)
```
-/
def chooseNextToken {vocabularySize recentCount : Nat} (scores : Tensor Float [vocabularySize])
    (options : GenerationOptions) (counter : Nat) (recentTokens : Tensor Nat [recentCount])
    (allowToken : Fin vocabularySize → Bool := fun _ => true) :
    Except String (Fin vocabularySize) := do
  unless options.repeatPenalty.isFinite && 0.0 <= options.repeatPenalty do
    throw "generation repeat penalty must be finite and nonnegative"
  unless options.topK = 1 || (options.temperature.isFinite && 0.0 < options.temperature) do
    throw "generation temperature must be finite and positive"
  let scores := Internal.penalizeRepeats scores recentTokens options.repeatPenalty
  let selected? :=
    if options.topK = 1 then
      greedyToken? scores allowToken
    else
      sampleTopKToken?
        scores options.temperature options.topK options.seed counter allowToken
  match selected? with
  | some token => pure token
  | none => throw "no allowed non-NaN token score is available for generation"

/--
Autoregressively extend token ids with a model-provided score callback.

The callback receives an exact-length context window and the sequence position whose logits should
be used for the next token. The shared policy crops to the last `sequenceLength` tokens, pads,
applies repeat penalties, samples by top-k/temperature, and writes one token per step.
The output includes the prompt followed by exactly `newTokenCount` tokens. A zero-length model
context is rejected when generation is requested.

Example:
```lean
-- The model arrives as a callback: given an exact-length window and the position whose logits to
-- read, return that position's scores. Cropping, padding, penalties, and sampling stay here.
def generate (options : text.GenerationOptions)
    (scoreWindow : Tensor Nat [16] → Fin 16 → IO (Tensor Float [256])) :
    IO (Tensor Nat [(text.Tokenizer.byte.encode options.prompt).size + options.newTokenCount]) :=
  text.autoregressiveTokenIds 16 (paddingTokenId := 0)
    (promptTokens := Tensor.from (text.Tokenizer.byte.encode options.prompt))
    (options := options) (scoreWindow := scoreWindow)
```
-/
def autoregressiveTokenIds {vocabularySize promptLength : Nat}
    (sequenceLength paddingTokenId : Nat)
    (promptTokens : Tensor Nat [promptLength])
    (options : GenerationOptions)
    (scoreWindow :
      Tensor Nat [sequenceLength] →
      Fin sequenceLength →
      IO (Tensor Float [vocabularySize]))
    (allowToken : Fin vocabularySize → Bool := fun _ => true) :
    IO (Tensor Nat [promptLength + options.newTokenCount]) := do
  let mut tokens : Tensor Nat [promptLength + options.newTokenCount] :=
    Tensor.ofFn fun index =>
      if h : index.val < promptLength then promptTokens[(⟨index.val, h⟩ : Fin promptLength)]
      else paddingTokenId
  if sequenceLengthIsZero : sequenceLength = 0 then
    if options.newTokenCount != 0 then
      throw (IO.userError "generation requires a nonempty model context")
    return tokens
  else
    for step in List.finRange options.newTokenCount do
      let activeLength := promptLength + step.val
      let start := activeLength - sequenceLength
      let contextCount := min activeLength sequenceLength
      let predictionPosition : Fin sequenceLength :=
        ⟨contextCount - 1, by omega⟩
      let padded := Tensor.window tokens sequenceLength start paddingTokenId
      let scores ← scoreWindow padded predictionPosition
      let recentCount := min activeLength options.repeatWindow
      let recentTokens :=
        Tensor.window tokens recentCount (activeLength - recentCount) paddingTokenId
      let nextToken ← IO.ofExcept <|
        chooseNextToken scores options step.val recentTokens allowToken
      let outputIndex : Fin (promptLength + options.newTokenCount) := ⟨activeLength, by omega⟩
      tokens := Tensor.set tokens (outputIndex, PUnit.unit) nextToken.val
    return tokens

/-!
The next six declarations come in three pairs: an operation on `(sequenceLength × vocabularySize)`
logits, and the same operation on a batch, which takes an extra `batchIndex` and works on one row.
The batched member of a pair is the unbatched name with a `batch` prefix, always in that position,
so knowing one spelling gives you the other.
-/

/-- Extract the vocabulary-score row at one statically valid sequence position. -/
def logitScoresAt {α : Type} [TorchLean.Storage α] {sequenceLength vocabularySize : Nat}
    (logits : Tensor α [sequenceLength, vocabularySize])
    (position : Fin sequenceLength) : Tensor α [vocabularySize] :=
  Tensor.get logits position

/-- Extract a vocabulary-score row from batched logits. -/
def batchLogitScoresAt
    {α : Type} [TorchLean.Storage α]
    {batchSize sequenceLength vocabularySize : Nat}
    (logits : Tensor α [batchSize, sequenceLength, vocabularySize])
    (batchIndex : Fin batchSize)
    (position : Fin sequenceLength) :
    Tensor α [vocabularySize] :=
  logitScoresAt (Tensor.get logits batchIndex) position

/--
Decode a matrix of token logits by taking `argmax` independently at each sequence position.

The shape is `(sequenceLength × vocabularySize)`, i.e. one logits vector per token position. This
helper is for inspection/debugging and is not differentiable.
-/
def argmaxTokens {α : Type} [TorchLean.Storage α] [LT α]
    [DecidableRel ((· > ·) : α → α → Prop)]
    {sequenceLength vocabularySize : Nat} (logits : Tensor α [sequenceLength, vocabularySize]) :
    Tensor Nat [sequenceLength] :=
  Tensor.ofFn fun position : Fin sequenceLength =>
    match TorchLean.Metrics.argmax? (α := α) (Tensor.get logits position) with
    | some token => token.val
    | none => 0

/-- Decode `(sequenceLength × vocabularySize)` logits as text using a tokenizer. -/
def decodeArgmaxLogits {α : Type} [TorchLean.Storage α] [LT α]
    [DecidableRel ((· > ·) : α → α → Prop)]
    (tokenizer : Tokenizer)
    {sequenceLength vocabularySize : Nat}
    (logits : Tensor α [sequenceLength, vocabularySize]) :
    String :=
  tokenizer.decode ((argmaxTokens (α := α) logits).to (Array Nat))

/-- Extract `batchIndex` from batched logits and return the per-position argmax token ids. -/
def batchArgmaxTokens {α : Type} [TorchLean.Storage α] [LT α]
    [DecidableRel ((· > ·) : α → α → Prop)]
    {batchSize sequenceLength vocabularySize : Nat}
    (logits : Tensor α [batchSize, sequenceLength, vocabularySize])
    (batchIndex : Fin batchSize) :
    Tensor Nat [sequenceLength] :=
  argmaxTokens (α := α) (Tensor.get logits batchIndex)

end text
end TorchLean
