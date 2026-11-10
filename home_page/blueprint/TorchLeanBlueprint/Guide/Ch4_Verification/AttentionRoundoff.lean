import VersoManual
import TorchLean.Tensor.Literal
import TorchLean.Verify.RuntimeApprox.OnlineSoftmax
import TorchLean.Verify.RuntimeApprox.TransformerBlock
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean

-- The online schedule is defined in `TorchLean.Softmax`, and its bounds live in one namespace,
-- opened here so that the displayed `#check` lines stay short. The FloatLib namespaces name the
-- configured binary formats that the rounding theorems quantify over.
open TorchLean
open OnlineSoftmax
open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange

-- Several signatures print wider than this file's 100-column limit, so their `leanOutput` blocks
-- ask for `whitespace := lax` and are wrapped in the source.
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Rounding Error In Attention" =>
%%%
tag := "attention-roundoff"
file := "Rounding-Error-In-Attention"
%%%

Attention computes, for each query, an average of value vectors weighted by a softmax of the
query's scores against the keys:

$$`o=\frac{\sum_j e^{s_j}v_j}{\sum_j e^{s_j}}.`

A direct implementation finds the largest score, exponentiates the shifted scores, sums them and
divides. It reads every score twice and keeps the whole row of weights. FlashAttention
{Informal.citep flashattention2022}[] reads the keys in blocks instead and keeps three running
quantities per query, so the row of weights is never stored. Over the reals the two procedures
compute the same number. In floating point they round at different places, and their errors
differ.

We state FlashAttention's procedure as a schedule of TorchLean's own softmax and attention, and
prove that it computes the two-pass value over the reals. Then we bound the schedule's rounding
error in every IEEE format, with kernel-checked numbers at 4096 keys. The last part bounds a whole
transformer block.

# The Online Softmax Schedule

For one query, the online schedule keeps a running maximum $`m` of the scores seen so far, a
normalizer $`l=\sum e^{s-m}` over those keys, and a weighted sum $`o=\sum e^{s-m}v`. A block of
keys first raises the maximum to $`m'` (the larger of $`m` and the block's scores). Both sums so
far were taken relative to $`m`, so they're multiplied by $`e^{m-m'}` before the block's own terms
are added relative to $`m'`. After the last block the output is $`o/l`. Every exponent is at most
zero, so no exponential overflows.

`List.toChunks size keys` splits the keys into consecutive blocks of `size` keys (the
last block may be shorter, and size `0` keeps all keys in one block). `Softmax.onlineStep`
is the update for one block:

```lean (name := arStep)
-- One block of the online schedule over any scalar type
-- with a `Context`.
#check @Softmax.onlineStep
```

```leanOutput arStep (whitespace := lax)
@Softmax.onlineStep : {β : Type} →
  [Context β] → (β → β) → (β → β) → (β → β) → β × β × β → List (β × β) → β × β × β
```

The three function arguments are the exponential and two roundings of each weight: one before
the weight enters the normalizer and one before it multiplies its value. In exact arithmetic
both are the identity. A later section uses them for weights stored in a narrower format.

Here are five keys, each a score and a value, in blocks of two, with the state $`(m,l,o)` after
each block:

```lean (name := arTrace)
-- Pairs of (score, value). The fold starts at the first
-- score with empty sums.
def arKeys : List (Float × Float) :=
  [(0.5, 1.0), (2.0, -1.0), (1.0, 3.0), (-1.0, 0.0),
   (3.0, 2.0)]

#eval List.toChunks 2 arKeys
#eval List.scanl
  (Softmax.onlineStep Float.exp id id) (0.5, 0, 0)
  (List.toChunks 2 arKeys)
```

```leanOutput arTrace (whitespace := lax)
[[(0.500000, 1.000000), (2.000000, -1.000000)], [(1.000000, 3.000000), (-1.000000, 0.000000)],
  [(3.000000, 2.000000)]]
```
```leanOutput arTrace (whitespace := lax)
[(0.500000, 0.000000, 0.000000),
 (2.000000, 1.223130, -0.776870),
 (2.000000, 1.640797, 0.326768),
 (3.000000, 1.603615, 2.120211)]
```

The first block raises the maximum from `0.5` to `2`, but its sums start empty, so the rescaling
changes nothing. In the second block the scores stay below `2`, so the scale is $`e^0=1` and the
block's terms are added. At the third block the maximum rises to `3`, so the normalizer `1.640797`
and the weighted sum `0.326768` are multiplied by $`e^{-1}` before the last key adds $`e^0=1` and
$`1\cdot 2`. `Softmax.onlineAverage` runs the fold and divides, and the direct formula
gives the same number to the digits shown:

```lean (name := arAverage)
-- The online average against the direct softmax average.
#eval Softmax.onlineAverage Float.exp id id 0.5
  (List.toChunks 2 arKeys)
#eval
  (arKeys.foldl (fun acc e => acc + Float.exp e.1 * e.2)
      (0 : Float)) /
    arKeys.foldl (fun acc e => acc + Float.exp e.1)
      (0 : Float)
```

```leanOutput arAverage (whitespace := lax)
1.322145
```
```leanOutput arAverage (whitespace := lax)
1.322145
```

`Tensor.softmax x dim mask schedule` and `Tensor.attention` take the schedule as an argument.
`.twoPass` (the default) finds the maximum and the normalizer before computing the weights.
`.online blocks` runs `onlineStep` over the blocks of a fiber's allowed keys, `blocks.size` keys
per block. A Boolean mask removes keys before the fold, so a blocked key enters neither the maximum
nor the normalizer, and a fiber without an allowed key is zero under both schedules. On these
scores the two schedules print the same weights:

```lean (name := arSchedules)
-- Softmax along axis 0 by each schedule, the online one in
-- blocks of two.
def arScores : Tensor Float [5] :=
  [0.5, 2.0, 1.0, -1.0, 3.0]

#eval Tensor.softmax arScores ⟨0, by decide⟩
#eval Tensor.softmax arScores ⟨0, by decide⟩ none
  (.online ⟨2⟩)
```

```leanOutput arSchedules (whitespace := lax)
[0.051187, 0.229406, 0.084394, 0.011421, 0.623591]
```
```leanOutput arSchedules (whitespace := lax)
[0.051187, 0.229406, 0.084394, 0.011421, 0.623591]
```

Six digits hide the last bits. The online normalizer is a different sequence of roundings, so in
general the two schedules need not agree bit for bit.

# The Two Schedules Agree Over The Reals

Over the reals the rescaling cancels exactly. After any prefix of blocks with maximum $`M`, the
state is $`(M,\sum e^{s-M},\sum e^{s-M}v)`, so the quotient is the softmax average whatever the
block sizes were. `attentionOnline_apply_real` states this for one entry of the online schedule
on score matrices with any leading batch axes, a mask and a multiplier on the weights. Two
corollaries state it for the public operators:

```lean (name := arReal)
-- Online and two-pass softmax and attention are equal
-- functions over the reals.
#check @Tensor.softmax_online_eq
#check @Tensor.attention_online_eq
```

```leanOutput arReal (whitespace := lax)
@Tensor.softmax_online_eq : ∀ {s : Shape} (x : Tensor ℝ s) (dim : Fin s.rank) (mask : Option
  (Tensor Bool s))
  (blocks : Softmax.BlockPartition), x.softmax dim mask (Softmax.Schedule.online blocks) =
    x.softmax dim mask
```
```leanOutput arReal (whitespace := lax)
@Tensor.attention_online_eq : ∀ {batch : Shape} {rows keys depth width : ℕ}
  (query : Tensor ℝ (batch ++ [rows, depth])) (key : Tensor ℝ (batch ++ [keys, depth]))
  (value : Tensor ℝ (batch ++ [keys, width])) (scale : ℝ) (mask : Option (Tensor Bool [rows,
    keys]))
  (weights : Option (Tensor ℝ (batch ++ [rows, keys]))) (blocks : Softmax.BlockPartition),
  query.attention key value scale mask weights (Softmax.Schedule.online blocks) =
    query.attention key value scale mask weights
```

Both equalities hold for every block size and every mask, and the attention theorem holds for every
scale and multiplier too. A real-valued theorem about the two-pass operators (such as a derivative
theorem of {ref "autograd-proofs"}[the autograd chapter]) carries over to the online schedule by
rewriting with these equalities. Neither says anything about rounding.

# Rounding Error Of The Online Pass

In floating point the cancellation isn't exact. Each rescaling rounds a subtraction, an
exponential, a product and a sum, and a key seen early is rescaled by every later block that raises
the maximum. `onlineAverage_error` bounds the computed output against the exact softmax average
$`O/S` for one query. That bound holds in every IEEE format, for every partition of the keys into
blocks of at most $`B` keys and every number of keys, with gradual underflow. Every operation is
the configured IEEE operation with nearest-even rounding and FloatLib's correctly rounded
exponential.

Our proof follows each key's weight $`e^{s-M}`, with $`M` the final running maximum. A key first
seen against the maximum $`m_k` has its shift $`s-m_k` rounded, and each later change of the
maximum that rescales it is rounded too. Each rounding is at most the unit roundoff $`u` times its
shift, and the shifts add up to $`M-s` because the maximum only grows. So the shift errors
telescope to $`u(M-s)`, the allowance of a single pass against the final maximum. On top of it, for
$`T` blocks, each key is charged $`B+3T-1` unit roundoffs in the normalizer and $`B+3T` in the
weighted sum. A later block charges the sums accumulated before it three roundings when it raises
the maximum (the exponential, the product and the sum) and only the sum when it doesn't. A key's
weight in the normalizer thus receives the relative allowance

$$`\exp\bigl(u(M-s)+u(B+3T-1)\bigr)-1,`

and the remaining terms are underflow allowances. The theorem needs finite inputs, scores at most
half the largest finite value in magnitude, values bounded by $`V`, a normalizer allowance below
the exact normalizer, and range conditions on these input-only quantities. We proved it by hand in
{src "TorchLean/Verify/RuntimeApprox/OnlineSoftmax.lean"}[the online-softmax proofs], because a
closed form needs an invariant over all the keys seen, which a generated recurrence doesn't supply.

Started at a value at least every score, as when the maximum is computed in a first pass, the
same fold is the blocked two-pass softmax. `onlineAverage_error_of_le` bounds it with $`B+T+1`
and $`B+T+2` units. The uniform bounds differ by $`2u(T-1)`: a block that raises the maximum adds
two roundings to the exponent budget of every sum accumulated before it. One block of all the keys
is the two-pass softmax with sequential sums. `onlineAverage_blocks_error` and
`onlineAverage_blocks_error_of_max` state both bounds in closed form for blocks of $`B` keys,
through Gibbs' inequality $`\sum w(M-s)\le S(\log N-\log S)` (`sum_exp_mul_sub_le`). When every
scaled shift $`u(M-s)` is at most one, the normalizer's relative error is at most
$`(e^c-1)+2ue^c\log N` plus its underflow residual.

`onlineWeight_error` bounds one weight of the online `Tensor.softmax` the same way, against the
softmax weight of its key among the allowed keys (`onlineWeight_reference_eq`). It uses the
normalizer allowance above and one rounded shift, exponential and division for the key itself.

# Bounds At 4096 Keys

Once the sizes are fixed, the closed forms become numbers. The theorems of
{src "TorchLean/Verify/RuntimeApprox/OnlineSoftmax/Binary32.lean"}[`OnlineSoftmax/Binary32.lean`]
evaluate them for a working format with the unit roundoff and underflow allowance of `binary32`,
values of magnitude at most one, and a largest finite value at least $`2^{100}`. `norm_num`
decides the rational inequalities in the kernel, from $`e^x\le 1/(1-x)` and
$`\log 2<0.6931471808`. Each theorem states that the budget and range premises of its closed form
hold at those sizes and bounds its conclusion:

:::table +header
*
  * Theorem
  * Schedule
  * Keys
  * Block size
  * Bound
*
  * `online_4096_128`
  * online
  * 4096
  * 128
  * `2.9e-5`
*
  * `twoPass_4096_128`
  * blocked two-pass
  * 4096
  * 128
  * `2.2e-5`
*
  * `online_4096_4096`
  * one block, sequential sums
  * 4096
  * 4096
  * `0.000492`
*
  * `online_16384_128`
  * online
  * 16384
  * 128
  * `6.4e-5`
:::

Keeping a running maximum costs little: at 4096 keys in blocks of 128 the online bound is about
thirty percent above the blocked two-pass bound. Block size matters more. One block of 4096 keys
sums the normalizer in one sequential fold, which charges each key about 4096 unit roundoffs
instead of about 128, and its bound is about seventeen times larger.

`online_4096_128_output` combines the first row with the closed-form theorem into a statement about
the computed output. Take any IEEE format with the precision and underflow of `binary32` and a
largest finite value at least $`2^{100}`, and any 4096 keys with finite scores of magnitude at most
$`2^{23}` and finite values of magnitude at most one. Then the online pass in blocks of 128 is
finite and within `2.9e-5` of the exact softmax average of the scores and values it is given. For
attention, those scores are computed products of queries and keys, whose own rounding adds to this
bound.

# Weights Stored In A Narrower Format

FlashAttention-2 computes the weights in `binary32` but stores them in `bfloat16` or `binary16` for
the tensor-core product with the values. Its normalizer sums the `binary32` weights, while the
weighted sum multiplies the values by their stored copies. The numerator then uses weights that the
denominator never saw, so the output isn't an average of the values: when every value is one, the
output need not be one. This mismatch was reported before, in flash-attention pull request #2721.
TorchLean bounds both recipes.

In `onlineStep` the two roundings are `roundSum` and `round`. FlashAttention-2's recipe is
`roundSum = id` and `round = roundTo low`, where `roundTo low x` stores `x` in the format `low` and
reads it back in the working format. Summing the stored weights instead is `roundSum = round`. With
128 keys of value one in blocks of 32 and native floats, weights rounded to the eight significant
bits of `bfloat16` give:

```lean (name := arStored)
-- Round a native float to the 8 significant bits of
-- bfloat16, then run both recipes on constant values.
def arBfloat16 (x : Float) : Float :=
  let (m, e) := x.frExp
  Float.scaleB (Float.scaleB m 8).round (e - 8)

def arConstantKeys : List (Float × Float) :=
  (0, 1) :: List.replicate 127 (-0.003, 1)

#eval Softmax.onlineAverage Float.exp id
  arBfloat16 0 (List.toChunks 32 arConstantKeys)
#eval Softmax.onlineAverage Float.exp arBfloat16
  arBfloat16 0 (List.toChunks 32 arConstantKeys)
```

```leanOutput arStored (whitespace := lax)
0.999094
```
```leanOutput arStored (whitespace := lax)
1.000000
```

The first recipe moves a constant value by almost $`10^{-3}`, while the second returns it exactly.
`roundTo_contract` states what storing does to one weight in $`[0,1]`, when the working format's
grid contains the grid of `low`:

```lean (name := arRoundTo)
-- Storing a weight in a narrower IEEE format and reading it
-- back meets the weight-rounding contract.
#check @roundTo_contract
```

```leanOutput arRoundTo (whitespace := lax)
@roundTo_contract : ∀ {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [inst : ExecFloat.ModelCodec plan (Model format) code],
  format.isIEEE = true →
    ∀ {low : FloatFormat},
      low.isIEEE = true →
        low.fracWidth ≤ format.fracWidth →
          format.minSubnormalExponent ≤ low.minSubnormalExponent →
            ∀ (x : ExecFloat (Configured.Family format code plan)),
              ExecFloat.Binary.isFinite x = true →
                ExecFloat.Binary.toReal x ∈ Set.Icc 0 1 →
                  ExecFloat.Binary.isFinite (roundTo low x) = true ∧
                    ExecFloat.Binary.toReal (roundTo low x) ∈ Set.Icc 0 1 ∧
                      |ExecFloat.Binary.toReal (roundTo low x) - ExecFloat.Binary.toReal x| ≤
                        Model.unitRoundoffAt low * |ExecFloat.Binary.toReal x| +
                          Model.underflowErrorAt low
```

Storing keeps the weight finite and in $`[0,1]`, and moves it by at most the unit roundoff of `low`
times its magnitude plus the underflow allowance of `low`. `onlineAverage_stored_error` compares
the computation with the exact sums of the weights it stores, not with the exact weights. For any
two roundings with
$`|\mathrm{round}(x)-\mathrm{roundSum}(x)|\le\delta\,\mathrm{roundSum}(x)+\eta_s`, the output is
within a closed-form bound of $`\beta z`, with $`z` in the range $`[lo,hi]` of the values and
$`|\beta-1|\le\delta+N\eta_s`. Its exponents are $`u(B+2T)` and $`u(B+1+2T)` for every partition,
with no shift term, since the shifts are those of the stored weights.

For FlashAttention-2's recipe, `onlineAverage_id_roundTo_error` gives $`|\beta-1|\le u_L+N\eta_L`,
with the unit roundoff and underflow allowance of `low`. So the output is a point of the values'
range scaled by a factor that can be off by a unit roundoff of `low`. Summing the stored weights
gives $`\beta=1` (`onlineAverage_roundTo_error`), so constant values are kept up to the
bound. At 4096 keys in 32 blocks of at most 128 with `bfloat16` weights, `flash2_4096_128_output`
bounds FlashAttention-2's recipe by `2.32e-5` around $`\beta z` with
$`|\beta-1|\le 2^{-8}+2^{-122}`, and `stored_4096_128_output` bounds the stored-sum recipe by
`2.31e-5` around a point $`z` of the values' range. Measured against the exact softmax average
instead, `online_4096_128_bfloat16` bounds the error with `bfloat16` weights by `0.004`, against
`2.9e-5` with `binary32` weights.

# Online Softmax In The Coverage Analysis

`roundoff` analyzes the online schedule too. For every IEEE format and every list of blocks of any
sizes, {src "TorchLean/Verify/RuntimeApprox/Roundoff/OnlineSoftmax.lean"}[the generated fold]
proves that the normalizer and the output of one query are finite and have mixed bounds against the
same schedule in real arithmetic (`blocked_normalizer_error`, `blocked_attention_error`). Its
budgets are recurrences over the blocks, not closed forms.

Online routes of `Tensor.softmax` and `Tensor.attention` fold over the allowed keys of a fiber (a
list that the mask selects), and the analysis doesn't read that list as a loop over stored entries.
So
{src "TorchLean/Verify/RuntimeApprox/Roundoff/Rules/OnlineSoftmax.lean"}[our online-softmax rules]
register the hand bounds `onlineWeight_error` and `onlineAverage_error` as `roundoff` rules for the
two iterators instead. For the attention iterator, the rule fixes `round = id`: it covers weights
kept in the working format, not weights stored in a narrower one. With them `#roundoff_coverage`
generates conditional theorems for the online forward routes of `Tensor.softmax`,
`Tensor.attention` and `Op.attention`, in both modes. Under both schedules, the backward routes of
`Op.attention` are unsupported or failed.

# A Transformer Block

Attention is one stage of a transformer block. Our
{src "TorchLean/Verify/RuntimeApprox/TransformerBlock.lean"}[transformer-block bound] treats a
pre-normalization block over the reals: layer normalization, attention of every head over the keys
of each token, an output projection and a residual sum, then a second layer normalization, a
two-layer perceptron and a second residual sum. `Arithmetic.block` evaluates the same block with
the functions of an `Arithmetic`: a rounding after each scalar operation and after each product
inside a sum, a rounding of each stored stage output, a computed sum of each list, and computed
exponential and activation. `Arithmetic.Contract` bounds each of them by a mixed error
$`|f(t)-t|\le u|t|+\eta` and charges a sum of $`n` terms $`g(n)` times the sum of their magnitudes
plus $`h(n)`.

`Arithmetic.sqrt_sum_sq_block_sub_le` bounds the Euclidean distance between each token of the
computed block and of the exact block. It holds for every arithmetic meeting the contract and every
finite model width, set of heads, head width and hidden width. Its hypotheses are Euclidean bounds
on the exact stages, spectral and Frobenius bounds on the weights (`LinearBound`), and one
inequality per stage whose right side is that stage's bound at the errors of the stages before it.
Two facts about the exact operations carry the errors forward. Layer normalization moves by at most
$`2G/\sqrt{\operatorname{var}(x)+\varepsilon}` times the change of its input
(`sqrt_sum_sq_layerNorm_sub_le`), and scores that move by at most $`t` move the softmax
by at most $`2\tanh(t/2)` in $`\ell^1`:

```lean (name := arSoftmaxL1)
-- Softmax is Lipschitz from the max norm on scores to the
-- l1 norm on weights, with a bound that saturates at two.
open TransformerBlock in
#check @sum_abs_softmax_sub_le
```

```leanOutput arSoftmaxL1 (whitespace := lax)
@sum_abs_softmax_sub_le : ∀ {ν : Type u_1} {s : Finset ν},
  s.Nonempty →
    ∀ {x x' : ν → ℝ} {t : ℝ},
      (∀ j ∈ s, |x' j - x j| ≤ t) →
        ∑ j ∈ s, |softmax s x' j - softmax s x j| ≤
          2 * (Real.exp t - 1) / (Real.exp t + 1)
```

Here the right side $`2(e^t-1)/(e^t+1)` is $`2\tanh(t/2)`, which never exceeds two (the largest
$`\ell^1` distance between two probability vectors). An attention output moves by at most that
times the bound on the values (`sqrt_sum_sq_attention_sub_le`).

`GPT2BlockBound.sqrt_sum_sq_block_sub_le` evaluates the block bound at the sizes of GPT-2 small:
width 768, 12 heads of width 64, hidden width 3072, at most 1024 keys per token, $`\varepsilon =
10^{-5}` and scale $`1/8`. With nearest-even rounding in `binary32` and unrounded stores, every
token of the computed block is within `0.538` of the exact block in Euclidean norm. The magnitudes
of the exact stages and the norms of the weights are hypotheses of that theorem, not facts proved
about a particular model. `norm_num` and `linarith` check the stage inequalities from rational
bounds on the square roots, exponentials and summation constants. The rounding is onto the real
grid of the format, which has no upper exponent bound, so the bound describes executable `binary32`
only when nothing overflows. It bounds the rounding schedule `Arithmetic.block`. The executed
op-table block normalizes and adds biases differently, and the bound does not cover it.

# What The Bounds Cover

Our online-softmax theorems are statements about the Lean schedule: `onlineAverage` for one query
and the weights of `Tensor.softmax` under `.online blocks`, in a configured IEEE format with
nearest-even operations and FloatLib's correctly rounded exponential. Each rounding theorem also
keeps its own hypotheses: finite inputs, scores at most half the largest finite value, bounded
values, a normalizer allowance below the exact normalizer, and range conditions on the inputs. A
native kernel has its own exponential and its own summation order inside a block, and it needs an
agreement statement with this schedule before these bounds describe it. For attention, the rounding
of the score product adds to the bounds (which take the scores as inputs). No theorem here bounds
the backward pass of the online schedule. The transformer-block bound holds for every arithmetic
meeting its contract, while its GPT-2 instance rests on stage bounds supplied as hypotheses.
