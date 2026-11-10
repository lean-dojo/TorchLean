import Verso
import VersoManual
import VersoBlueprint
import FloatLib.Floats
import FloatLib.Floats.Formats.BinaryInterchange
import FloatLib.Floats.Formats.Flocq.Theory.Rounding.Affine
import TorchLean.Verify.Floats.IEEE754
import TorchLean.Verify.NumericalCertificate
import TorchLean.Verify.RuntimeApprox.NF.Attention
import TorchLean.Verify.RuntimeApprox.NF.Convolution
import TorchLean.Verify.RuntimeApprox.NF.EndToEnd
import TorchLean.Verify.RuntimeApprox.NF.Normalization
import TorchLean.Verify.RuntimeApprox.NF.Optimizers
import TorchLean.Verify.RuntimeApprox.NF.ReductionOps
import TorchLean.Verify.RuntimeApprox.OptimizerContract
import TorchLean.Verify.RuntimeApprox.InverseTrigonometric
import TorchLean.Verify.RuntimeApprox.OnlineSoftmax
import TorchLean.Verify.RuntimeApprox.TransformerBlock

open Verso.Genre
open Verso.Genre.Manual
open Informal

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

#doc (Manual) "Floating-Point Numerics" =>

The numerical map starts with FloatLib's generic formats and configured executable scalars.
TorchLean keeps two binary32 specializations in view: `FP32` is the imported rounded-real model
with binary32's gradual-underflow grid, and `ExecFloat.Binary 8 23` includes signed zeros,
infinities, and NaNs. Bridge theorems connect their finite cases. TorchLean's runtime-approximation
layer then composes local operator bounds over whole forward and backward graphs. Other configured
widths share FloatLib's public scalar API, but a network error bound must use the format and
operation sequence actually selected.

:::group "generic_numerics"
Generic formats, rounding, and quantization.
:::

:::definition "radix_float_values" (parent := "generic_numerics") (lean := "FloatLib.Floats.Formats.Flocq.FloatRep")
`FloatRep` stores an integer mantissa and exponent at a chosen radix. Its real interpretation is
the exact value of that pair.
:::

:::definition "representable_float_grid" (parent := "generic_numerics") (lean := "FloatLib.Floats.Formats.Flocq.genericFormat")
`genericFormat` says when a real number belongs to the grid selected by a radix and a valid
exponent policy. Fixed, unbounded, and gradual-underflow formats are instances of this setup.
:::

:::definition "nearest_integer_rounding" (parent := "generic_numerics") (lean := "FloatLib.Floats.Formats.Flocq.ValidRndToNearest")
`ValidRndToNearest` is the contract required of an integer rounding rule: it is monotone,
fixes integers, and stays within one half of its input.
:::

:::definition "generic_grid_rounding" (parent := "generic_numerics") (lean := "FloatLib.Floats.Formats.Flocq.round")
`round` scales at the canonical exponent, applies its supplied integer rounding rule, and
maps the resulting {uses "radix_float_values"}[mantissa/exponent pair] back to a real value.
Nearestness and tie handling come from that supplied rule.

Scale selection uses the input magnitude and happens before integer rounding. Separating these
choices lets the same format support nearest and directed modes, but a nearest-error theorem can't
be transferred to a directed mode merely because their representable values coincide.
:::

:::theorem "nearest_rounding_error" (parent := "generic_numerics") (lean := "FloatLib.Floats.Formats.Flocq.error_bound_ulp")
When its integer rule satisfies {uses "nearest_integer_rounding"}[the nearest-rounding contract],
{uses "generic_grid_rounding"}[generic grid rounding] is within half an ULP.
:::

:::proof "nearest_rounding_error"
We apply {uses "nearest_integer_rounding"}[the half-integer error bound] to the scaled mantissa,
then rescale it at the exponent used by {uses "generic_grid_rounding"}[the grid rounder].
:::

:::definition "affine_quantizer" (parent := "generic_numerics") (lean := "FloatLib.Numerics.Quantization.RealAffineQuantizer")
A bounded affine quantizer records its positive scale, zero point, and nonempty integer code range.
Its `quantize` operation accepts the integer rounding rule separately.
:::

:::theorem "affine_quantization_accuracy" (parent := "generic_numerics") (lean := "FloatLib.Floats.Formats.Flocq.affine_dequantize_quantize_error_le_half")
An unclipped value passed through the {uses "affine_quantizer"}[affine quantizer] reconstructs
within half a scale step when its integer rule satisfies
{uses "nearest_integer_rounding"}[the nearest-rounding contract].

Unclipped means that the rounded code lies inside the allowed integer range, and the theorem needs
that hypothesis because saturation can introduce an error larger than the half-step rounding
budget.
:::

:::proof "affine_quantization_accuracy"
Here the proof applies {uses "nearest_integer_rounding"}[the half-integer error bound] before
multiplying by the positive scale from {uses "affine_quantizer"}[the quantizer].
:::

:::group "binary32_semantics"
Proof-oriented and executable accounts of IEEE 754 binary32.
:::

:::definition "rounded_real_fp32" (parent := "binary32_semantics") (lean := "TorchLean.FP32")
`FP32` specializes {uses "representable_float_grid"}[the generic format theory] to a proof-oriented
nearest-even model over real values. It leaves out NaNs, infinities, and the upper exponent cutoff,
so claims about those cases belong to FloatLib binary32.
:::

:::theorem "fp32_rounding_accuracy" (parent := "binary32_semantics") (lean := "FloatLib.Floats.Formats.BinaryInterchange.Model.abs_roundAt_sub_le")
Rounding in the {uses "rounded_real_fp32"}[rounded-real model] differs from its real input by at
most half an ULP.
:::

:::proof "fp32_rounding_accuracy"
Unfolding {uses "rounded_real_fp32"}[binary32 rounding] exposes the generic real-valued rounder
with binary32's exponent policy and nearest-even integer rounding, and the
{uses "nearest_rounding_error"}[generic half-ULP theorem] then supplies the error bound.
:::

:::definition "executable_binary32" (parent := "binary32_semantics") (lean := "FloatLib.Floats.ExecFloat.Binary")
Choose exponent and fraction widths in FloatLib's configured binary constructor. Binary32 is the
`8`, `23` configuration. Wider and custom formats use the same interface. Encodings drive
executable addition, multiplication, division, fused multiply-add, and square root, with explicit
rounding modes and operation-specific exception status.
:::

:::theorem "finite_ieee_refinement" (parent := "binary32_semantics") (lean := "FloatLib.Floats.ExecFloat.Binary.toReal_add_eq_roundAt")
On the stated finite-result path, {uses "executable_binary32"}[executable addition] agrees with
{uses "rounded_real_fp32"}[rounded-real binary32 addition]. Matching bridge theorems cover
subtraction, multiplication, fused multiply-add, square root, and division.
:::

:::proof "finite_ieee_refinement"
We decode {uses "executable_binary32"}[finite operands] to dyadics and identify the bit-level
result with {uses "rounded_real_fp32"}[nearest-even real rounding].

For a composed expression, each intermediate operation must meet the hypotheses of its bridge.
Finite source tensors alone do not establish this: addition or multiplication of finite operands
can overflow before the final output is formed.
:::

:::theorem "directed_addition_lower_bound" (parent := "binary32_semantics") (lean := "FloatLib.Floats.Formats.BinaryInterchange.Model.toEReal_add_down_le")
For finite inputs, downward-rounded {uses "executable_binary32"}[binary32 addition] is no greater
than the exact real sum, including overflow to negative infinity.
:::

:::proof "directed_addition_lower_bound"
We decode the finite {uses "executable_binary32"}[inputs] to dyadics, compute their exact dyadic
sum, and apply soundness of downward rounding in the extended reals.
:::

:::theorem "directed_addition_upper_bound" (parent := "binary32_semantics") (lean := "FloatLib.Floats.Formats.BinaryInterchange.Model.le_toEReal_add_up")
For finite inputs, the exact real sum is no greater than upward-rounded
{uses "executable_binary32"}[binary32 addition], including overflow to positive infinity.
:::

:::proof "directed_addition_upper_bound"
Decoding the finite {uses "executable_binary32"}[inputs] gives their exact dyadic sum, and
soundness of upward rounding in the extended reals finishes the proof.

Extended-real endpoints allow these two directed statements to remain useful when a finite exact
sum exceeds the largest binary32 value. A lower endpoint of negative infinity or an upper endpoint
of positive infinity still gives a valid one-sided bound (although it may be too wide for a later
verification claim). This differs from the finite refinement theorem, whose conclusion identifies a
decoded result with rounded-real arithmetic and therefore needs its finite-path premise.
:::

:::theorem "correctly_rounded_inverse_trigonometric" (parent := "binary32_semantics") (lean := "TorchLean.InverseTrigonometric.toReal_atan")
In every IEEE format, TorchLean's `atan` of a finite input is finite and is the nearest-even
rounding of the real arctangent. `toReal_asin`, `toReal_acos` and `toReal_atan2` state the same for
the inverse sine and cosine on $`[-1,1]` and for the principal argument. These functions form the
`InverseTrig` capability of every configured format, including
{uses "executable_binary32"}[binary32].
:::

:::proof "correctly_rounded_inverse_trigonometric"
FloatLib's exact comparisons of the inverse functions with rational boundaries decide the rounding
through {uses "generic_grid_rounding"}[the format's rounding]. With the real result at most $`\pi`
in magnitude (below the overflow threshold of every IEEE format), its rounding is finite.
:::

:::group "runtime_error_bounds"
Operator bounds composed across programs.
:::

:::theorem "compositional_forward_approximation" (parent := "runtime_error_bounds") (lean := "TorchLean.FwdGraph.eval_approx")
Local approximation contracts over {uses "shape_indexed_tensors"}[typed tensors] compose through
forward graph evaluation.
:::

:::proof "compositional_forward_approximation"
Forward graph induction carries every local contract through
{uses "shape_indexed_tensors"}[the stored typed context].
:::

:::theorem "compositional_reverse_approximation" (parent := "runtime_error_bounds") (lean := "TorchLean.RevGraph.backprop_approx")
Given approximate inputs and cotangent seeds, local forward and backward contracts compose into an
approximation bound for every gradient produced by reverse graph evaluation.
:::

:::proof "compositional_reverse_approximation"
Reverse graph induction reuses {uses "compositional_forward_approximation"}[forward composition]
and threads the local backward bounds through the accumulated cotangent context.

We state the accumulation law as a numerical premise: when two graph paths reach one variable,
their cotangents are added, so bounds for the two local VJPs must also account for that rounded
sum. And analytic correctness requires identifying the ideal VJP with the forward derivative.
:::

:::definition "nf_positive_division_bound" (parent := "runtime_error_bounds") (lean := "TorchLean.NF.divPosErrorBound")
`divPosErrorBound η epsx epsy xhat yhat` is an upper bound for a rounded division's forward error
(`xhat` and `yhat` are the approximate operands interpreted as reals). The exact denominator is at
least `η`, and the approximate denominator is within $`\mathtt{epsy} < \eta` of it. Three terms add
up to the bound: the numerator error divided by the effective margin $`\eta - \mathtt{epsy}`, the
denominator error times $`|\mathtt{xhat}| + \mathtt{epsx}` divided by the squared margin, and half
an ULP of the quotient $`\mathtt{xhat} / \mathtt{yhat}` under {uses "generic_grid_rounding"}[grid
rounding]. `NF.approx_div_of_pos_lb` proves the bound, and the sigmoid, softmax and attention
bounds are built on top of it.
:::

:::theorem "nf_sigmoid_bound_regression" (parent := "runtime_error_bounds") (lean := "TorchLean.NF.reciprocalSigmoidBoundScalar_le_one")
Under small-error hypotheses on the rounded one, the rounded sigmoid denominator, and the ULP of
the quotient, the error budget for the sequence $`1/(1+\exp(-x))` derived from
{uses "nf_positive_division_bound"}[the positive-division bound] is at most one. This is a
regression theorem: it pins the size of the budget so that a refactor can't silently make it
vacuous. On positive inputs the public sigmoid evaluates this sequence, and otherwise it uses
$`\exp(x)/(1+\exp(x))`. Its theorem `NF.approx_sigmoid` selects the error budget for the branch
that was evaluated.

Both branches agree as real functions, which allows the exact and rounded inputs to fall on
opposite sides of zero. But their budgets follow different rounded operation sequences (including
the numerator exponential in the nonpositive branch).
:::

:::proof "nf_sigmoid_bound_regression"
Each of the three terms of {uses "nf_positive_division_bound"}[the division bound] is estimated
against the hypotheses, and the estimates are summed.
:::

:::definition "numerical_optimizer_contract" (parent := "runtime_error_bounds") (lean := "TorchLean.OptimizerContract")
`OptimizerContract` packages specification and runtime optimizer states, their approximation
relation, an update bound, any domain checks, and the theorem that one update respects the bound.
:::

:::theorem "end_to_end_runtime_error_bounds" (parent := "runtime_error_bounds") (lean := "TorchLean.NF.update_approx_graphData")
For one parameter in an already constructed typed reverse graph,
{uses "compositional_reverse_approximation"}[the reverse approximation theorem] and a
{uses "numerical_optimizer_contract"}[valid optimizer contract] carry the stated input, seed,
parameter, state, and step-data bounds through backpropagation and one optimizer update.
:::

:::proof "end_to_end_runtime_error_bounds"
From {uses "compositional_reverse_approximation"}[reverse composition] the proof obtains the
indexed gradient bound, then applies the update-soundness field of
{uses "numerical_optimizer_contract"}[the supplied optimizer contract].

The parameter tensor being updated is a separate argument from the graph context: for ordinary
training, the application chooses the tensor at that context index, so the gradient is evaluated at
the parameter value that receives the update.
:::

:::theorem "online_softmax_real_value" (parent := "runtime_error_bounds") (lean := "TorchLean.Tensor.attention_online_eq")
Over the reals, attention by FlashAttention's online schedule (one pass over blocks of keys with a
running maximum and a rescaled normalizer) equals attention by the two-pass schedule, for every
block size, mask, scale and multiplier. `Tensor.softmax_online_eq` states the same for softmax at
every axis and rank.
:::

:::proof "online_softmax_real_value"
After any prefix of blocks with running maximum $`M`, the state holds the normalizer and the
weighted sum shifted by $`M`, so the rescalings cancel and the quotient is the softmax average.
:::

:::theorem "online_softmax_rounding" (parent := "runtime_error_bounds") (lean := "TorchLean.OnlineSoftmax.onlineAverage_error")
For one query, in every IEEE format, for every partition of the keys into blocks of at most $`B`
keys and with gradual underflow, the computed online softmax average is finite and within an
explicit bound of the exact average, given finite inputs, bounded scores and values, and range
conditions on the inputs. Each key's weight receives the relative allowance
$`\exp(u(M-s)+u(B+3T-1))-1` in the normalizer for $`T` blocks and final maximum $`M`.
:::

:::proof "online_softmax_rounding"
An invariant over the keys seen so far carries one aggregate error inequality through the fold.
The roundings of the shifts against the running maximum add up to $`u(M-s)`, because the maximum
only grows.
:::

:::theorem "transformer_block_rounding" (parent := "runtime_error_bounds") (lean := "TorchLean.TransformerBlock.Arithmetic.sqrt_sum_sq_block_sub_le")
For every arithmetic whose operations, products, stores, sums, exponential and activation meet
mixed-error contracts, each token of a pre-normalization transformer block evaluated in that
arithmetic is within an explicit Euclidean distance of the exact block, given bounds on the exact
stages and the weights and one inequality per stage.
:::

:::proof "transformer_block_rounding"
Each stage lemma bounds that stage's error from the errors of the stages before it. Layer
normalization and softmax are Lipschitz in the needed norms, with constants
$`2G/\sqrt{\operatorname{var}(x)+\varepsilon}` for layer normalization and $`2\tanh(t/2)` in
$`\ell^1` for softmax scores that move by at most $`t`.
:::

:::definition "checked_numerical_certificate" (parent := "runtime_error_bounds") (lean := "TorchLean.NumericalCertificate.Checked")
A checked numerical certificate retains the submitted artifact together with the
canonical source ranges, node-range trace, accepted kernel plan, and proofs that the recomputed
trace and audit match the artifact. Real-valued enclosure is a separate theorem.

This distinction separates consistency of submitted data from mathematical soundness of the range
transfer. Reconstructing the same interval twice proves agreement with the range transfers, but
enclosure still needs a connection to the operation's real denotation.
:::

:::definition "proved_real_enclosure_trace" (parent := "runtime_error_bounds") (lean := "TorchLean.NumericalCertificate.RealEnclosure")
For one {uses "checked_numerical_certificate"}[checked certificate],
`NumericalCertificate.RealEnclosure` stores a
real payload and input, the complete {uses "ir_denotation"}[IR execution trace], and a proof that
each real node value lies in its checked interval.
:::

:::theorem "checked_numerical_execution" (parent := "runtime_error_bounds") (lean := "TorchLean.NumericalCertificate.CheckedExecution.errorTrace")
A checked IEEE replay plus a separately supplied
{uses "proved_real_enclosure_trace"}[real-execution enclosure] for the same
{uses "checked_numerical_certificate"}[certificate] yields a graph-wide pointwise error trace whose
budget at each node is the width of its checked interval.
:::

:::proof "checked_numerical_execution"
The proof combines the IEEE replay's range check with
{uses "proved_real_enclosure_trace"}[the supplied real enclosure], node by node.

Both values lie between the same endpoints, so their distance is bounded by the interval width.
This is pointwise in the two supplied executions. A statement for every input in a region requires
real-enclosure and execution evidence quantified over that region.
:::
