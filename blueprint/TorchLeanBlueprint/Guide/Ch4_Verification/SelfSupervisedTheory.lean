import VersoManual
import VersoBlueprint

open Verso.Genre Manual

#doc (Manual) "Self Supervised Theory" =>
%%%
tag := "self-supervised-theory"
%%%

Self supervised learning is full of objectives whose code looks simple but whose meaning depends on
bookkeeping: which patches are masked, which view is predicted, which encoder receives gradients,
and which term prevents collapse.

TorchLean's SSL layer is about these objective semantics. It does not prove that MAE, JEPA, or
VICReg training learns useful representations. It proves smaller facts: masked sums decompose as
intended, MAE and JEPA instantiate a common predictive-view contract, and collapse guards are
positive under the stated hypotheses.

# The Minimal Pattern

Most SSL objectives in this layer have the same three ingredients:

- *View*: a finite piece of an input, image, graph, or sequence, such as visible patches versus
  masked patches.
- *Prediction*: a map from a context representation to a target representation, such as a JEPA
  style predictor head.
- *Guard term*: an extra scalar penalty that rules out a degenerate solution, such as a VICReg
  variance floor for collapse.

That lets us write theorem statements about the objective itself. We are not proving that every
training run learns good representations; we are proving that the named loss decomposes, respects
finite masks, and penalizes the degenerate cases it claims to penalize.

# Predictive Views and Masks

The common predictive view pattern compares a prediction from one view against a representation of
another view:

- encode a context view, then apply a predictor head to get `z_ctx`;
- encode the target view to get `z_target`;
- evaluate the view loss as `ell z_ctx z_target`.

The [predictive view API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/SelfSupervised/PredictiveView.lean) packages that
pattern as `PredictiveViewContract`. The corresponding objective decomposes over a finite list of
views by summing `predictiveLoss C v` over the selected views.

In symbols, the view contract has the shape:

$$`z_c=p(f_c(x_c)),
\qquad
z_t=f_t(x_t),
\qquad
L_{\mathrm{view}}=\ell(z_c,z_t).`

The theorem `predictiveViewObjective_decomposes` records that equation in Lean. The bridge theorems
`mae_is_predictive_view_loss`, `mae_is_predictive_view_objective`,
`jepa_is_predictive_view_loss`, and `jepa_is_predictive_view_objective` say that MAE and JEPA style
objectives are instances of the same contract rather than unrelated pieces of code.

Masks are equally concrete. The [masking API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/SelfSupervised/Masking.lean)
uses finite indices throughout: a mask for length `n` is a function from `Fin n` to `Bool`, and
`maskedLoss idxs ell` is the finite sum of `ell i` over the selected indices.

The masked objective is just a finite sum:

$$`L_M=\sum_{i\in M}\ell(\hat x_i,x_i).`

The key theorems are small but important: `maskedLoss_append`, `maskedLoss_reverse`, and
`maskedLoss_eq_zero_of_all_zero`. They make sure that serialization details such as splitting or
reversing the selected patch list do not silently change the algebra being proved.

A toy example is enough to see why this matters. If a mask selects indices `[0, 2]`, then the masked
loss is `loss 0 + loss 2`. Reversing the selected list should not change the sum, and appending two
disjoint chunks should give the same result as summing over the combined list. These are simple
algebraic facts, but they are exactly the facts that catch bookkeeping mistakes in masked
objectives.

# MAE and JEPA

The [MAE API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/SelfSupervised/MAE.lean) starts with a patch batch, represented as a
function from `Fin n` to patches. Exact reconstruction means every finite patch index agrees, and
`maeLoss` is the masked reconstruction loss over the selected indices.

The MAE shape is:

$$`L_{\mathrm{MAE}}
=
\sum_{i\in M}
\ell\!\left(\operatorname{dec}(\operatorname{enc}(x_{\mathrm{visible}}))_i,x_i\right).`

The theorem `exactReconstruction_identity` is the base case: the identity reconstruction is exact.
Theorems such as `maeLoss_append`, `maeLoss_reverse`, and
`maeLoss_eq_zero_of_patch_losses_zero` give the list algebra for masked reconstruction losses.

The [JEPA API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/SelfSupervised/JEPA.lean) keeps `encodeContext`, `encodeTarget`,
and `predict` abstract. Its loss compares `predict (encodeContext ctx)` with
`encodeTarget target` under the chosen scalar loss.

The JEPA shape is:

$$`L_{\mathrm{JEPA}}
=
\sum_{v\in V}
\ell\!\left(p(f_{\mathrm{ctx}}(x_{\mathrm{ctx}}^v)),
f_{\mathrm{target}}(x_{\mathrm{target}}^v)\right).`

The theorems `jepaLoss_append`, `jepaLoss_reverse`, and `jepaLoss_target_ext` are objective algebra
facts. They do not say that a representation is useful; they say that the target and context
terms in the objective are exactly the terms named in the statement.

# VICReg and Redundancy Reduction

The [VICReg API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/SelfSupervised/VICReg.lean) names the three pressure terms that
show up in VICReg and related redundancy reduction methods: invariance, variance, and covariance.
The variance floor penalty is the positive part of `gamma - sigma^2`, so a representation with
zero variance is penalized whenever the floor `gamma` is positive.

The objective has the familiar weighted shape:

$$`L
=
\lambda L_{\mathrm{inv}}
+\mu L_{\mathrm{var}}
+\nu L_{\mathrm{cov}}.`

A variance guard has the form:

$$`L_{\mathrm{var}}
=
\sum_j \max(0,\gamma-\sigma_j)^2.`

The useful theorem pattern is collapse detection. If every variance is zero and the variance floor
`γ` is positive, then the variance penalty is positive:

`varianceTerm_collapsed_positive` is the Lean theorem name for this collapsed representation case.

The predictive view API also has real valued graph SSL facts such as
`graphAlignmentEnergy_eq_zero_of_collapsed`, `realVarianceFloorGuard_zero_spread_positive`, and
`graphSSLObjective_collapsed_positive`. These make the common self supervised warning precise:
alignment alone may accept collapsed representations, so a guard term is needed if collapse is a
failure mode.

Collapse is one of the central semantic hazards in self supervised learning. A runtime experiment
may show that a loss decreased; the formal objective can additionally state whether the collapsed
case is penalized by the loss itself.

A loss going down does not tell us whether the implementation used the intended mask, whether
masked patches were ordered correctly, or whether the collapse penalty was active. The SSL theory
layer checks objective identities and degenerate cases directly.

# What We Claim

TorchLean formalizes selected SSL objective components and finite mask/list properties. It does not
prove that MAE, JEPA, or VICReg training learns useful representations. Useful external anchors are
[MAE by He et al.](https://arxiv.org/abs/2111.06377),
[VICReg by Bardes et al.](https://arxiv.org/abs/2105.04906), and JEPA style predictive embedding
objectives from LeCun and collaborators. Those papers motivate the shapes of the objectives; the
Lean declarations state the algebraic pieces checked in this layer.
