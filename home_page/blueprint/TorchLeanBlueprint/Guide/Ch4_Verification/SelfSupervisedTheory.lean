import VersoManual
import NN.API.SelfSupervised.BlockMask
import NN.MLTheory.SelfSupervised.JEPA
import NN.MLTheory.SelfSupervised.MAE
import NN.MLTheory.SelfSupervised.Masking
import NN.MLTheory.SelfSupervised.PredictiveView
import NN.MLTheory.SelfSupervised.VICReg
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean

-- Everything in the finite theory lives in one namespace, and the runtime mask lives in another,
-- so both are opened here to keep the displayed `#check` lines inside Verso's narrow code column.
-- The printed signatures still spell out every name in full, which is how you can tell which side
-- of the bridge a declaration is on: `NN.MLTheory.SelfSupervised` is the proof layer and
-- `TorchLean.ssl` is the executable one.
open NN.MLTheory.SelfSupervised
open TorchLean.ssl

-- A handful of signatures below print wider than this file's 100-column limit, so their
-- `leanOutput` blocks ask for `whitespace := lax` and are wrapped in the source. The rendered page
-- still shows each message exactly as Lean printed it.
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Self-Supervised Objectives" =>
%%%
tag := "self-supervised-theory"
%%%

An MAE loss depends on which patches were hidden. A JEPA loss depends on which branch supplies the
target representation. An alignment objective can be minimized by mapping every view to the same
vector unless another term prevents collapse.

TorchLean's present self-supervised theory isolates this bookkeeping. It is a finite algebraic
model of masks, target views, predictive losses, and collapse guards. It does not formalize
a complete MAE or JEPA training run, and it does not prove that minimizing one of these objectives
learns useful representations.

A plausible loss value can hide an index counted twice, a mask read with the wrong polarity, or
a variance floor that assigns no penalty to collapse. Small examples let us change one of these
choices and calculate its effect. Named Lean blocks and their outputs are checked while the page
is built; Python fragments illustrate related objectives.

# Masks And Index Arrays

A mask over `n` positions assigns a Boolean to each position:

```lean (name := maskDef)
-- Relate the Boolean mask representation to its
-- selected-position proposition.
example (n : Nat) : Mask n = (Fin n → Bool) := rfl

#check @selected
```

```leanOutput maskDef
@selected : {n : ℕ} → Mask n → Fin n → Prop
```

`selected m i` is the proposition `m i = true`, which is what lets a mask appear in a hypothesis
rather than only in a computation. The module supplies all-true and all-false masks, a pointwise
complement, and simplification lemmas for these operations. Evaluating the complement shows its
effect:

```lean (name := maskEval)
-- Complementing an all-selected mask removes the same
-- position from selection.
#eval allMask 4 2
#eval complement (allMask 4) 2
```

```leanOutput maskEval
true
```

```leanOutput maskEval
false
```

Here `true` means selected. For a reconstruction loss, selected positions are the hidden targets
to score; they are not the visible positions supplied to the encoder. Complementing that mask
exchanges those roles. The theory fixes the selected-index convention, and callers still need
to specify which set their objective should select.

The loss itself does not take a mask. It takes an explicit array of the selected indices:

```lean (name := maskedLossSig)
-- The loss accepts in-range index occurrences and returns
-- their natural-valued sum.
#check @maskedLoss
```

```leanOutput maskedLossSig
@maskedLoss : {n : ℕ} → Array (Fin n) → (Fin n → ℕ) → ℕ
```

The element type `Fin n` carries a proof that the index lies within the grid. This rules out an
out-of-range value at the loss interface, but a caller can still construct the wrong in-range
index. In particular, bounds safety does not prove mask polarity or patch identity. The runtime
mask materializer returns the same `Array (Fin n)` type, as the last section demonstrates.

A Boolean mask can select an index only once, whereas an index array can contain several
occurrences of it. Moving from the former representation to the latter therefore introduces a
property worth checking: does the materialized array enumerate exactly the selected positions,
with the intended multiplicity? The type `Fin n` answers only the range question. `selected m i`
answers whether one position belongs to a mask. A producer theorem relating that proposition to
array membership, together with a no-duplicates property when needed, would answer the stronger
selection question. The loss interface itself deliberately accepts any in-range array.

The container is `Array`, not `List`. The producer is executable code that filters
`Array.finRange`, and using the producer's own type means no conversion lemma has to sit between
the runtime index buffer and the theorems about it.

The result is a sum, so concatenating arrays adds their losses. With per-patch losses `1, 2, 3, 4`
the sum over `#[0]` is `1`, the sum over
`#[2]` is `3`, and the sum over the concatenation is `4`. The corresponding means are `1`, `3`, and
`2`. Combining group means requires their selected counts as weights. The finite objective leaves
normalization to a separate step, so its append theorem needs no count or nonempty-array hypotheses.

Run the objective on a concrete index array. The per-patch loss below is `i + 1`, so the four
positions cost `1, 2, 3, 4`:

```lean (name := lossTwo)
-- Sum the costs at positions zero and two.
#eval maskedLoss (#[0, 2] : Array (Fin 4))
  (fun i => i.val + 1)
```

```leanOutput lossTwo
4
```

Now select index `2` twice:

```lean (name := lossDup)
-- Repeating position two repeats its contribution to the
-- sum.
#eval maskedLoss (#[0, 2, 2] : Array (Fin 4))
  (fun i => i.val + 1)
```

```leanOutput lossDup
7
```

Index `2` contributes its loss of `3` twice, giving `7`. The definition sums array occurrences;
it does not deduplicate them. If a producer intends each hidden patch to contribute once, it
must establish that its index array has no duplicates. The repeated-index calculation also has
an exact proof:

```lean
-- Prove the repeated-index total by unfolding the finite
-- array sum.
example :
    maskedLoss (#[0, 2, 2] : Array (Fin 4))
        (fun i => i.val + 1) = 7 := by
  simp [maskedLoss]
```

Order, on the other hand, does not matter:

```lean (name := lossSwap)
-- Reversing these two indices retains both contributions.
#eval maskedLoss (#[2, 0] : Array (Fin 4))
  (fun i => i.val + 1)
```

```leanOutput lossSwap
4
```

Append and reverse have corresponding theorems:

```lean (name := lossThms)
-- The array laws split or reverse indices while keeping
-- per-patch scores fixed.
#check @maskedLoss_append
#check @maskedLoss_reverse
```

```leanOutput lossThms
@maskedLoss_append : ∀ {n : ℕ} (xs ys : Array (Fin n)) (perPatchLoss : Fin n → ℕ),
  maskedLoss (xs ++ ys) perPatchLoss = maskedLoss xs perPatchLoss + maskedLoss ys perPatchLoss
```

```leanOutput lossThms
@maskedLoss_reverse : ∀ {n : ℕ} (idxs : Array (Fin n)) (perPatchLoss : Fin n → ℕ),
  maskedLoss idxs.reverse perPatchLoss = maskedLoss idxs perPatchLoss
```

The reverse theorem proves one particular reordering preserves the sum; its statement does not
quantify over arbitrary permutations. Neither theorem eliminates duplicates.
`maskedLoss_eq_zero_of_all_zero`
completes the small set: if every selected index has zero loss then the objective is zero.

In `maskedLoss_append`, the same function `perPatchLoss` appears on both sides. The theorem
splits the index collection while holding the score assigned to each index fixed. It would not
justify separately running a model on two groups if that model's predictions changed with batch
composition. The reverse theorem has the same qualification: reversing the array leaves the
lookup function untouched. This is an algebraic fact about summing fixed natural values, useful
once a caller has identified those values with the per-patch quantities its model should score.

The scalar type also limits what these results say. The finite model computes in `ℕ`, so its
"losses" are already-computed nonnegative summaries. Exact array algebra applies to those
summaries; relating them to a mean-squared error over runtime floats requires a separate argument.

# Predictive View Contracts

MAE predicts pixels or patches. JEPA predicts a latent target representation. The surrounding index
algebra is almost identical, so
{src "NN/MLTheory/SelfSupervised/PredictiveView.lean"}[`PredictiveViewContract`]
keeps only the target types separate:

```lean (name := contractSig)
-- Keep context, raw targets, encoded targets, and
-- predictions as separate type roles.
#check PredictiveViewContract
```

```leanOutput contractSig (whitespace := lax)
NN.MLTheory.SelfSupervised.PredictiveViewContract (n : ℕ)
  (Context Target TargetRep Prediction : Type) : Type
```

The four type parameters name distinct roles, even when an application uses the same type for
several of them:

- `Context` is what the online branch sees;
- `Target` is the raw target-view value;
- `TargetRep` is what the target encoder produces from it;
- `Prediction` is what the context-side predictor produces.

The record then holds `targetIdxs`, a context value, a target function, a `targetEncoder`, a
`predict`, a `distance` that compares a `TargetRep` with a `Prediction`, and a `geometryGuard` that
defaults to zero. For a selected index $`i` the predictive term is

$$`\ell\!\left(
  \operatorname{targetEncoder}_i(\operatorname{target}_i),
  \operatorname{predict}(\operatorname{context},i)
\right),`

and summing over `targetIdxs` gives `predictiveLoss`:

```lean (name := predLossSig)
-- The predictive contract still returns a natural-valued
-- finite loss.
#check @predictiveLoss
```

```leanOutput predLossSig (whitespace := lax)
@predictiveLoss : {n : ℕ} →
  {Context Target TargetRep Prediction : Type} →
    PredictiveViewContract n Context Target TargetRep Prediction → ℕ
```

The full finite objective adds the guard:

$$`L_{\mathrm{SSL}}
=L_{\mathrm{predictive}}+L_{\mathrm{geometry}}.`

`predictiveViewObjective_zero_geometry` proves that a zero guard leaves the predictive term alone.
`withGeometryGuard` attaches a VICReg-style or Barlow-style term without changing the view
selection. The guard is a supplied numeric value. It can be changed without altering the stored
indices, but a caller may compute that value from those indices; the record does not enforce
independence.

The return type of `predictiveLoss` is still `ℕ`, even though the representation types can be
arbitrary. The record tells Lean how to turn a target representation and a prediction into that
number; it does not infer a metric from their types. In particular, the `distance` field comes
with no symmetry, triangle inequality, or “zero exactly when equal” law. A zero predictive loss
therefore means the supplied scores sum to zero. Interpreting it as exact reconstruction needs
an additional property of the chosen score. This flexibility is why the same contract can
accommodate pixels, latent vectors, and discrete summaries without silently identifying them.

The contract stores target values and an encoder function. It can describe the values supplied
by a stopped-gradient target branch, but it does not enforce that gradient behavior. A caller
could make those values depend on trainable parameters outside the contract. Proving which
parameters receive gradients requires a separate differentiation theorem.

# MAE Identity Targets

Give the contract a concrete four-patch problem. The target patches are `0, 10, 20, 30`, the model
guesses `0, 17, 25, 30`, and the per-patch loss is absolute difference written with truncated
natural subtraction:

```lean
-- Separate target patches, predictions, and the indices
-- selected for scoring.
def absDiff (a b : Nat) : Nat := (a - b) + (b - a)

def patches : PatchBatch 4 Nat := fun i => 10 * i.val

def guesses : Fin 4 → Nat := fun i =>
  if i.val = 1 then 17 else
  if i.val = 2 then 25 else 10 * i.val

def maskedIdxs : Array (Fin 4) := #[0, 2]
```

The per-position errors are `0, 7, 5, 0`. Hiding patches `0` and `2` costs:

```lean (name := maeMasked)
-- Score only the selected hidden patches, excluding the
-- error at position one.
#eval maeLoss maskedIdxs patches guesses absDiff
```

```leanOutput maeMasked
5
```

Scoring every position instead costs:

```lean (name := maeAll)
-- Scoring every position includes the additional error of
-- seven.
#eval maeLoss (Array.finRange 4) patches guesses absDiff
```

```leanOutput maeAll
12
```

The difference is the error of `7` at position `1`, which the masked objective excludes. A loss
value of `5` is consistent with the selected targets even though a larger error exists elsewhere.
Checking the value alone cannot establish that the intended patches were selected.

The MAE contract is the identity-target instance,

$$`\operatorname{targetEncoder}_i(x_i)=x_i,`

and its predictive loss is not merely equal to `maeLoss` but definitionally equal:

```lean (name := maeBridge)
-- The identity target encoder makes the common predictive
-- loss equal to MAE loss.
#check @mae_is_predictive_view_loss
```

```leanOutput maeBridge (whitespace := lax)
@mae_is_predictive_view_loss : ∀ {n : ℕ} {Patch Pred : Type}
  (maskedIdxs : Array (Fin n)) (target : PatchBatch n Patch)
  (pred : Fin n → Pred) (patchLoss : Patch → Pred → ℕ),
  predictiveLoss (maeAsPredictiveViewContract maskedIdxs target pred patchLoss) =
    maeLoss maskedIdxs target pred patchLoss
```

The signature binds the index array, target function, prediction function, and loss once, then
uses those same four arguments on each side of the equality. There is no existence claim about a
predictor and no premise saying it was trained. `maeAsPredictiveViewContract` simply places these
arguments in the common record, with identity target encoding. This is useful when a later proof
is already phrased using predictive views: it can rewrite an MAE objective into that vocabulary
without changing which patches or predictions are scored.

The proof is `rfl`: unfolding the MAE contract and the two loss definitions produces the same
expression. No assumptions about the supplied patch loss are needed. Applying the theorem to the
four-patch example gives:

```lean
-- Instantiate the objective identity using the same targets
-- and predictions.
example :
    predictiveLoss
        (maeAsPredictiveViewContract maskedIdxs patches
          guesses absDiff) =
      maeLoss maskedIdxs patches guesses absDiff :=
  mae_is_predictive_view_loss maskedIdxs patches guesses
    absDiff
```

`mae_is_predictive_view_objective` adds that the full objective is still `maeLoss`, because the
geometry guard of this contract is zero.

The MAE file also carries a reconstruction sanity theorem:

```lean (name := exactRecon)
-- Returning each supplied patch unchanged gives exact
-- reconstruction.
#check @exactReconstruction_identity
```

```leanOutput exactRecon
@exactReconstruction_identity : ∀ {n : ℕ} {Patch : Type} (x : PatchBatch n Patch),
  ExactReconstruction x (reconstruct (fun x p => p) x)
```

Here the decoder `fun x p => p` ignores its index and returns the supplied patch unchanged.
Reconstructing the original patch batch through that identity is exact. This is a property of
the specified map, with no trained decoder involved. The finite MAE loss
inherits append, reverse, and zero-per-patch theorems in the same spirit. They prove the objective
is assembled as intended, and they remain silent about patchification, pixel normalization, and
tensor decoders until those are connected to this contract.

# JEPA Target Representations

I-JEPA {Informal.citep ijepa2023}[] predicts target-block representations rather than pixels, so
the objective starts one encoder later:

$$`L_{\mathrm{JEPA}}
=\sum_{i\in I}
\ell\!\left(z_i^{\mathrm{target}},
p(z^{\mathrm{context}},i)\right).`

`jepaAsPredictiveViewContract` takes the supplied target as its own representation, while
`encodedTargetPredictiveViewContract` exposes a separate encoder for the general case, and
`jepa_is_predictive_view_loss` and `jepa_is_predictive_view_objective` identify the JEPA sum with
the common contract. Those are the same `rfl`-shaped bridges as on the MAE side.

The target extensionality theorem makes the dependence on selected indices explicit:

```lean (name := jepaExt)
-- Target equality is required only at indices occurring in
-- the selected array.
#check @jepaLoss_target_ext
```

```leanOutput jepaExt (whitespace := lax)
@jepaLoss_target_ext : ∀ {n : ℕ} {Context Target Pred : Type}
  (idxs : Array (Fin n)) (context : Context)
  (target₁ target₂ : Fin n → Target) (predict : Context → Fin n → Pred)
  (repLoss : Target → Pred → ℕ),
  (∀ i ∈ idxs, target₁ i = target₂ i) →
    jepaLoss idxs context target₁ predict repLoss = jepaLoss idxs context target₂ predict repLoss
```

The hypothesis quantifies over `i ∈ idxs`, and that scope is exactly as narrow as it looks. Take
three target branches: one baseline, one that changes `10` to `999` at the unselected position
`1`, and
one that differs by a single unit at the selected position `2`.

```lean
-- Change one unselected target and one selected target in
-- separate branches.
def targetsA : Fin 4 → Nat := fun i => 10 * i.val

def targetsB : Fin 4 → Nat := fun i =>
  if i.val = 1 then 999 else 10 * i.val

def targetsC : Fin 4 → Nat := fun i =>
  if i.val = 2 then 21 else 10 * i.val
```

The baseline reproduces the MAE number, since the predictor is the same:

```lean (name := jepaA)
-- The baseline target branch reproduces the masked
-- reconstruction total.
#eval jepaLoss maskedIdxs () targetsA
  (fun _ i => guesses i) absDiff
```

```leanOutput jepaA
5
```

Moving an unselected target from `10` to `999` changes nothing:

```lean (name := jepaB)
-- Changing an unselected target leaves both scored
-- positions unchanged.
#eval jepaLoss maskedIdxs () targetsB
  (fun _ i => guesses i) absDiff
```

```leanOutput jepaB
5
```

Moving a selected target by one unit does change the answer:

```lean (name := jepaC)
-- Changing the selected target at position two reduces its
-- error by one.
#eval jepaLoss maskedIdxs () targetsC
  (fun _ i => guesses i) absDiff
```

```leanOutput jepaC
4
```

The first equality is the theorem, and discharging its side condition is a finite check over the two
selected indices:

```lean
-- Reduce selected-array membership to the two target
-- equalities the theorem needs.
example :
    jepaLoss maskedIdxs () targetsA
        (fun _ i => guesses i) absDiff =
      jepaLoss maskedIdxs () targetsB
        (fun _ i => guesses i) absDiff := by
  refine jepaLoss_target_ext maskedIdxs () targetsA targetsB
    _ absDiff ?_
  intro i hi
  simp [maskedIdxs] at hi
  rcases hi with h | h <;> subst h <;> rfl
```

Two obligations, `targetsA 0 = targetsB 0` and `targetsA 2 = targetsB 2`, both true by computation.
Now try the same proof with `targetsC`. The `rcases` branch for index `2` leaves
`targetsA 2 = targetsC 2`, that is `20 = 21`. The hypothesis fails at a selected index, and the
computed losses differ. Extensionality requires equality on the selected targets, regardless of
how small a change is.

The proof follows the theorem's quantifier over membership. `intro i hi` introduces an arbitrary
selected position and evidence that it occurs in the array. Simplifying `maskedIdxs` reduces
that evidence to two alternatives; each branch then substitutes its concrete index. The change
at position `1` is never examined because neither alternative reaches it. The experiment with
`targetsC` also shows why “small target change” is a different theorem: extensionality requires
exact equality, while a perturbation bound would need a quantitative continuity assumption on
`repLoss` and would conclude an inequality rather than equal objectives.

This is target-value extensionality, not a stop-gradient theorem. With context, predictor, indices,
and loss fixed, changing only unselected target values leaves the objective unchanged.

# MAE Loss In PyTorch

The reference MAE implementation {Informal.citep mae2022}[] computes its loss like this, using the
framework's ordinary tensor operations {Informal.citep pytorch2019}[]:

```
# Average pixel errors within patches, then normalize over
# the selected patches.
loss = (pred - target) ** 2
loss = loss.mean(dim=-1)
loss = (loss * mask).sum() / mask.sum()
```

Relating this expression to `maskedLoss` requires identifying the selected positions, the
normalization, and the per-patch score.

`mask` is a float tensor of zeros and ones, so masking is multiplication. Multiplying by the
complement of the intended mask produces a perfectly finite number, and so does multiplying by a
mask with a broadcast-compatible but unintended shape. In the Lean version, the objective reads
the supplied `Array (Fin n)` directly. This makes selection explicit and bounds-safe, but choosing
the wrong array remains possible.

The division by `mask.sum()` is the normalization the finite theory deliberately leaves out. It is
also why `loss` over a concatenation of patch groups is not the sum of their mean losses. The
corresponding identity instead weights each group by its selected count.

`mask.sum()` can be zero. If a masking policy selects nothing, the displayed formula divides
zero by zero and produces `nan`; backpropagation can then contaminate the parameter update. The
finite version returns `0` for the empty array, and that is `maskedLoss_nil`.

Finally, the squaring and the `mean(dim=-1)` are the per-patch loss, which in Lean is the
`perPatchLoss` argument. Keeping it as a parameter rather than fixing it to squared error is what
lets the same theorems cover a JEPA latent distance, a quantized patch loss, or a codebook index
mismatch.

The finite theorems describe the selected-index sum. Transferring them to the differentiable
Python objective requires a relation between its tensor mask, its floating per-patch losses,
and its normalization.

# Alignment And Representation Collapse

The predictive-view file also carries a real-valued graph model for studying collapse. A
representation is

$$`z:\operatorname{Fin}(n)\to\mathbb R^d,`

written `EuclideanRep d` for the codomain. Squared distance sums the squared coordinate differences:

```lean (name := sqDistSig)
-- Squared distance returns a real scalar from two
-- coordinate representations.
#check @sqDist
```

```leanOutput sqDistSig
@sqDist : {d : ℕ} → EuclideanRep d → EuclideanRep d → ℝ
```

Two vectors that disagree by one in each of three coordinates are at
squared distance three:

```lean
-- Three unit coordinate differences contribute three
-- squared-error terms.
example :
    sqDist (fun _ : Fin 3 => (0 : ℝ)) (fun _ => (1 : ℝ))
      = 3 := by
  simp [sqDist]
```

An `SSLViewGraph n` stores positive pairs of views, and the alignment energy sums squared distance
over those edges:

$$`E_{\mathrm{align}}(z)
=\sum_{(i,k)\in E_+}\|z_i-z_k\|_2^2.`

Every term is nonnegative, which `graphAlignmentEnergy_nonneg` proves. To see why this objective
permits collapse, consider a representation that assigns the same vector to every view:

```lean (name := collapsedSig)
-- Collapse means one representation vector is shared by
-- every view.
#check @CollapsedRep
```

```leanOutput collapsedSig
@CollapsedRep : {n d : ℕ} → (Fin n → EuclideanRep d) → Prop
```

and for such a representation every edge contributes zero, no matter which edges the graph has:

```lean (name := collapseZero)
-- A collapsed representation makes every positive-pair
-- distance zero.
#check @graphAlignmentEnergy_eq_zero_of_collapsed
```

```leanOutput collapseZero (whitespace := lax)
@graphAlignmentEnergy_eq_zero_of_collapsed : ∀ {n d : ℕ} (graph : SSLViewGraph n)
  (rep : Fin n → EuclideanRep d),
  CollapsedRep rep → graphAlignmentEnergy graph rep = 0
```

Instantiating it takes one line, and the witness is the constant itself:

```lean
-- Supply the constant vector explicitly as the collapse
-- witness.
example (graph : SSLViewGraph 2) :
    graphAlignmentEnergy graph
        (fun _ _ => (7 : ℝ) : Fin 2 → EuclideanRep 3)
      = 0 :=
  graphAlignmentEnergy_eq_zero_of_collapsed graph _
    ⟨fun _ => 7, fun _ => rfl⟩

```

The theorem holds for every graph. Since the energy is nonnegative and every constant
representation attains zero, constant representations are global minimizers of alignment alone.

Graph structure affects how much zero alignment tells us. An edge of zero squared distance
forces its two endpoint representations to agree. Equality can then propagate along paths of
positive pairs. If the graph has disconnected components, each component can take its own
constant value while all alignment terms vanish. The displayed theorem uses the stronger
condition that every view shares one vector, which works for every graph without a connectivity
hypothesis. For an application, the chosen positive pairs therefore determine which differences
alignment can see before any geometry guard is added.

TorchLean's guard is a variance floor over coordinate spread. For a floor $`\gamma`,

$$`G_\gamma(z)
=\sum_{j=0}^{d-1}
\max\!\left(0,\gamma-\operatorname{spread}_j(z)\right),`

and the complete graph objective is $`E_{\mathrm{SSL}}(z)=E_{\mathrm{align}}(z)+G_\gamma(z)`. On a
collapsed representation the spread is zero in every coordinate. For a nonnegative floor, the
guard therefore pays $`d\gamma`. With
four coordinates and a floor of one half that is exactly two:

```lean
-- Four zero-spread coordinates each pay the floor of one
-- half.
example :
    realVarianceFloorGuard (d := 4) (1 / 2 : ℝ)
        (fun _ => 0) = 2 := by
  rw [realVarianceFloorGuard_zero_spread (d := 4)
    (gamma := (1 / 2 : ℝ)) (by norm_num)]
  norm_num
```

which is the arithmetic behind the positivity theorem:

```lean (name := collapsePos)
-- Positive dimension and a positive floor make collapse
-- cost strictly positive.
#check @graphSSLObjective_collapsed_positive
```

```leanOutput collapsePos (whitespace := lax)
@graphSSLObjective_collapsed_positive : ∀ {n d : ℕ} (graph : SSLViewGraph n)
  (rep : Fin n → EuclideanRep d) {gamma : ℝ},
  CollapsedRep rep → 0 < d → 0 < gamma → 0 < graphSSLObjective graph rep gamma
```

Both side conditions are necessary. With a zero floor, a collapsed representation pays nothing:

```lean
-- A zero floor removes the penalty even when every
-- coordinate has zero spread.
example :
    realVarianceFloorGuard (d := 4) (0 : ℝ)
        (fun _ => 0) = 0 := by
  rw [realVarianceFloorGuard_zero_spread (d := 4)
    (gamma := (0 : ℝ)) le_rfl]
  norm_num
```

Take the embedding dimension to zero and there is nothing to guard, whatever the floor:

```lean
-- With no coordinates, the guard is an empty sum for any
-- floor.
example (gamma : ℝ) :
    realVarianceFloorGuard (d := 0) gamma (fun _ => 0)
      = 0 := by
  simp [realVarianceFloorGuard]
```

With zero floor, each zero-spread coordinate has zero penalty. With zero dimension, the sum has no
terms. These calculations explain the positivity theorem's two side conditions.

Positive loss at collapse does not establish that collapsed representations cease to be global
minima. One needs a feasible noncollapsed representation with a smaller total objective. For a
single view, every representation is collapsed regardless of the positive floor:

```lean
-- One view always admits its own representation as a
-- shared-vector witness.
example (rep : Fin 1 → EuclideanRep 3) :
    CollapsedRep rep := by
  refine ⟨rep 0, ?_⟩
  intro i
  fin_cases i
  rfl
```

The witness in `CollapsedRep rep` is one vector shared by all view indices. In the one-view
proof, `rep 0` supplies that witness, and `fin_cases` checks the only possible index. This is
why positive embedding dimension does not ensure a noncollapsed configuration exists: there
must also be enough views to differ. For several views, proving that a guard rules out collapse
would require exhibiting a lower objective value or comparing minimizers under further
conditions. The positive-loss theorem is a useful ingredient in that comparison because it
computes the cost of collapse explicitly.

With several views, alignment costs can also compete with the spread reward.

# Pairwise Spread And Variance

The summary the guard reads is an unnormalized pairwise sum:

```lean (name := spreadSig)
-- Coordinate spread sums squared differences over all
-- ordered view pairs.
#check @coordinateSpread
```

```leanOutput spreadSig
@coordinateSpread : {n d : ℕ} → (Fin n → EuclideanRep d) → Fin d → ℝ
```

$$`\operatorname{spread}_j(z)
=\sum_i\sum_k(z_{ij}-z_{kj})^2.`

This sums ordered pairs without normalization. For two views and one coordinate, the diagonal
pairs contribute zero and the two off-diagonal pairs give twice the squared gap:

```lean
-- The two off-diagonal ordered pairs each contribute the
-- same squared gap.
example (a : Fin 2 → ℝ) :
    coordinateSpread (n := 2) (d := 1) (fun i _ => a i) 0
      = 2 * (a 0 - a 1) ^ 2 := by
  simp [coordinateSpread, Fin.sum_univ_two]
  ring
```

In general the pairwise sum expands to
$`2n\sum_i z_{ij}^2-2\left(\sum_i z_{ij}\right)^2`, which for a nonempty batch is $`2n^2` times
the biased variance of that coordinate. Each squared value occurs in two sums of length `n`;
the cross terms combine into the square of the coordinate sum. This explains the scale factor.

For two scalar views at zero and one, the ordered-pair spread is $`2`, while the biased variance
is $`1/4`; the factor $`2n^2=8` accounts for the difference. Duplicating every view preserves
that biased variance but quadruples the ordered-pair sum, since each old pair now occurs four
times. A fixed floor on the unnormalized spread consequently changes its effective meaning when
the batch is replicated. This is a concrete reason to read the normalization before transferring
a variance threshold between formulations, rather than treating the word “variance” as enough
to identify the quantity.

For a fixed batch, dividing by $`2n^2` would leave the zero-spread condition unchanged. It would
change which nonzero spreads meet a fixed floor, so the floor must be rescaled when comparing
objectives. The Lean guard also uses squared spread directly, without a square root.
$`\sqrt{\cdot}` is not differentiable at zero; the VICReg-style formula below
{Informal.citep vicreg2022}[] adds a positive constant before taking that root:

```
# The square-root offset and variance normalization differ
# from pairwise spread.
std_x = torch.sqrt(x.var(dim=0) + 0.0001)
std_loss = torch.mean(F.relu(1 - std_x))
```

The positive `1e-4` shifts the square-root input away from zero, making that derivative finite.
It also changes the objective and must appear in a theorem about that formula. Note also that
`F.relu(1 - s)` is literally $`\max(0,\gamma-s)` with $`\gamma=1`, so the hinge shape does carry
over exactly; it is the argument of the hinge that differs.

The consequence is that a floor of $`\gamma` in the Lean guard and a floor of $`\gamma` in VICReg
are not the same threshold, and a theorem proved about one does not transfer to the other by
renaming.
Relating them needs the $`2n^2` factor, the biased-versus-unbiased choice, and a bound on the
`sqrt` perturbation. The current finite theory does not supply that correspondence.

# Discrete VICReg Model

{src "NN/MLTheory/SelfSupervised/VICReg.lean"}[`VICReg.lean`]
also contains a simpler `ℕ` model of the guards, using truncated subtraction:

```lean (name := varFloorSig)
-- Natural subtraction implements the discrete shortfall
-- below the floor.
#check @varianceFloorPenalty
```

```leanOutput varFloorSig
varianceFloorPenalty : ℕ → ℕ → ℕ
```

The definition is `gamma - variance`, which over `ℕ` is already the hinge. A coordinate whose
spread exceeds the floor pays nothing:

```lean (name := varOver)
-- A summary above the floor has no shortfall.
#eval varianceFloorPenalty 3 5
```

```leanOutput varOver
0
```

and a coordinate below the floor pays the shortfall:

```lean (name := varUnder)
-- A summary of one falls two units below the floor of
-- three.
#eval varianceFloorPenalty 3 1
```

```leanOutput varUnder
2
```

The natural subtraction `3 - 5 = 0` supplies the hinge's truncation at zero. Written over the
reals, the same calculation is $`\max(0,3-5)`. A separate `max` operation and a nonnegativity
assumption on the penalty are unnecessary in this discrete definition.

Summing over coordinates gives `varianceTerm`. Three fully collapsed coordinates with a floor of
three pay nine:

```lean (name := varTermCollapsed)
-- Three collapsed summaries each contribute a penalty of
-- three.
#eval varianceTerm 3 #[0, 0, 0]
```

```leanOutput varTermCollapsed
9
```

and one collapsed coordinate among three pays three:

```lean (name := varTermOne)
-- Only the middle coordinate falls below the floor in this
-- array.
#eval varianceTerm 3 #[5, 0, 5]
```

```leanOutput varTermOne
3
```

The general statement is that $`d` collapsed coordinates pay $`d\gamma`, which is the discrete twin
of the real theorem from the previous section:

```lean (name := varReplicate)
-- The replicate theorem states the total collapsed cost for
-- any coordinate count.
#check @varianceTerm_replicate_zero
```

```leanOutput varReplicate
varianceTerm_replicate_zero : ∀ (gamma d : ℕ), varianceTerm gamma (Array.replicate d 0) = d * gamma
```

Barlow Twins {Informal.citep barlowtwins2021}[] pushes a cross-correlation matrix toward the
identity, so its penalties are a distance from one on the diagonal and a distance from zero off it.
The discrete diagonal penalty is written `(c - 1) + (1 - c)`, which is absolute difference spelled
with two truncated subtractions:

```lean (name := diagSig)
-- The diagonal penalty measures natural-valued distance
-- from the target one.
#check @diagonalRedundancyPenalty
```

```leanOutput diagSig
diagonalRedundancyPenalty : ℕ → ℕ
```

A collapsed diagonal entry pays one:

```lean (name := diagZero)
-- A collapsed diagonal summary is one unit below its
-- target.
#eval diagonalRedundancyPenalty 0
```

```leanOutput diagZero
1
```

the ideal entry pays nothing:

```lean (name := diagOne)
-- The ideal diagonal summary has zero deviation.
#eval diagonalRedundancyPenalty 1
```

```leanOutput diagOne
0
```

and a summary value above one pays the excess:

```lean (name := diagFive)
-- A summary of five is four units above the target.
#eval diagonalRedundancyPenalty 5
```

```leanOutput diagFive
4
```

The full objective weights the off-diagonal terms. An identity summary is free:

```lean (name := redIdeal)
-- Ideal diagonal and off-diagonal summaries give zero
-- redundancy penalty.
#eval redundancyReductionObjective 2 #[1, 1, 1] #[0, 0]
```

```leanOutput redIdeal
0
```

while collapsing one diagonal entry and leaving `3` of off-diagonal correlation costs the diagonal
unit plus twice the off-diagonal mass:

```lean (name := redBad)
-- One diagonal error plus twice the off-diagonal total
-- gives seven.
#eval redundancyReductionObjective 2 #[1, 0, 1] #[0, 3]
```

```leanOutput redBad
7
```

The Barlow Twins-style tensor formula computes squared penalties from normalized representations:

```
# Build the normalized cross-correlation before scoring
# squared entry deviations.
c = self.bn(z1).T @ self.bn(z2)
c.div_(self.args.batch_size)
on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
off_diag = off_diagonal(c).pow_(2).sum()
loss = on_diag + self.args.lambd * off_diag
```

The shape matches term for term: a diagonal deviation from one, an off-diagonal magnitude, and one
weight. The `ℕ` model omits squaring, batch normalization, and division by batch size; it also
cannot
represent signed correlation entries directly. `redundancyReductionObjective`
therefore takes `diag` and `offDiag` as already-computed summaries, and the theorems about it are
theorems about the penalty algebra, not about the estimator that produced the numbers.
`redundancyReductionObjective_identity` and `redundancyReductionObjective_collapsed_diag_positive`
are the two facts that survive that abstraction, and they are the two that the guard is actually
used for.

For the discrete bad example, the diagonal array contributes exactly one from its middle entry:
$`|1-1|+|0-1|+|1-1|=1`. The off-diagonal summaries sum to three, and the weight two makes their
contribution six, giving seven overall. The two arrays need not even encode a full square matrix
at this interface; their lengths are ordinary runtime array lengths. To connect the expression
to a correlation matrix, a caller must identify which entries were extracted into each array and
what preprocessing produced their summaries. The penalty theorem then handles the arithmetic
on those supplied values.

`VICRegGuard` and `BarlowGuard` in the predictive-view file package these penalties as geometry
guards. `withGeometryGuard` sets the contract's geometry penalty to the supplied value. The
objective adds that penalty to the unchanged predictive term.

# Block Masks And Tensor Objectives

The executable block masker supplies the index arrays used by the finite loss:
{src "NN/API/SelfSupervised/BlockMask.lean"}[`NN/API/SelfSupervised/BlockMask.lean`]
describes a mask by a rank-indexed policy tensor, applies it to a `Tensor Float`, and hands the
hidden positions back as the exact `Array (Fin n)` the theory expects.

The policy is one entry per axis. `none` means the axis does not participate in the block index,
and `some k` groups it into blocks of positive width `k`. The selected block coordinates are
flattened row-major, and one congruence class modulo a positive `period` is hidden. A zero period,
a zero block width, or a policy with no participating axes hides nothing under the implementation.
For a length-four signal cut into blocks of two:

```lean
-- Group the four signal positions into contiguous blocks of
-- width two.
def blocks : TorchLean.Tensor (Option Nat) [1] :=
  TorchLean.Tensor.from #[some 2]

def signal : TorchLean.Tensor Float [4] :=
  TorchLean.Tensor.from #[1.0, 2.0, 3.0, 4.0]
```

Hiding the even congruence class zeroes the first block and leaves the second alone:

```lean (name := blockApply)
-- Offset zero hides the first block under the period-two
-- policy.
#eval TorchLean.Tensor.to
  (BlockMask.apply signal blocks 2 0) (Array Float)
```

```leanOutput blockApply
#[0.000000, 0.000000, 3.000000, 4.000000]
```

That is an executable tensor operation using the block-mask definition. The mask
predicate is separately callable, so a single coordinate can be interrogated. Position `1` is
hidden:

```lean (name := hiddenOne)
-- Position one belongs to the hidden first block.
#eval BlockMask.hidden (shape := [4]) blocks 2 0
  (TorchLean.Tensor.from #[1])
```

```leanOutput hiddenOne
true
```

and position `2` is not:

```lean (name := hiddenTwo)
-- Position two belongs to the visible second block.
#eval BlockMask.hidden (shape := [4]) blocks 2 0
  (TorchLean.Tensor.from #[2])
```

```leanOutput hiddenTwo
false
```

For this one-dimensional policy, positions zero and one have block coordinate zero, and
positions two and three have block coordinate one. Taking the selected congruence class modulo
two therefore hides an entire pair of positions at once. The coordinate argument to `hidden` is
a tensor because a higher-rank input needs one coordinate per axis. In contrast, the output of
`hiddenIndices` uses flattened indices, so it can feed a loss over the complete data size.
Keeping these coordinate systems distinct prevents confusing an axis coordinate with a flattened
patch position.

The materializer turns the same policy into the index array:

```lean (name := hiddenIdxSig)
-- Materialized indices refer to the flattened data size,
-- not the number of axes.
#check @BlockMAE.hiddenIndices
```

```leanOutput hiddenIdxSig
@BlockMAE.hiddenIndices : {dataShape : Spec.Shape} →
  TorchLean.Tensor (Option ℕ) [dataShape.rank] → ℕ → ℕ → Array (Fin dataShape.size)
```

```lean (name := hiddenIdxEven)
-- Materialize the same hidden positions used by the tensor
-- mask.
#eval BlockMAE.hiddenIndices (dataShape := [4]) blocks 2 0
```

```leanOutput hiddenIdxEven
#[0, 1]
```

Shifting the offset selects the complementary block:

```lean (name := hiddenIdxOdd)
-- Offset one selects the other block with the same width
-- and period.
#eval BlockMAE.hiddenIndices (dataShape := [4]) blocks 2 1
```

```leanOutput hiddenIdxOdd
#[2, 3]
```

`#[0, 1]` has type `Array (Fin 4)`, exactly the index type `maskedLoss` takes. The materializer
computes it by running the mask on a tensor of ones and filtering for zeros, reusing
the same mask definition. Matching policy arguments are still essential:
the index type cannot prevent masking with one policy and scoring with another.

`BlockMAE.Proof.rowPredictiveContract` builds a `PredictiveViewContract` for one batch row out of
the masked sample, the materialized hidden indices, and a `Float → Float → ℕ` loss, and
`row_predictive_objective_eq_mae_loss` proves that its objective is the finite `maeLoss`. It
requires
the reconstruction width to be at most the flattened data size; the selected indices and target
lookups are restricted to that prefix. The full signatures are available on hover:

```lean (name := bridgeEntries)
-- Connect one tensor row to the finite MAE objective under
-- the width premise.
#check @BlockMAE.Proof.rowPredictiveContract
#check @BlockMAE.Proof.row_predictive_objective_eq_mae_loss
```

The target function uses `Spec.get` twice to read a row and coordinate of the sample's target
tensor, followed by `.item`; the prediction is `prediction[row][j]`. The theorem identifies the
resulting natural-valued row objective with `maeLoss`, so its append and reverse lemmas apply.
It does not identify the differentiable Float training loss with that summary.

In the row contract, `row : Fin batch` selects a valid sample and `reconstructionWidth` fixes
how many flattened coordinates are eligible as targets. The premise bounding that width ensures
that target lookups stay within the original data. It does not assert that the prefix contains
all hidden positions of the full sample: a narrower reconstruction task intentionally scores
only positions inside that prefix. The `Unit` context in this contract reflects that predictions
are already supplied as a tensor. The proof can therefore compare objective assembly directly,
without modeling an encoder or decoder evaluation inside the context value.

The supplied prediction tensor is arbitrary: the theorem does not prove that a model produced it
from the masked input. Connecting model evaluation, a differentiable loss, and optimizer updates
requires further statements.

# Proof And Runtime Boundary

The current theory establishes:

- finite index and array algebra, with duplicates and order handled explicitly;
- definitional identity of MAE, JEPA, and a common predictive-view contract;
- zero alignment energy for collapsed real representations, for every view graph;
- positivity of the explicit guard at collapse under positive dimension and floor;
- equality of a block-masked tensor row objective with the finite MAE objective.

It does not establish:

- correctness of patch extraction or data augmentation;
- equivalence to a full PyTorch MAE, JEPA, VICReg, or Barlow Twins training script;
- stopped-gradient behavior of a target encoder as a statement about differentiation;
- a correspondence between the coordinate-spread floor and VICReg's standard-deviation floor;
- exclusion of collapsed minimizers, or quality and downstream usefulness of representations;
- floating-point agreement for the runtime objective.

The tensor bridge reuses `Array (Fin n)` for indices and instantiates the existing MAE contract
with tensor lookups. Its proof applies `mae_is_predictive_view_objective` to those arguments.
That explains both the short proof and its scope: the equality concerns the assembled objective,
while mask generation, model evaluation, and gradient behavior need their own specifications.

The objective shapes are motivated by MAE {Informal.citep mae2022}[], I-JEPA
{Informal.citep ijepa2023}[], VICReg {Informal.citep vicreg2022}[], and Barlow Twins
{Informal.citep barlowtwins2021}[]. The Lean statements are narrower than those papers: they
formalize the finite algebra, the degenerate cases, and one executable bridge that the present
TorchLean definitions actually express.
