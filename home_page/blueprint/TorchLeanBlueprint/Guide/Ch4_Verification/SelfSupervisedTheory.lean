import VersoManual
import TorchLean.NN.SelfSupervised
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles
import TorchLean.Tensor.Constructors

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean

-- The block masker lives in `TorchLean.SelfSupervised`. Opening it keeps the displayed `#check`
-- lines inside Verso's narrow code column.
open TorchLean.SelfSupervised

#doc (Manual) "Self-Supervised Objectives" =>
%%%
tag := "self-supervised-theory"
%%%

An MAE loss depends on which patches were hidden. A JEPA loss depends on which branch supplies the
target representation. An alignment objective can be minimized by mapping every view to the same
vector unless another term prevents collapse.

TorchLean isolates the first of these choices in an executable block masker. Its theorems identify
every coordinate of the masked tensor, and the masker materializes the hidden positions as an index
array. TorchLean does not formalize a complete MAE or JEPA training run, and it does not prove that
minimizing one of these objectives learns useful representations.

A plausible loss value can hide a mask read with the wrong polarity. Small examples let us change
the mask policy and calculate its effect. Named Lean blocks and their outputs are checked while the
page is built.

# Block Masks And Tensor Objectives

TorchLean's executable block masker is defined in the self-supervised API:
{src "TorchLean/NN/SelfSupervised.lean"}[`TorchLean/NN/SelfSupervised.lean`] describes a
mask by a rank-indexed policy tensor, applies it to a tensor with scalar storage and a zero value,
and hands the hidden positions back as an `Array (Fin n)` of flattened indices.

A policy has one entry per axis: `none` means the axis doesn't participate in the block index, and
`some k` groups it into blocks of positive width `k`. The selected block coordinates are flattened
row-major, and one congruence class modulo a positive `period` is hidden. A zero period, a zero
block width, or a policy with no participating axes hides nothing under the implementation. For a
length-four signal cut into blocks of two:

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

TorchLean's `BlockMask.apply_scalar_at` theorem identifies every coordinate of the masked tensor: a
hidden entry becomes zero and a visible entry is preserved. Its two corollaries,
`hidden_scalar_eq_zero` and `visible_scalar_eq_input`, expose those cases. In the example above,
the scalar-generic operation is instantiated at `Float`. Called on its own, the mask predicate
reports that position `1` is hidden:

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

For this one-dimensional policy, positions zero and one have block coordinate zero, and positions
two and three have block coordinate one. So taking the selected congruence class modulo two hides
an entire pair of positions at once. `hidden` takes a tensor as its coordinate argument because a
higher-rank input needs one coordinate per axis. In contrast, the output of `hiddenIndices` uses
flattened indices, so it can feed a loss over the complete data size. Keeping these coordinate
systems distinct prevents confusing an axis coordinate with a flattened patch position.

From the same policy, the materializer builds the index array:

```lean (name := hiddenIdxSig)
-- Materialized indices refer to the flattened data size,
-- not the number of axes.
#check @BlockMAE.hiddenIndices
```

```leanOutput hiddenIdxSig
@BlockMAE.hiddenIndices : {dataShape : TorchLean.Shape} →
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

`#[0, 1]` has type `Array (Fin 4)`, indexed by the flattened data size. The materializer
computes it by filtering `Array.finRange` with the shared block-hidden predicate,
evaluated at each flattened position's unlinearized coordinate. Matching policy arguments
are still essential:
the index type cannot prevent masking with one policy and scoring with another.

# Proof And Runtime Boundary

The block-mask theorems establish:

- the value of `BlockMask.apply` at every coordinate (`BlockMask.apply_scalar_at`): zero at a
  hidden in-bounds coordinate, the input entry at a visible one, and no entry out of bounds
- the hidden and visible in-bounds cases, as the corollaries `BlockMask.hidden_scalar_eq_zero` and
  `BlockMask.visible_scalar_eq_input`.

They do not establish:

- correctness of patch extraction or data augmentation
- a link between `BlockMAE.hiddenIndices` and `BlockMask.apply`: the materializer filters flattened
  positions with the same block-hidden predicate, but no theorem states that its indices are the
  positions `apply` zeroes
- equivalence to a full PyTorch MAE, JEPA, VICReg, or Barlow Twins training script
- stopped-gradient behavior of a target encoder as a statement about differentiation
- exclusion of collapsed minimizers, or quality and downstream usefulness of representations
- floating-point agreement for the runtime objective.

The masking setup is motivated by MAE {Informal.citep mae2022}[] and I-JEPA
{Informal.citep ijepa2023}[]. The Lean statements are narrower than those papers: they fix which
positions a mask hides, not what a model learns from them.
