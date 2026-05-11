/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Nodes
public import NN.Proofs.Autograd.Tape.Nodes.Shape
public import NN.Proofs.Autograd.Tape.Util.Idx

public import Mathlib.Analysis.Calculus.Deriv.Add

/-!
# BatchNormChannelFirst

Pointwise analytic correctness for a **channel-first BatchNorm-like** graph.

This matches the existing spec/runtime operator `Spec.batchNorm_channel_first` used by
`Runtime.Autograd.Tape.batchnorm_channel_first`:
it normalizes each channel independently by computing mean/variance over the spatial
dimensions `(H,W)` and then applying per-channel affine parameters `(gamma,beta)`.

The proof is spec-level over `ℝ`. Because the graph uses `sqrt (max x 0)` and `inv`, the
statement is pointwise (`GraphFDerivCorrectAt`) with explicit domain assumptions.

Note: this is *not* PyTorch `BatchNorm2d` over `N×H×W` with running statistics; it is closer to
an InstanceNorm/GroupNorm-style normalization over spatial dimensions per channel.

## PyTorch correspondence / citations
- Reference `BatchNorm2d` and `InstanceNorm2d` docs (for naming/background; this file is a simpler,
  stateless normalization graph).
  https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
  https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html
-/

@[expose] public section


namespace Proofs
namespace Autograd

open Spec

open scoped BigOperators

noncomputable section

namespace BatchNormChannelFirst

open TapeNodes
open TapeNodes.ShapeOps

set_option maxHeartbeats 1200000

/-- Channel-first tensor shape `C×H×W`. -/
abbrev CHWShape (channels height width : Nat) : Shape :=
  .dim channels (.dim height (.dim width .scalar))

/-- Matrix shape `m×n`. -/
abbrev MatShape (m n : Nat) : Shape := .dim m (.dim n .scalar)
/-- Vector shape `k`. -/
abbrev VecShape (k : Nat) : Shape := .dim k .scalar

/-- Input context shapes: `[x, gamma, beta]` with `x : C×H×W` and `gamma/beta : C`. -/
abbrev ΓBN (channels height width : Nat) : List Shape :=
  [CHWShape channels height width, VecShape channels, VecShape channels]

/-- Flattened spatial size `H*W`. -/
abbrev hw (height width : Nat) : Nat := height * width

/-- Prefix intermediates up to `var_eps` (after flattening spatial dimensions). -/
abbrev ssPrefixVarEps (channels height width : Nat) : List Shape :=
  [ MatShape channels (hw height width) -- xMat
  , VecShape channels                  -- mean
  , MatShape channels (hw height width) -- mean_b
  , MatShape channels (hw height width) -- centered
  , MatShape channels (hw height width) -- centered_sq
  , VecShape channels                  -- var
  , VecShape channels                  -- var_eps
  ]

/-- Prefix intermediates up to `std` (adds one more vector). -/
abbrev ssPrefixStd (channels height width : Nat) : List Shape :=
  ssPrefixVarEps channels height width ++ [VecShape channels] -- std

/-- Full list of intermediates for the BatchNormChannelFirst graph in this file. -/
abbrev ssBatchNorm (channels height width : Nat) : List Shape :=
  ssPrefixStd channels height width ++
    [ VecShape channels                   -- inv_std
    , MatShape channels (hw height width) -- inv_std_b
    , MatShape channels (hw height width) -- normalized
    , MatShape channels (hw height width) -- gamma_b
    , MatShape channels (hw height width) -- scaled
    , MatShape channels (hw height width) -- beta_b
    , MatShape channels (hw height width) -- yMat
    , CHWShape channels height width      -- yChw
    ]

/-- Index of the input `x` in the base BatchNorm context `ΓBN channels height width ++ ss`. -/
def idxX {channels height width : Nat} {ss : List Shape} :
    Idx (ΓBN channels height width ++ ss) (CHWShape channels height width) :=
  ⟨⟨0, by simp [ΓBN]⟩, by simp [ΓBN]⟩

/-- Index of the scale vector `gamma` in the base BatchNorm context `ΓBN ++ ss`. -/
def idxGamma {channels height width : Nat} {ss : List Shape} :
    Idx (ΓBN channels height width ++ ss) (VecShape channels) :=
  ⟨⟨1, by simp [ΓBN]⟩, by simp [ΓBN]⟩

/-- Index of the shift vector `beta` in the base BatchNorm context `ΓBN ++ ss`. -/
def idxBeta {channels height width : Nat} {ss : List Shape} :
    Idx (ΓBN channels height width ++ ss) (VecShape channels) :=
  ⟨⟨2, by simp [ΓBN]⟩, by simp [ΓBN]⟩

/-- Index helper for the last element of an extended context `Γ ++ ss ++ [τ]`. -/
def idxLast {Γ : List Shape} {ss : List Shape} {τ : Shape} :
    Idx (Γ ++ ss ++ [τ]) τ :=
  _root_.Proofs.Autograd.Idx.last (Γ := Γ) (ss := ss) (τ := τ)

-- ---------------------------------------------------------------------------
-- BatchNorm graph
-- ---------------------------------------------------------------------------

/-!
Informal computation (per channel, flattening spatial dims):

Let `x : C×H×W`, flatten spatial dims to `xMat : C×(H*W)`. Then for each channel `c`:

`mean_c := (1/(H*W)) * ∑_{p} xMat[c,p]`
`centered := xMat - mean_b`
`var_c := (1/(H*W)) * ∑_{p} centered[c,p]^2`
`std_c := sqrt(var_c + ε)` (implemented as `sqrt_clamp`)
`inv_std_c := 1/std_c`
`normalized := centered ⊙ inv_std_b`
`scaled := normalized ⊙ gamma_b`
`yMat := scaled + beta_b`
`yChw := reshape yMat back to C×H×W`

This is the stateless, per-example normalization used by the runtime spec
`Spec.batchNorm_channel_first`; it is closer to InstanceNorm/GroupNorm than to BatchNorm with
running statistics.
-/

lemma hsz_chw_mat {channels height width : Nat} :
    Shape.size (CHWShape channels height width) = Shape.size (MatShape channels (hw height width))
      := by
  simp [hw, Shape.size]

/-- Reshape `x : C×H×W` into a matrix `xMat : C×(H*W)` (flatten spatial dimensions). -/
def nodeXMat {channels height width : Nat} :
    Node (ΓBN channels height width) (MatShape channels (hw height width)) :=
  reshape
    (Γ := ΓBN channels height width)
    (s₁ := CHWShape channels height width)
    (s₂ := MatShape channels (hw height width))
    (idx := idxX (channels := channels) (height := height) (width := width) (ss := []))
    (h := hsz_chw_mat (channels := channels) (height := height) (width := width))

/-- Graph prefix producing `[xMat]`. -/
def g1 {channels height width : Nat} :
    Graph (ΓBN channels height width) [MatShape channels (hw height width)] :=
  .snoc (.nil) (nodeXMat (channels := channels) (height := height) (width := width))

/-- Index of `xMat` in `ΓBN ++ [xMat]`. -/
def idxXMat {channels height width : Nat} :
    Idx (ΓBN channels height width ++ [MatShape channels (hw height width)]) (MatShape channels (hw
      height width)) :=
  idxLast (Γ := ΓBN channels height width) (ss := []) (τ := MatShape channels (hw height width))

/-- Per-channel mean over spatial dims: `mean : C×(H*W) → C`. -/
def nodeMean {channels height width : Nat} :
    Node (ΓBN channels height width ++ [MatShape channels (hw height width)]) (VecShape channels) :=
  rowMean
    (Γ := ΓBN channels height width ++ [MatShape channels (hw height width)])
    (m := channels) (n := hw height width)
    (idx := idxXMat (channels := channels) (height := height) (width := width))

/-- Graph prefix producing `[xMat, mean]`. -/
def g2 {channels height width : Nat} :
    Graph (ΓBN channels height width) [MatShape channels (hw height width), VecShape channels] :=
  .snoc (g1 (channels := channels) (height := height) (width := width)) (nodeMean (channels :=
    channels) (height := height) (width := width))

/-- Index of `mean` in `ΓBN ++ [xMat, mean]`. -/
def idxMean {channels height width : Nat} :
    Idx (ΓBN channels height width ++ [MatShape channels (hw height width), VecShape channels])
      (VecShape channels) :=
  idxLast (Γ := ΓBN channels height width) (ss := [MatShape channels (hw height width)]) (τ :=
    VecShape channels)

/-- Broadcast `mean` back to `C×(H*W)` (row-wise). -/
def nodeMeanB {channels height width : Nat} :
    Node (ΓBN channels height width ++ [MatShape channels (hw height width), VecShape channels])
      (MatShape channels (hw height width)) :=
  broadcastRow
    (Γ := ΓBN channels height width ++ [MatShape channels (hw height width), VecShape channels])
    (m := channels) (n := hw height width)
    (idx := idxMean (channels := channels) (height := height) (width := width))

/-- Graph prefix producing `[xMat, mean, mean_b]`. -/
def g3 {channels height width : Nat} :
    Graph (ΓBN channels height width)
      [MatShape channels (hw height width), VecShape channels, MatShape channels (hw height width)]
        :=
  .snoc (g2 (channels := channels) (height := height) (width := width)) (nodeMeanB (channels :=
    channels) (height := height) (width := width))

/-- Index of `mean_b` in the extended context. -/
def idxMeanB {channels height width : Nat} :
    Idx (ΓBN channels height width ++
        [MatShape channels (hw height width), VecShape channels, MatShape channels (hw height
          width)]) (MatShape channels (hw height width)) :=
  idxLast
    (Γ := ΓBN channels height width)
    (ss := [MatShape channels (hw height width), VecShape channels])
    (τ := MatShape channels (hw height width))

/-- Index of `xMat` in the extended context at stage `g3`. -/
def idxXMat3 {channels height width : Nat} :
    Idx (ΓBN channels height width ++
        [MatShape channels (hw height width), VecShape channels, MatShape channels (hw height
          width)]) (MatShape channels (hw height width)) :=
  _root_.Proofs.Autograd.Idx.weaken (Γ := ΓBN channels height width ++ [MatShape channels (hw height
    width)]) (idxXMat (channels := channels) (height := height) (width := width))
    (rest := [VecShape channels, MatShape channels (hw height width)])

/-- Center: `centered := xMat - mean_b`. -/
def nodeCentered {channels height width : Nat} :
    Node (ΓBN channels height width ++
        [MatShape channels (hw height width), VecShape channels, MatShape channels (hw height
          width)]) (MatShape channels (hw height width)) :=
  sub
    (Γ := ΓBN channels height width ++ [MatShape channels (hw height width), VecShape channels,
      MatShape channels (hw height width)])
    (s := MatShape channels (hw height width))
    (a := idxXMat3 (channels := channels) (height := height) (width := width))
    (b := idxMeanB (channels := channels) (height := height) (width := width))

/-- Graph prefix producing `[xMat, mean, mean_b, centered]`. -/
def g4 {channels height width : Nat} :
    Graph (ΓBN channels height width)
      [MatShape channels (hw height width), VecShape channels, MatShape channels (hw height width),
       MatShape channels (hw height width)] :=
  .snoc (g3 (channels := channels) (height := height) (width := width)) (nodeCentered (channels :=
    channels) (height := height) (width := width))

/-- Index of `centered` in the extended context at stage `g4`. -/
def idxCentered {channels height width : Nat} :
    Idx (ΓBN channels height width ++
        [MatShape channels (hw height width), VecShape channels, MatShape channels (hw height
          width),
         MatShape channels (hw height width)]) (MatShape channels (hw height width)) :=
  idxLast
    (Γ := ΓBN channels height width)
    (ss := [MatShape channels (hw height width), VecShape channels, MatShape channels (hw height
      width)])
    (τ := MatShape channels (hw height width))

/-- Square `centered`: `centered_sq := centered ⊙ centered`. -/
def nodeCenteredSq {channels height width : Nat} :
    Node (ΓBN channels height width ++
        [MatShape channels (hw height width), VecShape channels, MatShape channels (hw height
          width),
         MatShape channels (hw height width)]) (MatShape channels (hw height width)) :=
  mul
    (Γ := ΓBN channels height width ++
        [MatShape channels (hw height width), VecShape channels, MatShape channels (hw height
          width),
         MatShape channels (hw height width)])
    (s := MatShape channels (hw height width))
    (a := idxCentered (channels := channels) (height := height) (width := width))
    (b := idxCentered (channels := channels) (height := height) (width := width))

/-- Graph prefix producing `[xMat, mean, mean_b, centered, centered_sq]`. -/
def g5 {channels height width : Nat} :
    Graph (ΓBN channels height width)
      [MatShape channels (hw height width), VecShape channels, MatShape channels (hw height width),
       MatShape channels (hw height width), MatShape channels (hw height width)] :=
  .snoc (g4 (channels := channels) (height := height) (width := width)) (nodeCenteredSq (channels :=
    channels) (height := height) (width := width))

/-- Index of `centered_sq` in the extended context at stage `g5`. -/
def idxCenteredSq {channels height width : Nat} :
    Idx (ΓBN channels height width ++
        [MatShape channels (hw height width), VecShape channels, MatShape channels (hw height
          width),
         MatShape channels (hw height width), MatShape channels (hw height width)]) (MatShape
           channels (hw height width)) :=
  idxLast
    (Γ := ΓBN channels height width)
    (ss := [MatShape channels (hw height width), VecShape channels, MatShape channels (hw height
      width),
            MatShape channels (hw height width)])
    (τ := MatShape channels (hw height width))

/-- Per-channel variance over spatial dims: `var := mean(centered_sq)` producing a length-`channels`
  vector. -/
def nodeVar {channels height width : Nat} :
    Node (ΓBN channels height width ++
        [MatShape channels (hw height width), VecShape channels, MatShape channels (hw height
          width),
         MatShape channels (hw height width), MatShape channels (hw height width)]) (VecShape
           channels) :=
  rowMean
    (Γ := ΓBN channels height width ++
        [MatShape channels (hw height width), VecShape channels, MatShape channels (hw height
          width),
         MatShape channels (hw height width), MatShape channels (hw height width)])
    (m := channels) (n := hw height width)
    (idx := idxCenteredSq (channels := channels) (height := height) (width := width))

/-- Graph prefix producing `[xMat, mean, mean_b, centered, centered_sq, var]`. -/
def g6 {channels height width : Nat} :
    Graph (ΓBN channels height width)
      [MatShape channels (hw height width), VecShape channels, MatShape channels (hw height width),
       MatShape channels (hw height width), MatShape channels (hw height width), VecShape channels]
         :=
  .snoc (g5 (channels := channels) (height := height) (width := width)) (nodeVar (channels :=
    channels) (height := height) (width := width))

/-- Index of `var` in the extended context at stage `g6`. -/
def idxVar {channels height width : Nat} :
    Idx (ΓBN channels height width ++
        [MatShape channels (hw height width), VecShape channels, MatShape channels (hw height
          width),
         MatShape channels (hw height width), MatShape channels (hw height width), VecShape
           channels]) (VecShape channels) :=
  idxLast
    (Γ := ΓBN channels height width)
    (ss := [MatShape channels (hw height width), VecShape channels, MatShape channels (hw height
      width),
            MatShape channels (hw height width), MatShape channels (hw height width)])
    (τ := VecShape channels)

/-- Add epsilon: `var_eps := var + ε`. -/
def nodeVarEps {channels height width : Nat} (ε : ℝ) :
    Node (ΓBN channels height width ++
        [MatShape channels (hw height width), VecShape channels, MatShape channels (hw height
          width),
         MatShape channels (hw height width), MatShape channels (hw height width), VecShape
           channels]) (VecShape channels) :=
  elemwise
    (Γ := ΓBN channels height width ++
        [MatShape channels (hw height width), VecShape channels, MatShape channels (hw height
          width),
         MatShape channels (hw height width), MatShape channels (hw height width), VecShape
           channels])
    (s := VecShape channels)
    (idx := idxVar (channels := channels) (height := height) (width := width))
    (fun z => z + ε) (fun _ => 1)

/-- Graph prefix computing `ssPrefixVarEps` (up to `var_eps`). -/
def batchNormPrefixVarEps {channels height width : Nat} (ε : ℝ) :
    Graph (ΓBN channels height width) (ssPrefixVarEps channels height width) :=
  .snoc (g6 (channels := channels) (height := height) (width := width)) (nodeVarEps (channels :=
    channels) (height := height) (width := width) ε)

/-- Index of `var_eps` in `ΓBN ++ ssPrefixVarEps`. -/
def idxVarEps {channels height width : Nat} :
    Idx (ΓBN channels height width ++ ssPrefixVarEps channels height width) (VecShape channels) :=
  idxLast
    (Γ := ΓBN channels height width)
    (ss := [MatShape channels (hw height width), VecShape channels, MatShape channels (hw height
      width),
            MatShape channels (hw height width), MatShape channels (hw height width), VecShape
              channels])
    (τ := VecShape channels)

/--
Standard deviation: `std := sqrt_clamp(var_eps)`.

This is where the development becomes pointwise: differentiability depends on positivity of
  `var_eps`.
-/
def nodeStd {channels height width : Nat} :
    Node (ΓBN channels height width ++ ssPrefixVarEps channels height width) (VecShape channels) :=
  sqrtClamp
    (Γ := ΓBN channels height width ++ ssPrefixVarEps channels height width)
    (s := VecShape channels)
    (idx := idxVarEps (channels := channels) (height := height) (width := width))

/-- Graph prefix computing `ssPrefixStd` (adds `std`). -/
def batchNormPrefixStd {channels height width : Nat} (ε : ℝ) :
    Graph (ΓBN channels height width) (ssPrefixStd channels height width) :=
  .snoc (batchNormPrefixVarEps (channels := channels) (height := height) (width := width) ε)
    (nodeStd (channels := channels) (height := height) (width := width))

/-- Index of `std` in `ΓBN ++ ssPrefixStd`. -/
def idxStd {channels height width : Nat} :
    Idx (ΓBN channels height width ++ ssPrefixStd channels height width) (VecShape channels) :=
  idxLast
    (Γ := ΓBN channels height width)
    (ss := ssPrefixVarEps channels height width)
    (τ := VecShape channels)

/-- Inverse standard deviation: `inv_std := 1/std`. -/
def nodeInvStd {channels height width : Nat} :
    Node (ΓBN channels height width ++ ssPrefixStd channels height width) (VecShape channels) :=
  inv
    (Γ := ΓBN channels height width ++ ssPrefixStd channels height width)
    (s := VecShape channels)
    (idx := idxStd (channels := channels) (height := height) (width := width))

/-- Graph prefix adding `inv_std`. -/
def g8 {channels height width : Nat} (ε : ℝ) :
    Graph (ΓBN channels height width) (ssPrefixStd channels height width ++ [VecShape channels]) :=
  .snoc (batchNormPrefixStd (channels := channels) (height := height) (width := width) ε)
    (nodeInvStd (channels := channels) (height := height) (width := width))

/-- Index of `inv_std` in the extended context after `g8`. -/
def idxInvStd {channels height width : Nat} :
    Idx (ΓBN channels height width ++ (ssPrefixStd channels height width ++ [VecShape channels]))
      (VecShape channels) :=
  idxLast (Γ := ΓBN channels height width) (ss := ssPrefixStd channels height width) (τ := VecShape
    channels)

/-- Broadcast `inv_std` back to `C×(H*W)` (row-wise), producing `inv_std_b`. -/
def nodeInvStdB {channels height width : Nat} :
    Node (ΓBN channels height width ++ (ssPrefixStd channels height width ++ [VecShape channels]))
      (MatShape channels (hw height width)) :=
  broadcastRow
    (Γ := ΓBN channels height width ++ (ssPrefixStd channels height width ++ [VecShape channels]))
    (m := channels) (n := hw height width)
    (idx := idxInvStd (channels := channels) (height := height) (width := width))

/-- Graph prefix adding `inv_std_b`. -/
def g9 {channels height width : Nat} (ε : ℝ) :
    Graph (ΓBN channels height width) (ssPrefixStd channels height width ++ [VecShape channels,
      MatShape channels (hw height width)]) :=
  .snoc (g8 (channels := channels) (height := height) (width := width) ε) (nodeInvStdB (channels :=
    channels) (height := height) (width := width))

/--
Index of `centered` in the context at stage `g9`.

This is obtained by weakening the earlier `idxCentered` along the extended intermediate list.
-/
def idxCentered9 {channels height width : Nat} :
    Idx (ΓBN channels height width ++ (ssPrefixStd channels height width ++ [VecShape channels,
      MatShape channels (hw height width)]))
      (MatShape channels (hw height width)) :=
  -- `centered` sits at the tail of `g4` and is carried forward; weaken its index to the current
  -- context.
  _root_.Proofs.Autograd.Idx.weaken
    (Γ := ΓBN channels height width ++
      [ MatShape channels (hw height width)  -- xMat
      , VecShape channels                   -- mean
      , MatShape channels (hw height width) -- mean_b
      , MatShape channels (hw height width) -- centered
      ])
    (idxCentered (channels := channels) (height := height) (width := width))
    (rest :=
      -- remaining shapes after `centered` inside `ssPrefixStd ++ [inv_std, inv_std_b]`
      [ MatShape channels (hw height width) -- centered_sq
      , VecShape channels                  -- var
      , VecShape channels                  -- var_eps
      , VecShape channels                  -- std
      , VecShape channels                  -- inv_std
      , MatShape channels (hw height width) -- inv_std_b
      ])

/-- Index of `inv_std_b` in the context at stage `g9`. -/
def idxInvStdB9 {channels height width : Nat} :
    Idx (ΓBN channels height width ++ (ssPrefixStd channels height width ++ [VecShape channels,
      MatShape channels (hw height width)]))
      (MatShape channels (hw height width)) :=
  idxLast (Γ := ΓBN channels height width) (ss := ssPrefixStd channels height width ++ [VecShape
    channels]) (τ := MatShape channels (hw height width))

/-- Normalize: `normalized := centered ⊙ inv_std_b`. -/
def nodeNorm {channels height width : Nat} :
    Node (ΓBN channels height width ++ (ssPrefixStd channels height width ++ [VecShape channels,
      MatShape channels (hw height width)]))
      (MatShape channels (hw height width)) :=
  mul
    (Γ := ΓBN channels height width ++ (ssPrefixStd channels height width ++ [VecShape channels,
      MatShape channels (hw height width)]))
    (s := MatShape channels (hw height width))
    (a := idxCentered9 (channels := channels) (height := height) (width := width))
    (b := idxInvStdB9 (channels := channels) (height := height) (width := width))

/-- Graph prefix adding `normalized`. -/
def g10 {channels height width : Nat} (ε : ℝ) :
    Graph (ΓBN channels height width)
      (ssPrefixStd channels height width ++ [VecShape channels, MatShape channels (hw height width),
        MatShape channels (hw height width)]) :=
  .snoc (g9 (channels := channels) (height := height) (width := width) ε) (nodeNorm (channels :=
    channels) (height := height) (width := width))

/-- Index of `normalized` in the extended context at stage `g10`. -/
def idxNorm10 {channels height width : Nat} :
    Idx (ΓBN channels height width ++
      (ssPrefixStd channels height width ++ [VecShape channels, MatShape channels (hw height width),
        MatShape channels (hw height width)]))
      (MatShape channels (hw height width)) :=
  idxLast
    (Γ := ΓBN channels height width)
    (ss := ssPrefixStd channels height width ++ [VecShape channels, MatShape channels (hw height
      width)])
    (τ := MatShape channels (hw height width))

/-- Broadcast `gamma : C` to `C×(H*W)` (row-wise), producing `gamma_b`. -/
def nodeGammaB {channels height width : Nat} :
    Node (ΓBN channels height width ++
      (ssPrefixStd channels height width ++ [VecShape channels, MatShape channels (hw height width),
        MatShape channels (hw height width)]))
      (MatShape channels (hw height width)) :=
  broadcastRow
    (Γ := ΓBN channels height width ++
      (ssPrefixStd channels height width ++ [VecShape channels, MatShape channels (hw height width),
        MatShape channels (hw height width)]))
    (m := channels) (n := hw height width)
    (idx := idxGamma (channels := channels) (height := height) (width := width)
      (ss := ssPrefixStd channels height width ++ [VecShape channels, MatShape channels (hw height
        width), MatShape channels (hw height width)]))

/-- Graph prefix adding `gamma_b`. -/
def g11 {channels height width : Nat} (ε : ℝ) :
    Graph (ΓBN channels height width)
      (ssPrefixStd channels height width ++
        [VecShape channels, MatShape channels (hw height width), MatShape channels (hw height
          width),
         MatShape channels (hw height width)]) :=
  .snoc (g10 (channels := channels) (height := height) (width := width) ε) (nodeGammaB (channels :=
    channels) (height := height) (width := width))

/-- Index of `gamma_b` in the extended context at stage `g11`. -/
def idxGammaB11 {channels height width : Nat} :
    Idx (ΓBN channels height width ++
      (ssPrefixStd channels height width ++
        [VecShape channels, MatShape channels (hw height width), MatShape channels (hw height
          width),
         MatShape channels (hw height width)]))
      (MatShape channels (hw height width)) :=
  idxLast
    (Γ := ΓBN channels height width)
    (ss := ssPrefixStd channels height width ++ [VecShape channels, MatShape channels (hw height
      width), MatShape channels (hw height width)])
    (τ := MatShape channels (hw height width))

/-- Scale: `scaled := normalized ⊙ gamma_b`. -/
def nodeScaled {channels height width : Nat} :
    Node (ΓBN channels height width ++
      (ssPrefixStd channels height width ++
        [VecShape channels, MatShape channels (hw height width), MatShape channels (hw height
          width),
         MatShape channels (hw height width)]))
      (MatShape channels (hw height width)) :=
  mul
    (Γ := ΓBN channels height width ++
      (ssPrefixStd channels height width ++
        [VecShape channels, MatShape channels (hw height width), MatShape channels (hw height
          width),
         MatShape channels (hw height width)]))
    (s := MatShape channels (hw height width))
    (a :=
      _root_.Proofs.Autograd.Idx.weaken
        (Γ := ΓBN channels height width ++
          (ssPrefixStd channels height width ++ [VecShape channels, MatShape channels (hw height
            width), MatShape channels (hw height width)]))
        (idxNorm10 (channels := channels) (height := height) (width := width))
        (rest := [MatShape channels (hw height width)]))
    (b := idxGammaB11 (channels := channels) (height := height) (width := width))

/-- Graph prefix adding `scaled`. -/
def g12 {channels height width : Nat} (ε : ℝ) :
    Graph (ΓBN channels height width)
      (ssPrefixStd channels height width ++
        [VecShape channels, MatShape channels (hw height width), MatShape channels (hw height
          width),
         MatShape channels (hw height width), MatShape channels (hw height width)]) :=
  .snoc (g11 (channels := channels) (height := height) (width := width) ε) (nodeScaled (channels :=
    channels) (height := height) (width := width))

/-- Index of `scaled` in the extended context at stage `g12`. -/
def idxScaled12 {channels height width : Nat} :
    Idx (ΓBN channels height width ++
      (ssPrefixStd channels height width ++
        [VecShape channels, MatShape channels (hw height width), MatShape channels (hw height
          width),
         MatShape channels (hw height width), MatShape channels (hw height width)]))
      (MatShape channels (hw height width)) :=
  idxLast
    (Γ := ΓBN channels height width)
    (ss := ssPrefixStd channels height width ++ [VecShape channels, MatShape channels (hw height
      width), MatShape channels (hw height width),
      MatShape channels (hw height width)])
    (τ := MatShape channels (hw height width))

/-- Broadcast `beta : C` to `C×(H*W)` (row-wise), producing `beta_b`. -/
def nodeBetaB {channels height width : Nat} :
    Node (ΓBN channels height width ++
      (ssPrefixStd channels height width ++
        [VecShape channels, MatShape channels (hw height width), MatShape channels (hw height
          width),
         MatShape channels (hw height width), MatShape channels (hw height width)]))
      (MatShape channels (hw height width)) :=
  broadcastRow
    (Γ := ΓBN channels height width ++
      (ssPrefixStd channels height width ++
        [VecShape channels, MatShape channels (hw height width), MatShape channels (hw height
          width),
         MatShape channels (hw height width), MatShape channels (hw height width)]))
    (m := channels) (n := hw height width)
    (idx := idxBeta (channels := channels) (height := height) (width := width)
      (ss := ssPrefixStd channels height width ++
        [VecShape channels, MatShape channels (hw height width), MatShape channels (hw height
          width),
         MatShape channels (hw height width), MatShape channels (hw height width)]))

/-- Graph prefix adding `beta_b`. -/
def g13 {channels height width : Nat} (ε : ℝ) :
    Graph (ΓBN channels height width)
      (ssPrefixStd channels height width ++
        [VecShape channels, MatShape channels (hw height width), MatShape channels (hw height
          width),
         MatShape channels (hw height width), MatShape channels (hw height width), MatShape channels
           (hw height width)]) :=
  .snoc (g12 (channels := channels) (height := height) (width := width) ε) (nodeBetaB (channels :=
    channels) (height := height) (width := width))

/-- Index of `beta_b` in the extended context at stage `g13`. -/
def idxBetaB13 {channels height width : Nat} :
    Idx (ΓBN channels height width ++
      (ssPrefixStd channels height width ++
        [VecShape channels, MatShape channels (hw height width), MatShape channels (hw height
          width),
         MatShape channels (hw height width), MatShape channels (hw height width), MatShape channels
           (hw height width)]))
      (MatShape channels (hw height width)) :=
  idxLast
    (Γ := ΓBN channels height width)
    (ss := ssPrefixStd channels height width ++
      [VecShape channels, MatShape channels (hw height width), MatShape channels (hw height width),
       MatShape channels (hw height width), MatShape channels (hw height width)])
    (τ := MatShape channels (hw height width))

/-- Add bias: `yMat := scaled + beta_b`. -/
def nodeYMat {channels height width : Nat} :
    Node (ΓBN channels height width ++
      (ssPrefixStd channels height width ++
        [VecShape channels, MatShape channels (hw height width), MatShape channels (hw height
          width),
         MatShape channels (hw height width), MatShape channels (hw height width), MatShape channels
           (hw height width)]))
      (MatShape channels (hw height width)) :=
  add
    (Γ := ΓBN channels height width ++
      (ssPrefixStd channels height width ++
        [VecShape channels, MatShape channels (hw height width), MatShape channels (hw height
          width),
         MatShape channels (hw height width), MatShape channels (hw height width), MatShape channels
           (hw height width)]))
    (s := MatShape channels (hw height width))
    (a :=
      _root_.Proofs.Autograd.Idx.weaken
        (Γ := ΓBN channels height width ++
          (ssPrefixStd channels height width ++
            [VecShape channels, MatShape channels (hw height width), MatShape channels (hw height
              width),
             MatShape channels (hw height width), MatShape channels (hw height width)]))
        (idxScaled12 (channels := channels) (height := height) (width := width))
        (rest := [MatShape channels (hw height width)]))
    (b := idxBetaB13 (channels := channels) (height := height) (width := width))

/-- Graph prefix adding `yMat`. -/
def g14 {channels height width : Nat} (ε : ℝ) :
    Graph (ΓBN channels height width)
      (ssPrefixStd channels height width ++
        [VecShape channels, MatShape channels (hw height width), MatShape channels (hw height
          width),
         MatShape channels (hw height width), MatShape channels (hw height width), MatShape channels
           (hw height width),
         MatShape channels (hw height width)]) :=
  .snoc (g13 (channels := channels) (height := height) (width := width) ε) (nodeYMat (channels :=
    channels) (height := height) (width := width))

/-- Index of `yMat` in the extended context after `g14`. -/
def idxYMat {channels height width : Nat} :
    Idx (ΓBN channels height width ++
      (ssPrefixStd channels height width ++
        [VecShape channels, MatShape channels (hw height width), MatShape channels (hw height
          width),
         MatShape channels (hw height width), MatShape channels (hw height width), MatShape channels
           (hw height width),
         MatShape channels (hw height width)]))
      (MatShape channels (hw height width)) :=
  idxLast
    (Γ := ΓBN channels height width)
    (ss := ssPrefixStd channels height width ++
      [VecShape channels, MatShape channels (hw height width), MatShape channels (hw height width),
       MatShape channels (hw height width), MatShape channels (hw height width), MatShape channels
         (hw height width)])
    (τ := MatShape channels (hw height width))

/-- Reshape the matrix output `yMat : C×(H*W)` back into `yChw : C×H×W`. -/
def nodeYChw {channels height width : Nat} :
    Node (ΓBN channels height width ++
      (ssPrefixStd channels height width ++
        [VecShape channels, MatShape channels (hw height width), MatShape channels (hw height
          width),
         MatShape channels (hw height width), MatShape channels (hw height width), MatShape channels
           (hw height width),
         MatShape channels (hw height width)]))
      (CHWShape channels height width) :=
  reshape
    (Γ := ΓBN channels height width ++
      (ssPrefixStd channels height width ++
        [VecShape channels, MatShape channels (hw height width), MatShape channels (hw height
          width),
         MatShape channels (hw height width), MatShape channels (hw height width), MatShape channels
           (hw height width),
         MatShape channels (hw height width)]))
    (s₁ := MatShape channels (hw height width))
    (s₂ := CHWShape channels height width)
    (idx := idxYMat (channels := channels) (height := height) (width := width))
    (h := (hsz_chw_mat (channels := channels) (height := height) (width := width)).symm)

/-- Full BatchNormChannelFirst graph (explicit snoc chain). -/
def batchNormGraph {channels height width : Nat} (ε : ℝ) :
    Graph (ΓBN channels height width) (ssBatchNorm channels height width) :=
  .snoc (g14 (channels := channels) (height := height) (width := width) ε) (nodeYChw (channels :=
    channels) (height := height) (width := width))

-- ---------------------------------------------------------------------------
-- Pointwise `GraphFDerivCorrectAt`
-- ---------------------------------------------------------------------------

/--
Pointwise proof that `batchNormGraph` satisfies `GraphFDerivCorrectAt`.

As with `LayerNorm`, the hypotheses are explicit domain assumptions needed for differentiability
of `sqrt` (after clamp) and `inv` at the actual execution point.
-/
def batchNormGraphFderivCorrectAt
    {channels height width : Nat} (ε : ℝ) (xV : CtxVec (ΓBN channels height width))
    (hVarEpsPos :
      ∀ i : Fin (Shape.size (VecShape channels)),
        0 < CtxVec.get (Γ := ΓBN channels height width ++ ssPrefixVarEps channels height width)
              (s := VecShape channels)
              (idxVarEps (channels := channels) (height := height) (width := width))
              (Graph.evalVec (Γ := ΓBN channels height width) (ss := ssPrefixVarEps channels height
                width)
                (batchNormPrefixVarEps (channels := channels) (height := height) (width := width) ε)
                  xV) i)
    (hStdNe0 :
      ∀ i : Fin (Shape.size (VecShape channels)),
        CtxVec.get (Γ := ΓBN channels height width ++ ssPrefixStd channels height width)
              (s := VecShape channels)
              (idxStd (channels := channels) (height := height) (width := width))
              (Graph.evalVec (Γ := ΓBN channels height width) (ss := ssPrefixStd channels height
                width)
                (batchNormPrefixStd (channels := channels) (height := height) (width := width) ε)
                  xV) i ≠ 0) :
    GraphFDerivCorrectAt (Γ := ΓBN channels height width) (ss := ssBatchNorm channels height width)
      (batchNormGraph (channels := channels) (height := height) (width := width) ε) xV := by
  classical
  have hg0 : GraphFDerivCorrectAt (Γ := ΓBN channels height width) (ss := []) (.nil) xV :=
    PUnit.unit

  have hg1 :
      GraphFDerivCorrectAt (Γ := ΓBN channels height width) (ss := [MatShape channels (hw height
        width)])
        (g1 (channels := channels) (height := height) (width := width)) xV := by
    refine ⟨hg0, ?_⟩
    exact
      (reshapeFderiv (Γ := ΓBN channels height width)
        (s₁ := CHWShape channels height width) (s₂ := MatShape channels (hw height width))
        (idx := idxX (channels := channels) (height := height) (width := width) (ss := []))
        (h := hsz_chw_mat (channels := channels) (height := height) (width := width))).at
        (Graph.evalVec (Γ := ΓBN channels height width) (ss := []) (.nil) xV)

  have hg2 :
      GraphFDerivCorrectAt (Γ := ΓBN channels height width)
        (ss := [MatShape channels (hw height width), VecShape channels])
        (g2 (channels := channels) (height := height) (width := width)) xV := by
    refine ⟨hg1, ?_⟩
    exact
      (rowMeanFderiv (idx := idxXMat (channels := channels) (height := height) (width :=
        width))).at
        (Graph.evalVec (Γ := ΓBN channels height width)
          (ss := [MatShape channels (hw height width)]) (g1 (channels := channels) (height :=
            height) (width := width)) xV)

  have hg3 :
      GraphFDerivCorrectAt (Γ := ΓBN channels height width)
        (ss := [MatShape channels (hw height width), VecShape channels, MatShape channels (hw height
          width)])
        (g3 (channels := channels) (height := height) (width := width)) xV := by
    refine ⟨hg2, ?_⟩
    exact
      (broadcastRowFderiv (idx := idxMean (channels := channels) (height := height) (width :=
        width))).at
        (Graph.evalVec (Γ := ΓBN channels height width)
          (ss := [MatShape channels (hw height width), VecShape channels]) (g2 (channels :=
            channels) (height := height) (width := width)) xV)

  have hg4 :
      GraphFDerivCorrectAt (Γ := ΓBN channels height width)
        (ss := [MatShape channels (hw height width), VecShape channels, MatShape channels (hw height
          width),
               MatShape channels (hw height width)])
        (g4 (channels := channels) (height := height) (width := width)) xV := by
    refine ⟨hg3, ?_⟩
    exact
      (subFderiv (s := MatShape channels (hw height width))
        (a := idxXMat3 (channels := channels) (height := height) (width := width))
        (b := idxMeanB (channels := channels) (height := height) (width := width))).at
        (Graph.evalVec (Γ := ΓBN channels height width)
          (ss := [MatShape channels (hw height width), VecShape channels, MatShape channels (hw
            height width)])
          (g3 (channels := channels) (height := height) (width := width)) xV)

  have hg5 :
      GraphFDerivCorrectAt (Γ := ΓBN channels height width)
        (ss := [MatShape channels (hw height width), VecShape channels, MatShape channels (hw height
          width),
               MatShape channels (hw height width), MatShape channels (hw height width)])
        (g5 (channels := channels) (height := height) (width := width)) xV := by
    refine ⟨hg4, ?_⟩
    exact
      (mulFderiv (s := MatShape channels (hw height width))
        (a := idxCentered (channels := channels) (height := height) (width := width))
        (b := idxCentered (channels := channels) (height := height) (width := width))).at
        (Graph.evalVec (Γ := ΓBN channels height width)
          (ss := [MatShape channels (hw height width), VecShape channels, MatShape channels (hw
            height width),
                 MatShape channels (hw height width)])
          (g4 (channels := channels) (height := height) (width := width)) xV)

  have hg6 :
      GraphFDerivCorrectAt (Γ := ΓBN channels height width)
        (ss := [MatShape channels (hw height width), VecShape channels, MatShape channels (hw height
          width),
               MatShape channels (hw height width), MatShape channels (hw height width), VecShape
                 channels])
        (g6 (channels := channels) (height := height) (width := width)) xV := by
    refine ⟨hg5, ?_⟩
    exact
      (rowMeanFderiv (idx := idxCenteredSq (channels := channels) (height := height) (width :=
        width))).at
        (Graph.evalVec (Γ := ΓBN channels height width)
          (ss := [MatShape channels (hw height width), VecShape channels, MatShape channels (hw
            height width),
                 MatShape channels (hw height width), MatShape channels (hw height width)])
          (g5 (channels := channels) (height := height) (width := width)) xV)

  have hg7 :
      GraphFDerivCorrectAt (Γ := ΓBN channels height width)
        (ss := ssPrefixVarEps channels height width)
        (batchNormPrefixVarEps (channels := channels) (height := height) (width := width) ε) xV :=
          by
    refine ⟨hg6, ?_⟩
    have hderiv : NodeFDerivCorrect (nodeVarEps (channels := channels) (height := height) (width :=
      width) ε) :=
      elemwiseFderiv
        (Γ := ΓBN channels height width ++
          [MatShape channels (hw height width), VecShape channels, MatShape channels (hw height
            width),
           MatShape channels (hw height width), MatShape channels (hw height width), VecShape
             channels])
        (s := VecShape channels)
        (idx := idxVar (channels := channels) (height := height) (width := width))
        (f := fun z => z + ε)
        (f' := fun _ => 1)
        (hf := fun z => by simpa using (hasDerivAt_id (x := z)).add_const ε)
    exact
      hderiv.at
        (Graph.evalVec (Γ := ΓBN channels height width)
          (ss := [MatShape channels (hw height width), VecShape channels, MatShape channels (hw
            height width),
                 MatShape channels (hw height width), MatShape channels (hw height width), VecShape
                   channels])
          (g6 (channels := channels) (height := height) (width := width)) xV)

  have hg8 :
      GraphFDerivCorrectAt (Γ := ΓBN channels height width)
        (ss := ssPrefixStd channels height width)
        (batchNormPrefixStd (channels := channels) (height := height) (width := width) ε) xV := by
    refine ⟨hg7, ?_⟩
    have hStdAt :
        NodeFDerivCorrectAt (nodeStd (channels := channels) (height := height) (width := width))
          (Graph.evalVec (Γ := ΓBN channels height width) (ss := ssPrefixVarEps channels height
            width)
            (batchNormPrefixVarEps (channels := channels) (height := height) (width := width) ε) xV)
              :=
      sqrtClampFderivAt
        (Γ := ΓBN channels height width ++ ssPrefixVarEps channels height width)
        (s := VecShape channels)
        (idx := idxVarEps (channels := channels) (height := height) (width := width))
        (xV := Graph.evalVec (Γ := ΓBN channels height width) (ss := ssPrefixVarEps channels height
          width)
          (batchNormPrefixVarEps (channels := channels) (height := height) (width := width) ε) xV)
        (hx := hVarEpsPos)
    simpa [batchNormPrefixStd, nodeStd] using hStdAt

  have hg9 :
      GraphFDerivCorrectAt (Γ := ΓBN channels height width)
        (ss := ssPrefixStd channels height width ++ [VecShape channels])
        (g8 (channels := channels) (height := height) (width := width) ε) xV := by
    refine ⟨hg8, ?_⟩
    have hInvAt :
        NodeFDerivCorrectAt (nodeInvStd (channels := channels) (height := height) (width := width))
          (Graph.evalVec (Γ := ΓBN channels height width) (ss := ssPrefixStd channels height width)
            (batchNormPrefixStd (channels := channels) (height := height) (width := width) ε) xV) :=
      invFderivAt
        (Γ := ΓBN channels height width ++ ssPrefixStd channels height width)
        (s := VecShape channels)
        (idx := idxStd (channels := channels) (height := height) (width := width))
        (xV := Graph.evalVec (Γ := ΓBN channels height width) (ss := ssPrefixStd channels height
          width)
          (batchNormPrefixStd (channels := channels) (height := height) (width := width) ε) xV)
        (hx := hStdNe0)
    simpa [g8, nodeInvStd] using hInvAt

  have hg10 :
      GraphFDerivCorrectAt (Γ := ΓBN channels height width)
        (ss := ssPrefixStd channels height width ++ [VecShape channels, MatShape channels (hw height
          width)])
        (g9 (channels := channels) (height := height) (width := width) ε) xV := by
    refine ⟨hg9, ?_⟩
    exact
      (broadcastRowFderiv (idx := idxInvStd (channels := channels) (height := height) (width :=
        width))).at
        (Graph.evalVec (Γ := ΓBN channels height width)
          (ss := ssPrefixStd channels height width ++ [VecShape channels])
          (g8 (channels := channels) (height := height) (width := width) ε) xV)

  have hg11 :
      GraphFDerivCorrectAt (Γ := ΓBN channels height width)
        (ss := ssPrefixStd channels height width ++ [VecShape channels, MatShape channels (hw height
          width), MatShape channels (hw height width)])
        (g10 (channels := channels) (height := height) (width := width) ε) xV := by
    refine ⟨hg10, ?_⟩
    exact
      (mulFderiv (s := MatShape channels (hw height width))
        (a := idxCentered9 (channels := channels) (height := height) (width := width))
        (b := idxInvStdB9 (channels := channels) (height := height) (width := width))).at
        (Graph.evalVec (Γ := ΓBN channels height width)
          (ss := ssPrefixStd channels height width ++ [VecShape channels, MatShape channels (hw
            height width)])
          (g9 (channels := channels) (height := height) (width := width) ε) xV)

  have hg12 :
      GraphFDerivCorrectAt (Γ := ΓBN channels height width)
        (ss := ssPrefixStd channels height width ++
          [VecShape channels, MatShape channels (hw height width), MatShape channels (hw height
            width), MatShape channels (hw height width)])
        (g11 (channels := channels) (height := height) (width := width) ε) xV := by
    refine ⟨hg11, ?_⟩
    exact
      (broadcastRowFderiv
        (idx := idxGamma (channels := channels) (height := height) (width := width)
          (ss := ssPrefixStd channels height width ++ [VecShape channels, MatShape channels (hw
            height width), MatShape channels (hw height width)]))).at
        (Graph.evalVec (Γ := ΓBN channels height width)
          (ss := ssPrefixStd channels height width ++ [VecShape channels, MatShape channels (hw
            height width), MatShape channels (hw height width)])
          (g10 (channels := channels) (height := height) (width := width) ε) xV)

  have hg13 :
      GraphFDerivCorrectAt (Γ := ΓBN channels height width)
        (ss := ssPrefixStd channels height width ++
          [VecShape channels, MatShape channels (hw height width), MatShape channels (hw height
            width),
           MatShape channels (hw height width), MatShape channels (hw height width)])
        (g12 (channels := channels) (height := height) (width := width) ε) xV := by
    refine ⟨hg12, ?_⟩
    exact
      (mulFderiv (s := MatShape channels (hw height width))
        (a :=
          _root_.Proofs.Autograd.Idx.weaken
            (Γ := ΓBN channels height width ++
              (ssPrefixStd channels height width ++ [VecShape channels, MatShape channels (hw height
                width), MatShape channels (hw height width)]))
            (idxNorm10 (channels := channels) (height := height) (width := width))
            (rest := [MatShape channels (hw height width)]))
        (b := idxGammaB11 (channels := channels) (height := height) (width := width))).at
        (Graph.evalVec (Γ := ΓBN channels height width)
          (ss := ssPrefixStd channels height width ++
            [VecShape channels, MatShape channels (hw height width), MatShape channels (hw height
              width), MatShape channels (hw height width)])
          (g11 (channels := channels) (height := height) (width := width) ε) xV)

  have hg14 :
      GraphFDerivCorrectAt (Γ := ΓBN channels height width)
        (ss := ssPrefixStd channels height width ++
          [VecShape channels, MatShape channels (hw height width), MatShape channels (hw height
            width),
           MatShape channels (hw height width), MatShape channels (hw height width), MatShape
             channels (hw height width)])
        (g13 (channels := channels) (height := height) (width := width) ε) xV := by
    refine ⟨hg13, ?_⟩
    exact
      (broadcastRowFderiv
        (idx := idxBeta (channels := channels) (height := height) (width := width)
          (ss := ssPrefixStd channels height width ++
            [VecShape channels, MatShape channels (hw height width), MatShape channels (hw height
              width),
             MatShape channels (hw height width), MatShape channels (hw height width)]))).at
        (Graph.evalVec (Γ := ΓBN channels height width)
          (ss := ssPrefixStd channels height width ++
            [VecShape channels, MatShape channels (hw height width), MatShape channels (hw height
              width),
             MatShape channels (hw height width), MatShape channels (hw height width)])
          (g12 (channels := channels) (height := height) (width := width) ε) xV)

  have hg15 :
      GraphFDerivCorrectAt (Γ := ΓBN channels height width)
        (ss := ssPrefixStd channels height width ++
          [VecShape channels, MatShape channels (hw height width), MatShape channels (hw height
            width),
           MatShape channels (hw height width), MatShape channels (hw height width), MatShape
             channels (hw height width),
           MatShape channels (hw height width)])
        (g14 (channels := channels) (height := height) (width := width) ε) xV := by
    refine ⟨hg14, ?_⟩
    exact
      (addFderiv (s := MatShape channels (hw height width))
        (a :=
          _root_.Proofs.Autograd.Idx.weaken
            (Γ := ΓBN channels height width ++
              (ssPrefixStd channels height width ++
                [VecShape channels, MatShape channels (hw height width), MatShape channels (hw
                  height width),
                 MatShape channels (hw height width), MatShape channels (hw height width)]))
            (idxScaled12 (channels := channels) (height := height) (width := width))
            (rest := [MatShape channels (hw height width)]))
        (b := idxBetaB13 (channels := channels) (height := height) (width := width))).at
        (Graph.evalVec (Γ := ΓBN channels height width)
          (ss := ssPrefixStd channels height width ++
            [VecShape channels, MatShape channels (hw height width), MatShape channels (hw height
              width),
             MatShape channels (hw height width), MatShape channels (hw height width), MatShape
               channels (hw height width)])
          (g13 (channels := channels) (height := height) (width := width) ε) xV)

  refine ⟨hg15, ?_⟩
  exact
    (reshapeFderiv
      (Γ := ΓBN channels height width ++
        (ssPrefixStd channels height width ++
          [VecShape channels, MatShape channels (hw height width), MatShape channels (hw height
            width),
           MatShape channels (hw height width), MatShape channels (hw height width), MatShape
             channels (hw height width),
           MatShape channels (hw height width)]))
      (s₁ := MatShape channels (hw height width))
      (s₂ := CHWShape channels height width)
      (idx := idxYMat (channels := channels) (height := height) (width := width))
      (h := (hsz_chw_mat (channels := channels) (height := height) (width := width)).symm)).at
      (Graph.evalVec (Γ := ΓBN channels height width)
        (ss := ssPrefixStd channels height width ++
          [VecShape channels, MatShape channels (hw height width), MatShape channels (hw height
            width),
           MatShape channels (hw height width), MatShape channels (hw height width), MatShape
             channels (hw height width),
           MatShape channels (hw height width)])
        (g14 (channels := channels) (height := height) (width := width) ε) xV)

/--
Pointwise end-to-end result: backprop equals `(fderiv eval)†` for `batchNormGraph`.

This is the BatchNormChannelFirst analogue of the global DAG theorem, specialized to the explicit
graph construction and with explicit domain assumptions for `sqrt`/`inv`.
-/
theorem backprop_eq_adjoint_fderiv_batchNorm_channel_first_at
    {channels height width : Nat} (ε : ℝ)
    (xV : CtxVec (ΓBN channels height width))
    (seedV : CtxVec (ΓBN channels height width ++ ssBatchNorm channels height width))
    (hVarEpsPos :
      ∀ i : Fin (Shape.size (VecShape channels)),
        0 < CtxVec.get (Γ := ΓBN channels height width ++ ssPrefixVarEps channels height width)
              (s := VecShape channels)
              (idxVarEps (channels := channels) (height := height) (width := width))
              (Graph.evalVec (Γ := ΓBN channels height width) (ss := ssPrefixVarEps channels height
                width)
                (batchNormPrefixVarEps (channels := channels) (height := height) (width := width) ε)
                  xV) i)
    (hStdNe0 :
      ∀ i : Fin (Shape.size (VecShape channels)),
        CtxVec.get (Γ := ΓBN channels height width ++ ssPrefixStd channels height width)
              (s := VecShape channels)
              (idxStd (channels := channels) (height := height) (width := width))
              (Graph.evalVec (Γ := ΓBN channels height width) (ss := ssPrefixStd channels height
                width)
                (batchNormPrefixStd (channels := channels) (height := height) (width := width) ε)
                  xV) i ≠ 0) :
    Graph.backpropVec (Γ := ΓBN channels height width) (ss := ssBatchNorm channels height width)
        (batchNormGraph (channels := channels) (height := height) (width := width) ε) xV seedV
      =
    (fderiv ℝ
        (Graph.evalVec (Γ := ΓBN channels height width) (ss := ssBatchNorm channels height width)
          (batchNormGraph (channels := channels) (height := height) (width := width) ε))
        xV).adjoint seedV := by
  classical
  have hg :
      GraphFDerivCorrectAt (Γ := ΓBN channels height width) (ss := ssBatchNorm channels height
        width)
        (batchNormGraph (channels := channels) (height := height) (width := width) ε) xV :=
    batchNormGraphFderivCorrectAt (channels := channels) (height := height) (width := width)
      ε xV hVarEpsPos hStdNe0
  exact
    Graph.backpropVec_eq_adjoint_fderiv_at (Γ := ΓBN channels height width) (ss := ssBatchNorm
      channels height width)
      (g := batchNormGraph (channels := channels) (height := height) (width := width) ε) xV seedV hg

end BatchNormChannelFirst

end

end Autograd
end Proofs
