/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Module.Activation
public import NN.Spec.Module.Conv
public import NN.Spec.Module.Flatten
public import NN.Spec.Module.Linear
public import NN.Spec.Module.Pooling

/-!
# CNN (spec model)

This file wires together a small CNN in two styles:

1. A `SpecChain` description, which is great for "model wiring" and exporters:

- `cnn_spec`: `Conv2D → MaxPool2D → Conv2D → MaxPool2D → Flatten → Linear`
- `cnn_with_relu_spec`: same, but with ReLU inserted after each conv

2. A fully explicit forward/backward pair (`Models.Full.CNN2Spec`) for the classic training setup:

`Conv → ReLU → MaxPool → Conv → ReLU → MaxPool → Flatten → Linear`

PyTorch mental model (single image, no batch):

```python
nn.Sequential(
  nn.Conv2d(inC, c1, (kH,kW), stride=stride1, padding=padding1),
  nn.ReLU(),
  nn.MaxPool2d((poolKH,poolKW), stride=poolStride1),
  nn.Conv2d(c1, c2, (kH,kW), stride=stride2, padding=padding2),
  nn.ReLU(),
  nn.MaxPool2d((poolKH,poolKW), stride=poolStride2),
  nn.Flatten(),
  nn.Linear(c2 * H2 * W2, outDim),
)
```

All shapes are tracked at the type level; the feature dimension for the final `LinearSpec` is
computed as `Shape.size` of the post-pooling feature map.

This is intended as a reference/specification of model structure, not as a tuned implementation.
-/

@[expose] public section


namespace Models

open ModSpec
open Spec
open Tensor
open Activation

namespace CNN

/-- Output size for a conv along one spatial axis.

Matches the standard PyTorch formula (for `dilation = 1`, `groups = 1`):

`out = (in + 2*padding - k) / stride + 1`.
-/
abbrev convOut (input kernel stride padding : Nat) : Nat :=
  (input + 2 * padding - kernel) / stride + 1

/-- Output size for a pooling op along one spatial axis (no padding).

Matches the standard formula:

`out = (in - k) / stride + 1`.
-/
abbrev poolOut (input kernel stride : Nat) : Nat :=
  (input - kernel) / stride + 1

/-- Output height after the first convolution stage. -/
abbrev outH1 (inH kH stride1 padding1 : Nat) : Nat :=
  convOut inH kH stride1 padding1

/-- Output width after the first convolution stage. -/
abbrev outW1 (inW kW stride1 padding1 : Nat) : Nat :=
  convOut inW kW stride1 padding1

/-- Output height after the first pooling stage. -/
abbrev poolH1 (inH kH stride1 padding1 poolKH poolStride1 : Nat) : Nat :=
  poolOut (outH1 inH kH stride1 padding1) poolKH poolStride1

/-- Output width after the first pooling stage. -/
abbrev poolW1 (inW kW stride1 padding1 poolKW poolStride1 : Nat) : Nat :=
  poolOut (outW1 inW kW stride1 padding1) poolKW poolStride1

/-- Output height after the second convolution stage (after pool1). -/
abbrev outH2 (inH kH stride1 padding1 stride2 padding2 poolKH poolStride1 : Nat) : Nat :=
  convOut (poolH1 inH kH stride1 padding1 poolKH poolStride1) kH stride2 padding2

/-- Output width after the second convolution stage (after pool1). -/
abbrev outW2 (inW kW stride1 padding1 stride2 padding2 poolKW poolStride1 : Nat) : Nat :=
  convOut (poolW1 inW kW stride1 padding1 poolKW poolStride1) kW stride2 padding2

/-- Output height after the second pooling stage. -/
abbrev poolH2 (inH kH stride1 padding1 stride2 padding2 poolKH poolStride1 poolStride2 : Nat) : Nat
  :=
  poolOut (outH2 inH kH stride1 padding1 stride2 padding2 poolKH poolStride1) poolKH poolStride2

/-- Output width after the second pooling stage. -/
abbrev poolW2 (inW kW stride1 padding1 stride2 padding2 poolKW poolStride1 poolStride2 : Nat) : Nat
  :=
  poolOut (outW2 inW kW stride1 padding1 stride2 padding2 poolKW poolStride1) poolKW poolStride2

/-- Feature-map shape after the second pooling stage: `(c2, H2, W2)`. -/
abbrev featShape
  (c2 inH inW kH kW stride1 padding1 stride2 padding2 poolKH poolKW poolStride1 poolStride2 : Nat) :
    Shape :=
  Shape.dim c2
    (.dim (poolH2 inH kH stride1 padding1 stride2 padding2 poolKH poolStride1 poolStride2)
      (.dim (poolW2 inW kW stride1 padding1 stride2 padding2 poolKW poolStride1 poolStride2)
        .scalar))

/-- Flattened feature size after the second pooling stage: `c2 * H2 * W2`. -/
abbrev featSize
  (c2 inH inW kH kW stride1 padding1 stride2 padding2 poolKH poolKW poolStride1 poolStride2 : Nat) :
    Nat :=
  Shape.size (featShape c2 inH inW kH kW stride1 padding1 stride2 padding2 poolKH poolKW poolStride1
    poolStride2)

end CNN

-- CNN model specification using SpecChain composition
/--
CNN `SpecChain` wiring (no activations): `Conv2D -> MaxPool2D -> Conv2D -> MaxPool2D -> Flatten ->
  Linear`.

If you want the "classic" ReLU-after-conv variant, use `cnn_with_relu_spec`.
-/
def cnnSpec
  {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {inC outC inH inW kH kW stride1 padding1 stride2 padding2 poolKH poolKW poolstride1 poolstride2 :
    Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0} {h4 : outC ≠ 0} {h5 : poolKH ≠ 0} {h6 : poolKW ≠ 0}
  {hPoolStride1 : poolstride1 ≠ 0} {hPoolStride2 : poolstride2 ≠ 0}
  (conv1_spec : Conv2DSpec inC outC kH kW stride1 padding1 α h1 h2 h3)
  (conv2_spec : Conv2DSpec outC outC kH kW stride2 padding2 α h4 h2 h3)
  (pool1_spec : MaxPool2DSpec poolKH poolKW poolstride1 h5 h6 hPoolStride1)
  (pool2_spec : MaxPool2DSpec poolKH poolKW poolstride2 h5 h6 hPoolStride2)
  (linearSpec :
    LinearSpec α
      (CNN.featSize outC inH inW kH kW stride1 padding1 stride2 padding2 poolKH poolKW poolstride1
        poolstride2)
      outC) :
  SpecChain α (.dim inC (.dim inH (.dim inW .scalar))) (.dim outC .scalar) :=

  -- Create module specs
  let conv1_module := Conv2DModuleSpec conv1_spec
  let pool1_module := MaxPool2DModuleSpec pool1_spec
  let conv2_module := Conv2DModuleSpec conv2_spec
  let pool2_module := MaxPool2DModuleSpec pool2_spec
  let flatten_module :=
    FlattenModuleSpec α (CNN.featShape outC inH inW kH kW stride1 padding1 stride2 padding2 poolKH
      poolKW poolstride1 poolstride2)
  let linear_module := LinearModuleSpec linearSpec

  -- Compose the chain: Conv1 → Pool1 → Conv2 → Pool2 → Flatten → Linear
  SpecChain.single conv1_module
    |>.composeRight pool1_module
    |>.composeRight conv2_module
    |>.composeRight pool2_module
    |>.composeRight flatten_module
    |>.composeRight linear_module

/-- A slightly more "classic" CNN `SpecChain` with ReLU after each convolution.

PyTorch analogy: insert `nn.ReLU()` between the conv and pool blocks.
-/
def cnnWithReluSpec
  {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {inC outC inH inW kH kW stride1 padding1 stride2 padding2 poolKH poolKW poolstride1 poolstride2 :
    Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0} {h4 : outC ≠ 0} {h5 : poolKH ≠ 0} {h6 : poolKW ≠ 0}
  {hPoolStride1 : poolstride1 ≠ 0} {hPoolStride2 : poolstride2 ≠ 0}
  (conv1_spec : Conv2DSpec inC outC kH kW stride1 padding1 α h1 h2 h3)
  (conv2_spec : Conv2DSpec outC outC kH kW stride2 padding2 α h4 h2 h3)
  (pool1_spec : MaxPool2DSpec poolKH poolKW poolstride1 h5 h6 hPoolStride1)
  (pool2_spec : MaxPool2DSpec poolKH poolKW poolstride2 h5 h6 hPoolStride2)
  (linearSpec :
    LinearSpec α
      (CNN.featSize outC inH inW kH kW stride1 padding1 stride2 padding2 poolKH poolKW poolstride1
        poolstride2)
      outC) :
  SpecChain α (.dim inC (.dim inH (.dim inW .scalar))) (.dim outC .scalar) :=

  -- Create module specs
  let conv1_module := Conv2DModuleSpec conv1_spec
  let relu1_module :=
    ReLUModuleSpec (α := α)
      (.dim outC
        (.dim (CNN.outH1 inH kH stride1 padding1)
          (.dim (CNN.outW1 inW kW stride1 padding1) .scalar)))
  let pool1_module := MaxPool2DModuleSpec pool1_spec
  let conv2_module := Conv2DModuleSpec conv2_spec
  let relu2_module :=
    ReLUModuleSpec (α := α)
      (.dim outC
        (.dim (CNN.outH2 inH kH stride1 padding1 stride2 padding2 poolKH poolstride1)
          (.dim (CNN.outW2 inW kW stride1 padding1 stride2 padding2 poolKW poolstride1) .scalar)))
  let pool2_module := MaxPool2DModuleSpec pool2_spec
  let flatten_module :=
    FlattenModuleSpec α (CNN.featShape outC inH inW kH kW stride1 padding1 stride2 padding2 poolKH
      poolKW poolstride1 poolstride2)
  let linear_module := LinearModuleSpec linearSpec

  -- Compose the chain: Conv1 → ReLU1 → Pool1 → Conv2 → ReLU2 → Pool2 → Flatten → Linear
  SpecChain.single conv1_module
    |>.composeRight relu1_module
    |>.composeRight pool1_module
    |>.composeRight conv2_module
    |>.composeRight relu2_module
    |>.composeRight pool2_module
    |>.composeRight flatten_module
    |>.composeRight linear_module

/-- Run `cnn_spec` forward on a single input image.

This is a convenience wrapper around `SpecChain.forward`.
-/
def cnnForward
  {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {inC outC inH inW kH kW stride1 padding1 stride2 padding2 poolKH poolKW poolstride1 poolstride2 :
    Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0} {h4 : outC ≠ 0} {h5 : poolKH ≠ 0} {h6 : poolKW ≠ 0}
  {hPoolStride1 : poolstride1 ≠ 0} {hPoolStride2 : poolstride2 ≠ 0}
  (conv1_spec : Conv2DSpec inC outC kH kW stride1 padding1 α h1 h2 h3)
  (conv2_spec : Conv2DSpec outC outC kH kW stride2 padding2 α h4 h2 h3)
  (pool1_spec : MaxPool2DSpec poolKH poolKW poolstride1 h5 h6 hPoolStride1)
  (pool2_spec : MaxPool2DSpec poolKH poolKW poolstride2 h5 h6 hPoolStride2)
  (linearSpec :
    LinearSpec α
      (CNN.featSize outC inH inW kH kW stride1 padding1 stride2 padding2 poolKH poolKW poolstride1
        poolstride2)
      outC)
  (x : MultiChannelImage inC inH inW α) :
  Tensor α (.dim outC .scalar) :=
  let net := cnnSpec conv1_spec conv2_spec pool1_spec pool2_spec linearSpec
  SpecChain.forward (α:=α) net x

/-!
## A fully explicit CNN spec with backward pass

The `SpecChain` wiring above is convenient for model diagrams. For training/verification workflows
we also want an
explicit reverse-mode spec that returns parameter gradients.

This section provides a small CNN in the classic:

`Conv → ReLU → MaxPool → Conv → ReLU → MaxPool → Flatten → Linear`

form, with a complete backward pass using the per-layer backward specs:
- `conv2d_backward_spec`,
- `max_pool2d_multi_backward_spec`,
- `linear_backward_spec`,
- elementwise gating for ReLU.
-/

namespace Full

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-!
### Configuration

The explicit `CNN2Spec` below is intended to be a readable, end-to-end training reference.
To keep call sites ergonomic (and avoid hiding numeric architecture choices in long argument lists),
we bundle the architectural hyperparameters into a config record.
-/

/-- Hyperparameters for the small 2-layer CNN (`CNN2Spec`). -/
structure CNN2Config where
  /-- Channels after the first conv. -/
  c1 : Nat := 32
  /-- Channels after the second conv. -/
  c2 : Nat := 64
  /-- Output dimension of the linear head (e.g. number of classes). -/
  outDim : Nat := 10

  /-- Convolution kernel height. -/
  kH : Nat := 3
  /-- Convolution kernel width. -/
  kW : Nat := 3
  /-- Stride of the first conv. -/
  stride1 : Nat := 1
  /-- Padding of the first conv. -/
  padding1 : Nat := 1
  /-- Stride of the second conv. -/
  stride2 : Nat := 1
  /-- Padding of the second conv. -/
  padding2 : Nat := 1

  /-- Pooling kernel height. -/
  poolKH : Nat := 2
  /-- Pooling kernel width. -/
  poolKW : Nat := 2
  /-- Stride of the first pooling op. -/
  poolStride1 : Nat := 2
  /-- Stride of the second pooling op. -/
  poolStride2 : Nat := 2

/-- Well-formedness conditions for `CNN2Config` (the nonzero facts needed by some layer specs). -/
structure CNN2Config.WF (cfg : CNN2Config) : Prop where
  c1_ne0 : cfg.c1 ≠ 0
  c2_ne0 : cfg.c2 ≠ 0
  outDim_ne0 : cfg.outDim ≠ 0
  kH_ne0 : cfg.kH ≠ 0
  kW_ne0 : cfg.kW ≠ 0
  poolKH_ne0 : cfg.poolKH ≠ 0
  poolKW_ne0 : cfg.poolKW ≠ 0
  poolStride1_ne0 : cfg.poolStride1 ≠ 0
  poolStride2_ne0 : cfg.poolStride2 ≠ 0

/-- Default `CNN2Config` (a small "classic" CNN shape). -/
def cnn2DefaultConfig : CNN2Config := {}

/-- `cnn2DefaultConfig` is well-formed. -/
theorem cnn2DefaultConfig_wf : cnn2DefaultConfig.WF := by
  refine
    { c1_ne0 := by decide
      c2_ne0 := by decide
      outDim_ne0 := by decide
      kH_ne0 := by decide
      kW_ne0 := by decide
      poolKH_ne0 := by decide
      poolKW_ne0 := by decide
      poolStride1_ne0 := by decide
      poolStride2_ne0 := by decide }

/-- A small 2-block CNN with an explicit linear head.

Parameters:

- `conv1 : Conv2d(inC -> c1)`
- `conv2 : Conv2d(c1 -> c2)`
- `pool1`, `pool2 : MaxPool2d` (no padding in this spec)
- `head : Linear(c2 * H2 * W2 -> outDim)`

PyTorch mental model:

`Conv → ReLU → MaxPool → Conv → ReLU → MaxPool → Flatten → Linear`.

This structure is used by `forward` and `backward` below.
-/
structure CNN2Spec
  (cfg : CNN2Config) (inC inH inW : Nat)
  (α : Type)
  (h_inC : inC ≠ 0) (hCfg : cfg.WF) where
  conv1 :
    Spec.Conv2DSpec inC cfg.c1 cfg.kH cfg.kW cfg.stride1 cfg.padding1 α h_inC hCfg.kH_ne0 hCfg.kW_ne0
  conv2 :
    Spec.Conv2DSpec cfg.c1 cfg.c2 cfg.kH cfg.kW cfg.stride2 cfg.padding2 α hCfg.c1_ne0 hCfg.kH_ne0
      hCfg.kW_ne0
  pool1 :
    Spec.MaxPool2DSpec cfg.poolKH cfg.poolKW cfg.poolStride1 hCfg.poolKH_ne0 hCfg.poolKW_ne0
      hCfg.poolStride1_ne0
  pool2 :
    Spec.MaxPool2DSpec cfg.poolKH cfg.poolKW cfg.poolStride2 hCfg.poolKH_ne0 hCfg.poolKW_ne0
      hCfg.poolStride2_ne0
  head  :
    Spec.LinearSpec α
      (CNN.featSize cfg.c2 inH inW cfg.kH cfg.kW cfg.stride1 cfg.padding1 cfg.stride2 cfg.padding2
        cfg.poolKH cfg.poolKW cfg.poolStride1 cfg.poolStride2)
      cfg.outDim

/-- Gradients for `CNN2Spec` parameters (returned by `CNN2Spec.backward`). -/
structure CNN2Grads
  (cfg : CNN2Config) (inC inH inW : Nat)
  (α : Type) where
  d_conv1_kernel : Tensor α (.dim cfg.c1 (.dim inC (.dim cfg.kH (.dim cfg.kW .scalar))))
  d_conv1_bias   : Tensor α (.dim cfg.c1 .scalar)
  d_conv2_kernel : Tensor α (.dim cfg.c2 (.dim cfg.c1 (.dim cfg.kH (.dim cfg.kW .scalar))))
  d_conv2_bias   : Tensor α (.dim cfg.c2 .scalar)
  d_head_W       :
    Tensor α (.dim cfg.outDim (.dim (CNN.featSize cfg.c2 inH inW cfg.kH cfg.kW cfg.stride1 cfg.padding1
      cfg.stride2 cfg.padding2 cfg.poolKH cfg.poolKW cfg.poolStride1 cfg.poolStride2) .scalar))
  d_head_b       : Tensor α (.dim cfg.outDim .scalar)

/-- Forward pass for `CNN2Spec`.

This is the "full implementation" version of the model, written directly in terms of the
layer-level specs, rather than via `SpecChain`.
-/
def CNN2Spec.forward
  {cfg : CNN2Config} {inC inH inW : Nat}
  {h_inC : inC ≠ 0} {hCfg : cfg.WF}
  (m : CNN2Spec (α := α) cfg inC inH inW h_inC hCfg)
  (x : Spec.MultiChannelImage inC inH inW α) :
  Tensor α (.dim cfg.outDim .scalar) :=
  let y1 := Spec.conv2dSpec (α := α) m.conv1 x
  let r1 := Activation.reluSpec y1
  let p1 := Spec.maxPool2dMultiSpec (α := α) (layer := m.pool1) (input := r1)
  let y2 := Spec.conv2dSpec (α := α) m.conv2 p1
  let r2 := Activation.reluSpec y2
  let p2 := Spec.maxPool2dMultiSpec (α := α) (layer := m.pool2) (input := r2)
  let flat := Tensor.flattenSpec p2
  Spec.linearSpec (α := α) m.head flat

/-- Backward pass (reverse-mode / VJP) for `CNN2Spec`.

Returns parameter gradients plus the input gradient.

Each step uses the corresponding layer backward spec:

- `linear_backward_spec`
- `max_pool2d_multi_backward_spec`
- `conv2d_backward_spec`

and ReLU uses the standard pointwise gate `dY = dR ⊙ relu'(preact)`.
-/
def CNN2Spec.backward
  {cfg : CNN2Config} {inC inH inW : Nat}
  {h_inC : inC ≠ 0} {hCfg : cfg.WF}
  (m : CNN2Spec (α := α) cfg inC inH inW h_inC hCfg)
  (x : Spec.MultiChannelImage inC inH inW α)
  (grad_output : Tensor α (.dim cfg.outDim .scalar)) :
  (CNN2Grads cfg inC inH inW α ×
   Spec.MultiChannelImage inC inH inW α) :=

  -- Forward reconstruction.
  let y1 := Spec.conv2dSpec (α := α) m.conv1 x
  let r1 := Activation.reluSpec y1
  let p1 := Spec.maxPool2dMultiSpec (α := α) (layer := m.pool1) (input := r1)
  let y2 := Spec.conv2dSpec (α := α) m.conv2 p1
  let r2 := Activation.reluSpec y2
  let p2 := Spec.maxPool2dMultiSpec (α := α) (layer := m.pool2) (input := r2)
  let flat := Tensor.flattenSpec p2

  -- Linear head backward.
  let (dW_head, db_head, d_flat) := Spec.linearBackwardSpec (α := α) m.head flat grad_output

  -- Unflatten back to the pooled feature map.
  let featShape :=
    CNN.featShape cfg.c2 inH inW cfg.kH cfg.kW cfg.stride1 cfg.padding1 cfg.stride2 cfg.padding2
      cfg.poolKH cfg.poolKW cfg.poolStride1 cfg.poolStride2

  let d_p2 : Spec.MultiChannelImage cfg.c2
      (CNN.poolH2 inH cfg.kH cfg.stride1 cfg.padding1 cfg.stride2 cfg.padding2 cfg.poolKH cfg.poolStride1 cfg.poolStride2)
      (CNN.poolW2 inW cfg.kW cfg.stride1 cfg.padding1 cfg.stride2 cfg.padding2 cfg.poolKW cfg.poolStride1 cfg.poolStride2) α :=
    Tensor.unflattenSpec featShape d_flat

  -- Pool2 backward.
  let d_r2 := Spec.maxPool2dMultiBackwardSpec (α := α) (layer := m.pool2) (input := r2)
    (grad_output := d_p2)

  -- ReLU2 backward.
  let d_y2 := mulSpec d_r2 (Activation.reluDerivSpec y2)

  -- Conv2 backward.
  let (d_conv2_kernel, d_conv2_bias, d_p1) :=
    Spec.conv2dBackwardSpec (α := α)
      (inC := cfg.c1) (outC := cfg.c2) (kH := cfg.kH) (kW := cfg.kW)
      (stride := cfg.stride2) (padding := cfg.padding2)
      (inH := (CNN.poolH1 inH cfg.kH cfg.stride1 cfg.padding1 cfg.poolKH cfg.poolStride1))
      (inW := (CNN.poolW1 inW cfg.kW cfg.stride1 cfg.padding1 cfg.poolKW cfg.poolStride1))
      (h1 := hCfg.c1_ne0) (h2 := hCfg.kH_ne0) (h3 := hCfg.kW_ne0)
      m.conv2 p1 d_y2

  -- Pool1 backward.
  let d_r1 := Spec.maxPool2dMultiBackwardSpec (α := α) (layer := m.pool1) (input := r1)
    (grad_output := d_p1)

  -- ReLU1 backward.
  let d_y1 := mulSpec d_r1 (Activation.reluDerivSpec y1)

  -- Conv1 backward.
  let (d_conv1_kernel, d_conv1_bias, d_x) :=
    Spec.conv2dBackwardSpec (α := α)
      (inC := inC) (outC := cfg.c1) (kH := cfg.kH) (kW := cfg.kW)
      (stride := cfg.stride1) (padding := cfg.padding1)
      (inH := inH) (inW := inW)
      (h1 := h_inC) (h2 := hCfg.kH_ne0) (h3 := hCfg.kW_ne0)
      m.conv1 x d_y1

  let grads : CNN2Grads cfg inC inH inW α :=
    { d_conv1_kernel := d_conv1_kernel
      d_conv1_bias := d_conv1_bias
      d_conv2_kernel := d_conv2_kernel
      d_conv2_bias := d_conv2_bias
      d_head_W := dW_head
      d_head_b := db_head }

  (grads, d_x)

end Full

end Models
