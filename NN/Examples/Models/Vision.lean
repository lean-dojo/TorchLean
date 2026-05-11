/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Models.Vision.Cnn
public import NN.Examples.Models.Vision.Resnet
public import NN.Examples.Models.Vision.Vit

/-!
# Vision Model Examples

Runnable image-model examples. These read prepared real image arrays at the Lean boundary. For
CIFAR-10, run `python3 scripts/datasets/download_example_data.py --cifar10`; for ImageNet-style folders, use
`scripts/datasets/torchlean_data_convert.py image-folder`.
-/

@[expose] public section
