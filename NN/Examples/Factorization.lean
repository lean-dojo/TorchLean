/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Factorization.Common
public import NN.Examples.Factorization.Cholesky
public import NN.Examples.Factorization.QR
public import NN.Examples.Factorization.Check

/-!
# Matrix Factorization Examples

Public `Tensor.cholesky` and `Tensor.qr` examples over `Float`. They check Cholesky reconstruction,
square and wide reduced QR, full-rank orthonormality, and explicit negative controls. Run all checks
with `lake exe torchlean factorizations`.
-/
