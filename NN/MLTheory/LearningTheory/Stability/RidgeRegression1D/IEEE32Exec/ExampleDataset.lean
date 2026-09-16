/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.LearningTheory.Stability.RidgeRegression1D.IEEE32Exec.Core

/-!
# 1D ridge regression under `ExecFloat.Binary 8 23`: example dataset

This module provides a small concrete dataset value of the right type for interactive evaluation and
documentation examples.

It is small by design: its job is to show how the tensor-based dataset encoding plugs into the
executable `ExecFloat.Binary 8 23` ridge-regression algorithm.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


noncomputable section

namespace NN.MLTheory.LearningTheory.Stability.RidgeRegression1D.IEEE32Exec

open TorchLean.Floats
open TorchLean.Floats.IEEE754

namespace ExampleDataset

/-- Pack a scalar `x` into a length-1 vector tensor (shape `XShape`). -/
def mkVec1 (x : ExecFloat.Binary 8 23) : TorchLean.Tensor (ExecFloat.Binary 8 23) XShape :=
  TorchLean.Tensor.dim (fun _ : Fin 1 => TorchLean.Tensor.scalar x)

/-- A concrete dataset with $N=2$ examples (so $n=1$ in the $N=n+1$ convention). -/
def S : Dataset 2 ExampleIEEE32Vec1 :=
  Dataset.ofFn (n := 2) (Z := ExampleIEEE32Vec1) (fun i =>
    Fin.cases
      (mkVec1 (1 : ExecFloat.Binary 8 23), (2 : ExecFloat.Binary 8 23))
      (fun _ => (mkVec1 (3 : ExecFloat.Binary 8 23), (4 : ExecFloat.Binary 8 23)))
      i)

/-- A small regularization parameter for the example dataset. -/
def lam : ExecFloat.Binary 8 23 := (1 : ExecFloat.Binary 8 23)

/-- The computed ridge weight on the example dataset (executable IEEE32 semantics). -/
def wHat : ExecFloat.Binary 8 23 :=
  ridgeFit1DExecVec1 (n := 1) lam S

end ExampleDataset

end NN.MLTheory.LearningTheory.Stability.RidgeRegression1D.IEEE32Exec
