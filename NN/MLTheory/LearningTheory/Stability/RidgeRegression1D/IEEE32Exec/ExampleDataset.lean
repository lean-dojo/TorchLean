/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.LearningTheory.Stability.RidgeRegression1D.IEEE32Exec.Core

/-!
# 1D ridge regression under `IEEE32Exec`: example dataset

This module provides a small concrete dataset value of the right type for interactive evaluation and
documentation examples.

It is small by design: its job is to show how the tensor-based dataset encoding plugs into the
executable `IEEE32Exec` ridge-regression algorithm.
-/

@[expose] public section


noncomputable section

namespace NN.MLTheory.LearningTheory.Stability.RidgeRegression1D.IEEE32Exec

open TorchLean.Floats
open TorchLean.Floats.IEEE754

namespace ExampleDataset

/-- Pack a scalar `x` into a length-1 vector tensor (shape `XShape`). -/
def mkVec1 (x : IEEE32Exec) : Spec.Tensor IEEE32Exec XShape :=
  Spec.Tensor.dim (fun _ : Fin 1 => Spec.Tensor.ofScalar x)

/-- A concrete dataset with `N = 2` examples (so `n = 1` in the `N = n+1` convention). -/
def S : Dataset 2 ExampleIEEE32Vec1 :=
  Dataset.ofFn (n := 2) (Z := ExampleIEEE32Vec1) (fun i =>
    Fin.cases
      (mkVec1 (1 : IEEE32Exec), ((2 : Nat) : IEEE32Exec))
      (fun _ => (mkVec1 ((3 : Nat) : IEEE32Exec), ((4 : Nat) : IEEE32Exec)))
      i)

/-- A small regularization parameter for the example dataset. -/
def lam : IEEE32Exec := (1 : IEEE32Exec)

/-- The computed ridge weight on the example dataset (executable IEEE32 semantics). -/
def wHat : IEEE32Exec :=
  ridgeFit1DExecVec1 (n := 1) lam S

end ExampleDataset

end NN.MLTheory.LearningTheory.Stability.RidgeRegression1D.IEEE32Exec
