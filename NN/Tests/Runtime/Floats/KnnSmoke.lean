/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor
public import NN.Spec.Models.Knn

/-!
# KnnSmoke

 Smoke tests for the KNN specification over floats. -/

@[expose] public section

open Spec
open Tensor

namespace Tests
namespace Floats
namespace KNN

def run : IO Unit := do
  let exampleKNN : Spec.KNN Float String 2 :=
    Spec.KNN.fromData Float String 2 2
      [ (vectorTensor (fun i => [1.0, 2.0][i]!), "Class A")
      , (vectorTensor (fun i => [2.0, 3.0][i]!), "Class B")
      , (vectorTensor (fun i => [3.0, 4.0][i]!), "Class A")
      , (vectorTensor (fun i => [5.0, 6.0][i]!), "Class B")
      ]

  let predictedClass :=
    Spec.classify Float String 2 exampleKNN (vectorTensor (fun i => [4.0, 4.5][i]!))
  if predictedClass != "Class A" then
    throw <| IO.userError s!"knn_smoke classification failed: got {predictedClass}"

  let exampleKNNRegression : Spec.KNN Float Float 2 :=
    Spec.KNN.fromData Float Float 2 2
      [ (vectorTensor (fun i => [1.0, 2.0][i]!), 2.5)
      , (vectorTensor (fun i => [2.0, 3.0][i]!), 3.5)
      , (vectorTensor (fun i => [3.0, 4.0][i]!), 4.0)
      , (vectorTensor (fun i => [5.0, 6.0][i]!), 5.5)
      ]

  let predictedValue :=
    Spec.predict Float 2 exampleKNNRegression (vectorTensor (fun i => [3.0, 5.0][i]!))
  if Float.abs (predictedValue - 3.75) > 1e-6 then
    throw <| IO.userError s!"knn_smoke regression failed: got {predictedValue}"

  IO.println "knn_smoke: OK"

end KNN
end Floats
end Tests
/-!
KNN runtime smoke test (floats).

This is a small executable regression check used by CI to ensure basic model components and tensor
ops still run under the float backends.
-/
