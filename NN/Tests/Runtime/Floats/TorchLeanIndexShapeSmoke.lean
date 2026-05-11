/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN
public import NN.Tests.Runtime.Floats.Utils
public import Std

/-!
# TorchLeanIndexShapeSmoke

 Smoke tests for TorchLean indexing and shape helpers over floats. -/

@[expose] public section

open Spec
open Tensor
open NN.API
open Tests.Floats.Utils

namespace Tests
namespace Floats
namespace TorchLeanIndexShapeSmoke

instance : Fact (3 > 0) where
  out := by decide

def gradGatherVec (backend : TorchLean.Backend) : IO (Tensor Float (.dim 3 .scalar)) := do
  let sess ← TorchLean.Session.new (α := Float) (opts := { backend := backend })
  let xVal : Tensor Float (.dim 3 .scalar) := NN.Tensor.tensor1d (α := Float) [1.0, 2.0, 3.0]
  let x ← TorchLean.Session.input sess xVal (name := some "x") (requiresGrad := true)
  let y ← TorchLean.Session.gatherScalar sess (n := 3) x ⟨1, by decide⟩
  let grads ← TorchLean.Session.backwardScalarDenseAll sess y
  TorchLean.Session.grad grads x

def gradGatherScalarNat (backend : TorchLean.Backend) : IO (Tensor Float (.dim 3 .scalar)) := do
  let sess ← TorchLean.Session.new (α := Float) (opts := { backend := backend })
  let xVal : Tensor Float (.dim 3 .scalar) := NN.Tensor.tensor1d (α := Float) [1.0, 2.0, 3.0]
  let x ← TorchLean.Session.input sess xVal (name := some "x") (requiresGrad := true)
  let y ← TorchLean.Session.gatherScalarNat sess (n := 3) x 1
  let grads ← TorchLean.Session.backwardScalarDenseAll sess y
  TorchLean.Session.grad grads x

def gradGatherVecNat (backend : TorchLean.Backend) : IO (Tensor Float (.dim 3 .scalar)) := do
  let sess ← TorchLean.Session.new (α := Float) (opts := { backend := backend })
  let xVal : Tensor Float (.dim 3 .scalar) := NN.Tensor.tensor1d (α := Float) [1.0, 2.0, 3.0]
  let idx : Tensor Nat (.dim 4 .scalar) := NN.Tensor.tensor1d (α := Nat) [2, 0, 2, 10]
  let x ← TorchLean.Session.input sess xVal (name := some "x") (requiresGrad := true)
  let y ← TorchLean.Session.gatherVecNat sess (n := 3) (k := 4) x idx
  let total ← TorchLean.Session.sum sess (sh := .dim 4 .scalar) y
  let grads ← TorchLean.Session.backwardScalarDenseAll sess total
  TorchLean.Session.grad grads x

def gradGatherRowsNat (backend : TorchLean.Backend) : IO (Tensor Float (.dim 3 (.dim 2 .scalar))) :=
  do
  let sess ← TorchLean.Session.new (α := Float) (opts := { backend := backend })
  let xVal : Tensor Float (.dim 3 (.dim 2 .scalar)) :=
    Tensor.dim (fun r => Tensor.dim (fun c => Tensor.scalar (Float.ofNat (r.val * 10 + c.val + 1))))
  let idx : Tensor Nat (.dim 3 .scalar) := NN.Tensor.tensor1d (α := Nat) [2, 10, 2]
  let x ← TorchLean.Session.input sess xVal (name := some "x") (requiresGrad := true)
  let y ← TorchLean.Session.gatherRowsNat sess (rows := 3) (cols := 2) (k := 3) x idx
  let total ← TorchLean.Session.sum sess (sh := .dim 3 (.dim 2 .scalar)) y
  let grads ← TorchLean.Session.backwardScalarDenseAll sess total
  TorchLean.Session.grad grads x

def gradBroadcastScalar (backend : TorchLean.Backend) : IO Float := do
  let sess ← TorchLean.Session.new (α := Float) (opts := { backend := backend })
  let sVal : Tensor Float Shape.scalar := Tensor.scalar 2.0
  let sRef ← TorchLean.Session.input sess sVal (name := some "s") (requiresGrad := true)
  let cb : Shape.CanBroadcastTo Shape.scalar (.dim 4 .scalar) :=
    Shape.CanBroadcastTo.scalar_to_any (.dim 4 .scalar)
  let v ← TorchLean.Session.broadcastTo sess (sh1 := Shape.scalar) (sh2 := .dim 4 .scalar) cb sRef
  let total ← TorchLean.Session.sum sess (sh := .dim 4 .scalar) v
  let grads ← TorchLean.Session.backwardScalarDenseAll sess total
  let dsT ← TorchLean.Session.grad grads sRef
  pure (scalarVal dsT)

def gradReshapeMat (backend : TorchLean.Backend) : IO (Tensor Float (.dim 2 (.dim 3 .scalar))) := do
  let sess ← TorchLean.Session.new (α := Float) (opts := { backend := backend })
  let xVal : Tensor Float (.dim 2 (.dim 3 .scalar)) :=
    Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (Float.ofNat (i.val * 10 + j.val))))
  let x ← TorchLean.Session.input sess xVal (name := some "x") (requiresGrad := true)
  let h : Shape.size (.dim 2 (.dim 3 .scalar)) = Shape.size (.dim 6 .scalar) := by decide
  let y ← TorchLean.Session.reshape sess (sh1 := .dim 2 (.dim 3 .scalar)) (sh2 := .dim 6 .scalar) x
    h
  let total ← TorchLean.Session.sum sess (sh := .dim 6 .scalar) y
  let grads ← TorchLean.Session.backwardScalarDenseAll sess total
  TorchLean.Session.grad grads x

def gradTransposeMat (backend : TorchLean.Backend) : IO (Tensor Float (.dim 2 (.dim 3 .scalar))) :=
  do
  let sess ← TorchLean.Session.new (α := Float) (opts := { backend := backend })
  let xVal : Tensor Float (.dim 2 (.dim 3 .scalar)) :=
    Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (Float.ofNat (i.val + j.val + 1))))
  let x ← TorchLean.Session.input sess xVal (name := some "x") (requiresGrad := true)
  let xt ← TorchLean.Session.transpose2d sess (m := 2) (n := 3) x
  let total ← TorchLean.Session.sum sess (sh := .dim 3 (.dim 2 .scalar)) xt
  let grads ← TorchLean.Session.backwardScalarDenseAll sess total
  TorchLean.Session.grad grads x

def gradReduceMeanVec (backend : TorchLean.Backend) : IO (Tensor Float (.dim 3 .scalar)) := do
  let sess ← TorchLean.Session.new (α := Float) (opts := { backend := backend })
  let xVal : Tensor Float (.dim 3 .scalar) := NN.Tensor.tensor1d (α := Float) [1.0, 2.0, 3.0]
  let x ← TorchLean.Session.input sess xVal (name := some "x") (requiresGrad := true)
  let m ← TorchLean.Session.reduceMean sess (sh := .dim 3 .scalar) 0 x
  let grads ← TorchLean.Session.backwardScalarDenseAll sess m
  TorchLean.Session.grad grads x

def gradScatterAddVec (backend : TorchLean.Backend) : IO (Tensor Float (.dim 3 .scalar) × Float) :=
  do
  let sess ← TorchLean.Session.new (α := Float) (opts := { backend := backend })
  let xVal : Tensor Float (.dim 3 .scalar) := NN.Tensor.tensor1d (α := Float) [1.0, 2.0, 3.0]
  let vVal : Tensor Float Shape.scalar := Tensor.scalar 5.0
  let x ← TorchLean.Session.input sess xVal (name := some "x") (requiresGrad := true)
  let v ← TorchLean.Session.input sess vVal (name := some "v") (requiresGrad := true)
  let y ← TorchLean.Session.scatterAddVec sess (n := 3) x v ⟨2, by decide⟩
  let total ← TorchLean.Session.sum sess (sh := .dim 3 .scalar) y
  let grads ← TorchLean.Session.backwardScalarDenseAll sess total
  let dx ← TorchLean.Session.grad grads x
  let dvT ← TorchLean.Session.grad grads v
  pure (dx, scalarVal dvT)

def run : IO Unit := do
  IO.println "torchlean_index_shape_smoke: begin"

  let gE ← gradGatherVec .eager
  let gC ← gradGatherVec .compiled
  for i in List.finRange 3 do
    assertApprox s!"gather grad[{i.val}] eager/compiled" (vecVal gE i) (vecVal gC i)
  for i in List.finRange 3 do
    assertApprox s!"gather grad[{i.val}] expected" (vecVal gE i) (if i.val = 1 then 1.0 else 0.0)

  let gnE ← gradGatherScalarNat .eager
  let gnC ← gradGatherScalarNat .compiled
  for i in List.finRange 3 do
    assertApprox s!"gather_scalar_nat grad[{i.val}] eager/compiled" (vecVal gnE i) (vecVal gnC i)
    assertApprox s!"gather_scalar_nat grad[{i.val}] expected" (vecVal gnE i) (if i.val = 1 then 1.0
      else 0.0)

  let gvE ← gradGatherVecNat .eager
  let gvC ← gradGatherVecNat .compiled
  for i in List.finRange 3 do
    assertApprox s!"gather_vec_nat dx[{i.val}] eager/compiled" (vecVal gvE i) (vecVal gvC i)
  assertApprox "gather_vec_nat dx[0] expected" (vecVal gvE ⟨0, by decide⟩) 1.0
  assertApprox "gather_vec_nat dx[1] expected" (vecVal gvE ⟨1, by decide⟩) 0.0
  assertApprox "gather_vec_nat dx[2] expected" (vecVal gvE ⟨2, by decide⟩) 2.0

  let grE ← gradGatherRowsNat .eager
  let grC ← gradGatherRowsNat .compiled
  for i in List.finRange 3 do
    for j in List.finRange 2 do
      assertApprox s!"gather_rows_nat dx[{i.val},{j.val}] eager/compiled" (matVal grE i j) (matVal
        grC i j)
  for j in List.finRange 2 do
    assertApprox s!"gather_rows_nat dx[0,{j.val}] expected" (matVal grE ⟨0, by decide⟩ j) 0.0
    assertApprox s!"gather_rows_nat dx[1,{j.val}] expected" (matVal grE ⟨1, by decide⟩ j) 0.0
    assertApprox s!"gather_rows_nat dx[2,{j.val}] expected" (matVal grE ⟨2, by decide⟩ j) 2.0

  let bsE ← gradBroadcastScalar .eager
  let bsC ← gradBroadcastScalar .compiled
  assertApprox "broadcast scalar grad eager/compiled" bsE bsC
  assertApprox "broadcast scalar grad expected" bsE 4.0

  let rE ← gradReshapeMat .eager
  let rC ← gradReshapeMat .compiled
  for i in List.finRange 2 do
    for j in List.finRange 3 do
      assertApprox s!"reshape grad[{i.val},{j.val}] eager/compiled" (matVal rE i j) (matVal rC i j)
      assertApprox s!"reshape grad[{i.val},{j.val}] expected" (matVal rE i j) 1.0

  let tE ← gradTransposeMat .eager
  let tC ← gradTransposeMat .compiled
  for i in List.finRange 2 do
    for j in List.finRange 3 do
      assertApprox s!"transpose grad[{i.val},{j.val}] eager/compiled" (matVal tE i j) (matVal tC i
        j)
      assertApprox s!"transpose grad[{i.val},{j.val}] expected" (matVal tE i j) 1.0

  let mE ← gradReduceMeanVec .eager
  let mC ← gradReduceMeanVec .compiled
  for i in List.finRange 3 do
    assertApprox s!"reduce_mean grad[{i.val}] eager/compiled" (vecVal mE i) (vecVal mC i)
    assertApprox s!"reduce_mean grad[{i.val}] expected" (vecVal mE i) (1.0 / 3.0) 1e-6

  let (sxE, svE) ← gradScatterAddVec .eager
  let (sxC, svC) ← gradScatterAddVec .compiled
  for i in List.finRange 3 do
    assertApprox s!"scatter_add_vec dx[{i.val}] eager/compiled" (vecVal sxE i) (vecVal sxC i)
    assertApprox s!"scatter_add_vec dx[{i.val}] expected" (vecVal sxE i) 1.0
  assertApprox "scatter_add_vec dv eager/compiled" svE svC
  assertApprox "scatter_add_vec dv expected" svE 1.0

  IO.println "torchlean_index_shape_smoke: ok"

end TorchLeanIndexShapeSmoke
end Floats
end Tests
/-!
Indexing/shape regression tests (floats).

These checks focus on shape-manipulating ops and indexing helpers that are easy to break when
refactoring tensor APIs.
-/
