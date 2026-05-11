/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN
public import NN.Spec.Core.TensorOps
public import NN.Tests.Runtime.Floats.Utils
public import Std

/-!
# TorchLeanOpsSmoke

 Smoke tests for TorchLean operator wrappers over the float runtime. -/

@[expose] public section

open Spec
open Tensor
open NN.API
open Tests.Floats.Utils

namespace Tests
namespace Floats
namespace TorchLeanOpsSmoke

def evalMatmul (backend : TorchLean.Backend) : IO (Tensor Float (.dim 2 (.dim 2 .scalar))) := do
  let sess ← TorchLean.Session.new (α := Float) (opts := { backend := backend })
  let a : Tensor Float (.dim 2 (.dim 3 .scalar)) :=
    Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (Float.ofNat (i.val + 2 * j.val + 1))))
  let b : Tensor Float (.dim 3 (.dim 2 .scalar)) :=
    Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (Float.ofNat (3 * i.val + j.val + 1))))
  let aR ← TorchLean.Session.const sess (sh := .dim 2 (.dim 3 .scalar)) a
  let bR ← TorchLean.Session.const sess (sh := .dim 3 (.dim 2 .scalar)) b
  let cR ← TorchLean.Session.matmul sess (m := 2) (n := 3) (p := 2) aR bR
  TorchLean.Session.getValue sess (sh := .dim 2 (.dim 2 .scalar)) cR

def evalConcat (backend : TorchLean.Backend) : IO (Tensor Float (.dim 5 .scalar)) := do
  let sess ← TorchLean.Session.new (α := Float) (opts := { backend := backend })
  let a : Tensor Float (.dim 2 .scalar) := Tensor.dim (fun i => Tensor.scalar (Float.ofNat (i.val +
    1)))
  let b : Tensor Float (.dim 3 .scalar) := Tensor.dim (fun i => Tensor.scalar (10.0 + Float.ofNat
    i.val))
  let aR ← TorchLean.Session.const sess (sh := .dim 2 .scalar) a
  let bR ← TorchLean.Session.const sess (sh := .dim 3 .scalar) b
  let cR ← TorchLean.Session.concatVectors sess (n := 2) (m := 3) aR bR
  TorchLean.Session.getValue sess (sh := .dim 5 .scalar) cR

def evalMaxPool (backend : TorchLean.Backend) : IO (Tensor Float (.dim 1 (.dim 2 (.dim 2 .scalar))))
  := do
  let sess ← TorchLean.Session.new (α := Float) (opts := { backend := backend })
  let x : Tensor Float (.dim 1 (.dim 4 (.dim 4 .scalar))) :=
    Tensor.dim (fun _c =>
      Tensor.dim (fun i =>
        Tensor.dim (fun j =>
          Tensor.scalar (Float.ofNat (i.val * 10 + j.val)))))
  let xR ← TorchLean.Session.const sess (sh := .dim 1 (.dim 4 (.dim 4 .scalar))) x
  let yR ← TorchLean.Session.maxPool2d sess (kH := 2) (kW := 2) (inH := 4) (inW := 4) (inC := 1)
    (stride := 2)
    (h1 := by decide) (h2 := by decide) xR
  TorchLean.Session.getValue sess (sh := .dim 1 (.dim 2 (.dim 2 .scalar))) yR

def evalAvgPool (backend : TorchLean.Backend) : IO (Tensor Float (.dim 1 (.dim 2 (.dim 2 .scalar))))
  := do
  let sess ← TorchLean.Session.new (α := Float) (opts := { backend := backend })
  let x : Tensor Float (.dim 1 (.dim 4 (.dim 4 .scalar))) :=
    Tensor.dim (fun _c =>
      Tensor.dim (fun i =>
        Tensor.dim (fun j =>
          Tensor.scalar (Float.ofNat (i.val * 10 + j.val)))))
  let xR ← TorchLean.Session.const sess (sh := .dim 1 (.dim 4 (.dim 4 .scalar))) x
  let yR ← TorchLean.Session.avgPool2d sess (kH := 2) (kW := 2) (inH := 4) (inW := 4) (inC := 1)
    (stride := 2)
    (by decide) (by decide) xR
  TorchLean.Session.getValue sess (sh := .dim 1 (.dim 2 (.dim 2 .scalar))) yR

def run : IO Unit := do
  IO.println "torchlean_ops_smoke: begin"

  let mmE ← evalMatmul .eager
  let mmC ← evalMatmul .compiled
  for i in List.finRange 2 do
    for j in List.finRange 2 do
      assertApprox s!"matmul[{i.val},{j.val}] eager/compiled" (matVal mmE i j) (matVal mmC i j) 1e-5

  let cvE ← evalConcat .eager
  let cvC ← evalConcat .compiled
  for i in List.finRange 5 do
    assertApprox s!"concat[{i.val}] eager/compiled" (vecVal cvE i) (vecVal cvC i) 1e-5

  let mpE ← evalMaxPool .eager
  let mpC ← evalMaxPool .compiled
  for hi in List.finRange 2 do
    for wi in List.finRange 2 do
      assertApprox s!"max_pool2d[{hi.val},{wi.val}] eager/compiled"
        (chwVal mpE ⟨0, by decide⟩ hi wi)
        (chwVal mpC ⟨0, by decide⟩ hi wi)
        1e-5

  let apE ← evalAvgPool .eager
  let apC ← evalAvgPool .compiled
  for hi in List.finRange 2 do
    for wi in List.finRange 2 do
      assertApprox s!"avg_pool2d[{hi.val},{wi.val}] eager/compiled"
        (chwVal apE ⟨0, by decide⟩ hi wi)
        (chwVal apC ⟨0, by decide⟩ hi wi)
        1e-5

  IO.println "torchlean_ops_smoke: ok"

end TorchLeanOpsSmoke
end Floats
end Tests
/-!
TorchLean op-surface smoke tests (floats).

This file exercises a broad subset of the runtime op surface to catch missing instances, backend
breakage, and shape mismatches early.
-/
