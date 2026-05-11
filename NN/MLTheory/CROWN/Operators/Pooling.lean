/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Core
public import NN.Spec.Core.Tensor
public import NN.Spec.Core.TensorOps

/-!
# Pooling operator bounds

This file provides simple interval-bound-propagation (IBP) transfer rules for 2D max/average
pooling. These rules are used by the graph-level CROWN/LiRPA development when a model contains
pooling operators.

## References

- IBP: Gowal et al., *On the Effectiveness of Interval Bound Propagation for Training Verifiably
  Robust Models*, 2018. (arXiv:1810.12715)
- CROWN: Zhang et al., *Efficient Neural Network Robustness Certification with General Activation
  Functions*, NeurIPS 2018. (arXiv:1811.00866)
- auto_LiRPA: Xu et al., *Automatic Perturbation Analysis for Scalable Certified Robustness and
  Beyond*, NeurIPS 2020. (arXiv:2002.12920)
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Operators

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN

variable {α : Type} [Context α]

/--
Configuration for 2D pooling.

This matches the common `(kernel, stride, padding)` parameters used by deep-learning libraries.
-/
structure Pool2DConfig where
  /-- Kernel height. -/
  kH : Nat
  /-- Kernel width. -/
  kW : Nat
  /-- Stride (used for both height and width). -/
  stride : Nat
  /-- Padding (used for all sides). -/
  padding : Nat

/-- Compute output dimensions for pooling -/
def poolOutputSize (inSize padding kSize stride : Nat) : Nat :=
  if stride = 0 then 0 else (inSize + 2 * padding - kSize) / stride + 1

/-!
# Max Pooling IBP

For max pooling, the interval bounds are:
- Lower bound: max of all lower bounds in the window
- Upper bound: max of all upper bounds in the window

We use `Numbers.neg_thousand` as a practical lower bound for initialization
since the Context typeclass doesn't provide negative infinity.
-/

/-- Get element from 2D tensor with bounds checking, returns default if out of bounds -/
def get2D {h w : Nat} (t : Tensor α (.dim h (.dim w .scalar))) (i j : Nat) (default : α) : α :=
  if hi : i < h then
    match t with
    | .dim rows =>
      if hj : j < w then
        match rows ⟨i, hi⟩ with
        | .dim cols =>
          match cols ⟨j, hj⟩ with
          | .scalar v => v
      else default
  else default

/-- IBP for MaxPool2D on a single channel 2D input.
    Input shape: (H, W), Output shape: (outH, outW) -/
def ibpMaxpool2dChannel {inH inW : Nat}
    (cfg : Pool2DConfig)
    (xB : Box α (.dim inH (.dim inW .scalar))) :
    Box α (.dim (poolOutputSize inH cfg.padding cfg.kH cfg.stride)
              (.dim (poolOutputSize inW cfg.padding cfg.kW cfg.stride) .scalar)) :=
  let outH := poolOutputSize inH cfg.padding cfg.kH cfg.stride
  let outW := poolOutputSize inW cfg.padding cfg.kW cfg.stride
  -- Use a very negative number as initial max value
  let negInit : α := Numbers.neg_thousand
  let outLo := Tensor.dim (fun i : Fin outH =>
    Tensor.dim (fun j : Fin outW =>
      -- Find max of lower bounds in window
      let maxLo : α := (List.range cfg.kH).foldl (fun acc di =>
        (List.range cfg.kW).foldl (fun acc dj =>
          let pi := i.val * cfg.stride + di
          let pj := j.val * cfg.stride + dj
          -- Check if within padded bounds
          if pi ≥ cfg.padding ∧ pj ≥ cfg.padding then
            let ii := pi - cfg.padding
            let jj := pj - cfg.padding
            if ii < inH ∧ jj < inW then
              let v := get2D xB.lo ii jj negInit
              if v > acc then v else acc
            else acc
          else acc
        ) acc
      ) negInit
      Tensor.scalar maxLo))
  let outHi := Tensor.dim (fun i : Fin outH =>
    Tensor.dim (fun j : Fin outW =>
      -- Find max of upper bounds in window
      let maxHi : α := (List.range cfg.kH).foldl (fun acc di =>
        (List.range cfg.kW).foldl (fun acc dj =>
          let pi := i.val * cfg.stride + di
          let pj := j.val * cfg.stride + dj
          if pi ≥ cfg.padding ∧ pj ≥ cfg.padding then
            let ii := pi - cfg.padding
            let jj := pj - cfg.padding
            if ii < inH ∧ jj < inW then
              let v := get2D xB.hi ii jj negInit
              if v > acc then v else acc
            else acc
          else acc
        ) acc
      ) negInit
      Tensor.scalar maxHi))
  { lo := outLo, hi := outHi }

/-!
# Average Pooling IBP

For average pooling, the interval bounds are:
- Lower bound: average of all lower bounds in the window
- Upper bound: average of all upper bounds in the window
-/

/-- IBP for AvgPool2D on a single channel 2D input. -/
def ibpAvgpool2dChannel {inH inW : Nat}
    (cfg : Pool2DConfig)
    (xB : Box α (.dim inH (.dim inW .scalar))) :
    Box α (.dim (poolOutputSize inH cfg.padding cfg.kH cfg.stride)
              (.dim (poolOutputSize inW cfg.padding cfg.kW cfg.stride) .scalar)) :=
  let outH := poolOutputSize inH cfg.padding cfg.kH cfg.stride
  let outW := poolOutputSize inW cfg.padding cfg.kW cfg.stride
  let windowSize : α := (cfg.kH * cfg.kW : Nat)
  let outLo := Tensor.dim (fun i : Fin outH =>
    Tensor.dim (fun j : Fin outW =>
      -- Sum lower bounds in window and divide by window size
      let sumLo : α := (List.range cfg.kH).foldl (fun acc di =>
        (List.range cfg.kW).foldl (fun acc dj =>
          let pi := i.val * cfg.stride + di
          let pj := j.val * cfg.stride + dj
          if pi ≥ cfg.padding ∧ pj ≥ cfg.padding then
            let ii := pi - cfg.padding
            let jj := pj - cfg.padding
            if ii < inH ∧ jj < inW then
              let v := get2D xB.lo ii jj Numbers.zero
              acc + v
            else acc  -- Padding treated as 0
          else acc
        ) acc
      ) Numbers.zero
      Tensor.scalar (sumLo / windowSize)))
  let outHi := Tensor.dim (fun i : Fin outH =>
    Tensor.dim (fun j : Fin outW =>
      let sumHi : α := (List.range cfg.kH).foldl (fun acc di =>
        (List.range cfg.kW).foldl (fun acc dj =>
          let pi := i.val * cfg.stride + di
          let pj := j.val * cfg.stride + dj
          if pi ≥ cfg.padding ∧ pj ≥ cfg.padding then
            let ii := pi - cfg.padding
            let jj := pj - cfg.padding
            if ii < inH ∧ jj < inW then
              let v := get2D xB.hi ii jj Numbers.zero
              acc + v
            else acc
          else acc
        ) acc
      ) Numbers.zero
      Tensor.scalar (sumHi / windowSize)))
  { lo := outLo, hi := outHi }

/-!
# Global Pooling

Global max/average pooling reduces entire spatial dimensions to a single value.
-/

/-- Get element from 3D tensor (C, H, W) -/
def get3D {c h w : Nat} (t : Tensor α (.dim c (.dim h (.dim w .scalar)))) (ch i j : Nat) (default :
  α) : α :=
  if hc : ch < c then
    match t with
    | .dim chans =>
      if hi : i < h then
        match chans ⟨ch, hc⟩ with
        | .dim rows =>
          if hj : j < w then
            match rows ⟨i, hi⟩ with
            | .dim cols =>
              match cols ⟨j, hj⟩ with
              | .scalar v => v
          else default
      else default
  else default

/-- IBP for Global Average Pooling -/
def ibpGlobalAvgpool {c h w : Nat} (xB : Box α (.dim c (.dim h (.dim w .scalar)))) :
    Box α (.dim c .scalar) :=
  let spatialSize : α := (h * w : Nat)
  let outLo := Tensor.dim (fun ch : Fin c =>
    let sumLo := (List.range h).foldl (fun acc i =>
      (List.range w).foldl (fun acc j =>
        let v := get3D xB.lo ch.val i j Numbers.zero
        acc + v
      ) acc
    ) Numbers.zero
    Tensor.scalar (sumLo / spatialSize))
  let outHi := Tensor.dim (fun ch : Fin c =>
    let sumHi := (List.range h).foldl (fun acc i =>
      (List.range w).foldl (fun acc j =>
        let v := get3D xB.hi ch.val i j Numbers.zero
        acc + v
      ) acc
    ) Numbers.zero
    Tensor.scalar (sumHi / spatialSize))
  { lo := outLo, hi := outHi }

/-- IBP for Global Max Pooling -/
def ibpGlobalMaxpool {c h w : Nat} (xB : Box α (.dim c (.dim h (.dim w .scalar)))) :
    Box α (.dim c .scalar) :=
  let negInit : α := Numbers.neg_thousand
  let outLo := Tensor.dim (fun ch : Fin c =>
    let maxLo := (List.range h).foldl (fun acc i =>
      (List.range w).foldl (fun acc j =>
        let v := get3D xB.lo ch.val i j negInit
        if v > acc then v else acc
      ) acc
    ) negInit
    Tensor.scalar maxLo)
  let outHi := Tensor.dim (fun ch : Fin c =>
    let maxHi := (List.range h).foldl (fun acc i =>
      (List.range w).foldl (fun acc j =>
        let v := get3D xB.hi ch.val i j negInit
        if v > acc then v else acc
      ) acc
    ) negInit
    Tensor.scalar maxHi)
  { lo := outLo, hi := outHi }

end NN.MLTheory.CROWN.Operators
