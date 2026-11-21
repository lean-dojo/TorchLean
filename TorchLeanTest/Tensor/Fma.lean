/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.IR.Builder
public import TorchLean.NN.PyTorch.Export.IR
public import TorchLean.Tensor.ConfiguredBinary
public import TorchLean.Tensor.Conversion
public import TorchLean.Tensor.Op.Need
public meta import TorchLean.NN.IR.Builder
public meta import TorchLean.NN.PyTorch.Export.IR
public meta import TorchLean.Tensor.ConfiguredBinary
public meta import TorchLean.Tensor.Conversion
public meta import TorchLean.Tensor.Op.Need

/-!
# `fma` rounds once

In binary32 and binary64 the rule of `Op.fma` rounds the exact `x y + z` once (IEEE 754-2019
5.4.1): `(1 + 2⁻²³)² - (1 + 2⁻²²)` is `2⁻⁴⁶`, where rounding the product first gives `0`; the
product `2 × 0x7f7fffff` does not overflow on its own, so subtracting `0x7f7fffff` gives it back;
and `2⁻⁷⁵ × 2⁻⁷⁵ + 2⁻¹⁴⁹`, which lies halfway between the subnormals `2⁻¹⁴⁹` and `2⁻¹⁴⁸`, rounds
to the even `2⁻¹⁴⁸`, where rounding the product first gives `0` and then `2⁻¹⁴⁹`. An exact zero
sum is `+0` and the sum of two `-0` is `-0` (6.3), and an infinity times a zero, or opposite
infinities in the sum, give a NaN (7.2). The pullbacks of the seed `1` are `y`, `x` and `1`.
Native `Float32` and `Float`, through C's `fmaf` and `fma`, give the same words. The wire record
of `Op.fma` is tagged `fma` with the shape, and the PyTorch exporter refuses it, since PyTorch has
no fused multiply-add.
-/

@[expose] public section

namespace TorchLean.Test.Fma

open FloatLib.Floats

/-- The words of the result and of the three pullbacks of the seed `1` that the rule of `Op.fma`
gives at the word triples `ws`, read and written by `ofBits` and `toBits`. -/
def fmaWords {α : Type} [Storage α] [Context α] [Op.Capabilities α] (ofBits : Nat → α)
    (toBits : α → Nat) (ws : Array (Nat × Nat × Nat)) :
    Option (Array Nat × Array Nat × Array Nat × Array Nat) := do
  let tensor (f : Nat × Nat × Nat → Nat) : Tensor α [ws.size] :=
    Tensor.Rep.ofArray (ws.map (ofBits ∘ f)) (by simp [Shape.size])
  let xs := (tensor (·.1), tensor (·.2.1), tensor (·.2.2), ())
  let rule ← (Op.fma (s := [ws.size])).rule? (α := α)
  let y := rule.forward xs
  let g := rule.backward xs (Tensor.full _ 1, ())
  let words (t : Tensor α [ws.size]) := (t.to (Array α)).map toBits
  pure (words y.1, words g.1, words g.2.1, words g.2.2.1)

/-- Binary32 triples: `(1 + 2⁻²³, 1 + 2⁻²³, -(1 + 2⁻²²))`, `(0x7f7fffff, 2, -0x7f7fffff)`,
`(2⁻⁷⁵, 2⁻⁷⁵, 2⁻¹⁴⁹)`, `(1, -1, 1)`, `(-0, 1, -0)` and `(-0, 1, +0)`. -/
def triples32 : Array (Nat × Nat × Nat) :=
  #[(0x3f800001, 0x3f800001, 0xbf800002), (0x7f7fffff, 0x40000000, 0xff7fffff),
    (0x1a000000, 0x1a000000, 1), (0x3f800000, 0xbf800000, 0x3f800000),
    (0x80000000, 0x3f800000, 0x80000000), (0x80000000, 0x3f800000, 0)]

/-- The words of the result and of the three pullbacks of the seed `1` at `triples32`. -/
def words32 : Array Nat × Array Nat × Array Nat × Array Nat :=
  (#[0x28800000, 0x7f7fffff, 2, 0, 0x80000000, 0],
    #[0x3f800001, 0x40000000, 0x1a000000, 0xbf800000, 0x3f800000, 0x3f800000],
    #[0x3f800001, 0x7f7fffff, 0x1a000000, 0x3f800000, 0x80000000, 0x80000000],
    #[0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000])

#guard fmaWords (fun w => (ExecFloat.Binary.ofNatBits w : Binary32)) ExecFloat.Binary.toNatBits
  triples32 == some words32
#guard fmaWords (fun w => Float32.ofBits w.toUInt32) (·.toBits.toNat) triples32 == some words32

-- Rounding the product first gives `0`, `inf` and `2⁻¹⁴⁹` at the first three triples.
#guard
  let b (w : Nat) : Binary32 := ExecFloat.Binary.ofNatBits w
  [(0x3f800001, 0x3f800001, 0xbf800002), (0x7f7fffff, 0x40000000, 0xff7fffff),
      (0x1a000000, 0x1a000000, 1)].map
      (fun (x, y, z) => ExecFloat.Binary.toNatBits (b x * b y + b z)) ==
    [0, 0x7f800000, 1]

-- Binary64 and native `Float` at `(1 + 2⁻⁵², 1 + 2⁻⁵², -(1 + 2⁻⁵¹))`: `2⁻¹⁰⁴`.
#guard (fmaWords (fun w => (ExecFloat.Binary.ofNatBits w : Binary64)) ExecFloat.Binary.toNatBits
    #[(0x3ff0000000000001, 0x3ff0000000000001, 0xbff0000000000002)]).map (·.1) ==
  some #[0x3970000000000000]
#guard (fmaWords (fun w => Float.ofBits w.toUInt64) (·.toBits.toNat)
    #[(0x3ff0000000000001, 0x3ff0000000000001, 0xbff0000000000002)]).map (·.1) ==
  some #[0x3970000000000000]

-- `inf × 0 + 1`, `0 × -inf + 1`, `inf × 1 - inf` and a NaN addend are NaNs.
#guard
  let b (w : Nat) : Binary32 := ExecFloat.Binary.ofNatBits w
  [(0x7f800000, 0, 0x3f800000), (0, 0xff800000, 0x3f800000),
      (0x7f800000, 0x3f800000, 0xff800000), (0x3f800000, 0x3f800000, 0x7fc00000)].all
    fun (x, y, z) => ExecFloat.Binary.isNaN (Fma.fma (b x) (b y) (b z))

#guard ((Op.fma (s := [2])).rule? (α := ExecFloat.Binary 2 1)).isSome
#guard (Op.fma (s := [2])).need == .fma
#guard Op.encode ⟨_, _, Op.fma (s := [2, 3])⟩ == ⟨Op.Wire.formatVersion, "fma", [.shape [2, 3]]⟩
#guard ((Op.decode ⟨Op.Wire.formatVersion, "fma", [.shape [2, 3]]⟩).map Op.encode) ==
  some ⟨Op.Wire.formatVersion, "fma", [.shape [2, 3]]⟩

/-- The module `PyTorch.emit` writes for `Op.fma` of input 0, `v0` of shape `[3]`, in each operand,
or its refusal. -/
def emitted : Except String String := do
  let b : IR.Graph.Builder := {}
  let (b, x) := b.input 0 (.value [3])
  let (b, r) := b.op (Op.fma (s := [3])) [x, x, x]
  PyTorch.emit ⟨← b.build #[r.headD x], {}, #[]⟩ (options := { includeTrainingSkeleton := false })

#guard match emitted with
  | .ok _ => false
  | .error message => (message.splitOn "fma has no PyTorch translation").length > 1

end TorchLean.Test.Fma
