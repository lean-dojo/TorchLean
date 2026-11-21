/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.IR.Builder
public import TorchLean.NN.Op.Frexp
public import TorchLean.NN.Ops.Pure
public import TorchLean.NN.PyTorch.Export.IR
public import TorchLean.Tensor.ConfiguredBinary
public import TorchLean.Tensor.Conversion
public import TorchLean.Tensor.Op.Need
public meta import TorchLean.NN.IR.Builder
public meta import TorchLean.NN.Ops.Pure
public meta import TorchLean.NN.PyTorch.Export.IR
public meta import TorchLean.Tensor.ConfiguredBinary
public meta import TorchLean.Tensor.Conversion
public meta import TorchLean.Tensor.Op.Need

/-!
# `frexp` against PyTorch and C

In binary32 and binary64 the rule of `Op.frexp` gives the words of `torch.frexp` and of C's
`frexp` (CPython's `math.frexp`): the mantissa and the exponent of normal and subnormal inputs and
of the largest finite value, `(x, 0)` at both zeros and an infinity, and the quiet NaN with the
exponent `0` at a NaN, a signaling one included. Its pullback of the seed `1` is `ldexp 1 (-e)`,
rounded once. Torch autograd's `grad / exp2(e)` (PyTorch 2.13) gives the same words except where
`exp2 e`, which it computes in binary32, overflows or underflows. At the largest finite value
PyTorch gives `0`, while the rule gives the subnormal `2⁻¹²⁸` in binary32 and `2⁻¹⁰²⁴` in binary64.
At an exponent of `-150` or below PyTorch gives `inf`: binary64 `2⁻²⁰⁰` has the exponent `-199`,
where the rule gives `2¹⁹⁹`.

A configured format has `Frexp` only in an IEEE format of bias at least two, where every normalized
mantissa is a value, so the operator has a rule in binary32, in binary64 and in the format with
exponent width three and fraction width two. The minimal default-bias format, of bias one, has none:
its `frexp` would round the mantissa `3/4` of `3` to `1`. At the same widths with bias two the
mantissas are exact, but the format is not IEEE, and neither is E4M3's encoding without infinities,
so they have no rule either. Nor does the IEEE format of exponent width 32, whose exponent `2³¹` at
the largest finite value would wrap in the int32 exponent output, while exponent width 31 has one.
`Float` and `Float32` have no `Frexp`. Its wire record is tagged `frexp` with the shape, and the
PyTorch exporter writes `torch.frexp`.
-/

@[expose] public section

namespace TorchLean.Test.Frexp

open FloatLib.Floats

/-- The mantissa words, the exponents and the words of the pullback of the seed `1` that the rule
of `Op.frexp` gives at the words `ws`, read and written by `ofBits` and `toBits`. -/
def frexpWords {α : Type} [Storage α] [Context α] [Op.Capabilities α] (ofBits : Nat → α)
    (toBits : α → Nat) (ws : Array Nat) : Option (Array Nat × Array Int × Array Nat) := do
  let x := Tensor.from (ws.map ofBits)
  let rule ← (Op.frexp (s := [(ws.map ofBits).size])).rule? (α := α)
  let y := rule.forward (x, ())
  let g := rule.backward (x, ()) (Tensor.full _ 1, (), ())
  pure ((y.1.to (Array α)).map toBits, (y.2.1.to (Array (BitVec 32))).map BitVec.toInt,
    (g.1.to (Array α)).map toBits)

-- Binary32 at `8`, `-3`, `1`, `±0`, `inf`, a quiet and a signaling NaN, the least subnormal, the
-- subnormal `1e-40`, the largest finite value and `0.5`: the words of `torch.frexp` and
-- `torch.ldexp(1, -e)`.
#guard frexpWords (fun w => (ExecFloat.Binary.ofNatBits w : Binary32)) ExecFloat.Binary.toNatBits
    #[0x41000000, 0xc0400000, 0x3f800000, 0, 0x80000000, 0x7f800000, 0x7fc00000, 0x7fa00000, 1,
      0x116c2, 0x7f7fffff, 0x3f000000] ==
  some (#[0x3f000000, 0xbf400000, 0x3f000000, 0, 0x80000000, 0x7f800000, 0x7fc00000, 0x7fe00000,
      0x3f000000, 0x3f0b6100, 0x3f7fffff, 0x3f000000],
    #[4, 2, 1, 0, 0, 0, 0, 0, -148, -132, 128, 0],
    #[0x3d800000, 0x3e800000, 0x3f000000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000,
      0x3f800000, 0x7f800000, 0x7f800000, 0x200000, 0x3f800000])

-- Binary64 at the least subnormal, the largest finite value, `8`, `-3`, `2⁻²⁰⁰` and a signaling
-- NaN: the words of `torch.frexp`, `math.frexp` and `math.ldexp(1, -e)`.
#guard frexpWords (fun w => (ExecFloat.Binary.ofNatBits w : Binary64)) ExecFloat.Binary.toNatBits
    #[1, 0x7fefffffffffffff, 0x4020000000000000, 0xc008000000000000, 0x3370000000000000,
      0x7ff4000000000000] ==
  some (#[0x3fe0000000000000, 0x3fefffffffffffff, 0x3fe0000000000000, 0xbfe8000000000000,
      0x3fe0000000000000, 0x7ffc000000000000],
    #[-1073, 1024, 4, 2, -199, 0],
    #[0x7ff0000000000000, 0x4000000000000, 0x3fb0000000000000, 0x3fd0000000000000,
      0x4c60000000000000, 0x3ff0000000000000])

-- In the widths two and one, the word `0x5` is `3` at the default bias one and `3/2` at bias two.
-- At bias one `frexp` would round the mantissa `3/4` of `3` to `1` (word `0x2`), while at bias two
-- the mantissa `3/4` of `3/2` is the word `0x3`. Neither format has a rule: the first rounds, and
-- the second is not IEEE. The IEEE format of widths three and two, of bias three, has one.
#guard
  let biasOne (w : Nat) : ExecFloat.Binary 2 1 := ExecFloat.Binary.ofNatBits w
  let biasTwo (w : Nat) : ExecFloat.Binary 2 1 (bias := 2) := ExecFloat.Binary.ofNatBits w
  (ConfiguredBinary.frexp (biasOne 5)).map ExecFloat.Binary.toNatBits id == (0x2, 2) &&
    (ConfiguredBinary.frexp (biasTwo 5)).map ExecFloat.Binary.toNatBits id == (0x3, 1)
#guard ((Op.frexp (s := [2])).rule? (α := ExecFloat.Binary 2 1)).isNone
#guard ((Op.frexp (s := [2])).rule? (α := ExecFloat.Binary 2 1 (bias := 2))).isNone
#guard ((Op.frexp (s := [2])).rule? (α := ExecFloat.Binary 3 2)).isSome
#guard ((Op.frexp (s := [2])).rule? (α := ExecFloat.Binary 4 3 .finiteMaxNaN)).isNone

-- In the IEEE format of exponent width 32 and fraction width one, the largest finite value
-- `2 ^ (2³¹ - 1) * 3/2` has the exponent `2³¹`, whose int32 word `-2³¹` would reconstruct a value
-- near `2 ^ (-2³¹)`, so the format has no rule. At exponent width 31 every exponent of a finite
-- value lies in `[2 - 2³⁰, 2³⁰]` and the format has one.
#guard ((Op.frexp (s := [2])).rule? (α := ExecFloat.Binary 32 1)).isNone
#guard ((Op.frexp (s := [2])).rule? (α := ExecFloat.Binary 31 1)).isSome

/-- The mantissa the pure interpreter gives for `Op.frexp` in binary32. No instance of `Frexp` is
found at a configured format, so the call passes the scalar's capability record, with `decide`'s
proof that it holds `Frexp`. It also names the scalar's `Context`: `Ops.apply`, a projection of the
class `Ops`, takes it as an implicit argument, and at a concrete scalar (`Float` too) elaboration
does not fill it: the type of the arguments waits on `Ops.Ref` of an instance not yet found. -/
def pureMantissa (x : Tensor Binary32 [1]) : Tensor Binary32 [1] :=
  @Ops.apply Id Binary32 _ configuredBinaryContext _ _ _ _ (Op.frexp (s := [1]))
    ⟨inferInstance, by decide⟩ (x, ())

#guard
  let eight : Binary32 := ExecFloat.Binary.ofNatBits 0x41000000
  ((pureMantissa (Tensor.from #[eight])).to (Array Binary32)).map ExecFloat.Binary.toNatBits ==
    #[0x3f000000]

-- In the bias-one format the record does not hold `Frexp`.
example : ¬ Op.Need.frexp.Available (inferInstance : Op.Capabilities (ExecFloat.Binary 2 1)) := by
  decide

-- The exponents of finite binary64 values, `[-1073, 1024]`, fit in int32, so `decide` discharges
-- the format's premises of the reconstruction from the word.
example {x : Binary64} (hx : ExecFloat.Binary.isFinite x = true) :
    ExecFloat.Binary.toReal (ConfiguredBinary.frexp x).1 *
        2 ^ (BitVec.ofInt 32 (ConfiguredBinary.frexp x).2).toInt = ExecFloat.Binary.toReal x :=
  ConfiguredBinary.toReal_frexp_mul_toInt rfl (by decide) (by decide) (by decide) hx

-- The same premises give the reconstruction through the rule of `Op.frexp` at binary64.
example (x : Tensor Binary64 [2]) :
    ∃ rule ∈ (Op.frexp (s := [2])).rule? (α := Binary64), ∀ c,
      ExecFloat.Binary.isFinite (x c) = true →
        ExecFloat.Binary.toReal ((rule.forward (x, ())).1 c) *
            2 ^ ((rule.forward (x, ())).2.1 c).toInt = ExecFloat.Binary.toReal (x c) :=
  Op.frexp_forward_reconstruct_configured rfl (by decide) (by decide) (by decide) x

-- Every rule at a configured format reconstructs, with no premise on the format.
example {rule} (h : rule ∈ (Op.frexp (s := [2])).rule? (α := Binary64))
    (x : Tensor Binary64 [2]) (c) (hx : ExecFloat.Binary.isFinite (x c) = true) :
    ExecFloat.Binary.toReal ((rule.forward (x, ())).1 c) *
        2 ^ ((rule.forward (x, ())).2.1 c).toInt = ExecFloat.Binary.toReal (x c) :=
  Op.frexp_forward_reconstruct_of_mem h x c hx

#guard ((Op.frexp (s := [2])).rule? (α := Float)).isNone
#guard ((Op.frexp (s := [2])).rule? (α := Float32)).isNone
#guard (Op.frexp (s := [2])).need == .frexp
#guard Op.encode ⟨_, _, Op.frexp (s := [2, 3])⟩ ==
  ⟨Op.Wire.formatVersion, "frexp", [.shape [2, 3]]⟩
#guard ((Op.decode ⟨Op.Wire.formatVersion, "frexp", [.shape [2, 3]]⟩).map Op.encode) ==
  some ⟨Op.Wire.formatVersion, "frexp", [.shape [2, 3]]⟩

/-- The module `PyTorch.emit` writes for the mantissa of `Op.frexp` at input 0, `v0` of shape
`[3]`, or its refusal. -/
def emitted : Except String String := do
  let b : IR.Graph.Builder := {}
  let (b, x) := b.input 0 (.value [3])
  let (b, y) := b.op (Op.frexp (s := [3])) [x]
  PyTorch.emit ⟨← b.build #[y.headD x], {}, #[]⟩ (options := { includeTrainingSkeleton := false })

#guard match emitted with
  | .ok module => (module.splitOn "v1, v1_1 = torch.frexp(v0)").length > 1
  | .error _ => false

end TorchLean.Test.Frexp
