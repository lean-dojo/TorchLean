/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Init

/-!
# Tensor Initialization API Tests

Regression checks for shape inference, deterministic schemes, and scalar-polymorphic initialization.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat (Binary)
open FloatLib.Floats.ExecFloat.Binary (ofModel toModel)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

namespace NN.Tests.API.Init

open TorchLean

def expect (message : String) (condition : Bool) : IO Unit := do
  unless condition do
    throw <| IO.userError s!"tensor initialization check failed: {message}"

def run : IO Unit := do
  let zeros : Tensor Float [2, 3] := TorchLean.Init.tensor .zeros
  expect "generic tensor initialization infers shape"
    (zeros.to (Array Float) == #[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

  let xavier : Tensor Float [2, 3] := TorchLean.Init.xavierUniform (seed := 17)
  let xavierFromScheme : Tensor Float [2, 3] :=
    TorchLean.Init.tensor (.xavierUniform 3 2) (seed := 17)
  expect "Xavier dimensions are inferred from the result type"
    (xavier.to (Array Float) == xavierFromScheme.to (Array Float))

  let kaiming : Tensor Float [2, 3] := TorchLean.Init.kaimingUniform (seed := 23)
  let kaimingFromScheme : Tensor Float [2, 3] :=
    TorchLean.Init.tensor (.kaimingUniform 3) (seed := 23)
  expect "Kaiming fan-in is inferred from the result type"
    (kaiming.to (Array Float) == kaimingFromScheme.to (Array Float))

  let zeroFanXavier : Tensor Float [2] :=
    TorchLean.Init.tensor (.xavierUniform 0 0) (seed := 29)
  expect "zero-fan Xavier remains finite and degenerates to zero"
    ((zeroFanXavier.to (Array Float)).all fun value => value.isFinite && value == 0.0)

  let zeroFanKaiming : Tensor Float [2] :=
    TorchLean.Init.tensor (.kaimingUniform 0) (seed := 31)
  expect "zero-fan Kaiming remains finite and degenerates to zero"
    ((zeroFanKaiming.to (Array Float)).all fun value => value.isFinite && value == 0.0)

  let emptyXavier : Tensor Float [2, 0] := TorchLean.Init.xavierUniform (seed := 37)
  expect "empty Xavier matrices initialize without values"
    (emptyXavier.to (Array Float)).isEmpty

  let validUniform : TorchLean.Init.Scheme := .uniform (-1.0) 1.0
  let constantUniform : TorchLean.Init.Scheme := .uniform 2.0 2.0
  let validNormal : TorchLean.Init.Scheme := .normal 0.0 0.0
  let reversedUniform : TorchLean.Init.Scheme := .uniform 1.0 (-1.0)
  let negativeStd : TorchLean.Init.Scheme := .normal 0.0 (-1.0)
  let nanBound : TorchLean.Init.Scheme :=
    .uniform 0.0 (Float.ofBits 0x7ff8000000000000)
  let infiniteMean : TorchLean.Init.Scheme :=
    .normal (Float.ofBits 0x7ff0000000000000) 1.0
  expect "finite initializer schemes validate"
    (validUniform.validate.isOk && constantUniform.validate.isOk && validNormal.validate.isOk)
  expect "reversed uniform bounds are rejected" (!reversedUniform.validate.isOk)
  expect "negative normal standard deviation is rejected" (!negativeStd.validate.isOk)
  expect "NaN uniform bounds are rejected" (!nanBound.validate.isOk)
  expect "infinite normal parameters are rejected" (!infiniteMean.validate.isOk)

  let nativeBinary32 : Tensor Float32 [2, 2] := TorchLean.Init.tensor .ones
  expect "native binary32 uses the same initializer API"
    ((nativeBinary32.to (Array Float32)).map Float32.toFloat ==
      #[1.0, 1.0, 1.0, 1.0])

  let referenceBinary32 : Tensor (Binary 8 23) [2, 2] :=
    TorchLean.Init.tensor .ones
  expect "reference IEEE binary32 uses the same initializer API"
    ((referenceBinary32.to (Array (Binary 8 23))).map
      (fun x => Binary.toFloat (ofModel (Model.cast .binary32 .binary64 (toModel x)))) ==
        #[1.0, 1.0, 1.0, 1.0])

  let complex : Tensor (TorchLean.Complex Float) [2] := TorchLean.Init.tensor .ones
  expect "complex initialization embeds real samples with zero imaginary part"
    ((complex.to (Array (TorchLean.Complex Float))).all fun value =>
      value.re == 1.0 && value.im == 0.0)

  IO.println "  public tensor initialization: passed"

end NN.Tests.API.Init
