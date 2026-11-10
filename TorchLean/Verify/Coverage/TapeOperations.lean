/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.NN.Tape.Apply
public import TorchLean.NN.Tape.Backward
public import TorchLean.Verify.RuntimeApprox.Roundoff.Coverage
public import TorchLean.NN.Tape.CPU

/-!
# Roundoff coverage of the CPU tape operations

`#roundoff_coverage` reads the code of the tape builders (with `Tape.apply`, which records any
registered operator, and `Tape.record`, which appends its prepared node): the forward value of the
node it records and each contribution its backward closure sends to a parent, with the saved values
it uses. It runs `roundoff` on each of these routes and on the sum with which the reverse sweep
accumulates contributions, for the configured scalar of an arbitrary IEEE format, for native
`Float32` and for the rounded reals `FP32`. A route that applies a registered operator
specification to the saved values shares that specification's outcome.

The accumulation casts a contribution to the shape of the parent's gradient once the two shapes
agree. `roundoff` reads through that cast with `TorchLean.SomeTensor.cast_apply`.
-/

public section

namespace TorchLean.Coverage.TapeOperations

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange

#roundoff_coverage
  builder := Tape.leaf, Tape.apply, Tape.record
  accumulation := SomeTensor.add

end TorchLean.Coverage.TapeOperations
