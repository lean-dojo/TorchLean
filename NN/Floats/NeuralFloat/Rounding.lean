/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Rounding.Away
public import NN.Floats.NeuralFloat.Rounding.Core
public import NN.Floats.NeuralFloat.Rounding.Double
public import NN.Floats.NeuralFloat.Rounding.Generic
public import NN.Floats.NeuralFloat.Rounding.Nearest
public import NN.Floats.NeuralFloat.Rounding.Odd
public import NN.Floats.NeuralFloat.Rounding.Order
public import NN.Floats.NeuralFloat.Rounding.Predicates
public import NN.Floats.NeuralFloat.Rounding.Properties

/-!
# Rounding Semantics

This umbrella contains rounding functions and their semantic laws: directed modes, nearest choices,
order properties, round-to-odd, round-away, and double rounding.  It is independent of concrete
IEEE bit encodings; a format supplies the representable grid and a rounding policy selects a point
on that grid.

## References

- IEEE, *IEEE Standard for Floating-Point Arithmetic*, IEEE 754-2019, Sections 4 and 7.
- D. Goldberg, "What Every Computer Scientist Should Know About Floating-Point Arithmetic,"
  *ACM Computing Surveys* 23(1), 1991, doi:10.1145/103162.103163.
-/

@[expose] public section
