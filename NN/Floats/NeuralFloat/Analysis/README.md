# Floating-Point Analysis

This folder contains reusable facts about grid spacing and exact operations.

- `Ulp.lean` defines format-parametric units in the last place.
- `Neighbors.lean` characterizes predecessor and successor values.
- `Sterbenz.lean` proves exact subtraction for unbounded-exponent formats.
- `SterbenzFLT.lean` extends exact subtraction through the gradual-underflow boundary.

These results feed compensated summation, dot-product error bounds, and proofs about reduction
order.  See Sterbenz, *Floating-Point Computation* (1974), and Higham, *Accuracy and Stability of
Numerical Algorithms*, second edition (2002).
