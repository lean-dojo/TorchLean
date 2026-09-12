import VersoManual
import NN.MLTheory.Proofs.Hopfield
import NN.MLTheory.Proofs.ReLU.Approx.ReLUMulApprox
import NN.MLTheory.Proofs.ReLU.Bridge.ReLUMlpBridge
import NN.MLTheory.Proofs.StateSpace.MambaCausality
import NN.Spec.Models.Hopfield
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean

-- The three developments in this chapter live in four unrelated namespaces: the executable Hopfield
-- spec, its proofs, the ReLU bridge, and the state-space causality proofs. Opening them here lets
-- each `#check` and each small definition fit inside Verso's narrow code column.
open Spec.Hopfield
open NN.MLTheory.Proofs.Hopfield
open NN.MLTheory.Proofs.ReLUMlpBridge
open NN.MLTheory.Proofs.ReLUMulApprox
open NN.MLTheory.StateSpace
open TorchLean (Tensor Storage)

-- Most signatures here print wider than this file's 100-column limit, so their `leanOutput` blocks
-- ask for `whitespace := lax` and are wrapped in the source. The rendered page still shows each
-- message exactly as Lean printed it.
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Structural Model Proofs" =>
%%%
tag := "classical-ml-proofs"
%%%

Some neural-network properties are independent of any particular GPU kernel or training loop.
Hopfield networks have an energy argument. ReLU networks have exact algebraic identities that make
larger approximation constructions possible. Recurrent state-space models are causal because their
output at time `t` is computed before future inputs are seen. These are structural facts about the
mathematical models.

TorchLean formalizes such results beside its runtime developments so that later work can connect
them. These theorems concern the spec-level definitions; they do not by themselves certify a CUDA
implementation.

# Hopfield Dynamics

A TorchLean Hopfield state is a Boolean vector, and its parameters are a weight matrix together
with a threshold vector. The energy argument follows {Informal.citet hopfield1982}[] for
asynchronous updates. The following commands check the state and parameter types:

```lean (name := hopfieldTypes)
-- Inspect the dimension-indexed state and parameter types
-- used by both rational and real models.
#check @State
#check @Params
```

```leanOutput hopfieldTypes
State : ℕ → Type
```

`State n` unfolds to `Fin n → Bool`. The numeric activation map reads `true` as `+1` and `false` as
`-1`, so the Boolean carrier is a storage choice rather than a claim that the model is Boolean
valued. `Params α n` holds the fields `W : Fin n → Fin n → α` and `θ : Fin n → α`, and it is
polymorphic in the scalar: the same structure serves the executable `Rat` runs below and the
real-valued theorems that follow.

`Fin n` is the type of valid neuron indices, so a state is a function that returns one bit for each
neuron without admitting an out-of-range coordinate. `List.ofFn` in the examples merely displays
those bits in index order. It does not change the state representation or supply the numeric
activation map. This distinction matters when reading the energy table: a printed `false`
contributes minus one to the sums, not zero. In the type of `energy`, `[Field α]` requests the
arithmetic operations and laws used to form the expression. An energy-decrease result additionally
needs an order and assumptions connecting the weights to the update rule; those appear in the
real theorem rather than in the parameter record.

For state `s`, write $`x_i\in\{-1,+1\}` for its numeric activation. The net input to neuron `u` is

$$`\operatorname{net}_u(s)=\sum_j W_{uj}x_j.`

`updateAt p s u` changes only coordinate `u`, using

$$`x_u'=
\begin{cases}
+1,&\theta_u\leq\operatorname{net}_u(s),\\
-1,&\operatorname{net}_u(s)<\theta_u.
\end{cases}`

The non-strict comparison fixes a detail often omitted on paper: ties go to `+1`. That convention
becomes important in the convergence proof.

The energy is

$$`E(s)
=-\frac12\sum_i\sum_j W_{ij}x_i x_j
 +\sum_i\theta_i x_i.`

The factor of one half is why the scalar has to be more than a ring:

```lean (name := energySig)
-- The energy expression uses division by two, supplied by
-- the scalar field instance.
#check @energy
```

```leanOutput energySig (whitespace := lax)
@energy : {α : Type} → [Field α] → {n : ℕ} → Params α n → State n → α
```

`Rat` supplies this field structure and supports exact executable arithmetic. It therefore lets
the experiments below evaluate the energy without rounding.

Under symmetric weights and zero diagonal,

$$`W_{ij}=W_{ji},
\qquad W_{ii}=0,`

the theorem
{src "NN/MLTheory/Proofs/Hopfield/Energy.lean"}[`energy_updateAt_le`]
proves

$$`E(\operatorname{updateAt}(p,s,u))\leq E(s).`

```lean (name := energyLe)
-- The order conclusion applies to real weights satisfying
-- symmetry and a zero diagonal.
#check @energy_updateAt_le
```

```leanOutput energyLe (whitespace := lax)
@energy_updateAt_le : ∀ {n : ℕ} (p : Params ℝ n),
  SymmetricW p → DiagonalZero p →
    ∀ (s : State n) (u : Fin n), energy p (updateAt p s u) ≤ energy p s
```

`Params` allows arbitrary weights; `SymmetricW` and `DiagonalZero` restrict the networks covered
by the theorem. This separation also lets us execute asymmetric networks and examine where the
energy argument fails.

The proof expands the quadratic energy difference. Symmetry makes the changed row and column
contribute the same net-input term, while the zero diagonal removes the self-interaction. When the
coordinate changes and the net input is not tied with the threshold,
`energy_updateAt_lt_of_change_of_ne` strengthens the inequality to a strict decrease.

Writing the old and new activations at the changed coordinate as $`x_u` and $`x'_u`, the
cancellation gives

$$`E(s')-E(s)=(x'_u-x_u)(\theta_u-\operatorname{net}_u(s)).`

If a negative activation becomes positive, the first factor is two and the threshold comparison
makes the second nonpositive. If a positive activation becomes negative, the first factor is
minus two and the second is positive. Both cases give a nonpositive product. A tied negative
activation can become positive with the second factor zero, explaining why an actual state change
need not decrease energy strictly. This calculation tells us exactly where the tie example and
the secondary progress measure enter the argument.

# Two-Neuron Hopfield Updates

Start with two mutually excitatory neurons, zero thresholds, and initial state `[+1,-1]`.
Each neuron receives the other neuron's activation with weight one:

```lean (name := hopfieldRun)
-- Use mutual excitation so neuron one copies the positive
-- activation of neuron zero.
def excitatory : Params Rat 2 where
  W := fun i j => if i = j then 0 else 1
  θ := fun _ => 0

def before : State 2 := fun i => i = 0
def after : State 2 := updateAt excitatory before 1
```

The states before and after updating neuron `1`:

```lean (name := hopfieldStates)
-- Display the Boolean states in neuron order; true and
-- false encode positive and negative
-- activation.
#eval List.ofFn before
#eval List.ofFn after
```

```leanOutput hopfieldStates
[true, false]
```

```leanOutput hopfieldStates
[true, true]
```

And their energies:

```lean (name := hopfieldEnergy)
-- Compute the two rational energies exactly, before and
-- after the single-coordinate update.
#eval energy excitatory before
#eval energy excitatory after
```

```leanOutput hopfieldEnergy
1
```

```leanOutput hopfieldEnergy
-1
```

Neuron `1` receives net input `+1`, so its activation changes from `-1` to `+1`. The two
off-diagonal products in the energy change from negative to positive, lowering the exact rational
energy from `1` to `-1`.
Changing `W 0 1` without changing `W 1 0` still produces an executable state sequence, but it
prevents use of `energy_updateAt_le`: Lean asks for `SymmetricW p`. Setting a diagonal weight to a
nonzero value similarly leaves the program runnable while invalidating the theorem's
`DiagonalZero p` premise.

# Energy Ties And Convergence

If every state change strictly lowered energy, finiteness would immediately rule out cycles. Ties
make the argument subtler. With the convention “ties go to `+1`,” a state may change while energy
stays equal. TorchLean therefore uses the number of positive neurons,

$$`\operatorname{pluses}(s)
=|\{i\mid s_i=\texttt{true}\}|,`

as a secondary progress measure. For one full cyclic sweep, `cycleUpdate_progress` proves:

- either energy strictly decreases;
- or energy is unchanged and `pluses` strictly increases.

```lean (name := progress)
-- A changed sweep must decrease energy or increase the
-- positive count at unchanged energy.
#check @cycleUpdate_progress
```

```leanOutput progress (whitespace := lax)
@cycleUpdate_progress : ∀ {n : ℕ} (p : Params ℝ n),
  SymmetricW p →
    DiagonalZero p →
      ∀ (s : State n),
        cycleUpdate p s ≠ s →
          energy p (cycleUpdate p s) < energy p s ∨
            energy p (cycleUpdate p s) = energy p s ∧ pluses (cycleUpdate p s) > pluses s
```

The second case occurs when all
weights and thresholds are zero, so every net input ties with its threshold:

```lean (name := tieSetup)
-- Zero weights and thresholds force a tie at every
-- coordinate.
def zeroNet : Params Rat 2 where
  W := fun _ _ => 0
  θ := fun _ => 0

def tiedBefore : State 2 := fun _ => false
def tiedAfter : State 2 :=
  updateAt zeroNet tiedBefore 0
```

The state changes, because a tie resolves to `+1`:

```lean (name := tieState)
-- Updating a tied negative coordinate demonstrates the
-- convention that ties become positive.
#eval List.ofFn tiedBefore
#eval List.ofFn tiedAfter
```

```leanOutput tieState
[false, false]
```

```leanOutput tieState
[true, false]
```

Every term in the energy has a zero coefficient, so both energies are zero:

```lean (name := tieEnergy)
-- Every energy term vanishes here, even though the state
-- changes.
#eval energy zeroNet tiedBefore
#eval energy zeroNet tiedAfter
```

```leanOutput tieEnergy
0
```

```leanOutput tieEnergy
0
```

The positive count increases even though the energy stays fixed:

```lean (name := tiePluses)
-- The positive count detects the progress that energy alone
-- misses in the tied update.
#eval pluses tiedBefore
#eval pluses tiedAfter
```

```leanOutput tiePluses
0
```

```leanOutput tiePluses
1
```

This single-coordinate example exhibits equal energy with an increased positive count. A strict
comparison would instead send ties to `-1`: this initial state would stay fixed, but tied positive
states could still change at equal energy. Such a convention needs the opposite count as its
secondary measure. Keeping the previous activation on ties would avoid tie-induced changes.

The lexicographic pair

$$`\bigl(E(s),-\operatorname{pluses}(s)\bigr)`

therefore progresses whenever a sweep changes the state. Since `State n` is finite,
{src "NN/MLTheory/Proofs/Hopfield/Convergence.lean"}[`cycleUpdate_no_nontrivial_cycles`]
rules out a nontrivial cycle, and `cycleUpdate_exists_fixedpoint_le_card` gives a fixed point within
at most `Fintype.card (State n)` sweeps. The more explicit
`cycleUpdate_exists_fixedpoint_le_pow` states the corresponding $`2^n` bound.

The following declarations expose the strict-decrease lemma, the exclusion of cycles, and the two
fixed-point bounds:

```lean (name := convergence)
-- Inspect the strict-update lemma and the finite bound on
-- repeated cyclic sweeps.
#check @energy_updateAt_lt_of_change_of_ne
#check @cycleUpdate_no_nontrivial_cycles
#check @cycleUpdate_exists_fixedpoint_le_card
#check @cycleUpdate_exists_fixedpoint_le_pow
```

```leanOutput convergence (whitespace := lax)
@cycleUpdate_exists_fixedpoint_le_pow : ∀ {n : ℕ} (p : Params ℝ n),
  SymmetricW p → DiagonalZero p → ∀ (s0 : State n), ∃ m ≤ 2 ^ n, (f p)^[m + 1] s0 = (f p)^[m] s0
```

`(f p)^[m]` is Mathlib's iterated-function notation, so the conclusion reads: within $`2^n` sweeps,
one more sweep changes nothing. The `_le_card` variant says the same thing with
`Fintype.card (State n)` in place of $`2^n`; the explicit power is the one to quote, and the
cardinality form is the one the proof actually produces.

A decreasing real sequence need not reach its limit after finitely many steps. What makes this
argument terminate is that the energy and count are evaluated on a finite set of states. Before
a fixed point is reached, a repeated state would close a cycle, contradicting the progress result.
There are only `2 ^ n` possible Boolean functions, so enough distinct sweep states exhaust the
possibilities. The existential `∃ m ≤ 2 ^ n` supplies a stopping index, and the equality compares
successive iterates at that index. It does not say that the fixed point is unique, nor that every
starting state reaches the same one.

These are theorems about asynchronous coordinate updates arranged into cyclic sweeps. They do not
apply automatically to synchronous updates, stochastic schedules, modern continuous-state
Hopfield layers such as the one of {Informal.citet modernhopfield2021}[], or a floating-point
kernel. Each variation needs its own transition relation and
energy argument.

# Hebbian Storage And Recall

To use the dynamics as an associative memory, store a pattern $`\xi\in\{-1,+1\}^n` by the Hebbian
outer product

$$`W_{ij}=\xi_i\xi_j\ (i\neq j),
\qquad W_{ii}=0,
\qquad \theta=0,`

which is symmetric and has zero diagonal. These are the weight conditions needed by the real
convergence theorem. The following rational experiment uses the same construction with four
neurons, storing $`(+1,+1,-1,-1)`:

```lean (name := hebbSetup)
-- Build the zero-diagonal outer product and a rational
-- sweep in increasing neuron order.
def xi : Fin 4 → Rat :=
  fun i => if i.val < 2 then 1 else -1

def hebbNet : Params Rat 4 where
  W := fun i j => if i = j then 0 else xi i * xi j
  θ := fun _ => 0

-- The computable twin of `cycleUpdate`, which is
-- `noncomputable` because it is stated over `ℝ`.
def sweepRat (s : State 4) : State 4 :=
  (List.finRange 4).foldl
    (fun s u => updateAt hebbNet s u) s

def pattern : State 4 := fun i => i.val < 2
def noisy : State 4 := fun i => i.val = 0
```

`noisy` is the stored pattern with one bit wrong. One sweep repairs it:

```lean (name := hebbRun)
-- Check whether one sweep corrects this particular one-bit
-- corruption.
#eval List.ofFn noisy
#eval List.ofFn (sweepRat noisy)
#eval List.ofFn (sweepRat noisy) == List.ofFn pattern
```

```leanOutput hebbRun
[true, false, false, false]
```

```leanOutput hebbRun
[true, true, false, false]
```

```leanOutput hebbRun
true
```

The energy decreases and the positive count increases in this run. Strict energy decrease is
already sufficient for progress; the theorem only needs the count when the energy stays equal:

```lean (name := hebbEnergy)
-- Compare energy progress with the auxiliary count for the
-- stored pattern and noisy start.
#eval energy hebbNet noisy
#eval energy hebbNet pattern
#eval pluses noisy
#eval pluses (sweepRat noisy)
```

```leanOutput hebbEnergy
0
```

```leanOutput hebbEnergy
-6
```

```leanOutput hebbEnergy
1
```

```leanOutput hebbEnergy
2
```

For a single stored pattern, the energies can be calculated from its overlap with the current
state. The double sum collapses because
$`\sum_{i,j}\xi_i\xi_jx_ix_j=(\xi\cdot x)^2` and the zero diagonal removes the $`n` self terms:

$$`E(s)=\frac{n-(\xi\cdot x)^2}{2}.`

With $`n=4`, the pattern itself has $`\xi\cdot x=4` and energy $`(4-16)/2=-6`, while `noisy` has
$`\xi\cdot x=2` and energy $`0`. The formula also says what the minima are: energy is smallest
exactly when $`|\xi\cdot x|` is largest, which happens at $`x=\xi` and at $`x=-\xi`. For this
zero-threshold, single-pattern network, both patterns are stable, as checked here:

```lean (name := hebbComplement)
-- Test both the stored pattern and its complement for
-- stability under every coordinate update.
def complement : State 4 := fun i => ¬ (i.val < 2)

def stable (s : State 4) : Bool :=
  decide (∀ u : Fin 4, updateAt hebbNet s u = s)

#eval energy hebbNet complement
#eval stable pattern
#eval stable complement
```

```leanOutput hebbComplement
-6
```

```leanOutput hebbComplement
true
```

```leanOutput hebbComplement
true
```

`stable` is `IsStable` from the spec, evaluated: no single-coordinate update changes the state. Both
$`\xi` and $`-\xi` pass it. Since `State 4` has sixteen inhabitants, we can enumerate every
initial state and check where one sweep takes it:

```lean (name := hebbBasins)
-- Enumerate all four-bit starts and count the two basins
-- after one sweep.
def allStates : List (State 4) :=
  (List.range 16).map fun k =>
    fun i : Fin 4 => (k / 2 ^ i.val) % 2 == 1

def basin (s : State 4) : List Bool :=
  List.ofFn (sweepRat s)

#eval allStates.all (fun s => stable (sweepRat s))
#eval (allStates.filter
  (fun s => basin s == List.ofFn pattern)).length
#eval (allStates.filter
  (fun s => basin s == List.ofFn complement)).length
```

```leanOutput hebbBasins
true
```

```leanOutput hebbBasins
8
```

```leanOutput hebbBasins
8
```

Every one of the sixteen states is at a fixed point after a single sweep, and eight states reach
each attractor.

`cycleUpdate_exists_fixedpoint_le_pow` promises a fixed point within $`2^n=16` sweeps; this network
needs one. The bound counts states because its proof uses finiteness and lexicographic progress.
It does not analyze the particular weights or update order that make this example converge
quickly. A sharper bound for a family of networks would need that additional analysis.

The three outputs of `hebbBasins` answer three different questions. The Boolean asks whether
every enumerated start is stable after one sweep. The two natural numbers count starts whose
result equals each named pattern. Here `allStates` decodes all four-bit integers, so its sixteen
entries cover this state space rather than a sample of it. The split of eight and eight is a
cardinality statement. Interpreting it as a fifty-percent recall probability would require
choosing a uniform distribution on those starts. A noise process concentrated near the stored
pattern could have a different recall probability even with exactly the same dynamics.

The theorem guarantees that the sweep sequence stops, but does not identify the final state.
Half of these starts recover $`\xi` and half
recover $`-\xi`, and for a memory with several stored patterns there are spurious mixture states
too. Convergence and correct recall are different claims, and only the first one is proved here.

## Hebbian Cross-Talk

Add a second pattern that
differs from `xi` in a single coordinate, summing the two outer products as the Hebbian rule
prescribes:

```lean (name := crossTalk)
-- The correlated second pattern cancels every off-diagonal
-- connection to the last neuron.
/-- A second pattern, agreeing with `xi` except at the
last coordinate. -/
def xi2 : Fin 4 → Rat :=
  fun i => if i.val = 3 then 1 else xi i

def twoPatterns : Params Rat 4 where
  W := fun i j =>
    if i = j then 0 else xi i * xi j + xi2 i * xi2 j
  θ := fun _ => 0

def sweepWith (p : Params Rat 4) (s : State 4) : State 4 :=
  (List.finRange 4).foldl
    (fun s u => updateAt p s u) s

def stableWith (p : Params Rat 4) (s : State 4) : Bool :=
  decide (∀ u : Fin 4, updateAt p s u = s)

def pattern2 : State 4 :=
  fun i => i.val < 2 || i.val = 3

#eval (List.finRange 4).map fun i =>
  (List.finRange 4).map fun j => twoPatterns.W i j
```
```leanOutput crossTalk
[[0, 2, -2, 0], [2, 0, -2, 0], [-2, -2, 0, 0], [0, 0, 0, 0]]
```

The last row is zero because neuron `3` is the coordinate where the two
patterns disagree, so its two contributions $`\xi_3\xi_j` and $`\xi'_3\xi'_j` are equal and
opposite for every other neuron `j`. Its net input is therefore always zero. With a zero threshold,
every update of neuron `3` is a tie and sets its activation to `+1`.

Both stored patterns are still global energy minima, and the energy cannot even see the coordinate
they disagree on:

```lean (name := crossEnergy)
-- Compare minimum energy with coordinate stability; the tie
-- rule can distinguish equal-energy
-- states.
#eval energy twoPatterns pattern
#eval energy twoPatterns pattern2
#eval stableWith twoPatterns pattern
#eval stableWith twoPatterns pattern2
#eval List.ofFn (sweepWith twoPatterns pattern)
```
```leanOutput crossEnergy
-6
```
```leanOutput crossEnergy
-6
```
```leanOutput crossEnergy
false
```
```leanOutput crossEnergy
true
```
```leanOutput crossEnergy
[true, true, false, true]
```

Both patterns sit at energy $`-6`, which is the minimum over all sixteen states, and
only one of them is a fixed point. Starting the network *at* the pattern we stored moves it away,
to the other one, in a single sweep.

`twoPatterns` is symmetric with zero diagonal, so its real-valued counterpart satisfies the
hypotheses of `cycleUpdate_progress` and `cycleUpdate_exists_fixedpoint_le_pow`. Reaching a
fixed point is consistent with failing to recall one of the stored patterns. Enumerating the
initial states shows how the tie rule affects recall:

```lean (name := crossBasins)
-- Count convergence destinations and stable states
-- separately for the two-pattern memory.
#eval (allStates.filter (fun s =>
  List.ofFn (sweepWith twoPatterns s) ==
    List.ofFn pattern2)).length
#eval (allStates.filter (fun s =>
  stableWith twoPatterns (sweepWith twoPatterns s))).length
#eval (allStates.filter
  (stableWith twoPatterns)).length
```
```leanOutput crossBasins
12
```
```leanOutput crossBasins
16
```
```leanOutput crossBasins
2
```

Twelve of the sixteen starts land on `pattern2`, and every start reaches a fixed point after one
sweep. There are two fixed points in total. Of the four energy-minimizing states, the two with
neuron `3` negative change on a tie; the two with that neuron positive remain fixed. Thus the tie
rule affects both which minima are stable and the secondary measure used in the convergence proof.

This small example demonstrates cross-talk through an exactly zero row. Asymptotic capacity
estimates for large networks with random patterns do not supply a threshold for these four neurons
and deliberately correlated patterns. Here the printed matrix and exhaustive state checks explain
the failure directly.

The row of zeros also explains why a global energy minimum can fail the stability check. Energy
is indifferent to neuron three, but the update rule is not: it always chooses the positive
activation on that row. Thus the Boolean `false` for `stableWith twoPatterns pattern` is
consistent with the energy value `-6`; the two commands test different predicates. For an
associative-memory application, a useful further specification would ask that each stored
pattern be stable, and then describe which corrupted inputs return to it. The convergence theorem
supplies neither condition merely from the way the weights were constructed.

## Hopfield Updates In PyTorch

The same single-pattern update can be written in PyTorch:

```
# Mutate coordinates in order so each neuron reads the
# preceding updates in the same sweep.
import torch

xi = torch.tensor([1., 1., -1., -1.])
W = torch.outer(xi, xi)
W.fill_diagonal_(0.)

x = torch.tensor([1., -1., -1., -1.])
for u in range(4):
    x[u] = 1.0 if W[u] @ x >= 0 else -1.0

assert torch.equal(x, xi)
```

The loop directly exposes the asynchronous dynamics. What
it checks is one pattern, from one start, at one precision. The `#eval` above checks all sixteen
starts for this rational network. `cycleUpdate_no_nontrivial_cycles` proves the absence of
nontrivial sweep cycles for every real symmetric zero-diagonal network, at every finite size.

Two details of that loop deserve attention. It writes `>=`, so ties go to `+1`; writing `>` instead
would also run, would also pass this assertion, and would quietly change the dynamics at exactly
the states where `pluses` was needed. And `x` is mutated inside the loop, so neuron `u` already sees
the updates from neurons before it. That is the asynchronous schedule the theorems assume. A
vectorized `x = torch.sign(W @ x)` reads the old state for every coordinate, giving a synchronous
update that can oscillate. It also returns zero at a zero net input, unlike the Boolean model's
tie rule. The examples below avoid zero net inputs when comparing the two schedules.

## Symmetry And Update Scheduling

The convergence theorems assume `SymmetricW`, `DiagonalZero`, and a cyclic asynchronous schedule.
The next examples show what fails when symmetry or the asynchronous schedule is removed.

Drop symmetry first. Two neurons, `W 0 1 = 1` and `W 1 0 = -3`: neuron `0` is excited by neuron `1`
while neuron `1` is inhibited by neuron `0`.

```lean (name := asymRun)
-- Opposite signs across the off-diagonal entries
-- deliberately violate the symmetry premise.
def asymNet : Params Rat 2 where
  W := fun i j =>
    if i = j then 0
    else if i.val < j.val then 1 else -3
  θ := fun _ => 0

def sweep2 (p : Params Rat 2) (s : State 2) : State 2 :=
  (List.finRange 2).foldl
    (fun s u => updateAt p s u) s

def bothUp : State 2 := fun _ => true

#eval (List.range 5).map fun m =>
  List.ofFn ((sweep2 asymNet)^[m] bothUp)
```
```leanOutput asymRun (whitespace := lax)
[[true, true], [true, false], [false, true],
  [true, false], [false, true]]
```

The sweep sequence enters a two-cycle. Each sweep sets neuron `0` to the old value of neuron `1`,
then sets neuron `1` to the opposite of the new value of neuron `0`. A fixed point would therefore
have to make the two neurons both agree and disagree. No fixed point exists, and the conclusion
of `cycleUpdate_exists_fixedpoint_le_pow` fails for this network. The single-coordinate energy
inequality fails too:

```lean (name := asymEnergy)
-- Inspect the individual update where the asymmetric
-- network increases energy.
def asymMixed : State 2 := fun i => i.val = 0

#eval energy asymNet asymMixed
#eval energy asymNet (updateAt asymNet asymMixed 0)
```
```leanOutput asymEnergy
-1
```
```leanOutput asymEnergy
1
```

Energy increases from `-1` to `1`. The reason for `SymmetricW p` is visible in the
arithmetic: the energy only ever sees the symmetric part $`(W_{ij}+W_{ji})/2`, while the update rule
reads the raw row. When they disagree, an update can increase energy. In this example the energies
at the ends of full sweeps never increase, so inspecting only sweep boundaries would miss the
failed single-coordinate inequality.

Now keep symmetry and change only the schedule. The classic vectorized one-liner
`x = torch.sign(W @ x)` updates every neuron from the same old state, and on a symmetric network
with no diagonal it can cycle forever:

```lean (name := syncRun)
-- Keep weights and tie handling fixed while comparing
-- old-state and sequential updates.
def antiNet : Params Rat 2 where
  W := fun i j => if i = j then 0 else -1
  θ := fun _ => 0

/-- Every neuron reads the *old* state. This is the update
the theorems on this page do not cover. -/
def syncStep (p : Params Rat 2) (s : State 2) : State 2 :=
  fun u => decide (p.θ u ≤ net p s u)

#eval (List.range 4).map fun m =>
  List.ofFn ((syncStep antiNet)^[m] bothUp)
#eval (List.range 4).map fun m =>
  List.ofFn ((sweep2 antiNet)^[m] bothUp)
```
```leanOutput syncRun
[[true, true], [false, false], [true, true], [false, false]]
```
```leanOutput syncRun
[[true, true], [false, true], [false, true], [false, true]]
```

Same network, same start, same tie rule, two schedules. The synchronous one flips both neurons at
once and oscillates with period two forever. The asynchronous sweep reaches a fixed point on the
first pass, because neuron `1` already sees what neuron `0` decided. Synchronous Hopfield dynamics
have their own theory, with a Lyapunov function on pairs of consecutive states rather than on single
states; none of it is what `cycleUpdate_no_nontrivial_cycles` proves.

# Exact ReLU Network Algebra

The approximation developments use exact ReLU identities to assemble larger networks. The module
{src "NN/MLTheory/Proofs/ReLU/Bridge/ReLUMlpBridge.lean"}[`ReLUMlpBridge`]
proves

$$`\operatorname{ReLU}(u)-\operatorname{ReLU}(-u)=u.`

In Lean:

```lean (name := reluBridge)
-- The exact identity recovers a signed value from its two
-- nonnegative ReLU parts.
#check @relu_sub_relu_neg
```

```leanOutput reluBridge (whitespace := lax)
relu_sub_relu_neg : ∀ (u : ℝ),
  NN.MLTheory.Proofs.UniversalApproximation.relu u -
    NN.MLTheory.Proofs.UniversalApproximation.relu (-u) = u
```

The fully qualified `relu` in that output is the definition used by the approximation development.
Reusing that definition
lets these real-valued
lemmas compose without an additional equivalence lemma.

Since `relu u` is `max u 0`, one split on the sign of the input proves the identity:

```lean
-- On either side of zero one hinge vanishes, leaving the
-- signed input after subtraction.
example (u : ℝ) : max u 0 - max (-u) 0 = u := by
  rcases le_total 0 u with h | h
  · rw [max_eq_left h, max_eq_right (neg_nonpos.mpr h),
      sub_zero]
  · rw [max_eq_right h, max_eq_left (neg_nonneg.mpr h),
      zero_sub, neg_neg]
```

`le_total 0 u` produces the two possible order relations, including equality in both branches.
In the first branch, `h` proves that the positive hinge equals `u`, while
`neg_nonpos.mpr h` proves that the other hinge is zero. In the second, the roles reverse and
subtracting `-u` recovers `u`. The proof therefore covers zero without an extra exceptional case.
Each rewrite names the order fact that permits selecting an argument of `max`.

In `relu u - relu (-u)` one of the two operands is always the exact
zero, so for finite IEEE values the result has the same numerical value, including at zero. Bitwise
signed-zero preservation is a separate question:

```lean (name := reluFloat)
-- Compare numerical Float equality on these finite inputs;
-- this does not compare signed-zero bits.
def reluF (u : Float) : Float := max u 0.0

#eval reluF 2.5 - reluF (-2.5)
#eval List.map (fun u => reluF u - reluF (-u) == u)
  [0.0, 1.0e-45, -1.0e-45, 0.1, -3.5e37, 1.0e38]
```

```leanOutput reluFloat
2.500000
```

```leanOutput reluFloat
[true, true, true, true, true, true]
```

Lean's `Float` is binary64, and the list covers zero, magnitudes that would be subnormal in
binary32, a decimal that no binary format stores exactly, and a magnitude near the binary32
maximum. These comparisons use exact numerical equality: `x - 0` and `0 - x` are exact for
every finite `x` in every IEEE 754 format, so the identity holds in binary32 and binary16 for the
same reason it holds here. These tests do not establish a bitwise identity for signed zero or NaNs,
nor do they extend the
real theorem to a backend without a refinement argument.
This identity lets a ReLU network carry an affine term exactly even though each hidden unit clips
negative values. Two units retain its positive and negative parts; subtracting their outputs
recovers the affine term. The bounded-box multiplication construction uses this identity:

```lean (name := reluMul)
-- Inspect the bounded-domain hypotheses and the existential
-- pair of real network layers.
#check @relu_mul_universal_approximation_box
```

```leanOutput reluMul (whitespace := lax)
@relu_mul_universal_approximation_box : ∀ {M : ℝ},
  0 < M → ∀ ε > 0, ∃ hidDim l1 l2, ∀ x ∈ box M, |mulFun x - mlpEval l1 l2 x| < ε
```

The assumption `0 < M` specifies a nondegenerate bounded box. A finite ReLU network is piecewise
affine; along the diagonal of $`\mathbb R^2`, multiplication grows quadratically. Its error
therefore cannot remain uniformly small on the whole plane. Restricting the domain to a box makes
the stated approximation possible.

The construction reduces multiplication to two one-dimensional square approximations through
$`uv=((u+v)^2-(u-v)^2)/4`. If $`|u|,|v|\leq M`, both arguments of the square lie in
$`[-2M,2M]`, which is why a bounded input box gives a common interval for the two approximants.
The source chooses square error below $`2\varepsilon` for each occurrence. Subtraction adds
the two absolute-error bounds, and division by four leaves error below $`\varepsilon`.
The returned layers have input dimension two and output dimension one; `hidDim` records the
width needed to combine the constructions. This is a concrete way that a scalar approximation
lemma becomes a component for a model with several input features.

Read the existential witnesses in order: Lean first chooses the hidden dimension, then two
layers whose types use that dimension. Only after those choices does the statement quantify over
inputs in `box M`. A proof cannot satisfy it by selecting a new network for each input point.
The same returned pair of layers must meet the error budget throughout the entire box.

`relu_mul_universal_approximation_box` uses this algebra inside a uniform approximation argument.
For its construction, quantitative hypotheses, and the distinction between existence, checkpoint
verification, and finite-precision execution, return to *Approximation Theory*.

# Causality In State-Space Models

A recurrent sequence model should not revise an earlier output after future tokens arrive. For a
simple state-space recurrence,

$$`h_{t+1}=A_t h_t+B_t x_t,\qquad
y_t=C_t h_{t+1}+D_t x_t,`

the causal claim can be phrased without derivatives or probability:

$$`\operatorname{take}_{|xs|}
  \bigl(\operatorname{outputs}(\operatorname{run}(xs\mathbin{++}ys))\bigr)
=
\operatorname{outputs}(\operatorname{run}(xs)).`

The theorem says that appending a future suffix `ys` preserves every output already produced for
the prefix `xs`.

{src "NN/MLTheory/Proofs/StateSpace/MambaCausality.lean"}[`MambaCausality`]
proves this statement for three increasingly rich specifications, following the selective
state-space construction of {Informal.citet mamba2024}[]:

- `DiagonalS4Spec`, the diagonal case of the structured state space of
  {Informal.citet s4_2022}[];
- `MambaBlockSpec`;
- `SelectiveMambaBlockSpec`, including its carried convolution history.

The selective theorem includes both recurrent state and convolution history. The following
example states its prefix equality and proves it by applying the library theorem:

```lean (name := selective)
-- Both runs share the model and initial state; only the
-- future suffix is extended.
example {α : Type} [Storage α] [Context α]
    {inputDim stateDim outputDim innerDim convWidth : ℕ}
    (m : Models.SelectiveMambaBlockSpec α inputDim innerDim
      stateDim outputDim convWidth)
    (h0 : Tensor α [innerDim, stateDim])
    (xs ys : Array (Tensor α [inputDim])) :
    (m.runArray h0 (xs ++ ys)).snd.take xs.size =
      (m.runArray h0 xs).snd :=
  selectiveMamba_runArray_append_outputs_prefix m h0 xs ys
```

The five dimensions are arbitrary natural numbers. The scalar type needs `Storage` and `Context`
instances, which provide the tensor representation and operations used by the specification.
The same statement therefore covers `Float`, `Rat`, and `ℝ` instances.

Its content is one equation. `runArray` returns the final recurrent state together with the array of
outputs, and the statement is about the outputs component: run the block on `xs ++ ys`, keep the
first `xs.size` outputs, and you get exactly what running it on `xs` alone produces. Appending
future tokens cannot disturb an earlier output.

The example declaration is itself a proposition whose proof is the named theorem on its final
line. There is no `by` block because applying that theorem already produces a term of exactly
the required equality type. Square-bracketed arguments such as `[Context α]` are instances Lean
supplies from the scalar type, while `m`, `h0`, `xs`, and `ys` are the actual model and data. The
same `m` and `h0` occur on both sides. Comparing two runs that also change the initial state
would be a different statement: earlier outputs can depend on that state even when they cannot
depend on a later suffix.

The theorem is polymorphic over any scalar `α` with a TorchLean `Context`. Its proof is structural:
induct on `xs`, unfold one recurrent step, and apply the induction hypothesis to the updated state
and history. It does not require commutative or exact arithmetic because causality depends on
evaluation order, not algebraic rearrangement.

The selective block's causal convolution needs a window of
recent inputs, so the internal runner threads a newest-first history alongside the recurrent state.
That runner is where the induction happens, and it is stated separately:

```lean (name := history)
-- Keep the initial convolution history fixed when comparing
-- the two output prefixes.
example {α : Type} [Storage α] [Context α]
    {inputDim stateDim outputDim innerDim convWidth : ℕ}
    (m : Models.SelectiveMambaBlockSpec α inputDim innerDim
      stateDim outputDim convWidth)
    (h0 : Tensor α [innerDim, stateDim])
    (history : Array (Tensor α [innerDim]))
    (xs ys : Array (Tensor α [inputDim])) :
    (m.runArrayWithHistory h0 history (xs ++ ys)).snd.take
        xs.size =
      (m.runArrayWithHistory h0 history xs).snd :=
  selectiveMamba_runArrayWithHistory_append_outputs_prefix
    m h0 history xs ys
```

The public theorem above is the `history := #[]` instance of this one, obtained by
`simpa [Models.SelectiveMambaBlockSpec.runArray]`. A streaming implementation may resume with
earlier inputs still in its convolution window; the history theorem covers that case.

A carried history is part of the starting configuration, just like the recurrent tensor.
`runArrayWithHistory` compares runs with the same history on both sides, so a previously buffered
input may affect an early convolution output in both runs. The induction can keep this invariant
because one step consumes the current token and passes the updated state and history to the next
step. No step receives the unprocessed suffix as an additional argument. This identifies what a
streaming implementation must preserve when splitting a sequence into chunks: dropping or
reordering the history changes the initial configuration to which the theorem applies.

The reusable array argument is factored through
[`Scan`](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Proofs/StateSpace/Scan.lean),
which proves append and prefix laws for state-threading scans. `MambaCausality` instantiates that
structure with the S4/Mamba state and convolution history; it does not assert that an optimized
selective-scan kernel refines the specification.

The two simpler specifications satisfy the same output equation with a smaller state:

```lean (name := causality)
-- The smaller recurrent state gives the same prefix
-- equality for S4 and compact Mamba.
example {α : Type} [Storage α] [Context α]
    {inputDim stateDim outputDim : ℕ}
    (m : Models.DiagonalS4Spec α inputDim stateDim
      outputDim)
    (h0 : Tensor α [stateDim])
    (xs ys : Array (Tensor α [inputDim])) :
    (m.runArray h0 (xs ++ ys)).snd.take xs.size =
      (m.runArray h0 xs).snd :=
  diagonalS4_runArray_append_outputs_prefix m h0 xs ys

example {α : Type} [Storage α] [Context α]
    {inputDim stateDim outputDim : ℕ}
    (m : Models.MambaBlockSpec α inputDim stateDim
      outputDim)
    (h0 : Tensor α [stateDim])
    (xs ys : Array (Tensor α [inputDim])) :
    (m.runArray h0 (xs ++ ys)).snd.take xs.size =
      (m.runArray h0 xs).snd :=
  compactMamba_runArray_append_outputs_prefix m h0 xs ys
```


The equations are character for character identical; only the state shape and the spec
structure change. `DiagonalS4Spec` carries a single `[stateDim]` vector, `MambaBlockSpec` adds the
selection projections but keeps the same state shape, and the selective block above is the one that
grows to `[innerDim, stateDim]` plus a history.

The prefix length matters: the next output may depend on the first token of the suffix. To see
this, take an S4 block with one channel, `A = B = C = 1`, `D = 0`, and unit projections. Its state
accumulates the inputs, so output
`t` is the running total of tokens `0` through `t`.

```lean (name := boundary)
-- Use a running sum so the first suffix-dependent output
-- can be read directly from the numbers.
open TorchLean.Tensor (ofFn matrix getScalar)

-- One channel with `A = B = C = 1`, `D = 0`, so the state
-- just accumulates and output `t` is the running total.
def one1 : Tensor Float [1] := ofFn (fun _ => 1.0)
def zero1 : Tensor Float [1] := ofFn (fun _ => 0.0)

def accumS4 : Models.DiagonalS4Spec Float 1 1 1 :=
  { inProj := matrix (fun _ _ => 1.0)
    outProj := matrix (fun _ _ => 1.0)
    ssm := { A := one1, B := one1, C := one1, D := zero1 } }

def tok (v : Float) : Tensor Float [1] := ofFn (fun _ => v)

/-- The prefix `xs` of the causality statement. -/
def xsToks : Array (Tensor Float [1]) := #[tok 1.0, tok 2.0]

/-- Outputs of `xsToks ++ ys`, read back as plain floats. -/
def outs (ys : Array (Tensor Float [1])) : Array Float :=
  (accumS4.runArray (tok 0.0) (xsToks ++ ys)).snd.map
    (fun y => getScalar y 0)

#eval outs #[tok 5.0]
#eval outs #[tok 9.0]
#eval (outs #[]).take 2 == (outs #[tok 5.0]).take 2
#eval (outs #[tok 5.0]).take 3 == (outs #[tok 9.0]).take 3
```

```leanOutput boundary
#[1.000000, 3.000000, 8.000000]
```

```leanOutput boundary
#[1.000000, 3.000000, 12.000000]
```

```leanOutput boundary
true
```

```leanOutput boundary
false
```

The two futures follow the same prefix `#[1.0, 2.0]`. The first two outputs agree, `1.0` and
`1.0 + 2.0`. The third does not: `1 + 2 + 5 = 8` against
`1 + 2 + 9 = 12`. So `.take 2`, which is `.take xs.size`, matches the prefix-only run,
while `.take 3`, which is `.take (xs.size + 1)`, differs across the two futures.

Using zero-based indices, output `xs.size` is the first output that may depend on the suffix.
The theorem fixes all earlier outputs. Extending the equality by one position would fail in this
accumulator example.

## Causality Tests In PyTorch

A corresponding PyTorch test compares a prefix-only run with the start of a longer run:

```
# Compare outputs for one shared prefix under the same model
# and starting configuration.
t = 8
y_short = model(x[:, :t])
y_long = model(x)
torch.testing.assert_close(y_long[:, :t], y_short)
```

This checks approximate agreement for one `x`, one `t`, and one set of weights. The Lean
statement quantifies over every
input array, every suffix, every initial state, and every scalar type with a `Context` instance,
which is why the proof is an induction on `xs` and not a comparison of two tensors.

The proof also does not need `assert_close`, because there is no tolerance in it. Causality is an
equality of prefixes, and prefix equality survives rounding: whatever the arithmetic did to produce
the first `xs.size` outputs, appending `ys` cannot reach back and change it. That is the reason this
theorem is polymorphic in the scalar while the approximation results are not.

# Proof Boundary

The three developments establish different kinds of structure:

:::table +header
*
  * Development
  * Proved object
  * Not established by that theorem
*
  * Hopfield
  * finite real-valued energy and cyclic asynchronous dynamics
  * floating execution or arbitrary update schedules
*
  * ReLU approximation
  * existence of real MLP parameters with uniform error
  * training convergence or binary32 error
*
  * Mamba/S4
  * prefix preservation of spec-level array runners
  * equality with a particular fused scan kernel
:::

The Hopfield example executes over `Rat`; the energy theorem is stated over `ℝ`. The Mamba
causality theorem works over an abstract `Context`, but a runtime refinement theorem is still
needed to connect a backend kernel to the spec runner. The ReLU theorem constructs real-valued
layers, while quantization and rounded execution require the finite-precision bridge described in
the approximation chapter.

A backend proof can use these results once it establishes that the relevant executable operation
implements the specification, with any arithmetic error accounted for.
