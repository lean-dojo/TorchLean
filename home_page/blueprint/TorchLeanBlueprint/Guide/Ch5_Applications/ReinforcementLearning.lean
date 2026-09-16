import VersoManual
import NN.API
import NN.Proofs.RL.Core
import NN.Proofs.RL.Boundary
import NN.Proofs.RL.Environment
import NN.Proofs.RL.FiniteStochasticMDP
import NN.Proofs.RL.Envs.GridWorld
import NN.Proofs.RL.Replay
import NN.Proofs.RL.Algorithms.DQN
import NN.Runtime.RL.Numerics.Float32
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

#doc (Manual) "Reinforcement Learning" =>
%%%
tag := "reinforcement-learning"
%%%

The GridWorld run below raises its measured return from `-0.4` to `3.6`, yet the updated policy
still stops at a wall. I start with the saved path so we can explain both numbers and see what
the policy actually learned to do. In reinforcement learning, the current policy helps create
its own future training data, so understanding an update means following the whole loop:

$$`\text{environment}
\longrightarrow\text{transition}
\longrightarrow\text{rollout or replay}
\longrightarrow\text{return and advantage}
\longrightarrow\text{policy/value update}.`

Each arrow carries assumptions about observation shape, valid actions, finite rewards, episode
boundaries, and alignment of rollout fields. TorchLean represents these assumptions in types,
runtime checks, and theorem hypotheses; we need to know which one we have before using a result.

The implementation assigns these responsibilities to three source directories:

- {srcDir "NN/Spec/RL"}[`NN.Spec.RL`] defines environments, MDPs, Bellman operators, returns, and
  advantages. Numerical trajectories use shape-indexed tensors over the chosen scalar type;
- {srcDir "NN/Runtime/RL"}[`NN.Runtime.RL`] implements checked transitions, replay, rollouts, PPO,
  and Gymnasium communication;
- {srcDir "NN/Proofs/RL"}[`NN.Proofs.RL`] proves structural, numerical, and dynamic-programming
  facts about named objects from the first two layers.

The spec layer is polymorphic in the scalar type, so algebraic return and advantage recurrences can
be run at `ℚ` and inspected as exact fractions before running at `Float`. Exponentials, logarithms,
and other analytic operations need a different scalar interpretation; scalar polymorphism alone
does not make them exact rational computations. The examples use this distinction to separate the
recurrence from rounding.

The standard textbook for the material below is Sutton and Barto,
[*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html)
(second edition). The calculations below also draw on
{Informal.citet gae2015}[] for advantage estimation, {Informal.citet ppo2017}[] for the clipped
policy objective, and {Informal.citet dqn2015}[] for replay-based value learning.

# PPO GridWorld Run

A small CPU run connects the environment, rollout, and update. The environment is a
$`4\times4` GridWorld defined in Lean. The actor and critic use a fixed rollout horizon of 64.
Run one update and write the artifacts to files:

```terminal
# Keep the evaluation path and policy so the measured return
# can be reconstructed.
lake exe torchlean ppo_gridworld --device cpu \
  --updates 1 \
  --eval-every 1 --eval-episodes 1 --eval-max-steps 8 \
  --log /tmp/ppo-gridworld-trainlog.json \
  --policy /tmp/ppo-gridworld-policy.json \
  --path /tmp/ppo-gridworld-path.json
```

A captured run produced the following output and exited 0:

```terminal +output
[TorchLean] arithmetic: native binary32
[TorchLean] execution: eager
[TorchLean] device: cpu
ppo_gridworld: PPO on Lean-native GridWorld (4x4, horizon=64) (device=cpu)
  env: pure Lean dynamics + boundary contract check + formal MDP validity proof available
  eval(step=0) avg_return=-0.400000
  update=1 avg_return=3.600000
  wrote TrainLog JSON: /tmp/ppo-gridworld-trainlog.json
ppo_gridworld: wrote policy snapshot to /tmp/ppo-gridworld-policy.json
ppo_gridworld: wrote path snapshot to /tmp/ppo-gridworld-path.json
ppo_gridworld: done
ppo_gridworld: ok
```

The transcript records four kinds of information:

1. the three `[TorchLean]` banners are configuration facts, described in
   {ref "cli"}[the command-line reference];
2. `update=1` is an execution fact: the PPO program completed one rollout/update cycle without
   tripping a contract check; the configured optimization epochs reuse that collected batch;
3. the two `avg_return` numbers are measurements from one seeded run, not properties of PPO;
4. the `env:` line advertises that a formal MDP validity proof exists for this environment. That
   proof is a theorem about the specification, and we look at it directly later in this chapter.

The implementation is
{src "NN/Examples/Models/RL/PPOGridWorld.lean"}[`NN/Examples/Models/RL/PPOGridWorld.lean`],
and the pure environment is
{src "NN/Spec/RL/Envs/GridWorld.lean"}[`NN.Spec.RL.Envs.GridWorld`].

GridWorld is useful here because an observation has a concrete interpretation: one of sixteen
cells, encoded as a one-hot vector. Four actor logits select a direction, and the next cell can
be calculated without an external simulator. That makes the policy's behavior recoverable from
the saved path, rather than leaving a scalar reward curve as the only evidence of what happened.
The horizon of sixty-four concerns training collection; the eight-step evaluation cap concerns a
separate greedy episode. They need not have the same length.

The command uses eight optimization epochs per collected rollout. Thus `--updates 1` collects
one batch and reuses it for the configured optimization passes. The displayed counter is not a
count of individual environment actions or a promise of one gradient evaluation. This distinction
matters when comparing a longer rollout with more update epochs: one changes the collected data,
while the other changes how often the existing data influences the parameters.

# Episode Returns

The evaluation metric is
{src "NN/Runtime/RL/Eval.lean"}[`rl.eval.averageEpisodeTotalReward`], the undiscounted sum of
rewards over a greedy episode, capped at `--eval-max-steps`. The command also writes the greedy policy and
path. Together with the reward definition, these artifacts let us recompute the two reported
returns.

The path artifact records decoded `(row, column)` states:

```
before: [[0,0],[1,0],[1,1],[0,1],[0,0],[1,0],[1,1],[0,1],[0,0]]
after:  [[0,0],[0,1],[1,1],[1,2],[1,3],[1,3],[1,3],[1,3],[1,3]]
```

The untrained greedy policy walks a four-step cycle and comes back to the start twice. The updated
one moves toward the goal for four steps and then stalls against the right wall. The policy
artifact explains both, once you know that actions are encoded `0=up`, `1=down`, `2=left`,
`3=right`, and that state index `row * 4 + column` is the row-major flattening:

```
before: [1, 2, 0, 0, 3, 0, 3, 3, 3, 0, 1, 3, 2, 1, 0, 3]
after:  [3, 1, 0, 3, 3, 3, 3, 3, 3, 3, 1, 3, 2, 1, 0, 3]
```

Read the `after` row along the visited cells: cell 0 says right, cell 1 says down, cell 5 says
right, cell 6 says right, cell 7 says right. That is exactly the recorded path, and the final
`right` at cell 7 is why the episode stalls: column 3 is the last column, and this GridWorld clamps
at its borders instead of wrapping or failing.

The path contains nine states because eight transitions need both a starting state and a
successor for every action. The policy snapshot instead gives an action for every cell, including
cells the path never visits. Reading them together checks that the chosen actions explain the
observed transitions. It also avoids interpreting a larger return as reaching the goal: the last
four repeated coordinates show exactly what the updated greedy policy still fails to do.

## Reward Shaping

`NN.Spec.RL.Envs.GridWorld` pays $`-1` for every non-terminal step, so no episode can have a
positive total. The example program uses a different transition function with a dense progress
reward:

```
-- Reward distance reduction, then charge a small cost for
-- spending another step.
let progress := goalDistance pos - goalDistance nextPos
{ state := nextState
  reward := progress - 0.05
  terminated := false
  truncated := false }
```

with reward $`+1` for entering the goal, $`0` once at the goal, and a declared reward range of
$`[-1.05, 1]`. The source explains the choice:

> The original sparse `-1 until terminal` reward gave short runs too little learning signal:
> random rollouts rarely found the goal, so PPO received almost no useful signal.

Reward shaping changes the objective presented to the learner. The example records it next to the
transition code. There are two environments to distinguish: the sparse specification used by the
MDP theorems and the shaped variant used by the optimizer.

With that in hand the returns are arithmetic. Manhattan distance to the goal $`(3,3)` along the
`after` path runs $`6,5,4,3,2` and then stays at $`2`:

:::table +header
*
  * Step
  * From
  * To
  * Progress
  * Reward
*
  * 1
  * (0,0)
  * (0,1)
  * $`+1`
  * $`0.95`
*
  * 2
  * (0,1)
  * (1,1)
  * $`+1`
  * $`0.95`
*
  * 3
  * (1,1)
  * (1,2)
  * $`+1`
  * $`0.95`
*
  * 4
  * (1,2)
  * (1,3)
  * $`+1`
  * $`0.95`
*
  * 5 to 8
  * (1,3)
  * (1,3)
  * $`0`
  * $`-0.05` each
:::

Since the metric is an undiscounted sum, we can check it with the spec-layer return function at
$`\gamma=1` and exact rational arithmetic. The first entry of the result is the total return from
the first step:

```lean (name := rlAfterReturn)
-- Read the first entry as the undiscounted total of this
-- eight-step path.
#eval rl.core.discountedReturns (α := ℚ) 1
  ([19/20, 19/20, 19/20, 19/20,
    -1/20, -1/20, -1/20, -1/20] : Tensor ℚ [8])
```

```leanOutput rlAfterReturn (whitespace := lax)
[(18 : Rat)/5, (53 : Rat)/20, (17 : Rat)/10, (3 : Rat)/4,
 (-1 : Rat)/5, (-3 : Rat)/20, (-1 : Rat)/10, (-1 : Rat)/20]
```

$`18/5 = 3.6`, which is the number the command reported. The untrained cycle is the same
computation with its own rewards: two steps of progress and two steps away from the goal, twice.

```lean (name := rlBeforeReturn)
-- The cycle alternates progress toward the goal with
-- equally large movement away.
#eval rl.core.discountedReturns (α := ℚ) 1
  ([19/20, 19/20, -21/20, -21/20,
    19/20, 19/20, -21/20, -21/20] : Tensor ℚ [8])
```

```leanOutput rlBeforeReturn (whitespace := lax)
[(-2 : Rat)/5, (-27 : Rat)/20, (-23 : Rat)/10, (-5 : Rat)/4,
 (-1 : Rat)/5, (-23 : Rat)/20, (-21 : Rat)/10, (-21 : Rat)/20]
```

$`-2/5 = -0.4`. The two sums recover the reported returns from the paths. The improved return
measures progress toward the goal under the shaped reward; the updated policy still does not reach
the goal within this evaluation.

The return tensor lists a suffix total at every timestep. Its first coordinate summarizes the
whole evaluation, but the later coordinates tell the learner how much reward remains after each
visited state. In the updated path, those suffixes become negative once progress stops, because
only the small step penalties remain. Keeping the full vector is therefore important for training
even when the evaluation report displays only one total. A sum over the entire tensor would count
the same rewards repeatedly and would not reproduce the episode metric.

# GridWorld Definition

{src "NN/Spec/RL/Envs/GridWorld.lean"}[`NN.Spec.RL.Envs.GridWorld`] defines the state, actions,
movement, and sparse reward in Lean. The MDP theorems use these definitions directly.

The state is a pair of `Fin` coordinates and an action is one of four directions. Movement is
border-clamped rather than wrapping or erroring, which is the behaviour we saw stall the trained
policy at column 3:

```lean (name := rlClamp)
-- An upward action at the top boundary leaves the state
-- unchanged.
open Spec.RL.Envs.GridWorld in
#eval nextState (width := 4) (height := 4)
  (0, 0) Action.up
```

```leanOutput rlClamp (whitespace := lax)
(0, 0)
```

```lean (name := rlClampRight)
-- A rightward action at the final column is clamped at that
-- column.
open Spec.RL.Envs.GridWorld in
#eval nextState (width := 4) (height := 4)
  (1, 3) Action.right
```

```leanOutput rlClampRight (whitespace := lax)
(1, 3)
```

Moving up from the top row leaves you in the top row, and moving right from the last column leaves
you there. This is a deliberate choice: a clamped grid is a total function, so the environment
needs no error case, and the induced MDP has no absorbing failure state to reason about.

To hand the environment to the finite MDP layer, coordinates are flattened row-major through
mathlib's `finProdFinEquiv` {Informal.citep mathlib2020}[]:

```lean (name := rlEncode)
-- Flatten one coordinate pair and decode its corresponding
-- state index.
open Spec.RL.Envs.GridWorld in
#eval (encode (width := 4) (height := 4) (1, 3),
  decode (width := 4) (height := 4) 7)
```

```leanOutput rlEncode (whitespace := lax)
(7, 1, 3)
```

The mathlib equivalence supplies both the encoding and its inverse laws, so subsequent proofs can
use the flattening as a bijection.

Now the reward. Here is the specification environment, instantiated at the grid the examples use,
together with the reward on a step that does not reach the goal:

```lean (name := rlGridDefs)
/-- The 4x4 grid used throughout this chapter. -/
noncomputable def rlGrid : Spec.RL.Envs.GridWorld 4 4 :=
  { start := (0, 0), goal := (3, 3), discount := 99/100 }

open Spec.RL.Envs.GridWorld in
/-- A step that does not enter the goal costs one unit. -/
theorem rlGridStepCost :
    (rlGrid.step (0, 0) Action.right).reward = -1 := by
  norm_num [step, rlGrid, nextState, Action.right,
    Action.up, Action.down, Action.left, colRight,
    Prod.ext_iff, Fin.ext_iff]
```

The reward has type `ℝ`, so the block proves an equality rather than evaluating a float. The
specification assigns exactly $`-1`; rounding is not part of this statement.

After unfolding the step definition, `norm_num` discharges $`(0,0)\neq(3,3)` and
$`(0,1)\neq(3,3)` using `Fin.ext_iff` and reads off the reward field. A change to the reward
convention would require revisiting this proof.

The sparse reward theorem and the shaped training reward share movement rules, but they do
not define the same value function. A value is an expected accumulated reward under a particular
reward convention and discount. Changing progress rewards can therefore change which policy is
preferred unless an appropriate invariance result is supplied. Here the explicit distinction lets
us use the GridWorld theorem for its stated MDP without silently transferring an optimality claim
to the shaped PPO experiment.

# Actor And Critic Batch Shapes

The GridWorld command does not hide its actor and critic inside the training loop. It starts from a
small reusable configuration, which
{src "NN/API/Models/PPO.lean"}[`nn.models.PPO`]
turns into a two-layer `tanh` MLP per role: the actor ends in four action logits, the critic in one
value.

The constructors take a leading batch shape, so the same configuration describes both individual
observations and batches:

```lean (name := rlCfg)
-- One-hot state features feed separate action-logit and
-- scalar-value heads.
def rlPpoCfg : nn.models.PPO.Config :=
  { observationWidth := 16, hiddenWidth := 32,
    actionCount := 4 }

#eval (rlPpoCfg.input [], rlPpoCfg.actorOutput [],
  rlPpoCfg.criticOutput [])
```

```leanOutput rlCfg (whitespace := lax)
([16], [4], [1])
```

```lean (name := rlCfgBatch)
-- Add the rollout horizon as a shared leading axis for both
-- models.
#eval (rlPpoCfg.input [64], rlPpoCfg.actorOutput [64],
  rlPpoCfg.criticOutput [64])
```

```leanOutput rlCfgBatch (whitespace := lax)
([64, 16], [64, 4], [64, 1])
```

Scoring a single state during a rollout and a whole horizon during optimization uses the same model
at two batch shapes. PyTorch's `nn.Linear(16, 4)` accepts leading dimensions implicitly
{Informal.citep pytorch2019}[]. Here the prefix belongs to the sequential model's type, so
{ref "building-models"}[the model layer] can state and check it.

The configuration also validates, which matters because these numbers usually arrive from flags:

```lean (name := rlCfgOk)
-- The chosen widths provide at least one valid action and
-- one feature.
#eval rlPpoCfg.validate
```

```leanOutput rlCfgOk (whitespace := lax)
Except.ok ()
```

```lean (name := rlCfgBad)
-- An empty action axis cannot supply a categorical choice.
#eval { rlPpoCfg with actionCount := 0 :
  nn.models.PPO.Config }.validate
```

```leanOutput rlCfgBad (whitespace := lax)
Except.error "PPO: action count must be positive"
```

A zero action count would otherwise build a model whose logit axis has length zero, and the first
symptom would be a categorical sampler that cannot return anything.

GridWorld, CartPole, and Pong RAM reuse these two constructors with different observation and
action widths. Environment collection and boundary checking stay outside the model helper.
The actor definition can therefore serve both a single-state rollout and a full-horizon update.

The actor's four outputs are logits; their differences determine the action distribution after
normalization. The critic's one output estimates future reward in the units
of the chosen reward convention. These heads have different training targets even though they
share the same observation width and hidden-width configuration. The `[64, 1]` critic output
must eventually align with one scalar target per collected step; the singleton axis is a model
interface choice, not sixty-four additional value predictions.

# Rollout Data

For a discrete action space of size $`A`, one PPO step stores

$$`(s_t,\;a_t,\;\log\pi_{\mathrm{old}}(a_t\mid s_t),\;
r_t,\;d_t,\;\mathrm{terminated}_t,\;V(s_t),\;V(s_{t+1})).`

The Lean structure in
{src "NN/Runtime/RL/PPO/Rollout.lean"}[`NN.Runtime.RL.PPO.Rollout`]
holds one observation tensor, one `Fin nActions` action, four scalars, and two Boolean markers
per step. `done` marks either termination or truncation and stops advantage continuation.
`terminated` suppresses the next-state value bootstrap. When constructing a step manually,
omitting `terminated` defaults it to `done`, preserving the earlier single-mask behavior.
The rollout container also fixes the number of steps:

```
structure Rollout (α : Type) [TorchLean.Storage α]
    (obsShape : Shape) (nActions horizon : Nat) where
  steps : Array (Step α obsShape nActions)
  /-- Invariant: fixed-horizon rollouts always have exactly `horizon` steps. -/
  steps_size_eq_horizon : steps.size = horizon
```

The array carries its length as a proof field. With separate arrays for rewards, values, and
observations, an extra entry after an early episode end can misalign the fields. A single step
record keeps its fields together, and the proof fixes the number of records.
Neither prevents a caller from putting semantically mismatched observations and rewards into a step.

`Rollout.trainingBatch` converts the records into five named tensors whose shapes are computed
from the horizon:

:::table +header
*
  * Field
  * Shape for horizon 64, 16 features, 4 actions
*
  * `states`
  * $`64\times16`
*
  * `actionsOneHot`
  * $`64\times4`
*
  * `oldLogProb`
  * $`64`
*
  * `advantages`
  * $`64`
*
  * `valueTargets`
  * $`64\times1`
:::

The runtime's shape abbreviations compute these dimensions:

```lean (name := rlBatchShapes)
-- Distinguish the flat scalar batch from the critic’s
-- trailing singleton axis.
#eval (rl.ppo.StateBatchShape 64 [16],
  rl.ppo.LogitsBatchShape 64 4,
  rl.ppo.ScalarBatchShape 64,
  rl.ppo.ValueBatchShape 64)
```

```leanOutput rlBatchShapes (whitespace := lax)
([64, 16], [64, 4], [64], [64, 1])
```

The conversion from rollout to batch is total: because the horizon invariant is already in the
structure, no step of the packing can fail, and `trainingBatch` needs no error case at all. The one
subtlety is deliberate and documented in the source: value targets are computed from the
*unnormalized* advantages, and only the policy term sees z-score normalized advantages. Normalizing
the advantages that feed the value target would change what the critic regresses to, which is a
different algorithm than the one {Informal.citet ppo2017}[] describe.

The stored old log probability belongs to the policy that sampled the action. Recomputing it
after an optimizer update would change the denominator of the importance ratio and erase the
comparison PPO is trying to make. Likewise, `nextValue` refers to the successor before any
automatic environment reset. A correctly sized rollout could still be wrong if either field were
filled from the wrong policy version or episode. The record keeps related fields adjacent;
collection determines their semantic alignment.

# Returns And Generalized Advantage Estimation

Let $`d_t` mark an episode boundary and $`z_t` mark task termination. Both are zero or one.
An external time-limit truncation has $`d_t=1` and $`z_t=0`. The one-step temporal
difference residual used by PPO is

$$`\delta_t
=r_t+\gamma(1-z_t)V(s_{t+1})-V(s_t),`

generalized advantage estimation runs the backward recursion

$$`A_t
=\delta_t+\gamma\lambda(1-d_t)A_{t+1},`

and the matching value target is $`R_t=A_t+V(s_t)` {Informal.citep gae2015}[].

The generic `rl.core.generalizedAdvantageEstimation` API below has one mask and uses it in
both places. These examples have no external truncations, so $`z_t=d_t`. PPO's
`Rollout.generalizedAdvantages` keeps the two masks separate.

Begin with discounted returns at $`\gamma=1/2` over rewards $`1,2,3`:

```lean (name := rlReturns)
-- Accumulate rewards backwards with discount one half and
-- no episode boundaries.
#eval rl.core.discountedReturns (α := ℚ) (1/2)
  ([1, 2, 3] : Tensor ℚ [3])
```

```leanOutput rlReturns (whitespace := lax)
[(11 : Rat)/4, (7 : Rat)/2, 3]
```

By hand: $`G_2=3`, $`G_1=2+\tfrac12\cdot3=\tfrac72`, $`G_0=1+\tfrac12\cdot\tfrac72=\tfrac{11}4`. The
exact fractions are the reason to run this at `ℚ` first. At `Float` the same call gives the decimal
form, and here the two agree because every value is a dyadic rational:

```lean (name := rlReturnsFloat)
-- These particular dyadic results are also exactly
-- representable in binary64.
#eval rl.core.discountedReturns (α := Float) 0.5
  ([1.0, 2.0, 3.0] : Tensor Float [3])
```

```leanOutput rlReturnsFloat (whitespace := lax)
[2.750000, 3.500000, 3.000000]
```

A `done` flag stops the return recursion at an episode boundary:

```lean (name := rlReturnsDone)
-- Stop continuation after the middle reward, leaving the
-- third reward in another episode.
#eval rl.core.discountedReturnsDone (α := ℚ) (1/2)
  ([1, 2, 3] : Tensor ℚ [3]) [false, true, false]
```

```leanOutput rlReturnsDone (whitespace := lax)
[2, 2, 3]
```

Step 1 terminates, so its return is its own reward, $`2`, with no $`\tfrac12\cdot3` attached. Step 0
still bootstraps from step 1, giving $`1+\tfrac12\cdot2=2`. Compare with `[11/4, 7/2, 3]` above:
one Boolean changed two of the three targets. An incorrect boundary flag would silently change
these targets while leaving every tensor shape intact.

Now generalized advantage estimation. With $`\lambda=1` and a zero baseline it must degenerate to
the Monte Carlo return, which is a good self-check on the recursion direction:

```lean (name := rlGaeMonteCarlo)
-- A zero baseline and lambda one reduce this finite
-- recurrence to discounted returns.
#eval rl.core.generalizedAdvantageEstimation (α := ℚ)
  (1/2) 1
  ([1, 2, 3] : Tensor ℚ [3]) [0, 0, 0] [0, 0, 0]
  [false, false, false]
```

```leanOutput rlGaeMonteCarlo (whitespace := lax)
[(11 : Rat)/4, (7 : Rat)/2, 3]
```

This matches `discountedReturns`. The next example changes both the baseline values and
$`\lambda`, setting $`\lambda=1/2`. Its smaller advantages reflect both changes:

```lean (name := rlGae)
-- Nonzero baselines and a terminal final step expose each
-- temporal-difference residual.
#eval rl.core.generalizedAdvantageEstimation (α := ℚ)
  (1/2) (1/2)
  ([1, 2, 3] : Tensor ℚ [3]) [1, 2, 3] [2, 3, 0]
  [false, false, true]
```

```leanOutput rlGae (whitespace := lax)
[(11 : Rat)/8, (3 : Rat)/2, 0]
```

The four tensors contain rewards, values, next values, and termination flags, in that order.
Following the recursion: $`\delta_2=3+0-3=0` because the step is terminal,
$`\delta_1=2+\tfrac12\cdot3-2=\tfrac32`
and $`A_1=\tfrac32`, then $`\delta_0=1+\tfrac12\cdot2-1=1` and
$`A_0=1+\tfrac14\cdot\tfrac32=\tfrac{11}8`. Adding the baseline back recovers the value targets:

```lean (name := rlLambdaReturns)
-- Add the original baseline back before any normalization
-- of policy advantages.
#eval rl.core.returnsFromAdvantages (α := ℚ)
  ([11/8, 3/2, 0] : Tensor ℚ [3]) [1, 2, 3]
```

```leanOutput rlLambdaReturns (whitespace := lax)
[(19 : Rat)/8, (7 : Rat)/2, 3]
```

The parameter $`\lambda` changes how far a temporal-difference residual propagates backward.
With $`\gamma=\lambda=1/2`, each additional step multiplies that contribution by $`1/4`.
This is distinct from discounting the raw reward by $`1/2`: the residual has already compared a
reward-plus-bootstrap prediction against the current baseline. In the final terminal step here,
the baseline equals the reward, so its residual and advantage are zero even though the reward
itself is three. Adding the baseline back restores a value target of three.

## GAE In PyTorch

The same backward recursion can be written explicitly in PyTorch:

```
# Carry only the following advantage backwards through the
# episode mask.
rewards     = torch.tensor([1.0, 2.0, 3.0])
values      = torch.tensor([1.0, 2.0, 3.0])
next_values = torch.tensor([2.0, 3.0, 0.0])
dones       = torch.tensor([0.0, 0.0, 1.0])
terminated  = torch.tensor([0.0, 0.0, 1.0])
gamma, lam = 0.5, 0.5

adv = torch.zeros_like(rewards)
running = 0.0
for t in reversed(range(len(rewards))):
    bootstrap_mask = 1.0 - terminated[t]
    continuation_mask = 1.0 - dones[t]
    delta = rewards[t] + gamma * bootstrap_mask * next_values[t] - values[t]
    running = delta + gamma * lam * continuation_mask * running
    adv[t] = running
print("advantages", adv.tolist())
print("returns   ", (adv + values).tolist())
```

The loop prints:

```
advantages [1.375, 1.5, 0.0]
returns    [2.375, 3.5, 3.0]
```

$`11/8 = 1.375` and $`19/8 = 2.375`, so the implementations agree exactly on these inputs. The
generic TorchLean runtime reuses the spec-layer recurrence, which can first be checked at `ℚ`.
PPO uses the separate masks shown here when a rollout includes truncations.

The difference is what happens when the shapes do not line up. The `values` tensor above is
one-dimensional. Store it as a column instead, which is easy to do by accident when values come
from a critic whose output has a trailing axis of size one, and PyTorch will broadcast:

```
# A column baseline broadcasts against a vector into an
# unintended pairwise matrix.
adv = torch.tensor([1.375, 1.5, 0.0])
values_column = torch.tensor([[1.0], [2.0], [3.0]])
print(tuple((adv + values_column).shape))
```

```
(3, 3)
```

TorchLean requires the advantages and values to have the same tensor shape. A length mismatch
is rejected before the calculation runs:

```lean (name := rlShapeMismatch) +error
-- The return operation requires one value for every
-- advantage coordinate.
def rlAdvThree : Tensor Float [3] :=
  [1.375, 1.5, 0.0]

def rlValTwo : Tensor Float [2] :=
  [1.0, 2.0]

def rlMismatch : Tensor Float [3] :=
  rl.core.returnsFromAdvantages rlAdvThree rlValTwo
```

```leanOutput rlShapeMismatch (whitespace := lax)
Application type mismatch: The argument
  rlValTwo
has type
  Tensor Float [2]
but is expected to have type
  Tensor Float [3]
in the application
  Spec.RL.returnsFromAdvantages rlAdvThree
    rlValTwo
```

The message identifies the mismatched argument and required shape. The PyTorch column example
instead produces a valid tensor of an unintended shape.

The same horizon appears in every input and output tensor of GAE. The specification and runtime
share these functions, so there is one recurrence and no separate array implementation that can
silently shorten a trajectory.

The broadcast counterexample creates all pairwise sums of three advantages and three values.
Its nine entries can look numerically plausible, and taking a mean afterwards could even hide
the extra axis. The desired operation pairs step zero with value zero, step one with value one,
and so on. The shared shape parameter expresses that coordinatewise pairing; it is the reason
an accidental critic column needs an explicit reshape before it can participate in this target
calculation.

# The PPO Objective

For the sampled action at timestep $`t`, the importance ratio is

$$`r_t(\theta)
=\exp\!\left(
\log\pi_\theta(a_t\mid s_t)
-\log\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)
\right),`

and the clipped surrogate that {Informal.citet ppo2017}[] maximize is

$$`L^{\mathrm{clip}}_t(\theta)
=\min\!\left(
r_t(\theta)A_t,\;
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t
\right).`

The minimum makes clipping depend on the advantage sign. Take $`\epsilon=1/4` and a positive
advantage $`A_t=1`. At ratio one, the objective is one:

```lean (name := rlClipOne)
-- At ratio one, the new and behavior policies agree on the
-- sampled action probability.
#eval rl.policy.ppoClippedObjectiveFromRatio (α := Float)
  1.0 1.0 0.25
```

```leanOutput rlClipOne (whitespace := lax)
1.000000
```

At ratio $`3/2`, the clipped branch reaches $`1+\epsilon`:

```lean (name := rlClipUp)
-- A positive advantage stops rewarding ratio increases
-- beyond the upper clip.
#eval rl.policy.ppoClippedObjectiveFromRatio (α := Float)
  1.5 1.0 0.25
```

```leanOutput rlClipUp (whitespace := lax)
1.250000
```

$`1.25` rather than $`1.5`: the gradient of this term with respect to the ratio is zero, removing
this sample's incentive to increase the ratio further. This does not enforce a hard
bound on policy movement: other samples, shared parameters, and optimizer state still affect it.
Pull the ratio down to
$`1/2` and nothing is clipped, because the minimum picks the unclipped branch:

```lean (name := rlClipDown)
-- Reducing a positively advantaged action remains penalized
-- below the lower clip.
#eval rl.policy.ppoClippedObjectiveFromRatio (α := Float)
  0.5 1.0 0.25
```

```leanOutput rlClipDown (whitespace := lax)
0.500000
```

With a negative advantage, multiplication by $`A_t` reverses the order of the two candidate terms:

```lean (name := rlClipNegUp)
-- For a negative advantage, an increased action probability
-- stays on the penalized branch.
#eval rl.policy.ppoClippedObjectiveFromRatio (α := Float)
  1.5 (-1.0) 0.25
```

```leanOutput rlClipNegUp (whitespace := lax)
-1.500000
```

```lean (name := rlClipNegDown)
-- A negative advantage flattens after the ratio falls below
-- the lower clip.
#eval rl.policy.ppoClippedObjectiveFromRatio (α := Float)
  0.5 (-1.0) 0.25
```

```leanOutput rlClipNegDown (whitespace := lax)
-0.750000
```

At ratio $`3/2` with $`A_t=-1`, the objective is $`-1.5`, since
$`\min(-1.5,-1.25)=-1.5`. The clipped objective keeps penalizing an increase in this ratio beyond
$`1+\epsilon`; it does not flatten there. In the opposite direction it flattens at $`-0.75` once
the ratio falls below $`1-\epsilon`. Thus the objective is not a symmetric constraint on policy
movement. For a categorical action with fixed positive old probability, the ratio also has the
separate probability bound $`r_t\le 1/\pi_{\mathrm{old}}(a_t\mid s_t)`.

The PyTorch expression gives the same five numbers:

```
# Evaluate both signs with one shared clipping interval.
ratio = torch.tensor([1.0, 1.5, 0.5, 1.5, 0.5])
advan = torch.tensor([1.0, 1.0, 1.0, -1.0, -1.0])
eps = 0.25
obj = torch.min(ratio * advan,
                torch.clamp(ratio, 1 - eps, 1 + eps) * advan)
print(obj.tolist())
```

```
[1.0, 1.25, 0.5, -1.5, -0.75]
```

The full loss to minimize adds a critic term and an optional entropy bonus,

$$`-L^{\mathrm{clip}}_t
+c_v\,\bigl(V_\theta(s_t)-R_t\bigr)^2
-c_e\,H\!\left[\pi_\theta(\cdot\mid s_t)\right],`

and the differentiable batch version over backend references lives in
{src "NN/Runtime/RL/PolicyGradient/Autograd.lean"}[`NN.Runtime.RL.PolicyGradient.Autograd`],
where the categorical log probability is built from `logSoftmax` and a one-hot action tensor. The
pure scalar functions above are the scalar formulas without a tape. The ratio-based clipped
objective can be checked
exactly at `ℚ`; computing the ratio from log probabilities additionally requires an exponential.
Their tape implementations live inside {ref "runtime-autograd"}[the autograd runtime] and need
their own agreement arguments.

The ratio is computed for the action actually stored in the rollout, not for whichever action
the updated actor now prefers. A ratio above one says that sampled action has become more likely
under the new policy; the advantage supplies the judgment about whether that change is desirable.
The five examples separate those two pieces by providing ratios directly. They do not evaluate
log probabilities or sample an action, which is why they can isolate the clipped minimum without
involving a categorical distribution or a random seed.

A training loss minimizes the negative surrogate. At positive advantage, increasing the ratio
inside the unclipped interval improves the surrogate and lowers that loss. The critic term has
a different purpose: it fits the return target used to estimate advantages. The entropy term
rewards a less concentrated action distribution when its coefficient is positive. These terms
can pull shared parameters in different directions, so the flat branch of one sample's surrogate
is not a bound on the complete update.

# Checked Arithmetic For Returns And Ratios

Returns, advantages, and importance ratios give us three places to look for non-finite values.
A return accumulates over a horizon, a ratio is an exponential of a difference of logarithms, and
normalization divides by a standard deviation that can be zero. Binary32 arithmetic can produce
an infinity or a NaN and continue the calculation with that value
{Informal.citep goldberg1991}[].

The checked helpers under
{srcDir "NN/Runtime/RL/Numerics/Float32"}[`NN.Runtime.RL.Numerics.Float32`]
run the same recurrences over an executable binary32 semantics and return
`Except String ...`. They are described in {ref "fp32-soundness"}[the float32 soundness chapter];
here we use them.

Start at the boundary. Casting a host binary64 value into the executable binary32 type can
overflow, and the cast refuses rather than quietly returning an infinity:

```lean (name := rlCastReject)
-- Reject a host value whose conversion overflows the
-- binary32 range.
#eval match rl.numerics.float32.ofFloatChecked 1e39 with
  | .ok _ => "accepted"
  | .error _ => "rejected before it reached the update"
```

```leanOutput rlCastReject (whitespace := lax)
"rejected before it reached the update"
```

The full error message names the input and the resulting value. I match on the constructor here
to keep the rejection of $`10^{39}` readable. An accepted value carries its bit
pattern, which is a reminder that this type is a binary32 encoding rather than a real number:

```lean (name := rlCastOk)
-- Inspect the accepted binary32 encoding of one half.
#eval (rl.numerics.float32.ofFloatChecked 0.5).map
  ExecFloat.Binary.toBits32
```

```leanOutput rlCastOk (whitespace := lax)
Except.ok 1056964608
```

$`1056964608` is `0x3F000000`, the binary32 encoding of $`0.5`.

Now the discounted backup $`r+\gamma(1-d)\,V(s')`, first on ordinary inputs:

```lean (name := rlBackupOk)
-- Compute one reward plus a discounted, nonterminal
-- bootstrap.
open Runtime.RL.Numerics.Float32 in
#eval (discountedBackupChecked 1 (1 / 2) 2 false).map
  (Float32.toFloat ∘ ExecFloat.Binary.toFloat32)
```

```leanOutput rlBackupOk (whitespace := lax)
Except.ok 2.000000
```

Now supply a bootstrap value that is already infinite:

```lean (name := rlBackupInf)
-- Supply an infinite critic value to locate the failed
-- intermediate operation.
open Runtime.RL.Numerics.Float32 in
#eval discountedBackupChecked 1 (1 / 2)
  (ExecFloat.Binary.infinity false) false
```

```leanOutput rlBackupInf (whitespace := lax)
Except.error "RL float32: non-finite configured binary32 value at
  discountedBackup/mul(t1,bootstrap): inf"
```

The label identifies the multiplication that produced the non-finite value:
$`\gamma\cdot(1-d)\cdot V(s')`. Larger checked routines retain the labels of their primitive
operations, so the diagnostic locates the failing calculation within the update.

For background on formal floating-point analysis, see {Informal.citep flocq2011}[]. Here we can
inspect the same recurrence with an interval calculation: compute the returns in binary32 and
propagate intervals alongside them, then check whether each binary32 result lies inside its interval:

```lean (name := rlEnclosure)
-- Compare the checked recurrence’s values with
-- independently propagated intervals.
open rl.numerics.float32 in
#eval do
  let g ← ofFloatChecked 0.5
  let r ← castTensorChecked (s := [3])
    ([1.0, 2.0, 3.0] : Tensor Float [3])
  let out ← discountedReturnsChecked (n := 3) g r
  let iv := discountedReturnsIntervals (n := 3) g r
  pure (returnsWithinIntervals (n := 3) out iv)
```

```leanOutput rlEnclosure (whitespace := lax)
Except.ok true
```

The successful interval check returns `Except.ok true`: the outer constructor says the checked
binary32 calculations completed, and the Boolean says their returned values passed the enclosure
test. Those are separate conditions. A failed cast or intermediate overflow would return an error
before there were finite results to compare. The dyadic three-reward example is chosen so its
expected values are already known; it demonstrates how the numerical checks connect to the
recurrence rather than estimating error for an unknown long trajectory.

## Finiteness And Refinement Hypotheses

An equality proving that the checked routine returns `.ok` supplies the finiteness hypotheses
needed by refinement theorems:

```lean (name := rlBackupBridge)
-- Successful execution gives both the spec equality and
-- finite intermediate products.
open Runtime.RL.Numerics.Float32 Floats.IEEE754 Spec.RL in
#check @discountedBackup_eq_ok
```

```leanOutput rlBackupBridge (whitespace := lax)
discountedBackup_eq_ok : ∀ (reward gamma bootstrap : Float32Exec) (done : Bool) (out : Float32Exec),
  discountedBackupChecked reward gamma bootstrap done = Except.ok out →
    ExecFloat.Binary.isFinite (ExecFloat.mul gamma (continueMask done)) = true ∧
      ExecFloat.Binary.isFinite ((ExecFloat.mul gamma (continueMask done)).mul bootstrap) = true ∧
        ExecFloat.Binary.isFinite (ExecFloat.add reward ((ExecFloat.mul gamma (continueMask
          done)).mul bootstrap)) =
            true ∧
          out = discountedBackup reward gamma bootstrap done
```

Read the conclusion right to left. The last conjunct says the checked routine agrees with the
spec-layer formula, and the first three say every intermediate was finite. So one successful call
gives you both the value equality and the three finiteness conditions that the bridge theorems in
{src "NN/Proofs/RL/Floats/IEEE32Exec.lean"}[FloatLib binary32]
ask for.

These results cover selected scalar recurrences. They do not establish agreement with a parallel
CUDA implementation or bound the numerical error of a complete PPO training run.

# Bellman Operators And Fixed Points

The same vocabulary supports the classical theory. For a fixed policy $`\pi`,

$$`(T^\pi V)(s)
=r(s,\pi(s))
+\gamma\mathbb E_{s'\sim P(\cdot\mid s,\pi(s))}[V(s')],`

and the optimality operator takes the best action instead:

$$`(T^\star V)(s)
=\max_a\left(
r(s,a)+\gamma\mathbb E_{s'\sim P(\cdot\mid s,a)}[V(s')]
\right).`

For $`0\le\gamma<1` both are contractions in the sup metric, which is the single fact from which
value iteration, uniqueness of the fixed point, and the familiar geometric error bounds follow.
TorchLean proves it for finite stochastic MDPs:

```lean (name := rlContraction)
-- Validity supplies probability rows and a discount
-- strictly below one.
open Proofs.RL.MDP (valueSupDist) in
open Proofs.RL.FiniteStochastic Spec.RL.FiniteStochastic in
#check @bellmanOptimality_contraction
```

```leanOutput rlContraction (whitespace := lax)
@bellmanOptimality_contraction : ∀ {nStates nActions : ℕ}
  [inst : Fact (0 < nStates)] [inst_1 : Fact (0 < nActions)]
  (mdp : MDP nStates nActions),
  Valid mdp →
    ∀ (values₁ values₂ : Spec.RL.ValueFunction ℝ nStates),
      valueSupDist (bellmanOptimality mdp values₁)
            (bellmanOptimality mdp values₂) ≤
        mdp.discount * valueSupDist values₁ values₂
```

The discount bound arrives through
`Valid mdp`, which also requires the transition rows to be probability distributions; the state and
action counts are positive by instance arguments; and `valueSupDist` is a `Finset.sup'` over states,
which is why nonemptiness has to be available. Nothing here is inferred from a simulator run.

Iterating that inequality gives a geometric error bound for value iteration:

```lean (name := rlIterate)
-- Iterate toward a supplied fixed point and inspect the
-- discount power in the bound.
open Proofs.RL.MDP (valueSupDist) in
open Proofs.RL.FiniteStochastic Spec.RL.FiniteStochastic in
#check @bellmanOptimality_iterate_error_to_fixedPoint
```

```leanOutput rlIterate (whitespace := lax)
@bellmanOptimality_iterate_error_to_fixedPoint : ∀
  {nStates nActions : ℕ} [inst : Fact (0 < nStates)]
  [inst_1 : Fact (0 < nActions)] (mdp : MDP nStates nActions),
  Valid mdp →
    ∀ (v vStar : Spec.RL.ValueFunction ℝ nStates),
      bellmanOptimality mdp vStar = vStar →
        ∀ (k : ℕ),
          valueSupDist
              ((bellmanOptimality mdp)^[k] v) vStar ≤
            mdp.discount ^ k * valueSupDist v vStar
```

$`k` applications of the operator shrink the distance to a fixed point by $`\gamma^k`. The companion
theorems prove that such a fixed point is unique for both operators, and related results appear
for deterministic and stochastic settings: first for deterministic finite MDPs in
{src "NN/Proofs/RL/MDP.lean"}[`NN.Proofs.RL.MDP`],
once for stochastic ones in
{src "NN/Proofs/RL/FiniteStochasticMDP.lean"}[`FiniteStochasticMDP`],
and once more for general measurable state spaces in
{src "NN/Proofs/RL/MarkovMDP.lean"}[`NN.Proofs.RL.MarkovMDP`].

The sup distance asks for the largest error over all states. A stochastic transition row
averages successor errors using nonnegative weights that sum to one, so this averaging cannot
exceed that largest error. Discounting then multiplies it by $`\gamma`, which is the source of
the contraction factor. This explains both parts of `Valid`: probability-row conditions control
the averaging, and a discount below one makes the factor contractive. Neither condition concerns
how a neural network is initialized or optimized.

The iteration counter in this theorem counts exact Bellman applications. A PPO epoch instead
updates policy and value-network parameters using a sampled surrogate objective, so its logged
epoch number cannot be substituted for $`k` in the displayed bound. To use the bound for an
approximate value-iteration implementation, one would also need to quantify the error introduced
at each approximate application. The fixed-point hypothesis supplies the comparison function
$`vStar`; the theorem then controls distance to it from any initial value function.

## GridWorld Validity

A theorem about `Valid mdp` is only useful if some concrete environment satisfies it. GridWorld
does, and the proof obligation is exactly the discount condition:

```lean (name := rlValidCheck)
-- A deterministic GridWorld transition induces a valid
-- stochastic row.
open Proofs.RL.Envs.GridWorld Spec.RL.FiniteStochastic in
#check @toFiniteStochasticMDP_valid
```

```leanOutput rlValidCheck (whitespace := lax)
@toFiniteStochasticMDP_valid : ∀ {width height : ℕ}
  (gw : Spec.RL.Envs.GridWorld width height),
  0 ≤ gw.discount → gw.discount < 1 →
    Valid gw.toFiniteStochasticMDP
```

The theorem packages the proof that a deterministic successor state gives a row-stochastic
kernel. Applying it to the grid from earlier in this chapter takes two
`norm_num` calls:

```lean (name := rlValidGrid)
/-- The MDP induced by `rlGrid` is well formed. -/
theorem rlGridValid :
    Spec.RL.FiniteStochastic.Valid
      rlGrid.toFiniteStochasticMDP :=
  Proofs.RL.Envs.GridWorld.toFiniteStochasticMDP_valid
    rlGrid (by norm_num [rlGrid]) (by norm_num [rlGrid])
```

This is the theorem the `ppo_gridworld` banner refers to when it says a formal MDP validity proof is
available, and the example file proves the same statement for its own grid.

The Bellman results concern an operator on value functions over a finite state space with a known
kernel. PPO optimizes a parameterized stochastic policy from sampled finite trajectories with a
clipped surrogate and a learned critic. The operator theorems do not establish convergence of that
sampled optimization procedure.

# External Environments

GridWorld's transition function is a Lean definition that proofs can refer to. CartPole instead
runs in Gymnasium, a Python library. Its transitions enter Lean through a checked communication
boundary.

The bridge uses a subprocess speaking JSON lines, one object per line, and checks a contract on
the Lean side of the exchange. The server is
{src "scripts/rl/gymnasium_server.py"}[`scripts/rl/gymnasium_server.py`],
and its docstring explains that you can run the file directly and type the protocol by hand.
The exchange below shows what crosses the boundary:

```terminal +output
$ python3 scripts/rl/gymnasium_server.py --env-id CartPole-v1
{"cmd": "describe"}
{"ok":true,"n_actions":2,"obs_shape":[4],"obs_dtype":"float32"}
{"cmd": "reset", "seed": 0}
{"ok":true,"obs":[0.013696168549358845,
                  -0.023021329194307327,
                  -0.04590264707803726,
                  -0.04834723472595215]}
{"cmd": "step", "action": 0}
{"ok":true,"obs":[0.013235742226243019,
                  0.17272774875164032,
                  -0.04686959087848663,
                  -0.3551521897315979],
 "reward":1.0,"terminated":false,"truncated":false}
{"cmd": "step", "action": 99}
{"ok":false,"error":"AssertionError: 99 (<class 'int'>) invalid"}
{"cmd": "close"}
{"ok":true}
```

The observation lines are wrapped here to fit the page; on the wire each reply is one line.
The `describe` reply supplies the dimensions Lean uses to construct its tensor shapes.
Gymnasium's `terminated` and `truncated` arrive as separate flags: an external time-limit
truncation generally retains the value bootstrap, while termination ends it.
The PPO collector records both the `done` episode boundary and the task's `terminated`
flag. GAE uses `terminated` to mask the next-state value and `done` to stop advantage continuation.
A truncated step therefore bootstraps from the final observation before auto-reset, without
including rewards from the following episode. The rollout viewer uses the same calculation as
training. The generic single-mask GAE API retains its existing semantics.

Python refuses the out-of-range action. The `close` command gets an acknowledgement, so the
Lean side can distinguish a clean shutdown from a crash.

The `describe` handshake fixes the observation and action contract before training tensors
are constructed. CartPole supplies four real-valued observations and two actions, while Pong RAM
supplies byte-like observations and a different action space. Once that boundary has been checked,
the common collector can use the same rollout structure. The Python process still owns the
transition dynamics; successful JSON exchange establishes that the data reached Lean in the
expected form, rather than proving how the simulator produced it.

The rejected action `99` probes one boundary without changing the observation dimensions. It
shows why a successful shape handshake is only the beginning of the protocol: every subsequent
message still carries values that need checking. Conversely, a valid action does not imply that
its resulting reward is finite or its observations lie in the declared range. The following
contracts separate these conditions, so a failure can identify the field responsible instead of
reporting only that an episode could not continue.

## Transition Contracts

Every message that comes back is checked against a
{src "NN/Runtime/RL/Boundary/Core.lean"}[`rl.boundary.Contract`]
before it becomes a `Transition`. A contract records the range, finiteness, and flag checks
requested by its caller. The three environment examples declare:

:::table +header
*
  * Environment
  * `obsRange?`
  * `rewardRange?`
  * Finiteness
*
  * GridWorld (Lean-native)
  * `some (0, 1)`
  * `some (-1.05, 1)`
  * both on
*
  * CartPole (Gymnasium)
  * `none`
  * `none`
  * both on
*
  * Pong RAM (Gymnasium)
  * `some (0, 255)`
  * `none`
  * both on
:::

GridWorld's one-hot observations permit a $`[0,1]` range check, and Pong's RAM bytes permit a
$`[0,255]` check. CartPole's contract leaves the ranges unspecified. All three contracts require
finite observations and rewards, rejecting NaNs and infinities before they reach an update.

The following contract uses GridWorld's range settings and also enables the done-flag exclusivity
check that the shipped example leaves off:

```lean (name := rlContractDef)
/-- A contract for a 2-wide observation and 4 actions. -/
def rlContract : rl.boundary.Contract [2] 4 :=
  { checkObsFinite := true
    checkRewardFinite := true
    obsRange? := some (0, 1)
    rewardRange? := some (-1.05, 1)
    requireExclusiveDoneFlags := true }

/-- An in-range observation pair. -/
def rlObs : Tensor Float [2] := [0.25, 0.75]

/-- An observation with an entry above the upper bound. -/
def rlObsBad : Tensor Float [2] := [0.25, 1.75]
```

A well formed step passes and yields a typed transition, whose reward we print to show that the
`Except` carries a value rather than just a verdict:

```lean (name := rlCheckOk)
-- Preserve the accepted reward while checking both
-- observations and the action.
#eval (rl.boundary.checkTransition rlContract rlObs rlObs 2
  0.95 false false).map (fun tr => tr.reward)
```

```leanOutput rlCheckOk (whitespace := lax)
Except.ok 0.950000
```

A NaN reward is refused by name:

```lean (name := rlCheckNaN)
-- Exercise reward finiteness without changing any other
-- transition field.
#eval (rl.boundary.checkTransition rlContract rlObs rlObs 2
  (0.0 / 0.0) false false).map (fun tr => tr.reward)
```

```leanOutput rlCheckNaN (whitespace := lax)
Except.error "RL boundary: expected finite reward, got NaN."
```

A reward that is finite but outside the declared range is refused with both numbers, which is the
message you want when an environment wrapper has quietly rescaled its rewards:

```lean (name := rlCheckRange)
-- A finite reward can still violate the environment’s
-- declared range.
#eval (rl.boundary.checkTransition rlContract rlObs rlObs 2
  7.5 false false).map (fun tr => tr.reward)
```

```leanOutput rlCheckRange (whitespace := lax)
Except.error "RL boundary: reward out of range: got
  7.500000, expected in [-1.050000, 1.000000]."
```

The observation check names the field, so a bad `nextObservation` is not confused with a bad
`observation`:

```lean (name := rlCheckObs)
-- Change only the next observation to identify that field
-- in the error.
#eval (rl.boundary.checkTransition rlContract rlObs rlObsBad
  2 0.95 false false).map (fun tr => tr.reward)
```

```leanOutput rlCheckObs (whitespace := lax)
Except.error "RL boundary: nextObservation tensor out of
  range: expected all entries in [0.000000, 1.000000]."
```

With exclusivity requested, a step claiming both `terminated` and `truncated` is refused. Gymnasium
does not promise these are mutually exclusive, which is why the check is opt-in:

```lean (name := rlCheckFlags)
-- This contract explicitly forbids simultaneous termination
-- and truncation flags.
#eval (rl.boundary.checkTransition rlContract rlObs rlObs 2
  0.95 true true).map (fun tr => tr.reward)
```

```leanOutput rlCheckFlags (whitespace := lax)
Except.error "RL boundary: both `terminated` and
  `truncated` are true (contract requires exclusivity)."
```

Finally, the action check is what turns an untyped index from outside into a `Fin nActions`:

```lean (name := rlCheckAction)
-- Seven cannot index an action space containing four
-- choices.
#eval rl.boundary.checkAction 4 7
```

```leanOutput rlCheckAction (whitespace := lax)
Except.error "RL boundary: action out of range:
  7 (nActions=4)."
```

After this check, the action has type `Fin nActions`. Subsequent indexing can use that bounded
index without repeating the range check.

These failures exercise distinct clauses. A NaN reward violates finiteness; the reward `7.5`
violates a declared interval even though it is finite; the bad observation violates a tensor-wide
range condition; and the action index violates a discrete bound. Combining them into one invalid
transition would obscure which check was responsible. Varying one field at a time makes the error
messages useful specifications of the boundary and shows which accepted data downstream code may
rely on.

## Checker Soundness

The checker has a theorem connecting successful execution to the declared contract:

```lean (name := rlContractBridge)
-- The successful checker equality is the hypothesis used to
-- derive the contract proposition.
open Proofs.RL.Boundary Runtime.RL.Boundary in
#check @contractHolds_of_checkTransitionFin_eq_ok
```

```leanOutput rlContractBridge (whitespace := lax)
@contractHolds_of_checkTransitionFin_eq_ok : ∀
  {obsShape : Shape} {nActions : ℕ}
  (c : Contract obsShape nActions)
  (observation nextObservation : Tensor Float obsShape)
  (action : Fin nActions) (reward : Float)
  (terminated truncated : Bool)
  (t : Transition obsShape nActions),
  checkTransitionFin c observation nextObservation action
        reward terminated truncated =
      Except.ok t →
    ContractHolds c t
```

`ContractHolds` is a Prop-level structure with one field per clause of the contract: the done flags,
the reward, and both observations. The theorem says the executable checker is sound with respect to
it, so a proof of the checker's successful equality yields the propositions downstream lemmas
quantify over. A printed `.ok` from native execution does not itself supply that proof term. This
is the proof-carrying code arrangement in miniature
{Informal.citep necula1997}[]: the untrusted producer supplies data, the consumer checks it once at
the boundary, and downstream lemmas use the resulting contract hypotheses.

The theorem establishes the checked contract clauses. An environment with incorrect dynamics could
still emit finite, correctly shaped values that pass those checks. Agreement with CartPole's
intended dynamics requires a separate specification.

# CartPole And Pong RAM Results

CartPole and Pong RAM exercise the same collector with external environments.

`ppo_cartpole` runs the same PPO loop against Gymnasium. Here is a recorded 30-update run with the
shipped hyperparameters, evaluating every fifth update over five episodes with a 200-step cap:

```terminal +output
$ torchlean ppo_cartpole --updates 30 --eval-every 5 \
    --eval-episodes 5 --eval-max-steps 200 \
    --log-json /mnt/build/rl-cartpole-log.json
[TorchLean] arithmetic: native binary32
[TorchLean] execution: eager
[TorchLean] device: cpu
ppo_cartpole: PPO on CartPole-v1 (horizon=64) (device=cpu)
  env: Python Gymnasium subprocess (JSON-lines bridge)
       + Lean boundary contract
  eval(step=0) avg_return=9.600000
  update=5 avg_return=10.000000
  update=10 avg_return=9.800000
  update=15 avg_return=9.800000
  update=20 avg_return=9.800000
  update=25 avg_return=10.000000
  update=30 avg_return=10.000000
  wrote TrainLog JSON: /mnt/build/rl-cartpole-log.json
ppo_cartpole: done
ppo_cartpole: ok
```

The evaluated greedy policy survives about ten steps both before and after training. With
horizon $`64`, thirty updates collect roughly two thousand environment steps. The collector and
updates completed, but the measured return shows little improvement. That observation alone
does not tell us whether collecting more steps would improve this policy.

For CartPole, `avg_return` is capped by `--eval-max-steps`: an episode that survives all eight
steps under an eight-step cap reports `8.000000`. That measurement cannot distinguish policies that
survive eight steps from those that survive much longer. The run above uses $`200`, and the JSON
log records the cap alongside the curve.

`ppo_pong_ram` is the third example, and requires the external ALE package and environment assets.
An environment missing that
package reports the following failure:

```terminal +output
$ torchlean ppo_pong_ram --updates 1
[TorchLean] arithmetic: native binary32
[TorchLean] execution: eager
[TorchLean] device: cpu
ppo_pong_ram: PPO on ALE/Pong-v5
  (obs=ram, horizon=128) (device=cpu)
  env: Python Gymnasium subprocess (ALE)
       + Lean boundary contract
  starting env: ALE/Pong-v5 (obs_type=ram)
Traceback (most recent call last):
  ...
ModuleNotFoundError: No module named 'ale_py'
  ...
RuntimeError: Requested an `ALE/...` environment id, but
`ale-py` could not be imported/registered. Detected
gymnasium=1.3.0, ale-py=not installed. Try: python3 -m pip
install --user --upgrade ale-py 'gymnasium>=1.0'
error: Gymnasium: unexpected EOF from server
  (process exited?)
```

Two tracebacks are elided. The `RuntimeError` names the missing package, reports the installed
`gymnasium` version, and gives
the command to fix it. The Lean-side exception on the last line says only that the pipe closed. The
diagnostic is visible because the subprocess writes to the same standard error, not because the Lean
side captured it, so in a context that separates the two streams the useful half can go missing.
Keep both output streams when diagnosing an external environment failure.

The CartPole trace illustrates why evaluation settings belong beside the metric. Thirty update
cycles can finish successfully while the greedy policy's survival time remains near ten steps.
A different exploration distribution might collect different trajectories even when greedy
evaluation looks unchanged. The trace reports what this evaluation observed; it does not isolate
whether the limiting factor is optimization, advantage estimates, rollout coverage, or model
capacity. The Pong failure occurs earlier still, before the environment can provide a trajectory.

# Off-Policy Data: Replay Buffers And DQN

PPO reuses a collected rollout for its optimization epochs, then collects another rollout.
DQN retains past transitions for later updates {Informal.citep dqn2015}[]. Drawing random
minibatches from that buffer lets an update combine transitions collected at different times.
The small example below uses a deterministic sampler whose selected entries we can inspect.

A replay buffer must keep its size within its capacity. Pushing three transitions into a buffer of
capacity two demonstrates the full-buffer case:

```lean (name := rlReplay)
/-- A one-entry observation, keeping the print short. -/
def rlObsOne (x : Float) : Tensor Float [1] :=
  [x]

open Runtime.RL.Replay in
#eval
  let mk : Float → Transition Float [1] 2 := fun k =>
    { state := rlObsOne k, action := ⟨0, by decide⟩,
      reward := k, nextState := rlObsOne k, done := false }
  let b0 : Buffer Float [1] 2 := Buffer.empty 2
  let b := ((b0.push (mk 1)).push (mk 2)).push (mk 3)
  (b.size, b.items.map (fun tr => tr.reward))
```

```leanOutput rlReplay (whitespace := lax)
(2, #[2.000000, 3.000000])
```

The oldest transition is gone and the size stayed at the capacity. A bounded Python `deque`
has the same eviction behavior:

```
# Compare oldest-first eviction with a capacity-two Python
# container.
>>> from collections import deque
>>> b = deque(maxlen=2)
>>> for k in (1.0, 2.0, 3.0): b.append(k)
>>> list(b)
[2.0, 3.0]
```

TorchLean also proves the size after pushing into a full buffer:

```lean (name := rlPushFull)
-- Positivity and a full initial buffer are needed for this
-- capacity equality.
open Proofs.RL.Replay Runtime.RL.Replay in
#check @push_size_of_full
```

```leanOutput rlPushFull (whitespace := lax)
@push_size_of_full : ∀ {α : Type} {obsShape : Shape}
  {nActions : ℕ} (b : Buffer α obsShape nActions)
  (t : Transition α obsShape nActions),
  0 < b.capacity → b.items.size = b.capacity →
    (b.push t).items.size = b.capacity
```

`push_size_of_room` says the size increases by one when it is below capacity; `push_size_of_full`
says it stays at capacity when the buffer is full and capacity is positive. These are size
statements under their respective hypotheses. The push definition appends the new transition and,
if necessary, removes the oldest entry, but the displayed size theorem alone does not establish
which transitions were retained.

The type also pins the transition down. `Transition Float [1] 2` fixes the observation shape and the
action count, and the action field is a `Fin 2`, which is why the literal above needs `by decide`.
A `deque` does not check these fields when an item is appended; its callers must arrange those
checks separately.

The buffer stores oldest-first entries in a bounded FIFO array. Its capacity controls how much
past experience remains available, while the sampling policy controls which retained entries
participate in an update. These are separate choices. The capacity theorem assumes an initially
full valid buffer and positive capacity; a manually constructed oversized buffer is not covered
by that hypothesis. The public empty constructor and repeated pushes are the path illustrated
here, so the capacity-two example exposes eviction without requiring a large training run.

## DQN Replay Example

`dqn_replay` uses two transitions and two hand-written Q-functions, with no training loop. The small
example lets us follow each TD target into both loss functions.

```terminal +output
$ torchlean dqn_replay
dqn_replay: begin
stored transitions: 2
sampled transitions: 4
DQN minibatch MSE loss:   0.917800
DQN minibatch Huber loss: 0.452500
soft target update example: 1.000000
dqn_replay: ok
```

`sampleContiguous` indexes modulo the buffer size, so a batch of four from a buffer of two contains
each transition twice. This deterministic wraparound repeats data without decorrelating samples.
Callers needing randomized replay must choose a sampler with that policy.

Here are the pieces. The observations are two-dimensional, there are three actions, and both
Q-functions are affine in the features:

```lean (name := rlDqnSetup)
/-- The first observation in the `dqn_replay` example. -/
def rlObsA : Tensor Float [2] := [0.0, 1.0]

/-- The second observation, and the first's next state. -/
def rlObsB : Tensor Float [2] := [1.0, 0.0]

/-- The example's online Q-function. -/
def rlOnlineQ (obs : Tensor Float [2]) : Tensor Float [3] :=
  [obs[0] + 0.2, obs[1] + 1.0, 0.5]

/-- The example's frozen target Q-function. -/
def rlTargetQ (obs : Tensor Float [2]) : Tensor Float [3] :=
  [0.1 + obs[1], 1.4 + obs[0], 0.3]

/-- The non-terminal transition stored by the example. -/
def rlTrA : rl.core.Transition Float [2] 3 :=
  { state := rlObsA, action := 1, reward := 1.0,
    nextState := rlObsB, done := false }

/-- The terminal transition stored by the example. -/
def rlTrB : rl.core.Transition Float [2] 3 :=
  { state := rlObsB, action := 0, reward := 0.5,
    nextState := rlObsA, done := true }
```

Evaluate the two Q-functions at the states that matter for the first transition:

```lean (name := rlDqnQ)
-- Inspect the online action prediction and the target
-- network’s next-state scores.
#eval (rlOnlineQ rlObsA, rlTargetQ rlObsB)
```

```leanOutput rlDqnQ (whitespace := lax)
([0.200000, 2.000000, 0.500000],
  [0.100000, 2.400000, 0.300000])
```

Transition A took action $`1`, so the prediction is
$`Q_\theta(s,1)=2.0`. It is not terminal, so the target bootstraps from the frozen network:
$`y=r+\gamma\max_a Q_{\bar\theta}(s',a)=1.0+0.9\cdot 2.4=3.16`. The TD error is
$`2.0-3.16=-1.16`, so the squared loss is $`1.3456` and the Huber loss with $`\delta=1` is in the
linear regime, $`|e|-\tfrac12=0.66`. Transition B took action $`0` from $`s'=[1,0]`, predicting
$`1.2`, and it is terminal, so the target is just $`r=0.5`; the error is $`0.7`, squared $`0.49`,
Huber $`\tfrac12(0.7)^2=0.245`. Lean agrees with all four:

```lean (name := rlDqnLossA)
-- The nonterminal target includes the maximum next-state
-- target value.
#eval rl.dqn.transitionMSELoss rlOnlineQ rlTargetQ 0.9 rlTrA
```

```leanOutput rlDqnLossA (whitespace := lax)
1.345600
```

```lean (name := rlDqnLossB)
-- The terminal target contains only its immediate reward.
#eval rl.dqn.transitionMSELoss rlOnlineQ rlTargetQ 0.9 rlTrB
```

```leanOutput rlDqnLossB (whitespace := lax)
0.490000
```

```lean (name := rlDqnHuberA)
-- Error magnitude above one enters the linear branch of the
-- Huber loss.
#eval rl.dqn.transitionHuberLoss rlOnlineQ rlTargetQ 0.9 1.0
  rlTrA
```

```leanOutput rlDqnHuberA (whitespace := lax)
0.660000
```

```lean (name := rlDqnHuberB)
-- Error magnitude below one remains in the quadratic
-- branch.
#eval rl.dqn.transitionHuberLoss rlOnlineQ rlTargetQ 0.9 1.0
  rlTrB
```

```leanOutput rlDqnHuberB (whitespace := lax)
0.245000
```

The minibatch losses are means, and because the batch is `[A, B, A, B]` the mean over four equals
the mean over two: $`(1.3456+0.49)/2=0.9178` and $`(0.66+0.245)/2=0.4525`. Those are the two
numbers the example printed:

```lean (name := rlDqnBatch)
-- Repeating both transitions equally preserves their mean
-- squared loss.
#eval rl.dqn.minibatchMSELoss rlOnlineQ rlTargetQ 0.9
  #[rlTrA, rlTrB, rlTrA, rlTrB]
```

```leanOutput rlDqnBatch (whitespace := lax)
0.917800
```

```lean (name := rlDqnBatchHuber)
-- Apply the same repeated batch to the Huber objective.
#eval rl.dqn.minibatchHuberLoss rlOnlineQ rlTargetQ 0.9 1.0
  #[rlTrA, rlTrB, rlTrA, rlTrB]
```

```leanOutput rlDqnBatchHuber (whitespace := lax)
0.452500
```

The `done` flag determines whether the target includes a next-state value. Keeping the reward,
discount, and next-state values fixed isolates its effect:

```lean (name := rlDqnTarget)
-- The largest next-state value contributes when the
-- transition continues.
#eval rl.value.dqnTarget 1.0 0.5 false
  ([2.0, 10.0, 4.0] : Tensor Float [3])
```

```leanOutput rlDqnTarget (whitespace := lax)
6.000000
```

```lean (name := rlDqnTargetDone)
-- With finite inputs, termination removes that bootstrap
-- contribution.
#eval rl.value.dqnTarget 1.0 0.5 true
  ([2.0, 10.0, 4.0] : Tensor Float [3])
```

```leanOutput rlDqnTargetDone (whitespace := lax)
1.000000
```

With $`r=1`, $`\gamma=\tfrac12`, and next-state values $`(2,10,4)`, the non-terminal target is
$`1+\tfrac12\cdot 10=6` and the terminal target is $`1`. For these finite values, `done := true`
removes the bootstrap contribution. The implementation
uses multiplication by zero, not a short-circuit branch: an infinite or NaN next-state value can
still contaminate floating-point arithmetic. The checked numerical helpers reject that case.

The two transitions deliberately fall on opposite sides of the Huber threshold. For A, the
linear branch reduces the influence of a large residual compared with squared error. For B, the
quadratic branch still responds smoothly to a smaller residual. Repeating both transitions twice
preserves their relative weighting, which is why the minibatch mean stays unchanged. Repeating
only one would change the empirical objective even though no new environment information had
been collected.

## Target Networks

The last line of the example is a Polyak average {Informal.citep polyak1964}[], the standard way to
move a target network slowly toward the online one:

$$`\bar\theta\leftarrow\tau\theta+(1-\tau)\bar\theta.`

With $`\tau=\tfrac1{100}`, an online value of $`5` and a target of $`1`:

```lean (name := rlSoft)
-- Move the target one percent of the way toward the online
-- value.
#eval rl.dqn.softUpdateScalar 0.01 5.0 1.0
```

```leanOutput rlSoft (whitespace := lax)
1.040000
```

This example uses online value 5, target value 1, and mixing weight 0.01, giving
$`1+0.01\cdot(5-1)=1.04`. The earlier `dqn_replay` transcript instead uses online value 10,
target value 0, and weight
0.1, giving 1. The real-valued theorem expresses the displacement as a fraction of
the original gap:

```lean (name := rlSoftThm)
-- Over the reals, the displacement equals the mixing weight
-- times the original gap.
open Proofs.RL.DQN Runtime.RL.DQN in
#check @softUpdateScalar_sub_target_real
```

```leanOutput rlSoftThm (whitespace := lax)
softUpdateScalar_sub_target_real : ∀ (tau online target : ℝ),
  softUpdateScalar tau online target - target =
    tau * (online - target)
```

Rewriting the update as `target + tau * (online - target)` is what makes the contraction argument
explicit: for $`0<\tau\le 1` the gap to the online parameters shrinks by a factor of $`1-\tau` per
sync. Stating it as a lemma over $`\mathbb R` rather than over `Float` is deliberate, since the
identity is false in floating point for extreme values, so an executable claim requires the
binary32 treatment discussed earlier in this chapter.

This identity concerns one scalar update. The buffer capacity invariants, terminal TD targets, and
hand-computed losses above give us specific checks on the implementation. A DQN convergence argument
would additionally have to address the replay distribution and the interaction of bootstrapping,
off-policy data, and function approximation; the scalar identity does not supply those results.

The target network serves as a temporarily fixed source of bootstrap values. Its delayed update
prevents the target of every regression step from moving immediately with the online prediction.
The scalar averaging identity explains the synchronization rule when the online value is held
fixed. During learning, that online value also changes, so repeatedly applying the identity does
not by itself give a convergence theorem for the coupled online and target networks.

# Implementation And Proof Coverage

The source directories separate specifications, runtime code, and proofs; the command registry
identifies maintained examples. The table pairs each implementation area with its proof scope.

:::table +header
*
  * Area
  * What exists
  * What is proved
*
  * Returns and GAE
  * returns, GAE, targets
  * common shape; pointwise target addition
*
  * Environments
  * `Spec.RL.Environment`, GridWorld
  * `rollout_length`, step and encoding
*
  * MDP theory
  * deterministic, finite stochastic, measurable
  * contraction, fixed point, bound
*
  * PPO
  * clipped objective, rollout collection
  * shape facts; evaluations are tests
*
  * Binary32
  * checked returns and PPO helpers, intervals
  * finiteness, agreement with the spec
*
  * Trust boundary
  * contracts, checkers, JSON schema
  * checker soundness for `ContractHolds`
*
  * Replay and DQN
  * bounded FIFO buffer, TD targets, losses, Polyak
  * capacity invariants, Polyak over ℝ
:::

The MDP theorems apply to their named mathematical environments, including the sparse-reward
GridWorld. The executable PPO example uses the shaped reward described above. For a deployment,
check the selected environment contract, termination policy, arithmetic, and runtime; the earlier
sections state these contracts beside the corresponding code.

# Contract Exercises

1. Run `torchlean ppo_gridworld --updates 40 --eval-episodes 1 --eval-max-steps 8
   --log /tmp/rl.json --path /tmp/rl-path.json`. Recompute the final return from that path
   and the shaped reward, rather than assuming it follows the one-update path shown above.
2. Change the discount in the `rlGrid` definition above to $`1` and see which hypothesis of
   `toFiniteStochasticMDP_valid` fails, and what the error message says.
3. Evaluate `discountedReturns` at `ℚ` on a reward tensor of your own, then evaluate the same tensor
   at `Float`, and find the shortest trajectory where the two answers disagree once you convert.
4. Write the GAE recursion in NumPy for $`\lambda=0.5` and cross-check three horizons against
   `generalizedAdvantageEstimation`. Then break the shapes on purpose and compare the two error
   messages.
5. Set `requireExclusiveDoneFlags := false` in `rlContract` and rerun the both-flags-true check.
   Then decide which setting you would want for a Gymnasium wrapper you did not write.
6. Change `batchSize := 4` to `batchSize := 16` in the DQN replay example source and predict
   the repeated sample loss before rebuilding and running it; the command has no batch-size flag.
7. Feed `ofFloatChecked` the largest binary64 value that still converts, then the next one up, and
   read both messages.
8. Run `ppo_cartpole` with `--eval-max-steps 8` and then with `--eval-max-steps 500`, and compare
   how the cap limits what each return curve can establish.

# Related Verification Chapters

{ref "verification"}[Verification Overview] explains how executable checks connect to mathematical
claims. {ref "scientific-ml-verification"}[Scientific ML Verification] examines the corresponding
obligations for PDE surrogates, including the meaning of a measured residual.
