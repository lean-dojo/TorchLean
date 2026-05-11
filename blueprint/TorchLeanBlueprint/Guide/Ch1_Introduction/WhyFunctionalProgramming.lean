import VersoManual

open Verso.Genre Manual

#doc (Manual) "Why Functional Programming?" =>
%%%
tag := "why_functional"
%%%

The reason TorchLean uses a functional style is not aesthetic. It is about making state visible.

In ordinary ML code, a forward pass may read parameters, update normalization buffers, consult a
random generator, depend on train/eval mode, and leave gradients in mutable fields. That is
convenient, but it complicates the question a verifier eventually has to ask: which function did
this model compute?

Functional programming gives us a clean way to answer. A layer is a function from inputs and
parameters to outputs. A training step is a function from old parameters, data, gradients, optimizer
state, and random generator state to new values. Nothing essential is lost; the difference is that
every value that matters to a theorem has a name.

# The Problem with Mutable State

In an ML script, state is everywhere. BatchNorm has running statistics. Dropout depends on mode and
randomness. Optimizers carry momentum or Adam moments. Autoregressive models carry KV caches and
position counters. Tokenizers carry vocabularies and special-token conventions. Parameters can be
shared. None of these are bad. They are how practical systems work.

The problem is that a theorem cannot reason about state that never appears in the object being
checked. In many frameworks, a model is an object with mutable fields. A call that looks like an
ordinary forward pass may update a buffer, consult a hidden random generator, write to gradient
storage, or depend on whether some parameter tensor is shared with another module. That style is
convenient for experimentation, but it makes the mathematical question less direct: which function
did the network compute?

A Python-style sketch makes the issue concrete:

```
class Layer:
    def __call__(self, x):
        self.calls += 1          # hidden state change
        self.running_mean *= 0.9 # another hidden state change
        return self.weight * x + self.bias
```

The return value is incomplete evidence. The next call can behave differently because this call
changed the object. If a proof, exporter, or verifier ignores those mutations, it may reason about a
different computation from the one that actually ran.

# Pure Functions are Mathematical Functions

In a pure functional language such as Lean, ordinary functions have no side effects. A TorchLean
layer takes explicit inputs, including its parameters, and returns an explicit output. The simplest
version is just affine arithmetic:

```
structure Affine1D where
  w : Float
  b : Float

def affine1D (p : Affine1D) (x : Float) : Float :=
  p.w * x + p.b
```

Here the mathematical reading and the executable reading coincide: `affine1D p x` computes
`p.w * x + p.b`. There is no hidden `.grad` field, no object identity, and no accidental parameter
mutation that a theorem has to account for later.

The same idea scales to tensors. In TorchLean, a layer is still read as

$$`\operatorname{forward}(\theta, x) = y`

but the values now carry tensor shapes, scalar semantics, and graph structure as needed. We can
prove facts about the same definitions that examples and checkers inspect.

# Training Still Changes Things

Functional programming does not mean that training is static. It means that change is represented by
new values instead of silent updates to existing ones.

```
def sgdStep (eta gradW gradB : Float) (p : Affine1D) : Affine1D :=
  { w := p.w - eta * gradW
    b := p.b - eta * gradB }
```

This is still an update, but it is an update with a type and a result. An optimizer step takes an
old parameter bundle and returns a new parameter bundle. A logger takes an old log state and returns
an updated log state. A random generator takes an old seed or generator state and returns the next
one. The training loop remains inspectable because state changes appear at the places where the
program says they happen.

This also clarifies trust boundaries. If a CUDA kernel, a PyTorch exporter, or an external
certificate producer contributes a value, TorchLean can name that imported value and state what is
assumed about it. The proof does not have to pretend that an external side effect was a Lean
definition.

# Reference Counting And Practical Execution

The usual worry about pure code is that it allocates too much. Lean 4 uses deterministic reference
counting, and values with a unique owner can often be updated in place under the hood. That means
the functional surface does not require the runtime to behave naively.

For TorchLean, this matters because tensor code needs both a clean semantics and a realistic path to
performance. We can write programs as transformations of values, while the runtime is still allowed
to perform safe buffer reuse when uniqueness makes it possible.

# Related Design Ideas

The same design principle appears throughout formal methods: keep executable code, specifications,
and proof obligations close enough that they stay aligned. TorchLean applies that principle to
neural networks. State is data. Shapes are part of the interface. Semantics are named. Proof
artifacts are built around the same definitions that model examples use.

Functional programming is useful here because it turns hidden context into data. Parameters,
optimizer state, random seeds, logs, masks, imported artifacts, and certificate payloads can all be
passed, saved, inspected, and mentioned in theorem statements. TorchLean does not remove state from
ML. It makes the state part of the computation we can talk about.

## References

- Lean 4 documentation: https://lean-lang.org/doc/reference/latest/
- Ullrich and de Moura, "Counting Immutable Beans: Reference Counting Optimized for Purely
  Functional Programming", IFL 2019.
- George et al., "BRIDGE: Building Representations In Domain Guided Program Synthesis",
  arXiv:2511.21104.
