# NN/Runtime

`NN/Runtime` is TorchLean's executable layer. This is where typed tensor specs become runnable
programs, autograd tapes execute, optimizers update parameter lists, and optional bridges connect
TorchLean to CUDA, PyTorch, Julia, and Gymnasium style environments.

Most downstream code should not import modules from this directory directly. Prefer:

* `import NN` for the ordinary public surface,
* `import NN.API.Runtime` for the stable runtime facade, or
* `import NN.Entrypoint.Runtime` when you intentionally want the broad executable umbrella.

There is deliberately no top level `NN.Runtime` Lean file. The project keeps public import
surfaces under `NN.API.*` and `NN.Entrypoint.*` so that subsystem directories can contain
implementation modules without creating competing umbrella names.

## Directory Map

* `Autograd/Engine`: the small eager reverse-mode tape and runtime tape monad.
* `Autograd/Compiled`: typed IR/DAG execution that lowers into the same tape machinery.
* `Autograd/Torch`: lower level imperative session operations and linked compiled sessions.
* `Autograd/TorchLean`: the user facing runtime front end used by `NN.API.Runtime`.
* `Autograd/Train`: deterministic datasets, loaders, trainers, and optimizer integration.
* `Optim`: pure optimizer equations and scheduler utilities.
* `PyTorch`: import/export bridges for round trip checks against Python `torch.nn.Module`s.
* `RL`: typed reinforcement learning runtimes, Gymnasium bridges, PPO helpers, and rollout
  boundary checks.
* `External`: small process/Julia helpers used by executable integrations.
* `Training`: lightweight training-log records shared by runtime examples and widgets.

## Trust Boundaries

The Lean runtime specifies and executes TorchLean's own tensor and tape semantics. Native CUDA,
PyTorch, Julia, and Gymnasium integrations are external trust boundaries: their contracts, shape
checks, and regression tests live in Lean facing modules, but foreign C, CUDA, Python, and Julia code
is not proved correct inside Lean. Formal correctness results live in `NN.Proofs.*`; this directory
supplies the executable objects those theorems connect to.
