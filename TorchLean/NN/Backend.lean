/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# Backend Types

The vocabulary of backend selection and trust boundaries.

TorchLean owns the spec, the graph and the contracts that proofs refer to. Backends are execution
providers for parts of that graph: a Lean reference path, a TorchLean composition or LibTorch ATen
operations. The module holds data alone, so specs, runtime wrappers, docs generators and tests can
import it cheaply.
-/

@[expose] public section

namespace TorchLean
namespace Backend

/-- Hardware or execution target visible to the planner. -/
inductive Device where
  | cpu
  | cuda
  deriving DecidableEq, Repr

namespace Device

/-- Stable spelling used in profile names, reports, and CLI bridges. -/
def cliName : Device → String
  | .cpu => "cpu"
  | .cuda => "cuda"

/-- The device names `parse` accepts, separated by `|`, for help texts. -/
def help : String := "cpu|gpu|cuda"

/-- Parse a backend device name, failing with a diagnostic that lists the names in `help`. `gpu`
parses to `cuda` as well, and CLI layers resolve policy names such as `auto` before calling
`parse`. -/
def parse : String → Except String Device
  | "cpu" => pure .cpu
  | "cuda" => pure .cuda
  | "gpu" => pure .cuda
  | value => throw s!"unknown device {value} (known devices: {help})"

/-- `parse` reads back every `cliName`. -/
theorem parse_cliName (d : Device) : parse d.cliName = .ok d := by
  cases d <;> rfl

end Device

/-- Concrete provider family used to execute a kernel capsule. -/
inductive Provider where
  | reference
  | torchLean
  /-- LibTorch ATen execution, which does not imply LibTorch autograd ownership. -/
  | libTorch
  deriving DecidableEq, Repr

/--
How much TorchLean knows about an implementation.

`checked` records maintained runtime guards and regression evidence, wrappers around LibTorch
included, which test an implementation without proving it. `trustedExternal` records a boundary
accepted without that evidence, which only an assurance policy with `allowTrustedExternal` admits.
-/
inductive TrustLevel where
  | checked
  | trustedExternal
  deriving DecidableEq, Repr

/--
Which trust boundaries a kernel plan may cross.

One record decides which capsules the planner may select (by trust level) and which contract
evidence the selected capsules may rely on (by evidence kind), so a profile cannot select a capsule
under one policy and check it under another.
-/
structure AssurancePolicy where
  /-- Whether capsules and evidence that delegate to an external implementation are admitted. -/
  allowTrustedExternal : Bool := false
  deriving DecidableEq, Repr

namespace AssurancePolicy

/--
The maintained TorchLean runtime policy.

It accepts implementations backed by maintained runtime guards and regression evidence, LibTorch
wrappers with recorded CUDA comparisons included, and rejects `trustedExternal` capsules and named
trusted-boundary evidence. It does not verify LibTorch, the FFI or compiled code.
-/
def checked : AssurancePolicy := {}

/--
The external-provider policy.

Besides what `checked` accepts, it admits capsules and contract descriptors that rest on a recorded
assumption that an external implementation agrees with its contract. Every boundary selected under
it appears in the execution audit.
-/
def external : AssurancePolicy :=
  { allowTrustedExternal := true }

/-- Whether the policy admits a capsule with the given implementation trust level. -/
def acceptsTrust (policy : AssurancePolicy) : TrustLevel → Bool
  | .checked => true
  | .trustedExternal => policy.allowTrustedExternal

end AssurancePolicy

/-- How a backend capsule supplies a local reverse rule to the TorchLean tape. -/
inductive VJPMode where
  /-- No reverse rule is supplied. -/
  | none
  /-- TorchLean composes the local VJP from runtime operations. -/
  | torchLeanTape
  /-- The native bridge supplies the selected local VJP, without recording an autograd graph. -/
  | backendVJP
  deriving DecidableEq, Repr

/-- Provider preference used when selecting an implementation for an operation. -/
inductive Provider.Preference where
  | auto
  | prefer (provider : Provider)
  | only (provider : Provider)
  deriving DecidableEq, Repr

/-- Policy used to select kernel capsules for a device and assurance boundary. -/
structure KernelPolicy where
  /-- Device every selected capsule targets. -/
  device : Device := .cpu
  /-- The provider selection prefers (`prefer`) or is restricted to (`only`), if any. -/
  provider : Provider.Preference := .auto
  /-- Trust boundaries the selected capsules and their evidence may cross. -/
  assurance : AssurancePolicy := .checked
  /-- Gradient mode the selected capsules must support, `none` for inference. -/
  vjpMode : VJPMode := .torchLeanTape
  deriving Repr

end Backend
end TorchLean
