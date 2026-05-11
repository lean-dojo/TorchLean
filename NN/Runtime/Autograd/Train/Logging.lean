/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Train.Trainer

/-!
# Logging helpers for training loops

This module defines a small, pluggable logging interface used by the training utilities.
The core interface is monad-polymorphic (so it can stay pure), and `Logger.stdout` provides a
convenient `IO` implementation.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace Train

/-!
## Log levels and entries
-/
/-- Severity of a log message emitted during training/evaluation. -/
inductive LogLevel where
  | debug
  | info
  | warn
  | error
  deriving Repr, DecidableEq

/-- Render a `LogLevel` as a lower-case string (used by `LogEntry.render`). -/
def LogLevel.render : LogLevel -> String
  | .debug => "debug"
  | .info => "info"
  | .warn => "warn"
  | .error => "error"

/-- A structured log message (level + string payload). -/
structure LogEntry where
  /-- Severity level for filtering or rendering. -/
  level : LogLevel
  /-- Human-readable message payload. -/
  message : String

/-- Render a `LogEntry` as `[level] msg`. -/
def LogEntry.render (e : LogEntry) : String :=
  s!"[{LogLevel.render e.level}] {e.message}"

/-!
## Logger interface
-/
/--
A small pluggable logger interface used by training utilities.

The interface is compact: a logger consumes a level + string and can live in any monad `m`
(including pure test monads). See `Logger.stdout` for a basic `IO` implementation.
-/
structure Logger (m : Type -> Type) where
  /-- Emit one message at the requested severity. -/
  log : LogLevel -> String -> m Unit

namespace Logger

/-- A logger that discards every message. Useful as a default. -/
def noOp {m : Type -> Type} [Monad m] : Logger m :=
  { log := fun _ _ => pure () }

/-- A simple `IO` logger that prints to stdout. -/
def stdout : Logger IO :=
  { log := fun lvl msg => IO.println s!"[{LogLevel.render lvl}] {msg}" }

/-- Log a pre-built `LogEntry`. -/
def logEntry {m : Type -> Type} (logger : Logger m) (entry : LogEntry) : m Unit :=
  logger.log entry.level entry.message

/-- Convenience wrapper for an `.info` message. -/
def info {m : Type -> Type} (logger : Logger m) (msg : String) : m Unit :=
  logger.log .info msg

/-- Convenience wrapper for a `.warn` message. -/
def warn {m : Type -> Type} (logger : Logger m) (msg : String) : m Unit :=
  logger.log .warn msg

/-- Convenience wrapper for an `.error` message. -/
def error {m : Type -> Type} (logger : Logger m) (msg : String) : m Unit :=
  logger.log .error msg

end Logger

/-!
## Trainer integration
-/
namespace Trainer

/-- Attach an arbitrary logger hook to a `Trainer` (called once per step). -/
def withLogger {m : Type -> Type} {state a : Type}
  (t : Trainer m state a) (logger : Nat -> state -> StepReport a -> m Unit) :
  Trainer m state a :=
  { t with logger := logger }

/--
Attach a `Logger` to a `Trainer` by logging the pretty-printed `StepReport` each step.

This is analogous to printing per-step metrics in an imperative training script (e.g. a PyTorch
  loop).
-/
def withReportLogger {m : Type -> Type} [Monad m] {state a : Type} [ToString a]
  (t : Trainer m state a) (logger : Logger m) : Trainer m state a :=
  { t with logger := fun step _ report =>
      logger.log .info (renderReport step report) }

end Trainer

end Train
end Autograd
end Runtime
