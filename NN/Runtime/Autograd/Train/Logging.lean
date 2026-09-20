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
  /-- Message payload written to the training log. -/
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

/-- A simple `IO` logger that prints to stdout. -/
def stdout : Logger IO :=
  { log := fun lvl msg => IO.println s!"[{LogLevel.render lvl}] {msg}" }

/-- Emit an informational log message. -/
def info {m : Type -> Type} (logger : Logger m) (msg : String) : m Unit :=
  logger.log .info msg

/-- Emit a warning log message. -/
def warn {m : Type -> Type} (logger : Logger m) (msg : String) : m Unit :=
  logger.log .warn msg

/-- Emit an error log message. -/
def error {m : Type -> Type} (logger : Logger m) (msg : String) : m Unit :=
  logger.log .error msg

end Logger

/-!
## Trainer integration
-/
namespace Trainer

end Trainer

end Train
end Autograd
end Runtime
