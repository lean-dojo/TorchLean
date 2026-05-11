#!/usr/bin/env python3
"""Generate compact LiRPA/PINN certificates and optionally check them in Lean.

Python acts as the certificate producer. Lean remains the checker: this runner
can write JSON artifacts and then invoke `lake exe verify` for the selected
workflow.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Wrapper that computes the IBP cert in Python and optionally verifies in Lean.
# Supports multiple certificate workflows via --model: transformer (default), mlp, attention, cnn, gru, pinn.

HERE = Path(__file__).resolve().parent

def find_repo_root(start: Path) -> Path:
    """Walk upward from `start` until a Lake project root is found."""
    cur = start
    for _ in range(8):  # search up to 8 levels just in case
        if (cur / "lakefile.lean").exists() or (cur / "lakefile.toml").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start

REPO_ROOT = find_repo_root(HERE)
EXAMPLES_DIR = REPO_ROOT / "NN" / "Examples" / "Verification"

def load_exporter(pyfile: str):
    """Dynamically import one local exporter module by filename or absolute path."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            Path(pyfile).stem, (HERE / pyfile).as_posix()
        )
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        print(f"Error importing {pyfile}: {e}", file=sys.stderr)
        sys.exit(1)


class Model:
    """Workflow wrapper that can generate a Python IBP certificate and verify it in Lean."""

    def __init__(self, model: str = "transformer", cert_path: Path | None = None):
        """Select the exporter, default certificate path, and Lean verifier command."""
        self.model = model
        # Select exporter and default cert path
        if model == "transformer":
            self.exporter = load_exporter("export_crown_cert.py")
            default_cert = EXAMPLES_DIR / "LiRPA" / "transformer_encoder_cert.json"
            self.expected_cert = default_cert
            self.lean_tool = "lirpa-encoder"
        elif model == "mlp":
            self.exporter = load_exporter("export_mlp_cert.py")
            default_cert = EXAMPLES_DIR / "LiRPA" / "mlp_cert.json"
            self.expected_cert = default_cert
            self.lean_tool = "lirpa-mlp"
        elif model == "attention":
            self.exporter = load_exporter("export_attention_cert.py")
            default_cert = EXAMPLES_DIR / "LiRPA" / "attention_softmax_cert.json"
            self.expected_cert = default_cert
            self.lean_tool = "lirpa-attention"
        elif model == "cnn":
            self.exporter = load_exporter("export_cnn_cert.py")
            default_cert = EXAMPLES_DIR / "LiRPA" / "cnn_cert.json"
            self.expected_cert = default_cert
            self.lean_tool = "lirpa-cnn"
        elif model == "gru":
            self.exporter = load_exporter("export_gru_cert.py")
            default_cert = EXAMPLES_DIR / "LiRPA" / "gru_gate_cert.json"
            self.expected_cert = default_cert
            self.lean_tool = "lirpa-gru"
        elif model == "pinn":
            # The PINN exporter lives in the sibling `scripts/verification/pinn` directory.
            self.exporter = load_exporter((REPO_ROOT / "scripts" / "verification" / "pinn" / "export_pinn_cert.py").as_posix())
            default_cert = (EXAMPLES_DIR / "PINN" / "pinn_cert.json").resolve()
            self.expected_cert = default_cert
            self.lean_tool = "pinn-cert"
        else:
            print(f"Unknown model kind: {model}", file=sys.stderr)
            sys.exit(2)
        self.cert_path = Path(cert_path) if cert_path else default_cert

    def run_python_ibp(self) -> dict:
        """Run the selected Python exporter and write the certificate JSON."""
        cert = self.exporter.run_ibp()
        # Write to user-specified path
        self.cert_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cert_path, "w") as f:
            json.dump(cert, f, indent=2)
        # Also write to the model-expected path for Lean verifiers
        if self.cert_path.resolve() != self.expected_cert.resolve():
            self.expected_cert.parent.mkdir(parents=True, exist_ok=True)
            with open(self.expected_cert, "w") as f:
                json.dump(cert, f, indent=2)
        return cert

    def verify_in_lean(self, quiet: bool = False) -> int:
        """Build the Lean verifier and check the generated certificate."""
        cwd = REPO_ROOT.as_posix()
        build_cmd = ["lake", "build", "verify"]
        run_cmd = ["lake", "exe", "verify", "--", self.lean_tool]
        try:
            if not quiet:
                print(f"[lean] Building verifier (verify) in {cwd} ...")
            subprocess.run(
                build_cmd,
                cwd=cwd,
                check=True,
                stdout=subprocess.PIPE if quiet else None,
                stderr=subprocess.STDOUT,
            )
            if not quiet:
                print(f"[lean] Running verifier tool: {self.lean_tool} ...")
            proc = subprocess.run(run_cmd, cwd=cwd, check=False, capture_output=True, text=True)
            if not quiet:
                print(proc.stdout, end="")
                if proc.stderr:
                    print(proc.stderr, file=sys.stderr, end="")
            if proc.returncode != 0:
                return proc.returncode
            output = proc.stdout + proc.stderr
            # The verifier executable exits successfully for some certificate
            # rejections, so this wrapper keys off the exact rejection phrases
            # emitted by the Lean verifier rather than broad words like "error".
            mismatch_markers = (
                "IBP certificate mismatch",
                "certificate mismatch",
            )
            if any(marker in output for marker in mismatch_markers):
                return 1
            return proc.returncode
        except subprocess.CalledProcessError as e:
            if not quiet and e.stdout:
                print(e.stdout, end="")
            if e.stderr:
                print(e.stderr, file=sys.stderr, end="")
            return e.returncode or 1


def main():
    """CLI entry point for certificate generation and optional Lean checking."""
    p = argparse.ArgumentParser(description="Run Python IBP and optionally verify in Lean.")
    p.add_argument(
        "--model",
        choices=["transformer", "mlp", "attention", "cnn", "gru", "pinn"],
        default="transformer",
        help="Select which certificate workflow to run",
    )
    p.add_argument(
        "--cert",
        type=str,
        default=None,
        help="Optional path to write the certificate JSON; defaults to the model’s expected path",
    )
    p.add_argument("--lean", action="store_true", help="Also build and run the Lean verifier")
    p.add_argument("--quiet", action="store_true", help="Reduce Lean output verbosity")
    args = p.parse_args()

    model = Model(model=args.model, cert_path=Path(args.cert) if args.cert else None)
    cert = model.run_python_ibp()
    print(f"[python] Wrote certificate to {model.cert_path}")

    if args.lean:
        code = model.verify_in_lean(quiet=args.quiet)
        sys.exit(code)


if __name__ == "__main__":
    main()
