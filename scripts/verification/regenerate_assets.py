#!/usr/bin/env python3
"""Regenerate TorchLean verification artifacts.

This script is a small command catalog, not a verifier.  It exists so users can see which JSON
artifacts are source fixtures and which ones come from external producer scripts.

Default behavior is deliberately safe: commands are printed, not executed.  Pass `--run` to execute
one group locally.

Examples:
  python3 scripts/verification/regenerate_assets.py --list
  python3 scripts/verification/regenerate_assets.py --group digits --run
  python3 scripts/verification/regenerate_assets.py --group lirpa --run

The generated files are still untrusted.  After regeneration, run the Lean checker:
  lake exe verify -- all
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts" / "verification"


@dataclass(frozen=True)
class Command:
    """One regeneration command.

    `group` is the user-facing selector. `kind` distinguishes small bundled fixtures from optional
    benchmark/data workflows.  Keeping that distinction visible helps prevent accidentally treating
    generated JSON as hand-authored source.
    """

    group: str
    kind: str
    description: str
    argv: tuple[str, ...]

    def shell(self) -> str:
        """Render the command as a copy-pasteable shell line."""
        return " ".join(shlex.quote(x) for x in self.argv)


COMMANDS: tuple[Command, ...] = (
    Command(
        group="digits",
        kind="bundled fixture",
        description="Train/export sklearn-digits linear weights and test split.",
        argv=(
            "python3",
            str(SCRIPTS_DIR / "robustness" / "train_digits_linear.py"),
        ),
    ),
    Command(
        group="digits",
        kind="bundled fixture",
        description="Export the per-example logit-margin certificate for the digits workflow.",
        argv=(
            "python3",
            str(SCRIPTS_DIR / "robustness" / "export_margin_cert.py"),
        ),
    ),
    Command(
        group="lirpa",
        kind="bundled fixture",
        description="Regenerate the default compact LiRPA JSON certificate and check it in Lean.",
        argv=(
            "python3",
            str(SCRIPTS_DIR / "lirpa" / "cert_runner.py"),
            "--lean",
        ),
    ),
    Command(
        group="spline",
        kind="bundled fixture",
        description="Regenerate the small spline certificate through Julia and check it in Lean.",
        argv=(
            "lake",
            "exe",
            "verify",
            "--",
            "spline-cert",
            "--regen",
        ),
    ),
    Command(
        group="pinn-small",
        kind="bundled fixture",
        description="Regenerate the compact 1D PINN certificate used by the fast verifier suite.",
        argv=(
            "python3",
            str(SCRIPTS_DIR / "pinn" / "export_pinn_cert.py"),
        ),
    ),
    Command(
        group="pinn-train",
        kind="local output",
        description="Train a compact 1D PINN checkpoint/weights JSON under the ignored checkpoints directory.",
        argv=(
            "python3",
            str(SCRIPTS_DIR / "pinn" / "train_pinn_1d.py"),
            "--steps",
            "25",
            "--out-ckpt",
            "_external/pinn/checkpoints/pinn1d.pt",
            "--out-json",
            "_external/pinn/checkpoints/pinn1d.json",
        ),
    ),
    Command(
        group="pinn-train",
        kind="local output",
        description="Train a compact 2D PINN checkpoint/weights JSON under the ignored checkpoints directory.",
        argv=(
            "python3",
            str(SCRIPTS_DIR / "pinn" / "train_pinn_2d.py"),
            "--steps",
            "25",
            "--out-ckpt",
            "_external/pinn/checkpoints/pinn2d.pt",
            "--out-json",
            "_external/pinn/checkpoints/pinn2d.json",
        ),
    ),
    Command(
        group="ode",
        kind="bundled fixture",
        description="Check the curated ODE certificate fixture. ODE JSONs are hand-curated compact fixtures.",
        argv=(
            "lake",
            "exe",
            "verify",
            "--",
            "ode",
            "--cert=NN/Examples/Verification/ODE/sample_ode_cert.json",
        ),
    ),
    Command(
        group="geometry3d-real",
        kind="local output",
        description="Run DETR + Depth Anything V2 on a real image, export a 3D-box cert, and check it in Lean.",
        argv=(
            "python3",
            str(SCRIPTS_DIR / "geometry3d" / "export_hf_depth_box3d_cert.py"),
            "--batch-manifest",
            str(SCRIPTS_DIR / "geometry3d" / "realworld_manifest.json"),
            "--batch-out-dir",
            "_external/geometry3d/realworld",
            "--verify",
        ),
    ),
    Command(
        group="geometry3d-real",
        kind="local output",
        description="Render PNG overlays for the real-image Geometry3D certificates.",
        argv=(
            "python3",
            str(SCRIPTS_DIR / "geometry3d" / "render_box3d_cert_overlay.py"),
            "--glob",
            "_external/geometry3d/realworld/*.json",
            "--out-dir",
            "_external/geometry3d/overlays/realworld",
            "--contact-sheet",
        ),
    ),
    Command(
        group="geometry3d-wilddet3d",
        kind="local output",
        description="Run WildDet3D, export a 3D-box certificate, check it, and render an overlay.",
        argv=(
            "python3",
            str(SCRIPTS_DIR / "geometry3d" / "export_wilddet3d_box3d_cert.py"),
            "--text-prompt",
            "cat",
            "--out",
            "_external/geometry3d/wilddet3d/wilddet3d_cat_box3d_cert.json",
            "--verify",
            "--overlay",
        ),
    ),
    Command(
        group="geometry3d-wilddet3d",
        kind="diagnostic output",
        description="Export WildDet3D's strict model-2D-bbox claim so the overlay shows the 2D/3D mismatch Lean rejects.",
        argv=(
            "python3",
            str(SCRIPTS_DIR / "geometry3d" / "export_wilddet3d_box3d_cert.py"),
            "--text-prompt",
            "cat",
            "--bbox-source",
            "model2d",
            "--out",
            "_external/geometry3d/wilddet3d/wilddet3d_cat_model2d_strict_box3d_cert.json",
            "--overlay",
        ),
    ),
    Command(
        group="geometry3d-wilddet3d",
        kind="visual output",
        description="Render a WildDet3D contact sheet with the accepted projected-envelope cert and rejected strict-bbox cert.",
        argv=(
            "python3",
            str(SCRIPTS_DIR / "geometry3d" / "render_box3d_cert_overlay.py"),
            "--glob",
            "_external/geometry3d/wilddet3d/*.json",
            "--out-dir",
            "_external/geometry3d/wilddet3d",
            "--contact-sheet",
        ),
    ),
    Command(
        group="geometry3d-wilddet3d",
        kind="visual output",
        description="Plot the WildDet3D model-2D-bbox interval against the projected 3D-footprint interval.",
        argv=(
            "python3",
            str(SCRIPTS_DIR / "geometry3d" / "plot_box3d_bbox_diagnostic.py"),
            "--cert",
            "_external/geometry3d/wilddet3d/wilddet3d_cat_box3d_cert.json",
            "--out",
            "_external/geometry3d/wilddet3d/wilddet3d_bbox_diagnostic.png",
        ),
    ),
    Command(
        group="geometry3d-bad",
        kind="regression test",
        description="Generate broken 3D-box artifacts and require the Lean checker to reject them.",
        argv=(
            "python3",
            str(SCRIPTS_DIR / "geometry3d" / "test_bad_box3d_certs.py"),
        ),
    ),
    Command(
        group="geometry3d-visual",
        kind="regression demo",
        description="Generate broken Geometry3D artifacts, then render good/bad visual overlays and a contact sheet.",
        argv=(
            "python3",
            str(SCRIPTS_DIR / "geometry3d" / "test_bad_box3d_certs.py"),
        ),
    ),
    Command(
        group="geometry3d-visual",
        kind="regression demo",
        description="Render PNG overlays for the bundled good fixture and deliberately invalid certificates.",
        argv=(
            "python3",
            str(SCRIPTS_DIR / "geometry3d" / "render_box3d_cert_overlay.py"),
            "--include-default",
            "--glob",
            "_external/geometry3d/bad/*.json",
            "--out-dir",
            "_external/geometry3d/overlays/bugzoo",
            "--contact-sheet",
        ),
    ),
    Command(
        group="two-stage",
        kind="local output",
        description="Train a stage-1 Van controller/Lyapunov seed and export exact float32 bits.",
        argv=(
            "python3",
            str(SCRIPTS_DIR / "two_stage" / "export_van_stage1_bits.py"),
            "--out",
            "_external/van_stage1_w100_bits.json",
        ),
    ),
    Command(
        group="two-stage",
        kind="local output",
        description="Run the Python stage-2 baseline against the stage-1 bit export.",
        argv=(
            "python3",
            str(SCRIPTS_DIR / "two_stage" / "cegis_van_stage2_python_baseline.py"),
            "--weights",
            "_external/van_stage1_w100_bits.json",
        ),
    ),
)


def selected_commands(group: str) -> list[Command]:
    """Return commands matching a named regeneration group."""
    if group == "all-light":
        groups = {"digits", "lirpa", "pinn-small"}
        return [cmd for cmd in COMMANDS if cmd.group in groups]
    return [cmd for cmd in COMMANDS if cmd.group == group]


def main() -> None:
    """List or run selected verification artifact regeneration commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        default="all-light",
        choices=[
            "all-light",
            "digits",
            "lirpa",
            "spline",
            "pinn-small",
            "pinn-train",
            "ode",
            "geometry3d-real",
            "geometry3d-wilddet3d",
            "geometry3d-bad",
            "geometry3d-visual",
            "two-stage",
        ],
        help="Artifact group to regenerate.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute commands. Without this flag, only print the commands.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all known regeneration commands and exit.",
    )
    args = parser.parse_args()

    commands = COMMANDS if args.list else tuple(selected_commands(args.group))
    if not commands:
        raise SystemExit(f"No commands registered for group {args.group!r}")

    for cmd in commands:
        print(f"\n[{cmd.group}] {cmd.kind}: {cmd.description}", flush=True)
        print(f"$ {cmd.shell()}", flush=True)
        if args.run and not args.list:
            subprocess.run(cmd.argv, cwd=ROOT, check=True)

    if not args.run:
        print("\nDry run only. Re-run with --run to execute the selected group.")


if __name__ == "__main__":
    main()
