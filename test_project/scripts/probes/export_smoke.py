#!/usr/bin/env python3
"""Isolated Godot 4.4.1 Windows debug export gate for bundled neural snapshot (#50)."""
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


def run(args, cwd):
    p = subprocess.run([str(a) for a in args], cwd=cwd, capture_output=True, text=True, timeout=90)
    if p.returncode or "ERROR:" in p.stdout or "SCRIPT ERROR:" in p.stdout:
        raise RuntimeError(f"{' '.join(map(str, args))}\nexit={p.returncode}\n{p.stdout[-3500:]}\n{p.stderr[-3500:]}")
    return p.stdout + p.stderr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--godot", type=Path, required=True, help="Godot 4.4.1 console editor")
    parser.add_argument("--debug-template", type=Path, required=True, help="Windows 4.4.1 debug export template exe")
    parser.add_argument("--dll", type=Path, required=True, help="freshly built Windows debug MLGodotKit DLL")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    descriptor = root / "test_project" / "addons" / "mlgodotkit" / "mlgodotkit.gdextension"
    scripts = Path(__file__).parent
    for path in (args.godot, args.debug_template, args.dll, descriptor):
        if not path.is_file():
            parser.error(f"Missing: {path}")
    with tempfile.TemporaryDirectory(prefix="mlgodotkit-neural-export-") as scratch:
        project = Path(scratch)
        addon = project / "addons" / "mlgodotkit"
        (addon / "bin").mkdir(parents=True)
        shutil.copy2(descriptor, addon)
        shutil.copy2(args.dll, addon / "bin" / "mlgodotkit.windows.template_debug.x86_64.dll")
        assets = project / "scripts" / "probes"
        assets.mkdir(parents=True)
        for name in ("fixture_small_policy.json", "exported_snapshot_probe.gd",
                     "neural_snapshot_probe.gd", "shared_combat_policy_probe.gd"):
            shutil.copy2(scripts / name, assets / name)
        (project / "main.tscn").write_text(
            '[gd_scene load_steps=2 format=3]\n[ext_resource type="Script" path="res://scripts/probes/exported_snapshot_probe.gd" id="1"]\n[node name="SnapshotExportProbe" type="Node"]\nscript = ExtResource("1")\n', encoding="utf-8")
        (project / "project.godot").write_text(
            '[application]\nconfig/name="MLGodotKit isolated snapshot export"\nrun/main_scene="res://main.tscn"\n', encoding="utf-8")
        template = args.debug_template.resolve().as_posix()
        (project / "export_presets.cfg").write_text(
            '[preset.0]\nname="Windows Desktop"\nplatform="Windows Desktop"\nrunnable=true\n'
            'dedicated_server=false\ncustom_features=""\nexport_filter="all_resources"\n'
            'include_filter=""\nexclude_filter=""\nexport_path=""\n'
            'encrypt_pck=false\nencrypt_directory=false\n[preset.0.options]\n'
            f'custom_template/debug="{template}"\ncustom_template/release="{template}"\n'
            'binary_format/architecture="x86_64"\napplication/modify_resources=false\n', encoding="utf-8")
        godot = args.godot.resolve()
        run([godot, "--headless", "--path", project, "--editor", "--import"], project)
        snapshot = run([godot, "--headless", "--path", project, "--script",
                        "res://scripts/probes/neural_snapshot_probe.gd"], project)
        if "NEURAL_SNAPSHOT_PROBE PASSED" not in snapshot:
            raise RuntimeError(f"Snapshot gate failed: {snapshot[-3500:]}")
        performance = run([godot, "--headless", "--path", project, "--script",
                           "res://scripts/probes/shared_combat_policy_probe.gd"], project)
        if "SHARED_POLICY_PROBE" not in performance:
            raise RuntimeError(f"Inference gate failed: {performance[-3500:]}")
        print(next(line for line in performance.splitlines() if "SHARED_POLICY_PROBE" in line))
        output = project / "out"
        output.mkdir()
        game = output / "neural-export-probe.exe"
        run([godot, "--headless", "--path", project, "--export-debug", "Windows Desktop", game], project)
        report = run([game, "--headless"], output)
        if "NEURAL_EXPORT_PROBE_PASSED" not in report:
            raise RuntimeError(f"Exported game did not load fixture: {report[-3500:]}")
        print("NEURAL_EXPORT_PROBE_PASSED (isolated debug PCK, bundled JSON + DLL)")


if __name__ == "__main__":
    main()
