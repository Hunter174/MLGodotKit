# Shared-policy and pretrained-artifact gates (issue #50)

Build the native debug extension first, then run isolated Windows/Godot 4.4.1 import, snapshot roundtrip, headless inference and **actual debug export with a bundled JSON fixture** without altering `test_project/addons/` or other existing local changes:

```sh
cd mlgodotkit
scons platform=windows target=template_debug eigen_path=C:/libs/eigen-3.4.0 -j4
cd ..
python test_project/scripts/probes/export_smoke.py \
  --godot C:/path/to/Godot_v4.4.1-stable_win64_console.exe \
  --debug-template C:/path/to/windows_debug_x86_64.exe \
  --dll mlgodotkit/bin/mlgodotkit.windows.template_debug.x86_64.dll
```

This script constructs one 16→32→8 `NeuralNetworkNode`, copies its weights into a separate identical node, checks equal output, output shape and non-mutating inference, then reports p50/p95/p99 single-observation `predict()` latency plus p95 for **20** individual calls and one 20-observation batch (after warmups). The random network has not learned combat. Timings are reports, **not** portability guarantees or automatic pass thresholds. On the local Windows/Godot 4.4.1 CPU: single p95 **0.003 ms**, 20 individual p95 **0.053 ms**, 20 batched p95 **0.022 ms**. `neural_snapshot_probe.gd` verifies a versioned model architecture/weight dictionary, JSON roundtrip into a fresh node, bundled deterministic fixture, and **atomic rejection** of nonfinite weights, wrong shape and unsupported optimizer. `exported_snapshot_probe.gd` verifies a Windows debug PCK contains the fixture and loads it in an exported game alongside the GDExtension. `rendered_snapshot_probe.gd` optionally loads the fixture in a **non-headless synthetic** 400-rectangle scene, running 20 batched predictions every 100 ms. One Windows run: frame p95 **16.826 ms**, max **19.46 ms**, 20-agent batch p95 **0.111 ms**; not a real game performance claim.

This tests **inference persistence**, not bit-exact continuation of model training (optimizer moments are not serialized), real combat quality or integration with SaLMon. DQN training remains experimental under #37. See `docs/guides/neural_artifacts.rst` for the snapshot schema and asset handling.
