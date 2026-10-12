# Shared-policy inference baseline (issue #50)

From the repository root with a staged native addon on Windows and Godot 4.4.1:

```sh
Godot --headless --path test_project --script res://scripts/probes/shared_combat_policy_probe.gd
```

This script constructs one 16→32→8 `NeuralNetworkNode`, copies its weights into a separate identical node, checks equal output, output shape and non-mutating inference, then reports p50/p95/p99 single-observation `predict()` latency plus p95 for **20** individual calls and one 20-observation batch (after warmups). The random network has not learned combat. Timings are reports, **not** portability guarantees or automatic pass thresholds. On the local Windows/Godot 4.4.1 CPU: single p95 **0.003 ms**, 20 individual p95 **0.053 ms**, 20 batched p95 **0.022 ms**. No training, model save/load, Godot rendering or integration with SaLMon is tested here. Pretrained weight import/export remains part of #35/#46; stable DQN training remains part of #37. Keep #50 open until an actual artifact round trip and rendered scene are validated.
