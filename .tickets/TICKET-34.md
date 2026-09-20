# TICKET-34: Refactor NeuralNetworkNode into ML core model plus Godot wrapper
**Roadmap Phase:** v0.2 - Stable NN Core
**GitHub Issue:** #34
**Status:** TODO

## Objective
Separate neural-network computation from Godot object/binding concerns while preserving a small, documented Godot-facing API.

## Requirements
- [ ] Define a Godot-independent core model API.
- [ ] Keep `_bind_methods`, properties, and Variant conversion in the wrapper.
- [ ] Preserve or intentionally version the current public API.
- [ ] Add clone and weight-copy support needed by RL target networks.
- [ ] Add C++ and Godot tests for forward, backward, and lifecycle behavior.

## Verification Plan
1. Build the native extension with the documented Eigen configuration.
2. Run the complete GUT suite.
3. Exercise the model from a Godot headless smoke test.
4. Confirm the target-network clone/copy contract with a dedicated test.

## Agent Notes
The current native build and existing XOR test pass. This task must not begin until the public API and ownership model are explicitly designed.
