# Changelog

All notable changes to MLGodotKit are documented here.

## [Unreleased]

- Continue stabilizing the MLCore/MLNodes public API.
- Add optional bundled-model runtime support planning for ONNX and GGUF.

## [0.1.0] - 2026-09-21

### Added

- Canonical addon source under `mlgodotkit/addon/`.
- Configurable Eigen discovery through `EIGEN_PATH` or `eigen_path`.
- Windows native build and GitHub Release packaging workflow.
- Neural-network and PID controller Core/Node separation.
- Catch2 native test scaffold.
- RL framework scripts under the canonical addon source layout.

### Changed

- Generated native artifacts are excluded from source control.
- Model, loss, optimizer, and control APIs received initial lifecycle and
  validation cleanup.

### Experimental

- Reinforcement learning APIs and examples remain experimental.
- Model runtime integrations, including ONNX Runtime and GGUF/llama.cpp, are
  planned but are not included in this release.

### Compatibility

- Minimum declared Godot version: 4.1.
- This release currently publishes a Windows x86_64 native artifact.
- Other platforms are present in the extension manifest but are not release
  validated yet.
