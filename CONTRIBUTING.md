# Contributing to MLGodotKit

Welcome! This project is designed to be maintained by a mix of human developers and AI agents. Because maintenance happens intermittently, we prioritize **explicit documentation over implicit knowledge**.

## 🚀 The Agent Execution Workflow

If you are an AI agent, you **must** follow the "Isolated Execution" protocol:

1. **Branch**: Create a feature branch from `main` named `ticket/TICKET-XX`.
2. **Locate Task**: Read the corresponding ticket in `.tickets/TICKET-XX.md`.
3. **Execute**: Implement the change. Keep commits atomic and focused.
4. **Verify**: Execute the **Verification Plan** strictly. Record the output.
5. **PR**: Create a Pull Request to `main`.
6. **Handoff**: Update **Agent Notes** with decisions made, blockers encountered, and the exact next step for the next session.
7. **Merge**: Once verified, merge the PR into `main`.

## 🛠 Development Standards

### C++ (GDExtension)
- **Core vs Wrapper**: Keep Godot-specific logic (bindings, `_bind_methods`) in the `godot/` layer. Keep math and model logic in the `core/` layer.
- **Memory**: Use `std::unique_ptr` or `std::shared_ptr` for internal C++ objects. Use `godot::Ref<T>` for Godot-managed objects.
- **Math**: Leverage Eigen for all linear algebra. Avoid writing manual loops for matrix operations.

### GDScript
- **Type Safety**: Use static typing (`var x: int = 0`) wherever possible to help agents and developers understand the API.
- **RL Contracts**: Adhere to the base classes defined in the RL module (e.g., `AgentNode`, `RLEnvironment`).

### Git Commit Format
Please use the following format for commits:
- `feat: ...` (New feature)
- `fix: ...` (Bug fix)
- `refactor: ...` (Internal cleanup)
- `docs: ...` (Documentation)
- `ref: TICKET-XX ...` (Changes tied to a specific agent ticket)

## 🗺️ Project Roadmap
The high-level direction and phased milestones are documented in `docs/roadmap.rst`. Always check the roadmap before proposing major architectural changes.
