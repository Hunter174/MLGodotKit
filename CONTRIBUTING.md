# Contributing to MLGodotKit

MLGodotKit is maintained by human developers and AI agents. GitHub Issues and Pull Requests are the project record; do not create or depend on repository-local ticket files.

## Development workflow

1. Create or select a GitHub issue describing the problem, scope, acceptance criteria, and verification steps.
2. Start from an up-to-date `main` branch.
3. Create a focused issue branch, such as `issue/42-eigen-ci`.
4. Implement the smallest coherent change.
5. Run the relevant native, GDScript, documentation, and CI checks.
6. Commit focused changes and push the branch.
7. Open a pull request linked to the issue, for example `Closes #42`.
8. Merge only after review and successful required checks.

Useful commands:

```sh
gh issue list --state open
gh issue view 42
gh issue create --title "Improve ..." --body-file issue.md
gh pr create --body "Closes #42"
```

## Development standards

### C++ (GDExtension)

- Keep Godot bindings and lifecycle code in `*Node` wrappers.
- Keep engine-independent math and model logic in `*Core` classes.
- Use `std::unique_ptr` or `std::shared_ptr` for internal ownership.
- Use `godot::Ref<T>` for Godot-managed reference-counted objects.
- Use Eigen for linear algebra rather than handwritten matrix operations.
- Validate dimensions and empty inputs at public API boundaries.

### GDScript

- Prefer static typing (`var value: int = 0`).
- Preserve the contracts defined by the RL base classes.
- Keep high-level orchestration in the MLRL layer.

### Verification

Do not claim a build or test passed unless it was actually run. Include commands and results in the pull request. Native builds should be validated locally before relying on the slower GitHub Actions pipeline.

## Commit format

Use one of these prefixes:

- `feat: ...` — new functionality
- `fix: ...` — bug fix
- `refactor: ...` — internal cleanup
- `docs: ...` — documentation
- `test: ...` — tests
- `ci: ...` — CI/build changes
