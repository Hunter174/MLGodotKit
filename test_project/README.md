# MLGodotKit integration test project

This project is a development fixture for validating the addon and its
experimental examples. It is not the user-facing demo or a release package.

The native addon is generated into `addons/mlgodotkit/` by the SCons build and
is intentionally ignored by Git. The project targets Godot 4.4.1, matching the
`v0.1.0` compatibility matrix.

The runnable learning scenes under `examples/` are the preferred starting
point for validating common use cases. The GDScript tests under `test/` use
GUT and require that dependency to be installed separately. CI currently runs
the Godot addon smoke test and the Godot-independent Catch2 native tests; it
does not claim that the GUT suite runs in a clean checkout.
