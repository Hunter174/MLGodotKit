# MLGodotKit

MLGodotKit brings machine-learning building blocks and experimental
reinforcement-learning tools to Godot through a native GDExtension.

## 0.1.0 support

The first release is validated for:

- Godot 4.4.1
- Windows x86_64

Other platforms and Godot versions are not release-validated. RL APIs are
experimental, and ONNX/GGUF runtime integrations are planned rather than
included in this release.

## Install the release

1. Download `mlgodotkit-v0.1.0-windows.zip` from the
   [GitHub Releases](https://github.com/Hunter174/MLGodotKit/releases) page.
2. Extract `addons/mlgodotkit` into your Godot project's `addons/` directory.
3. Enable **mlgodotkit** under **Project > Project Settings > Plugins**.
4. Restart the editor if the native classes do not appear immediately.

The archive contains the Windows x86_64 native library required by the
GDExtension. Do not use the repository's debug build as a release dependency.

See the [Getting Started guide](https://mlgodotkit.readthedocs.io/en/latest/guides/getting_started.html)
and [API reference](https://mlgodotkit.readthedocs.io/en/latest/api/index.html)
for details.

## Build from source

Source builds are intended for contributors. See the
[native build guide](docs/guides/native_build.rst). The release build uses the
repository's pinned `godot-cpp` revision, Eigen 3.4.0, and SCons.

## Contributing

Use [GitHub Issues](https://github.com/Hunter174/MLGodotKit/issues) and pull
requests for bug reports, feature requests, and development work. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## License and credits

MLGodotKit is built on the [Eigen C++ library](https://eigen.tuxfamily.org/).
See [LICENSE](LICENSE) for license terms.
