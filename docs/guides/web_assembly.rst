WebAssembly Status
==================

WebAssembly is not a release-validated target for MLGodotKit ``0.1.0``. The
``0.1.0`` package contains only the Windows x86_64 native library.

The GDExtension manifest reserves Web library entries for future builds, but a
Web binary is not included in the release archive. Do not assume that the
Windows package can be exported to the Web.

Future Web support
------------------

A supported Web release will require all of the following:

* Building MLGodotKit and its pinned ``godot-cpp`` revision for Web.
* Packaging the resulting WASM library with the addon.
* Exporting a clean example project with the matching Godot version.
* Running browser and package smoke tests in CI.

Generated Godot Web exports are intentionally excluded from source control.
For details about building GDExtensions for Web, see the official
`Godot documentation <https://docs.godotengine.org/en/stable/contributing/development/compiling/compiling_for_web.html#gdextension>`_.
