Getting Started
===============

Requirements
------------

MLGodotKit ``0.1.0`` currently provides a validated native release for:

* Godot 4.4.1
* Windows x86_64

Other platforms and Godot versions are not release-validated yet. The RL APIs
and future ONNX/GGUF integrations are experimental or planned.

Installing a release
--------------------

1. Download ``mlgodotkit-v<version>-windows.zip`` from the
   `GitHub Releases page <https://github.com/Hunter174/MLGodotKit/releases>`_.
2. Extract the archive into your Godot project so the layout is::

       your-project/
       \- addons/
          \- mlgodotkit/
             \- mlgodotkit.gdextension
             \- bin/

3. Open the project in Godot and enable **mlgodotkit** under
   **Project > Project Settings > Plugins**.
4. Restart the editor if the native classes do not appear immediately.

The release archive already contains the platform-specific native library; do
not copy files from the repository's build directories into a user project.

Building from source
--------------------

Source builds are intended for contributors. See :doc:`native_build` for
Eigen installation, SCons commands, and native tests. A source build must use
the repository's pinned ``godot-cpp`` revision and the matching Godot version.

Examples
--------

The repository's ``test_project/examples`` directory contains standalone
scenes for common workflows:

* ``matrix_basics.tscn`` for matrix construction and multiplication.
* ``linear_regression.tscn`` for fitting and prediction.
* ``neural_network_xor.tscn`` for an explicit forward/loss/backward loop.
* ``pid_control.tscn`` for feedback control and tuning.

Open these scenes in Godot and use **Play Current Scene** after building the
local addon. The example scripts are intentionally short and can be copied
into a user's project.

Next steps
----------

The API reference is available in the :doc:`../api/index` documentation.
Experimental RL code is included in the repository but is not part of the
stable API promise for this pre-1.0 release.

Troubleshooting
---------------

* If Godot reports that the extension was built for a newer engine, use Godot
  4.4.1 for the ``0.1.0`` Windows package.
* If no native classes appear, check that ``addons/mlgodotkit/bin`` contains
  the Windows x86_64 DLL and that the plugin is enabled.
* Report reproducible problems through a
  `GitHub Issue <https://github.com/Hunter174/MLGodotKit/issues>`_.
