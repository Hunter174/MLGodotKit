Native Build Setup
==================

The native extension is built with SCons. The repository includes ``godot-cpp``
as a Git submodule, so initialize it after cloning::

    git submodule update --init --recursive

Eigen is an external dependency and is not committed to this repository. Set
its installation directory with ``EIGEN_PATH`` or pass ``eigen_path`` directly::

    scons platform=windows eigen_path=C:/libs/eigen-3.4.0

On PowerShell, the equivalent environment-variable form is::

    $env:EIGEN_PATH = 'C:/libs/eigen-3.4.0'
    scons platform=windows

The path must contain ``Eigen/Core``. A missing or invalid path produces an
explicit configuration error before compilation starts.

After a successful build, SCons copies the extension binary into the test
project at ``test_project/addons/mlgodotkit/bin``. Restart Godot after replacing
the binary so the editor reloads the extension.
