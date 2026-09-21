Releasing MLGodotKit
=====================

MLGodotKit uses `Semantic Versioning <https://semver.org/>`_. The version in
``MLGODOTKIT_VERSION`` is the single source of truth for releases.

Before 1.0.0, minor releases may include API changes:

* Patch releases fix bugs without intentionally changing the API.
* Minor releases add features or may change experimental APIs.
* Version 1.0.0 will identify the first stable public API.

Release checklist
-----------------

1. Update ``MLGODOTKIT_VERSION`` and ``mlgodotkit/addon/plugins/plugin.cfg`` together.
2. Run the native build and relevant tests from a clean checkout.
3. Update documentation and migration notes for API changes.
4. Commit the version change on ``main``.
5. Create a matching tag, for example::

       git tag v0.1.0
       git push origin v0.1.0

The release workflow rejects tags that do not match ``MLGODOTKIT_VERSION``. It builds a
release extension, packages the addon, includes ``MLGODOTKIT_VERSION`` in the archive, and
publishes the archive as a GitHub Release asset.

Release artifacts
-----------------

Actions artifacts from normal CI runs are temporary build outputs for
verification. GitHub Release assets are the supported user-downloadable
packages. Release packages should include the addon, license, README, version
metadata, and all required native binaries for the advertised platform.

Pre-release and experimental features
--------------------------------------

Use a pre-release identifier such as ``0.3.0-alpha.1`` when publishing an
unstable preview. RL and optional runtime backends remain experimental until
their examples, packaging, and compatibility matrix are verified.
