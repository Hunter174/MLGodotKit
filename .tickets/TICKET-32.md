# TICKET-32: Parameterize Eigen include path and document native build setup
**Roadmap Phase:** v0.1 - Hygiene and Direction
**GitHub Issue:** #32
**Status:** VERIFIED

## 🎯 Objective
Enable the project to be built on any machine without manually editing `SConstruct` to fix Eigen paths.

## 🛠 Requirements
- [ ] Replace the hard-coded path `C:/libs/eigen-3.4.0` in `SConstruct` with a configurable option (e.g., `env['EIGEN_PATH']` with a default or environment variable).
- [ ] Add a check in `SConstruct` that fails with a helpful error message if Eigen is not found.
- [ ] Update the build documentation to explain how to set the Eigen path.

## 🧪 Verification Plan
1. Change the Eigen path to an invalid location in the config/env.
2. Run `scons`.
3. Verify that the build fails with the helpful error message.
4. Set the correct path.
5. Verify the build succeeds.

## 📝 Agent Notes

Implemented and verified on the repository-hardening branch. `SConstruct` accepts `eigen_path=...` or `EIGEN_PATH`, validates `Eigen/Core`, and emits a clear configuration error. Native build documentation was added under `docs/guides/native_build.rst`.
