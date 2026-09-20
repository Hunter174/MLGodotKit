# TICKET-33: Establish canonical addon source layout outside test_project
**Roadmap Phase:** v0.1 - Hygiene and Direction
**GitHub Issue:** #33
**Status:** VERIFIED

## 🎯 Objective
Separate the reusable addon source from the test/demo project.

## 🛠 Requirements
- [ ] Move all reusable GDScript nodes and plugins from `test_project/addons/mlgodotkit/` to a canonical location (e.g., `mlgodotkit/addon/`).
- [ ] Update `SConstruct` to copy binaries and scripts from the new source to the `test_project` as part of the build process.
- [ ] Ensure `test_project` functions as a consumer of the addon rather than the primary source.
- [ ] Update the `.gdextension` file and plugin paths to match the new layout.

## 🧪 Verification Plan
1. Delete the `addons` folder inside `test_project`.
2. Run the build/copy command.
3. Open `test_project` in Godot.
4. Verify the plugin is detectable and functional.

## 📝 Agent Notes

Implemented and verified on the repository-hardening branch. Reusable addon files now live under `mlgodotkit/addon/`; SCons deploys that source plus the compiled extension into `test_project/addons/mlgodotkit`. The test project starts successfully headlessly after deployment.
