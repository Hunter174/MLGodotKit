# Agent Execution Protocol

GitHub Issues are the source of truth for planned work. Agents must not rely on conversation history or local ticket files.

## Required workflow

1. Start from an up-to-date `main` branch.
2. Inspect the relevant GitHub issue with `gh issue view <number>`.
3. Create an issue-specific branch, for example `issue/42-eigen-ci`.
4. Read the issue, roadmap, and relevant source before editing.
5. Make the smallest coherent change that satisfies the issue.
6. Run the verification steps described in the issue and add results to the PR.
7. Commit focused changes using the project commit conventions.
8. Push the branch and open a pull request linked to the issue:
   `gh pr create --body "Closes #42"`.
9. Do not commit directly to `main`, merge pull requests, or close issues unless explicitly authorized.

## Issue and PR hygiene

- Use `gh issue list --state open` to find active work.
- Create issues with clear context, scope, acceptance criteria, and verification steps.
- Keep implementation notes in the pull request rather than repository-local ticket files.
- Link related issues and PRs using GitHub references such as `#42`.
- Do not claim a build, test, or Godot run passed unless it was actually run.

## Session handoff

If work pauses before completion, update the PR or issue with:

- current branch and commit;
- files changed;
- checks run and their results;
- blockers;
- the first command or action for the next session.

## Sources of truth

- GitHub Issues and Pull Requests: active tasks, decisions, verification, and handoff notes;
- `docs/roadmap.rst`: product direction and architecture roadmap;
- `CONTRIBUTING.md`: contributor and code-quality standards.
