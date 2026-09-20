# Agent Execution Protocol

This repository uses short, isolated, artifact-driven tickets. Agents must not
rely on conversation history.

## Required workflow

1. Start from an up-to-date `main` branch.
2. Create `ticket/TICKET-XX` (or another ticket-specific feature branch).
3. Read the ticket, roadmap, and relevant source before editing.
4. Make the smallest coherent change that satisfies the ticket.
5. Run every verification step in the ticket. Add any extra validation results
   to the ticket's Agent Notes.
6. Mark a ticket `VERIFIED` only when all required checks pass. Use
   `BLOCKED` or `NEEDS-DESIGN` when they do not.
7. Commit code, tests, and ticket updates together in focused commits.
8. Push the feature branch and open a pull request to `main`.
9. Never commit directly to `main` and never merge an unverified ticket.

## Session handoff

If the session ends before completion, record:

- current branch and commit;
- files changed;
- checks run and their results;
- blockers;
- the first command or action for the next agent.

Do not claim a build, Godot run, or test passed unless it was actually run.

## Sources of truth

- `.tickets/`: executable task specifications and handoff notes;
- `docs/roadmap.rst`: product direction and architecture roadmap;
- `CONTRIBUTING.md`: contributor and code-quality standards.
