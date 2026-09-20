# Agent Execution Protocol

To maximize efficiency and maintain consistency across different AI sessions, this project uses an **Artifact-Driven Execution** model.

## 🔄 The Execution Loop

Agents should not rely on chat history. Every task must follow this loop:

1. **Initialization**: Read the corresponding ticket in `.tickets/TICKET-XX.md`.
2. **Execution**: Implement the minimum changes required to satisfy the **Requirements**.
3. **Verification**: Run the steps listed in the **Verification Plan**.
4. **Documentation**: Update the **Agent Notes** and mark the status as `VERIFIED`.
5. **Handoff**: If the session ends before completion, record the current state and the immediate next step in the ticket notes.

## 🛠 Technical Guidelines

- **Atomic Commits**: One ticket = one or more focused commits. Use the format `ref: TICKET-XX [Description]`.
- **No Guessing**: If a requirement is ambiguous, the agent must update the ticket as `status: needs-design` and stop until clarified.
- **Clean State**: Always verify that changes do not introduce new untracked generated files (build artifacts).

## 📂 Project Structure for Agents
- `.tickets/`: The source of truth for active work.
- `docs/roadmap.rst`: The high-level direction.
- `docs/architecture.md` (upcoming): The technical constraints.
