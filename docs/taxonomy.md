# MLGodotKit Project Taxonomy & Architectural Standards

This document serves as the "Source of Truth" for naming, structure, and ownership within the MLGodotKit repository. All contributors (human and AI) must adhere to these standards to prevent architectural drift.

## 🏛 The Three Pillars

The project is divided into three distinct layers. A class must belong to exactly one pillar.

### 1. MLCore (The Engine)
**Location:** `mlgodotkit/src/...`
**Nature:** Pure C++, Eigen-powered, Godot-agnostic.
**Responsibility:** Heavy mathematical computation, memory management, and raw logic.
**Naming Convention:** 
- Classes: `[Feature]Core` (e.g., `NeuralNetworkCore`, `MatrixCore`).
- Files: `[feature]_core.h/cpp`.
- **Strict Rule:** No `godot_cpp` headers allowed here.

### 2. MLNodes (The Interface)
**Location:** `mlgodotkit/src/...` (Wrappers) $\rightarrow$ deployed to `addons/mlgodotkit/bin`
**Nature:** GDExtension / Godot Nodes.
**Responsibility:** Binding Core logic to the Godot Editor, handling `Variant` conversions, and providing Inspector properties.
**Naming Convention:** 
- Classes: `[Feature]Node` (e.g., `NeuralNetworkNode`, `PIDControllerNode`).
- Files: `[feature]_node.h/cpp`.
- **Strict Rule:** Every Node should ideally own a `unique_ptr` to a corresponding `Core` class.

### 3. MLRL (The Framework)
**Location:** `mlgodotkit/addon/nodes/rl/...`
**Nature:** GDScript high-level orchestration.
**Responsibility:** Implementing RL loops, Agent behaviors, and Environment contracts.
**Naming Convention:** 
- Base Classes: `RL[Component]` (e.g., `RLPolicy`, `RLLearner`, `RLEnvironment`).
- Implementations: `[Specific]RL[Component]` (e.g., `DQNPolicy`, `PPOLearner`).
- **Strict Rule:** Must use a contract-based approach (inherited base classes).

---

## 📂 Directory Structure

```text
/
├── mlgodotkit/
│   ├── src/                 <-- MLCore & MLNodes (C++)
│   │   ├── linalg/          <-- Math Core
│   │   ├── models/          <-- NN Core & Nodes
│   │   └── control/         <-- Control Core & Nodes
│   └── addon/               <-- MLRL & GDScript Core
│       └── nodes/
│           └── rl/          <-- RL Framework
└── test_project/            <-- Consumer/Testing area (Temporary mirror of addon)
```

## 🛠 Lifecycle of a Feature
1. **Implement Core**: Write the math in `MLCore`.
2. **Wrap Node**: Expose the Core via `MLNodes`.
3. **Integrate RL**: Use the Node within the `MLRL` framework.
4. **Verify**: Test in `test_project`.
