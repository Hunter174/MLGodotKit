API Reference
=============
.. toctree::
    :maxdepth: 2

    matrix
    linalg
    models/index
    rl/index

Public API documentation for MLGodotKit.

This section provides detailed documentation for all public classes, nodes, and
utilities exposed by MLGodotKit. The API is organized by responsibility rather
than implementation language, and reflects the intended way components are
combined in practice.

The API favors explicit control and composability over automated pipelines.
Most components are designed to be orchestrated manually by user code.

----

Numerical Foundations
---------------------

Low-level numerical utilities used throughout the library.

``Matrix``
    Dense matrix container with basic linear algebra operations.

``Linalg``
    Stateless solvers and matrix decompositions backed by Eigen.

These components form the numerical backbone for learning models and
reinforcement learning algorithms.

----

Machine Learning Models
-----------------------

Core learning models implemented as native Godot nodes.

See :doc:`models/index` for a complete overview.

Included models:
- Linear regression
- Decision tree classification
- Feed-forward neural networks

Models expose explicit training and inference interfaces and do not assume
specific data pipelines or loss functions.

----

Reinforcement Learning
----------------------

Composable components for building reinforcement learning systems.

See :doc:`rl/index` for a conceptual overview.

Included components:
- Environment abstractions
- Policies and action selection
- Episode runners
- Trainers and experience buffers

The RL API is inspired by Gym-style design, while remaining engine-first and
explicit.

----

Design Notes
------------

The API is intentionally:

- **Explicit** — no hidden training loops or background processes
- **Composable** — components can be mixed and matched freely
- **Inspectable** — state, gradients, and decisions are accessible
- **Engine-integrated** — designed for real-time Godot execution

Users are encouraged to build their own workflows rather than rely on fixed
pipelines.