Reinforcement Learning
======================

Composable reinforcement learning components for Godot.

This section documents the reinforcement learning (RL) building blocks provided
by MLGodotKit. Rather than offering a monolithic RL framework, the library
exposes **small, composable components** that can be assembled into custom
training loops and learning systems.

The design is inspired by OpenAI Gym and modern RL libraries, while remaining
explicit, debuggable, and suitable for real-time execution inside the Godot
Engine.

----

Core Abstractions
-----------------

Environments
^^^^^^^^^^^^

``RLEnvironment``
    Base class defining episodic environment dynamics.

    - Explicit reset / step interface
    - Separation of internal state and observations
    - Signal-driven lifecycle events

----

Policies
^^^^^^^^

``RLPolicy``
    Abstract interface for action selection.

``DQNPolicy``
    ε-greedy policy for discrete-action Deep Q-Network agents.

    - Exploration scheduling
    - NN-based Q-value inference
    - Episode-aware behavior

----

Execution
^^^^^^^^^

``RLRunner``
    Coordinates environment interaction, policy execution, and training.

    - Deterministic episode loop
    - Optional rendering and step delays
    - Global step tracking

----

Training
^^^^^^^^

``DQNTrainer``
    Trainer implementing Double DQN with experience replay.

    - Online and target networks
    - Polyak (soft) target updates
    - Explicit gradient computation

----

Experience Buffers
^^^^^^^^^^^^^^^^^^

``ReplayBuffer``
    FIFO experience replay for off-policy learning.

``RolloutBuffer``
    Trajectory-based storage for on-policy algorithms (e.g. PPO).

----

Design Philosophy
-----------------

MLGodotKit reinforcement learning components intentionally avoid:

- Hidden background training
- Implicit loss computation
- Automatic environment wrapping
- Prescriptive algorithm pipelines

Instead, the library emphasizes:

- Explicit control over training logic
- Clear separation of responsibilities
- Step-by-step inspectability
- Easy experimentation and debugging

This makes the RL stack suitable for:
- Research and prototyping
- Educational use
- Game-integrated learning agents
- Custom algorithm development

----

.. toctree::
   :maxdepth: 1

   environment
   policy
   runner
   dqn_policy
   dqn_trainer
   buffers
