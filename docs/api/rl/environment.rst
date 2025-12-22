RLEnvironment
=============

Base class for reinforcement learning environments.

``RLEnvironment`` defines a minimal interface for episodic reinforcement learning
inside the Godot Engine. It manages episode lifecycle, step counting, reward
accumulation, and signal emission, while delegating environment-specific logic
to user-defined override methods.

The design is inspired by OpenAI Gym environments, with explicit separation
between **internal state** and **observations**.

----

Overview
--------

- Episodic environment abstraction
- Explicit reset / step API
- Raw internal state separated from observations
- Signal-driven interaction model
- Designed for real-time and interactive RL

----

Core Concepts
-------------

**Raw State**
Internal environment state (physics, game objects, simulation variables).
This state is never exposed directly to agents.

**Observation**
Processed representation of the raw state, returned to the agent and emitted
via signals. Observations are produced by ``_observe``.

----

Public API
----------

``reset()``
    Reset the environment and start a new episode.

    Returns
        ``Array``
            Initial observation.

    Behavior
        - Clears step counter and accumulated reward
        - Calls ``_reset`` to initialize raw state
        - Converts raw state to observation via ``_observe``
        - Emits ``episode_reset`` signal

----

``step(action)``
    Advance the environment by one step.

    Parameters
        ``action``
            Agent action (type defined by the environment).

    Returns
        ``Dictionary``
            Dictionary with keys:

            - ``state``: observation
            - ``reward``: float
            - ``done``: bool

    Behavior
        - Calls ``_step(action)`` to update raw state
        - Accumulates reward
        - Enforces ``max_steps`` termination
        - Emits ``step_completed`` signal
        - Emits ``episode_done`` when terminal

----

Override Points
---------------

Subclasses must implement the following methods.

``_reset()``
    Initialize and return the raw environment state.

    Returns
        ``Array``
            Raw state representation.

----

``_step(action)``
    Apply an action to the environment.

    Parameters
        ``action``
            Agent action.

    Returns
        ``Dictionary``
            Dictionary with keys:

            - ``state``: raw state
            - ``reward``: float
            - ``done``: bool

----

``_observe(raw_state)``
    Convert raw internal state to an observation.

    Parameters
        ``raw_state`` : Array
            Internal state.

    Returns
        ``Array``
            Observation passed to the agent.

    Notes
        - Default implementation returns the raw state unchanged.
        - Override to implement feature extraction or normalization.

----

Signals
-------

``episode_reset(initial_obs)``
    Emitted when a new episode begins.

``step_completed(obs, reward, done)``
    Emitted after each environment step.

``episode_done(total_reward)``
    Emitted when an episode terminates.

----

Configuration
-------------

``max_steps`` : int, default=1000
    Maximum number of steps per episode.
    Episodes terminate automatically when exceeded.

----

Design Philosophy
-----------------

``RLEnvironment`` intentionally avoids prescribing:

- Action space definitions
- Reward shaping strategies
- Observation formats
- Agent or training logic

This allows the environment to be used with:
- Custom RL agents
- ``NNNode``-based policies
- Manual or scripted controllers
- Online or offline learning loops

----

Example
-------

Minimal custom environment:

.. code-block:: gdscript

   extends RLEnvironment

   func _reset():
       return [0.0]

   func _step(action):
       var next_state = [state[0] + action]
       return {
           "state": next_state,
           "reward": -abs(next_state[0]),
           "done": false
       }

----

Limitations
-----------

- Single-agent only
- No built-in action or observation spaces
- No vectorized environments
- No parallel execution

----

See Also
--------

- ``NNNode`` for policy and value networks
- Reinforcement learning agents and trainers
