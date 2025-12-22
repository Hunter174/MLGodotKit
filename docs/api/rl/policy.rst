RLPolicy
========

Abstract policy interface for reinforcement learning agents.

``RLPolicy`` defines the minimal interface required for decision-making
components in reinforcement learning workflows. A policy maps observations
(or states) to actions and may optionally react to episode boundaries.

This class is intended to be subclassed. It does not implement any learning or
decision logic by default.

----

Overview
--------

- Stateless-by-default policy abstraction
- Explicit action selection interface
- Optional episode lifecycle hook
- Designed to integrate with ``RLEnvironment``

----

Core Concept
------------

A **policy** is responsible for selecting an action given the current
observation. How that action is computed—random sampling, heuristics, neural
networks, or learned value functions—is entirely up to the implementation.

``RLPolicy`` intentionally does not assume:
- Discrete or continuous action spaces
- Any particular observation format
- Any learning algorithm

----

Methods
-------

``act(state)``
    Select an action given the current observation.

    Parameters
        ``state``
            Observation returned by the environment.

    Returns
        Action compatible with the environment’s ``_step`` method.

    Notes
        - Must be overridden by subclasses.
        - Called once per environment step.

----

``on_episode_end()``
    Optional hook invoked at the end of an episode.

    Notes
        - Default implementation does nothing.
        - Override to reset internal state, update exploration schedules,
          or finalize learning updates.

----

Usage Pattern
-------------

Policies are typically used alongside an ``RLEnvironment``:

.. code-block:: gdscript

   var obs = env.reset()
   while true:
       var action = policy.act(obs)
       var result = env.step(action)
       obs = result.state
       if result.done:
           policy.on_episode_end()
           break

----

Design Philosophy
-----------------

``RLPolicy`` separates **decision logic** from:

- Environment dynamics
- Training loops
- Learning algorithms

This allows the same environment to be paired with:
- Random or scripted policies
- ``NNNode``-based policies
- Reinforcement learning agents
- Evaluation-only controllers

----

Limitations
-----------

- No built-in learning logic
- No action-space validation
- No observation preprocessing
- Single-agent only

----

See Also
--------

- ``RLEnvironment`` for environment interaction
- ``NNNode`` for neural-network-based policies
