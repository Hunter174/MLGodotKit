Buffers
=======

Experience storage utilities for reinforcement learning.

This module provides simple data structures for storing and sampling
environment interactions. Buffers are deliberately minimal and designed
to be paired with explicit training loops rather than automated pipelines.

----

ReplayBuffer
------------

First-in, first-out (FIFO) experience replay buffer.

``ReplayBuffer`` stores individual transitions and supports random sampling.
It is primarily intended for off-policy algorithms such as DQN.

----

Overview
^^^^^^^^

- Transition-based storage
- Fixed capacity with FIFO eviction
- Uniform random sampling
- Suitable for off-policy learning

----

Stored Transition Format
^^^^^^^^^^^^^^^^^^^^^^^^

Each entry in the buffer is a dictionary with keys:

- ``s``: state
- ``a``: action
- ``r``: reward
- ``s_next``: next state
- ``done``: terminal flag

----

Parameters
^^^^^^^^^^

``capacity`` : int, default=10000
    Maximum number of stored transitions.
    Oldest entries are discarded when capacity is exceeded.

----

Methods
^^^^^^^

``add(s, a, r, s_next, done)``
    Add a transition to the buffer.

    Notes
        - Transitions are appended in chronological order.
        - Oldest transitions are removed when capacity is exceeded.

----

``sample(batch_size)``
    Sample a batch of transitions uniformly at random.

    Returns
        ``Array``
            List of transition dictionaries.

    Notes
        - Sampling is with replacement.
        - Caller is responsible for ensuring sufficient buffer size.

----

``size()``
    Return the number of stored transitions.

----

Usage
^^^^^

Typical usage with ``DQNTrainer``:

.. code-block:: gdscript

   buffer.add(s, a, r, s_next, done)

   if buffer.size() >= batch_size:
       var batch = buffer.sample(batch_size)

----

Limitations
^^^^^^^^^^^

- No prioritized replay
- No n-step transitions
- No episode boundary tracking
- No sequence sampling

----

RolloutBuffer
-------------

Trajectory-based buffer for on-policy algorithms.

``RolloutBuffer`` stores sequences of transitions collected during an episode
or rollout. It is intended for on-policy methods such as PPO, A2C, or policy
gradient algorithms.

----

Overview
^^^^^^^^

- Step-by-step trajectory storage
- Maintains aligned arrays for each quantity
- Cleared after each rollout
- Suitable for advantage-based methods

----

Stored Data
^^^^^^^^^^^

The buffer stores the following arrays:

- ``states``
- ``actions``
- ``rewards``
- ``dones``
- ``log_probs``
- ``values``

All arrays are appended synchronously per step.

----

Methods
^^^^^^^

``store(s, a, r, done, log_p=0.0, v=0.0)``
    Store a single timestep of a rollout.

    Parameters
        ``s``
            Observation or state.

        ``a``
            Action taken.

        ``r``
            Reward received.

        ``done``
            Terminal flag.

        ``log_p``
            Log probability of the action under the policy.

        ``v``
            Estimated state value.

----

``clear()``
    Clear all stored rollout data.

----

``size()``
    Return the number of stored timesteps.

----

Usage
^^^^^

Typical usage pattern:

.. code-block:: gdscript

   rollout.store(s, a, r, done, log_p, value)

   if done:
       process_rollout(rollout)
       rollout.clear()

----

Limitations
^^^^^^^^^^^

- Single-trajectory only
- No batching or padding
- No return or advantage computation
- No time-limit handling

----

Design Notes
------------

``ReplayBuffer`` and ``RolloutBuffer`` intentionally expose raw data rather
than computed targets or advantages. This keeps learning logic explicit and
allows trainers to implement custom algorithms without hidden assumptions.

----

See Also
--------

- ``DQNTrainer`` for off-policy learning
- ``RLRunner`` for episode execution
- Policy gradient trainers for on-policy learning
