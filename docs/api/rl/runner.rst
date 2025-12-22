RLRunner
========

Execution coordinator for reinforcement learning episodes.

``RLRunner`` orchestrates the interaction between an environment, a policy, and
a trainer. It is responsible for driving the episode loop, collecting
transitions, triggering training steps, and managing optional rendering delays.

This class does **not** implement learning logic itself; instead, it provides a
deterministic and explicit control loop suitable for experimentation, debugging,
and real-time learning.

----

Overview
--------

- Centralized episode execution loop
- Coordinates environment, policy, and trainer
- Optional step delay for visualization
- Explicit global step tracking
- Signal-based episode completion

----

Core Components
---------------

``env``
    Instance of ``RLEnvironment`` defining the environment dynamics.

``policy``
    Instance of ``RLPolicy`` responsible for action selection.

``trainer``
    Training component responsible for learning updates.
    The runner treats this object as a black box.

----

Configuration
-------------

``configure(env, policy, trainer)``
    Attach the environment, policy, and trainer used during execution.

    Parameters
        ``env`` : RLEnvironment
            Environment instance.

        ``policy`` : RLPolicy
            Policy instance.

        ``trainer``
            Trainer object implementing the expected training interface.

----

Execution
---------

``run_episode()``
    Execute a single reinforcement learning episode.

    Returns
        ``float``
            Total accumulated reward for the episode.

    Behavior
        - Resets the environment
        - Repeatedly queries the policy for actions
        - Steps the environment
        - Passes transitions to the trainer
        - Triggers training updates when requested
        - Terminates on environment signal or step limit
        - Emits ``episode_finished`` signal

----

Trainer Interface Expectations
------------------------------

``RLRunner`` does not enforce a concrete trainer type. The provided trainer is
expected to implement the following methods:

- ``observe(state, action, reward, next_state, done)``
- ``should_train(global_step)``
- ``train_step()``

This design allows trainers to implement:
- Online or batch learning
- On-policy or off-policy algorithms
- Experience replay or direct updates

----

Signals
-------

``episode_finished(total_reward)``
    Emitted when an episode completes.

----

Configuration Parameters
------------------------

``step_delay`` : float, default=0.0
    Optional delay (in seconds) between environment steps.
    Useful for visualization and debugging.

``render_mode`` : bool, default=false
    If enabled, the runner yields execution between steps using ``step_delay``.

----

Global Step Tracking
--------------------

``global_step``
    Counter incremented after every environment step across all episodes.
    Typically used by trainers to schedule updates or exploration decay.

----

Execution Flow
--------------

Typical usage pattern:

.. code-block:: gdscript

   runner.configure(env, policy, trainer)

   while true:
       var reward = runner.run_episode()
       print("Episode reward:", reward)

----

Design Philosophy
-----------------

``RLRunner`` intentionally keeps the training loop explicit:

- No hidden background threads
- No implicit batching
- No enforced learning schedules

This makes it suitable for:
- Debugging RL algorithms
- Educational demonstrations
- Real-time interactive training
- Deterministic experimentation

----

Limitations
-----------

- Single environment execution
- No parallel or vectorized rollout support
- Trainer interface is conve
