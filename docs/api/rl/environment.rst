RLEnvironment
=============

Base class for environments.

Override Points
---------------

- ``_reset()``
- ``_step(action)``
- ``_observe(raw_state)``

Signals
-------

- ``episode_reset(obs)``
- ``step_completed(obs, reward, done)``
- ``episode_done(total_reward)``
