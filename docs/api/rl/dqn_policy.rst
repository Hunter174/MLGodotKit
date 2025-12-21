DQNPolicy
=========

ε-greedy policy for DQN.

Parameters
----------

- ``epsilon``
- ``epsilon_decay``
- ``epsilon_min``

Behavior
--------

- Uses ``NNNode`` for Q-values
- Decays epsilon after warmup
