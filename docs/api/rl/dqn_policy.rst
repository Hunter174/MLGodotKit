DQNPolicy
=========

ε-greedy policy for Deep Q-Network (DQN) agents.

``DQNPolicy`` implements an ε-greedy action selection strategy over Q-values
produced by a neural network. It selects random actions with probability
``epsilon`` and greedy actions otherwise, using the maximum predicted Q-value.

This policy is designed to be used with ``NNNode`` as the online Q-network and
integrates naturally with ``RLRunner`` and ``RLEnvironment``.

---

Overview
--------

- ε-greedy exploration strategy
- Discrete action spaces only
- Uses ``NNNode`` for Q-value inference
- Episode-based epsilon decay
- Stateless action selection

---

Parameters
----------

``epsilon`` : float, default=1.0
    Initial exploration probability.

``epsilon_decay`` : float, default=0.995
    Multiplicative decay applied to ``epsilon`` after each episode.

``epsilon_min`` : float, default=0.05
    Lower bound on ``epsilon``.

All parameters are inspector-visible and may be modified at runtime.

---

Configuration
-------------

``configure(q_online, action_size, train_batch_size)``
    Configure the policy with a Q-network and action space size.

    Parameters
        ``q_online`` : NNNode
            Neural network producing Q-values for each action.

        ``action_size`` : int
            Number of discrete actions.

        ``train_batch_size`` : int
            Unused placeholder for interface consistency.

    Notes
        - ``q_online`` must be non-null.
        - ``action_size`` must be greater than zero.

---

Action Selection
----------------

``act(state)``
    Select an action using ε-greedy exploration.

    Parameters
        ``state`` : Array
            Observation returned by the environment.

    Returns
        ``int``
            Selected action index.

    Behavior
        - With probability ``epsilon``, selects a random action.
        - Otherwise, selects the action with the highest predicted Q-value.
        - Temporarily sets the Q-network batch size to 1 for inference.

---

Episode Lifecycle
-----------------

``on_episode_end()``
    Update exploration schedule at the end of an episode.

    Behavior
        - Increments internal episode counter.
        - No decay is applied during the first 200 episodes (warmup period).
        - After warmup, ``epsilon`` is decayed multiplicatively and clipped
          to ``epsilon_min``.

---

Algorithm Details
-----------------

- Exploration strategy: ε-greedy
- Action selection: ``argmax_a Q(s, a)``
- Epsilon decay: episode-based, post-warmup
- Q-value computation: forward pass through ``NNNode``

---

Usage Pattern
-------------

Typical usage with ``RLRunner``:

.. code-block:: gdscript

   policy.configure(q_network_
