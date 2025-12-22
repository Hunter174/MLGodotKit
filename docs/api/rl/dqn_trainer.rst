DQNTrainer
==========

Trainer implementing Double DQN with experience replay and Polyak averaging.

``DQNTrainer`` is responsible for learning Q-values using a Deep Q-Network
(DQN)–style algorithm. It maintains an online network, a target network, and a
replay buffer, and performs gradient-based updates using sampled transitions.

This trainer is designed to be driven by ``RLRunner`` and paired with
``DQNPolicy`` for action selection.

---

Overview
--------

- Double DQN target computation
- Experience replay
- Online and target Q-networks
- Polyak (soft) target updates
- Explicit training step control

---

Core Components
---------------

``q_online``
    Online Q-network (``NNNode``) updated via backpropagation.

``q_target``
    Target Q-network (``NNNode``) updated via Polyak averaging.

``buffer``
    Replay buffer storing experience tuples
    ``(state, action, reward, next_state, done)``.

``action_size``
    Number of discrete actions.

---

Parameters
----------

``gamma`` : float, default=0.99
    Discount factor used for future rewards.

``batch_size`` : int, default=128
    Number of transitions sampled per training step.

``warmup`` : int, default=1000
    Minimum number of stored transitions before training begins.

``target_update_period`` : int, default=2000
    Interval (in environment steps) at which target updates are applied.

``polyak_tau`` : float, default=0.005
    Interpolation factor for Polyak averaging.

``huber_delta`` : float, default=1.0
    Threshold parameter for the Huber loss.

All parameters are inspector-visible and can be modified at runtime.

---

Configuration
-------------

``configure(q_online, q_target, buffer, action_size)``
    Configure the trainer.

    Parameters
        ``q_online`` : NNNode
            Online Q-network.

        ``q_target`` : NNNode
            Target Q-network.

        ``buffer``
            Replay buffer instance.

        ``action_size`` : int
            Number of discrete actions.

    Notes
        - Both networks and the buffer must be non-null.
        - ``action_size`` must be greater than zero.

---

Trainer Interface
-----------------

``observe(s, a, r, s_next, done)``
    Store a transition in the replay buffer.

``should_train(step)``
    Determine whether a training step should be performed.

    Returns
        ``bool``
            ``true`` if the replay buffer contains at least ``warmup`` entries
            and at least ``batch_size`` samples.

---

``train_step()``
    Perform a single training update.

    Behavior
        - Samples a batch from the replay buffer
        - Computes Double DQN targets
        - Computes Huber loss gradients
        - Backpropagates gradients through ``q_online``
        - Applies Polyak updates to ``q_target`` periodically

---

Algorithm Details
-----------------

**Target computation (Double DQN)**

For each transition:

- Action selection uses the online network
  ``a* = argmax_a Q_online(s', a)``

- Target evaluation uses the target network

  ``y = r + γ · Q_target(s', a*)`` (if not terminal)

---

**Loss Function**

Huber loss per sample:

- Quadratic for small errors
- Linear for large errors

Gradients are clipped implicitly by the Huber formulation.

---

**Target Network Update**

Polyak averaging is applied periodically:

