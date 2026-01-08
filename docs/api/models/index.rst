Models
======
.. toctree::
   :maxdepth: 1

   lrnode
   dtreenode
   nnnode

Machine learning models implemented as native Godot nodes.

This section documents the core learning models provided by MLGodotKit.
All models are exposed as GDExtension nodes and are designed for explicit,
engine-integrated training and inference.

The models in this section prioritize transparency and control over automation.
Training loops, loss functions, and data handling are intentionally left to the
user.

----

Design Philosophy
-----------------

MLGodotKit models deliberately avoid high-level abstractions such as
automatic fitting pipelines or implicit optimization routines.

This makes them well suited for:

- Real-time and interactive learning
- Reinforcement learning agents
- Educational and experimental workflows
- Tight integration with gameplay logic

----
