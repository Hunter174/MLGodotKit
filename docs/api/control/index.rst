Control
=======
.. toctree::
   :maxdepth: 2

   motion/index
   navigation/index
   controllers/index
   filters/index


Control theory and control-oriented utilities implemented as native Godot nodes.

This section documents the control systems provided by MLGodotKit. These nodes
implement foundational control-theoretic constructs—such as feedback controllers,
filters, and signal-processing utilities—designed to operate directly within the
Godot engine.

The control nodes in this section prioritize explicit behavior, numerical stability,
and tight integration with engine state. They are intended to be composed directly
with gameplay logic, physics systems, and AI decision layers rather than used as
black-box automation.

----

Design Philosophy
-----------------

The purpose of the control module is to provide deterministic, interpretable tools
for continuous decision-making and motion control. Unlike the learning models,
control nodes do not adapt through training; instead, they expose tunable parameters
that define system response and stability.

These nodes are designed to:

- Model and regulate continuous processes (movement, steering, smoothing, tracking)
- Provide predictable, frame-stable behavior under real-time constraints
- Serve as building blocks for higher-level AI, navigation, and animation systems

In practice, the control nodes complement the machine learning models by handling
low-level dynamics and feedback, enabling hybrid systems where learned policies
operate on top of stable, well-defined control primitives.
