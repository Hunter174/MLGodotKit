Models
======
.. toctree::
   :maxdepth: 1

   lrnode
   dtreenode
   nnnode

Machine learning models implemented as native Godot nodes.

This section documents the core learning models provided by MLGodotKit.
All models are exposed as GDExtension nodes and are designed to operate
entirely within the Godot engine for both training and inference.

These models are intentionally low-level. They do not yet provide automated
pipelines, dataset abstractions, or training orchestration. Instead, they
are built to integrate directly into gameplay systems, simulation loops,
and engine-driven logic.

----

Design Philosophy
-----------------

The primary goal of this library is to enable non-deterministic,
adaptive AI behavior inside the Godot engine without relying on
external machine learning frameworks or offline tooling.

Rather than replicating full-featured ML libraries, MLGodotKit focuses on
exposing foundational tools from machine learning, control theory, and
linear algebra in a form that is:

- Explicit and inspectable
- Deterministic when required, stochastic when desired
- Tightly coupled to engine state and game logic

Training loops, loss computation, data collection, and update schedules
are deliberately left to the user. This design allows models to be trained
incrementally during gameplay, simulation, or tooling workflows, and
avoids hiding behavior behind opaque automation.

In practice, these models are intended to support AI systems that extend
beyond fixed state machines or scripted logic, while remaining fully
debuggable and controllable within Godot.