API Reference
=============
.. toctree::
    :maxdepth: 2
    :hidden:

    matrix
    linalg
    models/index
    rl/index

.. contents::
   :local:
   :depth: 2

Public API documentation for MLGodotKit.

This section provides detailed documentation for all public classes, nodes, and
utilities exposed by MLGodotKit. The API is organized by responsibility rather
than implementation language, and reflects the intended way components are
combined in practice.

The API favors explicit control and composability over automated pipelines.
Most components are designed to be orchestrated manually by user code.

----

Numerical Foundations
---------------------

Low-level numerical utilities used throughout the library.

``Matrix`` This is a dense matrix container meant to act as a pseudo primitive class for downstream linear algebra tooling
and model development.

``Linalg`` Leveraging the support of Eigen (C++ library) this module acts as a stateless wrapper for many of the built in
utility presented by Eigen. This module is meant to work seamlessly with both native godot scripting and nodes as well as
the aforementioned matrix container class. If there is utility offered by eigen that we do not currently support please
add an issue on the github linking to that utility and I will do my best to implement it!

These tools form the numerical backbone of this plugin and serve as independent utility outside of the currently support
models.

----

Machine Learning Models
-----------------------

Core learning models implemented as native Godot nodes. See :doc:`models/index` for a complete overview. This plugin
currently supports three model implementation (so far). Those models are a simple stochastic linear regression, a
classification decision tree, and a feed forward neural network.

Models expose explicit training and inference interfaces and do not assume
specific data pipelines or loss functions.

I really enjoy the Sklearn implementation design of models so if there is a model you see there that you would like added
to the supported list please make an issue linking to the desired model for reference. I will try my best to expeditiously
implement said model or tool.

----

Reinforcement Learning
----------------------

Composable components for building reinforcement learning systems. See :doc:`rl/index` for a conceptual overview. There
are currently three components supported by this plugin. Those are: Environmental abstractions, policies and action selection
episode runners, as well as trainers and experience buffers. Each designed with the purpose of building flexible reinforcement
learning pipelines to build a Gym-style RL API, while remaining engine=first and explicit.

----

Design Notes
------------

The API is intentionally:

- **Explicit** — no hidden training loops or background processes
- **Composable** — components can be mixed and matched freely
- **Inspectable** — state, gradients, and decisions are accessible
- **Engine-integrated** — designed for real-time Godot execution

Users are encouraged to build their own workflows rather than rely on fixed
pipelines. That being said examples and recommendations can be found at :doc:`guides/index`