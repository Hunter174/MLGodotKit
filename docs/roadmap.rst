Technical Roadmap
=================

Purpose
-------

This roadmap is the technical operating plan for MLGodotKit. The project is
maintained intermittently, so the goal is to keep direction stable, reduce
context-reload cost, and make each work session produce durable progress.

Product Direction
-----------------

MLGodotKit should become a Godot-native toolkit for lightweight runtime machine
learning, reinforcement learning, and AI control.

The project should not try to become a full replacement for PyTorch,
TensorFlow, scikit-learn, or a large offline training stack. Its strength should
be tight Godot integration: scene-aware agents, runtime inference/training for
small models, useful math primitives, and workflows that are understandable to
GDScript users.

North-star statement:

    MLGodotKit provides Godot-first ML and RL primitives for game agents,
    simulations, and interactive AI behaviors.

Non-goals
---------

The following are explicitly out of scope for the near-term roadmap:

* Large-scale deep learning training.
* GPU kernels or custom accelerator support.
* Full automatic differentiation frameworks.
* Competing with Python ML ecosystems for offline experimentation.
* Broad algorithm coverage before core APIs are stable.
* Adding many new models while existing model lifecycle, tests, and examples are
  incomplete.

Core Design Principles
----------------------

1. Scene behavior belongs in Godot ``Node`` classes.

   Examples: RL agents, environments, observables, PID controllers, movement
   controllers.

2. Models, losses, optimizers, replay buffers, and data containers should be
   ``RefCounted`` or ``Resource``-style objects where possible.

   A neural network is data and computation, not inherently scene behavior.

3. C++ should own performance-sensitive math and training primitives.

   Examples: matrix operations, neural network forward/backward, losses,
   optimizers, serialization, model copying.

4. GDScript should own Godot orchestration.

   Examples: reward functions, observation collection, environment stepping,
   scene interaction, simple RL learners and policies until performance demands
   otherwise.

5. Public APIs should be small and stable before adding breadth.

6. Experimental features are allowed, but they must be labeled as experimental
   in docs, issue labels, and examples.

Target Module Boundaries
------------------------

The repository should converge toward these conceptual modules:

Core Math
~~~~~~~~~

Responsibilities:

* Matrix wrapper.
* Linear algebra operations.
* Godot/Eigen conversion utilities.
* Numeric validation helpers.

Primary public types:

* ``Matrix``
* ``Linalg``

ML Core
~~~~~~~

Responsibilities:

* Neural network model internals.
* Layers and activations.
* Losses.
* Optimizers.
* Classical lightweight models where they remain useful.

Primary public types:

* ``MLNeuralNetwork`` or equivalent final public name.
* ``MSELoss`` / ``BCELoss`` / ``CrossEntropyLoss``.
* ``AdamOptimizer`` / ``SGDOptimizer``.
* ``LinearRegression``.
* ``DecisionTree``.

RL Layer
~~~~~~~~

Responsibilities:

* Scene-integrated agent lifecycle.
* Environment stepping.
* Observation aggregation.
* Policies.
* Value functions.
* Replay buffers.
* DQN-style learners.

Likely implementation split:

* Godot-facing orchestration in GDScript.
* Neural-network and batch math in C++.

Control Layer
~~~~~~~~~~~~~

Responsibilities:

* PID controller.
* Filters.
* Movement helper nodes.
* Agent movement utilities.

This module supports the AI-agent story, but should not drive the ML API design.

Repository Layout Target
------------------------

The exact layout can evolve, but the long-term direction should separate core
implementation, Godot bindings, addon-distributed GDScript, and demo projects.

Potential target layout::

    mlgodotkit/
      SConstruct
      src/
        core/
          math/
          nn/
          losses/
          optimizers/
          models/
        godot/
          bindings/
          register_types.cpp
        utility/
      addon/
        mlgodotkit.gdextension
        plugins/
        nodes/
          rl/
          control/
        bin/
    test_project/
      project.godot
      examples only
    examples/
      rl/
      neural_network/
      control/
    docs/

Important direction: ``test_project`` should consume the addon; it should not be
the canonical source location for reusable addon scripts.

Public API Direction
--------------------

The desired high-level user experience should be code-first and Godot-friendly.

Model construction example::

    var model := MLNeuralNetwork.new()
    model.add_dense(2, 64, "relu")
    model.add_dense(64, 64, "relu")
    model.add_dense(64, 4, "linear")

    var q_values = model.predict([[0.25, -0.5]])

Training primitive example::

    var preds = model.forward(states)
    var loss_value = loss.forward(preds, targets)
    var grad = loss.backward()
    model.backward(grad)
    optimizer.step(model)

Convenience APIs may exist, but the primitives should remain clear:

* ``forward`` computes outputs and stores caches needed for training.
* ``backward`` computes gradients.
* ``optimizer.step`` updates parameters.
* ``predict`` performs inference-oriented forward execution.
* ``clone`` creates an independent model copy.
* ``copy_weights_from`` copies compatible parameters.

Roadmap Phases
--------------

Phase 0 — Stabilize Direction and Project Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Goal:
    Make the project easy to resume after long breaks.

Status:
    Current immediate phase.

Deliverables:

* Maintain this technical roadmap.
* Create or clean GitHub labels for roadmap-driven triage.
* Convert existing GitHub issues into epics/tasks aligned to this roadmap.
* Add a small contributor/developer workflow note.
* Define which current features are stable vs experimental.

Recommended GitHub labels:

* ``type:bug``
* ``type:feature``
* ``type:refactor``
* ``type:docs``
* ``type:test``
* ``type:release``
* ``area:build``
* ``area:docs``
* ``area:core-math``
* ``area:ml-core``
* ``area:rl``
* ``area:control``
* ``area:examples``
* ``priority:p0``
* ``priority:p1``
* ``priority:p2``
* ``status:blocked``
* ``status:needs-design``
* ``status:good-first-issue``
* ``stability:experimental``

Exit criteria:

* Roadmap exists in docs.
* Issues are grouped by area and priority.
* A small set of next-session tasks is obvious from GitHub.

Phase 1 — Build, Packaging, and Test Hygiene
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Goal:
    Make the project buildable, testable, and releasable with less manual state.

Problems to address:

* Build artifacts currently appear inside ``src``.
* Eigen path is hard-coded in ``SConstruct``.
* Addon source appears to live under ``test_project/addons``.
* Test/demo project and distributed addon are blurred.

Deliverables:

* Stop tracking generated files such as ``.o``, ``.obj``, ``.dll``, ``.lib``,
  ``.exp``, and local SCons artifacts unless intentionally released.
* Update ``.gitignore`` for generated native build outputs.
* Parameterize Eigen include location.
* Document Windows build prerequisites.
* Ensure ``scons platform=windows`` works from a clean checkout with documented
  setup.
* Make addon packaging deterministic.
* Decide canonical location for reusable addon GDScript.

Exit criteria:

* Clean checkout build instructions are accurate.
* Generated files do not pollute source directories.
* Addon package can be rebuilt/copied intentionally.

Phase 2 — Neural Network Core Refactor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Goal:
    Separate Godot bindings from neural-network implementation details.

Current issue:
    ``NeuralNetworkNode`` mixes scene node behavior, model state, training,
    optimizer ownership, inspector config, conversion, and logging.

Deliverables:

* Introduce an internal C++ neural network core class without Godot binding
  responsibilities.
* Keep or wrap existing ``NeuralNetworkNode`` temporarily for compatibility.
* Decide final public name, likely not ``Node``-based. Candidate names:

  * ``MLNeuralNetwork``
  * ``NeuralNetwork``
  * ``NeuralNetworkResource``

* Implement model lifecycle methods:

  * ``clone``
  * ``copy_weights_from``
  * ``clear`` / ``reset``
  * ``get_config``
  * ``from_config``

* Make batch behavior explicit and predictable.
* Add dimension validation for every public training/inference path.
* Add deterministic seeding support or a documented random initialization policy.

Exit criteria:

* RL code can create separate online and target networks safely.
* Model cloning no longer aliases the same underlying object.
* Existing simple examples still run through compatibility wrappers or updated
  APIs.

Phase 3 — Losses and Optimizers API Cleanup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Goal:
    Make training semantics explicit and composable.

Current issue:
    Model ``backward`` currently applies optimizer updates directly. This is
    convenient but conceptually surprising and makes future optimizer work hard.

Deliverables:

* Decide final training API:

  * Preferred: ``model.backward(grad)`` computes gradients and
    ``optimizer.step(model)`` updates parameters.
  * Compatibility option: keep ``train_backward`` or ``backward_and_step`` as a
    convenience method.

* Fix optimizer timestep semantics. In particular, ensure Adam increments once
  per optimization step, not once per parameter plus once per step.
* Add ``SGDOptimizer`` as the simplest reference optimizer.
* Ensure losses consistently handle batch and scalar output dimensions.
* Add shape/dimension error messages that are useful from GDScript.

Exit criteria:

* A minimal supervised training example is easy to read and explain.
* DQN learner can train without relying on hidden optimizer behavior inside the
  model.

Phase 4 — RL Module Consolidation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Goal:
    Make the RL layer a coherent Godot-first framework built on stable ML
    primitives.

Current issues:

* RL code exists under ``test_project/addons``.
* ``enviornment.gd`` has a typo and should become ``environment.gd``.
* DQN currently needs real model cloning/copying.
* Agent/environment contracts need documentation.

Deliverables:

* Move reusable RL scripts to the canonical addon source location.
* Define stable base contracts for:

  * ``AgentNode``
  * ``RLEnvironment``
  * ``ObservableNode``
  * ``Policy``
  * ``ValueFunction``
  * ``Learner``
  * ``MemoryBuffer``

* Update DQN to use independent online and target networks.
* Add one small, documented DQN example.
* Add one tabular Q-learning example.
* Mark RL as experimental until examples are reliable.

Exit criteria:

* A user can copy one RL example and understand which methods they must override.
* DQN target network behavior is correct.
* RL scripts are not maintained primarily inside ``test_project``.

Phase 5 — Matrix/Linalg Stabilization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Goal:
    Make math primitives reliable enough to treat as stable public API.

Deliverables:

* Audit ``Matrix`` and ``Linalg`` method names and edge-case behavior.
* Validate dimension errors and null ``Ref`` handling.
* Add docs/examples for common operations.
* Decide whether ``Matrix`` is a foundational public type for ML APIs or a
  separate utility for users.

Exit criteria:

* Matrix and linalg docs match implementation.
* Basic operations have regression coverage or executable examples.

Phase 6 — Classical Models and Control Modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Goal:
    Keep useful secondary features without allowing them to distract from the ML
    and RL core.

Deliverables:

* Mark linear regression, decision tree, PID, filters, and movement helpers as
  stable or experimental.
* Move classical models behind the same model lifecycle conventions where
  practical.
* Document control utilities separately from ML training APIs.
* Avoid adding new classical algorithms until existing APIs are coherent.

Exit criteria:

* Users can tell which parts are ML, RL, math, and control.
* Secondary modules do not impose design constraints on neural-network APIs.

Phase 7 — Documentation and Examples Pass
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Goal:
    Make the project understandable to users and future contributors.

Deliverables:

* Getting started guide from clean install to first inference.
* Neural network guide.
* Runtime training guide.
* RL guide.
* Control guide.
* Build-from-source guide.
* API docs updated to match renamed/moved classes.
* Examples organized by module.

Exit criteria:

* A new user can install the addon, create a tiny model, run inference, and run
  one RL example without reading source code.

Runtime Inference and Model Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Runtime model distribution is a first-class product concern: users should be
able to bundle model artifacts with exported games and run inference without
Python or a training environment.

The runtime plan is deliberately split into optional backends:

* Native MLCore models for the smallest Godot-integrated models.
* ONNX Runtime for portable tensor-model inference (issue ``#44``).
* GGUF through an optional llama.cpp backend for local LLM inference (issue
  ``#45``).

A backend-neutral loading and inference contract must be designed before adding
integrations (issue ``#43``). Packaging, artifact discovery, compatibility
metadata, licensing, and smoke tests are tracked in issue ``#46``.

The ONNX and GGUF integrations must remain optional. Projects should not pay
the binary-size or platform-support cost of a backend they do not use. LLM
execution must also be asynchronous or worker-thread based so generation does
not block the Godot main thread.

Phase 8 — Release Hardening
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Goal:
    Prepare predictable releases.

Deliverables:

* Versioning policy.
* Changelog process.
* Release checklist.
* Binary artifact build process.
* Compatibility matrix for Godot/platform/architecture.
* Smoke-test checklist using ``test_project``.

Exit criteria:

* A release can be reproduced from documented steps.
* Users can tell which Godot versions and platforms are supported.

Immediate Backlog
-----------------

The next useful work items, in order:

1. Finalize the public ``NeuralNetworkCore``/``NeuralNetworkNode`` API (issue
   ``#34``).
2. Define model serialization and reset/config lifecycle behavior (issue
   ``#35``).
3. Separate optimizer stepping and add the reference SGD optimizer (issue
   ``#36``).
4. Stabilize RL contracts and DQN target-network behavior (issue ``#37``).
5. Add native Catch2 and headless Godot verification to CI.
6. Create minimal supervised and RL tutorials (issue ``#11``).
7. Define a focused runtime model-evaluation API (issue ``#12``).
8. Keep deferred control/editor ideas out of the critical path until the core
   APIs and examples are stable.
9. Design the backend-neutral runtime and artifact-loading contract (issue
   ``#43``).
10. Add optional ONNX Runtime and GGUF/llama.cpp integrations (issues ``#44``
    and ``#45``).
11. Add model bundling and runtime smoke tests (issue ``#46``).

GitHub Issue Triage Plan
------------------------

When GitHub CLI access is available, issues should be triaged with this process:

1. Export/list all open issues.
2. Close issues that are obsolete, duplicates, or already completed.
3. Add area labels to every remaining issue.
4. Add type labels to every remaining issue.
5. Add priority labels only where useful.
6. Convert broad ideas into roadmap-aligned epics.
7. Break immediate work into small issues that can be completed in one session.
8. Link issues back to roadmap phases.

Suggested milestone structure:

* ``v0.1 - Hygiene and Direction``
* ``v0.2 - Stable NN Core``
* ``v0.3 - RL Experimental Preview``
* ``v0.4 - Docs and Examples``
* ``v1.0 - Stable Godot ML/RL Toolkit``

Operating Model for Infrequent Maintenance
------------------------------------------

Each work session should start with:

1. Read this roadmap.
2. Check the current milestone.
3. Pick one issue that can be completed in the available time.
4. Avoid broad refactors unless they are explicitly roadmap tasks.
5. Update docs or issue comments with any decisions made.

Each work session should end with:

1. Build or test what changed.
2. Update issue status.
3. Leave a short note on what should happen next.
4. Avoid leaving untracked architectural decisions only in chat.

Decision Log
------------

Decisions to make soon:

* Final public name and Godot base type for the neural network model.
* Canonical source location for addon GDScript.
* Whether public training API separates ``backward`` and ``optimizer.step``.
* Minimum supported Godot version.
* Supported platform priority order.

Current recommended decisions:

* Prefer ``RefCounted``/``Resource`` for models instead of ``Node``.
* Keep RL orchestration in GDScript for now.
* Keep C++ focused on math, model execution, losses, optimizers, and model
  lifecycle.
* Treat RL as experimental until DQN examples are correct and documented.
