NNNode
======

Feed-forward neural network with explicit forward and backward passes.

``NNNode`` implements a fully-connected neural network composed of sequential
layers with configurable activation functions. Training logic is intentionally
left explicit: users are expected to compute loss functions externally and
provide gradients directly to the network.

This design prioritizes transparency, flexibility, and real-time integration
inside the Godot Engine over automation or convenience abstractions.

The interface is inspired by low-level deep learning frameworks rather than
high-level APIs such as ``sklearn`` or ``Keras``.

----

Overview
--------

- Fully-connected feed-forward architecture
- Manual forward and backward passes
- Explicit gradient input (loss handled externally)
- Configurable activation functions per layer
- Designed for real-time and interactive learning scenarios

----

Model Definition
----------------

Networks are defined as an ordered sequence of layers. Layers are **not**
instantiated directly by the user; instead, they are configured via dictionaries
or added incrementally using ``add_layer``.

Each layer specifies:
- Input size
- Output size
- Activation function (by name)

Example layer configuration:

.. code-block:: gdscript

   nn.layers = [
     {"input_size": 4, "output_size": 16, "activation": "relu"},
     {"input_size": 16, "output_size": 2, "activation": "linear"}
   ]

Setting the ``layers`` property automatically rebuilds the model.

----

Supported Activations
---------------------

Activation functions are selected by string identifier. Internally, these map
to optimized C++ implementations.

Supported activations include:

- ``"linear"``
- ``"relu"``
- ``"leaky_relu"``
- ``"sigmoid"``
- ``"tanh"``

If an unknown activation name is provided, the layer defaults to ``relu``.

----

Parameters
----------

``learning_rate`` : float, default=0.01
    Learning rate used by all layers during weight updates.

``batch_size`` : int, default=1
    Batch size used when interpreting input arrays.

``verbosity`` : int, default=0
    Controls logging verbosity:

    - 0: silent
    - 1–3: increasing diagnostic detail

All parameters are inspector-visible and can be modified at runtime.

----

Methods
-------

Model Construction
^^^^^^^^^^^^^^^^^^

``add_layer(input_size, output_size, activation)``
    Append a new fully-connected layer to the model.

``set_layers(layers)``
    Define the model using an array of layer configuration dictionaries.
    Automatically rebuilds the network.

``build_model()``
    Rebuild the internal layer stack from the current configuration.

``model_summary()``
    Print a summary of the network architecture, including layer sizes and
    activation functions.

----

Forward Pass
^^^^^^^^^^^^

``forward(input)``
    Perform a forward pass through the network.

    Parameters
        ``input`` : Array
            A 2D array of shape ``(batch_size, input_dim)``.

    Returns
        ``Array``
            Network output as a 2D array.

    Notes
        - Input dimensions are validated against the first layer.
        - No output activation is applied automatically.
        - ``forward`` must be called before ``backward``.

----

Backward Pass
^^^^^^^^^^^^^

``backward(error)``
    Perform backpropagation given the gradient of the loss with respect to the
    network output.

    Parameters
        ``error`` : Array
            Gradient of the loss function with respect to the network output.

    Notes
        - Loss functions are not implemented internally.
        - Global gradient norm clipping is applied for stability.
        - Weight updates occur immediately after backpropagation.

----

Utilities
^^^^^^^^^

``copy_weights(source)``
    Copy weights and biases from another ``NNNode`` with identical architecture.

----

Algorithm Details
-----------------

- Weight initialization:
  - He initialization for ReLU-based activations
  - Xavier initialization for others

- Optimization:
  - Gradient descent with momentum
  - L2 weight decay
  - Global gradient norm clipping

- Backpropagation:
  - Layer-wise chain rule
  - Analytical activation derivatives
  - No automatic loss computation

----

Design Philosophy
-----------------

``NNNode`` deliberately avoids high-level training abstractions. This allows:

- Custom loss functions
- Reinforcement learning–style updates
- Online and non-epoch-based training
- Tight control over learning dynamics

This makes ``NNNode`` suitable for:
- Reinforcement learning agents
- Real-time adaptive behaviors
- Educational and experimental workflows

----

Limitations
-----------

- No built-in loss functions
- No automatic batching or dataset handling
- No convolutional or recurrent layers
- No serialization or checkpointing
- No GPU acceleration

----

Examples
--------

Minimal usage with an external loss:

.. code-block:: gdscript

   var nn = NNNode.new()
   nn.set_learning_rate(0.01)

   nn.add_layer(2, 4, "relu")
   nn.add_layer(4, 1, "sigmoid")

   var output = nn.forward(x)
   var error = output[0][0] - target
   nn.backward([[2.0 * error]])

----

See Also
--------

- ``LRNode`` for linear models
- ``DTreeNode`` for tree-based classification
- Reinforcement learning modules for agent-based training
