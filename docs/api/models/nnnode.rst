NNNode
======

Feed-forward neural network with manual backpropagation.

Layer Definition
----------------

Layers are defined via dictionaries:

.. code-block:: gdscript

   nn.layers = [
     {"input_size": 4, "output_size": 16, "activation": "relu"},
     {"input_size": 16, "output_size": 2, "activation": "linear"}
   ]

Training Interface
------------------

- ``forward(input)``
- ``backward(error)``
- ``copy_weights(source)``
- ``model_summary()``

Hyperparameters
---------------

- ``learning_rate``
- ``batch_size``
- ``verbosity``
