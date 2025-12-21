LRNode
======

Linear regression model trained with gradient descent.

Usage
-----

.. code-block:: gdscript

   var lr = LRNode.new()
   lr.initialize(3)
   lr.train(X, y, 500)

Methods
-------

- ``initialize(input_size)``
- ``predict(input)``
- ``train(inputs, targets, epochs)``
- ``set_learning_rate(lr)``
