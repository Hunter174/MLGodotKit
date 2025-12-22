LRNode
======

Linear regression model trained using batch gradient descent.

``LRNode`` implements ordinary least squares linear regression with a mean
squared error (MSE) loss function. The model learns a linear mapping from input
features to a continuous target variable using full-batch gradient descent.

This implementation is designed for simplicity, transparency, and real-time
use inside the Godot Engine via GDExtension.

The interface is loosely inspired by ``sklearn.linear_model.LinearRegression``,
with explicit control over training and learning rate.

---

Overview
--------

- Regression only
- Full-batch gradient descent
- Mean squared error (MSE) loss
- Deterministic training
- No regularization

---

Parameters
----------

``learning_rate`` : float, default=0.01
    Step size used during gradient descent updates.

The learning rate can be modified at any time using
``set_learning_rate``.

---

Methods
-------

Initialization
^^^^^^^^^^^^^^

``initialize(input_size)``
    Initialize model parameters.

    Parameters
        ``input_size`` : int
            Number of input features.

    Notes
        - Weights are initialized randomly.
        - Bias is initialized to zero.
        - This method must be called before training or prediction.

---

Training
^^^^^^^^

``train(inputs, targets, epochs)``
    Train the model using batch gradient descent.

    Parameters
        ``inputs`` : Array
            A 2D array of shape ``(n_samples, n_features)``.

        ``targets`` : Array
            A 1D array of shape ``(n_samples,)`` containing continuous values.

        ``epochs`` : int
            Number of training iterations.

    Notes
        - Training minimizes mean squared error.
        - Loss is printed every 100 epochs.
        - Training is deterministic given identical inputs.

---

Prediction
^^^^^^^^^^

``predict(input)``
    Predict continuous target values.

    Parameters
        ``input`` : Array
            A 2D array of shape ``(n_samples, n_features)``.

    Returns
        ``Array``
            Predicted values of shape ``(n_samples,)``.

---

Configuration
^^^^^^^^^^^^^

``set_learning_rate(lr)``
    Set the learning rate used during training.

---

Algorithm Details
-----------------

- Hypothesis function:
  ``y = Xw + b``

- Loss function:
  Mean squared error (MSE)

- Optimization method:
  Batch gradient descent

- Gradient computation:
  Analytical gradient of MSE with respect to weights and bias

---

Limitations
-----------

- Regression only
- No regularization (L1 / L2)
- No early stopping
- No validation split
- No feature scaling
- Single-output only

---

Examples
--------

Minimal usage from GDScript:

.. code-block:: gdscript

   var lr = LRNode.new()
   lr.initialize(3)
   lr.set_learning_rate(0.01)

   lr.train(X_train, y_train, 500)
   var predictions = lr.predict(X_test)

---

See Also
--------

- ``sklearn.linear_model.LinearRegression``
- ``DTreeNode`` for classification
- ``NNNode`` for nonlinear regression
