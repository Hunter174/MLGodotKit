DTreeNode
=========

Decision tree classifier based on Gini impurity.

``DTreeNode`` implements a deterministic, binary decision tree for
classification tasks. The model recursively partitions the feature space
using axis-aligned splits selected by minimizing weighted Gini impurity.

This implementation is designed for educational use, real-time inference,
and integration directly inside the Godot Engine via GDExtension.

The interface and behavior are inspired by ``sklearn.tree.DecisionTreeClassifier``,
with a deliberately reduced feature set.

----

Overview
--------

- Classification only
- Binary splits using ``feature <= threshold``
- Greedy, deterministic split selection
- No pruning or ensemble support
- CPU-only, single-threaded

----

Parameters
----------

``max_depth`` : int, default=10
    The maximum depth of the tree. A depth of 1 corresponds to a single split.

``min_samples_split`` : int, default=2
    The minimum number of samples required to split an internal node.

Both parameters can be set before training using the provided setter methods.

----

Methods
-------

Configuration
^^^^^^^^^^^^^

``set_max_depth(depth)``
    Set the maximum depth of the tree. Values less than 1 are clamped to 1.

``set_min_samples_split(n)``
    Set the minimum number of samples required to split a node.
    Values less than 2 are clamped to 2.

``get_max_depth()``
    Return the configured maximum tree depth.

----

Training
^^^^^^^^

``fit(inputs, targets)``
    Build the decision tree from training data.

    Parameters
        ``inputs`` : Array
            A 2D array of shape ``(n_samples, n_features)``.

        ``targets`` : Array
            A 1D array of integer class labels of shape ``(n_samples,)``.

    Notes
        - Any previously trained tree is discarded.
        - Training is deterministic given identical inputs.
        - All data is converted internally to Eigen matrices.

----

Prediction
^^^^^^^^^^

``predict(inputs)``
    Predict class labels for input samples.

    Parameters
        ``inputs`` : Array
            A 2D array of shape ``(n_samples, n_features)``.

    Returns
        ``Array``
            Predicted integer class labels.

    Notes
        - ``fit`` must be called before prediction.
        - Each sample is evaluated independently by traversing the tree.

----

Algorithm Details
-----------------

- Impurity metric: **Gini impurity**
- Threshold candidates: **unique feature values**
- Split selection: **minimum weighted impurity**
- Leaf prediction: **majority class (deterministic tie-breaking)**

Stopping criteria:
- ``depth >= max_depth``
- ``n_samples < min_samples_split``
- Pure node (all labels identical)
- No valid split found

----

Limitations
-----------

- Classification only (no regression)
- No pruning
- No feature subsampling
- No class weighting
- No missing-value handling
- Not optimized for large datasets

----

Examples
--------

Minimal usage from GDScript:

.. code-block:: gdscript

   var tree = DTreeNode.new()
   tree.set_max_depth(5)
   tree.set_min_samples_split(2)

   tree.fit(X_train, y_train)
   var predictions = tree.predict(X_test)

----

See Also
--------

- ``sklearn.tree.DecisionTreeClassifier``
- ``NNNode`` for neural network-based classification
- ``LRNode`` for linear models
