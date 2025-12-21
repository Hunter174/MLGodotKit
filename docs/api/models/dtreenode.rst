DTreeNode
=========

Decision tree classifier using Gini impurity.

Configuration
-------------

- ``set_max_depth(depth)``
- ``set_min_samples_split(n)``

Training
--------

- ``fit(inputs, targets)``
- ``predict(inputs)``

Notes
-----

- Classification only
- Deterministic splits
- No pruning (yet)
