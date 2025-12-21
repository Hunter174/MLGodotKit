Linalg
======

The ``Linalg`` class provides stateless linear algebra solvers
built on Eigen.

Solvers
-------

- ``solve(A, b)``
  Solves a square linear system.

- ``least_squares(A, b)``
  Computes a least-squares solution.

- ``pinv(A)``
  Computes the Moore–Penrose pseudoinverse.

Decompositions
--------------

Each decomposition returns a ``Dictionary`` of matrices.

- ``qr(A)`` → ``{Q, R}``
- ``svd(A)`` → ``{U, S, V}``
- ``eig(A)`` → ``{values, vectors}``
- ``lu(A)`` → ``{L, U, P}``

Example
-------

.. code-block:: gdscript

   var res = Linalg.qr(A)
   var Q = res["Q"]
   var R = res["R"]
