Linalg
======

Stateless linear algebra solvers and matrix decompositions.

The ``Linalg`` class provides a collection of static linear algebra routines
backed by the Eigen C++ library. All methods operate on ``Matrix`` objects and
do not maintain internal state.

This module is intended as a foundational utility for numerical computation,
optimization, and machine learning workflows inside the Godot Engine.

The API is inspired by ``numpy.linalg`` and ``scipy.linalg``, with a simplified,
engine-friendly interface.

----

Overview
--------

- Stateless functional API
- Eigen-backed numerical routines
- Designed for small to medium matrices
- Suitable for real-time and scripting use
- No GPU acceleration

----

Solvers
-------

``solve(A, b)``
    Solve a square linear system.

    Parameters
        ``A`` : Matrix
            Square coefficient matrix of shape ``(n, n)``.

        ``b`` : Matrix
            Right-hand side matrix of shape ``(n, k)``.

    Returns
        ``Matrix``
            Solution matrix ``x`` satisfying ``Ax = b``.

    Notes
        - Uses LU decomposition with partial pivoting.
        - Raises an error if dimensions are incompatible.

----

``least_squares(A, b)``
    Compute a least-squares solution to an overdetermined system.

    Parameters
        ``A`` : Matrix
            Coefficient matrix of shape ``(m, n)``.

        ``b`` : Matrix
            Right-hand side matrix of shape ``(m, k)``.

    Returns
        ``Matrix``
            Least-squares solution minimizing ``||Ax - b||₂``.

    Notes
        - Uses column-pivoted QR decomposition.
        - Suitable for full-rank problems.

----

``pinv(A)``
    Compute the Moore–Penrose pseudoinverse of a matrix.

    Parameters
        ``A`` : Matrix
            Input matrix of shape ``(m, n)``.

    Returns
        ``Matrix``
            Pseudoinverse matrix of shape ``(n, m)``.

    Notes
        - Computed via singular value decomposition (SVD).
        - Small singular values are thresholded for numerical stability.

----

Decompositions
--------------

All decomposition methods return a ``Dictionary`` containing the resulting
matrices.

``qr(A)`` → ``{"Q", "R"}``
    QR decomposition of ``A``.

    - ``Q``: orthonormal matrix
    - ``R``: upper triangular matrix

----

``svd(A)`` → ``{"U", "S", "V"}``
    Singular value decomposition.

    - ``U``: left singular vectors
    - ``S``: diagonal matrix of singular values
    - ``V``: right singular vectors

    Notes
        - Thin SVD is used for efficiency.

----

``eig(A)`` → ``{"values", "vectors"}``
    Eigenvalue decomposition of a symmetric matrix.

    Notes
        - ``A`` must be square and self-adjoint.
        - Eigenvalues are real-valued.

----

``lu(A)`` → ``{"L", "U", "P"}``
    LU decomposition with partial pivoting.

    - ``L``: unit lower triangular matrix
    - ``U``: upper triangular matrix
    - ``P``: permutation matrix

----

Error Handling
--------------

- Dimension mismatches raise runtime errors.
- Invalid inputs (e.g., null matrices) are rejected.
- Decomposition failures are reported explicitly.

----

Examples
--------

Solve a linear system:

.. code-block:: gdscript

   var x = Linalg.solve(A, b)

Least-squares fit:

.. code-block:: gdscript

   var x = Linalg.least_squares(A, b)

QR decomposition:

.. code-block:: gdscript

   var res = Linalg.qr(A)
   var Q = res["Q"]
   var R = res["R"]

----
