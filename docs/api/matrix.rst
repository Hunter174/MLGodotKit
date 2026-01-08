Matrix
======

Dense numerical matrix container.

The ``Matrix`` class is a lightweight wrapper around a dense, floating-point
matrix backed internally by the Eigen C++ library. It provides basic matrix
construction, linear algebra operations, and interoperability with Godot
built-in vector types.

``Matrix`` is the primary numerical container used throughout MLGodotKit and
serves as the input and output type for the ``Linalg`` module.

----

Overview
--------

- Dense float matrix storage
- Immutable-by-convention functional operations
- Explicit shape validation
- Designed for small to medium matrices
- CPU-only, single-precision floats

----

Construction
------------

Matrices are created using static factory methods.

.. code-block:: gdscript

   var A = Matrix.zeros(3, 3)
   var B = Matrix.ones(2, 4)
   var I = Matrix.identity(3)

Matrices may also be constructed from Godot arrays or vector types.

----

Static Methods
--------------

``zeros(rows, cols)``
    Create a matrix filled with zeros.

``ones(rows, cols)``
    Create a matrix filled with ones.

``identity(n)``
    Create an ``n × n`` identity matrix.

``from_array(A)``
    Construct a matrix from a 2D Godot ``Array``.

``from_vector2(v, column=true)``
``from_vector3(v, column=true)``
``from_vector4(v, column=true)``
    Construct a matrix from a Godot vector.

    If ``column`` is ``true``, the result is a column vector.
    Otherwise, a row vector is created.

----

Shape and Access
----------------

``rows()``
    Return the number of rows.

``cols()``
    Return the number of columns.

``get(i, j)``
    Return the element at position ``(i, j)``.

``set(i, j, value)``
    Set the element at position ``(i, j)``.

----

Linear Algebra Operations
-------------------------

``transpose()``
    Return the transpose of the matrix.

``matmul(B)``
    Matrix–matrix multiplication.

``inverse()``
    Return the matrix inverse.

    Notes
        - The matrix must be square and invertible.

``det()``
    Compute the determinant.

``trace()``
    Compute the matrix trace.

``norm()``
    Compute the Frobenius norm.

----

Vector Interoperability
-----------------------

``mul_vector2(v)``
``mul_vector3(v)``
``mul_vector4(v)``
    Multiply the matrix by a Godot vector.

    Notes
        - Matrix dimensions must match the vector size.
        - Raises an error on shape mismatch.

``to_vector2()``
``to_vector3()``
``to_vector4()``
    Convert a row or column matrix into a Godot vector.

    Notes
        - Matrix must have compatible shape (e.g. ``3×1`` or ``1×3``).

----

Conversion
----------

``to_array()``
    Convert the matrix to a 2D Godot ``Array``.

``copy()``
    Create a deep copy of the matrix.

----

Comparison and Introspection
----------------------------

``equals(other, eps=1e-6)``
    Test approximate equality with another matrix.

``info()``
    Return a ``Dictionary`` containing:

    - ``rows``
    - ``cols``
    - ``norm``

``_to_string()``
    Return a formatted string representation of the matrix.

----

Error Handling
--------------

- Shape mismatches raise runtime errors.
- Invalid vector conversions are rejected.
- No implicit broadcasting is performed.

----

Examples
--------

Matrix multiplication:

.. code-block:: gdscript

   var C = A.matmul(B)

Vector transformation:

.. code-block:: gdscript

   var R = Matrix.identity(3)
   var v2 = R.mul_vector3(v1)

----