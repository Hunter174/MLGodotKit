Matrix
======

The ``Matrix`` class is a lightweight wrapper around Eigen matrices,
exposed to Godot via GDExtension.

Construction
------------

.. code-block:: gdscript

   var A = Matrix.zeros(3, 3)
   var I = Matrix.identity(3)

Static Methods
--------------

- ``zeros(rows, cols)``
- ``ones(rows, cols)``
- ``identity(n)``
- ``from_array(A)``
- ``from_vector2(v, column := true)``
- ``from_vector3(v, column := true)``
- ``from_vector4(v, column := true)``

Instance Methods
----------------

Linear algebra:
- ``transpose()``
- ``matmul(B)``
- ``inverse()``
- ``det()``
- ``trace()``
- ``norm()``

Accessors:
- ``rows()``
- ``cols()``
- ``get(i, j)``
- ``set(i, j, value)``

Conversion:
- ``to_array()``
- ``to_vector2()``
- ``to_vector3()``
- ``to_vector4()``

Utilities:
- ``copy()``
- ``equals(other, eps := 1e-6)``
- ``info()``
