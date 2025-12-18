#ifndef MLGODOTKIT_LINALG_H
#define MLGODOTKIT_LINALG_H

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>

#include <Eigen/Dense>

#include "utility/logger.h"
#include "utility/utils.h"

namespace godot {

class Linalg : public RefCounted {
    GDCLASS(Linalg, RefCounted);

protected:
    static void _bind_methods();

public:
    // ======================================================
    // Core matrix operations
    // ======================================================
    static Array add(const Array &A, const Array &B);
    static Array mult(const Array &A, const Array &B);
    static Array transpose(const Array &A);
    static Array inv(const Array &A);
    static Array scalar_mult(const Array &A, float s);
    static Array elem_mult(const Array &A, const Array &B);

    // ======================================================
    // Matrix invariants
    // ======================================================
    static float det(const Array &A);
    static float trace(const Array &A);

    // ======================================================
    // Global reductions
    // ======================================================
    static float sum(const Array &A);
    static float mean(const Array &A);
    static float min(const Array &A);
    static float max(const Array &A);
    static float norm(const Array &A); // Frobenius (L2)

    // ======================================================
    // Row-wise / column-wise reductions
    // ======================================================
    static Array row_sum(const Array &A);
    static Array row_mean(const Array &A);
    static Array row_norm(const Array &A);

    static Array col_sum(const Array &A);
    static Array col_mean(const Array &A);
    static Array col_norm(const Array &A);

    // ======================================================
    // Indexing & slicing
    // ======================================================
    static Array diag(const Array &A);
    static Array block(const Array &A, int r0, int c0, int r, int c);
    static Array reshape(const Array &A, int rows, int cols);
    static Array get_row(const Array &A, int i);
    static Array get_col(const Array &A, int j);

    // ======================================================
    // Vector operations
    // ======================================================
    static float dot(const Array &a, const Array &b);
    static Array normalize(const Array &a);
    static Array cross(const Array &a, const Array &b);

    // ======================================================
    // Solvers
    // ======================================================
    static Array solve(const Array &A, const Array &b);
    static Array least_squares(const Array &A, const Array &b);
    static Array pinv(const Array &A);

    // ======================================================
    // Decompositions
    // ======================================================
    static Dictionary qr(const Array &A);
    static Dictionary svd(const Array &A);
    static Dictionary eig(const Array &A); // symmetric / self-adjoint
    static Dictionary lu(const Array &A);

    // ======================================================
    // Matrix constructors
    // ======================================================
    static Array zeros(int rows, int cols);
    static Array ones(int rows, int cols);
    static Array full(int rows, int cols, float value);
    static Array eye(int n);
    static Array rand(int rows, int cols);
    static Array randn(int rows, int cols);
};

} // namespace godot

#endif // MLGODOTKIT_LINALG_H