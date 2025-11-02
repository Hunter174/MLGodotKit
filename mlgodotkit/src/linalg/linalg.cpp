#include "linalg.h"
#include <godot_cpp/core/class_db.hpp>

namespace godot {

void Linalg::_bind_methods() {
    ClassDB::bind_static_method("Linalg", D_METHOD("add", "A", "B"), &Linalg::add);
    ClassDB::bind_static_method("Linalg", D_METHOD("mult", "A", "B"), &Linalg::mult);
    ClassDB::bind_static_method("Linalg", D_METHOD("transpose", "A"), &Linalg::transpose);
    ClassDB::bind_static_method("Linalg", D_METHOD("inv", "A"), &Linalg::inv);
}

Array Linalg::add(const Array &A, const Array &B) {
    Eigen::MatrixXf mA = Utils::godot_to_eigen(A);
    Eigen::MatrixXf mB = Utils::godot_to_eigen(B);
    if (mA.rows() != mB.rows() || mA.cols() != mB.cols()) {
        Logger::error_raise("Linalg.add(): dimension mismatch");
        return Array();
    }
    return Utils::eigen_to_godot(mA + mB);
}

Array Linalg::mult(const Array &A, const Array &B) {
    Eigen::MatrixXf mA = Utils::godot_to_eigen(A);
    Eigen::MatrixXf mB = Utils::godot_to_eigen(B);
    if (mA.cols() != mB.rows()) {
        Logger::error_raise("Linalg.mult(): inner dimension mismatch");
        return Array();
    }
    return Utils::eigen_to_godot(mA * mB);
}

Array Linalg::transpose(const Array &A) {
    Eigen::MatrixXf mA = Utils::godot_to_eigen(A);
    return Utils::eigen_to_godot(mA.transpose());
}

Array Linalg::inv(const Array &A) {
    Eigen::MatrixXf mA = Utils::godot_to_eigen(A);
    if (mA.rows() != mA.cols()) {
        Logger::error_raise("Linalg.inv(): matrix must be square");
        return Array();
    }
    return Utils::eigen_to_godot(mA.inverse());
}

} // namespace godot
