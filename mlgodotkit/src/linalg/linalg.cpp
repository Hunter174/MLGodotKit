#include "linalg.h"
#include <godot_cpp/core/class_db.hpp>

namespace godot {

void Linalg::_bind_methods() {
    ClassDB::bind_static_method("Linalg", D_METHOD("add", "A", "B"), &Linalg::add);
    ClassDB::bind_static_method("Linalg", D_METHOD("mult", "A", "B"), &Linalg::mult);
    ClassDB::bind_static_method("Linalg", D_METHOD("transpose", "A"), &Linalg::transpose);
    ClassDB::bind_static_method("Linalg", D_METHOD("inv", "A"), &Linalg::inv);
	ClassDB::bind_static_method("Linalg", D_METHOD("scalar_mult", "A", "s"), &Linalg::scalar_mult);
    ClassDB::bind_static_method("Linalg", D_METHOD("elem_mult", "A", "B"), &Linalg::elem_mult);
    ClassDB::bind_static_method("Linalg", D_METHOD("det", "A"), &Linalg::det);
    ClassDB::bind_static_method("Linalg", D_METHOD("trace", "A"), &Linalg::trace);
    ClassDB::bind_static_method("Linalg", D_METHOD("sum", "A"), &Linalg::sum);
	ClassDB::bind_static_method("Linalg", D_METHOD("mean", "A"), &Linalg::mean);
	ClassDB::bind_static_method("Linalg", D_METHOD("min", "A"), &Linalg::min);
	ClassDB::bind_static_method("Linalg", D_METHOD("max", "A"), &Linalg::max);
	ClassDB::bind_static_method("Linalg", D_METHOD("norm", "A"), &Linalg::norm);
	ClassDB::bind_static_method("Linalg", D_METHOD("row_sum", "A"), &Linalg::row_sum);
	ClassDB::bind_static_method("Linalg", D_METHOD("row_mean", "A"), &Linalg::row_mean);
	ClassDB::bind_static_method("Linalg", D_METHOD("row_norm", "A"), &Linalg::row_norm);
	ClassDB::bind_static_method("Linalg", D_METHOD("col_sum", "A"), &Linalg::col_sum);
	ClassDB::bind_static_method("Linalg", D_METHOD("col_mean", "A"), &Linalg::col_mean);
	ClassDB::bind_static_method("Linalg", D_METHOD("col_norm", "A"), &Linalg::col_norm);
    ClassDB::bind_static_method("Linalg", D_METHOD("diag", "A"), &Linalg::diag);
	ClassDB::bind_static_method("Linalg", D_METHOD("block", "A", "r0", "c0", "r", "c"), &Linalg::block);
	ClassDB::bind_static_method("Linalg", D_METHOD("reshape", "A", "rows", "cols"), &Linalg::reshape);
	ClassDB::bind_static_method("Linalg", D_METHOD("get_row", "A", "i"), &Linalg::get_row);
	ClassDB::bind_static_method("Linalg", D_METHOD("get_col", "A", "j"), &Linalg::get_col);

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

Array Linalg::scalar_mult(const Array &A, const float s) {
    Eigen::MatrixXf mA = Utils::godot_to_eigen(A);
    return Utils::eigen_to_godot(mA * s);
}


Array Linalg::elem_mult(const Array &A, const Array &B) {
    Eigen::MatrixXf mA = Utils::godot_to_eigen(A);
    Eigen::MatrixXf mB = Utils::godot_to_eigen(B);

    if (mA.rows() != mB.rows() || mA.cols() != mB.cols()) {
        Logger::error_raise("Linalg.elem_mult(): dimension mismatch");
        return Array();
    }

    return Utils::eigen_to_godot(mA.cwiseProduct(mB));
}

float Linalg::det(const Array &A) {
    Eigen::MatrixXf mA = Utils::godot_to_eigen(A);

    if (mA.rows() != mA.cols()) {
        Logger::error_raise("Linalg.det(): matrix must be square");
        return 0.0f;
    }

    return mA.determinant();
}

float Linalg::trace(const Array &A) {
    Eigen::MatrixXf mA = Utils::godot_to_eigen(A);

    if (mA.rows() != mA.cols()) {
        Logger::error_raise("Linalg.trace(): matrix must be square");
        return 0.0f;
    }

    return mA.trace();
}

float Linalg::sum(const Array &A) {
    Eigen::MatrixXf mA = Utils::godot_to_eigen(A);
    return mA.sum();
}

float Linalg::mean(const Array &A) {
    Eigen::MatrixXf mA = Utils::godot_to_eigen(A);
    return mA.mean();
}

float Linalg::min(const Array &A) {
    Eigen::MatrixXf mA = Utils::godot_to_eigen(A);
    return mA.minCoeff();
}

float Linalg::max(const Array &A) {
    Eigen::MatrixXf mA = Utils::godot_to_eigen(A);
    return mA.maxCoeff();
}

//L2 Frobenius Norm
float Linalg::norm(const Array &A) {
    Eigen::MatrixXf mA = Utils::godot_to_eigen(A);
    return mA.norm();
}

Array Linalg::row_sum(const Array &A) {
    Eigen::MatrixXf mA = Utils::godot_to_eigen(A);
    Eigen::VectorXf out = mA.rowwise().sum();
    return Utils::eigen_to_godot(out);
}

Array Linalg::row_mean(const Array &A) {
    Eigen::MatrixXf mA = Utils::godot_to_eigen(A);
    Eigen::VectorXf out = mA.rowwise().mean();
    return Utils::eigen_to_godot(out);
}

Array Linalg::row_norm(const Array &A) {
    Eigen::MatrixXf mA = Utils::godot_to_eigen(A);
    Eigen::VectorXf out(mA.rows());

    for (int i = 0; i < mA.rows(); i++)
        out(i) = mA.row(i).norm();

    return Utils::eigen_to_godot(out);
}

Array Linalg::col_sum(const Array &A) {
    Eigen::MatrixXf mA = Utils::godot_to_eigen(A);
    Eigen::RowVectorXf out = mA.colwise().sum();
    return Utils::eigen_to_godot(out);
}

Array Linalg::col_mean(const Array &A) {
    Eigen::MatrixXf mA = Utils::godot_to_eigen(A);
    Eigen::RowVectorXf out = mA.colwise().mean();
    return Utils::eigen_to_godot(out);
}

Array Linalg::col_norm(const Array &A) {
    Eigen::MatrixXf mA = Utils::godot_to_eigen(A);
    Eigen::RowVectorXf out(mA.cols());

    for (int j = 0; j < mA.cols(); j++)
        out(j) = mA.col(j).norm();

    return Utils::eigen_to_godot(out);
}

Array Linalg::diag(const Array &A) {
    Eigen::MatrixXf mA = Utils::godot_to_eigen(A);
    Eigen::VectorXf d = mA.diagonal();
    return Utils::eigen_to_godot(d);
}

Array Linalg::block(const Array &A, int r0, int c0, int r, int c) {
    Eigen::MatrixXf mA = Utils::godot_to_eigen(A);

    if (r0 < 0 || c0 < 0 || r <= 0 || c <= 0 ||
        r0 + r > mA.rows() || c0 + c > mA.cols()) {
        Logger::error_raise("Linalg.block(): invalid block indices");
        return Array();
    }

    return Utils::eigen_to_godot(mA.block(r0, c0, r, c));
}

Array Linalg::reshape(const Array &A, int rows, int cols) {
    Eigen::MatrixXf mA = Utils::godot_to_eigen(A);

    if (rows * cols != mA.size()) {
        Logger::error_raise("Linalg.reshape(): total size mismatch");
        return Array();
    }

    Eigen::MatrixXf out = Eigen::Map<Eigen::MatrixXf>(mA.data(), rows, cols);
    return Utils::eigen_to_godot(out);
}

Array Linalg::get_row(const Array &A, int i) {
    Eigen::MatrixXf mA = Utils::godot_to_eigen(A);

    if (i < 0 || i >= mA.rows()) {
        Logger::error_raise("Linalg.get_row(): index out of range");
        return Array();
    }

    return Utils::eigen_to_godot(mA.row(i));
}

Array Linalg::get_col(const Array &A, int j) {
    Eigen::MatrixXf mA = Utils::godot_to_eigen(A);

    if (j < 0 || j >= mA.cols()) {
        Logger::error_raise("Linalg.get_col(): index out of range");
        return Array();
    }

    return Utils::eigen_to_godot(mA.col(j));
}

} // namespace godot
