#include "matrix.h"
#include "utility/logger.h"
#include "utility/utils.h"
#include <godot_cpp/core/class_db.hpp>

namespace godot {

void Matrix::_bind_methods() {
    ClassDB::bind_static_method("Matrix", D_METHOD("zeros", "rows", "cols"), &Matrix::zeros);
    ClassDB::bind_static_method("Matrix", D_METHOD("ones", "rows", "cols"), &Matrix::ones);
    ClassDB::bind_static_method("Matrix", D_METHOD("identity", "n"), &Matrix::identity);
    ClassDB::bind_static_method("Matrix", D_METHOD("from_array", "A"), &Matrix::from_array);

    ClassDB::bind_static_method("Matrix", D_METHOD("from_vector2", "v", "column"), &Matrix::from_vector2, DEFVAL(true));
    ClassDB::bind_static_method("Matrix", D_METHOD("from_vector3", "v", "column"), &Matrix::from_vector3, DEFVAL(true));
    ClassDB::bind_static_method("Matrix", D_METHOD("from_vector4", "v", "column"), &Matrix::from_vector4, DEFVAL(true));

    ClassDB::bind_method(D_METHOD("to_vector2"), &Matrix::to_vector2);
    ClassDB::bind_method(D_METHOD("to_vector3"), &Matrix::to_vector3);
    ClassDB::bind_method(D_METHOD("to_vector4"), &Matrix::to_vector4);

    ClassDB::bind_method(D_METHOD("mul_vector2", "v"), &Matrix::mul_vector2);
    ClassDB::bind_method(D_METHOD("mul_vector3", "v"), &Matrix::mul_vector3);
    ClassDB::bind_method(D_METHOD("mul_vector4", "v"), &Matrix::mul_vector4);

    ClassDB::bind_method(D_METHOD("to_array"), &Matrix::to_array);
    ClassDB::bind_method(D_METHOD("rows"), &Matrix::rows);
    ClassDB::bind_method(D_METHOD("cols"), &Matrix::cols);

    ClassDB::bind_method(D_METHOD("transpose"), &Matrix::transpose);
    ClassDB::bind_method(D_METHOD("matmul", "B"), &Matrix::matmul);
    ClassDB::bind_method(D_METHOD("inverse"), &Matrix::inverse);

    ClassDB::bind_method(D_METHOD("det"), &Matrix::det);
    ClassDB::bind_method(D_METHOD("trace"), &Matrix::trace);
    ClassDB::bind_method(D_METHOD("norm"), &Matrix::norm);

    ClassDB::bind_method(D_METHOD("get", "i", "j"), &Matrix::get);
    ClassDB::bind_method(D_METHOD("set", "i", "j", "value"), &Matrix::set);

    ClassDB::bind_method(D_METHOD("copy"), &Matrix::copy);
    ClassDB::bind_method(D_METHOD("equals", "other", "eps"), &Matrix::equals);
    ClassDB::bind_method(D_METHOD("info"), &Matrix::info);
    ClassDB::bind_method(D_METHOD("_to_string"), &Matrix::_to_string);
}

void Matrix::_init() {}

Matrix::Matrix() {}

Ref<Matrix> Matrix::zeros(int rows, int cols) {
    Ref<Matrix> out = memnew(Matrix());
    out->m = Eigen::MatrixXf::Zero(rows, cols);
    return out;
}

Ref<Matrix> Matrix::ones(int rows, int cols) {
    Ref<Matrix> out = memnew(Matrix());
    out->m = Eigen::MatrixXf::Ones(rows, cols);
    return out;
}

Ref<Matrix> Matrix::identity(int n) {
    Ref<Matrix> out = memnew(Matrix());
    out->m = Eigen::MatrixXf::Identity(n, n);
    return out;
}

Ref<Matrix> Matrix::from_array(const Array &A) {
    Ref<Matrix> out = memnew(Matrix());
    out->m = Utils::godot_to_eigen(A);
    return out;
}

Ref<Matrix> Matrix::from_vector2(const Vector2 &v, bool column) {
    Ref<Matrix> out = memnew(Matrix());
    out->m = column ? Eigen::MatrixXf(2,1) : Eigen::MatrixXf(1,2);
    out->m << v.x, v.y;
    return out;
}

Ref<Matrix> Matrix::from_vector3(const Vector3 &v, bool column) {
    Ref<Matrix> out = memnew(Matrix());
    out->m = column ? Eigen::MatrixXf(3,1) : Eigen::MatrixXf(1,3);
    out->m << v.x, v.y, v.z;
    return out;
}

Ref<Matrix> Matrix::from_vector4(const Vector4 &v, bool column) {
    Ref<Matrix> out = memnew(Matrix());
    out->m = column ? Eigen::MatrixXf(4,1) : Eigen::MatrixXf(1,4);
    out->m << v.x, v.y, v.z, v.w;
    return out;
}

Vector3 Matrix::mul_vector3(const Vector3 &v) const {
    if (m.rows() != 3 || m.cols() != 3) {
        Logger::error_raise("Matrix.mul_vector3(): matrix must be 3x3");
        return Vector3();
    }
    Eigen::Vector3f r = m * Eigen::Vector3f(v.x, v.y, v.z);
    return Vector3(r.x(), r.y(), r.z());
}

Vector2 Matrix::mul_vector2(const Vector2 &v) const {
    if (m.rows() != 2 || m.cols() != 2) {
        Logger::error_raise("Matrix.mul_vector2(): matrix must be 2x2");
        return Vector2();
    }
    Eigen::Vector2f r = m * Eigen::Vector2f(v.x, v.y);
    return Vector2(r.x(), r.y());
}

Vector4 Matrix::mul_vector4(const Vector4 &v) const {
    if (m.rows() != 4 || m.cols() != 4) {
        Logger::error_raise("Matrix.mul_vector4(): matrix must be 4x4");
        return Vector4();
    }
    Eigen::Vector4f r = m * Eigen::Vector4f(v.x, v.y, v.z, v.w);
    return Vector4(r.x(), r.y(), r.z(), r.w());
}

Vector3 Matrix::to_vector3() const {
    if (!((m.rows() == 3 && m.cols() == 1) || (m.rows() == 1 && m.cols() == 3))) {
        Logger::error_raise("Matrix.to_vector3(): shape must be 3x1 or 1x3");
        return Vector3();
    }
    return Vector3(m(0), m(1), m(2));
}

Vector2 Matrix::to_vector2() const {
    if (!((m.rows() == 2 && m.cols() == 1) || (m.rows() == 1 && m.cols() == 2))) {
        Logger::error_raise("Matrix.to_vector2(): shape must be 2x1 or 1x2");
        return Vector2();
    }
    return Vector2(m(0), m(1));
}

Vector4 Matrix::to_vector4() const {
    if (!((m.rows() == 4 && m.cols() == 1) || (m.rows() == 1 && m.cols() == 4))) {
        Logger::error_raise("Matrix.to_vector4(): shape must be 4x1 or 1x4");
        return Vector4();
    }
    return Vector4(m(0), m(1), m(2), m(3));
}

Array Matrix::to_array() const {
    return Utils::eigen_to_godot(m);
}

int Matrix::rows() const { return m.rows(); }
int Matrix::cols() const { return m.cols(); }

Ref<Matrix> Matrix::transpose() const {
    Ref<Matrix> out = memnew(Matrix());
    out->m = m.transpose();
    return out;
}

Ref<Matrix> Matrix::matmul(const Ref<Matrix> &B) const {
    Ref<Matrix> out = memnew(Matrix());
    out->m = m * B->m;
    return out;
}

Ref<Matrix> Matrix::inverse() const {
    Ref<Matrix> out = memnew(Matrix());
    out->m = m.inverse();
    return out;
}

float Matrix::det() const { return m.determinant(); }
float Matrix::trace() const { return m.trace(); }
float Matrix::norm() const { return m.norm(); }

float Matrix::get(int i, int j) const { return m(i, j); }
void Matrix::set(int i, int j, float value) { m(i, j) = value; }

Ref<Matrix> Matrix::copy() const {
    Ref<Matrix> out = memnew(Matrix());
    out->m = m;
    return out;
}

bool Matrix::equals(const Ref<Matrix> &other, float eps) const {
    return (m - other->m).cwiseAbs().maxCoeff() <= eps;
}

Dictionary Matrix::info() const {
    Dictionary d;
    d["rows"] = m.rows();
    d["cols"] = m.cols();
    d["norm"] = m.norm();
    return d;
}

String Matrix::_to_string() const {
    String s = "Matrix(" + itos(m.rows()) + "x" + itos(m.cols()) + ")\n";
    for (int i = 0; i < m.rows(); ++i) {
        s += "[ ";
        for (int j = 0; j < m.cols(); ++j) {
            s += String::num(m(i, j), 4);
            if (j + 1 < m.cols()) s += ", ";
        }
        s += " ]\n";
    }
    return s;
}

} // namespace godot