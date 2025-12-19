#include "linalg.h"
#include "utility/logger.h"
#include "utility/utils.h"
#include <godot_cpp/core/class_db.hpp>
#include <Eigen/Dense>

namespace godot {

void Linalg::_bind_methods() {
    ClassDB::bind_static_method("Linalg", D_METHOD("solve", "A", "b"), &Linalg::solve);
    ClassDB::bind_static_method("Linalg", D_METHOD("least_squares", "A", "b"), &Linalg::least_squares);
    ClassDB::bind_static_method("Linalg", D_METHOD("pinv", "A"), &Linalg::pinv);

    ClassDB::bind_static_method("Linalg", D_METHOD("qr", "A"), &Linalg::qr);
    ClassDB::bind_static_method("Linalg", D_METHOD("svd", "A"), &Linalg::svd);
    ClassDB::bind_static_method("Linalg", D_METHOD("eig", "A"), &Linalg::eig);
    ClassDB::bind_static_method("Linalg", D_METHOD("lu", "A"), &Linalg::lu);
}

Ref<Matrix> Linalg::solve(const Ref<Matrix> &A, const Ref<Matrix> &b) {
    if (A.is_null() || b.is_null()) {
        Logger::error_raise("Linalg.solve(): null input");
        return Ref<Matrix>();
    }

    const auto &mA = A->eigen();
    const auto &mb = b->eigen();

    if (mA.rows() != mA.cols() || mA.rows() != mb.rows()) {
        Logger::error_raise("Linalg.solve(): dimension mismatch");
        return Ref<Matrix>();
    }

    Ref<Matrix> out = memnew(Matrix());
    out->eigen() = mA.partialPivLu().solve(mb);
    return out;
}

Ref<Matrix> Linalg::least_squares(const Ref<Matrix> &A, const Ref<Matrix> &b) {
    if (A.is_null() || b.is_null()) {
        Logger::error_raise("Linalg.least_squares(): null input");
        return Ref<Matrix>();
    }

    const auto &mA = A->eigen();
    const auto &mb = b->eigen();

    if (mA.rows() != mb.rows()) {
        Logger::error_raise("Linalg.least_squares(): dimension mismatch");
        return Ref<Matrix>();
    }

    Ref<Matrix> out = memnew(Matrix());
    out->eigen() = mA.colPivHouseholderQr().solve(mb);
    return out;
}

Ref<Matrix> Linalg::pinv(const Ref<Matrix> &A) {
    if (A.is_null()) {
        Logger::error_raise("Linalg.pinv(): null input");
        return Ref<Matrix>();
    }

    const auto &m = A->eigen();

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(
        m, Eigen::ComputeThinU | Eigen::ComputeThinV
    );

    float tol = 1e-6f * std::max(m.rows(), m.cols()) *
                svd.singularValues().array().abs().maxCoeff();

    Eigen::VectorXf inv_s = svd.singularValues();
    for (int i = 0; i < inv_s.size(); ++i)
        inv_s(i) = (inv_s(i) > tol) ? 1.0f / inv_s(i) : 0.0f;

    Ref<Matrix> out = memnew(Matrix());
    out->eigen() =
        svd.matrixV() * inv_s.asDiagonal() * svd.matrixU().transpose();

    return out;
}

Dictionary Linalg::qr(const Ref<Matrix> &A) {
    const auto &m = A->eigen();

    Eigen::HouseholderQR<Eigen::MatrixXf> qr(m);
    Eigen::MatrixXf Q = qr.householderQ() *
        Eigen::MatrixXf::Identity(m.rows(), m.cols());
    Eigen::MatrixXf R = qr.matrixQR().triangularView<Eigen::Upper>();

    Dictionary out;
    out["Q"] = Matrix::from_array(Utils::eigen_to_godot(Q));
    out["R"] = Matrix::from_array(Utils::eigen_to_godot(R));
    return out;
}

Dictionary Linalg::svd(const Ref<Matrix> &A) {
    const auto &m = A->eigen();

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(
        m, Eigen::ComputeThinU | Eigen::ComputeThinV
    );

    Dictionary out;
    out["U"] = Matrix::from_array(Utils::eigen_to_godot(svd.matrixU()));
    out["S"] = Matrix::from_array(Utils::eigen_to_godot(svd.singularValues().asDiagonal()));
    out["V"] = Matrix::from_array(Utils::eigen_to_godot(svd.matrixV()));
    return out;
}

Dictionary Linalg::eig(const Ref<Matrix> &A) {
    const auto &m = A->eigen();

    if (m.rows() != m.cols()) {
        Logger::error_raise("Linalg.eig(): matrix must be square");
        return Dictionary();
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eig(m);
    if (eig.info() != Eigen::Success) {
        Logger::error_raise("Linalg.eig(): decomposition failed");
        return Dictionary();
    }

    Dictionary out;
    out["values"] = Matrix::from_array(Utils::eigen_to_godot(eig.eigenvalues()));
    out["vectors"] = Matrix::from_array(Utils::eigen_to_godot(eig.eigenvectors()));
    return out;
}

Dictionary Linalg::lu(const Ref<Matrix> &A) {
    const auto &m = A->eigen();

    Eigen::PartialPivLU<Eigen::MatrixXf> lu(m);

    Dictionary out;
    out["L"] = Matrix::from_array(Utils::eigen_to_godot(
        Eigen::MatrixXf(lu.matrixLU().triangularView<Eigen::UnitLower>())
    ));
    out["U"] = Matrix::from_array(Utils::eigen_to_godot(
        Eigen::MatrixXf(lu.matrixLU().triangularView<Eigen::Upper>())
    ));
    out["P"] = Matrix::from_array(Utils::eigen_to_godot(lu.permutationP()));
    return out;
}

} // namespace godot