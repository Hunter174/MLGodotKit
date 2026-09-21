#include "adam_core.h"

void AdamCore::update(Eigen::MatrixXf& param, const Eigen::MatrixXf& grad, int index) {
    if (m.size() <= index) {
        m.resize(index + 1);
        v.resize(index + 1);
    }

    if (m[index].rows() != grad.rows() || m[index].cols() != grad.cols()) {
        m[index] = Eigen::MatrixXf::Zero(grad.rows(), grad.cols());
        v[index] = Eigen::MatrixXf::Zero(grad.rows(), grad.cols());
    }

    m[index] = beta1 * m[index] + (1 - beta1) * grad;
    v[index] = beta2 * v[index] + (1 - beta2) * grad.array().square().matrix();

    Eigen::MatrixXf m_hat = m[index] / (1 - std::pow(beta1, timestep));
    Eigen::MatrixXf v_hat = v[index] / (1 - std::pow(beta2, timestep));

    param -= (lr * m_hat.array() / (v_hat.array().sqrt() + eps)).matrix();
}
