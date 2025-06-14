#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <Eigen/Dense>

namespace Activations {
    Eigen::MatrixXf sigmoid(const Eigen::MatrixXf& x);
    Eigen::MatrixXf sigmoid_derivative(const Eigen::MatrixXf& z);

    Eigen::MatrixXf relu(const Eigen::MatrixXf& x);
    Eigen::MatrixXf relu_derivative(const Eigen::MatrixXf& z);

    Eigen::MatrixXf linear(const Eigen::MatrixXf& x);
    Eigen::MatrixXf linear_derivative(const Eigen::MatrixXf& z);
}

#endif
