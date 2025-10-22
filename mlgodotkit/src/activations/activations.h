#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <Eigen/Dense>
#include <unordered_map>
#include <functional>
#include <string>

namespace Activations {
    // --- Core activation APIs ---
    Eigen::MatrixXf sigmoid(const Eigen::MatrixXf& x);
    Eigen::MatrixXf sigmoid_derivative(const Eigen::MatrixXf& z);

    Eigen::MatrixXf relu(const Eigen::MatrixXf& x);
    Eigen::MatrixXf relu_derivative(const Eigen::MatrixXf& z);

    Eigen::MatrixXf linear(const Eigen::MatrixXf& x);
    Eigen::MatrixXf linear_derivative(const Eigen::MatrixXf& z);

    Eigen::MatrixXf leaky_relu(const Eigen::MatrixXf& x, float alpha = 0.01f);
    Eigen::MatrixXf leaky_relu_derivative(const Eigen::MatrixXf& z, float alpha = 0.01f);

    // --- Lookup helpers ---
    std::function<Eigen::MatrixXf(const Eigen::MatrixXf&)> get_activation(const std::string& name);
    std::function<Eigen::MatrixXf(const Eigen::MatrixXf&)> get_derivative(const std::string& name);
}

#endif