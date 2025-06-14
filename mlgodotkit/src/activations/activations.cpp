#include "activations.h"

// *** Activation Functions ***
static constexpr float epsilon = 1e-6; // Small value to avoid overflow or division by zero

// Linear Activation (identity)
Eigen::MatrixXf Activations::linear(const Eigen::MatrixXf& x) {
    return x;
}

// Derivative of Linear Activation
Eigen::MatrixXf Activations::linear_derivative(const Eigen::MatrixXf& z) {
    // Derivative of f(x) = x is 1
    return Eigen::MatrixXf::Ones(z.rows(), z.cols());
}

// Sigmoid Activation
Eigen::MatrixXf Activations::sigmoid(const Eigen::MatrixXf& x) {
    Eigen::MatrixXf result = 1.0 / (1.0 + (-x.array()).exp());

    // Clip values to prevent overflow in case of extreme values
    result = result.array().min(1.0 - epsilon).max(epsilon);
    return result;
}

// Sigmoid Derivative
Eigen::MatrixXf Activations::sigmoid_derivative(const Eigen::MatrixXf& z) {
    Eigen::MatrixXf s = sigmoid(z);

    // Ensure that derivative is numerically stable
    return s.array() * (1.0 - s.array()).max(epsilon);
}

// ReLU Activation
Eigen::MatrixXf Activations::relu(const Eigen::MatrixXf& x) {
    Eigen::MatrixXf result = x.cwiseMax(0);
    return result;
}

// ReLU Derivative
Eigen::MatrixXf Activations::relu_derivative(const Eigen::MatrixXf& z) {
    Eigen::MatrixXf result = (z.array() > 0).cast<float>();
    return result;
}
