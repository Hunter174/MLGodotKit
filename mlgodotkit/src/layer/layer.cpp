#include "layer/layer.h"

using namespace Activations;

Layer::Layer(int input_size, int out_features, float learning_rate, std::string activation_type) {

    lr = learning_rate;

    std::tie(weights, biases) = init_weights(input_size, out_features, activation_type);

    // Initialize the weights based on the activation func
    if (activation_type == "sigmoid") {
        activation_func = sigmoid;
        derivative_func = sigmoid_derivative;
    } else if (activation_type == "relu") {
        activation_func = relu;
        derivative_func = relu_derivative;
    } else if (activation_type == "linear") {
        activation_func = linear;
        derivative_func = linear_derivative;
    } else {
        // fallback default
        activation_func = relu;
        derivative_func = relu_derivative;
    }

    // Debugging: Print the initial weights and biases
    Logger::debug(2, "Initialized weights:\n" + GodotUtils::eigen_to_string(weights));
    Logger::debug(2, "Initialized biases:\n" + GodotUtils::eigen_to_string(biases));
}

Layer::~Layer() {}

Eigen::MatrixXf Layer::forward(const Eigen::MatrixXf& X) {
    input = X;

    // Debugging: Print input matrix and its shape
    Logger::debug(2, "Forward input (X):\n" + GodotUtils::eigen_to_string(input));

    // Compute the linear combination: z = X * weights + biases
    Eigen::MatrixXf z = (input * weights) + biases;

    // Debugging: Print the linear combination matrix z and its shape
    Logger::debug(2, "Linear combination (Z = XW + b):\n" + GodotUtils::eigen_to_string(z));

    output = activation_func(z);
    grad_z = z;

    // Debugging: Print the output of the activation function
    Logger::debug(2, "Activation output:\n" + GodotUtils::eigen_to_string(output));

    return output;
}

Eigen::MatrixXf Layer::backward(const Eigen::MatrixXf& error) {
    // Compute the delta (error term)
    Eigen::MatrixXf delta = error.cwiseProduct(derivative_func(grad_z));

    // Debugging: Print the delta (error term) and its shape
    Logger::debug(3, "Delta (error * activation'):\n" + GodotUtils::eigen_to_string(delta));

    // Compute gradients for weights and biases
    Eigen::MatrixXf dW = input.transpose() * delta;
    Eigen::MatrixXf db = delta.colwise().sum();

    // Debugging: Print gradients
    Logger::debug(3, "Weight gradients (dW):\n" + GodotUtils::eigen_to_string(dW));
    Logger::debug(3, "Bias gradients (db):\n" + GodotUtils::eigen_to_string(db));

    // Update weights and biases
    weights -= lr * dW;
    biases -= lr * db;

    Logger::debug(3, "Updated weights:\n" + GodotUtils::eigen_to_string(weights));
    Logger::debug(3, "Updated biases:\n" + GodotUtils::eigen_to_string(biases));

    // Compute the next error gradient
    Eigen::MatrixXf grad_input = delta * weights.transpose();

    // Debugging: Print the next error gradient and its shape
    Logger::debug(3, "Gradient passed to previous layer:\n" + GodotUtils::eigen_to_string(grad_input));

    return grad_input;
}

// *** Weight Initialization Method(s) ***
std::tuple<Eigen::MatrixXf, Eigen::MatrixXf> Layer::init_weights(int input_size, int out_features, const std::string& activation_type)
{
    Eigen::MatrixXf weights;
    Eigen::MatrixXf biases = Eigen::MatrixXf::Zero(1, out_features);

    if (activation_type == "sigmoid") {
        weights = Eigen::MatrixXf::Random(input_size, out_features) * std::sqrt(1.0f / input_size);
    } else if (activation_type == "relu") {
        weights = Eigen::MatrixXf::Random(input_size, out_features) * std::sqrt(2.0f / (input_size + out_features));
    } else {
        // Default to He initialization
        weights = Eigen::MatrixXf::Random(input_size, out_features) * std::sqrt(2.0f / input_size);
    }

    return std::make_tuple(weights, biases);
}

// *** Class Specific Utility Functions ***
std::string Layer::to_string() const {
    std::ostringstream stream;
    stream << "Layer Information:\n";
    stream << "  - Weights (Shape: " << weights.rows() << "x" << weights.cols() << "):\n\t"
           << weights << "\n";
    stream << "  - Biases (Shape: " << biases.size() << "):\n\t"
           << biases.transpose() << "\n";

    return stream.str();
}

// Rounding for numeric stability
Eigen::MatrixXf Layer::stable_round(const Eigen::MatrixXf& mat, int precision, float threshold) {
    Eigen::MatrixXf result = mat;
    float scale = std::pow(10.0f, precision);
    for (int i = 0; i < result.rows(); ++i) {
        for (int j = 0; j < result.cols(); ++j) {
            if (std::abs(result(i, j)) > threshold) {
                result(i, j) = std::round(result(i, j) * scale) / scale;
            }
        }
    }
    return result;
}