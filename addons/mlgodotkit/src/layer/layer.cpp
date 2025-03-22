#include "layer/layer.h"

using namespace Utils;

Layer::Layer(int input_size, int out_features, float learning_rate, std::string activation_type) {

    lr = learning_rate;

    if (activation_type == "sigmoid") {
		//Initialize weights and bias
      	weights = Eigen::MatrixXf::Random(input_size, out_features) * sqrt(1.0 / input_size);
		biases = Eigen::MatrixXf::Zero(1, out_features);

        activation_func = sigmoid;
        derivative_func = sigmoid_derivative;
    } else if (activation_type == "relu") {
      	weights = Eigen::MatrixXf::Random(input_size, out_features) * sqrt(2.0 / (input_size + out_features));
    	biases = Eigen::MatrixXf::Zero(1, out_features);

        activation_func = relu;
        derivative_func = relu_derivative;
    } else { //default to relu for now
        activation_func = relu;
        derivative_func = relu_derivative;
    }

    // Debugging: Print the initial weights and biases
	debug_print(verbosity, 1, "Initial weights -> Shape: " + godot::String::num(weights.rows()) + "x" + godot::String::num(weights.cols()) +
                          ", Values: " + godot::String(eigen_to_string(weights).c_str()));

	debug_print(verbosity, 1, "Initial biases -> Shape: " + godot::String::num(biases.rows()) + "x" + godot::String::num(biases.cols()) +
                          ", Values: " + godot::String(eigen_to_string(biases).c_str()));
}

Layer::~Layer() {}

Eigen::MatrixXf Layer::forward(const Eigen::MatrixXf& X) {
    input = X;

    // Debugging: Print input matrix and its shape
	debug_print(verbosity, 2, "Forward input -> Shape: " + godot::String::num(input.rows()) + "x" + godot::String::num(input.cols()) +
                          ", Values: " + godot::String(eigen_to_string(input).c_str()));


    // Compute the linear combination: z = X * weights + biases
    Eigen::MatrixXf z = (input * weights) + biases;

    // Debugging: Print the linear combination matrix z and its shape
	debug_print(verbosity, 2, "Linear combination (z) -> Shape: " + godot::String::num(z.rows()) + "x" + godot::String::num(z.cols()) +
                          ", Values: " + godot::String(eigen_to_string(z).c_str()));

    output = activation_func(z);
    grad_z = z;

    // Debugging: Print the output of the activation function
	debug_print(verbosity, 2, "Activation-> Shape: " + godot::String::num(output.rows()) + "x" + godot::String::num(output.cols()) +
                          ", Values: " + godot::String(eigen_to_string(output).c_str()));

    return output;
}

Eigen::MatrixXf Layer::backward(const Eigen::MatrixXf& error) {
    // Compute the delta (error term)
    Eigen::MatrixXf delta = error.cwiseProduct(derivative_func(grad_z));
    delta = stable_round(delta, precision);

    // Debugging: Print the delta (error term) and its shape
    debug_print(verbosity, 3, "Delta -> Shape: " + godot::String::num(delta.rows()) + "x" + godot::String::num(delta.cols()) +
                          ", Values: " + godot::String(eigen_to_string(delta).c_str()));

    // Compute gradients for weights and biases
    Eigen::MatrixXf dW = input.transpose() * delta;
    Eigen::MatrixXf db = delta.colwise().sum();

    dW = stable_round(dW, precision);
    db = stable_round(db, precision);

    // Debugging: Print gradients
    debug_print(verbosity, 3, "dW -> Shape: " + godot::String::num(dW.rows()) + "x" + godot::String::num(dW.cols()) +
                          ", Values: " + godot::String(eigen_to_string(dW).c_str()));
    debug_print(verbosity, 3, "db -> Shape: " + godot::String::num(db.rows()) + "x" + godot::String::num(db.cols()) +
                          ", Values: " + godot::String(eigen_to_string(db).c_str()));

    // Update weights and biases with gradient clipping
    weights -= lr * dW;
    biases -= lr * db;

	weights = stable_round(weights, precision);
	biases = stable_round(biases, precision);

    // Debugging: Print updated weights and biases
    debug_print(verbosity, 3, "Updated weights -> Shape: " + godot::String::num(weights.rows()) + "x" + godot::String::num(weights.cols()) +
                          ", Values: " + godot::String(eigen_to_string(weights).c_str()));
    debug_print(verbosity, 3, "Updated biases -> Shape: " + godot::String::num(biases.rows()) + "x" + godot::String::num(biases.cols()) +
                          ", Values: " + godot::String(eigen_to_string(biases).c_str()));

    // Compute the next error gradient
    Eigen::MatrixXf grad_input = delta * weights.transpose();
    grad_input = stable_round(grad_input, precision);

    // Debugging: Print the next error gradient and its shape
    debug_print(verbosity, 3, "grad_input -> Shape: " + godot::String::num(grad_input.rows()) + "x" + godot::String::num(grad_input.cols()) +
               	 			", Values: " + godot::String(eigen_to_string(grad_input).c_str()));

    return grad_input;
}

// *** Activation Functions ***

// Sigmoid Activation
Eigen::MatrixXf Layer::sigmoid(const Eigen::MatrixXf& x) {
    Eigen::MatrixXf result = 1.0 / (1.0 + (-x.array()).exp());

    // Clip values to prevent overflow in case of extreme values
    result = result.array().min(1.0 - epsilon).max(epsilon);
    return result;
}

// Sigmoid Derivative
Eigen::MatrixXf Layer::sigmoid_derivative(const Eigen::MatrixXf& z) {
    Eigen::MatrixXf s = sigmoid(z);

    // Ensure that derivative is numerically stable
    return s.array() * (1.0 - s.array()).max(epsilon);
}

// ReLU Activation
Eigen::MatrixXf Layer::relu(const Eigen::MatrixXf& x) {
    Eigen::MatrixXf result = x.cwiseMax(0);
    return result;
}

// ReLU Derivative
Eigen::MatrixXf Layer::relu_derivative(const Eigen::MatrixXf& z) {
    Eigen::MatrixXf result = (z.array() > 0).cast<float>();
    return result;
}

godot::String Layer::to_string() const {
    std::ostringstream stream;
    stream << "Layer Information:\n";
    stream << "  - Weights (Shape: " << weights.rows() << "x" << weights.cols() << "):\n\t"
           << weights << "\n";
    stream << "  - Biases (Shape: " << biases.size() << "):\n\t"
           << biases.transpose() << "\n";

    return godot::String(stream.str().c_str());
}

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