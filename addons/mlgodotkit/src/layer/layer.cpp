#include "layer/layer.h"

Layer::Layer(int input_size, int out_features, float learning_rate, std::string activation_type) {
    weights = Eigen::MatrixXd::Random(input_size, out_features) * sqrt(2.0 / (input_size + out_features));
    biases = Eigen::MatrixXd::Zero(out_features, 1);

    lr = learning_rate;

    if (activation_type == "sigmoid") {
        activation_func = sigmoid;
        derivative_func = sigmoid_derivative;
    } else if (activation_type == "relu") {
        activation_func = relu;
        derivative_func = relu_derivative;
    } else { //default to relu for now
        activation_func = relu;
        derivative_func = relu_derivative;
    }

    // Debugging: Print the initial weights and biases
    godot::UtilityFunctions::print("Layer initialization:");
    godot::UtilityFunctions::print("Initial weights shape: " + godot::String::num(weights.rows()) + "x" + godot::String::num(weights.cols()));
    godot::UtilityFunctions::print(eigen_to_godot(weights));
    godot::UtilityFunctions::print("Initial biases shape: " + godot::String::num(biases.rows()) + "x" + godot::String::num(biases.cols()));
    godot::UtilityFunctions::print(eigen_to_godot(biases));
}

Layer::~Layer() {}

Eigen::MatrixXd Layer::forward(const Eigen::MatrixXd& X) {
    input = X;

    // Debugging: Print input matrix and its shape
    godot::UtilityFunctions::print("Input to the forward pass:");
    godot::UtilityFunctions::print("Input shape: " + godot::String::num(input.rows()) + "x" + godot::String::num(input.cols()));
    godot::UtilityFunctions::print(eigen_to_godot(input));

    // Compute the linear combination: z = X * weights + biases
    Eigen::MatrixXd z = (input * weights) + biases;

    // Debugging: Print the linear combination matrix z and its shape
    godot::UtilityFunctions::print("Linear combination (z) from forward pass:");
    godot::UtilityFunctions::print("Shape: " + godot::String::num(z.rows()) + "x" + godot::String::num(z.cols()));
    godot::UtilityFunctions::print(eigen_to_godot(z));

    output = activation_func(z);
    grad_z = z;

    // Debugging: Print the output of the activation function
    godot::UtilityFunctions::print("Activation output:");
    godot::UtilityFunctions::print("Shape: " + godot::String::num(output.rows()) + "x" + godot::String::num(output.cols()));
    godot::UtilityFunctions::print(eigen_to_godot(output));

    return output;
}

Eigen::MatrixXd Layer::backward(const Eigen::MatrixXd& error) {
    // Compute the delta (error term)
    Eigen::MatrixXd delta = error.cwiseProduct(derivative_func(grad_z));

    // Debugging: Print the delta (error term) and its shape
    godot::UtilityFunctions::print("Delta (error term) for backward pass:");
    godot::UtilityFunctions::print("Shape: " + godot::String::num(delta.rows()) + "x" + godot::String::num(delta.cols()));
    godot::UtilityFunctions::print(eigen_to_godot(delta));

    // Compute gradients for weights and biases
    Eigen::MatrixXd dW = input.transpose() * delta;
    Eigen::MatrixXd db = delta.colwise().sum();

    // Debugging: Print the gradients for weights and biases
    godot::UtilityFunctions::print("Gradients for weights and biases:");
    godot::UtilityFunctions::print("dW shape: " + godot::String::num(dW.rows()) + "x" + godot::String::num(dW.cols()));
    godot::UtilityFunctions::print(eigen_to_godot(dW));
    godot::UtilityFunctions::print("db shape: " + godot::String::num(db.rows()) + "x" + godot::String::num(db.cols()));
    godot::UtilityFunctions::print(eigen_to_godot(db));

    // Update weights and biases with gradient clipping
    weights -= lr * dW;
    biases -= lr * db;

    // Debugging: Print the updated weights and biases
    godot::UtilityFunctions::print("Updated weights and biases:");
    godot::UtilityFunctions::print("Updated weights shape: " + godot::String::num(weights.rows()) + "x" + godot::String::num(weights.cols()));
    godot::UtilityFunctions::print(eigen_to_godot(weights));
    godot::UtilityFunctions::print("Updated biases shape: " + godot::String::num(biases.rows()) + "x" + godot::String::num(biases.cols()));
    godot::UtilityFunctions::print(eigen_to_godot(biases));

    // Compute the next error gradient
    Eigen::MatrixXd grad_input = delta * weights;

    // Debugging: Print the next error gradient and its shape
    godot::UtilityFunctions::print("Next error gradient (grad_input) for backward pass:");
    godot::UtilityFunctions::print("Shape: " + godot::String::num(grad_input.rows()) + "x" + godot::String::num(grad_input.cols()));
    godot::UtilityFunctions::print(eigen_to_godot(grad_input));

    return grad_input;
}

// *** Activation Functions ***

// Sigmoid Activation
Eigen::MatrixXd Layer::sigmoid(const Eigen::MatrixXd& x) {
    // Debugging: Print the input to sigmoid
    godot::UtilityFunctions::print("Sigmoid input:");
    godot::UtilityFunctions::print(eigen_to_godot(x));

    Eigen::MatrixXd result = 1.0 / (1.0 + (-x.array()).exp());

    // Debugging: Print the output of sigmoid
    godot::UtilityFunctions::print("Sigmoid output:");
    godot::UtilityFunctions::print(eigen_to_godot(result));

    return result;
}

// Sigmoid Derivative
Eigen::MatrixXd Layer::sigmoid_derivative(const Eigen::MatrixXd& z) {
    Eigen::MatrixXd s = sigmoid(z);

    // Debugging: Print the derivative of sigmoid
    godot::UtilityFunctions::print("Sigmoid derivative:");
    godot::UtilityFunctions::print(eigen_to_godot(s.array() * (1.0 - s.array())));

    return s.array() * (1.0 - s.array());
}

// ReLU Activation
Eigen::MatrixXd Layer::relu(const Eigen::MatrixXd& x) {
    // Debugging: Print the input to ReLU
    godot::UtilityFunctions::print("ReLU input:");
    godot::UtilityFunctions::print(eigen_to_godot(x));

    Eigen::MatrixXd result = x.cwiseMax(0);

    // Debugging: Print the output of ReLU
    godot::UtilityFunctions::print("ReLU output:");
    godot::UtilityFunctions::print(eigen_to_godot(result));

    return result;
}

// ReLU Derivative
Eigen::MatrixXd Layer::relu_derivative(const Eigen::MatrixXd& z) {
    // Debugging: Print the input to ReLU derivative
    godot::UtilityFunctions::print("ReLU derivative input:");
    godot::UtilityFunctions::print(eigen_to_godot(z));

    Eigen::MatrixXd result = (z.array() > 0).cast<double>();

    // Debugging: Print the output of ReLU derivative
    godot::UtilityFunctions::print("ReLU derivative output:");
    godot::UtilityFunctions::print(eigen_to_godot(result));

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

// Helper Functions
Eigen::MatrixXd Layer::godot_to_eigen(godot::Array array) {
    int rows = array.size();
    int cols = 0;

    // Handle 2D array case
    if (rows > 0 && array[0].get_type() == godot::Variant::ARRAY) {
        godot::Array first_row = array[0];
        cols = first_row.size();

        // Create an Eigen matrix for 2D arrays
        Eigen::MatrixXd out(rows, cols);
        for (int i = 0; i < rows; i++) {
            godot::Array row = array[i];
            for (int j = 0; j < cols; j++) {
                out(i, j) = static_cast<double>(row[j]);
            }
        }
        return out;
    }

    // Handle 1D array case (e.g., 1x2 or n×1)
    cols = rows; // A single 1D array will be treated as one row
    rows = 1;    // Force 1 row for 1D array inputs

    Eigen::MatrixXd out(rows, cols);
    for (int j = 0; j < cols; j++) {
        out(0, j) = static_cast<double>(array[j]);
    }

    return out;
}

godot::Array Layer::eigen_to_godot(Eigen::MatrixXd matrix){
    godot::Array out;

    for(int i=0;i<matrix.rows();i++){
        godot::Array row;
        for(int j=0;j<matrix.cols();j++){
            row.push_back(matrix(i,j));
        }
        if(matrix.rows() <= 1){
            return row;
        }
        out.push_back(row);
    }
    return out;
}