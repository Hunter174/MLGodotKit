#include "neural_network/nnnode.h"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/method_bind.hpp>
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/variant/array.hpp>
#include <cmath>
#include <random>
#include <algorithm>

using namespace godot;

NNNode::NNNode() {}
NNNode::~NNNode() {}

void NNNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("initialize", "layer_sizes"), &NNNode::initialize);
    ClassDB::bind_method(D_METHOD("predict", "input"), &NNNode::predict);
    ClassDB::bind_method(D_METHOD("train", "input", "target"), &NNNode::train);
    ClassDB::bind_method(D_METHOD("set_optimizer", "optimizer_type", "learning_rate"), &NNNode::set_optimizer);
    ClassDB::bind_method(D_METHOD("set_loss_function", "loss_function"), &NNNode::set_loss_function);
    ClassDB::bind_method(D_METHOD("get_loss"), &NNNode::get_loss);
    ClassDB::bind_method(D_METHOD("copy_from", "other"), &NNNode::copy_from);
}

void NNNode::initialize(Array layer_sizes) {
    if (layer_sizes.size() < 2) {
        UtilityFunctions::print("Error: Must have at least input and output layers.");
        return;
    }

    model_architecture = layer_sizes;
    weights.clear();
    biases.clear();

    std::vector<int> layers;
    for (int i = 0; i < layer_sizes.size(); ++i) {
        layers.push_back(int(layer_sizes[i]));
    }

    for (size_t i = 0; i < layers.size() - 1; ++i) {
        weights.push_back(init_weights(layers[i + 1], layers[i]));
        biases.push_back(VectorXd::Zero(layers[i + 1]));
    }
}

MatrixXd NNNode::init_weights(int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0, sqrt(2.0 / cols));  // He initialization for ReLU

    MatrixXd mat(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            mat(i, j) = dis(gen);

    return mat;
}

VectorXd NNNode::relu(const VectorXd& x) const {
    return x.cwiseMax(0);
}

VectorXd NNNode::relu_derivative(const VectorXd& x) const {
    return x.unaryExpr([](double val) { return val > 0 ? 1.0 : 0.0; });
}

VectorXd NNNode::softmax(const VectorXd& x) const {
    VectorXd exp_x = (x.array() - x.maxCoeff()).exp();
    return exp_x / exp_x.sum();
}

VectorXd NNNode::forward_pass(const VectorXd& input) const {
    VectorXd activation = input;
    for (size_t i = 0; i < weights.size() - 1; ++i) {
        activation = relu(weights[i] * activation + biases[i]);
    }
    return weights.back() * activation + biases.back();
}

Array NNNode::predict(Array input) {
    VectorXd input_vec(input.size());
    for (int i = 0; i < input.size(); ++i) {
        input_vec[i] = float(input[i]);
    }

    VectorXd output = forward_pass(input_vec);

    Array result;
    for (int i = 0; i < output.size(); ++i) {
        result.push_back(output[i]);
    }
    return result;
}

double NNNode::mean_squared_error(const VectorXd& output, const VectorXd& target) {
    return (output - target).squaredNorm() / target.size();
}

double NNNode::mean_absolute_error(const VectorXd& output, const VectorXd& target) {
    return (output - target).array().abs().sum() / target.size();
}

void NNNode::train(Array input, Array target) {
    VectorXd input_vec(input.size());
    VectorXd target_vec(target.size());

    for (int i = 0; i < input.size(); ++i) input_vec[i] = float(input[i]);
    for (int i = 0; i < target.size(); ++i) target_vec[i] = float(target[i]);

    std::vector<VectorXd> activations = {input_vec};
    std::vector<VectorXd> zs;

    // Forward pass
    VectorXd activation = input_vec;
    for (size_t i = 0; i < weights.size(); ++i) {
        VectorXd z = weights[i] * activation + biases[i];
        zs.push_back(z);
        activation = (i < weights.size() - 1) ? relu(z) : z;
        activations.push_back(activation);
    }

    // Compute and store loss
    last_loss = mean_squared_error(activations.back(), target_vec);

    // Backpropagation
    VectorXd delta = activations.back() - target_vec;

    for (int layer = weights.size() - 1; layer >= 0; --layer) {
        MatrixXd grad_w = delta * activations[layer].transpose();
        VectorXd grad_b = delta;

        // Gradient clipping for stability
        grad_w = grad_w.unaryExpr([](double val) { return std::clamp(val, -1.0, 1.0); });
        grad_b = grad_b.unaryExpr([](double val) { return std::clamp(val, -1.0, 1.0); });

        weights[layer] -= learning_rate * grad_w;
        biases[layer] -= learning_rate * grad_b;

        if (layer > 0) {
            delta = (weights[layer].transpose() * delta).cwiseProduct(relu_derivative(zs[layer - 1]));
        }
    }
}

void NNNode::set_optimizer(String optimizer_type, float lr) {
    if (optimizer_type == "SGD") {
        learning_rate = lr;
    } else {
        UtilityFunctions::print("Unsupported optimizer");
    }
}

void NNNode::set_loss_function(String loss_function) {
    if (loss_function == "MSE") {
        loss_function_type = loss_function;
    } else {
        UtilityFunctions::print("Unsupported loss function");
    }
}

double NNNode::get_loss() const {
    return last_loss;
}

void NNNode::decay_learning_rate() {
    learning_rate = std::max(0.0001, learning_rate * 0.99);
}

void NNNode::copy_from(const NNNode* other) {
    if (!other) {
        UtilityFunctions::print("Error: Null NNNode instance.");
        return;
    }

    if (model_architecture != other->model_architecture) {
        UtilityFunctions::print("Error: Model architectures do not match.");
        return;
    }

    weights = other->weights;
    biases = other->biases;
    learning_rate = other->learning_rate;
    loss_function_type = other->loss_function_type;
}
