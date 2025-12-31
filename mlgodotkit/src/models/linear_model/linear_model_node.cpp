#include "linear_model_node.h"

LinearModelNode::LinearModelNode() : learning_rate(0.01), bias(0) {}

LinearModelNode::~LinearModelNode() {}

void LinearModelNode::_bind_methods() {
    godot::ClassDB::bind_method(godot::D_METHOD("initialize", "input_size"), &LinearModelNode::initialize);
    godot::ClassDB::bind_method(godot::D_METHOD("predict", "input"), &LinearModelNode::predict);
    godot::ClassDB::bind_method(godot::D_METHOD("train", "inputs", "targets", "epochs"), &LinearModelNode::train);
    godot::ClassDB::bind_method(godot::D_METHOD("set_learning_rate", "lr"), &LinearModelNode::set_learning_rate);
}

void LinearModelNode::initialize(int input_size) {
    num_features = input_size;
    weights = Eigen::VectorXf::Random(input_size);
    bias = 0.0;
}

godot::Array LinearModelNode::predict(godot::Array input) {
    Eigen::MatrixXf x = Utils::godot_to_eigen(input);
    Eigen::VectorXf predictions = (x * weights).array() + bias;
    return Utils::eigen_to_godot(predictions);
}

void LinearModelNode::train(godot::Array inputs, godot::Array targets, int epochs) {
    Eigen::MatrixXf X = Utils::godot_to_eigen(inputs);
    Eigen::VectorXf y = Utils::godot_to_eigen(targets);

    for (int epoch = 0; epoch < epochs; epoch++) {
        Eigen::VectorXf predictions = (X * weights).array() + bias;
        Eigen::VectorXf gradients = compute_gradient(predictions, y, X);

        // Gradient descent update
        weights -= learning_rate * gradients;
        bias -= learning_rate * (predictions - y).mean();
    }
}

double LinearModelNode::compute_loss(const Eigen::VectorXf &predictions, const Eigen::VectorXf &targets) {
    return (predictions - targets).array().square().mean(); // MSE loss
}

Eigen::VectorXf LinearModelNode::compute_gradient(const Eigen::VectorXf &predictions, const Eigen::VectorXf &targets, const Eigen::MatrixXf &inputs) {
    int n = inputs.rows();
    return (2.0 / n) * (inputs.transpose() * (predictions - targets));
}

void LinearModelNode::set_learning_rate(double lr) {
    learning_rate = lr;
}
