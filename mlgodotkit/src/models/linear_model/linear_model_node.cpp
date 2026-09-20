#include "linear_model_node.h"

using namespace godot;

LinearModelNode::LinearModelNode() {
    core = std::make_unique<LinearModelCore>();
}

LinearModelNode::~LinearModelNode() {}

void LinearModelNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("initialize", "input_size"), &LinearModelNode::initialize);
    ClassDB::bind_method(D_METHOD("predict", "input"), &LinearModelNode::predict);
    ClassDB::bind_method(D_METHOD("train", "inputs", "targets", "epochs"), &LinearModelNode::train);
    ClassDB::bind_method(D_METHOD("set_learning_rate", "lr"), &LinearModelNode::set_learning_rate);
}

void LinearModelNode::initialize(int input_size) {
    core->initialize(input_size);
}

Array LinearModelNode::predict(Array input) {
    Eigen::VectorXf x = godot_to_eigen(input, 1);
    float p = core->predict_single(x);
    return Array::from(p);
}

void LinearModelNode::train(Array inputs_arr, Array targets_arr, int epochs) {
    Eigen::MatrixXf inputs = godot_to_eigen(inputs_arr, inputs_arr.size());
    Eigen::VectorXf targets = godot_to_eigen(targets_arr, targets_arr.size());
    core->train(inputs, targets, epochs);
}

void LinearModelNode::set_learning_rate(double lr) {
    core->learning_rate = lr;
}
