#include "nnnode.h"

using namespace Utils;

NNNode::NNNode() {}

NNNode::~NNNode() {}

void NNNode::_bind_methods() {
    godot::ClassDB::bind_method(godot::D_METHOD("add_layer", "input_size", "output_size", "activation"), &NNNode::add_layer);
    godot::ClassDB::bind_method(godot::D_METHOD("forward", "input"), &NNNode::forward);
    godot::ClassDB::bind_method(godot::D_METHOD("backward", "error"), &NNNode::backward);
    godot::ClassDB::bind_method(godot::D_METHOD("model_summary"), &NNNode::model_summary);
    godot::ClassDB::bind_method(godot::D_METHOD("set_learning_rate", "lr"), &NNNode::set_learning_rate);
}

void NNNode::add_layer(int input_size, int output_size, godot::String activation){
    std::string activation_type = activation.utf8().get_data();

    layers.push_back(Layer(input_size, output_size, learning_rate, activation_type));
}

godot::Array NNNode::forward(godot::Array input){
    Eigen::MatrixXd x = godot_to_eigen(input);

    for(int i=0;i<layers.size(); i++){
        x = layers[i].forward(x);
    }

    godot::Array output = eigen_to_godot(x);
    return output;
}

void NNNode::backward(godot::Array error) {
    Eigen::MatrixXd grad_loss = godot_to_eigen(error);

    // Backward pass through layers
    for (int i = layers.size() - 1; i >= 0; --i) {
        grad_loss = layers[i].backward(grad_loss);
    }
}

void NNNode::model_summary() {
    for (int i = 0; i < layers.size(); i++) {
        godot::UtilityFunctions::print(layers[i].to_string());
    }
}