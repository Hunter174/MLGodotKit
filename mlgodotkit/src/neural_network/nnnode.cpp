#include "nnnode.h"

using namespace Utils;

NNNode::NNNode() {}

NNNode::~NNNode() {}

void NNNode::_bind_methods() {
    godot::ClassDB::bind_method(godot::D_METHOD("add_layer", "input_size", "output_size", "activation"), &NNNode::add_layer);
    godot::ClassDB::bind_method(godot::D_METHOD("forward", "input"), &NNNode::forward);
    godot::ClassDB::bind_method(godot::D_METHOD("backward", "error"), &NNNode::backward);
//    godot::ClassDB::bind_method(godot::D_METHOD("model_summary"), &NNNode::model_summary);
    godot::ClassDB::bind_method(godot::D_METHOD("set_learning_rate", "lr"), &NNNode::set_learning_rate);
    godot::ClassDB::bind_method(godot::D_METHOD("set_verbosity", "level"), &NNNode::set_verbosity);
}

void NNNode::add_layer(int input_size, int output_size, godot::String activation){
    std::string activation_type = activation.utf8().get_data();
	Layer temp = Layer(input_size, output_size, learning_rate, activation_type);
    temp.set_verbosity(verbosity);
    layers.push_back(temp);
}

godot::Array NNNode::forward(godot::Array input){
    Eigen::MatrixXf x = godot_to_eigen(input);

    for(int i=0;i<layers.size(); i++){
        x = layers[i].forward(x);
    }

    godot::Array output = eigen_to_godot(x);
    return output;
}

void NNNode::backward(godot::Array error) {
    Eigen::MatrixXf grad_loss = godot_to_eigen(error);

    // Backward pass through layers
    for (int i = layers.size() - 1; i >= 0; --i) {
        grad_loss = layers[i].backward(grad_loss);
    }

    // Decay Learning Rate
//    learning_rate *= 0.995;

    // Clamp the learning rate to be within a specific range [min_lr, max_lr]
    double min_lr = 1e-5;  // Minimum learning rate
    double max_lr = 0.1;   // Maximum learning rate
    learning_rate = std::min(std::max(learning_rate, min_lr), max_lr);

    // Round the learning rate to 5 decimal places
    learning_rate = std::round(learning_rate * 100000.0) / 100000.0;

    // Update the learning rate in the layers
    set_learning_rate(learning_rate);
}

//void NNNode::model_summary() {
//    for (int i = 0; i < layers.size(); i++) {
//        godot::UtilityFunctions::print(layers[i].to_string());
//    }
//}

void NNNode::set_verbosity(int verbosity) {
    this->verbosity = verbosity;
    for(int i = 0; i < layers.size(); i++){
        layers[i].set_verbosity(verbosity);
    }
}

void NNNode::set_learning_rate(double lr){
	this->learning_rate = lr;
	for(int i = 0; i < layers.size(); i++){
		layers[i].set_learning_rate(lr);
    }
}