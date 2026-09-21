#include "neural_network_node.h"

using namespace Utils;

NeuralNetworkNode::NeuralNetworkNode() {
    core = std::make_unique<NeuralNetworkCore>();
}

NeuralNetworkNode::~NeuralNetworkNode() {}

void NeuralNetworkNode::_bind_methods() {
    using namespace godot;
    ClassDB::bind_method(D_METHOD("add_layer", "input_size", "output_size", "activation"), &NeuralNetworkNode::add_layer);
    ClassDB::bind_method(D_METHOD("forward", "input"), &NeuralNetworkNode::forward);
    ClassDB::bind_method(D_METHOD("backward", "error"), &NeuralNetworkNode::backward);
    ClassDB::bind_method(D_METHOD("predict", "input"), &NeuralNetworkNode::predict);
    ClassDB::bind_method(D_METHOD("model_summary"), &NeuralNetworkNode::model_summary);
    ClassDB::bind_method(D_METHOD("copy_weights", "source"), &NeuralNetworkNode::copy_weights);
    ClassDB::bind_method(D_METHOD("set_learning_rate", "lr"), &NeuralNetworkNode::set_learning_rate);
    ClassDB::bind_method(D_METHOD("get_learning_rate"), &NeuralNetworkNode::get_learning_rate);
    ClassDB::bind_method(D_METHOD("set_verbosity", "level"), &NeuralNetworkNode::set_verbosity);
    ClassDB::bind_method(D_METHOD("get_verbosity"), &NeuralNetworkNode::get_verbosity);
    ClassDB::bind_method(D_METHOD("set_layers", "layers"), &NeuralNetworkNode::set_layers);
    ClassDB::bind_method(D_METHOD("get_layers"), &NeuralNetworkNode::get_layers);
    ClassDB::bind_method(D_METHOD("set_batch_size", "batch_size"), &NeuralNetworkNode::set_batch_size);
    ClassDB::bind_method(D_METHOD("get_batch_size"), &NeuralNetworkNode::get_batch_size);
    ClassDB::bind_method(D_METHOD("build_model"), &NeuralNetworkNode::build_model);
    ClassDB::bind_method(D_METHOD("set_optimizer", "name"), &NeuralNetworkNode::set_optimizer);
	ClassDB::bind_method(D_METHOD("get_optimizer"), &NeuralNetworkNode::get_optimizer);


    // Inspector-visible properties
    ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "layers",
        PROPERTY_HINT_NONE, "",
        PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_EDITOR),
        "set_layers", "get_layers");


    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "learning_rate",
        PROPERTY_HINT_RANGE, "0.0,1.0,0.0001,precision:6"),
        "set_learning_rate", "get_learning_rate");

    ADD_PROPERTY(PropertyInfo(Variant::INT, "batch_size",
			PROPERTY_HINT_RANGE, "1,1024,1"),
			"set_batch_size", "get_batch_size");

    ADD_PROPERTY(PropertyInfo(Variant::INT, "verbosity",
        PROPERTY_HINT_RANGE, "0,3,1"),
        "set_verbosity", "get_verbosity");

    ADD_PROPERTY(
    	PropertyInfo(Variant::STRING, "optimizer"),
    	"set_optimizer",
    	"get_optimizer"
	);
}

void NeuralNetworkNode::add_layer(int input_size, int output_size, godot::String activation) {
    core->add_layer(input_size, output_size, activation.utf8().get_data());
}

godot::Array NeuralNetworkNode::forward(godot::Array input) {
    if (core->get_layers().empty()) {
        Logger::error_raise("NeuralNetworkNode::forward() - no layers defined");
        return godot::Array();
    }
    if (input.is_empty()) {
        Logger::error_raise("NeuralNetworkNode::forward() - empty input");
        return godot::Array();
    }

    const int expected_dim = core->get_layers().front().get_input_size();
    int provided_dim = (input[0].get_type() == godot::Variant::ARRAY)
        ? ((godot::Array)input[0]).size()
        : input.size();

    if (provided_dim != expected_dim) {
        std::ostringstream msg;
        msg << "Input dim mismatch (expected " << expected_dim << ", got " << provided_dim << ")";
        Logger::error_raise(msg.str());
        return godot::Array();
    }

    Eigen::MatrixXf x = godot_to_eigen(input, batch_size);
    Eigen::MatrixXf output = core->forward(x);
    return eigen_to_godot(output);
}

void NeuralNetworkNode::backward(godot::Array error) {
    Eigen::MatrixXf grad = godot_to_eigen(error, batch_size);

    if (grad.size() == 0 || !grad.allFinite()) {
        Logger::warn("NeuralNetworkNode::backward() - invalid gradient input");
        return;
    }

    core->backward(grad);
}

godot::Array NeuralNetworkNode::predict(godot::Array input) {
    if (core->get_layers().empty()) {
        Logger::error_raise("NeuralNetworkNode::predict() - no layers defined");
        return godot::Array();
    }

    if (input.is_empty()) {
        Logger::error_raise("NeuralNetworkNode::predict() - empty input");
        return godot::Array();
    }

    int actual_batch = input.size();
    Eigen::MatrixXf x = godot_to_eigen(input, actual_batch);
    Eigen::MatrixXf output = core->predict(x);
    return eigen_to_godot(output);
}

void NeuralNetworkNode::set_optimizer(godot::String name) {
    core->set_optimizer(name.to_lower().utf8().get_data());
}

godot::String NeuralNetworkNode::get_optimizer() const {
    return godot::String(core->get_optimizer_name().c_str());
}

void NeuralNetworkNode::set_learning_rate(double lr) {
    core->set_learning_rate(lr);
}

double NeuralNetworkNode::get_learning_rate() const {
    return core->get_learning_rate();
}

void NeuralNetworkNode::set_verbosity(int level) {
    core->set_verbosity(level);
}

void NeuralNetworkNode::copy_weights(const NeuralNetworkNode* source) {
    if (!source) {
        Logger::error("NeuralNetworkNode::copy_weights - null source");
        return;
    }
    core->copy_weights_from(*source->core);
}

void NeuralNetworkNode::set_layers(const godot::Array &p_layers) {
    layers_config = p_layers;
    if (layers_config.size() > 0)
        build_model();
}

godot::Array NeuralNetworkNode::get_layers() const {
    return layers_config;
}

void NeuralNetworkNode::build_model() {
    core = std::make_unique<NeuralNetworkCore>();
    for (int i = 0; i < layers_config.size(); ++i) {
        godot::Dictionary d = layers_config[i];
        int in_size = (int)d.get("input_size", 1);
        int out_size = (int)d.get("output_size", 1);
        godot::String act = d.get("activation", "relu");
        core->add_layer(in_size, out_size, act.utf8().get_data());
    }
    Logger::debug(1, "NeuralNetworkNode::build_model - model rebuilt");
}

void NeuralNetworkNode::model_summary() {
    Logger::info("----------- Model Summary -----------");
    const auto &layers = core->get_layers();
    for (int i = 0; i < (int)layers.size(); ++i) {
        const auto &layer = layers[i];
        std::ostringstream ss;
        ss << "Layer " << i << " | in=" << layer.get_input_size()
           << " out=" << layer.get_output_size()
           << " act=" << layer.get_activation_type();
        Logger::info(ss.str());
    }
    Logger::info("-------------------------------------");
}