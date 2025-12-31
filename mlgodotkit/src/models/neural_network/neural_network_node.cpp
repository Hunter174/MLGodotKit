#include "neural_network_node.h"
#include "utility/logger.h"
#include "utility/utils.h"
#include <sstream>
#include <iomanip>
#include <cmath>

using namespace Utils;

NeuralNetworkNode::NeuralNetworkNode() {}
NeuralNetworkNode::~NeuralNetworkNode() {}

void NeuralNetworkNode::_bind_methods() {
    using namespace godot;
    ClassDB::bind_method(D_METHOD("add_layer", "input_size", "output_size", "activation"), &NeuralNetworkNode::add_layer);
    ClassDB::bind_method(D_METHOD("forward", "input"), &NeuralNetworkNode::forward);
    ClassDB::bind_method(D_METHOD("backward", "error"), &NeuralNetworkNode::backward);
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
}

void NeuralNetworkNode::add_layer(int input_size, int output_size, godot::String activation) {
    std::string act_type = activation.utf8().get_data();
    Layer layer(input_size, output_size, learning_rate, act_type);
    layer.set_verbosity(verbosity);
    layers.push_back(layer);
}

godot::Array NeuralNetworkNode::forward(godot::Array input) {
    if (layers.empty()) {
        Logger::error_raise("NeuralNetworkNode::forward() - no layers defined");
        return godot::Array();
    }
    if (input.is_empty()) {
        Logger::error_raise("NeuralNetworkNode::forward() - empty input");
        return godot::Array();
    }

    // Validate dimensions
    const int expected_dim = layers.front().get_input_size();
    int provided_dim = (input[0].get_type() == godot::Variant::ARRAY)
        ? ((godot::Array)input[0]).size()
        : input.size();

    if (provided_dim != expected_dim) {
        std::ostringstream msg;
        msg << "Input dim mismatch (expected " << expected_dim << ", got " << provided_dim << ")";
        Logger::error_raise(msg.str());
        return godot::Array();
    }

    // Convert to Eigen
    Eigen::MatrixXf x = godot_to_eigen(input, batch_size);

    // Forward pass
    for (auto &layer : layers)
        x = layer.forward(x);

    // Output: no artificial squashing
    return eigen_to_godot(x);
}

void NeuralNetworkNode::backward(godot::Array error) {
    Eigen::MatrixXf grad = godot_to_eigen(error, batch_size);
    if (grad.size() == 0 || !grad.allFinite()) {
        Logger::warn("NeuralNetworkNode::backward() - invalid gradient input");
        return;
    }

    // 1. Backprop through layers
    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i)
        grad = layers[i].backward_compute(grad);

    // 2. Compute global gradient norm (for clipping)
    float global_norm = 0.0f;
    for (auto &layer : layers)
        global_norm += layer.get_dW().squaredNorm() + layer.get_db().squaredNorm();
    global_norm = std::sqrt(global_norm);

    const float max_norm = 2.5f;
    float scale = 1.0f;
    if (global_norm > max_norm && global_norm > 0.0f) {
        scale = max_norm / global_norm;
        Logger::warn("⚠️ Gradient clip applied, norm = " + std::to_string(global_norm));
    }

    // 3. Scale gradients (no extra per-layer normalization)
    for (auto &layer : layers)
        layer.normalize_gradients(scale);

    // 4. Update weights (weight decay handled inside layer)
    for (auto &layer : layers)
        layer.apply_update();
}

void NeuralNetworkNode::set_learning_rate(double lr) {
    learning_rate = lr;
    for (auto &layer : layers)
        layer.set_learning_rate(lr);
}

void NeuralNetworkNode::set_verbosity(int level) {
    verbosity = level;
    Logger::set_verbosity(level);
    for (auto &layer : layers)
        layer.set_verbosity(level);
}

void NeuralNetworkNode::copy_weights(const NeuralNetworkNode* source) {
    if (!source || source->layers.size() != layers.size()) {
        Logger::error("NeuralNetworkNode::copy_weights - incompatible network sizes");
        return;
    }
    for (size_t i = 0; i < layers.size(); ++i)
        layers[i].copy_weights(source->layers[i]);
    Logger::info("NeuralNetworkNode::copy_weights - success");
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
    layers.clear();
    for (int i = 0; i < layers_config.size(); ++i) {
        godot::Dictionary d = layers_config[i];
        int in_size = (int)d.get("input_size", 1);
        int out_size = (int)d.get("output_size", 1);
        godot::String act = d.get("activation", "relu");
        add_layer(in_size, out_size, act);
    }
    Logger::info("NeuralNetworkNode::build_model - model rebuilt");
}

void NeuralNetworkNode::model_summary() {
    Logger::info("----------- Model Summary -----------");
    for (int i = 0; i < layers.size(); ++i) {
        const auto &layer = layers[i];
        std::ostringstream ss;
        ss << "Layer " << i << " | in=" << layer.get_input_size()
           << " out=" << layer.get_output_size()
           << " act=" << layer.get_activation_type();
        Logger::info(ss.str());
    }
    Logger::info("-------------------------------------");
}