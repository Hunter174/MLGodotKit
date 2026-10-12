#include "neural_network_node.h"
#include <cmath>
#include <limits>

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
    ClassDB::bind_method(D_METHOD("get_snapshot"), &NeuralNetworkNode::get_snapshot);
    ClassDB::bind_method(D_METHOD("load_snapshot", "snapshot"), &NeuralNetworkNode::load_snapshot);
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
    godot::Dictionary config;
    config["input_size"] = input_size;
    config["output_size"] = output_size;
    config["activation"] = activation;
    layers_config.append(config);
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

godot::Dictionary NeuralNetworkNode::get_snapshot() const {
    using namespace godot;
    Dictionary snapshot;
    if (core->get_layers().empty() || core->get_layers().size() > 128 ||
        core->get_optimizer_name() != "adam" || !std::isfinite(core->get_learning_rate()) ||
        core->get_learning_rate() <= 0.0 || core->get_learning_rate() > 1.0) return snapshot;
    int64_t total_parameters = 0;
    int previous_output = 0;
    Array exported;
    for (const auto &layer : core->get_layers()) {
        const int in = layer.get_input_size(), out = layer.get_output_size();
        const auto &activation = layer.get_activation_type();
        total_parameters += static_cast<int64_t>(in) * out + out;
        if (in < 1 || in > 2048 || out < 1 || out > 2048 || total_parameters > 1000000 ||
            (previous_output != 0 && in != previous_output) ||
            (activation != "relu" && activation != "leaky_relu" && activation != "sigmoid" && activation != "linear")) return {};
        previous_output = out;
        Dictionary item;
        item["input_size"] = layer.get_input_size();
        item["output_size"] = layer.get_output_size();
        item["activation"] = String(layer.get_activation_type().c_str());
        Array rows;
        const auto &w = layer.get_weights();
        for (int i = 0; i < w.rows(); ++i) {
            Array row;
            for (int j = 0; j < w.cols(); ++j) {
                if (!std::isfinite(w(i, j))) return {};
                row.append(static_cast<double>(w(i, j)));
            }
            rows.append(row);
        }
        Array bias_row;
        const auto &b = layer.get_biases();
        for (int j = 0; j < b.cols(); ++j) {
            if (!std::isfinite(b(0, j))) return {};
            bias_row.append(static_cast<double>(b(0, j)));
        }
        Array biases;
        biases.append(bias_row);
        item["weights"] = rows;
        item["biases"] = biases;
        exported.append(item);
    }
    snapshot["version"] = 1;
    snapshot["optimizer"] = String(core->get_optimizer_name().c_str());
    snapshot["learning_rate"] = core->get_learning_rate();
    snapshot["layers"] = exported;
    return snapshot;
}

namespace {
bool snapshot_integer(const godot::Variant &value, int64_t &out) {
    if (value.get_type() != godot::Variant::INT && value.get_type() != godot::Variant::FLOAT) return false;
    const double number = static_cast<double>(value);
    if (!std::isfinite(number) || number < 0.0 || number > 1000000.0 || number != std::trunc(number)) return false;
    out = static_cast<int64_t>(number);
    return true;
}
bool snapshot_number(const godot::Variant &value, float &out) {
    if (value.get_type() != godot::Variant::INT && value.get_type() != godot::Variant::FLOAT) return false;
    const double number = static_cast<double>(value);
    if (!std::isfinite(number) || std::abs(number) > std::numeric_limits<float>::max()) return false;
    out = static_cast<float>(number);
    return std::isfinite(out);
}
}

bool NeuralNetworkNode::load_snapshot(const godot::Dictionary &snapshot) {
    using namespace godot;
    int64_t version = 0;
    if (!snapshot.has("version") || !snapshot_integer(snapshot["version"], version) ||
        version != 1 || !snapshot.has("layers") ||
        snapshot["layers"].get_type() != Variant::ARRAY || !snapshot.has("optimizer") ||
        snapshot["optimizer"].get_type() != Variant::STRING ||
        String(snapshot["optimizer"]) != "adam" || !snapshot.has("learning_rate")) return false;
    const Variant rate_value = snapshot["learning_rate"];
    float rate = 0.0f;
    if (!snapshot_number(rate_value, rate) || rate <= 0.0f || rate > 1.0f) return false;
    const Array entries = snapshot["layers"];
    if (entries.is_empty() || entries.size() > 128) return false;
    auto replacement = std::make_unique<NeuralNetworkCore>();
    Array rebuilt_config;
    int previous_output = 0;
    int64_t total_parameters = 0;
    for (int layer_index = 0; layer_index < entries.size(); ++layer_index) {
        if (entries[layer_index].get_type() != Variant::DICTIONARY) return false;
        const Dictionary item = entries[layer_index];
        if (!item.has("input_size") || !item.has("output_size") || !item.has("activation") ||
            !item.has("weights") || !item.has("biases") || item["activation"].get_type() != Variant::STRING ||
            item["weights"].get_type() != Variant::ARRAY || item["biases"].get_type() != Variant::ARRAY) return false;
        int64_t in = 0, out = 0;
        if (!snapshot_integer(item["input_size"], in) || !snapshot_integer(item["output_size"], out)) return false;
        if (in < 1 || in > 2048 || out < 1 || out > 2048 ||
            (layer_index > 0 && in != previous_output)) return false;
        previous_output = static_cast<int>(out);
        total_parameters += in * out + out;
        if (total_parameters > 1000000) return false;
        const String activation = item["activation"];
        if (activation != "linear" && activation != "relu" && activation != "leaky_relu" && activation != "sigmoid") return false;
        const Array rows = item["weights"];
        const Array biases = item["biases"];
        if (rows.size() != in || biases.size() != 1 || biases[0].get_type() != Variant::ARRAY) return false;
        const Array bias_row = biases[0];
        if (bias_row.size() != out) return false;
        Eigen::MatrixXf weights(static_cast<int>(in), static_cast<int>(out));
        Eigen::MatrixXf bias(1, static_cast<int>(out));
        for (int i = 0; i < in; ++i) {
            if (rows[i].get_type() != Variant::ARRAY) return false;
            const Array row = rows[i];
            if (row.size() != out) return false;
            for (int j = 0; j < out; ++j) {
                float value = 0.0f;
                if (!snapshot_number(row[j], value)) return false;
                weights(i, j) = value;
            }
        }
        for (int j = 0; j < out; ++j) {
            float value = 0.0f;
            if (!snapshot_number(bias_row[j], value)) return false;
            bias(0, j) = value;
        }
        replacement->add_layer(static_cast<int>(in), static_cast<int>(out), activation.utf8().get_data());
        auto &layer = replacement->mutable_layers().back();
        layer.get_weights() = std::move(weights);
        layer.get_biases() = std::move(bias);
        Dictionary config;
        config["input_size"] = in;
        config["output_size"] = out;
        config["activation"] = activation;
        rebuilt_config.append(config);
    }
    replacement->set_learning_rate(rate);
    replacement->set_optimizer("adam");
    core = std::move(replacement); // Atomic commit: failed validation never changes the old model.
    layers_config = rebuilt_config;
    optimizer_name = "adam";
    return true;
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