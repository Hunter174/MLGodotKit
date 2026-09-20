#include "neural_network_core.h"
#include "utility/logger.h"

using namespace Utils;

NeuralNetworkCore::NeuralNetworkCore() {
    optimizer = std::make_unique<AdamCore>();
}

Eigen::MatrixXf NeuralNetworkCore::forward(const Eigen::MatrixXf& input) {
    if (layers.empty()) {
        Logger::error_raise("NeuralNetworkCore::forward() - no layers defined");
        return Eigen::MatrixXf();
    }

    Eigen::MatrixXf x = input;
    for (auto &layer : layers) {
        x = layer.forward(x);
    }
    return x;
}

void NeuralNetworkCore::backward(const Eigen::MatrixXf& grad) {
    if (layers.empty()) return;

    Eigen::MatrixXf g = grad;
    if (g.size() == 0 || !g.allFinite()) {
        Logger::warn("NeuralNetworkCore::backward() - invalid gradient input");
        return;
    }

    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
        g = layers[i].backward_compute(g);
    }

    float global_norm = 0.0f;
    for (auto &layer : layers) {
        global_norm += layer.get_dW().squaredNorm() + layer.get_db().squaredNorm();
    }
    global_norm = std::sqrt(global_norm);

    const float max_norm = 2.5f;
    float scale = 1.0f;
    if (global_norm > max_norm && global_norm > 0.0f) {
        scale = max_norm / global_norm;
    }

    for (auto &layer : layers) {
        layer.normalize_gradients(scale);
    }

    if (!optimizer) return;

    optimizer->begin_step();
    int param_index = 0;
    for (auto& layer : layers) {
        optimizer->update(layer.get_weights(), layer.get_dW(), param_index++);
        optimizer->update(layer.get_biases(), layer.get_db(), param_index++);
    }
}

Eigen::MatrixXf NeuralNetworkCore::predict(const Eigen::MatrixXf& input) const {
    if (layers.empty()) {
        Logger::error_raise("NeuralNetworkCore::predict() - no layers defined");
        return Eigen::MatrixXf();
    }

    Eigen::MatrixXf x = input;
    for (const auto &layer : layers) {
        x = layer.forward(x);
    }
    return x;
}

std::unique_ptr<NeuralNetworkCore> NeuralNetworkCore::clone() const {
    auto new_core = std::make_unique<NeuralNetworkCore>();
    new_core->copy_weights_from(*this);
    new_core->learning_rate = this->learning_rate;
    new_core->optimizer_name = this->optimizer_name;
    new_core->verbosity = this->verbosity;
    return new_core;
}

void NeuralNetworkCore::copy_weights_from(const NeuralNetworkCore& other) {
    if (other.layers.size() != layers.size()) {
        // If sizes differ, we must rebuild layers to match
        this->layers.clear();
        for (const auto& l : other.layers) {
            this->add_layer(l.get_input_size(), l.get_output_size(), l.get_activation_type());
        }
    }
    
    for (size_t i = 0; i < layers.size(); ++i) {
        layers[i].copy_weights(other.layers[i]);
    }
}

void NeuralNetworkCore::add_layer(int input_size, int output_size, const std::string& activation) {
    LayerCore layer(input_size, output_size, activation);
    layer.set_verbosity(verbosity);
    layers.push_back(layer);
}

void NeuralNetworkCore::set_learning_rate(double lr) {
    learning_rate = lr;
    if (optimizer) {
        optimizer->set_learning_rate(lr);
    }
}

void NeuralNetworkCore::set_optimizer(const std::string& name) {
    optimizer_name = name;
    if (name == "adam") {
        optimizer = std::make_unique<AdamCore>();
        optimizer->set_learning_rate(learning_rate);
    } else {
        Logger::error_raise("Unknown optimizer: " + name);
    }
}

void NeuralNetworkCore::set_verbosity(int level) {
    verbosity = level;
    for (auto &layer : layers) {
        layer.set_verbosity(level);
    }
}
