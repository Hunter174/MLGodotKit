#ifndef NEURAL_NETWORK_NODE_H
#define NEURAL_NETWORK_NODE_H

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/core/class_db.hpp>
#include "models/neural_network/neural_network_core.h"
#include "utility/logger.h"
#include "utility/utils.h"
#include <sstream>
#include <iomanip>
#include <cmath>
#include <memory>

class NeuralNetworkNode : public godot::Node {
    GDCLASS(NeuralNetworkNode, godot::Node);

private:
    std::unique_ptr<NeuralNetworkCore> core;
    godot::Array layers_config;
    int verbosity = 0;
    int batch_size = 1;
    godot::String optimizer_name = "adam";

protected:
    static void _bind_methods();

public:
    NeuralNetworkNode();
    ~NeuralNetworkNode();

    void add_layer(int input_size, int output_size, godot::String activation);
    godot::Array forward(godot::Array input);
    void backward(godot::Array error);
    godot::Array predict(godot::Array input);

    void model_summary();
    void copy_weights(const NeuralNetworkNode* source);

    void set_verbosity(int level);
    int get_verbosity() const { return verbosity; }
    void set_learning_rate(double lr);
    double get_learning_rate() const;
    void set_batch_size(int bs) { batch_size = bs; }
    int get_batch_size() const { return batch_size; }
    NeuralNetworkCore* get_core() { return core.get(); }
    void set_optimizer(godot::String name);
    godot::String get_optimizer() const;

    void set_layers(const godot::Array &p_layers);
    godot::Array get_layers() const;
    void build_model();
};

#endif // NEURAL_NETWORK_NODE_H
