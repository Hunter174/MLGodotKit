#ifndef NNNODE_H
#define NNNODE_H

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/core/class_db.hpp>
#include "layer/layer.h"

class NNNode : public godot::Node {
    GDCLASS(NNNode, godot::Node);

private:
    double learning_rate = 0.001;   // safer default
    std::vector<Layer> layers;
    godot::Array layers_config;
    int verbosity = 0;
    int batch_size = 1;

protected:
    static void _bind_methods();

public:
    NNNode();
    ~NNNode();

    // Core
    void add_layer(int input_size, int output_size, godot::String activation);
    godot::Array forward(godot::Array input);
    void backward(godot::Array error);

    // Utilities
    void model_summary();
    void copy_weights(const NNNode* source);

    // Getters / Setters
    void set_verbosity(int level);
    int get_verbosity() const { return verbosity; }
    void set_learning_rate(double lr);
    double get_learning_rate() const { return learning_rate; }
    void set_batch_size(int bs) { batch_size = bs; }
    int get_batch_size() const { return batch_size; }

    // Inspector (Godot)
    void set_layers(const godot::Array &p_layers);
    godot::Array get_layers() const;
    void build_model();

};

#endif // NNNODE_H