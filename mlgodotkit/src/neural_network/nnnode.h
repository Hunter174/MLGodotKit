#ifndef NNNODE_H
#define NNNODE_H

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include "layer/layer.h"
#include "utility/utils.h"

class NNNode : public godot::Node {
    GDCLASS(NNNode, godot::Node);

private:
	// Hyper parameters
	double learning_rate = 0.01;

    std::vector<Layer> layers;

protected:
    static void _bind_methods();

public:

	//Debug flags
    int verbosity = 0;

    NNNode();
    ~NNNode();

    void add_layer(int input_size, int output_size, godot::String activation);

    godot::Array forward(godot::Array input);
    void backward(godot::Array error);

    // Debug Helpers
    void model_summary();

    //Getters and Setters
    void set_verbosity(int verbosity);
    void set_learning_rate(double lr);

    void copy_weights(const NNNode* source);

};

#endif // NNNODE_H