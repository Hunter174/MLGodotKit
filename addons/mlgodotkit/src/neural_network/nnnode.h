#ifndef NNNODE_H
#define NNNODE_H

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include "layer/layer.h"

class NNNode : public godot::Node {
    GDCLASS(NNNode, godot::Node);

private:
	// Hyper parameters
	float learning_rate = 0.01;

    std::vector<Layer> layers;

    // Helper Functions
    Eigen::MatrixXd godot_to_eigen(godot::Array array);
    godot::Array eigen_to_godot(Eigen::MatrixXd matrix);

protected:
    static void _bind_methods();

public:
    NNNode();
    ~NNNode();

    void add_layer(int input_size, int output_size, godot::String activation);

    godot::Array forward(godot::Array input);
    void backward(godot::Array error);

    // Debug Helpers
    void model_summary();

    //Getters and Setters
    void set_learning_rate(float lr){learning_rate = lr;}

};

#endif // NNNODE_H
