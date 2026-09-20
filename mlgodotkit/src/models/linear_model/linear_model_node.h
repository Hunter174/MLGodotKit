#ifndef LINEAR_MODEL_NODE_H
#define LINEAR_MODEL_NODE_H

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include "models/linear_model/linear_model_core.h"
#include "utility/utils.h"
#include <Eigen/Dense>
#include <vector>
#include <memory>

class LinearModelNode : public godot::Node {
	GDCLASS(LinearModelNode, godot::Node);

private:
    std::unique_ptr<LinearModelCore> core;

protected:
    static void _bind_methods();

public:
    LinearModelNode();
    ~LinearModelNode();

    void initialize(int input_size);
    godot::Array predict(godot::Array input);
    void train(godot::Array inputs, godot::Array targets, int epochs);
    
    void set_learning_rate(double lr);
};

#endif // LINEAR_MODEL_NODE_H