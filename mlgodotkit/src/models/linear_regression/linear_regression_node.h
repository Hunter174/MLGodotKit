#ifndef LINEAR_REGRESSION_NODE_H
#define LINEAR_REGRESSION_NODE_H

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <Eigen/Dense>
#include "utility/utils.h"

class LinearRegressionNode : public godot::Node {
    GDCLASS(LinearRegressionNode, godot::Node);

private:
    // Includes bias as last element
    Eigen::VectorXf weights;

protected:
    static void _bind_methods();

public:
    LinearRegressionNode() = default;

    void fit(godot::Array inputs, godot::Array targets);
    godot::Array predict(godot::Array inputs);
};

#endif