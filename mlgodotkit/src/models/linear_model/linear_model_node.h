#ifndef LINEAR_MODEL_NODE_H
#define LINEAR_MODEL_NODE_H

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include "utility/utils.h"
#include <Eigen/Dense>
#include <vector>

class LinearModelNode : public godot::Node {
	GDCLASS(LinearModelNode, godot::Node);

private:
    Eigen::VectorXf weights;
    double bias;
    double learning_rate;
    int num_features;

protected:
    static void _bind_methods();

public:
    LinearModelNode();
    ~LinearModelNode();

    void initialize(int input_size);
    godot::Array predict(godot::Array input);
    void train(godot::Array inputs, godot::Array targets, int epochs);
    double compute_loss(const Eigen::VectorXf &predictions, const Eigen::VectorXf &targets);
    Eigen::VectorXf compute_gradient(const Eigen::VectorXf &predictions, const Eigen::VectorXf &targets, const Eigen::MatrixXf &inputs);

    void set_learning_rate(double lr);
};

#endif // LINEAR_MODEL_NODE_H