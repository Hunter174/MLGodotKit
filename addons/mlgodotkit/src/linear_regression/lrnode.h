#ifndef LRNODE_H
#define LRNODE_H

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include "utility/utils.h"
#include <Eigen/Dense>
#include <vector>

class LRNode : public godot::Node {
    GDCLASS(LRNode, godot::Node);

private:
    Eigen::VectorXd weights;
    double bias;
    double learning_rate;
    int num_features;

protected:
    static void _bind_methods();

public:
    LRNode();
    ~LRNode();

    void initialize(int input_size);
    godot::Array predict(godot::Array input);
    void train(godot::Array inputs, godot::Array targets, int epochs);
    double compute_loss(const Eigen::VectorXd &predictions, const Eigen::VectorXd &targets);
    Eigen::VectorXd compute_gradient(const Eigen::VectorXd &predictions, const Eigen::VectorXd &targets, const Eigen::MatrixXd &inputs);

    void set_learning_rate(double lr);
};

#endif // LRNODE_H
