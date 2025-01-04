#ifndef TESTNNNODE_H
#define TESTNNNODE_H

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/method_bind.hpp>
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/classes/node.hpp>
#include <vector>
#include <functional>
#include <Eigen/Dense>
#include <sstream>
#include <cmath>
#include <random>
#include <algorithm>

class TestNNNode : public godot::Node {
    GDCLASS(TestNNNode, godot::Node);

private:
    Eigen::MatrixXd input;
    Eigen::MatrixXd labels;

    // Layer 1 (hidden)
    Eigen::MatrixXd W1;
    Eigen::MatrixXd b1;

    // Layer 2 (output)
    Eigen::MatrixXd W2;
    Eigen::MatrixXd b2;

    double learning_rate;
    int epochs;

protected:
  static void _bind_methods();

public:
    TestNNNode();
    ~TestNNNode();

    void test();

    // Activation Functions
    static Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& x);
    static Eigen::MatrixXd sigmoid_derivative(const Eigen::MatrixXd& x);

    static Eigen::MatrixXd relu(const Eigen::MatrixXd& x);
    static Eigen::MatrixXd relu_derivative(const Eigen::MatrixXd& x);

    // Loss functions
    double compute_loss(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& y);
    Eigen::MatrixXd loss_derivative(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& y);

    // Forward and backward propagation
    Eigen::MatrixXd forward(const Eigen::MatrixXd& X);
    void backward(const Eigen::MatrixXd& loss_gradient);
    void train();
};

#endif // TESTNNNODE_H