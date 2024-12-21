#ifndef NNNODE_H
#define NNNODE_H

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/godot.hpp>
#include <vector>
#include <Eigen/Dense>

using namespace godot;
using Eigen::MatrixXd;
using Eigen::VectorXd;

class NNNode : public Node {
    GDCLASS(NNNode, Node);

private:
    std::vector<MatrixXd> weights;
    std::vector<VectorXd> biases;
    float learning_rate = 0.01;
    String loss_function_type = "MSE";
    Array model_architecture;
    double last_loss = 0.0;

    // Activation Functions
    VectorXd relu(const VectorXd& x) const;
    VectorXd relu_derivative(const VectorXd& x) const;
    VectorXd softmax(const VectorXd& x) const;

    // Loss Functions
    double mean_squared_error(const VectorXd& output, const VectorXd& target);
    double mean_absolute_error(const VectorXd& output, const VectorXd& target);

    // Forward and Weight Initialization
    VectorXd forward_pass(const VectorXd& input) const;
    MatrixXd init_weights(int rows, int cols);

protected:
    static void _bind_methods();

public:
    NNNode();
    ~NNNode();

    // Core Neural Network Operations
    void initialize(Array layer_sizes);
    Array predict(Array input);
    void train(Array input, Array target);

    // Training Configuration
    void set_optimizer(String optimizer_type, float learning_rate);
    void set_loss_function(String loss_function);

    // Accessors
    double get_loss() const;
    void decay_learning_rate();
    void copy_from(const NNNode* other);
};

#endif  // NNNODE_H
