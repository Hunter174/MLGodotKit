#ifndef NNNODE_H
#define NNNODE_H

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <vector>
#include <functional>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class NNNode : public godot::Node {
    GDCLASS(NNNode, Node);

private:
    std::vector<MatrixXd> weights;
    std::vector<VectorXd> biases;
    std::vector<VectorXd> zs;
    std::vector<std::function<VectorXd(const VectorXd&)>> activations;
    std::vector<std::function<VectorXd(const VectorXd&)>> activation_derivatives;
    std::vector<VectorXd> activations_values;

    float learning_rate = 0.01;
    godot::String loss_function_type = "MSE";
    godot::Array model_architecture;
    double last_loss = 0.0;

    // Loss function and its derivative
    std::function<double(const VectorXd&, const VectorXd&)> loss_function;
    std::function<VectorXd(const VectorXd&, const VectorXd&)> loss_derivative;

    // Activation functions
    static VectorXd relu(const VectorXd& x);
    static VectorXd relu_derivative(const VectorXd& x);
    static VectorXd softmax(const VectorXd& x);
    static VectorXd sigmoid(const VectorXd& x);
    static VectorXd sigmoid_derivative(const VectorXd& x);

    // Loss functions
    static double mean_squared_error(const VectorXd& output, const VectorXd& target);
    static VectorXd mean_squared_error_derivative(const VectorXd& output, const VectorXd& target);
    static double binary_cross_entropy(const VectorXd& output, const VectorXd& target);
    static VectorXd binary_cross_entropy_derivative(const VectorXd& output, const VectorXd& target);

    // Helper functions
    MatrixXd init_weights(int rows, int cols, const godot::String &activation_type);

protected:
    static void _bind_methods();

public:
    NNNode();
    ~NNNode();

    // Initialization
    void initialize(godot::Array layer_sizes, godot::String hidden_activation, godot::String output_activation);

    // Core functions
    godot::Array predict(godot::Array input);
    void train(godot::Array input, godot::Array target);
    void backward(const VectorXd& target);
    VectorXd forward(const VectorXd& input, bool store_intermediate = true);

    // Configurations
    void set_optimizer(godot::String optimizer_type, float lr);
    void set_loss_function(godot::String loss_function_type);

    // Utilities
    double get_loss() const;
    void decay_learning_rate();
    void copy_from(const NNNode* other);
    bool verbose = false;
};

#endif // NNNODE_H