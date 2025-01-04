#ifndef NNNODE_H
#define NNNODE_H

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

class NNNode : public godot::Node {
    GDCLASS(NNNode, godot::Node);

private:
    std::vector<Eigen::MatrixXd> weights;
    std::vector<Eigen::VectorXd> biases;
    std::vector<Eigen::VectorXd> zs;
    std::vector<std::function<Eigen::VectorXd(const Eigen::VectorXd&)>> activations;
    std::vector<std::function<Eigen::VectorXd(const Eigen::VectorXd&)>> activation_derivatives;
    std::vector<Eigen::VectorXd> activations_values;
    godot::String _hidden_activation;
    godot::String _output_activation;

	// Hyper parameters
    float learning_rate = 0.01;
    godot::String loss_function_type = "MSE";
    godot::Array model_architecture;
    double last_loss = 0.0;
    static constexpr double epsilon = 1e-10; // Small term for numeric stability

    // Loss function and its derivative
    std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)> loss_function;
    std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> loss_derivative;

    // Activation functions
    static Eigen::VectorXd relu(const Eigen::VectorXd& x);
    static Eigen::VectorXd relu_derivative(const Eigen::VectorXd& x);
    static Eigen::VectorXd softmax(const Eigen::VectorXd& x);
    static Eigen::VectorXd sigmoid(const Eigen::VectorXd& x);
    static Eigen::VectorXd sigmoid_derivative(const Eigen::VectorXd& x);
    static Eigen::VectorXd tanh(const Eigen::VectorXd& x);
    static Eigen::VectorXd tanh_derivative(const Eigen::VectorXd& x);

    // Loss functions
    static double mean_squared_error(const Eigen::VectorXd& output, const Eigen::VectorXd& target);
    static Eigen::VectorXd mean_squared_error_derivative(const Eigen::VectorXd& output, const Eigen::VectorXd& target);
    static double binary_cross_entropy(const Eigen::VectorXd& output, const Eigen::VectorXd& target);
    static Eigen::VectorXd binary_cross_entropy_derivative(const Eigen::VectorXd& output, const Eigen::VectorXd& target);

    // Helper functions
    Eigen::MatrixXd init_weights(int rows, int cols, const godot::String &activation_type);
    void set_verbosity(int level);
    void debug_print(int level, const std::string& message);
    void model_summary();
    std::string vector_to_string(const Eigen::VectorXd& vec);
    Eigen::VectorXd array_to_eigenvec(const godot::Array);
    Eigen::MatrixXd array_to_eigenmat(const godot::Array);

    // Verbosity level for debugging
    int verbosity_level = 0;

protected:
    static void _bind_methods();

public:
    NNNode();
    ~NNNode();

    // Initialization
    void initialize(godot::Array layer_sizes, godot::String hidden_activation, godot::String output_activation);

    // Core functions
    godot::Array predict(godot::Array input);
    void train(godot::Array input, godot::Array target, int batch_size);
    void backward(const Eigen::VectorXd& target);
    Eigen::VectorXd forward(const Eigen::VectorXd& input, bool store_intermediate = true);

    // Configurations
    void set_optimizer(godot::String optimizer_type, float lr);
    void set_loss_function(godot::String loss_function_type);

    // Utilities
    double get_loss() const;
    void decay_learning_rate();
    void copy_from(const NNNode* other);
    std::pair<std::function<Eigen::VectorXd(const Eigen::VectorXd&)>, std::function<Eigen::VectorXd(const Eigen::VectorXd&)>> get_activation_function(const godot::String& activation);
    Eigen::VectorXd clip_gradient(const Eigen::VectorXd& gradient, double max_norm);
};

#endif // NNNODE_H