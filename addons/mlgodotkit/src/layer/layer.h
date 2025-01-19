#ifndef LAYER_H
#define LAYER_H

#include <godot_cpp/variant/utility_functions.hpp>
#include <sstream>
#include <Eigen/Dense>
#include "utility/utils.h"

class Layer {
private:
    int num_neurons;
    Eigen::MatrixXd weights;
    Eigen::MatrixXd biases;
    Eigen::MatrixXd input;
    Eigen::MatrixXd output;
    Eigen::MatrixXd grad_z;
    float lr = 0.01;

    std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> activation_func;
    std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> derivative_func;

public:
    Layer(int input_size, int out_features, float learning_rate, std::string activation_type);

    ~Layer();

    Eigen::MatrixXd forward(const Eigen::MatrixXd& X);
    Eigen::MatrixXd backward(const Eigen::MatrixXd& loss_gradient);

    static Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& x);
    static Eigen::MatrixXd sigmoid_derivative(const Eigen::MatrixXd& x);
    static Eigen::MatrixXd relu(const Eigen::MatrixXd& x);
    static Eigen::MatrixXd relu_derivative(const Eigen::MatrixXd& x);

    godot::String to_string() const;

    // Getters and Setters
    Eigen::MatrixXd get_weights() const { return weights; }
    Eigen::MatrixXd get_biases() const { return biases; }
    Eigen::MatrixXd get_input() const { return input; }
    Eigen::MatrixXd get_output() const { return output; }
    Eigen::MatrixXd get_gradient() const { return grad_z; }
};

#endif // LAYER_H
