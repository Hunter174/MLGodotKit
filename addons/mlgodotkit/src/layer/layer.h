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

    static constexpr double epsilon = 1e-6; // Small value to avoid overflow or division by zero


    std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> activation_func;
    std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> derivative_func;

public:
  	//Public Variables
    int verbosity = 1;
    double lr = 0.01;

    Layer(int input_size, int out_features, double learning_rate, std::string activation_type);

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

	void set_learning_rate(double learning_rate) { this->lr = learning_rate; }
    void set_verbosity(int verbosity) { this->verbosity = verbosity;};
};

#endif // LAYER_H
