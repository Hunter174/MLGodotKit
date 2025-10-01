#ifndef LAYER_H
#define LAYER_H

#include <sstream>
#include <Eigen/Dense>
#include <activations/activations.h>
#include "utility/logger.h"
#include "utility/godot_utility.h"

class Layer {
private:
    int precision = 8;
    int num_neurons;
    Eigen::MatrixXf weights;
    Eigen::MatrixXf biases;
    Eigen::MatrixXf input;
    Eigen::MatrixXf output;
    Eigen::MatrixXf grad_z;

    std::function<Eigen::MatrixXf(const Eigen::MatrixXf&)> activation_func;
    std::function<Eigen::MatrixXf(const Eigen::MatrixXf&)> derivative_func;

public:
  	//Public Variables
    int verbosity = 1;
    float lr = 0.01;

    Layer(int input_size, int out_features, float learning_rate, std::string activation_type);

    ~Layer();

    Eigen::MatrixXf forward(const Eigen::MatrixXf& X);
    Eigen::MatrixXf backward(const Eigen::MatrixXf& loss_gradient);

    std::string to_string() const;

    // Getters and Setters
    Eigen::MatrixXf get_weights() const { return weights; }
    Eigen::MatrixXf get_biases() const { return biases; }
    Eigen::MatrixXf get_input() const { return input; }
    Eigen::MatrixXf get_output() const { return output; }
    Eigen::MatrixXf get_gradient() const { return grad_z; }

	void set_learning_rate(float learning_rate) { this->lr = learning_rate; }
    void set_verbosity(int verbosity) { this->verbosity = verbosity;};

    // Utility
    Eigen::MatrixXf stable_round(const Eigen::MatrixXf& mat, int precision = 8, float threshold = 1e-4);
    std::tuple<Eigen::MatrixXf, Eigen::MatrixXf> init_weights(int input_size, int out_features, const std::string& activation_type);

	void copy_weights(const Layer& source);
    int get_input_size() const;
	int get_output_size() const;
};

#endif // LAYER_H
