#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Dense>
#include <functional>
#include <string>
#include <tuple>
#include "utility/logger.h"
#include <models/neural_network/activations/activations.h>

class Layer {
private:
    // Parameters
    Eigen::MatrixXf weights;
    Eigen::MatrixXf biases;

    // Cached forward data
    Eigen::MatrixXf input;
    Eigen::MatrixXf output;
    Eigen::MatrixXf grad_z;

    // Gradients
    Eigen::MatrixXf dW;
    Eigen::MatrixXf db;

    // Momentum buffers (per layer)
    Eigen::MatrixXf mW;
    Eigen::MatrixXf mb;

    // Config
    float lr = 0.01f;
    bool squash_enabled = false;
    float squash_scale_in = 10.0f;
    float squash_scale_out = 10.0f;
    std::string activation_type;

    // Activation functions
    std::function<Eigen::MatrixXf(const Eigen::MatrixXf&)> activation_func;
    std::function<Eigen::MatrixXf(const Eigen::MatrixXf&)> derivative_func;

public:
    int verbosity = 0;

    Layer(int input_size, int out_features, float learning_rate, const std::string& activation);
    ~Layer();

    // Core
    Eigen::MatrixXf forward(const Eigen::MatrixXf& X);
    Eigen::MatrixXf backward_compute(const Eigen::MatrixXf& loss_grad);
    void apply_update();
    void normalize_gradients(float scale);

    // Utilities
    void copy_weights(const Layer& source);
    void set_learning_rate(float lr);
    void set_verbosity(int v);
    void set_output_squash(bool enabled, float scale_in, float scale_out);

    // Getters
    int get_input_size() const;
    int get_output_size() const;
    std::string get_activation_type() const { return activation_type; }
    Eigen::MatrixXf get_weights() const { return weights; }
    Eigen::MatrixXf get_biases() const { return biases; }
    Eigen::MatrixXf get_dW() const { return dW; }
    Eigen::MatrixXf get_db() const { return db; }

private:
    std::tuple<Eigen::MatrixXf, Eigen::MatrixXf> init_weights(int in, int out, const std::string& activation);
};

#endif // LAYER_H