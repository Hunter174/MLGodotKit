#ifndef NeuralNetworkCore_H
#define NeuralNetworkCore_H

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <string>
#include "models/neural_network/layer/layer.h"
#include "optimizers/optimizer/optimizer.h"
#include "optimizers/adam/adam.h"

class NeuralNetworkCore {
public:
    NeuralNetworkCore();
    ~NeuralNetworkCore() = default;

    // Core Operations
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input);
    void backward(const Eigen::MatrixXf& grad);
    Eigen::MatrixXf predict(const Eigen::MatrixXf& input) const;

    // Lifecycle & RL Support
    std::unique_ptr<NeuralNetworkCore> clone() const;
    void copy_weights_from(const NeuralNetworkCore& other);

    // Configuration
    void add_layer(int input_size, int output_size, const std::string& activation);
    void set_learning_rate(double lr);
    void set_optimizer(const std::string& name);
    void set_verbosity(int level);

    // Accessors
    const std::vector<Layer>& get_layers() const { return layers; }
    double get_learning_rate() const { return learning_rate; }
    std::string get_optimizer_name() const { return optimizer_name; }
    int get_verbosity() const { return verbosity; }

private:
    std::vector<Layer> layers;
    std::unique_ptr<Optimizer> optimizer;
    double learning_rate = 0.001;
    std::string optimizer_name = "adam";
    int verbosity = 0;
};

#endif // NeuralNetworkCore_H