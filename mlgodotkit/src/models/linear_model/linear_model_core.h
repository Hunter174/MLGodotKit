#ifndef LINEAR_MODEL_CORE_H
#define LINEAR_MODEL_CORE_H

#include <Eigen/Dense>
#include <vector>

class LinearModelCore {
public:
    LinearModelCore() : bias(0.0), learning_rate(0.01), num_features(0) {}

    void initialize(int input_size) {
        num_features = input_size;
        weights = Eigen::VectorXf::Zero(num_features);
        bias = 0.0;
    }

    float predict_single(const Eigen::VectorXf& input) {
        return weights.dot(input) + bias;
    }

    void train(const Eigen::MatrixXf& inputs, const Eigen::VectorXf& targets, int epochs) {
        for (int e = 0; e < epochs; ++e) {
            Eigen::VectorXf predictions = (inputs * weights).array() + bias;
            Eigen::VectorXf error = predictions - targets;
            
            Eigen::VectorXf grad_w = (inputs.transpose() * error) / inputs.rows();
            float grad_b = error.mean();

            weights -= learning_rate * grad_w;
            bias -= learning_rate * grad_b;
        }
    }

    // State
    Eigen::VectorXf weights;
    double bias;
    double learning_rate = 0.01;
    int num_features;
};

#endif // LINEAR_MODEL_CORE_H
