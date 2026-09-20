#pragma once
#include <Eigen/Dense>

class Optimizer {
public:
    virtual ~Optimizer() = default;

    virtual void update(
        Eigen::MatrixXf& param,
        const Eigen::MatrixXf& grad,
        int param_index) = 0;

    virtual void set_learning_rate(float lr) = 0;
    virtual void begin_step() {}
};