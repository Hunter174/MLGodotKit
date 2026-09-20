#pragma once

#include "optimizers/optimizer/optimizer.h"
#include <vector>
#include <cmath>

class Adam : public Optimizer {
private:
    float lr = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    int timestep = 0;

    std::vector<Eigen::MatrixXf> m;
    std::vector<Eigen::MatrixXf> v;

public:
    void update(Eigen::MatrixXf& param, const Eigen::MatrixXf& grad, int index) override;
    void set_learning_rate(float learning_rate) override {lr = learning_rate;}
    void begin_step() override {timestep++;}
};
