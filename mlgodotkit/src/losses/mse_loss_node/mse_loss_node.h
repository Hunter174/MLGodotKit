#pragma once

#include "losses/loss_node/loss_node.h"
#include <Eigen/Dense>

class MSELossNode : public LossNode {
    GDCLASS(MSELossNode, LossNode);

private:
    Eigen::MatrixXf diff;
    Eigen::MatrixXf grad;

protected:
    static void _bind_methods();

public:
    float forward(godot::Array prediction, godot::Array target) override;
    godot::Array backward() override;
};