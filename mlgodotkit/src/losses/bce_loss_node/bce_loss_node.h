#pragma once

#include "losses/loss_node/loss_node.h"
#include <Eigen/Dense>

class BCELossNode : public LossNode {
    GDCLASS(BCELossNode, LossNode);

private:
    Eigen::MatrixXf prediction_cache;
    Eigen::MatrixXf target_cache;

protected:
    static void _bind_methods();

public:
    float forward(godot::Array prediction,godot::Array target) override;
    godot::Array backward() override;
};
