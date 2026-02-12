#include "mse_loss_node.h"
#include "utility/utils.h"

using namespace godot;
using namespace Utils;

void MSELossNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("forward", "prediction", "target"), &MSELossNode::forward);
    ClassDB::bind_method(D_METHOD("backward"), &MSELossNode::backward);
}

float MSELossNode::forward(Array prediction, Array target) {

    Eigen::MatrixXf pred = godot_to_eigen(prediction, prediction.size());
    Eigen::MatrixXf tgt  = godot_to_eigen(target, target.size());

    diff = pred - tgt;

    float loss = diff.array().square().mean();

    // Store gradient for backward()
    grad = 2.0f * diff / diff.rows();

    return loss;
}

Array MSELossNode::backward() {
    return eigen_to_godot(grad);
}