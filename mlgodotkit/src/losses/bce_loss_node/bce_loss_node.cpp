#include "bce_loss_node.h"
#include "utility/utils.h"

using namespace godot;
using namespace Utils;

void BCELossNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("forward", "prediction", "target"), &BCELossNode::forward);
    ClassDB::bind_method(D_METHOD("backward"), &BCELossNode::backward);
}

float BCELossNode::forward(Array prediction, Array target) {

    int batch = prediction.size();
    prediction_cache = godot_to_eigen(prediction, batch);
    target_cache = godot_to_eigen(target, batch);

    // Clamp for numerical stability
    Eigen::MatrixXf p = prediction_cache.array().max(1e-7f).min(1.0f - 1e-7f);
    Eigen::MatrixXf loss =
        - (target_cache.array() * p.array().log()
        + (1.0f - target_cache.array())
        * (1.0f - p.array()).log());

    return loss.mean();
}

Array BCELossNode::backward() {
    // For sigmoid output, simplified gradient:
    Eigen::MatrixXf grad = prediction_cache - target_cache;
    return eigen_to_godot(grad);
}