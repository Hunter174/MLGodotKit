#include "ce_loss_node.h"
#include "utility/utils.h"

using namespace godot;
using namespace Utils;

void CELossNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("forward", "prediction", "target"), &CELossNode::forward);
    ClassDB::bind_method(D_METHOD("backward"), &CELossNode::backward);
}

float CELossNode::forward(Array prediction, Array target) {

    int batch = prediction.size();
    prediction_cache = godot_to_eigen(prediction, batch);
    target_cache = godot_to_eigen(target, batch);

    Eigen::MatrixXf p = prediction_cache.array().max(1e-7f).min(1.0f);
    Eigen::MatrixXf loss = -(target_cache.array() * p.array().log());

    return loss.rowwise().sum().mean();
}

Array CELossNode::backward() {
    Eigen::MatrixXf grad = prediction_cache - target_cache;
    return eigen_to_godot(grad);
}
