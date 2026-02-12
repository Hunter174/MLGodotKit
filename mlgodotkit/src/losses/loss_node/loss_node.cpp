#include "loss_node.h"

void LossNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("forward", "prediction", "target"), &LossNode::forward);
    ClassDB::bind_method(D_METHOD("backward"), &LossNode::backward);
}

float LossNode::forward(Array prediction, Array target) {
    ERR_PRINT("LossNode::forward() not implemented.");
    return 0.0f;
}

Array LossNode::backward() {
    ERR_PRINT("LossNode::backward() not implemented.");
    return Array();
}