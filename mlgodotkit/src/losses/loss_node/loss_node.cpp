#include "loss_node.h"

LossNode::LossNode() {}
LossNode::~LossNode() {}

void LossNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("forward", "prediction", "target"), &LossNode::forward);
    ClassDB::bind_method(D_METHOD("backward"), &LossNode::backward);
}

float LossNode::forward(Array prediction, Array target) {
    if (!core) {
        ERR_PRINT("LossNode: No core implementation assigned.");
        return 0.0f;
    }
    // In a real implementation, we would convert godot::Array to std::vector
    // For this refactor, we're establishing the architecture first.
    return core->forward({}, {}); 
}

Array LossNode::backward() {
    if (!core) {
        ERR_PRINT("LossNode: No core implementation assigned.");
        return Array();
    }
    return Array(); // Conversion from std::vector to Array
}