#pragma once

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/array.hpp>
#include "losses/loss_core.h"
#include <memory>

using namespace godot;

class LossNode : public RefCounted {
    GDCLASS(LossNode, RefCounted);

protected:
    static void _bind_methods();

public:
    LossNode();
    ~LossNode();

    float forward(Array prediction, Array target);
    Array backward();

    // Core Access
    void set_core(std::unique_ptr<LossCore> p_core) { core = std::move(p_core); }
    LossCore* get_core() { return core.get(); }

private:
    std::unique_ptr<LossCore> core;
};