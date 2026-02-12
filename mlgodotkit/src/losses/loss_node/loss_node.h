#pragma once

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/array.hpp>

using namespace godot;

class LossNode : public RefCounted {
    GDCLASS(LossNode, RefCounted);

protected:
    static void _bind_methods();

public:
    virtual float forward(Array prediction, Array target);
    virtual Array backward();
};