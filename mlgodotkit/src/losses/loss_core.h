#pragma once

#include <godot_cpp/variant/array.hpp>
#include <vector>

class LossCore {
public:
    virtual ~LossCore() = default;
    virtual float forward(const std::vector<float>& prediction, const std::vector<float>& target) = 0;
    virtual std::vector<float> backward() = 0;
};
