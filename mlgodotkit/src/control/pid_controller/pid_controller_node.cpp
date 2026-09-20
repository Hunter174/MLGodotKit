#include "pid_controller_node.h"
#include <godot_cpp/core/class_db.hpp>
#include <algorithm>

using namespace godot;

PIDControllerNode::PIDControllerNode() {
    core = std::make_unique<PIDControllerCore>();
}

PIDControllerNode::~PIDControllerNode() {}

void PIDControllerNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("update", "setpoint", "measurement"), &PIDControllerNode::update);
    ClassDB::bind_method(D_METHOD("update_dt", "setpoint", "measurement", "dt"), &PIDControllerNode::update_dt);
    ClassDB::bind_method(D_METHOD("reset"), &PIDControllerNode::reset);

    ClassDB::bind_method(D_METHOD("set_kp", "value"), &PIDControllerNode::set_kp);
    ClassDB::bind_method(D_METHOD("get_kp"), &PIDControllerNode::get_kp);

    ClassDB::bind_method(D_METHOD("set_ki", "value"), &PIDControllerNode::set_ki);
    ClassDB::bind_method(D_METHOD("get_ki"), &PIDControllerNode::get_ki);

    ClassDB::bind_method(D_METHOD("set_kd", "value"), &PIDControllerNode::set_kd);
    ClassDB::bind_method(D_METHOD("get_kd"), &PIDControllerNode::get_kd);

    ClassDB::bind_method(D_METHOD("set_tau", "value"), &PIDControllerNode::set_tau);
    ClassDB::bind_method(D_METHOD("get_tau"), &PIDControllerNode::get_tau);

    ClassDB::bind_method(D_METHOD("set_limits", "min", "max"),  &PIDControllerNode::set_limits);
    ClassDB::bind_method(D_METHOD("get_limit_min"), &PIDControllerNode::get_limit_min);
    ClassDB::bind_method(D_METHOD("get_limit_max"), &PIDControllerNode::get_limit_max);

    ClassDB::bind_method(D_METHOD("set_sample_time", "value"), &PIDControllerNode::set_sample_time);
    ClassDB::bind_method(D_METHOD("get_sample_time"), &PIDControllerNode::get_sample_time);

    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "kp", PROPERTY_HINT_NONE, ""), "set_kp", "get_kp");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "ki", PROPERTY_HINT_NONE, ""), "set_ki", "get_ki");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "kd", PROPERTY_HINT_NONE, ""), "set_kd", "get_kd");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "tau", PROPERTY_HINT_NONE, ""), "set_tau", "get_tau");
}

float PIDControllerNode::update(float setpoint, float measurement) {
    if (!initialized) {
        T = 0.016f; // Default 60fps
        initialized = true;
    }
    out = core->update(setpoint, measurement, T);
    return out;
}

float PIDControllerNode::update_dt(float setpoint, float measurement, float dt) {
    out = core->update(setpoint, measurement, dt);
    return out;
}

void PIDControllerNode::reset() {
    core->reset();
    initialized = false;
}

void PIDControllerNode::set_kp(float v) { core->kp = v; }
float PIDControllerNode::get_kp() const { return core->kp; }
void PIDControllerNode::set_ki(float v) { core->ki = v; }
float PIDControllerNode::get_ki() const { return core->ki; }
void PIDControllerNode::set_kd(float v) { core->kd = v; }
float PIDControllerNode::get_kd() const { return core->kd; }
void PIDControllerNode::set_tau(float v) { core->tau = std::max(0.0f, v); }
float PIDControllerNode::get_tau() const { return core->tau; }
void PIDControllerNode::set_limits(float min, float max) {
    core->lim_min = min;
    core->lim_max = max;
}
float PIDControllerNode::get_limit_min() const { return core->lim_min; }
float PIDControllerNode::get_limit_max() const { return core->lim_max; }
void PIDControllerNode::set_sample_time(float v) { T = std::max(0.0f, v); }
float PIDControllerNode::get_sample_time() const { return T; }
