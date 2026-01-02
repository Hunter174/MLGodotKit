#include "pid_controller_node.h"
#include <godot_cpp/core/class_db.hpp>
#include <algorithm>

using namespace godot;

PIDControllerNode::PIDControllerNode() {}
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
}

float PIDControllerNode::update(float setpoint, float measurement) {

    // Error signal
    float error = setpoint - measurement;

    // Proportional
    float proportional = kp * error;

    if (!initialized) {
        prev_error = error;
        prev_measurement = measurement;
        differentiator = 0.0f;
        integrator = 0.0f;
        initialized = true;

        // P-only output on first step
        out = std::clamp(proportional, lim_min, lim_max);
        return out;
    }

    if (T <= 0.0f) {
        return out;
    }

    // Integral (trapezoidal)
    integrator += 0.5f * ki * T * (error + prev_error);

    // Dynamic integrator clamping (anti-windup)
    float lim_min_int, lim_max_int;

    if (lim_max > proportional) {
        lim_max_int = lim_max - proportional;
    } else {
        lim_max_int = 0.0f;
    }

    if (lim_min < proportional) {
        lim_min_int = lim_min - proportional;
    } else {
        lim_min_int = 0.0f;
    }

    integrator = std::clamp(integrator, lim_min_int, lim_max_int);

    // Derivative (band-limited, on measurement)
    differentiator =
        (2.0f * kd * (measurement - prev_measurement)
        + (2.0f * tau - T) * differentiator)
        / (2.0f * tau + T);

    // Output
    out = proportional + integrator - differentiator;
    out = std::clamp(out, lim_min, lim_max);

    // Store state
    prev_error = error;
    prev_measurement = measurement;

    return out;
}

float PIDControllerNode::update_dt(float setpoint, float measurement, float dt) {
    set_sample_time(dt);
    return update(setpoint, measurement);
}

void PIDControllerNode::reset() {
    integrator = 0.0f;
    prev_error = 0.0f;
    differentiator = 0.0f;
    prev_measurement = 0.0f;
    out = 0.0f;
    initialized = false;
}

// --- Getters / Setters ---

void PIDControllerNode::set_kp(float v) { kp = v; }
float PIDControllerNode::get_kp() const { return kp; }

void PIDControllerNode::set_ki(float v) { ki = v; }
float PIDControllerNode::get_ki() const { return ki; }

void PIDControllerNode::set_kd(float v) { kd = v; }
float PIDControllerNode::get_kd() const { return kd; }

void PIDControllerNode::set_tau(float v) { tau = std::max(0.0f, v); }
float PIDControllerNode::get_tau() const { return tau; }

void PIDControllerNode::set_limits(float min, float max) {
    lim_min = min;
    lim_max = max;
}
float PIDControllerNode::get_limit_min() const { return lim_min; }
float PIDControllerNode::get_limit_max() const { return lim_max; }

void PIDControllerNode::set_sample_time(float v) { T = std::max(0.0f, v); }
float PIDControllerNode::get_sample_time() const { return T; }