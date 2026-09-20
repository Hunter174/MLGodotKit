#include "pid_controller_core.h"
#include <algorithm>

float PIDControllerCore::update(float setpoint, float measurement, float dt) {
    if (dt <= 0.0f) return 0.0f;

    float error = setpoint - measurement;
    
    // Proportional
    float p_term = kp * error;

    // Integral
    integrator += error * dt;
    float i_term = ki * integrator;

    // Derivative (with filtered derivative to avoid spikes)
    float diff = (error - prev_error);
    differentiator = (tau * differentiator + diff * dt) / (tau + dt);
    float d_term = kd * differentiator;

    prev_error = error;

    float output = p_term + i_term + d_term;

    // Clamp output
    return std::clamp(output, lim_min, lim_max);
}

void PIDControllerCore::reset() {
    integrator = 0.0f;
    prev_error = 0.0f;
    differentiator = 0.0f;
    prev_measurement = 0.0f;
}
