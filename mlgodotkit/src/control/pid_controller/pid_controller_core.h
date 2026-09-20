#ifndef PID_CONTROLLER_CORE_H
#define PID_CONTROLLER_CORE_H

class PIDControllerCore {
public:
    PIDControllerCore() = default;

    float update(float setpoint, float measurement, float dt);
    void reset();

    // Gains
    float kp = 0.0f;
    float ki = 0.0f;
    float kd = 0.0f;
    float tau = 0.02f;
    float lim_min = -1.0f;
    float lim_max = 1.0f;

private:
    float integrator = 0.0f;
    float prev_error = 0.0f;
    float differentiator = 0.0f;
    float prev_measurement = 0.0f;
};

#endif // PID_CONTROLLER_CORE_H
