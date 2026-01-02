#ifndef PID_CONTROLLER_NODE_H
#define PID_CONTROLLER_NODE_H

#include <godot_cpp/classes/node.hpp>

class PIDControllerNode : public godot::Node {
    GDCLASS(PIDControllerNode, godot::Node);

private:
    // Must not derive and integrate in first update
    bool initialized = false;

    // Gains
    float kp = 0.0f;
    float ki = 0.0f;
    float kd = 0.0f;

    // Derivative filter time constant
    float tau = 0.02f;

    // Output limits
    float lim_min = -1.0f;
    float lim_max =  1.0f;

    // Sample time
    float T = 0.0f;

    // State
    float integrator = 0.0f;
    float prev_error = 0.0f;
    float differentiator = 0.0f;
    float prev_measurement = 0.0f;

    float out = 0.0f;

protected:
    static void _bind_methods();

public:
    PIDControllerNode();
    ~PIDControllerNode();

    float update(float setpoint, float measurement);
    float update_dt(float setpoint, float measurement, float dt);
    void reset();

    // Getters / Setters
    void set_kp(float v);
    float get_kp() const;

    void set_ki(float v);
    float get_ki() const;

    void set_kd(float v);
    float get_kd() const;

    void set_tau(float v);
    float get_tau() const;

    void set_limits(float min, float max);
    float get_limit_min() const;
    float get_limit_max() const;

    void set_sample_time(float v);
    float get_sample_time() const;
};

#endif // PID_CONTROLLER_NODE_H
