#include "gdexample.h"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/vector2.hpp>
#include <cmath> // For sin and cos functions

using namespace godot;

void GDExample::_bind_methods() {
    // Binding the getter and setter methods for amplitude
    ClassDB::bind_method(D_METHOD("get_amplitude"), &GDExample::get_amplitude);
    ClassDB::bind_method(D_METHOD("set_amplitude", "p_amplitude"), &GDExample::set_amplitude);

    // Binding the getter and setter methods for speed
    ClassDB::bind_method(D_METHOD("get_speed"), &GDExample::get_speed);
    ClassDB::bind_method(D_METHOD("set_speed", "p_speed"), &GDExample::set_speed);

    // Registering amplitude and speed as properties in Godot's inspector
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "amplitude"), "set_amplitude", "get_amplitude");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "speed", PROPERTY_HINT_RANGE, "0,20,0.01"), "set_speed", "get_speed");
}

GDExample::GDExample() {
    // Initialize variables
    time_passed = 0.0;
    amplitude = 10.0;  // Default value for amplitude
    speed = 1.0;       // Default value for speed
}

GDExample::~GDExample() {
    // Add any cleanup code here if necessary
}

void GDExample::_process(double delta) {
    // Increment time_passed by delta multiplied by speed
    time_passed += speed * delta;

    // Calculate new position in a figure-8 pattern using sine and cosine
    Vector2 new_position = Vector2(
        amplitude + (amplitude * sin(time_passed * 2.0)), // X-coordinate
        amplitude + (amplitude * cos(time_passed * 1.5))  // Y-coordinate
    );

    // Set the position of the node to the newly calculated position
    set_position(new_position);
}

// Setter for amplitude
void GDExample::set_amplitude(const double p_amplitude) {
    amplitude = p_amplitude;
}

// Getter for amplitude
double GDExample::get_amplitude() const {
    return amplitude;
}

// Setter for speed
void GDExample::set_speed(const double p_speed) {
    speed = p_speed;
}

// Getter for speed
double GDExample::get_speed() const {
    return speed;
}
