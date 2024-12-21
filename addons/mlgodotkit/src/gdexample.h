#ifndef GDEXAMPLE_H
#define GDEXAMPLE_H

#include <godot_cpp/classes/node2d.hpp>

namespace godot {

    class GDExample : public Node2D {
        GDCLASS(GDExample, Node2D)

    private:
        double time_passed;
        double amplitude;
        double speed;

    protected:
        static void _bind_methods();  // Used to bind methods and properties for Godot

    public:
        GDExample();
        ~GDExample();

        void _process(double delta) override;

        // Getter and setter for amplitude
        void set_amplitude(const double p_amplitude);
        double get_amplitude() const;

        // Getter and setter for speed
        void set_speed(const double p_speed);
        double get_speed() const;
    };

}

#endif // GDEXAMPLE_H
