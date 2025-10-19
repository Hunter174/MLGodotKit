#ifndef BENCHMARK_H
#define BENCHMARK_H
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/classes/time.hpp> // <-- ADD THIS!

using namespace godot;

class Benchmark : public Node {
    GDCLASS(Benchmark, Node)

public:
    static void _bind_methods() {
        // No bindings for now, but required for registration
    }

    void _ready() {
        UtilityFunctions::print("✅ Benchmark started");

        Time *time = Time::get_singleton();
        uint64_t start = time->get_ticks_usec();

        volatile int64_t total = 0; // volatile prevents optimization
        for (int64_t i = 0; i < 1'000'000; ++i) {
            total += i % 1000;
        }

        uint64_t end = time->get_ticks_usec();
        UtilityFunctions::print("✅ Benchmark finished, result = ", total);
        UtilityFunctions::print("⏱ Time taken: ", (end - start) / 1000.0, " ms");
    }

};

#endif //BENCHMARK_H
