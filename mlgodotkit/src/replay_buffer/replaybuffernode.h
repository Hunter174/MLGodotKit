// ReplayBufferNode.h
#ifndef REPLAYBUFFERNODE_H
#define REPLAYBUFFERNODE_H

#include <vector>
#include <random>
#include <cstddef>
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>

using namespace godot;

class ReplayBufferNode : public Node {
    GDCLASS(ReplayBufferNode, Node);

public:
    struct Experience {
        PackedFloat32Array state;
        int action;
        float reward;
        PackedFloat32Array next_state;
        bool done;
    };

    ReplayBufferNode();

    void add(const PackedFloat32Array &state, int action, float reward,
         const PackedFloat32Array &next_state, bool done);

    Array sample(size_t batch_size);
    int get_size() const;

    void set_max_size(int new_max_size);
    int get_max_size() const;

protected:
    static void _bind_methods();

private:
    size_t max_size;
    size_t current_size;
    std::vector<Experience> buffer;
    std::default_random_engine random_engine;
};

#endif // REPLAYBUFFERNODE_H