#include "replay_buffer/replaybuffernode.h"
#include <unordered_set>

ReplayBufferNode::ReplayBufferNode() : max_size(1000), current_size(0), random_engine(std::random_device{}()) {}

void ReplayBufferNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("add", "state", "action", "reward", "next_state", "done"), &ReplayBufferNode::add);
    ClassDB::bind_method(D_METHOD("sample", "batch_size"), &ReplayBufferNode::sample);
    ClassDB::bind_method(D_METHOD("get_size"), &ReplayBufferNode::get_size);
    ClassDB::bind_method(D_METHOD("set_max_size", "new_max_size"), &ReplayBufferNode::set_max_size);
    ClassDB::bind_method(D_METHOD("get_max_size"), &ReplayBufferNode::get_max_size);

    ADD_PROPERTY(PropertyInfo(Variant::INT, "max_size"), "set_max_size", "get_max_size");
}

void ReplayBufferNode::add(const PackedFloat32Array &state, int action, float reward,
                           const PackedFloat32Array &next_state, bool done) {
    if (buffer.size() < max_size) {
        buffer.push_back({state, action, reward, next_state, done});
    } else {
        buffer[current_size % max_size] = {state, action, reward, next_state, done};
    }
    current_size++;
}

Array ReplayBufferNode::sample(size_t batch_size) {
    Array samples;

    if (current_size == 0) {
        return samples; // Return an empty array if the buffer is empty
    }

    size_t sample_size = std::min(batch_size, std::min(current_size, max_size));
    std::unordered_set<size_t> sampled_indices;
    std::uniform_int_distribution<size_t> distribution(0, std::min(current_size, max_size) - 1);

    while (sampled_indices.size() < sample_size) {
        size_t index = distribution(random_engine);
        if (sampled_indices.insert(index).second) {
            const auto &exp = buffer[index];

            Dictionary experience;
            experience["state"] = exp.state;
            experience["action"] = exp.action;
            experience["reward"] = exp.reward;
            experience["next_state"] = exp.next_state;
            experience["done"] = exp.done;

            samples.push_back(experience);
        }
    }
    return samples;
}

int ReplayBufferNode::get_size() const {
    return static_cast<int>(std::min(current_size, max_size));
}

void ReplayBufferNode::set_max_size(int new_max_size) {
    max_size = static_cast<size_t>(new_max_size);
}

int ReplayBufferNode::get_max_size() const {
    return static_cast<int>(max_size);
}