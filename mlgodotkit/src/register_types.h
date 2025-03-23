#ifndef GDEXAMPLE_REGISTER_TYPES_H
#define GDEXAMPLE_REGISTER_TYPES_H

// Utility Classes
#include "utility/utils.h"

// Node Classes
#include "neural_network/nnnode.h"
#include "linear_regression/lrnode.h"
#include "replay_buffer/replaybuffernode.h"
#include "decision_tree/dtreenode.h"

// Godot Classes
#include <gdextension_interface.h>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>

using namespace godot;

void initialize_example_module(ModuleInitializationLevel p_level);
void uninitialize_example_module(ModuleInitializationLevel p_level);

#endif // GDEXAMPLE_REGISTER_TYPES_H