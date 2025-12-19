#ifndef ML_GODOT_KIT_REGISTER_TYPES_H
#define ML_GODOT_KIT_REGISTER_TYPES_H

//Primative Classes
#include "matrix/matrix.h"

// Utility Classes
#include "utility/utils.h"
#include "benchmark/benchmark.h"
#include "linalg/linalg.h"

// Node Classes
#include "neural_network/nnnode.h"
#include "linear_regression/lrnode.h"
#include "decision_tree/dtreenode.h"

// Godot Classes
#include <gdextension_interface.h>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>

using namespace godot;
void register_mlgogodot_types();
void initialize_example_module(ModuleInitializationLevel p_level);
void uninitialize_example_module(ModuleInitializationLevel p_level);

#endif // GDEXAMPLE_REGISTER_TYPES_H