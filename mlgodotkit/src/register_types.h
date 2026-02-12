#ifndef ML_GODOT_KIT_REGISTER_TYPES_H
#define ML_GODOT_KIT_REGISTER_TYPES_H

//Primative Classes
#include "matrix/matrix.h"

// Utility Classes
#include "utility/utils.h"
#include "linalg/linalg.h"

// Model Classes
#include "models/linear_regression/linear_regression_node.h"
#include "models/neural_network/neural_network_node.h"
#include "models/linear_model/linear_model_node.h"
#include "models/decision_tree/decision_tree_node.h"

// Loss Fucntions
#include "losses/loss_node/loss_node.h"
#include "losses/mse_loss_node/mse_loss_node.h"

// Control Theory
#include "control/pid_controller/pid_controller_node.h"


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