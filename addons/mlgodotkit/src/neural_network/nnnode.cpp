#include "neural_network/nnnode.h"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/method_bind.hpp>
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/variant/array.hpp>
#include <cmath>
#include <random>
#include <algorithm>

using namespace godot;

NNNode::NNNode() {}
NNNode::~NNNode() {}

void NNNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("initialize", "layer_sizes", "hidden_activation", "output_activation"), &NNNode::initialize);
    ClassDB::bind_method(D_METHOD("predict", "input"), &NNNode::predict);
    ClassDB::bind_method(D_METHOD("train", "input", "target"), &NNNode::train);
    ClassDB::bind_method(D_METHOD("set_optimizer", "optimizer_type", "learning_rate"), &NNNode::set_optimizer);
    ClassDB::bind_method(D_METHOD("set_loss_function", "loss_function"), &NNNode::set_loss_function);
    ClassDB::bind_method(D_METHOD("get_loss"), &NNNode::get_loss);
    ClassDB::bind_method(D_METHOD("copy_from", "other"), &NNNode::copy_from);
}

void NNNode::initialize(Array layer_sizes, String hidden_activation, String output_activation) {
	if (layer_sizes.size() < 2) {
		UtilityFunctions::print("Error: Must have at least input and output layers.");
		return;
	}

	model_architecture = layer_sizes;
	weights.clear();
	biases.clear();
	activations.clear();
	activation_derivatives.clear();

	// Define activation functions for each layer
	auto get_activation_function = [](const String& activation) -> std::pair<
	    std::function<VectorXd(const VectorXd&)>,
	    std::function<VectorXd(const VectorXd&)>> {
		if (activation == "relu") {
			return {relu, relu_derivative};
		} else if (activation == "sigmoid") {
			return {sigmoid, sigmoid_derivative};
		} else if (activation == "softmax") {
			return {softmax, nullptr}; // Softmax doesn't need a derivative here
		} else {
			UtilityFunctions::print("Unsupported activation function: ", activation);
			return {nullptr, nullptr};
		}
	};

	auto [hidden_func, hidden_derivative] = get_activation_function(hidden_activation);
	auto [output_func, output_derivative] = get_activation_function(output_activation);

	if (!hidden_func || !output_func) {
		UtilityFunctions::print("Error: Invalid activation functions.");
		return;
	}

	// Set activation functions for hidden layers and output layer
	for (size_t i = 0; i < layer_sizes.size() - 2; ++i) {
		activations.push_back(hidden_func);
		activation_derivatives.push_back(hidden_derivative);
	}
	activations.push_back(output_func);
	activation_derivatives.push_back(output_derivative);

	// Initialize weights and biases
	for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
		String activation_type = (i == layer_sizes.size() - 2) ? output_activation : hidden_activation;
		weights.push_back(init_weights(layer_sizes[i + 1].operator int(), layer_sizes[i].operator int(), activation_type));
		biases.push_back(VectorXd::Zero(layer_sizes[i + 1]));
	}
}

MatrixXd NNNode::init_weights(int rows, int cols, const String &activation_type) {
    std::random_device rd;
    std::mt19937 gen(rd());

    double std_dev;
    if (activation_type == "relu") {
        std_dev = sqrt(2.0 / cols); // He initialization
    } else if (activation_type == "sigmoid") {
        std_dev = 1.0 / sqrt(cols) * 1.5; // Scaled Xavier initialization
    } else {
        std_dev = 1.0 / sqrt(cols);
    }

    std::normal_distribution<> dis(0, std_dev);
    MatrixXd mat(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            mat(i, j) = dis(gen);

    return mat;
}

Array NNNode::predict(Array input) {
	VectorXd input_vec(input.size());
	for (int i = 0; i < input.size(); ++i) {
		input_vec[i] = float(input[i]);
	}

	// Use forward without storing intermediate states
	VectorXd output = forward(input_vec, false);

	// Convert the result to Godot's Array format
	Array result;
	for (int i = 0; i < output.size(); ++i) {
		result.push_back(output[i]);
	}
	return result;
}

VectorXd NNNode::forward(const VectorXd& input, bool store_intermediate) {
    VectorXd activation = input;

    if (store_intermediate) {
        activations_values.clear();
        zs.clear();
    }

    for (size_t i = 0; i < weights.size(); ++i) {
        VectorXd z = weights[i] * activation + biases[i];
        if (store_intermediate) zs.push_back(z);

        if (verbose == true)
        	UtilityFunctions::print("Layer ", i, " z norm: ", z.norm());

        // Use hidden layer activations for all but the last layer
        if (i == weights.size() - 1) {
            activation = activations.back()(z);  // Output layer activation
        } else {
            activation = activations[i](z);  // Hidden layer activations
        }

        if (verbose == true)
        	UtilityFunctions::print("Layer ", i, " activation norm: ", activation.norm());

        if (store_intermediate) activations_values.push_back(activation);
    }

    return activation;
}

void NNNode::backward(const VectorXd& target) {
    const double max_gradient = 5.0; // Gradient normalization threshold

    // Compute initial delta (output layer error)
    VectorXd delta = activations_values.back() - target;

    // Normalize delta to prevent exploding gradients
    delta /= std::max(delta.norm(), max_gradient);

    // Iterate backward through layers
    for (int layer = weights.size() - 1; layer >= 0; --layer) {
        // Compute gradients for weights and biases
        MatrixXd grad_w = delta * activations_values[layer].transpose();
        VectorXd grad_b = delta;

        // Normalize gradients to prevent exploding gradients
        double grad_w_norm = grad_w.norm();
        double grad_b_norm = grad_b.norm();
        if (grad_w_norm > max_gradient) {
            grad_w /= grad_w_norm / max_gradient;
        }
        if (grad_b_norm > max_gradient) {
            grad_b /= grad_b_norm / max_gradient;
        }

        // Debugging: Print norms of gradients
        if (verbose == true){
        	UtilityFunctions::print("Layer ", layer, " grad_w norm: ", grad_w_norm);
        	UtilityFunctions::print("Layer ", layer, " grad_b norm: ", grad_b_norm);}

        // Update weights and biases
        weights[layer] -= learning_rate * grad_w;
        biases[layer] -= learning_rate * grad_b;

        // Debugging: Print updated weights and biases (optional, can be removed if verbose)
        if (verbose == true){
        	UtilityFunctions::print("Updated weights for layer ", layer, ": ", weights[layer].sum());
        	UtilityFunctions::print("Updated biases for layer ", layer, ": ", biases[layer].sum());}

        // If not the first layer, propagate the error backward
        if (layer > 0) {
            auto derivative = activation_derivatives[layer - 1];
            if (derivative) {
                // Compute the delta for the previous layer
                delta = (weights[layer].transpose() * delta).cwiseProduct(derivative(zs[layer - 1]));

                // Normalize delta to prevent exploding gradients
                double delta_norm = delta.norm();
                if (delta_norm > max_gradient) {
                    delta /= delta_norm / max_gradient;
                }

                // Debugging: Print delta norm for the current layer
                if (verbose == true){
                	UtilityFunctions::print("Layer ", layer, " delta norm: ", delta_norm);}
            }
        }
    }
}

void NNNode::train(Array input, Array target) {
    VectorXd input_vec(input.size());
    VectorXd target_vec(target.size());

    for (int i = 0; i < input.size(); ++i) input_vec[i] = float(input[i]);
    for (int i = 0; i < target.size(); ++i) target_vec[i] = float(target[i]);

    VectorXd output = forward(input_vec);

    if (!loss_function || !loss_derivative) {
        UtilityFunctions::print("Error: Loss function not set.");
        return;
    }

    last_loss = loss_function(output, target_vec);
    backward(loss_derivative(output, target_vec));
}

void NNNode::set_optimizer(String optimizer_type, float lr) {
    if (optimizer_type == "SGD") {
        learning_rate = lr;
    } else {
        UtilityFunctions::print("Unsupported optimizer");
    }
}

double NNNode::get_loss() const {
    return last_loss;
}

void NNNode::decay_learning_rate() {
    learning_rate = std::max(0.0001, learning_rate * 0.99);
}

void NNNode::copy_from(const NNNode* other) {
    if (!other) {
        UtilityFunctions::print("Error: Null NNNode instance.");
        return;
    }

    if (model_architecture != other->model_architecture) {
        UtilityFunctions::print("Error: Model architectures do not match.");
        return;
    }

    weights = other->weights;
    biases = other->biases;
    learning_rate = other->learning_rate;
    loss_function_type = other->loss_function_type;
}

// Activation Functions
VectorXd NNNode::relu(const VectorXd& x) {
    return x.cwiseMax(0);
}

VectorXd NNNode::relu_derivative(const VectorXd& x) {
    return x.unaryExpr([](double val) { return val > 0 ? 1.0 : 0.0; });
}

VectorXd NNNode::sigmoid(const VectorXd& x) {
    const double epsilon = 1e-10;  // Prevent log(0)
    return (1.0 / (1.0 + (-x.array()).exp())).array().max(epsilon).min(1.0 - epsilon);
}


VectorXd NNNode::sigmoid_derivative(const VectorXd& x) {
    VectorXd sig = sigmoid(x);
    return sig.array() * (1.0 - sig.array());
}

VectorXd NNNode::softmax(const VectorXd& x) {
    VectorXd exp_x = (x.array() - x.maxCoeff()).exp();
    return exp_x / exp_x.sum();
}

// Loss functions
void NNNode::set_loss_function(String loss_function_type) {
    if (loss_function_type == "MSE") {
        loss_function = [](const VectorXd& output, const VectorXd& target) {
            return mean_squared_error(output, target);
        };
        loss_derivative = [](const VectorXd& output, const VectorXd& target) {
            return mean_squared_error_derivative(output, target);
        };
    } else if (loss_function_type == "BCE") {
        loss_function = [](const VectorXd& output, const VectorXd& target) {
            return binary_cross_entropy(output, target);
        };
        loss_derivative = [](const VectorXd& output, const VectorXd& target) {
            return binary_cross_entropy_derivative(output, target);
        };
    } else {
        UtilityFunctions::print("Error: Unsupported loss function: ", loss_function_type);
        loss_function = nullptr;
        loss_derivative = nullptr;
    }
}

double NNNode::mean_squared_error(const VectorXd& output, const VectorXd& target) {
    return (output - target).squaredNorm() / target.size();
}

VectorXd NNNode::mean_squared_error_derivative(const VectorXd& output, const VectorXd& target) {
    return output - target;
}

double NNNode::binary_cross_entropy(const VectorXd& output, const VectorXd& target) {
    const double epsilon = 1e-10;
    VectorXd clipped_output = output.array().max(epsilon).min(1.0 - epsilon);
    return -(target.array() * clipped_output.array().log() +
             (1.0 - target.array()) * (1.0 - clipped_output.array()).log()).mean();
}

VectorXd NNNode::binary_cross_entropy_derivative(const VectorXd& output, const VectorXd& target) {
    const double epsilon = 1e-10;  // Prevent division by 0
    VectorXd clipped_output = output.array().max(epsilon).min(1.0 - epsilon);
    return (clipped_output - target).array() / (clipped_output.array() * (1.0 - clipped_output.array()));
}