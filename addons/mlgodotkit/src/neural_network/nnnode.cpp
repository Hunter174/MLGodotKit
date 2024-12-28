#include "neural_network/nnnode.h"

NNNode::NNNode() {}
NNNode::~NNNode() {}

void NNNode::_bind_methods() {
    godot::ClassDB::bind_method(godot::D_METHOD("initialize", "layer_sizes", "hidden_activation", "output_activation"), &NNNode::initialize);
    godot::ClassDB::bind_method(godot::D_METHOD("predict", "input"), &NNNode::predict);
    godot::ClassDB::bind_method(godot::D_METHOD("train", "input", "target"), &NNNode::train);
    godot::ClassDB::bind_method(godot::D_METHOD("set_optimizer", "optimizer_type", "learning_rate"), &NNNode::set_optimizer);
    godot::ClassDB::bind_method(godot::D_METHOD("set_loss_function", "loss_function"), &NNNode::set_loss_function);
    godot::ClassDB::bind_method(godot::D_METHOD("get_loss"), &NNNode::get_loss);
    godot::ClassDB::bind_method(godot::D_METHOD("copy_from", "other"), &NNNode::copy_from);
    godot::ClassDB::bind_method(godot::D_METHOD("set_verbosity", "level"), &NNNode::set_verbosity);
    godot::ClassDB::bind_method(godot::D_METHOD("model_summary"), &NNNode::model_summary);
}

void NNNode::initialize(godot::Array layer_sizes, godot::String hidden_activation, godot::String output_activation) {
    this->_hidden_activation = hidden_activation;
    this->_output_activation = output_activation;

    if (layer_sizes.size() < 2) {
        debug_print(1, "Error: Must have at least input and output layers.");
        return;
    }

    debug_print(1, "Initializing neural network");

    model_architecture = layer_sizes;
    weights.clear();
    biases.clear();
    activations.clear();
    activation_derivatives.clear();

    for (int i = 0; i < layer_sizes.size() - 1; ++i) {
        godot::String current_activation;

        if (i == layer_sizes.size() - 2) {
            // Last transition to output layer
            current_activation = output_activation;
        } else {
            // All other transitions use hidden layer activation
            current_activation = hidden_activation;
        }

        auto [func, derivative] = get_activation_function(current_activation);
        if (!func) {
            godot::UtilityFunctions::print("Error: Invalid activation function specified for layer ", i);
            return;
        }

        activations.push_back(func);
        activation_derivatives.push_back(derivative);

        weights.push_back(init_weights(layer_sizes[i + 1].operator int(), layer_sizes[i].operator int(), current_activation));
        biases.push_back(Eigen::VectorXd::Constant(layer_sizes[i + 1], 0.01));

        std::string layer_info = "Layer " + std::to_string(i) + " initialized with activation " + current_activation.utf8().get_data();
        debug_print(1, layer_info);
    }
}

Eigen::MatrixXd NNNode::init_weights(int rows, int cols, const godot::String &activation_type) {
    std::random_device rd;
    std::mt19937 gen(rd());

    double std_dev;
    if (activation_type == "relu") {
        std_dev = sqrt(2.0 / cols); // He initialization
    } else if (activation_type == "sigmoid" || activation_type == "tanh") {
        std_dev = 1.0 / sqrt(cols); // Standard Xavier initialization
    } else {
        std_dev = 1.0 / sqrt(cols); // Default to Xavier
    }

    std::normal_distribution<> dis(0, std_dev);
    Eigen::MatrixXd mat(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat(i, j) = dis(gen);
        }
    }

    for (size_t i = 0; i < weights.size(); ++i) {
        debug_print(1, "Neural Network Layers Initialized with: \n");
        debug_print(1, "Layer " + std::to_string(i) + " weights norm: " + std::to_string(weights[i].norm()));
        debug_print(1, "Layer " + std::to_string(i) + " biases norm: " + std::to_string(biases[i].norm()));
    }

    return mat;
}

godot::Array NNNode::predict(godot::Array input) {
	Eigen::VectorXd input_vec(input.size());
	for (int i = 0; i < input.size(); ++i) {
		input_vec[i] = float(input[i]);
	}

	// Use forward without storing intermediate states
	Eigen::VectorXd output = forward(input_vec, false);

	// Convert the result to Godot's Array format
	godot::Array result;
	for (int i = 0; i < output.size(); ++i) {
		result.push_back(output[i]);
	}
	return result;
}

Eigen::VectorXd NNNode::forward(const Eigen::VectorXd& input, bool store_intermediate) {
    Eigen::VectorXd activation = input;

    if (store_intermediate) {
        activations_values.clear();
        zs.clear();
    }

    for (size_t i = 0; i < weights.size(); ++i) {
        Eigen::VectorXd z = weights[i] * activation + biases[i];
        if (store_intermediate) zs.push_back(z);

        debug_print(2, "Layer " + std::to_string(i) + " z norm: " + std::to_string(z.norm()));

        // Apply activation functions
        activation = activations[i](z);

        debug_print(2, "Layer " + std::to_string(i) + " activation norm: " + std::to_string(activation.norm()));

        if (store_intermediate) activations_values.push_back(activation);
    }

    return activation;
}

void NNNode::backward(const Eigen::VectorXd& target) {

    // Compute initial delta (output layer error)
    Eigen::VectorXd delta = target - activations_values.back();

    // Gradient Clipping threshold
    const double max_gradient_norm = 1.0;

    if (verbosity_level >= 3){
        for (double val: delta) {
          godot::UtilityFunctions::print(("Delta: " + std::to_string(val)).c_str());
        }
        for (double val: activations_values.back()) {
           godot::UtilityFunctions::print(("Prediction: " + std::to_string(val)).c_str());
        }
        for (double val: target) {
           godot::UtilityFunctions::print(("Target: " + std::to_string(val)).c_str());
        }
    }

    // Iterate backward through layers
    for (int layer = weights.size() - 1; layer >= 0; --layer) {
        // Compute gradients for weights and biases
        Eigen::MatrixXd grad_w = delta * activations_values[layer].transpose();
        Eigen::VectorXd grad_b = delta;

        // Clip gradients by norm
        Eigen::VectorXd grad_w_flattened = Eigen::Map<Eigen::VectorXd>(grad_w.data(), grad_w.size());
        grad_w_flattened = clip_gradient(grad_w_flattened, max_gradient_norm);
        grad_b = clip_gradient(grad_b, max_gradient_norm);

        // Update weights and biases
        weights[layer] -= learning_rate * grad_w;
        biases[layer] -= learning_rate * grad_b;

        // Debugging: Print updated weights and biases (optional, can be removed if verbose)
		debug_print(2, "Updated weights for layer " + std::to_string(layer) + " grad_b norm: " + std::to_string(weights[layer].sum()));
		debug_print(2, "Updated biases for layer " + std::to_string(layer) + " grad_b norm: " + std::to_string(biases[layer].sum()));

        // If not the first layer, propagate the error backward
        if (layer > 0) {
            auto derivative = activation_derivatives[layer - 1];
            if (derivative) {
                // Compute the delta for the previous layer
                delta = (weights[layer].transpose() * delta).cwiseProduct(derivative(zs[layer - 1]));
            }
        }
    }
}

void NNNode::train(godot::Array input, godot::Array target) {
    Eigen::VectorXd input_vec(input.size());
    Eigen::VectorXd target_vec(target.size());

    // Convert inputs and targets to floats
    for (int i = 0; i < input.size(); ++i) input_vec[i] = float(input[i]);
    for (int i = 0; i < target.size(); ++i) target_vec[i] = float(target[i]);

    Eigen::VectorXd output = forward(input_vec);

    last_loss = loss_function(output, target_vec);
    backward(loss_derivative(output, target_vec));

    // Decay the learning rate
    decay_learning_rate();
}

void NNNode::set_optimizer(godot::String optimizer_type, float lr) {
    if (optimizer_type == "SGD") {
        learning_rate = lr;
    } else {
        godot::UtilityFunctions::print("Unsupported optimizer");
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
        godot::UtilityFunctions::print("Error: Null NNNode instance.");
        return;
    }

    if (model_architecture != other->model_architecture) {
        godot::UtilityFunctions::print("Error: Model architectures do not match.");
        return;
    }

    weights = other->weights;
    biases = other->biases;
    learning_rate = other->learning_rate;
    loss_function_type = other->loss_function_type;
}

// Activation Functions
Eigen::VectorXd NNNode::relu(const Eigen::VectorXd& x) {
    return x.cwiseMax(0);
}

Eigen::VectorXd NNNode::relu_derivative(const Eigen::VectorXd& x) {
    return x.unaryExpr([](double val) { return val > 0 ? 1.0 : 0.0; });
}

Eigen::VectorXd NNNode::sigmoid(const Eigen::VectorXd& x) {
    return (1.0 / (1.0 + (-x.array()).exp())).array().max(NNNode::epsilon).min(1.0 - NNNode::epsilon);
}

Eigen::VectorXd NNNode::sigmoid_derivative(const Eigen::VectorXd& x) {
    Eigen::VectorXd sig = sigmoid(x);
    return (sig.array() * (1.0 - sig.array())).array().max(NNNode::epsilon).min(1.0 - NNNode::epsilon);
}

Eigen::VectorXd NNNode::softmax(const Eigen::VectorXd& x) {
    Eigen::VectorXd exp_x = (x.array() - x.maxCoeff()).exp(); // Numerical stability with maxCoeff
    return (exp_x / exp_x.sum()).array().max(NNNode::epsilon).min(1.0 - NNNode::epsilon);
}

Eigen::VectorXd NNNode::tanh(const Eigen::VectorXd& x) {
    return x.array().tanh().max(-1.0 + NNNode::epsilon).min(1.0 - NNNode::epsilon);
}

Eigen::VectorXd NNNode::tanh_derivative(const Eigen::VectorXd& x) {
    return (1.0 - x.array().tanh().square()).array().max(NNNode::epsilon).min(1.0 - NNNode::epsilon);
}


// Loss functions
void NNNode::set_loss_function(godot::String loss_function_type) {
    if (loss_function_type == "MSE") {
        loss_function = [](const Eigen::VectorXd& output, const Eigen::VectorXd& target) {
            return mean_squared_error(output, target);
        };
        loss_derivative = [](const Eigen::VectorXd& output, const Eigen::VectorXd& target) {
            return mean_squared_error_derivative(output, target);
        };
    } else if (loss_function_type == "BCE") {
        loss_function = [](const Eigen::VectorXd& output, const Eigen::VectorXd& target) {
            return binary_cross_entropy(output, target);
        };
        loss_derivative = [](const Eigen::VectorXd& output, const Eigen::VectorXd& target) {
            return binary_cross_entropy_derivative(output, target);
        };
    } else {
        godot::UtilityFunctions::print("Error: Unsupported loss function: ", loss_function_type);
        loss_function = nullptr;
        loss_derivative = nullptr;
    }
}

double NNNode::mean_squared_error(const Eigen::VectorXd& output, const Eigen::VectorXd& target) {
    return (output - target).squaredNorm() / target.size();
}

Eigen::VectorXd NNNode::mean_squared_error_derivative(const Eigen::VectorXd& output, const Eigen::VectorXd& target) {
    return 2*(output - target);
}

double NNNode::binary_cross_entropy(const Eigen::VectorXd& output, const Eigen::VectorXd& target) {
    const double epsilon = 1e-10;
    Eigen::VectorXd clipped_output = output.array().max(epsilon).min(1.0 - epsilon);
    return -(target.array() * clipped_output.array().log() +
             (1.0 - target.array()) * (1.0 - clipped_output.array()).log()).mean();
}

Eigen::VectorXd NNNode::binary_cross_entropy_derivative(const Eigen::VectorXd& output, const Eigen::VectorXd& target) {
    const double epsilon = 1e-10;  // Prevent division by 0
    Eigen::VectorXd clipped_output = output.array().max(epsilon).min(1.0 - epsilon);
    return (clipped_output - target).array() / (clipped_output.array() * (1.0 - clipped_output.array()));
}

// Utility
std::pair<std::function<Eigen::VectorXd(const Eigen::VectorXd&)>, std::function<Eigen::VectorXd(const Eigen::VectorXd&)>>
NNNode::get_activation_function(const godot::String& activation) {
    if (activation == "relu") {
        return {relu, relu_derivative};
    } else if (activation == "sigmoid") {
        return {sigmoid, sigmoid_derivative};
    } else if (activation == "tanh") {
        return {tanh, tanh_derivative};
    } else if (activation == "softmax") {
        // Note: softmax usually does not need a derivative in the output layer
        return {softmax, nullptr};
    } else {
        godot::UtilityFunctions::print("Unsupported activation function: " + activation);
        return {nullptr, nullptr};
    }
}

Eigen::VectorXd NNNode::clip_gradient(const Eigen::VectorXd& gradient, double max_norm) {
    double norm = gradient.norm();
    if (norm > max_norm) {
        return (gradient / norm) * max_norm; // Scale the gradient to have the norm of max_norm
    }
    return gradient;
}

void NNNode::set_verbosity(int level) {
    verbosity_level = level;
}

void NNNode::debug_print(int level, const std::string& message) {
    if (verbosity_level >= level) {
        godot::UtilityFunctions::print(message.c_str());
    }
}

void NNNode::model_summary(){
    std::stringstream summary;
    summary << "Neural Network Architecture\n";
    summary << "----------------------------\n";
    summary << "Total layers: " << model_architecture.size() << " (including input layer)\n\n";

    // Loop through each layer to print details
    for (size_t i = 0; i < model_architecture.size(); ++i) {
        summary << "Layer " << i << (i == 0 ? " (Input Layer)" :
                         (i == model_architecture.size() - 1 ? " (Output Layer)" : " (Hidden Layer)")) << ":\n";
        summary << "  - Nodes: " << int(model_architecture[i]) << "\n";

        if (i > 0) { // Starting from first hidden layer to output layer
            // Determine the activation function for the current layer
            std::string activation = (i == model_architecture.size() - 1) ? _output_activation.utf8().get_data() :
                                                                          _hidden_activation.utf8().get_data();
            summary << "  - Activation: " << activation << "\n";
            summary << "  - Weights Matrix: " << weights[i-1].rows() << " x " << weights[i-1].cols() << "\n";
            summary << "  - Biases: " << biases[i-1].size() << " nodes\n";
            summary << "    - Weights Values:\n";
            for (int row = 0; row < weights[i-1].rows(); ++row) {
                summary << "      ";
                for (int col = 0; col < weights[i-1].cols(); ++col) {
                    summary << weights[i-1](row, col) << (col == weights[i-1].cols() - 1 ? "" : ", ");
                }
                summary << "\n";
            }
            summary << "    - Bias Values:\n      ";
            for (int j = 0; j < biases[i-1].size(); ++j) {
                summary << biases[i-1][j] << (j == biases[i-1].size() - 1 ? "\n" : ", ");
            }
        }
        summary << "\n";
    }

    // Output the complete neural network architecture summary
    godot::UtilityFunctions::print(summary.str().c_str());
}
