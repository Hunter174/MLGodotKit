#include "layer/layer.h"
#include <cmath>

using namespace Activations;

Layer::Layer(int input_size, int out_features, float learning_rate, const std::string& activation)
    : lr(learning_rate), activation_type(activation) {

    std::tie(weights, biases) = init_weights(input_size, out_features, activation);

    // Initialize momentum buffers
    mW = Eigen::MatrixXf::Zero(input_size, out_features);
    mb = Eigen::MatrixXf::Zero(1, out_features);

    // Select activation
    if (activation == "sigmoid") {
        activation_func = sigmoid;
        derivative_func = sigmoid_derivative;
    } else if (activation == "relu") {
        activation_func = relu;
        derivative_func = relu_derivative;
    } else if (activation == "leaky_relu") {
        activation_func = [](const Eigen::MatrixXf& x){ return leaky_relu(x, 0.01f); };
        derivative_func = [](const Eigen::MatrixXf& z){ return leaky_relu_derivative(z, 0.01f); };
    } else {
        activation_func = linear;
        derivative_func = linear_derivative;
    }

    Logger::debug(2, "Layer initialized (" + activation +
        ") weights=" + std::to_string(input_size) + "x" + std::to_string(out_features));
}

Layer::~Layer() {}

std::tuple<Eigen::MatrixXf, Eigen::MatrixXf> Layer::init_weights(int in, int out, const std::string& activation) {
    Eigen::MatrixXf W;
    Eigen::MatrixXf b = Eigen::MatrixXf::Zero(1, out);

    if (activation == "relu" || activation == "leaky_relu") {
        float stddev = std::sqrt(2.0f / in);  // He init
        W = Eigen::MatrixXf::Random(in, out) * stddev;
    } else {
        float stddev = std::sqrt(1.0f / in);  // Xavier
        W = Eigen::MatrixXf::Random(in, out) * stddev;
    }
    return std::make_tuple(W, b);
}

Eigen::MatrixXf Layer::forward(const Eigen::MatrixXf& X) {
    input = X;

    // z = XW + b (bias broadcast)
	Eigen::MatrixXf z = (X * weights).rowwise() + biases.row(0);


    // Apply activation
    Eigen::MatrixXf a = activation_func(z);

    // Optional squash for stable Q-heads
    if (squash_enabled) {
        Eigen::MatrixXf z_scaled = z / squash_scale_in;
        grad_z = z_scaled;
        output = z_scaled.array().tanh() * squash_scale_out;
    } else {
        output = a;
        grad_z = z;
    }

    return output;
}

Eigen::MatrixXf Layer::backward_compute(const Eigen::MatrixXf& loss_grad) {
    if (loss_grad.size() == 0 || !loss_grad.allFinite()) {
        Logger::warn("Layer::backward_compute - invalid gradient input");
        return Eigen::MatrixXf::Zero(input.rows(), weights.rows());
    }

    // Activation derivative
    Eigen::MatrixXf act_prime;
    if (squash_enabled) {
        Eigen::ArrayXXf a = grad_z.array().tanh();
        act_prime = (1.0f - a.square()).matrix() * (1.0f / squash_scale_in);
    } else {
        act_prime = derivative_func(grad_z);
    }

    Eigen::MatrixXf delta = loss_grad.cwiseProduct(act_prime);

    // Compute gradients
    dW = input.transpose() * delta;
    db = delta.colwise().sum();

    // Ensure finite
    dW = dW.unaryExpr([](float v){ return std::isfinite(v) ? v : 0.0f; });
    db = db.unaryExpr([](float v){ return std::isfinite(v) ? v : 0.0f; });

    dW /= static_cast<float>(input.rows());
	db /= static_cast<float>(input.rows());

    // Return for chain rule
    return delta * weights.transpose();
}

void Layer::normalize_gradients(float scale) {
    dW *= scale;
    db *= scale;
}

void Layer::apply_update() {
    const float beta = 0.9f;           // momentum
    const float weight_decay = 1e-4f;  // L2 regularization

    // Momentum update (per layer)
    mW = beta * mW + (1.0f - beta) * dW;
    mb = beta * mb + (1.0f - beta) * db;

    weights -= lr * (mW + weight_decay * weights);
    biases  -= lr * (mb + 1e-6f * biases);
}

void Layer::copy_weights(const Layer& src) {
    weights = src.weights;
    biases  = src.biases;
}

void Layer::set_learning_rate(float learning_rate) { lr = learning_rate; }
void Layer::set_verbosity(int v) { verbosity = v; }
void Layer::set_output_squash(bool enabled, float scale_in, float scale_out) {
    squash_enabled   = enabled;
    squash_scale_in  = (scale_in  <= 0.0f ? 10.0f : scale_in);
    squash_scale_out = (scale_out <= 0.0f ? 10.0f : scale_out);
}

int Layer::get_input_size() const { return weights.rows(); }
int Layer::get_output_size() const { return weights.cols(); }