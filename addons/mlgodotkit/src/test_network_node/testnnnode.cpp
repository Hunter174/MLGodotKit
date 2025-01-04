#include "testnnnode.h"

TestNNNode::TestNNNode() {
    // XOR Input and Labels
    input = Eigen::MatrixXd(4, 2);
    input << 0, 0,
             0, 1,
             1, 0,
             1, 1;

    labels = Eigen::MatrixXd(4, 1);
    labels << 0,
              1,
              1,
              0;

    // Initialize Weights and Biases Randomly
    W1 = Eigen::MatrixXd(2, 4);  // 4 hidden neurons
    W1.setRandom();
    b1 = Eigen::MatrixXd(1, 4);
    b1.setZero();

    W2 = Eigen::MatrixXd(4, 1);  // 1 output neuron
    W2.setRandom();
    b2 = Eigen::MatrixXd(1, 1);
    b2.setZero();

    learning_rate = 0.01;
    epochs = 1000;  // Increase epochs for better training
}

TestNNNode::~TestNNNode() {}

void TestNNNode::_bind_methods() {
    godot::ClassDB::bind_method(godot::D_METHOD("test"), &TestNNNode::test);
}

// Testing method for Godot
void TestNNNode::test() {
    godot::UtilityFunctions::print("Initializing Neural Network...");

    // Perform initial forward pass
    Eigen::MatrixXd predictions = forward(input);

    godot::UtilityFunctions::print("Initial Predictions:");
    for (int i = 0; i < predictions.rows(); ++i) {
        std::stringstream ss;
        ss << "Input: " << input.row(i)
           << " Prediction: " << predictions(i, 0);
        godot::UtilityFunctions::print(ss.str().c_str());
    }

    // Train the network
    train();

    // Show predictions after training
    predictions = forward(input);
    godot::UtilityFunctions::print("Predictions after training:");

    for (int i = 0; i < predictions.rows(); ++i) {
        std::stringstream ss;
        ss << "Input: " << input.row(i)
           << " Prediction: " << predictions(i, 0);
        godot::UtilityFunctions::print(ss.str().c_str());
    }
}

// Sigmoid Activation
Eigen::MatrixXd TestNNNode::sigmoid(const Eigen::MatrixXd& x) {
    return 1.0 / (1.0 + (-x.array()).exp());
}

// Sigmoid Derivative
Eigen::MatrixXd TestNNNode::sigmoid_derivative(const Eigen::MatrixXd& x) {
    Eigen::MatrixXd s = sigmoid(x);
    return s.array() * (1.0 - s.array());
}

// ReLU Activation
Eigen::MatrixXd TestNNNode::relu(const Eigen::MatrixXd& x) {
    return x.cwiseMax(0);
}

// ReLU Derivative
Eigen::MatrixXd TestNNNode::relu_derivative(const Eigen::MatrixXd& x) {
    return (x.array() > 0).cast<double>();
}

// Forward Pass (ReLU in hidden, sigmoid in output)
Eigen::MatrixXd TestNNNode::forward(const Eigen::MatrixXd& X) {
    Eigen::MatrixXd z1 = X * W1 + b1.replicate(X.rows(), 1);
    Eigen::MatrixXd a1 = relu(z1);  // ReLU for hidden layer
    Eigen::MatrixXd z2 = a1 * W2 + b2.replicate(a1.rows(), 1);
    Eigen::MatrixXd a2 = sigmoid(z2);  // Sigmoid for output

    return a2;
}

// Mean Squared Error (MSE) Loss
double TestNNNode::compute_loss(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& y) {
    Eigen::MatrixXd diff = predictions - y;
    return (diff.array().square().sum()) / y.rows();
}

// MSE Loss Derivative
Eigen::MatrixXd TestNNNode::loss_derivative(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& y) {
    return 2 * (predictions - y) / y.rows();
}

// Backward Propagation
void TestNNNode::backward(const Eigen::MatrixXd& loss_gradient) {
    Eigen::MatrixXd z1 = input * W1 + b1.replicate(input.rows(), 1);
    Eigen::MatrixXd a1 = relu(z1);
    Eigen::MatrixXd z2 = a1 * W2 + b2.replicate(a1.rows(), 1);
    Eigen::MatrixXd a2 = sigmoid(z2);

    // Compute Deltas
    Eigen::MatrixXd delta2 = loss_gradient.cwiseProduct(sigmoid_derivative(z2));
    Eigen::MatrixXd delta1 = (delta2 * W2.transpose()).cwiseProduct(relu_derivative(z1));

    // Update Weights and Biases
    W2 -= learning_rate * (a1.transpose() * delta2);
    b2 -= learning_rate * delta2.colwise().sum();
    W1 -= learning_rate * (input.transpose() * delta1);
    b1 -= learning_rate * delta1.colwise().sum();
}

// Training Loop
void TestNNNode::train() {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        Eigen::MatrixXd predictions = forward(input);
        Eigen::MatrixXd loss_grad = loss_derivative(predictions, labels);

        double loss = compute_loss(predictions, labels);
        if (epoch % 100 == 0) {
            godot::UtilityFunctions::print("Epoch " + godot::String::num(epoch) + " Loss: " + godot::String::num(loss));
        }

        backward(loss_grad);
    }
}