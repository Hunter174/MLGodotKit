#include "catch_amalgamated.hpp"
#include "../src/layer/layer.h"
#include <iostream>

TEST_CASE("Layer forward produces correct shape") {
    Layer layer(3, 2, 0.01f, "linear");

    Eigen::MatrixXf x(1, 3);
    x << 1.0f, 2.0f, 3.0f;

    auto out = layer.forward(x);

    REQUIRE(out.rows() == 1);
    REQUIRE(out.cols() == 2);
}

TEST_CASE("Layer backward updates weights and biases") {
    Layer layer(2, 1, 0.1f, "linear");

    Eigen::MatrixXf x(1, 2);
    x << 1.0f, 2.0f;

    layer.forward(x);

    Eigen::MatrixXf error(1, 1);
    error << 1.0f;

    auto before_weights = layer.to_string(); // Snapshot

    layer.backward(error);
    auto after_weights = layer.to_string();

    REQUIRE(before_weights != after_weights);
}

TEST_CASE("Weight initialization sets correct shapes via constructor") {
    Layer layer(4, 3, 0.01f, "relu");

    // Use the Layer's internal to_string() for introspection
    std::string desc = layer.to_string();

    REQUIRE(desc.find("Weights (Shape: 4x3") != std::string::npos);
    REQUIRE(desc.find("Biases (Shape: 3") != std::string::npos);  // Biases is 1x3 but flattened in output
}

TEST_CASE("Layer to_string contains weight and bias shapes") {
    Layer layer(3, 2, 0.01f, "relu");
    auto str = layer.to_string();

    REQUIRE(str.find("Weights") != std::string::npos);
    REQUIRE(str.find("Biases") != std::string::npos);
}

TEST_CASE("Layer remains stable over many forward/backward passes") {
    const int input_size = 4;
    const int output_size = 4;
    const float lr = 0.01f;

    Layer layer(input_size, output_size, lr, "relu");

    Eigen::MatrixXf input(1, input_size);
    input.setRandom();

    for (int i = 0; i < 1000; ++i) {
        auto output = layer.forward(input);
        REQUIRE((output.array().isFinite()).all());

        Eigen::MatrixXf grad = Eigen::MatrixXf::Random(1, output_size);
        auto grad_input = layer.backward(grad);

        REQUIRE((grad_input.array().isFinite()).all());

        // Weights should remain finite
        auto w = layer.get_weights();
        auto b = layer.get_biases();
        REQUIRE((w.array().isFinite()).all());
        REQUIRE((b.array().isFinite()).all());
    }
}

TEST_CASE("Layer stack avoids vanishing/exploding activations") {
    std::vector<Layer> layers;
    layers.emplace_back(4, 8, 0.01f, "relu");
    layers.emplace_back(8, 8, 0.01f, "relu");
    layers.emplace_back(8, 4, 0.01f, "relu");

    Eigen::MatrixXf input(1, 4);
    input.setRandom();

    for (int step = 0; step < 500; ++step) {
        Eigen::MatrixXf x = input;

        // Forward pass
        for (auto& l : layers) {
            x = l.forward(x);
            REQUIRE((x.array().isFinite()).all());
        }

        // Gradients from loss (random for test)
        Eigen::MatrixXf grad = Eigen::MatrixXf::Random(1, 4);

        // Backward pass
        for (int i = layers.size() - 1; i >= 0; --i) {
            grad = layers[i].backward(grad);
            REQUIRE((grad.array().isFinite()).all());
        }

        // Optional: check for saturation
        for (const auto& l : layers) {
            auto w = l.get_weights();
            REQUIRE((w.array().abs() < 100.0f).all());  // catch explosion
        }
    }
}

TEST_CASE("Layer detects exploding or vanishing gradients") {
    Layer layer(10, 10, 0.01f, "sigmoid");

    Eigen::MatrixXf input = Eigen::MatrixXf::Random(1, 10);
    layer.forward(input);

    for (int i = 0; i < 100; ++i) {
        Eigen::MatrixXf grad = Eigen::MatrixXf::Ones(1, 10) * 0.001f;
        auto grad_input = layer.backward(grad);

        float grad_norm = grad_input.norm();
        REQUIRE(std::isfinite(grad_norm));
        REQUIRE(grad_norm < 100.0f);   // Exploding
        REQUIRE(grad_norm > 1e-6f);    // Vanishing
    }
}

TEST_CASE("Layer network learns XOR logic") {

    // Define XOR dataset
    std::vector<Eigen::Vector2f> inputs = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };

    std::vector<float> targets = {0.0f, 1.0f, 1.0f, 0.0f};

    // Define network: 2 → 4 → 1
    Layer hidden(2, 4, 0.1f, "relu");
    Layer output(4, 1, 0.1f, "sigmoid");

    int epochs = 5000;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int i = 0; i < 4; ++i) {
            Eigen::MatrixXf x(1, 2);
            x << inputs[i].transpose();

            Eigen::MatrixXf y(1, 1);
            y << targets[i];

            // Forward
            auto h = hidden.forward(x);
            auto y_pred = output.forward(h);

            // Compute gradient (MSE loss)
            Eigen::MatrixXf error = y_pred - y;

            // Backward
            auto grad_output = output.backward(error);
            auto grad_hidden = hidden.backward(grad_output);

            // Check for finite values
            REQUIRE((y_pred.array().isFinite()).all());
            REQUIRE((grad_output.array().isFinite()).all());
        }
    }

    // Final accuracy check
    int correct = 0;
    for (int i = 0; i < 4; ++i) {
        Eigen::MatrixXf x(1, 2);
        x << inputs[i].transpose();

        float prediction = output.forward(hidden.forward(x))(0, 0);
        bool predicted_class = prediction >= 0.5f;
        bool actual_class = targets[i] >= 0.5f;

        if (predicted_class == actual_class) {
            correct++;
        }
    }

    // Compute accuracy
    float accuracy = static_cast<float>(correct) / 4.0f;
    std::cout << "XOR Test Accuracy: " << accuracy * 100.0f << "%\n";

    REQUIRE(correct == 4);
    std::cout << "Non-linear decision boundary verified.\n";
}

// Optional Stress tests
TEST_CASE("Long-term stability under heavy load", "[stress]") {
    Layer layer(10, 10, 0.01f, "relu");
    Eigen::MatrixXf input = Eigen::MatrixXf::Random(1, 10);

    for (int i = 0; i < 1000000; ++i) {
        auto out = layer.forward(input);
        REQUIRE((out.array().isFinite()).all());

        Eigen::MatrixXf grad = Eigen::MatrixXf::Random(1, 10);
        auto back = layer.backward(grad);

        REQUIRE((back.array().isFinite()).all());

        // Optionally: fail if weights explode
        REQUIRE(layer.get_weights().array().abs().maxCoeff() < 100.0f);
    }
}

TEST_CASE("Deep network with varied layer sizes remains numerically stable", "[stress]") {

    // Create a deep and wide stack of layers with mixed sizes
    std::vector<Layer> layers = {
        Layer(16, 64, 0.01f, "relu"),
        Layer(64, 128, 0.01f, "relu"),
        Layer(128, 64, 0.01f, "relu"),
        Layer(64, 32, 0.01f, "relu"),
        Layer(32, 16, 0.01f, "relu"),
        Layer(16, 8,  0.01f, "relu"),
        Layer(8,  4,  0.01f, "relu"),
        Layer(4,  2,  0.01f, "relu")
    };

    Eigen::MatrixXf input(1, 16);
    input.setRandom();

    for (int step = 0; step < 5000; ++step) {
        Eigen::MatrixXf x = input;

        // Forward pass through all layers
        for (auto& l : layers) {
            x = l.forward(x);
            REQUIRE((x.array().isFinite()).all());
        }

        // Simulate a gradient from loss
        Eigen::MatrixXf grad = Eigen::MatrixXf::Random(1, 2);

        // Backward pass through all layers
        for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
            grad = layers[i].backward(grad);
            REQUIRE((grad.array().isFinite()).all());
        }

        // Check each layer's weights and biases for stability
        for (const auto& l : layers) {
            auto w = l.get_weights();
            auto b = l.get_biases();

            REQUIRE((w.array().isFinite()).all());
            REQUIRE((b.array().isFinite()).all());
            REQUIRE(w.array().abs().maxCoeff() < 1e3f);  // reasonable bound
            REQUIRE(b.array().abs().maxCoeff() < 1e3f);
        }
    }
}