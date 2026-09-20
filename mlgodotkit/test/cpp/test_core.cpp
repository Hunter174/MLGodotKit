#include "extern/catch_amalgamated.hpp"
#include "mlgodotkit/src/models/neural_network/neural_network_core.h"
#include "mlgodotkit/src/control/pid_controller/pid_controller_core.h"
#include <Eigen/Dense>

TEST_CASE("NeuralNetworkCore Basic Forward Pass", "[nn][core]") {
    NeuralNetworkCore nn;
    nn.add_layer(2, 4, "relu");
    nn.add_layer(4, 1, "sigmoid");

    Eigen::MatrixXf input(1, 2);
    input << 0.5f, -0.2f;

    Eigen::MatrixXf output = nn.forward(input);

    REQUIRE(output.rows() == 1);
    REQUIRE(output.cols() == 1);
    REQUIRE(output(0, 0) >= 0.0f);
    REQUIRE(output(0, 0) <= 1.0f);
}

TEST_CASE("PIDControllerCore Basic Logic", "[control][core]") {
    PIDControllerCore pid;
    pid.kp = 1.0f;
    pid.ki = 0.0f;
    pid.kd = 0.0f;
    pid.lim_min = -10.0f;
    pid.lim_max = 10.0f;

    // P-only: output = kp * (setpoint - measurement)
    float out = pid.update(10.0f, 5.0f, 0.1f);
    REQUIRE(out == 5.0f);

    // Test clamping
    pid.kp = 100.0f;
    out = pid.update(10.0f, 0.0f, 0.1f);
    REQUIRE(out == 10.0f);
}

TEST_CASE("NeuralNetworkCore Weight Copying", "[nn][core]") {
    NeuralNetworkCore nn1;
    nn1.add_layer(2, 2, "relu");
    
    NeuralNetworkCore nn2;
    nn2.add_layer(2, 2, "relu");
    
    nn2.copy_weights_from(nn1);
    
    Eigen::MatrixXf input(1, 2);
    input << 1.0f, 1.0f;
    
    REQUIRE(nn1.forward(input) == nn2.forward(input));
}
