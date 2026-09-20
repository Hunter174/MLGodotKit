#include <iostream>
#include <cassert>
#include <vector>
#include "mlgodotkit/src/models/neural_network/neural_network_core.h"
#include "mlgodotkit/src/matrix/matrix_node.h" // Note: MatrixNode is currently RefCounted, may need a MatrixCore

int main() {
    std::cout << "Running MLCore Unit Tests..." << std::endl;

    // Test 1: NeuralNetworkCore lifecycle
    {
        NeuralNetworkCore nn;
        nn.add_layer(2, 4, "relu");
        nn.add_layer(4, 1, "sigmoid");
        
        Eigen::MatrixXf input(1, 2);
        input << 0.5f, -0.2f;
        
        Eigen::MatrixXf output = nn.forward(input);
        
        if (output.rows() == 1 && output.cols() == 1) {
            std::cout << "[PASS] NeuralNetworkCore forward pass dimensions" << std::endl;
        } else {
            std::cerr << "[FAIL] NeuralNetworkCore forward pass dimensions" << std::endl;
            return 1;
        }
    }

    std::cout << "All core tests passed!" << std::endl;
    return 0;
}
