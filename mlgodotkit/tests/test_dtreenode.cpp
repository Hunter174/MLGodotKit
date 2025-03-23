#include "catch2/catch_amalgamated.hpp"

#include "../src/decision_tree/dtreenode.h"
#include "../src/utility/utils.h"

// Basic test to confirm Catch2 is running
TEST_CASE("Hello World Test", "[Hello]") {
    DTreeNode node;

    int a = 1;
    int b = 1;
    REQUIRE(a == b);  // Should pass
}


//TEST_CASE("Basic DTreeNode Functionality", "[DTreeNode]") {
//    DTreeNode node;
//    node.set_max_depth(3);
//    node.set_min_samples_split(2);
//
//    // Prepare Eigen matrix for input (2 samples, 2 features each)
//    Eigen::MatrixXf input_data(2, 2);
//    input_data << 1.0, 2.0,
//                  3.0, 4.0;
//
//    // Prepare Eigen vector for targets
//    Eigen::VectorXf target_data(2);
//    target_data << 0, 1;
//
//    // Convert Eigen data to Godot Arrays using existing Utils
//    godot::Array inputs = Utils::eigen_to_godot(input_data);
//    godot::Array targets = Utils::eigen_to_godot(target_data);  // Handles 1D vectors as rows
//
//    REQUIRE_NOTHROW(node.fit(inputs, targets));
//
//    // Now for prediction: prepare Eigen matrix and convert
//    Eigen::MatrixXf test_input_data(1, 2);  // Single test sample
//    test_input_data << 1.5, 2.5;
//
//    godot::Array test_input = Utils::eigen_to_godot(test_input_data);
//    godot::Array pred = node.predict(test_input);
//
//    REQUIRE(pred.size() == 1);
//    REQUIRE(int(pred[0]) == 0);  // Assuming the tree would classify this as 0
//}
