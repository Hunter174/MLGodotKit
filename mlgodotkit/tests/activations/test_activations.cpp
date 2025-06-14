#include "catch_amalgamated.hpp"
#include "../src/activations/activations.h"

using namespace Activations;

TEST_CASE("Sigmoid activation output is between 0 and 1") {
    Eigen::MatrixXf x(1, 3);
    x << -10.0f, 0.0f, 10.0f;

    auto y = sigmoid(x);

    REQUIRE(y.rows() == 1);
    REQUIRE(y.cols() == 3);
    REQUIRE(y(0, 0) > 0.0f);
    REQUIRE(y(0, 0) < 0.5f);
    REQUIRE(y(0, 1) == Catch::Approx(0.5f).margin(1e-5f));
    REQUIRE(y(0, 2) < 1.0f);
    REQUIRE(y(0, 2) > 0.5f);
}

TEST_CASE("Sigmoid derivative behaves correctly") {
    Eigen::MatrixXf x(1, 3);
    x << -1.0f, 0.0f, 1.0f;

    auto dy = sigmoid_derivative(x);

    REQUIRE(dy.rows() == 1);
    REQUIRE(dy.cols() == 3);
    REQUIRE(dy(0, 0) > 0.0f);
    REQUIRE(dy(0, 1) == Catch::Approx(0.25f).margin(0.01f));  // sigmoid'(0) = 0.25
    REQUIRE(dy(0, 2) > 0.0f);
}

TEST_CASE("ReLU activation sets negatives to zero") {
    Eigen::MatrixXf x(1, 3);
    x << -1.0f, 0.0f, 2.5f;

    auto y = relu(x);

    REQUIRE(y(0, 0) == 0.0f);
    REQUIRE(y(0, 1) == 0.0f);
    REQUIRE(y(0, 2) == 2.5f);
}

TEST_CASE("ReLU derivative is 0 for negatives and 1 for positives") {
    Eigen::MatrixXf x(1, 3);
    x << -1.0f, 0.0f, 2.0f;

    auto dy = relu_derivative(x);

    REQUIRE(dy(0, 0) == 0.0f);  // negative
    REQUIRE(dy(0, 1) == 0.0f);  // at zero
    REQUIRE(dy(0, 2) == 1.0f);  // positive
}

TEST_CASE("Linear activation returns identity") {
    Eigen::MatrixXf x(2, 2);
    x << 1.0f, -2.0f,
         3.0f, 4.0f;

    auto y = linear(x);

    REQUIRE((y - x).norm() == Catch::Approx(0.0f));
}

TEST_CASE("Linear derivative returns matrix of ones") {
    Eigen::MatrixXf x(2, 2);
    x << -5.0f, 0.0f,
          7.0f, 1.0f;

    auto dy = linear_derivative(x);

    REQUIRE(dy.rows() == 2);
    REQUIRE(dy.cols() == 2);
    REQUIRE((dy.array() == 1.0f).all());
}
