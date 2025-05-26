// tests/test_layer.cpp
#include "catch_amalgamated.hpp"
#include "../src/layer/layer.h"

TEST_CASE("Layer forward produces correct shape") {
    Layer layer(3, 2, 0.01f, "linear");

    Eigen::MatrixXf x(1, 3);
    x << 1.0f, 2.0f, 3.0f;

    auto out = layer.forward(x);

    REQUIRE(out.rows() == 1);
    REQUIRE(out.cols() == 2);
}
