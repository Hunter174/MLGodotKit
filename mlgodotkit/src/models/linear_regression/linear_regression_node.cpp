#include "linear_regression_node.h"

void LinearRegressionNode::_bind_methods() {
    godot::ClassDB::bind_method(
        godot::D_METHOD("fit", "inputs", "targets"),
        &LinearRegressionNode::fit
    );

    godot::ClassDB::bind_method(
        godot::D_METHOD("predict", "inputs"),
        &LinearRegressionNode::predict
    );
}

void LinearRegressionNode::fit(godot::Array inputs, godot::Array targets) {
    Eigen::MatrixXf X = Utils::godot_to_eigen(inputs);
    Eigen::MatrixXf y = Utils::godot_to_eigen(targets);

    ERR_FAIL_COND_MSG(X.rows() != y.rows(), "Inputs and targets row mismatch");
    ERR_FAIL_COND_MSG(y.cols() != 1, "Targets must be a column vector");

    // Add bias column (ones)
    Eigen::MatrixXf Xb(X.rows(), X.cols() + 1);
    Xb << X, Eigen::VectorXf::Ones(X.rows());

    // Closed-form using pseudoinverse for stability
    weights = Xb.completeOrthogonalDecomposition().pseudoInverse() * y;
}

godot::Array LinearRegressionNode::predict(godot::Array inputs) {
    ERR_FAIL_COND_V(weights.size() == 0, godot::Array());

    Eigen::MatrixXf X = Utils::godot_to_eigen(inputs);

    // Add bias column
    Eigen::MatrixXf Xb(X.rows(), X.cols() + 1);
    Xb << X, Eigen::VectorXf::Ones(X.rows());

    Eigen::VectorXf preds = Xb * weights;
    return Utils::eigen_to_godot(preds);
}
