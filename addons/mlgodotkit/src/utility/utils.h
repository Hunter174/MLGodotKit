#ifndef UTILS_H
#define UTILS_H

#include <godot_cpp/variant/utility_functions.hpp>
#include <Eigen/Dense>
#include <sstream>

namespace Utils {

    // Conversion tools from godot to eigen
    Eigen::MatrixXf godot_to_eigen(godot::Array array);
    Eigen::VectorXf godot_to_eigen_vector(godot::Array array);

    // Conversion tools from eigen to godot
    godot::Array eigen_to_godot(Eigen::MatrixXf matrix);

    // Other Utility functions
    void debug_print(int verbosity, int debug_level, godot::Variant msg);
    std::string eigen_to_string(const Eigen::MatrixXf& matrix);
    Eigen::MatrixXf round_matrix(const Eigen::MatrixXf& mat, int precision);
};

#endif //UTILS_H
