#ifndef UTILS_H
#define UTILS_H

#include <godot_cpp/variant/utility_functions.hpp>
#include <Eigen/Dense>
#include <sstream>

namespace Utils {
    Eigen::MatrixXd godot_to_eigen(godot::Array array);
    godot::Array eigen_to_godot(Eigen::MatrixXd matrix);
    void debug_print(int vebosity, int debug_level, godot::Variant msg);
    std::string eigen_to_string(const Eigen::MatrixXd& matrix);
    Eigen::MatrixXd round_matrix(const Eigen::MatrixXd& mat, int precision);

};

#endif //UTILS_H
