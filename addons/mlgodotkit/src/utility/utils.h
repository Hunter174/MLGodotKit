#ifndef UTILS_H
#define UTILS_H

#include <godot_cpp/variant/utility_functions.hpp>
#include <Eigen/Dense>

namespace Utils {
    Eigen::MatrixXd godot_to_eigen(godot::Array array);
    godot::Array eigen_to_godot(Eigen::MatrixXd matrix);
};

#endif //UTILS_H
