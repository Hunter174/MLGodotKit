#ifndef GODOT_UTILITY_H
#define GODOT_UTILITY_H

#include <string>
#include <sstream>
#include <Eigen/Dense>

namespace GodotUtils {

    // Inline to avoid multiple definition errors
    inline std::string eigen_to_string(const Eigen::MatrixXf& matrix) {
        std::ostringstream stream;
        stream << "[";
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                stream << matrix(i, j);
                if (j < matrix.cols() - 1) stream << ", ";
            }
            if (i < matrix.rows() - 1) stream << "; ";
        }
        stream << "]";
        return stream.str();
    }

}

#endif // GODOT_UTILITY_H