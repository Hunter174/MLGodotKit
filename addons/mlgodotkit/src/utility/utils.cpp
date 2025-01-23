#include "utils.h"

Eigen::MatrixXd Utils::godot_to_eigen(godot::Array array) {
    int rows = array.size();
    int cols = 0;

    // Handle 2D array case
    if (rows > 0 && array[0].get_type() == godot::Variant::ARRAY) {
        godot::Array first_row = array[0];
        cols = first_row.size();

        // Create an Eigen matrix for 2D arrays
        Eigen::MatrixXd out(rows, cols);
        for (int i = 0; i < rows; i++) {
            godot::Array row = array[i];
            for (int j = 0; j < cols; j++) {
                out(i, j) = static_cast<double>(row[j]);
            }
        }
        return out;
    }

    // Handle 1D array case (e.g., 1x2 or n×1)
    cols = rows; // A single 1D array will be treated as one row
    rows = 1;    // Force 1 row for 1D array inputs

    Eigen::MatrixXd out(rows, cols);
    for (int j = 0; j < cols; j++) {
        out(0, j) = static_cast<double>(array[j]);
    }

    return out;
}

godot::Array Utils::eigen_to_godot(Eigen::MatrixXd matrix){
    godot::Array out;

    for(int i=0;i<matrix.rows();i++){
        godot::Array row;
        for(int j=0;j<matrix.cols();j++){
            row.push_back(matrix(i,j));
        }
        if(matrix.rows() <= 1){
            return row;
        }
        out.push_back(row);
    }
    return out;
}

void Utils::debug_print(int verbosity, int debug_level, godot::Variant msg){
    if(verbosity == debug_level){
        godot::UtilityFunctions::print(msg);
    }
}

std::string Utils::eigen_to_string(const Eigen::MatrixXd& matrix) {
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

// Function to round the matrix values to a specific precision
Eigen::MatrixXd Utils::round_matrix(const Eigen::MatrixXd& mat, int precision) {
    // Compute the scaling factor based on the precision
    double scale = std::pow(10, precision);

    // Round the matrix elements to the desired precision
    Eigen::MatrixXd rounded = (mat.array() * scale).round() / scale;

    return rounded;
}


