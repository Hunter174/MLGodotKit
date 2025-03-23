#include "utils.h"

Eigen::MatrixXf Utils::godot_to_eigen(godot::Array array) {
    int rows = array.size();
    int cols = 0;

    // Handle 2D array case
    if (rows > 0 && array[0].get_type() == godot::Variant::ARRAY) {
        godot::Array first_row = array[0];
        cols = first_row.size();

        // Create an Eigen matrix for 2D arrays
        Eigen::MatrixXf out(rows, cols);
        for (int i = 0; i < rows; i++) {
            godot::Array row = array[i];
            for (int j = 0; j < cols; j++) {
                out(i, j) = static_cast<float>(row[j]);
            }
        }
        return out;
    }

    // Handle 1D array case (e.g., 1x2 or n×1)
    cols = rows; // A single 1D array will be treated as one row
    rows = 1;    // Force 1 row for 1D array inputs

    // Handle 1D array case (treat as Nx1 column vector)
    Eigen::MatrixXf out(rows, 1);
    for (int i = 0; i < rows; i++) {
        out(i, 0) = static_cast<float>(array[i]);
    }

    return out;
}

Eigen::VectorXf Utils::godot_to_eigen_vector(godot::Array array) {
    int size = array.size();
    Eigen::VectorXf vec(size);

    for (int i = 0; i < size; i++) {
        vec(i) = static_cast<float>(array[i]);
    }

    return vec;
}

godot::Array Utils::eigen_to_godot(Eigen::MatrixXf matrix){
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

std::string Utils::eigen_to_string(const Eigen::MatrixXf& matrix) {
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
Eigen::MatrixXf Utils::round_matrix(const Eigen::MatrixXf& mat, int precision) {
    // Compute the scaling factor based on the precision
    double scale = std::pow(10, precision);

    // Round the matrix elements to the desired precision
    Eigen::MatrixXf rounded = (mat.array() * scale).round() / scale;

    return rounded;
}


