#include "utils.h"
#include "utility/logger.h"

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

Eigen::MatrixXf Utils::godot_to_eigen(const godot::Array &arr, int batch_size) {
    if (arr.is_empty())
        return Eigen::MatrixXf();

    // Case 1: already a nested batch array [[...], [...]]
    if (arr[0].get_type() == godot::Variant::ARRAY) {
        int provided_batch = arr.size();
        int feature_size = ((godot::Array)arr[0]).size();
        Eigen::MatrixXf m(provided_batch, feature_size);
        for (int i = 0; i < provided_batch; i++) {
            godot::Array row = arr[i];
            for (int j = 0; j < feature_size; j++)
                m(i, j) = (float)row[j];
        }
        if (provided_batch != batch_size) {
            Logger::warn("Batch size mismatch: provided=" + std::to_string(provided_batch)
                + ", expected=" + std::to_string(batch_size));
        }
        return m;
    }

    // Case 2: flat 1D array treated as contiguous batch data
    int total_len = arr.size();
    if (batch_size == 1) {
        Eigen::MatrixXf m(1, total_len);
        for (int j = 0; j < total_len; j++)
            m(0, j) = (float)arr[j];
        return m;
    }

    if (total_len % batch_size != 0) {
        Logger::error_raise("godot_to_eigen() - Input size not divisible by batch size");
        return Eigen::MatrixXf();
    }

    int feature_size = total_len / batch_size;
    Eigen::MatrixXf m(batch_size, feature_size);
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < feature_size; j++)
            m(i, j) = (float)arr[i * feature_size + j];
    }
    return m;
}

Eigen::VectorXf Utils::godot_to_eigen_vector(godot::Array array) {
    int size = array.size();
    Eigen::VectorXf vec(size);

    for (int i = 0; i < size; i++) {
        vec(i) = static_cast<float>(array[i]);
    }

    return vec;
}

godot::Array Utils::eigen_to_godot(Eigen::MatrixXf matrix) {
    godot::Array out;
    for (int i = 0; i < matrix.rows(); i++) {
        godot::Array row;
        for (int j = 0; j < matrix.cols(); j++)
            row.push_back(matrix(i, j));
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


