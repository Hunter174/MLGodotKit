#include "nnnode.h"

NNNode::NNNode() {}

NNNode::~NNNode() {}

void NNNode::_bind_methods() {
    godot::ClassDB::bind_method(godot::D_METHOD("add_layer", "input_size", "output_size", "activation"), &NNNode::add_layer);
    godot::ClassDB::bind_method(godot::D_METHOD("forward", "input"), &NNNode::forward);
    godot::ClassDB::bind_method(godot::D_METHOD("backward", "error"), &NNNode::backward);
    godot::ClassDB::bind_method(godot::D_METHOD("model_summary"), &NNNode::model_summary);
    godot::ClassDB::bind_method(godot::D_METHOD("set_learning_rate", "lr"), &NNNode::set_learning_rate);
}

void NNNode::add_layer(int input_size, int output_size, godot::String activation){
    std::string activation_type = activation.utf8().get_data();

    layers.push_back(Layer(input_size, output_size, learning_rate, activation_type));
}

godot::Array NNNode::forward(godot::Array input){
    Eigen::MatrixXd x = godot_to_eigen(input);

    for(int i=0;i<layers.size(); i++){
        x = layers[i].forward(x);
    }

    godot::Array output = eigen_to_godot(x);
    return output;
}

void NNNode::backward(godot::Array error) {
    Eigen::MatrixXd grad_loss = godot_to_eigen(error);

    // Backward pass through layers
    for (int i = layers.size() - 1; i >= 0; --i) {
        grad_loss = layers[i].backward(grad_loss);
    }

//    // Decay the learning rate
//    set_learning_rate((learning_rate*.99));
}

// Helper Functions
Eigen::MatrixXd NNNode::godot_to_eigen(godot::Array array) {
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

godot::Array NNNode::eigen_to_godot(Eigen::MatrixXd matrix){
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

void NNNode::model_summary() {
    for (int i = 0; i < layers.size(); i++) {
        godot::UtilityFunctions::print(layers[i].to_string());
    }
}