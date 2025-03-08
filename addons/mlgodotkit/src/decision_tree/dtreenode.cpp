#include "dtreenode.h"

using namespace godot;

void DTreeNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("fit", "inputs", "targets"), &DTreeNode::fit);
    ClassDB::bind_method(D_METHOD("predict", "inputs"), &DTreeNode::predict);
    ClassDB::bind_method(D_METHOD("set_min_samples_split", "min_samples"), &DTreeNode::set_min_samples_split);
    ClassDB::bind_method(D_METHOD("set_max_depth", "depth"), &DTreeNode::set_max_depth);
    ClassDB::bind_method(D_METHOD("get_max_depth"), &DTreeNode::get_max_depth);
}

DTreeNode::DTreeNode() : root(nullptr), max_depth(10), min_samples_split(2) {}

DTreeNode::~DTreeNode() {
    freeTree(root);
}

void DTreeNode::freeTree(SubNode* node) {
    if (node) {
        freeTree(node->left);
        freeTree(node->right);
        delete node;
        node = nullptr;
    }
}


// Find the majority Vote
int DTreeNode::computeLeafValue(const Eigen::VectorXf& y){
  	// Ordered map for tie breaks
	std::map<int, int> class_counts;

    // Count the occurences for each class
	for(int i = 0; i < y.size(); i++){
          int label = static_cast<int>(y(i));
          class_counts[label]++;
	}

    int majority_class = -1;
    int max_count = 0;

    // Find the majority Class
    for(const auto& pair : class_counts){
    	if (pair.second > max_count){
          max_count = pair.second;
          majority_class = pair.first;
    	}
    }

    return majority_class;
}

float DTreeNode::calculateImpurity(const Eigen::VectorXf& y){
	if(y.size() <= 1){
          return 0.0f;
    }

    std::unordered_map<int, int> class_counts;
    int total_samples = y.size();

    //Count the occurences of each class
    for(int i = 0; i < total_samples; i++){
    	int label = static_cast<int>(y(i));
    	class_counts[label]++;
    }

    // Compute Gini Impurity
    float gini = 1.0f;
    for (const auto& pair : class_counts) {
    	float prob = static_cast<float>(pair.second) / total_samples;
        gini -= prob * prob; // Subtract the squared prob
	}

    return gini;
}

DTreeNode::SplitData DTreeNode::splitData(const Eigen::MatrixXf& X, const Eigen::VectorXf& y,
                                          int feature_idx, float threshold) {
	std::vector<int> left_indices, right_indices;

    // Partition the data left and right by threshold
    for(int i = 0; i < X.rows(); i++){
    	if (X(i, feature_idx) <= threshold) {
        	left_indices.push_back(i);
        } else {
        	right_indices.push_back(i);
        }
    }

    // Create left and right subsets
    Eigen::MatrixXf left_X(left_indices.size(), X.cols());
    Eigen::MatrixXf right_X(right_indices.size(), X.cols());
    Eigen::VectorXf left_y(left_indices.size());
    Eigen::VectorXf right_y(right_indices.size());

    // Fill new matrices
    for (size_t i = 0; i < left_indices.size(); i++) {
        left_X.row(i) = X.row(left_indices[i]);
        left_y(i) = y(left_indices[i]);
    }
    for (size_t i = 0; i < right_indices.size(); i++) {
        right_X.row(i) = X.row(right_indices[i]);
        right_y(i) = y(right_indices[i]);
    }

    return {left_X, right_X, left_y, right_y};
}

void DTreeNode::findBestSplit(const Eigen::MatrixXf& X, const Eigen::VectorXf& y,
                              int& best_feature, float& best_threshold) {
	float best_impurity = std::numeric_limits<float>::max();
    best_feature = -1;
    best_threshold = std::numeric_limits<float>::quiet_NaN();

    int num_samples = X.rows();
    int num_features = X.cols();

    for(int feature = 0; feature < num_features; feature++){
    	// Get the unique feature values
        std::vector<float> unique_values;
        for(int i = 0; i < num_samples; i++){
        	unique_values.push_back(X(i, feature));
        }

        // Sort and remove duplicates
        std::sort(unique_values.begin(), unique_values.end());
        unique_values.erase(std::unique(unique_values.begin(), unique_values.end()), unique_values.end());

        for(float threshold : unique_values){
        	SplitData split = splitData(X, y, feature, threshold);

            // Skip invalid splits
            if(split.left_y.size() == 0 || split.right_y.size() == 0){
            	continue;
            }

            // Find Gini impurity
            float left_impurity = calculateImpurity(split.left_y);
            float right_impurity = calculateImpurity(split.right_y);

            // Compute weighted impurity
            float total_size = split.left_y.size() + split.right_y.size();
			float weighted_impurity = (split.left_y.size() * left_impurity + split.right_y.size() * right_impurity) / total_size;


            // Get the weighted best split
            if(weighted_impurity < best_impurity){
            	best_impurity = weighted_impurity;
                best_feature = feature;
                best_threshold = threshold;
            }
        }
    }
}


DTreeNode::SubNode* DTreeNode::buildTree(const Eigen::MatrixXf& X, const Eigen::VectorXf& y, int depth) {

    // Base Case: Stop if we reach max depth or too few samples
    if (X.rows() < min_samples_split || depth >= max_depth) {
        SubNode* leaf = new SubNode();
        leaf->value = computeLeafValue(y);
        leaf->is_leaf = true;
        return leaf;
    }

    // Check if all labels are the same (pure node)
    if ((y.array() == y(0)).all()) {
        SubNode* leaf = new SubNode();
        leaf->value = static_cast<int>(y(0)); // Convert float to int
        leaf->is_leaf = true;
        return leaf;
    }

    // Find best split
    int best_feature;
    float best_threshold;
    findBestSplit(X, y, best_feature, best_threshold);

    // If no valid split is found, create a leaf node
    if (best_feature == -1 || std::isnan(static_cast<float>(best_threshold))) {
        SubNode* leaf = new SubNode();
        leaf->value = computeLeafValue(y);
        leaf->is_leaf = true;
        return leaf;
    }

    // Split data
    SplitData split = splitData(X, y, best_feature, best_threshold);

    // Create node and recursively build left & right children
    SubNode* node = new SubNode();
    node->feature_idx = best_feature;
    node->threshold = best_threshold;

    node->left = buildTree(split.left_X, split.left_y, depth + 1);
    node->right = buildTree(split.right_X, split.right_y, depth + 1);

    return node;
}

void DTreeNode::fit(godot::Array inputs, godot::Array targets) {
	// Convert Godot arrays to Eigen matrices
	Eigen::MatrixXf X = Utils::godot_to_eigen(inputs);
	Eigen::VectorXf y = Utils::godot_to_eigen_vector(targets);

    // Free tree if it was initialized
	if(root){
    	freeTree(root);
        root = nullptr;
    }

	root = buildTree(X, y, 0);
}

godot::Array DTreeNode::predict(godot::Array inputs) {
    // Ensure the tree is trained before prediction
    if (!root) {
        ERR_PRINT("Error: Decision Tree has not been fit.");
        return godot::Array();
    }


    // Convert Godot array to Eigen matrix
    Eigen::MatrixXf X = Utils::godot_to_eigen(inputs);
    int num_samples = X.rows();

    // Initialize output array
    godot::Array predictions;

    // Predict for each sample
    for (int i = 0; i < num_samples; i++) {
        Eigen::VectorXf sample = X.row(i);  // Extract row as a sample
        int pred = predictRecursive(root, sample);
        predictions.push_back(pred);
    }

    return predictions;
}

int DTreeNode::predictRecursive(SubNode* node, const Eigen::VectorXf& sample) const {
    // Base Case: If we reach a leaf node, return its stored class value
    if (node->is_leaf) {
        return node->value;
    }

    // Traverse the tree based on the feature value
    if (sample(node->feature_idx) <= node->threshold) {
        return predictRecursive(node->left, sample);  // Go left
    } else {
        return predictRecursive(node->right, sample); // Go right
    }
}


// GETTERS and SETTERS
void DTreeNode::set_min_samples_split(int min_samples) {
    // Ensure min_samples_split is at least 2
    if (min_samples < 2) {
        ERR_PRINT("Warning: min_samples_split must be at least 2. Setting to 2.");
        min_samples_split = 2;
    } else {
        min_samples_split = min_samples;
    }
}

void DTreeNode::set_max_depth(int depth) {
    if (depth < 1) {
        ERR_PRINT("Warning: max_depth must be at least 1. Setting to 1.");
        max_depth = 1;
    } else {
        max_depth = depth;
    }
}

int DTreeNode::get_max_depth() const {
    return max_depth;
}
