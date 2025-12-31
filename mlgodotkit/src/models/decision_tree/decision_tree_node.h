#ifndef DecisionTreeNode_H
#define DecisionTreeNode_H

#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/core/error_macros.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/classes/node.hpp>
#include "utility/utils.h"
#include <Eigen/Dense>
#include <vector>
#include <limits>
#include <map>

using namespace godot;

class DecisionTreeNode : public godot::Node {
    GDCLASS(DecisionTreeNode, godot::Node);

private:
  
    struct SplitData {
        Eigen::MatrixXf left_X, right_X;
        Eigen::VectorXf left_y, right_y;
    };
  
    struct SubNode {
        int feature_idx;
        float threshold;
        SubNode* left;
        SubNode* right;
        int value;
        bool is_leaf;

        SubNode() : left(nullptr), right(nullptr), is_leaf(false), value(-1) {}
    };

    SubNode* root;
    int max_depth;
    int min_samples_split;

    // Recursively build the tree
    SubNode* buildTree(const Eigen::MatrixXf& X,
                    const Eigen::VectorXf& y,
                    int depth);

    // Find best feature to split on
    void findBestSplit(const Eigen::MatrixXf& X,
                       const Eigen::VectorXf& y,
                       int& best_feature,
                       float& best_threshold);

    SplitData splitData(const Eigen::MatrixXf& X,
                        const Eigen::VectorXf& y,
                        int feature_idx, float threshold);

    float calculateImpurity(const Eigen::VectorXf& y);

    int computeLeafValue(const Eigen::VectorXf& y);

    int predictRecursive(SubNode* node, const Eigen::VectorXf& sample) const;

    void freeTree(SubNode* node);

public:
    DecisionTreeNode();
    ~DecisionTreeNode();

    static void _bind_methods();

    void fit(godot::Array inputs, godot::Array targets);
    void set_min_samples_split(int min_samples);

    void set_max_depth(int depth);
    int get_max_depth() const;

    godot::Array predict(godot::Array inputs);
};

#endif // DecisionTreeNode_H