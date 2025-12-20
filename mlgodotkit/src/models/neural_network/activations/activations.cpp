#include "activations.h"
#include <cmath>

static constexpr float epsilon = 1e-6f;

namespace Activations {

	// -----------------------------------------------------------------
	//   Internal activation map registry (static, translation-unit local)
	// -----------------------------------------------------------------
	static const std::unordered_map<std::string,
		std::function<Eigen::MatrixXf(const Eigen::MatrixXf&)>> ACTIVATIONS = {
		{"sigmoid", sigmoid},
		{"relu", relu},
		{"linear", linear},
		{"tanh",  [](const Eigen::MatrixXf& x) { return x.array().tanh().matrix(); }},
		{"leaky_relu", [](const Eigen::MatrixXf& x) { return leaky_relu(x, 0.01f); }}
	};

	static const std::unordered_map<std::string,
		std::function<Eigen::MatrixXf(const Eigen::MatrixXf&)>> DERIVATIVES = {
		{"sigmoid", sigmoid_derivative},
		{"relu", relu_derivative},
		{"linear", linear_derivative},
		{"tanh", [](const Eigen::MatrixXf& z) {
			Eigen::MatrixXf t = z.array().tanh().matrix();
			return (1.0f - t.array().square()).matrix();
		}},
		{"leaky_relu", [](const Eigen::MatrixXf& z) { return leaky_relu_derivative(z, 0.01f); }}
	};

	// -----------------------------------------------------------------
	//   Core functions
	// -----------------------------------------------------------------
	Eigen::MatrixXf linear(const Eigen::MatrixXf& x) {
		return x;
	}

	Eigen::MatrixXf linear_derivative(const Eigen::MatrixXf& z) {
		return Eigen::MatrixXf::Ones(z.rows(), z.cols());
	}

	Eigen::MatrixXf sigmoid(const Eigen::MatrixXf& x) {
		Eigen::MatrixXf result = x.unaryExpr([](float v) {
			if (v >= 0.0f) {
				float z = std::exp(-v);
				return 1.0f / (1.0f + z);
			} else {
				float z = std::exp(v);
				return z / (1.0f + z);
			}
		});
		result = result.array().min(1.0f - epsilon).max(epsilon);
		return result;
	}

	Eigen::MatrixXf sigmoid_derivative(const Eigen::MatrixXf& z) {
		Eigen::MatrixXf s = sigmoid(z);
		return s.array() * (1.0f - s.array());
	}



	Eigen::MatrixXf relu(const Eigen::MatrixXf& x) {
		return x.cwiseMax(0.0f);
	}

	Eigen::MatrixXf relu_derivative(const Eigen::MatrixXf& z) {
		return (z.array() > 0.0f).cast<float>();
	}

	Eigen::MatrixXf leaky_relu(const Eigen::MatrixXf& x, float alpha) {
		return x.unaryExpr([alpha](float v) { return v > 0.0f ? v : alpha * v; });
	}

	Eigen::MatrixXf leaky_relu_derivative(const Eigen::MatrixXf& z, float alpha) {
		return z.unaryExpr([alpha](float v) { return v > 0.0f ? 1.0f : alpha; });
	}

	// -----------------------------------------------------------------
	//   Public map accessors
	// -----------------------------------------------------------------
	std::function<Eigen::MatrixXf(const Eigen::MatrixXf&)> get_activation(const std::string& name) {
		auto it = ACTIVATIONS.find(name);
		if (it != ACTIVATIONS.end()) return it->second;
		return relu; // fallback
	}

	std::function<Eigen::MatrixXf(const Eigen::MatrixXf&)> get_derivative(const std::string& name) {
		auto it = DERIVATIVES.find(name);
		if (it != DERIVATIVES.end()) return it->second;
		return relu_derivative; // fallback
	}
} // namespace Activations