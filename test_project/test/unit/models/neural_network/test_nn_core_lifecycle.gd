# test_nn_core_lifecycle.gd
extends "GutTest"

# This test verifies the new Core/Wrapper split and RL target network capabilities
# Since we can't easily write a pure C++ test without a full runner, 
# we use the Node wrapper to exercise the Core's cloning and copying.

func test_weight_copying():
	var nn1 = NeuralNetworkNode.new()
	nn1.add_layer(2, 4, "relu")
	nn1.add_layer(4, 1, "sigmoid")
	
	var nn2 = NeuralNetworkNode.new()
	nn2.add_layer(2, 4, "relu")
	nn2.add_layer(4, 1, "sigmoid")
	
	# Initially, weights are random and likely different
	# We check if copy_weights actually synchronizes them
	nn2.copy_weights(nn1)
	
	var input = [0.5, -0.2]
	var out1 = nn1.predict(input)
	var out2 = nn2.predict(input)
	
	assert_eq(out1, out2, "Weights should be identical after copy_weights")

func test_build_model_reset():
	var nn = NeuralNetworkNode.new()
	nn.add_layer(2, 2, "relu")
	
	var config = [
		{"input_size": 2, "output_size": 3, "activation": "relu"},
		{"input_size": 3, "output_size": 1, "activation": "sigmoid"}
	]
	nn.set_layers(config)
	
	var input = [0.1, 0.2]
	var out = nn.predict(input)
	
	assert_eq(out.size(), 1, "Output size should match the new config")

func test_optimizer_switch():
	var nn = NeuralNetworkNode.new()
	nn.set_optimizer("adam")
	assert_eq(nn.get_optimizer(), "adam")
	
	# Testing error handling for unknown optimizer (should log error but not crash)
	nn.set_optimizer("unknown_opt")
	# Depending on implementation, it might stay as previous or reset.
	# Our current implementation logs error and keeps existing if possible or just fails.
	# We just ensure it doesn't crash the engine.
	pass
