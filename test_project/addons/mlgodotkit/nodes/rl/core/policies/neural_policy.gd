class_name NeuralPolicy
extends Policy

var model = NeuralNetworkNode.new()

func select_action(state):
	var logits = model.forward(state)
	return logits
