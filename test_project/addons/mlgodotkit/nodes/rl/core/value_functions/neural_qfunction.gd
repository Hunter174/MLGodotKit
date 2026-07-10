class_name NeuralQFunction
extends RefCounted

var model : NeuralNetworkNode

func _init(_model):
	model = _model

func evaluate(state, action):
	var q_values = model.predict([state])[0]
	return q_values[action]

func evaluate_all(state):
	return model.predict([state])[0]
