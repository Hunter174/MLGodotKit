extends Node2D

@onready var lr_model = $LRNode

# Example dataset: simple linear function y = 3x + 5 with noise
var inputs = [[1], [2], [3], [4], [5]]
var targets = [[8], [11], [14], [17], [20]]  # y = 3x + 5

func _ready():
	lr_model.set_learning_rate(0.01)
	lr_model.initialize(1)  # Single feature input
	lr_model.train(inputs, targets, 1000)

	print("Predictions after training:")
	for i in range(len(inputs)):
		var pred_y = lr_model.predict(inputs[i])
		print("Input: ", inputs[i], " Predicted: ", pred_y, " Actual: ", targets[i])
