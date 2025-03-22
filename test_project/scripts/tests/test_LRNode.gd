extends Node

@onready var lr_model = LRNode.new()

func _ready():
	print("========== Test 1: Linear Regression Node ==========\n")
	lr_model.set_learning_rate(0.01)
	lr_model.initialize(1)  # Single feature input

	# Example dataset: y = 3x + 5
	var inputs = [[1], [2], [3], [4], [5]]
	var targets = [[8], [11], [14], [17], [20]]

	lr_model.train(inputs, targets, 1000)
	print()

	var all_pass = true  # Flag to track success/failure
	var tolerance = 1.0  # Acceptable error margin for predictions

	for i in range(inputs.size()):
		var pred_y = lr_model.predict(inputs[i])
		var actual_y = targets[i][0]
		var error = abs(pred_y[0] - actual_y)
		if error > tolerance:
			print("❌ FAIL: Input: ", inputs[i], " Predicted: ", pred_y, " Actual: ", actual_y)
			all_pass = false
		else:
			print("✅ PASS: Input: ", inputs[i], " Predicted: ", pred_y, " Actual: ", actual_y)

	if all_pass:
		print("✅ Linear Regression test passed. \n")
	else:
		print("❌ Linear Regression test failed. \n")
