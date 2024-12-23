extends Node2D

@onready var test_nn = $TestNeuralNetwork

var xor_inputs = [
	[0, 0],  # XOR(0, 0) -> 0
	[0, 1],  # XOR(0, 1) -> 1
	[1, 0],  # XOR(1, 0) -> 1
	[1, 1]   # XOR(1, 1) -> 0
]
var xor_targets = [
	[0],     # Target for XOR(0, 0)
	[1],     # Target for XOR(0, 1)
	[1],     # Target for XOR(1, 0)
	[0]      # Target for XOR(1, 1)
]

func _ready():
	var layers = [2, 4, 2, 1]  # Input size 2, hidden layer size 3, output size 1
	test_nn.initialize(layers, "relu", "sigmoid")
	test_nn.set_optimizer("SGD", 0.1)  # Reduced learning rate
	test_nn.set_loss_function("BCE")
	train_xor()

func train_xor():
	var epochs = 5  # Reduced to 1000 for quicker debug cycles
	for epoch in range(epochs):
		var total_loss = 0.0
		for i in range(xor_inputs.size()):
			var input = xor_inputs[i]
			var target = xor_targets[i]
			test_nn.train(input, target)
			total_loss += test_nn.get_loss()
		
		if epochs % 1 == 0:
			print("Epoch ", epoch, ": Loss = ", total_loss / xor_inputs.size())

	print("Training complete. Testing network...")
	test_xor()

func test_xor():
	for i in range(xor_inputs.size()):
		var input = xor_inputs[i]
		var prediction = test_nn.predict(input)
		print(typeof(prediction), prediction)  # Debug the prediction type and content
		var result
		if prediction[0] >= 0.5:
			result = 1
		else:
			result = 0
		print("Input: ", input, ", Prediction: ", result)
