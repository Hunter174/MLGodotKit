extends Node2D

@onready var test_nn = $"../TestNeuralNetwork"

var and_inputs = [
	[0, 0],  # AND(0, 0) -> 0
	[0, 1],  # AND(0, 1) -> 0
	[1, 0],  # AND(1, 0) -> 0
	[1, 1]   # AND(1, 1) -> 1
]
var and_targets = [
	[0],     # Target for AND(0, 0)
	[0],     # Target for AND(0, 1)
	[0],     # Target for AND(1, 0)
	[1]      # Target for AND(1, 1)
]

func _ready():
	var layers = [2, 4, 1]  # Input size 2, one hidden layer size 4, output size 1
	test_nn.initialize(layers, "relu", "sigmoid")
	test_nn.set_optimizer("SGD", 0.01)  # Reduced learning rate
	test_nn.set_loss_function("MSE")
	test_nn.set_verbosity(1)

	train_and()

func train_and():
	var epochs = 2000  
	for epoch in range(epochs):
		var total_loss = 0.0
		for i in range(and_inputs.size()):
			var input = and_inputs[i]
			var target = and_targets[i]
			test_nn.train(input, target)
			total_loss += test_nn.get_loss()
		
		if epoch % 100 == 0:
			print("Epoch ", epoch, ": Loss = ", total_loss / and_inputs.size())

	print("Training complete. Testing network...\n\n")
	test_and()
	test_nn.model_summary()

func test_and():
	for i in range(and_inputs.size()):
		var input = and_inputs[i]
		var prediction = test_nn.predict(input)
		print("Q-value is: ", prediction)  # Debug the prediction type and content
		var result
		if prediction[0] >= 0.5:
			result = 1
		else:
			result = 0
		print("Input: ", input, ", Prediction: ", result)
