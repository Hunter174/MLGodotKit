extends Node

@onready var nn_model = NNNode.new()

# Simple XOR dataset (non-linear, good for testing)
var inputs = [
	[0.0, 0.0],
	[0.0, 1.0],
	[1.0, 0.0],
	[1.0, 1.0]
]
var targets = [
	[0.0],  # 0 XOR 0 = 0
	[1.0],  # 0 XOR 1 = 1
	[1.0],  # 1 XOR 0 = 1
	[0.0]   # 1 XOR 1 = 0
]

# Hyperparameters
var epochs = 5000
var print_interval = epochs/10  # How often to print loss

func _ready():
	# Initialize NN model
	nn_model.set_verbosity(0)  # Set to 1 if you want to see all internal prints
	nn_model.set_learning_rate(0.1)
	
	# Add layers
	nn_model.add_layer(2, 4, "relu")
	nn_model.add_layer(4, 4, "relu")
	nn_model.add_layer(4, 1, "sigmoid")
	
	# Print model summary
	nn_model.model_summary()

	print("\n🔥 Starting training on XOR dataset...\n")

	# Training loop
	for epoch in range(1, epochs + 1):
		var total_loss = 0.0
		
		var indices = [0, 1, 2, 3]
		indices.shuffle()
		for i in indices:
			var x = [inputs[i]]
			var y_true = targets[i][0]
			# then process x and y_true as before

			
			# Forward pass
			var output = nn_model.forward(x)
			var y_pred = output[0] 
			
			# Compute error (difference)
			var error = y_pred - y_true
			
			# Compute simple squared loss
			var loss = error * error
			total_loss += loss
			
			# Backward pass (send gradient of loss w.r.t output)
			var grad = [ [2.0 * error] ]  # d(Loss)/d(Prediction)
			nn_model.backward(grad)
		
		# Print loss at intervals
		if epoch % print_interval == 0:
			print("Epoch ", epoch, " | Avg Loss: ", total_loss / inputs.size())
	
	print("\n✅ Training completed!\n")

	# Inference and Evaluation
	print("🚀 Testing trained model on XOR:")
	for i in range(inputs.size()):
		var x = [inputs[i]]
		var output = nn_model.forward(x)
		print("Input: ", x, " | Predicted: ", output, " | Expected: ", targets[i])

	# Auto-check: classify and compare
	var correct = 0
	for i in range(inputs.size()):
		var x = [inputs[i]]
		var output = nn_model.forward(x)
		var prediction = 1 if output[0] >= 0.5 else 0
		if prediction == int(targets[i][0]):
			correct += 1
	
	var accuracy = float(correct) / inputs.size()
	print("\n🎯 Final Accuracy: ", accuracy * 100.0, "%")

	if accuracy >= 0.75:  # We expect at least 75% on XOR
		print("✅ NNNode passes XOR test!")
	else:
		print("❌ NNNode failed XOR test. Needs improvement.")
	
	nn_model.model_summary()
