extends Node2D

@onready var test_nn = $TestNeuralNetwork

# XOR Dataset
var inputs = [
	[0, 0], 
	[0, 1], 
	[1, 0], 
	[1, 1]
]
var outputs = [
	[0],  # 0 XOR 0 = 0
	[1],  # 0 XOR 1 = 1
	[1],  # 1 XOR 0 = 1
	[1]   # 1 XOR 1 = 0
]

func _ready():
	# Define the network architecture
	test_nn.set_learning_rate(0.01)
	test_nn.add_layer(2, 4, "sigmoid")  # Hidden Layer
	test_nn.add_layer(4, 1, "sigmoid")  # Output Layer
	test_nn.set_verbosity(1)
	#test_nn.model_summary()
	
	# Training parameters
	var epochs = 10000
	
	# Training loop - > Try using ADAM next time
	for epoch in range(epochs):
		var total_loss = 0
		var grad_loss_batch = []

		# Process the entire batch
		for i in range(len(inputs)):
			var pred_y = test_nn.forward(inputs[i])  # Forward pass
			#test_nn.model_summary()
			
			# Compute the error (MSE for one sample)
			var error = mse_loss(outputs[i], pred_y)
			total_loss += error
			
			# Compute the gradient of the loss w.r.t predictions
			var grad_loss = compute_mse_gradient(outputs[i], pred_y)
		
			# Perform backward pass (using averaged gradient)
			#print("================")
			test_nn.backward([grad_loss])
			#test_nn.model_summary()
		
		# Print loss for the epoch every 100 epochs
		if (epoch + 1) % 10 == 0:
			print("Epoch %d, Loss: %f" % [epoch + 1, total_loss / len(inputs)])
			#test_nn.model_summary()

	# Are the weights updating?
	test_nn.model_summary()

	# Evaluate the model
	evaluate_model()

# Helper function to compute Mean Squared Error (MSE)
func mse_loss(target, prediction):
	var loss = 0.0
	for i in range(len(target)):
		loss += pow((target[i] - prediction[i]), 2)  # Element-wise subtraction and squaring
	return loss / len(target)  # Average loss

# Compute the gradient of the MSE loss
func compute_mse_gradient(target, prediction):
	var grad_loss = []
	for i in range(len(target)):
		grad_loss.append(2 * (prediction[i] - target[i]) / len(target))  # Element-wise gradient
	return grad_loss

# Evaluate the model on the XOR dataset
func evaluate_model():
	print("Evaluating the model on the XOR dataset:")
	var correct_predictions = 0
	
	for i in range(len(inputs)):
		var pred_y = test_nn.forward(inputs[i])  # Forward pass
		var predicted
		print(pred_y)
		if pred_y[0] >= 0.5:
			predicted = 1
		else: predicted = 0
		
		var actual = outputs[i][0]
		print("Input: %s, Predicted: %d, Actual: %d" % [inputs[i], predicted, actual])
		
		if predicted == actual:
			correct_predictions += 1
	
	var accuracy = float(correct_predictions) / len(inputs) * 100
	print("Accuracy: %.2f%%" % accuracy)
