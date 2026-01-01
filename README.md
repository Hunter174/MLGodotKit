# MLGodotKit

**Empower your Godot projects with the power of machine learning!**  
MLGodotKit is a C++ GDExtension for Godot, enabling seamless integration of AI-driven features into your games and applications. With support for adaptive behaviors and real-time decision-making, it’s designed to inspire innovation and enhance gameplay.

<p align="center">
  <img src="docs/_static/MLGodotKit_logo.png" alt="MLGodotKit Logo" width="1000"/>
</p>

> *If you'd like to see something else add an issue!*

## Getting Started

Here are quick examples of how to use the three core models in **MLGodotKit**:

> *All models are GDExtension nodes and can be used directly in any Godot scene or script.*

---

## Linear Regression (`LRNode`)

```gdscript
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
			print("FAIL: Input: ", inputs[i], " Predicted: ", pred_y, " Actual: ", actual_y)
			all_pass = false
		else:
			print("PASS: Input: ", inputs[i], " Predicted: ", pred_y, " Actual: ", actual_y)

	if all_pass:
		print("Linear Regression test passed. \n")
	else:
		print("Linear Regression test failed. \n")
```

---

## Descision Tree Classifier (`DTreeNode`)

```gdscript
extends Node

@onready var tree = DTreeNode.new()

func _ready():
	print("========== Test 2: DTree Classification Node ==========\n")
	tree.set_min_samples_split(2)
	tree.set_max_depth(100)

	var X_train = [
		[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0],
		[11.0], [12.0], [13.0], [14.0], [15.0], [16.0], [17.0], [18.0], [19.0], [20.0]
	]
	var y_train = [
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1
	]

	tree.fit(X_train, y_train)

	var X_test = [[2.0], [5.0], [10.0], [11.0], [15.0], [19.0]]
	var y_true = [0, 0, 0, 1, 1, 1]
	var predictions = tree.predict(X_test)

	var correct = 0
	for i in range(y_true.size()):
		if predictions[i] == y_true[i]:
			print("PASS: Input ", X_test[i], " Predicted: ", predictions[i], " Expected: ", y_true[i])
			correct += 1
		else:
			print("FAIL: Input ", X_test[i], " Predicted: ", predictions[i], " Expected: ", y_true[i])

	var accuracy = float(correct) / y_true.size()
	print("\nAccuracy: \n", accuracy)

	if accuracy >= 0.9:
		print("Decision Tree test passed. \n")
	else:
		print("Decision Tree test failed. \n")
```
---

## Neural Network (`NNNode`)

> *Note: All loss functions are left to the user to implement please see examples for suggested implementation
will be implemented natively at a later date.*

```gdscript
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

	print("\nStarting training on XOR dataset...\n")

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
	
	print("\nTraining completed!\n")

	# Inference and Evaluation
	print("Testing trained model on XOR:")
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
	print("\n Final Accuracy: ", accuracy * 100.0, "%")

	if accuracy >= 0.75:  # We expect at least 75% on XOR
		print("NNNode passes XOR test!")
	else:
		print("NNNode failed XOR test. Needs improvement.")
	
	nn_model.model_summary()
```
---

## Credits
Built on the powerful [Eigen C++ library](https://eigen.tuxfamily.org/).


