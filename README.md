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

# Create a neural network instance.
# This is our function approximator.
@onready var nn_model := NeuralNetworkNode.new()

# Create a loss function.
# We explicitly define the loss instead of hardcoding gradients.
@onready var loss := MSELossNode.new()


# ============================================================
# XOR DATASET
# ============================================================
# XOR is a classic nonlinear problem.
# It cannot be solved by a single linear layer,
# which makes it a good test for neural networks.

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


# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================

var epochs := 5000
var print_interval := epochs / 10


func _ready():

	print("========== Neural Network XOR Example ==========\n")

	# ------------------------------------------------------------
	# STEP 1 — Configure Model
	# ------------------------------------------------------------

	# Verbosity controls internal debug logging.
	# Set to 1 to see detailed training information.
	nn_model.set_verbosity(0)

	# Learning rate controls how large weight updates are.
	nn_model.set_learning_rate(0.1)

	# We train one sample at a time (stochastic gradient descent).
	# Since batch size is explicitly enforced during training,
	# we set it to 1.
	nn_model.set_batch_size(1)

	# ------------------------------------------------------------
	# STEP 2 — Define Network Architecture
	# ------------------------------------------------------------

	# Input layer: 2 features
	nn_model.add_layer(2, 4, "relu")

	# Hidden layer
	nn_model.add_layer(4, 4, "relu")

	# Output layer: 1 neuron with sigmoid activation
	# Sigmoid squashes output into [0, 1] for classification.
	nn_model.add_layer(4, 1, "sigmoid")

	nn_model.model_summary()

	print("\nStarting training on XOR dataset...\n")


	# ============================================================
	# STEP 3 — TRAINING LOOP
	# ============================================================

	for epoch in range(1, epochs + 1):

		var total_loss := 0.0

		# Shuffle data each epoch for better learning
		var indices = [0, 1, 2, 3]
		indices.shuffle()

		for i in indices:

			# Wrap input into batch format (batch size = 1)
			var x = [inputs[i]]

			# Targets must match network output shape
			var target = [[targets[i][0]]]

			# -------------------------------
			# Forward pass
			# -------------------------------
			var prediction = nn_model.forward(x)

			# -------------------------------
			# Compute loss
			# -------------------------------
			# MSELossNode computes:
			# L = mean((prediction - target)^2)
			var loss_value = loss.forward(prediction, target)
			total_loss += loss_value

			# -------------------------------
			# Backward pass
			# -------------------------------
			# loss.backward() returns dL/dOutput
			var grad = loss.backward()

			# Pass gradient into network
			nn_model.backward(grad)

		# Print progress periodically
		if epoch % print_interval == 0:
			print("Epoch ", epoch,
			      " | Avg Loss: ",
			      total_loss / inputs.size())

	print("\nTraining completed!\n")


	# ============================================================
	# STEP 4 — INFERENCE (PREDICTION MODE)
	# ============================================================

	# IMPORTANT:
	# We use predict() instead of forward().
	#
	# forward() enforces training batch size.
	# predict() dynamically infers batch size
	# and is meant for evaluation/inference.

	print("Testing trained model on XOR:\n")

	var correct := 0

	for i in range(inputs.size()):

		var output = nn_model.predict([inputs[i]])

		# Output shape is [[value]]
		var value = output[0][0]

		print("Input: ", inputs[i],
		      " | Predicted: ", value,
		      " | Expected: ", targets[i][0])

		# Convert probability to class label
		var prediction = 1 if value >= 0.5 else 0

		if prediction == int(targets[i][0]):
			correct += 1

	var accuracy = float(correct) / inputs.size()

	print("\nFinal Accuracy: ", accuracy * 100.0, "%")

	if accuracy >= 0.75:
		print("NNNode passes XOR test!")
	else:
		print("NNNode failed XOR test. Needs improvement.")

	nn_model.model_summary()
```
---

## Credits
Built on the powerful [Eigen C++ library](https://eigen.tuxfamily.org/).


