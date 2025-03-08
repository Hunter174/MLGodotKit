extends Node

@onready var tree = $DTreeNode  # Ensure "DTreeNode" exists in the scene tree

func _ready():
	# Set minimum samples per split
	tree.set_min_samples_split(2)
	tree.set_max_depth(100)

	# Binary classification dataset (X_train and y_train)
	var X_train = [
		[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0], 
		[11.0], [12.0], [13.0], [14.0], [15.0], [16.0], [17.0], [18.0], [19.0], [20.0]
	]
	var y_train = [
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1
	]  # Binary classification: 0 for numbers <= 10, 1 for numbers > 10

	print("Training the Decision Tree on binary classification dataset...")
	tree.fit(X_train, y_train)

	# Binary classification test dataset
	var X_test = [[2.0], [5.0], [10.0], [11.0], [15.0], [19.0]]
	var y_true = [0, 0, 0, 1, 1, 1]  # Expected outputs (classification)

	var predictions = tree.predict(X_test)

	# Print predictions
	print("Predictions for test data: ", predictions)

	# Compute classification accuracy
	var accuracy = compute_accuracy(predictions, y_true)
	print("Classification Accuracy: ", accuracy)

	# Check accuracy threshold
	if accuracy >= 0.9:
		print("Decision Tree is performing well!")
	else:
		print("Decision Tree classification needs improvement.")

func compute_accuracy(predictions, y_true):
	var correct = 0
	var count = len(y_true)
	for i in range(count):
		if predictions[i] == y_true[i]:
			correct += 1
	return correct / count  # Accuracy formula
