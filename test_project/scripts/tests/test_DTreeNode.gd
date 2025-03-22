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
			print("✅ PASS: Input ", X_test[i], " Predicted: ", predictions[i], " Expected: ", y_true[i])
			correct += 1
		else:
			print("❌ FAIL: Input ", X_test[i], " Predicted: ", predictions[i], " Expected: ", y_true[i])

	var accuracy = float(correct) / y_true.size()
	print("\nAccuracy: \n", accuracy)

	if accuracy >= 0.9:
		print("✅ Decision Tree test passed. \n")
	else:
		print("❌ Decision Tree test failed. \n")
