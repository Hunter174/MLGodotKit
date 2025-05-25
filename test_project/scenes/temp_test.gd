extends Node

@onready var nn_model = NNNode.new()

# Majority-bit classification dataset
var inputs = [
	[0.0, 0.0, 0.0],  # Majority 0 → class 0
	[0.0, 0.0, 1.0],  # Majority 0 → class 0
	[1.0, 0.0, 0.0],  # Majority 0 → class 0
	[1.0, 1.0, 0.0],  # Majority 1 → class 1
	[0.0, 1.0, 1.0],  # Majority 1 → class 1
	[1.0, 1.0, 1.0],  # Majority 1 → class 1
]

var targets = [
	[0.0],
	[0.0],
	[0.0],
	[1.0],
	[1.0],
	[1.0],
]

var epochs = 3000
var print_interval = epochs / 10

func _ready():
	nn_model.set_learning_rate(0.1)
	nn_model.set_verbosity(0)

	# Define a basic classification network
	nn_model.add_layer(3, 4, "relu")
	nn_model.add_layer(4, 1, "sigmoid")

	nn_model.model_summary()
	print("\n🔥 Training on Majority Bit dataset...\n")

	for epoch in range(1, epochs + 1):
		var total_loss = 0.0
		for i in range(inputs.size()):
			var x = [inputs[i]]
			var y_true = targets[i][0]

			var output = nn_model.forward(x)
			var y_pred = output[0]
			var error = y_pred - y_true
			var loss = error * error
			total_loss += loss

			var grad = [[2.0 * error]]
			nn_model.backward(grad)

		if epoch % print_interval == 0:
			print("Epoch ", epoch, " | Avg Loss: ", total_loss / inputs.size())

	print("\n✅ Training complete!\n🚀 Evaluating model...")

	var correct = 0
	for i in range(inputs.size()):
		var x = [inputs[i]]
		var y_pred = nn_model.forward(x)[0]
		var prediction = 1 if y_pred >= 0.5 else 0
		var expected = int(targets[i][0])
		print("Input: ", x, " | Predicted: ", prediction, " (raw: ", y_pred, ") | Expected: ", expected)
		if prediction == expected:
			correct += 1

	var acc = float(correct) / inputs.size()
	print("\n🎯 Accuracy: ", acc * 100.0, "%")

	if acc >= 1.0:
		print("✅ NNNode passes Majority Bit classification!")
	else:
		print("❌ NNNode did not classify correctly. Needs tuning.")
