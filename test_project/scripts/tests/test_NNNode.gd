extends Node

@onready var nn_model: NNNode = NNNode.new()

var inputs = [
	[0.0, 0.0],
	[0.0, 1.0],
	[1.0, 0.0],
	[1.0, 1.0]
]
var targets = [
	[0.0],
	[1.0],
	[1.0],
	[0.0]
]

var epochs = 5000
var print_interval = epochs / 10

func _ready():
	nn_model.add_layer(2, 8, "relu")
	nn_model.add_layer(8, 8, "relu")
	nn_model.add_layer(8, 1, "sigmoid")
	nn_model.set_learning_rate(0.01)
	nn_model.set_batch_size(1)  # ✅ single-sample batches for XOR
	nn_model.model_summary()

	print("\n🔥 Starting training on XOR dataset...\n")

	for epoch in range(1, epochs + 1):
		var total_loss := 0.0
		var indices = [0, 1, 2, 3]
		indices.shuffle()

		for i in indices:
			var x = [inputs[i]]  # 1×2 batch
			var y_true = targets[i][0]

			# --- Forward pass ---
			var output = nn_model.forward(x)      # [[y_hat]]
			var y_pred = output[0][0]              # ✅ extract scalar

			# --- Compute loss ---
			var error = y_pred - y_true
			var loss = error * error
			total_loss += loss

			# --- Backprop ---
			var grad = [[2.0 * error]]  # 1×1 gradient batch
			nn_model.backward(grad)

		if epoch % print_interval == 0:
			print("Epoch ", epoch, " | Avg Loss: ", total_loss / inputs.size())

	print("\n✅ Training completed!\n")

	# --- Inference ---
	print("🚀 Testing trained model on XOR:")
	for i in range(inputs.size()):
		var x = [inputs[i]]
		var output = nn_model.forward(x)
		print("Input: ", x, " | Predicted: ", output, " | Expected: ", targets[i])

	# --- Accuracy ---
	var correct := 0
	for i in range(inputs.size()):
		var x = [inputs[i]]
		var output = nn_model.forward(x)
		var prediction = 1 if output[0][0] >= 0.5 else 0  # ✅ extract scalar
		if prediction == int(targets[i][0]):
			correct += 1

	var accuracy = float(correct) / inputs.size()
	print("\n🎯 Final Accuracy: ", accuracy * 100.0, "%")
	if accuracy >= 0.75:
		print("✅ NNNode passes XOR test!")
	else:
		print("❌ NNNode failed XOR test. Needs improvement.")
