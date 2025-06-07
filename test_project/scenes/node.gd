extends Node

@onready var nn_model = NNNode.new()

# Linear regression dataset: output = sum(inputs)
var inputs = [
	[0.0, 0.0, 0.0],  # sum = 0.0
	[1.0, 0.0, 0.0],  # sum = 1.0
	[1.0, 1.0, 0.0],  # sum = 2.0
	[1.0, 1.0, 1.0],  # sum = 3.0
	[0.5, 0.5, 0.5],  # sum = 1.5
	[0.2, 0.3, 0.5],  # sum = 1.0
]

var targets = [
	[0.0],
	[1.0],
	[2.0],
	[3.0],
	[1.5],
	[1.0],
]

var epochs = 2000
var print_interval = epochs / 10

func _ready():
	nn_model.set_learning_rate(0.05)
	nn_model.set_verbosity(0)

	# Define a simple linear regression model
	nn_model.add_layer(3, 4, "relu")
	nn_model.add_layer(4, 1, "linear")  # Output is raw: Wx + b

	nn_model.model_summary()
	print("\n📈 Training on Linear Regression dataset...\n")

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

	print("\n✅ Training complete!\n🔍 Evaluating model...")

	for i in range(inputs.size()):
		var x = [inputs[i]]
		var y_pred = nn_model.forward(x)[0]
		var expected = targets[i][0]
		print("Input: ", x, " | Predicted: %.4f" % y_pred, " | Expected: ", expected)

	print("\n🚀 Linear Regression test complete!")
