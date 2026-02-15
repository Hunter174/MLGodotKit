extends GutTest

func test_nn_learns_xor():
	var nn := NeuralNetworkNode.new()
	var loss := MSELossNode.new()

	# ---- Explicitly choose optimizer ----
	nn.set_optimizer("adam")   # Important: do not rely on constructor default

	# ---- Model Architecture ----
	nn.add_layer(2, 8, "relu")
	nn.add_layer(8, 8, "relu")
	nn.add_layer(8, 1, "sigmoid")

	# ---- Hyperparameters ----
	nn.set_learning_rate(0.01)
	nn.set_batch_size(4)
	nn.set_verbosity(0)  # keep tests quiet

	# ---- XOR Dataset ----
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

	# ---- Training Loop ----
	for epoch in range(3000):
		var prediction = nn.forward(inputs)
		loss.forward(prediction, targets)
		var grad = loss.backward()
		nn.backward(grad)

	# ---- Evaluation ----
	var correct := 0
	for i in range(inputs.size()):
		var out = nn.predict([inputs[i]])[0][0]
		var pred = 1 if out >= 0.5 else 0
		if pred == int(targets[i][0]):
			correct += 1

	var accuracy = float(correct) / inputs.size()

	nn.free()
	assert_true(accuracy >= 0.75)
