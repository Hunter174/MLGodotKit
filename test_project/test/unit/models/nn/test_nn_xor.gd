extends GutTest

func test_nn_learns_xor():
	var nn := NNNode.new()

	nn.add_layer(2, 8, "relu")
	nn.add_layer(8, 8, "relu")
	nn.add_layer(8, 1, "sigmoid")
	nn.set_learning_rate(0.01)
	nn.set_batch_size(1)
	nn.set_verbosity(0)

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

	for epoch in 3000:
		for i in inputs.size():
			var x = [inputs[i]]
			var y = targets[i][0]
			var out = nn.forward(x)[0][0]
			var err = out - y
			nn.backward([[2.0 * err]])

	var correct := 0
	for i in inputs.size():
		var out = nn.forward([inputs[i]])[0][0]
		var pred = 1 if out >= 0.5 else 0
		if pred == int(targets[i][0]):
			correct += 1
			
	nn.free()
	var accuracy = float(correct) / inputs.size()
	assert_true(accuracy >= 0.75)
