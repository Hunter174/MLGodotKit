extends Control

func _ready() -> void:
	var network := NeuralNetworkNode.new()
	var loss := MSELossNode.new()
	add_child(network)

	network.set_learning_rate(0.05)
	network.set_batch_size(4)
	network.add_layer(2, 4, "relu")
	network.add_layer(4, 1, "sigmoid")

	var inputs := [
		[0.0, 0.0],
		[0.0, 1.0],
		[1.0, 0.0],
		[1.0, 1.0],
	]
	var targets := [[0.0], [1.0], [1.0], [0.0]]

	for _epoch in range(500):
		var prediction = network.forward(inputs)
		loss.forward(prediction, targets)
		network.backward(loss.backward())

	var outputs := []
	for input in inputs:
		outputs.append(network.predict([input])[0][0])
	_show_result("Neural network XOR", [
		"Inputs: 00, 01, 10, 11",
		"Outputs: %s" % [outputs],
		"Training flow: forward → loss → backward",
	])

func _show_result(title: String, lines: Array) -> void:
	var label := Label.new()
	label.position = Vector2(32, 32)
	label.add_theme_font_size_override("font_size", 20)
	label.text = title + "\n\n" + "\n".join(lines)
	add_child(label)
