extends Node2D

const ORIGIN := Vector2(480.0, 290.0)
const GRID_SIZE := 18
const CELL_SIZE := 24.0

var network: NeuralNetworkNode
var loss: MSELossNode
var elapsed := 0.0
var paused := false
var status_label: Label
var outputs := []

var inputs := [
	[0.0, 0.0],
	[0.0, 1.0],
	[1.0, 0.0],
	[1.0, 1.0],
]
var targets := [[0.0], [1.0], [1.0], [0.0]]

func _ready() -> void:
	network = NeuralNetworkNode.new()
	loss = MSELossNode.new()
	add_child(network)

	network.set_learning_rate(0.05)
	network.set_batch_size(4)
	network.add_layer(2, 4, "relu")
	network.add_layer(4, 1, "sigmoid")
	_train(500)

	status_label = Label.new()
	status_label.position = Vector2(24.0, 20.0)
	status_label.add_theme_font_size_override("font_size", 18)
	add_child(status_label)
	_update_status()
	queue_redraw()

func _process(delta: float) -> void:
	if not paused:
		elapsed += delta
		# Continue learning so the scene demonstrates an online training loop.
		_train(2)
		_update_status()
	queue_redraw()

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		if event.keycode == KEY_SPACE:
			paused = not paused
		if event.keycode == KEY_R:
			_train(500)

func _train(epochs: int) -> void:
	for _epoch in range(epochs):
		var prediction = network.forward(inputs)
		loss.forward(prediction, targets)
		network.backward(loss.backward())
	outputs.clear()
	for input in inputs:
		outputs.append(network.predict([input])[0][0])

func _update_status() -> void:
	if status_label == null:
		return
	status_label.text = "Neural Network XOR Classifier\n" + \
		"The background is the live prediction field.\n" + \
		"Outputs: %s    SPACE: pause    R: retrain" % [outputs]

func _draw() -> void:
	draw_rect(Rect2(Vector2.ZERO, Vector2(960.0, 540.0)), Color("101827"))
	for y in range(GRID_SIZE):
		for x in range(GRID_SIZE):
			var input := [float(x) / float(GRID_SIZE - 1), float(y) / float(GRID_SIZE - 1)]
			var value: float = network.predict([input])[0][0]
			var color := Color("2f6da8").lerp(Color("d66b69"), value)
			var position := Vector2(
				ORIGIN.x + (float(x) - GRID_SIZE / 2.0) * CELL_SIZE,
				ORIGIN.y + (float(y) - GRID_SIZE / 2.0) * CELL_SIZE)
			draw_rect(Rect2(position, Vector2(CELL_SIZE - 1.0, CELL_SIZE - 1.0)), color)

	for i in range(inputs.size()):
		var point := _to_screen(inputs[i])
		var point_color := Color("f4d35e") if targets[i][0] > 0.5 else Color("ffffff")
		draw_circle(point, 10.0, point_color)
		draw_circle(point, 10.0, Color("101827"), false, 2.0)

func _to_screen(input: Array) -> Vector2:
	return ORIGIN + Vector2(
		(input[0] - 0.5) * GRID_SIZE * CELL_SIZE,
		(input[1] - 0.5) * GRID_SIZE * CELL_SIZE)
