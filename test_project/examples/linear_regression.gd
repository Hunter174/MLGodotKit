extends Node2D

const ORIGIN := Vector2(80.0, 430.0)
const X_SCALE := 75.0
const Y_SCALE := 28.0

var model: LinearRegressionNode
var elapsed := 0.0
var paused := false
var status_label: Label
var samples: Array[Vector2] = []

func _ready() -> void:
	model = LinearRegressionNode.new()
	add_child(model)
	status_label = Label.new()
	status_label.position = Vector2(24.0, 20.0)
	status_label.add_theme_font_size_override("font_size", 18)
	add_child(status_label)
	_generate_samples()
	_update_model()

func _process(delta: float) -> void:
	if not paused:
		elapsed += delta
		_generate_samples()
		_update_model()
	queue_redraw()

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		if event.keycode == KEY_SPACE:
			paused = not paused

func _generate_samples() -> void:
	samples.clear()
	for i in range(11):
		var x := float(i)
		var noise := sin(elapsed * 2.0 + x * 1.7) * 0.45
		samples.append(Vector2(x, 2.0 * x + 1.0 + noise))

func _update_model() -> void:
	var inputs := []
	var targets := []
	for sample in samples:
		inputs.append([sample.x])
		targets.append([sample.y])
	model.fit(inputs, targets)
	var prediction = model.predict([[5.0]])
	status_label.text = "Live Linear Regression\n" + \
		"The data stream changes every frame; the blue line is refit continuously.\n" + \
		"Prediction at x=5: %.2f    SPACE: pause" % [prediction[0][0]]

func _draw() -> void:
	draw_rect(Rect2(Vector2.ZERO, Vector2(960.0, 540.0)), Color("101827"))
	for x in range(0, 11):
		var screen_x := ORIGIN.x + x * X_SCALE
		draw_line(Vector2(screen_x, 80.0), Vector2(screen_x, ORIGIN.y), Color("1b2a40"), 1.0)
	for y in range(0, 11):
		var screen_y := ORIGIN.y - y * Y_SCALE
		draw_line(Vector2(ORIGIN.x, screen_y), Vector2(900.0, screen_y), Color("1b2a40"), 1.0)

	for sample in samples:
		draw_circle(_to_screen(sample), 7.0, Color("efb366"))

	var previous := _to_screen(Vector2(0.0, model.predict([[0.0]])[0][0]))
	for x in range(1, 81):
		var data_x := float(x) / 8.0
		var current := _to_screen(Vector2(data_x, model.predict([[data_x]])[0][0]))
		draw_line(previous, current, Color("62b0ff"), 4.0)
		previous = current

func _to_screen(point: Vector2) -> Vector2:
	return ORIGIN + Vector2(point.x * X_SCALE, -point.y * Y_SCALE)
