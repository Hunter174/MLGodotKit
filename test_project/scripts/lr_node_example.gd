extends Node2D

const SCALE := 100.0  # 1 unit = 100 pixels

@onready var lr_model = $LRNode
@onready var plot_area = $PlotArea
@onready var line = $PlotArea/Line2D
@onready var points_container = $PlotArea/DataPoints
@onready var input_slider = $UI/InputSlider
@onready var prediction_label = $UI/PredictionLabel
@onready var train_button = $UI/TrainButton

var inputs: Array = []
var targets: Array = []

func _ready():
	prediction_label.text = "Click to Add Points"

func _input(event):
	if event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_LEFT:
		# Restrict clicks to PlotArea bounds
		if not plot_area.get_global_rect().has_point(event.position):
			return

		var local_pos = event.position - plot_area.get_global_position()
		var input_val = local_pos.x / SCALE
		var target_val = local_pos.y / -SCALE  # Flip Y
		inputs.append([input_val])
		targets.append([target_val])
		_add_point_visual(local_pos)


func _add_point_visual(pos: Vector2):
	var point = ColorRect.new()
	point.color = Color.RED
	point.size = Vector2(8, 8)
	point.position = pos - point.size / 2
	points_container.add_child(point)

func _draw_line():
	var x0 = 0.0
	var x1 = 10.0
	var y0 = lr_model.predict([x0])[0]
	var y1 = lr_model.predict([x1])[0]

	line.points = [
		Vector2(x0 * SCALE, -y0 * SCALE),
		Vector2(x1 * SCALE, -y1 * SCALE)
	]

func _on_slider_changed(value):
	var x = value
	var y = lr_model.predict([x])[0]
	prediction_label.text = "x = %.2f → y = %.2f" % [x, y]

	if not plot_area.has_node("PredictionDot"):
		var dot = ColorRect.new()
		dot.name = "PredictionDot"
		dot.color = Color.BLUE
		dot.size = Vector2(10, 10)
		plot_area.add_child(dot)

	var pred_dot = plot_area.get_node("PredictionDot")
	pred_dot.position = Vector2(x * SCALE, -y * SCALE) - pred_dot.size / 2

func _on_button_pressed() -> void:
	if inputs.is_empty():
		print("No data!")
		return

	lr_model.set_learning_rate(0.01)
	lr_model.initialize(1)
	lr_model.train(inputs, targets, 1000)
	_draw_line()
