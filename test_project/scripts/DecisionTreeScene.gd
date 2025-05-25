extends Node2D

@onready var dtree = $DTreeNode
@onready var plot_area = $PlotArea
@onready var points_container = $PlotArea/DataPoints
@onready var train_button = $UI/TrainButton
@onready var toggle_button = $UI/ClassToggleButton

const SCALE := 100.0
const GRID_RES := 10  # Grid resolution for the background map
var current_class := 0  # Toggle between class 0 and 1

var inputs: Array = []
var targets: Array = []

func _ready():
	train_button.pressed.connect(_on_train_button_pressed)
	toggle_button.pressed.connect(_on_toggle_button_pressed)
	_update_toggle_label()

func _input(event):
	if event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_LEFT:
		if not plot_area.get_global_rect().has_point(event.position):
			return

		var local_pos = event.position - plot_area.get_global_position()
		var input_vec = [local_pos.x / SCALE, local_pos.y / -SCALE]
		inputs.append(input_vec)
		targets.append(current_class)
		_add_point_visual(local_pos, current_class)

func _add_point_visual(pos: Vector2, cls: int):
	var point = ColorRect.new()
	if cls == 0:
		point.color = Color.RED
	else:
		point.color = Color.GREEN

	point.size = Vector2(8, 8)
	point.position = pos - point.size / 2
	points_container.add_child(point)

func _update_toggle_label():
	toggle_button.text = "Current Class: %d" % current_class

func _on_toggle_button_pressed() -> void:
	current_class = 1 - current_class
	_update_toggle_label()

func _on_train_button_pressed() -> void:
	if inputs.is_empty():
		print("No data to train on!")
		return

	dtree.set_max_depth(5)
	dtree.fit(inputs, targets)
	_render_classification_map()

func _render_classification_map():
	var size = plot_area.get_size()
	var img = Image.create(int(size.x), int(size.y), false, Image.FORMAT_RGB8)

	for x in range(0, int(size.x), GRID_RES):
		for y in range(0, int(size.y), GRID_RES):
			var fx = x / SCALE
			var fy = y / -SCALE
			var prediction = dtree.predict([[fx, fy]])[0]

			var color: Color
			if prediction == 0:
				color = Color(1, 0.9, 0.9)  # Light red
			else:
				color = Color(0.9, 1, 0.9)  # Light green

			for dx in range(GRID_RES):
				for dy in range(GRID_RES):
					var px = x + dx
					var py = y + dy
					if px < img.get_width() and py < img.get_height():
						img.set_pixel(px, py, color)

	var tex = ImageTexture.create_from_image(img)
	var map_node = plot_area.get_node("ClassificationMap") as TextureRect
	map_node.texture = tex
