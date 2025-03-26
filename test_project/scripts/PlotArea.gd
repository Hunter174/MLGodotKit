extends Control

const GRID_SPACING := 100
const GRID_COLOR := Color(0.8, 0.8, 0.8, 0.4)
const AXIS_COLOR := Color(0.2, 0.2, 0.2, 0.8)

func _ready():
	queue_redraw()

func _draw():
	var size = get_size()
	# Draw vertical grid lines
	for x in range(0, int(size.x), GRID_SPACING):
		draw_line(Vector2(x, 0), Vector2(x, size.y), GRID_COLOR, 1)

	# Draw horizontal grid lines
	for y in range(0, int(size.y), GRID_SPACING):
		draw_line(Vector2(0, y), Vector2(size.x, y), GRID_COLOR, 1)

	# Draw axes (X axis across center horizontally, Y on left)
	var origin = Vector2(0, size.y / 2)
	draw_line(Vector2(0, origin.y), Vector2(size.x, origin.y), AXIS_COLOR, 2)  # X-axis
	draw_line(Vector2(0, 0), Vector2(0, size.y), AXIS_COLOR, 2)  # Y-axis
