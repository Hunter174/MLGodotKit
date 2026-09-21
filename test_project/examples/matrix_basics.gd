extends Node2D

const CENTER := Vector2(480.0, 290.0)
const SCALE := 150.0

var elapsed := 0.0
var status_label: Label

func _ready() -> void:
	status_label = Label.new()
	status_label.position = Vector2(24.0, 20.0)
	status_label.add_theme_font_size_override("font_size", 18)
	add_child(status_label)
	queue_redraw()

func _process(delta: float) -> void:
	elapsed += delta
	var angle := elapsed * 0.8
	status_label.text = "Matrix Transform\n" + \
		"A Matrix rotates and scales the square every frame.\n" + \
		"Matrix × Vector2 uses the same data path as gameplay transforms.\n" + \
		"SPACE: pause"
	queue_redraw()

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo and event.keycode == KEY_SPACE:
		set_process(not is_processing())

func _draw() -> void:
	draw_rect(Rect2(Vector2.ZERO, Vector2(960.0, 540.0)), Color("101827"))
	draw_line(CENTER - Vector2(420.0, 0), CENTER + Vector2(420.0, 0), Color("334b68"), 2.0)
	draw_line(CENTER - Vector2(0, 220.0), CENTER + Vector2(0, 220.0), Color("334b68"), 2.0)

	var angle := elapsed * 0.8
	var matrix := Matrix.from_array([
		[cos(angle), -sin(angle)],
		[sin(angle), cos(angle)],
	])
	var square := [
		Vector2(-1.0, -1.0),
		Vector2(1.0, -1.0),
		Vector2(1.0, 1.0),
		Vector2(-1.0, 1.0),
	]
	for i in range(square.size()):
		var transformed: Vector2 = matrix.mul_vector2(square[i]) * SCALE
		var next_transformed: Vector2 = matrix.mul_vector2(square[(i + 1) % square.size()]) * SCALE
		draw_line(CENTER + transformed, CENTER + next_transformed, Color("62b0ff"), 5.0)
		draw_circle(CENTER + transformed, 7.0, Color("efb366"))
