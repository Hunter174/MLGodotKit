extends Node2D

const DIST := 600.0  # Pixels to move

# Durations in seconds (from your benchmark results)
const DURATION_GD := 62.956 / 1000.0
const DURATION_CPP := 1.078 / 1000.0

func _ready():
	# Create GDScript Ball
	var ball_gd = make_ball(Color.RED)
	ball_gd.position = Vector2(0, 100)
	add_child(ball_gd)

	var label_gd := make_label("GDScript", Vector2(0, 70))
	add_child(label_gd)

	# Create C++ Ball
	var ball_cpp = make_ball(Color.BLUE)
	ball_cpp.position = Vector2(0, 300)
	add_child(ball_cpp)

	var label_cpp := make_label("C++", Vector2(0, 270))
	add_child(label_cpp)

	# Animate both
	animate_ball(ball_gd, DURATION_GD)
	animate_ball(ball_cpp, DURATION_CPP)

func make_ball(color: Color) -> Node2D:
	var ball := Node2D.new()
	var sprite := Sprite2D.new()
	sprite.texture = get_circle_texture(color)
	sprite.centered = true
	ball.add_child(sprite)
	return ball

func make_label(text: String, pos: Vector2) -> Label:
	var label := Label.new()
	label.text = text
	label.position = pos
	label.modulate = Color(1, 1, 1)
	return label
	
func animate_ball(ball: Node2D, duration: float) -> void:
	var tween := get_tree().create_tween()
	tween.tween_property(ball, "position", ball.position + Vector2(DIST, 0), duration)
	
func get_circle_texture(color: Color) -> Texture2D:
	var img := Image.create(32, 32, false, Image.FORMAT_RGBA8)
	img.fill(Color(0, 0, 0, 0))
	#img.lock()
	for x in 32:
		for y in 32:
			var dx = x - 16
			var dy = y - 16
			if dx * dx + dy * dy <= 16 * 16:
				img.set_pixel(x, y, color)
	#img.unlock()
	var tex := ImageTexture.create_from_image(img)
	return tex
