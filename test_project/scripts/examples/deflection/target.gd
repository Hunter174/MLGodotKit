extends Node2D

@export var move_radius := 200.0
@export var move_speed := 1.2
@export var velocity_scale := 180.0
@export var center_offset := Vector2.ZERO

var time := 0.0
var velocity := Vector2.ZERO
var center := Vector2.ZERO

func _ready():
	center = global_position + center_offset

func _process(delta):
	time += delta

	# Lissajous-style motion (smooth, non-trivial)
	var x = cos(time * move_speed) * move_radius
	var y = sin(time * move_speed * 1.7) * move_radius * 0.6

	var new_pos = center + Vector2(x, y)

	# Compute velocity explicitly (important for NN input)
	velocity = (new_pos - global_position) / max(delta, 0.0001)

	global_position = new_pos
