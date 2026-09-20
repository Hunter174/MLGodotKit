class_name MovementControllerNode2D
extends Node2D

## The MovementControllerNode2D handles the conversion of a target position
## into a desired velocity vector.

var target_position := Vector2.ZERO
var arrival_radius := 10.0
var stop_on_arrival := true
var limits := Vector2(-1000, 1000)

func set_target(pos: Vector2) -> void:
	target_position = pos

func set_limits(min_val: float, max_val: float) -> void:
	limits = Vector2(min_val, max_val)

func update(current_pos: Vector2, delta: float) -> Vector2:
	var dir = (target_position - current_pos).normalized()
	var dist = current_pos.distance_to(target_position)

	if stop_on_arrival and dist < arrival_radius:
		return Vector2.ZERO

	# Simple proportional control for demo purposes
	# This will be replaced by actual ML logic in the full kit
	var speed = clamp(dist * 2.0, limits.x, limits.y)
	return dir * speed
