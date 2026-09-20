extends Node3D
class_name SlewLimiter3D

@export var max_delta_per_sec := 600.0   # units per second
@export var enabled := true

var _prev: Vector3 = Vector3.ZERO
var _initialized := false

func reset(value: Vector3 = Vector3.ZERO) -> void:
	_prev = value
	_initialized = true

func filter(input: Vector3, delta: float) -> Vector3:
	if not enabled or delta <= 0.0:
		_prev = input
		_initialized = true
		return input

	if not _initialized:
		reset(input)
		return input

	var max_delta := max_delta_per_sec * delta
	var diff := input - _prev
	var len := diff.length()

	if len > max_delta:
		diff = diff / len * max_delta

	_prev += diff
	return _prev
