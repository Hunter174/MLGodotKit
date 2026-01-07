extends Node3D
class_name LowPassFilter3D

@export var cutoff_hz := 6.0   # higher = less smoothing
@export var enabled := true

var _state: Vector3 = Vector3.ZERO
var _initialized := false

func reset(value: Vector3 = Vector3.ZERO) -> void:
	_state = value
	_initialized = true

func filter(input: Vector3, delta: float) -> Vector3:
	if not enabled or delta <= 0.0:
		_state = input
		_initialized = true
		return input

	if not _initialized:
		reset(input)
		return input

	var alpha := 1.0 - exp(-TAU * cutoff_hz * delta)
	_state = _state.lerp(input, alpha)
	return _state
