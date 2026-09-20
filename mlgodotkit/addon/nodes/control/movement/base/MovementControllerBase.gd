extends Node
class_name MovementControllerBase

# ─────────────────────────────────────────────
# Signals
# ─────────────────────────────────────────────
signal arrived(target)

# ─────────────────────────────────────────────
# General behavior toggles
# ─────────────────────────────────────────────
@export var enabled := true

# Arrival behavior
@export var stop_on_arrival := true
@export var arrival_radius := 8.0

# Output smoothing
@export var use_low_pass_filter := true
@export var filter_strength := 0.15
# (0 = no smoothing, 1 = infinite smoothing)

# Internal state
var _filtered_output = null
var _arrival_emitted := false

func _ready():
	_reset_filter()

# ─────────────────────────────────────────────
# Arrival handling (dimension-agnostic)
# ─────────────────────────────────────────────
func _has_arrived(current, target) -> bool:
	return current.distance_to(target) <= arrival_radius


func handle_arrival(current, target) -> bool:
	"""
	Returns true if arrival should stop motion.
	Emits `arrived(target)` exactly once per target.
	"""
	if not stop_on_arrival:
		return false

	if _has_arrived(current, target):
		if not _arrival_emitted:
			_arrival_emitted = true
			emit_signal("arrived", target)
		return true

	return false


func reset_arrival():
	_arrival_emitted = false

# ─────────────────────────────────────────────
# Low-pass filter (Vector2 / Vector3 safe)
# ─────────────────────────────────────────────
func _apply_low_pass(raw_output):
	if not use_low_pass_filter:
		return raw_output

	if _filtered_output == null:
		_filtered_output = raw_output
	else:
		_filtered_output = _filtered_output.lerp(raw_output, filter_strength)

	return _filtered_output


func _reset_filter():
	_filtered_output = null
