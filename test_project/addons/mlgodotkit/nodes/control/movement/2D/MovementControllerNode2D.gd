extends MovementControllerBase
class_name MovementControllerNode2D

@export var sample_time := 1.0 / 60.0
@export var limit_min := -INF
@export var limit_max := INF

var pid_x: PIDControllerNode
var pid_y: PIDControllerNode

var target: Vector2 = Vector2.ZERO

func _ready():
	super()

	pid_x = PIDControllerNode.new()
	pid_y = PIDControllerNode.new()

	pid_x.set_sample_time(sample_time)
	pid_y.set_sample_time(sample_time)

	_set_default_pid()
	_apply_limits()

# ─────────────────────────────────────────────
# Defaults (hidden from beginners)
# ─────────────────────────────────────────────
func _set_default_pid():
	# Smooth, stable gameplay defaults
	configure_pid(6.0, 0.0, 3.0)

# Expert access
func configure_pid(kp: float, ki: float, kd: float):
	pid_x.set_kp(kp)
	pid_y.set_kp(kp)
	pid_x.set_ki(ki)
	pid_y.set_ki(ki)
	pid_x.set_kd(kd)
	pid_y.set_kd(kd)

func set_limits(min_val: float, max_val: float):
	limit_min = min_val
	limit_max = max_val
	_apply_limits()

func set_target(t: Vector2):
	if t != target:
		target = t
		reset()

func reset():
	pid_x.reset()
	pid_y.reset()
	_reset_filter()

# ─────────────────────────────────────────────
# Core update (position → velocity)
# ─────────────────────────────────────────────
func update(current: Vector2, delta: float) -> Vector2:
	if not enabled:
		return Vector2.ZERO

	if stop_on_arrival and _has_arrived(current, target):
		reset()
		return Vector2.ZERO

	var raw := Vector2(
		pid_x.update_dt(target.x, current.x, delta),
		pid_y.update_dt(target.y, current.y, delta)
	)

	raw = raw.clamp(
		Vector2(limit_min, limit_min),
		Vector2(limit_max, limit_max)
	)

	return _apply_low_pass(raw)

func _apply_limits():
	pid_x.set_limits(limit_min, limit_max)
	pid_y.set_limits(limit_min, limit_max)
