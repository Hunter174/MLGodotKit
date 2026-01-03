extends MovementControllerBase
class_name MovementControllerNode3D

@export var sample_time := 1.0 / 60.0
@export var limit_min := -INF
@export var limit_max := INF

var pid_x: PIDControllerNode
var pid_y: PIDControllerNode
var pid_z: PIDControllerNode

var target: Vector3 = Vector3.ZERO

func _ready():
	super()

	pid_x = PIDControllerNode.new()
	pid_y = PIDControllerNode.new()
	pid_z = PIDControllerNode.new()

	for pid in [pid_x, pid_y, pid_z]:
		pid.set_sample_time(sample_time)

	_set_default_pid()
	_apply_limits()

func _set_default_pid():
	configure_pid(6.0, 0.0, 3.0)

func configure_pid(kp: float, ki: float, kd: float):
	for pid in [pid_x, pid_y, pid_z]:
		pid.set_kp(kp)
		pid.set_ki(ki)
		pid.set_kd(kd)

func set_limits(min_val: float, max_val: float):
	limit_min = min_val
	limit_max = max_val
	_apply_limits()

func set_target(t: Vector3):
	if t != target:
		target = t
		reset()

func reset():
	for pid in [pid_x, pid_y, pid_z]:
		pid.reset()
	_reset_filter()

func update(current: Vector3, delta: float) -> Vector3:
	if not enabled:
		return Vector3.ZERO

	if stop_on_arrival and _has_arrived(current, target):
		reset()
		return Vector3.ZERO

	var raw := Vector3(
		pid_x.update_dt(target.x, current.x, delta),
		pid_y.update_dt(target.y, current.y, delta),
		pid_z.update_dt(target.z, current.z, delta)
	)

	raw.x = clamp(raw.x, limit_min, limit_max)
	raw.y = clamp(raw.y, limit_min, limit_max)
	raw.z = clamp(raw.z, limit_min, limit_max)

	return _apply_low_pass(raw)

func _apply_limits():
	for pid in [pid_x, pid_y, pid_z]:
		pid.set_limits(limit_min, limit_max)
