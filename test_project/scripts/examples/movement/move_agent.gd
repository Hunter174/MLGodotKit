extends CharacterBody2D

@export var max_speed := 300.0

# Look-ahead distance for anticipatory steering
@export var lookahead_distance := 48.0 

# ─────────────────────────────────────────────
# Control filters
# ─────────────────────────────────────────────
@export var steering_cutoff_hz := 6.0
@export var max_steering_delta := 800.0   # units/sec

@onready var controller: MovementControllerNode2D = $MovementControllerNode2D
@onready var navigator: Navigator2D = $Navigator2D
@onready var lowpass: LowPassFilter2D = $LowPassFilter2D
@onready var slew: SlewLimiter2D = $SlewLimiter2D

var goal: Vector2

func _ready() -> void:
	controller.set_limits(-max_speed, max_speed)
	controller.stop_on_arrival = true
	controller.arrival_radius = 20

	lowpass.cutoff_hz = steering_cutoff_hz
	slew.max_delta_per_sec = max_steering_delta

	# Demo patrol route
	navigator.patrol_points = [
		Vector2(200, 200),
		Vector2(600, 200),
		Vector2(600, 500),
		Vector2(200, 500)
	]

	_set_behavior(Navigator2D.Behavior.SEEK)

func _physics_process(delta: float) -> void:
	_handle_behavior_input()
	_handle_goal_input()

	# ─────────────────────────────────────────
	# Navigator → raw steering direction
	# ─────────────────────────────────────────
	var desired_dir := navigator.compute_direction(
		global_position,
		velocity,
		goal,
		delta
	)

	# ─────────────────────────────────────────
	# Look-ahead target selection
	# ─────────────────────────────────────────
	var to_goal := goal - global_position
	var dist := to_goal.length()

	var effective_target: Vector2
	if dist <= lookahead_distance:
		effective_target = goal
	else:
		effective_target = global_position + desired_dir * lookahead_distance

	controller.set_target(effective_target)

	# ─────────────────────────────────────────
	# Controller → desired velocity
	# ─────────────────────────────────────────
	var desired_velocity := controller.update(global_position, delta)

	# ─────────────────────────────────────────
	# Signal conditioning
	# ─────────────────────────────────────────
	desired_velocity = lowpass.filter(desired_velocity, delta)
	desired_velocity = slew.filter(desired_velocity, delta)

	velocity = desired_velocity
	move_and_slide()

# ─────────────────────────────────────────────
# Input handling
# ─────────────────────────────────────────────
func _handle_goal_input() -> void:
	if Input.is_mouse_button_pressed(MOUSE_BUTTON_LEFT):
		goal = get_global_mouse_position()

func _handle_behavior_input() -> void:
	if Input.is_key_pressed(KEY_1):
		_set_behavior(Navigator2D.Behavior.SEEK)
	elif Input.is_key_pressed(KEY_2):
		_set_behavior(Navigator2D.Behavior.FLEE)
	elif Input.is_key_pressed(KEY_3):
		_set_behavior(Navigator2D.Behavior.LOS_SEEK)
	elif Input.is_key_pressed(KEY_4):
		_set_behavior(Navigator2D.Behavior.MAINTAIN_RANGE)
	elif Input.is_key_pressed(KEY_5):
		_set_behavior(Navigator2D.Behavior.ORBIT)
	elif Input.is_key_pressed(KEY_6):
		_set_behavior(Navigator2D.Behavior.PATROL)

func _set_behavior(b) -> void:
	if navigator.behavior == b:
		return

	navigator.behavior = b

	# Important: reset filters on behavior change
	lowpass.reset(velocity)
	slew.reset(velocity)

	print("Navigator behavior → ", _behavior_name(b))

func _behavior_name(b: int) -> String:
	match b:
		Navigator2D.Behavior.SEEK:
			return "SEEK"
		Navigator2D.Behavior.FLEE:
			return "FLEE"
		Navigator2D.Behavior.LOS_SEEK:
			return "LOS_SEEK"
		Navigator2D.Behavior.MAINTAIN_RANGE:
			return "MAINTAIN_RANGE"
		Navigator2D.Behavior.ORBIT:
			return "ORBIT"
		Navigator2D.Behavior.PATROL:
			return "PATROL"
	return "UNKNOWN"
