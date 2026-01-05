extends CharacterBody2D

@export var max_speed := 300.0

# We look ahead a few steps to see were to move next
@export var lookahead_distance := 48.0 

@onready var controller: MovementControllerNode2D = $MovementControllerNode2D
@onready var navigator: Navigator2D = $Navigator2D

var goal: Vector2

func _ready() -> void:
	
	controller.set_limits(-max_speed, max_speed)
	controller.stop_on_arrival = true
	controller.arrival_radius = 20

	# Demo patrol route (static)
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
	
	var desired_dir := navigator.compute_direction(
		global_position,
		velocity,
		goal,
		delta
	)

	var to_goal := goal - global_position
	var dist := to_goal.length()

	var effective_target: Vector2

	if dist <= lookahead_distance:
		# Near goal → aim directly at the real target
		effective_target = goal
	else:
		# Far from goal → use lookahead
		effective_target = global_position + desired_dir * lookahead_distance

	controller.set_target(effective_target)

	var desired_velocity := controller.update(global_position, delta)
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
