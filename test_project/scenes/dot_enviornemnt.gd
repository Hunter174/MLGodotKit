extends RLEnvironment
class_name DotEnvironment

# ----------------------------------------------------------
# --- Tunable Parameters ---
# ----------------------------------------------------------
@export var move_speed: float = 1.0
@export var goal_threshold: float = 15.0
@export var max_range: float = 600.0
@export var min_range: float = 200.0
@export var arena_min: Vector2 = Vector2(-200, -200)
@export var arena_max: Vector2 = Vector2(800, 800)
@export var max_steps_per_episode: int = 500

# ----------------------------------------------------------
# --- Internal State ---
# ----------------------------------------------------------
var agent_pos: Vector2 = Vector2(100, 100)
var target_pos: Vector2 = Vector2(600, 600)
var done: bool = false

var ACTIONS = [
	Vector2(-1, 0),    # left
	Vector2(1, 0),     # right
	Vector2(0, -1),    # up
	Vector2(0, 1),     # down
	Vector2(-1, -1),   # up-left
	Vector2(1, -1),    # up-right
	Vector2(-1, 1),    # down-left
	Vector2(1, 1)      # down-right
]

# ----------------------------------------------------------
# --- Godot Lifecycle ---
# ----------------------------------------------------------
func _ready() -> void:
	randomize()
	
func _process(delta: float) -> void:
	queue_redraw()

func _draw() -> void:
	draw_circle(agent_pos, 8, Color.RED)
	draw_circle(target_pos, 10, Color.GREEN)
	draw_line(agent_pos, target_pos, Color(0.8, 0.8, 0.8), 1.0)

# ----------------------------------------------------------
# --- RL Environment API ---
# ----------------------------------------------------------
func get_state() -> Array:
	var rel = target_pos - agent_pos
	var rel_norm = rel / max_range
	var dist = clamp(rel.length() / max_range, 0.0, 1.0)
	return [rel_norm.x, rel_norm.y, dist, 1.0]

func reset() -> Array:
	step_count = 0
	total_reward = 0.0
	target_pos = Vector2(randf_range(min_range, max_range), randf_range(min_range, max_range))
	agent_pos = Vector2(randf_range(-max_range, max_range), randf_range(-max_range, max_range))
	_clamp_agent_in_bounds()
	queue_redraw()
	return get_state()

func step(action: Array) -> Dictionary:
	step_count += 1

	# Compute distances before and after move
	var prev_rel = target_pos - agent_pos
	var prev_dist = prev_rel.length()

	var move_vec = ACTIONS[int(action[0])].normalized()
	agent_pos += move_vec * move_speed
	_clamp_agent_in_bounds()

	var rel = target_pos - agent_pos
	var dist = rel.length()

	# Reward based on progress
	var progress = (prev_dist - dist) / max_range
	var reward = progress * 100.0 - 0.01
	reward = clamp(reward, -1.0, 1.0)

	var done := false
	if dist <= goal_threshold:
		reward += 1.0
		done = true
	elif step_count >= max_steps_per_episode:
		done = true

	total_reward += reward
	var s_next = get_state()

	if step_count % 50 == 0 or done:
		print("📘 Step %d | Dist: %.2f | Reward: %.3f | State: %s" % [step_count, dist, reward, str(s_next)])

	return {"state": s_next, "reward": reward, "done": done}

func _clamp_agent_in_bounds() -> void:
	agent_pos.x = clamp(agent_pos.x, arena_min.x, arena_max.x)
	agent_pos.y = clamp(agent_pos.y, arena_min.y, arena_max.y)
