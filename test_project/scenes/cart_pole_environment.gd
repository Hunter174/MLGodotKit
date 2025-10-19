extends RLEnvironment
class_name CartPoleEnvironment

# ----------------------------------------------------------
# --- Physics Parameters
# ----------------------------------------------------------
@export var gravity: float = 9.8
@export var mass_cart: float = 1.0
@export var mass_pole: float = 0.1
@export var pole_length: float = 0.5  # half-length in meters
@export var force_mag: float = 10.0
@export var tau: float = 1.0 / 60.0   # time step (s)

# Stability thresholds
@export var x_threshold: float = 10          # cart position limit (m)
@export var theta_threshold: float = deg_to_rad(25.0)  # pole angle limit (rad)
@export var max_steps_per_episode: int = 1000

# ----------------------------------------------------------
# --- Reward Parameters
# ----------------------------------------------------------
@export var reward_alive: float = 1.0
@export var reward_failure: float = -1.0
@export var angle_penalty_scale: float = 0.5
@export var position_penalty_scale: float = 0.05

# ----------------------------------------------------------
# --- Visualization
# ----------------------------------------------------------
@export var origin: Vector2 = Vector2(400, 300)
@export var scale_factor: float = 100.0
@export var cart_size: Vector2 = Vector2(60, 20)
@export var pole_width: float = 6.0
@export var pole_color: Color = Color(0.39, 0.58, 0.93)
@export var cart_color: Color = Color(0.4, 0.4, 0.4)

# ----------------------------------------------------------
# --- Internal State
# ----------------------------------------------------------
var x := 0.0
var x_dot := 0.0
var theta := 0.0
var theta_dot := 0.0
var last_reward := 0.0

# ----------------------------------------------------------
# --- Lifecycle
# ----------------------------------------------------------
func _ready() -> void:
	randomize()
	reset()

func _process(_delta: float) -> void:
	queue_redraw()

func _draw() -> void:
	var track_y := origin.y + 100
	draw_line(
		Vector2(origin.x - x_threshold * scale_factor - 50, track_y),
		Vector2(origin.x + x_threshold * scale_factor + 50, track_y),
		Color.GRAY, 4.0
	)

	var cart_px := Vector2(origin.x + x * scale_factor, track_y)
	draw_rect(Rect2(cart_px - cart_size / 2.0, cart_size), cart_color, true)

	var pole_len_px := pole_length * scale_factor * 2.0
	var pole_tip := cart_px + Vector2(pole_len_px * sin(theta), -pole_len_px * cos(theta))
	draw_line(cart_px, pole_tip, pole_color, pole_width)
	draw_circle(pole_tip, 6.0, Color.SKY_BLUE)

	var msg := "x=%.2f | θ=%.2f° | reward=%.2f" % [x, rad_to_deg(theta), last_reward]
	draw_string(ThemeDB.fallback_font, origin + Vector2(-120, -180), msg, 0, 1, 16, Color.WHITE, 16)

# ----------------------------------------------------------
# --- RL API
# ----------------------------------------------------------
func get_state() -> Array:
	# Slight normalization (not clamped)
	return [
		x / x_threshold,
		x_dot / 2.0,
		theta / theta_threshold,
		theta_dot / 2.0
	]

func reset() -> Array:
	x = randf_range(-0.4, 0.4)
	x_dot = randf_range(-0.05, 0.05)
	theta = randf_range(deg_to_rad(-5.0), deg_to_rad(5.0))
	theta_dot = randf_range(-0.05, 0.05)
	step_count = 0
	last_reward = 0.0
	queue_redraw()
	return get_state()

func step(action: Array) -> Dictionary:
	step_count += 1

	# Force: 0 = left, 1 = right
	var force := force_mag if int(action[0]) == 1 else -force_mag

	# --- Dynamics (Euler integration)
	var total_mass = mass_cart + mass_pole
	var polemass_len = mass_pole * pole_length
	var sin_t = sin(theta)
	var cos_t = cos(theta)

	var temp = (force + polemass_len * theta_dot * theta_dot * sin_t) / total_mass
	var theta_acc = (gravity * sin_t - cos_t * temp) / (pole_length * (4.0 / 3.0 - mass_pole * cos_t * cos_t / total_mass))
	var x_acc = temp - polemass_len * theta_acc * cos_t / total_mass

	x += tau * x_dot
	x_dot += tau * x_acc
	theta += tau * theta_dot
	theta_dot += tau * theta_acc

	# --- Termination & Reward Shaping ---
	var done = abs(x) > x_threshold or abs(theta) > theta_threshold or step_count >= max_steps

	# --- Progressive reward shaping ---
	# The closer the pole is to vertical (θ = 0) and the cart to center (x = 0), the higher the reward.
	# This encourages smooth control and recovery, not just survival.
	var angle_penalty = pow(abs(theta / theta_threshold), 1.5)     # penalize large tilts more strongly
	var pos_penalty   = pow(abs(x / x_threshold), 1.2)             # penalize drifting from center
	var vel_penalty   = 0.01 * (abs(x_dot) + abs(theta_dot))       # small penalty for jittery motion

	# Combine into a smooth reward signal in [−2, +1]
	var reward = reward_alive * (1.0 - angle_penalty - pos_penalty - vel_penalty)

	# Add a small bonus when the pole is very upright and centered
	if abs(theta) < deg_to_rad(3.0) and abs(x) < 0.2:
		reward += 0.5

	# If the episode ends, penalize heavily for failure
	if done:
		reward -= 2.0

	# Clamp to reasonable range
	reward = clamp(reward, -2.0, 2.0)

	last_reward = reward
	queue_redraw()

	return {"state": get_state(), "reward": reward, "done": done}
