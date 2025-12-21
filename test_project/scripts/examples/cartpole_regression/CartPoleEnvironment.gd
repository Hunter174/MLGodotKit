extends RLEnvironment
class_name CartPoleEnvironment

const GRAVITY := 9.8
const CART_MASS := 1.0
const POLE_MASS := 0.1
const TOTAL_MASS := CART_MASS + POLE_MASS
const LENGTH := 0.5
const FORCE_MAG := 10.0
const TAU := 0.02

const X_THRESH := 2.4
const THETA_THRESH := 0.2095  # ~12 degrees

func _reset() -> Array:
	# RAW physics state
	return [
		randf_range(-0.05, 0.05),
		randf_range(-0.05, 0.05),
		randf_range(-0.05, 0.05),
		randf_range(-0.05, 0.05),
	]

func _step(action) -> Dictionary:
	# --- robust action parsing ---
	var a: int
	if typeof(action) == TYPE_ARRAY:
		a = int(action[0])
	else:
		a = int(action)

	var x: float = state[0]
	var x_dot: float = state[1]
	var theta: float = state[2]
	var theta_dot: float = state[3]

	var force := FORCE_MAG if a == 1 else -FORCE_MAG

	var costheta := cos(theta)
	var sintheta := sin(theta)

	var temp := (force + POLE_MASS * LENGTH * theta_dot * theta_dot * sintheta) / TOTAL_MASS
	var theta_acc := (GRAVITY * sintheta - costheta * temp) / (
		LENGTH * (4.0 / 3.0 - POLE_MASS * costheta * costheta / TOTAL_MASS)
	)
	var x_acc := temp - POLE_MASS * LENGTH * theta_acc * costheta / TOTAL_MASS

	x += TAU * x_dot
	x_dot += TAU * x_acc
	theta += TAU * theta_dot
	theta_dot += TAU * theta_acc

	var raw_state := [x, x_dot, theta, theta_dot]

	var done = abs(x) > X_THRESH or abs(theta) > THETA_THRESH

	# Stable reward shaping (RAW units)
	var angle_cost := theta * theta
	var pos_cost := 0.1 * x * x
	var vel_cost := 0.01 * (x_dot * x_dot + theta_dot * theta_dot)

	var reward := 1.0 - angle_cost - pos_cost - vel_cost

	if done:
		reward = -5.0

	return {
		"state": raw_state,
		"reward": reward,
		"done": done
	}

func _observe(raw_state: Array) -> Array:
	return [
		clamp(raw_state[0] / X_THRESH, -1.0, 1.0),
		clamp(raw_state[1] / 5.0, -1.0, 1.0),
		clamp(raw_state[2] / THETA_THRESH, -1.0, 1.0),
		clamp(raw_state[3] / 5.0, -1.0, 1.0),
	]
