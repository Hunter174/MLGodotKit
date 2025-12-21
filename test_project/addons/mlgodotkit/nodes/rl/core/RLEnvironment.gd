extends Node2D
class_name RLEnvironment

signal episode_reset(initial_obs)
signal step_completed(obs, reward, done)
signal episode_done(total_reward)

var state := [] # RAW internal state (physics, game state, etc.)
var step_count := 0
var total_reward := 0.0
@export var max_steps := 1000

func reset() -> Array:
	step_count = 0
	total_reward = 0.0

	# _reset returns RAW state
	state = _reset()

	# observation derived from raw state
	var obs := _observe(state)

	emit_signal("episode_reset", obs)
	return obs

func step(action) -> Dictionary:
	# _step returns {"state": raw_state, "reward": r, "done": d}
	var result := _step(action)

	step_count += 1
	total_reward += float(result.reward)

	state = result.state  # RAW state always stored internally

	var done := bool(result.done) or step_count >= max_steps
	var obs := _observe(state)

	emit_signal("step_completed", obs, float(result.reward), done)

	if done:
		emit_signal("episode_done", total_reward)

	return {"state": obs, "reward": float(result.reward), "done": done}

# --- override points ---
func _reset() -> Array:
	push_error("RLEnvironment._reset not implemented")
	return []

func _step(action) -> Dictionary:
	push_error("RLEnvironment._step not implemented")
	return {"state": [], "reward": 0.0, "done": true}

func _observe(raw_state: Array) -> Array:
	# default: identity (raw == obs)
	return raw_state
