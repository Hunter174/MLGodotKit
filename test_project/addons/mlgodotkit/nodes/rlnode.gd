extends Node
class_name RLNode

signal episode_finished(total_reward: float)
signal policy_updated(epoch: int, loss: float)

@onready var env: CartPoleEnvironment = $"../CartPoleEnvironment"

# Optional maybe abstract out into an inspector view
@onready var graph: RewardGraph = $"../RewardGraph" 

# Hyper params
@export var gamma := 0.99
@export var learning_rate := 0.02

@export var epsilon: float = 1.0
@export var epsilon_decay: float = 0.995
@export var epsilon_min: float = 0.05

@export var max_steps_per_episode: int = 500
@export var step_delay: float = 0.01
@export var render_mode: bool = true

@export var batch_size: int = 128
@export var buffer_capacity: int = 10000
@export var buffer_warmup: int = 1000
@export var train_every_n_steps: int = 1
@export var target_update_period: int = 5000

@export var polyak_tau: float = 0.005
@export var max_grad_norm := 2.5
@export var huber_delta := 1.0

var q_online: NNNode
var q_target: NNNode

var replay_buffer: Array = []
var state: Array = []
var total_reward: float = 0.0
var episode_counter: int = 0
var global_step: int = 0

const STATE_SIZE := 4
const ACTION_SIZE := 2

func _ready() -> void:
	randomize()

	# Build online network
	q_online = NNNode.new()
	q_online.add_layer(STATE_SIZE, 64, "leaky_relu")
	q_online.add_layer(64, 64, "leaky_relu")
	q_online.add_layer(64, ACTION_SIZE, "linear")
	q_online.set_learning_rate(learning_rate)
	q_online.set_batch_size(batch_size)

	# Build target network
	q_target = NNNode.new()
	q_target.add_layer(STATE_SIZE, 64, "leaky_relu")
	q_target.add_layer(64, 64, "leaky_relu")
	q_target.add_layer(64, ACTION_SIZE, "linear")
	q_target.set_learning_rate(learning_rate)
	q_target.set_batch_size(batch_size)

	_copy_weights(q_online, q_target)

	state = env.get_state()
	await _main_loop()

func _main_loop() -> void:
	while true:
		await run_episode()

func run_episode() -> void:
	total_reward = 0.0
	episode_counter += 1
	state = env.reset()

	# Epsilon decay AFTER episode for better exploration at start
	if epsilon > epsilon_min:
		epsilon = max(epsilon * epsilon_decay, epsilon_min)

	await _run_step_loop()

	print("✅ Episode %d | Total Reward: %.2f | ε: %.3f" % [episode_counter, total_reward, epsilon])
	
	# Abstract out
	graph.add_reward(total_reward)
	emit_signal("episode_finished", total_reward)


func _run_step_loop() -> void:
	for step in range(max_steps_per_episode):
		var action := _select_action(state)
		var result := env.step([action])
		var reward: float = result.reward
		var done: bool = result.done

		_store_experience(state, action, reward, result.state, done)
		state = result.state
		total_reward += reward
		global_step += 1

		if replay_buffer.size() >= batch_size and (global_step % train_every_n_steps) == 0:
			_train_from_replay()

		if global_step % target_update_period == 0:
			_polyak_update()

		if done:
			break

		if render_mode:
			await get_tree().create_timer(step_delay).timeout


func _select_action(s: Array) -> int:
	# If a random tick is less than ε choose a random action
	if randf() < epsilon:
		return randi_range(0, ACTION_SIZE - 1)
		
	# Here we need to reset the batch size to single sample
	q_online.set_batch_size(1)
	var q_values = _flatten_1d(q_online.forward([s]))
	
	# Reset back to the origional batch size	
	q_online.set_batch_size(batch_size)
	return _argmax(q_values)

func _store_experience(s, a, r, s_next, done) -> void:
	replay_buffer.append({"s": s, "a": a, "r": r, 
	"s_next": s_next, "done": done})
	
	if replay_buffer.size() > buffer_capacity:
		replay_buffer.pop_front()

func _train_from_replay() -> void:
	
	if replay_buffer.size() < buffer_warmup:
		return
		
	# --- 1. Sample random batch ---
	var batch: Array = []
	for i in range(batch_size):
		batch.append(replay_buffer.pick_random())

	# --- 2. Split batch ---
	var states: Array = []
	var next_states: Array = []
	var actions: Array = []
	var rewards: Array = []
	var dones: Array = []

	for exp in batch:
		states.append(exp.s)
		next_states.append(exp.s_next)
		actions.append(exp.a)
		rewards.append(exp.r)
		dones.append(exp.done)

	# --- 3. Batch sizes ---
	q_online.set_batch_size(batch_size)
	q_target.set_batch_size(batch_size)

	# --- 4. Forward passes ---
	var q_values_batch: Array = q_online.forward(states)
	var next_q_online_batch: Array = q_online.forward(next_states)
	var next_q_target_batch: Array = q_target.forward(next_states)

	# --- 5. TD errors (Double DQN target) ---
	var td_errors: Array = []
	for i in range(batch_size):
		var q_values = _flatten_1d(q_values_batch[i])
		var next_q_online = _flatten_1d(next_q_online_batch[i])
		var next_q_target = _flatten_1d(next_q_target_batch[i])
		var a_star = _argmax(next_q_online)

		var td_target = rewards[i]
		if not dones[i]:
			td_target += gamma * next_q_target[a_star]

		var q_pred = q_values[actions[i]]
		var td_error = td_target - q_pred
		td_errors.append(td_error)

	# --- 6. Huber derivative, correct formula + correct sign ---
	var error_matrix: Array = []
	for i in range(batch_size):
		var row: Array = []
		for j in range(ACTION_SIZE):
			row.append(0.0)

		var err = td_errors[i]
		var grad_val: float
		if abs(err) <= huber_delta:
			# dL/d(err) = err
			grad_val = err
		else:
			# dL/d(err) = huber_delta * sign(err)
			grad_val = huber_delta * sign(err)

		# dL/dQ = - dL/d(err)
		row[actions[i]] = -grad_val
		error_matrix.append(row)

	# --- 7. Backprop ---
	q_online.backward(error_matrix)

	# --- 8. Diagnostics ---
	if (global_step % 500) == 0:
		var mean_td = abs(td_errors.reduce(func(a,b): return a+b) / td_errors.size())
		print("🧮 Step %d | Mean |TD|=%.4f | ε=%.3f" % [global_step, mean_td, epsilon])

func _polyak_update() -> void:
	for i in range(q_online.layers.size()):
		var w_online = q_online.layers[i].get_weights()
		var b_online = q_online.layers[i].get_biases()
		var w_target = q_target.layers[i].get_weights()
		var b_target = q_target.layers[i].get_biases()
		var w_new = (1.0 - polyak_tau) * w_target + polyak_tau * w_online
		var b_new = (1.0 - polyak_tau) * b_target + polyak_tau * b_online
		q_target.layers[i].set_weights(w_new)
		q_target.layers[i].set_biases(b_new)

func _copy_weights(src: NNNode, dst: NNNode) -> void:
	dst.copy_weights(src)

func _flatten_1d(arr: Array) -> Array:
	if arr.size() == 0:
		return []
	if typeof(arr[0]) == TYPE_ARRAY:
		return arr[0]
	return arr

# Simple argmax function
func _argmax(arr: Array) -> int:
	var best_idx := 0
	var best_val := -INF
	for i in range(arr.size()):
		if arr[i] > best_val:
			best_val = arr[i]
			best_idx = i
	return best_idx

# Why do we need hubber?
func _huber(td_error: float) -> float:
	if abs(td_error) < huber_delta:
		return 0.5 * td_error * td_error
	else:
		return huber_delta * (abs(td_error) - 0.5 * huber_delta)
