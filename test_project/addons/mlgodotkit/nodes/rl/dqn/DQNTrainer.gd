extends Node
class_name DQNTrainer

@export var gamma := 0.99
@export var batch_size := 128
@export var warmup := 1000
@export var target_update_period := 2000
@export var polyak_tau := 0.005
@export var huber_delta := 1.0

var q_online: NeuralNetworkNode
var q_target: NeuralNetworkNode
var buffer: ReplayBuffer
var action_size := 0
var global_step := 0

func configure(p_q_online: NeuralNetworkNode, p_q_target: NeuralNetworkNode, 
	p_buffer: ReplayBuffer, p_action_size: int) -> void:
	
	q_online = p_q_online
	q_target = p_q_target
	buffer = p_buffer
	action_size = p_action_size

	assert(q_online != null)
	assert(q_target != null)
	assert(buffer != null)
	assert(action_size > 0)

func observe(s, a, r, s_next, done) -> void:
	buffer.add(s, a, r, s_next, done)

func should_train(step: int) -> bool:
	global_step = step
	return buffer.size() >= warmup and buffer.size() >= batch_size

func train_step() -> void:
	var batch = buffer.sample(batch_size)

	var states := []
	var next_states := []
	var actions := []
	var rewards := []
	var dones := []

	for e in batch:
		states.append(e.s)
		next_states.append(e.s_next)
		actions.append(e.a)
		rewards.append(e.r)
		dones.append(e.done)

	q_online.set_batch_size(batch_size)
	q_target.set_batch_size(batch_size)

	var next_q_online := q_online.forward(next_states)
	var next_q_target := q_target.forward(next_states)
	var q_values_batch := q_online.forward(states)

	var grads := []

	for i in range(batch_size):
		var q_vals = q_values_batch[i]
		var next_online = next_q_online[i]
		var next_target = next_q_target[i]

		var a_star := _argmax(next_online)

		var target = rewards[i]
		if not dones[i]:
			target += gamma * next_target[a_star]

		var err = target - q_vals[actions[i]]
		var g = err if abs(err) <= huber_delta else huber_delta * sign(err)

		var row := []
		for _j in range(action_size):
			row.append(0.0)
		row[actions[i]] = -g
		grads.append(row)

	q_online.backward(grads)

	if global_step % target_update_period == 0:
		_polyak_update()

func _polyak_update():
	for i in range(q_online.layers.size()):
		var w := lerp(q_target.layers[i].get_weights(), q_online.layers[i].get_weights(), polyak_tau)
		var b := lerp(q_target.layers[i].get_biases(), q_online.layers[i].get_biases(), polyak_tau)
		q_target.layers[i].set_weights(w)
		q_target.layers[i].set_biases(b)

func _argmax(arr: Array) -> int:
	var bi := 0
	var bv := -INF
	for i in range(arr.size()):
		if arr[i] > bv:
			bv = arr[i]
			bi = i
	return bi
