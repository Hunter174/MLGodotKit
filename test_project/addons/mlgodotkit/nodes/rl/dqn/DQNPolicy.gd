extends RLPolicy
class_name DQNPolicy

@export var epsilon := 1.0
@export var epsilon_decay := 0.995
@export var epsilon_min := 0.05

var q_online: NeuralNetworkNode
var action_size := 0
var episode_count := 0

func configure(p_q_online: NeuralNetworkNode, p_action_size: int, _train_batch_size: int) -> void:
	q_online = p_q_online
	action_size = p_action_size
	assert(q_online != null)
	assert(action_size > 0)

func act(state: Array) -> int:
	if randf() < epsilon:
		return randi_range(0, action_size - 1)

	var old_bs := q_online.batch_size
	q_online.set_batch_size(1)
	var out := q_online.forward([state])
	q_online.set_batch_size(old_bs)

	var q_values = out[0]
	return _argmax(q_values)

func on_episode_end() -> void:
	episode_count += 1
	if episode_count < 200:
		return
	epsilon = max(epsilon * epsilon_decay, epsilon_min)

func _argmax(arr: Array) -> int:
	var bi := 0
	var bv := -INF
	for i in range(arr.size()):
		if arr[i] > bv:
			bv = arr[i]
			bi = i
	return bi
