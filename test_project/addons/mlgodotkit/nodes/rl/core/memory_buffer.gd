class_name MemoryBuffer extends RefCounted

var capacity : int
var buffer := []
var position := 0
var full := false

func _init(max_capacity := 10000):
	capacity = max_capacity
	buffer.resize(capacity)

func size() -> int:
	if full:
		return capacity
	return position

func is_ready(min_samples : int) -> bool:
	return size() >= min_samples

func store_experience(state, action, reward, next_state, done):
	var experience = {
		"state": state,
		"action": action,
		"reward": reward,
		"next_state": next_state,
		"done": done
	}

	buffer[position] = experience

	position += 1
	if position >= capacity:
		position = 0
		full = true

func sample(batch_size : int):
	var current_size = size()
	if batch_size > current_size:
		batch_size = current_size

	var indices = []
	while indices.size() < batch_size:
		var idx = randi() % current_size
		if not idx in indices:
			indices.append(idx)

	var batch = []
	for idx in indices:
		batch.append(buffer[idx])
		
	return batch
	
func clear():
	buffer.clear()
	buffer.resize(capacity)
	position = 0
	full = false
