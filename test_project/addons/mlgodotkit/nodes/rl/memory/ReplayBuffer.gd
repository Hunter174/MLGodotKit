extends Node
class_name ReplayBuffer

@export var capacity := 10000
var buffer := []

func add(s, a, r, s_next, done):
	buffer.append({
		"s": s,
		"a": a,
		"r": r,
		"s_next": s_next,
		"done": done
	})
	if buffer.size() > capacity:
		buffer.pop_front()

func sample(batch_size):
	var batch := []
	for _i in range(batch_size):
		batch.append(buffer.pick_random())
	return batch

func size() -> int:
	return buffer.size()
