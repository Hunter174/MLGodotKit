class_name QLearner
extends RefCounted

var q_function
var gamma := 0.99
var lr := 0.1

func _init(_q_function, _lr := 0.1, _gamma := 0.99):
	q_function = _q_function
	lr = _lr
	gamma = _gamma

func learn(batch):
	for exp in batch:
		var s = exp["state"]
		var a = exp["action"]
		var r = exp["reward"]
		var s1 = exp["next_state"]
		var done = exp["done"]

		var old_q = q_function.evaluate(s, a)

		var max_next := 0.0
		if not done:
			max_next = q_function.max_value(s1)

		var target = r + gamma * max_next
		var new_q = old_q + lr * (target - old_q)

		q_function.set_value(s, a, new_q)
