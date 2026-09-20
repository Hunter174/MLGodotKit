class_name DQNLearner
extends RefCounted

var q_network
var target_network
var gamma := 0.99
var loss := MSELossNode.new()
var update_counter := 0
var target_update_freq := 200

func _init(_q, _target):
	q_network = _q
	target_network = _target

func learn(batch):
	var states = []
	var targets = []

	for exp in batch:
		var s = exp["state"]
		var a = exp["action"]
		var r = exp["reward"]
		var s1 = exp["next_state"]
		var done = exp["done"]

		var q_values = q_network.evaluate_all(s)
		var target_q = q_values.duplicate()

		var next_q = target_network.evaluate_all(s1)
		var max_next = next_q.max()

		var y = r
		if not done:
			y += gamma * max_next

		target_q[a] = y

		states.append(s)
		targets.append(target_q)

	var preds = q_network.model.forward(states)
	loss.forward(preds, targets)
	var grad = loss.backward()
	q_network.model.backward(grad)

	update_counter += 1
	if update_counter % target_update_freq == 0:
		target_network.model.copy_weights(q_network.model)
