class_name QTable
extends RefCounted

var table := {}

func _key_state(state):
	return str(state)

func _key_action(action):
	return str(action)

func evaluate(state, action):
	var s = _key_state(state)
	var a = _key_action(action)

	if not table.has(s):
		table[s] = {}
	if not table[s].has(a):
		table[s][a] = 0.0

	return table[s][a]

func set_value(state, action, value):
	var s = _key_state(state)
	var a = _key_action(action)

	if not table.has(s):
		table[s] = {}
	table[s][a] = value

func max_value(state):
	var s = _key_state(state)

	if not table.has(s):
		return 0.0

	var max_val = -INF
	for v in table[s].values():
		if v > max_val:
			max_val = v
	return max_val
